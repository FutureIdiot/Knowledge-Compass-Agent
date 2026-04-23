from core.router import RouterAgent
from core.subagents import (
    InteractionAgent,
    KnowledgeManagerAgent,
    MemoryManagerAgent,
    ProfileManagerAgent,
    WebSearchAgent,
)
from memory.schemas import AgentRole, TaskResult, TaskSpec, TaskStatus
from llm.factory import get_role_llm


class MultiAgentOrchestrator:
    def __init__(self):
        self.router = RouterAgent(get_role_llm("router"))
        self.interaction_agent = InteractionAgent(get_role_llm("interaction"))
        self.agents = {
            AgentRole.MEMORY_MANAGER: MemoryManagerAgent(get_role_llm("memory_manager")),
            AgentRole.PROFILE_MANAGER: ProfileManagerAgent(get_role_llm("profile_manager")),
            AgentRole.KNOWLEDGE_MANAGER: KnowledgeManagerAgent(get_role_llm("knowledge_manager")),
            AgentRole.WEB_SEARCHER: WebSearchAgent(get_role_llm("web_searcher")),
            AgentRole.INTERACTION: self.interaction_agent,
        }

    def run_stream(self, user_input: str, history: list | None = None):
        history = history or []
        shared_context = self._bootstrap_context(history)
        plan = self.router.plan(
            user_input=user_input,
            history=history,
            runtime_state=shared_context["runtime_state"],
        )

        task_results = self._execute_plan(plan.tasks, shared_context, user_input)
        shared_context["task_results"] = task_results

        interaction_task = self._pick_interaction_task(plan.tasks, user_input)
        for chunk in self.interaction_agent.stream_answer(interaction_task, shared_context):
            yield chunk

    def _bootstrap_context(self, history: list) -> dict:
        return {
            "history": history,
            "memory_snapshot": "",
            "profile_snapshot": "",
            "knowledge_snapshot": "",
            "runtime_state": {
                "has_memory": self.agents[AgentRole.MEMORY_MANAGER].resource_exists(),
                "has_profile": self.agents[AgentRole.PROFILE_MANAGER].resource_exists(),
                "knowledge_file_count": self.agents[AgentRole.KNOWLEDGE_MANAGER].resource_count(),
            },
        }

    def _execute_plan(self, tasks: list[TaskSpec], shared_context: dict, user_input: str) -> dict[str, TaskResult]:
        results: dict[str, TaskResult] = {}
        for task in tasks:
            if task.owner == AgentRole.INTERACTION:
                continue

            unresolved = [task_id for task_id in task.depends_on if task_id not in results]
            if unresolved:
                results[task.id] = TaskResult(
                    task_id=task.id,
                    owner=task.owner,
                    status=TaskStatus.SKIPPED,
                    summary=f"依赖未完成，跳过任务: {', '.join(unresolved)}",
                )
                continue

            task.payload.setdefault("user_input", user_input)
            result = self.agents[task.owner].run(
                task,
                {**shared_context, "task_results": results},
            )
            results[task.id] = result

            if "memory_snapshot" in result.data:
                shared_context["memory_snapshot"] = result.data["memory_snapshot"]
            if "profile_snapshot" in result.data:
                shared_context["profile_snapshot"] = result.data["profile_snapshot"]
            if "knowledge_snapshot" in result.data:
                shared_context["knowledge_snapshot"] = result.data["knowledge_snapshot"]

        return results

    def _pick_interaction_task(self, tasks: list[TaskSpec], user_input: str) -> TaskSpec:
        for task in tasks:
            if task.owner == AgentRole.INTERACTION:
                task.payload.setdefault("user_input", user_input)
                return task
        return TaskSpec(
            id="final-interaction",
            owner=AgentRole.INTERACTION,
            goal="compose_final_answer",
            instructions="综合已有上下文回答用户。",
            payload={"user_input": user_input},
        )
