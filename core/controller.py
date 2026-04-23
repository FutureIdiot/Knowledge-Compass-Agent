import json
import re

from core.subagents import (
    KnowledgeManagerAgent,
    MemoryManagerAgent,
    ProfileManagerAgent,
    ResponderAgent,
    WebSearchAgent,
)
from llm.base import BaseLLM
from llm.factory import get_role_llm
from memory.schemas import AgentRole, SemanticDecision, TaskResult, TaskSpec, TaskStatus


CONTROLLER_PROMPT = """你是多 Agent 学习系统的 Controller。
你的职责不是生成任务列表，而是做语义判断，告诉代码层应该启用哪些能力。

请只输出 JSON，不要输出 Markdown，不要解释。
输出结构：
{
  "response_mode": "direct" | "composite",
  "reason": "一句话说明判断依据",
  "read_memory": true,
  "write_memory": false,
  "read_profile": false,
  "write_profile": false,
  "use_knowledge": false,
  "use_web": false,
  "knowledge_query": "",
  "web_query": ""
}

规则：
1. simple chit-chat 或无需额外上下文的短问题用 direct。
2. 是否读取 memory/profile 与是否写入 memory/profile 要分别判断。
3. 用户在问“接下来怎么做、结合我当前情况、复盘、总结”时，通常需要读 memory 和/或 profile。
4. 用户在陈述进度变化、计划变化、目标变化、偏好变化、约束变化时，通常需要写 memory 和/或 profile。
5. 本地知识检索只在确实需要知识库资料时启用；联网搜索只在需要外部/最新信息时启用。
6. knowledge_query 和 web_query 只在对应开关为 true 时填写，否则留空。
7. 不要生成任务 id，不要生成执行步骤，不要代替代码层编排流程。
"""


class LearningController:
    def __init__(self, llm: BaseLLM | None = None):
        self.llm = llm or get_role_llm("controller")
        self.responder = ResponderAgent(get_role_llm("responder"))
        self.agents = {
            AgentRole.MEMORY_MANAGER: MemoryManagerAgent(get_role_llm("memory_manager")),
            AgentRole.PROFILE_MANAGER: ProfileManagerAgent(get_role_llm("profile_manager")),
            AgentRole.KNOWLEDGE_MANAGER: KnowledgeManagerAgent(get_role_llm("knowledge_manager")),
            AgentRole.WEB_SEARCHER: WebSearchAgent(get_role_llm("web_searcher")),
        }

    def run_stream(self, user_input: str, history: list | None = None):
        history = history or []
        shared_context = self._bootstrap_context(history)
        decision = self._decide(user_input=user_input, history=history, runtime_state=shared_context["runtime_state"])
        tasks = self._build_task_list(user_input=user_input, decision=decision)
        task_results = self._run_task_pipeline(tasks=tasks, shared_context=shared_context)

        shared_context["task_results"] = task_results
        shared_context["task_board"] = self._serialize_task_board(tasks, task_results)
        shared_context["controller_decision"] = decision.model_dump()

        responder_task = self._build_responder_task(user_input=user_input, tasks=tasks, task_results=task_results)
        for chunk in self.responder.stream_answer(responder_task, shared_context):
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

    def _decide(self, user_input: str, history: list, runtime_state: dict) -> SemanticDecision:
        messages = [
            {"role": "system", "content": CONTROLLER_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_input": user_input,
                        "history_tail": history[-6:],
                        "runtime_state": runtime_state,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            content = self.llm.chat(messages).get("content", "")
            return SemanticDecision.model_validate(self._load_json(content))
        except Exception:
            return self._fallback_decision(user_input)

    def _build_task_list(self, user_input: str, decision: SemanticDecision) -> list[TaskSpec]:
        tasks: list[TaskSpec] = []

        if decision.read_memory or decision.write_memory:
            tasks.append(
                TaskSpec(
                    id="memory",
                    owner=AgentRole.MEMORY_MANAGER,
                    goal="manage_memory_context",
                    instructions="按需读取或写入长期记忆。",
                    payload={
                        "user_input": user_input,
                        "action": self._merge_action(decision.read_memory, decision.write_memory),
                    },
                    required_payload_fields=["user_input", "action"],
                    max_retries=1,
                )
            )

        if decision.read_profile or decision.write_profile:
            tasks.append(
                TaskSpec(
                    id="profile",
                    owner=AgentRole.PROFILE_MANAGER,
                    goal="manage_profile_context",
                    instructions="按需读取或写入用户画像。",
                    payload={
                        "user_input": user_input,
                        "action": self._merge_action(decision.read_profile, decision.write_profile),
                    },
                    required_payload_fields=["user_input", "action"],
                    max_retries=1,
                )
            )

        if decision.use_knowledge:
            tasks.append(
                TaskSpec(
                    id="knowledge",
                    owner=AgentRole.KNOWLEDGE_MANAGER,
                    goal="retrieve_local_knowledge",
                    instructions="只检索与当前问题最相关的本地知识。",
                    payload={
                        "user_input": user_input,
                        "query": decision.knowledge_query or user_input,
                    },
                    required_payload_fields=["query"],
                    run_if={"runtime_state.knowledge_file_count": {"gt": 0}},
                )
            )

        if decision.use_web:
            tasks.append(
                TaskSpec(
                    id="web_search",
                    owner=AgentRole.WEB_SEARCHER,
                    goal="search_web",
                    instructions="搜索外部或最新资料。",
                    payload={
                        "user_input": user_input,
                        "query": decision.web_query or user_input,
                    },
                    required_payload_fields=["query"],
                    max_retries=1,
                )
            )

        tasks.append(
            TaskSpec(
                id="responder",
                owner=AgentRole.RESPONDER,
                goal="compose_final_answer",
                instructions="整合所有上游结果，生成最终回答。",
                payload={"user_input": user_input},
                required_payload_fields=["user_input"],
            )
        )
        return tasks

    def _run_task_pipeline(self, tasks: list[TaskSpec], shared_context: dict) -> dict[str, TaskResult]:
        results: dict[str, TaskResult] = {}
        worker_tasks = [task for task in tasks if task.owner != AgentRole.RESPONDER]

        while not self._all_subtasks_returned(worker_tasks, results):
            progress_made = False

            for task in worker_tasks:
                if task.status in {TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.FAILED}:
                    continue

                if not self._dependencies_resolved(task, results):
                    continue

                if not self._branch_condition_matches(task, shared_context, results):
                    task.status = TaskStatus.SKIPPED
                    results[task.id] = TaskResult(
                        task_id=task.id,
                        owner=task.owner,
                        status=TaskStatus.SKIPPED,
                        summary="分支条件不满足，跳过执行。",
                        attempts=max(task.retry_count, 1),
                    )
                    progress_made = True
                    continue

                validation_error = self._validate_task(task, shared_context)
                if validation_error:
                    task.status = TaskStatus.FAILED
                    results[task.id] = TaskResult(
                        task_id=task.id,
                        owner=task.owner,
                        status=TaskStatus.FAILED,
                        summary=f"任务校验失败: {validation_error}",
                        attempts=max(task.retry_count, 1),
                        error=validation_error,
                    )
                    progress_made = True
                    continue

                task.status = TaskStatus.RUNNING
                attempt_number = task.retry_count + 1
                try:
                    result = self.agents[task.owner].run(task, {**shared_context, "task_results": results})
                except Exception as exc:
                    result = TaskResult(
                        task_id=task.id,
                        owner=task.owner,
                        status=TaskStatus.FAILED,
                        summary=f"执行异常: {exc}",
                        attempts=attempt_number,
                        error=str(exc),
                    )

                if result.status == TaskStatus.FAILED and task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    progress_made = True
                    continue

                task.status = result.status
                result.attempts = attempt_number
                results[task.id] = result
                self._update_shared_context(shared_context, result)
                progress_made = True

            if not progress_made:
                self._mark_stalled_tasks(worker_tasks, results)
                break

        return results

    def _build_responder_task(self, user_input: str, tasks: list[TaskSpec], task_results: dict[str, TaskResult]) -> TaskSpec:
        responder_task = next(task for task in tasks if task.owner == AgentRole.RESPONDER)
        responder_task.payload.setdefault("user_input", user_input)
        worker_ids = [task.id for task in tasks if task.owner != AgentRole.RESPONDER]
        unresolved = [task_id for task_id in worker_ids if task_id not in task_results]
        if unresolved:
            raise RuntimeError(f"responder 启动前仍有子任务未返回: {', '.join(unresolved)}")
        return responder_task

    def _update_shared_context(self, shared_context: dict, result: TaskResult) -> None:
        if "memory_snapshot" in result.data:
            shared_context["memory_snapshot"] = result.data["memory_snapshot"]
        if "profile_snapshot" in result.data:
            shared_context["profile_snapshot"] = result.data["profile_snapshot"]
        if "knowledge_snapshot" in result.data:
            shared_context["knowledge_snapshot"] = result.data["knowledge_snapshot"]

    def _dependencies_resolved(self, task: TaskSpec, results: dict[str, TaskResult]) -> bool:
        return all(dep in results for dep in task.depends_on)

    def _branch_condition_matches(self, task: TaskSpec, shared_context: dict, results: dict[str, TaskResult]) -> bool:
        if not task.run_if:
            return True

        for key, rule in task.run_if.items():
            value = self._resolve_path(shared_context, key)
            if isinstance(rule, dict):
                if "gt" in rule and not (isinstance(value, (int, float)) and value > rule["gt"]):
                    return False
                if "exists" in rule and bool(value is not None and value != "") != bool(rule["exists"]):
                    return False
            elif value != rule:
                return False
        return True

    def _validate_task(self, task: TaskSpec, shared_context: dict) -> str:
        for field_name in task.required_payload_fields:
            value = task.payload.get(field_name)
            if value is None or value == "":
                return f"payload 缺少必要字段: {field_name}"

        for field_name in task.required_context_fields:
            value = self._resolve_path(shared_context, field_name)
            if value is None or value == "":
                return f"context 缺少必要字段: {field_name}"
        return ""

    def _all_subtasks_returned(self, tasks: list[TaskSpec], results: dict[str, TaskResult]) -> bool:
        return all(task.id in results for task in tasks)

    def _mark_stalled_tasks(self, tasks: list[TaskSpec], results: dict[str, TaskResult]) -> None:
        for task in tasks:
            if task.id in results:
                continue
            task.status = TaskStatus.FAILED
            results[task.id] = TaskResult(
                task_id=task.id,
                owner=task.owner,
                status=TaskStatus.FAILED,
                summary="任务推进停滞，未能拿到全部子结果。",
                attempts=max(task.retry_count, 1),
                error="stalled_pipeline",
            )

    def _serialize_task_board(self, tasks: list[TaskSpec], results: dict[str, TaskResult]) -> list[dict]:
        board = []
        for task in tasks:
            task_result = results.get(task.id)
            board.append(
                {
                    "id": task.id,
                    "owner": task.owner.value,
                    "goal": task.goal,
                    "status": task_result.status.value if task_result else task.status.value,
                    "retry_count": task.retry_count,
                    "max_retries": task.max_retries,
                    "depends_on": task.depends_on,
                    "summary": task_result.summary if task_result else "",
                }
            )
        return board

    def _fallback_decision(self, user_input: str) -> SemanticDecision:
        needs_memory_write = any(word in user_input for word in ["记住", "记录", "刚刚学到", "进度", "状态", "完成", "卡住", "计划"])
        needs_profile_write = any(word in user_input for word in ["偏好", "习惯", "目标", "长期", "画像", "时间安排", "适合我"])
        needs_memory_read = any(word in user_input for word in ["我现在", "接下来", "下一步", "怎么安排", "复盘", "总结"])
        needs_profile_read = any(word in user_input for word in ["适合我", "按我的情况", "结合我的目标", "结合我的时间"])
        needs_knowledge = any(word in user_input for word in ["资料", "参考", "原理", "概念", "学习项目", "知识库", "文档"])
        needs_web = any(word in user_input for word in ["联网", "web", "搜索", "最新"])
        simple = len(user_input) < 40 and not any(
            [needs_memory_write, needs_profile_write, needs_memory_read, needs_profile_read, needs_knowledge, needs_web]
        )

        return SemanticDecision(
            response_mode="direct" if simple else "composite",
            reason="fallback 规则判断",
            read_memory=needs_memory_read,
            write_memory=needs_memory_write,
            read_profile=needs_profile_read,
            write_profile=needs_profile_write,
            use_knowledge=needs_knowledge,
            use_web=needs_web,
            knowledge_query=user_input if needs_knowledge else "",
            web_query=user_input if needs_web else "",
        )

    def _load_json(self, text: str) -> dict:
        text = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
        if fenced:
            text = fenced.group(1)
        return json.loads(text)

    def _merge_action(self, should_read: bool, should_write: bool) -> str:
        if should_read and should_write:
            return "read_write"
        if should_write:
            return "write"
        if should_read:
            return "read"
        raise ValueError("unreachable: both flags are False")

    def _resolve_path(self, data: dict, path: str):
        current = data
        for part in path.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current
