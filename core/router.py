import json
import re

from llm.base import BaseLLM
from memory.schemas import AgentRole, ExecutionPlan, TaskSpec


ROUTER_PROMPT = """你是多 Agent 学习系统的 Router。
你的任务不是回答用户，而是决定是否需要派发子任务。

请只输出 JSON，不要输出 Markdown，不要解释。
输出结构：
{
  "mode": "direct" | "composite",
  "rationale": "一句话说明为什么这样派发",
  "tasks": [
    {
      "id": "t1",
      "owner": "memory_manager|profile_manager|knowledge_manager|web_searcher|interaction",
      "goal": "任务目标",
      "instructions": "给子 agent 的执行说明",
      "payload": {},
      "depends_on": []
    }
  ]
}

规则：
1. 简单寒暄、日常问答、无需额外加工的短问题，输出 direct，只保留一个 interaction 任务。
2. 你需要同时判断“是否要读取上下文”和“是否要写入长期状态”。manager 只有在真的需要时才调用。
3. memory_manager 负责学习进度/计划/状态的读取与写入；profile_manager 负责稳定偏好、长期目标、约束的读取与写入。
4. 用户提到进度变化、计划变动、长期目标、偏好变化、需要记住的事情时，加入 memory_manager 和/或 profile_manager，并在 payload 里写明 action: "read"|"write"|"read_write"。
5. 用户询问项目相关客观资料、学习参考、术语、背景知识时，加入 knowledge_manager；只有需要本地知识时才调它。
6. 用户明确要求联网查资料、最新信息、网页搜索时，加入 web_searcher。
7. 如果 interaction 想利用 memory/profile/knowledge/web 结果，它必须依赖对应任务。
8. payload 只写完成任务真正需要的最小信息，可以包含 query / action / search_provider。
9. composite 模式下，interaction 任务放在最后，并依赖前面的任务结果。
"""


class RouterAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def plan(
        self,
        user_input: str,
        history: list | None = None,
        runtime_state: dict | None = None,
    ) -> ExecutionPlan:
        history = history or []
        runtime_state = runtime_state or {}
        messages = [
            {"role": "system", "content": ROUTER_PROMPT},
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
            return ExecutionPlan.model_validate(self._load_json(content))
        except Exception:
            return self._fallback_plan(user_input)

    def _load_json(self, text: str) -> dict:
        text = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
        if fenced:
            text = fenced.group(1)
        return json.loads(text)

    def _fallback_plan(self, user_input: str) -> ExecutionPlan:
        lowered = user_input.lower()
        tasks: list[TaskSpec] = []

        needs_memory_write = any(word in user_input for word in ["记住", "记录", "刚刚学到", "进度", "状态", "完成", "卡住", "计划"])
        needs_profile_write = any(word in user_input for word in ["偏好", "习惯", "目标", "长期", "画像", "时间安排", "适合我"])
        needs_memory_read = any(word in user_input for word in ["我现在", "接下来", "下一步", "怎么安排", "复盘", "总结"])
        needs_profile_read = any(word in user_input for word in ["适合我", "按我的情况", "结合我的目标", "结合我的时间"])
        needs_knowledge = any(word in user_input for word in ["资料", "参考", "原理", "概念", "学习项目"])
        needs_web = any(word in user_input for word in ["联网", "web", "搜索", "最新"])
        simple = len(user_input) < 40 and not any(
            [needs_memory_write, needs_profile_write, needs_memory_read, needs_profile_read, needs_knowledge, needs_web]
        )

        if simple:
            return ExecutionPlan(
                mode="direct",
                rationale="问题较短，且不需要额外加工。",
                tasks=[
                    TaskSpec(
                        id="t1",
                        owner=AgentRole.INTERACTION,
                        goal="direct_answer",
                        instructions="直接回答用户问题。",
                        payload={"user_input": user_input},
                    )
                ],
            )

        if needs_memory_read or needs_memory_write:
            tasks.append(
                TaskSpec(
                    id="t1",
                    owner=AgentRole.MEMORY_MANAGER,
                    goal="manage_memory_context",
                    instructions="按需读取或写入长期记忆，只保留和当前输入相关的状态上下文。",
                    payload={
                        "user_input": user_input,
                        "action": self._merge_action(needs_memory_read, needs_memory_write),
                    },
                )
            )
        if needs_profile_read or needs_profile_write:
            tasks.append(
                TaskSpec(
                    id=f"t{len(tasks) + 1}",
                    owner=AgentRole.PROFILE_MANAGER,
                    goal="manage_profile_context",
                    instructions="按需读取或写入用户画像，保留稳定偏好、长期目标和约束。",
                    payload={
                        "user_input": user_input,
                        "action": self._merge_action(needs_profile_read, needs_profile_write),
                    },
                )
            )
        if needs_knowledge:
            tasks.append(
                TaskSpec(
                    id=f"t{len(tasks) + 1}",
                    owner=AgentRole.KNOWLEDGE_MANAGER,
                    goal="retrieve_local_knowledge",
                    instructions="只检索本地 knowledge 中与当前问题最相关的资料，不要全量转储。",
                    payload={"user_input": user_input, "query": user_input},
                )
            )
        if needs_web:
            tasks.append(
                TaskSpec(
                    id=f"t{len(tasks) + 1}",
                    owner=AgentRole.WEB_SEARCHER,
                    goal="search_web",
                    instructions="联网搜索最新或外部资料。",
                    payload={"user_input": user_input, "query": user_input},
                )
            )

        dependencies = [task.id for task in tasks]
        tasks.append(
            TaskSpec(
                id=f"t{len(tasks) + 1}",
                owner=AgentRole.INTERACTION,
                goal="compose_final_answer",
                instructions="综合前置任务的结果，对用户输出最终回答。",
                payload={"user_input": user_input},
                depends_on=dependencies,
            )
        )

        return ExecutionPlan(
            mode="composite",
            rationale="问题需要结合记忆、画像或参考资料进行处理。",
            tasks=tasks,
        )

    def _merge_action(self, should_read: bool, should_write: bool) -> str:
        if should_read and should_write:
            return "read_write"
        if should_write:
            return "write"
        return "read"
