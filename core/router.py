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
2. 用户提到进度变化、计划变动、长期目标、偏好变化、需要记住的事情时，加入 memory_manager 和/或 profile_manager。
3. 用户询问项目相关客观资料、学习参考、术语、背景知识时，加入 knowledge_manager。
4. 用户明确要求联网查资料、最新信息、网页搜索时，加入 web_searcher。
5. composite 模式下，interaction 任务放在最后，并依赖前面的任务结果。
6. 不要发明不存在的数据；payload 只写完成任务真正需要的最小信息。
"""


class RouterAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def plan(
        self,
        user_input: str,
        history: list | None = None,
        memory_snapshot: str = "",
        profile_snapshot: str = "",
        knowledge_snapshot: str = "",
    ) -> ExecutionPlan:
        history = history or []
        messages = [
            {"role": "system", "content": ROUTER_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_input": user_input,
                        "history_tail": history[-6:],
                        "memory_snapshot": memory_snapshot[:2000],
                        "profile_snapshot": profile_snapshot[:2000],
                        "knowledge_snapshot": knowledge_snapshot[:2000],
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

        needs_memory = any(word in user_input for word in ["记住", "记录", "刚刚学到", "进度", "状态"])
        needs_profile = any(word in user_input for word in ["偏好", "习惯", "目标", "长期", "画像"])
        needs_knowledge = any(word in user_input for word in ["资料", "参考", "原理", "概念", "学习项目"])
        needs_web = any(word in user_input for word in ["联网", "web", "搜索", "最新"])
        simple = len(user_input) < 40 and not any([needs_memory, needs_profile, needs_knowledge, needs_web])

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

        if needs_memory:
            tasks.append(
                TaskSpec(
                    id="t1",
                    owner=AgentRole.MEMORY_MANAGER,
                    goal="review_memory_relevance",
                    instructions="判断这次输入是否值得写入长期记忆，并返回相关记忆摘要。",
                    payload={"user_input": user_input},
                )
            )
        if needs_profile:
            tasks.append(
                TaskSpec(
                    id=f"t{len(tasks) + 1}",
                    owner=AgentRole.PROFILE_MANAGER,
                    goal="review_profile_update",
                    instructions="判断用户画像是否需要更新，并给出结构化更新建议。",
                    payload={"user_input": user_input},
                )
            )
        if needs_knowledge:
            tasks.append(
                TaskSpec(
                    id=f"t{len(tasks) + 1}",
                    owner=AgentRole.KNOWLEDGE_MANAGER,
                    goal="retrieve_local_knowledge",
                    instructions="读取本地 knowledge 目录中和问题相关的参考信息。",
                    payload={"user_input": user_input},
                )
            )
        if needs_web:
            tasks.append(
                TaskSpec(
                    id=f"t{len(tasks) + 1}",
                    owner=AgentRole.WEB_SEARCHER,
                    goal="search_web",
                    instructions="联网搜索最新或外部资料。",
                    payload={"user_input": user_input},
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

