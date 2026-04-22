import json
from pathlib import Path

from llm.base import BaseLLM
from memory.schemas import AgentRole, TaskResult, TaskSpec, TaskStatus


class BaseSubAgent:
    role: AgentRole

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def run(self, task: TaskSpec, shared_context: dict) -> TaskResult:
        raise NotImplementedError


class MemoryManagerAgent(BaseSubAgent):
    role = AgentRole.MEMORY_MANAGER
    memory_path = Path("data/worklog.md")

    def run(self, task: TaskSpec, shared_context: dict) -> TaskResult:
        existing = self._read_text(self.memory_path, "未找到 worklog，用户可能是第一次使用。")
        user_input = task.payload.get("user_input", "")

        should_update = any(word in user_input for word in ["记住", "记录", "进度", "完成", "卡住", "计划"])
        note = ""
        if should_update:
            note = self._draft_note(
                system_prompt=(
                    "你是 memory manager。请把用户输入压缩成适合长期保留的 Markdown 记录。"
                    "只保留状态变化、学习进度、下一步计划，不要闲聊。"
                ),
                user_input=user_input,
                existing=existing,
            )
            self._write_text(self.memory_path, note)

        summary = "已检查长期记忆。"
        if should_update:
            summary += " 已更新 data/worklog.md。"
        else:
            summary += " 本轮没有足够强的记忆写入信号。"

        return TaskResult(
            task_id=task.id,
            owner=self.role,
            status=TaskStatus.COMPLETED,
            summary=summary,
            data={"memory_snapshot": self._read_text(self.memory_path, existing), "updated": should_update},
        )

    def _draft_note(self, system_prompt: str, user_input: str, existing: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"existing_memory": existing[:3000], "new_input": user_input},
                    ensure_ascii=False,
                ),
            },
        ]
        content = self.llm.chat(messages).get("content", "").strip()
        return content or existing

    def _read_text(self, path: Path, default: str) -> str:
        if not path.exists():
            return default
        return path.read_text(encoding="utf-8")

    def _write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


class ProfileManagerAgent(BaseSubAgent):
    role = AgentRole.PROFILE_MANAGER
    profile_path = Path("memory/profile.md")

    def run(self, task: TaskSpec, shared_context: dict) -> TaskResult:
        existing = self._read_text(
            self.profile_path,
            "# 用户画像\n- 学习偏好: \n- 长期目标: \n- 当前约束: \n",
        )
        user_input = task.payload.get("user_input", "")
        should_update = any(word in user_input for word in ["偏好", "习惯", "目标", "长期", "适合我", "时间"])

        if should_update:
            updated = self._draft_profile(existing, user_input)
            self._write_text(self.profile_path, updated)
        else:
            updated = existing

        return TaskResult(
            task_id=task.id,
            owner=self.role,
            status=TaskStatus.COMPLETED,
            summary="已检查用户画像，" + ("并完成更新。" if should_update else "本轮无需更新。"),
            data={"profile_snapshot": updated, "updated": should_update},
        )

    def _draft_profile(self, existing: str, user_input: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 profile manager。请根据用户新输入更新用户画像。"
                    "输出 Markdown，保留稳定偏好、长期目标、时间预算、常见阻碍。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"existing_profile": existing[:3000], "new_input": user_input},
                    ensure_ascii=False,
                ),
            },
        ]
        content = self.llm.chat(messages).get("content", "").strip()
        return content or existing

    def _read_text(self, path: Path, default: str) -> str:
        if not path.exists():
            return default
        return path.read_text(encoding="utf-8")

    def _write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


class KnowledgeManagerAgent(BaseSubAgent):
    role = AgentRole.KNOWLEDGE_MANAGER
    knowledge_dir = Path("knowledge")

    def run(self, task: TaskSpec, shared_context: dict) -> TaskResult:
        docs = []
        if self.knowledge_dir.exists():
            for path in sorted(self.knowledge_dir.glob("**/*")):
                if path.is_file() and path.suffix in {".md", ".txt"}:
                    docs.append(f"## {path.name}\n{path.read_text(encoding='utf-8')[:2000]}")

        snapshot = "\n\n".join(docs).strip() or "knowledge 目录下暂时没有可用参考资料。"
        return TaskResult(
            task_id=task.id,
            owner=self.role,
            status=TaskStatus.COMPLETED,
            summary="已读取本地 knowledge 资料。",
            data={"knowledge_snapshot": snapshot},
        )


class WebSearchAgent(BaseSubAgent):
    role = AgentRole.WEB_SEARCHER

    def run(self, task: TaskSpec, shared_context: dict) -> TaskResult:
        return TaskResult(
            task_id=task.id,
            owner=self.role,
            status=TaskStatus.SKIPPED,
            summary="当前仓库还没有接入真实 web search 工具，这个节点先保留为占位。",
            data={"web_search_available": False},
        )


class InteractionAgent(BaseSubAgent):
    role = AgentRole.INTERACTION

    def stream_answer(self, task: TaskSpec, shared_context: dict):
        messages = self._build_messages(task, shared_context)
        for chunk in self.llm.chat_stream(messages):
            if chunk["type"] == "text":
                yield chunk["content"]

    def run(self, task: TaskSpec, shared_context: dict) -> TaskResult:
        content = self.llm.chat(self._build_messages(task, shared_context)).get("content", "")
        return TaskResult(
            task_id=task.id,
            owner=self.role,
            status=TaskStatus.COMPLETED,
            summary="已生成最终回答。",
            data={"answer": content},
        )

    def _build_messages(self, task: TaskSpec, shared_context: dict) -> list[dict]:
        system_prompt = (
            "你是 interaction agent，负责对用户输出最终答案。"
            "请优先利用上游任务结果，回答要清楚、具体、自然。"
            "如果上游有未实现的能力，要诚实说明，不要假装已经联网搜索。"
        )
        user_payload = {
            "user_input": task.payload.get("user_input", ""),
            "history_tail": shared_context.get("history", [])[-6:],
            "memory_snapshot": shared_context.get("memory_snapshot", ""),
            "profile_snapshot": shared_context.get("profile_snapshot", ""),
            "knowledge_snapshot": shared_context.get("knowledge_snapshot", ""),
            "task_results": {
                task_id: result.model_dump()
                for task_id, result in shared_context.get("task_results", {}).items()
            },
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

