import json
import os
import re
from pathlib import Path
from urllib import error, parse, request

from llm.base import BaseLLM
from memory.schemas import AgentRole, TaskResult, TaskSpec, TaskStatus


class BaseSubAgent:
    role: AgentRole

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def run(self, task: TaskSpec, shared_context: dict) -> TaskResult:
        raise NotImplementedError

    def resource_exists(self) -> bool:
        return False

    def resource_count(self) -> int:
        return 0


class MemoryManagerAgent(BaseSubAgent):
    role = AgentRole.MEMORY_MANAGER
    memory_path = Path("data/worklog.md")

    def run(self, task: TaskSpec, shared_context: dict) -> TaskResult:
        existing = self._read_text(self.memory_path, "未找到 worklog，用户可能是第一次使用。")
        user_input = task.payload.get("user_input", "")
        action = task.payload.get("action", "read")
        should_read = action in {"read", "read_write"}
        should_update = action in {"write", "read_write"}
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

        latest_memory = self._read_text(self.memory_path, existing)
        memory_snapshot = self._extract_relevant_memory(user_input, latest_memory) if should_read else ""

        summary = "已检查长期记忆。"
        if should_update:
            summary += " 已更新 data/worklog.md。"
        if should_read:
            summary += " 已提取当前问题相关的记忆上下文。"
        if not should_update and not should_read:
            summary += " 本轮无需读取或写入。"

        return TaskResult(
            task_id=task.id,
            owner=self.role,
            status=TaskStatus.COMPLETED,
            summary=summary,
            data={"memory_snapshot": memory_snapshot, "updated": should_update, "action": action},
        )

    def resource_exists(self) -> bool:
        return self.memory_path.exists()

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

    def _extract_relevant_memory(self, user_input: str, text: str) -> str:
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return text[:800]

        query_terms = _extract_query_terms(user_input)
        selected = []
        for line in lines:
            lowered = line.lower()
            if line.startswith("#") or any(term in lowered for term in query_terms):
                selected.append(line)

        if not selected:
            selected = lines[:8]
        return "\n".join(selected[:12])


class ProfileManagerAgent(BaseSubAgent):
    role = AgentRole.PROFILE_MANAGER
    profile_path = Path("memory/profile.md")

    def run(self, task: TaskSpec, shared_context: dict) -> TaskResult:
        existing = self._read_text(
            self.profile_path,
            "# 用户画像\n- 学习偏好: \n- 长期目标: \n- 当前约束: \n",
        )
        user_input = task.payload.get("user_input", "")
        action = task.payload.get("action", "read")
        should_read = action in {"read", "read_write"}
        should_update = action in {"write", "read_write"}

        if should_update:
            updated = self._draft_profile(existing, user_input)
            self._write_text(self.profile_path, updated)
        else:
            updated = existing

        profile_snapshot = self._extract_relevant_profile(user_input, updated) if should_read else ""

        return TaskResult(
            task_id=task.id,
            owner=self.role,
            status=TaskStatus.COMPLETED,
            summary="已检查用户画像，" + ("并完成更新。" if should_update else "本轮无需更新。"),
            data={"profile_snapshot": profile_snapshot, "updated": should_update, "action": action},
        )

    def resource_exists(self) -> bool:
        return self.profile_path.exists()

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

    def _extract_relevant_profile(self, user_input: str, text: str) -> str:
        sections = [block for block in text.split("\n## ") if block.strip()]
        if not sections:
            return text[:800]

        query_terms = _extract_query_terms(user_input)
        scored = []
        for index, section in enumerate(sections):
            normalized = section.lower()
            score = sum(1 for term in query_terms if term in normalized)
            score += 1 if index == 0 else 0
            formatted = section if index == 0 else f"## {section}"
            scored.append((score, formatted))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [section for score, section in scored if score > 0][:3]
        if not selected:
            selected = [scored[0][1]]
        return "\n\n".join(selected)


class KnowledgeManagerAgent(BaseSubAgent):
    role = AgentRole.KNOWLEDGE_MANAGER
    knowledge_dir = Path("knowledge")

    def run(self, task: TaskSpec, shared_context: dict) -> TaskResult:
        query = task.payload.get("query") or task.payload.get("user_input", "")
        snapshot, matched_files = self._build_relevant_snapshot(query)
        return TaskResult(
            task_id=task.id,
            owner=self.role,
            status=TaskStatus.COMPLETED,
            summary="已筛选本地 knowledge 资料。" if matched_files else "knowledge 目录下暂时没有命中当前问题的资料。",
            data={"knowledge_snapshot": snapshot, "matched_files": matched_files},
        )

    def resource_count(self) -> int:
        if not self.knowledge_dir.exists():
            return 0
        return sum(1 for path in self.knowledge_dir.glob("**/*") if path.is_file() and path.suffix in {".md", ".txt"})

    def _build_relevant_snapshot(self, query: str) -> tuple[str, list[str]]:
        if not self.knowledge_dir.exists():
            return "knowledge 目录下暂时没有可用参考资料。", []

        query_terms = _extract_query_terms(query)
        candidates = []
        for path in sorted(self.knowledge_dir.glob("**/*")):
            if not path.is_file() or path.suffix not in {".md", ".txt"}:
                continue

            text = path.read_text(encoding="utf-8")
            excerpt, score = _score_document(path.name, text, query_terms)
            if score > 0:
                candidates.append((score, path.as_posix(), excerpt))

        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            return "knowledge 目录下暂时没有命中当前问题的资料。", []

        top_candidates = candidates[:3]
        snapshot = "\n\n".join(f"## {path}\n{excerpt}" for _, path, excerpt in top_candidates)
        return snapshot, [path for _, path, _ in top_candidates]


class WebSearchAgent(BaseSubAgent):
    role = AgentRole.WEB_SEARCHER

    def run(self, task: TaskSpec, shared_context: dict) -> TaskResult:
        query = task.payload.get("query") or task.payload.get("user_input", "")
        provider = task.payload.get("search_provider") or os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")
        max_results = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))

        try:
            results = self._search(query, provider, max_results)
        except Exception as exc:
            return TaskResult(
                task_id=task.id,
                owner=self.role,
                status=TaskStatus.FAILED,
                summary=f"web search 调用失败: {exc}",
                data={"web_search_available": False, "provider": provider},
            )

        return TaskResult(
            task_id=task.id,
            owner=self.role,
            status=TaskStatus.COMPLETED,
            summary=f"已使用 {provider} 检索 {len(results)} 条网页结果。",
            data={"web_search_available": True, "provider": provider, "web_results": results},
        )

    def _search(self, query: str, provider: str, max_results: int) -> list[dict]:
        normalized = provider.strip().lower()
        if normalized != "duckduckgo":
            raise ValueError(f"暂不支持的搜索提供商: {provider}")
        return self._search_duckduckgo(query, max_results)

    def _search_duckduckgo(self, query: str, max_results: int) -> list[dict]:
        if not query.strip():
            return []

        encoded_query = parse.urlencode({"q": query})
        req = request.Request(
            url=f"https://html.duckduckgo.com/html/?{encoded_query}",
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
                )
            },
        )
        try:
            with request.urlopen(req, timeout=15) as response:
                html = response.read().decode("utf-8", errors="ignore")
        except error.URLError as exc:
            raise RuntimeError(f"DuckDuckGo 请求失败: {exc}") from exc

        results = []
        title_pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            re.S,
        )
        snippet_pattern = re.compile(r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.S)
        snippets = [match.group("snippet") for match in snippet_pattern.finditer(html)]

        for index, match in enumerate(title_pattern.finditer(html)):
            if len(results) >= max_results:
                break
            results.append(
                {
                    "title": _strip_html(match.group("title")),
                    "url": _strip_html(match.group("href")),
                    "snippet": _strip_html(snippets[index]) if index < len(snippets) else "",
                }
            )
        return results


class ResponderAgent(BaseSubAgent):
    role = AgentRole.RESPONDER

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
            "你是 responder agent，负责对用户输出最终答案。"
            "请优先利用上游任务结果，回答要清楚、具体、自然。"
            "如果上游有失败或未实现的能力，要诚实说明，不要假装已经联网搜索。"
        )
        user_payload = {
            "user_input": task.payload.get("user_input", ""),
            "history_tail": shared_context.get("history", [])[-6:],
            "controller_decision": shared_context.get("controller_decision", {}),
            "task_board": shared_context.get("task_board", []),
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


def _extract_query_terms(text: str) -> list[str]:
    parts = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_+-]{3,}", text.lower())
    seen = set()
    terms = []
    for part in parts:
        if part not in seen:
            seen.add(part)
            terms.append(part)
    return terms[:12]


def _score_document(file_name: str, text: str, query_terms: list[str]) -> tuple[str, int]:
    lowered_text = text.lower()
    lowered_name = file_name.lower()
    score = 0
    hit_positions = []

    for term in query_terms:
        if term in lowered_name:
            score += 3
        if term in lowered_text:
            score += min(lowered_text.count(term), 5)
            hit_positions.append(lowered_text.find(term))

    if score == 0:
        return "", 0

    excerpt = _extract_excerpt(text, hit_positions[0] if hit_positions else 0)
    return excerpt, score


def _extract_excerpt(text: str, center: int, window: int = 240) -> str:
    if not text:
        return ""
    start = max(center - window // 2, 0)
    end = min(start + window, len(text))
    return text[start:end].strip()


def _strip_html(value: str) -> str:
    cleaned = re.sub(r"<[^>]+>", "", value)
    return (
        cleaned.replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#x27;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .strip()
    )
