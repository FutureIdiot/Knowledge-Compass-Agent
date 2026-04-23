import json
import os
import re
import uuid
from pathlib import Path
from urllib import error, parse, request

from llm.base import BaseLLM
from llm.factory import get_embedding_llm
from memory.archiver import SessionArchiver
from memory.schemas import AgentRole, TaskResult, TaskSpec, TaskStatus
from skills.memory_skills import rank_nodes_for_retrieval, should_archive
from tools.memory_store import (
    get_chunk_count,
    get_retrieval_threshold,
    memory_has_nodes,
    query_edges_by_node_ids,
    query_nodes_by_ids,
    query_nodes_by_keywords,
    query_similar_chunks,
    update_node_access,
)
from tools.session_store import get_unprocessed_buffer, insert_session_buffer


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

    def _read_text(self, path: Path, default: str) -> str:
        if not path.exists():
            return default
        return path.read_text(encoding="utf-8")

    def _write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


class MemoryManagerAgent(BaseSubAgent):
    role = AgentRole.MEMORY_MANAGER
    memory_snapshot_limit = 800

    def __init__(self, llm: BaseLLM):
        super().__init__(llm)
        self.archiver = SessionArchiver()
        self.embedder = get_embedding_llm()
        memory_has_nodes()

    def run(self, task: TaskSpec, shared_context: dict) -> TaskResult:
        user_input = task.payload.get("user_input", "")
        action = task.payload.get("action", "read")
        should_read = action in {"read", "read_write"}
        should_update = action in {"write", "read_write"}
        session_id = str(shared_context.get("session_id") or uuid.uuid4())
        shared_context["session_id"] = session_id

        if should_update:
            insert_session_buffer(session_id, user_input)

        archive_result = {"archived_node_id": None, "edge_count": 0}
        if should_update and should_archive(session_id):
            archive_result = self.archiver.archive(session_id, self.llm, self.embedder)
        retrieval_mode = ""
        if should_read:
            memory_snapshot, retrieval_mode = self._build_memory_snapshot(session_id, user_input)
        else:
            memory_snapshot = ""

        summary = "已检查长期记忆。"
        if should_update:
            summary += " 已写入 session buffer。"
        if should_read:
            summary += " 已提取当前问题相关的记忆上下文。"
            if retrieval_mode:
                summary += f" 检索模式: {retrieval_mode}。"
        if archive_result.get("archived_node_id"):
            summary += " 已触发归档。"
        if not should_update and not should_read:
            summary += " 本轮无需读取或写入。"

        return TaskResult(
            task_id=task.id,
            owner=self.role,
            status=TaskStatus.COMPLETED,
            summary=summary,
            data={
                "memory_snapshot": memory_snapshot,
                "updated": should_update,
                "action": action,
                "session_id": session_id,
                "archive_result": archive_result,
                "retrieval_mode": retrieval_mode,
            },
        )

    def resource_exists(self) -> bool:
        memory_db_path = Path("data/memory.db")
        return memory_db_path.exists() and memory_has_nodes()

    def _build_memory_snapshot(self, session_id: str, user_input: str) -> tuple[str, str]:
        buffer_items = get_unprocessed_buffer(session_id)
        recent_buffer = buffer_items[-5:]

        ranked_nodes, retrieval_mode = self._retrieve_ranked_nodes(user_input)
        ranked_nodes = ranked_nodes[:3]

        for node in ranked_nodes:
            if node.get("id"):
                update_node_access(node["id"])

        sections = []
        if recent_buffer:
            sections.append(
                "## Session Buffer\n"
                + "\n".join(
                    f"- {self._compact_text(item.get('content', ''), 120)}"
                    for item in recent_buffer
                    if item.get("content")
                )
            )
        if ranked_nodes:
            sections.append(
                "## Long-term Memory\n"
                + "\n".join(
                    f"- [{node.get('layer', 'episodic')}] "
                    f"{self._compact_text(node.get('summary') or node.get('content', ''), 200)}"
                    for node in ranked_nodes
                )
            )

        if not sections:
            return "暂无可用记忆。", retrieval_mode
        return self._truncate_memory_snapshot("\n\n".join(sections)), retrieval_mode

    def _retrieve_ranked_nodes(self, user_input: str) -> tuple[list[dict], str]:
        query_embeddings = self.embedder.embed([user_input])
        query_embedding = query_embeddings[0] if query_embeddings else []
        coarse_result = self._coarse_filter(user_input)

        if query_embedding and coarse_result["mode"] in {"direct_chroma", "like_prefilter"}:
            try:
                chunk_candidates = query_similar_chunks(
                    query_embedding,
                    top_k=8,
                    allowed_node_ids=coarse_result["node_ids"] or None,
                )
                retrieval_mode = coarse_result["mode"]
            except Exception:
                chunk_candidates = []
                retrieval_mode = "fallback_like"
        else:
            chunk_candidates = []
            retrieval_mode = "fallback_like"

        direct_node_ids: list[str] = []
        similarity_by_node_id: dict[str, float] = {}
        for chunk in chunk_candidates:
            node_id = str(chunk.get("node_id", ""))
            if not node_id:
                continue
            similarity_by_node_id[node_id] = max(
                similarity_by_node_id.get(node_id, 0.0),
                float(chunk.get("similarity", 0.0)),
            )
            if node_id not in direct_node_ids:
                direct_node_ids.append(node_id)

        if not direct_node_ids:
            keyword_nodes = query_nodes_by_ids(coarse_result["node_ids"]) if coarse_result["node_ids"] else query_nodes_by_keywords(user_input, limit=8)
            return (
                rank_nodes_for_retrieval(
                    keyword_nodes,
                    similarity_by_node_id={str(node.get("id", "")): 0.3 for node in keyword_nodes},
                    neighbor_node_ids=set(),
                ),
                "fallback_like",
            )

        neighbor_node_ids: set[str] = set()
        for edge in query_edges_by_node_ids(direct_node_ids):
            source_id = edge.get("source_id")
            target_id = edge.get("target_id")
            if source_id in direct_node_ids and target_id:
                neighbor_node_ids.add(str(target_id))
            if target_id in direct_node_ids and source_id:
                neighbor_node_ids.add(str(source_id))

        all_node_ids = direct_node_ids + [node_id for node_id in neighbor_node_ids if node_id not in direct_node_ids]
        return (
            rank_nodes_for_retrieval(
                query_nodes_by_ids(all_node_ids),
                similarity_by_node_id=similarity_by_node_id,
                neighbor_node_ids=neighbor_node_ids,
            ),
            retrieval_mode,
        )

    def _coarse_filter(self, user_input: str) -> dict[str, object]:
        chunk_count = get_chunk_count()
        threshold = get_retrieval_threshold()
        if chunk_count < threshold:
            return {"mode": "direct_chroma", "node_ids": []}

        keyword_nodes = query_nodes_by_keywords(user_input, limit=20)
        return {
            "mode": "like_prefilter",
            "node_ids": [str(node.get("id", "")) for node in keyword_nodes if node.get("id")],
        }

    def _truncate_memory_snapshot(self, snapshot: str) -> str:
        if len(snapshot) <= self.memory_snapshot_limit:
            return snapshot
        truncated = snapshot[: self.memory_snapshot_limit - 3].rstrip()
        return f"{truncated}..."

    def _compact_text(self, text: str, max_chars: int) -> str:
        compact = " ".join(text.split())
        if len(compact) <= max_chars:
            return compact
        return f"{compact[: max_chars - 3].rstrip()}..."


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
                task_id: result.summary
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
