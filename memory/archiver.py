import json
import uuid
from typing import Any

from llm.base import BaseLLM
from skills.embedding_skills import dump_embedding_blob
from skills.memory_skills import extract_relations, split_content_into_chunks
from tools.memory_store import insert_chunks, insert_edge, insert_node, list_recent_nodes
from tools.session_store import delete_processed_buffer, get_unprocessed_buffer, mark_buffer_processed


class SessionArchiver:
    def archive(self, session_id: str, llm: BaseLLM, embedder: BaseLLM) -> dict[str, Any]:
        buffer_items = get_unprocessed_buffer(session_id)
        if not buffer_items:
            return {"archived_node_id": None, "edge_count": 0}

        combined_content = "\n".join(item.get("content", "") for item in buffer_items if item.get("content"))
        summary = self._draft_summary(llm=llm, session_id=session_id, buffer_items=buffer_items, combined_content=combined_content)
        node_embedding = embedder.embed([combined_content])[0]
        embedding_blob = dump_embedding_blob(node_embedding)

        node_id = str(uuid.uuid4())
        insert_node(
            {
                "id": node_id,
                "content": combined_content,
                "summary": summary,
                "embedding": embedding_blob,
                "layer": "episodic",
                "access_count": 1,
            }
        )
        chunks = split_content_into_chunks(combined_content)
        chunk_embeddings = embedder.embed(chunks) if chunks else []
        insert_chunks(
            [
                {
                    "node_id": node_id,
                    "content": chunk_content,
                    "embedding_vector": chunk_embeddings[index],
                    "chunk_index": index,
                }
                for index, chunk_content in enumerate(chunks)
            ]
        )

        edge_count = 0
        recent_nodes = list_recent_nodes(limit=5, exclude_ids=[node_id])
        for existing_node in recent_nodes:
            related_summary = existing_node.get("summary", "") or existing_node.get("content", "")[:200]
            relations = extract_relations(summary or combined_content, related_summary)
            for relation in relations:
                insert_edge(
                    source_id=node_id,
                    target_id=existing_node["id"],
                    relation_type=relation["relation_type"],
                    weight=float(relation["weight"]),
                )
                edge_count += 1

        mark_buffer_processed([item["id"] for item in buffer_items if item.get("id")])
        delete_processed_buffer(session_id)
        return {"archived_node_id": node_id, "edge_count": edge_count}

    def _draft_summary(
        self,
        llm: BaseLLM,
        session_id: str,
        buffer_items: list[dict[str, Any]],
        combined_content: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 memory manager。请把多条 session buffer 内容压缩为适合长期记忆保存的摘要。"
                    "只保留状态变化、关键事实、学习进展、计划和阻碍。"
                    "输出 80 到 200 字的纯文本摘要，不要加标题，不要重复细节。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "session_id": session_id,
                        "buffer_items": buffer_items,
                        "combined_content": combined_content,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        content = llm.chat(messages).get("content", "").strip()
        summary = content or combined_content[:200]
        return summary[:200]
