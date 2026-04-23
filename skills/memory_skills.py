import math
import re
import time
from typing import Any

from tools.session_store import is_buffer_over_threshold


def compute_decay_weight(node: dict[str, Any]) -> float:
    last_accessed = float(node.get("last_accessed", node.get("created_at", time.time())))
    access_count = int(node.get("access_count", 0))
    decay_rate = float(node.get("decay_rate", 0.01))
    days_since_access = max((time.time() - last_accessed) / 86400.0, 0.0)
    return float(access_count * math.exp(-decay_rate * days_since_access))


def rank_candidates(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(nodes, key=compute_decay_weight, reverse=True)


def should_archive(session_id: str, buffer_size_threshold: int = 10) -> bool:
    return is_buffer_over_threshold(session_id, buffer_size_threshold)


def split_content_into_chunks(
    content: str,
    target_chars: int = 300,
    max_chars: int = 400,
    overlap_chars: int = 40,
) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", content) if paragraph.strip()]
    if not paragraphs:
        paragraphs = [content.strip()] if content.strip() else []

    chunks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue

        sentences = _split_sentences(paragraph)
        current = ""
        for sentence in sentences:
            if not current:
                current = sentence
                continue
            candidate = f"{current}{sentence}"
            if len(candidate) <= target_chars:
                current = candidate
                continue
            chunks.extend(_finalize_chunk(current, max_chars, overlap_chars))
            current = sentence

        if current:
            chunks.extend(_finalize_chunk(current, max_chars, overlap_chars))

    return [chunk for chunk in chunks if chunk.strip()]


def rank_nodes_for_retrieval(
    nodes: list[dict[str, Any]],
    similarity_by_node_id: dict[str, float],
    neighbor_node_ids: set[str],
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for node in nodes:
        node_id = node.get("id", "")
        similarity = float(similarity_by_node_id.get(node_id, 0.0))
        decay_score = compute_decay_weight(node)
        neighbor_bonus = 0.15 if node_id in neighbor_node_ids else 0.0
        combined_score = similarity + min(decay_score, 3.0) * 0.2 + neighbor_bonus
        ranked.append({**node, "retrieval_score": combined_score})
    ranked.sort(key=lambda item: item["retrieval_score"], reverse=True)
    return ranked


def extract_relations(content_a: str, content_b: str) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    lowered_a = content_a.lower()
    lowered_b = content_b.lower()

    temporal_words = ["然后", "接着", "之后", "今天", "明天", "昨天", "later", "next", "then", "after", "before"]
    if any(word in lowered_a for word in temporal_words) or any(word in lowered_b for word in temporal_words):
        relations.append({"relation_type": "temporal_follow", "weight": 0.6})

    tokens_a = set(_extract_terms(content_a))
    tokens_b = set(_extract_terms(content_b))
    overlap = tokens_a & tokens_b
    if overlap:
        overlap_weight = min(1.0, 0.2 + 0.1 * len(overlap))
        relations.append({"relation_type": "domain_related", "weight": overlap_weight})

    causal_words = ["因为", "所以", "导致", "caused", "because", "therefore", "so that"]
    if any(word in lowered_a for word in causal_words) or any(word in lowered_b for word in causal_words):
        relations.append({"relation_type": "causal", "weight": 0.5})

    return relations


def _extract_terms(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_+-]{3,}", text.lower())


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[。！？!?；;])", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _finalize_chunk(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    step = max(max_chars - overlap_chars, 1)
    while start < len(text):
        chunk = text[start : start + max_chars].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks
