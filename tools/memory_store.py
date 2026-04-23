import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from skills.embedding_skills import dump_embedding_blob


DB_PATH = Path("data/memory.db")
CHROMA_PATH = Path("data/chroma")
CHROMA_COLLECTION_NAME = "memory_chunks"


def _ensure_parent_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _get_connection() -> sqlite3.Connection:
    _ensure_parent_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _initialize_schema(conn)
    return conn


def _get_chroma_collection():
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("缺少 chromadb 依赖，请先安装 requirements.txt 中的 chromadb。") from exc

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_nodes (
            id TEXT PRIMARY KEY,
            content TEXT,
            summary TEXT,
            embedding BLOB,
            layer TEXT,
            created_at REAL,
            last_accessed REAL,
            access_count INTEGER DEFAULT 0,
            decay_rate REAL DEFAULT 0.01
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_edges (
            source_id TEXT,
            target_id TEXT,
            relation_type TEXT,
            weight REAL DEFAULT 1.0,
            created_at REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_chunks (
            id TEXT PRIMARY KEY,
            node_id TEXT,
            content TEXT,
            embedding BLOB,
            chunk_index INTEGER,
            created_at REAL
        )
        """
    )
    conn.commit()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def insert_node(node: dict[str, Any]) -> str:
    node_id = str(node.get("id") or uuid.uuid4())
    created_at = float(node.get("created_at", time.time()))
    last_accessed = float(node.get("last_accessed", created_at))
    access_count = int(node.get("access_count", 0))
    decay_rate = float(node.get("decay_rate", 0.01))

    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO memory_nodes (
                id, content, summary, embedding, layer,
                created_at, last_accessed, access_count, decay_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node_id,
                node.get("content", ""),
                node.get("summary", ""),
                node.get("embedding"),
                node.get("layer", "episodic"),
                created_at,
                last_accessed,
                access_count,
                decay_rate,
            ),
        )
        conn.commit()
    return node_id


def insert_edge(source_id: str, target_id: str, relation_type: str, weight: float) -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO memory_edges (
                source_id, target_id, relation_type, weight, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (source_id, target_id, relation_type, float(weight), time.time()),
        )
        conn.commit()


def insert_chunk(chunk: dict[str, Any]) -> str:
    chunk_id = str(chunk.get("id") or uuid.uuid4())
    embedding_vector = [float(value) for value in chunk.get("embedding_vector", [])]
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO memory_chunks (
                id, node_id, content, embedding, chunk_index, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                chunk.get("node_id", ""),
                chunk.get("content", ""),
                chunk.get("embedding") or dump_embedding_blob(embedding_vector),
                int(chunk.get("chunk_index", 0)),
                float(chunk.get("created_at", time.time())),
            ),
        )
        conn.commit()
    if embedding_vector:
        _get_chroma_collection().add(
            ids=[chunk_id],
            embeddings=[embedding_vector],
            documents=[chunk.get("content", "")],
            metadatas=[
                {
                    "node_id": chunk.get("node_id", ""),
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                }
            ],
        )
    return chunk_id


def insert_chunks(chunks: list[dict[str, Any]]) -> list[str]:
    if not chunks:
        return []
    chunk_ids: list[str] = []
    chroma_ids: list[str] = []
    chroma_embeddings: list[list[float]] = []
    chroma_documents: list[str] = []
    chroma_metadatas: list[dict[str, Any]] = []
    with _get_connection() as conn:
        for chunk in chunks:
            chunk_id = str(chunk.get("id") or uuid.uuid4())
            chunk_ids.append(chunk_id)
            embedding_vector = [float(value) for value in chunk.get("embedding_vector", [])]
            conn.execute(
                """
                INSERT INTO memory_chunks (
                    id, node_id, content, embedding, chunk_index, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    chunk.get("node_id", ""),
                    chunk.get("content", ""),
                    chunk.get("embedding") or dump_embedding_blob(embedding_vector),
                    int(chunk.get("chunk_index", 0)),
                    float(chunk.get("created_at", time.time())),
                ),
            )
            if embedding_vector:
                chroma_ids.append(chunk_id)
                chroma_embeddings.append(embedding_vector)
                chroma_documents.append(chunk.get("content", ""))
                chroma_metadatas.append(
                    {
                        "node_id": chunk.get("node_id", ""),
                        "chunk_index": int(chunk.get("chunk_index", 0)),
                    }
                )
        conn.commit()
    if chroma_ids:
        _get_chroma_collection().add(
            ids=chroma_ids,
            embeddings=chroma_embeddings,
            documents=chroma_documents,
            metadatas=chroma_metadatas,
        )
    return chunk_ids


def get_node(node_id: str) -> dict[str, Any] | None:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM memory_nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
    return _row_to_dict(row)


def update_node_access(node_id: str) -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            UPDATE memory_nodes
            SET last_accessed = ?, access_count = access_count + 1
            WHERE id = ?
            """,
            (time.time(), node_id),
        )
        conn.commit()


def query_nodes_by_ids(ids: list[str]) -> list[dict[str, Any]]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    with _get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM memory_nodes WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    node_map = {row["id"]: dict(row) for row in rows}
    return [node_map[node_id] for node_id in ids if node_id in node_map]


def query_chunks_by_node_id(node_id: str) -> list[dict[str, Any]]:
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM memory_chunks
            WHERE node_id = ?
            ORDER BY chunk_index ASC
            """,
            (node_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def query_all_chunks() -> list[dict[str, Any]]:
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM memory_chunks
            ORDER BY created_at DESC, chunk_index ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_chunk_count() -> int:
    with _get_connection() as conn:
        row = conn.execute("SELECT COUNT(1) AS count FROM memory_chunks").fetchone()
    return int(row["count"]) if row else 0


def query_similar_chunks(
    query_embedding: list[float],
    top_k: int = 8,
    allowed_node_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not query_embedding:
        return []

    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if allowed_node_ids:
        query_kwargs["where"] = {"node_id": {"$in": allowed_node_ids}}

    result = _get_chroma_collection().query(**query_kwargs)
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    matches: list[dict[str, Any]] = []
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata = metadata or {}
        similarity = 1.0 - float(distance)
        matches.append(
            {
                "id": chunk_id,
                "node_id": metadata.get("node_id", ""),
                "content": document or "",
                "chunk_index": int(metadata.get("chunk_index", 0)),
                "similarity": similarity,
            }
        )
    return matches


def query_edges_by_node_ids(node_ids: list[str]) -> list[dict[str, Any]]:
    if not node_ids:
        return []
    placeholders = ",".join("?" for _ in node_ids)
    params = node_ids + node_ids
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM memory_edges
            WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
            ORDER BY weight DESC, created_at DESC
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def query_nodes_by_keywords(query: str, limit: int = 20) -> list[dict[str, Any]]:
    terms = [term.strip() for term in query.split() if term.strip()]
    if not terms:
        return list_recent_nodes(limit=limit)

    summary_clauses = []
    content_clauses = []
    where_params: list[Any] = []
    score_params: list[Any] = []
    for term in terms[:8]:
        like_value = f"%{term}%"
        summary_clauses.append("summary LIKE ?")
        content_clauses.append("content LIKE ?")
        where_params.append(like_value)
        score_params.append(like_value)
    for term in terms[:8]:
        where_params.append(f"%{term}%")

    sql = (
        "SELECT *, "
        f"(CASE WHEN {' OR '.join(summary_clauses)} THEN 1 ELSE 0 END) AS summary_hit "
        "FROM memory_nodes "
        f"WHERE {' OR '.join(summary_clauses + content_clauses)} "
        "ORDER BY summary_hit DESC, created_at DESC LIMIT ?"
    )
    params = score_params + where_params + [limit]

    with _get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def get_retrieval_threshold() -> int:
    return max(int(os.getenv("MEMORY_COARSE_FILTER_THRESHOLD", "1000")), 1)


def list_recent_nodes(limit: int = 5, exclude_ids: list[str] | None = None) -> list[dict[str, Any]]:
    exclude_ids = exclude_ids or []
    sql = "SELECT * FROM memory_nodes"
    params: list[Any] = []
    if exclude_ids:
        placeholders = ",".join("?" for _ in exclude_ids)
        sql += f" WHERE id NOT IN ({placeholders})"
        params.extend(exclude_ids)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with _get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def memory_has_nodes() -> bool:
    with _get_connection() as conn:
        row = conn.execute("SELECT COUNT(1) AS count FROM memory_nodes").fetchone()
    return bool(row and row["count"] > 0)
