import json
import time
import uuid
from pathlib import Path
from typing import Any


RUNTIME_DIR = Path("runtime")


def _session_path(session_id: str) -> Path:
    return RUNTIME_DIR / f"session_{session_id}.json"


def _processed_path(session_id: str) -> Path:
    return RUNTIME_DIR / f"session_{session_id}.processed.json"


def _ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def insert_session_buffer(session_id: str, content: str) -> str:
    _ensure_runtime_dir()
    buffer_path = _session_path(session_id)
    items = _read_json(buffer_path, [])
    entry_id = str(uuid.uuid4())
    items.append({"id": entry_id, "content": content, "created_at": time.time()})
    _write_json(buffer_path, items)
    return entry_id


def get_unprocessed_buffer(session_id: str) -> list[dict[str, Any]]:
    _ensure_runtime_dir()
    buffer_path = _session_path(session_id)
    processed_path = _processed_path(session_id)
    items = _read_json(buffer_path, [])
    processed_ids = set(_read_json(processed_path, []))
    return [item for item in items if item.get("id") not in processed_ids]


def mark_buffer_processed(ids: list[str]) -> None:
    if not ids:
        return
    _ensure_runtime_dir()
    pending_ids = set(ids)
    for buffer_path in RUNTIME_DIR.glob("session_*.json"):
        if buffer_path.name.endswith(".processed.json"):
            continue
        session_id = buffer_path.stem.removeprefix("session_")
        items = _read_json(buffer_path, [])
        item_ids = {item.get("id") for item in items}
        matched_ids = sorted(pending_ids & item_ids)
        if not matched_ids:
            continue
        processed_path = _processed_path(session_id)
        existing = set(_read_json(processed_path, []))
        existing.update(matched_ids)
        _write_json(processed_path, sorted(existing))


def delete_processed_buffer(session_id: str) -> None:
    _ensure_runtime_dir()
    buffer_path = _session_path(session_id)
    processed_path = _processed_path(session_id)

    if not buffer_path.exists():
        if processed_path.exists():
            processed_path.unlink()
        return

    items = _read_json(buffer_path, [])
    processed_ids = set(_read_json(processed_path, []))
    remaining = [item for item in items if item.get("id") not in processed_ids]

    if remaining:
        _write_json(buffer_path, remaining)
    else:
        buffer_path.unlink()

    if processed_path.exists():
        processed_path.unlink()


def is_buffer_over_threshold(session_id: str, threshold: int) -> bool:
    return len(get_unprocessed_buffer(session_id)) >= threshold
