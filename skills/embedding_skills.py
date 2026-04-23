import json
from typing import Sequence


def load_embedding_blob(blob: bytes | bytearray | memoryview | str | None) -> list[float]:
    if blob is None:
        return []
    if isinstance(blob, memoryview):
        raw = blob.tobytes()
    elif isinstance(blob, (bytes, bytearray)):
        raw = bytes(blob)
    elif isinstance(blob, str):
        raw = blob.encode("utf-8")
    else:
        return []
    return [float(value) for value in json.loads(raw.decode("utf-8"))]


def dump_embedding_blob(vector: Sequence[float]) -> bytes:
    return json.dumps([float(value) for value in vector], separators=(",", ":")).encode("utf-8")
