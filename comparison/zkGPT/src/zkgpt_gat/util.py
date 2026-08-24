from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator


def zk_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_json(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_id(path: pathlib.Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest_file(path),
    }


def write_json(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: pathlib.Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def np_digest(array: Any, h: hashlib._Hash | None = None, label: str = "") -> str:
    import numpy as np

    local = h is None
    if h is None:
        h = hashlib.sha256()
    arr = np.ascontiguousarray(array)
    h.update(label.encode("utf-8"))
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    if arr.dtype == object:
        for value in arr.ravel().tolist():
            h.update(str(value).encode("ascii"))
            h.update(b",")
    else:
        h.update(memoryview(arr).cast("B"))
    return h.hexdigest() if local else ""


def update_digest_obj(h: hashlib._Hash, label: str, data: Any) -> None:
    h.update(label.encode("utf-8"))
    h.update(canonical_json(data).encode("utf-8"))


def update_digest_np(h: hashlib._Hash, label: str, array: Any) -> None:
    np_digest(array, h, label)


@dataclass
class TimerBook:
    values: Dict[str, float]

    def add(self, name: str, ms: float) -> None:
        self.values[name] = self.values.get(name, 0.0) + ms

    def get(self, name: str) -> float:
        return float(self.values.get(name, 0.0))


@contextlib.contextmanager
def timed(book: TimerBook, name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        book.add(name, (time.perf_counter() - start) * 1000.0)


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def stable_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    return env
