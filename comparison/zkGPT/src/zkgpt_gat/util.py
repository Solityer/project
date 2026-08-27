from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import stat
import tempfile
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
    """Write JSON without following a hostile artifact-file alias.

    Production proof files are rewritten during a rerun, so ``O_EXCL`` alone
    is insufficient.  Validate an existing destination as a one-link regular
    file, write a fresh regular temporary file in the same directory, and
    atomically replace the destination.  ``os.replace`` replaces a symlink
    itself rather than following it; refusing pre-existing symlinks, hardlinks,
    FIFOs, and other special files keeps the write path fail-closed.
    """
    payload = (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    write_bytes_atomic(path, payload, label="JSON output")


def _check_atomic_parent(path: pathlib.Path, label: str) -> pathlib.Path:
    """Authenticate every lexical parent before publishing a result file."""
    path = pathlib.Path(path).expanduser()
    parent = path.parent
    # ``mkdir -p`` and ordinary ``open`` follow a symlink in any ancestor.
    # Walk every existing component with lstat so result publication cannot be
    # redirected outside the baseline by a pre-existing alias.
    current = pathlib.Path(parent.anchor) if parent.is_absolute() else pathlib.Path.cwd()
    parts = parent.parts[1:] if parent.is_absolute() else parent.parts
    for component in parts:
        current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            current.mkdir()
            info = current.lstat()
        except OSError as exc:
            raise OSError(f"cannot inspect {label} parent: {current}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise OSError(f"{label} parent must be a real directory: {current}")
    return path


def write_bytes_atomic(path: pathlib.Path, payload: bytes, *, label: str = "output") -> None:
    """Atomically publish one baseline-local owned regular file.

    Existing destinations are replaceable only when they are one-link regular
    files.  Symlinks, hardlinks, FIFOs and other special files fail closed;
    the temporary file is created in the authenticated destination directory.
    """
    path = _check_atomic_parent(path, label)
    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise OSError(f"cannot inspect {label}: {path}") from exc
    if existing is not None:
        if stat.S_ISLNK(existing.st_mode):
            raise OSError(f"{label} must not be a symlink: {path}")
        if not stat.S_ISREG(existing.st_mode):
            raise OSError(f"{label} must be a regular file: {path}")
        if existing.st_nlink != 1:
            raise OSError(f"{label} hardlink count must be one: {path}")

    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        with os.fdopen(fd, "wb") as handle:
            handle.write(bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def write_text_atomic(path: pathlib.Path, text: str, *, label: str = "text output") -> None:
    """Atomically publish UTF-8 text through the same fail-closed path."""
    write_bytes_atomic(path, str(text).encode("utf-8"), label=label)


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
