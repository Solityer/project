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


def lexical_path(path: pathlib.Path | str) -> pathlib.Path:
    """Return an absolute lexical path without following aliases.

    Production writers use this helper before any call to ``resolve``.  A
    resolved path loses the evidence that a caller supplied a symlink alias,
    which is precisely the state that must fail closed for baseline-owned
    artifacts.
    """
    value = pathlib.Path(path).expanduser()
    if not value.is_absolute():
        value = pathlib.Path.cwd() / value
    return pathlib.Path(os.path.abspath(os.path.normpath(os.fspath(value))))


def _lstat_directory_chain(
    path: pathlib.Path | str,
    *,
    label: str,
    create: bool,
) -> pathlib.Path:
    """Authenticate/create a directory chain without following aliases.

    ``Path.mkdir(..., parents=True, exist_ok=True)`` follows a pre-existing
    symlink in an ancestor.  Walking each lexical component with ``lstat``
    makes pre-seeded symlink, FIFO, socket, and regular-file components fail
    closed.  Directory hardlinks cannot be created by an unprivileged caller;
    directory components are therefore authenticated by inode type while
    regular output files are checked separately for one-link ownership.
    """
    target = lexical_path(path)
    current = pathlib.Path(target.anchor)
    components = target.parts[1:] if target.is_absolute() else target.parts
    for component in components:
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create:
                raise FileNotFoundError(f"{label} directory is missing: {current}")
            try:
                current.mkdir()
            except FileExistsError:
                # A concurrent/pre-seeded entry appeared between the check
                # and mkdir.  Inspect it below rather than following it.
                pass
            info = current.lstat()
        except OSError as exc:
            raise OSError(f"cannot inspect {label} directory: {current}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise OSError(f"{label} path component must be a real directory: {current}")
    return target


def assert_contained_path(
    root: pathlib.Path | str,
    path: pathlib.Path | str,
    label: str,
    *,
    allow_missing: bool = True,
) -> pathlib.Path:
    """Return a lexical path under ``root`` after authenticating components.

    Existing aliases and special parent components are rejected even when the
    alias resolves back inside the baseline.  Missing components are allowed
    for callers that will create them through :func:`ensure_dir` or an
    atomic writer.
    """
    baseline = lexical_path(root)
    candidate_raw = pathlib.Path(path).expanduser()
    candidate = lexical_path(candidate_raw if candidate_raw.is_absolute() else baseline / candidate_raw)
    try:
        relative = candidate.relative_to(baseline)
    except ValueError as exc:
        raise ValueError(f"{label} escapes baseline root: {candidate}") from exc
    try:
        root_info = baseline.lstat()
    except OSError as exc:
        raise OSError(f"cannot inspect baseline root for {label}: {baseline}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise OSError(f"baseline root for {label} must be a real directory: {baseline}")
    current = baseline
    for index, component in enumerate(relative.parts):
        current /= component
        is_leaf = index == len(relative.parts) - 1
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not allow_missing:
                raise FileNotFoundError(f"{label} path is missing: {current}")
            continue
        except OSError as exc:
            raise OSError(f"cannot inspect {label} path: {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise OSError(f"{label} path component must not be a symlink: {current}")
        if not is_leaf and not stat.S_ISDIR(info.st_mode):
            raise OSError(f"{label} parent component must be a real directory: {current}")
    return candidate


def file_id(path: pathlib.Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest_file(path),
    }


def write_json(
    path: pathlib.Path,
    data: Any,
    *,
    root: pathlib.Path | None = None,
    label: str = "JSON output",
) -> None:
    """Write JSON without following a hostile artifact-file alias.

    Production proof files are rewritten during a rerun, so ``O_EXCL`` alone
    is insufficient.  Validate an existing destination as a one-link regular
    file, write a fresh regular temporary file in the same directory, and
    atomically replace the destination.  ``os.replace`` replaces a symlink
    itself rather than following it; refusing pre-existing symlinks, hardlinks,
    FIFOs, and other special files keeps the write path fail-closed.
    """
    payload = (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    write_bytes_atomic(path, payload, label=label, root=root)


def _check_atomic_parent(
    path: pathlib.Path,
    label: str,
    root: pathlib.Path | None = None,
) -> pathlib.Path:
    """Authenticate every lexical parent before publishing a result file."""
    path = lexical_path(path)
    if root is not None:
        path = assert_contained_path(root, path, label, allow_missing=True)
    parent = path.parent
    # ``mkdir -p`` and ordinary ``open`` follow a symlink in any ancestor.
    # Walk every component with lstat so result publication cannot be
    # redirected outside the baseline by a pre-existing alias.
    _lstat_directory_chain(parent, label=f"{label} parent", create=True)
    return path


def write_bytes_atomic(
    path: pathlib.Path,
    payload: bytes,
    *,
    label: str = "output",
    root: pathlib.Path | None = None,
) -> None:
    """Atomically publish one baseline-local owned regular file.

    Existing destinations are replaceable only when they are one-link regular
    files.  Symlinks, hardlinks, FIFOs and other special files fail closed;
    the temporary file is created in the authenticated destination directory.
    """
    path = _check_atomic_parent(path, label, root)
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


def write_text_atomic(
    path: pathlib.Path,
    text: str,
    *,
    label: str = "text output",
    root: pathlib.Path | None = None,
) -> None:
    """Atomically publish UTF-8 text through the same fail-closed path."""
    write_bytes_atomic(path, str(text).encode("utf-8"), label=label, root=root)


def read_json(path: pathlib.Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def ensure_dir(
    path: pathlib.Path,
    *,
    root: pathlib.Path | None = None,
    label: str = "directory",
) -> pathlib.Path:
    """Create/authenticate a directory without following pre-existing aliases."""
    if root is not None:
        target = assert_contained_path(root, path, label, allow_missing=True)
    else:
        target = lexical_path(path)
    return _lstat_directory_chain(target, label=label, create=True)


def require_dir(path: pathlib.Path | str, *, label: str = "directory") -> pathlib.Path:
    """Authenticate an existing directory chain without creating anything."""
    return _lstat_directory_chain(path, label=label, create=False)


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
