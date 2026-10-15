"""Fail-closed identity checks for one owned formal session directory."""
from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path

ROOT_ENV = "ZKDL_FORMAL_SESSION_ROOT"
DEVICE_ENV = "ZKDL_FORMAL_SESSION_ROOT_DEVICE"
INODE_ENV = "ZKDL_FORMAL_SESSION_ROOT_INODE"
SESSION_NAME = re.compile(r"formal_[0-9a-f]{32}\Z")


class SessionRootError(RuntimeError):
    """The active session root disappeared, changed identity, or was unbound."""


@dataclass(frozen=True)
class SessionRootIdentity:
    path: Path
    device: int
    inode: int
    root_fd: int | None = field(default=None, compare=False, repr=False)

    def child_environment(self) -> dict[str, str]:
        return {
            ROOT_ENV: str(self.path),
            DEVICE_ENV: str(self.device),
            INODE_ENV: str(self.inode),
        }


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def capture_session_root(path: Path) -> SessionRootIdentity:
    """Capture an already-created session root exactly once in its owner."""
    absolute = _absolute(path)
    if absolute.is_symlink():
        raise SessionRootError(f"formal session root is a symlink: {absolute}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise SessionRootError(f"formal session root is unavailable at owner capture: {absolute}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        path_info = absolute.lstat()
        resolved = absolute.resolve(strict=True)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(path_info.st_mode):
            raise SessionRootError(f"formal session root is not a non-symlink directory at owner capture: {absolute}")
        if resolved != absolute or (info.st_dev, info.st_ino) != (path_info.st_dev, path_info.st_ino):
            raise SessionRootError(f"formal session root path changed identity during owner capture: {absolute}")
        return SessionRootIdentity(absolute, info.st_dev, info.st_ino, descriptor)
    except BaseException:
        os.close(descriptor)
        raise


def identity_from_environment(*, required: bool = True) -> SessionRootIdentity | None:
    raw_path = os.environ.get(ROOT_ENV)
    raw_device = os.environ.get(DEVICE_ENV)
    raw_inode = os.environ.get(INODE_ENV)
    present = (raw_path is not None, raw_device is not None, raw_inode is not None)
    if not any(present) and not required:
        return None
    if not all(present):
        raise SessionRootError("active formal session root identity is incomplete")
    assert raw_path is not None and raw_device is not None and raw_inode is not None
    try:
        path = _absolute(raw_path)
        device = int(raw_device, 10)
        inode = int(raw_inode, 10)
    except (TypeError, ValueError) as exc:
        raise SessionRootError("active formal session root identity is malformed") from exc
    if not path.is_absolute() or device < 0 or inode <= 0:
        raise SessionRootError("active formal session root identity is malformed")
    return SessionRootIdentity(path, device, inode)


def require_session_root(identity: SessionRootIdentity | None = None,
                         session: Path | None = None) -> SessionRootIdentity:
    """Require the original non-symlink directory; never recapture a replacement."""
    expected = identity or identity_from_environment(required=True)
    assert expected is not None
    if session is not None and _absolute(session) != expected.path:
        raise SessionRootError(
            f"formal phase session path differs from its owner-bound root: {_absolute(session)}"
        )
    try:
        info = expected.path.lstat()
        resolved = expected.path.resolve(strict=True)
    except OSError as exc:
        raise SessionRootError(f"owner-bound formal session root disappeared: {expected.path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SessionRootError(f"owner-bound formal session root was replaced or is not a directory: {expected.path}")
    if (info.st_dev, info.st_ino) != (expected.device, expected.inode):
        raise SessionRootError(f"owner-bound formal session root identity changed: {expected.path}")
    if resolved != expected.path:
        raise SessionRootError(f"owner-bound formal session root path now traverses a symlink: {expected.path}")
    if expected.root_fd is not None:
        try:
            held = os.fstat(expected.root_fd)
        except OSError as exc:
            raise SessionRootError(f"owner-bound formal session directory descriptor is unavailable: {exc}") from exc
        if (held.st_dev, held.st_ino) != (expected.device, expected.inode) or not stat.S_ISDIR(held.st_mode):
            raise SessionRootError("owner-bound formal session directory descriptor identity changed")
    return expected


def require_bound_session(session: Path, project_root: Path) -> SessionRootIdentity:
    """Require a caller-supplied owner identity for a session path."""
    expected = identity_from_environment(required=True)
    assert expected is not None
    root = _absolute(project_root)
    session_path = _absolute(session)
    if session_path.parent != root / "runs" or SESSION_NAME.fullmatch(session_path.name) is None:
        raise SessionRootError(f"formal session is outside the canonical baseline-local runs tree: {session_path}")
    require_session_root(expected, session_path)
    return expected


def _session_ancestor(path: Path, project_root: Path) -> Path | None:
    candidate = _absolute(path)
    root = _absolute(project_root)
    current = candidate
    while current != current.parent:
        if current.parent == root / "runs" and SESSION_NAME.fullmatch(current.name):
            return current
        current = current.parent
    return None


def require_session_artifact(path: Path, project_root: Path) -> SessionRootIdentity | None:
    """Validate owner binding when a provenance artifact belongs to a session."""
    artifact = _absolute(path)
    session = _session_ancestor(artifact, project_root)
    expected = identity_from_environment(required=False)
    if expected is None:
        if session is not None:
            raise SessionRootError(
                f"formal session artifact requires the live owner's root identity: {artifact}"
            )
        return None
    require_session_root(expected)
    if session != expected.path or artifact == expected.path:
        raise SessionRootError(f"formal artifact path is outside the owner-bound session root: {artifact}")
    return expected
