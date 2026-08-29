#!/usr/bin/env python3
"""Materialize a Cargo binary as a baseline-local single-link regular file.

Cargo may hard-link a release binary from ``target/*/release/deps`` into the
top-level release directory.  The formal build identity intentionally rejects
hard-linked inputs, so this helper makes an independent copy in the same
directory and atomically replaces the top-level name before identity hashing.
The copy is deliberately narrow: all path components remain baseline-local,
the source must be a regular executable, and the destination is checked again
after replacement.  It never follows symlinks or writes outside the binary's
parent directory.
"""

from __future__ import annotations

import argparse
import os
import secrets
import stat
from pathlib import Path

from formal_source_manifest import checked_path, lexical_baseline_root


_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _destination(root: Path, value: Path | str) -> tuple[Path, str]:
    """Return a checked parent directory and final basename.

    ``checked_path`` cannot be used for the final component because it
    intentionally rejects the Cargo hard-link that is about to be copied.
    Checking the parent with the normal gate retains its full lexical,
    containment, and link policy; the final component is inspected below with
    ``lstat``/``O_NOFOLLOW``.
    """

    root = lexical_baseline_root(root)
    raw = Path(value).expanduser()
    if not raw.name or raw.name in {".", ".."}:
        raise ValueError(f"formal binary has no safe final component: {value}")
    if ".." in raw.parts:
        raise ValueError("formal binary contains an unsafe lexical component: '..'")
    parent_value = raw.parent if raw.is_absolute() else root / raw.parent
    parent = checked_path(root, parent_value, "formal binary parent")
    candidate = parent / raw.name
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"formal binary escapes baseline root: {candidate}") from exc
    return parent, raw.name


def _regular_executable(metadata: os.stat_result, label: str) -> None:
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file")
    if not metadata.st_mode & 0o111:
        raise ValueError(f"{label} is not executable")


def materialize_binary(root: Path, binary: Path | str) -> Path:
    """Copy and atomically replace ``binary`` so its final link count is one."""

    parent, name = _destination(root, binary)
    directory_fd = os.open(parent, os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC | _O_NOFOLLOW)
    source_fd: int | None = None
    temporary_fd: int | None = None
    temporary_name: str | None = None
    try:
        try:
            source_metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"formal binary is missing: {parent / name}") from exc
        _regular_executable(source_metadata, "formal binary")
        source_fd = os.open(
            name,
            os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        opened_metadata = os.fstat(source_fd)
        _regular_executable(opened_metadata, "formal binary")
        if (opened_metadata.st_dev, opened_metadata.st_ino) != (
            source_metadata.st_dev,
            source_metadata.st_ino,
        ):
            raise ValueError("formal binary changed while being opened")

        mode = stat.S_IMODE(opened_metadata.st_mode)
        temporary_name = f".{name}.materialize.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_CLOEXEC | _O_NOFOLLOW,
            mode,
            dir_fd=directory_fd,
        )
        remaining = opened_metadata.st_size
        while True:
            chunk = os.read(source_fd, 1 << 20)
            if not chunk:
                break
            offset = 0
            while offset < len(chunk):
                offset += os.write(temporary_fd, chunk[offset:])
        os.fchmod(temporary_fd, mode)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None

        temporary_metadata = os.stat(
            temporary_name, dir_fd=directory_fd, follow_symlinks=False
        )
        _regular_executable(temporary_metadata, "materialized formal binary")
        if temporary_metadata.st_nlink != 1:
            raise ValueError("materialized formal binary is hard-linked")
        if temporary_metadata.st_size != opened_metadata.st_size:
            raise ValueError(
                "materialized formal binary size mismatch: "
                f"expected {remaining}, got {temporary_metadata.st_size}"
            )

        # Reject a concurrent replacement rather than replacing an unrelated
        # path.  The destination is still allowed to have Cargo's original
        # hard-link count at this point.
        current_metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (current_metadata.st_dev, current_metadata.st_ino) != (
            opened_metadata.st_dev,
            opened_metadata.st_ino,
        ):
            raise ValueError("formal binary changed before atomic replacement")
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = None
        os.fsync(directory_fd)

        final_metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _regular_executable(final_metadata, "materialized formal binary")
        if final_metadata.st_nlink != 1:
            raise ValueError("materialized formal binary is hard-linked")
        if final_metadata.st_size != opened_metadata.st_size:
            raise ValueError("materialized formal binary size changed after replacement")
        return parent / name
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        if source_fd is not None:
            os.close(source_fd)
        os.close(directory_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = lexical_baseline_root(Path(__file__).absolute().parent.parent)
    materialize_binary(root, args.binary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
