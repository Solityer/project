"""Canonical recursive source inventory for the formal ezkl relation.

This is shared by the security-profile materializer and build-provenance
writer.  The Rust build script mirrors the same ordered roots so the
compile-time semantic ID and the externally recorded build identity cannot
silently cover different code.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


# The formal build identity covers every production input that can affect the
# native GAT route: the complete Rust/C++ source trees, every production
# script/configuration, the locked dependency graph, and the authenticated
# vendor tree.  Keeping this as coarse roots prevents a newly added source or
# runner file from silently falling outside the identity contract.
FORMAL_SOURCE_ROOTS = (
    "src",
    "cpp/gat_ezkl_adapter",
    "configs",
    "scripts",
    "Cargo.lock",
    "Cargo.toml",
    "build.rs",
    "rust-toolchain",
    "vendor",
)

FORMAL_VENDOR_INVENTORY = Path("configs") / "formal_vendor_files.txt"

# This is an explicit production-source filtering contract shared with
# ``build.rs``.  Source roots are intentionally narrow: an entry with an
# unknown name/suffix is an error, while only the listed generated artefacts
# are ignored.  This keeps normal Python test/import activity from changing
# the formal identity without turning the identity scan into a permissive
# "ignore anything unknown" walk.
SOURCE_FILTER_CONTRACT = "ezkl_formal_production_source_filter_v1"
GENERATED_SOURCE_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "target",
        "build",
        "build_gat_adapter",
        "tmp",
        "logs",
        "runs_tmp",
    }
)
GENERATED_SOURCE_SUFFIXES = frozenset({".pyc", ".pyo", ".log", ".tmp"})
GENERATED_SOURCE_TEMP_PREFIX = "."
GENERATED_SOURCE_TEMP_MARKER = ".tmp-"
PRODUCTION_SOURCE_SUFFIXES = {
    "src": frozenset({".rs", ".c"}),
    "cpp/gat_ezkl_adapter": frozenset({".cpp", ".hpp"}),
    "configs": frozenset({".cfg", ".json"}),
    "scripts": frozenset({".py", ".sh"}),
}
PRODUCTION_SOURCE_NAMES = {
    "cpp/gat_ezkl_adapter": frozenset({"CMakeLists.txt"}),
    "configs": frozenset({"formal_vendor_files.txt"}),
}


def _is_generated_source_name(name: str) -> bool:
    return (
        name in GENERATED_SOURCE_DIR_NAMES
        or name.endswith(tuple(GENERATED_SOURCE_SUFFIXES))
        or (
            name.startswith(GENERATED_SOURCE_TEMP_PREFIX)
            and GENERATED_SOURCE_TEMP_MARKER in name
        )
    )


def _production_source_allowed(source_root: str, path: Path) -> bool:
    name = path.name
    if name in PRODUCTION_SOURCE_NAMES.get(source_root, frozenset()):
        return True
    return path.suffix in PRODUCTION_SOURCE_SUFFIXES.get(source_root, frozenset())


def _lexical_absolute(path: Path | str) -> Path:
    """Return an absolute lexical path without resolving symlinks."""

    candidate = Path(path).expanduser()
    return Path(os.path.abspath(candidate))


def _checked_lstat(path: Path, label: str, *, allow_missing: bool = False) -> os.stat_result | None:
    """Reject links, hard-linked regular files, and special entries before resolve()."""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        if allow_missing:
            return None
        raise FileNotFoundError(f"{label} is missing: {path}") from None
    except OSError as exc:
        raise ValueError(f"cannot inspect {label}: {path}: {exc}") from exc
    mode = metadata.st_mode
    if stat.S_ISLNK(mode):
        raise ValueError(f"{label} contains a symlink component: {path}")
    if stat.S_ISREG(mode):
        if metadata.st_nlink != 1:
            raise ValueError(f"{label} contains a hard-linked regular file: {path}")
        return metadata
    if not stat.S_ISDIR(mode):
        raise ValueError(f"{label} contains a special filesystem entry: {path}")
    return metadata


def _check_lexical_components(
    lexical: Path,
    label: str,
    *,
    allow_missing_final: bool = False,
    allow_missing_tail: bool = False,
) -> None:
    """lstat every component without allowing resolution to hide a link."""

    current = Path(lexical.anchor)
    parts = lexical.parts[1:] if lexical.anchor else lexical.parts
    for index, part in enumerate(parts):
        if part in ("", ".", ".."):
            raise ValueError(f"{label} contains an unsafe lexical component: {part!r}")
        current = current / part
        try:
            _checked_lstat(
                current,
                label,
                allow_missing=allow_missing_final and index == len(parts) - 1,
            )
        except FileNotFoundError:
            if allow_missing_tail:
                return
            raise


def lexical_baseline_root(root: Path | str) -> Path:
    """Validate the baseline path lexically, then return its resolved root."""

    lexical = _lexical_absolute(Path(root))
    _check_lexical_components(lexical, "baseline root")
    metadata = _checked_lstat(lexical, "baseline root")
    assert metadata is not None
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"baseline root is not a directory: {lexical}")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"baseline root is not a directory: {resolved}")
    return resolved


def lexical_checked_path(
    root: Path | str,
    value: Path | str,
    label: str,
    *,
    allow_missing_final: bool = False,
    allow_missing_tail: bool = False,
) -> Path:
    """Validate a path lexically before any symlink resolution.

    ``allow_missing_tail`` is only for a caller that will create the missing
    directory components one at a time with :func:`safe_mkdir`; every
    existing component is still inspected with ``lstat`` first.  The returned
    path is lexical and must not be opened until the caller has completed any
    required safe creation and called :func:`checked_path` again.
    """

    lexical_root = _lexical_absolute(root)
    lexical_baseline_root(root)
    raw = Path(value).expanduser()
    if any(part in ("..",) for part in raw.parts):
        raise ValueError(f"{label} contains an unsafe lexical component: '..'")
    lexical = raw if raw.is_absolute() else lexical_root / raw
    lexical = _lexical_absolute(lexical)
    try:
        lexical.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes baseline root: {lexical}") from exc
    _check_lexical_components(
        lexical,
        label,
        allow_missing_final=allow_missing_final,
        allow_missing_tail=allow_missing_tail,
    )
    return lexical


def checked_path(
    root: Path | str,
    value: Path | str,
    label: str,
    *,
    allow_missing: bool = False,
) -> Path:
    """Check every lexical component before resolving and enforce root containment."""

    root_resolved = lexical_baseline_root(root)
    lexical = lexical_checked_path(
        root,
        value,
        label,
        allow_missing_final=allow_missing,
    )
    resolved = lexical.resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{label} escapes resolved baseline root: {resolved}") from exc
    return resolved


def checked_tree_files(
    root: Path | str,
    tree: Path | str,
    label: str,
    *,
    source_root: str | None = None,
) -> list[Path]:
    """Recursively inventory one checked source tree with deterministic ordering."""

    root_resolved = lexical_baseline_root(root)
    path = checked_path(root_resolved, tree, label)
    metadata = _checked_lstat(path, label)
    assert metadata is not None
    if stat.S_ISREG(metadata.st_mode):
        return [path]
    files: list[Path] = []

    def visit(current: Path) -> None:
        try:
            entries = list(os.scandir(current))
        except OSError as exc:
            raise ValueError(f"cannot read {label}: {current}: {exc}") from exc
        for entry in entries:
            child = Path(entry.path)
            child_metadata = _checked_lstat(child, label)
            assert child_metadata is not None
            if stat.S_ISDIR(child_metadata.st_mode):
                # A vendored checkout may retain VCS metadata for provenance;
                # it is not a production dependency and must not enter the
                # authenticated source inventory.
                if child.name == ".git":
                    continue
                if source_root is not None and _is_generated_source_name(child.name):
                    continue
                visit(child)
            else:
                if source_root is not None:
                    if _is_generated_source_name(child.name):
                        continue
                    if not _production_source_allowed(source_root, child):
                        raise ValueError(
                            "unknown formal production source entry: "
                            f"{child.relative_to(root_resolved).as_posix()}"
                        )
                files.append(child.resolve(strict=True))

    visit(path)
    return files


def formal_vendor_files(root: Path) -> list[Path]:
    """Return exactly the baseline-local files listed by the vendor lockfile."""

    root_resolved = lexical_baseline_root(root)
    vendor_root = checked_path(root_resolved, "vendor", "authenticated vendor tree")
    inventory_path = checked_path(
        root_resolved, FORMAL_VENDOR_INVENTORY, "formal vendor inventory"
    )
    metadata = _checked_lstat(inventory_path, "formal vendor inventory")
    assert metadata is not None
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"formal vendor inventory is not a regular file: {inventory_path}")
    try:
        raw = inventory_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read formal vendor inventory: {inventory_path}") from exc
    entries = raw.splitlines()
    if not entries or any(not entry or entry != entry.strip() for entry in entries):
        raise ValueError("formal vendor inventory contains blank or whitespace-padded entries")
    if entries != sorted(entries) or len(entries) != len(set(entries)):
        raise ValueError("formal vendor inventory must be sorted and duplicate-free")

    listed: list[Path] = []
    for entry in entries:
        relative = Path(entry)
        if relative.is_absolute() or "\\" in entry or entry.startswith("vendor/"):
            raise ValueError(f"formal vendor inventory contains an unsafe path: {entry}")
        if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
            raise ValueError(f"formal vendor inventory contains an unsafe path: {entry}")
        path = checked_path(vendor_root, relative, "formal vendor file")
        metadata = _checked_lstat(path, "formal vendor file")
        assert metadata is not None
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"formal vendor inventory entry is not a file: {entry}")
        listed.append(path)

    actual = checked_tree_files(root_resolved, "vendor", "authenticated vendor tree")
    actual_entries = sorted(path.relative_to(vendor_root).as_posix() for path in actual)
    if actual_entries != entries:
        missing = sorted(set(entries) - set(actual_entries))
        extra = sorted(set(actual_entries) - set(entries))
        raise ValueError(
            "formal vendor inventory does not match vendor tree"
            + (f"; missing={missing}" if missing else "")
            + (f"; extra={extra}" if extra else "")
        )
    return listed


def formal_source_files(root: Path) -> list[Path]:
    root_resolved = lexical_baseline_root(root)
    files: list[Path] = []
    for relative in FORMAL_SOURCE_ROOTS:
        if relative == "vendor":
            files.extend(formal_vendor_files(root_resolved))
            continue
        try:
            files.extend(
                checked_tree_files(
                    root_resolved,
                    relative,
                    "formal source",
                    source_root=relative,
                )
            )
        except FileNotFoundError:
            raise
    unique = sorted(set(files), key=lambda path: path.relative_to(root_resolved).as_posix())
    return unique


def tree_digest(root: Path, files: list[Path] | None = None) -> str:
    root_resolved = lexical_baseline_root(root)
    inventory = files if files is not None else formal_source_files(root_resolved)
    digest = hashlib.sha256()
    for path in inventory:
        checked = Path(path)
        try:
            relative = checked.relative_to(root_resolved).as_posix().encode()
        except ValueError as exc:
            raise ValueError(f"formal source file escapes baseline root: {checked}") from exc
        metadata = _checked_lstat(checked, "formal source file")
        assert metadata is not None
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"formal source file is not regular: {checked}")
        payload = checked.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def formal_source_id(root: Path) -> str:
    return tree_digest(root)
