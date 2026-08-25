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


FORMAL_SOURCE_ROOTS = (
    "src/circuit",
    "src/graph",
    "src/pfsys",
    "src/tensor",
    "src/bin/ezkl_formal.rs",
    "src/native_gat.rs",
    "src/gpu_telemetry.rs",
    "src/gpu_telemetry.c",
    "cpp/gat_ezkl_adapter",
    "configs/formal_operation_registry.json",
    "Cargo.lock",
    "Cargo.toml",
    "build.rs",
    "scripts/build.sh",
    "scripts/run_gat_ezkl_baseline.py",
    "scripts/collect_results.py",
    "scripts/validate_final_results.py",
    "scripts/formal_source_manifest.py",
    "scripts/build_identity_contract.py",
    "scripts/materialize_security_profile.py",
    "scripts/resource_policy.py",
    "scripts/write_build_identity.py",
    "configs/security_profile.json",
    "configs/full_datasets.json",
    "vendor/SOURCE_LOCK.md",
    "vendor/halo2/halo2_proofs",
    "vendor/halo2/halo2_gadgets",
    "vendor/halo2curves/halo2curves",
    "vendor/snark-verifier/snark-verifier",
    "vendor/icicle/rust/icicle-core",
    "vendor/icicle/rust/icicle-curves/icicle-bn254",
    "vendor/icicle/rust/icicle-runtime",
    "vendor/icicle/icicle/include",
    "vendor/icicle/icicle/src",
    "vendor/icicle/icicle/backend/cpu",
    "vendor/mopro-msm",
)


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
) -> None:
    """lstat every component without allowing resolution to hide a link."""

    current = Path(lexical.anchor)
    parts = lexical.parts[1:] if lexical.anchor else lexical.parts
    for index, part in enumerate(parts):
        if part in ("", ".", ".."):
            raise ValueError(f"{label} contains an unsafe lexical component: {part!r}")
        current = current / part
        _checked_lstat(
            current,
            label,
            allow_missing=allow_missing_final and index == len(parts) - 1,
        )


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


def checked_path(
    root: Path | str,
    value: Path | str,
    label: str,
    *,
    allow_missing: bool = False,
) -> Path:
    """Check every lexical component before resolving and enforce root containment."""

    root_resolved = lexical_baseline_root(root)
    lexical_root = _lexical_absolute(root)
    raw = Path(value).expanduser()
    if any(part in ("..",) for part in raw.parts):
        raise ValueError(f"{label} contains an unsafe lexical component: '..'")
    lexical = raw if raw.is_absolute() else lexical_root / raw
    lexical = _lexical_absolute(lexical)
    try:
        relative = lexical.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes baseline root: {lexical}") from exc
    _check_lexical_components(
        lexical,
        label,
        allow_missing_final=allow_missing,
    )
    resolved = lexical.resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{label} escapes resolved baseline root: {resolved}") from exc
    return resolved


def checked_tree_files(root: Path | str, tree: Path | str, label: str) -> list[Path]:
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
                visit(child)
            else:
                files.append(child.resolve(strict=True))

    visit(path)
    return files


def formal_source_files(root: Path) -> list[Path]:
    root_resolved = lexical_baseline_root(root)
    files: list[Path] = []
    for relative in FORMAL_SOURCE_ROOTS:
        try:
            files.extend(checked_tree_files(root_resolved, relative, "formal source"))
        except FileNotFoundError:
            # The optional mopro-msm source is required for a CUDA-capable
            # offline build, but a CPU-only checkout may be staged before it
            # is materialized.  The build/provenance checks reject that state
            # when the GPU feature is requested.
            if relative == "vendor/mopro-msm":
                continue
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
