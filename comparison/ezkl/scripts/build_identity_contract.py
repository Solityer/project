"""Independent validation of the formal ezkl build identity contract.

The runner, result collector, and final validator all call this module.  It
deliberately recomputes the inventory and digests from the current checkout;
the JSON identity file is evidence, not an authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from formal_source_manifest import (
    checked_path,
    formal_source_files,
    formal_vendor_files,
    formal_source_id,
    lexical_baseline_root,
    SOURCE_FILTER_CONTRACT,
    tree_digest,
)
from resource_policy import validate_cuda_backend_material


class BuildIdentityError(ValueError):
    """Raised when a build identity is incomplete or does not match source."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _vendor_tree_digest(root: Path) -> str:
    try:
        files = formal_vendor_files(root)
    except (FileNotFoundError, ValueError) as exc:
        raise BuildIdentityError(str(exc)) from exc
    if not files:
        raise BuildIdentityError("authenticated vendor tree is empty")
    return tree_digest(root, sorted(files, key=lambda item: item.relative_to(root).as_posix()))


def _cuda_backend_provenance(root: Path, backend: str) -> dict[str, Any]:
    if backend == "cuda":
        return validate_cuda_backend_material(root)
    return {"status": "not_applicable"}


def validate_build_identity(
    root: Path,
    identity_path: Path,
    *,
    backend: str,
    binary: Path,
    adapter: Path,
) -> dict[str, Any]:
    """Validate and return an identity payload, failing closed on any mismatch."""

    try:
        root = lexical_baseline_root(root)
        identity_path = checked_path(root, identity_path, "build identity")
        expected_binary = checked_path(root, binary, "formal binary")
        expected_adapter = checked_path(root, adapter, "formal adapter")
    except (FileNotFoundError, ValueError) as exc:
        raise BuildIdentityError(str(exc)) from exc
    if not identity_path.is_file() or identity_path.is_symlink():
        raise BuildIdentityError(f"missing or symlinked build identity: {identity_path}")
    try:
        payload = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildIdentityError(f"cannot parse build identity: {identity_path}") from exc
    if not isinstance(payload, dict):
        raise BuildIdentityError("build identity must be a JSON object")

    expected_files = formal_source_files(root)
    expected_manifest = [path.relative_to(root).as_posix() for path in expected_files]
    expected_source_id = formal_source_id(root)
    if not expected_binary.is_file() or expected_binary.is_symlink():
        raise BuildIdentityError(f"missing formal binary: {expected_binary}")
    if not expected_adapter.is_file() or expected_adapter.is_symlink():
        raise BuildIdentityError(f"missing formal adapter: {expected_adapter}")

    required = (
        "build_identity_kind",
        "backend",
        "features",
        "source_tree_digest",
        "formal_semantic_source_id",
        "source_filter_contract",
        "formal_source_manifest",
        "vendor_tree_digest",
        "dependency_source_policy",
        "dependency_lock_digest",
        "security_profile_digest",
        "binary_sha256",
        "adapter_sha256",
        "binary_path",
        "adapter_path",
        "gpu_telemetry_contract",
        "cuda_backend_provenance",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise BuildIdentityError("build identity missing fields: " + ",".join(missing))
    expected = {
        "build_identity_kind": "ezkl_formal_build_identity_v1",
        "backend": backend,
        "features": "formal,gpu-accelerated" if backend == "cuda" else "formal",
        "source_tree_digest": expected_source_id,
        "formal_semantic_source_id": expected_source_id,
        "source_filter_contract": SOURCE_FILTER_CONTRACT,
        "formal_source_manifest": expected_manifest,
        "vendor_tree_digest": _vendor_tree_digest(root),
        "dependency_source_policy": "exact-revision-or-authenticated-vendor-path-v1",
        "dependency_lock_digest": sha256(root / "Cargo.lock"),
        "security_profile_digest": sha256(root / "configs" / "security_profile.json"),
        "binary_sha256": sha256(expected_binary),
        "adapter_sha256": sha256(expected_adapter),
        "binary_path": expected_binary.relative_to(root).as_posix(),
        "adapter_path": expected_adapter.relative_to(root).as_posix(),
        "gpu_telemetry_contract": (
            "cupti_activity_kernel_scoped_v1" if backend == "cuda" else "not_applicable"
        ),
        "cuda_backend_provenance": _cuda_backend_provenance(root, backend),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise BuildIdentityError(f"build identity mismatch for {key}")
    return payload


def identity_digest(path: Path) -> str:
    """Return the digest of an identity file for result-manifest binding."""

    lexical = Path(path).expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    # This helper is called only with a baseline-local path, but still checks
    # every lexical component before opening it so a symlink cannot redirect
    # the digest computation.
    root = lexical.anchor or "/"
    checked = checked_path(Path(root), lexical, "build identity digest")
    return sha256(checked)
