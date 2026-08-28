#!/usr/bin/env python3
"""Shared fail-closed contracts for the formal zkDL execution chain.

This module is production runner infrastructure, not a test adapter.  It keeps
all paths, resources, manifests, and result artifacts inside this baseline and
uses a single exact cell layout for both the runner and the independent
collector/validator.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable


BASELINE_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv")
DEVICES = ("cpu", "gpu")
CONFIGS = {
    "cora": "cora_full.cfg",
    "citeseer": "citeseer_full.cfg",
    "pubmed": "pubmed_full.cfg",
    "ppi": "ppi_full.cfg",
    "ogbn-arxiv": "ogbn_arxiv_full.cfg",
}
PROTOCOL_PATH = BASELINE_ROOT / "GAT-ZKML-zkDL.md"
RESOURCE_MANIFEST_PATH = BASELINE_ROOT / "resources" / "resource_manifest.json"
STAGED_RESOURCE_MANIFEST_PATH = BASELINE_ROOT / "resources" / "staged_resource_manifest.json"
RESOURCE_POLICY_SCHEMA = "zkDL_gat_resource_policy_v1"
STAGED_RESOURCE_MANIFEST_SCHEMA = "zkDL_gat_staged_resource_manifest_v1"
# Kept as a public alias for callers that only need to identify the source
# policy.  Production code must call load_resource_manifest(require_staged=True)
# and therefore never consumes this file as an authenticated resource set.
RESOURCE_MANIFEST_SCHEMA = RESOURCE_POLICY_SCHEMA
PP_INFO_SCHEMA = "zkDL_gat_pp_info_v1"
PP_CANONICAL_RELATIVE = "artifacts/setup/pp.bin"
PP_DEFAULT_MAX_VECTOR_LEN = 1 << 28
PP_SETUP_ID = "zkDL-GAT-transparent-PP-v1"
PP_SECURITY_ID = "zkdl_gat_bls12_381_multi_session_v2"
PP_PROTOCOL = "zkDL-hiding-Hyrax-IPA-BLS12-381-v1"
PP_SECURITY_BITS = 128
PP_FIELDS = (
    "pp_path",
    "pp_sha256",
    "pp_size_bytes",
    "pp_schema",
    "pp_id",
    "pp_max_vector_len",
    "pp_setup_id",
    "pp_security_id",
    "pp_protocol",
    "pp_security_bits",
)
DATASET_CACHE_SCHEMA = "zkDL_gat_dataset_cache_v1"
CHECKPOINT_BUNDLE_SCHEMA = "zkDL_gat_checkpoint_bundle_v1"
BUILD_MANIFEST_SCHEMA = "zkDL_gat_build_manifest_v1"
CELL_SCHEMA = "zkDL_gat_cell_success_manifest_v1"
CELL_FILES = frozenset(
    {
        "formal_artifacts/preflight.json",
        "formal_artifacts/canonical_relation.json",
        "formal_artifacts/statement.bin",
        "formal_artifacts/proof.bin",
        "proof_manifest.json",
        "verifier_manifest.json",
        "result.json",
        "success_manifest.json",
    }
)
CELL_DIRS = frozenset({"formal_artifacts"})

# Private witness/model-opening bytes are process-local inputs.  These names
# are deliberately not part of the public cell schema; callers can use this
# small guard when checking JSON manifests for accidental publication.
PRIVATE_PUBLICATION_KEYS = frozenset(
    {
        "witness",
        "witness_path",
        "witness_sha256",
        "model_secret",
        "model_secret_path",
        "model_secret_sha256",
        "secret",
        "secret_path",
        "secret_sha256",
    }
)


def private_publication_fields(value: Any, *, prefix: str = "") -> list[str]:
    """Return public-manifest fields that would expose private inputs.

    The verifier-boundary boolean is intentionally allowed: it attests that
    the independent verifier did *not* read a witness and carries no secret
    bytes or path.  Raw witness/model-secret paths and digests are rejected.
    """
    findings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            location = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in PRIVATE_PUBLICATION_KEYS:
                findings.append(location)
            if key_text != "verifier_reads_witness":
                lowered = key_text.lower()
                if "witness.private.bin" in lowered or "model_opening.private.bin" in lowered:
                    findings.append(location)
            findings.extend(private_publication_fields(item, prefix=location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(private_publication_fields(item, prefix=f"{prefix}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if "witness.private.bin" in lowered or "model_opening.private.bin" in lowered:
            findings.append(prefix or "value")
    return findings


def _relative(path: Path, root: Path = BASELINE_ROOT) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def ensure_contained(path: Path, *, root: Path = BASELINE_ROOT, label: str = "path") -> Path:
    """Resolve *path* and reject absolute escapes and symlink components."""
    root_resolved = root.resolve()
    raw_candidate = path if path.is_absolute() else root_resolved / path
    # Check the lexical path before resolving it.  A symlink whose target is
    # still inside the baseline is forbidden as well: artifacts must have one
    # unambiguous owner and may not alias another cell/resource.
    raw_candidate = Path(os.path.abspath(str(raw_candidate)))
    try:
        raw_parts = raw_candidate.relative_to(root_resolved).parts
    except ValueError as error:
        raise ValueError(f"{label} escapes baseline root: {path}") from error
    current = root_resolved
    for part in raw_parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")
    candidate = raw_candidate.resolve(strict=False)
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"{label} must stay inside {root_resolved}: {path}")
    return candidate


def ensure_regular_file(path: Path, *, label: str, allow_missing: bool = False) -> Path:
    path = ensure_contained(path, label=label)
    if not path.exists():
        if allow_missing:
            return path
        raise FileNotFoundError(f"missing {label}: {path}")
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if st.st_nlink != 1:
        raise ValueError(f"{label} must not be hard-linked: {path}")
    return path


def ensure_directory(path: Path, *, label: str, allow_missing: bool = False) -> Path:
    path = ensure_contained(path, label=label)
    if not path.exists():
        if allow_missing:
            return path
        raise FileNotFoundError(f"missing {label}: {path}")
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory: {path}")
    return path


def sha256_file(path: Path) -> str:
    path = ensure_regular_file(path, label="hashed file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path = ensure_contained(path, label="output")
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_directory(path.parent, label="output parent")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any, *, mode: int = 0o644) -> None:
    atomic_write_bytes(path, canonical_json(value), mode=mode)


def read_json(path: Path, *, label: str = "JSON") -> dict[str, Any]:
    ensure_regular_file(path, label=label)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def config_path(dataset: str) -> Path:
    if dataset not in CONFIGS:
        raise ValueError(f"unknown dataset: {dataset}")
    return ensure_regular_file(
        BASELINE_ROOT / "configs" / CONFIGS[dataset], label=f"{dataset} config"
    )


def parse_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _manifest_digest(manifest: dict[str, Any]) -> str:
    material = dict(manifest)
    material.pop("manifest_id", None)
    return f"sha256:{sha256_bytes(canonical_json(material))}"


def _sha256_id(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("sha256 value must be a 64-character hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("sha256 value is not hexadecimal") from error
    return f"sha256:{value}"


def _require_sha256_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError(f"{label} must be a tagged sha256 digest")
    _sha256_id(value[7:])
    return value


def _validate_pp_policy(pp: Any) -> dict[str, Any]:
    """Validate the immutable PP policy without accepting an identity claim."""
    if not isinstance(pp, dict):
        raise ValueError("resource policy PP entry must be an object")
    expected = {
        "path": PP_CANONICAL_RELATIVE,
        "schema": PP_INFO_SCHEMA,
        "max_vector_len": PP_DEFAULT_MAX_VECTOR_LEN,
        "setup_id": PP_SETUP_ID,
        "security_id": PP_SECURITY_ID,
        "protocol": PP_PROTOCOL,
        "security_bits": PP_SECURITY_BITS,
    }
    if set(pp) - set(expected):
        raise ValueError("resource policy PP contains an identity or unknown field")
    for key, value in expected.items():
        if pp.get(key) != value:
            raise ValueError(f"resource policy PP {key} mismatch")
    maximum = pp["max_vector_len"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0 or maximum & (maximum - 1):
        raise ValueError("resource policy PP max_vector_len must be a power of two")
    return pp


def _validate_pp_spec(pp: Any) -> dict[str, Any]:
    """Validate an actual PP identity from the staged manifest."""
    if not isinstance(pp, dict):
        raise ValueError("staged resource manifest PP entry must be an object")
    policy = _validate_pp_policy({key: pp.get(key) for key in (
        "path", "schema", "max_vector_len", "setup_id", "security_id",
        "protocol", "security_bits",
    )})
    if not isinstance(pp.get("pp_id"), str) or len(pp["pp_id"]) != 64:
        raise ValueError("staged PP pp_id is malformed")
    try:
        int(pp["pp_id"], 16)
    except ValueError as error:
        raise ValueError("staged PP pp_id is not hexadecimal") from error
    if set(pp["pp_id"]) == {"0"}:
        raise ValueError("staged PP pp_id may not be a placeholder")
    _require_sha256_id(pp.get("sha256"), label="staged PP sha256")
    size = pp.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("staged PP size_bytes must be positive")
    return {**policy, "pp_id": pp["pp_id"], "sha256": pp["sha256"], "size_bytes": size}


def pp_path_from_manifest(manifest: dict[str, Any]) -> Path:
    pp = _validate_pp_spec(manifest.get("pp"))
    path = ensure_contained(BASELINE_ROOT / pp["path"], label="canonical PP path")
    if path.relative_to(BASELINE_ROOT).as_posix() != PP_CANONICAL_RELATIVE:
        raise ValueError("resource manifest PP path is not canonical")
    return path


def pp_binding_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    pp = _validate_pp_spec(manifest.get("pp"))
    return {
        "pp_path": pp["path"],
        "pp_sha256": pp["sha256"],
        "pp_size_bytes": pp["size_bytes"],
        "pp_schema": pp["schema"],
        "pp_id": pp["pp_id"],
        "pp_max_vector_len": pp["max_vector_len"],
        "pp_setup_id": pp["setup_id"],
        "pp_security_id": pp["security_id"],
        "pp_protocol": pp["protocol"],
        "pp_security_bits": pp["security_bits"],
    }


def validate_pp_file(path: Path, manifest: dict[str, Any], pp_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Authenticate the one baseline-local PP and its native pp-info record."""
    expected = pp_binding_from_manifest(manifest)
    canonical = pp_path_from_manifest(manifest)
    actual_path = ensure_contained(path, label="PP path")
    if actual_path != canonical:
        raise ValueError(f"PP path is not the manifest canonical path: {actual_path}")
    actual_path = ensure_regular_file(actual_path, label="PP")
    actual_sha = _sha256_id(sha256_file(actual_path))
    actual_size = actual_path.stat().st_size
    if actual_sha != expected["pp_sha256"]:
        raise ValueError("PP sha256 does not match the authenticated resource manifest")
    if actual_size != expected["pp_size_bytes"]:
        raise ValueError("PP size does not match the authenticated resource manifest")
    if pp_info is not None:
        # The native pp-info JSON has a canonical `pp_*` projection plus
        # compatibility aliases used by older setup tooling.  Authenticate
        # every field the binary emits, while keeping the parser forward
        # compatible with a binary that omits an optional alias.
        required_native = {
            "schema": "pp_schema",
            "pp_id": "pp_id",
            "max_vector_len": "pp_max_vector_len",
        }
        optional_native = {
            "pp_path": "pp_path",
            "pp_sha256": "pp_sha256",
            "pp_size_bytes": "pp_size_bytes",
            "pp_schema": "pp_schema",
            "pp_max_vector_len": "pp_max_vector_len",
            "pp_setup_id": "pp_setup_id",
            "pp_security_id": "pp_security_id",
            "pp_protocol": "pp_protocol",
            "pp_security_bits": "pp_security_bits",
            "setup_id": "pp_setup_id",
            "security_id": "pp_security_id",
            "protocol": "pp_protocol",
            "security_bits": "pp_security_bits",
            "sha256": "pp_sha256",
            "size_bytes": "pp_size_bytes",
            "digest": "pp_digest",
            "pp_digest": "pp_digest",
        }
        expected_aliases = {
            **expected,
            "pp_sha256": expected["pp_sha256"],
            "pp_size_bytes": expected["pp_size_bytes"],
            "pp_digest": f"sha256:{expected['pp_id']}",
        }
        for native_key, binding_key in required_native.items():
            if pp_info.get(native_key) != expected_aliases[binding_key]:
                raise ValueError(f"native PP info {native_key} mismatch")
        for native_key, binding_key in optional_native.items():
            if native_key in pp_info and pp_info.get(native_key) != expected_aliases[binding_key]:
                raise ValueError(f"native PP info {native_key} mismatch")
    return expected


def _production_source_files() -> list[Path]:
    """List source inputs that determine a formal binary/execution identity."""
    roots = [BASELINE_ROOT / "src", BASELINE_ROOT / "scripts", BASELINE_ROOT / "configs"]
    explicit = [BASELINE_ROOT / "Cargo.toml", BASELINE_ROOT / "Cargo.lock", BASELINE_ROOT / "build.rs"]
    paths = [path for path in explicit if path.is_file()]
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix not in {".rs", ".cu", ".cuh", ".py", ".sh", ".cfg", ".json", ".toml"}:
                continue
            if path.parts and "tests" in path.parts:
                continue
            paths.append(path)
    return sorted(set(paths))


def source_identity() -> str:
    records: list[dict[str, Any]] = []
    for path in _production_source_files():
        ensure_regular_file(path, label="production source")
        records.append({"path": _relative(path), "size": path.stat().st_size, "sha256": sha256_file(path)})
    return f"sha256:{sha256_bytes(canonical_json(records))}"


def build_manifest_path(device: str) -> Path:
    if device not in DEVICES:
        raise ValueError(f"invalid build device: {device}")
    return ensure_contained(BASELINE_ROOT / "bin" / f"zkdl-formal-{device}.build.json", label="build manifest")


def write_build_manifest(device: str, binary: Path) -> dict[str, Any]:
    binary = ensure_regular_file(binary, label=f"{device} formal binary")
    if binary != ensure_contained(BASELINE_ROOT / "bin" / f"zkdl-formal-{device}", label="formal binary"):
        raise ValueError("build binary path is not canonical")
    payload: dict[str, Any] = {
        "schema": BUILD_MANIFEST_SCHEMA,
        "baseline": "zkdl",
        "device": device,
        "binary_path": binary.relative_to(BASELINE_ROOT).as_posix(),
        "binary_sha256": _sha256_id(sha256_file(binary)),
        "binary_size_bytes": binary.stat().st_size,
        "source_identity": source_identity(),
        "cargo_lock_sha256": _sha256_id(sha256_file(BASELINE_ROOT / "Cargo.lock")),
        "profile": "release",
        "feature": "gpu" if device == "gpu" else "cpu",
    }
    material = canonical_json(payload)
    payload["build_identity"] = f"sha256:{sha256_bytes(material)}"
    write_path = build_manifest_path(device)
    atomic_write_json(write_path, payload)
    return payload


def load_build_manifest(device: str, binary: Path) -> dict[str, Any]:
    binary = ensure_regular_file(binary, label=f"{device} formal binary")
    manifest = read_json(build_manifest_path(device), label=f"{device} build manifest")
    if manifest.get("schema") != BUILD_MANIFEST_SCHEMA or manifest.get("baseline") != "zkdl":
        raise ValueError(f"{device} build manifest schema mismatch")
    if manifest.get("device") != device or manifest.get("profile") != "release":
        raise ValueError(f"{device} build manifest identity mismatch")
    expected_binary = ensure_contained(BASELINE_ROOT / str(manifest.get("binary_path", "")), label="build binary path")
    if expected_binary != binary:
        raise ValueError(f"{device} build manifest binary path mismatch")
    if manifest.get("binary_sha256") != _sha256_id(sha256_file(binary)):
        raise ValueError(f"{device} binary digest differs from build manifest")
    if manifest.get("binary_size_bytes") != binary.stat().st_size:
        raise ValueError(f"{device} binary size differs from build manifest")
    if manifest.get("source_identity") != source_identity():
        raise ValueError(f"{device} build is stale for current source")
    if manifest.get("cargo_lock_sha256") != _sha256_id(sha256_file(BASELINE_ROOT / "Cargo.lock")):
        raise ValueError(f"{device} build lockfile identity mismatch")
    material = dict(manifest)
    build_id = material.pop("build_identity", None)
    if build_id != f"sha256:{sha256_bytes(canonical_json(material))}":
        raise ValueError(f"{device} build manifest digest mismatch")
    return manifest


def _validate_dataset_policy_entry(dataset: str, entry: Any) -> dict[str, str]:
    if not isinstance(entry, dict):
        raise ValueError(f"resource policy entry is not an object: {dataset}")
    expected = {
        "cache": f"resources/datasets/{'ogbn_arxiv' if dataset == 'ogbn-arxiv' else dataset}",
        "checkpoint_bundle": f"resources/models/{dataset.replace('-', '_')}_gat",
    }
    for key, value in expected.items():
        raw = entry.get(key)
        if raw != value or Path(str(raw)).is_absolute():
            raise ValueError(f"resource policy {dataset} {key} path is not canonical")
        ensure_contained(BASELINE_ROOT / value, label=f"{dataset} {key}")
    return expected


def load_resource_policy() -> tuple[dict[str, Any], str]:
    """Load only the immutable source policy; it carries no PP/resource identity."""
    policy = read_json(RESOURCE_MANIFEST_PATH, label="resource policy")
    if policy.get("schema") != RESOURCE_POLICY_SCHEMA or policy.get("baseline") != "zkdl":
        raise ValueError("unsupported zkDL resource policy schema")
    allowed = {"schema", "baseline", "datasets", "pp", "static_tables"}
    if set(policy) != allowed:
        raise ValueError("source resource policy contains staged identity or unknown fields")
    _validate_pp_policy(policy.get("pp"))
    datasets = policy.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ValueError("resource policy must enumerate exactly five datasets")
    for dataset in DATASETS:
        _validate_dataset_policy_entry(dataset, datasets[dataset])
    static_tables = policy.get("static_tables")
    if static_tables != {"scope": "baseline-local formal relation"}:
        raise ValueError("resource policy static_tables scope mismatch")
    policy_id = _manifest_digest(policy)
    return policy, policy_id


def _resource_bundle_records(dataset: str, root: Path) -> list[dict[str, Any]]:
    root = ensure_directory(root, label=f"{dataset} resource bundle")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            if path.is_symlink():
                raise ValueError(f"resource directory is a symlink: {path}")
            continue
        ensure_regular_file(path, label=f"{dataset} resource")
        records.append({
            "path": _relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_id(sha256_file(path)),
        })
    if not records:
        raise ValueError(f"{dataset} resource bundle is empty")
    return records


def _bundle_identity(dataset: str, root: Path, schema: str) -> dict[str, Any]:
    records = _resource_bundle_records(dataset, root)
    return {
        "schema": schema,
        "size_bytes": sum(int(record["size_bytes"]) for record in records),
        "sha256": f"sha256:{sha256_bytes(canonical_json(records))}",
        "files": records,
    }


def build_staged_resource_manifest(
    *, require_all: bool = True, pp_info: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Authenticate staged bundles and the native PP into one atomic manifest."""
    policy, policy_id = load_resource_policy()
    staged: dict[str, Any] = {
        "schema": STAGED_RESOURCE_MANIFEST_SCHEMA,
        "baseline": "zkdl",
        "policy_id": policy_id,
        "datasets": {},
        "pp": {},
    }
    pp_policy = policy["pp"]
    pp_path = ensure_regular_file(
        BASELINE_ROOT / pp_policy["path"], label="canonical PP", allow_missing=not require_all
    )
    if pp_path.exists():
        pp_sha = _sha256_id(sha256_file(pp_path))
        pp_size = pp_path.stat().st_size
        if pp_size <= 0:
            raise ValueError("canonical PP must not be empty")
        pp_id = str((pp_info or {}).get("pp_id", ""))
    else:
        pp_sha = ""
        pp_size = 0
        pp_id = ""
    staged["pp"] = {
        **pp_policy,
        "pp_id": pp_id,
        "sha256": pp_sha,
        "size_bytes": pp_size,
    }
    for dataset in DATASETS:
        entry_policy = _validate_dataset_policy_entry(dataset, policy["datasets"][dataset])
        cache = BASELINE_ROOT / entry_policy["cache"]
        checkpoint = BASELINE_ROOT / entry_policy["checkpoint_bundle"]
        cache_exists = cache.exists()
        checkpoint_exists = checkpoint.exists()
        if require_all and (not cache_exists or not checkpoint_exists):
            missing = entry_policy["cache"] if not cache_exists else entry_policy["checkpoint_bundle"]
            raise FileNotFoundError(f"missing staged {dataset} resource bundle: {missing}")
        if not cache_exists or not checkpoint_exists:
            continue
        validate_staged_dataset_resources(dataset, {"datasets": {dataset: entry_policy}})
        cache_id = _bundle_identity(dataset, cache, DATASET_CACHE_SCHEMA)
        checkpoint_id = _bundle_identity(dataset, checkpoint, CHECKPOINT_BUNDLE_SCHEMA)
        staged["datasets"][dataset] = {
            **entry_policy,
            "cache_schema": cache_id["schema"],
            "cache_sha256": cache_id["sha256"],
            "cache_size_bytes": cache_id["size_bytes"],
            "cache_files": cache_id["files"],
            "checkpoint_schema": checkpoint_id["schema"],
            "checkpoint_sha256": checkpoint_id["sha256"],
            "checkpoint_size_bytes": checkpoint_id["size_bytes"],
            "checkpoint_files": checkpoint_id["files"],
        }
    if require_all and set(staged["datasets"]) != set(DATASETS):
        raise FileNotFoundError("all five dataset resource bundles must be staged before setup")
    if not staged["pp"]["pp_id"]:
        raise ValueError("native PP identity is required; invoke setup_resources with formal binary")
    if pp_info is not None:
        if pp_info.get("schema") != PP_INFO_SCHEMA:
            raise ValueError("native PP info schema mismatch")
        if pp_info.get("pp_id") != staged["pp"]["pp_id"]:
            raise ValueError("native PP info id mismatch")
        if int(pp_info.get("max_vector_len", -1)) != int(staged["pp"]["max_vector_len"]):
            raise ValueError("native PP info max_vector_len mismatch")
    staged["manifest_id"] = _manifest_digest(staged)
    return staged


def load_resource_manifest(*, require_staged: bool = True) -> tuple[dict[str, Any], str]:
    """Load the authenticated staged manifest; production defaults fail closed."""
    if not require_staged:
        return load_resource_policy()
    policy, policy_id = load_resource_policy()
    manifest = read_json(STAGED_RESOURCE_MANIFEST_PATH, label="staged resource manifest")
    if manifest.get("schema") != STAGED_RESOURCE_MANIFEST_SCHEMA or manifest.get("baseline") != "zkdl":
        raise ValueError("unsupported zkDL staged resource manifest schema")
    if manifest.get("policy_id") != policy_id:
        raise ValueError("staged resource manifest is for a different source policy")
    if manifest.get("manifest_id") != _manifest_digest(manifest):
        raise ValueError("staged resource manifest digest mismatch")
    pp = manifest.get("pp")
    _validate_pp_spec(pp)
    pp_path = pp_path_from_manifest(manifest)
    validate_pp_file(pp_path, manifest)
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ValueError("staged resource manifest must enumerate exactly five datasets")
    for dataset in DATASETS:
        entry = datasets[dataset]
        expected_paths = _validate_dataset_policy_entry(dataset, entry)
        for key, value in expected_paths.items():
            if entry.get(key) != value:
                raise ValueError(f"staged {dataset} {key} does not match source policy")
        validate_staged_dataset_resources(dataset, manifest)
        _validate_bundle_identity(dataset, entry, "cache", DATASET_CACHE_SCHEMA)
        _validate_bundle_identity(dataset, entry, "checkpoint", CHECKPOINT_BUNDLE_SCHEMA)
    return manifest, str(manifest["manifest_id"])


def validate_config_resources(dataset: str, config: dict[str, str], manifest: dict[str, Any]) -> None:
    entry = manifest["datasets"][dataset]
    if config.get("dataset") != dataset:
        raise ValueError(f"config dataset mismatch: {dataset}")
    for key in ("cache", "checkpoint_bundle"):
        if config.get(key) != entry[key]:
            raise ValueError(f"config/resource manifest mismatch for {dataset}: {key}")
        ensure_contained(BASELINE_ROOT / config[key], label=f"{dataset} {key}")


def validate_staged_dataset_resources(dataset: str, manifest: dict[str, Any]) -> None:
    """Require only the selected dataset's local resource bundle."""
    entry = manifest["datasets"][dataset]
    cache = ensure_directory(BASELINE_ROOT / entry["cache"], label=f"{dataset} cache")
    checkpoint = ensure_directory(
        BASELINE_ROOT / entry["checkpoint_bundle"], label=f"{dataset} checkpoint"
    )
    ensure_regular_file(cache / "meta.cfg", label=f"{dataset} cache metadata")
    ensure_regular_file(cache / "edges.txt", label=f"{dataset} edges")
    feature_files = [cache / "features.txt", cache / "features.npy"]
    for feature_file in feature_files:
        if feature_file.exists() or feature_file.is_symlink():
            ensure_regular_file(feature_file, label=f"{dataset} features")
    if not any(feature_file.is_file() for feature_file in feature_files):
        raise FileNotFoundError(f"{dataset} cache has no feature resource")
    ensure_regular_file(checkpoint / "manifest.json", label=f"{dataset} model manifest")
    ensure_regular_file(checkpoint / "tensors.txt", label=f"{dataset} model tensors")


def _validate_bundle_identity(
    dataset: str, entry: dict[str, Any], prefix: str, schema: str
) -> None:
    root_key = "cache" if prefix == "cache" else "checkpoint_bundle"
    records_key = f"{prefix}_files"
    root = ensure_directory(BASELINE_ROOT / entry[root_key], label=f"{dataset} {prefix}")
    actual = _resource_bundle_records(dataset, root)
    if entry.get(f"{prefix}_schema") != schema:
        raise ValueError(f"{dataset} {prefix} schema mismatch")
    if entry.get(records_key) != actual:
        raise ValueError(f"{dataset} {prefix} file inventory mismatch")
    if entry.get(f"{prefix}_size_bytes") != sum(int(item["size_bytes"]) for item in actual):
        raise ValueError(f"{dataset} {prefix} size mismatch")
    if entry.get(f"{prefix}_sha256") != f"sha256:{sha256_bytes(canonical_json(actual))}":
        raise ValueError(f"{dataset} {prefix} digest mismatch")


def resource_inventory_digest(dataset: str, manifest: dict[str, Any]) -> str:
    """Digest staged resource identities without exposing private model data."""
    entry = manifest["datasets"][dataset]
    _validate_bundle_identity(dataset, entry, "cache", DATASET_CACHE_SCHEMA)
    _validate_bundle_identity(dataset, entry, "checkpoint", CHECKPOINT_BUNDLE_SCHEMA)
    records = list(entry["cache_files"]) + list(entry["checkpoint_files"])
    return f"sha256:{sha256_bytes(canonical_json(records))}"


def cell_path(output_root: Path, dataset: str, device: str) -> Path:
    if dataset not in DATASETS or device not in DEVICES:
        raise ValueError(f"invalid cell: {dataset}/{device}")
    output_root = ensure_contained(output_root, label="runs root")
    if output_root == BASELINE_ROOT:
        raise ValueError("runs root must be a baseline-local child directory")
    if output_root.exists() and output_root.is_symlink():
        raise ValueError("runs root may not be a symlink")
    return ensure_contained(output_root / dataset / device / "warm", label="run cell")


def prepare_cell(run: Path, *, replace: bool = False) -> None:
    run = ensure_contained(run, label="run cell")
    if run.exists():
        if run.is_symlink() or not run.is_dir():
            raise ValueError(f"run cell is not a real directory: {run}")
        if any(run.iterdir()):
            if not replace:
                raise FileExistsError(
                    f"run cell already contains artifacts; pass --replace explicitly: {run}"
                )
            # The target is a single validated dataset/device cell, never a
            # workspace or recursive baseline root.
            for child in run.iterdir():
                if child.is_symlink() or child.is_dir():
                    if child.is_symlink():
                        child.unlink()
                    else:
                        import shutil

                        shutil.rmtree(child)
                else:
                    child.unlink()
    else:
        run.mkdir(parents=True, exist_ok=False)
    ensure_directory(run, label="run cell")


def artifact_inventory(run: Path, *, require_complete: bool = False) -> dict[str, dict[str, int | str]]:
    """Return a strict cell inventory and reject symlink/hardlink/special files."""
    run = ensure_directory(run, label="run cell")
    files: dict[str, dict[str, int | str]] = {}
    directories: set[str] = set()
    for path in sorted(run.rglob("*")):
        rel = path.relative_to(run).as_posix()
        if path.is_symlink():
            raise ValueError(f"run artifact symlink is forbidden: {rel}")
        if path.is_dir():
            directories.add(rel)
            continue
        ensure_regular_file(path, label=f"run artifact {rel}")
        if rel in files:
            raise ValueError(f"duplicate run artifact: {rel}")
        files[rel] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    if not directories.issubset(CELL_DIRS):
        extra = sorted(directories - CELL_DIRS)
        raise ValueError(f"unexpected run artifact directories: {extra}")
    if require_complete:
        missing = sorted(CELL_FILES - files.keys())
        extra = sorted(files.keys() - CELL_FILES)
        if missing or extra:
            raise ValueError(f"cell artifact inventory mismatch missing={missing} extra={extra}")
    return files


def relative_manifest_path(raw: Any, *, run: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ValueError(f"{label} must be a relative path")
    path = ensure_contained(BASELINE_ROOT / raw, label=label)
    try:
        path.relative_to(run)
    except ValueError as error:
        raise ValueError(f"{label} escapes run cell: {raw}") from error
    return path


def protocol_digest() -> str:
    return f"sha256:{sha256_file(PROTOCOL_PATH)}"


def config_digest(path: Path) -> str:
    return f"sha256:{sha256_file(path)}"
