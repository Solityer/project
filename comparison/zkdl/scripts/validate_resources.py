#!/usr/bin/env python3
"""Validate externally supplied zkDL resources without generating replacements.

The native CUDA bridge remains the authority for decoding BLS12-381 frames and
model commitments.  This short admission tool only checks the baseline-local
paths, trusted digest bindings, and canonical JSON identities before setup or
formal execution.  Missing, stale, or self-invented resources fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any


HEX64 = re.compile(r"^[0-9a-f]{64}$")
DATASETS = {"cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv"}


def fail(message: str) -> None:
    raise SystemExit(f"validate-resources: BLOCKED: {message}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                value.update(block)
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    return value.hexdigest()


def regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} is missing or symlinked: {path}")
    return path


def local(root: Path, raw: str, label: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        fail(f"{label} must be baseline-local: {raw}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        fail(f"{label} escapes baseline root: {raw}")
    return resolved


def config_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            fail(f"malformed config line {number}")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value or key in values:
            fail(f"invalid or duplicate config key at line {number}")
        values[key] = value
    return values


def json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(regular(path, label).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid {label} JSON: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object")
    return value


def required_string(obj: dict[str, Any], key: str, label: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        fail(f"{label} missing canonical field {key}")
    return value


def required_digest(obj: dict[str, Any], key: str, label: str) -> str:
    value = required_string(obj, key, label)
    if not HEX64.fullmatch(value):
        fail(f"{label}.{key} is not lowercase SHA-256")
    return value


def canonical_digest(obj: dict[str, Any], omit: set[str]) -> str:
    payload = {key: value for key, value in obj.items() if key not in omit}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def config_identity_digest(path: Path) -> str:
    values = config_values(path)
    values.pop("verifier_registry_digest", None)
    values.pop("model_blinder_sidecar_digest", None)
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def frame_path(root: Path, frame: dict[str, Any], label: str) -> Path:
    raw = required_string(frame, "path", label)
    if Path(raw).is_absolute() or ".." in Path(raw).parts:
        fail(f"{label}.path escapes SRS directory")
    path = local(root, raw, f"{label}.path")
    regular(path, f"SRS {label} frame")
    declared = required_digest(frame, "sha256", label)
    if digest(path) != declared:
        fail(f"SRS {label} frame digest mismatch")
    count = frame.get("count")
    if not isinstance(count, int) or count < 1:
        fail(f"SRS {label}.count is invalid")
    return path


def validate_srs(root: Path, values: dict[str, str]) -> tuple[Path, dict[str, Any], str]:
    srs_raw = values.get("srs_path")
    manifest_digest = values.get("srs_manifest_digest")
    srs_id = values.get("srs_id")
    if not srs_raw or not manifest_digest or not srs_id:
        fail("secure config lacks srs_path, srs_id, or srs_manifest_digest")
    if manifest_digest == "UNAVAILABLE" or not HEX64.fullmatch(manifest_digest):
        fail("srs_manifest_digest must be an externally supplied lowercase SHA-256")
    srs = local(root, srs_raw, "srs_path")
    if srs.is_symlink() or not srs.is_dir():
        fail(f"authenticated SRS directory is absent: {srs}")
    manifest_path = regular(srs / "srs_manifest.json", "SRS manifest")
    if digest(manifest_path) != manifest_digest:
        fail("srs_manifest.json does not match srs_manifest_digest")
    manifest = json_object(manifest_path, "SRS manifest")
    if required_string(manifest, "manifest_format", "SRS manifest") != "zkgat_srs_manifest":
        fail("SRS manifest format is not zkgat_srs_manifest")
    if required_string(manifest, "curve_id", "SRS manifest") != "bls12_381":
        fail("SRS curve is not bls12_381")
    if required_string(manifest, "encoding", "SRS manifest") != "GATZKG1C":
        fail("SRS encoding is not GATZKG1C")
    if required_string(manifest, "srs_id", "SRS manifest") != srs_id:
        fail("SRS id mismatch")
    version = manifest.get("manifest_version", 1)
    if not isinstance(version, int) or version < 1:
        fail("SRS manifest_version is invalid")
    if version >= 2 or "families" in manifest:
        if version < 2 or manifest.get("chunking_rule") != "per_family_prefix_v1":
            fail("v2 SRS manifest requires per_family_prefix_v1 chunking_rule")
        families = manifest.get("families")
        if not isinstance(families, list) or not families:
            fail("v2 SRS manifest has no families")
        selected: tuple[int, str, dict[str, Any]] | None = None
        default_family = manifest.get("default_family_id")
        default_chunk = manifest.get("default_chunk_id")
        seen: set[tuple[str, int]] = set()
        generator_id = required_string(manifest, "generator_id", "SRS manifest")
        transcript = required_digest(manifest, "ceremony_transcript_sha256", "SRS manifest")
        seed = required_digest(manifest, "ceremony_seed_sha256", "SRS manifest")
        for family in families:
            if not isinstance(family, dict) or not isinstance(family.get("family_id"), str) or not family["family_id"]:
                fail("v2 SRS family has no family_id")
            family_id = family["family_id"]
            chunks = family.get("chunks")
            if not isinstance(chunks, list) or not chunks:
                fail(f"v2 SRS family has no chunks: {family_id}")
            for ordinal, chunk in enumerate(chunks):
                if not isinstance(chunk, dict):
                    fail("v2 SRS chunk is not an object")
                for key in ("chunk_id", "generator_id", "ceremony_transcript_sha256"):
                    if not isinstance(chunk.get(key), str) or not chunk[key]:
                        fail(f"v2 SRS chunk omits {key}")
                if chunk["generator_id"] != generator_id or chunk["ceremony_transcript_sha256"] != transcript:
                    fail("v2 SRS chunk generator/transcript identity mismatch")
                if chunk.get("ceremony_seed_sha256") != seed:
                    fail("v2 SRS chunk seed identity mismatch")
                index = chunk.get("chunk_index")
                domain = chunk.get("domain")
                if not isinstance(index, int) or index < 0 or not isinstance(domain, int) or domain < 1 or domain & (domain - 1):
                    fail("v2 SRS chunk index/domain is invalid")
                if (family_id, index) in seen:
                    fail("v2 SRS duplicate family/chunk_index")
                seen.add((family_id, index))
                if chunk.get("point_offset") != 0 or chunk.get("point_count") != domain or chunk.get("degree") != max(0, domain - 1) or chunk.get("point_size") != 48:
                    fail("v2 SRS chunk domain/degree/point metadata mismatch")
                frames: dict[str, Path] = {}
                for label in ("G", "H", "U"):
                    frame = chunk.get(label)
                    if not isinstance(frame, dict):
                        fail(f"v2 SRS chunk omits {label} frame")
                    frames[label] = frame_path(srs, frame, f"{family_id}/{chunk['chunk_id']}/{label}")
                    expected = domain if label == "G" else 1
                    if frame.get("count") != expected:
                        fail(f"v2 SRS {label} count does not match chunk domain")
                if frames["H"].read_bytes() == frames["U"].read_bytes():
                    fail("v2 SRS H and U frames are byte-identical")
                if required_digest(chunk, "generator_digest", "v2 SRS chunk") != chunk["G"]["sha256"]:
                    fail("v2 SRS generator_digest does not bind G frame")
                candidate = (domain, family_id, chunk)
                if default_family is not None and default_chunk is not None:
                    if family_id == default_family and chunk["chunk_id"] == default_chunk:
                        selected = candidate
                elif selected is None or candidate[:2] > selected[:2]:
                    selected = candidate
        if selected is None:
            fail("v2 SRS has no selectable chunk")
        return srs, manifest, selected[2]["generator_digest"]
    domain = manifest.get("domain")
    degree = manifest.get("degree")
    point_size = manifest.get("point_size")
    if not isinstance(domain, int) or domain < 1 or domain & (domain - 1):
        fail("SRS domain is not a positive power of two")
    if not isinstance(degree, int) or degree < (domain - 1 if domain > 1 else 0):
        fail("SRS degree is smaller than its domain")
    if point_size != 48:
        fail("SRS point_size must be 48 for GATZKG1C")
    frames = {}
    for label in ("G", "H", "U"):
        frame = manifest.get(label)
        if not isinstance(frame, dict):
            fail(f"SRS manifest omits {label} frame")
        frames[label] = frame_path(srs, frame, label)
        expected_count = domain if label == "G" else 1
        if frame["count"] != expected_count:
            fail(f"SRS {label} count does not match the authenticated domain")
    if frames["H"].read_bytes() == frames["U"].read_bytes():
        fail("SRS H and U frames are byte-identical")
    return srs, manifest, digest(frames["G"])


def validate_registry(root: Path, values: dict[str, str], srs: Path,
                      srs_manifest: dict[str, Any], generator_digest: str) -> tuple[Path, dict[str, Any], str]:
    raw_path = values.get("model_registry_path")
    expected_digest = values.get("verifier_registry_digest")
    if not raw_path or not expected_digest or expected_digest == "UNAVAILABLE" or not HEX64.fullmatch(expected_digest):
        fail("verifier_registry_digest must be an externally supplied lowercase SHA-256")
    path = local(root, raw_path, "model_registry_path")
    regular(path, "model registry")
    actual = digest(path)
    if actual != expected_digest:
        fail("model registry digest does not match verifier_registry_digest")
    registry = json_object(path, "model registry")
    if required_string(registry, "schema", "model registry") != "zkDL_gat_model_registry_v2":
        fail("model registry schema is not zkDL_gat_model_registry_v2")
    for key in ("dataset", "model_arch_id", "model_param_id", "quant_cfg_id", "static_table_id", "srs_id"):
        if registry.get(key) != values.get(key):
            fail(f"model registry identity mismatch for {key}")
    if registry.get("source_root") != "baseline-local":
        fail("model registry source_root is not baseline-local")
    srs_manifest_digest = required_digest(registry, "srs_manifest_digest", "model registry")
    if srs_manifest_digest != values["srs_manifest_digest"]:
        fail("model registry SRS manifest digest mismatch")
    if required_digest(registry, "generator_digest", "model registry") != generator_digest:
        fail("model registry generator digest mismatch")
    config_path = local(root, values["_config_path"], "config")
    if required_digest(registry, "config_digest", "model registry") != config_identity_digest(config_path):
        fail("model registry config identity mismatch")
    identity = required_digest(registry, "registry_identity_digest", "model registry")
    if identity != canonical_digest(registry, {"registry_identity_digest", "verifier_registry_digest"}):
        fail("model registry non-self identity mismatch")
    if required_digest(registry, "verifier_registry_digest", "model registry") != identity:
        fail("model registry verifier identity mismatch")
    quant_path = root / "configs" / "quantization_scale_chain.cfg"
    static_path = local(root, values["static_table_descriptor"], "static_table_descriptor")
    ledger_path = local(root, values["integer_bound_ledger"], "integer_bound_ledger")
    manifest_path = local(root, values["checkpoint_bundle"], "checkpoint_bundle") / "manifest.json"
    if required_digest(registry, "scale_chain_digest", "model registry") != digest(quant_path):
        fail("model registry scale-chain digest mismatch")
    if required_digest(registry, "static_table_digest", "model registry") != digest(static_path):
        fail("model registry static-table digest mismatch")
    if required_digest(registry, "integer_bound_ledger_digest", "model registry") != digest(ledger_path):
        fail("model registry integer-ledger digest mismatch")
    if required_digest(registry, "model_manifest_digest", "model registry") != digest(regular(manifest_path, "checkpoint manifest")):
        fail("model registry checkpoint-manifest digest mismatch")
    tensors = registry.get("model_tensors")
    if not isinstance(tensors, list) or not tensors:
        fail("model registry has no model_tensors")
    ids: set[str] = set()
    for tensor in tensors:
        if not isinstance(tensor, dict):
            fail("model registry tensor entry is not an object")
        tensor_id = required_string(tensor, "tensor_id", "model tensor")
        if tensor_id in ids:
            fail(f"duplicate model tensor id: {tensor_id}")
        ids.add(tensor_id)
        domain = tensor.get("domain_size")
        if not isinstance(domain, int) or domain < 1 or domain & (domain - 1):
            fail(f"invalid model tensor domain: {tensor_id}")
        commitment = required_string(tensor, "commitment", "model tensor")
        if not re.fullmatch(r"[0-9a-f]{96}", commitment):
            fail(f"model tensor commitment is not canonical G1 wire: {tensor_id}")
    return path, registry, actual


def validate_blinders(root: Path, values: dict[str, str], registry: dict[str, Any], registry_digest: str) -> tuple[Path, str]:
    raw_path = values.get("model_blinder_sidecar_path")
    expected_digest = values.get("model_blinder_sidecar_digest")
    if not raw_path or not expected_digest or expected_digest == "UNAVAILABLE" or not HEX64.fullmatch(expected_digest):
        fail("model_blinder_sidecar_digest must be an externally supplied lowercase SHA-256")
    path = local(root, raw_path, "model_blinder_sidecar_path")
    regular(path, "model blinder sidecar")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077 or mode & 0o600 != 0o600:
        fail("model blinder sidecar must be owner-only")
    actual = digest(path)
    if actual != expected_digest:
        fail("model blinder sidecar digest does not match config")
    sidecar = json_object(path, "model blinder sidecar")
    if required_string(sidecar, "schema", "model blinder sidecar") != "zkDL_gat_model_blinder_sidecar_v1":
        fail("model blinder sidecar schema mismatch")
    if required_digest(sidecar, "registry_digest", "model blinder sidecar") != registry_digest:
        fail("model blinder sidecar registry digest mismatch")
    tensors = registry.get("model_tensors", [])
    entries = sidecar.get("entries")
    if not isinstance(entries, list) or len(entries) != len(tensors):
        fail("model blinder sidecar entry count mismatch")
    registry_ids = {tensor["tensor_id"] for tensor in tensors}
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            fail("model blinder sidecar entry is not an object")
        tensor_id = required_string(entry, "tensor_id", "model blinder entry")
        blinder = required_string(entry, "blinder", "model blinder entry")
        if tensor_id in seen or tensor_id not in registry_ids:
            fail(f"model blinder entry is not registry-bound: {tensor_id}")
        if not re.fullmatch(r"[0-9a-f]{64}", blinder):
            fail(f"model blinder is not canonical field wire: {tensor_id}")
        seen.add(tensor_id)
    if seen != registry_ids:
        fail("model blinder sidecar does not cover every registered model tensor")
    return path, actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if config_path.parent.name != "configs":
        fail("config must live under baseline-local configs/")
    root = config_path.parent.parent.resolve()
    values = config_values(config_path)
    values["_config_path"] = config_path.relative_to(root).as_posix()
    if values.get("dataset") not in DATASETS:
        fail("dataset is outside the canonical five-dataset set")
    for key in ("static_table_descriptor", "integer_bound_ledger", "checkpoint_bundle"):
        if key not in values:
            fail(f"config key missing: {key}")
    quant = regular(root / "configs" / "quantization_scale_chain.cfg", "scale chain")
    srs, srs_manifest, generator_digest = validate_srs(root, values)
    registry_path, registry, registry_digest = validate_registry(root, values, srs, srs_manifest, generator_digest)
    sidecar_path, sidecar_digest = validate_blinders(root, values, registry, registry_digest)
    print(json.dumps({
        "schema": "zkDL_gat_resource_admission_v1",
        "dataset": values["dataset"],
        "config": config_path.relative_to(root).as_posix(),
        "config_digest": digest(config_path),
        "quantization_scale_chain_digest": digest(quant),
        "srs_manifest_digest": values["srs_manifest_digest"],
        "srs_generator_digest": generator_digest,
        "model_registry_path": registry_path.relative_to(root).as_posix(),
        "model_registry_digest": registry_digest,
        "model_blinder_sidecar_path": sidecar_path.relative_to(root).as_posix(),
        "model_blinder_sidecar_digest": sidecar_digest,
        "status": "AUTHENTICATED",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
