#!/usr/bin/env python3
"""Validate baseline-local zkDL resources without generating replacements.

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
NATIVE_BASIS_DOMAIN = 1 << 18
MAX_NATIVE_DOMAIN = 1 << 30
SRS_FRAME_HEADER_BYTES = 8 + 4 + 4 + 8 + 8 + 8
G1_POINT_BYTES = 48
MAX_SRS_FRAME_BYTES = 1 << 32


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


def append_u32(value: int) -> bytes:
    return value.to_bytes(4, "little", signed=False)


def append_u64(value: int) -> bytes:
    return value.to_bytes(8, "little", signed=False)


def append_string(value: str) -> bytes:
    encoded = value.encode()
    return append_u64(len(encoded)) + encoded


def segment_vector_digest(tensor_id: str, visibility: int, domain: int,
                          version: int, segment_size: int, segment_count: int,
                          srs_manifest_digest: str, basis_digest: str,
                          segments: list[dict[str, Any]]) -> str:
    encoded = append_string("zkdl.segment-commitment-vector.v1")
    encoded += append_string(tensor_id)
    encoded += append_u32(visibility) + append_u32(domain)
    encoded += append_u32(version) + append_u32(segment_size)
    encoded += append_u32(segment_count)
    encoded += bytes.fromhex(srs_manifest_digest)
    encoded += bytes.fromhex(basis_digest)
    encoded += append_u64(len(segments))
    for segment in segments:
        encoded += append_u32(segment["segment_index"])
        encoded += append_u64(segment["logical_offset"])
        encoded += append_u32(segment["actual_length"])
        encoded += append_u32(segment["padded_length"])
        encoded += append_u32(segment["padding_length"])
        encoded += bytes.fromhex(segment["frame_digest"])
        encoded += bytes.fromhex(segment["commitment"])
    return hashlib.sha256(encoded).hexdigest()


def config_identity_digest(path: Path) -> str:
    values = config_values(path)
    values.pop("verifier_registry_digest", None)
    values.pop("model_blinder_sidecar_digest", None)
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def config_binding_digest(path: Path) -> str:
    values = config_values(path)
    values.pop("srs_manifest_digest", None)
    values.pop("verifier_registry_digest", None)
    values.pop("model_blinder_sidecar_digest", None)
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def frame_path(root: Path, frame: dict[str, Any], label: str,
               digest_cache: dict[Path, str] | None = None) -> Path:
    raw = required_string(frame, "path", label)
    if Path(raw).is_absolute() or ".." in Path(raw).parts:
        fail(f"{label}.path escapes SRS directory")
    path = local(root, raw, f"{label}.path")
    regular(path, f"SRS {label} frame")
    declared = required_digest(frame, "sha256", label)
    actual = digest_cache.get(path) if digest_cache is not None else None
    if actual is None:
        actual = digest(path)
        if digest_cache is not None:
            digest_cache[path] = actual
    if actual != declared:
        fail(f"SRS {label} frame digest mismatch")
    count = frame.get("count")
    frame_bytes = frame.get("bytes")
    if not isinstance(count, int) or count < 1:
        fail(f"SRS {label}.count is invalid")
    if (not isinstance(frame_bytes, int) or
            frame_bytes != SRS_FRAME_HEADER_BYTES + count * G1_POINT_BYTES or
            frame_bytes > MAX_SRS_FRAME_BYTES):
        fail(f"SRS {label}.bytes is invalid")
    if path.stat().st_size != frame_bytes:
        fail(f"SRS {label}.bytes does not match the authenticated frame")
    return path


def validate_srs(root: Path, values: dict[str, str]) -> tuple[Path, dict[str, Any], str]:
    srs_raw = values.get("srs_path")
    manifest_digest = values.get("srs_manifest_digest")
    srs_id = values.get("srs_id")
    if not srs_raw or not manifest_digest or not srs_id:
        fail("secure config lacks srs_path, srs_id, or srs_manifest_digest")
    if manifest_digest == "UNAVAILABLE" or not HEX64.fullmatch(manifest_digest):
        fail("srs_manifest_digest must be a lowercase SHA-256")
    if values.get("srs_type") != "native_local_ephemeral_random":
        fail("secure config srs_type must be native_local_ephemeral_random")
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
    setup = manifest.get("setup")
    if not isinstance(setup, dict):
        fail("SRS manifest lacks local random setup metadata")
    required_setup = {
        "schema": "zkDL_gat_local_random_setup_v1",
        "mode": "native_local_ephemeral_random",
        "trust_status": "native_local_ephemeral_random",
        "algorithm": "os_csprng_nonzero_scalar_mul_bls12_381_g1",
        "spec": "zkDL_Hyrax_IPA_Pedersen_independent_g1_basis_v1",
        "curve_id": "bls12_381",
    }
    for key, expected in required_setup.items():
        if setup.get(key) != expected:
            fail(f"SRS setup.{key} is not {expected}")
    version = manifest.get("manifest_version")
    if version != 2 or manifest.get("chunking_rule") != "per_family_prefix_v1":
        fail("local random SRS requires manifest_version=2 and per_family_prefix_v1")
    basis = required_digest(manifest, "basis_digest", "SRS manifest")
    if required_digest(setup, "basis_digest", "SRS setup") != basis:
        fail("SRS setup basis digest is not manifest-bound")
    config_path = local(root, values["_config_path"], "config")
    if required_digest(setup, "config_binding_digest", "SRS setup") != config_binding_digest(config_path):
        fail("SRS setup config binding digest mismatch")
    domain = setup.get("domain")
    basis_domain = setup.get("basis_domain")
    logical_domain = setup.get("logical_domain")
    segment_size = setup.get("segment_size")
    segment_count = setup.get("segment_count")
    counts = (setup.get("generator_count"), setup.get("h_count"), setup.get("u_count"))
    if (not isinstance(domain, int) or domain < 1 or domain & (domain - 1) or
            not isinstance(basis_domain, int) or basis_domain != NATIVE_BASIS_DOMAIN or
            not isinstance(logical_domain, int) or logical_domain < 1 or
            logical_domain > MAX_NATIVE_DOMAIN or logical_domain & (logical_domain - 1) or
            domain != basis_domain or
            not isinstance(segment_size, int) or segment_size != basis_domain or
            not isinstance(segment_count, int) or segment_count < 1 or
            segment_count != (logical_domain + basis_domain - 1) // basis_domain or
            counts != (basis_domain, 1, 1)):
        fail("SRS setup logical/basis segment metadata is invalid")
    families = manifest.get("families")
    if not isinstance(families, list) or not families:
        fail("local random SRS manifest has no families")
    selected: tuple[int, str, dict[str, Any]] | None = None
    default_family = manifest.get("default_family_id")
    default_chunk = manifest.get("default_chunk_id")
    seen: set[tuple[str, int]] = set()
    generator_id = required_string(manifest, "generator_id", "SRS manifest")
    frame_digest_cache: dict[Path, str] = {}
    for family in families:
        if not isinstance(family, dict) or not isinstance(family.get("family_id"), str) or not family["family_id"]:
            fail("SRS family has no family_id")
        family_id = family["family_id"]
        chunks = family.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            fail(f"SRS family has no chunks: {family_id}")
        for chunk in chunks:
            if not isinstance(chunk, dict):
                fail("SRS chunk is not an object")
            for key in ("chunk_id", "generator_id", "basis_digest"):
                if not isinstance(chunk.get(key), str) or not chunk[key]:
                    fail(f"SRS chunk omits {key}")
            if (chunk["generator_id"] != generator_id or chunk["basis_digest"] != basis or
                    required_digest(chunk, "basis_digest", "SRS chunk") != basis):
                fail("SRS chunk generator/basis identity mismatch")
            index = chunk.get("chunk_index")
            chunk_domain = chunk.get("domain")
            chunk_logical = chunk.get("logical_domain")
            chunk_basis = chunk.get("basis_domain")
            chunk_size = chunk.get("segment_size")
            chunk_count = chunk.get("segment_count")
            if (not isinstance(index, int) or index < 0 or not isinstance(chunk_domain, int) or
                    chunk_domain != basis_domain or not isinstance(chunk_logical, int) or
                    chunk_logical < 1 or chunk_logical > MAX_NATIVE_DOMAIN or
                    chunk_logical & (chunk_logical - 1) or
                    not isinstance(chunk_basis, int) or chunk_basis != basis_domain or
                    not isinstance(chunk_size, int) or chunk_size != basis_domain or
                    not isinstance(chunk_count, int) or chunk_count != (chunk_logical + basis_domain - 1) // basis_domain):
                fail("SRS chunk index/domain is invalid")
            if (family_id, index) in seen:
                fail("SRS duplicate family/chunk_index")
            seen.add((family_id, index))
            required_domain = chunk.get("required_domain", chunk_domain)
            if (not isinstance(required_domain, int) or required_domain != chunk_logical):
                fail("SRS required_domain is not bound to logical_domain")
            if (chunk.get("family_id") != family_id or chunk.get("point_offset") != 0 or
                    chunk.get("point_count") != basis_domain or chunk.get("domain") != basis_domain or
                    chunk.get("degree") != max(0, basis_domain - 1) or chunk.get("point_size") != 48):
                fail("SRS chunk domain/degree/point metadata mismatch")
            segments = chunk.get("segments")
            if not isinstance(segments, list) or len(segments) != chunk_count:
                fail("SRS chunk segment descriptor count mismatch")
            expected_offset = 0
            for segment_index, segment in enumerate(segments):
                if not isinstance(segment, dict) or segment.get("segment_index") != segment_index:
                    fail("SRS segment ordering is not canonical")
                actual = segment.get("actual_length")
                if (segment.get("logical_offset") != expected_offset or
                        not isinstance(actual, int) or actual < 1 or actual > basis_domain or
                        segment.get("padded_length") != basis_domain or
                        segment.get("basis_offset") != 0 or segment.get("basis_count") != basis_domain):
                    fail("SRS segment offset/length/basis metadata is invalid")
                expected_offset += actual
            if expected_offset != chunk_logical:
                fail("SRS segments do not cover the logical domain exactly")
            frames: dict[str, Path] = {}
            for label in ("G", "H", "U"):
                frame = chunk.get(label)
                if not isinstance(frame, dict):
                    fail(f"SRS chunk omits {label} frame")
                frames[label] = frame_path(srs, frame, f"{family_id}/{chunk['chunk_id']}/{label}", frame_digest_cache)
                expected = basis_domain if label == "G" else 1
                if frame.get("count") != expected:
                    fail(f"SRS {label} count does not match chunk domain")
            if frames["H"].read_bytes() == frames["U"].read_bytes():
                fail("SRS H and U frames are byte-identical")
            if required_digest(chunk, "generator_digest", "SRS chunk") != chunk["G"]["sha256"]:
                fail("SRS generator_digest does not bind G frame")
            expected_basis = canonical_digest({
                "G": chunk["G"]["sha256"], "H": chunk["H"]["sha256"],
                "U": chunk["U"]["sha256"], "schema": "zkDL_gat_basis_digest_v1",
            }, set())
            if expected_basis != basis:
                fail("SRS basis digest does not bind G/H/U frame digests")
            candidate = (domain, family_id, chunk)
            if default_family is not None or default_chunk is not None:
                if family_id == default_family and chunk["chunk_id"] == default_chunk:
                    selected = candidate
            elif selected is None or candidate[:2] > selected[:2]:
                selected = candidate
    if selected is None:
        fail("local random SRS has no selectable chunk")
    maximum_logical = max(chunk["logical_domain"] for family in families for chunk in family["chunks"])
    if maximum_logical != logical_domain or selected[2]["logical_domain"] != logical_domain:
        fail("SRS setup/default logical domain does not cover its family registry exactly")
    return srs, manifest, selected[2]["generator_digest"]


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
    if required_string(registry, "schema", "model registry") != "zkDL_gat_model_registry_v3":
        fail("model registry schema is not zkDL_gat_model_registry_v3")
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
        version = tensor.get("segment_binding_version")
        segment_size = tensor.get("segment_size")
        segment_count = tensor.get("segment_count")
        expected_segment_size = min(domain, NATIVE_BASIS_DOMAIN)
        if (version != 1 or not isinstance(segment_size, int) or segment_size < 1 or
                segment_size & (segment_size - 1) or segment_size != expected_segment_size or
                not isinstance(segment_count, int) or segment_count < 1 or
                segment_count != (domain + segment_size - 1) // segment_size):
            fail(f"model tensor segment layout is not canonical: {tensor_id}")
        srs_manifest_digest = required_digest(tensor, "srs_manifest_digest", "model tensor")
        basis_digest = required_digest(tensor, "basis_digest", "model tensor")
        vector_digest = required_digest(tensor, "segment_commitment_digest", "model tensor")
        if srs_manifest_digest != required_digest(registry, "srs_manifest_digest", "model registry"):
            fail(f"model tensor SRS manifest binding mismatch: {tensor_id}")
        segments = tensor.get("segment_commitments")
        if not isinstance(segments, list) or len(segments) != segment_count:
            fail(f"model tensor ordered segment vector is incomplete: {tensor_id}")
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                fail(f"model tensor segment is not an object: {tensor_id}/{index}")
            if (segment.get("segment_index") != index or
                    segment.get("logical_offset") != index * segment_size or
                    not isinstance(segment.get("actual_length"), int) or
                    segment["actual_length"] < 1 or segment["actual_length"] > segment_size or
                    segment.get("padded_length") != segment_size or
                    segment.get("padding_length") != segment_size - segment["actual_length"] or
                    not isinstance(segment.get("frame_digest"), str) or
                    not HEX64.fullmatch(segment["frame_digest"]) or
                    not isinstance(segment.get("commitment"), str) or
                    not re.fullmatch(r"[0-9a-f]{96}", segment["commitment"])):
                fail(f"model tensor segment layout/point is not canonical: {tensor_id}/{index}")
        if segment_vector_digest(
                tensor_id, 2, domain, version, segment_size, segment_count,
                srs_manifest_digest, basis_digest, segments) != vector_digest:
            fail(f"model tensor segment commitment digest mismatch: {tensor_id}")
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


def validate_sorted_lookup_descriptor(root: Path, values: dict[str, str],
                                      static_table_path: Path) -> tuple[Path, str]:
    raw_path = values.get("sorted_lookup_descriptor")
    if not raw_path:
        fail("config lacks sorted_lookup_descriptor")
    path = local(root, raw_path, "sorted_lookup_descriptor")
    descriptor = json_object(path, "sorted lookup descriptor")
    if descriptor.get("schema") != "zkDL_gat_sorted_lookup_descriptor_v1" or descriptor.get("version") != 1:
        fail("sorted lookup descriptor schema/version mismatch")
    if descriptor.get("source_root") != "baseline-local":
        fail("sorted lookup descriptor source_root is not baseline-local")
    if descriptor.get("static_table_digest") != digest(static_table_path):
        fail("sorted lookup descriptor is not bound to the authenticated static table")
    if descriptor.get("key_encoding") != "packed_pair_u128":
        fail("sorted lookup descriptor key encoding is not packed_pair_u128")
    if descriptor.get("active_mask_rule") != "prefix_one_then_zero_v1":
        fail("sorted lookup descriptor active-mask rule mismatch")
    if descriptor.get("sorted_delta_rule") != "nonnegative_bounded_delta_v1":
        fail("sorted lookup descriptor delta rule mismatch")
    if descriptor.get("marker_rule") != "group_first_v1" or descriptor.get("inverse_rule") != "delta_inverse_or_zero_v1":
        fail("sorted lookup descriptor marker/inverse rule mismatch")
    segment_size = descriptor.get("segment_size")
    if not isinstance(segment_size, int) or segment_size != NATIVE_BASIS_DOMAIN:
        fail("sorted lookup descriptor segment_size is not the bounded native basis")
    families = descriptor.get("table_families")
    if not isinstance(families, list) or len(families) != 3:
        fail("sorted lookup descriptor must contain exactly three canonical table families")
    seen: set[str] = set()
    for family in families:
        if not isinstance(family, dict):
            fail("sorted lookup table family is not an object")
        table_ref = family.get("table_ref")
        if not isinstance(table_ref, str) or table_ref in seen or table_ref not in {"static_lrelu", "static_elu", "static_exp"}:
            fail("sorted lookup table family is not canonical")
        seen.add(table_ref)
        logical_rows = family.get("logical_rows")
        padded_domain = family.get("padded_domain")
        if (not isinstance(logical_rows, int) or logical_rows < 1 or
                not isinstance(padded_domain, int) or padded_domain < logical_rows or
                padded_domain & (padded_domain - 1) or family.get("sorted_unique") is not True or
                family.get("padding_rule") != "zero_after_logical_rows_v1"):
            fail(f"sorted lookup family has invalid logical-domain metadata: {table_ref}")
        x_bits, y_bits = family.get("x_bits"), family.get("y_bits")
        if (not isinstance(x_bits, int) or not isinstance(y_bits, int) or
                x_bits < 1 or y_bits < 1 or x_bits + y_bits > 127 or
                family.get("key_encoding") != "packed_pair_u128"):
            fail(f"sorted lookup family has invalid u128 bounds: {table_ref}")
    if set(seen) != {"static_lrelu", "static_elu", "static_exp"}:
        fail("sorted lookup descriptor does not cover every canonical table")
    queries = descriptor.get("query_families")
    if not isinstance(queries, list) or not queries or any(
            not isinstance(item, dict) or not isinstance(item.get("query_rows"), int) or item["query_rows"] < 1
            for item in queries):
        fail("sorted lookup descriptor query families are incomplete")
    declared = descriptor.get("descriptor_digest")
    if not isinstance(declared, str) or not HEX64.fullmatch(declared):
        fail("sorted lookup descriptor digest is not canonical")
    if declared != canonical_digest(descriptor, {"descriptor_digest"}):
        fail("sorted lookup descriptor self-digest mismatch")
    return path, digest(path)


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
    static_table_path = local(root, values["static_table_descriptor"], "static_table_descriptor")
    sorted_lookup_path, sorted_lookup_digest = validate_sorted_lookup_descriptor(
        root, values, static_table_path)
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
        "sorted_lookup_descriptor_path": sorted_lookup_path.relative_to(root).as_posix(),
        "sorted_lookup_descriptor_digest": sorted_lookup_digest,
        "status": "AUTHENTICATED",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
