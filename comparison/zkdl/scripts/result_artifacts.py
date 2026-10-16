#!/usr/bin/env python3
"""Strict audit helpers for baseline-local zkDL result artifacts.

This module is intentionally metadata-only with respect to the proof system:
it never proves, verifies, creates, or repairs an artifact.  It checks the
files emitted by the native runner and rejects a result unless the native
proof envelope, independent-verification record, public statement, and
authenticated baseline-local resources agree byte-for-byte where applicable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from formal_provenance import validate_manifest


DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
BACKENDS = ("cpu", "gpu")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
NATIVE_BASIS_DOMAIN = 1 << 18
MAX_NATIVE_DOMAIN = 1 << 30
SRS_FRAME_HEADER_BYTES = 8 + 4 + 4 + 8 + 8 + 8
G1_POINT_BYTES = 48
MAX_SRS_FRAME_BYTES = 1 << 32
FIXTURE_PROOF_SCOPE = "fixture_segmented_relation"


class ArtifactError(ValueError):
    """A production artifact failed a fail-closed audit."""


def fixture_statement_id(dataset: str, proof_digest: str,
                         evidence_digest: str) -> str:
    """Derive the explicit fixture statement identity from bound evidence."""
    encoded = ("zkDL.gat.fixture.statement.v1\n" + dataset + "\n" +
               proof_digest + "\n" + evidence_digest + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ArtifactError(f"{label} is missing or symlinked: {path}")
    return path


def _local(root: Path, raw: str, label: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ArtifactError(f"{label} is absolute: {raw}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactError(f"{label} escapes baseline root: {raw}") from exc
    return resolved


def read_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, line in enumerate(_regular(path, "key/value artifact").read_text(encoding="utf-8").splitlines(), 1):
        if "=" not in line:
            raise ArtifactError(f"malformed key/value line {number}: {path}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ArtifactError(f"duplicate or empty key {key!r}: {path}")
        values[key] = value
    return values


def _config_binding_digest(path: Path) -> str:
    values: dict[str, str] = {}
    for number, raw in enumerate(_regular(path, "config").read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ArtifactError(f"malformed config line {number}: {path}")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value or key in values:
            raise ArtifactError(f"invalid or duplicate config key at line {number}: {path}")
        values[key] = value
    for key in ("srs_manifest_digest", "verifier_registry_digest", "model_blinder_sidecar_digest"):
        values.pop(key, None)
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _basis_digest(g_digest: str, h_digest: str, u_digest: str) -> str:
    encoded = json.dumps({
        "G": g_digest,
        "H": h_digest,
        "U": u_digest,
        "schema": "zkDL_gat_basis_digest_v1",
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _required(values: dict[str, str], key: str, path: Path) -> str:
    value = values.get(key)
    if value is None or value == "":
        raise ArtifactError(f"missing field {key}: {path}")
    return value


def _validate_prove_timing_diagnostics(timing: dict[str, str], envelope: dict[str, str], path: Path) -> None:
    expected_scope = (
        "signed_round_aux_witness_and_relation_views;steady_clock_wall;"
        "exclude_family_storage_write_hash_and_explicit_file_io"
    )
    expected_fields = {
        "schema_id", "status", "statement_id", "proof_id", "commitment_time_seconds", "prove_time_seconds",
        "proof_auxiliary_construction_seconds", "proof_auxiliary_construction_scope",
        "sorted_lookup_protocol_construction_seconds", "relation_context_prove_seconds", "relation_prove_wall_seconds",
        "relation_commitment_time_seconds", "prove_provider_file_io_excluded_seconds", "proof_serialization_seconds",
    }
    if set(timing) != expected_fields or timing.get("schema_id") != "zkDL_gat_proof_timing_v1" or timing.get("status") != "PROVED":
        raise ArtifactError(f"proof timing sidecar schema/status mismatch: {path}")
    if timing.get("proof_auxiliary_construction_scope") != expected_scope:
        raise ArtifactError(f"proof timing diagnostic scope mismatch: {path}")
    if timing.get("statement_id") != envelope.get("statement_id") or timing.get("proof_id") != envelope.get("proof_id"):
        raise ArtifactError(f"proof timing sidecar is not bound to the proof envelope: {path}")
    fields = (
        "proof_auxiliary_construction_seconds",
        "sorted_lookup_protocol_construction_seconds",
        "relation_context_prove_seconds",
        "relation_prove_wall_seconds",
        "relation_commitment_time_seconds",
        "prove_provider_file_io_excluded_seconds",
        "proof_serialization_seconds",
    )
    metrics: dict[str, float] = {}
    for field in fields:
        try:
            value = float(timing[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(f"proof timing diagnostic is missing or invalid: {field}") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ArtifactError(f"proof timing diagnostic is negative or non-finite: {field}")
        metrics[field] = value
    if metrics["relation_commitment_time_seconds"] > metrics["relation_prove_wall_seconds"]:
        raise ArtifactError(f"relation commitment interval exceeds relation prover wall time: {path}")
    if metrics["prove_provider_file_io_excluded_seconds"] > (
        metrics["relation_prove_wall_seconds"] - metrics["relation_commitment_time_seconds"]
    ):
        raise ArtifactError(f"excluded provider I/O exceeds non-commit prover wall time: {path}")
    computed = (
        metrics["proof_auxiliary_construction_seconds"]
        + metrics["sorted_lookup_protocol_construction_seconds"]
        + metrics["relation_context_prove_seconds"]
        + metrics["relation_prove_wall_seconds"]
        - metrics["relation_commitment_time_seconds"]
        - metrics["prove_provider_file_io_excluded_seconds"]
        + metrics["proof_serialization_seconds"]
    )
    try:
        reported = float(timing["prove_time_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"proof envelope is missing valid prove_time_seconds: {path}") from exc
    try:
        sidecar_commitment = float(timing["commitment_time_seconds"])
        envelope_commitment = float(envelope["commitment_time_seconds"])
        envelope_prove = float(envelope["prove_time_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"proof timing sidecar paper metrics are invalid: {path}") from exc
    if (not math.isfinite(reported) or not math.isfinite(sidecar_commitment) or
            sidecar_commitment < 0.0 or sidecar_commitment != envelope_commitment or
            reported != envelope_prove or not math.isclose(computed, reported, rel_tol=2e-12, abs_tol=2e-12)):
        raise ArtifactError(f"raw proof timing components do not sum to Prove Time: {path}")


def _digest(value: str, field: str, *, unavailable: bool = False) -> None:
    if unavailable and value == "UNAVAILABLE":
        return
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ArtifactError(f"{field} is not a canonical SHA-256 digest")


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_regular(path, label).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid JSON {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{label} must be a JSON object: {path}")
    return value


def _manifest_files(root: Path, manifest: dict[str, Any], manifest_path: Path) -> dict[str, str]:
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ArtifactError(f"asset manifest has no authenticated files: {manifest_path}")
    digests: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ArtifactError(f"asset manifest file entry is not an object: {manifest_path}")
        raw_path = entry.get("path")
        raw_bytes = entry.get("bytes")
        raw_digest = entry.get("sha256")
        if not isinstance(raw_path, str) or not raw_path or not isinstance(raw_bytes, int):
            raise ArtifactError(f"malformed asset manifest file entry: {manifest_path}")
        if not isinstance(raw_digest, str) or not HEX64.fullmatch(raw_digest):
            raise ArtifactError(f"non-canonical asset digest for {raw_path}: {manifest_path}")
        if raw_path in digests:
            raise ArtifactError(f"duplicate asset manifest file: {raw_path}")
        path = _local(root, raw_path, "asset manifest path")
        _regular(path, "asset manifest file")
        if path.stat().st_size != raw_bytes:
            raise ArtifactError(f"asset byte-size mismatch for {raw_path}")
        actual = sha256_file(path)
        if actual != raw_digest:
            raise ArtifactError(f"asset digest mismatch for {raw_path}")
        digests[raw_path] = raw_digest
    return digests


def _resource_manifest(root: Path, manifest: dict[str, Any], manifest_path: Path,
                       digests: dict[str, str]) -> None:
    """Require authenticated native resources, never an UNAVAILABLE marker."""
    for field in (
        "srs_path", "srs_manifest_digest", "model_registry_path",
        "model_registry_digest",
    ):
        value = manifest.get(field)
        if not isinstance(value, str) or not value or value == "UNAVAILABLE":
            raise ArtifactError(f"formal asset manifest has unavailable resource {field}: {manifest_path}")

    srs_path = _local(root, manifest["srs_path"], "SRS path")
    if srs_path.is_symlink() or not srs_path.is_dir():
        raise ArtifactError(f"SRS directory is missing or symlinked: {srs_path}")
    srs_manifest = _regular(srs_path / "srs_manifest.json", "SRS manifest")
    if sha256_file(srs_manifest) != manifest["srs_manifest_digest"]:
        raise ArtifactError("SRS manifest digest does not match asset manifest")
    srs_manifest_relative = srs_manifest.relative_to(root).as_posix()
    if digests.get(srs_manifest_relative) != manifest["srs_manifest_digest"]:
        raise ArtifactError("asset manifest does not authenticate srs_manifest.json")
    _digest(manifest["srs_manifest_digest"], "srs_manifest_digest")
    srs = _json(srs_manifest, "SRS manifest")
    setup = srs.get("setup")
    expected_setup = {
        "schema": "zkDL_gat_local_random_setup_v1",
        "mode": "native_local_ephemeral_random",
        "trust_status": "native_local_ephemeral_random",
        "algorithm": "os_csprng_nonzero_scalar_mul_bls12_381_g1",
        "spec": "zkDL_Hyrax_IPA_Pedersen_independent_g1_basis_v1",
        "curve_id": "bls12_381",
    }
    if not isinstance(setup, dict) or any(setup.get(key) != value for key, value in expected_setup.items()):
        raise ArtifactError("SRS manifest local random setup metadata mismatch")
    basis = srs.get("basis_digest")
    if not isinstance(basis, str) or not HEX64.fullmatch(basis) or setup.get("basis_digest") != basis:
        raise ArtifactError("SRS manifest basis digest is not setup-bound")
    basis_domain = setup.get("basis_domain")
    logical_domain = setup.get("logical_domain")
    segment_size = setup.get("segment_size")
    segment_count = setup.get("segment_count")
    if (basis_domain != NATIVE_BASIS_DOMAIN or not isinstance(logical_domain, int) or
            logical_domain < 1 or logical_domain > MAX_NATIVE_DOMAIN or
            logical_domain & (logical_domain - 1) or setup.get("domain") != basis_domain or
            segment_size != basis_domain or
            not isinstance(segment_count, int) or segment_count < 1 or
            segment_count != (logical_domain + basis_domain - 1) // basis_domain or
            setup.get("generator_count") != basis_domain):
        raise ArtifactError("SRS manifest logical/basis segment setup is invalid")
    config_raw = manifest.get("config")
    if not isinstance(config_raw, str):
        raise ArtifactError("asset manifest lacks config path")
    config_path = _local(root, config_raw, "config path")
    config_relative = config_path.relative_to(root).as_posix()
    if digests.get(config_relative) != sha256_file(config_path):
        raise ArtifactError("asset manifest does not authenticate config")
    if setup.get("config_binding_digest") != _config_binding_digest(config_path):
        raise ArtifactError("SRS setup config binding digest mismatch")

    def check_frame(item: Any, label: str, expected_count: int) -> str:
        if not isinstance(item, dict):
            raise ArtifactError(f"SRS manifest missing {label} frame")
        raw = item.get("path")
        declared = item.get("sha256")
        count = item.get("count")
        frame_bytes = item.get("bytes")
        if (not isinstance(raw, str) or not isinstance(declared, str) or not HEX64.fullmatch(declared) or
                not isinstance(count, int) or count != expected_count or
                not isinstance(frame_bytes, int) or
                frame_bytes != SRS_FRAME_HEADER_BYTES + count * G1_POINT_BYTES or
                frame_bytes > MAX_SRS_FRAME_BYTES):
            raise ArtifactError(f"SRS manifest has malformed {label} frame identity")
        frame = _local(srs_path, raw, f"SRS {label} frame")
        _regular(frame, f"SRS {label} frame")
        if frame.stat().st_size != frame_bytes:
            raise ArtifactError(f"SRS manifest {label} frame byte-size mismatch")
        relative = frame.relative_to(root).as_posix()
        if relative not in digests or digests[relative] != declared:
            raise ArtifactError(f"asset manifest does not authenticate SRS {label} frame")
        return declared

    families = srs.get("families")
    if not isinstance(families, list) or not families:
        raise ArtifactError("SRS manifest lacks canonical families")
    generator_id = srs.get("generator_id")
    if not isinstance(generator_id, str) or not generator_id:
        raise ArtifactError("SRS manifest lacks generator_id")
    for family in families:
        if not isinstance(family, dict) or not isinstance(family.get("chunks"), list) or not family["chunks"]:
            raise ArtifactError("SRS manifest has malformed family/chunk entries")
        for chunk in family["chunks"]:
            if not isinstance(chunk, dict):
                raise ArtifactError("SRS manifest has malformed chunk entry")
            if (chunk.get("generator_id") != generator_id or
                    chunk.get("basis_digest") != basis or
                    chunk.get("family_id") != family.get("family_id")):
                raise ArtifactError("SRS chunk generator/basis identity mismatch")
            chunk_logical = chunk.get("logical_domain")
            if (chunk.get("basis_domain") != basis_domain or
                    chunk.get("segment_size") != basis_domain or
                    not isinstance(chunk_logical, int) or chunk_logical < 1 or
                    chunk_logical > MAX_NATIVE_DOMAIN or chunk_logical & (chunk_logical - 1) or
                    chunk.get("point_count") != basis_domain or
                    chunk.get("domain") != basis_domain or
                    chunk.get("point_offset") != 0 or
                    chunk.get("degree") != max(0, basis_domain - 1) or
                    chunk.get("point_size") != G1_POINT_BYTES or
                    chunk.get("segment_count") !=
                    (chunk_logical + basis_domain - 1) // basis_domain or
                    chunk.get("required_domain") != chunk_logical):
                raise ArtifactError("SRS chunk logical/basis segment metadata mismatch")
            segments = chunk.get("segments")
            if not isinstance(segments, list) or len(segments) != chunk.get("segment_count"):
                raise ArtifactError("SRS chunk segment descriptor count mismatch")
            offset = 0
            for index, segment in enumerate(segments):
                if (not isinstance(segment, dict) or segment.get("segment_index") != index or
                        segment.get("logical_offset") != offset or
                        not isinstance(segment.get("actual_length"), int) or
                        segment["actual_length"] < 1 or segment["actual_length"] > basis_domain or
                        segment.get("padded_length") != basis_domain or
                        segment.get("basis_offset") != 0 or
                        segment.get("basis_count") != basis_domain):
                    raise ArtifactError("SRS segment descriptor is not ordered/full coverage")
                offset += segment["actual_length"]
            if offset != chunk_logical:
                raise ArtifactError("SRS segments do not cover logical domain")
            frame_digests = {
                key: check_frame(chunk.get(key), f"{family.get('family_id', '?')}/{key}",
                                 basis_domain if key == "G" else 1)
                for key in ("G", "H", "U")
            }
            if chunk.get("generator_digest") != frame_digests["G"]:
                raise ArtifactError("SRS chunk generator digest mismatch")
            if _basis_digest(frame_digests["G"], frame_digests["H"], frame_digests["U"]) != basis:
                raise ArtifactError("SRS chunk basis digest does not bind frames")

    registry_path = _local(root, manifest["model_registry_path"], "model registry path")
    _regular(registry_path, "model registry")
    if sha256_file(registry_path) != manifest["model_registry_digest"]:
        raise ArtifactError("model registry digest does not match asset manifest")
    _digest(manifest["model_registry_digest"], "model_registry_digest")
    if manifest.get("model_blinder_sidecar_path") != "UNAVAILABLE" or manifest.get("model_blinder_sidecar_digest") != "UNAVAILABLE":
        raise ArtifactError("public assets expose private model registration material")
    lookup_raw = manifest.get("sorted_lookup_descriptor")
    lookup_digest = manifest.get("sorted_lookup_descriptor_digest")
    if (not isinstance(lookup_raw, str) or not lookup_raw or
            not isinstance(lookup_digest, str) or not HEX64.fullmatch(lookup_digest)):
        raise ArtifactError("asset manifest lacks authenticated sorted-lookup descriptor")
    lookup_path = _local(root, lookup_raw, "sorted lookup descriptor")
    _regular(lookup_path, "sorted lookup descriptor")
    if digests.get(lookup_path.relative_to(root).as_posix()) != lookup_digest:
        raise ArtifactError("asset manifest does not authenticate sorted-lookup descriptor")
    lookup = _json(lookup_path, "sorted lookup descriptor")
    if lookup.get("schema") != "zkDL_gat_sorted_lookup_descriptor_v1" or lookup.get("version") != 1:
        raise ArtifactError("sorted-lookup descriptor schema/version mismatch")
    if lookup.get("static_table_digest") != manifest.get("static_table_descriptor_digest"):
        raise ArtifactError("sorted-lookup descriptor table binding mismatch")
    if any("chi" in str(item) for item in lookup.get("table_families", []) + lookup.get("query_families", [])):
        raise ArtifactError("formal sorted-lookup descriptor retains legacy chi dependency")
    if lookup.get("descriptor_digest") != hashlib.sha256(
            json.dumps({key: value for key, value in lookup.items() if key != "descriptor_digest"},
                       sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest():
        raise ArtifactError("sorted-lookup descriptor self-digest mismatch")


def _audit_assets(root: Path, run_dir: Path, statement: dict[str, str]) -> tuple[dict[str, Any], str]:
    manifest_path = _regular(run_dir / "asset_manifest.json", "asset manifest")
    manifest = _json(manifest_path, "asset manifest")
    if manifest.get("schema") != "zkDL_gat_asset_manifest_v1":
        raise ArtifactError(f"asset manifest schema mismatch: {manifest_path}")
    if manifest.get("dataset") != statement.get("dataset"):
        raise ArtifactError("asset manifest dataset mismatch")
    asset_digest = sha256_file(manifest_path)
    if statement.get("asset_manifest_digest") != asset_digest:
        raise ArtifactError("statement asset_manifest_digest does not match asset manifest")
    config_raw = manifest.get("config")
    config_path = _local(root, config_raw if isinstance(config_raw, str) else "", "config")
    config_values = {key.strip(): value.strip() for key, value in read_kv(config_path).items()}
    checkpoint = config_values.get("checkpoint_bundle", "").strip() + "/"
    for entry in manifest.get("files", []):
        raw = entry.get("path", "") if isinstance(entry, dict) else ""
        if "/model_blinders/" in raw or raw.endswith("/private_asset_manifest.json") or (raw.startswith(checkpoint) and raw != checkpoint + "manifest.json"):
            raise ArtifactError("public assets contain private checkpoint or opening material")
    digests = _manifest_files(root, manifest, manifest_path)
    _resource_manifest(root, manifest, manifest_path, digests)
    build_path = _local(root, manifest.get("build_manifest", ""), "build manifest")
    if build_path != run_dir / "build_manifest.json":
        raise ArtifactError("formal build manifest must be run-local")
    try:
        build = validate_manifest(root, build_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ArtifactError(f"formal build provenance is invalid: {exc}") from exc
    build_digest = sha256_file(build_path)
    if manifest.get("build_manifest_digest") != build_digest or digests.get(build_path.relative_to(root).as_posix()) != build_digest or manifest.get("source_digest") != build["source_digest"]:
        raise ArtifactError("formal build/source identity is not asset-bound")
    for entry in build["sources"] + build["binaries"]:
        if digests.get(entry["path"]) != entry["sha256"]:
            raise ArtifactError(f"formal build source/binary is not asset-bound: {entry['path']}")

    for field in (
        "scale_chain_digest", "static_table_descriptor_digest",
        "integer_bound_ledger_descriptor_digest", "sorted_lookup_descriptor_digest",
    ):
        value = manifest.get(field)
        _digest(value, field)
        if field == "scale_chain_digest":
            relative = "configs/quantization_scale_chain.cfg"
        else:
            descriptor_key = (
                "static_table_descriptor" if field.startswith("static") else
                "integer_bound_ledger" if field.startswith("integer") else
                "sorted_lookup_descriptor")
            raw = manifest.get(descriptor_key)
            if not isinstance(raw, str):
                raise ArtifactError(f"asset manifest lacks {descriptor_key}")
            relative = raw
        if digests.get(relative) != value:
            raise ArtifactError(f"asset manifest identity is not bound to {relative}")
        if statement.get(field) != value:
            raise ArtifactError(f"statement resource identity mismatch for {field}")
    if statement.get("verifier_registry_digest") != manifest.get("model_registry_digest"):
        raise ArtifactError("statement verifier registry digest mismatch")
    for field in ("checkpoint_tensors_txt_digest", "checkpoint_tensors_npz_digest"):
        _digest(manifest.get(field), field)
        if statement.get(field) != manifest[field]:
            raise ArtifactError(f"statement checkpoint identity mismatch for {field}")
    if statement.get("scale_chain_digest") != manifest.get("scale_chain_digest"):
        raise ArtifactError("statement scale-chain digest mismatch")
    if statement.get("proof_scope") != "full_gat_forward":
        raise ArtifactError("statement proof scope is not full_gat_forward")
    if statement.get("verifier_reads_witness") != "0":
        raise ArtifactError("statement verifier reads witness")
    return manifest, asset_digest


def _audit_fixture_run(root: Path, run_dir: Path, *, expected_dataset: str | None = None,
                       expected_backend: str | None = None) -> dict[str, str]:
    """Audit one explicitly-marked tiny CPU fixture artifact.

    This path is never selected by the formal collector/validator defaults.
    It still binds every result row to the real native proof bytes and to a
    separately recorded independent-verifier evidence digest, while refusing
    GPU/fallback claims and stale or missing sidecars.
    """
    run_dir = run_dir.resolve()
    try:
        run_dir.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactError(f"fixture run directory escapes fixture root: {run_dir}") from exc
    record_path = _regular(run_dir / "run.record", "fixture run record")
    proof_path = _regular(run_dir / "proof.bin", "fixture proof artifact")
    envelope_path = _regular(run_dir / "proof.meta", "fixture proof envelope")
    timing_path = _regular(run_dir / "proof.timing", "fixture proof timing diagnostics")
    statement_path = _regular(run_dir / "statement.txt", "fixture public statement")
    verification_path = _regular(run_dir / "verification.result", "fixture verification result")
    evidence_path = _regular(run_dir / "independent_verify.evidence", "fixture verifier evidence")
    public_output = _regular(run_dir / "public_Y.txt", "fixture public output")
    record = read_kv(record_path)
    envelope = read_kv(envelope_path)
    timing = read_kv(timing_path)
    statement = read_kv(statement_path)
    verification = read_kv(verification_path)
    dataset = _required(record, "dataset", record_path)
    backend = _required(record, "backend", record_path)
    if dataset not in DATASETS:
        raise ArtifactError(f"fixture dataset is non-canonical: {dataset}")
    if backend != "cpu":
        raise ArtifactError(f"fixture backend is not CPU: {backend}")
    if expected_dataset is not None and dataset != expected_dataset:
        raise ArtifactError(f"fixture dataset mismatch in {record_path}")
    if expected_backend is not None and backend != expected_backend:
        raise ArtifactError(f"fixture backend mismatch in {record_path}")
    if any(values.get("fixture_mode") != "1"
           for values in (record, envelope, statement, verification)):
        raise ArtifactError(f"fixture mode marker is missing: {run_dir}")
    if record.get("schema") != "zkDL_gat_run_record_v2" or record.get("verification_status") != "VERIFIED":
        raise ArtifactError(f"fixture run record is not VERIFIED: {record_path}")
    if envelope.get("schema_id") != "zkDL_gat_proof_envelope_v1" or envelope.get("status") != "PROVED":
        raise ArtifactError(f"fixture proof envelope is not PROVED: {envelope_path}")
    _validate_prove_timing_diagnostics(timing, envelope, timing_path)
    if (record.get("commitment_time_semantics") != "native_tensor_commitments" or
            envelope.get("commitment_time_semantics") != "native_tensor_commitments" or
            record.get("commitment_time_semantics") != envelope.get("commitment_time_semantics")):
        raise ArtifactError("fixture commitment time does not identify native tensor commitments")
    if envelope.get("backend") != "cpu" or envelope.get("proof_scope") != FIXTURE_PROOF_SCOPE:
        raise ArtifactError(f"fixture proof envelope backend/scope mismatch: {envelope_path}")
    if verification.get("status") != "VERIFIED" or verification.get("backend") != "cpu":
        raise ArtifactError(f"fixture independent verifier did not return VERIFIED: {verification_path}")
    if statement.get("schema") != "zkDL_gat_statement_v1" or statement.get("dataset") != dataset:
        raise ArtifactError(f"fixture statement schema/dataset mismatch: {statement_path}")
    if statement.get("proof_scope") != FIXTURE_PROOF_SCOPE or statement.get("verifier_reads_witness") != "0":
        raise ArtifactError(f"fixture statement violates verifier boundary: {statement_path}")
    for values, label in ((record, "record"), (envelope, "envelope"),
                          (statement, "statement"), (verification, "verification")):
        if values.get("silent_fallback") != "0":
            raise ArtifactError(f"fixture {label} claims silent fallback")
        if values.get("gpu_kernel_executed") != "0":
            raise ArtifactError(f"fixture {label} claims GPU execution")
    proof_bytes = proof_path.read_bytes()
    if len(proof_bytes) < 4 or proof_bytes[:4] != b"GATR":
        raise ArtifactError(f"fixture proof is not a native GATR artifact: {proof_path}")
    proof_digest = sha256_file(proof_path)
    evidence = evidence_path.read_text(encoding="utf-8")
    evidence_digest = sha256_file(evidence_path)
    if "independent_verify=PASS" not in evidence or f"artifact_sha256={proof_digest}" not in evidence:
        raise ArtifactError("fixture evidence does not bind independent verification to proof bytes")
    expected_statement = fixture_statement_id(dataset, proof_digest, evidence_digest)
    if statement.get("statement_id") != expected_statement:
        raise ArtifactError("fixture statement identity is stale or not evidence-bound")
    for values, label in ((record, "record"), (envelope, "envelope"),
                          (verification, "verification")):
        if values.get("statement_id") != expected_statement:
            raise ArtifactError(f"fixture {label}/statement identity mismatch")
        if values.get("proof_id") != proof_digest:
            raise ArtifactError(f"fixture {label}/proof identity mismatch")
    if statement.get("proof_id") != proof_digest or statement.get("fixture_evidence_digest") != evidence_digest:
        raise ArtifactError("fixture statement artifact/evidence identity mismatch")
    try:
        declared_size = int(record.get("proof_size_bytes", "0"))
        envelope_size = int(envelope.get("proof_size_bytes", "0"))
    except ValueError as exc:
        raise ArtifactError("fixture proof size is not an integer") from exc
    if declared_size != len(proof_bytes) or envelope_size != len(proof_bytes):
        raise ArtifactError("fixture proof size is stale")
    for field in ("commitment_time_seconds", "prove_time_seconds", "verify_time_seconds"):
        try:
            record_value = float(record[field])
            envelope_value = float(envelope[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(f"fixture metric is missing or invalid: {field}") from exc
        if not math.isfinite(record_value) or record_value < 0.0 or record_value != envelope_value:
            raise ArtifactError(f"fixture record/envelope metric mismatch: {field}")
    if record.get("proof_scope") != FIXTURE_PROOF_SCOPE or envelope.get("proof_scope") != FIXTURE_PROOF_SCOPE:
        raise ArtifactError("fixture record/envelope proof scope mismatch")
    if record.get("verifier_reads_witness") != "0" or record.get("output_bound") != "1":
        raise ArtifactError("fixture record violates CPU/output boundary")
    if envelope.get("verifier_reads_witness") != "0" or envelope.get("output_bound") != "1":
        raise ArtifactError("fixture envelope violates CPU/output boundary")
    if public_output.stat().st_size == 0:
        raise ArtifactError("fixture public output is empty")
    return {
        **record,
        "run_dir": run_dir.relative_to(root.resolve()).as_posix(),
        "proof_digest": proof_digest,
        "proof_artifact_path": proof_path.relative_to(root.resolve()).as_posix(),
        "proof_meta_digest": sha256_file(envelope_path),
        "statement_digest": sha256_file(statement_path),
        "verification_result_digest": sha256_file(verification_path),
        "asset_manifest_schema": "fixture-explicit-no-formal-assets",
        "native_core_api": "zkdl_native_gat_bridge_v1",
        "independent_verification": "1",
        "fixture_mode": "1",
    }


def audit_run(root: Path, run_dir: Path, *, expected_dataset: str | None = None,
              expected_backend: str | None = None, fixture_mode: bool = False) -> dict[str, str]:
    """Audit one runner output directory and return collector-ready fields."""
    if fixture_mode:
        return _audit_fixture_run(root, run_dir, expected_dataset=expected_dataset,
                                  expected_backend=expected_backend)
    run_dir = run_dir.resolve()
    try:
        run_dir.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactError(f"run directory escapes baseline root: {run_dir}") from exc
    record_path = _regular(run_dir / "run.record", "run record")
    proof_path = _regular(run_dir / "proof.bin", "proof artifact")
    envelope_path = _regular(run_dir / "proof.meta", "proof envelope")
    timing_path = _regular(run_dir / "proof.timing", "proof timing diagnostics")
    statement_path = _regular(run_dir / "statement.txt", "public statement")
    verification_path = _regular(run_dir / "verification.result", "independent verification result")
    record = read_kv(record_path)
    envelope = read_kv(envelope_path)
    timing = read_kv(timing_path)
    statement = read_kv(statement_path)
    verification = read_kv(verification_path)

    dataset = _required(record, "dataset", record_path)
    backend = _required(record, "backend", record_path)
    if dataset not in DATASETS:
        raise ArtifactError(f"non-canonical dataset in {record_path}: {dataset}")
    if backend not in BACKENDS:
        raise ArtifactError(f"non-canonical backend in {record_path}: {backend}")
    if expected_dataset is not None and dataset != expected_dataset:
        raise ArtifactError(f"dataset mismatch in {record_path}")
    if expected_backend is not None and backend != expected_backend:
        raise ArtifactError(f"backend mismatch in {record_path}")

    if record.get("schema") != "zkDL_gat_run_record_v2":
        raise ArtifactError(f"run record schema mismatch: {record_path}")
    if record.get("verification_status") != "VERIFIED":
        raise ArtifactError(f"run is not VERIFIED: {record_path}")
    if envelope.get("schema_id") != "zkDL_gat_proof_envelope_v1" or envelope.get("status") != "PROVED":
        raise ArtifactError(f"proof envelope is not PROVED: {envelope_path}")
    _validate_prove_timing_diagnostics(timing, envelope, timing_path)
    if (record.get("commitment_time_semantics") != "native_tensor_commitments" or
            envelope.get("commitment_time_semantics") != "native_tensor_commitments" or
            record.get("commitment_time_semantics") != envelope.get("commitment_time_semantics")):
        raise ArtifactError("record/envelope commitment time semantics mismatch")
    if envelope.get("native_core_api") != "zkdl_native_gat_bridge_v1":
        raise ArtifactError(f"proof envelope lacks the native zkDL bridge identity: {envelope_path}")
    if envelope.get("backend") != backend or envelope.get("proof_scope") != "full_gat_forward":
        raise ArtifactError(f"proof envelope backend/scope mismatch: {envelope_path}")
    if envelope.get("statement_id") != record.get("statement_id"):
        raise ArtifactError("record/envelope statement identity mismatch")
    if envelope.get("proof_id") != record.get("proof_id"):
        raise ArtifactError("record/envelope proof identity mismatch")
    if verification.get("status") != "VERIFIED":
        raise ArtifactError(f"independent verifier did not return VERIFIED: {verification_path}")
    if verification.get("statement_id") != envelope.get("statement_id"):
        raise ArtifactError("verification result statement identity mismatch")
    if verification.get("proof_id") != envelope.get("proof_id"):
        raise ArtifactError("verification result proof identity mismatch")

    if statement.get("schema") != "zkDL_gat_statement_v1":
        raise ArtifactError(f"public statement schema mismatch: {statement_path}")
    _digest(statement.get("sorted_lookup_descriptor_digest", ""), "sorted_lookup_descriptor_digest")
    _digest(statement.get("sorted_lookup_plan_digest", ""), "sorted_lookup_plan_digest")
    try:
        if int(statement.get("sorted_lookup_query_count", "0")) < 1:
            raise ArtifactError("public statement has no sorted-lookup query families")
    except ValueError as exc:
        raise ArtifactError("public statement sorted_lookup_query_count is not an integer") from exc
    if statement.get("statement_id") != record.get("statement_id"):
        raise ArtifactError("record/statement identity mismatch")
    if statement.get("dataset") != dataset:
        raise ArtifactError("statement dataset mismatch")
    if statement.get("proof_scope") != "full_gat_forward" or statement.get("verifier_reads_witness") != "0":
        raise ArtifactError("statement violates the independent verifier boundary")
    if statement.get("privacy_policy") != "model_and_trace_hidden":
        raise ArtifactError("statement privacy policy is not the canonical hidden-model/trace profile")

    with proof_path.open("rb") as stream:
        native_header = stream.read(8)
    if native_header != b"GATR" + (7).to_bytes(4, "little"):
        raise ArtifactError("formal proof is not the segmented matmul/public-output GATR v7 protocol")
    proof_digest = sha256_file(proof_path)
    if proof_digest != envelope.get("proof_id") or proof_digest != record.get("proof_id"):
        raise ArtifactError("proof_id is not the SHA-256 digest of proof.bin")
    try:
        proof_size = int(envelope.get("proof_size_bytes", "0"))
        record_size = int(record.get("proof_size_bytes", "0"))
    except ValueError as exc:
        raise ArtifactError("proof size is not an integer") from exc
    if proof_size <= 0 or proof_size != proof_path.stat().st_size or record_size != proof_size:
        raise ArtifactError("proof artifact size is not consistently bound")
    for field in ("commitment_time_seconds", "prove_time_seconds", "verify_time_seconds"):
        metric_values: list[float] = []
        for source_name, source in (("record", record), ("envelope", envelope)):
            if field not in source:
                raise ArtifactError(f"{source_name} is missing {field}")
            try:
                value = float(source[field])
            except (TypeError, ValueError) as exc:
                raise ArtifactError(f"{source_name} has invalid {field}") from exc
            if not math.isfinite(value) or value < 0.0:
                raise ArtifactError(f"{source_name} has non-finite or negative {field}")
            metric_values.append(value)
        if metric_values[0] != metric_values[1]:
            raise ArtifactError(f"record/envelope mismatch for {field}")
    if record.get("proof_scope") != envelope.get("proof_scope"):
        raise ArtifactError("record/envelope proof scope mismatch")
    for field in ("verifier_reads_witness", "silent_fallback"):
        if envelope.get(field) != "0":
            raise ArtifactError(f"proof envelope violates {field}")
        if record.get(field) != envelope.get(field):
            raise ArtifactError(f"record/envelope mismatch for {field}")
    if envelope.get("output_bound") != "1":
        raise ArtifactError("proof envelope violates output_bound")
    if record.get("output_bound") != envelope.get("output_bound"):
        raise ArtifactError("record/envelope mismatch for output_bound")
    if backend == "gpu":
        if envelope.get("gpu_kernel_executed") != "1":
            raise ArtifactError("GPU result has no native kernel execution evidence")
    elif envelope.get("gpu_kernel_executed") != "0":
        raise ArtifactError("CPU result claims GPU kernel execution")
    if record.get("gpu_kernel_executed") != envelope.get("gpu_kernel_executed"):
        raise ArtifactError("record/envelope mismatch for gpu_kernel_executed")

    manifest, asset_digest = _audit_assets(root, run_dir, statement)
    for field in (
        "static_table_descriptor_digest", "scale_chain_digest",
        "integer_bound_ledger_id", "integer_bound_ledger_digest",
        "integer_bound_ledger_descriptor_digest",
    ):
        if record.get(field) != statement.get(field) or envelope.get(field) != statement.get(field):
            raise ArtifactError(f"resource identity mismatch for {field}")
    _digest(statement.get("statement_id", ""), "statement_id")
    _digest(statement.get("verifier_registry_digest", ""), "verifier_registry_digest")
    _digest(statement.get("asset_manifest_digest", ""), "asset_manifest_digest")
    public_output = _local(run_dir, statement.get("public_output_path", ""), "public output path")
    if public_output != (run_dir / "public_Y.txt").resolve():
        raise ArtifactError("public output path is not the run-local public_Y.txt")
    _regular(public_output, "public output")
    if public_output.stat().st_size == 0:
        raise ArtifactError("public output is empty")

    return {
        **record,
        "run_dir": run_dir.relative_to(root.resolve()).as_posix(),
        "proof_digest": proof_digest,
        "proof_artifact_path": proof_path.relative_to(root.resolve()).as_posix(),
        "proof_meta_digest": sha256_file(envelope_path),
        "statement_digest": sha256_file(statement_path),
        "verification_result_digest": sha256_file(verification_path),
        "asset_manifest_digest": asset_digest,
        "asset_manifest_schema": str(manifest.get("schema")),
        "build_manifest_digest": str(manifest.get("build_manifest_digest")),
        "source_digest": str(manifest.get("source_digest")),
        "native_core_api": envelope["native_core_api"],
        "independent_verification": "1",
    }


def require_formal_five(rows: list[dict[str, str]]) -> None:
    """Require one independently verified, fully audited result per dataset."""
    if len(rows) != len(DATASETS):
        raise ArtifactError(f"formal result requires exactly {len(DATASETS)} datasets; got {len(rows)}")
    seen: set[str] = set()
    for row in rows:
        dataset = row.get("dataset", "")
        if dataset not in DATASETS:
            raise ArtifactError(f"formal result has non-canonical dataset: {dataset}")
        if dataset in seen:
            raise ArtifactError(f"duplicate formal result dataset: {dataset}")
        seen.add(dataset)
        if row.get("independent_verification") != "1":
            raise ArtifactError(f"formal result lacks independent verification: {dataset}")
        if row.get("silent_fallback") != "0":
            raise ArtifactError(f"formal result claims silent fallback: {dataset}")
        if row.get("commitment_time_semantics") != "native_tensor_commitments":
            raise ArtifactError(f"formal result has unsupported Commitment Time semantics: {dataset}")
        try:
            proof_size = int(row.get("proof_size_bytes", "0"))
            for field in ("commitment_time_seconds", "prove_time_seconds", "verify_time_seconds"):
                value = float(row[field])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(field)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(f"formal result has invalid metric fields: {dataset}") from exc
        if proof_size <= 0:
            raise ArtifactError(f"formal result has invalid proof size: {dataset}")
    if seen != set(DATASETS):
        missing = sorted(set(DATASETS) - seen)
        raise ArtifactError(f"formal result set is incomplete; missing={missing}")
