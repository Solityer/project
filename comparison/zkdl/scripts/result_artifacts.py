#!/usr/bin/env python3
"""Collect result metadata after native proof and independent verification.

This module is intentionally metadata-only with respect to the proof system:
it never proves, verifies, creates, or repairs an artifact.  It checks the
native run record, proof envelope, public statement, proof path, and the
independent-verification result before collecting five-dataset metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any



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
        "schema_id", "status", "statement_id", "commitment_time_seconds", "prove_time_seconds",
        "proof_auxiliary_construction_seconds", "proof_auxiliary_construction_scope",
        "sorted_lookup_protocol_construction_seconds", "relation_context_prove_seconds", "relation_prove_wall_seconds",
        "relation_commitment_time_seconds", "prove_provider_file_io_excluded_seconds", "proof_serialization_seconds",
    }
    if set(timing) != expected_fields or timing.get("schema_id") != "zkDL_gat_proof_timing_v1" or timing.get("status") != "PROVED":
        raise ArtifactError(f"proof timing sidecar schema/status mismatch: {path}")
    if timing.get("proof_auxiliary_construction_scope") != expected_scope:
        raise ArtifactError(f"proof timing diagnostic scope mismatch: {path}")
    if timing.get("statement_id") != envelope.get("statement_id"):
        raise ArtifactError(f"proof timing sidecar is not bound to the statement: {path}")
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


def _gpu_offload_evidence(record: dict[str, str], envelope: dict[str, str], verification: dict[str, str],
                          backend: str, expected_stage: str | None) -> dict[str, str]:
    stage = envelope.get("gpu_offload_stage", "none")
    if record.get("gpu_offload_stage", "none") != stage:
        raise ArtifactError("record/envelope mismatch for gpu_offload_stage")
    if expected_stage is not None and stage != expected_stage:
        raise ArtifactError(f"proof offload stage mismatch: expected {expected_stage}, got {stage}")
    if stage == "none":
        if backend == "gpu":
            if envelope.get("gpu_kernel_executed") != "1":
                raise ArtifactError("GPU result has no native kernel execution evidence")
        elif envelope.get("gpu_kernel_executed") != "0":
            raise ArtifactError("CPU result claims GPU kernel execution")
        if record.get("gpu_kernel_executed") != envelope.get("gpu_kernel_executed"):
            raise ArtifactError("record/envelope mismatch for gpu_kernel_executed")
        if backend == "cpu":
            for field in ("gpu_forward_kernel_executed", "gpu_proof_kernel_executed"):
                if field in envelope and envelope[field] != "0":
                    raise ArtifactError(f"CPU result claims GPU execution in {field}")
                if field in record and record[field] != envelope.get(field):
                    raise ArtifactError(f"record/envelope mismatch for {field}")
            for field in ("gpu_proof_kernel_count", "gpu_proof_peak_allocated_bytes"):
                if field in envelope and envelope[field] != "0":
                    raise ArtifactError(f"CPU result claims GPU allocation in {field}")
                if field in record and record[field] != envelope.get(field):
                    raise ArtifactError(f"record/envelope mismatch for {field}")
        return {}
    if stage != "hyrax_ipa_fold" or backend != "cpu":
        raise ArtifactError("unsupported CPU/GPU proof offload combination")
    fields = ("gpu_forward_kernel_executed", "gpu_proof_kernel_executed", "gpu_kernel_executed",
              "gpu_proof_kernel_count", "gpu_proof_peak_allocated_bytes", "gpu_device_name",
              "gpu_device_ordinal", "gpu_device_uuid")
    for field in fields:
        if not envelope.get(field) or record.get(field) != envelope.get(field):
            raise ArtifactError(f"record/envelope mismatch or missing {field}")
    for field in ("verifier_backend", "verify_gpu_kernel_executed"):
        if not envelope.get(field) or record.get(field) != envelope.get(field):
            raise ArtifactError(f"record/envelope mismatch or missing {field}")
    if (envelope["gpu_forward_kernel_executed"] != "0" or
            envelope["gpu_proof_kernel_executed"] != "1" or envelope["gpu_kernel_executed"] != "1"):
        raise ArtifactError("Hyrax IPA offload lacks consistent native GPU execution flags")
    for field in ("gpu_proof_kernel_count", "gpu_proof_peak_allocated_bytes"):
        if (not envelope[field].isascii() or not envelope[field].isdecimal() or int(envelope[field]) <= 0 or
                int(envelope[field]) > (1 << 64) - 1):
            raise ArtifactError(f"Hyrax IPA offload has invalid {field}")
    if (not envelope["gpu_device_name"].strip() or not envelope["gpu_device_uuid"].strip() or
            not envelope["gpu_device_ordinal"].isascii() or not envelope["gpu_device_ordinal"].isdecimal() or
            int(envelope["gpu_device_ordinal"]) > (1 << 31) - 1):
        raise ArtifactError("Hyrax IPA offload lacks a valid native CUDA device identity")
    if (verification.get("verifier_backend") != "cpu" or
            verification.get("verify_gpu_kernel_executed") != "0"):
        raise ArtifactError("independent verifier did not record CPU-only verification")
    if record["verifier_backend"] != verification["verifier_backend"]:
        raise ArtifactError("run record/verification result mismatch for verifier_backend")
    if record["verify_gpu_kernel_executed"] != verification["verify_gpu_kernel_executed"]:
        raise ArtifactError("run record/verification result mismatch for verify_gpu_kernel_executed")
    return {"gpu_offload_stage": stage, **{field: envelope[field] for field in fields},
            "verifier_backend": verification["verifier_backend"],
            "verify_gpu_kernel_executed": verification["verify_gpu_kernel_executed"]}


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
    proof_size = proof_path.stat().st_size
    with proof_path.open("rb") as proof_stream:
        magic = proof_stream.read(4)
    if proof_size < 4 or magic != b"GATR":
        raise ArtifactError(f"fixture proof is not a native GATR artifact: {proof_path}")
    evidence = evidence_path.read_text(encoding="utf-8")
    if "independent_verify=PASS" not in evidence:
        raise ArtifactError("fixture evidence does not report independent verification")
    expected_statement = _required(statement, "statement_id", statement_path)
    for values, label in ((record, "record"), (envelope, "envelope"),
                          (verification, "verification")):
        if values.get("statement_id") != expected_statement:
            raise ArtifactError(f"fixture {label}/statement identity mismatch")
    try:
        declared_size = int(record.get("proof_size_bytes", "0"))
        envelope_size = int(envelope.get("proof_size_bytes", "0"))
    except ValueError as exc:
        raise ArtifactError("fixture proof size is not an integer") from exc
    if declared_size != proof_size or envelope_size != proof_size:
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
        "proof_artifact_path": proof_path.relative_to(root.resolve()).as_posix(),
        "native_core_api": "zkdl_native_gat_bridge_v1",
        "independent_verification": "1",
        "fixture_mode": "1",
    }


def audit_run(root: Path, run_dir: Path, *, expected_dataset: str | None = None,
              expected_backend: str | None = None, expected_gpu_offload_stage: str | None = None,
              fixture_mode: bool = False) -> dict[str, str]:
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
    if verification.get("status") != "VERIFIED":
        raise ArtifactError(f"independent verifier did not return VERIFIED: {verification_path}")
    if verification.get("statement_id") != envelope.get("statement_id"):
        raise ArtifactError("verification result statement identity mismatch")

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
    offload = _gpu_offload_evidence(record, envelope, verification, backend, expected_gpu_offload_stage)

    for field in (
        "static_table_descriptor_digest", "scale_chain_digest",
        "integer_bound_ledger_id", "integer_bound_ledger_digest",
        "integer_bound_ledger_descriptor_digest",
    ):
        if record.get(field) != statement.get(field) or envelope.get(field) != statement.get(field):
            raise ArtifactError(f"resource identity mismatch for {field}")
    _digest(statement.get("statement_id", ""), "statement_id")
    _digest(statement.get("verifier_registry_digest", ""), "verifier_registry_digest")
    public_output = _local(run_dir, statement.get("public_output_path", ""), "public output path")
    if public_output != (run_dir / "public_Y.txt").resolve():
        raise ArtifactError("public output path is not the run-local public_Y.txt")
    _regular(public_output, "public output")
    if public_output.stat().st_size == 0:
        raise ArtifactError("public output is empty")

    return {
        **record,
        "run_dir": run_dir.relative_to(root.resolve()).as_posix(),
        "proof_artifact_path": proof_path.relative_to(root.resolve()).as_posix(),
        "native_core_api": envelope["native_core_api"],
        "independent_verification": "1",
        **offload,
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
