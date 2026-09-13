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
import re
from pathlib import Path
from typing import Any


DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
BACKENDS = ("cpu", "gpu")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ArtifactError(ValueError):
    """A production artifact failed a fail-closed audit."""


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


def _required(values: dict[str, str], key: str, path: Path) -> str:
    value = values.get(key)
    if value is None or value == "":
        raise ArtifactError(f"missing field {key}: {path}")
    return value


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
        "model_registry_digest", "model_blinder_sidecar_path",
        "model_blinder_sidecar_digest",
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
    for key in ("G", "H", "U"):
        item = srs.get(key)
        if not isinstance(item, dict):
            raise ArtifactError(f"SRS manifest missing {key} frame")
        raw = item.get("path")
        declared = item.get("sha256")
        if not isinstance(raw, str) or not isinstance(declared, str) or not HEX64.fullmatch(declared):
            raise ArtifactError(f"SRS manifest has malformed {key} frame identity")
        frame = _local(srs_path, raw, f"SRS {key} frame")
        _regular(frame, f"SRS {key} frame")
        relative = frame.relative_to(root).as_posix()
        if relative not in digests or digests[relative] != declared:
            raise ArtifactError(f"asset manifest does not authenticate SRS {key} frame")

    registry_path = _local(root, manifest["model_registry_path"], "model registry path")
    _regular(registry_path, "model registry")
    if sha256_file(registry_path) != manifest["model_registry_digest"]:
        raise ArtifactError("model registry digest does not match asset manifest")
    _digest(manifest["model_registry_digest"], "model_registry_digest")
    sidecar_path = _local(root, manifest["model_blinder_sidecar_path"], "model blinder sidecar path")
    _regular(sidecar_path, "model blinder sidecar")
    if sha256_file(sidecar_path) != manifest["model_blinder_sidecar_digest"]:
        raise ArtifactError("model blinder sidecar digest does not match asset manifest")
    _digest(manifest["model_blinder_sidecar_digest"], "model_blinder_sidecar_digest")
    if sidecar_path.stat().st_mode & 0o077:
        raise ArtifactError("model blinder sidecar is not owner-only")


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
    digests = _manifest_files(root, manifest, manifest_path)
    _resource_manifest(root, manifest, manifest_path, digests)

    # These identities must be backed by the exact authenticated files listed
    # in the manifest.  No formal result may downgrade them to UNAVAILABLE.
    for field in (
        "scale_chain_digest", "static_table_descriptor_digest",
        "integer_bound_ledger_descriptor_digest",
    ):
        value = manifest.get(field)
        _digest(value, field)
        if field == "scale_chain_digest":
            relative = "configs/quantization_scale_chain.cfg"
        else:
            descriptor_key = "static_table_descriptor" if field.startswith("static") else "integer_bound_ledger"
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
    if statement.get("scale_chain_digest") != manifest.get("scale_chain_digest"):
        raise ArtifactError("statement scale-chain digest mismatch")
    if statement.get("proof_scope") != "full_gat_forward":
        raise ArtifactError("statement proof scope is not full_gat_forward")
    if statement.get("verifier_reads_witness") != "0":
        raise ArtifactError("statement verifier reads witness")
    return manifest, asset_digest


def audit_run(root: Path, run_dir: Path, *, expected_dataset: str | None = None,
              expected_backend: str | None = None) -> dict[str, str]:
    """Audit one runner output directory and return collector-ready fields."""
    run_dir = run_dir.resolve()
    try:
        run_dir.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactError(f"run directory escapes baseline root: {run_dir}") from exc
    record_path = _regular(run_dir / "run.record", "run record")
    proof_path = _regular(run_dir / "proof.bin", "proof artifact")
    envelope_path = _regular(run_dir / "proof.meta", "proof envelope")
    statement_path = _regular(run_dir / "statement.txt", "public statement")
    verification_path = _regular(run_dir / "verification.result", "independent verification result")
    record = read_kv(record_path)
    envelope = read_kv(envelope_path)
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

    if record.get("schema") != "zkDL_gat_run_record_v1":
        raise ArtifactError(f"run record schema mismatch: {record_path}")
    if record.get("verification_status") != "VERIFIED":
        raise ArtifactError(f"run is not VERIFIED: {record_path}")
    if envelope.get("schema_id") != "zkDL_gat_proof_envelope_v1" or envelope.get("status") != "PROVED":
        raise ArtifactError(f"proof envelope is not PROVED: {envelope_path}")
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
    if statement.get("statement_id") != record.get("statement_id"):
        raise ArtifactError("record/statement identity mismatch")
    if statement.get("dataset") != dataset:
        raise ArtifactError("statement dataset mismatch")
    if statement.get("proof_scope") != "full_gat_forward" or statement.get("verifier_reads_witness") != "0":
        raise ArtifactError("statement violates the independent verifier boundary")
    if statement.get("privacy_policy") != "model_and_trace_hidden":
        raise ArtifactError("statement privacy policy is not the canonical hidden-model/trace profile")

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
        "native_core_api": envelope["native_core_api"],
        "independent_verification": "1",
    }


def expected_cells() -> set[tuple[str, str]]:
    return {(dataset, backend) for dataset in DATASETS for backend in BACKENDS}


def require_exact_matrix(rows: list[dict[str, str]]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        cell = (row.get("dataset", ""), row.get("backend", ""))
        if cell in seen:
            raise ArtifactError(f"duplicate formal result cell: {cell[0]}/{cell[1]}")
        seen.add(cell)
    expected = expected_cells()
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ArtifactError(f"exact formal matrix required; missing={missing} extra={extra}")
    for dataset in DATASETS:
        cpu = next(row for row in rows if row["dataset"] == dataset and row["backend"] == "cpu")
        gpu = next(row for row in rows if row["dataset"] == dataset and row["backend"] == "gpu")
        for field in ("statement_id", "scale_chain_digest", "static_table_descriptor_digest",
                      "integer_bound_ledger_id", "integer_bound_ledger_digest",
                      "integer_bound_ledger_descriptor_digest"):
            if cpu.get(field) != gpu.get(field):
                raise ArtifactError(f"CPU/GPU public identity mismatch for {dataset}: {field}")
