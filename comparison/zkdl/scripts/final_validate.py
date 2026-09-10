#!/usr/bin/env python3
"""Fail-closed validator for the formal CPU/GPU result matrix."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"final-validate: {message}")


def read_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            fail(f"malformed record line: {path}")
        key, value = line.split("=", 1)
        if key in values:
            fail(f"duplicate record key {key}: {path}")
        values[key] = value
    return values


IDENTITY_DIGESTS = {
    "static_table_descriptor_digest", "scale_chain_digest", "integer_bound_ledger_digest",
    "integer_bound_ledger_descriptor_digest",
}


def validate_identity_fields(row: dict[str, str], path: Path, manifest: dict[str, object]) -> None:
    for field in IDENTITY_DIGESTS:
        value = row.get(field, "")
        if not value or (value != "UNAVAILABLE" and not re.fullmatch(r"[0-9a-f]{64}", value)):
            fail(f"invalid identity field {field}: {path}")
    if not row.get("integer_bound_ledger_id"):
        fail(f"missing integer-bound ledger id: {path}")
    # A formal VERIFIED run must carry authenticated relation identities.  The
    # empty-file manifests are test fixtures only; real manifests have entries
    # and may not downgrade these bindings to UNAVAILABLE.
    if manifest.get("files") and any(row[field] == "UNAVAILABLE" for field in IDENTITY_DIGESTS):
        fail(f"VERIFIED run has unavailable relation identity: {path}")
    for field in ("scale_chain_digest", "static_table_descriptor_digest", "integer_bound_ledger_descriptor_digest"):
        manifest_value = manifest.get(field)
        if manifest_value and manifest_value != "UNAVAILABLE" and row[field] != manifest_value:
            fail(f"run/asset manifest identity mismatch for {field}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--require-pairs", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    runs = (root / args.runs).resolve()
    if not runs.is_relative_to(root) or not runs.is_dir():
        fail("runs directory must be baseline-local")
    records = sorted(runs.rglob("run.record"))
    if not records:
        fail("no production run records")
    rows = [read_kv(path) for path in records]
    for path, row in zip(records, rows):
        if row.get("schema") != "zkDL_gat_run_record_v1": fail(f"wrong schema: {path}")
        if row.get("verification_status") != "VERIFIED": fail(f"not verified: {path}")
        if row.get("proof_scope") != "full_gat_forward": fail(f"proof scope mismatch: {path}")
        if row.get("verifier_reads_witness") != "0": fail(f"verifier reads witness: {path}")
        if row.get("silent_fallback") != "0": fail(f"silent fallback: {path}")
        if row.get("output_bound") != "1": fail(f"public output not bound: {path}")
        if row.get("backend") == "gpu" and row.get("gpu_kernel_executed") != "1":
            fail(f"GPU run has no executed CUDA kernel: {path}")
        try:
            if int(row.get("proof_size_bytes", "0")) <= 0: fail(f"empty proof: {path}")
            if int(row.get("commitment_size_bytes", "0")) <= 0: fail(f"empty commitment: {path}")
        except ValueError:
            fail(f"invalid size field: {path}")
        manifest = path.parent / "asset_manifest.json"
        if not manifest.is_file(): fail(f"asset manifest missing beside run: {manifest}")
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(f"invalid asset manifest {manifest}: {exc}")
        if payload.get("schema") != "zkDL_gat_asset_manifest_v1": fail(f"asset schema mismatch: {manifest}")
        for field in ("scale_chain_digest", "static_table_descriptor", "static_table_descriptor_digest",
                      "integer_bound_ledger", "integer_bound_ledger_descriptor_digest"):
            if field not in payload: fail(f"asset identity field missing: {field}: {manifest}")
        validate_identity_fields(row, path, payload)
    if args.require_pairs:
        seen = {(row.get("dataset"), row.get("backend")) for row in rows}
        datasets = {dataset for dataset, _ in seen}
        for dataset in datasets:
            if (dataset, "cpu") not in seen or (dataset, "gpu") not in seen:
                fail(f"missing CPU/GPU pair for {dataset}")
    print(f"final_validation=PASS records={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
