#!/usr/bin/env python3
"""Collect only independently verified formal ezkl GAT run manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"


def main() -> int:
    rows: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for path in sorted(RUNS.glob("*/*/run_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = {"dataset": payload.get("dataset"), "device": payload.get("device")}
        if payload.get("verify_passed") is not True or payload.get("status") != "verified":
            excluded.append(
                {**identity, "status": payload.get("status"), "manifest": str(path.relative_to(ROOT))}
            )
            continue
        timings = payload.get("timings_ms", {})
        rows.append(
            {
                **identity,
                "route_id": payload.get("route_id"),
                "status": "verified",
                "setup_ms": timings.get("setup"),
                "gen_witness_ms": timings.get("gen_witness"),
                "prove_ms": timings.get("prove"),
                "verify_ms": timings.get("verify"),
                "proof_size_bytes": payload.get("proof_size_bytes"),
                "proof_sha256": payload.get("proof_sha256"),
                "public_statement_sha256": payload.get("public_statement_sha256"),
                "compiled_circuit_sha256": payload.get("compiled_circuit_sha256"),
                "vk_sha256": payload.get("vk_sha256"),
                "runtime_device_marker": payload.get("runtime_device_marker"),
                "verification_runtime_marker": payload.get("verification_runtime_marker"),
                "ezkl_security_profile_id": payload.get("ezkl_security_profile_id"),
                "quant_cfg_id": payload.get("quant_cfg_id"),
                "static_tables_id": payload.get("static_tables_id"),
                "backend_aux_tables_id": payload.get("backend_aux_tables_id"),
                "source_srs_sha256": payload.get("source_srs_sha256"),
                "source_srs_k": payload.get("source_srs_k"),
                "downsized_srs_k": payload.get("downsized_srs_k"),
                "proving_srs_sha256": payload.get("proving_srs_sha256"),
                "verification_srs_sha256": payload.get("verification_srs_sha256"),
                "gpu_backend_execution": payload.get("gpu_backend_execution"),
                "gpu_kernel_telemetry": payload.get("gpu_kernel_telemetry"),
                "manifest": str(path.relative_to(ROOT)),
            }
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    document = {
        "result_kind": "ezkl_formal_gat_results_v1",
        "verified_runs": rows,
        "non_verified_runs_excluded": excluded,
    }
    (RESULTS / "formal_results.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    columns = list(rows[0]) if rows else ["dataset", "device", "status"]
    with (RESULTS / "formal_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"verified_runs": len(rows), "excluded_non_verified": len(excluded)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
