#!/usr/bin/env python3
"""Keep zkGPT's shared capacity receipt aligned with the strict root schema."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_formal  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="zkgpt-capacity-profile-") as temporary:
        session = Path(temporary) / "runs" / "formal" / ("a" * 32)
        session.mkdir(parents=True)
        item = {
            "status": "READY",
            "proof_path": "formal_native_masked_lasso_hyrax+native_masked_layered_gkr",
            "full_relation_bound": True,
            "estimated_native_scratch_bytes": 2,
            "disk_requirement_bytes_upper_bound": 3,
            "artifact_disk_bytes_upper_bound": 1,
            "estimated_peak_host_memory_bytes": 5,
            "proof_artifact_wire_bytes_upper_bound": 7,
            "context_payload_bytes_bound": 11,
        }
        profile = run_formal._capacity_profile(
            "a" * 32, session,
            {"capacity": {dataset: dict(item) for dataset in run_formal.DATASETS}},
        )
        expected_profile_keys = {
            "schema", "baseline", "session_id", "fs_device_id",
            "preexisting_baseline_artifact_bytes", "finalization_disk_bytes_upper_bound",
            "bound_methods", "datasets", "memory_accounting_kind", "device",
        }
        if set(profile) != expected_profile_keys:
            raise AssertionError("zkGPT shared capacity receipt differs from root profile-v2 fields")
        if (profile["schema"] != "comparison-baseline-capacity-profile-v2" or
                profile["memory_accounting_kind"] != "host-only" or profile["device"] is not None):
            raise AssertionError("zkGPT shared capacity receipt has the wrong CPU memory identity")
        expected_cell_keys = {
            "additional_disk_peak_bytes", "cell_retained_artifact_bytes_upper_bound",
            "host_peak_bytes", "device_peak_bytes", "phase_coverage",
        }
        if set(profile["datasets"]) != set(run_formal.DATASETS):
            raise AssertionError("zkGPT capacity receipt does not name exactly five datasets")
        for cell in profile["datasets"].values():
            if set(cell) != expected_cell_keys or cell["device_peak_bytes"] != 0:
                raise AssertionError("zkGPT capacity cell differs from root's exact five-key CPU schema")
            if set(cell["phase_coverage"]) != {"setup", "witness", "commitment", "prove", "verify"}:
                raise AssertionError("zkGPT capacity cell omits a canonical phase")
    print("shared_capacity_profile_smoke status=PASS schema=v2 datasets=5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
