#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import inspect
import json
import pathlib
import sys
import tempfile
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_manifest(runner, path: pathlib.Path, row: dict[str, object]) -> None:
    row["run_manifest_hash"] = runner.manifest_hash(row)
    (path / "run_manifest.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def valid_cpu_row() -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": "cora",
        "backend": "cpu",
        "source_tree_hash": "source",
        "binary_hash": "binary",
        "runner_hash": "runner",
        "status": "VERIFY_OK",
        "verified": "true",
        "full_dataset": "true",
        "timeout": "false",
        "stale_result": "false",
        "exit_code": "0",
        "proof_payload_audit_status": "passed",
        "reference_consistency_passed": "true",
        "tamper_tests_passed": "true",
        "security_budget_passed": "true",
        "private_plain_value_count": "0",
        "verifier_reads_witness": "false",
        "timing_schema": "exclusive_commit_prove_verify_v2",
        "dynamic_commitment_object_profile_csv": "profile.csv",
        "performance_baseline_status": "pending_rebuild_current_protocol",
        "performance_evaluation_status": "informational_not_run_validity_gate",
    }
    for key in (
        "total_wall_clock_ms",
        "formal_time_sum_ms",
        "non_formal_overhead_ms",
        "context_build_ms",
        "reference_forward_ms",
        "trace_generation_ms",
        "quotient_build_ms",
        "quotient_commitment_ms",
        "domain_opening_ms",
        "external_opening_ms",
        "payload_audit_ms",
        "reference_consistency_ms",
        "tamper_tests_ms",
        "export_manifest_validation_ms",
        "runner_overhead_ms",
        "profile_logging_overhead_ms",
        "official_runner_wall_clock_ms",
    ):
        row[key] = "1400000000.000" if key == "total_wall_clock_ms" else "0.000"
    row["commitment_time_ms"] = "300000000.000"
    row["prove_time_ms"] = "900000000.000"
    row["verify_time_ms"] = "100000000.000"
    row["formal_time_sum_ms"] = "1300000000.000"
    row["non_formal_overhead_ms"] = "100000000.000"
    return row


def main() -> int:
    runner = load_module("secure_runner", ROOT / "scripts" / "run_secure_main_experiments.py")
    exporter = load_module("secure_exporter", ROOT / "scripts" / "export_paper_secure.py")

    execute_source = inspect.getsource(runner.execute_row)
    assert "timeout=timeout_seconds" in execute_source
    assert "row_hard_timeout" not in execute_source
    assert "compute_budget" not in execute_source

    with tempfile.TemporaryDirectory(prefix="gatzk-runner-test-") as temp_dir:
        temp = pathlib.Path(temp_dir)
        (temp / "profile.csv").write_text("label,total_commitment_ms\nP,1\n", encoding="utf-8")
        row = valid_cpu_row()
        write_manifest(runner, temp, row)
        assert runner.valid_manifest(temp, "cora", "cpu", "source", "binary", "runner")

        stale_timing_row = dict(row)
        stale_timing_row.pop("timing_schema")
        write_manifest(runner, temp, stale_timing_row)
        assert not runner.valid_manifest(temp, "cora", "cpu", "source", "binary", "runner")

        legacy_timing_row = dict(row)
        legacy_timing_row["commitment_ms"] = legacy_timing_row.pop(
            "commitment_time_ms"
        )
        legacy_timing_row["prove_ms"] = legacy_timing_row.pop("prove_time_ms")
        legacy_timing_row["verify_ms"] = legacy_timing_row.pop("verify_time_ms")
        write_manifest(runner, temp, legacy_timing_row)
        assert not runner.valid_manifest(
            temp, "cora", "cpu", "source", "binary", "runner"
        )
        write_manifest(runner, temp, row)

        row["performance_baseline_status"] = "deliberately_not_a_resume_gate"
        row["performance_evaluation_status"] = "deliberately_not_a_resume_gate"
        write_manifest(runner, temp, row)
        assert runner.valid_manifest(temp, "cora", "cpu", "source", "binary", "runner")

        row["security_budget_passed"] = "false"
        write_manifest(runner, temp, row)
        assert not runner.valid_manifest(temp, "cora", "cpu", "source", "binary", "runner")

        start = time.perf_counter()
        code = runner.run_command(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            temp / "timeout.log",
            timeout=1,
        )
        elapsed = time.perf_counter() - start
        assert code == 124
        assert 0.8 <= elapsed < 1.8

    legacy_fields = {
        "proof_core_total_ms",
        "target_2x_proof_core_total_ms",
        "near_2_2x_proof_core_total_ms",
        "budget_status",
        "row_hard_timeout_ms",
        "commitment_ms",
        "prove_ms",
        "verify_ms",
        "cold_end_to_end_wall_clock_ms",
        "warm_end_to_end_wall_clock_ms",
        "amortized_preprocessing_ms",
        "total_cost_for_fresh_reproduction_ms",
        "is_cold_run",
    }
    assert not legacy_fields.intersection(exporter.TIMING_FIELDS)
    assert not legacy_fields.intersection(exporter.OUTPUT_FIELDS)
    assert exporter.SECURE_FIELDS["opening_mode"] == "point_grouped_batch_opening"
    assert exporter.SECURE_FIELDS["security_budget_passed"] == "true"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
