#!/usr/bin/env python3
"""Focused regression checks for repetition profiles and source identity."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import pathlib
import shutil
import tempfile
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module("mechanism_runner_regression", ROOT / "scripts/run_mechanism_ablation.py")
provenance = load_module("mechanism_provenance_regression", ROOT / "tools/build_provenance.py")


def make_cells(repetitions: int):
    spec = runner.FORMAL_CONFIGS[0]
    config_id = runner.matrix_config_id(spec)
    cells = []
    for repetition in range(1, repetitions + 1):
        for variant in runner.VARIANTS:
            original = variant == "original"
            cells.append({
                "config_id": config_id,
                "mechanism": spec["mechanism"],
                "dataset": spec["dataset"],
                "variant": variant,
                "repetition": repetition,
                "pair_status": "paired",
                "N": 2708,
                "E": 13264,
                "D": 1,
                "G": 1,
                "component_ids": [0],
                "security_policy": {"target_bits": 120, "validated": True},
                "commitment_time_ms": 10.0 + repetition if original else 20.0 + repetition,
                "prove_time_ms": 30.0 + repetition if original else 40.0 + repetition,
                "verify_time_ms": 1.0 + repetition if original else 2.0 + repetition,
                "proof_size_bytes": 1000.0 + repetition if original else 2000.0 + repetition,
            })
    return cells


def test_reverify_archive_validation() -> None:
    with tempfile.TemporaryDirectory(prefix="mechanism_reverify_archive_regression_") as temporary:
        session = pathlib.Path(temporary)
        proof_sha = "1" * 64
        statement_sha = "2" * 64
        prover_sha = "3" * 64
        verifier_sha = "4" * 64
        source_hash = "5" * 64
        diff_hash = "6" * 64
        binary_hash = "7" * 64
        build_manifest_hash = "8" * 64
        srs_hash = "9" * 64
        for name, content in (("verify.json", "{}\n"), ("verify.log", "VERIFY_OK\n"), ("audit.log", "PASS\n")):
            (session / name).write_text(content, encoding="utf-8")
        cell = {
            "config_id": "synthetic-test-cell", "repetition": 1, "variant": "original",
            "proof_sha256": proof_sha, "statement_file_sha256": statement_sha,
            "prover_result_sha256": prover_sha, "verifier_result_sha256": verifier_sha,
        }
        source = {"source_tree_hash": source_hash, "git_diff_sha256": diff_hash}
        build = {"binary_sha256": binary_hash, "source_manifest_sha256": build_manifest_hash}
        datasets = {"test": {"srs": {"manifest_sha256": srs_hash}}}
        document = {
            "session_id": "synthetic-archive-regression", "expected_cell_count": 1,
            "cells": [cell], "result_files": {},
            "provenance": {"source": source, "build": build, "datasets": datasets},
        }
        runner.atomic_write_json(session / "summary.json", document)
        (session / "cells.csv").write_text("config_id\nsynthetic-test-cell\n", encoding="utf-8")
        summary_sha = runner.sha256_file(session / "summary.json")
        runner.atomic_write_json(session / "results_manifest.json", {
            "session_id": document["session_id"], "session_status": "complete", "files": {},
            "cells_csv": {"path": str(session / "cells.csv"), "sha256": runner.sha256_file(session / "cells.csv")},
            "summary_json": {"path": str(session / "summary.json"), "sha256": summary_sha},
        })
        result_manifest_sha = runner.sha256_file(session / "results_manifest.json")
        audit_cell = {
            **cell, "status": "VERIFY_OK", "exit_code": 0,
            "receipt_path": str(session / "verify.json"),
            "receipt_sha256": runner.sha256_file(session / "verify.json"),
            "log_path": str(session / "verify.log"),
            "log_sha256": runner.sha256_file(session / "verify.log"),
        }
        audit = {
            "format": "zkgat_real_mechanism_result_audit_v1", "session_id": document["session_id"],
            "status": "PASS", "reverified_independently": True,
            "summary_sha256": summary_sha, "cell_count": 1, "verified_cells": 1,
            "cells": [audit_cell],
            "provenance_bindings": {
                "source_tree_hash": source_hash, "git_diff_sha256": diff_hash,
                "binary_sha256": binary_hash, "build_source_manifest_sha256": build_manifest_hash,
                "srs_manifest_sha256": {"test": srs_hash},
                "results_manifest_sha256": result_manifest_sha,
            },
            "reverification_attempt": {
                "log_path": str(session / "audit.log"),
                "log_sha256": runner.sha256_file(session / "audit.log"),
                "time_path": str(session / "audit.time.json"),
            },
        }
        archive_bytes = (json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        archive_sha = runner.sha256_bytes(archive_bytes)
        runner.atomic_write_bytes(session / "audit_reverify.json", archive_bytes)
        runner.atomic_write_text(session / "audit_reverify.sha256", f"{archive_sha}  audit_reverify.json\n")
        runner.atomic_write_json(session / "audit.time.json", {
            "format": "zkgat_mechanism_reverify_timing_v1", "session_id": document["session_id"],
            "status": "PASS", "audit_sha256": archive_sha,
            "log_sha256": runner.sha256_file(session / "audit.log"), "wall_time_seconds": 1.0,
        })
        time_path = session / "audit.time.json"
        runner.atomic_write_text(time_path.with_suffix(time_path.suffix + ".sha256"),
                                 f"{runner.sha256_file(time_path)}  {time_path.name}\n")
        assert runner.validate_reverify_archive(session, document)["reverified_independently"] is True

        failed_archive = dict(audit)
        failed_archive["reverified_independently"] = False
        failed_bytes = (json.dumps(failed_archive, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        runner.atomic_write_bytes(session / "audit_reverify.json", failed_bytes)
        runner.atomic_write_text(session / "audit_reverify.sha256",
                                 f"{runner.sha256_bytes(failed_bytes)}  audit_reverify.json\n")
        preserved = (session / "audit_reverify.json").read_bytes()
        try:
            runner.validate_reverify_archive(session, document)
            raise AssertionError("a receipt-only audit was accepted as a full re-verification archive")
        except ValueError:
            pass
        assert (session / "audit_reverify.json").read_bytes() == preserved


def test_shared_gpu_authorization_and_activity_sampling() -> None:
    selected = {"index": "0", "uuid": "GPU-regression", "name": "NVIDIA GB10", "driver_version": "test"}
    device_snapshot = {"selected_device_index": 0, "selected_device": selected}
    gpu_activity = {
        "sampled_utc": "2026-10-10T00:00:00Z", "gpu_uuid": selected["uuid"], "gpu_name": selected["name"],
        "gpu_utilization_percent": "72", "memory_utilization_percent": "N/A",
        "gpu_memory_used_mib": "N/A", "gpu_memory_total_mib": "N/A",
        "external_compute_processes": [{
            "pid": "4321", "process_name": "external_test.py", "gpu_uuid": selected["uuid"],
            "used_memory_mib": "512",
        }],
    }
    result_queue = [
        types.SimpleNamespace(returncode=0, stdout="GPU-regression, NVIDIA GB10, 72, N/A, N/A, N/A\n", stderr=""),
        types.SimpleNamespace(returncode=0, stdout="4321, external_test.py, GPU-regression, 512\n", stderr=""),
    ]
    original_run = runner.subprocess.run
    try:
        runner.subprocess.run = lambda *args, **kwargs: result_queue.pop(0)
        sampled = runner.cuda_activity_snapshot(device_snapshot)
    finally:
        runner.subprocess.run = original_run
    assert sampled["gpu_uuid"] == selected["uuid"]
    assert sampled["gpu_utilization_percent"] == "72"
    assert sampled["external_compute_processes"] == gpu_activity["external_compute_processes"]
    try:
        runner.gpu_sharing_provenance(device_snapshot, gpu_activity, False)
        raise AssertionError("external CUDA use was accepted without explicit sharing authorization")
    except ValueError:
        pass
    authorized = runner.gpu_sharing_provenance(device_snapshot, gpu_activity, True)
    assert authorized["mode"] == "authorized_shared_external_compute"
    assert authorized["explicit_shared_gpu_authorization"] is True
    assert authorized["initial_activity_snapshot"] == gpu_activity

    with tempfile.TemporaryDirectory(prefix="mechanism_cuda_activity_regression_") as temporary:
        path = pathlib.Path(temporary) / "cuda_activity_samples.jsonl"
        original_snapshot = runner.cuda_activity_snapshot
        try:
            runner.cuda_activity_snapshot = lambda _device: dict(gpu_activity)
            sampler = runner.CudaActivitySampler(device_snapshot, path, interval_seconds=3600)
            sampler.start(gpu_activity)
            before = sampler.sample("cell_before", {
                "config_id": "test_config", "repetition": 1, "variant": "original",
            })
            after = sampler.sample("cell_after", {
                "config_id": "test_config", "repetition": 1, "variant": "original",
            })
            receipt = sampler.stop()
        finally:
            runner.cuda_activity_snapshot = original_snapshot
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert receipt["sample_count"] == len(rows) == 4
        assert receipt["sha256"] == runner.sha256_file(path)
        assert rows[0]["sample_kind"] == "session_start"
        assert rows[1] == before and rows[2] == after
        assert rows[-1]["sample_kind"] == "session_end"


def main() -> None:
    test_shared_gpu_authorization_and_activity_sampling()
    one_schedule = runner.formal_schedule(1)
    three_schedule = runner.formal_schedule(3)
    assert len(one_schedule) == 11 and {rep for _, rep in one_schedule} == {1}
    assert len(three_schedule) == 33 and {rep for _, rep in three_schedule} == {1, 2, 3}
    assert len(runner.expected_cell_keys(1)) == 22
    assert len(runner.expected_cell_keys(3)) == 66
    orders = [runner.pair_variant_order(index) for index in range(11)]
    assert sum(order[0] == "original" for order in orders) == 6
    assert sum(order[0] == "control" for order in orders) == 5

    one_cells = make_cells(1)
    runner.require_repetition_input_stability(one_cells, one_cells[0]["config_id"])
    one_pairs = runner.summarize_real_pairs(one_cells)
    one_groups = runner.summarize_real_groups(one_pairs, 1)
    group = one_groups[0]
    commitment = group["metrics"]["commitment_time_ms"]
    assert commitment["original"] == {"value": 11.0, "n": 1}
    assert commitment["control"] == {"value": 21.0, "n": 1}
    assert commitment["paired_original_minus_control"] == {"value": -10.0, "n": 1}
    assert commitment["saving_percent"]["n"] == 1
    assert math.isclose(commitment["saving_percent"]["value"], (21.0 - 11.0) / 21.0 * 100.0)
    assert commitment["improvement_factor"]["n"] == 1
    assert math.isclose(commitment["improvement_factor"]["value"], 21.0 / 11.0)
    assert all("sample_sd" not in item and "mean" not in item for item in commitment.values())

    three_cells = make_cells(3)
    runner.require_repetition_input_stability(three_cells, three_cells[0]["config_id"])
    three_pairs = runner.summarize_real_pairs(three_cells)
    three_group = runner.summarize_real_groups(three_pairs, 3)[0]
    assert three_group["metrics"]["commitment_time_ms"]["original"]["n"] == 3
    assert three_group["metrics"]["commitment_time_ms"]["original"]["sample_sd"] is not None

    unequal_cells = make_cells(2)
    for cell in unequal_cells:
        if cell["repetition"] == 1:
            cell["commitment_time_ms"] = 1.0 if cell["variant"] == "original" else 2.0
        else:
            cell["commitment_time_ms"] = 9.0 if cell["variant"] == "original" else 90.0
    unequal_pairs = runner.summarize_real_pairs(unequal_cells)
    unequal_group = runner.summarize_real_groups(unequal_pairs, 2)[0]
    paired_metrics = unequal_group["metrics"]["commitment_time_ms"]
    assert math.isclose(paired_metrics["improvement_factor"]["mean"], 6.0)
    assert not math.isclose(paired_metrics["improvement_factor"]["mean"],
                            paired_metrics["control"]["mean"] / paired_metrics["original"]["mean"])
    assert math.isclose(paired_metrics["saving_percent"]["mean"], 70.0)
    assert not math.isclose(paired_metrics["saving_percent"]["mean"],
                            (paired_metrics["control"]["mean"] - paired_metrics["original"]["mean"])
                            / paired_metrics["control"]["mean"] * 100.0)

    manifest = provenance.source_manifest_document(ROOT)
    paths = {entry["path"] for entry in manifest["files"]}
    assert manifest["excluded_reporting_paths"] == ["README.md", "docs/mechanism_ablation.md"]
    assert manifest["excluded_generated_output_paths"] == ["results/mechanism_ablation_final"]
    assert "README.md" not in paths and "docs/mechanism_ablation.md" not in paths
    for readonly in ("GAT-ZKML-多层多头.md", "PROJECT_CONTEXT_AND_EXECUTION_CONSTRAINTS.md",
                     "PERFORMANCE_OPTIMIZATION_CONSTRAINTS.md"):
        assert readonly in paths
    assert not any(path.startswith(("comparison/", "runs/", "tests/")) for path in paths)
    assert not any(path.startswith("results/mechanism_ablation_final/") for path in paths)
    assert not any(path.startswith("results/paper_secure/") for path in paths)
    exclusion_specs = provenance.git_exclusion_pathspecs()
    assert ":(exclude,glob)results/mechanism_ablation_final/**" in exclusion_specs

    assert runner.DEFAULT_OUTPUT_ROOT == ROOT / "results" / "mechanism_ablation_final"
    assert runner.safe_output_root(runner.DEFAULT_OUTPUT_ROOT) == runner.DEFAULT_OUTPUT_ROOT
    candidate_output = runner.DEFAULT_OUTPUT_ROOT / f"runner_test_{os.getpid()}"
    assert runner.safe_output_root(candidate_output) == candidate_output
    try:
        runner.safe_output_root(ROOT / "results" / "paper_secure")
        raise AssertionError("protected paper_secure root was accepted")
    except ValueError:
        pass
    parser_args = runner.build_parser().parse_args(["run", "--device", "0"])
    assert parser_args.output_root == runner.DEFAULT_OUTPUT_ROOT
    assert parser_args.allow_shared_gpu is False
    authorized_run_args = runner.build_parser().parse_args(["run", "--device", "0", "--allow-shared-gpu"])
    assert authorized_run_args.allow_shared_gpu is True
    launch_output = runner.DEFAULT_OUTPUT_ROOT / f"launch_test_{os.getpid()}"
    captured: dict[str, object] = {}
    original_popen = runner.subprocess.Popen

    class FakeChild:
        pid = 123456

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeChild()

    try:
        runner.subprocess.Popen = fake_popen
        runner.launch_formal(types.SimpleNamespace(
            repetitions=1, timeout_seconds=0, prepare_timeout_seconds=0,
            output_root=launch_output, device=0, binary=None, allow_shared_gpu=True,
        ))
        command = captured["command"]
        assert isinstance(command, list)
        assert "--output-root" in command
        assert command[command.index("--output-root") + 1] == str(launch_output)
        assert "--allow-shared-gpu" in command
    finally:
        runner.subprocess.Popen = original_popen
        shutil.rmtree(launch_output, ignore_errors=True)

    output_root_existed = runner.DEFAULT_OUTPUT_ROOT.exists()
    runner.DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    status_before = runner.git_output_excluding_identity_paths(
        "status", "--porcelain=v1", "--untracked-files=all")
    tree_before = provenance.source_manifest_document(ROOT)["source_tree_hash"]
    try:
        with tempfile.TemporaryDirectory(prefix="source_identity_regression_", dir=runner.DEFAULT_OUTPUT_ROOT) as temp_out:
            generated = pathlib.Path(temp_out) / "synthetic_test_only.json"
            generated.write_text("temporary non-formal artifact\n", encoding="utf-8")
            manifest_with_output = provenance.source_manifest_document(ROOT)
            status_with_output = runner.git_output_excluding_identity_paths(
                "status", "--porcelain=v1", "--untracked-files=all")
            assert manifest_with_output["source_tree_hash"] == tree_before
            assert status_with_output == status_before
            assert generated.relative_to(ROOT).as_posix() not in {entry["path"] for entry in manifest_with_output["files"]}
    finally:
        if not output_root_existed and not any(runner.DEFAULT_OUTPUT_ROOT.iterdir()):
            runner.DEFAULT_OUTPUT_ROOT.rmdir()

    test_reverify_archive_validation()

    with tempfile.TemporaryDirectory(prefix="mechanism-runner-regression-") as temporary:
        csv_path = pathlib.Path(temporary) / "single.csv"
        runner.write_paper_csv(csv_path, one_groups, "feature_folding")
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            assert "sample_sd" not in ",".join(reader.fieldnames or [])
            rows = list(reader)
        assert rows[0]["original"] == "11.0"
        assert math.isclose(float(rows[0]["saving_percent"]), (21.0 - 11.0) / 21.0 * 100.0)
        assert float(rows[0]["improvement_factor"]) == 21.0 / 11.0

        md_path = pathlib.Path(temporary) / "single.md"
        document = {
            "session_id": "unit-single", "session_status": "complete", "backend": "cuda",
            "verified_cells": 2, "expected_cell_count": 22, "repetitions": 1,
            "groups": one_groups, "cells": [], "prepared_inputs": [], "session_root": temporary,
            "provenance": {"source": {}, "build": {}, "device": {}, "datasets": {"cora": {"srs": {}}}},
        }
        runner.write_mechanism_paper_markdown(md_path, document, "feature_folding")
        text = md_path.read_text()
        header = next(line for line in text.splitlines() if line.startswith("| Dataset |"))
        assert "sample SD" not in header and "sample standard deviation" not in header
        assert "Commitment Time (ms)" in text
        assert "Each configuration has one paired measurement" in text
        assert "(Control − Original) / Control × 100" in text

    print("REAL_MECHANISM_RUNNER_REGRESSION_OK")


if __name__ == "__main__":
    main()
