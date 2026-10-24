#!/usr/bin/env python3
"""Focused regression checks for repetition profiles and source identity."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import pathlib
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
                "prover_wall_time_seconds": 0.1 + repetition if original else 0.2 + repetition,
                "verify_wall_time_seconds": 0.01 + repetition if original else 0.02 + repetition,
            })
    return cells


def make_complete_combined_document(session: pathlib.Path, repetitions: int = 3):
    document = runner.empty_summary("combined-summary-regression", session, repetitions)
    document.update({"session_status": "complete", "provenance_stable": True})
    document["provenance"] = {
        "source": {
            "source_tree_hash": "1" * 64,
            "git_diff_sha256": "2" * 64,
            "git_status_sha256": "3" * 64,
        },
        "build": {
            "binary_sha256": "4" * 64,
            "cmake_cache_sha256": "5" * 64,
            "source_manifest_sha256": "6" * 64,
        },
    }
    cells = []
    for config_index, spec in enumerate(runner.FORMAL_CONFIGS):
        cfgid = runner.matrix_config_id(spec)
        n, e, _ = runner.expected_geometry(spec)
        for repetition in range(1, repetitions + 1):
            for variant in runner.VARIANTS:
                cell_root = session / "synthetic_fixture" / cfgid / f"rep_{repetition}" / variant
                cell_root.mkdir(parents=True)
                original = variant == "original"
                width_factor = 1.0 if original else 1.7 + 0.2 * repetition
                cell = {
                    "mechanism": spec["mechanism"], "dataset": spec["dataset"], "config_id": cfgid,
                    "N": n, "E": e, "D": spec["width"], "G": spec["graphs"],
                    "component_ids": list(spec["components"] or (0,)),
                    "repetition": repetition, "variant": variant,
                    "pair_status": "paired", "cell_status": "complete", "verification_status": "VERIFY_OK",
                    "backend": "cuda", "cuda_actual": True, "cuda_kernel_invocations_delta": 2,
                }
                for metric_index, metric in enumerate(runner.METRIC_FIELDS):
                    measured = float(100 + config_index * 7 + repetition * (metric_index + 1))
                    if metric == "proof_size_bytes":
                        measured = float(1000 + config_index * 7 + repetition * 2)
                        if not original:
                            measured += 500
                    else:
                        measured = measured * width_factor
                    cell[metric] = measured
                cell["prover_wall_time_seconds"] = 0.3 + repetition / 10 + (0 if original else 0.2)
                cell["verify_wall_time_seconds"] = 0.04 + repetition / 100 + (0 if original else 0.02)
                artifact_pairs = (
                    ("proof_path", "proof_sha256", "proof.bin", b"proof"),
                    ("statement_path", "statement_file_sha256", "statement.json", b"statement"),
                    ("prover_result_path", "prover_result_sha256", "prover.json", b"prover receipt"),
                    ("verifier_result_path", "verifier_result_sha256", "verifier.json", b"VERIFY_OK"),
                    ("prover_log", "prover_log_sha256", "prover.log", b"prove log"),
                    ("verifier_log", "verifier_log_sha256", "verifier.log", b"verify log"),
                )
                for path_field, digest_field, name, content in artifact_pairs:
                    artifact = cell_root / name
                    artifact.write_bytes(content)
                    cell[path_field] = str(artifact)
                    cell[digest_field] = runner.sha256_file(artifact)
                cell["cell_json_path"] = str(cell_root / "cell.json")
                runner.atomic_write_json(pathlib.Path(cell["cell_json_path"]), cell)
                cells.append(cell)
    runner.update_summary_counts(document, cells)
    return document


def test_flat_report_export_and_raw_manifest_copy() -> None:
    with tempfile.TemporaryDirectory(prefix="mechanism_flat_report_regression_") as temporary:
        root = pathlib.Path(temporary)
        session = root / "complete"
        session.mkdir()
        document = make_complete_combined_document(session, repetitions=3)
        paths = runner.generate_reports(session, document)
        assert set(paths) == {"summary_markdown", "results_csv"}
        csv_bytes = (session / "results.csv").read_bytes()
        assert csv_bytes.startswith(b"\xef\xbb\xbf")
        with (session / "results.csv").open(newline="", encoding="utf-8-sig") as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(csv_rows) == 282
        assert {"机制", "数据集", "输入范围", "N", "E", "G", "D", "版本", "重复序号",
                "原生承诺时间_ms", "完整Prover进程时间_秒", "独立验证", "安全账本", "单元状态"} <= set(csv_rows[0])
        assert all(row["单元状态"] == "已完成" for row in csv_rows)
        assert "282" in (session / "summary.md").read_text(encoding="utf-8")
        manifest = json.loads((session / "results_manifest.json").read_text(encoding="utf-8"))
        assert set(paths) == set(manifest["files"])
        assert all(item["sha256"] == runner.sha256_file(pathlib.Path(item["path"]))
                   for item in manifest["files"].values())

        planned_session = root / "planned"
        planned_session.mkdir()
        planned = runner.empty_summary("planned-regression", planned_session, 3)
        runner.update_summary_counts(planned, [])
        runner.generate_reports(planned_session, planned)
        with (planned_session / "results.csv").open(newline="", encoding="utf-8-sig") as stream:
            planned_rows = list(csv.DictReader(stream))
        assert len(planned_rows) == 282
        assert all(row["单元状态"] == "未运行" and row["独立验证"] == "未验证" and
                   row["安全账本"] == "未验证" and row["原生承诺时间_ms"] == "" and row["证明字节数"] == ""
                   for row in planned_rows)

        selections = (
            ("feature_folding", "main", 40, 240),
            ("shared_neighborhood", "main", 5, 30),
            ("all", "ppi_full", 2, 12),
        )
        for mechanism, input_scope, expected_groups, expected_cells in selections:
            configs = runner.selected_formal_configs(mechanism, input_scope)
            assert len(configs) == expected_groups
            assert len(runner.formal_schedule(3, configs)) == expected_groups * 3
            assert len(runner.expected_cell_keys(3, configs)) == expected_cells
            selected_doc = runner.empty_summary(
                f"selection-{mechanism}-{input_scope}", planned_session, 3, mechanism, input_scope)
            runner.update_summary_counts(selected_doc, [])
            selected_rows_path = root / f"{mechanism}_{input_scope}.csv"
            runner.write_results_csv(selected_rows_path, selected_doc)
            with selected_rows_path.open(newline="", encoding="utf-8-sig") as stream:
                selected_rows = list(csv.DictReader(stream))
            assert len(selected_rows) == expected_cells
            expected_config_ids = {runner.matrix_config_id(spec) for spec in configs}
            assert {row["配置ID"] for row in selected_rows} == expected_config_ids
            assert all(row["单元状态"] == "未运行" for row in selected_rows)
            summary_text = root / f"{mechanism}_{input_scope}.md"
            runner.write_summary_markdown(summary_text, selected_doc)
            summary_contents = summary_text.read_text(encoding="utf-8")
            if input_scope == "main":
                assert "不是 zkGAT 主实验" in summary_contents
            if mechanism == "feature_folding":
                assert "## 共享邻域" not in summary_contents
            if mechanism == "shared_neighborhood":
                assert "## 特征折叠" not in summary_contents
            if input_scope == "ppi_full":
                assert "## 特征折叠" in summary_contents and "## 共享邻域" in summary_contents

            tampered = dict(selected_doc)
            tampered["formal_configs"] = selected_doc["formal_configs"][:-1]
            try:
                runner.configs_for_document(tampered)
                raise AssertionError("a free-form shortened configuration list was accepted")
            except ValueError:
                pass

        producer = root / "producer_source_manifest.json"
        copied = root / "copied_source_manifest.json"
        producer_bytes = b'{\n  "source_tree_hash": "abc",\n  "files": []\n}\n'
        producer.write_bytes(producer_bytes)
        runner.copy_build_source_manifest(producer, copied, runner.sha256_bytes(producer_bytes))
        assert copied.read_bytes() == producer_bytes
        try:
            runner.copy_build_source_manifest(producer, copied, "0" * 64)
            raise AssertionError("a build source manifest with a mismatched raw digest was copied")
        except ValueError:
            pass
        assert copied.read_bytes() == producer_bytes


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
        spec = runner.selected_formal_configs("feature_folding", "ppi_full")[0]
        config_id = runner.matrix_config_id(spec)
        nodes, edges, graphs = runner.expected_geometry(spec)
        cells = [{
            "config_id": config_id, "repetition": 1, "variant": variant,
            "N": nodes, "E": edges, "D": spec["width"], "G": graphs,
            "component_ids": list(spec["components"]), "pair_status": "unpaired",
            "proof_sha256": proof_sha, "statement_file_sha256": statement_sha,
            "prover_result_sha256": prover_sha, "verifier_result_sha256": verifier_sha,
        } for variant in runner.VARIANTS]
        source = {"source_tree_hash": source_hash, "git_diff_sha256": diff_hash}
        build = {"binary_sha256": binary_hash, "source_manifest_sha256": build_manifest_hash}
        datasets = {"test": {"srs": {"manifest_sha256": srs_hash}}}
        document = runner.empty_summary(
            "synthetic-archive-regression", session, 1, "feature_folding", "ppi_full")
        document.update({
            "session_status": "complete", "expected_cell_count": len(cells), "cells": cells,
            "provenance": {"source": source, "build": build, "datasets": datasets},
        })
        runner.update_summary_counts(document, cells)
        runner.generate_reports(session, document)
        summary_sha = runner.sha256_file(session / "summary.json")
        result_manifest_sha = runner.sha256_file(session / "results_manifest.json")
        audit_cell = {
            **cells[0], "status": "VERIFY_OK", "exit_code": 0,
            "receipt_path": str(session / "verify.json"),
            "receipt_sha256": runner.sha256_file(session / "verify.json"),
            "log_path": str(session / "verify.log"),
            "log_sha256": runner.sha256_file(session / "verify.log"),
        }
        audit = {
            "format": "zkgat_real_mechanism_result_audit_v1", "session_id": document["session_id"],
            "selection": document["selection"], "expected_config_ids": [config_id],
            "status": "PASS", "reverified_independently": True,
            "summary_sha256": summary_sha, "cell_count": len(cells), "verified_cells": len(cells),
            "cells": [audit_cell, {
                **audit_cell, "variant": "control",
            }],
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
    test_flat_report_export_and_raw_manifest_copy()
    one_schedule = runner.formal_schedule(1)
    three_schedule = runner.formal_schedule(3)
    assert len(runner.FORMAL_CONFIGS) == 47
    assert runner.registered_matrix_counts() == {
        "total_configurations": 47,
        "feature_folding_main_matrix_count": 40,
        "shared_neighborhood_main_matrix_count": 5,
        "ppi_component0_main_matrix_groups": 9,
        "ppi_full295_appendix_groups": 2,
    }
    assert sum(spec["mechanism"] == "feature_folding" and spec["graphs"] == 1
               for spec in runner.FORMAL_CONFIGS if spec["dataset"] != "ppi" or spec["components"] == (0,)) == 40
    assert sum(spec["mechanism"] == "shared_neighborhood" and spec["graphs"] == 1
               for spec in runner.FORMAL_CONFIGS if spec["dataset"] != "ppi" or spec["components"] == (0,)) == 5
    assert len(one_schedule) == 47 and {rep for _, rep in one_schedule} == {1}
    assert len(three_schedule) == 141 and {rep for _, rep in three_schedule} == {1, 2, 3}
    assert len(runner.expected_cell_keys(1)) == 94
    assert len(runner.expected_cell_keys(3)) == 282
    assert [len(runner.selected_formal_configs(mechanism, scope)) for mechanism, scope in (
        ("feature_folding", "main"), ("shared_neighborhood", "main"), ("all", "ppi_full"))] == [40, 5, 2]
    assert sum(spec["dataset"] == "ppi" and spec["components"] == runner.PPI_FULL_COMPONENTS
               and spec["graphs"] == 295 for spec in runner.FORMAL_CONFIGS) == 2
    assert sum(spec["dataset"] == "ppi" and spec["graphs"] == 2 for spec in runner.FORMAL_CONFIGS) == 0
    orders = [runner.pair_variant_order(index) for index in range(47)]
    assert sum(order[0] == "original" for order in orders) == 24
    assert sum(order[0] == "control" for order in orders) == 23

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
    assert three_group["metrics"]["prover_wall_time_seconds"]["original"]["n"] == 3

    partial_cells = make_cells(3)
    for cell in partial_cells:
        if cell["repetition"] == 2:
            cell["pair_status"] = "invalid_pair"
    partial_group = runner.summarize_real_groups(runner.summarize_real_pairs(partial_cells), 3)[0]
    assert partial_group["complete_repetitions"] == 2
    assert partial_group["metrics"]["commitment_time_ms"]["original"]["n"] == 2

    negative_saving = make_cells(1)
    for cell in negative_saving:
        if cell["variant"] == "original":
            cell["commitment_time_ms"] = 20.0
        else:
            cell["commitment_time_ms"] = 10.0
    negative_pair = runner.summarize_real_pairs(negative_saving)[0]
    assert negative_pair["metrics"]["commitment_time_ms"]["saving_percent"] == -100.0
    assert negative_pair["metrics"]["commitment_time_ms"]["improvement_factor"] == 0.5
    formatting_fixture = {
        "original": runner._sample([1.0, 1.0, 1.0]),
        "control": runner._sample([2.0, 2.0, 2.0]),
        "saving_percent": runner._sample([50.0, 50.0, 50.0]),
        "improvement_factor": runner._sample([2.0, 2.0, 2.0]),
    }
    formatted = runner._format_metric_effect(formatting_fixture, "commitment_time_ms", 3)
    assert "节省 50 ± 0%" in formatted and "C/O 2 ± 0×" in formatted and "节省 50 ± 0 ms" not in formatted

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
    preflight_args = runner.build_parser().parse_args(["preflight", "--device", "0"])
    assert preflight_args.repetitions == 3
    assert preflight_args.output_root == runner.DEFAULT_OUTPUT_ROOT
    assert preflight_args.mechanism == "all" and preflight_args.input_scope == "all"
    split_args = runner.build_parser().parse_args([
        "launch", "--device", "0", "--mechanism", "feature_folding", "--input-scope", "main"])
    assert split_args.mechanism == "feature_folding" and split_args.input_scope == "main"
    captured: dict[str, object] = {}
    original_popen = runner.subprocess.Popen
    original_default_output_root = runner.DEFAULT_OUTPUT_ROOT

    class FakeChild:
        pid = 123456

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeChild()

    try:
        runner.subprocess.Popen = fake_popen
        with tempfile.TemporaryDirectory(prefix="ablation_launch_bootstrap_", dir=runner.DEFAULT_OUTPUT_ROOT) as temp_root:
            runner.DEFAULT_OUTPUT_ROOT = pathlib.Path(temp_root) / "isolated_default_root"
            runner.DEFAULT_OUTPUT_ROOT.mkdir()
            launch_output = runner.DEFAULT_OUTPUT_ROOT / "ff"
            runner.launch_formal(types.SimpleNamespace(
                repetitions=1, timeout_seconds=0, prepare_timeout_seconds=0,
                output_root=launch_output, device=0, binary=None, allow_shared_gpu=True,
            ))
            command = captured["command"]
            kwargs = captured["kwargs"]
            assert isinstance(command, list) and isinstance(kwargs, dict)
            assert "--output-root" in command
            assert command[command.index("--output-root") + 1] == str(launch_output)
            assert command[command.index("--mechanism") + 1] == "all"
            assert command[command.index("--input-scope") + 1] == "all"
            assert "--allow-shared-gpu" in command
            assert not launch_output.exists()
            assert runner.safe_output_root(launch_output) == launch_output
            launch_log = pathlib.Path(kwargs["stdout"].name)
            launcher_dir = launch_log.parent
            launcher_metadata = json.loads((launcher_dir / "launcher.json").read_text(encoding="utf-8"))
            assert launcher_metadata["runner_output_root"] == str(launch_output)
            assert launcher_metadata["command"] == command
            assert pathlib.Path(launcher_metadata["log_path"]) == launch_log
            assert (launcher_dir / "launcher.pid").read_text(encoding="ascii") == "123456\n"

            selected_output = runner.DEFAULT_OUTPUT_ROOT / "sn"
            runner.launch_formal(types.SimpleNamespace(
                repetitions=3, timeout_seconds=0, prepare_timeout_seconds=0,
                output_root=selected_output, device=0, binary=None, allow_shared_gpu=False,
                mechanism="shared_neighborhood", input_scope="main",
            ))
            selected_command = captured["command"]
            assert isinstance(selected_command, list)
            assert selected_command[selected_command.index("--mechanism") + 1] == "shared_neighborhood"
            assert selected_command[selected_command.index("--input-scope") + 1] == "main"
            assert not selected_output.exists()
            assert runner.safe_output_root(selected_output) == selected_output

            launch_output.mkdir()
            try:
                runner.safe_output_root(launch_output)
                raise AssertionError("an existing custom formal output root was accepted")
            except ValueError:
                pass
    finally:
        runner.subprocess.Popen = original_popen
        runner.DEFAULT_OUTPUT_ROOT = original_default_output_root

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

    print("REAL_MECHANISM_RUNNER_REGRESSION_OK")


if __name__ == "__main__":
    main()
