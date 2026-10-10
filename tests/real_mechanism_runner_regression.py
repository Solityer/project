#!/usr/bin/env python3
"""Focused regression checks for repetition profiles and source identity."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import pathlib
import tempfile


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


def main() -> None:
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

    manifest = provenance.source_manifest_document(ROOT)
    paths = {entry["path"] for entry in manifest["files"]}
    assert manifest["excluded_reporting_paths"] == ["README.md", "docs/mechanism_ablation.md"]
    assert "README.md" not in paths and "docs/mechanism_ablation.md" not in paths
    for readonly in ("GAT-ZKML-多层多头.md", "PROJECT_CONTEXT_AND_EXECUTION_CONSTRAINTS.md",
                     "PERFORMANCE_OPTIMIZATION_CONSTRAINTS.md"):
        assert readonly in paths
    assert not any(path.startswith(("comparison/", "runs/", "tests/")) for path in paths)

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
