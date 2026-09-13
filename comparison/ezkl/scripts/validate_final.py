#!/usr/bin/env python3
"""Validate the complete ezkl 5x2 result matrix without declaring readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from collect_results import PAIR_FIELDS, check_pair_consistency, validate_cell
from formal_common import (
    BACKENDS,
    CANONICAL_RELATION,
    DATASETS,
    ROOT,
    ContractError,
    abort,
    local_path,
    print_error,
    resource_manifest_hash,
    source_tree_hash,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--matrix", default="results/formal_matrix.json")
    return result


def main(args: argparse.Namespace) -> int:
    matrix_path = local_path(args.matrix, "formal matrix")
    if not matrix_path.is_file() or matrix_path.is_symlink():
        abort(f"missing formal matrix: {matrix_path}")
    try:
        matrix = json.loads(matrix_path.read_text())
    except json.JSONDecodeError as exc:
        abort(f"formal matrix is invalid JSON: {exc}")
    if not isinstance(matrix, dict):
        abort("formal matrix must be a JSON object")
    required = {
        "schema_version",
        "baseline",
        "collection_status",
        "relation_id",
        "datasets",
        "backends",
        "cell_count",
        "source_tree_sha256",
        "resource_manifest_sha256",
        "pair_consistency_checked",
        "collected_at_utc",
        "cells",
    }
    actual = set(matrix)
    missing = required - actual
    extra = actual - required
    if missing or extra:
        abort(f"formal matrix schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    if matrix["schema_version"] != 1 or matrix["baseline"] != "ezkl":
        abort("formal matrix schema/baseline mismatch")
    if matrix["collection_status"] != "COLLECTED_PENDING_FINAL_VALIDATION":
        abort("formal matrix was not produced by the strict collector")
    if matrix["relation_id"] != CANONICAL_RELATION:
        abort("formal matrix relation mismatch")
    if matrix["datasets"] != list(DATASETS) or matrix["backends"] != list(BACKENDS):
        abort("formal matrix dataset/backend axes are not canonical")
    cells = matrix["cells"]
    if not isinstance(cells, list) or len(cells) != len(DATASETS) * len(BACKENDS):
        abort("formal matrix does not contain exactly ten cells")
    if matrix["cell_count"] != len(cells) or matrix["pair_consistency_checked"] is not True:
        abort("formal matrix cell count/pair attestation is invalid")
    if not isinstance(matrix["collected_at_utc"], str) or not matrix["collected_at_utc"]:
        abort("formal matrix has no collection timestamp")
    if matrix["source_tree_sha256"] != source_tree_hash() or matrix["resource_manifest_sha256"] != resource_manifest_hash():
        abort("formal matrix source/resource identity is stale")

    keys: set[tuple[str, str]] = set()
    refreshed: list[dict] = []
    for cell in cells:
        if not isinstance(cell, dict) or not isinstance(cell.get("_result_path"), str):
            abort("formal matrix cell has no baseline-local result path")
        key = (cell.get("dataset"), cell.get("backend"))
        if key in keys:
            abort(f"duplicate formal matrix cell: {key}")
        keys.add(key)
        result_path = (ROOT / cell["_result_path"]).resolve()
        try:
            result_path.relative_to(ROOT.resolve())
        except ValueError:
            abort(f"formal matrix result path escapes baseline: {result_path}")
        refreshed_cell = validate_cell(result_path)
        for field in cell:
            if field.startswith("_"):
                continue
            if cell[field] != refreshed_cell.get(field):
                abort(f"formal matrix cell is stale for {key}: field={field}")
        refreshed.append(refreshed_cell)
    expected = {(dataset, backend) for dataset in DATASETS for backend in BACKENDS}
    if keys != expected:
        abort(f"formal matrix is missing or has extra cells: expected={sorted(expected)}, got={sorted(keys)}")
    check_pair_consistency(refreshed)
    print("validated 10 ezkl result cells: external statement/Y, native instances, circuit artifacts, timing, and CPU/GPU relation are consistent")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(print_error(exc))
