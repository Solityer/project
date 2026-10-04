#!/usr/bin/env python3
"""Validate exactly five canonical EZKL/Halo2 formal results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from collect_results import check_pair_consistency, validate_cell
from formal_common import (
    BACKENDS,
    CANONICAL_RELATION,
    DATASETS,
    FORMAL_SESSION_ID,
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
    result.add_argument("--session-id", help="active formal session; required for five-dataset final validation")
    return result


def main(args: argparse.Namespace) -> int:
    return validate_matrix(local_path(args.matrix, "formal matrix"), session_id=args.session_id)


def validate_matrix(
    matrix_path: Path,
    *,
    selected_datasets: tuple[str, ...] = DATASETS,
    selected_backends: tuple[str, ...] | None = None,
    session_id: str | None = None,
) -> int:
    """Validate formal dataset cells; explicit native selections serve only tiny/local checks."""
    formal = selected_backends is None
    if formal and selected_datasets != DATASETS:
        abort("formal validation requires exactly the five canonical datasets")
    if formal:
        if not isinstance(session_id, str) or not FORMAL_SESSION_ID.fullmatch(session_id):
            abort("formal validation requires the active --session-id")
    elif session_id is not None:
        abort("--session-id cannot be combined with local-only validation")
    elif not selected_backends or any(backend not in BACKENDS for backend in selected_backends):
        abort("local validation requires explicit native backend selections")
    elif len(selected_datasets) != 1 or len(selected_backends) != 1:
        abort("local-only validation is restricted to one tiny dataset and one backend")
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
        "cell_count",
        "source_tree_sha256",
        "resource_manifest_sha256",
        "collected_at_utc",
        "cells",
    }
    if formal:
        required.add("formal_session_id")
    else:
        required.update({"backends", "pair_consistency_checked"})
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
    if matrix["datasets"] != list(selected_datasets) or (not formal and matrix["backends"] != list(selected_backends)):
        abort("result matrix dataset set is not canonical")
    if formal and matrix.get("formal_session_id") != session_id:
        abort("formal matrix does not belong to the active session")
    cells = matrix["cells"]
    expected_count = len(selected_datasets) if formal else len(selected_datasets) * len(selected_backends)
    if not isinstance(cells, list) or len(cells) != expected_count:
        abort("result matrix does not contain the required cells")
    if matrix["cell_count"] != len(cells):
        abort("result matrix cell count is invalid")
    if not formal and matrix["pair_consistency_checked"] is not (set(selected_backends) == set(BACKENDS)):
        abort("local result matrix pair attestation is invalid")
    if not isinstance(matrix["collected_at_utc"], str) or not matrix["collected_at_utc"]:
        abort("formal matrix has no collection timestamp")
    if matrix["source_tree_sha256"] != source_tree_hash() or matrix["resource_manifest_sha256"] != resource_manifest_hash():
        abort("formal matrix source/resource identity is stale")

    keys: set[str] | set[tuple[str, str]] = set()
    refreshed: list[dict] = []
    for cell in cells:
        if not isinstance(cell, dict) or not isinstance(cell.get("_result_path"), str):
            abort("formal matrix cell has no baseline-local result path")
        key = cell.get("dataset") if formal else (cell.get("dataset"), cell.get("backend"))
        if key in keys:
            abort(f"duplicate formal matrix cell: {key}")
        keys.add(key)
        result_path = (ROOT / cell["_result_path"]).resolve()
        try:
            result_path.relative_to(ROOT.resolve())
        except ValueError:
            abort(f"formal matrix result path escapes baseline: {result_path}")
        refreshed_cell = validate_cell(
            result_path,
            expected_session_id=session_id if formal else None,
        )
        if set(cell) != set(refreshed_cell):
            abort(f"formal matrix cell fields are stale for {key}")
        for field in cell:
            if field.startswith("_"):
                continue
            if cell[field] != refreshed_cell.get(field):
                abort(f"formal matrix cell is stale for {key}: field={field}")
        refreshed.append(refreshed_cell)
    expected = set(selected_datasets) if formal else {(dataset, backend) for dataset in selected_datasets for backend in selected_backends}
    if keys != expected:
        abort(f"formal matrix is missing or has extra cells: expected={sorted(expected)}, got={sorted(keys)}")
    if not formal:
        check_pair_consistency(refreshed)
    print(
        f"validated {len(refreshed)} ezkl result cells: external statement/Y, native instances, "
        f"circuit artifacts, timing, and {'formal session identity' if formal else 'local execution'} are consistent"
    )
    print(f"FINAL_VALIDATION_OK cells={len(refreshed)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(print_error(exc))
