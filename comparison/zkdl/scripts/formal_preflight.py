#!/usr/bin/env python3
"""Build the five-dataset shape and resource plan used by formal admission."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
CANONICAL_FULL_SHAPES = {
    "cora": (2708, 1433, 7, 1, 10858),
    "citeseer": (3327, 3703, 6, 1, 9464),
    "pubmed": (19717, 500, 3, 1, 88676),
    "ppi": (56944, 50, 121, 295, 818716),
    "ogbn_arxiv": (169343, 128, 40, 1, 1166243),
}


def audit_resource_prerequisites(config: Path) -> dict[str, object]:
    from prepare_form_resources import audit_formal_prerequisites

    evidence = audit_formal_prerequisites(config)
    if not isinstance(evidence, dict):
        raise ValueError(f"formal resource plan is invalid: {config}")
    return evidence


def check_config(root: Path, dataset: str) -> dict[str, object]:
    config = root / "configs" / f"{dataset}_secure.cfg"
    evidence = audit_resource_prerequisites(config)
    shape = evidence.get("input_shape")
    graph = evidence.get("graph")
    if not isinstance(shape, dict) or not isinstance(graph, dict):
        raise ValueError(f"formal resource plan lacks graph/model shape: {dataset}")
    expected_nodes, expected_features, expected_classes, expected_graphs, expected_raw_edges = CANONICAL_FULL_SHAPES[dataset]
    actual_shape = (shape.get("nodes"), shape.get("features"), shape.get("classes"), shape.get("graphs"))
    if actual_shape != (expected_nodes, expected_features, expected_classes, expected_graphs):
        raise ValueError(f"formal full-scale shape mismatch for {dataset}: {actual_shape}")
    if graph.get("raw_edge_rows") != expected_raw_edges:
        raise ValueError(f"formal raw edge count mismatch for {dataset}: {graph.get('raw_edge_rows')}")
    return {"dataset": dataset, "input_shape": shape, "resource_prerequisites": evidence}


def main() -> int:
    argparse.ArgumentParser().parse_args()
    root = Path(__file__).resolve().parent.parent
    report: dict[str, dict[str, object]] = {}
    for dataset in DATASETS:
        try:
            row = check_config(root, dataset)
            row["status"] = "PASS"
            report[dataset] = row
        except SystemExit as exc:
            detail = exc.code if isinstance(exc.code, str) else f"exit={exc.code}"
            report[dataset] = {"dataset": dataset, "status": "FAILED", "failure": str(detail)}
        except Exception as exc:
            report[dataset] = {"dataset": dataset, "status": "FAILED",
                               "failure": f"{type(exc).__name__}: {exc}"}

    report_raw_path = os.environ.get("ZKDL_PREFLIGHT_REPORT")
    if report_raw_path:
        report_path = Path(report_raw_path)
        try:
            report_path.resolve().relative_to(root / "runs")
        except ValueError as exc:
            raise SystemExit("formal_preflight: report path must be under runs/") from exc
        if report_path.exists() or not report_path.parent.is_dir():
            raise SystemExit("formal_preflight: report path must be new and its directory must exist")
        filesystem = os.statvfs(report_path.parent)
        available = filesystem.f_bavail * filesystem.f_frsize
        planned = 0
        for dataset in DATASETS:
            row = report[dataset]
            if row.get("status") != "PASS":
                continue
            evidence = row["resource_prerequisites"]
            srs_bytes = int(evidence["srs_plan"]["estimated_materialized_bytes"])
            if planned + srs_bytes > available:
                row["status"] = "FAILED"
                row["failure"] = f"SRS setup needs {planned + srs_bytes} bytes; {available} are available"
            else:
                row["resource_reservation_bytes"] = srs_bytes
                planned += srs_bytes
        document = {
            "schema": "zkDL_formal_input_preflight_v1",
            "datasets": report,
            "combined_capacity": {"srs_materialized_bytes": planned, "available_bytes": available},
        }
        report_path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    passed = sum(row.get("status") == "PASS" for row in report.values())
    failed = len(DATASETS) - passed
    status = "PASS" if not failed else "PARTIAL"
    print(f"formal_input_preflight={status} datasets=5 ready={passed} failed={failed}")
    return 0 if report_raw_path or not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
