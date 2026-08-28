#!/usr/bin/env python3
"""Low-cost source/config/resource preflight for the formal ezkl route."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import shutil
from pathlib import Path

from resource_policy import (
    DATASET_ORDER,
    FORMAL_ROUTE,
    ROOT,
    atomic_write_json,
    check_resource_layout,
    safe_mkdir,
    read_resource_manifest,
)
from formal_source_manifest import checked_path, lexical_baseline_root, lexical_checked_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "preflight.json")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--require-staged", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "cuda"))
    args = parser.parse_args()
    try:
        # Keep the caller's path lexical until every component has been
        # inspected.  Resolving --output first would turn a symlink alias into
        # the external target and could redirect the report write.
        root = lexical_baseline_root(ROOT)
        output_lexical = lexical_checked_path(
            root,
            args.output,
            "preflight output",
            allow_missing_tail=True,
        )
        # Safely create a missing output parent after the complete lexical
        # path has been checked, then re-check the destination itself.
        safe_mkdir(root, output_lexical.parent, "preflight output parent")
        output = checked_path(root, output_lexical, "preflight output", allow_missing=True)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    selected = DATASET_ORDER if args.datasets == "all" else tuple(
        item.strip() for item in args.datasets.split(",") if item.strip()
    )
    report: dict[str, object] = {
        "schema": "ezkl_formal_preflight_v1",
        "baseline": "ezkl",
        "route_id": FORMAL_ROUTE,
        "source_only": not args.require_staged,
        "python3": shutil.which("python3") is not None,
        "cmake": shutil.which("cmake") is not None,
        "datasets": list(selected),
        "backend": args.backend or "both",
        "status": "ready",
        "errors": [],
    }
    try:
        manifest = read_resource_manifest(root)
        report["resource_manifest_schema"] = manifest["schema"]
        report["resource_check"] = check_resource_layout(
            root, selected, require_staged=args.require_staged, backend=args.backend
        )
        for dataset in selected:
            config = root / "configs" / {
                "cora_full": "cora_full.cfg",
                "citeseer_full": "citeseer_full.cfg",
                "pubmed_full": "pubmed_full.cfg",
                "ppi_full_formal": "ppi_full.cfg",
                "ogbn_arxiv_full": "ogbn_arxiv_full.cfg",
            }.get(dataset, "")
            if not config.is_file():
                raise FileNotFoundError(f"missing formal config: {config}")
        if args.backend:
            binary = root / (
                "target-gpu/release/ezkl_formal"
                if args.backend == "cuda"
                else "target/release/ezkl_formal"
            )
            report["binary"] = str(binary.relative_to(root))
            report["binary_present"] = binary.is_file()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["status"] = "blocked"
        report["errors"] = [str(exc)]
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
