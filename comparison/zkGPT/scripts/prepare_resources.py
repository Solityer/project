#!/usr/bin/env python3
"""Prepare/check the baseline-local resource layout without loading data.

This command is intentionally a manifest/layout operation.  It never copies
from an external project directory and never opens a
dataset, checkpoint, or public-output payload.  The formal owner stages
authenticated resources below this baseline before running ``--check`` and
the proof runner.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.constants import DATASET_ORDER  # noqa: E402
from zkgpt_gat.resources import (  # noqa: E402
    MODEL_REGISTRY_ROOT_RELATIVE_PATH,
    PRIVATE_MODEL_SETUP_ROOT_RELATIVE_PATH,
    check_resource_layout,
    canonical_dataset_name,
    required_resource_paths,
    resource_manifest_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare/check baseline-local zkGPT resources")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--check", action="store_true", help="check staged file presence without opening contents")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if root != ROOT.resolve():
        raise SystemExit("--root must be comparison/zkGPT; external project resource roots are forbidden")
    selected = DATASET_ORDER[:] if args.datasets == "all" else [item.strip() for item in args.datasets.split(",") if item.strip()]
    selected = [canonical_dataset_name(item) for item in selected]
    unknown = sorted(set(selected) - set(DATASET_ORDER))
    if unknown:
        raise SystemExit(f"unsupported dataset(s): {unknown}")
    resource_dir = root / "resources"
    resource_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resource_dir / "resource_manifest.json"
    # Re-emit the canonical registration manifest on every layout operation.
    # This changes only baseline-owned descriptors and never reads or rewrites
    # dataset, checkpoint, or public-output contents.
    manifest_path.write_text(
        json.dumps(resource_manifest_payload(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.check:
        report = check_resource_layout(root, selected, require_staged=True)
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        # Layout creation is limited to empty directories; it does not create
        # placeholder data/model/table files that could be mistaken for proof
        # inputs.
        (root / "data").mkdir(parents=True, exist_ok=True)
        (root / "data" / "cache").mkdir(parents=True, exist_ok=True)
        (root / "artifacts" / "checkpoints").mkdir(parents=True, exist_ok=True)
        (root / MODEL_REGISTRY_ROOT_RELATIVE_PATH).mkdir(parents=True, exist_ok=True)
        (root / PRIVATE_MODEL_SETUP_ROOT_RELATIVE_PATH).mkdir(parents=True, exist_ok=True)
        for dataset in selected:
            paths = required_resource_paths(root, dataset)
            paths["cache_dir"].mkdir(parents=True, exist_ok=True)
            paths["checkpoint_bundle"].mkdir(parents=True, exist_ok=True)
        (root / "resources" / "public_y").mkdir(parents=True, exist_ok=True)
        (root / "resources" / "static_tables").mkdir(parents=True, exist_ok=True)
        print(f"baseline-local resource layout prepared at {root}; authenticated files remain required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
