#!/usr/bin/env python3
"""Create/check ezkl's baseline-local resource layout.

The command never copies or opens dataset, checkpoint, SRS, or public-output
contents.  Resource owners must stage authenticated files below this baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from resource_policy import (
    DATASET_ORDER,
    ROOT,
    atomic_write_json,
    check_resource_layout,
    dataset_paths,
    manifest_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--check", action="store_true", help="require staged file names")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if root != ROOT.resolve():
        raise SystemExit("--root must be comparison/ezkl")
    selected = DATASET_ORDER if args.datasets == "all" else tuple(
        item.strip() for item in args.datasets.split(",") if item.strip()
    )
    unknown = sorted(set(selected) - set(DATASET_ORDER))
    if unknown:
        raise SystemExit(f"unsupported dataset(s): {unknown}")
    (root / "resources").mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "resources/resource_manifest.json", manifest_payload(root))
    for dataset in selected:
        paths = dataset_paths(root, dataset)
        paths["dataset_root"].mkdir(parents=True, exist_ok=True)
        paths["checkpoint"].mkdir(parents=True, exist_ok=True)
    (root / "resources/public_y").mkdir(parents=True, exist_ok=True)
    if args.check:
        report = check_resource_layout(root, selected, require_staged=True)
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        report = check_resource_layout(root, selected, require_staged=False)
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

