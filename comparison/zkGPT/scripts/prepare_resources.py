#!/usr/bin/env python3
"""Prepare/check the baseline-local zkGPT resource layout.

The default operation creates only the directory/manifest skeleton.  The
``--check`` operation is presence-only.  Content generation is intentionally a
separate official ``gat_zkgpt_experiment.py`` preparation stage so a direct
formal start can authenticate and cache canonical checkpoints/public outputs
without asking the operator to copy or convert anything by hand.
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
from zkgpt_gat.util import ensure_dir, lexical_path, require_dir, write_json  # noqa: E402


def _require_stable_resource_file(path: pathlib.Path, label: str) -> None:
    """Reject aliases/special files before descriptor validation reads them."""
    import stat

    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise SystemExit(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SystemExit(f"{label} must be an owned regular file: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare/check baseline-local zkGPT resources")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--check", action="store_true", help="check staged file presence without opening contents")
    args = parser.parse_args()
    root = lexical_path(args.root)
    expected_root = require_dir(ROOT, label="zkGPT baseline root")
    if root != expected_root:
        raise SystemExit("--root must be comparison/zkGPT; external project resource roots are forbidden")
    selected = DATASET_ORDER[:] if args.datasets == "all" else [item.strip() for item in args.datasets.split(",") if item.strip()]
    selected = [canonical_dataset_name(item) for item in selected]
    unknown = sorted(set(selected) - set(DATASET_ORDER))
    if unknown:
        raise SystemExit(f"unsupported dataset(s): {unknown}")
    resource_dir = ensure_dir(root / "resources", root=root, label="resource root")
    descriptor_path = root / "resources" / "static_tables" / "exact_integer_gat_v1.json"
    _require_stable_resource_file(descriptor_path, "static-table descriptor")
    manifest_path = resource_dir / "resource_manifest.json"
    # Re-emit the canonical registration manifest on every layout operation.
    # This changes only baseline-owned descriptors and never reads or rewrites
    # dataset, checkpoint, or public-output contents.
    write_json(
        manifest_path,
        resource_manifest_payload(root),
        root=root,
        label="resource manifest",
    )
    if args.check:
        report = check_resource_layout(root, selected, require_staged=True)
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        # Layout creation is limited to empty directories; it does not create
        # placeholder data/model/table files that could be mistaken for proof
        # inputs.
        ensure_dir(root / "data", root=root, label="data root")
        ensure_dir(root / "data" / "cache", root=root, label="cache root")
        ensure_dir(root / "artifacts" / "checkpoints", root=root, label="checkpoint root")
        ensure_dir(root / MODEL_REGISTRY_ROOT_RELATIVE_PATH, root=root, label="model registry root")
        ensure_dir(root / PRIVATE_MODEL_SETUP_ROOT_RELATIVE_PATH, root=root, label="private model setup root")
        for dataset in selected:
            paths = required_resource_paths(root, dataset)
            ensure_dir(paths["cache_dir"], root=root, label=f"{dataset} cache")
            ensure_dir(paths["checkpoint_bundle"], root=root, label=f"{dataset} checkpoint bundle")
        ensure_dir(root / "resources" / "public_y", root=root, label="public output root")
        ensure_dir(root / "resources" / "static_tables", root=root, label="static table root")
        print(f"baseline-local resource layout prepared at {root}; authenticated files remain required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
