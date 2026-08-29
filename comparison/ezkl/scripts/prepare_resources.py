#!/usr/bin/env python3
"""Create/check ezkl's baseline-local resource layout.

The command never copies or opens dataset, checkpoint, SRS, or public-output
contents.  Resource owners must stage authenticated files below this baseline.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path

from resource_policy import (
    CEREMONY_ROOT,
    CUDA_BACKEND_MANIFEST,
    CUDA_BACKEND_LIBRARY_PATHS,
    ICICLE_CUDA_ABI,
    ICICLE_SOURCE_COMMIT,
    DATASET_ORDER,
    DATASET_SLUG,
    BASELINE,
    FORMAL_ROUTE,
    ROOT,
    atomic_write_json,
    check_resource_layout,
    manifest_payload,
    safe_mkdir,
)
from formal_source_manifest import lexical_baseline_root, lexical_checked_path


def _resource_layout_targets(root: Path, selected: tuple[str, ...]) -> tuple[list[Path], list[Path]]:
    directories = [
        root / "resources",
        root / CEREMONY_ROOT,
        root / "resources/public_y",
    ]
    for dataset in selected:
        slug = DATASET_SLUG[dataset]
        directories.extend(
            (
                root / "data/cache" / slug,
                root / "artifacts/checkpoints" / f"{slug}_gat",
            )
        )
    write_targets = [
        root / "resources/resource_manifest.json",
        root / CUDA_BACKEND_MANIFEST,
    ]
    return directories, write_targets


def _validate_resource_layout_paths(root: Path, selected: tuple[str, ...]) -> list[Path]:
    """Validate every resource directory and write target before mutation."""

    directories, write_targets = _resource_layout_targets(root, selected)
    for directory in directories:
        lexical_checked_path(
            root,
            directory,
            "baseline-local resource directory",
            allow_missing_tail=True,
        )
    for target in write_targets:
        lexical_checked_path(
            root,
            target,
            "baseline-local resource manifest",
            allow_missing_tail=True,
        )
    return directories


def _blocked_resource_report(
    root: Path,
    selected: tuple[str, ...],
    backend: str,
    error: BaseException,
) -> dict[str, object]:
    """Return a machine-readable fail-closed report for missing resources."""

    config_names = {
        "cora_full": "cora_full.cfg",
        "citeseer_full": "citeseer_full.cfg",
        "pubmed_full": "pubmed_full.cfg",
        "ppi_full_formal": "ppi_full.cfg",
        "ogbn_arxiv_full": "ogbn_arxiv_full.cfg",
    }
    return {
        "schema": "ezkl_formal_resource_check_v1",
        "baseline": BASELINE,
        "route_id": FORMAL_ROUTE,
        "require_staged": True,
        "datasets": list(selected),
        "configs": [
            str((root / "configs" / config_names[dataset]).relative_to(root))
            for dataset in selected
        ],
        "status": "blocked",
        "errors": [str(error)],
        "manifest": "resources/resource_manifest.json",
        "contents_read": False,
        "backend": backend,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--check", action="store_true", help="require staged file names")
    args = parser.parse_args()
    try:
        # Validate the exact baseline root lexically before resolving it.  A
        # symlink alias must not be accepted as a staging root.
        requested_root = Path(os.path.abspath(args.root.expanduser()))
        expected_root = Path(os.path.abspath(ROOT))
        if requested_root != expected_root:
            raise ValueError("--root must be comparison/ezkl")
        root = lexical_baseline_root(requested_root)
        if root != lexical_baseline_root(ROOT):
            raise ValueError("--root must be comparison/ezkl")
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    selected = DATASET_ORDER if args.datasets == "all" else tuple(
        item.strip() for item in args.datasets.split(",") if item.strip()
    )
    unknown = sorted(set(selected) - set(DATASET_ORDER))
    if unknown:
        raise SystemExit(f"unsupported dataset(s): {unknown}")
    # Preflight every path that may be created or written before the first
    # mutation.  This catches an existing resources/ceremony alias (including
    # one pointing outside the baseline) before any manifest or directory is
    # touched.
    directories = _validate_resource_layout_paths(root, selected)
    for directory in directories:
        safe_mkdir(root, directory, "baseline-local resource directory")
    atomic_write_json(root / "resources/resource_manifest.json", manifest_payload(root))
    backend_manifest = root / CUDA_BACKEND_MANIFEST
    if not backend_manifest.exists():
        atomic_write_json(
            backend_manifest,
            {
                "schema": "ezkl_formal_icicle_cuda_backend_v1",
                "backend": "cuda",
                "icicle_commit": ICICLE_SOURCE_COMMIT,
                "abi": ICICLE_CUDA_ABI,
                "library_inventory": [
                    {"path": path, "size": None, "sha256": None}
                    for path in CUDA_BACKEND_LIBRARY_PATHS
                ],
            },
        )
    try:
        report = check_resource_layout(
            root, selected, require_staged=args.check, backend=args.backend
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if not args.check:
            raise
        report = _blocked_resource_report(root, selected, args.backend, exc)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
