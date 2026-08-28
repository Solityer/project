#!/usr/bin/env python3
"""Create and validate the baseline-local zkDL resource layout.

The user supplies every formal dataset cache and checkpoint bundle directly
under ``resources/`` in this baseline.  This command never accepts a source
checkout, follows an external path, or copies files from another project.
It only creates the canonical empty directories and, unless ``--init`` is
requested, checks the local bundles required by the formal setup command.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from runner_contract import (
    BASELINE_ROOT,
    DATASETS,
    ensure_contained,
    ensure_directory,
    ensure_regular_file,
    load_resource_policy,
)


def _create_layout(dataset: str, policy: dict) -> tuple[Path, Path]:
    """Create only the canonical baseline-local resource directories."""
    entry = policy["datasets"][dataset]
    cache = ensure_contained(BASELINE_ROOT / entry["cache"], label=f"{dataset} cache")
    checkpoint = ensure_contained(
        BASELINE_ROOT / entry["checkpoint_bundle"], label=f"{dataset} checkpoint"
    )
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    return (
        ensure_directory(cache, label=f"{dataset} cache"),
        ensure_directory(checkpoint, label=f"{dataset} checkpoint"),
    )


def _check_local_bundle(dataset: str, cache: Path, checkpoint: Path) -> None:
    """Check the minimum local files consumed by preflight/setup."""
    ensure_regular_file(cache / "meta.cfg", label=f"{dataset} cache metadata")
    ensure_regular_file(cache / "edges.txt", label=f"{dataset} edges")
    feature_files = [cache / "features.txt", cache / "features.npy"]
    for feature_file in feature_files:
        if feature_file.exists() or feature_file.is_symlink():
            ensure_regular_file(feature_file, label=f"{dataset} features")
    if not any(feature_file.is_file() for feature_file in feature_files):
        raise FileNotFoundError(f"{dataset} cache has no feature resource")
    ensure_regular_file(checkpoint / "manifest.json", label=f"{dataset} model manifest")
    ensure_regular_file(checkpoint / "tensors.txt", label=f"{dataset} model tensors")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create/check resources supplied under comparison/zkdl/resources; "
            "no external source paths are accepted."
        )
    )
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--init",
        action="store_true",
        help="create the selected empty baseline-local directories and stop",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="check the selected local bundles after creating their directories",
    )
    args = parser.parse_args()

    policy, policy_id = load_resource_policy()
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    local = {dataset: _create_layout(dataset, policy) for dataset in selected}
    if args.init:
        for dataset in selected:
            cache, checkpoint = local[dataset]
            print(
                f"ZKDL_RESOURCE_LAYOUT_READY dataset={dataset} "
                f"cache={cache.relative_to(BASELINE_ROOT)} "
                f"checkpoint={checkpoint.relative_to(BASELINE_ROOT)}"
            )
        print(f"policy_id={policy_id}")
        return 0

    for dataset in selected:
        cache, checkpoint = local[dataset]
        _check_local_bundle(dataset, cache, checkpoint)
        print(f"ZKDL_RESOURCE_CHECK_OK dataset={dataset} policy_id={policy_id}")
    print("Local resources are complete. Run scripts/setup_resources.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
