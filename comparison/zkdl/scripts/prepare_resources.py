#!/usr/bin/env python3
"""Explicitly stage baseline-local formal resources.

This command is the only supported bridge from a user's source checkout into
the baseline.  It copies data/model inputs once; the formal runner never reads
the source root and fails closed when staging is incomplete.
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
from pathlib import Path

from runner_contract import (
    BASELINE_ROOT,
    DATASETS,
    ensure_contained,
    ensure_directory,
    ensure_regular_file,
    load_resource_manifest,
    resource_inventory_digest,
)


def source_directory(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(str(path)))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        st = current.lstat()
        if stat.S_ISLNK(st.st_mode):
            raise ValueError(f"{label} contains a symlink component: {current}")
        if not stat.S_ISDIR(st.st_mode):
            raise ValueError(f"{label} path component is not a directory: {current}")
    if not path.is_dir():
        raise ValueError(f"{label} must be a real source directory: {path}")
    for item in path.rglob("*"):
        st = item.lstat()
        if stat.S_ISLNK(st.st_mode):
            raise ValueError(f"{label} contains a symlink: {item}")
        if item.is_dir():
            continue
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise ValueError(f"{label} contains a non-regular or hard-linked file: {item}")
    return path


def copy_tree(source: Path, target: Path, *, replace: bool) -> None:
    source = source_directory(source, "source resource")
    target = ensure_contained(target, label="staged resource")
    if target.exists():
        if target.is_symlink() or not target.is_dir():
            raise ValueError(f"staged target is not a real directory: {target}")
        if not replace:
            raise FileExistsError(f"staged target already exists; pass --replace: {target}")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, symlinks=False, copy_function=shutil.copy2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    source_root = source_directory(args.source_root, "source root")
    if not source_root.is_dir() or source_root.is_symlink():
        raise SystemExit(f"invalid source root: {source_root}")
    manifest, manifest_id = load_resource_manifest(require_staged=False)
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        entry = manifest["datasets"][dataset]
        source_cache = source_root / "data" / "cache" / dataset
        source_model_name = entry["checkpoint_bundle"].rsplit("/", 1)[-1]
        source_model = source_root / "artifacts" / "checkpoints" / source_model_name
        copy_tree(source_cache, BASELINE_ROOT / entry["cache"], replace=args.replace)
        copy_tree(source_model, BASELINE_ROOT / entry["checkpoint_bundle"], replace=args.replace)
        # Ensure the minimum files required by preflight are present before
        # reporting staging success.  No proof or verifier is run here.
        cache = ensure_directory(BASELINE_ROOT / entry["cache"], label=f"{dataset} cache")
        checkpoint = ensure_directory(
            BASELINE_ROOT / entry["checkpoint_bundle"], label=f"{dataset} checkpoint"
        )
        ensure_regular_file(cache / "meta.cfg", label=f"{dataset} cache metadata")
        ensure_regular_file(cache / "edges.txt", label=f"{dataset} edges")
        if not (cache / "features.txt").is_file() and not (cache / "features.npy").is_file():
            raise FileNotFoundError(f"{dataset} cache has no feature resource")
        ensure_regular_file(checkpoint / "manifest.json", label=f"{dataset} model manifest")
        ensure_regular_file(checkpoint / "tensors.txt", label=f"{dataset} model tensors")
        print(f"STAGED dataset={dataset} manifest={manifest_id} inventory={resource_inventory_digest(dataset, manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
