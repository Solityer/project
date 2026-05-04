#!/usr/bin/env python3
"""Create a portable strict GPU experiment bundle for an x86-64 CUDA host."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


EZKL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = EZKL_ROOT / "gpu_experiment_bundle"
INCLUDE_PATHS = [
    "Cargo.toml",
    "Cargo.lock",
    "src",
    "cpp/gat_ezkl_adapter",
    "configs",
    "scripts",
    "README.md",
    "results/control_variable_audit.md",
    "results/large_public_output_policy_review.md",
    "results/gpu_remote_runbook.md",
    "results/gpu_backend_source_audit.md",
    "results/gpu_backend_build_attempt.md",
]
EXCLUDE_NAMES = {
    "target",
    "build_gat_adapter",
    "build_gpu_backend",
    ".cargo_local",
    ".rustup",
    ".icicle_backend_local",
    ".gpu_backend_local",
    ".vendor_local",
    "gpu_experiment_bundle",
    "__pycache__",
}
EXCLUDE_SUFFIXES = {
    ".so",
    ".dylib",
    ".a",
    ".o",
    ".pyc",
    ".pyo",
    ".tmp",
    ".bak",
    ".old",
    ".orig",
    ".compiled",
    ".key",
    ".srs",
    ".bin",
}
EXCLUDE_FILENAMES = {
    "proof.json",
    "witness.json",
    "model.compiled",
    "pk.key",
    "vk.key",
    "pk.srs",
    "vk.srs",
}


def is_excluded(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_NAMES:
        return True
    if path.name in EXCLUDE_FILENAMES:
        return True
    return path.suffix in EXCLUDE_SUFFIXES


def copy_path(source: Path, destination: Path) -> None:
    if is_excluded(source.relative_to(EZKL_ROOT)):
        return
    if source.is_dir():
        for child in source.iterdir():
            copy_path(child, destination / child.name)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def create_bundle(bundle_dir: Path) -> None:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    for relative in INCLUDE_PATHS:
        source = EZKL_ROOT / relative
        if source.exists():
            copy_path(source, bundle_dir / relative)
    manifest = bundle_dir / "BUNDLE_CONTENTS.md"
    manifest.write_text(
        "\n".join(
            [
                "# ezkl GPU 远端实验包",
                "",
                "本包只包含 strict GPU route 运行所需的源码、配置、脚本和审计文档。",
                "不包含 proof、key、witness、compiled circuit、构建目录或 Icicle 后端二进制。",
                "",
                "远端运行步骤见 `results/gpu_remote_runbook.md`。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def create_tarball(bundle_dir: Path, tarball: Path) -> None:
    if tarball.exists():
        tarball.unlink()
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(bundle_dir, arcname=bundle_dir.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--tarball", default=str(EZKL_ROOT / "gpu_experiment_bundle.tar.gz"))
    parser.add_argument("--no-tarball", action="store_true")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    if bundle_dir == EZKL_ROOT or EZKL_ROOT not in bundle_dir.parents:
        raise SystemExit(f"bundle dir must stay under {EZKL_ROOT}: {bundle_dir}")
    create_bundle(bundle_dir)
    if not args.no_tarball:
        create_tarball(bundle_dir, Path(args.tarball).resolve())
    print(f"gpu_experiment_bundle={bundle_dir}")
    if not args.no_tarball:
        print(f"gpu_experiment_tarball={Path(args.tarball).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
