#!/usr/bin/env python3
"""Write the authenticated baseline-local CUDA build receipt."""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.build_identity import write_build_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the zkGPT CUDA build manifest")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--nvcc", required=True)
    parser.add_argument("--target-compute-capability", required=True)
    parser.add_argument("--arch-flag", required=True)
    # This option is deliberately last: nvcc accepts single-dash flags which
    # argparse would otherwise treat as unknown options.
    parser.add_argument("--compile-command", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args(argv)
    root = pathlib.Path(args.root)
    if root.expanduser().absolute() != ROOT.absolute():
        raise SystemExit(f"--root must equal canonical zkGPT baseline root: {ROOT}")
    manifest = write_build_manifest(
        root,
        binary=args.binary,
        nvcc=args.nvcc,
        target_compute_capability=args.target_compute_capability,
        arch_flag=args.arch_flag,
        compile_command=args.compile_command,
    )
    print(
        "zkGPT_cuda_build_manifest="
        f"{manifest['manifest_digest']} binary_sha256={manifest['binary']['sha256']} "
        f"source_manifest_sha256={manifest['source']['manifest_sha256']} "
        f"target={manifest['target']['code']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
