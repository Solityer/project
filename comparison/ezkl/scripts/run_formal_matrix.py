#!/usr/bin/env python3
"""The baseline-local production entry for five EZKL/Halo2 dataset results."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def restart_without_inherited_preload() -> None:
    """Start a clean interpreter before importing or launching formal code."""
    if not os.environ.get("LD_PRELOAD"):
        return
    clean_environment = dict(os.environ)
    clean_environment.pop("LD_PRELOAD", None)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_ld-preload-cleared"]
    os.execve(sys.executable, command, clean_environment)


restart_without_inherited_preload()

from formal_common import ContractError, ROOT, abort, print_error  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute-formal", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--preflight-only", action="store_true", help="authenticate prerequisites without setup or proof")
    result.add_argument("--binary", default="target/gpu/release/ezkl")
    result.add_argument("--output-root", default="runs/formal")
    result.add_argument("--matrix", default="results/formal_matrix.json")
    result.add_argument("--_ld-preload-cleared", action="store_true", help=argparse.SUPPRESS)
    return result


def main(args: argparse.Namespace) -> int:
    if args.dry_run and (args.execute_formal or args.preflight_only):
        abort("--dry-run cannot be combined with --execute-formal or --preflight-only")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_hybrid_formal.py"),
        "--binary",
        args.binary,
        "--output-root",
        args.output_root,
        "--matrix",
        args.matrix,
        "--_from-ezkl-entry",
    ]
    if args.preflight_only:
        command.append("--preflight-only")
    if not args.execute_formal and not args.preflight_only:
        print("five-dataset EZKL/Halo2 formal chain: no setup/prove/verify was run")
        print("the EZKL-local production command is:")
        print("python3 scripts/run_formal_matrix.py --execute-formal")
        return 0
    os.chdir(ROOT)
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError) as exc:
        raise SystemExit(print_error(exc))
