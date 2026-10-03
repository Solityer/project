#!/usr/bin/env python3
"""The production entry for five hybrid CPU+GPU full-dataset results."""

from __future__ import annotations

import argparse
import subprocess
import sys

from formal_common import ROOT, ContractError, abort, local_path, print_error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute-formal", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--binary", default="target/gpu/release/ezkl")
    result.add_argument("--output-root", default="runs/formal")
    result.add_argument("--matrix", default="results/formal_matrix.json")
    return result


def main(args: argparse.Namespace) -> int:
    if args.execute_formal and args.dry_run:
        abort("--dry-run cannot be combined with --execute-formal")
    command = [
        sys.executable,
        "scripts/run_hybrid_formal.py",
        "--binary",
        args.binary,
        "--output-root",
        args.output_root,
        "--matrix",
        args.matrix,
    ]
    if not args.execute_formal:
        print("ezkl formal entry point: no setup/prove/verify was run")
        print("run the following command to execute the five-dataset hybrid CPU+GPU chain:")
        print(" ".join([sys.executable, "scripts/run_formal_matrix.py", "--execute-formal", "--binary", args.binary, "--output-root", args.output_root, "--matrix", args.matrix]))
        return 0
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        abort(f"hybrid formal runner failed with exit code {completed.returncode}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(print_error(exc))
