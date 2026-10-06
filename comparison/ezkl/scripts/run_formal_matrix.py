#!/usr/bin/env python3
"""The production entry for five full-dataset EZKL/Halo2 results."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def restart_without_inherited_preload() -> None:
    """Start a clean interpreter before formal environment validation."""
    if not os.environ.get("LD_PRELOAD"):
        return
    clean_environment = dict(os.environ)
    clean_environment.pop("LD_PRELOAD", None)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_ld-preload-cleared"]
    os.execve(sys.executable, command, clean_environment)


restart_without_inherited_preload()

from formal_common import (
    ROOT,
    ContractError,
    abort,
    local_path,
    print_error,
    validate_root_owned_phase,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute-formal", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--binary", default="target/gpu/release/ezkl")
    result.add_argument("--output-root", default="runs/formal")
    result.add_argument("--matrix", default="results/formal_matrix.json")
    result.add_argument("--internal-phase", choices=("prepare", "cell", "finalize"), help=argparse.SUPPRESS)
    result.add_argument("--session-dir", help=argparse.SUPPRESS)
    result.add_argument("--dataset", help=argparse.SUPPRESS)
    result.add_argument("--_ld-preload-cleared", action="store_true", help=argparse.SUPPRESS)
    return result


def main(args: argparse.Namespace) -> int:
    if args.execute_formal and args.dry_run:
        abort("--dry-run cannot be combined with --execute-formal")
    internal_options = (args.internal_phase is not None, args.session_dir is not None, args.dataset is not None)
    if any(internal_options) and not (all(internal_options[:2]) and (args.dataset is not None or args.internal_phase != "cell")):
        abort("internal formal phases require --internal-phase and --session-dir; --dataset is required only for cell")
    if any(internal_options) and not args.execute_formal:
        abort("internal formal phases require --execute-formal")
    if args.internal_phase == "cell" and args.dataset is None:
        abort("internal cell phase requires --dataset")
    if args.internal_phase != "cell" and args.dataset is not None:
        abort("--dataset is valid only for internal cell phase")
    if args.execute_formal and args.internal_phase is None:
        abort("direct EZKL full-matrix execution is disabled; use ./comparison/run_formal.sh")
    if args.internal_phase is not None:
        session_dir = Path(args.session_dir)
        session_id = session_dir.parent.name
        validate_root_owned_phase(session_dir, session_id, args.internal_phase, args.dataset)
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
    if args.internal_phase is not None:
        command.extend(["--internal-phase", args.internal_phase, "--session-dir", args.session_dir])
        if args.dataset is not None:
            command.extend(["--dataset", args.dataset])
    if not args.execute_formal:
        print("ezkl formal entry point: no setup/prove/verify was run")
        print("run the repository's single official comparison command:")
        print("./comparison/run_formal.sh")
        return 0
    os.chdir(ROOT)
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError) as exc:
        raise SystemExit(print_error(exc))
