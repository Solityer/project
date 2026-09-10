#!/usr/bin/env python3
"""Plan or execute the user-owned five-dataset ezkl formal matrix.

The default mode is a non-executing plan.  Full-dataset work is reachable only
with the explicit ``--execute-formal`` switch and requires baseline-local
external public statements, public logits, authentic SRS files, and their
operator-supplied SHA-256 digests.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from formal_common import (
    BACKENDS,
    DATASETS,
    ROOT,
    ContractError,
    abort,
    local_path,
    print_error,
    reject_forbidden_path,
    reject_reduced_srs,
    require_digest,
    require_file,
    require_formal_srs,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute-formal", action="store_true", help="execute all ten user-owned formal cells")
    result.add_argument("--dry-run", action="store_true", help="print the matrix plan (the default)")
    result.add_argument("--runner", default="scripts/run_single_dataset.py")
    result.add_argument("--public-root", help="baseline-local directory containing <dataset>.statement.json and <dataset>.Y.json")
    result.add_argument("--srs-root", help="baseline-local directory containing <dataset>_formal_srs")
    result.add_argument("--srs-digests", help="baseline-local JSON object mapping dataset to trusted SRS SHA-256")
    result.add_argument("--binary-cpu", default="target/release/ezkl")
    result.add_argument("--binary-gpu", default="target/release/ezkl")
    result.add_argument("--config-root", default="resources/canonical/configs")
    result.add_argument("--output-root", default="runs/formal")
    return result


def path_arg(value: str, label: str) -> Path:
    path = local_path(value, label)
    reject_forbidden_path(path, label)
    return path


def load_digests(path: Path) -> dict[str, str]:
    require_file(path, "trusted SRS digest map")
    try:
        value: Any = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        abort(f"trusted SRS digest map is invalid JSON: {exc}")
    if not isinstance(value, dict) or set(value) != set(DATASETS):
        abort(f"trusted SRS digest map must contain exactly {list(DATASETS)}")
    return {dataset: require_digest(value[dataset], f"SRS digest for {dataset}") for dataset in DATASETS}


def cell_paths(public_root: Path, srs_root: Path, dataset: str) -> tuple[Path, Path, Path]:
    statement = public_root / f"{dataset}.statement.json"
    public_output = public_root / f"{dataset}.Y.json"
    srs = srs_root / f"{dataset}_formal_srs"
    for path, label in (
        (statement, f"{dataset} public statement"),
        (public_output, f"{dataset} public output"),
    ):
        reject_forbidden_path(path, label)
        require_file(path, label)
    reject_forbidden_path(srs, f"{dataset} SRS")
    require_formal_srs(srs, f"{dataset} trusted official ezkl SRS")
    return statement, public_output, srs


def command_for(
    runner: Path,
    dataset: str,
    backend: str,
    *,
    config: Path,
    statement: Path,
    public_output: Path,
    srs: Path,
    srs_digest: str | None,
    binary: Path,
    output: Path,
    execute: bool,
) -> list[str]:
    def cli_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            # Dry-run placeholders are intentionally not local filesystem paths.
            return str(path)

    command = [
        sys.executable,
        str(runner),
        "--dataset",
        dataset,
        "--backend",
        backend,
        "--config",
        cli_path(config),
        "--public-statement",
        cli_path(statement),
        "--public-output",
        cli_path(public_output),
        "--srs",
        cli_path(srs),
        "--binary",
        cli_path(binary),
        "--output-dir",
        cli_path(output),
    ]
    if srs_digest is not None:
        insert_at = command.index("--binary")
        command[insert_at:insert_at] = ["--srs-sha256", srs_digest]
    if execute:
        command.append("--execute-formal")
    else:
        command.append("--dry-run")
    return command


def main(args: argparse.Namespace) -> int:
    reject_reduced_srs(sys.argv[1:])
    if args.execute_formal and args.dry_run:
        abort("--dry-run cannot be combined with --execute-formal")
    execute = bool(args.execute_formal)
    runner = path_arg(args.runner, "runner")
    require_file(runner, "formal single-cell runner")
    config_root = path_arg(args.config_root, "config-root")
    output_root = path_arg(args.output_root, "output-root")

    print("ezkl formal matrix plan: 5 datasets x 2 backends")
    print("execution=" + ("requested" if execute else "dry-run; no compile/setup/prove/verify was run"))
    if not execute:
        print("execute mode requires baseline-local --public-root, --srs-root and --srs-digests with real operator-provided SHA-256 values")

    if not execute:
        for dataset in DATASETS:
            config = config_root / f"{dataset}_secure.cfg"
            for backend in BACKENDS:
                print(
                    " ".join(
                        command_for(
                            runner,
                            dataset,
                            backend,
                            config=config,
                            statement=Path(f"<public-root>/{dataset}.statement.json"),
                            public_output=Path(f"<public-root>/{dataset}.Y.json"),
                            srs=Path(f"<srs-root>/{dataset}_formal_srs"),
                            srs_digest=None,
                            binary=Path(args.binary_gpu if backend == "gpu" else args.binary_cpu),
                            output=output_root / dataset / backend,
                            execute=False,
                        )
                    )
                )
        return 0

    if not args.public_root or not args.srs_root or not args.srs_digests:
        abort("--execute-formal requires --public-root, --srs-root and --srs-digests")
    public_root = path_arg(args.public_root, "public-root")
    srs_root = path_arg(args.srs_root, "srs-root")
    digest_map = load_digests(path_arg(args.srs_digests, "srs-digests"))
    binaries = {"cpu": path_arg(args.binary_cpu, "binary-cpu"), "gpu": path_arg(args.binary_gpu, "binary-gpu")}
    for backend, binary in binaries.items():
        require_file(binary, f"{backend} official ezkl binary")
    cells: list[tuple[str, str, list[str]]] = []
    for dataset in DATASETS:
        statement, public_output, srs = cell_paths(public_root, srs_root, dataset)
        config = config_root / f"{dataset}_secure.cfg"
        reject_forbidden_path(config, f"{dataset} config")
        require_file(config, f"{dataset} canonical secure config")
        for backend in BACKENDS:
            command = command_for(
                runner,
                dataset,
                backend,
                config=config,
                statement=statement,
                public_output=public_output,
                srs=srs,
                srs_digest=digest_map[dataset],
                binary=binaries[backend],
                output=output_root / dataset / backend,
                execute=True,
            )
            cells.append((dataset, backend, command))

    for dataset, backend, command in cells:
        print(f"executing dataset={dataset} backend={backend}")
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            abort(f"formal cell failed closed: dataset={dataset} backend={backend} exit={completed.returncode}")
    print("formal matrix commands completed; collect and validate the produced artifacts separately")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise SystemExit(print_error(exc))
