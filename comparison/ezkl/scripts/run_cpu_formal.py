#!/usr/bin/env python3
"""Run the single baseline-local ezkl CPU formal chain.

The operator supplies one authentic official Halo2 KZG SRS and its SHA-256
per dataset.  This entry point stages those files, regenerates the exact
canonical resources (including the large ogbn-arxiv table/ledger), builds the
official CPU binary, runs all five setup/produce/final/witness/prove/verify
cells, then collects and validates the CPU result matrix.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from formal_common import DATASETS, ROOT, ContractError, abort, local_path, print_error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--srs-source",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help="operator-supplied official SRS source; repeat once per dataset",
    )
    result.add_argument(
        "--srs-sha256",
        action="append",
        required=True,
        metavar="DATASET=SHA256",
        help="lowercase SHA-256 for the matching SRS source; repeat once per dataset",
    )
    result.add_argument("--binary", default="target/release/ezkl")
    result.add_argument("--output-root", default="runs/formal")
    result.add_argument("--matrix", default="results/formal_matrix.json")
    return result


def keyed(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        dataset, separator, payload = value.partition("=")
        if not separator or dataset not in DATASETS or not payload or dataset in result:
            abort(f"{label} must contain each dataset exactly once as DATASET=VALUE")
        result[dataset] = payload
    if set(result) != set(DATASETS):
        abort(f"{label} must contain exactly {list(DATASETS)}")
    return result


def run(command: list[str], label: str) -> None:
    print(f"[ezkl-cpu] {label}: {' '.join(command)}")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        abort(f"{label} failed with exit code {completed.returncode}")


def local_cli(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        abort(f"formal path is outside the ezkl baseline: {path}")


def main(args: argparse.Namespace) -> int:
    sources = keyed(args.srs_source, "--srs-source")
    digests = keyed(args.srs_sha256, "--srs-sha256")
    for dataset, value in digests.items():
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            abort(f"--srs-sha256 {dataset} must be a lowercase SHA-256 digest")

    binary = local_path(args.binary, "binary")
    output_root = local_path(args.output_root, "output-root")
    matrix = local_path(args.matrix, "matrix")
    if binary.exists() and binary.is_symlink():
        abort(f"binary must not be a symlink: {binary}")
    if output_root.exists() and output_root.is_symlink():
        abort(f"output-root must not be a symlink: {output_root}")
    if matrix.exists() and matrix.is_symlink():
        abort(f"matrix must not be a symlink: {matrix}")

    python = sys.executable
    provision = ROOT / "scripts" / "provision_srs.py"
    prepare = ROOT / "scripts" / "prepare_gat_resources.py"
    build = ROOT / "scripts" / "build_ezkl.sh"
    single = ROOT / "scripts" / "run_single_dataset.py"
    collect = ROOT / "scripts" / "collect_results.py"
    validate = ROOT / "scripts" / "validate_final.py"
    for path in (provision, prepare, build, single, collect, validate):
        if not path.is_file():
            abort(f"missing formal runner component: {path}")

    # Authenticate and stage each external SRS before build/config validation.
    for dataset in DATASETS:
        source = Path(sources[dataset]).expanduser()
        if not source.is_absolute():
            source = (Path.cwd() / source).resolve()
        run(
            [
                python,
                local_cli(provision),
                "--dataset",
                dataset,
                "--source",
                str(source),
                "--sha256",
                digests[dataset],
                "--replace",
            ],
            f"provision {dataset} official SRS",
        )

    # This is the only resource-preparation entry point.  It regenerates all
    # five exact tables/ledgers and atomically refreshes the authenticated
    # manifest; no OGBN generation is performed by the development checks.
    run([python, local_cli(prepare), "--commit-manifest"], "prepare authenticated five-dataset resources")
    run([build, "--cpu"], "build official ezkl CPU binary")

    for dataset in DATASETS:
        output = output_root / dataset / "cpu"
        srs = ROOT / "resources" / "native_srs" / f"{dataset}_formal_srs"
        if output.exists() or output.is_symlink():
            abort(f"formal output already exists; refusing to mix runs: {output}")
        output.mkdir(parents=True)
        setup = [
            local_cli(binary),
            "gat-prepare",
            "--dataset",
            dataset,
            "--phase",
            "setup",
            "--resource-manifest",
            "resources/resource_manifest.json",
            "--output-dir",
            local_cli(output),
            "--srs-path",
            local_cli(srs),
        ]
        run(setup, f"{dataset} native setup")
        run(
            [
                local_cli(binary),
                "gat-produce",
                "--dataset",
                dataset,
                "--resource-manifest",
                "resources/resource_manifest.json",
                "--output-dir",
                local_cli(output),
            ],
            f"{dataset} independent gat-produce",
        )
        run(
            [
                python,
                local_cli(single),
                "--dataset",
                dataset,
                "--backend",
                "cpu",
                "--execute-formal",
                "--reuse-setup",
                "--config",
                local_cli(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg"),
                "--public-statement",
                local_cli(output / "statement.json"),
                "--public-output",
                local_cli(output / "public_output.json"),
                "--srs",
                local_cli(srs),
                "--srs-sha256",
                digests[dataset],
                "--binary",
                local_cli(binary),
                "--output-dir",
                local_cli(output),
            ],
            f"{dataset} final/witness/prove/independent verify",
        )

    run(
        [
            python,
            local_cli(collect),
            "--runs-root",
            local_cli(output_root),
            "--output",
            local_cli(matrix),
            "--backends",
            "cpu",
        ],
        "collect CPU result matrix",
    )
    run(
        [
            python,
            local_cli(validate),
            "--matrix",
            local_cli(matrix),
            "--backends",
            "cpu",
        ],
        "validate CPU result matrix",
    )
    print(f"validated CPU matrix: {matrix.relative_to(ROOT)}")
    print(f"view: python3 -m json.tool {matrix.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise SystemExit(print_error(exc))
