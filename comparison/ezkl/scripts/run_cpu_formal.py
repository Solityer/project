#!/usr/bin/env python3
"""Run the baseline-local ezkl CPU formal chain for all five datasets.

The command owns the complete preparation order.  It regenerates and gates the
five canonical resources, builds the current CPU binary, provisions one
source-matched official SRS through native ``get-srs`` (the native loader
downsizes it for each circuit), and then runs setup, independent production,
final binding, witness, prove, independent verify, collection and validation.
No SRS path, digest, or other ceremony input is accepted from the operator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from formal_common import (
    DATASETS,
    ROOT,
    ContractError,
    abort,
    local_path,
    print_error,
    validate_native_srs_manifest,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary", default="target/release/ezkl")
    result.add_argument("--output-root", default="runs/formal")
    result.add_argument("--matrix", default="results/formal_matrix.json")
    return result


def run(command: list[str], label: str) -> None:
    print(f"[ezkl-cpu] {label}: {' '.join(command)}")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        abort(f"{label} failed with exit code {completed.returncode}")


def local_cli(path: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        abort(f"{label} is outside the ezkl baseline: {path}")


def main(args: argparse.Namespace) -> int:
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
    resource_validator = ROOT / "scripts" / "validate_resources.py"
    build = ROOT / "scripts" / "build_ezkl.sh"
    single = ROOT / "scripts" / "run_single_dataset.py"
    collect = ROOT / "scripts" / "collect_results.py"
    validate = ROOT / "scripts" / "validate_final.py"
    for path in (provision, prepare, resource_validator, build, single, collect, validate):
        if not path.is_file():
            abort(f"missing formal runner component: {path}")

    # This prepares all five exact tables/ledgers, including the OGBN canonical
    # table/ledger, and writes a fresh authenticated resource manifest.
    run(
        [python, local_cli(prepare, "resource preparer"), "--commit-manifest"],
        "prepare five-dataset canonical resources",
    )
    run(
        [python, local_cli(resource_validator, "resource validator"), "--require-complete"],
        "require complete authenticated resources",
    )
    run([local_cli(build, "build script"), "--cpu"], "build official ezkl CPU binary")

    # The native binary is now source-matched to the code used by the build.
    # provision_srs.py parses the native PUBLIC_SRS_SHA256_HASHES table, uses
    # its maximum supported k, and updates all five secure configs atomically.
    run(
        [python, local_cli(provision, "SRS provisioner"), "--binary", local_cli(binary, "binary")],
        "provision official SRS through native get-srs",
    )
    srs_manifest = validate_native_srs_manifest()
    entries = srs_manifest["entries"]
    shared_paths = {entry["path"] for entry in entries.values()}
    if len(shared_paths) != 1:
        abort("native SRS manifest does not expose one shared official cache")
    srs = ROOT / next(iter(shared_paths))
    if not srs.is_file() or srs.is_symlink():
        abort(f"shared official SRS is missing or symlinked: {srs}")

    for dataset in DATASETS:
        output = output_root / dataset / "cpu"
        if output.exists() or output.is_symlink():
            abort(f"formal output already exists; refusing to mix runs: {output}")
        output.mkdir(parents=True)
        setup = [
            local_cli(binary, "binary"),
            "gat-prepare",
            "--dataset",
            dataset,
            "--phase",
            "setup",
            "--resource-manifest",
            "resources/resource_manifest.json",
            "--output-dir",
            local_cli(output, "output-dir"),
            "--srs-path",
            local_cli(srs, "SRS"),
        ]
        run(setup, f"{dataset} native setup")
        run(
            [
                local_cli(binary, "binary"),
                "gat-produce",
                "--dataset",
                dataset,
                "--resource-manifest",
                "resources/resource_manifest.json",
                "--output-dir",
                local_cli(output, "output-dir"),
            ],
            f"{dataset} independent gat-produce",
        )
        run(
            [
                python,
                local_cli(single, "single formal runner"),
                "--dataset",
                dataset,
                "--backend",
                "cpu",
                "--execute-formal",
                "--reuse-setup",
                "--config",
                local_cli(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg", "config"),
                "--public-statement",
                local_cli(output / "statement.json", "public statement"),
                "--public-output",
                local_cli(output / "public_output.json", "public output"),
                "--binary",
                local_cli(binary, "binary"),
                "--output-dir",
                local_cli(output, "output-dir"),
            ],
            f"{dataset} final/witness/prove/independent verify",
        )

    run(
        [
            python,
            local_cli(collect, "result collector"),
            "--runs-root",
            local_cli(output_root, "output-root"),
            "--output",
            local_cli(matrix, "matrix"),
            "--backends",
            "cpu",
        ],
        "collect CPU result matrix",
    )
    run(
        [
            python,
            local_cli(validate, "result validator"),
            "--matrix",
            local_cli(matrix, "matrix"),
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
