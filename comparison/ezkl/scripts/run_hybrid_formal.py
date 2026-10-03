#!/usr/bin/env python3
"""Run the baseline-local ezkl formal chain for all five datasets.

The command authenticates the prepared five-dataset resources and official
native ``get-srs`` cache, reuses a source-matched GPU-enabled build or builds it once,
and then runs setup, independent production,
final binding, witness, prove, independent verify, collection and validation.
No SRS path, digest, or other ceremony input is accepted from the operator.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from formal_common import (
    CONTRACT,
    DATASETS,
    FORMAL_BACKEND,
    NATIVE_FORMAL_BACKEND,
    ROOT,
    ContractError,
    abort,
    local_path,
    native_backend_identity,
    load_cli_contract,
    load_json,
    parse_cfg,
    print_error,
    require_file,
    resource_manifest_hash,
    sha256,
    source_tree_hash,
    validate_config,
    validate_native_srs_manifest,
    validate_gpu_environment,
    validate_resources,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary")
    result.add_argument("--output-root", default="runs/formal")
    result.add_argument("--matrix", default="results/formal_matrix.json")
    return result


def run(command: list[str], label: str, *, gpu: bool = False) -> float:
    print(f"[ezkl-formal] {label}: {' '.join(command)}", flush=True)
    env = os.environ.copy()
    for key in ("ENABLE_ICICLE_GPU", "ICICLE_SMALL_K", "ICICLE_SMALL_K_FFT"): env.pop(key, None)
    if gpu:
        env["ENABLE_ICICLE_GPU"] = "1"
        env["EZKL_GAT_BACKEND"] = "gpu"
        env["ICICLE_SMALL_K"] = "2"
        env["ICICLE_SMALL_K_FFT"] = "2"
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode != 0:
        abort(f"{label} failed with exit code {completed.returncode}")
    return (time.monotonic() - started) * 1000.0


def local_cli(path: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        abort(f"{label} is outside the ezkl baseline: {path}")


def reusable_build(binary: Path, backend: str) -> bool:
    receipt_path = ROOT / "artifacts" / "build" / backend / "build_receipt.json"
    if not binary.is_file() or binary.is_symlink() or not os.access(binary, os.X_OK):
        return False
    if not receipt_path.is_file() or receipt_path.is_symlink() or not CONTRACT.is_file():
        return False
    receipt = load_json(receipt_path, "native build receipt")
    contract = load_cli_contract()
    expected = {
        "backend": backend, "binary_path": binary.relative_to(ROOT).as_posix(), "binary_sha256": sha256(binary),
        "source_tree_sha256": source_tree_hash(), "resource_manifest_sha256": resource_manifest_hash(),
        "adapter_contract_id": contract["contract_id"], "adapter_contract_sha256": sha256(CONTRACT),
    }
    if backend == "gpu":
        expected.update(native_backend_identity())
    return all(receipt.get(key) == value for key, value in expected.items()) and receipt.get("features") == ["ezkl"] + (["gpu-accelerated"] if backend == "gpu" else [])


def check_output_paths(output_root: Path, matrix: Path, backend: str) -> None:
    for path, label in ((output_root, "output-root"), (matrix.parent, "matrix parent")):
        for parent in (path, *path.parents):
            if parent == ROOT.parent:
                break
            if parent.exists() and not parent.is_dir():
                abort(f"{label} has a non-directory path: {parent}")
    if matrix.exists() and not matrix.is_file():
        abort(f"matrix is not a regular file: {matrix}")
    for dataset in DATASETS:
        output = output_root / dataset / backend
        for path in (output_root / dataset, output, output_root / ".incomplete", output_root / ".incomplete" / dataset):
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                abort(f"formal output path is not a regular directory: {path}")
        result = output / "result.json"
        if result.exists() or result.is_symlink():
            abort(f"formal output already exists; refusing to mix runs: {output}")


def prepare_output_dir(output_root: Path, dataset: str, backend: str) -> Path:
    output = output_root / dataset / backend
    if not (output.exists() or output.is_symlink()):
        output.mkdir(parents=True)
        return output
    if output.is_symlink() or not output.is_dir():
        abort(f"formal output is not a directory or is symlinked: {output}")
    result = output / "result.json"
    if result.exists() or result.is_symlink():
        abort(f"formal output already exists; refusing to mix runs: {output}")

    incomplete = output_root / ".incomplete"
    if incomplete.exists() or incomplete.is_symlink():
        if incomplete.is_symlink() or not incomplete.is_dir():
            abort(f"incomplete archive root is not a directory or is symlinked: {incomplete}")
    else:
        incomplete.mkdir()
    dataset_archive = incomplete / dataset
    if dataset_archive.exists() or dataset_archive.is_symlink():
        if dataset_archive.is_symlink() or not dataset_archive.is_dir():
            abort(f"incomplete archive dataset path is not a directory or is symlinked: {dataset_archive}")
    else:
        dataset_archive.mkdir()
    archive = dataset_archive / datetime.now(timezone.utc).strftime(f"{backend}-%Y%m%dT%H%M%S%fZ")
    if archive.exists() or archive.is_symlink():
        abort(f"incomplete archive already exists; refusing to overwrite: {archive}")
    output.rename(archive)
    print(f"[ezkl-formal] archived incomplete formal output: {output} -> {archive}")
    output.mkdir(parents=True)
    return output


def main(args: argparse.Namespace) -> int:
    binary = local_path(args.binary or f"target/{NATIVE_FORMAL_BACKEND}/release/ezkl", "binary")
    output_root = local_path(args.output_root, "output-root")
    matrix = local_path(args.matrix, "matrix")
    if binary.exists() and binary.is_symlink():
        abort(f"binary must not be a symlink: {binary}")
    if output_root.exists() and output_root.is_symlink():
        abort(f"output-root must not be a symlink: {output_root}")
    if matrix.exists() and matrix.is_symlink():
        abort(f"matrix must not be a symlink: {matrix}")

    python = sys.executable
    resource_validator = ROOT / "scripts" / "validate_resources.py"
    build = ROOT / "scripts" / "build_ezkl.sh"
    producer = ROOT / "scripts" / "produce_public_statement.py"
    single = ROOT / "scripts" / "run_single_dataset.py"
    collect = ROOT / "scripts" / "collect_results.py"
    validate = ROOT / "scripts" / "validate_final.py"
    for path in (resource_validator, build, producer, single, collect, validate):
        if not path.is_file():
            abort(f"missing formal runner component: {path}")

    check_output_paths(output_root, matrix, FORMAL_BACKEND)
    try:
        importlib.import_module("numpy")
    except (ImportError, OSError) as exc:
        abort(f"independent public producer requires a working numpy installation: {exc}")
    validate_resources()
    require_file(ROOT / "resources/canonical/configs/quantization_scale_chain.cfg", "canonical scale chain")
    for dataset in DATASETS:
        values = validate_config(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg", dataset)
        require_file(ROOT / values["checkpoint_bundle"] / "tensors.txt", f"{dataset} canonical producer tensors")
        data_dir = ROOT / values["cache_root"] / dataset
        meta = parse_cfg(require_file(data_dir / "meta.cfg", f"{dataset} canonical graph metadata"))
        require_file(data_dir / "edges.txt", f"{dataset} canonical edges")
        features = data_dir / "features.npy"
        require_file(features if features.exists() else data_dir / "features.txt", f"{dataset} canonical features")
        try:
            graph_count = int(meta.get("graph_count", "1"))
        except ValueError:
            abort(f"{dataset} graph_count is not an integer")
        if graph_count <= 0:
            abort(f"{dataset} graph_count must be positive")
        if graph_count > 1:
            require_file(data_dir / "node_ptr.txt", f"{dataset} canonical graph boundaries")
    print("[ezkl-formal] authenticated prepared five-dataset resources/configs", flush=True)
    try:
        srs_manifest = validate_native_srs_manifest()
    except ContractError:
        run([python, "scripts/provision_srs.py"], "provision source-authenticated official SRS")
        srs_manifest = validate_native_srs_manifest()
    entries = srs_manifest["entries"]
    shared_paths = {entry["path"] for entry in entries.values()}
    if len(shared_paths) != 1:
        abort("native SRS manifest does not expose one shared official cache")
    srs = ROOT / next(iter(shared_paths))
    if not srs.is_file() or srs.is_symlink():
        abort(f"shared official SRS is missing or symlinked: {srs}")
    print(f"[ezkl-formal] reuse authenticated official SRS: {local_cli(srs, 'SRS')}", flush=True)
    if NATIVE_FORMAL_BACKEND == "gpu":
        run(["bash", "setup-gpu.sh", "--yes"], "prepare source-matched native ICICLE CUDA libraries")
        validate_gpu_environment()
    if not reusable_build(binary, NATIVE_FORMAL_BACKEND):
        if binary != ROOT / "target" / NATIVE_FORMAL_BACKEND / "release/ezkl":
            abort(f"stale custom --binary cannot be rebuilt; use target/{NATIVE_FORMAL_BACKEND}/release/ezkl")
        run(["bash", local_cli(build, "build script"), f"--{NATIVE_FORMAL_BACKEND}"], "build source-matched official ezkl native binary")
        if not reusable_build(binary, NATIVE_FORMAL_BACKEND):
            abort("native build did not produce a matching binary/resource/adapter receipt")
    else:
        print("[ezkl-formal] reuse source-matched official ezkl native build", flush=True)

    for dataset in DATASETS:
        output = prepare_output_dir(output_root, dataset, FORMAL_BACKEND)
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
        if NATIVE_FORMAL_BACKEND == "gpu":
            setup.append("--gpu")
        setup_ms = run(setup, f"{dataset} native setup", gpu=NATIVE_FORMAL_BACKEND == "gpu")
        (output / "setup_timing.json").write_text(json.dumps({"duration_ms": setup_ms}) + "\n")
        run(
            [
                python,
                local_cli(producer, "independent producer"),
                "--dataset",
                dataset,
                "--resource-manifest",
                "resources/resource_manifest.json",
                "--setup-context",
                local_cli(output / "producer_context.json", "producer setup context"),
                "--output-dir",
                local_cli(output, "output-dir"),
                "--config",
                local_cli(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg", "config"),
                "--statement",
                local_cli(output / "statement.json", "public statement"),
                "--public-output",
                local_cli(output / "public_output.json", "public output"),
                "--producer-manifest",
                local_cli(output / "producer_manifest.json", "producer manifest"),
            ],
            f"{dataset} independent public statement/Y producer",
        )
        run(
            [
                python,
                local_cli(single, "single formal runner"),
                "--dataset",
                dataset,
                "--backend",
                NATIVE_FORMAL_BACKEND,
                "--execute-formal",
                "--reuse-setup",
                "--config",
                local_cli(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg", "config"),
                "--public-statement",
                local_cli(output / "statement.json", "public statement"),
                "--public-output",
                local_cli(output / "public_output.json", "public output"),
                "--producer-manifest",
                local_cli(output / "producer_manifest.json", "producer manifest"),
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
        ],
        "collect five-dataset result matrix",
    )
    run(
        [
            python,
            local_cli(validate, "result validator"),
            "--matrix",
            local_cli(matrix, "matrix"),
        ],
        "validate five-dataset result matrix",
    )
    print(f"validated {FORMAL_BACKEND} formal results: {matrix.relative_to(ROOT)}")
    print(f"view: python3 -m json.tool {matrix.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise SystemExit(print_error(exc))
