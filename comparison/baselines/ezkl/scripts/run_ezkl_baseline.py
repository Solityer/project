#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_full_dataset import export_dataset, load_dataset_specs, load_routes


ALLOWED_DATASETS = ["cora_full", "citeseer_full", "pubmed_full", "ppi_full_formal", "ogbn_arxiv_full"]
FORBIDDEN_MODES = {"smoke", "toy", "subset", "subgraph", "scaled", "shard", "debug"}
DEFAULT_ROUTE = json.loads((BASE_DIR / "configs" / "ezkl_routes.json").read_text())["default_route"]


class StepError(RuntimeError):
    def __init__(self, step: str, record: dict[str, Any]) -> None:
        super().__init__(f"{step} failed")
        self.step = step
        self.record = record


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def artifact_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def find_ezkl_binary() -> Path:
    env_bin = os.environ.get("EZKL_BIN")
    candidates = []
    if env_bin:
        candidates.append(Path(env_bin))
    candidates.append(BASE_DIR / "target" / "release" / "ezkl")
    path_bin = shutil.which("ezkl")
    if path_bin:
        candidates.append(Path(path_bin))
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError("ezkl binary not found; set EZKL_BIN or build comparison/baselines/ezkl/target/release/ezkl")


def ceil_log2(value: int) -> int:
    return int(math.ceil(math.log2(max(1, value))))


def command_record(command: list[str], cwd: Path, artifact_paths: list[Path]) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    duration_ms = (time.perf_counter() - started) * 1000.0
    record = {
        "duration_ms": duration_ms,
        "command": command,
        "exit_code": result.returncode,
        "status": "ok" if result.returncode == 0 else "failed",
        "artifacts": {path.name: artifact_size(path) for path in artifact_paths},
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    if result.returncode != 0:
        record["signal"] = -result.returncode if result.returncode < 0 else None
    return record


def run_step(
    *,
    step: str,
    command: list[str],
    cwd: Path,
    artifact_paths: list[Path],
    resume: bool,
    required_existing: list[Path] | None = None,
) -> dict[str, Any]:
    required = required_existing if required_existing is not None else artifact_paths
    if resume and required and all(path.exists() for path in required):
        return {
            "duration_ms": 0.0,
            "command": command,
            "exit_code": 0,
            "status": "resumed",
            "artifacts": {path.name: artifact_size(path) for path in artifact_paths},
            "stdout_tail": "",
            "stderr_tail": "",
        }
    record = command_record(command, cwd, artifact_paths)
    if record["exit_code"] != 0:
        raise StepError(step, record)
    return record


def clean_run_dir(run_dir: Path) -> None:
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)


def classify_failure(step: str, record: dict[str, Any]) -> tuple[str, str]:
    text = (record.get("stdout_tail", "") + "\n" + record.get("stderr_tail", "")).lower()
    signal = record.get("signal")
    if signal in {9, 6} or "out of memory" in text or "cannot allocate memory" in text or "killed" in text:
        return "blocked", f"{step}_resource_failure"
    return "failed", f"{step}_failed"


def manifest_from_steps(
    *,
    dataset_id: str,
    route_id: str,
    status: str,
    block_reason: str | None,
    verify_passed: bool,
    steps: dict[str, dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    model_info = load_json(run_dir / "model_info.json")
    settings = load_json(run_dir / "settings.json")
    run_args = settings.get("run_args", {})
    proof_path = run_dir / "proof.json"
    manifest = {
        "dataset": dataset_id,
        "full_dataset": True,
        "route_id": route_id,
        "status": status,
        "node_count": model_info.get("node_count"),
        "edge_count": model_info.get("edge_count"),
        "num_classes": model_info.get("num_classes"),
        "model_arch_id": model_info.get("model_arch_id"),
        "model_param_id": model_info.get("model_param_id"),
        "quant_cfg_id": model_info.get("quant_cfg_id"),
        "preprocess_id": model_info.get("preprocess_id"),
        "batching_rule": model_info.get("batching_rule"),
        "subgraph_rule": model_info.get("subgraph_rule"),
        "self_loop_rule": model_info.get("self_loop_rule"),
        "edge_sort_rule": model_info.get("edge_sort_rule"),
        "chunking_rule": model_info.get("chunking_rule"),
        "input_visibility": model_info.get("input_visibility"),
        "output_visibility": model_info.get("output_visibility"),
        "param_visibility": model_info.get("param_visibility"),
        "k": run_args.get("logrows"),
        "num_inner_cols": run_args.get("num_inner_cols"),
        "gen_settings_ms": steps.get("gen-settings", {}).get("duration_ms"),
        "compile_ms": steps.get("compile-circuit", {}).get("duration_ms"),
        "setup_ms": steps.get("setup", {}).get("duration_ms"),
        "gen_witness_ms": steps.get("gen-witness", {}).get("duration_ms"),
        "prove_time_ms": steps.get("prove", {}).get("duration_ms") if status == "verified" else None,
        "verify_time_ms": steps.get("verify", {}).get("duration_ms") if status == "verified" else None,
        "proof_size_bytes": artifact_size(proof_path) if status == "verified" else None,
        "proof_artifact_json_bytes": artifact_size(proof_path) if proof_path.exists() else None,
        "commitment_time_ms": None,
        "commitment_time_semantics": "included_in_ezkl_prove_ms_unless_instrumented",
        "verify_passed": verify_passed,
        "block_reason": block_reason,
        "statement": model_info.get("statement"),
        "strict_public_input_equivalent": model_info.get("strict_public_input_equivalent"),
        "fixed_artifact_hash": model_info.get("fixed_artifact_hash"),
        "public_output_hash": model_info.get("public_output_hash"),
        "onnx_graph_id": model_info.get("onnx_graph_id"),
        "artifacts": {
            "network.onnx": artifact_size(run_dir / "network.onnx"),
            "input.json": artifact_size(run_dir / "input.json"),
            "settings.json": artifact_size(run_dir / "settings.json"),
            "model.compiled": artifact_size(run_dir / "model.compiled"),
            "witness.json": artifact_size(run_dir / "witness.json"),
            "pk.key": artifact_size(run_dir / "pk.key"),
            "vk.key": artifact_size(run_dir / "vk.key"),
            "proof.json": artifact_size(run_dir / "proof.json"),
            "verify.json": artifact_size(run_dir / "verify.json"),
            "model_info.json": artifact_size(run_dir / "model_info.json"),
            "onnx_stats.json": artifact_size(run_dir / "onnx_stats.json"),
        },
        "steps": steps,
    }
    return manifest


def write_verify_json(run_dir: Path, record: dict[str, Any]) -> bool:
    passed = record["exit_code"] == 0
    payload = {
        "verify_passed": passed,
        "exit_code": record["exit_code"],
        "duration_ms": record["duration_ms"],
        "stdout_tail": record.get("stdout_tail", ""),
        "stderr_tail": record.get("stderr_tail", ""),
    }
    (run_dir / "verify.json").write_text(json.dumps(payload, indent=2))
    return passed


def validate_route(route_id: str) -> None:
    routes = load_routes()
    if route_id not in routes:
        known = ", ".join(sorted(routes))
        raise SystemExit(f"illegal route '{route_id}'. Allowed by note: {known}")
    if route_id != DEFAULT_ROUTE:
        raise SystemExit(f"route '{route_id}' is allowed by the note but is not the runnable full timing route")


def run_one(dataset_id: str, route_id: str, *, clean: bool, run: bool, resume: bool) -> dict[str, Any] | None:
    specs = load_dataset_specs()
    spec = specs[dataset_id]
    run_dir = spec.run_dir
    if clean:
        clean_run_dir(run_dir)
    if not run:
        print(json.dumps({"dataset": dataset_id, "cleaned": clean, "run_dir": run_dir.as_posix()}, indent=2))
        return None

    validate_route(route_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    steps: dict[str, dict[str, Any]] = {}
    manifest_path = run_dir / "run_manifest.json"
    if resume and manifest_path.exists():
        existing = load_json(manifest_path)
        expected_artifacts = [
            "network.onnx",
            "input.json",
            "settings.json",
            "model.compiled",
            "witness.json",
            "pk.key",
            "vk.key",
            "proof.json",
            "verify.json",
        ]
        if (
            existing.get("dataset") == dataset_id
            and existing.get("route_id") == route_id
            and existing.get("status") == "verified"
            and all((run_dir / name).exists() for name in expected_artifacts)
        ):
            print(json.dumps({"dataset": dataset_id, "status": "verified", "run_manifest": manifest_path.as_posix()}, indent=2))
            return existing

    try:
        export_started = time.perf_counter()
        if resume and (run_dir / "network.onnx").exists() and (run_dir / "input.json").exists() and (run_dir / "model_info.json").exists():
            export_info = load_json(run_dir / "model_info.json")
            export_status = "resumed"
            export_ms = 0.0
        else:
            export_info = export_dataset(dataset_id, route_id)
            export_status = "ok"
            export_ms = (time.perf_counter() - export_started) * 1000.0
        steps["export"] = {
            "duration_ms": export_ms,
            "command": [
                sys.executable,
                str(BASE_DIR / "scripts" / "export_full_dataset.py"),
                "--dataset",
                dataset_id,
                "--route",
                route_id,
            ],
            "exit_code": 0,
            "status": export_status,
            "artifacts": {
                "network.onnx": artifact_size(run_dir / "network.onnx"),
                "input.json": artifact_size(run_dir / "input.json"),
                "model_info.json": artifact_size(run_dir / "model_info.json"),
                "onnx_stats.json": artifact_size(run_dir / "onnx_stats.json"),
            },
        }

        ezkl = find_ezkl_binary()
        output_elements = int(export_info["node_count"]) * int(export_info["num_classes"])
        logrows = max(17, ceil_log2(output_elements + 64))
        srs_path = run_dir / f"kzg{logrows}.srs"
        settings_path = run_dir / "settings.json"
        model_path = run_dir / "network.onnx"
        compiled_path = run_dir / "model.compiled"
        witness_path = run_dir / "witness.json"
        pk_path = run_dir / "pk.key"
        vk_path = run_dir / "vk.key"
        proof_path = run_dir / "proof.json"

        steps["gen-settings"] = run_step(
            step="gen-settings",
            command=[
                str(ezkl),
                "gen-settings",
                "-M",
                str(model_path),
                "-O",
                str(settings_path),
                "-K",
                str(logrows),
                "-N",
                "2",
                "--input-visibility",
                "private",
                "--output-visibility",
                "public",
                "--param-visibility",
                "fixed",
                "--check-mode",
                "unsafe",
                "--ignore-range-check-inputs-outputs",
            ],
            cwd=run_dir,
            artifact_paths=[settings_path],
            resume=resume,
        )
        steps["compile-circuit"] = run_step(
            step="compile-circuit",
            command=[
                str(ezkl),
                "compile-circuit",
                "-M",
                str(model_path),
                "--compiled-circuit",
                str(compiled_path),
                "-S",
                str(settings_path),
            ],
            cwd=run_dir,
            artifact_paths=[compiled_path],
            resume=resume,
        )
        steps["get-srs"] = run_step(
            step="get-srs",
            command=[str(ezkl), "get-srs", "-S", str(settings_path), "--srs-path", str(srs_path)],
            cwd=run_dir,
            artifact_paths=[srs_path],
            resume=resume,
        )
        steps["setup"] = run_step(
            step="setup",
            command=[
                str(ezkl),
                "setup",
                "-M",
                str(compiled_path),
                "--srs-path",
                str(srs_path),
                "--vk-path",
                str(vk_path),
                "--pk-path",
                str(pk_path),
            ],
            cwd=run_dir,
            artifact_paths=[pk_path, vk_path],
            resume=resume,
        )
        steps["gen-witness"] = run_step(
            step="gen-witness",
            command=[
                str(ezkl),
                "gen-witness",
                "-D",
                str(run_dir / "input.json"),
                "-M",
                str(compiled_path),
                "-O",
                str(witness_path),
            ],
            cwd=run_dir,
            artifact_paths=[witness_path],
            resume=resume,
        )
        steps["prove"] = run_step(
            step="prove",
            command=[
                str(ezkl),
                "prove",
                "-W",
                str(witness_path),
                "-M",
                str(compiled_path),
                "--pk-path",
                str(pk_path),
                "--proof-path",
                str(proof_path),
                "--srs-path",
                str(srs_path),
                "--check-mode",
                "unsafe",
            ],
            cwd=run_dir,
            artifact_paths=[proof_path],
            resume=resume,
        )
        verify_record = run_step(
            step="verify",
            command=[
                str(ezkl),
                "verify",
                "-S",
                str(settings_path),
                "--proof-path",
                str(proof_path),
                "--vk-path",
                str(vk_path),
                "--srs-path",
                str(srs_path),
            ],
            cwd=run_dir,
            artifact_paths=[run_dir / "verify.json"],
            resume=False,
        )
        verify_passed = write_verify_json(run_dir, verify_record)
        steps["verify"] = verify_record
        status = "verified" if verify_passed else "failed"
        block_reason = None if verify_passed else "verify_failed"
    except StepError as error:
        steps[error.step] = error.record
        status, block_reason = classify_failure(error.step, error.record)
        verify_passed = False
    except Exception as error:
        status = "failed"
        block_reason = f"runner_exception:{type(error).__name__}:{error}"
        verify_passed = False

    manifest = manifest_from_steps(
        dataset_id=dataset_id,
        route_id=route_id,
        status=status,
        block_reason=block_reason,
        verify_passed=verify_passed,
        steps=steps,
        run_dir=run_dir,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"dataset": dataset_id, "status": status, "run_manifest": manifest_path.as_posix()}, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full-dataset ezkl baseline")
    parser.add_argument("--dataset", required=True, choices=[*ALLOWED_DATASETS, "all"])
    parser.add_argument("--route", default=DEFAULT_ROUTE)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.mode in FORBIDDEN_MODES or args.mode != "full":
        raise SystemExit("Only --mode full is allowed; smoke/toy/subset/subgraph/scaled/shard/debug routes are rejected")
    validate_route(args.route)
    datasets = ALLOWED_DATASETS if args.dataset == "all" else [args.dataset]
    for dataset_id in datasets:
        run_one(dataset_id, args.route, clean=args.clean, run=args.run, resume=args.resume)


if __name__ == "__main__":
    main()
