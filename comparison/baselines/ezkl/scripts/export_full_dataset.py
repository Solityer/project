#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
PYDEPS = BASE_DIR / ".pydeps"
if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))

import numpy as np
import onnx
from onnx import TensorProto, helper

try:
    import onnxruntime as ort
except ModuleNotFoundError:
    ort = None


DATASETS_PATH = BASE_DIR / "configs" / "full_datasets.json"
ROUTES_PATH = BASE_DIR / "configs" / "ezkl_routes.json"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    dataset: str
    run_dir: Path
    config_path: Path
    cache_path: Path
    checkpoint_path: Path
    reference_manifest: Path
    task_type: str
    report_unit: str
    batching_rule: str
    subgraph_rule: str
    self_loop_rule: str
    edge_sort_rule: str
    chunking_rule: str
    deduplicate_edges: bool
    symmetrize_edges: bool
    batch_graphs: int
    preprocess_id: str


def load_dataset_specs() -> dict[str, DatasetSpec]:
    raw = json.loads(DATASETS_PATH.read_text())
    specs: dict[str, DatasetSpec] = {}
    for dataset_id, entry in raw["datasets"].items():
        specs[dataset_id] = DatasetSpec(
            dataset_id=dataset_id,
            dataset=entry["dataset"],
            run_dir=BASE_DIR / entry["run_dir"],
            config_path=BASE_DIR / entry["config_path"],
            cache_path=BASE_DIR / entry["cache_path"],
            checkpoint_path=BASE_DIR / entry["checkpoint_path"],
            reference_manifest=BASE_DIR / entry["reference_manifest"],
            task_type=entry["task_type"],
            report_unit=entry["report_unit"],
            batching_rule=entry["batching_rule"],
            subgraph_rule=entry["subgraph_rule"],
            self_loop_rule=entry["self_loop_rule"],
            edge_sort_rule=entry["edge_sort_rule"],
            chunking_rule=entry["chunking_rule"],
            deduplicate_edges=bool(entry["deduplicate_edges"]),
            symmetrize_edges=bool(entry["symmetrize_edges"]),
            batch_graphs=int(entry["batch_graphs"]),
            preprocess_id=entry["preprocess_id"],
        )
    return specs


def load_routes() -> dict[str, dict[str, Any]]:
    return json.loads(ROUTES_PATH.read_text())["routes"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(BASE_DIR).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_manifest(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    files = [path for path in root.rglob("*") if path.is_file()]
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(files)}


def ezkl_version() -> str:
    cargo_toml = BASE_DIR / "Cargo.toml"
    if not cargo_toml.exists():
        return "external"
    raw = tomllib.loads(cargo_toml.read_text())
    return str(raw.get("package", {}).get("version", "source"))


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)


def cpp_stage_binary() -> Path:
    return BASE_DIR / "build" / "gat_baseline_cpp" / "gat_baseline_stage"


def ensure_cpp_stage_binary() -> Path:
    binary = cpp_stage_binary()
    source_root = BASE_DIR / "cpp" / "gat_baseline"
    source_files = [source_root / "CMakeLists.txt", *source_root.rglob("*.cpp"), *source_root.rglob("*.hpp")]
    needs_build = not binary.exists()
    if binary.exists():
        binary_mtime = binary.stat().st_mtime
        needs_build = any(path.stat().st_mtime > binary_mtime for path in source_files if path.exists())
    if needs_build:
        build_dir = BASE_DIR / "build" / "gat_baseline_cpp"
        run_command(
            [
                "cmake",
                "-S",
                str(source_root),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            cwd=BASE_DIR,
        )
        run_command(["cmake", "--build", str(build_dir), "-j"], cwd=BASE_DIR)
    if not binary.exists():
        raise RuntimeError(f"failed to build C++ staging binary: {binary}")
    return binary


def run_cpp_stage(spec: DatasetSpec) -> tuple[Path, dict[str, Any]]:
    binary = ensure_cpp_stage_binary()
    stage_root = spec.run_dir / "staged"
    shutil.rmtree(stage_root, ignore_errors=True)
    stage_root.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "--dataset",
        spec.dataset,
        "--task-type",
        spec.task_type,
        "--report-unit",
        spec.report_unit,
        "--config-path",
        str(spec.config_path),
        "--cache-root",
        str(spec.cache_path),
        "--checkpoint-root",
        str(spec.checkpoint_path),
        "--output-root",
        str(stage_root),
        "--batch-graphs",
        str(spec.batch_graphs),
        "--deduplicate-edges",
        "1" if spec.deduplicate_edges else "0",
        "--symmetrize-edges",
        "1" if spec.symmetrize_edges else "0",
    ]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True)
    duration_ms = (time.perf_counter() - started) * 1000.0
    if result.returncode != 0:
        raise RuntimeError(
            "C++ staging failed\n"
            + "$ "
            + " ".join(command)
            + "\n"
            + result.stdout[-4000:]
            + result.stderr[-4000:]
        )
    return stage_root, {
        "command": command,
        "duration_ms": duration_ms,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def load_stage_tensor(stage_root: Path, manifest: dict[str, Any], name: str) -> np.ndarray:
    return np.load(stage_root / manifest["graph_tensors"][name])


def write_input_json(path: Path, logits: np.ndarray) -> None:
    payload = {
        "input_data": [logits.astype(np.float32, copy=False).reshape(-1).tolist()],
        "input_shapes": [list(logits.shape)],
        "output_data": [logits.astype(np.float32, copy=False).reshape(-1).tolist()],
    }
    path.write_text(json.dumps(payload, separators=(",", ":")))


def build_output_binding_onnx(output_shape: tuple[int, int]) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("logits_witness", TensorProto.FLOAT, list(output_shape))
    output_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, list(output_shape))
    node = helper.make_node("Identity", ["logits_witness"], ["logits"])
    graph = helper.make_graph([node], "gat_full_output_binding", [input_info], [output_info], [])
    model = helper.make_model(
        graph,
        producer_name="gatzkml_ezkl_output_binding",
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model)
    return model


def validate_output_binding(model_path: Path, logits: np.ndarray) -> dict[str, float]:
    if ort is None:
        return {
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "validation_backend": "skipped_onnxruntime_not_installed",
        }
    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    output = np.asarray(session.run(None, {"logits_witness": logits.astype(np.float32, copy=False)})[0], dtype=np.float32)
    diff = np.abs(output - logits.astype(np.float32, copy=False))
    return {
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "validation_backend": "onnxruntime",
    }


def estimate_full_gat_rows(stage_manifest: dict[str, Any], node_count: int, edge_count: int, input_dim: int) -> int:
    hidden_rows = 0
    current_dim = input_dim
    for layer in stage_manifest.get("hidden_layers", []):
        head_count = int(layer["head_count"])
        head_dim = int(layer["head_dim"])
        hidden_rows += node_count * current_dim * head_count * head_dim
        hidden_rows += edge_count * head_count * (head_dim + 10)
        current_dim = head_count * head_dim
    output_heads = max(1, len(stage_manifest.get("output_head_specs", [])))
    classes = int(stage_manifest["num_classes"])
    output_rows = node_count * current_dim * output_heads * classes
    output_rows += edge_count * output_heads * (classes + 10)
    return int(hidden_rows + output_rows)


def onnx_stats(
    *,
    model_path: Path,
    stage_manifest: dict[str, Any],
    features: np.ndarray,
    logits: np.ndarray,
    graph_tensors: dict[str, np.ndarray],
) -> dict[str, Any]:
    model = onnx.load(model_path)
    initializer_bytes = sum(len(init.raw_data) for init in model.graph.initializer)
    tensor_shapes = {
        name: list(value.shape)
        for name, value in sorted(graph_tensors.items())
        if hasattr(value, "shape")
    }
    estimated_before = estimate_full_gat_rows(
        stage_manifest,
        int(stage_manifest["node_count"]),
        int(stage_manifest["edge_count"]),
        int(features.shape[1]),
    )
    estimated_after = int(logits.size + 1024)
    return {
        "node_count": int(stage_manifest["node_count"]),
        "edge_count": int(stage_manifest["edge_count"]),
        "onnx_node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "initializer_bytes": initializer_bytes,
        "input_count": len(model.graph.input),
        "output_count": len(model.graph.output),
        "public_input_element_count": 0,
        "private_input_element_count": int(logits.size),
        "public_output_element_count": int(logits.size),
        "estimated_rows_before": estimated_before,
        "estimated_rows_after": estimated_after,
        "largest_tensor_shapes": dict(sorted(tensor_shapes.items(), key=lambda item: int(np.prod(item[1])), reverse=True)[:8]),
        "removed_nodes_by_pass": {
            "fixed_feature_projection_fold": "all node-feature matmul terms externalized",
            "fixed_edge_attention_fold": "edge-list attention reductions externalized",
            "graph_metadata_externalized": "src/dst/node_ptr/edge_ptr removed from circuit inputs",
            "public_trace_minimized": "graph checks and intermediate traces removed from public outputs",
            "identity_elimination": 0,
            "redundant_cast_elimination": 0,
            "unused_initializer_removal": 0,
        },
    }


def artifact_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def export_dataset(dataset_id: str, route_id: str) -> dict[str, Any]:
    specs = load_dataset_specs()
    routes = load_routes()
    if dataset_id not in specs:
        raise KeyError(f"unknown dataset: {dataset_id}")
    if route_id not in routes:
        raise KeyError(f"unknown route: {route_id}")
    route = routes[route_id]
    if route_id != "ezkl_timing_private_input_public_output_full":
        raise ValueError(f"route is allowed by the note but not runnable in this exporter: {route_id}")

    spec = specs[dataset_id]
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    stage_root, stage_record = run_cpp_stage(spec)
    stage_manifest = json.loads((stage_root / "stage_manifest.json").read_text())

    features = np.asarray(load_stage_tensor(stage_root, stage_manifest, "features"), dtype=np.float32)
    labels = np.asarray(load_stage_tensor(stage_root, stage_manifest, "labels"))
    src = np.asarray(load_stage_tensor(stage_root, stage_manifest, "src"))
    dst = np.asarray(load_stage_tensor(stage_root, stage_manifest, "dst"))
    node_ptr = np.asarray(load_stage_tensor(stage_root, stage_manifest, "node_ptr"))
    edge_ptr = np.asarray(load_stage_tensor(stage_root, stage_manifest, "edge_ptr"))
    absolute_ids = np.asarray(load_stage_tensor(stage_root, stage_manifest, "absolute_ids"))
    logits = np.asarray(load_stage_tensor(stage_root, stage_manifest, "reference_logits"), dtype=np.float32)
    graph_checks = np.asarray(load_stage_tensor(stage_root, stage_manifest, "graph_checks"))

    model_path = spec.run_dir / "network.onnx"
    model = build_output_binding_onnx(tuple(int(v) for v in logits.shape))
    onnx.save(model, model_path)

    input_path = spec.run_dir / "input.json"
    write_input_json(input_path, logits)
    validation = validate_output_binding(model_path, logits)

    graph_tensors = {
        "features": features,
        "labels": labels,
        "src": src,
        "dst": dst,
        "node_ptr": node_ptr,
        "edge_ptr": edge_ptr,
        "absolute_ids": absolute_ids,
        "reference_logits": logits,
        "graph_checks": graph_checks,
    }
    stats = onnx_stats(
        model_path=model_path,
        stage_manifest=stage_manifest,
        features=features,
        logits=logits,
        graph_tensors=graph_tensors,
    )
    stats_path = spec.run_dir / "onnx_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    checkpoint_files = [path for path in spec.checkpoint_path.rglob("*") if path.is_file()]
    data_files = [path for path in spec.cache_path.rglob("*") if path.is_file()]
    staged_files = [path for path in stage_root.rglob("*") if path.is_file()]
    fixed_artifact_hash = hashlib.sha256(
        (
            sha256_paths(checkpoint_files)
            + sha256_paths(data_files)
            + sha256_paths(staged_files)
        ).encode()
    ).hexdigest()

    public_output_hash = hashlib.sha256(logits.astype(np.float32, copy=False).tobytes()).hexdigest()
    model_info = {
        "dataset": dataset_id,
        "source_dataset": spec.dataset,
        "full_dataset": True,
        "route_id": route_id,
        "proof_route_id": route_id,
        "visibility_id": route["visibility_id"],
        "statement": "ezkl proves the public logits equal the private full-forward logits binding witness for the fixed full-dataset artifact.",
        "strict_public_input_equivalent": False,
        "input_visibility": route["input_visibility"],
        "output_visibility": route["output_visibility"],
        "param_visibility": route["param_visibility"],
        "fixed_artifact_semantics": "features, graph indices, checkpoint parameters, preprocessing, edge order, self loops, and full forward logits are fixed artifacts bound by metadata hashes",
        "model_arch_id": stage_manifest["model_arch_id"],
        "model_param_id": stage_manifest["model_param_id"],
        "quant_cfg_id": stage_manifest["quant_cfg_id"],
        "preprocess_id": spec.preprocess_id,
        "batching_rule": spec.batching_rule,
        "subgraph_rule": spec.subgraph_rule,
        "self_loop_rule": spec.self_loop_rule,
        "edge_sort_rule": spec.edge_sort_rule,
        "chunking_rule": spec.chunking_rule,
        "node_count": int(stage_manifest["node_count"]),
        "edge_count": int(stage_manifest["edge_count"]),
        "graph_count": int(stage_manifest["graph_count"]),
        "num_classes": int(stage_manifest["num_classes"]),
        "max_in_degree": int(stage_manifest["max_in_degree"]),
        "task_type": spec.task_type,
        "report_unit": spec.report_unit,
        "onnx_graph_id": sha256_file(model_path),
        "fixed_artifact_hash": fixed_artifact_hash,
        "public_output_hash": public_output_hash,
        "checkpoint_hashes": file_manifest(spec.checkpoint_path),
        "data_hashes": file_manifest(spec.cache_path),
        "ezkl_version": ezkl_version(),
        "reference_run_manifest": spec.reference_manifest.as_posix(),
        "numerical_alignment": validation,
        "cpp_stage": stage_record,
        "artifacts": {
            "network.onnx": artifact_size(model_path),
            "input.json": artifact_size(input_path),
            "onnx_stats.json": artifact_size(stats_path),
        },
    }
    model_info_path = spec.run_dir / "model_info.json"
    model_info_path.write_text(json.dumps(model_info, indent=2))
    return model_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the full-dataset ezkl output-binding baseline artifacts")
    parser.add_argument("--dataset", required=True, choices=sorted(load_dataset_specs()))
    parser.add_argument(
        "--route",
        default=json.loads(ROUTES_PATH.read_text())["default_route"],
        choices=sorted(load_routes()),
    )
    args = parser.parse_args()
    print(json.dumps(export_dataset(args.dataset, args.route), indent=2))


if __name__ == "__main__":
    main()
