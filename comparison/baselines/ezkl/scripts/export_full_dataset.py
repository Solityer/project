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
from onnx import TensorProto, helper, numpy_helper

try:
    import onnxruntime as ort
except ModuleNotFoundError:
    ort = None


DATASETS_PATH = BASE_DIR / "configs" / "full_datasets.json"
ROUTES_PATH = BASE_DIR / "configs" / "ezkl_routes.json"

from output_binding import (
    DIGEST_ROUTE_ID,
    IDENTITY_ROUTE_ID,
    build_binding,
    choose_binding_plan,
)


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


def write_input_json(path: Path, private_input: np.ndarray, public_output: np.ndarray) -> None:
    payload = {
        "input_data": [private_input.astype(np.float32, copy=False).reshape(-1).tolist()],
        "input_shapes": [list(private_input.shape)],
        "output_data": [public_output.astype(np.float32, copy=False).reshape(-1).tolist()],
    }
    path.write_text(json.dumps(payload, separators=(",", ":")))


def build_identity_output_binding_onnx(output_shape: tuple[int, int]) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("logits_witness", TensorProto.FLOAT, list(output_shape))
    output_info = helper.make_tensor_value_info("output_binding", TensorProto.FLOAT, [1, int(np.prod(output_shape))])
    shape_name = "flat_shape"
    shape_tensor = numpy_helper.from_array(np.asarray([1, int(np.prod(output_shape))], dtype=np.int64), name=shape_name)
    node = helper.make_node("Reshape", ["logits_witness", shape_name], ["output_binding"])
    graph = helper.make_graph([node], "gat_full_identity_output_binding", [input_info], [output_info], [shape_tensor])
    model = helper.make_model(
        graph,
        producer_name="gatzkml_ezkl_output_binding",
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model)
    return model


def build_field_folding_onnx(output_shape: tuple[int, int], public_tail: np.ndarray, challenge_values: np.ndarray) -> onnx.ModelProto:
    element_count = int(np.prod(output_shape))
    block_size = int(challenge_values.shape[0])
    block_count = element_count // block_size
    public_count = int(challenge_values.shape[1] + public_tail.size)
    input_info = helper.make_tensor_value_info("logits_witness", TensorProto.FLOAT, list(output_shape))
    output_info = helper.make_tensor_value_info("output_binding", TensorProto.FLOAT, [1, public_count])
    block_shape = numpy_helper.from_array(np.asarray([block_count, block_size], dtype=np.int64), name="block_shape")
    challenge_tensor = numpy_helper.from_array(challenge_values.astype(np.float32, copy=False), name="fold_challenges")
    tail_values = np.asarray(public_tail, dtype=np.float32, order="C").reshape(1, -1)
    tail_tensor = numpy_helper.from_array(tail_values, name="public_hash_tail")
    nodes = [
        helper.make_node("Reshape", ["logits_witness", "block_shape"], ["blocked_logits"]),
        helper.make_node("MatMul", ["blocked_logits", "fold_challenges"], ["fold_values_by_block"]),
        helper.make_node("ReduceSum", ["fold_values_by_block"], ["fold_values"], axes=[0], keepdims=1),
        helper.make_node("Concat", ["fold_values", "public_hash_tail"], ["output_binding"], axis=1),
    ]
    initializers = [block_shape, challenge_tensor, tail_tensor]
    graph = helper.make_graph(nodes, "gat_full_field_folding_output_binding", [input_info], [output_info], initializers)
    model = helper.make_model(
        graph,
        producer_name="gatzkml_ezkl_output_binding",
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model)
    return model


def validate_output_binding(model_path: Path, logits: np.ndarray, expected_output: np.ndarray) -> dict[str, float | str]:
    if ort is None:
        return {
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "validation_backend": "skipped_onnxruntime_not_installed",
        }
    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    output = np.asarray(session.run(None, {"logits_witness": logits.astype(np.float32, copy=False)})[0], dtype=np.float32)
    diff = np.abs(output.reshape(-1) - expected_output.astype(np.float32, copy=False).reshape(-1))
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
    route_id: str,
    binding_backend: str,
    public_output_element_count: int,
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
    if binding_backend == "field_folding":
        estimated_after = int(logits.size * 1.2 + public_output_element_count + 4096)
    else:
        estimated_after = int(logits.size + 1024)
    return {
        "dataset": stage_manifest["dataset"],
        "route_id": route_id,
        "binding_backend": binding_backend,
        "node_count": int(stage_manifest["node_count"]),
        "edge_count": int(stage_manifest["edge_count"]),
        "onnx_node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "initializer_bytes": initializer_bytes,
        "input_count": len(model.graph.input),
        "output_count": len(model.graph.output),
        "public_input_element_count": 0,
        "private_input_element_count": int(logits.size),
        "public_output_element_count": int(public_output_element_count),
        "estimated_rows_before": estimated_before,
        "estimated_rows_after": estimated_after,
        "public_instances_before": int(logits.size),
        "public_instances_after": int(public_output_element_count),
        "reads_full_logits_input": True,
        "binding_computed_inside_onnx": True,
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


def export_dataset(dataset_id: str, route_id: str = "auto", binding_backend: str = "auto") -> dict[str, Any]:
    specs = load_dataset_specs()
    routes = load_routes()
    if dataset_id not in specs:
        raise KeyError(f"unknown dataset: {dataset_id}")
    plan = choose_binding_plan(dataset_id, binding_backend, route_id)
    route_id = plan.route_id
    binding_backend = plan.binding_backend
    if route_id not in routes:
        raise KeyError(f"unknown route: {route_id}")
    route = routes[route_id]

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

    binding_metadata = {
        "dataset": dataset_id,
        "source_dataset": spec.dataset,
        "route_id": route_id,
        "binding_backend": binding_backend,
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
        "fixed_artifact_hash": fixed_artifact_hash,
    }

    output_logits_path = spec.run_dir / "output_logits.bin"
    output_binding, public_output, challenge_values = build_binding(
        dataset_id=dataset_id,
        route_id=route_id,
        binding_backend=binding_backend,
        logits=logits,
        metadata=binding_metadata,
        output_logits_path=output_logits_path,
    )

    if binding_backend == "field_folding":
        if challenge_values is None:
            raise RuntimeError("field_folding selected without challenge values")
        public_tail = public_output.reshape(1, -1)[:, int(challenge_values.shape[1]) :]
        model = build_field_folding_onnx(tuple(int(v) for v in logits.shape), public_tail, challenge_values)
    else:
        model = build_identity_output_binding_onnx(tuple(int(v) for v in logits.shape))

    model_path = spec.run_dir / "network.onnx"
    onnx.save(model, model_path)

    input_path = spec.run_dir / "input.json"
    write_input_json(input_path, logits, public_output)
    validation = validate_output_binding(model_path, logits, public_output)

    output_binding_path = spec.run_dir / "output_binding.json"
    output_binding_path.write_text(json.dumps(output_binding, indent=2))

    public_output_hash = hashlib.sha256(public_output.astype(np.float32, copy=False).tobytes()).hexdigest()
    onnx_graph_id = sha256_file(model_path)

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
        route_id=route_id,
        binding_backend=binding_backend,
        public_output_element_count=int(public_output.size),
    )
    stats_path = spec.run_dir / "onnx_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    model_info = {
        "dataset": dataset_id,
        "source_dataset": spec.dataset,
        "full_dataset": True,
        "route_id": route_id,
        "proof_route_id": route_id,
        "visibility_id": route["visibility_id"],
        "statement": route["statement"],
        "strict_public_input_equivalent": False,
        "input_visibility": route["input_visibility"],
        "output_visibility": route["output_visibility"],
        "param_visibility": route["param_visibility"],
        "binding_backend": binding_backend,
        "binding_security": output_binding["binding_security"],
        "binding_note": output_binding["binding_note"],
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
        "onnx_graph_id": onnx_graph_id,
        "fixed_artifact_hash": fixed_artifact_hash,
        "public_output_hash": public_output_hash,
        "output_artifact_hash": output_binding["output_artifact_hash"],
        "metadata_hash": output_binding["metadata_hash"],
        "shape_hash": output_binding["shape_hash"],
        "public_output_element_count_before": output_binding["public_output_element_count_before"],
        "public_output_element_count_after": output_binding["public_output_element_count_after"],
        "public_reduction_ratio": output_binding["public_reduction_ratio"],
        "reads_full_logits_input": True,
        "binding_computed_inside_onnx": True,
        "output_binding_path": output_binding_path.as_posix(),
        "output_logits_path": output_logits_path.as_posix(),
        "binding_metadata": binding_metadata,
        "checkpoint_hashes": file_manifest(spec.checkpoint_path),
        "data_hashes": file_manifest(spec.cache_path),
        "ezkl_version": ezkl_version(),
        "reference_run_manifest": spec.reference_manifest.as_posix(),
        "numerical_alignment": validation,
        "cpp_stage": stage_record,
        "artifacts": {
            "network.onnx": artifact_size(model_path),
            "input.json": artifact_size(input_path),
            "output_logits.bin": artifact_size(output_logits_path),
            "output_binding.json": artifact_size(output_binding_path),
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
        default="auto",
        choices=["auto", *sorted(load_routes())],
    )
    parser.add_argument("--binding-backend", default="auto", choices=["auto", "cryptographic_hash", "field_folding", "identity_public_logits"])
    args = parser.parse_args()
    print(json.dumps(export_dataset(args.dataset, args.route, args.binding_backend), indent=2))


if __name__ == "__main__":
    main()
