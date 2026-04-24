from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from .manifest import ResolvedDatasetSpec


@dataclass(frozen=True)
class DegreeBucket:
    degree: int
    node_index: np.ndarray
    incoming_src: np.ndarray


@dataclass(frozen=True)
class GraphInputBundle:
    features: np.ndarray
    labels: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    node_ptr: np.ndarray
    edge_ptr: np.ndarray
    absolute_ids: np.ndarray
    reference_logits: np.ndarray
    graph_checks: np.ndarray
    degree_buckets: tuple[DegreeBucket, ...]
    max_in_degree: int

    @property
    def num_nodes(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.src.shape[0])

    @property
    def graph_count(self) -> int:
        return int(self.node_ptr.shape[0] - 1)


@dataclass(frozen=True)
class HeadWeights:
    seq_kernel: np.ndarray
    attn_src_kernel: np.ndarray
    attn_src_bias: np.ndarray
    attn_dst_kernel: np.ndarray
    attn_dst_bias: np.ndarray
    output_bias: np.ndarray


@dataclass(frozen=True)
class LayerWeights:
    layer_index: int
    input_dim: int
    head_count: int
    head_dim: int
    heads: tuple[HeadWeights, ...]


@dataclass(frozen=True)
class GATCheckpoint:
    model_arch_id: str
    model_param_id: str
    quant_cfg_id: str
    output_average_rule: str
    hidden_layers: tuple[LayerWeights, ...]
    output_heads: tuple[HeadWeights, ...]
    num_classes: int


def _load_reference_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _ezkl_version(repo_root: Path) -> str:
    cargo_toml = tomllib.loads((repo_root / "Cargo.toml").read_text())
    raw_version = str(cargo_toml.get("package", {}).get("version", "0.0.0"))
    if raw_version == "0.0.0":
        return "source - no compatibility guaranteed"
    return raw_version


def _weight_hash(spec: ResolvedDatasetSpec) -> str:
    digest = hashlib.sha256()
    for path in (
        spec.checkpoint_path / "manifest.json",
        spec.checkpoint_path / "tensors.npz",
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _artifact_sizes(output_root: Path) -> dict[str, int]:
    return {
        path.name: path.stat().st_size
        for path in sorted(output_root.iterdir())
        if path.is_file()
    }


def _write_export_log(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines))


def _cpp_stage_binary(repo_root: Path) -> Path:
    return repo_root / "build" / "gat_baseline_cpp" / "gat_baseline_stage"


def _run_checked(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)


def _ensure_cpp_stage_binary(repo_root: Path) -> Path:
    binary = _cpp_stage_binary(repo_root)
    source_roots = [repo_root / "cpp" / "gat_baseline" / "CMakeLists.txt", *sorted((repo_root / "cpp" / "gat_baseline" / "src").rglob("*"))]
    needs_build = not binary.exists()
    if binary.exists():
        binary_mtime = binary.stat().st_mtime
        needs_build = any(path.is_file() and path.stat().st_mtime > binary_mtime for path in source_roots)
    if not needs_build:
        return binary

    _run_checked(
        [
            "cmake",
            "-S",
            str(repo_root / "cpp" / "gat_baseline"),
            "-B",
            str(repo_root / "build" / "gat_baseline_cpp"),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=repo_root,
    )
    _run_checked(["cmake", "--build", str(repo_root / "build" / "gat_baseline_cpp"), "-j"], cwd=repo_root)
    if not binary.exists():
        raise RuntimeError(f"failed to build C++ stage binary: {binary}")
    return binary


def _run_cpp_stage(spec: ResolvedDatasetSpec) -> tuple[Path, str]:
    binary = _ensure_cpp_stage_binary(spec.repo_root)
    stage_root = spec.output_root / "staged"
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
    result = subprocess.run(command, cwd=spec.repo_root, capture_output=True, text=True)
    log_text = f"$ {' '.join(command)}\n"
    if result.stdout:
        log_text += f"--- stdout ---\n{result.stdout}"
        if not log_text.endswith("\n"):
            log_text += "\n"
    if result.stderr:
        log_text += f"--- stderr ---\n{result.stderr}"
        if not log_text.endswith("\n"):
            log_text += "\n"
    if result.returncode != 0:
        raise RuntimeError(f"cpp stage failed with status {result.returncode}\n{log_text}")
    return stage_root, log_text


def _degree_buckets(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> tuple[tuple[DegreeBucket, ...], int]:
    degree_groups: dict[int, list[tuple[int, np.ndarray]]] = {}
    edge_offset = 0
    max_in_degree = 0

    for node_index in range(num_nodes):
        edge_begin = edge_offset
        while edge_offset < dst.shape[0] and int(dst[edge_offset]) == node_index:
            edge_offset += 1
        if edge_begin == edge_offset:
            raise ValueError(f"node {node_index} has no incoming edges after self-loop expansion")

        node_sources = src[edge_begin:edge_offset].astype(np.int64, copy=False)
        degree = int(node_sources.shape[0])
        max_in_degree = max(max_in_degree, degree)
        degree_groups.setdefault(degree, []).append((node_index, node_sources.copy()))

    if edge_offset != dst.shape[0]:
        raise ValueError("edge list is not fully covered by dst-stable ordering")

    buckets: list[DegreeBucket] = []
    for degree in sorted(degree_groups):
        entries = degree_groups[degree]
        node_index = np.asarray([item[0] for item in entries], dtype=np.int64)
        incoming_src = np.asarray([item[1] for item in entries], dtype=np.int64)
        buckets.append(DegreeBucket(degree=degree, node_index=node_index, incoming_src=incoming_src))
    return tuple(buckets), max_in_degree


def _load_stage_manifest(stage_root: Path) -> dict[str, object]:
    return json.loads((stage_root / "stage_manifest.json").read_text())


def _load_stage_tensor(stage_root: Path, relpath: str) -> np.ndarray:
    return np.load(stage_root / relpath)


def _load_stage_graph(stage_root: Path, manifest: dict[str, object]) -> GraphInputBundle:
    graph_tensors = manifest["graph_tensors"]
    assert isinstance(graph_tensors, dict)
    features = np.asarray(_load_stage_tensor(stage_root, str(graph_tensors["features"])), dtype=np.float32)
    labels = np.asarray(_load_stage_tensor(stage_root, str(graph_tensors["labels"])), dtype=np.int64)
    src = np.asarray(_load_stage_tensor(stage_root, str(graph_tensors["src"])), dtype=np.int64)
    dst = np.asarray(_load_stage_tensor(stage_root, str(graph_tensors["dst"])), dtype=np.int64)
    node_ptr = np.asarray(_load_stage_tensor(stage_root, str(graph_tensors["node_ptr"])), dtype=np.int64)
    edge_ptr = np.asarray(_load_stage_tensor(stage_root, str(graph_tensors["edge_ptr"])), dtype=np.int64)
    absolute_ids = np.asarray(_load_stage_tensor(stage_root, str(graph_tensors["absolute_ids"])), dtype=np.int64)
    reference_logits = np.asarray(_load_stage_tensor(stage_root, str(graph_tensors["reference_logits"])), dtype=np.float32)
    graph_checks = np.asarray(_load_stage_tensor(stage_root, str(graph_tensors["graph_checks"])), dtype=np.float32)
    degree_buckets, max_in_degree = _degree_buckets(src, dst, int(features.shape[0]))
    expected_max = int(manifest["max_in_degree"])
    if max_in_degree != expected_max:
        raise ValueError(f"staged max_in_degree mismatch: {max_in_degree} != {expected_max}")
    return GraphInputBundle(
        features=features,
        labels=labels,
        src=src,
        dst=dst,
        node_ptr=node_ptr,
        edge_ptr=edge_ptr,
        absolute_ids=absolute_ids,
        reference_logits=reference_logits,
        graph_checks=graph_checks,
        degree_buckets=degree_buckets,
        max_in_degree=max_in_degree,
    )


def _reshape_seq_kernel(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"expected seq kernel [1,in,out], got {tensor.shape}")
    return tensor.reshape(tensor.shape[1], tensor.shape[2]).astype(np.float32, copy=False)


def _reshape_attention_kernel(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != 1:
        raise ValueError(f"expected attention kernel [1,width,1], got {tensor.shape}")
    return tensor.reshape(tensor.shape[1], 1).astype(np.float32, copy=False)


def _reshape_scalar_bias(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim != 1 or tensor.shape[0] != 1:
        raise ValueError(f"expected scalar bias [1], got {tensor.shape}")
    return tensor.astype(np.float32, copy=False)


def _reshape_output_bias(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim != 1:
        raise ValueError(f"expected output bias [width], got {tensor.shape}")
    return tensor.astype(np.float32, copy=False)


def _load_head(stage_root: Path, manifest: dict[str, object], spec: dict[str, object]) -> HeadWeights:
    tensor_files = manifest["tensor_files"]
    assert isinstance(tensor_files, dict)
    return HeadWeights(
        seq_kernel=_reshape_seq_kernel(_load_stage_tensor(stage_root, str(tensor_files[str(spec["seq_kernel"])]))),
        attn_src_kernel=_reshape_attention_kernel(_load_stage_tensor(stage_root, str(tensor_files[str(spec["attn_src_kernel"])]))),
        attn_src_bias=_reshape_scalar_bias(_load_stage_tensor(stage_root, str(tensor_files[str(spec["attn_src_bias"])]))),
        attn_dst_kernel=_reshape_attention_kernel(_load_stage_tensor(stage_root, str(tensor_files[str(spec["attn_dst_kernel"])]))),
        attn_dst_bias=_reshape_scalar_bias(_load_stage_tensor(stage_root, str(tensor_files[str(spec["attn_dst_bias"])]))),
        output_bias=_reshape_output_bias(_load_stage_tensor(stage_root, str(tensor_files[str(spec["output_bias"])]))),
    )


def _load_stage_checkpoint(stage_root: Path, manifest: dict[str, object]) -> GATCheckpoint:
    hidden_specs_by_layer: dict[int, list[dict[str, object]]] = {}
    for head_spec in manifest["hidden_head_specs"]:
        assert isinstance(head_spec, dict)
        hidden_specs_by_layer.setdefault(int(head_spec["layer_index"]), []).append(head_spec)

    hidden_layers: list[LayerWeights] = []
    for layer_info in manifest["hidden_layers"]:
        assert isinstance(layer_info, dict)
        layer_index = int(layer_info["layer_index"])
        head_specs = sorted(hidden_specs_by_layer[layer_index], key=lambda item: int(item["local_head_index"]))
        heads = tuple(_load_head(stage_root, manifest, head_spec) for head_spec in head_specs)
        hidden_layers.append(
            LayerWeights(
                layer_index=layer_index,
                input_dim=int(layer_info["input_dim"]),
                head_count=int(layer_info["head_count"]),
                head_dim=int(layer_info["head_dim"]),
                heads=heads,
            )
        )

    output_heads = tuple(
        _load_head(stage_root, manifest, head_spec)
        for head_spec in sorted(manifest["output_head_specs"], key=lambda item: int(item["head_index"]))
    )

    return GATCheckpoint(
        model_arch_id=str(manifest["model_arch_id"]),
        model_param_id=str(manifest["model_param_id"]),
        quant_cfg_id=str(manifest["quant_cfg_id"]),
        output_average_rule=str(manifest["output_average_rule"]),
        hidden_layers=tuple(hidden_layers),
        output_heads=output_heads,
        num_classes=int(manifest["num_classes"]),
    )


def write_input_json(path: Path, tensors: list[np.ndarray], outputs: list[np.ndarray]) -> None:
    payload = {
        "input_data": [tensor.reshape(-1).tolist() for tensor in tensors],
        "input_shapes": [list(tensor.shape) for tensor in tensors],
        "output_data": [tensor.reshape(-1).tolist() for tensor in outputs],
    }
    path.write_text(json.dumps(payload))


class OnnxGraphBuilder:
    def __init__(self) -> None:
        self._nodes: list[onnx.NodeProto] = []
        self._initializers: list[onnx.TensorProto] = []
        self._counter = 0

    def add_initializer(self, name: str, array: np.ndarray) -> str:
        self._initializers.append(numpy_helper.from_array(np.asarray(array), name=name))
        return name

    def make_name(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def add_node(self, op_type: str, inputs: list[str], *, outputs: list[str] | None = None, **attrs: object) -> str:
        if outputs is None:
            outputs = [self.make_name(op_type.lower())]
        self._nodes.append(helper.make_node(op_type, inputs, outputs, **attrs))
        return outputs[0]

    @property
    def nodes(self) -> list[onnx.NodeProto]:
        return self._nodes

    @property
    def initializers(self) -> list[onnx.TensorProto]:
        return self._initializers


def _input_info(name: str, proto_type: int, shape: tuple[int, ...] | np.ndarray.shape) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, proto_type, list(shape))


def _output_info(name: str, proto_type: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, proto_type, shape)


def _graph_check(builder: OnnxGraphBuilder, input_name: str, expected: np.ndarray, prefix: str) -> str:
    expected_name = builder.add_initializer(f"{prefix}_expected", expected.astype(np.float32, copy=False))
    cast_name = builder.add_node("Cast", [input_name], to=TensorProto.FLOAT)
    diff_name = builder.add_node("Sub", [cast_name, expected_name])
    abs_name = builder.add_node("Abs", [diff_name])
    return builder.add_node("ReduceSum", [abs_name], keepdims=1, axes=[0])


@dataclass(frozen=True)
class BucketBindings:
    degree: int
    node_index: np.ndarray
    node_index_name: str
    incoming_src_name: str


def _bucket_bindings(builder: OnnxGraphBuilder, graph: GraphInputBundle) -> tuple[BucketBindings, ...]:
    bindings: list[BucketBindings] = []
    for bucket_index, bucket in enumerate(graph.degree_buckets):
        bindings.append(
            BucketBindings(
                degree=bucket.degree,
                node_index=bucket.node_index,
                node_index_name=builder.add_initializer(
                    f"degree_bucket_{bucket.degree}_{bucket_index}_nodes",
                    bucket.node_index.astype(np.int64, copy=False),
                ),
                incoming_src_name=builder.add_initializer(
                    f"degree_bucket_{bucket.degree}_{bucket_index}_incoming_src",
                    bucket.incoming_src.astype(np.int64, copy=False),
                ),
            )
        )
    return tuple(bindings)


def _scatter_index_name(
    builder: OnnxGraphBuilder,
    cache: dict[tuple[int, int], str],
    bucket: BucketBindings,
    width: int,
) -> str:
    key = (bucket.degree, width)
    if key not in cache:
        scatter_index = np.repeat(bucket.node_index[:, None], width, axis=1).astype(np.int64, copy=False)
        cache[key] = builder.add_initializer(f"degree_bucket_{bucket.degree}_scatter_idx_w{width}", scatter_index)
    return cache[key]


def _zero_output_name(builder: OnnxGraphBuilder, cache: dict[int, str], num_nodes: int, width: int) -> str:
    if width not in cache:
        cache[width] = builder.add_initializer(f"zero_output_w{width}", np.zeros((num_nodes, width), dtype=np.float32))
    return cache[width]


def _run_head_graph(
    builder: OnnxGraphBuilder,
    graph: GraphInputBundle,
    bucket_bindings: tuple[BucketBindings, ...],
    zero_output_cache: dict[int, str],
    scatter_index_cache: dict[tuple[int, int], str],
    input_name: str,
    head: HeadWeights,
    prefix: str,
    apply_output_bias: bool,
    apply_activation: bool,
) -> str:
    seq_kernel = builder.add_initializer(f"{prefix}_seq_kernel", head.seq_kernel)
    attn_src_kernel = builder.add_initializer(f"{prefix}_attn_src_kernel", head.attn_src_kernel)
    attn_src_bias = builder.add_initializer(f"{prefix}_attn_src_bias", head.attn_src_bias)
    attn_dst_kernel = builder.add_initializer(f"{prefix}_attn_dst_kernel", head.attn_dst_kernel)
    attn_dst_bias = builder.add_initializer(f"{prefix}_attn_dst_bias", head.attn_dst_bias)
    output_bias = None
    if apply_output_bias:
        output_bias = builder.add_initializer(f"{prefix}_output_bias", head.output_bias.reshape(1, -1))

    h_prime = builder.add_node("MatMul", [input_name, seq_kernel])
    e_src = builder.add_node("Add", [builder.add_node("MatMul", [h_prime, attn_src_kernel]), attn_src_bias])
    e_dst = builder.add_node("Add", [builder.add_node("MatMul", [h_prime, attn_dst_kernel]), attn_dst_bias])

    output_width = head.seq_kernel.shape[1]
    output_name = _zero_output_name(builder, zero_output_cache, graph.num_nodes, output_width)

    for bucket in bucket_bindings:
        neighbor_h = builder.add_node("Gather", [h_prime, bucket.incoming_src_name], axis=0)
        neighbor_src = builder.add_node("Gather", [e_src, bucket.incoming_src_name], axis=0)
        node_dst = builder.add_node("Gather", [e_dst, bucket.node_index_name], axis=0)
        node_dst_expanded = builder.add_node("Unsqueeze", [node_dst], axes=[1])
        scores = builder.add_node("Add", [neighbor_src, node_dst_expanded])
        z = builder.add_node("LeakyRelu", [scores], alpha=0.2)
        max_per_node = builder.add_node("ReduceMax", [z], keepdims=1, axes=[1])
        exp_scores = builder.add_node("Exp", [builder.add_node("Sub", [z, max_per_node])])
        denominator = builder.add_node("ReduceSum", [exp_scores], keepdims=1, axes=[1])
        alpha = builder.add_node("Div", [exp_scores, denominator])
        weighted = builder.add_node("Mul", [alpha, neighbor_h])
        aggregated = builder.add_node("ReduceSum", [weighted], keepdims=0, axes=[1])

        if apply_output_bias:
            aggregated = builder.add_node("Add", [aggregated, output_bias])
        if apply_activation:
            aggregated = builder.add_node("Elu", [aggregated], alpha=1.0)

        output_name = builder.add_node(
            "ScatterElements",
            [output_name, _scatter_index_name(builder, scatter_index_cache, bucket, output_width), aggregated],
            axis=0,
        )

    return output_name


def build_onnx_model(spec: ResolvedDatasetSpec, graph: GraphInputBundle, checkpoint: GATCheckpoint) -> onnx.ModelProto:
    builder = OnnxGraphBuilder()
    bucket_bindings = _bucket_bindings(builder, graph)
    zero_output_cache: dict[int, str] = {}
    scatter_index_cache: dict[tuple[int, int], str] = {}

    inputs = [
        _input_info("features", TensorProto.FLOAT, graph.features.shape),
        _input_info("src", TensorProto.INT64, graph.src.shape),
        _input_info("dst", TensorProto.INT64, graph.dst.shape),
        _input_info("node_ptr", TensorProto.INT64, graph.node_ptr.shape),
        _input_info("edge_ptr", TensorProto.INT64, graph.edge_ptr.shape),
    ]

    hidden_name = "features"
    for layer in checkpoint.hidden_layers:
        head_outputs = []
        for head_index, head in enumerate(layer.heads):
            head_outputs.append(
                _run_head_graph(
                    builder,
                    graph,
                    bucket_bindings,
                    zero_output_cache,
                    scatter_index_cache,
                    hidden_name,
                    head,
                    prefix=f"hidden_l{layer.layer_index}_h{head_index}",
                    apply_output_bias=False,
                    apply_activation=True,
                )
            )
        hidden_name = builder.add_node("Concat", head_outputs, axis=1)

    output_heads = []
    for head_index, head in enumerate(checkpoint.output_heads):
        output_heads.append(
            _run_head_graph(
                builder,
                graph,
                bucket_bindings,
                zero_output_cache,
                scatter_index_cache,
                hidden_name,
                head,
                prefix=f"output_h{head_index}",
                apply_output_bias=True,
                apply_activation=False,
            )
        )

    if len(output_heads) == 1:
        logits_name = output_heads[0]
    else:
        logits_name = output_heads[0]
        for other_output in output_heads[1:]:
            logits_name = builder.add_node("Add", [logits_name, other_output])
        average_scale = builder.add_initializer(
            "output_average_scale",
            np.asarray([1.0 / float(len(output_heads))], dtype=np.float32),
        )
        logits_name = builder.add_node("Mul", [logits_name, average_scale])

    graph_checks = builder.add_node(
        "Concat",
        [
            _graph_check(builder, "src", graph.src, "src"),
            _graph_check(builder, "dst", graph.dst, "dst"),
            _graph_check(builder, "node_ptr", graph.node_ptr, "node_ptr"),
            _graph_check(builder, "edge_ptr", graph.edge_ptr, "edge_ptr"),
        ],
        axis=0,
    )

    graph_proto = helper.make_graph(
        builder.nodes,
        f"gat_formal_{spec.dataset}",
        inputs,
        [
            _output_info(logits_name, TensorProto.FLOAT, [graph.num_nodes, checkpoint.num_classes]),
            _output_info(graph_checks, TensorProto.FLOAT, [4]),
        ],
        builder.initializers,
    )
    model = helper.make_model(
        graph_proto,
        producer_name="gat_baseline",
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model)
    return model


def _validate_model(model_path: Path, graph: GraphInputBundle) -> dict[str, float]:
    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    outputs = session.run(
        None,
        {
            "features": graph.features,
            "src": graph.src,
            "dst": graph.dst,
            "node_ptr": graph.node_ptr,
            "edge_ptr": graph.edge_ptr,
        },
    )
    ort_logits = np.asarray(outputs[0], dtype=np.float32)
    ort_checks = np.asarray(outputs[1], dtype=np.float32).reshape(4)
    return {
        "max_abs_diff": float(np.max(np.abs(ort_logits - graph.reference_logits))),
        "mean_abs_diff": float(np.mean(np.abs(ort_logits - graph.reference_logits))),
        "graph_check_max": float(np.max(np.abs(ort_checks - graph.graph_checks.reshape(4)))),
    }


def export_dataset(spec: ResolvedDatasetSpec) -> dict[str, object]:
    started = time.perf_counter()
    spec.output_root.mkdir(parents=True, exist_ok=True)

    stage_started = time.perf_counter()
    stage_root, stage_log = _run_cpp_stage(spec)
    stage_elapsed_ms = (time.perf_counter() - stage_started) * 1000.0

    stage_manifest = _load_stage_manifest(stage_root)
    graph = _load_stage_graph(stage_root, stage_manifest)
    checkpoint = _load_stage_checkpoint(stage_root, stage_manifest)

    build_started = time.perf_counter()
    model = build_onnx_model(spec, graph, checkpoint)
    build_elapsed_ms = (time.perf_counter() - build_started) * 1000.0
    model_path = spec.output_root / "network.onnx"

    save_started = time.perf_counter()
    onnx.save(model, model_path)
    save_elapsed_ms = (time.perf_counter() - save_started) * 1000.0

    onnx_graph_id = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model_weight_hash = _weight_hash(spec)
    reference_manifest = _load_reference_manifest(spec.reference_run_manifest_path)
    ezkl_version = _ezkl_version(spec.repo_root)

    input_json_path = spec.output_root / "input.json"
    write_input_json(
        input_json_path,
        [graph.features, graph.src, graph.dst, graph.node_ptr, graph.edge_ptr],
        [graph.reference_logits, graph.graph_checks.reshape(4)],
    )

    validation_started = time.perf_counter()
    validation = _validate_model(model_path, graph)
    validation_elapsed_ms = (time.perf_counter() - validation_started) * 1000.0
    validation_path = spec.output_root / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2))

    model_info = {
        "dataset": spec.dataset,
        "task_type": spec.task_type,
        "report_unit": spec.report_unit,
        "protocol_id": spec.protocol_id,
        "model_arch_id": checkpoint.model_arch_id,
        "model_param_id": checkpoint.model_param_id,
        "onnx_graph_id": onnx_graph_id,
        "quant_cfg_id": checkpoint.quant_cfg_id,
        "model_weight_hash": model_weight_hash,
        "ezkl_version": ezkl_version,
        "preprocess_id": spec.preprocess_id,
        "attestation_mode": spec.attestation_mode,
        "encoding_id": spec.encoding_id,
        "degree_bound_id": spec.degree_bound_id,
        "proof_system_tag": spec.proof_system_tag,
        "node_count": graph.num_nodes,
        "edge_count": graph.num_edges,
        "graph_count": graph.graph_count,
        "max_in_degree": graph.max_in_degree,
        "batching_rule": spec.batching_rule,
        "self_loop_rule": spec.self_loop_rule,
        "edge_sort_rule": spec.edge_sort_rule,
        "chunking_rule": spec.chunking_rule,
        "reference_run_manifest": spec.reference_run_manifest_path.as_posix(),
        "reference_metrics": {
            "prove_time_ms": reference_manifest.get("prove_time_ms"),
            "verify_time_ms": reference_manifest.get("verify_time_ms"),
            "proof_size_bytes": reference_manifest.get("proof_size_bytes"),
            "node_count": reference_manifest.get("node_count"),
            "edge_count": reference_manifest.get("edge_count"),
        },
        "graph_binding": {
            "mode": "public_tensor_equality_checks",
            "outputs": ["graph_checks"],
        },
        "public_io": stage_manifest.get("public_io"),
        "numerical_alignment": validation,
        "cpp_stage": {
            "stage_root": stage_root.as_posix(),
            "stage_manifest": (stage_root / "stage_manifest.json").as_posix(),
            "config": stage_manifest.get("config"),
        },
        "artifacts": {
            "network": model_path.name,
            "input": input_json_path.name,
            "validation": validation_path.name,
        },
    }
    model_info_path = spec.output_root / "model_info.json"
    model_info_path.write_text(json.dumps(model_info, indent=2))

    (spec.output_root / "cpp_stage.log").write_text(stage_log)
    _write_export_log(
        spec.output_root / "export.log",
        [
            f"dataset={spec.dataset}",
            f"node_count={graph.num_nodes}",
            f"edge_count={graph.num_edges}",
            f"graph_count={graph.graph_count}",
            f"max_in_degree={graph.max_in_degree}",
            f"onnx_graph_id={onnx_graph_id}",
            f"model_weight_hash={model_weight_hash}",
            f"max_abs_diff={validation['max_abs_diff']}",
            f"mean_abs_diff={validation['mean_abs_diff']}",
            f"graph_check_max={validation['graph_check_max']}",
            f"cpp_stage_ms={stage_elapsed_ms:.3f}",
            f"build_onnx_ms={build_elapsed_ms:.3f}",
            f"save_onnx_ms={save_elapsed_ms:.3f}",
            f"validate_onnxruntime_ms={validation_elapsed_ms:.3f}",
        ],
    )
    return model_info
