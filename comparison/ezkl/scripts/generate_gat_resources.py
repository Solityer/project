#!/usr/bin/env python3
"""Generate authenticated ezkl-local GAT lookup tables and bound ledgers.

The native ezkl adapter consumes complete, contiguous integer lookup tables and
an explicit no-wrap ledger. This generator is deliberately baseline-local: it
reads only the packaged canonical cache/checkpoint/config files and emits the
ezkl schema, rather than importing a proof resource from another baseline. The
output is subsequently authenticated by validate_resources.py.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path
from typing import Any

DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
BN254_MODULUS = ("21888242871839275222246405745257275088548364400416034343698204186575808495617")
U64_MAX = (1 << 64) - 1
U128_MAX = (1 << 128) - 1
I128_MIN = -(1 << 127)
I128_MAX = (1 << 127) - 1
TABLE_MAGIC = b"EZKL-GAT-TABLES\x00"
TABLE_FORMAT_VERSION = 1
TABLE_ORDER = ("lrelu", "exp", "elu", "max_relu")
NPY_MAGIC = b"\x93NUMPY"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def append_string(out: bytearray, value: str) -> None:
    encoded = value.encode()
    out.extend(struct.pack("<Q", len(encoded)))
    out.extend(encoded)


def append_i128(out: bytearray, value: int) -> None:
    out.extend(int(value).to_bytes(16, "little", signed=True))


def append_usize(out: bytearray, value: int) -> None:
    out.extend(struct.pack("<Q", int(value) & U64_MAX))


def hash_string(hasher: Any, value: str) -> None:
    encoded = value.encode()
    hasher.update(struct.pack("<Q", len(encoded)))
    hasher.update(encoded)


def hash_i128(hasher: Any, value: int) -> None:
    hasher.update(int(value).to_bytes(16, "little", signed=True))


def hash_u64(hasher: Any, value: int) -> None:
    hasher.update(struct.pack("<Q", int(value) & U64_MAX))


def round_away(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def round_ratio_away(numerator: int, denominator: int) -> int:
    sign = -1 if numerator < 0 else 1
    value = abs(numerator)
    return sign * ((2 * value + denominator) // (2 * denominator))


def parse_cfg(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"malformed config line {path}:{number}")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or key in result:
            raise ValueError(f"duplicate/empty config key {path}:{number}")
        result[key] = value
    return result


def scale_list(config: dict[str, str], key: str) -> list[int]:
    value = config.get(key)
    if value is None:
        raise ValueError(f"missing scale-chain key {key}")
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"invalid scale-chain key {key}: {value}") from exc
    if not values or any(item <= 0 or item > U128_MAX for item in values):
        raise ValueError(f"scale-chain key {key} must contain positive u128 integers")
    return values


def scale_value(config: dict[str, str], key: str) -> int:
    values = scale_list(config, key)
    if len(values) != 1:
        raise ValueError(f"scale-chain key {key} must contain one value")
    return values[0]


def scale_multipliers(config: dict[str, str], key: str, count: int) -> list[int]:
    values = scale_list(config, key)
    if len(values) != count:
        raise ValueError(f"scale-chain key {key} width {len(values)} does not match authenticated width {count}")
    if any(value & (value - 1) for value in values):
        raise ValueError(f"scale-chain key {key} must contain positive power-of-two multipliers")
    return values


def write_cfg(path: Path, updates: dict[str, str]) -> None:
    lines = path.read_text().splitlines()
    seen: set[str] = set()
    output: list[str] = []
    for raw in lines:
        stripped = raw.split("#", 1)[0].strip()
        if "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                output.append(f"{key} = {updates[key]}")
                seen.add(key)
                continue
        output.append(raw)
    for key in updates:
        if key not in seen:
            output.append(f"{key} = {updates[key]}")
    atomic_write(path, ("\n".join(output) + "\n").encode())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=False) + "\n").encode())


def atomic_write(path: Path, payload: bytes) -> None:
    """Install a small generated artifact without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def load_tensors(path: Path) -> dict[str, tuple[list[int], list[float]]]:
    tokens = path.read_text().split()
    cursor = 0
    result: dict[str, tuple[list[int], list[float]]] = {}
    while cursor < len(tokens):
        if tokens[cursor] != "TENSOR" or cursor + 3 >= len(tokens):
            raise ValueError(f"malformed tensor archive {path}")
        name = tokens[cursor + 1]
        if not name or name in result:
            raise ValueError(f"duplicate/empty tensor name in checkpoint archive: {name}")
        rank = int(tokens[cursor + 2])
        shape_start = cursor + 3
        shape = [int(x) for x in tokens[shape_start:shape_start + rank]]
        count = int(tokens[shape_start + rank])
        value_start = shape_start + rank + 1
        values = [float(x) for x in tokens[value_start:value_start + count]]
        if len(values) != count or math.prod(shape) != count:
            raise ValueError(f"tensor count mismatch: {name}")
        result[name] = (shape, values)
        cursor = value_start + count
    return result


def quantized_tensor_abs_max(
    tensors: dict[str, tuple[list[int], list[float]]], name: str, multiplier: int, expected_count: int,
    authenticated_shape: list[int], output_bias_classes: int | None = None,
) -> int:
    if multiplier <= 0 or multiplier > U128_MAX or multiplier & (multiplier - 1):
        raise ValueError(f"quantization multiplier for {name} must be a positive power-of-two u128")
    tensor = tensors.get(name)
    if tensor is None:
        raise ValueError(f"checkpoint is missing authenticated model tensor {name}")
    shape, values = tensor
    valid_shape = (
        isinstance(authenticated_shape, list)
        and bool(authenticated_shape)
        and all(isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0 for dimension in authenticated_shape)
    )
    output_bias_shapes = (
        ([output_bias_classes], [1, output_bias_classes, 1])
        if output_bias_classes is not None
        else ()
    )
    shape_matches = shape == authenticated_shape
    if output_bias_shapes:
        shape_matches = shape_matches or (shape in output_bias_shapes and authenticated_shape in output_bias_shapes)
    if (
        not valid_shape or not shape_matches or not shape
        or any(dimension <= 0 for dimension in shape)
        or not values or len(values) != expected_count or math.prod(shape) != expected_count
    ):
        raise ValueError(f"checkpoint tensor {name} shape/count disagrees with its authenticated role")
    maximum = 0
    for value in values:
        quantized = quantized_value(value, multiplier, name)
        if quantized == I128_MIN:
            raise ValueError(f"quantized checkpoint absolute bound exceeds signed i128 in {name}")
        maximum = max(maximum, abs(quantized))
    return maximum


def quantized_value(value: float, multiplier: int, label: str) -> int:
    if (
        isinstance(multiplier, bool) or not isinstance(multiplier, int)
        or multiplier <= 0 or multiplier > U128_MAX or multiplier & (multiplier - 1)
    ):
        raise ValueError(f"quantization multiplier for {label} must be a positive power-of-two u128")
    if not math.isfinite(value):
        raise ValueError(f"non-finite checkpoint value in {label}")
    scaled = value * multiplier
    if not math.isfinite(scaled):
        raise ValueError(f"non-finite scaled checkpoint value in {label}")
    if scaled > float(I128_MAX) or scaled < float(I128_MIN):
        raise ValueError(f"scaled checkpoint value exceeds signed i128 range in {label}")
    quantized = round_away(scaled)
    # Preserve the native cast result at the representable +/-2^127 endpoints.
    return min(I128_MAX, max(I128_MIN, quantized))


def quantized_model_maxima(
    tensors: dict[str, tuple[list[int], list[float]]], checkpoint: dict[str, Any], scales: dict[str, str],
) -> dict[str, int]:
    """Bound the exact checkpoint tensors using their authenticated per-head scales."""
    hidden_layers = checkpoint.get("hidden_layers")
    hidden_specs = checkpoint.get("hidden_head_specs")
    output_specs = checkpoint.get("output_head_specs")
    if (
        not isinstance(hidden_layers, list) or not hidden_layers
        or not isinstance(hidden_specs, list) or not isinstance(output_specs, list)
        or checkpoint.get("L") != len(hidden_layers) + 1
        or checkpoint.get("K_out") != len(output_specs)
    ):
        raise ValueError("checkpoint manifest has an incomplete authenticated layer/head schema")

    def manifest_int(value: Any, label: str, minimum: int = 0) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"checkpoint manifest {label} is invalid")
        return value

    maxima: dict[str, int] = {}
    used_tensors: set[str] = set()
    tensor_index = checkpoint.get("tensor_index")
    if not isinstance(tensor_index, dict):
        raise ValueError("checkpoint manifest has no authenticated tensor_index")

    def include(
        name: Any, multiplier: int, expected_count: int, output_bias_classes: int | None = None,
    ) -> None:
        if not isinstance(name, str) or not name or name in used_tensors:
            raise ValueError("checkpoint manifest has a missing or duplicate model tensor reference")
        used_tensors.add(name)
        metadata = tensor_index.get(name)
        if not isinstance(metadata, dict):
            raise ValueError(f"checkpoint tensor_index is missing authenticated shape for {name}")
        authenticated_shape = metadata.get("shape")
        family = name.rsplit("/", 1)[0]
        maximum = quantized_tensor_abs_max(
            tensors, name, multiplier, expected_count, authenticated_shape, output_bias_classes,
        )
        maxima[family] = max(maxima.get(family, 0), maximum)

    grouped_specs: dict[int, list[dict[str, Any]]] = {}
    for spec in hidden_specs:
        if not isinstance(spec, dict):
            raise ValueError("checkpoint hidden head spec is malformed")
        layer_index = manifest_int(spec.get("layer_index"), "hidden layer index")
        head_index = manifest_int(spec.get("local_head_index"), "hidden local head index")
        if layer_index >= len(hidden_layers):
            raise ValueError("checkpoint hidden head spec references an unknown layer")
        grouped_specs.setdefault(layer_index, []).append(spec)

    current_dim: int | None = None
    for layer_index, layer in enumerate(hidden_layers):
        if not isinstance(layer, dict) or manifest_int(layer.get("layer_index"), "hidden layer index") != layer_index:
            raise ValueError("checkpoint hidden layer indices are not contiguous")
        input_dim = manifest_int(layer.get("input_dim"), "hidden input dimension", 1)
        head_count = manifest_int(layer.get("head_count"), "hidden head count", 1)
        head_dim = manifest_int(layer.get("head_dim"), "hidden head dimension", 1)
        if current_dim is not None and input_dim != current_dim:
            raise ValueError("checkpoint hidden layer input dimension is not closed")
        specs = sorted(grouped_specs.get(layer_index, []), key=lambda value: int(value["local_head_index"]))
        if len(specs) != head_count or [spec["local_head_index"] for spec in specs] != list(range(head_count)):
            raise ValueError(f"checkpoint hidden layer {layer_index} head specifications are not closed")
        weight_scales = scale_multipliers(scales, f"s_w_hidden_{layer_index}", head_count)
        src_scales = scale_multipliers(scales, f"s_a_src_hidden_{layer_index}", head_count)
        dst_scales = scale_multipliers(scales, f"s_a_dst_hidden_{layer_index}", head_count)
        for head_index, spec in enumerate(specs):
            include(spec.get("seq_kernel"), weight_scales[head_index], input_dim * head_dim)
            include(spec.get("attn_src_kernel"), src_scales[head_index], head_dim)
            include(spec.get("attn_dst_kernel"), dst_scales[head_index], head_dim)
        current_dim = head_count * head_dim

    output_count = manifest_int(checkpoint.get("K_out"), "output head count", 1)
    classes = manifest_int(checkpoint.get("C"), "output class count", 1)
    if current_dim is None or output_count != len(output_specs):
        raise ValueError("checkpoint output head count is not closed")
    if any(not isinstance(spec, dict) for spec in output_specs):
        raise ValueError("checkpoint output head spec is malformed")
    for spec in output_specs:
        manifest_int(spec.get("head_index"), "output head index")
    outputs = sorted(output_specs, key=lambda value: value["head_index"])
    if [spec["head_index"] for spec in outputs] != list(range(output_count)):
        raise ValueError("checkpoint output head specifications are not closed")
    output_weight_scales = scale_multipliers(scales, "s_w_output", output_count)
    output_src_scales = scale_multipliers(scales, "s_a_src_output", output_count)
    output_dst_scales = scale_multipliers(scales, "s_a_dst_output", output_count)
    output_bias_scales = scale_multipliers(scales, "s_b_output", output_count)
    for head_index, spec in enumerate(outputs):
        include(spec.get("seq_kernel"), output_weight_scales[head_index], current_dim * classes)
        include(spec.get("attn_src_kernel"), output_src_scales[head_index], classes)
        include(spec.get("attn_dst_kernel"), output_dst_scales[head_index], classes)
        include(spec.get("output_bias"), output_bias_scales[head_index], classes, classes)

    checkpoint_tensors = {name for name in tensors if name.startswith(("hidden/", "output/"))}
    if checkpoint_tensors != used_tensors:
        missing = sorted(used_tensors - checkpoint_tensors)
        extra = sorted(checkpoint_tensors - used_tensors)
        raise ValueError(f"checkpoint model tensor manifest mismatch: missing={missing}, unreferenced={extra}")
    return maxima


def quantized_model(
    root: Path, dataset: str, checkpoint: dict[str, Any], scales: dict[str, str],
) -> dict[str, int]:
    checkpoint_dir = root / "resources/canonical/checkpoints" / f"{dataset}_gat"
    tensors = load_tensors(checkpoint_dir / "tensors.txt")
    return quantized_model_maxima(tensors, checkpoint, scales)


def graph_info(root: Path, dataset: str, meta: dict[str, str]) -> tuple[int, int, int]:
    nodes = int(meta["num_nodes"])
    graph_count = int(meta.get("graph_count", "1"))
    if nodes <= 0 or graph_count <= 0 or graph_count > nodes:
        raise ValueError(f"invalid graph dimensions in {dataset}: nodes={nodes}, graph_count={graph_count}")
    if dataset == "ppi":
        node_ptr = [int(x) for x in (root / "resources/canonical/cache/ppi/node_ptr.txt").read_text().split()]
    else:
        if graph_count != 1:
            raise ValueError(f"non-PPI dataset {dataset} must contain exactly one graph")
        node_ptr = [0, nodes]
    if len(node_ptr) != graph_count + 1 or node_ptr[0] != 0 or node_ptr[-1] != nodes or any(start >= end for start, end in zip(node_ptr, node_ptr[1:])):
        raise ValueError(f"node_ptr does not match graph metadata in {dataset}")
    graph_for_node = [0] * nodes
    for graph in range(graph_count):
        for node in range(node_ptr[graph], node_ptr[graph + 1]):
            graph_for_node[node] = graph
    edges = root / "resources/canonical/cache" / dataset / "edges.txt"
    print(f"[resource-preparation] {dataset}: reading graph edges from {edges} ({edges.stat().st_size:,} bytes)", flush=True)
    raw: list[tuple[int, int, int]] = []
    with edges.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number % (1 << 20) == 0:
                print(f"[resource-preparation] {dataset}: graph edges processed {line_number:,} lines", flush=True)
            if not line.strip():
                continue
            parts = line.split()
            try:
                values = [int(x) for x in parts]
            except ValueError as exc:
                raise ValueError(f"invalid edge integer at line {line_number} in {dataset}") from exc
            if dataset == "ppi":
                if len(values) != 3:
                    raise ValueError(f"invalid PPI edge row at line {line_number} in {dataset}")
                declared_graph, source, destination = values
            else:
                if len(values) != 2:
                    raise ValueError(f"invalid edge row at line {line_number} in {dataset}")
                declared_graph = None
                source, destination = values
            if source < 0 or source >= nodes or destination < 0 or destination >= nodes:
                raise ValueError(f"edge endpoint outside node range at line {line_number} in {dataset}")
            graph = graph_for_node[source]
            if declared_graph is not None and declared_graph != graph:
                raise ValueError(f"edge graph id disagrees at line {line_number} in {dataset}")
            if graph_for_node[destination] != graph:
                raise ValueError(f"edge crosses graph boundary at line {line_number} in {dataset}")
            raw.append((graph, source, destination))
    raw.sort(key=lambda item: (item[0], item[2]))
    active_destination: tuple[int, int] | None = None
    seen_sources: set[int] = set()
    write = 0
    for edge in raw:
        graph, source, destination = edge
        key = (graph, destination)
        if active_destination != key:
            active_destination = key
            seen_sources.clear()
        if source in seen_sources:
            continue
        seen_sources.add(source)
        raw[write] = edge
        write += 1
    del raw[write:]
    has_self_loop = [False] * nodes
    for _graph, source, destination in raw:
        if source == destination:
            has_self_loop[source] = True
    for node in range(nodes):
        if not has_self_loop[node]:
            raw.append((graph_for_node[node], node, node))
    raw.sort(key=lambda item: (item[0], item[2]))
    degrees = [0] * nodes
    for _graph, _source, destination in raw:
        degrees[destination] += 1
    return nodes, len(raw), max(degrees)


def npy_contract(path: Path) -> dict[str, Any]:
    """Read only the bounded NPY header and stream its authenticated payload."""
    with path.open("rb") as stream:
        prefix = stream.read(10)
        if len(prefix) < 10 or prefix[:6] != NPY_MAGIC:
            raise ValueError(f"invalid npy file {path}")
        major = prefix[6]
        if major == 1:
            header_len = int.from_bytes(prefix[8:10], "little")
            start = 10
        elif major == 2:
            prefix += stream.read(2)
            if len(prefix) < 12:
                raise ValueError(f"truncated npy header {path}")
            header_len = int.from_bytes(prefix[8:12], "little")
            start = 12
        else:
            raise ValueError(f"unsupported npy version {major}")
        if header_len > 16 * 1024 * 1024:
            raise ValueError(f"npy header is unreasonably large: {path}")
        header_bytes = stream.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"truncated npy header {path}")
    try:
        header = ast.literal_eval(header_bytes.decode("latin1").strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid npy header {path}: {exc}") from exc
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise ValueError(f"unsupported npy header schema {path}")
    descr = header["descr"]
    if descr == "<f4":
        item_size, fmt, dtype = 4, "<f", "float32"
    elif descr == "<f8":
        item_size, fmt, dtype = 8, "<d", "float64"
    else:
        raise ValueError(f"unsupported npy dtype {descr!r} in {path}")
    if header["fortran_order"] is not False:
        raise ValueError(f"Fortran-order npy arrays are unsupported: {path}")
    shape = header["shape"]
    if not isinstance(shape, tuple) or not shape or any(not isinstance(x, int) or x <= 0 for x in shape):
        raise ValueError(f"invalid npy shape in {path}")
    length = math.prod(shape)
    payload_size = length * item_size
    actual_size = path.stat().st_size
    data_offset = start + header_len
    if actual_size != data_offset + payload_size:
        raise ValueError(f"npy data length mismatch in {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return {
        "dtype": dtype,
        "descr": descr,
        "shape": list(shape),
        "order": "C",
        "length": length,
        "item_size": item_size,
        "format": fmt,
        "data_offset": data_offset,
        "sha256": hasher.hexdigest(),
    }


def iter_npy_values(path: Path, contract: dict[str, Any]):
    """Yield NPY values in bounded chunks, never materializing the tensor."""
    item_size = int(contract["item_size"])
    fmt = str(contract["format"])
    carry = b""
    with path.open("rb") as stream:
        stream.seek(int(contract["data_offset"]))
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            data = carry + block
            usable = len(data) - (len(data) % item_size)
            for (value,) in struct.iter_unpack(fmt, data[:usable]):
                yield float(value)
            carry = data[usable:]
    if carry:
        raise ValueError(f"truncated npy value payload: {path}")


def feature_bound(root: Path, dataset: str, meta: dict[str, str]) -> int:
    path = root / "resources/canonical/cache" / dataset
    scale = 16
    if dataset in {"ppi", "ogbn_arxiv"}:
        feature_path = path / "features.npy"
        contract = npy_contract(feature_path)
        expected_shape = [int(meta["num_nodes"]), int(meta["num_features"])]
        if contract["shape"] != expected_shape:
            raise ValueError(f"{dataset} feature NPY shape does not match metadata")
        preprocess_id = meta.get("input_preprocess_id", "")
        expected_preprocess = "row_normalize" if dataset == "ppi" else "ogbn_arxiv_raw_float32"
        if preprocess_id != expected_preprocess:
            raise ValueError(f"{dataset} feature input preprocessing identity is invalid: {preprocess_id!r}")
        maximum = 1
        rows, columns = contract["shape"]
        total = int(contract["length"])
        print(f"[resource-preparation] {dataset}: scanning NPY feature bound ({total:,} values)", flush=True)
        values = iter_npy_values(feature_path, contract)
        processed = 0
        for _row_index in range(rows):
            row = [next(values) for _column_index in range(columns)]
            row = preprocess_npy_feature_row(row, dataset, preprocess_id)
            for value in row:
                maximum = max(maximum, _quantized_feature_abs(value, scale))
                processed += 1
                if processed % (1 << 20) == 0:
                    print(f"[resource-preparation] {dataset}: NPY feature bound {processed:,}/{total:,}", flush=True)
        return maximum
    maximum = 0
    preprocess = meta.get("input_preprocess_id", "")
    if preprocess not in {"row_normalize", "none"}:
        raise ValueError(f"unsupported sparse feature preprocessing identity in {dataset}: {preprocess!r}")
    expected_nodes, expected_features = int(meta["num_nodes"]), int(meta["num_features"])
    with (path / "features.txt").open(encoding="utf-8") as stream:
        header = stream.readline().split()
        if len(header) != 2 or tuple(int(value) for value in header) != (expected_nodes, expected_features):
            raise ValueError(f"{dataset} sparse feature shape does not match metadata")
        rows = 0
        for line in stream:
            if rows >= expected_nodes:
                raise ValueError(f"{dataset} sparse feature row count does not match metadata")
            dense = [0.0] * expected_features
            seen: set[int] = set()
            for token in line.split():
                try:
                    index_text, value_text = token.split(":", 1)
                    index = int(index_text)
                    value = float(value_text)
                except ValueError as exc:
                    raise ValueError(f"malformed sparse feature at row {rows} in {dataset}: {token!r}") from exc
                if index < 0 or index >= expected_features:
                    raise ValueError(f"sparse feature index outside shape at row {rows} in {dataset}")
                if index in seen:
                    raise ValueError(f"duplicate sparse feature column at row {rows} in {dataset}: {index}")
                if not math.isfinite(value):
                    raise ValueError(f"non-finite sparse feature at row {rows}, column {index} in {dataset}")
                seen.add(index)
                dense[index] = value
            norm = 1.0
            if preprocess == "row_normalize":
                norm = 0.0
                for value in dense:
                    norm += value
            if not math.isfinite(norm):
                raise ValueError(f"invalid feature normalization in {dataset}")
            if norm == 0:
                norm = 1.0
            normalized = [value / norm for value in dense]
            if any(not math.isfinite(value) for value in normalized):
                raise ValueError(f"feature normalization produced NaN/Inf in {dataset} row {rows}")
            maximum = max(maximum, max((_quantized_feature_abs(value, scale) for value in normalized), default=0))
            rows += 1
        if rows != expected_nodes:
            raise ValueError(f"{dataset} sparse feature row count does not match metadata")
    return max(1, maximum)


def _quantized_feature_abs(value: float, scale: int) -> int:
    quantized = round_away(value * scale)
    if quantized < -(1 << 63) or quantized >= (1 << 63):
        raise ValueError("canonical quantized feature exceeds signed int64 range")
    return abs(quantized)


def preprocess_npy_feature_row(values: list[float], dataset: str, preprocess_id: str) -> list[float]:
    if dataset == "ppi" and preprocess_id == "row_normalize":
        row = [float(value) for value in values]
        if any(not math.isfinite(value) for value in row):
            raise ValueError("row normalization rejects NaN/Inf")
        rowsum = 0.0
        for value in row:
            rowsum += float(value)
        if not math.isfinite(rowsum):
            raise ValueError("row normalization sum is not finite")
        normalized = row if rowsum == 0.0 else [float(value) / rowsum for value in row]
        if any(not math.isfinite(value) for value in normalized):
            raise ValueError("row normalization produced NaN/Inf")
        return normalized
    if dataset == "ogbn_arxiv" and preprocess_id == "ogbn_arxiv_raw_float32":
        try:
            row = [struct.unpack("<f", struct.pack("<f", float(value)))[0] for value in values]
        except OverflowError as exc:
            raise ValueError("raw float32 input preprocessing produced NaN/Inf") from exc
        if any(not math.isfinite(value) for value in row):
            raise ValueError("raw float32 input preprocessing rejects NaN/Inf")
        return row
    raise ValueError(f"unsupported NPY feature preprocessing identity for {dataset}: {preprocess_id!r}")


def table_value(table: dict[str, Any], input_value: int) -> int:
    function_id = table["function_id"]
    if function_id == "lrelu":
        return input_value if input_value >= 0 else round_ratio_away(input_value, 5)
    if function_id == "exp_neg":
        return max(1, round_away(math.exp(-input_value / 256.0) * 16.0))
    if function_id == "elu":
        return input_value if input_value >= 0 else round_away((math.exp(input_value / 16.0) - 1.0) * 16.0)
    if function_id == "max_relu":
        return max(0, input_value)
    raise ValueError(f"unsupported exact table function: {function_id}")


def iter_table_rows(table: dict[str, Any]):
    for input_value in range(int(table["input_min"]), int(table["input_max"]) + 1):
        yield input_value, table_value(table, input_value)


def table_canonical_length(table: dict[str, Any]) -> int:
    """Length of ExactTable::canonical_bytes without constructing it."""
    string_fields = ("table_id", "function_id", "domain_id", "generator_id")
    return (
        len(b"ezkl-gat-exact-table-v1") + sum(8 + len(str(table[field]).encode()) for field in string_fields) + 4 + 4 + 16 + 16 + int(table["row_count"]) * 32
    )


def table_digest(table: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"ezkl-gat-exact-table-v1")
    hash_string(hasher, table["table_id"])
    hasher.update(struct.pack("<i", table["input_scale"]))
    hasher.update(struct.pack("<i", table["output_scale"]))
    hash_string(hasher, table["function_id"])
    hash_string(hasher, table["domain_id"])
    hash_string(hasher, table["generator_id"])
    hash_i128(hasher, table["input_min"])
    hash_i128(hasher, table["input_max"])
    for input_value, output_value in iter_table_rows(table):
        hash_i128(hasher, input_value)
        hash_i128(hasher, output_value)
    return hasher.hexdigest()


def static_tables_digest(tables: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"ezkl-gat-static-tables-v1")
    hash_string(hasher, tables["static_tables_id"])
    hash_string(hasher, tables["backend_aux_tables_id"])
    for key in TABLE_ORDER:
        table = tables[key]
        hash_u64(hasher, table_canonical_length(table))
        hasher.update(b"ezkl-gat-exact-table-v1")
        hash_string(hasher, table["table_id"])
        hasher.update(struct.pack("<i", table["input_scale"]))
        hasher.update(struct.pack("<i", table["output_scale"]))
        hash_string(hasher, table["function_id"])
        hash_string(hasher, table["domain_id"])
        hash_string(hasher, table["generator_id"])
        hash_i128(hasher, table["input_min"])
        hash_i128(hasher, table["input_max"])
        for input_value, output_value in iter_table_rows(table):
            hash_i128(hasher, input_value)
            hash_i128(hasher, output_value)
        hash_string(hasher, table["digest"])
    return hasher.hexdigest()


def ledger_digest(ledger: dict[str, Any]) -> str:
    out = bytearray(b"ezkl-gat-ledger-v1")
    append_string(out, ledger["ledger_id"])
    append_string(out, ledger["field_modulus"])
    for key in (
        "max_score_abs",
        "max_table_output_abs",
        "max_delta",
        "max_exp",
        "max_sum",
        "max_aggregation_num",
        "max_dividend_abs",
        "max_aux_abs",
        "max_divisor",
    ):
        append_i128(out, ledger[key])
    for family in ledger["families"]:
        append_usize(out, family["layer"])
        append_usize(out, family["head"])
        out.append(1 if family["is_output"] else 0)
        for key in (
            "max_score_abs",
            "max_weight_abs",
            "max_attention_abs",
            "max_bias_abs",
            "max_table_output_abs",
            "max_delta",
            "max_exp",
            "max_sum",
            "max_aggregation_num",
            "max_dividend_abs",
            "max_aux_abs",
            "max_divisor",
        ):
            append_i128(out, family[key])
    return sha256(bytes(out))


def make_table(
    table_id: str,
    function_id: str,
    domain_id: str,
    generator_id: str,
    input_scale: int,
    output_scale: int,
    low: int,
    high: int,
) -> dict[str, Any]:
    table = {
        "table_id": table_id,
        "input_scale": input_scale,
        "output_scale": output_scale,
        "function_id": function_id,
        "domain_id": domain_id,
        "generator_id": generator_id,
        "input_min": low,
        "input_max": high,
        "row_count": high - low + 1,
        "digest": "",
    }
    table["digest"] = table_digest(table)
    return table


def derive_bound_plan(
    meta: dict[str, str],
    checkpoint: dict[str, Any],
    scales: dict[str, str],
    maxima: dict[str, int],
    feature_max: int,
    degree: int,
) -> dict[str, Any]:
    hidden_layers = checkpoint.get("hidden_layers")
    if not isinstance(hidden_layers, list) or not hidden_layers:
        raise ValueError("checkpoint hidden_layers must be a non-empty list")
    if int(checkpoint.get("L", 0)) != len(hidden_layers) + 1:
        raise ValueError("checkpoint layer count is not closed")
    output_heads = int(checkpoint.get("K_out", 0))
    if output_heads <= 0:
        raise ValueError("checkpoint output head count must be positive")
    s_agg_pre = scale_list(scales, "s_agg_pre")
    if len(s_agg_pre) != len(hidden_layers):
        raise ValueError("hidden aggregation scale count does not match checkpoint")
    s_h_prime = [scale_list(scales, f"s_h_prime_hidden_{layer_idx}") for layer_idx in range(len(hidden_layers))]
    if any(len(layer_scales) != int(layer["head_count"]) for layer_scales, layer in zip(s_h_prime, hidden_layers)):
        raise ValueError("hidden projection scale count does not match checkpoint")
    s_y_prime = scale_list(scales, "s_y_prime")
    if len(s_y_prime) != output_heads:
        raise ValueError("output projection scale count does not match checkpoint")
    s_out_pre = scale_value(scales, "s_out_pre")
    if degree <= 0 or feature_max < 0 or any(value <= 0 for value in s_agg_pre + s_y_prime + [s_out_pre]):
        raise ValueError("graph, feature, and output scale bounds must be non-negative/positive")

    exp_zero_output = table_value({"function_id": "exp_neg"}, 0)
    current_bound = feature_max
    current_dim = int(meta["num_features"])
    families: list[dict[str, Any]] = []
    hidden_post_bounds: list[int] = []
    for layer_idx, layer_meta in enumerate(hidden_layers):
        layer = int(layer_meta["layer_index"])
        if layer != layer_idx:
            raise ValueError("checkpoint hidden layer indices are not contiguous")
        input_dim = int(layer_meta["input_dim"])
        head_count = int(layer_meta["head_count"])
        head_dim = int(layer_meta["head_dim"])
        if input_dim != current_dim or head_count <= 0 or head_dim <= 0:
            raise ValueError("checkpoint hidden dimensions are not closed")
        layer_post_bound = 0
        for head in range(head_count):
            weight = max(0, maxima.get(f"hidden/layer{layer}/head{head}/seq", 0))
            attention = max(
                0,
                maxima.get(f"hidden/layer{layer}/head{head}/attn_src", 0),
                maxima.get(f"hidden/layer{layer}/head{head}/attn_dst", 0),
            )
            projected = input_dim * current_bound * weight
            score = 2 * head_dim * projected * attention
            aggregation_num = degree * exp_zero_output * projected
            divisor_min = exp_zero_output * s_h_prime[layer_idx][head]
            post_bound = round_ratio_away(aggregation_num * s_agg_pre[layer_idx], divisor_min)
            layer_post_bound = max(layer_post_bound, post_bound)
            families.append(
                {
                    "layer": layer,
                    "head": head,
                    "is_output": False,
                    "projected": projected,
                    "score": score,
                    "weight": weight,
                    "attention": attention,
                    "bias": 0,
                    "recovery_scale": s_agg_pre[layer_idx],
                    "denominator_scale": s_h_prime[layer_idx][head],
                }
            )
        hidden_post_bounds.append(layer_post_bound)
        current_dim = head_count * head_dim
        current_bound = layer_post_bound

    classes = int(checkpoint["C"])
    if classes != int(meta["num_classes"]):
        raise ValueError("checkpoint class count disagrees with dataset metadata")
    for head in range(output_heads):
        weight = max(0, maxima.get(f"output/head{head}/seq", 0))
        attention = max(
            0,
            maxima.get(f"output/head{head}/attn_src", 0),
            maxima.get(f"output/head{head}/attn_dst", 0),
        )
        bias = max(0, maxima.get(f"output/head{head}", 0))
        projected = current_dim * current_bound * weight
        score = 2 * classes * projected * attention
        families.append(
            {
                "layer": U64_MAX,
                "head": head,
                "is_output": True,
                "projected": projected,
                "score": score,
                "weight": weight,
                "attention": attention,
                "bias": bias,
                "recovery_scale": s_out_pre,
                "denominator_scale": s_y_prime[head],
            }
        )

    return {
        "families": families,
        "lrelu_bound": max((family["score"] for family in families), default=0),
        "elu_bound": max(hidden_post_bounds, default=0),
        "hidden_post_bounds": hidden_post_bounds,
        "exp_zero_output": exp_zero_output,
    }


def build_tables(
    dataset: str,
    range_bits: int,
    nodes: int,
    features: int,
    edges: int,
    degree: int,
    whole_graph_single: bool,
    lrelu_bound: int,
    elu_bound: int,
) -> dict[str, Any]:
    large = whole_graph_single and (edges >= (1 << 18) or nodes * features >= (1 << 24))
    lrelu_low = -lrelu_bound
    lrelu_high = lrelu_bound
    lrelu_probe = {"function_id": "lrelu", "input_min": lrelu_low, "input_max": lrelu_high}
    lrelu_output_bound = max(abs(table_value(lrelu_probe, lrelu_low)), abs(table_value(lrelu_probe, lrelu_high)))
    exp_bound = 2 * lrelu_output_bound
    elu_probe = {"function_id": "elu", "input_min": -elu_bound, "input_max": elu_bound}
    elu_output_bound = max(abs(table_value(elu_probe, -elu_bound)), abs(table_value(elu_probe, elu_bound)))
    max_relu_bound = max(2 * lrelu_output_bound, degree)
    domain_id = (f"dataset:{dataset}:range_bits:{range_bits}:large:{str(large).lower()}")
    generator_id = "ezkl_baseline_local_static_table_generator_v1"
    prefix = f"tables:gat:{dataset}:v1"
    tables = {
        "lrelu": make_table(
            f"{prefix}:lrelu",
            "lrelu",
            domain_id,
            generator_id,
            8,
            8,
            lrelu_low,
            lrelu_high,
        ),
        "exp": make_table(
            f"{prefix}:exp",
            "exp_neg",
            domain_id,
            generator_id,
            8,
            4,
            0,
            exp_bound,
        ),
        "elu": make_table(
            f"{prefix}:elu",
            "elu",
            domain_id,
            generator_id,
            4,
            4,
            -elu_bound,
            elu_bound,
        ),
        "max_relu": make_table(
            f"{prefix}:max_relu",
            "max_relu",
            domain_id,
            generator_id,
            8,
            8,
            -max_relu_bound,
            max_relu_bound,
        ),
        "static_tables_id": prefix,
        "backend_aux_tables_id": f"tables:gat:{dataset}:max_relu:v1",
        "static_tables_digest": "",
    }
    tables["static_tables_digest"] = static_tables_digest(tables)
    return tables


def build_ledger(
    dataset: str,
    tables: dict[str, Any],
    checkpoint: dict[str, Any],
    scales: dict[str, str],
    degree: int,
    bound_plan: dict[str, Any],
) -> dict[str, Any]:
    lrelu_output = max(
        abs(table_value(tables["lrelu"], tables["lrelu"]["input_min"])),
        abs(table_value(tables["lrelu"], tables["lrelu"]["input_max"])),
    )
    exp_output = table_value(tables["exp"], tables["exp"]["input_min"])
    elu_input = max(abs(tables["elu"]["input_min"]), abs(tables["elu"]["input_max"]))
    elu_output = max(
        abs(table_value(tables["elu"], tables["elu"]["input_min"])),
        abs(table_value(tables["elu"], tables["elu"]["input_max"])),
    )
    max_relu_input = max(abs(tables["max_relu"]["input_min"]), abs(tables["max_relu"]["input_max"]))
    table_output = max(lrelu_output, elu_output, max_relu_input)
    s_out_pre = scale_value(scales, "s_out_pre")
    families: list[dict[str, Any]] = []
    output_head_post_bounds: list[int] = []
    classes = int(checkpoint["C"])
    exp_zero_output = int(bound_plan["exp_zero_output"])
    for planned in bound_plan["families"]:
        projected = int(planned["projected"])
        recovery_scale = int(planned["recovery_scale"])
        denominator_scale = int(planned["denominator_scale"])
        summation = degree * exp_output
        numerator = summation * projected
        dividend = max(elu_input, numerator * recovery_scale)
        divisor = max(1, summation * denominator_scale)
        minimum_divisor = exp_zero_output * denominator_scale
        if planned["is_output"]:
            divisor = max(divisor, s_out_pre)
            minimum_divisor = min(minimum_divisor, s_out_pre)
        if minimum_divisor <= 0:
            raise ValueError("the reachable exponential denominator must be positive")
        quotient = round_ratio_away(numerator * recovery_scale, minimum_divisor)
        if planned["is_output"]:
            # The native relation adds each public output bias after this
            # head's recovery division, then sums heads before the final
            # arithmetic-mean division. Keep each family dividend above as
            # the true pre-bias division bound; collect the post-bias result
            # separately for the final global dividend bound.
            output_head_post_bounds.append(quotient + int(planned["bias"]))
        aux = max(
            max_relu_input,
            2 * dividend + divisor,
            2 * divisor * (quotient + 1),
            classes if planned["is_output"] else 0,
        )
        families.append(
            {
                "layer": int(planned["layer"]),
                "head": int(planned["head"]),
                "is_output": bool(planned["is_output"]),
                "max_score_abs": int(planned["score"]),
                "max_weight_abs": int(planned["weight"]),
                "max_attention_abs": int(planned["attention"]),
                "max_bias_abs": int(planned["bias"]),
                "max_table_output_abs": table_output,
                "max_delta": 2 * int(planned["score"]),
                "max_exp": exp_output,
                "max_sum": summation,
                "max_aggregation_num": numerator,
                "max_dividend_abs": dividend,
                "max_aux_abs": aux,
                "max_divisor": divisor,
            }
        )
    output_heads = int(checkpoint["K_out"])
    if output_heads <= 0 or len(output_head_post_bounds) != output_heads:
        raise ValueError("output head bounds do not match the authenticated checkpoint head count")

    # The final output relation computes sum(head_output_h) / output_heads.
    # Each head_output_h is bounded by its own recovered quotient plus its
    # authenticated bias absolute bound, so use the triangle bound across all
    # heads rather than the largest per-head bound.
    final_dividend_abs = sum(output_head_post_bounds)
    final_quotient_abs = round_ratio_away(final_dividend_abs, output_heads)
    final_aux_abs = max(
        2 * final_dividend_abs + output_heads,
        2 * output_heads * (final_quotient_abs + 1),
    )
    field_half = (int(BN254_MODULUS) - 1) // 2
    for label, value in (
        ("final output dividend", final_dividend_abs),
        ("final output quotient", final_quotient_abs),
        ("final output auxiliary", final_aux_abs),
        ("final output divisor", output_heads),
    ):
        if value < 0 or value > I128_MAX or value >= field_half:
            raise ValueError(f"{label} does not satisfy the signed-i128 BN254 no-wrap bound")
    fields = (
        "max_score_abs",
        "max_table_output_abs",
        "max_delta",
        "max_exp",
        "max_sum",
        "max_aggregation_num",
        "max_dividend_abs",
        "max_aux_abs",
        "max_divisor",
    )
    ledger = {key: max(int(family[key]) for family in families) for key in fields}
    # The global ledger also covers the final post-bias head sum and its
    # nearest-ties-away division. Per-family entries remain the exact bounds
    # for their respective pre-bias aggregation divisions.
    ledger["max_dividend_abs"] = max(ledger["max_dividend_abs"], final_dividend_abs)
    ledger["max_aux_abs"] = max(ledger["max_aux_abs"], final_aux_abs)
    ledger["max_divisor"] = max(ledger["max_divisor"], output_heads)
    ledger["max_table_output_abs"] = max(ledger["max_table_output_abs"], table_output)
    ledger["field_modulus"] = BN254_MODULUS
    ledger["ledger_id"] = f"ledger:gat:{dataset}:v1"
    ledger["ledger_digest"] = ""
    ledger["families"] = families
    ledger["ledger_digest"] = ledger_digest(ledger)
    return ledger


def table_metadata(table: dict[str, Any]) -> dict[str, Any]:
    return {
        "table_id": table["table_id"],
        "input_scale": table["input_scale"],
        "output_scale": table["output_scale"],
        "function_id": table["function_id"],
        "domain_id": table["domain_id"],
        "generator_id": table["generator_id"],
        "input_min": table["input_min"],
        "input_max": table["input_max"],
        "row_count": table["row_count"],
        "digest": table["digest"],
        "canonical_len": table_canonical_length(table),
    }


def write_table_archive(path: Path, tables: dict[str, Any]) -> None:
    """Write the authenticated lookup registry as a bounded-header stream.

    Rows are generated directly into the temporary file.  The archive uses
    u64 lengths/counts and signed i128 row values, matching the Rust loader's
    checked parser and the existing canonical table digest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(TABLE_MAGIC)
            stream.write(struct.pack("<I", TABLE_FORMAT_VERSION))
            stream.write(struct.pack("<I", len(TABLE_ORDER)))
            registry = {
                "format": "ezkl-gat-static-tables-bin",
                "version": TABLE_FORMAT_VERSION,
                "static_tables_id": tables["static_tables_id"],
                "backend_aux_tables_id": tables["backend_aux_tables_id"],
                "static_tables_digest": tables["static_tables_digest"],
                "table_order": list(TABLE_ORDER),
            }
            registry_bytes = json.dumps(registry, separators=(",", ":"), sort_keys=True).encode()
            stream.write(struct.pack("<Q", len(registry_bytes)))
            stream.write(registry_bytes)
            for key in TABLE_ORDER:
                table = tables[key]
                print(
                    f"[resource-preparation] {path.parent.name}/{table['function_id']}: writing "
                    f"{table['row_count']:,} exact rows",
                    flush=True,
                )
                metadata = json.dumps(table_metadata(table), separators=(",", ":"), sort_keys=True).encode()
                stream.write(struct.pack("<Q", len(metadata)))
                stream.write(metadata)
                stream.write(struct.pack("<Q", table["row_count"]))
                for row_index, (input_value, output_value) in enumerate(iter_table_rows(table), 1):
                    stream.write(int(input_value).to_bytes(16, "little", signed=True))
                    stream.write(int(output_value).to_bytes(16, "little", signed=True))
                    if row_index % (1 << 20) == 0:
                        print(
                            f"[resource-preparation] {path.parent.name}/{table['function_id']}: "
                            f"rows {row_index:,}/{table['row_count']:,}",
                            flush=True,
                        )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def generate_dataset(root: Path, dataset: str) -> tuple[str, str]:
    if dataset not in DATASETS:
        raise SystemExit(f"generate_gat_resources: unsupported dataset {dataset}")
    config = parse_cfg(root / "resources/canonical/configs" / f"{dataset}_secure.cfg")
    meta = parse_cfg(root / "resources/canonical/cache" / dataset / "meta.cfg")
    scales = parse_cfg(root / "resources/canonical/configs/quantization_scale_chain.cfg")
    manifest_path = root / "resources/canonical/checkpoints" / f"{dataset}_gat/manifest.json"
    checkpoint = json.loads(manifest_path.read_text())
    if (
        checkpoint.get("dataset") != dataset
        or not checkpoint.get("quant_cfg_id")
        or checkpoint.get("quant_cfg_id") != config.get("quant_cfg_id")
        or checkpoint.get("quant_cfg_id") != scales.get("quant_cfg_id")
    ):
        raise ValueError(f"{dataset} secure config, checkpoint, and scale chain identities disagree")
    nodes, edges, degree = graph_info(root, dataset, meta)
    maxima = quantized_model(root, dataset, checkpoint, scales)
    feature_max = feature_bound(root, dataset, meta)
    bound_plan = derive_bound_plan(meta, checkpoint, scales, maxima, feature_max, degree)
    tables = build_tables(
        dataset,
        int(config["range_bits"]),
        nodes,
        int(meta["num_features"]),
        edges,
        degree,
        config.get("batching_rule") == "whole_graph_single",
        int(bound_plan["lrelu_bound"]),
        int(bound_plan["elu_bound"]),
    )
    ledger = build_ledger(dataset, tables, checkpoint, scales, degree, bound_plan)
    table_path = root / "resources/canonical/tables" / dataset / "gat_static_tables.bin"
    ledger_path = root / "resources/canonical/ledgers" / dataset / "gat_integer_ledger.json"
    write_table_archive(table_path, tables)
    old_table_path = table_path.with_suffix(".json")
    if old_table_path.exists() and not old_table_path.is_symlink():
        old_table_path.unlink()
    write_json(ledger_path, ledger)
    feature_updates: dict[str, str] = {}
    if dataset in {"ppi", "ogbn_arxiv"}:
        contract = npy_contract(root / "resources/canonical/cache" / dataset / "features.npy")
        feature_updates = {
            "feature_format": "npy",
            "feature_dtype": contract["dtype"],
            "feature_shape": ",".join(str(x) for x in contract["shape"]),
            "feature_order": contract["order"],
            "feature_length": str(contract["length"]),
            "feature_sha256": contract["sha256"],
        }
    write_cfg(
        root / "resources/canonical/configs" / f"{dataset}_secure.cfg",
        {
            "static_table_id": tables["static_tables_id"],
            "static_table_digest": tables["static_tables_digest"],
            "degree_bound_id": ledger["ledger_id"],
            "degree_bound_digest": ledger["ledger_digest"],
        },
    )
    if feature_updates:
        write_cfg(
            root / "resources/canonical/cache" / dataset / "meta.cfg",
            feature_updates,
        )
    checkpoint["static_table_id"] = tables["static_tables_id"]
    checkpoint["static_table_digest"] = tables["static_tables_digest"]
    checkpoint["degree_bound_id"] = ledger["ledger_id"]
    checkpoint["degree_bound_digest"] = ledger["ledger_digest"]
    write_json(manifest_path, checkpoint)
    return tables["static_tables_digest"], ledger["ledger_digest"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        action="append",
    )
    args = parser.parse_args()
    baseline_root = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    if root != baseline_root:
        raise SystemExit(f"generate_gat_resources: --root must be this ezkl baseline ({baseline_root})")
    datasets = args.dataset or list(DATASETS)
    for dataset in datasets:
        table_digest_value, ledger_digest_value = generate_dataset(root, dataset)
        print(f"{dataset}: static_tables={table_digest_value} ledger={ledger_digest_value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
