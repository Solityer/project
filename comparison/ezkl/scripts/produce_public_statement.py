#!/usr/bin/env python3
"""Produce the external ezkl GAT statement and public Y.

This is deliberately a separate producer from the ezkl binary.  It reads the
authenticated baseline resources and the setup-only public context, evaluates
the canonical integer GAT relation, and emits only verifier-facing statement
objects.  It never invokes ``gat-witness``, ``prove`` or ``verify`` and never
loads a compiled-circuit witness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import struct
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from formal_common import DATASETS, ROOT, ContractError, abort, load_json, require_file, sha256


RELATION = "R_GAT-ezkl"
PRODUCER_ID = "ezkl-gat-independent-python"
PRODUCER_VERSION = "1"
TABLE_MAGIC = b"EZKL-GAT-TABLES\x00"
BN254_MODULUS = int(
    "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001", 16
)
NATIVE_FIELD_HEX_PREFIX = "0x"


def fail(message: str) -> None:
    raise SystemExit(f"independent GAT producer failed: {message}")


def local_path(value: str, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        fail(f"{label} must be baseline-local: {value}")
    resolved = (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        fail(f"{label} escapes the baseline: {value}")
    if resolved.is_symlink():
        fail(f"{label} must not be a symlink: {value}")
    return resolved


def parse_cfg(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            fail(f"malformed config line {path}:{number}")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or key in result:
            fail(f"duplicate/empty config key {path}:{number}")
        result[key] = value
    return result


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def i128_bytes(value: int) -> bytes:
    try:
        return int(value).to_bytes(16, "little", signed=True)
    except OverflowError as exc:
        fail(f"integer does not fit signed i128: {value}")
        raise AssertionError from exc


def u64_bytes(value: int) -> bytes:
    if value < 0 or value >= 1 << 64:
        fail(f"unsigned identity value is outside u64: {value}")
    return int(value).to_bytes(8, "little")


def digest_i128(values: Iterable[int]) -> str:
    payload = bytearray(b"ezkl-gat-public-y-v1")
    for value in values:
        payload.extend(i128_bytes(value))
    return sha256_bytes(bytes(payload))


def canonical_model_digest(values: Iterable[int], hash_domain: str) -> str:
    if not hash_domain:
        fail("authenticated model hash domain is empty")
    payload = bytearray(hash_domain.encode())
    for value in values:
        payload.extend(i128_bytes(value))
    return sha256_bytes(bytes(payload))


def authenticated_model_binding(context: dict[str, Any]) -> tuple[str, str, str]:
    """Read the explicit setup model binding without rewriting either value."""

    model = context.get("model")
    if not isinstance(model, dict):
        fail("setup producer context is missing authenticated model binding")
    if set(model) != {"model_hash_field", "model_digest_sha256", "hash_domain"}:
        fail("setup producer context model binding schema is not canonical")
    model_hash = model.get("model_hash_field")
    if (
        not isinstance(model_hash, str)
        or len(model_hash) != 66
        or not model_hash.startswith(NATIVE_FIELD_HEX_PREFIX)
        or any(char not in "0123456789abcdef" for char in model_hash[2:])
    ):
        fail("setup producer context model hash is not a canonical Halo2 field encoding")
    model_digest = model.get("model_digest_sha256")
    if (
        not isinstance(model_digest, str)
        or len(model_digest) != 64
        or any(char not in "0123456789abcdef" for char in model_digest)
    ):
        fail("setup producer context model digest is not a canonical SHA-256")
    model_hash_domain = model.get("hash_domain")
    if not isinstance(model_hash_domain, str) or not model_hash_domain:
        fail("setup producer context model hash domain is empty")
    return model_hash, model_digest, model_hash_domain


def rust_digest_identity(value: dict[str, Any], label: str) -> dict[str, str]:
    """Serialize a DigestIdentity in the native Rust struct field order.

    ``producer_context.json`` is a serde_json::Value and therefore exposes
    object keys in its canonical map order.  The verifier-facing statement is
    a typed Rust ``PublicStatement``; its ``DigestIdentity`` fields serialize
    as ``id`` then ``digest``.  The statement digest covers bytes, so this
    order is part of the cross-language boundary and cannot be left to the
    Python dictionary inherited from the setup context.
    """

    if set(value) != {"id", "digest"}:
        fail(f"{label} is not a canonical digest identity")
    identity = value.get("id")
    digest = value.get("digest")
    if not isinstance(identity, str) or not identity:
        fail(f"{label}.id is empty")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        fail(f"{label}.digest is not a canonical SHA-256")
    return {"id": identity, "digest": digest}


def topology_digest(
    n_total: int,
    n: int,
    e: int,
    g_batch: int,
    node_ids: list[int],
    src: list[int],
    dst: list[int],
    node_ptr: list[int],
    edge_ptr: list[int],
) -> str:
    payload = bytearray(b"ezkl-gat-topology-v1")
    for value in (n_total, n, e, g_batch):
        payload.extend(u64_bytes(value))
    for values in (node_ids, src, dst, node_ptr, edge_ptr):
        payload.extend(u64_bytes(len(values)))
        for value in values:
            payload.extend(u64_bytes(value))
    return sha256_bytes(bytes(payload))


def round_away(value: float) -> int:
    if not math.isfinite(value):
        fail("non-finite value in canonical quantization")
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def quantize(value: float, exponent: int) -> int:
    scaled = float(value) * float(2**exponent)
    if not math.isfinite(scaled):
        fail("non-finite scaled value in canonical quantization")
    return round_away(scaled)


def parse_shape(value: str) -> list[int]:
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def scale_exponent(raw_multiplier: int, key: str) -> int:
    if raw_multiplier <= 0 or raw_multiplier & (raw_multiplier - 1):
        fail(
            f"quantization scale {key} must be a positive power-of-two multiplier"
        )
    return raw_multiplier.bit_length() - 1


def scale_list(config: dict[str, str], key: str, count: int) -> list[int]:
    raw_values = [int(piece.strip()) for piece in config[key].split(",") if piece.strip()]
    if len(raw_values) != count:
        fail(f"quantization scale {key} does not match authenticated width {count}")
    return [scale_exponent(value, key) for value in raw_values]


def scale_value(config: dict[str, str], key: str) -> int:
    return scale_exponent(int(config[key]), key)


def tensor_archive(path: Path) -> dict[str, tuple[list[int], list[float]]]:
    result: dict[str, tuple[list[int], list[float]]] = {}
    current_name: str | None = None
    current_shape: list[int] = []
    values: list[float] = []

    def flush() -> None:
        nonlocal current_name, current_shape, values
        if current_name is None:
            return
        expected = math.prod(current_shape)
        if expected != len(values):
            fail(f"tensor {current_name} count mismatch")
        result[current_name] = (current_shape, values)
        current_name = None
        current_shape = []
        values = []

    with path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("TENSOR "):
                flush()
                pieces = line.split()
                if len(pieces) < 4:
                    fail(f"malformed tensor header in {path}")
                current_name = pieces[1]
                rank = int(pieces[2])
                if len(pieces) != 4 + rank:
                    fail(f"tensor header shape mismatch for {current_name}")
                current_shape = [int(x) for x in pieces[3 : 3 + rank]]
                declared = int(pieces[3 + rank])
                if declared != math.prod(current_shape):
                    fail(f"tensor header count mismatch for {current_name}")
                continue
            if current_name is None:
                fail("tensor values precede first tensor header")
            for token in line.split():
                number = float(token)
                if not math.isfinite(number):
                    fail(f"tensor {current_name} contains a non-finite value")
                values.append(number)
    flush()
    return result


def quantized_tensor(
    tensors: dict[str, tuple[list[int], list[float]]], name: str, exponent: int
) -> np.ndarray:
    if name not in tensors:
        fail(f"missing canonical tensor: {name}")
    shape, values = tensors[name]
    return np.asarray([quantize(value, exponent) for value in values], dtype=np.int64).reshape(shape)


class BinaryTable:
    """Bounded binary-search view over one authenticated table section."""

    def __init__(self, mapped: mmap.mmap, offset: int, metadata: dict[str, Any], count: int):
        self.mapped = mapped
        self.offset = offset
        self.metadata = metadata
        self.count = count
        self.minimum = int(metadata["input_min"])
        self.maximum = int(metadata["input_max"])
        if count != self.maximum - self.minimum + 1:
            fail(f"table row/domain mismatch for {metadata.get('table_id')}")

    def lookup(self, value: int) -> int:
        if value < self.minimum or value > self.maximum:
            fail(
                f"{self.metadata.get('table_id')} lookup {value} outside "
                f"[{self.minimum},{self.maximum}]"
            )
        index = value - self.minimum
        start = self.offset + index * 32
        raw = self.mapped[start + 16 : start + 32]
        return int.from_bytes(raw, "little", signed=True)


def open_tables(path: Path, expected_id: str, expected_digest: str) -> tuple[mmap.mmap, dict[str, BinaryTable], dict[str, Any]]:
    stream = path.open("rb")
    mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    if mapped[:16] != TABLE_MAGIC:
        fail(f"invalid static-table archive magic: {path}")
    offset = 16
    version, table_count = struct.unpack_from("<II", mapped, offset)
    offset += 8
    if version != 1 or table_count != 4:
        fail("unsupported static-table archive version/count")
    (registry_size,) = struct.unpack_from("<Q", mapped, offset)
    offset += 8
    registry = json.loads(bytes(mapped[offset : offset + registry_size]))
    offset += registry_size
    if registry.get("format") != "ezkl-gat-static-tables-bin" or registry.get("version") != 1:
        fail("static-table registry format mismatch")
    if registry.get("static_tables_id") != expected_id or registry.get("static_tables_digest") != expected_digest:
        fail("static-table registry does not match the authenticated setup context")
    tables: dict[str, BinaryTable] = {}
    for expected_function in ("lrelu", "exp_neg", "elu", "max_relu"):
        (metadata_size,) = struct.unpack_from("<Q", mapped, offset)
        offset += 8
        metadata = json.loads(bytes(mapped[offset : offset + metadata_size]))
        offset += metadata_size
        (count,) = struct.unpack_from("<Q", mapped, offset)
        offset += 8
        if metadata.get("function_id") != expected_function:
            fail(f"static table function mismatch: {expected_function}")
        tables[expected_function] = BinaryTable(mapped, offset, metadata, int(count))
        offset += int(count) * 32
    return mapped, tables, registry


def graph_from_resources(data_dir: Path, meta: dict[str, str]) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    nodes = int(meta["num_nodes"])
    graph_count = int(meta.get("graph_count", "1"))
    if graph_count > 1:
        node_ptr = [int(x) for x in (data_dir / "node_ptr.txt").read_text().split()]
    else:
        node_ptr = [0, nodes]
    if len(node_ptr) != graph_count + 1 or node_ptr[0] != 0 or node_ptr[-1] != nodes:
        fail("node_ptr does not match authenticated graph metadata")
    graph_for_node = [0] * nodes
    for graph in range(graph_count):
        for node in range(node_ptr[graph], node_ptr[graph + 1]):
            graph_for_node[node] = graph
    raw: list[tuple[int, int, int]] = []
    incoming: set[int] = set()
    with (data_dir / "edges.txt").open() as stream:
        for number, line in enumerate(stream, 1):
            pieces = line.split()
            if not pieces:
                continue
            if graph_count > 1:
                if len(pieces) != 3:
                    fail(f"invalid batched edge at line {number}")
                graph, source, destination = (int(x) for x in pieces)
                if graph != graph_for_node[source]:
                    fail(f"batched edge graph mismatch at line {number}")
            else:
                if len(pieces) != 2:
                    fail(f"invalid edge at line {number}")
                source, destination = (int(x) for x in pieces)
                graph = graph_for_node[source]
            if source < 0 or destination < 0 or source >= nodes or destination >= nodes:
                fail(f"edge endpoint outside graph at line {number}")
            if graph_for_node[destination] != graph:
                fail(f"edge crosses graph boundary at line {number}")
            raw.append((graph, source, destination))
            incoming.add(destination)
    for node in range(nodes):
        if node not in incoming:
            raw.append((graph_for_node[node], node, node))
    raw.sort(key=lambda value: (value[0], value[2]))
    src = [value[1] for value in raw]
    dst = [value[2] for value in raw]
    edge_ptr = [0]
    cursor = 0
    for graph in range(graph_count):
        while cursor < len(raw) and raw[cursor][0] == graph:
            cursor += 1
        edge_ptr.append(cursor)
    if cursor != len(raw):
        fail("edge grouping failed")
    node_ids = list(range(nodes))
    return node_ids, src, dst, node_ptr, edge_ptr


def feature_values(
    data_dir: Path, meta: dict[str, str], exponent: int, preprocess_id: str
) -> tuple[np.ndarray, Path]:
    nodes, columns = int(meta["num_nodes"]), int(meta["num_features"])
    npy_path = data_dir / "features.npy"
    text_path = data_dir / "features.txt"
    if npy_path.is_file():
        feature_path = npy_path
        raw = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if raw.shape != (nodes, columns) or raw.dtype not in (np.dtype("<f4"), np.dtype("<f8")):
            fail("authenticated feature NPY shape/dtype mismatch")
        result = np.empty((nodes, columns), dtype=np.int64)
        for row in range(nodes):
            values = np.asarray(raw[row], dtype=np.float64)
            scaled = values * float(2**exponent)
            quantized = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
            if not np.all(np.isfinite(quantized)):
                fail("feature NPY contains a non-finite value")
            result[row, :] = quantized.astype(np.int64, copy=False)
        return result, feature_path
    if not text_path.is_file():
        fail(f"authenticated feature payload is missing: {data_dir}")
    result = np.zeros((nodes, columns), dtype=np.int64)
    with text_path.open() as stream:
        header = stream.readline().split()
        if len(header) != 2 or tuple(int(x) for x in header) != (nodes, columns):
            fail("sparse feature shape mismatch")
        row_count = 0
        for row, line in enumerate(stream):
            if row >= nodes:
                fail("sparse feature file has too many rows")
            raw: list[tuple[int, float]] = []
            for token in line.split():
                index, value = token.split(":", 1)
                raw.append((int(index), float(value)))
            normalizer = sum(value for _, value in raw) if preprocess_id == "row_normalize" else 1.0
            if not math.isfinite(normalizer) or normalizer < 0.0:
                fail(f"invalid sparse feature normalization at row {row}")
            if normalizer == 0.0:
                normalizer = 1.0
            for index, value in raw:
                if index < 0 or index >= columns:
                    fail(f"feature index outside shape at row {row}")
                result[row, index] = quantize(value / normalizer, exponent)
            row_count += 1
        if row_count != nodes:
            fail("sparse feature row count mismatch")
    return result, text_path


def signed_round_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        fail("canonical signed division has a non-positive divisor")
    negative = value < 0
    absolute = abs(int(value))
    quotient, remainder = divmod(absolute, divisor)
    if remainder * 2 >= divisor:
        quotient += 1
    return -quotient if negative else quotient


def checked_matmul(left: np.ndarray, right: np.ndarray, label: str) -> np.ndarray:
    if left.dtype != np.int64 or right.dtype != np.int64:
        fail(f"{label} does not use canonical int64 intermediates")
    estimate = int(np.max(np.abs(left))) * int(np.max(np.abs(right))) * left.shape[1]
    if estimate >= 1 << 62:
        fail(f"{label} exceeds the bounded signed accumulator")
    result = left @ right
    if result.size and int(np.max(np.abs(result))) >= 1 << 62:
        fail(f"{label} overflowed the bounded signed accumulator")
    return result


def aggregate(
    projected: np.ndarray,
    src: list[int],
    dst: list[int],
    groups: list[tuple[int, int, int]],
    a_src: np.ndarray,
    a_dst: np.ndarray,
    lrelu: BinaryTable,
    exp: BinaryTable,
    elu: BinaryTable | None,
    numerator_scale: int,
    denominator_scale: int,
) -> np.ndarray:
    d_out = projected.shape[1]
    src_index = np.asarray(src, dtype=np.int64)
    dst_index = np.asarray(dst, dtype=np.int64)
    src_scores = np.sum(projected[src_index] * a_src.reshape(1, -1), axis=1, dtype=np.int64)
    dst_scores = np.sum(projected[dst_index] * a_dst.reshape(1, -1), axis=1, dtype=np.int64)
    scores = [lrelu.lookup(int(x)) for x in (src_scores + dst_scores).tolist()]
    result = np.zeros((projected.shape[0], d_out), dtype=np.int64)
    for node, start, end in groups:
        group_scores = scores[start:end]
        maximum = max(group_scores)
        exponents = [exp.lookup(maximum - score) for score in group_scores]
        denominator = sum(exponents)
        if denominator <= 0:
            fail("stable-softmax denominator is not positive")
        rows = projected[np.asarray(src[start:end], dtype=np.int64)]
        weighted = rows * np.asarray(exponents, dtype=np.int64).reshape(-1, 1)
        numerators = np.sum(weighted, axis=0, dtype=np.int64).tolist()
        values = [
            signed_round_div(int(numerator) * numerator_scale, denominator * denominator_scale)
            for numerator in numerators
        ]
        if elu is not None:
            values = [elu.lookup(value) for value in values]
        result[node, :] = np.asarray(values, dtype=np.int64)
    return result


def canonical_forward(
    features: np.ndarray,
    src: list[int],
    dst: list[int],
    node_ids: list[int],
    groups: list[tuple[int, int, int]],
    hidden: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    tables: dict[str, BinaryTable],
    hidden_agg_pre: list[int],
    hidden_h_prime: list[list[int]],
    output_pre: int,
    output_y_prime: list[int],
    output_head: int,
) -> list[int]:
    current = features[np.asarray(node_ids, dtype=np.int64)].copy()
    for layer_index, layer in enumerate(hidden):
        heads: list[np.ndarray] = []
        for head_index, (projection, a_src, a_dst) in enumerate(layer):
            projected = checked_matmul(current, projection, "hidden projection")
            heads.append(
                aggregate(
                    projected,
                    src,
                    dst,
                    groups,
                    a_src,
                    a_dst,
                    tables["lrelu"],
                    tables["exp_neg"],
                    tables["elu"],
                    1 << hidden_agg_pre[layer_index],
                    1 << hidden_h_prime[layer_index][head_index],
                )
            )
        current = np.concatenate(heads, axis=1)
    head_outputs: list[np.ndarray] = []
    for head_index, (projection, a_src, a_dst, bias) in enumerate(outputs):
        projected = checked_matmul(current, projection, "output projection")
        aggregated = aggregate(
            projected,
            src,
            dst,
            groups,
            a_src,
            a_dst,
            tables["lrelu"],
            tables["exp_neg"],
            None,
            1 << output_pre,
            1 << output_y_prime[head_index],
        )
        aligned = np.asarray(
            [signed_round_div(int(value) * (1 << output_head), 1 << output_pre) for value in aggregated.ravel()],
            dtype=np.int64,
        ).reshape(aggregated.shape)
        head_outputs.append(aligned + bias.reshape(1, -1))
    summed = np.sum(np.stack(head_outputs, axis=0), axis=0, dtype=np.int64)
    output = np.asarray(
        [signed_round_div(int(value), len(head_outputs)) for value in summed.ravel()],
        dtype=np.int64,
    )
    return [int(value) for value in output.tolist()]


def groups_for_graph(node_ids: list[int], dst: list[int], node_ptr: list[int], edge_ptr: list[int]) -> list[tuple[int, int, int]]:
    groups: list[tuple[int, int, int]] = []
    for graph in range(len(node_ptr) - 1):
        node_start, node_end = node_ptr[graph], node_ptr[graph + 1]
        edge_start, edge_end = edge_ptr[graph], edge_ptr[graph + 1]
        cursor = edge_start
        for node in range(node_start, node_end):
            begin = cursor
            while cursor < edge_end and dst[cursor] == node:
                cursor += 1
            if begin == cursor:
                fail(f"destination node {node} has no incoming edge")
            groups.append((node, begin, cursor))
        if cursor != edge_end:
            fail("edge groups are not sorted by destination")
    return groups


def write_new(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        fail(f"refusing to overwrite existing producer output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=DATASETS, required=True)
    result.add_argument("--resource-manifest", default="resources/resource_manifest.json")
    result.add_argument("--setup-context", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--config")
    result.add_argument("--statement", default="statement.json")
    result.add_argument("--public-output", default="public_output.json")
    result.add_argument("--producer-manifest", default="producer_manifest.json")
    result.add_argument(
        "--tiny-fixture",
        action="store_true",
        help="explicitly enable the bounded local fixture path; formal runners never pass this flag",
    )
    return result


def main(args: argparse.Namespace) -> int:
    if args.tiny_fixture and os.environ.get("EZKL_ALLOW_TINY_FIXTURE") != "1":
        fail("tiny fixture mode requires EZKL_ALLOW_TINY_FIXTURE=1")
    manifest_path = local_path(args.resource_manifest, "resource manifest")
    expected_manifest = ROOT / "resources/resource_manifest.json"
    if not args.tiny_fixture and manifest_path != expected_manifest:
        fail("producer requires the baseline canonical resource manifest")
    if not manifest_path.is_file():
        fail(f"missing resource manifest: {manifest_path}")
    config = local_path(
        args.config or f"resources/canonical/configs/{args.dataset}_secure.cfg", "config"
    )
    output_dir = local_path(args.output_dir, "output directory")
    setup_context_path = local_path(args.setup_context, "setup context")
    statement_path = local_path(args.statement, "statement")
    public_output_path = local_path(args.public_output, "public output")
    producer_manifest_path = local_path(args.producer_manifest, "producer manifest")
    for path, label in (
        (config, "config"),
        (setup_context_path, "setup context"),
        (manifest_path, "resource manifest"),
    ):
        require_file(path, label)
    if statement_path.parent != output_dir or public_output_path.parent != output_dir or producer_manifest_path.parent != output_dir:
        fail("statement, public output and producer manifest must be in output-dir")

    from validate_resources import validate_manifest  # imported only after ROOT is fixed

    resource_document = load_json(manifest_path, "resource manifest")
    validate_manifest(ROOT, resource_document, False)
    resource_digest = sha256(manifest_path)
    context = load_json(setup_context_path, "setup producer context")
    expected_context = {
        "schema_version",
        "relation_id",
        "dataset",
        "model_arch_id",
        "quant_cfg_id",
        "quantization",
        "static_tables",
        "ledger",
        "topology",
        "model",
        "circuit",
        "ezkl_security_profile_id",
    }
    if set(context) != expected_context or context["schema_version"] != 1 or context["relation_id"] != RELATION:
        fail("setup producer context schema/identity mismatch")
    if context["dataset"] != args.dataset:
        fail("setup producer context dataset mismatch")
    model_hash_field, model_digest_sha256_expected, model_hash_domain = authenticated_model_binding(
        context
    )
    secure = parse_cfg(config)
    data_root = local_path(secure["data_root"], "authenticated data_root")
    data_dir = data_root / args.dataset
    checkpoint_root = local_path(secure["checkpoint_bundle"], "authenticated checkpoint_bundle")
    checkpoint_manifest_path = checkpoint_root / "manifest.json"
    tensor_path = checkpoint_root / "tensors.txt"
    meta = parse_cfg(data_dir / "meta.cfg")
    checkpoint = load_json(checkpoint_manifest_path, "checkpoint manifest")
    tensors = tensor_archive(tensor_path)
    scales_cfg = parse_cfg(ROOT / "resources/canonical/configs/quantization_scale_chain.cfg")
    if (
        checkpoint.get("dataset") != args.dataset
        or checkpoint.get("model_arch_id") != context["model_arch_id"]
        or secure.get("model_arch_id") != context["model_arch_id"]
        or checkpoint.get("quant_cfg_id") != context["quant_cfg_id"]
        or secure.get("quant_cfg_id") != context["quant_cfg_id"]
        or checkpoint.get("input_preprocess_id") != meta.get("input_preprocess_id")
        or secure.get("input_preprocess_id") != meta.get("input_preprocess_id")
    ):
        fail("authenticated config/checkpoint/metadata identity mismatch")
    preprocess_id = meta.get("input_preprocess_id", "")
    if preprocess_id not in {"row_normalize", "none"}:
        fail(f"unsupported authenticated input_preprocess_id: {preprocess_id}")
    if (
        scales_cfg.get("quant_cfg_id") != context["quant_cfg_id"]
        or secure.get("quant_cfg_id") != context["quant_cfg_id"]
    ):
        fail("quantization config identity mismatch")
    s_input = scale_value(scales_cfg, "s_input")
    features, feature_path = feature_values(data_dir, meta, s_input, preprocess_id)
    node_ids, src, dst, node_ptr, edge_ptr = graph_from_resources(data_dir, meta)
    topo_digest = topology_digest(len(features), len(node_ids), len(src), len(node_ptr) - 1, node_ids, src, dst, node_ptr, edge_ptr)
    if context["topology"]["digest"] != topo_digest or context["topology"]["id"] != f"topology:{args.dataset}:{len(node_ptr)-1}":
        fail("setup topology identity disagrees with independently reconstructed graph")

    hidden_meta = checkpoint.get("hidden_layers")
    hidden_specs = checkpoint.get("hidden_head_specs")
    output_specs = checkpoint.get("output_head_specs")
    if not isinstance(hidden_meta, list) or not isinstance(hidden_specs, list) or not isinstance(output_specs, list):
        fail("checkpoint manifest is missing authenticated layer/head specifications")
    output_head_count = len(output_specs)
    checkpoint_k_out = checkpoint.get("K_out")
    config_k_out = secure.get("K_out")
    if (
        checkpoint.get("L") != len(hidden_meta) + 1
        or checkpoint_k_out is None
        or str(checkpoint_k_out) != str(output_head_count)
        or secure.get("L") != str(len(hidden_meta) + 1)
        or (config_k_out is not None and config_k_out != str(output_head_count))
    ):
        fail("authenticated layer/head schema is not closed")
    hidden: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    model_values: list[int] = []
    current_dim = features.shape[1]
    hidden_agg_pre: list[int] = []
    hidden_h_prime: list[list[int]] = []
    agg_pre_scales = scale_list(scales_cfg, "s_agg_pre", len(hidden_meta))
    for layer_meta in sorted(hidden_meta, key=lambda value: int(value["layer_index"])):
        layer_index = int(layer_meta["layer_index"])
        if layer_index != len(hidden):
            fail("checkpoint hidden layer indices are not contiguous")
        head_count = int(layer_meta["head_count"])
        head_dim = int(layer_meta["head_dim"])
        input_dim = int(layer_meta["input_dim"])
        if input_dim != current_dim:
            fail(f"checkpoint hidden layer {layer_index} input dimension is not closed")
        layer_specs = sorted(
            (
                spec
                for spec in hidden_specs
                if int(spec["layer_index"]) == layer_index
            ),
            key=lambda value: int(value["local_head_index"]),
        )
        if len(layer_specs) != head_count:
            fail(f"checkpoint hidden layer {layer_index} head count is not closed")
        w_scales = scale_list(scales_cfg, f"s_w_hidden_{layer_index}", head_count)
        src_scales = scale_list(scales_cfg, f"s_a_src_hidden_{layer_index}", head_count)
        dst_scales = scale_list(scales_cfg, f"s_a_dst_hidden_{layer_index}", head_count)
        h_prime = scale_list(scales_cfg, f"s_h_prime_hidden_{layer_index}", head_count)
        layer_heads: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for head_index, spec in enumerate(layer_specs):
            projection = quantized_tensor(tensors, spec["seq_kernel"], w_scales[head_index])
            projection = projection.reshape(current_dim, head_dim)
            a_src = quantized_tensor(tensors, spec["attn_src_kernel"], src_scales[head_index]).reshape(-1)
            a_dst = quantized_tensor(tensors, spec["attn_dst_kernel"], dst_scales[head_index]).reshape(-1)
            if projection.shape != (current_dim, head_dim) or a_src.shape != (head_dim,) or a_dst.shape != (head_dim,):
                fail(f"checkpoint hidden layer {layer_index} head {head_index} tensor shape is not closed")
            layer_heads.append((projection, a_src, a_dst))
            model_values.extend(int(x) for x in projection.ravel())
            model_values.extend(int(x) for x in a_src)
            model_values.extend(int(x) for x in a_dst)
        hidden.append(layer_heads)
        hidden_agg_pre.append(agg_pre_scales[layer_index])
        hidden_h_prime.append(h_prime)
        current_dim = head_count * head_dim
    outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    output_count = output_head_count
    if checkpoint_k_out is None or str(checkpoint_k_out) != str(output_count):
        fail("checkpoint output head count is not closed")
    output_w_scales = scale_list(scales_cfg, "s_w_output", output_count)
    output_src_scales = scale_list(scales_cfg, "s_a_src_output", output_count)
    output_dst_scales = scale_list(scales_cfg, "s_a_dst_output", output_count)
    output_bias_scales = scale_list(scales_cfg, "s_b_output", output_count)
    for head_index, spec in enumerate(sorted(output_specs, key=lambda value: int(value["head_index"]))):
        projection = quantized_tensor(tensors, spec["seq_kernel"], output_w_scales[head_index]).reshape(
            current_dim, int(meta["num_classes"])
        )
        a_src = quantized_tensor(tensors, spec["attn_src_kernel"], output_src_scales[head_index]).reshape(-1)
        a_dst = quantized_tensor(tensors, spec["attn_dst_kernel"], output_dst_scales[head_index]).reshape(-1)
        bias = quantized_tensor(tensors, spec["output_bias"], output_bias_scales[head_index]).reshape(-1)
        if (
            projection.shape != (current_dim, int(meta["num_classes"]))
            or a_src.shape != (int(meta["num_classes"]),)
            or a_dst.shape != (int(meta["num_classes"]),)
            or bias.shape != (int(meta["num_classes"]),)
        ):
            fail(f"checkpoint output head {head_index} tensor shape is not closed")
        outputs.append((projection, a_src, a_dst, bias))
        model_values.extend(int(x) for x in projection.ravel())
        model_values.extend(int(x) for x in a_src)
        model_values.extend(int(x) for x in a_dst)
        model_values.extend(int(x) for x in bias)
    model_digest_sha256 = canonical_model_digest(model_values, model_hash_domain)
    if model_digest_sha256 != model_digest_sha256_expected:
        fail("independent canonical model digest disagrees with setup context")

    table_path = ROOT / "resources/canonical/tables" / args.dataset / "gat_static_tables.bin"
    table_map, tables, _registry = open_tables(
        table_path, context["static_tables"]["id"], context["static_tables"]["digest"]
    )
    try:
        groups = groups_for_graph(node_ids, dst, node_ptr, edge_ptr)
        output = canonical_forward(
            features,
            src,
            dst,
            node_ids,
            groups,
            hidden,
            outputs,
            tables,
            hidden_agg_pre,
            hidden_h_prime,
            scale_value(scales_cfg, "s_out_pre"),
            scale_list(scales_cfg, "s_y_prime", output_count),
            scale_value(scales_cfg, "s_out_head"),
        )
    finally:
        table_map.close()

    public_output = {"dataset": args.dataset, "Y": output}
    output_bytes = (json.dumps(public_output, indent=2, ensure_ascii=False) + "\n").encode()
    output_digest = digest_i128(output)
    public_output_sha = sha256_bytes(output_bytes)
    circuit = context["circuit"]
    statement: dict[str, Any] = {
        "schema_version": 1,
        "relation_id": RELATION,
        "proof_scope": "full_gat",
        "privacy_policy": "model_and_trace_hidden",
        "public_statement_mode": "external",
        "output_mode": "public_logits",
        "dataset": args.dataset,
        "model_arch_id": context["model_arch_id"],
        "quant_cfg_id": context["quant_cfg_id"],
        "static_table_id": context["static_tables"]["id"],
        "degree_bound_id": context["ledger"]["id"],
        "ezkl_security_profile_id": context["ezkl_security_profile_id"],
        "public_ezkl": {
            "topology": rust_digest_identity(context["topology"], "context.topology"),
            "static_tables": rust_digest_identity(
                context["static_tables"], "context.static_tables"
            ),
            "quant": rust_digest_identity(context["quantization"], "context.quantization"),
            "ledger": rust_digest_identity(context["ledger"], "context.ledger"),
            "circuit": circuit,
            "model_hash": model_hash_field,
            "model": {
                "hash": model_hash_field,
                "hash_domain": model_hash_domain,
                "canonical_model_digest": model_digest_sha256_expected,
            },
            "feature_table": [int(x) for x in features.ravel().tolist()],
            "node_ids": node_ids,
            "src": src,
            "dst": dst,
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
            "y": output,
        },
        "output_digest": output_digest,
        "public_output_sha256": public_output_sha,
        "statement_digest": "",
    }
    # Rust hashes the typed statement after converting it through
    # serde_json::Value, whose default map is recursively key-ordered.  Keep
    # the producer's bytes independent of Python insertion order and Rust
    # struct/flatten field order.
    statement_digest = sha256_bytes(
        json.dumps(
            statement,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    statement["statement_digest"] = statement_digest
    script_path = ROOT / "scripts/produce_public_statement.py"
    producer_manifest = {
        "schema_version": 1,
        "producer_id": PRODUCER_ID,
        "producer_version": PRODUCER_VERSION,
        "independent": True,
        "relation_id": RELATION,
        "dataset": args.dataset,
        "producer_script": "scripts/produce_public_statement.py",
        "producer_script_sha256": sha256(script_path),
        "config_path": config.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config),
        "resource_manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "resource_manifest_sha256": resource_digest,
        "checkpoint_manifest_path": checkpoint_manifest_path.relative_to(ROOT).as_posix(),
        "checkpoint_manifest_sha256": sha256(checkpoint_manifest_path),
        "tensor_archive_path": tensor_path.relative_to(ROOT).as_posix(),
        "tensor_archive_sha256": sha256(tensor_path),
        "feature_path": feature_path.relative_to(ROOT).as_posix(),
        "feature_sha256": sha256(feature_path),
        "setup_context_path": setup_context_path.relative_to(ROOT).as_posix(),
        "setup_context_sha256": sha256(setup_context_path),
        "model_arch_id": context["model_arch_id"],
        "model_hash": model_hash_field,
        "hash_domain": model_hash_domain,
        "canonical_model_digest": model_digest_sha256_expected,
        "circuit_ref": circuit,
        "input_digest": sha256_bytes(
            b"ezkl-gat-public-input-v1"
            + b"".join(i128_bytes(int(x)) for x in features.ravel().tolist())
            + bytes.fromhex(topo_digest)
        ),
        "output_digest": output_digest,
        "public_output_sha256": public_output_sha,
        "statement_sha256": sha256_bytes((json.dumps(statement, indent=2, ensure_ascii=False) + "\n").encode()),
        "public_statement_path": statement_path.relative_to(ROOT).as_posix(),
        "public_output_path": public_output_path.relative_to(ROOT).as_posix(),
        "provenance": {
            "language": "python",
            "numpy_version": np.__version__,
            "algorithm": "canonical_integer_gat_v1",
            "native_ezkl_commands": [],
            "witness_loaded": False,
            "proof_commands": [],
        },
    }
    write_new(public_output_path, public_output)
    write_new(statement_path, statement)
    write_new(producer_manifest_path, producer_manifest)
    print(
        json.dumps(
            {
                "producer": PRODUCER_ID,
                "dataset": args.dataset,
                "statement": statement_path.relative_to(ROOT).as_posix(),
                "public_output": public_output_path.relative_to(ROOT).as_posix(),
                "producer_manifest": producer_manifest_path.relative_to(ROOT).as_posix(),
                "output_digest": output_digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (OSError, ValueError, json.JSONDecodeError, ContractError) as exc:
        fail(str(exc))
