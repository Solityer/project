#!/usr/bin/env python3
"""Generate the baseline-local independent public-logit resource.

This is deliberately a small, dependency-free reference implementation.  It
reads the staged dataset and canonical checkpoint bundle itself; it never
reads adapter ``canonical_forward.json`` or any proof artifact.  The runner
invokes it during setup, and the native verifier consumes the resulting JSON
as an external public statement.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROUTE = "ezkl_gat_native_compressed_csr_attention_full"
QUANT_ID = "gat_l3_scale_chain"
ROUNDING = "nearest_ties_away"


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def formal_bytes(domain: str, fields: list[tuple[str, bytes]]) -> str:
    payload = bytearray()

    def put(value: bytes) -> None:
        payload.extend(struct.pack("<Q", len(value)))
        payload.extend(value)

    put(b"GAT-ZKML/ezkl/canonical/v1")
    put(domain.encode())
    payload.extend(struct.pack("<Q", len(fields)))
    for role, value in fields:
        put(role.encode())
        put(value)
    return digest(bytes(payload))


def u64_vector(values: list[int]) -> bytes:
    return struct.pack("<Q", len(values)) + b"".join(
        struct.pack("<Q", value) for value in values
    )


def signed_i64(value: int) -> bytes:
    return struct.pack("<q", value)


def round_away(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("reference input is non-finite")
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def signed_div(a: int, d: int) -> int:
    if d <= 0:
        raise ValueError("reference SignedRoundDiv denominator is not positive")
    magnitude = abs(a)
    q = (2 * magnitude + d) // (2 * d)
    return -q if a < 0 else q


def read_meta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def file_bundle_digest(root: Path, names: list[str]) -> str:
    """Hash the exact baseline-local files, including names and lengths."""

    payload = bytearray()
    for name in names:
        path = root / name
        data = path.read_bytes()
        encoded = name.encode("utf-8")
        payload.extend(struct.pack("<Q", len(encoded)))
        payload.extend(encoded)
        payload.extend(struct.pack("<Q", len(data)))
        payload.extend(data)
    return digest(bytes(payload))


def validate_checkpoint_manifest(manifest: dict[str, Any], checkpoint: Path, dataset: str, preprocess: str) -> None:
    """Reject non-canonical/legacy bundles before reference evaluation."""

    required = {
        "manifest_kind", "dataset", "input_preprocess_id", "model_arch_id",
        "model_param_id", "hidden_layers", "hidden_head_specs",
        "output_head_specs", "K_out", "tensors_npz_sha256",
    }
    if not required.issubset(manifest):
        raise ValueError("checkpoint manifest is missing canonical bundle fields")
    if "checkpoint_prefix" in manifest or any(
        "/home/pzh/project/" in str(value) for value in manifest.values()
    ):
        raise ValueError("checkpoint manifest contains a main-experiment absolute path")
    expected_dataset = {
        "cora_full": "cora", "citeseer_full": "citeseer", "pubmed_full": "pubmed",
        "ppi_full_formal": "ppi", "ogbn_arxiv_full": "ogbn_arxiv",
    }[dataset]
    if manifest["dataset"] not in (dataset, expected_dataset, "ogbn-arxiv"):
        raise ValueError("checkpoint dataset identity mismatch")
    if manifest["input_preprocess_id"] != preprocess:
        raise ValueError("checkpoint preprocessing identity mismatch")
    npz = checkpoint / "tensors.npz"
    if not npz.is_file():
        raise ValueError("canonical checkpoint bundle is missing tensors.npz")
    expected = manifest["tensors_npz_sha256"]
    if expected != digest(npz.read_bytes()):
        raise ValueError("checkpoint tensors.npz digest mismatch")
    hidden = manifest["hidden_head_specs"]
    if not isinstance(hidden, list) or len(hidden) != 16:
        raise ValueError("canonical checkpoint must contain exactly 16 hidden heads")
    for spec in hidden:
        if not isinstance(spec, dict) or not all(
            key in spec for key in (
                "layer_index", "local_head_index", "global_head_index",
                "seq_kernel", "attn_src_kernel", "attn_dst_kernel",
                "attn_src_bias", "attn_dst_bias", "output_bias",
            )
        ):
            raise ValueError("hidden bias is not part of the canonical GAT relation")
    outputs = manifest["output_head_specs"]
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise ValueError("canonical checkpoint must contain exactly two output heads")
    if manifest["K_out"] != 2:
        raise ValueError("checkpoint K_out is not canonical")


def read_npy(path: Path) -> tuple[tuple[int, ...], str, list[float]]:
    data = path.read_bytes()
    if data[:6] != b"\x93NUMPY":
        raise ValueError(f"invalid npy magic: {path}")
    major, minor = data[6], data[7]
    del minor
    if major == 1:
        header_len = struct.unpack_from("<H", data, 8)[0]
        offset = 10
    elif major in (2, 3):
        header_len = struct.unpack_from("<I", data, 8)[0]
        offset = 12
    else:
        raise ValueError(f"unsupported npy version: {path}")
    header = data[offset : offset + header_len].decode("latin1")
    info = ast.literal_eval(header.strip())
    if info.get("fortran_order") is not False:
        raise ValueError("reference does not accept Fortran-order npy")
    descr = info.get("descr")
    if descr not in ("<f4", "|f4", "<f8", "|f8"):
        raise ValueError("reference requires little-endian float32/float64 npy")
    shape = tuple(int(value) for value in info["shape"])
    count = math.prod(shape)
    payload = data[offset + header_len :]
    width = 4 if descr.endswith("f4") else 8
    if len(payload) != count * width:
        raise ValueError("npy payload size mismatch")
    fmt = "<" + ("f" if width == 4 else "d") * count
    return shape, descr, list(struct.unpack(fmt, payload))


def load_features(root: Path, rows: int, cols: int, preprocess: str) -> list[list[float]]:
    if (root / "features.txt").is_file():
        if preprocess != "row_normalize":
            raise ValueError("raw float32 preprocessing requires features.npy")
        raw = [[0.0] * cols for _ in range(rows)]
        seen = [set() for _ in range(rows)]
        with (root / "features.txt").open(encoding="utf-8") as handle:
            first = handle.readline().split()
            if len(first) != 2 or tuple(map(int, first)) != (rows, cols):
                raise ValueError("features.txt shape mismatch")
            for row in range(rows):
                for entry in handle.readline().split():
                    column, text = entry.split(":", 1)
                    column = int(column)
                    if not 0 <= column < cols:
                        raise ValueError("malformed or duplicate sparse feature")
                    if column in seen[row]:
                        raise ValueError("malformed or duplicate sparse feature")
                    seen[row].add(column)
                    raw[row][column] = float(text)
    else:
        shape, descr, values = read_npy(root / "features.npy")
        if shape != (rows, cols):
            raise ValueError("features.npy shape mismatch")
        if preprocess == "ogbn_arxiv_raw_float32" and descr not in ("<f4", "|f4"):
            raise ValueError("OGBN-Arxiv reference requires float32 source features")
        raw = [values[row * cols : (row + 1) * cols] for row in range(rows)]
    if preprocess == "row_normalize":
        for row in raw:
            total = 0.0
            for value in row:
                if not math.isfinite(value):
                    raise ValueError("feature is non-finite")
                total += value
            if not math.isfinite(total):
                raise ValueError("feature row sum is non-finite")
            if total:
                for index, value in enumerate(row):
                    row[index] = value / total
    elif preprocess != "ogbn_arxiv_raw_float32":
        raise ValueError("unknown canonical preprocessing")
    return raw


def load_graph(root: Path, config: dict[str, str], meta: dict[str, str], features: list[list[float]]) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int], int, int]:
    del features
    nodes = int(meta["num_nodes"])
    graphs = int(meta.get("graph_count", "1"))
    if (root / "node_ptr.txt").is_file():
        node_ptr = [int(value) for value in (root / "node_ptr.txt").read_text().split()]
    else:
        node_ptr = [0, nodes]
    if len(node_ptr) != graphs + 1 or node_ptr[0] != 0 or node_ptr[-1] != nodes:
        raise ValueError("node_ptr does not define the canonical graph partition")
    edges: list[tuple[int, int, int, int]] = []
    for stable, raw in enumerate((root / "edges.txt").read_text().splitlines()):
        parts = [int(item) for item in raw.split()]
        if len(parts) == 2:
            gid, src, dst = 0, parts[0], parts[1]
        elif len(parts) == 3:
            gid, src, dst = parts
        else:
            raise ValueError("malformed edge")
        if not 0 <= gid < graphs:
            raise ValueError("edge graph id is outside the canonical graph partition")
        edges.append((gid, src, dst, stable))
    symmetrize = config.get("symmetrize_edges", "false").lower() in ("1", "true", "yes", "on")
    result: list[tuple[int, int, int, int]] = []
    edge_ptr = [0]
    for gid in range(graphs):
        begin, end = node_ptr[gid], node_ptr[gid + 1]
        local: set[tuple[int, int]] = set()
        candidates: list[tuple[int, int, int, int]] = []
        for edge_gid, src, dst, stable in edges:
            if edge_gid != gid:
                continue
            if not (begin <= src < end and begin <= dst < end):
                raise ValueError("edge endpoint escapes its graph block")
            for left, right in ((src, dst), (dst, src)) if symmetrize and src != dst else ((src, dst),):
                if (left, right) in local:
                    raise ValueError("duplicate canonical directed edge")
                local.add((left, right))
                candidates.append((gid, left, right, stable))
        for node in range(begin, end):
            if (node, node) in local:
                raise ValueError("input already contains a canonical self-loop")
            candidates.append((gid, node, node, len(edges) + node))
        candidates.sort(key=lambda item: (item[0], item[2], item[1], item[3]))
        result.extend(candidates)
        edge_ptr.append(len(result))
    src = [item[1] for item in result]
    dst = [item[2] for item in result]
    # The C++ full-graph path keeps global IDs for a single graph and remaps
    # each PPI graph into a contiguous block.  PPI cache IDs are already
    # contiguous, so this is the same representation here.
    segment = [0] * (nodes + 1)
    for value in dst:
        if not 0 <= value < nodes:
            raise ValueError("edge endpoint escapes dataset node range")
        segment[value + 1] += 1
    for index in range(1, len(segment)):
        segment[index] += segment[index - 1]
    return src, dst, node_ptr, edge_ptr, list(range(nodes)), segment, graphs, len(result)


def read_tensors(path: Path) -> dict[str, tuple[tuple[int, ...], list[float]]]:
    # Canonical checkpoint dumps are token streams: a header may be followed
    # by its values on one or several physical lines.  Parse by tokens so the
    # independent reference consumes exactly the same logical payload as the
    # native adapter.
    tensors: dict[str, tuple[tuple[int, ...], list[float]]] = {}
    tokens = path.read_text(encoding="utf-8").split()
    index = 0
    while index < len(tokens):
        if tokens[index] != "TENSOR" or index + 2 >= len(tokens):
            raise ValueError("malformed tensors.txt record")
        name, rank = tokens[index + 1], int(tokens[index + 2])
        index += 3
        if rank < 0 or index + rank >= len(tokens):
            raise ValueError(f"malformed tensor shape: {name}")
        shape = tuple(int(value) for value in tokens[index : index + rank])
        index += rank
        count = int(tokens[index])
        index += 1
        if count < 0 or math.prod(shape) != count or index + count > len(tokens):
            raise ValueError(f"malformed tensor payload: {name}")
        values = [float(value) for value in tokens[index : index + count]]
        index += count
        if name in tensors or any(not math.isfinite(value) for value in values):
            raise ValueError(f"malformed tensor payload: {name}")
        tensors[name] = (shape, values)
    return tensors


def validate_hidden_bias_tensors(
    manifest: dict[str, Any],
    tensors: dict[str, tuple[tuple[int, ...], list[float]]],
) -> None:
    """Require every structurally unused hidden bias to be present and zero."""

    for spec in manifest.get("hidden_head_specs", []):
        for name in (spec["attn_src_bias"], spec["attn_dst_bias"], spec["output_bias"]):
            values = tensors.get(name, ((), []))[1]
            if not values or any(value != 0.0 for value in values):
                raise ValueError("canonical hidden bias tensors must be present and exactly zero")


def qtensor(tensors: dict[str, tuple[tuple[int, ...], list[float]]], name: str, scale: int) -> list[int]:
    shape, values = tensors[name]
    del shape
    return [round_away(value * scale) for value in values]


def matmul(rows: list[list[int]], weights: list[list[int]]) -> list[list[int]]:
    return [[sum(left * weights[index][column] for index, left in enumerate(row))
             for column in range(len(weights[0]))] for row in rows]


def forward_head(rows: list[list[int]], edges: list[tuple[int, int, int, int]], weights: list[list[int]], src_attn: list[int], dst_attn: list[int], hidden: bool, bias: list[int] | None = None) -> list[list[int]]:
    projected = matmul(rows, weights)
    src_score = [sum(value * src_attn[index] for index, value in enumerate(row)) for row in projected]
    dst_score = [sum(value * dst_attn[index] for index, value in enumerate(row)) for row in projected]
    activated = [signed_div(src_score[src] + dst_score[dst], 5) if src_score[src] + dst_score[dst] < 0 else src_score[src] + dst_score[dst] for _, src, dst, _ in edges]
    maximum = [-(1 << 63) for _ in rows]
    for edge, (_, _src, dst, _) in enumerate(edges):
        maximum[dst] = max(maximum[dst], activated[edge])
    exponential = [max(0, round_away(16.0 * math.exp(-(maximum[dst] - activated[index]) / 256.0))) for index, (_, _src, dst, _) in enumerate(edges)]
    denominator = [0] * len(rows)
    numerator = [[0] * len(projected[0]) for _ in rows]
    for index, (_, src, dst, _) in enumerate(edges):
        denominator[dst] += exponential[index]
        for column, value in enumerate(projected[src]):
            numerator[dst][column] += exponential[index] * value
    output: list[list[int]] = []
    for node, row in enumerate(numerator):
        output.append([])
        for value in row:
            recovered = signed_div(16 * value, 64 * denominator[node])
            output[-1].append(round_away(16.0 * (math.exp(recovered / 16.0) - 1.0)) if hidden and recovered < 0 else recovered)
    if bias is not None:
        output = [[value + bias[column] for column, value in enumerate(row)] for row in output]
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = {}
    for line in args.config.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and "=" in line:
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip().strip("[]")
    meta = read_meta(args.dataset_root / "meta.cfg")
    dataset_config_id = config.get("dataset")
    preprocess = config.get("input_preprocess_id")
    if dataset_config_id not in ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv") or preprocess is None:
        raise ValueError("config is missing canonical dataset/preprocess identity")
    dataset = args.dataset
    rows, cols, classes = int(meta["num_nodes"]), int(meta["num_features"]), int(meta["num_classes"])
    floating = load_features(args.dataset_root, rows, cols, preprocess)
    src, dst, node_ptr, edge_ptr, indices, segment_ptr, graph_count, edge_count = load_graph(args.dataset_root, config, meta, floating)
    features = [[round_away(value * 16.0) for value in row] for row in floating]
    manifest = json.loads((args.checkpoint / "manifest.json").read_text(encoding="utf-8"))
    validate_checkpoint_manifest(manifest, args.checkpoint, dataset, preprocess)
    tensors = read_tensors(args.checkpoint / "tensors.txt")
    hidden = manifest["hidden_head_specs"]
    validate_hidden_bias_tensors(manifest, tensors)
    hidden.sort(key=lambda item: (item["layer_index"], item["local_head_index"], item["global_head_index"]))
    current = features
    for layer in range(2):
        outputs = []
        for head in [item for item in hidden if item["layer_index"] == layer]:
            weights = qtensor(tensors, head["seq_kernel"], 4)
            dim = int(manifest["hidden_layers"][layer]["head_dim"])
            width = int(config["d_in_profile"].split(",")[layer].strip())
            matrix = [weights[row * dim : (row + 1) * dim] for row in range(width)]
            edges = [(0, source, target, edge) for edge, (source, target) in enumerate(zip(src, dst))]
            outputs.append(forward_head(current, edges, matrix,
                                        qtensor(tensors, head["attn_src_kernel"], 4),
                                        qtensor(tensors, head["attn_dst_kernel"], 4), True))
        current = [sum((output[node] for output in outputs), []) for node in range(rows)]
    output_specs = sorted(manifest["output_head_specs"], key=lambda item: item["head_index"])
    heads = []
    for head in output_specs:
        weights = qtensor(tensors, head["seq_kernel"], 4)
        width = len(current[0])
        matrix = [weights[row * classes : (row + 1) * classes] for row in range(width)]
        edges = [(0, source, target, edge) for edge, (source, target) in enumerate(zip(src, dst))]
        heads.append(forward_head(current, edges, matrix,
                                  qtensor(tensors, head["attn_src_kernel"], 4),
                                  qtensor(tensors, head["attn_dst_kernel"], 4), False,
                                  qtensor(tensors, head["output_bias"], 16)))
    logits = [signed_div(sum(head[node][column] for head in heads), len(heads))
              for node in range(rows) for column in range(classes)]
    dimensions = b"".join(struct.pack("<Q", value) for value in (rows, rows, edge_count, graph_count, cols, classes))
    quantized_graph_features = struct.pack("<Q", len(floating) * cols) + b"".join(
        signed_i64(round_away(value)) for row in floating for value in row
    )
    graph_hash = formal_bytes("public_graph", [("dimensions", dimensions), ("absolute_ids", u64_vector(indices)),
                                                ("node_ptr", u64_vector(node_ptr)), ("edge_ptr", u64_vector(edge_ptr)),
                                                ("src", u64_vector(src)), ("dst", u64_vector(dst)),
                                                ("features_i64", quantized_graph_features)])
    checkpoint_hash = formal_bytes("checkpoint_bundle", [("manifest", (args.checkpoint / "manifest.json").read_bytes()),
                                                           ("tensors", (args.checkpoint / "tensors.txt").read_bytes()),
                                                           ("tensors_npz", (args.checkpoint / "tensors.npz").read_bytes())])
    file_names = ["meta.cfg", "edges.txt"]
    for optional in ("node_ptr.txt", "features.txt", "features.npy", "labels.txt", "labels_multi.txt", "train_mask.txt", "val_mask.txt", "test_mask.txt"):
        if (args.dataset_root / optional).is_file():
            file_names.append(optional)
    file_digest = file_bundle_digest(args.dataset_root, file_names)
    checkpoint_digest = file_bundle_digest(args.checkpoint, ["manifest.json", "tensors.npz", "tensors.txt"])
    quant_core = {
        "rounding_mode": ROUNDING, "limb_bits": int(config.get("range_bits", "21")),
        "input_preprocess_id": preprocess, "feature_scale": 16,
        "hidden_weight_scale": 4, "hidden_attention_scale": 4,
        "hidden_projection_scale": 64, "delta_scale": 256, "exp_scale": 16,
        "hidden_recovery_num_scale": 16, "hidden_recovery_den_scale": 64,
        "elu_input_scale": 16, "elu_output_scale": 16, "agg_scale": 16,
        "output_weight_scale": 4, "output_attention_scale": 4,
        "output_projection_scale": 64, "output_pre_scale": 16,
        "output_head_scale": 16, "output_bias_scale": 16, "output_scale": 16,
    }
    quant_profile = formal_bytes("formal_quant_profile", [("canonical_json", json.dumps(quant_core, separators=(",", ":")).encode())])
    payload: dict[str, Any] = {
        "kind": "ezkl_independent_public_y_v1", "route_id": FORMAL_ROUTE,
        "dataset": dataset, "run_id": args.dataset,
        "model_arch_id": manifest["model_arch_id"], "model_param_id": manifest["model_param_id"],
        "input_preprocess_id": preprocess, "quant_cfg_id": quant_profile,
        "source_quant_cfg_id": QUANT_ID, "rounding_mode": ROUNDING,
        "node_count": rows, "feature_dim": cols, "class_count": classes,
        "graph_count": graph_count, "edge_count": edge_count,
        "graph_hash": graph_hash, "checkpoint_hash": checkpoint_hash,
        "dataset_files_digest": file_digest, "checkpoint_bundle_digest": checkpoint_digest,
        "reference_source_digest": digest(Path(__file__).read_bytes()), "logits": logits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"status": "public_y_generated", "resource": str(args.output), "graph_hash": graph_hash}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
