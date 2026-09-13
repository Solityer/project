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
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


DATASETS = ("cora", "citeseer", "pubmed", "ppi")
UNSUPPORTED_DATASETS = ("ogbn_arxiv",)
BN254_MODULUS = (
    "21888242871839275222246405745257275088548364400416034343698204186575808495617"
)
U64_MAX = (1 << 64) - 1


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
    path.write_text("\n".join(output) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def load_tensors(path: Path) -> dict[str, tuple[list[int], list[float]]]:
    tokens = path.read_text().split()
    cursor = 0
    result: dict[str, tuple[list[int], list[float]]] = {}
    while cursor < len(tokens):
        if tokens[cursor] != "TENSOR" or cursor + 3 >= len(tokens):
            raise ValueError(f"malformed tensor archive {path}")
        name = tokens[cursor + 1]
        rank = int(tokens[cursor + 2])
        shape_start = cursor + 3
        shape = [int(x) for x in tokens[shape_start : shape_start + rank]]
        count = int(tokens[shape_start + rank])
        value_start = shape_start + rank + 1
        values = [float(x) for x in tokens[value_start : value_start + count]]
        if len(values) != count or math.prod(shape) != count:
            raise ValueError(f"tensor count mismatch: {name}")
        result[name] = (shape, values)
        cursor = value_start + count
    return result


def quantized_model(root: Path, dataset: str) -> dict[str, int]:
    checkpoint = root / "resources/canonical/checkpoints" / f"{dataset}_gat"
    tensors = load_tensors(checkpoint / "tensors.txt")
    maxima: dict[str, int] = {}
    for name, (_shape, values) in tensors.items():
        if name.startswith("hidden/"):
            scale = 4
        elif name.startswith("output/"):
            scale = 16 if name.endswith("/output_bias") else 4
        else:
            continue
        quantized = [round_away(value * scale) for value in values]
        prefix = name.rsplit("/", 1)[0]
        maxima[prefix] = max(maxima.get(prefix, 0), max((abs(v) for v in quantized), default=0))
    return maxima


def graph_info(root: Path, dataset: str, meta: dict[str, str]) -> tuple[int, int, int]:
    nodes = int(meta["num_nodes"])
    graph_count = int(meta.get("graph_count", "1"))
    if dataset == "ppi":
        node_ptr = [int(x) for x in (root / "resources/canonical/cache/ppi/node_ptr.txt").read_text().split()]
    else:
        node_ptr = [0, nodes]
    graph_for_node = [0] * nodes
    for graph in range(graph_count):
        for node in range(node_ptr[graph], node_ptr[graph + 1]):
            graph_for_node[node] = graph
    edges = root / "resources/canonical/cache" / dataset / "edges.txt"
    raw: list[tuple[int, int, int]] = []
    incoming: set[int] = set()
    for line in edges.read_text().splitlines():
        if not line.strip():
            continue
        parts = [int(x) for x in line.split()]
        if dataset == "ppi":
            if len(parts) != 3:
                raise ValueError(f"invalid PPI edge row in {dataset}")
            declared_graph, source, destination = parts
        else:
            if len(parts) != 2:
                raise ValueError(f"invalid edge row in {dataset}")
            declared_graph = None
            source, destination = parts
        graph = graph_for_node[source]
        if declared_graph is not None and declared_graph != graph:
            raise ValueError(f"edge graph id disagrees in {dataset}")
        if graph_for_node[destination] != graph:
            raise ValueError(f"edge crosses graph boundary in {dataset}")
        raw.append((graph, source, destination))
        incoming.add(destination)
    for node in range(nodes):
        if node not in incoming:
            raw.append((graph_for_node[node], node, node))
    raw.sort(key=lambda item: (item[0], item[2]))
    degree = 0
    cursor = 0
    for graph in range(graph_count):
        counts: dict[int, int] = {}
        while cursor < len(raw) and raw[cursor][0] == graph:
            counts[raw[cursor][2]] = counts.get(raw[cursor][2], 0) + 1
            cursor += 1
        degree = max(degree, max(counts.values(), default=0))
    return nodes, len(raw), max(1, degree)


def npy_values(path: Path) -> list[float]:
    data = path.read_bytes()
    if data[:6] != b"\x93NUMPY":
        raise ValueError(f"invalid npy file {path}")
    major = data[6]
    if major == 1:
        header_len = int.from_bytes(data[8:10], "little")
        start = 10
    elif major == 2:
        header_len = int.from_bytes(data[8:12], "little")
        start = 12
    else:
        raise ValueError(f"unsupported npy version {major}")
    header = data[start : start + header_len].decode()
    item_size = 4 if "f4" in header else 8 if "f8" in header else 0
    if not item_size:
        raise ValueError(f"unsupported npy dtype in {path}")
    payload = data[start + header_len :]
    return [
        (
            struct.unpack("<f", payload[i : i + 4])[0]
            if item_size == 4
            else struct.unpack("<d", payload[i : i + 8])[0]
        )
        for i in range(0, len(payload), item_size)
    ]


def feature_bound(root: Path, dataset: str, meta: dict[str, str]) -> int:
    path = root / "resources/canonical/cache" / dataset
    scale = 16
    if dataset in {"ppi", "ogbn_arxiv"}:
        return max((abs(round_away(x * scale)) for x in npy_values(path / "features.npy")), default=1)
    maximum = 0
    lines = (path / "features.txt").read_text().splitlines()
    preprocess = meta.get("input_preprocess_id", "")
    for line in lines[1:]:
        entries = []
        for token in line.split():
            _index, value = token.split(":", 1)
            entries.append(float(value))
        norm = sum(entries) if preprocess == "row_normalize" else 1.0
        if norm < 0 or not math.isfinite(norm):
            raise ValueError(f"invalid feature normalization in {dataset}")
        if norm == 0:
            norm = 1.0
        maximum = max(maximum, max((abs(round_away(value / norm * scale)) for value in entries), default=0))
    return max(1, maximum)


def table_digest(table: dict[str, Any]) -> str:
    out = bytearray(b"ezkl-gat-exact-table-v1")
    append_string(out, table["table_id"])
    out.extend(struct.pack("<i", table["input_scale"]))
    out.extend(struct.pack("<i", table["output_scale"]))
    append_string(out, table["function_id"])
    append_string(out, table["domain_id"])
    append_string(out, table["generator_id"])
    append_i128(out, table["input_min"])
    append_i128(out, table["input_max"])
    for input_value, output_value in table["rows"]:
        append_i128(out, input_value)
        append_i128(out, output_value)
    return sha256(bytes(out))


def static_tables_digest(tables: dict[str, Any]) -> str:
    out = bytearray(b"ezkl-gat-static-tables-v1")
    append_string(out, tables["static_tables_id"])
    append_string(out, tables["backend_aux_tables_id"])
    for key in ("lrelu", "exp", "elu", "max_relu"):
        encoded = bytearray(b"ezkl-gat-exact-table-v1")
        table = tables[key]
        append_string(encoded, table["table_id"])
        encoded.extend(struct.pack("<i", table["input_scale"]))
        encoded.extend(struct.pack("<i", table["output_scale"]))
        append_string(encoded, table["function_id"])
        append_string(encoded, table["domain_id"])
        append_string(encoded, table["generator_id"])
        append_i128(encoded, table["input_min"])
        append_i128(encoded, table["input_max"])
        for input_value, output_value in table["rows"]:
            append_i128(encoded, input_value)
            append_i128(encoded, output_value)
        append_usize(out, len(encoded))
        out.extend(encoded)
        append_string(out, table["digest"])
    return sha256(bytes(out))


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
    rows: list[tuple[int, int]],
) -> dict[str, Any]:
    table = {
        "table_id": table_id,
        "input_scale": input_scale,
        "output_scale": output_scale,
        "function_id": function_id,
        "domain_id": domain_id,
        "generator_id": generator_id,
        "rows": rows,
        "input_min": low,
        "input_max": high,
        "digest": "",
    }
    table["digest"] = table_digest(table)
    return table


def build_tables(
    dataset: str,
    range_bits: int,
    nodes: int,
    features: int,
    edges: int,
    degree: int,
    whole_graph_single: bool,
) -> dict[str, Any]:
    configured = 1 << range_bits
    large = whole_graph_single and (
        edges >= (1 << 18) or nodes * features >= (1 << 24)
    )
    if large:
        negative_rows = 3 << 18
        interval_rows = (1 << 23) - 2
        lrelu_low = -negative_rows
        lrelu_high = interval_rows - negative_rows - 1
        elu_bound = 1 << 20
    else:
        lrelu_low = -configured
        lrelu_high = configured
        elu_bound = configured
    lrelu_rows = [
        (x, x if x >= 0 else round_ratio_away(x, 5))
        for x in range(lrelu_low, lrelu_high + 1)
    ]
    lrelu_output_bound = max(abs(y) for _, y in lrelu_rows)
    exp_bound = 2 * lrelu_output_bound
    exp_rows = [
        (x, max(1, round_away(math.exp(-x / 256.0) * 16.0)))
        for x in range(exp_bound + 1)
    ]
    elu_rows = [
        (x, x if x >= 0 else round_away((math.exp(x / 16.0) - 1.0) * 16.0))
        for x in range(-elu_bound, elu_bound + 1)
    ]
    max_relu_bound = max(2 * lrelu_output_bound, degree)
    max_relu_rows = [(x, max(0, x)) for x in range(-max_relu_bound, max_relu_bound + 1)]
    domain_id = (
        f"dataset:{dataset}:range_bits:{range_bits}:large:{str(large).lower()}"
    )
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
            lrelu_rows,
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
            exp_rows,
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
            elu_rows,
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
            max_relu_rows,
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
    meta: dict[str, str],
    maxima: dict[str, int],
    feature_max: int,
    degree: int,
) -> dict[str, Any]:
    lrelu_input = max(abs(tables["lrelu"]["input_min"]), abs(tables["lrelu"]["input_max"]))
    lrelu_output = max(abs(y) for _x, y in tables["lrelu"]["rows"])
    exp_input = tables["exp"]["input_max"]
    exp_output = max(y for _x, y in tables["exp"]["rows"])
    elu_input = max(abs(tables["elu"]["input_min"]), abs(tables["elu"]["input_max"]))
    elu_output = max(abs(y) for _x, y in tables["elu"]["rows"])
    max_relu_input = max(
        abs(tables["max_relu"]["input_min"]), abs(tables["max_relu"]["input_max"])
    )
    table_output = max(lrelu_output, elu_output, max_relu_input)
    families: list[dict[str, Any]] = []
    current_bound = max(1, feature_max)
    for layer in range(2):
        input_dim = int(meta["num_features"]) if layer == 0 else 64
        for head in range(8):
            weight = max(
                1, maxima.get(f"hidden/layer{layer}/head{head}/seq", 1)
            )
            attention = max(
                1,
                maxima.get(f"hidden/layer{layer}/head{head}/attn_src", 1),
                maxima.get(f"hidden/layer{layer}/head{head}/attn_dst", 1),
            )
            projected = max(1, input_dim * current_bound * weight)
            score = max(lrelu_input, 2 * projected * attention)
            summation = max(1, degree * exp_output)
            numerator = max(1, summation * projected)
            dividend = max(elu_input, numerator * 16)
            divisor = max(1, summation * 64)
            quotient = max(1, (dividend + divisor // 2) // divisor)
            aux = max(
                max_relu_input,
                2 * dividend + divisor,
                2 * divisor * (quotient + 1),
            )
            families.append(
                {
                    "layer": layer,
                    "head": head,
                    "is_output": False,
                    "max_score_abs": score,
                    "max_weight_abs": weight,
                    "max_attention_abs": attention,
                    "max_bias_abs": 0,
                    "max_table_output_abs": table_output,
                    "max_delta": max(exp_input, 2 * lrelu_output),
                    "max_exp": exp_output,
                    "max_sum": summation,
                    "max_aggregation_num": numerator,
                    "max_dividend_abs": dividend,
                    "max_aux_abs": aux,
                    "max_divisor": divisor,
                }
            )
        current_bound = max(1, elu_output)
    classes = int(meta["num_classes"])
    for head in range(2):
        weight = max(1, maxima.get(f"output/head{head}/seq", 1))
        attention = max(
            1,
            maxima.get(f"output/head{head}/attn_src", 1),
            maxima.get(f"output/head{head}/attn_dst", 1),
        )
        bias = max(0, maxima.get(f"output/head{head}/output_bias", 0))
        projected = max(1, 64 * current_bound * weight)
        score = max(lrelu_input, 2 * projected * attention)
        summation = max(1, degree * exp_output)
        numerator = max(1, summation * projected)
        dividend = max(1, numerator * 16, elu_input)
        divisor = max(1, summation * 64, 16)
        quotient = max(1, (dividend + divisor // 2) // divisor)
        aux = max(
            max_relu_input,
            2 * dividend + divisor,
            2 * divisor * (quotient + 1),
            classes,
        )
        families.append(
            {
                "layer": U64_MAX,
                "head": head,
                "is_output": True,
                "max_score_abs": score,
                "max_weight_abs": weight,
                "max_attention_abs": attention,
                "max_bias_abs": bias,
                "max_table_output_abs": table_output,
                "max_delta": max(exp_input, 2 * lrelu_output),
                "max_exp": exp_output,
                "max_sum": summation,
                "max_aggregation_num": numerator,
                "max_dividend_abs": dividend,
                "max_aux_abs": aux,
                "max_divisor": divisor,
            }
        )
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
    ledger = {key: max([int(family[key]) for family in families] + [1]) for key in fields}
    ledger.update(
        {
            "max_score_abs": max(ledger["max_score_abs"], lrelu_input),
            "max_table_output_abs": max(ledger["max_table_output_abs"], table_output),
            "max_delta": max(ledger["max_delta"], exp_input),
            "max_exp": max(ledger["max_exp"], exp_output),
            "max_dividend_abs": max(ledger["max_dividend_abs"], elu_input),
            "max_aux_abs": max(ledger["max_aux_abs"], max_relu_input),
        }
    )
    ledger["field_modulus"] = BN254_MODULUS
    ledger["ledger_id"] = f"ledger:gat:{dataset}:v1"
    ledger["ledger_digest"] = ""
    ledger["families"] = families
    ledger["ledger_digest"] = ledger_digest(ledger)
    return ledger


def generate_dataset(root: Path, dataset: str) -> tuple[str, str]:
    if dataset in UNSUPPORTED_DATASETS:
        raise SystemExit(
            f"generate_gat_resources: unsupported dataset {dataset}; "
            "large OGBN-Arxiv exact tables are not generated in this baseline-local package"
        )
    if dataset not in DATASETS:
        raise SystemExit(f"generate_gat_resources: unsupported dataset {dataset}")
    config = parse_cfg(root / "resources/canonical/configs" / f"{dataset}_secure.cfg")
    meta = parse_cfg(root / "resources/canonical/cache" / dataset / "meta.cfg")
    nodes, edges, degree = graph_info(root, dataset, meta)
    maxima = quantized_model(root, dataset)
    feature_max = feature_bound(root, dataset, meta)
    tables = build_tables(
        dataset,
        int(config["range_bits"]),
        nodes,
        int(meta["num_features"]),
        edges,
        degree,
        config.get("batching_rule") == "whole_graph_single",
    )
    ledger = build_ledger(dataset, tables, meta, maxima, feature_max, degree)
    table_path = root / "resources/canonical/tables" / dataset / "gat_static_tables.json"
    ledger_path = root / "resources/canonical/ledgers" / dataset / "gat_integer_ledger.json"
    write_json(table_path, tables)
    write_json(ledger_path, ledger)
    write_cfg(
        root / "resources/canonical/configs" / f"{dataset}_secure.cfg",
        {
            "static_table_id": tables["static_tables_id"],
            "static_table_digest": tables["static_tables_digest"],
            "degree_bound_id": ledger["ledger_id"],
            "degree_bound_digest": ledger["ledger_digest"],
        },
    )
    manifest_path = root / "resources/canonical/checkpoints" / f"{dataset}_gat/manifest.json"
    checkpoint = json.loads(manifest_path.read_text())
    checkpoint["static_table_id"] = tables["static_tables_id"]
    checkpoint["static_table_digest"] = tables["static_tables_digest"]
    checkpoint["degree_bound_id"] = ledger["ledger_id"]
    checkpoint["degree_bound_digest"] = ledger["ledger_digest"]
    write_json(manifest_path, checkpoint)
    return tables["static_tables_digest"], ledger["ledger_digest"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS + UNSUPPORTED_DATASETS,
        action="append",
    )
    args = parser.parse_args()
    datasets = args.dataset or list(DATASETS)
    for dataset in datasets:
        table_digest_value, ledger_digest_value = generate_dataset(args.root, dataset)
        print(
            f"{dataset}: static_tables={table_digest_value} ledger={ledger_digest_value}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
