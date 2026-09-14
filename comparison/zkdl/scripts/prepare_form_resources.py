#!/usr/bin/env python3
"""Create baseline-local zkDL formal resources from an explicit ceremony input.

This tool is intentionally fail-closed.  A seed is never inferred from the
repository and is never described as a public trusted ceremony: the supplied
transcript must bind its SHA-256.  The default mode materializes the small Cora
resource set only; the other datasets receive deterministic plan manifests
until ``--full-dataset`` is explicitly requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
R = int("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001", 16)
P = int("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab", 16)
GX = int("17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb", 16)
GY = int("08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1", 16)
G = (GX, GY)

# These bounds mirror the native proof/resource wire contracts.  The former
# preparation path used a 2^19 shortcut, which truncated valid relation
# families for larger datasets.  Keep only representation and frame-size
# guards here; the requested domain itself is still derived from the
# authenticated relation shapes below.
MAX_NATIVE_DOMAIN = 1 << 30
SRS_FRAME_HEADER_BYTES = 8 + 4 + 4 + 8 + 8 + 8
G1_POINT_BYTES = 48
MAX_SRS_FRAME_BYTES = 1 << 32


def die(message: str) -> None:
    raise SystemExit(f"prepare-form-resources: BLOCKED: {message}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compact(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.write_bytes(data)
    return sha256_bytes(data)


def parse_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            die(f"malformed config line {number}: {path}")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value or key in values:
            die(f"duplicate/empty config key at {path}:{number}")
        values[key] = value
    return values


def update_config(path: Path, updates: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    output: list[str] = []
    for raw in lines:
        stripped = raw.split("#", 1)[0].strip()
        if "=" not in stripped:
            output.append(raw)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            output.append(f"{key} = {updates[key]}")
            seen.add(key)
        else:
            output.append(raw)
    missing = set(updates) - seen
    if missing:
        output.extend(f"{key} = {updates[key]}" for key in sorted(missing))
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def config_identity(path: Path) -> str:
    values = parse_config(path)
    values.pop("verifier_registry_digest", None)
    values.pop("model_blinder_sidecar_digest", None)
    return sha256_bytes(compact(values))


def next_pow2(value: int) -> int:
    if value < 1:
        return 1
    result = 1 << (value - 1).bit_length()
    # Native tensor descriptors currently admit at most 2^30 coordinates.
    # This is a representation/overflow boundary, not a dataset-size cap.
    if result > MAX_NATIVE_DOMAIN:
        die(f"relation domain {result} exceeds the native 2^30 coordinate safety boundary")
    return result


def round_away(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def hash_string(h: "hashlib._Hash", value: str) -> None:
    encoded = value.encode()
    h.update(struct.pack("<Q", len(encoded)))
    h.update(encoded)


def hash_u64(h: "hashlib._Hash", value: int) -> None:
    h.update(struct.pack("<Q", value & ((1 << 64) - 1)))


def table_digest(table: dict[str, Any]) -> str:
    h = hashlib.sha256()
    hash_string(h, "zkDL_gat_static_tables_v1")
    for key in ("generator_id", "version", "scale_id", "domain_id", "row_order", "spec_hash"):
        hash_string(h, table[key])
    hash_string(h, table["padding_rule"])
    for key in ("lrelu_padded_domain", "elu_padded_domain", "exp_padded_domain"):
        hash_u64(h, int(table[key]))
    for key in ("lrelu_min", "lrelu_max", "elu_min", "elu_max", "exp_max"):
        hash_u64(h, int(table[key]))
    for key in ("lrelu", "elu", "exp"):
        values = table[key]
        hash_u64(h, len(values))
        for value in values:
            hash_u64(h, int(value))
    return h.hexdigest()


def ledger_digest(ledger: dict[str, Any]) -> str:
    h = hashlib.sha256()
    hash_string(h, "zkDL_gat_integer_bound_ledger_v1")
    for key in ("ledger_id", "quant_cfg_id", "static_table_id", "field_modulus", "scale_chain_digest", "static_table_descriptor_digest"):
        hash_string(h, ledger[key])
    for key in ("max_score_abs", "max_table_output_abs", "max_delta", "max_exp", "max_sum", "max_aggregate_num", "max_dividend_abs", "max_aux_abs", "max_divisor"):
        hash_u64(h, int(ledger[key]))
    families = ledger["families"]
    hash_u64(h, len(families))
    fields = ("max_input_abs", "max_weight_abs", "max_attention_abs", "max_bias_abs", "max_score_abs", "max_table_output_abs", "max_delta", "max_exp", "max_sum", "max_aggregate_num", "max_dividend_abs", "max_aux_abs", "max_divisor", "max_field_product_abs")
    for family in families:
        hash_string(h, family["family_id"])
        hash_u64(h, int(family["layer_index"]))
        hash_u64(h, int(family["head_index"]))
        hash_u64(h, int(family["output"]))
        for key in fields:
            hash_u64(h, int(family[key]))
    return h.hexdigest()


def point_add(left: tuple[int, int] | None, right: tuple[int, int] | None) -> tuple[int, int] | None:
    if left is None:
        return right
    if right is None:
        return left
    x1, y1 = left
    x2, y2 = right
    if x1 == x2:
        if (y1 + y2) % P == 0:
            return None
        slope = (3 * x1 * x1) * pow(2 * y1, P - 2, P) % P
    else:
        slope = (y2 - y1) * pow((x2 - x1) % P, P - 2, P) % P
    x3 = (slope * slope - x1 - x2) % P
    return x3, (slope * (x1 - x3) - y1) % P


def point_mul(point: tuple[int, int] | None, scalar: int) -> tuple[int, int] | None:
    result: tuple[int, int] | None = None
    base = point
    scalar %= R
    while scalar:
        if scalar & 1:
            result = point_add(result, base)
        base = point_add(base, base)
        scalar >>= 1
    return result


def encode_g1(point: tuple[int, int] | None) -> bytes:
    if point is None:
        return bytes([0xC0]) + bytes(47)
    x, y = point
    out = bytearray(x.to_bytes(48, "big"))
    out[0] |= 0x80
    if y > P - y:
        out[0] |= 0x20
    return bytes(out)


def ceremony(seed_file: Path, transcript_file: Path) -> tuple[bytes, str, str]:
    seed = seed_file.read_bytes()
    if not seed:
        die("ceremony seed file is empty")
    seed_digest = sha256_bytes(seed)
    transcript = transcript_file.read_text(encoding="utf-8")
    match = re.search(r"(?m)^seed_sha256=([0-9a-f]{64})$", transcript)
    if match is None or match.group(1) != seed_digest:
        die("ceremony transcript must bind seed_sha256 to the supplied seed")
    return seed, seed_digest, sha256_bytes(transcript.encode())


def derive_scalar(seed: bytes, label: bytes) -> int:
    value = int.from_bytes(hashlib.sha256(seed + b"|zkDL-explicit-ceremony-v1|" + label).digest(), "big") % R
    return value or 1


def _config_bool(values: dict[str, str] | None, key: str, default: bool) -> bool:
    if values is None or key not in values:
        return default
    raw = values[key].strip().lower()
    if raw not in {"true", "false"}:
        die(f"{key} must be true or false")
    return raw == "true"


def graph_degree(root: Path, dataset: str, values: dict[str, str] | None = None) -> tuple[int, int, int]:
    """Reproduce the native load_graph edge canonicalization exactly.

    The old preparation path trusted ``num_edges``/``max_in_degree`` from
    metadata and counted raw destination columns.  Native ``load_graph``
    first applies configured symmetrization and deduplication, then appends a
    self-loop for every node and computes its incoming-edge layout.  SRS
    relation domains must use that same post-canonicalization graph.
    """
    values = values or {}
    cache_raw = values.get("cache_root", "data/cache")
    cache_path = Path(cache_raw)
    if cache_path.is_absolute() or ".." in cache_path.parts:
        die(f"cache_root must be baseline-local: {cache_raw}")
    directory = root / cache_path / dataset
    meta: dict[str, str] = {}
    meta_path = directory / "meta.cfg"
    if not meta_path.is_file():
        die(f"native graph metadata is missing: {meta_path}")
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = (item.strip() for item in line.split("=", 1))
            meta[key] = value
    if meta.get("name") != dataset:
        die(f"cache dataset name mismatch: {dataset}")
    try:
        nodes = int(meta["num_nodes"])
    except (KeyError, ValueError) as exc:
        die(f"cache num_nodes is invalid for {dataset}: {exc}")
    if nodes < 1:
        die(f"cache num_nodes must be positive for {dataset}")
    graph_count = int(meta.get("graph_count", "1"))
    if graph_count < 1:
        die(f"cache graph_count must be positive for {dataset}")
    if dataset != "ppi" and graph_count != 1:
        die("non-PPI canonical graph must contain exactly one graph")
    node_ptr: list[int] = [0, nodes]
    if dataset == "ppi":
        ptr_path = directory / "node_ptr.txt"
        if not ptr_path.is_file():
            die(f"missing PPI node_ptr: {ptr_path}")
        try:
            node_ptr = [int(item) for item in ptr_path.read_text(encoding="utf-8").split()]
        except ValueError as exc:
            die(f"malformed PPI node_ptr: {exc}")
        if (len(node_ptr) != graph_count + 1 or not node_ptr or node_ptr[0] != 0 or
                node_ptr[-1] != nodes):
            die("PPI node_ptr does not match native graph metadata")
        if any(left >= right or right > nodes for left, right in zip(node_ptr, node_ptr[1:])):
            die("PPI node_ptr must be strictly increasing and bounded")

    edges_path = directory / "edges.txt"
    if not edges_path.is_file():
        die(f"native edge file is missing: {edges_path}")
    symmetrize = _config_bool(values, "symmetrize_edges", False)
    deduplicate = _config_bool(values, "deduplicate_edges", True)
    seen: set[tuple[int, int, int]] = set()
    incoming = [0] * nodes
    edge_count = 0

    def graph_for_node(node: int) -> int:
        if dataset != "ppi":
            return 0
        # node_ptr is short (one entry per PPI graph), matching native's
        # upper_bound lookup rather than scanning the full edge list.
        import bisect
        return bisect.bisect_right(node_ptr, node) - 1

    def add(gid: int, src: int, dst: int) -> None:
        nonlocal edge_count
        if src < 0 or src >= nodes or dst < 0 or dst >= nodes:
            die("edge endpoint is outside the native graph bounds")
        key = (gid, src, dst)
        if deduplicate and key in seen:
            return
        if deduplicate:
            seen.add(key)
        incoming[dst] += 1
        edge_count += 1

    for number, raw in enumerate(edges_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        expected = 3 if dataset == "ppi" else 2
        if len(parts) != expected:
            die(f"malformed edge row {number}: expected {expected} columns")
        try:
            parsed = [int(item) for item in parts]
        except ValueError as exc:
            die(f"malformed edge row {number}: {exc}")
        gid = parsed[0] if dataset == "ppi" else 0
        if dataset == "ppi":
            if gid < 0 or gid >= graph_count:
                die(f"PPI edge graph id out of range at row {number}")
            graph_nodes = node_ptr[gid + 1] - node_ptr[gid]
            if parsed[1] < 0 or parsed[1] >= graph_nodes or parsed[2] < 0 or parsed[2] >= graph_nodes:
                die(f"PPI local edge endpoint out of range at row {number}")
            src = node_ptr[gid] + parsed[1]
            dst = node_ptr[gid] + parsed[2]
        else:
            src, dst = parsed
        add(gid, src, dst)
        if symmetrize:
            add(gid, dst, src)
    for node in range(nodes):
        add(graph_for_node(node), node, node)
    return nodes, edge_count, max(1, max(incoming, default=0))


def graph_metadata(root: Path, dataset: str, config_values: dict[str, str] | None = None) -> dict[str, int]:
    metadata: dict[str, int] = {}
    cache_raw = (config_values or {}).get("cache_root", "data/cache")
    cache_path = Path(cache_raw)
    if cache_path.is_absolute() or ".." in cache_path.parts:
        die(f"cache_root must be baseline-local: {cache_raw}")
    path = root / cache_path / dataset / "meta.cfg"
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            if "=" not in raw:
                continue
            key, value = (part.strip() for part in raw.split("=", 1))
            if value.isdigit():
                metadata[key] = int(value)
    return metadata


def parse_profile(value: str) -> list[tuple[int, int]]:
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        die(f"invalid hidden_profile: {value}")
    result: list[tuple[int, int]] = []
    for item in text[1:-1].split(","):
        item = item.strip()
        if "x" not in item:
            die(f"invalid hidden_profile item: {item}")
        heads, width = (part.strip() for part in item.split("x", 1))
        result.append((int(heads), int(width)))
    if not result or any(heads < 1 or width < 1 for heads, width in result):
        die("hidden_profile has no positive layer/head dimensions")
    return result


def parse_list(value: str) -> list[int]:
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        die(f"invalid list value: {value}")
    result = [int(item.strip()) for item in text[1:-1].split(",") if item.strip()]
    if not result or any(item < 1 for item in result):
        die(f"list has no positive entries: {value}")
    return result


def relation_family_domains(values: dict[str, str], nodes: int, edges: int,
                            table: dict[str, Any], model_domains: list[int],
                            graph_count: int | None = None) -> dict[str, int]:
    """Mirror the authenticated RelationPlan's padded tensor families.

    The native runner derives every family from these same public dimensions.
    Keeping this calculation in the resource tool makes SRS preparation
    independent of a prover-side witness and prevents a guessed global cap
    from truncating rounddiv_aux or edge-feature families.
    """
    profile = parse_profile(values["hidden_profile"])
    input_dims = parse_list(values["d_in_profile"])
    classes = int(values["num_classes"])
    output_heads = int(values["K_out"])
    range_bits = int(values["range_bits"])
    graph_count = graph_count if graph_count is not None else max(1, int(values.get("batch_graphs", "1")))
    if graph_count < 1:
        die("native graph_count must be positive")
    feature_dim = int(input_dims[0])
    domains: dict[str, int] = {}

    def shape_domain(*shape: int) -> int:
        result = 1
        for dimension in shape:
            if dimension < 1:
                die(f"relation family has a non-positive logical dimension: {shape}")
            padded = next_pow2(dimension)
            if result > MAX_NATIVE_DOMAIN // padded:
                die(f"relation family shape overflows the native coordinate boundary: {shape}")
            result *= padded
        return result

    def add(name: str, *shape: int) -> None:
        domains[name] = shape_domain(*shape)

    add("input_features", nodes, feature_dim)
    add("sparse_route", edges, 2)
    add("group_selectors", edges, 2)
    add("node_ptr", graph_count + 1)
    add("edge_ptr", graph_count + 1)
    add("node_edge_ptr", nodes + 1)
    add("output_head_count", 1)
    table_bound = 1 << range_bits
    add("static_lrelu", 2 * table_bound + 1)
    add("static_elu", 2 * table_bound + 1)
    add("static_exp", table_bound + 1)
    current_input = feature_dim
    for layer, (head_count, head_dim) in enumerate(profile):
        for head in range(head_count):
            prefix = f"hidden/layer{layer}/head{head}"
            add(prefix + "/model/seq_kernel", 1, current_input, head_dim)
            add(prefix + "/model/attn_src", head_dim)
            add(prefix + "/model/attn_dst", head_dim)
            add(prefix + "/trace/input", nodes, current_input)
            add(prefix + "/trace/h_prime", nodes, head_dim)
            for name in ("e_src", "e_dst", "score", "leaky_relu", "delta", "exp_weight",
                         "lrelu_idx", "exp_idx", "max_selector"):
                add(prefix + "/trace/" + name, edges)
            add(prefix + "/trace/lrelu_bits", edges, range_bits)
            add(prefix + "/trace/exp_bits", edges, range_bits)
            for name in ("inverse_field", "alpha_field"):
                add(prefix + "/trace/" + name, edges)
            add(prefix + "/trace/maximum", nodes)
            add(prefix + "/trace/denominator", nodes)
            add(prefix + "/trace/range_bits", edges, range_bits)
            add(prefix + "/trace/rounddiv_aux", nodes, head_dim, 8)
            for name in ("rounddiv_a_inverse", "rounddiv_divisor_inverse"):
                add(prefix + "/trace/" + name, nodes, head_dim)
            for name in ("routed_h_prime", "product_u_h_prime", "product_alpha_h_prime_field",
                         "aggregate_num", "aggregate_pre_field", "consistency_residual_field",
                         "aggregate_pre", "elu_input", "aggregate"):
                add(prefix + "/trace/" + name, edges, head_dim) if name.startswith(("routed", "product")) else add(prefix + "/trace/" + name, nodes, head_dim)
            add(prefix + "/trace/" + "elu_idx", nodes, head_dim)
            add(prefix + "/trace/" + "elu_bits", nodes, head_dim, range_bits)
        add(f"hidden/layer{layer}/concat", nodes, head_count, head_dim)
        current_input = head_count * head_dim
    output_input = current_input
    for head in range(output_heads):
        prefix = f"output/head{head}"
        add(prefix + "/model/seq_kernel", 1, output_input, classes)
        add(prefix + "/model/attn_src", classes)
        add(prefix + "/model/attn_dst", classes)
        add(prefix + "/model/output_bias", classes)
        add(prefix + "/trace/input", nodes, output_input)
        add(prefix + "/trace/y_prime", nodes, classes)
        for name in ("e_src", "e_dst", "score", "leaky_relu", "delta", "exp_weight",
                     "lrelu_idx", "exp_idx", "max_selector"):
            add(prefix + "/trace/" + name, edges)
        add(prefix + "/trace/lrelu_bits", edges, range_bits)
        add(prefix + "/trace/exp_bits", edges, range_bits)
        for name in ("inverse_field", "alpha_field"):
            add(prefix + "/trace/" + name, edges)
        add(prefix + "/trace/maximum", nodes)
        add(prefix + "/trace/denominator", nodes)
        add(prefix + "/trace/range_bits", edges, range_bits)
        add(prefix + "/trace/rounddiv_aux", nodes, classes, 8)
        for name in ("rounddiv_a_inverse", "rounddiv_divisor_inverse"):
            add(prefix + "/trace/" + name, nodes, classes)
        for name in ("routed_y_prime", "product_u_y_prime", "product_alpha_y_prime_field",
                     "aggregate_num", "aggregate_pre_field", "consistency_residual_field",
                     "aggregate_pre", "y_quantized", "aligned", "output"):
            add(prefix + "/trace/" + name, edges, classes) if name.startswith(("routed", "product")) else add(prefix + "/trace/" + name, nodes, classes)
    add("output_heads", nodes, output_heads, classes)
    add("output_sum", nodes, classes)
    add("public_Y", nodes, classes)

    # Operation descriptors add derived TensorFamily entries for every
    # route/broadcast/prefix/lookup/rounding view.  They are real relation
    # families in the native witness (not prover-only labels), so register
    # each exact view domain here as well.  The SRS writer may still bind all
    # of these logical requirements to the authenticated common prefix.
    def view(name: str, *shape: int) -> None:
        add(name, *shape)

    def edge_shape(width: int = 1) -> tuple[int, ...]:
        return (edges,) if width == 1 else (edges, width)

    def node_shape(width: int = 1) -> tuple[int, ...]:
        return (nodes,) if width == 1 else (nodes, width)

    def table_depth(size: int) -> int:
        depth = 0
        width = 1
        while width < size:
            width <<= 1
            depth += 1
        return depth

    def append_round_views(operation: str, width: int) -> None:
        shape = node_shape(width)
        for role in ("signed_numerator", "divisor", "signed_quotient", "a", "q_abs",
                     "r", "delta", "sigma", "z", "a_inverse", "divisor_inverse"):
            view(operation + "/view/" + role, *shape)
        for value in ("a", "q_abs", "r", "delta", "divisor"):
            for bit in range(range_bits):
                view(operation + f"/view/range/{value}/{bit}", *shape)

    def append_table_views(operation: str, edge_domain: bool,
                           feature_width: int, table_size: int) -> None:
        shape = edge_shape(feature_width) if edge_domain else node_shape(feature_width)
        for role in ("query", "index", "output0"):
            view(operation + "/view/" + role, *shape)
        depth = table_depth(table_size)
        for bit in range(depth):
            view(operation + f"/view/bit/{bit}", *shape)
        for level in range(depth + 1):
            for index in range(1 << level):
                view(operation + f"/view/chi/{level}/{index}", *shape)

    def append_stage_views(stage: str, width: int, output: bool) -> None:
        edge = edge_shape()
        edge_feature = edge_shape(width)
        node_feature = node_shape(width)
        inner = stage + ("/output_attention_inner_product" if output else "/attention_inner_product")
        for role in ("src_matrix", "dst_matrix"):
            view(inner + "/view/" + role, *edge_feature)
        for role in ("src_output", "dst_output"):
            view(inner + "/view/" + role, *edge)

        maximum = stage + "/max_range"
        for role in ("value", "maximum_broadcast", "selector", "delta", "group_new",
                     "group_end", "selector_prefix", "selector_prev"):
            view(maximum + "/view/" + role, *edge)
        for bit in range(range_bits):
            view(maximum + f"/view/range_bit/{bit}", *edge)

        group = stage + "/group_sum"
        for role in ("group_new", "group_end", "input0", "input1", "output0", "output1",
                     "prefix0", "prefix_prev0", "prefix1", "prefix_prev1"):
            view(group + "/view/" + role, *edge_feature)

        inverse = stage + "/inverse_alpha"
        for role in ("input0", "input1", "output0", "output1"):
            view(inverse + "/view/" + role, *edge)

        for operation in (stage + "/field_aggregate", stage + "/integer_numerator"):
            for role in ("group_new", "group_end", "input0", "output0", "prefix0", "prefix_prev0"):
                view(operation + "/view/" + role, *edge_feature)

        consistency = stage + "/consistency"
        for role in ("input0", "input1", "input2", "output0"):
            view(consistency + "/view/" + role, *node_feature)

        append_round_views(stage + "/signed_round_div", width)
        append_table_views(stage + "/leaky_relu", True, 1, 2 * table_bound + 1)
        append_table_views(stage + "/exp_map", True, 1, table_bound + 1)
        if not output:
            append_table_views(stage + "/elu", False, width, 2 * table_bound + 1)
        else:
            append_round_views(stage + "/output_scale_alignment", width)
            bias = stage + "/output_bias"
            for role in ("input0", "input1", "output0"):
                view(bias + "/view/" + role, *node_feature)

    for layer, (head_count, head_dim) in enumerate(profile):
        for head in range(head_count):
            append_stage_views(f"hidden/layer{layer}/head{head}", head_dim, False)
        concat = f"hidden/layer{layer}/concat"
        concat_shape = (nodes, head_count, head_dim)
        for head in range(head_count):
            view(concat + f"/view/input{head}", *concat_shape)
        view(concat + "/view/output", *concat_shape)
    for head in range(output_heads):
        append_stage_views(f"output/head{head}", classes, True)
    append_round_views("final_average", classes)

    table_domains = {
        "static_lrelu": int(table["lrelu_padded_domain"]),
        "static_elu": int(table["elu_padded_domain"]),
        "static_exp": int(table["exp_padded_domain"]),
    }
    domains.update(table_domains)
    for domain in model_domains:
        if domain > MAX_NATIVE_DOMAIN:
            die(f"model family domain exceeds native coordinate boundary: {domain}")
    domains["*"] = max(max(domains.values(), default=1), max(model_domains, default=1))
    return domains


def load_tensors(path: Path) -> list[tuple[str, list[int], list[float]]]:
    tokens = path.read_text(encoding="utf-8").split()
    index = 0
    result: list[tuple[str, list[int], list[float]]] = []
    while index < len(tokens):
        if tokens[index] != "TENSOR" or index + 3 >= len(tokens):
            die(f"malformed tensor export: {path}")
        name = tokens[index + 1]
        rank = int(tokens[index + 2])
        begin = index + 3
        shape = [int(value) for value in tokens[begin:begin + rank]]
        count = int(tokens[begin + rank])
        value_begin = begin + rank + 1
        values = [float(value) for value in tokens[value_begin:value_begin + count]]
        if len(values) != count or math.prod(shape) != count:
            die(f"tensor count mismatch: {name}")
        result.append((name, shape, values))
        index = value_begin + count
    return result


def model_entries(root: Path, values: dict[str, str], tau: int, h_point: tuple[int, int], blinders: dict[str, int]) -> list[dict[str, Any]]:
    tensors_path = root / values["checkpoint_bundle"] / "tensors.txt"
    entries: list[dict[str, Any]] = []
    power = 1
    for name, shape, floats in load_tensors(tensors_path):
        if "/hidden/" in "/" + name:
            prefix = name.split("/seq/kernel")[0].split("/attn_")[0]
            mapped = name.replace("/seq/kernel", "/model/seq_kernel").replace("/attn_src/kernel", "/model/attn_src").replace("/attn_dst/kernel", "/model/attn_dst")
            scale = 4
        elif name.startswith("output/"):
            mapped = name.replace("/seq/kernel", "/model/seq_kernel").replace("/attn_src/kernel", "/model/attn_src").replace("/attn_dst/kernel", "/model/attn_dst").replace("/output_bias", "/model/output_bias")
            scale = 16 if name.endswith("output_bias") else 4
        else:
            continue
        logical = shape[1:] if len(shape) == 3 and shape[0] == 1 else shape
        padded = [next_pow2(int(axis)) for axis in logical]
        domain = math.prod(padded)
        scalar = 0
        tau_power = 1
        for value in floats:
            quantized = round_away(value * scale)
            scalar = (scalar + quantized * tau_power) % R
            tau_power = tau_power * tau % R
        blind = blinders[mapped]
        commitment = point_add(point_mul(G, scalar), point_mul(h_point, blind))
        if commitment is None:
            commitment = G
        entries.append({"tensor_id": mapped, "domain_size": domain, "logical_shape": logical, "padded_shape": padded, "commitment": encode_g1(commitment).hex()})
        power = max(power, domain)
    if not entries:
        die(f"checkpoint has no canonical GAT model tensors: {tensors_path}")
    return entries


def make_tables(values: dict[str, str], quant_digest: str) -> dict[str, Any]:
    bound = 1 << int(values["range_bits"])
    scale_id = "scale_chain:s_delta=256,s_agg_pre=16,s_agg=16,s_exp=16"
    domain_id = f"range_bits:{values['range_bits']}"
    generator_id = "zkDL_baseline_local_static_table_generator_v1"
    version = "v1"
    row_order = "ascending_index_min_to_max"
    padding_rule = "next_power_of_two_zero_v1"
    lrelu_rows = 2 * bound + 1
    elu_rows = 2 * bound + 1
    exp_rows = bound + 1
    lrelu_padded_domain = next_pow2(lrelu_rows)
    elu_padded_domain = next_pow2(elu_rows)
    exp_padded_domain = next_pow2(exp_rows)
    spec = "\n".join((values["static_table_id"], generator_id, version, scale_id, domain_id, row_order,
                       padding_rule, values["range_bits"], str(-bound), str(bound), str(bound),
                       str(lrelu_padded_domain), str(elu_padded_domain), str(exp_padded_domain))) + "\n"
    table: dict[str, Any] = {"schema": "zkDL_gat_static_table_descriptor_v1", "quant_cfg_id": values["quant_cfg_id"], "generator_id": generator_id, "version": version, "scale_id": scale_id, "domain_id": domain_id, "row_order": row_order, "padding_rule": padding_rule, "lrelu_padded_domain": lrelu_padded_domain, "elu_padded_domain": elu_padded_domain, "exp_padded_domain": exp_padded_domain, "spec_hash": sha256_bytes(spec.encode()), "static_table_id": values["static_table_id"], "lrelu_min": -bound, "lrelu_max": bound, "elu_min": -bound, "elu_max": bound, "exp_max": bound}
    table["lrelu"] = [x if x >= 0 else round_away(x * 0.2) for x in range(-bound, bound + 1)]
    table["elu"] = [x if x >= 0 else round_away((math.exp(x / 16.0) - 1.0) * 16.0) for x in range(-bound, bound + 1)]
    table["exp"] = [max(1, round_away(math.exp(-x / 256.0) * 16.0)) for x in range(bound + 1)]
    table["digest"] = table_digest(table)
    return table


def make_ledger(root: Path, values: dict[str, str], tables: dict[str, Any], table_file_digest: str, quant_digest: str) -> dict[str, Any]:
    _, _, degree = graph_degree(root, values["dataset"], values)
    tensors = load_tensors(root / values["checkpoint_bundle"] / "tensors.txt")
    model_max = max((abs(round_away(value * (16 if name.endswith("output_bias") else 4))) for name, _, data in tensors for value in data), default=1)
    bound = max(1, model_max)
    ledger_id = f"zkdl_integer_ledger_v2_{values['dataset']}_degree_{degree}"
    degree_id = f"zkdl_degree_v2_{values['dataset']}_{degree}"
    families: list[dict[str, Any]] = []
    def family(fid: str, layer: int, head: int, output: int, input_bound: int, weight: int, attention: int, bias: int) -> None:
        score = max(1, input_bound * max(1, weight) * max(1, attention) * 8)
        table_out = max(1, score)
        delta = 2 * table_out
        exp = 16
        summation = max(1, degree * exp)
        aggregate = max(1, summation * max(1, input_bound * weight))
        dividend = max(1, aggregate * 16)
        divisor = max(1, summation * 64)
        aux = max(1, 2 * dividend + divisor, 2 * divisor * max(1, (dividend + divisor - 1) // divisor) + 2 * divisor)
        families.append({"family_id": fid, "layer_index": layer, "head_index": head, "output": output, "max_input_abs": input_bound, "max_weight_abs": max(1, weight), "max_attention_abs": max(1, attention), "max_bias_abs": max(0, bias), "max_score_abs": score, "max_table_output_abs": table_out, "max_delta": delta, "max_exp": exp, "max_sum": summation, "max_aggregate_num": aggregate, "max_dividend_abs": dividend, "max_aux_abs": aux, "max_divisor": divisor, "max_field_product_abs": max(1, exp * input_bound * max(1, weight))})
    current = max(1, 1 << min(int(values["range_bits"]), 20))
    for layer in range(2):
        for head in range(8):
            family(f"hidden/layer{layer}/head{head}", layer, head, 0, current, bound, bound, 0)
    for head in range(2):
        family(f"output/head{head}", 2, head, 1, current, bound, bound, bound)
    fields = ("max_score_abs", "max_table_output_abs", "max_delta", "max_exp", "max_sum", "max_aggregate_num", "max_dividend_abs", "max_aux_abs", "max_divisor")
    global_values = {key: max(int(item[key]) for item in families) for key in fields}
    ledger = {"schema": "zkDL_gat_integer_bound_ledger_v1", "ledger_id": ledger_id, "degree_bound_id": degree_id, "quant_cfg_id": values["quant_cfg_id"], "static_table_id": values["static_table_id"], "field_modulus": str(R), "scale_chain_digest": quant_digest, "static_table_descriptor_digest": table_file_digest, **global_values, "families": families}
    ledger["digest"] = ledger_digest(ledger)
    return ledger


def write_srs(root: Path, values: dict[str, str], seed: bytes, seed_digest: str, transcript_digest: str, family_domains: dict[str, int]) -> tuple[str, str, tuple[int, int], tuple[int, int]]:
    srs_dir = root / values["srs_path"]
    srs_dir.mkdir(parents=True, exist_ok=True)
    if not family_domains or "*" not in family_domains:
        die("authenticated SRS family-domain map must contain the common '*' prefix")
    for family_id, chunk_domain in family_domains.items():
        if chunk_domain < 1 or chunk_domain > MAX_NATIVE_DOMAIN or chunk_domain & (chunk_domain - 1):
            die(f"SRS family {family_id} has an invalid native power-of-two domain: {chunk_domain}")
        if chunk_domain > MAX_SRS_FRAME_BYTES // G1_POINT_BYTES:
            die(f"SRS family {family_id} frame exceeds the native authenticated frame byte bound")
    # Check aggregate output size before deriving points.  This is an
    # operational disk-capacity guard, not a proof-domain restriction.
    common_domain = max(family_domains.values())
    # All family chunks authenticate one common prefix basis.  The manifest
    # retains per-family required domains, but the large G frame and the H/U
    # frames are written once and referenced by every family.  Include a
    # conservative metadata allowance in the disk admission check.
    estimated_bytes = (
        (SRS_FRAME_HEADER_BYTES + G1_POINT_BYTES * common_domain) +
        2 * (SRS_FRAME_HEADER_BYTES + G1_POINT_BYTES) +
        len(family_domains) * 4096
    )
    try:
        statvfs = os.statvfs(srs_dir)
        free_bytes = statvfs.f_bavail * statvfs.f_frsize
    except OSError as exc:
        die(f"cannot inspect SRS filesystem capacity: {exc}")
    if estimated_bytes > free_bytes:
        die(f"SRS materialization needs {estimated_bytes} bytes but only {free_bytes} are available")
    tau = derive_scalar(seed, b"tau:" + values["dataset"].encode())
    h_scalar = derive_scalar(seed, b"H:" + values["dataset"].encode())
    u_scalar = derive_scalar(seed, b"U:" + values["dataset"].encode())
    domain = common_domain
    h_point = point_mul(G, h_scalar)
    u_point = point_mul(G, u_scalar)
    def write_frame(label: str, payloads: Iterable[bytes], count: int) -> dict[str, Any]:
        """Stream one authenticated common frame without retaining points."""
        relative = Path("wildcard") / f"{label}.chunk0.bin"
        if count < 1 or SRS_FRAME_HEADER_BYTES + count * G1_POINT_BYTES > MAX_SRS_FRAME_BYTES:
            die(f"SRS common {label} frame exceeds the native authenticated frame bound")
        header = b"GATZKG1C" + struct.pack(
            "<IIQQQ", 1, G1_POINT_BYTES, count, domain, domain - 1
        )
        path = srs_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        written = 0
        try:
            with path.open("wb") as stream:
                stream.write(header)
                digest.update(header)
                for point in payloads:
                    if len(point) != G1_POINT_BYTES:
                        die(f"SRS common {label} point has an invalid encoded size")
                    stream.write(point)
                    digest.update(point)
                    written += 1
        except (OSError, MemoryError, OverflowError) as exc:
            die(f"SRS common {label} frame materialization failed: {exc}")
        if written != count:
            die(f"SRS common {label} frame point count mismatch: {written} != {count}")
        return {"path": relative.as_posix(), "count": count, "sha256": digest.hexdigest()}

    try:
        tau_point = point_mul(G, tau)

        def generator_points() -> Iterable[bytes]:
            current = G
            for _ in range(domain):
                yield encode_g1(current)
                current = point_add(current, tau_point)

        g_meta = write_frame("G", generator_points(), domain)
        h_meta = write_frame("H", (encode_g1(h_point) for _ in range(1)), 1)
        u_meta = write_frame("U", (encode_g1(u_point) for _ in range(1)), 1)
    except (MemoryError, OverflowError) as exc:
        die(f"SRS generator materialization exceeded process capacity: {exc}")
    families: list[dict[str, Any]] = []
    for family_id, required_domain in sorted(family_domains.items()):
        # The logical family requirement is retained explicitly, while the
        # authenticated frame metadata points to the single common basis.
        chunk = {"family_id": family_id, "chunk_id": f"{family_id}:chunk0", "chunk_index": 0, "point_offset": 0, "point_count": domain, "domain": domain, "required_domain": required_domain, "degree": domain - 1, "point_size": 48, "generator_id": "zkDL_explicit_ceremony_g1_v2", "ceremony_transcript_sha256": transcript_digest, "ceremony_seed_sha256": seed_digest, "generator_digest": g_meta["sha256"], "G": dict(g_meta), "H": dict(h_meta), "U": dict(u_meta)}
        families.append({"family_id": family_id, "chunks": [chunk]})
    default = next(item for item in families if item["family_id"] == "*")["chunks"][0]
    manifest = {"manifest_format": "zkgat_srs_manifest", "manifest_version": 2, "curve_id": "bls12_381", "encoding": "GATZKG1C", "srs_id": values["srs_id"], "chunking_rule": "per_family_prefix_v1", "frame_sharing_rule": "authenticated_common_prefix_v1", "default_family_id": "*", "default_chunk_id": "*:chunk0", "generator_id": "zkDL_explicit_ceremony_g1_v2", "ceremony_transcript_sha256": transcript_digest, "ceremony_seed_sha256": seed_digest, "setup": {"mode": "explicit_ceremony_input", "trust_status": "not_an_external_trust_root", "seed_sha256": seed_digest, "transcript_sha256": transcript_digest}, "families": families, "generator_digest": default["generator_digest"]}
    manifest_digest = write_json(srs_dir / "srs_manifest.json", manifest)
    return manifest_digest, default["generator_digest"], h_point, u_point


def write_plan(root: Path, values: dict[str, str], degree_id: str) -> dict[str, str]:
    dataset = values["dataset"]
    srs_dir = root / values["srs_path"]
    srs_dir.mkdir(parents=True, exist_ok=True)
    srs_plan = {"manifest_format": "zkgat_srs_manifest", "manifest_version": 2, "status": "ceremony_required", "srs_id": values["srs_id"], "curve_id": "bls12_381", "encoding": "GATZKG1C", "chunking_rule": "per_family_prefix_v1", "generator_id": "zkDL_explicit_ceremony_g1_v2", "ceremony_transcript_sha256": "CEREMONY_REQUIRED", "families": [], "generator": "prepare_form_resources.py --full-dataset", "source_root": "baseline-local"}
    srs_digest = write_json(srs_dir / "srs_manifest.json", srs_plan)
    table_path = root / values["static_table_descriptor"]
    table_digest = write_json(table_path, {"schema": "zkDL_gat_static_table_descriptor_plan_v1", "status": "ceremony_required", "dataset": dataset, "generator": "prepare_form_resources.py --config <config> --full-dataset", "source_root": "baseline-local"})
    ledger_path = root / values["integer_bound_ledger"]
    ledger_digest_value = write_json(ledger_path, {"schema": "zkDL_gat_integer_bound_ledger_plan_v1", "status": "ceremony_required", "degree_bound_id": degree_id, "source_root": "baseline-local"})
    registry_path = root / values["model_registry_path"]
    registry_digest = write_json(registry_path, {"schema": "zkDL_gat_model_registry_plan_v1", "status": "ceremony_required", "dataset": dataset, "source_root": "baseline-local"})
    sidecar_path = root / values["model_blinder_sidecar_path"]
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_digest = write_json(sidecar_path, {"schema": "zkDL_gat_model_blinder_sidecar_plan_v1", "status": "ceremony_required", "registry_digest": registry_digest, "source_root": "baseline-local"})
    sidecar_path.chmod(0o600)
    return {"degree_bound_id": degree_id, "srs_manifest_digest": srs_digest, "static_table_digest": table_digest, "integer_bound_ledger_digest": ledger_digest_value, "verifier_registry_digest": registry_digest, "model_blinder_sidecar_digest": sidecar_digest}


def generate(root: Path, config_path: Path, seed: bytes, seed_digest: str, transcript_digest: str, full: bool) -> str:
    values = parse_config(config_path)
    dataset = values.get("dataset")
    if dataset not in DATASETS:
        die(f"unsupported dataset in {config_path}")
    nodes, edges, degree = graph_degree(root, dataset, values)
    degree_id = f"zkdl_degree_v2_{dataset}_{degree}"
    if dataset != "cora" and not full:
        result = write_plan(root, values, degree_id)
        update_config(config_path, {"degree_bound_id": result["degree_bound_id"], "srs_manifest_digest": result["srs_manifest_digest"], "static_table_descriptor": values["static_table_descriptor"], "integer_bound_ledger": values["integer_bound_ledger"], "model_registry_path": values["model_registry_path"], "verifier_registry_digest": result["verifier_registry_digest"], "model_blinder_sidecar_digest": result["model_blinder_sidecar_digest"]})
        return "PLAN_ONLY"
    quant_path = root / "configs" / "quantization_scale_chain.cfg"
    quant_digest = sha256_file(quant_path)
    values = parse_config(config_path)
    table = make_tables(values, quant_digest)
    table_path = root / values["static_table_descriptor"]
    table_file_digest = write_json(table_path, table)
    ledger = make_ledger(root, values, table, table_file_digest, quant_digest)
    ledger_path = root / values["integer_bound_ledger"]
    ledger_file_digest = write_json(ledger_path, ledger)
    model_domains = [math.prod([next_pow2(int(axis)) for axis in (shape[1:] if len(shape) == 3 and shape[0] == 1 else shape)]) for _, shape, _ in load_tensors(root / values["checkpoint_bundle"] / "tensors.txt")]
    metadata = graph_metadata(root, dataset, values)
    family_domains = relation_family_domains(
        values, nodes, edges, table, model_domains,
        graph_count=metadata.get("graph_count", max(1, int(values.get("batch_graphs", "1")))),
    )
    srs_digest, generator_digest, h_point, _ = write_srs(root, values, seed, seed_digest, transcript_digest, family_domains)
    # Install the ceremony-bound SRS identity before deriving the registry's
    # config identity.  Registry and sidecar file digests are written later
    # and are intentionally excluded from config_identity().
    update_config(config_path, {"degree_bound_id": degree_id, "srs_manifest_digest": srs_digest})
    values = parse_config(config_path)
    blinders: dict[str, int] = {}
    for name, _, _ in load_tensors(root / values["checkpoint_bundle"] / "tensors.txt"):
        mapped = name.replace("/seq/kernel", "/model/seq_kernel").replace("/attn_src/kernel", "/model/attn_src").replace("/attn_dst/kernel", "/model/attn_dst").replace("/output_bias", "/model/output_bias")
        if "/model/" not in mapped:
            continue
        blinders[mapped] = derive_scalar(seed, b"blinder:" + dataset.encode() + b":" + mapped.encode())
    manifest_path = root / values["checkpoint_bundle"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["degree_bound_id"] = ledger["degree_bound_id"]
    write_json(manifest_path, manifest)
    registry_entries = model_entries(root, values, derive_scalar(seed, b"tau:" + dataset.encode()), h_point, blinders)
    registry = {"schema": "zkDL_gat_model_registry_v2", "dataset": dataset, "model_arch_id": values["model_arch_id"], "model_param_id": values["model_param_id"], "quant_cfg_id": values["quant_cfg_id"], "static_table_id": values["static_table_id"], "srs_id": values["srs_id"], "srs_manifest_digest": srs_digest, "generator_digest": generator_digest, "config_digest": config_identity(config_path), "scale_chain_digest": quant_digest, "static_table_digest": table_file_digest, "integer_bound_ledger_digest": ledger_file_digest, "model_manifest_digest": sha256_file(root / values["checkpoint_bundle"] / "manifest.json"), "model_tensors": registry_entries, "source_root": "baseline-local"}
    registry["registry_identity_digest"] = sha256_bytes(compact(registry))
    registry["verifier_registry_digest"] = registry["registry_identity_digest"]
    registry_path = root / values["model_registry_path"]
    registry_digest = write_json(registry_path, registry)
    sidecar_entries = [{"tensor_id": item["tensor_id"], "blinder": blinders[item["tensor_id"]].to_bytes(32, "little").hex()} for item in registry_entries]
    sidecar = {"schema": "zkDL_gat_model_blinder_sidecar_v1", "registry_digest": registry_digest, "entries": sidecar_entries, "source_root": "baseline-local"}
    sidecar_path = root / values["model_blinder_sidecar_path"]
    sidecar_digest = write_json(sidecar_path, sidecar)
    sidecar_path.chmod(0o600)
    update_config(config_path, {"degree_bound_id": ledger["degree_bound_id"], "srs_manifest_digest": srs_digest, "verifier_registry_digest": registry_digest, "model_blinder_sidecar_digest": sidecar_digest})
    return "MATERIALIZED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ceremony-seed-file", required=True, type=Path)
    parser.add_argument("--ceremony-transcript-file", required=True, type=Path)
    parser.add_argument("--full-dataset", action="store_true")
    args = parser.parse_args()
    config = args.config.resolve()
    root = config.parent.parent.resolve()
    if config.parent.name != "configs" or root.name != "zkdl":
        die("config must be under comparison/zkdl/configs")
    seed, seed_digest, transcript_digest = ceremony(args.ceremony_seed_file.resolve(), args.ceremony_transcript_file.resolve())
    status = generate(root, config, seed, seed_digest, transcript_digest, args.full_dataset)
    print(json.dumps({"schema": "zkDL_gat_resource_generation_v1", "config": config.relative_to(root).as_posix(), "seed_sha256": seed_digest, "transcript_sha256": transcript_digest, "status": status}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
