#!/usr/bin/env python3
"""Create baseline-local zkDL formal resources.

The native zkDL Hyrax/IPA/Pedersen path uses a process-local G1 basis setup.
The setup is sampled with the operating system CSPRNG, serialized only as
strict curve points, and reused only while the complete authenticated resource
set remains valid.  No ceremony bundle, deterministic seed, or tau-power basis
is accepted by the formal path.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

from formal_provenance import native_binary


DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
R = int("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001", 16)
P = int("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab", 16)
GX = int("17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb", 16)
GY = int("08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1", 16)
G = (GX, GY)


MAX_NATIVE_DOMAIN = 1 << 30
MAX_STATIC_TABLE_ROWS = 1 << 27
NATIVE_BASIS_DOMAIN = 1 << 18
SRS_FRAME_HEADER_BYTES = 8 + 4 + 4 + 8 + 8 + 8
G1_POINT_BYTES = 48
MAX_SRS_FRAME_BYTES = 1 << 32
CANONICAL_SCALE_CHAIN = {
    "schema": "gat_l3_scale_chain", "quant_cfg_id": "gat_l3_scale_chain", "s_input": "16", "s_delta": "256", "s_exp": "16",
    "s_h": "16,16,16",
    "s_w_hidden_0": "4,4,4,4,4,4,4,4", "s_w_hidden_1": "4,4,4,4,4,4,4,4",
    "s_h_prime_hidden_0": "64,64,64,64,64,64,64,64", "s_h_prime_hidden_1": "64,64,64,64,64,64,64,64",
    "s_a_src_hidden_0": "4,4,4,4,4,4,4,4", "s_a_src_hidden_1": "4,4,4,4,4,4,4,4",
    "s_a_dst_hidden_0": "4,4,4,4,4,4,4,4", "s_a_dst_hidden_1": "4,4,4,4,4,4,4,4",
    "s_agg_pre": "16,16", "s_agg": "16,16", "s_cat": "16,16",
    "s_w_output": "4,4", "s_y_prime": "64,64", "s_a_src_output": "4,4", "s_a_dst_output": "4,4",
    "s_b_output": "16,16", "s_out_pre": "16", "s_out_head": "16", "s_out": "16",
    "round": "nearest_ties_away", "rescale": "none", "output_average": "signed_round_div",
}


def die(message: str) -> None:
    raise SystemExit(f"prepare-form-resources: BLOCKED: {message}")


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            die(f"duplicate JSON object key in checkpoint manifest: {key}")
        result[key] = value
    return result


def decimal_u64(text: str, label: str) -> int:
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)", text):
        die(f"{label} is not an unsigned decimal integer")
    value = int(text, 10)
    if value > (1 << 64) - 1:
        die(f"{label} exceeds u64")
    return value


def validated_scale_chain(path: Path, quant_cfg_id: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        die(f"quantization scale-chain file is missing or symlinked: {path}")
    actual = parse_config(path)
    expected = dict(CANONICAL_SCALE_CHAIN)
    expected["quant_cfg_id"] = quant_cfg_id
    if actual != expected:
        die("quantization scale-chain differs from the native canonical fixed-point contract")
    return actual


def formal_input_scale_for(config_path: Path) -> int:
    values = parse_config(config_path)
    root = config_path.resolve().parent.parent
    chain = validated_scale_chain(root / "configs" / "quantization_scale_chain.cfg", values.get("quant_cfg_id", ""))
    return decimal_u64(chain["s_input"], "s_input")


def srs_frame_bytes(count: int) -> int:
    if count < 1:
        die(f"SRS frame point count is invalid: {count}")
    total = SRS_FRAME_HEADER_BYTES + count * G1_POINT_BYTES
    if total > MAX_SRS_FRAME_BYTES:
        die(f"SRS frame exceeds the native authenticated frame byte bound: {total}")
    return total


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


def write_json_atomic(path: Path, value: Any, mode: int | None = None) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temp = Path(raw)
    try:
        os.fchmod(fd, mode if mode is not None else 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temp.unlink(missing_ok=True)
        raise
    if mode is not None:
        path.chmod(mode)
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


def config_binding(values: dict[str, str]) -> str:
    bound = dict(values)
    bound.pop("srs_manifest_digest", None)
    bound.pop("verifier_registry_digest", None)
    bound.pop("model_blinder_sidecar_digest", None)
    return sha256_bytes(compact(bound))


def next_pow2(value: int) -> int:
    if value < 1:
        return 1
    result = 1 << (value - 1).bit_length()

    if result > MAX_NATIVE_DOMAIN:
        die(f"relation domain {result} exceeds the native 2^30 coordinate safety boundary")
    return result


def checked_shape_domain(shape: Iterable[int], label: str) -> int:
    """Return the padded tensor product under the native u32-domain contract."""
    product = 1
    axes = list(shape)
    if not axes:
        die(f"empty relation shape: {label}")
    for axis in axes:
        if axis < 1:
            die(f"non-positive relation shape axis in {label}: {axis}")
        padded = next_pow2(axis)
        if product > MAX_NATIVE_DOMAIN // padded:
            die(f"relation shape product exceeds native coordinate boundary: {label}")
        product *= padded
    return product


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
    for key in (
        "max_score_abs", "max_table_output_abs", "max_delta", "max_exp", "max_sum", "max_aggregate_num",
        "max_dividend_abs", "max_aux_abs", "max_divisor"
    ):
        hash_u64(h, int(ledger[key]))
    families = ledger["families"]
    hash_u64(h, len(families))
    fields = ("max_input_abs", "max_weight_abs", "max_attention_abs", "max_bias_abs", "max_score_abs", "max_table_output_abs", "max_delta",
              "max_exp", "max_sum", "max_aggregate_num", "max_dividend_abs", "max_aux_abs", "max_divisor", "max_field_product_abs")
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
        slope = (3 * x1 * x1) * pow(2 * y1, -1, P) % P
    else:
        slope = (y2 - y1) * pow((x2 - x1) % P, -1, P) % P
    x3 = (slope * slope - x1 - x2) % P
    return x3, (slope * (x1 - x3) - y1) % P


def point_mul(point: tuple[int, int] | None, scalar: int) -> tuple[int, int] | None:
    """Exact BLS12-381 scalar multiplication with projective intermediates."""
    scalar %= R
    if point is None or scalar == 0:
        return None
    px, py = point
    if scalar > R // 2:
        scalar, py = R - scalar, (-py) % P
    x, y, z = 0, 1, 0
    for bit in bin(scalar)[2:]:
        if z:
            aa, bb = x * x % P, y * y % P
            cc = bb * bb % P
            dd = 2 * ((x + bb) ** 2 - aa - cc) % P
            ee = 3 * aa % P
            xx = (ee * ee - 2 * dd) % P
            x, y, z = xx, (ee * (dd - xx) - 8 * cc) % P, 2 * y * z % P
        if bit == '1':
            if not z:
                x, y, z = px, py, 1
                continue
            zz = z * z % P
            hh = (px * zz - x) % P
            rr = 2 * (py * z * zz - y) % P
            if hh == 0:
                if rr != 0:
                    x, y, z = 0, 1, 0
                else:
                    aa, bb = x * x % P, y * y % P
                    cc = bb * bb % P
                    dd = 2 * ((x + bb) ** 2 - aa - cc) % P
                    ee = 3 * aa % P
                    xx = (ee * ee - 2 * dd) % P
                    x, y, z = xx, (ee * (dd - xx) - 8 * cc) % P, 2 * y * z % P
                continue
            h2 = hh * hh % P
            ii = 4 * h2 % P
            jj = hh * ii % P
            vv = x * ii % P
            xx = (rr * rr - jj - 2 * vv) % P
            x, y, z = xx, (rr * (vv - xx) - 2 * y * jj) % P, ((z + hh) ** 2 - zz - h2) % P
    if not z:
        return None
    inverse = pow(z, -1, P)
    inverse2 = inverse * inverse % P
    return x * inverse2 % P, y * inverse2 * inverse % P


def encode_g1(point: tuple[int, int] | None) -> bytes:
    if point is None:
        return bytes([0xC0]) + bytes(47)
    x, y = point
    out = bytearray(x.to_bytes(48, "big"))
    out[0] |= 0x80
    if y > P - y:
        out[0] |= 0x20
    return bytes(out)


def fresh_scalar() -> int:
    """Sample one nonzero field scalar from the operating-system CSPRNG."""
    return secrets.randbelow(R - 1) + 1


def _config_bool(values: dict[str, str] | None, key: str, default: bool) -> bool:
    if values is None or key not in values:
        return default
    raw = values[key].strip().lower()
    if raw not in {"true", "false"}:
        die(f"{key} must be true or false")
    return raw == "true"


def audit_graph(root: Path, dataset: str, values: dict[str, str] | None = None) -> dict[str, Any]:
    values = values or {}
    cache_raw = values.get("cache_root", "data/cache")
    cache_path = Path(cache_raw)
    if cache_path.is_absolute() or ".." in cache_path.parts:
        die(f"cache_root must be baseline-local: {cache_raw}")
    directory = root / cache_path / dataset
    meta: dict[str, str] = {}
    meta_path = directory / "meta.cfg"
    if meta_path.is_symlink() or not meta_path.is_file():
        die(f"native graph metadata is missing or symlinked: {meta_path}")
    meta = parse_config(meta_path)
    if meta.get("name") != dataset:
        die(f"cache dataset name mismatch: {dataset}")
    try:
        nodes = decimal_u64(meta["num_nodes"], "num_nodes")
        graph_count = decimal_u64(meta.get("graph_count", "1"), "graph_count")
    except (KeyError, ValueError) as exc:
        die(f"cache graph dimensions are invalid for {dataset}: {exc}")
    if nodes < 1 or nodes > MAX_NATIVE_DOMAIN or graph_count < 1 or graph_count > MAX_NATIVE_DOMAIN:
        die(f"cache node/graph count exceeds native bounds for {dataset}")
    if dataset != "ppi" and graph_count != 1:
        die("non-PPI canonical graph must contain exactly one graph")
    node_ptr: list[int] = [0, nodes]
    node_ptr_path: Path | None = None
    if dataset == "ppi":
        node_ptr_path = directory / "node_ptr.txt"
        if node_ptr_path.is_symlink() or not node_ptr_path.is_file():
            die(f"missing or symlinked PPI node_ptr: {node_ptr_path}")
        try:
            node_ptr = [decimal_u64(item, "PPI node_ptr") for item in node_ptr_path.read_text(encoding="utf-8").split()]
        except ValueError as exc:
            die(f"malformed PPI node_ptr: {exc}")
        if len(node_ptr) != graph_count + 1 or node_ptr[0] != 0 or node_ptr[-1] != nodes:
            die("PPI node_ptr does not match native graph metadata")
        if any(left >= right or right > nodes for left, right in zip(node_ptr, node_ptr[1:])):
            die("PPI node_ptr must be strictly increasing and bounded")
    edges_path = directory / "edges.txt"
    if edges_path.is_symlink() or not edges_path.is_file():
        die(f"native edge file is missing or symlinked: {edges_path}")
    symmetrize = _config_bool(values, "symmetrize_edges", False)
    deduplicate = _config_bool(values, "deduplicate_edges", True)
    seen: set[tuple[int, int, int]] = set()
    incoming = [0] * nodes
    edge_counts = [0] * graph_count
    edge_count = 0
    duplicate_edges = 0
    raw_edge_rows = 0
    raw_self_loops = 0
    symmetrized_edges = 0

    def graph_for_node(node: int) -> int:
        if dataset != "ppi":
            return 0
        import bisect
        return bisect.bisect_right(node_ptr, node) - 1

    def add(gid: int, src: int, dst: int) -> bool:
        nonlocal edge_count, duplicate_edges
        if gid < 0 or gid >= graph_count or src < 0 or src >= nodes or dst < 0 or dst >= nodes:
            die("edge graph id or endpoint is outside native bounds")
        if not node_ptr[gid] <= src < node_ptr[gid + 1] or not node_ptr[gid] <= dst < node_ptr[gid + 1]:
            die("edge endpoint graph ownership does not match node_ptr")
        key = (gid, src, dst)
        if deduplicate and key in seen:
            duplicate_edges += 1
            return False
        if deduplicate:
            seen.add(key)
        if edge_count >= MAX_NATIVE_DOMAIN // 2:
            die("normalized edge count exceeds native sparse-route domain bound")
        incoming[dst] += 1
        edge_counts[gid] += 1
        edge_count += 1
        return True

    with edges_path.open(encoding="utf-8") as stream:
        for number, raw in enumerate(stream, 1):
            parts = raw.split()
            if not parts:
                continue
            raw_edge_rows += 1
            expected = 3 if dataset == "ppi" else 2
            if len(parts) != expected:
                die(f"malformed edge row {number}: expected {expected} columns")
            parsed = [decimal_u64(item, f"edge row {number}") for item in parts]
            gid = parsed[0] if dataset == "ppi" else 0
            if gid < 0 or gid >= graph_count:
                die(f"edge graph id out of range at row {number}")
            src, dst = (parsed[1], parsed[2]) if dataset == "ppi" else (parsed[0], parsed[1])
            if dataset == "ppi" and (not node_ptr[gid] <= src < node_ptr[gid + 1] or not node_ptr[gid] <= dst < node_ptr[gid + 1]):
                die(f"PPI global edge endpoint out of graph bounds at row {number}")
            if src == dst:
                raw_self_loops += 1
            add(gid, src, dst)
            if symmetrize and add(gid, dst, src):
                symmetrized_edges += 1
    for key in ("num_edges", "edge_count", "raw_edge_count", "num_raw_edges"):
        if key in meta:
            try:
                declared_raw_edges = decimal_u64(meta[key], f"cache metadata {key}")
            except SystemExit as exc:
                die(f"cache metadata {key} is not an integer: {exc}")
            if declared_raw_edges != raw_edge_rows:
                die(f"cache metadata raw edge count mismatch: {key}={declared_raw_edges} rows={raw_edge_rows}")
    added_self_loops = 0
    for node in range(nodes):
        if add(graph_for_node(node), node, node):
            added_self_loops += 1
    node_edge_ptr = [0]
    for degree in incoming:
        node_edge_ptr.append(node_edge_ptr[-1] + degree)
    edge_ptr = [0]
    for count in edge_counts:
        if count == 0:
            die("normalized graph edge_ptr contains an empty graph block")
        edge_ptr.append(edge_ptr[-1] + count)
    if node_edge_ptr[-1] != edge_count or edge_ptr[-1] != edge_count:
        die("normalized edge/CSR prefix pointers do not cover the canonical edge list")
    if any(left > right for left, right in zip(node_edge_ptr, node_edge_ptr[1:])) or any(left >= right for left, right in zip(edge_ptr, edge_ptr[1:])):
        die("normalized edge/CSR prefix pointers are not monotone")
    if any(degree == 0 for degree in incoming):
        die("normalized CSR contains a destination without an incoming self-loop or edge")
    def pointer_digest(pointer: list[int]) -> str:
        return sha256_bytes(("\n".join(map(str, pointer)) + "\n").encode())
    return {"dataset": dataset, "nodes": nodes, "graphs": graph_count, "node_id_space": "canonical_flat_equals_proof_flat",
            "raw_edge_id_space": "gid_src_flat_dst_flat" if dataset == "ppi" else "src_flat_dst_flat", "raw_edge_rows": raw_edge_rows,
            "raw_edge_ownership_checks": raw_edge_rows, "raw_self_loops": raw_self_loops, "symmetrized_edges_added": symmetrized_edges,
            "duplicate_edges_removed": duplicate_edges, "self_loops_added": added_self_loops, "normalized_edges": edge_count,
            "normalized_edge_counts_by_graph": edge_counts, "node_ptr_count": len(node_ptr), "node_ptr_prefix": node_ptr[:3],
            "node_ptr_suffix": node_ptr[-3:], "node_ptr_sha256": pointer_digest(node_ptr), "edge_ptr_count": len(edge_ptr),
            "edge_ptr_prefix": edge_ptr[:3], "edge_ptr_suffix": edge_ptr[-3:], "edge_ptr_sha256": pointer_digest(edge_ptr),
            "csr_pointer_count": len(node_edge_ptr), "csr_pointer_sha256": pointer_digest(node_edge_ptr),
            "csr_destinations_checked": nodes, "csr_min_in_degree": min(incoming, default=0),
            "csr_max_in_degree": max(incoming, default=0), "destination_groups_checked": nodes,
            "edge_blocks_checked": graph_count, "edge_file_sha256": sha256_file(edges_path),
            "meta_sha256": sha256_file(meta_path),
            "node_ptr_file_sha256": sha256_file(node_ptr_path) if node_ptr_path is not None else None}

def graph_degree(root: Path, dataset: str, values: dict[str, str] | None = None) -> tuple[int, int, int]:
    audit = audit_graph(root, dataset, values)
    return int(audit["nodes"]), int(audit["normalized_edges"]), int(audit["csr_max_in_degree"])


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
            if re.fullmatch(r"(?:0|[1-9][0-9]*)", value):
                metadata[key] = decimal_u64(value, f"cache metadata {key}")
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
        result.append((decimal_u64(heads, "hidden head count"), decimal_u64(width, "hidden head dimension")))
    if not result or any(heads < 1 or width < 1 for heads, width in result):
        die("hidden_profile has no positive layer/head dimensions")
    return result


def parse_list(value: str) -> list[int]:
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        die(f"invalid list value: {value}")
    result = [decimal_u64(item.strip(), "shape axis") for item in text[1:-1].split(",") if item.strip()]
    if not result or any(item < 1 for item in result):
        die(f"list has no positive entries: {value}")
    return result


def relation_family_domains(values: dict[str, str], nodes: int, edges: int,
                            table: dict[str, Any], model_domains: list[int],
                            graph_count: int | None = None,
                            ledger_families: list[dict[str, Any]] | None = None,
                            quantization: dict[str, str] | None = None) -> dict[str, int]:
    """Mirror the authenticated RelationPlan's padded tensor families.

    The native runner derives every family from these same public dimensions.
    Keeping this calculation in the resource tool makes SRS preparation
    independent of a prover-side witness and prevents a guessed global cap
    from truncating rounddiv_aux or edge-feature families.
    """
    profile = parse_profile(values["hidden_profile"])
    input_dims = parse_list(values["d_in_profile"])
    classes = decimal_u64(values["num_classes"], "num_classes")
    output_heads = decimal_u64(values["K_out"], "K_out")
    range_bits = decimal_u64(values["range_bits"], "range_bits")
    graph_count = graph_count if graph_count is not None else max(1, decimal_u64(values.get("batch_graphs", "1"), "batch_graphs"))
    if graph_count < 1:
        die("native graph_count must be positive")
    if ledger_families is None or quantization is None:
        die("relation-family planning requires the authenticated ledger families and canonical quantization chain")
    ledger_by_id: dict[str, dict[str, Any]] = {}
    for family in ledger_families:
        family_id = family.get("family_id")
        if not isinstance(family_id, str) or not family_id or family_id in ledger_by_id:
            die("integer-bound ledger has a malformed or duplicate family id")
        ledger_by_id[family_id] = family

    def ledger_value(family: dict[str, Any], key: str, label: str) -> int:
        value = family.get(key)
        if type(value) is not int or value < 0:
            die(f"integer-bound ledger {label} is not a nonnegative integer")
        return value

    def quantization_vector(key: str) -> list[int]:
        raw = quantization.get(key)
        if not isinstance(raw, str) or not raw:
            die(f"canonical quantization chain is missing {key}")
        return [decimal_u64(part.strip(), f"quantization {key}") for part in raw.split(",")]

    def unsigned_width(inclusive_max: int, label: str) -> int:
        if inclusive_max < 0:
            die(f"signed-round bound is negative: {label}")
        return max(1, inclusive_max.bit_length())

    def ceil_positive_div(numerator: int, denominator: int, label: str) -> int:
        if numerator < 0 or denominator <= 0:
            die(f"signed-round division bound is invalid: {label}")
        return (numerator + denominator - 1) // denominator

    output_heads = decimal_u64(values["K_out"], "K_out")
    output_pre = decimal_u64(quantization.get("s_out_pre", ""), "quantization s_out_pre")
    output_y_prime = quantization_vector("s_y_prime")
    hidden_profile = parse_profile(values["hidden_profile"])
    hidden_scales = [quantization_vector(f"s_h_prime_hidden_{layer}") for layer in range(len(hidden_profile))]

    def signed_round_widths(operation: str) -> dict[str, int]:
        if operation == "final_average":
            output_families = [
                ledger_by_id.get(f"output/head{head}") for head in range(output_heads)
            ]
            if not output_families or any(family is None for family in output_families):
                die("final-average signed-round ledger families are incomplete")
            a_max = max(ledger_value(family, "max_dividend_abs", operation) for family in output_families if family is not None)
            aux_max = max(ledger_value(family, "max_aux_abs", operation) for family in output_families if family is not None)
            divisor_max = output_heads
            q_max = ceil_positive_div(a_max, divisor_max, operation + "/q")
            r_max = 2 * divisor_max - 1
            auxiliary_max = aux_max
        else:
            hidden_match = re.fullmatch(r"hidden/layer([0-9]+)/head([0-9]+)/signed_round_div", operation)
            output_match = re.fullmatch(r"output/head([0-9]+)/(signed_round_div|output_scale_alignment)", operation)
            if hidden_match:
                layer, head = (int(value, 10) for value in hidden_match.groups())
                if layer >= len(hidden_profile) or head >= hidden_profile[layer][0] or head >= len(hidden_scales[layer]):
                    die(f"hidden signed-round operation is outside the canonical quantization chain: {operation}")
                family = ledger_by_id.get(f"hidden/layer{layer}/head{head}")
                if family is None:
                    die(f"hidden signed-round ledger family is missing: {operation}")
                denominator_scale = hidden_scales[layer][head]
                a_max = ledger_value(family, "max_dividend_abs", operation)
                divisor_max = ledger_value(family, "max_divisor", operation)
                q_max = ceil_positive_div(a_max, denominator_scale, operation + "/q")
                binds_raw_denominator = True
            elif output_match:
                head = int(output_match.group(1), 10)
                role = output_match.group(2)
                if head >= output_heads or head >= len(output_y_prime):
                    die(f"output signed-round operation is outside the canonical quantization chain: {operation}")
                family = ledger_by_id.get(f"output/head{head}")
                if family is None:
                    die(f"output signed-round ledger family is missing: {operation}")
                a_max = ledger_value(family, "max_dividend_abs", operation)
                if role == "signed_round_div":
                    denominator_scale = output_y_prime[head]
                    divisor_max = ledger_value(family, "max_divisor", operation)
                    q_max = ceil_positive_div(a_max, denominator_scale, operation + "/q")
                    binds_raw_denominator = True
                else:
                    denominator_scale = 1
                    divisor_max = output_pre
                    q_max = ceil_positive_div(a_max, divisor_max, operation + "/q")
                    binds_raw_denominator = False
                auxiliary_max = ledger_value(family, "max_aux_abs", operation)
            else:
                die(f"noncanonical signed-round operation id in relation-family planner: {operation}")
            r_max = 2 * divisor_max - 1
            if hidden_match:
                auxiliary_max = ledger_value(family, "max_aux_abs", operation)

        if max(a_max, q_max, r_max, divisor_max) > auxiliary_max:
            die(f"signed-round range exceeds its authenticated family ledger: {operation}")

        widths = {
            "a": unsigned_width(a_max, operation + "/a"),
            "q_abs": unsigned_width(q_max, operation + "/q"),
            "r": unsigned_width(r_max, operation + "/r"),
            "delta": unsigned_width(r_max, operation + "/delta"),
            "divisor": unsigned_width(divisor_max, operation + "/divisor"),
        }
        if any(width > 63 for width in widths.values()):
            die(f"signed-round role width exceeds native int64 decomposition: {operation}")
        return widths
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
    lrelu_rows = decimal_u64(str(table["lrelu_max"] - table["lrelu_min"] + 1), "lrelu table rows")
    elu_rows = decimal_u64(str(table["elu_max"] - table["elu_min"] + 1), "elu table rows")
    exp_rows = decimal_u64(str(table["exp_max"] + 1), "exp table rows")
    def table_index_bits(logical_rows: int) -> int:
        if logical_rows < 2 or logical_rows > MAX_NATIVE_DOMAIN:
            die(f"lookup table has invalid logical row count: {logical_rows}")
        return (logical_rows - 1).bit_length()
    lrelu_index_bits = table_index_bits(lrelu_rows)
    elu_index_bits = table_index_bits(elu_rows)
    exp_index_bits = table_index_bits(exp_rows)
    # max-range delta is an index into the authenticated ExpMap table. Its
    # decomposition width follows the table's actual largest legal index,
    # independently of range_bits, which defines the LReLU/ELU input domain.
    delta_range_bits = exp_index_bits
    add("static_lrelu", lrelu_rows)
    add("static_elu", elu_rows)
    add("static_exp", exp_rows)
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
            add(prefix + "/trace/lrelu_bits", edges, lrelu_index_bits)
            add(prefix + "/trace/exp_bits", edges, exp_index_bits)
            for name in ("inverse_field", "alpha_field"):
                add(prefix + "/trace/" + name, edges)
            add(prefix + "/trace/maximum", nodes)
            add(prefix + "/trace/denominator", nodes)
            add(prefix + "/trace/range_bits", edges, delta_range_bits)
            add(prefix + "/trace/rounddiv_aux", nodes, head_dim, 8)
            for name in ("rounddiv_a_inverse", "rounddiv_divisor_inverse"):
                add(prefix + "/trace/" + name, nodes, head_dim)
            for name in ("routed_h_prime", "product_u_h_prime", "product_alpha_h_prime_field",
                         "aggregate_num", "aggregate_pre_field", "consistency_residual_field",
                         "aggregate_pre", "elu_input", "aggregate"):
                add(prefix + "/trace/" + name, edges, head_dim) if name.startswith(("routed", "product")) else add(prefix + "/trace/" + name, nodes, head_dim)
            add(prefix + "/trace/" + "elu_idx", nodes, head_dim)
            add(prefix + "/trace/" + "elu_bits", nodes, head_dim, elu_index_bits)
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
        add(prefix + "/trace/lrelu_bits", edges, lrelu_index_bits)
        add(prefix + "/trace/exp_bits", edges, exp_index_bits)
        for name in ("inverse_field", "alpha_field"):
            add(prefix + "/trace/" + name, edges)
        add(prefix + "/trace/maximum", nodes)
        add(prefix + "/trace/denominator", nodes)
        add(prefix + "/trace/range_bits", edges, delta_range_bits)
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
        padded_shape = node_shape(next_pow2(width))
        for role in ("signed_numerator", "divisor", "signed_quotient", "a", "q_abs",
                     "r", "delta", "sigma", "z", "a_inverse", "divisor_inverse"):
            view(operation + "/view/" + role, *(padded_shape if role == "divisor" else shape))
        widths = signed_round_widths(operation)
        if operation != "final_average" and not operation.endswith("/output_scale_alignment"):
            view(operation + "/view/divisor_aux", *padded_shape)
        for value in ("a", "q_abs", "r", "delta", "divisor"):
            for bit in range(widths[value]):
                view(operation + f"/view/range/{value}/{bit}", *shape)

    def append_table_views(operation: str, edge_domain: bool,
                           feature_width: int, table_size: int) -> None:
        shape = edge_shape(feature_width) if edge_domain else node_shape(feature_width)
        for role in ("query", "index", "output0"):
            view(operation + "/view/" + role, *shape)
        depth = table_depth(table_size)
        for bit in range(depth):
            view(operation + f"/view/bit/{bit}", *shape)

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
        for bit in range(delta_range_bits):
            view(maximum + f"/view/range_bit/{bit}", *edge)

        group = stage + "/group_sum"
        for role in ("group_new", "group_end", "input0", "output0", "prefix0", "prefix_prev0"):
            view(group + "/view/" + role, *edge_shape(next_pow2(width)))
        for role in ("input1", "output1", "prefix1", "prefix_prev1"):
            view(group + "/view/" + role, *edge_feature)

        alpha_product = stage + ("/alpha_y_prime" if output else "/alpha_h_prime")
        view(alpha_product + "/view/input0", *edge_shape(next_pow2(width)))

        inverse = stage + "/inverse_alpha"
        for role in ("input0", "input1", "output0", "output1"):
            view(inverse + "/view/" + role, *edge)

        for operation in (stage + "/field_aggregate", stage + "/integer_numerator"):
            for role in ("group_new", "group_end"):
                view(operation + "/view/" + role, *edge_shape(next_pow2(width)))
            for role in ("input0", "output0", "prefix0", "prefix_prev0"):
                view(operation + "/view/" + role, *edge_feature)

        consistency = stage + "/consistency"
        view(consistency + "/view/input0", *node_shape(next_pow2(width)))
        for role in ("input1", "input2", "output0"):
            view(consistency + "/view/" + role, *node_feature)

        append_round_views(stage + "/signed_round_div", width)
        append_table_views(stage + "/leaky_relu", True, 1, lrelu_rows)
        append_table_views(stage + "/exp_map", True, 1, exp_rows)
        if not output:
            append_table_views(stage + "/elu", False, width, elu_rows)
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


def _text_tokens(path: Path) -> Iterable[str]:
    """Stream whitespace tokens without retaining the tensor export in memory."""
    pending = ""
    with path.open("r", encoding="utf-8") as stream:
        while True:
            block = stream.read(1 << 16)
            if not block:
                yield from pending.split()
                return
            combined = pending + block
            tokens = combined.split()
            if tokens and not combined[-1].isspace():
                pending = tokens.pop()
            else:
                pending = ""
            if len(pending) > 4096:
                die(f"tensor export contains an oversized token: {path}")
            yield from tokens


def audit_tensor_export(path: Path, values: dict[str, str]) -> dict[str, Any]:
    """Check model names/shapes and quantized bounds without retaining tensor values."""
    profile = parse_profile(values["hidden_profile"])
    input_dims = parse_list(values["d_in_profile"])
    classes = decimal_u64(values["num_classes"], "num_classes")
    output_heads = decimal_u64(values["K_out"], "K_out")
    expected: dict[str, list[int]] = {}
    current_input = input_dims[0]
    for layer, (head_count, head_dim) in enumerate(profile):
        for head in range(head_count):
            prefix = f"hidden/layer{layer}/head{head}"
            expected[prefix + "/seq/kernel"] = [1, current_input, head_dim]
            expected[prefix + "/attn_src/kernel"] = [1, head_dim, 1]
            expected[prefix + "/attn_dst/kernel"] = [1, head_dim, 1]
        current_input = head_count * head_dim
    for head in range(output_heads):
        prefix = f"output/head{head}"
        expected[prefix + "/seq/kernel"] = [1, current_input, classes]
        expected[prefix + "/attn_src/kernel"] = [1, classes, 1]
        expected[prefix + "/attn_dst/kernel"] = [1, classes, 1]
        expected[prefix + "/output_bias"] = [classes]

    tokens = iter(_text_tokens(path))

    def take(label: str) -> str:
        try:
            return next(tokens)
        except StopIteration:
            die(f"truncated tensor export while reading {label}: {path}")

    seen: set[str] = set()
    domains: list[int] = []
    model_max = 1
    tensor_count = 0
    model_shapes: dict[str, list[int]] = {}
    try:
        marker = next(tokens)
    except StopIteration:
        die(f"tensor export is empty: {path}")
    while True:
        if marker != "TENSOR":
            die(f"malformed tensor export marker {marker!r}: {path}")
        name = take("tensor name")
        if name in seen:
            die(f"duplicate model tensor in export: {name}")
        seen.add(name)
        try:
            rank = decimal_u64(take("tensor rank"), "tensor rank")
            if rank < 1 or rank > 8:
                die(f"model tensor rank is outside native metadata bounds: {name}")
            shape = [decimal_u64(take(f"{name} axis"), f"{name} axis") for _ in range(rank)]
            count = decimal_u64(take(f"{name} value count"), f"{name} value count")
        except ValueError as exc:
            die(f"malformed tensor shape/count for {name}: {exc}")
        if any(axis < 1 for axis in shape):
            die(f"model tensor has a non-positive axis: {name}")
        logical_count = 1
        for axis in shape:
            if logical_count > MAX_NATIVE_DOMAIN // axis:
                die(f"model tensor logical count exceeds native domain: {name}")
            logical_count *= axis
        if logical_count != count:
            die(f"tensor count mismatch: {name}")
        if name not in expected or shape != expected[name]:
            die(f"model tensor name/shape differs from the canonical architecture: {name}")
        model_shapes[name] = shape
        logical = shape[1:] if len(shape) == 3 and shape[0] == 1 else shape
        domain = checked_shape_domain(logical, name)
        domains.append(domain)
        scale = 16 if name.endswith("output_bias") else 4
        for _ in range(count):
            raw = take(f"{name} value")
            try:
                value = float(raw)
            except ValueError as exc:
                die(f"malformed model scalar in {name}: {exc}")
            if not math.isfinite(value):
                die(f"non-finite model scalar in {name}")
            quantized = round_away(value * scale)
            if quantized < -(1 << 63) or quantized > (1 << 63) - 1:
                die(f"quantized model scalar exceeds signed native range: {name}")
            model_max = max(model_max, abs(quantized))
        tensor_count += 1
        try:
            marker = next(tokens)
        except StopIteration:
            break
    missing = sorted(set(expected) - seen)
    if missing:
        die(f"checkpoint tensor export is missing canonical model parameters: {missing[0]}")
    return {"tensor_count": tensor_count, "model_shapes": model_shapes, "model_padded_domains": domains,
            "max_model_padded_domain": max(domains, default=1), "max_quantized_model_abs": model_max,
            "tensor_export_sha256": sha256_file(path)}


def registered_model_segment_blinders(tensor_id: str, total_blinder: int, domain: int, segment_size: int, segment_count: int, srs_manifest_digest: str) -> list[int]:
    if not 0 < total_blinder < R or domain < 1 or domain > MAX_NATIVE_DOMAIN or domain & (domain - 1) or segment_size < 1 or segment_size > NATIVE_BASIS_DOMAIN or segment_size & (segment_size - 1) or segment_count < 1 or segment_count > 4096 or domain != segment_size * segment_count or not re.fullmatch(r"[0-9a-f]{64}", srs_manifest_digest):
        die("registered model segment blinder inputs are invalid")
    if segment_count == 1:
        return [total_blinder]
    key = total_blinder.to_bytes(32, "little")
    prefix = append_string("zkdl.registered-model.segment-blinders.v1") + append_string(tensor_id) + append_u32(domain) + append_u32(segment_size) + append_u32(segment_count) + bytes.fromhex(srs_manifest_digest)
    for nonce in range(1 << 16):
        shares = [0] * segment_count
        for segment_index in range(1, segment_count):
            for attempt in range(1 << 16):
                raw = hmac.new(key, prefix + append_u32(nonce) + append_u32(segment_index) + append_u32(attempt), hashlib.sha256).digest()
                scalar = int.from_bytes(raw, "little")
                if 0 < scalar < R:
                    shares[segment_index] = scalar
                    break
            else:
                die("registered model segment blinder rejection bound exceeded")
        shares[0] = (total_blinder - sum(shares[1:])) % R
        if shares[0] != 0:
            return shares
    die("registered model segment blinder nonce bound exceeded")


def cartesian_model_values(logical: list[int], padded: list[int], values: list[int]) -> list[int]:
    if not logical or len(logical) != len(padded) or any(left < 1 or right < left or right & (right - 1) for left, right in zip(logical, padded)) or math.prod(logical) != len(values):
        die("model tensor logical/padded shape is inconsistent")
    result = [0] * math.prod(padded)
    for index, value in enumerate(values):
        remaining, destination, stride = index, 0, 1
        for left, right in reversed(list(zip(logical, padded))):
            destination += (remaining % left) * stride
            remaining //= left
            stride *= right
        result[destination] = value
    return result


def model_entries(root: Path, values: dict[str, str],
                  h_point: tuple[int, int], u_point: tuple[int, int],
                  blinders: dict[str, int], generators: list[tuple[int, int]],
                  srs_manifest_digest: str) -> list[dict[str, Any]]:
    tensors_path = root / values["checkpoint_bundle"] / "tensors.txt"
    entries: list[dict[str, Any]] = []
    power = 1
    for name, shape, floats in load_tensors(tensors_path):
        if "/hidden/" in "/" + name:
            prefix = name.split("/seq/kernel")[0].split("/attn_")[0]
            mapped = name.replace("/seq/kernel", "/model/seq_kernel").replace("/attn_src/kernel",
                                                                              "/model/attn_src").replace("/attn_dst/kernel", "/model/attn_dst")
            scale = 4
        elif name.startswith("output/"):
            mapped = name.replace("/seq/kernel", "/model/seq_kernel").replace(
                "/attn_src/kernel", "/model/attn_src"
            ).replace("/attn_dst/kernel", "/model/attn_dst").replace("/output_bias", "/model/output_bias")
            scale = 16 if name.endswith("output_bias") else 4
        else:
            continue
        logical = shape[1:] if len(shape) == 3 and shape[0] == 1 else shape
        padded = [next_pow2(int(axis)) for axis in logical]
        domain = math.prod(padded)
        quantized = cartesian_model_values(logical, padded, [round_away(value * scale) % R for value in floats])
        segment_size = min(domain, len(generators))
        if segment_size < 1 or domain % segment_size != 0 or segment_size & (segment_size - 1):
            die(f"model tensor domain is not covered by authenticated IPA segments: {mapped}")
        segment_count = domain // segment_size
        frame_digest = ipa_basis_digest(
            generators[:segment_size], h_point, u_point,
            1 << (segment_size.bit_length()))
        basis_digest_text = ipa_basis_digest(
            generators, h_point, u_point, 1 << (len(generators).bit_length()))
        segment_blinders = registered_model_segment_blinders(mapped, blinders[mapped], domain, segment_size, segment_count, srs_manifest_digest)
        segments: list[dict[str, Any]] = []
        for segment_index in range(segment_count):
            offset = segment_index * segment_size
            values_segment = quantized[offset:offset + segment_size]
            segment_point: tuple[int, int] | None = None
            for index, quantized_value in enumerate(values_segment):
                segment_point = point_add(
                    segment_point,
                    point_mul(generators[index], quantized_value))

            segment_point = point_add(segment_point, point_mul(h_point, segment_blinders[segment_index]))
            if segment_point is None:
                die(f"model tensor segment commitment is the identity: {mapped}/{segment_index}")
            actual_length = min(segment_size, domain - offset)
            segments.append({
                "segment_index": segment_index,
                "logical_offset": offset,
                "actual_length": actual_length,
                "padded_length": segment_size,
                "padding_length": segment_size - actual_length,
                "frame_digest": frame_digest,
                "commitment": encode_g1(segment_point).hex(),
            })
        commitment: tuple[int, int] | None = None
        for segment in segments:
            commitment = point_add(commitment, decode_g1(bytes.fromhex(segment["commitment"]), mapped))
        if commitment is None:
            die(f"model tensor commitment is the identity: {mapped}")
        vector_digest = segment_vector_digest(
            mapped, 2, domain, 1, segment_size, segment_count,
            srs_manifest_digest, basis_digest_text, segments)
        entries.append({
            "tensor_id": mapped,
            "domain_size": domain,
            "logical_shape": logical,
            "padded_shape": padded,
            "commitment": encode_g1(commitment).hex(),
            "segment_binding_version": 1,
            "segment_size": segment_size,
            "segment_count": segment_count,
            "srs_manifest_digest": srs_manifest_digest,
            "basis_digest": basis_digest_text,
            "segment_commitment_digest": vector_digest,
            "segment_commitments": segments,
        })
        power = max(power, domain)
    if not entries:
        die(f"checkpoint has no canonical GAT model tensors: {tensors_path}")
    return entries


def decode_g1(encoded: bytes, label: str) -> tuple[int, int]:
    if len(encoded) != G1_POINT_BYTES:
        die(f"local SRS {label} point has invalid encoded size")
    flags = encoded[0]
    if not (flags & 0x80) or flags & 0x40:
        die(f"local SRS {label} point is not a finite compressed G1 point")
    x_bytes = bytearray(encoded)
    sort_flag = bool(x_bytes[0] & 0x20)
    x_bytes[0] &= 0x1F
    x = int.from_bytes(x_bytes, "big")
    if x >= P or P % 4 != 3:
        die(f"local SRS {label} point has an invalid field encoding")
    rhs = (x * x * x + 4) % P
    y = pow(rhs, (P + 1) // 4, P)
    if y * y % P != rhs:
        die(f"local SRS {label} point is not on the BLS12-381 G1 curve")
    if (y > P - y) != sort_flag:
        y = P - y
    return x, y


def srs_frame_path(srs_dir: Path, descriptor: dict[str, Any], label: str) -> tuple[Path, str, int, int]:
    if not isinstance(descriptor, dict):
        die(f"local SRS {label} frame descriptor is not an object")
    raw = descriptor.get("path")
    expected = descriptor.get("sha256")
    count = descriptor.get("count")
    frame_bytes = descriptor.get("bytes")
    if (not isinstance(raw, str) or not raw or Path(raw).is_absolute() or ".." in Path(raw).parts or
            not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or
            not isinstance(count, int) or count < 1 or not isinstance(frame_bytes, int) or
            frame_bytes != srs_frame_bytes(count)):
        die(f"local SRS {label} frame descriptor size/count is malformed")
    path = (srs_dir / Path(raw)).resolve()
    try:
        path.relative_to(srs_dir.resolve())
    except ValueError:
        die(f"local SRS {label} frame escapes SRS directory")
    if path.is_symlink() or not path.is_file():
        die(f"local SRS {label} frame is missing")
    if path.stat().st_size != frame_bytes:
        die(f"local SRS {label} frame byte size mismatch")
    if sha256_file(path) != expected:
        die(f"local SRS {label} frame digest mismatch")
    return path, expected, count, frame_bytes


def read_srs_frame_prefix(srs_dir: Path, descriptor: dict[str, Any], label: str,
                          required_count: int) -> list[tuple[int, int]]:
    path, _, declared_count, _ = srs_frame_path(srs_dir, descriptor, label)
    if required_count < 1 or required_count > declared_count:
        die(f"local SRS {label} frame does not cover required model domain")
    root = Path(__file__).resolve().parent.parent
    helper = native_binary(root, "gat_resource_setup")
    if helper.is_symlink() or not helper.is_file():
        die("baseline-local native SRS decoder is missing")
    try:
        result = subprocess.run([str(helper), "export-srs-prefix", "--frame", str(path), "--count", str(required_count), "--baseline-root", str(root)], cwd=root, check=True, stdout=subprocess.PIPE, text=True)
        exported = json.loads(result.stdout)
        if exported.get("schema") != "zkDL_gat_srs_prefix_v1" or not isinstance(exported.get("points"), list) or len(exported["points"]) != required_count:
            die(f"native SRS {label} prefix export identity/count mismatch")
        points = []
        for raw in exported["points"]:
            if not isinstance(raw, list) or len(raw) != 2 or any(not isinstance(axis, str) or not re.fullmatch(r"[0-9]+", axis) for axis in raw):
                die(f"native SRS {label} prefix has malformed affine coordinates")
            x, y = int(raw[0]), int(raw[1])
            if not (0 <= x < P and 0 <= y < P) or y * y % P != (x * x * x + 4) % P:
                die(f"native SRS {label} prefix has invalid affine coordinates")
            points.append((x, y))
        return points
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        die(f"cannot decode authenticated native SRS {label} frame: {exc}")


def basis_digest(g_meta: dict[str, Any], h_meta: dict[str, Any], u_meta: dict[str, Any]) -> str:
    return sha256_bytes(compact({
        "G": g_meta["sha256"],
        "H": h_meta["sha256"],
        "U": u_meta["sha256"],
        "schema": "zkDL_gat_basis_digest_v1",
    }))


def append_u32(value: int) -> bytes:
    return struct.pack("<I", value)


def append_u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def append_string(value: str) -> bytes:
    encoded = value.encode()
    return append_u64(len(encoded)) + encoded


def ipa_basis_digest(generators: list[tuple[int, int]], h_point: tuple[int, int],
                     u_point: tuple[int, int], padded_size: int) -> str:
    encoded = append_u32(len(generators)) + append_u32(padded_size)
    encoded += b"".join(encode_g1(point) for point in generators)
    encoded += encode_g1(h_point) + encode_g1(u_point)
    return sha256_bytes(encoded)


def segment_vector_digest(tensor_id: str, visibility: int, domain: int,
                          version: int, segment_size: int, segment_count: int,
                          srs_manifest_digest: str, basis_digest_text: str,
                          segments: list[dict[str, Any]]) -> str:
    encoded = append_string("zkdl.segment-commitment-vector.v1")
    encoded += append_string(tensor_id)
    encoded += append_u32(visibility) + append_u32(domain)
    encoded += append_u32(version) + append_u32(segment_size)
    encoded += append_u32(segment_count)
    encoded += bytes.fromhex(srs_manifest_digest)
    encoded += bytes.fromhex(basis_digest_text)
    encoded += append_u64(len(segments))
    for segment in segments:
        encoded += append_u32(segment["segment_index"])
        encoded += append_u64(segment["logical_offset"])
        encoded += append_u32(segment["actual_length"])
        encoded += append_u32(segment["padded_length"])
        encoded += append_u32(segment["padding_length"])
        encoded += bytes.fromhex(segment["frame_digest"])
        encoded += bytes.fromhex(segment["commitment"])
    return sha256_bytes(encoded)


def load_srs_basis(root: Path, values: dict[str, str], required_domain: int) -> tuple[
        list[tuple[int, int]], tuple[int, int], tuple[int, int], str, str]:
    srs_dir = root / values["srs_path"]
    manifest_path = srs_dir / "srs_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_json_keys)
    except (OSError, json.JSONDecodeError) as exc:
        die(f"cannot read installed local SRS manifest: {exc}")
    families = manifest.get("families") if isinstance(manifest, dict) else None
    if not isinstance(families, list) or not families:
        die("installed local SRS manifest has no authenticated families")
    default_family = manifest.get("default_family_id")
    default_chunk = manifest.get("default_chunk_id")
    selected: dict[str, Any] | None = None
    for family in families:
        if not isinstance(family, dict):
            continue
        if default_family is not None and family.get("family_id") != default_family:
            continue
        chunks = family.get("chunks")
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if isinstance(chunk, dict) and (default_chunk is None or chunk.get("chunk_id") == default_chunk):
                selected = chunk
                break
        if selected is not None:
            break
    if selected is None:
        for family in families:
            if not isinstance(family, dict) or family.get("family_id") != "*":
                continue
            chunks = family.get("chunks")
            if isinstance(chunks, list) and chunks and isinstance(chunks[0], dict):
                selected = chunks[0]
                break
    if selected is None:
        die("installed local SRS manifest has no selectable default chunk")
    basis_domain = selected.get("basis_domain")
    if not isinstance(basis_domain, int) or basis_domain < 1 or basis_domain & (basis_domain - 1):
        die("installed local SRS default chunk has no canonical basis_domain")

    generators = read_srs_frame_prefix(srs_dir, selected.get("G"), "G", basis_domain)
    if len(generators) != basis_domain:
        die("installed local SRS common G frame does not cover its basis_domain")
    h_points = read_srs_frame_prefix(srs_dir, selected.get("H"), "H", 1)
    u_points = read_srs_frame_prefix(srs_dir, selected.get("U"), "U", 1)
    generator_digest = selected.get("generator_digest")
    if not isinstance(generator_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", generator_digest):
        die("installed local SRS default chunk has no canonical generator digest")
    if generator_digest != selected["G"].get("sha256"):
        die("installed local SRS generator digest is not bound to G")
    selected_basis_digest = selected.get("basis_digest")
    if not isinstance(selected_basis_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", selected_basis_digest):
        die("installed local SRS default chunk has no canonical basis digest")
    return generators, h_points[0], u_points[0], generator_digest, selected_basis_digest


def plan_srs_segments(family_domains: dict[str, int], basis_domain: int = NATIVE_BASIS_DOMAIN,
                      materialize_segments: bool = True) -> dict[str, Any]:
    """Plan bounded basis frames without materializing any curve points.

    ``logical_domain`` is the relation's padded coordinate domain.  It is
    deliberately not a frame-size input: every family reuses the bounded
    authenticated prefix basis through ordered logical segments.  The 4 GiB
    guard therefore applies only to the actually materialized frame payload.
    """
    if basis_domain < 1 or basis_domain > MAX_NATIVE_DOMAIN or basis_domain & (basis_domain - 1):
        die(f"SRS basis domain is not a native power of two: {basis_domain}")
    frame_bytes = srs_frame_bytes(basis_domain)
    if not family_domains or "*" not in family_domains:
        die("authenticated SRS family-domain map must contain the common '*' prefix")
    families: dict[str, dict[str, Any]] = {}
    for family_id, logical_domain in sorted(family_domains.items()):
        if (logical_domain < 1 or logical_domain > MAX_NATIVE_DOMAIN or
                logical_domain & (logical_domain - 1)):
            die(f"SRS family {family_id} has an invalid native logical power-of-two domain: {logical_domain}")
        segment_count = (logical_domain + basis_domain - 1) // basis_domain
        if segment_count > 4096 or segment_count > (1 << 32) - 1:
            die(f"SRS family {family_id} segment count exceeds native serialization bounds: {segment_count}")
        families[family_id] = {
            "logical_domain": logical_domain,
            "basis_domain": basis_domain,
            "segment_size": basis_domain,
            "segment_count": segment_count,
            "segments": [
                {
                    "segment_index": index,
                    "logical_offset": index * basis_domain,
                    "actual_length": min(basis_domain, logical_domain - index * basis_domain),
                    "padded_length": basis_domain,
                    "basis_offset": 0,
                    "basis_count": basis_domain,
                }
                for index in range(segment_count)
            ] if materialize_segments else [],
        }
    return {"basis_domain": basis_domain, "frame_bytes": frame_bytes, "families": families}


def table_shape_plan(values: dict[str, str], quant_digest: str) -> dict[str, Any]:
    range_bits = decimal_u64(values["range_bits"], "range_bits")
    if range_bits < 1 or range_bits > 25:
        die(f"range_bits cannot be represented by the native static-table rows: {range_bits}")
    bound = 1 << range_bits
    scale_id = "scale_chain:s_delta=256,s_agg_pre=16,s_agg=16,s_exp=16"
    domain_id = f"range_bits:{values['range_bits']}"
    generator_id = "zkDL_baseline_local_static_table_generator_v1"
    version = "v1"
    row_order = "ascending_index_min_to_max"
    padding_rule = "next_power_of_two_zero_v1"
    lrelu_rows = 2 * bound + 1
    elu_rows = 2 * bound + 1
    # Every ExpMap query is max(LReLU outputs for a destination) - one
    # LReLU output. Its complete legal input domain is therefore the exact
    # output span of this canonical monotone LReLU table, not just `bound`.
    lrelu_output_min = round_away(-bound * 0.2)
    lrelu_output_max = bound
    exp_max = lrelu_output_max - lrelu_output_min
    exp_rows = exp_max + 1
    lrelu_padded_domain = next_pow2(lrelu_rows)
    elu_padded_domain = next_pow2(elu_rows)
    exp_padded_domain = next_pow2(exp_rows)
    if max(lrelu_rows, elu_rows, exp_rows, lrelu_padded_domain, elu_padded_domain, exp_padded_domain) > MAX_STATIC_TABLE_ROWS:
        die("static table row/padded domain exceeds native table safety bound")
    spec = "\n".join((values["static_table_id"], generator_id, version, scale_id, domain_id, row_order,
                      padding_rule, values["range_bits"], str(-bound), str(bound), str(exp_max),
                      str(lrelu_padded_domain), str(elu_padded_domain), str(exp_padded_domain))) + "\n"
    table: dict[str, Any] = {
        "schema": "zkDL_gat_static_table_descriptor_v1",
        "quant_cfg_id": values["quant_cfg_id"],
        "generator_id": generator_id,
        "version": version,
        "scale_id": scale_id,
        "domain_id": domain_id,
        "row_order": row_order,
        "padding_rule": padding_rule,
        "lrelu_padded_domain": lrelu_padded_domain,
        "elu_padded_domain": elu_padded_domain,
        "exp_padded_domain": exp_padded_domain,
        "spec_hash": sha256_bytes(spec.encode()),
        "static_table_id": values["static_table_id"],
        "lrelu_min": -bound,
        "lrelu_max": bound,
        "elu_min": -bound,
        "elu_max": bound,
        "exp_max": exp_max
    }
    return table

def make_tables(values: dict[str, str], quant_digest: str) -> dict[str, Any]:
    table = table_shape_plan(values, quant_digest)
    bound = 1 << decimal_u64(values["range_bits"], "range_bits")
    table["lrelu"] = [x if x >= 0 else round_away(x * 0.2) for x in range(-bound, bound + 1)]
    table["elu"] = [x if x >= 0 else round_away((math.exp(x / 16.0) - 1.0) * 16.0) for x in range(-bound, bound + 1)]
    table["exp"] = [max(1, round_away(math.exp(-x / 256.0) * 16.0)) for x in range(int(table["exp_max"]) + 1)]
    table["digest"] = table_digest(table)
    return table


def sorted_lookup_path(static_table_path: Path) -> Path:
    """Return the sibling descriptor used by the native sorted-lookup plan."""
    name = static_table_path.name
    if name.endswith("_gat_static_tables.json"):
        name = name[:-len("_gat_static_tables.json")] + "_gat_sorted_lookup.json"
    else:
        name = static_table_path.stem + "_sorted_lookup.json"
    return static_table_path.with_name(name)


def sorted_lookup_family_layout(table_ref: str, operation: str, logical_rows: int, input_min: int,
                                input_max: int, output_min: int, output_max: int) -> dict[str, Any]:
    if logical_rows < 1 or input_max < input_min or output_max < output_min:
        die(f"sorted lookup table range is invalid: {table_ref}")
    x_bits = max(1, (logical_rows - 1).bit_length())
    y_bits = max(1, (output_max - output_min).bit_length())
    if x_bits + y_bits > 127:
        die(f"sorted lookup packed key exceeds u128 bound: {table_ref}")
    key_max = ((logical_rows - 1) << y_bits) | (output_max - output_min)
    return {"table_ref": table_ref, "operation": operation, "logical_rows": logical_rows,
            "padded_domain": next_pow2(logical_rows), "sorted_unique": True, "input_min": input_min,
            "input_max": input_max, "output_min": output_min, "output_max": output_max,
            "x_bits": x_bits, "y_bits": y_bits, "key_encoding": "packed_pair_u128",
            "key_max_u128": str(key_max), "padding_rule": "zero_after_logical_rows_v1"}

def sorted_lookup_query_instances(values: dict[str, str], nodes: int,
                                  profile: list[tuple[int, int]], edges: int) -> list[dict[str, Any]]:
    """Mirror RelationPlan's active hidden-stage/head and output-head lookup calls."""
    instances: list[dict[str, Any]] = []

    def add(operation_id: str, operation: str, rows: int, shape: list[int]) -> None:
        instances.append({"operation_id": operation_id, "operation": operation,
                          "query_rows": rows, "query_shape": shape})

    for layer, (head_count, width) in enumerate(profile):
        for head in range(head_count):
            stage = f"hidden/layer{layer}/head{head}"
            add(f"{stage}/leaky_relu", "leaky_relu", edges, [edges])
            add(f"{stage}/exp_map", "exp_map", edges, [edges])
            add(f"{stage}/elu", "elu", nodes * width, [nodes, width])
    output_heads = decimal_u64(values["K_out"], "K_out")
    for head in range(output_heads):
        stage = f"output/head{head}"
        add(f"{stage}/leaky_relu", "leaky_relu", edges, [edges])
        add(f"{stage}/exp_map", "exp_map", edges, [edges])
    return instances

def analytic_sorted_lookup_plan(values: dict[str, str], table: dict[str, Any], nodes: int,
                                profile: list[tuple[int, int]], edges: int) -> dict[str, Any]:
    bound = -int(table["lrelu_min"])
    table_specs = (
        ("static_lrelu", "leaky_relu", 2 * bound + 1, -bound, bound,
         round_away(-bound * 0.2), bound),
        ("static_elu", "elu", 2 * bound + 1, -bound, bound,
         round_away((math.exp(-bound / 16.0) - 1.0) * 16.0), bound),
        ("static_exp", "exp_map", int(table["exp_max"]) + 1, 0, int(table["exp_max"]),
         max(1, round_away(math.exp(-int(table["exp_max"]) / 256.0) * 16.0)), 16),
    )
    families = [sorted_lookup_family_layout(*spec) for spec in table_specs]
    query_instances = sorted_lookup_query_instances(values, nodes, profile, edges)
    table_by_operation = {family["operation"]: family for family in families}
    queries = []
    scratch_bytes_total = 0
    scratch_payload_total = 0
    query_rows_total = 0
    table_domain_total = 0
    query_domain_total = 0
    merge_domain_total = 0
    role_map = ("table_key,query_x,query_y,query_key,query_prefix,merge_key,merge_delta,"
                "merge_marker,merge_inverse,merge_prefix,merge_bits,merge_delta_bits,query_bits")
    for instance in query_instances:
        operation_id = str(instance["operation_id"])
        operation = str(instance["operation"])
        rows = int(instance["query_rows"])
        shape = instance["query_shape"]
        if rows < 1 or rows > MAX_NATIVE_DOMAIN:
            die(f"sorted lookup query count exceeds native serialization bounds: {operation_id}")
        table_family = table_by_operation[operation]
        table_rows = int(table_family["logical_rows"])
        if table_rows > MAX_NATIVE_DOMAIN or rows > MAX_NATIVE_DOMAIN - table_rows:
            die(f"sorted lookup merge row count exceeds native domain: {operation_id}")
        merge_rows = rows + table_rows
        table_domain = next_pow2(table_rows)
        query_domain = next_pow2(rows)
        merge_domain = next_pow2(merge_rows)
        segment_size = min(NATIVE_BASIS_DOMAIN, table_domain, query_domain, merge_domain)
        if any(domain % segment_size != 0 for domain in (table_domain, query_domain, merge_domain)):
            die(f"sorted lookup segment layout is not power-of-two aligned: {operation_id}")
        scratch_domains = table_domain + query_domain + merge_domain
        if scratch_domains > ((1 << 64) - 1) // 107:
            die(f"sorted lookup scratch capacity overflows u64: {operation_id}")
        record_payload_bytes = scratch_domains * 107
        key_max_u128 = str(table_family["key_max_u128"])
        header_bytes = (124 + len(operation_id.encode("utf-8")) + len(role_map.encode("utf-8")) +
                        len(key_max_u128.encode("ascii")))
        scratch_file_bytes = record_payload_bytes + header_bytes
        if scratch_bytes_total > (1 << 64) - 1 - scratch_file_bytes:
            die("aggregate sorted lookup scratch capacity overflows u64")
        if scratch_payload_total > (1 << 64) - 1 - record_payload_bytes:
            die("aggregate sorted lookup record payload overflows u64")
        scratch_bytes_total += scratch_file_bytes
        scratch_payload_total += record_payload_bytes
        query_rows_total += rows
        table_domain_total += table_domain
        query_domain_total += query_domain
        merge_domain_total += merge_domain
        queries.append({"operation_id": operation_id, "operation": operation,
                        "table_ref": table_family["table_ref"], "table_rows": table_rows,
                        "query_rows": rows, "query_shape": shape,
                        "table_domain": table_domain, "query_domain": query_domain,
                        "merge_rows": merge_rows, "merge_domain": merge_domain,
                        "segment_size": segment_size,
                        "table_segment_count": table_domain // segment_size,
                        "query_segment_count": query_domain // segment_size,
                        "merge_segment_count": merge_domain // segment_size,
                        "x_bits": table_family["x_bits"],
                        "y_bits": table_family["y_bits"],
                        "key_max_u128": key_max_u128,
                        "scratch_header_bytes": header_bytes,
                        "scratch_record_payload_bytes": record_payload_bytes,
                        "scratch_bytes_upper_bound": scratch_file_bytes})
    return {"segment_size": NATIVE_BASIS_DOMAIN, "table_families": families,
            "query_families": queries, "query_count": len(queries),
            "query_rows_total": query_rows_total,
            "table_domain_total": table_domain_total,
            "query_domain_total": query_domain_total,
            "merge_domain_total": merge_domain_total,
            "scratch_record_payload_bytes_total": scratch_payload_total,
            "scratch_bytes_total": scratch_bytes_total,
            "scratch_bytes_max_single_file": max(
                (int(item["scratch_bytes_upper_bound"]) for item in queries), default=0)}

def make_sorted_lookup_descriptor(values: dict[str, str], table: dict[str, Any],
                                  static_table_digest: str, nodes: int,
                                  profile: list[tuple[int, int]], edges: int) -> dict[str, Any]:
    """Describe logical unique table rows and bounded Q/T/S layouts.

    The descriptor intentionally contains no chi tree or padded row material.
    Padding is a native provider rule, while table rows remain public, sorted,
    and unique by their packed (input-offset, output-offset) key.
    """
    segment_size = NATIVE_BASIS_DOMAIN
    table_specs = (
        ("static_lrelu", table["lrelu"], int(table["lrelu_min"]), "leaky_relu"),
        ("static_elu", table["elu"], int(table["elu_min"]), "elu"),
        ("static_exp", table["exp"], 0, "exp_map"),
    )
    families: list[dict[str, Any]] = []
    for table_ref, outputs, input_min, operation in table_specs:
        if not outputs:
            die(f"sorted lookup table is empty: {table_ref}")
        input_max = input_min + len(outputs) - 1
        families.append(sorted_lookup_family_layout(table_ref, operation, len(outputs), input_min, input_max,
                                                    min(int(value) for value in outputs), max(int(value) for value in outputs)))
    descriptor: dict[str, Any] = {
        "schema": "zkDL_gat_sorted_lookup_descriptor_v1",
        "version": 1,
        "static_table_id": values["static_table_id"],
        "static_table_digest": static_table_digest,
        "key_encoding": "packed_pair_u128",
        "active_mask_rule": "prefix_one_then_zero_v1",
        "sorted_delta_rule": "nonnegative_bounded_delta_v1",
        "marker_rule": "group_first_v1",
        "inverse_rule": "delta_inverse_or_zero_v1",
        "segment_size": segment_size,
        "table_families": families,
        "query_families": sorted_lookup_query_instances(values, nodes, profile, edges),
        "source_root": "baseline-local",
    }
    descriptor["descriptor_digest"] = sha256_bytes(compact(descriptor))
    return descriptor


def ledger_bounds_plan(values: dict[str, str], degree: int, model_max: int) -> tuple[str, str, list[dict[str, Any]], dict[str, int]]:
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
        families.append({
            "family_id": fid,
            "layer_index": layer,
            "head_index": head,
            "output": output,
            "max_input_abs": input_bound,
            "max_weight_abs": max(1, weight),
            "max_attention_abs": max(1, attention),
            "max_bias_abs": max(0, bias),
            "max_score_abs": score,
            "max_table_output_abs": table_out,
            "max_delta": delta,
            "max_exp": exp,
            "max_sum": summation,
            "max_aggregate_num": aggregate,
            "max_dividend_abs": dividend,
            "max_aux_abs": aux,
            "max_divisor": divisor,
            "max_field_product_abs": max(1, exp * input_bound * max(1, weight))
        })
    current = max(1, 1 << min(decimal_u64(values["range_bits"], "range_bits"), 20))
    for layer, (head_count, _head_dim) in enumerate(parse_profile(values["hidden_profile"])):
        for head in range(head_count):
            family(f"hidden/layer{layer}/head{head}", layer, head, 0, current, bound, bound, 0)
    for head in range(decimal_u64(values["K_out"], "K_out")):
        family(f"output/head{head}", 2, head, 1, current, bound, bound, bound)
    fields = ("max_score_abs", "max_table_output_abs", "max_delta", "max_exp", "max_sum", "max_aggregate_num", "max_dividend_abs", "max_aux_abs", "max_divisor")
    global_values = {key: max(int(item[key]) for item in families) for key in fields}
    return ledger_id, degree_id, families, global_values


def make_ledger(root: Path, values: dict[str, str], tables: dict[str, Any], table_file_digest: str,
                quant_digest: str, degree: int | None = None, model_max: int | None = None) -> dict[str, Any]:
    if degree is None:
        _, _, degree = graph_degree(root, values["dataset"], values)
    if model_max is None:
        tensors = load_tensors(root / values["checkpoint_bundle"] / "tensors.txt")
        model_max = max((abs(round_away(value * (16 if name.endswith("output_bias") else 4))) for name, _, data in tensors for value in data), default=1)
    ledger_id, degree_id, families, global_values = ledger_bounds_plan(values, degree, model_max)
    ledger = {
        "schema": "zkDL_gat_integer_bound_ledger_v1",
        "ledger_id": ledger_id,
        "degree_bound_id": degree_id,
        "quant_cfg_id": values["quant_cfg_id"],
        "static_table_id": values["static_table_id"],
        "field_modulus": str(R),
        "scale_chain_digest": quant_digest,
        "static_table_descriptor_digest": table_file_digest,
        **global_values,
        "families": families
    }
    ledger["digest"] = ledger_digest(ledger)
    return ledger


def audit_formal_prerequisites(cfg_path: Path) -> dict[str, Any]:
    """Read-only, analytic audit for one canonical formal dataset configuration.

    This intentionally shares graph normalization and all resource shape/domain
    planners with generation, but does not create table rows, SRS points,
    descriptors, registries, witnesses, or snapshots.
    """
    cfg_path = Path(cfg_path)
    if cfg_path.is_symlink() or not cfg_path.is_file():
        die(f"formal prerequisite config is missing or symlinked: {cfg_path}")
    config_path = cfg_path.resolve()
    root = config_path.parent.parent.resolve()
    if config_path.parent.name != "configs" or root.name != "zkdl":
        die("formal prerequisite config must be under comparison/zkdl/configs")
    values = parse_config(config_path)
    dataset = values.get("dataset")
    if dataset not in DATASETS:
        die(f"unsupported formal prerequisite dataset in {config_path}")
    if values.get("auto_prepare_dataset") != "false" or values.get("dump_trace") != "false":
        die(f"formal input/trace policy mismatch: {dataset}")
    if values.get("batching_rule") != ("multi_graph_batch" if dataset == "ppi" else "whole_graph_single"):
        die(f"formal batching contract mismatch: {dataset}")
    if values.get("self_loop_rule") != "per_node" or values.get("edge_sort_rule") != "edge_gid_then_dst_stable":
        die(f"formal self-loop/edge-order contract mismatch: {dataset}")

    graph = audit_graph(root, dataset, values)
    metadata = graph_metadata(root, dataset, values)
    try:
        features = metadata["num_features"]
        classes = metadata["num_classes"]
        configured_classes = decimal_u64(values["num_classes"], "num_classes")
        configured_graphs = decimal_u64(values.get("batch_graphs", str(graph["graphs"])), "batch_graphs")
        layers = decimal_u64(values["L"], "L")
        output_heads = decimal_u64(values["K_out"], "K_out")
    except (KeyError, ValueError) as exc:
        die(f"formal graph/model dimensions are incomplete for {dataset}: {exc}")
    profile = parse_profile(values["hidden_profile"])
    input_dims = parse_list(values["d_in_profile"])
    if (features < 1 or classes < 1 or configured_classes != classes or input_dims[0] != features or
            len(input_dims) != len(profile) or layers != len(profile) + 1 or output_heads < 1 or
            configured_graphs != int(graph["graphs"])):
        die(f"formal graph/model shapes or graph batching count mismatch: {dataset}")
    for layer in range(1, len(profile)):
        if input_dims[layer] != profile[layer - 1][0] * profile[layer - 1][1]:
            die(f"formal hidden-layer input width does not match previous concat shape: {dataset}/layer{layer}")

    quant_path = root / "configs" / "quantization_scale_chain.cfg"
    if quant_path.is_symlink() or not quant_path.is_file():
        die(f"quantization scale-chain file is missing or symlinked: {quant_path}")
    quant = validated_scale_chain(quant_path, values.get("quant_cfg_id", ""))
    quant_digest = sha256_file(quant_path)

    checkpoint_path = Path(values["checkpoint_bundle"])
    if checkpoint_path.is_absolute() or ".." in checkpoint_path.parts:
        die(f"checkpoint bundle must be baseline-local: {values['checkpoint_bundle']}")
    tensor_export = root / checkpoint_path / "tensors.txt"
    manifest_path = root / checkpoint_path / "manifest.json"
    for path in (tensor_export, manifest_path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            die(f"formal checkpoint prerequisite is missing, empty or symlinked: {path}")
    tensor_plan = audit_tensor_export(tensor_export, values)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        die(f"formal checkpoint manifest is invalid: {exc}")
    if (manifest.get("dataset") != dataset or manifest.get("L") != layers or manifest.get("K_out") != output_heads or
            manifest.get("C") != classes or manifest.get("tensor_count") != tensor_plan["tensor_count"] or
            manifest.get("d_in_profile") != input_dims):
        die(f"formal checkpoint manifest shape/count mismatch: {dataset}")
    tensor_index = manifest.get("tensor_index")
    if not isinstance(tensor_index, dict) or set(tensor_index) != set(tensor_plan["model_shapes"]):
        die(f"formal checkpoint manifest tensor_index keys differ from canonical exported tensors: {dataset}")
    for name, expected_shape in tensor_plan["model_shapes"].items():
        descriptor = tensor_index.get(name)
        shape = descriptor.get("shape") if isinstance(descriptor, dict) else None
        if (not isinstance(shape, list) or not shape or
                any(type(axis) is not int or axis < 1 for axis in shape) or shape != expected_shape):
            die(f"formal checkpoint manifest tensor shape differs from canonical export: {dataset}/{name}")

    table = table_shape_plan(values, quant_digest)
    ledger_id, degree_id, ledger_families, ledger_global = ledger_bounds_plan(
        values, int(graph["csr_max_in_degree"]), int(tensor_plan["max_quantized_model_abs"]),
    )
    family_domains = relation_family_domains(
        values, int(graph["nodes"]), int(graph["normalized_edges"]), table,
        tensor_plan["model_padded_domains"], graph_count=int(graph["graphs"]),
        ledger_families=ledger_families, quantization=quant,
    )
    lookup_plan = analytic_sorted_lookup_plan(
        values, table, int(graph["nodes"]), profile, int(graph["normalized_edges"]),
    )
    srs_plan = plan_srs_segments(family_domains, materialize_segments=False)
    max_domain = max(family_domains.values(), default=1)
    max_families = sorted(name for name, domain in family_domains.items() if domain == max_domain)
    max_segment_count = max((int(item["segment_count"]) for item in srs_plan["families"].values()), default=0)
    srs_bytes = ((SRS_FRAME_HEADER_BYTES + G1_POINT_BYTES * int(srs_plan["basis_domain"])) +
                 2 * (SRS_FRAME_HEADER_BYTES + G1_POINT_BYTES) + len(family_domains) * 4096)
    try:
        statvfs = os.statvfs(root)
        free_bytes = statvfs.f_bavail * statvfs.f_frsize
    except OSError as exc:
        die(f"cannot inspect formal SRS filesystem capacity: {exc}")
    if srs_bytes > free_bytes:
        die(f"formal SRS plan needs {srs_bytes} bytes but only {free_bytes} are available")

    all_ledger_bounds = list(ledger_global.values()) + [
        int(item[key]) for item in ledger_families
        for key in ("max_input_abs", "max_weight_abs", "max_attention_abs", "max_bias_abs", "max_score_abs",
                    "max_table_output_abs", "max_delta", "max_exp", "max_sum", "max_aggregate_num",
                    "max_dividend_abs", "max_aux_abs", "max_divisor", "max_field_product_abs")
    ]
    if any(value < 0 or value > (1 << 63) - 1 or value >= R // 2 for value in all_ledger_bounds):
        die(f"integer-bound ledger exceeds signed native/field range: {dataset}")

    return {
        "schema": "zkDL_gat_formal_prerequisites_v1",
        "dataset": dataset,
        "config_sha256": sha256_file(config_path),
        "graph": graph,
        "input_shape": {"nodes": int(graph["nodes"]), "features": features, "classes": classes,
                        "graphs": int(graph["graphs"]), "hidden_profile": [list(item) for item in profile],
                        "input_dims": input_dims, "output_heads": output_heads},
        "checkpoint": tensor_plan | {"manifest_sha256": sha256_file(manifest_path)},
        "table_plan": {key: table[key] for key in ("lrelu_padded_domain", "elu_padded_domain", "exp_padded_domain",
                                                        "lrelu_min", "lrelu_max", "elu_min", "elu_max", "exp_max", "spec_hash")},
        "sorted_lookup_plan": lookup_plan,
        "family_domain_count": len(family_domains),
        "family_domain_sha256": sha256_bytes(compact(dict(sorted(family_domains.items())))),
        "family_domains": dict(sorted(family_domains.items())),
        "max_family_domain": max_domain,
        "max_family_domain_ids": max_families,
        "srs_plan": {"basis_domain": srs_plan["basis_domain"], "frame_bytes": srs_plan["frame_bytes"],
                     "families": {name: {"logical_domain": int(item["logical_domain"]),
                                          "segment_size": int(item["segment_size"]),
                                          "segment_count": int(item["segment_count"])}
                                  for name, item in sorted(srs_plan["families"].items())},
                     "max_segment_count": max_segment_count, "estimated_materialized_bytes": srs_bytes,
                     "available_bytes": free_bytes},
        "ledger_plan": {"ledger_id": ledger_id, "degree_bound_id": degree_id,
                        "maxima": ledger_global,
                        "max_family_aux_abs": max(int(item["max_aux_abs"]) for item in ledger_families),
                        "max_field_product_abs": max(int(item["max_field_product_abs"]) for item in ledger_families)},
        "quantization_scale_chain_sha256": quant_digest,
    }


def write_srs(root: Path, values: dict[str, str], family_domains: dict[str, int]) -> tuple[str, str]:
    srs_dir = root / values["srs_path"]
    srs_dir.parent.mkdir(parents=True, exist_ok=True)
    if srs_dir.is_symlink():
        die(f"local SRS destination is symlinked: {srs_dir}")

    segment_plan = plan_srs_segments(family_domains)
    common_domain = int(segment_plan["basis_domain"])

    estimated_bytes = (
        (SRS_FRAME_HEADER_BYTES + G1_POINT_BYTES * common_domain) +
        2 * (SRS_FRAME_HEADER_BYTES + G1_POINT_BYTES) +
        len(family_domains) * 4096
    )
    try:
        statvfs = os.statvfs(srs_dir.parent)
        free_bytes = statvfs.f_bavail * statvfs.f_frsize
    except OSError as exc:
        die(f"cannot inspect SRS filesystem capacity: {exc}")
    if estimated_bytes > free_bytes:
        die(f"SRS materialization needs {estimated_bytes} bytes but only {free_bytes} are available")
    domain = common_domain
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{srs_dir.name}.tmp-", dir=srs_dir.parent))
    helper = native_binary(root, "gat_resource_setup")
    if helper.is_symlink() or not helper.is_file():
        shutil.rmtree(temp_dir)
        die("matched baseline-local gat_resource_setup is missing; run the formal build entry")
    def frame_metadata(label: str, count: int) -> dict[str, Any]:
        relative = Path("wildcard") / f"{label}.chunk0.bin"
        path = temp_dir / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != srs_frame_bytes(count):
            die(f"native SRS helper produced an invalid {label} frame")
        with path.open("rb") as stream:
            header = stream.read(SRS_FRAME_HEADER_BYTES)
        if header != b"GATZKG1C" + struct.pack("<IIQQQ", 1, G1_POINT_BYTES, count, domain, domain - 1):
            die(f"native SRS helper produced an invalid {label} header")
        return {"path": relative.as_posix(), "count": count, "bytes": srs_frame_bytes(count), "sha256": sha256_file(path)}
    try:
        print(f"native_srs_setup=START generators={domain} output={srs_dir}", flush=True)
        subprocess.run([str(helper), "generate-srs", "--output", str(temp_dir), "--baseline-root", str(root)], cwd=root, check=True)
        g_meta, h_meta, u_meta = frame_metadata("G", domain), frame_metadata("H", 1), frame_metadata("U", 1)
        basis = basis_digest(g_meta, h_meta, u_meta)
        generator_id = "zkDL_native_local_csprng_g1_v1"
        families: list[dict[str, Any]] = []
        for family_id, required_domain in sorted(family_domains.items()):
            family_plan = segment_plan["families"][family_id]
            segment_count = int(family_plan["segment_count"])
            chunk = {
                "family_id": family_id, "chunk_id": f"{family_id}:chunk0",
                "chunk_index": 0, "point_offset": 0, "point_count": domain,
                "domain": domain, "logical_domain": required_domain,
                "basis_domain": common_domain, "segment_size": common_domain,
                "segment_count": segment_count, "required_domain": required_domain,
                "degree": domain - 1, "point_size": 48,
                "generator_id": generator_id, "basis_digest": basis,
                "generator_digest": g_meta["sha256"], "G": dict(g_meta),
                "H": dict(h_meta), "U": dict(u_meta),
            }
            chunk["segments"] = list(family_plan["segments"])
            families.append({"family_id": family_id, "chunks": [chunk]})
        default = next(item for item in families if item["family_id"] == "*")["chunks"][0]
        manifest = {
            "manifest_format": "zkgat_srs_manifest", "manifest_version": 2,
            "curve_id": "bls12_381", "encoding": "GATZKG1C",
            "srs_id": values["srs_id"], "chunking_rule": "per_family_prefix_v1",
            "frame_sharing_rule": "authenticated_common_prefix_v1",
            "default_family_id": "*", "default_chunk_id": "*:chunk0",
            "generator_id": generator_id, "basis_digest": basis,
            "setup": {
                "schema": "zkDL_gat_local_random_setup_v1",
                "mode": "native_local_ephemeral_random",
                "trust_status": "native_local_ephemeral_random",
                "algorithm": "os_csprng_nonzero_scalar_mul_bls12_381_g1",
                "spec": "zkDL_Hyrax_IPA_Pedersen_independent_g1_basis_v1",
                "curve_id": "bls12_381",
                "config_binding_digest": config_binding(values),
                "basis_digest": basis, "domain": domain,
                "basis_domain": common_domain,
                "generator_count": common_domain, "h_count": 1, "u_count": 1,
                "logical_domain": max(family_domains.values()),
                "segment_size": common_domain,
                "segment_count": max((int(v) + common_domain - 1) // common_domain
                                     for v in family_domains.values()),
            },
            "families": families, "generator_digest": default["generator_digest"],
        }
        manifest_digest = write_json(temp_dir / "srs_manifest.json", manifest)
        backup: Path | None = None
        if srs_dir.exists():
            backup = srs_dir.with_name(f".{srs_dir.name}.old-{os.getpid()}")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(srs_dir, backup)
        try:
            os.replace(temp_dir, srs_dir)
        except OSError:
            if backup is not None and not srs_dir.exists():
                os.replace(backup, srs_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return manifest_digest, default["generator_digest"]
    except BaseException:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def complete_resources_valid(root: Path, config_path: Path) -> bool:
    validator = Path(__file__).with_name("validate_resources.py")
    result = subprocess.run(
        [sys.executable, str(validator), "--config", str(config_path)],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def existing_blinders(root: Path, values: dict[str, str], tensor_ids: set[str]) -> dict[str, int] | None:
    path = root / values["model_blinder_sidecar_path"]
    expected = values.get("model_blinder_sidecar_digest", "")
    if (path.is_symlink() or not path.is_file() or expected == "UNAVAILABLE" or
            not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(path) != expected):
        return None
    if (path.stat().st_mode & 0o077) or path.stat().st_mode & 0o600 != 0o600:
        return None
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(sidecar, dict) or sidecar.get("schema") != "zkDL_gat_model_blinder_sidecar_v1":
        return None
    if sidecar.get("registry_digest") != values.get("verifier_registry_digest"):
        return None
    entries = sidecar.get("entries")
    if not isinstance(entries, list) or len(entries) != len(tensor_ids):
        return None
    result: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("tensor_id"), str):
            return None
        tensor_id = entry["tensor_id"]
        raw = entry.get("blinder")
        if tensor_id not in tensor_ids or tensor_id in result or not isinstance(raw, str) or not re.fullmatch(r"[0-9a-f]{64}", raw):
            return None
        scalar = int.from_bytes(bytes.fromhex(raw), "little")
        if scalar <= 0 or scalar >= R:
            return None
        result[tensor_id] = scalar
    return result if set(result) == tensor_ids else None


def generate(root: Path, config_path: Path, full: bool, regenerate_srs: bool,
             admission_only: bool = False) -> str:
    values = parse_config(config_path)
    dataset = values.get("dataset")
    if dataset not in DATASETS:
        die(f"unsupported dataset in {config_path}")
    quant_path = root / "configs" / "quantization_scale_chain.cfg"
    quant = validated_scale_chain(quant_path, values.get("quant_cfg_id", ""))
    nodes, edges, degree = graph_degree(root, dataset, values)
    degree_id = f"zkdl_degree_v2_{dataset}_{degree}"
    if not full:
        die("formal setup requires --full-dataset")
    quant_digest = sha256_file(quant_path)
    values = parse_config(config_path)
    table = make_tables(values, quant_digest)
    table_path = root / values["static_table_descriptor"]
    table_file_digest = write_json(table_path, table)
    lookup_path = sorted_lookup_path(table_path)
    lookup_descriptor = make_sorted_lookup_descriptor(
        values, table, table_file_digest, nodes, parse_profile(values["hidden_profile"]), edges)
    write_json(lookup_path, lookup_descriptor)
    ledger = make_ledger(root, values, table, table_file_digest, quant_digest)
    ledger_path = root / values["integer_bound_ledger"]
    ledger_file_digest = write_json(ledger_path, ledger)
    ledger_checker = native_binary(root, "gat_resource_setup")
    subprocess.run([str(ledger_checker), "verify-ledger", "--path", str(ledger_path), "--baseline-root", str(root)],
                   cwd=root, check=True)
    model_domains = [math.prod([next_pow2(int(axis)) for axis in (shape[1:] if len(shape) == 3 and shape[0] == 1 else shape)])
                     for _, shape, _ in load_tensors(root / values["checkpoint_bundle"] / "tensors.txt")]
    metadata = graph_metadata(root, dataset, values)
    family_domains = relation_family_domains(
        values, nodes, edges, table, model_domains,
        graph_count=metadata.get("graph_count", max(1, decimal_u64(values.get("batch_graphs", "1"), "batch_graphs"))),
        ledger_families=ledger["families"], quantization=quant,
    )
    update_config(config_path, {"degree_bound_id": degree_id})
    values = parse_config(config_path)
    if admission_only:
        checkpoint_manifest_path = root / values["checkpoint_bundle"] / "manifest.json"
        checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        checkpoint_manifest["degree_bound_id"] = ledger["degree_bound_id"]
        write_json(checkpoint_manifest_path, checkpoint_manifest)
        update_config(config_path, {
            "srs_manifest_digest": "UNAVAILABLE",
            "verifier_registry_digest": "UNAVAILABLE",
            "model_blinder_sidecar_digest": "UNAVAILABLE",
        })
        return "ADMISSION_ONLY"
    resources_valid = complete_resources_valid(root, config_path)
    if not regenerate_srs and resources_valid:
        return "REUSED"
    srs_path = root / values["srs_path"]

    if regenerate_srs or not srs_path.is_dir() or not resources_valid:
        srs_digest, generator_digest = write_srs(root, values, family_domains)
    else:
        srs_digest = sha256_file(srs_path / "srs_manifest.json")
        generator_digest = ""
    update_config(config_path, {"srs_manifest_digest": srs_digest})
    values = parse_config(config_path)
    tensors = load_tensors(root / values["checkpoint_bundle"] / "tensors.txt")
    tensor_ids = set()
    for name, _, _ in tensors:
        mapped = name.replace("/seq/kernel", "/model/seq_kernel").replace(
            "/attn_src/kernel", "/model/attn_src"
        ).replace("/attn_dst/kernel", "/model/attn_dst").replace("/output_bias", "/model/output_bias")
        if "/model/" in mapped:
            tensor_ids.add(mapped)
    old_blinders = existing_blinders(root, values, tensor_ids)
    blinders = old_blinders or {tensor_id: fresh_scalar() for tensor_id in tensor_ids}
    generators, h_point, u_point, generator_digest, _srs_basis_digest = load_srs_basis(
        root, values, max(model_domains, default=1))
    manifest_path = root / values["checkpoint_bundle"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["degree_bound_id"] = ledger["degree_bound_id"]
    write_json(manifest_path, manifest)
    registry_entries = model_entries(
        root, values, h_point, u_point, blinders, generators, srs_digest)
    registry = {
        "schema": "zkDL_gat_model_registry_v3",
        "dataset": dataset,
        "model_arch_id": values["model_arch_id"],
        "model_param_id": values["model_param_id"],
        "quant_cfg_id": values["quant_cfg_id"],
        "static_table_id": values["static_table_id"],
        "srs_id": values["srs_id"],
        "srs_manifest_digest": srs_digest,
        "generator_digest": generator_digest,
        "config_digest": config_identity(config_path),
        "scale_chain_digest": quant_digest,
        "static_table_digest": table_file_digest,
        "integer_bound_ledger_digest": ledger_file_digest,
        "model_manifest_digest": sha256_file(root / values["checkpoint_bundle"] / "manifest.json"),
        "model_tensors": registry_entries,
        "source_root": "baseline-local"
    }
    registry["registry_identity_digest"] = sha256_bytes(compact(registry))
    registry["verifier_registry_digest"] = registry["registry_identity_digest"]
    registry_path = root / values["model_registry_path"]
    registry_digest = write_json(registry_path, registry)
    sidecar_entries = [{"tensor_id": item["tensor_id"], "blinder": blinders[item["tensor_id"]].to_bytes(32, "little").hex()} for item in registry_entries]
    sidecar = {"schema": "zkDL_gat_model_blinder_sidecar_v1", "registry_digest": registry_digest, "entries": sidecar_entries, "source_root": "baseline-local"}
    sidecar_path = root / values["model_blinder_sidecar_path"]
    sidecar_digest = write_json_atomic(sidecar_path, sidecar, 0o600)
    update_config(config_path, {"degree_bound_id": ledger["degree_bound_id"], "srs_manifest_digest": srs_digest,
                  "verifier_registry_digest": registry_digest, "model_blinder_sidecar_digest": sidecar_digest})
    return "MATERIALIZED" if old_blinders is None else "MATERIALIZED_REUSING_BLINDERS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--full-dataset", action="store_true")
    parser.add_argument("--admission-only", action="store_true",
                        help="materialize/authenticate ledgers and relation tables without SRS or registry setup")
    parser.add_argument("--regenerate-srs", action="store_true",
                        help="explicitly replace the local random G1 basis")
    args = parser.parse_args()
    config = args.config.resolve()
    root = config.parent.parent.resolve()
    if config.parent.name != "configs" or root.name != "zkdl":
        die("config must be under comparison/zkdl/configs")
    status = generate(root, config, args.full_dataset, args.regenerate_srs, args.admission_only)
    print(json.dumps({"schema": "zkDL_gat_local_resource_setup_v1",
                      "config": config.relative_to(root).as_posix(),
                      "status": status}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
