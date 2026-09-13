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
    return 1 << (value - 1).bit_length()


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


def graph_degree(root: Path, dataset: str) -> tuple[int, int, int]:
    meta: dict[str, str] = {}
    meta_path = root / "data" / "cache" / dataset / "meta.cfg"
    if meta_path.is_file():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = (item.strip() for item in line.split("=", 1))
                meta[key] = value
    nodes = int(meta.get("num_nodes", "1"))
    edges = int(meta.get("num_edges", "1"))
    degree = int(meta.get("max_in_degree", "0"))
    edge_path = root / "data" / "cache" / dataset / "edges.txt"
    if edge_path.is_file() and degree == 0:
        counts: dict[int, int] = {}
        for line in edge_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                dst = int(parts[1])
                counts[dst] = counts.get(dst, 0) + 1
        if counts:
            degree = max(counts.values())
            edges = max(edges, sum(counts.values()))
    return nodes, edges, max(1, degree)


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
    spec = "\n".join((values["static_table_id"], generator_id, version, scale_id, domain_id, row_order, values["range_bits"], str(-bound), str(bound), str(bound))) + "\n"
    table: dict[str, Any] = {"schema": "zkDL_gat_static_table_descriptor_v1", "quant_cfg_id": values["quant_cfg_id"], "generator_id": generator_id, "version": version, "scale_id": scale_id, "domain_id": domain_id, "row_order": row_order, "spec_hash": sha256_bytes(spec.encode()), "static_table_id": values["static_table_id"], "lrelu_min": -bound, "lrelu_max": bound, "elu_min": -bound, "elu_max": bound, "exp_max": bound}
    table["lrelu"] = [x if x >= 0 else round_away(x * 0.2) for x in range(-bound, bound + 1)]
    table["elu"] = [x if x >= 0 else round_away((math.exp(x / 16.0) - 1.0) * 16.0) for x in range(-bound, bound + 1)]
    table["exp"] = [max(1, round_away(math.exp(-x / 256.0) * 16.0)) for x in range(bound + 1)]
    table["digest"] = table_digest(table)
    return table


def make_ledger(root: Path, values: dict[str, str], tables: dict[str, Any], table_file_digest: str, quant_digest: str) -> dict[str, Any]:
    _, _, degree = graph_degree(root, values["dataset"])
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
    tau = derive_scalar(seed, b"tau:" + values["dataset"].encode())
    h_scalar = derive_scalar(seed, b"H:" + values["dataset"].encode())
    u_scalar = derive_scalar(seed, b"U:" + values["dataset"].encode())
    domain = max(family_domains.values())
    points: list[bytes] = []
    current = G
    tau_point = point_mul(G, tau)
    for _ in range(domain):
        points.append(encode_g1(current))
        current = point_add(current, tau_point)
    h_point = point_mul(G, h_scalar)
    u_point = point_mul(G, u_scalar)
    def frame(family: str, label: str, payload: bytes, count: int, chunk_domain: int) -> dict[str, Any]:
        safe_family = family.replace("*", "wildcard").replace("/", "_")
        relative = Path(safe_family) / f"{label}.chunk0.bin"
        data = b"GATZKG1C" + struct.pack("<IIQQQ", 1, 48, count, chunk_domain, chunk_domain - 1) + payload
        (srs_dir / relative).parent.mkdir(parents=True, exist_ok=True)
        (srs_dir / relative).write_bytes(data)
        return {"path": relative.as_posix(), "count": count, "sha256": sha256_bytes(data)}
    families: list[dict[str, Any]] = []
    for family_id, chunk_domain in sorted(family_domains.items()):
        g_meta = frame(family_id, "G", b"".join(points[:chunk_domain]), chunk_domain, chunk_domain)
        h_meta = frame(family_id, "H", encode_g1(h_point), 1, chunk_domain)
        u_meta = frame(family_id, "U", encode_g1(u_point), 1, chunk_domain)
        chunk = {"family_id": family_id, "chunk_id": f"{family_id}:chunk0", "chunk_index": 0, "point_offset": 0, "point_count": chunk_domain, "domain": chunk_domain, "degree": chunk_domain - 1, "point_size": 48, "generator_id": "zkDL_explicit_ceremony_g1_v2", "ceremony_transcript_sha256": transcript_digest, "ceremony_seed_sha256": seed_digest, "generator_digest": g_meta["sha256"], "G": g_meta, "H": h_meta, "U": u_meta}
        families.append({"family_id": family_id, "chunks": [chunk]})
    default = next(item for item in families if item["family_id"] == "*")["chunks"][0]
    manifest = {"manifest_format": "zkgat_srs_manifest", "manifest_version": 2, "curve_id": "bls12_381", "encoding": "GATZKG1C", "srs_id": values["srs_id"], "chunking_rule": "per_family_prefix_v1", "default_family_id": "*", "default_chunk_id": "*:chunk0", "generator_id": "zkDL_explicit_ceremony_g1_v2", "ceremony_transcript_sha256": transcript_digest, "ceremony_seed_sha256": seed_digest, "setup": {"mode": "explicit_ceremony_input", "trust_status": "not_an_external_trust_root", "seed_sha256": seed_digest, "transcript_sha256": transcript_digest}, "families": families, "generator_digest": default["generator_digest"]}
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


def generate(root: Path, config_path: Path, seed: bytes, seed_digest: str, transcript_digest: str, full: bool) -> None:
    values = parse_config(config_path)
    dataset = values.get("dataset")
    if dataset not in DATASETS:
        die(f"unsupported dataset in {config_path}")
    nodes, edges, degree = graph_degree(root, dataset)
    degree_id = f"zkdl_degree_v2_{dataset}_{degree}"
    if dataset != "cora" and not full:
        result = write_plan(root, values, degree_id)
        update_config(config_path, {"degree_bound_id": result["degree_bound_id"], "srs_manifest_digest": result["srs_manifest_digest"], "static_table_descriptor": values["static_table_descriptor"], "integer_bound_ledger": values["integer_bound_ledger"], "model_registry_path": values["model_registry_path"], "verifier_registry_digest": result["verifier_registry_digest"], "model_blinder_sidecar_digest": result["model_blinder_sidecar_digest"]})
        return
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
    domain = next_pow2(max(nodes, edges, max(model_domains, default=1)))
    if domain > (1 << 19):
        die(f"{dataset} requires SRS domain {domain}; use --full-dataset only with approved capacity")
    family_domains = {"*": domain, "model": max(model_domains, default=1), "node": next_pow2(nodes), "edge": next_pow2(edges), "lookup": next_pow2(max(nodes, edges)), "relation": domain}
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
    generate(root, config, seed, seed_digest, transcript_digest, args.full_dataset)
    print(json.dumps({"schema": "zkDL_gat_resource_generation_v1", "config": config.relative_to(root).as_posix(), "seed_sha256": seed_digest, "transcript_sha256": transcript_digest, "status": "GENERATED"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
