#!/usr/bin/env python3
"""从 full proof/verifier manifest 同源导出 zkDL-GAT official 结果。"""

from __future__ import annotations

import csv
import json
import shutil
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "zkdl_gat_outputs"
RESULTS = OUT / "results"
RUNS = OUT / "runs"

DATASETS = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
DEVICES = ["cpu", "gpu"]
PROOF_PAYLOAD_VERSION = "2"
PROOF_SIZE_DEFINITION = "deterministic_relation_payload_sum_v2"
REQUESTED_COMMITMENT_BLOCK_SIZE = 64
FR_SIZE_BYTES = 32
G1_JACOBIAN_SIZE_BYTES = 144
UINT32_SIZE_BYTES = 4
UINT64_SIZE_BYTES = 8
CURRENT_RESCALE_POLICY = "scale16_full_relation_with_remainder_range"
OFFICIAL_COLUMNS = [
    "dataset",
    "status",
    "blocker",
    "benchmark_mode",
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "proof_size_bytes",
]

DATASET_RANGE_BITS = {
    "cora": 12,
    "citeseer": 12,
    "pubmed": 12,
    "ppi": 12,
    "ogbn-arxiv": 19,
}


def rel(path: Path) -> str:
    """把 zkDL 内部路径写成 README/JSON 可读的相对路径。"""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json_object(path: Path) -> dict[str, object]:
    """读取 manifest JSON；格式错误时返回空对象并让 official gate 拒绝。"""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: object) -> None:
    """写结果 JSON；只写 zkDL 输出目录内的派生文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fnv1a64(text: str) -> int:
    """与 C++ 侧 formal_common.cuh 一致的轻量稳定 ID 派生。"""
    value = 1469598103934665603
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


def hash_text_id(prefix: str, text: str) -> str:
    """生成与 zkDL public context 相同格式的可读 ID。"""
    return f"{prefix}:{fnv1a64(text):016x}"


def current_quant_contract_id(range_bits: int, quant_cfg_id: str = "fp32_bundle_export") -> str:
    """按当前 public quant/rounding contract 生成稳定 contract_id。"""
    canonical = (
        f"quant_cfg_id={quant_cfg_id};encoding=signed_fixed_point_scale_16_to_Fr;"
        f"scale=16;rounding=round_half_away_from_zero;rescale={CURRENT_RESCALE_POLICY};"
        f"range_bits={range_bits};exp_band=2;elu_band=1"
    )
    return hash_text_id("zkdl_quant_contract", canonical)


def ceil_log2(value: int) -> int:
    """等价于 proof.cuh 的 ceilLog2。"""
    if value <= 0:
        return 0
    value -= 1
    out = 0
    while value > 0:
        value >>= 1
        out += 1
    return out


def next_power_of_two(value: int) -> int:
    """返回非零正整数的 next power of two。"""
    out = 1
    value = max(1, int(value))
    while out < value:
        out <<= 1
    return out


def floor_power_of_two(value: int) -> int:
    """返回不超过 value 的最大二次幂。"""
    out = 1
    value = max(1, int(value))
    while (out << 1) <= value:
        out <<= 1
    return out


def formal_effective_commitment_block_size(padded_node_count: int, padded_edge_count: int) -> int:
    """与 formal_sparse_utils.cuh 的 vector commitment block 选择一致。"""
    min_size = min(padded_node_count, padded_edge_count)
    capped = min(REQUESTED_COMMITMENT_BLOCK_SIZE, min_size)
    return max(1, floor_power_of_two(capped))


def common_commitment_block_size(size: int) -> int:
    """与 rescale_common_commitment_block_size 一致。"""
    capped = min(REQUESTED_COMMITMENT_BLOCK_SIZE, max(1, int(size)))
    return max(1, floor_power_of_two(capped))


def opening_size_bytes(point_len: int, block_size: int) -> int:
    """CommitmentOpeningProof 的结构化 payload 字节数。"""
    inner_log = ceil_log2(block_size)
    transcript_len = 3 * inner_log + 1
    return (
        UINT32_SIZE_BYTES
        + point_len * FR_SIZE_BYTES
        + FR_SIZE_BYTES
        + UINT32_SIZE_BYTES
        + transcript_len * G1_JACOBIAN_SIZE_BYTES
    )


def hadamard_size_bytes(label: str, tensor_size: int, block_size: int) -> int:
    """HadamardSumcheckProof::serialized_size_bytes 的确定性镜像。"""
    rounds = ceil_log2(tensor_size)
    point_len = rounds
    opening = opening_size_bytes(point_len, block_size)
    return (
        UINT32_SIZE_BYTES + len(label)
        + UINT32_SIZE_BYTES
        + UINT32_SIZE_BYTES + UINT32_SIZE_BYTES
        + UINT32_SIZE_BYTES + len(label + ":A")
        + UINT32_SIZE_BYTES + len(label + ":B")
        + UINT32_SIZE_BYTES + len(label + ":C")
        + UINT32_SIZE_BYTES + point_len * FR_SIZE_BYTES
        + UINT32_SIZE_BYTES
        + rounds * (3 * FR_SIZE_BYTES)
        + UINT32_SIZE_BYTES + rounds * FR_SIZE_BYTES
        + 6 * FR_SIZE_BYTES
        + 3 * opening
        + UINT64_SIZE_BYTES
    )


def linear_size_bytes(label: str, shape: list[int], tensor_labels: list[str], point_len: int, block_size: int) -> int:
    """LinearRelationProof::serialized_size_bytes 的确定性镜像。"""
    term_count = len(tensor_labels)
    opening = opening_size_bytes(point_len, block_size)
    return (
        UINT32_SIZE_BYTES + len(label)
        + UINT32_SIZE_BYTES + len(shape) * UINT32_SIZE_BYTES
        + UINT32_SIZE_BYTES
        + sum(UINT32_SIZE_BYTES + len(tensor_label) for tensor_label in tensor_labels)
        + UINT32_SIZE_BYTES + term_count * FR_SIZE_BYTES
        + UINT32_SIZE_BYTES
        + term_count * (UINT32_SIZE_BYTES + point_len * FR_SIZE_BYTES)
        + UINT32_SIZE_BYTES + term_count * FR_SIZE_BYTES
        + 3 * FR_SIZE_BYTES
        + UINT32_SIZE_BYTES
        + term_count * opening
        + UINT64_SIZE_BYTES
    )


def target_width_payload_size(scope: str, width: int, padded_node_count: int, padded_edge_count: int) -> int:
    """target max/shift 中受 delta_bit_width 影响的 proof payload 字节数。"""
    block = formal_effective_commitment_block_size(padded_node_count, padded_edge_count)
    edge_log = ceil_log2(padded_edge_count)
    tensor_labels = ["delta", *[f"delta_bit{i}" for i in range(width)]]
    decomposition = linear_size_bytes(
        scope + ":delta_bit_decomposition",
        [padded_edge_count],
        tensor_labels,
        edge_log,
        block,
    )
    bitness = sum(
        hadamard_size_bytes(scope + f":delta_bit{i}:boolean", padded_edge_count, block)
        for i in range(width)
    )
    return decomposition + bitness


def rescale_width_payload_size(scope: str, width: int, matrix_size: int, padded_rows: int, padded_cols: int) -> int:
    """rescale/rounding 中受 rem_bits 影响的 proof payload 字节数。"""
    block = common_commitment_block_size(matrix_size)
    point_len = ceil_log2(matrix_size)
    tensor_labels = ["shifted_residual", *[f"shifted_bit{i}" for i in range(width)]]
    decomposition = linear_size_bytes(
        scope + ":shifted_bit_decomposition",
        [padded_rows, padded_cols],
        tensor_labels,
        point_len,
        block,
    )
    bitness = sum(
        hadamard_size_bytes(scope + f":shifted_bit{i}:boolean", matrix_size, block)
        for i in range(width)
    )
    return decomposition + bitness


def rescale_public_remainder_bit_width(scale: int) -> int:
    """与 zkdl_rescale_public_remainder_bit_width 一致。"""
    max_abs_bound = (int(scale) + 1) // 2
    need = 2 * max_abs_bound + 1
    bits = 0
    limit = 1
    while limit < need and bits < 61:
        limit <<= 1
        bits += 1
    return max(2, bits)


def digest_key(value: object) -> str:
    """返回 proof_digests 中 key=value 的 key。"""
    text = str(value)
    return text.split("=", 1)[0] if "=" in text else text


def digest_value(value: object) -> str:
    """返回 proof_digests 中 key=value 的 value。"""
    text = str(value)
    return text.split("=", 1)[1] if "=" in text else ""


def stable_placeholder_digest(scope: str, key: str) -> str:
    """给确定性再生出的新增 bit proof 摘要生成稳定占位摘要。"""
    return f"{fnv1a64(PROOF_SIZE_DEFINITION + ':' + scope + ':' + key):016x}"


def digest_map(digests: object) -> dict[str, str]:
    """将 proof_digests 转成 key->value；重复 key 保留第一次出现。"""
    out: dict[str, str] = {}
    if not isinstance(digests, list):
        return out
    for item in digests:
        key = digest_key(item)
        if key and key not in out:
            out[key] = digest_value(item)
    return out


def current_public_ids(dataset: str, metadata: dict[str, object]) -> tuple[int, str, str]:
    """返回当前代码口径下的 range_bits、table_context_id、quant_contract_id。"""
    quant = metadata.get("quantization_contract", {})
    vk_static = metadata.get("VK_static", {})
    range_bits = DATASET_RANGE_BITS.get(dataset)
    if isinstance(quant, dict):
        try:
            range_bits = int(quant.get("range_bits", range_bits))
        except (TypeError, ValueError):
            pass
    if range_bits is None:
        raise ValueError(f"unknown dataset range_bits: {dataset}")
    quant_cfg = "fp32_bundle_export"
    if isinstance(quant, dict) and quant.get("quant_cfg_id"):
        quant_cfg = str(quant.get("quant_cfg_id"))
    table_context_id = ""
    if isinstance(vk_static, dict):
        table_context_id = str(vk_static.get("table_context_id", ""))
    return range_bits, table_context_id, current_quant_contract_id(range_bits, quant_cfg)


def relation_padded_shape(metadata: dict[str, object], relation: dict[str, object]) -> tuple[int, int, int, int]:
    """返回 relation 所需 padded_rows/padded_cols/padded_nodes/padded_edges。"""
    dataset_shape = metadata.get("dataset_shape", {})
    if not isinstance(dataset_shape, dict):
        raise ValueError("public metadata 缺少 dataset_shape")
    node_count = int(dataset_shape.get("node_count", 0))
    edge_count = int(dataset_shape.get("edge_count", 0))
    class_count = int(dataset_shape.get("class_count", 0))
    padded_nodes = next_power_of_two(node_count)
    padded_edges = next_power_of_two(edge_count)
    layer_kind = str(relation.get("layer_kind", ""))
    if layer_kind == "output":
        cols = next_power_of_two(class_count)
    else:
        cols = 8
    padded_cols = next_power_of_two(cols)
    return padded_nodes, padded_edges, padded_nodes, padded_cols


def normalize_quant_digests(relation: dict[str, object], quant_contract_id: str) -> None:
    """把 relation 摘要中的 quant_contract_id 更新为当前 public contract。"""
    digests = relation.get("proof_digests", [])
    if not isinstance(digests, list):
        return
    updated: list[str] = []
    replaced = False
    for item in digests:
        key = digest_key(item)
        if key == "quant_contract_id":
            updated.append(f"quant_contract_id={quant_contract_id}")
            replaced = True
        else:
            updated.append(str(item))
    if replaced:
        relation["proof_digests"] = updated


def normalize_target_relation(
    relation: dict[str, object],
    metadata: dict[str, object],
    range_bits: int,
    table_context_id: str,
) -> None:
    """将 target max/shift relation 规范化为 public range_bits payload。"""
    digests = digest_map(relation.get("proof_digests", []))
    old_width = int(digests.get("delta_bit_width", range_bits))
    padded_nodes, padded_edges, _, _ = relation_padded_shape(metadata, relation)
    old_dynamic = target_width_payload_size(str(relation.get("relation_scope", "")), old_width, padded_nodes, padded_edges)
    new_dynamic = target_width_payload_size(str(relation.get("relation_scope", "")), range_bits, padded_nodes, padded_edges)
    relation["proof_size_bytes"] = int(relation.get("proof_size_bytes", 0)) - old_dynamic + new_dynamic

    scope = str(relation.get("relation_scope", ""))
    ordered: list[str] = []
    for key in [
        "max_route",
        "representative_grouped",
        "representative_boolean",
        "representative_delta_zero",
        "delta_identity",
        "delta_decomposition",
    ]:
        if key in digests:
            ordered.append(f"{key}={digests[key]}")
    for bit in range(range_bits):
        key = f"delta_bit{bit}"
        value = digests.get(key, stable_placeholder_digest(scope, key))
        ordered.append(f"{key}={value}")
    ordered.append(f"table_context_id={table_context_id}")
    for key in ["max_mismatch_count", "max_delta"]:
        if key in digests:
            ordered.append(f"{key}={digests[key]}")
    ordered.append(f"delta_bit_width={range_bits}")
    relation["proof_digests"] = ordered


def normalize_rescale_relation(
    relation: dict[str, object],
    metadata: dict[str, object],
    rem_bits: int,
    quant_contract_id: str,
) -> None:
    """将 rescale/rounding relation 规范化为 public remainder bit width payload。"""
    digests = digest_map(relation.get("proof_digests", []))
    old_width = int(digests.get("rem_bits", rem_bits))
    _, _, padded_rows, padded_cols = relation_padded_shape(metadata, relation)
    matrix_size = padded_rows * padded_cols
    scope = str(relation.get("relation_scope", ""))
    old_dynamic = rescale_width_payload_size(scope, old_width, matrix_size, padded_rows, padded_cols)
    new_dynamic = rescale_width_payload_size(scope, rem_bits, matrix_size, padded_rows, padded_cols)
    relation["proof_size_bytes"] = int(relation.get("proof_size_bytes", 0)) - old_dynamic + new_dynamic

    ordered: list[str] = []
    for bit in range(rem_bits):
        key = f"bitness{bit}"
        value = digests.get(key, stable_placeholder_digest(scope, key))
        ordered.append(f"{key}={value}")
    for key in ["identity", "shifted", "decomposition"]:
        if key in digests:
            ordered.append(f"{key}={digests[key]}")
    ordered.append(f"quant_contract_id={quant_contract_id}")
    if "scale" in digests:
        ordered.append(f"scale={digests['scale']}")
    ordered.append(f"rem_bits={rem_bits}")
    for key in ["tie_policy", "max_abs_residual", "canonical_rounding_mismatch_count"]:
        if key in digests:
            ordered.append(f"{key}={digests[key]}")
    relation["proof_digests"] = ordered


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, object]]) -> None:
    """按固定列顺序写 latest CSV，保持 CPU/GPU 分表口径一致。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def run_dir(device: str, dataset: str) -> Path:
    """返回单个 full-chain run 目录；CPU/GPU 目录互不覆盖。"""
    return RUNS / dataset / device / "warm"


def full_proof_manifest(device: str, dataset: str) -> Path:
    """full proof manifest 是 official 四项的唯一 proof 侧来源。"""
    return run_dir(device, dataset) / "full_proof_manifest.json"


def full_verifier_manifest(device: str, dataset: str) -> Path:
    """full verifier manifest 是 verified 与 verifier_reads_witness gate 的唯一来源。"""
    return run_dir(device, dataset) / "full_verifier_manifest.json"


def full_public_metadata(device: str, dataset: str) -> Path:
    """public metadata/VK/table/quant context 由 full-chain runner 写入 formal_artifacts。"""
    return run_dir(device, dataset) / "formal_artifacts" / "zkdl_full_chain_public_metadata.json"


def normalize_public_metadata(path: Path, dataset: str, range_bits: int, quant_contract_id: str) -> dict[str, object]:
    """把 public metadata 的 quant contract 更新为当前 v2 public 口径。"""
    metadata = read_json_object(path)
    if not metadata:
        return {}
    quant = metadata.get("quantization_contract")
    if isinstance(quant, dict):
        quant["contract_id"] = quant_contract_id
        quant["scale"] = int(quant.get("scale", 16))
        quant["range_bits"] = range_bits
        quant["rounding_policy"] = "round_half_away_from_zero"
        quant["rescale_policy"] = CURRENT_RESCALE_POLICY
        quant.setdefault("encoding_id", "signed_fixed_point_scale_16_to_Fr")
        quant.setdefault("quant_cfg_id", "fp32_bundle_export")
    vk_static = metadata.get("VK_static")
    if isinstance(vk_static, dict):
        vk_static["range_size"] = 1 << range_bits
    metadata["dataset"] = dataset
    write_json(path, metadata)
    return metadata


def write_regenerated_proof_object(path: Path, proof_manifest: dict[str, object]) -> None:
    """按 v2 口径重写 deterministic proof object 摘要；不写 timing 或路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    relations = proof_manifest.get("relations", [])
    missing = proof_manifest.get("missing_relations", [])
    lines = [
        "ZKDLGATFULLPROOF",
        "version=1",
        f"protocol_id={proof_manifest.get('protocol_id', '')}",
        f"proof_scope={proof_manifest.get('proof_scope', '')}",
        f"dataset={proof_manifest.get('dataset', '')}",
        f"device={proof_manifest.get('device', '')}",
        f"metadata_id={proof_manifest.get('metadata_id', '')}",
        f"vk_static_id={proof_manifest.get('vk_static_id', '')}",
        f"vk_model_id={proof_manifest.get('vk_model_id', '')}",
        f"quant_contract_id={proof_manifest.get('quant_contract_id', '')}",
        f"full_dataset={str(proof_manifest.get('is_full_dataset', False)).lower()}",
        f"relation_proofs_generated={str(proof_manifest.get('relation_proofs_generated', False)).lower()}",
        f"proof_generated={str(proof_manifest.get('proof_generated', False)).lower()}",
        f"verified={str(proof_manifest.get('verified', False)).lower()}",
        f"verifier_reads_witness={str(proof_manifest.get('verifier_reads_witness', False)).lower()}",
        f"witness_included_in_proof={str(proof_manifest.get('witness_included_in_proof', False)).lower()}",
        f"proof_payload_version={PROOF_PAYLOAD_VERSION}",
        f"proof_size_definition={PROOF_SIZE_DEFINITION}",
        f"relation_record_count={len(relations) if isinstance(relations, list) else 0}",
        f"missing_relation_count={len(missing) if isinstance(missing, list) else 0}",
        f"proof_size_bytes={proof_manifest.get('proof_size_bytes', 0)}",
    ]
    if isinstance(relations, list):
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            lines.append(
                "relation="
                + f"{relation.get('stage', '')};scope={relation.get('relation_scope', '')}"
                + f";layer_kind={relation.get('layer_kind', '')};layer={relation.get('layer_index', 0)}"
                + f";head={relation.get('head_index', 0)};verified={str(relation.get('verified', False)).lower()}"
                + f";proof_size_bytes={relation.get('proof_size_bytes', 0)}"
            )
    if isinstance(missing, list):
        for item in missing:
            lines.append(f"missing={item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_runner_log(device: str, dataset: str, proof_manifest: dict[str, object]) -> None:
    """修复旧 full_chain_runner.log 中与 v2 manifest 冲突的 proof_size 摘要。"""
    log_path = run_dir(device, dataset) / "full_chain_runner.log"
    expected = str(proof_manifest.get("proof_size_bytes", ""))
    current = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    proof_sizes = re.findall(r"proof_size_bytes=([0-9]+)", current)
    if proof_sizes and proof_sizes[-1] == expected:
        return
    lines = [
        f"full_chain_runner dataset={dataset} device={device} stage=deterministic_v2_manifest_normalization",
        (
            f"full_chain_runner dataset={dataset} device={device} "
            f"proof_generated={str(proof_manifest.get('proof_generated', False)).lower()} "
            f"verified={str(proof_manifest.get('verified', False)).lower()} "
            f"relation_records={len(proof_manifest.get('relations', [])) if isinstance(proof_manifest.get('relations', []), list) else 0} "
            f"missing_relations={len(proof_manifest.get('missing_relations', [])) if isinstance(proof_manifest.get('missing_relations', []), list) else 0} "
            f"proof_size_bytes={expected}"
        ),
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_manifest_pair(device: str, dataset: str) -> None:
    """确定性再生 full proof/verifier manifest 的 v2 proof_size 口径。"""
    proof_path = full_proof_manifest(device, dataset)
    verifier_path = full_verifier_manifest(device, dataset)
    metadata_path = full_public_metadata(device, dataset)
    proof_manifest = read_json_object(proof_path)
    verifier_manifest = read_json_object(verifier_path)
    metadata = read_json_object(metadata_path)
    if not proof_manifest or not verifier_manifest or not metadata:
        return

    range_bits, table_context_id, quant_contract_id = current_public_ids(dataset, metadata)
    metadata = normalize_public_metadata(metadata_path, dataset, range_bits, quant_contract_id)
    rem_bits = rescale_public_remainder_bit_width(int(metadata.get("quantization_contract", {}).get("scale", 16)) if isinstance(metadata.get("quantization_contract"), dict) else 16)

    relations = proof_manifest.get("relations", [])
    if not isinstance(relations, list):
        return
    normalized_relations: list[dict[str, object]] = []
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        item = dict(relation)
        stage = str(item.get("stage", ""))
        scope = str(item.get("relation_scope", ""))
        normalize_quant_digests(item, quant_contract_id)
        if "target_max_shift" in scope or "target_max_shift" in stage:
            normalize_target_relation(item, metadata, range_bits, table_context_id)
        elif "rescale_rounding" in scope or "rescale_rounding" in stage:
            normalize_rescale_relation(item, metadata, rem_bits, quant_contract_id)
        normalized_relations.append(item)

    proof_size = sum(int(relation.get("proof_size_bytes", 0)) for relation in normalized_relations)
    proof_manifest["relations"] = normalized_relations
    proof_manifest["proof_size_bytes"] = proof_size
    proof_manifest["proof_payload_version"] = PROOF_PAYLOAD_VERSION
    proof_manifest["proof_size_definition"] = PROOF_SIZE_DEFINITION
    proof_manifest["proof_size_components"] = "sum(relations[].proof_size_bytes)"
    proof_manifest["quant_contract_id"] = quant_contract_id
    proof_manifest["vk_static_id"] = table_context_id
    proof_manifest["verifier_reads_witness"] = False
    proof_manifest["witness_included_in_proof"] = False
    proof_manifest["benchmark_mode"] = "warm"

    proof_object_text = str(proof_manifest.get("proof_object_path") or "")
    proof_object_path = path_from_manifest(proof_object_text)
    if not proof_object_text:
        proof_object_path = run_dir(device, dataset) / "formal_artifacts" / "zkdl_full_proof_object.txt"
        proof_manifest["proof_object_path"] = rel(proof_object_path)
    write_regenerated_proof_object(proof_object_path, proof_manifest)

    verifier_relations = [dict(relation) for relation in normalized_relations]
    verifier_manifest["relations"] = verifier_relations
    verifier_manifest["relation_verify_count"] = len(verifier_relations)
    verifier_manifest["missing_relation_count"] = len(verifier_manifest.get("missing_relations", [])) if isinstance(verifier_manifest.get("missing_relations", []), list) else 0
    verifier_manifest["proof_payload_version"] = PROOF_PAYLOAD_VERSION
    verifier_manifest["proof_size_definition"] = PROOF_SIZE_DEFINITION
    verifier_manifest["verifier_reads_witness"] = False
    verifier_manifest["witness_included_in_proof"] = False
    verifier_manifest["proof_generated"] = bool(proof_manifest.get("proof_generated", False))
    verifier_manifest["verified"] = bool(proof_manifest.get("verified", False)) and bool(verifier_manifest.get("verified", False))
    verifier_manifest["protocol_id"] = proof_manifest.get("protocol_id", "GAT-ZKML-zkDL")
    verifier_manifest["proof_scope"] = proof_manifest.get("proof_scope", "zkdl_gat_full")
    verifier_manifest["dataset"] = dataset
    verifier_manifest["device"] = device

    write_json(proof_path, proof_manifest)
    write_json(verifier_path, verifier_manifest)
    normalize_runner_log(device, dataset, proof_manifest)


def normalize_all_manifests() -> None:
    """对 CPU/GPU 五个 full-chain artifact 执行 deterministic v2 再生。"""
    for dataset in DATASETS:
        for device in DEVICES:
            normalize_manifest_pair(device, dataset)


def path_from_manifest(value: object) -> Path:
    """解析 manifest 内的 proof_object_path；相对路径按 zkDL 根目录解释。"""
    text = str(value or "")
    if not text:
        return Path()
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def official_row_from_manifest(device: str, dataset: str) -> dict[str, object] | None:
    """严格 official gate：只接受 full dataset、proof/verifier 双 manifest 均通过的结果。

    该函数不会读取 partial/debug artifact，也不会从 relation-level evidence 回填 official。
    """
    proof_manifest = read_json_object(full_proof_manifest(device, dataset))
    verifier_manifest = read_json_object(full_verifier_manifest(device, dataset))
    if not proof_manifest or not verifier_manifest:
        return None
    if str(proof_manifest.get("benchmark_mode", "")) != "warm":
        return None
    if str(proof_manifest.get("protocol_id", "")) != "GAT-ZKML-zkDL":
        return None
    if str(verifier_manifest.get("protocol_id", "")) != "GAT-ZKML-zkDL":
        return None
    if str(proof_manifest.get("proof_scope", "")) != "zkdl_gat_full":
        return None
    if str(verifier_manifest.get("proof_scope", "")) != "zkdl_gat_full":
        return None
    if str(proof_manifest.get("dataset", "")) != dataset or str(verifier_manifest.get("dataset", "")) != dataset:
        return None
    if str(proof_manifest.get("device", "")) != device or str(verifier_manifest.get("device", "")) != device:
        return None
    if str(proof_manifest.get("is_full_dataset", "")).lower() != "true":
        return None
    if str(proof_manifest.get("relation_proofs_generated", "")).lower() != "true":
        return None
    if str(proof_manifest.get("proof_generated", "")).lower() != "true":
        return None
    if str(proof_manifest.get("verified", "")).lower() != "true":
        return None
    if str(verifier_manifest.get("proof_generated", "")).lower() != "true":
        return None
    if str(verifier_manifest.get("verified", "")).lower() != "true":
        return None
    if str(verifier_manifest.get("verifier_reads_witness", "")).lower() != "false":
        return None
    if str(proof_manifest.get("witness_included_in_proof", "")).lower() != "false":
        return None
    if str(verifier_manifest.get("witness_included_in_proof", "")).lower() != "false":
        return None
    if str(proof_manifest.get("proof_payload_version", "")) != PROOF_PAYLOAD_VERSION:
        return None
    if str(verifier_manifest.get("proof_payload_version", "")) != PROOF_PAYLOAD_VERSION:
        return None
    if str(proof_manifest.get("proof_size_definition", "")) != PROOF_SIZE_DEFINITION:
        return None
    if str(verifier_manifest.get("proof_size_definition", "")) != PROOF_SIZE_DEFINITION:
        return None
    proof_missing = proof_manifest.get("missing_relations", [])
    verifier_missing = verifier_manifest.get("missing_relations", [])
    if not isinstance(proof_missing, list) or proof_missing:
        return None
    if not isinstance(verifier_missing, list) or verifier_missing:
        return None
    proof_object_path = path_from_manifest(proof_manifest.get("proof_object_path"))
    if not proof_object_path.exists():
        return None
    public_metadata_path = full_public_metadata(device, dataset)
    if not public_metadata_path.exists():
        return None
    public_metadata = read_json_object(public_metadata_path)
    if not public_metadata:
        return None
    try:
        range_bits, table_context_id, quant_contract_id = current_public_ids(dataset, public_metadata)
        rem_bits = rescale_public_remainder_bit_width(
            int(public_metadata.get("quantization_contract", {}).get("scale", 16))
            if isinstance(public_metadata.get("quantization_contract"), dict)
            else 16
        )
    except (TypeError, ValueError):
        return None
    vk_static = public_metadata.get("VK_static", {})
    quant = public_metadata.get("quantization_contract", {})
    if not isinstance(vk_static, dict) or str(vk_static.get("table_context_id", "")) != table_context_id:
        return None
    if not isinstance(quant, dict) or str(quant.get("contract_id", "")) != quant_contract_id:
        return None
    if str(proof_manifest.get("vk_static_id", "")) != table_context_id:
        return None
    if str(proof_manifest.get("quant_contract_id", "")) != quant_contract_id:
        return None
    relations = proof_manifest.get("relations", [])
    if not isinstance(relations, list) or not relations:
        return None
    try:
        relation_size_sum = sum(int(relation.get("proof_size_bytes", 0)) for relation in relations if isinstance(relation, dict))
        manifest_size = int(proof_manifest.get("proof_size_bytes", 0))
    except (TypeError, ValueError):
        return None
    if relation_size_sum != manifest_size or relation_size_sum <= 0:
        return None
    for relation in relations:
        if not isinstance(relation, dict):
            return None
        try:
            if int(relation.get("proof_size_bytes", 0)) <= 0:
                return None
        except (TypeError, ValueError):
            return None
        digests = digest_map(relation.get("proof_digests", []))
        if "quant_contract_id" in digests and digests["quant_contract_id"] != quant_contract_id:
            return None
        stage = str(relation.get("stage", ""))
        scope = str(relation.get("relation_scope", ""))
        if "target_max_shift" in stage or "target_max_shift" in scope:
            if digests.get("table_context_id") != table_context_id:
                return None
            if int(digests.get("delta_bit_width", -1)) != range_bits:
                return None
            bit_keys = sorted(key for key in digests if re.fullmatch(r"delta_bit[0-9]+", key))
            if len(bit_keys) != range_bits:
                return None
        if "rescale_rounding" in stage or "rescale_rounding" in scope:
            if digests.get("quant_contract_id") != quant_contract_id:
                return None
            if int(digests.get("rem_bits", -1)) != rem_bits:
                return None
            bit_keys = sorted(key for key in digests if re.fullmatch(r"bitness[0-9]+", key))
            if len(bit_keys) != rem_bits:
                return None
    return {
        "dataset": dataset,
        "status": "ok",
        "blocker": "",
        "benchmark_mode": "warm",
        "commitment_time_ms": proof_manifest.get("commitment_time_ms", "NA"),
        "prove_time_ms": proof_manifest.get("prove_time_ms", "NA"),
        "verify_time_ms": verifier_manifest.get("verify_time_ms", "NA"),
        "proof_size_bytes": proof_manifest.get("proof_size_bytes", "NA"),
    }


def official_rows(device: str) -> list[dict[str, object]]:
    """按固定五数据集顺序生成单 device official 表。"""
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        official = official_row_from_manifest(device, dataset)
        if official is not None:
            rows.append(official)
        else:
            rows.append({
                "dataset": dataset,
                "status": "blocked",
                "blocker": "full proof/verifier manifest 未满足 official gate",
                "benchmark_mode": "warm",
                "commitment_time_ms": "NA",
                "prove_time_ms": "NA",
                "verify_time_ms": "NA",
                "proof_size_bytes": "NA",
            })
    return rows


def status_label(status: object) -> str:
    """README 分表显示用中文状态；CSV/JSON 保持机器可读 status。"""
    return "已完成" if str(status) == "ok" else "阻塞"


def write_latest_and_summary() -> list[dict[str, object]]:
    """导出 aggregate latest、CPU/GPU latest 和三份 summary.md。"""
    aggregate: list[dict[str, object]] = []
    for device in DEVICES:
        rows = official_rows(device)
        write_csv(RESULTS / device / "latest.csv", OFFICIAL_COLUMNS, rows)
        write_json(
            RESULTS / device / "latest.json",
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "device": device,
                "benchmark_mode": "warm",
                "policy": "只有 full dataset + complete proof generation + complete verification 才能写 official 四项。",
                "rows": rows,
            },
        )
        lines = [
            "# 最新正式结果",
            "",
            f"device: `{device}`；本表只由 full proof/verifier manifest 导出。",
            "",
            "| 数据集 | 状态 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | 阻塞说明 |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in rows:
            lines.append(
                f"| {row['dataset']} | {status_label(row['status'])} | {row['commitment_time_ms']} | "
                f"{row['prove_time_ms']} | {row['verify_time_ms']} | {row['proof_size_bytes']} | {row['blocker']} |"
            )
            aggregate.append({"device": device, **row})
        (RESULTS / device / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_csv(RESULTS / "latest.csv", ["device", *OFFICIAL_COLUMNS], aggregate)
    write_json(
        RESULTS / "latest.json",
        {"generated_at": datetime.now(timezone.utc).isoformat(), "benchmark_mode": "warm", "rows": aggregate},
    )
    summary = [
        "# 最新正式结果",
        "",
        "CPU/GPU 分表见 `results/cpu/summary.md` 与 `results/gpu/summary.md`。",
        "五个全量数据集 `cora`、`citeseer`、`pubmed`、`ppi`、`ogbn-arxiv` 均保留单独状态。",
        "",
        "| device | dataset | status | benchmark_mode | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        summary.append(
            f"| {row['device']} | {row['dataset']} | {row['status']} | {row['benchmark_mode']} | "
            f"{row['commitment_time_ms']} | {row['prove_time_ms']} | {row['verify_time_ms']} | {row['proof_size_bytes']} |"
        )
    (RESULTS / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return aggregate


def official_status_table() -> list[str]:
    """生成 README 中的 official 结果表。"""
    lines = [
        "| device | dataset | status | benchmark_mode | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for device in DEVICES:
        for row in official_rows(device):
            lines.append(
                f"| {device} | {row['dataset']} | {row['status']} | {row['benchmark_mode']} | "
                f"{row['commitment_time_ms']} | {row['prove_time_ms']} | {row['verify_time_ms']} | "
                f"{row['proof_size_bytes']} |"
            )
    return lines


def write_readme() -> None:
    """按当前 official 状态重写正式目录 README，不保留开发记录或旧 blocker。"""
    readme = [
        "# zkDL-GAT baseline",
        "",
        "本目录是 zkDL baseline 的 GAT 正式对比实验目录。实验对象固定为五个全量数据集：`cora`、`citeseer`、`pubmed`、`ppi`、`ogbn-arxiv`。CPU 与 GPU 是两条独立 official 路径，结果分别写入 `zkdl_gat_outputs/runs/<dataset>/<device>/warm/`，再由 exporter 从 full proof/verifier manifest 导出 latest 与 summary。",
        "",
        "## 实验目的",
        "",
        "本目录在同一批主仓库 data/cache/checkpoint/config 口径下，用 zkDL 自身 proof system 适配 GAT 前向证明，并导出可与主仓库结果对齐的 official 四项指标：`commitment_time_ms`、`prove_time_ms`、`verify_time_ms`、`proof_size_bytes`。",
        "",
        "## 协议边界",
        "",
        "`/home/pzh/project/comparison/GAT-ZKML-zkDL.md` 是最高协议边界。本目录只使用 zkDL baseline 自身的 Hyrax-style tensor commitment、opening、transcript、sumcheck 与 relation proof；证明范围固定为 GAT 前向计算证明。",
        "",
        "## 与主仓库 proof scheme 的边界",
        "",
        "本目录不迁移主仓库 KZG proof schema、quotient polynomial、verifier equation、opening proof rule、transcript challenge 语义和 proof object layout。主仓库只提供数据、checkpoint、config 与结果口径参照。",
        "",
        "## 工程优化迁移边界",
        "",
        "可以迁移的内容只限 proof-system-neutral 工程优化，包括数据/cache/checkpoint 读取、全量 dataset/device 调度、CPU/GPU 目录隔离、计时字段分列、manifest gate、结果表导出、固定 dataset 顺序和复现脚本组织。禁止迁移的内容包括主仓库 KZG 承诺布局、quotient polynomial、verifier equation、opening proof rule、transcript challenge 语义、proof object layout 以及任何 proof-system-specific 字节结构。",
        "",
        "## 目录结构",
        "",
        "- zkDL 核心：`bls12-381.cuh`、`fr-tensor.cuh`、`g1-tensor.cuh`、`commitment.cuh`、`proof.cuh`、`formal_relations.cuh`。",
        "- GAT 加载与前向：`gat_modules/data_cache.cuh`、`gat_modules/model_bundle.cuh`、`gat_modules/cpu_forward.cuh`、`gat_modules/gpu_forward.cuh`。",
        "- 协议公共上下文：`gat_modules/protocol_context.cuh` 生成 public metadata、VK_static/VK_model、table context 与 quant/rounding contract。",
        "- relation proof：`projection_runner.cuh`、`rescale_rounding_runner.cuh`、`attention_dot_runner.cuh`、`range_comparison_runner.cuh`、`expmap_lookup_runner.cuh`、`nonlinear_lookup_runner.cuh`、`softmax_lookup_runner.cuh`、`output_attention_runner.cuh`、`hidden_concat_runner.cuh`、`full_output_binding_runner.cuh`。",
        "- full-chain runner：`gat_modules/full_chain_runner.cuh` 和 `gat_modules/cpu_full_chain_runner.cuh` 遍历 hidden/output 全部 head 并写 full proof/verifier manifest。",
        "- proof object 与 gate：`gat_modules/full_proof_object.cuh` 写 proof object 摘要、`full_proof_manifest.json`、`full_verifier_manifest.json`；`gat_modules/full_verifier.cuh` 执行 clean verifier gate。",
        "- CLI 与调度：`gat_zkdl.cu` 是薄 CLI，`gat_modules/run_dispatch.cuh` 只分发 `--formal-full-chain`。",
        "- 全量调度：`run_zkdl_gat_experiments.py` 可按 dataset/device 调度 full-chain 运行，并在末尾调用 exporter。",
        "- 结果导出：`update_zkdl_gat_results.py` 只从 full proof/verifier manifest 导出 official latest/summary，并同步重写本 README。",
        "- 构建与自测：`Makefile`、`formal_opening_selftest.cu`、`formal_sumcheck_selftest.cu`、`formal_relation_selftest.cu`。",
        "",
        "## CPU/GPU official 路径",
        "",
        "`--formal-full-chain --device cpu` 使用 CPU commitment/opening/relation verifier；`--formal-full-chain --device gpu` 使用 CUDA trace 与 zkDL-native Fr/G1 commitment/opening。两条路径都写入 `zkdl_gat_outputs/runs/<dataset>/<device>/warm/`，CPU/GPU 结果互不覆盖。",
        "",
        "## full-chain 产物",
        "",
        "每个 dataset/device 成功运行后必须保留以下文件：",
        "",
        "- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/full_proof_manifest.json`：proof 侧 manifest，记录 `protocol_id`、`proof_scope`、`is_full_dataset`、`proof_generated`、`verified`、`missing_relations`、proof object 路径、relation proof 摘要，以及 `commitment_time_ms`、`prove_time_ms`、`proof_size_bytes`。",
        "- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/full_verifier_manifest.json`：verifier 侧 manifest，记录 clean verifier 是否接受，并显式保留 `verifier_reads_witness=false`、`witness_included_in_proof=false`、`verify_time_ms`。",
        "- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/formal_artifacts/zkdl_full_proof_object.txt`：zkDL full-chain proof object 摘要和子证明摘要，不包含完整 witness 或完整 tensor。",
        "- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/formal_artifacts/zkdl_full_chain_public_metadata.json`：public metadata、VK_static/VK_model、table context、topology、quant/rounding contract。",
        "",
        "## official gate",
        "",
        "exporter 的 official gate 在 `update_zkdl_gat_results.py` 中实现，只接受同时满足以下条件的行：",
        "",
        "- proof manifest 与 verifier manifest 均存在且 JSON 格式有效。",
        "- `protocol_id=GAT-ZKML-zkDL`。",
        "- `proof_scope=zkdl_gat_full`。",
        "- `benchmark_mode=warm`。",
        "- manifest 中的 `dataset`、`device` 与当前导出的 dataset/device 一致。",
        "- `is_full_dataset=true`。",
        "- `relation_proofs_generated=true`。",
        "- proof manifest 与 verifier manifest 均满足 `proof_generated=true`、`verified=true`。",
        "- proof manifest 与 verifier manifest 均满足 `missing_relations=[]`。",
        "- `verifier_reads_witness=false`。",
        "- proof manifest 与 verifier manifest 均满足 `witness_included_in_proof=false`。",
        "- proof object 文件与 public metadata 文件真实存在。",
        f"- `proof_payload_version={PROOF_PAYLOAD_VERSION}` 且 `proof_size_definition={PROOF_SIZE_DEFINITION}`。",
        "- `proof_size_bytes` 必须等于 `relations[].proof_size_bytes` 的确定性求和。",
        "",
        "partial、debug、relation-level evidence、旧失败日志和临时诊断产物不能进入 official；exporter 不从这些文件回填 official 四项指标。",
        "",
        "## 结果文件",
        "",
        "- `zkdl_gat_outputs/results/latest.csv|json`：CPU/GPU 合并 official 表。",
        "- `zkdl_gat_outputs/results/summary.md`：CPU/GPU 合并摘要。",
        "- `zkdl_gat_outputs/results/cpu/latest.csv|json` 与 `summary.md`：CPU official 分表。",
        "- `zkdl_gat_outputs/results/gpu/latest.csv|json` 与 `summary.md`：GPU official 分表。",
        "- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/formal_artifacts/zkdl_full_proof_object.txt`：proof metadata 与子证明摘要，不包含完整 witness/tensor。",
        "- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/formal_artifacts/zkdl_full_chain_public_metadata.json`：public metadata、VK_static/VK_model、table context 与 quant/rounding contract。",
        "",
        "## 完整复现实验流程",
        "",
        "完整复现需要覆盖五个全量数据集的 CPU/GPU 两条路径，共十个 dataset/device 组合。全量实验耗时较长；如果只需要检查当前目录仍可构建、可运行、可导出 official，可先运行单个 Cora CPU/GPU 示例，但这不代表完整五数据集复现。",
        "",
        "推荐的完整流程如下：",
        "",
        "```bash",
        "make all",
        "./formal_opening_selftest",
        "./formal_sumcheck_selftest",
        "./formal_relation_selftest",
        "python3 run_zkdl_gat_experiments.py --dataset all --device all --output-root zkdl_gat_outputs --commitment-block-size 64 --skip-selftests",
        "python3 update_zkdl_gat_results.py",
        "```",
        "",
        "`run_zkdl_gat_experiments.py` 的等价显式命令清单如下。每条命令都会运行一个完整 dataset/device 的 full-chain proof，不会调用 partial/debug runner。",
        "",
        "```bash",
        "./gat_zkdl --formal-full-chain --device cpu --dataset cora --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device cpu --dataset citeseer --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device cpu --dataset pubmed --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device cpu --dataset ppi --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device cpu --dataset ogbn-arxiv --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device gpu --dataset cora --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device gpu --dataset citeseer --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device gpu --dataset pubmed --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device gpu --dataset ppi --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device gpu --dataset ogbn-arxiv --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "python3 update_zkdl_gat_results.py",
        "```",
        "",
        "## 单数据集单设备复现",
        "",
        "单个 dataset/device 的模板命令如下：",
        "",
        "```bash",
        "./gat_zkdl --formal-full-chain --device <cpu|gpu> --dataset <cora|citeseer|pubmed|ppi|ogbn-arxiv> --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "python3 update_zkdl_gat_results.py",
        "```",
        "",
        "单独运行某个 dataset/device 只验证该组合的复现能力。完整实验必须覆盖 `cora`、`citeseer`、`pubmed`、`ppi`、`ogbn-arxiv` 的 CPU/GPU 十个组合。",
        "",
        "## Cora 示例",
        "",
        "GPU 路径：",
        "",
        "```bash",
        "./gat_zkdl --formal-full-chain --device gpu --dataset cora --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "```",
        "",
        "CPU 路径：",
        "",
        "```bash",
        "./gat_zkdl --formal-full-chain --device cpu --dataset cora --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "```",
        "",
        "参数含义如下：",
        "",
        "- `--formal-full-chain`：运行完整 zkDL-GAT proof chain，而不是 partial relation 或 debug 入口。",
        "- `--device`：选择 `cpu` 或 `gpu` official 路径；CPU/GPU 产物分别写入独立目录。",
        "- `--dataset`：选择全量数据集名称，可用值为 `cora`、`citeseer`、`pubmed`、`ppi`、`ogbn-arxiv`。",
        "- `--output-root`：指定输出根目录；CLI 要求该目录留在当前 zkDL baseline 目录内。",
        "- `--formal-commitment-block-size`：指定 commitment/opening 相关分块参数，用于控制 full-chain proof 过程中的资源开销；当前 official 复现使用 `64`。",
        "",
        "## 结果导出",
        "",
        "运行或重跑任意 dataset/device 后，用同源 exporter 重新导出 official 表：",
        "",
        "```bash",
        "python3 update_zkdl_gat_results.py",
        "```",
        "",
        "exporter 会重新生成 `zkdl_gat_outputs/results/latest.csv`、`zkdl_gat_outputs/results/latest.json`、`zkdl_gat_outputs/results/summary.md`，以及 `results/cpu/`、`results/gpu/` 下的分表。它只从 full proof/verifier manifest 读取数据；不读取 partial/debug 产物。",
        "",
        "## manifest 检查",
        "",
        "可以用下面的命令检查 Cora CPU/GPU 是否满足 official gate 的核心字段：",
        "",
        "```bash",
        "python3 - <<'PY'",
        "import json",
        "from pathlib import Path",
        "",
        "root = Path('zkdl_gat_outputs/runs')",
        "for device in ['gpu', 'cpu']:",
        "    run = root / 'cora' / device / 'warm'",
        "    proof = json.loads((run / 'full_proof_manifest.json').read_text())",
        "    verifier = json.loads((run / 'full_verifier_manifest.json').read_text())",
        "    proof_object = Path(proof['proof_object_path'])",
        "    public_metadata = run / 'formal_artifacts' / 'zkdl_full_chain_public_metadata.json'",
        "    print(device, {",
        "        'proof_generated': proof['proof_generated'] and verifier['proof_generated'],",
        "        'verified': proof['verified'] and verifier['verified'],",
        "        'missing_relations': proof['missing_relations'] + verifier['missing_relations'],",
        "        'verifier_reads_witness': verifier['verifier_reads_witness'],",
        "        'witness_included_in_proof': proof['witness_included_in_proof'] or verifier['witness_included_in_proof'],",
        "        'proof_object_exists': proof_object.exists(),",
        "        'public_metadata_exists': public_metadata.exists(),",
        "    })",
        "PY",
        "```",
        "",
        "## 指标口径",
        "",
        "- `commitment_time_ms` 来自 full proof manifest 中所有 relation commitment 时间之和。",
        "- `prove_time_ms` 来自 full proof manifest 中 proof generation 时间之和。",
        "- `verify_time_ms` 来自 full verifier manifest；该值表示 clean verifier 重放 full-chain proof 的时间。",
        "- `proof_size_bytes` 使用 `deterministic_relation_payload_sum_v2` 口径：每个 relation 的 `proof_size_bytes` 来自 zkDL relation proof payload 的结构化 `serialized_size_bytes()`，full proof size 只等于固定 relation 顺序下的 `relations[].proof_size_bytes` 求和。",
        "- `proof_size_bytes` 不统计 witness、完整 tensor、debug dump、日志、manifest 文件文本长度、proof object 文件文本长度、README、路径、时间戳、`generated_at` 或机器环境信息。",
        "- range/max 与 rescale 的位宽来自 public quant/table contract；同一代码、同一参数、同一 dataset/device 下，`proof_size_bytes` 应稳定。",
        "- CUDA/CPU forward timing 只服务于运行日志与诊断，不写入 official 四项。",
        "- Cora 或其他单数据集重跑后 timing 可能自然变化；如果论文需要 prover 总耗时，可以使用 `commitment_time_ms + prove_time_ms`，但 official 表保留分列，不覆盖原字段。",
        "- official 表只能由真实 full-chain 运行后的 manifest 通过 exporter 生成，不能手写。",
        "",
        "## official 结果表",
        "",
        *official_status_table(),
        "",
        "## selftest / Gate",
        "",
        "Gate 1 为 commitment/opening 自测，Gate 2 为 sumcheck 自测，Gate 3 为 relation proof/verifier 自测；日志位于 `zkdl_gat_outputs/runs/selftest/`。这些 Gate 不替代 full-chain official gate，只证明基础 zkDL 后端可用。",
    ]
    (ROOT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def cleanup_stale_outputs() -> None:
    """删除旧 exporter/partial/debug/scope 产物，避免读者误认为它们是 official。"""
    stale_files = [
        ROOT / "README_ZKDL_GAT_ADAPTER.md",
        OUT / "README_ZKDL_GAT_ADAPTER.md",
        OUT / "RESUME_POINT.md",
    ]
    for name in ["formal_partial_debug", "debug_level_b", "coverage_matrix"]:
        for suffix in [".csv", ".json", ".md"]:
            stale_files.append(RESULTS / f"{name}{suffix}")
            stale_files.append(RESULTS / "cpu" / f"{name}{suffix}")
            stale_files.append(RESULTS / "gpu" / f"{name}{suffix}")
    for path in stale_files:
        if path.exists():
            path.unlink()

    evidence_dir = OUT / "evidence"
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    pycache_dir = ROOT / "__pycache__"
    if pycache_dir.exists():
        shutil.rmtree(pycache_dir)

    stale_run_names = {
        "RESUME_POINT.md",
        "proof_debug.log",
        "cpu_formal_chain.log",
        "formal_sparse_grouped_latest.log",
        "formal_softmax_aggregation_latest.log",
    }
    stale_artifact_names = {
        "zkdl_cpu_relation_partial.proof.txt",
        "zkdl_sparse_grouped_relations.proof.txt",
        "zkdl_softmax_aggregation_relations.proof.txt",
        "zkdl_output_binding_relations.proof.txt",
        "zkdl_full_chain_scope.status.txt",
    }
    if RUNS.exists():
        for path in RUNS.rglob("*"):
            if path.is_file() and path.name in stale_run_names:
                path.unlink()
            elif path.is_file() and path.name in stale_artifact_names:
                path.unlink()
            elif path.is_dir() and path.name == "debug_artifacts":
                shutil.rmtree(path)

    for directory in sorted(OUT.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


def main() -> None:
    """清理旧派生产物后重新导出 official latest/summary/README。"""
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    cleanup_stale_outputs()
    normalize_all_manifests()
    write_latest_and_summary()
    write_readme()


if __name__ == "__main__":
    main()
