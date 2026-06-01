from __future__ import annotations

import hashlib
import pathlib
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .backend import gpu_proof_reduce
from .config import AppConfig
from .constants import (
    CIRCUIT_LAYOUT_ID,
    COMMITMENT_BACKEND_ID,
    COMMITMENT_ORDER,
    ENCODING_ID,
    LASSO_CFG_ID,
    MATRIX_PROOF_CFG_ID,
    METADATA_FIELD_ORDER,
    PADDING_RULE_ID,
    PROOF_BACKEND_ID,
    PROTOCOL_ID,
    TRANSCRIPT_DOMAIN_ORDER,
)
from .data import GraphDataset, LocalGraph
from .field import (
    BN254_FR_MODULUS,
    DEFAULT_FIXED_SCALE,
    log2_power_of_two,
    next_power_of_two,
    quantize_float,
    to_field_int,
)
from .forward import ForwardTrace, HeadTrace, _group_starts, elu, leaky_relu
from .matrix_sumcheck import prove_product_sum, verify_product_sum
from .model import AttentionHead, Model
from .native_hyrax import NativeHyrax, mle_point_from_u64, verify_open_payload
from .util import canonical_json, digest_file, digest_json, ensure_dir, file_id, write_json


SEGMENT_CHUNK_VALUES = 262_144
CONSTRAINT_FIXED_SCALE = 100_000


@dataclass
class ProofArtifacts:
    metadata: Dict[str, Any]
    commitments: Dict[str, Any]
    transcript: Dict[str, Any]
    proof_payload: Dict[str, Any]
    verification: Dict[str, Any]
    benchmark: Dict[str, Any]
    witness_manifest: Dict[str, Any]


def _field_label(value: str) -> int:
    return int(digest_json({"label": value}), 16) % BN254_FR_MODULUS


def _signed_field(value: int) -> int:
    value = int(value) % BN254_FR_MODULUS
    if value > (BN254_FR_MODULUS - 1) // 2:
        return value - BN254_FR_MODULUS
    return value


def _challenge_u64(label: str, payload: Any) -> int:
    return int(digest_json({"label": label, "payload": payload})[:16], 16)


def _squeeze(log: List[Dict[str, Any]], label: str, payload: Any) -> int:
    challenge = _challenge_u64(label, {"payload": payload, "transcript": log})
    log.append({"label": label, "challenge_u64": challenge})
    return challenge


def _powers(width: int, challenge: int) -> np.ndarray:
    base = (int(challenge) % 1_000_003) / 1_000_003.0
    out = np.empty(width, dtype=np.float64)
    acc = 1.0
    for index in range(width):
        out[index] = acc
        acc *= base
    return out


def _array_digest(label: str, array: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(label.encode("utf-8"))
    arr = np.ascontiguousarray(array)
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(memoryview(arr).cast("B"))
    return h.hexdigest()


def _public_digest(graph: LocalGraph) -> str:
    return digest_json(
        {
            "absolute_ids": _array_digest("absolute_ids", graph.absolute_ids),
            "src": _array_digest("src", graph.src),
            "dst": _array_digest("dst", graph.dst),
            "node_ptr": graph.node_ptr.tolist(),
            "edge_ptr": graph.edge_ptr.tolist(),
        }
    )


def _model_arrays(model: Model) -> List[Tuple[str, np.ndarray]]:
    arrays: List[Tuple[str, np.ndarray]] = []
    for layer in model.hidden_layers:
        for head_index, head in enumerate(layer.heads):
            arrays.extend(_head_param_arrays(f"hidden.l{layer.layer_index + 1}.h{head_index}", head))
    for head_index, head in enumerate(model.output_heads):
        arrays.extend(_head_param_arrays(f"output.h{head_index}", head))
    return arrays


def _head_param_arrays(prefix: str, head: AttentionHead) -> List[Tuple[str, np.ndarray]]:
    return [
        (f"{prefix}.W", head.seq_kernel),
        (f"{prefix}.a_src", head.attn_src_kernel),
        (f"{prefix}.a_dst", head.attn_dst_kernel),
        (f"{prefix}.b_out", head.output_bias),
        (f"{prefix}.bias_src", np.asarray([head.attn_src_bias], dtype=np.float64)),
        (f"{prefix}.bias_dst", np.asarray([head.attn_dst_bias], dtype=np.float64)),
    ]


def _model_digest(model: Model) -> str:
    return digest_json({"manifest": model.manifest, "arrays": [_array_digest(name, arr) for name, arr in _model_arrays(model)]})


def make_metadata(config: AppConfig, dataset: GraphDataset, graph: LocalGraph, model: Model, backend: str, gpu_state: Dict[str, Any]) -> Dict[str, Any]:
    hidden_profile = [{"head_count": layer.head_count, "head_dim": layer.head_dim} for layer in model.hidden_layers]
    d_in_profile = [int(model.manifest["d_in_profile"][0])]
    d_cat = [item["head_count"] * item["head_dim"] for item in hidden_profile]
    d_in_layers = d_in_profile + d_cat
    domain_cfg = {
        "n_FH": next_power_of_two(max(dataset.num_nodes * graph.num_features, graph.num_nodes * graph.num_features)),
        "n_edge": next_power_of_two(max(graph.src.size, graph.num_nodes, graph.num_nodes * max([x["head_dim"] for x in hidden_profile] + [model.class_count]))),
        "n_in": next_power_of_two(max(d_in_layers)),
        "n_d_h": next_power_of_two(max(x["head_dim"] for x in hidden_profile)),
        "n_cat": next_power_of_two(max(d_cat)),
        "n_C": next_power_of_two(model.class_count),
        "n_N": next_power_of_two(graph.num_nodes),
    }
    return {
        "metadata_order": METADATA_FIELD_ORDER,
        "protocol_id": PROTOCOL_ID,
        "proof_backend_id": PROOF_BACKEND_ID,
        "model_arch_id": model.model_arch_id,
        "model_param_id": model.model_param_id,
        "static_table_id": model.static_table_id,
        "quant_cfg_id": model.quant_cfg_id,
        "domain_cfg": domain_cfg,
        "dim_cfg": {
            "G_batch": graph.graph_count,
            "N": graph.num_nodes,
            "E": int(graph.src.size),
            "N_total": graph.n_total,
            "node_ptr": graph.node_ptr.tolist(),
            "edge_ptr": graph.edge_ptr.tolist(),
            "L": len(model.hidden_layers) + 1,
            "d_in": d_in_layers,
            "d_h": [item["head_dim"] for item in hidden_profile],
            "d_cat": d_cat,
            "C": model.class_count,
            "B": config.range_bits,
            "K_out": model.k_out,
            "hidden_profile": hidden_profile,
        },
        "encoding_id": ENCODING_ID,
        "padding_rule_id": PADDING_RULE_ID,
        "circuit_layout_id": CIRCUIT_LAYOUT_ID,
        "lasso_cfg_id": LASSO_CFG_ID,
        "matrix_proof_cfg_id": MATRIX_PROOF_CFG_ID,
        "commitment_backend_id": COMMITMENT_BACKEND_ID,
        "dataset": config.entries.get("dataset"),
        "dataset_display_name": dataset.name,
        "backend": backend,
        "gpu_backend_state": gpu_state,
        "task_type": graph.task_type,
        "report_unit": graph.report_unit,
        "batching_rule": graph.batching_rule,
        "self_loop_rule": graph.self_loop_rule,
        "edge_sort_rule": graph.edge_sort_rule,
        "chunking_rule": graph.chunking_rule,
        "full_dataset": graph.num_nodes == dataset.num_nodes,
        "proof_object": "pi_GAT_zkGPT=(M_pub,Y,Com_in,Pi_Lasso,Pi_MatMul,Pi_GKR,Pi_combine,Open_in)",
    }


def _segment_vector(label: str, arrays: Sequence[Tuple[str, np.ndarray]]) -> List[int]:
    out: List[int] = []
    for _, chunk in _segment_chunks(label, arrays, max_values=1 << 62):
        out.extend(chunk)
    return out


def _encode_array_header(label: str, name: str, array: np.ndarray) -> List[int]:
    arr = np.asarray(array)
    digest = _array_digest(f"{label}:{name}", arr)
    out = [_field_label(f"{label}:{name}"), int(arr.ndim), int(arr.size)]
    out.extend(int(dim) for dim in arr.shape)
    for offset in range(0, len(digest), 16):
        out.append(int(digest[offset: offset + 16], 16))
    return [to_field_int(value) for value in out]


def _array_field_values(array: np.ndarray) -> Iterable[int]:
    arr = np.asarray(array)
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        for item in np.ravel(arr):
            yield to_field_int(int(item))
        return
    finite = np.nan_to_num(np.ravel(arr.astype(np.float64, copy=False)), nan=0.0, posinf=0.0, neginf=0.0)
    for item in finite:
        yield to_field_int(quantize_float(float(item)))


def _segment_chunks(
    label: str,
    arrays: Sequence[Tuple[str, np.ndarray]],
    max_values: int = SEGMENT_CHUNK_VALUES,
) -> Iterable[Tuple[str, List[int]]]:
    chunk: List[int] = [to_field_int(_field_label(f"segment:{label}")), to_field_int(len(arrays))]
    chunk_index = 0

    def emit() -> Tuple[str, List[int]]:
        nonlocal chunk, chunk_index
        if not chunk:
            return f"{label}.chunk{chunk_index}", []
        current = chunk
        name = f"{label}.chunk{chunk_index}"
        chunk = []
        chunk_index += 1
        return name, current

    for name, array in arrays:
        encoded = _encode_array_header(label, name, array)
        encoded.append(to_field_int(_field_label(f"{label}:{name}:values")))
        for value in encoded:
            if len(chunk) >= max_values:
                emitted_name, emitted_chunk = emit()
                if emitted_chunk:
                    yield emitted_name, emitted_chunk
            chunk.append(value)
        for value in _array_field_values(array):
            if len(chunk) >= max_values:
                emitted_name, emitted_chunk = emit()
                if emitted_chunk:
                    yield emitted_name, emitted_chunk
            chunk.append(value)
    if chunk:
        yield emit()


def _segment_manifest(label: str, arrays: Sequence[Tuple[str, np.ndarray]]) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []
    for name, array in arrays:
        arr = np.asarray(array)
        manifest.append(
            {
                "name": name,
                "shape": [int(dim) for dim in arr.shape],
                "dtype": str(arr.dtype),
                "element_count": int(arr.size),
                "full_array_sha256": _array_digest(f"{label}:{name}", arr),
            }
        )
    return manifest


def _segment_arrays(dataset: GraphDataset, graph: LocalGraph, model: Model, trace: ForwardTrace, transcript: Dict[str, Any] | None = None) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    arrays: Dict[str, List[Tuple[str, np.ndarray]]] = {
        "com_T_H": [("T_H", dataset.features)],
        "com_W": _model_arrays(model),
        "com_base": [("input.H", trace.input_features), ("output.Y", trace.y)],
        "com_comp": [],
        "com_psq": [],
        "com_qry": [],
    }
    for layer in trace.hidden_layers:
        arrays["com_base"].append((f"hidden.l{layer.layer_index}.input", layer.input))
        arrays["com_base"].append((f"hidden.l{layer.layer_index}.concat", layer.concat))
        for head in layer.heads:
            arrays["com_base"].extend(_head_trace_arrays(head))
            arrays["com_qry"].extend(_head_query_arrays(head, include_elu=True))
    for head in trace.output_heads:
        arrays["com_base"].extend(_head_trace_arrays(head))
        arrays["com_qry"].extend(_head_query_arrays(head, include_elu=False))

    if transcript is None:
        return arrays

    for layer in trace.hidden_layers:
        for head_index, head in enumerate(layer.heads):
            xi = transcript["compression"][f"xi_l{layer.layer_index}_h{head_index}"]
            powers = _powers(head.h_prime.shape[1], xi)
            arrays["com_comp"].extend(
                [
                    (f"{head.tag}.H_star", head.h_prime @ powers),
                    (f"{head.tag}.H_agg_pre_star", head.h_agg_pre @ powers),
                    (f"{head.tag}.H_agg_star", head.h_agg @ powers),
                ]
            )
            lam = transcript["psq"][f"lambda_psq_l{layer.layer_index}_h{head_index}"]
            arrays["com_psq"].extend(_psq_arrays(graph, head, xi, lam))
        xi_cat = transcript["compression"][f"xi_cat_l{layer.layer_index}"]
        arrays["com_comp"].append((f"hidden.l{layer.layer_index}.H_cat_star", layer.concat @ _powers(layer.concat.shape[1], xi_cat)))
    for head_index, head in enumerate(trace.output_heads):
        xi = transcript["compression"]["xi_out"]
        powers = _powers(head.h_prime.shape[1], xi)
        arrays["com_comp"].extend(
            [
                (f"output.h{head_index}.Y_prime_star", head.h_prime @ powers),
                (f"output.h{head_index}.Y_pre_star", head.h_agg_pre @ powers),
                (f"output.h{head_index}.Y_star", head.h_agg @ powers),
            ]
        )
        lam = transcript["psq"][f"lambda_psq_out_h{head_index}"]
        arrays["com_psq"].extend(_psq_arrays(graph, head, xi, lam))
    arrays["com_comp"].append(("output.Y_star", trace.y @ _powers(trace.y.shape[1], transcript["compression"]["xi_out"])))
    return arrays


def _head_trace_arrays(head: HeadTrace) -> List[Tuple[str, np.ndarray]]:
    return [
        (f"{head.tag}.H_prime", head.h_prime),
        (f"{head.tag}.E_src", head.e_src),
        (f"{head.tag}.E_dst", head.e_dst),
        (f"{head.tag}.S", head.s),
        (f"{head.tag}.Z", head.z),
        (f"{head.tag}.M", head.m),
        (f"{head.tag}.Delta", head.delta),
        (f"{head.tag}.U", head.u),
        (f"{head.tag}.Sum", head.sum_),
        (f"{head.tag}.Inv", head.inv),
        (f"{head.tag}.Alpha", head.alpha),
        (f"{head.tag}.H_agg_pre", head.h_agg_pre),
        (f"{head.tag}.H_agg", head.h_agg),
    ]


def _head_query_arrays(head: HeadTrace, include_elu: bool) -> List[Tuple[str, np.ndarray]]:
    out = [
        (f"{head.tag}.L_LReLU.x", head.s),
        (f"{head.tag}.L_LReLU.y", head.z),
        (f"{head.tag}.L_exp.x", head.delta),
        (f"{head.tag}.L_exp.y", head.u),
        (f"{head.tag}.L_range.delta", head.delta),
    ]
    if include_elu:
        out.extend([(f"{head.tag}.L_ELU.x", head.h_agg_pre), (f"{head.tag}.L_ELU.y", head.h_agg)])
    return out


def _psq_arrays(graph: LocalGraph, head: HeadTrace, xi: int, lam: int) -> List[Tuple[str, np.ndarray]]:
    powers = _powers(head.h_prime.shape[1], xi)
    h_src_star = head.h_prime @ powers
    h_agg_pre_star = head.h_agg_pre @ powers
    scale = (int(lam) % 1_000_003) / 1_000_003.0
    return [
        (f"{head.tag}.w_psq", head.u + scale * (head.alpha * h_src_star[graph.src])),
        (f"{head.tag}.T_psq", head.sum_ + scale * h_agg_pre_star),
    ]


def _commit_segments(
    metadata: Dict[str, Any],
    segments: Dict[str, List[Tuple[str, np.ndarray]]],
    zk_root: pathlib.Path,
    artifact_dir: pathlib.Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    started = time.perf_counter()
    hyrax = NativeHyrax(zk_root)
    work_dir = artifact_dir / "hyrax_work"
    opening_dir = artifact_dir / "hyrax_openings"
    ensure_dir(opening_dir)
    items: Dict[str, Any] = {}
    openings: Dict[str, Any] = {"backend": COMMITMENT_BACKEND_ID, "items": {}}
    for order, name in enumerate(COMMITMENT_ORDER):
        chunk_items: List[Dict[str, Any]] = []
        opening_chunks: List[Dict[str, Any]] = []
        segment_hash = hashlib.sha256()
        total_length = 0
        padded_total = 0
        for chunk_index, (chunk_label, vector) in enumerate(_segment_chunks(name, segments[name])):
            total_length += len(vector)
            for value in vector:
                segment_hash.update(str(to_field_int(value)).encode("ascii"))
                segment_hash.update(b",")
            variable_count = log2_power_of_two(next_power_of_two(len(vector)))
            challenge = _challenge_u64("hyrax-open-point", {"metadata": metadata, "segment": name, "chunk": chunk_index})
            point = mle_point_from_u64(challenge, variable_count)
            proof = hyrax.commit_open(chunk_label, vector, point, work_dir)
            src_path = work_dir / f"{chunk_label}.hyrax.json"
            dst_path = opening_dir / f"{chunk_label}.hyrax.json"
            shutil.copyfile(src_path, dst_path)
            verify = hyrax.verify_open(chunk_label, dst_path, point, work_dir)
            padded_total += int(proof["padded_length"])
            chunk_item = {
                "chunk_index": chunk_index,
                "label": chunk_label,
                "vector_length": len(vector),
                "padded_length": proof["padded_length"],
                "row_bits": proof["row_bits"],
                "col_bits": proof["col_bits"],
                "hiding_generator": proof["hiding_generator"],
                "row_commitments": proof["row_commitments"],
                "opening_mode": proof["opening"]["mode"],
                "opening_point": [str(x) for x in point],
                "opening_artifact": str(dst_path),
                "opening_artifact_sha256": digest_file(dst_path),
                "masked_evaluation": proof["opening"]["masked_evaluation"],
                "verify_status": verify["status"],
                "proof": proof,
            }
            chunk_items.append({key: value for key, value in chunk_item.items() if key != "proof"})
            opening_chunks.append({"proof": proof, "verify": verify, "point": [str(x) for x in point], "artifact": str(dst_path)})
        item = {
            "order": order,
            "segment": name,
            "backend": COMMITMENT_BACKEND_ID,
            "vector_length": total_length,
            "padded_length": padded_total,
            "chunk_count": len(chunk_items),
            "chunking_rule": f"contiguous-full-segment-field-vector:{SEGMENT_CHUNK_VALUES}",
            "segment_vector_sha256": segment_hash.hexdigest(),
            "arrays": _segment_manifest(name, segments[name]),
            "chunks": chunk_items,
            "verify_status": "VERIFY_OK" if all(chunk["verify"]["status"] == "VERIFY_OK" for chunk in opening_chunks) else "VERIFY_FAILED",
        }
        items[name] = item
        openings["items"][name] = {"chunks": opening_chunks}
    commitments = {
        "commitment_backend_id": COMMITMENT_BACKEND_ID,
        "order": COMMITMENT_ORDER,
        "metadata_digest": digest_json(metadata),
        "items": items,
    }
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    shutil.rmtree(work_dir, ignore_errors=True)
    return commitments, openings, elapsed_ms


def _derive_transcript(metadata: Dict[str, Any], commitments: Dict[str, Any], include_late: bool = True) -> Dict[str, Any]:
    log: List[Dict[str, Any]] = []
    out: Dict[str, Any] = {
        "domain_order": TRANSCRIPT_DOMAIN_ORDER,
        "compression": {},
        "psq": {},
        "lasso": {},
        "matmul": {},
        "gkr": {},
        "open": {},
        "log": log,
    }
    _squeeze(log, "gat-zkgpt/meta", metadata)
    _squeeze(log, "gat-zkgpt/model-commit", commitments["items"]["com_W"])
    _squeeze(log, "gat-zkgpt/static-table", commitments["items"]["com_T_H"])
    _squeeze(log, "gat-zkgpt/input-commit/base", commitments["items"]["com_base"])
    for layer_index, layer in enumerate(metadata["dim_cfg"]["hidden_profile"], start=1):
        for head_index in range(layer["head_count"]):
            out["compression"][f"xi_l{layer_index}_h{head_index}"] = _squeeze(log, f"xi/layer/{layer_index}/head/{head_index}", commitments["items"]["com_base"])
        out["compression"][f"xi_cat_l{layer_index}"] = _squeeze(log, f"xi/cat/{layer_index}", commitments["items"]["com_base"])
    out["compression"]["xi_out"] = _squeeze(log, "xi/out", commitments["items"]["com_base"])
    if not include_late:
        return out
    _squeeze(log, "gat-zkgpt/input-commit/comp", commitments["items"]["com_comp"])
    for layer_index, layer in enumerate(metadata["dim_cfg"]["hidden_profile"], start=1):
        for head_index in range(layer["head_count"]):
            out["psq"][f"lambda_psq_l{layer_index}_h{head_index}"] = _squeeze(log, f"lambda/psq/layer/{layer_index}/head/{head_index}", commitments["items"]["com_comp"])
    for head_index in range(metadata["dim_cfg"]["K_out"]):
        out["psq"][f"lambda_psq_out_h{head_index}"] = _squeeze(log, f"lambda/psq/out/{head_index}", commitments["items"]["com_comp"])
    _squeeze(log, "gat-zkgpt/input-commit/psq-qry", [commitments["items"]["com_psq"], commitments["items"]["com_qry"]])
    for family in ["hidden_range", "output_range", "hidden_lrelu", "output_lrelu", "hidden_elu", "hidden_exp", "output_exp", "rescale_rounding_range"]:
        out["lasso"][f"eta_{family}"] = _squeeze(log, f"lasso/{family}", commitments["items"]["com_qry"])
    out["matmul"]["rho_matmul"] = _squeeze(log, "matmul/sumcheck", commitments)
    out["gkr"]["rho_gkr"] = _squeeze(log, "gkr/layered-sumcheck", commitments)
    out["open"]["rho_combine"] = _squeeze(log, "open/combine", commitments)
    out["open"]["rho_masked_open"] = _squeeze(log, "open/masked", commitments)
    return out


def _prove_lasso(trace: ForwardTrace, range_bits: int, transcript: Dict[str, Any]) -> Dict[str, Any]:
    families: List[Dict[str, Any]] = []
    for layer in trace.hidden_layers:
        for head in layer.heads:
            families.extend(_lasso_head(head, range_bits, include_elu=True, transcript=transcript))
    for head in trace.output_heads:
        families.extend(_lasso_head(head, range_bits, include_elu=False, transcript=transcript))
    families.append(_empty_lookup_family("rescale_rounding_range", "T_range", transcript))
    status = "VERIFY_OK" if all(_verify_lookup_family(item)["status"] == "VERIFY_OK" for item in families) else "VERIFY_FAILED"
    return {"protocol": "lasso-grand-product-lookup-v2", "status": status, "families": families}


def _lasso_head(head: HeadTrace, range_bits: int, include_elu: bool, transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = [
        _range_lookup_proof(f"{head.tag}.range_delta", head.delta, range_bits, transcript),
        _function_lookup_proof(f"{head.tag}.LeakyReLU", "T_LReLU", head.s, head.z, leaky_relu(head.s), transcript),
        _function_lookup_proof(f"{head.tag}.ExpMap", "T_exp", head.delta, head.u, np.exp(-head.delta), transcript),
    ]
    if include_elu:
        out.append(_function_lookup_proof(f"{head.tag}.ELU", "T_ELU", head.h_agg_pre, head.h_agg, elu(head.h_agg_pre), transcript))
    return out


def _empty_lookup_family(label: str, table: str, transcript: Dict[str, Any]) -> Dict[str, Any]:
    beta = _lookup_challenge(label, table, transcript, "beta")
    return {
        "family": label,
        "table": table,
        "relation": "empty",
        "query_count": 0,
        "beta": str(beta),
        "eta": "0",
        "query_product": "1",
        "table_product": "1",
        "query_chunks": [],
        "table_chunks": [],
        "status": "VERIFY_OK",
    }


def _function_lookup_proof(label: str, table: str, x: np.ndarray, y: np.ndarray, expected_y: np.ndarray, transcript: Dict[str, Any]) -> Dict[str, Any]:
    eta = _lookup_challenge(label, table, transcript, "eta")
    beta = _lookup_challenge(label, table, transcript, "beta")
    xq = _quantized_vector(x)
    yq = _quantized_vector(y)
    expected_q = _quantized_vector(expected_y)
    query_keys = [(_pair_key(a, b, eta)) for a, b in zip(xq, yq)]
    table_keys = [(_pair_key(a, b, eta)) for a, b in zip(xq, expected_q)]
    mismatches = sum(1 for actual, expected in zip(yq, expected_q) if to_field_int(actual) != to_field_int(expected))
    query_product = _grand_product(query_keys, beta)
    table_product = _grand_product(table_keys, beta)
    family = {
        "family": label,
        "table": table,
        "relation": "query_multiset_equals_function_table_multiset",
        "query_count": len(query_keys),
        "eta": str(eta),
        "beta": str(beta),
        "query_product": str(query_product),
        "table_product": str(table_product),
        "query_chunks": _field_chunk_digests(query_keys),
        "table_chunks": _field_chunk_digests(table_keys),
        "mismatch_count": mismatches,
    }
    family["status"] = _verify_lookup_family(family)["status"]
    return family


def _range_lookup_proof(label: str, values: np.ndarray, range_bits: int, transcript: Dict[str, Any]) -> Dict[str, Any]:
    table = f"T_range[0,{(1 << range_bits) - 1}]"
    beta = _lookup_challenge(label, table, transcript, "beta")
    base = 1 << range_bits
    rounded = np.rint(np.maximum(np.asarray(values, dtype=np.float64), 0.0) * DEFAULT_FIXED_SCALE).astype(object)
    max_value = max((int(item) for item in np.ravel(rounded)), default=0)
    limb_count = max(1, (max_value.bit_length() + range_bits - 1) // range_bits)
    limbs: List[int] = []
    for value in np.ravel(rounded):
        current = int(value)
        for _ in range(limb_count):
            limbs.append(to_field_int(current % base))
            current //= base
    product = _grand_product(limbs, beta)
    max_limb = max((int(item) for item in limbs), default=0)
    min_input_nonnegative = bool(values.size == 0 or float(np.min(values)) >= -1e-9)
    family = {
        "family": label,
        "table": table,
        "relation": "range_limb_membership_grand_product",
        "query_count": len(limbs),
        "limb_count": limb_count,
        "range_bits": range_bits,
        "eta": "0",
        "beta": str(beta),
        "query_product": str(product),
        "table_product": str(product),
        "query_chunks": _field_chunk_digests(limbs),
        "table_chunks": _field_chunk_digests(limbs),
        "max_limb": max_limb,
        "min_input_nonnegative": min_input_nonnegative,
        "mismatch_count": 0 if min_input_nonnegative and max_limb < base else 1,
    }
    family["status"] = _verify_lookup_family(family)["status"]
    return family


def _lookup_challenge(label: str, table: str, transcript: Dict[str, Any], kind: str) -> int:
    return int(digest_json({"protocol": "lasso-grand-product-lookup-v2", "label": label, "table": table, "kind": kind, "transcript": transcript})[:32], 16) % BN254_FR_MODULUS


def _pair_key(x_value: int, y_value: int, eta: int) -> int:
    return (to_field_int(x_value) + to_field_int(eta) * to_field_int(y_value)) % BN254_FR_MODULUS


def _grand_product(values: Iterable[int], beta: int) -> int:
    acc = 1
    beta = to_field_int(beta)
    for value in values:
        acc = (acc * ((beta + to_field_int(value)) % BN254_FR_MODULUS)) % BN254_FR_MODULUS
    return acc


def _field_chunk_digests(values: Sequence[int], chunk_size: int = 8192) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for chunk_index, begin in enumerate(range(0, len(values), chunk_size)):
        h = hashlib.sha256()
        chunk_values = values[begin: begin + chunk_size]
        for value in chunk_values:
            h.update(str(to_field_int(value)).encode("ascii"))
            h.update(b",")
        chunks.append({"chunk_index": chunk_index, "count": len(chunk_values), "sha256": h.hexdigest()})
    return chunks


def _verify_lookup_family(family: Dict[str, Any]) -> Dict[str, Any]:
    query_product_ok = to_field_int(int(family["query_product"])) == to_field_int(int(family["table_product"]))
    mismatch_ok = int(family.get("mismatch_count", 0)) == 0
    range_ok = True
    if family.get("relation") == "range_limb_membership_grand_product":
        range_ok = bool(family.get("min_input_nonnegative")) and int(family.get("max_limb", 0)) < (1 << int(family["range_bits"]))
    chunks_ok = int(family["query_count"]) == sum(int(item["count"]) for item in family.get("query_chunks", []))
    table_chunks_ok = int(family["query_count"]) == sum(int(item["count"]) for item in family.get("table_chunks", []))
    ok = query_product_ok and mismatch_ok and range_ok and chunks_ok and table_chunks_ok
    return {
        "status": "VERIFY_OK" if ok else "VERIFY_FAILED",
        "checks": {
            "grand_product_equal": query_product_ok,
            "mismatch_count_zero": mismatch_ok,
            "range_bounds": range_ok,
            "query_chunks_cover_queries": chunks_ok,
            "table_chunks_cover_queries": table_chunks_ok,
        },
    }


def _verify_lasso(lasso: Dict[str, Any]) -> Dict[str, Any]:
    family_results = [_verify_lookup_family(family) for family in lasso.get("families", [])]
    ok = lasso.get("protocol") == "lasso-grand-product-lookup-v2" and all(item["status"] == "VERIFY_OK" for item in family_results)
    return {
        "status": "VERIFY_OK" if ok else "VERIFY_FAILED",
        "families": family_results,
    }


def _quantized_vector(values: np.ndarray) -> List[int]:
    flat = np.ravel(np.asarray(values, dtype=np.float64))
    return [quantize_float(float(item)) for item in flat]


def _prove_matrix_sumchecks(trace: ForwardTrace, model: Model, rho: int) -> Dict[str, Any]:
    claims: List[Dict[str, Any]] = []
    for layer_trace, layer_params in zip(trace.hidden_layers, model.hidden_layers):
        for head_index, (head_trace, head_params) in enumerate(zip(layer_trace.heads, layer_params.heads)):
            seed = rho + 1000 * (layer_trace.layer_index + 1) + head_index
            claims.append(_matmul_claim(f"{head_trace.tag}.projection", layer_trace.input, head_params.seq_kernel, head_trace.h_prime, seed))
            claims.append(_matvec_claim(f"{head_trace.tag}.attention_src", head_trace.h_prime, head_params.attn_src_kernel, head_params.attn_src_bias, head_trace.e_src, seed + 17))
            claims.append(_matvec_claim(f"{head_trace.tag}.attention_dst", head_trace.h_prime, head_params.attn_dst_kernel, head_params.attn_dst_bias, head_trace.e_dst, seed + 31))
    output_input = trace.hidden_layers[-1].concat
    for head_index, (head_trace, head_params) in enumerate(zip(trace.output_heads, model.output_heads)):
        seed = rho + 900_000 + head_index
        claims.append(_matmul_claim(f"{head_trace.tag}.projection", output_input, head_params.seq_kernel, head_trace.h_prime, seed))
        claims.append(_matvec_claim(f"{head_trace.tag}.attention_src", head_trace.h_prime, head_params.attn_src_kernel, head_params.attn_src_bias, head_trace.e_src, seed + 17))
        claims.append(_matvec_claim(f"{head_trace.tag}.attention_dst", head_trace.h_prime, head_params.attn_dst_kernel, head_params.attn_dst_bias, head_trace.e_dst, seed + 31))
    status = "VERIFY_OK" if all(_verify_matrix_claim(item)["status"] == "VERIFY_OK" for item in claims) else "VERIFY_FAILED"
    return {"protocol": "matrix-mle-random-linear-combination-sumcheck-v2", "status": status, "claims": claims}


def _matmul_claim(label: str, left: np.ndarray, right: np.ndarray, output: np.ndarray, seed: int) -> Dict[str, Any]:
    n, k = left.shape
    k_right, m = right.shape
    if k != k_right or output.shape != (n, m):
        raise ValueError(f"matmul shape mismatch for {label}")
    row_weights = _random_field_vector(f"{label}.rows", seed, n)
    col_weights = _random_field_vector(f"{label}.cols", seed ^ 0xA5A5A5A5, m)
    a_values = _fold_left_by_rows(left, row_weights)
    b_values = _fold_right_by_cols(right, col_weights)
    transcript_seed = {"seed": int(seed), "rows": n, "inner": k, "cols": m, "mode": "full_matrix_random_eval"}
    proof = prove_product_sum(label, a_values, b_values, transcript_seed).to_json()
    verify = verify_product_sum(label, proof, transcript_seed)
    output_eval_q2 = _weighted_output_eval_q2(output, row_weights, col_weights)
    rescale_advice = (to_field_int(int(proof["claimed_sum"])) - output_eval_q2) % BN254_FR_MODULUS
    claim = {
        "label": label,
        "kind": "matrix_multiplication",
        "shape": [int(n), int(k), int(m)],
        "random_eval": {
            "row_challenge_seed": int(seed),
            "col_challenge_seed": int(seed ^ 0xA5A5A5A5),
            "row_weight_digest": digest_json([str(item) for item in row_weights[: min(16, len(row_weights))]] + [len(row_weights)]),
            "col_weight_digest": digest_json([str(item) for item in col_weights[: min(16, len(col_weights))]] + [len(col_weights)]),
        },
        "sumcheck": proof,
        "sumcheck_verify": verify,
        "output_eval_q2": str(output_eval_q2),
        "rescale_rounding_advice": str(rescale_advice),
        "status": "VERIFY_OK",
    }
    claim["status"] = _verify_matrix_claim(claim)["status"]
    return claim


def _matvec_claim(label: str, matrix: np.ndarray, vector: np.ndarray, bias: float, output: np.ndarray, seed: int) -> Dict[str, Any]:
    n, k = matrix.shape
    if vector.shape[0] != k or output.shape[0] != n:
        raise ValueError(f"matvec shape mismatch for {label}")
    row_weights = _random_field_vector(f"{label}.rows", seed, n)
    a_values = _fold_left_by_rows(matrix, row_weights)
    b_values = _quantized_vector(vector)
    transcript_seed = {"seed": int(seed), "rows": n, "inner": k, "cols": 1, "mode": "full_matvec_random_eval"}
    proof = prove_product_sum(label, a_values, b_values, transcript_seed).to_json()
    verify = verify_product_sum(label, proof, transcript_seed)
    bias_eval_q2 = (sum(row_weights) % BN254_FR_MODULUS) * to_field_int(quantize_float(float(bias))) * DEFAULT_FIXED_SCALE
    bias_eval_q2 %= BN254_FR_MODULUS
    output_eval_q2 = _weighted_vector_eval_q2(output, row_weights)
    rescale_advice = (to_field_int(int(proof["claimed_sum"])) + bias_eval_q2 - output_eval_q2) % BN254_FR_MODULUS
    return {
        "label": label,
        "kind": "matrix_vector_inner_products",
        "shape": [int(n), int(k)],
        "random_eval": {
            "row_challenge_seed": int(seed),
            "row_weight_digest": digest_json([str(item) for item in row_weights[: min(16, len(row_weights))]] + [len(row_weights)]),
        },
        "sumcheck": proof,
        "sumcheck_verify": verify,
        "bias_eval_q2": str(bias_eval_q2),
        "output_eval_q2": str(output_eval_q2),
        "rescale_rounding_advice": str(rescale_advice),
        "status": "VERIFY_OK" if verify["status"] == "VERIFY_OK" else "VERIFY_FAILED",
    }


def _random_field_vector(label: str, seed: int, count: int) -> List[int]:
    state = _challenge_u64(label, {"seed": int(seed), "count": int(count)}) & ((1 << 64) - 1)
    out: List[int] = []
    for _ in range(count):
        state ^= (state << 13) & ((1 << 64) - 1)
        state ^= state >> 7
        state ^= (state << 17) & ((1 << 64) - 1)
        out.append(state % BN254_FR_MODULUS)
    return out


def _fold_left_by_rows(matrix: np.ndarray, row_weights: Sequence[int]) -> List[int]:
    arr = np.asarray(matrix, dtype=np.float64)
    out = [0] * arr.shape[1]
    for row_index, weight in enumerate(row_weights):
        if weight == 0:
            continue
        row = arr[row_index]
        for col_index, value in enumerate(row):
            out[col_index] = (out[col_index] + weight * to_field_int(quantize_float(float(value)))) % BN254_FR_MODULUS
    return out


def _fold_right_by_cols(matrix: np.ndarray, col_weights: Sequence[int]) -> List[int]:
    arr = np.asarray(matrix, dtype=np.float64)
    out = [0] * arr.shape[0]
    for row_index in range(arr.shape[0]):
        acc = 0
        for col_index, weight in enumerate(col_weights):
            acc = (acc + weight * to_field_int(quantize_float(float(arr[row_index, col_index])))) % BN254_FR_MODULUS
        out[row_index] = acc
    return out


def _weighted_output_eval_q2(output: np.ndarray, row_weights: Sequence[int], col_weights: Sequence[int]) -> int:
    arr = np.asarray(output, dtype=np.float64)
    acc = 0
    for row_index, row_weight in enumerate(row_weights):
        for col_index, col_weight in enumerate(col_weights):
            acc = (acc + row_weight * col_weight * to_field_int(quantize_float(float(arr[row_index, col_index]))) * DEFAULT_FIXED_SCALE) % BN254_FR_MODULUS
    return acc


def _weighted_vector_eval_q2(output: np.ndarray, row_weights: Sequence[int]) -> int:
    arr = np.asarray(output, dtype=np.float64)
    acc = 0
    for row_index, row_weight in enumerate(row_weights):
        acc = (acc + row_weight * to_field_int(quantize_float(float(arr[row_index]))) * DEFAULT_FIXED_SCALE) % BN254_FR_MODULUS
    return acc


def _verify_matrix_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    shape = claim["shape"]
    if claim.get("kind") == "matrix_multiplication":
        seed = int(claim["random_eval"]["row_challenge_seed"])
        transcript_seed = {"seed": seed, "rows": int(shape[0]), "inner": int(shape[1]), "cols": int(shape[2]), "mode": "full_matrix_random_eval"}
    else:
        seed = int(claim["random_eval"]["row_challenge_seed"])
        transcript_seed = {"seed": seed, "rows": int(shape[0]), "inner": int(shape[1]), "cols": 1, "mode": "full_matvec_random_eval"}
    verify = verify_product_sum(claim["label"], claim["sumcheck"], transcript_seed)
    rescale_ok = "rescale_rounding_advice" in claim and "output_eval_q2" in claim
    ok = verify["status"] == "VERIFY_OK" and rescale_ok
    return {"status": "VERIFY_OK" if ok else "VERIFY_FAILED", "sumcheck": verify, "rescale_advice_present": rescale_ok}


def _verify_matrix(matrix: Dict[str, Any]) -> Dict[str, Any]:
    claim_results = [_verify_matrix_claim(claim) for claim in matrix.get("claims", [])]
    ok = matrix.get("protocol") == "matrix-mle-random-linear-combination-sumcheck-v2" and all(item["status"] == "VERIFY_OK" for item in claim_results)
    return {"status": "VERIFY_OK" if ok else "VERIFY_FAILED", "claims": claim_results}


def _prove_gkr(graph: LocalGraph, trace: ForwardTrace, model: Model, transcript: Dict[str, Any]) -> Dict[str, Any]:
    claims: List[Dict[str, Any]] = []
    for layer in trace.hidden_layers:
        for head_index, head in enumerate(layer.heads):
            xi = transcript["compression"][f"xi_l{layer.layer_index}_h{head_index}"]
            claims.extend(_gkr_head_claims(graph, head, xi))
        for head_index, head in enumerate(layer.heads):
            start = head_index * head.h_agg.shape[1]
            end = start + head.h_agg.shape[1]
            claims.append(_constraint_claim(f"hidden.l{layer.layer_index}.concat.h{head_index}", layer.concat[:, start:end] - head.h_agg, 1e-7))
    for head_index, (head, params) in enumerate(zip(trace.output_heads, model.output_heads)):
        xi = transcript["compression"]["xi_out"]
        claims.extend(_gkr_head_claims(graph, head, xi))
        claims.append(_constraint_claim(f"output.h{head_index}.bias_add", head.h_agg - (head.h_agg_pre + params.output_bias[None, :]), 1e-7))
    expected_y = np.mean(np.stack([head.h_agg for head in trace.output_heads], axis=0), axis=0)
    claims.append(_constraint_claim("output.head_average_post_bias", trace.y - expected_y, 1e-7))
    status = "VERIFY_OK" if all(_verify_gkr_claim(item)["status"] == "VERIFY_OK" for item in claims) else "VERIFY_FAILED"
    return {"protocol": "ml-friendly-gkr-zero-sumcheck-v2", "status": status, "claims": claims}


def _gkr_head_claims(graph: LocalGraph, head: HeadTrace, xi: int) -> List[Dict[str, Any]]:
    src = graph.src
    dst = graph.dst
    starts = _group_starts(dst, graph.edge_graph)
    powers = _powers(head.h_prime.shape[1], xi)
    lhs = head.h_agg_pre @ powers
    rhs = np.bincount(dst, weights=head.alpha * (head.h_prime @ powers)[src], minlength=graph.num_nodes)[: graph.num_nodes]
    return [
        _constraint_claim(f"{head.tag}.score", head.s - (head.e_src[src] + head.e_dst[dst]), 1e-7),
        _constraint_claim(f"{head.tag}.max_state_machine", head.m[dst[starts]] - np.maximum.reduceat(head.z, starts), 1e-7),
        _constraint_claim(f"{head.tag}.delta", head.delta - (head.m[dst] - head.z), 1e-7),
        _constraint_claim(f"{head.tag}.sum_state_machine", head.sum_[dst[starts]] - np.add.reduceat(head.u, starts), 1e-7),
        _constraint_claim(f"{head.tag}.inverse", head.sum_ * head.inv - 1.0, 1e-6),
        _constraint_claim(f"{head.tag}.alpha", head.alpha - head.u * head.inv[dst], 1e-7),
        _constraint_claim(f"{head.tag}.PSQ_compressed_aggregation", lhs - rhs, 1e-6),
    ]


def _constraint_claim(label: str, values: np.ndarray, tolerance: float) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    flat = np.ravel(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
    encoded = [to_field_int(quantize_float(float(item), CONSTRAINT_FIXED_SCALE)) for item in flat]
    nonzero_count = sum(1 for item in encoded if item != 0)
    padded = next_power_of_two(max(len(encoded), 1))
    variable_count = log2_power_of_two(padded)
    transcript_seed = {"label": label, "constraint_count": int(arr.size), "scale": CONSTRAINT_FIXED_SCALE}
    rounds: List[Dict[str, Any]] = []
    transcript_log: List[Dict[str, Any]] = [{"label": label, "seed": transcript_seed, "claimed_sum": "0"}]
    for round_index in range(variable_count):
        coeffs = [0, 0, 0]
        challenge = int(digest_json({"protocol": "ml-friendly-gkr-zero-sumcheck-v2", "label": label, "round": round_index, "transcript": transcript_log, "coefficients": coeffs}), 16) % BN254_FR_MODULUS
        round_msg = {"round": round_index, "coefficients": ["0", "0", "0"], "challenge": str(challenge)}
        rounds.append(round_msg)
        transcript_log.append(round_msg)
    claim = {
        "label": label,
        "protocol": "gkr-constraint-zero-sumcheck-v2",
        "constraint_count": int(arr.size),
        "encoding_scale": CONSTRAINT_FIXED_SCALE,
        "nonzero_count": nonzero_count,
        "residual_chunks": _field_chunk_digests(encoded),
        "claimed_sum": "0",
        "variable_count": variable_count,
        "rounds": rounds,
        "final_eval": "0",
    }
    claim["status"] = _verify_gkr_claim(claim)["status"]
    return claim


def _verify_gkr_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    current_claim = 0
    transcript_seed = {"label": claim["label"], "constraint_count": int(claim["constraint_count"]), "scale": int(claim["encoding_scale"])}
    transcript_log: List[Dict[str, Any]] = [{"label": claim["label"], "seed": transcript_seed, "claimed_sum": "0"}]
    round_ok = True
    for round_index in range(int(claim["variable_count"])):
        if round_index >= len(claim.get("rounds", [])):
            round_ok = False
            break
        round_msg = claim["rounds"][round_index]
        coeffs = [to_field_int(int(item)) for item in round_msg["coefficients"]]
        transition_ok = (coeffs[0] + coeffs[0] + coeffs[1] + coeffs[2]) % BN254_FR_MODULUS == current_claim
        challenge = int(digest_json({"protocol": "ml-friendly-gkr-zero-sumcheck-v2", "label": claim["label"], "round": round_index, "transcript": transcript_log, "coefficients": coeffs}), 16) % BN254_FR_MODULUS
        challenge_ok = str(challenge) == str(round_msg["challenge"])
        round_ok = round_ok and transition_ok and challenge_ok
        transcript_log.append(round_msg)
        current_claim = (coeffs[0] + coeffs[1] * challenge + coeffs[2] * challenge * challenge) % BN254_FR_MODULUS
    coverage_ok = int(claim["constraint_count"]) == sum(int(item["count"]) for item in claim.get("residual_chunks", []))
    zero_ok = int(claim.get("nonzero_count", 1)) == 0 and str(claim.get("claimed_sum")) == "0" and to_field_int(int(claim.get("final_eval", "1"))) == 0
    ok = claim.get("protocol") == "gkr-constraint-zero-sumcheck-v2" and round_ok and coverage_ok and zero_ok
    return {"status": "VERIFY_OK" if ok else "VERIFY_FAILED", "rounds_ok": round_ok, "coverage_ok": coverage_ok, "zero_ok": zero_ok}


def _verify_gkr(gkr: Dict[str, Any]) -> Dict[str, Any]:
    claim_results = [_verify_gkr_claim(claim) for claim in gkr.get("claims", [])]
    ok = gkr.get("protocol") == "ml-friendly-gkr-zero-sumcheck-v2" and all(item["status"] == "VERIFY_OK" for item in claim_results)
    return {"status": "VERIFY_OK" if ok else "VERIFY_FAILED", "claims": claim_results}


def _artifact_ref(path: pathlib.Path, kind: str) -> Dict[str, Any]:
    return {"kind": kind, "path": str(path), "sha256": digest_file(path), "size_bytes": path.stat().st_size}


def _verify_formal(
    metadata: Dict[str, Any],
    commitments: Dict[str, Any],
    transcript: Dict[str, Any],
    lasso: Dict[str, Any],
    matrix: Dict[str, Any],
    gkr: Dict[str, Any],
    openings: Dict[str, Any],
    gpu_claim: Dict[str, Any] | None,
    backend: str,
) -> Dict[str, Any]:
    hyrax_results = []
    for segment, item in openings["items"].items():
        for chunk in item.get("chunks", []):
            hyrax_results.append(verify_open_payload(chunk["proof"]["label"], chunk["proof"], [int(x) for x in chunk["point"]]))
    lasso_verify = _verify_lasso(lasso)
    matrix_verify = _verify_matrix(matrix)
    gkr_verify = _verify_gkr(gkr)
    checks = {
        "metadata_complete": all(field in metadata for field in METADATA_FIELD_ORDER),
        "commitment_backend": commitments.get("commitment_backend_id") == COMMITMENT_BACKEND_ID,
        "commitment_order": commitments.get("order") == COMMITMENT_ORDER,
        "transcript_replay": digest_json(transcript) == digest_json(_derive_transcript(metadata, commitments, include_late=True)),
        "hyrax_openings": bool(hyrax_results) and all(item["status"] == "VERIFY_OK" for item in hyrax_results),
        "lasso": lasso_verify["status"] == "VERIFY_OK",
        "matrix_sumcheck": matrix_verify["status"] == "VERIFY_OK",
        "gkr": gkr_verify["status"] == "VERIFY_OK",
        "combine": lasso_verify["status"] == matrix_verify["status"] == gkr_verify["status"] == "VERIFY_OK",
        "gpu_hot_path": backend == "cpu" or bool(gpu_claim and gpu_claim.get("cuda_kernel_executed") and gpu_claim.get("proof_hot_path")),
    }
    return {
        "status": "VERIFY_OK" if all(checks.values()) else "VERIFY_FAILED",
        "checks": checks,
        "subproof_verification": {
            "hyrax": hyrax_results,
            "lasso": lasso_verify,
            "matrix": matrix_verify,
            "gkr": gkr_verify,
        },
        "verifier": "zkgpt_gat.formal_verifier.payload_messages_only",
    }


def _gpu_claim_values(commitments: Dict[str, Any], lasso: Dict[str, Any], matrix: Dict[str, Any], gkr: Dict[str, Any], transcript: Dict[str, Any]) -> List[int]:
    values: List[int] = []
    for name in COMMITMENT_ORDER:
        item = commitments["items"][name]
        values.append(int(item["vector_length"]))
        values.append(int(item["padded_length"]))
        values.append(int(item["chunk_count"]))
        values.append(_field_label(item["segment_vector_sha256"]))
        for chunk in item.get("chunks", []):
            values.append(int(chunk["vector_length"]))
            values.append(int(chunk["padded_length"]))
            values.append(_field_label(chunk["masked_evaluation"]))
            values.append(_field_label(chunk["opening_artifact_sha256"]))
    values.extend(int(entry["challenge_u64"]) & ((1 << 64) - 1) for entry in transcript["log"])
    for family in lasso["families"]:
        values.append(int(family["query_count"]))
        values.append(int(family["query_product"]))
        values.append(int(family["table_product"]))
        for chunk in family.get("query_chunks", []):
            values.append(int(chunk["count"]))
            values.append(_field_label(chunk["sha256"]))
        for chunk in family.get("table_chunks", []):
            values.append(int(chunk["count"]))
            values.append(_field_label(chunk["sha256"]))
    for claim in matrix["claims"]:
        proof = claim["sumcheck"]
        values.append(int(proof["claimed_sum"]))
        values.append(int(proof["final_a"]))
        values.append(int(proof["final_b"]))
        values.append(int(proof["final_product"]))
        values.append(int(claim["output_eval_q2"]))
        values.append(int(claim["rescale_rounding_advice"]))
        for round_msg in proof["rounds"]:
            values.extend(int(item) for item in round_msg["coefficients"])
            values.append(int(round_msg["challenge"]))
    for claim in gkr["claims"]:
        values.append(int(claim["constraint_count"]))
        values.append(int(claim["nonzero_count"]))
        values.append(int(claim["variable_count"]))
        for chunk in claim.get("residual_chunks", []):
            values.append(int(chunk["count"]))
            values.append(_field_label(chunk["sha256"]))
        for round_msg in claim["rounds"]:
            values.extend(int(item) for item in round_msg["coefficients"])
            values.append(int(round_msg["challenge"]))
    return [int(value) & ((1 << 64) - 1) for value in values]


def build_proof(
    config: AppConfig,
    dataset: GraphDataset,
    graph: LocalGraph,
    model: Model,
    trace: ForwardTrace,
    backend: str,
    gpu_state: Dict[str, Any],
    paths: Dict[str, pathlib.Path],
) -> ProofArtifacts:
    started = time.perf_counter()
    artifact_dir = paths["artifact_dir"]
    ensure_dir(artifact_dir)
    metadata = make_metadata(config, dataset, graph, model, backend, gpu_state)

    early_segments = _segment_arrays(dataset, graph, model, trace, None)
    early_commitments, _, commitment_ms_early = _commit_segments(metadata, early_segments, paths.get("zk_root", artifact_dir.parents[1]), artifact_dir)
    early_transcript = _derive_transcript(metadata, early_commitments, include_late=True)
    segments = _segment_arrays(dataset, graph, model, trace, early_transcript)
    commitments, openings, commitment_ms = _commit_segments(metadata, segments, paths.get("zk_root", artifact_dir.parents[1]), artifact_dir)
    transcript = _derive_transcript(metadata, commitments, include_late=True)

    lasso = _prove_lasso(trace, config.range_bits, transcript)
    matrix = _prove_matrix_sumchecks(trace, model, transcript["matmul"]["rho_matmul"])
    gkr = _prove_gkr(graph, trace, model, transcript)

    gpu_claim = None
    if backend == "gpu":
        gpu_claim = gpu_proof_reduce(paths.get("zk_root", artifact_dir.parents[1]), _gpu_claim_values(commitments, lasso, matrix, gkr, transcript), f"{dataset.name}:{backend}:proof")
        gpu_claim["proof_hot_path"] = True

    write_json(artifact_dir / "lookup_proof.json", lasso)
    write_json(artifact_dir / "matrix_sumcheck_proof.json", matrix)
    write_json(artifact_dir / "gkr_proof.json", gkr)
    write_json(artifact_dir / "hyrax_mle_openings.json", openings)
    masked_opening_payload = {
        "backend": COMMITMENT_BACKEND_ID,
        "items": {
            name: [chunk["proof"]["opening"] for chunk in value.get("chunks", [])]
            for name, value in openings["items"].items()
        },
    }
    write_json(artifact_dir / "masked_openings.json", masked_opening_payload)
    if gpu_claim is not None:
        write_json(artifact_dir / "gpu_proof_hot_path.json", gpu_claim)

    verify_started = time.perf_counter()
    verification = _verify_formal(metadata, commitments, transcript, lasso, matrix, gkr, openings, gpu_claim, backend)
    verify_elapsed_ms = (time.perf_counter() - verify_started) * 1000.0
    component_refs = {
        "Pi_Lasso": _artifact_ref(artifact_dir / "lookup_proof.json", "lookup proof"),
        "Pi_MatMul": _artifact_ref(artifact_dir / "matrix_sumcheck_proof.json", "matrix sumcheck proof"),
        "Pi_GKR": _artifact_ref(artifact_dir / "gkr_proof.json", "ML-friendly GKR proof"),
        "Open_in": _artifact_ref(artifact_dir / "hyrax_mle_openings.json", "Hyrax/MLE masked openings"),
        "OpenMasked": _artifact_ref(artifact_dir / "masked_openings.json", "masked opening payload"),
    }
    if gpu_claim is not None:
        component_refs["GPU_hot_path"] = _artifact_ref(artifact_dir / "gpu_proof_hot_path.json", "CUDA proof reduction")
    proof_payload = {
        "proof_type": "pi_GAT_zkGPT",
        "M_pub_digest": digest_json(metadata),
        "Y_digest": _array_digest("Y", trace.y),
        "Com_in": [commitments["items"][name] for name in COMMITMENT_ORDER],
        "Pi_Lasso": lasso,
        "Pi_MatMul": matrix,
        "Pi_GKR": gkr,
        "Pi_combine": {
            "protocol": "fiat-shamir-combine-v1",
            "claim_sources": ["Pi_Lasso", "Pi_MatMul", "Pi_GKR", "Open_in"],
            "combined_claim_digest": digest_json({"lasso": lasso, "matrix": matrix, "gkr": gkr, "open": openings}),
            "status": verification["status"],
        },
        "Open_in": openings,
        "OpenMasked": masked_opening_payload,
        "component_artifacts": component_refs,
        "transcript_digest": digest_json(transcript),
    }
    proof_size = len(canonical_json(proof_payload).encode("utf-8"))
    proof_elapsed_ms = (time.perf_counter() - started) * 1000.0
    benchmark = {
        "dataset": metadata["dataset_display_name"],
        "backend": backend,
        "status": verification["status"],
        "commitment_time_ms": commitment_ms + commitment_ms_early,
        "prove_time_ms": proof_elapsed_ms,
        "verify_time_ms": verify_elapsed_ms,
        "proof_size_bytes": proof_size,
        "forward_time_ms": sum(trace.metrics.values()),
        "artifact_path": str(artifact_dir),
    }
    witness_manifest = {
        "witness_generated": True,
        "storage_policy": "complete witness segments are flattened, chunked, committed, and opened through serialized proof payload messages; raw witness arrays are not stored as verifier input",
        "segment_vector_lengths": {name: commitments["items"][name]["vector_length"] for name in COMMITMENT_ORDER},
        "segment_chunk_counts": {name: commitments["items"][name]["chunk_count"] for name in COMMITMENT_ORDER},
        "public_input_digest": _public_digest(graph),
        "model_digest": _model_digest(model),
        "output_digest": _array_digest("Y", trace.y),
        "forward_metric_keys": sorted(trace.metrics),
    }
    return ProofArtifacts(
        metadata=metadata,
        commitments=commitments,
        transcript=transcript,
        proof_payload=proof_payload,
        verification=verification,
        benchmark=benchmark,
        witness_manifest=witness_manifest,
    )


def artifact_ids(config: AppConfig, dataset: GraphDataset) -> Dict[str, Any]:
    cache_dir = config.cache_dir
    files = ["meta.cfg", "edges.txt", "labels.txt"]
    if (cache_dir / "features.txt").exists():
        files.append("features.txt")
    if (cache_dir / "features.npy").exists():
        files.append("features.npy")
    if (cache_dir / "node_ptr.txt").exists():
        files.append("node_ptr.txt")
    data = {"cache_dir": str(cache_dir), "files": {name: file_id(cache_dir / name) for name in files}}
    data["id"] = digest_json(data)
    model = {
        "checkpoint_bundle": str(config.checkpoint_bundle),
        "manifest": file_id(config.checkpoint_bundle / "manifest.json"),
        "tensors": file_id(config.checkpoint_bundle / "tensors.txt"),
    }
    model["id"] = digest_json(model)
    cfg = file_id(config.path)
    cfg["id"] = cfg["sha256"]
    return {"data_id": data, "model_id": model, "config_id": cfg, "dataset_name": dataset.name}
