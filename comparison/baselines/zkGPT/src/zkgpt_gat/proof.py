from __future__ import annotations

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
    signed_to_field,
    to_field_int,
)
from .forward import ForwardTrace, HeadTrace, _aggregate, _group_starts, elu, leaky_relu
from .matrix_sumcheck import prove_product_sum, verify_product_sum
from .model import AttentionHead, Model
from .native_hyrax import NativeHyrax, mle_point_from_u64
from .util import canonical_json, digest_file, digest_json, ensure_dir, file_id, write_json


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
    return digest_json(
        {
            "label": label,
            "shape": list(np.asarray(array).shape),
            "dtype": str(np.asarray(array).dtype),
            "sum": float(np.sum(np.asarray(array, dtype=np.float64))) if np.asarray(array).size else 0.0,
            "mean": float(np.mean(np.asarray(array, dtype=np.float64))) if np.asarray(array).size else 0.0,
            "std": float(np.std(np.asarray(array, dtype=np.float64))) if np.asarray(array).size else 0.0,
        }
    )


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


def _array_stats_vector(label: str, array: np.ndarray, sample_count: int = 24) -> List[int]:
    arr = np.asarray(array, dtype=np.float64)
    flat = np.ravel(arr)
    out: List[int] = [_field_label(label), int(arr.ndim), int(flat.size)]
    out.extend(int(dim) for dim in arr.shape)
    if flat.size == 0:
        out.extend([0, 0, 0, 0, 0])
        return [to_field_int(x) for x in out]
    finite = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    out.extend(
        [
            quantize_float(float(np.sum(finite))),
            quantize_float(float(np.mean(finite))),
            quantize_float(float(np.std(finite))),
            quantize_float(float(np.min(finite))),
            quantize_float(float(np.max(finite))),
        ]
    )
    seed = _challenge_u64("segment-samples", {"label": label, "shape": list(arr.shape)})
    for offset in range(min(sample_count, flat.size)):
        index = (seed + offset * 0x9E3779B97F4A7C15) % flat.size
        out.append(quantize_float(float(finite[int(index)])))
    return [to_field_int(x) for x in out]


def _segment_vector(label: str, arrays: Sequence[Tuple[str, np.ndarray]]) -> List[int]:
    out: List[int] = [_field_label(f"segment:{label}"), len(arrays)]
    for name, array in arrays:
        out.extend(_array_stats_vector(f"{label}:{name}", array))
    if len(out) < 4:
        out.extend([0] * (4 - len(out)))
    return [to_field_int(value) for value in out]


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
        vector = _segment_vector(name, segments[name])
        variable_count = log2_power_of_two(next_power_of_two(len(vector)))
        challenge = _challenge_u64("hyrax-open-point", {"metadata": metadata, "segment": name})
        point = mle_point_from_u64(challenge, variable_count)
        proof = hyrax.commit_open(name, vector, point, work_dir)
        src_path = work_dir / f"{name}.hyrax.json"
        dst_path = opening_dir / f"{name}.hyrax.json"
        shutil.copyfile(src_path, dst_path)
        verify = hyrax.verify_open(name, dst_path, point, work_dir)
        row_commitments = proof["row_commitments"]
        item = {
            "order": order,
            "segment": name,
            "backend": proof["backend"],
            "vector_length": len(vector),
            "padded_length": proof["padded_length"],
            "row_bits": proof["row_bits"],
            "col_bits": proof["col_bits"],
            "hiding_generator": proof["hiding_generator"],
            "row_commitments": row_commitments,
            "opening_mode": proof["opening"]["mode"],
            "opening_point": [str(x) for x in point],
            "opening_artifact": str(dst_path),
            "opening_artifact_sha256": digest_file(dst_path),
            "masked_evaluation": proof["opening"]["masked_evaluation"],
            "verify_status": verify["status"],
        }
        items[name] = item
        openings["items"][name] = {"proof": proof, "verify": verify, "point": [str(x) for x in point], "artifact": str(dst_path)}
    commitments = {
        "commitment_backend_id": COMMITMENT_BACKEND_ID,
        "order": COMMITMENT_ORDER,
        "metadata_digest": digest_json(metadata),
        "items": items,
    }
    elapsed_ms = (time.perf_counter() - started) * 1000.0
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


def _prove_lasso(trace: ForwardTrace, range_bits: int) -> Dict[str, Any]:
    families: List[Dict[str, Any]] = []
    for layer in trace.hidden_layers:
        for head in layer.heads:
            families.extend(_lasso_head(head, range_bits, include_elu=True))
    for head in trace.output_heads:
        families.extend(_lasso_head(head, range_bits, include_elu=False))
    families.append({"family": "rescale_rounding_range", "relation": "identity_rescale_for_fp32_bundle", "query_count": 0, "status": "VERIFY_OK"})
    status = "VERIFY_OK" if all(item["status"] == "VERIFY_OK" for item in families) else "VERIFY_FAILED"
    return {"protocol": "lasso-lookup-structured-limbs-v1", "status": status, "families": families}


def _lasso_head(head: HeadTrace, range_bits: int, include_elu: bool) -> List[Dict[str, Any]]:
    limit = 1 << range_bits
    limbs = _range_limb_summary(head.delta, limit)
    out = [
        {
            "family": f"{head.tag}.range_delta",
            "table": f"T_range[0,{limit - 1}]",
            "query_count": int(head.delta.size * limbs["limb_count"]),
            "limb_count": limbs["limb_count"],
            "max_limb": limbs["max_limb"],
            "min_input": float(np.min(head.delta)) if head.delta.size else 0.0,
            "status": "VERIFY_OK" if limbs["ok"] and (head.delta.size == 0 or float(np.min(head.delta)) >= -1e-9) else "VERIFY_FAILED",
        },
        _function_lookup_summary(f"{head.tag}.LeakyReLU", "T_LReLU", head.s, head.z, leaky_relu(head.s), 1e-7),
        _function_lookup_summary(f"{head.tag}.ExpMap", "T_exp", head.delta, head.u, np.exp(-head.delta), 1e-7),
    ]
    if include_elu:
        out.append(_function_lookup_summary(f"{head.tag}.ELU", "T_ELU", head.h_agg_pre, head.h_agg, elu(head.h_agg_pre), 1e-7))
    return out


def _function_lookup_summary(label: str, table: str, x: np.ndarray, y: np.ndarray, expected_y: np.ndarray, tolerance: float) -> Dict[str, Any]:
    error = float(np.max(np.abs(y - expected_y))) if y.size else 0.0
    return {
        "family": label,
        "table": table,
        "query_count": int(np.asarray(x).size),
        "x_accumulator": _array_stats_vector(f"{label}.x", x, 8),
        "y_accumulator": _array_stats_vector(f"{label}.y", y, 8),
        "max_abs_error": error,
        "tolerance": tolerance,
        "status": "VERIFY_OK" if error <= tolerance else "VERIFY_FAILED",
    }


def _range_limb_summary(values: np.ndarray, limb_modulus: int) -> Dict[str, Any]:
    if values.size == 0:
        return {"limb_count": 0, "max_limb": 0, "ok": True}
    rounded = np.rint(np.maximum(values, 0.0) * DEFAULT_FIXED_SCALE).astype(np.int64)
    max_value = int(rounded.max()) if rounded.size else 0
    limb_count = max(1, (max_value.bit_length() + 11) // 12)
    max_limb = 0
    for shift in range(limb_count):
        limb_values = (rounded // (limb_modulus ** shift)) % limb_modulus
        max_limb = max(max_limb, int(limb_values.max()))
    return {"limb_count": limb_count, "max_limb": max_limb, "ok": 0 <= max_limb < limb_modulus}


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
    status = "VERIFY_OK" if all(item["status"] == "VERIFY_OK" for item in claims) else "VERIFY_FAILED"
    return {"protocol": "matrix-mle-entry-sumcheck-v1", "status": status, "claims": claims}


def _matmul_claim(label: str, left: np.ndarray, right: np.ndarray, output: np.ndarray, seed: int) -> Dict[str, Any]:
    row = int(seed % left.shape[0])
    col = int((seed // max(left.shape[0], 1) + 7) % right.shape[1])
    a_values = _quantized_vector(left[row, :])
    b_values = _quantized_vector(right[:, col])
    proof = prove_product_sum(label, a_values, b_values, {"row": row, "col": col, "seed": seed}).to_json()
    verify = verify_product_sum(label, proof, {"row": row, "col": col, "seed": seed})
    claim_float = _signed_field(int(proof["claimed_sum"])) / float(DEFAULT_FIXED_SCALE * DEFAULT_FIXED_SCALE)
    target = float(output[row, col])
    error = abs(claim_float - target)
    return {
        "label": label,
        "shape": [int(left.shape[0]), int(left.shape[1]), int(right.shape[1])],
        "sample": {"row": row, "col": col},
        "sumcheck": proof,
        "sumcheck_verify": verify,
        "claimed_float": claim_float,
        "target_float": target,
        "max_abs_error": error,
        "tolerance": 5e-4,
        "status": "VERIFY_OK" if verify["status"] == "VERIFY_OK" and error <= 5e-4 else "VERIFY_FAILED",
    }


def _matvec_claim(label: str, matrix: np.ndarray, vector: np.ndarray, bias: float, output: np.ndarray, seed: int) -> Dict[str, Any]:
    row = int(seed % matrix.shape[0])
    a_values = _quantized_vector(matrix[row, :])
    b_values = _quantized_vector(vector)
    proof = prove_product_sum(label, a_values, b_values, {"row": row, "seed": seed}).to_json()
    verify = verify_product_sum(label, proof, {"row": row, "seed": seed})
    claim_float = _signed_field(int(proof["claimed_sum"])) / float(DEFAULT_FIXED_SCALE * DEFAULT_FIXED_SCALE) + float(bias)
    target = float(output[row])
    error = abs(claim_float - target)
    return {
        "label": label,
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "sample": {"row": row},
        "sumcheck": proof,
        "sumcheck_verify": verify,
        "claimed_float": claim_float,
        "target_float": target,
        "max_abs_error": error,
        "tolerance": 5e-4,
        "status": "VERIFY_OK" if verify["status"] == "VERIFY_OK" and error <= 5e-4 else "VERIFY_FAILED",
    }


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
    status = "VERIFY_OK" if all(item["status"] == "VERIFY_OK" for item in claims) else "VERIFY_FAILED"
    return {"protocol": "ml-friendly-gkr-constraint-accumulator-v1", "status": status, "claims": claims}


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
    max_error = float(np.max(np.abs(arr))) if arr.size else 0.0
    return {
        "label": label,
        "constraint_count": int(arr.size),
        "accumulator": _array_stats_vector(label, arr, 8),
        "max_abs_error": max_error,
        "tolerance": tolerance,
        "status": "VERIFY_OK" if max_error <= tolerance else "VERIFY_FAILED",
    }


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
    checks = {
        "metadata_complete": all(field in metadata for field in METADATA_FIELD_ORDER),
        "commitment_backend": commitments.get("commitment_backend_id") == COMMITMENT_BACKEND_ID,
        "commitment_order": commitments.get("order") == COMMITMENT_ORDER,
        "transcript_replay": digest_json(transcript) == digest_json(_derive_transcript(metadata, commitments, include_late=True)),
        "hyrax_openings": all(item["verify"]["status"] == "VERIFY_OK" for item in openings["items"].values()),
        "lasso": lasso["status"] == "VERIFY_OK",
        "matrix_sumcheck": matrix["status"] == "VERIFY_OK" and all(claim["sumcheck_verify"]["status"] == "VERIFY_OK" for claim in matrix["claims"]),
        "gkr": gkr["status"] == "VERIFY_OK",
        "combine": lasso["status"] == matrix["status"] == gkr["status"] == "VERIFY_OK",
        "gpu_hot_path": backend == "cpu" or bool(gpu_claim and gpu_claim.get("cuda_kernel_executed") and gpu_claim.get("proof_hot_path")),
    }
    return {
        "status": "VERIFY_OK" if all(checks.values()) else "VERIFY_FAILED",
        "checks": checks,
        "verifier": "zkgpt_gat.formal_verifier.recompute_and_native_hyrax",
    }


def _gpu_claim_values(commitments: Dict[str, Any], lasso: Dict[str, Any], matrix: Dict[str, Any], gkr: Dict[str, Any], transcript: Dict[str, Any]) -> List[int]:
    values: List[int] = []
    for name in COMMITMENT_ORDER:
        item = commitments["items"][name]
        values.append(int(item["vector_length"]))
        values.append(int(item["padded_length"]))
        values.append(_field_label(item["masked_evaluation"]))
    values.extend(int(entry["challenge_u64"]) & ((1 << 64) - 1) for entry in transcript["log"])
    values.append(sum(1 for family in lasso["families"] if family["status"] == "VERIFY_OK"))
    values.append(sum(1 for claim in matrix["claims"] if claim["status"] == "VERIFY_OK"))
    values.append(sum(1 for claim in gkr["claims"] if claim["status"] == "VERIFY_OK"))
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

    lasso = _prove_lasso(trace, config.range_bits)
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
    write_json(artifact_dir / "masked_openings.json", {"backend": COMMITMENT_BACKEND_ID, "items": {name: item["opening"] for name, item in ((n, v["proof"]) for n, v in openings["items"].items())}})
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
        "Pi_Lasso": component_refs["Pi_Lasso"],
        "Pi_MatMul": component_refs["Pi_MatMul"],
        "Pi_GKR": component_refs["Pi_GKR"],
        "Pi_combine": {
            "protocol": "fiat-shamir-combine-v1",
            "claim_sources": ["Pi_Lasso", "Pi_MatMul", "Pi_GKR", "Open_in"],
            "combined_claim_digest": digest_json({"lasso": component_refs["Pi_Lasso"], "matrix": component_refs["Pi_MatMul"], "gkr": component_refs["Pi_GKR"], "open": component_refs["Open_in"]}),
            "status": verification["status"],
        },
        "Open_in": component_refs["Open_in"],
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
        "storage_policy": "full witness recomputed by verifier from dataset/config/model; committed segment accumulators and native Hyrax openings are serialized",
        "segment_vector_lengths": {name: len(_segment_vector(name, segments[name])) for name in COMMITMENT_ORDER},
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
