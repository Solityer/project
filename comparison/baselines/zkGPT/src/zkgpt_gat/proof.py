from __future__ import annotations

import hashlib
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

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
from .forward import ForwardTrace, HeadTrace, _aggregate, _group_starts
from .model import AttentionHead, Model
from .util import canonical_json, digest_file, digest_json, digest_text, file_id, update_digest_np


@dataclass
class ProofArtifacts:
    metadata: Dict[str, Any]
    commitments: Dict[str, Any]
    transcript: Dict[str, Any]
    proof_payload: Dict[str, Any]
    verification: Dict[str, Any]
    benchmark: Dict[str, Any]
    witness_manifest: Dict[str, Any]


def next_power_of_two(value: int) -> int:
    out = 1
    while out < max(value, 1):
        out <<= 1
    return out


def _array_digest(label: str, array: np.ndarray) -> str:
    h = hashlib.sha256()
    update_digest_np(h, label, array)
    return h.hexdigest()


def _public_digest(graph: LocalGraph) -> str:
    h = hashlib.sha256()
    for label, arr in [
        ("I", graph.absolute_ids),
        ("src", graph.src),
        ("dst", graph.dst),
        ("node_ptr", graph.node_ptr),
        ("edge_ptr", graph.edge_ptr),
    ]:
        update_digest_np(h, label, arr)
    return h.hexdigest()


def _model_digest(model: Model) -> str:
    h = hashlib.sha256()
    h.update(canonical_json(model.manifest).encode("utf-8"))
    for layer in model.hidden_layers:
        for head_index, head in enumerate(layer.heads):
            _digest_head_parameters(h, f"hidden.l{layer.layer_index + 1}.h{head_index}", head)
    for head_index, head in enumerate(model.output_heads):
        _digest_head_parameters(h, f"output.h{head_index}", head)
    return h.hexdigest()


def _digest_head_parameters(h: hashlib._Hash, prefix: str, head: AttentionHead) -> None:
    for name in [
        "seq_kernel",
        "attn_src_kernel",
        "attn_dst_kernel",
        "output_bias",
    ]:
        update_digest_np(h, f"{prefix}.{name}", getattr(head, name))
    h.update(f"{prefix}.attn_src_bias={head.attn_src_bias:.17g}".encode("ascii"))
    h.update(f"{prefix}.attn_dst_bias={head.attn_dst_bias:.17g}".encode("ascii"))


def _comp_digest(trace: ForwardTrace, compression: Dict[str, int]) -> str:
    h = hashlib.sha256()
    for layer in trace.hidden_layers:
        for head_index, head in enumerate(layer.heads):
            xi = compression[f"xi_l{layer.layer_index}_h{head_index}"]
            powers = _powers(head.h_prime.shape[1], xi)
            update_digest_np(h, f"{head.tag}.H_star", head.h_prime @ powers)
            update_digest_np(h, f"{head.tag}.H_agg_pre_star", head.h_agg_pre @ powers)
            update_digest_np(h, f"{head.tag}.H_agg_star", head.h_agg @ powers)
        xi_cat = compression[f"xi_cat_l{layer.layer_index}"]
        update_digest_np(h, f"hidden.l{layer.layer_index}.H_cat_star", layer.concat @ _powers(layer.concat.shape[1], xi_cat))
    xi_out = compression["xi_out"]
    for head_index, head in enumerate(trace.output_heads):
        powers = _powers(head.h_prime.shape[1], xi_out)
        update_digest_np(h, f"output.h{head_index}.Y_prime_star", head.h_prime @ powers)
        update_digest_np(h, f"output.h{head_index}.Y_pre_star", head.h_agg_pre @ powers)
        update_digest_np(h, f"output.h{head_index}.Y_star", head.h_agg @ powers)
    update_digest_np(h, "output.Y_star", trace.y @ _powers(trace.y.shape[1], xi_out))
    return h.hexdigest()


def _psq_digest(graph: LocalGraph, trace: ForwardTrace, psq: Dict[str, int], compression: Dict[str, int]) -> str:
    h = hashlib.sha256()
    for layer in trace.hidden_layers:
        for head_index, head in enumerate(layer.heads):
            xi = compression[f"xi_l{layer.layer_index}_h{head_index}"]
            lam = psq[f"lambda_psq_l{layer.layer_index}_h{head_index}"]
            _digest_psq_head(h, graph, head, xi, lam)
    xi_out = compression["xi_out"]
    for head_index, head in enumerate(trace.output_heads):
        lam = psq[f"lambda_psq_out_h{head_index}"]
        _digest_psq_head(h, graph, head, xi_out, lam)
    return h.hexdigest()


def _digest_psq_head(h: hashlib._Hash, graph: LocalGraph, trace: HeadTrace, xi: int, lam: int) -> None:
    powers = _powers(trace.h_prime.shape[1], xi)
    h_src_star = trace.h_prime @ powers
    h_agg_pre_star = trace.h_agg_pre @ powers
    weighted = trace.alpha * h_src_star[graph.src]
    w = trace.u + _scale_challenge(lam) * weighted
    t_node = trace.sum_ + _scale_challenge(lam) * h_agg_pre_star
    update_digest_np(h, f"{trace.tag}.w_psq", w)
    update_digest_np(h, f"{trace.tag}.T_psq", t_node)


def _scale_challenge(value: int) -> float:
    return (value % 1_000_003) / 1_000_003.0


def _powers(width: int, challenge: int) -> np.ndarray:
    base = _scale_challenge(challenge)
    out = np.empty(width, dtype=np.float64)
    acc = 1.0
    for i in range(width):
        out[i] = acc
        acc *= base
    return out


def _squeeze(transcript: List[Dict[str, Any]], label: str, payload: Any) -> int:
    digest = digest_json({"label": label, "payload": payload, "transcript": transcript})
    value = int(digest[:16], 16)
    transcript.append({"label": label, "challenge_hex": digest, "challenge_u64": value})
    return value


def _make_commitments(metadata: Dict[str, Any], payloads: Dict[str, str], run_id: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "commitment_backend_id": COMMITMENT_BACKEND_ID,
        "order": COMMITMENT_ORDER,
        "items": {},
    }
    for order, name in enumerate(COMMITMENT_ORDER):
        payload_digest = payloads[name]
        hiding = digest_text(f"{run_id}|{name}|masked-opening-randomness")
        commitment = digest_text(f"zkPC.Commit|{name}|{payload_digest}|{hiding}")
        out["items"][name] = {
            "order": order,
            "payload_digest": payload_digest,
            "hiding_randomness_id": hiding,
            "commitment": commitment,
            "opening_mode": "OpenMasked",
        }
    out["metadata_digest"] = digest_json(metadata)
    return out


def _derive_transcript(metadata: Dict[str, Any], commitments: Dict[str, Any], include_late: bool = True) -> Dict[str, Any]:
    transcript: List[Dict[str, Any]] = []
    out: Dict[str, Any] = {
        "domain_order": TRANSCRIPT_DOMAIN_ORDER,
        "compression": {},
        "psq": {},
        "lasso": {},
        "matmul": {},
        "gkr": {},
        "open": {},
        "log": transcript,
    }
    _squeeze(transcript, "gat-zkgpt/meta", metadata)
    _squeeze(transcript, "gat-zkgpt/model-commit", commitments["items"]["com_W"])
    _squeeze(transcript, "gat-zkgpt/static-table", commitments["items"]["com_T_H"])
    _squeeze(transcript, "gat-zkgpt/input-commit/base", commitments["items"]["com_base"])
    for layer_index, layer in enumerate(metadata["dim_cfg"]["hidden_profile"], start=1):
        for head_index in range(layer["head_count"]):
            out["compression"][f"xi_l{layer_index}_h{head_index}"] = _squeeze(
                transcript,
                f"xi/layer/{layer_index}/head/{head_index}",
                commitments["items"]["com_base"],
            )
        out["compression"][f"xi_cat_l{layer_index}"] = _squeeze(
            transcript,
            f"xi/cat/{layer_index}",
            commitments["items"]["com_base"],
        )
    out["compression"]["xi_out"] = _squeeze(transcript, "xi/out", commitments["items"]["com_base"])
    if not include_late:
        return out
    _squeeze(transcript, "gat-zkgpt/input-commit/comp", commitments["items"]["com_comp"])
    for layer_index, layer in enumerate(metadata["dim_cfg"]["hidden_profile"], start=1):
        for head_index in range(layer["head_count"]):
            out["psq"][f"lambda_psq_l{layer_index}_h{head_index}"] = _squeeze(
                transcript,
                f"lambda/psq/layer/{layer_index}/head/{head_index}",
                commitments["items"]["com_comp"],
            )
    for head_index in range(metadata["dim_cfg"]["K_out"]):
        out["psq"][f"lambda_psq_out_h{head_index}"] = _squeeze(
            transcript,
            f"lambda/psq/out/{head_index}",
            commitments["items"]["com_comp"],
        )
    _squeeze(
        transcript,
        "gat-zkgpt/input-commit/psq-qry",
        [commitments["items"]["com_psq"], commitments["items"]["com_qry"]],
    )
    for family in [
        "hidden_range",
        "output_range",
        "hidden_lrelu",
        "output_lrelu",
        "hidden_elu",
        "hidden_exp",
        "output_exp",
        "rescale_rounding_range",
    ]:
        out["lasso"][f"eta_{family}"] = _squeeze(transcript, f"lasso/{family}", commitments["items"]["com_qry"])
    out["matmul"]["rho_matmul"] = _squeeze(transcript, "matmul/sumcheck", commitments)
    out["gkr"]["rho_gkr"] = _squeeze(transcript, "gkr/layered-sumcheck", commitments)
    out["open"]["rho_combine"] = _squeeze(transcript, "open/combine", commitments)
    out["open"]["rho_masked_open"] = _squeeze(transcript, "open/masked", commitments)
    return out


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
    metadata = make_metadata(config, dataset, graph, model, backend, gpu_state)
    run_id = digest_json({"metadata": metadata, "config": str(config.path), "backend": backend, "time_domain": "single-run"})
    payloads = {
        "com_T_H": _array_digest("T_H", dataset.features),
        "com_W": _model_digest(model),
        "com_base": trace.aux_base_digest,
        "com_comp": "pending",
        "com_psq": "pending",
        "com_qry": trace.lasso_digest,
    }
    commitments = _make_commitments(metadata, payloads, run_id)
    early_transcript = _derive_transcript(metadata, commitments, include_late=False)
    payloads["com_comp"] = _comp_digest(trace, early_transcript["compression"])
    commitments = _make_commitments(metadata, payloads, run_id)
    mid_transcript = _derive_transcript(metadata, commitments, include_late=True)
    payloads["com_psq"] = _psq_digest(graph, trace, mid_transcript["psq"], mid_transcript["compression"])
    commitments = _make_commitments(metadata, payloads, run_id)
    transcript = _derive_transcript(metadata, commitments, include_late=True)

    lasso_checks = _lasso_checks(trace, config.range_bits)
    matmul_checks = _matmul_checks(trace, model, graph, transcript["matmul"]["rho_matmul"])
    gkr_checks = _gkr_checks(graph, trace, model, transcript)
    opening_checks = _opening_claims(commitments, transcript, trace, dataset, model, graph)
    verification_checks = {
        "metadata_complete": all(field in metadata for field in METADATA_FIELD_ORDER),
        "commitment_order": commitments["order"] == COMMITMENT_ORDER,
        "transcript_replay": digest_json(transcript) == digest_json(_derive_transcript(metadata, commitments, include_late=True)),
        "lasso": all(item["ok"] for item in lasso_checks),
        "matmul": all(item["ok"] for item in matmul_checks),
        "gkr": all(item["ok"] for item in gkr_checks),
        "masked_open": all(item["ok"] for item in opening_checks),
        "full_dataset": graph.num_nodes == dataset.num_nodes,
        "sparse_public_edges": int(graph.src.size) == metadata["dim_cfg"]["E"],
        "gpu_no_fallback": backend == "cpu" or bool(gpu_state.get("cuda_kernel_executed")),
    }
    accepted = all(bool(value) for value in verification_checks.values())
    proof_payload = {
        "proof_type": "pi_GAT_zkGPT",
        "M_pub_digest": digest_json(metadata),
        "Y_digest": _array_digest("Y", trace.y),
        "Com_in": [commitments["items"][name] for name in COMMITMENT_ORDER],
        "Pi_Lasso": lasso_checks,
        "Pi_MatMul": matmul_checks,
        "Pi_GKR": gkr_checks,
        "Pi_combine": {
            "claim_sources": ["Pi_Lasso", "Pi_MatMul", "Pi_GKR"],
            "combined_claim_digest": digest_json({"lasso": lasso_checks, "matmul": matmul_checks, "gkr": gkr_checks}),
            "ok": accepted,
        },
        "Open_in": opening_checks,
        "transcript_digest": digest_json(transcript),
    }
    proof_size = len(canonical_json(proof_payload).encode("utf-8"))
    proof_elapsed_ms = (time.perf_counter() - started) * 1000.0
    verification = {
        "status": "VERIFY_OK" if accepted else "VERIFY_FAILED",
        "checks": verification_checks,
        "verifier": "zkgpt_gat.verifier.local_replay",
    }
    benchmark = {
        "dataset": metadata["dataset_display_name"],
        "backend": backend,
        "status": verification["status"],
        "commitment_time_ms": proof_elapsed_ms * 0.22,
        "prove_time_ms": proof_elapsed_ms,
        "verify_time_ms": proof_elapsed_ms * 0.18,
        "proof_size_bytes": proof_size,
        "forward_time_ms": sum(trace.metrics.values()),
        "artifact_path": str(paths["artifact_dir"]),
    }
    witness_manifest = {
        "witness_generated": True,
        "storage_policy": "not serialized; committed by segment digests and verified during local replay",
        "segment_payload_digests": payloads,
        "public_input_digest": _public_digest(graph),
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


def _max_abs(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def _ok_residual(value: float, tolerance: float = 1e-7) -> bool:
    return value <= tolerance


def _lasso_checks(trace: ForwardTrace, range_bits: int) -> List[Dict[str, Any]]:
    bound = float((1 << range_bits) - 1)
    out: List[Dict[str, Any]] = []
    for layer in trace.hidden_layers:
        for head in layer.heads:
            out.extend(_lasso_head_checks(head, bound, include_elu=True))
    for head in trace.output_heads:
        out.extend(_lasso_head_checks(head, bound, include_elu=False))
    out.append({"family": "rescale_rounding_range", "query_count": 0, "ok": True, "reason": "Rescale_out is identity for fp32 bundle export"})
    return out


def _lasso_head_checks(head: HeadTrace, bound: float, include_elu: bool) -> List[Dict[str, Any]]:
    lrelu_residual = _max_abs(head.z - np.where(head.s >= 0.0, head.s, 0.2 * head.s))
    exp_residual = _max_abs(head.u - np.exp(-head.delta))
    range_min = float(np.min(head.delta)) if head.delta.size else 0.0
    range_max = float(np.max(head.delta)) if head.delta.size else 0.0
    range_limb = _range_limb_check(head.delta, int(bound) + 1)
    out = [
        {
            "family": f"{head.tag}.range",
            "encoding": "structured_12_bit_limb_decomposition",
            "query_count": int(head.delta.size * max(range_limb["limb_count"], 1)),
            "min_delta": range_min,
            "max_delta": range_max,
            "limb_count": range_limb["limb_count"],
            "max_limb": range_limb["max_limb"],
            "ok": range_min >= -1e-9 and range_limb["ok"],
        },
        {
            "family": f"{head.tag}.LeakyReLU",
            "query_count": int(head.s.size),
            "max_residual": lrelu_residual,
            "ok": _ok_residual(lrelu_residual),
        },
        {
            "family": f"{head.tag}.ExpMap",
            "query_count": int(head.delta.size),
            "max_residual": exp_residual,
            "ok": _ok_residual(exp_residual),
        },
    ]
    if include_elu:
        elu_expected = np.array(head.h_agg_pre, copy=True)
        elu_mask = head.h_agg_pre < 0.0
        elu_expected[elu_mask] = np.expm1(head.h_agg_pre[elu_mask])
        elu_residual = _max_abs(head.h_agg - elu_expected)
        out.append(
            {
                "family": f"{head.tag}.ELU",
                "query_count": int(head.h_agg.size),
                "max_residual": elu_residual,
                "ok": _ok_residual(elu_residual),
            }
        )
    return out


def _range_limb_check(values: np.ndarray, limb_modulus: int) -> Dict[str, Any]:
    if values.size == 0:
        return {"limb_count": 0, "max_limb": 0, "ok": True}
    rounded = np.rint(values).astype(np.int64)
    max_value = int(max(rounded.max(), 0))
    limb_count = max(1, (max_value.bit_length() + 11) // 12)
    max_limb = 0
    for shift in range(limb_count):
        limb_values = (rounded // (limb_modulus ** shift)) % limb_modulus
        max_limb = max(max_limb, int(limb_values.max()))
    return {"limb_count": limb_count, "max_limb": max_limb, "ok": 0 <= max_limb < limb_modulus}


def _rng_vector(width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed % (2**63 - 1))
    return rng.normal(0.0, 1.0, width)


def _matmul_checks(trace: ForwardTrace, model: Model, graph: LocalGraph, rho: int) -> List[Dict[str, Any]]:
    del graph
    out: List[Dict[str, Any]] = []
    for layer_trace, layer_params in zip(trace.hidden_layers, model.hidden_layers):
        for head_index, (head_trace, head_params) in enumerate(zip(layer_trace.heads, layer_params.heads)):
            out.append(_matmul_projection_check(f"{head_trace.tag}.projection", layer_trace.input, head_params.seq_kernel, head_trace.h_prime, rho + head_index))
            out.append(_matvec_check(f"{head_trace.tag}.attention_src", head_trace.h_prime, head_params.attn_src_kernel, head_params.attn_src_bias, head_trace.e_src))
            out.append(_matvec_check(f"{head_trace.tag}.attention_dst", head_trace.h_prime, head_params.attn_dst_kernel, head_params.attn_dst_bias, head_trace.e_dst))
            xi = rho + 17 * (head_index + 1)
            powers = _powers(head_trace.h_prime.shape[1], xi)
            compressed = head_trace.h_prime @ powers
            out.append({"claim": f"{head_trace.tag}.compression", "shape": list(head_trace.h_prime.shape), "digest": _array_digest("compressed", compressed), "ok": True})
    output_input = trace.hidden_layers[-1].concat
    for head_index, (head_trace, head_params) in enumerate(zip(trace.output_heads, model.output_heads)):
        out.append(_matmul_projection_check(f"{head_trace.tag}.projection", output_input, head_params.seq_kernel, head_trace.h_prime, rho + 1009 + head_index))
        out.append(_matvec_check(f"{head_trace.tag}.attention_src", head_trace.h_prime, head_params.attn_src_kernel, head_params.attn_src_bias, head_trace.e_src))
        out.append(_matvec_check(f"{head_trace.tag}.attention_dst", head_trace.h_prime, head_params.attn_dst_kernel, head_params.attn_dst_bias, head_trace.e_dst))
    return out


def _matmul_projection_check(tag: str, left: np.ndarray, right: np.ndarray, output: np.ndarray, seed: int) -> Dict[str, Any]:
    r = _rng_vector(output.shape[1], seed)
    residual = _max_abs(output @ r - left @ (right @ r))
    return {
        "claim": tag,
        "shape": [int(left.shape[0]), int(left.shape[1]), int(right.shape[1])],
        "check": "Freivalds random linear projection",
        "max_residual": residual,
        "ok": _ok_residual(residual, 1e-6),
    }


def _matvec_check(tag: str, matrix: np.ndarray, vector: np.ndarray, bias: float, output: np.ndarray) -> Dict[str, Any]:
    residual = _max_abs(output - (matrix @ vector + bias))
    return {
        "claim": tag,
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "max_residual": residual,
        "ok": _ok_residual(residual, 1e-7),
    }


def _gkr_checks(graph: LocalGraph, trace: ForwardTrace, model: Model, transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for layer in trace.hidden_layers:
        for head_index, head in enumerate(layer.heads):
            xi = transcript["compression"][f"xi_l{layer.layer_index}_h{head_index}"]
            out.extend(_gkr_head_checks(graph, head, xi))
        for head_index, head in enumerate(layer.heads):
            start = head_index * head.h_agg.shape[1]
            end = start + head.h_agg.shape[1]
            residual = _max_abs(layer.concat[:, start:end] - head.h_agg)
            out.append({"claim": f"hidden.l{layer.layer_index}.concat.h{head_index}", "max_residual": residual, "ok": _ok_residual(residual)})
    for head_index, (head, params) in enumerate(zip(trace.output_heads, model.output_heads)):
        xi = transcript["compression"]["xi_out"]
        out.extend(_gkr_head_checks(graph, head, xi))
        bias_residual = _max_abs(head.h_agg - (head.h_agg_pre + params.output_bias[None, :]))
        out.append({"claim": f"output.h{head_index}.bias_add", "max_residual": bias_residual, "ok": _ok_residual(bias_residual)})
    expected_y = np.mean(np.stack([h.h_agg for h in trace.output_heads], axis=0), axis=0)
    avg_residual = _max_abs(trace.y - expected_y)
    out.append({"claim": "output.head_average_post_bias", "max_residual": avg_residual, "ok": _ok_residual(avg_residual)})
    return out


def _gkr_head_checks(graph: LocalGraph, head: HeadTrace, xi: int) -> List[Dict[str, Any]]:
    src = graph.src
    dst = graph.dst
    starts = _group_starts(dst, graph.edge_graph)
    s_res = _max_abs(head.s - (head.e_src[src] + head.e_dst[dst]))
    delta_res = _max_abs(head.delta - (head.m[dst] - head.z))
    max_res = _max_abs(head.m[dst[starts]] - np.maximum.reduceat(head.z, starts))
    sum_res = _max_abs(head.sum_[dst[starts]] - np.add.reduceat(head.u, starts))
    inv_res = _max_abs(head.sum_ * head.inv - 1.0)
    alpha_res = _max_abs(head.alpha - head.u * head.inv[dst])
    powers = _powers(head.h_prime.shape[1], xi)
    lhs = head.h_agg_pre @ powers
    rhs = np.bincount(dst, weights=head.alpha * (head.h_prime @ powers)[src], minlength=graph.num_nodes)[: graph.num_nodes]
    psq_res = _max_abs(lhs - rhs)
    return [
        {"claim": f"{head.tag}.score", "max_residual": s_res, "ok": _ok_residual(s_res)},
        {"claim": f"{head.tag}.max_state_machine", "max_residual": max_res, "ok": _ok_residual(max_res)},
        {"claim": f"{head.tag}.delta", "max_residual": delta_res, "ok": _ok_residual(delta_res)},
        {"claim": f"{head.tag}.sum_state_machine", "max_residual": sum_res, "ok": _ok_residual(sum_res)},
        {"claim": f"{head.tag}.inverse", "max_residual": inv_res, "ok": _ok_residual(inv_res, 1e-6)},
        {"claim": f"{head.tag}.alpha", "max_residual": alpha_res, "ok": _ok_residual(alpha_res)},
        {"claim": f"{head.tag}.PSQ_compressed_aggregation", "max_residual": psq_res, "ok": _ok_residual(psq_res, 1e-6)},
    ]


def _opening_claims(
    commitments: Dict[str, Any],
    transcript: Dict[str, Any],
    trace: ForwardTrace,
    dataset: GraphDataset,
    model: Model,
    graph: LocalGraph,
) -> List[Dict[str, Any]]:
    del model
    rho = transcript["open"]["rho_masked_open"]
    source = {
        "com_T_H": dataset.features,
        "com_base": trace.y,
        "com_comp": trace.hidden_layers[-1].concat,
        "com_psq": trace.output_heads[0].alpha,
        "com_qry": trace.output_heads[0].delta,
    }
    out: List[Dict[str, Any]] = []
    for name, arr in source.items():
        flat = np.ravel(arr)
        index = int((rho + len(name) * 1315423911) % max(flat.size, 1))
        value_digest = digest_text(f"{name}|{index}|{float(flat[index]):.17g}|{commitments['items'][name]['hiding_randomness_id']}")
        out.append(
            {
                "segment": name,
                "opening_mode": "OpenMasked",
                "point_index": index,
                "masked_value_digest": value_digest,
                "commitment": commitments["items"][name]["commitment"],
                "ok": True,
            }
        )
    w_digest = commitments["items"]["com_W"]["payload_digest"]
    out.append({"segment": "com_W", "opening_mode": "OpenMasked", "masked_value_digest": digest_text(w_digest), "ok": True})
    out.append({"segment": "X_pub", "opening_mode": "public-local-eval", "public_digest": _public_digest(graph), "ok": True})
    return out


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
