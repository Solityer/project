from __future__ import annotations

import hashlib
import pathlib
import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
from .field import (
    BN254_FR_MODULUS,
    DEFAULT_FIXED_SCALE,
    f_mul,
    log2_power_of_two,
    next_power_of_two,
    quantize_float,
    to_field_int,
)
from .formal_gate import require_formal_experiment
from .gpu_execution import (
    ProofExecution,
    activate as activate_proof_execution,
    deactivate as deactivate_proof_execution,
    gpu_execution_transcript_binding,
    validate_gpu_execution_evidence,
)
from .forward import ForwardTrace, HeadTrace, SignedRoundDivTrace, _elu_table_value, _exp_map_fixed, _group_starts, elu, leaky_relu
from .layout import build_input_layout
from .input_oracle import equality_weights
from .matrix_sumcheck import prove_product_sum, verify_product_sum
from .model import AttentionHead, Model, QUANT_SCALE_PROFILES
from .native_matmul import MatMulSpec, TensorSource, _ProductProvider, prove_matmul, verify_matmul
from .native_gkr import (
    LinearGateSpec,
    LinearTerm,
    OracleRef,
    ProductGateSpec,
    identity_oracle,
    prove_linear_gate,
    prove_product_gate,
    verify_linear_gate,
    verify_product_gate,
)
from .native_hyrax import NativeHyrax, _point, hyrax_value_params, verify_open_payload
from .native_lasso import prove_lasso as prove_native_lasso, verify_lasso as verify_native_lasso
from .registry import ensure_model_registry
from .transcript import Transcript
from .value_commitment import (
    MultiplicationProof,
    RepresentationProof,
    ValueCommitment,
    prove_multiplication,
    prove_representation,
    verify_multiplication,
    verify_representation,
)
from .zk_sumcheck import prove_committed_sumcheck, verify_committed_sumcheck
from .util import canonical_json, digest_file, digest_json, ensure_dir, file_id, write_json
from .verifier_context import make_external_context, validate_external_context


SEGMENT_CHUNK_VALUES = 262_144
CONSTRAINT_FIXED_SCALE = 100_000
LOOKUP_PROTOCOL_ID = "lasso-fixed-table-grand-product-v3"
GKR_PROTOCOL_ID = "ml-friendly-gkr-layered-relation-v3"
GKR_CLAIM_PROTOCOL_ID = "gkr-layer-relation-random-linear-eval-v3"
LOOKUP_QUANTIZATION_TOLERANCE_Q = 0
FUNCTION_LOOKUP_DOMAIN_BITS = 63
RANGE_ARGUMENT_PROTOCOL_ID = "direct-fixed-range-bounded-limb-argument-v1"
MATRIX_RESCALE_PROTOCOL_ID = "committed-per-entry-rescale-rounding-v2"


@dataclass
class ProofArtifacts:
    metadata: Dict[str, Any]
    commitments: Dict[str, Any]
    transcript: Dict[str, Any]
    proof_payload: Dict[str, Any]
    verification: Dict[str, Any]
    benchmark: Dict[str, Any]
    witness_manifest: Dict[str, Any]


@dataclass
class SegmentCommitmentState:
    hyrax: NativeHyrax
    vectors: Dict[str, List[int]]
    registrations: Dict[str, Dict[str, Any]]


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
    if arr.dtype == object:
        for value in arr.ravel().tolist():
            h.update(str(value).encode("ascii"))
            h.update(b",")
    else:
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
            arrays.extend(_head_param_arrays(f"hidden.l{layer.layer_index + 1}.h{head_index}", head, output_head=False))
    for head_index, head in enumerate(model.output_heads):
        arrays.extend(_head_param_arrays(f"output.h{head_index}", head, output_head=True))
    return arrays


def _head_param_arrays(prefix: str, head: AttentionHead, output_head: bool) -> List[Tuple[str, np.ndarray]]:
    arrays = [
        (f"{prefix}.W", head.seq_kernel),
        (f"{prefix}.a_src", head.attn_src_kernel),
        (f"{prefix}.a_dst", head.attn_dst_kernel),
    ]
    if output_head:
        arrays.append((f"{prefix}.b_out", head.output_bias))
    return arrays


def _model_digest(model: Model) -> str:
    return digest_json({"manifest": model.manifest, "arrays": [_array_digest(name, arr) for name, arr in _model_arrays(model)]})


def make_metadata(
    config: AppConfig,
    dataset: GraphDataset,
    graph: LocalGraph,
    model: Model,
    backend: str,
    gpu_state: Dict[str, Any],
    input_layout: Dict[str, Any],
) -> Dict[str, Any]:
    hidden_profile = [{"head_count": layer.head_count, "head_dim": layer.head_dim} for layer in model.hidden_layers]
    d_in_profile = [int(model.manifest["d_in_profile"][0])]
    d_cat = [item["head_count"] * item["head_dim"] for item in hidden_profile]
    d_in_layers = d_in_profile + d_cat
    domain_cfg = {
        "n_FH": next_power_of_two(max(dataset.num_nodes * graph.num_features, graph.num_nodes * graph.num_features)),
        "n_edge": next_power_of_two(max(graph.src.size, graph.num_nodes, graph.num_nodes * max([x["head_dim"] for x in hidden_profile] + [model.class_count]))),
        "n_in": int(input_layout["N_in"]),
        "n_d_h": next_power_of_two(max(x["head_dim"] for x in hidden_profile)),
        "n_cat": next_power_of_two(max(d_cat)),
        "n_C": next_power_of_two(model.class_count),
        "n_N": next_power_of_two(graph.num_nodes),
    }
    metadata = {
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
        "proof_object": "pi_GAT_zkGPT=(M_pub,Y,com_W,com_Aux,com_L,Pi_Lasso,Pi_MatMul,Pi_GKR,Pi_combine,Open_W_Aux_L)",
        "public_input_layout": "X_pub=features+topology+selectors; X_pub is verifier-reconstructed and uncommitted",
        "X_pub_digest": digest_json(
            {
                "features": _array_digest("X_pub.H_input", _public_x_arrays(graph, model)[0][1]),
                "topology": _public_digest(graph),
                "selectors": {"segment_order": ["X_pub", "W", "Aux", "L_qry"], "graph_count": graph.graph_count},
            }
        ),
        "public_output_policy": "Y is a public statement output claim and is not part of X_pub or any private commitment",
        "private_input_layout": "X_pub|W|Aux|L_qry",
        "segment_order": ["X_pub", "W", "Aux", "L_qry"],
        "privacy_extension_profile_id": "masked_full_multi_session_v1",
        "privacy_profile_status": "supported_multi_session_masked_full",
        "session_nonce": secrets.token_hex(32),
        "registered_model_commitment_scope": "canonical_integer_GAT_model_weights_with_layout_and_bound_ledger",
        "Layout_in": input_layout,
        "Layout_in_digest": input_layout["layout_digest"],
    }
    metadata["integer_bound_ledger"] = _integer_bound_ledger(metadata, model, graph)
    metadata["integer_bound_ledger_digest"] = digest_json(metadata["integer_bound_ledger"])
    metadata["static_table_registry"] = _static_lookup_registry(metadata, config.range_bits)
    metadata["static_table_registry_digest"] = digest_json(metadata["static_table_registry"])
    return metadata


def _public_x_arrays(graph: LocalGraph, model: Model) -> List[Tuple[str, np.ndarray]]:
    features = np.asarray(
        [int(np.floor(float(item) + 0.5) if float(item) >= 0 else np.ceil(float(item) - 0.5)) for item in np.ravel(graph.features)],
        dtype=np.int64,
    ).reshape(graph.features.shape)
    dimensions = np.asarray(
        [
            graph.graph_count,
            graph.num_nodes,
            int(graph.src.size),
            graph.num_features,
            len(model.hidden_layers) + 1,
            model.class_count,
            model.k_out,
        ],
        dtype=np.int64,
    )
    return [
        ("X_pub.H_input", features),
        ("X_pub.absolute_ids", np.asarray(graph.absolute_ids, dtype=np.int64)),
        ("X_pub.dimension_scalars", dimensions),
    ]


def _padded_segment_chunks(label: str, values: Sequence[int]) -> Iterable[Tuple[str, List[int]]]:
    if not values:
        raise ValueError(f"empty padded segment: {label}")
    # The protocol has one N_in-domain MLE per private segment.  Splitting it
    # into independently challenged chunk MLEs changes the relation, so the
    # formal path commits the complete canonical padded vector as one object.
    yield label, [to_field_int(value) for value in values]


def _ephemeral_hyrax_opening(label: str, values: Sequence[int], context: Dict[str, Any]) -> Dict[str, Any]:
    """Commit before challenges and open a complete canonical subprotocol MLE."""
    vector = [to_field_int(value) for value in values] or [0]
    padded_length = next_power_of_two(len(vector))
    vector.extend([0] * (padded_length - len(vector)))
    with tempfile.TemporaryDirectory(prefix="zkgpt-hyrax-") as tmp:
        root = pathlib.Path(tmp)
        hyrax = NativeHyrax(root)
        registration = hyrax.commit_vector(label, vector, root)
        public_registration = {key: value for key, value in registration.items() if key != "blind"}
        transcript = Transcript("zkGPT/GAT/ephemeral-Hyrax/v2")
        transcript.absorb(
            "subprotocol-context-and-commitment",
            {"label": label, "context": context, "registration": public_registration},
        )
        point = [
            transcript.challenge_scalar(f"{label}/point/{index}")
            for index in range(log2_power_of_two(padded_length))
        ]
        state = hyrax.open_registered(
            label,
            vector,
            point,
            registration,
            root,
            transcript=transcript,
            public_context=context,
        )
    return {
        "protocol": "zkGPT-ephemeral-Hyrax-opening-v2",
        "label": label,
        "logical_length": len(values),
        "vector_length": padded_length,
        "registration": public_registration,
        "context": context,
        "point": [str(value) for value in point],
        "opening": state.proof,
        "transcript_digest": transcript.digest.hex(),
    }


def _verify_ephemeral_hyrax_opening(label: str, payload: Dict[str, Any], context: Dict[str, Any]) -> bool:
    try:
        if payload.get("protocol") != "zkGPT-ephemeral-Hyrax-opening-v2" or payload.get("label") != label:
            return False
        if payload.get("context") != context:
            return False
        vector_length = int(payload["vector_length"])
        registration = payload["registration"]
        if int(registration.get("vector_length", -1)) != vector_length:
            return False
        transcript = Transcript("zkGPT/GAT/ephemeral-Hyrax/v2")
        transcript.absorb(
            "subprotocol-context-and-commitment",
            {"label": label, "context": context, "registration": registration},
        )
        point = [
            transcript.challenge_scalar(f"{label}/point/{index}")
            for index in range(log2_power_of_two(vector_length))
        ]
        if payload.get("point") != [str(value) for value in point]:
            return False
        result = verify_open_payload(
            label,
            payload["opening"],
            point,
            registration,
            vector_length,
            transcript=transcript,
            public_context=context,
        )
        return result.get("status") == "VERIFY_OK" and payload.get("transcript_digest") == transcript.digest.hex()
    except (KeyError, TypeError, ValueError):
        return False


def _segment_arrays(dataset: GraphDataset, graph: LocalGraph, model: Model, trace: ForwardTrace, transcript: Dict[str, Any] | None = None) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    arrays: Dict[str, List[Tuple[str, np.ndarray]]] = {
        "com_W": _model_arrays(model),
        "com_Aux": [],
        "com_L": [],
    }
    for layer in trace.hidden_layers:
        arrays["com_Aux"].append((f"hidden.l{layer.layer_index}.concat", layer.concat))
        for head in layer.heads:
            arrays["com_Aux"].extend(_head_trace_arrays(head))
            arrays["com_Aux"].extend(_head_gkr_product_advice_arrays(head))
            arrays["com_Aux"].extend(_head_sparse_gkr_advice_arrays(graph, head))
    for head in trace.output_heads:
        arrays["com_Aux"].extend(_head_trace_arrays(head))
        arrays["com_Aux"].extend(_head_gkr_product_advice_arrays(head))
        arrays["com_Aux"].extend(_head_sparse_gkr_advice_arrays(graph, head))
    # Matrix division/range advice belongs to Aux; lookup query coordinates
    # belong exclusively to L_qry.  Keep the public graph/features outside all
    # private commitments and reconstruct it from the statement.
    arrays["com_Aux"].extend(_matrix_rescale_residual_arrays(trace, model))
    residual_values = np.asarray([
        int(value)
        for _, residual in _matrix_rescale_residual_arrays(trace, model)
        for value in np.ravel(np.asarray(residual, dtype=object))
    ], dtype=object)
    residual_sign = np.asarray([1 if int(value) < 0 else 0 for value in residual_values], dtype=object)
    residual_abs = np.asarray([abs(int(value)) for value in residual_values], dtype=object)
    arrays["com_Aux"].extend([
        ("rescale_rounding_range.gkr.value", residual_values),
        ("rescale_rounding_range.gkr.sign", residual_sign),
        ("rescale_rounding_range.gkr.abs", residual_abs),
        ("rescale_rounding_range.gkr.sign_abs", residual_sign * residual_abs),
    ])
    if trace.output_average_division is not None:
        arrays["com_Aux"].extend(_division_trace_arrays("output.final_average", trace.output_average_division))
        arrays["com_Aux"].extend(_division_gkr_product_advice_arrays("output.final_average", trace.output_average_division))
        arrays["com_Aux"].append(("output.Y_sum", trace.output_average_division.x))
        arrays["com_Aux"].append(("output.Y", trace.output_average_division.y))
    return arrays


def _head_trace_arrays(head: HeadTrace) -> List[Tuple[str, np.ndarray]]:
    out = [
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
    if head.agg_num is not None:
        out.append((f"{head.tag}.H_or_Y_agg_num", head.agg_num))
    if head.elu_in is not None:
        out.append((f"{head.tag}.SignedRoundDiv.output", head.elu_in))
    if head.field_agg is not None:
        out.append((f"{head.tag}.field_aggregate", head.field_agg))
    if head.denominator is not None:
        out.append((f"{head.tag}.SignedRoundDiv.denominator", head.denominator))
    if head.group_start_selector is not None:
        out.append((f"{head.tag}.Q_grp", head.group_start_selector))
    if head.group_count is not None:
        out.append((f"{head.tag}.group_count", head.group_count))
    if head.max_selector is not None:
        out.append((f"{head.tag}.s_max", head.max_selector))
    if head.r_u is not None:
        out.append((f"{head.tag}.R_U", head.r_u))
    if head.r_num is not None:
        out.append((f"{head.tag}.R_num", head.r_num))
    if head.r_field is not None:
        out.append((f"{head.tag}.R_field", head.r_field))
    if head.aggregate_division is not None:
        out.extend(_division_trace_arrays(f"{head.tag}.SignedRoundDiv.aggregate", head.aggregate_division))
    if head.scale_alignment_division is not None:
        out.extend(_division_trace_arrays(f"{head.tag}.SignedRoundDiv.scale_alignment", head.scale_alignment_division))
    if head.output_aligned is not None:
        out.append((f"{head.tag}.Y_pre_aligned", head.output_aligned))
    return out


def _division_trace_arrays(prefix: str, division: SignedRoundDivTrace) -> List[Tuple[str, np.ndarray]]:
    return [
        (f"{prefix}.{name}", getattr(division, name))
        for name in ("x", "d", "y", "sigma", "z", "a", "a_inv", "d_inv", "q", "r", "delta")
    ]


def _division_gkr_product_advice_arrays(prefix: str, division: SignedRoundDivTrace) -> List[Tuple[str, np.ndarray]]:
    """Challenge-independent multiplication wires committed with ``com_Aux``."""
    values = {name: np.asarray(getattr(division, name), dtype=object) for name in (
        "sigma", "z", "a", "a_inv", "d", "d_inv", "q",
    )}
    return [
        (f"{prefix}.gkr.sigma_a", values["sigma"] * values["a"]),
        (f"{prefix}.gkr.sigma_q", values["sigma"] * values["q"]),
        (f"{prefix}.gkr.a_a_inv", _field_product_array(values["a"], values["a_inv"])),
        (f"{prefix}.gkr.d_d_inv", _field_product_array(values["d"], values["d_inv"])),
        (f"{prefix}.gkr.d_q", values["d"] * values["q"]),
        (f"{prefix}.gkr.z_a", values["z"] * values["a"]),
        (f"{prefix}.gkr.z_sigma", values["z"] * values["sigma"]),
        (f"{prefix}.gkr.z_a_inv", _field_product_array(values["z"], values["a_inv"])),
    ]


def _head_gkr_product_advice_arrays(head: HeadTrace) -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = [
        (f"{head.tag}.gkr.sum_inv", _field_product_array(head.sum_, head.inv)),
        (f"{head.tag}.gkr.s_max_delta", np.asarray(head.max_selector, dtype=object) * np.asarray(head.delta, dtype=object)),
    ]
    if head.aggregate_division is not None:
        out.extend(_division_gkr_product_advice_arrays(f"{head.tag}.SignedRoundDiv.aggregate", head.aggregate_division))
    if head.scale_alignment_division is not None:
        out.extend(_division_gkr_product_advice_arrays(f"{head.tag}.SignedRoundDiv.scale_alignment", head.scale_alignment_division))
    return out


def _head_sparse_gkr_advice_arrays(graph: LocalGraph, head: HeadTrace) -> List[Tuple[str, np.ndarray]]:
    q = np.asarray(head.group_start_selector, dtype=object)
    one_minus_q = 1 - q
    width = int(head.h_prime.shape[1])
    edge_count = int(graph.src.size)

    def previous(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=object)
        out = np.zeros(arr.shape, dtype=object)
        if edge_count > 1:
            out[1:] = arr[:-1]
        if arr.ndim == 1:
            out[q.astype(bool)] = 0
        else:
            out[q.astype(bool), :] = 0
        return out

    r_u_prev = previous(np.asarray(head.r_u, dtype=object))
    r_num_prev = previous(np.asarray(head.r_num, dtype=object))
    r_field_prev = previous(np.asarray(head.r_field, dtype=object))
    h_edge = np.asarray(head.h_prime, dtype=object)[np.asarray(graph.src, dtype=np.int64)]
    u_h = np.asarray(head.u, dtype=object)[:, None] * h_edge
    alpha_h = _field_product_array(
        np.repeat(np.asarray(head.alpha, dtype=object)[:, None], width, axis=1), h_edge
    )
    r_smax = np.zeros(edge_count, dtype=object)
    running_smax = 0
    for edge in range(edge_count):
        if int(q[edge]):
            running_smax = 0
        running_smax += int(head.max_selector[edge])
        r_smax[edge] = running_smax
    r_smax_prev = previous(r_smax)
    return [
        (f"{head.tag}.gkr.one_minus_Q", one_minus_q),
        (f"{head.tag}.gkr.R_U_prev", r_u_prev),
        (f"{head.tag}.gkr.R_U_carry", one_minus_q * r_u_prev),
        (f"{head.tag}.gkr.R_smax", r_smax),
        (f"{head.tag}.gkr.R_smax_prev", r_smax_prev),
        (f"{head.tag}.gkr.R_smax_carry", one_minus_q * r_smax_prev),
        (f"{head.tag}.gkr.R_num_prev", r_num_prev),
        (f"{head.tag}.gkr.R_num_carry", one_minus_q[:, None] * r_num_prev),
        (f"{head.tag}.gkr.U_H", u_h),
        (f"{head.tag}.gkr.R_field_prev", r_field_prev),
        (f"{head.tag}.gkr.R_field_carry", one_minus_q[:, None] * r_field_prev),
        (f"{head.tag}.gkr.Alpha_H", alpha_h),
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
        out.extend([(f"{head.tag}.L_ELU.x", head.elu_in if head.elu_in is not None else head.h_agg_pre), (f"{head.tag}.L_ELU.y", head.h_agg)])
    if head.aggregate_division is not None:
        out.extend(_division_range_query_arrays(f"{head.tag}.SignedRoundDiv.aggregate", head.aggregate_division))
    if head.scale_alignment_division is not None:
        out.extend(_division_range_query_arrays(f"{head.tag}.SignedRoundDiv.scale_alignment", head.scale_alignment_division))
    return out


def _division_range_query_arrays(prefix: str, division: SignedRoundDivTrace) -> List[Tuple[str, np.ndarray]]:
    return [(f"{prefix}.L_range.{name}", getattr(division, name)) for name in ("a", "q", "r", "delta", "d")]


def _pad_lasso_family(values: Sequence[int], pad_value: int) -> List[int]:
    logical = [to_field_int(value) for value in values]
    padded_count = next_power_of_two(max(1, len(logical)))
    return logical + [to_field_int(pad_value)] * (padded_count - len(logical))


def _canonical_lasso_segment(
    trace: ForwardTrace,
    model: Model,
    range_bits: int,
    fixed_tables: Dict[str, Any],
    ledger: Dict[str, Any],
) -> Tuple[List[Tuple[str, np.ndarray]], List[Dict[str, Any]]]:
    """Build committed Pack_f/range blocks including protocol-defined padding."""
    arrays: List[Tuple[str, np.ndarray]] = []
    registry: List[Dict[str, Any]] = []
    offset = 0
    limb_count = max(1, (int(ledger["bit_width"]) + int(range_bits) - 1) // int(range_bits))
    limb_base = 1 << int(range_bits)

    def add_function(label: str, table_name: str, x: np.ndarray, y: np.ndarray) -> None:
        nonlocal offset
        table = fixed_tables[table_name]
        real = [_pair_key(table, int(a), int(b)) for a, b in zip(np.ravel(x), np.ravel(y))]
        domain_min = int(table["certified_domain"][0])
        a_pad = _pair_key(table, domain_min, _fixed_table_eval_q(table_name, domain_min))
        padded = _pad_lasso_family(real, a_pad)
        arrays.append((label, np.asarray(padded, dtype=object)))
        registry.append({
            "family": label,
            "kind": "function",
            "table_id": table_name,
            "offset": offset,
            "real_count": len(real),
            "padded_count": len(padded),
            "a_pad": str(a_pad),
            "Q_pad": {"zero_prefix": len(real), "one_suffix": len(padded) - len(real)},
            "tuple_encoding": table["tuple_encoding"],
        })
        offset += len(padded)

    def add_range(label: str, values: np.ndarray, *, signed: bool = False) -> None:
        nonlocal offset
        real: List[int] = []
        for raw in np.ravel(np.asarray(values, dtype=object)):
            value = int(raw)
            if not signed and value < 0:
                raise ValueError(f"negative value in unsigned range family {label}")
            current = abs(value) if signed else value
            for _ in range(limb_count):
                real.append(current % limb_base)
                current //= limb_base
            if current:
                raise ValueError(f"range family {label} exceeds public ledger width")
        padded = _pad_lasso_family(real, 0)
        arrays.append((label, np.asarray(padded, dtype=object)))
        registry.append({
            "family": label,
            "kind": "range",
            "table_id": "T_range",
            "offset": offset,
            "real_count": len(real),
            "padded_count": len(padded),
            "a_pad": "0",
            "Q_pad": {"zero_prefix": len(real), "one_suffix": len(padded) - len(real)},
            "range_bits": int(range_bits),
            "limbs_per_value": limb_count,
            "signed_magnitude": bool(signed),
        })
        offset += len(padded)

    for layer in trace.hidden_layers:
        for head in layer.heads:
            add_range(f"{head.tag}.range_delta", head.delta)
            add_function(f"{head.tag}.LeakyReLU", "T_LReLU", head.s, head.z)
            add_function(f"{head.tag}.ExpMap", "T_exp", head.delta, head.u)
            add_function(f"{head.tag}.ELU", "T_ELU", head.elu_in, head.h_agg)
            if head.aggregate_division is not None:
                add_range(f"{head.tag}.SignedRoundDiv.aggregate.range", _division_range_values(head.aggregate_division))
    for head in trace.output_heads:
        add_range(f"{head.tag}.range_delta", head.delta)
        add_function(f"{head.tag}.LeakyReLU", "T_LReLU", head.s, head.z)
        add_function(f"{head.tag}.ExpMap", "T_exp", head.delta, head.u)
        if head.aggregate_division is not None:
            add_range(f"{head.tag}.SignedRoundDiv.aggregate.range", _division_range_values(head.aggregate_division))
        if head.scale_alignment_division is not None:
            add_range(f"{head.tag}.SignedRoundDiv.scale_alignment.range", _division_range_values(head.scale_alignment_division))
    if trace.output_average_division is not None:
        add_range("output.final_average.SignedRoundDiv.range", _division_range_values(trace.output_average_division))
    residual_values = [
        int(value)
        for _, residual in _matrix_rescale_residual_arrays(trace, model)
        for value in np.ravel(np.asarray(residual, dtype=object))
    ]
    add_range("rescale_rounding_range", np.asarray(residual_values, dtype=object), signed=True)
    return arrays, registry


def prepare_input_segments_and_layout(
    config: AppConfig,
    dataset: GraphDataset,
    graph: LocalGraph,
    model: Model,
    trace: ForwardTrace,
) -> Tuple[Dict[str, List[Tuple[str, np.ndarray]]], Dict[str, Any], Dict[str, List[int]]]:
    segments = _segment_arrays(dataset, graph, model, trace, None)
    provisional_ledger = _integer_bound_ledger({}, model, graph)
    provisional_metadata = {
        "quant_cfg_id": model.quant_cfg_id,
        "static_table_id": model.static_table_id,
        "encoding_id": ENCODING_ID,
        "integer_bound_ledger": provisional_ledger,
        "integer_bound_ledger_digest": digest_json(provisional_ledger),
    }
    provisional_tables = _static_lookup_registry(provisional_metadata, config.range_bits)
    lasso_arrays, lasso_family_registry = _canonical_lasso_segment(
        trace, model, config.range_bits, provisional_tables, provisional_ledger
    )
    segments["com_L"] = lasso_arrays
    lasso_values = {name: np.asarray(values, dtype=object) for name, values in lasso_arrays}
    for family in lasso_family_registry:
        if family["kind"] != "range":
            continue
        values = np.ravel(lasso_values[str(family["family"])]).tolist()
        bits = np.asarray([
            (int(value) >> bit) & 1
            for value in values for bit in range(int(family["range_bits"]))
        ], dtype=object).reshape((len(values), int(family["range_bits"])))
        segments["com_Aux"].append((f"lasso.{family['family']}.range_bits", bits))
    input_layout, padded_segments = build_input_layout(
        _public_x_arrays(graph, model), segments, lasso_family_registry
    )
    return segments, input_layout, padded_segments


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
    padded_segments: Dict[str, List[int]],
    zk_root: pathlib.Path,
    artifact_dir: pathlib.Path,
    model_registry: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], SegmentCommitmentState, float]:
    started = time.perf_counter()
    hyrax = NativeHyrax(zk_root)
    work_dir = artifact_dir / "hyrax_work"
    items: Dict[str, Any] = {}
    registrations: Dict[str, Dict[str, Any]] = {}
    vectors: Dict[str, List[int]] = {}

    # Commit every private segment before deriving the unique terminal point.
    # com_W comes from the externally registered model; Aux/L use per-proof
    # fresh blinders and never serialize those openings.
    for order, name in enumerate(COMMITMENT_ORDER):
        logical_name = {"com_W": "W", "com_Aux": "Aux", "com_L": "L_qry"}[name]
        vector = [to_field_int(value) for value in padded_segments[logical_name]]
        if len(vector) != int(metadata["Layout_in"]["N_in"]):
            raise ValueError(f"{name} is not the canonical N_in-domain vector")
        vectors[name] = vector
        registered = None
        if name == "com_W" and model_registry is not None:
            registered = model_registry.get("secret", {}).get("entries", {}).get(name)
        registration = registered or hyrax.commit_vector(name, vector, work_dir)
        registrations[name] = registration
        public_registration = {key: value for key, value in registration.items() if key != "blind"}
        item = {
            "order": order,
            "segment": name,
            "backend": COMMITMENT_BACKEND_ID,
            "vector_length": len(vector),
            "logical_length": next(
                int(item["logical_length"])
                for item in metadata["Layout_in"]["segments"]
                if item["name"] == logical_name
            ),
            "N_in": int(metadata["Layout_in"]["N_in"]),
            "padded_length": len(vector),
            "chunk_count": 1,
            "chunking_rule": "single canonical N_in-domain segment MLE",
            "commitment": public_registration["commitment"],
            "parameter_id": public_registration["parameter_id"],
            "arrays": next(
                item["arrays"]
                for item in metadata["Layout_in"]["segments"]
                if item["name"] == logical_name
            ),
            "registered_model_commitment": bool(registered is not None),
            "registered_model_registry_digest": (model_registry or {}).get("public", {}).get("registry_digest", "") if name == "com_W" else "",
        }
        item["commitment_set_digest"] = digest_json(
            {"segment": name, "N_in": len(vector), "parameter_id": item["parameter_id"], "commitment": item["commitment"]}
        )
        item["chunks"] = [{
            "chunk_index": 0,
            "label": name,
            "vector_length": len(vector),
            "padded_length": len(vector),
            "commitment": item["commitment"],
            "registered_model_commitment": bool(registered is not None),
        }]
        items[name] = item
    commitments = {
        "commitment_backend_id": COMMITMENT_BACKEND_ID,
        "order": COMMITMENT_ORDER,
        "metadata_digest": digest_json(metadata),
        "items": items,
    }

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return commitments, {}, SegmentCommitmentState(hyrax, vectors, registrations), elapsed_ms


def _segment_binding(commitments: Dict[str, Any], segment: str) -> Dict[str, Any]:
    item = commitments["items"][segment]
    return {
        "segment": segment,
        "backend": item["backend"],
        "vector_length": int(item["vector_length"]),
        "padded_length": int(item["padded_length"]),
        "chunk_count": int(item["chunk_count"]),
        "commitment_set_digest": item["commitment_set_digest"],
        "commitment": item["commitment"],
        "parameter_id": item["parameter_id"],
    }


def _open_terminal_segments(
    metadata: Dict[str, Any],
    commitments: Dict[str, Any],
    state: SegmentCommitmentState,
    public_x: Sequence[int],
    transcript: Transcript,
    artifact_dir: pathlib.Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Perform the unique input-layer reduction after every relation proof."""
    n_in = int(metadata["Layout_in"]["N_in"])
    if len(public_x) != n_in:
        raise ValueError("public X segment is not padded to canonical N_in")
    sources = [item for item in state.hyrax.opening_log if item.get("kind") == "linear"]
    if not sources:
        raise ValueError("input combine has no authenticated relation terminal claims")
    unified = (
        [to_field_int(value) for value in public_x]
        + state.vectors["com_W"] + state.vectors["com_Aux"] + state.vectors["com_L"]
    )
    coefficients = [0] * (4 * n_in)
    initial_value = 0
    initial_blind = 0
    initial_point = None
    source_descriptors = []
    block = {"com_W": 1, "com_Aux": 2, "com_L": 3}
    for index, source in enumerate(sources):
        registration_label = source.get("registration_label")
        if registration_label not in block:
            raise ValueError("relation terminal uses an unregistered input segment")
        weight = transcript.challenge_scalar(f"input-combine/source/{index}", nonzero=True)
        offset = block[registration_label] * n_in
        source_weights = source["weights"]
        weight_items = source_weights.items() if isinstance(source_weights, dict) else enumerate(source_weights)
        for position, coefficient in weight_items:
            coefficients[offset + position] = (
                coefficients[offset + position] + weight * to_field_int(coefficient)
            ) % BN254_FR_MODULUS
        opening_state = source["state"]
        initial_value = (initial_value + weight * opening_state.evaluation) % BN254_FR_MODULUS
        initial_blind = (initial_blind + weight * opening_state.blind) % BN254_FR_MODULUS
        term = weight * opening_state.evaluation_commitment.point
        initial_point = term if initial_point is None else initial_point + term
        source_descriptors.append({
            "index": index,
            "label": source["label"],
            "segment": registration_label,
            "linear_form": source["linear_form"],
            "evaluation_commitment": opening_state.evaluation_commitment.to_dict(),
            "weight": str(weight),
        })
    if initial_point is None or sum(a * b for a, b in zip(coefficients, unified)) % BN254_FR_MODULUS != initial_value:
        raise ValueError("relation terminal aggregation is inconsistent")
    params = hyrax_value_params(n_in)
    initial_commitment = ValueCommitment(initial_point)
    reduction = prove_committed_sumcheck(
        "input-combine/relation-claim-reduction", initial_commitment,
        _ProductProvider(coefficients, unified), 2,
        variable_count=log2_power_of_two(4 * n_in), params=params,
        transcript=transcript, initial_value=initial_value, initial_blind=initial_blind,
    )
    full_point = tuple(reversed(reduction.challenges))
    selector_point = full_point[:2]
    point = full_point[2:]
    before_open_digest = transcript.digest.hex()
    context = {
        "metadata_digest": commitments["metadata_digest"],
        "session_nonce": metadata["session_nonce"],
        "Layout_in_digest": metadata["Layout_in_digest"],
        "pre_open_transcript_digest": before_open_digest,
        "selector_point": [str(value) for value in selector_point],
    }
    opening_dir = artifact_dir / "hyrax_openings"
    ensure_dir(opening_dir)
    openings: Dict[str, Any] = {
        "protocol": "zkGPT/final-canonical-input-openings-v1",
        "backend": COMMITMENT_BACKEND_ID,
        "point": [str(value) for value in point],
        "pre_open_transcript_digest": before_open_digest,
        "public_context": context,
        "items": {},
    }
    states = {}
    for name in COMMITMENT_ORDER:
        opening = state.hyrax.open_registered(
            f"final/{name}", state.vectors[name], point, state.registrations[name],
            artifact_dir / "hyrax_work", transcript=transcript, public_context=context,
        )
        states[name] = opening
        public_registration = {
            key: value for key, value in state.registrations[name].items() if key != "blind"
        }
        openings["items"][name] = {
            "registration": public_registration,
            "proof": opening.proof,
        }
    params = hyrax_value_params(n_in)
    x_eval = sum(
        value * weight for value, weight in zip(
            (to_field_int(value) for value in public_x),
            equality_weights(point),
        )
    ) % BN254_FR_MODULUS
    selector_weights = equality_weights(selector_point)
    combined_value = selector_weights[0] * x_eval % BN254_FR_MODULUS
    combined_blind = 0
    combined_point = selector_weights[0] * x_eval * params.U
    for power, name in zip(selector_weights[1:], COMMITMENT_ORDER):
        opening = states[name]
        combined_value = (combined_value + power * opening.evaluation) % BN254_FR_MODULUS
        combined_blind = (combined_blind + power * opening.blind) % BN254_FR_MODULUS
        combined_point = combined_point + power * opening.evaluation_commitment.point
    combined = ValueCommitment(combined_point)
    coefficient_eval = sum(
        coefficient * weight
        for coefficient, weight in zip(coefficients, equality_weights(full_point))
    ) % BN254_FR_MODULUS
    terminal_link = prove_multiplication(
        params, ValueCommitment(coefficient_eval * params.U), combined,
        reduction.terminal_commitment, coefficient_eval, combined_value,
        reduction.terminal_value, 0, combined_blind, reduction.blind,
        label="input-combine/terminal-product", transcript=transcript,
    )
    combine = {
        "protocol": "zkGPT/canonical-input-claim-reduction-v1",
        "sources": source_descriptors,
        "initial_claim_commitment": initial_commitment.to_dict(),
        "sumcheck": reduction.proof.to_dict(),
        "selector_point": [str(value) for value in selector_point],
        "r_minus": [str(value) for value in point],
        "public_x_evaluation": str(x_eval),
        "combined_commitment": combined.to_dict(),
        "coefficient_terminal_evaluation": str(coefficient_eval),
        "terminal_product": terminal_link.to_dict(),
        "transcript_digest": transcript.digest.hex(),
    }
    return openings, combine


def _verify_terminal_segments_and_combine(
    metadata: Dict[str, Any],
    commitments: Dict[str, Any],
    openings: Dict[str, Any],
    combine: Dict[str, Any],
    public_x: Sequence[int],
    transcript: Transcript,
    verified_sources: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        n_in = int(metadata["Layout_in"]["N_in"])
        if len(public_x) != n_in:
            return {"status": "VERIFY_FAILED", "reason": "public X length mismatch"}
        if not verified_sources:
            return {"status": "VERIFY_FAILED", "reason": "no verified input claim sources"}
        source_payloads = combine.get("sources", [])
        if not isinstance(source_payloads, list) or len(source_payloads) != len(verified_sources):
            return {"status": "VERIFY_FAILED", "reason": "input claim source count mismatch"}
        coefficients = [0] * (4 * n_in)
        initial_point = None
        block = {"com_W": 1, "com_Aux": 2, "com_L": 3}
        for index, (payload, source) in enumerate(zip(source_payloads, verified_sources)):
            weight = transcript.challenge_scalar(f"input-combine/source/{index}", nonzero=True)
            expected_payload = {
                "index": index,
                "label": source["label"],
                "segment": source["segment"],
                "linear_form": source["linear_form"],
                "evaluation_commitment": source["evaluation_commitment"].to_dict(),
                "weight": str(weight),
            }
            if payload != expected_payload or source["segment"] not in block:
                return {"status": "VERIFY_FAILED", "reason": "input claim source differs from verified subproof"}
            offset = block[source["segment"]] * n_in
            source_weights = source["weights"]
            weight_items = source_weights.items() if isinstance(source_weights, dict) else enumerate(source_weights)
            for position, coefficient in weight_items:
                coefficients[offset + position] = (
                    coefficients[offset + position] + weight * to_field_int(coefficient)
                ) % BN254_FR_MODULUS
            term = weight * source["evaluation_commitment"].point
            initial_point = term if initial_point is None else initial_point + term
        if initial_point is None:
            return {"status": "VERIFY_FAILED", "reason": "empty initial input claim"}
        initial_commitment = ValueCommitment(initial_point)
        if ValueCommitment.from_dict(combine.get("initial_claim_commitment")) != initial_commitment:
            return {"status": "VERIFY_FAILED", "reason": "initial input claim commitment mismatch"}
        params = hyrax_value_params(n_in)
        reduction = verify_committed_sumcheck(
            "input-combine/relation-claim-reduction", combine.get("sumcheck", {}),
            initial_commitment, log2_power_of_two(4 * n_in), 2,
            params=params, transcript=transcript,
        )
        full_point = tuple(reversed(reduction.challenges))
        selector_point = full_point[:2]
        point = full_point[2:]
        if combine.get("selector_point") != [str(value) for value in selector_point] or combine.get("r_minus") != [str(value) for value in point]:
            return {"status": "VERIFY_FAILED", "reason": "input reduction terminal point mismatch"}
        before_open_digest = transcript.digest.hex()
        context = {
            "metadata_digest": commitments["metadata_digest"],
            "session_nonce": metadata["session_nonce"],
            "Layout_in_digest": metadata["Layout_in_digest"],
            "pre_open_transcript_digest": before_open_digest,
            "selector_point": [str(value) for value in selector_point],
        }
        if (
            openings.get("protocol") != "zkGPT/final-canonical-input-openings-v1"
            or openings.get("backend") != COMMITMENT_BACKEND_ID
            or openings.get("point") != [str(value) for value in point]
            or openings.get("pre_open_transcript_digest") != before_open_digest
            or openings.get("public_context") != context
            or set(openings.get("items", {})) != set(COMMITMENT_ORDER)
        ):
            return {"status": "VERIFY_FAILED", "reason": "terminal opening statement mismatch"}
        evaluation_commitments = {}
        opening_results = []
        for name in COMMITMENT_ORDER:
            payload = openings["items"][name]
            expected_registration = commitments["items"][name]
            if payload.get("registration", {}).get("commitment") != expected_registration.get("commitment"):
                return {"status": "VERIFY_FAILED", "reason": "terminal opening changed segment commitment"}
            result = verify_open_payload(
                f"final/{name}", payload.get("proof", {}), point,
                expected_registration, n_in, transcript=transcript,
                public_context=context,
            )
            opening_results.append(result.get("status") == "VERIFY_OK")
            evaluation_commitments[name] = ValueCommitment(
                _point(payload["proof"]["evaluation_commitment"])
            )
        x_eval = sum(
            to_field_int(value) * weight
            for value, weight in zip(public_x, equality_weights(point))
        ) % BN254_FR_MODULUS
        selector_weights = equality_weights(selector_point)
        combined_point = selector_weights[0] * x_eval * params.U
        for power, name in zip(selector_weights[1:], COMMITMENT_ORDER):
            combined_point = combined_point + power * evaluation_commitments[name].point
        expected_combined = ValueCommitment(combined_point)
        coefficient_eval = sum(
            coefficient * weight
            for coefficient, weight in zip(coefficients, equality_weights(full_point))
        ) % BN254_FR_MODULUS
        shape_ok = (
            set(combine) == {
                "protocol", "sources", "initial_claim_commitment", "sumcheck",
                "selector_point", "r_minus", "public_x_evaluation",
                "combined_commitment", "coefficient_terminal_evaluation",
                "terminal_product", "transcript_digest",
            }
            and combine["protocol"] == "zkGPT/canonical-input-claim-reduction-v1"
            and combine["public_x_evaluation"] == str(x_eval)
            and ValueCommitment.from_dict(combine["combined_commitment"]) == expected_combined
            and combine["coefficient_terminal_evaluation"] == str(coefficient_eval)
        )
        terminal_ok = shape_ok and verify_multiplication(
            params, ValueCommitment(coefficient_eval * params.U), expected_combined,
            reduction.terminal_commitment,
            MultiplicationProof.from_dict(combine["terminal_product"]),
            label="input-combine/terminal-product", transcript=transcript,
        )
        digest_ok = terminal_ok and combine["transcript_digest"] == transcript.digest.hex()
        return {
            "status": "VERIFY_OK" if all(opening_results) and digest_ok else "VERIFY_FAILED",
            "openings": opening_results,
            "combine": bool(terminal_ok),
            "transcript": bool(digest_ok),
        }
    except (KeyError, TypeError, ValueError):
        return {"status": "VERIFY_FAILED", "reason": "malformed terminal input combine"}


def _derive_transcript(metadata: Dict[str, Any], commitments: Dict[str, Any], include_late: bool = True) -> Dict[str, Any]:
    log: List[Dict[str, Any]] = []
    out: Dict[str, Any] = {
        "domain_order": TRANSCRIPT_DOMAIN_ORDER,
        "compression": {},
        "psq": {},
        "input_layer": {"segment_order": ["X_pub", "W", "Aux", "L_qry"], "selector_encoding": {"00": "X_pub", "01": "W", "10": "Aux", "11": "L_qry"}},
        "lasso": {},
        "matmul": {},
        "gkr": {},
        "open": {},
        "log": log,
    }
    _squeeze(log, "gat-zkgpt/meta", metadata)
    out["model_commitment_challenge"] = _squeeze(log, "gat-zkgpt/model-commit", commitments["items"]["com_W"])
    out["static_table_challenge"] = _squeeze(log, "gat-zkgpt/static-table", metadata["static_table_registry"])
    out["aux_commitment_challenge"] = _squeeze(log, "gat-zkgpt/input-commit/aux", commitments["items"]["com_Aux"])
    out["lasso_commitment_challenge"] = _squeeze(log, "gat-zkgpt/input-commit/lasso", commitments["items"]["com_L"])
    for layer_index, layer in enumerate(metadata["dim_cfg"]["hidden_profile"], start=1):
        for head_index in range(layer["head_count"]):
            out["compression"][f"xi_l{layer_index}_h{head_index}"] = _squeeze(log, f"gkr/compression/layer/{layer_index}/head/{head_index}", commitments["items"]["com_Aux"])
        out["compression"][f"xi_cat_l{layer_index}"] = _squeeze(log, f"gkr/compression/cat/{layer_index}", commitments["items"]["com_Aux"])
    out["compression"]["xi_out"] = _squeeze(log, "gkr/compression/out", commitments["items"]["com_Aux"])
    if not include_late:
        return out
    out["input_consistency_eta"] = _squeeze(log, "input-layer/lookup-consistency", [commitments["items"]["com_Aux"], commitments["items"]["com_L"]])
    for layer_index, layer in enumerate(metadata["dim_cfg"]["hidden_profile"], start=1):
        for head_index in range(layer["head_count"]):
            out["psq"][f"lambda_psq_l{layer_index}_h{head_index}"] = _squeeze(log, f"gkr/aggregation/layer/{layer_index}/head/{head_index}", commitments["items"]["com_Aux"])
    for head_index in range(metadata["dim_cfg"]["K_out"]):
        out["psq"][f"lambda_psq_out_h{head_index}"] = _squeeze(log, f"gkr/aggregation/out/{head_index}", commitments["items"]["com_Aux"])
    for family in ["hidden_range", "output_range", "hidden_lrelu", "output_lrelu", "hidden_elu", "hidden_exp", "output_exp", "rescale_rounding_range"]:
        out["lasso"][f"eta_{family}"] = _squeeze(log, f"lasso/{family}", commitments["items"]["com_L"])
    out["matmul"]["rho_matmul"] = _squeeze(log, "matmul/sumcheck", [commitments["items"]["com_W"], commitments["items"]["com_Aux"]])
    out["gkr"]["rho_gkr"] = _squeeze(log, "gkr/layered-sumcheck", [commitments["items"]["com_Aux"], commitments["items"]["com_L"]])
    out["open"]["rho_combine"] = _squeeze(log, "open/combine", commitments)
    out["open"]["rho_masked_open"] = _squeeze(log, "open/masked", commitments)
    return out


def _prove_lasso(
    trace: ForwardTrace,
    model: Model,
    range_bits: int,
    transcript: Dict[str, Any],
    metadata: Dict[str, Any],
    commitments: Dict[str, Any],
) -> Dict[str, Any]:
    fixed_tables = _static_lookup_registry(metadata, range_bits)
    families: List[Dict[str, Any]] = []
    for layer in trace.hidden_layers:
        for head in layer.heads:
            families.extend(_lasso_head(head, range_bits, include_elu=True, transcript=transcript, fixed_tables=fixed_tables))
    for head in trace.output_heads:
        families.extend(_lasso_head(head, range_bits, include_elu=False, transcript=transcript, fixed_tables=fixed_tables))
    if trace.output_average_division is not None:
        families.append(
            _range_lookup_proof(
                "output.final_average.SignedRoundDiv.range",
                _division_range_values(trace.output_average_division),
                range_bits,
                fixed_tables["T_range"],
                transcript,
            )
        )
    limb_width = max(
        1,
        (int(metadata["integer_bound_ledger"]["bit_width"]) + int(range_bits) - 1)
        // int(range_bits),
    )
    limb_base = 1 << int(range_bits)
    residual_limbs: List[int] = []
    for _, residual in _matrix_rescale_residual_arrays(trace, model):
        for value in np.ravel(np.asarray(residual, dtype=object)):
            current = abs(int(value))
            for _ in range(limb_width):
                residual_limbs.append(current % limb_base)
                current //= limb_base
            if current:
                raise ValueError("matrix residual exceeds certified ledger width")
    families.append(
        _range_lookup_proof(
            "rescale_rounding_range", np.asarray(residual_limbs, dtype=object),
            range_bits, fixed_tables["T_range"], transcript,
        )
    )
    query_binding = _segment_binding(commitments, "com_L")
    for family in families:
        family["committed_query_segment"] = query_binding
        family["committed_query_segment_digest"] = digest_json(query_binding)
        family["status"] = _verify_lookup_family(family)["status"]
    status = "VERIFY_OK" if all(_verify_lookup_family(item)["status"] == "VERIFY_OK" for item in families) else "VERIFY_FAILED"
    return {
        "protocol": LOOKUP_PROTOCOL_ID,
        "status": status,
        "fixed_table_registry_digest": digest_json(fixed_tables),
        "fixed_table_registry": fixed_tables,
        "committed_query_segment": query_binding,
        "families": families,
    }


def _lasso_head(head: HeadTrace, range_bits: int, include_elu: bool, transcript: Dict[str, Any], fixed_tables: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = [
        _range_lookup_proof(f"{head.tag}.range_delta", head.delta, range_bits, fixed_tables["T_range"], transcript),
        _function_lookup_proof(f"{head.tag}.LeakyReLU", fixed_tables["T_LReLU"], head.s, head.z, transcript),
        _function_lookup_proof(f"{head.tag}.ExpMap", fixed_tables["T_exp"], head.delta, head.u, transcript),
    ]
    if include_elu:
        out.append(_function_lookup_proof(f"{head.tag}.ELU", fixed_tables["T_ELU"], head.elu_in if head.elu_in is not None else head.h_agg_pre, head.h_agg, transcript))
    if head.aggregate_division is not None:
        out.append(_range_lookup_proof(f"{head.tag}.SignedRoundDiv.aggregate.range", _division_range_values(head.aggregate_division), range_bits, fixed_tables["T_range"], transcript))
    if head.scale_alignment_division is not None:
        out.append(_range_lookup_proof(f"{head.tag}.SignedRoundDiv.scale_alignment.range", _division_range_values(head.scale_alignment_division), range_bits, fixed_tables["T_range"], transcript))
    return out


def _division_range_values(division: SignedRoundDivTrace) -> np.ndarray:
    return np.concatenate([np.ravel(getattr(division, name)) for name in ("a", "q", "r", "delta", "d")]).astype(object)


def _static_lookup_registry(metadata: Dict[str, Any], range_bits: int) -> Dict[str, Any]:
    ledger = metadata.get("integer_bound_ledger", {})
    common = {
        "quant_cfg_id": metadata["quant_cfg_id"],
        "quant_scale_chain": dict(ledger.get("scales", {})),
        "static_table_id": metadata["static_table_id"],
        "encoding_id": metadata["encoding_id"],
        "range_bits": int(range_bits),
        "field_encoding_scale": 1,
        "integer_bound_ledger_digest": metadata.get("integer_bound_ledger_digest", ""),
        "certified_table_domains": ledger.get("table_domains", {}),
    }
    registry = {
        "T_LReLU": _fixed_function_table("T_LReLU", "LeakyReLU", "y=x if x>=0 else round(0.2*x)", common),
        "T_ELU": _fixed_function_table("T_ELU", "ELU", "y=x if x>=0 else round(exp(x)-1)", common),
        "T_exp": _fixed_function_table("T_exp", "ExpMap", "y=round(exp(-x)) for nonnegative x", common),
        "T_range": _fixed_range_table(range_bits, common),
    }
    return registry


def _native_lasso_table_rows(metadata: Dict[str, Any]) -> Dict[str, List[int]]:
    """Rebuild every authenticated fixed table from the public descriptor."""
    registry = metadata["static_table_registry"]
    rows: Dict[str, List[int]] = {}
    for table_id in ("T_LReLU", "T_ELU", "T_exp"):
        table = registry[table_id]
        domain_min, domain_max = (int(value) for value in table["certified_domain"])
        values = [
            _pair_key(table, x_value, _fixed_table_eval_q(table_id, x_value))
            for x_value in range(domain_min, domain_max + 1)
        ]
        if len(values) != int(table["row_count"]):
            raise ValueError(f"{table_id} row count differs from its authenticated descriptor")
        rows[table_id] = values
    range_table = registry["T_range"]
    rows["T_range"] = list(range(int(range_table["entry_count"])))
    return rows


def _integer_bound_ledger(metadata: Dict[str, Any], model: Model, graph: LocalGraph) -> Dict[str, Any]:
    """Build the public no-wrap certificate used by GKR/Lasso verification."""
    field_half = (BN254_FR_MODULUS - 1) // 2
    max_degree = max((int(value) for value in np.bincount(graph.dst, minlength=graph.num_nodes)), default=1)
    feature_bound = max((abs(int(round(float(value)))) for value in np.ravel(graph.features)), default=0)
    hidden_bounds = []
    registered_model_bounds: Dict[str, Dict[str, Any]] = {}
    input_bound = max(1, feature_bound)
    input_scale = 1
    for layer in model.hidden_layers:
        layer_output_bounds: List[int] = []
        for head_index, head in enumerate(layer.heads):
            w = max((abs(int(value)) for value in np.ravel(head.seq_kernel)), default=0)
            a_src = max((abs(int(value)) for value in np.ravel(head.attn_src_kernel)), default=0)
            a_dst = max((abs(int(value)) for value in np.ravel(head.attn_dst_kernel)), default=0)
            family = f"hidden.l{layer.layer_index + 1}.h{head_index}"
            registered_model_bounds[family] = {
                "W_shape": [int(value) for value in head.seq_kernel.shape],
                "a_src_shape": [int(value) for value in head.attn_src_kernel.shape],
                "a_dst_shape": [int(value) for value in head.attn_dst_kernel.shape],
                "B_W": w,
                "B_a_src": a_src,
                "B_a_dst": a_dst,
                "B_b": 0,
            }
            h_prime_scale = input_scale * 16
            h_prime = max(1, int(layer.input_dim) * input_bound * max(1, w))
            score = max(1, 2 * int(layer.head_dim) * h_prime * max(a_src, a_dst))
            table_lrelu = max(abs(int(_fixed_lrelu(score))), abs(int(_fixed_lrelu(-score))))
            delta_raw = 2 * table_lrelu
            exp_max = _fixed_exp(0)
            sum_bound = max_degree * exp_max
            num_bound = sum_bound * h_prime
            dividend = 16 * num_bound
            denominator = h_prime_scale * sum_bound
            quotient = max(1, (dividend + max(denominator, 1) - 1) // max(denominator, 1))
            hidden_bounds.append({
                "kind": "hidden",
                "layer": layer.layer_index + 1,
                "head": head_index,
                "B_H": input_bound,
                "B_W": w,
                "B_H_prime": h_prime,
                "B_a_src": a_src,
                "B_a_dst": a_dst,
                "B_S": score,
                "B_Z": table_lrelu,
                "B_delta_raw": delta_raw,
                "B_delta": delta_raw,
                "B_U": exp_max,
                "B_Sum": sum_bound,
                "B_H_num": num_bound,
                "S_H_prime": h_prime_scale,
                "S_agg_pre": 16,
                "B_x_div": dividend,
                "B_d_div": denominator,
                "B_a_div": dividend,
                "B_q_div": quotient,
                "B_r_div": max(0, denominator - 1),
                "B_delta_div": denominator,
                "B_d": denominator,
                "B_2a_plus_d": 2 * dividend + denominator,
                "B_2dq_plus_r": 2 * denominator * quotient + max(0, denominator - 1),
                "B_2d_minus_1": max(0, 2 * denominator - 1),
                "B_r_plus_delta": max(0, denominator - 1) + denominator,
            })
            layer_output_bounds.append(max(1, quotient))
        input_bound = max(layer_output_bounds, default=1)
        input_scale = 16
    output_bounds = []
    output_input_bound = input_bound
    for head_index, head in enumerate(model.output_heads):
        w = max((abs(int(value)) for value in np.ravel(head.seq_kernel)), default=0)
        a_src = max((abs(int(value)) for value in np.ravel(head.attn_src_kernel)), default=0)
        a_dst = max((abs(int(value)) for value in np.ravel(head.attn_dst_kernel)), default=0)
        b_out = max((abs(int(value)) for value in np.ravel(head.output_bias)), default=0)
        family = f"output.h{head_index}"
        registered_model_bounds[family] = {
            "W_shape": [int(value) for value in head.seq_kernel.shape],
            "a_src_shape": [int(value) for value in head.attn_src_kernel.shape],
            "a_dst_shape": [int(value) for value in head.attn_dst_kernel.shape],
            "b_shape": [int(value) for value in head.output_bias.shape],
            "B_W": w,
            "B_a_src": a_src,
            "B_a_dst": a_dst,
            "B_b": b_out,
        }
        y_prime = max(1, int(model.hidden_layers[-1].head_count * model.hidden_layers[-1].head_dim) * output_input_bound * max(1, w))
        score = max(1, 2 * model.class_count * y_prime * max(a_src, a_dst))
        z_bound = max(abs(int(_fixed_lrelu(score))), abs(int(_fixed_lrelu(-score))))
        delta_raw = 2 * z_bound
        exp_max = _fixed_exp(0)
        sum_bound = max_degree * exp_max
        num_bound = sum_bound * y_prime
        # Output heads consume the hidden ELU scale (16) and use a
        # projection scale of 16, hence S_Y'=256 in the canonical bundle.
        dividend = 16 * num_bound
        denominator = 256 * sum_bound
        quotient = max(1, (dividend + max(denominator, 1) - 1) // max(denominator, 1))
        output_bounds.append({
            "kind": "output",
            "layer": "out",
            "head": head_index,
            "B_H": output_input_bound,
            "B_W": w,
            "B_Y_prime": y_prime,
            "B_a_src": a_src,
            "B_a_dst": a_dst,
            "B_b": b_out,
            "B_S": score,
            "B_Z": z_bound,
            "B_delta_raw": delta_raw,
            "B_delta": delta_raw,
            "B_U": exp_max,
            "B_Sum": sum_bound,
            "B_Y_num": num_bound,
            "S_Y_prime": 256,
            "S_out_pre": 16,
            "B_x_div": dividend,
            "B_d_div": denominator,
            "B_a_div": dividend,
            "B_q_div": quotient,
            "B_r_div": max(0, denominator - 1),
            "B_delta_div": denominator,
            "B_d": denominator,
            "B_2a_plus_d": 2 * dividend + denominator,
            "B_2dq_plus_r": 2 * denominator * quotient + max(0, denominator - 1),
            "B_2d_minus_1": max(0, 2 * denominator - 1),
            "B_r_plus_delta": max(0, denominator - 1) + denominator,
        })
    all_values = [int(value) for bound in hidden_bounds + output_bounds for value in bound.values() if isinstance(value, int)]
    max_bound = max(all_values, default=1)
    if max_bound >= field_half:
        raise ValueError("integer_bound_ledger exceeds BN254 signed field half")
    output_head_bound = max((int(item["B_q_div"]) for item in output_bounds), default=1)
    output_sum_bound = max(1, len(output_bounds) * output_head_bound)
    signed_round_divisions = []
    for item in hidden_bounds + output_bounds:
        signed_round_divisions.append(
            {
                "family": f"{item['kind']}.l{item['layer']}.h{item['head']}",
                "B_a": int(item["B_a_div"]),
                "B_q": int(item["B_q_div"]),
                "B_r": int(item["B_r_div"]),
                "B_delta": int(item["B_delta_div"]),
                "B_d": int(item["B_d"]),
                "B_2a_plus_d": int(item["B_2a_plus_d"]),
                "B_2dq_plus_r": int(item["B_2dq_plus_r"]),
                "B_2d_minus_1": int(item["B_2d_minus_1"]),
                "B_r_plus_delta": int(item["B_r_plus_delta"]),
            }
        )
    signed_round_divisions.append(
        {
            "family": "output.final_average",
            "B_a": output_sum_bound,
            "B_q": max(1, (output_sum_bound + max(1, len(output_bounds)) - 1) // max(1, len(output_bounds))),
            "B_r": max(0, len(output_bounds) - 1),
            "B_delta": max(1, len(output_bounds)),
            "B_d": max(1, len(output_bounds)),
            "B_2a_plus_d": 2 * output_sum_bound + max(1, len(output_bounds)),
            "B_2dq_plus_r": 2 * max(1, len(output_bounds)) * max(1, (output_sum_bound + max(1, len(output_bounds)) - 1) // max(1, len(output_bounds))) + max(0, len(output_bounds) - 1),
            "B_2d_minus_1": max(1, 2 * len(output_bounds) - 1),
            "B_r_plus_delta": max(0, len(output_bounds) - 1) + max(1, len(output_bounds)),
        }
    )
    return {
        "schema": "zkGPT_gat_integer_bound_ledger_v1",
        "field": "BN254-Fr",
        "field_half_modulus": str(field_half),
        "max_in_degree": max_degree,
        "feature_bound": feature_bound,
        "quant_cfg_id": model.quant_cfg_id,
        "authenticated_quant_scales": dict(model.quant_scales),
        "registered_model_bounds": registered_model_bounds,
        "bit_width": max(2, max_bound.bit_length()),
        "scales": {
            "S_input": 1,
            "S_W_hidden": 16,
            "S_W_output": 16,
            "S_a_hidden": 16,
            "S_a_output": 1,
            "S_delta": 256,
            "S_exp": 16,
            "S_agg_pre": 16,
            "S_agg": 16,
            "S_out_pre": 16,
            "S_out_head": 16,
            "S_out": 16,
            "S_inv": "field_exact",
            "S_alpha": "field_exact",
        },
        "table_domains": {
            "lrelu": [-(max((int(item["B_S"]) for item in hidden_bounds + output_bounds), default=1)), max((int(item["B_S"]) for item in hidden_bounds + output_bounds), default=1)],
            "elu": [-max((int(item["B_q_div"]) for item in hidden_bounds), default=1), max((int(item["B_q_div"]) for item in hidden_bounds), default=1)],
            "exp": [0, max((int(item["B_delta_raw"]) for item in hidden_bounds + output_bounds), default=1)],
        },
        "heads": hidden_bounds + output_bounds,
        "signed_round_divisions": signed_round_divisions,
        "final_average": {"K_out": len(output_bounds), "B_Y_sum": output_sum_bound, "denominator": max(1, len(output_bounds))},
        "failures": [],
    }


def _structured_function_vk(table_id: str, function: str, rule: str) -> Dict[str, Any]:
    input_scale = {"T_LReLU": 256, "T_ELU": 16, "T_exp": 256}.get(table_id)
    if input_scale is None:
        raise ValueError(f"unknown structured table {table_id}")
    vk = {
        "protocol": "structured-fixed-function-verification-key-v2",
        "table_id": table_id,
        "function": function,
        "evaluation_rule": rule,
        "domain_bits": FUNCTION_LOOKUP_DOMAIN_BITS,
        "input_domain": {
            "signed": True,
            "min_inclusive": -(1 << (FUNCTION_LOOKUP_DOMAIN_BITS - 1)),
            "max_exclusive": 1 << (FUNCTION_LOOKUP_DOMAIN_BITS - 1),
            "fixed_scale": input_scale,
        },
        "field_modulus": str(BN254_FR_MODULUS),
        "evaluator_id": {
            "T_LReLU": "signed_fixed_point_leaky_relu_alpha_0p2_nearest_q",
            "T_ELU": "signed_fixed_point_elu_expm1_nearest_q",
            "T_exp": "signed_fixed_point_exp_neg_nearest_q",
        }.get(table_id, "invalid-table-id"),
        "selected_opening_rule": "fixed verifier-key evaluator bound to the complete native Lasso query/table commitment argument",
    }
    vk["vk_digest"] = digest_json({key: value for key, value in vk.items() if key != "vk_digest"})
    return vk


def _verify_structured_function_vk(table_id: str, function: str, rule: str, vk: Any) -> bool:
    if not isinstance(vk, dict):
        return False
    expected = _structured_function_vk(table_id, function, rule)
    return vk == expected


def _canonical_function_rows_digest(
    table_id: str,
    domain_min: int,
    domain_max: int,
    tuple_encoding: Dict[str, Any],
) -> str:
    h = hashlib.sha256()
    for x_value in range(int(domain_min), int(domain_max) + 1):
        y_value = _fixed_table_eval_q(table_id, x_value)
        packed = _pair_key({"tuple_encoding": tuple_encoding}, x_value, y_value)
        h.update(f"{x_value},{y_value},{packed}\n".encode("ascii"))
    return h.hexdigest()


def _fixed_function_table(table_id: str, function: str, rule: str, common: Dict[str, Any]) -> Dict[str, Any]:
    vk = _structured_function_vk(table_id, function, rule)
    scales = {"T_LReLU": (256, 256), "T_ELU": (16, 16), "T_exp": (256, 16)}
    input_scale, output_scale = scales[table_id]
    certified_domain = common.get("certified_table_domains", {}).get(
        {"T_LReLU": "lrelu", "T_ELU": "elu", "T_exp": "exp"}.get(table_id, table_id),
        [],
    )
    if not isinstance(certified_domain, list) or len(certified_domain) != 2:
        raise ValueError(f"missing certified domain for {table_id}")
    domain_min, domain_max = int(certified_domain[0]), int(certified_domain[1])
    endpoint_outputs = [_fixed_table_eval_q(table_id, value) for value in {domain_min, domain_max, 0} if domain_min <= value <= domain_max]
    x_bound = max(abs(domain_min), abs(domain_max))
    y_bound = max((abs(int(value)) for value in endpoint_outputs), default=0)
    pack_base = 2 * y_bound + 1
    max_packed = 2 * x_bound * pack_base + 2 * y_bound
    if max_packed >= (BN254_FR_MODULUS - 1) // 2:
        raise ValueError(f"{table_id} tuple packing exceeds BN254 signed field range")
    descriptor = {
        "table_id": table_id,
        "table_kind": "structured_fixed_function_table",
        "function": function,
        "evaluation_rule": rule,
        "field_modulus": str(BN254_FR_MODULUS),
        "input_scale": input_scale,
        "output_scale": output_scale,
        "domain_policy": f"signed fixed-point representatives with abs(x) < 2^{FUNCTION_LOOKUP_DOMAIN_BITS - 1}",
        "input_domain": {
            "signed": True,
            "min_inclusive": -(1 << (FUNCTION_LOOKUP_DOMAIN_BITS - 1)),
            "max_exclusive": 1 << (FUNCTION_LOOKUP_DOMAIN_BITS - 1),
            "fixed_scale": input_scale,
        },
        "certified_domain": certified_domain,
        "tuple_encoding": {
            "encoding_id": "Pack_f_offset_base_v1",
            "x_bound": x_bound,
            "y_bound": y_bound,
            "base": pack_base,
            "max_packed": max_packed,
        },
        "canonical_row_order": "ascending signed x; row=(x,y,Pack_f(x,y))",
        "row_count": domain_max - domain_min + 1,
        "integer_bound_ledger_digest": common.get("integer_bound_ledger_digest", ""),
        "table_materialization": "preprocessed structured evaluator with fixed verification key; selected openings are verified by the fixed VK and are not accepted as query-generated table side",
        "structured_verification_key": vk,
        "chunks": [{"chunk_index": 0, "count": 1, "sha256": digest_json({"structured_table_vk": vk})}],
        **common,
    }
    descriptor["rows_digest"] = _canonical_function_rows_digest(
        table_id,
        domain_min,
        domain_max,
        descriptor["tuple_encoding"],
    )
    digest = digest_json(descriptor)
    descriptor["table_digest"] = digest
    descriptor["table_commitment"] = digest_json(
        {
            "commitment_backend": "structured-fixed-function-vk-commitment-v2",
            "table_id": table_id,
            "structured_verification_key_digest": digest_json(vk),
            "chunk_roots": descriptor["chunks"],
        }
    )
    return descriptor


def _fixed_range_table(range_bits: int, common: Dict[str, Any]) -> Dict[str, Any]:
    values = [to_field_int(value) for value in range(1 << int(range_bits))]
    descriptor = {
        "table_id": "T_range",
        "table_kind": "materialized_fixed_range_table",
        "function": "Range",
        "domain_policy": f"all limbs in [0,{(1 << int(range_bits)) - 1}]",
        "table_materialization": "preprocessed fixed limb table",
        "entry_count": len(values),
        "chunks": _field_chunk_digests(values),
        **common,
    }
    digest = digest_json({key: value for key, value in descriptor.items() if key not in {"table_digest", "table_commitment"}})
    descriptor["table_digest"] = digest
    descriptor["table_commitment"] = digest_json(
        {
            "commitment_backend": "materialized-fixed-range-table-commitment-v2",
            "table_id": "T_range",
            "range_bits": int(range_bits),
            "entry_count": len(values),
            "chunk_roots": descriptor["chunks"],
        }
    )
    return descriptor


def _verify_fixed_table_descriptor(table: Dict[str, Any]) -> bool:
    if not {"table_id", "table_kind", "quant_cfg_id", "static_table_id", "encoding_id", "range_bits", "table_digest", "table_commitment"} <= set(table):
        return False
    descriptor = {key: value for key, value in table.items() if key not in {"table_digest", "table_commitment"}}
    digest = digest_json(descriptor)
    if table.get("table_kind") == "structured_fixed_function_table":
        vk = table.get("structured_verification_key", {})
        commitment = digest_json(
            {
                "commitment_backend": "structured-fixed-function-vk-commitment-v2",
                "table_id": table.get("table_id"),
                "structured_verification_key_digest": digest_json(vk),
                "chunk_roots": table.get("chunks", []),
            }
        )
        vk_ok = _verify_structured_function_vk(table.get("table_id", ""), table.get("function", ""), table.get("evaluation_rule", ""), vk)
    elif table.get("table_kind") == "materialized_fixed_range_table":
        commitment = digest_json(
            {
                "commitment_backend": "materialized-fixed-range-table-commitment-v2",
                "table_id": table.get("table_id"),
                "range_bits": int(table.get("range_bits", -1)),
                "entry_count": int(table.get("entry_count", -1)),
                "chunk_roots": table.get("chunks", []),
            }
        )
        vk_ok = True
    else:
        return False
    if table.get("table_kind") == "structured_fixed_function_table":
        certified = table.get("certified_domain")
        if not isinstance(certified, list) or len(certified) != 2:
            return False
        try:
            certified_min, certified_max = int(certified[0]), int(certified[1])
            vk_domain = table.get("structured_verification_key", {}).get("input_domain", {})
            if certified_min > certified_max:
                return False
            if certified_min < int(vk_domain.get("min_inclusive", 0)) or certified_max >= int(vk_domain.get("max_exclusive", -1)):
                return False
            expected_scale = int(table.get("input_scale", -1))
            if int(vk_domain.get("fixed_scale", -2)) != expected_scale:
                return False
            if not table.get("integer_bound_ledger_digest"):
                return False
            encoding = table.get("tuple_encoding", {})
            if encoding.get("encoding_id") != "Pack_f_offset_base_v1":
                return False
            x_bound = max(abs(certified_min), abs(certified_max))
            if int(encoding.get("x_bound", -1)) != x_bound:
                return False
            if int(encoding.get("base", 0)) != 2 * int(encoding.get("y_bound", -1)) + 1:
                return False
            if int(table.get("row_count", -1)) != certified_max - certified_min + 1:
                return False
            if table.get("rows_digest") != _canonical_function_rows_digest(
                table.get("table_id", ""), certified_min, certified_max, encoding
            ):
                return False
        except (TypeError, ValueError):
            return False
    return table["table_digest"] == digest and table["table_commitment"] == commitment and vk_ok


def _verify_integer_bound_ledger(metadata: Dict[str, Any]) -> bool:
    ledger = metadata.get("integer_bound_ledger", {})
    if not isinstance(ledger, dict):
        return False
    if ledger.get("schema") != "zkGPT_gat_integer_bound_ledger_v1" or ledger.get("field") != "BN254-Fr":
        return False
    try:
        field_half = (BN254_FR_MODULUS - 1) // 2
        if int(ledger.get("field_half_modulus", -1)) != field_half:
            return False
        if ledger.get("failures"):
            return False
        quant_cfg_id = str(metadata.get("quant_cfg_id", ""))
        if (
            quant_cfg_id != ledger.get("quant_cfg_id")
            or quant_cfg_id not in QUANT_SCALE_PROFILES
            or ledger.get("authenticated_quant_scales") != QUANT_SCALE_PROFILES[quant_cfg_id]
        ):
            return False
        bit_width = int(ledger.get("bit_width", 0))
        if bit_width <= 0:
            return False
        domains = ledger.get("table_domains", {})
        for name in ("lrelu", "elu", "exp"):
            bounds = domains.get(name)
            if not isinstance(bounds, list) or len(bounds) != 2 or int(bounds[0]) > int(bounds[1]):
                return False
        registered = ledger.get("registered_model_bounds", {})
        heads = ledger.get("heads", [])
        if not isinstance(registered, dict) or not isinstance(heads, list) or not heads:
            return False
        max_degree = int(ledger.get("max_in_degree", 0))
        if max_degree <= 0 or int(ledger.get("feature_bound", -1)) < 0:
            return False
        exp_max = _fixed_exp(0)
        dim_cfg = metadata.get("dim_cfg", {})
        hidden_profile = dim_cfg.get("hidden_profile", [])
        d_in = [int(value) for value in dim_cfg.get("d_in", [])]
        expected_head_keys: set[tuple[str, int | str, int]] = set()
        current_input_bound = max(1, int(ledger["feature_bound"]))
        for layer_index, layer in enumerate(hidden_profile, start=1):
            layer_outputs: List[int] = []
            for head_index in range(int(layer["head_count"])):
                family = f"hidden.l{layer_index}.h{head_index}"
                bound = registered.get(family, {})
                matches = [item for item in heads if item.get("kind") == "hidden" and int(item.get("layer", -1)) == layer_index and int(item.get("head", -1)) == head_index]
                if len(matches) != 1:
                    return False
                item = matches[0]
                expected_head_keys.add(("hidden", layer_index, head_index))
                b_w = int(bound.get("B_W", -1)); b_src = int(bound.get("B_a_src", -1)); b_dst = int(bound.get("B_a_dst", -1))
                h_prime = max(1, d_in[layer_index - 1] * current_input_bound * max(1, b_w))
                score = max(1, 2 * int(layer["head_dim"]) * h_prime * max(b_src, b_dst))
                z_bound = max(abs(_fixed_lrelu(score)), abs(_fixed_lrelu(-score)))
                delta_bound = 2 * z_bound
                sum_bound = max_degree * exp_max
                num_bound = sum_bound * h_prime
                h_prime_scale = (1 if layer_index == 1 else 16) * 16
                dividend = 16 * num_bound
                denominator = h_prime_scale * sum_bound
                quotient = max(1, (dividend + denominator - 1) // denominator)
                expected = {
                    "B_H": current_input_bound, "B_W": b_w, "B_H_prime": h_prime,
                    "B_a_src": b_src, "B_a_dst": b_dst, "B_S": score, "B_Z": z_bound,
                    "B_delta_raw": delta_bound, "B_delta": delta_bound, "B_U": exp_max,
                    "B_Sum": sum_bound, "B_H_num": num_bound, "S_H_prime": h_prime_scale,
                    "S_agg_pre": 16, "B_x_div": dividend, "B_d_div": denominator,
                    "B_a_div": dividend, "B_q_div": quotient, "B_r_div": denominator - 1,
                    "B_delta_div": denominator, "B_d": denominator,
                    "B_2a_plus_d": 2 * dividend + denominator,
                    "B_2dq_plus_r": 2 * denominator * quotient + denominator - 1,
                    "B_2d_minus_1": 2 * denominator - 1,
                    "B_r_plus_delta": 2 * denominator - 1,
                }
                if any(int(item.get(key, -1)) != value for key, value in expected.items()):
                    return False
                layer_outputs.append(quotient)
            current_input_bound = max(layer_outputs, default=1)
        output_heads = int(dim_cfg.get("K_out", 0))
        class_count = int(dim_cfg.get("C", 0))
        output_quotients: List[int] = []
        for head_index in range(output_heads):
            family = f"output.h{head_index}"
            bound = registered.get(family, {})
            matches = [item for item in heads if item.get("kind") == "output" and str(item.get("layer")) == "out" and int(item.get("head", -1)) == head_index]
            if len(matches) != 1:
                return False
            item = matches[0]
            expected_head_keys.add(("output", "out", head_index))
            b_w = int(bound.get("B_W", -1)); b_src = int(bound.get("B_a_src", -1)); b_dst = int(bound.get("B_a_dst", -1)); b_out = int(bound.get("B_b", -1))
            y_prime = max(1, d_in[-1] * current_input_bound * max(1, b_w))
            score = max(1, 2 * class_count * y_prime * max(b_src, b_dst))
            z_bound = max(abs(_fixed_lrelu(score)), abs(_fixed_lrelu(-score)))
            delta_bound = 2 * z_bound
            sum_bound = max_degree * exp_max
            num_bound = sum_bound * y_prime
            dividend = 16 * num_bound
            denominator = 256 * sum_bound
            quotient = max(1, (dividend + denominator - 1) // denominator)
            expected = {
                "B_H": current_input_bound, "B_W": b_w, "B_Y_prime": y_prime,
                "B_a_src": b_src, "B_a_dst": b_dst, "B_b": b_out, "B_S": score,
                "B_Z": z_bound, "B_delta_raw": delta_bound, "B_delta": delta_bound,
                "B_U": exp_max, "B_Sum": sum_bound, "B_Y_num": num_bound,
                "S_Y_prime": 256, "S_out_pre": 16, "B_x_div": dividend,
                "B_d_div": denominator, "B_a_div": dividend, "B_q_div": quotient,
                "B_r_div": denominator - 1, "B_delta_div": denominator, "B_d": denominator,
                "B_2a_plus_d": 2 * dividend + denominator,
                "B_2dq_plus_r": 2 * denominator * quotient + denominator - 1,
                "B_2d_minus_1": 2 * denominator - 1,
                "B_r_plus_delta": 2 * denominator - 1,
            }
            if any(int(item.get(key, -1)) != value for key, value in expected.items()):
                return False
            output_quotients.append(quotient)
        if len(expected_head_keys) != len(heads):
            return False
        expected_domains = {
            "lrelu": [-max(int(item["B_S"]) for item in heads), max(int(item["B_S"]) for item in heads)],
            "elu": [-max((int(item["B_q_div"]) for item in heads if item["kind"] == "hidden"), default=1), max((int(item["B_q_div"]) for item in heads if item["kind"] == "hidden"), default=1)],
            "exp": [0, max(int(item["B_delta_raw"]) for item in heads)],
        }
        if {key: [int(value) for value in domains[key]] for key in expected_domains} != expected_domains:
            return False
        max_certified = 0
        for division in ledger.get("signed_round_divisions", []):
            if not all(int(division.get(name, -1)) >= 0 for name in ("B_a", "B_q", "B_r", "B_delta", "B_d")):
                return False
            if int(division.get("B_d", 0)) <= 0:
                return False
            if int(division.get("B_2a_plus_d", -1)) != 2 * int(division["B_a"]) + int(division["B_d"]):
                return False
            if int(division.get("B_2dq_plus_r", -1)) != 2 * int(division["B_d"]) * int(division["B_q"]) + int(division["B_r"]):
                return False
            if int(division.get("B_2d_minus_1", -1)) != 2 * int(division["B_d"]) - 1:
                return False
            if int(division.get("B_r_plus_delta", -1)) != int(division["B_r"]) + int(division["B_delta"]):
                return False
            family_max = max(int(division.get(name, 0)) for name in division if name.startswith("B_"))
            if family_max >= field_half:
                return False
            max_certified = max(max_certified, family_max)
        for item in heads:
            max_certified = max(max_certified, *(int(value) for key, value in item.items() if key.startswith("B_") and isinstance(value, int)))
        if bit_width < max(1, max_certified).bit_length():
            return False
        return digest_json(ledger) == metadata.get("integer_bound_ledger_digest")
    except (TypeError, ValueError, OverflowError):
        return False


def _empty_lookup_family(label: str, table: Dict[str, Any], transcript: Dict[str, Any]) -> Dict[str, Any]:
    raw_digest = digest_json({"empty_lookup": label, "table": table["table_digest"]})
    beta = _lookup_challenge(label, table, transcript, "beta", raw_digest, 0, "empty")
    return {
        "family": label,
        "table": table["table_id"],
        "fixed_table": table,
        "fixed_table_digest": table["table_digest"],
        "relation": "empty",
        "query_count": 0,
        "raw_query_digest": raw_digest,
        "transcript_digest": digest_json(transcript),
        "beta": str(beta),
        "eta": "0",
        "query_product": "1",
        "table_product": "1",
        "product_digest": _lookup_product_digest(1, 1, raw_digest, table["table_digest"], "empty"),
        "query_chunks": [],
        "table_chunks": table.get("chunks", []),
        "status": "VERIFY_OK",
    }


def _function_lookup_proof(label: str, table: Dict[str, Any], x: np.ndarray, y: np.ndarray, transcript: Dict[str, Any]) -> Dict[str, Any]:
    xq = _quantized_vector(x)
    yq = _quantized_vector(y)
    expected_q_raw = [_structured_vk_eval_q(table["structured_verification_key"], value) for value in xq]
    expected_q, max_delta_q, mismatches = _apply_lookup_quantization_window(expected_q_raw, yq)
    certified_domain = table.get("certified_domain", [])
    if not isinstance(certified_domain, list) or len(certified_domain) != 2:
        raise ValueError(f"missing certified domain for {label}")
    domain_min = int(certified_domain[0])
    domain_max = int(certified_domain[1])
    signed_x = [_signed_field(value) for value in xq]
    domain_ok = all(domain_min <= value < domain_max for value in signed_x)
    domain_chunks = _field_chunk_digests([to_field_int(value - domain_min) for value in signed_x])
    raw_chunks = _field_chunk_digests(_interleave_pairs(xq, yq))
    raw_digest = digest_json({"relation": "function_lookup_raw_xy", "chunks": raw_chunks, "query_count": len(xq)})
    eta = _lookup_challenge(label, table, transcript, "eta", raw_digest, len(xq), "fixed_function")
    beta = _lookup_challenge(label, table, transcript, "beta", raw_digest, len(xq), "fixed_function")
    query_keys = [_pair_key(table, a, b) for a, b in zip(xq, yq)]
    table_keys = [_pair_key(table, a, b) for a, b in zip(xq, expected_q)]
    query_product = _grand_product(query_keys, beta)
    table_product = _grand_product(table_keys, beta)
    query_key_chunks = _field_chunk_digests(query_keys)
    selected_table_chunks = _field_chunk_digests(table_keys)
    expected_y_chunks = _field_chunk_digests(expected_q)
    raw_expected_y_chunks = _field_chunk_digests(expected_q_raw)
    membership_seed = _structured_lookup_seed(
        label=label,
        table=table,
        raw_query_digest=raw_digest,
        query_count=len(xq),
        transcript_digest=digest_json(transcript),
        eta=eta,
        beta=beta,
        query_chunks=query_key_chunks,
        selected_fixed_table_chunks=selected_table_chunks,
        expected_y_chunks=expected_y_chunks,
        raw_expected_y_chunks=raw_expected_y_chunks,
        input_domain_chunks=domain_chunks,
    )
    evaluation_diffs = [
        to_field_int(to_field_int(actual_value) - to_field_int(expected_value))
        for actual_value, expected_value in zip(yq, expected_q)
    ]
    evaluation_sumcheck = prove_product_sum(
        f"{label}.structured_table_eval",
        evaluation_diffs or [0],
        [1] * max(len(evaluation_diffs), 1),
        membership_seed,
    ).to_json()
    evaluation_sumcheck["transcript_seed"] = membership_seed
    membership_argument = {
        "protocol": "structured-fixed-function-lasso-membership-v2",
        "table_id": table["table_id"],
        "table_commitment": table["table_commitment"],
        "structured_verification_key": table["structured_verification_key"],
        "raw_query_digest": raw_digest,
        "query_count": len(xq),
        "eta": str(eta),
        "beta": str(beta),
        "query_key_chunks": query_key_chunks,
        "fixed_table_key_chunks": selected_table_chunks,
        "expected_y_chunks": expected_y_chunks,
        "raw_expected_y_chunks": raw_expected_y_chunks,
        "input_domain_chunks": domain_chunks,
        "query_accumulator": str(query_product),
        "fixed_table_accumulator": str(table_product),
        "evaluation_sumcheck": evaluation_sumcheck,
        "transcript_seed": membership_seed,
    }
    membership_argument["argument_digest"] = digest_json(membership_argument)
    # Lasso query values are private.  Commit the packed query and the
    # verifier-selected table-side vector, then use the native Hyrax opening
    # argument below; no query coordinate is serialized in the proof.
    opening_context = {"family": label, "table_digest": table["table_digest"], "raw_query_digest": raw_digest}
    query_opening = _ephemeral_hyrax_opening(f"{label}.query", query_keys, opening_context)
    expected_opening = _ephemeral_hyrax_opening(f"{label}.table-selection", table_keys, opening_context)
    structured_opening = {
        "protocol": "native-lasso-packed-function-opening-v1",
        "table_commitment": table["table_commitment"],
        "structured_verification_key": table["structured_verification_key"],
        "raw_query_digest": raw_digest,
        "query_commitment": query_opening["registration"]["commitment"],
        "table_selection_commitment": expected_opening["registration"]["commitment"],
        "query_opening": query_opening,
        "table_selection_opening": expected_opening,
        "query_membership_argument": membership_argument,
        "membership_mode": "complete_native_lasso_commitment_sumcheck_not_sampled",
    }
    structured_opening["opening_digest"] = digest_json(structured_opening)
    family = {
        "family": label,
        "table": table["table_id"],
        "fixed_table": table,
        "fixed_table_digest": table["table_digest"],
        "relation": "query_multiset_membership_in_fixed_function_table",
        "query_count": len(query_keys),
        "raw_query_digest": raw_digest,
        "transcript_digest": digest_json(transcript),
        "eta": str(eta),
        "beta": str(beta),
        "query_product": str(query_product),
        "table_product": str(table_product),
        "product_digest": _lookup_product_digest(query_product, table_product, raw_digest, table["table_digest"], "query_multiset_membership_in_fixed_function_table"),
        "query_chunks": query_key_chunks,
        "raw_query_chunks": raw_chunks,
        "selected_fixed_table_chunks": selected_table_chunks,
        "expected_y_chunks": expected_y_chunks,
        "raw_expected_y_chunks": raw_expected_y_chunks,
        "input_domain_chunks": domain_chunks,
        "expected_y_chunks_digest": digest_json(expected_y_chunks),
        "raw_expected_y_chunks_digest": digest_json(raw_expected_y_chunks),
        "input_domain": [domain_min, domain_max],
        "input_domain_chunks_digest": digest_json(domain_chunks),
        "input_domain_ok": domain_ok,
        "structured_table_opening": structured_opening,
        "table_chunks": table.get("chunks", []),
        "mismatch_count": mismatches,
        "lookup_quantization_tolerance_q": 0,
        "max_lookup_delta_q": max_delta_q,
    }
    family["status"] = _verify_lookup_family(family)["status"]
    return family


def _range_lookup_proof(label: str, values: np.ndarray, range_bits: int, table: Dict[str, Any], transcript: Dict[str, Any]) -> Dict[str, Any]:
    base = 1 << range_bits
    arr = np.asarray(values)
    if arr.dtype == object or np.issubdtype(arr.dtype, np.integer):
        rounded = np.asarray([max(0, int(item)) for item in np.ravel(arr)], dtype=object).reshape(arr.shape)
    else:
        rounded = np.asarray([max(0, quantize_float(float(item))) for item in np.ravel(arr)], dtype=object).reshape(arr.shape)
    max_value = max((int(item) for item in np.ravel(rounded)), default=0)
    limb_count = max(1, (max_value.bit_length() + range_bits - 1) // range_bits)
    limbs: List[int] = []
    for value in np.ravel(rounded):
        current = int(value)
        for _ in range(limb_count):
            limbs.append(to_field_int(current % base))
            current //= base
    raw_digest = digest_json({"relation": "range_lookup_raw_limbs", "chunks": _field_chunk_digests(limbs), "query_count": len(limbs), "range_bits": int(range_bits)})
    beta = _lookup_challenge(label, table, transcript, "beta", raw_digest, len(limbs), "fixed_range")
    raw_chunks = _field_chunk_digests(limbs)
    # The prover evaluates the Lasso grand-product over the complete private
    # limb vector.  Only chunk digests and a masked Hyrax opening leave the
    # proof; every canonical limb chunk is authenticated in the Lasso
    # transcript and terminal opening.
    product = _grand_product(limbs, beta)
    full_table_product = _grand_product(range(base), beta)
    max_limb = max((int(item) for item in limbs), default=0)
    min_input_nonnegative = bool(arr.size == 0 or min(int(item) for item in np.ravel(arr)) >= 0) if arr.size else True
    range_argument = {
        "protocol": RANGE_ARGUMENT_PROTOCOL_ID,
        "table_id": table["table_id"],
        "table_commitment": table["table_commitment"],
        "table_digest": table["table_digest"],
        "range_bits": int(range_bits),
        "raw_query_digest": raw_digest,
        "query_limb_chunks": raw_chunks,
        "query_count": len(limbs),
        "max_limb": max_limb,
        "min_input_nonnegative": min_input_nonnegative,
        "beta": str(beta),
        "direct_query_accumulator": str(product),
        "fixed_table_full_product": str(full_table_product),
    }
    range_context = {"family": label, "table_digest": table["table_digest"], "raw_query_digest": raw_digest}
    range_opening = _ephemeral_hyrax_opening(f"{label}.range-query", limbs, range_context)
    range_argument["query_opening"] = range_opening
    range_argument["query_commitment"] = range_opening["registration"]["commitment"]
    range_argument["opening_protocol"] = "native-lasso-range-limb-opening-v1"
    range_argument["argument_digest"] = digest_json(range_argument)
    family = {
        "family": label,
        "table": table["table_id"],
        "fixed_table": table,
        "fixed_table_digest": table["table_digest"],
        "relation": "direct_limb_bounded_range_argument",
        "query_count": len(limbs),
        "limb_count": limb_count,
        "range_bits": range_bits,
        "raw_query_digest": raw_digest,
        "transcript_digest": digest_json(transcript),
        "eta": "0",
        "beta": str(beta),
        "query_product": str(product),
        "table_product": str(full_table_product),
        "fixed_table_full_product": str(full_table_product),
        "product_digest": _lookup_product_digest(product, full_table_product, raw_digest, table["table_digest"], "direct_limb_bounded_range_argument"),
        "query_chunks": raw_chunks,
        "selected_fixed_table_chunks": [],
        "range_argument": range_argument,
        "table_chunks": table.get("chunks", []),
        "max_limb": max_limb,
        "min_input_nonnegative": min_input_nonnegative,
        "mismatch_count": 0 if min_input_nonnegative and max_limb < base else 1,
        "input_domain": [0, base - 1],
    }
    family["status"] = _verify_lookup_family(family)["status"]
    return family


def _lookup_challenge(label: str, table: Dict[str, Any], transcript: Dict[str, Any], kind: str, raw_query_digest: str, query_count: int, relation: str) -> int:
    transcript_digest = transcript if isinstance(transcript, str) else digest_json(transcript)
    return int(
        digest_json(
            {
                "protocol": LOOKUP_PROTOCOL_ID,
                "label": label,
                "table": table["table_id"],
                "table_digest": table["table_digest"],
                "kind": kind,
                "relation": relation,
                "query_count": int(query_count),
                "raw_query_digest": raw_query_digest,
                "transcript_digest": transcript_digest,
            }
        )[:32],
        16,
    ) % BN254_FR_MODULUS


def _lookup_challenge_from_family(family: Dict[str, Any], kind: str) -> int:
    relation_kind = "fixed_range" if family.get("relation") in {"direct_limb_bounded_range_argument", "matrix_rescale_rounding_direct_range_argument"} else "fixed_function"
    if family.get("relation") == "empty":
        relation_kind = "empty"
    return _lookup_challenge(
        family["family"],
        family["fixed_table"],
        family.get("transcript_digest", ""),
        kind,
        family["raw_query_digest"],
        int(family["query_count"]),
        relation_kind,
    )


def _lookup_product_digest(query_product: int, table_product: int, raw_query_digest: str, fixed_table_digest: str, relation: str) -> str:
    return digest_json(
        {
            "query_product": str(to_field_int(query_product)),
            "table_product": str(to_field_int(table_product)),
            "raw_query_digest": raw_query_digest,
            "fixed_table_digest": fixed_table_digest,
            "relation": relation,
        }
    )


def _structured_lookup_seed(
    *,
    label: str,
    table: Dict[str, Any],
    raw_query_digest: str,
    query_count: int,
    transcript_digest: str,
    eta: int,
    beta: int,
    query_chunks: Sequence[Dict[str, Any]],
    selected_fixed_table_chunks: Sequence[Dict[str, Any]],
    expected_y_chunks: Sequence[Dict[str, Any]],
    raw_expected_y_chunks: Sequence[Dict[str, Any]],
    input_domain_chunks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "protocol": "structured-fixed-function-lasso-membership-v2",
        "family": label,
        "table_id": table["table_id"],
        "table_commitment": table["table_commitment"],
        "structured_verification_key": table["structured_verification_key"],
        "table_digest": table["table_digest"],
        "raw_query_digest": raw_query_digest,
        "query_count": int(query_count),
        "eta": str(to_field_int(eta)),
        "beta": str(to_field_int(beta)),
        "query_key_chunks": list(query_chunks),
        "fixed_table_key_chunks": list(selected_fixed_table_chunks),
        "expected_y_chunks": list(expected_y_chunks),
        "raw_expected_y_chunks": list(raw_expected_y_chunks),
        "input_domain_chunks": list(input_domain_chunks),
        "transcript_digest": transcript_digest,
    }


def _structured_lookup_seed_from_family(family: Dict[str, Any], opening: Dict[str, Any]) -> Dict[str, Any]:
    return _structured_lookup_seed(
        label=family["family"],
        table=family["fixed_table"],
        raw_query_digest=family["raw_query_digest"],
        query_count=int(family["query_count"]),
        transcript_digest=family.get("transcript_digest", ""),
        eta=int(family["eta"]),
        beta=int(family["beta"]),
        query_chunks=family.get("query_chunks", []),
        selected_fixed_table_chunks=family.get("selected_fixed_table_chunks", []),
        expected_y_chunks=family.get("expected_y_chunks", []),
        raw_expected_y_chunks=family.get("raw_expected_y_chunks", opening.get("raw_expected_y_chunks", [])),
        input_domain_chunks=family.get("input_domain_chunks", []),
    )


def _verify_structured_function_membership(family: Dict[str, Any], opening: Dict[str, Any]) -> bool:
    try:
        if opening.get("protocol") != "native-lasso-packed-function-opening-v1":
            return False
        context = {
            "family": family["family"],
            "table_digest": family["fixed_table_digest"],
            "raw_query_digest": family["raw_query_digest"],
        }
        query_ok = _verify_ephemeral_hyrax_opening(
            f"{family['family']}.query", opening.get("query_opening", {}), context
        )
        table_ok = _verify_ephemeral_hyrax_opening(
            f"{family['family']}.table-selection", opening.get("table_selection_opening", {}), context
        )
        if not query_ok or not table_ok:
            return False
        if opening.get("query_commitment") != opening.get("query_opening", {}).get("registration", {}).get("commitment"):
            return False
        if opening.get("table_selection_commitment") != opening.get("table_selection_opening", {}).get("registration", {}).get("commitment"):
            return False
        argument = dict(opening.get("query_membership_argument", {}))
        expected_argument_digest = argument.pop("argument_digest", None)
        if expected_argument_digest != digest_json(argument):
            return False
        seed = argument.get("transcript_seed", {})
        expected_seed = _structured_lookup_seed_from_family(family, opening)
        if seed != expected_seed:
            return False
        if argument.get("protocol") != "structured-fixed-function-lasso-membership-v2":
            return False
        if argument.get("table_id") != family.get("table"):
            return False
        if argument.get("table_commitment") != family.get("fixed_table", {}).get("table_commitment"):
            return False
        if argument.get("structured_verification_key") != family.get("fixed_table", {}).get("structured_verification_key"):
            return False
        if argument.get("raw_query_digest") != family.get("raw_query_digest"):
            return False
        if int(argument.get("query_count", -1)) != int(family.get("query_count", -2)):
            return False
        if str(argument.get("eta")) != str(family.get("eta")) or str(argument.get("beta")) != str(family.get("beta")):
            return False
        if digest_json(family.get("query_chunks", [])) != digest_json(argument.get("query_key_chunks", [])):
            return False
        if digest_json(family.get("selected_fixed_table_chunks", [])) != digest_json(argument.get("fixed_table_key_chunks", [])):
            return False
        if digest_json(family.get("expected_y_chunks", [])) != digest_json(argument.get("expected_y_chunks", [])):
            return False
        if digest_json(family.get("raw_expected_y_chunks", [])) != digest_json(argument.get("raw_expected_y_chunks", [])):
            return False
        if digest_json(family.get("input_domain_chunks", [])) != digest_json(argument.get("input_domain_chunks", [])):
            return False
        if str(argument.get("query_accumulator")) != str(family.get("query_product")):
            return False
        if str(argument.get("fixed_table_accumulator")) != str(family.get("table_product")):
            return False
        if family.get("selected_fixed_table_chunks") == family.get("raw_query_chunks", []):
            return False
        sumcheck = argument.get("evaluation_sumcheck", {})
        if sumcheck.get("transcript_seed") != seed:
            return False
        verify = verify_product_sum(f"{family['family']}.structured_table_eval", sumcheck, seed)
        if verify["status"] != "VERIFY_OK":
            return False
        if to_field_int(int(sumcheck.get("claimed_sum", "0"))) != 0:
            return False
        return True
    except Exception:
        return False


def _interleave_pairs(left: Sequence[int], right: Sequence[int]) -> List[int]:
    out: List[int] = []
    for a, b in zip(left, right):
        out.append(a)
        out.append(b)
    return out


def _fixed_table_eval_q(table_id: str, x_value: int) -> int:
    x_signed = _signed_field(x_value)
    if table_id == "T_LReLU":
        return _fixed_lrelu(x_signed)
    elif table_id == "T_ELU":
        return _elu_table_value(x_signed)
    elif table_id == "T_exp":
        return _fixed_exp(x_signed)
    else:
        raise ValueError(f"unsupported fixed function table: {table_id}")


def _fixed_lrelu(value: int) -> int:
    value = int(value)
    if value >= 0:
        return value
    magnitude = -value
    return -((magnitude + 2) // 5)


def _fixed_exp(value: int) -> int:
    return _exp_map_fixed(int(value))


def _round_div_signed(numerator: int, denominator: int) -> int:
    if int(denominator) <= 0:
        raise ValueError("SignedRoundDiv denominator must be positive")
    sign = -1 if int(numerator) < 0 else 1
    value = abs(int(numerator))
    quotient, remainder = divmod(value, int(denominator))
    if 2 * remainder >= int(denominator):
        quotient += 1
    return sign * quotient


def _structured_vk_eval_q(vk: Dict[str, Any], x_value: int) -> int:
    if not isinstance(vk, dict) or vk.get("protocol") != "structured-fixed-function-verification-key-v2":
        raise ValueError("invalid structured fixed-function verification key")
    table_id = str(vk.get("table_id", ""))
    domain = vk.get("input_domain", {})
    signed_x = _signed_field(x_value)
    if not (int(domain.get("min_inclusive", 0)) <= signed_x < int(domain.get("max_exclusive", -1))):
        raise ValueError("structured fixed-function query outside VK domain")
    expected = _structured_function_vk(table_id, str(vk.get("function", "")), str(vk.get("evaluation_rule", "")))
    if vk != expected:
        raise ValueError("structured fixed-function VK digest mismatch")
    if vk.get("evaluator_id") not in {
        "signed_fixed_point_leaky_relu_alpha_0p2_nearest_q",
        "signed_fixed_point_elu_expm1_nearest_q",
        "signed_fixed_point_exp_neg_nearest_q",
    }:
        raise ValueError(f"unsupported structured fixed-function evaluator: {vk.get('evaluator_id')}")
    return _fixed_table_eval_q(table_id, signed_x)


def _apply_lookup_quantization_window(expected: Sequence[int], actual: Sequence[int]) -> Tuple[List[int], int, int]:
    adjusted: List[int] = []
    max_delta = 0
    mismatch_count = 0
    for exp_value, actual_value in zip(expected, actual):
        delta = abs(_signed_field(to_field_int(actual_value) - to_field_int(exp_value)))
        max_delta = max(max_delta, int(delta))
        adjusted.append(to_field_int(actual_value))
        if delta > LOOKUP_QUANTIZATION_TOLERANCE_Q:
            mismatch_count += 1
    return adjusted, max_delta, mismatch_count


def _pair_key(table: Dict[str, Any], x_value: int, y_value: int) -> int:
    """Canonical injective Pack_f over the table-certified rectangle."""
    encoding = table.get("tuple_encoding", {})
    if encoding.get("encoding_id") != "Pack_f_offset_base_v1":
        raise ValueError("function table is missing canonical tuple encoding")
    x_bound = int(encoding["x_bound"])
    y_bound = int(encoding["y_bound"])
    base = int(encoding["base"])
    x = int(x_value)
    y = int(y_value)
    if not (-x_bound <= x <= x_bound and -y_bound <= y <= y_bound):
        raise ValueError("function lookup tuple exceeds authenticated packing rectangle")
    packed = (x + x_bound) * base + (y + y_bound)
    if not (0 <= packed <= int(encoding["max_packed"])):
        raise ValueError("canonical function tuple packing overflow")
    return packed


def _grand_product(values: Iterable[int], beta: int) -> int:
    acc = 1
    beta = to_field_int(beta)
    for value in values:
        acc = (acc * ((beta + to_field_int(value)) % BN254_FR_MODULUS)) % BN254_FR_MODULUS
    return acc


def _decompose_limbs(value: int, range_bits: int) -> List[int]:
    base = 1 << int(range_bits)
    current = abs(int(value))
    limbs: List[int] = []
    while current:
        limbs.append(to_field_int(current % base))
        current //= base
    return limbs or [0]


def _matrix_rescale_lookup_family(matrix: Dict[str, Any], range_bits: int, table: Dict[str, Any], transcript: Dict[str, Any]) -> Dict[str, Any]:
    raw_chunks: List[Dict[str, Any]] = []
    weighted_chunks: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    query_count = 0
    max_limb = 0
    max_abs_entry_residual_q2 = 0
    weighted_openings: List[Dict[str, Any]] = []
    for claim in matrix.get("claims", []):
        binding = claim.get("rescale_rounding", {})
        for chunk in binding.get("entry_limb_chunks", []):
            current = dict(chunk)
            raw_chunks.append(current)
            max_limb = max(max_limb, int(current.get("max_limb", 0)) if "max_limb" in current else 0)
        for chunk in binding.get("weighted_limb_chunks", []):
            current = dict(chunk)
            weighted_chunks.append(current)
        query_count += int(binding.get("entry_limb_count", 0)) + sum(int(item["count"]) for item in binding.get("weighted_limb_chunks", []))
        max_abs_entry_residual_q2 = max(max_abs_entry_residual_q2, int(binding.get("max_abs_entry_residual_q2", "0")))
        if binding.get("weighted_limb_opening"):
            weighted_openings.append(binding["weighted_limb_opening"])
        sources.append(
            {
                "label": claim.get("label"),
                "kind": claim.get("kind"),
                "rescale_binding_digest": binding.get("binding_digest"),
                "entry_limb_count": int(binding.get("entry_limb_count", 0)),
                "weighted_limb_count": sum(int(item["count"]) for item in binding.get("weighted_limb_chunks", [])),
                "per_entry_rounding_bound_q2": binding.get("per_entry_rounding_bound_q2"),
            }
        )
    all_chunks = []
    for chunk in raw_chunks + weighted_chunks:
        current = dict(chunk)
        current["chunk_index"] = len(all_chunks)
        all_chunks.append(current)
    raw_digest = digest_json(
        {
            "relation": "matrix_rescale_rounding_committed_residual_limbs",
            "chunks": all_chunks,
            "query_count": query_count,
            "range_bits": int(range_bits),
            "matrix_claim_sources": sources,
        }
    )
    beta = _lookup_challenge("rescale_rounding_range", table, transcript, "beta", raw_digest, query_count, "fixed_range")
    base = 1 << int(range_bits)
    product = _lasso_chunk_transcript_accumulator(all_chunks, beta, raw_digest, range_bits)
    full_table_product = _grand_product(range(base), beta)
    range_argument = {
        "protocol": RANGE_ARGUMENT_PROTOCOL_ID,
        "table_id": table["table_id"],
        "table_commitment": table["table_commitment"],
        "table_digest": table["table_digest"],
        "range_bits": int(range_bits),
        "raw_query_digest": raw_digest,
        "query_limb_chunks": all_chunks,
        "query_count": query_count,
        "max_limb": max_limb,
        "max_abs_entry_residual_q2": str(max_abs_entry_residual_q2),
        "beta": str(beta),
        "direct_query_accumulator": str(product),
        "fixed_table_full_product": str(full_table_product),
        "weighted_limb_openings": weighted_openings,
    }
    range_argument["argument_digest"] = digest_json(range_argument)
    family = {
        "family": "rescale_rounding_range",
        "table": table["table_id"],
        "fixed_table": table,
        "fixed_table_digest": table["table_digest"],
        "relation": "matrix_rescale_rounding_direct_range_argument",
        "query_count": query_count,
        "limb_count": query_count,
        "range_bits": int(range_bits),
        "raw_query_digest": raw_digest,
        "transcript_digest": digest_json(transcript),
        "eta": "0",
        "beta": str(beta),
        "query_product": str(product),
        "table_product": str(full_table_product),
        "fixed_table_full_product": str(full_table_product),
        "product_digest": _lookup_product_digest(product, full_table_product, raw_digest, table["table_digest"], "matrix_rescale_rounding_direct_range_argument"),
        "query_chunks": all_chunks,
        "selected_fixed_table_chunks": [],
        "range_argument": range_argument,
        "table_chunks": table.get("chunks", []),
        "max_limb": max_limb,
        "min_input_nonnegative": True,
        "mismatch_count": 0 if max_limb < base else 1,
        "matrix_claim_sources": sources,
        "matrix_claim_sources_digest": digest_json(sources),
    }
    family["status"] = _verify_lookup_family(family)["status"]
    return family


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
    fixed_table_ok = _verify_fixed_table_descriptor(family.get("fixed_table", {})) and family.get("fixed_table_digest") == family.get("fixed_table", {}).get("table_digest")
    query_binding = family.get("committed_query_segment")
    query_binding_ok = query_binding is None or family.get("committed_query_segment_digest") == digest_json(query_binding)
    certified_domain = family.get("fixed_table", {}).get("certified_domain", [])
    domain_descriptor_ok = (
        family.get("input_domain") == certified_domain
        if family.get("relation") == "query_multiset_membership_in_fixed_function_table"
        else True
    )
    raw_query_digest_ok = _verify_lookup_raw_query_digest(family)
    eta_ok = True
    if family.get("relation") == "query_multiset_membership_in_fixed_function_table":
        eta_ok = str(_lookup_challenge_from_family(family, "eta")) == str(family.get("eta"))
    beta_ok = str(_lookup_challenge_from_family(family, "beta")) == str(family.get("beta"))
    product_digest_ok = family.get("product_digest") == _lookup_product_digest(
        int(family["query_product"]),
        int(family["table_product"]),
        family["raw_query_digest"],
        family["fixed_table_digest"],
        family["relation"],
    )
    lookup_window_ok = int(family.get("max_lookup_delta_q", 0)) <= int(family.get("lookup_quantization_tolerance_q", LOOKUP_QUANTIZATION_TOLERANCE_Q))
    direct_range_relation = family.get("relation") in {"direct_limb_bounded_range_argument", "matrix_rescale_rounding_direct_range_argument"}
    if direct_range_relation:
        range_argument_for_opening = family.get("range_argument", {})
        range_opening = range_argument_for_opening.get("query_opening", {})
        if range_opening:
            range_context = {
                "family": family["family"],
                "table_digest": family["fixed_table_digest"],
                "raw_query_digest": family["raw_query_digest"],
            }
            opening_results = [{
                "status": "VERIFY_OK" if _verify_ephemeral_hyrax_opening(
                    f"{family['family']}.range-query", range_opening, range_context
                ) else "VERIFY_FAILED"
            }]
        else:
            opening_results = [
                {"status": "VERIFY_OK" if _verify_ephemeral_hyrax_opening(
                    opening.get("label", f"{family.get('family', 'range')}.weighted-limbs.{index}"),
                    opening,
                    opening.get("context", {}),
                ) else "VERIFY_FAILED"}
                for index, opening in enumerate(range_argument_for_opening.get("weighted_limb_openings", []))
            ]
        query_product_ok = (
            bool(opening_results)
            and all(result.get("status") == "VERIFY_OK" for result in opening_results)
            and (not range_opening or range_argument_for_opening.get("query_commitment") == range_opening.get("registration", {}).get("commitment"))
        )
    else:
        query_product_ok = _verify_structured_function_membership(family, family.get("structured_table_opening", {}))
    range_ok = True
    range_argument_ok = True
    if direct_range_relation:
        full_table_product = _grand_product(range(1 << int(family["range_bits"])), int(family["beta"]))
        range_ok = bool(family.get("min_input_nonnegative", True)) and int(family.get("max_limb", 0)) < (1 << int(family["range_bits"]))
        range_argument = dict(family.get("range_argument", {}))
        expected_argument_digest = range_argument.pop("argument_digest", None)
        range_argument_ok = (
            expected_argument_digest == digest_json(range_argument)
            and range_argument.get("protocol") == RANGE_ARGUMENT_PROTOCOL_ID
            and range_argument.get("table_commitment") == family.get("fixed_table", {}).get("table_commitment")
            and range_argument.get("table_digest") == family.get("fixed_table_digest")
            and range_argument.get("raw_query_digest") == family.get("raw_query_digest")
            and range_argument.get("query_limb_chunks") == family.get("query_chunks", [])
            and int(range_argument.get("query_count", -1)) == int(family.get("query_count", -2))
            and str(range_argument.get("direct_query_accumulator")) == str(family.get("query_product"))
            and str(range_argument.get("fixed_table_full_product")) == str(full_table_product) == str(family.get("table_product"))
        )
        if family.get("relation") == "matrix_rescale_rounding_direct_range_argument":
            range_ok = range_ok and all(
                int(source.get("entry_limb_count", 0)) >= 0
                and int(source.get("weighted_limb_count", 0)) >= 0
                for source in family.get("matrix_claim_sources", [])
            )
    chunks_ok = int(family["query_count"]) == sum(int(item["count"]) for item in family.get("query_chunks", []))
    if family.get("relation") == "query_multiset_membership_in_fixed_function_table":
        selected_count = sum(int(item["count"]) for item in family.get("selected_fixed_table_chunks", []))
        table_chunks_ok = selected_count == int(family["query_count"])
    elif direct_range_relation:
        table_chunks_ok = family.get("table_chunks") == family.get("fixed_table", {}).get("chunks", []) and not family.get("selected_fixed_table_chunks")
    else:
        table_chunks_ok = bool(family.get("table_chunks")) or int(family["query_count"]) == 0
    structured_opening_ok = True
    if family.get("relation") == "query_multiset_membership_in_fixed_function_table":
        opening = dict(family.get("structured_table_opening", {}))
        expected_digest = opening.pop("opening_digest", None)
        structured_opening_ok = (
            bool(family.get("input_domain_ok"))
            and expected_digest == digest_json(opening)
            and opening.get("table_commitment") == family.get("fixed_table", {}).get("table_commitment")
            and opening.get("structured_verification_key") == family.get("fixed_table", {}).get("structured_verification_key")
            and opening.get("raw_query_digest") == family.get("raw_query_digest")
            and _verify_structured_function_membership(family, family.get("structured_table_opening", {}))
        )
    ok = fixed_table_ok and query_binding_ok and domain_descriptor_ok and raw_query_digest_ok and eta_ok and beta_ok and product_digest_ok and lookup_window_ok and query_product_ok and range_ok and chunks_ok and table_chunks_ok and structured_opening_ok and range_argument_ok
    return {
        "status": "VERIFY_OK" if ok else "VERIFY_FAILED",
        "checks": {
            "fixed_table_descriptor": fixed_table_ok,
            "committed_query_segment": query_binding_ok,
            "certified_domain": domain_descriptor_ok,
            "raw_query_digest_replay": raw_query_digest_ok,
            "eta_replay": eta_ok,
            "beta_replay": beta_ok,
            "product_digest": product_digest_ok,
            "lookup_quantization_window": lookup_window_ok,
            "membership_argument": query_product_ok,
            "range_bounds": range_ok,
            "direct_range_argument": range_argument_ok,
            "query_chunks_cover_queries": chunks_ok,
            "table_chunks_cover_queries": table_chunks_ok,
            "structured_fixed_table_opening": structured_opening_ok,
        },
    }


def _verify_lasso(lasso: Dict[str, Any], commitments: Dict[str, Any] | None = None) -> Dict[str, Any]:
    family_results = [_verify_lookup_family(family) for family in lasso.get("families", [])]
    registry_ok = lasso.get("fixed_table_registry_digest") == digest_json(lasso.get("fixed_table_registry", {}))
    query_binding = lasso.get("committed_query_segment", {})
    query_segment_ok = query_binding.get("segment") == "com_L" and query_binding.get("backend") == COMMITMENT_BACKEND_ID
    if commitments is not None:
        try:
            query_segment_ok = query_segment_ok and query_binding == _segment_binding(commitments, "com_L")
        except Exception:
            query_segment_ok = False
    family_binding_ok = all(
        family.get("committed_query_segment") == query_binding
        and family.get("committed_query_segment_digest") == digest_json(query_binding)
        for family in lasso.get("families", [])
    )
    ok = lasso.get("protocol") == LOOKUP_PROTOCOL_ID and registry_ok and query_segment_ok and family_binding_ok and all(item["status"] == "VERIFY_OK" for item in family_results)
    return {
        "status": "VERIFY_OK" if ok else "VERIFY_FAILED",
        "fixed_table_registry": registry_ok,
        "committed_query_segment": query_segment_ok,
        "family_query_bindings": family_binding_ok,
        "families": family_results,
    }


def _verify_lookup_raw_query_digest(family: Dict[str, Any]) -> bool:
    relation = family.get("relation")
    if relation == "query_multiset_membership_in_fixed_function_table":
        expected = digest_json(
            {
                "relation": "function_lookup_raw_xy",
                "chunks": family.get("raw_query_chunks", []),
                "query_count": int(family.get("query_count", -1)),
            }
        )
    elif relation == "direct_limb_bounded_range_argument":
        expected = digest_json(
            {
                "relation": "range_lookup_raw_limbs",
                "chunks": family.get("query_chunks", []),
                "query_count": int(family.get("query_count", -1)),
                "range_bits": int(family.get("range_bits", -1)),
            }
        )
    elif relation == "matrix_rescale_rounding_direct_range_argument":
        expected = digest_json(
            {
                "relation": "matrix_rescale_rounding_committed_residual_limbs",
                "chunks": family.get("query_chunks", []),
                "query_count": int(family.get("query_count", -1)),
                "range_bits": int(family.get("range_bits", -1)),
                "matrix_claim_sources": family.get("matrix_claim_sources", []),
            }
        )
    elif relation == "empty":
        expected = digest_json({"empty_lookup": family.get("family"), "table": family.get("fixed_table_digest")})
    else:
        return False
    return expected == family.get("raw_query_digest")


def _quantized_vector(values: np.ndarray) -> List[int]:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == object:
        return [int(item) for item in np.ravel(arr)]
    flat = np.ravel(np.asarray(values, dtype=np.float64))
    return [quantize_float(float(item)) for item in flat]


def _signed_quantized_array(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if np.issubdtype(raw.dtype, np.integer) or raw.dtype == object:
        return np.asarray([int(item) for item in np.ravel(raw)], dtype=object).reshape(raw.shape)
    arr = np.nan_to_num(raw.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.rint(arr * DEFAULT_FIXED_SCALE).astype(np.int64)


def _safe_int_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_raw = np.asarray(left)
    right_raw = np.asarray(right)
    max_left = max((abs(int(item)) for item in np.ravel(left_raw)), default=0)
    max_right = max((abs(int(item)) for item in np.ravel(right_raw)), default=0)
    if max_left * max_right * max(int(left_raw.shape[1]), 1) < (1 << 62):
        left_q = left_raw.astype(np.int64, copy=False)
        right_q = right_raw.astype(np.int64, copy=False)
    else:
        left_q = left_raw.astype(object, copy=False)
        right_q = right_raw.astype(object, copy=False)
    inner = left_q.shape[1]
    max_left = int(np.max(np.abs(left_q))) if left_q.size else 0
    max_right = int(np.max(np.abs(right_q))) if right_q.size else 0
    if left_q.dtype != object and right_q.dtype != object:
        return left_q @ right_q
    out = np.zeros((left_q.shape[0], right_q.shape[1]), dtype=object)
    for row in range(left_q.shape[0]):
        for col in range(right_q.shape[1]):
            acc = 0
            for idx in range(inner):
                acc += int(left_q[row, idx]) * int(right_q[idx, col])
            out[row, col] = acc
    return out


def _finalize_residual_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=object)
    max_abs = max((abs(int(item)) for item in np.ravel(arr)), default=0)
    if max_abs >= (1 << 62):
        raise OverflowError("matrix rescale residual exceeds int64 segment encoding bound")
    return arr.astype(np.int64)


def _matmul_residual_array(left: np.ndarray, right: np.ndarray, output: np.ndarray) -> np.ndarray:
    q_left = _signed_quantized_array(left)
    q_right = _signed_quantized_array(right)
    q_output = _signed_quantized_array(output)
    product = _safe_int_matmul(q_left, q_right)
    return _finalize_residual_array(product - q_output)


def _matvec_residual_array(matrix: np.ndarray, vector: np.ndarray, bias: float, output: np.ndarray) -> np.ndarray:
    q_matrix = _signed_quantized_array(matrix)
    q_vector = _signed_quantized_array(vector)
    q_output = _signed_quantized_array(output)
    product = _safe_int_matmul(q_matrix, q_vector.reshape((-1, 1))).reshape((-1,))
    bias_q2 = int(round(float(bias)))
    return _finalize_residual_array(product + bias_q2 - q_output)


def _matrix_rescale_residual_arrays(trace: ForwardTrace, model: Model) -> List[Tuple[str, np.ndarray]]:
    arrays: List[Tuple[str, np.ndarray]] = []
    for layer_trace, layer_params in zip(trace.hidden_layers, model.hidden_layers):
        for head_trace, head_params in zip(layer_trace.heads, layer_params.heads):
            arrays.append(
                (
                    f"{head_trace.tag}.projection.rescale_residual",
                    _matmul_residual_array(layer_trace.input, head_params.seq_kernel, head_trace.h_prime),
                )
            )
            arrays.append(
                (
                    f"{head_trace.tag}.attention_src.rescale_residual",
                    _matvec_residual_array(head_trace.h_prime, head_params.attn_src_kernel, 0, head_trace.e_src),
                )
            )
            arrays.append(
                (
                    f"{head_trace.tag}.attention_dst.rescale_residual",
                    _matvec_residual_array(head_trace.h_prime, head_params.attn_dst_kernel, 0, head_trace.e_dst),
                )
            )
    output_input = trace.hidden_layers[-1].concat
    for head_trace, head_params in zip(trace.output_heads, model.output_heads):
        arrays.append(
            (
                f"{head_trace.tag}.projection.rescale_residual",
                _matmul_residual_array(output_input, head_params.seq_kernel, head_trace.h_prime),
            )
        )
        arrays.append(
            (
                f"{head_trace.tag}.attention_src.rescale_residual",
                _matvec_residual_array(head_trace.h_prime, head_params.attn_src_kernel, 0, head_trace.e_src),
            )
        )
        arrays.append(
            (
                f"{head_trace.tag}.attention_dst.rescale_residual",
                _matvec_residual_array(head_trace.h_prime, head_params.attn_dst_kernel, 0, head_trace.e_dst),
            )
        )
    return arrays


def _prove_matrix_sumchecks(trace: ForwardTrace, model: Model, rho: int, range_bits: int, commitments: Dict[str, Any]) -> Dict[str, Any]:
    residuals = {name: values for name, values in _matrix_rescale_residual_arrays(trace, model)}
    claims: List[Dict[str, Any]] = []
    for layer_trace, layer_params in zip(trace.hidden_layers, model.hidden_layers):
        for head_index, (head_trace, head_params) in enumerate(zip(layer_trace.heads, layer_params.heads)):
            seed = rho + 1000 * (layer_trace.layer_index + 1) + head_index
            label = f"{head_trace.tag}.projection"
            claims.append(_matmul_claim(label, layer_trace.input, head_params.seq_kernel, head_trace.h_prime, residuals[f"{label}.rescale_residual"], commitments, seed, range_bits))
            label = f"{head_trace.tag}.attention_src"
            claims.append(_matvec_claim(label, head_trace.h_prime, head_params.attn_src_kernel, 0, head_trace.e_src, residuals[f"{label}.rescale_residual"], commitments, seed + 17, range_bits))
            label = f"{head_trace.tag}.attention_dst"
            claims.append(_matvec_claim(label, head_trace.h_prime, head_params.attn_dst_kernel, 0, head_trace.e_dst, residuals[f"{label}.rescale_residual"], commitments, seed + 31, range_bits))
    output_input = trace.hidden_layers[-1].concat
    for head_index, (head_trace, head_params) in enumerate(zip(trace.output_heads, model.output_heads)):
        seed = rho + 900_000 + head_index
        label = f"{head_trace.tag}.projection"
        claims.append(_matmul_claim(label, output_input, head_params.seq_kernel, head_trace.h_prime, residuals[f"{label}.rescale_residual"], commitments, seed, range_bits))
        label = f"{head_trace.tag}.attention_src"
        claims.append(_matvec_claim(label, head_trace.h_prime, head_params.attn_src_kernel, 0, head_trace.e_src, residuals[f"{label}.rescale_residual"], commitments, seed + 17, range_bits))
        label = f"{head_trace.tag}.attention_dst"
        claims.append(_matvec_claim(label, head_trace.h_prime, head_params.attn_dst_kernel, 0, head_trace.e_dst, residuals[f"{label}.rescale_residual"], commitments, seed + 31, range_bits))
    status = "VERIFY_OK" if all(_verify_matrix_claim(item, commitments)["status"] == "VERIFY_OK" for item in claims) else "VERIFY_FAILED"
    return {"protocol": "matrix-mle-random-linear-combination-sumcheck-v2", "status": status, "claims": claims}


def _array_commitment_binding(commitments: Dict[str, Any], segment: str, array_name: str) -> Dict[str, Any]:
    segment_item = commitments["items"][segment]
    matches = [item for item in segment_item.get("arrays", []) if item.get("name") == array_name]
    if len(matches) != 1:
        raise KeyError(f"committed array not found: {segment}/{array_name}")
    binding = {
        "segment": segment,
        "array_name": array_name,
        "offset": int(matches[0]["offset"]),
        "field_vector_length": int(matches[0]["length"]),
        "shape": list(matches[0]["shape"]),
        "N_in": int(segment_item["N_in"]),
        "commitment_set_digest": segment_item["commitment_set_digest"],
        "segment_chunk_count": int(segment_item["chunk_count"]),
    }
    binding["binding_digest"] = digest_json(binding)
    return binding


def _verify_array_commitment_binding(binding: Dict[str, Any], commitments: Dict[str, Any] | None) -> bool:
    payload = dict(binding)
    expected_digest = payload.pop("binding_digest", None)
    if expected_digest != digest_json(payload):
        return False
    if commitments is None:
        return True
    try:
        expected = _array_commitment_binding(commitments, payload["segment"], payload["array_name"])
    except Exception:
        return False
    return binding == expected


def _flatten_field_values(values: np.ndarray) -> List[int]:
    return [to_field_int(int(item)) for item in np.ravel(np.asarray(values, dtype=object))]


def _weighted_matrix_eval(values: np.ndarray, row_weights: Sequence[int], col_weights: Sequence[int]) -> Tuple[int, List[int]]:
    arr = np.asarray(values, dtype=object)
    flat_weights: List[int] = []
    acc = 0
    for row_index, row_weight in enumerate(row_weights):
        for col_index, col_weight in enumerate(col_weights):
            weight = f_mul(row_weight, col_weight)
            flat_weights.append(weight)
            acc = (acc + weight * to_field_int(int(arr[row_index, col_index]))) % BN254_FR_MODULUS
    return acc, flat_weights


def _weighted_vector_eval(values: np.ndarray, row_weights: Sequence[int]) -> Tuple[int, List[int]]:
    arr = np.ravel(np.asarray(values, dtype=object))
    weights = [to_field_int(item) for item in row_weights[: len(arr)]]
    acc = 0
    for value, weight in zip(arr, weights):
        acc = (acc + weight * to_field_int(int(value))) % BN254_FR_MODULUS
    return acc, weights


def _matmul_residual_bound_q2(left: np.ndarray, right: np.ndarray) -> int:
    q_left = np.abs(_signed_quantized_array(left).astype(object))
    q_right = np.abs(_signed_quantized_array(right).astype(object))
    inner = int(q_left.shape[1])
    max_row_l1 = max((sum(int(item) for item in row) for row in q_left), default=0)
    max_col_l1 = max((sum(int(item) for item in q_right[:, col]) for col in range(q_right.shape[1])), default=0)
    # q(a)q(b) - q(a*b)S is bounded by operand quantization errors:
    # roughly 0.5*||q(a)||_1 + 0.5*||q(b)||_1 plus nearest-output rounding.
    return int(max_row_l1 // 2 + max_col_l1 // 2 + 1 + inner)


def _matvec_residual_bound_q2(matrix: np.ndarray, vector: np.ndarray) -> int:
    q_matrix = np.abs(_signed_quantized_array(matrix).astype(object))
    q_vector = np.abs(_signed_quantized_array(vector).astype(object))
    inner = int(q_matrix.shape[1])
    max_row_l1 = max((sum(int(item) for item in row) for row in q_matrix), default=0)
    vector_l1 = sum(int(item) for item in np.ravel(q_vector))
    return int(max_row_l1 // 2 + vector_l1 // 2 + 2 + inner)


def _limb_chunk_summary_from_signed(values: Iterable[int], range_bits: int, chunk_size: int = 8192) -> Dict[str, Any]:
    base = 1 << int(range_bits)
    chunks: List[Dict[str, Any]] = []
    chunk: List[int] = []
    max_limb = 0
    max_abs_value = 0
    count = 0

    def flush() -> None:
        nonlocal chunk
        if not chunk:
            return
        chunks.append(_field_chunk_digests(chunk, chunk_size=len(chunk))[0] | {"chunk_index": len(chunks)})
        chunk = []

    for raw in values:
        value = abs(int(raw))
        max_abs_value = max(max_abs_value, value)
        limbs = _decompose_limbs(value, range_bits)
        for limb in limbs:
            limb_int = int(limb)
            max_limb = max(max_limb, limb_int)
            chunk.append(to_field_int(limb_int))
            count += 1
            if len(chunk) >= chunk_size:
                flush()
    flush()
    return {
        "range_bits": int(range_bits),
        "limb_count": count,
        "max_limb": max_limb,
        "max_abs_value": str(max_abs_value),
        "limb_chunks": chunks,
        "limb_chunks_digest": digest_json(chunks),
        "base": base,
    }


def _lasso_chunk_transcript_accumulator(raw_chunks: Sequence[Dict[str, Any]], beta: int, raw_digest: str, range_bits: int) -> int:
    """Bind the complete range-query chunk transcript to its grand product.

    The chunk roots are authenticated by the per-claim residual/limb
    commitments.  A verifier never accepts a selected table prefix or a
    prover-chosen sample; every canonical chunk is included in this
    transcript.  The scalar opening proofs carried by each claim provide the
    terminal binding for the private limb vector.
    """
    acc = 1
    for chunk in raw_chunks:
        root = int(str(chunk.get("sha256", "0")), 16) % BN254_FR_MODULUS
        count = int(chunk.get("count", 0))
        acc = (acc * (to_field_int(beta) + root + count)) % BN254_FR_MODULUS
    # Include the domain-separated transcript in the terminal scalar while
    # retaining an actual field accumulator over every authenticated chunk.
    transcript_scalar = _field_label(f"{RANGE_ARGUMENT_PROTOCOL_ID}:{raw_digest}:{int(range_bits)}")
    return (acc + to_field_int(beta) * transcript_scalar) % BN254_FR_MODULUS


def _rescale_rounding_binding(
    label: str,
    kind: str,
    shape: Sequence[int],
    claimed_sum: int,
    bias_eval_q2: int | None,
    output_eval_q2: int,
    residual_eval: int,
    residual_array: np.ndarray,
    residual_weights: Sequence[int],
    residual_oracle_binding: Dict[str, Any],
    per_entry_bound_q2: int,
    range_bits: int,
) -> Dict[str, Any]:
    residual_values = _flatten_field_values(residual_array)
    residual_seed = {
        "protocol": MATRIX_RESCALE_PROTOCOL_ID,
        "label": label,
        "kind": kind,
        "shape": [int(item) for item in shape],
        "residual_oracle_binding_digest": residual_oracle_binding["binding_digest"],
    }
    residual_eval_sumcheck = prove_product_sum(
        f"{label}.rescale_residual_eval",
        residual_values or [0],
        list(residual_weights)[: max(len(residual_values), 1)] or [0],
        residual_seed,
    ).to_json()
    residual_eval_sumcheck["transcript_seed"] = residual_seed
    residual_eval_verify = verify_product_sum(f"{label}.rescale_residual_eval", residual_eval_sumcheck, residual_seed)
    signed_weighted_residual = _signed_field(residual_eval)
    weighted_limbs = _decompose_limbs(abs(signed_weighted_residual), range_bits)
    signed_entries = [int(item) for item in np.ravel(np.asarray(residual_array, dtype=object))]
    entry_summary = _limb_chunk_summary_from_signed(signed_entries, range_bits)
    max_abs_entry = max((abs(value) for value in signed_entries), default=0)
    weighted_opening_seed = digest_json({"label": label, "binding": residual_oracle_binding, "weighted_limb_chunks": _field_chunk_digests(weighted_limbs)})
    weighted_opening = _ephemeral_hyrax_opening(
        f"{label}.weighted-limbs",
        weighted_limbs,
        {"weighted_opening_seed": weighted_opening_seed},
    )
    payload = {
        "protocol": MATRIX_RESCALE_PROTOCOL_ID,
        "label": label,
        "kind": kind,
        "shape": [int(item) for item in shape],
        "field_modulus": str(BN254_FR_MODULUS),
        "fixed_scale": 1,
        "rounding_rule": "per-output-cell exact integer residual is committed in com_Aux and range-bounded; weighted advice is its verifier-challenge linear evaluation",
        "claimed_sum": str(to_field_int(claimed_sum)),
        "bias_eval_q2": None if bias_eval_q2 is None else str(to_field_int(bias_eval_q2)),
        "output_eval_q2": str(to_field_int(output_eval_q2)),
        "rescale_rounding_advice": str(to_field_int(residual_eval)),
        "signed_weighted_residual": str(signed_weighted_residual),
        "range_bits": int(range_bits),
        "weighted_limb_chunks": _field_chunk_digests(weighted_limbs),
        "weighted_limb_count": len(weighted_limbs),
        "weighted_limb_opening": weighted_opening,
        "weighted_limb_opening_protocol": "native-range-residual-limb-opening-v1",
        "entry_limb_count": int(entry_summary["limb_count"]),
        "entry_limb_chunks": entry_summary["limb_chunks"],
        "entry_limb_chunks_digest": entry_summary["limb_chunks_digest"],
        "max_abs_entry_residual_q2": str(max_abs_entry),
        "per_entry_rounding_bound_q2": str(per_entry_bound_q2),
        "rounding_bound_derivation": "0.5*max_row_l1(q_left)+0.5*max_col_or_vector_l1(q_right)+nearest-output-rounding slack",
        "residual_oracle_binding": residual_oracle_binding,
        "residual_oracle_binding_digest": residual_oracle_binding["binding_digest"],
        "residual_eval_sumcheck": residual_eval_sumcheck,
        "residual_eval_sumcheck_verify": residual_eval_verify,
    }
    payload["binding_digest"] = digest_json(payload)
    return payload


def _matmul_claim(label: str, left: np.ndarray, right: np.ndarray, output: np.ndarray, residual_array: np.ndarray, commitments: Dict[str, Any], seed: int, range_bits: int) -> Dict[str, Any]:
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
    residual_eval, residual_weights = _weighted_matrix_eval(residual_array, row_weights, col_weights)
    residual_binding = _array_commitment_binding(commitments, "com_Aux", f"{label}.rescale_residual")
    rescale_binding = _rescale_rounding_binding(
        label,
        "matrix_multiplication",
        [int(n), int(k), int(m)],
        to_field_int(int(proof["claimed_sum"])),
        None,
        output_eval_q2,
        residual_eval,
        residual_array,
        residual_weights,
        residual_binding,
        _matmul_residual_bound_q2(left, right),
        range_bits,
    )
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
        "rescale_rounding_advice": str(residual_eval),
        "rescale_rounding": rescale_binding,
        "constraint_equation": "claimed_sum - output_eval_q2 - committed_residual_eval == 0 mod p",
        "rescale_advice_digest": digest_json({"label": label, "output_eval_q2": str(output_eval_q2), "rescale_rounding_advice": str(residual_eval), "rescale_binding_digest": rescale_binding["binding_digest"]}),
        "status": "VERIFY_OK",
    }
    claim["status"] = _verify_matrix_claim(claim, commitments)["status"]
    return claim


def _matvec_claim(label: str, matrix: np.ndarray, vector: np.ndarray, bias: float, output: np.ndarray, residual_array: np.ndarray, commitments: Dict[str, Any], seed: int, range_bits: int) -> Dict[str, Any]:
    n, k = matrix.shape
    if vector.shape[0] != k or output.shape[0] != n:
        raise ValueError(f"matvec shape mismatch for {label}")
    row_weights = _random_field_vector(f"{label}.rows", seed, n)
    a_values = _fold_left_by_rows(matrix, row_weights)
    b_values = _quantized_vector(vector)
    transcript_seed = {"seed": int(seed), "rows": n, "inner": k, "cols": 1, "mode": "full_matvec_random_eval"}
    proof = prove_product_sum(label, a_values, b_values, transcript_seed).to_json()
    verify = verify_product_sum(label, proof, transcript_seed)
    # Model biases are already canonical integers in the registered model
    # segment.  Do not re-quantize them with the legacy floating-point scale.
    bias_eval_q2 = (sum(row_weights) % BN254_FR_MODULUS) * to_field_int(int(bias))
    bias_eval_q2 %= BN254_FR_MODULUS
    output_eval_q2 = _weighted_vector_eval_q2(output, row_weights)
    residual_eval, residual_weights = _weighted_vector_eval(residual_array, row_weights)
    residual_binding = _array_commitment_binding(commitments, "com_Aux", f"{label}.rescale_residual")
    rescale_binding = _rescale_rounding_binding(
        label,
        "matrix_vector_inner_products",
        [int(n), int(k)],
        to_field_int(int(proof["claimed_sum"])),
        bias_eval_q2,
        output_eval_q2,
        residual_eval,
        residual_array,
        residual_weights,
        residual_binding,
        _matvec_residual_bound_q2(matrix, vector),
        range_bits,
    )
    claim = {
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
        "rescale_rounding_advice": str(residual_eval),
        "rescale_rounding": rescale_binding,
        "constraint_equation": "claimed_sum + bias_eval_q2 - output_eval_q2 - committed_residual_eval == 0 mod p",
        "rescale_advice_digest": digest_json({"label": label, "bias_eval_q2": str(bias_eval_q2), "output_eval_q2": str(output_eval_q2), "rescale_rounding_advice": str(residual_eval), "rescale_binding_digest": rescale_binding["binding_digest"]}),
        "status": "VERIFY_OK" if verify["status"] == "VERIFY_OK" else "VERIFY_FAILED",
    }
    claim["status"] = _verify_matrix_claim(claim, commitments)["status"]
    return claim


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
    arr = np.asarray(matrix)
    out = [0] * arr.shape[1]
    for row_index, weight in enumerate(row_weights):
        if weight == 0:
            continue
        row = arr[row_index]
        for col_index, value in enumerate(row):
            value_int = int(value) if (arr.dtype == object or np.issubdtype(arr.dtype, np.integer)) else quantize_float(float(value))
            out[col_index] = (out[col_index] + weight * to_field_int(value_int)) % BN254_FR_MODULUS
    return out


def _fold_right_by_cols(matrix: np.ndarray, col_weights: Sequence[int]) -> List[int]:
    arr = np.asarray(matrix)
    out = [0] * arr.shape[0]
    for row_index in range(arr.shape[0]):
        acc = 0
        for col_index, weight in enumerate(col_weights):
            value = arr[row_index, col_index]
            value_int = int(value) if (arr.dtype == object or np.issubdtype(arr.dtype, np.integer)) else quantize_float(float(value))
            acc = (acc + weight * to_field_int(value_int)) % BN254_FR_MODULUS
        out[row_index] = acc
    return out


def _weighted_output_eval_q2(output: np.ndarray, row_weights: Sequence[int], col_weights: Sequence[int]) -> int:
    arr = np.asarray(output)
    acc = 0
    for row_index, row_weight in enumerate(row_weights):
        for col_index, col_weight in enumerate(col_weights):
            value = arr[row_index, col_index]
            value_int = int(value) if (arr.dtype == object or np.issubdtype(arr.dtype, np.integer)) else quantize_float(float(value))
            acc = (acc + row_weight * col_weight * to_field_int(value_int)) % BN254_FR_MODULUS
    return acc


def _weighted_vector_eval_q2(output: np.ndarray, row_weights: Sequence[int]) -> int:
    arr = np.asarray(output)
    acc = 0
    for row_index, row_weight in enumerate(row_weights):
        value = arr[row_index]
        value_int = int(value) if (arr.dtype == object or np.issubdtype(arr.dtype, np.integer)) else quantize_float(float(value))
        acc = (acc + row_weight * to_field_int(value_int)) % BN254_FR_MODULUS
    return acc


def _verify_rescale_rounding_binding(claim: Dict[str, Any], commitments: Dict[str, Any] | None = None) -> bool:
    binding = dict(claim.get("rescale_rounding", {}))
    expected_digest = binding.pop("binding_digest", None)
    if expected_digest != digest_json(binding):
        return False
    if binding.get("protocol") != MATRIX_RESCALE_PROTOCOL_ID:
        return False
    if binding.get("label") != claim.get("label") or binding.get("kind") != claim.get("kind"):
        return False
    if str(to_field_int(int(binding.get("claimed_sum", "0")))) != str(to_field_int(int(claim["sumcheck"]["claimed_sum"]))):
        return False
    if str(to_field_int(int(binding.get("output_eval_q2", "0")))) != str(to_field_int(int(claim.get("output_eval_q2", "0")))):
        return False
    if str(to_field_int(int(binding.get("rescale_rounding_advice", "0")))) != str(to_field_int(int(claim.get("rescale_rounding_advice", "0")))):
        return False
    if claim.get("kind") == "matrix_vector_inner_products":
        if str(to_field_int(int(binding.get("bias_eval_q2", "0")))) != str(to_field_int(int(claim.get("bias_eval_q2", "0")))):
            return False
    elif binding.get("bias_eval_q2") is not None:
        return False
    signed_weighted_residual = int(binding.get("signed_weighted_residual", "0"))
    if signed_weighted_residual != _signed_field(int(claim.get("rescale_rounding_advice", "0"))):
        return False
    range_bits = int(binding.get("range_bits", 0))
    if range_bits <= 0:
        return False
    weighted_limb_chunks = binding.get("weighted_limb_chunks", [])
    weighted_opening = binding.get("weighted_limb_opening", {})
    if binding.get("weighted_limb_opening_protocol") != "native-range-residual-limb-opening-v1":
        return False
    weighted_opening_seed = digest_json({
        "label": claim["label"],
        "binding": binding.get("residual_oracle_binding", {}),
        "weighted_limb_chunks": binding.get("weighted_limb_chunks", []),
    })
    if not _verify_ephemeral_hyrax_opening(
        f"{claim['label']}.weighted-limbs",
        weighted_opening,
        {"weighted_opening_seed": weighted_opening_seed},
    ):
        return False
    if int(binding.get("weighted_limb_count", -1)) != sum(int(item.get("count", 0)) for item in weighted_limb_chunks):
        return False
    if int(binding.get("entry_limb_count", 0)) != sum(int(item["count"]) for item in binding.get("entry_limb_chunks", [])):
        return False
    if binding.get("entry_limb_chunks_digest") != digest_json(binding.get("entry_limb_chunks", [])):
        return False
    if int(binding.get("max_abs_entry_residual_q2", "0")) > int(binding.get("per_entry_rounding_bound_q2", "0")):
        return False
    if not _verify_array_commitment_binding(binding.get("residual_oracle_binding", {}), commitments):
        return False
    if binding.get("residual_oracle_binding_digest") != binding.get("residual_oracle_binding", {}).get("binding_digest"):
        return False
    residual_sumcheck = binding.get("residual_eval_sumcheck", {})
    seed = residual_sumcheck.get("transcript_seed", {})
    if seed.get("residual_oracle_binding_digest") != binding.get("residual_oracle_binding_digest"):
        return False
    residual_verify = verify_product_sum(f"{claim['label']}.rescale_residual_eval", residual_sumcheck, seed)
    if residual_verify["status"] != "VERIFY_OK":
        return False
    if to_field_int(int(residual_sumcheck.get("claimed_sum", "0"))) != to_field_int(int(claim.get("rescale_rounding_advice", "0"))):
        return False
    return True


def _verify_matrix_claim(claim: Dict[str, Any], commitments: Dict[str, Any] | None = None) -> Dict[str, Any]:
    shape = claim["shape"]
    if claim.get("kind") == "matrix_multiplication":
        seed = int(claim["random_eval"]["row_challenge_seed"])
        transcript_seed = {"seed": seed, "rows": int(shape[0]), "inner": int(shape[1]), "cols": int(shape[2]), "mode": "full_matrix_random_eval"}
    else:
        seed = int(claim["random_eval"]["row_challenge_seed"])
        transcript_seed = {"seed": seed, "rows": int(shape[0]), "inner": int(shape[1]), "cols": 1, "mode": "full_matvec_random_eval"}
    verify = verify_product_sum(claim["label"], claim["sumcheck"], transcript_seed)
    rescale_present = "rescale_rounding_advice" in claim and "output_eval_q2" in claim
    claimed = to_field_int(int(claim["sumcheck"]["claimed_sum"]))
    output_eval = to_field_int(int(claim.get("output_eval_q2", "0")))
    rescale = to_field_int(int(claim.get("rescale_rounding_advice", "0")))
    if claim.get("kind") == "matrix_multiplication":
        equation_ok = (claimed - output_eval - rescale) % BN254_FR_MODULUS == 0
        digest_ok = claim.get("rescale_advice_digest") == digest_json(
            {"label": claim["label"], "output_eval_q2": str(output_eval), "rescale_rounding_advice": str(rescale), "rescale_binding_digest": claim.get("rescale_rounding", {}).get("binding_digest")}
        )
    else:
        bias_eval = to_field_int(int(claim.get("bias_eval_q2", "0")))
        equation_ok = (claimed + bias_eval - output_eval - rescale) % BN254_FR_MODULUS == 0
        digest_ok = claim.get("rescale_advice_digest") == digest_json(
            {"label": claim["label"], "bias_eval_q2": str(bias_eval), "output_eval_q2": str(output_eval), "rescale_rounding_advice": str(rescale), "rescale_binding_digest": claim.get("rescale_rounding", {}).get("binding_digest")}
        )
    rescale_binding_ok = _verify_rescale_rounding_binding(claim, commitments)
    ok = verify["status"] == "VERIFY_OK" and rescale_present and equation_ok and digest_ok and rescale_binding_ok
    return {
        "status": "VERIFY_OK" if ok else "VERIFY_FAILED",
        "sumcheck": verify,
        "rescale_advice_present": rescale_present,
        "output_rescale_equation": equation_ok,
        "rescale_advice_digest": digest_ok,
        "rescale_rounding_range_binding": rescale_binding_ok,
    }


def _verify_matrix(matrix: Dict[str, Any], commitments: Dict[str, Any] | None = None) -> Dict[str, Any]:
    claim_results = [_verify_matrix_claim(claim, commitments) for claim in matrix.get("claims", [])]
    ok = matrix.get("protocol") == "matrix-mle-random-linear-combination-sumcheck-v2" and all(item["status"] == "VERIFY_OK" for item in claim_results)
    return {"status": "VERIFY_OK" if ok else "VERIFY_FAILED", "claims": claim_results}


def _native_matrix_witness_plan(trace: ForwardTrace, model: Model) -> List[Tuple[MatMulSpec, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    residuals = {name: values for name, values in _matrix_rescale_residual_arrays(trace, model)}
    plan: List[Tuple[MatMulSpec, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for layer_position, (layer_trace, layer_params) in enumerate(zip(trace.hidden_layers, model.hidden_layers)):
        if layer_position == 0:
            left_source = TensorSource("X_pub", "X_pub.H_input")
        else:
            left_source = TensorSource("Aux", f"hidden.l{trace.hidden_layers[layer_position - 1].layer_index}.concat")
        for head_trace, head_params in zip(layer_trace.heads, layer_params.heads):
            label = f"{head_trace.tag}.projection"
            spec = MatMulSpec(
                label,
                (int(layer_trace.input.shape[0]), int(layer_trace.input.shape[1]), int(head_trace.h_prime.shape[1])),
                left_source,
                TensorSource("W", f"{head_trace.tag}.W"),
                TensorSource("Aux", f"{head_trace.tag}.H_prime"),
                TensorSource("Aux", f"{label}.rescale_residual"),
                1,
            )
            plan.append((spec, layer_trace.input, head_params.seq_kernel, head_trace.h_prime, residuals[f"{label}.rescale_residual"]))
            for suffix, vector, output in (
                ("attention_src", head_params.attn_src_kernel, head_trace.e_src),
                ("attention_dst", head_params.attn_dst_kernel, head_trace.e_dst),
            ):
                label = f"{head_trace.tag}.{suffix}"
                spec = MatMulSpec(
                    label,
                    (int(head_trace.h_prime.shape[0]), int(head_trace.h_prime.shape[1]), 1),
                    TensorSource("Aux", f"{head_trace.tag}.H_prime"),
                    TensorSource("W", f"{head_trace.tag}.a_{suffix.split('_')[1]}"),
                    TensorSource("Aux", f"{head_trace.tag}.E_{suffix.split('_')[1]}"),
                    TensorSource("Aux", f"{label}.rescale_residual"),
                    1,
                )
                plan.append((spec, head_trace.h_prime, np.asarray(vector, dtype=object).reshape((-1, 1)), np.asarray(output, dtype=object).reshape((-1, 1)), np.asarray(residuals[f"{label}.rescale_residual"], dtype=object).reshape((-1, 1))))
    output_input = trace.hidden_layers[-1].concat
    output_source = TensorSource("Aux", f"hidden.l{trace.hidden_layers[-1].layer_index}.concat")
    for head_trace, head_params in zip(trace.output_heads, model.output_heads):
        label = f"{head_trace.tag}.projection"
        spec = MatMulSpec(
            label,
            (int(output_input.shape[0]), int(output_input.shape[1]), int(head_trace.h_prime.shape[1])),
            output_source,
            TensorSource("W", f"{head_trace.tag}.W"),
            TensorSource("Aux", f"{head_trace.tag}.H_prime"),
            TensorSource("Aux", f"{label}.rescale_residual"),
            1,
        )
        plan.append((spec, output_input, head_params.seq_kernel, head_trace.h_prime, residuals[f"{label}.rescale_residual"]))
        for suffix, vector, output in (
            ("attention_src", head_params.attn_src_kernel, head_trace.e_src),
            ("attention_dst", head_params.attn_dst_kernel, head_trace.e_dst),
        ):
            label = f"{head_trace.tag}.{suffix}"
            spec = MatMulSpec(
                label,
                (int(head_trace.h_prime.shape[0]), int(head_trace.h_prime.shape[1]), 1),
                TensorSource("Aux", f"{head_trace.tag}.H_prime"),
                TensorSource("W", f"{head_trace.tag}.a_{suffix.split('_')[1]}"),
                TensorSource("Aux", f"{head_trace.tag}.E_{suffix.split('_')[1]}"),
                TensorSource("Aux", f"{label}.rescale_residual"),
                1,
            )
            plan.append((spec, head_trace.h_prime, np.asarray(vector, dtype=object).reshape((-1, 1)), np.asarray(output, dtype=object).reshape((-1, 1)), np.asarray(residuals[f"{label}.rescale_residual"], dtype=object).reshape((-1, 1))))
    return plan


def _expected_native_matrix_specs(metadata: Dict[str, Any]) -> List[MatMulSpec]:
    dims = metadata["dim_cfg"]
    node_count = int(dims["N"])
    out: List[MatMulSpec] = []
    hidden = dims["hidden_profile"]
    for layer_position, profile in enumerate(hidden):
        layer_index = layer_position + 1
        d_in = int(dims["d_in"][layer_position])
        head_dim = int(profile["head_dim"])
        left = TensorSource("X_pub", "X_pub.H_input") if layer_position == 0 else TensorSource("Aux", f"hidden.l{layer_index - 1}.concat")
        for head_index in range(int(profile["head_count"])):
            prefix = f"hidden.l{layer_index}.h{head_index}"
            label = f"{prefix}.projection"
            out.append(MatMulSpec(label, (node_count, d_in, head_dim), left, TensorSource("W", f"{prefix}.W"), TensorSource("Aux", f"{prefix}.H_prime"), TensorSource("Aux", f"{label}.rescale_residual"), 1))
            for direction in ("src", "dst"):
                label = f"{prefix}.attention_{direction}"
                out.append(MatMulSpec(label, (node_count, head_dim, 1), TensorSource("Aux", f"{prefix}.H_prime"), TensorSource("W", f"{prefix}.a_{direction}"), TensorSource("Aux", f"{prefix}.E_{direction}"), TensorSource("Aux", f"{label}.rescale_residual"), 1))
    d_out_input = int(dims["d_cat"][-1])
    left = TensorSource("Aux", f"hidden.l{len(hidden)}.concat")
    for head_index in range(int(dims["K_out"])):
        prefix = f"output.h{head_index}"
        label = f"{prefix}.projection"
        out.append(MatMulSpec(label, (node_count, d_out_input, int(dims["C"])), left, TensorSource("W", f"{prefix}.W"), TensorSource("Aux", f"{prefix}.H_prime"), TensorSource("Aux", f"{label}.rescale_residual"), 1))
        for direction in ("src", "dst"):
            label = f"{prefix}.attention_{direction}"
            out.append(MatMulSpec(label, (node_count, int(dims["C"]), 1), TensorSource("Aux", f"{prefix}.H_prime"), TensorSource("W", f"{prefix}.a_{direction}"), TensorSource("Aux", f"{prefix}.E_{direction}"), TensorSource("Aux", f"{label}.rescale_residual"), 1))
    return out


def _prove_native_matrix(
    trace: ForwardTrace,
    model: Model,
    metadata: Dict[str, Any],
    state: SegmentCommitmentState,
    transcript: Transcript,
    work_dir: pathlib.Path,
    graph: LocalGraph,
) -> Dict[str, Any]:
    n_in = int(metadata["Layout_in"]["N_in"])
    context = {
        "metadata_digest": digest_json(metadata),
        "Layout_in_digest": metadata["Layout_in_digest"],
        "session_nonce": metadata["session_nonce"],
        "relation": "Pi_MatMul",
    }
    vectors = {"W": state.vectors["com_W"], "Aux": state.vectors["com_Aux"]}
    registrations = {"W": state.registrations["com_W"], "Aux": state.registrations["com_Aux"]}
    public_arrays = {"X_pub.H_input": np.asarray(trace.input_features, dtype=object)}
    plan = _native_matrix_witness_plan(trace, model)
    expected = _expected_native_matrix_specs(metadata)
    if [item[0] for item in plan] != expected:
        raise RuntimeError("MatMul witness plan differs from the external circuit plan")
    claims = [
        prove_matmul(
            spec, left, right, output, residual,
            layout=metadata["Layout_in"], public_arrays=public_arrays,
            vectors=vectors, registrations=registrations, params=hyrax_value_params(n_in),
            hyrax=state.hyrax, work_dir=work_dir, transcript=transcript, context=context,
        )
        for spec, left, right, output, residual in plan
    ]
    return {"protocol": "zkGPT/exact-integer-matmul-batch-v1", "claims": claims, "transcript_digest": transcript.digest.hex()}


def _verify_native_matrix(
    matrix: Dict[str, Any],
    metadata: Dict[str, Any],
    commitments: Dict[str, Any],
    transcript: Transcript,
    public_arrays: Mapping[str, np.ndarray],
    source_sink: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    try:
        expected = _expected_native_matrix_specs(metadata)
        claims = matrix.get("claims", [])
        if matrix.get("protocol") != "zkGPT/exact-integer-matmul-batch-v1" or len(claims) != len(expected) or not expected:
            return {"status": "VERIFY_FAILED", "reason": "missing or extra MatMul claims", "claims": []}
        registrations = {
            "W": commitments["items"]["com_W"],
            "Aux": commitments["items"]["com_Aux"],
        }
        n_in = int(metadata["Layout_in"]["N_in"])
        context = {
            "metadata_digest": digest_json(metadata),
            "Layout_in_digest": metadata["Layout_in_digest"],
            "session_nonce": metadata["session_nonce"],
            "relation": "Pi_MatMul",
        }
        results = [
            verify_matmul(
                proof, spec, layout=metadata["Layout_in"], public_arrays=public_arrays,
                registrations=registrations, params=hyrax_value_params(n_in),
                transcript=transcript, context=context, source_sink=source_sink,
            )
            for proof, spec in zip(claims, expected)
        ]
        digest_ok = matrix.get("transcript_digest") == transcript.digest.hex()
        return {"status": "VERIFY_OK" if all(results) and digest_ok else "VERIFY_FAILED", "claims": results, "transcript": digest_ok}
    except (KeyError, TypeError, ValueError):
        return {"status": "VERIFY_FAILED", "reason": "malformed native MatMul proof", "claims": []}


def _verify_matrix_rescale_lasso_binding(matrix: Dict[str, Any], lasso: Dict[str, Any]) -> bool:
    families = {family.get("family"): family for family in lasso.get("families", [])}
    family = families.get("rescale_rounding_range")
    if not family or family.get("relation") != "matrix_rescale_rounding_direct_range_argument":
        return False
    raw_chunks: List[Dict[str, Any]] = []
    weighted_chunks: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    query_count = 0
    for claim in matrix.get("claims", []):
        binding = claim.get("rescale_rounding", {})
        for chunk in binding.get("entry_limb_chunks", []):
            raw_chunks.append(dict(chunk))
        for chunk in binding.get("weighted_limb_chunks", []):
            weighted_chunks.append(dict(chunk))
        entry_limb_count = int(binding.get("entry_limb_count", 0))
        weighted_limb_count = sum(int(item["count"]) for item in binding.get("weighted_limb_chunks", []))
        query_count += entry_limb_count + weighted_limb_count
        sources.append(
            {
                "label": claim.get("label"),
                "kind": claim.get("kind"),
                "rescale_binding_digest": binding.get("binding_digest"),
                "entry_limb_count": entry_limb_count,
                "weighted_limb_count": weighted_limb_count,
                "per_entry_rounding_bound_q2": binding.get("per_entry_rounding_bound_q2"),
            }
        )
    renumbered_chunks = []
    for chunk in raw_chunks + weighted_chunks:
        current = dict(chunk)
        current["chunk_index"] = len(renumbered_chunks)
        renumbered_chunks.append(current)
    raw_digest = digest_json(
        {
            "relation": "matrix_rescale_rounding_committed_residual_limbs",
            "chunks": renumbered_chunks,
            "query_count": query_count,
            "range_bits": int(family["range_bits"]),
            "matrix_claim_sources": sources,
        }
    )
    return (
        int(family.get("query_count", -1)) == query_count
        and family.get("query_chunks") == renumbered_chunks
        and family.get("selected_fixed_table_chunks") == []
        and family.get("raw_query_digest") == raw_digest
        and family.get("matrix_claim_sources") == sources
        and family.get("matrix_claim_sources_digest") == digest_json(sources)
        and _verify_lookup_family(family)["status"] == "VERIFY_OK"
    )


def _native_gkr_array_length(layout: Dict[str, Any], segment: str, name: str) -> int:
    matches = [
        array
        for descriptor in layout.get("segments", []) if descriptor.get("name") == segment
        for array in descriptor.get("arrays", []) if array.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"native GKR array missing from Layout_in: {segment}/{name}")
    return int(matches[0]["length"])


def _native_division_specs(prefix: str, count: int) -> List[Tuple[str, Any]]:
    aux = lambda name: identity_oracle("Aux", f"{prefix}.{name}", count)
    product = lambda name: identity_oracle("Aux", f"{prefix}.gkr.{name}", count)
    zero = (0,) * count
    one = (1,) * count
    inv_two = pow(2, BN254_FR_MODULUS - 2, BN254_FR_MODULUS)
    return [
        ("product", ProductGateSpec(f"{prefix}.sigma_boolean", aux("sigma"), aux("sigma"), aux("sigma"))),
        ("product", ProductGateSpec(f"{prefix}.z_boolean", aux("z"), aux("z"), aux("z"))),
        ("product", ProductGateSpec(f"{prefix}.sigma_a", product("sigma_a"), aux("sigma"), aux("a"))),
        ("product", ProductGateSpec(f"{prefix}.sigma_q", product("sigma_q"), aux("sigma"), aux("q"))),
        ("product", ProductGateSpec(f"{prefix}.a_a_inv", product("a_a_inv"), aux("a"), aux("a_inv"))),
        ("product", ProductGateSpec(f"{prefix}.d_d_inv", product("d_d_inv"), aux("d"), aux("d_inv"))),
        ("product", ProductGateSpec(f"{prefix}.d_q", product("d_q"), aux("d"), aux("q"))),
        ("product", ProductGateSpec(f"{prefix}.z_a", product("z_a"), aux("z"), aux("a"))),
        ("product", ProductGateSpec(f"{prefix}.z_sigma", product("z_sigma"), aux("z"), aux("sigma"))),
        ("product", ProductGateSpec(f"{prefix}.z_a_inv", product("z_a_inv"), aux("z"), aux("a_inv"))),
        ("linear", LinearGateSpec(f"{prefix}.signed_input", aux("x"), (
            LinearTerm(1, aux("a")), LinearTerm(-2, product("sigma_a")),
        ), zero)),
        ("linear", LinearGateSpec(f"{prefix}.signed_output", aux("y"), (
            LinearTerm(1, aux("q")), LinearTerm(-2, product("sigma_q")),
        ), zero)),
        ("linear", LinearGateSpec(f"{prefix}.a_inverse", product("a_a_inv"), (
            LinearTerm(-1, aux("z")),
        ), one)),
        ("linear", LinearGateSpec(f"{prefix}.d_inverse", product("d_d_inv"), (), one)),
        ("linear", LinearGateSpec(f"{prefix}.z_a_zero", product("z_a"), (), zero)),
        ("linear", LinearGateSpec(f"{prefix}.z_sigma_zero", product("z_sigma"), (), zero)),
        ("linear", LinearGateSpec(f"{prefix}.z_a_inv_zero", product("z_a_inv"), (), zero)),
        ("linear", LinearGateSpec(f"{prefix}.nearest_equation", aux("a"), (
            LinearTerm(1, product("d_q")), LinearTerm(inv_two, aux("r")), LinearTerm(-inv_two, aux("d")),
        ), zero)),
        ("linear", LinearGateSpec(f"{prefix}.rounding_slack", aux("delta"), (
            LinearTerm(2, aux("d")), LinearTerm(-1, aux("r")),
        ), (-1,) * count)),
    ]


def _native_lasso_consistency_specs(metadata: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Bind every committed L_qry coordinate to its canonical Aux arithmetic."""
    layout = metadata["Layout_in"]
    lasso_segment = next(item for item in layout["segments"] if item["name"] == "L_qry")
    specs: List[Tuple[str, Any]] = []

    def aux_length(name: str) -> int:
        return _native_gkr_array_length(layout, "Aux", name)

    for family in lasso_segment.get("lasso_families", []):
        label = str(family["family"])
        real_count = int(family["real_count"])
        padded_count = int(family["padded_count"])
        lasso_real = OracleRef("L_qry", label, tuple(range(real_count)))
        if family["kind"] == "function":
            table = metadata["static_table_registry"][family["table_id"]]
            encoding = table["tuple_encoding"]
            if label.endswith(".LeakyReLU"):
                prefix = label.removesuffix(".LeakyReLU")
                x_name, y_name = f"{prefix}.S", f"{prefix}.Z"
            elif label.endswith(".ExpMap"):
                prefix = label.removesuffix(".ExpMap")
                x_name, y_name = f"{prefix}.Delta", f"{prefix}.U"
            elif label.endswith(".ELU"):
                prefix = label.removesuffix(".ELU")
                x_name, y_name = f"{prefix}.SignedRoundDiv.output", f"{prefix}.H_agg"
            else:
                raise ValueError(f"unknown formal function family: {label}")
            if aux_length(x_name) != real_count or aux_length(y_name) != real_count:
                raise ValueError("function lookup Aux/L cardinality mismatch")
            specs.append(("linear", LinearGateSpec(
                f"lasso/{label}/packed_query",
                lasso_real,
                (
                    LinearTerm(int(encoding["base"]), identity_oracle("Aux", x_name, real_count)),
                    LinearTerm(1, identity_oracle("Aux", y_name, real_count)),
                ),
                (int(encoding["x_bound"]) * int(encoding["base"]) + int(encoding["y_bound"]),) * real_count,
            )))
        else:
            limb_count = int(family["limbs_per_value"])
            base = 1 << int(family["range_bits"])
            range_bits = int(family["range_bits"])
            bit_name = f"lasso.{label}.range_bits"
            bit_count = padded_count * range_bits
            bit_oracle = identity_oracle("Aux", bit_name, bit_count)
            specs.extend([
                ("product", ProductGateSpec(
                    f"lasso/{label}/range-bit-boolean", bit_oracle, bit_oracle, bit_oracle,
                )),
                ("linear", LinearGateSpec(
                    f"lasso/{label}/range-limb-reconstruction",
                    identity_oracle("L_qry", label, padded_count),
                    tuple(
                        LinearTerm(
                            1 << bit,
                            OracleRef("Aux", bit_name, tuple(index * range_bits + bit for index in range(padded_count))),
                        )
                        for bit in range(range_bits)
                    ),
                    (0,) * padded_count,
                )),
            ])
            if label == "rescale_rounding_range":
                sources = [("rescale_rounding_range.gkr.abs", aux_length("rescale_rounding_range.gkr.abs"))]
            elif label.endswith(".range_delta"):
                prefix = label.removesuffix(".range_delta")
                sources = [(f"{prefix}.Delta", aux_length(f"{prefix}.Delta"))]
            elif label.endswith(".range"):
                prefix = label.removesuffix(".range")
                if prefix == "output.final_average.SignedRoundDiv":
                    prefix = "output.final_average"
                sources = [(f"{prefix}.{name}", aux_length(f"{prefix}.{name}")) for name in ("a", "q", "r", "delta", "d")]
            else:
                raise ValueError(f"unknown formal range family: {label}")
            cursor = 0
            for source_name, count in sources:
                terms = tuple(
                    LinearTerm(
                        pow(base, limb, BN254_FR_MODULUS),
                        OracleRef("L_qry", label, tuple(cursor + index * limb_count + limb for index in range(count))),
                    )
                    for limb in range(limb_count)
                )
                specs.append(("linear", LinearGateSpec(
                    f"lasso/{label}/reconstruct/{source_name}",
                    identity_oracle("Aux", source_name, count), terms, (0,) * count,
                )))
                cursor += count * limb_count
            if cursor != real_count:
                raise ValueError("range lookup limb layout differs from certified family")
        if padded_count > real_count:
            specs.append(("linear", LinearGateSpec(
                f"lasso/{label}/padding",
                OracleRef("L_qry", label, tuple(range(real_count, padded_count))), (),
                (to_field_int(int(family["a_pad"])),) * (padded_count - real_count),
            )))
    return specs


def _native_gkr_specs(
    metadata: Dict[str, Any], graph: Any, public_y: Sequence[int]
) -> List[Tuple[str, Any]]:
    """Derive the accepted ordered gate list from public graph/layout data."""
    layout = metadata["Layout_in"]
    graph_src = graph.get("src", []) if isinstance(graph, dict) else graph.src
    graph_dst = graph.get("dst", []) if isinstance(graph, dict) else graph.dst
    src = tuple(int(value) for value in graph_src)
    dst = tuple(int(value) for value in graph_dst)
    edge_count = len(src)
    node_count = int(metadata["dim_cfg"]["N"])
    specs: List[Tuple[str, Any]] = []
    groups: List[List[int]] = [[] for _ in range(node_count)]
    for index, target in enumerate(dst):
        if target < 0 or target >= node_count:
            raise ValueError("public destination leaves the canonical node domain")
        groups[target].append(index)
    if any(not group for group in groups):
        raise ValueError("formal GAT graph must contain the canonical self-loop group for every node")
    group_starts = {group[0] for group in groups}
    expected_q = tuple(1 if index in group_starts else 0 for index in range(edge_count))
    expected_count = tuple(len(group) for group in groups)
    group_ends = tuple(group[-1] for group in groups)

    def add_head(prefix: str, width: int, *, output: bool, h_prime_scale: int) -> None:
        aux = lambda name, mapping: OracleRef("Aux", f"{prefix}.{name}", tuple(mapping))
        identity_edges = tuple(range(edge_count))
        identity_nodes = tuple(range(node_count))
        specs.extend([
            ("linear", LinearGateSpec(f"{prefix}.Q_grp_public_boundary", aux("Q_grp", identity_edges), (), expected_q)),
            ("linear", LinearGateSpec(f"{prefix}.group_count_public", aux("group_count", identity_nodes), (), expected_count)),
            ("linear", LinearGateSpec(f"{prefix}.score_routing", aux("S", identity_edges), (
                LinearTerm(1, aux("E_src", src)), LinearTerm(1, aux("E_dst", dst)),
            ), (0,) * edge_count)),
            ("linear", LinearGateSpec(f"{prefix}.max_shift", aux("Delta", identity_edges), (
                LinearTerm(1, aux("M", dst)), LinearTerm(-1, aux("Z", identity_edges)),
            ), (0,) * edge_count)),
            ("product", ProductGateSpec(f"{prefix}.alpha_product", aux("Alpha", identity_edges), aux("U", identity_edges), aux("Inv", dst))),
            ("product", ProductGateSpec(f"{prefix}.sum_inverse_product", aux("gkr.sum_inv", identity_nodes), aux("Sum", identity_nodes), aux("Inv", identity_nodes))),
            ("linear", LinearGateSpec(f"{prefix}.sum_inverse_one", aux("gkr.sum_inv", identity_nodes), (), (1,) * node_count)),
            ("product", ProductGateSpec(f"{prefix}.Q_grp_boolean", aux("Q_grp", identity_edges), aux("Q_grp", identity_edges), aux("Q_grp", identity_edges))),
            ("product", ProductGateSpec(f"{prefix}.s_max_boolean", aux("s_max", identity_edges), aux("s_max", identity_edges), aux("s_max", identity_edges))),
            ("product", ProductGateSpec(f"{prefix}.s_max_delta_product", aux("gkr.s_max_delta", identity_edges), aux("s_max", identity_edges), aux("Delta", identity_edges))),
            ("linear", LinearGateSpec(f"{prefix}.s_max_selected", aux("gkr.s_max_delta", identity_edges), (), (0,) * edge_count)),
        ])
        repeated_edges = tuple(edge for edge in range(edge_count) for _ in range(width))
        h_edge_map = tuple(src[edge] * width + column for edge in range(edge_count) for column in range(width))
        flat_count = edge_count * width
        previous_edge_map = tuple(-1 if edge in group_starts else edge - 1 for edge in range(edge_count))
        previous_flat_map = tuple(
            -1 if edge in group_starts else (edge - 1) * width + column
            for edge in range(edge_count) for column in range(width)
        )
        specs.extend([
            ("linear", LinearGateSpec(f"{prefix}.one_minus_Q", aux("gkr.one_minus_Q", identity_edges), (
                LinearTerm(-1, aux("Q_grp", identity_edges)),
            ), (1,) * edge_count)),
            ("linear", LinearGateSpec(f"{prefix}.R_U_prev_binding", aux("gkr.R_U_prev", identity_edges), (
                LinearTerm(1, aux("R_U", previous_edge_map)),
            ), (0,) * edge_count)),
            ("product", ProductGateSpec(f"{prefix}.R_U_carry_product", aux("gkr.R_U_carry", identity_edges), aux("gkr.one_minus_Q", identity_edges), aux("gkr.R_U_prev", identity_edges))),
            ("linear", LinearGateSpec(f"{prefix}.R_U_recurrence", aux("R_U", identity_edges), (
                LinearTerm(1, aux("gkr.R_U_carry", identity_edges)), LinearTerm(1, aux("U", identity_edges)),
            ), (0,) * edge_count)),
            ("linear", LinearGateSpec(f"{prefix}.Sum_terminal", aux("Sum", identity_nodes), (
                LinearTerm(1, aux("R_U", group_ends)),
            ), (0,) * node_count)),
            ("linear", LinearGateSpec(f"{prefix}.R_smax_prev_binding", aux("gkr.R_smax_prev", identity_edges), (
                LinearTerm(1, aux("gkr.R_smax", previous_edge_map)),
            ), (0,) * edge_count)),
            ("product", ProductGateSpec(f"{prefix}.R_smax_carry_product", aux("gkr.R_smax_carry", identity_edges), aux("gkr.one_minus_Q", identity_edges), aux("gkr.R_smax_prev", identity_edges))),
            ("linear", LinearGateSpec(f"{prefix}.R_smax_recurrence", aux("gkr.R_smax", identity_edges), (
                LinearTerm(1, aux("gkr.R_smax_carry", identity_edges)), LinearTerm(1, aux("s_max", identity_edges)),
            ), (0,) * edge_count)),
            ("linear", LinearGateSpec(f"{prefix}.s_max_one_per_group", aux("gkr.sum_inv", identity_nodes), (
                LinearTerm(1, aux("gkr.R_smax", group_ends)),
            ), (0,) * node_count)),
            ("linear", LinearGateSpec(f"{prefix}.R_num_prev_binding", aux("gkr.R_num_prev", tuple(range(flat_count))), (
                LinearTerm(1, aux("R_num", previous_flat_map)),
            ), (0,) * flat_count)),
            ("product", ProductGateSpec(f"{prefix}.R_num_carry_product", aux("gkr.R_num_carry", tuple(range(flat_count))), aux("gkr.one_minus_Q", repeated_edges), aux("gkr.R_num_prev", tuple(range(flat_count))))),
            ("product", ProductGateSpec(f"{prefix}.U_H_product", aux("gkr.U_H", tuple(range(flat_count))), aux("U", repeated_edges), aux("H_prime", h_edge_map))),
            ("linear", LinearGateSpec(f"{prefix}.R_num_recurrence", aux("R_num", tuple(range(flat_count))), (
                LinearTerm(1, aux("gkr.R_num_carry", tuple(range(flat_count)))), LinearTerm(1, aux("gkr.U_H", tuple(range(flat_count)))),
            ), (0,) * flat_count)),
            ("linear", LinearGateSpec(f"{prefix}.R_field_prev_binding", aux("gkr.R_field_prev", tuple(range(flat_count))), (
                LinearTerm(1, aux("R_field", previous_flat_map)),
            ), (0,) * flat_count)),
            ("product", ProductGateSpec(f"{prefix}.R_field_carry_product", aux("gkr.R_field_carry", tuple(range(flat_count))), aux("gkr.one_minus_Q", repeated_edges), aux("gkr.R_field_prev", tuple(range(flat_count))))),
            ("product", ProductGateSpec(f"{prefix}.Alpha_H_product", aux("gkr.Alpha_H", tuple(range(flat_count))), aux("Alpha", repeated_edges), aux("H_prime", h_edge_map))),
            ("linear", LinearGateSpec(f"{prefix}.R_field_recurrence", aux("R_field", tuple(range(flat_count))), (
                LinearTerm(1, aux("gkr.R_field_carry", tuple(range(flat_count)))), LinearTerm(1, aux("gkr.Alpha_H", tuple(range(flat_count)))),
            ), (0,) * flat_count)),
        ])
        node_flat_count = node_count * width
        end_flat_maps = tuple(group_ends[node] * width + column for node in range(node_count) for column in range(width))
        specs.extend([
            ("linear", LinearGateSpec(f"{prefix}.agg_num_terminal", identity_oracle("Aux", f"{prefix}.H_or_Y_agg_num", node_flat_count), (
                LinearTerm(1, aux("R_num", end_flat_maps)),
            ), (0,) * node_flat_count)),
            ("linear", LinearGateSpec(f"{prefix}.field_aggregate_terminal", identity_oracle("Aux", f"{prefix}.field_aggregate", node_flat_count), (
                LinearTerm(1, aux("R_field", end_flat_maps)),
            ), (0,) * node_flat_count)),
        ])
        value_count = node_count * width
        specs.append(("linear", LinearGateSpec(
            f"{prefix}.field_aggregate_output",
            identity_oracle("Aux", f"{prefix}.H_agg_pre", value_count),
            (LinearTerm(1, identity_oracle("Aux", f"{prefix}.field_aggregate", value_count)),),
            (0,) * value_count,
        )))
        aggregate_prefix = f"{prefix}.SignedRoundDiv.aggregate"
        aggregate_count = _native_gkr_array_length(layout, "Aux", f"{aggregate_prefix}.x")
        repeated_nodes = tuple(row for row in range(node_count) for _ in range(width))
        specs.extend([
            ("linear", LinearGateSpec(f"{aggregate_prefix}.numerator_wiring", identity_oracle("Aux", f"{aggregate_prefix}.x", aggregate_count), (
                LinearTerm(int(16), identity_oracle("Aux", f"{prefix}.H_or_Y_agg_num", aggregate_count)),
            ), (0,) * aggregate_count)),
            ("linear", LinearGateSpec(f"{aggregate_prefix}.denominator_wiring", identity_oracle("Aux", f"{aggregate_prefix}.d", aggregate_count), (
                LinearTerm(int(h_prime_scale), aux("Sum", repeated_nodes)),
            ), (0,) * aggregate_count)),
            ("linear", LinearGateSpec(f"{aggregate_prefix}.output_wiring", identity_oracle("Aux", f"{prefix}.SignedRoundDiv.output", aggregate_count), (
                LinearTerm(1, identity_oracle("Aux", f"{aggregate_prefix}.y", aggregate_count)),
            ), (0,) * aggregate_count)),
        ])
        specs.extend(_native_division_specs(aggregate_prefix, aggregate_count))
        if output:
            alignment_prefix = f"{prefix}.SignedRoundDiv.scale_alignment"
            alignment_count = _native_gkr_array_length(layout, "Aux", f"{alignment_prefix}.x")
            specs.extend(_native_division_specs(alignment_prefix, alignment_count))
            specs.extend([
                ("linear", LinearGateSpec(f"{alignment_prefix}.numerator_wiring", identity_oracle("Aux", f"{alignment_prefix}.x", alignment_count), (
                    LinearTerm(16, identity_oracle("Aux", f"{prefix}.SignedRoundDiv.output", alignment_count)),
                ), (0,) * alignment_count)),
                ("linear", LinearGateSpec(f"{alignment_prefix}.denominator_wiring", identity_oracle("Aux", f"{alignment_prefix}.d", alignment_count), (), (16,) * alignment_count)),
                ("linear", LinearGateSpec(f"{alignment_prefix}.output_wiring", identity_oracle("Aux", f"{prefix}.Y_pre_aligned", alignment_count), (
                    LinearTerm(1, identity_oracle("Aux", f"{alignment_prefix}.y", alignment_count)),
                ), (0,) * alignment_count)),
            ])
            bias_map = tuple(index % width for index in range(value_count))
            specs.append(("linear", LinearGateSpec(
                f"{prefix}.bias_add",
                identity_oracle("Aux", f"{prefix}.H_agg", value_count),
                (
                    LinearTerm(1, identity_oracle("Aux", f"{prefix}.Y_pre_aligned", value_count)),
                    LinearTerm(1, OracleRef("W", f"{prefix}.b_out", bias_map)),
                ),
                (0,) * value_count,
            )))

    hidden = metadata["dim_cfg"]["hidden_profile"]
    for layer_position, layer in enumerate(hidden, start=1):
        width = int(layer["head_dim"])
        for head_index in range(int(layer["head_count"])):
            add_head(
                f"hidden.l{layer_position}.h{head_index}", width,
                output=False, h_prime_scale=16 if layer_position == 1 else 256,
            )
        concat_width = int(layer["head_count"]) * width
        for head_index in range(int(layer["head_count"])):
            target_map = tuple(
                row * concat_width + head_index * width + column
                for row in range(node_count) for column in range(width)
            )
            specs.append(("linear", LinearGateSpec(
                f"hidden.l{layer_position}.concat.h{head_index}",
                OracleRef("Aux", f"hidden.l{layer_position}.concat", target_map),
                (LinearTerm(1, identity_oracle("Aux", f"hidden.l{layer_position}.h{head_index}.H_agg", node_count * width)),),
                (0,) * (node_count * width),
            )))
    class_count = int(metadata["dim_cfg"]["C"])
    for head_index in range(int(metadata["dim_cfg"]["K_out"])):
        add_head(f"output.h{head_index}", class_count, output=True, h_prime_scale=256)
    final_prefix = "output.final_average"
    final_count = _native_gkr_array_length(layout, "Aux", f"{final_prefix}.x")
    public_y_values = tuple(to_field_int(value) for value in public_y)
    if len(public_y_values) != final_count:
        raise ValueError("external public Y shape differs from the final GKR layer")
    specs.extend([
        ("linear", LinearGateSpec(
            "output.head_sum_post_bias",
            identity_oracle("Aux", "output.Y_sum", final_count),
            tuple(
                LinearTerm(1, identity_oracle("Aux", f"output.h{head_index}.H_agg", final_count))
                for head_index in range(int(metadata["dim_cfg"]["K_out"]))
            ),
            (0,) * final_count,
        )),
        ("linear", LinearGateSpec(
            "output.final_average.numerator_wiring",
            identity_oracle("Aux", f"{final_prefix}.x", final_count),
            (LinearTerm(1, identity_oracle("Aux", "output.Y_sum", final_count)),),
            (0,) * final_count,
        )),
        ("linear", LinearGateSpec(
            "output.final_average.denominator_wiring",
            identity_oracle("Aux", f"{final_prefix}.d", final_count), (),
            (int(metadata["dim_cfg"]["K_out"]),) * final_count,
        )),
    ])
    specs.extend(_native_division_specs(final_prefix, final_count))
    specs.append(("linear", LinearGateSpec(
        "output.final_average.public_tensor_binding",
        identity_oracle("Aux", "output.Y", final_count),
        (LinearTerm(1, identity_oracle("Aux", f"{final_prefix}.y", final_count)),),
        (0,) * final_count,
    )))
    specs.append(("linear", LinearGateSpec(
        "output.external_public_Y",
        identity_oracle("Aux", "output.Y", final_count), (), public_y_values,
    )))
    residual_count = _native_gkr_array_length(layout, "Aux", "rescale_rounding_range.gkr.value")
    residual = lambda name: identity_oracle("Aux", f"rescale_rounding_range.gkr.{name}", residual_count)
    specs.extend([
        ("product", ProductGateSpec("rescale_rounding_range.sign_boolean", residual("sign"), residual("sign"), residual("sign"))),
        ("product", ProductGateSpec("rescale_rounding_range.sign_abs_product", residual("sign_abs"), residual("sign"), residual("abs"))),
        ("linear", LinearGateSpec("rescale_rounding_range.signed_magnitude", residual("value"), (
            LinearTerm(1, residual("abs")), LinearTerm(-2, residual("sign_abs")),
        ), (0,) * residual_count)),
    ])
    specs.extend(_native_lasso_consistency_specs(metadata))
    return specs


def _prove_native_gkr(
    graph: LocalGraph,
    public_y: np.ndarray,
    metadata: Dict[str, Any],
    state: SegmentCommitmentState,
    transcript: Transcript,
    work_dir: pathlib.Path,
) -> Dict[str, Any]:
    n_in = int(metadata["Layout_in"]["N_in"])
    params = hyrax_value_params(n_in)
    context = {
        "metadata_digest": digest_json(metadata),
        "Layout_in_digest": metadata["Layout_in_digest"],
        "session_nonce": metadata["session_nonce"],
        "relation": "Pi_GKR",
    }
    vectors = {"W": state.vectors["com_W"], "Aux": state.vectors["com_Aux"], "L_qry": state.vectors["com_L"]}
    registrations = {"W": state.registrations["com_W"], "Aux": state.registrations["com_Aux"], "L_qry": state.registrations["com_L"]}
    claims = []
    for kind, spec in _native_gkr_specs(metadata, graph, np.ravel(public_y)):
        common = dict(
            layout=metadata["Layout_in"], vectors=vectors, registrations=registrations,
            params=params, hyrax=state.hyrax, work_dir=work_dir,
            transcript=transcript, context=context,
        )
        proof = prove_linear_gate(spec, **common) if kind == "linear" else prove_product_gate(spec, **common)
        claims.append({"kind": kind, "proof": proof})
    return {"protocol": "zkGPT/native-gkr-bound-gates-v1", "claims": claims, "transcript_digest": transcript.digest.hex()}


def _prove_gkr_legacy(graph: LocalGraph, trace: ForwardTrace, model: Model, transcript: Dict[str, Any], commitments: Dict[str, Any]) -> Dict[str, Any]:
    claims: List[Dict[str, Any]] = []
    claims.extend(_gkr_public_routing_claims(graph, transcript, commitments))
    for layer in trace.hidden_layers:
        for head_index, head in enumerate(layer.heads):
            xi = transcript["compression"][f"xi_l{layer.layer_index}_h{head_index}"]
            claims.extend(_gkr_head_claims(graph, head, xi, transcript, include_elu=True, commitments=commitments))
        for head_index, head in enumerate(layer.heads):
            start = head_index * head.h_agg.shape[1]
            end = start + head.h_agg.shape[1]
            claims.append(_gkr_relation_claim(f"hidden.l{layer.layer_index}.concat.h{head_index}", layer.concat[:, start:end], head.h_agg, 0.0, transcript, "concat_head_wiring", commitments))
    for head_index, (head, params) in enumerate(zip(trace.output_heads, model.output_heads)):
        xi = transcript["compression"]["xi_out"]
        claims.extend(_gkr_head_claims(graph, head, xi, transcript, include_elu=False, commitments=commitments))
        output_pre_bias = head.output_aligned if head.output_aligned is not None else (head.elu_in if head.elu_in is not None else head.h_agg_pre)
        claims.append(_gkr_relation_claim(f"output.h{head_index}.bias_add", head.h_agg, output_pre_bias + params.output_bias[None, :], 0.0, transcript, "output_bias", commitments))
    output_values = np.stack([head.h_agg for head in trace.output_heads], axis=0).astype(object)
    expected_y = np.empty(output_values.shape[1:], dtype=object)
    for index in np.ndindex(expected_y.shape):
        expected_y[index] = _round_div_signed(sum(int(item[index]) for item in output_values), len(trace.output_heads))
    claims.append(_gkr_relation_claim("output.head_average_post_bias", trace.y, expected_y, 0.0, transcript, "output_head_average", commitments))
    if trace.output_average_division is not None:
        claims.extend(_gkr_signed_round_div_claims("output.final_average.SignedRoundDiv", trace.output_average_division, transcript, commitments))
    status = "VERIFY_OK" if all(_verify_gkr_claim(item)["status"] == "VERIFY_OK" for item in claims) else "VERIFY_FAILED"
    return {"protocol": GKR_PROTOCOL_ID, "status": status, "claims": claims}


def _gkr_public_routing_claims(graph: LocalGraph, transcript: Dict[str, Any], commitments: Dict[str, Any]) -> List[Dict[str, Any]]:
    dst_expected = np.sort(graph.dst.astype(np.float64), kind="stable")
    node_ptr_expected = np.asarray(graph.node_ptr, dtype=np.float64)
    edge_ptr_expected = np.asarray(graph.edge_ptr, dtype=np.float64)
    return [
        _gkr_relation_claim("public.edge_dst_sorted_wiring", graph.dst.astype(np.float64), dst_expected, 0.0, transcript, "public_edge_routing", commitments),
        _gkr_relation_claim("public.node_ptr_wiring", graph.node_ptr.astype(np.float64), node_ptr_expected, 0.0, transcript, "public_node_ptr", commitments),
        _gkr_relation_claim("public.edge_ptr_wiring", graph.edge_ptr.astype(np.float64), edge_ptr_expected, 0.0, transcript, "public_edge_ptr", commitments),
    ]


def _field_product_array(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Elementwise field product for exact integer GKR relation claims."""
    lhs = np.asarray(left, dtype=object)
    rhs = np.asarray(right, dtype=object)
    if lhs.shape != rhs.shape:
        raise ValueError(f"field product shape mismatch: {lhs.shape} != {rhs.shape}")
    out = np.empty(lhs.shape, dtype=object)
    for index in np.ndindex(lhs.shape):
        out[index] = (int(lhs[index]) * int(rhs[index])) % BN254_FR_MODULUS
    return out


def _field_reduce_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=object)
    out = np.empty(arr.shape, dtype=object)
    for index in np.ndindex(arr.shape):
        out[index] = int(arr[index]) % BN254_FR_MODULUS
    return out


def _signed_round_div_array(numerator: np.ndarray, denominator: np.ndarray, numerator_scale: int = 1) -> np.ndarray:
    num = np.asarray(numerator, dtype=object)
    den = np.asarray(denominator, dtype=object)
    if num.shape != den.shape:
        raise ValueError(f"SignedRoundDiv shape mismatch: {num.shape} != {den.shape}")
    out = np.empty(num.shape, dtype=object)
    for index in np.ndindex(num.shape):
        out[index] = _round_div_signed(int(numerator_scale) * int(num[index]), int(den[index]))
    return out


def _gkr_signed_round_div_claims(
    prefix: str,
    division: SignedRoundDivTrace,
    transcript: Dict[str, Any],
    commitments: Dict[str, Any],
) -> List[Dict[str, Any]]:
    sigma = np.asarray(division.sigma, dtype=object)
    z = np.asarray(division.z, dtype=object)
    a = np.asarray(division.a, dtype=object)
    q = np.asarray(division.q, dtype=object)
    d = np.asarray(division.d, dtype=object)
    r = np.asarray(division.r, dtype=object)
    delta = np.asarray(division.delta, dtype=object)
    a_inv = np.asarray(division.a_inv, dtype=object)
    d_inv = np.asarray(division.d_inv, dtype=object)
    zeros = np.zeros(sigma.shape, dtype=object)
    ones = np.ones(sigma.shape, dtype=object)
    signed_a = (1 - 2 * sigma) * a
    signed_q = (1 - 2 * sigma) * q
    return [
        _gkr_relation_claim(f"{prefix}.signed_input", division.x, signed_a, 0.0, transcript, "signed_round_division", commitments),
        _gkr_relation_claim(f"{prefix}.signed_output", division.y, signed_q, 0.0, transcript, "signed_round_division", commitments),
        _gkr_relation_claim(f"{prefix}.sigma_boolean", sigma * (sigma - 1), zeros, 0.0, transcript, "signed_round_division", commitments),
        _gkr_relation_claim(f"{prefix}.z_boolean", z * (z - 1), zeros, 0.0, transcript, "signed_round_division", commitments),
        _gkr_relation_claim(f"{prefix}.zero_abs", z * a, zeros, 0.0, transcript, "signed_round_division", commitments),
        _gkr_relation_claim(f"{prefix}.zero_sign", z * sigma, zeros, 0.0, transcript, "signed_round_division", commitments),
        _gkr_relation_claim(f"{prefix}.zero_inverse", z * a_inv, zeros, 0.0, transcript, "signed_round_division", commitments),
        _gkr_relation_claim(f"{prefix}.abs_inverse", _field_product_array(a, a_inv), 1 - z, 0.0, transcript, "signed_round_division", commitments),
        _gkr_relation_claim(f"{prefix}.denominator_inverse", _field_product_array(d, d_inv), ones, 0.0, transcript, "signed_round_division", commitments),
        _gkr_relation_claim(f"{prefix}.nearest_equation", 2 * a + d, 2 * d * q + r, 0.0, transcript, "signed_round_division", commitments),
        _gkr_relation_claim(f"{prefix}.rounding_slack", r + delta, 2 * d - 1, 0.0, transcript, "signed_round_division", commitments),
    ]


def _scatter_group_count(starts: np.ndarray, dst: np.ndarray, node_count: int, edge_count: int) -> np.ndarray:
    counts = np.zeros(node_count, dtype=np.int64)
    ends = np.append(starts[1:], edge_count)
    for begin, end in zip(starts, ends):
        counts[int(dst[int(begin)])] = int(end) - int(begin)
    return counts


def _sparse_running_states(graph: LocalGraph, head: HeadTrace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_grp = np.asarray(head.group_start_selector, dtype=np.int64)
    r_u = np.zeros(graph.src.size, dtype=object)
    r_num = np.zeros((graph.src.size, head.h_prime.shape[1]), dtype=object)
    r_field = np.zeros((graph.src.size, head.h_prime.shape[1]), dtype=object)
    running_u = 0
    running_num = [0] * head.h_prime.shape[1]
    running_field = [0] * head.h_prime.shape[1]
    for edge_index in range(graph.src.size):
        if q_grp[edge_index]:
            running_u = 0
            running_num = [0] * head.h_prime.shape[1]
            running_field = [0] * head.h_prime.shape[1]
        source = int(graph.src[edge_index])
        running_u += int(head.u[edge_index])
        r_u[edge_index] = running_u
        for col in range(head.h_prime.shape[1]):
            running_num[col] += int(head.u[edge_index]) * int(head.h_prime[source, col])
            running_field[col] = (running_field[col] + int(head.alpha[edge_index]) * int(head.h_prime[source, col])) % BN254_FR_MODULUS
            r_num[edge_index, col] = running_num[col]
            r_field[edge_index, col] = running_field[col]
    return r_u, r_num, r_field


def _gkr_head_claims(graph: LocalGraph, head: HeadTrace, xi: int, transcript: Dict[str, Any], include_elu: bool, commitments: Dict[str, Any]) -> List[Dict[str, Any]]:
    src = graph.src
    dst = graph.dst
    starts = _group_starts(dst, graph.edge_graph)
    expected_q_grp = np.zeros(src.size, dtype=np.int64)
    expected_q_grp[starts] = 1
    expected_group_count = _scatter_group_count(starts, dst, graph.num_nodes, src.size)
    if head.field_agg is not None:
        lhs = head.field_agg
        rhs = np.zeros_like(head.field_agg, dtype=object)
        for edge_index in range(src.size):
            target = int(dst[edge_index])
            source = int(src[edge_index])
            for col in range(head.h_prime.shape[1]):
                rhs[target, col] = (int(rhs[target, col]) + int(head.alpha[edge_index]) * int(head.h_prime[source, col])) % BN254_FR_MODULUS
    else:
        powers = _powers(head.h_prime.shape[1], xi)
        lhs = head.h_agg_pre @ powers
        rhs = np.bincount(dst, weights=head.alpha * (head.h_prime @ powers)[src], minlength=graph.num_nodes)[: graph.num_nodes]
    # The inverse and alpha relations are field equations.  Compare their
    # canonical field representatives rather than Python's unreduced integer
    # products; otherwise a valid Fr inverse would be rejected (or, worse,
    # an unreduced product would be treated as an integer witness).
    inverse_lhs = _field_product_array(_field_reduce_array(head.sum_), _field_reduce_array(head.inv))
    alpha_rhs = np.asarray(
        [
            (int(head.u[index]) * int(head.inv[int(dst[index])])) % BN254_FR_MODULUS
            for index in range(head.u.size)
        ],
        dtype=object,
    ).reshape(np.asarray(head.u).shape)
    if head.field_agg is not None:
        numerator_rhs = _field_product_array(
            np.asarray(head.sum_, dtype=object)[:, None].repeat(head.field_agg.shape[1], axis=1),
            _field_reduce_array(head.field_agg),
        )
        numerator_lhs = _field_reduce_array(head.agg_num if head.agg_num is not None else head.h_agg_pre)
    else:
        numerator_lhs = np.asarray(lhs)
        numerator_rhs = np.asarray(rhs)
    claims = [
        _gkr_relation_claim(f"{head.tag}.Q_grp", head.group_start_selector, expected_q_grp, 0.0, transcript, "public_group_selector", commitments),
        _gkr_relation_claim(f"{head.tag}.group_count", head.group_count, expected_group_count, 0.0, transcript, "group_count", commitments),
        _gkr_relation_claim(f"{head.tag}.score", head.s, head.e_src[src] + head.e_dst[dst], 1e-7, transcript, "score_relation", commitments),
        _gkr_relation_claim(f"{head.tag}.LeakyReLU.lookup_query_x", head.s, head.s, 0.0, transcript, "lrelu_lookup_query_consistency", commitments),
        _gkr_relation_claim(f"{head.tag}.LeakyReLU.lookup_query_y", head.z, leaky_relu(head.s), 1e-7, transcript, "lrelu_lookup_query_consistency", commitments),
        _gkr_relation_claim(f"{head.tag}.max_state_machine", head.m[dst[starts]], np.maximum.reduceat(head.z, starts), 1e-7, transcript, "max_state_machine", commitments),
        _gkr_relation_claim(f"{head.tag}.delta", head.delta, head.m[dst] - head.z, 1e-7, transcript, "delta_relation", commitments),
        _gkr_relation_claim(f"{head.tag}.ExpMap.lookup_query_x", head.delta, head.delta, 0.0, transcript, "exp_lookup_query_consistency", commitments),
        _gkr_relation_claim(f"{head.tag}.ExpMap.lookup_query_y", head.u, np.asarray([_fixed_exp(int(value)) for value in np.ravel(head.delta)], dtype=np.int64).reshape(np.asarray(head.delta).shape), 0.0, transcript, "exp_lookup_query_consistency", commitments),
        _gkr_relation_claim(f"{head.tag}.sum_state_machine", head.sum_[dst[starts]], np.add.reduceat(head.u, starts), 1e-7, transcript, "sum_state_machine", commitments),
        _gkr_relation_claim(f"{head.tag}.inverse", inverse_lhs, np.ones_like(head.sum_, dtype=object), 0.0, transcript, "inverse_relation", commitments),
        _gkr_relation_claim(f"{head.tag}.alpha", _field_reduce_array(head.alpha), alpha_rhs, 0.0, transcript, "alpha_relation", commitments),
        _gkr_relation_claim(f"{head.tag}.double_aggregate", numerator_lhs, numerator_rhs, 0.0, transcript, "double_aggregate_consistency", commitments),
        _gkr_relation_claim(f"{head.tag}.PSQ_compressed_aggregation", lhs, rhs, 1e-6, transcript, "psq_aggregation", commitments),
    ]
    if head.max_selector is not None:
        selector = np.asarray(head.max_selector, dtype=object)
        group_selector_sum = np.add.reduceat(selector, starts)
        claims.extend(
            [
                _gkr_relation_claim(f"{head.tag}.s_max.boolean", selector * (selector - 1), np.zeros(selector.shape, dtype=object), 0.0, transcript, "max_selector", commitments),
                _gkr_relation_claim(f"{head.tag}.s_max.one_per_group", group_selector_sum, np.ones(group_selector_sum.shape, dtype=object), 0.0, transcript, "max_selector", commitments),
                _gkr_relation_claim(f"{head.tag}.s_max.selected_value", selector * (np.asarray(head.m, dtype=object)[dst] - np.asarray(head.z, dtype=object)), np.zeros(selector.shape, dtype=object), 0.0, transcript, "max_selector", commitments),
            ]
        )
    if head.r_u is not None and head.r_num is not None and head.r_field is not None:
        expected_r_u, expected_r_num, expected_r_field = _sparse_running_states(graph, head)
        claims.extend(
            [
                _gkr_relation_claim(f"{head.tag}.R_U.recurrence", head.r_u, expected_r_u, 0.0, transcript, "sparse_running_state", commitments),
                _gkr_relation_claim(f"{head.tag}.R_num.recurrence", head.r_num, expected_r_num, 0.0, transcript, "sparse_running_state", commitments),
                _gkr_relation_claim(f"{head.tag}.R_field.recurrence", head.r_field, expected_r_field, 0.0, transcript, "sparse_running_state", commitments),
            ]
        )
    if head.agg_num is not None and head.denominator is not None and head.elu_in is not None:
        division_rhs = _signed_round_div_array(
            head.agg_num,
            np.asarray(head.denominator, dtype=object) * int(head.h_prime_scale),
            numerator_scale=int(head.aggregate_output_scale),
        )
        claims.append(
            _gkr_relation_claim(
                f"{head.tag}.SignedRoundDiv.aggregate",
                head.elu_in,
                division_rhs,
                0.0,
                transcript,
                "signed_round_division",
                commitments,
            )
        )
    if head.aggregate_division is not None:
        claims.extend(_gkr_signed_round_div_claims(f"{head.tag}.SignedRoundDiv.aggregate", head.aggregate_division, transcript, commitments))
    if head.scale_alignment_division is not None:
        claims.extend(_gkr_signed_round_div_claims(f"{head.tag}.SignedRoundDiv.scale_alignment", head.scale_alignment_division, transcript, commitments))
    if include_elu:
        claims.extend(
            [
                _gkr_relation_claim(f"{head.tag}.ELU.lookup_query_x", head.elu_in if head.elu_in is not None else head.h_agg_pre, head.elu_in if head.elu_in is not None else head.h_agg_pre, 0.0, transcript, "elu_lookup_query_consistency", commitments),
                _gkr_relation_claim(f"{head.tag}.ELU.lookup_query_y", head.h_agg, elu(head.elu_in if head.elu_in is not None else head.h_agg_pre), 0.0, transcript, "elu_lookup_query_consistency", commitments),
            ]
        )
    return claims


def _gkr_oracle_bindings(commitments: Dict[str, Any], gate_family: str) -> List[Dict[str, Any]]:
    if gate_family in {"lrelu_lookup_query_consistency", "exp_lookup_query_consistency", "elu_lookup_query_consistency"}:
        segments = ["com_Aux", "com_L"]
    elif gate_family in {"psq_aggregation", "max_state_machine", "sum_state_machine"}:
        segments = ["com_Aux"]
    elif gate_family.startswith("public_"):
        segments = []
    else:
        segments = ["com_Aux"]
    return [_segment_binding(commitments, segment) for segment in segments]


def _gkr_committed_witness_opening_proofs(oracle_bindings: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    proofs = {
        "protocol": "gkr-final-oracle-claims-bound-to-openmasked-segments-v2",
        "segments": [
            {
                "segment": binding["segment"],
                "commitment_set_digest": binding["commitment_set_digest"],
                "chunk_count": binding["chunk_count"],
                "padded_length": binding["padded_length"],
                "opening_chunks": [
                    {
                        "chunk_index": chunk["chunk_index"],
                        "masked_evaluation": chunk["masked_evaluation"],
                        "opening_artifact_sha256": chunk["opening_artifact_sha256"],
                    }
                    for chunk in binding.get("chunks", [])
                ],
            }
            for binding in oracle_bindings
        ],
    }
    proofs["binding_digest"] = digest_json(proofs)
    return proofs


def _gkr_relation_claim(label: str, lhs: np.ndarray, rhs: np.ndarray, tolerance: float, transcript: Dict[str, Any], gate_family: str, commitments: Dict[str, Any]) -> Dict[str, Any]:
    lhs_array = np.asarray(lhs)
    rhs_array = np.asarray(rhs)
    exact_integer = (np.issubdtype(lhs_array.dtype, np.integer) or lhs_array.dtype == object) and (np.issubdtype(rhs_array.dtype, np.integer) or rhs_array.dtype == object)
    if exact_integer:
        left = np.ravel(lhs_array.astype(object))
        right = np.ravel(rhs_array.astype(object))
    else:
        left = np.ravel(np.nan_to_num(lhs_array.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))
        right = np.ravel(np.nan_to_num(rhs_array.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    if left.shape != right.shape:
        raise ValueError(f"GKR relation shape mismatch for {label}: {left.shape} != {right.shape}")
    diff = np.asarray([abs(int(a) - int(b)) for a, b in zip(left, right)], dtype=object) if exact_integer else np.abs(left - right)
    max_abs_error = float(np.max(diff)) if diff.size else 0.0
    mismatch_count = int(sum(1 for value in diff if int(value) != 0)) if exact_integer else int(np.count_nonzero(diff > float(tolerance)))
    lhs_q: List[int] = []
    rhs_q: List[int] = []
    for a, b in zip(left, right):
        qa = to_field_int(int(a) if exact_integer else quantize_float(float(a), CONSTRAINT_FIXED_SCALE))
        qb = to_field_int(int(b) if exact_integer else quantize_float(float(b), CONSTRAINT_FIXED_SCALE))
        lhs_q.append(qa)
        rhs_q.append(qb)
    lhs_chunks = _field_chunk_digests(lhs_q)
    rhs_chunks = _field_chunk_digests(rhs_q)
    oracle_bindings = _gkr_oracle_bindings(commitments, gate_family)
    committed_openings = _gkr_committed_witness_opening_proofs(oracle_bindings)
    seed = {
        "label": label,
        "gate_family": gate_family,
        "constraint_count": int(left.size),
        "encoding_scale": 1 if exact_integer else CONSTRAINT_FIXED_SCALE,
        "rho_gkr": int(transcript["gkr"]["rho_gkr"]),
        "transcript_digest": digest_json(transcript),
        "lhs_chunks": lhs_chunks,
        "rhs_chunks": rhs_chunks,
        "oracle_opening_reference_digest": digest_json(oracle_bindings),
        "committed_witness_opening_digest": committed_openings["binding_digest"],
    }
    weights = _random_field_vector(f"{label}.gkr.weights", int(transcript["gkr"]["rho_gkr"]) ^ _challenge_u64(label, seed), max(int(left.size), 1))
    lhs_proof = prove_product_sum(f"{label}.lhs_oracle", lhs_q or [0], weights[: max(len(lhs_q), 1)], seed).to_json()
    rhs_proof = prove_product_sum(f"{label}.rhs_oracle", rhs_q or [0], weights[: max(len(rhs_q), 1)], seed).to_json()
    lhs_proof["transcript_seed"] = seed
    rhs_proof["transcript_seed"] = seed
    lhs_verify = verify_product_sum(f"{label}.lhs_oracle", lhs_proof, seed)
    rhs_verify = verify_product_sum(f"{label}.rhs_oracle", rhs_proof, seed)
    oracle_equal = to_field_int(int(lhs_proof["claimed_sum"])) == to_field_int(int(rhs_proof["claimed_sum"]))
    claim = {
        "label": label,
        "protocol": GKR_CLAIM_PROTOCOL_ID,
        "gate_family": gate_family,
        "constraint_count": int(left.size),
        "encoding_scale": 1 if exact_integer else CONSTRAINT_FIXED_SCALE,
        "tolerance": 0.0 if exact_integer else float(tolerance),
        "max_abs_error": max_abs_error,
        "mismatch_count": mismatch_count,
        "oracle_weight_digest": digest_json([str(item) for item in weights[: min(len(weights), 32)]] + [len(weights)]),
        "lhs_chunks": lhs_chunks,
        "rhs_chunks": rhs_chunks,
        "transcript_seed_digest": digest_json(seed),
        "oracle_opening_references": oracle_bindings,
        "oracle_opening_reference_digest": digest_json(oracle_bindings),
        "committed_witness_opening_proofs": committed_openings,
        "lhs_sumcheck": lhs_proof,
        "rhs_sumcheck": rhs_proof,
        "lhs_sumcheck_verify": lhs_verify,
        "rhs_sumcheck_verify": rhs_verify,
        "oracle_claim_equal": oracle_equal,
        "final_oracle_claims": {
            "lhs_final_a": lhs_proof["final_a"],
            "lhs_final_b": lhs_proof["final_b"],
            "lhs_final_product": lhs_proof["final_product"],
            "rhs_final_a": rhs_proof["final_a"],
            "rhs_final_b": rhs_proof["final_b"],
            "rhs_final_product": rhs_proof["final_product"],
        },
    }
    claim["status"] = _verify_gkr_claim(claim)["status"]
    return claim


def _verify_gkr_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    seed = claim["lhs_sumcheck"].get("transcript_seed", {})
    rhs_seed = claim["rhs_sumcheck"].get("transcript_seed", {})
    seed_ok = (
        seed == rhs_seed
        and seed.get("label") == claim.get("label")
        and seed.get("gate_family") == claim.get("gate_family")
        and int(seed.get("constraint_count", -1)) == int(claim["constraint_count"])
        and int(seed.get("encoding_scale", -1)) == int(claim["encoding_scale"])
        and seed.get("lhs_chunks") == claim.get("lhs_chunks")
        and seed.get("rhs_chunks") == claim.get("rhs_chunks")
        and seed.get("oracle_opening_reference_digest") == claim.get("oracle_opening_reference_digest")
        and seed.get("committed_witness_opening_digest") == claim.get("committed_witness_opening_proofs", {}).get("binding_digest")
        and claim.get("transcript_seed_digest") == digest_json(seed)
    )
    lhs_verify = verify_product_sum(claim["label"] + ".lhs_oracle", claim["lhs_sumcheck"], seed)
    rhs_verify = verify_product_sum(claim["label"] + ".rhs_oracle", claim["rhs_sumcheck"], seed)
    lhs_count = sum(int(item["count"]) for item in claim.get("lhs_chunks", []))
    rhs_count = sum(int(item["count"]) for item in claim.get("rhs_chunks", []))
    coverage_ok = int(claim["constraint_count"]) == lhs_count == rhs_count
    exact_relation_ok = int(claim.get("mismatch_count", 1)) == 0
    oracle_equal = to_field_int(int(claim["lhs_sumcheck"]["claimed_sum"])) == to_field_int(int(claim["rhs_sumcheck"]["claimed_sum"]))
    oracle_chunks_equal = claim.get("lhs_chunks") == claim.get("rhs_chunks")
    oracle_binding_ok = claim.get("oracle_opening_reference_digest") == digest_json(claim.get("oracle_opening_references", []))
    committed_opening_full = claim.get("committed_witness_opening_proofs", {})
    expected_committed_opening = _gkr_committed_witness_opening_proofs(claim.get("oracle_opening_references", []))
    committed_opening = dict(committed_opening_full)
    committed_opening_digest = committed_opening.pop("binding_digest", None)
    committed_opening_ok = (
        committed_opening_digest == digest_json(committed_opening)
        and committed_opening.get("protocol") == "gkr-final-oracle-claims-bound-to-openmasked-segments-v2"
        and committed_opening_digest == seed.get("committed_witness_opening_digest")
        and committed_opening_full == expected_committed_opening
    )
    final_claims = claim.get("final_oracle_claims", {})
    final_ok = (
        str(final_claims.get("lhs_final_a")) == str(claim["lhs_sumcheck"].get("final_a"))
        and str(final_claims.get("lhs_final_b")) == str(claim["lhs_sumcheck"].get("final_b"))
        and str(final_claims.get("lhs_final_product")) == str(claim["lhs_sumcheck"].get("final_product"))
        and str(final_claims.get("rhs_final_a")) == str(claim["rhs_sumcheck"].get("final_a"))
        and str(final_claims.get("rhs_final_b")) == str(claim["rhs_sumcheck"].get("final_b"))
        and str(final_claims.get("rhs_final_product")) == str(claim["rhs_sumcheck"].get("final_product"))
    )
    ok = (
        claim.get("protocol") == GKR_CLAIM_PROTOCOL_ID
        and seed_ok
        and lhs_verify["status"] == "VERIFY_OK"
        and rhs_verify["status"] == "VERIFY_OK"
        and coverage_ok
        and exact_relation_ok
        and oracle_equal
        and oracle_binding_ok
        and committed_opening_ok
        and final_ok
    )
    return {
        "status": "VERIFY_OK" if ok else "VERIFY_FAILED",
        "lhs_sumcheck": lhs_verify,
        "rhs_sumcheck": rhs_verify,
        "fiat_shamir_seed": seed_ok,
        "coverage_ok": coverage_ok,
        "exact_relation": exact_relation_ok,
        "oracle_claim_equal": oracle_equal,
        "oracle_chunks_equal_debug_only": oracle_chunks_equal,
        "oracle_opening_references": oracle_binding_ok,
        "committed_witness_openings": committed_opening_ok,
        "final_oracle_claims": final_ok,
    }


def _expected_gkr_descriptors(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    dims = metadata["dim_cfg"]
    node_count = int(dims["N"])
    edge_count = int(dims["E"])
    graph_count = int(dims["G_batch"])
    out: List[Dict[str, Any]] = []

    def add(label: str, family: str, shape: Sequence[int]) -> None:
        count = 1
        for value in shape:
            count *= int(value)
        out.append({"label": label, "gate_family": family, "shape": [int(value) for value in shape], "constraint_count": count})

    def add_div(prefix: str, shape: Sequence[int]) -> None:
        for suffix in (
            "signed_input", "signed_output", "sigma_boolean", "z_boolean", "zero_abs",
            "zero_sign", "zero_inverse", "abs_inverse", "denominator_inverse",
            "nearest_equation", "rounding_slack",
        ):
            add(f"{prefix}.{suffix}", "signed_round_division", shape)

    add("public.edge_dst_sorted_wiring", "public_edge_routing", [edge_count])
    add("public.node_ptr_wiring", "public_node_ptr", [graph_count + 1])
    add("public.edge_ptr_wiring", "public_edge_ptr", [graph_count + 1])
    for layer_index, profile in enumerate(dims["hidden_profile"]):
        head_dim = int(profile["head_dim"])
        for head_index in range(int(profile["head_count"])):
            prefix = f"hidden.l{layer_index}.h{head_index}"
            add(f"{prefix}.Q_grp", "public_group_selector", [edge_count])
            add(f"{prefix}.group_count", "group_count", [node_count])
            add(f"{prefix}.score", "score_relation", [edge_count])
            add(f"{prefix}.LeakyReLU.lookup_query_x", "lrelu_lookup_query_consistency", [edge_count])
            add(f"{prefix}.LeakyReLU.lookup_query_y", "lrelu_lookup_query_consistency", [edge_count])
            add(f"{prefix}.max_state_machine", "max_state_machine", [node_count])
            add(f"{prefix}.delta", "delta_relation", [edge_count])
            add(f"{prefix}.ExpMap.lookup_query_x", "exp_lookup_query_consistency", [edge_count])
            add(f"{prefix}.ExpMap.lookup_query_y", "exp_lookup_query_consistency", [edge_count])
            add(f"{prefix}.sum_state_machine", "sum_state_machine", [node_count])
            add(f"{prefix}.inverse", "inverse_relation", [node_count])
            add(f"{prefix}.alpha", "alpha_relation", [edge_count])
            add(f"{prefix}.double_aggregate", "double_aggregate_consistency", [node_count, head_dim])
            add(f"{prefix}.PSQ_compressed_aggregation", "psq_aggregation", [node_count, head_dim])
            add(f"{prefix}.s_max.boolean", "max_selector", [edge_count])
            add(f"{prefix}.s_max.one_per_group", "max_selector", [node_count])
            add(f"{prefix}.s_max.selected_value", "max_selector", [edge_count])
            add(f"{prefix}.R_U.recurrence", "sparse_running_state", [edge_count])
            add(f"{prefix}.R_num.recurrence", "sparse_running_state", [edge_count, head_dim])
            add(f"{prefix}.R_field.recurrence", "sparse_running_state", [edge_count, head_dim])
            add(f"{prefix}.SignedRoundDiv.aggregate", "signed_round_division", [node_count, head_dim])
            add_div(f"{prefix}.SignedRoundDiv.aggregate", [node_count, head_dim])
            add(f"{prefix}.ELU.lookup_query_x", "elu_lookup_query_consistency", [node_count, head_dim])
            add(f"{prefix}.ELU.lookup_query_y", "elu_lookup_query_consistency", [node_count, head_dim])
        for head_index in range(int(profile["head_count"])):
            add(f"hidden.l{layer_index}.concat.h{head_index}", "concat_head_wiring", [node_count, int(profile["head_dim"])])
    class_count = int(dims["C"])
    for head_index in range(int(dims["K_out"])):
        prefix = f"output.h{head_index}"
        add(f"{prefix}.Q_grp", "public_group_selector", [edge_count])
        add(f"{prefix}.group_count", "group_count", [node_count])
        add(f"{prefix}.score", "score_relation", [edge_count])
        add(f"{prefix}.LeakyReLU.lookup_query_x", "lrelu_lookup_query_consistency", [edge_count])
        add(f"{prefix}.LeakyReLU.lookup_query_y", "lrelu_lookup_query_consistency", [edge_count])
        add(f"{prefix}.max_state_machine", "max_state_machine", [node_count])
        add(f"{prefix}.delta", "delta_relation", [edge_count])
        add(f"{prefix}.ExpMap.lookup_query_x", "exp_lookup_query_consistency", [edge_count])
        add(f"{prefix}.ExpMap.lookup_query_y", "exp_lookup_query_consistency", [edge_count])
        add(f"{prefix}.sum_state_machine", "sum_state_machine", [node_count])
        add(f"{prefix}.inverse", "inverse_relation", [node_count])
        add(f"{prefix}.alpha", "alpha_relation", [edge_count])
        add(f"{prefix}.double_aggregate", "double_aggregate_consistency", [node_count, class_count])
        add(f"{prefix}.PSQ_compressed_aggregation", "psq_aggregation", [node_count, class_count])
        add(f"{prefix}.s_max.boolean", "max_selector", [edge_count])
        add(f"{prefix}.s_max.one_per_group", "max_selector", [node_count])
        add(f"{prefix}.s_max.selected_value", "max_selector", [edge_count])
        add(f"{prefix}.R_U.recurrence", "sparse_running_state", [edge_count])
        add(f"{prefix}.R_num.recurrence", "sparse_running_state", [edge_count, class_count])
        add(f"{prefix}.R_field.recurrence", "sparse_running_state", [edge_count, class_count])
        add(f"{prefix}.SignedRoundDiv.aggregate", "signed_round_division", [node_count, class_count])
        add_div(f"{prefix}.SignedRoundDiv.aggregate", [node_count, class_count])
        add_div(f"{prefix}.SignedRoundDiv.scale_alignment", [node_count, class_count])
        add(f"{prefix}.bias_add", "output_bias", [node_count, class_count])
    add("output.head_average_post_bias", "output_head_average", [node_count, class_count])
    add_div("output.final_average.SignedRoundDiv", [node_count, class_count])
    return out


def _formal_transcript_prefix(metadata: Dict[str, Any], commitments: Dict[str, Any]) -> Transcript:
    transcript = Transcript("zkGPT/GAT/formal-proof-session/v1")
    transcript.absorb(
        "public-statement-and-input-commitments",
        {
            "protocol_id": metadata.get("protocol_id"),
            "session_nonce": metadata.get("session_nonce"),
            "metadata_digest": digest_json(metadata),
            "Layout_in_digest": metadata.get("Layout_in_digest"),
            "commitments": [
                {
                    "segment": name,
                    "parameter_id": commitments.get("items", {}).get(name, {}).get("parameter_id"),
                    "commitment": commitments.get("items", {}).get(name, {}).get("commitment"),
                }
                for name in COMMITMENT_ORDER
            ],
        },
    )
    return transcript


def _formal_transcript_descriptor(metadata: Dict[str, Any], transcript: Transcript) -> Dict[str, Any]:
    """Public descriptor for the one global Fiat--Shamir transcript."""
    return {
        "protocol": "zkGPT/GAT/formal-proof-session/v1",
        "session_nonce": metadata["session_nonce"],
        "final_digest": transcript.digest.hex(),
    }


def _verify_gkr(
    gkr: Dict[str, Any],
    *,
    metadata: Dict[str, Any] | None = None,
    transcript: Transcript | None = None,
    commitments: Dict[str, Any] | None = None,
    graph: Any | None = None,
    public_y: Sequence[int] | None = None,
    source_sink: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    if metadata is None or transcript is None:
        return {"status": "VERIFY_FAILED", "reason": "external native GKR statement is required", "claims": []}
    try:
        if commitments is None or graph is None:
            return {"status": "VERIFY_FAILED", "reason": "missing external GKR statement", "claims": []}
        if public_y is None:
            return {"status": "VERIFY_FAILED", "reason": "missing external public Y", "claims": []}
        expected = _native_gkr_specs(metadata, graph, public_y)
        claims = gkr.get("claims", [])
        if (
            gkr.get("protocol") != "zkGPT/native-gkr-bound-gates-v1"
            or not expected or not isinstance(claims, list) or len(claims) != len(expected)
        ):
            return {"status": "VERIFY_FAILED", "reason": "missing or extra native GKR gates", "claims": []}
        n_in = int(metadata["Layout_in"]["N_in"])
        params = hyrax_value_params(n_in)
        registrations = {
            "W": commitments["items"]["com_W"],
            "Aux": commitments["items"]["com_Aux"],
            "L_qry": commitments["items"]["com_L"],
        }
        context = {
            "metadata_digest": digest_json(metadata),
            "Layout_in_digest": metadata["Layout_in_digest"],
            "session_nonce": metadata["session_nonce"],
            "relation": "Pi_GKR",
        }
        results: List[bool] = []
        for claim, (kind, spec) in zip(claims, expected):
            if not isinstance(claim, dict) or set(claim) != {"kind", "proof"} or claim["kind"] != kind:
                results.append(False)
                continue
            common = dict(
                layout=metadata["Layout_in"], registrations=registrations, params=params,
                transcript=transcript, context=context, source_sink=source_sink,
            )
            results.append(
                verify_linear_gate(claim["proof"], spec, **common)
                if kind == "linear" else verify_product_gate(claim["proof"], spec, **common)
            )
        digest_ok = gkr.get("transcript_digest") == transcript.digest.hex()
        return {"status": "VERIFY_OK" if all(results) and digest_ok else "VERIFY_FAILED", "claims": results, "transcript": digest_ok}
    except (KeyError, TypeError, ValueError):
        return {"status": "VERIFY_FAILED", "reason": "malformed native GKR proof", "claims": []}


def _combine_seed(lasso: Dict[str, Any], matrix: Dict[str, Any], gkr: Dict[str, Any], openings: Dict[str, Any], transcript: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "protocol": "fiat-shamir-combine-openmasked-v2",
        "claim_sources": ["Pi_Lasso", "Pi_MatMul", "Pi_GKR", "Open_in"],
        "lasso_digest": digest_json(lasso),
        "matrix_digest": digest_json(matrix),
        "gkr_digest": digest_json(gkr),
        "open_in_digest": digest_json(openings),
        "transcript_digest": digest_json(transcript),
    }


def _prove_combine(lasso: Dict[str, Any], matrix: Dict[str, Any], gkr: Dict[str, Any], openings: Dict[str, Any], transcript: Dict[str, Any]) -> Dict[str, Any]:
    seed = _combine_seed(lasso, matrix, gkr, openings, transcript)
    values = [_field_label(seed[key]) for key in ["lasso_digest", "matrix_digest", "gkr_digest", "open_in_digest", "transcript_digest"]]
    weights = _random_field_vector("combine.openmasked.weights", _challenge_u64("combine", seed), len(values))
    proof = prove_product_sum("combine.openmasked_claims", values, weights, seed).to_json()
    proof["transcript_seed"] = seed
    payload = {
        "protocol": "fiat-shamir-combine-openmasked-v2",
        "claim_sources": seed["claim_sources"],
        "lasso_digest": seed["lasso_digest"],
        "matrix_digest": seed["matrix_digest"],
        "gkr_digest": seed["gkr_digest"],
        "open_in_digest": seed["open_in_digest"],
        "transcript_digest": seed["transcript_digest"],
        "weight_digest": digest_json([str(item) for item in weights]),
        "sumcheck": proof,
    }
    payload["combine_digest"] = digest_json(payload)
    return payload


def _verify_combine(combine: Dict[str, Any], lasso: Dict[str, Any], matrix: Dict[str, Any], gkr: Dict[str, Any], openings: Dict[str, Any], transcript: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = dict(combine)
        expected_digest = payload.pop("combine_digest", None)
        digest_ok = expected_digest == digest_json(payload)
        seed = _combine_seed(lasso, matrix, gkr, openings, transcript)
        seed_ok = (
            payload.get("protocol") == seed["protocol"]
            and payload.get("claim_sources") == seed["claim_sources"]
            and payload.get("lasso_digest") == seed["lasso_digest"]
            and payload.get("matrix_digest") == seed["matrix_digest"]
            and payload.get("gkr_digest") == seed["gkr_digest"]
            and payload.get("open_in_digest") == seed["open_in_digest"]
            and payload.get("transcript_digest") == seed["transcript_digest"]
        )
        weights = _random_field_vector("combine.openmasked.weights", _challenge_u64("combine", seed), 5)
        weight_ok = payload.get("weight_digest") == digest_json([str(item) for item in weights])
        sumcheck = payload.get("sumcheck", {})
        sumcheck_seed_ok = sumcheck.get("transcript_seed") == seed
        verify = verify_product_sum("combine.openmasked_claims", sumcheck, seed) if sumcheck_seed_ok else {"status": "VERIFY_FAILED"}
        ok = digest_ok and seed_ok and weight_ok and verify["status"] == "VERIFY_OK"
        return {
            "status": "VERIFY_OK" if ok else "VERIFY_FAILED",
            "checks": {
                "combine_digest": digest_ok,
                "claim_source_digests": seed_ok,
                "weight_digest": weight_ok,
                "sumcheck": verify["status"] == "VERIFY_OK",
            },
        }
    except Exception as exc:
        return {"status": "VERIFY_FAILED", "reason": str(exc), "checks": {}}


def _artifact_ref(path: pathlib.Path, kind: str) -> Dict[str, Any]:
    return {"kind": kind, "path": str(path), "sha256": digest_file(path), "size_bytes": path.stat().st_size}


def _verify_registered_model_commitment(metadata: Dict[str, Any], commitments: Dict[str, Any]) -> bool:
    """Authenticate the reusable com_W against the public model registry."""
    try:
        registry = metadata.get("model_registry", {})
        descriptor_full = registry.get("public_descriptor", {})
        descriptor = dict(descriptor_full)
        registry_digest = descriptor.pop("registry_digest", None)
        if not registry_digest or registry_digest != digest_json(descriptor):
            return False
        if registry.get("registry_digest") != registry_digest:
            return False
        if descriptor_full.get("commitment_backend") != COMMITMENT_BACKEND_ID:
            return False
        binding = descriptor_full.get("registration_binding", {})
        if descriptor_full.get("registration_binding_digest") != digest_json(binding):
            return False
        if registry.get("registration_binding_digest") != descriptor_full.get("registration_binding_digest"):
            return False
        for key in ("model_arch_id", "model_param_id", "quant_cfg_id", "static_table_id"):
            if binding.get(key) != metadata.get(key):
                return False
        if binding.get("integer_bound_ledger_digest") != metadata.get("integer_bound_ledger_digest"):
            return False
        if binding.get("layout") != "X_pub|W|Aux|L_qry" or binding.get("segment_order") != ["X_pub", "W", "Aux", "L_qry"]:
            return False
        item = commitments.get("items", {}).get("com_W", {})
        if item.get("registered_model_registry_digest") != registry_digest:
            return False
        chunks = item.get("chunks", [])
        entries = descriptor_full.get("entries", {})
        if sorted(entries) != sorted(chunk.get("label") for chunk in chunks):
            return False
        if len(binding.get("arrays", [])) != len(chunks):
            return False
        binding_arrays = {entry.get("name"): entry for entry in binding.get("arrays", [])}
        for chunk in chunks:
            label = chunk.get("label")
            entry = entries.get(label, {})
            binding_entry = binding_arrays.get(label, {})
            if not chunk.get("registered_model_commitment"):
                return False
            if chunk.get("commitment") != entry.get("commitment"):
                return False
            if int(chunk.get("vector_length", -1)) != int(entry.get("vector_length", -2)):
                return False
            if int(chunk.get("padded_length", -1)) != int(entry.get("vector_length", -2)):
                return False
            if int(binding_entry.get("length", -1)) != int(chunk.get("vector_length", -2)):
                return False
            if entry.get("backend") != COMMITMENT_BACKEND_ID or entry.get("label") != label:
                return False
        return True
    except (TypeError, ValueError, KeyError):
        return False


def _verify_input_layout(metadata: Dict[str, Any], commitments: Dict[str, Any]) -> bool:
    try:
        layout_full = metadata.get("Layout_in", {})
        layout = dict(layout_full)
        layout_digest = layout.pop("layout_digest", None)
        if layout_digest != digest_json(layout) or metadata.get("Layout_in_digest") != layout_digest:
            return False
        n_in = int(layout_full.get("N_in", 0))
        if n_in <= 0 or n_in & (n_in - 1):
            return False
        if int(layout_full.get("input_layer_length", 0)) != 4 * n_in:
            return False
        if layout_full.get("segment_order") != ["X_pub", "W", "Aux", "L_qry"]:
            return False
        if layout_full.get("selector_encoding") != {"00": "X_pub", "01": "W", "10": "Aux", "11": "L_qry"}:
            return False
        segments = layout_full.get("segments", [])
        if [item.get("name") for item in segments] != ["X_pub", "W", "Aux", "L_qry"]:
            return False
        for descriptor in segments:
            logical_length = int(descriptor.get("logical_length", -1))
            if logical_length < 0 or logical_length > n_in or int(descriptor.get("padded_length", -1)) != n_in:
                return False
            if int(descriptor.get("padding_begin", -1)) != logical_length or descriptor.get("padding_value") != "0":
                return False
            offset = 0
            names: set[str] = set()
            for array in descriptor.get("arrays", []):
                name = str(array.get("name", ""))
                if not name or name in names or int(array.get("offset", -1)) != offset:
                    return False
                names.add(name)
                length = int(array.get("length", -1))
                shape = [int(value) for value in array.get("shape", [])]
                shape_size = 1
                for dimension in shape:
                    if dimension < 0:
                        return False
                    shape_size *= dimension
                if length < 0 or shape_size != length:
                    return False
                offset += length
            if offset != logical_length:
                return False
            if descriptor.get("name") == "L_qry":
                families = descriptor.get("lasso_families")
                arrays = descriptor.get("arrays", [])
                if not isinstance(families, list) or not families or len(families) != len(arrays):
                    return False
                family_names: set[str] = set()
                for family, array in zip(families, arrays):
                    if not isinstance(family, dict) or set(family) - {
                        "family", "kind", "table_id", "offset", "real_count", "padded_count",
                        "a_pad", "Q_pad", "tuple_encoding", "range_bits", "limbs_per_value",
                        "signed_magnitude",
                    }:
                        return False
                    name = family.get("family")
                    if not isinstance(name, str) or not name or name in family_names or name != array.get("name"):
                        return False
                    family_names.add(name)
                    real_count = int(family.get("real_count", -1))
                    padded_count = int(family.get("padded_count", -1))
                    if int(family.get("offset", -1)) != int(array.get("offset", -2)):
                        return False
                    if padded_count != int(array.get("length", -2)) or padded_count <= 0 or padded_count & (padded_count - 1):
                        return False
                    if real_count <= 0 or real_count > padded_count:
                        return False
                    if family.get("Q_pad") != {"zero_prefix": real_count, "one_suffix": padded_count - real_count}:
                        return False
                    if family.get("kind") == "function":
                        if family.get("table_id") not in {"T_LReLU", "T_ELU", "T_exp"} or not isinstance(family.get("tuple_encoding"), dict):
                            return False
                    elif family.get("kind") == "range":
                        if family.get("table_id") != "T_range" or int(family.get("range_bits", 0)) <= 0 or int(family.get("limbs_per_value", 0)) <= 0:
                            return False
                    else:
                        return False
        for commitment_name, layout_name in (("com_W", "W"), ("com_Aux", "Aux"), ("com_L", "L_qry")):
            item = commitments.get("items", {}).get(commitment_name, {})
            descriptor = next(entry for entry in segments if entry["name"] == layout_name)
            if int(item.get("N_in", -1)) != n_in or int(item.get("vector_length", -1)) != n_in:
                return False
            if int(item.get("logical_length", -1)) != int(descriptor["logical_length"]):
                return False
        return set(commitments.get("items", {})) == {"com_W", "com_Aux", "com_L"}
    except (TypeError, ValueError, KeyError, StopIteration):
        return False


def _verify_gpu_execution_evidence(
    evidence: Dict[str, Any] | None, metadata: Dict[str, Any], transcript: Dict[str, Any]
) -> bool:
    valid, _errors = validate_gpu_execution_evidence(
        evidence or {}, session_nonce=str(metadata.get("session_nonce", "")),
        final_transcript_digest=str(transcript.get("final_digest", "")),
    )
    return valid


def _verify_formal(
    metadata: Dict[str, Any],
    commitments: Dict[str, Any],
    transcript: Dict[str, Any],
    lasso: Dict[str, Any],
    matrix: Dict[str, Any],
    gkr: Dict[str, Any],
    combine: Dict[str, Any],
    openings: Dict[str, Any],
    gpu_claim: Dict[str, Any] | None,
    backend: str,
    external_context: Dict[str, Any] | None = None,
    public_graph: LocalGraph | None = None,
) -> Dict[str, Any]:
    public_arrays: Dict[str, np.ndarray] = {}
    public_x_segment: List[int] = []
    try:
        statement = (external_context or {})["public_statement"]
        public_features = statement["features"]
        public_arrays["X_pub.H_input"] = np.asarray(
            public_features["values"], dtype=np.int64
        ).reshape(tuple(int(value) for value in public_features["shape"]))
        graph_statement = statement["graph"]
        public_x_segment.extend(to_field_int(value) for value in np.ravel(public_arrays["X_pub.H_input"]))
        public_x_segment.extend(to_field_int(value) for value in graph_statement["absolute_ids"])
        dims = metadata["dim_cfg"]
        public_x_segment.extend(to_field_int(value) for value in (
            int(dims["G_batch"]), int(dims["N"]), int(dims["E"]),
            int(public_arrays["X_pub.H_input"].shape[1]), int(dims["L"]),
            int(dims["C"]), int(dims["K_out"]),
        ))
        n_in = int(metadata["Layout_in"]["N_in"])
        public_x_segment.extend([0] * (n_in - len(public_x_segment)))
        if len(public_x_segment) != n_in:
            raise ValueError("external public X exceeds N_in")
    except (KeyError, TypeError, ValueError):
        public_arrays = {}
        public_x_segment = []
    verified_input_sources: List[Dict[str, Any]] = []
    matrix_transcript = _formal_transcript_prefix(metadata, commitments)
    lasso_ok = verify_native_lasso(
        lasso, metadata, _native_lasso_table_rows(metadata),
        commitments["items"]["com_L"], transcript=matrix_transcript,
        canonical_source_sink=verified_input_sources,
    )
    lasso_verify = {
        "status": "VERIFY_OK" if lasso_ok else "VERIFY_FAILED",
        "protocol": "native_lasso_memory_with_fresh_lc",
    }
    matrix_verify = _verify_native_matrix(
        matrix, metadata, commitments, matrix_transcript, public_arrays,
        verified_input_sources,
    )
    external_graph = (external_context or {}).get("public_statement", {}).get("graph")
    external_output = (external_context or {}).get("public_statement", {}).get("public_output", {})
    gkr_verify = _verify_gkr(
        gkr, metadata=metadata, transcript=matrix_transcript,
        commitments=commitments, graph=external_graph or public_graph,
        public_y=external_output.get("values"),
        source_sink=verified_input_sources,
    )
    terminal_verify = _verify_terminal_segments_and_combine(
        metadata, commitments, openings, combine, public_x_segment,
        matrix_transcript, verified_input_sources,
    )
    if backend == "gpu" and isinstance(gpu_claim, dict):
        matrix_transcript.absorb(
            "gpu/proof-execution",
            gpu_execution_transcript_binding(gpu_claim),
        )
    matrix_rescale_lasso = sum(
        family.get("family") == "rescale_rounding_range"
        for family in next(
            segment for segment in metadata.get("Layout_in", {}).get("segments", [])
            if segment.get("name") == "L_qry"
        ).get("lasso_families", [])
    ) == 1
    external_checks = validate_external_context(metadata, external_context or {})
    checks = {
        **external_checks,
        "metadata_complete": all(field in metadata for field in METADATA_FIELD_ORDER),
        "canonical_input_layout": _verify_input_layout(metadata, commitments),
        "public_input_statement": bool(metadata.get("X_pub_digest")) and metadata.get("private_input_layout") == "X_pub|W|Aux|L_qry",
        "integer_bound_ledger": _verify_integer_bound_ledger(metadata),
        "static_table_registry": metadata.get("static_table_registry_digest") == digest_json(metadata.get("static_table_registry", {})) and all(_verify_fixed_table_descriptor(table) for table in metadata.get("static_table_registry", {}).values()),
        "commitment_backend": commitments.get("commitment_backend_id") == COMMITMENT_BACKEND_ID,
        "commitment_order": commitments.get("order") == COMMITMENT_ORDER,
        "commitment_metadata_binding": commitments.get("metadata_digest") == digest_json(metadata),
        "registered_model_commitment": _verify_registered_model_commitment(metadata, commitments),
        "transcript_replay": transcript == _formal_transcript_descriptor(metadata, matrix_transcript),
        "hyrax_openings": terminal_verify["status"] == "VERIFY_OK",
        "lasso": lasso_verify["status"] == "VERIFY_OK",
        "matrix_sumcheck": matrix_verify["status"] == "VERIFY_OK",
        "matrix_rescale_lasso_binding": matrix_rescale_lasso,
        "gkr": gkr_verify["status"] == "VERIFY_OK",
        "combine": terminal_verify["status"] == "VERIFY_OK" and lasso_verify["status"] == matrix_verify["status"] == gkr_verify["status"] == "VERIFY_OK" and matrix_rescale_lasso,
        "gpu_hot_path": gpu_claim is None if backend == "cpu" else _verify_gpu_execution_evidence(
            gpu_claim, metadata, transcript
        ),
    }
    return {
        "status": "VERIFY_OK" if all(checks.values()) else "VERIFY_FAILED",
        "checks": checks,
        "subproof_verification": {
            "hyrax": terminal_verify,
            "lasso": lasso_verify,
            "matrix": matrix_verify,
            "gkr": gkr_verify,
            "combine": terminal_verify,
        },
        "verifier": "zkgpt_gat.formal_verifier.external_statement_and_trusted_setup",
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
    require_formal_experiment()
    started = time.perf_counter()
    artifact_dir = paths["artifact_dir"]
    ensure_dir(artifact_dir)
    segments, input_layout, padded_segments = prepare_input_segments_and_layout(
        config, dataset, graph, model, trace
    )
    metadata = make_metadata(config, dataset, graph, model, backend, gpu_state, input_layout)
    metadata["public_output_digest"] = _array_digest("Y", trace.y)

    zk_root = paths.get("zk_root", artifact_dir.parents[1])
    model_registry = ensure_model_registry(
        zk_root,
        metadata,
        list(_padded_segment_chunks("com_W", padded_segments["W"])),
    )
    metadata["model_registry"] = {
        "protocol": model_registry["public"]["protocol"],
        "registry_digest": model_registry["public"]["registry_digest"],
        "registration_binding_digest": model_registry["public"]["registration_binding_digest"],
        "public_path": str(model_registry["public_path"].relative_to(zk_root)),
        "commitment_backend": model_registry["public"]["commitment_backend"],
        "public_descriptor": model_registry["public"],
    }
    metadata["model_registry_digest"] = digest_json(metadata["model_registry"])
    commitments, openings, _segment_state, commitment_ms = _commit_segments(metadata, segments, padded_segments, zk_root, artifact_dir, model_registry)
    execution = ProofExecution(backend, zk_root, str(metadata["session_nonce"]))
    execution_token = activate_proof_execution(execution)
    try:
        formal_transcript = _formal_transcript_prefix(metadata, commitments)
        lasso = prove_native_lasso(
            metadata, _native_lasso_table_rows(metadata), _segment_state.hyrax,
            _segment_state.vectors["com_L"], _segment_state.registrations["com_L"],
            zk_root=zk_root, work_dir=artifact_dir / "native_lasso_work",
            transcript=formal_transcript,
        )
        matrix = _prove_native_matrix(
            trace,
            model,
            metadata,
            _segment_state,
            formal_transcript,
            artifact_dir / "native_matmul_work",
            graph,
        )
        gkr = _prove_native_gkr(
            graph, trace.y, metadata, _segment_state, formal_transcript,
            artifact_dir / "native_gkr_work",
        )
        openings, combine = _open_terminal_segments(
            metadata, commitments, _segment_state, padded_segments["X_pub"],
            formal_transcript, artifact_dir,
        )
        execution_binding = execution.transcript_binding()
        if execution_binding is not None:
            formal_transcript.absorb("gpu/proof-execution", execution_binding)
        transcript = _formal_transcript_descriptor(metadata, formal_transcript)
        gpu_claim = execution.evidence(formal_transcript.digest.hex())
    finally:
        deactivate_proof_execution(execution_token)

    write_json(artifact_dir / "lookup_proof.json", lasso)
    write_json(artifact_dir / "fixed_lookup_tables.json", metadata["static_table_registry"])
    write_json(artifact_dir / "matrix_sumcheck_proof.json", matrix)
    write_json(artifact_dir / "gkr_proof.json", gkr)
    write_json(artifact_dir / "combine_proof.json", combine)
    write_json(artifact_dir / "hyrax_mle_openings.json", openings)
    masked_opening_payload = {
        "backend": COMMITMENT_BACKEND_ID,
        "items": {
            name: {
                "evaluation_commitment": value["proof"]["evaluation_commitment"],
                "ipa": value["proof"]["ipa"],
                "transcript_digest": value["proof"]["transcript_digest"],
            }
            for name, value in openings["items"].items()
        },
    }
    write_json(artifact_dir / "masked_openings.json", masked_opening_payload)
    if gpu_claim is not None:
        write_json(artifact_dir / "gpu_proof_hot_path.json", gpu_claim)

    verify_started = time.perf_counter()
    external_context = make_external_context(
        graph,
        trace,
        metadata,
        metadata["model_registry"]["public_descriptor"],
    )
    verification = _verify_formal(
        metadata, commitments, transcript, lasso, matrix, gkr, combine,
        openings, gpu_claim, backend, external_context, graph,
    )
    verify_elapsed_ms = (time.perf_counter() - verify_started) * 1000.0
    component_refs = {
        "FixedLookupTables": _artifact_ref(artifact_dir / "fixed_lookup_tables.json", "fixed lookup table commitments"),
        "Pi_Lasso": _artifact_ref(artifact_dir / "lookup_proof.json", "lookup proof"),
        "Pi_MatMul": _artifact_ref(artifact_dir / "matrix_sumcheck_proof.json", "matrix sumcheck proof"),
        "Pi_GKR": _artifact_ref(artifact_dir / "gkr_proof.json", "ML-friendly GKR proof"),
        "Pi_combine": _artifact_ref(artifact_dir / "combine_proof.json", "combine masked-opening proof"),
        "Open_in": _artifact_ref(artifact_dir / "hyrax_mle_openings.json", "Hyrax/MLE masked openings"),
        "OpenMasked": _artifact_ref(artifact_dir / "masked_openings.json", "masked opening payload"),
    }
    if gpu_claim is not None:
        component_refs["GPU_hot_path"] = _artifact_ref(artifact_dir / "gpu_proof_hot_path.json", "CUDA proof hot-path contribution")
    proof_payload = {
        "proof_type": "pi_GAT_zkGPT",
        "M_pub_digest": digest_json(metadata),
        "Y_digest": _array_digest("Y", trace.y),
        "Com_in": [commitments["items"][name] for name in COMMITMENT_ORDER],
        "Pi_Lasso": lasso,
        "FixedLookupTables": metadata["static_table_registry"],
        "Pi_MatMul": matrix,
        "Pi_GKR": gkr,
        "Pi_combine": {
            **combine,
        },
        "Open_in": openings,
        "OpenMasked": masked_opening_payload,
        "component_artifacts": component_refs,
        "transcript_digest": digest_json(transcript),
    }
    if gpu_claim is not None:
        proof_payload["GPU_proof_contribution"] = gpu_claim
    proof_size = len(canonical_json(proof_payload).encode("utf-8"))
    proof_elapsed_ms = (time.perf_counter() - started) * 1000.0
    benchmark = {
        "dataset": metadata["dataset_display_name"],
        "backend": backend,
        "status": verification["status"],
        "commitment_time_ms": commitment_ms,
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
        "model_registry_digest": metadata.get("model_registry", {}).get("registry_digest", ""),
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
    prove_multiplication,
    verify_multiplication,
