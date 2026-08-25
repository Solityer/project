from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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
)
from .data import GraphDataset, LocalGraph
from .field import (
    BN254_FR_MODULUS,
    DEFAULT_FIXED_SCALE,
    log2_power_of_two,
    next_power_of_two,
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
from .forward import ForwardTrace, HeadTrace, SignedRoundDivTrace, _elu_table_value, _exp_map_fixed
from .layout import build_input_layout
from .input_oracle import equality_weights
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
from .privacy import (
    PROFILE_ID,
    bind_privacy_profile,
    build_privacy_profile,
    build_profile_anchor,
    fresh_session_nonce,
    transcript_domain,
    verify_privacy_profile,
)
from .registry import ensure_model_registry
from .resources import validate_static_table_descriptor
from .rounding import validate_rounding_mode
from .bound_ledger import build_integer_bound_ledger
from .transcript import Transcript
from .value_commitment import (
    MultiplicationProof,
    ValueCommitment,
    prove_multiplication,
    verify_multiplication,
)
from .zk_sumcheck import prove_committed_sumcheck, verify_committed_sumcheck
from .util import canonical_json, digest_file, digest_json, ensure_dir, file_id, write_json
from .verifier_context import make_external_context, validate_external_context


FUNCTION_LOOKUP_DOMAIN_BITS = 63


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


def _proof_work_dir(zk_root: pathlib.Path) -> pathlib.Path:
    """Create one authenticated, baseline-local proof work directory.

    Native Hyrax/GKR/Lasso helpers emit intermediate JSON transcripts.  Those
    files are implementation material only and must never become part of the
    serialized result cell, so each proof receives a unique directory under
    ``runs_tmp/proof-work``.  Every existing path component is checked before
    creation to prevent a symlink from redirecting work outside this baseline.
    """
    root = pathlib.Path(os.path.abspath(os.fspath(zk_root)))
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"proof work root must be a real directory: {root}")
    runs_tmp = root / "runs_tmp"
    if runs_tmp.exists() or os.path.lexists(runs_tmp):
        if runs_tmp.is_symlink() or not runs_tmp.is_dir():
            raise ValueError(f"runs_tmp must be a real directory: {runs_tmp}")
    else:
        runs_tmp.mkdir()
    work_parent = runs_tmp / "proof-work"
    if work_parent.exists() or os.path.lexists(work_parent):
        if work_parent.is_symlink() or not work_parent.is_dir():
            raise ValueError(f"proof-work root must be a real directory: {work_parent}")
    else:
        work_parent.mkdir()
    if work_parent.resolve(strict=True) != work_parent:
        raise ValueError(f"proof-work root resolves through an alias: {work_parent}")
    session = pathlib.Path(tempfile.mkdtemp(prefix="session-", dir=str(work_parent)))
    try:
        if session.is_symlink() or not session.is_dir() or session.resolve(strict=True).parent != work_parent:
            raise ValueError(f"proof work session is not contained: {session}")
        session.resolve(strict=True).relative_to(root.resolve(strict=True))
    except Exception:
        # The session name is unique and the parent has already been
        # authenticated; remove only this exact session on failed validation.
        if session.is_dir() and not session.is_symlink():
            shutil.rmtree(session)
        raise
    return session


def _cleanup_proof_work_dir(session: pathlib.Path, zk_root: pathlib.Path) -> None:
    """Remove one authenticated proof session without touching its parent."""
    root = pathlib.Path(os.path.abspath(os.fspath(zk_root))).resolve(strict=True)
    work_parent = (root / "runs_tmp" / "proof-work").resolve(strict=True)
    candidate = pathlib.Path(os.path.abspath(os.fspath(session)))
    try:
        if candidate.parent != work_parent:
            raise ValueError(f"proof work cleanup path is not session-local: {candidate}")
        if candidate.is_symlink():
            raise ValueError(f"proof work cleanup path is a symlink: {candidate}")
        candidate.resolve(strict=True).relative_to(work_parent)
        if candidate.exists():
            if not candidate.is_dir():
                raise ValueError(f"proof work cleanup path is not a directory: {candidate}")
            shutil.rmtree(candidate)
    except FileNotFoundError:
        return


def _signed_field(value: int) -> int:
    value = int(value) % BN254_FR_MODULUS
    if value > (BN254_FR_MODULUS - 1) // 2:
        return value - BN254_FR_MODULUS
    return value


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


def make_metadata(
    config: AppConfig,
    dataset: GraphDataset,
    graph: LocalGraph,
    model: Model,
    backend: str,
    gpu_state: Dict[str, Any],
    input_layout: Dict[str, Any],
) -> Dict[str, Any]:
    static_descriptor = validate_static_table_descriptor(config.project_root)
    rounding_mode = config.rounding_mode
    if validate_rounding_mode(model.rounding_mode) != rounding_mode:
        raise ValueError("model and formal config SignedRoundDiv rounding_mode differ")
    if config.entries.get("static_table_descriptor_digest") != static_descriptor["descriptor_digest"]:
        raise ValueError("formal config is not bound to the canonical static-table descriptor")
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
    session_nonce = fresh_session_nonce()
    metadata = {
        "metadata_order": METADATA_FIELD_ORDER,
        "protocol_id": PROTOCOL_ID,
        "proof_backend_id": PROOF_BACKEND_ID,
        "model_arch_id": model.model_arch_id,
        "model_param_id": model.model_param_id,
        "static_table_id": model.static_table_id,
        "quant_cfg_id": model.quant_cfg_id,
        "rounding_mode": rounding_mode,
        "static_table_descriptor_schema": static_descriptor["schema"],
        "static_table_content_digest": static_descriptor["content_digest"],
        "static_table_descriptor_digest": static_descriptor["descriptor_digest"],
        "static_table_descriptor_path": str(config.static_table_path.relative_to(config.project_root)),
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
        "privacy_extension_profile_id": PROFILE_ID,
        "privacy_profile_status": "supported_multi_session_masked_full",
        # Sampled before any commitment/transcript challenge and never
        # derived from ``com_W`` or public statement material.
        "session_nonce": session_nonce,
        "privacy_transcript_domain": transcript_domain(session_nonce),
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
        "protocol": "zkGPT-ephemeral-Hyrax-opening-v3-masked",
        "label": label,
        "logical_length": len(values),
        "vector_length": padded_length,
        "registration": public_registration,
        "context": context,
        "point": [str(value) for value in point],
        "opening": state.proof,
        # Duplicate only public mask schema fields at the active ephemeral
        # envelope boundary.  The nested native Hyrax proof remains the
        # authoritative IPA statement; no mask scalar or private vector is
        # serialized here.
        "mask_protocol_id": state.proof["mask_protocol_id"],
        "mask_domain": state.proof["mask_domain"],
        "mask_version": state.proof["mask_version"],
        "mask_commitment": state.proof["mask_commitment"],
        "mask_challenge": state.proof["mask_challenge"],
        "transcript_digest": transcript.digest.hex(),
    }


def _verify_ephemeral_hyrax_opening(label: str, payload: Dict[str, Any], context: Dict[str, Any]) -> bool:
    try:
        if payload.get("protocol") != "zkGPT-ephemeral-Hyrax-opening-v3-masked" or payload.get("label") != label:
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
        opening = payload["opening"]
        if not isinstance(opening, dict):
            return False
        for field in (
            "mask_protocol_id", "mask_domain", "mask_version",
            "mask_commitment", "mask_challenge",
        ):
            if payload.get(field) != opening.get(field):
                return False
        result = verify_open_payload(
            label,
            opening,
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
        (
            f"{head.tag}.gkr.double_aggregate",
            _field_product_array(
                np.repeat(np.asarray(head.sum_, dtype=object)[:, None], int(head.h_agg_pre.shape[1]), axis=1),
                np.asarray(head.h_agg_pre, dtype=object),
            ),
        ),
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


def _pad_lasso_family(values: Sequence[int], pad_value: int) -> List[int]:
    logical = [to_field_int(value) for value in values]
    # Native committed Sumcheck has at least one folding round.  A legal
    # one-edge/one-node GAT instance therefore needs the smallest non-empty
    # family to be padded to two coordinates; verifier-side family rebuilding
    # mirrors this rule.
    padded_count = next_power_of_two(max(2, len(logical)))
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
    limb_base = 1 << int(range_bits)
    family_policy = ledger.get("range_family_policy", {}).get("families", {})

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
        policy = family_policy.get(label)
        if not isinstance(policy, dict):
            raise ValueError(f"missing canonical range-family policy: {label}")
        if int(policy.get("range_bits", -1)) != int(range_bits):
            raise ValueError(f"range-family bit width differs from authenticated ledger: {label}")
        limb_count = int(policy.get("limb_count", 0))
        bit_width = int(policy.get("bit_width", 0))
        if limb_count <= 0 or bit_width <= 0:
            raise ValueError(f"invalid canonical range-family policy: {label}")
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
            "bit_width": bit_width,
            "limbs_per_value": limb_count,
            "bound": int(policy.get("bound", -1)),
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
    static_descriptor = validate_static_table_descriptor(config.project_root)
    segments = _segment_arrays(dataset, graph, model, trace, None)
    provisional_ledger = _integer_bound_ledger(
        {"rounding_mode": config.rounding_mode, "dim_cfg": {"B": config.range_bits}},
        model,
        graph,
    )
    provisional_metadata = {
        "quant_cfg_id": model.quant_cfg_id,
        "rounding_mode": config.rounding_mode,
        "static_table_id": model.static_table_id,
        "static_table_descriptor_schema": static_descriptor["schema"],
        "static_table_content_digest": static_descriptor["content_digest"],
        "static_table_descriptor_digest": static_descriptor["descriptor_digest"],
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


def _commit_segments(
    metadata: Dict[str, Any],
    segments: Dict[str, List[Tuple[str, np.ndarray]]],
    padded_segments: Dict[str, List[int]],
    zk_root: pathlib.Path,
    work_root: pathlib.Path,
    model_registry: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], SegmentCommitmentState, float]:
    started = time.perf_counter()
    hyrax = NativeHyrax(zk_root)
    work_dir = work_root / "hyrax_work"
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
        if registered is not None:
            # A persisted model-registry secret belongs to an earlier setup
            # session.  Authenticate it once in this proof session so all
            # subsequent registered linear claims use the bounded Hyrax cache.
            registration = hyrax.register_vector(name, vector, registered, work_dir)
        else:
            registration = hyrax.commit_vector(name, vector, work_dir)
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
    # Bind the profile to the exact public segment commitments before any
    # native sub-proof is generated.  The anchor excludes ``metadata_digest``
    # and is therefore not circular; the resulting metadata digest then
    # authenticates the anchor itself.
    metadata["privacy_profile_anchor"] = build_profile_anchor(
        metadata,
        {"items": items},
    )
    commitments = {
        "commitment_backend_id": COMMITMENT_BACKEND_ID,
        "order": COMMITMENT_ORDER,
        "metadata_digest": digest_json(metadata),
        "items": items,
    }

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return commitments, {}, SegmentCommitmentState(hyrax, vectors, registrations), elapsed_ms


def _open_terminal_segments(
    metadata: Dict[str, Any],
    commitments: Dict[str, Any],
    state: SegmentCommitmentState,
    public_x: Sequence[int],
    transcript: Transcript,
    work_root: pathlib.Path,
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
            work_root / "hyrax_work", transcript=transcript, public_context=context,
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


def _division_range_values(division: SignedRoundDivTrace) -> np.ndarray:
    return np.concatenate([np.ravel(getattr(division, name)) for name in ("a", "q", "r", "delta", "d")]).astype(object)


def _static_lookup_registry(metadata: Dict[str, Any], range_bits: int) -> Dict[str, Any]:
    ledger = metadata.get("integer_bound_ledger", {})
    common = {
        "quant_cfg_id": metadata["quant_cfg_id"],
        "rounding_mode": validate_rounding_mode(metadata.get("rounding_mode")),
        "quant_scale_chain": dict(ledger.get("scales", {})),
        "static_table_id": metadata["static_table_id"],
        "static_table_descriptor_schema": metadata.get("static_table_descriptor_schema"),
        "static_table_content_digest": metadata.get("static_table_content_digest"),
        "static_table_descriptor_digest": metadata.get("static_table_descriptor_digest"),
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
    """Build the mode-authenticated public ledger shared with the verifier."""
    mode = validate_rounding_mode(metadata.get("rounding_mode"))
    registered: Dict[str, Dict[str, Any]] = {}
    for layer in model.hidden_layers:
        for head_index, head in enumerate(layer.heads):
            family = f"hidden.l{layer.layer_index + 1}.h{head_index}"
            registered[family] = {
                "W_shape": [int(value) for value in head.seq_kernel.shape],
                "a_src_shape": [int(value) for value in head.attn_src_kernel.shape],
                "a_dst_shape": [int(value) for value in head.attn_dst_kernel.shape],
                "B_W": max((abs(int(value)) for value in np.ravel(head.seq_kernel)), default=0),
                "B_a_src": max((abs(int(value)) for value in np.ravel(head.attn_src_kernel)), default=0),
                "B_a_dst": max((abs(int(value)) for value in np.ravel(head.attn_dst_kernel)), default=0),
                "B_b": 0,
            }
    for head_index, head in enumerate(model.output_heads):
        family = f"output.h{head_index}"
        registered[family] = {
            "W_shape": [int(value) for value in head.seq_kernel.shape],
            "a_src_shape": [int(value) for value in head.attn_src_kernel.shape],
            "a_dst_shape": [int(value) for value in head.attn_dst_kernel.shape],
            "b_shape": [int(value) for value in head.output_bias.shape],
            "B_W": max((abs(int(value)) for value in np.ravel(head.seq_kernel)), default=0),
            "B_a_src": max((abs(int(value)) for value in np.ravel(head.attn_src_kernel)), default=0),
            "B_a_dst": max((abs(int(value)) for value in np.ravel(head.attn_dst_kernel)), default=0),
            "B_b": max((abs(int(value)) for value in np.ravel(head.output_bias)), default=0),
        }
    hidden_profile = [
        {"layer_index": int(layer.layer_index), "head_count": int(layer.head_count), "head_dim": int(layer.head_dim)}
        for layer in model.hidden_layers
    ]
    feature_bound = max((abs(int(round(float(value)))) for value in np.ravel(graph.features)), default=0)
    return build_integer_bound_ledger(
        mode=mode,
        quant_cfg_id=model.quant_cfg_id,
        authenticated_quant_scales=model.quant_scales,
        registered_model_bounds=registered,
        hidden_profile=hidden_profile,
        d_in=[int(model.manifest["d_in_profile"][0])] + [int(layer.head_count * layer.head_dim) for layer in model.hidden_layers],
        classes=int(model.class_count),
        k_out=int(model.k_out),
        max_degree=max((int(value) for value in np.bincount(graph.dst, minlength=graph.num_nodes)), default=1),
        feature_bound=feature_bound,
        node_count=int(graph.num_nodes),
        range_bits=int(metadata.get("dim_cfg", {}).get("B", 12)),
    )


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
    if not {"table_id", "table_kind", "quant_cfg_id", "rounding_mode", "static_table_id", "encoding_id", "range_bits", "table_digest", "table_commitment"} <= set(table):
        return False
    try:
        validate_rounding_mode(table.get("rounding_mode"))
    except ValueError:
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
    """Independently re-derive and authenticate every public bound instance.

    The v2 ledger is a complete relation manifest, not a best-effort summary.
    Rebuilding it from the public architecture, registered model bounds and
    authenticated graph scalars catches missing families, mode/denominator
    substitutions, stale range geometry, and all formula-level tampering.
    """

    try:
        ledger = metadata.get("integer_bound_ledger")
        if not isinstance(ledger, dict) or ledger.get("schema") != "zkGPT_gat_integer_bound_ledger_v2":
            return False
        if ledger.get("field") != "BN254-Fr" or int(ledger.get("field_half_modulus", -1)) != (BN254_FR_MODULUS - 1) // 2:
            return False
        if ledger.get("failures"):
            return False
        mode = validate_rounding_mode(metadata.get("rounding_mode"))
        if ledger.get("rounding_mode") != mode:
            return False
        quant_cfg_id = str(metadata.get("quant_cfg_id", ""))
        if quant_cfg_id not in QUANT_SCALE_PROFILES or ledger.get("quant_cfg_id") != quant_cfg_id:
            return False
        if ledger.get("authenticated_quant_scales") != QUANT_SCALE_PROFILES[quant_cfg_id]:
            return False
        dims = metadata.get("dim_cfg", {})
        hidden_profile = dims.get("hidden_profile", [])
        d_in = [int(value) for value in dims.get("d_in", [])]
        if not isinstance(hidden_profile, list) or not hidden_profile or len(d_in) != len(hidden_profile) + 1:
            return False
        registered = ledger.get("registered_model_bounds")
        if not isinstance(registered, dict):
            return False
        expected_families = {
            *(f"hidden.l{layer_index}.h{head}" for layer_index, layer in enumerate(hidden_profile, start=1) for head in range(int(layer["head_count"]))),
            *(f"output.h{head}" for head in range(int(dims.get("K_out", 0)))),
        }
        if set(registered) != expected_families:
            return False
        expected = build_integer_bound_ledger(
            mode=mode,
            quant_cfg_id=quant_cfg_id,
            authenticated_quant_scales=QUANT_SCALE_PROFILES[quant_cfg_id],
            registered_model_bounds=registered,
            hidden_profile=[
                {"layer_index": int(index), "head_count": int(item["head_count"]), "head_dim": int(item["head_dim"])}
                for index, item in enumerate(hidden_profile)
            ],
            d_in=d_in,
            classes=int(dims.get("C", 0)),
            k_out=int(dims.get("K_out", 0)),
            max_degree=int(ledger.get("max_in_degree", 0)),
            feature_bound=int(ledger.get("feature_bound", -1)),
            node_count=int(dims.get("N", 0)),
            range_bits=int(dims.get("B", 0)),
        )
        if ledger != expected:
            return False
        if digest_json(ledger) != metadata.get("integer_bound_ledger_digest"):
            return False

        divisions = ledger.get("signed_round_divisions")
        ids = ledger.get("division_instance_ids")
        cardinality = ledger.get("division_cardinality")
        if not isinstance(divisions, list) or not isinstance(ids, list) or not isinstance(cardinality, dict):
            return False
        expected_ids = [str(item["id"]) for item in divisions]
        if ids != expected_ids or len(ids) != len(set(ids)) or set(cardinality) != set(ids):
            return False
        for division in divisions:
            if not isinstance(division, dict) or division.get("rounding_mode") != mode:
                return False
            if int(division.get("count", 0)) <= 0 or int(cardinality.get(division["id"], -1)) != int(division["count"]):
                return False
            a_bound = int(division.get("B_a", -1)); q_bound = int(division.get("B_q", -1)); r_bound = int(division.get("B_r", -1)); delta_bound = int(division.get("B_delta", -1)); d_bound = int(division.get("B_d", -1)); d_min = int(division.get("B_d_min", -1))
            if min(a_bound, q_bound, r_bound, delta_bound) < 0 or d_bound <= 0 or d_min <= 0 or d_min > d_bound:
                return False
            constant = 2 * d_bound - 1 if mode == "nearest_ties_away" else d_bound - 1
            if r_bound != constant or delta_bound != constant or int(division.get("B_rounding_constant", -1)) != constant or int(division.get("B_2d_minus_1", -1)) != constant:
                return False
            if q_bound != (a_bound + d_min - 1) // d_min:
                return False
            lhs_bound = 2 * a_bound + d_bound if mode == "nearest_ties_away" else a_bound
            rhs_bound = (2 * d_bound * q_bound + constant) if mode == "nearest_ties_away" else (d_bound * q_bound + constant)
            if int(division.get("B_2a_plus_d", -1)) != lhs_bound or int(division.get("B_2dq_plus_r", -1)) != rhs_bound:
                return False
            if int(division.get("B_r_plus_delta", -1)) != constant or int(division.get("B_r_plus_delta_independent", -1)) != 2 * constant:
                return False
            if int(division.get("equality_lhs_bound", -1)) != lhs_bound or int(division.get("equality_rhs_bound", -1)) != rhs_bound:
                return False
            if int(division.get("B_x_independent", -1)) < 0 or int(division.get("B_x_independent", -1)) > a_bound:
                return False
        final = ledger.get("final_average")
        if not isinstance(final, dict) or final.get("division_id") != "output.final_average.SignedRoundDiv":
            return False
        if final.get("denominator") != int(dims.get("K_out", 0)) or final.get("K_out") != int(dims.get("K_out", 0)):
            return False
        policy = ledger.get("range_family_policy", {})
        families = policy.get("families") if isinstance(policy, dict) else None
        expected_range_families = {
            f"{item['id']}.range" for item in divisions
            if str(item["id"]) != "output.final_average.SignedRoundDiv"
        } | {
            "output.final_average.SignedRoundDiv.range",
            *(
                (f"hidden.l{item['layer']}.h{item['head']}.range_delta" if item["kind"] == "hidden" else f"output.h{item['head']}.range_delta")
                for item in ledger.get("heads", [])
            ),
            "rescale_rounding_range",
        }
        if not isinstance(families, dict) or set(families) != expected_range_families:
            return False
        for name, item in families.items():
            if not isinstance(item, dict) or int(item.get("bound", -1)) < 0 or int(item.get("bit_width", 0)) <= 0 or int(item.get("range_bits", 0)) != int(dims.get("B", 0)) or int(item.get("limb_count", 0)) != max(1, (int(item["bit_width"]) + int(dims["B"]) - 1) // int(dims["B"])):
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


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


def _verify_matrix_rescale_lasso_binding(
    matrix: Dict[str, Any], lasso: Dict[str, Any], gkr: Dict[str, Any],
) -> bool:
    """Require native MatMul's exact-zero residual relation to be wired.

    Projection and attention are exact integer products in ``forward.py``.
    Their residual arrays therefore are not SignedRoundDiv witnesses.  The
    native MatMul proof binds the product to output plus residual, while the
    native GKR residual-zero gates bind every committed residual coordinate to
    zero.  The Lasso range family remains a canonical, signed range witness
    for that same residual vector and is checked here for cardinality/source
    consistency.  This function intentionally does not accept the removed
    legacy matrix-rescale payload.
    """
    try:
        families = [item for item in lasso.get("families", []) if item.get("family") == "rescale_rounding_range"]
        if len(families) != 1:
            return False
        family_wire = families[0]
        descriptor = family_wire.get("descriptor")
        if not isinstance(descriptor, dict):
            return False
        if (
            descriptor.get("family") != "rescale_rounding_range"
            or descriptor.get("kind") != "range"
            or descriptor.get("table_id") != "T_range"
            or descriptor.get("signed_magnitude") is not True
            or family_wire.get("table_id") != "T_range"
            or int(family_wire.get("query_count", -1)) != int(descriptor.get("padded_count", -2))
        ):
            return False
        claims = matrix.get("claims")
        if not isinstance(claims, list) or not claims:
            return False
        residual_sources: list[dict[str, str]] = []
        residual_count = 0
        expected_zero_labels: list[str] = []
        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("spec"), dict):
                return False
            spec = claim["spec"]
            if spec.get("residual_semantics") != "exact_zero" or int(spec.get("output_scale", -1)) != 1:
                return False
            shape = spec.get("shape")
            residual = spec.get("residual")
            if not isinstance(shape, list) or len(shape) != 3 or not isinstance(residual, dict):
                return False
            if residual.get("segment") != "Aux" or not residual.get("array_name"):
                return False
            entry_count = int(shape[0]) * int(shape[2])
            if entry_count <= 0:
                return False
            residual_count += entry_count
            source = {"segment": str(residual["segment"]), "array_name": str(residual["array_name"])}
            if source in residual_sources:
                return False
            residual_sources.append(source)
            expected_zero_labels.append(f"{spec.get('label')}.residual_zero")
            binding = claim.get("residual_binding")
            if binding != {
                "protocol": "zkGPT/native-matmul-exact-zero-residual-v1",
                "semantics": "exact_zero",
                "source": residual,
                "shape": [int(shape[0]), int(shape[2])],
                "value": "0",
            }:
                return False
        limbs = int(descriptor.get("limbs_per_value", 0))
        if limbs <= 0 or int(descriptor.get("real_count", -1)) != residual_count * limbs:
            return False
        if int(descriptor.get("padded_count", 0)) < int(descriptor.get("real_count", 0)):
            return False
        gkr_claims = gkr.get("claims")
        if not isinstance(gkr_claims, list):
            return False
        actual_zero_labels = [
            item.get("proof", {}).get("spec", {}).get("label")
            for item in gkr_claims
            if isinstance(item, dict) and isinstance(item.get("proof"), dict)
            and str(item.get("proof", {}).get("spec", {}).get("label", "")).endswith(".residual_zero")
        ]
        return actual_zero_labels == expected_zero_labels and len(actual_zero_labels) == len(set(actual_zero_labels))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _native_gkr_array_length(layout: Dict[str, Any], segment: str, name: str) -> int:
    matches = [
        array
        for descriptor in layout.get("segments", []) if descriptor.get("name") == segment
        for array in descriptor.get("arrays", []) if array.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"native GKR array missing from Layout_in: {segment}/{name}")
    return int(matches[0]["length"])


def _native_division_specs(
    prefix: str, count: int, rounding_mode: str
) -> List[Tuple[str, Any]]:
    rounding_mode = validate_rounding_mode(rounding_mode)
    aux = lambda name: identity_oracle("Aux", f"{prefix}.{name}", count)
    product = lambda name: identity_oracle("Aux", f"{prefix}.gkr.{name}", count)
    zero = (0,) * count
    one = (1,) * count
    inv_two = pow(2, BN254_FR_MODULUS - 2, BN254_FR_MODULUS)
    equation = (
        ("nearest_equation", aux("a"), (
            LinearTerm(1, product("d_q")),
            LinearTerm(inv_two, aux("r")),
            LinearTerm(-inv_two, aux("d")),
        ), zero)
        if rounding_mode == "nearest_ties_away"
        else ("toward_zero_equation", aux("a"), (
            LinearTerm(1, product("d_q")), LinearTerm(1, aux("r")),
        ), zero)
    )
    slack_terms = (
        (LinearTerm(2, aux("d")), LinearTerm(-1, aux("r"))),
        (-1,) * count,
    ) if rounding_mode == "nearest_ties_away" else (
        (LinearTerm(1, aux("d")), LinearTerm(-1, aux("r"))),
        (-1,) * count,
    )
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
        ("linear", LinearGateSpec(f"{prefix}.{equation[0]}", equation[1], equation[2], equation[3])),
        ("linear", LinearGateSpec(f"{prefix}.rounding_slack", aux("delta"), slack_terms[0], slack_terms[1])),
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
            policy = metadata.get("integer_bound_ledger", {}).get("range_family_policy", {}).get("families", {}).get(label)
            if not isinstance(policy, dict):
                raise ValueError(f"range family is absent from authenticated ledger: {label}")
            if any(int(family.get(key, -1)) != int(policy.get(source, -2)) for key, source in (("range_bits", "range_bits"), ("bit_width", "bit_width"), ("limbs_per_value", "limb_count"), ("bound", "bound"))):
                raise ValueError(f"range family geometry differs from authenticated ledger: {label}")
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
    rounding_mode = validate_rounding_mode(metadata.get("rounding_mode"))
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
        specs.extend([
            ("product", ProductGateSpec(
                f"{prefix}.double_aggregate_product",
                identity_oracle("Aux", f"{prefix}.gkr.double_aggregate", value_count),
                aux("Sum", tuple(row for row in range(node_count) for _ in range(width))),
                identity_oracle("Aux", f"{prefix}.H_agg_pre", value_count),
            )),
            ("linear", LinearGateSpec(
                f"{prefix}.double_aggregate_equality",
                identity_oracle("Aux", f"{prefix}.H_or_Y_agg_num", value_count),
                (LinearTerm(1, identity_oracle("Aux", f"{prefix}.gkr.double_aggregate", value_count)),),
                (0,) * value_count,
            )),
        ])
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
        specs.extend(_native_division_specs(aggregate_prefix, aggregate_count, rounding_mode))
        if output:
            alignment_prefix = f"{prefix}.SignedRoundDiv.scale_alignment"
            alignment_count = _native_gkr_array_length(layout, "Aux", f"{alignment_prefix}.x")
            specs.extend(_native_division_specs(alignment_prefix, alignment_count, rounding_mode))
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
    specs.extend(_native_division_specs(final_prefix, final_count, rounding_mode))
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
    for matrix_spec in _expected_native_matrix_specs(metadata):
        residual_size = int(matrix_spec.shape[0]) * int(matrix_spec.shape[2])
        residual_oracle = identity_oracle("Aux", matrix_spec.residual.array_name, residual_size)
        specs.append(("linear", LinearGateSpec(
            f"{matrix_spec.label}.residual_zero",
            residual_oracle,
            (),
            (0,) * residual_size,
        )))
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
    return {
        "protocol": "zkGPT/native-gkr-bound-gates-v1",
        "rounding_mode": validate_rounding_mode(metadata.get("rounding_mode")),
        "claims": claims,
        "transcript_digest": transcript.digest.hex(),
    }


def _field_product_array(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Elementwise BN254 field product for committed witness advice arrays."""
    lhs = np.asarray(left, dtype=object)
    rhs = np.asarray(right, dtype=object)
    if lhs.shape != rhs.shape:
        raise ValueError(f"field product shape mismatch: {lhs.shape} != {rhs.shape}")
    out = np.empty(lhs.shape, dtype=object)
    for index in np.ndindex(lhs.shape):
        out[index] = (int(lhs[index]) * int(rhs[index])) % BN254_FR_MODULUS
    return out


def _formal_transcript_prefix(metadata: Dict[str, Any], commitments: Dict[str, Any]) -> Transcript:
    transcript = Transcript(transcript_domain(str(metadata.get("session_nonce", ""))))
    transcript.absorb(
        "public-statement-and-input-commitments",
        {
            "protocol_id": metadata.get("protocol_id"),
            "session_nonce": metadata.get("session_nonce"),
            "rounding_mode": validate_rounding_mode(metadata.get("rounding_mode")),
            "integer_bound_ledger_digest": metadata.get("integer_bound_ledger_digest"),
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


def _formal_transcript_descriptor(
    metadata: Dict[str, Any], transcript: Transcript, privacy_profile_digest: str | None = None,
    pre_profile_transcript_digest: str | None = None,
) -> Dict[str, Any]:
    """Public descriptor for the one global Fiat--Shamir transcript."""
    descriptor = {
        "protocol": "zkGPT/GAT/formal-proof-session/v1",
        "transcript_domain": transcript_domain(metadata["session_nonce"]),
        "session_nonce": metadata["session_nonce"],
        "final_digest": transcript.digest.hex(),
    }
    if privacy_profile_digest is not None:
        descriptor["privacy_profile_digest"] = privacy_profile_digest
    if pre_profile_transcript_digest is not None:
        descriptor["pre_profile_transcript_digest"] = pre_profile_transcript_digest
    return descriptor


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
        rounding_mode = validate_rounding_mode(metadata.get("rounding_mode"))
        if (
            gkr.get("protocol") != "zkGPT/native-gkr-bound-gates-v1"
            or gkr.get("rounding_mode") != rounding_mode
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
        for key in ("model_arch_id", "model_param_id", "quant_cfg_id", "rounding_mode", "static_table_id"):
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
        if not isinstance(chunks, list) or not isinstance(entries, dict):
            return False
        if sorted(entries) != sorted(chunk.get("label") for chunk in chunks):
            return False
        binding_array_values = binding.get("arrays", [])
        if (
            not isinstance(binding_array_values, list)
            or any(not isinstance(array, dict) for array in binding_array_values)
            or len(binding_array_values) != len(chunks)
        ):
            return False
        binding_arrays = {entry.get("name"): entry for entry in binding_array_values}
        for chunk in chunks:
            if not isinstance(chunk, dict):
                return False
            label = chunk.get("label")
            entry = entries.get(label, {})
            binding_entry = binding_arrays.get(label, {})
            if not isinstance(entry, dict) or not isinstance(binding_entry, dict):
                return False
            if not chunk.get("registered_model_commitment"):
                return False
            if chunk.get("commitment") != entry.get("commitment"):
                return False
            if int(chunk.get("vector_length", -1)) != int(entry.get("vector_length", -2)):
                return False
            if "padded_length" not in entry:
                return False
            if (
                int(entry.get("padded_length", -1)) != int(entry.get("vector_length", -2))
                or int(chunk.get("padded_length", -1)) != int(entry.get("padded_length", -2))
            ):
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
                        "a_pad", "Q_pad", "tuple_encoding", "range_bits", "bit_width",
                        "limbs_per_value", "bound", "signed_magnitude",
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
                    if padded_count != int(array.get("length", -2)) or padded_count < 2 or padded_count & (padded_count - 1):
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
                        policy = metadata.get("integer_bound_ledger", {}).get("range_family_policy", {}).get("families", {}).get(name)
                        if not isinstance(policy, dict):
                            return False
                        if any(int(family.get(key, -1)) != int(policy.get(source, -2)) for key, source in (("range_bits", "range_bits"), ("bit_width", "bit_width"), ("limbs_per_value", "limb_count"), ("bound", "bound"))):
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
    privacy_profile: Dict[str, Any],
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
    pre_profile_transcript_digest = matrix_transcript.digest.hex()
    privacy_payload = {
        "Pi_Lasso": lasso,
        "Pi_MatMul": matrix,
        "Pi_GKR": gkr,
        "Pi_combine": combine,
        "Open_in": openings,
        "PrivacyProfile": privacy_profile,
        "privacy_profile_digest": privacy_profile.get("profile_digest"),
    }
    privacy_verify = verify_privacy_profile(
        metadata,
        commitments,
        privacy_payload,
        pre_profile_transcript_digest=pre_profile_transcript_digest,
        components={
            "lasso": lasso,
            "matmul": matrix,
            "gkr": gkr,
            "input_combine": combine,
            "terminal_openings": openings,
        },
    )
    if privacy_verify["status"] == "VERIFY_OK":
        bind_privacy_profile(matrix_transcript, privacy_profile)
    matrix_rescale_lasso = _verify_matrix_rescale_lasso_binding(matrix, lasso, gkr)
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
        "transcript_replay": transcript == _formal_transcript_descriptor(
            metadata,
            matrix_transcript,
            privacy_profile.get("profile_digest") if isinstance(privacy_profile, dict) else None,
            pre_profile_transcript_digest,
        ),
        "privacy_profile": privacy_verify["status"] == "VERIFY_OK",
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
            "privacy_profile": privacy_verify,
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
    proof_work_dir = _proof_work_dir(zk_root)
    try:
        commitments, openings, _segment_state, commitment_ms = _commit_segments(
            metadata, segments, padded_segments, zk_root, proof_work_dir, model_registry
        )
        execution = ProofExecution(backend, zk_root, str(metadata["session_nonce"]))
        execution_token = activate_proof_execution(execution)
        try:
            formal_transcript = _formal_transcript_prefix(metadata, commitments)
            lasso = prove_native_lasso(
                metadata, _native_lasso_table_rows(metadata), _segment_state.hyrax,
                _segment_state.vectors["com_L"], _segment_state.registrations["com_L"],
                zk_root=zk_root, work_dir=proof_work_dir / "native_lasso_work",
                transcript=formal_transcript,
            )
            matrix = _prove_native_matrix(
                trace,
                model,
                metadata,
                _segment_state,
                formal_transcript,
                proof_work_dir / "native_matmul_work",
                graph,
            )
            gkr = _prove_native_gkr(
                graph, trace.y, metadata, _segment_state, formal_transcript,
                proof_work_dir / "native_gkr_work",
            )
            openings, combine = _open_terminal_segments(
                metadata, commitments, _segment_state, padded_segments["X_pub"],
                formal_transcript, proof_work_dir,
            )
            execution_binding = execution.transcript_binding()
            if execution_binding is not None:
                formal_transcript.absorb("gpu/proof-execution", execution_binding)
            privacy_profile = build_privacy_profile(
                metadata,
                commitments,
                lasso=lasso,
                matrix=matrix,
                gkr=gkr,
                input_combine=combine,
                terminal_openings=openings,
                pre_profile_transcript_digest=formal_transcript.digest.hex(),
            )
            bind_privacy_profile(formal_transcript, privacy_profile)
            transcript = _formal_transcript_descriptor(
                metadata, formal_transcript, privacy_profile["profile_digest"],
                privacy_profile["pre_profile_transcript_digest"],
            )
            gpu_claim = execution.evidence(formal_transcript.digest.hex())
        finally:
            deactivate_proof_execution(execution_token)
    finally:
        _cleanup_proof_work_dir(proof_work_dir, zk_root)

    write_json(artifact_dir / "lookup_proof.json", lasso)
    write_json(artifact_dir / "fixed_lookup_tables.json", metadata["static_table_registry"])
    write_json(artifact_dir / "matrix_sumcheck_proof.json", matrix)
    write_json(artifact_dir / "gkr_proof.json", gkr)
    write_json(artifact_dir / "combine_proof.json", combine)
    write_json(artifact_dir / "hyrax_mle_openings.json", openings)
    write_json(artifact_dir / "privacy_profile.json", privacy_profile)
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
        openings, privacy_profile, gpu_claim, backend, external_context, graph,
    )
    verify_elapsed_ms = (time.perf_counter() - verify_started) * 1000.0
    component_refs = {
        "FixedLookupTables": _artifact_ref(artifact_dir / "fixed_lookup_tables.json", "fixed lookup table commitments"),
        "Pi_Lasso": _artifact_ref(artifact_dir / "lookup_proof.json", "lookup proof"),
        "Pi_MatMul": _artifact_ref(artifact_dir / "matrix_sumcheck_proof.json", "matrix sumcheck proof"),
        "Pi_GKR": _artifact_ref(artifact_dir / "gkr_proof.json", "ML-friendly GKR proof"),
        "Pi_combine": _artifact_ref(artifact_dir / "combine_proof.json", "combine masked-opening proof"),
        "Open_in": _artifact_ref(artifact_dir / "hyrax_mle_openings.json", "Hyrax/MLE masked openings"),
        "PrivacyProfile": _artifact_ref(artifact_dir / "privacy_profile.json", "canonical proof privacy profile"),
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
        "PrivacyProfile": privacy_profile,
        "privacy_profile_digest": privacy_profile["profile_digest"],
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
    }
    model["id"] = digest_json(model)
    cfg = file_id(config.path)
    cfg["id"] = cfg["sha256"]
    return {"data_id": data, "model_id": model, "config_id": cfg, "dataset_name": dataset.name}
    prove_multiplication,
    verify_multiplication,
