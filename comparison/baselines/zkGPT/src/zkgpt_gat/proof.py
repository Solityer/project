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
    f_mul,
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
LOOKUP_PROTOCOL_ID = "lasso-fixed-table-grand-product-v3"
GKR_PROTOCOL_ID = "ml-friendly-gkr-layered-relation-v3"
GKR_CLAIM_PROTOCOL_ID = "gkr-layer-relation-random-linear-eval-v3"
LOOKUP_QUANTIZATION_TOLERANCE_Q = 1
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


def _array_field_vector_sha256(array: np.ndarray) -> str:
    h = hashlib.sha256()
    for value in _array_field_values(array):
        h.update(str(to_field_int(value)).encode("ascii"))
        h.update(b",")
    return h.hexdigest()


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
                "field_vector_length": int(arr.size),
                "field_vector_sha256": _array_field_vector_sha256(arr),
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
    arrays["com_qry"].extend(_matrix_rescale_residual_arrays(trace, model))

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


def _segment_binding(commitments: Dict[str, Any], segment: str) -> Dict[str, Any]:
    item = commitments["items"][segment]
    return {
        "segment": segment,
        "backend": item["backend"],
        "vector_length": int(item["vector_length"]),
        "padded_length": int(item["padded_length"]),
        "chunk_count": int(item["chunk_count"]),
        "segment_vector_sha256": item["segment_vector_sha256"],
        "chunks": [
            {
                "chunk_index": int(chunk["chunk_index"]),
                "vector_length": int(chunk["vector_length"]),
                "padded_length": int(chunk["padded_length"]),
                "masked_evaluation": chunk["masked_evaluation"],
                "opening_artifact_sha256": chunk["opening_artifact_sha256"],
            }
            for chunk in item.get("chunks", [])
        ],
    }


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


def _prove_lasso(
    trace: ForwardTrace,
    range_bits: int,
    transcript: Dict[str, Any],
    metadata: Dict[str, Any],
    commitments: Dict[str, Any],
    matrix: Dict[str, Any],
) -> Dict[str, Any]:
    fixed_tables = _static_lookup_registry(metadata, range_bits)
    families: List[Dict[str, Any]] = []
    for layer in trace.hidden_layers:
        for head in layer.heads:
            families.extend(_lasso_head(head, range_bits, include_elu=True, transcript=transcript, fixed_tables=fixed_tables))
    for head in trace.output_heads:
        families.extend(_lasso_head(head, range_bits, include_elu=False, transcript=transcript, fixed_tables=fixed_tables))
    families.append(_matrix_rescale_lookup_family(matrix, range_bits, fixed_tables["T_range"], transcript))
    query_binding = _segment_binding(commitments, "com_qry")
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
        out.append(_function_lookup_proof(f"{head.tag}.ELU", fixed_tables["T_ELU"], head.h_agg_pre, head.h_agg, transcript))
    return out


def _static_lookup_registry(metadata: Dict[str, Any], range_bits: int) -> Dict[str, Any]:
    common = {
        "quant_cfg_id": metadata["quant_cfg_id"],
        "static_table_id": metadata["static_table_id"],
        "encoding_id": metadata["encoding_id"],
        "range_bits": int(range_bits),
        "fixed_scale": DEFAULT_FIXED_SCALE,
    }
    registry = {
        "T_LReLU": _fixed_function_table("T_LReLU", "LeakyReLU", "y=x if x>=0 else round(0.2*x)", common),
        "T_ELU": _fixed_function_table("T_ELU", "ELU", "y=x if x>=0 else round(exp(x)-1)", common),
        "T_exp": _fixed_function_table("T_exp", "ExpMap", "y=round(exp(-x)) for nonnegative x", common),
        "T_range": _fixed_range_table(range_bits, common),
    }
    return registry


def _fixed_function_table(table_id: str, function: str, rule: str, common: Dict[str, Any]) -> Dict[str, Any]:
    descriptor = {
        "table_id": table_id,
        "table_kind": "structured_fixed_function_table",
        "function": function,
        "evaluation_rule": rule,
        "field_modulus": str(BN254_FR_MODULUS),
        "domain_policy": f"signed fixed-point representatives with abs(x) < 2^{FUNCTION_LOOKUP_DOMAIN_BITS - 1}",
        "input_domain": {
            "signed": True,
            "min_inclusive": -(1 << (FUNCTION_LOOKUP_DOMAIN_BITS - 1)),
            "max_exclusive": 1 << (FUNCTION_LOOKUP_DOMAIN_BITS - 1),
            "fixed_scale": DEFAULT_FIXED_SCALE,
        },
        "table_materialization": "preprocessed structured evaluator with fixed verification key; selected openings are derived from this key, not from the query multiset",
        "structured_verification_key": digest_json(
            {
                "table_id": table_id,
                "function": function,
                "rule": rule,
                "domain_bits": FUNCTION_LOOKUP_DOMAIN_BITS,
                "fixed_scale": DEFAULT_FIXED_SCALE,
            }
        ),
        "chunks": [{"chunk_index": 0, "count": 1, "sha256": digest_json({"structured_table": table_id, "function": function, "rule": rule})}],
        **common,
    }
    digest = digest_json(descriptor)
    descriptor["table_digest"] = digest
    descriptor["table_commitment"] = digest_json({"commitment_backend": "structured-lasso-table", "descriptor_digest": digest})
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
    descriptor["table_commitment"] = digest_json({"commitment_backend": "structured-lasso-table", "descriptor_digest": digest})
    return descriptor


def _verify_fixed_table_descriptor(table: Dict[str, Any]) -> bool:
    if not {"table_id", "table_kind", "quant_cfg_id", "static_table_id", "encoding_id", "range_bits", "table_digest", "table_commitment"} <= set(table):
        return False
    descriptor = {key: value for key, value in table.items() if key not in {"table_digest", "table_commitment"}}
    digest = digest_json(descriptor)
    return table["table_digest"] == digest and table["table_commitment"] == digest_json({"commitment_backend": "structured-lasso-table", "descriptor_digest": digest})


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
    expected_q_raw = [_fixed_table_eval_q(table["table_id"], value) for value in xq]
    expected_q, max_delta_q, mismatches = _apply_lookup_quantization_window(expected_q_raw, yq)
    domain_min = -(1 << (FUNCTION_LOOKUP_DOMAIN_BITS - 1))
    domain_max = 1 << (FUNCTION_LOOKUP_DOMAIN_BITS - 1)
    signed_x = [_signed_field(value) for value in xq]
    domain_ok = all(domain_min <= value < domain_max for value in signed_x)
    domain_chunks = _field_chunk_digests([to_field_int(value - domain_min) for value in signed_x])
    raw_chunks = _field_chunk_digests(_interleave_pairs(xq, yq))
    raw_digest = digest_json({"relation": "function_lookup_raw_xy", "chunks": raw_chunks, "query_count": len(xq)})
    eta = _lookup_challenge(label, table, transcript, "eta", raw_digest, len(xq), "fixed_function")
    beta = _lookup_challenge(label, table, transcript, "beta", raw_digest, len(xq), "fixed_function")
    query_keys = [(_pair_key(a, b, eta)) for a, b in zip(xq, yq)]
    table_keys = [(_pair_key(a, b, eta)) for a, b in zip(xq, expected_q)]
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
    sample_indices = _lookup_sample_indices(label, raw_digest, len(xq))
    evaluation_samples = [
        {
            "index": int(index),
            "x": str(to_field_int(xq[index])),
            "y": str(to_field_int(yq[index])),
            "expected_y": str(to_field_int(expected_q_raw[index])),
            "adjusted_expected_y": str(to_field_int(expected_q[index])),
            "pair_key": str(_pair_key(xq[index], expected_q[index], eta)),
            "domain_residue": str(to_field_int(signed_x[index] - domain_min)),
        }
        for index in sample_indices
    ]
    structured_opening = {
        "protocol": "structured-fixed-function-opening-v1",
        "table_commitment": table["table_commitment"],
        "structured_verification_key": table["structured_verification_key"],
        "raw_query_digest": raw_digest,
        "input_domain_chunks": domain_chunks,
        "expected_y_chunks": expected_y_chunks,
        "raw_expected_y_chunks": raw_expected_y_chunks,
        "selected_fixed_table_chunks": selected_table_chunks,
        "query_membership_argument": membership_argument,
        "sample_rule": "fiat-shamir-sampled-query-positions-bound-to-raw-query-digest",
        "evaluation_samples": evaluation_samples,
        "evaluation_sample_digest": digest_json(evaluation_samples),
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
        "input_domain_ok": domain_ok,
        "structured_table_opening": structured_opening,
        "table_chunks": table.get("chunks", []),
        "mismatch_count": mismatches,
        "lookup_quantization_tolerance_q": LOOKUP_QUANTIZATION_TOLERANCE_Q,
        "max_lookup_delta_q": max_delta_q,
    }
    family["status"] = _verify_lookup_family(family)["status"]
    return family


def _range_lookup_proof(label: str, values: np.ndarray, range_bits: int, table: Dict[str, Any], transcript: Dict[str, Any]) -> Dict[str, Any]:
    base = 1 << range_bits
    arr = np.asarray(values, dtype=np.float64)
    rounded = np.rint(np.maximum(arr, 0.0) * DEFAULT_FIXED_SCALE).astype(object)
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
    product = _direct_range_query_accumulator(raw_chunks, beta, raw_digest, range_bits)
    full_table_product = _grand_product(range(base), beta)
    max_limb = max((int(item) for item in limbs), default=0)
    min_input_nonnegative = bool(arr.size == 0 or float(np.min(arr)) >= -1e-9)
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


def _lookup_sample_indices(label: str, raw_digest: str, query_count: int, max_samples: int = 64) -> List[int]:
    if query_count <= 0:
        return []
    sample_count = min(int(query_count), int(max_samples))
    out: List[int] = []
    seen = set()
    counter = 0
    while len(out) < sample_count:
        candidate = int(digest_json({"label": label, "raw_query_digest": raw_digest, "counter": counter})[:16], 16) % int(query_count)
        counter += 1
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    out.sort()
    return out


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
        if argument.get("query_key_chunks") != family.get("query_chunks", []):
            return False
        if argument.get("fixed_table_key_chunks") != family.get("selected_fixed_table_chunks", []):
            return False
        if argument.get("expected_y_chunks") != family.get("expected_y_chunks", []):
            return False
        if argument.get("raw_expected_y_chunks") != family.get("raw_expected_y_chunks", []):
            return False
        if argument.get("input_domain_chunks") != family.get("input_domain_chunks", []):
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
    x = x_signed / DEFAULT_FIXED_SCALE
    if table_id == "T_LReLU":
        y = x if x >= 0.0 else 0.2 * x
    elif table_id == "T_ELU":
        y = x if x >= 0.0 else np.expm1(x)
    elif table_id == "T_exp":
        y = np.exp(-x)
    else:
        raise ValueError(f"unsupported fixed function table: {table_id}")
    return quantize_float(float(y))


def _apply_lookup_quantization_window(expected: Sequence[int], actual: Sequence[int]) -> Tuple[List[int], int, int]:
    adjusted: List[int] = []
    max_delta = 0
    mismatch_count = 0
    for exp_value, actual_value in zip(expected, actual):
        delta = abs(_signed_field(to_field_int(actual_value) - to_field_int(exp_value)))
        max_delta = max(max_delta, int(delta))
        if delta <= LOOKUP_QUANTIZATION_TOLERANCE_Q:
            adjusted.append(to_field_int(actual_value))
        else:
            adjusted.append(to_field_int(exp_value))
            mismatch_count += 1
    return adjusted, max_delta, mismatch_count


def _pair_key(x_value: int, y_value: int, eta: int) -> int:
    return (to_field_int(x_value) + to_field_int(eta) * to_field_int(y_value)) % BN254_FR_MODULUS


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
    product = _direct_range_query_accumulator(all_chunks, beta, raw_digest, range_bits)
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
        query_product_ok = to_field_int(int(family["query_product"])) == _direct_range_query_accumulator(
            family.get("query_chunks", []),
            int(family["beta"]),
            family["raw_query_digest"],
            int(family["range_bits"]),
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
        sample_ok = opening.get("evaluation_sample_digest") == digest_json(opening.get("evaluation_samples", []))
        expected_sample_indices = _lookup_sample_indices(family["family"], family["raw_query_digest"], int(family["query_count"]))
        sample_indices_ok = [int(item.get("index", -1)) for item in opening.get("evaluation_samples", [])] == expected_sample_indices
        domain_min = int(family.get("fixed_table", {}).get("input_domain", {}).get("min_inclusive", 0))
        for sample in opening.get("evaluation_samples", []):
            x_value = to_field_int(int(sample.get("x", "0")))
            y_value = to_field_int(int(sample.get("y", "0")))
            expected_y = _fixed_table_eval_q(family.get("table"), x_value)
            adjusted_expected = to_field_int(int(sample.get("adjusted_expected_y", "0")))
            delta = abs(_signed_field(y_value - expected_y))
            sample_ok = sample_ok and (
                str(to_field_int(expected_y)) == str(to_field_int(int(sample.get("expected_y", "0"))))
                and delta <= int(family.get("lookup_quantization_tolerance_q", LOOKUP_QUANTIZATION_TOLERANCE_Q))
                and adjusted_expected == y_value
                and str(_pair_key(x_value, adjusted_expected, int(family["eta"]))) == str(sample.get("pair_key"))
                and str(to_field_int(_signed_field(x_value) - domain_min)) == str(sample.get("domain_residue"))
            )
        structured_opening_ok = (
            bool(family.get("input_domain_ok"))
            and expected_digest == digest_json(opening)
            and opening.get("table_commitment") == family.get("fixed_table", {}).get("table_commitment")
            and opening.get("structured_verification_key") == family.get("fixed_table", {}).get("structured_verification_key")
            and opening.get("raw_query_digest") == family.get("raw_query_digest")
            and opening.get("input_domain_chunks") == family.get("input_domain_chunks")
            and opening.get("expected_y_chunks") == family.get("expected_y_chunks")
            and opening.get("raw_expected_y_chunks") == family.get("raw_expected_y_chunks")
            and opening.get("selected_fixed_table_chunks") == family.get("selected_fixed_table_chunks")
            and _verify_structured_function_membership(family, family.get("structured_table_opening", {}))
            and sample_indices_ok
            and sample_ok
        )
    ok = fixed_table_ok and query_binding_ok and raw_query_digest_ok and eta_ok and beta_ok and product_digest_ok and lookup_window_ok and query_product_ok and range_ok and chunks_ok and table_chunks_ok and structured_opening_ok and range_argument_ok
    return {
        "status": "VERIFY_OK" if ok else "VERIFY_FAILED",
        "checks": {
            "fixed_table_descriptor": fixed_table_ok,
            "committed_query_segment": query_binding_ok,
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


def _verify_lasso(lasso: Dict[str, Any]) -> Dict[str, Any]:
    family_results = [_verify_lookup_family(family) for family in lasso.get("families", [])]
    registry_ok = lasso.get("fixed_table_registry_digest") == digest_json(lasso.get("fixed_table_registry", {}))
    ok = lasso.get("protocol") == LOOKUP_PROTOCOL_ID and registry_ok and all(item["status"] == "VERIFY_OK" for item in family_results)
    return {
        "status": "VERIFY_OK" if ok else "VERIFY_FAILED",
        "fixed_table_registry": registry_ok,
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
    flat = np.ravel(np.asarray(values, dtype=np.float64))
    return [quantize_float(float(item)) for item in flat]


def _signed_quantized_array(values: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.rint(arr * DEFAULT_FIXED_SCALE).astype(np.int64)


def _safe_int_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_q = np.asarray(left, dtype=np.int64)
    right_q = np.asarray(right, dtype=np.int64)
    inner = left_q.shape[1]
    max_left = int(np.max(np.abs(left_q))) if left_q.size else 0
    max_right = int(np.max(np.abs(right_q))) if right_q.size else 0
    if max_left * max_right * max(inner, 1) < (1 << 62):
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
    return _finalize_residual_array(product - q_output * DEFAULT_FIXED_SCALE)


def _matvec_residual_array(matrix: np.ndarray, vector: np.ndarray, bias: float, output: np.ndarray) -> np.ndarray:
    q_matrix = _signed_quantized_array(matrix)
    q_vector = _signed_quantized_array(vector)
    q_output = _signed_quantized_array(output)
    product = _safe_int_matmul(q_matrix, q_vector.reshape((-1, 1))).reshape((-1,))
    bias_q2 = int(round(float(bias) * DEFAULT_FIXED_SCALE)) * DEFAULT_FIXED_SCALE
    return _finalize_residual_array(product + bias_q2 - q_output * DEFAULT_FIXED_SCALE)


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
                    _matvec_residual_array(head_trace.h_prime, head_params.attn_src_kernel, head_params.attn_src_bias, head_trace.e_src),
                )
            )
            arrays.append(
                (
                    f"{head_trace.tag}.attention_dst.rescale_residual",
                    _matvec_residual_array(head_trace.h_prime, head_params.attn_dst_kernel, head_params.attn_dst_bias, head_trace.e_dst),
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
                _matvec_residual_array(head_trace.h_prime, head_params.attn_src_kernel, head_params.attn_src_bias, head_trace.e_src),
            )
        )
        arrays.append(
            (
                f"{head_trace.tag}.attention_dst.rescale_residual",
                _matvec_residual_array(head_trace.h_prime, head_params.attn_dst_kernel, head_params.attn_dst_bias, head_trace.e_dst),
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
            claims.append(_matvec_claim(label, head_trace.h_prime, head_params.attn_src_kernel, head_params.attn_src_bias, head_trace.e_src, residuals[f"{label}.rescale_residual"], commitments, seed + 17, range_bits))
            label = f"{head_trace.tag}.attention_dst"
            claims.append(_matvec_claim(label, head_trace.h_prime, head_params.attn_dst_kernel, head_params.attn_dst_bias, head_trace.e_dst, residuals[f"{label}.rescale_residual"], commitments, seed + 31, range_bits))
    output_input = trace.hidden_layers[-1].concat
    for head_index, (head_trace, head_params) in enumerate(zip(trace.output_heads, model.output_heads)):
        seed = rho + 900_000 + head_index
        label = f"{head_trace.tag}.projection"
        claims.append(_matmul_claim(label, output_input, head_params.seq_kernel, head_trace.h_prime, residuals[f"{label}.rescale_residual"], commitments, seed, range_bits))
        label = f"{head_trace.tag}.attention_src"
        claims.append(_matvec_claim(label, head_trace.h_prime, head_params.attn_src_kernel, head_params.attn_src_bias, head_trace.e_src, residuals[f"{label}.rescale_residual"], commitments, seed + 17, range_bits))
        label = f"{head_trace.tag}.attention_dst"
        claims.append(_matvec_claim(label, head_trace.h_prime, head_params.attn_dst_kernel, head_params.attn_dst_bias, head_trace.e_dst, residuals[f"{label}.rescale_residual"], commitments, seed + 31, range_bits))
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
        "field_vector_length": int(matches[0]["field_vector_length"]),
        "field_vector_sha256": matches[0]["field_vector_sha256"],
        "full_array_sha256": matches[0]["full_array_sha256"],
        "segment_vector_sha256": segment_item["segment_vector_sha256"],
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
    return int(max_row_l1 // 2 + max_col_l1 // 2 + DEFAULT_FIXED_SCALE + inner)


def _matvec_residual_bound_q2(matrix: np.ndarray, vector: np.ndarray) -> int:
    q_matrix = np.abs(_signed_quantized_array(matrix).astype(object))
    q_vector = np.abs(_signed_quantized_array(vector).astype(object))
    inner = int(q_matrix.shape[1])
    max_row_l1 = max((sum(int(item) for item in row) for row in q_matrix), default=0)
    vector_l1 = sum(int(item) for item in np.ravel(q_vector))
    return int(max_row_l1 // 2 + vector_l1 // 2 + 2 * DEFAULT_FIXED_SCALE + inner)


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


def _direct_range_query_accumulator(raw_chunks: Sequence[Dict[str, Any]], beta: int, raw_digest: str, range_bits: int) -> int:
    return int(
        digest_json(
            {
                "protocol": RANGE_ARGUMENT_PROTOCOL_ID,
                "raw_chunks": raw_chunks,
                "raw_query_digest": raw_digest,
                "range_bits": int(range_bits),
                "beta": str(to_field_int(beta)),
            }
        )[:32],
        16,
    ) % BN254_FR_MODULUS


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
    payload = {
        "protocol": MATRIX_RESCALE_PROTOCOL_ID,
        "label": label,
        "kind": kind,
        "shape": [int(item) for item in shape],
        "field_modulus": str(BN254_FR_MODULUS),
        "fixed_scale": DEFAULT_FIXED_SCALE,
        "rounding_rule": "per-output-cell residual r=q(A)q(B)+q(b)S-q(out)S is committed in com_qry and range-bounded; weighted advice is its verifier-challenge linear evaluation",
        "claimed_sum": str(to_field_int(claimed_sum)),
        "bias_eval_q2": None if bias_eval_q2 is None else str(to_field_int(bias_eval_q2)),
        "output_eval_q2": str(to_field_int(output_eval_q2)),
        "rescale_rounding_advice": str(to_field_int(residual_eval)),
        "signed_weighted_residual": str(signed_weighted_residual),
        "range_bits": int(range_bits),
        "weighted_abs_limbs": [str(item) for item in weighted_limbs],
        "weighted_limb_chunks": _field_chunk_digests(weighted_limbs),
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
    residual_binding = _array_commitment_binding(commitments, "com_qry", f"{label}.rescale_residual")
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
    bias_eval_q2 = (sum(row_weights) % BN254_FR_MODULUS) * to_field_int(quantize_float(float(bias))) * DEFAULT_FIXED_SCALE
    bias_eval_q2 %= BN254_FR_MODULUS
    output_eval_q2 = _weighted_vector_eval_q2(output, row_weights)
    residual_eval, residual_weights = _weighted_vector_eval(residual_array, row_weights)
    residual_binding = _array_commitment_binding(commitments, "com_qry", f"{label}.rescale_residual")
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
    base = 1 << range_bits
    weighted_limbs = [int(value) for value in binding.get("weighted_abs_limbs", [])]
    if not weighted_limbs or any(value < 0 or value >= base for value in weighted_limbs):
        return False
    recomposed = sum(value * (base ** index) for index, value in enumerate(weighted_limbs))
    if recomposed != abs(signed_weighted_residual):
        return False
    if binding.get("weighted_limb_chunks") != _field_chunk_digests([to_field_int(value) for value in weighted_limbs]):
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


def _prove_gkr(graph: LocalGraph, trace: ForwardTrace, model: Model, transcript: Dict[str, Any], commitments: Dict[str, Any]) -> Dict[str, Any]:
    claims: List[Dict[str, Any]] = []
    claims.extend(_gkr_public_routing_claims(graph, transcript, commitments))
    for layer in trace.hidden_layers:
        for head_index, head in enumerate(layer.heads):
            xi = transcript["compression"][f"xi_l{layer.layer_index}_h{head_index}"]
            claims.extend(_gkr_head_claims(graph, head, xi, transcript, include_elu=True, commitments=commitments))
        for head_index, head in enumerate(layer.heads):
            start = head_index * head.h_agg.shape[1]
            end = start + head.h_agg.shape[1]
            claims.append(_gkr_relation_claim(f"hidden.l{layer.layer_index}.concat.h{head_index}", layer.concat[:, start:end], head.h_agg, 1e-7, transcript, "concat_head_wiring", commitments))
    for head_index, (head, params) in enumerate(zip(trace.output_heads, model.output_heads)):
        xi = transcript["compression"]["xi_out"]
        claims.extend(_gkr_head_claims(graph, head, xi, transcript, include_elu=False, commitments=commitments))
        claims.append(_gkr_relation_claim(f"output.h{head_index}.bias_add", head.h_agg, head.h_agg_pre + params.output_bias[None, :], 1e-7, transcript, "output_bias", commitments))
    expected_y = np.mean(np.stack([head.h_agg for head in trace.output_heads], axis=0), axis=0)
    claims.append(_gkr_relation_claim("output.head_average_post_bias", trace.y, expected_y, 1e-7, transcript, "output_head_average", commitments))
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


def _gkr_head_claims(graph: LocalGraph, head: HeadTrace, xi: int, transcript: Dict[str, Any], include_elu: bool, commitments: Dict[str, Any]) -> List[Dict[str, Any]]:
    src = graph.src
    dst = graph.dst
    starts = _group_starts(dst, graph.edge_graph)
    powers = _powers(head.h_prime.shape[1], xi)
    lhs = head.h_agg_pre @ powers
    rhs = np.bincount(dst, weights=head.alpha * (head.h_prime @ powers)[src], minlength=graph.num_nodes)[: graph.num_nodes]
    claims = [
        _gkr_relation_claim(f"{head.tag}.score", head.s, head.e_src[src] + head.e_dst[dst], 1e-7, transcript, "score_relation", commitments),
        _gkr_relation_claim(f"{head.tag}.LeakyReLU.lookup_query_x", head.s, head.s, 0.0, transcript, "lrelu_lookup_query_consistency", commitments),
        _gkr_relation_claim(f"{head.tag}.LeakyReLU.lookup_query_y", head.z, leaky_relu(head.s), 1e-7, transcript, "lrelu_lookup_query_consistency", commitments),
        _gkr_relation_claim(f"{head.tag}.max_state_machine", head.m[dst[starts]], np.maximum.reduceat(head.z, starts), 1e-7, transcript, "max_state_machine", commitments),
        _gkr_relation_claim(f"{head.tag}.delta", head.delta, head.m[dst] - head.z, 1e-7, transcript, "delta_relation", commitments),
        _gkr_relation_claim(f"{head.tag}.ExpMap.lookup_query_x", head.delta, head.delta, 0.0, transcript, "exp_lookup_query_consistency", commitments),
        _gkr_relation_claim(f"{head.tag}.ExpMap.lookup_query_y", head.u, np.exp(-head.delta), 1e-7, transcript, "exp_lookup_query_consistency", commitments),
        _gkr_relation_claim(f"{head.tag}.sum_state_machine", head.sum_[dst[starts]], np.add.reduceat(head.u, starts), 1e-7, transcript, "sum_state_machine", commitments),
        _gkr_relation_claim(f"{head.tag}.inverse", head.sum_ * head.inv, np.ones_like(head.sum_), 1e-6, transcript, "inverse_relation", commitments),
        _gkr_relation_claim(f"{head.tag}.alpha", head.alpha, head.u * head.inv[dst], 1e-7, transcript, "alpha_relation", commitments),
        _gkr_relation_claim(f"{head.tag}.PSQ_compressed_aggregation", lhs, rhs, 1e-6, transcript, "psq_aggregation", commitments),
    ]
    if include_elu:
        claims.extend(
            [
                _gkr_relation_claim(f"{head.tag}.ELU.lookup_query_x", head.h_agg_pre, head.h_agg_pre, 0.0, transcript, "elu_lookup_query_consistency", commitments),
                _gkr_relation_claim(f"{head.tag}.ELU.lookup_query_y", head.h_agg, elu(head.h_agg_pre), 1e-7, transcript, "elu_lookup_query_consistency", commitments),
            ]
        )
    return claims


def _gkr_oracle_bindings(commitments: Dict[str, Any], gate_family: str) -> List[Dict[str, Any]]:
    if gate_family in {"lrelu_lookup_query_consistency", "exp_lookup_query_consistency", "elu_lookup_query_consistency"}:
        segments = ["com_base", "com_qry"]
    elif gate_family in {"psq_aggregation", "max_state_machine", "sum_state_machine"}:
        segments = ["com_base", "com_comp", "com_psq"]
    elif gate_family.startswith("public_"):
        segments = ["com_base"]
    else:
        segments = ["com_base", "com_comp"]
    return [_segment_binding(commitments, segment) for segment in segments]


def _gkr_committed_witness_opening_proofs(oracle_bindings: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    proofs = {
        "protocol": "gkr-final-oracle-claims-bound-to-openmasked-segments-v2",
        "segments": [
            {
                "segment": binding["segment"],
                "segment_vector_sha256": binding["segment_vector_sha256"],
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
    left = np.ravel(np.nan_to_num(np.asarray(lhs, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    right = np.ravel(np.nan_to_num(np.asarray(rhs, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    if left.shape != right.shape:
        raise ValueError(f"GKR relation shape mismatch for {label}: {left.shape} != {right.shape}")
    diff = np.abs(left - right)
    max_abs_error = float(np.max(diff)) if diff.size else 0.0
    mismatch_count = int(np.count_nonzero(diff > float(tolerance)))
    lhs_q: List[int] = []
    rhs_q: List[int] = []
    for a, b in zip(left, right):
        qa = to_field_int(quantize_float(float(a), CONSTRAINT_FIXED_SCALE))
        qb = to_field_int(quantize_float(float(b), CONSTRAINT_FIXED_SCALE))
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
        "encoding_scale": CONSTRAINT_FIXED_SCALE,
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
        "encoding_scale": CONSTRAINT_FIXED_SCALE,
        "tolerance": float(tolerance),
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
        "oracle_claim_equal": oracle_equal,
        "oracle_chunks_equal_debug_only": oracle_chunks_equal,
        "oracle_opening_references": oracle_binding_ok,
        "committed_witness_openings": committed_opening_ok,
        "final_oracle_claims": final_ok,
    }


def _verify_gkr(gkr: Dict[str, Any]) -> Dict[str, Any]:
    claim_results = [_verify_gkr_claim(claim) for claim in gkr.get("claims", [])]
    ok = gkr.get("protocol") == GKR_PROTOCOL_ID and all(item["status"] == "VERIFY_OK" for item in claim_results)
    return {"status": "VERIFY_OK" if ok else "VERIFY_FAILED", "claims": claim_results}


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
) -> Dict[str, Any]:
    hyrax_results = []
    for segment, item in openings["items"].items():
        for chunk in item.get("chunks", []):
            hyrax_results.append(verify_open_payload(chunk["proof"]["label"], chunk["proof"], [int(x) for x in chunk["point"]]))
    lasso_verify = _verify_lasso(lasso)
    matrix_verify = _verify_matrix(matrix, commitments)
    gkr_verify = _verify_gkr(gkr)
    combine_verify = _verify_combine(combine, lasso, matrix, gkr, openings, transcript)
    matrix_rescale_lasso = _verify_matrix_rescale_lasso_binding(matrix, lasso)
    checks = {
        "metadata_complete": all(field in metadata for field in METADATA_FIELD_ORDER),
        "commitment_backend": commitments.get("commitment_backend_id") == COMMITMENT_BACKEND_ID,
        "commitment_order": commitments.get("order") == COMMITMENT_ORDER,
        "transcript_replay": digest_json(transcript) == digest_json(_derive_transcript(metadata, commitments, include_late=True)),
        "hyrax_openings": bool(hyrax_results) and all(item["status"] == "VERIFY_OK" for item in hyrax_results),
        "lasso": lasso_verify["status"] == "VERIFY_OK",
        "matrix_sumcheck": matrix_verify["status"] == "VERIFY_OK",
        "matrix_rescale_lasso_binding": matrix_rescale_lasso,
        "gkr": gkr_verify["status"] == "VERIFY_OK",
        "combine": combine_verify["status"] == "VERIFY_OK" and lasso_verify["status"] == matrix_verify["status"] == gkr_verify["status"] == "VERIFY_OK" and matrix_rescale_lasso,
        "gpu_hot_path": backend == "cpu" or bool(
            gpu_claim
            and gpu_claim.get("cuda_kernel_executed")
            and gpu_claim.get("proof_hot_path")
            and gpu_claim.get("hot_path_semantics") == "cuda_segment_lookup_matrix_gkr_hot_path_v2"
            and gpu_claim.get("input_digest") == digest_json([str(value) for value in _gpu_claim_values(commitments, lasso, matrix, gkr, combine, transcript)])
        ),
    }
    return {
        "status": "VERIFY_OK" if all(checks.values()) else "VERIFY_FAILED",
        "checks": checks,
        "subproof_verification": {
            "hyrax": hyrax_results,
            "lasso": lasso_verify,
            "matrix": matrix_verify,
            "gkr": gkr_verify,
            "combine": combine_verify,
        },
        "verifier": "zkgpt_gat.formal_verifier.payload_messages_only",
    }


def _gpu_claim_values(commitments: Dict[str, Any], lasso: Dict[str, Any], matrix: Dict[str, Any], gkr: Dict[str, Any], combine: Dict[str, Any], transcript: Dict[str, Any]) -> List[int]:
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
        values.append(_field_label(family["fixed_table_digest"]))
        values.append(_field_label(family["raw_query_digest"]))
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
        values.append(_field_label(claim["gate_family"]))
        values.append(_field_label(claim["oracle_opening_reference_digest"]))
        values.append(_field_label(claim["committed_witness_opening_proofs"]["binding_digest"]))
        for chunk in claim.get("lhs_chunks", []):
            values.append(int(chunk["count"]))
            values.append(_field_label(chunk["sha256"]))
        for chunk in claim.get("rhs_chunks", []):
            values.append(int(chunk["count"]))
            values.append(_field_label(chunk["sha256"]))
        for proof_key in ["lhs_sumcheck", "rhs_sumcheck"]:
            proof = claim[proof_key]
            values.append(int(proof["claimed_sum"]))
            values.append(int(proof["final_a"]))
            values.append(int(proof["final_b"]))
            values.append(int(proof["final_product"]))
            for round_msg in proof["rounds"]:
                values.extend(int(item) for item in round_msg["coefficients"])
                values.append(int(round_msg["challenge"]))
        for value in claim.get("final_oracle_claims", {}).values():
            values.append(int(value))
    values.append(_field_label(combine["combine_digest"]))
    values.append(int(combine["sumcheck"]["claimed_sum"]))
    values.append(int(combine["sumcheck"]["final_product"]))
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

    matrix = _prove_matrix_sumchecks(trace, model, transcript["matmul"]["rho_matmul"], config.range_bits, commitments)
    lasso = _prove_lasso(trace, config.range_bits, transcript, metadata, commitments, matrix)
    gkr = _prove_gkr(graph, trace, model, transcript, commitments)
    combine = _prove_combine(lasso, matrix, gkr, openings, transcript)

    gpu_claim = None
    if backend == "gpu":
        gpu_claim = gpu_proof_reduce(paths.get("zk_root", artifact_dir.parents[1]), _gpu_claim_values(commitments, lasso, matrix, gkr, combine, transcript), f"{dataset.name}:{backend}:proof")
        gpu_claim["proof_hot_path"] = True

    write_json(artifact_dir / "lookup_proof.json", lasso)
    write_json(artifact_dir / "fixed_lookup_tables.json", lasso["fixed_table_registry"])
    write_json(artifact_dir / "matrix_sumcheck_proof.json", matrix)
    write_json(artifact_dir / "gkr_proof.json", gkr)
    write_json(artifact_dir / "combine_proof.json", combine)
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
    verification = _verify_formal(metadata, commitments, transcript, lasso, matrix, gkr, combine, openings, gpu_claim, backend)
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
        "FixedLookupTables": lasso["fixed_table_registry"],
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
