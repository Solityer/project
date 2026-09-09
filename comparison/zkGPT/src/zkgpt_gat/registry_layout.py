"""Descriptor-only model-registration layout derivation.

The formal prover derives ``Layout_in`` from the complete forward trace.  A
one-time model registration only commits the model weights, however, and must
not execute or retain that trace merely to compute the binding digest.  This
module derives the same descriptor geometry from graph/model shapes and the
authenticated integer-bound ledger.  It never supplies values to the proof
path and is therefore suitable only for ``register_model.py``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .data import GraphDataset, LocalGraph
from .field import next_power_of_two
from .layout import ArrayShapeList, build_input_layout_from_shapes
from .model import Model
from .proof import (
    ENCODING_ID,
    _fixed_table_eval_q,
    _integer_bound_ledger,
    _model_arrays,
    _pair_key,
    _static_lookup_registry,
)
from .resources import validate_static_table_descriptor
from .util import digest_json


ShapeArrayList = List[Tuple[str, Tuple[int, ...]]]


def _shape_length(shape: Sequence[int]) -> int:
    length = 1
    for dimension in shape:
        length *= int(dimension)
    return length


def _head_aux_shapes(tag: str, node_count: int, edge_count: int, width: int, output: bool) -> ShapeArrayList:
    """Return the exact ordered shape list emitted by ``_segment_arrays``."""
    n = int(node_count)
    e = int(edge_count)
    d = int(width)
    scalar_n = (n,)
    scalar_e = (e,)
    matrix_n = (n, d)
    matrix_e = (e, d)
    shapes: ShapeArrayList = [
        (f"{tag}.H_prime", matrix_n),
        (f"{tag}.E_src", scalar_n),
        (f"{tag}.E_dst", scalar_n),
        (f"{tag}.S", scalar_e),
        (f"{tag}.Z", scalar_e),
        (f"{tag}.M", scalar_n),
        (f"{tag}.Delta", scalar_e),
        (f"{tag}.U", scalar_e),
        (f"{tag}.Sum", scalar_n),
        (f"{tag}.Inv", scalar_n),
        (f"{tag}.Alpha", scalar_e),
        (f"{tag}.H_agg_pre", matrix_n),
        (f"{tag}.H_agg", matrix_n),
        (f"{tag}.H_or_Y_agg_num", matrix_n),
        (f"{tag}.SignedRoundDiv.output", matrix_n),
        (f"{tag}.field_aggregate", matrix_n),
        (f"{tag}.SignedRoundDiv.denominator", scalar_n),
        (f"{tag}.Q_grp", scalar_e),
        (f"{tag}.group_count", scalar_n),
        (f"{tag}.s_max", scalar_e),
        (f"{tag}.R_U", scalar_e),
        (f"{tag}.R_num", matrix_e),
        (f"{tag}.R_field", matrix_e),
    ]
    for name in ("x", "d", "y", "sigma", "z", "a", "a_inv", "d_inv", "q", "r", "delta"):
        shapes.append((f"{tag}.SignedRoundDiv.aggregate.{name}", matrix_n))
    if output:
        for name in ("x", "d", "y", "sigma", "z", "a", "a_inv", "d_inv", "q", "r", "delta"):
            shapes.append((f"{tag}.SignedRoundDiv.scale_alignment.{name}", matrix_n))
        shapes.append((f"{tag}.Y_pre_aligned", matrix_n))
    for name in (
        "gkr.sum_inv",
        "gkr.s_max_delta",
        "gkr.double_aggregate",
    ):
        shape = scalar_n if name == "gkr.sum_inv" else (scalar_e if name == "gkr.s_max_delta" else matrix_n)
        shapes.append((f"{tag}.{name}", shape))
    for name in (
        "sigma_a", "sigma_q", "a_a_inv", "d_d_inv", "d_q", "z_a", "z_sigma", "z_a_inv"
    ):
        shapes.append((f"{tag}.SignedRoundDiv.aggregate.gkr.{name}", matrix_n))
    if output:
        for name in (
            "sigma_a", "sigma_q", "a_a_inv", "d_d_inv", "d_q", "z_a", "z_sigma", "z_a_inv"
        ):
            shapes.append((f"{tag}.SignedRoundDiv.scale_alignment.gkr.{name}", matrix_n))
    for name, shape in (
        ("one_minus_Q", scalar_e),
        ("R_U_prev", scalar_e),
        ("R_U_carry", scalar_e),
        ("R_smax", scalar_e),
        ("R_smax_prev", scalar_e),
        ("R_smax_carry", scalar_e),
        ("R_num_prev", matrix_e),
        ("R_num_carry", matrix_e),
        ("U_H", matrix_e),
        ("R_field_prev", matrix_e),
        ("R_field_carry", matrix_e),
        ("Alpha_H", matrix_e),
    ):
        shapes.append((f"{tag}.gkr.{name}", shape))
    return shapes


def _append_function_family(
    arrays: ShapeArrayList,
    registry: List[Dict[str, Any]],
    label: str,
    table_name: str,
    real_count: int,
    table: Mapping[str, Any],
    offset: int,
) -> int:
    real_count = int(real_count)
    padded_count = next_power_of_two(max(2, real_count))
    domain_min = int(table["certified_domain"][0])
    a_pad = _pair_key(
        table,
        domain_min,
        _fixed_table_eval_q(
            table_name,
            domain_min,
            table.get("authenticated_quant_scales"),
        ),
    )
    arrays.append((label, (padded_count,)))
    registry.append(
        {
            "family": label,
            "kind": "function",
            "table_id": table_name,
            "offset": int(offset),
            "real_count": real_count,
            "padded_count": padded_count,
            "a_pad": str(a_pad),
            "Q_pad": {"zero_prefix": real_count, "one_suffix": padded_count - real_count},
            "tuple_encoding": dict(table["tuple_encoding"]),
        }
    )
    return int(offset) + padded_count


def _append_range_family(
    arrays: ShapeArrayList,
    registry: List[Dict[str, Any]],
    label: str,
    value_count: int,
    policy: Mapping[str, Any],
    range_bits: int,
    offset: int,
    *,
    signed: bool = False,
) -> int:
    limb_count = int(policy.get("limb_count", 0))
    bit_width = int(policy.get("bit_width", 0))
    encoding = str(policy.get("encoding", "magnitude"))
    encoding_offset = int(policy.get("encoding_offset", 0))
    if limb_count <= 0 or bit_width <= 0:
        raise ValueError(f"invalid canonical range-family policy: {label}")
    if encoding not in {"magnitude", "offset"}:
        raise ValueError(f"invalid range-family encoding: {label}")
    if (encoding == "magnitude" and encoding_offset != 0) or (
        encoding == "offset" and encoding_offset <= 0
    ):
        raise ValueError(f"invalid range-family encoding offset: {label}")
    expected_signed = encoding == "offset" or label == "rescale_rounding_range"
    if bool(signed) != expected_signed:
        raise ValueError(f"range-family signedness does not match encoding: {label}")
    real_count = int(value_count) * limb_count
    padded_count = next_power_of_two(max(2, real_count))
    arrays.append((label, (padded_count,)))
    registry.append(
        {
            "family": label,
            "kind": "range",
            "table_id": "T_range",
            "offset": int(offset),
            "real_count": real_count,
            "padded_count": padded_count,
            "a_pad": "0",
            "Q_pad": {"zero_prefix": real_count, "one_suffix": padded_count - real_count},
            "range_bits": int(range_bits),
            "bit_width": bit_width,
            "limbs_per_value": limb_count,
            "bound": int(policy.get("bound", -1)),
            "signed_magnitude": bool(signed),
            "encoded_bound": int(policy.get("encoded_bound", policy.get("bound", -1))),
            "encoding": encoding,
            "encoding_offset": encoding_offset,
        }
    )
    return int(offset) + padded_count


def _shape_only_lasso(
    model: Model,
    graph: LocalGraph,
    range_bits: int,
    tables: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> Tuple[ShapeArrayList, List[Dict[str, Any]]]:
    arrays: ShapeArrayList = []
    registry: List[Dict[str, Any]] = []
    offset = 0
    n = int(graph.num_nodes)
    e = int(graph.src.size)
    family_policy = ledger.get("range_family_policy", {}).get("families", {})

    def add_range(label: str, value_count: int, *, signed: bool = False) -> None:
        nonlocal offset
        policy = family_policy.get(label)
        if not isinstance(policy, Mapping):
            raise ValueError(f"missing canonical range-family policy: {label}")
        if int(policy.get("range_bits", -1)) != int(range_bits):
            raise ValueError(f"range-family bit width differs from authenticated ledger: {label}")
        offset = _append_range_family(
            arrays, registry, label, value_count, policy, range_bits, offset, signed=signed
        )

    def add_function(label: str, table_name: str, real_count: int) -> None:
        nonlocal offset
        table = tables[table_name]
        offset = _append_function_family(arrays, registry, label, table_name, real_count, table, offset)

    def add_head(tag: str, width: int, output: bool) -> None:
        add_range(f"{tag}.range_delta", e)
        add_range(f"{tag}.M_range", n, signed=True)
        add_function(f"{tag}.LeakyReLU", "T_LReLU", e)
        add_range(f"{tag}.LeakyReLU.x_range", e, signed=True)
        add_range(f"{tag}.LeakyReLU.y_range", e, signed=True)
        add_function(f"{tag}.ExpMap", "T_exp", e)
        add_range(f"{tag}.ExpMap.x_range", e)
        add_range(f"{tag}.ExpMap.y_range", e)
        if not output:
            add_function(f"{tag}.ELU", "T_ELU", n * int(width))
            add_range(f"{tag}.ELU.x_range", n * int(width), signed=True)
            add_range(f"{tag}.ELU.y_range", n * int(width), signed=True)
        add_range(f"{tag}.SignedRoundDiv.aggregate.range", n * int(width) * 5, signed=True)
        if output:
            add_range(f"{tag}.SignedRoundDiv.scale_alignment.range", n * int(width) * 5, signed=True)

    for layer in model.hidden_layers:
        for head_index in range(int(layer.head_count)):
            add_head(f"hidden.l{int(layer.layer_index) + 1}.h{head_index}", int(layer.head_dim), False)
    for head_index in range(int(model.k_out)):
        add_head(f"output.h{head_index}", int(model.class_count), True)
    add_range("output.final_average.SignedRoundDiv.range", n * int(model.class_count) * 5, signed=True)
    residual_count = 0
    for layer in model.hidden_layers:
        for _head in layer.heads:
            residual_count += n * (int(layer.head_dim) + 2)
    residual_count += sum(n * (int(head.output_bias.size) + 2) for head in model.output_heads)
    add_range("rescale_rounding_range", residual_count, signed=True)
    return arrays, registry


def _shape_only_aux(model: Model, graph: LocalGraph) -> ShapeArrayList:
    n = int(graph.num_nodes)
    e = int(graph.src.size)
    arrays: ShapeArrayList = []
    for layer in model.hidden_layers:
        arrays.append((f"hidden.l{int(layer.layer_index) + 1}.concat", (n, int(layer.head_count) * int(layer.head_dim))))
        for head_index, _head in enumerate(layer.heads):
            tag = f"hidden.l{int(layer.layer_index) + 1}.h{head_index}"
            arrays.extend(_head_aux_shapes(tag, n, e, int(layer.head_dim), False))
    for head_index, head in enumerate(model.output_heads):
        arrays.extend(_head_aux_shapes(f"output.h{head_index}", n, e, int(head.output_bias.size), True))

    residual_arrays: ShapeArrayList = []
    for layer in model.hidden_layers:
        for head_index, _head in enumerate(layer.heads):
            tag = f"hidden.l{int(layer.layer_index) + 1}.h{head_index}"
            residual_arrays.extend(
                [
                    (f"{tag}.projection.rescale_residual", (n, int(layer.head_dim))),
                    (f"{tag}.attention_src.rescale_residual", (n,)),
                    (f"{tag}.attention_dst.rescale_residual", (n,)),
                ]
            )
    for head_index, head in enumerate(model.output_heads):
        tag = f"output.h{head_index}"
        residual_arrays.extend(
            [
                (f"{tag}.projection.rescale_residual", (n, int(head.output_bias.size))),
                (f"{tag}.attention_src.rescale_residual", (n,)),
                (f"{tag}.attention_dst.rescale_residual", (n,)),
            ]
        )
    arrays.extend(residual_arrays)
    residual_count = sum(_shape_length(shape) for _, shape in residual_arrays)
    arrays.extend(
        [
            ("rescale_rounding_range.gkr.value", (residual_count,)),
            ("rescale_rounding_range.gkr.sign", (residual_count,)),
            ("rescale_rounding_range.gkr.abs", (residual_count,)),
            ("rescale_rounding_range.gkr.sign_abs", (residual_count,)),
        ]
    )
    for name in ("x", "d", "y", "sigma", "z", "a", "a_inv", "d_inv", "q", "r", "delta"):
        arrays.append((f"output.final_average.{name}", (n, int(model.class_count))))
    for name in (
        "gkr.sigma_a", "gkr.sigma_q", "gkr.a_a_inv", "gkr.d_d_inv",
        "gkr.d_q", "gkr.z_a", "gkr.z_sigma", "gkr.z_a_inv",
    ):
        arrays.append((f"output.final_average.{name}", (n, int(model.class_count))))
    arrays.extend(
        [
            ("output.Y_sum", (n, int(model.class_count))),
            ("output.Y", (n, int(model.class_count))),
        ]
    )
    return arrays


def build_model_registry_layout(config: Any, dataset: GraphDataset, graph: LocalGraph, model: Model) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return the proof-equivalent layout and ledger without a forward trace."""
    descriptor = validate_static_table_descriptor(config.project_root)
    ledger = _integer_bound_ledger(
        {"rounding_mode": config.rounding_mode, "dim_cfg": {"B": config.range_bits}},
        model,
        graph,
    )
    provisional = {
        "quant_cfg_id": model.quant_cfg_id,
        "rounding_mode": config.rounding_mode,
        "static_table_id": model.static_table_id,
        "static_table_descriptor_schema": descriptor["schema"],
        "static_table_content_digest": descriptor["content_digest"],
        "static_table_descriptor_digest": descriptor["descriptor_digest"],
        "encoding_id": ENCODING_ID,
        "integer_bound_ledger": ledger,
        "integer_bound_ledger_digest": digest_json(ledger),
    }
    tables = _static_lookup_registry(provisional, config.range_bits)
    lasso_shapes, lasso_registry = _shape_only_lasso(model, graph, config.range_bits, tables, ledger)
    aux_shapes = _shape_only_aux(model, graph)
    for family in lasso_registry:
        if family["kind"] == "range":
            aux_shapes.append((f"lasso.{family['family']}.range_bits", (int(family["padded_count"]), int(config.range_bits))))
    private: Dict[str, ShapeArrayList] = {
        "com_W": [(name, tuple(int(dim) for dim in np.asarray(values).shape)) for name, values in _model_arrays(model)],
        "com_Aux": aux_shapes,
        "com_L": lasso_shapes,
    }
    x_public = [
        ("X_pub.H_input", (int(graph.num_nodes), int(graph.num_features))),
        ("X_pub.absolute_ids", (int(graph.num_nodes),)),
        ("X_pub.dimension_scalars", (7,)),
    ]
    return build_input_layout_from_shapes(x_public, private, lasso_registry), ledger
