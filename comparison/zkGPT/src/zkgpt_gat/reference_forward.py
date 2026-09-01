"""Independent scalar reference for the public canonical GAT output.

This module intentionally does not import or call ``forward_full`` or
``forward_public_y``.  It is a small, straightforward replay of the integer
relation used only to cross-check the production adapter's public output at
resource-generation and verification boundaries.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import Iterable

import numpy as np

from .data import LocalGraph
from .model import Model
from .rounding import signed_round_div, validate_rounding_mode
from .quantization import quantize_ties_away


def _exp(value: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 96
        return context.exp(value)


def _leaky(value: int) -> int:
    value = int(value)
    return value if value >= 0 else -((-value + 2) // 5)


def _elu(value: int) -> int:
    value = int(value)
    if value >= 0:
        return value
    raw = Decimal(16) * (_exp(Decimal(value) / Decimal(16)) - Decimal(1))
    return int(raw.to_integral_value(rounding=ROUND_HALF_UP))


def _exp_map(delta: int) -> int:
    delta = int(delta)
    if delta < 0:
        raise ValueError("reference ExpMap input must be nonnegative")
    raw = Decimal(16) * _exp(-Decimal(delta) / Decimal(256))
    return max(1, int(raw.to_integral_value(rounding=ROUND_HALF_UP)))


def _matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("reference matrix dimensions are inconsistent")
    output = np.empty((left.shape[0], right.shape[1]), dtype=object)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            total = 0
            for inner in range(left.shape[1]):
                total += int(left[row, inner]) * int(right[inner, column])
            output[row, column] = total
    return output


def _groups(graph: LocalGraph) -> Iterable[tuple[int, int, int]]:
    if graph.dst.size == 0:
        return
    begin = 0
    for index in range(1, int(graph.dst.size) + 1):
        if index == graph.dst.size or graph.dst[index] != graph.dst[begin] or graph.edge_graph[index] != graph.edge_graph[begin]:
            yield begin, index, int(graph.dst[begin])
            begin = index


def _head(
    graph: LocalGraph,
    values: np.ndarray,
    head,
    *,
    output: bool,
    h_prime_scale: int,
    aggregate_output_scale: int,
    output_head_scale: int,
    rounding_mode: str,
) -> np.ndarray:
    h_prime = _matmul(values, head.seq_kernel)
    e_src = _matmul(h_prime, np.asarray(head.attn_src_kernel, dtype=object).reshape((-1, 1))).reshape((-1,))
    e_dst = _matmul(h_prime, np.asarray(head.attn_dst_kernel, dtype=object).reshape((-1, 1))).reshape((-1,))
    aggregate = np.zeros((graph.num_nodes, h_prime.shape[1]), dtype=object)
    for begin, end, target in _groups(graph):
        scores = [_leaky(int(e_src[int(graph.src[index])]) + int(e_dst[int(graph.dst[index])])) for index in range(begin, end)]
        maximum = max(scores)
        weights = [_exp_map(maximum - score) for score in scores]
        denominator = sum(weights)
        for index, weight in zip(range(begin, end), weights):
            source = int(graph.src[index])
            for column in range(h_prime.shape[1]):
                aggregate[target, column] = int(aggregate[target, column]) + int(weight) * int(h_prime[source, column])
        for column in range(h_prime.shape[1]):
            numerator = int(aggregate[target, column]) * int(aggregate_output_scale)
            denominator_scaled = int(denominator) * int(h_prime_scale)
            value = signed_round_div(numerator, denominator_scaled, rounding_mode)
            if output:
                aligned = signed_round_div(int(value) * int(output_head_scale), int(aggregate_output_scale), rounding_mode)
                aggregate[target, column] = int(aligned) + int(head.output_bias[column])
            else:
                aggregate[target, column] = _elu(value)
    return aggregate


def reference_forward(graph: LocalGraph, model: Model, rounding_mode: str | None = None) -> np.ndarray:
    mode = validate_rounding_mode(model.rounding_mode if rounding_mode is None else rounding_mode)
    values = np.asarray(quantize_ties_away(graph.features, int(model.quant_scales["input"])), dtype=object)
    scales = model.quant_scales
    input_scale = int(scales["input"])
    for layer in model.hidden_layers:
        heads = [
            _head(
                graph,
                values,
                head,
                output=False,
                h_prime_scale=input_scale * int(scales["hidden_weight"]),
                aggregate_output_scale=int(scales["aggregate_output"]),
                output_head_scale=int(scales["output_head"]),
                rounding_mode=mode,
            )
            for head in layer.heads
        ]
        values = np.concatenate(heads, axis=1).astype(object)
        input_scale = int(scales["elu_output"])
    heads = [
        _head(
            graph,
            values,
            head,
            output=True,
            h_prime_scale=input_scale * int(scales["output_weight"]),
            aggregate_output_scale=int(scales["aggregate_output"]),
            output_head_scale=int(scales["output_head"]),
            rounding_mode=mode,
        )
        for head in model.output_heads
    ]
    if not heads:
        raise ValueError("reference model has no output heads")
    output = np.empty(heads[0].shape, dtype=object)
    for index in np.ndindex(output.shape):
        output[index] = signed_round_div(sum(int(head[index]) for head in heads), len(heads), mode)
    return output


__all__ = ["reference_forward"]
