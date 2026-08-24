from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import Dict, List

import numpy as np

from .data import LocalGraph
from .model import AttentionHead, Model
from .field import BN254_FR_MODULUS, to_field_int
from .util import TimerBook, timed, update_digest_np


@dataclass
class SignedRoundDivTrace:
    x: np.ndarray
    d: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    z: np.ndarray
    a: np.ndarray
    a_inv: np.ndarray
    d_inv: np.ndarray
    q: np.ndarray
    r: np.ndarray
    delta: np.ndarray


@dataclass
class HeadTrace:
    tag: str
    h_prime: np.ndarray
    e_src: np.ndarray
    e_dst: np.ndarray
    s: np.ndarray
    z: np.ndarray
    m: np.ndarray
    delta: np.ndarray
    u: np.ndarray
    sum_: np.ndarray
    inv: np.ndarray
    alpha: np.ndarray
    h_agg_pre: np.ndarray
    h_agg: np.ndarray
    # Canonical integer/advice tensors used by the zkGPT relation.  The first
    # two fields are kept separate from the legacy floating-point names so an
    # artifact cannot accidentally prove a rescaled approximation.
    agg_num: np.ndarray | None = None
    elu_in: np.ndarray | None = None
    field_agg: np.ndarray | None = None
    denominator: np.ndarray | None = None
    h_prime_scale: int = 16
    aggregate_output_scale: int = 16
    aggregate_division: SignedRoundDivTrace | None = None
    scale_alignment_division: SignedRoundDivTrace | None = None
    output_aligned: np.ndarray | None = None
    group_start_selector: np.ndarray | None = None
    group_count: np.ndarray | None = None
    max_selector: np.ndarray | None = None
    r_u: np.ndarray | None = None
    r_num: np.ndarray | None = None
    r_field: np.ndarray | None = None


@dataclass
class LayerTrace:
    layer_index: int
    input: np.ndarray
    heads: List[HeadTrace]
    concat: np.ndarray


@dataclass
class ForwardTrace:
    input_features: np.ndarray
    hidden_layers: List[LayerTrace]
    output_heads: List[HeadTrace]
    output_head_values: List[np.ndarray]
    output_pre_bias_values: List[np.ndarray]
    y_pre_bias_avg: np.ndarray
    y: np.ndarray
    aux_base_digest: str
    lasso_digest: str
    metrics: Dict[str, float]
    output_average_division: SignedRoundDivTrace | None = None


def leaky_relu(values: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    value_dtype = np.asarray(values).dtype
    if not (np.issubdtype(value_dtype, np.integer) or value_dtype == object):
        raise TypeError("formal LeakyReLU accepts canonical integers only")
    arr = np.asarray(values, dtype=object)
    out = arr.copy()
    for index in np.ndindex(arr.shape):
        value = int(arr[index])
        out[index] = value if value >= 0 else -((-value + 2) // 5)
    return out


def elu(values: np.ndarray) -> np.ndarray:
    value_dtype = np.asarray(values).dtype
    if not (np.issubdtype(value_dtype, np.integer) or value_dtype == object):
        raise TypeError("formal ELU accepts canonical integers only")
    arr = np.asarray(values, dtype=object)
    out = arr.copy()
    for index in np.ndindex(arr.shape):
        value = int(arr[index])
        if value < 0:
            out[index] = _elu_table_value(value)
    return out


def _group_starts(dst: np.ndarray, edge_graph: np.ndarray) -> np.ndarray:
    if dst.size == 0:
        return np.array([], dtype=np.int64)
    new_group = np.empty(dst.size, dtype=bool)
    new_group[0] = True
    new_group[1:] = (dst[1:] != dst[:-1]) | (edge_graph[1:] != edge_graph[:-1])
    return np.flatnonzero(new_group)


def _scatter_group_values(starts: np.ndarray, dst: np.ndarray, node_count: int, values: np.ndarray) -> np.ndarray:
    out = np.zeros(node_count, dtype=np.asarray(values).dtype)
    out[dst[starts]] = values
    return out


def _aggregate(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray, source_matrix: np.ndarray, node_count: int) -> np.ndarray:
    width = source_matrix.shape[1]
    out = np.zeros((node_count, width), dtype=np.float64)
    for col in range(width):
        out[:, col] = np.bincount(
            dst,
            weights=alpha * source_matrix[src, col],
            minlength=node_count,
        )[:node_count]
    return out


def _safe_int_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Exact integer matrix product with a wide-integer ledger-sized route."""
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    if lhs.size == 0 or rhs.size == 0:
        return np.zeros((lhs.shape[0], rhs.shape[1]), dtype=np.int64)
    max_left = int(np.max(np.abs(lhs.astype(object))))
    max_right = int(np.max(np.abs(rhs.astype(object))))
    if max_left * max_right * max(lhs.shape[1], 1) < (1 << 62):
        return (lhs.astype(np.int64) @ rhs.astype(np.int64)).astype(np.int64)
    result = np.zeros((lhs.shape[0], rhs.shape[1]), dtype=object)
    for row in range(lhs.shape[0]):
        for col in range(rhs.shape[1]):
            result[row, col] = sum(int(lhs[row, idx]) * int(rhs[idx, col]) for idx in range(lhs.shape[1]))
    return result


def _round_div_away(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("SignedRoundDiv denominator must be positive")
    sign = -1 if numerator < 0 else 1
    value = abs(int(numerator))
    quotient, remainder = divmod(value, int(denominator))
    if 2 * remainder >= int(denominator):
        quotient += 1
    return sign * quotient


def _signed_round_div_trace(numerator: np.ndarray, denominator: np.ndarray) -> SignedRoundDivTrace:
    x_raw, d_raw = np.broadcast_arrays(np.asarray(numerator, dtype=object), np.asarray(denominator, dtype=object))
    shape = x_raw.shape
    arrays = {name: np.empty(shape, dtype=object) for name in ("y", "sigma", "z", "a", "a_inv", "d_inv", "q", "r", "delta")}
    for index in np.ndindex(shape):
        x = int(x_raw[index])
        d = int(d_raw[index])
        if d <= 0:
            raise ValueError("SignedRoundDiv denominator must be positive")
        y = _round_div_away(x, d)
        sigma = 1 if x < 0 else 0
        a = abs(x)
        q = abs(y)
        z = 1 if a == 0 else 0
        r = 2 * a + d - 2 * d * q
        delta = 2 * d - 1 - r
        if r < 0 or delta < 0:
            raise RuntimeError("nearest_ties_away SignedRoundDiv advice is inconsistent")
        arrays["y"][index] = y
        arrays["sigma"][index] = sigma
        arrays["z"][index] = z
        arrays["a"][index] = a
        arrays["a_inv"][index] = 0 if a == 0 else pow(a % BN254_FR_MODULUS, BN254_FR_MODULUS - 2, BN254_FR_MODULUS)
        arrays["d_inv"][index] = pow(d % BN254_FR_MODULUS, BN254_FR_MODULUS - 2, BN254_FR_MODULUS)
        arrays["q"][index] = q
        arrays["r"][index] = r
        arrays["delta"][index] = delta
    return SignedRoundDivTrace(
        x=np.asarray(x_raw, dtype=object),
        d=np.asarray(d_raw, dtype=object),
        y=arrays["y"],
        sigma=arrays["sigma"],
        z=arrays["z"],
        a=arrays["a"],
        a_inv=arrays["a_inv"],
        d_inv=arrays["d_inv"],
        q=arrays["q"],
        r=arrays["r"],
        delta=arrays["delta"],
    )


def _decimal_exp(value: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 96
        return context.exp(value)


def _elu_table_value(value: int) -> int:
    if int(value) >= 0:
        return int(value)
    raw = Decimal(16) * (_decimal_exp(Decimal(int(value)) / Decimal(16)) - Decimal(1))
    return int(raw.to_integral_value(rounding=ROUND_HALF_UP))


def _exp_map_fixed(delta: int) -> int:
    if int(delta) < 0:
        raise ValueError("ExpMap input must be nonnegative")
    value = Decimal(16) * _decimal_exp(-Decimal(int(delta)) / Decimal(256))
    return max(1, int(value.to_integral_value(rounding=ROUND_HALF_UP)))


def attention_head_forward(
    graph: LocalGraph,
    features: np.ndarray,
    head: AttentionHead,
    tag: str,
    apply_output_bias: bool,
    apply_activation: bool,
    timers: TimerBook,
    h_prime_scale: int,
    aggregate_output_scale: int,
    output_head_scale: int,
) -> HeadTrace:
    src = graph.src
    dst = graph.dst
    edge_graph = graph.edge_graph
    starts = _group_starts(dst, edge_graph)
    feature_dtype = np.asarray(features).dtype
    kernel_dtype = np.asarray(head.seq_kernel).dtype
    integer_path = (
        np.issubdtype(feature_dtype, np.integer)
        or feature_dtype == object
        or np.issubdtype(kernel_dtype, np.integer)
        or kernel_dtype == object
    )
    if not integer_path:
        raise TypeError("formal GAT forward requires canonical integer features and model parameters")
    if integer_path:
        with timed(timers, f"{tag}.projection_ms"):
            h_prime = _safe_int_matmul(np.asarray(features, dtype=np.int64), np.asarray(head.seq_kernel, dtype=np.int64))
        with timed(timers, f"{tag}.attention_ms"):
            e_dst = _safe_int_matmul(h_prime, np.asarray(head.attn_dst_kernel, dtype=np.int64).reshape((-1, 1))).reshape((-1,))
            e_src = _safe_int_matmul(h_prime, np.asarray(head.attn_src_kernel, dtype=np.int64).reshape((-1, 1))).reshape((-1,))
            s = e_src[src] + e_dst[dst]
            z = leaky_relu(s)
            max_values = np.maximum.reduceat(z, starts)
            m = _scatter_group_values(starts, dst, graph.num_nodes, max_values)
            delta = m[dst] - z
            u = np.asarray([_exp_map_fixed(int(value)) for value in delta], dtype=np.int64)
            sum_values = np.add.reduceat(u, starts)
            sum_ = _scatter_group_values(starts, dst, graph.num_nodes, sum_values)
            inv = np.asarray([pow(int(value), BN254_FR_MODULUS - 2, BN254_FR_MODULUS) for value in sum_], dtype=object)
            alpha = np.asarray([(int(u[index]) * int(inv[int(dst[index])])) % BN254_FR_MODULUS for index in range(u.size)], dtype=object)
            agg_num = np.zeros((graph.num_nodes, h_prime.shape[1]), dtype=object)
            field_agg = np.zeros((graph.num_nodes, h_prime.shape[1]), dtype=object)
            group_start_selector = np.zeros(src.size, dtype=np.int64)
            group_start_selector[starts] = 1
            group_count_values = np.diff(np.append(starts, src.size)).astype(np.int64)
            group_count = _scatter_group_values(starts, dst, graph.num_nodes, group_count_values)
            max_selector = np.zeros(src.size, dtype=np.int64)
            r_u = np.zeros(src.size, dtype=object)
            r_num = np.zeros((src.size, h_prime.shape[1]), dtype=object)
            r_field = np.zeros((src.size, h_prime.shape[1]), dtype=object)
            running_u = 0
            running_num = [0] * h_prime.shape[1]
            running_field = [0] * h_prime.shape[1]
            for edge_index in range(src.size):
                if group_start_selector[edge_index]:
                    running_u = 0
                    running_num = [0] * h_prime.shape[1]
                    running_field = [0] * h_prime.shape[1]
                target = int(dst[edge_index])
                source = int(src[edge_index])
                running_u += int(u[edge_index])
                r_u[edge_index] = running_u
                for col in range(h_prime.shape[1]):
                    running_num[col] += int(u[edge_index]) * int(h_prime[source, col])
                    running_field[col] = (running_field[col] + int(alpha[edge_index]) * int(h_prime[source, col])) % BN254_FR_MODULUS
                    r_num[edge_index, col] = running_num[col]
                    r_field[edge_index, col] = running_field[col]
                    agg_num[target, col] = running_num[col]
                    field_agg[target, col] = running_field[col]
            for begin_index, end_index in zip(starts, np.append(starts[1:], src.size)):
                local = int(begin_index) + int(np.argmax(z[int(begin_index):int(end_index)]))
                max_selector[local] = 1
            h_agg_pre = np.asarray(field_agg, dtype=object)
            denominator = np.asarray([int(value) for value in sum_], dtype=object)
            division_denominator = np.repeat((np.asarray(denominator, dtype=object) * int(h_prime_scale))[:, None], h_prime.shape[1], axis=1)
            aggregate_division = _signed_round_div_trace(
                np.asarray(agg_num, dtype=object) * int(aggregate_output_scale),
                division_denominator,
            )
            elu_in = aggregate_division.y
        with timed(timers, f"{tag}.activation_ms"):
            if apply_output_bias:
                scale_alignment_division = _signed_round_div_trace(
                    np.asarray(elu_in, dtype=object) * int(output_head_scale),
                    np.full(np.asarray(elu_in).shape, int(aggregate_output_scale), dtype=object),
                )
                aligned = scale_alignment_division.y
                h_agg = aligned + np.asarray(head.output_bias, dtype=object)[None, :]
            else:
                scale_alignment_division = None
                aligned = None
                h_agg = elu(elu_in)
        return HeadTrace(
            tag=tag,
            h_prime=h_prime,
            e_src=e_src,
            e_dst=e_dst,
            s=s,
            z=z,
            m=m,
            delta=delta,
            u=u,
            sum_=sum_,
            inv=inv,
            alpha=alpha,
            h_agg_pre=h_agg_pre,
            h_agg=h_agg,
            agg_num=agg_num,
            elu_in=elu_in,
            field_agg=field_agg,
            denominator=denominator,
            h_prime_scale=h_prime_scale,
            aggregate_output_scale=aggregate_output_scale,
            aggregate_division=aggregate_division,
            scale_alignment_division=scale_alignment_division,
            output_aligned=aligned,
            group_start_selector=group_start_selector,
            group_count=group_count,
            max_selector=max_selector,
            r_u=r_u,
            r_num=r_num,
            r_field=r_field,
        )
    raise AssertionError("unreachable non-integer formal forward branch")


def _digest_head_base(h: hashlib._Hash, trace: HeadTrace) -> None:
    for name in ["h_prime", "e_src", "e_dst", "s", "z", "m", "delta", "u", "sum_", "inv", "alpha", "h_agg_pre", "h_agg"]:
        update_digest_np(h, f"{trace.tag}.{name}", getattr(trace, name))


def _digest_lasso(h: hashlib._Hash, trace: HeadTrace, include_elu: bool) -> None:
    update_digest_np(h, f"{trace.tag}.L_LReLU.x", trace.s)
    update_digest_np(h, f"{trace.tag}.L_LReLU.y", trace.z)
    update_digest_np(h, f"{trace.tag}.L_exp.x", trace.delta)
    update_digest_np(h, f"{trace.tag}.L_exp.y", trace.u)
    update_digest_np(h, f"{trace.tag}.L_range.delta", trace.delta)
    if include_elu:
        update_digest_np(h, f"{trace.tag}.L_ELU.x", trace.elu_in if trace.elu_in is not None else trace.h_agg_pre)
        update_digest_np(h, f"{trace.tag}.L_ELU.y", trace.h_agg)


def forward_full(graph: LocalGraph, model: Model) -> ForwardTrace:
    timers = TimerBook({})
    base_hash = hashlib.sha256()
    lasso_hash = hashlib.sha256()
    hidden_layers: List[LayerTrace] = []
    # Public features are canonical integers at S_H^(1)=1.  The model loader
    # has already converted every private parameter to its registered integer
    # scale, so all following operations use the exact branch above.
    layer_input = np.asarray([int(np.floor(float(item) + 0.5) if float(item) >= 0 else np.ceil(float(item) - 0.5)) for item in np.ravel(graph.features)], dtype=np.int64).reshape(graph.features.shape)
    scales = model.quant_scales
    layer_input_scale = int(scales["input"])
    update_digest_np(base_hash, "H.input_features", layer_input)

    for layer in model.hidden_layers:
        heads: List[HeadTrace] = []
        outputs: List[np.ndarray] = []
        for head_index, head in enumerate(layer.heads):
            tag = f"hidden.l{layer.layer_index + 1}.h{head_index}"
            trace = attention_head_forward(
                graph,
                layer_input,
                head,
                tag,
                apply_output_bias=False,
                apply_activation=True,
                timers=timers,
                h_prime_scale=layer_input_scale * int(scales["hidden_weight"]),
                aggregate_output_scale=int(scales["aggregate_output"]),
                output_head_scale=int(scales["output_head"]),
            )
            heads.append(trace)
            outputs.append(trace.h_agg)
            _digest_head_base(base_hash, trace)
            _digest_lasso(lasso_hash, trace, include_elu=True)
        with timed(timers, f"hidden.l{layer.layer_index + 1}.concat_ms"):
            concat = np.concatenate(outputs, axis=1)
        update_digest_np(base_hash, f"hidden.l{layer.layer_index + 1}.H_cat", concat)
        hidden_layers.append(LayerTrace(layer_index=layer.layer_index + 1, input=layer_input, heads=heads, concat=concat))
        layer_input = concat
        # The registered ELU output scale is S_agg=16 and concat does not
        # change a coordinate's scale.
        layer_input_scale = int(scales["elu_output"])

    output_heads: List[HeadTrace] = []
    output_values: List[np.ndarray] = []
    output_pre_bias: List[np.ndarray] = []
    for head_index, head in enumerate(model.output_heads):
        tag = f"output.h{head_index}"
        trace = attention_head_forward(
            graph,
            layer_input,
            head,
            tag,
            apply_output_bias=True,
            apply_activation=False,
            timers=timers,
            h_prime_scale=layer_input_scale * int(scales["output_weight"]),
            aggregate_output_scale=int(scales["aggregate_output"]),
            output_head_scale=int(scales["output_head"]),
        )
        output_heads.append(trace)
        output_pre_bias.append(trace.output_aligned if trace.output_aligned is not None else trace.elu_in)
        output_values.append(trace.h_agg)
        _digest_head_base(base_hash, trace)
        _digest_lasso(lasso_hash, trace, include_elu=False)

    with timed(timers, "output.average_ms"):
        output_dtype = np.asarray(output_pre_bias[0]).dtype if output_pre_bias else np.float64
        if output_pre_bias and (np.issubdtype(output_dtype, np.integer) or output_dtype == object):
            pre_stack = np.stack(output_pre_bias, axis=0).astype(object)
            value_stack = np.stack(output_values, axis=0).astype(object)
            y_pre = np.empty(pre_stack.shape[1:], dtype=object)
            y_sum = np.empty(value_stack.shape[1:], dtype=object)
            for index in np.ndindex(y_pre.shape):
                y_pre[index] = _round_div_away(sum(int(item[index]) for item in pre_stack), len(output_pre_bias))
                y_sum[index] = sum(int(item[index]) for item in value_stack)
            output_average_division = _signed_round_div_trace(
                y_sum,
                np.full(y_sum.shape, len(output_values), dtype=object),
            )
            y = output_average_division.y
        else:
            y_pre = np.mean(np.stack(output_pre_bias, axis=0), axis=0)
            y = np.mean(np.stack(output_values, axis=0), axis=0)
            output_average_division = None
    update_digest_np(base_hash, "output.Y_pre_bias_average", y_pre)
    update_digest_np(base_hash, "output.Y", y)
    return ForwardTrace(
        input_features=graph.features,
        hidden_layers=hidden_layers,
        output_heads=output_heads,
        output_head_values=output_values,
        output_pre_bias_values=output_pre_bias,
        y_pre_bias_avg=y_pre,
        y=y,
        aux_base_digest=base_hash.hexdigest(),
        lasso_digest=lasso_hash.hexdigest(),
        metrics=timers.values,
        output_average_division=output_average_division,
    )
