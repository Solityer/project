from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .data import LocalGraph
from .model import AttentionHead, Model
from .util import TimerBook, timed, update_digest_np


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


def leaky_relu(values: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return np.where(values >= 0.0, values, alpha * values)


def elu(values: np.ndarray) -> np.ndarray:
    out = np.array(values, copy=True)
    mask = values < 0.0
    out[mask] = np.expm1(values[mask])
    return out


def _group_starts(dst: np.ndarray, edge_graph: np.ndarray) -> np.ndarray:
    if dst.size == 0:
        return np.array([], dtype=np.int64)
    new_group = np.empty(dst.size, dtype=bool)
    new_group[0] = True
    new_group[1:] = (dst[1:] != dst[:-1]) | (edge_graph[1:] != edge_graph[:-1])
    return np.flatnonzero(new_group)


def _scatter_group_values(starts: np.ndarray, dst: np.ndarray, node_count: int, values: np.ndarray) -> np.ndarray:
    out = np.zeros(node_count, dtype=np.float64)
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


def attention_head_forward(
    graph: LocalGraph,
    features: np.ndarray,
    head: AttentionHead,
    tag: str,
    apply_output_bias: bool,
    apply_activation: bool,
    timers: TimerBook,
) -> HeadTrace:
    src = graph.src
    dst = graph.dst
    edge_graph = graph.edge_graph
    starts = _group_starts(dst, edge_graph)
    with timed(timers, f"{tag}.projection_ms"):
        h_prime = features @ head.seq_kernel
    with timed(timers, f"{tag}.attention_ms"):
        e_dst = h_prime @ head.attn_dst_kernel + head.attn_dst_bias
        e_src = h_prime @ head.attn_src_kernel + head.attn_src_bias
        s = e_src[src] + e_dst[dst]
        z = leaky_relu(s)
        max_values = np.maximum.reduceat(z, starts)
        m = _scatter_group_values(starts, dst, graph.num_nodes, max_values)
        delta = m[dst] - z
        # The protocol's ExpMap consumes Delta+ and maps it to exp(-Delta+).
        u = np.exp(-delta)
        sum_values = np.add.reduceat(u, starts)
        sum_ = _scatter_group_values(starts, dst, graph.num_nodes, sum_values)
        inv = 1.0 / sum_
        alpha = u * inv[dst]
        h_agg_pre = _aggregate(dst, src, alpha, h_prime, graph.num_nodes)
    with timed(timers, f"{tag}.activation_ms"):
        if apply_output_bias:
            pre_bias = h_agg_pre
            h_agg = pre_bias + head.output_bias[None, :]
        else:
            h_agg = h_agg_pre
        if apply_activation:
            h_agg = elu(h_agg)
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
    )


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
        update_digest_np(h, f"{trace.tag}.L_ELU.x", trace.h_agg_pre)
        update_digest_np(h, f"{trace.tag}.L_ELU.y", trace.h_agg)


def forward_full(graph: LocalGraph, model: Model) -> ForwardTrace:
    timers = TimerBook({})
    base_hash = hashlib.sha256()
    lasso_hash = hashlib.sha256()
    hidden_layers: List[LayerTrace] = []
    layer_input = graph.features
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
        )
        output_heads.append(trace)
        output_pre_bias.append(trace.h_agg_pre)
        output_values.append(trace.h_agg)
        _digest_head_base(base_hash, trace)
        _digest_lasso(lasso_hash, trace, include_elu=False)

    with timed(timers, "output.average_ms"):
        y_pre = np.mean(np.stack(output_pre_bias, axis=0), axis=0)
        y = np.mean(np.stack(output_values, axis=0), axis=0)
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
    )
