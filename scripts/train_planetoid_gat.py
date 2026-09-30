#!/usr/bin/env python3
"""Train an explicit multi-layer, multi-head GAT on Planetoid datasets.

The structured torch state layout is part of the checkpoint contract:

* ``hidden_layers.<layer>.heads.<head>`` for every hidden attention head;
* ``output_heads.<head>`` for every output attention head.

Every output head applies its own post-aggregation bias and the final logits
are the arithmetic mean of all biased output heads.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from checkpoint_bundle import canonical_checkpoint_semantics, validate_checkpoint_metadata
from planetoid_utils import build_sorted_edge_index, labels_to_int, load_canonical_planetoid_features, load_planetoid_dataset_with_masks
from quantization_scale_chain import (
    CANONICAL_QUANT_CFG_ID,
    QuantizationScaleChain,
    canonical_quantization_scale_chain,
    canonical_input_preprocess,
    canonical_input_preprocess_id,
    quantize_array_ties_away,
)

HiddenProfile = List[Dict[str, int]]
OUTPUT_AVERAGE_RULE = "per_head_bias_then_arithmetic_mean"

CANONICAL_MODEL_SPECS = {
    "cora": ([1433, 64], 7, "gat_family_L3_hid8x8-8x8_kout2_c7", "cora_planetoid_seed11"),
    "citeseer": ([3703, 64], 6, "gat_family_L3_hid8x8-8x8_kout2_c6", "citeseer_planetoid_seed11"),
    "pubmed": ([500, 64], 3, "gat_family_L3_hid8x8-8x8_kout2_c3", "pubmed_planetoid_seed11"),
    "ppi": ([50, 64], 121, "gat_family_L3_hid8x8-8x8_kout2_c121", "ppi_gat_seed11"),
    "ogbn_arxiv": ([128, 64], 40, "gat_family_L3_hid8x8-8x8_kout2_c40", "ogbn_arxiv_seed11"),
}


def parse_cfg(path: pathlib.Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected key=value")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{line_number}: empty config key")
        if key in values:
            raise ValueError(f"{path}:{line_number}: duplicate config key {key!r}")
        values[key] = value.strip()
    return values


def cfg_int(cfg: Mapping[str, str], key: str, default: int) -> int:
    return int(cfg.get(key, default))


def cfg_float(cfg: Mapping[str, str], key: str, default: float) -> float:
    return float(cfg.get(key, default))


def cfg_bool(cfg: Mapping[str, str], key: str, default: bool) -> bool:
    value = cfg.get(key)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"config {key} must be boolean, got {value!r}")


def input_quantization_diagnostic(values: torch.Tensor | np.ndarray, *, scale: int | None = None, range_bits: int | None = None) -> Dict[str, object]:
    """Return a lightweight provenance diagnostic for input quantization.

    This helper does not alter the registered chain or clip values; it reports
    the canonical nearest-integer quantizer's information loss for the actual
    formal ``S_input`` and is persisted with the training summary.
    """
    if scale is None:
        scale = canonical_quantization_scale_chain().s_input
    if scale <= 0:
        raise ValueError("input quantization diagnostic scale must be positive")
    if isinstance(values, torch.Tensor):
        raw = values.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        raw = np.asarray(values, dtype=np.float64)
    raw = raw.reshape(-1)
    if raw.size == 0:
        raise ValueError("input quantization diagnostic requires non-empty input")
    if not np.all(np.isfinite(raw)):
        raise ValueError("input quantization diagnostic rejects NaN/Inf")
    scaled = raw * float(scale)
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    quantized = rounded / float(scale)
    non_integer = np.abs(scaled - np.trunc(scaled)) > 1e-12
    changed = np.abs(quantized - raw) > 1e-12
    result: Dict[str, object] = {
        "scale": int(scale),
        "element_count": int(raw.size),
        "zero_ratio": float(np.mean(rounded == 0.0)),
        "changed_ratio": float(np.mean(changed)),
        "non_integer_ratio": float(np.mean(non_integer)),
        "max_abs_input": float(np.max(np.abs(raw))),
        "max_abs_quantized": float(np.max(np.abs(quantized))),
        "abs_input_quantiles": {str(q): float(np.quantile(np.abs(raw), q)) for q in (0.0, 0.5, 0.9, 0.99, 1.0)},
        "max_abs_error": float(np.max(np.abs(quantized - raw))),
        # S_input quantization has no clipping/saturation branch.  Exposing
        # both fields makes that fact explicit and still reports values which
        # would exceed the configured unsigned limb/range capacity.
        # When a formal limb/range is supplied, saturation is exactly the
        # fraction of rounded values outside that representable range.  Do
        # not report a hard-coded zero for a lossy input sample.
        "saturation_ratio": 0.0,
    }
    if range_bits is not None:
        if range_bits <= 0 or range_bits >= 63:
            raise ValueError("diagnostic range_bits must be in [1, 62]")
        capacity = float(1 << range_bits)
        out_of_range = np.abs(rounded) >= capacity
        result["range_capacity_abs_exclusive"] = int(1 << range_bits)
        result["out_of_range_ratio"] = float(np.mean(out_of_range))
        result["out_of_range_count"] = int(np.count_nonzero(out_of_range))
        result["saturation_ratio"] = float(np.mean(out_of_range))
    return result


def quantize_torch_ste(values: torch.Tensor, scale: int) -> torch.Tensor:
    """Canonical ties-away quantization with a straight-through gradient.

    The forward value is exactly q/S, while the backward pass remains useful
    for quantization-aware training.  This is the same operation used by the
    formal integer path at every registered parameter/activation boundary.
    """
    if scale <= 0:
        raise ValueError("quantization scale must be positive")
    scaled = values * float(scale)
    rounded = torch.where(scaled >= 0.0, torch.floor(scaled + 0.5), torch.ceil(scaled - 0.5))
    quantized = rounded / float(scale)
    return values + (quantized - values).detach()


def quantized_input_features(
    input_features: np.ndarray, chain: QuantizationScaleChain, *, range_bits: int | None = None
) -> tuple[np.ndarray, Dict[str, object]]:
    """Materialize the formal public input q/S view for training."""
    input_view = np.asarray(input_features, dtype=np.float64)
    quantized_raw = quantize_array_ties_away(input_view, chain.s_input)
    quantized = quantized_raw.astype(np.float64) / float(chain.s_input)
    diagnostic = input_quantization_diagnostic(input_view, scale=chain.s_input, range_bits=range_bits)
    return quantized, diagnostic


def parse_hidden_profile_expr(raw: str) -> HiddenProfile:
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    entries = [item.strip() for item in text.split(",") if item.strip()]
    profile: HiddenProfile = []
    for entry in entries:
        parts = entry.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"invalid hidden_profile entry: {entry}")
        head_count, head_dim = (int(part.strip()) for part in parts)
        if type(head_count) is not int or type(head_dim) is not int or head_count <= 0 or head_dim <= 0:
            raise ValueError("hidden_profile head counts and dimensions must be positive")
        profile.append({"head_count": head_count, "head_dim": head_dim})
    if not profile:
        raise ValueError("hidden_profile must be non-empty")
    return profile


def require_canonical_model_config(cfg: Mapping[str, str], dataset: str) -> HiddenProfile:
    """Require the one architecture used by every registered formal dataset."""
    if dataset not in CANONICAL_MODEL_SPECS:
        raise ValueError(f"unsupported formal dataset: {dataset}")
    expected_d_in, expected_classes, expected_arch, expected_param = CANONICAL_MODEL_SPECS[dataset]
    try:
        hidden_profile = parse_hidden_profile_expr(cfg["hidden_profile"])
        raw_d_in = cfg["d_in_profile"].strip().strip("[]")
        d_in_profile = [int(value.strip()) for value in raw_d_in.split(",") if value.strip()]
    except (KeyError, ValueError) as error:
        raise ValueError(f"{dataset} formal config must expose canonical model shape") from error
    if hidden_profile != [{"head_count": 8, "head_dim": 8}, {"head_count": 8, "head_dim": 8}]:
        raise ValueError(f"{dataset} formal training requires hidden_profile=[8x8,8x8]")
    if d_in_profile != expected_d_in:
        raise ValueError(f"{dataset} formal training requires d_in_profile={expected_d_in}")
    if cfg_int(cfg, "L", 0) != 3 or cfg_int(cfg, "K_out", 0) != 2:
        raise ValueError(f"{dataset} formal training requires L=3 and K_out=2")
    if cfg_int(cfg, "num_classes", 0) != expected_classes:
        raise ValueError(f"{dataset} formal config has an unexpected class count")
    if cfg.get("model_arch_id") != expected_arch:
        raise ValueError(f"{dataset} formal config has an unexpected model_arch_id")
    if cfg.get("model_param_id") != expected_param:
        raise ValueError(f"{dataset} formal config has an unexpected model_param_id")
    expected_preprocess_id = canonical_input_preprocess_id(dataset)
    if cfg.get("input_preprocess_id") != expected_preprocess_id:
        raise ValueError(f"{dataset} formal config requires input_preprocess_id={expected_preprocess_id}")
    canonical_quantization_scale_chain(cfg.get("quant_cfg_id", ""), hidden_profile, 2)
    return hidden_profile


def normalize_hidden_profile(raw: Sequence[Mapping[str, object]]) -> HiddenProfile:
    profile: HiddenProfile = []
    for layer_index, entry in enumerate(raw):
        try:
            head_count = entry["head_count"]
            head_dim = entry["head_dim"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid hidden_profile layer {layer_index}") from error
        if head_count <= 0 or head_dim <= 0:
            raise ValueError(f"hidden_profile layer {layer_index} must have positive shape")
        profile.append({"head_count": head_count, "head_dim": head_dim})
    if not profile:
        raise ValueError("hidden_profile must contain at least one hidden layer")
    return profile


def derive_d_in_profile(in_dim: int, hidden_profile: Sequence[Mapping[str, object]]) -> List[int]:
    if in_dim <= 0:
        raise ValueError("input dimension must be positive")
    profile = normalize_hidden_profile(hidden_profile)
    d_in_profile = [in_dim]
    for layer in profile[:-1]:
        d_in_profile.append(layer["head_count"] * layer["head_dim"])
    return d_in_profile


def architecture_id(hidden_profile: Sequence[Mapping[str, object]], k_out: int, num_classes: int) -> str:
    profile = normalize_hidden_profile(hidden_profile)
    hidden_label = "-".join(f"{layer['head_count']}x{layer['head_dim']}" for layer in profile)
    return f"gat_family_L{len(profile) + 1}_hid{hidden_label}" f"_kout{k_out}_c{num_classes}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DatasetTensors:
    features: torch.Tensor
    labels: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    num_nodes: int
    num_features: int
    num_classes: int
    input_quantization_diagnostic: Dict[str, object]


def load_dataset(
    data_root: pathlib.Path,
    dataset: str,
    device: torch.device,
    *,
    cache_root: pathlib.Path | None = None,
    chain: QuantizationScaleChain | None = None,
    range_bits: int | None = None,
) -> DatasetTensors:
    sparse_features, labels, adjacency, train_mask, val_mask, test_mask = load_planetoid_dataset_with_masks(data_root, dataset)
    chain = chain or canonical_quantization_scale_chain()
    cache_base = cache_root if cache_root is not None else data_root / "cache"
    input_features = load_canonical_planetoid_features(cache_base, dataset)
    if input_features.shape != sparse_features.shape:
        raise ValueError("canonical Planetoid cache feature shape does not match dataset masks")
    input_features = canonical_input_preprocess(input_features, dataset, canonical_input_preprocess_id(dataset))
    features, input_diagnostic = quantized_input_features(input_features, chain, range_bits=range_bits)
    label_ids = labels_to_int(labels)
    edge_src, edge_dst = build_sorted_edge_index(adjacency, features.shape[0], add_self_loops=True)
    return DatasetTensors(
        # q/S is exactly representable for the canonical power-of-two S_input;
        # keep the preprocessing itself float64 before this device transfer.
        features=torch.tensor(features, dtype=torch.float32, device=device),
        labels=torch.tensor(label_ids, dtype=torch.long, device=device),
        train_mask=torch.tensor(train_mask, dtype=torch.bool, device=device),
        val_mask=torch.tensor(val_mask, dtype=torch.bool, device=device),
        test_mask=torch.tensor(test_mask, dtype=torch.bool, device=device),
        edge_src=torch.tensor(edge_src, dtype=torch.long, device=device),
        edge_dst=torch.tensor(edge_dst, dtype=torch.long, device=device),
        num_nodes=int(features.shape[0]),
        num_features=int(features.shape[1]),
        num_classes=int(labels.shape[1]),
        input_quantization_diagnostic=input_diagnostic,
    )


class AttentionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, output_bias: bool = False) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("attention head dimensions must be positive")
        self.seq = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_dst = nn.Linear(out_dim, 1, bias=False)
        self.attn_src = nn.Linear(out_dim, 1, bias=False)
        if output_bias:
            self.output_bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("output_bias", None)

    def forward(
        self,
        inputs: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        num_nodes: int,
        *,
        dropout: float,
        training: bool,
        apply_activation: bool,
        return_trace: bool = False,
        quantization: Mapping[str, int] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor] | None]:
        working_inputs = F.dropout(inputs, p=dropout, training=training) if dropout > 0.0 else inputs
        if quantization is None:
            h_prime = self.seq(working_inputs)
            e_dst = self.attn_dst(h_prime).squeeze(-1)
            e_src = self.attn_src(h_prime).squeeze(-1)
            linear_scores = e_dst[edge_dst] + e_src[edge_src]
            scores = F.leaky_relu(linear_scores, negative_slope=0.2)
        else:
            # Forward values are the registered fixed-point values q/S.  The
            # straight-through wrapper keeps this path trainable while making
            # checkpoint selection observe the same parameter and activation
            # quantization semantics as the formal trace.
            working_inputs = quantize_torch_ste(working_inputs, quantization["input"])
            seq_weight = quantize_torch_ste(self.seq.weight, quantization["weight"])
            attn_dst_weight = quantize_torch_ste(self.attn_dst.weight, quantization["a_dst"])
            attn_src_weight = quantize_torch_ste(self.attn_src.weight, quantization["a_src"])
            h_prime = F.linear(working_inputs, seq_weight)
            h_prime = quantize_torch_ste(h_prime, quantization["h_prime"])
            e_dst = F.linear(h_prime, attn_dst_weight).squeeze(-1)
            e_src = F.linear(h_prime, attn_src_weight).squeeze(-1)
            e_dst = quantize_torch_ste(e_dst, quantization["delta"])
            e_src = quantize_torch_ste(e_src, quantization["delta"])
            linear_scores = quantize_torch_ste(e_dst[edge_dst] + e_src[edge_src], quantization["delta"])
            scores = quantize_torch_ste(F.leaky_relu(linear_scores, negative_slope=0.2), quantization["delta"])

        max_per_dst = torch.full((num_nodes,), -torch.inf, device=inputs.device, dtype=inputs.dtype)
        max_per_dst.scatter_reduce_(0, edge_dst, scores, reduce="amax", include_self=True)
        shifted = scores - max_per_dst[edge_dst]
        if quantization is None:
            exp_scores = torch.exp(shifted)
        else:
            delta = quantize_torch_ste(max_per_dst[edge_dst] - scores, quantization["delta"])
            exp_scores = quantize_torch_ste(torch.exp(-delta), quantization["exp"])
            # The formal ExpMap table is defined on exp(-Delta) with a
            # positive unit entry at Delta=0.  Keep QAT on that same
            # registered support when a floating-point exp underflows to a
            # value which would quantize to zero.
            exp_scores = torch.clamp(exp_scores, min=1.0 / float(quantization["exp"]))
        sum_per_dst = torch.zeros((num_nodes,), device=inputs.device, dtype=inputs.dtype)
        sum_per_dst.index_add_(0, edge_dst, exp_scores)
        if quantization is not None:
            sum_per_dst = quantize_torch_ste(sum_per_dst, quantization["exp"])
        inverse = sum_per_dst.clamp_min(1e-12).reciprocal()
        alpha = exp_scores * inverse[edge_dst]
        if dropout > 0.0:
            alpha = F.dropout(alpha, p=dropout, training=training)

        message_inputs = F.dropout(h_prime, p=dropout, training=training) if dropout > 0.0 else h_prime
        aggregated = torch.zeros((num_nodes, h_prime.shape[1]), device=inputs.device, dtype=inputs.dtype)
        aggregated.index_add_(0, edge_dst, message_inputs[edge_src] * alpha.unsqueeze(-1))
        if quantization is not None:
            aggregated = quantize_torch_ste(aggregated, quantization["agg_pre"])
        if apply_activation:
            output = F.elu(aggregated)
            if quantization is not None:
                output = quantize_torch_ste(output, quantization["agg"])
        elif self.output_bias is not None:
            if quantization is None:
                output = aggregated + self.output_bias
            else:
                output = aggregated + quantize_torch_ste(self.output_bias, quantization["bias"])
                output = quantize_torch_ste(output, quantization["out"])
        else:
            output = aggregated

        if not return_trace:
            return output, None
        return output, {
            "H_prime": h_prime.detach().cpu(),
            "E_src": e_src.detach().cpu(),
            "E_dst": e_dst.detach().cpu(),
            "S": linear_scores.detach().cpu(),
            "Z": scores.detach().cpu(),
            "M": max_per_dst.detach().cpu(),
            "Delta": (max_per_dst[edge_dst] - scores).detach().cpu(),
            "U": exp_scores.detach().cpu(),
            "Sum": sum_per_dst.detach().cpu(),
            "inv": inverse.detach().cpu(),
            "alpha": alpha.detach().cpu(),
            "H_agg_pre_bias": aggregated.detach().cpu(),
            "H_agg": output.detach().cpu(),
        }


class HiddenAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, head_count: int, head_dim: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(in_dim, head_dim) for _ in range(head_count)])


class PlanetoidGAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int | None = None,
        *,
        hidden_profile: Sequence[Mapping[str, object]],
        k_out: int = 2,
        quantization: QuantizationScaleChain | None = None,
    ) -> None:
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes is required")
        self.hidden_profile = normalize_hidden_profile(hidden_profile)
        if k_out <= 0:
            raise ValueError("K_out must be positive")
        self.k_out = int(k_out)
        self.num_classes = int(num_classes)
        self.quantization = quantization
        d_in_profile = derive_d_in_profile(in_dim, self.hidden_profile)
        self.hidden_layers = nn.ModuleList(
            [
                HiddenAttentionLayer(d_in_profile[layer_index], int(layer["head_count"]), int(layer["head_dim"]))
                for layer_index, layer in enumerate(self.hidden_profile)
            ]
        )
        output_input_dim = int(self.hidden_profile[-1]["head_count"]) * int(self.hidden_profile[-1]["head_dim"])
        self.output_heads = nn.ModuleList([AttentionHead(output_input_dim, self.num_classes, output_bias=True) for _ in range(self.k_out)])

    def forward(
        self, features: torch.Tensor, edge_src: torch.Tensor, edge_dst: torch.Tensor, *, dropout: float, training: bool, return_trace: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, object] | None]:
        layer_input = features
        hidden_layer_traces: List[Dict[str, object]] = []
        for layer_index, layer in enumerate(self.hidden_layers):
            head_outputs: List[torch.Tensor] = []
            head_traces: List[Dict[str, torch.Tensor]] = []
            for local_head_index, head in enumerate(layer.heads):
                head_quantization = self.quantization.hidden_head(layer_index, local_head_index) if self.quantization is not None else None
                output, trace = head(
                    layer_input,
                    edge_src,
                    edge_dst,
                    features.shape[0],
                    dropout=dropout,
                    training=training,
                    apply_activation=True,
                    return_trace=return_trace,
                    quantization=head_quantization,
                )
                head_outputs.append(output)
                if trace is not None:
                    head_traces.append(trace)
            layer_input = torch.cat(head_outputs, dim=-1)
            if return_trace:
                hidden_layer_traces.append({"layer_index": layer_index, "concat": layer_input.detach().cpu(), "heads": head_traces})

        output_values: List[torch.Tensor] = []
        output_traces: List[Dict[str, torch.Tensor]] = []
        for head_index, head in enumerate(self.output_heads):
            head_quantization = self.quantization.output_head(head_index) if self.quantization is not None else None
            output, trace = head(
                layer_input,
                edge_src,
                edge_dst,
                features.shape[0],
                dropout=dropout,
                training=training,
                apply_activation=False,
                return_trace=return_trace,
                quantization=head_quantization,
            )
            output_values.append(output)
            if trace is not None:
                output_traces.append(trace)
        logits = torch.stack(output_values, dim=0).mean(dim=0)
        if self.quantization is not None:
            logits = quantize_torch_ste(logits, self.quantization.s_out)
        if not return_trace:
            return logits, None
        trace: Dict[str, object] = {"hidden_layers": hidden_layer_traces, "output_heads": output_traces, "logits": logits.detach().cpu()}
        return logits, trace


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params = []
    nodecay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or name.endswith(".output_bias"):
            nodecay_params.append(parameter)
        else:
            decay_params.append(parameter)
    return torch.optim.Adam([{"params": decay_params, "weight_decay": weight_decay}, {"params": nodecay_params, "weight_decay": 0.0}], lr=learning_rate)


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    predictions = torch.argmax(logits[mask], dim=-1)
    return float((predictions == labels[mask]).float().mean().item())


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[mask], labels[mask])


def checkpoint_metadata(
    *,
    dataset: str,
    in_dim: int,
    hidden_profile: Sequence[Mapping[str, object]],
    k_out: int,
    num_classes: int,
    seed: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    model_param_id: str,
    extra: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    profile = normalize_hidden_profile(hidden_profile)
    chain = canonical_quantization_scale_chain(CANONICAL_QUANT_CFG_ID, profile, k_out)
    metadata: Dict[str, object] = {
        "dataset": dataset,
        "family_schema": "multi_layer_multi_head",
        "L": len(profile) + 1,
        "d_in_profile": derive_d_in_profile(in_dim, profile),
        "hidden_profile": profile,
        "K_out": int(k_out),
        "C": int(num_classes),
        "num_features": int(in_dim),
        "num_classes": int(num_classes),
        "seed": int(seed),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "task_type": "transductive_node_classification",
        "report_unit": "node",
        "selection_metric": "validation_loss",
        "model_arch_id": architecture_id(profile, k_out, num_classes),
        "model_param_id": model_param_id,
        "quant_cfg_id": chain.quant_cfg_id,
        "input_preprocess_id": canonical_input_preprocess_id(dataset),
        "quantization_scale_chain": chain.canonical_serialization(),
        "static_table_id": "tables:lrelu+elu+exp+range",
        "degree_bound_id": "auto",
        "output_average_rule": OUTPUT_AVERAGE_RULE,
    }
    metadata.update(canonical_checkpoint_semantics(dataset))
    if extra is not None:
        metadata.update(extra)
    validate_checkpoint_metadata(metadata)
    return metadata


def load_checkpoint(
    path: Union[str, pathlib.Path], device: torch.device | str | None = None, *, map_location: torch.device | str | None = None
) -> Dict[str, object]:
    """Load one canonical structured formal checkpoint."""
    if device is not None and map_location is not None:
        raise ValueError("pass either device or map_location, not both")
    location = map_location if map_location is not None else device
    if location is None:
        location = torch.device("cpu")
    checkpoint = torch.load(str(path), map_location=location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint at {path} must be a dict")
    if "bundle" not in checkpoint or not isinstance(checkpoint["bundle"], dict):
        raise ValueError(f"checkpoint at {path} is missing bundle metadata")
    if "state_dict" not in checkpoint or not isinstance(checkpoint["state_dict"], dict):
        raise ValueError(f"checkpoint at {path} is missing state_dict")
    state = checkpoint["state_dict"]
    for key in state:
        if ("attn_src" in str(key) or "attn_dst" in str(key)) and (str(key).endswith(".bias") or str(key).endswith("_bias")):
            raise ValueError("bias-enabled attention checkpoint is not accepted by the formal loader: " + str(key))
    bundle = checkpoint["bundle"]
    validate_checkpoint_metadata(bundle)
    if bundle["family_schema"] != "multi_layer_multi_head":
        raise ValueError("checkpoint family_schema is not canonical")
    if bundle["L"] != 3 or bundle["K_out"] != 2:
        raise ValueError("checkpoint must declare the canonical L=3 and K_out=2")
    bundle_chain = bundle["quantization_scale_chain"]
    if not isinstance(bundle_chain, str) or not bundle_chain:
        raise ValueError("checkpoint is missing the complete quantization_scale_chain; " "quant_cfg_id alone is not a formal identity")
    try:
        profile = checkpoint_hidden_profile(bundle)
        chain = canonical_quantization_scale_chain(bundle["quant_cfg_id"], profile, bundle["K_out"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint quantization metadata is not canonical") from error
    dataset = bundle["dataset"]
    expected_preprocess_id = canonical_input_preprocess_id(dataset)
    if bundle["input_preprocess_id"] != expected_preprocess_id:
        raise ValueError("checkpoint input_preprocess_id does not match its formal dataset")
    if bundle_chain != chain.canonical_serialization():
        raise ValueError("checkpoint quantization_scale_chain does not match the canonical source")
    return checkpoint


def checkpoint_hidden_profile(bundle: Mapping[str, object]) -> HiddenProfile:
    raw = bundle["hidden_profile"]
    if isinstance(raw, list) and raw:
        return normalize_hidden_profile(raw)
    raise ValueError("checkpoint metadata must expose a non-empty hidden_profile")


def _resolve_output_dir(project_root: pathlib.Path, cfg: Mapping[str, str], dataset: str, cli_output_dir: str | None) -> pathlib.Path:
    if cli_output_dir is not None:
        return pathlib.Path(cli_output_dir).expanduser().resolve()
    return (project_root / cfg.get("checkpoint_bundle", f"artifacts/checkpoints/{dataset}_gat")).resolve()


def _override(value: object | None, fallback: object) -> object:
    return fallback if value is None else value


def add_training_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", help="canonical checkpoint destination (defaults to checkpoint_bundle)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use-cuda", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action="store_true", help="allow replacing best_model.pt in the selected output directory")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_training_cli(parser)
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config).resolve()
    cfg = parse_cfg(cfg_path)
    dataset = cfg["dataset"]
    if dataset not in {"cora", "citeseer", "pubmed"}:
        raise ValueError("Planetoid trainer supports cora/citeseer/pubmed")
    project_root = cfg_path.parent.parent.resolve()
    data_root = (project_root / cfg.get("data_root", "data")).resolve()
    output_dir = _resolve_output_dir(project_root, cfg, dataset, args.output_dir)
    checkpoint_path = output_dir / "best_model.pt"
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {checkpoint_path}; pass a fresh --output-dir " "or explicitly use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_profile = require_canonical_model_config(cfg, dataset)
    k_out = 2
    chain = canonical_quantization_scale_chain(cfg["quant_cfg_id"], hidden_profile, k_out)
    epochs = int(_override(args.epochs, cfg_int(cfg, "epochs", 500)))
    patience = int(_override(args.patience, cfg_int(cfg, "patience", 100)))
    learning_rate = float(_override(args.learning_rate, cfg_float(cfg, "learning_rate", 0.005)))
    weight_decay = float(_override(args.weight_decay, cfg_float(cfg, "weight_decay", 0.0005)))
    dropout = float(_override(args.dropout, cfg_float(cfg, "dropout", 0.6)))
    seed = int(_override(args.seed, cfg_int(cfg, "seed", 11)))
    use_cuda = bool(_override(args.use_cuda, True))
    expected_training = {"cora": (500, 100, 0.005, 0.0005, 0.6), "citeseer": (500, 100, 0.005, 0.0005, 0.6), "pubmed": (500, 100, 0.01, 0.001, 0.6)}[dataset]
    actual_training = (epochs, patience, learning_rate, weight_decay, dropout)
    if actual_training != expected_training or seed != 11:
        raise ValueError(f"{dataset} formal training requires epochs/patience/lr/weight_decay/" f"dropout={expected_training} and seed=11")
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and patience must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if not use_cuda:
        raise RuntimeError("formal training requires CUDA; refusing to create a CPU checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training is required but torch.cuda.is_available() is false")

    device = torch.device("cuda" if use_cuda else "cpu")
    set_seed(seed)
    cache_root = (project_root / cfg.get("cache_root", "data/cache")).resolve()
    tensors = load_dataset(data_root, dataset, device, cache_root=cache_root, chain=chain, range_bits=cfg_int(cfg, "range_bits", 12))
    input_quantization = tensors.input_quantization_diagnostic
    model = PlanetoidGAT(tensors.num_features, num_classes=tensors.num_classes, hidden_profile=hidden_profile, k_out=k_out, quantization=chain).to(device)
    optimizer = build_optimizer(model, learning_rate, weight_decay)
    metadata = checkpoint_metadata(
        dataset=dataset,
        in_dim=tensors.num_features,
        hidden_profile=hidden_profile,
        k_out=k_out,
        num_classes=tensors.num_classes,
        seed=seed,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        model_param_id=f"{dataset}_planetoid_seed{seed}",
    )

    best_state = None
    best_metrics = None
    best_val_loss = math.inf
    bad_epochs = 0
    history: List[Dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(tensors.features, tensors.edge_src, tensors.edge_dst, dropout=dropout, training=True)
        loss = masked_loss(logits, tensors.labels, tensors.train_mask)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(tensors.features, tensors.edge_src, tensors.edge_dst, dropout=0.0, training=False)
            train_accuracy = masked_accuracy(eval_logits, tensors.labels, tensors.train_mask)
            validation_loss = float(masked_loss(eval_logits, tensors.labels, tensors.val_mask).item())
            validation_accuracy = masked_accuracy(eval_logits, tensors.labels, tensors.val_mask)
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": float(loss.item()),
            "train_acc": train_accuracy,
            "val_loss": validation_loss,
            "val_acc": validation_accuracy,
        }
        history.append(epoch_metrics)
        print(
            f"[{dataset}] epoch={epoch:04d} train_loss={loss.item():.6f} "
            f"train_acc={train_accuracy:.4f} val_loss={validation_loss:.6f} "
            f"val_acc={validation_accuracy:.4f}",
            flush=True,
        )
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = {
                "best_epoch": epoch,
                "train_loss": float(loss.item()),
                "train_acc": train_accuracy,
                "val_loss": validation_loss,
                "val_acc": validation_accuracy,
            }
            torch.save({"bundle": metadata, "state_dict": best_state}, checkpoint_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("training did not produce a checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_logits, _ = model(tensors.features, tensors.edge_src, tensors.edge_dst, dropout=0.0, training=False)
        final_test_accuracy = masked_accuracy(best_logits, tensors.labels, tensors.test_mask)
        final_test_loss = float(masked_loss(best_logits, tensors.labels, tensors.test_mask).item())
    np.save(output_dir / "qat_best_logits.npy", best_logits.detach().cpu().numpy().astype(np.float32, copy=False))
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "qat_float32_non_formal": True,
                "device": str(device),
                "wall_time_sec": time.perf_counter() - start,
                "best_metrics": best_metrics,
                "final_test_metrics": {"test_loss": final_test_loss, "test_acc": final_test_accuracy},
                "input_quantization_diagnostic": input_quantization,
                "history_tail": history[-10:],
                "checkpoint_metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
