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

from planetoid_utils import (
    build_sorted_edge_index,
    labels_to_int,
    load_planetoid_dataset_with_masks,
    row_normalize_features,
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
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
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
            raise ValueError(
                f"{path}:{line_number}: duplicate config key {key!r}"
            )
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
        if head_count <= 0 or head_dim <= 0:
            raise ValueError(
                "hidden_profile head counts and dimensions must be positive"
            )
        profile.append({"head_count": head_count, "head_dim": head_dim})
    if not profile:
        raise ValueError("hidden_profile must be non-empty")
    return profile


def require_canonical_model_config(
    cfg: Mapping[str, str], dataset: str
) -> HiddenProfile:
    """Require the one architecture used by every registered formal dataset."""
    if dataset not in CANONICAL_MODEL_SPECS:
        raise ValueError(f"unsupported formal dataset: {dataset}")
    expected_d_in, expected_classes, expected_arch, expected_param = CANONICAL_MODEL_SPECS[
        dataset
    ]
    try:
        hidden_profile = parse_hidden_profile_expr(cfg["hidden_profile"])
        raw_d_in = cfg["d_in_profile"].strip().strip("[]")
        d_in_profile = [int(value.strip()) for value in raw_d_in.split(",") if value.strip()]
    except (KeyError, ValueError) as error:
        raise ValueError(f"{dataset} formal config must expose canonical model shape") from error
    if hidden_profile != [
        {"head_count": 8, "head_dim": 8},
        {"head_count": 8, "head_dim": 8},
    ]:
        raise ValueError(f"{dataset} formal training requires hidden_profile=[8x8,8x8]")
    if d_in_profile != expected_d_in:
        raise ValueError(
            f"{dataset} formal training requires d_in_profile={expected_d_in}"
        )
    if cfg_int(cfg, "L", 0) != 3 or cfg_int(cfg, "K_out", 0) != 2:
        raise ValueError(f"{dataset} formal training requires L=3 and K_out=2")
    if cfg_int(cfg, "num_classes", 0) != expected_classes:
        raise ValueError(f"{dataset} formal config has an unexpected class count")
    if cfg.get("model_arch_id") != expected_arch:
        raise ValueError(f"{dataset} formal config has an unexpected model_arch_id")
    if cfg.get("model_param_id") != expected_param:
        raise ValueError(f"{dataset} formal config has an unexpected model_param_id")
    if cfg.get("quant_cfg_id") != "fp32_torch_checkpoint":
        raise ValueError(f"{dataset} formal training requires quant_cfg_id=fp32_torch_checkpoint")
    return hidden_profile


def normalize_hidden_profile(raw: Sequence[Mapping[str, object]]) -> HiddenProfile:
    profile: HiddenProfile = []
    for layer_index, entry in enumerate(raw):
        try:
            head_count = int(entry["head_count"])
            head_dim = int(entry["head_dim"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid hidden_profile layer {layer_index}"
            ) from error
        if head_count <= 0 or head_dim <= 0:
            raise ValueError(
                f"hidden_profile layer {layer_index} must have positive shape"
            )
        profile.append({"head_count": head_count, "head_dim": head_dim})
    if not profile:
        raise ValueError("hidden_profile must contain at least one hidden layer")
    return profile


def derive_d_in_profile(in_dim: int, hidden_profile: Sequence[Mapping[str, object]]) -> List[int]:
    if in_dim <= 0:
        raise ValueError("input dimension must be positive")
    profile = normalize_hidden_profile(hidden_profile)
    d_in_profile = [int(in_dim)]
    for layer in profile[:-1]:
        d_in_profile.append(int(layer["head_count"]) * int(layer["head_dim"]))
    return d_in_profile


def architecture_id(
    hidden_profile: Sequence[Mapping[str, object]],
    k_out: int,
    num_classes: int,
) -> str:
    profile = normalize_hidden_profile(hidden_profile)
    hidden_label = "-".join(
        f"{layer['head_count']}x{layer['head_dim']}" for layer in profile
    )
    return (
        f"gat_family_L{len(profile) + 1}_hid{hidden_label}"
        f"_kout{k_out}_c{num_classes}"
    )


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


def load_dataset(
    data_root: pathlib.Path,
    dataset: str,
    device: torch.device,
) -> DatasetTensors:
    (
        sparse_features,
        labels,
        adjacency,
        train_mask,
        val_mask,
        test_mask,
    ) = load_planetoid_dataset_with_masks(data_root, dataset)
    features = row_normalize_features(sparse_features)
    label_ids = labels_to_int(labels)
    edge_src, edge_dst = build_sorted_edge_index(
        adjacency, features.shape[0], add_self_loops=True
    )
    return DatasetTensors(
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
    )


class AttentionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, output_bias: bool = False) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("attention head dimensions must be positive")
        self.seq = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_dst = nn.Linear(out_dim, 1, bias=True)
        self.attn_src = nn.Linear(out_dim, 1, bias=True)
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor] | None]:
        working_inputs = (
            F.dropout(inputs, p=dropout, training=training)
            if dropout > 0.0
            else inputs
        )
        h_prime = self.seq(working_inputs)
        e_dst = self.attn_dst(h_prime).squeeze(-1)
        e_src = self.attn_src(h_prime).squeeze(-1)
        linear_scores = e_dst[edge_dst] + e_src[edge_src]
        scores = F.leaky_relu(linear_scores, negative_slope=0.2)

        max_per_dst = torch.full(
            (num_nodes,), -torch.inf, device=inputs.device, dtype=inputs.dtype
        )
        max_per_dst.scatter_reduce_(
            0, edge_dst, scores, reduce="amax", include_self=True
        )
        shifted = scores - max_per_dst[edge_dst]
        exp_scores = torch.exp(shifted)
        sum_per_dst = torch.zeros(
            (num_nodes,), device=inputs.device, dtype=inputs.dtype
        )
        sum_per_dst.index_add_(0, edge_dst, exp_scores)
        inverse = sum_per_dst.clamp_min(1e-12).reciprocal()
        alpha = exp_scores * inverse[edge_dst]
        if dropout > 0.0:
            alpha = F.dropout(alpha, p=dropout, training=training)

        message_inputs = (
            F.dropout(h_prime, p=dropout, training=training)
            if dropout > 0.0
            else h_prime
        )
        aggregated = torch.zeros(
            (num_nodes, h_prime.shape[1]),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        aggregated.index_add_(
            0, edge_dst, message_inputs[edge_src] * alpha.unsqueeze(-1)
        )
        if apply_activation:
            output = F.elu(aggregated)
        elif self.output_bias is not None:
            output = aggregated + self.output_bias
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
        self.heads = nn.ModuleList(
            [AttentionHead(in_dim, head_dim) for _ in range(head_count)]
        )


class PlanetoidGAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        num_classes: int | None = None,
        hidden_heads: int = 8,
        *,
        hidden_profile: Sequence[Mapping[str, object]] | None = None,
        k_out: int = 1,
    ) -> None:
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes is required")
        if hidden_profile is None:
            if hidden_dim is None:
                raise ValueError("hidden_dim or hidden_profile is required")
            hidden_profile = [
                {"head_count": int(hidden_heads), "head_dim": int(hidden_dim)}
            ]
        self.hidden_profile = normalize_hidden_profile(hidden_profile)
        if k_out <= 0:
            raise ValueError("K_out must be positive")
        self.k_out = int(k_out)
        self.num_classes = int(num_classes)
        d_in_profile = derive_d_in_profile(in_dim, self.hidden_profile)
        self.hidden_layers = nn.ModuleList(
            [
                HiddenAttentionLayer(
                    d_in_profile[layer_index],
                    int(layer["head_count"]),
                    int(layer["head_dim"]),
                )
                for layer_index, layer in enumerate(self.hidden_profile)
            ]
        )
        output_input_dim = (
            int(self.hidden_profile[-1]["head_count"])
            * int(self.hidden_profile[-1]["head_dim"])
        )
        self.output_heads = nn.ModuleList(
            [
                AttentionHead(
                    output_input_dim, self.num_classes, output_bias=True
                )
                for _ in range(self.k_out)
            ]
        )

    def forward(
        self,
        features: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        *,
        dropout: float,
        training: bool,
        return_trace: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object] | None]:
        layer_input = features
        hidden_layer_traces: List[Dict[str, object]] = []
        for layer_index, layer in enumerate(self.hidden_layers):
            head_outputs: List[torch.Tensor] = []
            head_traces: List[Dict[str, torch.Tensor]] = []
            for head in layer.heads:
                output, trace = head(
                    layer_input,
                    edge_src,
                    edge_dst,
                    features.shape[0],
                    dropout=dropout,
                    training=training,
                    apply_activation=True,
                    return_trace=return_trace,
                )
                head_outputs.append(output)
                if trace is not None:
                    head_traces.append(trace)
            layer_input = torch.cat(head_outputs, dim=-1)
            if return_trace:
                hidden_layer_traces.append(
                    {
                        "layer_index": layer_index,
                        "concat": layer_input.detach().cpu(),
                        "heads": head_traces,
                    }
                )

        output_values: List[torch.Tensor] = []
        output_traces: List[Dict[str, torch.Tensor]] = []
        for head in self.output_heads:
            output, trace = head(
                layer_input,
                edge_src,
                edge_dst,
                features.shape[0],
                dropout=dropout,
                training=training,
                apply_activation=False,
                return_trace=return_trace,
            )
            output_values.append(output)
            if trace is not None:
                output_traces.append(trace)
        logits = torch.stack(output_values, dim=0).mean(dim=0)
        if not return_trace:
            return logits, None
        trace: Dict[str, object] = {
            "hidden_layers": hidden_layer_traces,
            "output_heads": output_traces,
            "output_head_values": [value.detach().cpu() for value in output_values],
            "logits": logits.detach().cpu(),
            # Non-formal parity-only aliases for older trace consumers. The
            # canonical checkpoint/export/proof path uses the structured keys.
            "hidden_concat": hidden_layer_traces[-1]["concat"],
            "hidden_heads": (
                hidden_layer_traces[0]["heads"]
                if len(hidden_layer_traces) == 1
                else [
                    head_trace
                    for layer_trace in hidden_layer_traces
                    for head_trace in layer_trace["heads"]
                ]
            ),
        }
        if len(output_traces) == 1:
            trace["output"] = output_traces[0]
        return logits, trace


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decay_params = []
    nodecay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or name.endswith(".output_bias"):
            nodecay_params.append(parameter)
        else:
            decay_params.append(parameter)
    return torch.optim.Adam(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def masked_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    predictions = torch.argmax(logits[mask], dim=-1)
    return float((predictions == labels[mask]).float().mean().item())


def masked_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
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
    metadata: Dict[str, object] = {
        "dataset": dataset,
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
        "quant_cfg_id": "fp32_torch_checkpoint",
        "static_table_id": "tables:lrelu+elu+exp+range",
        "degree_bound_id": "auto",
        "output_average_rule": OUTPUT_AVERAGE_RULE,
    }
    if extra is not None:
        metadata.update(extra)
    return metadata


def load_checkpoint(
    path: Union[str, pathlib.Path],
    device: torch.device | str | None = None,
    *,
    map_location: torch.device | str | None = None,
) -> Dict[str, object]:
    """Load a checkpoint while preserving the historical public interface."""
    if device is not None and map_location is not None:
        raise ValueError("pass either device or map_location, not both")
    location = map_location if map_location is not None else device
    if location is None:
        location = torch.device("cpu")
    checkpoint = torch.load(
        str(path), map_location=location, weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint at {path} must be a dict")
    if not isinstance(checkpoint.get("bundle"), dict):
        raise ValueError(f"checkpoint at {path} is missing bundle metadata")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError(f"checkpoint at {path} is missing state_dict")
    return checkpoint


def checkpoint_hidden_profile(bundle: Mapping[str, object]) -> HiddenProfile:
    raw = bundle.get("hidden_profile")
    if isinstance(raw, list) and raw:
        return normalize_hidden_profile(raw)
    if "hidden_heads" in bundle and "hidden_dim" in bundle:
        return normalize_hidden_profile(
            [
                {
                    "head_count": int(bundle["hidden_heads"]),
                    "head_dim": int(bundle["hidden_dim"]),
                }
            ]
        )
    raise ValueError("checkpoint metadata does not expose hidden_profile")


def structured_checkpoint_state(
    state: Mapping[str, torch.Tensor],
    hidden_profile: Sequence[Mapping[str, object]],
    k_out: int,
    num_classes: int,
) -> Dict[str, torch.Tensor]:
    """Normalize historical one-layer checkpoints for parity replay only."""
    if any(key.startswith("hidden_layers.") for key in state):
        return dict(state)
    if not any(key.startswith("hidden_heads.") for key in state):
        raise ValueError("checkpoint state_dict has no supported GAT layout")
    profile = normalize_hidden_profile(hidden_profile)
    if len(profile) != 1 or k_out != 1:
        raise ValueError(
            "legacy checkpoint layout only supports one hidden layer and K_out=1"
        )
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("hidden_heads."):
            normalized[f"hidden_layers.0.heads.{key[len('hidden_heads.'):]}"] = value
        elif key.startswith("output_head."):
            normalized[f"output_heads.0.{key[len('output_head.'):]}"] = value
        else:
            raise ValueError(f"unsupported legacy state key: {key}")
    normalized.setdefault(
        "output_heads.0.output_bias",
        torch.zeros(num_classes, dtype=next(iter(state.values())).dtype),
    )
    return normalized


def load_bundle_state(
    bundle_dir: pathlib.Path,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    manifest = json.loads(
        (bundle_dir / "manifest.json").read_text(encoding="utf-8")
    )
    hidden_profile = normalize_hidden_profile(manifest["hidden_profile"])
    d_in_profile = [int(value) for value in manifest["d_in_profile"]]
    if len(d_in_profile) != len(hidden_profile):
        raise ValueError("bundle d_in_profile must align with hidden_profile")
    k_out = int(manifest["K_out"])
    num_classes = int(manifest["C"])
    tensor_index = manifest["tensor_index"]

    with np.load(bundle_dir / "tensors.npz") as archive:
        def tensor(name: str) -> np.ndarray:
            key = tensor_index[name]["archive_key"]
            return np.asarray(archive[key], dtype=np.float32).copy()

        state: Dict[str, torch.Tensor] = {}
        for head in manifest["hidden_head_specs"]:
            layer_index = int(head["layer_index"])
            local_head_index = int(head["local_head_index"])
            prefix = f"hidden_layers.{layer_index}.heads.{local_head_index}"
            seq = tensor(head["seq_kernel"])
            state[f"{prefix}.seq.weight"] = torch.from_numpy(
                seq.reshape(seq.shape[1], seq.shape[2]).T.copy()
            )
            state[f"{prefix}.attn_dst.weight"] = torch.from_numpy(
                tensor(head["attn_dst_kernel"]).reshape(1, -1)
            )
            state[f"{prefix}.attn_dst.bias"] = torch.from_numpy(
                tensor(head["attn_dst_bias"]).reshape(1)
            )
            state[f"{prefix}.attn_src.weight"] = torch.from_numpy(
                tensor(head["attn_src_kernel"]).reshape(1, -1)
            )
            state[f"{prefix}.attn_src.bias"] = torch.from_numpy(
                tensor(head["attn_src_bias"]).reshape(1)
            )

        for head in manifest["output_head_specs"]:
            head_index = int(head["head_index"])
            prefix = f"output_heads.{head_index}"
            seq = tensor(head["seq_kernel"])
            state[f"{prefix}.seq.weight"] = torch.from_numpy(
                seq.reshape(seq.shape[1], seq.shape[2]).T.copy()
            )
            state[f"{prefix}.attn_dst.weight"] = torch.from_numpy(
                tensor(head["attn_dst_kernel"]).reshape(1, -1)
            )
            state[f"{prefix}.attn_dst.bias"] = torch.from_numpy(
                tensor(head["attn_dst_bias"]).reshape(1)
            )
            state[f"{prefix}.attn_src.weight"] = torch.from_numpy(
                tensor(head["attn_src_kernel"]).reshape(1, -1)
            )
            state[f"{prefix}.attn_src.bias"] = torch.from_numpy(
                tensor(head["attn_src_bias"]).reshape(1)
            )
            state[f"{prefix}.output_bias"] = torch.from_numpy(
                tensor(head["output_bias"]).reshape(-1)
            )

    metadata = {
        "L": len(hidden_profile) + 1,
        "d_in_profile": d_in_profile,
        "hidden_profile": hidden_profile,
        "K_out": k_out,
        "C": num_classes,
        "num_features": d_in_profile[0],
        "num_classes": num_classes,
        "output_average_rule": manifest.get("output_average_rule"),
    }
    return metadata, state


def _numpy_trace(logits: torch.Tensor, trace: Dict[str, object]) -> Dict[str, object]:
    return {
        "logits": logits.detach().cpu().numpy(),
        "hidden_layers": [
            {
                "concat": layer["concat"].numpy(),
                "heads": [
                    {key: value.numpy() for key, value in head.items()}
                    for head in layer["heads"]
                ],
            }
            for layer in trace["hidden_layers"]
        ],
        "output_heads": [
            {key: value.numpy() for key, value in head.items()}
            for head in trace["output_heads"]
        ],
        "output_head_values": [value.numpy() for value in trace["output_head_values"]],
    }


def _run_trace(
    model: PlanetoidGAT,
    tensors: DatasetTensors,
) -> Dict[str, object]:
    model.eval()
    with torch.no_grad():
        logits, trace = model(
            tensors.features,
            tensors.edge_src,
            tensors.edge_dst,
            dropout=0.0,
            training=False,
            return_trace=True,
        )
    assert trace is not None
    result = _numpy_trace(logits, trace)
    # Non-formal parity-only aliases; canonical consumers use hidden_layers and
    # output_heads directly.
    result["hidden_concat"] = result["hidden_layers"][-1]["concat"]
    result["hidden_heads"] = result["hidden_layers"][0]["heads"]
    if len(result["output_heads"]) == 1:
        result["output"] = result["output_heads"][0]
    return result


def run_trace_from_checkpoint(
    checkpoint_path: pathlib.Path,
    data_root: pathlib.Path,
    dataset: str,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    bundle = checkpoint["bundle"]
    hidden_profile = checkpoint_hidden_profile(bundle)
    k_out = int(bundle.get("K_out", 1))
    tensors = load_dataset(data_root, dataset, device)
    model = PlanetoidGAT(
        tensors.num_features,
        num_classes=tensors.num_classes,
        hidden_profile=hidden_profile,
        k_out=k_out,
    ).to(device)
    model.load_state_dict(
        structured_checkpoint_state(
            checkpoint["state_dict"],
            hidden_profile,
            k_out,
            tensors.num_classes,
        )
    )
    return _run_trace(model, tensors)


def run_trace_from_bundle(
    bundle_dir: pathlib.Path,
    data_root: pathlib.Path,
    dataset: str,
    device: torch.device,
) -> Dict[str, object]:
    bundle, state = load_bundle_state(bundle_dir)
    tensors = load_dataset(data_root, dataset, device)
    model = PlanetoidGAT(
        tensors.num_features,
        num_classes=tensors.num_classes,
        hidden_profile=bundle["hidden_profile"],
        k_out=int(bundle["K_out"]),
    ).to(device)
    model.load_state_dict(state)
    return _run_trace(model, tensors)


def _resolve_output_dir(
    project_root: pathlib.Path,
    cfg: Mapping[str, str],
    dataset: str,
    cli_output_dir: str | None,
) -> pathlib.Path:
    if cli_output_dir is not None:
        return pathlib.Path(cli_output_dir).expanduser().resolve()
    return (
        project_root
        / cfg.get(
            "checkpoint_bundle",
            f"artifacts/checkpoints/{dataset}_gat",
        )
    ).resolve()


def _override(value: object | None, fallback: object) -> object:
    return fallback if value is None else value


def add_training_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-dir",
        help="canonical checkpoint destination (defaults to checkpoint_bundle)",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing best_model.pt in the selected output directory",
    )


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
    output_dir = _resolve_output_dir(
        project_root, cfg, dataset, args.output_dir
    )
    checkpoint_path = output_dir / "best_model.pt"
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {checkpoint_path}; pass a fresh --output-dir "
            "or explicitly use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_profile = require_canonical_model_config(cfg, dataset)
    k_out = 2
    epochs = int(_override(args.epochs, cfg_int(cfg, "epochs", 500)))
    patience = int(_override(args.patience, cfg_int(cfg, "patience", 100)))
    learning_rate = float(
        _override(args.learning_rate, cfg_float(cfg, "learning_rate", 0.005))
    )
    weight_decay = float(
        _override(args.weight_decay, cfg_float(cfg, "weight_decay", 0.0005))
    )
    dropout = float(_override(args.dropout, cfg_float(cfg, "dropout", 0.6)))
    seed = int(_override(args.seed, cfg_int(cfg, "seed", 11)))
    use_cuda = bool(_override(args.use_cuda, True))
    expected_training = {
        "cora": (500, 100, 0.005, 0.0005, 0.6),
        "citeseer": (500, 100, 0.005, 0.0005, 0.6),
        "pubmed": (500, 100, 0.01, 0.001, 0.6),
    }[dataset]
    actual_training = (epochs, patience, learning_rate, weight_decay, dropout)
    if actual_training != expected_training or seed != 11:
        raise ValueError(
            f"{dataset} formal training requires epochs/patience/lr/weight_decay/"
            f"dropout={expected_training} and seed=11"
        )
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and patience must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if not use_cuda:
        raise RuntimeError(
            "formal training requires CUDA; refusing to create a CPU checkpoint"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training is required but torch.cuda.is_available() is false"
        )

    device = torch.device("cuda" if use_cuda else "cpu")
    set_seed(seed)
    tensors = load_dataset(data_root, dataset, device)
    model = PlanetoidGAT(
        tensors.num_features,
        num_classes=tensors.num_classes,
        hidden_profile=hidden_profile,
        k_out=k_out,
    ).to(device)
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
        logits, _ = model(
            tensors.features,
            tensors.edge_src,
            tensors.edge_dst,
            dropout=dropout,
            training=True,
        )
        loss = masked_loss(logits, tensors.labels, tensors.train_mask)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(
                tensors.features,
                tensors.edge_src,
                tensors.edge_dst,
                dropout=0.0,
                training=False,
            )
            train_accuracy = masked_accuracy(
                eval_logits, tensors.labels, tensors.train_mask
            )
            validation_loss = float(
                masked_loss(eval_logits, tensors.labels, tensors.val_mask).item()
            )
            validation_accuracy = masked_accuracy(
                eval_logits, tensors.labels, tensors.val_mask
            )
            test_accuracy = masked_accuracy(
                eval_logits, tensors.labels, tensors.test_mask
            )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(loss.item()),
                "train_acc": train_accuracy,
                "val_loss": validation_loss,
                "val_acc": validation_accuracy,
                "test_acc": test_accuracy,
            }
        )
        print(
            f"[{dataset}] epoch={epoch:04d} train_loss={loss.item():.6f} "
            f"train_acc={train_accuracy:.4f} val_loss={validation_loss:.6f} "
            f"val_acc={validation_accuracy:.4f} test_acc={test_accuracy:.4f}",
            flush=True,
        )
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_metrics = {
                "best_epoch": epoch,
                "train_loss": float(loss.item()),
                "train_acc": train_accuracy,
                "val_loss": validation_loss,
                "val_acc": validation_accuracy,
                "test_acc": test_accuracy,
            }
            torch.save(
                {"bundle": metadata, "state_dict": best_state},
                checkpoint_path,
            )
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
        best_logits, _ = model(
            tensors.features,
            tensors.edge_src,
            tensors.edge_dst,
            dropout=0.0,
            training=False,
        )
    np.save(
        output_dir / "best_logits.npy",
        best_logits.detach().cpu().numpy().astype(np.float32, copy=False),
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "device": str(device),
                "wall_time_sec": time.perf_counter() - start,
                "best_metrics": best_metrics,
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
