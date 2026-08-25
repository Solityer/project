from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .rounding import validate_rounding_mode


QUANT_SCALE_PROFILES: Dict[str, Dict[str, int]] = {
    "exact_integer_gat_v1": {
        "input": 1,
        "hidden_weight": 16,
        "hidden_attention": 16,
        "output_weight": 16,
        "output_attention": 1,
        "output_bias": 16,
        "lrelu_input": 256,
        "lrelu_output": 256,
        "exp_input": 256,
        "exp_output": 16,
        "elu_input": 16,
        "elu_output": 16,
        "aggregate_output": 16,
        "output_head": 16,
    },
    # Raw checkpoint manifests use this source-format name.  Formal configs
    # authenticate the exact-integer profile above; both names intentionally
    # resolve to the same explicit scale chain.
    "fp32_bundle_export": {
        "input": 1,
        "hidden_weight": 16,
        "hidden_attention": 16,
        "output_weight": 16,
        "output_attention": 1,
        "output_bias": 16,
        "lrelu_input": 256,
        "lrelu_output": 256,
        "exp_input": 256,
        "exp_output": 16,
        "elu_input": 16,
        "elu_output": 16,
        "aggregate_output": 16,
        "output_head": 16,
    },
}


@dataclass
class AttentionHead:
    seq_kernel: np.ndarray
    attn_dst_kernel: np.ndarray
    attn_src_kernel: np.ndarray
    output_bias: np.ndarray


@dataclass
class HiddenLayer:
    layer_index: int
    input_dim: int
    head_count: int
    head_dim: int
    heads: List[AttentionHead]


@dataclass
class Model:
    manifest: Dict
    hidden_layers: List[HiddenLayer]
    output_heads: List[AttentionHead]
    class_count: int
    k_out: int
    model_arch_id: str
    model_param_id: str
    quant_cfg_id: str
    static_table_id: str
    output_average_rule: str
    quant_scales: Dict[str, int]
    # Manual/synthetic models retain the canonical nearest mode for backwards
    # compatible construction; formal loaders always replace it with the
    # explicit config value and validate it.
    rounding_mode: str = "nearest_ties_away"


def _load_text_tensors(path: pathlib.Path) -> Dict[str, np.ndarray]:
    tokens = iter(path.read_text(encoding="utf-8").split())
    out: Dict[str, np.ndarray] = {}
    while True:
        try:
            marker = next(tokens)
        except StopIteration:
            break
        if marker != "TENSOR":
            raise ValueError(f"malformed tensor marker in {path}: {marker}")
        name = next(tokens)
        rank = int(next(tokens))
        shape = [int(next(tokens)) for _ in range(rank)]
        count = int(next(tokens))
        values = [float(next(tokens)) for _ in range(count)]
        array = np.asarray(values, dtype=np.float64)
        if int(np.prod(shape)) != count:
            raise ValueError(f"tensor count mismatch for {name}")
        out[name] = array.reshape(shape)
    return out


def _reshape_seq(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(f"expected seq kernel shape [1,in,out], got {array.shape}")
    return np.asarray(array[0], dtype=np.float64)


def _reshape_attn(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] == 1 and array.shape[2] == 1:
        return np.asarray(array[0, :, 0], dtype=np.float64)
    if array.ndim == 1:
        return np.asarray(array, dtype=np.float64)
    raise ValueError(f"expected attention kernel shape [1,d,1], got {array.shape}")


def _scalar(array: np.ndarray) -> float:
    flat = np.ravel(array)
    if flat.size != 1:
        raise ValueError(f"expected scalar tensor, got {array.shape}")
    return float(flat[0])


def _bias(array: np.ndarray) -> np.ndarray:
    return np.ravel(array).astype(np.float64)


def _round_ties_away(value: float) -> int:
    return int(np.floor(value + 0.5) if value >= 0.0 else np.ceil(value - 0.5))


def _quantize(values: np.ndarray, scale: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64)
    return np.asarray([_round_ties_away(float(item) * scale) for item in np.ravel(flat)], dtype=np.int64).reshape(flat.shape)


def _head_from_spec(spec: Dict, tensors: Dict[str, np.ndarray], scales: Dict[str, int], output_head: bool = False) -> AttentionHead:
    # These scales are the public fp32_bundle_export quant_cfg used by the
    # formal GAT relation.  Once converted, every forward tensor is an exact
    # integer and no floating-point operation is part of the proof witness.
    return AttentionHead(
        seq_kernel=_quantize(_reshape_seq(tensors[spec["seq_kernel"]]), scales["output_weight"] if output_head else scales["hidden_weight"]),
        attn_dst_kernel=_quantize(_reshape_attn(tensors[spec["attn_dst_kernel"]]), scales["output_attention"] if output_head else scales["hidden_attention"]),
        attn_src_kernel=_quantize(_reshape_attn(tensors[spec["attn_src_kernel"]]), scales["output_attention"] if output_head else scales["hidden_attention"]),
        output_bias=_quantize(_bias(tensors[spec["output_bias"]]), scales["output_bias"]) if output_head else np.zeros_like(_bias(tensors[spec["output_bias"]]), dtype=np.int64),
    )


def load_model(
    bundle: pathlib.Path,
    *,
    model_arch_id: str | None = None,
    model_param_id: str | None = None,
    quant_cfg_id: str | None = None,
    static_table_id: str | None = None,
    rounding_mode: str | None = None,
) -> Model:
    """Load and validate the raw checkpoint, with optional formal identities.

    The manifest remains the authenticated checkpoint source.  Formal runs
    may bind baseline-local parameter/quantization/table identities from their
    immutable config while retaining the raw manifest IDs in ``manifest``;
    default arguments preserve the historical raw-bundle behavior for tools
    that do not have a formal config.
    """
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("output_average_rule") != "per_head_bias_then_arithmetic_mean":
        raise ValueError("output layer must average per-head post-bias values")
    tensors = _load_text_tensors(bundle / "tensors.txt")
    resolved_quant_cfg = str(manifest["quant_cfg_id"] if quant_cfg_id is None else quant_cfg_id)
    if resolved_quant_cfg not in QUANT_SCALE_PROFILES:
        raise ValueError(f"unsupported authenticated quantization profile: {resolved_quant_cfg}")
    quant_scales = dict(QUANT_SCALE_PROFILES[resolved_quant_cfg])
    resolved_rounding_mode = validate_rounding_mode(
        manifest.get("rounding_mode") if rounding_mode is None else rounding_mode
    )

    hidden_layers: List[HiddenLayer] = []
    for layer_info in manifest["hidden_layers"]:
        specs = [
            item for item in manifest["hidden_head_specs"]
            if int(item["layer_index"]) == int(layer_info["layer_index"])
        ]
        specs.sort(key=lambda item: int(item["local_head_index"]))
        heads = [_head_from_spec(spec, tensors, quant_scales, output_head=False) for spec in specs]
        hidden_layers.append(
            HiddenLayer(
                layer_index=int(layer_info["layer_index"]),
                input_dim=int(layer_info["input_dim"]),
                head_count=int(layer_info["head_count"]),
                head_dim=int(layer_info["head_dim"]),
                heads=heads,
            )
        )
    hidden_layers.sort(key=lambda layer: layer.layer_index)

    output_specs = sorted(manifest["output_head_specs"], key=lambda item: int(item["head_index"]))
    output_heads = [_head_from_spec(spec, tensors, quant_scales, output_head=True) for spec in output_specs]

    model = Model(
        manifest=manifest,
        hidden_layers=hidden_layers,
        output_heads=output_heads,
        class_count=int(manifest["C"]),
        k_out=int(manifest["K_out"]),
        model_arch_id=str(manifest["model_arch_id"] if model_arch_id is None else model_arch_id),
        model_param_id=str(manifest["model_param_id"] if model_param_id is None else model_param_id),
        quant_cfg_id=resolved_quant_cfg,
        static_table_id=str(manifest["static_table_id"] if static_table_id is None else static_table_id),
        output_average_rule=str(manifest["output_average_rule"]),
        quant_scales=quant_scales,
        rounding_mode=resolved_rounding_mode,
    )
    _validate_model(model)
    return model


def _validate_model(model: Model) -> None:
    if model.k_out != len(model.output_heads):
        raise ValueError("K_out conflicts with output_head_specs")
    prev_dim = int(model.manifest["d_in_profile"][0])
    for layer in model.hidden_layers:
        if layer.input_dim != prev_dim:
            raise ValueError("hidden layer input dimension chain mismatch")
        if layer.head_count != len(layer.heads):
            raise ValueError("hidden layer head count mismatch")
        for head in layer.heads:
            if head.seq_kernel.shape != (layer.input_dim, layer.head_dim):
                raise ValueError(f"bad hidden seq shape: {head.seq_kernel.shape}")
            if head.attn_src_kernel.shape[0] != layer.head_dim or head.attn_dst_kernel.shape[0] != layer.head_dim:
                raise ValueError("hidden attention vector dimension mismatch")
        prev_dim = layer.head_count * layer.head_dim
    for head in model.output_heads:
        if head.seq_kernel.shape != (prev_dim, model.class_count):
            raise ValueError(f"bad output seq shape: {head.seq_kernel.shape}")
        if head.output_bias.shape[0] != model.class_count:
            raise ValueError("output bias dimension mismatch")
