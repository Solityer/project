from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class AttentionHead:
    seq_kernel: np.ndarray
    attn_dst_kernel: np.ndarray
    attn_dst_bias: float
    attn_src_kernel: np.ndarray
    attn_src_bias: float
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


def _head_from_spec(spec: Dict, tensors: Dict[str, np.ndarray]) -> AttentionHead:
    return AttentionHead(
        seq_kernel=_reshape_seq(tensors[spec["seq_kernel"]]),
        attn_dst_kernel=_reshape_attn(tensors[spec["attn_dst_kernel"]]),
        attn_dst_bias=_scalar(tensors[spec["attn_dst_bias"]]),
        attn_src_kernel=_reshape_attn(tensors[spec["attn_src_kernel"]]),
        attn_src_bias=_scalar(tensors[spec["attn_src_bias"]]),
        output_bias=_bias(tensors[spec["output_bias"]]),
    )


def load_model(bundle: pathlib.Path) -> Model:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("output_average_rule") != "per_head_bias_then_arithmetic_mean":
        raise ValueError("output layer must average per-head post-bias values")
    tensors = _load_text_tensors(bundle / "tensors.txt")

    hidden_layers: List[HiddenLayer] = []
    for layer_info in manifest["hidden_layers"]:
        specs = [
            item for item in manifest["hidden_head_specs"]
            if int(item["layer_index"]) == int(layer_info["layer_index"])
        ]
        specs.sort(key=lambda item: int(item["local_head_index"]))
        heads = [_head_from_spec(spec, tensors) for spec in specs]
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
    output_heads = [_head_from_spec(spec, tensors) for spec in output_specs]

    model = Model(
        manifest=manifest,
        hidden_layers=hidden_layers,
        output_heads=output_heads,
        class_count=int(manifest["C"]),
        k_out=int(manifest["K_out"]),
        model_arch_id=str(manifest["model_arch_id"]),
        model_param_id=str(manifest["model_param_id"]),
        quant_cfg_id=str(manifest["quant_cfg_id"]),
        static_table_id=str(manifest["static_table_id"]),
        output_average_rule=str(manifest["output_average_rule"]),
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

