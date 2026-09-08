#!/usr/bin/env python3
"""Test-only checkpoint/bundle forward parity helpers.

The trainer owns the formal model, quantization, checkpoint loader, and
canonical metadata checks.  This module owns only the extra state reconstruction
and trace replay needed to compare a source checkpoint with its exported
bundle.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict, Mapping, Tuple, Union

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from checkpoint_bundle import (  # noqa: E402
    load_checkpoint_manifest,
    validate_checkpoint_manifest,
    validate_tensor_encodings,
)
from quantization_scale_chain import canonical_quantization_scale_chain  # noqa: E402
from train_planetoid_gat import (  # noqa: E402
    DatasetTensors,
    PlanetoidGAT,
    checkpoint_hidden_profile,
    load_checkpoint,
    load_dataset,
)


def structured_checkpoint_state(
    state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Validate and copy the structured state for parity replay only."""
    for key in state:
        if (
            "attn_src.bias" in key
            or "attn_dst.bias" in key
            or "attn_src_bias" in key
            or "attn_dst_bias" in key
        ):
            raise ValueError(
                "bias-enabled attention checkpoint is not accepted by the formal loader"
            )
    if not state or any(
        not key.startswith(("hidden_layers.", "output_heads.")) for key in state
    ):
        raise ValueError("checkpoint state_dict must expose only structured model keys")
    return dict(state)


def load_bundle_state(
    bundle_dir: pathlib.Path,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    """Reconstruct a trainer-compatible state from a canonical bundle."""
    manifest = load_checkpoint_manifest(bundle_dir / "manifest.json")
    validate_checkpoint_manifest(manifest)
    tensor_map = validate_tensor_encodings(
        manifest,
        bundle_dir / "tensors.npz",
        bundle_dir / "tensors.txt",
    )
    manifest_chain = manifest["quantization_scale_chain"]
    if manifest["family_schema"] != "multi_layer_multi_head":
        raise ValueError("formal bundle family_schema is not canonical")
    if manifest["L"] != 3 or manifest["K_out"] != 2:
        raise ValueError("formal bundle must declare the canonical L=3 and K_out=2")
    if not isinstance(manifest_chain, str) or not manifest_chain:
        raise ValueError(
            "formal bundle is missing quantization_scale_chain; "
            "quant_cfg_id alone is insufficient"
        )
    expected_chain = canonical_quantization_scale_chain(
        manifest["quant_cfg_id"],
        manifest["hidden_profile"],
        manifest["K_out"],
    )
    if manifest_chain != expected_chain.canonical_serialization():
        raise ValueError("formal bundle quantization_scale_chain is not canonical")

    dataset = manifest["dataset"]
    # Keep the canonical preprocessing identity check in the production
    # checkpoint/bundle validators; this helper only reconstructs tensors.
    from quantization_scale_chain import canonical_input_preprocess_id

    expected_preprocess_id = canonical_input_preprocess_id(dataset)
    if manifest["input_preprocess_id"] != expected_preprocess_id:
        raise ValueError("formal bundle input_preprocess_id does not match its dataset")
    hidden_profile = manifest["hidden_profile"]
    d_in_profile = list(manifest["d_in_profile"])
    if len(d_in_profile) != len(hidden_profile):
        raise ValueError("bundle d_in_profile must align with hidden_profile")
    k_out = manifest["K_out"]
    num_classes = manifest["C"]

    def tensor(name: str) -> np.ndarray:
        return tensor_map[name]

    state: Dict[str, torch.Tensor] = {}
    for head in manifest["hidden_head_specs"]:
        layer_index = head["layer_index"]
        local_head_index = head["local_head_index"]
        prefix = f"hidden_layers.{layer_index}.heads.{local_head_index}"
        seq = tensor(head["seq_kernel"])
        state[f"{prefix}.seq.weight"] = torch.from_numpy(
            seq.reshape(seq.shape[1], seq.shape[2]).T.copy()
        )
        state[f"{prefix}.attn_dst.weight"] = torch.from_numpy(
            tensor(head["attn_dst_kernel"]).reshape(1, -1)
        )
        state[f"{prefix}.attn_src.weight"] = torch.from_numpy(
            tensor(head["attn_src_kernel"]).reshape(1, -1)
        )

    for head in manifest["output_head_specs"]:
        head_index = head["head_index"]
        prefix = f"output_heads.{head_index}"
        seq = tensor(head["seq_kernel"])
        state[f"{prefix}.seq.weight"] = torch.from_numpy(
            seq.reshape(seq.shape[1], seq.shape[2]).T.copy()
        )
        state[f"{prefix}.attn_dst.weight"] = torch.from_numpy(
            tensor(head["attn_dst_kernel"]).reshape(1, -1)
        )
        state[f"{prefix}.attn_src.weight"] = torch.from_numpy(
            tensor(head["attn_src_kernel"]).reshape(1, -1)
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
        "output_average_rule": manifest["output_average_rule"],
        "quant_cfg_id": manifest["quant_cfg_id"],
        "quantization_scale_chain": manifest_chain,
        "input_preprocess_id": manifest["input_preprocess_id"],
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
    }


def _run_trace(model: PlanetoidGAT, tensors: DatasetTensors) -> Dict[str, object]:
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
    if trace is None:
        raise RuntimeError("parity replay did not return a trace")
    return _numpy_trace(logits, trace)


def run_trace_from_checkpoint(
    checkpoint_path: pathlib.Path,
    data_root: pathlib.Path,
    dataset: str,
    device: torch.device,
    *,
    cache_root: pathlib.Path | None = None,
) -> Dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    bundle = checkpoint["bundle"]
    hidden_profile = checkpoint_hidden_profile(bundle)
    k_out = bundle["K_out"]
    chain = canonical_quantization_scale_chain(
        bundle["quant_cfg_id"], hidden_profile, k_out
    )
    tensors = load_dataset(
        data_root, dataset, device, cache_root=cache_root, chain=chain
    )
    model = PlanetoidGAT(
        tensors.num_features,
        num_classes=tensors.num_classes,
        hidden_profile=hidden_profile,
        k_out=k_out,
        quantization=chain,
    ).to(device)
    model.load_state_dict(structured_checkpoint_state(checkpoint["state_dict"]))
    return _run_trace(model, tensors)


def run_trace_from_bundle(
    bundle_dir: pathlib.Path,
    data_root: pathlib.Path,
    dataset: str,
    device: torch.device,
    *,
    cache_root: pathlib.Path | None = None,
) -> Dict[str, object]:
    bundle, state = load_bundle_state(bundle_dir)
    hidden_profile = bundle["hidden_profile"]
    k_out = bundle["K_out"]
    chain = canonical_quantization_scale_chain(
        bundle["quant_cfg_id"], hidden_profile, k_out
    )
    tensors = load_dataset(
        data_root, dataset, device, cache_root=cache_root, chain=chain
    )
    model = PlanetoidGAT(
        tensors.num_features,
        num_classes=tensors.num_classes,
        hidden_profile=hidden_profile,
        k_out=k_out,
        quantization=chain,
    ).to(device)
    model.load_state_dict(state)
    return _run_trace(model, tensors)
