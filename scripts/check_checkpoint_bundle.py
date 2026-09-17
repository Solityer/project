#!/usr/bin/env python3
"""Strictly check one exported checkpoint bundle against a formal dataset shape."""
from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Iterable, Tuple


from checkpoint_bundle import (
    load_checkpoint_manifest,
    validate_checkpoint_manifest,
    validate_checkpoint_metadata,
    validate_tensor_encodings,
)
from quantization_scale_chain import (
    OGBN_ARXIV_INTERNAL_NAME,
    canonical_input_preprocess_id,
    canonical_quantization_scale_chain,
)


EXPECTED: Dict[str, Tuple[Tuple[int, ...], int, str, str, str, str]] = {
    "cora": ((1433, 64), 7, "gat_family_L3_hid8x8-8x8_kout2_c7", "cora_planetoid_seed11", "transductive_node_classification", "node"),
    "citeseer": ((3703, 64), 6, "gat_family_L3_hid8x8-8x8_kout2_c6", "citeseer_planetoid_seed11", "transductive_node_classification", "node"),
    "pubmed": ((500, 64), 3, "gat_family_L3_hid8x8-8x8_kout2_c3", "pubmed_planetoid_seed11", "transductive_node_classification", "node"),
    "ppi": ((50, 64), 121, "gat_family_L3_hid8x8-8x8_kout2_c121", "ppi_gat_seed11", "inductive_multi_graph_node_classification", "graph"),
    OGBN_ARXIV_INTERNAL_NAME: ((128, 64), 40, "gat_family_L3_hid8x8-8x8_kout2_c40", "ogbn_arxiv_seed11", "transductive_node_classification", "node"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def tensor_shape(tensor_index: object, name: object) -> tuple[int, ...]:
    require(isinstance(tensor_index, dict), "manifest tensor_index must be an object")
    require(isinstance(name, str) and name in tensor_index, f"manifest missing tensor {name}")
    entry = tensor_index[name]
    require(isinstance(entry, dict) and isinstance(entry["shape"], list), f"manifest missing shape for {name}")
    return tuple(int(value) for value in entry["shape"])


def head_names(spec: object, *, require_output_bias: bool) -> Iterable[str]:
    require(isinstance(spec, dict), "head spec must be an object")
    for forbidden_field in ("attn_dst_bias", "attn_src_bias"):
        require(
            forbidden_field not in spec,
            f"formal head spec must reject attention scalar bias: {forbidden_field}",
        )
    names = ["seq_kernel", "attn_dst_kernel", "attn_src_kernel"]
    if require_output_bias:
        names.append("output_bias")
    else:
        require("output_bias" not in spec, "hidden head spec must not expose output_bias")
    for name in names:
        require(name in spec, f"head spec missing {name}")
        yield str(spec[name])


def check_head(
    spec: object,
    tensor_index: object,
    input_dim: int,
    output_dim: int,
    *,
    require_output_bias: bool,
) -> None:
    names = list(head_names(spec, require_output_bias=require_output_bias))
    require(tensor_shape(tensor_index, names[0]) == (1, input_dim, output_dim), "seq kernel shape mismatch")
    require(tensor_shape(tensor_index, names[1]) == (1, output_dim, 1), "destination attention shape mismatch")
    require(tensor_shape(tensor_index, names[2]) == (1, output_dim, 1), "source attention shape mismatch")
    if require_output_bias:
        require(tensor_shape(tensor_index, names[3]) == (output_dim,), "output bias shape mismatch")


def check_source_checkpoint(
    checkpoint: pathlib.Path,
    manifest: Dict[str, object],
    dataset: str,
    d_in_profile: Tuple[int, ...],
    classes: int,
    arch_id: str,
    model_param_id: str,
) -> None:
    import torch

    source = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    require(isinstance(source, dict), "source checkpoint must be an object")
    bundle = source["bundle"] if "bundle" in source else None
    state = source["state_dict"] if "state_dict" in source else None
    require(isinstance(bundle, dict), "source checkpoint is missing bundle metadata")
    require(isinstance(state, dict), "source checkpoint is missing state_dict")
    validate_checkpoint_metadata(bundle)
    require(bundle["dataset"] == dataset, "source checkpoint dataset mismatch")
    require(
        bundle["family_schema"] == "multi_layer_multi_head",
        "source checkpoint family schema mismatch",
    )
    require(bundle["L"] == 3, "source checkpoint L mismatch")
    require(bundle["d_in_profile"] == list(d_in_profile), "source checkpoint d_in_profile mismatch")
    require(bundle["hidden_profile"] == [{"head_count": 8, "head_dim": 8}, {"head_count": 8, "head_dim": 8}], "source checkpoint hidden_profile mismatch")
    require(bundle["K_out"] == 2, "source checkpoint K_out mismatch")
    require(bundle["C"] == classes, "source checkpoint class count mismatch")
    require(bundle["model_arch_id"] == arch_id, "source checkpoint model_arch_id mismatch")
    require(
        bundle["input_preprocess_id"] == canonical_input_preprocess_id(dataset),
        "source checkpoint input_preprocess_id mismatch",
    )
    require(bundle["quant_cfg_id"] == "gat_l3_scale_chain", "source checkpoint quantization metadata mismatch")
    expected_chain = canonical_quantization_scale_chain(
        bundle["quant_cfg_id"],
        bundle["hidden_profile"],
        bundle["K_out"],
    )
    require(
        bundle["quantization_scale_chain"] == expected_chain.canonical_serialization(),
        "source checkpoint complete quantization_scale_chain metadata mismatch",
    )
    require(bundle["model_param_id"] == model_param_id, "source checkpoint model parameter identity mismatch")
    if dataset == "ppi":
        require(bundle["label_format"] == "multi_hot", "source checkpoint PPI label_format mismatch")
        require(bundle["loss"] == "BCEWithLogitsLoss", "source checkpoint PPI loss metadata mismatch")
        require(bundle["selection_metric"] == "validation_loss", "source checkpoint PPI selection metric mismatch")
        require(bundle["split_format"] == "node_masks", "source checkpoint PPI split metadata mismatch")
    require(
        manifest["checkpoint_prefix"]
        in {str(checkpoint.resolve()), str(checkpoint)},
        "source checkpoint provenance does not match the bundle check path",
    )

    expected: Dict[str, Tuple[int, ...]] = {}
    for layer_index, input_dim in enumerate(d_in_profile):
        for head_index in range(8):
            prefix = f"hidden_layers.{layer_index}.heads.{head_index}"
            expected.update(
                {
                    f"{prefix}.seq.weight": (8, input_dim),
                    f"{prefix}.attn_dst.weight": (1, 8),
                    f"{prefix}.attn_src.weight": (1, 8),
                }
            )
    for head_index in range(2):
        prefix = f"output_heads.{head_index}"
        expected.update(
            {
                f"{prefix}.seq.weight": (classes, 64),
                f"{prefix}.attn_dst.weight": (1, classes),
                f"{prefix}.attn_src.weight": (1, classes),
                f"{prefix}.output_bias": (classes,),
            }
        )
    require(set(state) == set(expected), "source checkpoint state_dict keys do not match canonical GAT")
    for name, shape in expected.items():
        value = state[name]
        require(isinstance(value, torch.Tensor), f"source checkpoint tensor is invalid: {name}")
        require(tuple(value.shape) == shape, f"source checkpoint tensor shape mismatch: {name}")


def check_bundle(bundle_dir: pathlib.Path, dataset: str, checkpoint: pathlib.Path | None) -> None:
    d_in_profile, classes, arch_id, model_param_id, task_type, report_unit = EXPECTED[dataset]
    manifest_path = bundle_dir / "manifest.json"
    require(manifest_path.exists(), f"missing {manifest_path}")
    require((bundle_dir / "tensors.txt").exists(), f"missing {bundle_dir / 'tensors.txt'}")
    require((bundle_dir / "tensors.npz").exists(), f"missing {bundle_dir / 'tensors.npz'}")
    if checkpoint is not None:
        require(checkpoint.exists() and checkpoint.is_file(), f"missing source checkpoint {checkpoint}")
    manifest = load_checkpoint_manifest(manifest_path)
    validate_checkpoint_manifest(manifest)
    require(manifest["L"] == 3, "formal bundle requires L=3")
    require(manifest["K_out"] == 2, "formal bundle requires K_out=2")
    require(manifest["C"] == classes, "bundle class count mismatch")
    require(manifest["d_in_profile"] == list(d_in_profile), "bundle d_in_profile mismatch")
    require(manifest["hidden_profile"] == [{"head_count": 8, "head_dim": 8}, {"head_count": 8, "head_dim": 8}], "bundle hidden_profile mismatch")
    require(manifest["model_arch_id"] == arch_id, "bundle model_arch_id mismatch")
    require(manifest["model_param_id"] == model_param_id, "bundle model_param_id mismatch")
    require(manifest["dataset"] == dataset, "bundle dataset identity mismatch")
    require(
        manifest["input_preprocess_id"] == canonical_input_preprocess_id(dataset),
        "bundle input_preprocess_id mismatch",
    )
    expected_chain = canonical_quantization_scale_chain(
        manifest["quant_cfg_id"],
        manifest["hidden_profile"],
        manifest["K_out"],
    )
    require(
        manifest["quantization_scale_chain"] == expected_chain.canonical_serialization(),
        "bundle complete quantization_scale_chain metadata mismatch",
    )
    for spec in [*manifest["hidden_head_specs"], *manifest["output_head_specs"]]:
        require(isinstance(spec, dict), "head spec must be an object")
        require("attn_dst_bias" not in spec and "attn_src_bias" not in spec,
                "attention scalar bias fields are forbidden")
    require(manifest["task_type"] == task_type, "bundle task_type mismatch")
    require(manifest["report_unit"] == report_unit, "bundle report_unit mismatch")
    require(manifest["output_average_rule"] == "per_head_bias_then_arithmetic_mean", "bundle output average rule mismatch")
    if dataset == "ppi":
        require(manifest["label_format"] == "multi_hot", "PPI bundle label_format mismatch")
        require(manifest["loss"] == "BCEWithLogitsLoss", "PPI bundle loss metadata mismatch")
        require(manifest["selection_metric"] == "validation_loss", "PPI bundle selection metric mismatch")
        require(manifest["split_format"] == "node_masks", "PPI bundle split metadata mismatch")
    if checkpoint is not None:
        check_source_checkpoint(
            checkpoint,
            manifest,
            dataset,
            d_in_profile,
            classes,
            arch_id,
            model_param_id,
        )
    tensor_index = manifest["tensor_index"]
    hidden_layers = manifest["hidden_layers"]
    hidden_specs = manifest["hidden_head_specs"]
    output_specs = manifest["output_head_specs"]
    require(isinstance(hidden_layers, list) and len(hidden_layers) == 2, "bundle must contain two hidden layers")
    require(isinstance(hidden_specs, list) and len(hidden_specs) == 16, "bundle must contain sixteen hidden heads")
    require(isinstance(output_specs, list) and len(output_specs) == 2, "bundle must contain two output heads")
    require(all(isinstance(layer, dict) for layer in hidden_layers), "hidden layer entries must be objects")
    require(all(isinstance(spec, dict) for spec in hidden_specs), "hidden head entries must be objects")
    require(all(isinstance(spec, dict) for spec in output_specs), "output head entries must be objects")
    # Formal checkpoint identity is the explicit (layer_index, local_head_index)
    # pair.  A global-only entry is rejected fail-closed.
    expected_pairs = [(layer_index, head_index) for layer_index in range(2) for head_index in range(8)]
    actual_pairs = []
    for spec in hidden_specs:
        require("layer_index" in spec and "local_head_index" in spec, "hidden head spec must expose layer_index and local_head_index")
        actual_pairs.append((int(spec["layer_index"]), int(spec["local_head_index"])))
    require(actual_pairs == expected_pairs, "hidden head specs must use canonical layer/local pair order")
    require(
        [
            (int(layer["input_dim"]), int(layer["head_count"]), int(layer["head_dim"]))
            for layer in hidden_layers
        ]
        == [(d_in_profile[0], 8, 8), (d_in_profile[1], 8, 8)],
        "bundle hidden layer shape mismatch",
    )
    for layer_index, input_dim in enumerate(d_in_profile):
        layer_heads = [spec for spec in hidden_specs if int(spec["layer_index"]) == layer_index]
        require(len(layer_heads) == 8, f"bundle hidden layer {layer_index} head count mismatch")
        for spec in layer_heads:
            check_head(spec, tensor_index, input_dim, 8, require_output_bias=False)
    for spec in output_specs:
        check_head(spec, tensor_index, 64, classes, require_output_bias=True)

    validate_tensor_encodings(
        manifest,
        bundle_dir / "tensors.npz",
        bundle_dir / "tensors.txt",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(EXPECTED))
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    check_bundle(
        pathlib.Path(args.bundle_dir).resolve(),
        args.dataset,
        pathlib.Path(args.checkpoint).resolve() if args.checkpoint else None,
    )
    print(f"checkpoint bundle shape check passed: {args.dataset}")


if __name__ == "__main__":
    main()
