#!/usr/bin/env python3
"""Tiny, dataset-free checkpoint/export roundtrip for GAT family scaling."""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import numpy as np
import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from export_gat_checkpoint import archive_key, build_family_manifest  # noqa: E402
from export_torch_gat_bundle import build_tensors_and_manifest_spec  # noqa: E402
from check_planetoid_parity import compare_arrays  # noqa: E402
from train_planetoid_gat import (  # noqa: E402
    OUTPUT_AVERAGE_RULE,
    PlanetoidGAT,
    checkpoint_hidden_profile,
    checkpoint_metadata,
    load_bundle_state,
    load_checkpoint,
    parse_cfg,
    structured_checkpoint_state,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    shape_mismatch = compare_arrays(
        "shape_mismatch",
        np.zeros((2, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
    )
    require(
        not shape_mismatch["passed"]
        and shape_mismatch["shape"] == [2, 1]
        and shape_mismatch["actual_shape"] == [1, 1],
        "parity comparison accepted a broadcastable shape mismatch",
    )

    torch.manual_seed(17)
    hidden_profile = [
        {"head_count": 2, "head_dim": 3},
        {"head_count": 3, "head_dim": 2},
    ]
    model = PlanetoidGAT(
        5,
        num_classes=4,
        hidden_profile=hidden_profile,
        k_out=2,
    )
    state = model.state_dict()
    require(
        all(
            key.startswith("hidden_layers.") or key.startswith("output_heads.")
            for key in state
        ),
        "state_dict contains a non-structured key",
    )
    require(
        "hidden_layers.1.heads.2.seq.weight" in state
        and "output_heads.1.output_bias" in state,
        "structured layer/head state keys are incomplete",
    )

    legacy_model = PlanetoidGAT(5, 3, 4, hidden_heads=2)
    legacy_state = {}
    for key, value in legacy_model.state_dict().items():
        if key == "output_heads.0.output_bias":
            continue
        if key.startswith("hidden_layers.0.heads."):
            legacy_key = "hidden_heads." + key[len("hidden_layers.0.heads."):]
        elif key.startswith("output_heads.0."):
            legacy_key = "output_head." + key[len("output_heads.0."):]
        else:
            raise AssertionError(f"unexpected compatibility key: {key}")
        legacy_state[legacy_key] = value
    legacy_profile = checkpoint_hidden_profile(
        {"hidden_heads": 2, "hidden_dim": 3}
    )
    normalized_legacy_state = structured_checkpoint_state(
        legacy_state, legacy_profile, 1, 4
    )
    require(
        set(normalized_legacy_state) == set(legacy_model.state_dict()),
        "legacy parity checkpoint normalization changed state keys",
    )

    features = torch.randn(7, 5)
    edge_src = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 0, 2, 4, 6], dtype=torch.long
    )
    edge_dst = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 1, 3, 5, 0], dtype=torch.long
    )
    model.eval()
    with torch.no_grad():
        logits, trace = model(
            features,
            edge_src,
            edge_dst,
            dropout=0.0,
            training=False,
            return_trace=True,
        )
    assert trace is not None
    require(tuple(logits.shape) == (7, 4), "unexpected logits shape")
    require(
        len(trace["hidden_layers"]) == 2
        and len(trace["hidden_layers"][0]["heads"]) == 2
        and len(trace["hidden_layers"][1]["heads"]) == 3,
        "heterogeneous hidden profile was not executed",
    )
    require(
        len(trace["output_heads"]) == 2,
        "K_out=2 output traces are missing",
    )
    expected_logits = torch.stack(trace["output_head_values"], dim=0).mean(dim=0)
    require(
        torch.equal(logits.cpu(), expected_logits),
        "output heads were not averaged after per-head bias",
    )

    metadata = checkpoint_metadata(
        dataset="synthetic",
        in_dim=5,
        hidden_profile=hidden_profile,
        k_out=2,
        num_classes=4,
        seed=17,
        dropout=0.0,
        learning_rate=0.005,
        weight_decay=0.0005,
        model_param_id="synthetic_scaling_roundtrip",
    )
    require(
        metadata["L"] == 3
        and metadata["d_in_profile"] == [5, 6]
        and metadata["hidden_profile"] == hidden_profile
        and metadata["K_out"] == 2
        and metadata["C"] == 4
        and metadata["output_average_rule"] == OUTPUT_AVERAGE_RULE,
        "checkpoint metadata does not describe the scaling architecture",
    )

    with tempfile.TemporaryDirectory(prefix="gatzk_torch_scaling_") as raw_dir:
        temp_dir = pathlib.Path(raw_dir)
        duplicate_cfg = temp_dir / "duplicate.cfg"
        duplicate_cfg.write_text(
            "dataset=cora\ndataset=pubmed\n", encoding="utf-8"
        )
        try:
            parse_cfg(duplicate_cfg)
        except ValueError as error:
            require(
                "duplicate config key" in str(error),
                "duplicate config rejection used the wrong error",
            )
        else:
            raise AssertionError("duplicate Python config key was accepted")

        checkpoint_path = temp_dir / "model.pt"
        bundle_dir = temp_dir / "bundle"
        bundle_dir.mkdir()
        torch.save(
            {"bundle": metadata, "state_dict": model.state_dict()},
            checkpoint_path,
        )
        loaded = load_checkpoint(checkpoint_path, torch.device("cpu"))
        require(loaded["bundle"] == metadata, "checkpoint metadata changed")

        (
            tensors,
            d_in_profile,
            hidden_layers,
            hidden_head_specs,
            output_head_specs,
            num_classes,
            exported_metadata,
        ) = build_tensors_and_manifest_spec(checkpoint_path)
        require(
            d_in_profile == [5, 6]
            and len(hidden_layers) == 2
            and len(hidden_head_specs) == 5
            and len(output_head_specs) == 2
            and num_classes == 4,
            "exporter did not preserve structured architecture",
        )
        manifest = build_family_manifest(
            tensors,
            str(checkpoint_path),
            d_in_profile=d_in_profile,
            hidden_layers=hidden_layers,
            hidden_head_specs=hidden_head_specs,
            output_head_specs=output_head_specs,
            num_classes=num_classes,
            task_type=str(exported_metadata["task_type"]),
            report_unit=str(exported_metadata["report_unit"]),
            model_arch_id=str(exported_metadata["model_arch_id"]),
            model_param_id=str(exported_metadata["model_param_id"]),
        )
        np.savez_compressed(
            bundle_dir / "tensors.npz",
            **{archive_key(name): value for name, value in tensors.items()},
        )
        (bundle_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        bundle_metadata, roundtrip_state = load_bundle_state(bundle_dir)
        require(
            bundle_metadata["L"] == 3
            and bundle_metadata["d_in_profile"] == [5, 6]
            and bundle_metadata["K_out"] == 2,
            "exported bundle metadata changed",
        )
        require(
            set(roundtrip_state) == set(state),
            "exported bundle state keys changed",
        )
        for key, expected in state.items():
            require(
                torch.equal(expected.cpu(), roundtrip_state[key]),
                f"exported tensor changed: {key}",
            )

        roundtrip_model = PlanetoidGAT(
            5,
            num_classes=4,
            hidden_profile=hidden_profile,
            k_out=2,
        )
        roundtrip_model.load_state_dict(roundtrip_state)
        roundtrip_model.eval()
        with torch.no_grad():
            roundtrip_logits, _ = roundtrip_model(
                features,
                edge_src,
                edge_dst,
                dropout=0.0,
                training=False,
            )
        require(
            torch.equal(logits, roundtrip_logits),
            "checkpoint/export roundtrip changed synthetic logits",
        )

    print("torch GAT scaling roundtrip passed")


if __name__ == "__main__":
    main()
