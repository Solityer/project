#!/usr/bin/env python3
"""Validate installed baseline-local canonical checkpoint snapshots."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.config import load_app_config  # noqa: E402
from zkgpt_gat.constants import CONFIG_BY_DATASET, DATASET_ORDER  # noqa: E402
from zkgpt_gat.model import load_model  # noqa: E402
from zkgpt_gat.resources import canonical_dataset_name, required_resource_paths  # noqa: E402
from zkgpt_gat.util import digest_file  # noqa: E402


def validate_one(dataset_name: str) -> dict[str, object]:
    dataset = canonical_dataset_name(dataset_name)
    config_path = ROOT / "configs" / "formal" / CONFIG_BY_DATASET[dataset]
    config = load_app_config(config_path, ROOT)
    bundle = required_resource_paths(ROOT, dataset)["checkpoint_bundle"]
    model = load_model(
        bundle,
        model_arch_id=config.entries["model_arch_id"],
        model_param_id=config.entries["model_param_id"],
        quant_cfg_id=config.entries["quant_cfg_id"],
        static_table_id=config.entries["static_table_id"],
        rounding_mode=config.rounding_mode,
    )
    if model.input_preprocess_id != config.input_preprocess_id:
        raise RuntimeError("checkpoint preprocessing identity differs from formal config")
    return {
        "dataset": dataset,
        "bundle": str(bundle.relative_to(ROOT)),
        "tensor_count": int(model.manifest["tensor_count"]),
        "logical_payload_sha256": model.manifest["logical_payload_sha256"],
        "manifest_sha256": digest_file(bundle / "manifest.json"),
        "tensors_txt_sha256": digest_file(bundle / "tensors.txt"),
        "tensors_npz_sha256": digest_file(bundle / "tensors.npz"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all")
    args = parser.parse_args()
    selected = DATASET_ORDER if args.datasets == "all" else [
        canonical_dataset_name(item.strip())
        for item in args.datasets.split(",")
        if item.strip()
    ]
    for dataset in selected:
        print(json.dumps(validate_one(dataset), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
