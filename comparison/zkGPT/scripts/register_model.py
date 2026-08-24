#!/usr/bin/env python3
"""One-time model preprocessing for the zkGPT GAT baseline."""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.config import generated_config, load_app_config  # noqa: E402
from zkgpt_gat.data import load_dataset, normalize_graph  # noqa: E402
from zkgpt_gat.forward import forward_full  # noqa: E402
from zkgpt_gat.model import load_model  # noqa: E402
from zkgpt_gat.proof import _padded_segment_chunks, make_metadata, prepare_input_segments_and_layout  # noqa: E402
from zkgpt_gat.registry import ensure_model_registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Register canonical zkGPT GAT model weights")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--project-root", type=pathlib.Path, default=PROJECT)
    parser.add_argument("--zk-root", type=pathlib.Path, default=ROOT)
    args = parser.parse_args()
    config_path = generated_config(args.dataset, "cpu", args.project_root.resolve(), args.zk_root.resolve())
    config = load_app_config(config_path, args.project_root.resolve())
    dataset = load_dataset(config)
    graph = normalize_graph(dataset, config)
    model = load_model(
        config.checkpoint_bundle,
        model_arch_id=config.entries.get("model_arch_id"),
        model_param_id=config.entries.get("model_param_id"),
        quant_cfg_id=config.entries.get("quant_cfg_id"),
        static_table_id=config.entries.get("static_table_id"),
    )
    trace = forward_full(graph, model)
    _, input_layout, padded_segments = prepare_input_segments_and_layout(config, dataset, graph, model, trace)
    metadata = make_metadata(config, dataset, graph, model, "cpu", {"backend": "cpu"}, input_layout)
    registry = ensure_model_registry(args.zk_root.resolve(), metadata, list(_padded_segment_chunks("com_W", padded_segments["W"])))
    print(registry["public_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
