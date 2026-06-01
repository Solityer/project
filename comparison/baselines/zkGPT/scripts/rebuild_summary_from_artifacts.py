#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, List


def main() -> None:
    zk_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(zk_root / "src"))

    from zkgpt_gat.constants import DATASET_ORDER
    from zkgpt_gat.runner import write_summary
    from zkgpt_gat.util import digest_json, read_json

    results_dir = zk_root / "results"
    rows: List[Dict[str, Any]] = []
    for backend in ["cpu", "gpu"]:
        for dataset in DATASET_ORDER:
            artifact_dir = results_dir / backend / dataset
            manifest = read_json(artifact_dir / "run_manifest.json")
            metadata = read_json(artifact_dir / "metadata.json")
            benchmark = read_json(artifact_dir / "benchmark_summary.json")
            data_id = read_json(artifact_dir / "data_id.json")
            model_id = read_json(artifact_dir / "model_id.json")
            config_id = read_json(artifact_dir / "config_id.json")
            hidden_profile = metadata["dim_cfg"]["hidden_profile"]
            row = {
                "dataset": dataset,
                "backend": backend,
                "status": manifest["status"],
                "N": manifest["N"],
                "E": manifest["E"],
                "G_batch": manifest["G_batch"],
                "L": metadata["dim_cfg"]["L"],
                "hidden_profile": ",".join(f"{item['head_count']}x{item['head_dim']}" for item in hidden_profile),
                "d_in_profile": ",".join(str(item) for item in metadata["dim_cfg"]["d_in"]),
                "K_out": metadata["dim_cfg"]["K_out"],
                "C": metadata["dim_cfg"]["C"],
                "commitment_time_ms": benchmark["commitment_time_ms"],
                "prove_time_ms": benchmark["prove_time_ms"],
                "verify_time_ms": benchmark["verify_time_ms"],
                "proof_size_bytes": benchmark["proof_size_bytes"],
                "config_id": config_id["id"],
                "model_id": model_id["id"],
                "data_id": data_id["id"],
                "artifact_path": str(artifact_dir),
                "baseline_identity": manifest["baseline_identity"],
                "proof_scope": manifest["proof_scope"],
                "scope_mismatch": manifest["scope_mismatch"],
                "full_gat_forward_proved": manifest["full_gat_forward_proved"],
                "final_logits_bound": manifest["final_logits_bound"],
                "setup_assumption": manifest["setup_assumption"],
                "proof_size_definition": manifest["proof_size_definition"],
                "verifier_reads_witness": manifest["verifier_reads_witness"],
                "witness_included_in_proof": manifest["witness_included_in_proof"],
                "public_input_layout": manifest["public_input_layout"],
                "task_type": manifest["task_type"],
                "model_arch": digest_json(manifest["model_arch"]),
                "model_params": digest_json(manifest["model_params"]),
                "same_checkpoint": manifest["same_checkpoint"],
                "same_quantization": manifest["same_quantization"],
                "same_graph_preprocessing": manifest["same_graph_preprocessing"],
                "same_self_loop_rule": manifest["same_self_loop_rule"],
                "same_edge_sort_rule": manifest["same_edge_sort_rule"],
                "same_public_output_policy": manifest["same_public_output_policy"],
                "self_loop_rule": manifest["self_loop_rule"],
                "edge_sort_rule": manifest["edge_sort_rule"],
                "batching_rule": manifest["batching_rule"],
                "hardware_and_device": manifest["hardware_and_device"],
                "run_status": manifest["run_status"],
                "failure_reason_if_not_comparable": manifest["failure_reason_if_not_comparable"],
            }
            rows.append(row)
    write_summary(results_dir, rows)


if __name__ == "__main__":
    main()
