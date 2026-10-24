#!/usr/bin/env python3
"""Exercise formal capacity parsing against a real native CLI and tiny inputs."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import formal_preflight  # noqa: E402
import gat_zkgpt_experiment  # noqa: E402
from common import (  # noqa: E402
    PAPER_TIMING_SEMANTICS,
    REPORT_SCHEMA,
    RESULT_SCHEMA,
    artifact_fields,
    load_json,
    paper_metrics,
    require_result_shape,
    resource_identity,
)

PROVER = ROOT / "build" / "cpu" / "gat_adapter" / "zkgpt_gat_prover"
VERIFIER = ROOT / "build" / "cpu" / "gat_adapter" / "zkgpt_gat_verifier"
NATIVE_TIMEOUT_SECONDS = 120
def _authenticated(directory: Path, names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    entries = [f"{hashlib.sha256((directory / name).read_bytes()).hexdigest()}  {name}\n" for name in names]
    (directory / "integrity.sha256").write_text("".join(entries), encoding="utf-8")


def _config_sidecar(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")


def _write_resources(root: Path) -> None:
    configs = root / "baseline-local" / "configs"
    configs.mkdir(parents=True)
    config = dict(
        dataset="cora", data_root="baseline-local/data", cache_root="baseline-local/data/cache",
        export_dir="baseline-local/runs/cora_secure", checkpoint_bundle="baseline-local/checkpoints/cora_gat",
        task_type="transductive_node_classification", report_unit="node", batching_rule="whole_graph_single",
        self_loop_rule="per_node", edge_sort_rule="edge_gid_then_dst_stable", chunking_rule="none", L=3,
        hidden_profile="[1x1,1x1]", d_in_profile="[1,1]", K_out=1, num_classes=1, range_bits=12,
        seed=11, dump_trace="false", auto_prepare_dataset="false",
        model_arch_id="gat_family_L3_hid1x1-1x1_kout1_c1", model_param_id="synthetic-formal-model-v1",
        quant_cfg_id="gat_l3_scale_chain", input_preprocess_id="identity",
        static_table_id="tables:lrelu+elu+exp+range", degree_bound_id="auto",
        setup_mode="hyrax_native_local", setup_resource_scope="baseline-local",
        privacy_policy="model_and_trace_hidden", input_privacy="public_graph_public_features",
        output_mode="public_logits", opening_mode="point_grouped_batch_opening",
        binding_mode="zk_subproof_for_crpc", proof_task="node_classification_full_gat",
        security_profile_id="zkgpt_secure_profile",
    )
    config_path = configs / "cora_secure.cfg"
    config_path.write_text("".join(f"{key}={value}\n" for key, value in config.items()), encoding="utf-8")
    _config_sidecar(config_path)
    quant_path = configs / "quantization_scale_chain.cfg"
    quant = dict(schema="gat_l3_scale_chain", quant_cfg_id="gat_l3_scale_chain", s_input=16,
                 s_delta=256, s_exp=16, s_h="16,16,16", s_agg_pre="16,16", s_agg="16,16",
                 s_cat="16,16", s_out_pre=16, s_out_head=16, s_out=16, s_w_output="4,4",
                 s_y_prime="64,64", s_a_src_output="4,4", s_a_dst_output="4,4", s_b_output="16,16",
                 round="nearest_ties_away", rescale="none", output_average="signed_round_div",
                 static_table_id="tables:lrelu+elu+exp+range", static_table_generator_id="zkgpt-gat-static-tables-v2",
                 static_row_order="ascending_input", exp_approx_id="exp_neg_quantized",
                 elu_approx_id="elu_expm1_quantized", lrelu_slope_num=1, lrelu_slope_den=3, range_bits=12)
    for layer in range(2):
        for key, value in (("s_w_hidden", 4), ("s_h_prime_hidden", 64), ("s_a_src_hidden", 4), ("s_a_dst_hidden", 4)):
            quant[f"{key}_{layer}"] = ",".join([str(value)] * 8)
    quant_path.write_text("".join(f"{key}={value}\n" for key, value in quant.items()), encoding="utf-8")
    _config_sidecar(quant_path)

    cache = root / "baseline-local" / "data" / "cache" / "cora"
    cache.mkdir(parents=True)
    nodes = 64
    files = {
        "meta.cfg": f"num_nodes={nodes}\nnum_features=1\nnum_classes=1\ngraph_count=1\nlabel_format=class_index\n",
        "edges.txt": "".join(f"{node} {node}\n" for node in range(nodes)),
        "features.txt": f"{nodes} 1\n" + "0:0.25\n" * nodes,
        "labels.txt": "0\n" * nodes,
        "test.index": "0\n",
    }
    for name, contents in files.items():
        (cache / name).write_text(contents, encoding="utf-8")
    _authenticated(cache, list(files))

    checkpoint = root / "baseline-local" / "checkpoints" / "cora_gat"
    checkpoint.mkdir(parents=True)
    tensors = []
    shapes = {
        "hidden/layer0/head0/seq/kernel": (1, 1, 1),
        "hidden/layer0/head0/attn_src/kernel": (1, 1, 1),
        "hidden/layer0/head0/attn_dst/kernel": (1, 1, 1),
        "hidden/layer1/head0/seq/kernel": (1, 1, 1),
        "hidden/layer1/head0/attn_src/kernel": (1, 1, 1),
        "hidden/layer1/head0/attn_dst/kernel": (1, 1, 1),
        "output/head0/seq/kernel": (1, 1, 1),
        "output/head0/attn_src/kernel": (1, 1, 1),
        "output/head0/attn_dst/kernel": (1, 1, 1),
        "output/head0/output_bias": (1,),
    }
    for name, shape in shapes.items():
        count = 1
        for extent in shape:
            count *= extent
        dims = " ".join(str(extent) for extent in shape)
        tensors.append(f"TENSOR {name} {len(shape)} {dims} {count}\n0.25\n")
    (checkpoint / "tensors.txt").write_text("".join(tensors), encoding="utf-8")
    manifest = {
        "dataset": "cora", "model_arch_id": "gat_family_L3_hid1x1-1x1_kout1_c1",
        "quant_cfg_id": "gat_l3_scale_chain", "model_param_id": "synthetic-formal-model-v1",
        "static_table_id": "tables:lrelu+elu+exp+range", "input_preprocess_id": "identity",
        "batching_rule": "whole_graph_single", "edge_sort_rule": "edge_gid_then_dst_stable",
        "self_loop_rule": "per_node", "label_format": "class_index",
        "output_average_rule": "per_head_bias_then_arithmetic_mean",
        "quantization_scale_chain": "gat_l3_scale_chain", "tensor_count": 10,
        "L": 3, "K_out": 1, "C": 1,
    }
    (checkpoint / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("tensors.npz", "best_model.pt", "qat_best_logits.npy", "training_summary.json"):
        (checkpoint / name).write_bytes(b"")
    _authenticated(checkpoint, ["manifest.json", "tensors.txt", "tensors.npz", "best_model.pt",
                                "qat_best_logits.npy", "training_summary.json"])


class T2CapacityConsumerSmoke(unittest.TestCase):
    def test_formal_consumer_accepts_compact_t2_profile_from_native_cli(self) -> None:
        self.assertTrue(PROVER.is_file(), f"build the source-matched CPU prover first: {PROVER}")
        self.assertTrue(VERIFIER.is_file(), f"build the source-matched CPU verifier first: {VERIFIER}")
        with tempfile.TemporaryDirectory(prefix="zkgpt-t2-capacity-") as temporary:
            root = Path(temporary)
            _write_resources(root)
            with mock.patch.object(formal_preflight, "build_binary", return_value=PROVER), \
                    contextlib.redirect_stderr(io.StringIO()):
                payload = formal_preflight.run_capacity(
                    root, "cora", "cpu", root / "logs",
                    timeout_seconds=NATIVE_TIMEOUT_SECONDS,
                )
            self.assertEqual(payload["schema"], "zkgpt-gat-capacity-v12")
            self.assertEqual(payload["status"], "READY", payload.get("reason"))
            self.assertIs(payload["forward_value_admission_ready"], False)
            self.assertEqual(payload["forward_value_admission_mode"], "deferred_to_owned_forward_v1")
            self.assertEqual(payload["canonical_graph_identity"]["nodes"], 64)
            t2 = next(item["workspace"] for item in payload["lasso_workspace_profile"]
                      if item["workspace"]["t2_booleanity"])
            q = t2["padded_queries"]
            qbits = q.bit_length() - 1
            self.assertGreaterEqual(qbits, 4)
            self.assertEqual(t2["private_witness_bytes"], q * t2["private_witness_element_bytes"])
            self.assertEqual(t2["t2_column_one_bytes"], (q // 16) * t2["private_witness_element_bytes"])
            compact_fields = 65536 + max((1 << min(12, qbits - 4)) + qbits - 4, q // 16, 278)
            expected_peak = t2["row_ledger_retained_bytes"] + compact_fields * t2["private_witness_element_bytes"]
            self.assertEqual(t2["booleanity_peak_bytes"], expected_peak)
            l_query_logical = payload["input_segments"][3]["logical_length"]
            l_query_active = 1 << (l_query_logical - 1).bit_length()
            l_query_bits = l_query_active.bit_length() - 1
            l_query_columns = 1 << (l_query_bits // 2)
            l_query_rows = l_query_active // l_query_columns
            g1_bytes = t2["column_generator_bytes"] // t2["columns"]
            expected_lut_bytes = 32 * l_query_columns * g1_bytes \
                if l_query_rows >= 128 and l_query_columns >= 8 and l_query_columns % 8 == 0 else 0
            self.assertGreater(expected_lut_bytes, 0)
            self.assertEqual(payload["hyrax_l_qry_tensor_columns"], l_query_columns)
            self.assertEqual(payload["hyrax_l_qry_binary_lut_workspace_bytes"], expected_lut_bytes)

            session_id = "a" * 32
            run_relative = f"baseline-local/runs/formal/{session_id}/cora"
            run_dir = root / run_relative
            run_dir.mkdir(parents=True)
            config_relative = "baseline-local/configs/cora_secure.cfg"
            registration_relative = f"{run_relative}/model_registration.json"
            secret_relative = f"{run_relative}/com_W.prover.secret"

            def run_native(command: list[str], stage: str) -> subprocess.CompletedProcess[str]:
                completed = subprocess.run(
                    command, cwd=root, text=True, capture_output=True,
                    timeout=NATIVE_TIMEOUT_SECONDS, check=False,
                )
                self.assertEqual(
                    completed.returncode, 0,
                    f"native {stage} failed: stdout={completed.stdout}\nstderr={completed.stderr}",
                )
                return completed

            common_args = [
                "--dataset", "cora", "--backend", "cpu",
                "--baseline-root", str(root), "--config", config_relative,
                "--formal-session-id", session_id,
                "--formal-run-directory", run_relative,
            ]
            run_native([
                str(PROVER), *common_args, "--register-only",
                "--registration", registration_relative,
                "--com-w-secret", secret_relative,
            ], "model registration")
            self.assertTrue((root / registration_relative).is_file())
            self.assertTrue((root / secret_relative).is_file())

            run_native([
                str(PROVER), *common_args,
                "--registration", registration_relative,
                "--com-w-secret", secret_relative,
                "--defer-self-verify",
            ], "deferred proof")
            result_path = run_dir / "result.json"
            result = require_result_shape(load_json(result_path, "native result"), result_path)
            self.assertEqual(result["schema"], RESULT_SCHEMA)
            self.assertEqual(result["paper_timing_semantics"], PAPER_TIMING_SEMANTICS)
            self.assertEqual(result["session_id"], session_id)
            self.assertEqual(result["status"], "PROOF_GENERATED_PENDING_VERIFY")
            self.assertNotIn("run_nonce", result)
            self.assertNotIn("session_manifest_sha256", result)
            self.assertNotIn("artifact_sha256", result)
            self.assertNotIn("artifact_readback", result["timing_ms"])
            artifact = artifact_fields(root, result, run_dir)
            artifact_size = artifact.stat().st_size
            self.assertGreater(artifact_size, 0)
            self.assertEqual(result["proof_size_bytes"], artifact_size)

            report_path = run_dir / "verifier_report.json"
            if report_path.exists():
                report_path.unlink()
            run_native([
                str(VERIFIER), "--baseline-root", str(root),
                "--verifier-context", result["verifier_context_path"],
                "--public-logits", result["public_logits_path"],
                "--artifact", result["artifact_path"],
                "--report", f"{run_relative}/verifier_report.json",
            ], "independent verification")
            report = load_json(report_path, "independent verifier report")
            self.assertEqual(report["schema"], REPORT_SCHEMA)
            self.assertEqual(report["status"], "VERIFY_OK")
            self.assertIs(report["independent"], True)
            self.assertIs(report["masked_full"], True)
            self.assertEqual(report["artifact_size_bytes"], artifact_size)
            self.assertEqual(report["proof_digest"], result["proof_digest"])
            self.assertNotIn("artifact_sha256", report)
            paths = {
                "artifact": result["artifact_path"],
                "context": result["verifier_context_path"],
                "logits": result["public_logits_path"],
                "statement": result["statement_path"],
            }
            gat_zkgpt_experiment._validate_report(
                report, result, "cora", "cpu", resource_identity(root, "cora"),
                paths, artifact_size, report_path,
            )
            self.assertEqual(
                paper_metrics(result, report, report_path)["proof_size_bytes"],
                artifact_size,
            )


if __name__ == "__main__":
    unittest.main()
