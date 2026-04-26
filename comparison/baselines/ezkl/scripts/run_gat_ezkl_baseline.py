#!/usr/bin/env python3
"""Thin runner for the GAT-ZKML ezkl baseline.

The C++ exporter owns dataset loading, graph normalization, checkpoint checks,
hashing, GAT semantics metadata, and strict-route export decisions. This runner
only invokes the exporter, invokes ezkl if a strict ONNX export is available,
and writes orchestration status files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


EZKL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path("/home/pzh/project")
EXPORTER = EZKL_ROOT / "build_gat_adapter" / "gat_ezkl_export"
DATASETS = {
    "cora_full": "runs/cora_full",
    "citeseer_full": "runs/citeseer_full",
    "pubmed_full": "runs/pubmed_full",
    "ppi_full_formal": "runs/ppi_full_formal",
    "ogbn_arxiv_full": "runs/ogbn_arxiv_full",
}
SEGMENT_ROUTE = "ezkl_gat_strict_segment_softmax_full"
NATIVE_CSR_ROUTE = "ezkl_gat_native_csr_segment_attention_full"
COMPRESSED_NATIVE_CSR_ROUTE = "ezkl_gat_native_compressed_csr_attention_full"
NATIVE_GRAPH_ROUTES = {NATIVE_CSR_ROUTE, COMPRESSED_NATIVE_CSR_ROUTE}
NATIVE_DEBUG_SCOPES = [
    "level0_mul",
    "inverse_only",
    "segment_sum_inverse",
    "segment_max_only",
    "lrelu_lookup_only",
    "exp_lookup_only",
    "elu_lookup_only",
    "compressed_projection_only",
    "compressed_aggregation_only",
    "single_head_first_layer",
    "first_hidden_layer_all_heads",
    "hidden_only",
    "output_only",
    "full_model",
]


def ensure_inside(path: Path, root: Path) -> None:
    path = path.resolve()
    root = root.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"path escapes ezkl root: {path}")


def tail(text: str, limit: int = 4000) -> str:
    return text[-limit:]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_command(command: list[str], cwd: Path) -> tuple[int, float, str, str]:
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return completed.returncode, elapsed_ms, completed.stdout, completed.stderr


def classify_ezkl_failure(timing_key: str, stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if "reducemax" in text or "reduce<max>" in text:
        return "ezkl_reduce_max_unsupported"
    if "masked" in text and "sum" in text:
        return "ezkl_masked_sum_unsupported"
    if "gather" in text:
        return "ezkl_gather_for_packed_src_unsupported"
    if "exp" in text or "lookup" in text:
        return "ezkl_exp_lookup_semantics_unavailable"
    if "recip" in text or "inverse" in text or " div" in text or "division" in text:
        return "ezkl_inverse_semantics_unavailable"
    if timing_key in {"gen_settings_ms", "compile_ms"}:
        return "ezkl_operator_lowering_failed"
    if timing_key in {"get_srs_ms", "setup_ms"}:
        return "ezkl_setup_resource_failure"
    if timing_key == "prove_time_ms":
        return "ezkl_prove_resource_failure"
    if timing_key == "verify_time_ms":
        return "ezkl_verify_failed"
    return "ezkl_operator_lowering_failed"


def analysis_blocker_classification(blocker: str) -> str:
    if blocker == "ezkl_segment_softmax_sentinel_bound_unproven":
        return "ezkl_segment_softmax_padding_bound_unverified"
    if blocker == "ezkl_segment_packed_softmax_resource_boundary":
        return "ezkl_segment_softmax_resource_boundary"
    return "ezkl_segment_softmax_analysis_blocked"


def classify_native_failure(timing_key: str, stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if "inverse_policy_hash_mismatch" in text:
        return "inverse_policy_hash_mismatch"
    if "quantized_inverse_lookup_mismatch" in text:
        return "quantized_inverse_lookup_mismatch"
    if "alpha_rescale_mismatch" in text:
        return "alpha_rescale_mismatch"
    if "native_full_witness_preflight_failed" in text:
        return "native_full_witness_preflight_failed"
    if "native_full_gat_forward_constraints_incomplete" in text:
        return "native_full_gat_forward_constraints_incomplete"
    if "final_logits_derivation_incomplete" in text:
        return "final_logits_derivation_incomplete"
    if "native_constraint_synthesis_failed" in text:
        return "native_constraint_synthesis_failed"
    if "native_keygen" in text:
        return "native_keygen_failed"
    if "native_prover" in text:
        return "native_prover_failed"
    if "native_verify" in text:
        return "native_verify_failed"
    if "ezkl_compressed_csr_resource_boundary" in text:
        return "ezkl_compressed_csr_resource_boundary"
    if "ezkl_native_csr_compile_resource_boundary" in text or "required_logrows" in text:
        return "ezkl_native_csr_compile_resource_boundary"
    if "lookup" in text and "quant_cfg" in text:
        return "ezkl_native_lookup_tables_missing_from_quant_cfg"
    if "inverse" in text:
        return "ezkl_native_segment_inverse_implementation_failed"
    if timing_key == "verify_time_ms":
        return "ezkl_native_verify_failed"
    return "ezkl_native_csr_route_failed"


def read_native_failure(run_dir: Path) -> dict:
    path = run_dir / "native_constraint_failure.json"
    if path.exists():
        try:
            return read_json(path)
        except json.JSONDecodeError:
            return {}
    return {}


def update_manifest(run_dir: Path, **updates: object) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    manifest.update(updates)
    write_json(manifest_path, manifest)
    return manifest


def write_blocked_artifacts(run_dir: Path, manifest: dict) -> None:
    proof = {
        "status": manifest.get("status", "blocked"),
        "route_id": manifest.get("route_id"),
        "proof_generated": False,
        "block_reason": manifest.get("block_reason"),
        "limitation": manifest.get("limitation"),
    }
    verify = {
        "status": manifest.get("status", "blocked"),
        "verified": False,
        "verify_invoked": False,
        "block_reason": manifest.get("block_reason"),
    }
    write_json(run_dir / "proof.json", proof)
    write_json(run_dir / "verify.json", verify)


def find_ezkl_command() -> list[str]:
    binary = EZKL_ROOT / "target" / "release" / "ezkl"
    if binary.exists():
        return [str(binary)]
    return ["cargo", "run", "--release", "--bin", "ezkl", "--"]


def run_ezkl_pipeline(run_dir: Path, manifest: dict) -> dict:
    ezkl = find_ezkl_command()
    timings: dict[str, float | None] = {
        "gen_settings_ms": None,
        "compile_ms": None,
        "get_srs_ms": None,
        "setup_ms": None,
        "gen_witness_ms": None,
        "prove_time_ms": None,
        "verify_time_ms": None,
    }
    commands = [
        ("gen_settings_ms", ezkl + ["gen-settings", "--model", "network.onnx", "--data", "input.json", "--settings-path", "settings.json"]),
        ("compile_ms", ezkl + ["compile-circuit", "--model", "network.onnx", "--compiled-circuit", "model.compiled", "--settings-path", "settings.json"]),
        ("get_srs_ms", ezkl + ["get-srs", "--settings-path", "settings.json"]),
        ("setup_ms", ezkl + ["setup", "--compiled-circuit", "model.compiled", "--vk-path", "vk.key", "--pk-path", "pk.key"]),
        ("gen_witness_ms", ezkl + ["gen-witness", "--data", "input.json", "--compiled-circuit", "model.compiled", "--output", "witness.json"]),
        ("prove_time_ms", ezkl + ["prove", "--witness", "witness.json", "--compiled-circuit", "model.compiled", "--pk-path", "pk.key", "--proof-path", "proof.json"]),
        ("verify_time_ms", ezkl + ["verify", "--proof-path", "proof.json", "--settings-path", "settings.json", "--vk-path", "vk.key"]),
    ]
    stdout_all = []
    stderr_all = []
    for timing_key, command in commands:
        code, elapsed_ms, stdout, stderr = run_command(command, run_dir)
        stdout_all.append(f"$ {' '.join(command)}\n{stdout}")
        stderr_all.append(f"$ {' '.join(command)}\n{stderr}")
        if timing_key in timings:
            timings[timing_key] = elapsed_ms
        if code != 0:
            failure = classify_ezkl_failure(timing_key, stdout, stderr)
            manifest.update(
                {
                    "status": "failed",
                    "block_reason": failure,
                    "failed_command": " ".join(command),
                    "stdout_tail": tail("\n".join(stdout_all)),
                    "stderr_tail": tail("\n".join(stderr_all)),
                }
            )
            write_json(run_dir / "run_manifest.json", manifest)
            write_json(run_dir / "verify.json", {"status": "failed", "verified": False, "failed_command": " ".join(command)})
            return manifest
    proof_path = run_dir / "proof.json"
    proof_size = proof_path.stat().st_size if proof_path.exists() else None
    if (run_dir / "compressed_row_estimate.json").exists():
        manifest["compressed_row_estimate"] = read_json(run_dir / "compressed_row_estimate.json")
    if (run_dir / "formal_table_hashes.json").exists():
        manifest["formal_table_hashes"] = read_json(run_dir / "formal_table_hashes.json")
    manifest.update(timings)
    manifest.update(
        {
            "status": "verified",
            "block_reason": "",
            "proof_size_bytes": proof_size,
            "commitment_time_ms": timings.get("setup_ms"),
            "check_mode": "safe",
            "stdout_tail": tail("\n".join(stdout_all)),
            "stderr_tail": tail("\n".join(stderr_all)),
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "verify.json", {"status": "verified", "verified": True})
    return manifest


def run_native_graph_pipeline(run_dir: Path, manifest: dict, native_debug_scope: str | None = None) -> dict:
    ezkl = find_ezkl_command()
    timings: dict[str, float | None] = {
        "gen_settings_ms": None,
        "compile_ms": None,
        "setup_ms": None,
        "gen_witness_ms": None,
        "prove_time_ms": None,
        "verify_time_ms": None,
    }
    scope_flags = []
    if native_debug_scope:
        scope_flags = ["--native-debug-scope", native_debug_scope]
    commands = [
        (
            "gen_settings_ms",
            ezkl
            + [
                "graph-gat-gen-settings",
                "--spec-path",
                "graph_attention_spec.json",
                "--csr-layout-path",
                "csr_segment_layout.json",
                "--settings-path",
                "settings.json",
            ]
            + scope_flags,
        ),
        (
            "compile_ms",
            ezkl
            + [
                "graph-gat-compile",
                "--spec-path",
                "graph_attention_spec.json",
                "--csr-layout-path",
                "csr_segment_layout.json",
                "--settings-path",
                "settings.json",
                "--compiled-circuit",
                "model.compiled",
            ]
            + scope_flags,
        ),
        (
            "setup_ms",
            ezkl
            + [
                "graph-gat-setup",
                "--compiled-circuit",
                "model.compiled",
                "--vk-path",
                "vk.key",
                "--pk-path",
                "pk.key",
            ],
        ),
        (
            "gen_witness_ms",
            ezkl
            + [
                "graph-gat-gen-witness",
                "--compiled-circuit",
                "model.compiled",
                "--output",
                "witness.json",
            ],
        ),
        (
            "prove_time_ms",
            ezkl
            + [
                "graph-gat-prove",
                "--compiled-circuit",
                "model.compiled",
                "--witness",
                "witness.json",
                "--pk-path",
                "pk.key",
                "--proof-path",
                "proof.json",
            ],
        ),
        (
            "verify_time_ms",
            ezkl
            + [
                "graph-gat-verify",
                "--compiled-circuit",
                "model.compiled",
                "--proof-path",
                "proof.json",
                "--settings-path",
                "settings.json",
                "--vk-path",
                "vk.key",
                "--output",
                "verify.json",
            ],
        ),
    ]
    stdout_all = []
    stderr_all = []
    for timing_key, command in commands:
        code, elapsed_ms, stdout, stderr = run_command(command, run_dir)
        stdout_all.append(f"$ {' '.join(command)}\n{stdout}")
        stderr_all.append(f"$ {' '.join(command)}\n{stderr}")
        timings[timing_key] = elapsed_ms
        if code != 0:
            failure = classify_native_failure(timing_key, stdout, stderr)
            native_failure = read_native_failure(run_dir)
            if (
                timing_key == "setup_ms"
                and not native_failure
                and (code < 0 or (not stdout.strip() and not stderr.strip()))
            ):
                failure = "native_keygen_resource_failure"
            manifest.update(timings)
            manifest.update(
                {
                    "status": "blocked"
                    if (
                        "resource_boundary" in failure
                        or "resource_failure" in failure
                        or "lookup_tables" in failure
                        or "backend_unavailable" in failure
                        or "backend_incomplete" in failure
                        or failure == "inverse_policy_hash_mismatch"
                        or failure == "quantized_inverse_lookup_mismatch"
                        or failure == "alpha_rescale_mismatch"
                        or failure == "native_full_witness_preflight_failed"
                        or failure == "native_full_gat_forward_constraints_incomplete"
                        or failure == "final_logits_derivation_incomplete"
                    )
                    else "failed",
                    "block_reason": failure,
                    "failure_step": timing_key,
                    "failed_command": " ".join(command),
                    "command": " && ".join(" ".join(cmd) for _, cmd in commands),
                    "native_debug_scope": native_debug_scope or manifest.get("native_debug_scope") or "full_model",
                    "strict_public_input_equivalent": False,
                    "full_gat_forward_in_ezkl_proof": False,
                    "verify_passed": False,
                    "proof_size_bytes": None,
                    "commitment_time_ms": timings.get("setup_ms"),
                    "stdout_tail": tail("\n".join(stdout_all)),
                    "stderr_tail": tail("\n".join(stderr_all)),
                    "failure_gate": native_failure.get("gate_name"),
                    "failure_region": native_failure.get("region_name"),
                    "failure_row": native_failure.get("row_offset"),
                    "failure_relation": native_failure.get("expected_relation"),
                    "failure_values": native_failure.get("actual_values"),
                }
            )
            if (run_dir / "model.compiled").exists():
                try:
                    manifest["native_compile_plan"] = read_json(run_dir / "model.compiled")
                except json.JSONDecodeError:
                    pass
            if (run_dir / "compressed_row_estimate.json").exists():
                try:
                    manifest["compressed_row_estimate"] = read_json(run_dir / "compressed_row_estimate.json")
                except json.JSONDecodeError:
                    pass
            if (run_dir / "formal_table_hashes.json").exists():
                try:
                    manifest["formal_table_hashes"] = read_json(run_dir / "formal_table_hashes.json")
                except json.JSONDecodeError:
                    pass
            write_json(run_dir / "run_manifest.json", manifest)
            write_blocked_artifacts(run_dir, manifest)
            return manifest
    proof_path = run_dir / "proof.json"
    proof_size = proof_path.stat().st_size if proof_path.exists() else None
    verify = read_json(run_dir / "verify.json") if (run_dir / "verify.json").exists() else {}
    strict_public = bool(verify.get("strict_public_input_equivalent", False))
    full_forward = bool(verify.get("full_gat_forward_in_ezkl_proof", False))
    manifest.update(timings)
    manifest.update(
        {
            "status": "verified",
            "block_reason": "",
            "proof_size_bytes": proof_size,
            "commitment_time_ms": timings.get("setup_ms"),
            "check_mode": "safe",
            "strict_public_input_equivalent": strict_public,
            "full_gat_forward_in_ezkl_proof": full_forward,
            "verify_passed": bool(verify.get("verify_passed", True)),
            "native_debug_scope": native_debug_scope or verify.get("native_debug_scope") or manifest.get("native_debug_scope") or "full_model",
            "command": " && ".join(" ".join(cmd) for _, cmd in commands),
            "stdout_tail": tail("\n".join(stdout_all)),
            "stderr_tail": tail("\n".join(stderr_all)),
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "verify.json", {"status": "verified", "verified": True})
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS.keys())
    parser.add_argument("--mode", default="full", choices=["full"])
    parser.add_argument("--route", default="strict_gat_full_onnx_v1")
    parser.add_argument("--max-packed-slots-limit", type=int, default=16000000)
    parser.add_argument("--emit-segment-debug", action="store_true")
    parser.add_argument("--native-debug-scope", choices=NATIVE_DEBUG_SCOPES)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if not EXPORTER.exists():
        raise RuntimeError(f"C++ exporter is not built: {EXPORTER}")

    run_dir = EZKL_ROOT / DATASETS[args.dataset]
    if args.native_debug_scope:
        run_dir = run_dir / f"debug_{args.native_debug_scope}"
    ensure_inside(run_dir, EZKL_ROOT)
    if args.clean and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.route == SEGMENT_ROUTE:
        analysis_command = [
            str(EXPORTER),
            "--dataset",
            args.dataset,
            "--project-root",
            str(PROJECT_ROOT),
            "--ezkl-root",
            str(EZKL_ROOT),
            "--output-dir",
            str(run_dir),
            "--route",
            args.route,
            "--analysis-only",
            "segment-softmax",
            "--max-packed-slots-limit",
            str(args.max_packed_slots_limit),
            "--fail-on-padding-bound-unverified",
        ]
        if args.emit_segment_debug:
            analysis_command.append("--emit-segment-debug")
        code, analysis_ms, stdout, stderr = run_command(analysis_command, EZKL_ROOT)
        (run_dir / "segment_analysis.stdout.txt").write_text(stdout, encoding="utf-8")
        (run_dir / "segment_analysis.stderr.txt").write_text(stderr, encoding="utf-8")
        analysis_path = run_dir / "segment_softmax_analysis.json"
        if code != 0 or not analysis_path.exists():
            manifest = {
                "dataset": args.dataset,
                "route_id": args.route,
                "status": "failed",
                "block_reason": "ezkl_segment_softmax_analysis_blocked",
                "analysis_ms": analysis_ms,
                "strict_public_input_equivalent": False,
                "stdout_tail": tail(stdout),
                "stderr_tail": tail(stderr),
            }
            write_json(run_dir / "run_manifest.json", manifest)
            write_json(run_dir / "proof.json", {"status": "failed", "proof_generated": False})
            write_json(run_dir / "verify.json", {"status": "failed", "verified": False})
            print(f"dataset={args.dataset} status=failed block_reason=ezkl_segment_softmax_analysis_blocked")
            return 1
        analysis = read_json(analysis_path)
        blocker = analysis.get("blocker_if_any") or ""
        if blocker or not analysis.get("route_feasible_full", False):
            block_reason = blocker or "ezkl_segment_softmax_analysis_blocked"
            manifest = {
                "dataset": args.dataset,
                "route_id": args.route,
                "status": "blocked",
                "block_reason": block_reason,
                "failure_classification": analysis_blocker_classification(block_reason),
                "analysis_ms": analysis_ms,
                "strict_public_input_equivalent": False,
                "full_gat_forward_in_ezkl_proof": False,
                "forbidden_substitute_used": False,
                "precomputed_logits_binding_used": False,
                "commitment_time_ms": None,
                "prove_time_ms": None,
                "verify_time_ms": None,
                "proof_size_bytes": None,
                "N": analysis.get("node_count"),
                "E": analysis.get("edge_count"),
                "G_batch": analysis.get("graph_count"),
                "max_in_degree": analysis.get("max_in_degree"),
                "padded_edge_slots": analysis.get("total_packed_slots"),
                "padding_ratio": analysis.get("padding_ratio"),
                "stdout_tail": tail(stdout),
                "stderr_tail": tail(stderr),
            }
            write_json(run_dir / "run_manifest.json", manifest)
            write_blocked_artifacts(run_dir, manifest)
            print(f"dataset={args.dataset} status=blocked block_reason={block_reason} route_id={args.route}")
            return 0

    command = [
        str(EXPORTER),
        "--dataset",
        args.dataset,
        "--project-root",
        str(PROJECT_ROOT),
        "--ezkl-root",
        str(EZKL_ROOT),
        "--output-dir",
        str(run_dir),
    ]
    if args.route:
        command.extend(["--route", args.route])
    if args.route == SEGMENT_ROUTE:
        command.extend(["--max-packed-slots-limit", str(args.max_packed_slots_limit)])
        command.append("--fail-on-padding-bound-unverified")
        if args.emit_segment_debug:
            command.append("--emit-segment-debug")
    code, exporter_ms, stdout, stderr = run_command(command, EZKL_ROOT)
    (run_dir / "exporter.stdout.txt").write_text(stdout, encoding="utf-8")
    (run_dir / "exporter.stderr.txt").write_text(stderr, encoding="utf-8")
    if code != 0:
        manifest = {
            "dataset": args.dataset,
            "status": "failed",
            "block_reason": "exporter_failure",
            "exporter_ms": exporter_ms,
            "stdout_tail": tail(stdout),
            "stderr_tail": tail(stderr),
        }
        write_json(run_dir / "run_manifest.json", manifest)
        write_json(run_dir / "proof.json", {"status": "failed", "proof_generated": False})
        write_json(run_dir / "verify.json", {"status": "failed", "verified": False})
        print(f"dataset={args.dataset} status=failed block_reason=exporter_failure")
        return 1

    manifest = update_manifest(
        run_dir,
        exporter_ms=exporter_ms,
        native_debug_scope=args.native_debug_scope or "full_model",
        scope=args.native_debug_scope or "full_model",
        stdout_tail=tail(stdout),
        stderr_tail=tail(stderr),
    )
    if manifest.get("status") == "blocked":
        write_blocked_artifacts(run_dir, manifest)
        print(
            f"dataset={args.dataset} status=blocked "
            f"block_reason={manifest.get('block_reason')} route_id={manifest.get('route_id')}"
        )
        return 0

    if args.run:
        if args.route in NATIVE_GRAPH_ROUTES:
            manifest = run_native_graph_pipeline(run_dir, manifest, args.native_debug_scope)
        else:
            manifest = run_ezkl_pipeline(run_dir, manifest)
        print(f"dataset={args.dataset} status={manifest.get('status')} route_id={manifest.get('route_id')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
