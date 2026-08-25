#!/usr/bin/env python3
"""Run one formal zkDL GAT cell with a fail-closed artifact contract."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from runner_contract import (
    BASELINE_ROOT,
    CELL_FILES,
    CONFIGS,
    atomic_write_json,
    artifact_inventory,
    cell_path,
    config_digest,
    config_path,
    ensure_contained,
    ensure_regular_file,
    load_resource_manifest,
    parse_config,
    prepare_cell,
    protocol_digest,
    relative_manifest_path,
    resource_inventory_digest,
    read_json,
    sha256_file,
    validate_config_resources,
)


def path_arg(path: Path) -> str:
    path = ensure_contained(path, label="formal command path")
    return path.relative_to(BASELINE_ROOT).as_posix()


def call(command: list[object], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in command],
        cwd=BASELINE_ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )


def manifest_path_matches(value: object, expected: Path, *, run: Path, label: str) -> None:
    actual = relative_manifest_path(value, run=run, label=label)
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=CONFIGS, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--project-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=BASELINE_ROOT / "runs")
    parser.add_argument("--pp", type=Path, default=BASELINE_ROOT / "artifacts" / "setup" / "pp.bin")
    parser.add_argument("--max-vector-len", type=int, default=1 << 28)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    project_root = ensure_contained(args.project_root, label="project-root")
    if project_root != BASELINE_ROOT:
        raise SystemExit("project-root must be comparison/zkdl; stage resources locally first")
    output_root = ensure_contained(args.output_root, label="runs root")
    config = config_path(args.dataset)
    config_values = parse_config(config)
    resource_manifest, resource_manifest_id = load_resource_manifest(require_staged=True)
    validate_config_resources(args.dataset, config_values, resource_manifest)
    run = cell_path(output_root, args.dataset, args.device)
    prepare_cell(run, replace=args.replace)
    artifacts = ensure_contained(run / "formal_artifacts", label="formal artifact directory")
    artifacts.mkdir(parents=False, exist_ok=False)
    preflight = artifacts / "preflight.json"
    result_path = run / "result.json"

    cpu_binary = ensure_regular_file(BASELINE_ROOT / "bin" / "zkdl-formal-cpu", label="CPU formal binary")
    prover_binary = ensure_regular_file(
        BASELINE_ROOT / "bin" / f"zkdl-formal-{args.device}", label=f"{args.device} formal binary"
    )

    try:
        call(
            [
                "python3",
                "scripts/preflight.py",
                "--config",
                path_arg(config),
                "--project-root",
                ".",
                "--output",
                path_arg(preflight),
            ]
        )
    except subprocess.CalledProcessError as error:
        if preflight.is_file():
            report = read_json(preflight, label="preflight report")
            blocker = report.get("ledger", {}).get("failures") or [str(error)]
        else:
            blocker = [str(error)]
        atomic_write_json(
            result_path,
            {
                "protocol_id": "GAT-ZKML-zkDL",
                "dataset": args.dataset,
                "device": args.device,
                "status": "blocked",
                "proof_generated": False,
                "verified": False,
                "formal_experiment_eligible": False,
                "resource_manifest_id": resource_manifest_id,
                "blocker": blocker,
            },
        )
        return error.returncode

    info = read_json(preflight, label="preflight report")
    if info.get("status") != "accepted":
        raise RuntimeError("formal preflight did not accept the instance")
    if info.get("resource_manifest_id") != resource_manifest_id:
        raise RuntimeError("preflight resource manifest identity mismatch")

    pp_path = ensure_contained(args.pp, label="PP path")
    if not pp_path.exists():
        pp_path.parent.mkdir(parents=True, exist_ok=True)
        call(
            [
                path_arg(cpu_binary),
                "setup",
                "--output",
                path_arg(pp_path),
                "--max-vector-len",
                args.max_vector_len,
            ]
        )
    pp_path = ensure_regular_file(pp_path, label="PP")
    pp_info = json.loads(
        call([path_arg(cpu_binary), "pp-info", "--pp", path_arg(pp_path)], capture=True).stdout
    )
    if int(pp_info.get("max_vector_len", 0)) < args.max_vector_len:
        raise RuntimeError(
            f"existing PP max_vector_len={pp_info.get('max_vector_len')} is below requested {args.max_vector_len}"
        )

    relation_input = artifacts / "canonical_relation.json"
    statement = artifacts / "statement.bin"
    witness = artifacts / "witness.private.bin"
    adapter_start = time.monotonic()
    call(
        [
            "python3",
            "scripts/canonical_adapter.py",
            "--config",
            path_arg(config),
            "--project-root",
            ".",
            "--preflight",
            path_arg(preflight),
            "--pp-id",
            pp_info["pp_id"],
            "--output",
            path_arg(relation_input),
        ]
    )
    adapter_time_ms = (time.monotonic() - adapter_start) * 1000.0
    compile_start = time.monotonic()
    call(
        [
            path_arg(cpu_binary),
            "compile",
            "--input",
            path_arg(relation_input),
            "--statement",
            path_arg(statement),
            "--witness",
            path_arg(witness),
        ]
    )
    call([path_arg(cpu_binary), "check-statement", "--pp", path_arg(pp_path), "--statement", path_arg(statement)])
    compile_time_ms = (time.monotonic() - compile_start) * 1000.0

    model_dir = ensure_contained(BASELINE_ROOT / "artifacts" / "model_registry" / args.dataset, label="model registry directory")
    secret_dir = ensure_contained(BASELINE_ROOT / "artifacts" / "private" / "model_opening" / args.dataset, label="model secret directory")
    model = model_dir / "model_registry.bin"
    secret = secret_dir / "model_opening.private.bin"
    if model.exists() != secret.exists():
        raise RuntimeError("registered-model object/secret is incomplete; restore or remove both explicitly")
    if model.exists():
        ensure_regular_file(model, label="model registry")
        ensure_regular_file(secret, label="model opening secret")
    else:
        model_dir.mkdir(parents=True, exist_ok=True)
        secret_dir.mkdir(parents=True, exist_ok=True)
        call(
            [
                path_arg(cpu_binary),
                "register-model",
                "--pp",
                path_arg(pp_path),
                "--statement",
                path_arg(statement),
                "--witness",
                path_arg(witness),
                "--model-registry",
                path_arg(model),
                "--model-secret",
                path_arg(secret),
                "--certified-bounds-id",
                info["ledger"]["integer_bound_ledger_id"],
            ]
        )
        ensure_regular_file(model, label="model registry")
        ensure_regular_file(secret, label="model opening secret")

    proof = artifacts / "proof.bin"
    proof_manifest = run / "proof_manifest.json"
    verifier_manifest = run / "verifier_manifest.json"
    call(
        [
            path_arg(prover_binary),
            "prove",
            "--pp",
            path_arg(pp_path),
            "--statement",
            path_arg(statement),
            "--model-registry",
            path_arg(model),
            "--model-secret",
            path_arg(secret),
            "--witness",
            path_arg(witness),
            "--proof",
            path_arg(proof),
            "--manifest",
            path_arg(proof_manifest),
            "--device",
            args.device,
        ]
    )
    call(
        [
            path_arg(cpu_binary),
            "verify",
            "--pp",
            path_arg(pp_path),
            "--statement",
            path_arg(statement),
            "--model-registry",
            path_arg(model),
            "--proof",
            path_arg(proof),
            "--manifest",
            path_arg(verifier_manifest),
            "--device",
            args.device,
        ]
    )
    proof_info = read_json(proof_manifest, label="prover manifest")
    verifier_info = read_json(verifier_manifest, label="verifier manifest")
    ensure_regular_file(proof, label="verifier-facing proof")
    manifest_path_matches(proof_info.get("proof_path"), proof, run=run, label="prover proof path")
    manifest_path_matches(verifier_info.get("proof_path"), proof, run=run, label="verifier proof path")
    if not proof_info.get("proof_generated") or not verifier_info.get("verified"):
        raise RuntimeError("formal proof/verification manifest did not report success")
    for key in (
        "proof_format",
        "statement_id",
        "model_registry_id",
        "device",
        "proof_sha256",
        "proof_size_bytes",
        "model_session_commitments",
        "model_rerandomization_proofs",
    ):
        if proof_info.get(key) != verifier_info.get(key):
            raise RuntimeError(f"prover/verifier manifest mismatch: {key}")
    if proof_info.get("proof_format") != "zkDL-GAT-proof-envelope-v3":
        raise RuntimeError("unexpected formal proof envelope version")
    if proof_info.get("model_session_rerandomization") != "pedersen_blinder_schnorr_v1" or not verifier_info.get("model_rerandomization_verified"):
        raise RuntimeError("registered-model session re-randomization evidence missing")
    if not verifier_info.get("independent_verifier") or verifier_info.get("verifier_reads_witness"):
        raise RuntimeError("independent verifier boundary was not preserved")
    if sha256_file(proof) != proof_info.get("proof_sha256") or proof.stat().st_size != int(proof_info.get("proof_size_bytes", -1)):
        raise RuntimeError("proof digest/size does not match prover manifest")
    if args.device == "gpu":
        telemetry = proof_info.get("gpu_telemetry") or {}
        required = (
            "msm_kernel_invocations",
            "field_inner_product_invocations",
            "scalar_fold_invocations",
            "point_fold_invocations",
            "sumcheck_kernel_invocations",
        )
        if not all(int(telemetry.get(key, 0)) > 0 for key in required):
            raise RuntimeError("GPU proof lacks required native-kernel coverage telemetry")
        if not verifier_info.get("gpu_telemetry_validated"):
            raise RuntimeError("independent verifier did not validate GPU telemetry")

    provenance = {
        "config_sha256": config_digest(config),
        "protocol_sha256": protocol_digest(),
        "resource_manifest_id": resource_manifest_id,
        "resource_inventory_digest": resource_inventory_digest(args.dataset, resource_manifest),
        "preflight_sha256": sha256_file(preflight),
        "canonical_relation_sha256": sha256_file(relation_input),
        "statement_sha256": sha256_file(statement),
        "witness_sha256": sha256_file(witness),
        "proof_manifest_sha256": sha256_file(proof_manifest),
        "verifier_manifest_sha256": sha256_file(verifier_manifest),
        "cpu_binary_sha256": sha256_file(cpu_binary),
        "prover_binary_sha256": sha256_file(prover_binary),
        "pp_sha256": sha256_file(pp_path),
    }
    result_payload = {
        "protocol_id": "GAT-ZKML-zkDL",
        "dataset": args.dataset,
        "device": args.device,
        "status": "ok",
        "proof_generated": True,
        "verified": True,
        "independent_verifier": True,
        "verifier_reads_witness": False,
        "statement_id": proof_info["statement_id"],
        "model_registry_id": proof_info["model_registry_id"],
        "pp_id": pp_info["pp_id"],
        "resource_manifest_id": resource_manifest_id,
        "config_sha256": config_digest(config),
        "protocol_sha256": protocol_digest(),
        "integer_bound_ledger_id": info["ledger"]["integer_bound_ledger_id"],
        "node_count": info["node_count"],
        "edge_count": info["edge_count"],
        "graph_count": info["graph_count"],
        "feature_dim": info["feature_dim"],
        "class_count": info["class_count"],
        "graph_features_digest": info["graph_features_digest"],
        "topology_digest": info["topology_digest"],
        "checkpoint_manifest_id": info["checkpoint_manifest_id"],
        "model_shape_id": info["model_shape_id"],
        "static_tables_id": info["static_tables_id"],
        "static_table_rows_id": info["static_table_rows_id"],
        "model_arch_id": info["model_arch_id"],
        "model_param_id": info["model_param_id"],
        "resource_inventory_digest": info["resource_inventory_digest"],
        "public_instance": info["public_instance"],
        "proof_sha256": proof_info["proof_sha256"],
        "proof_size_bytes": proof_info["proof_size_bytes"],
        "adapter_time_ms": adapter_time_ms,
        "compile_time_ms": compile_time_ms,
        "prove_time_ms": proof_info["prove_time_ms"],
        "verify_time_ms": verifier_info["verify_time_ms"],
        "gpu_telemetry": proof_info.get("gpu_telemetry"),
        "proof_format": proof_info["proof_format"],
        "model_session_commitments": proof_info.get("model_session_commitments", 0),
        "model_rerandomization_proofs": proof_info.get("model_rerandomization_proofs", 0),
        "model_session_rerandomization": proof_info.get("model_session_rerandomization"),
        "privacy_profile_status": "supported_multi_session_registered_model",
        "formal_experiment_eligible": True,
        "provenance": provenance,
    }
    atomic_write_json(result_path, result_payload)
    before_success = artifact_inventory(run)
    required_before_success = set(CELL_FILES) - {"success_manifest.json"}
    if set(before_success) != required_before_success:
        raise RuntimeError(
            f"cell artifacts are not exact before success manifest: missing={sorted(required_before_success - set(before_success))} extra={sorted(set(before_success) - required_before_success)}"
        )
    success_payload = {
        "schema": "zkDL_gat_cell_success_manifest_v1",
        "status": "verified",
        "dataset": args.dataset,
        "device": args.device,
        "protocol_id": result_payload["protocol_id"],
        "proof_sha256": proof_info["proof_sha256"],
        "statement_id": proof_info["statement_id"],
        "model_registry_id": proof_info["model_registry_id"],
        "config_sha256": result_payload["config_sha256"],
        "protocol_sha256": result_payload["protocol_sha256"],
        "resource_manifest_id": result_payload["resource_manifest_id"],
        "resource_inventory_digest": result_payload["resource_inventory_digest"],
        "integer_bound_ledger_id": result_payload["integer_bound_ledger_id"],
        "node_count": result_payload["node_count"],
        "edge_count": result_payload["edge_count"],
        "graph_count": result_payload["graph_count"],
        "feature_dim": result_payload["feature_dim"],
        "class_count": result_payload["class_count"],
        "graph_features_digest": result_payload["graph_features_digest"],
        "topology_digest": result_payload["topology_digest"],
        "public_instance": result_payload["public_instance"],
        "checkpoint_manifest_id": result_payload["checkpoint_manifest_id"],
        "model_shape_id": result_payload["model_shape_id"],
        "static_tables_id": result_payload["static_tables_id"],
        "static_table_rows_id": result_payload["static_table_rows_id"],
        "model_arch_id": result_payload["model_arch_id"],
        "model_param_id": result_payload["model_param_id"],
        "gpu_telemetry": result_payload["gpu_telemetry"],
        "provenance": result_payload["provenance"],
        "files": before_success,
    }
    atomic_write_json(run / "success_manifest.json", success_payload)
    artifact_inventory(run, require_complete=True)
    print(f"ZKDL_CELL_OK dataset={args.dataset} device={args.device} proof_sha256={proof_info['proof_sha256']}")
    return 0


def failure_result(error: BaseException) -> None:
    def option(name: str, default: str | None = None) -> str | None:
        try:
            return sys.argv[sys.argv.index(name) + 1]
        except (ValueError, IndexError):
            return default

    dataset = option("--dataset")
    device = option("--device")
    if dataset not in CONFIGS or device not in {"cpu", "gpu"}:
        return
    try:
        output_root = ensure_contained(Path(option("--output-root", str(BASELINE_ROOT / "runs"))), label="runs root")
        run = cell_path(output_root, dataset, device)
        if run.exists() and (run / "success_manifest.json").exists():
            return
        atomic_write_json(
            run / "result.json",
            {
                "protocol_id": "GAT-ZKML-zkDL",
                "dataset": dataset,
                "device": device,
                "status": "failed",
                "proof_generated": False,
                "verified": False,
                "formal_experiment_eligible": False,
                "failure_type": type(error).__name__,
                "failure": str(error),
            },
        )
    except Exception:
        return


if __name__ == "__main__":
    try:
        return_code = main()
    except (Exception, SystemExit) as error:
        failure_result(error)
        if isinstance(error, SystemExit) and isinstance(error.code, int):
            return_code = error.code
        elif isinstance(error, subprocess.CalledProcessError):
            return_code = error.returncode
        else:
            return_code = 1
        print(f"zkDL formal row failed: {error}", file=sys.stderr)
    raise SystemExit(return_code)
