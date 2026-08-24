#!/usr/bin/env python3
"""Run one formal zkDL GAT dataset/device row without cross-baseline code."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "cora": "cora_full.cfg",
    "citeseer": "citeseer_full.cfg",
    "pubmed": "pubmed_full.cfg",
    "ppi": "ppi_full.cfg",
    "ogbn-arxiv": "ogbn_arxiv_full.cfg",
}


def call(command: list[object], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in command], check=True, text=True, capture_output=capture
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=CONFIGS, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--pp", type=Path, default=ROOT / "artifacts" / "pp.bin")
    parser.add_argument("--max-vector-len", type=int, default=1 << 28)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    baseline_root = ROOT.resolve()
    if output_root != baseline_root and baseline_root not in output_root.parents:
        raise SystemExit("output-root must stay inside comparison/zkdl")
    project_root = args.project_root.resolve()
    config = ROOT / "configs" / CONFIGS[args.dataset]
    run = output_root / args.dataset / args.device / "warm"
    artifacts = run / "formal_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    preflight = artifacts / "preflight.json"
    result_path = run / "result.json"

    cpu_binary = ROOT / "bin" / "zkdl-formal-cpu"
    prover_binary = ROOT / "bin" / f"zkdl-formal-{args.device}"
    for binary in {cpu_binary, prover_binary}:
        if not binary.is_file():
            raise SystemExit(f"missing formal binary: {binary}; run scripts/build.sh")

    try:
        call(
            [
                "python3",
                ROOT / "scripts" / "preflight.py",
                "--config",
                config,
                "--project-root",
                project_root,
                "--output",
                preflight,
            ]
        )
    except subprocess.CalledProcessError as error:
        info = json.loads(preflight.read_text(encoding="utf-8"))
        write_json(
            result_path,
            {
                "protocol_id": "GAT-ZKML-zkDL",
                "dataset": args.dataset,
                "device": args.device,
                "status": info["status"],
                "blocker": info["ledger"]["failures"],
                "proof_generated": False,
                "verified": False,
                "formal_experiment_eligible": False,
            },
        )
        return error.returncode

    info = json.loads(preflight.read_text(encoding="utf-8"))
    pp_path = args.pp.resolve()
    if not pp_path.exists():
        if pp_path != baseline_root and baseline_root not in pp_path.parents:
            raise SystemExit("new PP path must stay inside comparison/zkdl")
        pp_path.parent.mkdir(parents=True, exist_ok=True)
        call(
            [
                cpu_binary,
                "setup",
                "--output",
                pp_path,
                "--max-vector-len",
                args.max_vector_len,
            ]
        )
    pp_info = json.loads(call([cpu_binary, "pp-info", "--pp", pp_path], capture=True).stdout)
    if pp_info["max_vector_len"] < args.max_vector_len:
        raise SystemExit(
            f"existing PP max_vector_len={pp_info['max_vector_len']} is below required {args.max_vector_len}"
        )

    relation_input = artifacts / "canonical_relation.json"
    statement = artifacts / "statement.bin"
    witness = artifacts / "witness.private.bin"
    adapter_start = time.monotonic()
    call(
        [
            "python3",
            ROOT / "scripts" / "canonical_adapter.py",
            "--config",
            config,
            "--project-root",
            project_root,
            "--preflight",
            preflight,
            "--pp-id",
            pp_info["pp_id"],
            "--output",
            relation_input,
        ]
    )
    adapter_time_ms = (time.monotonic() - adapter_start) * 1000.0
    compile_start = time.monotonic()
    call(
        [
            cpu_binary,
            "compile",
            "--input",
            relation_input,
            "--statement",
            statement,
            "--witness",
            witness,
        ]
    )
    call([cpu_binary, "check-statement", "--pp", pp_path, "--statement", statement])
    compile_time_ms = (time.monotonic() - compile_start) * 1000.0

    model_dir = ROOT / "artifacts" / args.dataset
    model = model_dir / "model_registry.bin"
    secret = model_dir / "model_opening.private.bin"
    if model.exists() != secret.exists():
        raise SystemExit("registered-model object/secret is incomplete; restore or remove both explicitly")
    if not model.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        call(
            [
                cpu_binary,
                "register-model",
                "--pp",
                pp_path,
                "--statement",
                statement,
                "--witness",
                witness,
                "--model-registry",
                model,
                "--model-secret",
                secret,
                "--certified-bounds-id",
                info["ledger"]["integer_bound_ledger_id"],
            ]
        )

    proof = artifacts / "proof.bin"
    proof_manifest = run / "proof_manifest.json"
    verifier_manifest = run / "verifier_manifest.json"
    call(
        [
            prover_binary,
            "prove",
            "--pp",
            pp_path,
            "--statement",
            statement,
            "--model-registry",
            model,
            "--model-secret",
            secret,
            "--witness",
            witness,
            "--proof",
            proof,
            "--manifest",
            proof_manifest,
            "--device",
            args.device,
        ]
    )
    call(
        [
            cpu_binary,
            "verify",
            "--pp",
            pp_path,
            "--statement",
            statement,
            "--model-registry",
            model,
            "--proof",
            proof,
            "--manifest",
            verifier_manifest,
            "--device",
            args.device,
        ]
    )
    proof_info = json.loads(proof_manifest.read_text(encoding="utf-8"))
    verifier_info = json.loads(verifier_manifest.read_text(encoding="utf-8"))
    if not proof_info.get("proof_generated") or not verifier_info.get("verified"):
        raise SystemExit("formal proof/verification manifest did not report success")
    for key in ("proof_format", "statement_id", "model_registry_id", "device", "proof_sha256", "proof_size_bytes", "model_session_commitments", "model_rerandomization_proofs"):
        if proof_info.get(key) != verifier_info.get(key):
            raise SystemExit(f"prover/verifier manifest mismatch: {key}")
    if proof_info.get("proof_format") != "zkDL-GAT-proof-envelope-v3":
        raise SystemExit("unexpected formal proof envelope version")
    if proof_info.get("model_session_rerandomization") != "pedersen_blinder_schnorr_v1" or not verifier_info.get("model_rerandomization_verified"):
        raise SystemExit("registered-model session re-randomization evidence missing")
    if not verifier_info.get("independent_verifier") or verifier_info.get("verifier_reads_witness"):
        raise SystemExit("independent verifier boundary was not preserved")
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
            raise SystemExit("GPU proof lacks required native-kernel coverage telemetry")
        if not verifier_info.get("gpu_telemetry_validated"):
            raise SystemExit("independent verifier did not validate GPU telemetry")

    write_json(
        result_path,
        {
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
            "integer_bound_ledger_id": info["ledger"]["integer_bound_ledger_id"],
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
        },
    )
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
    output_root = Path(option("--output-root", str(ROOT / "runs"))).resolve()
    base = ROOT.resolve()
    if output_root != base and base not in output_root.parents:
        return
    write_json(
        output_root / dataset / device / "warm" / "result.json",
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
