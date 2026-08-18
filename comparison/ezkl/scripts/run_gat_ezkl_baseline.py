#!/usr/bin/env python3
"""Single-dataset formal ezkl/Halo2 GAT runner.

The independent public statement is emitted by witness generation and supplied
as a separate verifier input.  The proof JSON is never used as the statement
source.  CPU and CUDA use separately built binaries and no fallback exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASETS_FILE = ROOT / "configs" / "full_datasets.json"
SECURITY_PROFILE_TEMPLATE = ROOT / "configs" / "security_profile.json"
FORMAL_ROUTE = "ezkl_gat_native_compressed_csr_attention_full"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_datasets() -> dict[str, dict[str, Any]]:
    payload = json.loads(DATASETS_FILE.read_text(encoding="utf-8"))
    return {entry["run_id"]: entry for entry in payload["datasets"]}


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_stage(command: list[str], cwd: Path) -> tuple[float, str, str]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"formal command failed ({completed.returncode}): {rendered}\n"
            f"stdout:\n{completed.stdout[-8000:]}\nstderr:\n{completed.stderr[-8000:]}"
        )
    return elapsed_ms, completed.stdout, completed.stderr


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    datasets = load_datasets()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(datasets))
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--srs-path", type=Path, required=True, help="exact trusted ceremony max_k=28 KZG SRS")
    parser.add_argument("--ceremony-descriptor", type=Path, required=True,
                        help="descriptor binding the ceremony id, exact SRS SHA-256 and original max_k")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-logrows", type=int, default=28)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--run",
        action="store_true",
        help="continue through settings, compile, setup, witness, proof and verify",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entry = load_datasets()[args.dataset]
    config = (args.config or resolve(ROOT, entry["config"])).resolve()
    dataset_root = (args.dataset_root or Path(entry["dataset_root"])).resolve()
    checkpoint = (args.checkpoint or Path(entry["checkpoint"])).resolve()
    srs_path = args.srs_path.resolve()
    ceremony_descriptor = args.ceremony_descriptor.resolve()
    default_run = ROOT / entry["run_dir"] / args.device
    run_dir = (args.output_dir or default_run).resolve()
    runs_root = (ROOT / "runs").resolve()
    if runs_root not in run_dir.parents:
        raise RuntimeError("output directory must be a dataset/device directory below runs/")
    if args.clean and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter = ROOT / "build_gat_adapter" / "gat_ezkl_export"
    ezkl = ROOT / ("target-gpu/release/ezkl" if args.device == "cuda" else "target/release/ezkl")
    if not adapter.is_file():
        raise RuntimeError(f"missing formal adapter; run scripts/build.sh: {adapter}")
    if args.run and not ezkl.is_file():
        target = "GPU" if args.device == "cuda" else "CPU"
        raise RuntimeError(f"missing distinct {target} ezkl binary: {ezkl}")
    if not srs_path.is_file() or not ceremony_descriptor.is_file():
        raise RuntimeError("adapter/settings require an existing --srs-path and --ceremony-descriptor")
    resolved_security_profile = run_dir / "resolved_security_profile.json"
    materialize_command = [
        sys.executable, str(ROOT / "scripts" / "materialize_security_profile.py"),
        "--template", str(SECURITY_PROFILE_TEMPLATE),
        "--ceremony-descriptor", str(ceremony_descriptor),
        "--srs-path", str(srs_path), "--output", str(resolved_security_profile),
    ]
    materialize_ms, materialize_stdout, materialize_stderr = run_stage(materialize_command, ROOT)

    manifest: dict[str, Any] = {
        "manifest_kind": "ezkl_formal_gat_run_manifest_v1",
        "dataset": args.dataset,
        "device": args.device,
        "route_id": FORMAL_ROUTE,
        "config": str(config),
        "dataset_root": str(dataset_root),
        "checkpoint": str(checkpoint),
        "security_profile": str(resolved_security_profile),
        "ceremony_descriptor": str(ceremony_descriptor),
        "srs_path": str(srs_path),
        "run_dir": str(run_dir),
        "formal_run_requested": args.run,
        "status": "preparing",
        "verify_passed": False,
        "timings_ms": {"security_profile_materialization": materialize_ms},
        "commands": [materialize_command],
        "security_profile_materialization_stdout": materialize_stdout[-4000:],
        "security_profile_materialization_stderr": materialize_stderr[-4000:],
    }
    manifest_path = run_dir / "run_manifest.json"
    write_json(manifest_path, manifest)

    adapter_command = [
        str(adapter),
        "--dataset",
        args.dataset,
        "--config",
        str(config),
        "--dataset-root",
        str(dataset_root),
        "--checkpoint",
        str(checkpoint),
        "--security-profile",
        str(resolved_security_profile),
        "--ezkl-root",
        str(ROOT),
        "--output-dir",
        str(run_dir),
        "--route",
        FORMAL_ROUTE,
    ]
    elapsed, stdout, stderr = run_stage(adapter_command, ROOT)
    manifest["timings_ms"]["adapter"] = elapsed
    manifest["commands"].append(adapter_command)
    manifest["adapter_stdout"] = stdout[-4000:]
    manifest["adapter_stderr"] = stderr[-4000:]
    manifest["status"] = "canonical_artifacts_ready"
    spec_identity = json.loads((run_dir / "graph_attention_spec.json").read_text(encoding="utf-8"))
    manifest["quant_cfg_id"] = spec_identity["quant_cfg_id"]
    manifest["static_tables_id"] = spec_identity["static_tables_id"]
    manifest["backend_aux_tables_id"] = spec_identity["backend_aux_tables_id"]
    manifest["ezkl_security_profile_id"] = spec_identity["security_profile_id"]
    write_json(manifest_path, manifest)
    if not args.run:
        print(json.dumps({"status": manifest["status"], "run_dir": str(run_dir)}))
        return 0

    paths = {
        "spec": run_dir / "graph_attention_spec.json",
        "csr": run_dir / "csr_segment_layout.json",
        "settings": run_dir / "settings.json",
        "compiled": run_dir / "network.compiled",
        "vk": run_dir / "vk.key",
        "pk": run_dir / "pk.key",
        "witness": run_dir / "witness.json",
        "statement": run_dir / "public_statement.json",
        "proof": run_dir / "proof.json",
        "verify": run_dir / "verify.json",
    }
    commands: list[tuple[str, list[str]]] = [
        (
            "gen_settings",
            [
                str(ezkl), "graph-gat-gen-settings", "--spec-path", str(paths["spec"]),
                "--csr-layout-path", str(paths["csr"]), "--settings-path",
                str(paths["settings"]), "--max-logrows", str(args.max_logrows),
            ],
        ),
        (
            "compile",
            [
                str(ezkl), "graph-gat-compile", "--spec-path", str(paths["spec"]),
                "--csr-layout-path", str(paths["csr"]), "--settings-path",
                str(paths["settings"]), "--compiled-circuit", str(paths["compiled"]),
                "--max-logrows", str(args.max_logrows),
            ],
        ),
        (
            "setup",
            [
                str(ezkl), "graph-gat-setup", "--compiled-circuit", str(paths["compiled"]),
                "--vk-path", str(paths["vk"]), "--pk-path", str(paths["pk"]),
                "--srs-path", str(srs_path),
                "--device", args.device,
            ],
        ),
        (
            "gen_witness",
            [
                str(ezkl), "graph-gat-gen-witness", "--compiled-circuit",
                str(paths["compiled"]), "--output", str(paths["witness"]),
                "--public-statement", str(paths["statement"]),
            ],
        ),
        (
            "prove",
            [
                str(ezkl), "graph-gat-prove", "--compiled-circuit", str(paths["compiled"]),
                "--witness", str(paths["witness"]), "--public-statement",
                str(paths["statement"]), "--pk-path", str(paths["pk"]),
                "--proof-path", str(paths["proof"]), "--device", args.device,
            ],
        ),
        (
            "verify",
            [
                str(ezkl), "graph-gat-verify", "--compiled-circuit", str(paths["compiled"]),
                "--proof-path", str(paths["proof"]), "--public-statement",
                str(paths["statement"]), "--settings-path", str(paths["settings"]),
                "--vk-path", str(paths["vk"]), "--output", str(paths["verify"]),
            ],
        ),
    ]
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    for stage, command in commands:
        manifest["status"] = f"running_{stage}"
        write_json(manifest_path, manifest)
        try:
            elapsed, stdout, stderr = run_stage(command, ROOT)
        except Exception as error:
            manifest["status"] = "failed"
            manifest["failure_step"] = stage
            manifest["failure"] = str(error)[-8000:]
            write_json(manifest_path, manifest)
            raise
        manifest["timings_ms"][stage] = elapsed
        manifest["commands"].append(command)
        (logs / f"{stage}.stdout.log").write_text(stdout, encoding="utf-8")
        (logs / f"{stage}.stderr.log").write_text(stderr, encoding="utf-8")

    verify = json.loads(paths["verify"].read_text(encoding="utf-8"))
    if verify.get("verify_passed") is not True:
        raise RuntimeError("independent Halo2 verification did not return verify_passed=true")
    circuit_manifest = json.loads((run_dir / "circuit_manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "verified",
            "verify_passed": True,
            "proof_size_bytes": paths["proof"].stat().st_size,
            "proof_sha256": sha256(paths["proof"]),
            "public_statement_sha256": sha256(paths["statement"]),
            "compiled_circuit_sha256": sha256(paths["compiled"]),
            "vk_sha256": sha256(paths["vk"]),
            "runtime_device_marker": verify.get("prover_device_build"),
            "verification_runtime_marker": verify.get("verification_runtime_marker"),
            "source_srs_sha256": circuit_manifest.get("source_srs_sha256"),
            "source_srs_k": circuit_manifest.get("source_srs_k"),
            "downsized_srs_k": circuit_manifest.get("downsized_srs_k"),
            "proving_srs_sha256": circuit_manifest.get("proving_srs_sha256"),
            "verification_srs_sha256": circuit_manifest.get("verification_srs_sha256"),
            "proving_srs_path": circuit_manifest.get("proving_srs_path"),
            "verification_srs_path": circuit_manifest.get("verification_srs_path"),
            "gpu_kernel_execution": "待正式实验验证" if args.device == "cuda" else "not_applicable",
        }
    )
    write_json(manifest_path, manifest)
    result = {
        "dataset": args.dataset,
        "device": args.device,
        "route_id": FORMAL_ROUTE,
        "status": "verified",
        "verify_passed": True,
        "timings_ms": manifest["timings_ms"],
        "proof_size_bytes": manifest["proof_size_bytes"],
        "manifest": str(manifest_path.relative_to(ROOT)),
    }
    write_json(ROOT / "results" / f"{args.dataset}.{args.device}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # Fail closed; never synthesize partial results.
        print(f"formal ezkl runner failed: {error}", file=sys.stderr)
        sys.exit(2)
