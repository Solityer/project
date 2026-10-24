#!/usr/bin/env python3
"""Bounded collector/final-validator check for the native CPU fixture.

The input proof must be an artifact emitted by the already-passed native
segmented fixture.  This script creates only explicit fixture-mode records in
one TemporaryDirectory; formal collector/validator mode is checked to reject
those records.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from result_artifacts import (  # noqa: E402
    DATASETS,
    FIXTURE_PROOF_SCOPE,
)


def write_kv(path: Path, values: dict[str, object]) -> None:
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
    )


def invoke(command: list[str], expected: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=20)
    if result.returncode != expected:
        raise RuntimeError(
            f"unexpected exit={result.returncode} expected={expected}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def make_records(root: Path, artifact: Path, evidence: str) -> int:
    artifact_size = artifact.stat().st_size
    with artifact.open("rb") as stream:
        magic = stream.read(4)
    if artifact_size < 4 or magic != b"GATR":
        raise RuntimeError("input artifact is not a native GATR artifact")
    evidence_text = (
        evidence.strip() + "\n" +
        f"source_artifact={artifact.resolve()}\n" +
        "independent_verify=PASS\n"
    )
    runs = root / "runs"
    runs.mkdir()
    for dataset in DATASETS:
        run_dir = runs / f"{dataset}-fixture"
        run_dir.mkdir()
        shutil.copyfile(artifact, run_dir / "proof.bin")
        (run_dir / "public_Y.txt").write_text("fixture-public-output\n", encoding="utf-8")
        (run_dir / "independent_verify.evidence").write_text(evidence_text, encoding="utf-8")
        statement_id = f"fixture-statement-{dataset}"
        record = {
            "schema": "zkDL_gat_run_record_v2",
            "fixture_mode": 1,
            "dataset": dataset,
            "backend": "cpu",
            "statement_id": statement_id,
            "verification_status": "VERIFIED",
            "proof_scope": FIXTURE_PROOF_SCOPE,
            "proof_size_bytes": len(artifact_bytes),
            "commitment_size_bytes": 0,
            "commitment_time_semantics": "native_tensor_commitments",
            "commitment_time_seconds": "0",
            "prove_time_seconds": "0",
            "verify_time_seconds": "0",
            "verifier_reads_witness": 0,
            "gpu_kernel_executed": 0,
            "silent_fallback": 0,
            "output_bound": 1,
        }
        envelope = {
            "schema_id": "zkDL_gat_proof_envelope_v1",
            "fixture_mode": 1,
            "status": "PROVED",
            "native_core_api": "zkdl_native_gat_bridge_v1",
            "backend": "cpu",
            "proof_scope": FIXTURE_PROOF_SCOPE,
            "statement_id": statement_id,
            "proof_size_bytes": artifact_size,
            "commitment_size_bytes": 0,
            "commitment_time_semantics": "native_tensor_commitments",
            "commitment_time_seconds": "0",
            "prove_time_seconds": "0",
            "verify_time_seconds": "0",
            "verifier_reads_witness": 0,
            "gpu_kernel_executed": 0,
            "silent_fallback": 0,
            "output_bound": 1,
        }
        statement = {
            "schema": "zkDL_gat_statement_v1",
            "fixture_mode": 1,
            "dataset": dataset,
            "statement_id": statement_id,
            "proof_scope": FIXTURE_PROOF_SCOPE,
            "verifier_reads_witness": 0,
            "gpu_kernel_executed": 0,
            "silent_fallback": 0,
            "output_bound": 1,
            "privacy_policy": "fixture_cpu_only",
            "public_output_path": "public_Y.txt",
        }
        verification = {
            "schema": "zkDL_gat_verification_result_v1",
            "fixture_mode": 1,
            "status": "VERIFIED",
            "backend": "cpu",
            "statement_id": statement_id,
            "independent_verify": "PASS",
            "gpu_kernel_executed": 0,
            "silent_fallback": 0,
        }
        write_kv(run_dir / "run.record", record)
        write_kv(run_dir / "proof.meta", envelope)
        write_kv(run_dir / "proof.timing", {
            "schema_id": "zkDL_gat_proof_timing_v1",
            "status": "PROVED",
            "statement_id": statement_id,
            "commitment_time_seconds": "0",
            "prove_time_seconds": "0",
            "proof_auxiliary_construction_seconds": "0",
            "proof_auxiliary_construction_scope": "signed_round_aux_witness_and_relation_views;steady_clock_wall;exclude_family_storage_write_hash_and_explicit_file_io",
            "sorted_lookup_protocol_construction_seconds": "0",
            "relation_context_prove_seconds": "0",
            "relation_prove_wall_seconds": "0",
            "relation_commitment_time_seconds": "0",
            "prove_provider_file_io_excluded_seconds": "0",
            "proof_serialization_seconds": "0",
        })
        write_kv(run_dir / "statement.txt", statement)
        write_kv(run_dir / "verification.result", verification)
        (run_dir / "independent_verify.evidence").chmod(0o600)
    return artifact_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--evidence", required=True,
                        help="captured native fixture evidence containing independent_verify=PASS")
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    if artifact.is_symlink() or not artifact.is_file():
        raise SystemExit(f"collector-fixture: artifact is missing or symlinked: {artifact}")
    if "independent_verify=PASS" not in args.evidence:
        raise SystemExit("collector-fixture: evidence must contain independent_verify=PASS")
    with tempfile.TemporaryDirectory(prefix="zkdl-collector-fixture-") as raw_root:
        root = Path(raw_root)
        artifact_size = make_records(root, artifact, args.evidence)
        common = ["--root", str(root), "--runs-dir", "runs"]
        collect = [sys.executable, str(SCRIPTS / "collect_results.py"),
                   *common, "--output", "results/collected.json",
                   "--fixture-mode"]
        final = [sys.executable, str(SCRIPTS / "final_validate.py"),
                 *common, "--fixture-mode", "--collection", "results/collected.json"]
        collected = invoke(collect, 0)
        validated = invoke(final, 0)
        formal_reject = invoke([
            sys.executable, str(SCRIPTS / "final_validate.py"),
            *common, "--collection", "results/collected.json"], 1)

        first = root / "runs" / "cora-fixture"
        checks: list[tuple[str, callable]] = []

        def mutate_gpu() -> None:
            path = first / "run.record"
            path.write_text(path.read_text(encoding="utf-8").replace("backend=cpu", "backend=gpu"), encoding="utf-8")

        def mutate_fallback() -> None:
            path = first / "run.record"
            path.write_text(path.read_text(encoding="utf-8").replace("silent_fallback=0", "silent_fallback=1"), encoding="utf-8")

        def mutate_stale() -> None:
            path = first / "proof.bin"
            path.write_bytes(path.read_bytes() + b"stale")

        def mutate_missing() -> None:
            (first / "verification.result").unlink()

        def mutate_metric() -> None:
            path = first / "run.record"
            path.write_text(path.read_text(encoding="utf-8").replace(
                "commitment_time_seconds=0\n", "commitment_time_seconds=0.25\n"), encoding="utf-8")

        for label, mutate in (("gpu", mutate_gpu), ("fallback", mutate_fallback),
                              ("stale", mutate_stale), ("missing", mutate_missing),
                              ("metric", mutate_metric)):
            snapshot = {path: path.read_bytes() for path in first.iterdir() if path.is_file()}
            mutate()
            rejected = invoke(collect, 1)
            for path in list(first.iterdir()):
                if path.is_file() and path not in snapshot:
                    path.unlink()
            for path, contents in snapshot.items():
                path.write_bytes(contents)
            checks.append((label, rejected))

        summary_path = root / "results" / "collected.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["rows"][0]["prove_time_seconds"] = "0.5"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        summary_mismatch = invoke(final, 1)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["rows"] = summary["rows"][1:]
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        summary_missing = invoke(final, 1)
        invoke(collect, 0)
        final_after = invoke(final, 0)
        print(
            f"collector_fixture PASS artifact_bytes={artifact_size} "
            f"{collected.stdout.strip()} {validated.stdout.strip()} "
            f"formal_reject={formal_reject.returncode} "
            f"negative_checks={','.join(label for label, _ in checks)},summary_mismatch,summary_missing "
            f"{final_after.stdout.strip()}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        print(f"collector_fixture FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
