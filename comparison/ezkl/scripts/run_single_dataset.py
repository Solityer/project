#!/usr/bin/env python3
"""Run one ezkl GAT CPU/GPU formal cell with explicit fail-closed gates.

Without ``--execute-formal`` this command is a non-executing plan.  A formal
cell runs native setup first, then consumes an externally produced public
statement/output, finalizes the setup binding, and only then invokes the
official witness, prove, and verify commands.  No statement or public output
is synthesized by this runner.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from formal_common import (
    BACKENDS,
    CANONICAL_OUTPUT_MODE,
    CANONICAL_PRIVACY,
    CANONICAL_PROOF_SCOPE,
    CANONICAL_RELATION,
    CANONICAL_STATEMENT_MODE,
    CONTRACT,
    ROOT,
    ContractError,
    abort,
    canonical_json_digest,
    load_cli_contract,
    load_json,
    local_path,
    print_error,
    reject_forbidden_path,
    reject_reduced_srs,
    require_digest,
    require_formal_srs,
    require_file,
    require_circuit_ref,
    public_output_values,
    require_exact_keys,
    resource_manifest_hash,
    sha256,
    source_tree_hash,
    validate_artifact_identity,
    validate_config,
    validate_gpu_environment,
    validate_gpu_evidence,
    validate_native_srs_manifest,
    validate_public_statement,
    validate_resources,
)


DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--backend", choices=BACKENDS, required=True)
    p.add_argument("--execute-formal", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--reuse-setup",
        action="store_true",
        help="reuse an existing setup directory created before the external gat-produce step",
    )
    p.add_argument("--binary", default="target/release/ezkl")
    p.add_argument("--config")
    p.add_argument("--public-statement", help="external verifier-facing public statement")
    p.add_argument("--public-output", help="external verifier-facing public output/Y")
    p.add_argument("--producer-manifest", help="external producer provenance/digest manifest")
    p.add_argument("--settings")
    p.add_argument("--compiled-circuit")
    p.add_argument("--vk")
    p.add_argument("--pk")
    p.add_argument("--artifact-manifest")
    p.add_argument("--output-dir")
    return p


def default_config(dataset: str) -> Path:
    return ROOT / "resources" / "canonical" / "configs" / f"{dataset}_secure.cfg"


def default_path(output: Path, name: str) -> Path:
    return output / name


def baseline_relative(path: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        abort(f"{label} is outside baseline root: {path}")


def run_command(command: list[str], log_path: Path, *, phase: str, env: dict[str, str]) -> float:
    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = (time.monotonic() - start) * 1000.0
    log_path.write_text(completed.stdout)
    if completed.returncode != 0:
        abort(f"{phase} failed with exit code {completed.returncode}; see {log_path}")
    return elapsed


def ensure_relative_arg(value: str | None, label: str, default: Path | None = None) -> Path:
    if value is None:
        if default is None:
            abort(f"{label} is required")
        return default
    return local_path(value, label)


def read_proof_instances(proof: Path, *, statement_sha256: str, output_sha256: str) -> dict[str, Any]:
    """Extract native proof instances without recomputing the GAT relation."""

    document = load_json(proof, "verifier-facing proof")
    instances = document.get("instances")
    if not isinstance(instances, list) or not instances or any(not isinstance(item, list) for item in instances):
        abort("verifier-facing proof has no native instances array")
    count = sum(len(item) for item in instances)
    if count <= 0:
        abort("verifier-facing proof has no native public instances")
    return {
        "proof_field": "instances",
        "encoding": "canonical-json",
        "sha256": canonical_json_digest(instances),
        "count": count,
        "public_statement_sha256": statement_sha256,
        "public_output_sha256": output_sha256,
        "native_binding": True,
    }


def validate_core_manifest(
    path: Path,
    *,
    dataset: str,
    backend: str,
    statement: Path,
    public_output: Path,
    circuit_ref: dict[str, str],
) -> dict[str, Any]:
    core = load_json(path, "core run manifest")
    required = {
        "schema_version",
        "relation_id",
        "dataset",
        "backend",
        "proof_scope",
        "privacy_policy",
        "public_statement_mode",
        "output_mode",
        "native_verify_result",
        "native_public_instance_binding",
        "verifier_reads_witness",
        "proof_contains_private_witness",
        "public_statement_sha256",
        "public_output_sha256",
        "proof_instances_sha256",
        "proof_instance_count",
        "circuit_ref",
    }
    require_exact_keys(core, required, "core run manifest")
    if core["schema_version"] != 1 or core["relation_id"] != CANONICAL_RELATION:
        abort("core run manifest has an unsupported schema or relation")
    if core["dataset"] != dataset or core["backend"] != backend:
        abort("core run manifest dataset/backend mismatch")
    if core["proof_scope"] != CANONICAL_PROOF_SCOPE:
        abort("core run manifest is not full_gat")
    if core["privacy_policy"] != CANONICAL_PRIVACY:
        abort("core run manifest privacy policy mismatch")
    if core["public_statement_mode"] != CANONICAL_STATEMENT_MODE:
        abort("core run manifest did not consume external public statement")
    if core["output_mode"] != CANONICAL_OUTPUT_MODE:
        abort("core run manifest output mode mismatch")
    if core["native_verify_result"] is not True:
        abort("native ezkl verifier did not report success")
    if core["native_public_instance_binding"] is not True:
        abort("core run manifest does not attest native public-instance binding")
    if core["verifier_reads_witness"] is not False:
        abort("core run manifest does not prove verifier witness isolation")
    if core["proof_contains_private_witness"] is not False:
        abort("core run manifest indicates witness leakage")
    if core["public_statement_sha256"] != sha256(statement):
        abort("core run manifest public statement identity mismatch")
    if core["public_output_sha256"] != sha256(public_output):
        abort("core run manifest public output identity mismatch")
    if not isinstance(core["proof_instances_sha256"], str):
        abort("core run manifest has no proof instance digest")
    require_digest(core["proof_instances_sha256"], "core proof instance digest")
    if not isinstance(core["proof_instance_count"], int) or core["proof_instance_count"] <= 0:
        abort("core run manifest has no positive proof instance count")
    if require_circuit_ref(core["circuit_ref"], "core run manifest circuit_ref") != circuit_ref:
        abort("core run manifest circuit_ref mismatch")
    return core


def print_plan(args: argparse.Namespace, config: Path, output: Path) -> None:
    print("formal execution not requested; no setup, statement wait, final, prove, verify or full dataset work was run")
    print(f"dataset={args.dataset} backend={args.backend}")
    print(f"config={baseline_relative(config, 'config')}")
    print(f"output_dir={baseline_relative(output, 'output-dir')}")
    print("execution order: gat-prepare --phase setup -> independent public-statement producer -> gat-prepare --phase final -> gat-witness -> prove -> verify")
    print("rerun with --execute-formal, --public-statement, --public-output and --producer-manifest; native SRS comes from the authenticated manifest")


def main(args: argparse.Namespace) -> int:
    reject_reduced_srs(sys.argv[1:])
    config = ensure_relative_arg(args.config, "config", default_config(args.dataset))
    output = ensure_relative_arg(
        args.output_dir,
        "output-dir",
        ROOT / "runs" / "formal" / args.dataset / args.backend,
    )
    if not args.execute_formal:
        print_plan(args, config, output)
        return 0

    if args.dry_run:
        abort("--dry-run cannot be combined with --execute-formal")
    if not args.public_statement or not args.public_output or not args.producer_manifest:
        abort("--execute-formal requires --public-statement, --public-output and --producer-manifest")
    statement = ensure_relative_arg(args.public_statement, "public-statement")
    public_output = ensure_relative_arg(args.public_output, "public-output")
    producer_manifest = ensure_relative_arg(args.producer_manifest, "producer-manifest")
    reject_forbidden_path(statement, "public-statement")
    reject_forbidden_path(public_output, "public-output")
    reject_forbidden_path(producer_manifest, "producer-manifest")
    require_file(statement, "external public statement")
    require_file(public_output, "external public output Y")
    require_file(producer_manifest, "external producer manifest")
    public_values = public_output_values(public_output, dataset=args.dataset)
    validate_public_statement(
        statement,
        dataset=args.dataset,
        public_output_sha256=sha256(public_output),
    )
    validate_resources()
    contract = load_cli_contract()
    values = validate_config(config, args.dataset)
    if args.backend == "gpu":
        gpu_probe = validate_gpu_environment()
    else:
        gpu_probe = None

    validate_native_srs_manifest()
    srs = local_path(values["srs_path"], "config srs_path")
    require_formal_srs(srs)
    trusted_srs_sha = require_digest(values["trusted_srs_sha256"], "config trusted_srs_sha256")
    if sha256(srs) != trusted_srs_sha:
        abort("trusted SRS SHA-256 does not match the selected SRS")
    if values.get("trusted_srs_sha256") != trusted_srs_sha:
        abort("canonical secure config trusted_srs_sha256 does not match the selected SRS")

    binary = ensure_relative_arg(args.binary, "binary")
    require_file(binary, "official ezkl binary")
    if not os.access(binary, os.X_OK):
        abort(f"official ezkl binary is not executable: {binary}")
    build_receipt = ROOT / "artifacts" / "build" / args.backend / "build_receipt.json"
    receipt = json.loads(require_file(build_receipt, "backend build receipt").read_text())
    if receipt.get("backend") != args.backend:
        abort("build receipt backend does not match requested backend")
    if receipt.get("binary_sha256") != sha256(binary):
        abort("build receipt binary SHA-256 does not match current binary")
    if receipt.get("source_tree_sha256") != source_tree_hash():
        abort("build receipt source tree identity is stale")
    if receipt.get("resource_manifest_sha256") != resource_manifest_hash():
        abort("build receipt resource identity is stale")
    if receipt.get("adapter_contract_id") != contract["contract_id"]:
        abort("build receipt adapter contract identity is stale")
    if receipt.get("adapter_contract_sha256") != sha256(CONTRACT):
        abort("build receipt adapter contract digest is stale")
    features = receipt.get("features", [])
    if args.backend == "gpu" and "gpu-accelerated" not in features:
        abort("GPU build receipt lacks the gpu-accelerated feature")
    if args.backend == "cpu" and "gpu-accelerated" in features:
        abort("CPU run cannot use a GPU-featured build receipt")

    if args.reuse_setup:
        if not output.is_dir() or output.is_symlink():
            abort(f"--reuse-setup requires an existing regular output directory: {output}")
    else:
        if output.exists() or output.is_symlink():
            abort(f"output-dir already exists; run setup and independent producer first, then use --reuse-setup: {output}")
        output.mkdir(parents=True)
    logs = output / "logs"
    if logs.exists() and logs.is_symlink():
        abort(f"logs directory must not be a symlink: {logs}")
    logs.mkdir(exist_ok=True)
    settings = ensure_relative_arg(args.settings, "settings", default_path(output, "settings.json"))
    compiled = ensure_relative_arg(args.compiled_circuit, "compiled-circuit", default_path(output, "compiled.circuit"))
    vk = ensure_relative_arg(args.vk, "vk", default_path(output, "vk.key"))
    pk = ensure_relative_arg(args.pk, "pk", default_path(output, "pk.key"))
    artifact_manifest = ensure_relative_arg(
        args.artifact_manifest, "artifact-manifest", default_path(output, "artifact_manifest.json")
    )
    generated_artifact_manifest = output / "artifact_manifest.json"
    if artifact_manifest != generated_artifact_manifest:
        abort("artifact-manifest must be the native gat-prepare artifact_manifest.json")
    witness = output / "witness.json"
    proof = output / "proof.json"
    env = os.environ.copy()
    env["EZKL_WORKING_DIR"] = str(ROOT)
    env["EZKL_GAT_BACKEND"] = args.backend
    env["EZKL_GAT_RUN_DIR"] = str(output)
    if args.backend == "gpu":
        env["EZKL_GAT_REQUIRE_GPU_EVIDENCE"] = "1"

    def cli_path(path: Path, label: str) -> str:
        return baseline_relative(path, label)

    setup_ms = 0.0
    if not args.reuse_setup:
        setup_command = [
            str(binary),
            contract["prepare_command"],
            "--dataset",
            args.dataset,
            "--phase",
            "setup",
            "--resource-manifest",
            "resources/resource_manifest.json",
            "--output-dir",
            cli_path(output, "output-dir"),
            "--srs-path",
            cli_path(srs, "SRS"),
        ]
        if args.backend == "gpu":
            setup_command.append("--gpu")
        setup_ms = run_command(
            setup_command,
            logs / "setup.log",
            phase="GAT prepare setup (settings/compile/setup)",
            env=env,
        )
    for path, label in (
        (settings, "settings"),
        (compiled, "compiled circuit"),
        (vk, "verification key"),
        (pk, "proving key"),
    ):
        require_file(path, label)

    require_file(output / "circuit_ref.json", "setup circuit reference")
    require_file(output / "producer_context.json", "producer setup context")
    if producer_manifest.parent != output:
        abort("producer-manifest must be inside output-dir")

    if "qat_best_logits" in public_output.name or "qat_best_logits" in public_output.as_posix():
        abort("qat_best_logits is not a formal public output")
    env["EZKL_GAT_PUBLIC_STATEMENT"] = str(statement)
    env["EZKL_GAT_PUBLIC_OUTPUT"] = str(public_output)

    final_command = [
        str(binary),
        contract["prepare_command"],
        "--dataset",
        args.dataset,
        "--phase",
        "final",
        "--resource-manifest",
        "resources/resource_manifest.json",
        "--output-dir",
        cli_path(output, "output-dir"),
        "--srs-path",
        cli_path(srs, "SRS"),
        "--public-statement",
        cli_path(statement, "public statement"),
        "--public-output",
        cli_path(public_output, "public output"),
        "--producer-manifest",
        cli_path(producer_manifest, "producer manifest"),
    ]
    if args.backend == "gpu":
        final_command.append("--gpu")
    final_ms = run_command(
        final_command,
        logs / "final.log",
        phase="GAT prepare final (producer statement binding)",
        env=env,
    )
    require_file(artifact_manifest, "artifact identity manifest")
    prepare_ms = setup_ms + final_ms
    artifact = validate_artifact_identity(
        artifact_manifest,
        dataset=args.dataset,
        backend=args.backend,
        binary=binary,
        srs=srs,
        settings=settings,
        compiled=compiled,
        vk=vk,
        pk=pk,
    )
    circuit_ref = require_circuit_ref(artifact.get("circuit_ref"), "artifact circuit_ref")
    statement_document = validate_public_statement(
        statement,
        dataset=args.dataset,
        circuit_ref=circuit_ref,
        public_output_sha256=sha256(public_output),
        quant_cfg_id=values["quant_cfg_id"],
        static_tables_id=values["static_table_id"],
    )
    if statement_document["public_ezkl"]["y"] != public_values:
        abort("external public output Y disagrees element-by-element with statement")
    if artifact["statement_sha256"] != sha256(statement):
        abort("artifact statement_sha256 does not bind the selected external statement")
    if artifact["producer_manifest_sha256"] != sha256(producer_manifest):
        abort("artifact producer_manifest_sha256 does not bind the selected producer manifest")
    artifact_file_records = {record["path"]: record for record in artifact["files"]}
    for name, digest in (("statement.json", sha256(statement)), ("public_output.json", sha256(public_output))):
        if name not in artifact_file_records or artifact_file_records[name]["sha256"] != digest:
            abort(f"artifact file {name} does not bind the selected external public boundary")
    for name, digest in (
        ("producer_context.json", sha256(output / "producer_context.json")),
        ("producer_manifest.json", sha256(producer_manifest)),
    ):
        if name not in artifact_file_records or artifact_file_records[name]["sha256"] != digest:
            abort(f"artifact file {name} does not bind the independent producer chain")

    witness_ms = run_command(
        [
            str(binary),
            contract["witness_command"],
            "--dataset",
            args.dataset,
            "--resource-manifest",
            "resources/resource_manifest.json",
            "--compiled-circuit",
            cli_path(compiled, "compiled circuit"),
            "--statement",
            cli_path(statement, "public statement"),
            "--output",
            cli_path(witness, "witness"),
            *( ["--gpu"] if args.backend == "gpu" else [] ),
        ],
        logs / "witness.log",
        phase="GAT witness generation",
        env=env,
    )
    require_file(witness, "private witness")

    prove_ms = run_command(
        [
            str(binary),
            contract["prove_command"],
            "--witness",
            cli_path(witness, "witness"),
            "--compiled-circuit",
            cli_path(compiled, "compiled circuit"),
            "--pk-path",
            cli_path(pk, "proving key"),
            "--proof-path",
            cli_path(proof, "proof"),
            "--srs-path",
            cli_path(srs, "SRS"),
            "--check-mode",
            "safe",
        ],
        logs / "prove.log",
        phase="official ezkl prove",
        env=env,
    )
    require_file(proof, "verifier-facing proof")

    verify_ms = run_command(
        [
            str(binary),
            contract["verify_command"],
            "--settings-path",
            cli_path(settings, "settings"),
            "--proof-path",
            cli_path(proof, "proof"),
            "--vk-path",
            cli_path(vk, "verification key"),
            "--srs-path",
            cli_path(srs, "SRS"),
            "--statement",
            cli_path(statement, "public statement"),
            "--public-output",
            cli_path(public_output, "public output"),
            "--artifact-manifest",
            cli_path(artifact_manifest, "artifact identity manifest"),
            "--resource-manifest",
            "resources/resource_manifest.json",
        ],
        logs / "verify.log",
        phase="independent official ezkl verify",
        env=env,
    )

    core_manifest = output / "core_run_manifest.json"
    core = validate_core_manifest(
        core_manifest,
        dataset=args.dataset,
        backend=args.backend,
        statement=statement,
        public_output=public_output,
        circuit_ref=circuit_ref,
    )
    proof_instances = read_proof_instances(
        proof,
        statement_sha256=sha256(statement),
        output_sha256=sha256(public_output),
    )
    if core["proof_instances_sha256"] != proof_instances["sha256"]:
        abort("core run manifest proof instance digest mismatch")
    if core["proof_instance_count"] != proof_instances["count"]:
        abort("core run manifest proof instance count mismatch")

    evidence_path = output / "gpu_evidence.json"
    evidence = validate_gpu_evidence(evidence_path, backend=args.backend)
    if core["native_verify_result"] is not True or core["native_public_instance_binding"] is not True:
        abort("native verify/public-instance binding attestation is not successful")
    status = "VERIFY_OK"
    result = {
        "schema_version": 1,
        "status": status,
        "dataset": args.dataset,
        "backend": args.backend,
        "proof_scope": "full_gat",
        "privacy_policy": "model_and_trace_hidden",
        "public_statement_mode": "external",
        "output_mode": "public_logits",
        "relation_id": "R_GAT-ezkl",
        "quant_cfg_id": values["quant_cfg_id"],
        "static_tables_id": values["static_table_id"],
        "public_statement_sha256": sha256(statement),
        "public_output_sha256": sha256(public_output),
        "srs_sha256": trusted_srs_sha,
        "srs_provenance": artifact["srs_provenance"],
        "settings_sha256": sha256(settings),
        "compiled_circuit_sha256": sha256(compiled),
        "vk_sha256": sha256(vk),
        "pk_sha256": sha256(pk),
        "proof_sha256": sha256(proof),
        "proof_size_bytes": proof.stat().st_size,
        "proof_path": baseline_relative(proof, "proof"),
        "public_statement_path": baseline_relative(statement, "public statement"),
        "public_output_path": baseline_relative(public_output, "public output"),
        "srs_path": baseline_relative(srs, "srs"),
        "settings_path": baseline_relative(settings, "settings"),
        "compiled_circuit_path": baseline_relative(compiled, "compiled circuit"),
        "vk_path": baseline_relative(vk, "verification key"),
        "pk_path": baseline_relative(pk, "proving key"),
        "binary_path": baseline_relative(binary, "binary"),
        "artifact_manifest_path": baseline_relative(artifact_manifest, "artifact identity manifest"),
        "artifact_manifest_sha256": sha256(artifact_manifest),
        "circuit_ref": circuit_ref,
        "proof_instances": proof_instances,
        "native_verify_result": core["native_verify_result"],
        "native_public_instance_binding": core["native_public_instance_binding"],
        "binary_sha256": sha256(binary),
        "source_tree_sha256": source_tree_hash(),
        "resource_manifest_sha256": resource_manifest_hash(),
        "verifier_reads_witness": False,
        "proof_contains_private_witness": False,
        "core_run_manifest_sha256": sha256(core_manifest),
        "phase_timings_ms": {
            "prepare_compile_setup": round(prepare_ms, 3),
            "witness_generation": round(witness_ms, 3),
            "prove": round(prove_ms, 3),
            "verify": round(verify_ms, 3),
        },
        "gpu_evidence": evidence,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        parsed = parser().parse_args()
        raise SystemExit(main(parsed))
    except (ContractError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise SystemExit(print_error(exc))
