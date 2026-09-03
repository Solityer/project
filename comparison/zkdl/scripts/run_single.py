#!/usr/bin/env python3
"""Run one formal zkDL GAT cell with a fail-closed artifact contract."""
from __future__ import annotations

import argparse
import atexit
import json
import os
import secrets
import shutil
import stat
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
    build_manifest_path,
    cell_path,
    config_digest,
    config_path,
    ensure_contained,
    ensure_directory,
    ensure_regular_file,
    load_resource_manifest,
    parse_config,
    prepare_cell,
    pp_binding_from_manifest,
    pp_path_from_manifest,
    protocol_digest,
    relative_manifest_path,
    resource_inventory_digest,
    read_json,
    sha256_file,
    load_build_manifest,
    validate_pp_file,
    validate_staged_dataset_resources,
    validate_config_resources,
    validate_model_structure_config,
    validate_quantization_config,
)


_PRIVATE_SESSION_TO_CLEAN: Path | None = None


def _harden_private_directory(path: Path, *, create: bool = False) -> Path:
    """Create/check a private directory with no group/other access."""
    path = ensure_contained(path, label="private directory")
    if not path.exists():
        if not create:
            raise FileNotFoundError(f"missing private directory: {path}")
        path.mkdir(parents=True, exist_ok=False, mode=0o700)
    ensure_directory(path, label="private directory")
    if os.name == "posix":
        os.chmod(path, 0o700)
        if stat.S_IMODE(path.stat().st_mode) != 0o700:
            raise PermissionError(f"private directory permissions are not 0700: {path}")
    return path


def _private_file(path: Path, *, label: str) -> Path:
    """Authenticate a private regular file and its enclosing directory."""
    path = ensure_regular_file(path, label=label)
    _harden_private_directory(path.parent)
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise PermissionError(f"{label} permissions are not 0600: {path}")
    return path


def _cleanup_private_session() -> None:
    global _PRIVATE_SESSION_TO_CLEAN
    session = _PRIVATE_SESSION_TO_CLEAN
    _PRIVATE_SESSION_TO_CLEAN = None
    if session is None:
        return
    try:
        session = ensure_contained(session, label="private session")
        if session.is_symlink():
            session.unlink()
        elif session.exists():
            if not session.is_dir():
                session.unlink()
            else:
                shutil.rmtree(session)
    except (OSError, ValueError):
        # Cleanup must never turn a successful proof into a published private
        # artifact.  The native prove command also removes the witness in a
        # finally-equivalent path; this is only the runner-side last resort.
        pass


atexit.register(_cleanup_private_session)


def _new_private_session(dataset: str, device: str) -> tuple[Path, Path]:
    private_root = _harden_private_directory(BASELINE_ROOT / "artifacts" / "private", create=True)
    witness_root = _harden_private_directory(private_root / "witness", create=True)
    dataset_root = _harden_private_directory(witness_root / dataset, create=True)
    device_root = _harden_private_directory(dataset_root / device, create=True)
    for _ in range(16):
        token = f"session-{os.getpid()}-{secrets.token_hex(16)}"
        session = ensure_contained(device_root / token, label="private witness session")
        try:
            session.mkdir(mode=0o700)
        except FileExistsError:
            continue
        _harden_private_directory(session)
        witness = session / "witness.private.bin"
        return session, witness
    raise RuntimeError("failed to allocate a unique private witness session")


def path_arg(path: Path) -> str:
    path = ensure_contained(path, label="formal command path")
    return path.relative_to(BASELINE_ROOT).as_posix()


def native_command(binary: Path, subcommand: str) -> list[object]:
    """Build a native invocation with the mandatory exact baseline root."""
    return [
        path_arg(binary),
        "--project-root",
        str(BASELINE_ROOT),
        subcommand,
        # clap exposes the required root on each subcommand as well as on the
        # top-level parser; keep both positions explicit for native calls.
        "--project-root",
        str(BASELINE_ROOT),
    ]


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
    global _PRIVATE_SESSION_TO_CLEAN
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=CONFIGS, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--project-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=BASELINE_ROOT / "runs")
    parser.add_argument("--pp", type=Path, default=None)
    parser.add_argument("--max-vector-len", type=int, default=None)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    project_root = ensure_contained(args.project_root, label="project-root")
    if project_root != BASELINE_ROOT:
        raise SystemExit("project-root must be comparison/zkdl; stage resources locally first")
    output_root = ensure_contained(args.output_root, label="runs root")
    config = config_path(args.dataset)
    config_values = parse_config(config)
    validate_model_structure_config(config_values)
    validate_quantization_config(config_values)
    # Formal execution accepts only the authenticated manifest emitted by
    # setup_resources.py.  The source policy intentionally carries no PP
    # identity and cannot authorize a proof.
    resource_manifest, resource_manifest_id = load_resource_manifest(require_staged=True)
    validate_config_resources(args.dataset, config_values, resource_manifest)
    validate_staged_dataset_resources(args.dataset, resource_manifest)
    pp_binding = pp_binding_from_manifest(resource_manifest)
    canonical_pp = pp_path_from_manifest(resource_manifest)
    requested_pp = canonical_pp if args.pp is None else ensure_contained(args.pp, label="PP path")
    if requested_pp != canonical_pp:
        raise SystemExit("--pp must equal the canonical PP path authenticated by resource_manifest.json")
    requested_max = pp_binding["pp_max_vector_len"] if args.max_vector_len is None else args.max_vector_len
    if requested_max != pp_binding["pp_max_vector_len"]:
        raise SystemExit("--max-vector-len must equal the authenticated PP max_vector_len")
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
    cpu_build = load_build_manifest("cpu", cpu_binary)
    prover_build = load_build_manifest(args.device, prover_binary)

    try:
        call(
            [
                "python3",
                "scripts/preflight.py",
                "--config",
                path_arg(config),
                "--project-root",
                str(BASELINE_ROOT),
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

    pp_path = canonical_pp
    pp_path = ensure_regular_file(pp_path, label="PP")
    pp_info = json.loads(
        call(
            native_command(cpu_binary, "pp-info")
            + [
                "--pp",
                path_arg(pp_path),
                "--manifest",
                "artifacts/setup/pp_info.json",
            ],
            capture=True,
        ).stdout
    )
    validate_pp_file(pp_path, resource_manifest, pp_info)
    if int(pp_info.get("max_vector_len", 0)) != requested_max:
        raise RuntimeError(
            f"existing PP max_vector_len={pp_info.get('max_vector_len')} differs from requested {requested_max}"
        )

    relation_input = artifacts / "canonical_relation.json"
    statement = artifacts / "statement.bin"
    private_session, witness = _new_private_session(args.dataset, args.device)
    _PRIVATE_SESSION_TO_CLEAN = private_session
    adapter_start = time.monotonic()
    call(
        [
            "python3",
            "scripts/canonical_adapter.py",
            "--config",
            path_arg(config),
            "--project-root",
            str(BASELINE_ROOT),
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
        native_command(cpu_binary, "compile")
        + [
            "--input",
            path_arg(relation_input),
            "--statement",
            path_arg(statement),
            "--witness",
            path_arg(witness),
        ]
    )
    call(
        native_command(cpu_binary, "check-statement")
        + ["--pp", path_arg(pp_path), "--statement", path_arg(statement)]
    )
    compile_time_ms = (time.monotonic() - compile_start) * 1000.0

    model_dir = ensure_contained(BASELINE_ROOT / "artifacts" / "model_registry" / args.dataset, label="model registry directory")
    secret_dir = ensure_contained(BASELINE_ROOT / "artifacts" / "private" / "model_opening" / args.dataset, label="model secret directory")
    model = model_dir / "model_registry.bin"
    secret = secret_dir / "model_opening.private.bin"
    registration_manifest = model_dir / "registration_manifest.json"
    if model.exists() != secret.exists() or model.exists() != registration_manifest.exists():
        raise RuntimeError("registered-model object/secret is incomplete; restore or remove both explicitly")
    if model.exists():
        ensure_regular_file(model, label="model registry")
        _private_file(secret, label="model opening secret")
        ensure_regular_file(registration_manifest, label="model registration manifest")
        registration_info = read_json(registration_manifest, label="model registration manifest")
        if registration_info.get("manifest_schema") != "zkDL_gat_registration_manifest_v1":
            raise RuntimeError("registered-model manifest schema mismatch")
        if registration_info.get("registry_path") != model.relative_to(BASELINE_ROOT).as_posix():
            raise RuntimeError("registered-model manifest path mismatch")
        if registration_info.get("model_registry_id") is None:
            raise RuntimeError("registered-model manifest lacks registry identity")
        for key in (
            "dataset",
            "input_preprocess_id",
            "data_identity",
            "source_input_id",
            "quant_cfg_id",
            "checkpoint_manifest_id",
            "model_shape_id",
            "model_arch_id",
            "model_param_id",
            "graph_features_digest",
            "topology_digest",
        ):
            if registration_info.get(key) != info.get(key):
                raise RuntimeError(f"registered-model manifest identity mismatch: {key}")
    else:
        model_dir.mkdir(parents=True, exist_ok=True)
        _harden_private_directory(secret_dir, create=True)
        call(
            native_command(cpu_binary, "register-model")
            + [
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
                "--manifest",
                path_arg(registration_manifest),
            ]
        )
        ensure_regular_file(model, label="model registry")
        _private_file(secret, label="model opening secret")
        ensure_regular_file(registration_manifest, label="model registration manifest")

    proof = artifacts / "proof.bin"
    proof_manifest = run / "proof_manifest.json"
    verifier_manifest = run / "verifier_manifest.json"
    try:
        call(
            native_command(prover_binary, "prove")
            + [
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
    finally:
        # The native prover removes the witness after deserializing it; this
        # runner-side cleanup covers native crashes and Python exceptions.
        _cleanup_private_session()
    call(
        native_command(cpu_binary, "verify")
        + [
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
        "dataset",
        "input_preprocess_id",
        "data_identity",
        "source_input_id",
        "quant_cfg_id",
        "checkpoint_manifest_id",
        "model_shape_id",
        "model_arch_id",
        "model_param_id",
        "graph_features_digest",
        "topology_digest",
        "proof_sha256",
        "proof_size_bytes",
        "commitment_time_ms",
        "prove_time_ms",
        "model_session_commitments",
        "model_rerandomization_proofs",
    ):
        if proof_info.get(key) != verifier_info.get(key):
            raise RuntimeError(f"prover/verifier manifest mismatch: {key}")
    for key in (
        "dataset",
        "input_preprocess_id",
        "data_identity",
        "quant_cfg_id",
        "checkpoint_manifest_id",
        "model_shape_id",
        "model_arch_id",
        "model_param_id",
        "graph_features_digest",
        "topology_digest",
    ):
        if proof_info.get(key) != info.get(key) or verifier_info.get(key) != info.get(key):
            raise RuntimeError(f"proof identity mismatch: {key}")
    if not isinstance(proof_info.get("source_input_id"), str) or proof_info.get("source_input_id") != verifier_info.get("source_input_id"):
        raise RuntimeError("proof source-input identity is missing or mismatched")
    for key in ("commitment_time_ms", "prove_time_ms"):
        try:
            if float(proof_info.get(key, -1.0)) < 0.0:
                raise ValueError
        except (TypeError, ValueError):
            raise RuntimeError(f"proof timing is malformed: {key}")
    for key, expected in pp_binding.items():
        if proof_info.get(key) != expected:
            raise RuntimeError(f"prover manifest PP binding mismatch: {key}")
        if verifier_info.get(key) != expected:
            raise RuntimeError(f"verifier manifest PP binding mismatch: {key}")
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

    registry_sha256 = sha256_file(model)
    registry_size_bytes = model.stat().st_size
    source_id = cpu_build["source_identity"]
    if prover_build["source_identity"] != source_id:
        raise RuntimeError("CPU and prover binaries were built from different source identities")
    build_identity = {
        "cpu": cpu_build["build_identity"],
        "prover": prover_build["build_identity"],
    }
    provenance = {
        "config_sha256": config_digest(config),
        "protocol_sha256": protocol_digest(),
        "resource_manifest_id": resource_manifest_id,
        "resource_inventory_digest": resource_inventory_digest(args.dataset, resource_manifest),
        "preflight_sha256": sha256_file(preflight),
        "canonical_relation_sha256": sha256_file(relation_input),
        "statement_sha256": sha256_file(statement),
        "proof_manifest_sha256": sha256_file(proof_manifest),
        "verifier_manifest_sha256": sha256_file(verifier_manifest),
        "cpu_binary_sha256": cpu_build["binary_sha256"],
        "prover_binary_sha256": prover_build["binary_sha256"],
        "cpu_binary_size_bytes": cpu_build["binary_size_bytes"],
        "prover_binary_size_bytes": prover_build["binary_size_bytes"],
        "cpu_build_manifest_sha256": sha256_file(build_manifest_path("cpu")),
        "prover_build_manifest_sha256": sha256_file(build_manifest_path(args.device)),
        "cpu_build_identity": cpu_build["build_identity"],
        "prover_build_identity": prover_build["build_identity"],
        "source_identity": source_id,
        "input_preprocess_id": info["input_preprocess_id"],
        "data_identity": info["data_identity"],
        "quant_cfg_id": info["quant_cfg_id"],
        "source_input_id": proof_info.get("source_input_id"),
        "registry_sha256": registry_sha256,
        "registry_size_bytes": registry_size_bytes,
        **pp_binding,
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
        **pp_binding,
        "resource_manifest_id": resource_manifest_id,
        "config_sha256": config_digest(config),
        "protocol_sha256": protocol_digest(),
        "integer_bound_ledger_id": info["ledger"]["integer_bound_ledger_id"],
        "input_preprocess_id": info["input_preprocess_id"],
        "data_identity": info["data_identity"],
        "quant_cfg_id": info["quant_cfg_id"],
        "source_input_id": proof_info.get("source_input_id"),
        "L": info["L"],
        "hidden_profile": info["hidden_profile"],
        "d_in_profile": info["d_in_profile"],
        "K_out": info["K_out"],
        "checkpoint_binding": info["checkpoint_binding"],
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
        "registry_sha256": registry_sha256,
        "registry_size_bytes": registry_size_bytes,
        "source_identity": source_id,
        "build_identity": build_identity,
        "public_instance": info["public_instance"],
        "proof_sha256": proof_info["proof_sha256"],
        "proof_size_bytes": proof_info["proof_size_bytes"],
        "adapter_time_ms": adapter_time_ms,
        "compile_time_ms": compile_time_ms,
        "prove_time_ms": proof_info["prove_time_ms"],
        "commitment_time_ms": proof_info["commitment_time_ms"],
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
        "proof_size_bytes": proof_info["proof_size_bytes"],
        "statement_id": proof_info["statement_id"],
        "model_registry_id": proof_info["model_registry_id"],
        **{key: result_payload[key] for key in pp_binding},
        "registry_sha256": result_payload["registry_sha256"],
        "registry_size_bytes": result_payload["registry_size_bytes"],
        "source_identity": result_payload["source_identity"],
        "build_identity": result_payload["build_identity"],
        "input_preprocess_id": result_payload["input_preprocess_id"],
        "quant_cfg_id": result_payload["quant_cfg_id"],
        "data_identity": result_payload["data_identity"],
        "source_input_id": result_payload["source_input_id"],
        "config_sha256": result_payload["config_sha256"],
        "protocol_sha256": result_payload["protocol_sha256"],
        "resource_manifest_id": result_payload["resource_manifest_id"],
        "resource_inventory_digest": result_payload["resource_inventory_digest"],
        "integer_bound_ledger_id": result_payload["integer_bound_ledger_id"],
        "L": result_payload["L"],
        "hidden_profile": result_payload["hidden_profile"],
        "d_in_profile": result_payload["d_in_profile"],
        "K_out": result_payload["K_out"],
        "checkpoint_binding": result_payload["checkpoint_binding"],
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
        "commitment_time_ms": result_payload["commitment_time_ms"],
        "prove_time_ms": result_payload["prove_time_ms"],
        "gpu_telemetry": result_payload["gpu_telemetry"],
        "provenance": result_payload["provenance"],
        "files": before_success,
    }
    atomic_write_json(run / "success_manifest.json", success_payload)
    artifact_inventory(run, require_complete=True)
    print(f"ZKDL_CELL_OK dataset={args.dataset} device={args.device} proof_sha256={proof_info['proof_sha256']}")
    return 0


def failure_result(error: BaseException) -> None:
    _cleanup_private_session()

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
