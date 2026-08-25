#!/usr/bin/env python3
"""Single-dataset formal ezkl/Halo2 GAT runner.

The independent public statement is emitted by witness generation and supplied
as a separate verifier input.  The proof JSON is never used as the statement
source.  CPU and CUDA use separately built binaries and no fallback exists.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASETS_FILE = ROOT / "configs" / "full_datasets.json"
SECURITY_PROFILE_TEMPLATE = ROOT / "configs" / "security_profile.json"
FORMAL_ROUTE = "ezkl_gat_native_compressed_csr_attention_full"
FORMAL_SOURCE_SRS_MAX_K = 28
RESULTS_ROOT = ROOT / "results"
RUNS_ROOT = RESULTS_ROOT / "runs"
RUNS_TMP_ROOT = RESULTS_ROOT / "runs_tmp"
LOGS_ROOT = RESULTS_ROOT / "logs"

from resource_policy import (  # noqa: E402
    CELL_ARTIFACTS,
    DATASET_ORDER,
    atomic_write_json,
    canonical_cell_dir,
    check_resource_layout,
    resolve_local,
    validate_cell_inventory,
)
from build_identity_contract import (  # noqa: E402
    BuildIdentityError,
    identity_digest,
    validate_build_identity,
)
from formal_source_manifest import checked_path, formal_source_id  # noqa: E402


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
    return resolve_local(root, value, "formal input path")


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
    atomic_write_json(path, payload)


def parse_native_json(stdout: str, stage: str) -> dict[str, Any]:
    """Parse the final JSON object emitted by one native formal command."""

    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError(f"{stage} did not emit a native JSON result")


def source_metadata(payload: dict[str, Any], stage: str) -> dict[str, Any]:
    fields = (
        "semantic_source_id",
        "formal_source_manifest_id",
        "source_tree_digest",
        "vendor_tree_digest",
        "prover_backend",
        "verifier_backend",
    )
    result = {field: payload.get(field) for field in fields}
    expected = formal_source_id(ROOT)
    for field in ("semantic_source_id", "formal_source_manifest_id", "source_tree_digest"):
        if result[field] != expected:
            raise RuntimeError(f"{stage} source identity is not current: {field}")
    if not result["vendor_tree_digest"] or not str(result["vendor_tree_digest"]).startswith("sha256:"):
        raise RuntimeError(f"{stage} vendor source identity is missing")
    return result


def validate_gpu_telemetry(payload: Any, scope: str) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{scope} did not return scoped CUPTI telemetry")
    required = {
        "schema": "ezkl_cuda_kernel_telemetry_v1",
        "scope": scope,
        "kernel_count": int,
        "dropped_records": 0,
        "active_device_id": 0,
        "kernel_names": list,
    }
    for key, expected in required.items():
        actual = payload.get(key)
        if expected is int:
            if not isinstance(actual, int) or actual <= 0:
                raise RuntimeError(f"{scope} CUPTI kernel_count is not positive")
        elif isinstance(expected, type):
            if not isinstance(actual, expected):
                raise RuntimeError(f"{scope} CUPTI telemetry field {key} is invalid")
        elif actual != expected:
            raise RuntimeError(f"{scope} CUPTI telemetry field {key} is invalid")
    if (
        not payload.get("scope_nonce")
        or payload.get("scope_end_ns", 0) < payload.get("scope_start_ns", 1)
        or payload.get("active_device_id") != 0
    ):
        raise RuntimeError(f"{scope} CUPTI scope nonce/timestamps are invalid")
    names = payload["kernel_names"]
    if not names or any(
        not isinstance(item, dict)
        or not item.get("name")
        or not item.get("category")
        or not isinstance(item.get("count"), int)
        or item["count"] <= 0
        or item.get("device_id") != 0
        for item in names
    ):
        raise RuntimeError(f"{scope} CUPTI kernel identity inventory is incomplete")
    if sum(item["count"] for item in names) != payload["kernel_count"]:
        raise RuntimeError(f"{scope} CUPTI kernel inventory count mismatch")


def parse_args() -> argparse.Namespace:
    datasets = load_datasets()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(datasets))
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--srs-path", type=Path, required=True,
                        help="trusted ceremony KZG SRS whose declared source_srs_max_k covers the circuit")
    parser.add_argument("--ceremony-descriptor", type=Path, required=True,
                        help="descriptor binding the ceremony id, exact SRS SHA-256 and original max_k")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-logrows", type=int, default=FORMAL_SOURCE_SRS_MAX_K)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--run",
        action="store_true",
        help="continue through settings, compile, setup, witness, proof and verify",
    )
    return parser.parse_args()


def _clean_canonical_cell(path: Path, parent: Path) -> None:
    """Remove only an exact cell containing allowed one-link regular files."""

    path = Path(path)
    parent = Path(parent)
    try:
        checked_path(Path("/"), parent, "canonical result parent")
        checked = checked_path(Path("/"), path, "canonical result cell", allow_missing=True)
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    if not checked.exists():
        return
    parent_resolved = parent.resolve()
    if checked == parent_resolved or parent_resolved not in checked.parents:
        raise RuntimeError(f"refusing to clean path outside canonical result root: {checked}")
    metadata = checked.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"canonical result cell is not a directory: {checked}")
    entries = list(checked.iterdir())
    for entry in entries:
        item = entry.lstat()
        if not stat.S_ISREG(item.st_mode) or stat.S_ISLNK(item.st_mode):
            raise RuntimeError(f"refusing to clean unsafe cell entry: {entry}")
        if item.st_nlink != 1:
            raise RuntimeError(f"refusing to clean hard-linked cell entry: {entry}")
        if entry.name not in CELL_ARTIFACTS:
            raise RuntimeError(f"refusing to clean extra cell entry: {entry.name}")
    for entry in entries:
        entry.unlink()
    checked.rmdir()


def _safe_remove_work_tree(path: Path, parent: Path) -> None:
    """Recursively remove only a controlled temporary work tree."""

    path = Path(path)
    parent = Path(parent)
    try:
        checked_path(Path("/"), parent, "temporary work parent")
        checked = checked_path(Path("/"), path, "temporary work tree", allow_missing=True)
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    if not checked.exists():
        return
    parent_resolved = parent.resolve()
    if checked == parent_resolved or parent_resolved not in checked.parents:
        raise RuntimeError(f"refusing to remove temporary path outside root: {checked}")
    if not stat.S_ISDIR(checked.lstat().st_mode):
        raise RuntimeError(f"temporary work root is not a directory: {checked}")

    def remove_tree(directory: Path) -> None:
        for entry in list(directory.iterdir()):
            item = entry.lstat()
            if stat.S_ISLNK(item.st_mode):
                entry.unlink()
            elif stat.S_ISDIR(item.st_mode):
                remove_tree(entry)
            elif stat.S_ISREG(item.st_mode):
                if item.st_nlink != 1:
                    raise RuntimeError(f"refusing to clean hard-linked work file: {entry}")
                entry.unlink()
            else:
                raise RuntimeError(f"refusing to clean special work entry: {entry}")
        directory.rmdir()

    remove_tree(checked)


def _prepare_stage(dataset: str, device: str, final_dir: Path, clean: bool) -> tuple[Path, Path]:
    """Create a private proof stage and optionally clean only this cell."""

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    if final_dir.is_symlink():
        raise RuntimeError(f"canonical result cell is a symlink: {final_dir}")
    if clean:
        _clean_canonical_cell(final_dir, RUNS_ROOT)
    elif final_dir.exists():
        raise RuntimeError(
            f"canonical result cell already exists; use --clean to replace exactly {final_dir}"
        )
    # Every invocation gets an isolated stage.  It is removed in main's
    # finally block, so no work directory can become part of a result cell.
    stage = (RUNS_TMP_ROOT / f"{device}-{dataset}-{uuid.uuid4().hex}").resolve()
    if RUNS_TMP_ROOT.resolve() not in stage.parents:
        raise RuntimeError("temporary proof stage escaped results/runs_tmp")
    stage.mkdir(parents=True, exist_ok=False)
    cell = stage / "cell"
    cell.mkdir()
    return stage, cell


def _rewrite_stage_paths(value: Any, stage_root: Path, final_dir: Path) -> Any:
    """Make committed command provenance refer to the canonical result cell."""

    stage_text = str(stage_root)
    final_text = str(final_dir)
    if isinstance(value, str):
        return value.replace(stage_text, final_text)
    if isinstance(value, list):
        return [_rewrite_stage_paths(item, stage_root, final_dir) for item in value]
    if isinstance(value, dict):
        return {
            key: _rewrite_stage_paths(item, stage_root, final_dir)
            for key, item in value.items()
        }
    return value


def main() -> int:
    args = parse_args()
    if args.max_logrows != FORMAL_SOURCE_SRS_MAX_K:
        raise RuntimeError("formal ezkl requires --max-logrows=28; derived settings remain below the exact source SRS")
    entry = load_datasets()[args.dataset]
    # Formal data and checkpoint inputs are baseline-local resources.  SRS and
    # ceremony descriptors remain explicit trusted ceremony inputs and are not
    # silently copied into the baseline.
    config = resolve(ROOT, args.config or entry["config"])
    dataset_root = resolve(ROOT, args.dataset_root or entry["dataset_root"])
    checkpoint = resolve(ROOT, args.checkpoint or entry["checkpoint"])
    try:
        srs_input = args.srs_path.expanduser()
        descriptor_input = args.ceremony_descriptor.expanduser()
        if not srs_input.is_absolute():
            srs_input = Path.cwd() / srs_input
        if not descriptor_input.is_absolute():
            descriptor_input = Path.cwd() / descriptor_input
        srs_path = checked_path(Path("/"), srs_input, "trusted ceremony SRS")
        ceremony_descriptor = checked_path(
            Path("/"), descriptor_input, "trusted ceremony descriptor"
        )
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    default_run = canonical_cell_dir(ROOT, args.device, args.dataset)
    run_dir = (args.output_dir.resolve() if args.output_dir else default_run).resolve()
    if run_dir != default_run:
        raise RuntimeError(
            "output directory must be exactly results/runs/<backend>/<dataset>"
        )
    # This is a path-only preflight.  It checks names and presence but never
    # opens full dataset/checkpoint contents; the native adapter performs the
    # authoritative authenticated loading later.
    check_resource_layout(ROOT, [args.dataset], require_staged=True)
    stage_root, work_dir = _prepare_stage(args.dataset, args.device, run_dir, args.clean)
    run_dir = work_dir
    final_run_dir = default_run
    committed = False

    def cleanup_stage() -> None:
        if not committed:
            try:
                _safe_remove_work_tree(stage_root, RUNS_TMP_ROOT)
            except Exception:
                pass

    atexit.register(cleanup_stage)

    adapter = ROOT / "build_gat_adapter" / "gat_ezkl_export"
    prover_binary = ROOT / (
        "target-gpu/release/ezkl_formal"
        if args.device == "cuda"
        else "target/release/ezkl_formal"
    )
    verifier_binary = ROOT / "target/release/ezkl_formal"
    prover_build_identity = ROOT / "build_identity" / f"{args.device}.json"
    verifier_build_identity = ROOT / "build_identity" / "cpu.json"
    if not adapter.is_file():
        raise RuntimeError(f"missing formal adapter; run scripts/build.sh: {adapter}")
    if args.run and not prover_binary.is_file():
        target = "GPU" if args.device == "cuda" else "CPU"
        raise RuntimeError(f"missing distinct {target} ezkl binary: {prover_binary}")
    if args.run and not verifier_binary.is_file():
        raise RuntimeError(f"missing independent CPU verifier binary: {verifier_binary}")
    if args.run and not prover_build_identity.is_file():
        raise RuntimeError(
            f"missing authenticated {args.device} prover build identity: {prover_build_identity}"
        )
    if args.run and not verifier_build_identity.is_file():
        raise RuntimeError(
            f"missing authenticated CPU verifier build identity: {verifier_build_identity}"
        )
    prover_identity: dict[str, Any] | None = None
    verifier_identity: dict[str, Any] | None = None
    if args.run:
        try:
            prover_identity = validate_build_identity(
                ROOT,
                prover_build_identity,
                backend=args.device,
                binary=prover_binary,
                adapter=adapter,
            )
            verifier_identity = validate_build_identity(
                ROOT,
                verifier_build_identity,
                backend="cpu",
                binary=verifier_binary,
                adapter=adapter,
            )
        except BuildIdentityError as exc:
            raise RuntimeError(f"authenticated formal build identity rejected: {exc}") from exc
        if prover_identity["formal_source_manifest"] != verifier_identity["formal_source_manifest"]:
            raise RuntimeError("prover/verifier formal source inventories differ")
    if not srs_path.is_file() or not ceremony_descriptor.is_file():
        raise RuntimeError("adapter/settings require an existing --srs-path and --ceremony-descriptor")
    resolved_security_profile = stage_root / "resolved_security_profile.json"
    materialize_command = [
        sys.executable, str(ROOT / "scripts" / "materialize_security_profile.py"),
        "--template", str(SECURITY_PROFILE_TEMPLATE),
        "--ceremony-descriptor", str(ceremony_descriptor),
        "--srs-path", str(srs_path), "--output", str(resolved_security_profile),
    ]
    materialize_ms, materialize_stdout, materialize_stderr = run_stage(materialize_command, ROOT)
    resolved_profile_payload = json.loads(resolved_security_profile.read_text(encoding="utf-8"))
    current_source_id = formal_source_id(ROOT)
    if resolved_profile_payload.get("semantic_source_id") != current_source_id:
        raise RuntimeError("materialized security profile is not bound to the current formal source inventory")

    manifest: dict[str, Any] = {
        "manifest_kind": "ezkl_formal_gat_run_manifest_v1",
        "dataset": args.dataset,
        "device": args.device,
        "route_id": FORMAL_ROUTE,
        "config": str(config),
        "dataset_root": str(dataset_root),
        "checkpoint": str(checkpoint),
        "security_profile": str(final_run_dir / "security_profile.json"),
        "resolved_security_profile_stage": str(resolved_security_profile),
        "ceremony_descriptor": str(ceremony_descriptor),
        "srs_path": str(srs_path),
        "run_dir": str(final_run_dir),
        "stage_id": stage_root.name,
        "formal_run_requested": args.run,
        "prover_build_identity": str(prover_build_identity),
        "verifier_build_identity": str(verifier_build_identity),
        "prover_backend": "cuda_icicle" if args.device == "cuda" else "cpu",
        "verifier_backend": "cpu_independent",
        "status": "preparing",
        "verify_passed": False,
        "timings_ms": {"security_profile_materialization": materialize_ms},
        "commands": [materialize_command],
        "security_profile_materialization_stdout": materialize_stdout[-4000:],
        "security_profile_materialization_stderr": materialize_stderr[-4000:],
        "formal_semantic_source_id": current_source_id,
        "resolved_security_profile_semantic_source_id": resolved_profile_payload.get("semantic_source_id"),
    }
    if prover_identity is not None and verifier_identity is not None:
        manifest.update({
            "formal_semantic_source_id": prover_identity["formal_semantic_source_id"],
            "formal_source_manifest": prover_identity["formal_source_manifest"],
            "formal_source_manifest_count": len(prover_identity["formal_source_manifest"]),
            "vendor_tree_digest": prover_identity["vendor_tree_digest"],
            "prover_build_identity_sha256": identity_digest(prover_build_identity),
            "verifier_build_identity_sha256": identity_digest(verifier_build_identity),
            "prover_formal_semantic_source_id": prover_identity["formal_semantic_source_id"],
            "verifier_formal_semantic_source_id": verifier_identity["formal_semantic_source_id"],
            "prover_source_tree_digest": prover_identity["source_tree_digest"],
            "verifier_source_tree_digest": verifier_identity["source_tree_digest"],
            "prover_vendor_tree_digest": prover_identity["vendor_tree_digest"],
            "verifier_vendor_tree_digest": verifier_identity["vendor_tree_digest"],
            "prover_dependency_lock_digest": prover_identity["dependency_lock_digest"],
            "verifier_dependency_lock_digest": verifier_identity["dependency_lock_digest"],
            "prover_formal_source_manifest": prover_identity["formal_source_manifest"],
            "verifier_formal_source_manifest": verifier_identity["formal_source_manifest"],
        })
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
        print(json.dumps({
            "status": "canonical_artifacts_ready_not_committed",
            "stage_removed_after_exit": True,
            "dataset": args.dataset,
            "device": args.device,
        }, sort_keys=True))
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
                str(prover_binary), "graph-gat-gen-settings", "--spec-path", str(paths["spec"]),
                "--csr-layout-path", str(paths["csr"]), "--settings-path",
                str(paths["settings"]), "--max-logrows", str(args.max_logrows),
            ],
        ),
        (
            "compile",
            [
                str(prover_binary), "graph-gat-compile", "--spec-path", str(paths["spec"]),
                "--csr-layout-path", str(paths["csr"]), "--settings-path",
                str(paths["settings"]), "--compiled-circuit", str(paths["compiled"]),
                "--max-logrows", str(args.max_logrows),
            ],
        ),
        (
            "setup",
            [
                str(prover_binary), "graph-gat-setup", "--compiled-circuit", str(paths["compiled"]),
                "--vk-path", str(paths["vk"]), "--pk-path", str(paths["pk"]),
                "--srs-path", str(srs_path),
                "--device", args.device,
            ],
        ),
        (
            "gen_witness",
            [
                str(prover_binary), "graph-gat-gen-witness", "--compiled-circuit",
                str(paths["compiled"]), "--output", str(paths["witness"]),
                "--public-statement", str(paths["statement"]),
            ],
        ),
        (
            "prove",
            [
                str(prover_binary), "graph-gat-prove", "--compiled-circuit", str(paths["compiled"]),
                "--witness", str(paths["witness"]), "--public-statement",
                str(paths["statement"]), "--pk-path", str(paths["pk"]),
                "--proof-path", str(paths["proof"]), "--device", args.device,
            ],
        ),
        (
            "verify",
            [
                str(verifier_binary), "graph-gat-verify", "--compiled-circuit", str(paths["compiled"]),
                "--proof-path", str(paths["proof"]), "--public-statement",
                str(paths["statement"]), "--settings-path", str(paths["settings"]),
                "--vk-path", str(paths["vk"]), "--output", str(paths["verify"]),
                "--device", "cpu",
            ],
        ),
    ]
    logs = (LOGS_ROOT / args.device / args.dataset / stage_root.name).resolve()
    if LOGS_ROOT.resolve() not in logs.parents:
        raise RuntimeError("command log directory escaped results/logs")
    logs.mkdir(parents=True, exist_ok=True)
    stage_outputs: dict[str, tuple[str, str]] = {}
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
        stage_outputs[stage] = (stdout, stderr)
        (logs / f"{stage}.stdout.log").write_text(stdout, encoding="utf-8")
        (logs / f"{stage}.stderr.log").write_text(stderr, encoding="utf-8")

    verify = json.loads(paths["verify"].read_text(encoding="utf-8"))
    circuit_manifest = json.loads((run_dir / "circuit_manifest.json").read_text(encoding="utf-8"))
    expected_device_marker = (
        "halo2_icicle_cuda_device_0" if args.device == "cuda" else "halo2_cpu"
    )
    expected_verifier_marker = "halo2_cpu"
    prove_output = parse_native_json(stage_outputs["prove"][0], "prove")
    verify_output = parse_native_json(stage_outputs["verify"][0], "verify")
    prove_metadata = source_metadata(prove_output, "prove")
    verify_metadata = source_metadata(verify_output, "verify")
    if prove_metadata["prover_backend"] != manifest["prover_backend"]:
        raise RuntimeError("prove output backend does not match the authenticated build identity")
    if verify_metadata["verifier_backend"] != "cpu_independent":
        raise RuntimeError("verify output does not identify the independent CPU verifier")
    if args.device == "cuda":
        validate_gpu_telemetry(prove_output.get("gpu_prove_telemetry"), "formal_prove")
    if "gpu_runtime_telemetry" in verify_output or "gpu_verification_backend" in verify_output:
        raise RuntimeError("independent CPU verifier returned forbidden GPU telemetry/claim")
    required_verify_bindings = {
        "verify_passed": True,
        "native_halo2_verification": True,
        "independent_public_statement_reconstruction": True,
        "proof_instances_trusted": False,
        "compiled_circuit_id": circuit_manifest.get("compiled_circuit_id"),
        "settings_id": circuit_manifest.get("settings_id"),
        "vk_id": circuit_manifest.get("vk_id"),
        "static_tables_id": manifest["static_tables_id"],
        "backend_aux_tables_id": manifest["backend_aux_tables_id"],
        "prover_device_build": expected_device_marker,
        "prover_backend": "cuda_icicle" if args.device == "cuda" else "cpu",
        "verifier_backend": "cpu_independent",
        "semantic_source_id": manifest["formal_semantic_source_id"],
        "formal_source_manifest_id": manifest["formal_semantic_source_id"],
        "source_tree_digest": manifest["formal_semantic_source_id"],
        "vendor_tree_digest": manifest["vendor_tree_digest"],
        "verification_device_runtime_marker": expected_verifier_marker,
        "full_gat_forward_in_ezkl_proof": True,
        "strict_public_input_equivalent": True,
    }
    mismatched = {
        key: {"expected": expected, "actual": verify.get(key)}
        for key, expected in required_verify_bindings.items()
        if verify.get(key) != expected
    }
    if mismatched:
        manifest["status"] = "failed"
        manifest["failure_step"] = "verify_result_binding"
        manifest["failure"] = mismatched
        write_json(manifest_path, manifest)
        raise RuntimeError(f"independent Halo2 verifier output binding mismatch: {mismatched}")
    manifest.update(
        {
            "status": "verified",
            "verify_passed": True,
            "proof_size_bytes": paths["proof"].stat().st_size,
            "proof_sha256": sha256(paths["proof"]),
            "public_statement_sha256": sha256(paths["statement"]),
            "compiled_circuit_sha256": sha256(paths["compiled"]),
            "vk_sha256": sha256(paths["vk"]),
            "runtime_device_marker": expected_device_marker,
            "prover_backend": "cuda_icicle" if args.device == "cuda" else "cpu",
            "verifier_backend": "cpu_independent",
            "verification_runtime_marker": verify.get("verification_runtime_marker"),
            "verification_device_runtime_marker": verify.get("verification_device_runtime_marker"),
            "source_srs_sha256": circuit_manifest.get("source_srs_sha256"),
            "source_srs_k": circuit_manifest.get("source_srs_k"),
            "downsized_srs_k": circuit_manifest.get("downsized_srs_k"),
            "proving_srs_sha256": circuit_manifest.get("proving_srs_sha256"),
            "verification_srs_sha256": circuit_manifest.get("verification_srs_sha256"),
            "proving_srs_path": circuit_manifest.get("proving_srs_path"),
            "verification_srs_path": circuit_manifest.get("verification_srs_path"),
            "gpu_backend_execution": (
                "cuda_icicle_prove_only"
                if args.device == "cuda" else "not_applicable"
            ),
            "gpu_prove_telemetry": prove_output.get("gpu_prove_telemetry", "not_applicable"),
            "prove_metadata": prove_metadata,
            "verify_metadata": verify_metadata,
        }
    )
    # No result is committed until the native verifier has passed and the
    # staged cell has the exact production inventory.  This prevents partial
    # setup/proof output from being mistaken for a formal result.
    validate_cell_inventory(run_dir)
    if final_run_dir.exists() or final_run_dir.is_symlink():
        raise RuntimeError(
            "canonical result cell already exists; rerun with --clean to replace exactly this cell"
        )
    manifest = _rewrite_stage_paths(manifest, stage_root, final_run_dir)
    manifest.pop("resolved_security_profile_stage", None)
    manifest["run_dir"] = str(final_run_dir)
    manifest["manifest_path"] = str((final_run_dir / "run_manifest.json").relative_to(ROOT))
    manifest["artifact_inventory"] = sorted(CELL_ARTIFACTS)
    manifest["command_logs"] = str(logs.relative_to(ROOT))
    write_json(manifest_path, manifest)
    final_run_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(run_dir, final_run_dir)
    _safe_remove_work_tree(stage_root, RUNS_TMP_ROOT)
    committed = True
    result = {
        "dataset": args.dataset,
        "device": args.device,
        "route_id": FORMAL_ROUTE,
        "status": "verified",
        "verify_passed": True,
        "timings_ms": manifest["timings_ms"],
        "proof_size_bytes": manifest["proof_size_bytes"],
        "formal_semantic_source_id": manifest["formal_semantic_source_id"],
        "source_tree_digest": manifest["formal_semantic_source_id"],
        "vendor_tree_digest": manifest["vendor_tree_digest"],
        "prover_build_identity_sha256": manifest["prover_build_identity_sha256"],
        "verifier_build_identity_sha256": manifest["verifier_build_identity_sha256"],
        "manifest": str((final_run_dir / "run_manifest.json").relative_to(ROOT)),
    }
    # Per-cell production state lives only in the canonical run manifest.  The
    # collector owns the sole matrix summary under results/formal_results.*;
    # do not emit legacy results/<dataset>.<device>.json sidecars.
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # Fail closed; never synthesize partial results.
        print(f"formal ezkl runner failed: {error}", file=sys.stderr)
        sys.exit(2)
