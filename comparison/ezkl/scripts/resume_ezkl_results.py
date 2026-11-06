#!/usr/bin/env python3
"""Verify the existing PPI proof, then run OGBN end to end (or resume OGBN only).

``--check`` only parses and reports the planned commands and input presence. A
normal invocation owns the shared EZKL baseline lock and executes the recovery
stages in order. It deliberately does not authenticate source/build/file hashes
or reuse the old formal-session state machine; native Verify still checks the
proof against its public statement, instances, VK and SRS.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import importlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("runs/formal_ppi_ogbn_fresh_20261102_001")
SESSION_ID = "f8790043f13346f896b4ca47b1c94329"
DATASET = "ogbn_arxiv"
DEFAULT_PROVE_THREADS = 4
PROVENANCE_ENV = "EZKL_SKIP_PROVENANCE_CHECKS"
NATIVE_PREPARE_ENV = (
    "EZKL_NATIVE_PREPARE_BUDGET_BYTES",
    "EZKL_NATIVE_PREPARE_DEVICE_HEADROOM_BYTES",
    "EZKL_NATIVE_LOOKUP_CARDINALITY_BOUNDS",
)

ACTIVE_PROCESS: subprocess.Popen[bytes] | None = None
STOP_SIGNAL: int | None = None


class RecoveryError(RuntimeError):
    pass


class RecoveryInterrupted(RecoveryError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def short_time_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must be a JSON object: {path}")
    return value


def regular_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()


def nonempty_file(path: Path) -> bool:
    return regular_file(path) and path.stat().st_size > 0


def cli_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise RecoveryError(f"command path is outside the EZKL baseline: {path}") from exc


def require_inputs(output_root: Path, binary: Path, *, include_ppi: bool = True) -> list[str]:
    ppi = output_root / "ppi"
    ogbn_root = output_root / DATASET
    admission = output_root / ".sessions" / SESSION_ID / "admission" / DATASET
    capacity_root = output_root / ".sessions" / SESSION_ID / "native_capacity_query" / "capacity"
    required = [
        (binary, "GPU EZKL binary"),
        (ROOT / "resources/resource_manifest.json", "resource manifest"),
        (ROOT / "resources/native_srs/kzg26.srs", "official k26 SRS"),
        (ROOT / "resources/canonical/configs/ogbn_arxiv_secure.cfg", "OGBN canonical config"),
        (ROOT / "scripts/produce_public_statement.py", "independent producer"),
        (admission / "admission.json", "OGBN prepared admission"),
        (admission / "settings.json", "OGBN prepared settings"),
        (ROOT / "artifacts/build/icicle-cuda/native_backend_manifest.json", "GPU runtime configuration"),
        (ROOT / "artifacts/build/icicle-cuda/lib", "ICICLE runtime library directory"),
        (ROOT / "artifacts/build/gpu/build_receipt.json", "GPU build diagnostic receipt"),
        (capacity_root / "native_msm_workspace_query.json", "native CUB capacity geometry"),
        (capacity_root / "native_msm_workspace_query", "native live CUDA memory sampler"),
    ]
    if include_ppi:
        required.extend(
            (
                (ppi / "settings.json", "PPI settings"),
                (ppi / "vk.key", "PPI verification key"),
                (ppi / "srs.kzg", "PPI artifact SRS"),
                (ppi / "proof.json", "existing PPI proof"),
                (ppi / "statement.json", "PPI public statement"),
                (ppi / "public_output.json", "PPI public output"),
                (ppi / "artifact_manifest.json", "PPI artifact manifest"),
            )
        )
    missing = [f"missing {label}: {path}" for path, label in required if not path.exists()]
    for path, label in required:
        if path.exists() and path.is_symlink():
            missing.append(f"{label} must not be a symlink: {path}")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        missing.append(f"GPU EZKL binary is not executable: {binary}")
    if not ogbn_root.is_dir():
        missing.append(f"OGBN output directory is missing: {ogbn_root}")
    return missing


def formal_helpers() -> Any:
    """Load the official runner's pure geometry and capacity helpers."""
    try:
        # The read-only preflight imports this module only for pure calculations.
        sys.dont_write_bytecode = True
        return importlib.import_module("run_hybrid_formal")
    except Exception as exc:
        raise RecoveryError(f"cannot load official EZKL capacity helpers: {exc}") from exc


def native_capacity_helper(label: str, operation: Any) -> Any:
    """Translate source-helper ContractError into the recovery checkpoint error type."""
    try:
        contract_error = getattr(importlib.import_module("formal_common"), "ContractError", None)
    except (ImportError, AttributeError):
        contract_error = None
    try:
        return operation()
    except Exception as exc:
        if isinstance(contract_error, type) and isinstance(exc, contract_error):
            raise RecoveryError(f"OGBN {label} failed: {exc}") from exc
        raise


def admission_inputs(output_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    admission_dir = output_root / ".sessions" / SESSION_ID / "admission" / DATASET
    admission_path = admission_dir / "admission.json"
    settings_path = admission_dir / "settings.json"
    if not regular_file(admission_path) or not regular_file(settings_path):
        raise RecoveryError(f"prepared OGBN admission inputs are missing or unsafe: {admission_dir}")
    admission = load_json(admission_path, "OGBN prepared admission")
    if admission.get("dataset") != DATASET:
        raise RecoveryError(f"prepared admission identifies {admission.get('dataset')!r}, expected {DATASET}")
    helpers = formal_helpers()
    # These are freshly computed diagnostics for the current input files. The
    # recovery path never compares them with an earlier source/file identity.
    admission["sha256"] = helpers.sha256(admission_path)
    admission["settings_sha256"] = helpers.sha256(settings_path)
    admission["prepared_dir"] = str(admission_dir.resolve())
    return admission_dir, admission_path, admission


def capacity_workspace(output_root: Path) -> tuple[Any, dict[str, Any], dict[str, Any], Path]:
    helpers = formal_helpers()
    capacity_root = output_root / ".sessions" / SESSION_ID / "native_capacity_query" / "capacity"
    query_path = capacity_root / "native_msm_workspace_query.json"
    query_binary = capacity_root / "native_msm_workspace_query"
    query = load_json(query_path, "native CUB capacity geometry")
    runtime = load_json(ROOT / "artifacts/build/icicle-cuda/native_backend_manifest.json", "GPU runtime configuration")
    selected_device = runtime.get("selected_cuda_device")
    if not isinstance(selected_device, dict) or not isinstance(selected_device.get("uuid"), str):
        raise RecoveryError("GPU runtime configuration has no selected CUDA device UUID")
    try:
        validated = helpers._validate_cub_workspace_query(
            query,
            source_sha256=str(query.get("icicle_source_tree_sha256", "")),
            backend_manifest_sha256=str(query.get("native_backend_manifest_sha256", "")),
            selected_device=selected_device,
            session_id=str(query.get("session_id", "")),
        )
    except Exception as exc:
        raise RecoveryError(f"cached CUB geometry is malformed for the current CUDA device: {exc}") from exc
    return helpers, validated, selected_device, query_binary


def sample_cuda_memory(
    output_root: Path,
    run_dir: Path,
    session_id: str,
    workspace: dict[str, Any],
    query_binary: Path,
) -> dict[str, Any]:
    query_session = session_id.replace("-", "")
    if re.fullmatch(r"[0-9a-f]{32}", query_session) is None:
        raise RecoveryError("recovery session id cannot be represented for the CUDA memory query")
    env = gpu_environment(run_dir, DATASET, session_id)
    try:
        measured = subprocess.run(
            [str(query_binary.resolve()), "--memory-only", "--session-id", query_session],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RecoveryError(f"bounded CUDA memory-only query failed: {exc}") from exc
    if measured.returncode != 0:
        raise RecoveryError(f"CUDA memory-only query exited {measured.returncode}: {measured.stderr.strip()}")
    try:
        sample = json.loads(measured.stdout)
    except json.JSONDecodeError as exc:
        raise RecoveryError(f"CUDA memory-only query emitted malformed JSON: {exc}") from exc
    for name in ("free_memory_bytes", "total_memory_bytes"):
        value = sample.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RecoveryError(f"CUDA memory-only query has invalid {name}")
    if (
        sample.get("schema") != "ezkl.icicle.device_memory_sample.v1"
        or sample.get("device_uuid") != workspace.get("device_uuid")
        or sample.get("total_memory_bytes") != workspace.get("total_memory_bytes")
        or sample.get("integrated_device") != workspace.get("integrated_device")
        or sample["free_memory_bytes"] > sample["total_memory_bytes"]
    ):
        raise RecoveryError("live CUDA memory sample does not match the prepared OGBN device geometry")
    return sample


def rayon_thread_policy(admission: dict[str, Any], requested_threads: int) -> dict[str, Any]:
    memory = admission.get("native_memory_model")
    admitted = memory.get("fft_parallel_thread_count") if isinstance(memory, dict) else None
    if (
        isinstance(requested_threads, bool)
        or not isinstance(requested_threads, int)
        or requested_threads <= 0
        or isinstance(admitted, bool)
        or not isinstance(admitted, int)
        or admitted <= 0
    ):
        raise RecoveryError("OGBN Rayon thread override or admitted upper bound is invalid")
    if requested_threads > admitted:
        raise RecoveryError(
            f"OGBN Rayon thread override {requested_threads} exceeds the admitted Setup profile limit {admitted}"
        )
    return {
        "schema": "ezkl.recovery.rayon-thread-policy.v1",
        "admission_thread_limit": admitted,
        "effective_threads": requested_threads,
        "capacity_model_threads": requested_threads,
        "environment_variable": "RAYON_NUM_THREADS",
        "stages": ["final", "witness", "prove", "verify"],
        "capacity_model_stages": ["final", "witness", "prove"],
    }


def current_capacity_plan(
    output_root: Path,
    run_dir: Path,
    session_id: str,
    *,
    native_census: dict[str, Any] | None = None,
    srs_pending: bool,
    witness_already_written: bool = False,
    rayon_threads: int | None = None,
) -> dict[str, Any]:
    helpers, workspace, _selected_device, query_binary = capacity_workspace(output_root)
    admission_dir, admission_path, admission = admission_inputs(output_root)
    if run_dir != output_root / DATASET and regular_file(run_dir / "admission.json"):
        admission_path = run_dir / "admission.json"
        admission = load_json(admission_path, "recovery admission snapshot")
        settings_path = run_dir / "settings.json"
        if not regular_file(settings_path):
            raise RecoveryError(f"Setup-generated OGBN settings are missing: {settings_path}")
        admission["sha256"] = helpers.sha256(admission_path)
        admission["settings_sha256"] = helpers.sha256(settings_path)
        admission["prepared_dir"] = str(run_dir.resolve())
        admission_dir = run_dir
    runtime_policy: dict[str, Any] | None = None
    capacity_admission = admission
    if native_census is not None:
        memory = admission.get("native_memory_model")
        default_threads = memory.get("fft_parallel_thread_count") if isinstance(memory, dict) else None
        runtime_policy = rayon_thread_policy(admission, rayon_threads if rayon_threads is not None else default_threads)
        capacity_admission = copy.deepcopy(admission)
        capacity_admission["native_memory_model"]["fft_parallel_thread_count"] = runtime_policy["capacity_model_threads"]
    elif rayon_threads is not None:
        raise RecoveryError("a Rayon thread override requires the completed native Setup census")
    requirements = native_capacity_helper(
        "native resource capacity calculation",
        lambda: helpers._native_capacity_requirements(
            capacity_admission,
            workspace["native_type_sizes"],
            native_census=native_census,
        ),
    )
    if native_census is not None:
        if requirements.get("gat_transient_witness_bytes_upper") != load_json(
            run_dir / "native_keygen_budget.json", "native keygen budget"
        ).get("gat_transient_witness_bytes_upper"):
            raise RecoveryError("native keygen census and prepared admission disagree on GAT transient workspace")
        requirements["host_known_bytes"] = requirements["proof_host_peak_bytes"]

    settings_dir = run_dir if regular_file(run_dir / "settings.json") else admission_dir
    json_bounds = native_capacity_helper(
        "native JSON disk bound calculation",
        lambda: helpers._native_json_output_bounds(
            admission,
            DATASET,
            prepared_dir_override=settings_dir.resolve(),
        ),
    )
    pending_json = native_capacity_helper(
        "native output JSON disk budget calculation",
        lambda: helpers._pending_json_output_disk_budget({DATASET: json_bounds}, [DATASET]),
    )
    srs_size = native_capacity_helper(
        "authenticated SRS disk size calculation",
        lambda: helpers._authenticated_srs_copy_size_bytes(ROOT / "resources/native_srs/kzg26.srs"),
    )
    pending_srs = native_capacity_helper(
        "pending SRS disk budget calculation",
        lambda: helpers._pending_srs_copy_disk_budget(
            srs_size,
            {DATASET} if srs_pending else set(),
            datasets=(DATASET,),
        ),
    )
    if native_census is None:
        disk_components = native_capacity_helper(
            "Setup disk budget calculation",
            lambda: helpers._disk_budget_components(
                pending_srs=pending_srs,
                setup_key_artifact_upper_bytes=requirements["setup_key_artifact_upper_bytes"],
                setup_mapped_peak_bytes=requirements["setup_mapped_peak_bytes"],
                pending_witness_json_upper_bytes=int(pending_json["pending_witness_json_upper_bytes"]),
                pending_proof_json_upper_bytes=int(pending_json["pending_proof_json_upper_bytes"]),
                pending_json_datasets=[DATASET],
            ),
        )
        phase_name = "setup"
    else:
        proof_extra = max(0, int(requirements["proof_spill_known_bytes"]) - int(json_bounds["proof_json_upper_bytes"]))
        disk_components = native_capacity_helper(
            "Prove disk budget calculation",
            lambda: helpers._disk_budget_components(
                pending_srs=pending_srs,
                proof_spill_known_bytes=int(requirements["proof_spill_known_bytes"]),
                proof_spill_extra_bytes=proof_extra,
                pending_witness_json_upper_bytes=int(pending_json["pending_witness_json_upper_bytes"]),
                pending_proof_json_upper_bytes=int(pending_json["pending_proof_json_upper_bytes"]),
                pending_json_datasets=[DATASET],
            ),
        )
        phase_name = "witness_prove_verify"
        if witness_already_written:
            witness_path = run_dir / "witness.json"
            if not nonempty_file(witness_path):
                raise RecoveryError("cannot refresh Prove capacity without the completed OGBN witness")
            consumed = witness_path.stat().st_size
            disk_components = dict(disk_components)
            disk_components["preexisting_srs_and_witness_bytes"] = consumed
            disk_components["total_required_bytes"] = max(0, int(disk_components["total_required_bytes"]) - consumed)
    sample = sample_cuda_memory(output_root, run_dir, session_id, workspace, query_binary)
    try:
        check = helpers._check_dataset_capacity(
            DATASET,
            requirements,
            output_root=run_dir if run_dir.exists() else run_dir.parent,
            remaining_known_disk_bytes=int(disk_components["total_required_bytes"]),
            phase=phase_name,
            device_sample=sample,
            workspace_query=workspace,
            disk_budget_components=disk_components,
        )
    except Exception as exc:
        raise RecoveryError(f"OGBN {phase_name} live capacity gate failed: {exc}") from exc
    native_prepare_exports: dict[str, str] | None = None
    if runtime_policy is not None:
        prepare_budget = native_capacity_helper(
            "native prepare task budget calculation",
            lambda: helpers._native_prepare_budget(
                requirements,
                initial_available_bytes=int(check["host_available_bytes"]),
                workspace_query=workspace,
            ),
        )
        expected_lookup_groups = int(admission["constraint_system"]["num_unique_lookup_table_expression_groups"])
        taskcap, device_headroom, cardinalities = native_capacity_helper(
            "native prepare export validation",
            lambda: helpers._validated_native_prepare_exports(
                prepare_budget,
                requirements.get("lookup_map_bounds"),
                domain_size=int(admission["domain_size"]),
                expected_lookup_groups=expected_lookup_groups,
                expected_h_resident_device_upper_bytes=int(requirements["icicle_h_resident_device_upper_bytes"]),
            ),
        )
        native_prepare_exports = {
            "EZKL_NATIVE_PREPARE_BUDGET_BYTES": taskcap,
            "EZKL_NATIVE_PREPARE_DEVICE_HEADROOM_BYTES": device_headroom,
            "EZKL_NATIVE_LOOKUP_CARDINALITY_BOUNDS": cardinalities,
        }
        check["rayon_thread_policy"] = runtime_policy
        check["proof_host_peak_bytes"] = int(requirements["proof_host_peak_bytes"])
    return {
        "helpers": helpers,
        "admission": admission,
        "admission_dir": admission_dir,
        "admission_path": admission_path,
        "workspace": workspace,
        "requirements": requirements,
        "json_bounds": json_bounds,
        "pending_srs": pending_srs,
        "disk_components": disk_components,
        "sample": sample,
        "check": check,
        "srs_size_bytes": srs_size,
        "rayon_thread_policy": runtime_policy,
        "native_prepare_exports": native_prepare_exports,
    }


def apply_native_runtime_environment(
    env: dict[str, str],
    runtime_policy: dict[str, Any],
    prepare_exports: dict[str, str] | None,
) -> dict[str, str]:
    threads = runtime_policy.get("effective_threads")
    admission_limit = runtime_policy.get("admission_thread_limit")
    if (
        runtime_policy.get("schema") != "ezkl.recovery.rayon-thread-policy.v1"
        or runtime_policy.get("capacity_model_threads") != threads
        or runtime_policy.get("environment_variable") != "RAYON_NUM_THREADS"
        or isinstance(threads, bool)
        or not isinstance(threads, int)
        or threads <= 0
        or isinstance(admission_limit, bool)
        or not isinstance(admission_limit, int)
        or admission_limit <= 0
        or threads > admission_limit
    ):
        raise RecoveryError("OGBN native subprocess thread policy is malformed or exceeds its Setup profile")
    if not isinstance(prepare_exports, dict) or set(prepare_exports) != set(NATIVE_PREPARE_ENV):
        raise RecoveryError("OGBN native stage is missing validated native prepare budget exports")
    if any(not isinstance(prepare_exports[key], str) or not prepare_exports[key] for key in NATIVE_PREPARE_ENV):
        raise RecoveryError("OGBN native prepare budget exports are malformed")
    env["RAYON_NUM_THREADS"] = str(threads)
    env.update(prepare_exports)
    return env


def gpu_environment(
    run_dir: Path,
    dataset: str,
    session_id: str | None = None,
    *,
    runtime_policy: dict[str, Any] | None = None,
    prepare_exports: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build only the native GPU loader/device environment, without hash pins."""
    manifest_path = ROOT / "artifacts/build/icicle-cuda/native_backend_manifest.json"
    manifest = load_json(manifest_path, "GPU runtime configuration")
    selected = manifest.get("selected_cuda_device")
    architecture = manifest.get("architecture")
    if not isinstance(selected, dict) or not isinstance(selected.get("uuid"), str):
        raise RecoveryError(f"GPU runtime configuration has no selected device UUID: {manifest_path}")
    if not isinstance(architecture, str) or not architecture.startswith("sm_"):
        raise RecoveryError(f"GPU runtime configuration has no CUDA architecture: {manifest_path}")
    install_dir = ROOT / "artifacts/build/icicle-cuda"
    lib_dir = install_dir / "lib"
    if not install_dir.is_dir() or not lib_dir.is_dir():
        raise RecoveryError(f"ICICLE GPU runtime directories are missing: {install_dir}")

    env = os.environ.copy()
    for key in (
        "EZKL_FORMAL_SESSION_ID",
        "ICICLE_NATIVE_MANIFEST_SHA256",
        "EZKL_GAT_SOURCE_TREE_SHA256",
        "EZKL_GAT_BUILD_SHA256",
        "EZKL_GAT_BINARY_SHA256",
    ):
        env.pop(key, None)
    env.update(
        {
            PROVENANCE_ENV: "1",
            "EZKL_WORKING_DIR": str(ROOT),
            "EZKL_GAT_BACKEND": "gpu",
            "EZKL_GAT_RUN_DIR": str(run_dir.resolve()),
            "EZKL_GAT_DATASET": dataset,
            "EZKL_GAT_REQUIRE_GPU_EVIDENCE": "1",
            "ENABLE_ICICLE_GPU": "1",
            "ICICLE_BACKEND_INSTALL_DIR": str(install_dir),
            "ICICLE_NATIVE_LIB_DIR": str(lib_dir),
            "CUDA_ARCH": architecture.removeprefix("sm_"),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": selected["uuid"],
            "ICICLE_DEVICE_ID": "0",
            "EZKL_FORMAL_SELECTED_CUDA_DEVICE": json.dumps(selected, sort_keys=True, separators=(",", ":")),
        }
    )
    if session_id is not None:
        env["EZKL_FORMAL_SESSION_ID"] = session_id
    toolkit = manifest.get("cuda_toolkit_root")
    runtime_paths = [lib_dir]
    if isinstance(toolkit, str) and toolkit:
        runtime_paths.extend((Path(toolkit) / "lib64", Path(toolkit) / "targets/sbsa-linux/lib"))
    existing = [str(path) for path in runtime_paths if path.is_dir()]
    if env.get("LD_LIBRARY_PATH"):
        existing.append(env["LD_LIBRARY_PATH"])
    if existing:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(existing)
    if runtime_policy is not None:
        apply_native_runtime_environment(env, runtime_policy, prepare_exports)
    elif prepare_exports is not None:
        raise RecoveryError("OGBN native prepare budget exports require an explicit runtime thread policy")
    return env


def cpu_environment(
    run_dir: Path,
    dataset: str,
    session_id: str | None = None,
    *,
    runtime_policy: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Use the native binary's independent CPU verifier with provenance unbound."""
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("ICICLE_") or key.startswith("CUDA_"):
            env.pop(key, None)
    for key in (
        "EZKL_FORMAL_SESSION_ID",
        "EZKL_FORMAL_SELECTED_CUDA_DEVICE",
        "EZKL_GAT_REQUIRE_GPU_EVIDENCE",
        "ENABLE_ICICLE_GPU",
        "EZKL_GAT_SOURCE_TREE_SHA256",
        "EZKL_GAT_BUILD_SHA256",
        "EZKL_GAT_BINARY_SHA256",
        "ICICLE_NATIVE_MANIFEST_SHA256",
    ):
        env.pop(key, None)
    env.update(
        {
            PROVENANCE_ENV: "1",
            "EZKL_WORKING_DIR": str(ROOT),
            "EZKL_GAT_BACKEND": "cpu",
            "EZKL_GAT_RUN_DIR": str(run_dir.resolve()),
            "EZKL_GAT_DATASET": dataset,
        }
    )
    if session_id is not None:
        env["EZKL_FORMAL_SESSION_ID"] = session_id
    for name in NATIVE_PREPARE_ENV:
        env.pop(name, None)
    if runtime_policy is not None:
        threads = runtime_policy.get("effective_threads")
        admission_limit = runtime_policy.get("admission_thread_limit")
        if (
            runtime_policy.get("schema") != "ezkl.recovery.rayon-thread-policy.v1"
            or runtime_policy.get("capacity_model_threads") != threads
            or isinstance(threads, bool)
            or not isinstance(threads, int)
            or threads <= 0
            or isinstance(admission_limit, bool)
            or not isinstance(admission_limit, int)
            or admission_limit <= 0
            or threads > admission_limit
        ):
            raise RecoveryError("OGBN CPU verifier thread policy is malformed or exceeds its Setup profile")
        env["RAYON_NUM_THREADS"] = str(threads)
    return env


def producer_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("EZKL_FORMAL_SESSION_ID", None)
    env[PROVENANCE_ENV] = "1"
    env["EZKL_WORKING_DIR"] = str(ROOT)
    return env


def ppi_verify_command(binary: Path, output_root: Path) -> list[str]:
    ppi = output_root / "ppi"
    return [
        str(binary), "verify",
        "--settings-path", cli_path(ppi / "settings.json"),
        "--proof-path", cli_path(ppi / "proof.json"),
        "--vk-path", cli_path(ppi / "vk.key"),
        "--srs-path", cli_path(ppi / "srs.kzg"),
        "--statement", cli_path(ppi / "statement.json"),
        "--public-output", cli_path(ppi / "public_output.json"),
        "--artifact-manifest", cli_path(ppi / "artifact_manifest.json"),
        "--resource-manifest", "resources/resource_manifest.json",
    ]


def ogbn_commands(binary: Path, output_root: Path, output: Path) -> dict[str, list[str]]:
    admission = output_root / ".sessions" / SESSION_ID / "admission" / DATASET
    common = ["--resource-manifest", "resources/resource_manifest.json"]
    return {
        "setup": [
            str(binary), "gat-prepare", "--dataset", DATASET, "--phase", "setup",
            *common, "--output-dir", cli_path(output), "--admission-dir", cli_path(admission),
            "--srs-path", "resources/native_srs/kzg26.srs", "--gpu",
        ],
        "producer": [
            sys.executable, "-u", "scripts/produce_public_statement.py",
            "--dataset", DATASET, *common,
            "--setup-context", cli_path(output / "producer_context.json"),
            "--output-dir", cli_path(output),
            "--config", "resources/canonical/configs/ogbn_arxiv_secure.cfg",
            "--statement", cli_path(output / "statement.json"),
            "--public-output", cli_path(output / "public_output.json"),
            "--producer-manifest", cli_path(output / "producer_manifest.json"),
        ],
        "final": [
            str(binary), "gat-prepare", "--dataset", DATASET, "--phase", "final",
            *common, "--output-dir", cli_path(output),
            "--srs-path", "resources/native_srs/kzg26.srs",
            "--public-statement", cli_path(output / "statement.json"),
            "--public-output", cli_path(output / "public_output.json"),
            "--producer-manifest", cli_path(output / "producer_manifest.json"), "--gpu",
        ],
        "witness": [
            str(binary), "gat-witness", "--dataset", DATASET, *common,
            "--compiled-circuit", cli_path(output / "compiled.circuit"),
            "--statement", cli_path(output / "statement.json"),
            "--output", cli_path(output / "witness.json"), "--gpu",
        ],
        "prove": [
            str(binary), "prove",
            "--witness", cli_path(output / "witness.json"),
            "--compiled-circuit", cli_path(output / "compiled.circuit"),
            "--pk-path", cli_path(output / "pk.key"),
            "--proof-path", cli_path(output / "proof.json"),
            "--srs-path", "resources/native_srs/kzg26.srs", "--check-mode", "unsafe",
        ],
        "verify": [
            str(binary), "verify",
            "--settings-path", cli_path(output / "settings.json"),
            "--proof-path", cli_path(output / "proof.json"),
            "--vk-path", cli_path(output / "vk.key"),
            "--srs-path", cli_path(output / "srs.kzg"),
            "--statement", cli_path(output / "statement.json"),
            "--public-output", cli_path(output / "public_output.json"),
            "--artifact-manifest", cli_path(output / "artifact_manifest.json"),
            "--resource-manifest", "resources/resource_manifest.json",
        ],
    }


def load_budget_failed_ogbn_state(output_root: Path) -> tuple[Path, dict[str, Any], Path]:
    ogbn_root = output_root / DATASET
    session_path = ogbn_root / "recovery_session.json"
    if not regular_file(session_path):
        raise RecoveryError(f"OGBN-only resume requires the existing recovery session: {session_path}")
    state = load_json(session_path, "OGBN recovery session")
    if (
        state.get("schema") != "ezkl.recovery_session.v1"
        or state.get("dataset") != DATASET
        or state.get("status") != "BUDGET_FAILED"
        or state.get("current_stage") != "prove_budget"
        or state.get("stages", {}).get("prove_budget", {}).get("status") != "failed"
    ):
        raise RecoveryError(
            "OGBN-only resume is limited to the current BUDGET_FAILED/prove_budget session; "
            f"inspect {session_path} before choosing another recovery path"
        )
    output_text = state.get("output_dir")
    if not isinstance(output_text, str) or not output_text:
        raise RecoveryError(f"OGBN recovery state has no output directory: {session_path}")
    output = Path(output_text)
    if not output.is_absolute():
        output = ROOT / output
    if output.is_symlink() or not output.is_dir():
        raise RecoveryError(f"existing OGBN Setup output directory is missing or unsafe: {output}")
    try:
        output.resolve().relative_to(ogbn_root.resolve())
    except ValueError as exc:
        raise RecoveryError(f"OGBN Setup output escapes its dataset directory: {output}") from exc
    if not stage_is_complete(state, output, "setup") or not stage_is_complete(state, output, "setup_receipts"):
        raise RecoveryError(f"OGBN Setup or its receipt is incomplete; preserving {output}")
    if state.get("stages", {}).get("prove", {}).get("status") not in (None, "pending"):
        raise RecoveryError("OGBN-only budget retry refuses a session where Prove has already started")
    return session_path, state, output


def print_ogbn_only_check(output_root: Path, binary: Path, prove_threads: int) -> int:
    missing = require_inputs(output_root, binary, include_ppi=False)
    state_path: Path | None = None
    state: dict[str, Any] | None = None
    output: Path | None = None
    plan: dict[str, Any] | None = None
    if not missing:
        try:
            state_path, state, output = load_budget_failed_ogbn_state(output_root)
            validate_native_setup_outputs(output, str(state["session_id"]))
            census = load_json(output / "native_keygen_receipt.json", "OGBN native Setup census")["native_census"]
            plan = current_capacity_plan(
                output_root,
                output,
                str(state["session_id"]),
                native_census=census,
                srs_pending=not nonempty_file(output / "srs.kzg"),
                witness_already_written=stage_is_complete(state, output, "witness"),
                rayon_threads=prove_threads,
            )
        except (RecoveryError, KeyError, TypeError) as exc:
            missing.append(str(exc))
    commands: dict[str, list[str]] = {}
    if output is not None:
        commands = ogbn_commands(binary, output_root, output)
        commands.pop("setup", None)
    report = {
        "mode": "read_only_ogbn_resume_capacity_check",
        "input_status": "READY" if not missing else "MISSING_OR_CAPACITY_FAILED",
        "missing_or_failure": missing,
        "scope": "ogbn_arxiv_only",
        "ppi_verify": "skipped",
        "setup": "reused_existing_authenticated_setup",
        "formal_stage_started": False,
        "proving_key_or_srs_loaded": False,
        "files_written": False,
        "live_cuda_sampler": "bounded --memory-only query" if plan else "not reached",
        "capacity_sampled_at_utc": plan["check"].get("sampled_at_utc") if plan else None,
        "session_path": str(state_path) if state_path else None,
        "session_id": state.get("session_id") if state else None,
        "setup_output_dir": str(output) if output else None,
        "commands_after_preflight": commands,
        "rayon_thread_policy": plan["rayon_thread_policy"] if plan else None,
        "native_prepare_environment": plan["native_prepare_exports"] if plan else None,
        "capacity_check": plan["check"] if plan else None,
        "capacity_peak_bytes": (
            {
                "proof_host_peak_bytes": int(plan["requirements"]["proof_host_peak_bytes"]),
                "host_required_known_bytes": int(plan["check"]["host_required_known_bytes"]),
                "host_available_bytes": int(plan["check"]["host_available_bytes"]),
                "cuda_required_workspace_bytes": int(plan["check"]["cuda_required_workspace_bytes"]),
                "cuda_free_bytes": int(plan["check"]["cuda_free_bytes"]),
                "disk_required_known_bytes": int(plan["check"]["disk_required_known_bytes"]),
                "disk_available_bytes": int(plan["check"]["disk_available_bytes"]),
            }
            if plan else None
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not missing else 2


def print_check(output_root: Path, binary: Path) -> int:
    missing = require_inputs(output_root, binary)
    ppi = output_root / "ppi"
    placeholder = Path("<recovery-attempt>")
    try:
        cpu_environment(ppi / placeholder, "ppi", "check-only")
    except RecoveryError as exc:
        missing.append(str(exc))
    setup_capacity: dict[str, Any] | None = None
    prove_capacity: dict[str, Any] | None = None
    if not missing:
        try:
            setup_capacity = current_capacity_plan(
                output_root,
                output_root / DATASET / placeholder,
                "00000000000000000000000000000001",
                native_census=None,
                srs_pending=True,
            )
            prove_capacity = preliminary_prove_capacity(
                output_root,
                output_root / DATASET / placeholder,
                "00000000000000000000000000000001",
                live_sample=setup_capacity["sample"],
            )
        except RecoveryError as exc:
            missing.append(str(exc))
    commands = {
        "ppi_verify": ppi_verify_command(binary, output_root),
        "ogbn_arxiv": ogbn_commands(binary, output_root, output_root / DATASET / placeholder),
    }
    plan = {
        "mode": "read_only_check",
        "provenance_checks_disabled": True,
        "input_status": "READY" if not missing else "MISSING_INPUTS",
        "missing": missing,
        "commands": commands,
        "ogbn_setup_capacity": setup_capacity["check"] if setup_capacity else None,
        "ogbn_prove_capacity": prove_capacity,
        "ppi_existing_proof_bytes": (ppi / "proof.json").stat().st_size if regular_file(ppi / "proof.json") else None,
        "ogbn_old_spill_dirs_will_be_atomically_archived_on_execution": [
            path.name for path in sorted((output_root / DATASET).glob("ezkl-halo2-spill-*")) if path.is_dir()
        ],
        "global_lock": str(ROOT / "runs/.ezkl-formal.lock"),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0 if not missing else 2


def acquire_global_lock() -> int:
    lock_dir = ROOT / "runs"
    if lock_dir.is_symlink() or not lock_dir.is_dir():
        raise RecoveryError(f"official EZKL lock directory is missing or unsafe: {lock_dir}")
    lock_path = lock_dir / ".ezkl-formal.lock"
    if lock_path.is_symlink():
        raise RecoveryError(f"official EZKL global lock must not be a symlink: {lock_path}")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RecoveryError("another EZKL baseline task currently holds runs/.ezkl-formal.lock") from exc
    return fd


def install_signal_handlers() -> None:
    def handle(signum: int, _frame: Any) -> None:
        global STOP_SIGNAL
        STOP_SIGNAL = signum
        process = ACTIVE_PROCESS
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def wait_for_group_exit(pgid: int, grace_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return


def run_stage(
    *,
    name: str,
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    session_path: Path,
    state: dict[str, Any],
    runtime_policy: dict[str, Any] | None = None,
    prepare_exports: dict[str, str] | None = None,
) -> float:
    global ACTIVE_PROCESS
    if STOP_SIGNAL is not None:
        raise RecoveryInterrupted(f"received signal {STOP_SIGNAL} before {name}")
    if runtime_policy is not None:
        threads = runtime_policy.get("effective_threads")
        if isinstance(threads, bool) or not isinstance(threads, int) or env.get("RAYON_NUM_THREADS") != str(threads):
            raise RecoveryError(f"{name} subprocess environment does not match its OGBN Rayon runtime policy")
        if name in {"final", "witness", "prove"}:
            if not isinstance(prepare_exports, dict) or any(
                env.get(key) != prepare_exports.get(key) for key in NATIVE_PREPARE_ENV
            ):
                raise RecoveryError(f"{name} subprocess environment does not match its validated native prepare exports")
        elif prepare_exports is not None:
            raise RecoveryError(f"unexpected native prepare exports for OGBN {name}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = state["stages"].setdefault(name, {})
    record.update({"status": "running", "started_at": utc_now(), "argv": command, "log": str(log_path)})
    if runtime_policy is not None:
        record["runtime_policy"] = runtime_policy
        record["rayon_num_threads"] = int(env["RAYON_NUM_THREADS"])
    if prepare_exports is not None:
        record["native_prepare_environment"] = {key: env[key] for key in NATIVE_PREPARE_ENV}
    state.update({"status": "RUNNING", "current_stage": name, "active_process": None, "updated_at": utc_now()})
    atomic_json(session_path, state)
    print(f"[recovery] START {name}; log={log_path}; argv={' '.join(command)}", flush=True)
    started = time.monotonic()
    with log_path.open("ab") as stream:
        stream.write((f"\n--- recovery stage={name} started={utc_now()} ---\n").encode())
        stream.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            record.update({"status": "failed", "finished_at": utc_now(), "error": f"spawn failed: {exc}"})
            state.update({"status": "FAILED", "active_process": None, "updated_at": utc_now()})
            atomic_json(session_path, state)
            raise RecoveryError(f"cannot start {name}: {exc}") from exc
        ACTIVE_PROCESS = process
        if STOP_SIGNAL is not None:
            try:
                os.killpg(process.pid, STOP_SIGNAL)
            except ProcessLookupError:
                pass
        state["active_process"] = {"pid": process.pid, "stage": name, "argv": command, "process_group": process.pid}
        state["updated_at"] = utc_now()
        atomic_json(session_path, state)
        exit_code = process.wait()
        ACTIVE_PROCESS = None
        elapsed_ms = (time.monotonic() - started) * 1000.0
        stream.write((f"\n--- recovery stage={name} exit_code={exit_code} wall_ms={elapsed_ms:.3f} finished={utc_now()} ---\n").encode())
        stream.flush()
    if STOP_SIGNAL is not None:
        wait_for_group_exit(process.pid)
        record.update({"status": "interrupted", "exit_code": exit_code, "wall_ms": elapsed_ms, "finished_at": utc_now()})
        state.update({"status": "INTERRUPTED", "active_process": None, "updated_at": utc_now()})
        atomic_json(session_path, state)
        raise RecoveryInterrupted(f"{name} interrupted by signal {STOP_SIGNAL}")
    if exit_code != 0:
        record.update({"status": "failed", "exit_code": exit_code, "wall_ms": elapsed_ms, "finished_at": utc_now()})
        state.update({"status": "FAILED", "active_process": None, "updated_at": utc_now()})
        atomic_json(session_path, state)
        raise RecoveryError(f"{name} exited with code {exit_code}; see {log_path}")
    record.update({"status": "exited_zero", "exit_code": 0, "wall_ms": elapsed_ms, "finished_at": utc_now()})
    state.update({"active_process": None, "updated_at": utc_now()})
    atomic_json(session_path, state)
    return elapsed_ms


def validate_verify_outputs(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = load_json(run_dir / "native_verify_metrics.json", "native Verify metrics")
    core = load_json(run_dir / "core_run_manifest.json", "native core run manifest")
    duration = metrics.get("verify_time_ms")
    if (
        metrics.get("schema_version") != 1
        or metrics.get("operation") != "native_halo2_independent_verifier"
        or metrics.get("included_in") != "native_verify_proof"
        or metrics.get("provenance_checks_disabled") is not True
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) <= 0
    ):
        raise RecoveryError("native Verify metrics are incomplete or do not record the unbound mode")
    if core.get("native_verify_result") is not True or core.get("native_public_instance_binding") is not True:
        raise RecoveryError("native Verify did not attest successful proof and public-instance verification")
    return metrics, core


def validate_native_setup_outputs(output: Path, session_id: str) -> dict[str, Any]:
    required = (
        "settings.json", "compiled.circuit", "pk.key", "vk.key", "native_keygen_receipt.json",
        "native_keygen_budget.json", "gat_static_tables.json", "gat_integer_ledger.json",
        "producer_context.json", "circuit_ref.json",
    )
    missing = [name for name in required if not nonempty_file(output / name)]
    if missing:
        raise RecoveryError(f"OGBN setup returned success but is missing required outputs: {', '.join(missing)}")
    receipt = load_json(output / "native_keygen_receipt.json", "native keygen receipt")
    budget = load_json(output / "native_keygen_budget.json", "native keygen budget")
    relation = receipt.get("relation_identity")
    caps = receipt.get("stage_budget_caps")
    digest_fields = (
        "settings_sha256", "compiled_circuit_sha256", "vk_sha256", "pk_sha256", "budget_receipt_sha256",
    )
    if (
        receipt.get("schema") != "ezkl.native-keygen-receipt.v4"
        or receipt.get("dataset") != DATASET
        or receipt.get("session_id") != session_id
        or not isinstance(receipt.get("native_census"), dict)
        or not isinstance(relation, dict)
        or relation.get("dataset") != DATASET
        or receipt.get("native_pk_format_version") != 4
        or receipt.get("native_pk_sigma_wire_encoding") != "packed_source_target_indices_lsb0_v1"
        or not isinstance(caps, list)
        or len(caps) != 2
        or [item.get("stage") if isinstance(item, dict) else None for item in caps] != ["vk", "pk"]
        or any(not isinstance(receipt.get(name), str) or re.fullmatch(r"[0-9a-f]{64}", receipt[name]) is None for name in digest_fields)
    ):
        raise RecoveryError("native keygen receipt does not identify a complete OGBN setup census")
    if (
        budget.get("schema") != "ezkl.native-keygen-budget.v3"
        or budget.get("dataset") != DATASET
        or budget.get("session_id") != session_id
    ):
        raise RecoveryError("native keygen budget does not identify this OGBN setup")
    return receipt


def prepare_setup_receipts(output: Path, admission_dir: Path, session_id: str) -> dict[str, Any]:
    native_receipt = validate_native_setup_outputs(output, session_id)
    source = admission_dir / "admission.json"
    destination = output / "admission.json"
    if not regular_file(source):
        raise RecoveryError(f"prepared admission is missing or unsafe: {source}")
    if not destination.exists() and not destination.is_symlink():
        try:
            with source.open("rb") as input_stream, destination.open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream)
                output_stream.flush()
                os.fsync(output_stream.fileno())
        except OSError as exc:
            raise RecoveryError(f"cannot preserve the prepared OGBN admission snapshot: {exc}") from exc
    elif not regular_file(destination):
        raise RecoveryError(f"recovery admission snapshot is unsafe: {destination}")
    snapshot = load_json(destination, "recovery OGBN admission snapshot")
    if snapshot.get("dataset") != DATASET:
        raise RecoveryError("copied OGBN admission snapshot has an unexpected dataset")
    receipt_path = output / "setup_receipt.json"
    artifact_sizes: dict[str, int] = {}
    for name in (
        "admission.json", "settings.json", "compiled.circuit", "pk.key", "vk.key",
        "native_keygen_budget.json", "native_keygen_receipt.json", "gat_static_tables.json",
        "gat_integer_ledger.json", "producer_context.json", "circuit_ref.json",
    ):
        path = output / name
        if not nonempty_file(path):
            raise RecoveryError(f"successful OGBN Setup is missing {name}")
        artifact_sizes[name] = path.stat().st_size
    setup_receipt = {
        "schema": "ezkl.recovery-setup-receipt.v1",
        "session_id": session_id,
        "dataset": DATASET,
        "setup_command_exit_code": 0,
        "setup_native_keygen_receipt_schema": native_receipt["schema"],
        "setup_native_census_present": True,
        "setup_artifact_sizes_bytes": artifact_sizes,
        "provenance_checks_disabled": True,
        "recorded_at_utc": utc_now(),
    }
    if receipt_path.exists() or receipt_path.is_symlink():
        existing = load_json(receipt_path, "recovery Setup receipt")
        if (
            existing.get("schema") != setup_receipt["schema"]
            or existing.get("session_id") != session_id
            or existing.get("dataset") != DATASET
            or existing.get("setup_command_exit_code") != 0
            or existing.get("setup_native_census_present") is not True
        ):
            raise RecoveryError(f"existing recovery Setup receipt is incompatible: {receipt_path}")
    else:
        atomic_json(receipt_path, setup_receipt)
    return native_receipt


def diagnostic_resume_identity(binary: Path) -> dict[str, str]:
    helpers = formal_helpers()
    receipt_path = ROOT / "artifacts/build/gpu/build_receipt.json"
    receipt = load_json(receipt_path, "GPU build diagnostic receipt")
    environment_sha = receipt.get("build_environment_sha256")
    if not isinstance(environment_sha, str) or re.fullmatch(r"[0-9a-f]{64}", environment_sha) is None:
        raise RecoveryError("GPU build diagnostic receipt has no real build-environment digest")
    # These current-build digests are carried as diagnostic metadata only; no
    # source, build, binary, or artifact digest is used as a recovery gate.
    return {
        "binary_sha256": helpers.sha256(binary),
        "build_receipt_sha256": helpers.sha256(receipt_path),
        "build_environment_sha256": environment_sha,
    }


def prepare_native_keygen_budget(
    output: Path,
    binary: Path,
    recovery_id: str,
    plan: dict[str, Any],
) -> None:
    path = output / "native_keygen_budget.json"
    if path.exists() or path.is_symlink():
        existing = load_json(path, "existing native keygen budget")
        if (
            not regular_file(path)
            or existing.get("schema") != "ezkl.native-keygen-budget.v3"
            or existing.get("dataset") != DATASET
            or existing.get("session_id") != recovery_id
        ):
            raise RecoveryError(f"existing OGBN native keygen budget is incompatible: {path}")
        return
    plan["helpers"]._write_native_keygen_budget_receipt(
        output,
        dataset=DATASET,
        session_id=recovery_id,
        admission=plan["admission"],
        resume_identity=diagnostic_resume_identity(binary),
        capacity=plan["requirements"],
        setup_check=plan["check"],
        workspace_query=plan["workspace"],
    )
    budget = load_json(path, "new native keygen budget")
    if (
        budget.get("schema") != "ezkl.native-keygen-budget.v3"
        or budget.get("dataset") != DATASET
        or budget.get("session_id") != recovery_id
    ):
        raise RecoveryError("official native keygen budget helper emitted an incompatible OGBN budget")


def prepare_native_prove_budget(
    output: Path,
    output_root: Path,
    binary: Path,
    recovery_id: str,
    rayon_threads: int,
    capacity_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not nonempty_file(output / "admission.json") or not nonempty_file(output / "setup_receipt.json"):
        raise RecoveryError("cannot prepare native Prove budget before OGBN admission and successful Setup receipt")
    native_receipt = validate_native_setup_outputs(output, recovery_id)
    plan = capacity_plan or current_capacity_plan(
        output_root,
        output,
        recovery_id,
        native_census=native_receipt["native_census"],
        srs_pending=not nonempty_file(output / "srs.kzg"),
        rayon_threads=rayon_threads,
    )
    if plan.get("rayon_thread_policy", {}).get("effective_threads") != rayon_threads:
        raise RecoveryError("OGBN Prove budget capacity plan does not match the selected Rayon thread count")
    helpers = plan["helpers"]
    budget_path = output / "native_prove_budget.json"
    if budget_path.exists() or budget_path.is_symlink():
        budget = load_json(budget_path, "native prove budget")
        if (
            not regular_file(budget_path)
            or budget.get("schema") != "ezkl.native-prove-budget.v8"
            or budget.get("dataset") != DATASET
            or budget.get("session_id") != recovery_id
        ):
            raise RecoveryError("existing recovery native Prove budget has an unsupported schema/dataset")
        return plan

    native_budget = load_json(output / "native_keygen_budget.json", "native keygen budget")
    setup_receipt_path = output / "setup_receipt.json"
    setup_receipt = load_json(setup_receipt_path, "recovery Setup receipt")
    if (
        setup_receipt.get("schema") != "ezkl.recovery-setup-receipt.v1"
        or setup_receipt.get("session_id") != recovery_id
        or setup_receipt.get("dataset") != DATASET
        or setup_receipt.get("setup_command_exit_code") != 0
        or setup_receipt.get("setup_native_census_present") is not True
    ):
        raise RecoveryError("recovery Setup receipt does not record a successful native keygen census")
    trusted_setup_files = {
        "settings_sha256": native_receipt["settings_sha256"],
        "compiled_circuit_sha256": native_receipt["compiled_circuit_sha256"],
        "vk_sha256": native_receipt["vk_sha256"],
        "pk_sha256": native_receipt["pk_sha256"],
        "static_tables_sha256": helpers.sha256(output / "gat_static_tables.json"),
        "integer_ledger_sha256": helpers.sha256(output / "gat_integer_ledger.json"),
    }
    helpers._write_native_prove_budget_receipt(
        output,
        dataset=DATASET,
        session_id=recovery_id,
        admission=plan["admission"],
        resume_identity=diagnostic_resume_identity(binary),
        setup_receipt_sha256=helpers.sha256(setup_receipt_path),
        native_keygen_budget_sha256=helpers.sha256(output / "native_keygen_budget.json"),
        native_keygen_receipt_sha256=helpers.sha256(output / "native_keygen_receipt.json"),
        capacity=plan["requirements"],
        prove_check=plan["check"],
        workspace_query=plan["workspace"],
        trusted_setup_files=trusted_setup_files,
    )
    plan["native_keygen_budget"] = native_budget
    plan["native_prove_budget_path"] = str(budget_path)
    return plan


def preliminary_prove_capacity(
    output_root: Path,
    placeholder: Path,
    recovery_id: str,
    *,
    live_sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    helpers, workspace, _selected_device, query_binary = capacity_workspace(output_root)
    admission_dir, _admission_path, admission = admission_inputs(output_root)
    requirements = helpers._native_capacity_requirements(admission, workspace["native_type_sizes"])
    bounds = helpers._native_json_output_bounds(admission, DATASET, prepared_dir_override=admission_dir.resolve())
    pending_json = helpers._pending_json_output_disk_budget({DATASET: bounds}, [DATASET])
    srs_size = helpers._authenticated_srs_copy_size_bytes(ROOT / "resources/native_srs/kzg26.srs")
    pending_srs = helpers._pending_srs_copy_disk_budget(srs_size, {DATASET}, datasets=(DATASET,))
    proof_extra = max(0, int(requirements["proof_spill_known_bytes"]) - int(bounds["proof_json_upper_bytes"]))
    proof_components = helpers._disk_budget_components(
        pending_srs=pending_srs,
        proof_spill_known_bytes=int(requirements["proof_spill_known_bytes"]),
        proof_spill_extra_bytes=proof_extra,
        pending_witness_json_upper_bytes=int(pending_json["pending_witness_json_upper_bytes"]),
        pending_proof_json_upper_bytes=int(pending_json["pending_proof_json_upper_bytes"]),
        pending_json_datasets=[DATASET],
    )
    sample = live_sample or sample_cuda_memory(output_root, placeholder, recovery_id, workspace, query_binary)
    host_available, _ = helpers._effective_host_memory_available_bytes()
    return {
        "status": "NOT_ESTABLISHED_REQUIRES_REAL_KEYGEN_CENSUS",
        "host_available_bytes_now": host_available,
        "host_required_upper_bound_bytes_without_census": requirements["proof_host_peak_bytes"],
        "host_upper_bound_fits_current_available": requirements["proof_host_peak_bytes"] <= host_available,
        "cuda_free_bytes_now": sample["free_memory_bytes"],
        "cuda_workspace_upper_bound_bytes": workspace["device_workspace_peak_bytes"]
        + requirements["icicle_ntt_device_peak_upper_bytes"]
        + requirements["icicle_msm_precomputed_basis_device_upper_bytes"]
        + requirements["icicle_h_resident_device_upper_bytes"],
        "disk_available_bytes_now": helpers._filesystem_available_bytes((output_root / DATASET).resolve())[0],
        "disk_required_upper_bound_bytes_before_keygen_census": proof_components["total_required_bytes"],
        "proof_spill_upper_bound_bytes": requirements["proof_spill_known_bytes"],
        "witness_json_upper_bound_bytes": bounds["witness_json_upper_bytes"],
        "proof_json_upper_bound_bytes": bounds["proof_json_upper_bytes"],
        "srs_copy_bytes": srs_size,
        "note": "This is a conservative source-derived estimate only. The actual Prove gate is recalculated from the completed native keygen census before Final/Witness/Prove.",
    }


def validate_proof_metrics(output: Path) -> dict[str, Any]:
    metrics = load_json(output / "native_proof_metrics.json", "native proof metrics")
    if (
        metrics.get("schema_version") != 4
        or metrics.get("operation") != "native_halo2_initial_instance_and_phase0_advice_commitments"
        or metrics.get("included_in") != "native_create_proof_wall_ms"
        or metrics.get("provenance_checks_disabled") is not True
    ):
        raise RecoveryError("native proof metrics are incomplete or do not record the unbound mode")
    for name in ("commitment_time_ms", "native_create_proof_wall_ms", "native_create_proof_residual_ms", "prove_time_ms"):
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            raise RecoveryError(f"native proof metric {name} is not a positive finite duration")
    materialize = metrics.get("pk_materialize_time_ms")
    if isinstance(materialize, bool) or not isinstance(materialize, (int, float)) or not math.isfinite(float(materialize)) or float(materialize) < 0:
        raise RecoveryError("native proof metric pk_materialize_time_ms is invalid")
    if not math.isclose(
        float(metrics["native_create_proof_wall_ms"]),
        float(metrics["commitment_time_ms"]) + float(metrics["native_create_proof_residual_ms"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RecoveryError("native create_proof wall timing does not equal commitment plus residual")
    if not math.isclose(
        float(metrics["prove_time_ms"]),
        float(metrics["native_create_proof_residual_ms"]) + float(materialize),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RecoveryError("native prove timing does not equal create_proof residual plus PK materialization")
    return metrics


def stage_outputs(output: Path, stage: str) -> tuple[Path, ...]:
    names = {
        "producer": ("statement.json", "public_output.json", "producer_manifest.json"),
        "final": ("artifact_manifest.json", "srs.kzg"),
        "witness": ("witness.json",),
        "prove": ("proof.json", "native_proof_metrics.json", "native_pk_read_timing.json", "gpu_evidence.json", "gpu_prover_record.json"),
        "verify": ("native_verify_metrics.json", "core_run_manifest.json"),
    }.get(stage, ())
    return tuple(output / name for name in names)


def stage_is_complete(
    state: dict[str, Any],
    output: Path,
    stage: str,
    *,
    rayon_threads: int | None = None,
) -> bool:
    record = state.get("stages", {}).get(stage, {})
    setup_exit_zero_recoverable = (
        stage == "setup"
        and record.get("status") in {"interrupted", "exited_zero"}
        and record.get("exit_code") == 0
    )
    if record.get("status") != "success" and not setup_exit_zero_recoverable:
        return False
    if stage == "setup":
        try:
            validate_native_setup_outputs(output, str(state.get("session_id", "")))
            return True
        except RecoveryError:
            return False
    if stage == "setup_receipts":
        try:
            admission = load_json(output / "admission.json", "recovery OGBN admission")
            receipt = load_json(output / "setup_receipt.json", "recovery Setup receipt")
            return (
                admission.get("dataset") == DATASET
                and receipt.get("schema") == "ezkl.recovery-setup-receipt.v1"
                and receipt.get("dataset") == DATASET
                and receipt.get("session_id") == state.get("session_id")
                and receipt.get("setup_command_exit_code") == 0
            )
        except RecoveryError:
            return False
    if stage == "prove_budget":
        try:
            budget = load_json(output / "native_prove_budget.json", "native prove budget")
            policy = state.get("native_runtime_policy")
            snapshot = record.get("capacity_snapshot")
            helpers = formal_helpers()
            return (
                budget.get("schema") == "ezkl.native-prove-budget.v8"
                and budget.get("dataset") == DATASET
                and budget.get("session_id") == state.get("session_id")
                and isinstance(policy, dict)
                and record.get("runtime_policy") == policy
                and snapshot.get("rayon_thread_policy") == policy
                and record.get("capacity_snapshot_sha256") == helpers.canonical_json_digest(snapshot)
                and record.get("native_prove_budget_sha256") == helpers.sha256(output / "native_prove_budget.json")
                and (rayon_threads is None or policy.get("effective_threads") == rayon_threads)
            )
        except (RecoveryError, AttributeError, TypeError):
            return False
    required = stage_outputs(output, stage)
    if not required or not all(nonempty_file(path) for path in required):
        return False
    if stage == "verify":
        try:
            validate_verify_outputs(output)
            return True
        except RecoveryError:
            return False
    return True


def archive_original_spills(ogbn_root: Path, run_id: str) -> None:
    spills = [path for path in ogbn_root.glob("ezkl-halo2-spill-*") if path.is_dir() and not path.is_symlink()]
    if not spills:
        return
    archive = ogbn_root / "interrupted_setup" / f"original_{run_id}"
    archive.mkdir(parents=True, exist_ok=False)
    for spill in spills:
        os.replace(spill, archive / spill.name)


def archive_attempt(output: Path, ogbn_root: Path, run_id: str) -> None:
    archive_root = ogbn_root / "interrupted_setup" / run_id
    archive_root.parent.mkdir(parents=True, exist_ok=True)
    if archive_root.exists():
        raise RecoveryError(f"interrupted setup archive already exists: {archive_root}")
    os.replace(output, archive_root)


def archive_stage_outputs(output: Path, ogbn_root: Path, stage: str, run_id: str) -> None:
    existing = [path for path in stage_outputs(output, stage) if path.exists() or path.is_symlink()]
    if not existing:
        return
    archive = ogbn_root / "interrupted_stages" / run_id / stage
    archive.mkdir(parents=True, exist_ok=False)
    for path in existing:
        os.replace(path, archive / path.name)


def new_ogbn_state(ogbn_root: Path, session_id: str, run_id: str) -> dict[str, Any]:
    output = ogbn_root / f"recovery_attempt_{run_id}_{session_id[:8]}"
    return {
        "schema": "ezkl.recovery_session.v1",
        "session_id": session_id,
        "dataset": DATASET,
        "status": "PREPARED",
        "provenance_checks_disabled": True,
        "output_dir": str(output),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "current_stage": None,
        "active_process": None,
        "stages": {
            name: {"status": "pending"}
            for name in ("setup_budget", "setup", "setup_receipts", "prove_budget", "producer", "final", "witness", "prove", "verify")
        },
    }


def capacity_snapshot(plan: dict[str, Any], output: Path) -> dict[str, Any]:
    check = plan["check"]
    requirements = plan["requirements"]
    helpers = plan["helpers"]
    return {
        "schema": "ezkl.recovery.prove-capacity-snapshot.v1",
        "admission_sha256": plan["admission"].get("sha256"),
        "native_keygen_receipt_sha256": helpers.sha256(output / "native_keygen_receipt.json"),
        "rayon_thread_policy": plan["rayon_thread_policy"],
        "proof_host_peak_bytes": int(requirements["proof_host_peak_bytes"]),
        "host_required_known_bytes": int(check["host_required_known_bytes"]),
        "host_available_bytes": int(check["host_available_bytes"]),
        "cuda_required_workspace_bytes": int(check["cuda_required_workspace_bytes"]),
        "cuda_free_bytes": int(check["cuda_free_bytes"]),
        "disk_required_known_bytes": int(check["disk_required_known_bytes"]),
        "disk_available_bytes": int(check["disk_available_bytes"]),
        "native_prepare_exports": plan["native_prepare_exports"],
    }


def archive_unbound_native_prove_budget(
    state: dict[str, Any],
    session_path: Path,
    output: Path,
    output_root: Path,
    *,
    reason: str,
) -> None:
    budget_path = output / "native_prove_budget.json"
    if not (budget_path.exists() or budget_path.is_symlink()):
        return
    prove_record = state.get("stages", {}).get("prove", {})
    if (
        prove_record.get("status") not in (None, "pending")
        or any((output / name).exists() or (output / name).is_symlink() for name in ("proof.json", "native_proof_metrics.json"))
    ):
        raise RecoveryError("cannot replace an OGBN Prove budget after the Prove stage has started")
    if not regular_file(budget_path):
        raise RecoveryError(f"existing OGBN native Prove budget is unsafe: {budget_path}")
    helpers = formal_helpers()
    digest = helpers.sha256(budget_path)
    archive_dir = output_root / ".sessions" / str(state["session_id"]) / "recovery_budget_history"
    if archive_dir.is_symlink():
        raise RecoveryError(f"OGBN recovery budget archive is unsafe: {archive_dir}")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"native_prove_budget_{digest[:16]}.json"
    if archive_path.exists() or archive_path.is_symlink():
        raise RecoveryError(f"OGBN recovery budget archive already exists: {archive_path}")
    os.replace(budget_path, archive_path)
    history = state.setdefault("native_runtime_policy_history", [])
    if not isinstance(history, list):
        raise RecoveryError("OGBN runtime policy history is malformed")
    history.append({
        "reason": reason,
        "native_prove_budget_sha256": digest,
        "archived_budget_path": str(archive_path),
        "recorded_at_utc": utc_now(),
    })
    atomic_json(session_path, state)


def bind_native_runtime_policy(
    state: dict[str, Any],
    session_path: Path,
    output: Path,
    output_root: Path,
    policy: dict[str, Any],
) -> None:
    previous = state.get("native_runtime_policy")
    prove_record = state.get("stages", {}).get("prove", {})
    if previous != policy and (
        prove_record.get("status") not in (None, "pending")
        or any((output / name).exists() or (output / name).is_symlink() for name in ("proof.json", "native_proof_metrics.json"))
    ):
        raise RecoveryError("cannot change the OGBN Rayon policy after Prove has started")
    budget_path = output / "native_prove_budget.json"
    budget_bound = previous == policy and stage_is_complete(
        state,
        output,
        "prove_budget",
        rayon_threads=int(policy["effective_threads"]),
    )
    if previous != policy or ((budget_path.exists() or budget_path.is_symlink()) and not budget_bound):
        archive_unbound_native_prove_budget(
            state,
            session_path,
            output,
            output_root,
            reason="runtime thread policy changed or prior budget lacks a matching session checkpoint",
        )
        state["stages"]["prove_budget"] = {"status": "pending"}
    state["native_runtime_policy"] = policy
    state["updated_at"] = utc_now()
    atomic_json(session_path, state)


def mark_stage_success(state: dict[str, Any], session_path: Path, stage: str) -> None:
    record = state["stages"][stage]
    record.update({"status": "success", "finished_at": utc_now()})
    state.update({"status": "RUNNING", "active_process": None, "updated_at": utc_now()})
    atomic_json(session_path, state)


def ensure_setup_receipts(state: dict[str, Any], session_path: Path, output: Path, output_root: Path) -> None:
    if stage_is_complete(state, output, "setup_receipts"):
        return
    if not stage_is_complete(state, output, "setup"):
        raise RecoveryError("cannot create OGBN Setup receipts before the native Setup outputs are complete")
    record = state["stages"].setdefault("setup_receipts", {"status": "pending"})
    record.update({"status": "running", "started_at": utc_now()})
    state.update({"status": "RUNNING", "current_stage": "setup_receipts", "updated_at": utc_now()})
    atomic_json(session_path, state)
    try:
        prepare_setup_receipts(output, admission_inputs(output_root)[0], str(state["session_id"]))
    except RecoveryError as exc:
        record.update({"status": "failed", "error": str(exc), "finished_at": utc_now()})
        state.update({"status": "BUDGET_FAILED", "current_stage": "setup_receipts", "active_process": None, "error": str(exc), "updated_at": utc_now()})
        atomic_json(session_path, state)
        raise
    record.update({"status": "success", "finished_at": utc_now()})
    state.update({"status": "RUNNING", "current_stage": "setup_receipts", "updated_at": utc_now()})
    atomic_json(session_path, state)


def ensure_prove_budget(
    state: dict[str, Any],
    session_path: Path,
    output: Path,
    output_root: Path,
    binary: Path,
    summary: dict[str, Any],
    summary_path: Path,
    rayon_threads: int,
    capacity_plan: dict[str, Any] | None = None,
) -> None:
    if stage_is_complete(state, output, "prove_budget", rayon_threads=rayon_threads):
        current_exports = capacity_plan.get("native_prepare_exports") if capacity_plan else None
        existing_snapshot = state.get("stages", {}).get("prove_budget", {}).get("capacity_snapshot")
        if current_exports is None or (
            isinstance(existing_snapshot, dict)
            and existing_snapshot.get("native_prepare_exports") == current_exports
        ):
            return
        archive_unbound_native_prove_budget(
            state,
            session_path,
            output,
            output_root,
            reason="current Prove capacity check derived different validated native prepare exports",
        )
        state["stages"]["prove_budget"] = {"status": "pending"}
        atomic_json(session_path, state)
    ensure_setup_receipts(state, session_path, output, output_root)
    record = state["stages"].setdefault("prove_budget", {"status": "pending"})
    record.update({"status": "running", "started_at": utc_now()})
    state.update({"status": "RUNNING", "current_stage": "prove_budget", "updated_at": utc_now()})
    atomic_json(session_path, state)
    try:
        plan = prepare_native_prove_budget(
            output,
            output_root,
            binary,
            str(state["session_id"]),
            rayon_threads,
            capacity_plan,
        )
    except RecoveryError as exc:
        record.update({"status": "failed", "error": str(exc), "finished_at": utc_now()})
        state.update({"status": "BUDGET_FAILED", "current_stage": "prove_budget", "active_process": None, "error": str(exc), "updated_at": utc_now()})
        atomic_json(session_path, state)
        summary["datasets"]["ogbn_arxiv"] = {
            "status": "BUDGET_FAILED",
            "current_stage": "prove_budget",
            "output_dir": str(output),
            "formal_four_metrics_complete": False,
            "session_path": str(session_path),
            "error": str(exc),
        }
        summary["status"] = "BUDGET_FAILED"
        write_summary(summary_path, summary)
        raise
    record.update({
        "status": "success",
        "finished_at": utc_now(),
        "capacity_check": plan["check"],
        "budget_path": str(output / "native_prove_budget.json"),
        "runtime_policy": plan["rayon_thread_policy"],
        "capacity_snapshot": capacity_snapshot(plan, output),
        "native_prove_budget_sha256": plan["helpers"].sha256(output / "native_prove_budget.json"),
    })
    record["capacity_snapshot_sha256"] = plan["helpers"].canonical_json_digest(record["capacity_snapshot"])
    state.update({"status": "RUNNING", "current_stage": "prove_budget", "active_process": None, "updated_at": utc_now()})
    atomic_json(session_path, state)


def mark_prove_capacity_failed(
    state: dict[str, Any],
    session_path: Path,
    output: Path,
    summary: dict[str, Any],
    summary_path: Path,
    error: str,
) -> None:
    record = state["stages"].setdefault("prove_budget", {"status": "pending"})
    record.update({"status": "failed", "error": error, "finished_at": utc_now()})
    state.update({
        "status": "BUDGET_FAILED",
        "current_stage": "prove_budget",
        "active_process": None,
        "error": error,
        "updated_at": utc_now(),
    })
    atomic_json(session_path, state)
    summary["datasets"]["ogbn_arxiv"] = {
        "status": "BUDGET_FAILED",
        "current_stage": "prove_budget",
        "output_dir": str(output),
        "formal_four_metrics_complete": False,
        "session_path": str(session_path),
        "error": error,
    }
    summary["status"] = "BUDGET_FAILED"
    write_summary(summary_path, summary)


def original_ppi_timing(ppi: Path) -> tuple[float | None, float | None]:
    create_proof_ms: float | None = None
    prove_log = ppi / "logs/prove.log"
    if regular_file(prove_log):
        try:
            values = re.findall(r"EZKL_NATIVE_PHASE DONE stage=native_create_proof_total elapsed_ms=([0-9]+(?:\.[0-9]+)?)", prove_log.read_text(errors="replace"))
            if values:
                create_proof_ms = float(values[-1])
        except OSError:
            pass
    pk_materialize_ms: float | None = None
    pk_timing_path = ppi / "native_pk_read_timing.json"
    if regular_file(pk_timing_path):
        try:
            value = load_json(pk_timing_path, "PPI native PK read timing").get("pk_materialize_ms")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0:
                pk_materialize_ms = float(value)
        except RecoveryError:
            pass
    return create_proof_ms, pk_materialize_ms


def write_summary(path: Path, value: dict[str, Any]) -> None:
    value["updated_at"] = utc_now()
    atomic_json(path, value)


def run_recovery(
    output_root: Path,
    binary: Path,
    *,
    ogbn_only: bool = False,
    prove_threads: int | None = None,
) -> int:
    missing = require_inputs(output_root, binary, include_ppi=not ogbn_only)
    if missing:
        raise RecoveryError("preflight inputs are not ready:\n  " + "\n  ".join(missing))
    lock_fd = acquire_global_lock()
    install_signal_handlers()
    run_id = short_time_id()
    recovery_id = str(uuid.uuid4())
    ppi = output_root / "ppi"
    ogbn_root = output_root / DATASET
    summary_path = output_root / "recovery_summary.json"
    summary: dict[str, Any] = {
        "schema": "ezkl.recovery_summary.v1",
        "session_id": recovery_id,
        "status": "RUNNING",
        "provenance_checks_disabled": True,
        "datasets": {
            "ppi": {"status": "PENDING", "formal_four_metrics_complete": False},
            "ogbn_arxiv": {"status": "PENDING", "formal_four_metrics_complete": False},
        },
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    ppi_result: dict[str, Any] | None = None
    if ogbn_only:
        summary["status"] = "PREPARING_OGBN_ONLY"
    try:
        state: dict[str, Any] | None = None
        output: Path | None = None
        session_path = ogbn_root / "recovery_session.json"
        if ogbn_only:
            prove_threads = DEFAULT_PROVE_THREADS if prove_threads is None else prove_threads
            session_path, state, output = load_budget_failed_ogbn_state(output_root)
            setup_admission = load_json(output / "admission.json", "Setup admission for OGBN runtime policy")
            requested_policy = rayon_thread_policy(setup_admission, prove_threads)

            if regular_file(summary_path):
                summary = load_json(summary_path, "existing recovery summary")
                datasets = summary.get("datasets")
                if not isinstance(datasets, dict):
                    raise RecoveryError("existing recovery summary has a malformed datasets object")
                summary["datasets"] = datasets
            else:
                summary["datasets"] = {}
            summary.update({
                "session_id": recovery_id,
                "status": "RUNNING",
                "provenance_checks_disabled": True,
                "scope": "ogbn_arxiv_only",
                "prove_thread_policy": requested_policy,
            })
            summary["datasets"]["ogbn_arxiv"] = {
                "status": "RUNNING",
                "current_stage": "prove_budget",
                "output_dir": str(output),
                "formal_four_metrics_complete": False,
                "session_path": str(session_path),
            }
        else:
            prove_threads = None
        atomic_json(summary_path, summary)
        if not ogbn_only:
            ppi_verify_dir = ppi / f"recovery_verify_{run_id}_{recovery_id[:8]}"
            ppi_verify_dir.mkdir(parents=True, exist_ok=False)
            ppi_state_path = ppi / "recovery_session.json"
            ppi_state: dict[str, Any] = {
                "schema": "ezkl.recovery_session.v1",
                "session_id": recovery_id,
                "dataset": "ppi",
                "status": "RUNNING",
                "provenance_checks_disabled": True,
                "output_dir": str(ppi_verify_dir),
                "current_stage": "verify",
                "active_process": None,
                "stages": {"verify": {"status": "pending"}},
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            atomic_json(ppi_state_path, ppi_state)
            ppi_env = cpu_environment(ppi_verify_dir, "ppi", recovery_id)
            ppi_verify_command_line = ppi_verify_command(binary, output_root)
            try:
                ppi_verify_wall = run_stage(
                    name="verify",
                    command=ppi_verify_command_line,
                    env=ppi_env,
                    log_path=ppi_verify_dir / "verify.log",
                    session_path=ppi_state_path,
                    state=ppi_state,
                )
                ppi_metrics, ppi_core = validate_verify_outputs(ppi_verify_dir)
            except RecoveryInterrupted as exc:
                ppi_state["status"] = "INTERRUPTED"
                ppi_state["error"] = str(exc)
                ppi_state["updated_at"] = utc_now()
                atomic_json(ppi_state_path, ppi_state)
                create_proof_ms, pk_materialize_ms = original_ppi_timing(ppi)
                atomic_json(ppi / "recovery_result.json", {
                    "schema": "ezkl.recovery_result.v1", "dataset": "ppi",
                    "verification_status": "INTERRUPTED", "timing_status": "incomplete_original_split",
                    "formal_four_metrics_complete": False, "commitment_time_ms": None, "prove_time_ms": None,
                    "native_create_proof_wall_ms": create_proof_ms, "pk_materialize_time_ms": pk_materialize_ms,
                    "proof_size_bytes": (ppi / "proof.json").stat().st_size,
                    "verify_command_wall_ms": ppi_state["stages"]["verify"].get("wall_ms"),
                    "error": str(exc), "recovery_session_id": recovery_id,
                    "provenance_checks_disabled": True, "updated_at": utc_now(),
                })
                summary["datasets"]["ppi"] = {"status": "INTERRUPTED", "formal_four_metrics_complete": False}
                summary["status"] = "INTERRUPTED"
                summary["error"] = str(exc)
                write_summary(summary_path, summary)
                raise
            except RecoveryError as exc:
                ppi_state["status"] = "INTERRUPTED" if isinstance(exc, RecoveryInterrupted) else "FAILED"
                ppi_state["error"] = str(exc)
                ppi_state["updated_at"] = utc_now()
                atomic_json(ppi_state_path, ppi_state)
                proof_size = (ppi / "proof.json").stat().st_size
                create_proof_ms, pk_materialize_ms = original_ppi_timing(ppi)
                ppi_result = {
                    "schema": "ezkl.recovery_result.v1",
                    "dataset": "ppi",
                    "verification_status": "VERIFY_FAILED",
                    "timing_status": "incomplete_original_split",
                    "formal_four_metrics_complete": False,
                    "commitment_time_ms": None,
                    "prove_time_ms": None,
                    "native_create_proof_wall_ms": create_proof_ms,
                    "pk_materialize_time_ms": pk_materialize_ms,
                    "proof_size_bytes": proof_size,
                    "verify_command_wall_ms": ppi_state["stages"]["verify"].get("wall_ms"),
                    "error": str(exc),
                    "recovery_session_id": recovery_id,
                    "provenance_checks_disabled": True,
                    "updated_at": utc_now(),
                }
                atomic_json(ppi / "recovery_result.json", ppi_result)
                summary["datasets"]["ppi"] = {"status": "VERIFY_FAILED", "formal_four_metrics_complete": False, "error": str(exc)}
                summary["status"] = "FAILED_PPI_VERIFY"
                write_summary(summary_path, summary)
                raise RecoveryError(f"PPI Verify failed; OGBN was not started: {exc}") from exc

            ppi_state["stages"]["verify"].update({"status": "success", "wall_ms": ppi_verify_wall, "finished_at": utc_now()})
            ppi_state.update({"status": "VERIFY_OK", "active_process": None, "updated_at": utc_now()})
            atomic_json(ppi_state_path, ppi_state)
            create_proof_ms, pk_materialize_ms = original_ppi_timing(ppi)
            ppi_result = {
                "schema": "ezkl.recovery_result.v1",
                "dataset": "ppi",
                "verification_status": "VERIFY_OK" if ppi_core.get("native_verify_result") is True else "VERIFY_FAILED",
                "timing_status": "incomplete_original_split",
                "formal_four_metrics_complete": False,
                "commitment_time_ms": None,
                "prove_time_ms": None,
                "native_create_proof_wall_ms": create_proof_ms,
                "pk_materialize_time_ms": pk_materialize_ms,
                "proof_size_bytes": (ppi / "proof.json").stat().st_size,
                "verify_time_ms": ppi_metrics["verify_time_ms"],
                "verify_command_wall_ms": ppi_verify_wall,
                "verify_metrics_path": str(ppi_verify_dir / "native_verify_metrics.json"),
                "core_manifest_path": str(ppi_verify_dir / "core_run_manifest.json"),
                "proof_size_definition": "proof.json filesystem stat size in bytes (JSON artifact)",
                "recovery_session_id": recovery_id,
                "provenance_checks_disabled": True,
                "updated_at": utc_now(),
            }
            atomic_json(ppi / "recovery_result.json", ppi_result)
            summary["datasets"]["ppi"] = {
                "status": ppi_result["verification_status"],
                "formal_four_metrics_complete": False,
                "result_path": str(ppi / "recovery_result.json"),
            }
            write_summary(summary_path, summary)

        session_path = ogbn_root / "recovery_session.json"
        if state is None and regular_file(session_path):
            state = load_json(session_path, "OGBN recovery session")
            if state.get("schema") != "ezkl.recovery_session.v1" or state.get("dataset") != DATASET:
                raise RecoveryError(f"unexpected OGBN recovery state schema: {session_path}")
        if state is not None and state.get("status") == "FAILED":
            raise RecoveryError(
                f"previous OGBN stage failed ({state.get('current_stage')}); inspect {session_path} and its log before retrying"
            )
        if state is not None and state.get("status") == "BUDGET_FAILED" and state.get("current_stage") not in {
            "setup_budget", "setup_receipts", "prove_budget"
        }:
            raise RecoveryError(f"unexpected OGBN budget-preparation failure state: {session_path}")
        if state is not None and state.get("status") == "RUNNING" and state.get("active_process"):
            active = state["active_process"]
            pgid = active.get("process_group")
            if isinstance(pgid, int) and process_group_exists(pgid):
                raise RecoveryError(
                    f"recovery state reports an active child process group {pgid}; global lock was acquired but "
                    f"the group is still present. Inspect {session_path}; no stage was started."
                )
            interrupted_stage = active.get("stage")
            if interrupted_stage in state.get("stages", {}):
                state["stages"][interrupted_stage]["status"] = "interrupted"
                state["current_stage"] = interrupted_stage
            state["status"] = "INTERRUPTED"
            state["active_process"] = None
            state["updated_at"] = utc_now()
            atomic_json(session_path, state)
        elif state is not None and state.get("status") == "RUNNING":
            current = state.get("current_stage")
            record = state.get("stages", {}).get(current, {})
            if record.get("status") in {"running", "exited_zero"}:
                record["status"] = "interrupted"
                state["status"] = "INTERRUPTED"
                state["updated_at"] = utc_now()
                atomic_json(session_path, state)
        if state is not None and state.get("status") == "INTERRUPTED" and state.get("current_stage") == "prove":
            raise RecoveryError(
                "OGBN Prove was interrupted and is not resumable from memory; existing completed stages are preserved, "
                f"but automatic re-proving is disabled. Inspect {session_path}."
            )
        output = Path(state["output_dir"]) if state is not None else Path()
        if state is not None and not output.is_absolute():
            output = ROOT / output
        if state is not None and not stage_is_complete(state, output, "setup"):
            if state.get("status") == "INTERRUPTED" and state.get("current_stage") == "setup":
                archive_attempt(output, ogbn_root, f"{run_id}_{recovery_id[:8]}_{state.get('session_id', 'unknown')[:8]}")
                state = None
            elif state.get("stages", {}).get("setup", {}).get("status") == "success":
                archive_attempt(output, ogbn_root, f"{run_id}_{recovery_id[:8]}_{state.get('session_id', 'unknown')[:8]}")
                state = None
        if state is None:
            archive_original_spills(ogbn_root, f"{run_id}_{recovery_id[:8]}")
            state = new_ogbn_state(ogbn_root, recovery_id, run_id)
            output = Path(state["output_dir"])
            if not output.is_absolute():
                output = ROOT / output
            atomic_json(session_path, state)
        else:
            # Reuse completed native stages only when the recovery journal and their real outputs agree.
            output = Path(state["output_dir"])
            if not output.is_absolute():
                output = ROOT / output
            state["status"] = "RUNNING"
            state["updated_at"] = utc_now()
            for stage in ("producer", "final", "witness", "prove", "verify"):
                if state.get("stages", {}).get(stage, {}).get("status") == "success" and not stage_is_complete(state, output, stage):
                    raise RecoveryError(f"recorded OGBN {stage} success is missing required outputs; preserved state at {session_path}")
            atomic_json(session_path, state)

        ogbn_recovery_id = str(state.get("session_id", ""))
        if re.fullmatch(r"[0-9a-fA-F-]{36}", ogbn_recovery_id) is None:
            raise RecoveryError(f"OGBN recovery session has an invalid diagnostic session id: {session_path}")

        policy_admission = (
            load_json(output / "admission.json", "Setup admission for OGBN runtime policy")
            if regular_file(output / "admission.json")
            else admission_inputs(output_root)[2]
        )
        memory_model = policy_admission.get("native_memory_model")
        admitted_threads = memory_model.get("fft_parallel_thread_count") if isinstance(memory_model, dict) else None
        effective_threads = admitted_threads if prove_threads is None else prove_threads
        runtime_policy = rayon_thread_policy(policy_admission, effective_threads)
        bind_native_runtime_policy(state, session_path, output, output_root, runtime_policy)

        commands = ogbn_commands(binary, output_root, output)
        required_by_stage = {
            "producer": lambda: all(nonempty_file(output / name) for name in ("statement.json", "public_output.json", "producer_manifest.json")),
            "final": lambda: all(nonempty_file(output / name) for name in ("artifact_manifest.json", "srs.kzg")),
            "witness": lambda: nonempty_file(output / "witness.json"),
            "prove": lambda: all(nonempty_file(output / name) for name in ("proof.json", "native_proof_metrics.json")),
            "verify": lambda: all(nonempty_file(output / name) for name in ("native_verify_metrics.json", "core_run_manifest.json")),
        }
        run_order = ("setup", "producer", "final", "witness", "prove", "verify")
        for stage in run_order:
            if stage != "setup" and not stage_is_complete(
                state, output, "prove_budget", rayon_threads=effective_threads
            ):
                ensure_prove_budget(
                    state,
                    session_path,
                    output,
                    output_root,
                    binary,
                    summary,
                    summary_path,
                    effective_threads,
                )
            if stage_is_complete(state, output, stage):
                print(f"[recovery] REUSE completed OGBN stage {stage}", flush=True)
                continue
            prior = state.get("stages", {}).get(stage, {})
            if prior.get("status") == "failed":
                raise RecoveryError(f"previous OGBN {stage} failed; inspect its log before any retry")
            if prior.get("status") == "interrupted":
                if stage == "setup":
                    raise RecoveryError("interrupted OGBN setup should have been archived and restarted")
                if stage == "prove":
                    raise RecoveryError("interrupted OGBN Prove cannot resume; automatic re-proving is disabled")
                archive_stage_outputs(output, ogbn_root, stage, f"{run_id}_{recovery_id[:8]}_{stage}")
                state["stages"][stage] = {"status": "pending"}
                atomic_json(session_path, state)
            if stage == "setup":
                if not stage_is_complete(state, output, "setup"):
                    setup_budget_record = state["stages"].setdefault("setup_budget", {"status": "pending"})
                    setup_budget_record.update({"status": "running", "started_at": utc_now()})
                    state.update({"status": "RUNNING", "current_stage": "setup_budget", "updated_at": utc_now()})
                    atomic_json(session_path, state)
                    try:
                        setup_plan = current_capacity_plan(
                            output_root,
                            output,
                            ogbn_recovery_id,
                            native_census=None,
                            srs_pending=True,
                        )
                        if output.exists():
                            if not (
                                state.get("current_stage") == "setup_budget"
                                and not nonempty_file(output / "native_keygen_receipt.json")
                                and not nonempty_file(output / "pk.key")
                            ):
                                raise RecoveryError(f"fresh OGBN Setup output path unexpectedly contains prior work: {output}")
                        else:
                            output.mkdir(parents=True)
                        prepare_native_keygen_budget(output, binary, ogbn_recovery_id, setup_plan)
                    except RecoveryError as exc:
                        setup_budget_record.update({"status": "failed", "error": str(exc), "finished_at": utc_now()})
                        state.update({"status": "BUDGET_FAILED", "current_stage": "setup_budget", "active_process": None, "error": str(exc), "updated_at": utc_now()})
                        atomic_json(session_path, state)
                        summary["datasets"]["ogbn_arxiv"] = {
                            "status": "BUDGET_FAILED",
                            "current_stage": "setup_budget",
                            "output_dir": str(output),
                            "formal_four_metrics_complete": False,
                            "session_path": str(session_path),
                            "error": str(exc),
                        }
                        write_summary(summary_path, summary)
                        raise
                    setup_budget_record.update({"status": "success", "finished_at": utc_now(), "capacity_check": setup_plan["check"]})
                    state.update({"status": "RUNNING", "current_stage": "setup_budget", "updated_at": utc_now()})
                    atomic_json(session_path, state)
                stage_env = gpu_environment(output, DATASET, ogbn_recovery_id)
            elif stage == "producer":
                stage_env = producer_environment()
            else:
                if stage == "verify":
                    stage_env = cpu_environment(
                        output,
                        DATASET,
                        ogbn_recovery_id,
                        runtime_policy=runtime_policy,
                    )
                    stage_runtime_policy = runtime_policy
                    stage_prepare_exports = None
                else:
                    census_receipt = validate_native_setup_outputs(output, ogbn_recovery_id)
                    try:
                        stage_plan = current_capacity_plan(
                            output_root,
                            output,
                            ogbn_recovery_id,
                            native_census=census_receipt["native_census"],
                            srs_pending=not nonempty_file(output / "srs.kzg"),
                            witness_already_written=stage == "prove",
                            rayon_threads=effective_threads,
                        )
                    except RecoveryError as exc:
                        mark_prove_capacity_failed(
                            state,
                            session_path,
                            output,
                            summary,
                            summary_path,
                            str(exc),
                        )
                        raise
                    stage_prepare_exports = stage_plan["native_prepare_exports"]
                    if stage == "prove":
                        ensure_prove_budget(
                            state,
                            session_path,
                            output,
                            output_root,
                            binary,
                            summary,
                            summary_path,
                            effective_threads,
                            stage_plan,
                        )
                    stage_env = gpu_environment(
                        output,
                        DATASET,
                        ogbn_recovery_id,
                        runtime_policy=runtime_policy,
                        prepare_exports=stage_prepare_exports,
                    )
                    stage_runtime_policy = runtime_policy
            if stage in {"setup", "producer"}:
                stage_runtime_policy = None
                stage_prepare_exports = None
            log_path = output / "logs" / f"recovery_{stage}.log"
            wall_ms = run_stage(
                name=stage,
                command=commands[stage],
                env=stage_env,
                log_path=log_path,
                session_path=session_path,
                state=state,
                runtime_policy=stage_runtime_policy,
                prepare_exports=stage_prepare_exports,
            )
            try:
                if stage == "setup":
                    validate_native_setup_outputs(output, ogbn_recovery_id)
                elif not required_by_stage[stage]():
                    raise RecoveryError(f"OGBN {stage} exited successfully but required artifacts are missing")
                elif stage == "prove":
                    validate_proof_metrics(output)
                elif stage == "verify":
                    validate_verify_outputs(output)
            except RecoveryError as exc:
                state["stages"][stage].update({"status": "failed", "error": str(exc)})
                state.update({"status": "FAILED", "current_stage": stage, "active_process": None, "updated_at": utc_now()})
                atomic_json(session_path, state)
                raise
            if stage == "setup":
                state["stages"][stage]["wall_ms"] = wall_ms
                mark_stage_success(state, session_path, stage)
                ensure_setup_receipts(state, session_path, output, output_root)
            else:
                state["stages"][stage]["wall_ms"] = wall_ms
                mark_stage_success(state, session_path, stage)
            summary["datasets"]["ogbn_arxiv"] = {
                "status": "RUNNING",
                "current_stage": stage,
                "output_dir": str(output),
                "formal_four_metrics_complete": False,
            }
            write_summary(summary_path, summary)

        proof_metrics = validate_proof_metrics(output)
        verify_metrics, core = validate_verify_outputs(output)
        gpu_profile_path = output / "gpu_profile_conformance.json"
        gpu_profile = load_json(gpu_profile_path, "OGBN GPU profile conformance") if regular_file(gpu_profile_path) else {}
        proof_size = (output / "proof.json").stat().st_size
        four_complete = proof_size > 0 and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0
            for value in (
                proof_metrics.get("commitment_time_ms"),
                proof_metrics.get("prove_time_ms"),
                verify_metrics.get("verify_time_ms"),
            )
        )
        ogbn_result = {
            "schema": "ezkl.recovery_result.v1",
            "dataset": DATASET,
            "verification_status": "VERIFY_OK" if core.get("native_verify_result") is True else "VERIFY_FAILED",
            "timing_status": "complete" if four_complete else "incomplete",
            "formal_four_metrics_complete": four_complete,
            "commitment_time_ms": proof_metrics.get("commitment_time_ms"),
            "prove_time_ms": proof_metrics.get("prove_time_ms"),
            "native_create_proof_wall_ms": proof_metrics.get("native_create_proof_wall_ms"),
            "pk_materialize_time_ms": proof_metrics.get("pk_materialize_time_ms"),
            "proof_size_bytes": proof_size,
            "proof_size_definition": "proof.json filesystem stat size in bytes (JSON artifact)",
            "verify_time_ms": verify_metrics.get("verify_time_ms"),
            "verify_command_wall_ms": state["stages"]["verify"].get("wall_ms"),
            "stage_command_wall_ms": {stage: state["stages"][stage].get("wall_ms") for stage in run_order},
            "native_proof_metrics_path": str(output / "native_proof_metrics.json"),
            "native_verify_metrics_path": str(output / "native_verify_metrics.json"),
            "gpu_profile_conformance": gpu_profile.get("profile_conformance"),
            "gpu_profile_conformance_detail": gpu_profile.get("detail"),
            "gpu_profile_conformance_path": str(gpu_profile_path) if gpu_profile else None,
            "core_manifest_path": str(output / "core_run_manifest.json"),
            "recovery_session_id": state.get("session_id"),
            "provenance_checks_disabled": True,
            "updated_at": utc_now(),
        }
        result_path = ogbn_root / "recovery_result.json"
        atomic_json(result_path, ogbn_result)
        state.update({"status": "VERIFY_OK", "current_stage": None, "active_process": None, "updated_at": utc_now()})
        atomic_json(session_path, state)
        summary["datasets"]["ogbn_arxiv"] = {
            "status": ogbn_result["verification_status"],
            "formal_four_metrics_complete": four_complete,
            "result_path": str(result_path),
            "output_dir": str(output),
        }
        if ogbn_only:
            summary["status"] = ogbn_result["verification_status"]
            summary["formal_four_metrics_complete"] = four_complete
        else:
            assert ppi_result is not None
            summary["status"] = "VERIFY_OK" if ppi_result["verification_status"] == "VERIFY_OK" and ogbn_result["verification_status"] == "VERIFY_OK" else "FAILED"
            summary["formal_four_metrics_complete"] = bool(ppi_result["formal_four_metrics_complete"] and four_complete)
        write_summary(summary_path, summary)
        prefix = "VERIFY_OK OGBN" if ogbn_only else "VERIFY_OK PPI; OGBN"
        print(f"[recovery] {prefix} result={result_path}; summary={summary_path}", flush=True)
        return 0
    except RecoveryInterrupted as exc:
        summary["status"] = "INTERRUPTED"
        summary["error"] = str(exc)
        if summary["datasets"]["ogbn_arxiv"].get("status") == "RUNNING":
            session_path = ogbn_root / "recovery_session.json"
            if regular_file(session_path):
                try:
                    interrupted = load_json(session_path, "OGBN recovery session")
                    summary["datasets"]["ogbn_arxiv"].update({
                        "status": "INTERRUPTED",
                        "current_stage": interrupted.get("current_stage"),
                        "session_path": str(session_path),
                    })
                except RecoveryError:
                    pass
        write_summary(summary_path, summary)
        print(f"[recovery] {exc}", file=sys.stderr, flush=True)
        return 128 + (STOP_SIGNAL or signal.SIGTERM)
    except RecoveryError as exc:
        if summary.get("status") == "RUNNING":
            summary["status"] = "FAILED"
            summary["error"] = str(exc)
            session_path = ogbn_root / "recovery_session.json"
            if regular_file(session_path):
                try:
                    state = load_json(session_path, "OGBN recovery session")
                    current = state.get("current_stage")
                    summary["datasets"]["ogbn_arxiv"].update({
                        "status": state.get("status", "FAILED"),
                        "current_stage": current,
                        "session_path": str(session_path),
                        "output_dir": state.get("output_dir"),
                    })
                except RecoveryError:
                    pass
            write_summary(summary_path, summary)
        print(f"[recovery] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        ACTIVE_PROCESS = None
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    result.add_argument("--binary", default="target/gpu/release/ezkl")
    result.add_argument(
        "--check",
        action="store_true",
        help="read-only capacity/command check; writes no files or formal outputs, and may launch the bounded CUDA --memory-only sampler",
    )
    result.add_argument("--ogbn-only", action="store_true", help="resume only the existing OGBN BUDGET_FAILED/prove_budget session; skip PPI Verify")
    result.add_argument("--prove-threads", type=int, help="cap Final/Witness/Prove/Verify Rayon workers at or below the Setup admission limit")
    return result


def main() -> int:
    args = parser().parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    binary = Path(args.binary)
    if not binary.is_absolute():
        binary = ROOT / binary
    try:
        if args.check:
            if args.ogbn_only:
                return print_ogbn_only_check(
                    output_root,
                    binary,
                    DEFAULT_PROVE_THREADS if args.prove_threads is None else args.prove_threads,
                )
            return print_check(output_root, binary)
        return run_recovery(
            output_root,
            binary,
            ogbn_only=args.ogbn_only,
            prove_threads=args.prove_threads,
        )
    except RecoveryError as exc:
        print(f"[recovery] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
