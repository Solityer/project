from __future__ import annotations

import csv
import io
import json
import math
import os
import pathlib
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping

import numpy as np

from .backend import backend_state
from .config import generated_config, load_app_config
from .constants import COMMITMENT_BACKEND_ID, DATASET_ORDER, PROOF_BACKEND_ID, PROTOCOL_ID
from .data import load_dataset, normalize_graph
from .forward import forward_full
from .formal_gate import require_formal_experiment
from .model import load_model
from .quantization import resolve_quant_scale_profile
from .proof import artifact_ids, build_proof
from .resources import check_resource_layout, required_resource_paths, resolve_local
from .registry import (
    canonical_registry_public_path,
    canonical_registry_secret_path,
    validate_registry_pair,
)
from .gpu_execution import validate_gpu_execution_evidence
from .result_validation import (
    artifact_digest_map,
    canonical_artifact_dir,
    manifest_run_digest,
    required_artifact_file,
    relative_path,
    resolve_results_dir,
    required_artifact_names,
    validate_artifact_inventory,
    validate_canonical_cell_dir_from_path,
)
from .util import digest_file, digest_json, ensure_dir, read_json, require_dir, write_json, write_text_atomic
from .verifier_context import _array_digest, _x_digest, canonical_graph_identity, load_public_y


def selected_datasets(value: str) -> List[str]:
    if value == "all":
        return DATASET_ORDER[:]
    out = [item.strip() for item in value.split(",") if item.strip()]
    bad = sorted(set(out) - set(DATASET_ORDER))
    if bad:
        raise ValueError(f"unsupported datasets: {bad}")
    return out


def selected_backends(value: str) -> List[str]:
    if value == "all":
        return ["cpu", "gpu"]
    out = [item.strip() for item in value.split(",") if item.strip()]
    bad = sorted(set(out) - {"cpu", "gpu"})
    if bad:
        raise ValueError(f"unsupported backends: {bad}")
    return out


def _same_quantization(
    config: Any,
    registry_binding: Mapping[str, Any],
    proof_metadata: Mapping[str, Any],
) -> bool:
    """Compare authenticated quantization identity across all run inputs."""

    try:
        config_id = str(config.entries.get("quant_cfg_id", ""))
        proof_id = str(proof_metadata.get("quant_cfg_id", ""))
        registered_profile = registry_binding.get("authenticated_quant_scales")
        proof_profile = proof_metadata.get("authenticated_quant_scales")
        if not isinstance(registered_profile, Mapping) or not isinstance(
            proof_profile, Mapping
        ):
            return False
        registered = resolve_quant_scale_profile(
            config_id,
            registered_profile,
        )
        proved = resolve_quant_scale_profile(
            proof_id,
            proof_profile,
        )
        return (
            config_id == proof_id
            and registered == proved
            and registry_binding.get("quant_scale_chain")
            == proof_metadata.get("quant_scale_chain")
        )
    except (KeyError, TypeError, ValueError):
        return False


def _ensure_real_directory(path: pathlib.Path) -> None:
    """Create one already-contained directory and authenticate its inode."""
    try:
        ensure_dir(path, label="result directory")
    except OSError as exc:
        raise RuntimeError(f"cannot inspect result directory: {path}") from exc


def _prepare_artifact_directory(
    baseline: pathlib.Path,
    results_root: pathlib.Path,
    artifact_dir: pathlib.Path,
    force: bool,
) -> pathlib.Path:
    """Prepare one canonical cell before any proof bytes are written."""
    # Create and authenticate each component separately.  Calling mkdir on
    # the full path first would permit a pre-existing ``runs``/backend alias
    # to redirect proof output outside the baseline.
    _ensure_real_directory(results_root)
    runs_dir = results_root / "runs"
    _ensure_real_directory(runs_dir)
    backend_dir = artifact_dir.parent
    _ensure_real_directory(backend_dir)
    if artifact_dir.exists() or os.path.lexists(artifact_dir):
        # Authenticate the exact canonical cell before inspecting or
        # removing any stale entry.  In particular, never recursively remove
        # this directory: a nested stale work directory may contain aliases
        # or paths outside the cell.
        artifact_dir = validate_canonical_cell_dir_from_path(baseline, artifact_dir)
    else:
        _ensure_real_directory(artifact_dir)
    artifact_dir = validate_canonical_cell_dir_from_path(baseline, artifact_dir)

    # A cell is an exact top-level wire directory.  Existing entries are
    # accepted only as owned one-link regular files before an explicit
    # ``--force`` rerun; aliases, nested work directories, FIFOs, and other
    # special files are always rejected.  This prevents a stale/partial cell
    # from being mistaken for a newly produced proof.
    backend = artifact_dir.parent.name
    expected_names = set(required_artifact_names(backend, include_manifest=True))
    children = list(artifact_dir.iterdir())
    for child in children:
        if child.name not in expected_names:
            raise RuntimeError(f"non-canonical artifact entry is forbidden: {child}")
        info = child.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"artifact entry must not be a symlink: {child}")
        if stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"nested artifact directory is forbidden: {child}")
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"artifact entry must be an owned regular file: {child}")
    if children and not force:
        raise RuntimeError(
            f"artifact cell is stale or partial; refuse to reuse without --force: {artifact_dir}"
        )
    if children:
        # The target is the already authenticated canonical cell, and every
        # child was checked above.  Remove only these exact entries; never
        # recursively clear a parent/results directory.
        for child in children:
            child.unlink()
    return artifact_dir


def _resolve_public_y(config: Any, baseline_root: pathlib.Path, dataset: str) -> pathlib.Path:
    """Resolve the single canonical baseline-local public-Y source.

    The resource contract is the sole source of the canonical path.  The
    configured path must match it exactly; the proof artifact and
    reference/run directories are never searched.
    """
    baseline_root = pathlib.Path(baseline_root).expanduser().resolve()
    expected = required_resource_paths(baseline_root, dataset)["public_y_path"]
    configured = resolve_local(baseline_root, config.entries.get("public_y_path", ""), "public_y_path")
    if configured != expected:
        raise ValueError(f"config public_y_path must equal canonical resource path: {expected}")
    if configured.name != expected.name:
        raise ValueError(f"config public_y_path has non-canonical filename: {configured}")
    try:
        info = configured.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"canonical baseline-local public Y is missing: {configured}") from exc
    except OSError as exc:
        raise OSError(f"cannot inspect canonical baseline-local public Y: {configured}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"canonical baseline-local public Y must not be a symlink: {configured}")
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"canonical baseline-local public Y must be an owned regular file: {configured}")
    return configured


def _load_and_check_public_y_descriptor(
    path: pathlib.Path,
    baseline: pathlib.Path,
    dataset: str,
    registry_binding: Mapping[str, Any],
) -> tuple[pathlib.Path, Dict[str, Any], Any]:
    """Load the canonical Y descriptor and compare trusted setup identities."""

    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read canonical public-Y descriptor: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("canonical public-Y descriptor must be an object")
    descriptor = dict(payload)
    digest = descriptor.pop("descriptor_digest", None)
    if not isinstance(digest, str) or digest != digest_json(descriptor):
        raise RuntimeError("canonical public-Y descriptor digest is invalid")
    expected_path = str(required_resource_paths(baseline, dataset)["public_y_path"].relative_to(baseline))
    checks = {
        "schema": payload.get("schema") == registry_binding.get("public_y_schema"),
        "public_y_schema": payload.get("public_y_schema") == registry_binding.get("public_y_schema"),
        "public_y_path": payload.get("public_y_path") == expected_path == registry_binding.get("public_y_path"),
        "descriptor_digest": digest == registry_binding.get("public_y_descriptor_digest"),
        "y_digest": payload.get("y_digest") == registry_binding.get("public_y_digest"),
        "reference_y_digest": payload.get("reference_y_digest") == registry_binding.get("reference_y_digest"),
        "shape": payload.get("public_y_shape") == registry_binding.get("public_y_shape"),
        "reference_forward_profile_id": payload.get("reference_forward_profile_id") == registry_binding.get("reference_forward_profile_id"),
        "reference_forward_engine": payload.get("reference_forward_engine") == registry_binding.get("reference_forward_engine"),
        "reference_forward_source_sha256": payload.get("reference_forward_source_sha256") == registry_binding.get("reference_forward_source_sha256"),
        "X_pub_digest": payload.get("X_pub_digest") == registry_binding.get("X_pub_digest"),
        "full_dataset": payload.get("full_dataset") == registry_binding.get("full_dataset"),
        "graph_identity": payload.get("graph_identity") == registry_binding.get("graph_identity"),
        "independent_reference_forward": payload.get("independent_reference_forward") == registry_binding.get("independent_reference_forward") is True,
    }
    if not all(checks.values()):
        raise RuntimeError("canonical public-Y descriptor does not match trusted registry: " + ",".join(sorted(key for key, value in checks.items() if not value)))
    output = load_public_y(path)
    return path, payload, output


def _baseline_root(project_root: pathlib.Path, zk_root: pathlib.Path) -> pathlib.Path:
    project = require_dir(project_root, label="formal project root")
    baseline = require_dir(zk_root, label="formal zkGPT root")
    if project != baseline:
        raise ValueError("zkGPT formal execution cannot use an external project root")
    return baseline


def run_one(
    dataset_name: str,
    backend: str,
    project_root: pathlib.Path,
    zk_root: pathlib.Path,
    results_dir: pathlib.Path,
    force: bool,
) -> Dict[str, Any]:
    require_formal_experiment()
    baseline = _baseline_root(project_root, zk_root)
    results_root = resolve_results_dir(baseline, results_dir)
    if dataset_name not in DATASET_ORDER or backend not in {"cpu", "gpu"}:
        raise ValueError(f"invalid formal cell: {dataset_name}/{backend}")
    # This checks only baseline-local paths and file presence; no dataset file
    # is opened here.  The native loader below is the formal owner of content
    # validation and is intentionally never called by prepare_resources.py.
    check_resource_layout(baseline, [dataset_name], require_staged=True)
    config_path = generated_config(dataset_name, backend, baseline, baseline)
    config = load_app_config(config_path, baseline)
    model_param_id = str(config.entries["model_param_id"])
    try:
        expected_registry_public_path = canonical_registry_public_path(baseline, model_param_id)
        expected_registry_secret_path = canonical_registry_secret_path(baseline, model_param_id)
        registry_pair = validate_registry_pair(
            baseline,
            model_param_id,
            expected_model_arch_id=config.entries.get("model_arch_id"),
            expected_commitment_backend=COMMITMENT_BACKEND_ID,
        )
        if (
            registry_pair["public_path"] != expected_registry_public_path
            or registry_pair["secret_path"] != expected_registry_secret_path
        ):
            raise RuntimeError("model registry pair is not at the canonical public/private paths")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "canonical model registry public/private setup is missing or invalid; "
            "run scripts/register_model.py after staging resources"
        ) from exc
    registry_path = registry_pair["public_path"]
    registry_transaction_id = registry_pair["transaction_id"]
    public_y_path = _resolve_public_y(config, baseline, dataset_name)
    _public_y_path, public_y_descriptor, public_y_values = _load_and_check_public_y_descriptor(
        public_y_path,
        baseline,
        dataset_name,
        registry_pair["public"]["registration_binding"],
    )
    artifact_dir = canonical_artifact_dir(baseline, backend, dataset_name)
    artifact_dir = _prepare_artifact_directory(baseline, results_root, artifact_dir, force)
    started = time.perf_counter()
    dataset = load_dataset(config)
    graph = normalize_graph(dataset, config)
    if graph.num_nodes != dataset.num_nodes:
        raise RuntimeError(f"{dataset_name}/{backend} is not full dataset")
    model = load_model(
        config.checkpoint_bundle,
        model_arch_id=config.entries.get("model_arch_id"),
        model_param_id=config.entries.get("model_param_id"),
        quant_cfg_id=config.entries.get("quant_cfg_id"),
        static_table_id=config.entries.get("static_table_id"),
        rounding_mode=config.rounding_mode,
    )
    registry_binding = registry_pair["public"]["registration_binding"]
    expected_graph_identity = canonical_graph_identity(
        graph,
        dataset_name,
        full_dataset=graph.num_nodes == dataset.num_nodes,
    )
    if registry_binding.get("graph_identity") != expected_graph_identity or registry_binding.get("full_dataset") is not True:
        raise RuntimeError(f"{dataset_name}/{backend} graph/full-dataset identity does not match trusted setup")
    expected_x_digest = _x_digest(
        graph,
        len(model.hidden_layers) + 1,
        model.class_count,
        model.k_out,
        int(model.quant_scales["input"]),
    )
    if registry_binding.get("X_pub_digest") != expected_x_digest:
        raise RuntimeError(f"{dataset_name}/{backend} X_pub digest does not match trusted setup")
    if tuple(public_y_values.shape) != (int(graph.num_nodes), int(model.class_count)):
        raise RuntimeError(f"{dataset_name}/{backend} canonical public-Y shape does not match model/graph")
    gpu_seed = {
        "dataset": dataset_name,
        "backend": backend,
        "N": graph.num_nodes,
        "E": int(graph.src.size),
        "checkpoint": str(config.checkpoint_bundle),
    }
    gpu_info = backend_state(backend, zk_root, gpu_seed)
    forward_started = time.perf_counter()
    trace = forward_full(graph, model, config.rounding_mode)
    forward_ms = (time.perf_counter() - forward_started) * 1000.0
    trace_y = np.asarray(trace.y, dtype=object)
    if not np.array_equal(trace_y, public_y_values):
        raise RuntimeError(f"{dataset_name}/{backend} prover trace.Y does not match canonical public Y")
    if _array_digest("Y", trace_y) != registry_binding.get("public_y_digest"):
        raise RuntimeError(f"{dataset_name}/{backend} prover trace.Y digest does not match trusted setup")
    ids = artifact_ids(config, dataset)
    paths = {"artifact_dir": artifact_dir, "zk_root": baseline}
    proof = build_proof(config, dataset, graph, model, trace, backend, gpu_info, paths)
    if proof.verification["status"] != "VERIFY_OK":
        raise RuntimeError(f"{dataset_name}/{backend} verifier rejected: {proof.verification}")
    # Persist only run-scoped proof-authoring evidence.  The initial device
    # probe remains part of M_pub but cannot satisfy the result collector.
    if backend == "gpu":
        gpu_info = dict(proof.proof_payload.get("GPU_proof_contribution", {}))
        hot_path = read_json(required_artifact_file(artifact_dir, "gpu_proof_hot_path.json"))
        if gpu_info != hot_path:
            raise RuntimeError("GPU proof contribution and hot-path evidence differ")
        gpu_valid, gpu_errors = validate_gpu_execution_evidence(
            hot_path,
            session_nonce=str(proof.metadata.get("session_nonce", "")),
            final_transcript_digest=str(proof.transcript.get("final_digest", "")),
        )
        if not gpu_valid:
            raise RuntimeError("GPU proof evidence failed before runner persistence: " + ";".join(gpu_errors))
    write_json(artifact_dir / "metadata.json", proof.metadata, root=baseline)
    write_json(artifact_dir / "commitments.json", proof.commitments, root=baseline)
    write_json(artifact_dir / "transcript.json", proof.transcript, root=baseline)
    write_json(artifact_dir / "proof_payload.json", proof.proof_payload, root=baseline)
    write_json(artifact_dir / "verification.json", proof.verification, root=baseline)
    write_json(artifact_dir / "witness_manifest.json", proof.witness_manifest, root=baseline)
    write_json(artifact_dir / "gpu_backend.json", gpu_info, root=baseline)
    write_json(artifact_dir / "data_id.json", ids["data_id"], root=baseline)
    write_json(artifact_dir / "model_id.json", ids["model_id"], root=baseline)
    write_json(artifact_dir / "config_id.json", ids["config_id"], root=baseline)
    benchmark = dict(proof.benchmark)
    benchmark["forward_time_ms"] = forward_ms
    benchmark["wall_time_ms"] = (time.perf_counter() - started) * 1000.0
    write_json(artifact_dir / "benchmark_summary.json", benchmark, root=baseline)
    validate_artifact_inventory(
        artifact_dir, backend, require_independent=False, require_manifest=False
    )
    # The proof metadata is untrusted and is never allowed to nominate the
    # registry used by the independent replay.  Derive the trust root from the
    # canonical generated config/model identity instead.
    final_registry_pair = validate_registry_pair(
        baseline,
        model_param_id,
        expected_model_arch_id=config.entries.get("model_arch_id"),
        expected_commitment_backend=COMMITMENT_BACKEND_ID,
    )
    if final_registry_pair["transaction_id"] != registry_transaction_id:
        raise RuntimeError("model registry transaction changed during formal run")
    same_quantization = _same_quantization(
        config,
        final_registry_pair["public"]["registration_binding"],
        proof.metadata,
    )
    registry_path = final_registry_pair["public_path"]
    public_y_path = _resolve_public_y(config, baseline, dataset_name)
    replay = subprocess.run(
        [
            sys.executable,
            str(baseline / "scripts" / "verify_proof.py"),
            "--project-root", str(baseline),
            "--config", str(config_path),
            "--trusted-registry-public", str(registry_path),
            "--artifact-dir", str(artifact_dir),
            "--public-y", str(public_y_path),
        ],
        cwd=str(baseline),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if replay.returncode != 0:
        raise RuntimeError(
            f"{dataset_name}/{backend} independent verifier rejected\n"
            f"stdout:\n{replay.stdout}\n"
            f"stderr:\n{replay.stderr}"
        )
    independent_verification_path = artifact_dir / "independent_verification.json"
    independent_verification = read_json(
        required_artifact_file(artifact_dir, "independent_verification.json")
    )
    verified_fields = independent_verification.get("verified_fields")
    if independent_verification.get("status") != "VERIFY_OK" or not isinstance(verified_fields, dict):
        raise RuntimeError(f"{dataset_name}/{backend} independent verifier did not produce verified fields")
    timing_tokens = [token for token in replay.stdout.split() if token.startswith("verify_time_ms=")]
    if len(timing_tokens) != 1:
        raise RuntimeError("independent verifier did not report one canonical verify_time_ms")
    try:
        independent_verify_ms = float(timing_tokens[0].split("=", 1)[1])
    except ValueError as exc:
        raise RuntimeError("independent verifier reported malformed verify_time_ms") from exc
    if not math.isfinite(independent_verify_ms) or independent_verify_ms < 0:
        raise RuntimeError("independent verifier reported invalid verify_time_ms")
    benchmark["verify_time_ms"] = independent_verify_ms
    write_json(artifact_dir / "benchmark_summary.json", benchmark, root=baseline)
    required_verified_true = (
        "full_dataset",
        "full_gat_forward_proved",
        "final_logits_bound",
        "proof_schema_scan",
        "privacy_masking_verified",
        "independent_reference_forward",
    )
    if any(verified_fields.get(key) is not True for key in required_verified_true):
        raise RuntimeError(
            f"{dataset_name}/{backend} independent verifier did not close the full formal relation"
        )
    if verified_fields.get("verifier_reads_witness") is not False or verified_fields.get("witness_included_in_proof") is not False:
        raise RuntimeError(
            f"{dataset_name}/{backend} independent verifier witness boundary is not closed"
        )
    if verified_fields.get("backend") != backend or verified_fields.get("dataset") != dataset_name:
        raise RuntimeError(
            f"{dataset_name}/{backend} independent verifier identity does not match the requested cell"
        )
    expected_hidden = [
        {"layer_index": int(layer.layer_index), "head_count": int(layer.head_count), "head_dim": int(layer.head_dim)}
        for layer in model.hidden_layers
    ]
    expected_d_in = [int(value) for value in model.manifest["d_in_profile"]]
    if (
        verified_fields.get("model_arch_id") != model.model_arch_id
        or verified_fields.get("model_param_id") != model.model_param_id
        or verified_fields.get("quant_cfg_id") != model.quant_cfg_id
        or verified_fields.get("input_preprocess_id") != config.input_preprocess_id
        or verified_fields.get("L") != len(model.hidden_layers) + 1
        or verified_fields.get("hidden_profile") != expected_hidden
        or verified_fields.get("d_in_profile") != expected_d_in
        or verified_fields.get("K_out") != model.k_out
        or verified_fields.get("C") != model.class_count
    ):
        raise RuntimeError(
            f"{dataset_name}/{backend} independent verifier architecture identity does not match the loaded L3 model"
        )
    manifest = {
        "schema": "zkGPT_gat_run_manifest_v4",
        "status": verified_fields["status"],
        "dataset": verified_fields["dataset"],
        "dataset_display_name": dataset.name,
        "backend": verified_fields["backend"],
        "protocol_id": PROTOCOL_ID,
        "proof_backend_id": PROOF_BACKEND_ID,
        "model_arch_id": verified_fields["model_arch_id"],
        "model_param_id": verified_fields["model_param_id"],
        "quant_cfg_id": verified_fields["quant_cfg_id"],
        "input_preprocess_id": verified_fields["input_preprocess_id"],
        "L": verified_fields["L"],
        "hidden_profile": ",".join(
            f"{item['head_count']}x{item['head_dim']}" for item in verified_fields["hidden_profile"]
        ),
        "d_in_profile": ",".join(str(value) for value in verified_fields["d_in_profile"]),
        "K_out": verified_fields["K_out"],
        "C": verified_fields["C"],
        "config": relative_path(baseline, config_path),
        "artifact_path": relative_path(baseline, artifact_dir),
        "metadata_path": relative_path(baseline, artifact_dir / "metadata.json"),
        "transcript_path": relative_path(baseline, artifact_dir / "transcript.json"),
        "proof_payload_path": relative_path(baseline, artifact_dir / "proof_payload.json"),
        "verification_path": relative_path(baseline, artifact_dir / "verification.json"),
        "independent_verification_path": relative_path(baseline, independent_verification_path),
        "independent_verification_sha256": digest_file(independent_verification_path),
        "independent_verification_status": independent_verification["status"],
        "independent_report_digest": digest_json(independent_verification),
        "witness_manifest_path": relative_path(baseline, artifact_dir / "witness_manifest.json"),
        "N": graph.num_nodes,
        "E": int(graph.src.size),
        "G_batch": graph.graph_count,
        "node_ptr": graph.node_ptr.tolist(),
        "edge_ptr": graph.edge_ptr.tolist(),
        "full_dataset": verified_fields["full_dataset"],
        # The manifest's backend/evidence status is report-derived.  Detailed
        # CUDA evidence remains in the canonical artifact and is checked by
        # the independent verifier before this manifest is written.
        "gpu_backend": {
            "backend": verified_fields["backend"],
            "status": verified_fields["status"],
            "gpu_hot_path": verified_fields["gpu_hot_path"],
        },
        "proof_size_bytes_accounting": "exact canonical proof_payload.json byte length; public statement/setup and audit JSON are excluded",
        "proof_digest": digest_file(artifact_dir / "proof_payload.json"),
        "proof_size_bytes": benchmark["proof_size_bytes"],
        "timings_ms": {key: benchmark[key] for key in ("commitment_time_ms", "prove_time_ms", "verify_time_ms")},
        "proof_contribution": verified_fields["proof_contribution"],
        "proof_scope": verified_fields["proof_scope"],
        "full_gat_forward_proved": verified_fields["full_gat_forward_proved"],
        "final_logits_bound": verified_fields["final_logits_bound"],
        "privacy_extension_profile_id": verified_fields["privacy_extension_profile_id"],
        "privacy_profile_status": verified_fields["privacy_profile_status"],
        "verifier_reads_witness": verified_fields["verifier_reads_witness"],
        "witness_included_in_proof": verified_fields["witness_included_in_proof"],
        "public_output_digest": verified_fields["public_output_digest"],
        "public_y_schema": verified_fields.get("public_y_schema"),
        "public_y_path": verified_fields.get("public_y_path"),
        "public_y_digest": verified_fields.get("public_y_digest"),
        "public_y_descriptor_digest": verified_fields.get("public_y_descriptor_digest"),
        "public_y_shape": verified_fields.get("public_y_shape"),
        "X_pub_digest": verified_fields.get("X_pub_digest"),
        "graph_identity": verified_fields.get("graph_identity"),
        "com_W_identity": verified_fields.get("com_W_identity"),
        "reference_y_digest": verified_fields.get("reference_y_digest"),
        "reference_forward_profile_id": verified_fields.get("reference_forward_profile_id"),
        "reference_forward_engine": verified_fields.get("reference_forward_engine"),
        "reference_forward_source_sha256": verified_fields.get("reference_forward_source_sha256"),
        "public_output_binding_digest": verified_fields.get("public_output_binding_digest"),
        "registration_binding_digest": verified_fields.get("registration_binding_digest"),
        "gpu_hot_path": verified_fields["gpu_hot_path"],
        "proof_schema_scan": verified_fields["proof_schema_scan"],
        "privacy_masking_verified": verified_fields["privacy_masking_verified"],
        "independent_reference_forward": verified_fields["independent_reference_forward"],
        "verifier_input_boundary": verified_fields["verifier_input_boundary"],
        "artifact_digests": artifact_digest_map(artifact_dir, backend),
    }
    manifest["run_digest"] = manifest_run_digest(manifest)
    write_json(artifact_dir / "run_manifest.json", manifest, root=baseline)
    validate_artifact_inventory(
        artifact_dir, backend, require_independent=True, require_manifest=True
    )
    row = {
        "dataset": dataset_name,
        "backend": backend,
        "status": verified_fields["status"],
        "N": graph.num_nodes,
        "E": int(graph.src.size),
        "G_batch": graph.graph_count,
        "L": len(model.hidden_layers) + 1,
        "hidden_profile": ",".join(f"{layer.head_count}x{layer.head_dim}" for layer in model.hidden_layers),
        "d_in_profile": ",".join(str(item) for item in model.manifest["d_in_profile"]),
        "K_out": model.k_out,
        "C": model.class_count,
        "commitment_time_ms": benchmark["commitment_time_ms"],
        "prove_time_ms": benchmark["prove_time_ms"],
        "verify_time_ms": benchmark["verify_time_ms"],
        "proof_size_bytes": benchmark["proof_size_bytes"],
        "config_id": ids["config_id"]["id"],
        "model_id": ids["model_id"]["id"],
        "data_id": ids["data_id"]["id"],
        "artifact_path": relative_path(baseline, artifact_dir),
        "baseline_identity": verified_fields.get("baseline_identity", ""),
        "proof_scope": verified_fields["proof_scope"],
        "scope_mismatch": not bool(verified_fields["full_gat_forward_proved"]),
        "full_gat_forward_proved": verified_fields["full_gat_forward_proved"],
        "final_logits_bound": verified_fields["final_logits_bound"],
        "setup_assumption": verified_fields.get("setup_assumption", ""),
        "proof_size_definition": verified_fields.get("proof_size_definition", ""),
        "verifier_reads_witness": verified_fields["verifier_reads_witness"],
        "witness_included_in_proof": verified_fields["witness_included_in_proof"],
        "public_input_layout": verified_fields.get("public_input_layout", ""),
        "public_y_schema": verified_fields.get("public_y_schema", ""),
        "public_y_path": verified_fields.get("public_y_path", ""),
        "public_y_digest": verified_fields.get("public_y_digest", ""),
        "public_y_descriptor_digest": verified_fields.get("public_y_descriptor_digest", ""),
        "public_y_shape": str(verified_fields.get("public_y_shape", "")),
        "X_pub_digest": verified_fields.get("X_pub_digest", ""),
        "graph_identity": str(verified_fields.get("graph_identity", "")),
        "com_W_identity": str(verified_fields.get("com_W_identity", "")),
        "reference_y_digest": verified_fields.get("reference_y_digest", ""),
        "reference_forward_profile_id": verified_fields.get("reference_forward_profile_id", ""),
        "reference_forward_engine": verified_fields.get("reference_forward_engine", ""),
        "reference_forward_source_sha256": verified_fields.get("reference_forward_source_sha256", ""),
        "public_output_binding_digest": verified_fields.get("public_output_binding_digest", ""),
        "registration_binding_digest": verified_fields.get("registration_binding_digest", ""),
        "task_type": verified_fields.get("task_type", proof.metadata.get("task_type", "")),
        "model_arch": verified_fields.get("model_arch", ""),
        "model_params": verified_fields.get("model_params", ""),
        "model_registry_digest": verified_fields.get("model_registry_digest", ""),
        "same_checkpoint": verified_fields.get("same_checkpoint", ""),
        "same_quantization": "true" if same_quantization else "false",
        "same_graph_preprocessing": verified_fields.get("same_graph_preprocessing", ""),
        "same_self_loop_rule": verified_fields.get("same_self_loop_rule", ""),
        "same_edge_sort_rule": verified_fields.get("same_edge_sort_rule", ""),
        "same_public_output_policy": verified_fields.get("same_public_output_policy", ""),
        "self_loop_rule": verified_fields.get("self_loop_rule", ""),
        "edge_sort_rule": verified_fields.get("edge_sort_rule", ""),
        "batching_rule": verified_fields.get("batching_rule", ""),
        "hardware_and_device": verified_fields.get("hardware_and_device", ""),
        "gpu_hot_path_status": "VERIFY_OK" if verified_fields["gpu_hot_path"] else "NOT_APPLICABLE",
        "cpu_fallback": "false" if verified_fields["gpu_hot_path"] else "not_applicable",
        "run_status": verified_fields["status"],
        "failure_reason_if_not_comparable": verified_fields.get("failure_reason_if_not_comparable", ""),
    }
    return row


def write_summary(
    results_dir: pathlib.Path,
    rows: List[Dict[str, Any]],
    baseline_root: pathlib.Path | None = None,
) -> None:
    baseline = (
        pathlib.Path(baseline_root).expanduser().resolve()
        if baseline_root is not None
        else pathlib.Path(results_dir).expanduser().resolve().parent
    )
    results_dir = resolve_results_dir(baseline, results_dir)
    columns = [
        "dataset",
        "backend",
        "status",
        "N",
        "E",
        "G_batch",
        "L",
        "hidden_profile",
        "d_in_profile",
        "K_out",
        "C",
        "commitment_time_ms",
        "prove_time_ms",
        "verify_time_ms",
        "proof_size_bytes",
        "config_id",
        "model_id",
        "data_id",
        "artifact_path",
        "baseline_identity",
        "proof_scope",
        "scope_mismatch",
        "full_gat_forward_proved",
        "final_logits_bound",
        "setup_assumption",
        "proof_size_definition",
        "verifier_reads_witness",
        "witness_included_in_proof",
        "public_input_layout",
        "public_y_schema",
        "public_y_path",
        "public_y_digest",
        "public_y_descriptor_digest",
        "public_y_shape",
        "X_pub_digest",
        "graph_identity",
        "com_W_identity",
        "reference_y_digest",
        "reference_forward_profile_id",
        "reference_forward_engine",
        "reference_forward_source_sha256",
        "public_output_binding_digest",
        "registration_binding_digest",
        "task_type",
        "model_arch",
        "model_params",
        "model_registry_digest",
        "same_checkpoint",
        "same_quantization",
        "same_graph_preprocessing",
        "same_self_loop_rule",
        "same_edge_sort_rule",
        "same_public_output_policy",
        "self_loop_rule",
        "edge_sort_rule",
        "batching_rule",
        "hardware_and_device",
        "gpu_hot_path_status",
        "cpu_fallback",
        "run_status",
        "failure_reason_if_not_comparable",
    ]
    rows = sorted(rows, key=lambda row: (0 if row["backend"] == "cpu" else 1, DATASET_ORDER.index(row["dataset"])))
    ensure_dir(results_dir, root=baseline, label="results root")
    for backend in ["cpu", "gpu"]:
        csv_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(csv_buffer, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if row["backend"] == backend:
                writer.writerow({name: row.get(name, "") for name in columns})
        write_text_atomic(
            results_dir / f"{backend}_results.csv",
            csv_buffer.getvalue(),
            label=f"{backend} result CSV",
            root=baseline,
        )
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in columns})
    write_text_atomic(
        results_dir / "summary.csv",
        csv_buffer.getvalue(),
        label="summary CSV",
        root=baseline,
    )
    write_json(
        results_dir / "summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol_id": PROTOCOL_ID,
            "proof_backend_id": PROOF_BACKEND_ID,
            "rows": rows,
            "model_registration": "com_W is registered once per model/layout reuse key; Aux and L_qry are fresh per inference",
        },
        root=baseline,
    )
    readme = [
        "# GAT-ZKML-zkGPT Results",
        "",
        "This directory contains the official rerun independent zkGPT-style baseline results.",
        "Model registration artifacts live under `model_registry/`; per-inference Aux/L_qry openings remain fresh.",
        "",
        "Official tables:",
        "",
        "- `summary.csv` / `summary.json`",
        "- `cpu_results.csv`",
        "- `gpu_results.csv`",
        "",
        "Each dataset/backend directory includes metadata, full segment commitments, transcript, lookup proof, matrix sumcheck proof, GKR proof, complete masked Hyrax openings, canonical privacy profile, final proof payload, independent verification report, witness manifest, benchmark summary, and run manifest.",
        "",
        "`proof_size_bytes` is the canonical serialized `proof_payload.json` payload size, including verifier-facing proof messages. Witness manifests, raw witness materialization, datasets, checkpoints, logs, and audit metadata are not counted.",
        "",
    ]
    write_text_atomic(
        results_dir / "README_REPRODUCIBILITY.md",
        "\n".join(readme),
        label="result reproducibility Markdown",
        root=baseline,
    )


def run_many(
    datasets: List[str],
    backends: List[str],
    project_root: pathlib.Path,
    zk_root: pathlib.Path,
    results_dir: pathlib.Path,
    force: bool,
) -> List[Dict[str, Any]]:
    require_formal_experiment()
    baseline = _baseline_root(project_root, zk_root)
    results_dir = resolve_results_dir(baseline, results_dir)
    ensure_dir(
        resolve_local(baseline, "runs_tmp", "runs_tmp"),
        root=baseline,
        label="runs_tmp",
    )
    rows: List[Dict[str, Any]] = []
    for backend in backends:
        for dataset in datasets:
            print(f"[zkGPT] running {dataset}/{backend}", flush=True)
            row = run_one(dataset, backend, baseline, baseline, results_dir, force)
            rows.append(row)
            print(
                f"[zkGPT] {dataset}/{backend} VERIFY_OK "
                f"prove_ms={row['prove_time_ms']:.3f} verify_ms={row['verify_time_ms']:.3f} "
                f"proof_size_bytes={row['proof_size_bytes']}",
                flush=True,
            )
    write_summary(results_dir, rows, baseline)
    return rows
