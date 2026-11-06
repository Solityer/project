#!/usr/bin/env python3
"""Collect selected formal EZKL/Halo2 results or explicitly local-only checks.

Formal collection has one result per selected canonical dataset and no backend axis.
Explicit local-backend collection exists only for the tiny harness.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from formal_common import (
    BACKENDS,
    CANONICAL_OUTPUT_MODE,
    CANONICAL_PRIVACY,
    CANONICAL_PROOF_SCOPE,
    CANONICAL_RELATION,
    CANONICAL_STATEMENT_MODE,
    DATASETS,
    FORMAL_GPU_FEATURES,
    FORMAL_GPU_EXECUTION_POLICY,
    FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES,
    FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES,
    FORMAL_PHASE_BACKENDS,
    FORMAL_SESSION_ID,
    NATIVE_FORMAL_BACKEND,
    ROOT,
    ContractError,
    abort,
    canonical_json_digest,
    load_json,
    load_json_with_sha256,
    local_path,
    print_error,
    reject_forbidden_path,
    require_circuit_ref,
    require_digest,
    require_exact_keys,
    require_file,
    require_positive_timings,
    public_output_values,
    normalize_selected_datasets,
    require_formal_session_selection_binding,
    validate_compact_native_proof_v2,
    resource_manifest_hash,
    sha256,
    source_tree_hash,
    validate_artifact_identity,
    validate_canonical_public_inputs,
    validate_config,
    validate_native_proof_metrics,
    validate_native_pk_read_timing,
    validate_native_verify_metrics,
    validate_gpu_evidence,
    validate_gpu_evidence_document,
    validate_public_output,
    validate_public_statement,
    require_authenticated_formal_context,
    native_pk_v4_read_backing_geometry,
)
from compatible_resume import (
    BASELINE_MANIFEST_SHA256 as COMPAT_BASELINE_MANIFEST_SHA256,
    BASELINE_SESSION_ID as COMPAT_SESSION_ID,
    admission_semantics_digest,
    compare_admission_semantics,
    validate_compatibility_receipt,
    validate_compatible_resume_request,
    validate_original_artifact_manifest,
    validate_retained_witness_chain,
    validate_retained_setup_chain,
    validate_source_review,
)

COMPAT_OUTPUT_ROOT = Path("runs/formal_ppi_ogbn_20261029_001")
COMPAT_RECEIPT_RELATIVE = (
    COMPAT_OUTPUT_ROOT
    / ".sessions"
    / COMPAT_SESSION_ID
    / "compatibility"
    / "compatible_resume_receipt.json"
).as_posix()
COMPAT_RESULT_FIELDS = {
    "compatible_resume_receipt_path",
    "compatible_resume_receipt_sha256",
}

REQUIRED_RESULT_KEYS = {
    "schema_version",
    "status",
    "dataset",
    "proof_scope",
    "privacy_policy",
    "public_statement_mode",
    "output_mode",
    "relation_id",
    "quant_cfg_id",
    "static_tables_id",
    "public_statement_sha256",
    "public_output_sha256",
    "srs_sha256",
    "srs_provenance",
    "settings_sha256",
    "compiled_circuit_sha256",
    "vk_sha256",
    "pk_sha256",
    "proof_sha256",
    "proof_size_bytes",
    "binary_sha256",
    "source_tree_sha256",
    "resource_manifest_sha256",
    "verifier_reads_witness",
    "proof_contains_private_witness",
    "native_verify_result",
    "native_public_instance_binding",
    "phase_timings_ms",
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "commitment_metric",
    "verify_metric",
    "gpu_evidence",
    "circuit_ref",
    "proof_instances",
    "public_statement_path",
    "public_output_path",
    "srs_path",
    "settings_path",
    "compiled_circuit_path",
    "vk_path",
    "pk_path",
    "binary_path",
    "artifact_manifest_path",
    "artifact_manifest_sha256",
    "proof_path",
    "core_run_manifest_sha256",
}
PATH_DIGESTS = {
    "public_statement_path": "public_statement_sha256",
    "public_output_path": "public_output_sha256",
    "srs_path": "srs_sha256",
    "settings_path": "settings_sha256",
    "compiled_circuit_path": "compiled_circuit_sha256",
    "vk_path": "vk_sha256",
    "pk_path": "pk_sha256",
    "binary_path": "binary_sha256",
    "proof_path": "proof_sha256",
    "artifact_manifest_path": "artifact_manifest_sha256",
}
PAIR_FIELDS = (
    "relation_id",
    "proof_scope",
    "privacy_policy",
    "public_statement_mode",
    "output_mode",
    "quant_cfg_id",
    "static_tables_id",
    "public_statement_sha256",
    "public_output_sha256",
    "srs_sha256",
    "srs_provenance",
    "settings_sha256",
    "compiled_circuit_sha256",
    "vk_sha256",
    "source_tree_sha256",
    "resource_manifest_sha256",
    "circuit_ref",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--runs-root", default="runs/formal")
    result.add_argument("--output", default="results/formal_matrix.json")
    result.add_argument("--session-id", help="active formal session; required for formal collection")
    result.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=None,
        help="canonical dataset subset to collect; omitted means all five datasets",
    )
    result.add_argument("--local-backends", nargs="+", choices=BACKENDS, help="tiny/local-only results; cannot produce a formal result matrix")
    return result


def path_from_result(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        abort(f"result field {label} is not a path")
    path = local_path(value, label)
    reject_forbidden_path(path, label)
    return require_file(path, label)


def _load_compatible_resume_context(
    path: Path,
    result: dict[str, Any],
    *,
    session_id: str | None,
) -> dict[str, Any] | None:
    """Load the explicitly anchored PPI Setup-compatibility receipt, if requested."""
    supplied_fields = COMPAT_RESULT_FIELDS.intersection(result)
    if session_id != COMPAT_SESSION_ID:
        if supplied_fields:
            abort(f"{path} carries compatible-resume fields outside the exact baseline session")
        return None

    expected_root = (ROOT / COMPAT_OUTPUT_ROOT).resolve()
    result_root = path.parent.parent.resolve()
    if result_root != expected_root:
        if supplied_fields:
            abort(f"{path} carries compatible-resume fields outside the exact output root")
        return None

    session_path = expected_root / ".sessions" / f"{COMPAT_SESSION_ID}.json"
    if session_path.is_symlink() or not session_path.is_file():
        if supplied_fields:
            abort(f"{path} carries compatible-resume fields without an anchored session request")
        return None
    session_document = load_json(session_path, "compatible formal session")
    if not isinstance(session_document, dict):
        abort("compatible formal session is not a JSON object")
    if "compatible_resume_request" not in session_document:
        if supplied_fields:
            abort(f"{path} carries compatible-resume fields without an explicit session request")
        return None

    request, old_session = validate_compatible_resume_request(
        session_document,
        expected_session_id=COMPAT_SESSION_ID,
    )
    dataset = result.get("dataset")
    if dataset not in request.get("selected_datasets", []):
        abort(f"{path} is outside the immutable compatible-resume selection")
    expected_result_path = expected_root / str(dataset) / "result.json"
    if path.resolve() != expected_result_path.resolve():
        abort(f"{path} is not the exact selected compatible-resume result path")

    context: dict[str, Any] = {
        "session_path": session_path,
        "session": session_document,
        "request": request,
        "old_session": old_session,
        "receipt": None,
        "receipt_sha256": None,
    }
    if dataset != "ppi":
        if supplied_fields:
            abort(f"{path} must not carry PPI-only compatible-resume receipt fields")
        return context

    if supplied_fields != COMPAT_RESULT_FIELDS:
        missing = sorted(COMPAT_RESULT_FIELDS - supplied_fields)
        abort(f"{path} has an incomplete PPI compatible-resume receipt binding: missing={missing}")
    if result.get("compatible_resume_receipt_path") != COMPAT_RECEIPT_RELATIVE:
        abort(f"{path} compatible-resume receipt path is not the exact session sidecar")
    require_digest(result.get("compatible_resume_receipt_sha256"), f"{path} compatible-resume receipt digest")

    receipt_path = ROOT / COMPAT_RECEIPT_RELATIVE
    receipt, receipt_sha256 = validate_compatibility_receipt(
        receipt_path,
        session_id=COMPAT_SESSION_ID,
    )
    if receipt_sha256 != result["compatible_resume_receipt_sha256"]:
        abort(f"{path} compatible-resume receipt digest differs from the result binding")

    source_review_path = ROOT / request["source_review_path"]
    review, review_sha256, delta = validate_source_review(
        source_review_path,
        expected_session_id=COMPAT_SESSION_ID,
    )
    source_review_ref = receipt["source_review"]
    if (
        source_review_path.relative_to(ROOT).as_posix() != request["source_review_path"]
        or review_sha256 != request["source_review_sha256"]
        or review_sha256 != source_review_ref["sha256"]
        or source_review_ref["path"] != request["source_review_path"]
        or delta["diff_summary"]["combined_diff_sha256"] != request["source_delta_sha256"]
        or source_review_ref["source_delta_sha256"] != request["source_delta_sha256"]
        or source_review_ref["snapshot_index_inventory_sha256"] != request["snapshot_index_inventory_sha256"]
        or review.get("decision") != "APPROVED"
    ):
        abort("PPI compatible-resume receipt does not bind the approved source review and exact source delta")

    snapshot_path = ROOT / request["old_session_snapshot_path"]
    if (
        receipt["old_session_snapshot"]["path"] != request["old_session_snapshot_path"]
        or receipt["old_session_snapshot"]["sha256"] != request["old_session_snapshot_sha256"]
        or sha256(snapshot_path) != request["old_session_snapshot_sha256"]
    ):
        abort("PPI compatible-resume receipt does not bind the immutable original session snapshot")
    old_resume_identity = old_session.get("resume_identity")
    current_identity = session_document.get("compatible_prover_identity")
    if not isinstance(old_resume_identity, dict) or not isinstance(current_identity, dict):
        abort("PPI compatible-resume session is missing its frozen Setup or current Prover identity")
    old_resume_identity_sha256 = canonical_json_digest(old_resume_identity)
    old_setup_identity, old_setup_identity_sha256, setup_chain_refs = validate_retained_setup_chain(
        old_session,
        session_id=COMPAT_SESSION_ID,
    )
    current_identity_sha256 = canonical_json_digest(current_identity)
    current_identity_anchor = session_document.get("compatible_prover_identity_sha256")
    if (
        receipt["baseline_manifest_sha256"] != COMPAT_BASELINE_MANIFEST_SHA256
        or receipt["old_setup_identity_sha256"] != old_setup_identity_sha256
        or receipt["setup_chain"] != {
            key: value for key, value in setup_chain_refs.items() if key != "bound_setup_admission"
        }
        or receipt["bound_setup_admission"] != setup_chain_refs["bound_setup_admission"]
        or old_setup_identity.get("old_resume_identity_sha256") != old_resume_identity_sha256
        or receipt["current_prover_identity_sha256"] != current_identity_sha256
        or current_identity_anchor != current_identity_sha256
    ):
        abort("PPI compatible-resume receipt does not recompute from the frozen Setup/current Prover identities")

    old_admissions = old_session.get("native_admissions")
    old_ppi_admission = old_admissions.get("ppi") if isinstance(old_admissions, dict) else None
    bound_ref = receipt["bound_setup_admission"]
    if (
        not isinstance(old_ppi_admission, dict)
        or bound_ref.get("path") != old_ppi_admission.get("path")
        or bound_ref.get("sha256") != old_ppi_admission.get("sha256")
        or bound_ref.get("settings_sha256") != old_ppi_admission.get("settings_sha256")
    ):
        abort("PPI compatible-resume receipt does not bind the original session Admission anchor")

    context.update({
        "receipt": receipt,
        "receipt_sha256": receipt_sha256,
        "review": review,
        "reviewed_delta": delta,
        "old_identity": old_resume_identity,
        "old_identity_sha256": old_setup_identity_sha256,
        "old_resume_identity_sha256": old_resume_identity_sha256,
        "old_setup_identity": old_setup_identity,
        "setup_chain_refs": setup_chain_refs,
        "current_identity": current_identity,
        "current_identity_sha256": current_identity_sha256,
    })
    return context


def _repository_relative(path: Path, label: str) -> str:
    try:
        return path.resolve(strict=True).relative_to(ROOT.resolve(strict=True)).as_posix()
    except (OSError, ValueError):
        abort(f"{label} is outside the EZKL repository")


def _compatible_config_digest(identity: dict[str, Any], dataset: str) -> str:
    entries = identity.get("config_sha256s")
    if not isinstance(entries, list):
        abort(f"compatible {dataset} stage identity has no config digest map")
    matches = [item[1] for item in entries if isinstance(item, list) and len(item) == 2 and item[0] == dataset]
    if len(matches) != 1:
        abort(f"compatible {dataset} stage identity has no unique secure-config digest")
    require_digest(matches[0], f"compatible {dataset} secure-config digest")
    return matches[0]


def _validate_compatible_stage_receipt(
    anchor: Any,
    *,
    stage: str,
    expected_identity_sha256: str,
    expected_admission_sha256: str,
    expected_setup_receipt_sha256: str,
    expected_prove_budget_sha256: str | None,
    expected_inputs: dict[str, str],
    expected_outputs: dict[str, str],
    label: str,
    extra_keys: set[str] | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(anchor, dict)
        or set(anchor) != {"receipt", "sha256"}
        or not isinstance(anchor.get("receipt"), dict)
    ):
        abort(f"{label} {stage} stage anchor is malformed")
    stage_receipt = anchor["receipt"]
    expected_keys = {
        "schema", "session_id", "dataset", "stage", "resume_identity_sha256", "admission_sha256",
        "setup_receipt_sha256", "inputs_sha256", "outputs_sha256", "elapsed_ms", "recorded_at_utc",
    }
    if expected_prove_budget_sha256 is not None:
        expected_keys.add("native_prove_budget_sha256")
    if extra_keys:
        expected_keys.update(extra_keys)
    if set(stage_receipt) != expected_keys:
        abort(f"{label} {stage} stage receipt has unknown or missing fields")
    require_digest(anchor["sha256"], f"{label} {stage} stage receipt digest")
    if canonical_json_digest(stage_receipt) != anchor["sha256"]:
        abort(f"{label} {stage} stage receipt digest is stale")
    elapsed = stage_receipt.get("elapsed_ms")
    try:
        recorded_at = stage_receipt.get("recorded_at_utc")
        parsed_time = datetime.fromisoformat(recorded_at.replace("Z", "+00:00")) if isinstance(recorded_at, str) else None
    except ValueError:
        parsed_time = None
    if (
        stage_receipt.get("schema") != "ezkl.formal-stage-receipt.v1"
        or stage_receipt.get("session_id") != COMPAT_SESSION_ID
        or stage_receipt.get("dataset") != "ppi"
        or stage_receipt.get("stage") != stage
        or stage_receipt.get("resume_identity_sha256") != expected_identity_sha256
        or stage_receipt.get("admission_sha256") != expected_admission_sha256
        or stage_receipt.get("setup_receipt_sha256") != expected_setup_receipt_sha256
        or (expected_prove_budget_sha256 is not None and stage_receipt.get("native_prove_budget_sha256") != expected_prove_budget_sha256)
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed <= 0
        or parsed_time is None
        or parsed_time.tzinfo is None
    ):
        abort(f"{label} {stage} stage receipt has a mismatched identity, budget, or timing")
    for key, expected in (("inputs_sha256", expected_inputs), ("outputs_sha256", expected_outputs)):
        actual = stage_receipt.get(key)
        if (
            not isinstance(actual, dict)
            or set(actual) != set(expected)
            or any(not isinstance(value, str) or len(value) != 64 for value in actual.values())
            or actual != expected
        ):
            abort(f"{label} {stage} stage receipt {key} does not bind the exact stage artifacts")
        for name, value in actual.items():
            require_digest(value, f"{label} {stage} {key}.{name}")
    return stage_receipt


def _validate_retained_producer_final_chain(
    context: dict[str, Any],
    *,
    result_path: Path,
    original_manifest: dict[str, Any],
    retained_witness: dict[str, Any],
) -> None:
    """Check the frozen old PPI producer/final chain against its stage and manifest anchors."""
    old_session = context["old_session"]
    old_identity = context["old_identity"]
    old_resume_sha256 = context["old_resume_identity_sha256"]
    old_admissions = old_session.get("native_admissions")
    old_setup_map = old_session.get("setup_reuse_receipts")
    old_ppi_admission = old_admissions.get("ppi") if isinstance(old_admissions, dict) else None
    old_setup_sha256 = old_setup_map.get("ppi") if isinstance(old_setup_map, dict) else None
    if not isinstance(old_ppi_admission, dict) or not isinstance(old_setup_sha256, str):
        abort(f"{result_path} frozen PPI producer/final chain lacks old Admission or Setup anchors")
    require_digest(old_ppi_admission.get("sha256"), "frozen PPI Admission digest")
    require_digest(old_setup_sha256, "frozen PPI Setup receipt digest")

    files = original_manifest.get("files")
    records = {record.get("path"): record for record in files if isinstance(record, dict)} if isinstance(files, list) else {}
    if len(records) != len(files or []):
        abort(f"{result_path} frozen PPI manifest file map is malformed")
    old_output = ROOT / COMPAT_OUTPUT_ROOT / "ppi"
    stage_map = old_session.get("stage_receipts")
    ppi_stages = stage_map.get("ppi") if isinstance(stage_map, dict) else None
    current_stage_map = context["session"].get("stage_receipts")
    current_ppi_stages = current_stage_map.get("ppi") if isinstance(current_stage_map, dict) else None
    if (
        not isinstance(ppi_stages, dict)
        or set(ppi_stages) != {"producer", "final", "witness"}
        or current_ppi_stages != ppi_stages
    ):
        abort(f"{result_path} frozen PPI stage receipt map is absent")

    def anchored(stage: str) -> dict[str, Any]:
        anchor = ppi_stages.get(stage)
        if not isinstance(anchor, dict) or set(anchor) != {"receipt", "sha256"} or not isinstance(anchor.get("receipt"), dict):
            abort(f"{result_path} frozen PPI {stage} stage receipt is missing or malformed")
        if anchor.get("sha256") != canonical_json_digest(anchor["receipt"]):
            abort(f"{result_path} frozen PPI {stage} stage receipt digest is stale")
        return anchor

    def record_digest(name: str) -> str:
        record = records.get(name)
        digest = record.get("sha256") if isinstance(record, dict) else None
        require_digest(digest, f"frozen PPI manifest {name} digest")
        return digest

    producer = anchored("producer")
    producer_receipt = producer["receipt"]
    expected_producer_inputs = {
        "setup_context": record_digest("producer_context.json"),
        "settings": record_digest("settings.json"),
        "secure_config": _compatible_config_digest(old_identity, "ppi"),
        "resource_manifest": old_identity.get("resource_manifest_sha256"),
        "native_admission": old_ppi_admission.get("sha256"),
    }
    expected_producer_outputs = {
        "statement": record_digest("statement.json"),
        "public_output": record_digest("public_output.json"),
        "producer_manifest": record_digest("producer_manifest.json"),
    }
    for name, digest in {**expected_producer_inputs, **expected_producer_outputs}.items():
        require_digest(digest, f"frozen PPI producer {name} digest")
    _validate_compatible_stage_receipt(
        producer,
        stage="producer",
        expected_identity_sha256=old_resume_sha256,
        expected_admission_sha256=old_ppi_admission["sha256"],
        expected_setup_receipt_sha256=old_setup_sha256,
        expected_prove_budget_sha256=None,
        expected_inputs=expected_producer_inputs,
        expected_outputs=expected_producer_outputs,
        label="frozen old PPI",
    )

    final_anchor = anchored("final")
    final_receipt = final_anchor["receipt"]
    expected_final_inputs = {
        "statement": record_digest("statement.json"),
        "public_output": record_digest("public_output.json"),
        "producer_manifest": record_digest("producer_manifest.json"),
        "producer_context": record_digest("producer_context.json"),
        "settings": record_digest("settings.json"),
        "compiled_circuit": record_digest("compiled.circuit"),
        "verification_key": record_digest("vk.key"),
        "proving_key": record_digest("pk.key"),
        "secure_config": _compatible_config_digest(old_identity, "ppi"),
        "resource_manifest": old_identity.get("resource_manifest_sha256"),
        "native_prove_budget": retained_witness["old_native_prove_budget"]["sha256"],
        "admission": old_ppi_admission["sha256"],
    }
    for name, digest in expected_final_inputs.items():
        require_digest(digest, f"frozen PPI final {name} digest")
    final_log = old_output / "logs" / "final.log"
    if final_log.is_symlink() or not final_log.is_file():
        abort(f"{result_path} frozen PPI final log is missing or unsafe")
    projection = copy.deepcopy(original_manifest)
    projection["files"] = [
        record for record in original_manifest["files"]
        if record.get("path") not in {"proof.json", "native_pk_read_timing.json"}
    ]
    expected_final_outputs = {
        "artifact_manifest": canonical_json_digest(projection),
        "srs": record_digest("srs.kzg"),
        "final_log": sha256(final_log),
    }
    _validate_compatible_stage_receipt(
        final_anchor,
        stage="final",
        expected_identity_sha256=old_resume_sha256,
        expected_admission_sha256=old_ppi_admission["sha256"],
        expected_setup_receipt_sha256=old_setup_sha256,
        expected_prove_budget_sha256=None,
        expected_inputs=expected_final_inputs,
        expected_outputs=expected_final_outputs,
        label="frozen old PPI",
        extra_keys={"artifact_manifest_snapshot_sha256"},
    )
    require_digest(final_receipt.get("artifact_manifest_snapshot_sha256"), "frozen PPI final manifest snapshot digest")


def _validate_compatible_stage_receipts(
    context: dict[str, Any],
    *,
    result_path: Path,
    result: dict[str, Any],
    retained_witness: dict[str, Any],
    current_manifest_sha256: str,
) -> None:
    """Authenticate current PPI Prove and independent Verify against the immutable stage map."""
    receipt = context["receipt"]
    output = ROOT / COMPAT_OUTPUT_ROOT / "ppi"
    stage_map = context["session"].get("compatible_stage_receipts")
    expected_map_keys = {"schema", "session_id", "current_prover_identity_sha256", "ppi"}
    if (
        not isinstance(stage_map, dict)
        or set(stage_map) != expected_map_keys
        or stage_map.get("schema") != "ezkl.compatible-stage-receipts.v1"
        or stage_map.get("session_id") != COMPAT_SESSION_ID
        or stage_map.get("current_prover_identity_sha256") != context["current_identity_sha256"]
        or not isinstance(stage_map.get("ppi"), dict)
        or set(stage_map["ppi"]) != {"prove", "verify"}
    ):
        abort(f"{result_path} compatible session Prove/Verify receipt map is missing or malformed")
    current_prover = receipt["current_prover_receipts"]
    prove_budget_sha256 = current_prover["native_prove_budget"]["sha256"]
    bound_admission_sha256 = receipt["bound_setup_admission"]["sha256"]
    old_setup_sha256 = receipt["setup_chain"]["setup_receipt"]["sha256"]
    output_file = lambda name, label: _digest_compatible_output(output / name, label)

    expected_prove_inputs = {
        "witness": retained_witness["witness"]["sha256"],
        "compiled_circuit": receipt["shared_artifacts"]["compiled_circuit_sha256"],
        "proving_key": receipt["shared_artifacts"]["pk_sha256"],
        "settings": receipt["shared_artifacts"]["settings_sha256"],
        "srs": result["srs_sha256"],
        "native_prove_budget": prove_budget_sha256,
        "resource_manifest": result["resource_manifest_sha256"],
    }
    expected_prove_outputs = {
        "proof": result["proof_sha256"],
        "native_proof_metrics": output_file("native_proof_metrics.json", "PPI native proof metrics"),
        "native_pk_read_timing": output_file("native_pk_read_timing.json", "PPI native PK timing"),
        "gpu_evidence": output_file("gpu_evidence.json", "PPI GPU evidence"),
        "gpu_prover_record": output_file("gpu_prover_record.json", "PPI GPU prover record"),
        "artifact_manifest": current_manifest_sha256,
        "prove_log": output_file("logs/prove.log", "PPI Prove log"),
    }
    _validate_compatible_stage_receipt(
        stage_map["ppi"]["prove"],
        stage="prove",
        expected_identity_sha256=context["current_identity_sha256"],
        expected_admission_sha256=bound_admission_sha256,
        expected_setup_receipt_sha256=old_setup_sha256,
        expected_prove_budget_sha256=prove_budget_sha256,
        expected_inputs=expected_prove_inputs,
        expected_outputs=expected_prove_outputs,
        label="current PPI",
    )

    expected_verify_inputs = {
        "settings": receipt["shared_artifacts"]["settings_sha256"],
        "proof": result["proof_sha256"],
        "verification_key": receipt["shared_artifacts"]["vk_sha256"],
        "srs": result["srs_sha256"],
        "statement": receipt["shared_artifacts"]["statement_sha256"],
        "public_output": receipt["shared_artifacts"]["public_output_sha256"],
        "artifact_manifest": current_manifest_sha256,
        "resource_manifest": result["resource_manifest_sha256"],
    }
    expected_verify_outputs = {
        "native_verify_metrics": output_file("native_verify_metrics.json", "PPI native verify metrics"),
        "core_run_manifest": result["core_run_manifest_sha256"],
        "verify_log": result["independent_verifier_log_sha256"],
    }
    _validate_compatible_stage_receipt(
        stage_map["ppi"]["verify"],
        stage="verify",
        expected_identity_sha256=context["current_identity_sha256"],
        expected_admission_sha256=bound_admission_sha256,
        expected_setup_receipt_sha256=old_setup_sha256,
        expected_prove_budget_sha256=prove_budget_sha256,
        expected_inputs=expected_verify_inputs,
        expected_outputs=expected_verify_outputs,
        label="current PPI",
    )
    for stage, field, precision in (
        ("prove", "prove_command_wall_ms", 6),
        ("verify", "verify", 3),
    ):
        anchor = stage_map["ppi"][stage]
        elapsed = anchor["receipt"]["elapsed_ms"]
        timing = result.get(field) if stage == "prove" else result.get("phase_timings_ms", {}).get(field)
        if isinstance(timing, bool) or not isinstance(timing, (int, float)) or not math.isfinite(timing) or round(float(elapsed), precision) != timing:
            abort(f"{result_path} current PPI {stage} stage timing differs from the verified result")


def _digest_compatible_output(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        abort(f"{label} is missing or is not a regular file")
    return sha256(path)


def _validate_compatible_receipt_artifacts(
    context: dict[str, Any],
    *,
    result_path: Path,
    result: dict[str, Any],
    artifact_manifest: Path,
    artifact: dict[str, Any],
    artifact_records: dict[str, dict[str, Any]],
    setup_receipt: dict[str, Any],
    keygen_budget: dict[str, Any],
    keygen_receipt: dict[str, Any],
    prove_budget: dict[str, Any],
    receipt_digests: dict[str, str],
    admission: dict[str, Any],
    admission_path: Path,
) -> None:
    """Cross-bind the retained PPI Setup, current Prover, and current capacity sidecars."""
    receipt = context.get("receipt")
    if not isinstance(receipt, dict) or result.get("dataset") != "ppi":
        abort(f"{result_path} has no PPI-only compatible-resume receipt context")
    output_root = ROOT / COMPAT_OUTPUT_ROOT
    output = output_root / "ppi"
    old_session = context["old_session"]
    old_identity = context["old_identity"]
    current_identity = context["current_identity"]
    shared = receipt["shared_artifacts"]
    setup_chain = receipt["setup_chain"]
    setup_files = setup_chain["setup_files"]
    current_prover = receipt["current_prover_receipts"]

    original_manifest_ref = shared["original_artifact_manifest"]
    original_artifact, original_manifest_sha256, expected_original_manifest_ref = validate_original_artifact_manifest(
        old_session,
        session_id=COMPAT_SESSION_ID,
    )
    retained_witness = validate_retained_witness_chain(
        old_session,
        session_id=COMPAT_SESSION_ID,
    )
    if (
        original_manifest_ref != expected_original_manifest_ref
        or original_manifest_sha256 != expected_original_manifest_ref.get("sha256")
        or receipt["current_prover_receipts"].get("retained_witness") != retained_witness
    ):
        abort(f"{result_path} compatible receipt does not bind the frozen original manifest and retained Witness")
    stage_receipts = old_session.get("stage_receipts")
    ppi_stages = stage_receipts.get("ppi") if isinstance(stage_receipts, dict) else None
    final_stage_anchor = ppi_stages.get("final") if isinstance(ppi_stages, dict) else None
    final_stage_receipt = final_stage_anchor.get("receipt") if isinstance(final_stage_anchor, dict) else None
    if not isinstance(final_stage_receipt, dict):
        abort(f"{result_path} frozen original PPI final-stage receipt is malformed")

    current_manifest_relative = f"{COMPAT_OUTPUT_ROOT.as_posix()}/ppi/artifact_manifest.json"
    current_manifest_digest = sha256(artifact_manifest)
    receipt_copy_path = output / "compatible_resume_receipt.json"
    authoritative_receipt_path = ROOT / COMPAT_RECEIPT_RELATIVE
    if (
        _repository_relative(artifact_manifest, "PPI artifact manifest") != current_manifest_relative
        or current_manifest_digest != result["artifact_manifest_sha256"]
        or receipt_copy_path.is_symlink()
        or not receipt_copy_path.is_file()
        or authoritative_receipt_path.is_symlink()
        or not authoritative_receipt_path.is_file()
        or receipt_copy_path.read_bytes() != authoritative_receipt_path.read_bytes()
        or sha256(receipt_copy_path) != context["receipt_sha256"]
    ):
        abort(f"{result_path} current artifact manifest or local compatible-receipt copy is not bound to the authoritative session receipt")

    if (
        artifact.get("binary_sha256") != result["binary_sha256"]
        or artifact.get("source_tree_sha256") != result["source_tree_sha256"]
        or artifact.get("binary_sha256") != current_identity.get("binary_sha256")
        or artifact.get("source_tree_sha256") != current_identity.get("source_tree_sha256")
        or shared["statement_sha256"] != result["public_statement_sha256"]
        or shared["public_output_sha256"] != result["public_output_sha256"]
        or shared["settings_sha256"] != result["settings_sha256"]
        or shared["compiled_circuit_sha256"] != result["compiled_circuit_sha256"]
        or shared["vk_sha256"] != result["vk_sha256"]
        or shared["pk_sha256"] != result["pk_sha256"]
        or artifact.get("statement_sha256") != shared["statement_sha256"]
    ):
        abort(f"{result_path} compatible-resume receipt does not bind the current Prover and public/circuit/key identities")

    if (
        original_artifact.get("binary_sha256") != old_identity.get("binary_sha256")
        or original_artifact.get("source_tree_sha256") != old_identity.get("source_tree_sha256")
        or original_artifact.get("dataset") != "ppi"
        or original_artifact.get("backend") != NATIVE_FORMAL_BACKEND
    ):
        abort(f"{result_path} frozen original manifest does not match the original PPI Setup/producer identity")

    expected_current_artifact = copy.deepcopy(original_artifact)
    expected_current_artifact["binary_sha256"] = result["binary_sha256"]
    expected_current_artifact["source_tree_sha256"] = result["source_tree_sha256"]
    expected_records = expected_current_artifact["files"]
    original_budget_records = [
        record for record in expected_records
        if isinstance(record, dict) and record.get("path") == "native_prove_budget.json"
    ]
    if len(original_budget_records) != 1 or any(
        isinstance(record, dict) and record.get("path") == "compatible_resume_receipt.json"
        for record in expected_records
    ):
        abort(f"{result_path} frozen original manifest has an unexpected native prove-budget/compatibility-receipt inventory")
    budget_record = artifact_records.get("native_prove_budget.json")
    if not isinstance(budget_record, dict):
        abort(f"{result_path} current artifact manifest omits its native prove-budget record")
    original_budget_records[0]["size"] = budget_record.get("size")
    original_budget_records[0]["sha256"] = budget_record.get("sha256")
    expected_records.append({
        "path": "compatible_resume_receipt.json",
        "size": receipt_copy_path.stat().st_size,
        "sha256": context["receipt_sha256"],
    })
    proof_path = output / "proof.json"
    timing_path = output / "native_pk_read_timing.json"
    if (
        proof_path.is_symlink() or not proof_path.is_file()
        or timing_path.is_symlink() or not timing_path.is_file()
        or proof_path.stat().st_size != result.get("proof_size_bytes")
        or sha256(timing_path) != result.get("native_pk_read_timing_sha256")
    ):
        abort(f"{result_path} current Prove output files are not regular or do not match the verified result")
    expected_records.extend((
        {"path": "proof.json", "size": proof_path.stat().st_size, "sha256": result["proof_sha256"]},
        {"path": "native_pk_read_timing.json", "size": timing_path.stat().st_size, "sha256": result["native_pk_read_timing_sha256"]},
    ))
    if artifact != expected_current_artifact:
        abort(f"{result_path} current artifact manifest changes fields or file records outside the approved prove-budget/receipt/Prove-output transform")

    for filename, digest in (
        ("statement.json", shared["statement_sha256"]),
        ("public_output.json", shared["public_output_sha256"]),
        ("settings.json", shared["settings_sha256"]),
        ("compiled.circuit", shared["compiled_circuit_sha256"]),
        ("vk.key", shared["vk_sha256"]),
        ("pk.key", shared["pk_sha256"]),
        ("gat_static_tables.json", shared["static_tables_sha256"]),
        ("gat_integer_ledger.json", shared["integer_ledger_sha256"]),
    ):
        record = artifact_records.get(filename)
        if not isinstance(record, dict) or record.get("sha256") != digest:
            abort(f"{result_path} compatible-resume shared digest is not bound by artifact file {filename}")

    old_admissions = old_session.get("native_admissions")
    old_ppi_admission = old_admissions.get("ppi") if isinstance(old_admissions, dict) else None
    bound = receipt["bound_setup_admission"]
    current_ref = receipt["current_capacity_admission"]
    current_admission_path = ROOT / current_ref["path"]
    current_settings_path = current_admission_path.parent / "settings.json"
    if (
        not isinstance(old_ppi_admission, dict)
        or bound.get("path") != old_ppi_admission.get("path")
        or bound.get("sha256") != old_ppi_admission.get("sha256")
        or bound.get("settings_sha256") != old_ppi_admission.get("settings_sha256")
        or bound.get("path") != _repository_relative(admission_path, "bound PPI Setup Admission")
        or bound.get("sha256") != sha256(admission_path)
        or bound.get("settings_sha256") != result["settings_sha256"]
        or admission.get("settings_sha256") != bound.get("settings_sha256")
    ):
        abort(f"{result_path} retained PPI Setup Admission does not match its immutable session anchor")
    if (
        final_stage_receipt.get("admission_sha256") != bound["sha256"]
        or final_stage_receipt.get("setup_receipt_sha256") != receipt_digests["setup_receipt.json"]
    ):
        abort(f"{result_path} frozen original final-stage receipt does not bind the retained PPI Setup/Admission")
    if (
        current_admission_path.is_symlink()
        or not current_admission_path.is_file()
        or current_settings_path.is_symlink()
        or not current_settings_path.is_file()
        or sha256(current_admission_path) != current_ref["sha256"]
        or sha256(current_settings_path) != current_ref["settings_sha256"]
    ):
        abort(f"{result_path} current PPI capacity Admission/settings bytes differ from the receipt")
    current_admission = load_json(current_admission_path, "current PPI capacity Admission")
    if (
        current_admission.get("dataset") != "ppi"
        or current_admission.get("settings_sha256") != current_ref["settings_sha256"]
    ):
        abort(f"{result_path} current PPI capacity Admission identity is inconsistent")
    admission_semantics_sha256 = compare_admission_semantics(
        admission,
        current_admission,
        dataset="ppi",
        old_settings_path=output / "settings.json",
        current_settings_path=current_settings_path,
    )
    if receipt["admission_semantics_sha256"] != admission_semantics_sha256:
        abort(f"{result_path} PPI compatible Admission semantics do not match the receipt")

    setup_refs = setup_chain
    old_setup_anchor = old_session.get("setup_reuse_receipts", {}).get("ppi")
    if (
        setup_refs["setup_receipt"]["path"] != f"{COMPAT_OUTPUT_ROOT.as_posix()}/ppi/setup_receipt.json"
        or setup_refs["native_keygen_budget"]["path"] != f"{COMPAT_OUTPUT_ROOT.as_posix()}/ppi/native_keygen_budget.json"
        or setup_refs["native_keygen_receipt"]["path"] != f"{COMPAT_OUTPUT_ROOT.as_posix()}/ppi/native_keygen_receipt.json"
        or setup_refs["setup_receipt"]["sha256"] != receipt_digests["setup_receipt.json"]
        or setup_refs["native_keygen_budget"]["sha256"] != receipt_digests["native_keygen_budget.json"]
        or setup_refs["native_keygen_receipt"]["sha256"] != receipt_digests["native_keygen_receipt.json"]
        or old_setup_anchor != receipt_digests["setup_receipt.json"]
        or setup_chain["setup_identity_sha256"] != context["old_identity_sha256"]
        or setup_receipt.get("resume_identity_sha256") != context["old_resume_identity_sha256"]
        or setup_receipt.get("admission_sha256") != bound["sha256"]
        or setup_receipt.get("native_keygen_receipt_sha256") != receipt_digests["native_keygen_receipt.json"]
        or keygen_budget.get("admission_sha256") != bound["sha256"]
        or keygen_receipt.get("budget_receipt_sha256") != receipt_digests["native_keygen_budget.json"]
    ):
        abort(f"{result_path} retained PPI Setup/keygen chain is not bound to its frozen session anchors")

    old_identity_expectations = (
        ("binary_sha256", "binary_sha256"),
        ("build_receipt_sha256", "build_receipt_sha256"),
        ("build_environment_sha256", "build_environment_sha256"),
        ("resource_manifest_sha256", "resource_manifest_sha256"),
        ("settings_sha256", "settings_sha256"),
    )
    for budget_field, identity_field in old_identity_expectations:
        expected_value = old_identity.get(identity_field)
        if not isinstance(expected_value, str) or keygen_budget.get(budget_field) != expected_value:
            abort(f"{result_path} retained native keygen budget does not bind old Setup identity field {identity_field}")

    for name, filename in (
        ("settings_sha256", "settings.json"),
        ("compiled_circuit_sha256", "compiled.circuit"),
        ("vk_sha256", "vk.key"),
        ("pk_sha256", "pk.key"),
        ("static_tables_sha256", "gat_static_tables.json"),
        ("integer_ledger_sha256", "gat_integer_ledger.json"),
    ):
        recorded = setup_receipt.get("setup_files", {}).get(filename)
        if setup_files.get(name) != recorded or recorded != shared[name]:
            abort(f"{result_path} compatibility receipt does not preserve the old Setup {filename} digest")

    _validate_retained_producer_final_chain(
        context,
        result_path=result_path,
        original_manifest=original_artifact,
        retained_witness=retained_witness,
    )

    prove_ref = current_prover["native_prove_budget"]
    if (
        prove_ref["path"] != f"{COMPAT_OUTPUT_ROOT.as_posix()}/ppi/native_prove_budget.json"
        or prove_ref["sha256"] != receipt_digests["native_prove_budget.json"]
        or current_prover["binary_sha256"] != result["binary_sha256"]
        or current_prover["build_receipt_sha256"] != result.get("build_receipt_sha256")
        or current_prover["build_environment_sha256"] != result.get("build_environment_sha256")
        or prove_budget.get("admission_sha256") != bound["sha256"]
        or prove_budget.get("binary_sha256") != current_prover["binary_sha256"]
        or prove_budget.get("build_receipt_sha256") != current_prover["build_receipt_sha256"]
        or prove_budget.get("build_environment_sha256") != current_prover["build_environment_sha256"]
        or current_identity.get("binary_sha256") != current_prover["binary_sha256"]
        or current_identity.get("build_receipt_sha256") != current_prover["build_receipt_sha256"]
        or current_identity.get("build_environment_sha256") != current_prover["build_environment_sha256"]
    ):
        abort(f"{result_path} native prove budget is not bound to the old Admission and current Prover identity")

    _validate_compatible_stage_receipts(
        context,
        result_path=result_path,
        result=result,
        retained_witness=retained_witness,
        current_manifest_sha256=current_manifest_digest,
    )
    _validate_compatible_capacity_precheck(context, result_path=result_path, result=result)


def _validate_capacity_check(
    check: Any,
    *,
    dataset: str,
    phase: str,
    admission: dict[str, Any],
    expected_prove_budget_sha256: str | None = None,
) -> None:
    keys = {
        "dataset", "phase", "host_available_bytes", "host_required_known_bytes", "host_accounting",
        "host_os_diagnostic_reserve_bytes", "host_reserve_accounting", "host_unified_device_headroom_bytes",
        "msm_resident_cache_upper_bytes", "icicle_msm_precompute_factor", "icicle_msm_precomputed_basis_count",
        "icicle_msm_precomputed_basis_device_upper_bytes", "icicle_msm_point_staging_host_upper_bytes",
        "disk_available_bytes", "disk_required_known_bytes", "disk_budget_components", "disk_accounting",
        "cuda_free_bytes", "cuda_required_workspace_bytes", "cuda_total_bytes", "cuda_device_uuid",
        "integrated_device", "cuda_msm_workspace_bytes", "cuda_ntt_workspace_upper_bytes",
        "h_kernel_local_memory_upper_bytes", "h_program_metadata_upper_bytes", "coverage", "sampled_at_utc",
    }
    prepare_budget_keys = {
        "total_limit_bytes", "initial_available_bytes", "baseline_heap_bytes", "baseline_hot_mmap_bytes",
        "tablecache_bytes", "shplonk_extensions_bytes", "h_program_metadata_upper_bytes", "icicle_staging_bytes",
        "icicle_ntt_host_staging_upper_bytes", "msm_device_workspace_upper_bytes", "ntt_device_workspace_upper_bytes",
        "h_kernel_local_memory_upper_bytes", "icicle_msm_precompute_factor", "icicle_msm_precomputed_basis_count",
        "icicle_msm_precomputed_basis_device_upper_bytes", "icicle_msm_point_staging_host_upper_bytes",
        "integrated_device", "device_headroom_bytes", "reserve_bytes", "incremental_envelope_bytes",
        "future_phase_increment_bytes", "minimum_task_bytes", "taskcap_bytes", "icicle_msm_max_concurrency",
        "icicle_ntt_min_log_size", "icicle_ntt_max_log_size", "icicle_ntt_capacity_elements",
        "icicle_h_resident_device_upper_bytes", "icicle_ntt_device_workspace_upper_bytes",
        "icicle_ntt_domain_cache_upper_bytes", "icicle_ntt_init_scratch_upper_bytes",
        "icicle_ntt_device_peak_upper_bytes", "icicle_ntt_host_staging_upper_bytes",
    }
    budget_binding_keys = {"native_prove_budget_sha256", "native_prepare_budget"}
    expected_check_keys = keys if phase == "setup" else keys | budget_binding_keys
    if not isinstance(check, dict) or set(check) != expected_check_keys:
        abort(f"compatible capacity {dataset}/{phase} check has unknown or missing fields")
    if check.get("dataset") != dataset or check.get("phase") != phase:
        abort(f"compatible capacity {dataset}/{phase} check has a mismatched dataset/phase")

    def integer(value: Any, label: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize:
            abort(f"compatible capacity {dataset}/{phase} check has an invalid {label}")

    for name in keys - {"dataset", "phase", "host_accounting", "host_reserve_accounting", "disk_budget_components", "disk_accounting", "cuda_device_uuid", "integrated_device", "coverage", "sampled_at_utc"}:
        integer(check.get(name), name)
    host_accounting = check["host_accounting"]
    if not isinstance(host_accounting, dict) or set(host_accounting) != {"mem_available_bytes", "cgroup_limits", "effective_available_bytes"}:
        abort(f"compatible capacity {dataset}/{phase} host-accounting record is malformed")
    integer(host_accounting.get("mem_available_bytes"), "host MemAvailable")
    integer(host_accounting.get("effective_available_bytes"), "effective host availability")
    if not isinstance(host_accounting.get("cgroup_limits"), list):
        abort(f"compatible capacity {dataset}/{phase} cgroup memory limits are malformed")
    for item in host_accounting["cgroup_limits"]:
        if not isinstance(item, dict) or set(item) != {"path", "limit_bytes", "current_bytes", "available_bytes"}:
            abort(f"compatible capacity {dataset}/{phase} cgroup memory entry is malformed")
        if not isinstance(item.get("path"), str) or not item["path"]:
            abort(f"compatible capacity {dataset}/{phase} cgroup memory path is malformed")
        for name in ("limit_bytes", "current_bytes", "available_bytes"):
            integer(item.get(name), f"cgroup {name}")

    reserve = check["host_reserve_accounting"]
    if not isinstance(reserve, dict) or set(reserve) != {"host_mem_total_bytes", "effective_memory_limit_bytes", "cgroup_limits", "os_diagnostic_reserve_bytes"}:
        abort(f"compatible capacity {dataset}/{phase} host-reserve accounting is malformed")
    for name in ("host_mem_total_bytes", "effective_memory_limit_bytes", "os_diagnostic_reserve_bytes"):
        integer(reserve.get(name), f"host reserve {name}")
    if not isinstance(reserve.get("cgroup_limits"), list):
        abort(f"compatible capacity {dataset}/{phase} host-reserve cgroup limits are malformed")
    for item in reserve["cgroup_limits"]:
        if not isinstance(item, dict) or set(item) != {"path", "limit_bytes"} or not isinstance(item.get("path"), str) or not item["path"]:
            abort(f"compatible capacity {dataset}/{phase} host-reserve cgroup entry is malformed")
        integer(item.get("limit_bytes"), "host-reserve cgroup limit")

    disk = check["disk_accounting"]
    if not isinstance(disk, dict) or set(disk) != {
        "available_bytes", "device_id", "filesystem_type", "native_temp_path", "native_temp_filesystem_type",
        "native_temp_device_id", "native_temp_available_bytes",
    }:
        abort(f"compatible capacity {dataset}/{phase} disk accounting is malformed")
    for name in ("available_bytes", "device_id", "native_temp_device_id", "native_temp_available_bytes"):
        integer(disk.get(name), f"disk {name}")
    for name in ("filesystem_type", "native_temp_path", "native_temp_filesystem_type"):
        if not isinstance(disk.get(name), str) or not disk[name]:
            abort(f"compatible capacity {dataset}/{phase} disk {name} is malformed")
    if disk["available_bytes"] != check["disk_available_bytes"]:
        abort(f"compatible capacity {dataset}/{phase} disk accounting disagrees with its outer sample")

    components = check["disk_budget_components"]
    component_keys = {
        "future_setup_key_artifact_upper_bytes", "setup_mapped_peak_bytes", "proof_spill_known_bytes",
        "pending_witness_json_upper_bytes", "pending_proof_json_upper_bytes", "output_srs_copy_size_bytes",
        "pending_output_srs_copy_count", "pending_output_srs_copy_bytes", "proof_spill_extra_bytes",
        "pending_output_json_upper_bytes", "output_volume_required_bytes", "scratch_volume_required_bytes",
        "pending_output_srs_copy_datasets", "pending_output_json_datasets", "output_json_budget_kind",
        "total_required_bytes",
    }
    if not isinstance(components, dict) or set(components) != component_keys:
        abort(f"compatible capacity {dataset}/{phase} disk-budget component record is malformed")
    for name in component_keys - {"pending_output_srs_copy_datasets", "pending_output_json_datasets", "output_json_budget_kind"}:
        integer(components.get(name), f"disk component {name}")
    for name in ("pending_output_srs_copy_datasets", "pending_output_json_datasets"):
        value = components.get(name)
        if not isinstance(value, list) or any(not isinstance(item, str) or item not in DATASETS for item in value):
            abort(f"compatible capacity {dataset}/{phase} disk component {name} is malformed")
    if components.get("output_json_budget_kind") != "source_derived_future_persistent_json_upper_bound":
        abort(f"compatible capacity {dataset}/{phase} disk JSON budget kind is unsupported")
    if components["total_required_bytes"] != check["disk_required_known_bytes"]:
        abort(f"compatible capacity {dataset}/{phase} disk requirement differs from its authenticated components")

    memory = admission.get("native_memory_model")
    if not isinstance(memory, dict):
        abort(f"compatible capacity {dataset}/{phase} Admission has no native memory model")
    total = memory.get("cuda_total_memory_bytes")
    uuid = memory.get("cuda_uuid")
    integer(total, "Admission CUDA total memory")
    free = check["cuda_free_bytes"]
    if (
        not isinstance(uuid, str) or not uuid
        or check.get("cuda_device_uuid") != uuid
        or check.get("cuda_total_bytes") != total
        or free > total
        or check["cuda_required_workspace_bytes"] > free
        or check["host_required_known_bytes"] > check["host_available_bytes"]
        or check["host_os_diagnostic_reserve_bytes"] != reserve["os_diagnostic_reserve_bytes"]
        or host_accounting["effective_available_bytes"] != check["host_available_bytes"]
        or check.get("integrated_device") not in (True, False)
        or not isinstance(check.get("coverage"), str)
        or check.get("coverage") != "source-derived-known-allocation-gates; Prove includes one MSM, persistent NTT allocation envelope, fused H local-memory reserve, and compiled H metadata reserve"
    ):
        abort(f"compatible capacity {dataset}/{phase} sample does not pass its authenticated current device/host/disk gate")

    if budget_binding_keys.issubset(check):
        budget_digest = check.get("native_prove_budget_sha256")
        require_digest(budget_digest, f"compatible capacity {dataset}/{phase} native prove-budget digest")
        if expected_prove_budget_sha256 is None or budget_digest != expected_prove_budget_sha256:
            abort(f"compatible capacity {dataset}/{phase} gate is not bound to the current native prove budget")
        budget = check.get("native_prepare_budget")
        if (
            not isinstance(budget, dict)
            or set(budget) != prepare_budget_keys
            or not isinstance(budget.get("integrated_device"), bool)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize
                for name, value in budget.items()
                if name != "integrated_device"
            )
            or budget.get("total_limit_bytes", 0) <= 0
            or budget.get("initial_available_bytes") != check["host_available_bytes"]
            or budget.get("minimum_task_bytes", 0) <= 0
            or budget.get("taskcap_bytes", 0) < budget.get("minimum_task_bytes", 0)
            or budget.get("taskcap_bytes", 0) > budget.get("initial_available_bytes", 0)
            or budget.get("integrated_device") is not check.get("integrated_device")
            or budget.get("h_program_metadata_upper_bytes") != check["h_program_metadata_upper_bytes"]
            or budget.get("h_kernel_local_memory_upper_bytes") != check["h_kernel_local_memory_upper_bytes"]
            or budget.get("msm_device_workspace_upper_bytes") != check["cuda_msm_workspace_bytes"]
            or budget.get("ntt_device_workspace_upper_bytes") != check["cuda_ntt_workspace_upper_bytes"]
            or budget.get("icicle_msm_precompute_factor") != check["icicle_msm_precompute_factor"]
            or budget.get("icicle_msm_precomputed_basis_count") != check["icicle_msm_precomputed_basis_count"]
            or budget.get("icicle_msm_precomputed_basis_device_upper_bytes") != check["icicle_msm_precomputed_basis_device_upper_bytes"]
            or budget.get("icicle_msm_point_staging_host_upper_bytes") != check["icicle_msm_point_staging_host_upper_bytes"]
            or budget.get("icicle_ntt_device_peak_upper_bytes") != check["cuda_ntt_workspace_upper_bytes"]
            or budget.get("icicle_msm_max_concurrency") != 1
            or budget.get("icicle_ntt_min_log_size") != 16
            or budget.get("icicle_ntt_max_log_size") != 25
        ):
            abort(f"compatible capacity {dataset}/{phase} native prepare budget is malformed or inconsistent with its gate")
        domain_size = admission.get("domain_size")
        proof_domain = memory.get("proof_domain_size")
        constraint_system = admission.get("constraint_system")
        permutation_columns = constraint_system.get("num_permutation_columns") if isinstance(constraint_system, dict) else None
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (domain_size, proof_domain)
        ) or isinstance(permutation_columns, bool) or not isinstance(permutation_columns, int) or permutation_columns < 0:
            abort(f"compatible capacity {dataset}/{phase} Admission has malformed prepare-budget geometry")
        ntt_capacity = min(max(domain_size, proof_domain), 1 << 25)
        h_resident = 32 * ntt_capacity * (4 if permutation_columns > 0 else 1)
        expected_ntt = {
            "icicle_ntt_capacity_elements": ntt_capacity,
            "icicle_h_resident_device_upper_bytes": h_resident,
            "icicle_ntt_device_workspace_upper_bytes": 96 * ntt_capacity,
            "icicle_ntt_domain_cache_upper_bytes": 32 * (ntt_capacity + 1 + 64 + 6),
            "icicle_ntt_init_scratch_upper_bytes": 4 * 64 * 32,
            "icicle_ntt_device_peak_upper_bytes": 128 * ntt_capacity + 10464,
            "icicle_ntt_host_staging_upper_bytes": 32 * ntt_capacity,
            "icicle_msm_precomputed_basis_device_upper_bytes": domain_size * 64 * 4,
            "icicle_msm_point_staging_host_upper_bytes": min(domain_size, 1 << 20) * 64,
        }
        if any(budget.get(name) != expected for name, expected in expected_ntt.items()):
            abort(f"compatible capacity {dataset}/{phase} native prepare budget differs from its authenticated Admission geometry")
        expected_device_headroom = (
            budget["msm_device_workspace_upper_bytes"]
            + budget["ntt_device_workspace_upper_bytes"]
            + budget["icicle_msm_precomputed_basis_device_upper_bytes"]
            + budget["icicle_h_resident_device_upper_bytes"]
            + budget["h_kernel_local_memory_upper_bytes"]
            if budget["integrated_device"] else 0
        )
        expected_incremental = max(
            0,
            budget["initial_available_bytes"]
            - budget["baseline_heap_bytes"]
            - budget["baseline_hot_mmap_bytes"]
            - expected_device_headroom
            - budget["reserve_bytes"],
        )
        expected_taskcap = max(
            0,
            budget["initial_available_bytes"]
            - budget["baseline_heap_bytes"]
            - budget["baseline_hot_mmap_bytes"]
            - budget["tablecache_bytes"]
            - budget["h_program_metadata_upper_bytes"]
            - budget["shplonk_extensions_bytes"]
            - budget["icicle_staging_bytes"]
            - budget["icicle_ntt_host_staging_upper_bytes"]
            - expected_device_headroom
            - budget["reserve_bytes"],
        )
        expected_future_increment = (
            budget["tablecache_bytes"]
            + budget["h_program_metadata_upper_bytes"]
            + budget["shplonk_extensions_bytes"]
            + budget["icicle_staging_bytes"]
            + budget["icicle_ntt_host_staging_upper_bytes"]
            + budget["taskcap_bytes"]
        )
        if (
            budget["total_limit_bytes"] < budget["initial_available_bytes"]
            or budget["h_kernel_local_memory_upper_bytes"] != FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES
            or budget["h_program_metadata_upper_bytes"] != FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES
            or budget["icicle_msm_precompute_factor"] != 4
            or budget["icicle_msm_precomputed_basis_count"] != 1
            or budget["icicle_msm_max_concurrency"] != 1
            or budget["icicle_ntt_min_log_size"] != 16
            or budget["icicle_ntt_max_log_size"] != 25
            or budget["device_headroom_bytes"] != expected_device_headroom
            or budget["incremental_envelope_bytes"] != expected_incremental
            or budget["taskcap_bytes"] != expected_taskcap
            or budget["future_phase_increment_bytes"] != expected_future_increment
            or budget["reserve_bytes"] != check["host_os_diagnostic_reserve_bytes"]
        ):
            abort(f"compatible capacity {dataset}/{phase} native prepare budget does not satisfy its source-derived formulas")
    try:
        if not isinstance(check.get("sampled_at_utc"), str) or datetime.fromisoformat(check["sampled_at_utc"].replace("Z", "+00:00")).tzinfo is None:
            raise ValueError
    except ValueError:
        abort(f"compatible capacity {dataset}/{phase} timestamp is malformed")


def _validate_compatible_capacity_precheck(
    context: dict[str, Any],
    *,
    result_path: Path,
    result: dict[str, Any],
    require_ogbn_gate: bool = False,
) -> None:
    receipt = context["receipt"]
    output_root = ROOT / COMPAT_OUTPUT_ROOT
    session_id = COMPAT_SESSION_ID
    capacity_ref = receipt["capacity_precheck"]
    report_path = ROOT / capacity_ref["path"]
    recheck_path = ROOT / capacity_ref["recheck_path"]
    expected_report_relative = (
        COMPAT_OUTPUT_ROOT / ".sessions" / session_id / "compatibility" / "capacity_precheck_ppi.json"
    ).as_posix()
    if capacity_ref.get("path") != expected_report_relative:
        abort(f"{result_path} compatible capacity report does not use the immutable PPI-boundary snapshot path")
    for candidate, expected_digest, label in (
        (report_path, capacity_ref["sha256"], "capacity precheck"),
        (recheck_path, capacity_ref["recheck_sha256"], "capacity recheck snapshot"),
    ):
        if candidate.is_symlink() or not candidate.is_file() or sha256(candidate) != expected_digest:
            abort(f"{result_path} compatible {label} is missing, symlinked, or has a stale digest")

    report = load_json(report_path, "compatible current-capacity report")
    report_keys = {
        "schema", "report_kind", "baseline", "session_id", "sampled_at_utc", "selected_datasets", "gate_status",
        "readiness", "source_identity", "current_provenance", "actual_current_hardware_sample", "disk_floor_gate",
        "datasets", "source_model_estimate_scope", "scope",
    }
    if (
        not isinstance(report, dict) or set(report) != report_keys
        or report.get("schema") != "ezkl.resource_capacity_precheck.v2"
        or report.get("report_kind") != "current_fresh_admission_pre_setup"
        or report.get("baseline") != "ezkl"
        or report.get("session_id") != session_id
        or report.get("selected_datasets") != ["ppi", "ogbn_arxiv"]
        or report.get("gate_status") not in {"current_disk_floor_pass", "current_v4_pk_payload_lower_bound_unavailable"}
        or report.get("readiness") != "NOT_ESTABLISHED"
        or report.get("scope") != "Admission geometry is authenticated; source-model Setup upper and actual-census Prove gates remain mandatory and this report never establishes run readiness"
    ):
        abort(f"{result_path} compatible capacity report is incomplete or makes an unsupported readiness claim")

    try:
        if not isinstance(report.get("sampled_at_utc"), str) or datetime.fromisoformat(report["sampled_at_utc"].replace("Z", "+00:00")).tzinfo is None:
            raise ValueError
    except ValueError:
        abort(f"{result_path} compatible capacity report timestamp is malformed")

    provenance = report["current_provenance"]
    provenance_keys = {
        "capacity_source_identity", "resume_identity_sha256", "build_receipt_sha256", "binary_path", "binary_sha256",
        "source_tree_sha256", "resource_manifest_sha256", "config_sha256s", "authenticated_formal_context",
    }
    current_identity = context["current_identity"]
    if (
        not isinstance(provenance, dict) or set(provenance) != provenance_keys
        or provenance.get("capacity_source_identity") != report.get("source_identity")
        or provenance.get("resume_identity_sha256") != context["current_identity_sha256"]
        or provenance.get("binary_sha256") != result["binary_sha256"]
        or provenance.get("build_receipt_sha256") != result.get("build_receipt_sha256")
        or provenance.get("source_tree_sha256") != result["source_tree_sha256"]
        or provenance.get("resource_manifest_sha256") != result["resource_manifest_sha256"]
        or provenance.get("binary_path") != current_identity.get("binary_path")
    ):
        abort(f"{result_path} compatible capacity report does not bind the current Prover identity")
    require_digest(capacity_ref["capacity_source_identity_sha256"], "compatible capacity source identity digest")
    capacity_source_digest = canonical_json_digest(report["source_identity"])
    if capacity_source_digest != capacity_ref["capacity_source_identity_sha256"]:
        abort(f"{result_path} compatible capacity report source identity digest differs from its receipt")
    try:
        from resource_capacity_precheck import capacity_source_identity

        if capacity_source_identity() != report["source_identity"]:
            abort(f"{result_path} compatible capacity report source identity is stale")
    except ImportError as exc:
        abort(f"compatible current capacity source identity validator is unavailable: {exc}")

    rechecks = load_json(recheck_path, "compatible current-capacity recheck snapshot")
    recheck_keys = {
        "schema", "session_id", "selected_datasets", "current_prover_identity_sha256", "capacity_source_identity_sha256",
        "current_admissions", "capacity_checks_prefix", "checks",
    }
    if (
        not isinstance(rechecks, dict) or set(rechecks) != recheck_keys
        or rechecks.get("schema") != "ezkl.compatible-current-capacity-rechecks.v1"
        or rechecks.get("session_id") != session_id
        or rechecks.get("selected_datasets") != ["ppi", "ogbn_arxiv"]
        or rechecks.get("current_prover_identity_sha256") != context["current_identity_sha256"]
        or rechecks.get("capacity_source_identity_sha256") != capacity_source_digest
    ):
        abort(f"{result_path} compatible current-capacity recheck snapshot is not bound to the report and Prover")
    admissions = rechecks.get("current_admissions")
    if not isinstance(admissions, dict) or set(admissions) != {"ppi", "ogbn_arxiv"}:
        abort(f"{result_path} compatible current capacity Admission map is malformed")
    session_admissions = context["session"].get("compatible_native_admissions")
    if not isinstance(session_admissions, dict) or set(session_admissions) != {"ppi", "ogbn_arxiv"}:
        abort(f"{result_path} compatible session Admission anchors do not match the exact selection")
    session_admission_receipt_keys = {
        "path", "sha256", "prepared_dir", "settings_sha256", "scalar_domain_vector_bytes", "total_assignments",
        "logrows", "domain_size", "constraint_system", "native_memory_model", "relation_identity",
    }

    report_rows = report.get("datasets")
    if not isinstance(report_rows, list) or [row.get("dataset") for row in report_rows if isinstance(row, dict)] != ["ppi", "ogbn_arxiv"] or len(report_rows) != 2:
        abort(f"{result_path} compatible capacity report rows do not match the fixed selection")
    rows_by_dataset = {row["dataset"]: row for row in report_rows}
    session_directory = output_root / ".sessions" / session_id
    sidecar_root = session_directory / "compatibility" / "admission"
    old_admissions = context["old_session"].get("native_admissions")
    if not isinstance(old_admissions, dict) or set(old_admissions) != {"ppi", "ogbn_arxiv"}:
        abort(f"{result_path} frozen original Admission map does not match the exact selection")
    loaded_admissions: dict[str, dict[str, Any]] = {}
    for dataset in ("ppi", "ogbn_arxiv"):
        reference = admissions.get(dataset)
        expected_admission_path = sidecar_root / dataset / "admission.json"
        expected_admission_rel = expected_admission_path.relative_to(ROOT).as_posix()
        if (
            not isinstance(reference, dict)
            or set(reference) != {"path", "sha256", "settings_sha256"}
            or reference.get("path") != expected_admission_rel
        ):
            abort(f"{result_path} compatible current {dataset} Admission reference is malformed")
        current_admission_path = ROOT / reference["path"]
        settings_path = current_admission_path.parent / "settings.json"
        if (
            current_admission_path.is_symlink() or not current_admission_path.is_file()
            or settings_path.is_symlink() or not settings_path.is_file()
            or sha256(current_admission_path) != reference["sha256"]
            or sha256(settings_path) != reference["settings_sha256"]
        ):
            abort(f"{result_path} compatible current {dataset} Admission/settings bytes differ from the recheck snapshot")
        current_admission = load_json(current_admission_path, f"compatible current {dataset} Admission")
        if (
            not isinstance(current_admission, dict)
            or current_admission.get("dataset") != dataset
            or current_admission.get("settings_sha256") != reference["settings_sha256"]
        ):
            abort(f"{result_path} compatible current {dataset} Admission identity is malformed")
        session_reference = session_admissions[dataset]
        if (
            not isinstance(session_reference, dict)
            or set(session_reference) != session_admission_receipt_keys
            or {
                name: session_reference.get(name)
                for name in ("path", "sha256", "settings_sha256")
            } != reference
            or session_reference.get("prepared_dir") != expected_admission_path.parent.relative_to(ROOT).as_posix()
            or any(
                session_reference.get(name) != current_admission.get(name)
                for name in (
                    "scalar_domain_vector_bytes", "total_assignments", "logrows", "domain_size",
                    "constraint_system", "native_memory_model", "relation_identity",
                )
            )
        ):
            abort(f"{result_path} compatible full {dataset} Admission receipt differs from its exact recheck/path/content anchors")
        old_reference = old_admissions[dataset]
        expected_old_admission_path = session_directory / "admission" / dataset / "admission.json"
        expected_old_admission_rel = expected_old_admission_path.relative_to(ROOT).as_posix()
        if (
            not isinstance(old_reference, dict)
            or set(old_reference) != session_admission_receipt_keys
            or old_reference.get("path") != expected_old_admission_rel
            or old_reference.get("settings_sha256") is None
            or old_reference.get("sha256") is None
            or old_reference.get("prepared_dir") != expected_old_admission_path.parent.relative_to(ROOT).as_posix()
        ):
            abort(f"{result_path} frozen original {dataset} Admission anchor is malformed")
        old_admission_path = ROOT / expected_old_admission_rel
        old_settings_path = old_admission_path.parent / "settings.json"
        if (
            old_admission_path.is_symlink() or not old_admission_path.is_file()
            or old_settings_path.is_symlink() or not old_settings_path.is_file()
            or sha256(old_admission_path) != old_reference["sha256"]
            or sha256(old_settings_path) != old_reference["settings_sha256"]
        ):
            abort(f"{result_path} frozen original {dataset} Admission/settings bytes differ from their session anchors")
        old_admission = load_json(old_admission_path, f"frozen original {dataset} Admission")
        if (
            not isinstance(old_admission, dict)
            or old_admission.get("dataset") != dataset
            or old_admission.get("settings_sha256") != old_reference["settings_sha256"]
            or any(
                old_reference.get(name) != old_admission.get(name)
                for name in (
                    "scalar_domain_vector_bytes", "total_assignments", "logrows", "domain_size",
                    "constraint_system", "native_memory_model", "relation_identity",
                )
            )
        ):
            abort(f"{result_path} frozen original {dataset} Admission identity is malformed")
        compare_admission_semantics(
            old_admission,
            current_admission,
            dataset=dataset,
            old_settings_path=old_settings_path,
            current_settings_path=settings_path,
        )
        row = rows_by_dataset[dataset]
        if (
            row.get("admission_status") != "authenticated_current_session_admission"
            or row.get("admission_sha256") != reference["sha256"]
            or row.get("settings_sha256") != reference["settings_sha256"]
            or row.get("setup_already_retained") is not (dataset == "ppi")
        ):
            abort(f"{result_path} compatible capacity report row does not bind the current {dataset} Admission")
        sample = row.get("admission_device_sample")
        memory = current_admission.get("native_memory_model")
        if (
            not isinstance(sample, dict)
            or set(sample) != {"cuda_free_bytes_at_admission", "cuda_total_memory_bytes", "cuda_uuid", "evidence_kind"}
            or not isinstance(memory, dict)
            or sample.get("cuda_free_bytes_at_admission") != memory.get("cuda_free_bytes_at_admission")
            or sample.get("cuda_total_memory_bytes") != memory.get("cuda_total_memory_bytes")
            or sample.get("cuda_uuid") != memory.get("cuda_uuid")
            or sample.get("evidence_kind") != "sampled by native Admission, not by this precheck"
        ):
            abort(f"{result_path} compatible capacity report row has a mismatched current {dataset} device sample")
        loaded_admissions[dataset] = current_admission

    ppi_capacity_ref = receipt["current_capacity_admission"]
    if admissions["ppi"] != ppi_capacity_ref:
        abort(f"{result_path} compatible receipt and recheck snapshot disagree on the current PPI Admission")
    ppi_output_admission = output_root / "ppi" / "admission.json"
    ppi_output_settings = output_root / "ppi" / "settings.json"
    if (
        ppi_output_admission.is_symlink() or not ppi_output_admission.is_file()
        or sha256(ppi_output_admission) != receipt["bound_setup_admission"]["sha256"]
    ):
        abort(f"{result_path} original PPI Admission bytes were not preserved in output/ppi/admission.json")
    ppi_admission_semantics_sha256 = compare_admission_semantics(
        load_json(ppi_output_admission, "retained PPI Setup Admission"),
        loaded_admissions["ppi"],
        dataset="ppi",
        old_settings_path=ppi_output_settings,
        current_settings_path=(ROOT / admissions["ppi"]["path"]).parent / "settings.json",
    )
    if ppi_admission_semantics_sha256 != receipt["admission_semantics_sha256"]:
        abort(f"{result_path} current PPI Admission does not match the receipt's semantic digest")

    ogbn_output = output_root / "ogbn_arxiv"
    ogbn_output_admission = ogbn_output / "admission.json"
    ogbn_output_settings = ogbn_output / "settings.json"
    if (
        ogbn_output_admission.is_symlink() or not ogbn_output_admission.is_file()
        or ogbn_output_settings.is_symlink() or not ogbn_output_settings.is_file()
        or sha256(ogbn_output_admission) != admissions["ogbn_arxiv"]["sha256"]
        or sha256(ogbn_output_settings) != admissions["ogbn_arxiv"]["settings_sha256"]
    ):
        abort(f"{result_path} current OGBN Setup Admission does not match its current capacity sidecar")

    checks = rechecks.get("checks")
    if not isinstance(checks, list) or len(checks) != 1:
        abort(f"{result_path} compatible PPI capacity recheck must contain exactly one PPI gate")
    ppi_entry = checks[0]
    if (
        not isinstance(ppi_entry, dict)
        or set(ppi_entry) != {"dataset", "phase", "check", "check_sha256"}
        or ppi_entry.get("dataset") != "ppi"
        or ppi_entry.get("phase") != "witness_prove_verify"
        or ppi_entry.get("check_sha256") != canonical_json_digest(ppi_entry.get("check"))
    ):
        abort(f"{result_path} compatible PPI capacity recheck has an unsupported dataset/phase/object digest")
    prove_budget_ref = receipt["current_prover_receipts"]["native_prove_budget"]
    _validate_capacity_check(
        ppi_entry["check"],
        dataset="ppi",
        phase="witness_prove_verify",
        admission=loaded_admissions["ppi"],
        expected_prove_budget_sha256=prove_budget_ref["sha256"],
    )

    old_checks = context["old_session"].get("capacity_checks", [])
    current_checks = context["session"].get("capacity_checks", [])
    prefix_anchor = rechecks.get("capacity_checks_prefix")
    if (
        not isinstance(old_checks, list)
        or not isinstance(current_checks, list)
        or current_checks[: len(old_checks)] != old_checks
        or not isinstance(prefix_anchor, dict)
        or set(prefix_anchor) != {"count", "sha256"}
        or isinstance(prefix_anchor.get("count"), bool)
        or not isinstance(prefix_anchor.get("count"), int)
        or prefix_anchor["count"] < len(old_checks)
        or prefix_anchor["count"] >= len(current_checks)
    ):
        abort(f"{result_path} compatible session does not preserve the exact original/PPI-boundary capacity-check prefix")
    require_digest(prefix_anchor.get("sha256"), "compatible capacity-check prefix digest")
    prefix_count = prefix_anchor["count"]
    if (
        canonical_json_digest(current_checks[:prefix_count]) != prefix_anchor["sha256"]
        or current_checks[prefix_count] != ppi_entry["check"]
    ):
        abort(f"{result_path} compatible session does not preserve the exact PPI-boundary capacity prefix and gate position")
    appended_checks = current_checks[prefix_count + 1 :]
    phase_counts: dict[str, dict[str, int]] = {"ogbn_arxiv": {"setup": 0, "witness_prove_verify": 0}}
    ogbn_wpv_seen = False
    for check in appended_checks:
        if not isinstance(check, dict):
            abort(f"{result_path} compatible session has a malformed appended capacity check")
        dataset, phase = check.get("dataset"), check.get("phase")
        if dataset != "ogbn_arxiv" or phase not in {"setup", "witness_prove_verify"}:
            abort(f"{result_path} compatible session has an unknown appended capacity dataset/phase")
        phase_counts[dataset][phase] += 1
        if phase_counts[dataset][phase] > 1:
            abort(f"{result_path} compatible session has a duplicate post-PPI OGBN {phase} capacity gate")
        if phase == "setup" and ogbn_wpv_seen:
            abort(f"{result_path} compatible OGBN Setup capacity gate occurs after its Prove gate")
        if phase == "witness_prove_verify":
            if phase_counts[dataset]["setup"] != 1:
                abort(f"{result_path} compatible OGBN Prove gate has no preceding current Setup gate")
            ogbn_wpv_seen = True
        expected_budget_sha256 = None
        if phase == "witness_prove_verify":
            budget_path = output_root / dataset / "native_prove_budget.json"
            if budget_path.is_symlink() or not budget_path.is_file():
                abort(f"{result_path} current OGBN native prove budget is missing for its session gate")
            expected_budget_sha256 = sha256(budget_path)
        _validate_capacity_check(
            check,
            dataset=dataset,
            phase=phase,
            admission=loaded_admissions[dataset],
            expected_prove_budget_sha256=expected_budget_sha256,
        )
    if (
        (require_ogbn_gate and phase_counts["ogbn_arxiv"]["witness_prove_verify"] != 1)
        or (require_ogbn_gate and phase_counts["ogbn_arxiv"]["setup"] != 1)
    ):
        abort(f"{result_path} completed compatible session lacks its unique post-PPI OGBN Setup/Prove gates")


def validate_proof_instances(
    result: dict[str, Any],
    proof: Path,
    *,
    statement: Path,
    public_output: Path,
    artifact_manifest: Path,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    expected = result["proof_instances"]
    if not isinstance(expected, dict):
        abort("result proof_instances must be an object")
    required = {
        "proof_field",
        "encoding",
        "sha256",
        "count",
        "public_statement_sha256",
        "public_output_sha256",
        "native_binding",
    }
    if not required.issubset(expected):
        abort(f"result proof_instances is missing fields: {sorted(required - set(expected))}")
    reference = validate_compact_native_proof_v2(
        proof,
        statement_path=statement,
        public_output_path=public_output,
        artifact_manifest_path=artifact_manifest,
        artifact=artifact,
    )
    if reference is not None:
        if expected["proof_field"] != "instances_ref" or expected["encoding"] != "canonical-json":
            abort("result proof_instances does not identify the compact native reference encoding")
        if (
            "column_counts" not in expected
            or expected["sha256"] != reference["sha256"]
            or expected["count"] != reference["count"]
            or expected["column_counts"] != reference["column_counts"]
        ):
            abort("compact proof instance digest/count/column shape mismatch")
    else:
        document = load_json(proof, "verifier-facing proof")
        instances = document.get("instances")
        if not isinstance(instances, list) or not instances or any(not isinstance(item, list) for item in instances):
            abort(f"{proof} does not expose a native instances array")
        count = sum(len(item) for item in instances)
        if count <= 0:
            abort(f"{proof} has no native public instances")
        digest = canonical_json_digest(instances)
        if expected["proof_field"] != "instances" or expected["encoding"] != "canonical-json":
            abort("result proof_instances encoding is not canonical native instances")
        if expected["sha256"] != digest or expected["count"] != count:
            abort("proof instance digest/count mismatch")
    if expected["public_statement_sha256"] != result["public_statement_sha256"]:
        abort("proof instances are not bound to the external public statement")
    if expected["public_output_sha256"] != result["public_output_sha256"]:
        abort("proof instances are not bound to external public output Y")
    if expected["native_binding"] is not True:
        abort("proof instances lack native public-instance binding")
    return expected


def _validate_native_ntt_profile(
    path: Path,
    admission_path: Path,
    admission: dict[str, Any],
    prove_budget: dict[str, Any],
) -> dict[str, int]:
    memory = admission.get("native_memory_model")
    if not isinstance(memory, dict) or memory.get("schema") != "ezkl-halo2-static-envelope-v22":
        abort(f"{path} admission has an incompatible native NTT memory model")
    proof_domain = memory.get("proof_domain_size")
    scalar_bytes = memory.get("size_of_fr_bytes")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (proof_domain, scalar_bytes)):
        abort(f"{path} admission has malformed native NTT geometry")
    if scalar_bytes != 32 or proof_domain & (proof_domain - 1):
        abort(f"{path} admission native NTT geometry is not BN254 power-of-two Fr")
    base_domain = admission.get("domain_size")
    if isinstance(base_domain, bool) or not isinstance(base_domain, int) or base_domain <= 0:
        abort(f"{path} admission has malformed base-domain geometry")
    capacity = min(max(base_domain, proof_domain), 1 << 25)
    constraint_system = admission.get("constraint_system")
    if not isinstance(constraint_system, dict):
        abort(f"{path} admission has no constraint-system geometry for resident H capacity")
    permutation_columns = constraint_system.get("num_permutation_columns")
    if isinstance(permutation_columns, bool) or not isinstance(permutation_columns, int) or permutation_columns < 0:
        abort(f"{path} admission has malformed permutation-column geometry for resident H capacity")
    h_resident_upper = 32 * capacity * (4 if permutation_columns > 0 else 1)
    plan = {
        "icicle_ntt_min_log_size": 16,
        "icicle_ntt_max_log_size": 25,
        "icicle_ntt_capacity_elements": capacity,
        "icicle_h_resident_device_upper_bytes": h_resident_upper,
        "icicle_ntt_device_workspace_upper_bytes": 96 * capacity,
        "icicle_ntt_domain_cache_upper_bytes": 32 * (capacity + 1 + 64 + 6),
        "icicle_ntt_init_scratch_upper_bytes": 4 * 64 * 32,
        "icicle_ntt_device_peak_upper_bytes": 128 * capacity + 10464,
        "icicle_ntt_host_staging_upper_bytes": 32 * capacity,
        "icicle_msm_max_concurrency": 1,
        "icicle_msm_precompute_factor": 4,
        "icicle_msm_precomputed_basis_count": 1,
        "icicle_msm_precomputed_basis_device_upper_bytes": base_domain * 64 * 4,
        "icicle_msm_point_staging_host_upper_bytes": min(base_domain, 1 << 20) * 64,
    }
    for document_name, document in (("Admission", memory), ("native prove budget", prove_budget)):
        for key, expected in plan.items():
            observed = document.get(key)
            if isinstance(observed, bool) or not isinstance(observed, int) or observed != expected:
                abort(f"{path} {document_name} {key} does not match its authenticated source formula")
    if (
        prove_budget.get("schema") != "ezkl.native-prove-budget.v8"
        or prove_budget.get("native_execution_policy") != FORMAL_GPU_EXECUTION_POLICY
        or prove_budget.get("admission_sha256") != sha256(admission_path)
        or prove_budget.get("h_workspace_borrows_ntt") is not True
        or prove_budget.get("h_kernel_local_memory_upper_bytes") != FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES
        or prove_budget.get("h_program_metadata_upper_bytes") != FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES
    ):
        abort(f"{path} native prove budget does not bind the current GPU NTT Admission profile")
    if prove_budget.get("icicle_msm_workspace_query_schema") != "ezkl.icicle.cub_msm_workspace_query.v3":
        abort(f"{path} native prove budget does not bind the two-path CUB workspace schema")
    ordinary_peak = prove_budget.get("icicle_msm_ordinary_factor1_device_workspace_peak_bytes")
    prepared_peak = prove_budget.get("icicle_msm_prepared_factor4_device_workspace_peak_bytes")
    query_peak = prove_budget.get("device_workspace_peak_bytes")
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (ordinary_peak, prepared_peak, query_peak))
        or query_peak != max(ordinary_peak, prepared_peak)
    ):
        abort(f"{path} native prove budget has inconsistent CUB workspace path peaks")
    expected_cuda_required = (
        query_peak
        + plan["icicle_ntt_device_peak_upper_bytes"]
        + plan["icicle_msm_precomputed_basis_device_upper_bytes"]
        + plan["icicle_h_resident_device_upper_bytes"]
        + FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES
    )
    host_required = prove_budget.get("host_required_known_bytes")
    host_available = prove_budget.get("host_available_bytes")
    integrated_device = prove_budget.get("integrated_device")
    minimum_host_required = FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES + (
        expected_cuda_required if integrated_device is True else 0
    )
    if (
        prove_budget.get("cuda_required_workspace_bytes") != expected_cuda_required
        or isinstance(host_required, bool)
        or not isinstance(host_required, int)
        or isinstance(host_available, bool)
        or not isinstance(host_available, int)
        or not isinstance(integrated_device, bool)
        or host_required < minimum_host_required
        or host_required > host_available
    ):
        abort(f"{path} native prove budget omits the fused H CUDA or host metadata reserve")
    plan["h_kernel_local_memory_upper_bytes"] = FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES
    plan["h_program_metadata_upper_bytes"] = FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES
    plan["cuda_required_workspace_bytes"] = expected_cuda_required
    return plan


def _validate_shplonk_fixed_cache_budget(
    path: Path,
    admission: dict[str, Any],
    prove_budget: dict[str, Any],
) -> dict[str, int]:
    memory = admission.get("native_memory_model")
    if not isinstance(memory, dict) or memory.get("schema") != "ezkl-halo2-static-envelope-v22":
        abort(f"{path} admission has an incompatible SHPLONK cache memory model")
    cache_domain = admission.get("domain_size")
    cache_extended_domain = memory.get("proof_domain_size")
    cache_scalar = memory.get("size_of_fr_bytes")
    cache_group_layout = memory.get("size_of_shplonk_fixed_aggregate_cache_group_bytes")
    cache_usize = memory.get("size_of_usize_bytes")
    cache_vec_header = memory.get("size_of_vec_header_bytes")
    cache_sets = memory.get("shplonk_rotation_set_count")
    cache_fixed = memory.get("shplonk_max_commitments_per_rotation_set")
    cache_points = memory.get("shplonk_max_rotation_count")
    cache_source_values = (
        cache_domain,
        cache_extended_domain,
        cache_scalar,
        cache_group_layout,
        cache_usize,
        cache_vec_header,
        cache_sets,
        cache_fixed,
        cache_points,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > sys.maxsize
        for value in cache_source_values
    ):
        abort(
            f"{path} admission has malformed fixed-cache source bounds: "
            f"base_domain={cache_domain!r}, extended_proof_domain={cache_extended_domain!r}, "
            f"Fr_bytes={cache_scalar!r}, aggregate_group_bytes={cache_group_layout!r}"
        )

    cache_group_host = (
        cache_group_layout
        + cache_domain * cache_scalar
        + cache_fixed * cache_usize
        + cache_points * cache_scalar
        + 3 * 4096
    )
    cache_group_payload = cache_domain * cache_scalar
    cache_group_disk = ((cache_group_payload + 4095) // 4096) * 4096 + 4096
    cache_bank_header = cache_vec_header + 4096
    cache_static = {
        "shplonk_fixed_cache_group_host_bound_bytes": cache_group_host,
        "shplonk_fixed_cache_group_disk_bound_bytes": cache_group_disk,
        "shplonk_fixed_cache_bank_header_bytes": cache_bank_header,
        "shplonk_fixed_cache_domain_size": cache_domain,
        "shplonk_fixed_cache_scalar_bytes": cache_scalar,
        "shplonk_fixed_cache_max_fixed_sources_per_group": cache_fixed,
        "shplonk_fixed_cache_max_points_per_group": cache_points,
        "shplonk_fixed_cache_rotation_set_count": cache_sets,
    }
    cache_static_mismatches = {
        name: {"actual": prove_budget.get(name), "expected": value}
        for name, value in cache_static.items()
        if prove_budget.get(name) != value
    }
    if cache_static_mismatches:
        abort(
            f"{path} native prove budget SHPLONK cache shape does not match its admitted source geometry: "
            f"base_domain_expected={cache_domain}, extended_proof_domain={cache_extended_domain}, "
            f"Fr_bytes={cache_scalar}, aggregate_group_bytes={cache_group_layout}, "
            f"group_host_bytes_budget={prove_budget.get('shplonk_fixed_cache_group_host_bound_bytes')!r} "
            f"expected={cache_group_host}, "
            f"group_disk_bytes_budget={prove_budget.get('shplonk_fixed_cache_group_disk_bound_bytes')!r} "
            f"expected={cache_group_disk}, mismatches={cache_static_mismatches!r}"
        )

    for name in (
        "disk_available_bytes",
        "disk_required_known_bytes",
        "host_available_bytes",
        "host_required_known_bytes",
        "shplonk_fixed_cache_max_groups",
        "shplonk_fixed_cache_host_limit_bytes",
        "shplonk_fixed_cache_bank_header_limit_bytes",
        "shplonk_fixed_cache_disk_limit_bytes",
    ):
        value = prove_budget.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize:
            abort(f"{path} native prove budget has invalid {name}")

    expected_cache_groups = min(
        2,
        cache_sets,
        max(
            0,
            prove_budget["host_available_bytes"]
            - prove_budget["host_required_known_bytes"]
            - cache_bank_header,
        )
        // cache_group_host,
        max(
            0,
            prove_budget["disk_available_bytes"]
            - prove_budget["disk_required_known_bytes"],
        )
        // cache_group_disk,
    )
    expected_cache_host = cache_bank_header + expected_cache_groups * cache_group_host if expected_cache_groups else 0
    expected_cache_bank_header = cache_bank_header if expected_cache_groups else 0
    expected_cache_disk = expected_cache_groups * cache_group_disk
    cache_limit_expected = {
        "shplonk_fixed_cache_max_groups": expected_cache_groups,
        "shplonk_fixed_cache_host_limit_bytes": expected_cache_host,
        "shplonk_fixed_cache_bank_header_limit_bytes": expected_cache_bank_header,
        "shplonk_fixed_cache_disk_limit_bytes": expected_cache_disk,
    }
    cache_limit_mismatches = {
        name: {"actual": prove_budget.get(name), "expected": value}
        for name, value in cache_limit_expected.items()
        if prove_budget.get(name) != value
    }
    if cache_limit_mismatches:
        abort(
            f"{path} native prove budget SHPLONK cache limits do not match the admitted resource envelope: "
            f"base_domain={cache_domain}, extended_proof_domain={cache_extended_domain}, "
            f"Fr_bytes={cache_scalar}, aggregate_group_bytes={cache_group_layout}, "
            f"group_host_bytes={cache_group_host}, group_disk_bytes={cache_group_disk}, "
            f"host_available={prove_budget['host_available_bytes']}, "
            f"host_required={prove_budget['host_required_known_bytes']}, "
            f"disk_available={prove_budget['disk_available_bytes']}, "
            f"disk_required={prove_budget['disk_required_known_bytes']}, "
            f"mismatches={cache_limit_mismatches!r}"
        )
    return {
        "rotation_set_count": cache_sets,
        "group_host_bytes": cache_group_host,
        "group_disk_bytes": cache_group_disk,
        "bank_header_bytes": cache_bank_header,
    }


def require_positive_finite_duration(result: dict[str, Any], field: str, label: str) -> int | float:
    if field not in result:
        abort(f"{label} is missing {field}")
    value = result[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        abort(f"{label} {field} must be a positive finite duration")
    return value


def validate_cell(
    path: Path,
    *,
    expected_session_id: str | None = None,
    expected_source_tree_sha256: str | None = None,
    expected_resource_manifest_sha256: str | None = None,
    expected_config_values: dict[str, str] | None = None,
    expected_result_sha256: str | None = None,
) -> dict[str, Any]:
    if expected_result_sha256 is None:
        result = load_json(path, f"result cell {path}")
    else:
        require_digest(expected_result_sha256, f"{path} expected checkpoint digest")
        result, actual_result_sha256 = load_json_with_sha256(path, f"result cell {path}")
        if actual_result_sha256 != expected_result_sha256:
            abort(f"{path} result bytes differ from the active session checkpoint")
    formal_gpu_identity_fields = {
        "selected_cuda_device", "build_receipt_sha256", "build_environment_sha256",
        "gpu_prover_record_sha256", "independent_verifier_log_sha256",
    }
    formal_pk_timing_fields = {
        "native_pk_read_timing_sha256",
        "prove_command_wall_ms",
        "prove_wall_scope",
        "prove_command_residual_excluding_pk_artifact_read_and_sigma_prepare_ms",
    }
    unknown = (
        set(result)
        - REQUIRED_RESULT_KEYS
        - {"backend", "execution_mode", "phase_backends", "native_backend", "formal_session_id"}
        - formal_gpu_identity_fields
        - formal_pk_timing_fields
        - COMPAT_RESULT_FIELDS
    )
    if unknown:
        abort(f"{path} has unknown result fields: {sorted(unknown)}")
    missing = REQUIRED_RESULT_KEYS - set(result)
    if missing:
        abort(f"{path} is missing result fields: {sorted(missing)}")
    if result["schema_version"] != 1 or result["status"] != "VERIFY_OK":
        abort(f"{path} is not a verified native result")
    dataset = result["dataset"]
    formal = "formal_session_id" in result
    compatibility_context: dict[str, Any] | None = None
    if formal and not formal_pk_timing_fields.issubset(result):
        abort(f"{path} formal result omits its authenticated PK materialization timing boundary")
    secure_config = None
    backend = result.get("backend")
    if dataset not in DATASETS:
        abort(f"{path} has an unknown dataset")
    native_ntt_plan: dict[str, int] | None = None
    if formal:
        if not formal_pk_timing_fields.issubset(result):
            abort(f"{path} formal result omits its authenticated compact PK restore timing")
        if backend is not None or "execution_mode" in result or "phase_backends" in result:
            abort(f"{path} exposes a backend execution axis in the formal result")
        native_backend = result.get("native_backend")
        if native_backend != NATIVE_FORMAL_BACKEND:
            abort(f"{path} does not identify the authenticated native formal engine")
        session_id = result.get("formal_session_id")
        if not isinstance(session_id, str) or not FORMAL_SESSION_ID.fullmatch(session_id):
            abort(f"{path} has no valid formal session identity")
        if expected_session_id is None:
            abort(f"{path} is a formal result but no active session was supplied")
        if expected_session_id is not None and session_id != expected_session_id:
            abort(f"{path} belongs to a different formal session")
        if path.parent.name != dataset:
            abort(f"{path} is outside its canonical dataset result directory")
        compatibility_context = _load_compatible_resume_context(
            path,
            result,
            session_id=session_id,
        )
        if compatibility_context is not None and dataset == "ogbn_arxiv":
            ppi_result_path = ROOT / COMPAT_OUTPUT_ROOT / "ppi" / "result.json"
            ppi_result = load_json(ppi_result_path, "completed compatible PPI result")
            if not isinstance(ppi_result, dict):
                abort("completed compatible PPI result is not a JSON object")
            ppi_context = _load_compatible_resume_context(
                ppi_result_path,
                ppi_result,
                session_id=session_id,
            )
            if ppi_context is None or ppi_context.get("receipt") is None:
                abort("completed compatible OGBN result has no paired PPI compatibility receipt")
            _validate_compatible_capacity_precheck(
                ppi_context,
                result_path=ppi_result_path,
                result=ppi_result,
                require_ogbn_gate=True,
            )
        secure_config = (
            validate_config(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg", dataset)
            if expected_config_values is None
            else dict(expected_config_values)
        )
        if secure_config.get("dataset") != dataset:
            abort(f"{path} authenticated config values belong to another dataset")
        if result["quant_cfg_id"] != secure_config["quant_cfg_id"] or result["static_tables_id"] != secure_config["static_table_id"]:
            abort(f"{path} quantization/static-table identities differ from the canonical config")
        for field in formal_gpu_identity_fields:
            if field not in result:
                abort(f"{path} is missing current-session GPU identity field {field}")
        selected_device = result["selected_cuda_device"]
        device_keys = {"index", "uuid", "name", "driver_version", "compute_capability", "logical_device_id"}
        if not isinstance(selected_device, dict) or set(selected_device) != device_keys:
            abort(f"{path} selected CUDA device identity has missing or unknown fields")
        if (
            isinstance(selected_device["index"], bool) or not isinstance(selected_device["index"], int) or selected_device["index"] < 0
            or isinstance(selected_device["logical_device_id"], bool) or selected_device["logical_device_id"] != 0
            or any(not isinstance(selected_device[key], str) or not selected_device[key] for key in ("uuid", "name", "driver_version", "compute_capability"))
        ):
            abort(f"{path} selected CUDA device identity contains invalid values")
        for field in ("build_receipt_sha256", "build_environment_sha256", "gpu_prover_record_sha256", "independent_verifier_log_sha256"):
            require_digest(result[field], f"{path} {field}")
        receipt_path = ROOT / "artifacts/build/gpu/build_receipt.json"
        require_file(receipt_path, "current GPU build receipt")
        if sha256(receipt_path) != result["build_receipt_sha256"]:
            abort(f"{path} build receipt does not belong to the current formal session")
        receipt = load_json(receipt_path, "current GPU build receipt")
        environment = receipt.get("build_environment")
        if (
            receipt.get("selected_cuda_device") != selected_device
            or receipt.get("features") != list(FORMAL_GPU_FEATURES)
            or receipt.get("build_environment_sha256") != result["build_environment_sha256"]
            or not isinstance(environment, dict)
            or environment.get("selected_cuda_device") != selected_device
            or environment.get("features") != list(FORMAL_GPU_FEATURES)
            or environment.get("build_environment_sha256") != result["build_environment_sha256"]
        ):
            abort(f"{path} GPU result/build identity does not match")
        environment_payload = dict(environment)
        recorded_environment_digest = environment_payload.pop("build_environment_sha256", None)
        if canonical_json_digest(environment_payload) != recorded_environment_digest:
            abort(f"{path} GPU build environment digest is invalid")
        verifier_log = require_file(path.parent / "logs/verify.log", "independent verifier log")
        if sha256(verifier_log) != result["independent_verifier_log_sha256"]:
            abort(f"{path} independent verifier log identity is stale")
    else:
        if COMPAT_RESULT_FIELDS.intersection(result):
            abort(f"{path} nonformal result contains formal compatible-resume receipt fields")
        if formal_pk_timing_fields.intersection(result):
            abort(f"{path} nonformal result contains formal compact PK restore timing fields")
        if expected_session_id is not None:
            abort(f"{path} is not bound to the active formal session")
        if backend not in BACKENDS:
            abort(f"{path} is not an explicitly local native result")
        native_backend = backend
    if not formal and ("execution_mode" in result or "phase_backends" in result):
        expected_mode = "mixed" if backend == "gpu" else "cpu"
        legacy_phases = {"prepare_compile": "cpu", "setup": expected_mode, "public_statement": "cpu", "witness_generation": "cpu", "model_commitment": "cpu", "prove": expected_mode, "verify": "cpu"}
        cpu_phases = {phase: {"cpu": True, "gpu": False} for phase in FORMAL_PHASE_BACKENDS}
        allowed_phases = (legacy_phases, cpu_phases) if backend == "cpu" else (legacy_phases,)
        if result.get("execution_mode") != expected_mode or result.get("phase_backends") not in allowed_phases:
            abort(f"{path} has an invalid native phase device schedule")
    if "native_backend" in result and result["native_backend"] != native_backend:
        abort(f"{path} has inconsistent native engine identity")
    if result["proof_scope"] != CANONICAL_PROOF_SCOPE:
        abort(f"{path} is not full_gat")
    if result["privacy_policy"] != CANONICAL_PRIVACY:
        abort(f"{path} has the wrong privacy policy")
    if result["public_statement_mode"] != CANONICAL_STATEMENT_MODE or result["output_mode"] != CANONICAL_OUTPUT_MODE:
        abort(f"{path} has the wrong public statement/output mode")
    if result["relation_id"] != CANONICAL_RELATION:
        abort(f"{path} has the wrong relation id")
    if not isinstance(result["quant_cfg_id"], str) or not result["quant_cfg_id"]:
        abort(f"{path} has no quantization identity")
    if not isinstance(result["static_tables_id"], str) or not result["static_tables_id"]:
        abort(f"{path} has no static-table identity")
    if result["verifier_reads_witness"] is not False or result["proof_contains_private_witness"] is not False:
        abort(f"{path} violates the verifier/witness boundary")
    if result["native_verify_result"] is not True or result["native_public_instance_binding"] is not True:
        abort(f"{path} has no native verify/public-instance attestation")
    circuit_ref = require_circuit_ref(result["circuit_ref"])
    for key in (
        "public_statement_sha256", "public_output_sha256", "srs_sha256", "settings_sha256", "compiled_circuit_sha256", "vk_sha256", "pk_sha256", "proof_sha256",
        "binary_sha256", "source_tree_sha256", "resource_manifest_sha256", "artifact_manifest_sha256", "core_run_manifest_sha256"
    ):
        require_digest(result[key], f"{path}:{key}")
    if isinstance(result["proof_size_bytes"], bool) or not isinstance(result["proof_size_bytes"], int) or result["proof_size_bytes"] <= 0:
        abort(f"{path} has no positive proof size")
    if formal:
        production_timing_fields = ("public_statement_production", "final_statement_binding")
        phase_timings = result["phase_timings_ms"]
        if not isinstance(phase_timings, dict) or any(
            field not in phase_timings
            or isinstance(phase_timings[field], bool)
            or not isinstance(phase_timings[field], (int, float))
            or not math.isfinite(phase_timings[field])
            or phase_timings[field] <= 0
            for field in production_timing_fields
        ):
            abort(f"{path} formal result omits its authenticated producer/final stage wall timings")
    require_positive_timings(result["phase_timings_ms"], f"{path}:phase_timings_ms")
    for field, digest_field in PATH_DIGESTS.items():
        artifact = path_from_result(result[field], field)
        if sha256(artifact) != result[digest_field]:
            abort(f"{path} digest mismatch for {field}")
    proof = path_from_result(result["proof_path"], "proof")
    if proof.stat().st_size != result["proof_size_bytes"]:
        abort(f"{path} proof_size_bytes does not match proof file")
    statement = path_from_result(result["public_statement_path"], "public statement")
    public_output = path_from_result(result["public_output_path"], "public output")
    statement_document = validate_public_statement(
        statement,
        dataset=dataset,
        circuit_ref=circuit_ref,
        public_output_sha256=result["public_output_sha256"],
        quant_cfg_id=secure_config["quant_cfg_id"] if secure_config is not None else result["quant_cfg_id"],
        static_tables_id=secure_config["static_table_id"] if secure_config is not None else result["static_tables_id"],
        degree_bound_id=secure_config["degree_bound_id"] if secure_config is not None else None,
        degree_bound_digest=secure_config["degree_bound_digest"] if secure_config is not None else None,
    )
    if public_output_values(public_output, dataset=dataset) != statement_document["public_ezkl"]["y"]:
        abort(f"{path} public output Y disagrees element-by-element with statement")
    metric = require_exact_keys(result["commitment_metric"], {"operation", "included_in", "path", "sha256"}, "result commitment metric")
    commitment_path = path_from_result(metric["path"], "native proof metrics")
    if commitment_path != path.parent / "native_proof_metrics.json":
        abort(f"{path} native proof metrics do not belong to this formal cell")
    require_digest(metric["sha256"], "native proof metrics digest")
    if sha256(commitment_path) != metric["sha256"]:
        abort(f"{path} native proof metrics digest mismatch")
    proof_metrics = validate_native_proof_metrics(commitment_path, proof_sha256=result["proof_sha256"])
    if metric["operation"] != proof_metrics["operation"] or metric["included_in"] != proof_metrics["included_in"]:
        abort(f"{path} native commitment measurement scope mismatch")
    duration = require_positive_finite_duration(result, "commitment_time_ms", str(path))
    if duration != proof_metrics["commitment_time_ms"]:
        abort(f"{path} commitment_time_ms does not match the native measured duration")
    prove_time = require_positive_finite_duration(result, "prove_time_ms", str(path))
    if prove_time != proof_metrics["prove_time_ms"]:
        abort(f"{path} prove_time_ms does not match native proof timing")
    if not math.isclose(
        float(result["phase_timings_ms"].get("prove", float("nan"))),
        float(proof_metrics["prove_time_ms"]),
        rel_tol=0.0,
        abs_tol=0.000001,
    ):
        abort(f"{path} phase prove timing does not match the native composed proof metric")
    verify_metric = require_exact_keys(result["verify_metric"], {"operation", "included_in", "path", "sha256"}, "result native verification metric")
    verify_metrics_path = path_from_result(verify_metric["path"], "native verification metrics")
    if verify_metrics_path != path.parent / "native_verify_metrics.json":
        abort(f"{path} native verification metrics do not belong to this formal cell")
    require_digest(verify_metric["sha256"], "native verification metrics digest")
    if sha256(verify_metrics_path) != verify_metric["sha256"]:
        abort(f"{path} native verification metrics digest mismatch")
    native_verify_metrics = validate_native_verify_metrics(verify_metrics_path, proof_sha256=result["proof_sha256"])
    if verify_metric["operation"] != native_verify_metrics["operation"] or verify_metric["included_in"] != native_verify_metrics["included_in"]:
        abort(f"{path} native verification measurement scope mismatch")
    verify_time = require_positive_finite_duration(result, "verify_time_ms", str(path))
    if verify_time != native_verify_metrics["verify_time_ms"]:
        abort(f"{path} verify_time_ms does not match native verification timing")
    artifact_manifest = path_from_result(result["artifact_manifest_path"], "artifact identity manifest")
    artifact = validate_artifact_identity(
        artifact_manifest,
        dataset=dataset,
        backend=native_backend,
        binary=path_from_result(result["binary_path"], "binary"),
        srs=path_from_result(result["srs_path"], "SRS"),
        settings=path_from_result(result["settings_path"], "settings"),
        compiled=path_from_result(result["compiled_circuit_path"], "compiled circuit"),
        vk=path_from_result(result["vk_path"], "verification key"),
        pk=path_from_result(result["pk_path"], "proving key"),
    )
    validate_proof_instances(
        result,
        proof,
        statement=statement,
        public_output=public_output,
        artifact_manifest=artifact_manifest,
        artifact=artifact,
    )
    if secure_config is not None:
        validate_canonical_public_inputs(
            statement_document,
            dataset=dataset,
            secure_config=secure_config,
            setup_context_path=artifact_manifest.parent / "producer_context.json",
            expected_resource_manifest_sha256=result["resource_manifest_sha256"],
        )
    if artifact["statement_sha256"] != result["public_statement_sha256"]:
        abort(f"{path} artifact statement identity is not externally bound")
    artifact_records = {record["path"]: record for record in artifact["files"]}
    for name, digest in (
        ("statement.json", result["public_statement_sha256"]),
        ("public_output.json", result["public_output_sha256"]),
    ):
        if name not in artifact_records or artifact_records[name]["sha256"] != digest:
            abort(f"{path} artifact file {name} is not bound to the public boundary")
    if formal:
        native_receipt_names = (
            "setup_receipt.json",
            "native_keygen_budget.json",
            "native_keygen_receipt.json",
            "native_prove_budget.json",
            "admission.json",
        )
        if any(name not in artifact_records for name in native_receipt_names):
            abort(f"{path} artifact manifest omits native setup/prove resource receipts")
        setup_receipt = load_json(artifact_manifest.parent / native_receipt_names[0], "formal setup resource receipt")
        keygen_budget = load_json(artifact_manifest.parent / native_receipt_names[1], "native keygen budget")
        keygen_receipt = load_json(artifact_manifest.parent / native_receipt_names[2], "native keygen census receipt")
        prove_budget = load_json(artifact_manifest.parent / native_receipt_names[3], "native proving-key read budget")
        admission_path = artifact_manifest.parent / "admission.json"
        admission = load_json(admission_path, "formal native admission")
        current_admission_sha256 = sha256(admission_path)
        if any(not isinstance(value, dict) for value in (setup_receipt, keygen_budget, keygen_receipt, prove_budget)):
            abort(f"{path} native resource receipts must be JSON objects")
        native_ntt_plan = _validate_native_ntt_profile(path, admission_path, admission, prove_budget)
        receipt_digests = {name: artifact_records[name]["sha256"] for name in native_receipt_names}
        pk_timing_path = artifact_manifest.parent / "native_pk_read_timing.json"
        timing_file = artifact_records.get("native_pk_read_timing.json")
        if timing_file is None or timing_file.get("sha256") != result["native_pk_read_timing_sha256"] or sha256(pk_timing_path) != result["native_pk_read_timing_sha256"]:
            abort(f"{path} artifact manifest does not bind the native PK restore timing")
        pk_timing = validate_native_pk_read_timing(
            pk_timing_path,
            session_id=session_id,
            dataset=dataset,
            native_prove_budget_sha256=receipt_digests["native_prove_budget.json"],
            pk_sha256=result["pk_sha256"],
        )
        if any(
            pk_timing.get(name) != value
            for name, value in (
                ("admission_sha256", prove_budget.get("admission_sha256")),
                ("resource_manifest_sha256", result["resource_manifest_sha256"]),
                ("setup_receipt_sha256", receipt_digests["setup_receipt.json"]),
                ("native_keygen_budget_sha256", receipt_digests["native_keygen_budget.json"]),
                ("native_keygen_receipt_sha256", receipt_digests["native_keygen_receipt.json"]),
                ("build_receipt_sha256", result["build_receipt_sha256"]),
                ("build_environment_sha256", result["build_environment_sha256"]),
                ("binary_sha256", result["binary_sha256"]),
                ("settings_sha256", result["settings_sha256"]),
                ("compiled_circuit_sha256", result["compiled_circuit_sha256"]),
                ("vk_sha256", result["vk_sha256"]),
            )
        ):
            abort(f"{path} native PK restore timing disagrees with authenticated setup/build identities")
        phase_timings = result["phase_timings_ms"]
        if (
            not math.isclose(phase_timings.get("pk_artifact_read_validate", float("nan")), pk_timing["artifact_read_validate_ms"], rel_tol=0.0, abs_tol=0.000001)
            or not math.isclose(phase_timings.get("pk_materialize", float("nan")), pk_timing["pk_materialize_ms"], rel_tol=0.0, abs_tol=0.000001)
            or not math.isclose(phase_timings.get("setup_pk_sigma_deferred_source_prepare", float("nan")), pk_timing["sigma_deferred_source_prepare_wall_ms"], rel_tol=0.0, abs_tol=0.000001)
            or not math.isclose(proof_metrics["pk_materialize_time_ms"], pk_timing["pk_materialize_ms"], rel_tol=0.0, abs_tol=0.000001)
        ):
            abort(f"{path} phase timing does not preserve the measured PK restore subphases")
        command_wall = result["prove_command_wall_ms"]
        if isinstance(command_wall, bool) or not isinstance(command_wall, (int, float)) or not math.isfinite(command_wall) or command_wall <= 0:
            abort(f"{path} native prove command wall duration is invalid")
        residual = command_wall - pk_timing["artifact_read_validate_ms"] - pk_timing["sigma_deferred_source_prepare_wall_ms"]
        recorded_residual = result["prove_command_residual_excluding_pk_artifact_read_and_sigma_prepare_ms"]
        if (
            isinstance(recorded_residual, bool)
            or not isinstance(recorded_residual, (int, float))
            or not math.isfinite(recorded_residual)
            or not math.isclose(recorded_residual, residual, rel_tol=0.0, abs_tol=0.001)
        ):
            abort(f"{path} outer prove residual does not preserve its mutually exclusive restore phases")
        if residual > 0.0:
            if result["prove_wall_scope"] != "outer_command_residual_excludes_artifact_read_and_sigma_prepare_includes_pk_materialize":
                abort(f"{path} outer prove wall residual does not preserve PK materialization")
        elif residual >= -0.01:
            if result["prove_wall_scope"] != "outer_command_subphase_overlap_within_0_01ms_includes_pk_materialize":
                abort(f"{path} sub-resolution PK restore overlap is not transparently recorded")
        else:
            abort(f"{path} PK restore stages exceed its prove command wall time")
        expected_setup = {
            "schema": "ezkl.setup-reuse-receipt.v1",
            "session_id": session_id,
            "dataset": dataset,
            "admission_sha256": prove_budget.get("admission_sha256"),
            "native_keygen_receipt_sha256": receipt_digests["native_keygen_receipt.json"],
        }
        if any(setup_receipt.get(name) != value for name, value in expected_setup.items()):
            abort(f"{path} setup receipt identity does not match its formal cell")
        setup_files = setup_receipt.get("setup_files")
        if not isinstance(setup_files, dict) or any(
            setup_files.get(name) != digest
            for name, digest in (
                ("admission.json", current_admission_sha256),
                ("settings.json", result["settings_sha256"]),
                ("compiled.circuit", result["compiled_circuit_sha256"]),
                ("vk.key", result["vk_sha256"]),
                ("pk.key", result["pk_sha256"]),
                ("native_keygen_budget.json", receipt_digests["native_keygen_budget.json"]),
                ("native_keygen_receipt.json", receipt_digests["native_keygen_receipt.json"]),
            )
        ):
            abort(f"{path} setup receipt does not bind its exact key/circuit artifacts")
        expected_keygen = {
            "schema": "ezkl.native-keygen-receipt.v4",
            "session_id": session_id,
            "dataset": dataset,
            "budget_receipt_sha256": receipt_digests["native_keygen_budget.json"],
            "settings_sha256": result["settings_sha256"],
            "compiled_circuit_sha256": result["compiled_circuit_sha256"],
            "vk_sha256": result["vk_sha256"],
            "pk_sha256": result["pk_sha256"],
            "evidence_kind": "mandatory_vk_pk_native_keygen_synthesis_census",
            "coverage": "actual_vk_and_pk_synthesis_census; exact_selector_advice_fixed_permutation_and_lookup_bounds",
        }
        if any(keygen_receipt.get(name) != value for name, value in expected_keygen.items()):
            abort(f"{path} native keygen census receipt does not match its formal cell")
        relation_identity = keygen_receipt.get("relation_identity")
        if not isinstance(relation_identity, dict) or relation_identity.get("dataset") != dataset or relation_identity.get("resource_manifest_sha256") != result["resource_manifest_sha256"]:
            abort(f"{path} native keygen relation identity does not match the canonical result")
        expected_native_budget = {
            "schema": "ezkl.native-keygen-budget.v3",
            "session_id": session_id,
            "dataset": dataset,
            "admission_sha256": setup_receipt.get("admission_sha256"),
            "settings_sha256": result["settings_sha256"],
            "resource_manifest_sha256": result["resource_manifest_sha256"],
        }
        if compatibility_context is not None and dataset == "ppi":
            old_identity = compatibility_context["old_identity"]
            expected_native_budget.update({
                "binary_sha256": old_identity.get("binary_sha256"),
                "build_receipt_sha256": old_identity.get("build_receipt_sha256"),
                "build_environment_sha256": old_identity.get("build_environment_sha256"),
            })
        else:
            expected_native_budget.update({
                "binary_sha256": result["binary_sha256"],
                "build_receipt_sha256": result["build_receipt_sha256"],
                "build_environment_sha256": result["build_environment_sha256"],
            })
        if any(keygen_budget.get(name) != value for name, value in expected_native_budget.items()):
            abort(f"{path} native keygen budget does not match its formal build/resource identity")
        setup_disk_required = keygen_budget.get("setup_disk_required_known_bytes")
        proof_spill_reserve = keygen_budget.get("proof_spill_reserve_bytes")
        disk_required = keygen_budget.get("disk_required_known_bytes")
        disk_available = keygen_budget.get("disk_available_bytes")
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > sys.maxsize
                for value in (setup_disk_required, proof_spill_reserve, disk_required, disk_available)
            )
            or proof_spill_reserve != 0
            or keygen_budget.get("disk_budget_scope")
            != "setup_keygen_peak_and_source_bounded_permutation_backing; proof_disk_checked_after_actual_setup_census"
            or setup_disk_required != disk_required
            or disk_required > disk_available
        ):
            abort(f"{path} native keygen budget confuses Setup storage with the post-census proof disk gate")
        native_census = keygen_receipt.get("native_census")
        if not isinstance(native_census, dict) or not isinstance(native_census.get("preflight"), dict):
            abort(f"{path} native keygen receipt has no exact synthesis census")
        preflight = native_census["preflight"]
        stage_caps = keygen_receipt.get("stage_budget_caps")
        if not isinstance(stage_caps, list) or len(stage_caps) != 2:
            abort(f"{path} native keygen receipt has no VK/PK stage caps")
        caps: dict[str, dict[str, int]] = {}
        for index, stage in enumerate(("vk", "pk")):
            value = stage_caps[index]
            if not isinstance(value, dict) or value.get("stage") != stage:
                abort(f"{path} native keygen stage caps are not ordered VK then PK")
            selector_cap = value.get("max_selector_activation_bytes")
            permutation_cap = value.get("max_permutation_backing_bytes")
            if any(isinstance(cap, bool) or not isinstance(cap, int) or cap < 0 or cap >= 1 << 64 for cap in (selector_cap, permutation_cap)):
                abort(f"{path} native {stage.upper()} allocation cap is invalid")
            host_snapshot = keygen_budget.get("host_available_bytes")
            disk_snapshot = keygen_budget.get("native_permutation_backing_budget_bytes")
            selector_initial = preflight.get("selector_activation_initial_backing_bytes")
            source_components = (
                preflight.get("permutation_data_virtual_bytes"),
                preflight.get("permutation_directory_virtual_bytes"),
                keygen_budget.get("permutation_successor_backing_file_upper_bytes"),
            )
            if (
                any(isinstance(bound, bool) or not isinstance(bound, int) or bound < 0 or bound > sys.maxsize for bound in (host_snapshot, disk_snapshot, selector_initial, *source_components))
            ):
                abort(f"{path} native {stage.upper()} resource ceiling is malformed")
            source_backing_upper = sum(source_components)
            if (
                source_backing_upper > sys.maxsize
                or selector_cap < selector_initial
                or selector_cap > host_snapshot
                or permutation_cap > min(disk_snapshot, source_backing_upper)
            ):
                abort(f"{path} native {stage.upper()} allocation cap exceeds its authenticated budget envelope")
            caps[stage] = {"selector": selector_cap, "permutation": permutation_cap}
        selector_peak = native_census.get("selector_activation_storage_peak_bytes")
        permutation_values = [native_census.get(name) for name in ("permutation_data_allocated_bytes", "permutation_directory_allocated_bytes", "permutation_successor_edges")]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize for value in permutation_values):
            abort(f"{path} native permutation census byte counts are invalid")
        constraint_system = admission.get("constraint_system")
        permutation_domain = admission.get("domain_size")
        permutation_columns = constraint_system.get("num_permutation_columns") if isinstance(constraint_system, dict) else None
        if (
            isinstance(permutation_domain, bool) or not isinstance(permutation_domain, int) or permutation_domain <= 0
            or isinstance(permutation_columns, bool) or not isinstance(permutation_columns, int) or permutation_columns <= 0
        ):
            abort(f"{path} native admission has invalid permutation bitmap geometry")
        permutation_cells = permutation_domain * permutation_columns
        if permutation_cells > sys.maxsize or permutation_values[2] > permutation_cells:
            abort(f"{path} native permutation census exceeds its admitted bitmap domain")
        successor_logical_bytes = 16 * permutation_values[2]
        successor_reserved_bytes = ((successor_logical_bytes + 4095) // 4096) * 4096
        target_bitmap_geometry = _validate_native_pk_v4_read_backing_budget(
            admission, permutation_values[2], prove_budget
        )
        target_bitmap_virtual_bytes = target_bitmap_geometry["target_bitmap_virtual_bytes"]
        target_bitmap_page_count = target_bitmap_geometry["target_bitmap_page_count"]
        target_bitmap_reserved_bytes = target_bitmap_geometry["target_bitmap_backing_bytes"]
        exact_pk_read_backing_bytes = target_bitmap_reserved_bytes
        permutation_backing = permutation_values[0] + permutation_values[1] + successor_reserved_bytes
        if (
            successor_reserved_bytes > sys.maxsize
            or target_bitmap_reserved_bytes > sys.maxsize
            or exact_pk_read_backing_bytes > sys.maxsize
            or isinstance(selector_peak, bool)
            or not isinstance(selector_peak, int)
            or selector_peak < 0
            or selector_peak > sys.maxsize
            or selector_peak > min(caps["vk"]["selector"], caps["pk"]["selector"])
            or permutation_backing < 0
            or permutation_backing > sys.maxsize
            or permutation_backing > min(caps["vk"]["permutation"], caps["pk"]["permutation"])
        ):
            abort(f"{path} native keygen census exceeds its VK/PK resource caps")
        expected_prove = {
            "schema": "ezkl.native-prove-budget.v8",
            "session_id": session_id,
            "dataset": dataset,
            "admission_sha256": setup_receipt.get("admission_sha256"),
            "setup_receipt_sha256": receipt_digests["setup_receipt.json"],
            "native_keygen_budget_sha256": receipt_digests["native_keygen_budget.json"],
            "native_keygen_receipt_sha256": receipt_digests["native_keygen_receipt.json"],
            "binary_sha256": result["binary_sha256"],
            "settings_sha256": result["settings_sha256"],
            "compiled_circuit_sha256": result["compiled_circuit_sha256"],
            "vk_sha256": result["vk_sha256"],
            "pk_sha256": result["pk_sha256"],
            "resource_manifest_sha256": result["resource_manifest_sha256"],
            "build_receipt_sha256": result["build_receipt_sha256"],
            "build_environment_sha256": result["build_environment_sha256"],
            "native_execution_policy": FORMAL_GPU_EXECUTION_POLICY,
            "h_workspace_borrows_ntt": True,
            **native_ntt_plan,
        }
        prove_budget_exact_keys = set(expected_prove) | {
            "icicle_msm_workspace_query_schema",
            "icicle_msm_ordinary_factor1_device_workspace_peak_bytes",
            "icicle_msm_prepared_factor4_device_workspace_peak_bytes",
            "native_pk_format_version",
            "native_pk_sigma_wire_encoding",
            "native_pk_sigma_index_width_bits",
            "native_pk_sigma_packed_payload_bytes",
            "disk_available_bytes",
            "disk_required_known_bytes",
            "permutation_grand_product_set_count",
            "permutation_grand_product_bank_mapped_bytes",
            "permutation_grand_product_window_mapped_bytes",
            "permutation_grand_product_commit_workspace_bytes",
            "proof_permutation_grand_product_commit_host_peak_bytes",
            "shplonk_fixed_cache_group_host_bound_bytes",
            "shplonk_fixed_cache_group_disk_bound_bytes",
            "shplonk_fixed_cache_bank_header_bytes",
            "shplonk_fixed_cache_domain_size",
            "shplonk_fixed_cache_scalar_bytes",
            "shplonk_fixed_cache_max_fixed_sources_per_group",
            "shplonk_fixed_cache_max_points_per_group",
            "shplonk_fixed_cache_rotation_set_count",
            "shplonk_fixed_cache_max_groups",
            "shplonk_fixed_cache_host_limit_bytes",
            "shplonk_fixed_cache_bank_header_limit_bytes",
            "shplonk_fixed_cache_disk_limit_bytes",
            "native_pk_read_backing_required_bytes",
            "native_pk_read_mapped_peak_upper_bytes",
            "native_pk_read_mapped_extra_upper_bytes",
            "max_total_backing_bytes",
            "native_pk_read_host_scratch_upper_bytes",
            "native_pk_materialize_host_scratch_upper_bytes",
            "native_pk_read_successor_cache_upper_bytes",
            "native_pk_read_target_bitmap_fixed_cache_upper_bytes",
            "native_pk_read_target_bitmap_allocation_bitmap_upper_bytes",
            "native_pk_read_validation_buffers_upper_bytes",
            "native_pk_read_stream_buffer_upper_bytes",
            "host_available_bytes",
            "host_required_known_bytes",
            "h_kernel_local_memory_upper_bytes",
            "h_program_metadata_upper_bytes",
            "cuda_required_workspace_bytes",
            "integrated_device",
            "device_workspace_peak_bytes",
            "backing_scope",
            "host_scope",
        }
        require_exact_keys(prove_budget, prove_budget_exact_keys, "native prove budget")
        if any(prove_budget.get(name) != value for name, value in expected_prove.items()):
            abort(f"{path} native prove budget does not bind its setup, keygen, and binary identities")
        if (
            not isinstance(admission, dict)
            or receipt_digests["admission.json"] != current_admission_sha256
            or current_admission_sha256 != prove_budget.get("admission_sha256")
            or current_admission_sha256 != keygen_budget.get("admission_sha256")
            or current_admission_sha256 != setup_receipt.get("admission_sha256")
        ):
            abort(f"{path} formal setup/prove budgets do not bind the current Admission bytes")
        if compatibility_context is not None and dataset == "ppi":
            _validate_compatible_receipt_artifacts(
                compatibility_context,
                result_path=path,
                result=result,
                artifact_manifest=artifact_manifest,
                artifact=artifact,
                artifact_records=artifact_records,
                setup_receipt=setup_receipt,
                keygen_budget=keygen_budget,
                keygen_receipt=keygen_receipt,
                prove_budget=prove_budget,
                receipt_digests=receipt_digests,
                admission=admission,
                admission_path=admission_path,
            )
        _validate_shplonk_fixed_cache_budget(path, admission, prove_budget)
        for name in (
            "permutation_grand_product_set_count",
            "permutation_grand_product_bank_mapped_bytes",
            "permutation_grand_product_window_mapped_bytes",
            "permutation_grand_product_commit_workspace_bytes",
            "proof_permutation_grand_product_commit_host_peak_bytes",
            "native_pk_read_backing_required_bytes",
            "max_total_backing_bytes",
            "host_available_bytes",
            "host_required_known_bytes",
            "native_pk_read_host_scratch_upper_bytes",
            "native_pk_read_successor_cache_upper_bytes",
            "native_pk_read_target_bitmap_fixed_cache_upper_bytes",
            "native_pk_read_target_bitmap_allocation_bitmap_upper_bytes",
            "native_pk_read_validation_buffers_upper_bytes",
            "native_pk_read_stream_buffer_upper_bytes",
            "icicle_msm_max_concurrency",
            "icicle_msm_precompute_factor",
            "icicle_msm_precomputed_basis_count",
            "icicle_msm_precomputed_basis_device_upper_bytes",
            "icicle_msm_point_staging_host_upper_bytes",
            "icicle_ntt_min_log_size",
            "icicle_ntt_max_log_size",
            "icicle_ntt_capacity_elements",
            "icicle_ntt_device_workspace_upper_bytes",
            "icicle_ntt_domain_cache_upper_bytes",
            "icicle_ntt_init_scratch_upper_bytes",
            "icicle_ntt_device_peak_upper_bytes",
            "icicle_h_resident_device_upper_bytes",
            "icicle_ntt_host_staging_upper_bytes",
        ):
            value = prove_budget.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize:
                abort(f"{path} native prove budget has invalid {name}")
        target_bitmap_allocation_upper = target_bitmap_geometry["target_bitmap_allocation_bitmap_bytes"]
        expected_read_host_scratch = (
            prove_budget["native_pk_read_successor_cache_upper_bytes"]
            + prove_budget["native_pk_read_target_bitmap_fixed_cache_upper_bytes"]
            + target_bitmap_allocation_upper
            + prove_budget["native_pk_read_validation_buffers_upper_bytes"]
            + prove_budget["native_pk_read_stream_buffer_upper_bytes"]
        )
        if (
            prove_budget["disk_required_known_bytes"] > prove_budget["disk_available_bytes"]
            or prove_budget["permutation_grand_product_bank_mapped_bytes"]
            != prove_budget["permutation_grand_product_set_count"]
            * prove_budget["permutation_grand_product_window_mapped_bytes"]
            or prove_budget["permutation_grand_product_commit_workspace_bytes"]
            != 4 * prove_budget["permutation_grand_product_window_mapped_bytes"]
            or prove_budget["permutation_grand_product_bank_mapped_bytes"]
            > prove_budget["disk_required_known_bytes"]
            or prove_budget["proof_permutation_grand_product_commit_host_peak_bytes"]
            > prove_budget["host_required_known_bytes"]
            or prove_budget["native_pk_read_backing_required_bytes"] != exact_pk_read_backing_bytes
            or prove_budget["max_total_backing_bytes"] != exact_pk_read_backing_bytes
            or prove_budget["native_pk_read_target_bitmap_allocation_bitmap_upper_bytes"] != target_bitmap_allocation_upper
            or prove_budget["native_pk_read_validation_buffers_upper_bytes"] != 4 * 4096 * 16
            or prove_budget["native_pk_read_successor_cache_upper_bytes"] != keygen_budget.get("native_successor_read_cache_bytes")
            or prove_budget["native_pk_read_target_bitmap_fixed_cache_upper_bytes"] != keygen_budget.get("native_target_bitmap_fixed_cache_bytes")
            or prove_budget["native_pk_read_stream_buffer_upper_bytes"] != keygen_budget.get("native_pk_read_buffer_capacity_bytes")
            or prove_budget["native_pk_read_host_scratch_upper_bytes"] != expected_read_host_scratch
            or prove_budget["native_pk_read_host_scratch_upper_bytes"] > prove_budget["host_required_known_bytes"]
            or prove_budget["host_required_known_bytes"] > prove_budget["host_available_bytes"]
        ):
            abort(f"{path} native PK reader resources do not fit the authenticated host/disk envelope")
    core_path = path.parent / "core_run_manifest.json"
    core = load_json(core_path, "core run manifest")
    require_exact_keys(
        core,
        {
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
        },
        "core run manifest",
    )
    if (
        core["schema_version"] != 1 or core["relation_id"] != CANONICAL_RELATION or core["dataset"] != dataset or core["backend"] != native_backend or
        core["proof_scope"] != CANONICAL_PROOF_SCOPE or core["privacy_policy"] != CANONICAL_PRIVACY or
        core["public_statement_mode"] != CANONICAL_STATEMENT_MODE or core["output_mode"] != CANONICAL_OUTPUT_MODE or core["native_verify_result"] is not True or
        core["native_public_instance_binding"] is not True or core["verifier_reads_witness"] is not False or
        core["proof_contains_private_witness"] is not False or core["public_statement_sha256"] != result["public_statement_sha256"] or
        core["public_output_sha256"] != result["public_output_sha256"] or core["circuit_ref"] != artifact["circuit_ref"]
    ):
        abort(f"{path} core run manifest is not a successful native external binding")
    require_digest(core["public_statement_sha256"], "core public statement digest")
    require_digest(core["public_output_sha256"], "core public output digest")
    require_digest(core["proof_instances_sha256"], "core proof instance digest")
    if not isinstance(core["proof_instance_count"], int) or core["proof_instance_count"] <= 0:
        abort(f"{path} core proof instance count is invalid")
    if (core["proof_instances_sha256"] != result["proof_instances"]["sha256"] or core["proof_instance_count"] != result["proof_instances"]["count"]):
        abort(f"{path} core proof-instance attestation does not match the native proof")
    if sha256(core_path) != result["core_run_manifest_sha256"]:
        abort(f"{path} core run manifest digest mismatch")
    if require_circuit_ref(artifact["circuit_ref"], "artifact circuit_ref") != circuit_ref:
        abort(f"{path} circuit_ref does not match artifact identity")
    if result["srs_provenance"] != artifact["srs_provenance"]:
        abort(f"{path} SRS provenance does not match artifact identity")
    validate_gpu_evidence_from_result(result)
    if native_backend == "gpu":
        record = load_json(proof.parent / "gpu_prover_record.json", "native GPU prover record")
        require_exact_keys(record, {
            "schema_version", "proof_sha256", "admission_sha256", "native_execution_policy", "icicle_msm_max_concurrency",
            "icicle_msm_precompute_factor", "gpu_msm_precompute_calls", "gpu_msm_precomputed_chunk_calls",
            "gpu_msm_precomputed_cache_hits", "icicle_msm_precomputed_basis_device_upper_bytes",
            "icicle_msm_precomputed_basis_device_bytes", "icicle_msm_precomputed_basis_device_peak_bytes",
            "msm_operations", "gpu_msm_chunk_calls", "ntt_calls", "cpu_fft_calls", "synchronized_calls", "device_id",
            "gpu_ntt_elements", "ntt_max_size", "ntt_workspace_capacity_bytes", "ntt_domain_capacity_bytes",
            "ntt_host_staging_capacity_bytes", "ntt_device_peak_upper_bound_bytes",
            "gpu_h_eval_groups", "gpu_h_eval_calls", "gpu_h_eval_rows", "gpu_h_eval_fallbacks", "gpu_h_eval_chunks",
            "gpu_h_eval_syncs", "gpu_h_transfer_syncs", "gpu_h_workspace_high_water_bytes",
            "gpu_h_local_memory_peak_upper_bytes", "gpu_h_resident_high_water_bytes", "evidence",
        }, "native GPU prover record")
        counts = [record[key] for key in (
            "schema_version", "icicle_msm_max_concurrency", "icicle_msm_precompute_factor", "gpu_msm_precompute_calls",
            "gpu_msm_precomputed_chunk_calls", "gpu_msm_precomputed_cache_hits",
            "icicle_msm_precomputed_basis_device_upper_bytes", "icicle_msm_precomputed_basis_device_bytes",
            "icicle_msm_precomputed_basis_device_peak_bytes", "msm_operations", "gpu_msm_chunk_calls", "ntt_calls",
            "cpu_fft_calls", "synchronized_calls", "device_id", "gpu_ntt_elements", "ntt_max_size",
            "ntt_workspace_capacity_bytes", "ntt_domain_capacity_bytes", "ntt_host_staging_capacity_bytes",
            "ntt_device_peak_upper_bound_bytes", "gpu_h_eval_groups", "gpu_h_eval_calls", "gpu_h_eval_rows", "gpu_h_eval_fallbacks",
            "gpu_h_eval_chunks", "gpu_h_eval_syncs", "gpu_h_transfer_syncs", "gpu_h_workspace_high_water_bytes",
            "gpu_h_local_memory_peak_upper_bytes", "gpu_h_resident_high_water_bytes",
        )]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            abort("native GPU prover record has invalid device/call counters")
        kernel_calls = record["gpu_msm_chunk_calls"] + record["gpu_msm_precompute_calls"] + record["ntt_calls"] + record["gpu_h_eval_chunks"]
        sync_calls = record["gpu_msm_chunk_calls"] + record["gpu_msm_precompute_calls"] + record["ntt_calls"] + record["gpu_h_eval_syncs"] + record["gpu_h_transfer_syncs"]
        max_size = record["ntt_max_size"]
        plan = native_ntt_plan
        if plan is None:
            abort("formal GPU prover record has no authenticated NTT Admission plan")
        expected_workspace = max_size * 96
        expected_domain = 0 if max_size == 0 else 32 * (max_size + 1 + 64 + 6)
        expected_host = max_size * 32
        expected_device_peak = 0 if max_size == 0 else 128 * max_size + 10464
        planned_ntt_below_minimum = plan["icicle_ntt_capacity_elements"] < (1 << 16)
        minimum_ntt_elements = (1 << 16) * record["ntt_calls"]
        maximum_ntt_elements = max_size * record["ntt_calls"]
        if (
            record["schema_version"] != 5
            or record["proof_sha256"] != result["proof_sha256"]
            or record["admission_sha256"] != sha256(proof.parent / "admission.json")
            or record["native_execution_policy"] != FORMAL_GPU_EXECUTION_POLICY
            or record["icicle_msm_max_concurrency"] != plan["icicle_msm_max_concurrency"]
            or record["icicle_msm_precompute_factor"] != plan["icicle_msm_precompute_factor"]
            or record["icicle_msm_precomputed_basis_device_upper_bytes"] != plan["icicle_msm_precomputed_basis_device_upper_bytes"]
            or record["icicle_msm_precomputed_basis_device_bytes"] > record["icicle_msm_precomputed_basis_device_upper_bytes"]
            or record["icicle_msm_precomputed_basis_device_peak_bytes"] < record["icicle_msm_precomputed_basis_device_bytes"]
            or record["icicle_msm_precomputed_basis_device_peak_bytes"] > record["icicle_msm_precomputed_basis_device_upper_bytes"]
            or record["msm_operations"] <= 0
            or record["gpu_msm_chunk_calls"] < record["msm_operations"]
            or record["gpu_msm_precomputed_chunk_calls"] > record["gpu_msm_chunk_calls"]
            or record["gpu_msm_precompute_calls"] > record["gpu_msm_precomputed_chunk_calls"]
            or record["gpu_msm_precomputed_cache_hits"] > record["gpu_msm_precomputed_chunk_calls"]
            or record["gpu_msm_precomputed_chunk_calls"] != record["gpu_msm_precompute_calls"] + record["gpu_msm_precomputed_cache_hits"]
            or (planned_ntt_below_minimum and record["ntt_calls"] != 0)
            or (record["ntt_calls"] == 0 and record["gpu_ntt_elements"] != 0)
            or (record["ntt_calls"] > 0 and max_size < (1 << 16))
            or (record["ntt_calls"] > 0 and record["gpu_ntt_elements"] < minimum_ntt_elements)
            or (record["ntt_calls"] > 0 and record["gpu_ntt_elements"] > maximum_ntt_elements)
            or record["gpu_h_eval_calls"] <= 0
            or record["gpu_h_eval_groups"] != record["gpu_h_eval_calls"] + record["gpu_h_eval_fallbacks"]
            or record["gpu_h_eval_rows"] <= 0
            or record["gpu_h_eval_chunks"] <= 0
            or not (0 < record["gpu_h_eval_syncs"] <= record["gpu_h_eval_chunks"])
            or record["gpu_h_workspace_high_water_bytes"] > 64 * plan["icicle_ntt_capacity_elements"]
            or record["gpu_h_local_memory_peak_upper_bytes"] > plan["h_kernel_local_memory_upper_bytes"]
            or record["gpu_h_resident_high_water_bytes"] > plan["icicle_h_resident_device_upper_bytes"]
            or (max_size != 0 and max_size & (max_size - 1) != 0)
            or max_size > plan["icicle_ntt_capacity_elements"]
            or record["ntt_workspace_capacity_bytes"] != expected_workspace
            or record["ntt_domain_capacity_bytes"] != expected_domain
            or record["ntt_host_staging_capacity_bytes"] != expected_host
            or record["ntt_device_peak_upper_bound_bytes"] != expected_device_peak
            or expected_workspace > plan["icicle_ntt_device_workspace_upper_bytes"]
            or expected_domain > plan["icicle_ntt_domain_cache_upper_bytes"]
            or expected_device_peak > plan["icicle_ntt_device_peak_upper_bytes"]
            or expected_host > plan["icicle_ntt_host_staging_upper_bytes"]
            or kernel_calls <= 0
            or record["synchronized_calls"] != sync_calls
            or record["evidence"] != result["gpu_evidence"]
            or result["gpu_evidence"]["gpu_kernel_invocation_count"] != kernel_calls
        ):
            abort("native GPU prover record is not synchronized and bound to the verified proof")
        if formal:
            if record["device_id"] != result["selected_cuda_device"]["logical_device_id"]:
                abort("native proof logical device ordinal does not match the selected physical CUDA UUID")
            if sha256(proof.parent / "gpu_prover_record.json") != result["gpu_prover_record_sha256"]:
                abort("native GPU prover record identity is stale")

    source_identity = source_tree_hash() if expected_source_tree_sha256 is None else expected_source_tree_sha256
    resource_identity = resource_manifest_hash() if expected_resource_manifest_sha256 is None else expected_resource_manifest_sha256
    if result["source_tree_sha256"] != source_identity or result["resource_manifest_sha256"] != resource_identity:
        abort(f"{path} source/resource identity is stale")
    result["_result_path"] = path.relative_to(ROOT).as_posix()
    return result


def validate_gpu_evidence_from_result(result: dict[str, Any]) -> None:
    validate_gpu_evidence_document(result["gpu_evidence"], backend=result.get("native_backend", result.get("backend")))


def check_pair_consistency(cells: list[dict[str, Any]]) -> None:
    if not cells or any("backend" not in cell for cell in cells) or {cell["backend"] for cell in cells} != set(BACKENDS):
        return
    by_dataset = {(cell["dataset"], cell["backend"]): cell for cell in cells}
    for dataset in DATASETS:
        cpu = by_dataset[(dataset, "cpu")]
        gpu = by_dataset[(dataset, "gpu")]
        for field in PAIR_FIELDS:
            if cpu[field] != gpu[field]:
                abort(f"CPU/GPU relation mismatch for {dataset}: field={field}")


def build_matrix_document(cells: list[dict[str, Any]], *, datasets: tuple[str, ...], source_sha256: str, resource_sha256: str, collected_at_utc: str, formal_session_id: str | None, backends: tuple[str, ...] | None) -> dict[str, Any]:
    document = {"schema_version": 1, "baseline": "ezkl", "collection_status": "COLLECTED_PENDING_FINAL_VALIDATION", "relation_id": CANONICAL_RELATION, "datasets": list(datasets), "cell_count": len(cells), "source_tree_sha256": source_sha256, "resource_manifest_sha256": resource_sha256, "collected_at_utc": collected_at_utc, "cells": cells}
    if formal_session_id is not None:
        document["formal_session_id"] = formal_session_id
    else:
        if backends is None:
            abort("local matrix construction requires explicit backends")
        document["backends"] = list(backends)
        document["pair_consistency_checked"] = set(backends) == set(BACKENDS)
    return document


def _validate_native_pk_v4_read_backing_budget(
    admission: dict[str, Any], successor_edges: int, prove_budget: dict[str, Any]
) -> dict[str, int]:
    constraint_system = admission.get("constraint_system")
    permutation_domain = admission.get("domain_size")
    permutation_columns = constraint_system.get("num_permutation_columns") if isinstance(constraint_system, dict) else None
    if (
        isinstance(permutation_domain, bool)
        or not isinstance(permutation_domain, int)
        or permutation_domain <= 0
        or isinstance(permutation_columns, bool)
        or not isinstance(permutation_columns, int)
        or permutation_columns <= 0
    ):
        abort("formal native admission has invalid permutation bitmap geometry")
    permutation_cells = permutation_domain * permutation_columns
    geometry = native_pk_v4_read_backing_geometry(permutation_cells, successor_edges)
    expected = {
        "native_pk_read_backing_required_bytes": geometry["target_bitmap_backing_bytes"],
        "max_total_backing_bytes": geometry["target_bitmap_backing_bytes"],
        "native_pk_read_target_bitmap_allocation_bitmap_upper_bytes": geometry[
            "target_bitmap_allocation_bitmap_bytes"
        ],
    }
    if any(
        isinstance(prove_budget.get(name), bool)
        or not isinstance(prove_budget.get(name), int)
        or prove_budget.get(name) != value
        for name, value in expected.items()
    ):
        abort("native prove budget does not match the v4 PK reader's target-bitmap backing")
    return geometry


def _pending_formal_cell(
    path: Path,
    *,
    dataset: str,
    session_id: str,
    source_sha256: str,
    resource_sha256: str,
) -> dict[str, Any]:
    """Load only enough identity to construct the pending matrix; final validation owns acceptance."""
    result = load_json(path, f"pending formal result cell {path}")
    if (
        result.get("dataset") != dataset
        or result.get("status") != "VERIFY_OK"
        or result.get("formal_session_id") != session_id
        or result.get("native_verify_result") is not True
        or result.get("source_tree_sha256") != source_sha256
        or result.get("resource_manifest_sha256") != resource_sha256
        or path.parent.name != dataset
    ):
        abort(f"{path} is not a current-session independently verified result for the pending matrix")
    missing = REQUIRED_RESULT_KEYS - set(result)
    if missing:
        abort(f"{path} is missing pending formal result fields: {sorted(missing)}")
    _load_compatible_resume_context(path, result, session_id=session_id)
    result["_result_path"] = path.relative_to(ROOT).as_posix()
    return result


def main(
    args: argparse.Namespace,
    *,
    authenticated_context: Any | None = None,
    formal_session_id: str | None = None,
) -> int:
    selected_datasets = normalize_selected_datasets(getattr(args, "datasets", None))
    selected_backends = tuple(args.local_backends or ())
    formal = not selected_backends
    if formal:
        if not isinstance(args.session_id, str) or not FORMAL_SESSION_ID.fullmatch(args.session_id):
            abort("formal collection requires the active --session-id")
        if authenticated_context is not None:
            if formal_session_id != args.session_id:
                abort("sealed result collection session does not match its command-line session")
            require_authenticated_formal_context(
                authenticated_context,
                session_id=args.session_id,
                dataset=DATASETS[0],
                config=ROOT / "resources/canonical/configs" / f"{DATASETS[0]}_secure.cfg",
            )
    elif args.session_id is not None:
        abort("--session-id cannot be combined with local-only collection")
    elif len(selected_datasets) != 1 or len(selected_backends) != 1:
        abort("local-only collection is restricted to one tiny dataset and one backend")
    runs_root = local_path(args.runs_root, "runs-root")
    output = local_path(args.output, "collector output")
    reject_forbidden_path(output, "collector output")
    if not formal and output == ROOT / "results/formal_matrix.json":
        abort("local-only collection cannot write the formal result matrix path")
    if output.is_symlink():
        abort(f"collector output must not be a symlink: {output}")
    if formal:
        require_formal_session_selection_binding(
            args.session_id,
            runs_root=runs_root,
            matrix_path=output,
            selected_datasets=selected_datasets,
        )
    if formal and authenticated_context is not None:
        source_sha256 = authenticated_context.source_tree_sha256
        resource_sha256 = authenticated_context.resource_manifest_sha256
    else:
        source_sha256 = source_tree_hash()
        resource_sha256 = resource_manifest_hash()
    cells: list[dict[str, Any]] = []
    for dataset in selected_datasets:
        for backend in selected_backends if not formal else (None,):
            result_path = runs_root / dataset / "result.json" if formal else runs_root / dataset / str(backend) / "result.json"
            reject_forbidden_path(result_path, f"{dataset}/{backend} result")
            if formal:
                cell = _pending_formal_cell(
                    result_path,
                    dataset=dataset,
                    session_id=args.session_id,
                    source_sha256=source_sha256,
                    resource_sha256=resource_sha256,
                )
            else:
                cell = validate_cell(result_path, expected_session_id=None)
            if cell["dataset"] != dataset or (not formal and cell["backend"] != backend):
                abort(f"result identity does not match its collection path: {result_path}")
            cells.append(cell)
    expected_count = len(selected_datasets) if formal else len(selected_datasets) * len(selected_backends)
    if len(cells) != expected_count:
        abort("collector did not obtain the requested result cells")
    check_pair_consistency(cells)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = build_matrix_document(cells, datasets=selected_datasets, source_sha256=source_sha256, resource_sha256=resource_sha256, collected_at_utc=datetime.now(timezone.utc).isoformat(), formal_session_id=args.session_id if formal else None, backends=None if formal else selected_backends)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    mode = "formal dataset results" if formal else "local-only results"
    print(f"collected {len(cells)} verified {mode} into {output.relative_to(ROOT)}; final validation remains required")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(print_error(exc))
