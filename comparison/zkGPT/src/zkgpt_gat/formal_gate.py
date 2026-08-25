from __future__ import annotations

import json
import pathlib
from typing import Any, Dict


CAPABILITY_SCHEMA = "zkGPT_gat_formal_capabilities_v2"
_REQUIRED_FIELDS = {
    "schema",
    "protocol_id",
    "formal_experiment_eligible",
    "privacy_profile_status",
    "proof_task",
    "privacy_policy",
    "input_privacy",
    "output_mode",
    "rounding_mode",
    "static_table_id",
    "quant_cfg_id",
    "static_table_descriptor_schema",
    "static_table_descriptor_path",
    "static_table_descriptor_digest",
    "native_components",
    "gpu_fail_closed",
    "cpu_formal_path",
    "gpu_formal_path",
    "gpu_proof_message_contract",
    "gpu_proof_execution_contract",
    "gpu_proof_execution_semantic_id",
    "gpu_cuda_kernel",
    "implemented_reference_code",
    "required_privacy_extension_profile_id",
    "blockers",
}


def baseline_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def validate_capabilities(value: Any) -> list[str]:
    """Validate the one capability gate used by every formal entry point.

    This intentionally validates the execution contracts as a single
    fail-closed gate.  A source preflight must not be able to reinterpret a
    capability manifest differently from the runner or proof builder.
    """
    if not isinstance(value, dict):
        return ["manifest is not a JSON object"]
    errors: list[str] = []
    missing = sorted(_REQUIRED_FIELDS - set(value))
    if missing:
        errors.append("missing fields: " + ", ".join(missing))
    extra = sorted(set(value) - _REQUIRED_FIELDS)
    if extra:
        errors.append("unsupported fields: " + ", ".join(extra))
    expected = {
        "schema": CAPABILITY_SCHEMA,
        "protocol_id": "GAT-ZKML-zkGPT",
        "privacy_profile_status": "supported_multi_session_masked_full",
        "proof_task": "node_classification_full_gat",
        "privacy_policy": "model_and_trace_hidden",
        "input_privacy": "public_graph_public_features",
        "output_mode": "public_logits",
        "rounding_mode": "nearest_ties_away",
        "static_table_id": "tables:lrelu+elu+exp+range",
        "quant_cfg_id": "exact_integer_gat_v1",
        "static_table_descriptor_schema": "zkGPT_static_tables_descriptor_v1",
        "static_table_descriptor_path": "resources/static_tables/exact_integer_gat_v1.json",
        "static_table_descriptor_digest": "84ad4f14fa96f66b7f9f6a49fdee986530d16c07aa7b2a1794499170ba0118d3",
        "cpu_formal_path": "native_zkgpt_gkr_lasso_matmul_hyrax",
        "gpu_formal_path": "native_zkgpt_cuda_gkr_lasso_matmul_hyrax",
        "gpu_proof_message_contract": "zkGPT_gpu_proof_messages_v3",
        "gpu_proof_execution_contract": "zkGPT_gpu_proof_execution_v3",
        "gpu_proof_execution_semantic_id": "zkgpt_cuda_authors_bn254_fr_proof_messages_v3",
        "gpu_cuda_kernel": "zkgpt_native_bn254_fr_batch_v3",
        "implemented_reference_code": "formal_zkgpt_gat_native_v1",
        "required_privacy_extension_profile_id": "masked_full_multi_session_v1",
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            errors.append(f"{field} != {wanted}")
    if value.get("formal_experiment_eligible") is not True:
        errors.append("formal_experiment_eligible must be true for the supported implementation")
    if value.get("native_components") != [
        "native_hyrax_mle",
        "native_lasso",
        "native_matmul_sumcheck",
        "native_gkr",
        "native_bn254_cuda_proof_primitives",
    ]:
        errors.append("native_components does not match the supported native proof path")
    if value.get("gpu_fail_closed") != {
        "backend": "gpu",
        "requires_cuda_kernel": True,
        "requires_proof_message_evidence": True,
        "rejects_cpu_fallback": True,
        "failure_on_unavailable": True,
        "execution_schema": "zkGPT_gpu_proof_execution_v3",
        "execution_semantic_id": "zkgpt_cuda_authors_bn254_fr_proof_messages_v3",
    }:
        errors.append("gpu_fail_closed declaration does not match the supported GPU path")
    if value.get("blockers") != []:
        errors.append("blockers must be exactly [] for the supported implementation")
    return errors


def load_capabilities(path: pathlib.Path | None = None) -> Dict[str, Any]:
    manifest_path = path or (baseline_root() / "configs" / "formal_capabilities.json")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"unable to parse zkGPT formal capability manifest: {exc}") from exc
    errors = validate_capabilities(value)
    if errors:
        raise RuntimeError("invalid zkGPT formal capability manifest: " + "; ".join(errors))
    return value


def capabilities() -> Dict[str, Any]:
    return load_capabilities()


def formal_gate_status(path: pathlib.Path | None = None) -> Dict[str, Any]:
    manifest_path = path or (baseline_root() / "configs" / "formal_capabilities.json")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        return {"status": "BLOCKED", "errors": [f"unable to parse zkGPT formal capability manifest: {exc}"], "capabilities": None}
    errors = validate_capabilities(value)
    if errors:
        return {"status": "BLOCKED", "errors": errors, "capabilities": value if isinstance(value, dict) else None}
    return {"status": "READY", "errors": [], "capabilities": value}


def require_formal_experiment() -> None:
    gate = formal_gate_status()
    if gate["status"] != "READY":
        raise RuntimeError("zkGPT formal capability gate blocked: " + "; ".join(gate["errors"]))
