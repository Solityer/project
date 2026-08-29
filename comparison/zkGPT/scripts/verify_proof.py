#!/usr/bin/env python3
"""Replay a zkGPT proof only after external statement authentication.

All model/layout/table identities are supplied through an external
``VerifierContext``.  The serialized artifact is treated as untrusted proof
messages and metadata; its self-description cannot select the context.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Dict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The external verifier's context is intentionally a small public whitelist.
# ``model_manifest`` means the canonical architecture projection only; raw
# checkpoint tensor/value files are never verifier inputs.
PUBLIC_CONTEXT_FILES = ["config", "model_manifest", "trusted_registry_public", "public_y"]

from zkgpt_gat.proof import _verify_formal  # noqa: E402
from zkgpt_gat.config import load_app_config  # noqa: E402
from zkgpt_gat.gpu_execution import validate_gpu_execution_evidence  # noqa: E402
from zkgpt_gat.formal_gate import require_formal_experiment  # noqa: E402
from zkgpt_gat.privacy import verify_privacy_profile  # noqa: E402
from zkgpt_gat.registry import canonical_registry_public_path  # noqa: E402
from zkgpt_gat.resources import (  # noqa: E402
    STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH,
    canonical_dataset_name,
    required_resource_paths,
    resolve_local,
    validate_static_table_descriptor,
)
from zkgpt_gat.result_validation import (  # noqa: E402
    ResultValidationError,
    required_artifact_file,
    validate_artifact_inventory,
    validate_canonical_cell_dir_from_path,
)
from zkgpt_gat.util import canonical_json, digest_json, write_json  # noqa: E402
from zkgpt_gat.verifier_context import ContextError, VerifierContext, build_verifier_context  # noqa: E402


def _read(artifact_dir: pathlib.Path, name: str) -> Dict[str, Any]:
    path = required_artifact_file(artifact_dir, name)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _read_canonical_object(artifact_dir: pathlib.Path, name: str) -> Dict[str, Any]:
    """Read one public proof object with its canonical wire encoding intact.

    Production artifacts are emitted by ``write_json``.  Requiring the same
    sorted, UTF-8 JSON representation here makes duplicate/reordered privacy
    envelopes observable at the independent-verifier boundary instead of
    allowing two byte-distinct files to collapse to the same Python mapping.
    The parsed object is still checked by the protocol-specific verifier.
    """
    path = required_artifact_file(artifact_dir, name)
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not canonical JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    expected = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if raw != expected:
        raise ValueError(f"{path} is not canonically serialized")
    return value


def _validate_static_table_closure(
    project_root: pathlib.Path,
    metadata: Dict[str, Any],
    proof_payload: Dict[str, Any],
    fixed_lookup_tables: Dict[str, Any],
    lasso: Dict[str, Any],
) -> tuple[bool, list[str]]:
    """Check the fixed-table file as an independent trust-boundary input."""
    errors: list[str] = []
    try:
        descriptor = validate_static_table_descriptor(project_root)
    except (OSError, ValueError, TypeError) as exc:
        return False, [f"canonical_static_table_descriptor_invalid:{exc}"]
    if not isinstance(fixed_lookup_tables, dict):
        errors.append("fixed_lookup_tables_not_object")
        fixed_lookup_tables = {}
    if proof_payload.get("FixedLookupTables") != fixed_lookup_tables:
        errors.append("proof_payload_fixed_lookup_tables_mismatch")
    if metadata.get("static_table_registry") != fixed_lookup_tables:
        errors.append("metadata_static_table_registry_mismatch")
    if metadata.get("static_table_registry_digest") != digest_json(fixed_lookup_tables):
        errors.append("metadata_static_table_registry_digest_mismatch")
    if lasso.get("fixed_table_registry") != fixed_lookup_tables:
        errors.append("lasso_fixed_table_registry_mismatch")
    if lasso.get("fixed_table_registry_digest") != digest_json(fixed_lookup_tables):
        errors.append("lasso_fixed_table_registry_digest_mismatch")
    if metadata.get("static_table_descriptor_schema") != descriptor.get("schema"):
        errors.append("metadata_static_table_descriptor_schema_mismatch")
    if metadata.get("static_table_content_digest") != descriptor.get("content_digest"):
        errors.append("metadata_static_table_content_digest_mismatch")
    if metadata.get("static_table_descriptor_digest") != descriptor.get("descriptor_digest"):
        errors.append("metadata_static_table_descriptor_digest_mismatch")
    if metadata.get("static_table_descriptor_path") != STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH:
        errors.append("metadata_static_table_descriptor_path_mismatch")
    expected_id = descriptor.get("static_tables_id")
    expected_quant = descriptor.get("quant_cfg_id")
    expected_schema = descriptor.get("schema")
    expected_content = descriptor.get("content_digest")
    expected_descriptor = descriptor.get("descriptor_digest")
    if set(fixed_lookup_tables) != {"T_LReLU", "T_ELU", "T_exp", "T_range"}:
        errors.append("fixed_lookup_tables_family_set_invalid")
    for table_id, table in fixed_lookup_tables.items():
        if not isinstance(table, dict):
            errors.append(f"fixed_lookup_table_not_object:{table_id}")
            continue
        expected = {
            "table_id": table_id,
            "static_table_id": expected_id,
            "quant_cfg_id": expected_quant,
            "static_table_descriptor_schema": expected_schema,
            "static_table_content_digest": expected_content,
            "static_table_descriptor_digest": expected_descriptor,
        }
        for key, value in expected.items():
            if table.get(key) != value:
                errors.append(f"fixed_lookup_table_{table_id}_{key}_mismatch")
    return not errors, errors


def _validate_gpu_artifact_closure(
    artifact_dir: pathlib.Path,
    metadata: Dict[str, Any],
    transcript: Dict[str, Any],
    proof_payload: Dict[str, Any],
) -> tuple[bool, list[str]]:
    """Require all three public GPU records to be byte-equivalent claims."""
    errors: list[str] = []
    try:
        hot_path = _read(artifact_dir, "gpu_proof_hot_path.json")
        backend = _read(artifact_dir, "gpu_backend.json")
    except (OSError, ValueError) as exc:
        return False, [f"gpu_evidence_missing:{exc}"]
    contribution = proof_payload.get("GPU_proof_contribution")
    if not isinstance(contribution, dict):
        errors.append("proof_payload_gpu_contribution_missing")
    else:
        if contribution != hot_path:
            errors.append("proof_payload_gpu_contribution_mismatch")
        if contribution != backend:
            errors.append("gpu_backend_mismatch")
    valid, validation_errors = validate_gpu_execution_evidence(
        hot_path,
        session_nonce=str(metadata.get("session_nonce", "")),
        final_transcript_digest=str(transcript.get("final_digest", "")),
    )
    if not valid:
        errors.extend(f"gpu_hot_path:{item}" for item in validation_errors)
    return not errors, errors


def _canonical_public_y(project_root: pathlib.Path, config_path: pathlib.Path, public_y_path: pathlib.Path) -> pathlib.Path:
    """Require every verifier invocation to use the one baseline-local Y file.

    The proof artifact is untrusted and cannot nominate a replacement public
    output.  The config and CLI argument are both checked against the
    canonical resource layout, including the relative config spelling, before
    the independent verifier context is constructed.
    """
    root = pathlib.Path(project_root).expanduser().resolve()
    config = pathlib.Path(config_path).expanduser().resolve()
    if config != root and root not in config.parents:
        raise ValueError(f"config must stay inside baseline root: {config}")
    entries = load_app_config(config, root).entries
    dataset = canonical_dataset_name(entries.get("dataset", ""))
    expected = required_resource_paths(root, dataset)["public_y_path"]
    configured = resolve_local(root, entries.get("public_y_path", ""), "config public_y_path")
    if configured != expected or entries.get("public_y_path", "") != str(expected.relative_to(root)):
        raise ValueError(f"config public_y_path must be canonical baseline-local path: {expected.relative_to(root)}")
    supplied = pathlib.Path(public_y_path).expanduser().resolve()
    if supplied != expected:
        raise ValueError(f"--public-y must equal canonical baseline-local path: {expected}")
    if not expected.is_file():
        raise FileNotFoundError(f"canonical baseline-local public Y is missing: {expected}")
    return expected


def _strict_baseline_root(raw: str | pathlib.Path) -> pathlib.Path:
    """Authenticate the one lexical baseline root accepted by this verifier.

    The command-line root is a trust-boundary selector, so accepting a path
    that merely resolves to the baseline would permit aliases such as a
    symlink, a relative path, or a ``..`` spelling.  Normalize separators and
    trailing components lexically, reject dot components before normalization,
    then require both the normalized spelling and the resolved path to equal
    the module's canonical baseline root.
    """
    raw_text = os.fspath(raw)
    if not isinstance(raw_text, str):
        raise ValueError("--project-root must be a filesystem path string")
    supplied = pathlib.Path(raw_text)
    if not supplied.is_absolute():
        raise ValueError("--project-root must be absolute")
    if any(part in {".", ".."} for part in supplied.parts):
        raise ValueError("--project-root must not contain '.' or '..' path components")
    lexical = pathlib.Path(os.path.abspath(os.path.normpath(raw_text)))
    if str(lexical) != str(ROOT):
        raise ValueError(f"--project-root must equal canonical zkGPT baseline root: {ROOT}")

    current = pathlib.Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"--project-root contains a symlink component: {current}")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"--project-root cannot be resolved: {lexical}") from exc
    if resolved != ROOT:
        raise ValueError(f"--project-root resolves outside canonical baseline root: {resolved}")
    return lexical


def _invoke_formal_verifier(
    *,
    metadata: Dict[str, Any],
    commitments: Dict[str, Any],
    transcript: Dict[str, Any],
    lasso: Dict[str, Any],
    matrix: Dict[str, Any],
    gkr: Dict[str, Any],
    combine: Dict[str, Any],
    openings: Dict[str, Any],
    privacy_profile: Dict[str, Any],
    gpu_claim: Dict[str, Any] | None,
    backend: str,
    context: VerifierContext,
) -> Dict[str, Any]:
    """Call the strict proof verifier API with the rebuilt context."""
    # Keep the call keyworded against the real proof verifier signature.  A
    # missing serialized PrivacyProfile must never shift gpu_claim/backend
    # into the wrong positional slots or silently weaken the relation.
    return {
        **_verify_formal(
            metadata=metadata,
            commitments=commitments,
            transcript=transcript,
            lasso=lasso,
            matrix=matrix,
            gkr=gkr,
            combine=combine,
            openings=openings,
            privacy_profile=privacy_profile,
            gpu_claim=gpu_claim,
            backend=backend,
            external_context=context.proof_context(),
            public_graph=context.graph,
        ),
        "external_context_mode": "external_context",
    }


def verify_artifact_directory(artifact_dir: pathlib.Path, context: VerifierContext) -> Dict[str, Any]:
    metadata = _read(artifact_dir, "metadata.json")
    commitments = _read(artifact_dir, "commitments.json")
    transcript = _read(artifact_dir, "transcript.json")
    lasso = _read(artifact_dir, "lookup_proof.json")
    matrix = _read(artifact_dir, "matrix_sumcheck_proof.json")
    gkr = _read(artifact_dir, "gkr_proof.json")
    combine = _read(artifact_dir, "combine_proof.json")
    openings = _read(artifact_dir, "hyrax_mle_openings.json")
    privacy_profile = _read_canonical_object(artifact_dir, "privacy_profile.json")
    proof_payload = _read_canonical_object(artifact_dir, "proof_payload.json")
    fixed_lookup_tables = _read(artifact_dir, "fixed_lookup_tables.json")
    backend = str(metadata.get("backend", ""))
    if backend not in {"cpu", "gpu"}:
        raise ValueError(f"invalid proof backend: {backend!r}")

    external_metadata = context.validate_metadata(metadata)
    if external_metadata["status"] != "VERIFY_OK":
        return {
            "status": "VERIFY_FAILED",
            "reason": "external verifier context rejected artifact metadata",
            "checks": {"external_context": False},
            "external_context": external_metadata,
        }
    external_payload = context.validate_proof_payload(metadata, proof_payload)
    external_families = context.validate_subproof_families(lasso, matrix, gkr)
    if external_payload["status"] != "VERIFY_OK" or external_families["status"] != "VERIFY_OK":
        return {
            "status": "VERIFY_FAILED",
            "reason": "external verifier context rejected public output or subproof family descriptors",
            "checks": {"external_context": True, "external_public_y": external_payload["status"] == "VERIFY_OK", "external_families": external_families["status"] == "VERIFY_OK"},
            "external_context": external_metadata,
            "external_public_y": external_payload,
            "external_families": external_families,
        }

    static_table_ok, static_table_errors = _validate_static_table_closure(
        context.project_root, metadata, proof_payload, fixed_lookup_tables, lasso,
    )
    gpu_closure_ok, gpu_closure_errors = (True, [])
    if backend == "gpu":
        gpu_closure_ok, gpu_closure_errors = _validate_gpu_artifact_closure(
            artifact_dir, metadata, transcript, proof_payload,
        )
    gpu_claim = _read(artifact_dir, "gpu_proof_hot_path.json") if backend == "gpu" else None
    privacy_errors: list[str] = []
    payload_profile = proof_payload.get("PrivacyProfile")
    if payload_profile != privacy_profile:
        privacy_errors.append("privacy_profile_file_payload_mismatch")
    if not isinstance(payload_profile, dict):
        privacy_errors.append("privacy_profile_payload_not_object")
    else:
        if proof_payload.get("privacy_profile_digest") != payload_profile.get("profile_digest"):
            privacy_errors.append("privacy_profile_payload_digest_mismatch")
        if privacy_profile.get("profile_digest") != digest_json({key: value for key, value in privacy_profile.items() if key != "profile_digest"}):
            privacy_errors.append("privacy_profile_file_digest_invalid")
        if transcript.get("privacy_profile_digest") != privacy_profile.get("profile_digest"):
            privacy_errors.append("privacy_profile_transcript_digest_mismatch")
        if privacy_profile.get("pre_profile_transcript_digest") != transcript.get("pre_profile_transcript_digest"):
            privacy_errors.append("privacy_profile_pre_transcript_digest_mismatch")
        if proof_payload.get("transcript_digest") != digest_json(transcript):
            privacy_errors.append("proof_payload_transcript_digest_mismatch")
        commitment_items = commitments.get("items", {})
        if not isinstance(commitment_items, dict):
            commitment_items = {}
        expected_com_in = [commitment_items.get(name) for name in ("com_W", "com_Aux", "com_L")]
        if proof_payload.get("Com_in") != expected_com_in:
            privacy_errors.append("proof_payload_commitment_order_or_binding_mismatch")
        profile_check = verify_privacy_profile(
            metadata,
            commitments,
            proof_payload,
            pre_profile_transcript_digest=str(transcript.get("pre_profile_transcript_digest", "")),
            components={
                "lasso": lasso,
                "matmul": matrix,
                "gkr": gkr,
                "input_combine": combine,
                "terminal_openings": openings,
            },
        )
        if profile_check.get("status") != "VERIFY_OK":
            privacy_errors.append("privacy_profile_crypto_binding_invalid")
    if privacy_errors:
        result = {
            "status": "VERIFY_FAILED",
            "reason": "privacy profile or public proof binding rejected before formal replay",
            "checks": {},
            "subproof_verification": {},
        }
    else:
        result = _invoke_formal_verifier(
            metadata=metadata,
            commitments=commitments,
            transcript=transcript,
            lasso=lasso,
            matrix=matrix,
            gkr=gkr,
            combine=combine,
            openings=openings,
            privacy_profile=privacy_profile,
            gpu_claim=gpu_claim,
            backend=backend,
            context=context,
        )
    result["external_context"] = external_metadata
    result["external_public_y"] = external_payload
    result["external_families"] = external_families
    result["context_digest"] = context.context_digest
    checks = dict(result.get("checks", {}))
    checks["external_context"] = True
    checks["external_public_y"] = True
    checks["external_families"] = True
    checks["fixed_lookup_table_closure"] = static_table_ok
    checks["gpu_evidence_closure"] = gpu_closure_ok
    checks["privacy_profile_file"] = not privacy_errors
    if static_table_errors:
        result["fixed_lookup_table_errors"] = static_table_errors
    if gpu_closure_errors:
        result["gpu_evidence_errors"] = gpu_closure_errors
    if privacy_errors:
        result["privacy_profile_errors"] = privacy_errors
    checks = result.get("checks", {})
    required_payload_keys = {
        "proof_type", "M_pub_digest", "Y_digest", "Com_in", "Pi_Lasso",
        "FixedLookupTables", "Pi_MatMul", "Pi_GKR", "Pi_combine",
        "Open_in", "PrivacyProfile", "privacy_profile_digest", "component_artifacts", "transcript_digest",
    }
    allowed_payload_keys = required_payload_keys | {"GPU_proof_contribution"}
    payload_keys_ok = set(proof_payload) == allowed_payload_keys or (
        backend == "cpu" and set(proof_payload) == required_payload_keys
    )

    def _contains_raw_private(value: Any) -> bool:
        forbidden = {"raw_witness", "private_witness", "witness_values", "witness_arrays", "trace_arrays", "private_trace"}
        if isinstance(value, dict):
            for key, child in value.items():
                key_lower = str(key).lower()
                if (
                    key_lower in forbidden
                    or "witness" in key_lower
                    or "trace" in key_lower
                    or key_lower in {"blind", "blinder", "secret"}
                ):
                    return True
                if _contains_raw_private(child):
                    return True
            return False
        if isinstance(value, list):
            return any(_contains_raw_private(child) for child in value)
        return False

    raw_private_payload = _contains_raw_private(proof_payload)
    proof_schema_scan = payload_keys_ok and not raw_private_payload
    checks["proof_payload_schema"] = proof_schema_scan
    result["checks"] = checks
    if result.get("status") != "VERIFY_OK" or not static_table_ok or not gpu_closure_ok or privacy_errors or not proof_schema_scan:
        result["status"] = "VERIFY_FAILED"
    verifier_input_files = {
        "metadata.json", "commitments.json", "transcript.json", "lookup_proof.json",
        "matrix_sumcheck_proof.json", "gkr_proof.json", "combine_proof.json",
        "hyrax_mle_openings.json", "proof_payload.json",
    }
    private_input_files = [name for name in verifier_input_files if any(token in name.lower() for token in ("witness", "trace", "private"))]
    verifier_input_boundary = {
        "proof_artifact_files": sorted(verifier_input_files),
        "public_context_files": list(PUBLIC_CONTEXT_FILES),
        "private_input_files_read": sorted(private_input_files),
    }
    masking_ok = (
        checks.get("privacy_profile") is True
        and isinstance(result.get("subproof_verification", {}).get("privacy_profile"), dict)
        and result["subproof_verification"]["privacy_profile"].get("status") == "VERIFY_OK"
        and isinstance(proof_payload.get("Open_in"), dict)
        and isinstance(proof_payload.get("PrivacyProfile"), dict)
        and proof_payload.get("privacy_profile_digest") == proof_payload["PrivacyProfile"].get("profile_digest")
    )
    contribution = {
        "lookup": checks.get("lasso") is True,
        "matmul": checks.get("matrix_sumcheck") is True,
        "gkr": checks.get("gkr") is True,
        "combine": checks.get("combine") is True,
        "terminal_openings": checks.get("hyrax_openings") is True,
        "public_output": checks.get("external_public_y") is True and bool(metadata.get("public_output_digest")),
    }
    full_gat = result.get("status") == "VERIFY_OK" and all(contribution.values()) and checks.get("external_context") is True and proof_schema_scan
    witness_included = raw_private_payload
    privacy_profile = str(metadata.get("privacy_profile_status", ""))
    privacy_id = str(metadata.get("privacy_extension_profile_id", ""))
    dim_cfg = metadata.get("dim_cfg")
    if not isinstance(dim_cfg, dict):
        dim_cfg = {}
    result["schema"] = "zkGPT_independent_verification_v4"
    result["verified_fields"] = {
        "status": result.get("status"),
        "protocol_id": metadata.get("protocol_id"),
        "proof_backend_id": metadata.get("proof_backend_id"),
        "baseline_identity": "zkGPT",
        "dataset": canonical_dataset_name(str(metadata.get("dataset", ""))),
        "backend": backend,
        "model_arch_id": metadata.get("model_arch_id", ""),
        "model_param_id": metadata.get("model_param_id", ""),
        "quant_cfg_id": metadata.get("quant_cfg_id", ""),
        "L": dim_cfg.get("L"),
        "hidden_profile": dim_cfg.get("hidden_profile", []),
        "d_in_profile": dim_cfg.get("d_in_profile", []),
        "K_out": dim_cfg.get("K_out"),
        "C": dim_cfg.get("C"),
        "full_dataset": metadata.get("full_dataset") is True,
        "proof_scope": "full_gat_zkgpt_segment_commitments_lookup_matmul_gkr_openings" if full_gat else "",
        "full_gat_forward_proved": full_gat,
        "final_logits_bound": contribution["public_output"] and contribution["gkr"] and contribution["combine"],
        "privacy_extension_profile_id": privacy_id,
        "privacy_profile_status": privacy_profile,
        "verifier_reads_witness": bool(private_input_files),
        "witness_included_in_proof": witness_included,
        "public_output_digest": metadata.get("public_output_digest"),
        "proof_contribution": contribution,
        "gpu_hot_path": backend == "gpu" and checks.get("gpu_hot_path") is True,
        "proof_schema_scan": proof_schema_scan,
        "privacy_masking_verified": masking_ok,
        "verifier_input_boundary": verifier_input_boundary,
        "task_type": metadata.get("task_type", ""),
        "public_input_layout": metadata.get("public_input_layout", ""),
        "model_registry_digest": metadata.get("model_registry_digest", ""),
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a serialized zkGPT proof with an external verifier context")
    parser.add_argument("--project-root", type=pathlib.Path, required=True, help="trusted project root used to resolve config/model/data")
    parser.add_argument("--config", type=pathlib.Path, required=True, help="trusted canonical zkGPT config")
    parser.add_argument("--trusted-registry-public", "--registry-public", dest="registry_public", type=pathlib.Path, required=True, help="trusted model registry public JSON outside artifact")
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True, help="serialized proof artifact directory")
    parser.add_argument("--public-y", type=pathlib.Path, required=True, help="independent public Y JSON/NPY/text file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact_dir: pathlib.Path | None = None
    try:
        # The formal capability gate is the first trust-boundary operation in
        # every verifier process.  Reject blocked source configurations before
        # opening a caller-selected project, config, registry, or artifact.
        require_formal_experiment()
        project_root = _strict_baseline_root(args.project_root)
        # Authenticate the lexical canonical artifact directory before reading
        # any proof bytes.  A final run manifest is deliberately not required:
        # this verifier writes the independent report that the runner later
        # binds into the fresh VERIFY_OK manifest.
        artifact_dir = validate_canonical_cell_dir_from_path(
            project_root, args.artifact_dir,
        )
        path_backend = artifact_dir.parent.name
        path_dataset = artifact_dir.name
        # The fresh verifier is allowed to create the independent report, but
        # it must see an exact producer-side inventory before reading any
        # proof bytes.  The shared checker still rejects every extra entry,
        # nested work directory, alias, hardlink, or special file.
        validate_artifact_inventory(
            artifact_dir, path_backend, require_independent=False, require_manifest=False
        )
        metadata = _read(artifact_dir, "metadata.json")
        metadata_backend = str(metadata.get("backend", ""))
        metadata_dataset = canonical_dataset_name(str(metadata.get("dataset", "")))
        if metadata_backend != path_backend or metadata_dataset != path_dataset:
            raise ValueError(
                "metadata dataset/backend does not match canonical artifact path"
            )
        config_entries = load_app_config(args.config.expanduser().resolve(), project_root).entries
        model_param_id = config_entries.get("model_param_id", "")
        expected_registry = canonical_registry_public_path(project_root, model_param_id)
        supplied_registry = args.registry_public.expanduser().resolve()
        if supplied_registry != expected_registry.resolve():
            raise ValueError(
                f"--trusted-registry-public must equal canonical config/model registry: {expected_registry}"
            )
        public_y = _canonical_public_y(project_root, args.config, args.public_y)
        context = build_verifier_context(project_root, args.config, args.registry_public, artifact_dir, public_y)
        result = verify_artifact_directory(artifact_dir, context)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ContextError, ResultValidationError) as exc:
        result = {"status": "VERIFY_FAILED", "reason": str(exc), "checks": {"external_context": False}}
    if artifact_dir is not None and artifact_dir.is_dir():
        output = artifact_dir / "independent_verification.json"
        write_json(output, result)
        try:
            validate_artifact_inventory(
                artifact_dir,
                artifact_dir.parent.name,
                require_independent=True,
                require_manifest=False,
            )
        except (OSError, ValueError, ResultValidationError) as exc:
            # Never publish a verifier-success report for a directory that
            # changed shape while verification was running.
            result = {
                "status": "VERIFY_FAILED",
                "reason": f"post-verification artifact inventory rejected: {exc}",
                "checks": {"artifact_inventory": False},
            }
            write_json(output, result)
    shown_artifact = artifact_dir if artifact_dir is not None else args.artifact_dir.expanduser()
    print(f"zkgpt_proof_replay={result.get('status', 'VERIFY_FAILED')} artifact={shown_artifact} context={result.get('context_digest', 'unavailable')}")
    return 0 if result.get("status") == "VERIFY_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
