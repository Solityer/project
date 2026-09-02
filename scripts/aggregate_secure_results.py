#!/usr/bin/env python3
"""Aggregate the ten verified formal runs into one source-locked result set.

This command is intentionally fail-closed.  It does not create any result
file until exactly one complete warm CPU and CUDA run has passed all manifest,
proof, provenance, and security checks for each canonical dataset.  Controlled
smoke/stage runs therefore cannot be promoted into the paper result directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import pathlib
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple


_PROVENANCE_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "build_provenance.py"
_PROVENANCE_SPEC = importlib.util.spec_from_file_location(
    "gatzk_build_provenance", _PROVENANCE_PATH
)
if _PROVENANCE_SPEC is None or _PROVENANCE_SPEC.loader is None:
    raise RuntimeError(f"cannot load provenance implementation: {_PROVENANCE_PATH}")
_PROVENANCE = importlib.util.module_from_spec(_PROVENANCE_SPEC)
_PROVENANCE_SPEC.loader.exec_module(_PROVENANCE)


DATASET_ORDER = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
CANONICAL_DATASETS = frozenset(DATASET_ORDER)
# The verifier timing boundary is part of the result contract.  Every accepted
# manifest must use this one timing definition so the paper table has a single
# measurement meaning.
CANONICAL_TIMING_SCHEMA = "exclusive_commit_prove_protocol_verify"
EXPECTED_MODEL_ARCH = {
    "cora": "gat_family_L3_hid8x8-8x8_kout2_c7",
    "citeseer": "gat_family_L3_hid8x8-8x8_kout2_c6",
    "pubmed": "gat_family_L3_hid8x8-8x8_kout2_c3",
    "ppi": "gat_family_L3_hid8x8-8x8_kout2_c121",
    "ogbn_arxiv": "gat_family_L3_hid8x8-8x8_kout2_c40",
}
EXPECTED_MODEL_PARAM = {
    "cora": "cora_planetoid_seed11",
    "citeseer": "citeseer_planetoid_seed11",
    "pubmed": "pubmed_planetoid_seed11",
    "ppi": "ppi_gat_seed11",
    "ogbn_arxiv": "ogbn_arxiv_seed11",
}
EXPECTED_SRS_ID = {
    "cora": "cora_formal_srs",
    "citeseer": "citeseer_formal_srs",
    "pubmed": "pubmed_formal_srs",
    "ppi": "ppi_formal_srs",
    "ogbn_arxiv": "ogbn_arxiv_formal_srs",
}
EXPECTED_D_IN = {
    "cora": [1433, 64],
    "citeseer": [3703, 64],
    "pubmed": [500, 64],
    "ppi": [50, 64],
    "ogbn_arxiv": [128, 64],
}
EXPECTED_CLASSES = {
    "cora": 7,
    "citeseer": 6,
    "pubmed": 3,
    "ppi": 121,
    "ogbn_arxiv": 40,
}
EXPECTED_INPUT_PREPROCESS_ID = {
    "cora": "row_normalize",
    "citeseer": "row_normalize",
    "pubmed": "row_normalize",
    "ppi": "row_normalize",
    "ogbn_arxiv": "ogbn_arxiv_raw_float32",
}
EXPECTED_SRS_DEGREE = {
    "cora": 16777215,
    "citeseer": 67108863,
    "pubmed": 67108863,
    "ppi": 33554431,
    "ogbn_arxiv": 134217727,
}
EXPECTED_SRS_DOMAINS = {
    "cora": {"C": 16, "FH": 4194304, "N": 4096, "cat": 128, "d": 16, "edge": 32768, "in": 2048},
    "citeseer": {"C": 8, "FH": 16777216, "N": 4096, "cat": 128, "d": 16, "edge": 32768, "in": 4096},
    "pubmed": {"C": 8, "FH": 16777216, "N": 32768, "cat": 128, "d": 16, "edge": 262144, "in": 512},
    "ppi": {"C": 128, "FH": 4194304, "N": 65536, "cat": 64, "d": 16, "edge": 8388608, "in": 64},
    "ogbn_arxiv": {"C": 64, "FH": 33554432, "N": 262144, "cat": 64, "d": 16, "edge": 16777216, "in": 256},
}
REQUIRED_HASHES = (
    "proof_hash",
    "quant_config_hash",
    "srs_manifest_hash",
    "public_metadata_hash",
    "static_table_hash",
    "public_input_digest",
    "topology_digest",
    "source_tree_hash",
    "source_manifest_hash",
    "binary_hash",
    "build_identity",
    "build_receipt_hash",
    "verifier_report_hash",
    "trusted_registry_digest",
    "verifier_registry_digest",
    "security_budget_ledger_digest",
)

# Actual same-dataset semantic fields that must agree between CPU and CUDA
# manifests.  public_metadata_hash is intentionally excluded because
# proof_nonce differs.  Build/binary/report provenance remains mandatory and
# is independently validated for each manifest, but those artifacts are
# backend-specific and therefore must not be equal across CPU and CUDA.
PAIR_CONSISTENCY_FIELDS = (
    "protocol_schema_id",
    "timing_schema",
    "instance_schema_hash",
    "dim_cfg",
    "model_params",
    "quant_config",
    "static_tables",
    "public_output",
    "secure_trace_logits_hash",
    "reference_quant_logits_hash",
    "proof_public_output_hash",
    "public_input_digest",
    "topology_digest",
    "input_preprocess_id",
    "source_tree_hash",
    "trusted_registry_digest",
    "verifier_registry_digest",
    "security_budget_ledger_digest",
    "security_budget_exact",
    "tamper_test_case_map",
    "config",
    "config_sha256",
    "quant_config_hash",
    "srs_manifest_hash",
    "static_table_hash",
    "srs_id",
    "model_arch",
    "srs_degree_bound",
    "source_manifest_hash",
)

REGISTRY_INSPECTION_KEYS = {
    "format",
    "artifact_sha256",
    "registry_digest",
    "registry_id",
    "model_arch_id",
    "model_param_id",
    "quant_cfg_id",
    "static_table_id",
    "srs_id",
    "protocol_schema_id",
    "instance_schema_hash",
    "srs_manifest_hash",
    "model_layout_digest",
    "quant_chain_digest",
    "static_table_digest",
    "static_table_generator_id",
    "static_table_spec_hash",
    "range_and_scale_certificate",
    "integer_bound_ledger_digest",
    "public_input_digest",
    "topology_digest",
    "security_budget_ledger_digest",
    "security_budget_exact",
    "integer_bound_ledger_count",
    "security_budget_ledger_count",
    "model_entry_count",
    "static_entry_count",
    "security_budget_passed",
}

VERIFIER_REPORT_KEYS = {
    "format",
    "trusted_registry_digest",
    "authenticated_registry_digest",
    "proof_bin_sha256",
    "parsed_output_digest",
    "proof_public_output_hash",
    "parsed_output_count",
    "verify_result",
    "reference_consistency_passed",
    "tamper_tests_passed",
    "tamper_test_case_map",
    "security_budget_passed",
    "security_budget_ledger_digest",
    "security_budget_exact",
    "public_input_digest",
    "topology_digest",
    "source_tree_hash",
    "binary_hash",
    "build_identity",
}

REQUIRED_TAMPER_CASES = frozenset(
    {
        "codec_tag",
        "codec_length",
        "codec_trailing",
        "public_output",
        "metadata",
        "schema",
        "ids",
        "dynamic_commitment",
        "quotient_commitment",
        "opening",
        "reblind_commitment",
        "static_registry",
        "model_registry",
        "reference_mismatch",
        "private_opening",
        "route_padding",
        "route_query",
        "lookup_inverse",
        "lookup_multiplicity",
        "psq",
        "shift",
        "logup",
        "aggregate",
        "average",
        "division",
        "transcript_challenge",
        "transcript_value",
        "srs_metadata",
        "srs_consistency",
        "srs_resource_identity",
        "srs_resource_compromised",
        "proof_scope",
        "observation_ledger",
        "random_identity_ledger",
        "integer_bound_ledger",
        "integer_bound_certificate",
        "security_budget_ledger",
        "security_budget_exact",
        "public_input_digest",
    }
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_assignment(value: str, flag: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{flag} expects dataset=run_manifest.json_or_dir")
    dataset, payload = value.split("=", 1)
    dataset = dataset.strip()
    payload = payload.strip()
    require(dataset in CANONICAL_DATASETS, f"{flag} has unsupported dataset {dataset!r}")
    require(payload != "", f"{flag} has an empty path for {dataset}")
    return dataset, payload


def as_bool(value: Any, field: str, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise ValueError(f"{context}: {field} must be a boolean or true/false string")


def as_int(value: Any, field: str, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context}: {field} must be an integer")
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {field} must be an integer") from exc
    return parsed


def as_float(value: Any, field: str, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context}: {field} must be numeric")
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {field} must be numeric") from exc
    require(math.isfinite(parsed), f"{context}: {field} must be finite")
    return parsed


def nonempty_string(value: Any, field: str, context: str) -> str:
    require(isinstance(value, str) and value.strip() != "", f"{context}: {field} is missing")
    return value.strip()


def require_canonical_timing_schema(manifest: Dict[str, Any], context: str) -> str:
    timing_schema = nonempty_string(
        manifest.get("timing_schema"), "timing_schema", context
    )
    require(
        timing_schema == CANONICAL_TIMING_SCHEMA,
        f"{context}: unsupported timing_schema {timing_schema!r}; expected {CANONICAL_TIMING_SCHEMA!r}",
    )
    return timing_schema


def sha256_string(value: Any, field: str, context: str) -> str:
    encoded = nonempty_string(value, field, context)
    require(
        len(encoded) == 64
        and all(char in "0123456789abcdef" for char in encoded),
        f"{context}: {field} must be lowercase SHA-256",
    )
    return encoded


def resolve_input_path(project_root: pathlib.Path, value: str) -> pathlib.Path:
    candidate = pathlib.Path(value)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def load_manifest(project_root: pathlib.Path, value: str) -> Tuple[pathlib.Path, pathlib.Path, Dict[str, Any]]:
    input_path = resolve_input_path(project_root, value)
    manifest_path = input_path / "run_manifest.json" if input_path.is_dir() else input_path
    require(manifest_path.is_file(), f"run manifest does not exist: {manifest_path}")
    run_dir = manifest_path.parent
    proof_path = run_dir / "proof.bin"
    require(proof_path.is_file(), f"run manifest has no proof.bin: {proof_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read run manifest {manifest_path}: {exc}") from exc
    require(isinstance(manifest, dict), f"run manifest must be a JSON object: {manifest_path}")
    nonempty_string(
        manifest.get("input_preprocess_id"),
        "input_preprocess_id",
        f"run manifest/{manifest_path.parent}",
    )
    return run_dir, proof_path, manifest


def load_json_object(path: pathlib.Path, context: str) -> Dict[str, Any]:
    require(path.is_file(), f"{context}: missing JSON artifact {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}: cannot read JSON artifact {path}: {exc}") from exc
    require(isinstance(value, dict), f"{context}: JSON artifact must be an object: {path}")
    return value


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_local_artifact(
    run_dir: pathlib.Path,
    value: Any,
    field: str,
    context: str,
) -> pathlib.Path:
    """Resolve a run-local artifact without accepting path aliases."""
    relative = nonempty_string(value, field, context)
    candidate = pathlib.Path(relative)
    require(
        not candidate.is_absolute()
        and candidate.parent == pathlib.Path("."),
        f"{context}: {field} must be a run-local basename",
    )
    path = run_dir / candidate
    require(path.is_file() and not path.is_symlink(), f"{context}: missing {field}: {path}")
    return path


def canonical_tamper_case_map(value: Any, field: str, context: str) -> Dict[str, str]:
    """Parse the complete production tamper map without accepting aliases."""
    encoded = nonempty_string(value, field, context)
    parsed: Dict[str, str] = {}
    for item in encoded.split(","):
        require(item.count("=") == 1, f"{context}: {field} has malformed case {item!r}")
        name, result = (part.strip() for part in item.split("=", 1))
        require(name in REQUIRED_TAMPER_CASES, f"{context}: {field} has unknown case {name!r}")
        require(name not in parsed, f"{context}: {field} has duplicate case {name!r}")
        require(
            result in {"reject", "serialization_rejected"},
            f"{context}: {field} case {name!r} is not fail-closed",
        )
        parsed[name] = result
    require(
        set(parsed) == set(REQUIRED_TAMPER_CASES),
        f"{context}: {field} is incomplete",
    )
    return parsed


def _resolve_regular_file(project_root: pathlib.Path, value: Any, field: str, context: str) -> pathlib.Path:
    path = resolve_input_path(project_root, nonempty_string(value, field, context))
    require(path.is_file() and not path.is_symlink(), f"{context}: {field} must be a regular non-symlink file")
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"{context}: {field} must be inside the project root") from exc
    return path


def _resolve_inspector(project_root: pathlib.Path, value: str, context: str) -> pathlib.Path:
    path = _resolve_regular_file(project_root, value, "registry_inspector", context)
    return path


def inspect_registration_artifact(
    project_root: pathlib.Path,
    dataset: str,
    registry_value: str,
    expected_digest: str,
    inspector_value: str,
    context: str,
) -> Dict[str, Any]:
    """Require an externally anchored registry and independently inspect it.

    The expected digest is supplied independently of the run manifest.  The
    inspector is a separately built short-lived utility which invokes the
    formal registration loader; aggregate does not accept a manifest field as
    a substitute for either input.
    """
    registry_path = _resolve_regular_file(
        project_root, registry_value, "verifier_registry_path", context
    )
    expected = sha256_string(expected_digest, "trusted_registry_digest", context)
    artifact_hash = sha256_file(registry_path)
    require(
        artifact_hash == expected,
        f"{context}: registration artifact bytes do not match the external trusted digest",
    )
    inspector = _resolve_inspector(project_root, inspector_value, context)
    try:
        completed = subprocess.run(
            [
                str(inspector),
                "--registry",
                str(registry_path),
                "--trusted-digest",
                expected,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"{context}: independent registry inspection failed: {exc}") from exc
    require(
        completed.returncode == 0,
        f"{context}: independent registry inspection failed: {completed.stderr.strip() or 'non-zero exit'}",
    )
    try:
        inspection = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context}: registry inspector did not emit JSON") from exc
    require(isinstance(inspection, dict), f"{context}: registry inspection must be a JSON object")
    require(
        set(inspection) == REGISTRY_INSPECTION_KEYS,
        f"{context}: registry inspection has an unexpected or missing field",
    )
    require(
        inspection["format"] == "zkgat_registry_inspection",
        f"{context}: registry inspection format mismatch",
    )
    require(
        inspection["artifact_sha256"] == artifact_hash
        and inspection["registry_digest"] == expected,
        f"{context}: registry inspection digest mismatch",
    )
    for field in (
        "registry_digest",
        "artifact_sha256",
        "protocol_schema_id",
        "instance_schema_hash",
        "srs_manifest_hash",
        "model_layout_digest",
        "quant_chain_digest",
        "static_table_digest",
        "static_table_spec_hash",
        "range_and_scale_certificate",
        "integer_bound_ledger_digest",
        "public_input_digest",
        "topology_digest",
        "security_budget_ledger_digest",
    ):
        sha256_string(inspection.get(field), field, context)
    for field in (
        "registry_id",
        "model_arch_id",
        "model_param_id",
        "quant_cfg_id",
        "static_table_id",
        "srs_id",
        "static_table_generator_id",
        "security_budget_exact",
    ):
        nonempty_string(inspection.get(field), field, context)
    for field in (
        "integer_bound_ledger_count",
        "security_budget_ledger_count",
        "model_entry_count",
        "static_entry_count",
    ):
        require(
            as_int(inspection.get(field), field, context) > 0,
            f"{context}: {field} must be positive",
        )
    require(
        as_bool(inspection.get("security_budget_passed"), "security_budget_passed", context),
        f"{context}: authenticated SecurityBudgetLedger did not pass",
    )
    budget_exact = nonempty_string(inspection["security_budget_exact"], "security_budget_exact", context)
    require(
        "/" in budget_exact
        and all(
            part.isdigit() and (part == "0" or not part.startswith("0"))
            for part in budget_exact.split("/", 1)
        ),
        f"{context}: authenticated security budget is not canonical",
    )
    require(
        inspection["registry_id"] == EXPECTED_MODEL_PARAM[dataset],
        f"{context}: registry model identity is not canonical",
    )
    require(
        inspection["model_arch_id"] == EXPECTED_MODEL_ARCH[dataset],
        f"{context}: registry architecture identity is not canonical",
    )
    require(
        inspection["model_param_id"] == EXPECTED_MODEL_PARAM[dataset]
        and inspection["quant_cfg_id"] == "gat_l3_scale_chain"
        and inspection["static_table_id"] == "tables:lrelu+elu+exp+range"
        and inspection["srs_id"] == EXPECTED_SRS_ID[dataset],
        f"{context}: registry model/static/SRS identity is not canonical",
    )
    return inspection


def check_source_provenance(
    project_root: pathlib.Path,
    run_dir: pathlib.Path,
    run_manifest: Dict[str, Any],
    context: str,
) -> Dict[str, str]:
    """Recompute the source inventory and build receipt before aggregation."""
    source_manifest_path = _require_local_artifact(
        run_dir,
        run_manifest.get("source_manifest_path"),
        "source_manifest_path",
        context,
    )
    source_manifest = load_json_object(source_manifest_path, context)
    expected_source = _PROVENANCE.source_manifest_document(project_root)
    require(
        source_manifest == expected_source,
        f"{context}: source_manifest.json does not match the current source inventory",
    )
    source_manifest_hash = sha256_file(source_manifest_path)
    require(
        source_manifest_hash == nonempty_string(
            run_manifest.get("source_manifest_hash"),
            "source_manifest_hash",
            context,
        ),
        f"{context}: source_manifest_hash does not match source_manifest.json",
    )
    source_tree_hash = sha256_string(
        source_manifest.get("source_tree_hash"),
        "source_tree_hash",
        context,
    )
    require(
        source_tree_hash == _PROVENANCE.source_tree_hash(project_root),
        f"{context}: source_tree_hash does not match the current source tree",
    )
    require(
        nonempty_string(run_manifest.get("source_tree_hash"), "source_tree_hash", context)
        == source_tree_hash,
        f"{context}: run manifest source_tree_hash is not source_manifest-bound",
    )

    receipt_path = _require_local_artifact(
        run_dir,
        run_manifest.get("build_receipt_path"),
        "build_receipt_path",
        context,
    )
    receipt = load_json_object(receipt_path, context)
    expected_receipt_keys = {
        "format",
        "source_manifest_path",
        "source_manifest_hash",
        "source_inventory_id",
        "source_tree_hash",
        "compiler_id",
        "compiler_version",
        "build_type",
        "compile_flags",
        "binary_path",
        "binary_hash",
        "build_identity_algorithm",
        "build_identity",
    }
    require(
        set(receipt) == expected_receipt_keys,
        f"{context}: build_receipt.json has an unexpected or missing field",
    )
    require(receipt["format"] == "zkgat_build_receipt", f"{context}: build receipt format mismatch")
    require(receipt["source_manifest_path"] == "source_manifest.json", f"{context}: build receipt source manifest path mismatch")
    require(receipt["source_manifest_hash"] == source_manifest_hash, f"{context}: build receipt source manifest hash mismatch")
    require(receipt["source_inventory_id"] == _PROVENANCE.SOURCE_INVENTORY_ID, f"{context}: build receipt inventory id mismatch")
    require(receipt["source_tree_hash"] == source_tree_hash, f"{context}: build receipt source hash mismatch")
    require(receipt["build_identity_algorithm"] == _PROVENANCE.BUILD_IDENTITY_ALGORITHM, f"{context}: build identity algorithm mismatch")
    binary_path_value = nonempty_string(receipt.get("binary_path"), "binary_path", context)
    binary_path = pathlib.Path(binary_path_value)
    if not binary_path.is_absolute():
        binary_path = project_root / binary_path
    binary_path = binary_path.resolve()
    require(binary_path.is_file() and not binary_path.is_symlink(), f"{context}: build receipt binary is not a regular non-symlink file")
    binary_hash = sha256_file(binary_path)
    require(
        binary_hash == sha256_string(receipt.get("binary_hash"), "binary_hash", context),
        f"{context}: build receipt binary_hash does not match binary bytes",
    )
    identity = _PROVENANCE.build_identity(
        source_tree_hash,
        nonempty_string(receipt.get("compiler_id"), "compiler_id", context),
        nonempty_string(receipt.get("compiler_version"), "compiler_version", context),
        str(receipt.get("build_type", "")),
        str(receipt.get("compile_flags", "")),
        binary_hash,
    )
    require(identity == sha256_string(receipt.get("build_identity"), "build_identity", context), f"{context}: build_identity does not recompute")
    for field in ("source_manifest_hash", "source_tree_hash", "binary_hash", "build_identity"):
        require(run_manifest.get(field) == receipt[field], f"{context}: run manifest {field} is not build-receipt-bound")
    receipt_hash = sha256_file(receipt_path)
    require(receipt_hash == sha256_string(run_manifest.get("build_receipt_hash"), "build_receipt_hash", context), f"{context}: build_receipt_hash mismatch")

    return {
        "source_manifest_hash": source_manifest_hash,
        "source_tree_hash": source_tree_hash,
        "binary_hash": binary_hash,
        "build_identity": identity,
        "build_receipt_hash": receipt_hash,
    }


def check_verifier_report(
    run_dir: pathlib.Path,
    proof_path: pathlib.Path,
    run_manifest: Dict[str, Any],
    provenance: Dict[str, str],
    registration: Dict[str, Any],
    context: str,
) -> Dict[str, Any]:
    """Validate the independently parsed-proof verifier report binding."""
    report_path = _require_local_artifact(
        run_dir,
        run_manifest.get("verifier_report_path"),
        "verifier_report_path",
        context,
    )
    report = load_json_object(report_path, context)
    require(set(report) == VERIFIER_REPORT_KEYS, f"{context}: verifier_report.json has an unexpected or missing field")
    require(report["format"] == "zkgat_verifier_report", f"{context}: verifier report format mismatch")
    trusted = sha256_string(report.get("trusted_registry_digest"), "trusted_registry_digest", context)
    authenticated = sha256_string(report.get("authenticated_registry_digest"), "authenticated_registry_digest", context)
    require(trusted == authenticated, f"{context}: verifier report registry trust binding mismatch")
    require(trusted == registration["registry_digest"], f"{context}: verifier report registry digest does not match inspected artifact")
    require(trusted == sha256_string(run_manifest.get("trusted_registry_digest"), "trusted_registry_digest", context), f"{context}: verifier report trusted registry digest mismatch")
    require(trusted == sha256_string(run_manifest.get("verifier_registry_digest"), "verifier_registry_digest", context), f"{context}: verifier report verifier registry digest mismatch")
    proof_hash = sha256_file(proof_path)
    require(proof_hash == sha256_string(report.get("proof_bin_sha256"), "proof_bin_sha256", context), f"{context}: verifier report proof hash mismatch")
    require(proof_hash == nonempty_string(run_manifest.get("proof_hash"), "proof_hash", context), f"{context}: proof_hash is not bound to verifier report")
    parsed_digest = sha256_string(report.get("parsed_output_digest"), "parsed_output_digest", context)
    report_output_digest = sha256_string(report.get("proof_public_output_hash"), "proof_public_output_hash", context)
    require(parsed_digest == report_output_digest, f"{context}: verifier report output digests disagree")
    require(parsed_digest == sha256_string(run_manifest.get("proof_public_output_hash"), "proof_public_output_hash", context), f"{context}: parsed output digest is not bound to run manifest")
    require(as_int(report.get("parsed_output_count"), "parsed_output_count", context) > 0, f"{context}: verifier report parsed output is empty")
    require(as_bool(report.get("verify_result"), "verify_result", context), f"{context}: verifier report verify_result is not true")
    require(as_bool(report.get("reference_consistency_passed"), "reference_consistency_passed", context), f"{context}: verifier report reference consistency did not pass")
    require(as_bool(report.get("tamper_tests_passed"), "tamper_tests_passed", context), f"{context}: verifier report tamper tests did not pass")
    report_tamper_map = nonempty_string(report.get("tamper_test_case_map"), "tamper_test_case_map", context)
    canonical_tamper_case_map(report_tamper_map, "tamper_test_case_map", context)
    require(report_tamper_map == nonempty_string(run_manifest.get("tamper_test_case_map"), "tamper_test_case_map", context), f"{context}: verifier report tamper map is not manifest-bound")
    require(as_bool(report.get("security_budget_passed"), "security_budget_passed", context), f"{context}: verifier report security budget did not pass")
    require(report.get("security_budget_ledger_digest") == registration["security_budget_ledger_digest"], f"{context}: verifier report SecurityBudgetLedger digest mismatch")
    require(report.get("security_budget_ledger_digest") == sha256_string(run_manifest.get("security_budget_ledger_digest"), "security_budget_ledger_digest", context), f"{context}: verifier report budget digest is not manifest-bound")
    report_budget_exact = nonempty_string(report.get("security_budget_exact"), "security_budget_exact", context)
    require(report_budget_exact == registration["security_budget_exact"], f"{context}: verifier report exact budget mismatch")
    require(report_budget_exact == nonempty_string(run_manifest.get("security_budget_exact"), "security_budget_exact", context), f"{context}: verifier report exact budget is not manifest-bound")
    require(report.get("public_input_digest") == registration["public_input_digest"], f"{context}: verifier report public-input digest mismatch")
    require(report.get("public_input_digest") == sha256_string(run_manifest.get("public_input_digest"), "public_input_digest", context), f"{context}: verifier report public-input digest is not manifest-bound")
    require(report.get("topology_digest") == registration["topology_digest"], f"{context}: verifier report topology digest mismatch")
    require(report.get("topology_digest") == sha256_string(run_manifest.get("topology_digest"), "topology_digest", context), f"{context}: verifier report topology digest is not manifest-bound")
    require(report.get("source_tree_hash") == provenance["source_tree_hash"], f"{context}: verifier report source tree is not current-source-bound")
    require(report.get("binary_hash") == provenance["binary_hash"], f"{context}: verifier report binary is not build-receipt-bound")
    require(report.get("build_identity") == provenance["build_identity"], f"{context}: verifier report build identity is not build-receipt-bound")
    report_hash = sha256_file(report_path)
    require(report_hash == sha256_string(run_manifest.get("verifier_report_hash"), "verifier_report_hash", context), f"{context}: verifier_report_hash mismatch")
    return {
        "verifier_report_hash": report_hash,
        "trusted_registry_digest": trusted,
        "parsed_output_digest": parsed_digest,
        "tamper_tests_passed": True,
        "tamper_test_case_map": report_tamper_map,
        "reference_consistency_passed": True,
        "security_budget_passed": True,
        "security_budget_ledger_digest": report["security_budget_ledger_digest"],
        "security_budget_exact": report_budget_exact,
    }


def resolve_config(project_root: pathlib.Path, manifest: Dict[str, Any], context: str) -> pathlib.Path:
    config_value = nonempty_string(manifest.get("config"), "config", context)
    config_path = resolve_input_path(project_root, config_value)
    require(config_path.is_file(), f"{context}: config does not exist: {config_path}")
    try:
        config_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"{context}: config must be inside the project root") from exc
    return config_path


def load_config_entries(path: pathlib.Path, context: str) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"{context}: cannot read config {path}: {exc}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{context}: malformed config line {line_number}")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        require(key and value, f"{context}: malformed config line {line_number}")
        entries[key] = value
    return entries


def check_canonical_config(path: pathlib.Path, dataset: str, context: str) -> Dict[str, str]:
    entries = load_config_entries(path, context)
    expected_dataset = dataset
    expected_values = {
        "dataset": expected_dataset,
        "L": "3",
        "hidden_profile": "[8x8,8x8]",
        "d_in_profile": "[" + ",".join(str(value) for value in EXPECTED_D_IN[dataset]) + "]",
        "K_out": "2",
        "num_classes": str(EXPECTED_CLASSES[dataset]),
        "range_bits": str({"cora": 12, "citeseer": 12, "pubmed": 12, "ppi": 14, "ogbn_arxiv": 21}[dataset]),
        "model_arch_id": EXPECTED_MODEL_ARCH[dataset],
        "model_param_id": EXPECTED_MODEL_PARAM[dataset],
        "quant_cfg_id": "gat_l3_scale_chain",
        "input_preprocess_id": EXPECTED_INPUT_PREPROCESS_ID[dataset],
        "static_table_id": "tables:lrelu+elu+exp+range",
        "degree_bound_id": "auto",
        "checkpoint_bundle": f"artifacts/checkpoints/{dataset}_gat",
        "srs_id": EXPECTED_SRS_ID[dataset],
        "srs_path": f"artifacts/srs/{EXPECTED_SRS_ID[dataset]}",
    }
    for key, expected in expected_values.items():
        actual = entries.get(key, "")
        if key in {"hidden_profile", "d_in_profile"}:
            actual = actual.replace(" ", "")
            expected = expected.replace(" ", "")
        require(actual == expected, f"{context}: canonical config {key} mismatch")
    return entries


def check_manifest_config_binding(
    project_root: pathlib.Path,
    dataset: str,
    manifest: Dict[str, Any],
    context: str,
) -> Tuple[pathlib.Path, str, str]:
    """Bind the public manifest to the canonical config and preprocessing ID."""
    require(dataset in CANONICAL_DATASETS, f"{context}: unsupported canonical dataset {dataset!r}")
    expected_manifest_dataset = dataset
    require(
        manifest.get("dataset") == expected_manifest_dataset,
        f"{context}: manifest dataset is not {expected_manifest_dataset}",
    )
    config_path = resolve_config(project_root, manifest, context)
    expected_config = (
        project_root / "configs" / f"{dataset}_secure.cfg"
    ).resolve()
    require(
        config_path == expected_config,
        f"{context}: manifest config must be canonical {expected_config.relative_to(project_root)}",
    )
    config_entries = check_canonical_config(config_path, dataset, context)
    config_hash = sha256_file(config_path)
    recorded_config_hash = nonempty_string(
        manifest.get("config_sha256"), "config_sha256", context
    )
    require(
        recorded_config_hash == config_hash,
        f"{context}: config_sha256 does not match {config_path}",
    )
    config_preprocess_id = nonempty_string(
        config_entries.get("input_preprocess_id"), "input_preprocess_id", context
    )
    manifest_preprocess_id = nonempty_string(
        manifest.get("input_preprocess_id"), "input_preprocess_id", context
    )
    require(
        manifest_preprocess_id == config_preprocess_id,
        f"{context}: manifest input_preprocess_id is not canonical-config bound",
    )
    return config_path, recorded_config_hash, manifest_preprocess_id


def check_srs_artifacts(
    project_root: pathlib.Path,
    dataset: str,
    run_manifest: Dict[str, Any],
    context: str,
) -> Dict[str, Any]:
    srs_id = EXPECTED_SRS_ID[dataset]
    srs_root = (project_root / "artifacts" / "srs" / srs_id).resolve()
    srs_manifest_path = srs_root / "srs_manifest.json"
    srs_manifest = load_json_object(srs_manifest_path, context)
    require(srs_manifest.get("srs_id") == srs_id, f"{context}: SRS id is not canonical")
    recorded_srs_id = nonempty_string(run_manifest.get("srs_id"), "srs_id", context)
    require(recorded_srs_id == srs_id, f"{context}: run manifest srs_id is not canonical")
    require(srs_manifest.get("srs_type") == "trusted_srs", f"{context}: SRS type is not trusted_srs")
    require(as_int(srs_manifest.get("degree_bound"), "SRS degree_bound", context) == EXPECTED_SRS_DEGREE[dataset],
            f"{context}: SRS degree_bound is not canonical")
    raw_domains = srs_manifest.get("domain_sizes")
    require(isinstance(raw_domains, dict), f"{context}: SRS domain_sizes is missing")
    try:
        domain_sizes = {str(key): int(value) for key, value in raw_domains.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: SRS domain_sizes are malformed") from exc
    require(domain_sizes == EXPECTED_SRS_DOMAINS[dataset],
            f"{context}: SRS domain catalogue is not canonical")

    raw_manifest = srs_manifest_path.read_bytes()
    actual_hash = hashlib.sha256(raw_manifest).hexdigest()
    recorded_hash = nonempty_string(run_manifest.get("srs_manifest_hash"), "srs_manifest_hash", context)
    require(recorded_hash == actual_hash,
            f"{context}: srs_manifest_hash does not match {srs_manifest_path}")
    declared_hash = nonempty_string(srs_manifest.get("manifest_hash"), "manifest_hash", context)
    marker = b'  "manifest_hash"'
    marker_position = raw_manifest.rfind(marker)
    require(marker_position >= 0 and hashlib.sha256(raw_manifest[:marker_position]).hexdigest() == declared_hash,
            f"{context}: SRS manifest canonical hash field is invalid")
    return {"srs_manifest_hash": recorded_hash, "srs_id": srs_id}


def check_security(
    manifest: Dict[str, Any],
    proof_path: pathlib.Path,
    report: Dict[str, Any],
    registration: Dict[str, Any],
    context: str,
) -> Dict[str, Any]:
    """Validate security from independently bound report/registration data."""
    require(as_bool(manifest.get("verifier_reads_witness"), "verifier_reads_witness", context) is False,
            f"{context}: verifier_reads_witness must be false")
    require(as_int(manifest.get("private_plain_value_count"), "private_plain_value_count", context) == 0,
            f"{context}: private_plain_value_count must be zero")
    require(nonempty_string(manifest.get("proof_payload_audit_status"), "proof_payload_audit_status", context) == "passed",
            f"{context}: proof payload audit did not pass")
    require(as_bool(report.get("reference_consistency_passed"), "reference_consistency_passed", context), f"{context}: verifier report reference consistency did not pass")
    require(as_bool(report.get("tamper_tests_passed"), "tamper_tests_passed", context), f"{context}: verifier report tamper tests did not pass")
    for field in ("reference_consistency_passed", "tamper_tests_passed"):
        require(as_bool(manifest.get(field), field, context), f"{context}: {field} must be true")
    require(
        nonempty_string(manifest.get("tamper_test_case_map"), "tamper_test_case_map", context)
        == nonempty_string(report.get("tamper_test_case_map"), "tamper_test_case_map", context),
        f"{context}: manifest tamper map is not independently report-bound",
    )
    canonical_tamper_case_map(report.get("tamper_test_case_map"), "tamper_test_case_map", context)
    tamper_ms = as_float(manifest.get("tamper_tests_ms"), "tamper_tests_ms", context)
    require(tamper_ms > 0.0, f"{context}: tamper_tests_ms must be positive")

    for field in REQUIRED_HASHES:
        sha256_string(manifest.get(field), field, context)
    # There are exactly three output producers in the acceptance gate:
    # independent quantized reference, secure trace, and independently parsed
    # proof output.
    output_hashes = [
        sha256_string(manifest.get(field), field, context)
        for field in (
            "secure_trace_logits_hash",
            "reference_quant_logits_hash",
            "proof_public_output_hash",
        )
    ]
    require(
        len(set(output_hashes)) == 1,
        f"{context}: independent public-output hashes disagree",
    )
    require(as_bool(report.get("security_budget_passed"), "security_budget_passed", context), f"{context}: verifier report security budget did not pass")
    require(as_bool(manifest.get("security_budget_passed"), "security_budget_passed", context), f"{context}: security budget was not reconstructed and passed")
    require(
        manifest.get("security_budget_ledger_digest") == registration["security_budget_ledger_digest"]
        == report["security_budget_ledger_digest"],
        f"{context}: SecurityBudgetLedger digest is not registry/report bound",
    )
    budget_exact = nonempty_string(
        manifest.get("security_budget_exact"), "security_budget_exact", context
    )
    require(
        "/" in budget_exact
        and all(part.isdigit() and (part == "0" or not part.startswith("0"))
                for part in budget_exact.split("/", 1)),
        f"{context}: security_budget_exact is not canonical",
    )
    require(
        budget_exact == registration["security_budget_exact"] == report["security_budget_exact"],
        f"{context}: exact SecurityBudgetLedger budget is not registry/report bound",
    )
    tamper_case_map = nonempty_string(
        manifest.get("tamper_test_case_map"), "tamper_test_case_map", context
    )
    canonical_tamper_case_map(tamper_case_map, "tamper_test_case_map", context)
    actual_proof_hash = sha256_file(proof_path)
    require(manifest["proof_hash"] == actual_proof_hash,
            f"{context}: proof_hash does not match proof.bin ({actual_proof_hash})")
    return {
        "verifier_reads_witness": False,
        "private_plain_value_count": 0,
        "proof_payload_audit_status": "passed",
        "reference_consistency_passed": True,
        "tamper_tests_passed": True,
        "tamper_test_case_map": tamper_case_map,
        "security_budget_passed": True,
        "security_budget_ledger_digest": manifest["security_budget_ledger_digest"],
        "security_budget_exact": budget_exact,
        "tamper_tests_ms": tamper_ms,
        "proof_hash_audit_passed": True,
    }


def check_single_run_acceptance_state(
    manifest: Dict[str, Any],
    context: str,
) -> None:
    """Validate independent verifier/final states before accepting a run."""
    verifier_accepted = as_bool(
        manifest.get("verifier_accepted"),
        "verifier_accepted",
        context,
    )
    final_accepted = as_bool(
        manifest.get("final_formal_run_accepted"),
        "final_formal_run_accepted",
        context,
    )
    verified = as_bool(manifest.get("verified"), "verified", context)
    require(verifier_accepted, f"{context}: verifier_accepted must be true")
    require(final_accepted, f"{context}: final_formal_run_accepted must be true")
    require(
        verified is verifier_accepted,
        f"{context}: verified must equal verifier_accepted",
    )
    require(
        manifest.get("status") != "VERIFY_OK" or final_accepted,
        f"{context}: VERIFY_OK status requires final_formal_run_accepted",
    )
    require(manifest.get("status") == "VERIFY_OK", f"{context}: status is not VERIFY_OK")


def check_formal_acceptance_state(
    manifest: Dict[str, Any],
    context: str,
) -> None:
    """Validate every manifest-level gate required for an accepted run.

    The independent verifier report, proof bytes, registry, and source/build
    provenance are checked by their corresponding artifact gates below.  This
    helper deliberately covers only the acceptance state that is serialized in
    the public run manifest so benchmark exporters cannot promote a verifier
    pass whose reference audit, security audit, or final acceptance failed.
    """
    check_single_run_acceptance_state(manifest, context)
    for field in (
        "reference_consistency_passed",
        "tamper_tests_passed",
        "security_budget_passed",
    ):
        require(as_bool(manifest.get(field), field, context), f"{context}: {field} must be true")
    require(
        as_bool(manifest.get("verifier_reads_witness"), "verifier_reads_witness", context) is False,
        f"{context}: verifier_reads_witness must be false",
    )
    require(
        as_int(manifest.get("private_plain_value_count"), "private_plain_value_count", context) == 0,
        f"{context}: private_plain_value_count must be zero",
    )
    require(
        nonempty_string(manifest.get("proof_payload_audit_status"), "proof_payload_audit_status", context)
        == "passed",
        f"{context}: proof payload audit did not pass",
    )

    # Security acceptance is not represented by a single boolean.  Keep the
    # canonical tamper coverage, exact budget, and independent output hashes
    # attached to the acceptance gate as well.
    tamper_map = nonempty_string(
        manifest.get("tamper_test_case_map"), "tamper_test_case_map", context
    )
    canonical_tamper_case_map(tamper_map, "tamper_test_case_map", context)
    tamper_ms = as_float(manifest.get("tamper_tests_ms"), "tamper_tests_ms", context)
    require(tamper_ms > 0.0, f"{context}: tamper_tests_ms must be positive")
    sha256_string(
        manifest.get("security_budget_ledger_digest"),
        "security_budget_ledger_digest",
        context,
    )
    budget_exact = nonempty_string(
        manifest.get("security_budget_exact"), "security_budget_exact", context
    )
    require(
        "/" in budget_exact
        and all(
            part.isdigit() and (part == "0" or not part.startswith("0"))
            for part in budget_exact.split("/", 1)
        ),
        f"{context}: security_budget_exact is not canonical",
    )
    for field in REQUIRED_HASHES:
        sha256_string(manifest.get(field), field, context)
    output_hashes = [
        sha256_string(manifest.get(field), field, context)
        for field in (
            "secure_trace_logits_hash",
            "reference_quant_logits_hash",
            "proof_public_output_hash",
        )
    ]
    require(
        len(set(output_hashes)) == 1,
        f"{context}: independent public-output hashes disagree",
    )


def check_manifest(
    project_root: pathlib.Path,
    dataset: str,
    backend: str,
    value: str,
    registry_value: str | None = None,
    registry_digest: str | None = None,
    registry_inspector: str | None = None,
) -> Dict[str, Any]:
    run_dir, proof_path, manifest = load_manifest(project_root, value)
    context = f"{backend}/{dataset} ({run_dir})"
    require(
        not any(
            key.startswith("checkpoint") or key.startswith("tensor")
            for key in manifest
        ),
        f"{context}: public run manifest must not contain checkpoint/tensor provenance",
    )
    expected_manifest_dataset = dataset
    timing_schema = require_canonical_timing_schema(manifest, context)
    check_formal_acceptance_state(manifest, context)
    require(as_bool(manifest.get("full_dataset"), "full_dataset", context), f"{context}: full_dataset is not true")
    require(as_bool(manifest.get("is_full_dataset"), "is_full_dataset", context), f"{context}: is_full_dataset is not true")
    require(as_bool(manifest.get("stale_result"), "stale_result", context) is False,
            f"{context}: stale_result must be false")
    require(manifest.get("benchmark_mode") == "warm", f"{context}: benchmark_mode must be warm")

    expected_compute_backend = "cpu" if backend == "cpu" else "cuda_hotspots"
    expected_backend = "cpu" if backend == "cpu" else "gpu"
    require(manifest.get("backend") == expected_backend,
            f"{context}: backend must be {expected_backend}")
    require(manifest.get("compute_backend_name") == expected_compute_backend,
            f"{context}: compute_backend_name must be {expected_compute_backend}")
    requested_gpu = as_bool(manifest.get("gpu_backend_requested"), "gpu_backend_requested", context)
    require(requested_gpu is (backend == "cuda"), f"{context}: gpu_backend_requested is inconsistent")
    invocation_count = as_int(manifest.get("gpu_kernel_invocation_count"), "gpu_kernel_invocation_count", context)
    if backend == "cpu":
        require(invocation_count == 0, f"{context}: CPU run reports CUDA kernel invocations")
    else:
        runtime_present = as_bool(manifest.get("gpu_runtime_present"), "gpu_runtime_present", context)
        runtime_loaded = as_bool(manifest.get("gpu_runtime_loaded"), "gpu_runtime_loaded", context)
        require(runtime_present or runtime_loaded, f"{context}: CUDA runtime evidence is missing")
        require(invocation_count > 0, f"{context}: CUDA run reports no kernel invocations")
        nonempty_string(manifest.get("gpu_device"), "gpu_device", context)
        nonempty_string(manifest.get("gpu_acceleration_status"), "gpu_acceleration_status", context)

    config_path, recorded_config_hash, input_preprocess_id = check_manifest_config_binding(
        project_root, dataset, manifest, context
    )
    require(
        registry_value is not None
        and registry_digest is not None
        and registry_inspector is not None,
        f"{context}: external registry path, trusted digest, and inspector are required",
    )
    registration = inspect_registration_artifact(
        project_root,
        dataset,
        registry_value,
        registry_digest,
        registry_inspector,
        context,
    )
    require(
        manifest.get("trusted_registry_digest") == registration["registry_digest"],
        f"{context}: manifest trusted registry digest is not externally anchored",
    )
    require(
        manifest.get("verifier_registry_digest") == registration["registry_digest"],
        f"{context}: manifest verifier registry digest is not externally anchored",
    )
    require(
        manifest.get("verifier_registry_id") == registration["registry_id"],
        f"{context}: manifest verifier registry identity mismatch",
    )
    for manifest_field, registration_field in (
        ("protocol_schema_id", "protocol_schema_id"),
        ("instance_schema_hash", "instance_schema_hash"),
        ("srs_manifest_hash", "srs_manifest_hash"),
        ("public_input_digest", "public_input_digest"),
        ("topology_digest", "topology_digest"),
    ):
        require(
            manifest.get(manifest_field) == registration[registration_field],
            f"{context}: {manifest_field} is not registration-bound",
        )
    srs_info = check_srs_artifacts(project_root, dataset, manifest, context)
    provenance = check_source_provenance(project_root, run_dir, manifest, context)
    verifier_report = check_verifier_report(
        run_dir,
        proof_path,
        manifest,
        provenance,
        registration,
        context,
    )
    security = check_security(
        manifest,
        proof_path,
        verifier_report,
        registration,
        context,
    )
    timings = {
        "commitment_time_ms": as_float(manifest.get("commitment_time_ms"), "commitment_time_ms", context),
        "prove_time_ms": as_float(manifest.get("prove_time_ms"), "prove_time_ms", context),
        "verify_time_ms": as_float(manifest.get("verify_time_ms"), "verify_time_ms", context),
        "total_wall_clock_ms": as_float(manifest.get("total_wall_clock_ms"), "total_wall_clock_ms", context),
    }
    for field, value_number in timings.items():
        require(value_number >= 0.0, f"{context}: {field} must not be negative")
    proof_size = as_int(manifest.get("proof_size_bytes"), "proof_size_bytes", context)
    require(proof_size > 0 and proof_size == proof_path.stat().st_size,
            f"{context}: proof_size_bytes does not match proof.bin")

    row: Dict[str, Any] = {
        "dataset": dataset,
        "manifest_dataset": expected_manifest_dataset,
        "backend": backend,
        "status": "VERIFY_OK",
        "benchmark_mode": "warm",
        "timing_schema": timing_schema,
        "run_dir": str(run_dir.relative_to(project_root)),
        "config": str(config_path.relative_to(project_root)),
        "config_sha256": recorded_config_hash,
        "input_preprocess_id": input_preprocess_id,
        "protocol_schema_id": nonempty_string(manifest.get("protocol_schema_id"), "protocol_schema_id", context),
        "instance_schema_hash": nonempty_string(manifest.get("instance_schema_hash"), "instance_schema_hash", context),
        "dim_cfg": nonempty_string(manifest.get("dim_cfg"), "dim_cfg", context),
        "public_output": nonempty_string(manifest.get("public_output"), "public_output", context),
        "secure_trace_logits_hash": nonempty_string(manifest.get("secure_trace_logits_hash"), "secure_trace_logits_hash", context),
        "reference_quant_logits_hash": nonempty_string(manifest.get("reference_quant_logits_hash"), "reference_quant_logits_hash", context),
        "proof_public_output_hash": sha256_string(manifest.get("proof_public_output_hash"), "proof_public_output_hash", context),
        "public_input_digest": sha256_string(manifest.get("public_input_digest"), "public_input_digest", context),
        "topology_digest": sha256_string(manifest.get("topology_digest"), "topology_digest", context),
        "source_tree_hash": sha256_string(manifest.get("source_tree_hash"), "source_tree_hash", context),
        "source_manifest_hash": provenance["source_manifest_hash"],
        "binary_hash": sha256_string(manifest.get("binary_hash"), "binary_hash", context),
        "build_identity": sha256_string(manifest.get("build_identity"), "build_identity", context),
        "build_receipt_hash": provenance["build_receipt_hash"],
        "verifier_report_hash": verifier_report["verifier_report_hash"],
        "trusted_registry_digest": verifier_report["trusted_registry_digest"],
        "verifier_registry_digest": registration["registry_digest"],
        "proof_size_bytes": proof_size,
        "proof_hash": manifest["proof_hash"],
        "quant_config_hash": manifest["quant_config_hash"],
        "srs_manifest_hash": manifest["srs_manifest_hash"],
        "public_metadata_hash": manifest["public_metadata_hash"],
        "static_table_hash": manifest["static_table_hash"],
        "srs_id": srs_info["srs_id"],
        "model_arch": nonempty_string(manifest.get("model_arch"), "model_arch", context),
        "model_params": nonempty_string(manifest.get("model_params"), "model_params", context),
        "quant_config": nonempty_string(manifest.get("quant_config"), "quant_config", context),
        "static_tables": nonempty_string(manifest.get("static_tables"), "static_tables", context),
        "srs_degree_bound": nonempty_string(manifest.get("srs_degree_bound"), "srs_degree_bound", context),
        "gpu_runtime_present": as_bool(manifest.get("gpu_runtime_present"), "gpu_runtime_present", context),
        "gpu_runtime_loaded": as_bool(manifest.get("gpu_runtime_loaded"), "gpu_runtime_loaded", context),
        "gpu_kernel_invocation_count": invocation_count,
        "gpu_device": str(manifest.get("gpu_device", "")),
        "gpu_acceleration_status": str(manifest.get("gpu_acceleration_status", "")),
        "tamper_tests_ms": security["tamper_tests_ms"],
        **timings,
        **security,
    }
    require(row["srs_id"] == EXPECTED_SRS_ID[dataset],
            f"{context}: srs_id is not the canonical {EXPECTED_SRS_ID[dataset]}")
    require(row["model_arch"] == EXPECTED_MODEL_ARCH[dataset],
            f"{context}: model_arch is not the canonical architecture")
    require(row["model_params"].startswith("registry_sha256="),
            f"{context}: model_params is not a bound model registry")
    require(row["quant_config"].startswith("gat_l3_scale_chain|"),
            f"{context}: quant_config is not the canonical scale-chain binding")
    require(row["static_tables"].startswith("registry_sha256="),
            f"{context}: static_tables is not a bound static-table registry")
    return row


def compare_cpu_cuda_pair(
    dataset: str,
    cpu: Dict[str, Any],
    cuda: Dict[str, Any],
) -> None:
    """Fail closed unless every canonical pair field exists and is equal."""
    for field in PAIR_CONSISTENCY_FIELDS:
        require(field in cpu, f"{dataset}: CPU manifest is missing pair field {field}")
        require(field in cuda, f"{dataset}: CUDA manifest is missing pair field {field}")
        require(cpu[field] == cuda[field], f"{dataset}: CPU/CUDA {field} mismatch")


def ensure_pair_consistency(rows: Iterable[Dict[str, Any]]) -> bool:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["dataset"], {})[row["backend"]] = row
    for dataset in DATASET_ORDER:
        require(dataset in grouped, f"{dataset}: CPU/CUDA pair is missing")
        pair = grouped[dataset]
        require(set(pair) == {"cpu", "cuda"}, f"{dataset}: CPU/CUDA pair is incomplete")
        cpu = pair["cpu"]
        cuda = pair["cuda"]
        compare_cpu_cuda_pair(dataset, cpu, cuda)
    return True


def write_outputs(
    output_dir: pathlib.Path,
    rows: List[Dict[str, Any]],
    cpu_gpu_reference_consistency_passed: bool,
) -> None:
    # All validation, including pair consistency, has completed before this function is called.
    output_dir.mkdir(parents=True, exist_ok=True)
    security_rows = [
        {
            "dataset": row["dataset"],
            "backend": row["backend"],
            "run_dir": row["run_dir"],
            "proof_hash": row["proof_hash"],
            "proof_hash_audit_passed": row["proof_hash_audit_passed"],
            "verifier_reads_witness": row["verifier_reads_witness"],
            "private_plain_value_count": row["private_plain_value_count"],
            "proof_payload_audit_status": row["proof_payload_audit_status"],
            "reference_consistency_passed": row["reference_consistency_passed"],
            "tamper_tests_passed": row["tamper_tests_passed"],
            "tamper_test_case_map": row["tamper_test_case_map"],
            "security_budget_passed": row["security_budget_passed"],
            "security_budget_ledger_digest": row["security_budget_ledger_digest"],
            "security_budget_exact": row["security_budget_exact"],
            "tamper_tests_ms": row["tamper_tests_ms"],
        }
        for row in rows
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_mode": "warm",
        "run_count": len(rows),
        "rows": rows,
        "security_tests": {
            "all_passed": all(
                item["proof_hash_audit_passed"]
                and item["verifier_reads_witness"] is False
                and item["private_plain_value_count"] == 0
                and item["proof_payload_audit_status"] == "passed"
                and item["reference_consistency_passed"]
                and item["tamper_tests_passed"]
                and item["security_budget_passed"]
                for item in security_rows
            ) and cpu_gpu_reference_consistency_passed,
            "run_count": len(security_rows),
            "cpu_gpu_reference_consistency_passed": cpu_gpu_reference_consistency_passed,
            "checks": security_rows,
        },
    }
    (output_dir / "latest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_columns = [
        "dataset",
        "backend",
        "status",
        "benchmark_mode",
        "commitment_time_ms",
        "prove_time_ms",
        "verify_time_ms",
        "total_wall_clock_ms",
        "proof_size_bytes",
        "proof_hash",
        "config",
        "config_sha256",
        "source_tree_hash",
        "source_manifest_hash",
        "binary_hash",
        "build_identity",
        "build_receipt_hash",
        "verifier_report_hash",
        "trusted_registry_digest",
        "verifier_registry_digest",
        "quant_config_hash",
        "srs_manifest_hash",
        "static_table_hash",
        "srs_id",
        "gpu_runtime_present",
        "gpu_kernel_invocation_count",
        "gpu_device",
    ]
    with (output_dir / "latest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in csv_columns})

    lines = [
        "# 最新正式结果",
        "",
        "本表只由十个 `VERIFY_OK`、full-dataset、warm manifest 聚合生成；CPU/CUDA 每个数据集的配置、模型、量化、静态表和 SRS 标识已逐项配对校验。",
        "",
        "| 数据集 | 后端 | 状态 | commitment (ms) | prove (ms) | verify (ms) | proof bytes | GPU kernels |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['backend']} | {row['status']} | "
            f"{row['commitment_time_ms']:.3f} | {row['prove_time_ms']:.3f} | "
            f"{row['verify_time_ms']:.3f} | {row['proof_size_bytes']} | "
            f"{row['gpu_kernel_invocation_count']} |"
        )
    lines.extend(
        [
            "",
            "安全字段详见 `security_tests_report.md`；当前仓库只有在十组正式运行完成后才应生成本目录。",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    security_lines = [
        "# 正式运行安全核验",
        "",
        "安全状态由独立 verifier report、外部认证 registration artifact 和 run manifest 的交叉绑定共同核验；同时复核了 `proof.bin` 的 SHA-256。所有行都必须通过，聚合器才会写出本报告。",
        "",
        "| 数据集 | 后端 | proof hash | proof 审计 | witness 读取 | private plain values | payload audit | reference | tamper | tamper ms |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- | --- | ---: |",
    ]
    for item in security_rows:
        security_lines.append(
            f"| {item['dataset']} | {item['backend']} | `{item['proof_hash']}` | "
            f"{str(item['proof_hash_audit_passed']).lower()} | "
            f"{str(item['verifier_reads_witness']).lower()} | {item['private_plain_value_count']} | "
            f"{item['proof_payload_audit_status']} | "
            f"{str(item['reference_consistency_passed']).lower()} | "
            f"{str(item['tamper_tests_passed']).lower()} | {item['tamper_tests_ms']:.3f} |"
        )
    security_lines.extend(
        [
            "",
            f"CPU/CUDA pair-level reference consistency: `{str(cpu_gpu_reference_consistency_passed).lower()}`.",
            "",
            "结论：`all_passed=true`（由 `latest.json` 的 `security_tests` 字段记录）。这不是未执行 run 的占位结论。",
            "",
        ]
    )
    (output_dir / "security_tests_report.md").write_text("\n".join(security_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed aggregation of the five CPU and five CUDA formal runs")
    parser.add_argument("--output-dir", required=True, help="root results/paper_secure directory")
    parser.add_argument("--cpu-run", action="append", default=[], help="dataset=run_manifest.json_or_dir (exactly five)")
    parser.add_argument("--cuda-run", action="append", default=[], help="dataset=run_manifest.json_or_dir (exactly five)")
    parser.add_argument(
        "--registry",
        action="append",
        default=[],
        help="dataset=canonical verifier registration artifact (exactly five; externally anchored)",
    )
    parser.add_argument(
        "--registry-digest",
        action="append",
        default=[],
        help="dataset=lowercase SHA-256 of the trusted registration artifact (exactly five)",
    )
    parser.add_argument(
        "--registry-inspector",
        required=True,
        help="path to the independently built gatzk_registry_inspect utility",
    )
    args = parser.parse_args()

    project_root = pathlib.Path(__file__).resolve().parents[1]
    registry_paths: Dict[str, str] = {}
    for value in args.registry:
        dataset, path_value = parse_assignment(value, "--registry")
        require(dataset not in registry_paths, f"duplicate registry assignment: {dataset}")
        registry_paths[dataset] = path_value
    registry_digests: Dict[str, str] = {}
    for value in args.registry_digest:
        dataset, digest_value = parse_assignment(value, "--registry-digest")
        require(dataset not in registry_digests, f"duplicate registry digest assignment: {dataset}")
        registry_digests[dataset] = digest_value
    require(
        set(registry_paths) == set(DATASET_ORDER),
        "--registry must cover exactly " + ", ".join(DATASET_ORDER),
    )
    require(
        set(registry_digests) == set(DATASET_ORDER),
        "--registry-digest must cover exactly " + ", ".join(DATASET_ORDER),
    )
    assignments = {"cpu": args.cpu_run, "cuda": args.cuda_run}
    rows: List[Dict[str, Any]] = []
    for backend, values in assignments.items():
        require(len(values) == len(DATASET_ORDER), f"exactly five --{backend}-run assignments are required")
        seen = set()
        for value in values:
            dataset, path_value = parse_assignment(value, f"--{backend}-run")
            require(dataset not in seen, f"duplicate {backend} dataset assignment: {dataset}")
            seen.add(dataset)
            rows.append(
                check_manifest(
                    project_root,
                    dataset,
                    backend,
                    path_value,
                    registry_paths[dataset],
                    registry_digests[dataset],
                    args.registry_inspector,
                )
            )
        require(seen == set(DATASET_ORDER), f"--{backend}-run must cover exactly {', '.join(DATASET_ORDER)}")

    require(len(rows) == 10, "exactly ten formal runs are required")
    pair_consistency_passed = ensure_pair_consistency(rows)
    ordered_rows = sorted(rows, key=lambda row: (DATASET_ORDER.index(row["dataset"]), row["backend"]))
    write_outputs(
        resolve_input_path(project_root, args.output_dir),
        ordered_rows,
        pair_consistency_passed,
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"aggregate_secure_results: {exc}") from exc
