#!/usr/bin/env python3
"""Shared fail-closed checks for the ezkl GAT formal execution scripts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "resources" / "resource_manifest.json"
CONTRACT = ROOT / "artifacts" / "gat_adapter" / "formal_cli.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
COMMAND = re.compile(r"^[a-z][a-z0-9-]*$")

DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
BACKENDS = ("cpu", "gpu")
CANONICAL_RELATION = "R_GAT-ezkl"
CANONICAL_PROOF_SCOPE = "full_gat"
CANONICAL_PRIVACY = "model_and_trace_hidden"
CANONICAL_STATEMENT_MODE = "external"
CANONICAL_OUTPUT_MODE = "public_logits"
NATIVE_SECURITY_PROFILE_ID = "halo2-kzg-shplonk"
CANONICAL_CONTRACT_KEYS = {
    "schema_version",
    "baseline",
    "contract_id",
    "generated_by",
    "adapter_version",
    "ready",
    "relation_id",
    "proof_scope",
    "public_statement_mode",
    "output_mode",
    "privacy_policy",
    "native_verify_required",
    "gpu_evidence_required",
    "proof_instance_binding",
    "official_chain",
    "prepare_command",
    "producer_command",
    "witness_command",
    "prove_command",
    "verify_command",
    "required_flags",
    "native_backend",
    "gpu_backend",
}
CONTRACT_CHAIN = ("gat-prepare", "gat-produce", "gat-witness", "prove", "verify")
FORBIDDEN_PATH_MARKERS = (
    "dummy",
    "gen-srs",
    "reduced-srs",
    "reduced_srs",
    "placeholder",
    "fake",
    "mock",
)


class ContractError(RuntimeError):
    """A missing or invalid formal-run prerequisite."""


def abort(message: str) -> NoReturn:
    raise ContractError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def require_file(path: Path, label: str, *, regular: bool = True) -> Path:
    if not path.is_file():
        abort(f"missing {label}: {path}")
    if regular and path.is_symlink():
        abort(f"{label} must be a regular non-symlink file: {path}")
    if path.stat().st_size <= 0:
        abort(f"{label} is empty: {path}")
    return path


def local_path(value: str, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        abort(f"{label} must be a baseline-local relative path: {value}")
    resolved = (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        abort(f"{label} escapes the ezkl baseline root: {value}")
    return resolved


def reject_forbidden_path(path: Path, label: str) -> None:
    """Reject test-only, generated, reduced, or placeholder formal artifacts."""

    lowered = "/".join(path.parts).lower()
    if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
        abort(f"{label} uses a forbidden dummy/generated/reduced/placeholder path: {path}")
    if "tests" in {part.lower() for part in path.parts} or "assets" in {
        part.lower() for part in path.parts
    }:
        abort(f"{label} must not come from tests/assets: {path}")


def require_formal_srs(path: Path, label: str = "trusted official ezkl SRS") -> Path:
    reject_forbidden_path(path, label)
    if path.name.lower().endswith((".dummy", ".placeholder")):
        abort(f"{label} is not an authentic SRS: {path}")
    return require_file(path, label)


def resource_manifest_hash() -> str:
    require_file(MANIFEST, "resource manifest")
    return sha256(MANIFEST)


def _source_files() -> list[Path]:
    paths: list[Path] = []
    for relative in ("src", "gat_adapter", ".cargo"):
        directory = ROOT / relative
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file() and not path.is_symlink())
    for relative in ("Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain"):
        path = ROOT / relative
        if path.is_file() and not path.is_symlink():
            paths.append(path)
    return sorted(set(paths), key=lambda p: p.relative_to(ROOT).as_posix())


def source_tree_hash() -> str:
    files = _source_files()
    if not files:
        abort("no official ezkl source files found")
    h = hashlib.sha256()
    for path in files:
        relative = path.relative_to(ROOT).as_posix().encode()
        file_hash = sha256(path).encode()
        h.update(len(relative).to_bytes(8, "big"))
        h.update(relative)
        h.update(file_hash)
    return h.hexdigest()


def parse_cfg(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            abort(f"malformed config line {path}:{line_number}")
        key, value = (piece.strip() for piece in line.split("=", 1))
        if not key or key in values:
            abort(f"duplicate or empty config key {path}:{line_number}")
        values[key] = value
    return values


def validate_resources() -> None:
    validator = ROOT / "scripts" / "validate_resources.py"
    require_file(validator, "resource validator")
    result = subprocess.run(
        [sys.executable, str(validator), "--require-complete"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        abort(result.stderr.strip() or "resource manifest validation failed")


def load_cli_contract() -> dict[str, Any]:
    require_file(CONTRACT, "GAT adapter CLI contract")
    try:
        contract = json.loads(CONTRACT.read_text())
    except json.JSONDecodeError as exc:
        abort(f"invalid GAT adapter CLI contract: {exc}")
    if not isinstance(contract, dict) or set(contract) != CANONICAL_CONTRACT_KEYS:
        missing = sorted(CANONICAL_CONTRACT_KEYS - set(contract)) if isinstance(contract, dict) else sorted(CANONICAL_CONTRACT_KEYS)
        extra = sorted(set(contract) - CANONICAL_CONTRACT_KEYS) if isinstance(contract, dict) else []
        abort(f"GAT adapter CLI contract schema mismatch; missing={missing}, extra={extra}")
    if contract.get("schema_version") != 1:
        abort("GAT adapter CLI contract has unsupported schema_version")
    if contract.get("baseline") != "ezkl":
        abort("GAT adapter CLI contract baseline is not ezkl")
    if contract.get("contract_id") != "ezkl-gat-native-v1":
        abort("GAT adapter CLI contract has an unknown contract_id")
    if contract.get("generated_by") != "ezkl::gat_adapter":
        abort("GAT adapter CLI contract was not generated by the Rust adapter")
    if not isinstance(contract.get("adapter_version"), str) or not contract["adapter_version"]:
        abort("GAT adapter CLI contract has no adapter_version")
    if contract.get("ready") is not True:
        abort("GAT adapter CLI contract is not marked ready=true")
    if contract.get("relation_id") != CANONICAL_RELATION:
        abort("GAT adapter CLI contract relation_id is not R_GAT-ezkl")
    if contract.get("proof_scope") != CANONICAL_PROOF_SCOPE:
        abort("GAT adapter contract is not full_gat")
    if contract.get("public_statement_mode") != CANONICAL_STATEMENT_MODE:
        abort("GAT adapter must consume an external public statement")
    if contract.get("output_mode") != CANONICAL_OUTPUT_MODE:
        abort("GAT adapter output mode is not public_logits")
    if contract.get("privacy_policy") != CANONICAL_PRIVACY:
        abort("GAT adapter privacy policy is not model_and_trace_hidden")
    if contract.get("native_verify_required") is not True:
        abort("GAT adapter contract does not require native verification")
    if contract.get("gpu_evidence_required") is not True:
        abort("GAT adapter contract does not require GPU evidence")
    stages = contract.get("official_chain")
    if not isinstance(stages, list) or tuple(stages) != CONTRACT_CHAIN:
        abort("GAT adapter contract does not enumerate the complete official ezkl chain")
    commands = {
        key: contract.get(key)
        for key in (
            "prepare_command",
            "producer_command",
            "witness_command",
            "prove_command",
            "verify_command",
        )
    }
    for key, command in commands.items():
        if not isinstance(command, str) or not COMMAND.fullmatch(command):
            abort(f"GAT adapter contract has an invalid {key}")
        if command in {"gen-srs", "get-srs", "mock", "gen-random-data", "calibrate-settings"}:
            abort(f"GAT adapter contract maps {key} to a forbidden non-formal command")
    if (
        commands["prepare_command"] != "gat-prepare"
        or commands["producer_command"] != "gat-produce"
        or commands["witness_command"] != "gat-witness"
    ):
        abort("GAT adapter contract must expose gat-prepare, gat-produce and gat-witness")
    if commands["prove_command"] != "prove" or commands["verify_command"] != "verify":
        abort("GAT adapter contract must use official prove and verify commands")
    if contract.get("proof_instance_binding") != "native_public_instances":
        abort("GAT adapter contract does not bind native public instances")
    if contract.get("native_backend") != "halo2-kzg-shplonk":
        abort("GAT adapter contract does not identify the native Halo2/KZG/SHPLONK backend")
    if contract.get("gpu_backend") != "icicle-cuda":
        abort("GAT adapter contract does not identify the ICICLE CUDA backend")
    flags = contract.get("required_flags")
    if not isinstance(flags, dict) or set(flags) != {
        "prepare",
        "producer",
        "witness",
        "prove",
        "verify",
    }:
        abort("GAT adapter contract required_flags schema mismatch")
    for stage, values in flags.items():
        if not isinstance(values, list) or not values or any(
            not isinstance(value, str) or not value.startswith("--") for value in values
        ):
            abort(f"GAT adapter contract required_flags.{stage} is invalid")
    required_flag_sets = {
        "prepare": {
            "--dataset",
            "--phase",
            "--resource-manifest",
            "--output-dir",
            "--srs-path",
            "--public-statement",
            "--public-output",
            "--gpu",
        },
        "producer": {"--dataset", "--resource-manifest", "--output-dir", "--gpu"},
        "witness": {
            "--dataset",
            "--resource-manifest",
            "--compiled-circuit",
            "--statement",
            "--output",
            "--gpu",
        },
        "prove": {"--witness", "--compiled-circuit", "--pk-path", "--proof-path", "--srs-path", "--check-mode"},
        "verify": {
            "--settings-path", "--proof-path", "--vk-path", "--srs-path", "--statement",
            "--public-output", "--artifact-manifest", "--resource-manifest",
        },
    }
    for stage, required in required_flag_sets.items():
        if not required.issubset(set(flags[stage])):
            abort(f"GAT adapter contract required_flags.{stage} omits required official flags")
        if set(flags[stage]) != required:
            abort(f"GAT adapter contract required_flags.{stage} does not match the Rust Clap command exactly")
        if "--reduced-srs" in flags[stage] or "--reduced_srs" in flags[stage]:
            abort("GAT adapter contract enables reduced SRS verification")
    return contract


def validate_config(config: Path, dataset: str) -> dict[str, str]:
    require_file(config, "canonical secure config")
    values = parse_cfg(config)
    if values.get("dataset") != dataset:
        abort(f"config dataset mismatch: expected {dataset}, got {values.get('dataset')}")
    expected_roots = {
        "data_root": "resources/canonical/cache",
        "cache_root": "resources/canonical/cache",
        "checkpoint_bundle": f"resources/canonical/checkpoints/{'ogbn_arxiv_gat' if dataset == 'ogbn_arxiv' else dataset + '_gat'}",
    }
    for key, expected in expected_roots.items():
        if values.get(key) != expected:
            abort(f"{config} has non-canonical {key}: {values.get(key)!r}")
        require_file(ROOT / expected / "meta.cfg" if key != "checkpoint_bundle" else ROOT / expected / "manifest.json", key)
    if values.get("auto_prepare_dataset") != "false":
        abort(f"{config} must disable dataset auto-preparation")
    srs_path = values.get("srs_path")
    if not srs_path:
        abort(f"{config} has no srs_path")
    srs = local_path(srs_path, "config srs_path")
    reject_forbidden_path(srs, "config srs_path")
    if values.get("trusted_srs_sha256") == "REQUIRED_EXTERNAL_OFFICIAL_EZKL_SRS_SHA256":
        # The command-line hash is checked later.  The marker makes the
        # missing ceremony explicit without fabricating a digest in config.
        pass
    elif not HEX64.fullmatch(values.get("trusted_srs_sha256", "")):
        abort(f"{config} has an invalid trusted_srs_sha256")
    if values.get("ezkl_security_profile_id") != NATIVE_SECURITY_PROFILE_ID:
        abort(f"{config} must pin native ezkl security profile {NATIVE_SECURITY_PROFILE_ID}")
    for key in ("static_table_id", "degree_bound_id"):
        if not values.get(key) or values[key] == "auto":
            abort(f"{config} must bind an explicit {key}")
    for key in ("static_table_digest", "degree_bound_digest"):
        require_digest(values.get(key, ""), f"{config} {key}")
    if "verifier_registry_digest" in values:
        abort(f"{config} still references the main-experiment verifier registry")
    return values


def require_digest(value: str, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        abort(f"{label} must be a lowercase SHA-256 digest")
    return value


def load_json(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        abort(f"invalid {label}: {exc}")
    if not isinstance(value, dict):
        abort(f"{label} must be a JSON object")
    return value


def require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        abort(f"{label} must be a JSON object")
    actual = set(value)
    if actual != expected:
        abort(f"{label} schema mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    return value


def validate_public_statement(
    path: Path,
    *,
    dataset: str,
    circuit_ref: dict[str, str] | None = None,
    public_output_sha256: str | None = None,
    quant_cfg_id: str | None = None,
    static_tables_id: str | None = None,
) -> dict[str, Any]:
    statement = load_json(path, "external public statement")
    required = {
        "schema_version",
        "relation_id",
        "dataset",
        "proof_scope",
        "privacy_policy",
        "public_statement_mode",
        "output_mode",
        "model_arch_id",
        "quant_cfg_id",
        "static_table_id",
        "degree_bound_id",
        "ezkl_security_profile_id",
        "public_ezkl",
        "output_digest",
        "public_output_sha256",
        "statement_digest",
    }
    require_exact_keys(statement, required, "external public statement")
    if statement["schema_version"] != 1 or statement["relation_id"] != CANONICAL_RELATION:
        abort("external public statement has an unsupported schema or relation")
    if statement["dataset"] != dataset or statement["proof_scope"] != CANONICAL_PROOF_SCOPE:
        abort("external public statement dataset/proof scope mismatch")
    if statement["privacy_policy"] != CANONICAL_PRIVACY:
        abort("external public statement privacy policy mismatch")
    if statement["public_statement_mode"] != CANONICAL_STATEMENT_MODE:
        abort("external public statement mode is not external")
    if statement["output_mode"] != CANONICAL_OUTPUT_MODE:
        abort("external public statement output mode is not public_logits")
    for key in ("model_arch_id", "quant_cfg_id", "static_table_id", "degree_bound_id", "ezkl_security_profile_id"):
        if not isinstance(statement[key], str) or not statement[key]:
            abort(f"external public statement has no {key}")
    require_digest(statement["output_digest"], "external public statement output_digest")
    require_digest(statement["public_output_sha256"], "external public statement public_output_sha256")
    require_digest(statement["statement_digest"], "external public statement statement_digest")
    public = require_exact_keys(
        statement["public_ezkl"],
        {
            "topology", "static_tables", "quant", "ledger", "circuit", "model_hash", "model",
            "feature_table", "node_ids", "src", "dst", "node_ptr", "edge_ptr", "y",
        },
        "external public statement public_ezkl",
    )
    for key in ("topology", "static_tables", "quant", "ledger"):
        identity = require_exact_keys(public[key], {"id", "digest"}, f"statement {key}")
        if not isinstance(identity["id"], str) or not identity["id"]:
            abort(f"statement {key}.id is empty")
        require_digest(identity["digest"], f"statement {key}.digest")
    circuit = require_exact_keys(
        public["circuit"],
        {
            "relation_id", "compiled_circuit_id", "compiled_circuit_digest", "settings_id",
            "settings_digest", "vk_id", "vk_digest", "model_graph_id", "model_graph_digest",
            "ezkl_security_profile_id",
        },
        "statement circuit_ref",
    )
    parsed_ref = require_circuit_ref(circuit, "external public statement circuit_ref")
    if circuit["ezkl_security_profile_id"] != statement["ezkl_security_profile_id"]:
        abort("external public statement security profile mismatch")
    model = require_exact_keys(public["model"], {"hash", "hash_domain", "canonical_model_digest"}, "statement model")
    if model["hash"] != public["model_hash"] or not isinstance(public["model_hash"], str) or not public["model_hash"]:
        abort("statement native model H(W) binding mismatch")
    if (
        len(public["model_hash"]) != 66
        or not public["model_hash"].startswith("0x")
        or any(character not in "0123456789abcdef" for character in public["model_hash"][2:])
    ):
        abort("statement native model H(W) is not a canonical Halo2 field encoding")
    require_digest(model["canonical_model_digest"], "statement canonical model digest")
    if not isinstance(public["feature_table"], list) or not public["feature_table"]:
        abort("statement feature_table is empty")
    if not isinstance(public["node_ids"], list) or not public["node_ids"]:
        abort("statement node_ids is empty")
    if not isinstance(public["y"], list) or not public["y"]:
        abort("statement public Y is empty")
    for key in ("feature_table", "node_ids", "src", "dst", "node_ptr", "edge_ptr"):
        values = public[key]
        if not isinstance(values, list) or not values:
            abort(f"statement {key} is empty")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            abort(f"statement {key} must contain integers")
        if any(value < 0 for value in values):
            abort(f"statement {key} contains a negative value")
    if len(public["src"]) != len(public["dst"]):
        abort("statement src/dst lengths disagree")
    if len(public["node_ptr"]) < 2 or len(public["edge_ptr"]) < 2:
        abort("statement node_ptr/edge_ptr must contain at least two boundaries")
    if public["node_ptr"][0] != 0 or public["edge_ptr"][0] != 0:
        abort("statement graph pointers must start at zero")
    if public["node_ptr"][-1] != len(public["node_ids"]) or public["edge_ptr"][-1] != len(public["src"]):
        abort("statement graph pointers do not cover the public graph")
    if any(a >= b for a, b in zip(public["node_ptr"], public["node_ptr"][1:])) or any(
        a >= b for a, b in zip(public["edge_ptr"], public["edge_ptr"][1:])
    ):
        abort("statement graph pointers must be strictly increasing")
    if len(set(public["node_ids"])) != len(public["node_ids"]):
        abort("statement node_ids contain duplicates")
    for node_start, node_end, edge_start, edge_end in zip(
        public["node_ptr"], public["node_ptr"][1:], public["edge_ptr"], public["edge_ptr"][1:]
    ):
        if any(
            src < node_start or src >= node_end or dst < node_start or dst >= node_end
            for src, dst in zip(public["src"][edge_start:edge_end], public["dst"][edge_start:edge_end])
        ):
            abort("statement edge crosses a graph batch boundary")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in public["y"]):
        abort("statement public Y must contain integers")
    for value in public["y"]:
        if value < -(1 << 127) or value >= (1 << 127):
            abort("statement public Y contains an out-of-range i128")
    output_bytes = bytearray(b"ezkl-gat-public-y-v1")
    for value in public["y"]:
        output_bytes.extend(value.to_bytes(16, "little", signed=True))
    if statement["output_digest"] != sha256_bytes(bytes(output_bytes)):
        abort("statement output_digest does not match public Y")
    if public["quant"]["id"] != statement["quant_cfg_id"] or public["static_tables"]["id"] != statement["static_table_id"] or public["ledger"]["id"] != statement["degree_bound_id"]:
        abort("statement registry ids do not match public boundary")
    if quant_cfg_id is not None and statement["quant_cfg_id"] != quant_cfg_id:
        abort("external public statement quant_cfg_id does not match the canonical config")
    if static_tables_id is not None and statement["static_table_id"] != static_tables_id:
        abort("external public statement static_table_id does not match the canonical config")
    require_digest(statement["public_output_sha256"], "external public statement public_output_sha256")
    if public_output_sha256 is not None and statement["public_output_sha256"] != public_output_sha256:
        abort("external public statement does not bind the selected public output Y")
    if circuit_ref is not None and parsed_ref != circuit_ref:
        abort("external public statement circuit_ref does not match the run circuit")
    return statement


def validate_public_output(path: Path, *, dataset: str) -> Any:
    require_file(path, "external public output Y")
    try:
        output = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        abort(f"external public output Y is not valid JSON: {exc}")
    if output in (None, [], {}):
        abort("external public output Y is empty")
    if not isinstance(output, dict) or set(output) != {"dataset", "Y"}:
        abort("external public output must use the canonical {dataset,Y} object")
    if output.get("dataset") != dataset:
        abort("external public output dataset mismatch")
    payload = output.get("Y")
    if payload in (None, [], {}):
        abort("external public output Y payload is empty")
    return output


def public_output_values(path: Path, *, dataset: str) -> list[int]:
    """Return the one-dimensional integer Y vector used by the Rust adapter."""

    output = validate_public_output(path, dataset=dataset)
    payload = output["Y"]
    if not isinstance(payload, list) or not payload:
        abort("external public output Y must be a non-empty one-dimensional array")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in payload):
        abort("external public output Y must contain integers")
    if any(value < -(1 << 127) or value >= (1 << 127) for value in payload):
        abort("external public output Y contains an out-of-range i128")
    return payload


def validate_artifact_identity(
    artifact_manifest: Path,
    *,
    dataset: str,
    backend: str,
    binary: Path,
    srs: Path,
    settings: Path,
    compiled: Path,
    vk: Path,
    pk: Path,
) -> dict[str, Any]:
    manifest = load_json(artifact_manifest, "artifact identity manifest")
    required = {
        "schema_version",
        "baseline",
        "relation_id",
        "dataset",
        "backend",
        "proof_scope",
        "privacy_policy",
        "public_statement_mode",
        "output_mode",
        "quant_cfg_id",
        "static_tables_id",
        "circuit_ref",
        "compiled_circuit_id",
        "settings_id",
        "vk_id",
        "binary_sha256",
        "source_tree_sha256",
        "resource_manifest_sha256",
        "srs_sha256",
        "settings_sha256",
        "compiled_circuit_sha256",
        "vk_sha256",
        "pk_sha256",
        "srs_provenance",
        "files",
        "source_snapshot",
        "statement_sha256",
        "ezkl_security_profile_id",
    }
    require_exact_keys(manifest, required, "artifact identity manifest")
    if manifest["schema_version"] != 1 or manifest["dataset"] != dataset or manifest["backend"] != backend:
        abort("artifact identity manifest dataset/backend/schema mismatch")
    if (
        manifest["baseline"] != "ezkl"
        or manifest["proof_scope"] != CANONICAL_PROOF_SCOPE
        or manifest["relation_id"] != CANONICAL_RELATION
        or manifest["privacy_policy"] != CANONICAL_PRIVACY
        or manifest["public_statement_mode"] != CANONICAL_STATEMENT_MODE
        or manifest["output_mode"] != CANONICAL_OUTPUT_MODE
    ):
        abort("artifact identity manifest is not full_gat")
    circuit_ref = require_circuit_ref(manifest["circuit_ref"], "artifact identity circuit_ref")
    if circuit_ref["compiled_circuit_id"] != manifest["compiled_circuit_id"]:
        abort("artifact circuit_ref compiled_circuit_id mismatch")
    if circuit_ref["settings_id"] != manifest["settings_id"]:
        abort("artifact circuit_ref settings_id mismatch")
    if circuit_ref["vk_id"] != manifest["vk_id"]:
        abort("artifact circuit_ref vk_id mismatch")
    for key in ("quant_cfg_id", "static_tables_id", "ezkl_security_profile_id"):
        if not isinstance(manifest[key], str) or not manifest[key] or "pending" in manifest[key].lower():
            abort(f"artifact identity manifest has invalid {key}")
    circuit_document = manifest["circuit_ref"]
    if circuit_document["ezkl_security_profile_id"] != manifest["ezkl_security_profile_id"]:
        abort("artifact circuit_ref security profile mismatch")
    srs_provenance = manifest["srs_provenance"]
    srs_provenance = require_exact_keys(
        srs_provenance,
        {"source", "trusted", "reduced", "generated", "ceremony_id", "sha256"},
        "artifact SRS provenance",
    )
    if srs_provenance.get("source") != "official_ezkl_ceremony":
        abort("artifact SRS is not identified as official ezkl ceremony material")
    if srs_provenance.get("trusted") is not True or srs_provenance.get("reduced") is not False:
        abort("artifact SRS provenance is not trusted full-size material")
    if srs_provenance.get("generated") is not False:
        abort("artifact SRS provenance indicates generated material")
    if not isinstance(srs_provenance.get("ceremony_id"), str) or not srs_provenance["ceremony_id"]:
        abort("artifact SRS provenance has no ceremony_id")
    if srs_provenance.get("sha256") != manifest["srs_sha256"]:
        abort("artifact SRS provenance digest does not match artifact identity")
    require_digest(manifest["srs_sha256"], "artifact srs_sha256")
    require_digest(srs_provenance["sha256"], "artifact SRS provenance sha256")
    source_snapshot = require_exact_keys(
        manifest["source_snapshot"], {"root", "git_commit", "git_tree"}, "artifact source_snapshot"
    )
    for key in ("root", "git_commit", "git_tree"):
        if not isinstance(source_snapshot[key], str) or not source_snapshot[key]:
            abort(f"artifact source_snapshot.{key} is empty")
    artifact_files = manifest["files"]
    if not isinstance(artifact_files, list) or not artifact_files:
        abort("artifact identity manifest has no authenticated files")
    artifact_root = artifact_manifest.parent
    file_records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(artifact_files):
        record = require_exact_keys(record, {"path", "size", "sha256"}, f"artifact files[{index}]")
        relative = record["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).parts[0] == "."
        ):
            abort(f"artifact files[{index}].path is not baseline-local")
        if relative in file_records:
            abort(f"artifact files contains duplicate path: {relative}")
        require_digest(record["sha256"], f"artifact files[{index}].sha256")
        if isinstance(record["size"], bool) or not isinstance(record["size"], int) or record["size"] <= 0:
            abort(f"artifact files[{index}].size is invalid")
        resolved = artifact_root / relative
        cursor = artifact_root
        for component in Path(relative).parts:
            cursor = cursor / component
            if cursor.is_symlink():
                abort(f"artifact file path contains a symlink: {relative}")
        require_file(resolved, f"artifact file {relative}")
        if resolved.stat().st_size != record["size"] or sha256(resolved) != record["sha256"]:
            abort(f"artifact file digest/size mismatch: {relative}")
        file_records[relative] = record
    files = {
        "binary_sha256": binary,
        "srs_sha256": srs,
        "settings_sha256": settings,
        "compiled_circuit_sha256": compiled,
        "vk_sha256": vk,
        "pk_sha256": pk,
    }
    for key, path in files.items():
        require_digest(manifest[key], f"artifact {key}")
        if manifest[key] != sha256(path):
            abort(f"artifact identity mismatch for {key}: {path}")
    require_digest(manifest["statement_sha256"], "artifact statement_sha256")
    if manifest["source_tree_sha256"] != source_tree_hash():
        abort("artifact identity source_tree_sha256 does not match current source")
    if manifest["resource_manifest_sha256"] != resource_manifest_hash():
        abort("artifact identity resource_manifest_sha256 does not match current resources")
    for key in ("source_tree_sha256", "resource_manifest_sha256"):
        require_digest(manifest[key], key)
    return manifest


def validate_gpu_environment() -> dict[str, Any]:
    install_dir = os.environ.get("ICICLE_BACKEND_INSTALL_DIR", "")
    if not install_dir:
        abort("GPU execution requires ICICLE_BACKEND_INSTALL_DIR")
    directory = Path(install_dir)
    if not directory.is_dir() or directory.is_symlink():
        abort(f"ICICLE_BACKEND_INSTALL_DIR is not a regular directory: {install_dir}")
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        abort(f"CUDA device probe failed: {exc}")
    if probe.returncode != 0 or not probe.stdout.strip():
        abort(f"CUDA device probe failed: {probe.stderr.strip() or 'no GPU reported'}")
    return {"install_dir": str(directory.resolve()), "nvidia_smi": probe.stdout.strip()}


def validate_gpu_evidence(path: Path, *, backend: str) -> dict[str, Any]:
    evidence = load_json(path, "GPU/backend evidence")
    return validate_gpu_evidence_document(evidence, backend=backend)


def validate_gpu_evidence_document(evidence: Any, *, backend: str) -> dict[str, Any]:
    """Validate an already-loaded CPU/GPU attestation without weakening it."""

    evidence = require_exact_keys(
        evidence,
        {
            "backend",
            "gpu_backend",
            "backend_loaded",
            "cpu_fallback",
            "device_synchronized",
            "gpu_kernel_invocation_count",
            "provenance",
            "invocation_trace_sha256",
        },
        "GPU/backend evidence",
    )
    if not isinstance(evidence["gpu_kernel_invocation_count"], int) or isinstance(
        evidence["gpu_kernel_invocation_count"], bool
    ):
        abort("GPU evidence kernel invocation count is not an integer")
    require_digest(evidence["invocation_trace_sha256"], "GPU invocation trace digest")
    if backend == "gpu":
        if evidence.get("backend") != "CUDA":
            abort("GPU evidence does not identify CUDA backend")
        if evidence.get("gpu_backend") != "icicle-cuda":
            abort("GPU evidence does not identify the ICICLE CUDA backend")
        if evidence.get("backend_loaded") is not True:
            abort("GPU evidence does not confirm backend loading")
        if evidence.get("cpu_fallback") is not False:
            abort("GPU evidence indicates CPU fallback or omits the fail-closed flag")
        if evidence.get("device_synchronized") is not True:
            abort("GPU evidence lacks device synchronization")
        count = evidence.get("gpu_kernel_invocation_count")
        if not isinstance(count, int) or count <= 0:
            abort("GPU evidence has no positive GPU kernel invocation count")
        provenance = evidence.get("provenance")
        provenance = require_exact_keys(
            provenance,
            {"frontend_commit", "abi", "device", "architecture", "runtime", "manifest_sha256"},
            "GPU evidence provenance",
        )
        for key in ("frontend_commit", "abi", "device", "architecture", "runtime", "manifest_sha256"):
            value = provenance.get(key)
            if not isinstance(value, str) or not value:
                abort(f"GPU evidence provenance is missing {key}")
        require_digest(provenance["manifest_sha256"], "GPU backend manifest_sha256")
    elif (
        evidence.get("backend") != "CPU"
        or evidence.get("gpu_backend") != "none"
        or evidence.get("backend_loaded") is not True
        or evidence.get("cpu_fallback") is not False
        or evidence.get("device_synchronized") is not True
        or evidence["gpu_kernel_invocation_count"] != 0
        or evidence.get("provenance") is not None
    ):
        abort("CPU run has invalid backend evidence")
    return evidence


def reject_reduced_srs(argv: list[str]) -> None:
    if any("reduced-srs" in arg or "reduced_srs" in arg for arg in argv):
        abort("--reduced-srs is forbidden for formal comparison runs")


def require_positive_timings(timings: Any, label: str = "phase timings") -> dict[str, float]:
    if not isinstance(timings, dict):
        abort(f"{label} must be an object")
    result: dict[str, float] = {}
    for key in ("prepare_compile_setup", "witness_generation", "prove", "verify"):
        value = timings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            abort(f"{label}.{key} must be a positive finite duration")
        result[key] = float(value)
    return result


def require_circuit_ref(value: Any, label: str = "circuit_ref") -> dict[str, str]:
    value = require_exact_keys(
        value,
        {
            "relation_id", "compiled_circuit_id", "compiled_circuit_digest", "settings_id",
            "settings_digest", "vk_id", "vk_digest", "model_graph_id", "model_graph_digest",
            "ezkl_security_profile_id",
        },
        label,
    )
    required = ("relation_id", "compiled_circuit_id", "settings_id", "vk_id")
    if any(not isinstance(value.get(key), str) or not value[key] for key in required):
        abort(f"{label} is missing a non-empty identity field")
    if value["relation_id"] != CANONICAL_RELATION:
        abort(f"{label}.relation_id is not R_GAT-ezkl")
    if any(
        any(
            marker in value[key].lower()
            for marker in ("unbound", "pending", "placeholder", "dummy", "fake")
        )
        for key in required[1:]
    ):
        abort(f"{label} contains a placeholder identity")
    for key in ("compiled_circuit_digest", "settings_digest", "vk_digest", "model_graph_digest"):
        require_digest(value[key], f"{label}.{key}")
    if not isinstance(value["model_graph_id"], str) or not value["model_graph_id"]:
        abort(f"{label}.model_graph_id is empty")
    if not isinstance(value["ezkl_security_profile_id"], str) or not value["ezkl_security_profile_id"]:
        abort(f"{label}.ezkl_security_profile_id is empty")
    return {key: value[key] for key in required}


def print_error(exc: Exception) -> int:
    print(f"formal prerequisite failed: {exc}", file=sys.stderr)
    return 2
