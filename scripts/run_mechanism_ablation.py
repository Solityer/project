#!/usr/bin/env python3
"""Run paired, independently verified zkGAT mechanism experiments.

The runner is deliberately small: each side invokes the native mechanism
prover, then a second native process verifies the serialized artifact. Results
are written under runs/mechanism_ablation_secure/<session-id>/ and never into
the canonical end-to-end result directory.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "mechanism_ablation_secure"
MECHANISMS = ("feature_folding", "shared_neighborhood")
VARIANTS = ("original", "control")
METRIC_FIELDS = (
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "proof_size_bytes",
)
HEX_256 = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_UINT = re.compile(r"^(0|[1-9][0-9]*)$")
POWER_OF_TWO_LOWER_BOUND = re.compile(r"^2\^([0-9]+)$")
RUN_FORMAT = "zkgat_mechanism_result_v1"
TIMING_SCHEMA = "mechanism_commitment_disjoint_verify_parse_accept_v2"
TIMING_SEMANTICS = {
    "commitment_time_ms": "cold native KZG commitment work only, including first-use static-base preparation invoked inside commitment; Taylor-shift coefficient materialization is excluded",
    "prove_time_ms": "native online proof work, Taylor-shift coefficient materialization, and canonical proof encoding, excluding KZG commitment work and artifact file I/O",
    "verify_time_ms": "independent verifier parsing through accept/reject",
    "proof_size_bytes": "actual canonical serialized mechanism proof bytes",
}
ACTIVE_SECRET_FILES: set[pathlib.Path] = set()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def cleanup_secret_files() -> None:
    for path in list(ACTIVE_SECRET_FILES):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    ACTIVE_SECRET_FILES.clear()


atexit.register(cleanup_secret_files)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def atomic_write_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def emit(message: str, session_log: pathlib.Path | None = None) -> None:
    print(message, flush=True)
    if session_log is not None:
        with session_log.open("a", encoding="utf-8") as stream:
            stream.write(f"{utc_now()} {message}\n")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"JSON has duplicate field {key!r}")
        result[key] = value
    return result


def load_json(path: pathlib.Path, context: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{context} is missing or not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot parse {context}: {exc}") from exc
    require(isinstance(value, dict), f"{context} must be a JSON object")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_contains_bytes(path: pathlib.Path, needle: bytes) -> bool:
    overlap = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            block = overlap + chunk
            if needle in block:
                return True
            overlap = block[-(len(needle) - 1):] if len(needle) > 1 else b""
    return False


def load_build_provenance_module() -> Any:
    path = PROJECT_ROOT / "tools" / "build_provenance.py"
    spec = importlib.util.spec_from_file_location("mechanism_runner_build_provenance", path)
    require(spec is not None and spec.loader is not None, f"cannot load build provenance helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_output(*args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def source_snapshot() -> dict[str, Any]:
    provenance = load_build_provenance_module()
    manifest = provenance.source_manifest_document(PROJECT_ROOT)
    diff = git_output("diff", "--binary", "HEAD")
    status = git_output("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "git_head": git_output("rev-parse", "HEAD").decode("ascii").strip(),
        "git_diff_sha256": sha256_bytes(diff),
        "git_status_sha256": sha256_bytes(status),
        "source_inventory_id": manifest["inventory_id"],
        "source_tree_hash": manifest["source_tree_hash"],
        "source_file_count": manifest["file_count"],
        "source_manifest": manifest,
    }


def integer_arg(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_arg(value: str) -> int:
    parsed = integer_arg(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_workload(value: str) -> tuple[int, int, int, int]:
    fields = value.split(",")
    if len(fields) != 4:
        raise argparse.ArgumentTypeError("workload must be N,E,D,G (nodes,edges,width,graphs)")
    try:
        nodes, edges, width, graphs = (int(field, 10) for field in fields)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workload fields must be integers") from exc
    if min(nodes, edges, width, graphs) <= 0:
        raise argparse.ArgumentTypeError("workload fields must all be positive")
    if graphs > nodes or edges < nodes:
        raise argparse.ArgumentTypeError("workload requires nodes >= graphs and edges >= nodes")
    return nodes, edges, width, graphs


def validate_workloads(args: argparse.Namespace) -> list[dict[str, int]]:
    if args.workload:
        require(
            all(value is None for value in (args.nodes, args.edges, args.width, args.graphs)),
            "--workload cannot be combined with --nodes/--edges/--width/--graphs",
        )
        tuples = args.workload
    else:
        values = (args.nodes, args.edges, args.width, args.graphs)
        if all(value is None for value in values):
            tuples = [(16, 48, 2, 1), (64, 256, 4, 2), (256, 1024, 8, 4)]
        else:
            require(all(value is not None for value in values), "set all of --nodes/--edges/--width/--graphs together")
            tuples = [tuple(values)]
        for item in tuples:
            require(min(item) > 0, "nodes, edges, width, and graphs must all be positive")
            require(item[3] <= item[0] and item[1] >= item[0], "workload requires nodes >= graphs and edges >= nodes")
    return [
        {"nodes": item[0], "edges": item[1], "width": item[2], "graphs": item[3]}
        for item in tuples
    ]


def resolve_binary(args: argparse.Namespace) -> pathlib.Path:
    if args.binary is not None:
        path = args.binary.expanduser().resolve()
    else:
        build_name = "build_cuda" if args.backend == "cuda" else "build"
        path = (PROJECT_ROOT / build_name / "gatzk_mechanism").resolve()
    require(path.is_file() and os.access(path, os.X_OK), f"mechanism executable is missing or not executable: {path}")
    return path


def build_snapshot(binary: pathlib.Path, backend: str) -> dict[str, Any]:
    build_root = binary.parent
    cache_path = build_root / "CMakeCache.txt"
    source_manifest_path = build_root / "generated" / "gatzk" / "source_manifest.json"
    require(cache_path.is_file(), f"build cache is missing: {cache_path}")
    require(source_manifest_path.is_file(), f"build source manifest is missing: {source_manifest_path}")
    cache_bytes = cache_path.read_bytes()
    cache_text = cache_bytes.decode("utf-8", errors="replace")
    cache_values: dict[str, str] = {}
    for line in cache_text.splitlines():
        if line and not line.startswith("//") and not line.startswith("#") and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            cache_values[key.split(":", 1)[0]] = value
    manifest = load_json(source_manifest_path, "build source manifest")
    if backend == "cuda":
        require(cache_values.get("GATZK_ENABLE_CUDA_BACKEND") == "ON", "selected build was not configured with CUDA support")
    return {
        "build_root": str(build_root),
        "cmake_cache_sha256": sha256_bytes(cache_bytes),
        "cmake_cache_size_bytes": len(cache_bytes),
        "cmake_generator": cache_values.get("CMAKE_GENERATOR"),
        "build_type": cache_values.get("CMAKE_BUILD_TYPE"),
        "cxx_compiler": cache_values.get("CMAKE_CXX_COMPILER"),
        "cxx_compiler_id": cache_values.get("CMAKE_CXX_COMPILER_ID"),
        "cuda_enabled": cache_values.get("GATZK_ENABLE_CUDA_BACKEND"),
        "binary_path": str(binary),
        "binary_sha256": sha256_file(binary),
        "binary_size_bytes": binary.stat().st_size,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "built_source_tree_hash": manifest.get("source_tree_hash"),
        "source_manifest": manifest,
    }


def srs_snapshot(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    require(resolved.is_dir() and not path.is_symlink(), f"--srs must be a regular SRS directory: {path}")
    manifest_path = resolved / "srs_manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), f"SRS manifest is missing: {manifest_path}")
    manifest = load_json(manifest_path, "SRS manifest")
    raw = manifest_path.read_bytes()
    return {
        "path": str(resolved),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_bytes(raw),
        "manifest_size_bytes": len(raw),
        "manifest_srs_id": manifest.get("srs_id"),
        "manifest_protocol_schema_id": manifest.get("protocol_schema_id"),
        "manifest_degree_bound": manifest.get("degree_bound"),
    }


def cuda_snapshot(device: int) -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(result.returncode == 0, f"cannot identify CUDA devices with nvidia-smi: {result.stderr.strip()}")
    devices: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        require(len(fields) == 4, f"malformed nvidia-smi device row: {line!r}")
        devices.append({"index": fields[0], "uuid": fields[1], "name": fields[2], "driver_version": fields[3]})
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        selected_token = str(device)
    else:
        tokens = [token.strip() for token in visible.split(",")]
        require(device < len(tokens), f"CUDA device index {device} is outside CUDA_VISIBLE_DEVICES={visible!r}")
        selected_token = tokens[device]
    selected = next(
        (
            item for item in devices
            if item["index"] == selected_token
            or item["uuid"] == selected_token
            or item["uuid"].startswith(selected_token)
        ),
        None,
    )
    require(selected is not None, f"selected CUDA device token {selected_token!r} is not present in nvidia-smi inventory")
    return {
        "backend": "cuda",
        "selected_device_index": device,
        "selected_device_token": selected_token,
        "selected_device": selected,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "inventory": devices,
    }


def safe_output_root(value: pathlib.Path) -> pathlib.Path:
    allowed_root = DEFAULT_OUTPUT_ROOT.resolve()
    candidate = value.expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    require(not candidate.is_symlink(), "output root cannot be a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"output root must be inside the isolated {allowed_root} tree") from exc
    forbidden = (PROJECT_ROOT / "results" / "paper_secure").resolve()
    require(resolved != forbidden and forbidden not in resolved.parents and resolved not in forbidden.parents,
            "mechanism output cannot overlap results/paper_secure")
    if resolved != allowed_root:
        require(not resolved.exists(), "custom output root must be new; existing run directories are immutable")
    return resolved


def create_session_dir(output_root: pathlib.Path) -> tuple[str, pathlib.Path]:
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(8):
        session_id = f"mechanism_{timestamp}_{uuid.uuid4().hex[:12]}"
        session_root = output_root / session_id
        try:
            session_root.mkdir(mode=0o700)
            return session_id, session_root
        except FileExistsError:
            continue
    raise ValueError(f"cannot reserve unique session directory under {output_root}")


def workload_slug(workload: dict[str, int]) -> str:
    return "n{nodes}_e{edges}_d{width}_g{graphs}".format(**workload)


def validate_security_receipt(security: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate each variant's own exact budget and extract shared policy fields.

    The detailed challenge, opening-game, and identity counts are deliberately
    retained per variant. They are evidence for each ledger, not pair-equality
    fields: the mechanism changes those counts by design.
    """
    require(type(security.get("target_bits")) is int and security["target_bits"] == 120,
            "mechanism result does not use the required 120-bit security target")
    require(security.get("budget_exact") is True, "mechanism security budget is not exact")
    require(security.get("bound_kind") == "conservative_exact", "mechanism security budget is not conservatively exact")
    ledger_hash = security.get("budget_ledger_sha256")
    require(isinstance(ledger_hash, str) and HEX_256.fullmatch(ledger_hash), "invalid security budget ledger digest")

    assumptions = security.get("assumptions")
    require(isinstance(assumptions, dict), "mechanism security receipt has no assumption profile")
    computational_bits = security.get("computational_assumption_bits")
    kzg_bits = assumptions.get("kzg_assumption_bits")
    require(type(computational_bits) is int and computational_bits >= 120,
            "mechanism security receipt has an invalid computational assumption strength")
    require(type(kzg_bits) is int and kzg_bits >= 120,
            "mechanism security receipt has an invalid KZG assumption strength")
    kzg_profile_id = assumptions.get("kzg_assumption_profile_id")
    registry_profile_id = assumptions.get("registry_assumption_id")
    require(isinstance(kzg_profile_id, str) and kzg_profile_id,
            "mechanism security receipt has no KZG assumption profile identity")
    require(isinstance(registry_profile_id, str) and registry_profile_id,
            "mechanism security receipt has no registry/Fiat-Shamir assumption profile identity")
    fs_query_budget_max = security.get("fiat_shamir_query_budget_max")
    require(type(fs_query_budget_max) is int and fs_query_budget_max == (1 << 20),
            "mechanism security receipt has no supported 2^20 Fiat-Shamir query budget")
    fs_profile_id = assumptions.get("fiat_shamir_assumption_profile_id")
    require(isinstance(fs_profile_id, str) and fs_profile_id,
            "mechanism security receipt has no Fiat-Shamir assumption profile identity")
    fs_security_bits = assumptions.get("fiat_shamir_security_bits")
    fs_ro_terms = assumptions.get("fiat_shamir_random_oracle_terms_128bit")
    require(type(fs_security_bits) is int and fs_security_bits == 128,
            "mechanism security receipt has an invalid Fiat-Shamir assumption strength")
    require(type(fs_ro_terms) is int and fs_ro_terms == 1,
            "mechanism security receipt has an invalid random-oracle assumption term count")
    fs_challenge_count = assumptions.get("fiat_shamir_challenge_count")
    fs_inventory = security.get("fiat_shamir_challenge_inventory")
    require(type(fs_challenge_count) is int and fs_challenge_count > 0,
            "mechanism security receipt has no variant-specific Fiat-Shamir challenge count")
    require(isinstance(fs_inventory, list) and fs_challenge_count == len(fs_inventory),
            "Fiat-Shamir challenge count does not match the variant-specific challenge inventory")
    kzg_game_count = assumptions.get("kzg_point_group_games")
    kzg_inventory = security.get("kzg_point_group_game_inventory")
    require(type(kzg_game_count) is int and kzg_game_count > 0,
            "mechanism security receipt has no variant-specific KZG game count")
    require(isinstance(kzg_inventory, list) and kzg_game_count == len(kzg_inventory),
            "KZG point-group game count does not match the variant-specific game inventory")
    field_modulus = security.get("field_modulus_decimal")
    require(isinstance(field_modulus, str) and CANONICAL_UINT.fullmatch(field_modulus) and int(field_modulus) > 0,
            "mechanism security receipt has an invalid field modulus")
    field_denominator = security.get("field_denominator_lower_bound")
    require(isinstance(field_denominator, str) and POWER_OF_TWO_LOWER_BOUND.fullmatch(field_denominator),
            "mechanism security receipt has no valid field denominator lower bound")

    total_error = security.get("total_error_bound")
    require(isinstance(total_error, dict), "mechanism security receipt has no total error bound")
    require(total_error.get("meets_target") is True, "mechanism security budget did not meet its target")
    require(total_error.get("denominator_lower_bound") == field_denominator,
            "total security bound uses a different denominator from the field policy")
    numerator_text = total_error.get("numerator")
    threshold_text = total_error.get("target_numerator_threshold")
    require(isinstance(numerator_text, str) and CANONICAL_UINT.fullmatch(numerator_text),
            "mechanism security receipt has an invalid total error numerator")
    require(isinstance(threshold_text, str) and CANONICAL_UINT.fullmatch(threshold_text),
            "mechanism security receipt has an invalid target numerator threshold")
    numerator, threshold = int(numerator_text), int(threshold_text)
    denominator_match = POWER_OF_TWO_LOWER_BOUND.fullmatch(field_denominator)
    assert denominator_match is not None
    denominator_bits = int(denominator_match.group(1))
    require(str(denominator_bits) == denominator_match.group(1),
            "mechanism field denominator lower bound is not canonical")
    require(denominator_bits >= 120 and threshold == (1 << (denominator_bits - 120)),
            "mechanism security target threshold does not match the declared denominator and 120-bit target")
    require(numerator <= threshold, "mechanism security error numerator exceeds the 120-bit target threshold")

    # These are the common proof-policy settings. Keep the complete ledger and
    # variant-specific counts separately; never equate them across variants.
    policy = {
        "target_bits": 120,
        "budget_exact": True,
        "bound_kind": "conservative_exact",
        "field_modulus_decimal": field_modulus,
        "field_denominator_lower_bound": field_denominator,
        "computational_assumption_bits": computational_bits,
        "kzg_assumption_bits": kzg_bits,
        "kzg_assumption_profile_id": kzg_profile_id,
        "fiat_shamir_query_budget_max": fs_query_budget_max,
        "fiat_shamir_assumption_profile_id": fs_profile_id,
        "fiat_shamir_security_bits": fs_security_bits,
        "fiat_shamir_random_oracle_terms_128bit": fs_ro_terms,
        "registry_assumption_id": registry_profile_id,
    }
    budget = {
        "valid": True,
        "ledger_sha256": ledger_hash,
        "total_error_bound": total_error,
        "fiat_shamir_challenge_count": fs_challenge_count,
        "kzg_point_group_games": kzg_game_count,
        "total_128bit_terms": assumptions.get("total_128bit_terms"),
        "identity_count": security.get("identity_count"),
    }
    return policy, budget


def validate_run_result(
    result: dict[str, Any],
    *,
    mechanism: str,
    variant: str,
    workload: dict[str, int],
    seed: int,
    backend: str,
    device: int | None,
    proof_path: pathlib.Path,
    statement_path: pathlib.Path,
    proof_bytes: int,
    expected_srs: dict[str, Any],
) -> dict[str, Any]:
    require(result.get("format") == RUN_FORMAT, f"unexpected prover result format: {result.get('format')!r}")
    require(result.get("timing_schema") == TIMING_SCHEMA,
            f"unexpected prover timing schema: {result.get('timing_schema')!r}")
    require(result.get("mechanism") == mechanism, "prover result mechanism does not match requested mechanism")
    require(result.get("variant") == variant, "prover result variant does not match requested variant")
    expected_workload = {**workload, "seed": seed}
    result_workload = result.get("workload")
    require(isinstance(result_workload, dict), "prover result has no workload identity")
    require(all(result_workload.get(key) == value for key, value in expected_workload.items()),
            "prover result workload/seed differs from the requested cell")

    backend_result = result.get("backend")
    require(isinstance(backend_result, dict), "prover result has no backend identity")
    require(backend_result.get("prover") == backend, "prover result backend differs from requested backend")
    require(backend_result.get("verifier") == "cpu", "native verifier backend must identify the strict CPU verifier")
    require(backend_result.get("device") == device, "prover result device differs from requested device")
    cuda_kernel_delta = backend_result.get("cuda_kernel_invocations_delta")
    require(type(cuda_kernel_delta) is int and cuda_kernel_delta >= 0,
            "prover result has no valid native CUDA kernel invocation delta")
    cuda_stats_delta = backend_result.get("cuda_stats_delta")
    require(isinstance(cuda_stats_delta, dict), "prover result has no native CUDA statistics delta")
    require(all(isinstance(name, str) and type(value) is int and value >= 0
                for name, value in cuda_stats_delta.items()),
            "prover result has malformed native CUDA statistics deltas")
    if backend == "cuda":
        require(backend_result.get("actual_cuda") is True, "CUDA was requested but the prover did not attest actual CUDA execution")
        require(cuda_kernel_delta > 0, "CUDA was requested but the native CUDA kernel invocation delta is zero")
    else:
        require(backend_result.get("actual_cuda") is False, "CPU run has inconsistent CUDA execution attestation")
        require(cuda_kernel_delta == 0, "CPU run has a nonzero native CUDA kernel invocation delta")
        require(all(value == 0 for value in cuda_stats_delta.values()),
                "CPU run unexpectedly changed native CUDA statistics")

    security = result.get("security")
    require(isinstance(security, dict), "prover result has no security budget receipt")
    if "security_budget_passed" in security:
        require(security.get("security_budget_passed") is True, "mechanism security budget did not pass")
    security_policy, security_budget = validate_security_receipt(security)

    srs = result.get("srs")
    require(isinstance(srs, dict), "prover result has no SRS identity")
    require(isinstance(srs.get("id"), str) and srs["id"], "prover result has no SRS id")
    require(isinstance(srs.get("manifest_sha256"), str) and HEX_256.fullmatch(srs["manifest_sha256"]),
            "prover result has an invalid SRS manifest digest")
    require(isinstance(srs.get("protocol_schema_id"), str) and srs["protocol_schema_id"],
            "prover result has no protocol schema identity")
    require(srs.get("id") == expected_srs.get("manifest_srs_id"), "prover loaded a different SRS id than the selected manifest")
    require(srs.get("manifest_sha256") == expected_srs.get("manifest_sha256"),
            "prover loaded a different SRS manifest than the selected path")

    pair_digest = result.get("pair_input_commitment_sha256")
    require(isinstance(pair_digest, str) and HEX_256.fullmatch(pair_digest),
            "prover result has no valid pair input commitment digest")
    statement_digest = result.get("statement_sha256")
    require(isinstance(statement_digest, str) and HEX_256.fullmatch(statement_digest),
            "prover result has no valid public statement digest")
    require(statement_path.is_file() and sha256_file(statement_path) == statement_digest,
            "public statement bytes do not match the prover result digest")

    proof = result.get("proof")
    require(isinstance(proof, dict), "prover result has no proof artifact receipt")
    reported_path = proof.get("path")
    require(isinstance(reported_path, str) and pathlib.Path(reported_path).resolve() == proof_path.resolve(),
            "prover result proof path differs from the isolated cell artifact path")
    reported_proof_digest = proof.get("sha256")
    require(isinstance(reported_proof_digest, str) and HEX_256.fullmatch(reported_proof_digest),
            "prover result has an invalid proof digest")
    require(proof_path.is_file() and not proof_path.is_symlink(), "serialized proof artifact is missing")
    actual_proof_digest = sha256_file(proof_path)
    actual_proof_size = proof_path.stat().st_size
    require(actual_proof_digest == reported_proof_digest, "serialized proof bytes differ from prover receipt")
    require(type(proof.get("size_bytes")) is int and proof["size_bytes"] == actual_proof_size == proof_bytes,
            "proof_size_bytes is not the actual canonical serialized artifact length")

    metrics = result.get("metrics")
    require(isinstance(metrics, dict), "prover result has no metrics")
    for name in ("commitment_time_ms", "prove_time_ms"):
        value = metrics.get(name)
        require(type(value) in (int, float) and math.isfinite(float(value)) and value >= 0,
                f"invalid {name}")
    return {
        "security": security,
        "security_policy": security_policy,
        "security_budget": security_budget,
        "cuda_kernel_invocations_delta": cuda_kernel_delta,
        "cuda_stats_delta": cuda_stats_delta,
        "srs": srs,
        "pair_input_commitment_sha256": pair_digest,
        "statement_sha256": statement_digest,
        "proof_sha256": actual_proof_digest,
        "proof_size_bytes": actual_proof_size,
        "commitment_time_ms": float(metrics["commitment_time_ms"]),
        "prove_time_ms": float(metrics["prove_time_ms"]),
        "protocol_schema_id": srs["protocol_schema_id"],
    }


def validate_verify_result(
    result: dict[str, Any],
    *,
    proof_sha256: str,
    statement_sha256: str,
    srs: dict[str, Any],
) -> dict[str, float | int]:
    require(result.get("format") == RUN_FORMAT, f"unexpected verifier result format: {result.get('format')!r}")
    require(result.get("timing_schema") == TIMING_SCHEMA,
            f"unexpected verifier timing schema: {result.get('timing_schema')!r}")
    require(result.get("status") == "VERIFY_OK", f"strict independent verifier status is {result.get('status')!r}")
    require(result.get("proof_sha256") == proof_sha256, "verifier consumed a different proof artifact")
    require(result.get("statement_sha256") == statement_sha256, "verifier consumed a different public statement")
    verify_srs = result.get("srs")
    require(isinstance(verify_srs, dict), "verifier result has no SRS identity")
    require(verify_srs.get("id") == srs.get("id"), "prover and verifier SRS ids differ")
    require(verify_srs.get("manifest_sha256") == srs.get("manifest_sha256"),
            "prover and verifier SRS manifest digests differ")
    require(result.get("verifier_backend") == "cpu", "independent verifier was not the strict CPU verifier")
    cuda_kernel_delta = result.get("verifier_cuda_kernel_invocations_delta")
    require(type(cuda_kernel_delta) is int and cuda_kernel_delta == 0,
            "strict CPU verifier has a nonzero or invalid CUDA kernel invocation delta")
    metrics = result.get("metrics")
    require(isinstance(metrics, dict), "verifier result has no verification metric")
    value = metrics.get("verify_time_ms")
    require(type(value) in (int, float) and math.isfinite(float(value)) and value >= 0,
            "invalid verify_time_ms")
    return {
        "verify_time_ms": float(value),
        "verifier_cuda_kernel_invocations_delta": cuda_kernel_delta,
    }


def run_process(
    command: list[str],
    log_path: pathlib.Path,
    timeout_seconds: float,
    label: str,
    redaction_secret: bytes | None = None,
) -> dict[str, Any]:
    start_utc = utc_now()
    started = time.monotonic()
    return_code: int | None = None
    timed_out = False
    pid: int | None = None
    secret_leaked = False
    output_error: list[BaseException] = []
    with log_path.open("wb") as log:
        rendered = ["<private-seed-file>" if value.endswith(".witness_seed") else value for value in command]
        log.write(f"stage={label}\nstart_utc={start_utc}\ncommand={shlex.join(rendered)}\n".encode("utf-8"))
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid = process.pid

        def drain_output() -> None:
            nonlocal secret_leaked
            assert process.stdout is not None
            pending = b""
            try:
                while True:
                    chunk = process.stdout.read(65536)
                    if not chunk:
                        break
                    data = pending + chunk
                    if redaction_secret and redaction_secret in data:
                        secret_leaked = True
                        data = data.replace(redaction_secret, b"*" * len(redaction_secret))
                    if redaction_secret:
                        emit_length = max(0, len(data) - len(redaction_secret) + 1)
                        log.write(data[:emit_length])
                        pending = data[emit_length:]
                    else:
                        log.write(data)
                if pending:
                    if redaction_secret and redaction_secret in pending:
                        secret_leaked = True
                        pending = pending.replace(redaction_secret, b"*" * len(redaction_secret))
                    log.write(pending)
                log.flush()
            except BaseException as exc:
                output_error.append(exc)

        reader = threading.Thread(target=drain_output, daemon=True)
        reader.start()
        try:
            return_code = process.wait(timeout=max(0.0, timeout_seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                return_code = process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return_code = process.wait()
        reader.join(timeout=10)
        if reader.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            reader.join()
        if output_error:
            raise OSError(f"failed to capture {label} output: {output_error[0]}")
        end_utc = utc_now()
        elapsed = time.monotonic() - started
        log.write(
            f"end_utc={end_utc}\nprocess_pid={pid}\nreturn_code={return_code}\n"
            f"timed_out={str(timed_out).lower()}\nwall_time_seconds={elapsed:.6f}\n"
            .encode("utf-8")
        )
    return {
        "pid": pid,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "return_code": return_code,
        "timed_out": timed_out,
        "secret_leaked": secret_leaked,
        "wall_time_seconds": elapsed,
        "log": str(log_path),
    }


def write_csv(path: pathlib.Path, cells: list[dict[str, Any]]) -> None:
    fields = [
        "mechanism", "variant", "workload_id", "nodes", "edges", "width", "graphs",
        "repetition", "seed", "backend", "device", "cell_status", "pair_status",
        "verification_status", *METRIC_FIELDS, "pair_input_commitment_sha256", "srs_id",
        "srs_manifest_sha256", "protocol_schema_id", "security_budget_ledger_sha256",
        "security_budget_valid", "security_target_bits", "security_bound_kind",
        "security_error_numerator", "security_target_numerator_threshold",
        "security_fiat_shamir_query_budget_max", "security_fiat_shamir_challenge_count",
        "security_kzg_point_group_games",
        "security_total_128bit_terms", "security_identity_count", "proof_path", "proof_sha256",
        "prover_exit_code", "verifier_exit_code", "prover_log", "verifier_log",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for cell in cells:
            writer.writerow({field: cell.get(field, "") for field in fields})


def summarize_pairs(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int, int, int, int], dict[str, dict[str, Any]]] = {}
    for cell in cells:
        key = (
            cell["mechanism"], cell["workload_id"], cell["nodes"], cell["edges"],
            cell["width"], cell["graphs"], cell["repetition"],
        )
        grouped.setdefault(key, {})[cell["variant"]] = cell
    pairs: list[dict[str, Any]] = []
    for key, variants in sorted(grouped.items()):
        original = variants.get("original")
        control = variants.get("control")
        valid = bool(
            original and control
            and original.get("verification_status") == "VERIFY_OK"
            and control.get("verification_status") == "VERIFY_OK"
            and original.get("pair_status") == "paired"
            and control.get("pair_status") == "paired"
        )
        pair = {
            "mechanism": key[0], "workload_id": key[1], "nodes": key[2], "edges": key[3],
            "width": key[4], "graphs": key[5], "repetition": key[6],
            "status": "complete" if valid else "incomplete",
            "original": original,
            "control": control,
        }
        if original and control:
            pair["security_policy"] = {
                "original": original.get("security_policy"),
                "control": control.get("security_policy"),
                "matches": original.get("security_policy") == control.get("security_policy"),
            }
            # Preserve each full, variant-specific receipt independently. The
            # two ledgers intentionally have different challenge/opening counts.
            pair["security_ledgers"] = {
                "original": original.get("security"),
                "control": control.get("security"),
            }
        if valid:
            pair["metrics"] = {
                "original": {name: original.get(name) for name in METRIC_FIELDS},
                "control": {name: control.get(name) for name in METRIC_FIELDS},
            }
            deltas: dict[str, dict[str, float | None]] = {}
            for name in METRIC_FIELDS:
                a = float(original[name])
                b = float(control[name])
                deltas[name] = {"control_minus_original": b - a, "control_over_original_percent": None if a == 0 else (b - a) * 100.0 / a}
            pair["deltas"] = deltas
        pairs.append(pair)
    return pairs


def grouped_summary(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, int, int, int, int], list[dict[str, Any]]] = {}
    for pair in pairs:
        key = (pair["mechanism"], pair["workload_id"], pair["nodes"], pair["edges"], pair["width"], pair["graphs"])
        buckets.setdefault(key, []).append(pair)
    rows: list[dict[str, Any]] = []
    for key, group in sorted(buckets.items()):
        valid_pairs = [pair for pair in group if pair["status"] == "complete"]
        metrics: dict[str, dict[str, Any]] = {}
        for name in METRIC_FIELDS:
            entry: dict[str, Any] = {}
            metric_pairs: list[tuple[float, float]] = []
            for variant in VARIANTS:
                values = [float(pair["metrics"][variant][name]) for pair in valid_pairs]
                entry[variant] = sum(values) / len(values) if values else None
            metric_pairs = [
                (float(pair["metrics"]["original"][name]), float(pair["metrics"]["control"][name]))
                for pair in valid_pairs
            ]
            original = entry["original"]
            control = entry["control"]
            entry["control_minus_original"] = None if original is None or control is None else control - original
            entry["control_over_original_percent"] = (
                None if original is None or control is None or original == 0
                else (control - original) * 100.0 / original
            )
            paired_deltas = [control_value - original_value for original_value, control_value in metric_pairs]
            entry["sample_stddev"] = {
                "original": statistics.stdev([pair[0] for pair in metric_pairs]) if len(metric_pairs) >= 2 else None,
                "control": statistics.stdev([pair[1] for pair in metric_pairs]) if len(metric_pairs) >= 2 else None,
                "control_minus_original": statistics.stdev(paired_deltas) if len(paired_deltas) >= 2 else None,
            }
            metrics[name] = entry
        rows.append({
            "mechanism": key[0], "workload_id": key[1], "nodes": key[2], "edges": key[3],
            "width": key[4], "graphs": key[5], "requested_pairs": len(group),
            "complete_pairs": len(valid_pairs), "metrics": metrics,
        })
    return rows


def write_summary_markdown(path: pathlib.Path, document: dict[str, Any]) -> None:
    lines = [
        "# zkGAT Mechanism Ablation Results",
        "",
        f"Session: `{document['session_id']}`",
        f"Status: **{document['session_status']}**",
        f"Mechanism(s): `{', '.join(document['mechanisms'])}`; backend: `{document['backend']}`",
        f"Independent verifier: strict CPU verifier in a separate process; accepted cells: {document['verified_cells']}/{document['cell_count']}",
        "",
        "Only complete pairs with two serialized proofs and independent `VERIFY_OK` results contribute to aggregate metrics. Means and sample standard deviations use complete repetitions; standard deviation is shown as an em dash when fewer than two pairs are available.",
        "Commitment and prove are disjoint native prover partitions; verify includes parsing through accept/reject; proof size is the actual canonical proof file length.",
        "",
        "| Mechanism | Workload (N/E/d/G) | Complete pairs | Metric | Original mean | Original sample SD | Control mean | Control sample SD | Paired delta mean | Paired delta sample SD | Change % |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in document["groups"]:
        workload = f"{row['nodes']}/{row['edges']}/{row['width']}/{row['graphs']}"
        for name in METRIC_FIELDS:
            metric = row["metrics"][name]
            def render(value: Any) -> str:
                return "—" if value is None else (str(int(value)) if name == "proof_size_bytes" else f"{float(value):.6f}")
            percent = metric["control_over_original_percent"]
            standard_deviation = metric["sample_stddev"]
            lines.append(
                f"| {row['mechanism']} | {workload} | {row['complete_pairs']}/{row['requested_pairs']} | `{name}` | "
                f"{render(metric['original'])} | {render(standard_deviation['original'])} | "
                f"{render(metric['control'])} | {render(standard_deviation['control'])} | "
                f"{render(metric['control_minus_original'])} | {render(standard_deviation['control_minus_original'])} | "
                f"{'—' if percent is None else f'{percent:.3f}%'} |"
            )
    lines.extend([
        "", "## Cell results", "",
        "| Mechanism | Workload | Rep | Variant | Verification | Pair | Commitment ms | Prove ms | Verify ms | Proof bytes | Budget valid | Ledger SHA-256 | FS challenges | KZG games |",
        "|---|---|---:|---|---|---|---:|---:|---:|---:|---|---|---:|---:|",
    ])
    for cell in document["cells"]:
        def cell_metric(name: str) -> str:
            value = cell.get(name)
            if value is None:
                return "—"
            return str(int(value)) if name == "proof_size_bytes" else f"{float(value):.6f}"
        lines.append(
            f"| {cell['mechanism']} | {cell['workload_id']} | {cell['repetition']} | {cell['variant']} | "
            f"{cell.get('verification_status', 'NOT_RUN')} | {cell.get('pair_status', 'incomplete')} | "
            f"{cell_metric('commitment_time_ms')} | {cell_metric('prove_time_ms')} | "
            f"{cell_metric('verify_time_ms')} | {cell_metric('proof_size_bytes')} | "
            f"{cell.get('security_budget_valid', '—')} | `{cell.get('security_budget_ledger_sha256', '—')}` | "
            f"{cell.get('security_fiat_shamir_challenge_count', '—')} | {cell.get('security_kzg_point_group_games', '—')} |"
        )
    lines.extend(["", "## Variant-specific security ledgers", ""])
    for pair in document["pairs"]:
        ledgers = pair.get("security_ledgers")
        if not isinstance(ledgers, dict):
            continue
        lines.append(
            f"- `{pair['mechanism']}` `{pair['workload_id']}` rep {pair['repetition']}: "
            f"common policy match = `{pair['security_policy']['matches']}`."
        )
        for variant in VARIANTS:
            ledger = ledgers.get(variant)
            if not isinstance(ledger, dict):
                lines.append(f"  - {variant}: no accepted security receipt.")
                continue
            total_error = ledger.get("total_error_bound", {})
            assumptions = ledger.get("assumptions", {})
            lines.append(
                f"  - {variant}: ledger `{ledger.get('budget_ledger_sha256', '—')}`, "
                f"total error numerator `{total_error.get('numerator', '—')}` / target "
                f"`{total_error.get('target_numerator_threshold', '—')}`, "
                f"FS challenges `{assumptions.get('fiat_shamir_challenge_count', '—')}`, "
                f"KZG games `{assumptions.get('kzg_point_group_games', '—')}`, "
                f"identity count `{ledger.get('identity_count', '—')}`."
            )
    lines.extend([
        "", "## Provenance", "",
        f"- Git HEAD: `{document['provenance']['source']['git_head']}`",
        f"- Source tree SHA-256: `{document['provenance']['source']['source_tree_hash']}`",
        f"- Git diff SHA-256: `{document['provenance']['source']['git_diff_sha256']}`",
        f"- Binary SHA-256: `{document['provenance']['build']['binary_sha256']}`",
        f"- Build cache SHA-256: `{document['provenance']['build']['cmake_cache_sha256']}`",
        f"- SRS id / manifest SHA-256: `{document['provenance']['srs']['manifest_srs_id']}` / `{document['provenance']['srs']['manifest_sha256']}`",
        f"- Device: `{document['provenance']['device'].get('selected_device', document['provenance']['device'])}`",
        f"- Output directory: `{document['session_root']}`",
        "",
    ])
    atomic_write_text(path, "\n".join(lines))


def run_cell(
    *,
    binary: pathlib.Path,
    session_root: pathlib.Path,
    mechanism: str,
    variant: str,
    workload: dict[str, int],
    repetition: int,
    seed: int,
    backend: str,
    device: int | None,
    srs_path: pathlib.Path,
    witness_seed_file: pathlib.Path,
    expected_srs: dict[str, Any],
    timeout_seconds: int,
    pair_index: int,
) -> dict[str, Any]:
    workload_id = workload_slug(workload)
    cell_root = session_root / mechanism / workload_id / f"rep_{repetition:03d}" / variant
    cell_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    proof_path = cell_root / "proof.bin"
    statement_path = cell_root / "public_statement.json"
    prover_result_path = cell_root / "prover_result.json"
    verifier_result_path = cell_root / "verifier_result.json"
    prover_log = cell_root / "prover.log"
    verifier_log = cell_root / "verifier.log"
    common = {
        "mechanism": mechanism,
        "variant": variant,
        "workload_id": workload_id,
        **workload,
        "repetition": repetition,
        "seed": seed,
        "backend": backend,
        "device": device,
        "cell_status": "failed",
        "pair_status": "incomplete",
        "verification_status": "NOT_RUN",
        "proof_path": str(proof_path),
        "prover_log": str(prover_log),
        "verifier_log": str(verifier_log),
        "prover_exit_code": None,
        "verifier_exit_code": None,
        "prover_process_pid": None,
        "verifier_process_pid": None,
        "failure_reason": None,
    }

    run_command = [
        str(binary), "run",
        "--mechanism", mechanism,
        "--variant", variant,
        "--nodes", str(workload["nodes"]),
        "--edges", str(workload["edges"]),
        "--width", str(workload["width"]),
        "--graphs", str(workload["graphs"]),
        "--seed", str(seed),
        "--backend", backend,
    ]
    if backend == "cuda":
        run_command.extend(["--device", str(device)])
    run_command.extend([
        "--srs", str(srs_path),
        "--witness-seed-file", str(witness_seed_file),
        "--proof-out", str(proof_path),
        "--statement-out", str(statement_path),
        "--result-json", str(prover_result_path),
    ])
    started = time.monotonic()
    try:
        secret_bytes = witness_seed_file.read_bytes().strip()
        require(len(secret_bytes) == 64, "private witness seed file has an unexpected encoding")
        prover_proc = run_process(run_command, prover_log, timeout_seconds, "prover", secret_bytes)
        common["prover_exit_code"] = prover_proc["return_code"]
        common["prover_process_pid"] = prover_proc["pid"]
        common["prover_wall_time_seconds"] = prover_proc["wall_time_seconds"]
        if prover_proc["timed_out"]:
            raise ValueError(f"prover exceeded the {timeout_seconds}s per-cell limit")
        require(not prover_proc["secret_leaked"], "native prover attempted to write the private witness seed to its log")
        require(prover_proc["return_code"] == 0, f"prover exited with code {prover_proc['return_code']}")
        for output_path in (prover_result_path, statement_path, proof_path):
            if output_path.is_file():
                require(not file_contains_bytes(output_path, secret_bytes), f"private witness seed appeared in {output_path.name}")
        prover_result = load_json(prover_result_path, "prover result")
        require(proof_path.is_file() and not proof_path.is_symlink(), "prover did not serialize a regular proof artifact")
        proof_size = proof_path.stat().st_size
        receipt = validate_run_result(
            prover_result,
            mechanism=mechanism,
            variant=variant,
            workload=workload,
            seed=seed,
            backend=backend,
            device=device,
            proof_path=proof_path,
            statement_path=statement_path,
            proof_bytes=proof_size,
            expected_srs=expected_srs,
        )
        total_error_bound = receipt["security_budget"]["total_error_bound"]
        common.update({
            "pair_input_commitment_sha256": receipt["pair_input_commitment_sha256"],
            "statement_sha256": receipt["statement_sha256"],
            "proof_sha256": receipt["proof_sha256"],
            "srs_id": receipt["srs"]["id"],
            "srs_manifest_sha256": receipt["srs"]["manifest_sha256"],
            "protocol_schema_id": receipt["protocol_schema_id"],
            "security": receipt["security"],
            "security_policy": receipt["security_policy"],
            "security_budget": receipt["security_budget"],
            "security_budget_ledger_sha256": receipt["security_budget"]["ledger_sha256"],
            "security_budget_valid": receipt["security_budget"]["valid"],
            "security_target_bits": receipt["security_policy"]["target_bits"],
            "security_bound_kind": receipt["security_policy"]["bound_kind"],
            "security_error_numerator": total_error_bound["numerator"],
            "security_target_numerator_threshold": total_error_bound["target_numerator_threshold"],
            "security_fiat_shamir_query_budget_max": receipt["security_policy"]["fiat_shamir_query_budget_max"],
            "security_fiat_shamir_challenge_count": receipt["security_budget"]["fiat_shamir_challenge_count"],
            "security_kzg_point_group_games": receipt["security_budget"]["kzg_point_group_games"],
            "security_total_128bit_terms": receipt["security_budget"]["total_128bit_terms"],
            "security_identity_count": receipt["security_budget"]["identity_count"],
            "cuda_kernel_invocations_delta": receipt["cuda_kernel_invocations_delta"],
            "cuda_stats_delta": receipt["cuda_stats_delta"],
            "commitment_time_ms": receipt["commitment_time_ms"],
            "prove_time_ms": receipt["prove_time_ms"],
            "proof_size_bytes": receipt["proof_size_bytes"],
        })

        remaining = timeout_seconds - (time.monotonic() - started)
        require(remaining > 0, f"prover consumed the {timeout_seconds}s per-cell limit before independent verification")
        verify_command = [
            str(binary), "verify",
            "--proof", str(proof_path),
            "--statement", str(statement_path),
            "--srs", str(srs_path),
            "--result-json", str(verifier_result_path),
        ]
        verifier_proc = run_process(verify_command, verifier_log, remaining, "independent-verifier")
        common["verifier_exit_code"] = verifier_proc["return_code"]
        common["verifier_process_pid"] = verifier_proc["pid"]
        common["verify_wall_time_seconds"] = verifier_proc["wall_time_seconds"]
        if verifier_proc["timed_out"]:
            raise ValueError(f"independent verifier exceeded the remaining {timeout_seconds}s per-cell limit")
        require(verifier_proc["return_code"] == 0, f"independent verifier exited with code {verifier_proc['return_code']}")
        verify_result = load_json(verifier_result_path, "independent verifier result")
        require(not file_contains_bytes(verifier_result_path, secret_bytes), "private witness seed appeared in verifier output")
        verifier_receipt = validate_verify_result(
            verify_result,
            proof_sha256=receipt["proof_sha256"],
            statement_sha256=receipt["statement_sha256"],
            srs=receipt["srs"],
        )
        require(common["prover_process_pid"] != common["verifier_process_pid"],
                "proof and verification did not run in separate processes")
        common["verify_time_ms"] = verifier_receipt["verify_time_ms"]
        common["verifier_cuda_kernel_invocations_delta"] = verifier_receipt["verifier_cuda_kernel_invocations_delta"]
        common["verification_status"] = "VERIFY_OK"
        common["cell_status"] = "complete"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        common["failure_reason"] = str(exc)
        common["cell_status"] = "failed"
        if verifier_result_path.exists():
            try:
                diagnostic = load_json(verifier_result_path, "verifier diagnostic")
                if diagnostic.get("status") == "VERIFY_OK":
                    common["verification_status"] = "VERIFY_OK"
            except ValueError:
                pass
    common["cell_wall_time_seconds"] = time.monotonic() - started
    common["pair_order_index"] = pair_index
    atomic_write_json(cell_root / "cell.json", common)
    return common


def mark_pair(cells: list[dict[str, Any]], mechanism: str, workload_id: str, repetition: int) -> None:
    pair_cells = [
        cell for cell in cells
        if cell["mechanism"] == mechanism and cell["workload_id"] == workload_id and cell["repetition"] == repetition
    ]
    by_variant = {cell["variant"]: cell for cell in pair_cells}
    original, control = by_variant.get("original"), by_variant.get("control")
    pair_ok = bool(
        original and control
        and original.get("cell_status") == "complete"
        and control.get("cell_status") == "complete"
        and original.get("pair_input_commitment_sha256") == control.get("pair_input_commitment_sha256")
        and original.get("srs_id") == control.get("srs_id")
        and original.get("srs_manifest_sha256") == control.get("srs_manifest_sha256")
        and original.get("protocol_schema_id") == control.get("protocol_schema_id")
        and original.get("backend") == control.get("backend")
        and original.get("device") == control.get("device")
        and original.get("security_budget_valid") is True
        and control.get("security_budget_valid") is True
        and original.get("security_policy") == control.get("security_policy")
        and (original.get("nodes"), original.get("edges"), original.get("width"), original.get("graphs"), original.get("seed"))
            == (control.get("nodes"), control.get("edges"), control.get("width"), control.get("graphs"), control.get("seed"))
    )
    for cell in pair_cells:
        cell["pair_status"] = "paired" if pair_ok else "invalid_pair"
        if not pair_ok and cell.get("failure_reason") is None:
            cell["failure_reason"] = "paired workloads, witness commitment, SRS identity, or common security policy differ"
        cell_path = pathlib.Path(cell["prover_log"]).parent / "cell.json"
        atomic_write_json(cell_path, cell)


def execute(args: argparse.Namespace) -> int:
    workloads = validate_workloads(args)
    require(1 <= args.timeout_seconds <= 599, "--timeout-seconds must be between 1 and 599 seconds")
    require(args.repetitions > 0, "--repetitions must be positive")
    require(args.seed + args.repetitions - 1 <= (1 << 63) - 1, "seed plus repetition exceeds signed 64-bit range")
    mechanisms = list(MECHANISMS) if args.mechanism == "all" else [args.mechanism]
    device = args.device if args.backend == "cuda" else None
    if args.backend == "cuda":
        require(device is not None and device >= 0, "CUDA experiments require a non-negative --device index")
        device_snapshot = cuda_snapshot(device)
    else:
        require(args.device is None, "--device is only valid with --backend cuda")
        device_snapshot = {"backend": "cpu", "selected_device": "cpu", "cuda_visible_devices": None, "inventory": []}

    binary = resolve_binary(args)
    source_start = source_snapshot()
    build = build_snapshot(binary, args.backend)
    require(build.get("built_source_tree_hash") == source_start["source_tree_hash"],
            "selected binary was built from a different source inventory; rebuild before running mechanism proofs")
    srs = srs_snapshot(args.srs)
    srs_path = pathlib.Path(srs["path"])

    output_root = safe_output_root(args.output_root)
    session_id, session_root = create_session_dir(output_root)
    os.umask(0o077)
    atomic_write_json(session_root / "source_manifest.json", source_start["source_manifest"])
    atomic_write_json(session_root / "build_source_manifest.json", build["source_manifest"])
    provenance = {
        "source": {key: value for key, value in source_start.items() if key != "source_manifest"},
        "build": {key: value for key, value in build.items() if key != "source_manifest"},
        "srs": srs,
        "device": device_snapshot,
    }
    atomic_write_json(session_root / "provenance.json", provenance)
    session_log = session_root / "session.log"
    emit(f"MECHANISM_SESSION_STARTED session_id={session_id} output={session_root}", session_log)

    cells: list[dict[str, Any]] = []
    pair_index = 0
    for workload in workloads:
        for repetition in range(args.repetitions):
            seed = args.seed + repetition
            for mechanism in mechanisms:
                pair_dir = session_root / ".pair_secrets"
                pair_dir.mkdir(mode=0o700, exist_ok=True)
                witness_seed_file = pair_dir / f"pair_{pair_index:04d}.witness_seed"
                fd = os.open(witness_seed_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                ACTIVE_SECRET_FILES.add(witness_seed_file)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(os.urandom(32).hex().encode("ascii") + b"\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                except Exception:
                    witness_seed_file.unlink(missing_ok=True)
                    raise
                variants = list(VARIANTS)
                if pair_index % 2:
                    variants.reverse()
                workload_id = workload_slug(workload)
                emit(
                    f"PAIR_START mechanism={mechanism} workload={workload_id} repetition={repetition} "
                    f"seed={seed} order={','.join(variants)} backend={args.backend} device={device}",
                    session_log,
                )
                for variant in variants:
                    emit(f"CELL_START mechanism={mechanism} workload={workload_id} repetition={repetition} variant={variant}", session_log)
                    cell = run_cell(
                        binary=binary,
                        session_root=session_root,
                        mechanism=mechanism,
                        variant=variant,
                        workload=workload,
                        repetition=repetition,
                        seed=seed,
                        backend=args.backend,
                        device=device,
                        srs_path=srs_path,
                        witness_seed_file=witness_seed_file,
                        expected_srs=srs,
                        timeout_seconds=args.timeout_seconds,
                        pair_index=pair_index,
                    )
                    cells.append(cell)
                    emit(
                        f"CELL_END mechanism={mechanism} workload={workload_id} repetition={repetition} "
                        f"variant={variant} status={cell['cell_status']} verify={cell['verification_status']} "
                        f"exit={cell['prover_exit_code']}/{cell['verifier_exit_code']}",
                        session_log,
                    )
                mark_pair(cells, mechanism, workload_id, repetition)
                witness_seed_file.unlink(missing_ok=True)
                ACTIVE_SECRET_FILES.discard(witness_seed_file)
                try:
                    pair_dir.rmdir()
                except OSError:
                    pass
                pair_index += 1

    source_end = source_snapshot()
    build_end = build_snapshot(binary, args.backend)
    srs_end = srs_snapshot(srs_path)
    provenance_stable = (
        source_start["source_tree_hash"] == source_end["source_tree_hash"]
        and source_start["git_diff_sha256"] == source_end["git_diff_sha256"]
        and source_start["git_status_sha256"] == source_end["git_status_sha256"]
        and build["binary_sha256"] == build_end["binary_sha256"]
        and build["cmake_cache_sha256"] == build_end["cmake_cache_sha256"]
        and build["source_manifest_sha256"] == build_end["source_manifest_sha256"]
        and srs["manifest_sha256"] == srs_end["manifest_sha256"]
    )
    for cell in cells:
        if not provenance_stable:
            cell["pair_status"] = "invalid_provenance"
            if cell.get("failure_reason") is None:
                cell["failure_reason"] = "source/build/SRS provenance changed during the session"
            atomic_write_json(pathlib.Path(cell["prover_log"]).parent / "cell.json", cell)

    pairs = summarize_pairs(cells)
    groups = grouped_summary(pairs)
    verified_cells = sum(cell.get("verification_status") == "VERIFY_OK" for cell in cells)
    complete_pairs = sum(pair["status"] == "complete" for pair in pairs)
    session_status = "complete" if provenance_stable and complete_pairs == len(pairs) and len(pairs) > 0 else "incomplete"
    document = {
        "format": "zkgat_mechanism_paired_experiment_v1",
        "session_id": session_id,
        "session_root": str(session_root),
        "created_utc": utc_now(),
        "session_status": session_status,
        "mechanisms": mechanisms,
        "variants": list(VARIANTS),
        "backend": args.backend,
        "device": device,
        "srs": srs,
        "workloads": workloads,
        "repetitions": args.repetitions,
        "base_seed": args.seed,
        "timeout_seconds_per_cell": args.timeout_seconds,
        "timing_semantics": TIMING_SEMANTICS,
        "provenance_stable": provenance_stable,
        "provenance": provenance,
        "provenance_end": {
            "source_tree_hash": source_end["source_tree_hash"],
            "git_diff_sha256": source_end["git_diff_sha256"],
            "git_status_sha256": source_end["git_status_sha256"],
            "binary_sha256": build_end["binary_sha256"],
            "cmake_cache_sha256": build_end["cmake_cache_sha256"],
            "source_manifest_sha256": build_end["source_manifest_sha256"],
            "srs_manifest_sha256": srs_end["manifest_sha256"],
        },
        "cell_count": len(cells),
        "verified_cells": verified_cells,
        "pair_count": len(pairs),
        "complete_pairs": complete_pairs,
        "cells": cells,
        "pairs": pairs,
        "groups": groups,
    }
    atomic_write_json(session_root / "summary.json", document)
    write_csv(session_root / "cells.csv", cells)
    write_summary_markdown(session_root / "summary.md", document)
    emit(
        f"MECHANISM_SESSION_END session_id={session_id} status={session_status} "
        f"cells={len(cells)} verified={verified_cells} pairs={complete_pairs}/{len(pairs)} "
        f"summary={session_root / 'summary.md'}",
        session_log,
    )
    return 0 if session_status == "complete" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paired zkGAT mechanism proofs and separate-process verification.")
    parser.add_argument("--mechanism", choices=(*MECHANISMS, "all"), default="all",
                        help="mechanism to compare, or all (default: all)")
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--device", type=integer_arg, help="CUDA device index; required for cuda")
    parser.add_argument("--binary", type=pathlib.Path, help="mechanism executable (default: build[_cuda]/gatzk_mechanism)")
    parser.add_argument("--srs", type=pathlib.Path, required=True, help="trusted SRS directory with srs_manifest.json")
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT,
                        help="new-session parent under runs/mechanism_ablation_secure")
    parser.add_argument("--nodes", type=positive_arg)
    parser.add_argument("--edges", type=positive_arg)
    parser.add_argument("--width", type=positive_arg)
    parser.add_argument("--graphs", type=positive_arg)
    parser.add_argument("--workload", type=parse_workload, action="append",
                        help="repeatable N,E,D,G workload (nodes,edges,width,graphs)")
    parser.add_argument("--repetitions", type=positive_arg, default=1)
    parser.add_argument("--seed", type=integer_arg, default=11, help="public topology seed; paired variants share it")
    parser.add_argument("--timeout-seconds", type=positive_arg, default=540,
                        help="maximum prover plus independent-verifier wall time per side (1..599)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return execute(args)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"mechanism-runner: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
