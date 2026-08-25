#!/usr/bin/env python3
"""Fail-closed structural validation of collected formal ezkl GAT results."""

from __future__ import annotations

import hashlib
import json
import argparse
import struct
import math
from pathlib import Path

from resource_policy import (  # noqa: E402
    CELL_ARTIFACTS,
    DATASET_ORDER,
    canonical_cell_dir,
    validate_cell_inventory,
)
from build_identity_contract import identity_digest, validate_build_identity  # noqa: E402
from formal_source_manifest import formal_source_id  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROUTE_ID = "ezkl_gat_native_compressed_csr_attention_full"
FORMAL_SOURCE_SRS_MAX_K = 28
FORMAL_SEMANTIC_VERSION = "ezkl_formal_gat_semantics_v2"
REQUIRED = tuple(sorted(CELL_ARTIFACTS - {"run_manifest.json"}))


def _valid_cuda_prove_telemetry(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("schema") == "ezkl_cuda_kernel_telemetry_v1"
        and payload.get("scope") == "formal_prove"
        and isinstance(payload.get("kernel_count"), int)
        and payload.get("kernel_count", 0) > 0
        and payload.get("dropped_records") == 0
        and payload.get("active_device_id") == 0
        and bool(payload.get("scope_nonce"))
        and payload.get("scope_end_ns", 0) >= payload.get("scope_start_ns", 1)
        and isinstance(payload.get("kernel_names"), list)
        and bool(payload.get("kernel_names"))
        and all(
            isinstance(kernel, dict)
            and kernel.get("name")
            and kernel.get("category")
            and isinstance(kernel.get("count"), int)
            and kernel.get("count", 0) > 0
            and kernel.get("device_id") == 0
            for kernel in payload.get("kernel_names", [])
        )
        and sum(kernel["count"] for kernel in payload["kernel_names"])
        == payload["kernel_count"]
    )


def validate_matrix_records(
    records: list[dict[str, object]],
    *,
    allow_incomplete: bool = False,
    fixture: bool = False,
) -> list[str]:
    """Validate matrix indexing and backend claims independently of artifacts.

    ``fixture=True`` is reserved for tests-only mock contracts.  It never
    accepts or emits a production proof/verification result and additionally
    requires explicit fixture attestations for source, SRS, resources, and
    proof digest/size binding.
    """

    expected = {(name, device) for name in DATASET_ORDER for device in ("cpu", "cuda")}
    seen: set[tuple[object, object]] = set()
    failures: list[str] = []
    for record in records:
        if fixture and record.get("fixture_kind") != "tests_only_mock_contract_v1":
            failures.append("fixture record is not explicitly tests-only")
        dataset = record.get("dataset")
        device = record.get("device")
        cell = (dataset, device)
        if cell not in expected:
            failures.append(f"unknown matrix cell: {dataset}/{device}")
            continue
        if cell in seen:
            failures.append(f"duplicate matrix cell: {dataset}/{device}")
        seen.add(cell)
        if fixture:
            if record.get("verified_flag") is not True:
                failures.append(f"fixture cell is not verified: {dataset}/{device}")
            attestation = record.get("attestation")
            if not isinstance(attestation, dict):
                failures.append(f"fixture attestation missing: {dataset}/{device}")
            else:
                if attestation.get("build_identity_bound") is not True:
                    failures.append(f"fixture build identity binding missing: {dataset}/{device}")
                if attestation.get("source_bound") is not True:
                    failures.append(f"fixture source binding missing: {dataset}/{device}")
                if attestation.get("source_srs_k") != FORMAL_SOURCE_SRS_MAX_K:
                    failures.append(f"fixture source SRS k mismatch: {dataset}/{device}")
                if attestation.get("config_bound") is not True:
                    failures.append(f"fixture config binding missing: {dataset}/{device}")
                if attestation.get("resource_bound") is not True:
                    failures.append(f"fixture resource binding missing: {dataset}/{device}")
                if not isinstance(attestation.get("proof_size_bytes"), int) or attestation.get("proof_size_bytes", 0) <= 0:
                    failures.append(f"fixture proof size is invalid: {dataset}/{device}")
                if not isinstance(attestation.get("proof_sha256"), str) or not attestation["proof_sha256"].startswith("sha256:"):
                    failures.append(f"fixture proof digest is invalid: {dataset}/{device}")
            if device == "cpu":
                if record.get("gpu_prove_telemetry") != "not_applicable":
                    failures.append(f"CPU fixture carries GPU telemetry: {dataset}")
                if record.get("verifier_backend") != "cpu_independent":
                    failures.append(f"CPU fixture verifier is not independent: {dataset}")
            elif (
                record.get("gpu_backend_execution") != "cuda_icicle_prove_only"
                or not _valid_cuda_prove_telemetry(record.get("gpu_prove_telemetry"))
                or record.get("verifier_backend") != "cpu_independent"
            ):
                failures.append(f"CUDA fixture telemetry/verifier contract mismatch: {dataset}")
        elif not allow_incomplete and (
            record.get("status") != "verified" or record.get("verify_passed") is not True
        ):
            failures.append(f"required matrix cell is not verified: {dataset}/{device}")
        if not fixture and device == "cpu" and record.get("gpu_prove_telemetry") != "not_applicable":
            failures.append(f"CPU result carries GPU prove telemetry: {dataset}")
        if not fixture and device == "cuda" and not _valid_cuda_prove_telemetry(record.get("gpu_prove_telemetry")):
            failures.append(f"CUDA result lacks valid prove telemetry: {dataset}")
    if not allow_incomplete:
        for dataset, device in sorted(expected - seen):
            failures.append(f"missing required matrix cell: {dataset}/{device}")
    return failures


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def formal_digest(domain: str, role: str, payload: bytes) -> str:
    prefix = b"GAT-ZKML/ezkl/canonical/v1"
    encoded = struct.pack("<Q", len(prefix)) + prefix
    encoded += struct.pack("<Q", len(domain)) + domain.encode()
    encoded += struct.pack("<Q", 1)
    encoded += struct.pack("<Q", len(role)) + role.encode()
    encoded += struct.pack("<Q", len(payload)) + payload
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def canonical_json(payload: object) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()


def operation_registry_id() -> str:
    return formal_digest(
        "ezkl_formal_operation_registry", "canonical_json",
        (ROOT / "configs" / "formal_operation_registry.json").read_bytes().rstrip(b"\r\n"),
    )


def round_away(value: float) -> int:
    return int(value + 0.5) if value >= 0 else int(value - 0.5)


def signed_round_div(value: int, denominator: int) -> int:
    magnitude = abs(value)
    quotient = (2 * magnitude + denominator) // (2 * denominator)
    return -quotient if value < 0 else quotient


def validate_rows(path: Path, domain: str, expected_digest: str,
                  domains: dict[int, tuple[int, int]]) -> list[str]:
    failures: list[str] = []
    payload = path.read_bytes()
    if formal_digest(domain, "canonical_rows", payload) != expected_digest:
        return [f"{path}: canonical rows digest mismatch"]
    if not payload.startswith(b"EZKLGATTABLE1\n") or (len(payload) - 14) % 24:
        return [f"{path}: invalid canonical row encoding"]
    counts = {kind: 0 for kind in domains}
    for offset in range(14, len(payload), 24):
        kind, value, output = struct.unpack_from("<Qqq", payload, offset)
        if kind not in domains:
            failures.append(f"{path}: unexpected table kind {kind}")
            break
        start, end = domains[kind]
        expected_value = start + counts[kind]
        if value != expected_value or value > end:
            failures.append(f"{path}: noncanonical kind/input order")
            break
        if kind == 1:
            expected_output = value
        elif kind == 2:
            expected_output = value if value >= 0 else -signed_round_div(-value, 5)
        elif kind == 3:
            expected_output = value if value >= 0 else round_away(
                16.0 * (math.exp(value / 16.0) - 1.0)
            )
        elif kind == 4:
            expected_output = max(0, round_away(16.0 * math.exp(-value / 256.0)))
        else:
            expected_output = max(value, 0)
        if output != expected_output:
            failures.append(f"{path}: kind {kind} row semantics mismatch at {value}")
            break
        counts[kind] += 1
    for kind, (start, end) in domains.items():
        if counts[kind] != end - start + 1:
            failures.append(f"{path}: kind {kind} domain/row count mismatch")
    return failures


def validate_table_artifacts(run_dir: Path) -> list[str]:
    failures: list[str] = []
    main = json.loads((run_dir / "formal_table_spec.json").read_text(encoding="utf-8"))
    aux = json.loads((run_dir / "backend_aux_table_spec.json").read_text(encoding="utf-8"))
    spec = json.loads((run_dir / "graph_attention_spec.json").read_text(encoding="utf-8"))
    quant = json.loads((run_dir / "formal_quant_config.json").read_text(encoding="utf-8"))
    for profile, domain in ((main, "formal_static_table_profile"),
                            (aux, "ezkl_backend_aux_table_profile")):
        supplied = profile.get("canonical_digest")
        profile["canonical_digest"] = ""
        canonical = json.dumps(profile, separators=(",", ":"), ensure_ascii=False).encode()
        if supplied != formal_digest(domain, "canonical_json", canonical):
            failures.append(f"{run_dir}: {domain} canonical digest mismatch")
        profile["canonical_digest"] = supplied
    configured = 1 << quant["limb_bits"]
    # Table domains are derived from the registered scalar family ledger.
    # `range_bits` fixes limb rows, but must not silently cap score/ExpMap
    # domains for a valid model.
    lrelu_min = main.get("lrelu_min")
    lrelu_max = main.get("lrelu_max")
    elu_min = main.get("elu_min")
    elu_max = main.get("elu_max")
    exp_max = main.get("exp_max")
    if not all(isinstance(value, int) for value in
               (lrelu_min, lrelu_max, elu_min, elu_max, exp_max)):
        failures.append(f"{run_dir}: non-integer main table domain")
        return failures
    if lrelu_min >= lrelu_max or elu_min >= elu_max or exp_max < 0:
        failures.append(f"{run_dir}: empty main table domain")
    expected_metadata = {
        "generator_id": "main_gatzk_quantized_tables_v1",
        "profile_version": "main_gatzk_table_profile_v1",
        "rounding_mode": "nearest_ties_away",
        "clipping_rule": "exp_nonnegative_clamp_zero",
        "canonical_order": "kind_then_signed_input_ascending",
        "lrelu_scale": 256, "elu_input_scale": 16, "elu_output_scale": 16,
        "exp_input_scale": 256, "exp_output_scale": 16,
    }
    if any(main.get(key) != value for key, value in expected_metadata.items()):
        failures.append(f"{run_dir}: main table metadata mismatch")
    failures.extend(validate_rows(
        run_dir / main["rows_path"], "formal_static_table_rows", main["rows_digest"],
        {2: (lrelu_min, lrelu_max), 3: (elu_min, elu_max), 4: (0, exp_max)},
    ))
    ledger = json.loads((run_dir / "integer_bound_ledger.json").read_text(encoding="utf-8"))
    for family in ledger.get("families", []):
        label = family.get("label", "<unknown>")
        score_bound = abs(family.get("scalar_score", 0))
        if lrelu_min > -score_bound or lrelu_max < score_bound:
            failures.append(f"{run_dir}: LReLU table does not cover scalar family {label}")
        if family.get("scalar_delta", 0) > exp_max:
            failures.append(f"{run_dir}: ExpMap table does not cover scalar family {label}")
        if label.startswith("hidden."):
            recovery_bound = family.get("recovery", {}).get("q", 0)
            if elu_min > -recovery_bound or elu_max < recovery_bound:
                failures.append(f"{run_dir}: ELU table does not cover scalar family {label}")
    required_aux = max(
        max((family.get("delta", 0) + 1 for family in ledger["families"]),
            default=0),
        json.loads((run_dir / "csr_segment_layout.json").read_text(encoding="utf-8"))["max_in_degree"] - 1,
    )
    if (aux.get("generator_id") != "ezkl_backend_aux_tables_v1"
            or aux.get("profile_version") != "ezkl_backend_aux_table_profile_v1"
            or aux.get("canonical_order") != "kind_then_signed_input_ascending"
            or aux.get("limb_max") != configured - 1 or aux.get("max_relu_min", 1) > -required_aux
            or aux.get("max_relu_max", -1) < required_aux):
        failures.append(f"{run_dir}: auxiliary table domain is undersized")
    failures.extend(validate_rows(
        run_dir / aux["rows_path"], "ezkl_backend_aux_table_rows", aux["rows_digest"],
        {1: (0, configured - 1), 5: (aux["max_relu_min"], aux["max_relu_max"])},
    ))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="validate present verified cells without requiring the full matrix")
    args = parser.parse_args()
    failures: list[str] = []
    verified_count = 0
    dataset_cfg = json.loads((ROOT / "configs" / "full_datasets.json").read_text(encoding="utf-8"))
    expected = {(entry["run_id"], device) for entry in dataset_cfg["datasets"] for device in ("cpu", "cuda")}
    present: set[tuple[str, str]] = set()
    matrix_records: list[dict[str, object]] = []
    for manifest_path in sorted((ROOT / "results" / "runs").glob("*/*/run_manifest.json")):
        if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_nlink != 1:
            failures.append(f"{manifest_path}: unsafe manifest file")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            matrix_records.append(manifest)
        cell = (manifest.get("dataset"), manifest.get("device"))
        if cell in expected:
            present.add(cell)
        if manifest.get("verify_passed") is not True or manifest.get("status") != "verified":
            if cell in expected and not args.allow_incomplete:
                failures.append(f"{manifest_path}: required matrix cell is not verified")
            continue
        verified_count += 1
        run_dir = manifest_path.parent
        try:
            expected_cell = canonical_cell_dir(ROOT, str(manifest.get("device")), str(manifest.get("dataset")))
            if run_dir.resolve() != expected_cell.resolve():
                failures.append(f"{manifest_path}: canonical result cell path mismatch")
            validate_cell_inventory(run_dir)
        except (OSError, ValueError) as exc:
            failures.append(f"{manifest_path}: exact artifact inventory failure: {exc}")
            continue
        missing = [name for name in REQUIRED if not (run_dir / name).is_file()]
        if missing:
            failures.append(f"{manifest_path}: missing {', '.join(missing)}")
            continue
        device = manifest.get("device")
        if device not in ("cpu", "cuda"):
            failures.append(f"{manifest_path}: invalid formal backend")
            continue
        expected_source_id = formal_source_id(ROOT)
        expected_prover_binary = ROOT / (
            "target-gpu/release/ezkl_formal" if device == "cuda" else "target/release/ezkl_formal"
        )
        verifier_binary = ROOT / "target/release/ezkl_formal"
        adapter_binary = ROOT / "build_gat_adapter" / "gat_ezkl_export"
        try:
            prover_identity_path = ROOT / "build_identity" / f"{device}.json"
            verifier_identity_path = ROOT / "build_identity" / "cpu.json"
            prover_identity = validate_build_identity(
                ROOT, prover_identity_path, backend=str(device),
                binary=expected_prover_binary, adapter=adapter_binary,
            )
            verifier_identity = validate_build_identity(
                ROOT, verifier_identity_path, backend="cpu",
                binary=verifier_binary, adapter=adapter_binary,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{manifest_path}: build identity validation failed: {exc}")
            continue
        if manifest.get("formal_semantic_source_id") != expected_source_id:
            failures.append(f"{manifest_path}: stale formal semantic source identity")
        if manifest.get("formal_source_manifest") != prover_identity["formal_source_manifest"]:
            failures.append(f"{manifest_path}: recursive formal source inventory mismatch")
        if manifest.get("prover_build_identity_sha256") != identity_digest(prover_identity_path):
            failures.append(f"{manifest_path}: prover build identity digest mismatch")
        if manifest.get("verifier_build_identity_sha256") != identity_digest(verifier_identity_path):
            failures.append(f"{manifest_path}: verifier build identity digest mismatch")
        for prefix, identity in (("prover", prover_identity), ("verifier", verifier_identity)):
            if manifest.get(f"{prefix}_formal_semantic_source_id") != identity["formal_semantic_source_id"]:
                failures.append(f"{manifest_path}: {prefix} semantic source binding mismatch")
            if manifest.get(f"{prefix}_source_tree_digest") != identity["source_tree_digest"]:
                failures.append(f"{manifest_path}: {prefix} source tree binding mismatch")
            if manifest.get(f"{prefix}_dependency_lock_digest") != identity["dependency_lock_digest"]:
                failures.append(f"{manifest_path}: {prefix} dependency lock binding mismatch")
        verify = json.loads((run_dir / "verify.json").read_text(encoding="utf-8"))
        if verify.get("verify_passed") is not True:
            failures.append(f"{manifest_path}: independent verifier marker absent")
        for payload, label in ((manifest, "manifest"), (verify, "verify")):
            for field in ("semantic_source_id", "formal_source_manifest_id", "source_tree_digest"):
                if payload.get(field) != expected_source_id:
                    failures.append(f"{manifest_path}: {label} {field} is not current")
            if payload.get("vendor_tree_digest") != prover_identity["vendor_tree_digest"]:
                failures.append(f"{manifest_path}: {label} vendor tree binding mismatch")
        prove_metadata = manifest.get("prove_metadata")
        verify_metadata = manifest.get("verify_metadata")
        for metadata, label in ((prove_metadata, "prove_metadata"), (verify_metadata, "verify_metadata")):
            if not isinstance(metadata, dict):
                failures.append(f"{manifest_path}: missing {label}")
                continue
            for field in ("semantic_source_id", "formal_source_manifest_id", "source_tree_digest"):
                if metadata.get(field) != expected_source_id:
                    failures.append(f"{manifest_path}: {label} {field} is not current")
        if verify.get("proof_instances_trusted") is not False:
            failures.append(f"{manifest_path}: verifier trusted proof-carried instances")
        circuit = json.loads((run_dir / "circuit_manifest.json").read_text(encoding="utf-8"))
        security = json.loads((run_dir / "security_profile.json").read_text(encoding="utf-8"))
        spec = json.loads((run_dir / "graph_attention_spec.json").read_text(encoding="utf-8"))
        settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
        compiled = json.loads((run_dir / "network.compiled").read_text(encoding="utf-8"))
        statement = json.loads((run_dir / "public_statement.json").read_text(encoding="utf-8"))
        registration = json.loads((run_dir / "model_registration.json").read_text(encoding="utf-8"))
        aux = json.loads((run_dir / "backend_aux_table_spec.json").read_text(encoding="utf-8"))
        main_table = json.loads((run_dir / "formal_table_spec.json").read_text(encoding="utf-8"))
        for payload in (manifest, spec, settings, compiled, statement, verify):
            if payload.get("route_id") != FORMAL_ROUTE_ID:
                failures.append(f"{manifest_path}: formal route identity mismatch")
        main_id = main_table.get("canonical_digest")
        if any(payload.get("static_tables_id") != main_id for payload in
               (manifest, circuit, spec, settings, statement, verify)):
            failures.append(f"{manifest_path}: main static table identity mismatch")
        aux_id = aux.get("canonical_digest")
        if any(payload.get("backend_aux_tables_id") != aux_id for payload in
               (manifest, circuit, spec, settings, statement, verify)):
            failures.append(f"{manifest_path}: backend auxiliary table identity mismatch")
        settings_for_id = dict(settings)
        settings_id = settings_for_id.get("settings_id")
        settings_for_id["settings_id"] = ""
        if settings_id != formal_digest(
                "formal_gat_settings", "canonical_json", canonical_json(settings_for_id)):
            failures.append(f"{manifest_path}: domain-separated settings identity mismatch")
        compiled_for_id = dict(compiled)
        compiled_id = compiled_for_id.get("compiled_circuit_id")
        compiled_for_id["compiled_circuit_id"] = ""
        if compiled_id != formal_digest(
                "compiled_gat_circuit", "canonical_json", canonical_json(compiled_for_id)):
            failures.append(f"{manifest_path}: domain-separated compiled circuit identity mismatch")
        if any(payload.get("compiled_circuit_id") != compiled_id for payload in
               (circuit, statement, verify)):
            failures.append(f"{manifest_path}: compiled circuit cross-binding mismatch")
        expected_bounds_id = formal_digest(
            "registered_model_bounds", "canonical_json",
            canonical_json(registration.get("parameter_bounds", {})),
        )
        if (registration.get("model_bounds_digest") != expected_bounds_id
                or compiled.get("ledger", {}).get("model_bounds_digest") != expected_bounds_id):
            failures.append(f"{manifest_path}: registered model bounds identity mismatch")
        expected_vk_id = formal_digest(
            "halo2_verification_key", "serialized_bytes", (run_dir / "vk.key").read_bytes()
        )
        expected_pk_id = formal_digest(
            "halo2_proving_key", "serialized_bytes", (run_dir / "pk.key").read_bytes()
        )
        if (circuit.get("vk_id") != expected_vk_id or statement.get("vk_id") != expected_vk_id
                or verify.get("vk_id") != expected_vk_id or circuit.get("pk_id") != expected_pk_id):
            failures.append(f"{manifest_path}: domain-separated proving/verifying key identity mismatch")
        failures.extend(validate_table_artifacts(run_dir))
        security_bytes = (run_dir / "security_profile.json").read_text(encoding="utf-8").rstrip("\r\n").encode()
        security_id = formal_digest("ezkl_security_profile", "canonical_json", security_bytes)
        if (security.get("source_srs_max_k") != FORMAL_SOURCE_SRS_MAX_K
                or security.get("semantic_version") != FORMAL_SEMANTIC_VERSION
                or security.get("operation_registry_id") != operation_registry_id()
                or security.get("semantic_source_id") != expected_source_id):
            failures.append(f"{manifest_path}: exact SRS/semantic security profile binding absent")
        for payload in (settings, compiled):
            if (payload.get("semantic_version") != FORMAL_SEMANTIC_VERSION
                    or payload.get("operation_registry_id") != operation_registry_id()
                    or payload.get("semantic_source_id") != expected_source_id):
                failures.append(f"{manifest_path}: semantic source/operation binding mismatch")
        if any(payload.get(field) != security_id for payload, field in (
            (manifest, "ezkl_security_profile_id"), (circuit, "security_profile_id"),
            (spec, "security_profile_id"), (settings, "security_profile_id"),
            (statement, "security_profile_id"))):
            failures.append(f"{manifest_path}: resolved security profile identity mismatch")
        if circuit.get("downsized_srs_k") != settings.get("logrows"):
            failures.append(f"{manifest_path}: downsized SRS k/settings mismatch")
        if circuit.get("proving_srs_sha256") != "sha256:" + digest(run_dir / "pk.srs"):
            failures.append(f"{manifest_path}: proving SRS digest mismatch")
        if circuit.get("verification_srs_sha256") != "sha256:" + digest(run_dir / "vk.srs"):
            failures.append(f"{manifest_path}: verification SRS digest mismatch")
        if (Path(str(circuit.get("proving_srs_path", ""))).resolve() != (run_dir / "pk.srs").resolve()
                or Path(str(circuit.get("verification_srs_path", ""))).resolve()
                != (run_dir / "vk.srs").resolve()):
            failures.append(f"{manifest_path}: downsized SRS path binding mismatch")
        if not str(circuit.get("source_srs_sha256", "")).startswith("sha256:"):
            failures.append(f"{manifest_path}: source SRS provenance absent")
        if (circuit.get("source_srs_sha256") != security.get("source_srs_sha256") or
                circuit.get("source_srs_k") != FORMAL_SOURCE_SRS_MAX_K or
                circuit.get("source_srs_k") != security.get("source_srs_max_k")):
            failures.append(f"{manifest_path}: SRS differs from resolved security profile")
        if (not isinstance(circuit.get("source_srs_k"), int)
                or settings.get("logrows", 0) <= 0
                or settings.get("logrows", 0) > FORMAL_SOURCE_SRS_MAX_K
                or circuit.get("source_srs_k", 0) < settings.get("logrows", 0)):
            failures.append(f"{manifest_path}: source SRS k does not cover compiled logrows")
        source_srs = Path(str(circuit.get("source_srs_path", "")))
        if not source_srs.is_file() or circuit.get("source_srs_sha256") != "sha256:" + digest(source_srs):
            failures.append(f"{manifest_path}: source SRS bytes/provenance mismatch")
        for field, path in {
            "proof_sha256": run_dir / "proof.json",
            "public_statement_sha256": run_dir / "public_statement.json",
            "compiled_circuit_sha256": run_dir / "network.compiled",
            "vk_sha256": run_dir / "vk.key",
        }.items():
            if manifest.get(field) != digest(path):
                failures.append(f"{manifest_path}: {field} mismatch")
        expected_device = (
            "halo2_icicle_cuda_device_0" if manifest.get("device") == "cuda" else "halo2_cpu"
        )
        expected_prover_backend = "cuda_icicle" if manifest.get("device") == "cuda" else "cpu"
        if manifest.get("runtime_device_marker") != expected_device:
            failures.append(f"{manifest_path}: runtime device marker mismatch")
        if manifest.get("prover_backend") != expected_prover_backend:
            failures.append(f"{manifest_path}: prover backend identity mismatch")
        if manifest.get("verifier_backend") != "cpu_independent":
            failures.append(f"{manifest_path}: verifier backend is not independent CPU")
        if verify.get("prover_backend") != expected_prover_backend:
            failures.append(f"{manifest_path}: verifier prover-backend binding mismatch")
        if verify.get("verifier_backend") != "cpu_independent":
            failures.append(f"{manifest_path}: verifier backend is not independent CPU")
        if verify.get("verification_device_runtime_marker") != "halo2_cpu":
            failures.append(f"{manifest_path}: verification device marker is not CPU")
        if "gpu_runtime_telemetry" in verify or "gpu_verification_backend" in verify:
            failures.append(f"{manifest_path}: verifier carries forbidden GPU telemetry/claim")
        if manifest.get("device") == "cuda":
            prove_telemetry = manifest.get("gpu_prove_telemetry")
            telemetry_valid = (
                isinstance(prove_telemetry, dict)
                and prove_telemetry.get("schema") == "ezkl_cuda_kernel_telemetry_v1"
                and prove_telemetry.get("scope") == "formal_prove"
                and isinstance(prove_telemetry.get("kernel_count"), int)
                and prove_telemetry.get("kernel_count", 0) > 0
                and prove_telemetry.get("dropped_records") == 0
                and prove_telemetry.get("active_device_id") == 0
                and prove_telemetry.get("scope_nonce")
                and prove_telemetry.get("scope_end_ns", 0)
                >= prove_telemetry.get("scope_start_ns", 1)
                and isinstance(prove_telemetry.get("kernel_names"), list)
                and prove_telemetry.get("kernel_names")
                and all(
                    isinstance(kernel, dict)
                    and kernel.get("name")
                    and kernel.get("category")
                    and isinstance(kernel.get("count"), int)
                    and kernel.get("count", 0) > 0
                    and kernel.get("device_id") == 0
                    for kernel in prove_telemetry.get("kernel_names", [])
                )
                and sum(
                    kernel["count"] for kernel in prove_telemetry["kernel_names"]
                )
                == prove_telemetry["kernel_count"]
            )
            if manifest.get("gpu_backend_execution") != "cuda_icicle_prove_only" or not telemetry_valid:
                failures.append(f"{manifest_path}: CUDA backend execution marker mismatch")
        elif manifest.get("gpu_prove_telemetry") != "not_applicable":
            failures.append(f"{manifest_path}: CPU result carries GPU prove telemetry")
    failures.extend(validate_matrix_records(matrix_records, allow_incomplete=args.allow_incomplete))
    if not args.allow_incomplete:
        for dataset, device in sorted(expected - present):
            failures.append(f"missing required matrix cell: {dataset}/{device}")
    if failures:
        print("\n".join(failures))
        return 2
    print(json.dumps({"verified_formal_runs_validated": verified_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
