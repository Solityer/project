#!/usr/bin/env python3
"""Fail-closed structural validation of collected formal ezkl GAT results."""

from __future__ import annotations

import hashlib
import json
import argparse
import struct
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "graph_attention_spec.json", "csr_segment_layout.bin", "node_features.i64.bin",
    "model_parameters.i64.txt", "formal_quant_config.json", "formal_table_spec.json",
    "formal_main_tables.i64.bin", "backend_aux_table_spec.json", "backend_aux_tables.i64.bin",
    "security_profile.json",
    "integer_bound_ledger.json", "model_registration.json", "settings.json",
    "network.compiled", "vk.key", "pk.key", "witness.json", "public_statement.json",
    "proof.json", "verify.json", "circuit_manifest.json", "pk.srs", "vk.srs",
)


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
    large = spec.get("batching_rule") == "whole_graph_single" and (
        spec["edge_count"] >= (1 << 18) or spec["node_count"] * spec["feature_dim"] >= (1 << 24)
    )
    range_size = max(configured, (1 << 23) - 2) if large else configured
    lrelu_min = -(3 << 18) if large else -configured
    lrelu_max = ((1 << 23) - 2) - (3 << 18) - 1 if large else configured
    elu_bound = max(configured, 1 << 20) if large else configured
    expected_main = (lrelu_min, lrelu_max, -elu_bound, elu_bound, range_size - 1)
    actual_main = (main.get("lrelu_min"), main.get("lrelu_max"), main.get("elu_min"),
                   main.get("elu_max"), main.get("exp_max"))
    expected_metadata = {
        "generator_id": "main_gatzk_quantized_tables_v1",
        "profile_version": "main_gatzk_table_profile_v1",
        "rounding_mode": "nearest_ties_away",
        "clipping_rule": "exp_nonnegative_clamp_zero",
        "canonical_order": "kind_then_signed_input_ascending",
        "lrelu_scale": 256, "elu_input_scale": 16, "elu_output_scale": 16,
        "exp_input_scale": 256, "exp_output_scale": 16,
    }
    if actual_main != expected_main or any(main.get(key) != value for key, value in expected_metadata.items()):
        failures.append(f"{run_dir}: main table domain selection mismatch")
    failures.extend(validate_rows(
        run_dir / main["rows_path"], "formal_static_table_rows", main["rows_digest"],
        {2: (lrelu_min, lrelu_max), 3: (-elu_bound, elu_bound), 4: (0, range_size - 1)},
    ))
    required_aux = max(
        max((family.get("delta", 0) + 1 for family in
             json.loads((run_dir / "integer_bound_ledger.json").read_text(encoding="utf-8"))["families"]),
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
    for manifest_path in sorted((ROOT / "runs").glob("*/*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cell = (manifest.get("dataset"), manifest.get("device"))
        if cell in expected:
            present.add(cell)
        if manifest.get("verify_passed") is not True or manifest.get("status") != "verified":
            if cell in expected and not args.allow_incomplete:
                failures.append(f"{manifest_path}: required matrix cell is not verified")
            continue
        verified_count += 1
        run_dir = manifest_path.parent
        missing = [name for name in REQUIRED if not (run_dir / name).is_file()]
        if missing:
            failures.append(f"{manifest_path}: missing {', '.join(missing)}")
            continue
        verify = json.loads((run_dir / "verify.json").read_text(encoding="utf-8"))
        if verify.get("verify_passed") is not True:
            failures.append(f"{manifest_path}: independent verifier marker absent")
        if verify.get("proof_instances_trusted") is not False:
            failures.append(f"{manifest_path}: verifier trusted proof-carried instances")
        circuit = json.loads((run_dir / "circuit_manifest.json").read_text(encoding="utf-8"))
        security = json.loads((run_dir / "security_profile.json").read_text(encoding="utf-8"))
        spec = json.loads((run_dir / "graph_attention_spec.json").read_text(encoding="utf-8"))
        settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
        statement = json.loads((run_dir / "public_statement.json").read_text(encoding="utf-8"))
        aux = json.loads((run_dir / "backend_aux_table_spec.json").read_text(encoding="utf-8"))
        main_table = json.loads((run_dir / "formal_table_spec.json").read_text(encoding="utf-8"))
        main_id = main_table.get("canonical_digest")
        if any(payload.get("static_tables_id") != main_id for payload in
               (manifest, circuit, spec, settings, statement, verify)):
            failures.append(f"{manifest_path}: main static table identity mismatch")
        aux_id = aux.get("canonical_digest")
        if any(payload.get("backend_aux_tables_id") != aux_id for payload in
               (manifest, circuit, spec, settings, statement, verify)):
            failures.append(f"{manifest_path}: backend auxiliary table identity mismatch")
        failures.extend(validate_table_artifacts(run_dir))
        security_bytes = (run_dir / "security_profile.json").read_text(encoding="utf-8").rstrip("\r\n").encode()
        security_id = formal_digest("ezkl_security_profile", "canonical_json", security_bytes)
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
                circuit.get("source_srs_k") != security.get("source_srs_max_k")):
            failures.append(f"{manifest_path}: SRS differs from resolved security profile")
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
        if manifest.get("runtime_device_marker") != expected_device:
            failures.append(f"{manifest_path}: runtime device marker mismatch")
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
