#!/usr/bin/env python3
"""One isolated native full-GAT prove/independent-verify fixture.

The caller supplies a real build snapshot made by formal_provenance.py.  This
script uses the canonical Cora architecture and scale chain with a separately
identified synthetic two-node graph; it never enters the five-dataset runner
or result collector.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import formal_provenance  # noqa: E402
import prepare_form_resources as resources  # noqa: E402


FIXTURE_PREFIX = "native_fixture_TEST_"
PROFILE = {
    "L": "3",
    "hidden_profile": "[8x8,8x8]",
    "d_in_profile": "[1433,64]",
    "K_out": "2",
    "num_classes": "7",
    "quant_cfg_id": "gat_l3_scale_chain",
    "input_preprocess_id": "row_normalize",
}
PHASE_TIMEOUT_SECONDS = {
    "resource_setup": 300,
    "resource_validation": 60,
    "asset_manifest": 60,
    "native_prove": 300,
    "native_verify": 180,
}


class FixtureError(RuntimeError):
    pass


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_kv(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if key in fields:
            raise FixtureError(f"duplicate {key!r} in {path}:{number}")
        fields[key] = value.strip()
    return fields


def run_phase(name: str, argv: list[str], run_dir: Path, env: dict[str, str],
              timeout: int | None = None) -> None:
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    seconds = timeout if timeout is not None else PHASE_TIMEOUT_SECONDS[name]
    print(f"fixture_phase={name} status=START timeout_seconds={seconds} stdout={rel(stdout_path)} stderr={rel(stderr_path)}",
          flush=True)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            return_code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise FixtureError(
                f"{name} timed out after {seconds}s; artifacts/logs retained under {rel(run_dir)}"
            ) from exc
    if return_code != 0:
        raise FixtureError(
            f"{name} exited {return_code}; stdout={rel(stdout_path)} stderr={rel(stderr_path)}"
        )
    print(f"fixture_phase={name} status=PASS", flush=True)


def write_fixture_config(config_path: Path, fixture_root: Path, run_dir: Path) -> None:
    template_path = ROOT / "configs" / "cora_secure.cfg"
    values = resources.parse_config(template_path)
    suffix = run_dir.name.removeprefix(FIXTURE_PREFIX)
    ids = {
        "model_arch_id": f"gat_family_L3_hid8x8-8x8_kout2_c7_TEST_FIXTURE_{suffix}",
        "model_param_id": f"cora_planetoid_seed11_TEST_FIXTURE_{suffix}",
        "static_table_id": f"cora_native_tables_TEST_FIXTURE_{suffix}",
    }
    fixture_paths = {
        "data_root": fixture_root / "data",
        "cache_root": fixture_root / "cache",
        "checkpoint_bundle": fixture_root / "checkpoint",
        "export_dir": run_dir,
        "static_table_descriptor": fixture_root / "resources" / "cora_static_tables.json",
        "sorted_lookup_descriptor": fixture_root / "resources" / "cora_static_tables_sorted_lookup.json",
        "integer_bound_ledger": fixture_root / "resources" / "cora_integer_bound_ledger.json",
        "srs_path": fixture_root / "resources" / "srs",
        "model_registry_path": fixture_root / "resources" / "model_registry.json",
        "model_blinder_sidecar_path": fixture_root / "resources" / "model_blinders.json",
    }
    updates = {key: value for key, value in PROFILE.items()}
    updates.update(ids)
    updates.update({key: rel(path) for key, path in fixture_paths.items()})
    updates["srs_id"] = f"cora_native_TEST_FIXTURE_{suffix}"
    updates["degree_bound_id"] = f"zkdl_degree_v2_cora_TEST_FIXTURE_{suffix}"
    lines = [
        "# TEST_FIXTURE only: synthetic N=2/F=1433 graph using the canonical Cora GAT profile.",
        "# This config is not admitted to the formal five-dataset runner or result collector.",
    ]
    seen: set[str] = set()
    for raw in template_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.split("#", 1)[0].strip()
        if "=" not in stripped:
            lines.append(raw)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            lines.append(f"{key} = {updates[key]}")
            seen.add(key)
        else:
            lines.append(raw)
    missing = set(updates) - seen
    if missing:
        raise FixtureError(f"Cora profile config lacks required keys: {sorted(missing)}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def initialize_fixture(run_dir: Path) -> tuple[Path, Path]:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise FixtureError(f"real formal_provenance build snapshot is missing: {run_dir}")
    if run_dir.parent != ROOT / "runs" or not run_dir.name.startswith(FIXTURE_PREFIX):
        raise FixtureError(f"run root must be runs/{FIXTURE_PREFIX}<id> and outside formal collection")
    if re.fullmatch(r"native_fixture_TEST_[A-Za-z0-9_]+", run_dir.name) is None:
        raise FixtureError(f"unsafe test run-root name: {run_dir.name}")
    if re.fullmatch(r"formal_[0-9a-f]{32}", run_dir.name):
        raise FixtureError("synthetic fixture must not use a formal session directory")
    manifest_path = run_dir / "build_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FixtureError(f"build manifest from formal_provenance is missing: {manifest_path}")
    manifest = formal_provenance.validate_manifest(ROOT, manifest_path)
    binary_paths = {item["name"]: ROOT / item["path"] for item in manifest["binaries"]}
    if set(binary_paths) != {"gat_runner", "gat_resource_setup"}:
        raise FixtureError("verified build manifest does not contain both native binaries")

    fixture_root = ROOT / "tests" / "fixtures" / "full_gat_native_cora" / run_dir.name
    config_path = ROOT / "configs" / f"{run_dir.name}.cfg"
    if fixture_root.exists() or config_path.exists():
        raise FixtureError(
            f"fixture output already exists; preserving it: fixture={fixture_root} config={config_path}"
        )
    cache_dir = fixture_root / "cache" / "cora"
    checkpoint_dir = fixture_root / "checkpoint"
    (fixture_root / "data").mkdir(parents=True)
    (fixture_root / "resources").mkdir()
    cache_dir.mkdir(parents=True)
    write_fixture_config(config_path, fixture_root, run_dir)

    source_checkpoint = ROOT / "models" / "checkpoints" / "cora_gat"
    if any(path.is_symlink() for path in source_checkpoint.rglob("*")):
        raise FixtureError(f"refusing to copy symlinked source checkpoint: {source_checkpoint}")
    shutil.copytree(source_checkpoint, checkpoint_dir)
    config_values = resources.parse_config(config_path)
    checkpoint_manifest_path = checkpoint_dir / "manifest.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    for key in ("model_arch_id", "model_param_id", "static_table_id"):
        checkpoint_manifest[key] = config_values[key]
    checkpoint_manifest["checkpoint_prefix"] = rel(checkpoint_dir / "best_model.pt")
    checkpoint_manifest_path.write_text(
        json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    (cache_dir / "meta.cfg").write_text(
        "name=cora\nnum_nodes=2\nnum_features=1433\ngraph_count=1\n"
        "input_preprocess_id=row_normalize\nnum_classes=7\n"
        "task_type=transductive_node_classification\nreport_unit=node\n"
        f"fixture_id=TEST_FIXTURE_{run_dir.name}\nsource_graph=synthetic\n",
        encoding="utf-8",
    )
    (cache_dir / "features.txt").write_text(
        "2 1433\n0:1.0 17:0.5\n1:1.0 18:0.5\n", encoding="utf-8"
    )
    (cache_dir / "edges.txt").write_text("0 0\n0 1\n1 0\n1 1\n", encoding="utf-8")
    (cache_dir / "fixture_identity.json").write_text(json.dumps({
        "schema": "zkDL_native_full_gat_TEST_FIXTURE_v1",
        "fixture_id": run_dir.name,
        "dataset_selector": "cora",
        "input_graph": "synthetic directed graph",
        "nodes": 2,
        "features": 1433,
        "directed_edges": [[0, 0], [0, 1], [1, 0], [1, 1]],
        "layers": 3,
        "hidden_profile": "[8x8,8x8]",
        "output_heads": 2,
        "classes": 7,
        "formal_admission": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"fixture_identity=TEST_FIXTURE id={run_dir.name} config={rel(config_path)} input_root={rel(fixture_root)}",
          flush=True)
    print(f"native_runner={rel(binary_paths['gat_runner'])} resource_helper={rel(binary_paths['gat_resource_setup'])}",
          flush=True)
    return config_path, fixture_root


def prepare_resources(config_path: Path, run_dir: Path) -> None:
    """Call only the existing authenticated resource producers used by formal setup."""
    values = resources.parse_config(config_path)
    if values.get("dataset") != "cora" or "TEST_FIXTURE" not in values.get("model_param_id", ""):
        raise FixtureError("resource preparation requires the explicitly marked synthetic Cora fixture")
    quant_path = ROOT / "configs" / "quantization_scale_chain.cfg"
    quant = resources.validated_scale_chain(quant_path, values["quant_cfg_id"])
    quant_digest = resources.sha256_file(quant_path)
    nodes, edges, degree = resources.graph_degree(ROOT, "cora", values)
    if (nodes, edges) != (2, 4):
        raise FixtureError(f"synthetic graph identity changed: nodes={nodes} edges={edges}")

    table = resources.make_tables(values, quant_digest)
    table_path = ROOT / values["static_table_descriptor"]
    table_digest = resources.write_json(table_path, table)
    lookup_path = resources.sorted_lookup_path(table_path)
    configured_lookup = ROOT / values["sorted_lookup_descriptor"]
    if lookup_path != configured_lookup:
        raise FixtureError("configured sorted-lookup path does not match the official producer path")
    lookup = resources.make_sorted_lookup_descriptor(
        values, table, table_digest, nodes, resources.parse_profile(values["hidden_profile"]), edges
    )
    resources.write_json(lookup_path, lookup)

    ledger = resources.make_ledger(ROOT, values, table, table_digest, quant_digest, degree=degree)
    ledger_path = ROOT / values["integer_bound_ledger"]
    ledger_digest = resources.write_json(ledger_path, ledger)
    helper = formal_provenance.native_binary(ROOT, "gat_resource_setup")
    subprocess.run(
        [str(helper), "verify-ledger", "--path", str(ledger_path), "--baseline-root", str(ROOT)],
        cwd=ROOT, check=True,
    )

    checkpoint_manifest_path = ROOT / values["checkpoint_bundle"] / "manifest.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    checkpoint_manifest["degree_bound_id"] = ledger["degree_bound_id"]
    checkpoint_manifest_path.write_text(
        json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    resources.update_config(config_path, {"degree_bound_id": ledger["degree_bound_id"]})
    values = resources.parse_config(config_path)

    tensors = resources.load_tensors(ROOT / values["checkpoint_bundle"] / "tensors.txt")
    model_domains = [
        math.prod(resources.next_pow2(int(axis)) for axis in
                  (shape[1:] if len(shape) == 3 and shape[0] == 1 else shape))
        for _, shape, _ in tensors
    ]
    metadata = resources.graph_metadata(ROOT, "cora", values)
    family_domains = resources.relation_family_domains(
        values, nodes, edges, table, model_domains,
        graph_count=metadata.get("graph_count", 1),
        ledger_families=ledger["families"], quantization=quant,
    )
    srs_digest, generated_digest = resources.write_srs(ROOT, values, family_domains)
    resources.update_config(config_path, {"srs_manifest_digest": srs_digest})
    values = resources.parse_config(config_path)

    generators, h_point, u_point, loaded_generator_digest, _basis_digest = resources.load_srs_basis(
        ROOT, values, max(model_domains, default=1)
    )
    if loaded_generator_digest != generated_digest:
        raise FixtureError("native SRS producer and authenticated loader returned different generator digests")
    tensor_ids: set[str] = set()
    for name, _, _ in tensors:
        mapped = name.replace("/seq/kernel", "/model/seq_kernel").replace(
            "/attn_src/kernel", "/model/attn_src"
        ).replace("/attn_dst/kernel", "/model/attn_dst").replace("/output_bias", "/model/output_bias")
        if "/model/" in mapped:
            tensor_ids.add(mapped)
    if not tensor_ids:
        raise FixtureError("copied Cora checkpoint contains no canonical full-GAT tensors")
    blinders = {tensor_id: resources.fresh_scalar() for tensor_id in tensor_ids}
    registry_entries = resources.model_entries(
        ROOT, values, h_point, u_point, blinders, generators, srs_digest
    )
    registry: dict[str, Any] = {
        "schema": "zkDL_gat_model_registry_v3",
        "dataset": "cora",
        "model_arch_id": values["model_arch_id"],
        "model_param_id": values["model_param_id"],
        "quant_cfg_id": values["quant_cfg_id"],
        "static_table_id": values["static_table_id"],
        "srs_id": values["srs_id"],
        "srs_manifest_digest": srs_digest,
        "generator_digest": loaded_generator_digest,
        "config_digest": resources.config_identity(config_path),
        "scale_chain_digest": quant_digest,
        "static_table_digest": table_digest,
        "integer_bound_ledger_digest": ledger_digest,
        "model_manifest_digest": resources.sha256_file(checkpoint_manifest_path),
        "model_tensors": registry_entries,
        "source_root": "baseline-local",
    }
    registry_identity_digest = resources.sha256_bytes(resources.compact(registry))
    registry["registry_identity_digest"] = registry_identity_digest
    registry["verifier_registry_digest"] = registry_identity_digest
    registry_path = ROOT / values["model_registry_path"]
    registry_digest = resources.write_json(registry_path, registry)
    sidecar = {
        "schema": "zkDL_gat_model_blinder_sidecar_v1",
        "registry_digest": registry_digest,
        "entries": [
            {"tensor_id": item["tensor_id"],
             "blinder": blinders[item["tensor_id"]].to_bytes(32, "little").hex()}
            for item in registry_entries
        ],
        "source_root": "baseline-local",
    }
    sidecar_path = ROOT / values["model_blinder_sidecar_path"]
    sidecar_digest = resources.write_json_atomic(sidecar_path, sidecar, 0o600)
    resources.update_config(config_path, {
        "degree_bound_id": ledger["degree_bound_id"],
        "srs_manifest_digest": srs_digest,
        "verifier_registry_digest": registry_digest,
        "model_blinder_sidecar_digest": sidecar_digest,
    })
    print(
        f"fixture_resources=PASS nodes={nodes} edges={edges} model_tensors={len(registry_entries)} "
        f"srs_basis_domain={len(generators)} srs_manifest_digest={srs_digest} registry_digest={registry_digest}",
        flush=True,
    )


def remove_prover_private_material(config_path: Path, run_dir: Path) -> None:
    values = resources.parse_config(config_path)
    sidecar = ROOT / values["model_blinder_sidecar_path"]
    if sidecar.is_symlink():
        raise FixtureError("refusing to remove a symlinked fixture blinder sidecar")
    sidecar.unlink(missing_ok=True)
    private_manifest = run_dir / "private_asset_manifest.json"
    if private_manifest.is_symlink():
        raise FixtureError("refusing to remove a symlinked private asset manifest")
    private_manifest.unlink(missing_ok=True)
    scratch = run_dir / ".private_native"
    if scratch.is_symlink():
        raise FixtureError("refusing to remove symlinked native witness scratch")
    if scratch.exists():
        shutil.rmtree(scratch)


def audit_closure(config_path: Path, fixture_root: Path, run_dir: Path) -> dict[str, Any]:
    record = parse_kv(run_dir / "run.record")
    envelope = parse_kv(run_dir / "proof.meta")
    statement = parse_kv(run_dir / "statement.txt")
    receipt = parse_kv(run_dir / "verification.result")
    proof_path = run_dir / "proof.bin"
    if proof_path.is_symlink() or not proof_path.is_file():
        raise FixtureError("native proof artifact is missing or symlinked")
    proof_bytes = proof_path.stat().st_size
    if proof_bytes == 0:
        raise FixtureError("native proof artifact is empty")
    required_identity = {
        "status": "PROVED",
        "proof_scope": "full_gat_forward",
        "commitment_time_semantics": "native_tensor_commitments",
        "verifier_reads_witness": "0",
        "silent_fallback": "0",
        "output_bound": "1",
    }
    for key, expected in required_identity.items():
        if envelope.get(key) != expected:
            raise FixtureError(f"proof envelope {key}={envelope.get(key)!r}, expected {expected!r}")
    if record.get("verification_status") != "VERIFIED":
        raise FixtureError(f"run record verification status is not VERIFIED: {record.get('verification_status')}")
    if record.get("proof_scope") != "full_gat_forward" or record.get("verifier_reads_witness") != "0":
        raise FixtureError("run record does not attest full-GAT scope and witnessless verification")
    if record.get("silent_fallback") != "0" or record.get("output_bound") != "1" or record.get("gpu_kernel_executed") != "0":
        raise FixtureError("run record contains fallback/output/backend provenance mismatch")
    if receipt.get("status") != "VERIFIED":
        raise FixtureError(f"independent verification receipt is not VERIFIED: {receipt}")
    if receipt.get("statement_id") != statement.get("statement_id") or receipt.get("proof_id") != record.get("proof_id"):
        raise FixtureError("independent verification receipt does not bind the proof and statement")
    if record.get("statement_id") != statement.get("statement_id") or record.get("proof_id") != envelope.get("proof_id"):
        raise FixtureError("proof, statement, envelope, and run-record identities disagree")
    if record.get("proof_size_bytes") != str(proof_bytes) or envelope.get("proof_size_bytes") != str(proof_bytes):
        raise FixtureError("recorded proof size does not match proof.bin")
    if statement.get("nodes") != "2" or statement.get("graphs") != "1" or statement.get("classes") != "7":
        raise FixtureError("proof statement is not bound to the two-node synthetic full-GAT fixture")
    if statement.get("proof_scope") != "full_gat_forward":
        raise FixtureError("public statement is not a complete full-GAT statement")
    try:
        sorted_lookup_query_count = int(statement.get("sorted_lookup_query_count", "0"))
    except ValueError as exc:
        raise FixtureError("public statement has an invalid sorted-lookup query count") from exc
    if sorted_lookup_query_count <= 0:
        raise FixtureError("public statement does not bind any sorted-lookup queries")
    if "TEST_FIXTURE" not in resources.parse_config(config_path)["model_param_id"]:
        raise FixtureError("configuration lost the explicit TEST_FIXTURE identity")
    fixture_base = ROOT / "tests" / "fixtures" / "full_gat_native_cora"
    try:
        fixture_root.resolve().relative_to(fixture_base.resolve())
    except ValueError as exc:
        raise FixtureError("fixture assets escaped the test-only fixture tree") from exc

    metrics: dict[str, float] = {}
    for label, field in (("commitment", "commitment_time_seconds"),
                         ("prove", "prove_time_seconds"), ("verify", "verify_time_seconds")):
        raw = record.get(field)
        try:
            value = float(raw or "nan")
        except ValueError as exc:
            raise FixtureError(f"invalid {field}: {raw!r}") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise FixtureError(f"{field} is not a positive finite measured duration: {raw!r}")
        try:
            envelope_value = float(envelope.get(field, "nan"))
        except ValueError as exc:
            raise FixtureError(f"invalid proof-envelope {field}: {envelope.get(field)!r}") from exc
        tolerance = 2e-12 * max(1.0, abs(value), abs(envelope_value))
        if not math.isfinite(envelope_value) or abs(value - envelope_value) > tolerance:
            raise FixtureError(
                f"proof-envelope and run-record {field} disagree: {envelope_value!r} vs {value!r}"
            )
        metrics[label] = value
    commitment_bytes = int(record.get("commitment_size_bytes", "0"))
    if commitment_bytes <= 0:
        raise FixtureError("recorded native commitment size is not positive")
    return {
        "fixture_id": run_dir.name,
        "dataset_selector": "cora",
        "input_graph": "synthetic N=2 F=1433, four directed edges",
        "proof_scope": "full_gat_forward",
        "commitment_time_seconds": metrics["commitment"],
        "prove_time_seconds": metrics["prove"],
        "verify_time_seconds": metrics["verify"],
        "commitment_size_bytes": commitment_bytes,
        "proof_size_bytes": proof_bytes,
        "sorted_lookup_query_count": sorted_lookup_query_count,
        "proof_id": record["proof_id"],
        "statement_id": record["statement_id"],
        "independent_verification": "VERIFIED",
        "run_dir": rel(run_dir),
        "fixture_root": rel(fixture_root),
    }


def execute_fixture(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    config_path, fixture_root = initialize_fixture(run_dir)
    build_manifest = run_dir / "build_manifest.json"
    manifest = formal_provenance.validate_manifest(ROOT, build_manifest)
    binary_paths = {item["name"]: ROOT / item["path"] for item in manifest["binaries"]}
    env = os.environ.copy()
    for variable in ("ZKDL_FORMAL_SESSION_ROOT", "ZKDL_FORMAL_SESSION_ROOT_DEVICE", "ZKDL_FORMAL_SESSION_ROOT_INODE"):
        env.pop(variable, None)
    env["ZKDL_BUILD_MANIFEST"] = str(build_manifest)

    run_phase("resource_setup", [
        sys.executable, str(Path(__file__).resolve()), "--prepare-resources",
        "--config", str(config_path), "--run-dir", str(run_dir),
    ], run_dir, env)
    run_phase("resource_validation", [
        sys.executable, str(SCRIPTS / "validate_resources.py"), "--config", str(config_path),
    ], run_dir, env)
    run_phase("asset_manifest", [
        sys.executable, str(SCRIPTS / "asset_manifest.py"), "--config", str(config_path),
        "--output", str(run_dir / "asset_manifest.json"),
    ], run_dir, env)

    runner = binary_paths["gat_runner"]
    run_phase("native_prove", [
        str(runner), "prove", "--config", str(config_path), "--backend", "cpu", "--run-dir", str(run_dir),
    ], run_dir, env)
    remove_prover_private_material(config_path, run_dir)
    verifier_env = dict(env)
    verifier_env.pop("ZKDL_BUILD_MANIFEST", None)
    run_phase("native_verify", [
        str(runner), "verify", "--config", str(config_path), "--backend", "cpu", "--run-dir", str(run_dir),
    ], run_dir, verifier_env)
    result = audit_closure(config_path, fixture_root, run_dir)
    print("native_full_gat_TEST_FIXTURE_closure=PASS " + json.dumps(result, sort_keys=True), flush=True)
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="existing runs/native_fixture_TEST_* root with formal_provenance build snapshot")
    parser.add_argument("--prepare-resources", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
        if args.prepare_resources:
            if args.config is None:
                raise FixtureError("internal resource stage requires --config")
            prepare_resources(args.config.resolve(), run_dir.resolve())
        else:
            execute_fixture(run_dir)
        return 0
    except (FixtureError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"native-full-gat-fixture: FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
