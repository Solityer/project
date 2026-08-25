"""Private test support for the smallest complete zkGPT GAT relation.

The fixture is deliberately kept under ``tests/`` and is materialised only
inside a ``TemporaryDirectory`` rooted at ``runs_tmp``.  It does not stage a
dataset, checkpoint, public result, or production registry.  The prover uses
the real ``build_proof`` path; the only test-only override is the capability
gate and the static-table descriptor loader for the authenticated range-bits
fixture.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import platform
import shutil
import sys
import tempfile
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from zkgpt_gat.config import AppConfig
from zkgpt_gat.data import GraphDataset, LocalGraph
from zkgpt_gat.forward import forward_full
from zkgpt_gat.model import AttentionHead, HiddenLayer, Model, QUANT_SCALE_PROFILES
from zkgpt_gat.proof import build_proof
from zkgpt_gat.util import digest_json, write_json


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS_TMP = ROOT / "runs_tmp"
CACHE_ENV = "ZKGPT_SYNTHETIC_E2E_CACHE_DIR"
KEEP_FAILED_ENV = "ZKGPT_SYNTHETIC_E2E_KEEP_FAILED"
CACHE_MANIFEST_NAME = "synthetic_e2e_manifest.json"
CACHE_SCHEMA = "zkGPT_test_synthetic_e2e_cache_v1"
FAILED_DIAGNOSTIC_SCHEMA = "zkGPT_test_synthetic_e2e_failed_diagnostic_v1"
PUBLIC_INPUT_AUDIT_NAME = "public_input_access_audit.json"
PUBLIC_CONFIG_SCHEMA = "zkGPT_test_synthetic_config_v1"
PUBLIC_GRAPH_SCHEMA = "zkGPT_test_public_graph_v1"
PUBLIC_STATEMENT_SCHEMA = "zkGPT_external_public_statement_v1"
SYNTHETIC_PROOF_FILES = (
    "metadata.json",
    "commitments.json",
    "transcript.json",
    "lookup_proof.json",
    "fixed_lookup_tables.json",
    "matrix_sumcheck_proof.json",
    "gkr_proof.json",
    "combine_proof.json",
    "hyrax_mle_openings.json",
    "privacy_profile.json",
    "proof_payload.json",
    "verification.json",
)
SYNTHETIC_RANGE_BITS = 4
SYNTHETIC_DATASET = "synthetic-e2e"
MODEL_ARCH_ID = "zkgpt-test-gat-n1-h1-c1-v1"
MODEL_PARAM_ID = "zkgpt-test-model-zero-small-v1"
STATIC_TABLE_ID = "tables:lrelu+elu+exp+range"
QUANT_CFG_ID = "exact_integer_gat_v1"


def _read_json(path: pathlib.Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _source_files() -> list[pathlib.Path]:
    """Return the source/test/resource files that define this fixture.

    The cache is deliberately source-locked.  It must never become a way to
    replay a proof produced by a different adapter or native verifier.  Cache
    and bytecode directories are excluded because they are generated test
    state rather than source inputs.
    """
    files: list[pathlib.Path] = []
    for directory in (ROOT / "src", ROOT / "tests", ROOT / "resources"):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def synthetic_source_hash() -> str:
    """Hash the exact baseline-local inputs used by synthetic proving/replay."""
    digest = hashlib.sha256()
    for path in _source_files():
        relative = path.relative_to(ROOT).as_posix()
        payload = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _configured_cache_dir() -> pathlib.Path | None:
    raw = os.environ.get(CACHE_ENV, "").strip()
    if not raw:
        return None
    RUNS_TMP.mkdir(parents=True, exist_ok=True)
    candidate = pathlib.Path(raw).expanduser()
    # Relative values are intentionally rooted below runs_tmp.  Absolute
    # values are accepted only after resolving symlinks and checking the same
    # boundary, so a cache cannot point at production results or another
    # baseline.
    path = (RUNS_TMP / candidate if not candidate.is_absolute() else candidate).resolve()
    try:
        path.relative_to(RUNS_TMP.resolve())
    except ValueError as exc:
        raise ValueError(f"{CACHE_ENV} must point below {RUNS_TMP}") from exc
    if path == RUNS_TMP.resolve():
        raise ValueError(f"{CACHE_ENV} must name a cache directory below {RUNS_TMP}")
    return path


def _file_inventory(root: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    """Return a digest inventory, excluding mutable test reports/manifest."""
    inventory: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {CACHE_MANIFEST_NAME, PUBLIC_INPUT_AUDIT_NAME}:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory[relative] = {"sha256": digest, "size_bytes": path.stat().st_size}
    return inventory


def _python_identity() -> Dict[str, str | None]:
    implementation = platform.python_implementation()
    version = platform.python_version()
    cache_tag = getattr(sys.implementation, "cache_tag", None)
    return {
        "implementation": implementation,
        "version": version,
        "cache_tag": str(cache_tag) if cache_tag is not None else None,
    }


def _proof_schema_inventory(root: pathlib.Path) -> Dict[str, Any]:
    """Return the exact serialized test-bundle schema/file inventory.

    The inventory is deliberately explicit.  It is not enough for a cache to
    contain some JSON files that happen to verify: a partial bundle, a stale
    adapter output, or an extra untracked proof message must force a rebuild.
    """
    schemas: Dict[str, Any] = {
        "public_root": {
            "public_config.json": PUBLIC_CONFIG_SCHEMA,
            "public_graph.json": PUBLIC_GRAPH_SCHEMA,
            "public_model.json": "public_model_architecture_v1",
            "public_y.json": "public_output_v1",
            "static_descriptor.json": "zkGPT_static_tables_descriptor_v1",
        },
        "proof_files": list(SYNTHETIC_PROOF_FILES),
        "statement_schema": PUBLIC_STATEMENT_SCHEMA,
    }
    # The model and output descriptors intentionally have no trust-bearing
    # schema field.  Record their exact expected shape marker in the cache
    # inventory instead of accepting an arbitrary user-supplied schema.
    model_path = root / "public_model.json"
    y_path = root / "public_y.json"
    if model_path.is_file():
        model = _read_json(model_path)
        if not isinstance(model.get("manifest"), dict):
            raise ValueError("synthetic model descriptor has no manifest")
        schemas["public_root"]["public_model.json"] = "public_model_architecture_v1"
    if y_path.is_file():
        y_payload = _read_json(y_path)
        if not isinstance(y_payload.get("shape"), list) or not isinstance(y_payload.get("values"), list):
            raise ValueError("synthetic public Y descriptor is malformed")
        schemas["public_root"]["public_y.json"] = "public_output_v1"
    schemas["proof_files"] = sorted(schemas["proof_files"])
    return schemas


def _expected_bundle_files() -> set[str]:
    return {
        "public_config.json",
        "public_graph.json",
        "public_model.json",
        "public_y.json",
        "static_descriptor.json",
        "synthetic.cfg",
        f"results/model_registry/{MODEL_PARAM_ID}/registry.json",
        *(f"proof/{name}" for name in SYNTHETIC_PROOF_FILES),
    }


def _validate_cache_manifest(root: pathlib.Path, manifest: Mapping[str, Any]) -> None:
    """Fail closed unless *manifest* exactly describes this source/runtime."""
    expected_source = synthetic_source_hash()
    expected = {
        "schema": CACHE_SCHEMA,
        "dataset": SYNTHETIC_DATASET,
        "backend": "cpu",
        "source_tree_hash": expected_source,
        "python": _python_identity(),
        "production_results": False,
        "status": "TEST_FIXTURE_VALID",
        "proof_schema_inventory": _proof_schema_inventory(root),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"synthetic cache manifest mismatch: {key}")
    if manifest.get("cache_root") != str(root.relative_to(RUNS_TMP.resolve())):
        raise ValueError("synthetic cache manifest root mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != _expected_bundle_files():
        raise ValueError("synthetic cache file inventory is incomplete or has extras")
    if set(_file_inventory(root)) != set(files):
        raise ValueError("synthetic cache files differ from manifest inventory")
    internal = manifest.get("internal_verify")
    if not isinstance(internal, dict):
        raise ValueError("synthetic cache internal verification evidence is missing")
    if internal.get("status") != "VERIFY_OK":
        raise ValueError("synthetic cache internal verification status is not VERIFY_OK")
    if internal.get("statement_schema") != PUBLIC_STATEMENT_SCHEMA:
        raise ValueError("synthetic cache public statement schema evidence is missing")
    statement_digest = internal.get("statement_digest")
    if not isinstance(statement_digest, str) or len(statement_digest) != 64:
        raise ValueError("synthetic cache public statement digest evidence is missing")
    checks = internal.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise ValueError("synthetic cache internal verification checks are incomplete")
    if internal.get("verifier_reads_witness") is not False or internal.get("witness_paths_read") != [] or internal.get("witness_fields_read") != []:
        raise ValueError("synthetic cache witness boundary evidence is invalid")
    actual_inventory = _file_inventory(root)
    if files != actual_inventory:
        raise ValueError("synthetic cache digest inventory mismatch")


def _cache_manifest(root: pathlib.Path, report: Mapping[str, Any]) -> Dict[str, Any]:
    audit = report.get("public_input_access_audit")
    checks = report.get("checks")
    if not isinstance(audit, dict) or not isinstance(checks, dict):
        raise AssertionError("synthetic verifier did not expose internal checks/audit")
    if report.get("status") != "VERIFY_OK" or not all(value is True for value in checks.values()):
        raise AssertionError("synthetic verifier did not return a complete VERIFY_OK report")
    if report.get("statement_schema") != PUBLIC_STATEMENT_SCHEMA:
        raise AssertionError("synthetic verifier did not return the expected public statement schema")
    statement_digest = report.get("statement_digest")
    if not isinstance(statement_digest, str) or len(statement_digest) != 64:
        raise AssertionError("synthetic verifier did not return a public statement digest")
    if audit.get("verifier_reads_witness") is not False:
        raise AssertionError("synthetic verifier witness boundary is not closed")
    if audit.get("witness_paths_read") != [] or audit.get("witness_fields_read") != []:
        raise AssertionError("synthetic verifier accessed private witness material")
    return {
        "schema": CACHE_SCHEMA,
        "dataset": SYNTHETIC_DATASET,
        "backend": "cpu",
        "source_tree_hash": synthetic_source_hash(),
        "python": _python_identity(),
        "cache_root": str(root.relative_to(RUNS_TMP.resolve())),
        "production_results": False,
        "status": "TEST_FIXTURE_VALID",
        "proof_schema_inventory": _proof_schema_inventory(root),
        "internal_verify": {
            "status": report["status"],
            "verifier": report.get("verifier", ""),
            "checks": {str(key): bool(value) for key, value in sorted(checks.items())},
            "statement_schema": report["statement_schema"],
            "statement_digest": statement_digest,
            "verifier_reads_witness": False,
            "witness_paths_read": [],
            "witness_fields_read": [],
        },
        "files": _file_inventory(root),
    }


def _reset_cache_dir(root: pathlib.Path) -> None:
    resolved = root.expanduser().resolve()
    runs_tmp = RUNS_TMP.expanduser().resolve()
    try:
        resolved.relative_to(runs_tmp)
    except ValueError as exc:
        raise ValueError(f"refusing to reset cache outside {runs_tmp}: {resolved}") from exc
    if resolved == runs_tmp:
        raise ValueError("refusing to reset the runs_tmp directory itself")
    if resolved.exists():
        if resolved.is_symlink() or not resolved.is_dir():
            resolved.unlink()
        else:
            shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _bundle_paths(root: pathlib.Path, *, cached: bool) -> Dict[str, Any]:
    """Describe a cached bundle without fabricating a proof object/trace."""
    registry = root / "results" / "model_registry" / MODEL_PARAM_ID / "registry.json"
    return {
        "root": root,
        "artifact": root / "proof",
        "config": root / "public_config.json",
        "graph": root / "public_graph.json",
        "model": root / "public_model.json",
        "public_y": root / "public_y.json",
        "static_descriptor": root / "static_descriptor.json",
        "registry": registry,
        "cached": cached,
        "cache_manifest": root / CACHE_MANIFEST_NAME,
        "proof": None,
        "trace": None,
    }


class _PersistentSyntheticDirectory:
    """TemporaryDirectory-compatible handle for an opt-in persistent cache."""

    def __init__(self, path: pathlib.Path, *, source: str) -> None:
        self.name = str(path)
        self.cached = True
        self.cache_source = source

    def cleanup(self) -> None:
        # The user explicitly opted into persistence; never remove the cache
        # during test teardown.  Invalid caches are replaced only after a
        # complete fresh bundle has verified successfully.
        return None

    def __enter__(self) -> "_PersistentSyntheticDirectory":
        return self

    def __exit__(self, *_args: object) -> None:
        self.cleanup()


class SyntheticBundleVerificationError(ValueError):
    """A failed adapter check with structured test-only diagnostics."""

    def __init__(self, message: str, diagnostic: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostic = dict(diagnostic)


def make_static_descriptor(root: pathlib.Path) -> pathlib.Path:
    """Create an authenticated test-only copy with a four-bit range table."""
    source = _read_json(ROOT / "resources" / "static_tables" / "exact_integer_gat_v1.json")
    descriptor = copy.deepcopy(source)
    descriptor["content"]["range_table"]["range_bits"] = SYNTHETIC_RANGE_BITS
    descriptor["content_digest"] = digest_json(descriptor["content"])
    without_digest = dict(descriptor)
    without_digest.pop("descriptor_digest", None)
    descriptor["descriptor_digest"] = digest_json(without_digest)
    path = root / "static_descriptor.json"
    write_json(path, descriptor)
    return path


def synthetic_graph() -> Tuple[GraphDataset, LocalGraph]:
    """Return N=1, E=1, one graph, and an explicit self-loop."""
    features = np.asarray([[1]], dtype=np.int64)
    labels = np.asarray([0], dtype=np.int64)
    dataset = GraphDataset(
        name=SYNTHETIC_DATASET,
        num_nodes=1,
        num_features=1,
        num_classes=1,
        features=features,
        labels=labels,
        edges_src=np.asarray([0], dtype=np.int64),
        edges_dst=np.asarray([0], dtype=np.int64),
        edge_graph=np.asarray([0], dtype=np.int64),
        node_ptr=np.asarray([0, 1], dtype=np.int64),
        graph_count=1,
        task_type="transductive_node_classification",
        report_unit="node",
    )
    graph = LocalGraph(
        name=f"{SYNTHETIC_DATASET}_normalized",
        num_nodes=1,
        num_features=1,
        num_classes=1,
        features=features,
        labels=labels,
        absolute_ids=np.asarray([0], dtype=np.int64),
        src=np.asarray([0], dtype=np.int64),
        dst=np.asarray([0], dtype=np.int64),
        edge_graph=np.asarray([0], dtype=np.int64),
        node_ptr=np.asarray([0, 1], dtype=np.int64),
        edge_ptr=np.asarray([0, 1], dtype=np.int64),
        graph_count=1,
        n_total=1,
        task_type="transductive_node_classification",
        report_unit="node",
        batching_rule="whole_graph_single",
        self_loop_rule="per_node",
        edge_sort_rule="edge_gid_then_dst_stable",
        chunking_rule="none",
    )
    return dataset, graph


def synthetic_model() -> Model:
    """Build a one-hidden-head/one-output-head integer model.

    The hidden projection is negative so LeakyReLU and ELU each receive a
    negative representative.  The output bias keeps the final public logit
    negative after the output head's scale alignment.  All dimensions and
    values remain tiny while every protocol family is still instantiated.
    """
    manifest = {
        "model_arch_id": MODEL_ARCH_ID,
        "model_param_id": MODEL_PARAM_ID,
        "quant_cfg_id": QUANT_CFG_ID,
        "rounding_mode": "nearest_ties_away",
        "static_table_id": STATIC_TABLE_ID,
        "hidden_layers": [{"layer_index": 0, "input_dim": 1, "head_count": 1, "head_dim": 1}],
        "hidden_head_specs": [{"layer_index": 0, "local_head_index": 0}],
        "output_head_specs": [{"head_index": 0}],
        "C": 1,
        "K_out": 1,
        "L": 2,
        "d_in_profile": [1],
        "output_average_rule": "per_head_bias_then_arithmetic_mean",
    }
    hidden_head = AttentionHead(
        seq_kernel=np.asarray([[-1]], dtype=np.int64),
        attn_dst_kernel=np.asarray([1], dtype=np.int64),
        attn_src_kernel=np.asarray([1], dtype=np.int64),
        output_bias=np.asarray([0], dtype=np.int64),
    )
    output_head = AttentionHead(
        seq_kernel=np.asarray([[1]], dtype=np.int64),
        attn_dst_kernel=np.asarray([1], dtype=np.int64),
        attn_src_kernel=np.asarray([1], dtype=np.int64),
        output_bias=np.asarray([-1], dtype=np.int64),
    )
    return Model(
        manifest=manifest,
        hidden_layers=[HiddenLayer(0, 1, 1, 1, [hidden_head])],
        output_heads=[output_head],
        class_count=1,
        k_out=1,
        model_arch_id=MODEL_ARCH_ID,
        model_param_id=MODEL_PARAM_ID,
        quant_cfg_id=QUANT_CFG_ID,
        static_table_id=STATIC_TABLE_ID,
        output_average_rule="per_head_bias_then_arithmetic_mean",
        quant_scales=dict(QUANT_SCALE_PROFILES[QUANT_CFG_ID]),
    )


def synthetic_config(root: pathlib.Path, static_descriptor: pathlib.Path) -> AppConfig:
    entries = {
        "dataset": SYNTHETIC_DATASET,
        "static_table_path": static_descriptor.name,
        "static_table_descriptor_path": static_descriptor.name,
        "static_table_descriptor_digest": _read_json(static_descriptor)["descriptor_digest"],
        "range_bits": str(SYNTHETIC_RANGE_BITS),
        "model_arch_id": MODEL_ARCH_ID,
        "model_param_id": MODEL_PARAM_ID,
        "quant_cfg_id": QUANT_CFG_ID,
        "rounding_mode": "nearest_ties_away",
        "static_table_id": STATIC_TABLE_ID,
        "static_tables_id": STATIC_TABLE_ID,
        "task_type": "transductive_node_classification",
        "report_unit": "node",
        "batching_rule": "whole_graph_single",
        "self_loop_rule": "per_node",
        "edge_sort_rule": "edge_gid_then_dst_stable",
        "chunking_rule": "none",
        "L": "2",
        "hidden_profile": "1x1",
        "d_in_profile": "1",
        "K_out": "1",
        "num_classes": "1",
    }
    path = root / "synthetic.cfg"
    path.write_text(
        "\n".join(f"{key} = {value}" for key, value in sorted(entries.items())) + "\n",
        encoding="utf-8",
    )
    return AppConfig(entries=entries, path=path, project_root=root)


def _public_graph_payload(graph: LocalGraph) -> Dict[str, Any]:
    return {
        "schema": "zkGPT_test_public_graph_v1",
        "name": graph.name,
        "features": {"shape": list(graph.features.shape), "values": [int(v) for v in graph.features.ravel()]},
        "labels": {"shape": list(graph.labels.shape), "values": [int(v) for v in graph.labels.ravel()]},
        "absolute_ids": [int(v) for v in graph.absolute_ids],
        "src": [int(v) for v in graph.src],
        "dst": [int(v) for v in graph.dst],
        "edge_graph": [int(v) for v in graph.edge_graph],
        "node_ptr": [int(v) for v in graph.node_ptr],
        "edge_ptr": [int(v) for v in graph.edge_ptr],
        "graph_count": graph.graph_count,
        "n_total": graph.n_total,
        "num_nodes": graph.num_nodes,
        "num_features": graph.num_features,
        "num_classes": graph.num_classes,
        "task_type": graph.task_type,
        "report_unit": graph.report_unit,
        "batching_rule": graph.batching_rule,
        "self_loop_rule": graph.self_loop_rule,
        "edge_sort_rule": graph.edge_sort_rule,
        "chunking_rule": graph.chunking_rule,
    }


def _public_model_payload(model: Model) -> Dict[str, Any]:
    """Serialize public architecture/shapes and bounds, never tensor values."""
    return {
        "manifest": model.manifest,
        "hidden_layers": [
            {
                "layer_index": layer.layer_index,
                "input_dim": layer.input_dim,
                "head_count": layer.head_count,
                "head_dim": layer.head_dim,
            }
            for layer in model.hidden_layers
        ],
        "output_heads": [{"head_index": index, "class_count": model.class_count} for index in range(model.k_out)],
        "quant_scales": model.quant_scales,
        "model_arch_id": model.model_arch_id,
        "model_param_id": model.model_param_id,
        "quant_cfg_id": model.quant_cfg_id,
        "static_table_id": model.static_table_id,
        "class_count": model.class_count,
        "k_out": model.k_out,
    }


def _write_proof_artifacts(root: pathlib.Path, proof: Any) -> pathlib.Path:
    artifact = root / "proof"
    artifact.mkdir(parents=True, exist_ok=True)
    write_json(artifact / "metadata.json", proof.metadata)
    write_json(artifact / "commitments.json", proof.commitments)
    write_json(artifact / "transcript.json", proof.transcript)
    write_json(artifact / "lookup_proof.json", proof.proof_payload["Pi_Lasso"])
    write_json(artifact / "fixed_lookup_tables.json", proof.proof_payload["FixedLookupTables"])
    write_json(artifact / "matrix_sumcheck_proof.json", proof.proof_payload["Pi_MatMul"])
    write_json(artifact / "gkr_proof.json", proof.proof_payload["Pi_GKR"])
    write_json(artifact / "combine_proof.json", proof.proof_payload["Pi_combine"])
    write_json(artifact / "hyrax_mle_openings.json", proof.proof_payload["Open_in"])
    write_json(artifact / "privacy_profile.json", proof.proof_payload["PrivacyProfile"])
    write_json(artifact / "proof_payload.json", proof.proof_payload)
    write_json(artifact / "verification.json", proof.verification)
    return artifact


def _create_synthetic_bundle_uncached(root: pathlib.Path) -> Dict[str, Any]:
    """Run the real CPU prover and persist only serialized public/test data."""
    static_descriptor = make_static_descriptor(root)
    config = synthetic_config(root, static_descriptor)
    dataset, graph = synthetic_graph()
    model = synthetic_model()
    trace = forward_full(graph, model)

    import zkgpt_gat.formal_gate as formal_gate
    import zkgpt_gat.proof as proof_module

    original_gate = formal_gate.require_formal_experiment
    original_proof_gate = proof_module.require_formal_experiment
    original_descriptor_loader = proof_module.validate_static_table_descriptor
    descriptor = _read_json(static_descriptor)
    formal_gate.require_formal_experiment = lambda: None
    proof_module.require_formal_experiment = lambda: None
    proof_module.validate_static_table_descriptor = lambda _root: copy.deepcopy(descriptor)
    try:
        proof = build_proof(
            config,
            dataset,
            graph,
            model,
            trace,
            "cpu",
            {"backend": "cpu", "cuda_kernel_executed": False, "gpu_runtime_present": False},
            {"artifact_dir": root / "proof", "zk_root": root},
        )
    finally:
        formal_gate.require_formal_experiment = original_gate
        proof_module.require_formal_experiment = original_proof_gate
        proof_module.validate_static_table_descriptor = original_descriptor_loader

    artifact = _write_proof_artifacts(root, proof)
    registry_path = root / "results" / "model_registry" / MODEL_PARAM_ID / "registry.json"
    if not registry_path.is_file():
        raise AssertionError(f"missing test-only public com_W registry: {registry_path}")
    public_y = root / "public_y.json"
    write_json(public_y, {"shape": list(trace.y.shape), "values": [int(v) for v in np.ravel(trace.y)]})
    graph_path = root / "public_graph.json"
    write_json(graph_path, _public_graph_payload(graph))
    model_path = root / "public_model.json"
    write_json(model_path, _public_model_payload(model))
    config_payload = {
        "schema": "zkGPT_test_synthetic_config_v1",
        "dataset": SYNTHETIC_DATASET,
        "dataset_display_name": dataset.name,
        "model": _public_model_payload(model),
        "range_bits": SYNTHETIC_RANGE_BITS,
        "static_descriptor_path": static_descriptor.name,
        "static_descriptor": descriptor,
        "static_table_id": STATIC_TABLE_ID,
        "quant_cfg_id": QUANT_CFG_ID,
        "rounding_mode": model.rounding_mode,
        "task_type": graph.task_type,
        "report_unit": graph.report_unit,
        "batching_rule": graph.batching_rule,
        "self_loop_rule": graph.self_loop_rule,
        "edge_sort_rule": graph.edge_sort_rule,
        "chunking_rule": graph.chunking_rule,
        "registry_relative_path": str(registry_path.relative_to(root)),
    }
    config_fixture = root / "public_config.json"
    write_json(config_fixture, config_payload)
    return {
        "root": root,
        "artifact": artifact,
        "config": config_fixture,
        "graph": graph_path,
        "model": model_path,
        "public_y": public_y,
        "static_descriptor": static_descriptor,
        "registry": registry_path,
        "proof": proof,
        "trace": trace,
    }


def _adapter_diagnostic(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep the adapter's failure evidence intact for test-only diagnostics."""
    return {
        "status": report.get("status"),
        "error": report.get("error", report.get("reason")),
        "checks": report.get("checks", {}),
        "audit": report.get("public_input_access_audit", {}),
    }


def _verify_synthetic_bundle(root: pathlib.Path) -> Dict[str, Any]:
    """Run the test adapter's real verifier and require every check to pass."""
    try:
        from verify_synthetic_proof import verify_serialized_bundle  # type: ignore
    except ImportError:  # pragma: no cover - package-style test invocation
        from tests.verify_synthetic_proof import verify_serialized_bundle  # type: ignore
    report = verify_serialized_bundle(root)
    diagnostic = _adapter_diagnostic(report)
    audit = report.get("public_input_access_audit")
    checks = report.get("checks")
    if (
        report.get("status") != "VERIFY_OK"
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or not isinstance(audit, dict)
        or audit.get("verifier_reads_witness") is not False
        or audit.get("witness_paths_read") != []
        or audit.get("witness_fields_read") != []
    ):
        detail = json.dumps({"adapter": diagnostic}, sort_keys=True, default=str)
        raise SyntheticBundleVerificationError(
            "synthetic bundle verifier did not return complete VERIFY_OK evidence: "
            + detail,
            diagnostic,
        )
    return report


def _bundle_with_verification(root: pathlib.Path, *, cached: bool, source: str, report: Mapping[str, Any]) -> Dict[str, Any]:
    bundle = _bundle_paths(root, cached=cached)
    bundle["cache_source"] = source
    bundle["internal_verify"] = {
        "status": report.get("status"),
        "verifier": report.get("verifier", ""),
        "checks": dict(report.get("checks", {})),
        "statement_schema": report.get("statement_schema", ""),
        "statement_digest": report.get("statement_digest", ""),
        "verifier_reads_witness": False,
        "witness_paths_read": [],
        "witness_fields_read": [],
    }
    return bundle


def _load_cached_bundle(root: pathlib.Path) -> Dict[str, Any] | None:
    """Return a cache bundle only after exact manifest/inventory/verifier checks."""
    manifest_path = root / CACHE_MANIFEST_NAME
    if not root.is_dir() or not manifest_path.is_file():
        return None
    try:
        manifest = _read_json(manifest_path)
        # A retained failed staging directory is diagnostic evidence only.  It
        # must never be interpreted as a usable cache, even if a future
        # schema validator happens to accept some of its files.
        if (
            manifest.get("schema") == FAILED_DIAGNOSTIC_SCHEMA
            or manifest.get("status") == "FAILED_DIAGNOSTIC"
        ):
            return None
        _validate_cache_manifest(root, manifest)
        report = _verify_synthetic_bundle(root)
        expected = _cache_manifest(root, report)
        if manifest != expected:
            raise ValueError("synthetic cache manifest evidence is not source-matched")
        write_json(root / PUBLIC_INPUT_AUDIT_NAME, report["public_input_access_audit"])
        return _bundle_with_verification(root, cached=True, source="cache", report=report)
    except (OSError, TypeError, ValueError, KeyError, AssertionError, ImportError):
        # A stale, partial, or tampered cache is never used.  The caller builds
        # a new bundle into an isolated staging directory before replacing it.
        return None


def _keep_failed_diagnostics() -> bool:
    return os.environ.get(KEEP_FAILED_ENV, "") == "1"


def _failed_diagnostic_payload(exc: Exception) -> Dict[str, Any]:
    adapter = getattr(exc, "diagnostic", None)
    if not isinstance(adapter, Mapping):
        adapter = {
            "status": "BUILD_FAILED",
            "error": str(exc),
            "checks": {},
            "audit": {},
        }
    # Reports are JSON by contract, but normalize an unexpected test double
    # or exception payload so retaining diagnostics cannot mask the original
    # build/verifier failure with a serialization error.
    normalized = json.loads(json.dumps(dict(adapter), sort_keys=True, default=str))
    return {
        "schema": FAILED_DIAGNOSTIC_SCHEMA,
        "status": "FAILED_DIAGNOSTIC",
        "test_only": True,
        "production_results": False,
        "dataset": SYNTHETIC_DATASET,
        "backend": "cpu",
        "adapter": normalized,
        "error": str(exc),
    }


def _retain_failed_staging(
    staging: pathlib.Path,
    root: pathlib.Path,
    source_hash: str,
    exc: Exception,
) -> pathlib.Path:
    """Move failed test staging to a deterministic, non-production sibling."""
    parent = root.parent.resolve()
    if staging.parent.resolve() != parent:
        raise ValueError("synthetic failed diagnostic staging parent mismatch")
    failed_root = parent / f".failed-{source_hash}"
    _reset_cache_dir(failed_root)
    # _reset_cache_dir creates the validated target after removing any stale
    # diagnostic with the same source hash; os.replace then gives us an
    # atomic move of the complete failed staging tree.
    failed_root.rmdir()
    os.replace(staging, failed_root)
    diagnostic = _failed_diagnostic_payload(exc)
    adapter = diagnostic.get("adapter")
    if isinstance(adapter, Mapping):
        audit = adapter.get("audit")
        if isinstance(audit, Mapping):
            write_json(failed_root / PUBLIC_INPUT_AUDIT_NAME, dict(audit))
    diagnostic.update(
        {
            "cache_root": str(failed_root.relative_to(RUNS_TMP.resolve())),
            "source_tree_hash": source_hash,
            "files": _file_inventory(failed_root),
        }
    )
    write_json(failed_root / CACHE_MANIFEST_NAME, diagnostic)
    return failed_root


def _build_cached_bundle(root: pathlib.Path) -> Dict[str, Any]:
    root.parent.mkdir(parents=True, exist_ok=True)
    source_hash = synthetic_source_hash()
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=str(root.parent)))
    try:
        _create_synthetic_bundle_uncached(staging)
        report = _verify_synthetic_bundle(staging)
        # Only a complete real prover output which passed the independent
        # adapter is eligible to become persistent cache state.
        _reset_cache_dir(root)
        root.rmdir()
        os.replace(staging, root)
        audit = dict(report["public_input_access_audit"])
        audit["root"] = str(root)
        write_json(root / PUBLIC_INPUT_AUDIT_NAME, audit)
        write_json(root / CACHE_MANIFEST_NAME, _cache_manifest(root, report))
        return _bundle_with_verification(root, cached=True, source="rebuilt", report=report)
    except Exception as exc:
        if _keep_failed_diagnostics() and staging.exists():
            try:
                _retain_failed_staging(staging, root, source_hash, exc)
            except (OSError, TypeError, ValueError):
                # Diagnostic retention is best effort; preserve the original
                # build/verifier exception and let the default cleanup below
                # remove staging if retention could not be completed.
                pass
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def create_synthetic_bundle(root: pathlib.Path) -> Dict[str, Any]:
    """Create or reuse the test-only bundle under the explicit cache policy."""
    root = pathlib.Path(root).expanduser().resolve()
    cache_root = _configured_cache_dir()
    if cache_root is not None and root == cache_root:
        cached = _load_cached_bundle(root)
        if cached is not None:
            return cached
        return _build_cached_bundle(root)
    bundle = _create_synthetic_bundle_uncached(root)
    bundle["cached"] = False
    bundle["cache_source"] = "temporary"
    bundle["internal_verify"] = None
    return bundle


def temporary_synthetic_root() -> tempfile.TemporaryDirectory[str]:
    cache_root = _configured_cache_dir()
    if cache_root is not None:
        cache_root.parent.mkdir(parents=True, exist_ok=True)
        return _PersistentSyntheticDirectory(cache_root, source="configured")  # type: ignore[return-value]
    RUNS_TMP.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="zkgpt-synthetic-e2e-", dir=RUNS_TMP)
    setattr(temporary, "cached", False)
    setattr(temporary, "cache_source", "temporary")
    return temporary


__all__ = [
    "MODEL_ARCH_ID",
    "MODEL_PARAM_ID",
    "QUANT_CFG_ID",
    "ROOT",
    "STATIC_TABLE_ID",
    "SYNTHETIC_RANGE_BITS",
    "create_synthetic_bundle",
    "temporary_synthetic_root",
]
