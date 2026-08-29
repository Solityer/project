#!/usr/bin/env python3
"""Lightweight contract tests; no proof, verifier, or dataset execution."""
from __future__ import annotations

import sys
import tempfile
import unittest
import os
import hashlib
import json
import ast
import subprocess
from pathlib import Path

BASELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE / "scripts"))

from collect_results import inspect_cell  # noqa: E402
import preflight as preflight_module  # noqa: E402
from runner_contract import (  # noqa: E402
    CELL_FILES,
    DATASETS,
    DEVICES,
    RESOURCE_MANIFEST_SCHEMA,
    artifact_inventory,
    config_path,
    ensure_contained,
    ensure_regular_file,
    load_resource_manifest,
    load_resource_policy,
    parse_config,
    static_table_row_cap_blocker,
    STATIC_TABLE_ROW_CAP,
    validate_preflight_metadata,
    validate_checkpoint_structure,
    validate_checkpoint_tensor_shapes,
    validate_model_structure_config,
    pp_binding_from_manifest,
)
import validate_final_results as final_validator  # noqa: E402
from validate_final_results import check_cell  # noqa: E402


class RunnerContractTests(unittest.TestCase):
    def test_adapter_rejects_each_preflight_metadata_binding_field(self) -> None:
        expected = {
            "L": 3,
            "hidden_profile": [[8, 8], [8, 8]],
            "d_in_profile": [1433, 64],
            "K_out": 2,
            "model_arch_id": "arch",
            "model_param_id": "param",
            "checkpoint_manifest_id": "checkpoint",
            "config_sha256": "config",
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                tampered = dict(expected)
                tampered[key] = "tampered" if isinstance(value, str) else -1
                with self.assertRaisesRegex(ValueError, rf"canonical/preflight metadata mismatch: {key}"):
                    validate_preflight_metadata(tampered, expected)

    def test_native_static_table_row_cap_boundary(self) -> None:
        profile = {
            "lrelu": {"min": 0, "max": STATIC_TABLE_ROW_CAP - 1, "rows": STATIC_TABLE_ROW_CAP},
            "elu": {"min": 0, "max": 0, "rows": 1},
            "exp": {"min": 0, "max": 0, "rows": 1},
        }
        self.assertIsNone(static_table_row_cap_blocker(profile))
        profile["lrelu"]["max"] = STATIC_TABLE_ROW_CAP
        profile["lrelu"]["rows"] = STATIC_TABLE_ROW_CAP + 1
        blocker = static_table_row_cap_blocker(profile)
        self.assertIsNotNone(blocker)
        assert blocker is not None
        self.assertEqual(blocker["code"], "static_table_domain_exceeds_native_cap")
        self.assertEqual(blocker["native_row_cap"], STATIC_TABLE_ROW_CAP)
        self.assertEqual(blocker["offending_tables"][0]["rows"], STATIC_TABLE_ROW_CAP + 1)

    def test_preflight_cap_skips_full_table_generator(self) -> None:
        original = preflight_module.static_table_profile

        def unexpected_generator(*args, **kwargs):
            raise AssertionError("full static table generator must not run over native cap")

        preflight_module.static_table_profile = unexpected_generator
        try:
            descriptor, blocker = preflight_module.build_static_tables(
                12,
                "fixture-static-source",
                STATIC_TABLE_ROW_CAP,
                0,
                0,
                "nearest_ties_away",
            )
        finally:
            preflight_module.static_table_profile = original
        self.assertEqual(descriptor["lrelu"]["rows"], 2 * STATIC_TABLE_ROW_CAP + 1)
        self.assertIsNotNone(blocker)
        assert blocker is not None
        self.assertEqual(blocker["code"], "static_table_domain_exceeds_native_cap")

    def test_resource_preparation_is_baseline_local_only(self) -> None:
        script = BASELINE / "scripts" / "prepare_resources.py"
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(script))
        self.assertNotIn("source-root", source)
        self.assertNotIn("source_root", source)
        self.assertNotIn("copytree", source)
        self.assertNotIn("shutil", source)
        options = {
            argument.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for argument in node.args[:1]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        }
        self.assertNotIn("--source-root", options)
        help_result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=BASELINE,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertNotIn("source-root", help_result.stdout)
        self.assertIn("--init", help_result.stdout)
        self.assertIn("--check", help_result.stdout)

    def test_removed_external_source_option_fails_without_writes(self) -> None:
        script = BASELINE / "scripts" / "prepare_resources.py"
        resource_root = BASELINE / "resources"
        sentinels = (
            resource_root,
            resource_root / "resource_manifest.json",
            resource_root / "staged_resource_manifest.json",
            resource_root / "datasets",
            resource_root / "models",
        )

        def signature(path: Path) -> tuple[bool, bool, int | None, int | None]:
            if not path.exists() and not path.is_symlink():
                return False, False, None, None
            stat_result = path.lstat()
            return True, path.is_symlink(), stat_result.st_mtime_ns, stat_result.st_size

        before = {path: signature(path) for path in sentinels}
        with tempfile.TemporaryDirectory() as external_path:
            result = subprocess.run(
                [sys.executable, str(script), "--source-root", external_path],
                cwd=BASELINE,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unrecognized arguments", result.stderr)
            self.assertEqual(list(Path(external_path).iterdir()), [])
        self.assertEqual(before, {path: signature(path) for path in sentinels})

    def test_manifest_is_baseline_local_and_enumerates_exact_matrix(self) -> None:
        manifest, manifest_id = load_resource_policy()
        self.assertEqual(manifest["schema"], RESOURCE_MANIFEST_SCHEMA)
        self.assertTrue(manifest_id.startswith("sha256:"))
        self.assertEqual(set(manifest["datasets"]), set(DATASETS))
        self.assertEqual(set(DEVICES), {"cpu", "gpu"})
        for dataset in DATASETS:
            config = config_path(dataset).read_text(encoding="utf-8")
            self.assertIn("resources/datasets/", config)
            self.assertIn("resources/models/", config)

    def test_source_policy_has_no_placeholder_pp_identity(self) -> None:
        policy, _ = load_resource_policy()
        self.assertNotIn("pp_id", policy["pp"])
        self.assertNotIn("sha256", policy["pp"])
        self.assertNotIn("size_bytes", policy["pp"])
        staged_pp = dict(policy["pp"])
        staged_pp.update(
            {
                "pp_id": "0" * 64,
                "sha256": "sha256:" + "0" * 64,
                "size_bytes": 1,
            }
        )
        staged = dict(policy)
        staged["pp"] = staged_pp
        with self.assertRaises(ValueError):
            pp_binding_from_manifest(staged)

    def test_current_l3_pair_profile_is_accepted(self) -> None:
        config = parse_config(config_path("cora"))
        structure = validate_model_structure_config(config, feature_dim=1433)
        self.assertEqual(structure["L"], 3)
        self.assertEqual(structure["hidden_profile"], [(8, 8), (8, 8)])
        self.assertEqual(structure["d_in_profile"], [1433, 64])
        self.assertEqual(structure["K_out"], 2)
        manifest = {
            "dataset": None,
            "L": 3,
            "K_out": 2,
            "C": 7,
            "model_arch_id": config["model_arch_id"],
            "model_param_id": config["model_param_id"],
            "hidden_profile": [
                {"head_count": 8, "head_dim": 8},
                {"head_count": 8, "head_dim": 8},
            ],
            "d_in_profile": [1433, 64],
            "hidden_layers": [
                {"layer_index": 0, "input_dim": 1433, "head_count": 8, "head_dim": 8},
                {"layer_index": 1, "input_dim": 64, "head_count": 8, "head_dim": 8},
            ],
            "hidden_head_specs": [
                {
                    "layer_index": layer,
                    "local_head_index": local,
                    "global_head_index": layer * 8 + local,
                    "seq_kernel": f"hidden/layer{layer}/head{local}/seq/kernel",
                    "attn_src_kernel": f"hidden/layer{layer}/head{local}/attn_src/kernel",
                    "attn_dst_kernel": f"hidden/layer{layer}/head{local}/attn_dst/kernel",
                    "attn_src_bias": f"hidden/layer{layer}/head{local}/attn_src/bias",
                    "attn_dst_bias": f"hidden/layer{layer}/head{local}/attn_dst/bias",
                    "output_bias": f"hidden/layer{layer}/head{local}/output_bias",
                }
                for layer in range(2)
                for local in range(8)
            ],
            "output_head_specs": [
                {
                    "head_index": index,
                    "seq_kernel": f"output/head{index}/seq/kernel",
                    "attn_src_kernel": f"output/head{index}/attn_src/kernel",
                    "attn_dst_kernel": f"output/head{index}/attn_dst/kernel",
                    "attn_src_bias": f"output/head{index}/attn_src/bias",
                    "attn_dst_bias": f"output/head{index}/attn_dst/bias",
                    "output_bias": f"output/head{index}/output_bias",
                }
                for index in range(2)
            ],
        }
        binding = validate_checkpoint_structure(config, manifest, graph_count=1, feature_dim=1433)
        self.assertEqual(binding["d_in_profile"], [1433, 64])
        self.assertEqual(binding["K_out"], 2)
        tensor_shapes = {}
        for spec in manifest["hidden_head_specs"]:
            layer = manifest["hidden_layers"][spec["layer_index"]]
            tensor_shapes.update(
                {
                    spec["seq_kernel"]: (1, layer["input_dim"], layer["head_dim"]),
                    spec["attn_src_kernel"]: (1, layer["head_dim"], 1),
                    spec["attn_dst_kernel"]: (1, layer["head_dim"], 1),
                    spec["attn_src_bias"]: (1,),
                    spec["attn_dst_bias"]: (1,),
                    spec["output_bias"]: (layer["head_dim"],),
                }
            )
        for spec in manifest["output_head_specs"]:
            tensor_shapes.update(
                {
                    spec["seq_kernel"]: (1, 64, 7),
                    spec["attn_src_kernel"]: (1, 7, 1),
                    spec["attn_dst_kernel"]: (1, 7, 1),
                    spec["attn_src_bias"]: (1,),
                    spec["attn_dst_bias"]: (1,),
                    spec["output_bias"]: (7,),
                }
            )
        validate_checkpoint_tensor_shapes(manifest, tensor_shapes)
        malformed_shapes = dict(tensor_shapes)
        malformed_shapes[manifest["output_head_specs"][1]["seq_kernel"]] = (1, 63, 7)
        with self.assertRaises(ValueError):
            validate_checkpoint_tensor_shapes(manifest, malformed_shapes)
        pair_manifest = dict(manifest)
        pair_manifest["hidden_head_specs"] = [
            {key: value for key, value in spec.items() if key != "global_head_index"}
            for spec in manifest["hidden_head_specs"]
        ]
        self.assertEqual(
            validate_checkpoint_structure(config, pair_manifest, graph_count=1, feature_dim=1433)["L"],
            3,
        )

        duplicate_local = dict(manifest)
        duplicate_local["hidden_head_specs"] = [dict(spec) for spec in manifest["hidden_head_specs"]]
        duplicate_local["hidden_head_specs"][1]["local_head_index"] = 0
        with self.assertRaises(ValueError):
            validate_checkpoint_structure(config, duplicate_local, graph_count=1, feature_dim=1433)

        wrong_layer = dict(manifest)
        wrong_layer["hidden_head_specs"] = [dict(spec) for spec in manifest["hidden_head_specs"]]
        wrong_layer["hidden_head_specs"][0]["layer_index"] = 1
        with self.assertRaises(ValueError):
            validate_checkpoint_structure(config, wrong_layer, graph_count=1, feature_dim=1433)

        duplicate_global = dict(manifest)
        duplicate_global["hidden_head_specs"] = [dict(spec) for spec in manifest["hidden_head_specs"]]
        duplicate_global["hidden_head_specs"][1]["global_head_index"] = 0
        with self.assertRaises(ValueError):
            validate_checkpoint_structure(config, duplicate_global, graph_count=1, feature_dim=1433)

    def test_legacy_l2_k1_singleton_profiles_are_rejected(self) -> None:
        config = parse_config(config_path("cora"))
        for field, value in (("L", "2"), ("K_out", "1"), ("d_in_profile", "1433"), ("hidden_profile", "8x8")):
            with self.subTest(field=field):
                legacy = dict(config)
                legacy[field] = value
                with self.assertRaises(ValueError):
                    validate_model_structure_config(legacy, feature_dim=1433)

    def test_collector_marks_missing_fixture_without_success(self) -> None:
        with tempfile.TemporaryDirectory(dir=BASELINE / "tests") as directory:
            row, errors = inspect_cell(Path(directory) / "cora" / "cpu" / "warm", "cora", "cpu")
        self.assertEqual(row["status"], "missing")
        self.assertFalse(row["artifact_inventory_valid"])
        self.assertIn("missing_result", errors)

    def test_validator_rejects_incomplete_fixture(self) -> None:
        manifest, manifest_id = load_resource_manifest(require_staged=False)
        with tempfile.TemporaryDirectory(dir=BASELINE / "tests") as directory:
            errors = check_cell(Path(directory) / "cora" / "cpu" / "warm", "cora", "cpu", manifest, manifest_id)
        self.assertTrue(errors)

    def test_inventory_rejects_extra_artifact(self) -> None:
        with tempfile.TemporaryDirectory(dir=BASELINE / "tests") as directory:
            run = Path(directory) / "cora" / "cpu" / "warm"
            artifact = run / "formal_artifacts"
            artifact.mkdir(parents=True)
            (artifact / "unexpected.bin").write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                artifact_inventory(run, require_complete=True)

    def test_path_contract_rejects_symlink_and_hardlink(self) -> None:
        with tempfile.TemporaryDirectory(dir=BASELINE / "tests") as directory:
            root = Path(directory)
            target = root / "target.bin"
            target.write_bytes(b"fixture")
            link = root / "link.bin"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                ensure_contained(link, label="fixture symlink")
            hard = root / "hard.bin"
            os.link(target, hard)
            with self.assertRaises(ValueError):
                ensure_regular_file(hard, label="fixture hardlink")

    def _valid_cell_fixture(self, root: Path, device: str = "cpu") -> tuple[Path, dict, str]:
        """Build a tiny structural fixture; no production result is written."""
        manifest, manifest_id = load_resource_manifest(require_staged=False)
        run = root / "cora" / device / "warm"
        artifacts = run / "formal_artifacts"
        artifacts.mkdir(parents=True)
        current = {
            "config_sha256": "sha256:fixture-config",
            "protocol_sha256": "sha256:fixture-protocol",
            "resource_manifest_id": manifest_id,
            "resource_inventory_digest": "sha256:fixture-inventory",
            "checkpoint_manifest_id": "sha256:fixture-checkpoint",
            "checkpoint_manifest_sha256": "fixture-checkpoint",
            "checkpoint_tensors_sha256": "fixture-tensors",
            "model_shape_id": "sha256:fixture-shape",
            "model_arch_id": "fixture-arch",
            "model_param_id": "fixture-param",
            "source_static_table_id": "fixture-static-source",
            "checkpoint_binding": {
                "dataset": "cora",
                "L": 3,
                "hidden_profile": [[8, 8], [8, 8]],
                "d_in_profile": [1, 64],
                "K_out": 2,
                "model_arch_id": "fixture-arch",
                "model_param_id": "fixture-param",
                "batching_rule": "whole_graph_single",
                "batch_graphs": 1,
            },
        }
        profile = {
            "schema": "zkDL_gat_static_tables_v1",
            "source_static_table_id": current["source_static_table_id"],
            "generator_id": "fixture",
            "rounding_mode": "nearest_ties_away",
            "canonical_order": "kind_then_signed_input_ascending",
            "lrelu": {"min": -1, "max": 1, "input_scale": 256, "output_scale": 256, "rows": 3},
            "elu": {"min": -1, "max": 1, "input_scale": 16, "output_scale": 16, "rows": 3},
            "exp": {"min": 0, "max": 1, "input_scale": 256, "output_scale": 16, "rows": 2},
            "rows_digest": "sha256:fixture-rows",
        }
        profile["static_tables_id"] = final_validator._json_id(profile, domain=b"zkDL-GAT-static-table-profile-v1\0")
        operation_families = [{"residuals": [{"PublicTableLookup": {"values": [1, 2], "bit_columns": [3]}}]}]
        relation = {
            "schema": "zkDL_gat_canonical_adapter_v2",
            "dataset": "cora",
            "node_count": 1,
            "edge_count": 1,
            "graph_count": 1,
            "feature_dim": 1,
            "class_count": 1,
            "public_features": [3],
            "node_ptr": [0, 1],
            "edge_ptr": [0, 1],
            "src": [0],
            "dst": [0],
            "static_table_profile": profile,
            "operation_families": operation_families,
            "resource_manifest_id": manifest_id,
            "config_sha256": current["config_sha256"],
            "checkpoint_manifest_id": current["checkpoint_manifest_id"],
            "model_shape_id": current["model_shape_id"],
            "static_tables_id": profile["static_tables_id"],
            "static_table_rows_id": final_validator._table_rows_digest(operation_families),
            "model_arch_id": current["model_arch_id"],
            "model_param_id": current["model_param_id"],
            "canonical_schema": {
                "schema": "zkDL_gat_canonical_relation_schema_v1",
                "version": 1,
                "architecture": {
                    "L": 3,
                    "hidden_profile": [
                        {"head_count": 8, "head_dim": 8},
                        {"head_count": 8, "head_dim": 8},
                    ],
                    "d_in_profile": [1, 64],
                    "K_out": 2,
                },
            },
        }
        public = final_validator._public_instance(relation)
        relation["graph_features_digest"] = "sha256:" + final_validator._feature_digest(relation, public)
        relation["topology_digest"] = "sha256:" + final_validator._topology_digest(relation, public)
        ledger = {
            "schema": "zkDL_gat_integer_bound_ledger_v3",
            "dataset": "cora",
            "node_count": 1,
            "edge_count": 1,
            "max_in_degree": 1,
            "range_bits": 1,
            "bit_width": 8,
            "table_domains": {key: [profile[key]["min"], profile[key]["max"]] for key in ("lrelu", "elu", "exp")},
            "scales": {},
            "heads": [],
            "field": "BLS12-381-Fr",
            "field_half_modulus": "fixture",
            "field_fit": "fixture",
            "failures": [],
            "graph_count": 1,
            "feature_dim": 1,
            "class_count": 1,
            "L": 3,
            "hidden_profile": [[8, 8], [8, 8]],
            "d_in_profile": [1, 64],
            "K_out": 2,
            "resource_manifest_id": manifest_id,
            "config_sha256": current["config_sha256"],
            "checkpoint_manifest_id": current["checkpoint_manifest_id"],
            "model_shape_id": current["model_shape_id"],
            "static_tables_id": profile["static_tables_id"],
            "static_table_rows_id": relation["static_table_rows_id"],
            "model_arch_id": current["model_arch_id"],
            "model_param_id": current["model_param_id"],
        }
        ledger["integer_bound_ledger_id"] = final_validator._json_id({key: value for key, value in ledger.items()})
        relation["integer_bound_ledger"] = ledger
        relation["integer_bound_ledger_id"] = ledger["integer_bound_ledger_id"]
        preflight = {
            "schema": "zkDL_gat_preflight_v1",
            "status": "accepted",
            "dataset": "cora",
            **public,
            "L": 3,
            "hidden_profile": [[8, 8], [8, 8]],
            "d_in_profile": [1, 64],
            "K_out": 2,
            "config_sha256": current["config_sha256"],
            "protocol_sha256": current["protocol_sha256"],
            "resource_manifest_id": manifest_id,
            "resource_inventory_digest": current["resource_inventory_digest"],
            "graph_features_digest": relation["graph_features_digest"],
            "topology_digest": relation["topology_digest"],
            "checkpoint_manifest_id": current["checkpoint_manifest_id"],
            "checkpoint_manifest_sha256": current["checkpoint_manifest_sha256"],
            "checkpoint_tensors_sha256": current["checkpoint_tensors_sha256"],
            "model_shape_id": current["model_shape_id"],
            "static_tables_id": profile["static_tables_id"],
            "static_table_rows_id": relation["static_table_rows_id"],
            "static_tables": profile,
            "model_arch_id": current["model_arch_id"],
            "model_param_id": current["model_param_id"],
            "checkpoint_binding": current["checkpoint_binding"],
            "public_instance": public,
            "ledger": ledger,
        }
        relation_path = artifacts / "canonical_relation.json"
        preflight_path = artifacts / "preflight.json"
        relation_path.write_text(json.dumps(relation, sort_keys=True), encoding="utf-8")
        preflight_path.write_text(json.dumps(preflight, sort_keys=True), encoding="utf-8")
        (artifacts / "statement.bin").write_bytes(b"statement")
        proof_file = artifacts / "proof.bin"
        proof_file.write_bytes(b"proof")
        proof_sha = hashlib.sha256(proof_file.read_bytes()).hexdigest()
        proof = {
            "protocol_id": "GAT-ZKML-zkDL",
            "proof_format": "zkDL-GAT-proof-envelope-v3",
            "proof_generated": True,
            "verified": False,
            "statement_id": "fixture-statement",
            "model_registry_id": "fixture-registry",
            "device": device,
            "gpu_telemetry": None,
            "proof_path": "formal_artifacts/proof.bin",
            "proof_sha256": proof_sha,
            "proof_size_bytes": len(b"proof"),
            "model_session_commitments": 1,
            "model_rerandomization_proofs": 1,
            "model_session_rerandomization": "pedersen_blinder_schnorr_v1",
        }
        verifier = {
            **proof,
            "proof_generated": None,
            "verified": True,
            "independent_verifier": True,
            "verifier_reads_witness": False,
            "gpu_telemetry_validated": True,
            "model_rerandomization_verified": True,
        }
        (run / "proof_manifest.json").write_text(json.dumps(proof, sort_keys=True), encoding="utf-8")
        (run / "verifier_manifest.json").write_text(json.dumps(verifier, sort_keys=True), encoding="utf-8")
        result = {
            "protocol_id": "GAT-ZKML-zkDL",
            "dataset": "cora",
            "device": device,
            "status": "ok",
            "proof_generated": True,
            "verified": True,
            "formal_experiment_eligible": True,
            "statement_id": proof["statement_id"],
            "model_registry_id": proof["model_registry_id"],
            "proof_sha256": proof_sha,
            "proof_size_bytes": len(b"proof"),
            "proof_format": proof["proof_format"],
            "model_session_commitments": 1,
            "model_rerandomization_proofs": 1,
            "model_session_rerandomization": "pedersen_blinder_schnorr_v1",
            "privacy_profile_status": "supported_multi_session_registered_model",
            "gpu_telemetry": None,
            **{key: preflight[key] for key in ("config_sha256", "protocol_sha256", "resource_manifest_id", "resource_inventory_digest", "graph_features_digest", "topology_digest", "checkpoint_manifest_id", "model_shape_id", "static_tables_id", "static_table_rows_id", "model_arch_id", "model_param_id")},
            "L": 3,
            "hidden_profile": [[8, 8], [8, 8]],
            "d_in_profile": [1, 64],
            "K_out": 2,
            "checkpoint_binding": current["checkpoint_binding"],
            "integer_bound_ledger_id": ledger["integer_bound_ledger_id"],
            "public_instance": public,
        }
        provenance = {
            "config_sha256": current["config_sha256"],
            "protocol_sha256": current["protocol_sha256"],
            "resource_manifest_id": manifest_id,
            "resource_inventory_digest": current["resource_inventory_digest"],
            "preflight_sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
            "canonical_relation_sha256": hashlib.sha256(relation_path.read_bytes()).hexdigest(),
            "statement_sha256": hashlib.sha256((artifacts / "statement.bin").read_bytes()).hexdigest(),
            "proof_manifest_sha256": hashlib.sha256((run / "proof_manifest.json").read_bytes()).hexdigest(),
            "verifier_manifest_sha256": hashlib.sha256((run / "verifier_manifest.json").read_bytes()).hexdigest(),
        }
        result["provenance"] = provenance
        result_path = run / "result.json"
        result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        from runner_contract import artifact_inventory

        inventory = artifact_inventory(run)
        success = {
            "schema": "zkDL_gat_cell_success_manifest_v1",
            "status": "verified",
            "dataset": "cora",
            "device": device,
            **{key: result[key] for key in ("protocol_id", "config_sha256", "protocol_sha256", "resource_manifest_id", "resource_inventory_digest", "integer_bound_ledger_id", "graph_features_digest", "topology_digest", "public_instance", "checkpoint_manifest_id", "model_shape_id", "static_tables_id", "static_table_rows_id", "model_arch_id", "model_param_id")},
            "proof_sha256": proof_sha,
            "statement_id": proof["statement_id"],
            "model_registry_id": proof["model_registry_id"],
            "gpu_telemetry": None,
            "L": 3,
            "hidden_profile": [[8, 8], [8, 8]],
            "d_in_profile": [1, 64],
            "K_out": 2,
            "checkpoint_binding": current["checkpoint_binding"],
            "provenance": provenance,
            "files": inventory,
        }
        (run / "success_manifest.json").write_text(json.dumps(success, sort_keys=True), encoding="utf-8")
        return run, current, manifest_id

    def test_validator_rejects_current_binding_tamper_fixtures(self) -> None:
        cases = (
            ("config_sha256", "preflight_config_sha256"),
            ("protocol_sha256", "preflight_protocol_sha256"),
            ("resource_inventory_digest", "preflight_resource_inventory_digest"),
        )
        for field, expected_error in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory(dir=BASELINE / "tests") as directory:
                run, current, manifest_id = self._valid_cell_fixture(Path(directory))
                current = dict(current)
                current[field] = "sha256:tampered"
                errors = check_cell(run, "cora", "cpu", {}, manifest_id, current_identity=current)
                self.assertIn(f"cora/cpu:{expected_error}", errors)

    def test_validator_rejects_ledger_topology_public_and_success_tamper(self) -> None:
        tamper_cases = ("ledger", "topology", "public_instance", "success")
        for kind in tamper_cases:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=BASELINE / "tests") as directory:
                run, current, manifest_id = self._valid_cell_fixture(Path(directory))
                if kind == "ledger":
                    path = run / "formal_artifacts" / "preflight.json"
                    value = json.loads(path.read_text(encoding="utf-8"))
                    value["ledger"]["graph_count"] = 2
                    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
                elif kind == "topology":
                    path = run / "formal_artifacts" / "canonical_relation.json"
                    value = json.loads(path.read_text(encoding="utf-8"))
                    value["edge_ptr"] = [0, 0]
                    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
                elif kind == "public_instance":
                    path = run / "formal_artifacts" / "preflight.json"
                    value = json.loads(path.read_text(encoding="utf-8"))
                    value["public_instance"]["node_count"] = 2
                    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
                else:
                    path = run / "success_manifest.json"
                    value = json.loads(path.read_text(encoding="utf-8"))
                    value["config_sha256"] = "sha256:tampered"
                    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
                errors = check_cell(run, "cora", "cpu", {}, manifest_id, current_identity=current)
                self.assertTrue(errors)

    def test_collector_and_validator_reject_cpu_fake_gpu_telemetry(self) -> None:
        with tempfile.TemporaryDirectory(dir=BASELINE / "tests") as directory:
            run, current, manifest_id = self._valid_cell_fixture(Path(directory))
            result_path = run / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["gpu_telemetry"] = {"provider_id": "fake"}
            result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
            errors = check_cell(run, "cora", "cpu", {}, manifest_id, current_identity=current)
            self.assertIn("cora/cpu:cpu_result_gpu_telemetry_present", errors)
            _, collector_errors = inspect_cell(run, "cora", "cpu")
            self.assertIn("cpu_result_gpu_telemetry_present", collector_errors)


if __name__ == "__main__":
    unittest.main()
