from __future__ import annotations

import json
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from zkgpt_gat.config import load_app_config, read_config
from zkgpt_gat.constants import (
    DATASET_ORDER,
    FORMAL_GAT_CLASS_COUNTS,
    FORMAL_GAT_D_IN_PROFILE,
    FORMAL_GAT_HIDDEN_PROFILE,
    FORMAL_GAT_K_OUT,
    FORMAL_GAT_LAYER_COUNT,
    PROOF_BACKEND_ID,
    PROTOCOL_ID,
)
from zkgpt_gat.formal_gate import formal_gate_status, load_capabilities, validate_capabilities
from zkgpt_gat.reference_forward import REFERENCE_FORWARD_ENGINE, reference_forward_source_sha256
from zkgpt_gat.resources import check_resource_layout, read_resource_manifest
from zkgpt_gat.result_validation import (
    ResultValidationError,
    canonical_artifact_dir,
    canonical_manifest_path,
    canonical_results_dir,
    load_canonical_manifest,
    required_artifact_names,
    required_artifact_file,
    resolve_inside,
    resolve_results_dir,
    validate_artifact_inventory,
    validate_canonical_cell_dir_from_path,
    validate_artifact,
    validate_matrix_rows,
)
from zkgpt_gat.util import digest_json, write_json
from zkgpt_gat.proof import _cleanup_proof_work_dir, _proof_work_dir
from zkgpt_gat.runner import _prepare_artifact_directory


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ResultValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # A repository checkout intentionally ignores runtime scratch state,
        # but these tests use it as the parent of isolated temporary fixtures.
        # Create only the local directory; no production artifact is seeded.
        (ROOT / "runs_tmp").mkdir(parents=True, exist_ok=True)

    def _canonical_cell_fixture(self):
        """Create one isolated canonical cell without a production manifest."""
        return tempfile.TemporaryDirectory(prefix="zkgpt-containment-")

    @staticmethod
    def _cell_paths(name: str) -> tuple[pathlib.Path, pathlib.Path]:
        root = pathlib.Path(name)
        cell = root / "results" / "runs" / "cpu" / "cora"
        cell.mkdir(parents=True)
        return root, cell

    @staticmethod
    def _load_collector_module():
        script = ROOT / "scripts" / "collect_results.py"
        spec = importlib.util.spec_from_file_location("zkgpt_test_collect_results", script)
        if spec is None or spec.loader is None:
            raise AssertionError(f"cannot load collector script: {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_formal_capability_gate_is_exactly_ready(self) -> None:
        gate = formal_gate_status()
        self.assertEqual(gate["status"], "READY")
        self.assertEqual(gate["errors"], [])
        capabilities = load_capabilities()
        self.assertTrue(capabilities["formal_experiment_eligible"])
        self.assertEqual(capabilities["blockers"], [])
        tampered = dict(capabilities)
        tampered["output_mode"] = "proof_only"
        self.assertTrue(validate_capabilities(tampered))

    def _fixture_manifest(self, directory: pathlib.Path) -> pathlib.Path:
        path = directory / "run_manifest.json"
        path.write_text(
            json.dumps({
                "schema": "zkGPT_test_fixture_v1",
                "baseline": "zkGPT",
                "dataset": "cora",
                "backend": "cpu",
                "status": "TEST_FIXTURE_VALID",
                "proof_generated": False,
                "verified": False,
            }) + "\n",
            encoding="utf-8",
        )
        return path

    def _cache_manifest(self, directory: pathlib.Path) -> pathlib.Path:
        path = directory / "run_manifest.json"
        path.write_text(
            json.dumps({
                "schema": "zkGPT_synthetic_e2e_cache_v1",
                "baseline": "zkGPT",
                "dataset": "synthetic-e2e",
                "backend": "cpu",
                "status": "TEST_FIXTURE_VALID",
            }) + "\n",
            encoding="utf-8",
        )
        return path

    def _metadata_order_fixture(
        self,
        root: pathlib.Path,
        metadata: dict[str, object] | None,
    ) -> tuple[pathlib.Path, dict[str, object]]:
        """Create the authenticated prefix needed to exercise metadata ordering.

        The remaining proof files are deliberately empty JSON objects: the
        valid-metadata case must reach the producer-verifier check after the
        independent-report/metadata binding, while malformed metadata must
        fail at that binding point.  This keeps the regression small without
        manufacturing a proof.
        """
        cell = root / "results" / "runs" / "cpu" / "cora"
        cell.mkdir(parents=True)
        for name in required_artifact_names("cpu", include_manifest=True):
            if name == "metadata.json" and metadata is None:
                continue
            (cell / name).write_text("{}\n", encoding="utf-8")

        hidden_profile = [
            {"layer_index": 0, "head_count": 8, "head_dim": 8},
            {"layer_index": 1, "head_count": 8, "head_dim": 8},
        ]
        fields: dict[str, object] = {
            "status": "VERIFY_OK",
            "protocol_id": PROTOCOL_ID,
            "proof_backend_id": PROOF_BACKEND_ID,
            "dataset": "cora",
            "backend": "cpu",
            "full_dataset": True,
            "proof_scope": "full_gat_forward",
            "full_gat_forward_proved": True,
            "final_logits_bound": True,
            "privacy_extension_profile_id": "privacy-test",
            "privacy_profile_status": "VERIFY_OK",
            "verifier_reads_witness": False,
            "witness_included_in_proof": False,
            "public_output_digest": "public-output-digest",
            "proof_contribution": {"Pi_GKR": True},
            "gpu_hot_path": False,
            "proof_schema_scan": True,
            "privacy_masking_verified": True,
            "verifier_input_boundary": {"private_input_files_read": []},
            "model_arch_id": "test-model-arch",
            "model_param_id": "test-model-param",
            "quant_cfg_id": "gat_l3_scale_chain",
            "input_preprocess_id": "row_normalize",
            "L": 3,
            "hidden_profile": hidden_profile,
            "d_in_profile": [1433, 64],
            "K_out": 2,
            "C": 7,
            "independent_reference_forward": True,
            "independent_reference_provenance": {
                "status": "VERIFY_OK",
                "checks": {"independent_reference_forward": True},
                "forward_engine": REFERENCE_FORWARD_ENGINE,
                "reference_forward_source_sha256": reference_forward_source_sha256(),
                "dataset": "cora",
                "quant_cfg_id": "gat_l3_scale_chain",
                "input_preprocess_id": "row_normalize",
            },
        }
        report: dict[str, object] = {
            "schema": "zkGPT_independent_verification_v4",
            "status": "VERIFY_OK",
            "checks": {"all_checks": True},
            "verified_fields": fields,
        }
        (cell / "independent_verification.json").write_text(
            json.dumps(report) + "\n", encoding="utf-8"
        )
        if metadata is not None:
            (cell / "metadata.json").write_text(
                json.dumps(metadata) + "\n", encoding="utf-8"
            )

        manifest: dict[str, object] = {
            "schema": "zkGPT_gat_run_manifest_v4",
            "dataset": "cora",
            "backend": "cpu",
            "artifact_path": str(cell),
            "status": "VERIFY_OK",
            "protocol_id": PROTOCOL_ID,
            "proof_backend_id": PROOF_BACKEND_ID,
            "gpu_backend": {"backend": "cpu", "status": "VERIFY_OK", "gpu_hot_path": False},
            "independent_verification_status": "VERIFY_OK",
            "independent_verification_sha256": "artifact-digest",
            "independent_report_digest": digest_json(report),
        }
        for key in (
            "proof_scope", "full_gat_forward_proved", "final_logits_bound",
            "privacy_extension_profile_id", "privacy_profile_status",
            "verifier_reads_witness", "witness_included_in_proof",
            "public_output_digest", "proof_contribution", "gpu_hot_path",
            "proof_schema_scan", "privacy_masking_verified", "verifier_input_boundary",
            "independent_reference_forward",
        ):
            manifest[key] = fields[key]
        manifest["full_dataset"] = fields["full_dataset"]
        manifest["artifact_digests"] = {
            name: {
                "sha256": "artifact-digest",
                "size_bytes": (cell / name).stat().st_size,
            }
            for name in required_artifact_names("cpu")
            if (cell / name).exists()
        }
        manifest_path = cell / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        return manifest_path, manifest

    def test_metadata_is_loaded_before_independent_binding(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="zkgpt-metadata-order-", dir=str(ROOT / "runs_tmp")
        ) as name:
            root = pathlib.Path(name)
            metadata = {
                "dataset": "cora",
                "backend": "cpu",
                "protocol_id": PROTOCOL_ID,
                "proof_backend_id": PROOF_BACKEND_ID,
                "model_arch_id": "test-model-arch",
                "model_param_id": "test-model-param",
                "quant_cfg_id": "gat_l3_scale_chain",
                "input_preprocess_id": "row_normalize",
                "dim_cfg": {
                    "L": 3,
                    "hidden_profile": [
                        {"layer_index": 0, "head_count": 8, "head_dim": 8},
                        {"layer_index": 1, "head_count": 8, "head_dim": 8},
                    ],
                    "d_in_profile": [1433, 64],
                    "K_out": 2,
                    "C": 7,
                },
            }
            manifest_path, manifest = self._metadata_order_fixture(root, metadata)
            cell = manifest_path.parent
            with (
                patch("zkgpt_gat.result_validation.load_canonical_manifest", return_value=(cell, manifest)),
                patch("zkgpt_gat.result_validation.validate_required_artifacts"),
                patch("zkgpt_gat.result_validation.sha256_file", return_value="artifact-digest"),
            ):
                with self.assertRaisesRegex(
                    ResultValidationError, "producer verifier report is not successful"
                ):
                    validate_artifact(root, manifest_path)

    def test_missing_or_malformed_metadata_fails_closed(self) -> None:
        cases = (
            ("missing", None, "artifact file is missing"),
            (
                "malformed",
                {
                    "dataset": "cora",
                    "backend": "cpu",
                    "protocol_id": PROTOCOL_ID,
                    "proof_backend_id": PROOF_BACKEND_ID,
                },
                "metadata dim_cfg is missing",
            ),
        )
        for label, metadata, expected in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                prefix="zkgpt-metadata-fail-", dir=str(ROOT / "runs_tmp")
            ) as name:
                root = pathlib.Path(name)
                manifest_path, manifest = self._metadata_order_fixture(root, metadata)
                cell = manifest_path.parent
                with (
                    patch("zkgpt_gat.result_validation.load_canonical_manifest", return_value=(cell, manifest)),
                    patch("zkgpt_gat.result_validation.validate_required_artifacts"),
                    patch("zkgpt_gat.result_validation.sha256_file", return_value="artifact-digest"),
                ):
                    with self.assertRaisesRegex(ResultValidationError, expected):
                        validate_artifact(root, manifest_path)

    def test_fixture_is_explicit_and_never_production_accepted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-result-fixture-", dir=str(ROOT / "runs_tmp")) as name:
            manifest = self._fixture_manifest(pathlib.Path(name))
            self.assertEqual(validate_artifact(ROOT, manifest, allow_test_fixture=True)["status"], "TEST_FIXTURE_VALID")
            with self.assertRaises(ResultValidationError):
                validate_artifact(ROOT, manifest)

    def test_fixture_matrix_logic_is_test_only(self) -> None:
        rows = [
            {"dataset": dataset, "backend": backend, "status": "TEST_FIXTURE_VALID", "verified": False}
            for backend in ("cpu", "gpu")
            for dataset in DATASET_ORDER
        ]
        validate_matrix_rows(rows, allow_test_fixture=True)
        with self.assertRaises(ResultValidationError):
            validate_matrix_rows(rows)

    def test_synthetic_e2e_cache_schema_is_explicitly_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-cache-schema-", dir=str(ROOT / "runs_tmp")) as name:
            manifest = self._cache_manifest(pathlib.Path(name))
            with self.assertRaisesRegex(ResultValidationError, "synthetic_e2e_cache_v1"):
                validate_artifact(ROOT, manifest, allow_test_fixture=True)

    def test_failed_diagnostic_schema_is_explicitly_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-failed-diagnostic-", dir=str(ROOT / "runs_tmp")) as name:
            path = pathlib.Path(name) / "run_manifest.json"
            path.write_text(
                json.dumps({
                    "schema": "zkGPT_test_synthetic_e2e_failed_diagnostic_v1",
                    "status": "FAILED_DIAGNOSTIC",
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResultValidationError, "test-only schema"):
                validate_artifact(ROOT, path, allow_test_fixture=True)

    def test_resource_manifest_and_generated_configs_are_baseline_local(self) -> None:
        manifest = read_resource_manifest(ROOT)
        self.assertEqual(set(manifest["datasets"]), set(DATASET_ORDER))
        report = check_resource_layout(ROOT, DATASET_ORDER, require_staged=False)
        self.assertEqual(report["schema"], manifest["schema"])
        for backend in ("cpu", "gpu"):
            for path in sorted((ROOT / "configs" / "generated" / backend).glob("*.cfg")):
                entries = read_config(path)
                for key in ("data_root", "cache_root", "checkpoint_bundle", "reference_output_dir", "public_y_path", "static_table_path", "resource_manifest"):
                    self.assertFalse(pathlib.Path(entries[key]).is_absolute(), (path, key))
                self.assertEqual(entries["static_tables_id"], entries["static_table_id"])
                self.assertEqual(entries["edge_rule_id"], "per_node_self_loop+edge_gid_then_dst_stable")
                config = load_app_config(path, ROOT)
                self.assertEqual(config.layer_count, FORMAL_GAT_LAYER_COUNT)
                self.assertEqual(
                    [(item.head_count, item.head_dim) for item in config.hidden_profile],
                    list(FORMAL_GAT_HIDDEN_PROFILE),
                )
                canonical_dataset = "ogbn-arxiv" if config.dataset == "ogbn_arxiv" else config.dataset
                self.assertEqual(config.d_in_profile, list(FORMAL_GAT_D_IN_PROFILE[canonical_dataset]))
                self.assertEqual(config.k_out, FORMAL_GAT_K_OUT)
                self.assertEqual(config.num_classes, FORMAL_GAT_CLASS_COUNTS[canonical_dataset])

    def test_path_escape_is_rejected(self) -> None:
        with self.assertRaises(ResultValidationError):
            resolve_inside(ROOT, "../../outside", "fixture path")

    def test_canonical_cell_directory_is_valid_before_manifest_exists(self) -> None:
        with self._canonical_cell_fixture() as name:
            root, cell = self._cell_paths(name)
            self.assertFalse((cell / "run_manifest.json").exists())
            self.assertEqual(validate_canonical_cell_dir_from_path(root, cell), cell)

    def test_canonical_cell_directory_symlink_is_rejected(self) -> None:
        with self._canonical_cell_fixture() as name:
            root = pathlib.Path(name)
            parent = root / "results" / "runs" / "cpu"
            parent.mkdir(parents=True)
            target = root / "outside-cell"
            target.mkdir()
            alias = parent / "cora"
            os.symlink(target, alias, target_is_directory=True)
            with self.assertRaisesRegex(ResultValidationError, "symlink|real directory|alias"):
                validate_canonical_cell_dir_from_path(root, alias)

    def test_canonical_parent_component_symlink_is_rejected(self) -> None:
        with self._canonical_cell_fixture() as name:
            root = pathlib.Path(name)
            runs = root / "results" / "runs"
            runs.mkdir(parents=True)
            target = root / "outside-backend"
            (target / "cora").mkdir(parents=True)
            alias = runs / "cpu"
            os.symlink(target, alias, target_is_directory=True)
            with self.assertRaisesRegex(ResultValidationError, "symlink|real directory|alias"):
                validate_canonical_cell_dir_from_path(root, alias / "cora")

    def test_canonical_cell_path_escape_is_rejected(self) -> None:
        with self._canonical_cell_fixture() as name:
            root, _cell = self._cell_paths(name)
            with self.assertRaises(ResultValidationError):
                validate_canonical_cell_dir_from_path(root, root.parent / "escaped-cell")

    def test_metadata_and_proof_symlinks_to_dummy_tensors_are_rejected(self) -> None:
        for artifact_name in ("metadata.json", "proof_payload.json"):
            with self.subTest(artifact_name=artifact_name):
                with self._canonical_cell_fixture() as name:
                    _root, cell = self._cell_paths(name)
                    dummy = cell.parent.parent.parent.parent / "dummy tensors.txt"
                    dummy.write_text("not an artifact\n", encoding="utf-8")
                    os.symlink(dummy, cell / artifact_name)
                    with self.assertRaisesRegex(ResultValidationError, "symlink"):
                        required_artifact_file(cell, artifact_name)

    def test_hardlinked_artifact_is_rejected(self) -> None:
        with self._canonical_cell_fixture() as name:
            _root, cell = self._cell_paths(name)
            dummy = cell.parent.parent.parent.parent / "dummy tensors.txt"
            dummy.write_text("not an artifact\n", encoding="utf-8")
            os.link(dummy, cell / "proof_payload.json")
            with self.assertRaisesRegex(ResultValidationError, "hardlink"):
                required_artifact_file(cell, "proof_payload.json")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO tests require os.mkfifo")
    def test_fifo_artifact_is_rejected(self) -> None:
        with self._canonical_cell_fixture() as name:
            _root, cell = self._cell_paths(name)
            os.mkfifo(cell / "metadata.json")
            with self.assertRaisesRegex(ResultValidationError, "regular file"):
                required_artifact_file(cell, "metadata.json")

    @staticmethod
    def _complete_inventory_fixture(cell: pathlib.Path) -> None:
        for name in required_artifact_names("cpu", include_manifest=True):
            (cell / name).write_text("{}\n", encoding="utf-8")

    def test_exact_artifact_inventory_accepts_canonical_cell(self) -> None:
        with self._canonical_cell_fixture() as name:
            _root, cell = self._cell_paths(name)
            self._complete_inventory_fixture(cell)
            inventory = validate_artifact_inventory(
                cell, "cpu", require_independent=True, require_manifest=True
            )
            self.assertEqual(set(inventory), set(required_artifact_names("cpu", include_manifest=True)))

    def test_exact_artifact_inventory_rejects_extra_file_and_nested_directory(self) -> None:
        for kind in ("file", "directory", "nested_symlink"):
            with self.subTest(kind=kind), self._canonical_cell_fixture() as name:
                _root, cell = self._cell_paths(name)
                self._complete_inventory_fixture(cell)
                if kind == "file":
                    (cell / "extra-proof.json").write_text("{}\n", encoding="utf-8")
                elif kind == "directory":
                    (cell / "native_gkr_work").mkdir()
                else:
                    nested = cell / "native_gkr_work"
                    nested.mkdir()
                    target = cell / "metadata.json"
                    os.symlink(target, nested / "escape.json")
                with self.assertRaisesRegex(ResultValidationError, "non-canonical|nested artifact"):
                    validate_artifact_inventory(cell, "cpu", require_independent=True, require_manifest=True)

    def test_exact_artifact_inventory_rejects_hardlink_and_fifo(self) -> None:
        for kind in ("hardlink", "fifo"):
            with self.subTest(kind=kind), self._canonical_cell_fixture() as name:
                _root, cell = self._cell_paths(name)
                self._complete_inventory_fixture(cell)
                if kind == "hardlink":
                    outside = cell.parent.parent.parent.parent / "inventory-source.json"
                    outside.write_text("{}\n", encoding="utf-8")
                    cell.joinpath("metadata.json").unlink()
                    os.link(outside, cell / "metadata.json")
                else:
                    cell.joinpath("metadata.json").unlink()
                    os.mkfifo(cell / "metadata.json")
                with self.assertRaisesRegex(ResultValidationError, "hardlink|regular file"):
                    validate_artifact_inventory(cell, "cpu", require_independent=True, require_manifest=True)

    def test_proof_work_session_is_baseline_local_and_cleaned(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-proof-work-") as name:
            root = pathlib.Path(name)
            session = _proof_work_dir(root)
            nested = session / "native_gkr_work"
            nested.mkdir()
            (nested / "intermediate.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(session.parent, root / "runs_tmp" / "proof-work")
            self.assertEqual(session.resolve().parent, session.parent)
            _cleanup_proof_work_dir(session, root)
            self.assertFalse(session.exists())

    @staticmethod
    def _runner_cell(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
        results = root / "results"
        cell = results / "runs" / "cpu" / "cora"
        cell.mkdir(parents=True)
        return results, cell

    def test_runner_force_rejects_unsafe_stale_entries(self) -> None:
        cases = ("extra", "nested", "symlink", "hardlink", "fifo")
        for kind in cases:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix="zkgpt-runner-force-") as name:
                root = pathlib.Path(name)
                results, cell = self._runner_cell(root)
                if kind == "extra":
                    (cell / "extra-proof.json").write_text("{}\n", encoding="utf-8")
                elif kind == "nested":
                    (cell / "native_gkr_work").mkdir()
                elif kind == "symlink":
                    target = root / "outside.json"
                    target.write_text("outside\n", encoding="utf-8")
                    os.symlink(target, cell / "metadata.json")
                elif kind == "hardlink":
                    target = root / "outside.json"
                    target.write_text("outside\n", encoding="utf-8")
                    os.link(target, cell / "metadata.json")
                else:
                    os.mkfifo(cell / "metadata.json")
                with self.assertRaisesRegex(RuntimeError, "non-canonical|nested|symlink|hardlink|owned regular"):
                    _prepare_artifact_directory(root, results, cell, True)
                self.assertTrue(cell.exists())

    def test_runner_force_clears_only_canonical_regular_stale_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-runner-force-clean-") as name:
            root = pathlib.Path(name)
            results, cell = self._runner_cell(root)
            (cell / "metadata.json").write_text("{}\n", encoding="utf-8")
            (cell / "run_manifest.json").write_text("{}\n", encoding="utf-8")
            prepared = _prepare_artifact_directory(root, results, cell, True)
            self.assertEqual(prepared, cell)
            self.assertEqual(list(cell.iterdir()), [])
            # An empty canonical cell can be reused without force.
            self.assertEqual(_prepare_artifact_directory(root, results, cell, False), cell)

    def test_manifest_dataset_and_backend_mismatch_is_rejected(self) -> None:
        with self._canonical_cell_fixture() as name:
            root, cell = self._cell_paths(name)
            manifest_path = cell / "run_manifest.json"
            for dataset, backend in (("citeseer", "cpu"), ("cora", "gpu")):
                with self.subTest(dataset=dataset, backend=backend):
                    write_json(
                        manifest_path,
                        {
                            "schema": "zkGPT_gat_run_manifest_v4",
                            "dataset": dataset,
                            "backend": backend,
                            "artifact_path": str(cell),
                        },
                    )
                    with self.assertRaisesRegex(ResultValidationError, "does not match canonical path"):
                        load_canonical_manifest(root, manifest_path)

    def test_collector_missing_manifests_is_ineligible(self) -> None:
        """A missing cell manifest cannot become a formal result row."""
        collector = self._load_collector_module()
        with tempfile.TemporaryDirectory(prefix="zkgpt-collector-missing-") as name:
            root = pathlib.Path(name)
            results = root / "results"
            original_root = collector.ROOT
            collector.ROOT = root
            try:
                rows, errors = collector._collect(results)
                self.assertEqual(len(rows), 10)
                self.assertEqual(len(errors), 10)
                self.assertTrue(all(row["status"] == "not_run" for row in rows))
                self.assertFalse(collector._write_outputs(results, rows, errors))
                payload = json.loads(
                    (results / "final_zkgpt_results.json").read_text(encoding="utf-8")
                )
                self.assertEqual(payload["status"], "not_complete")
                self.assertFalse(payload["formal_experiment_eligible"])
            finally:
                collector.ROOT = original_root

    def test_write_json_rejects_aliases_and_special_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-write-json-") as name:
            root = pathlib.Path(name)
            dummy = root / "dummy tensors.txt"
            dummy.write_text("keep me\n", encoding="utf-8")

            symlink = root / "symlink.json"
            os.symlink(dummy, symlink)
            with self.assertRaisesRegex(OSError, "symlink"):
                write_json(symlink, {"changed": True})
            self.assertEqual(dummy.read_text(encoding="utf-8"), "keep me\n")

            hardlink = root / "hardlink.json"
            os.link(dummy, hardlink)
            with self.assertRaisesRegex(OSError, "hardlink"):
                write_json(hardlink, {"changed": True})
            self.assertEqual(dummy.read_text(encoding="utf-8"), "keep me\n")

            fifo = root / "fifo.json"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(OSError, "regular file"):
                write_json(fifo, {"changed": True})

    def test_write_json_atomic_write_is_regular_and_cleans_temp_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-write-json-atomic-") as name:
            root = pathlib.Path(name)
            output = root / "nested" / "result.json"
            write_json(output, {"step": 1})
            write_json(output, {"step": 2})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"step": 2})
            info = output.lstat()
            self.assertTrue(info.st_mode & 0o170000 == 0o100000)
            self.assertEqual(info.st_nlink, 1)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_canonical_result_layout_and_aliases_are_fail_closed(self) -> None:
        self.assertEqual(canonical_results_dir(ROOT), ROOT / "results")
        self.assertEqual(
            canonical_artifact_dir(ROOT, "cpu", "cora"),
            ROOT / "results" / "runs" / "cpu" / "cora",
        )
        self.assertEqual(
            canonical_manifest_path(ROOT, "gpu", "ogbn-arxiv"),
            ROOT / "results" / "runs" / "gpu" / "ogbn-arxiv" / "run_manifest.json",
        )
        self.assertEqual(resolve_results_dir(ROOT, ROOT / "results"), ROOT / "results")
        with self.assertRaises(ResultValidationError):
            resolve_results_dir(ROOT, ROOT / "results" / "runs")
        with self.assertRaises(ResultValidationError):
            resolve_results_dir(ROOT, ROOT.parent / "results")

    def test_collect_and_final_validator_reject_noncanonical_cli_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-tool-fixture-", dir=str(ROOT / "runs_tmp")) as name:
            temp_root = pathlib.Path(name)
            results = temp_root / "results"
            manifest_dir = results / "runs" / "cpu" / "cora"
            manifest_dir.mkdir(parents=True)
            self._fixture_manifest(manifest_dir)
            collect = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "collect_results.py"), "--results-dir", str(results)],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertNotEqual(collect.returncode, 0)
            validate = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "validate_final_results.py"), "--results-dir", str(results)],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertNotEqual(validate.returncode, 0)
            self.assertFalse((results / "final_zkgpt_results.json").exists())
            self.assertFalse((results / "local_validation.json").exists())


if __name__ == "__main__":
    unittest.main()
