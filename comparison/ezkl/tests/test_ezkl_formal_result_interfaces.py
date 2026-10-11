"""Bounded metadata and final-result interface checks; no formal proof is run."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
sys.path.insert(0, str(ROOT / "scripts"))

import collect_results  # noqa: E402
import formal_common  # noqa: E402
import produce_public_statement  # noqa: E402
import validate_final  # noqa: E402


class EzklFormalResultInterfacesTest(unittest.TestCase):
    def test_real_metadata_normalizes_five_graph_counts_without_file_changes(self) -> None:
        expected = dict(zip(formal_common.DATASETS, ("1", "1", "1", "295", "1")))
        resource_paths = [
            path for dataset in formal_common.DATASETS
            for path in (
                ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg",
                ROOT / "resources/canonical/cache" / dataset / "meta.cfg",
            )
        ]
        resource_hashes = {path: formal_common.sha256(path) for path in resource_paths}
        source_hash = formal_common.sha256(ROOT / "scripts/formal_common.py")
        for dataset in formal_common.DATASETS:
            with self.subTest(dataset=dataset):
                config = formal_common.validate_config(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg", dataset)
                metadata_path = ROOT / "resources/canonical/cache" / dataset / "meta.cfg"
                raw_metadata = formal_common.parse_cfg(metadata_path)
                metadata = formal_common.validate_dataset_metadata(metadata_path, dataset, config)
                self.assertEqual(metadata["graph_count"], expected[dataset])
                self.assertEqual(set(metadata), set(raw_metadata) | ({"graph_count"} if "graph_count" not in raw_metadata else set()))
                self.assertEqual(("feature_format" in metadata), dataset in {"ppi", "ogbn_arxiv"})
        self.assertEqual(resource_hashes, {path: formal_common.sha256(path) for path in resource_paths})
        self.assertEqual(source_hash, formal_common.sha256(ROOT / "scripts/formal_common.py"))

    def _canonical_fixture(self, dataset: str) -> dict[str, object]:
        root = Path(tempfile.mkdtemp(prefix="ezkl-formal-interface-", dir=TESTS))
        self.addCleanup(shutil.rmtree, root)
        config_source = ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg"
        metadata_source = ROOT / "resources/canonical/cache" / dataset / "meta.cfg"
        scales_source = ROOT / "resources/canonical/configs/quantization_scale_chain.cfg"
        config_path = root / "resources/canonical/configs" / f"{dataset}_secure.cfg"
        metadata_path = root / "resources/canonical/cache" / dataset / "meta.cfg"
        scales_path = root / "resources/canonical/configs/quantization_scale_chain.cfg"
        data_dir = root / "resources/canonical/cache" / dataset
        for path in (config_path, metadata_path, scales_path, data_dir):
            path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_bytes(config_source.read_bytes())
        scales_path.write_bytes(scales_source.read_bytes())
        config = formal_common.parse_cfg(config_path)
        metadata = formal_common.parse_cfg(metadata_source)
        graph_count = int(metadata.get("graph_count", "1"))
        nodes = graph_count if dataset == "ppi" else 2
        features = classes = 1
        metadata.update({"num_nodes": str(nodes), "num_features": str(features), "num_classes": str(classes)})
        if dataset in {"ppi", "ogbn_arxiv"}:
            metadata["feature_shape"] = f"{nodes},{features}"
            metadata["feature_length"] = str(nodes * features)
            feature_path = data_dir / "features.npy"
            feature_values = np.ones((nodes, features), dtype=np.float32)
            if dataset == "ogbn_arxiv":
                feature_values[:, 0] = np.arange(1, nodes + 1, dtype=np.float32)
            np.save(feature_path, feature_values, allow_pickle=False)
            metadata["feature_sha256"] = formal_common.sha256(feature_path)
        else:
            feature_path = data_dir / "features.txt"
            feature_path.write_text(f"{nodes} {features}\n" + "".join("0:1.0\n" for _ in range(nodes)), encoding="utf-8")
        edges_path = data_dir / "edges.txt"
        edges_path.write_text(
            "".join(f"{index} {index} {index}\n" if graph_count > 1 else f"{index} {index}\n" for index in range(nodes)),
            encoding="utf-8",
        )
        metadata_path.write_text("".join(f"{key} = {value}\n" for key, value in metadata.items()), encoding="utf-8")
        metadata = formal_common.parse_cfg(metadata_path)
        scales = formal_common.parse_cfg(scales_path)
        canonical_features, resolved_feature_path = produce_public_statement.feature_values(
            data_dir, metadata, produce_public_statement.scale_value(scales, "s_input"), metadata["input_preprocess_id"], dataset,
        )
        self.assertEqual(resolved_feature_path, feature_path)
        node_ptr = list(range(nodes + 1)) if graph_count > 1 else [0, nodes]
        edge_ptr = list(range(nodes + 1)) if graph_count > 1 else [0, nodes]
        digest = "1" * 64
        public = {
            "topology": {"id": f"topology:{dataset}:{graph_count}", "digest": digest},
            "static_tables": {"id": config["static_table_id"], "digest": config["static_table_digest"]},
            "quant": {"id": config["quant_cfg_id"], "digest": digest},
            "ledger": {"id": config["degree_bound_id"], "digest": config["degree_bound_digest"]},
            "circuit": {
                "relation_id": formal_common.CANONICAL_RELATION,
                "compiled_circuit_id": f"compiled-{dataset}",
                "compiled_circuit_digest": digest,
                "settings_id": f"settings-{dataset}",
                "settings_digest": digest,
                "vk_id": f"vk-{dataset}",
                "vk_digest": digest,
                "model_graph_id": f"model-graph-{dataset}",
                "model_graph_digest": digest,
                "ezkl_security_profile_id": config["ezkl_security_profile_id"],
            },
            "model_hash": "0x" + "2" * 64,
            "model": {"hash": "0x" + "2" * 64, "hash_domain": "ezkl-gat-model-v1", "canonical_model_digest": digest},
            "feature_table": [int(value) for value in canonical_features.flat],
            "node_ids": list(range(nodes)),
            "src": list(range(nodes)),
            "dst": list(range(nodes)),
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
            "y": [0] * (nodes * classes),
        }
        output_bytes = b"ezkl-gat-public-y-v1" + b"".join(value.to_bytes(16, "little", signed=True) for value in public["y"])
        statement = {
            "schema_version": 1,
            "relation_id": formal_common.CANONICAL_RELATION,
            "dataset": dataset,
            "proof_scope": formal_common.CANONICAL_PROOF_SCOPE,
            "privacy_policy": formal_common.CANONICAL_PRIVACY,
            "public_statement_mode": formal_common.CANONICAL_STATEMENT_MODE,
            "output_mode": formal_common.CANONICAL_OUTPUT_MODE,
            "model_arch_id": config["model_arch_id"],
            "quant_cfg_id": config["quant_cfg_id"],
            "static_table_id": config["static_table_id"],
            "degree_bound_id": config["degree_bound_id"],
            "ezkl_security_profile_id": config["ezkl_security_profile_id"],
            "public_ezkl": public,
            "output_digest": hashlib.sha256(output_bytes).hexdigest(),
            "public_output_sha256": "3" * 64,
            "statement_digest": "4" * 64,
        }
        statement_path = root / "statement.json"
        statement_path.write_text(json.dumps(statement, sort_keys=True), encoding="utf-8")
        records = []
        for path, record_dataset, role in (
            (config_path, dataset, "canonical_secure_config"),
            (metadata_path, dataset, "canonical_dataset_cache"),
            (scales_path, "all", "canonical_quantization_config"),
            (feature_path, dataset, "canonical_dataset_cache"),
            (edges_path, dataset, "canonical_dataset_cache"),
        ):
            records.append({"path": path.relative_to(root).as_posix(), "dataset": record_dataset, "role": role, "size": path.stat().st_size, "sha256": formal_common.sha256(path)})
        manifest_path = root / "resources/resource_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"baseline": "ezkl", "resource_root": "resources/canonical", "files": records}, sort_keys=True), encoding="utf-8")
        topology = public["topology"]
        context = {
            "schema_version": 1,
            "relation_id": formal_common.CANONICAL_RELATION,
            "dataset": dataset,
            "model_arch_id": config["model_arch_id"],
            "quant_cfg_id": config["quant_cfg_id"],
            "quantization": public["quant"],
            "static_tables": public["static_tables"],
            "ledger": public["ledger"],
            "topology": topology,
            "model": {},
            "circuit": {},
            "ezkl_security_profile_id": config["ezkl_security_profile_id"],
        }
        context_path = root / "runs" / dataset / "producer_context.json"
        context_path.parent.mkdir(parents=True, exist_ok=True)
        context_path.write_text(json.dumps(context, sort_keys=True), encoding="utf-8")
        resource_digest = formal_common.sha256(manifest_path)
        producer_manifest = {
            "dataset": dataset,
            "relation_id": formal_common.CANONICAL_RELATION,
            "resource_manifest_sha256": resource_digest,
            "resource_manifest_path": manifest_path.relative_to(root).as_posix(),
            "config_path": config_path.relative_to(root).as_posix(),
            "config_sha256": formal_common.sha256(config_path),
            "feature_path": feature_path.relative_to(root).as_posix(),
            "feature_sha256": formal_common.sha256(feature_path),
            "setup_context_path": context_path.relative_to(root).as_posix(),
            "setup_context_sha256": formal_common.sha256(context_path),
        }
        (context_path.parent / "producer_manifest.json").write_text(json.dumps(producer_manifest, sort_keys=True), encoding="utf-8")
        return {
            "root": root,
            "dataset": dataset,
            "config": config,
            "statement_path": statement_path,
            "context_path": context_path,
            "manifest_path": manifest_path,
            "resource_digest": resource_digest,
            "statement": statement,
            "feature_path": feature_path,
        }

    def _assert_canonical_fixture(self, dataset: str) -> None:
        fixture = self._canonical_fixture(dataset)
        root = fixture["root"]
        config = fixture["config"]
        with patch.object(formal_common, "ROOT", root), patch.object(formal_common, "MANIFEST", fixture["manifest_path"]):
            statement = formal_common.validate_public_statement(
                fixture["statement_path"],
                dataset=dataset,
                quant_cfg_id=config["quant_cfg_id"],
                static_tables_id=config["static_table_id"],
                degree_bound_id=config["degree_bound_id"],
                degree_bound_digest=config["degree_bound_digest"],
            )
            formal_common.validate_canonical_public_inputs(
                statement,
                dataset=dataset,
                secure_config=config,
                setup_context_path=fixture["context_path"],
                expected_resource_manifest_sha256=fixture["resource_digest"],
            )
        graph_count = int(formal_common.parse_cfg(fixture["root"] / "resources/canonical/cache" / dataset / "meta.cfg").get("graph_count", "1"))
        self.assertEqual(len(fixture["statement"]["public_ezkl"]["node_ptr"]), graph_count + 1)

    def test_canonical_public_inputs_cora_default_graph_count(self) -> None:
        self._assert_canonical_fixture("cora")

    def test_canonical_public_inputs_citeseer_default_graph_count(self) -> None:
        self._assert_canonical_fixture("citeseer")

    def test_canonical_public_inputs_pubmed_default_graph_count(self) -> None:
        self._assert_canonical_fixture("pubmed")

    def test_canonical_public_inputs_ppi_all_295_graphs(self) -> None:
        self._assert_canonical_fixture("ppi")

    def test_canonical_public_inputs_ogbn_arxiv_npy_schema(self) -> None:
        self._assert_canonical_fixture("ogbn_arxiv")

    def test_invalid_graph_count_npy_schema_and_feature_identity_reject(self) -> None:
        ppi_config = formal_common.parse_cfg(ROOT / "resources/canonical/configs/ppi_secure.cfg")
        ppi_metadata = formal_common.parse_cfg(ROOT / "resources/canonical/cache/ppi/meta.cfg")
        with tempfile.TemporaryDirectory(prefix="ezkl-invalid-meta-", dir=TESTS) as directory:
            path = Path(directory) / "meta.cfg"
            for invalid_count in (None, "294"):
                values = dict(ppi_metadata)
                if invalid_count is None:
                    values.pop("graph_count")
                else:
                    values["graph_count"] = invalid_count
                path.write_text("".join(f"{key} = {value}\n" for key, value in values.items()), encoding="utf-8")
                with self.assertRaisesRegex(formal_common.ContractError, "graph_count"):
                    formal_common.validate_dataset_metadata(path, "ppi", ppi_config)
            arxiv_config = formal_common.parse_cfg(ROOT / "resources/canonical/configs/ogbn_arxiv_secure.cfg")
            arxiv_metadata = dict(formal_common.parse_cfg(ROOT / "resources/canonical/cache/ogbn_arxiv/meta.cfg"))
            arxiv_metadata["feature_dtype"] = "float64"
            path.write_text("".join(f"{key} = {value}\n" for key, value in arxiv_metadata.items()), encoding="utf-8")
            with self.assertRaisesRegex(formal_common.ContractError, "dtype/order"):
                formal_common.validate_dataset_metadata(path, "ogbn_arxiv", arxiv_config)
        fixture = self._canonical_fixture("ogbn_arxiv")
        manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
        expected_feature = fixture["feature_path"].relative_to(fixture["root"]).as_posix()
        next(record for record in manifest["files"] if record["path"] == expected_feature)["path"] = expected_feature + ".stale"
        fixture["manifest_path"].write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        with patch.object(formal_common, "ROOT", fixture["root"]), patch.object(formal_common, "MANIFEST", fixture["manifest_path"]):
            statement = formal_common.validate_public_statement(fixture["statement_path"], dataset="ogbn_arxiv")
            with self.assertRaisesRegex(formal_common.ContractError, "exactly one record"):
                formal_common.validate_canonical_public_inputs(
                    statement,
                    dataset="ogbn_arxiv",
                    secure_config=fixture["config"],
                    setup_context_path=fixture["context_path"],
                    expected_resource_manifest_sha256=formal_common.sha256(fixture["manifest_path"]),
                )

    def test_collector_and_final_matrix_status_path_and_identity_interfaces(self) -> None:
        session_id = "a" * 32
        source_digest, resource_digest = "b" * 64, "c" * 64
        with tempfile.TemporaryDirectory(prefix="ezkl-final-matrix-", dir=TESTS) as directory:
            root = Path(directory)
            cells = []
            for index, dataset in enumerate(formal_common.DATASETS):
                relative = f"runs/formal/{dataset}/result.json"
                result_path = root / relative
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text("{}\n", encoding="utf-8")
                cells.append({"dataset": dataset, "status": "VERIFY_OK", "formal_session_id": session_id, "proof_sha256": str(index + 1) * 64, "_result_path": relative})
            matrix = collect_results.build_matrix_document(
                cells,
                datasets=formal_common.DATASETS,
                source_sha256=source_digest,
                resource_sha256=resource_digest,
                collected_at_utc="2026-10-11T00:00:00+00:00",
                formal_session_id=session_id,
                backends=None,
            )
            matrix_path = root / "results/formal_matrix.json"
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            matrix_path.write_text(json.dumps(matrix, sort_keys=True), encoding="utf-8")
            by_path = {cell["_result_path"]: cell for cell in cells}

            def revalidate_cell(path: Path, *, expected_session_id: str | None = None) -> dict[str, object]:
                self.assertEqual(expected_session_id, session_id)
                return by_path[path.relative_to(root).as_posix()]

            with (
                patch.object(validate_final, "ROOT", root),
                patch.object(validate_final, "source_tree_hash", return_value=source_digest),
                patch.object(validate_final, "resource_manifest_hash", return_value=resource_digest),
                patch.object(validate_final, "validate_cell", side_effect=revalidate_cell),
            ):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(validate_final.validate_matrix(matrix_path, session_id=session_id), 0)
                self.assertIn("FINAL_VALIDATION_OK cells=5", output.getvalue())
                for mutation, expected_error in (
                    (lambda value: value["cells"][0].update(proof_sha256="f" * 64), "stale for"),
                    (lambda value: value["cells"][0].update(_result_path="../../outside/result.json"), "escapes baseline"),
                    (lambda value: value.update(collection_status="FINAL_VALIDATION_OK"), "not produced by the strict collector"),
                ):
                    changed = copy.deepcopy(matrix)
                    mutation(changed)
                    matrix_path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
                    with self.assertRaisesRegex(formal_common.ContractError, expected_error):
                        validate_final.validate_matrix(matrix_path, session_id=session_id)
            required = {key: None for key in collect_results.REQUIRED_RESULT_KEYS}
            required.update({"schema_version": 1, "status": "NOT_READY", "dataset": "cora"})
            invalid_result = root / "invalid" / "cora" / "result.json"
            invalid_result.parent.mkdir(parents=True)
            invalid_result.write_text(json.dumps(required), encoding="utf-8")
            with self.assertRaisesRegex(formal_common.ContractError, "not a verified native result"):
                collect_results.validate_cell(invalid_result, expected_session_id=session_id)
            required.update({"status": "VERIFY_OK", "native_backend": formal_common.NATIVE_FORMAL_BACKEND, "formal_session_id": "d" * 32})
            invalid_result.write_text(json.dumps(required), encoding="utf-8")
            with self.assertRaisesRegex(formal_common.ContractError, "different formal session"):
                collect_results.validate_cell(invalid_result, expected_session_id=session_id)
            with patch.object(formal_common, "ROOT", root):
                with self.assertRaisesRegex(formal_common.ContractError, "baseline-local relative path"):
                    collect_results.path_from_result("../outside/result.json", "fixture result path")


if __name__ == "__main__":
    unittest.main()
