from __future__ import annotations

import pathlib
import json
import inspect
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from zkgpt_gat.bound_ledger import build_integer_bound_ledger
from zkgpt_gat.data import LocalGraph
from synthetic_e2e_support import TEST_QUANT_SCALE_PROFILE
from zkgpt_gat.rounding import ROUNDING_MODE_NEAREST
from zkgpt_gat.verifier_context import (
    VerifierContext,
    _canonical_model_architecture,
    _aux_arrays,
    _gkr_families,
    _matrix_families,
    _manifest,
)


def _graph() -> LocalGraph:
    return LocalGraph(
        name="context-test",
        num_nodes=2,
        num_features=2,
        num_classes=2,
        features=np.zeros((2, 2), dtype=np.int64),
        labels=np.zeros(2, dtype=np.int64),
        absolute_ids=np.asarray([0, 1], dtype=np.int64),
        src=np.asarray([0, 1], dtype=np.int64),
        dst=np.asarray([0, 1], dtype=np.int64),
        edge_graph=np.asarray([0, 0], dtype=np.int64),
        node_ptr=np.asarray([0, 2], dtype=np.int64),
        edge_ptr=np.asarray([0, 2], dtype=np.int64),
        graph_count=1,
        n_total=2,
        task_type="transductive_node_classification",
        report_unit="node",
        batching_rule="whole_graph_single",
        self_loop_rule="per_node",
        edge_sort_rule="edge_gid_then_dst_stable",
        chunking_rule="none",
    )


def _graph_with_more_edges_than_nodes() -> LocalGraph:
    return LocalGraph(
        name="context-test-non-equal",
        num_nodes=3,
        num_features=2,
        num_classes=2,
        features=np.zeros((3, 2), dtype=np.int64),
        labels=np.zeros(3, dtype=np.int64),
        absolute_ids=np.asarray([0, 1, 2], dtype=np.int64),
        src=np.asarray([0, 1, 2, 2], dtype=np.int64),
        dst=np.asarray([0, 1, 2, 2], dtype=np.int64),
        edge_graph=np.asarray([0, 0, 0, 0], dtype=np.int64),
        node_ptr=np.asarray([0, 3], dtype=np.int64),
        edge_ptr=np.asarray([0, 4], dtype=np.int64),
        graph_count=1,
        n_total=3,
        task_type="transductive_node_classification",
        report_unit="node",
        batching_rule="whole_graph_single",
        self_loop_rule="per_node",
        edge_sort_rule="edge_gid_then_dst_stable",
        chunking_rule="none",
    )


MANIFEST = {
    "hidden_layers": [{"layer_index": 0, "input_dim": 2, "head_count": 1, "head_dim": 2}],
    "output_head_specs": [{"head_index": 0}],
    "C": 2,
    "K_out": 1,
    "L": 2,
    "d_in_profile": [2],
}


def _ledger(graph: LocalGraph) -> dict[str, object]:
    """Build the authenticated v2 ledger used by descriptor fixtures.

    The production helpers intentionally do not infer a rounding mode.  Test
    fixtures must therefore carry the same explicit mode and range-family
    policy as a real public ledger instead of using a partial dictionary.
    """

    bounds = {
        "hidden.l1.h0": {"B_W": 1, "B_a_src": 1, "B_a_dst": 1, "B_b": 0},
        "output.h0": {"B_W": 1, "B_a_src": 1, "B_a_dst": 1, "B_b": 1},
    }
    return build_integer_bound_ledger(
        mode=ROUNDING_MODE_NEAREST,
        quant_cfg_id="exact_integer_gat",
        authenticated_quant_scales=dict(TEST_QUANT_SCALE_PROFILE),
        registered_model_bounds=bounds,
        hidden_profile=[{"layer_index": 0, "head_count": 1, "head_dim": 2}],
        d_in=[2, 2],
        classes=2,
        k_out=1,
        max_degree=max((int(value) for value in np.bincount(graph.dst, minlength=graph.num_nodes)), default=1),
        feature_bound=0,
        node_count=graph.num_nodes,
        range_bits=12,
    )


class ExternalVerifierContextTests(unittest.TestCase):
    def test_formal_verifier_requires_keyword_bound_statement(self) -> None:
        import zkgpt_gat.proof as proof

        parameters = inspect.signature(proof._verify_formal).parameters.values()
        self.assertTrue(all(item.kind is inspect.Parameter.KEYWORD_ONLY for item in parameters))

    def test_verify_cli_rejects_noncanonical_privacy_wire_order(self) -> None:
        import importlib.util

        script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "verify_proof.py"
        spec = importlib.util.spec_from_file_location("zkgpt_verify_canonical_wire", script)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix="zkgpt-privacy-wire-") as name:
            artifact = pathlib.Path(name)
            (artifact / "privacy_profile.json").write_text('{"b": 2, "a": 1}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                module._read_canonical_object(artifact, "privacy_profile.json")
            (artifact / "privacy_profile.json").write_text(
                json.dumps({"b": 2, "a": 1}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                module._read_canonical_object(artifact, "privacy_profile.json"),
                {"a": 1, "b": 2},
            )

    def test_manifest_projection_discards_tensor_and_producer_fields(self) -> None:
        raw = {
            "model_arch_id": "architecture-v1",
            "hidden_layers": [{"layer_index": 0, "input_dim": 2, "head_count": 1, "head_dim": 2}],
            "output_head_specs": [{"head_index": 0, "seq_kernel": "private.W"}],
            "C": 2,
            "K_out": 1,
            "L": 2,
            "d_in_profile": [2],
            "output_average_rule": "per_head_bias_then_arithmetic_mean",
            "tensors": [{"name": "private.W", "values": [1, 2]}],
            "tensors_sha256": "private-digest",
        }
        projected = _canonical_model_architecture(raw)
        self.assertNotIn("tensors", projected)
        self.assertNotIn("tensors_sha256", projected)
        self.assertNotIn("seq_kernel", projected["output_head_specs"][0])
        self.assertEqual(projected["hidden_layers"], raw["hidden_layers"])

    def test_manifest_loader_never_opens_tensor_value_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-public-manifest-") as name:
            root = pathlib.Path(name)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                '{"model_arch_id":"architecture-v1","hidden_layers":[],"output_head_specs":[],"C":1,"K_out":0,"L":1,"d_in_profile":[1],"output_average_rule":"per_head_bias_then_arithmetic_mean"}\n',
                encoding="utf-8",
            )
            tensors_path = root / "tensors.txt"
            tensors_path.write_text("private values must never be read\n", encoding="utf-8")
            original_open = pathlib.Path.open

            def reject_tensor_open(path: pathlib.Path, *args: object, **kwargs: object):
                if path.name == "tensors.txt":
                    raise AssertionError("independent context attempted to open private tensor values")
                return original_open(path, *args, **kwargs)

            with mock.patch.object(pathlib.Path, "open", new=reject_tensor_open):
                loaded = _manifest(root)
            self.assertIn("model_arch_id", loaded)

    def test_aux_layout_contains_native_sparse_wires_and_public_counts(self) -> None:
        ledger = _ledger(_graph())
        arrays = dict(_aux_arrays(_graph(), MANIFEST, 12, ledger))
        self.assertEqual(arrays["hidden.l1.h0.gkr.R_smax"], [2])
        self.assertEqual(arrays["hidden.l1.h0.gkr.R_num_carry"], [2, 2])
        self.assertFalse(any(".memory." in name for name in arrays))

    def test_non_equal_edge_node_aux_cardinality_uses_node_projections(self) -> None:
        graph = _graph_with_more_edges_than_nodes()
        ledger = _ledger(graph)
        arrays = dict(_aux_arrays(graph, MANIFEST, 12, ledger))
        self.assertEqual(arrays["hidden.l1.h0.E_src"], [3])
        self.assertEqual(arrays["hidden.l1.h0.E_dst"], [3])
        # Regression negative: an edge-shaped fresh-process descriptor must
        # not compare equal to the canonical node-shaped descriptor.
        self.assertNotEqual(arrays["hidden.l1.h0.E_src"], [4])
        self.assertNotEqual(arrays["hidden.l1.h0.E_dst"], [4])

    def test_matrix_vector_descriptor_keeps_three_dimensional_shape(self) -> None:
        specs = _matrix_families(_graph(), MANIFEST)
        attention = next(item for item in specs if item["label"] == "hidden.l1.h0.attention_src")
        self.assertEqual(attention["spec"]["shape"], [2, 2, 1])

    def test_gkr_nested_spec_is_exact_and_tamper_rejected(self) -> None:
        ledger = _ledger(_graph())
        expected = _gkr_families(_graph(), MANIFEST, [1, 2, 3, 4], ledger=ledger, range_bits=12)
        context = object.__new__(VerifierContext)
        object.__setattr__(context, "expected_gkr_families", tuple(expected))
        object.__setattr__(context, "context_digest", "test")
        claims = [{"kind": item["kind"], "proof": {"spec": item["spec"]}} for item in expected]
        self.assertEqual(context.validate_subproof_families(None, None, {"claims": claims})["status"], "VERIFY_OK")
        tampered = [dict(item) for item in claims]
        tampered[0] = {"kind": tampered[0]["kind"], "proof": {"spec": dict(tampered[0]["proof"]["spec"])}}
        tampered[0]["proof"]["spec"]["label"] = "attacker-selected"
        self.assertEqual(context.validate_subproof_families(None, None, {"claims": tampered})["status"], "VERIFY_FAILED")

    def test_gkr_requires_explicit_supported_rounding_mode(self) -> None:
        graph = _graph()
        with self.assertRaises(ValueError):
            _gkr_families(graph, MANIFEST, [1, 2, 3, 4], ledger=None)
        invalid = _ledger(graph)
        invalid["rounding_mode"] = "implicit_default"
        with self.assertRaises(ValueError):
            _gkr_families(graph, MANIFEST, [1, 2, 3, 4], ledger=invalid)

    def test_verify_cli_requires_external_context_flags(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(root / "scripts" / "verify_proof.py"), "--help"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        for flag in ("--project-root", "--config", "--trusted-registry-public", "--artifact-dir", "--public-y"):
            self.assertIn(flag, completed.stdout)

    def test_verify_cli_requires_strict_canonical_baseline_root(self) -> None:
        import importlib.util

        script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "verify_proof.py"
        spec = importlib.util.spec_from_file_location("zkgpt_verify_strict_root", script)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        root = pathlib.Path(module.ROOT)
        self.assertEqual(module._strict_baseline_root(str(root)), root)
        with self.assertRaises(ValueError):
            module._strict_baseline_root(str(root / ".." / root.name))
        with self.assertRaises(ValueError):
            module._strict_baseline_root(str(root.parent))
        with self.assertRaises(ValueError):
            module._strict_baseline_root("comparison/zkGPT")
        with tempfile.TemporaryDirectory(prefix="zkgpt-root-alias-") as name:
            alias = pathlib.Path(name) / "baseline-link"
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                module._strict_baseline_root(str(alias))

    def test_verify_cli_runs_formal_gate_before_artifact_path(self) -> None:
        import importlib.util

        script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "verify_proof.py"
        spec = importlib.util.spec_from_file_location("zkgpt_verify_gate_order", script)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        root = pathlib.Path(module.ROOT)
        with mock.patch.object(module, "require_formal_experiment", side_effect=RuntimeError("blocked")), mock.patch.object(
            module,
            "validate_canonical_cell_dir_from_path",
            side_effect=AssertionError("artifact path was inspected before formal gate"),
        ):
            code = module.main([
                "--project-root", str(root),
                "--config", str(root / "missing.cfg"),
                "--trusted-registry-public", str(root / "missing-registry.json"),
                "--artifact-dir", str(root / "results" / "runs" / "cpu" / "cora"),
                "--public-y", str(root / "missing-y.json"),
            ])
        self.assertEqual(code, 1)

    def test_verify_report_public_context_whitelist_excludes_private_values(self) -> None:
        import importlib.util

        script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "verify_proof.py"
        spec = importlib.util.spec_from_file_location("zkgpt_verify_proof_boundary", script)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertEqual(
            module.PUBLIC_CONTEXT_FILES,
            ["config", "model_manifest", "trusted_registry_public", "public_y"],
        )
        self.assertNotIn("tensors", " ".join(module.PUBLIC_CONTEXT_FILES).lower())


if __name__ == "__main__":
    unittest.main()
