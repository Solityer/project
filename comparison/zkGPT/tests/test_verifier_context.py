from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest

import numpy as np

from zkgpt_gat.data import LocalGraph
from zkgpt_gat.verifier_context import (
    VerifierContext,
    _aux_arrays,
    _gkr_families,
    _matrix_families,
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


MANIFEST = {
    "hidden_layers": [{"layer_index": 0, "input_dim": 2, "head_count": 1, "head_dim": 2}],
    "output_head_specs": [{"head_index": 0}],
    "C": 2,
    "K_out": 1,
    "L": 2,
    "d_in_profile": [2],
}


class ExternalVerifierContextTests(unittest.TestCase):
    def test_aux_layout_contains_native_sparse_wires_and_public_counts(self) -> None:
        ledger = {"bit_width": 4, "table_domains": {"lrelu": [-1, 1], "elu": [-1, 1], "exp": [0, 1]}}
        arrays = dict(_aux_arrays(_graph(), MANIFEST, 12, ledger))
        self.assertEqual(arrays["hidden.l1.h0.gkr.R_smax"], [2])
        self.assertEqual(arrays["hidden.l1.h0.gkr.R_num_carry"], [2, 2])
        self.assertFalse(any(".memory." in name for name in arrays))

    def test_matrix_vector_descriptor_keeps_three_dimensional_shape(self) -> None:
        specs = _matrix_families(_graph(), MANIFEST)
        attention = next(item for item in specs if item["label"] == "hidden.l1.h0.attention_src")
        self.assertEqual(attention["spec"]["shape"], [2, 2, 1])

    def test_gkr_nested_spec_is_exact_and_tamper_rejected(self) -> None:
        expected = _gkr_families(_graph(), MANIFEST, [1, 2, 3, 4])
        context = object.__new__(VerifierContext)
        object.__setattr__(context, "expected_gkr_families", tuple(expected))
        object.__setattr__(context, "context_digest", "test")
        claims = [{"kind": item["kind"], "proof": {"spec": item["spec"]}} for item in expected]
        self.assertEqual(context.validate_subproof_families(None, None, {"claims": claims})["status"], "VERIFY_OK")
        tampered = [dict(item) for item in claims]
        tampered[0] = {"kind": tampered[0]["kind"], "proof": {"spec": dict(tampered[0]["proof"]["spec"])}}
        tampered[0]["proof"]["spec"]["label"] = "attacker-selected"
        self.assertEqual(context.validate_subproof_families(None, None, {"claims": tampered})["status"], "VERIFY_FAILED")

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


if __name__ == "__main__":
    unittest.main()
