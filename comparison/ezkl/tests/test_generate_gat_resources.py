from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_gat_resources import (  # noqa: E402
    graph_info,
    quantized_tensor_abs_max,
    quantized_model_maxima,
    quantized_value,
    round_away,
)


class GenerateGatResourcesTest(unittest.TestCase):
    def _root(self, temporary: str, dataset: str, edges: str, node_ptr: str | None = None) -> Path:
        root = Path(temporary)
        cache = root / "resources/canonical/cache"
        (cache / dataset).mkdir(parents=True)
        (cache / dataset / "edges.txt").write_text(edges)
        if node_ptr is not None:
            (cache / "ppi").mkdir(exist_ok=True)
            (cache / "ppi/node_ptr.txt").write_text(node_ptr)
        return root

    def test_matches_runtime_dedup_and_missing_self_loop_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary, "cora", "0 1\n0 1\n2 1\n1 0\n1 1\n2 2\n")
            self.assertEqual(graph_info(root, "cora", {"num_nodes": "4", "graph_count": "1"}), (4, 7, 3))

    def test_ppi_degree_and_edge_count_respect_two_graph_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary, "ppi", "0 0 1\n0 1 1\n1 2 3\n", "0 2 4\n")
            self.assertEqual(graph_info(root, "ppi", {"num_nodes": "4", "graph_count": "2"}), (4, 6, 2))

    def test_rejects_invalid_node_pointer_endpoint_and_cross_graph_edge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary, "ppi", "0 1 2\n", "0 0 4\n")
            with self.assertRaisesRegex(ValueError, "node_ptr"):
                graph_info(root, "ppi", {"num_nodes": "4", "graph_count": "2"})
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary, "cora", "4 0\n")
            with self.assertRaisesRegex(ValueError, "endpoint"):
                graph_info(root, "cora", {"num_nodes": "4", "graph_count": "1"})
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary, "ppi", "0 1 2\n", "0 2 4\n")
            with self.assertRaisesRegex(ValueError, "crosses graph boundary"):
                graph_info(root, "ppi", {"num_nodes": "4", "graph_count": "2"})

    def test_rejects_non_ppi_multigraph_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary, "cora", "")
            with self.assertRaisesRegex(ValueError, "exactly one graph"):
                graph_info(root, "cora", {"num_nodes": "4", "graph_count": "2"})

    def test_quantized_model_uses_authenticated_per_head_multipliers(self) -> None:
        fixture = Path(__file__).parent / "fixtures/quantized_model_scale_chain.json"
        document = json.loads(fixture.read_text(encoding="utf-8"))
        tensors = {
            name: (entry["shape"], entry["values"])
            for name, entry in document["tensors"].items()
        }

        maxima = quantized_model_maxima(tensors, document["checkpoint"], document["scales"])

        self.assertEqual(quantized_value(0.03125, 32, "scale-32 fixture"), 1)
        self.assertEqual(round_away(0.03125 * 4), 0)
        self.assertEqual(quantized_value(-0.5, 1, "negative-half fixture"), -1)
        self.assertEqual(maxima["hidden/layer0/head0/seq"], 1)
        self.assertEqual(maxima["hidden/layer0/head0/attn_src"], 1)
        self.assertEqual(maxima["hidden/layer0/head0/attn_dst"], 1)
        self.assertEqual(maxima["output/head0"], 0)
        self.assertEqual(maxima["output/head1/seq"], 1)
        self.assertEqual(maxima["output/head1/attn_src"], 1)
        self.assertEqual(maxima["output/head1/attn_dst"], 1)
        self.assertEqual(maxima["output/head1"], 1)

        invalid_scales = dict(document["scales"], s_a_src_hidden_0="3")
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            quantized_model_maxima(tensors, document["checkpoint"], invalid_scales)

    def test_quantizer_enforces_authenticated_shapes_and_native_i128_boundaries(self) -> None:
        fixture = Path(__file__).parent / "fixtures/quantized_model_scale_chain.json"
        document = json.loads(fixture.read_text(encoding="utf-8"))
        checkpoint = document["checkpoint"]
        tensors = {
            name: (entry["shape"], entry["values"])
            for name, entry in document["tensors"].items()
        }

        reshaped = dict(tensors)
        name = "hidden/layer0/head0/seq/kernel"
        reshaped[name] = ([1, 1], tensors[name][1])
        with self.assertRaisesRegex(ValueError, "shape/count"):
            quantized_model_maxima(reshaped, checkpoint, document["scales"])

        # Rust accepts either canonical output-bias representation when the
        # tensor_index records either one.
        alternate_bias = dict(tensors)
        bias_name = "output/head1/output_bias"
        alternate_bias[bias_name] = ([1, 2, 1], tensors[bias_name][1])
        quantized_model_maxima(alternate_bias, checkpoint, document["scales"])

        too_wide_scales = dict(document["scales"], s_a_src_hidden_0=str(1 << 128))
        with self.assertRaisesRegex(ValueError, "u128"):
            quantized_model_maxima(tensors, checkpoint, too_wide_scales)
        with self.assertRaisesRegex(ValueError, "u128"):
            quantized_value(1.0, 1 << 128, "u128 overflow fixture")

        self.assertEqual(quantized_value(float(1 << 100), 1, "i128 supported fixture"), 1 << 100)
        self.assertEqual(quantized_value(float(1 << 127), 1, "i128 positive saturation fixture"), (1 << 127) - 1)
        self.assertEqual(quantized_value(-float(1 << 127), 1, "i128 negative saturation fixture"), -(1 << 127))

    def test_native_quantizer_boundary_equivalence(self) -> None:
        positive_boundary = float(1 << 127)
        negative_boundary = -float(1 << 127)
        self.assertEqual(quantized_value(positive_boundary, 1, "positive endpoint"), (1 << 127) - 1)
        self.assertEqual(quantized_value(negative_boundary, 1, "negative endpoint"), -(1 << 127))

        outside_positive = float.fromhex("0x1.0000000000001p+127")
        outside_negative = -outside_positive
        with self.assertRaisesRegex(ValueError, "exceeds signed i128 range"):
            quantized_value(outside_positive, 1, "outside positive endpoint")
        with self.assertRaisesRegex(ValueError, "exceeds signed i128 range"):
            quantized_value(outside_negative, 1, "outside negative endpoint")

        largest_finite = float.fromhex("0x1.fffffffffffffp+1023")
        with self.assertRaisesRegex(ValueError, "non-finite scaled"):
            quantized_value(largest_finite, 1 << 127, "finite source product overflow")
        with self.assertRaisesRegex(ValueError, "non-finite checkpoint"):
            quantized_value(float("inf"), 1, "non-finite source")

        # Native (scaled + 0.5).floor behavior increments this exact f64.
        self.assertEqual(round_away(float((1 << 52) + 1)), (1 << 52) + 2)

        with self.assertRaisesRegex(ValueError, "absolute bound exceeds signed i128"):
            quantized_tensor_abs_max(
                {"weight": ([1], [negative_boundary])}, "weight", 1, 1, [1],
            )


if __name__ == "__main__":
    unittest.main()
