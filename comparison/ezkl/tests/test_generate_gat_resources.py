from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_gat_resources import graph_info  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
