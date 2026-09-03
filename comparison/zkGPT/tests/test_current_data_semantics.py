from __future__ import annotations

import pathlib
import sys
import tempfile
import types
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from synthetic_e2e_support import synthetic_graph, synthetic_model  # type: ignore  # noqa: E402
from zkgpt_gat.data import _validate_dataset_meta  # type: ignore  # noqa: E402
from zkgpt_gat.forward import canonical_input_integers, forward_full  # type: ignore  # noqa: E402
from zkgpt_gat.proof import artifact_ids  # type: ignore  # noqa: E402


class CurrentDataSemanticsTests(unittest.TestCase):
    def test_forward_trace_retains_original_canonical_input(self) -> None:
        _dataset, graph = synthetic_graph()
        model = synthetic_model()
        expected = canonical_input_integers(graph, model)
        trace = forward_full(graph, model)
        np.testing.assert_array_equal(trace.input_features, expected)
        np.testing.assert_array_equal(trace.hidden_layers[0].input, expected)

    def test_meta_identities_reject_old_ppi_and_ogbn_spellings(self) -> None:
        ppi = types.SimpleNamespace(
            dataset="ppi",
            entries={
                "batching_rule": "multi_graph_batch",
                "task_type": "inductive_multi_graph_node_classification",
                "report_unit": "graph",
            },
        )
        ppi_meta = {
            "name": "ppi",
            "num_classes": "121",
            "graph_count": "295",
            "task_type": "inductive_multi_graph_node_classification",
            "report_unit": "graph",
            "label_format": "multi_hot",
            "split_format": "node_masks",
        }
        _validate_dataset_meta(ppi, ppi_meta)
        with self.assertRaisesRegex(ValueError, "PPI metadata identity mismatch"):
            _validate_dataset_meta(ppi, {**ppi_meta, "split_format": "node_masks_v1"})

        ogbn = types.SimpleNamespace(dataset="ogbn_arxiv", entries={})
        ogbn_meta = {
            "name": "ogbn_arxiv",
            "num_classes": "40",
            "graph_count": "1",
            "task_type": "transductive_node_classification",
            "report_unit": "node",
        }
        _validate_dataset_meta(ogbn, ogbn_meta)
        with self.assertRaisesRegex(ValueError, "dataset meta name mismatch"):
            _validate_dataset_meta(ogbn, {**ogbn_meta, "name": "ogbn-arxiv"})

    def test_ppi_artifact_identity_binds_multilabel_masks_and_preprocess(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-artifact-ids-") as name:
            root = pathlib.Path(name)
            cache = root / "cache"
            bundle = root / "checkpoint"
            cache.mkdir()
            bundle.mkdir()
            for filename in (
                "meta.cfg",
                "edges.txt",
                "features.npy",
                "labels_multi.npy",
                "node_ptr.txt",
                "train_mask.npy",
                "val_mask.npy",
                "test_mask.npy",
            ):
                (cache / filename).write_bytes(filename.encode("ascii"))
            (bundle / "manifest.json").write_text("{}\n", encoding="utf-8")
            config_path = root / "ppi.cfg"
            config_path.write_text("dataset = ppi\n", encoding="utf-8")
            config = types.SimpleNamespace(
                dataset="ppi",
                cache_dir=cache,
                checkpoint_bundle=bundle,
                path=config_path,
                input_preprocess_id="row_normalize",
            )
            ids = artifact_ids(config, types.SimpleNamespace(name="ppi"))
            self.assertEqual(ids["data_id"]["dataset"], "ppi")
            self.assertEqual(ids["data_id"]["input_preprocess_id"], "row_normalize")
            self.assertEqual(
                set(ids["data_id"]["files"]),
                {
                    "meta.cfg",
                    "edges.txt",
                    "features.npy",
                    "labels_multi.npy",
                    "node_ptr.txt",
                    "train_mask.npy",
                    "val_mask.npy",
                    "test_mask.npy",
                },
            )


if __name__ == "__main__":
    unittest.main()
