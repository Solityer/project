"""Small producer/CLI contract tests; no dataset, SRS, or proof execution."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_public_y import (  # noqa: E402
    load_graph,
    read_tensors,
    validate_checkpoint_manifest,
    validate_hidden_bias_tensors,
)
from resource_policy import _is_private_regular_file  # noqa: E402


def graph_root(root: Path, edges: str) -> dict[str, str]:
    (root / "meta.cfg").write_text(
        "num_nodes=2\nnum_features=1\nnum_classes=2\ngraph_count=1\n",
        encoding="utf-8",
    )
    (root / "edges.txt").write_text(edges, encoding="utf-8")
    return {"num_nodes": "2", "num_features": "1", "num_classes": "2", "graph_count": "1"}


def hidden_spec(name: str = "hidden/layer0/head0") -> dict[str, str]:
    return {
        "attn_src_bias": name + "/attn_src/bias",
        "attn_dst_bias": name + "/attn_dst/bias",
        "output_bias": name + "/output_bias",
    }


class PublicYContractTests(unittest.TestCase):
    def test_duplicate_directed_edge_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                load_graph(Path(temporary), {"symmetrize_edges": "false"}, graph_root(Path(temporary), "0 1\n0 1\n"), [])

    def test_out_of_range_endpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                load_graph(Path(temporary), {"symmetrize_edges": "false"}, graph_root(Path(temporary), "0 2\n"), [])

    def test_graph_id_is_bound_to_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                load_graph(Path(temporary), {"symmetrize_edges": "false"}, graph_root(Path(temporary), "1 0 1\n"), [])

    def test_checkpoint_source_digest_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            payload = b"not-the-canonical-npz"
            (checkpoint / "tensors.npz").write_bytes(payload)
            manifest = {
                "manifest_kind": "ezkl_canonical_checkpoint_bundle_v1",
                "dataset": "cora",
                "input_preprocess_id": "row_normalize",
                "model_arch_id": "arch",
                "model_param_id": "params",
                "hidden_layers": [{"head_count": 8, "head_dim": 8}] * 2,
                "hidden_head_specs": [hidden_spec(str(i)) for i in range(16)],
                "output_head_specs": [{"head_index": 0}, {"head_index": 1}],
                "K_out": 2,
                "tensors_npz_sha256": "sha256:" + "0" * 64,
            }
            with self.assertRaises(ValueError):
                validate_checkpoint_manifest(manifest, checkpoint, "cora_full", "row_normalize")
            self.assertNotEqual(manifest["tensors_npz_sha256"], "sha256:" + hashlib.sha256(payload).hexdigest())

    def test_multiline_dump_and_nonzero_hidden_bias_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dump = Path(temporary) / "tensors.txt"
            dump.write_text("TENSOR h 1 1 1\n1.0\n", encoding="utf-8")
            tensors = read_tensors(dump)
            self.assertEqual(tensors["h"][1], [1.0])
            with self.assertRaises(ValueError):
                validate_hidden_bias_tensors(
                    {"hidden_head_specs": [hidden_spec()]},
                    {"hidden/layer0/head0/attn_src/bias": ((1,), [1.0]),
                     "hidden/layer0/head0/attn_dst/bias": ((1,), [0.0]),
                     "hidden/layer0/head0/output_bias": ((8,), [0.0] * 8)},
                )

    def test_external_public_y_and_proof_carried_replacement_are_cli_bound(self) -> None:
        cli = (Path(__file__).parents[1] / "src/bin/ezkl_formal.rs").read_text(encoding="utf-8")
        native = (Path(__file__).parents[1] / "src/native_gat.rs").read_text(encoding="utf-8")
        self.assertGreaterEqual(cli.count('required(options, "public-y")'), 3)
        self.assertIn("proof-carried instances differ from verifier-reconstructed public statement", native)
        self.assertIn("load_public_y_resource(&plan, &public_y_path)", native)

    def test_public_y_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            link = root / "public_y.json"
            target.write_text(json.dumps({"kind": "fixture"}), encoding="utf-8")
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                _is_private_regular_file(link)


if __name__ == "__main__":
    unittest.main()
