#!/usr/bin/env python3
"""Short fail-closed checks for the canonical checkpoint tensor encodings.

The fixture is a synthetic final-shape PPI bundle.  It does not train, load a
checkpoint, create an SRS, or execute a formal proof.  Each mutation changes
only one encoding/property so NPZ/TXT cross-checking cannot be replaced by a
shape-only or one-file fallback.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import contextmanager
from typing import Iterator

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from checkpoint_bundle import (  # noqa: E402
    archive_key,
    build_family_manifest,
    load_checkpoint_manifest,
    validate_tensor_encodings,
    write_text_tensor_dump,
)
from quantization_scale_chain import (  # noqa: E402
    canonical_input_preprocess_id,
    canonical_quantization_scale_chain,
)


def _tensor(shape: tuple[int, ...], seed: float) -> np.ndarray:
    count = int(np.prod(shape, dtype=np.int64))
    values = (np.arange(count, dtype=np.float32) + np.float32(seed)).reshape(shape)
    return np.asarray(values, dtype=np.float32)


@contextmanager
def canonical_fixture() -> Iterator[tuple[pathlib.Path, dict[str, object]]]:
    with tempfile.TemporaryDirectory(prefix="checkpoint_encodings_") as raw_root:
        root = pathlib.Path(raw_root)
        tensors: dict[str, np.ndarray] = {}
        hidden_layers: list[dict[str, int]] = []
        hidden_head_specs: list[dict[str, object]] = []
        hidden_profile = [
            {"head_count": 8, "head_dim": 8},
            {"head_count": 8, "head_dim": 8},
        ]

        for layer_index, input_dim in enumerate((50, 64)):
            hidden_layers.append(
                {
                    "layer_index": layer_index,
                    "input_dim": input_dim,
                    "head_count": 8,
                    "head_dim": 8,
                }
            )
            for local_head_index in range(8):
                prefix = f"hidden/layer{layer_index}/head{local_head_index}"
                seq_kernel = f"{prefix}/seq/kernel"
                attn_dst_kernel = f"{prefix}/attn_dst/kernel"
                attn_src_kernel = f"{prefix}/attn_src/kernel"
                tensors[seq_kernel] = _tensor(
                    (1, input_dim, 8),
                    1.0 + layer_index + local_head_index,
                )
                tensors[attn_dst_kernel] = _tensor(
                    (1, 8, 1),
                    101.0 + layer_index + local_head_index,
                )
                tensors[attn_src_kernel] = _tensor(
                    (1, 8, 1),
                    201.0 + layer_index + local_head_index,
                )
                hidden_head_specs.append(
                    {
                        "layer_index": layer_index,
                        "local_head_index": local_head_index,
                        "seq_kernel": seq_kernel,
                        "attn_dst_kernel": attn_dst_kernel,
                        "attn_src_kernel": attn_src_kernel,
                    }
                )

        output_head_specs: list[dict[str, object]] = []
        for head_index in range(2):
            prefix = f"output/head{head_index}"
            seq_kernel = f"{prefix}/seq/kernel"
            attn_dst_kernel = f"{prefix}/attn_dst/kernel"
            attn_src_kernel = f"{prefix}/attn_src/kernel"
            output_bias = f"{prefix}/output_bias"
            tensors[seq_kernel] = _tensor((1, 64, 121), 301.0 + head_index)
            tensors[attn_dst_kernel] = _tensor((1, 121, 1), 401.0 + head_index)
            tensors[attn_src_kernel] = _tensor((1, 121, 1), 501.0 + head_index)
            tensors[output_bias] = _tensor((121,), 601.0 + head_index)
            output_head_specs.append(
                {
                    "head_index": head_index,
                    "seq_kernel": seq_kernel,
                    "attn_dst_kernel": attn_dst_kernel,
                    "attn_src_kernel": attn_src_kernel,
                    "output_bias": output_bias,
                }
            )

        chain = canonical_quantization_scale_chain(
            "gat_l3_scale_chain", hidden_profile, 2
        ).canonical_serialization()
        manifest = build_family_manifest(
            tensors,
            "/tmp/canonical_checkpoint.pt",
            d_in_profile=[50, 64],
            hidden_layers=hidden_layers,
            hidden_head_specs=hidden_head_specs,
            output_head_specs=output_head_specs,
            num_classes=121,
            task_type="inductive_multi_graph_node_classification",
            report_unit="graph",
            quant_cfg_id="gat_l3_scale_chain",
            quantization_scale_chain=chain,
            static_table_id="static_tables_test",
            degree_bound_id="degree_bound_test",
            model_arch_id="gat_family_L3_hid8x8-8x8_kout2_c121",
            model_param_id="ppi_gat_seed11",
            dataset="ppi",
            input_preprocess_id=canonical_input_preprocess_id("ppi"),
        )
        np.savez_compressed(
            root / "tensors.npz",
            **{archive_key(name): tensor for name, tensor in tensors.items()},
        )
        write_text_tensor_dump(root / "tensors.txt", tensors)
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        loaded_manifest = load_checkpoint_manifest(root / "manifest.json")
        yield root, loaded_manifest


def _rewrite_npz(path: pathlib.Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)


def _first_archive_key(manifest: dict[str, object]) -> str:
    tensor_index = manifest["tensor_index"]
    assert isinstance(tensor_index, dict)
    name = sorted(tensor_index)[0]
    entry = tensor_index[name]
    assert isinstance(entry, dict)
    return str(entry["archive_key"])


class CheckpointBundleEncodingTests(unittest.TestCase):
    def test_canonical_fixture_passes_and_returns_logical_map(self) -> None:
        with canonical_fixture() as (root, manifest):
            tensor_map = validate_tensor_encodings(
                manifest,
                root / "tensors.npz",
                root / "tensors.txt",
            )
            self.assertEqual(len(tensor_map), 56)
            self.assertIn("hidden/layer0/head0/seq/kernel", tensor_map)

    def test_npz_only_value_tamper_fails_closed(self) -> None:
        with canonical_fixture() as (root, manifest):
            key = _first_archive_key(manifest)

            def mutate(arrays: dict[str, np.ndarray]) -> None:
                arrays[key].reshape(-1)[0] += np.float32(1.0)

            _rewrite_npz(root / "tensors.npz", mutate)
            with self.assertRaises(ValueError):
                validate_tensor_encodings(manifest, root / "tensors.npz", root / "tensors.txt")

    def test_txt_only_value_tamper_fails_closed(self) -> None:
        with canonical_fixture() as (root, manifest):
            tokens = (root / "tensors.txt").read_text(encoding="utf-8").split()
            cursor = tokens.index("TENSOR") + 1
            cursor += 1  # logical name
            rank = int(tokens[cursor])
            cursor += 1 + rank
            cursor += 1  # element count
            tokens[cursor] = "-123.25"
            (root / "tensors.txt").write_text(" ".join(tokens) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_tensor_encodings(manifest, root / "tensors.npz", root / "tensors.txt")

    def test_npz_key_mismatch_fails_closed(self) -> None:
        with canonical_fixture() as (root, manifest):
            key = _first_archive_key(manifest)

            def mutate(arrays: dict[str, np.ndarray]) -> None:
                arrays[key + "_tampered"] = arrays.pop(key)

            _rewrite_npz(root / "tensors.npz", mutate)
            with self.assertRaises(ValueError):
                validate_tensor_encodings(manifest, root / "tensors.npz", root / "tensors.txt")

    def test_npz_shape_mismatch_fails_closed(self) -> None:
        with canonical_fixture() as (root, manifest):
            key = _first_archive_key(manifest)

            def mutate(arrays: dict[str, np.ndarray]) -> None:
                arrays[key] = arrays[key].reshape(1, -1)

            _rewrite_npz(root / "tensors.npz", mutate)
            with self.assertRaises(ValueError):
                validate_tensor_encodings(manifest, root / "tensors.npz", root / "tensors.txt")

    def test_npz_dtype_mismatch_fails_closed(self) -> None:
        with canonical_fixture() as (root, manifest):
            key = _first_archive_key(manifest)

            def mutate(arrays: dict[str, np.ndarray]) -> None:
                arrays[key] = arrays[key].astype(np.float64)

            _rewrite_npz(root / "tensors.npz", mutate)
            with self.assertRaises(ValueError):
                validate_tensor_encodings(manifest, root / "tensors.npz", root / "tensors.txt")

    def test_manifest_archive_key_mismatch_fails_closed(self) -> None:
        with canonical_fixture() as (root, manifest):
            tensor_index = manifest["tensor_index"]
            assert isinstance(tensor_index, dict)
            name = sorted(tensor_index)[0]
            entry = tensor_index[name]
            assert isinstance(entry, dict)
            entry["archive_key"] = "not_the_canonical_encoding"
            with self.assertRaises(ValueError):
                validate_tensor_encodings(manifest, root / "tensors.npz", root / "tensors.txt")

    def test_nonfinite_npz_value_fails_closed(self) -> None:
        with canonical_fixture() as (root, manifest):
            key = _first_archive_key(manifest)

            def mutate(arrays: dict[str, np.ndarray]) -> None:
                arrays[key].reshape(-1)[0] = np.float32(np.inf)

            _rewrite_npz(root / "tensors.npz", mutate)
            with self.assertRaises(ValueError):
                validate_tensor_encodings(manifest, root / "tensors.npz", root / "tensors.txt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
