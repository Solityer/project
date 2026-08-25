from __future__ import annotations

import copy
import os
import pathlib
import tempfile
import unittest

from zkgpt_gat.constants import COMMITMENT_BACKEND_ID
from zkgpt_gat.registry import (
    canonical_registry_public_path,
    canonical_registry_secret_path,
    ensure_model_registry,
    validate_registry_pair,
)
from zkgpt_gat.util import read_json, write_json


class RegistryPairSecurityTests(unittest.TestCase):
    @staticmethod
    def _metadata() -> dict[str, object]:
        return {
            "model_arch_id": "tests-registry-pair-arch-v1",
            "model_param_id": "tests-registry-pair-model-v1",
            "quant_cfg_id": "exact_integer_gat_v1",
            "static_table_id": "tables:test-v1",
            "rounding_mode": "nearest_ties_away",
            "private_input_layout": "X_pub|W|Aux|L_qry",
            "segment_order": ["X_pub", "W", "Aux", "L_qry"],
            "padding_rule_id": "power_of_two_zero_extension_v1",
            "integer_bound_ledger_digest": "a" * 64,
            "Layout_in_digest": "b" * 64,
            "Layout_in": {"N_in": 2},
            "integer_bound_ledger": {
                "registered_model_bounds": {"hidden.l1.h0": {"B_W": 1}},
            },
            "commitment_backend_id": COMMITMENT_BACKEND_ID,
        }

    @classmethod
    def _make_pair(cls, root: pathlib.Path) -> tuple[dict[str, object], pathlib.Path, pathlib.Path]:
        metadata = cls._metadata()
        result = ensure_model_registry(root, metadata, [("com_W", [1, 2])])
        return metadata, result["public_path"], result["secret_path"]

    def test_canonical_pair_succeeds_and_legacy_error_path_is_unused(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-registry-pair-ok-") as name:
            root = pathlib.Path(name)
            metadata, public_path, secret_path = self._make_pair(root)
            pair = validate_registry_pair(
                root,
                str(metadata["model_param_id"]),
                expected_model_arch_id=str(metadata["model_arch_id"]),
                expected_commitment_backend=COMMITMENT_BACKEND_ID,
            )
            self.assertEqual(pair["public_path"], canonical_registry_public_path(root, str(metadata["model_param_id"])))
            self.assertEqual(pair["secret_path"], canonical_registry_secret_path(root, str(metadata["model_param_id"])))
            self.assertTrue(public_path.is_file())
            self.assertTrue(secret_path.is_file())
            self.assertFalse(public_path.parent.joinpath("registry.secret.json").exists())

    def test_missing_public_or_private_side_fails_closed(self) -> None:
        for missing in ("public", "private"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory(prefix="zkgpt-registry-pair-missing-") as name:
                root = pathlib.Path(name)
                metadata, public_path, secret_path = self._make_pair(root)
                (public_path if missing == "public" else secret_path).unlink()
                with self.assertRaises((FileNotFoundError, RuntimeError)):
                    validate_registry_pair(root, str(metadata["model_param_id"]))

    def test_symlink_and_hardlink_files_fail_closed(self) -> None:
        for link_kind in ("symlink", "hardlink"):
            with self.subTest(link_kind=link_kind), tempfile.TemporaryDirectory(prefix="zkgpt-registry-pair-link-") as name:
                root = pathlib.Path(name)
                metadata, public_path, secret_path = self._make_pair(root)
                target = public_path if link_kind == "symlink" else secret_path
                original = target.read_bytes()
                outside = root / f"{link_kind}-target.json"
                outside.write_bytes(original)
                target.unlink()
                if link_kind == "symlink":
                    target.symlink_to(outside)
                else:
                    os.link(outside, target)
                with self.assertRaisesRegex(RuntimeError, "(symlink|owned regular file)"):
                    validate_registry_pair(root, str(metadata["model_param_id"]))

    def test_symlinked_registry_directory_and_identity_escape_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-registry-pair-dir-") as name:
            root = pathlib.Path(name)
            metadata, public_path, _ = self._make_pair(root)
            public_dir = public_path.parent
            outside = root / "outside-public"
            outside.mkdir()
            saved = outside / "registry.json"
            saved.write_bytes(public_path.read_bytes())
            public_path.unlink()
            public_dir.rename(root / "public-real")
            public_dir.symlink_to(root / "public-real", target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                validate_registry_pair(root, str(metadata["model_param_id"]))
            with self.assertRaises((RuntimeError, ValueError)):
                validate_registry_pair(root, "../escape")

    def test_transaction_id_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-registry-pair-tx-") as name:
            root = pathlib.Path(name)
            metadata, _, secret_path = self._make_pair(root)
            secret = copy.deepcopy(read_json(secret_path))
            secret["registration_transaction_id"] = "f" * 64
            write_json(secret_path, secret)
            with self.assertRaisesRegex(RuntimeError, "transaction id"):
                validate_registry_pair(root, str(metadata["model_param_id"]))


if __name__ == "__main__":
    unittest.main()
