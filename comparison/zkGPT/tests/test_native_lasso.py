from __future__ import annotations

import copy
import pathlib
import tempfile
import unittest

from zkgpt_gat.native_hyrax import NativeHyrax
from zkgpt_gat.native_lasso import _layout, prove_lasso, verify_lasso
from zkgpt_gat.transcript import Transcript
from zkgpt_gat.util import digest_json


class NativeLassoTest(unittest.TestCase):
    def _statement(self):
        family = {
            "family": "toy.range",
            "kind": "range",
            "table_id": "T_range",
            "offset": 0,
            "real_count": 2,
            "padded_count": 2,
            "a_pad": "0",
            "Q_pad": {"zero_prefix": 2, "one_suffix": 0},
            "range_bits": 1,
            "limbs_per_value": 1,
            "signed_magnitude": False,
        }
        layout = {
            "N_in": 2,
            "segments": [{
                "name": "L_qry",
                "logical_length": 2,
                "arrays": [{"name": "toy.range", "offset": 0, "length": 2}],
                "lasso_families": [family],
            }],
        }
        layout["layout_digest"] = digest_json(layout)
        metadata = {
            "Layout_in": layout,
            "Layout_in_digest": layout["layout_digest"],
            "static_table_registry_digest": digest_json({"T_range": "toy"}),
        }
        return metadata, {"T_range": [0, 1]}, [0, 1]

    def test_padded_count_one_is_rejected_by_native_layout_and_verifier(self):
        metadata, tables, _vector = self._statement()
        family = metadata["Layout_in"]["segments"][0]["lasso_families"][0]
        family["padded_count"] = 1
        with self.assertRaises(ValueError):
            _layout(family, len(tables["T_range"]))
        self.assertFalse(
            verify_lasso(
                {}, metadata, tables, {},
                transcript=Transcript("native-lasso-test"), canonical_source_sink=[],
            )
        )

    def test_fresh_lc_memory_argument_and_tamper(self):
        metadata, tables, vector = self._statement()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            hyrax = NativeHyrax(root)
            registration = hyrax.commit_vector("com_L", vector, root)
            proof = prove_lasso(
                metadata, tables, hyrax, vector, registration,
                zk_root=root, work_dir=root / "prove", transcript=Transcript("native-lasso-test"),
            )
            public_registration = {key: value for key, value in registration.items() if key != "blind"}
            sources = []
            self.assertTrue(verify_lasso(
                proof, metadata, tables, public_registration,
                transcript=Transcript("native-lasso-test"), canonical_source_sink=sources,
            ))
            self.assertEqual(len(sources), 1)
            tampered = copy.deepcopy(proof)
            tampered["families"][0]["beta"] = str(int(tampered["families"][0]["beta"]) + 1)
            self.assertFalse(verify_lasso(
                tampered, metadata, tables, public_registration,
                transcript=Transcript("native-lasso-test"), canonical_source_sink=[],
            ))


if __name__ == "__main__":
    unittest.main()
