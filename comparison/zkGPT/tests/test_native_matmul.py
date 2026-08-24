from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np

from zkgpt_gat.native_hyrax import NativeHyrax, hyrax_value_params
from zkgpt_gat.native_matmul import MatMulSpec, TensorSource, prove_matmul, verify_matmul
from zkgpt_gat.transcript import Transcript


class NativeMatMulTests(unittest.TestCase):
    def test_matmul_is_bound_to_registered_segments(self) -> None:
        left = np.asarray([[1, 2], [3, 4]], dtype=object)
        right = np.asarray([[2, 1], [1, 3]], dtype=object)
        output = left @ right
        residual = np.zeros_like(output)
        n_in = 8
        w_vector = list(right.ravel()) + [0] * 4
        aux_vector = list(output.ravel()) + list(residual.ravel())
        layout = {
            "N_in": n_in,
            "segments": [
                {"name": "W", "arrays": [{"name": "B", "offset": 0, "length": 4, "shape": [2, 2]}]},
                {"name": "Aux", "arrays": [
                    {"name": "C", "offset": 0, "length": 4, "shape": [2, 2]},
                    {"name": "R", "offset": 4, "length": 4, "shape": [2, 2]},
                ]},
            ],
        }
        spec = MatMulSpec(
            "test.matmul", (2, 2, 2), TensorSource("X_pub", "A"),
            TensorSource("W", "B"), TensorSource("Aux", "C"),
            TensorSource("Aux", "R"), 1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            hyrax = NativeHyrax(root)
            registrations = {
                "W": hyrax.commit_vector("W", w_vector, root),
                "Aux": hyrax.commit_vector("Aux", aux_vector, root),
            }
            public_registrations = {
                name: {key: value for key, value in registration.items() if key != "blind"}
                for name, registration in registrations.items()
            }
            prover = Transcript("test/native-matmul")
            verifier = Transcript("test/native-matmul")
            proof = prove_matmul(
                spec, left, right, output, residual,
                layout=layout,
                public_arrays={"A": left},
                vectors={"W": w_vector, "Aux": aux_vector},
                registrations=registrations,
                params=hyrax_value_params(n_in),
                hyrax=hyrax,
                work_dir=root,
                transcript=prover,
                context={"test": "native-matmul"},
            )
            self.assertTrue(verify_matmul(
                proof, spec,
                layout=layout,
                public_arrays={"A": left},
                registrations=public_registrations,
                params=hyrax_value_params(n_in),
                transcript=verifier,
                context={"test": "native-matmul"},
            ))
            self.assertEqual(prover.digest, verifier.digest)
            tampered = dict(proof)
            tampered["row_point"] = list(proof["row_point"])
            tampered["row_point"][0] = str(int(tampered["row_point"][0]) + 1)
            self.assertFalse(verify_matmul(
                tampered, spec,
                layout=layout,
                public_arrays={"A": left},
                registrations=public_registrations,
                params=hyrax_value_params(n_in),
                transcript=Transcript("test/native-matmul"),
                context={"test": "native-matmul"},
            ))


if __name__ == "__main__":
    unittest.main()
