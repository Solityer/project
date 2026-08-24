from __future__ import annotations

import pathlib
import tempfile
import unittest

from zkgpt_gat.native_gkr import (
    LinearGateSpec, LinearTerm, OracleRef, ProductGateSpec, identity_oracle,
    prove_linear_gate, prove_product_gate, verify_linear_gate, verify_product_gate,
)
from zkgpt_gat.native_hyrax import NativeHyrax, hyrax_value_params
from zkgpt_gat.transcript import Transcript


class NativeGKRTests(unittest.TestCase):
    def test_linear_gate_uses_registered_aux_arrays(self) -> None:
        n_in = 8
        a = [2, 3, 5]
        b = [7, 11, 13]
        c = [a[index] + 2 * b[index] + 4 for index in range(3)]
        vector = a + b + c + [0] * (n_in - len(a) - len(b) - len(c))
        # Use N=16 because the three arrays require nine coordinates.
        n_in = 16
        vector += [0] * (n_in - len(vector))
        layout = {
            "N_in": n_in,
            "segments": [{"name": "Aux", "arrays": [
                {"name": "a", "offset": 0, "length": 3, "shape": [3]},
                {"name": "b", "offset": 3, "length": 3, "shape": [3]},
                {"name": "c", "offset": 6, "length": 3, "shape": [3]},
            ]}],
        }
        spec = LinearGateSpec(
            "linear",
            identity_oracle("Aux", "c", 3),
            (
                LinearTerm(1, identity_oracle("Aux", "a", 3)),
                LinearTerm(2, identity_oracle("Aux", "b", 3)),
            ),
            (4, 4, 4),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            hyrax = NativeHyrax(root)
            registration = hyrax.commit_vector("Aux", vector, root)
            public_registration = {key: value for key, value in registration.items() if key != "blind"}
            prover = Transcript("test/native-gkr")
            verifier = Transcript("test/native-gkr")
            proof = prove_linear_gate(
                spec, layout=layout, vectors={"Aux": vector}, registrations={"Aux": registration},
                params=hyrax_value_params(n_in), hyrax=hyrax, work_dir=root,
                transcript=prover, context={"test": "linear"},
            )
            self.assertTrue(verify_linear_gate(
                proof, spec, layout=layout, registrations={"Aux": public_registration},
                params=hyrax_value_params(n_in), transcript=verifier,
                context={"test": "linear"},
            ))
            self.assertEqual(prover.digest, verifier.digest)
            bad_spec = LinearGateSpec(
                "linear", identity_oracle("Aux", "c", 3),
                (LinearTerm(1, identity_oracle("Aux", "a", 3)), LinearTerm(3, identity_oracle("Aux", "b", 3))),
                (4, 4, 4),
            )
            self.assertFalse(verify_linear_gate(
                proof, bad_spec, layout=layout, registrations={"Aux": public_registration},
                params=hyrax_value_params(n_in), transcript=Transcript("test/native-gkr"),
                context={"test": "linear"},
            ))

    def test_constant_gate_has_no_prover_selected_term(self) -> None:
        n_in = 4
        vector = [1, 1, 1, 0]
        layout = {"N_in": n_in, "segments": [{"name": "Aux", "arrays": [
            {"name": "ones", "offset": 0, "length": 3, "shape": [3]},
        ]}]}
        spec = LinearGateSpec(
            "constant", identity_oracle("Aux", "ones", 3), (), (1, 1, 1),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            hyrax = NativeHyrax(root)
            registration = hyrax.commit_vector("Aux", vector, root)
            public_registration = {key: value for key, value in registration.items() if key != "blind"}
            prover = Transcript("test/native-gkr-constant")
            verifier = Transcript("test/native-gkr-constant")
            proof = prove_linear_gate(
                spec, layout=layout, vectors={"Aux": vector}, registrations={"Aux": registration},
                params=hyrax_value_params(n_in), hyrax=hyrax, work_dir=root,
                transcript=prover, context={"test": "constant"},
            )
            self.assertTrue(verify_linear_gate(
                proof, spec, layout=layout, registrations={"Aux": public_registration},
                params=hyrax_value_params(n_in), transcript=verifier,
                context={"test": "constant"},
            ))
            self.assertEqual(prover.digest, verifier.digest)

    def test_product_gate_binds_registered_operands(self) -> None:
        n_in = 16
        a = [2, 3, 5]
        b = [7, 11, 13]
        c = [left * right for left, right in zip(a, b)]
        vector = a + b + c + [0] * (n_in - 9)
        layout = {
            "N_in": n_in,
            "segments": [{"name": "Aux", "arrays": [
                {"name": "a", "offset": 0, "length": 3, "shape": [3]},
                {"name": "b", "offset": 3, "length": 3, "shape": [3]},
                {"name": "c", "offset": 6, "length": 3, "shape": [3]},
            ]}],
        }
        spec = ProductGateSpec(
            "product", identity_oracle("Aux", "c", 3),
            identity_oracle("Aux", "a", 3), identity_oracle("Aux", "b", 3),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            hyrax = NativeHyrax(root)
            registration = hyrax.commit_vector("Aux", vector, root)
            public_registration = {key: value for key, value in registration.items() if key != "blind"}
            prover = Transcript("test/native-gkr-product")
            verifier = Transcript("test/native-gkr-product")
            proof = prove_product_gate(
                spec, layout=layout, vectors={"Aux": vector}, registrations={"Aux": registration},
                params=hyrax_value_params(n_in), hyrax=hyrax, work_dir=root,
                transcript=prover, context={"test": "product"},
            )
            self.assertTrue(verify_product_gate(
                proof, spec, layout=layout, registrations={"Aux": public_registration},
                params=hyrax_value_params(n_in), transcript=verifier,
                context={"test": "product"},
            ))
            self.assertEqual(prover.digest, verifier.digest)

    def test_public_zero_sentinel_is_not_python_negative_index(self) -> None:
        n_in = 8
        source = [5, 7, 9]
        target = [0, 5, 7]
        vector = source + target + [0] * 2
        layout = {"N_in": n_in, "segments": [{"name": "Aux", "arrays": [
            {"name": "source", "offset": 0, "length": 3, "shape": [3]},
            {"name": "target", "offset": 3, "length": 3, "shape": [3]},
        ]}]}
        spec = LinearGateSpec(
            "shift", identity_oracle("Aux", "target", 3),
            (LinearTerm(1, OracleRef("Aux", "source", (-1, 0, 1))),),
            (0, 0, 0),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            hyrax = NativeHyrax(root)
            registration = hyrax.commit_vector("Aux", vector, root)
            public_registration = {key: value for key, value in registration.items() if key != "blind"}
            proof = prove_linear_gate(
                spec, layout=layout, vectors={"Aux": vector}, registrations={"Aux": registration},
                params=hyrax_value_params(n_in), hyrax=hyrax, work_dir=root,
                transcript=Transcript("test/native-gkr-zero"), context={"test": "zero"},
            )
            self.assertTrue(verify_linear_gate(
                proof, spec, layout=layout, registrations={"Aux": public_registration},
                params=hyrax_value_params(n_in), transcript=Transcript("test/native-gkr-zero"),
                context={"test": "zero"},
            ))
            bad = LinearGateSpec(
                "shift", identity_oracle("Aux", "target", 3),
                (LinearTerm(1, OracleRef("Aux", "source", (-2, 0, 1))),), (0, 0, 0),
            )
            self.assertFalse(verify_linear_gate(
                proof, bad, layout=layout, registrations={"Aux": public_registration},
                params=hyrax_value_params(n_in), transcript=Transcript("test/native-gkr-zero"),
                context={"test": "zero"},
            ))


if __name__ == "__main__":
    unittest.main()
