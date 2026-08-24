from __future__ import annotations

import pathlib
import tempfile
import unittest

from zkgpt_gat.native_gkr import LinearGateSpec, LinearTerm, identity_oracle, prove_linear_gate, verify_linear_gate
from zkgpt_gat.native_hyrax import NativeHyrax, hyrax_value_params
from zkgpt_gat.proof import SegmentCommitmentState, _open_terminal_segments, _verify_terminal_segments_and_combine
from zkgpt_gat.transcript import Transcript


class InputCombineTests(unittest.TestCase):
    def test_verified_claims_reduce_to_one_selector_point(self) -> None:
        n_in = 4
        public_x = [9, 0, 0, 0]
        w = [3, 0, 0, 0]
        aux = [5, 8, 0, 0]
        l_qry = [13, 0, 0, 0]
        layout = {
            "N_in": n_in,
            "segments": [{"name": "Aux", "arrays": [
                {"name": "a", "offset": 0, "length": 1, "shape": [1]},
                {"name": "b", "offset": 1, "length": 1, "shape": [1]},
            ]}],
        }
        spec = LinearGateSpec(
            "identity", identity_oracle("Aux", "b", 1),
            (LinearTerm(1, identity_oracle("Aux", "a", 1)),), (3,),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            hyrax = NativeHyrax(root)
            registrations = {
                "com_W": hyrax.commit_vector("com_W", w, root),
                "com_Aux": hyrax.commit_vector("com_Aux", aux, root),
                "com_L": hyrax.commit_vector("com_L", l_qry, root),
            }
            public_registrations = {
                key: {name: value for name, value in registration.items() if name != "blind"}
                for key, registration in registrations.items()
            }
            prover = Transcript("test/input-combine")
            verifier = Transcript("test/input-combine")
            proof = prove_linear_gate(
                spec, layout=layout, vectors={"Aux": aux}, registrations={"Aux": registrations["com_Aux"]},
                params=hyrax_value_params(n_in), hyrax=hyrax, work_dir=root,
                transcript=prover, context={"test": "combine"},
            )
            verified_sources = []
            self.assertTrue(verify_linear_gate(
                proof, spec, layout=layout, registrations={"Aux": public_registrations["com_Aux"]},
                params=hyrax_value_params(n_in), transcript=verifier,
                context={"test": "combine"}, source_sink=verified_sources,
            ))
            metadata = {"Layout_in": {"N_in": n_in}, "Layout_in_digest": "layout", "session_nonce": "nonce"}
            commitments = {"metadata_digest": "metadata", "items": public_registrations}
            state = SegmentCommitmentState(
                hyrax=hyrax, vectors={"com_W": w, "com_Aux": aux, "com_L": l_qry}, registrations=registrations,
            )
            openings, combine = _open_terminal_segments(
                metadata, commitments, state, public_x, prover, root,
            )
            result = _verify_terminal_segments_and_combine(
                metadata, commitments, openings, combine, public_x, verifier, verified_sources,
            )
            self.assertEqual(result["status"], "VERIFY_OK")
            self.assertEqual(prover.digest, verifier.digest)


if __name__ == "__main__":
    unittest.main()
