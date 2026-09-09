"""Short unit tests for the standalone zkGPT cryptographic primitives."""
from __future__ import annotations

import copy
import unittest
from dataclasses import replace

from zkgpt_gat.field import BN254_FR_MODULUS
from zkgpt_gat.transcript import EncodingError, Transcript, decode_point, encode_point, encode_scalar
from zkgpt_gat.value_commitment import (
    CommitmentError,
    PedersenParams,
    commit_value,
    prove_cross_params_same_value,
    prove_multiplication,
    prove_representation,
    prove_same_value,
    verify_multiplication,
    verify_cross_params_same_value,
    verify_representation,
    verify_same_value,
)
from zkgpt_gat.zk_sumcheck import (
    SumcheckVerificationError,
    bind_terminal_claim,
    bind_terminal_commitment,
    prove_committed_sumcheck,
    verify_committed_sumcheck,
    verify_committed_sumcheck_bool,
)


class CryptoPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # This is an externally selected setup object shared by every family
        # in a session, not a per-label Sumcheck default.
        cls.params = PedersenParams.derive("tests/authenticated-params-v1")

    def test_canonical_transcript_order_and_encoding(self) -> None:
        first = Transcript("tests/transcript")
        first.absorb("left", {"x": 1, "y": [2, 3]})
        first.absorb("right", b"message")
        challenge_first = first.challenge_scalar("round")

        second = Transcript("tests/transcript")
        second.absorb("right", b"message")
        second.absorb("left", {"x": 1, "y": [2, 3]})
        challenge_second = second.challenge_scalar("round")
        self.assertNotEqual(challenge_first, challenge_second)

        scalar = 17
        self.assertEqual(len(encode_scalar(scalar)), 32)
        self.assertEqual(scalar, int.from_bytes(encode_scalar(scalar), "big"))
        self.assertEqual(encode_point(decode_point(encode_point(self.params.U))), encode_point(self.params.U))
        with self.assertRaises(EncodingError):
            encode_scalar(BN254_FR_MODULUS)

    def test_selected_params_are_authenticated(self) -> None:
        wire = self.params.to_dict()
        with self.assertRaises(CommitmentError):
            PedersenParams.from_dict(wire)
        self.assertIsInstance(PedersenParams.from_dict(wire, allow_derived=True), PedersenParams)
        expected = PedersenParams.from_dict(wire, expected=self.params)
        self.assertEqual(expected, self.params)
        tampered = dict(wire)
        tampered["U"] = tampered["H"]
        with self.assertRaises(CommitmentError):
            PedersenParams.from_dict(tampered)
        with self.assertRaises(CommitmentError):
            PedersenParams.from_dict(tampered, expected=self.params)

    def test_representation_and_same_value_proofs(self) -> None:
        first, first_blind = commit_value(self.params, 17, 101)
        second, second_blind = commit_value(self.params, 17, 103)
        representation = prove_representation(
            self.params,
            first,
            17,
            first_blind,
            label="tests/representation",
            transcript=Transcript("tests/representation-session"),
        )
        self.assertTrue(
            verify_representation(
                self.params,
                first,
                representation,
                label="tests/representation",
                transcript=Transcript("tests/representation-session"),
            )
        )
        self.assertTrue(
            verify_representation(
                self.params,
                first,
                representation.to_dict(),
                label="tests/representation",
                transcript=Transcript("tests/representation-session"),
            )
        )
        self.assertFalse(
            verify_representation(
                self.params,
                first,
                replace(representation, z_u=(representation.z_u + 1) % BN254_FR_MODULUS),
                label="tests/representation",
                transcript=Transcript("tests/representation-session"),
            )
        )

        same = prove_same_value(
            self.params,
            first,
            second,
            first_blind,
            second_blind,
            label="tests/same-value",
            transcript=Transcript("tests/same-value-session"),
        )
        self.assertTrue(
            verify_same_value(
                self.params,
                first,
                second,
                same,
                label="tests/same-value",
                transcript=Transcript("tests/same-value-session"),
            )
        )
        tampered = replace(same, z_h=(same.z_h + 1) % BN254_FR_MODULUS)
        self.assertFalse(
            verify_same_value(
                self.params,
                first,
                second,
                tampered,
                label="tests/same-value",
                transcript=Transcript("tests/same-value-session"),
            )
        )

    def test_cross_parameter_same_value(self) -> None:
        other = PedersenParams.derive("tests/authenticated-params-v2")
        first, first_blind = commit_value(self.params, 23, 107)
        second, second_blind = commit_value(other, 23, 109)
        prover = Transcript("tests/cross-params")
        verifier = Transcript("tests/cross-params")
        proof = prove_cross_params_same_value(
            self.params, first, other, second, 23, first_blind, second_blind,
            label="tests/cross-params/value", transcript=prover,
        )
        self.assertTrue(verify_cross_params_same_value(
            self.params, first, other, second, proof,
            label="tests/cross-params/value", transcript=verifier,
        ))
        self.assertEqual(prover.digest, verifier.digest)

    def test_multiplication_sigma_proof(self) -> None:
        a, ra = commit_value(self.params, 7, 11)
        b, rb = commit_value(self.params, 9, 13)
        product, rc = commit_value(self.params, 63, 17)
        proof = prove_multiplication(
            self.params,
            a,
            b,
            product,
            7,
            9,
            63,
            ra,
            rb,
            rc,
            label="tests/multiplication",
            transcript=Transcript("tests/multiplication-session"),
        )
        self.assertTrue(
            verify_multiplication(
                self.params,
                a,
                b,
                product,
                proof,
                label="tests/multiplication",
                transcript=Transcript("tests/multiplication-session"),
            )
        )
        tampered = replace(proof, z_product_h=(proof.z_product_h + 1) % BN254_FR_MODULUS)
        self.assertFalse(
            verify_multiplication(
                self.params,
                a,
                b,
                product,
                tampered,
                label="tests/multiplication",
                transcript=Transcript("tests/multiplication-session"),
            )
        )
        with self.assertRaises(CommitmentError):
            prove_multiplication(
                self.params,
                a,
                b,
                product,
                7,
                9,
                64,
                ra,
                rb,
                rc,
                label="tests/multiplication",
                transcript=Transcript("tests/multiplication-session"),
            )

    @staticmethod
    def _provider(round_index: int, claim: int, _transcript: Transcript):
        # First round: g(0)+g(1)=2*3+4+5=15.  Later rounds are selected
        # after the previous FS challenge and remain degree <= 2.
        return [3, 4, 5] if round_index == 0 else [0, claim, 0]

    def _prove_sumcheck(self, *, initial_claim=15, initial_value=None, initial_blind=None, label="tests/sumcheck"):
        prover_transcript = Transcript("tests/sumcheck-session")
        if initial_value is None:
            claim = initial_claim
        else:
            claim = initial_claim
        return prove_committed_sumcheck(
            label,
            claim,
            self._provider,
            2,
            variable_count=2,
            params=self.params,
            transcript=prover_transcript,
            initial_value=initial_value,
            initial_blind=initial_blind,
        )

    def test_sumcheck_states_terminal_binding_and_wire_privacy(self) -> None:
        prover_state = self._prove_sumcheck()
        wire = prover_state.proof.to_dict()
        self.assertFalse(hasattr(prover_state, "to_dict"))
        self.assertEqual(prover_state.terminal_value, prover_state.terminal_value % BN254_FR_MODULUS)

        verifier_state = verify_committed_sumcheck(
            "tests/sumcheck",
            wire,
            15,
            2,
            2,
            params=self.params,
            transcript=Transcript("tests/sumcheck-session"),
        )
        self.assertTrue(verifier_state.round_consistent)
        self.assertFalse(verifier_state.terminal_bound)
        self.assertEqual(verifier_state.challenges, prover_state.challenges)
        self.assertEqual(verifier_state.terminal_claim, prover_state.terminal_commitment)
        self.assertEqual(
            verifier_state.terminal_mask_commitment.point,
            prover_state.proof.terminal_mask_commitment,
        )
        self.assertNotEqual(
            prover_state.proof.terminal_claim_commitment,
            prover_state.proof.terminal_mask_commitment,
        )
        for round_message in wire["rounds"]:
            self.assertEqual(
                len(round_message["coefficient_commitments"]),
                len(round_message["coefficient_mask_commitments"]),
            )
            self.assertEqual(
                len(round_message["coefficient_poks"]),
                len(round_message["coefficient_mask_poks"]),
            )
            self.assertIn(round_message["mask_recurrence"], {"zero_sum", "challenge_recurrence"})
        self.assertTrue(bind_terminal_claim(
            verifier_state,
            lambda actual, _challenges: actual == prover_state.terminal_commitment,
        ).terminal_bound)
        self.assertEqual(bind_terminal_commitment(verifier_state, prover_state.terminal_commitment).terminal_bound, True)
        self.assertTrue(
            verify_committed_sumcheck_bool(
                "tests/sumcheck",
                wire,
                15,
                2,
                2,
                params=self.params,
                transcript=Transcript("tests/sumcheck-session"),
                terminal_predicate=lambda actual, expected: actual == prover_state.terminal_commitment,
            )
        )
        self.assertFalse(
            verify_committed_sumcheck_bool(
                "tests/sumcheck",
                wire,
                15,
                2,
                2,
                params=self.params,
                transcript=Transcript("tests/sumcheck-session"),
            )
        )
        self.assertFalse(
            verify_committed_sumcheck_bool(
                "tests/sumcheck",
                wire,
                15,
                2,
                2,
                params=self.params,
                transcript=Transcript("tests/sumcheck-session"),
                terminal_predicate=lambda _actual, _challenges: False,
            )
        )

        def assert_wire_clean(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    self.assertNotIn(key, {"value", "blind", "coefficients"})
                    assert_wire_clean(item)
            elif isinstance(value, list):
                for item in value:
                    assert_wire_clean(item)

        assert_wire_clean(wire)

        tampered_mask = copy.deepcopy(wire)
        tampered_mask["rounds"][0]["coefficient_mask_commitments"][0] = copy.deepcopy(
            tampered_mask["rounds"][0]["coefficient_commitments"][0]
        )
        with self.assertRaises(SumcheckVerificationError):
            verify_committed_sumcheck(
                "tests/sumcheck",
                tampered_mask,
                15,
                2,
                2,
                params=self.params,
                transcript=Transcript("tests/sumcheck-session"),
            )

    def test_sumcheck_witness_initial_and_external_statement(self) -> None:
        initial, initial_blind = commit_value(self.params, 15, 29)
        prover_transcript = Transcript("tests/witness-initial-session")
        prover_state = prove_committed_sumcheck(
            "tests/witness-initial",
            initial,
            self._provider,
            2,
            variable_count=2,
            params=self.params,
            transcript=prover_transcript,
            initial_value=15,
            initial_blind=initial_blind,
        )
        self.assertEqual(prover_state.proof.initial_claim_commitment, initial.point)
        verifier_state = verify_committed_sumcheck(
            "tests/witness-initial",
            prover_state.proof.to_dict(),
            initial,
            2,
            2,
            params=self.params,
            transcript=Transcript("tests/witness-initial-session"),
        )
        self.assertTrue(verifier_state.round_consistent)

        other_initial, _ = commit_value(self.params, 15, 31)
        with self.assertRaises(SumcheckVerificationError):
            verify_committed_sumcheck(
                "tests/witness-initial",
                prover_state.proof.to_dict(),
                other_initial,
                2,
                2,
                params=self.params,
                transcript=Transcript("tests/witness-initial-session"),
            )

    def test_sumcheck_session_chaining_and_negative_shapes(self) -> None:
        prover_transcript = Transcript("tests/chained-session")
        first = prove_committed_sumcheck(
            "tests/family-a", 15, self._provider, 2, variable_count=2,
            params=self.params, transcript=prover_transcript,
        )
        second = prove_committed_sumcheck(
            "tests/family-b", 15, self._provider, 2, variable_count=2,
            params=self.params, transcript=prover_transcript,
        )
        verifier_transcript = Transcript("tests/chained-session")
        first_state = verify_committed_sumcheck(
            "tests/family-a", first.proof.to_dict(), 15, 2, 2,
            params=self.params, transcript=verifier_transcript,
        )
        second_state = verify_committed_sumcheck(
            "tests/family-b", second.proof.to_dict(), 15, 2, 2,
            params=self.params, transcript=verifier_transcript,
        )
        self.assertEqual(first_state.challenges, first.challenges)
        self.assertEqual(second_state.challenges, second.challenges)
        with self.assertRaises(SumcheckVerificationError):
            verify_committed_sumcheck(
                "tests/family-b", second.proof.to_dict(), 15, 2, 2,
                params=self.params, transcript=Transcript("tests/chained-session"),
            )

        no_rounds = copy.deepcopy(first.proof.to_dict())
        no_rounds["rounds"] = []
        with self.assertRaises(SumcheckVerificationError):
            verify_committed_sumcheck(
                "tests/family-a", no_rounds, 15, 2, 2,
                params=self.params, transcript=Transcript("tests/chained-session"),
            )
        extra_round = copy.deepcopy(first.proof.to_dict())
        extra_round["rounds"].append(copy.deepcopy(extra_round["rounds"][0]))
        with self.assertRaises(SumcheckVerificationError):
            verify_committed_sumcheck(
                "tests/family-a", extra_round, 15, 2, 2,
                params=self.params, transcript=Transcript("tests/chained-session"),
            )
        reordered = copy.deepcopy(first.proof.to_dict())
        reordered["rounds"].reverse()
        with self.assertRaises(SumcheckVerificationError):
            verify_committed_sumcheck(
                "tests/family-a", reordered, 15, 2, 2,
                params=self.params, transcript=Transcript("tests/chained-session"),
            )
        challenge_tampered = copy.deepcopy(first.proof.to_dict())
        challenge_tampered["rounds"][0]["challenge"] = encode_scalar(1).hex()
        with self.assertRaises(SumcheckVerificationError):
            verify_committed_sumcheck(
                "tests/family-a", challenge_tampered, 15, 2, 2,
                params=self.params, transcript=Transcript("tests/chained-session"),
            )


if __name__ == "__main__":
    unittest.main()
