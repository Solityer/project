from __future__ import annotations

import copy
import pathlib
import random
import tempfile
import unittest

from zkgpt_gat.bn254 import BN254_FR_MODULUS, G1Point, generator_vector, multi_scalar_mul
from zkgpt_gat.constants import COMMITMENT_BACKEND_ID
from zkgpt_gat.registry import (
    canonical_registry_dir,
    canonical_registry_public_path,
    canonical_registry_secret_path,
    ensure_model_registry,
    registration_binding,
)
from zkgpt_gat.transcript import Transcript
from zkgpt_gat.value_commitment import PedersenParams
from zkgpt_gat.proof import _verify_registered_model_commitment
from zkgpt_gat.zk_sumcheck import (
    SumcheckVerificationError,
    bind_terminal_commitment,
    prove_committed_sumcheck,
    verify_committed_sumcheck,
)
from zkgpt_gat.util import digest_json, read_json, write_json


def _naive_msm(points: list[G1Point], scalars: list[int]) -> G1Point:
    result = G1Point.zero()
    for point, scalar in zip(points, scalars):
        result = result + int(scalar) * point
    return result


class NativeArithmeticRegressionTests(unittest.TestCase):
    def test_pippenger_matches_naive_randomized_small_vectors_and_boundaries(self) -> None:
        points = generator_vector("tests/msm/randomized", 16)
        points[0] = G1Point.zero()
        rng = random.Random(0x5A17)
        for trial in range(32):
            scalars = [rng.randrange(-(2 * BN254_FR_MODULUS), 2 * BN254_FR_MODULUS) for _ in points]
            # Explicitly exercise zero, identity, scalar modulo-r, and a
            # non-canonical negative scalar in every run.
            scalars[0] = 7
            scalars[1] = 0
            scalars[2] = BN254_FR_MODULUS
            scalars[3] = BN254_FR_MODULUS + 7
            scalars[4] = -BN254_FR_MODULUS - 11
            expected = _naive_msm(points, scalars)
            self.assertEqual(multi_scalar_mul(points, scalars), expected, trial)
        self.assertEqual(multi_scalar_mul([], []), G1Point.zero())
        self.assertEqual(multi_scalar_mul([G1Point.zero()], [1]), G1Point.zero())

    @staticmethod
    def _two_round_provider(round_index: int, claim: int, _transcript: Transcript) -> list[int]:
        return [3, 4, 5] if round_index == 0 else [0, claim, 0]

    def test_padding2_sumcheck_roundtrip_and_tamper(self) -> None:
        params = PedersenParams.derive("tests/padding2-sumcheck")
        prover = prove_committed_sumcheck(
            "tests/padding2-sumcheck",
            15,
            self._two_round_provider,
            2,
            variable_count=2,
            params=params,
            transcript=Transcript("tests/padding2-sumcheck-session"),
        )
        wire = prover.proof.to_dict()
        verifier = verify_committed_sumcheck(
            "tests/padding2-sumcheck",
            wire,
            15,
            2,
            2,
            params=params,
            transcript=Transcript("tests/padding2-sumcheck-session"),
        )
        self.assertEqual(verifier.challenges, prover.challenges)
        self.assertTrue(bind_terminal_commitment(verifier, prover.terminal_commitment).terminal_bound)

        tampered = copy.deepcopy(wire)
        tampered["rounds"][1]["challenge"] = tampered["rounds"][0]["challenge"]
        with self.assertRaises(SumcheckVerificationError):
            verify_committed_sumcheck(
                "tests/padding2-sumcheck",
                tampered,
                15,
                2,
                2,
                params=params,
                transcript=Transcript("tests/padding2-sumcheck-session"),
            )


class RegistrySchemaRegressionTests(unittest.TestCase):
    @staticmethod
    def _metadata() -> dict[str, object]:
        return {
            "model_arch_id": "tests-registry-arch-v1",
            "model_param_id": "tests-registry-model-v1",
            "quant_cfg_id": "exact_integer_gat_v1",
            "static_table_id": "tables:test-v1",
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

    def test_registry_public_schema_and_digest_are_canonical(self) -> None:
        metadata = self._metadata()
        arrays = [("com_W", [1, 2])]
        binding = registration_binding(metadata, arrays)
        self.assertEqual(binding["N_in"], 2)
        self.assertEqual(binding["arrays"], [{"name": "com_W", "length": 2}])
        self.assertEqual(binding["segment_order"], ["X_pub", "W", "Aux", "L_qry"])

        with tempfile.TemporaryDirectory(prefix="zkgpt-registry-schema-") as tmp:
            root = pathlib.Path(tmp)
            result = ensure_model_registry(root, metadata, arrays)
            public_path = canonical_registry_public_path(root, metadata["model_param_id"])
            self.assertEqual(result["public_path"], public_path)
            self.assertEqual(result["secret_path"], canonical_registry_secret_path(root, metadata["model_param_id"]))
            self.assertNotEqual(result["secret_path"].parent, public_path.parent)
            self.assertEqual(result["secret_path"].stat().st_mode & 0o777, 0o600)
            public = read_json(public_path)
            payload = dict(public)
            registry_digest = payload.pop("registry_digest")
            self.assertEqual(registry_digest, digest_json(payload))
            self.assertEqual(
                set(public),
                {
                    "protocol", "model_param_id", "model_arch_id", "registration_binding",
                    "registration_binding_digest", "registration_transaction_id", "commitment_backend",
                    "entries", "registry_digest",
                },
            )
            entry = public["entries"]["com_W"]
            self.assertEqual(
                set(entry),
                {"backend", "parameter_id", "label", "vector_length", "commitment", "padded_length"},
            )
            self.assertEqual(entry["padded_length"], entry["vector_length"])
            self.assertEqual(public["registration_binding_digest"], digest_json(public["registration_binding"]))

            # Existing registration is revalidated rather than silently
            # replaced, and the canonical path rejects traversal identities.
            second = ensure_model_registry(root, metadata, arrays)
            self.assertEqual(second["public_path"], public_path)
            with self.assertRaises(ValueError):
                canonical_registry_dir(root, "../escape")

    def test_existing_legacy_public_registry_fails_closed(self) -> None:
        metadata = self._metadata()
        arrays = [("com_W", [1, 2])]
        with tempfile.TemporaryDirectory(prefix="zkgpt-registry-legacy-") as tmp:
            root = pathlib.Path(tmp)
            result = ensure_model_registry(root, metadata, arrays)
            public_path = result["public_path"]
            legacy = copy.deepcopy(read_json(public_path))
            legacy["entries"]["com_W"].pop("padded_length")
            payload = dict(legacy)
            payload.pop("registry_digest")
            legacy["registry_digest"] = digest_json(payload)
            write_json(public_path, legacy)
            with self.assertRaises(RuntimeError):
                ensure_model_registry(root, metadata, arrays)

    def test_public_registry_rejects_nested_private_tensor_fields(self) -> None:
        metadata = self._metadata()
        arrays = [("com_W", [1, 2])]
        with tempfile.TemporaryDirectory(prefix="zkgpt-registry-private-field-") as tmp:
            root = pathlib.Path(tmp)
            result = ensure_model_registry(root, metadata, arrays)
            public_path = result["public_path"]
            public = copy.deepcopy(read_json(public_path))
            public["registration_binding"]["registered_model_bounds"]["tensor_values"] = [1, 2]
            payload = dict(public)
            payload.pop("registry_digest")
            public["registry_digest"] = digest_json(payload)
            write_json(public_path, public)
            with self.assertRaisesRegex(RuntimeError, "private tensor"):
                ensure_model_registry(root, metadata, arrays)

    def test_registered_commitment_rejects_missing_public_padding(self) -> None:
        metadata = self._metadata()
        arrays = [("com_W", [1, 2])]
        with tempfile.TemporaryDirectory(prefix="zkgpt-registry-verify-") as tmp:
            result = ensure_model_registry(pathlib.Path(tmp), metadata, arrays)
            public = result["public"]
            metadata["model_registry"] = {
                "registry_digest": public["registry_digest"],
                "registration_binding_digest": public["registration_binding_digest"],
                "public_descriptor": public,
            }
            entry = public["entries"]["com_W"]
            commitments = {
                "items": {
                    "com_W": {
                        "registered_model_registry_digest": public["registry_digest"],
                        "chunks": [{
                            "label": "com_W",
                            "registered_model_commitment": True,
                            "commitment": entry["commitment"],
                            "vector_length": 2,
                            "padded_length": 2,
                        }],
                    },
                },
            }
            self.assertTrue(_verify_registered_model_commitment(metadata, commitments))

            legacy_descriptor = copy.deepcopy(public)
            legacy_descriptor["entries"]["com_W"].pop("padded_length")
            legacy_payload = dict(legacy_descriptor)
            legacy_payload.pop("registry_digest")
            legacy_digest = digest_json(legacy_payload)
            legacy_descriptor["registry_digest"] = legacy_digest
            legacy_metadata = copy.deepcopy(metadata)
            legacy_metadata["model_registry"] = {
                "registry_digest": legacy_digest,
                "registration_binding_digest": legacy_descriptor["registration_binding_digest"],
                "public_descriptor": legacy_descriptor,
            }
            legacy_commitments = copy.deepcopy(commitments)
            legacy_commitments["items"]["com_W"]["registered_model_registry_digest"] = legacy_digest
            self.assertFalse(_verify_registered_model_commitment(legacy_metadata, legacy_commitments))


if __name__ == "__main__":
    unittest.main()
