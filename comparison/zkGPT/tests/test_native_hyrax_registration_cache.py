from __future__ import annotations

import copy
import pathlib
import tempfile
import unittest

from zkgpt_gat.native_hyrax import NativeHyrax
from zkgpt_gat.transcript import Transcript


class NativeHyraxRegistrationCacheTests(unittest.TestCase):
    def test_linear_claims_hit_session_cache_without_rebuilding_msm(self) -> None:
        values = [3, 5, 7, 11]
        weights = [13, 17, 19, 23]
        with tempfile.TemporaryDirectory(prefix="zkgpt-registration-cache-") as tmp:
            root = pathlib.Path(tmp)
            hyrax = NativeHyrax(root)
            registration = hyrax.commit_vector("v", values, root)
            self.assertEqual(hyrax.registration_msm_calls, 1)
            hyrax.commit_linear_claim_registered(
                "claim/one", values, weights, registration, root,
                transcript=Transcript("cache/one"),
                linear_form={"kind": "fixture", "index": 1},
            )
            hyrax.commit_linear_claim_registered(
                "claim/two", values, weights, registration, root,
                transcript=Transcript("cache/two"),
                linear_form={"kind": "fixture", "index": 2},
            )
            self.assertEqual(hyrax.registration_msm_calls, 1)
            self.assertEqual(hyrax.registration_cache_size, 1)

    def test_registration_cache_binds_vector_label_parameters_and_commitment(self) -> None:
        values = [3, 5, 7, 11]
        weights = [1, 2, 3, 4]
        with tempfile.TemporaryDirectory(prefix="zkgpt-registration-cache-reject-") as tmp:
            root = pathlib.Path(tmp)
            hyrax = NativeHyrax(root)
            registration = hyrax.commit_vector("v", values, root)

            def claim(candidate_values, candidate_registration):
                return hyrax.commit_linear_claim_registered(
                    "claim/reject", candidate_values, weights,
                    candidate_registration, root,
                    transcript=Transcript("cache/reject"),
                    linear_form={"kind": "fixture"},
                )

            with self.assertRaises(ValueError):
                claim([3, 5, 7, 12], registration)  # same label, changed field

            wrong_label = copy.deepcopy(registration)
            wrong_label["label"] = "other"
            with self.assertRaises(ValueError):
                claim(values, wrong_label)

            wrong_params = copy.deepcopy(registration)
            wrong_params["parameter_id"] = "zkGPT-bn254-hyrax-generators-v2/N=8"
            with self.assertRaises(ValueError):
                claim(values, wrong_params)

            wrong_commitment = copy.deepcopy(registration)
            other = hyrax.commit_vector("other", [4, 6, 8, 10], root)
            wrong_commitment["commitment"] = other["commitment"]
            with self.assertRaises(ValueError):
                claim(values, wrong_commitment)

            # Rejections above must not trigger an authentication fallback MSM.
            self.assertEqual(hyrax.registration_msm_calls, 2)

    def test_persisted_registration_is_authenticated_once_then_cached(self) -> None:
        values = [2, 4, 8, 16]
        weights = [7, 3, 5, 9]
        with tempfile.TemporaryDirectory(prefix="zkgpt-registration-cache-register-") as tmp:
            root = pathlib.Path(tmp)
            setup = NativeHyrax(root)
            persisted = setup.commit_vector("v", values, root)
            session = NativeHyrax(root)
            session.register_vector("v", values, persisted, root)
            self.assertEqual(session.registration_msm_calls, 1)
            session.commit_linear_claim_registered(
                "claim", values, weights, persisted, root,
                transcript=Transcript("cache/register"),
                linear_form={"kind": "fixture"},
            )
            self.assertEqual(session.registration_msm_calls, 1)


if __name__ == "__main__":
    unittest.main()
