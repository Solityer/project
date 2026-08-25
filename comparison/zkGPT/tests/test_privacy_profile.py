from __future__ import annotations

import copy
import tempfile
import unittest
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from zkgpt_gat.constants import COMMITMENT_BACKEND_ID
from zkgpt_gat.native_hyrax import MASK_DOMAIN, MASK_PROTOCOL_ID, MASK_VERSION
from zkgpt_gat.privacy import (
    PROFILE_ID,
    PROFILE_STATUS,
    PrivacyProfileError,
    build_privacy_profile,
    build_profile_anchor,
    check_multi_session_profiles,
    fresh_session_nonce,
    verify_privacy_profile,
)
from zkgpt_gat.util import digest_json, read_json, write_json


def _registration(label: str, value: int) -> dict[str, object]:
    """Return the public portion of a native Hyrax registration."""

    return {
        "backend": COMMITMENT_BACKEND_ID,
        "parameter_id": "test-params",
        "vector_length": 4,
        "label": label,
        "commitment": {"x": str(value), "y": str(value + 1), "infinity": False},
    }


def _opening(value: int, transcript_digest: str, *, identity_mask: bool = False) -> dict[str, object]:
    """A serialized native masked opening with no private blinder/scalar."""

    return {
        "evaluation_commitment": {"x": str(value), "y": str(value + 1), "infinity": False},
        "ipa": {
            "rounds": [{"L": {"x": str(value + 10), "y": str(value + 11), "infinity": False}}],
            "final_a": str(value + 12),
        },
        "mask_protocol_id": MASK_PROTOCOL_ID,
        "mask_domain": MASK_DOMAIN,
        "mask_version": MASK_VERSION,
        "mask_commitment": {
            "x": str(value + 2),
            "y": str(value + 3),
            "infinity": identity_mask,
        },
        "mask_challenge": str(value + 4),
        "transcript_digest": transcript_digest,
    }


def _lasso(index: int, transcript_digest: str) -> dict[str, object]:
    """Small proof-shaped Lasso object containing LC_f and GP_f registrations."""

    family = "T_range"
    return {
        "protocol": "zkGPT/native-lasso-memory-v1",
        "transcript_digest": transcript_digest,
        "lc_header": [{
            "index": 0,
            "family": family,
            "registration": _registration(f"LC_f/0/{family}", 100 + index),
        }],
        "families": [{
            "index": 0,
            "family": family,
            "gp_registration": _registration(f"GP_f/0/{family}", 110 + index),
            "query_link": {
                "lc_claim": _opening(120 + index, transcript_digest),
                "canonical_claim": _opening(130 + index, transcript_digest),
            },
        }],
    }


def _components(
    index: int,
    *,
    dynamic_index: int | None = None,
    transcript_index: int | None = None,
    identity_mask: bool = False,
) -> dict[str, dict[str, Any]]:
    dynamic_index = index if dynamic_index is None else dynamic_index
    transcript_index = index if transcript_index is None else transcript_index
    digest = f"{transcript_index + 1:064x}"
    components: dict[str, dict[str, Any]] = {
        "lasso": _lasso(index, digest),
        "matmul": {
            "active": _opening(200 + index, digest, identity_mask=identity_mask),
            "linear_equality": {"commitment": f"matmul-{index}"},
            "transcript_digest": digest,
        },
        "gkr": {
            "active": _opening(300 + index, digest, identity_mask=identity_mask),
            "terminal_product": {"commitment": f"gkr-{index}"},
            "transcript_digest": digest,
        },
        "input_combine": {
            "active": _opening(400 + index, digest, identity_mask=identity_mask),
            "same_value": {"commitment": f"combine-{index}"},
            "transcript_digest": digest,
        },
        "terminal_openings": {
            "items": {
                "com_W": {
                    "registration": _registration("com_W", 11),
                    "proof": _opening(500 + index, digest, identity_mask=identity_mask),
                },
                "com_Aux": {
                    "registration": _registration("com_Aux", 20 + dynamic_index),
                    "proof": _opening(510 + index, digest, identity_mask=identity_mask),
                },
                "com_L": {
                    "registration": _registration("com_L", 30 + dynamic_index),
                    "proof": _opening(520 + index, digest, identity_mask=identity_mask),
                },
            },
            "transcript_digest": digest,
        },
    }
    return components


def _session(
    index: int,
    *,
    model_commitment_value: int = 11,
    dynamic_index: int | None = None,
    mask_index: int | None = None,
    transcript_index: int | None = None,
    identity_mask: bool = False,
) -> dict[str, Any]:
    """Build an authenticated in-memory artifact bundle, never production files."""

    dynamic_index = index if dynamic_index is None else dynamic_index
    mask_index = index if mask_index is None else mask_index
    components = _components(
        mask_index,
        dynamic_index=dynamic_index,
        transcript_index=transcript_index,
        identity_mask=identity_mask,
    )
    metadata: dict[str, Any] = {
        "privacy_extension_profile_id": PROFILE_ID,
        "privacy_profile_status": PROFILE_STATUS,
        "session_nonce": fresh_session_nonce(),
    }
    commitments: dict[str, Any] = {
        "items": {
            # com_W is intentionally shared by sessions in the positive
            # multi-session case; Aux and L are per-proof commitments.
            "com_W": _registration("com_W", model_commitment_value),
            "com_Aux": _registration("com_Aux", 20 + dynamic_index),
            "com_L": _registration("com_L", 30 + dynamic_index),
        }
    }
    metadata["privacy_profile_anchor"] = build_profile_anchor(metadata, commitments)
    pre_digest = f"{index + 1:064x}"
    profile = build_privacy_profile(
        metadata,
        commitments,
        lasso=components["lasso"],
        matrix=components["matmul"],
        gkr=components["gkr"],
        input_combine=components["input_combine"],
        terminal_openings=components["terminal_openings"],
        pre_profile_transcript_digest=pre_digest,
    )
    proof_payload: dict[str, Any] = {
        "proof_type": "pi_GAT_zkGPT",
        "M_pub_digest": digest_json(metadata),
        "Y_digest": digest_json({"fixture": index}),
        "Com_in": [commitments["items"][name] for name in ("com_W", "com_Aux", "com_L")],
        "Pi_Lasso": components["lasso"],
        "Pi_MatMul": components["matmul"],
        "Pi_GKR": components["gkr"],
        "Pi_combine": components["input_combine"],
        "Open_in": components["terminal_openings"],
        "PrivacyProfile": profile,
        "privacy_profile_digest": profile["profile_digest"],
    }
    transcript = {
        "protocol": "zkGPT/GAT/formal-proof-session/v1",
        "session_nonce": metadata["session_nonce"],
        "transcript_domain": profile["transcript_domain"],
        "pre_profile_transcript_digest": pre_digest,
        "privacy_profile_digest": profile["profile_digest"],
    }
    transcript["final_digest"] = digest_json(transcript)
    proof_payload["transcript_digest"] = digest_json(transcript)
    required_context = {
        "manifest": {
            "status": "VERIFY_OK",
            "privacy_profile_digest": profile["profile_digest"],
        },
        "verification": {"status": "VERIFY_OK"},
        "independent_verification": {
            "status": "VERIFY_OK",
            "privacy_masking_verified": True,
            "privacy_profile_digest": profile["profile_digest"],
        },
    }
    return {
        "metadata": metadata,
        "commitments": commitments,
        "proof_payload": proof_payload,
        "transcript": transcript,
        "required_context": required_context,
        "pre_profile_transcript_digest": pre_digest,
    }


def _reverse_mapping_order(value: Any) -> Any:
    """Rebuild a proof object with every mapping insertion order reversed."""

    if isinstance(value, dict):
        return {
            key: _reverse_mapping_order(child)
            for key, child in reversed(list(value.items()))
        }
    if isinstance(value, list):
        return [_reverse_mapping_order(child) for child in value]
    return value


class _DuplicateKeyMapping(Mapping[str, Any]):
    """Mapping fixture exposing duplicate entries through ``items()``."""

    def __init__(self, items: list[tuple[str, Any]]) -> None:
        self._items = items

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)


class PrivacyProfileTests(unittest.TestCase):
    def test_real_artifact_bundle_positive_and_single_member(self) -> None:
        first = _session(0)
        second = _session(1)
        single = check_multi_session_profiles([first])
        self.assertEqual(single["status"], "VERIFY_OK")
        self.assertEqual(single["session_count"], 1)
        result = check_multi_session_profiles([first, second])
        self.assertEqual(result["status"], "VERIFY_OK")
        self.assertEqual(result["session_count"], 2)
        self.assertEqual(result["unique_dynamic_segment_commitments"], 4)

    def test_anchor_profile_and_evidence_digests_are_authenticated(self) -> None:
        bundle = _session(0)
        profile = bundle["proof_payload"]["PrivacyProfile"]
        anchor = bundle["metadata"]["privacy_profile_anchor"]
        self.assertEqual(profile["anchor_digest"], anchor["anchor_digest"])
        self.assertEqual(profile["evidence_digest"], digest_json(profile["evidence"]))
        self.assertEqual(
            profile["profile_digest"],
            digest_json({key: value for key, value in profile.items() if key != "profile_digest"}),
        )
        result = verify_privacy_profile(
            bundle["metadata"],
            bundle["commitments"],
            bundle["proof_payload"],
            pre_profile_transcript_digest=bundle["pre_profile_transcript_digest"],
        )
        self.assertEqual(result["status"], "VERIFY_OK")
        self.assertEqual(result["profile_digest"], profile["profile_digest"])
        self.assertEqual(result["evidence_digest"], profile["evidence_digest"])

    def test_nested_mapping_insertion_order_is_not_protocol_order(self) -> None:
        bundle = _session(0)
        reordered = copy.deepcopy(bundle)
        proof = reordered["proof_payload"]
        for key in ("Pi_Lasso", "Pi_MatMul", "Pi_GKR", "Pi_combine", "Open_in"):
            proof[key] = _reverse_mapping_order(proof[key])

        original_profile = bundle["proof_payload"]["PrivacyProfile"]
        rebuilt_profile = build_privacy_profile(
            reordered["metadata"],
            reordered["commitments"],
            lasso=proof["Pi_Lasso"],
            matrix=proof["Pi_MatMul"],
            gkr=proof["Pi_GKR"],
            input_combine=proof["Pi_combine"],
            terminal_openings=proof["Open_in"],
            pre_profile_transcript_digest=reordered["pre_profile_transcript_digest"],
        )
        self.assertEqual(rebuilt_profile["evidence"], original_profile["evidence"])
        self.assertEqual(rebuilt_profile["evidence_digest"], original_profile["evidence_digest"])
        self.assertEqual(rebuilt_profile["profile_digest"], original_profile["profile_digest"])

        terminal_registrations = [
            item["active_label"]
            for item in rebuilt_profile["evidence"]
            if item["component"] == "terminal_openings" and item["role"] == "registration_commitment"
        ]
        self.assertEqual(terminal_registrations, ["com_Aux", "com_L", "com_W"])
        lasso_registrations = [
            item["active_label"]
            for item in rebuilt_profile["evidence"]
            if item["component"] == "lasso" and item["role"] == "registration_commitment"
        ]
        self.assertEqual(lasso_registrations, ["GP_f/0/T_range", "LC_f/0/T_range"])

        check = verify_privacy_profile(
            reordered["metadata"],
            reordered["commitments"],
            proof,
            pre_profile_transcript_digest=reordered["pre_profile_transcript_digest"],
        )
        self.assertEqual(check["status"], "VERIFY_OK")

    def test_sorted_json_round_trip_preserves_privacy_profile(self) -> None:
        bundle = _session(0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("metadata", "commitments", "proof_payload"):
                write_json(root / f"{name}.json", bundle[name])
            metadata = read_json(root / "metadata.json")
            commitments = read_json(root / "commitments.json")
            proof_payload = read_json(root / "proof_payload.json")
            check = verify_privacy_profile(
                metadata,
                commitments,
                proof_payload,
                pre_profile_transcript_digest=bundle["pre_profile_transcript_digest"],
            )
        self.assertEqual(check["status"], "VERIFY_OK")
        self.assertEqual(check["profile_digest"], bundle["proof_payload"]["PrivacyProfile"]["profile_digest"])

    def test_non_string_and_duplicate_mapping_keys_fail_closed(self) -> None:
        bundle = _session(0)
        bad_key = copy.deepcopy(bundle["proof_payload"]["Pi_GKR"])
        bad_key["nested"] = {0: "not a protocol key"}
        with self.assertRaises(PrivacyProfileError):
            build_privacy_profile(
                bundle["metadata"], bundle["commitments"],
                lasso=bundle["proof_payload"]["Pi_Lasso"],
                matrix=bundle["proof_payload"]["Pi_MatMul"],
                gkr=bad_key,
                input_combine=bundle["proof_payload"]["Pi_combine"],
                terminal_openings=bundle["proof_payload"]["Open_in"],
                pre_profile_transcript_digest=bundle["pre_profile_transcript_digest"],
            )

        duplicate_key = copy.deepcopy(bundle["proof_payload"]["Pi_GKR"])
        duplicate_key["nested"] = _DuplicateKeyMapping([("same", 1), ("same", 2)])
        with self.assertRaises(PrivacyProfileError):
            build_privacy_profile(
                bundle["metadata"], bundle["commitments"],
                lasso=bundle["proof_payload"]["Pi_Lasso"],
                matrix=bundle["proof_payload"]["Pi_MatMul"],
                gkr=duplicate_key,
                input_combine=bundle["proof_payload"]["Pi_combine"],
                terminal_openings=bundle["proof_payload"]["Open_in"],
                pre_profile_transcript_digest=bundle["pre_profile_transcript_digest"],
            )

    def test_identity_mask_and_missing_registration_fail_closed(self) -> None:
        with self.assertRaises(PrivacyProfileError):
            _session(0, identity_mask=True)
        bundle = _session(0)
        bundle["proof_payload"]["Open_in"]["items"].pop("com_Aux")
        bundle["proof_payload"]["Open_in"]["items"].pop("com_L")
        result = verify_privacy_profile(
            bundle["metadata"],
            bundle["commitments"],
            bundle["proof_payload"],
            pre_profile_transcript_digest=bundle["pre_profile_transcript_digest"],
        )
        self.assertEqual(result["status"], "VERIFY_FAILED")

    def test_missing_and_tampered_payload_fields_fail_closed(self) -> None:
        bundle = _session(0)
        missing = copy.deepcopy(bundle)
        missing["proof_payload"].pop("Pi_GKR")
        result = verify_privacy_profile(
            missing["metadata"], missing["commitments"], missing["proof_payload"],
            pre_profile_transcript_digest=missing["pre_profile_transcript_digest"],
        )
        self.assertEqual(result["status"], "VERIFY_FAILED")

        tampered = copy.deepcopy(bundle)
        tampered["metadata"]["privacy_profile_anchor"]["anchor_digest"] = "0" * 64
        result = verify_privacy_profile(
            tampered["metadata"], tampered["commitments"], tampered["proof_payload"],
            pre_profile_transcript_digest=tampered["pre_profile_transcript_digest"],
        )
        self.assertEqual(result["status"], "VERIFY_FAILED")

        tampered = copy.deepcopy(bundle)
        tampered["proof_payload"]["PrivacyProfile"]["evidence"][0]["payload_digest"] = "f" * 64
        result = verify_privacy_profile(
            tampered["metadata"], tampered["commitments"], tampered["proof_payload"],
            pre_profile_transcript_digest=tampered["pre_profile_transcript_digest"],
        )
        self.assertEqual(result["status"], "VERIFY_FAILED")

    def test_nonce_mask_dynamic_and_transcript_reuse_fail_closed(self) -> None:
        first = _session(0)
        reused_nonce = check_multi_session_profiles([first, copy.deepcopy(first)])
        self.assertEqual(reused_nonce["status"], "VERIFY_FAILED")
        self.assertIn("nonce", reused_nonce["reason"])

        reused_mask = check_multi_session_profiles([first, _session(1, mask_index=0)])
        self.assertEqual(reused_mask["status"], "VERIFY_FAILED")
        self.assertIn("mask", reused_mask["reason"])

        reused_dynamic = check_multi_session_profiles([first, _session(1, dynamic_index=0)])
        self.assertEqual(reused_dynamic["status"], "VERIFY_FAILED")
        self.assertIn("dynamic", reused_dynamic["reason"])

        reused_transcript = check_multi_session_profiles([first, _session(1, transcript_index=0)])
        self.assertEqual(reused_transcript["status"], "VERIFY_FAILED")
        self.assertIn("transcript", reused_transcript["reason"])

    def test_unmasked_opening_and_private_blinder_fail_closed(self) -> None:
        bundle = _session(0)
        components = copy.deepcopy(bundle["proof_payload"]["Pi_GKR"])
        for field in ("mask_protocol_id", "mask_domain", "mask_version", "mask_commitment", "mask_challenge"):
            components["active"].pop(field)
        with self.assertRaises(PrivacyProfileError):
            build_privacy_profile(
                bundle["metadata"], bundle["commitments"],
                lasso=bundle["proof_payload"]["Pi_Lasso"],
                matrix=bundle["proof_payload"]["Pi_MatMul"],
                gkr=components,
                input_combine=bundle["proof_payload"]["Pi_combine"],
                terminal_openings=bundle["proof_payload"]["Open_in"],
                pre_profile_transcript_digest=bundle["pre_profile_transcript_digest"],
            )

        private_registration = copy.deepcopy(bundle["commitments"]["items"]["com_Aux"])
        private_registration["blind"] = "must-not-serialize"
        bad_commitments = copy.deepcopy(bundle["commitments"])
        bad_commitments["items"]["com_Aux"] = private_registration
        with self.assertRaises(PrivacyProfileError):
            build_profile_anchor(bundle["metadata"], bad_commitments)


if __name__ == "__main__":
    unittest.main()
