from __future__ import annotations

import copy
import pathlib
import tempfile
import unittest

from zkgpt_gat.native_hyrax import (
    MASK_DOMAIN,
    MASK_PROTOCOL_ID,
    MASK_VERSION,
    NativeHyrax,
    verify_linear_payload,
    verify_open_payload,
)
from zkgpt_gat.field import log2_power_of_two, next_power_of_two, to_field_int
from zkgpt_gat.transcript import Transcript


def _ephemeral_hyrax_opening(label: str, values: list[int], context: dict[str, object]) -> dict[str, object]:
    """Test-only adapter for the legacy-shaped ephemeral fixture envelope."""
    vector = [to_field_int(value) for value in values] or [0]
    padded_length = next_power_of_two(len(vector))
    vector.extend([0] * (padded_length - len(vector)))
    with tempfile.TemporaryDirectory(prefix="zkgpt-hyrax-test-") as tmp:
        root = pathlib.Path(tmp)
        hyrax = NativeHyrax(root)
        registration = hyrax.commit_vector(label, vector, root)
        public_registration = {key: value for key, value in registration.items() if key != "blind"}
        transcript = Transcript("zkGPT/GAT/ephemeral-Hyrax/v2")
        transcript.absorb("subprotocol-context-and-commitment", {"label": label, "context": context, "registration": public_registration})
        point = [transcript.challenge_scalar(f"{label}/point/{index}") for index in range(log2_power_of_two(padded_length))]
        state = hyrax.open_registered(label, vector, point, registration, root, transcript=transcript, public_context=context)
    return {
        "protocol": "zkGPT-ephemeral-Hyrax-opening-v3-masked",
        "label": label,
        "logical_length": len(values),
        "vector_length": padded_length,
        "registration": public_registration,
        "context": context,
        "point": [str(value) for value in point],
        "opening": state.proof,
        "mask_protocol_id": state.proof["mask_protocol_id"],
        "mask_domain": state.proof["mask_domain"],
        "mask_version": state.proof["mask_version"],
        "mask_commitment": state.proof["mask_commitment"],
        "mask_challenge": state.proof["mask_challenge"],
        "transcript_digest": transcript.digest.hex(),
    }


def _verify_ephemeral_hyrax_opening(label: str, payload: dict[str, object], context: dict[str, object]) -> bool:
    try:
        if payload.get("protocol") != "zkGPT-ephemeral-Hyrax-opening-v3-masked" or payload.get("label") != label or payload.get("context") != context:
            return False
        vector_length = int(payload["vector_length"])
        registration = payload["registration"]
        if not isinstance(registration, dict) or int(registration.get("vector_length", -1)) != vector_length:
            return False
        transcript = Transcript("zkGPT/GAT/ephemeral-Hyrax/v2")
        transcript.absorb("subprotocol-context-and-commitment", {"label": label, "context": context, "registration": registration})
        point = [transcript.challenge_scalar(f"{label}/point/{index}") for index in range(log2_power_of_two(vector_length))]
        if payload.get("point") != [str(value) for value in point]:
            return False
        opening = payload["opening"]
        if not isinstance(opening, dict):
            return False
        for field in ("mask_protocol_id", "mask_domain", "mask_version", "mask_commitment", "mask_challenge"):
            if payload.get(field) != opening.get(field):
                return False
        result = verify_open_payload(label, opening, point, registration, vector_length, transcript=transcript, public_context=context)
        return result.get("status") == "VERIFY_OK" and payload.get("transcript_digest") == transcript.digest.hex()
    except (KeyError, TypeError, ValueError):
        return False


def _public_registration(registration: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in registration.items() if key != "blind"}


def _walk_wire(value: object, path: str = "wire") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in {"blind", "mask_scalars", "mask_vector", "private_vector", "sigma", "s"}:
                found.append(child_path)
            found.extend(_walk_wire(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_wire(child, f"{path}[{index}]"))
    return found


class MaskedHyraxTests(unittest.TestCase):
    def test_registered_openings_are_fresh_and_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-masked-test-") as tmp:
            root = pathlib.Path(tmp)
            hyrax = NativeHyrax(root)
            values = [3, 5, 7, 11]
            point = [13, 17]
            registration = hyrax.commit_vector("v", values, root)
            first = hyrax.open_registered(
                "v", values, point, registration, root / "first",
                transcript=Transcript("tests/masked-registered"),
            )
            second = hyrax.open_registered(
                "v", values, point, registration, root / "second",
                transcript=Transcript("tests/masked-registered"),
            )
            self.assertNotEqual(first.proof["ipa"]["rounds"][0]["L"], second.proof["ipa"]["rounds"][0]["L"])
            self.assertNotEqual(first.proof["mask_commitment"], second.proof["mask_commitment"])
            self.assertNotEqual(first.proof["ipa"]["final_a"], second.proof["ipa"]["final_a"])
            for state in (first, second):
                result = verify_open_payload(
                    "v",
                    state.proof,
                    point,
                    _public_registration(registration),
                    len(values),
                    transcript=Transcript("tests/masked-registered"),
                )
                self.assertEqual(result["status"], "VERIFY_OK")

    def test_linear_opening_round_trip_and_wire_privacy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-masked-linear-") as tmp:
            root = pathlib.Path(tmp)
            hyrax = NativeHyrax(root)
            values = [2, 4, 8, 16]
            weights = [7, 3, 5, 9]
            registration = hyrax.commit_vector("v", values, root)
            state = hyrax.open_linear_registered(
                "v/linear", values, weights, registration, root,
                transcript=Transcript("tests/masked-linear"),
                linear_form={"kind": "unit", "weights_digest": "fixture"},
            )
            result = verify_linear_payload(
                "v/linear",
                state.proof,
                weights,
                _public_registration(registration),
                len(values),
                transcript=Transcript("tests/masked-linear"),
                linear_form={"kind": "unit", "weights_digest": "fixture"},
            )
            self.assertEqual(result["status"], "VERIFY_OK")
            self.assertEqual(
                state.proof["mask_protocol_id"], MASK_PROTOCOL_ID,
            )
            self.assertEqual(state.proof["mask_domain"], MASK_DOMAIN)
            self.assertEqual(state.proof["mask_version"], MASK_VERSION)
            self.assertFalse(_walk_wire(state.proof), _walk_wire(state.proof))

    def test_mask_schema_and_relation_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-masked-tamper-") as tmp:
            root = pathlib.Path(tmp)
            hyrax = NativeHyrax(root)
            values = [1, 2, 3, 4]
            point = [23, 29]
            registration = hyrax.commit_vector("v", values, root)
            state = hyrax.open_registered(
                "v", values, point, registration, root,
                transcript=Transcript("tests/masked-tamper"),
            )
            public = _public_registration(registration)
            for field, replacement in (
                ("mask_commitment", {"x": "1", "y": "2", "infinity": True}),
                ("mask_domain", "wrong-domain"),
                ("mask_version", MASK_VERSION + 1),
                ("mask_challenge", "1"),
            ):
                tampered = copy.deepcopy(state.proof)
                tampered[field] = replacement
                result = verify_open_payload(
                    "v", tampered, point, public, len(values),
                    transcript=Transcript("tests/masked-tamper"),
                )
                self.assertEqual(result["status"], "VERIFY_FAILED", field)
            legacy = copy.deepcopy(state.proof)
            for field in (
                "mask_protocol_id", "mask_domain", "mask_version",
                "mask_commitment", "mask_challenge",
            ):
                legacy.pop(field)
            result = verify_open_payload(
                "v", legacy, point, public, len(values),
                transcript=Transcript("tests/masked-tamper"),
            )
            self.assertEqual(result["status"], "VERIFY_FAILED")

    def test_ephemeral_opening_round_trip_and_legacy_rejection(self) -> None:
        context = {"family": "fixture", "purpose": "ephemeral"}
        payload = _ephemeral_hyrax_opening("fixture.ephemeral", [6, 10, 14], context)
        self.assertTrue(_verify_ephemeral_hyrax_opening("fixture.ephemeral", payload, context))
        self.assertEqual(payload["protocol"], "zkGPT-ephemeral-Hyrax-opening-v3-masked")
        self.assertFalse(_walk_wire(payload), _walk_wire(payload))
        legacy = copy.deepcopy(payload)
        legacy["protocol"] = "zkGPT-ephemeral-Hyrax-opening-v2"
        self.assertFalse(_verify_ephemeral_hyrax_opening("fixture.ephemeral", legacy, context))


if __name__ == "__main__":
    unittest.main()
