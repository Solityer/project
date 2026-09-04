from __future__ import annotations

import copy
import unittest

from zkgpt_gat.proof import CANONICAL_PROOF_PAYLOAD_SCHEMA, parse_proof_payload, serialize_proof_payload
from zkgpt_gat.util import digest_json


def payload() -> dict:
    value = {
        "schema": CANONICAL_PROOF_PAYLOAD_SCHEMA, "proof_type": "pi_GAT_zkGPT",
        "M_pub_digest": "a" * 64, "Y_digest": "b" * 64,
        "Commitments": {"items": {}}, "Transcript": {"final_digest": "c" * 64},
        "Com_in": [], "Pi_Lasso": {}, "FixedLookupTables": {}, "Pi_MatMul": {},
        "Pi_GKR": {}, "Pi_combine": {}, "Open_in": {}, "PrivacyProfile": {},
        "privacy_profile_digest": "d" * 64, "transcript_digest": "e" * 64,
    }
    names = {"Commitments", "Transcript", "Com_in", "Pi_Lasso", "FixedLookupTables", "Pi_MatMul", "Pi_GKR", "Pi_combine", "Open_in", "PrivacyProfile"}
    value["component_digests"] = {name: digest_json(value[name]) for name in names}
    return value


class ProofWireTest(unittest.TestCase):
    def test_exact_canonical_round_trip_and_size(self) -> None:
        value = payload(); wire = serialize_proof_payload(value)
        self.assertEqual(parse_proof_payload(wire), value)
        self.assertEqual(len(wire), len(serialize_proof_payload(parse_proof_payload(wire))))

    def test_unknown_missing_duplicate_and_component_tamper_rejected(self) -> None:
        base = payload(); unknown = copy.deepcopy(base); unknown["unexpected"] = 1
        missing = copy.deepcopy(base); del missing["Pi_GKR"]
        tampered = copy.deepcopy(base); tampered["Pi_GKR"]["changed"] = True
        for value in (unknown, missing, tampered):
            with self.assertRaises(ValueError): parse_proof_payload(serialize_proof_payload(value))
        duplicate = serialize_proof_payload(base).replace(b'{\n  "Com_in":', b'{\n  "Com_in": [],\n  "Com_in":', 1)
        with self.assertRaises(ValueError): parse_proof_payload(duplicate)

    def test_noncanonical_bytes_rejected(self) -> None:
        with self.assertRaises(ValueError): parse_proof_payload(serialize_proof_payload(payload()).rstrip())


if __name__ == "__main__":
    unittest.main()
