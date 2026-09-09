from __future__ import annotations

import copy
import tempfile
import unittest
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from zkgpt_gat.constants import COMMITMENT_BACKEND_ID
from zkgpt_gat.native_hyrax import (
    AFFINE_MASK_DOMAIN,
    AFFINE_MASK_PROTOCOL_ID,
    AFFINE_MASK_VERSION,
    MASK_DOMAIN,
    MASK_HEADER_SCHEMA,
    MASK_PROTOCOL_ID,
    MASK_VERSION,
    PROTOCOL_ID as HYRAX_MLE_PROTOCOL_ID,
    build_mask_header,
)
from zkgpt_gat.bn254 import G1Point, hash_point
from zkgpt_gat.field import BN254_FR_MODULUS
from zkgpt_gat.value_commitment import (
    CrossParamsSameValueProof,
    MultiplicationProof,
    RepresentationProof,
    SameValueProof,
    ValueCommitment,
)
from zkgpt_gat.zk_sumcheck import COMMITTED_SUMCHECK_PROTOCOL
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


def _point_json(seed: object, *, identity: bool = False) -> dict[str, object]:
    if identity:
        return G1Point.zero().to_json()
    return hash_point("zkGPT/tests/privacy/point", {"seed": str(seed)}).to_json()


def _registration(label: str, value: int) -> dict[str, object]:
    """Return a v2-shaped public native Hyrax registration fixture."""

    return {
        "backend": COMMITMENT_BACKEND_ID,
        "parameter_id": "test-params",
        "vector_length": 4,
        "label": label,
        "commitment": _point_json(("registration", label, value)),
    }


def _opening(value: int, transcript_digest: str, *, identity_mask: bool = False) -> dict[str, object]:
    """A v2 affine-masked native opening fixture with no private scalar."""

    masked = _point_json(("masked", value, transcript_digest))
    affine_mask = _point_json(("affine-mask", value, transcript_digest), identity=identity_mask)
    return {
        "protocol": HYRAX_MLE_PROTOCOL_ID,
        "parameter_id": "test-params",
        "vector_length": 4,
        "evaluation_commitment": masked,
        "affine_mask_protocol_id": AFFINE_MASK_PROTOCOL_ID,
        "affine_mask_domain": AFFINE_MASK_DOMAIN,
        "affine_mask_version": AFFINE_MASK_VERSION,
        "masked_evaluation_commitment": masked,
        "mask_evaluation_commitment": affine_mask,
        "mask_evaluation_pok": {"fixture": f"affine-mask-{value}-{transcript_digest}"},
        "ipa": {
            "rounds": [{"L": _point_json(("ipa", value, transcript_digest))}],
            "final_a": str(value + 12),
        },
        "mask_protocol_id": MASK_PROTOCOL_ID,
        "mask_domain": MASK_DOMAIN,
        "mask_version": MASK_VERSION,
        "mask_commitment": _point_json(("vector-mask", value, transcript_digest), identity=identity_mask),
        "mask_challenge": str(value + 4),
        "transcript_digest": transcript_digest,
    }


def _representation(seed: object) -> dict[str, str]:
    return RepresentationProof(
        hash_point("zkGPT/tests/privacy/representation", {"seed": str(seed)}),
        1,
        2,
        3,
    ).to_dict()


def _same_value(seed: object) -> dict[str, str]:
    return SameValueProof(
        hash_point("zkGPT/tests/privacy/same-value", {"seed": str(seed)}),
        1,
        2,
    ).to_dict()


def _cross_value(seed: object) -> dict[str, str]:
    point = lambda side: hash_point(
        "zkGPT/tests/privacy/cross-value", {"seed": str(seed), "side": side}
    )
    return CrossParamsSameValueProof(
        point("first"), point("second"), 1, 2, 3, 4
    ).to_dict()


def _multiplication(seed: object) -> dict[str, str]:
    point = lambda role: hash_point(
        "zkGPT/tests/privacy/multiplication", {"seed": str(seed), "role": role}
    )
    return MultiplicationProof(
        point("a"), point("b"), point("product"), point("cross"),
        1, 2, 3, 4, 5, 6, 7,
    ).to_dict()


def _value_commitment(seed: object) -> dict[str, str]:
    return ValueCommitment(
        hash_point("zkGPT/tests/privacy/value-commitment", {"seed": str(seed)})
    ).to_dict()


def _seed_digest(seed: object, role: str) -> str:
    return digest_json({"seed": str(seed), "role": role})


def _sumcheck(seed: object) -> dict[str, object]:
    return {
        "protocol": COMMITTED_SUMCHECK_PROTOCOL,
        "field_modulus": str(BN254_FR_MODULUS),
        "variable_count": 1,
        "degree": 1,
        "initial_claim_commitment": _value_commitment((seed, "initial"))["point"],
        "rounds": [{
            "round": 0,
            "coefficient_commitments": [
                _value_commitment((seed, "coefficient", index)) for index in range(2)
            ],
            "coefficient_mask_commitments": [
                _value_commitment((seed, "coefficient-mask", index)) for index in range(2)
            ],
            "coefficient_poks": [_representation((seed, "coefficient-pok", index)) for index in range(2)],
            "coefficient_mask_poks": [_representation((seed, "coefficient-mask-pok", index)) for index in range(2)],
            "group_equality_pok": _representation((seed, "group-equality")),
            "mask_recurrence": "zero_sum",
            "mask_recurrence_pok": _representation((seed, "mask-recurrence")),
            "challenge": "00" * 32,
        }],
        "terminal_claim_commitment": _value_commitment((seed, "terminal"))["point"],
        "terminal_mask_commitment": _value_commitment((seed, "terminal-mask"))["point"],
    }


def _linear_gate(label: str, seed: object, header: Mapping[str, object]) -> dict[str, object]:
    return {
        "protocol": "zkGPT/native-gkr-linear-gate-v1",
        "spec": {"label": label},
        "point": ["1"],
        "target_opening": {"proof": _opening(1000 + hash(str(seed)) % 100, _seed_digest(seed, label + "/target"))},
        "term_openings": [{"proof": _opening(1100 + hash(str(seed)) % 100, _seed_digest(seed, label + "/term"))}],
        "linear_equality": _representation((seed, "linear-equality")),
        "mask_header": dict(header),
    }


def _product_gate(label: str, seed: object, header: Mapping[str, object]) -> dict[str, object]:
    digest = _seed_digest(seed, label + "/target")
    return {
        "protocol": "zkGPT/native-gkr-product-gate-v1",
        "spec": {"label": label},
        "output_point": ["1"],
        "target_opening": {"proof": _opening(1200 + hash(str(seed)) % 100, digest)},
        "sumcheck": _sumcheck((seed, "product")),
        "left_opening": {"proof": _opening(1300 + hash(str(seed)) % 100, _seed_digest(seed, label + "/left"))},
        "right_opening": {"proof": _opening(1400 + hash(str(seed)) % 100, _seed_digest(seed, label + "/right"))},
        "product_commitment": _value_commitment((seed, "product-commitment")),
        "operand_product": _multiplication((seed, "operand-product")),
        "terminal_product": _multiplication((seed, "terminal-product")),
        "mask_header": dict(header),
    }


def _lasso(index: int, transcript_digest: str, session_nonce: str) -> dict[str, object]:
    """Small v2 proof-shaped Lasso object covering every native sub-relation."""

    family = "T_range"
    value_offset = index * 1000
    header = build_mask_header(
        "lasso",
        {"session_nonce": session_nonce},
        ("LC_f", "GP_f", "M_f", "ell_f", "membership-sumcheck", "lookup-input-consistency"),
    )
    gate_header = build_mask_header(
        "lasso-gates",
        {"session_nonce": session_nonce},
        ("target-evaluation", "term-evaluations", "linear-equality"),
    )
    product_header = build_mask_header(
        "lasso-products",
        {"session_nonce": session_nonce},
        ("target-evaluation", "product-sumcheck", "left-evaluation", "right-evaluation", "terminal-relation"),
    )
    membership_booleanity = _product_gate("lasso/T_range/membership/booleanity", (index, "booleanity"), product_header)
    return {
        "protocol": "zkGPT/native-lasso-memory-v1",
        "transcript_digest": transcript_digest,
        "mask_header": header,
        "lc_header": [{
            "index": 0,
            "family": family,
            "registration": _registration(f"LC_f/0/{family}", 100 + index),
        }],
        "families": [{
            "index": 0,
            "family": family,
            "descriptor": {"family": family, "kind": "function", "table_id": "T_range", "padded_count": 2},
            "table_id": "T_range",
            "query_count": 2,
            "beta": str(150 + index),
            "gp_registration": _registration(f"GP_f/0/{family}", 110 + index),
            "query_link": {
                "point": ["1"],
                "lc_form": {"kind": "lasso-lc-query"},
                "lc_claim": _opening(value_offset + 120 + index, transcript_digest),
                "canonical_form": {"kind": "lasso-canonical-query"},
                "canonical_claim": _opening(value_offset + 130 + index, transcript_digest),
                "same_value": _cross_value((index, "query-link")),
            },
            "membership": {
                "matrix_registration": _registration(f"M_f/0/{family}", 160 + index),
                "padding_row": {"row": 1, "address": 0},
                "booleanity": membership_booleanity,
                "one_hot": {
                    "point": ["1"],
                    "linear_form": {"kind": "lasso-membership-row-sum"},
                    "claim": _opening(value_offset + 170 + index, transcript_digest),
                    "same_value": _same_value((index, "one-hot")),
                },
                "ell_f": _opening(value_offset + 180 + index, transcript_digest),
                "ell_link": _cross_value((index, "ell-link")),
                "sumcheck": _sumcheck((index, "membership")),
                "matrix_opening": {
                    "linear_form": {"kind": "lasso-membership-matrix"},
                    "proof": _opening(value_offset + 190 + index, transcript_digest),
                },
                "terminal_product": _multiplication((index, "membership-terminal")),
            },
            "gates": [
                {"kind": "linear", "proof": _linear_gate("memory/write-timestamp", (index, "write"), gate_header)},
                {"kind": "product", "proof": _product_gate("memory/left/accumulator-recurrence", (index, "left"), product_header)},
                {"kind": "product", "proof": _product_gate("memory/right/accumulator-recurrence", (index, "right"), product_header)},
                {"kind": "linear", "proof": _linear_gate("memory/grand-product-equality", (index, "grand"), gate_header)},
            ],
            "local_combine": {
                "sources": [],
                "initial_claim_commitment": _value_commitment((index, "local-initial")),
                "sumcheck": _sumcheck((index, "local-combine")),
                "selector_point": ["1"],
                "point": ["1"],
                "openings": {
                    "LC": _opening(value_offset + 200 + index, transcript_digest),
                    "GP": _opening(value_offset + 210 + index, transcript_digest),
                },
                "combined_commitment": _value_commitment((index, "combined")),
                "coefficient_evaluation": str(220 + index),
                "terminal_product": _multiplication((index, "local-terminal")),
                "transcript_digest": transcript_digest,
            },
        }],
    }


def _components(
    index: int,
    *,
    session_nonce: str,
    dynamic_index: int | None = None,
    transcript_index: int | None = None,
    identity_mask: bool = False,
) -> dict[str, dict[str, Any]]:
    dynamic_index = index if dynamic_index is None else dynamic_index
    transcript_index = index if transcript_index is None else transcript_index
    digest = f"{transcript_index + 1:064x}"
    lasso = _lasso(index, digest, session_nonce)
    relation_header = lambda label, coverage: build_mask_header(
        label, {"session_nonce": session_nonce}, coverage
    )
    matmul_header = relation_header(
        "matmul",
        ("output-evaluation", "residual-evaluation", "inner-product-sumcheck", "left-evaluation", "right-evaluation", "terminal-relation"),
    )
    gkr_header = relation_header(
        "gkr",
        ("linear-equality", "product-sumcheck", "operand-product", "terminal-selector", "lookup-input-consistency"),
    )
    combine_header = relation_header(
        "input_combine",
        ("source-claims", "combine-sumcheck", "Open_W", "Open_Aux", "Open_L", "terminal-relation"),
    )
    components: dict[str, dict[str, Any]] = {
        "lasso": lasso,
        "matmul": {
            "mask_header": matmul_header,
            "claims": [{
                "spec": {"label": f"matmul/claim/{index}"},
                "output_opening": {"linear_form": {}, "proof": _opening(300 + index * 1000, digest, identity_mask=identity_mask)},
                "residual_opening": {"linear_form": {}, "proof": _opening(310 + index * 1000, digest, identity_mask=identity_mask)},
                "left_opening": {"linear_form": {}, "proof": _opening(320 + index * 1000, digest, identity_mask=identity_mask)},
                "right_opening": {"linear_form": {}, "proof": _opening(330 + index * 1000, digest, identity_mask=identity_mask)},
                "residual_binding": _representation((index, "matmul-residual")),
                "sumcheck": _sumcheck((index, "matmul")),
                "terminal_multiplication": _multiplication((index, "matmul-terminal")),
            }],
            "transcript_digest": digest,
        },
        "gkr": {
            "mask_header": gkr_header,
            "claims": [
                {"kind": "linear", "proof": _linear_gate(f"lasso/T_range/packed_query", (index, "lookup"), relation_header("gkr-linear", ("target-evaluation", "term-evaluations", "linear-equality")))},
                {"kind": "product", "proof": _product_gate(f"gkr/product/{index}", (index, "gkr-product"), relation_header("gkr-product", ("target-evaluation", "product-sumcheck", "left-evaluation", "right-evaluation", "terminal-relation")))},
            ],
            "transcript_digest": digest,
        },
        "input_combine": {
            "mask_header": combine_header,
            "sources": [
                {"segment": "com_W", "linear_form": {}, "weight": "1", "evaluation_claim": _opening(400 + index * 1000, digest, identity_mask=identity_mask)},
                {"segment": "com_Aux", "linear_form": {}, "weight": "1", "evaluation_claim": _opening(410 + index * 1000, digest, identity_mask=identity_mask)},
                {"segment": "com_L", "linear_form": {}, "weight": "1", "evaluation_claim": _opening(420 + index * 1000, digest, identity_mask=identity_mask)},
            ],
            "sumcheck": _sumcheck((index, "input-combine")),
            "terminal_product": _multiplication((index, "input-combine-terminal")),
            "transcript_digest": digest,
        },
        "terminal_openings": {
            "items": {
                "com_W": {
                    "registration": _registration("com_W", 11),
                    "proof": _opening(500 + index * 1000, digest, identity_mask=identity_mask),
                },
                "com_Aux": {
                    "registration": _registration("com_Aux", 20 + dynamic_index),
                    "proof": _opening(510 + index * 1000, digest, identity_mask=identity_mask),
                },
                "com_L": {
                    "registration": _registration("com_L", 30 + dynamic_index),
                    "proof": _opening(520 + index * 1000, digest, identity_mask=identity_mask),
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
    session_nonce = fresh_session_nonce()
    metadata: dict[str, Any] = {
        "privacy_extension_profile_id": PROFILE_ID,
        "privacy_profile_status": PROFILE_STATUS,
        "session_nonce": session_nonce,
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
    components = _components(
        mask_index,
        session_nonce=session_nonce,
        dynamic_index=dynamic_index,
        transcript_index=transcript_index,
        identity_mask=identity_mask,
    )
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
        self.assertEqual(
            lasso_registrations,
            ["GP_f/0/T_range", "M_f/0/T_range", "LC_f/0/T_range"],
        )

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
        active = components["claims"][0]["proof"]["target_opening"]["proof"]
        for field in ("mask_protocol_id", "mask_domain", "mask_version", "mask_commitment", "mask_challenge"):
            active.pop(field)
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

    def test_masked_claim_rejects_unclassified_nested_secret_field(self) -> None:
        bundle = _session(0)
        components = copy.deepcopy(bundle["proof_payload"]["Pi_GKR"])
        claim = components["claims"][0]["proof"]["target_opening"]["proof"]
        claim["blind"] = "must-not-serialize"
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


if __name__ == "__main__":
    unittest.main()
