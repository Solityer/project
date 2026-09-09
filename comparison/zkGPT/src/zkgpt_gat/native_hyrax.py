"""Native BN254 Hyrax commitment and masked zero-knowledge MLE evaluation proof.

The committed polynomial is one complete power-of-two vector.  An opening
proves that a fresh Pedersen commitment ``D = f(r) U + tau H`` contains the
evaluation of an externally supplied vector commitment.  The evaluation and
all commitment/masking scalars stay private.  Every active opening applies a
fresh, transcript-bound masking vector before the logarithmic IPA; the wire
contains only the public mask commitment, Fiat--Shamir challenge, IPA group
messages and final scalar.  The verifier reconstructs the masked statement
and folded public equality vector itself.

There is intentionally no verifier API that trusts a proof-provided base
commitment or point.  The trusted commitment, common N_in domain, random point,
and public-statement context are mandatory verifier inputs.
"""
from __future__ import annotations

import hashlib
import pathlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .bn254 import BN254_FQ_MODULUS, G1Point, generator_vector, hash_point, multi_scalar_mul, scalar_random
from .field import BN254_FR_MODULUS, f_inv, log2_power_of_two, parse_canonical_decimal, to_field_int
from .gpu_execution import active_mle_fold
from .transcript import Transcript
from .util import digest_json, ensure_dir, read_json, write_json
from .value_commitment import (
    PedersenParams,
    RepresentationProof,
    ValueCommitment,
    commit_value,
    prove_representation,
    verify_representation,
)


PROTOCOL_ID = "zkGPT-hyrax-mle-evaluation-commitment-v3-masked-affine"
LINEAR_PROTOCOL_ID = "zkGPT-hyrax-linear-evaluation-commitment-v2-masked-affine"
BATCHED_LINEAR_PROTOCOL_ID = "zkGPT-hyrax-batched-linear-claim-v2-masked-affine"
PARAMETER_ID = "zkGPT-bn254-hyrax-generators-v2"
MASK_PROTOCOL_ID = "zkGPT-hyrax-masked-ipa-v2"
MASK_DOMAIN = "zkGPT/Hyrax/Mask/V_mask/v1"
MASK_VERSION = 1

# An evaluation commitment is itself a private scalar commitment.  The
# masked wire carries only an affine sum and a separately committed mask;
# the verifier subtracts the latter in G1 before replaying the native Hyrax
# relation.  This is deliberately distinct from the vector-level V_mask used
# by the IPA below.
AFFINE_MASK_PROTOCOL_ID = "zkGPT/value-commitment-affine-mask-v1"
AFFINE_MASK_DOMAIN = "zkGPT/Hyrax/Mask/evaluation/v1"
AFFINE_MASK_VERSION = 1
AFFINE_MASK_FIELDS = {
    "affine_mask_protocol_id",
    "affine_mask_domain",
    "affine_mask_version",
    "masked_evaluation_commitment",
    "mask_evaluation_commitment",
    "mask_evaluation_pok",
}
MASK_HEADER_SCHEMA = "zkGPT/masking-header-v1"
MASK_HEADER_VERSION = 1


def build_mask_header(
    label: str,
    context: Mapping[str, Any] | None,
    coverage: Sequence[str],
) -> Dict[str, Any]:
    """Build a public, versioned coverage header for a native component.

    The session nonce is the only proof-instance entropy used for these
    deterministic coverage commitments.  It is sampled from the OS CSPRNG
    before the formal transcript is created; no model blinder, prior
    transcript state, or predictable label is used as randomness.
    """

    if not isinstance(label, str) or not label:
        raise ValueError("mask header label must be non-empty")
    context = dict(context or {})
    nonce = str(context.get("session_nonce") or context.get("transcript_domain") or "standalone")
    ordered = [str(item) for item in coverage]
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("mask header coverage must be non-empty and unique")
    commitments = {
        item: hash_point(
            AFFINE_MASK_DOMAIN + "/coverage",
            {"session_nonce": nonce, "component": label, "coverage": item},
        ).to_json()
        for item in ordered
    }
    header: Dict[str, Any] = {
        "schema": MASK_HEADER_SCHEMA,
        "version": MASK_HEADER_VERSION,
        "component": label,
        "session_nonce": nonce,
        "coverage": ordered,
        "coverage_commitments": commitments,
    }
    header["header_digest"] = digest_json(header)
    return header


def absorb_mask_header(transcript: Transcript, header: Mapping[str, Any]) -> None:
    """Absorb a complete MaskHeader before any component challenge."""

    if not isinstance(header, Mapping):
        raise ValueError("mask header must be a mapping")
    if header.get("schema") != MASK_HEADER_SCHEMA or header.get("version") != MASK_HEADER_VERSION:
        raise ValueError("mask header schema/version mismatch")
    if header.get("header_digest") != digest_json({key: value for key, value in header.items() if key != "header_digest"}):
        raise ValueError("mask header digest mismatch")
    transcript.absorb("masking/MaskHeader", dict(header))


def _eq_table(point: Sequence[int]) -> List[int]:
    values = [1]
    for raw in point:
        r = to_field_int(raw)
        one_minus = (1 - r) % BN254_FR_MODULUS
        values = [item for value in values for item in (
            value * one_minus % BN254_FR_MODULUS,
            value * r % BN254_FR_MODULUS,
        )]
    return values


def _dot(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) != len(right):
        raise ValueError("inner-product vector length mismatch")
    return sum(to_field_int(a) * to_field_int(b) for a, b in zip(left, right)) % BN254_FR_MODULUS


def _require_power_of_two(value: int, label: str) -> int:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{label} must be a positive power of two")
    return value


def _point(value: Any) -> G1Point:
    if isinstance(value, G1Point):
        point = value
    else:
        if not isinstance(value, Mapping) or set(value) != {"x", "y", "infinity"}:
            raise ValueError("non-canonical BN254 G1 encoding")
        if not isinstance(value["infinity"], bool):
            raise ValueError("non-canonical BN254 G1 infinity flag")
        x = parse_canonical_decimal(value["x"], label="G1.x", nonnegative=True)
        y = parse_canonical_decimal(value["y"], label="G1.y", nonnegative=True)
        infinity = value["infinity"]
        if infinity:
            if x != 0 or y != 0:
                raise ValueError("non-canonical BN254 infinity coordinates")
        elif x >= BN254_FQ_MODULUS or y >= BN254_FQ_MODULUS:
            raise ValueError("BN254 coordinate is outside the base field")
        point = G1Point(x, y, infinity)
    if point.infinity:
        if point.x != 0 or point.y != 0:
            raise ValueError("non-canonical BN254 infinity coordinates")
    elif not (0 <= point.x < BN254_FQ_MODULUS and 0 <= point.y < BN254_FQ_MODULUS):
        raise ValueError("BN254 coordinate is outside the base field")
    if not point.is_on_curve():
        raise ValueError("invalid BN254 G1 point")
    return point


@lru_cache(maxsize=8)
def _params(vector_length: int) -> Dict[str, Any]:
    n = _require_power_of_two(int(vector_length), "Hyrax vector length")
    ipa_length = 2 * n
    pp_label = f"{PARAMETER_ID}/N={n}"
    vector_generators = generator_vector(pp_label, n, "G/vector")
    blind_generator = hash_point("zkGPT/Hyrax/G/blind", pp_label)
    padding_generators = generator_vector(pp_label, n - 1, "G/padding")
    ipa_generators = vector_generators + [blind_generator] + padding_generators
    equality_generators = generator_vector(pp_label, ipa_length, "J/equality")
    evaluation_generator = hash_point("zkGPT/Hyrax/U/evaluation", pp_label)
    if any(point == blind_generator for point in vector_generators):
        raise RuntimeError("Hyrax generator derivation collision")
    params = {
        "parameter_id": pp_label,
        "vector_generators": vector_generators,
        "blind_generator": blind_generator,
        "ipa_generators": ipa_generators,
        "equality_generators": equality_generators,
        "evaluation_generator": evaluation_generator,
        "ipa_length": ipa_length,
    }
    params["generator_set_identity"] = _generator_set_identity(params)
    return params


@lru_cache(maxsize=8)
def _masked_ipa_generators(vector_length: int) -> Dict[str, Any]:
    """Return the power-of-two IPA bases with the independent mask base.

    The unmasked Hyrax statement has ``2*N`` witness/base coordinates.  A
    single coordinate carrying ``y*sigma`` and ``V_mask`` is added, then all
    bases are padded to the next power of two.  The first ``2*N`` generators
    are exactly the native Hyrax generators, which is required for the
    equation ``P' = P + y*S`` to be sound.
    """

    params = _params(vector_length)
    base_length = int(params["ipa_length"])
    extended_length = 1 << (base_length + 1).bit_length()
    mask_generator = hash_point(
        MASK_DOMAIN,
        {
            "parameter_id": params["parameter_id"],
            "vector_length": int(vector_length),
            "base_ipa_length": base_length,
        },
    )
    if mask_generator in params["ipa_generators"]:
        raise RuntimeError("masked Hyrax V_mask collides with an IPA generator")
    padding_count = extended_length - base_length - 1
    g_padding = generator_vector(
        params["parameter_id"], padding_count, "G/masked-ipa-padding"
    )
    h_padding = generator_vector(
        params["parameter_id"], extended_length - base_length, "J/masked-ipa-padding"
    )
    generators = list(params["ipa_generators"]) + [mask_generator] + g_padding
    equality_generators = list(params["equality_generators"]) + h_padding
    if len(generators) != extended_length or len(equality_generators) != extended_length:
        raise RuntimeError("masked Hyrax generator extension has invalid length")
    if any(point == mask_generator for point in g_padding):
        raise RuntimeError("masked Hyrax V_mask collides with padding generator")
    return {
        "mask_generator": mask_generator,
        "generators": generators,
        "equality_generators": equality_generators,
        "length": extended_length,
    }


def hyrax_value_params(vector_length: int) -> PedersenParams:
    """Return the authenticated value-commitment bases for this N_in domain.

    Every committed Sumcheck and terminal same-value proof must use these
    exact U/H bases; unrelated module defaults cannot compose with a Hyrax
    evaluation commitment.
    """
    params = _params(vector_length)
    return PedersenParams(
        params["evaluation_generator"],
        params["blind_generator"],
        f"{params['parameter_id']}/value-commitment",
    )


def _statement_transcript(
    transcript: Transcript,
    label: str,
    vector_length: int,
    expected_commitment: G1Point,
    point: Sequence[int],
    evaluation_commitment: G1Point,
    context: Mapping[str, Any],
) -> None:
    transcript.absorb(
        "hyrax/statement",
        {
            "protocol": PROTOCOL_ID,
            "parameter_id": f"{PARAMETER_ID}/N={vector_length}",
            "label": label,
            "vector_length": vector_length,
            "commitment": expected_commitment,
            "point": list(point),
            "evaluation_commitment": evaluation_commitment,
            "public_context": dict(context),
        },
    )


def _linear_statement_transcript(
    transcript: Transcript,
    label: str,
    vector_length: int,
    expected_commitment: G1Point,
    linear_form: Mapping[str, Any],
    evaluation_commitment: G1Point,
    context: Mapping[str, Any],
) -> None:
    transcript.absorb(
        "hyrax/linear-statement",
        {
            "protocol": LINEAR_PROTOCOL_ID,
            "parameter_id": f"{PARAMETER_ID}/N={vector_length}",
            "label": label,
            "vector_length": vector_length,
            "commitment": expected_commitment,
            "linear_form": dict(linear_form),
            "evaluation_commitment": evaluation_commitment,
            "public_context": dict(context),
        },
    )


def _batched_linear_statement_transcript(
    transcript: Transcript,
    label: str,
    vector_length: int,
    expected_commitment: G1Point,
    linear_form: Mapping[str, Any],
    evaluation_commitment: G1Point,
    context: Mapping[str, Any],
) -> None:
    transcript.absorb(
        "hyrax/batched-linear-claim",
        {
            "protocol": BATCHED_LINEAR_PROTOCOL_ID,
            "parameter_id": f"{PARAMETER_ID}/N={vector_length}",
            "label": label,
            "vector_length": vector_length,
            "commitment": expected_commitment,
            "linear_form": dict(linear_form),
            "evaluation_commitment": evaluation_commitment,
            "public_context": dict(context),
        },
    )


@dataclass(frozen=True)
class HyraxOpeningState:
    """Serialized proof plus prover-local opening retained only in memory."""

    proof: Dict[str, Any]
    evaluation_commitment: ValueCommitment
    evaluation: int
    blind: int


@dataclass(frozen=True)
class _RegistrationCacheEntry:
    """Session-local authentication record for one registered vector.

    The vector itself is deliberately not retained.  Every later claim still
    hashes the supplied vector using the canonical field encoding, while this
    bounded record avoids rebuilding the expensive commitment MSM.
    """

    parameter_id: str
    vector_length: int
    registration_label: str
    commitment: G1Point
    vector_digest: str
    generator_set_identity: str
    blind: int


REGISTRATION_CACHE_LIMIT = 256


def _canonical_vector_digest(vector: Sequence[int]) -> str:
    """Hash a vector using an unambiguous canonical BN254-Fr encoding."""

    digest = hashlib.sha256()
    digest.update(b"zkGPT/Hyrax/registered-vector/v1\0")
    digest.update(len(vector).to_bytes(8, "big", signed=False))
    for value in vector:
        digest.update(to_field_int(value).to_bytes(32, "big", signed=False))
    return digest.hexdigest()


def _generator_set_identity(params: Mapping[str, Any]) -> str:
    """Return the identity of the bases authenticating a registered vector."""

    digest = hashlib.sha256()
    digest.update(b"zkGPT/Hyrax/registration-generators/v1\0")
    parameter_id = str(params["parameter_id"]).encode("utf-8")
    digest.update(len(parameter_id).to_bytes(4, "big", signed=False))
    digest.update(parameter_id)
    generators = list(params["vector_generators"]) + [params["blind_generator"]]
    digest.update(len(generators).to_bytes(8, "big", signed=False))
    for point in generators:
        digest.update(b"\1" if point.infinity else b"\0")
        digest.update(int(point.x).to_bytes(32, "big", signed=False))
        digest.update(int(point.y).to_bytes(32, "big", signed=False))
    return digest.hexdigest()


def _prove_ipa(
    transcript: Transcript,
    a: Sequence[int],
    b: Sequence[int],
    generators: Sequence[G1Point],
    equality_generators: Sequence[G1Point],
    inner_product_generator: G1Point,
    statement_point: G1Point,
) -> Dict[str, Any]:
    if not (len(a) == len(b) == len(generators) == len(equality_generators)):
        raise ValueError("Hyrax IPA vector length mismatch")
    _require_power_of_two(len(a), "Hyrax IPA length")
    av = [to_field_int(value) for value in a]
    bv = [to_field_int(value) for value in b]
    gs = list(generators)
    hs = list(equality_generators)
    current = statement_point
    rounds: List[Dict[str, Any]] = []
    while len(av) > 1:
        round_index = len(rounds)
        half = len(av) // 2
        a_l, a_r = av[:half], av[half:]
        b_l, b_r = bv[:half], bv[half:]
        g_l, g_r = gs[:half], gs[half:]
        h_l, h_r = hs[:half], hs[half:]
        left = multi_scalar_mul(g_r, a_l) + multi_scalar_mul(h_l, b_r) + _dot(a_l, b_r) * inner_product_generator
        right = multi_scalar_mul(g_l, a_r) + multi_scalar_mul(h_r, b_l) + _dot(a_r, b_l) * inner_product_generator
        transcript.absorb(f"hyrax/ipa/{round_index}/LR", {"L": left, "R": right})
        challenge = transcript.challenge_scalar(f"hyrax/ipa/{round_index}/challenge")
        challenge_inv = f_inv(challenge)
        challenge_sq = challenge * challenge % BN254_FR_MODULUS
        current = challenge_sq * left + current + f_inv(challenge_sq) * right
        av = [
            (challenge * left_value + challenge_inv * right_value) % BN254_FR_MODULUS
            for left_value, right_value in zip(a_l, a_r)
        ]
        bv = [
            (challenge_inv * left_value + challenge * right_value) % BN254_FR_MODULUS
            for left_value, right_value in zip(b_l, b_r)
        ]
        gs = [
            challenge_inv * left_generator + challenge * right_generator
            for left_generator, right_generator in zip(g_l, g_r)
        ]
        hs = [
            challenge * left_generator + challenge_inv * right_generator
            for left_generator, right_generator in zip(h_l, h_r)
        ]
        rounds.append({"L": left.to_json(), "R": right.to_json()})
    expected = av[0] * gs[0] + bv[0] * hs[0] + (av[0] * bv[0] % BN254_FR_MODULUS) * inner_product_generator
    if current != expected:
        raise RuntimeError("Hyrax IPA prover terminal relation failed")
    transcript.absorb("hyrax/ipa/final", {"a": av[0]})
    return {"rounds": rounds, "final_a": str(av[0])}


def _verify_ipa(
    transcript: Transcript,
    proof: Mapping[str, Any],
    b: Sequence[int],
    generators: Sequence[G1Point],
    equality_generators: Sequence[G1Point],
    inner_product_generator: G1Point,
    statement_point: G1Point,
) -> bool:
    if set(proof) != {"rounds", "final_a"}:
        return False
    bv = [to_field_int(value) for value in b]
    gs = list(generators)
    hs = list(equality_generators)
    _require_power_of_two(len(bv), "Hyrax IPA length")
    rounds = proof["rounds"]
    if not isinstance(rounds, list) or len(rounds) != log2_power_of_two(len(bv)):
        return False
    current = statement_point
    for round_index, item in enumerate(rounds):
        if not isinstance(item, dict) or set(item) != {"L", "R"}:
            return False
        left = _point(item["L"])
        right = _point(item["R"])
        transcript.absorb(f"hyrax/ipa/{round_index}/LR", {"L": left, "R": right})
        challenge = transcript.challenge_scalar(f"hyrax/ipa/{round_index}/challenge")
        challenge_inv = f_inv(challenge)
        challenge_sq = challenge * challenge % BN254_FR_MODULUS
        current = challenge_sq * left + current + f_inv(challenge_sq) * right
        half = len(bv) // 2
        bv = [
            (challenge_inv * left_value + challenge * right_value) % BN254_FR_MODULUS
            for left_value, right_value in zip(bv[:half], bv[half:])
        ]
        gs = [
            challenge_inv * left_generator + challenge * right_generator
            for left_generator, right_generator in zip(gs[:half], gs[half:])
        ]
        hs = [
            challenge * left_generator + challenge_inv * right_generator
            for left_generator, right_generator in zip(hs[:half], hs[half:])
        ]
    final_a_raw = proof["final_a"]
    try:
        final_a = parse_canonical_decimal(final_a_raw, label="Hyrax IPA final_a", nonnegative=True)
    except ValueError:
        return False
    if final_a >= BN254_FR_MODULUS:
        return False
    transcript.absorb("hyrax/ipa/final", {"a": final_a})
    return current == final_a * gs[0] + bv[0] * hs[0] + (final_a * bv[0] % BN254_FR_MODULUS) * inner_product_generator


def _mask_wire_fields(mask_commitment: G1Point, challenge: int) -> Dict[str, Any]:
    """Return the public masked-IPA schema (never include mask scalars)."""

    return {
        "mask_protocol_id": MASK_PROTOCOL_ID,
        "mask_domain": MASK_DOMAIN,
        "mask_version": MASK_VERSION,
        "mask_commitment": mask_commitment.to_json(),
        "mask_challenge": str(to_field_int(challenge)),
    }


def _absorb_mask_and_challenge(
    transcript: Transcript,
    mask_commitment: G1Point,
    *,
    vector_length: int,
) -> tuple[Dict[str, Any], int]:
    """Absorb the public mask commitment, then derive its nonzero challenge."""

    # A masking commitment at infinity carries no binding information.  Keep
    # this guard at the transcript boundary as well as in the prover and
    # verifier callers so a future native caller cannot accidentally turn an
    # identity point into an apparently valid masked opening.
    if mask_commitment.infinity:
        raise ValueError("masked Hyrax commitment must be a finite G1 point")

    fields = _mask_wire_fields(mask_commitment, 0)
    # The challenge is derived after S is absorbed.  The placeholder value is
    # not absorbed and is replaced only in the returned public wire fields.
    transcript.absorb(
        "hyrax/mask/commitment",
        {
            "mask_protocol_id": MASK_PROTOCOL_ID,
            "mask_domain": MASK_DOMAIN,
            "mask_version": MASK_VERSION,
            "vector_length": int(vector_length),
            "mask_commitment": mask_commitment,
        },
    )
    challenge = transcript.challenge_scalar("hyrax/mask/challenge", nonzero=True)
    fields["mask_challenge"] = str(challenge)
    return fields, challenge


def _affine_mask_wire(
    masked_commitment: G1Point,
    mask_commitment: G1Point,
    mask_pok: RepresentationProof,
) -> Dict[str, Any]:
    """Return the public affine evaluation-mask envelope.

    ``masked_commitment`` is the only commitment to the evaluation exposed by
    the proof.  The raw evaluation commitment is intentionally reconstructed
    as ``masked_commitment - mask_commitment`` by the verifier and is never a
    serialized field.  The representation proof is a Schnorr PoK for the
    fresh mask commitment; its responses do not reveal the mask opening.
    """

    return {
        "affine_mask_protocol_id": AFFINE_MASK_PROTOCOL_ID,
        "affine_mask_domain": AFFINE_MASK_DOMAIN,
        "affine_mask_version": AFFINE_MASK_VERSION,
        "masked_evaluation_commitment": masked_commitment.to_json(),
        "mask_evaluation_commitment": mask_commitment.to_json(),
        "mask_evaluation_pok": mask_pok.to_dict(),
    }


def _absorb_affine_mask_header(
    transcript: Transcript,
    label: str,
    masked_commitment: G1Point,
    mask_commitment: G1Point,
) -> None:
    """Absorb both affine mask commitments before deriving any challenge."""

    transcript.absorb(
        f"{label}/affine-mask/header",
        {
            "protocol": AFFINE_MASK_PROTOCOL_ID,
            "domain": AFFINE_MASK_DOMAIN,
            "version": AFFINE_MASK_VERSION,
            "masked_evaluation_commitment": masked_commitment,
            "mask_evaluation_commitment": mask_commitment,
        },
    )


def _prove_affine_mask(
    transcript: Transcript,
    label: str,
    raw_commitment: G1Point,
    params: PedersenParams,
) -> tuple[G1Point, Dict[str, Any]]:
    """Create a fresh affine mask and a PoK for its commitment."""

    # A nonzero blind is not required for correctness, but rejecting an
    # identity mask makes accidental zero masking fail closed and gives the
    # privacy verifier an unambiguous finite point to audit.
    for _ in range(64):
        mask_value = scalar_random(nonzero=False)
        mask_blind = scalar_random(nonzero=True)
        mask, _ = commit_value(params, mask_value, mask_blind)
        if mask.point.infinity:
            continue
        masked = raw_commitment + mask.point
        if masked.infinity:
            continue
        _absorb_affine_mask_header(transcript, label, masked, mask.point)
        pok = prove_representation(
            params,
            mask,
            mask_value,
            mask_blind,
            label=f"{label}/affine-mask/representation",
            transcript=transcript,
        )
        return masked, _affine_mask_wire(masked, mask.point, pok)
    raise RuntimeError("affine evaluation mask sampling produced only identity points")


def _recover_affine_masked_commitment(
    proof: Mapping[str, Any],
    *,
    params: PedersenParams,
) -> tuple[ValueCommitment, G1Point, G1Point]:
    """Parse a masked evaluation envelope and recover its raw commitment."""

    if not isinstance(proof, Mapping) or not AFFINE_MASK_FIELDS.issubset(proof):
        raise ValueError("masked Hyrax evaluation envelope is incomplete")
    if (
        proof.get("affine_mask_protocol_id") != AFFINE_MASK_PROTOCOL_ID
        or proof.get("affine_mask_domain") != AFFINE_MASK_DOMAIN
        or proof.get("affine_mask_version") != AFFINE_MASK_VERSION
    ):
        raise ValueError("masked Hyrax affine schema mismatch")
    masked = _point(proof.get("masked_evaluation_commitment"))
    mask = _point(proof.get("mask_evaluation_commitment"))
    if mask.infinity:
        raise ValueError("affine evaluation mask must be finite")
    # The historical field name remains part of the canonical wire for
    # compatibility with the native opening schema, but it is not an
    # independent statement.  Refuse an alias/tamper instead of letting a
    # caller compose a different public point than the one authenticated by
    # the affine mask header.
    if "evaluation_commitment" in proof and _point(proof["evaluation_commitment"]) != masked:
        raise ValueError("affine evaluation commitment does not equal masked point")
    # ``mask_evaluation_pok`` is checked by the transcript-aware verifier;
    # this parser only performs strict point/schema validation and derives the
    # raw group element used by the algebraic caller.
    raw = masked - mask
    return ValueCommitment(raw), masked, mask


def evaluation_commitment_from_payload(
    proof: Mapping[str, Any],
    *,
    params: PedersenParams | None = None,
) -> ValueCommitment:
    """Recover the raw evaluation commitment from a public masked claim.

    The helper is intentionally group-only.  Callers that need acceptance
    must separately run :func:`verify_representation` through the opening
    verifier; this function exists so native GKR/MatMul/Lasso composition uses
    the recovered raw commitment instead of accidentally composing masked
    points.
    """

    raw, _masked, _mask = _recover_affine_masked_commitment(proof, params=params)
    return raw


def masked_evaluation_claim_from_payload(proof: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the canonical public masked-claim fields from a proof wire."""

    fields = (
        "evaluation_commitment",
        "affine_mask_protocol_id",
        "affine_mask_domain",
        "affine_mask_version",
        "masked_evaluation_commitment",
        "mask_evaluation_commitment",
        "mask_evaluation_pok",
    )
    if not isinstance(proof, Mapping) or any(key not in proof for key in fields):
        raise ValueError("evaluation proof has no canonical masked claim envelope")
    if _point(proof["evaluation_commitment"]) != _point(proof["masked_evaluation_commitment"]):
        raise ValueError("evaluation claim historical field is not the masked point")
    return {key: proof[key] for key in fields}


def _verify_affine_mask(
    transcript: Transcript,
    label: str,
    proof: Mapping[str, Any],
    params: PedersenParams,
) -> tuple[ValueCommitment, G1Point, G1Point]:
    """Verify the affine mask PoK and return the recovered raw commitment."""

    raw, masked, mask = _recover_affine_masked_commitment(proof, params=params)
    _absorb_affine_mask_header(transcript, label, masked, mask)
    mask_pok = proof.get("mask_evaluation_pok")
    if not verify_representation(
        params,
        ValueCommitment(mask),
        mask_pok,
        label=f"{label}/affine-mask/representation",
        transcript=transcript,
    ):
        raise ValueError("affine evaluation mask representation proof failed")
    return raw, masked, mask


def prove_masked_scalar_claim(
    label: str,
    value: int,
    params: PedersenParams,
    transcript: Transcript,
    *,
    blind: int | None = None,
) -> tuple[ValueCommitment, int, Dict[str, Any]]:
    """Commit one private scalar and expose only its affine-masked claim."""

    raw, opening_blind = commit_value(params, value, blind)
    masked, affine = _prove_affine_mask(transcript, label, raw.point, params)
    proof = {
        "protocol": "zkGPT/masked-scalar-claim-v1",
        "evaluation_commitment": masked.to_json(),
        **affine,
    }
    return raw, opening_blind, proof


def verify_masked_scalar_claim(
    label: str,
    proof: Mapping[str, Any],
    params: PedersenParams,
    transcript: Transcript,
) -> ValueCommitment:
    """Verify and recover a private scalar commitment from its public claim."""

    if not isinstance(proof, Mapping) or proof.get("protocol") != "zkGPT/masked-scalar-claim-v1":
        raise ValueError("masked scalar claim protocol mismatch")
    raw, masked, _mask = _verify_affine_mask(transcript, label, proof, params)
    if proof.get("evaluation_commitment") != masked.to_json():
        raise ValueError("masked scalar claim commitment mismatch")
    return raw


def _prove_masked_ipa(
    transcript: Transcript,
    a: Sequence[int],
    b: Sequence[int],
    generators: Sequence[G1Point],
    equality_generators: Sequence[G1Point],
    inner_product_generator: G1Point,
    statement_point: G1Point,
    *,
    vector_length: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Mask a native IPA statement and prove the resulting extended claim.

    For ``s`` and ``sigma`` sampled from the OS CSPRNG, the public mask point
    is ``S = <s,G> + <s,b>U' + sigma*V_mask``.  After absorbing S, the
    transcript challenge ``y`` defines ``a' = (a+y*s, y*sigma, 0...)`` and
    ``b' = (b,0...)``.  Thus the extended IPA statement is exactly
    ``P' = P + y*S`` while the hidden mask material never reaches the wire.
    """

    if len(a) != len(b) or len(a) != len(generators):
        raise ValueError("masked Hyrax IPA base vector length mismatch")
    if len(equality_generators) != len(a):
        raise ValueError("masked Hyrax equality vector length mismatch")
    extension = _masked_ipa_generators(vector_length)
    base_length = len(a)
    if base_length != _params(vector_length)["ipa_length"]:
        raise ValueError("masked Hyrax base length does not match domain")
    # The linear combination can very rarely land at the group identity.  A
    # zero/infinity mask is not a hiding commitment, so resample the complete
    # mask (including sigma) instead of emitting a degenerate proof.  The
    # bounded retry also makes a broken RNG fail closed rather than loop
    # forever in production or tests.
    mask_commitment = None
    mask_scalars: list[int] = []
    sigma = 0
    for _ in range(64):
        candidate_scalars = [scalar_random(nonzero=False) for _ in range(base_length)]
        candidate_sigma = scalar_random(nonzero=False)
        candidate = (
            multi_scalar_mul(generators, candidate_scalars)
            + _dot(candidate_scalars, b) * inner_product_generator
            + candidate_sigma * extension["mask_generator"]
        )
        if not candidate.infinity:
            mask_scalars = candidate_scalars
            sigma = candidate_sigma
            mask_commitment = candidate
            break
    if mask_commitment is None:
        raise RuntimeError("native Hyrax mask sampling produced only identity commitments")
    fields, challenge = _absorb_mask_and_challenge(
        transcript, mask_commitment, vector_length=vector_length
    )
    masked_a = [
        (to_field_int(value) + challenge * scalar) % BN254_FR_MODULUS
        for value, scalar in zip(a, mask_scalars)
    ]
    masked_a.append(challenge * sigma % BN254_FR_MODULUS)
    masked_a.extend([0] * (extension["length"] - len(masked_a)))
    masked_b = [to_field_int(value) for value in b]
    masked_b.extend([0] * (extension["length"] - len(masked_b)))
    masked_statement = statement_point + challenge * mask_commitment
    ipa = _prove_ipa(
        transcript,
        masked_a,
        masked_b,
        extension["generators"],
        extension["equality_generators"],
        inner_product_generator,
        masked_statement,
    )
    return ipa, fields


def _verify_masked_ipa(
    transcript: Transcript,
    proof: Mapping[str, Any],
    b: Sequence[int],
    generators: Sequence[G1Point],
    equality_generators: Sequence[G1Point],
    inner_product_generator: G1Point,
    statement_point: G1Point,
    *,
    vector_length: int,
) -> bool:
    """Reconstruct and verify the masked IPA without private witness data."""

    required = {
        "mask_protocol_id",
        "mask_domain",
        "mask_version",
        "mask_commitment",
        "mask_challenge",
    }
    if any(key not in proof for key in required):
        return False
    if (
        proof.get("mask_protocol_id") != MASK_PROTOCOL_ID
        or proof.get("mask_domain") != MASK_DOMAIN
        or proof.get("mask_version") != MASK_VERSION
    ):
        return False
    challenge_raw = proof.get("mask_challenge")
    try:
        parse_canonical_decimal(challenge_raw, label="Hyrax mask_challenge", nonnegative=True)
    except ValueError:
        return False
    mask_commitment = _point(proof.get("mask_commitment"))
    if mask_commitment.infinity:
        return False
    fields, challenge = _absorb_mask_and_challenge(
        transcript, mask_commitment, vector_length=vector_length
    )
    if fields["mask_challenge"] != challenge_raw:
        return False
    extension = _masked_ipa_generators(vector_length)
    base_length = len(b)
    if (
        base_length != len(generators)
        or base_length != len(equality_generators)
        or base_length != _params(vector_length)["ipa_length"]
    ):
        return False
    masked_b = [to_field_int(value) for value in b]
    masked_b.extend([0] * (extension["length"] - base_length))
    masked_statement = statement_point + challenge * mask_commitment
    return _verify_ipa(
        transcript,
        proof.get("ipa", {}),
        masked_b,
        extension["generators"],
        extension["equality_generators"],
        inner_product_generator,
        masked_statement,
    )


@dataclass(frozen=True)
class NativeHyrax:
    zk_root: pathlib.Path
    opening_log: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False, compare=False)
    _registration_cache: OrderedDict[tuple[Any, ...], _RegistrationCacheEntry] = field(
        default_factory=OrderedDict, init=False, repr=False, compare=False
    )
    _registration_cache_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False
    )
    _registration_stats: dict[str, int] = field(
        default_factory=lambda: {"msm_calls": 0}, init=False, repr=False, compare=False
    )

    @property
    def registration_msm_calls(self) -> int:
        """Number of full registration-authentication MSMs in this session."""

        with self._registration_cache_lock:
            return int(self._registration_stats.get("msm_calls", 0))

    @property
    def registration_cache_size(self) -> int:
        """Current bounded number of authenticated registrations."""

        with self._registration_cache_lock:
            return len(self._registration_cache)

    @staticmethod
    def _registration_cache_key(
        *,
        parameter_id: str,
        vector_length: int,
        registration_label: str,
        commitment: G1Point,
        vector_digest: str,
        generator_set_identity: str,
        blind: int,
    ) -> tuple[Any, ...]:
        # Include every authenticated identity component in the key.  The
        # commitment point is represented by its canonical immutable G1 value,
        # rather than by object identity or a caller-controlled label alone.
        return (
            parameter_id,
            int(vector_length),
            registration_label,
            commitment,
            vector_digest,
            generator_set_identity,
            int(blind),
        )

    @staticmethod
    def _registration_details(
        registration: Mapping[str, Any],
        vector_length: int,
        params: Mapping[str, Any],
    ) -> tuple[str, int, G1Point]:
        if not isinstance(registration, Mapping):
            raise ValueError("Hyrax registration must be a mapping")
        if registration.get("backend") != PROTOCOL_ID:
            raise ValueError("registered Hyrax backend mismatch")
        try:
            registered_length = int(registration.get("vector_length", -1))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("registered Hyrax vector length is invalid") from exc
        if registered_length != int(vector_length):
            raise ValueError("registered Hyrax domain mismatch")
        if registration.get("parameter_id") != params["parameter_id"]:
            raise ValueError("registered Hyrax parameter mismatch")
        registration_label = registration.get("label")
        if not isinstance(registration_label, str) or not registration_label:
            raise ValueError("registered Hyrax label is invalid")
        try:
            commitment = _point(registration["commitment"])
            blind = to_field_int(int(registration["blind"]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("registered Hyrax authentication fields are invalid") from exc
        return registration_label, blind, commitment

    @classmethod
    def _registration_entry(
        cls,
        registration: Mapping[str, Any],
        vector: Sequence[int],
        params: Mapping[str, Any],
    ) -> tuple[_RegistrationCacheEntry, tuple[Any, ...]]:
        registration_label, blind, commitment = cls._registration_details(
            registration, len(vector), params
        )
        vector_digest = _canonical_vector_digest(vector)
        generator_set_identity = str(params["generator_set_identity"])
        entry = _RegistrationCacheEntry(
            parameter_id=str(params["parameter_id"]),
            vector_length=len(vector),
            registration_label=registration_label,
            commitment=commitment,
            vector_digest=vector_digest,
            generator_set_identity=generator_set_identity,
            blind=blind,
        )
        key = cls._registration_cache_key(
            parameter_id=entry.parameter_id,
            vector_length=entry.vector_length,
            registration_label=entry.registration_label,
            commitment=entry.commitment,
            vector_digest=entry.vector_digest,
            generator_set_identity=entry.generator_set_identity,
            blind=entry.blind,
        )
        return entry, key

    def _authenticate_registration(
        self,
        registration: Mapping[str, Any],
        vector: Sequence[int],
        params: Mapping[str, Any],
        *,
        known_commitment: G1Point | None = None,
    ) -> _RegistrationCacheEntry:
        """Authenticate once and retain only a bounded session-local record."""

        entry, key = self._registration_entry(registration, vector, params)
        with self._registration_cache_lock:
            existing = self._registration_cache.get(key)
            if existing is not None:
                if existing != entry:
                    raise ValueError("registered Hyrax cache entry mismatch")
                self._registration_cache.move_to_end(key)
                return existing
            if known_commitment is None:
                self._registration_stats["msm_calls"] = self._registration_stats.get("msm_calls", 0) + 1
                recomputed = (
                    multi_scalar_mul(params["vector_generators"], vector)
                    + entry.blind * params["blind_generator"]
                )
            else:
                recomputed = known_commitment
            if recomputed != entry.commitment:
                raise ValueError("registered commitment does not bind supplied vector")
            self._registration_cache[key] = entry
            self._registration_cache.move_to_end(key)
            while len(self._registration_cache) > REGISTRATION_CACHE_LIMIT:
                self._registration_cache.popitem(last=False)
            return entry

    def _require_cached_registration(
        self,
        registration: Mapping[str, Any],
        vector: Sequence[int],
        params: Mapping[str, Any],
    ) -> _RegistrationCacheEntry:
        """Validate vector digest and require prior session registration.

        Claim/opening paths never authenticate a cache miss themselves: a
        caller must first use ``commit_vector`` or ``register_vector``.  This
        prevents a malformed registration from silently turning every claim
        into an unbounded fallback MSM.
        """

        entry, key = self._registration_entry(registration, vector, params)
        with self._registration_cache_lock:
            cached = self._registration_cache.get(key)
            if cached is None:
                raise ValueError("registered Hyrax authentication cache miss")
            if cached != entry:
                raise ValueError("registered Hyrax cache entry mismatch")
            self._registration_cache.move_to_end(key)
            return cached

    def register_vector(
        self,
        label: str,
        values: Iterable[int],
        registration: Mapping[str, Any],
        work_dir: pathlib.Path | None = None,
    ) -> Dict[str, Any]:
        """Authenticate an externally persisted registration in this session.

        Model-registry secrets are intentionally persisted across proof
        sessions.  Loading one therefore requires one explicit full MSM in the
        new session before registered claims may use the cache.
        """

        del work_dir
        vector = [to_field_int(value) for value in values]
        n = _require_power_of_two(len(vector), "committed vector length")
        params = _params(n)
        if registration.get("label") != label:
            raise ValueError("registered Hyrax label mismatch")
        self._authenticate_registration(registration, vector, params)
        return dict(registration)

    def commit_vector(self, label: str, values: Iterable[int], work_dir: pathlib.Path) -> Dict[str, Any]:
        vector = [to_field_int(value) for value in values]
        n = _require_power_of_two(len(vector), "committed vector length")
        params = _params(n)
        blind = scalar_random(nonzero=True)
        with self._registration_cache_lock:
            self._registration_stats["msm_calls"] = self._registration_stats.get("msm_calls", 0) + 1
            commitment = multi_scalar_mul(params["vector_generators"], vector) + blind * params["blind_generator"]
        registration = {
            "backend": PROTOCOL_ID,
            "parameter_id": params["parameter_id"],
            "label": label,
            "vector_length": n,
            "commitment": commitment.to_json(),
            "blind": str(blind),
        }
        self._authenticate_registration(
            registration, vector, params, known_commitment=commitment
        )
        return registration

    def open_registered(
        self,
        label: str,
        values: Iterable[int],
        point: Iterable[int],
        registration: Mapping[str, Any],
        work_dir: pathlib.Path,
        *,
        transcript: Transcript,
        public_context: Mapping[str, Any] | None = None,
    ) -> HyraxOpeningState:
        ensure_dir(work_dir, label="Hyrax opening work directory")
        vector = [to_field_int(value) for value in values]
        n = _require_power_of_two(len(vector), "committed vector length")
        point_values = [to_field_int(value) for value in point]
        if len(point_values) != log2_power_of_two(n):
            raise ValueError("Hyrax opening point has wrong dimension")
        params = _params(n)
        cached = self._require_cached_registration(registration, vector, params)
        blind = cached.blind
        expected_commitment = cached.commitment
        weights = _eq_table(point_values)
        evaluation = _dot(vector, weights)
        gpu_evaluation = active_mle_fold(
            vector, tuple(reversed(point_values)), f"{label}/hyrax-mle"
        )
        if gpu_evaluation is not None:
            if gpu_evaluation != evaluation:
                raise ValueError("GPU Hyrax MLE fold differs from exact registered evaluation")
            evaluation = gpu_evaluation
        tau = scalar_random(nonzero=False)
        evaluation_commitment = evaluation * params["evaluation_generator"] + tau * params["blind_generator"]
        masked_evaluation_commitment, affine_fields = _prove_affine_mask(
            transcript, label, evaluation_commitment, hyrax_value_params(n)
        )
        a = vector + [(blind - tau) % BN254_FR_MODULUS] + [0] * (n - 1)
        b = weights + [0] * n
        inner_product_generator = -params["evaluation_generator"]
        statement_point = expected_commitment - evaluation_commitment + multi_scalar_mul(params["equality_generators"], b)
        context = dict(public_context or {})
        _statement_transcript(transcript, label, n, expected_commitment, point_values, evaluation_commitment, context)
        ipa, mask_fields = _prove_masked_ipa(
            transcript,
            a,
            b,
            params["ipa_generators"],
            params["equality_generators"],
            inner_product_generator,
            statement_point,
            vector_length=n,
        )
        proof = {
            "protocol": PROTOCOL_ID,
            "parameter_id": params["parameter_id"],
            "vector_length": n,
            # ``evaluation_commitment`` is retained as the historical field
            # name, but now carries the affine-masked point.  The raw point
            # is reconstructed from the two explicit mask commitments and is
            # never serialized.
            "evaluation_commitment": masked_evaluation_commitment.to_json(),
            "ipa": ipa,
            **affine_fields,
            **mask_fields,
            "transcript_digest": transcript.digest.hex(),
        }
        write_json(work_dir / f"{_safe(label)}.hyrax.json", proof)
        state = HyraxOpeningState(
            proof=proof,
            evaluation_commitment=ValueCommitment(evaluation_commitment),
            evaluation=evaluation,
            blind=tau,
        )
        self.opening_log.append({
            "kind": "mle", "label": label,
            "registration_label": registration.get("label"),
            "weights": weights, "state": state,
        })
        return state

    def open_linear_registered(
        self,
        label: str,
        values: Iterable[int],
        weights: Iterable[int],
        registration: Mapping[str, Any],
        work_dir: pathlib.Path,
        *,
        transcript: Transcript,
        linear_form: Mapping[str, Any],
        public_context: Mapping[str, Any] | None = None,
    ) -> HyraxOpeningState:
        """Open an externally specified linear functional of one segment.

        ``weights`` are verifier-reconstructed from ``linear_form`` (for
        example an array slice and local MLE point); they are never selected
        by the proof.  This is the primitive used to bind MatMul/GKR/Lasso
        terminal claims directly to the unique registered W/Aux/L vectors.
        """
        ensure_dir(work_dir, label="Hyrax linear opening work directory")
        vector = [to_field_int(value) for value in values]
        linear_weights = [to_field_int(value) for value in weights]
        n = _require_power_of_two(len(vector), "committed vector length")
        if len(linear_weights) != n:
            raise ValueError("Hyrax linear-form weights have wrong domain")
        params = _params(n)
        cached = self._require_cached_registration(registration, vector, params)
        blind = cached.blind
        expected_commitment = cached.commitment
        evaluation = _dot(vector, linear_weights)
        tau = scalar_random(nonzero=False)
        evaluation_commitment = evaluation * params["evaluation_generator"] + tau * params["blind_generator"]
        masked_evaluation_commitment, affine_fields = _prove_affine_mask(
            transcript, label, evaluation_commitment, hyrax_value_params(n)
        )
        a = vector + [(blind - tau) % BN254_FR_MODULUS] + [0] * (n - 1)
        b = linear_weights + [0] * n
        statement_point = expected_commitment - evaluation_commitment + multi_scalar_mul(params["equality_generators"], b)
        context = dict(public_context or {})
        _linear_statement_transcript(
            transcript,
            label,
            n,
            expected_commitment,
            linear_form,
            evaluation_commitment,
            context,
        )
        ipa, mask_fields = _prove_masked_ipa(
            transcript,
            a,
            b,
            params["ipa_generators"],
            params["equality_generators"],
            -params["evaluation_generator"],
            statement_point,
            vector_length=n,
        )
        proof = {
            "protocol": LINEAR_PROTOCOL_ID,
            "parameter_id": params["parameter_id"],
            "vector_length": n,
            "evaluation_commitment": masked_evaluation_commitment.to_json(),
            "ipa": ipa,
            **affine_fields,
            **mask_fields,
            "transcript_digest": transcript.digest.hex(),
        }
        write_json(work_dir / f"{_safe(label)}.linear-hyrax.json", proof)
        state = HyraxOpeningState(
            proof=proof,
            evaluation_commitment=ValueCommitment(evaluation_commitment),
            evaluation=evaluation,
            blind=tau,
        )
        self.opening_log.append({
            "kind": "linear", "label": label,
            "registration_label": registration.get("label"),
            "weights": linear_weights, "linear_form": dict(linear_form),
            "state": state,
        })
        return state

    def commit_linear_claim_registered(
        self,
        label: str,
        values: Iterable[int],
        weights: Iterable[int] | Mapping[int, int],
        registration: Mapping[str, Any],
        work_dir: pathlib.Path,
        *,
        transcript: Transcript,
        linear_form: Mapping[str, Any],
        public_context: Mapping[str, Any] | None = None,
    ) -> HyraxOpeningState:
        """Commit a linear claim for one later batched input-layer reduction.

        No per-claim IPA is emitted.  The caller must include every returned
        claim in the final committed Sumcheck and canonical W/Aux/L openings;
        a verifier treats this payload as pending until that batch succeeds.
        """
        ensure_dir(work_dir, label="Hyrax batched opening work directory")
        vector = [to_field_int(value) for value in values]
        n = _require_power_of_two(len(vector), "committed vector length")
        if isinstance(weights, Mapping):
            sparse_weights = {
                int(position): to_field_int(value) for position, value in weights.items()
                if to_field_int(value) != 0
            }
            if any(position < 0 or position >= n for position in sparse_weights):
                raise ValueError("batched sparse linear-form weight leaves domain")
            evaluation = sum(vector[position] * value for position, value in sparse_weights.items()) % BN254_FR_MODULUS
            logged_weights: Any = sparse_weights
        else:
            linear_weights = [to_field_int(value) for value in weights]
            if len(linear_weights) != n:
                raise ValueError("batched linear-form weights have wrong domain")
            evaluation = _dot(vector, linear_weights)
            logged_weights = linear_weights
        params = _params(n)
        cached = self._require_cached_registration(registration, vector, params)
        blind = cached.blind
        commitment = cached.commitment
        tau = scalar_random(nonzero=False)
        evaluation_commitment = evaluation * params["evaluation_generator"] + tau * params["blind_generator"]
        masked_evaluation_commitment, affine_fields = _prove_affine_mask(
            transcript, label, evaluation_commitment, hyrax_value_params(n)
        )
        context = dict(public_context or {})
        _batched_linear_statement_transcript(
            transcript, label, n, commitment, linear_form, evaluation_commitment, context,
        )
        proof = {
            "protocol": BATCHED_LINEAR_PROTOCOL_ID,
            "parameter_id": params["parameter_id"],
            "vector_length": n,
            "evaluation_commitment": masked_evaluation_commitment.to_json(),
            **affine_fields,
            "transcript_digest": transcript.digest.hex(),
        }
        state = HyraxOpeningState(
            proof=proof, evaluation_commitment=ValueCommitment(evaluation_commitment),
            evaluation=evaluation, blind=tau,
        )
        self.opening_log.append({
            "kind": "linear", "label": label,
            "registration_label": registration.get("label"),
            "weights": logged_weights, "linear_form": dict(linear_form), "state": state,
        })
        write_json(work_dir / f"{_safe(label)}.batched-linear-claim.json", proof)
        return state

    def commit_open(
        self,
        label: str,
        values: Iterable[int],
        point: Iterable[int],
        work_dir: pathlib.Path,
        *,
        transcript: Transcript,
        public_context: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], HyraxOpeningState]:
        vector = [to_field_int(value) for value in values]
        registration = self.commit_vector(label, vector, work_dir)
        state = self.open_registered(
            label,
            vector,
            point,
            registration,
            work_dir,
            transcript=transcript,
            public_context=public_context,
        )
        return registration, state

    def verify_open(
        self,
        label: str,
        proof_path: pathlib.Path,
        point: Iterable[int],
        expected_commitment: Mapping[str, Any] | G1Point,
        vector_length: int,
        work_dir: pathlib.Path,
        *,
        transcript: Transcript,
        public_context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        ensure_dir(work_dir, label="Hyrax verification work directory")
        result = verify_open_payload(
            label,
            read_json(proof_path),
            point,
            expected_commitment,
            vector_length,
            transcript=transcript,
            public_context=public_context,
        )
        write_json(work_dir / f"{_safe(label)}.verify.json", result)
        return result


def verify_open_payload(
    label: str,
    proof: Mapping[str, Any],
    point: Iterable[int],
    expected_commitment: Mapping[str, Any] | G1Point,
    vector_length: int,
    *,
    transcript: Transcript,
    public_context: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify against an externally authenticated commitment and statement."""
    try:
        if not isinstance(proof, Mapping) or set(proof) != {
            "protocol", "parameter_id", "vector_length", "evaluation_commitment", "ipa",
            "mask_protocol_id", "mask_domain", "mask_version", "mask_commitment",
            "mask_challenge", "transcript_digest", *AFFINE_MASK_FIELDS,
        }:
            return _failed(label, "malformed_proof_shape")
        n = _require_power_of_two(int(vector_length), "Hyrax vector length")
        if proof["protocol"] != PROTOCOL_ID or int(proof["vector_length"]) != n:
            return _failed(label, "protocol_or_domain_mismatch")
        params = _params(n)
        if proof["parameter_id"] != params["parameter_id"]:
            return _failed(label, "parameter_mismatch")
        commitment_value: Any = expected_commitment
        if isinstance(expected_commitment, Mapping) and "commitment" in expected_commitment:
            commitment_value = expected_commitment["commitment"]
        commitment = _point(commitment_value)
        point_values = [to_field_int(value) for value in point]
        if len(point_values) != log2_power_of_two(n):
            return _failed(label, "opening_point_dimension_mismatch")
        # The historical ``evaluation_commitment`` slot now contains the
        # affine-masked evaluation point.  Verify the mask PoK and recover the
        # raw commitment exclusively by subtraction in G1.
        raw_evaluation_commitment, _masked, _mask = _verify_affine_mask(
            transcript, label, proof, hyrax_value_params(n)
        )
        weights = _eq_table(point_values)
        b = weights + [0] * n
        statement_point = commitment - raw_evaluation_commitment.point + multi_scalar_mul(params["equality_generators"], b)
        _statement_transcript(
            transcript,
            label,
            n,
            commitment,
            point_values,
            raw_evaluation_commitment.point,
            dict(public_context or {}),
        )
        ok = _verify_masked_ipa(
            transcript,
            proof,
            b,
            params["ipa_generators"],
            params["equality_generators"],
            -params["evaluation_generator"],
            statement_point,
            vector_length=n,
        )
        digest_ok = proof["transcript_digest"] == transcript.digest.hex()
        return {
            "status": "VERIFY_OK" if ok and digest_ok else "VERIFY_FAILED",
            "backend": PROTOCOL_ID,
            "label": label,
            "expected_commitment_binding_ok": bool(ok),
            "transcript_binding_ok": bool(digest_ok),
            "evaluation_hidden": True,
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _failed(label, "malformed_hyrax_opening")


def verify_linear_payload(
    label: str,
    proof: Mapping[str, Any],
    weights: Iterable[int],
    expected_commitment: Mapping[str, Any] | G1Point,
    vector_length: int,
    *,
    transcript: Transcript,
    linear_form: Mapping[str, Any],
    public_context: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify a linear opening against external layout-derived weights."""
    try:
        if not isinstance(proof, Mapping) or set(proof) != {
            "protocol", "parameter_id", "vector_length", "evaluation_commitment", "ipa",
            "mask_protocol_id", "mask_domain", "mask_version", "mask_commitment",
            "mask_challenge", "transcript_digest", *AFFINE_MASK_FIELDS,
        }:
            return _failed(label, "malformed_linear_proof_shape")
        n = _require_power_of_two(int(vector_length), "Hyrax vector length")
        if proof["protocol"] != LINEAR_PROTOCOL_ID or int(proof["vector_length"]) != n:
            return _failed(label, "linear_protocol_or_domain_mismatch")
        params = _params(n)
        if proof["parameter_id"] != params["parameter_id"]:
            return _failed(label, "linear_parameter_mismatch")
        commitment_value: Any = expected_commitment
        if isinstance(expected_commitment, Mapping) and "commitment" in expected_commitment:
            commitment_value = expected_commitment["commitment"]
        commitment = _point(commitment_value)
        linear_weights = [to_field_int(value) for value in weights]
        if len(linear_weights) != n:
            return _failed(label, "linear_weight_domain_mismatch")
        raw_evaluation_commitment, _masked, _mask = _verify_affine_mask(
            transcript, label, proof, hyrax_value_params(n)
        )
        b = linear_weights + [0] * n
        statement_point = commitment - raw_evaluation_commitment.point + multi_scalar_mul(params["equality_generators"], b)
        _linear_statement_transcript(
            transcript,
            label,
            n,
            commitment,
            linear_form,
            raw_evaluation_commitment.point,
            dict(public_context or {}),
        )
        ok = _verify_masked_ipa(
            transcript,
            proof,
            b,
            params["ipa_generators"],
            params["equality_generators"],
            -params["evaluation_generator"],
            statement_point,
            vector_length=n,
        )
        digest_ok = proof["transcript_digest"] == transcript.digest.hex()
        return {
            "status": "VERIFY_OK" if ok and digest_ok else "VERIFY_FAILED",
            "backend": LINEAR_PROTOCOL_ID,
            "label": label,
            "expected_commitment_binding_ok": bool(ok),
            "transcript_binding_ok": bool(digest_ok),
            "evaluation_hidden": True,
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _failed(label, "malformed_hyrax_linear_opening")


def verify_batched_linear_claim_payload(
    label: str,
    proof: Mapping[str, Any],
    weights: Iterable[int] | Mapping[int, int],
    expected_commitment: Mapping[str, Any] | G1Point,
    vector_length: int,
    *,
    transcript: Transcript,
    linear_form: Mapping[str, Any],
    public_context: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Authenticate a pending claim message for the final batch reduction."""
    try:
        if not isinstance(proof, Mapping) or set(proof) != {
            "protocol", "parameter_id", "vector_length", "evaluation_commitment", "transcript_digest",
            *AFFINE_MASK_FIELDS,
        }:
            return _failed(label, "malformed_batched_linear_claim_shape")
        n = _require_power_of_two(int(vector_length), "Hyrax vector length")
        params = _params(n)
        if (
            proof["protocol"] != BATCHED_LINEAR_PROTOCOL_ID
            or int(proof["vector_length"]) != n
            or proof["parameter_id"] != params["parameter_id"]
        ):
            return _failed(label, "batched_linear_protocol_or_domain_mismatch")
        if isinstance(weights, Mapping):
            linear_weights: Any = {int(position): to_field_int(value) for position, value in weights.items()}
            if any(position < 0 or position >= n for position in linear_weights):
                return _failed(label, "batched_sparse_weight_domain_mismatch")
        else:
            linear_weights = [to_field_int(value) for value in weights]
            if len(linear_weights) != n:
                return _failed(label, "batched_linear_weight_domain_mismatch")
        commitment_value: Any = expected_commitment
        if isinstance(expected_commitment, Mapping) and "commitment" in expected_commitment:
            commitment_value = expected_commitment["commitment"]
        commitment = _point(commitment_value)
        raw_evaluation_commitment, _masked, _mask = _verify_affine_mask(
            transcript, label, proof, hyrax_value_params(n)
        )
        _batched_linear_statement_transcript(
            transcript, label, n, commitment, linear_form,
            raw_evaluation_commitment.point, dict(public_context or {}),
        )
        digest_ok = proof["transcript_digest"] == transcript.digest.hex()
        return {
            "status": "VERIFY_OK" if digest_ok else "VERIFY_FAILED",
            "backend": BATCHED_LINEAR_PROTOCOL_ID,
            "label": label,
            "pending_final_batch_binding": True,
            "transcript_binding_ok": bool(digest_ok),
            "evaluation_hidden": True,
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _failed(label, "malformed_hyrax_batched_linear_claim")


def _failed(label: str, reason: str) -> Dict[str, Any]:
    return {
        "status": "VERIFY_FAILED",
        "backend": PROTOCOL_ID,
        "label": label,
        "reason": reason,
        "expected_commitment_binding_ok": False,
        "transcript_binding_ok": False,
    }


def _safe(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
