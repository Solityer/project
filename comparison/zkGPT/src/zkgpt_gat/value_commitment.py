"""Pedersen value commitments and small Sigma proofs for zkGPT.

The GAT adaptation needs commitments to scalar advice in addition to the
native Hyrax vector commitments.  This module keeps that primitive explicit:

``VC(v;r) = v*U + r*H``

where ``U`` and ``H`` are independent BN254 G1 bases.  Proof objects contain
only group points, Fiat--Shamir challenges, and Schnorr responses; openings
and witness coefficients remain prover-local.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .bn254 import BN254_FQ_MODULUS, G1Point, hash_point, scalar_random
from .field import BN254_FR_MODULUS
from .transcript import (
    EncodingError,
    Transcript,
    decode_point,
    decode_scalar,
    encode_point,
    encode_scalar,
)


class CommitmentError(ValueError):
    """Raised for malformed commitments or invalid witness inputs."""


def _strict_scalar(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CommitmentError("scalar must be an integer")
    if value < 0 or value >= BN254_FR_MODULUS:
        raise CommitmentError("scalar is outside canonical Fr")
    return value


def _wire_scalar(value: int) -> str:
    return encode_scalar(_strict_scalar(value)).hex()


def _parse_wire_scalar(value: Any) -> int:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise CommitmentError("scalar wire value must be a 64-character hex string")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise CommitmentError("invalid scalar hex") from exc
    if raw.hex() != value.lower():
        raise CommitmentError("scalar hex is not canonical lowercase")
    return decode_scalar(raw)


def _wire_point(point: G1Point) -> str:
    return encode_point(_check_point(point)).hex()


def _parse_wire_point(value: Any) -> G1Point:
    if not isinstance(value, str):
        raise CommitmentError("point wire value must be hex text")
    if value.lower() != value or len(value) not in (2, 130):
        raise CommitmentError("point wire value is not canonical")
    try:
        point = decode_point(bytes.fromhex(value))
    except (ValueError, EncodingError) as exc:
        raise CommitmentError("invalid point wire value") from exc
    return point


def _check_point(point: G1Point) -> G1Point:
    if not isinstance(point, G1Point) or not point.is_on_curve():
        raise CommitmentError("invalid BN254 G1 point")
    if point.infinity:
        if point.x != 0 or point.y != 0:
            raise CommitmentError("non-canonical infinity coordinates")
        return point
    if not (0 <= point.x < BN254_FQ_MODULUS):
        raise CommitmentError("non-canonical point x coordinate")
    if not (0 <= point.y < BN254_FQ_MODULUS):
        raise CommitmentError("non-canonical point y coordinate")
    return point


@dataclass(frozen=True)
class PedersenParams:
    U: G1Point
    H: G1Point
    parameter_id: str

    def __post_init__(self) -> None:
        u = _check_point(self.U)
        h = _check_point(self.H)
        if u.infinity or h.infinity or u == h or u == -h:
            raise CommitmentError("Pedersen bases must be distinct nonzero points")

    @classmethod
    def derive(cls, label: str) -> "PedersenParams":
        if not isinstance(label, str) or not label:
            raise CommitmentError("parameter label must be non-empty text")
        # U is the canonical BN254 generator used by the native code.  H is
        # an independent hash-to-G1 base, not a known scalar multiple chosen
        # by the caller.
        u = G1Point.generator()
        h = hash_point("zkGPT/BN254/Pedersen/H", {"label": label})
        if h == u or h == -u:
            raise CommitmentError("derived Pedersen bases collided")
        return cls(u, h, label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_id": self.parameter_id,
            "U": _wire_point(self.U),
            "H": _wire_point(self.H),
            # The modulus is a parameter, not an Fr element, so it is
            # represented as its canonical decimal parameter string.
            "field_modulus": str(BN254_FR_MODULUS),
        }

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        expected: "PedersenParams | None" = None,
        allow_derived: bool = False,
    ) -> "PedersenParams":
        if not isinstance(value, dict):
            raise CommitmentError("invalid Pedersen parameter object")
        if expected is not None:
            _check_params(expected)
            if value != expected.to_dict():
                raise CommitmentError("parameter wire does not match authenticated external parameters")
            return expected
        if not allow_derived:
            raise CommitmentError("formal parameter parsing requires externally authenticated params")
        # A serialized parameter object is never allowed to choose arbitrary
        # bases.  It may only name a deterministic setup profile; the points
        # must exactly equal the profile's derived bases.
        parameter_id = value.get("parameter_id")
        if not isinstance(parameter_id, str) or not parameter_id:
            raise CommitmentError("missing Pedersen parameter id")
        derived = cls.derive(parameter_id)
        if value != derived.to_dict():
            raise CommitmentError("parameter points do not match deterministic setup")
        return derived


@dataclass(frozen=True)
class ValueCommitment:
    point: G1Point

    def __post_init__(self) -> None:
        _check_point(self.point)

    def to_dict(self) -> dict[str, str]:
        return {"point": _wire_point(self.point)}

    @classmethod
    def from_dict(cls, value: Any) -> "ValueCommitment":
        if not isinstance(value, dict) or set(value) != {"point"}:
            raise CommitmentError("invalid value commitment encoding")
        return cls(_parse_wire_point(value["point"]))


@dataclass(frozen=True)
class RepresentationProof:
    announcement: G1Point
    challenge: int
    z_u: int
    z_h: int

    def to_dict(self) -> dict[str, str]:
        return {
            "announcement": _wire_point(self.announcement),
            "challenge": _wire_scalar(self.challenge),
            "z_u": _wire_scalar(self.z_u),
            "z_h": _wire_scalar(self.z_h),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "RepresentationProof":
        if not isinstance(value, dict) or set(value) != {"announcement", "challenge", "z_u", "z_h"}:
            raise CommitmentError("invalid representation proof encoding")
        return cls(
            _parse_wire_point(value["announcement"]),
            _parse_wire_scalar(value["challenge"]),
            _parse_wire_scalar(value["z_u"]),
            _parse_wire_scalar(value["z_h"]),
        )


@dataclass(frozen=True)
class SameValueProof:
    announcement: G1Point
    challenge: int
    z_h: int

    def to_dict(self) -> dict[str, str]:
        return {
            "announcement": _wire_point(self.announcement),
            "challenge": _wire_scalar(self.challenge),
            "z_h": _wire_scalar(self.z_h),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SameValueProof":
        if not isinstance(value, dict) or set(value) != {"announcement", "challenge", "z_h"}:
            raise CommitmentError("invalid same-value proof encoding")
        return cls(
            _parse_wire_point(value["announcement"]),
            _parse_wire_scalar(value["challenge"]),
            _parse_wire_scalar(value["z_h"]),
        )


@dataclass(frozen=True)
class CrossParamsSameValueProof:
    first_announcement: G1Point
    second_announcement: G1Point
    challenge: int
    z_value: int
    z_first_blind: int
    z_second_blind: int

    def to_dict(self) -> dict[str, str]:
        return {
            "first_announcement": _wire_point(self.first_announcement),
            "second_announcement": _wire_point(self.second_announcement),
            "challenge": _wire_scalar(self.challenge),
            "z_value": _wire_scalar(self.z_value),
            "z_first_blind": _wire_scalar(self.z_first_blind),
            "z_second_blind": _wire_scalar(self.z_second_blind),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "CrossParamsSameValueProof":
        keys = {"first_announcement", "second_announcement", "challenge", "z_value", "z_first_blind", "z_second_blind"}
        if not isinstance(value, dict) or set(value) != keys:
            raise CommitmentError("invalid cross-parameter same-value proof encoding")
        return cls(
            _parse_wire_point(value["first_announcement"]),
            _parse_wire_point(value["second_announcement"]),
            _parse_wire_scalar(value["challenge"]),
            _parse_wire_scalar(value["z_value"]),
            _parse_wire_scalar(value["z_first_blind"]),
            _parse_wire_scalar(value["z_second_blind"]),
        )


@dataclass(frozen=True)
class MultiplicationProof:
    a_announcement: G1Point
    b_announcement: G1Point
    product_announcement: G1Point
    cross_announcement: G1Point
    challenge: int
    z_a: int
    z_b: int
    z_ra: int
    z_rb: int
    z_cross_h: int
    z_product_h: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "a_announcement": _wire_point(self.a_announcement),
            "b_announcement": _wire_point(self.b_announcement),
            "product_announcement": _wire_point(self.product_announcement),
            "cross_announcement": _wire_point(self.cross_announcement),
            "challenge": _wire_scalar(self.challenge),
            "z_a": _wire_scalar(self.z_a),
            "z_b": _wire_scalar(self.z_b),
            "z_ra": _wire_scalar(self.z_ra),
            "z_rb": _wire_scalar(self.z_rb),
            "z_cross_h": _wire_scalar(self.z_cross_h),
            "z_product_h": _wire_scalar(self.z_product_h),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "MultiplicationProof":
        keys = {
            "a_announcement", "b_announcement", "product_announcement", "cross_announcement",
            "challenge", "z_a", "z_b", "z_ra", "z_rb", "z_cross_h", "z_product_h",
        }
        if not isinstance(value, dict) or set(value) != keys:
            raise CommitmentError("invalid multiplication proof encoding")
        return cls(
            _parse_wire_point(value["a_announcement"]),
            _parse_wire_point(value["b_announcement"]),
            _parse_wire_point(value["product_announcement"]),
            _parse_wire_point(value["cross_announcement"]),
            _parse_wire_scalar(value["challenge"]),
            _parse_wire_scalar(value["z_a"]),
            _parse_wire_scalar(value["z_b"]),
            _parse_wire_scalar(value["z_ra"]),
            _parse_wire_scalar(value["z_rb"]),
            _parse_wire_scalar(value["z_cross_h"]),
            _parse_wire_scalar(value["z_product_h"]),
        )


def commit_value(params: PedersenParams, value: int, blind: int | None = None) -> tuple[ValueCommitment, int]:
    """Create ``VC(value; blind)`` and return the opening only to the caller."""

    _check_params(params)
    value = _strict_scalar(value)
    if blind is None:
        blind = scalar_random(nonzero=False)
    blind = _strict_scalar(blind)
    return ValueCommitment(value * params.U + blind * params.H), blind


def _check_params(params: PedersenParams) -> None:
    if not isinstance(params, PedersenParams):
        raise CommitmentError("invalid Pedersen parameters")
    _check_point(params.U)
    _check_point(params.H)


def _require_transcript(transcript: Transcript) -> Transcript:
    if not isinstance(transcript, Transcript):
        raise CommitmentError("formal proof APIs require one external Transcript")
    return transcript


def _representation_statement(t: Transcript, label: str, params: PedersenParams, commitment: ValueCommitment, announcement: G1Point) -> int:
    t.absorb(f"{label}/statement", {"params": params.to_dict(), "commitment": commitment.point, "announcement": announcement})
    return t.challenge_scalar(f"{label}/challenge", nonzero=True)


def prove_representation(
    params: PedersenParams,
    commitment: ValueCommitment,
    value: int,
    blind: int,
    *,
    label: str,
    transcript: Transcript,
) -> RepresentationProof:
    _check_params(params)
    value = _strict_scalar(value)
    blind = _strict_scalar(blind)
    _check_point(commitment.point)
    if commitment.point != value * params.U + blind * params.H:
        raise CommitmentError("opening does not match commitment")
    t = _require_transcript(transcript)
    alpha = scalar_random(nonzero=True)
    beta = scalar_random(nonzero=True)
    announcement = alpha * params.U + beta * params.H
    challenge = _representation_statement(t, label, params, commitment, announcement)
    proof = RepresentationProof(
        announcement,
        challenge,
        (alpha + challenge * value) % BN254_FR_MODULUS,
        (beta + challenge * blind) % BN254_FR_MODULUS,
    )
    t.absorb(f"{label}/response", {"z_u": proof.z_u, "z_h": proof.z_h})
    return proof


def verify_representation(
    params: PedersenParams,
    commitment: ValueCommitment,
    proof: RepresentationProof | dict[str, Any],
    *,
    label: str,
    transcript: Transcript,
) -> bool:
    try:
        _check_params(params)
        _check_point(commitment.point)
        proof = RepresentationProof.from_dict(proof) if isinstance(proof, dict) else proof
        if not isinstance(proof, RepresentationProof):
            return False
        _check_point(proof.announcement)
        if not all(0 <= item < BN254_FR_MODULUS for item in (proof.challenge, proof.z_u, proof.z_h)):
            return False
        t = _require_transcript(transcript)
        expected = _representation_statement(t, label, params, commitment, proof.announcement)
        if proof.challenge != expected:
            return False
        t.absorb(f"{label}/response", {"z_u": proof.z_u, "z_h": proof.z_h})
        return proof.z_u * params.U + proof.z_h * params.H == proof.announcement + expected * commitment.point
    except (CommitmentError, EncodingError, KeyError, TypeError, ValueError):
        return False


def prove_same_value(
    params: PedersenParams,
    first: ValueCommitment,
    second: ValueCommitment,
    first_blind: int,
    second_blind: int,
    *,
    label: str,
    transcript: Transcript,
) -> SameValueProof:
    _check_params(params)
    _check_point(first.point)
    _check_point(second.point)
    first_blind = _strict_scalar(first_blind)
    second_blind = _strict_scalar(second_blind)
    difference = first.point - second.point
    delta = (first_blind - second_blind) % BN254_FR_MODULUS
    if difference != delta * params.H:
        raise CommitmentError("same-value openings do not match")
    t = _require_transcript(transcript)
    nonce = scalar_random(nonzero=True)
    announcement = nonce * params.H
    t.absorb(f"{label}/statement", {"params": params.to_dict(), "first": first.point, "second": second.point, "announcement": announcement})
    challenge = t.challenge_scalar(f"{label}/challenge", nonzero=True)
    proof = SameValueProof(announcement, challenge, (nonce + challenge * delta) % BN254_FR_MODULUS)
    t.absorb(f"{label}/response", {"z_h": proof.z_h})
    return proof


def verify_same_value(
    params: PedersenParams,
    first: ValueCommitment,
    second: ValueCommitment,
    proof: SameValueProof | dict[str, Any],
    *,
    label: str,
    transcript: Transcript,
) -> bool:
    try:
        _check_params(params)
        _check_point(first.point)
        _check_point(second.point)
        proof = SameValueProof.from_dict(proof) if isinstance(proof, dict) else proof
        if not isinstance(proof, SameValueProof):
            return False
        _check_point(proof.announcement)
        if not all(0 <= item < BN254_FR_MODULUS for item in (proof.challenge, proof.z_h)):
            return False
        t = _require_transcript(transcript)
        t.absorb(f"{label}/statement", {"params": params.to_dict(), "first": first.point, "second": second.point, "announcement": proof.announcement})
        expected = t.challenge_scalar(f"{label}/challenge", nonzero=True)
        if proof.challenge != expected:
            return False
        t.absorb(f"{label}/response", {"z_h": proof.z_h})
        return proof.z_h * params.H == proof.announcement + expected * (first.point - second.point)
    except (CommitmentError, EncodingError, KeyError, TypeError, ValueError):
        return False


def _cross_params_statement(
    transcript: Transcript,
    label: str,
    first_params: PedersenParams,
    first: ValueCommitment,
    second_params: PedersenParams,
    second: ValueCommitment,
    first_announcement: G1Point,
    second_announcement: G1Point,
) -> int:
    transcript.absorb(f"{label}/statement", {
        "first_params": first_params.to_dict(), "first": first.point,
        "second_params": second_params.to_dict(), "second": second.point,
        "first_announcement": first_announcement,
        "second_announcement": second_announcement,
    })
    return transcript.challenge_scalar(f"{label}/challenge", nonzero=True)


def prove_cross_params_same_value(
    first_params: PedersenParams,
    first: ValueCommitment,
    second_params: PedersenParams,
    second: ValueCommitment,
    value: int,
    first_blind: int,
    second_blind: int,
    *,
    label: str,
    transcript: Transcript,
) -> CrossParamsSameValueProof:
    for params in (first_params, second_params):
        _check_params(params)
    value = _strict_scalar(value); first_blind = _strict_scalar(first_blind); second_blind = _strict_scalar(second_blind)
    if first.point != value * first_params.U + first_blind * first_params.H:
        raise CommitmentError("first cross-parameter opening mismatch")
    if second.point != value * second_params.U + second_blind * second_params.H:
        raise CommitmentError("second cross-parameter opening mismatch")
    nonce_value = scalar_random(nonzero=True)
    nonce_first = scalar_random(nonzero=True)
    nonce_second = scalar_random(nonzero=True)
    first_announcement = nonce_value * first_params.U + nonce_first * first_params.H
    second_announcement = nonce_value * second_params.U + nonce_second * second_params.H
    t = _require_transcript(transcript)
    challenge = _cross_params_statement(t, label, first_params, first, second_params, second, first_announcement, second_announcement)
    proof = CrossParamsSameValueProof(
        first_announcement, second_announcement, challenge,
        (nonce_value + challenge * value) % BN254_FR_MODULUS,
        (nonce_first + challenge * first_blind) % BN254_FR_MODULUS,
        (nonce_second + challenge * second_blind) % BN254_FR_MODULUS,
    )
    t.absorb(f"{label}/response", {"z_value": proof.z_value, "z_first_blind": proof.z_first_blind, "z_second_blind": proof.z_second_blind})
    return proof


def verify_cross_params_same_value(
    first_params: PedersenParams,
    first: ValueCommitment,
    second_params: PedersenParams,
    second: ValueCommitment,
    proof: CrossParamsSameValueProof | dict[str, Any],
    *,
    label: str,
    transcript: Transcript,
) -> bool:
    try:
        for params in (first_params, second_params): _check_params(params)
        proof = CrossParamsSameValueProof.from_dict(proof) if isinstance(proof, dict) else proof
        if not isinstance(proof, CrossParamsSameValueProof): return False
        t = _require_transcript(transcript)
        challenge = _cross_params_statement(t, label, first_params, first, second_params, second, proof.first_announcement, proof.second_announcement)
        if proof.challenge != challenge: return False
        t.absorb(f"{label}/response", {"z_value": proof.z_value, "z_first_blind": proof.z_first_blind, "z_second_blind": proof.z_second_blind})
        return (
            proof.z_value * first_params.U + proof.z_first_blind * first_params.H
            == proof.first_announcement + challenge * first.point
            and proof.z_value * second_params.U + proof.z_second_blind * second_params.H
            == proof.second_announcement + challenge * second.point
        )
    except (CommitmentError, EncodingError, KeyError, TypeError, ValueError):
        return False


def _multiplication_statement(
    t: Transcript,
    label: str,
    params: PedersenParams,
    a_commitment: ValueCommitment,
    b_commitment: ValueCommitment,
    product_commitment: ValueCommitment,
    a_announcement: G1Point,
    b_announcement: G1Point,
    product_announcement: G1Point,
    cross_announcement: G1Point,
) -> int:
    t.absorb(
        f"{label}/statement",
        {
            "params": params.to_dict(),
            "a": a_commitment.point,
            "b": b_commitment.point,
            "product": product_commitment.point,
            "a_announcement": a_announcement,
            "b_announcement": b_announcement,
            "product_announcement": product_announcement,
            "cross_announcement": cross_announcement,
        },
    )
    return t.challenge_scalar(f"{label}/challenge", nonzero=True)


def prove_multiplication(
    params: PedersenParams,
    a_commitment: ValueCommitment,
    b_commitment: ValueCommitment,
    product_commitment: ValueCommitment,
    a: int,
    b: int,
    product: int,
    a_blind: int,
    b_blind: int,
    product_blind: int,
    *,
    label: str,
    transcript: Transcript,
) -> MultiplicationProof:
    _check_params(params)
    a, b, product = _strict_scalar(a), _strict_scalar(b), _strict_scalar(product)
    ra, rb, rc = _strict_scalar(a_blind), _strict_scalar(b_blind), _strict_scalar(product_blind)
    for commitment in (a_commitment, b_commitment, product_commitment):
        _check_point(commitment.point)
    if product != (a * b) % BN254_FR_MODULUS:
        raise CommitmentError("product witness is not a*b")
    if a_commitment.point != a * params.U + ra * params.H or b_commitment.point != b * params.U + rb * params.H:
        raise CommitmentError("multiplication input opening mismatch")
    if product_commitment.point != product * params.U + rc * params.H:
        raise CommitmentError("multiplication output opening mismatch")
    alpha, beta = scalar_random(nonzero=True), scalar_random(nonzero=True)
    r_alpha = scalar_random(nonzero=True)
    r_beta = scalar_random(nonzero=True)
    r_product = scalar_random(nonzero=True)
    a_announcement = alpha * params.U + r_alpha * params.H
    b_announcement = beta * params.U + r_beta * params.H
    product_announcement = (alpha * beta) * params.U + r_product * params.H
    # The cross announcement is B^alpha A^beta.  Its H coefficient is
    # alpha*rb + beta*ra, but that scalar is not emitted.
    cross_announcement = alpha * b_commitment.point + beta * a_commitment.point
    t = _require_transcript(transcript)
    challenge = _multiplication_statement(
        t,
        label,
        params,
        a_commitment,
        b_commitment,
        product_commitment,
        a_announcement,
        b_announcement,
        product_announcement,
        cross_announcement,
    )
    e2 = (challenge * challenge) % BN254_FR_MODULUS
    # The cross equation uses the witness-dependent H coefficient
    # a*rb + b*ra - 2*rc (the alpha/beta terms are already in the
    # cross-announcement itself).
    cross_blind = (a * rb + b * ra - 2 * rc) % BN254_FR_MODULUS
    product_blind_response = (
        r_product
        + challenge * (alpha * rb + beta * ra)
        + e2 * rc
    ) % BN254_FR_MODULUS
    proof = MultiplicationProof(
        a_announcement,
        b_announcement,
        product_announcement,
        cross_announcement,
        challenge,
        (alpha + challenge * a) % BN254_FR_MODULUS,
        (beta + challenge * b) % BN254_FR_MODULUS,
        (r_alpha + challenge * ra) % BN254_FR_MODULUS,
        (r_beta + challenge * rb) % BN254_FR_MODULUS,
        (challenge * cross_blind) % BN254_FR_MODULUS,
        product_blind_response,
    )
    t.absorb(
        f"{label}/response",
        {
            "z_a": proof.z_a,
            "z_b": proof.z_b,
            "z_ra": proof.z_ra,
            "z_rb": proof.z_rb,
            "z_cross_h": proof.z_cross_h,
            "z_product_h": proof.z_product_h,
        },
    )
    return proof


def verify_multiplication(
    params: PedersenParams,
    a_commitment: ValueCommitment,
    b_commitment: ValueCommitment,
    product_commitment: ValueCommitment,
    proof: MultiplicationProof | dict[str, Any],
    *,
    label: str,
    transcript: Transcript,
) -> bool:
    try:
        _check_params(params)
        for commitment in (a_commitment, b_commitment, product_commitment):
            _check_point(commitment.point)
        proof = MultiplicationProof.from_dict(proof) if isinstance(proof, dict) else proof
        if not isinstance(proof, MultiplicationProof):
            return False
        for point in (proof.a_announcement, proof.b_announcement, proof.product_announcement, proof.cross_announcement):
            _check_point(point)
        scalars = (
            proof.challenge, proof.z_a, proof.z_b, proof.z_ra, proof.z_rb,
            proof.z_cross_h, proof.z_product_h,
        )
        if not all(0 <= item < BN254_FR_MODULUS for item in scalars):
            return False
        t = _require_transcript(transcript)
        expected = _multiplication_statement(
            t,
            label,
            params,
            a_commitment,
            b_commitment,
            product_commitment,
            proof.a_announcement,
            proof.b_announcement,
            proof.product_announcement,
            proof.cross_announcement,
        )
        if expected != proof.challenge:
            return False
        t.absorb(
            f"{label}/response",
            {
                "z_a": proof.z_a,
                "z_b": proof.z_b,
                "z_ra": proof.z_ra,
                "z_rb": proof.z_rb,
                "z_cross_h": proof.z_cross_h,
                "z_product_h": proof.z_product_h,
            },
        )
        e = expected
        e2 = (e * e) % BN254_FR_MODULUS
        repr_a = proof.z_a * params.U + proof.z_ra * params.H == proof.a_announcement + e * a_commitment.point
        repr_b = proof.z_b * params.U + proof.z_rb * params.H == proof.b_announcement + e * b_commitment.point
        cross = (
            proof.z_a * b_commitment.point + proof.z_b * a_commitment.point
            == proof.cross_announcement + (2 * e) * product_commitment.point + proof.z_cross_h * params.H
        )
        product = (
            (proof.z_a * proof.z_b) * params.U + proof.z_product_h * params.H
            == proof.product_announcement + e * proof.cross_announcement + e2 * product_commitment.point
        )
        return bool(repr_a and repr_b and cross and product)
    except (CommitmentError, EncodingError, KeyError, TypeError, ValueError):
        return False


# Protocol-facing aliases used by adapters that call these relations "PoK".
pedersen_commit = commit_value
prove_representation_pok = prove_representation
verify_representation_pok = verify_representation
prove_same_value_pok = prove_same_value
verify_same_value_pok = verify_same_value
prove_multiplication_sigma = prove_multiplication
verify_multiplication_sigma = verify_multiplication
