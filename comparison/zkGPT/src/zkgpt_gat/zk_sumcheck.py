"""Committed-coefficient Sumcheck primitives native to the zkGPT baseline.

This module verifies *round consistency* only.  A Sumcheck terminal claim is
still an application-level claim: the caller must bind it to the surrounding
GKR/MatMul/Lasso relation or to a separate opening proof.  The explicit
``VerifierState.terminal_bound`` bit and ``bind_terminal_claim`` helper keep
that boundary visible in the API.

Every formal call receives one externally authenticated :class:`PedersenParams`
and one mutable session :class:`Transcript`.  No per-label setup or private
transcript is silently created.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Sequence

from .bn254 import G1Point
from .field import BN254_FR_MODULUS
from .transcript import EncodingError, Transcript, decode_point, decode_scalar, encode_point, encode_scalar
from .value_commitment import (
    CommitmentError,
    PedersenParams,
    RepresentationProof,
    ValueCommitment,
    commit_value,
    prove_representation,
    verify_representation,
)


COMMITTED_SUMCHECK_PROTOCOL = "zkGPT/committed-coefficient-sumcheck-v1"


class SumcheckVerificationError(CommitmentError):
    """Raised when the committed round relation or terminal binding fails."""


def _scalar(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= BN254_FR_MODULUS:
        raise CommitmentError("non-canonical Sumcheck scalar")
    return value


def _scalar_wire(value: int) -> str:
    return encode_scalar(_scalar(value)).hex()


def _parse_scalar(value: Any) -> int:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise CommitmentError("non-canonical Sumcheck scalar wire value")
    try:
        return decode_scalar(bytes.fromhex(value))
    except (ValueError, EncodingError) as exc:
        raise CommitmentError("invalid Sumcheck scalar wire value") from exc


def _point_wire(point: G1Point) -> str:
    return encode_point(point).hex()


def _parse_point(value: Any) -> G1Point:
    if not isinstance(value, str) or value.lower() != value or len(value) not in (2, 130):
        raise CommitmentError("non-canonical Sumcheck point wire value")
    try:
        return decode_point(bytes.fromhex(value))
    except (ValueError, EncodingError) as exc:
        raise CommitmentError("invalid Sumcheck point wire value") from exc


@dataclass(frozen=True)
class CommittedSumcheckProof:
    """Serializable public messages; no coefficient, value, or opening is stored."""

    protocol: str
    variable_count: int
    degree: int
    initial_claim_commitment: G1Point
    rounds: tuple[dict[str, Any], ...]
    terminal_claim_commitment: G1Point

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "field_modulus": str(BN254_FR_MODULUS),
            "variable_count": self.variable_count,
            "degree": self.degree,
            "initial_claim_commitment": _point_wire(self.initial_claim_commitment),
            "rounds": [dict(item) for item in self.rounds],
            "terminal_claim_commitment": _point_wire(self.terminal_claim_commitment),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "CommittedSumcheckProof":
        if not isinstance(value, dict):
            raise CommitmentError("Sumcheck proof must be an object")
        required = {
            "protocol", "field_modulus", "variable_count", "degree",
            "initial_claim_commitment", "rounds", "terminal_claim_commitment",
        }
        if set(value) != required:
            raise CommitmentError("Sumcheck proof has missing or extra fields")
        if value["protocol"] != COMMITTED_SUMCHECK_PROTOCOL or value["field_modulus"] != str(BN254_FR_MODULUS):
            raise CommitmentError("wrong Sumcheck protocol or scalar field")
        variable_count = value["variable_count"]
        degree = value["degree"]
        if isinstance(variable_count, bool) or not isinstance(variable_count, int) or variable_count <= 0:
            raise CommitmentError("invalid Sumcheck variable count")
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
            raise CommitmentError("invalid Sumcheck degree")
        rounds = value["rounds"]
        if not isinstance(rounds, list):
            raise CommitmentError("Sumcheck rounds must be a list")
        return cls(
            COMMITTED_SUMCHECK_PROTOCOL,
            variable_count,
            degree,
            _parse_point(value["initial_claim_commitment"]),
            tuple(dict(item) for item in rounds),
            _parse_point(value["terminal_claim_commitment"]),
        )


@dataclass(frozen=True)
class ProverState:
    """Non-serializable prover-local state returned with a proof."""

    proof: CommittedSumcheckProof
    terminal_value: int
    blind: int
    terminal_commitment: ValueCommitment
    challenges: tuple[int, ...]


@dataclass(frozen=True)
class VerifierState:
    """Round-verifier state; terminal binding is intentionally separate."""

    terminal_claim: ValueCommitment
    challenges: tuple[int, ...]
    round_consistent: bool = True
    terminal_bound: bool = False

    @property
    def terminal_commitment(self) -> ValueCommitment:
        return self.terminal_claim


TerminalPredicate = Callable[[ValueCommitment, tuple[int, ...]], bool]


def _require_params(params: PedersenParams) -> PedersenParams:
    if not isinstance(params, PedersenParams):
        raise CommitmentError("formal Sumcheck APIs require external PedersenParams")
    return params


def _require_transcript(transcript: Transcript) -> Transcript:
    if not isinstance(transcript, Transcript):
        raise CommitmentError("formal Sumcheck APIs require one external Transcript")
    return transcript


def _initial_prover_state(
    initial_claim: int | ValueCommitment,
    params: PedersenParams,
    initial_value: int | None,
    initial_blind: int | None,
) -> tuple[ValueCommitment, int, int]:
    if isinstance(initial_claim, bool):
        raise CommitmentError("initial claim must be Fr scalar or ValueCommitment")
    if isinstance(initial_claim, int):
        if initial_value is not None or initial_blind is not None:
            raise CommitmentError("public initial claim cannot also carry a private opening")
        value = _scalar(initial_claim)
        return ValueCommitment(value * params.U), value, 0
    if not isinstance(initial_claim, ValueCommitment):
        raise CommitmentError("invalid initial claim")
    if initial_value is None or initial_blind is None:
        raise CommitmentError("witness initial commitment requires value and blind locally")
    value = _scalar(initial_value)
    blind = _scalar(initial_blind)
    if initial_claim.point != value * params.U + blind * params.H:
        raise CommitmentError("initial commitment opening does not match")
    return initial_claim, value, blind


def _initial_verifier_commitment(initial_claim: int | ValueCommitment, params: PedersenParams) -> ValueCommitment:
    if isinstance(initial_claim, bool):
        raise CommitmentError("initial claim must be Fr scalar or ValueCommitment")
    if isinstance(initial_claim, int):
        return ValueCommitment(_scalar(initial_claim) * params.U)
    if isinstance(initial_claim, ValueCommitment):
        return initial_claim
    raise CommitmentError("invalid initial claim")


def _absorb_statement(
    transcript: Transcript,
    label: str,
    params: PedersenParams,
    initial_commitment: ValueCommitment,
    variable_count: int,
    degree: int,
) -> None:
    transcript.absorb(
        "sumcheck/statement",
        {
            "label": label,
            "parameter_id": params.parameter_id,
            "params": params.to_dict(),
            "initial_claim_commitment": initial_commitment.point,
            "variable_count": variable_count,
            "degree": degree,
        },
    )


def _evaluate_commitments(commitments: Sequence[G1Point], challenge: int) -> G1Point:
    result = G1Point.zero()
    power = 1
    for point in commitments:
        result = result + power * point
        power = (power * challenge) % BN254_FR_MODULUS
    return result


def _sum_commitments(commitments: Sequence[G1Point]) -> G1Point:
    result = G1Point.zero()
    for point in commitments:
        result = result + point
    return result


def prove_committed_sumcheck(
    label: str,
    initial_claim: int | ValueCommitment,
    coefficient_rounds: Sequence[Sequence[int]] | Callable[[int, int, Transcript], Sequence[int]],
    degree: int,
    *,
    params: PedersenParams,
    transcript: Transcript,
    variable_count: Optional[int] = None,
    initial_value: int | None = None,
    initial_blind: int | None = None,
) -> ProverState:
    """Generate public round messages and return private terminal state."""

    if not isinstance(label, str) or not label:
        raise CommitmentError("Sumcheck label must be non-empty text")
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise CommitmentError("Sumcheck degree must be non-negative")
    params = _require_params(params)
    transcript = _require_transcript(transcript)
    if callable(coefficient_rounds):
        if variable_count is None or isinstance(variable_count, bool) or not isinstance(variable_count, int):
            raise CommitmentError("a dynamic Sumcheck provider requires variable_count")
        variable_count = int(variable_count)
        provider = coefficient_rounds
    else:
        if variable_count is not None and variable_count != len(coefficient_rounds):
            raise CommitmentError("variable_count disagrees with coefficient round list")
        variable_count = len(coefficient_rounds)
        provider = None
    if variable_count <= 0:
        raise CommitmentError("committed Sumcheck cannot have zero rounds")

    initial_commitment, current_claim, current_blind = _initial_prover_state(
        initial_claim, params, initial_value, initial_blind
    )
    _absorb_statement(transcript, label, params, initial_commitment, variable_count, degree)
    current_point = initial_commitment.point
    rounds: list[dict[str, Any]] = []
    challenges: list[int] = []

    for round_index in range(variable_count):
        raw_coefficients = provider(round_index, current_claim, transcript) if provider is not None else coefficient_rounds[round_index]
        coefficients = [_scalar(int(value)) for value in raw_coefficients]
        if len(coefficients) != degree + 1:
            raise CommitmentError("each Sumcheck round must use the exact public coefficient width")
        if (coefficients[0] + sum(coefficients)) % BN254_FR_MODULUS != current_claim:
            raise CommitmentError("round polynomial does not satisfy the current claim")

        commitments: list[ValueCommitment] = []
        blinds: list[int] = []
        poks: list[RepresentationProof] = []
        for coefficient_index, coefficient in enumerate(coefficients):
            commitment, blind = commit_value(params, coefficient)
            commitments.append(commitment)
            blinds.append(blind)
            poks.append(
                prove_representation(
                    params,
                    commitment,
                    coefficient,
                    blind,
                    label=f"{label}/round/{round_index}/coefficient/{coefficient_index}",
                    transcript=transcript,
                )
            )

        equality_point = _sum_commitments([item.point for item in commitments]) - current_point + commitments[0].point
        equality_blind = (sum(blinds) + blinds[0] - current_blind) % BN254_FR_MODULUS
        equality = prove_representation(
            params,
            ValueCommitment(equality_point),
            0,
            equality_blind,
            label=f"{label}/round/{round_index}/group-equality",
            transcript=transcript,
        )

        challenge = transcript.challenge_scalar(f"{label}/round/{round_index}/challenge", nonzero=True)
        challenges.append(challenge)
        current_point = _evaluate_commitments([item.point for item in commitments], challenge)
        current_blind = 0
        power = 1
        for blind in blinds:
            current_blind = (current_blind + power * blind) % BN254_FR_MODULUS
            power = (power * challenge) % BN254_FR_MODULUS
        current_claim = sum(
            coefficient * pow(challenge, coefficient_index, BN254_FR_MODULUS)
            for coefficient_index, coefficient in enumerate(coefficients)
        ) % BN254_FR_MODULUS
        advance = getattr(provider, "advance", None) if provider is not None else None
        if advance is not None:
            advance(challenge)
        rounds.append(
            {
                "round": round_index,
                "coefficient_commitments": [item.to_dict() for item in commitments],
                "coefficient_poks": [item.to_dict() for item in poks],
                "group_equality_pok": equality.to_dict(),
                "challenge": _scalar_wire(challenge),
            }
        )

    proof = CommittedSumcheckProof(
        COMMITTED_SUMCHECK_PROTOCOL,
        variable_count,
        degree,
        initial_commitment.point,
        tuple(rounds),
        current_point,
    )
    return ProverState(
        proof=proof,
        terminal_value=current_claim,
        blind=current_blind,
        terminal_commitment=ValueCommitment(current_point),
        challenges=tuple(challenges),
    )


def _terminal_predicate_result(
    state: VerifierState,
    terminal_predicate: TerminalPredicate | None,
) -> VerifierState:
    if terminal_predicate is None:
        return state
    try:
        bound = bool(terminal_predicate(state.terminal_claim, state.challenges))
    except Exception as exc:
        raise SumcheckVerificationError("terminal predicate raised") from exc
    if not bound:
        raise SumcheckVerificationError("caller terminal predicate rejected")
    return replace(state, terminal_bound=True)


def bind_terminal_claim(state: VerifierState, terminal_predicate: TerminalPredicate) -> VerifierState:
    """Bind round state to the caller's external terminal relation."""

    if not isinstance(state, VerifierState):
        raise SumcheckVerificationError("invalid verifier state")
    return _terminal_predicate_result(state, terminal_predicate)


def bind_terminal_commitment(state: VerifierState, expected: ValueCommitment) -> VerifierState:
    """Convenience binding for an externally authenticated terminal commitment."""

    if not isinstance(expected, ValueCommitment):
        raise SumcheckVerificationError("expected terminal must be ValueCommitment")
    return bind_terminal_claim(state, lambda actual, _challenges: actual == expected)


def verify_committed_sumcheck(
    label: str,
    proof: CommittedSumcheckProof | dict[str, Any],
    initial_claim: int | ValueCommitment,
    expected_variable_count: int,
    degree: int,
    *,
    params: PedersenParams,
    transcript: Transcript,
    terminal_predicate: TerminalPredicate | None = None,
) -> VerifierState:
    """Verify round consistency and return terminal commitment state.

    Omitting ``terminal_predicate`` intentionally returns a state with
    ``terminal_bound=False``; it must not be mistaken for a complete proof.
    """

    try:
        if isinstance(proof, dict):
            proof = CommittedSumcheckProof.from_dict(proof)
        if not isinstance(proof, CommittedSumcheckProof):
            raise SumcheckVerificationError("invalid proof object")
        if not isinstance(expected_variable_count, int) or isinstance(expected_variable_count, bool) or expected_variable_count <= 0:
            raise SumcheckVerificationError("invalid expected round count")
        if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
            raise SumcheckVerificationError("invalid expected degree")
        if proof.protocol != COMMITTED_SUMCHECK_PROTOCOL or proof.variable_count != expected_variable_count or proof.degree != degree:
            raise SumcheckVerificationError("proof shape does not match public expected shape")
        if len(proof.rounds) != expected_variable_count:
            raise SumcheckVerificationError("proof has missing or extra rounds")
        params = _require_params(params)
        transcript = _require_transcript(transcript)
        initial_commitment = _initial_verifier_commitment(initial_claim, params)
        if proof.initial_claim_commitment != initial_commitment.point:
            raise SumcheckVerificationError("proof initial commitment differs from external statement")
        _absorb_statement(transcript, label, params, initial_commitment, expected_variable_count, degree)
        current_point = initial_commitment.point
        challenges: list[int] = []
        seen_rounds: set[int] = set()

        for expected_round, round_message in enumerate(proof.rounds):
            if not isinstance(round_message, dict) or set(round_message) != {
                "round", "coefficient_commitments", "coefficient_poks", "group_equality_pok", "challenge"
            }:
                raise SumcheckVerificationError("round has missing or extra fields")
            round_index = round_message["round"]
            if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index != expected_round or round_index in seen_rounds:
                raise SumcheckVerificationError("round index is missing, duplicated, or out of order")
            seen_rounds.add(round_index)
            commitment_wires = round_message["coefficient_commitments"]
            pok_wires = round_message["coefficient_poks"]
            if not isinstance(commitment_wires, list) or not isinstance(pok_wires, list):
                raise SumcheckVerificationError("round commitments/PoKs must be lists")
            if len(commitment_wires) != degree + 1 or len(pok_wires) != degree + 1:
                raise SumcheckVerificationError("round coefficient width differs from public degree")
            commitments = [ValueCommitment.from_dict(item) for item in commitment_wires]
            poks = [RepresentationProof.from_dict(item) for item in pok_wires]
            for coefficient_index, (commitment, pok) in enumerate(zip(commitments, poks)):
                if not verify_representation(
                    params,
                    commitment,
                    pok,
                    label=f"{label}/round/{expected_round}/coefficient/{coefficient_index}",
                    transcript=transcript,
                ):
                    raise SumcheckVerificationError("coefficient representation PoK failed")
            equality_point = _sum_commitments([item.point for item in commitments]) - current_point + commitments[0].point
            equality = RepresentationProof.from_dict(round_message["group_equality_pok"])
            if not verify_representation(
                params,
                ValueCommitment(equality_point),
                equality,
                label=f"{label}/round/{expected_round}/group-equality",
                transcript=transcript,
            ):
                raise SumcheckVerificationError("round group equality failed")
            challenge = _parse_scalar(round_message["challenge"])
            expected_challenge = transcript.challenge_scalar(f"{label}/round/{expected_round}/challenge", nonzero=True)
            if challenge != expected_challenge:
                raise SumcheckVerificationError("Fiat--Shamir challenge mismatch")
            challenges.append(challenge)
            current_point = _evaluate_commitments([item.point for item in commitments], challenge)

        terminal = ValueCommitment(_check_point(proof.terminal_claim_commitment))
        if terminal.point != current_point or len(seen_rounds) != expected_variable_count:
            raise SumcheckVerificationError("terminal commitment does not match round evaluation")
        state = VerifierState(terminal_claim=terminal, challenges=tuple(challenges), round_consistent=True)
        return _terminal_predicate_result(state, terminal_predicate)
    except SumcheckVerificationError:
        raise
    except (CommitmentError, EncodingError, KeyError, TypeError, ValueError, IndexError) as exc:
        raise SumcheckVerificationError("malformed or invalid committed Sumcheck proof") from exc


def verify_committed_sumcheck_bool(*args: Any, terminal_predicate: TerminalPredicate | None = None, **kwargs: Any) -> bool:
    """Compatibility wrapper; true only after explicit terminal binding."""

    try:
        state = verify_committed_sumcheck(*args, terminal_predicate=terminal_predicate, **kwargs)
        return state.round_consistent and state.terminal_bound
    except SumcheckVerificationError:
        return False


def _check_point(point: G1Point) -> G1Point:
    if not isinstance(point, G1Point) or not point.is_on_curve():
        raise CommitmentError("invalid terminal Sumcheck point")
    try:
        return _parse_point(encode_point(point).hex())
    except (EncodingError, CommitmentError, ValueError) as exc:
        raise CommitmentError("non-canonical terminal Sumcheck point") from exc
