"""Native BN254 Hyrax commitment and zero-knowledge MLE evaluation proof.

The committed polynomial is one complete power-of-two vector.  An opening
proves that a fresh Pedersen commitment ``D = f(r) U + tau H`` contains the
evaluation of an externally supplied vector commitment.  The evaluation and
both commitment blinders stay private.  The logarithmic IPA emits only L/R
group elements and its final masked scalar; the verifier computes the folded
public equality vector itself.

There is intentionally no verifier API that trusts a proof-provided base
commitment or point.  The trusted commitment, common N_in domain, random point,
and public-statement context are mandatory verifier inputs.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .bn254 import BN254_FQ_MODULUS, G1Point, generator_vector, hash_point, multi_scalar_mul, scalar_random
from .field import BN254_FR_MODULUS, f_inv, log2_power_of_two, to_field_int
from .gpu_execution import active_mle_fold
from .transcript import Transcript
from .util import read_json, write_json
from .value_commitment import PedersenParams, ValueCommitment


PROTOCOL_ID = "zkGPT-hyrax-mle-evaluation-commitment-v2"
LINEAR_PROTOCOL_ID = "zkGPT-hyrax-linear-evaluation-commitment-v1"
BATCHED_LINEAR_PROTOCOL_ID = "zkGPT-hyrax-batched-linear-claim-v1"
PARAMETER_ID = "zkGPT-bn254-hyrax-generators-v2"


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
        if not isinstance(value["x"], str) or not value["x"].isdigit():
            raise ValueError("non-canonical BN254 G1 x coordinate")
        if not isinstance(value["y"], str) or not value["y"].isdigit():
            raise ValueError("non-canonical BN254 G1 y coordinate")
        if not isinstance(value["infinity"], bool):
            raise ValueError("non-canonical BN254 G1 infinity flag")
        x = int(value["x"])
        y = int(value["y"])
        infinity = value["infinity"]
        if infinity:
            if x != 0 or y != 0:
                raise ValueError("non-canonical BN254 infinity coordinates")
        elif x >= BN254_FQ_MODULUS or y >= BN254_FQ_MODULUS:
            raise ValueError("BN254 coordinate is outside the base field")
        point = G1Point(x, y, infinity)
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
    return {
        "parameter_id": pp_label,
        "vector_generators": vector_generators,
        "blind_generator": blind_generator,
        "ipa_generators": ipa_generators,
        "equality_generators": equality_generators,
        "evaluation_generator": evaluation_generator,
        "ipa_length": ipa_length,
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
    if not isinstance(final_a_raw, str) or not final_a_raw.isdigit():
        return False
    final_a = int(final_a_raw)
    if final_a >= BN254_FR_MODULUS:
        return False
    transcript.absorb("hyrax/ipa/final", {"a": final_a})
    return current == final_a * gs[0] + bv[0] * hs[0] + (final_a * bv[0] % BN254_FR_MODULUS) * inner_product_generator


@dataclass(frozen=True)
class NativeHyrax:
    zk_root: pathlib.Path
    opening_log: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False, compare=False)

    def require_binary(self) -> None:
        return None

    def commit_vector(self, label: str, values: Iterable[int], work_dir: pathlib.Path) -> Dict[str, Any]:
        vector = [to_field_int(value) for value in values]
        n = _require_power_of_two(len(vector), "committed vector length")
        params = _params(n)
        blind = scalar_random(nonzero=True)
        commitment = multi_scalar_mul(params["vector_generators"], vector) + blind * params["blind_generator"]
        return {
            "backend": PROTOCOL_ID,
            "parameter_id": params["parameter_id"],
            "label": label,
            "vector_length": n,
            "commitment": commitment.to_json(),
            "blind": str(blind),
        }

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
        work_dir.mkdir(parents=True, exist_ok=True)
        vector = [to_field_int(value) for value in values]
        n = _require_power_of_two(len(vector), "committed vector length")
        point_values = [to_field_int(value) for value in point]
        if len(point_values) != log2_power_of_two(n):
            raise ValueError("Hyrax opening point has wrong dimension")
        if registration.get("backend") != PROTOCOL_ID or int(registration.get("vector_length", -1)) != n:
            raise ValueError("registered Hyrax domain mismatch")
        params = _params(n)
        if registration.get("parameter_id") != params["parameter_id"]:
            raise ValueError("registered Hyrax parameter mismatch")
        blind = to_field_int(int(registration["blind"]))
        expected_commitment = _point(registration["commitment"])
        recomputed = multi_scalar_mul(params["vector_generators"], vector) + blind * params["blind_generator"]
        if recomputed != expected_commitment:
            raise ValueError("registered commitment does not bind supplied vector")
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
        a = vector + [(blind - tau) % BN254_FR_MODULUS] + [0] * (n - 1)
        b = weights + [0] * n
        inner_product_generator = -params["evaluation_generator"]
        statement_point = expected_commitment - evaluation_commitment + multi_scalar_mul(params["equality_generators"], b)
        context = dict(public_context or {})
        _statement_transcript(transcript, label, n, expected_commitment, point_values, evaluation_commitment, context)
        ipa = _prove_ipa(
            transcript,
            a,
            b,
            params["ipa_generators"],
            params["equality_generators"],
            inner_product_generator,
            statement_point,
        )
        proof = {
            "protocol": PROTOCOL_ID,
            "parameter_id": params["parameter_id"],
            "vector_length": n,
            "evaluation_commitment": evaluation_commitment.to_json(),
            "ipa": ipa,
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
        work_dir.mkdir(parents=True, exist_ok=True)
        vector = [to_field_int(value) for value in values]
        linear_weights = [to_field_int(value) for value in weights]
        n = _require_power_of_two(len(vector), "committed vector length")
        if len(linear_weights) != n:
            raise ValueError("Hyrax linear-form weights have wrong domain")
        if registration.get("backend") != PROTOCOL_ID or int(registration.get("vector_length", -1)) != n:
            raise ValueError("registered Hyrax domain mismatch")
        params = _params(n)
        if registration.get("parameter_id") != params["parameter_id"]:
            raise ValueError("registered Hyrax parameter mismatch")
        blind = to_field_int(int(registration["blind"]))
        expected_commitment = _point(registration["commitment"])
        recomputed = multi_scalar_mul(params["vector_generators"], vector) + blind * params["blind_generator"]
        if recomputed != expected_commitment:
            raise ValueError("registered commitment does not bind supplied vector")
        evaluation = _dot(vector, linear_weights)
        tau = scalar_random(nonzero=False)
        evaluation_commitment = evaluation * params["evaluation_generator"] + tau * params["blind_generator"]
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
        ipa = _prove_ipa(
            transcript,
            a,
            b,
            params["ipa_generators"],
            params["equality_generators"],
            -params["evaluation_generator"],
            statement_point,
        )
        proof = {
            "protocol": LINEAR_PROTOCOL_ID,
            "parameter_id": params["parameter_id"],
            "vector_length": n,
            "evaluation_commitment": evaluation_commitment.to_json(),
            "ipa": ipa,
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
        work_dir.mkdir(parents=True, exist_ok=True)
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
        if registration.get("backend") != PROTOCOL_ID or int(registration.get("vector_length", -1)) != n:
            raise ValueError("registered Hyrax domain mismatch")
        params = _params(n)
        if registration.get("parameter_id") != params["parameter_id"]:
            raise ValueError("registered Hyrax parameter mismatch")
        blind = to_field_int(int(registration["blind"]))
        commitment = _point(registration["commitment"])
        if multi_scalar_mul(params["vector_generators"], vector) + blind * params["blind_generator"] != commitment:
            raise ValueError("registered commitment does not bind supplied vector")
        tau = scalar_random(nonzero=False)
        evaluation_commitment = evaluation * params["evaluation_generator"] + tau * params["blind_generator"]
        context = dict(public_context or {})
        _batched_linear_statement_transcript(
            transcript, label, n, commitment, linear_form, evaluation_commitment, context,
        )
        proof = {
            "protocol": BATCHED_LINEAR_PROTOCOL_ID,
            "parameter_id": params["parameter_id"],
            "vector_length": n,
            "evaluation_commitment": evaluation_commitment.to_json(),
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
        work_dir.mkdir(parents=True, exist_ok=True)
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
            "protocol", "parameter_id", "vector_length", "evaluation_commitment", "ipa", "transcript_digest"
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
        evaluation_commitment = _point(proof["evaluation_commitment"])
        weights = _eq_table(point_values)
        b = weights + [0] * n
        statement_point = commitment - evaluation_commitment + multi_scalar_mul(params["equality_generators"], b)
        _statement_transcript(
            transcript,
            label,
            n,
            commitment,
            point_values,
            evaluation_commitment,
            dict(public_context or {}),
        )
        ok = _verify_ipa(
            transcript,
            proof["ipa"],
            b,
            params["ipa_generators"],
            params["equality_generators"],
            -params["evaluation_generator"],
            statement_point,
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
            "protocol", "parameter_id", "vector_length", "evaluation_commitment", "ipa", "transcript_digest"
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
        evaluation_commitment = _point(proof["evaluation_commitment"])
        b = linear_weights + [0] * n
        statement_point = commitment - evaluation_commitment + multi_scalar_mul(params["equality_generators"], b)
        _linear_statement_transcript(
            transcript,
            label,
            n,
            commitment,
            linear_form,
            evaluation_commitment,
            dict(public_context or {}),
        )
        ok = _verify_ipa(
            transcript,
            proof["ipa"],
            b,
            params["ipa_generators"],
            params["equality_generators"],
            -params["evaluation_generator"],
            statement_point,
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
            "protocol", "parameter_id", "vector_length", "evaluation_commitment", "transcript_digest"
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
        evaluation_commitment = _point(proof["evaluation_commitment"])
        _batched_linear_statement_transcript(
            transcript, label, n, commitment, linear_form,
            evaluation_commitment, dict(public_context or {}),
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


def mle_point_from_u64(challenge: int, variable_count: int) -> List[int]:
    """Compatibility helper; formal code must source challenge from Fiat-Shamir."""
    state = int(challenge) & ((1 << 64) - 1)
    out: List[int] = []
    for _ in range(variable_count):
        state ^= state << 13 & ((1 << 64) - 1)
        state ^= state >> 7
        state ^= state << 17 & ((1 << 64) - 1)
        out.append(state % BN254_FR_MODULUS)
    return out


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
