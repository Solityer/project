from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .field import (
    BN254_FR_MODULUS,
    f_eval_quadratic,
    f_mul,
    f_sub,
    log2_power_of_two,
    next_power_of_two,
    to_field_int,
)
from .util import digest_json


@dataclass(frozen=True)
class SumcheckProof:
    protocol: str
    variable_count: int
    claimed_sum: int
    rounds: List[Dict[str, Any]]
    final_a: int
    final_b: int
    final_product: int

    def to_json(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "field_modulus": str(BN254_FR_MODULUS),
            "variable_count": self.variable_count,
            "claimed_sum": str(self.claimed_sum),
            "rounds": self.rounds,
            "final_a": str(self.final_a),
            "final_b": str(self.final_b),
            "final_product": str(self.final_product),
        }


def prove_product_sum(label: str, a_values: Iterable[int], b_values: Iterable[int], transcript_seed: Dict[str, Any]) -> SumcheckProof:
    a = [to_field_int(item) for item in a_values]
    b = [to_field_int(item) for item in b_values]
    if len(a) != len(b):
        raise ValueError("sumcheck vector length mismatch")
    padded = next_power_of_two(len(a))
    a.extend([0] * (padded - len(a)))
    b.extend([0] * (padded - len(b)))
    variable_count = log2_power_of_two(padded)
    claim = sum(f_mul(x, y) for x, y in zip(a, b)) % BN254_FR_MODULUS
    rounds: List[Dict[str, Any]] = []
    current_claim = claim
    transcript: List[Dict[str, Any]] = [{"label": label, "seed": transcript_seed, "claimed_sum": str(claim)}]

    for round_index in range(variable_count):
        c0 = 0
        c1 = 0
        c2 = 0
        next_a: List[int] = []
        next_b: List[int] = []
        for idx in range(0, len(a), 2):
            a0 = a[idx]
            a1 = a[idx + 1]
            b0 = b[idx]
            b1 = b[idx + 1]
            da = f_sub(a1, a0)
            db = f_sub(b1, b0)
            c0 = (c0 + a0 * b0) % BN254_FR_MODULUS
            c1 = (c1 + a0 * db + b0 * da) % BN254_FR_MODULUS
            c2 = (c2 + da * db) % BN254_FR_MODULUS
        coeffs = [c0, c1, c2]
        if (f_eval_quadratic(coeffs, 0) + f_eval_quadratic(coeffs, 1)) % BN254_FR_MODULUS != current_claim:
            raise RuntimeError("internal sumcheck claim transition failed")
        challenge = _challenge(label, round_index, transcript, coeffs)
        for idx in range(0, len(a), 2):
            a0 = a[idx]
            a1 = a[idx + 1]
            b0 = b[idx]
            b1 = b[idx + 1]
            next_a.append((a0 + f_sub(a1, a0) * challenge) % BN254_FR_MODULUS)
            next_b.append((b0 + f_sub(b1, b0) * challenge) % BN254_FR_MODULUS)
        round_msg = {
            "round": round_index,
            "coefficients": [str(item) for item in coeffs],
            "challenge": str(challenge),
        }
        rounds.append(round_msg)
        transcript.append(round_msg)
        current_claim = f_eval_quadratic(coeffs, challenge)
        a = next_a
        b = next_b

    final_product = f_mul(a[0], b[0])
    if final_product != current_claim:
        raise RuntimeError("internal sumcheck final product failed")
    return SumcheckProof(
        protocol="matrix-product-sumcheck-v1",
        variable_count=variable_count,
        claimed_sum=claim,
        rounds=rounds,
        final_a=a[0],
        final_b=b[0],
        final_product=final_product,
    )


def verify_product_sum(label: str, proof: Dict[str, Any], transcript_seed: Dict[str, Any]) -> Dict[str, Any]:
    claimed = to_field_int(int(proof["claimed_sum"]))
    variable_count = int(proof["variable_count"])
    rounds = list(proof["rounds"])
    transcript: List[Dict[str, Any]] = [{"label": label, "seed": transcript_seed, "claimed_sum": str(claimed)}]
    current_claim = claimed
    checks: List[Dict[str, Any]] = []
    for expected_round in range(variable_count):
        if expected_round >= len(rounds):
            return _failed("missing_round", checks)
        round_msg = rounds[expected_round]
        coeffs = [to_field_int(int(item)) for item in round_msg["coefficients"]]
        sum01 = (f_eval_quadratic(coeffs, 0) + f_eval_quadratic(coeffs, 1)) % BN254_FR_MODULUS
        transition_ok = sum01 == current_claim
        challenge = _challenge(label, expected_round, transcript, coeffs)
        challenge_ok = str(challenge) == str(round_msg["challenge"])
        checks.append(
            {
                "round": expected_round,
                "transition_ok": transition_ok,
                "challenge_ok": challenge_ok,
            }
        )
        if not transition_ok or not challenge_ok:
            return _failed("round_check_failed", checks)
        transcript.append(round_msg)
        current_claim = f_eval_quadratic(coeffs, challenge)
    final_a = to_field_int(int(proof["final_a"]))
    final_b = to_field_int(int(proof["final_b"]))
    final_product = to_field_int(int(proof["final_product"]))
    final_ok = final_product == f_mul(final_a, final_b) == current_claim
    checks.append({"final_ok": final_ok})
    return {
        "status": "VERIFY_OK" if final_ok else "VERIFY_FAILED",
        "reason": "ok" if final_ok else "final_product_mismatch",
        "checks": checks,
    }


def _challenge(label: str, round_index: int, transcript: List[Dict[str, Any]], coeffs: List[int]) -> int:
    digest = digest_json(
        {
            "protocol": "matrix-product-sumcheck-v1",
            "label": label,
            "round": round_index,
            "transcript": transcript,
            "coefficients": [str(item) for item in coeffs],
        }
    )
    return int(digest, 16) % BN254_FR_MODULUS


def _failed(reason: str, checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"status": "VERIFY_FAILED", "reason": reason, "checks": checks}
