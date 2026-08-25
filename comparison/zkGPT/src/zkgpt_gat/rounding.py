"""Canonical rounding-mode and SignedRoundDiv helpers for the zkGPT baseline.

The mode is part of the authenticated quantization/configuration identity.  A
caller must therefore pass an explicit supported value; this module never
silently turns an absent or unknown mode into the nearest-rounding relation.
"""
from __future__ import annotations

from typing import Any, Mapping


ROUNDING_MODE_NEAREST = "nearest_ties_away"
ROUNDING_MODE_TOWARD_ZERO = "toward_zero"
ROUNDING_MODES = (ROUNDING_MODE_NEAREST, ROUNDING_MODE_TOWARD_ZERO)


def validate_rounding_mode(value: Any) -> str:
    """Return a canonical mode or fail closed."""

    if not isinstance(value, str) or value not in ROUNDING_MODES:
        raise ValueError(
            f"unsupported or missing SignedRoundDiv rounding_mode: {value!r}"
        )
    return value


def configured_rounding_mode(entries: Mapping[str, Any]) -> str:
    """Resolve the mode from an authenticated config mapping.

    Formal configs must contain the field explicitly.  Keeping this check in a
    shared helper prevents producer and external verifier from choosing
    different implicit defaults.
    """

    return validate_rounding_mode(entries.get("rounding_mode"))


def signed_round_div(numerator: int, denominator: int, mode: str) -> int:
    """Compute the protocol's signed integer division for one value."""

    mode = validate_rounding_mode(mode)
    denominator = int(denominator)
    if denominator <= 0:
        raise ValueError("SignedRoundDiv denominator must be positive")
    numerator = int(numerator)
    sign = -1 if numerator < 0 else 1
    magnitude = abs(numerator)
    quotient, remainder = divmod(magnitude, denominator)
    if mode == ROUNDING_MODE_NEAREST and 2 * remainder >= denominator:
        quotient += 1
    return sign * quotient


def rounding_constant(denominator_bound: int, mode: str) -> int:
    """Return the certified RHS constant for ``r + delta = c``."""

    mode = validate_rounding_mode(mode)
    denominator_bound = int(denominator_bound)
    if denominator_bound <= 0:
        raise ValueError("SignedRoundDiv denominator bound must be positive")
    return (
        2 * denominator_bound - 1
        if mode == ROUNDING_MODE_NEAREST
        else denominator_bound - 1
    )


def division_bound_record(
    family: str,
    mode: str,
    *,
    a_bound: int,
    d_bound: int,
    d_min: int,
    count: int,
    source: str,
    numerator_source: str,
    denominator_source: str,
) -> dict[str, Any]:
    """Build one explicit no-wrap ledger entry.

    ``B_a`` is the absolute bound of the actual equality dividend.  The
    separate ``B_x_independent`` field records the independent operand bound
    used to derive it, so the equality-side bounds cannot be confused with a
    sum of unrelated operands.
    """

    mode = validate_rounding_mode(mode)
    a_bound, d_bound, d_min, count = map(int, (a_bound, d_bound, d_min, count))
    if a_bound < 0 or d_bound <= 0 or d_min <= 0 or d_min > d_bound or count <= 0:
        raise ValueError("invalid SignedRoundDiv ledger bounds")
    q_bound = (a_bound + d_min - 1) // d_min
    constant = rounding_constant(d_bound, mode)
    if mode == ROUNDING_MODE_NEAREST:
        lhs_bound = 2 * a_bound + d_bound
    else:
        lhs_bound = a_bound
    rhs_bound = (
        2 * d_bound * q_bound + constant
        if mode == ROUNDING_MODE_NEAREST
        else d_bound * q_bound + constant
    )
    return {
        "id": str(family),
        "family": str(family),
        "source": str(source),
        "numerator_source": str(numerator_source),
        "denominator_source": str(denominator_source),
        "count": count,
        "rounding_mode": mode,
        "B_x_independent": a_bound,
        "B_a": a_bound,
        "B_q": q_bound,
        "B_r": constant,
        "B_delta": constant,
        "B_d": d_bound,
        "B_d_min": d_min,
        "B_rounding_constant": constant,
        # Retain the historical key for schema compatibility, but its value
        # is explicitly mode-dependent (d-1 for toward_zero).
        "B_2d_minus_1": constant,
        "B_2a_plus_d": lhs_bound,
        "B_2dq_plus_r": rhs_bound,
        # This is the bound of the *equality* ``r + delta = c``.  It is not
        # the independent-operand sum ``B_r + B_delta``; retaining the latter
        # separately prevents the ledger from weakening the certified gate.
        "B_r_plus_delta": constant,
        "B_r_plus_delta_independent": 2 * constant,
        "equality_lhs_bound": lhs_bound,
        "equality_rhs_bound": rhs_bound,
        "bound_semantics": {
            "B_a": "absolute_actual_equality_dividend",
            "B_x_independent": "independent_operand_sum_bound_before_division",
            "B_d": "absolute_actual_equality_denominator",
            "B_q": "absolute_quotient_bound_from_B_a_over_B_d_min",
            "B_r": "exact_mode_remainder_bound",
            "B_delta": "exact_mode_slack_bound",
            "B_r_plus_delta": "actual_equality_bound",
            "B_r_plus_delta_independent": "independent_operand_sum_bound",
        },
    }
