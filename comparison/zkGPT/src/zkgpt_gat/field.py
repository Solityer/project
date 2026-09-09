from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

BN254_FR_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617
DEFAULT_FIXED_SCALE = 1_000_000


def parse_canonical_decimal(
    value: object,
    *,
    label: str = "decimal",
    nonnegative: bool = False,
) -> int:
    """Parse the only decimal wire spelling accepted by zkGPT.

    JSON integer aliases such as 01, +1, -0, and surrounding whitespace are
    rejected before conversion. Native proof serializers always emit strings,
    so accepting Python integers here would create a second wire grammar.
    """

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be canonical decimal text")
    if value == "0":
        return 0
    negative = value.startswith("-")
    digits = value[1:] if negative else value
    if negative and nonnegative:
        raise ValueError(f"{label} must be nonnegative")
    if not digits or not digits.isascii() or not digits.isdigit() or digits[0] == "0":
        raise ValueError(f"{label} is not canonical decimal text")
    return -int(digits) if negative else int(digits)


def to_field_int(value: int) -> int:
    return int(value) % BN254_FR_MODULUS


def f_inv(value: int) -> int:
    value = int(value) % BN254_FR_MODULUS
    if value == 0:
        raise ZeroDivisionError("field inverse of zero")
    return pow(value, BN254_FR_MODULUS - 2, BN254_FR_MODULUS)


def next_power_of_two(value: int) -> int:
    out = 1
    while out < max(int(value), 1):
        out <<= 1
    return out


def sumcheck_domain(count: int) -> int:
    """Return the power-of-two domain used by a committed Sumcheck.

    The native MatMul and GKR product relations both need at least one
    variable so that a one-element relation still has a real Sumcheck round.
    Keep that padding rule in one field-level helper; callers retain their
    real output/index cardinality and only pad the relation oracle with zero.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Sumcheck domain count must be a positive integer")
    return next_power_of_two(max(2, count))


def log2_power_of_two(value: int) -> int:
    value = int(value)
    if value <= 0 or value & (value - 1):
        raise ValueError(f"not a power of two: {value}")
    return value.bit_length() - 1


def signed_to_field(value: int) -> int:
    return to_field_int(value)


def quantize_float(value: float, scale: int = DEFAULT_FIXED_SCALE) -> int:
    return signed_to_field(int(round(float(value) * scale)))


def quantize_array(values: np.ndarray, scale: int = DEFAULT_FIXED_SCALE) -> List[int]:
    flat = np.ravel(np.asarray(values, dtype=np.float64))
    return [quantize_float(float(item), scale) for item in flat]


@dataclass(frozen=True)
class FieldEncoding:
    modulus: int = BN254_FR_MODULUS
    fixed_scale: int = DEFAULT_FIXED_SCALE
    field_bytes: int = 32
    signed_representation: str = "centered_mod_p"
