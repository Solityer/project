from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

BN254_FR_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617
DEFAULT_FIXED_SCALE = 1_000_000


def to_field_int(value: int) -> int:
    return int(value) % BN254_FR_MODULUS


def f_add(left: int, right: int) -> int:
    return (int(left) + int(right)) % BN254_FR_MODULUS


def f_sub(left: int, right: int) -> int:
    return (int(left) - int(right)) % BN254_FR_MODULUS


def f_mul(left: int, right: int) -> int:
    return (int(left) * int(right)) % BN254_FR_MODULUS


def f_inv(value: int) -> int:
    value = int(value) % BN254_FR_MODULUS
    if value == 0:
        raise ZeroDivisionError("field inverse of zero")
    return pow(value, BN254_FR_MODULUS - 2, BN254_FR_MODULUS)


def f_eval_quadratic(coeffs: Iterable[int], point: int) -> int:
    c0, c1, c2 = [to_field_int(item) for item in coeffs]
    x = to_field_int(point)
    return (c0 + c1 * x + c2 * x * x) % BN254_FR_MODULUS


def next_power_of_two(value: int) -> int:
    out = 1
    while out < max(int(value), 1):
        out <<= 1
    return out


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


def write_field_vector(path, values: Iterable[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        first = True
        for value in values:
            if not first:
                handle.write(" ")
            handle.write(str(to_field_int(int(value))))
            first = False
        handle.write("\n")


@dataclass(frozen=True)
class FieldEncoding:
    modulus: int = BN254_FR_MODULUS
    fixed_scale: int = DEFAULT_FIXED_SCALE
    field_bytes: int = 32
    signed_representation: str = "centered_mod_p"
