"""Small, self-contained BN254 G1 implementation used by the zkGPT baseline.

The formal Python path keeps the group arithmetic in this baseline-local
module, so the comparison runner has no external cryptographic source-root or
vendor dependency.  It implements the affine short-Weierstrass G1 group used
by the Hyrax commitment layer; scalar arithmetic is performed in
:data:`BN254_FR_MODULUS`.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

from .field import BN254_FR_MODULUS, f_inv, parse_canonical_decimal, to_field_int

try:
    from ._openssl_msm import ACCELERATOR as _OPENSSL_MSM_ACCELERATOR
except (ImportError, OSError, RuntimeError):  # pragma: no cover - host capability gate
    _OPENSSL_MSM_ACCELERATOR = None

# BN254 base-field modulus (the scalar modulus is kept in field.py).
BN254_FQ_MODULUS = 21888242871839275222246405745257275088696311157297823662689037894645226208583


def scalar_random(nonzero: bool = False) -> int:
    while True:
        value = secrets.randbelow(BN254_FR_MODULUS)
        if value or not nonzero:
            return value


@dataclass(frozen=True)
class G1Point:
    x: int = 0
    y: int = 0
    infinity: bool = True

    @staticmethod
    def zero() -> "G1Point":
        return G1Point()

    @staticmethod
    def generator() -> "G1Point":
        return G1Point(1, 2, False)

    def normalized(self) -> "G1Point":
        if self.infinity:
            return self
        return G1Point(self.x % BN254_FQ_MODULUS, self.y % BN254_FQ_MODULUS, False)

    def __neg__(self) -> "G1Point":
        if self.infinity:
            return self
        return G1Point(self.x, (-self.y) % BN254_FQ_MODULUS, False)

    def __add__(self, other: "G1Point") -> "G1Point":
        if self.infinity:
            return other
        if other.infinity:
            return self
        p = BN254_FQ_MODULUS
        if self.x == other.x:
            if (self.y + other.y) % p == 0:
                return G1Point.zero()
            slope = ((3 * self.x * self.x) * pow(2 * self.y, p - 2, p)) % p
        else:
            slope = ((other.y - self.y) * pow(other.x - self.x, p - 2, p)) % p
        x3 = (slope * slope - self.x - other.x) % p
        y3 = (slope * (self.x - x3) - self.y) % p
        return G1Point(x3, y3, False)

    def __sub__(self, other: "G1Point") -> "G1Point":
        return self + (-other)

    def __rmul__(self, scalar: int) -> "G1Point":
        n = int(scalar) % BN254_FR_MODULUS
        if self.infinity or n == 0:
            return G1Point.zero()
        if _OPENSSL_MSM_ACCELERATOR is not None:
            accelerated = _OPENSSL_MSM_ACCELERATOR.scalar_mul(self, n)
            if accelerated is not None:
                return accelerated
        # Keep the wire representation affine, but perform the scalar ladder
        # in Jacobian coordinates so setup/opening does not pay one field
        # inversion per bit of every generator multiplication.
        x, y, z = self.x % BN254_FQ_MODULUS, self.y % BN254_FQ_MODULUS, 1
        rx, ry, rz = 0, 1, 0
        while n:
            if n & 1:
                rx, ry, rz = _jacobian_add(rx, ry, rz, x, y, z)
            x, y, z = _jacobian_double(x, y, z)
            n >>= 1
        return _jacobian_to_affine(rx, ry, rz)

    def __mul__(self, scalar: int) -> "G1Point":
        return self.__rmul__(scalar)

    def to_json(self) -> dict[str, Any]:
        point = self.normalized()
        return {"x": str(point.x), "y": str(point.y), "infinity": bool(point.infinity)}

    @staticmethod
    def from_json(value: Any) -> "G1Point":
        if not isinstance(value, dict):
            raise ValueError("invalid G1 point encoding")
        if set(value) != {"x", "y", "infinity"} or not isinstance(value["infinity"], bool):
            raise ValueError("invalid G1 point encoding")
        x = parse_canonical_decimal(value["x"], label="G1.x", nonnegative=True)
        y = parse_canonical_decimal(value["y"], label="G1.y", nonnegative=True)
        if x >= BN254_FQ_MODULUS or y >= BN254_FQ_MODULUS:
            raise ValueError("non-canonical G1 coordinate")
        if value["infinity"] and (x != 0 or y != 0):
            raise ValueError("non-canonical infinity coordinates")
        point = G1Point(x, y, value["infinity"])
        if not point.infinity and not point.is_on_curve():
            raise ValueError("G1 point is not on BN254 curve")
        return point

    def is_on_curve(self) -> bool:
        if self.infinity:
            return True
        p = BN254_FQ_MODULUS
        return (self.y * self.y - (self.x * self.x * self.x + 3)) % p == 0


def hash_scalar(domain: str, payload: Any, nonzero: bool = True) -> int:
    encoded = json.dumps({"domain": domain, "payload": payload}, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    value = int.from_bytes(hashlib.sha256(encoded).digest(), "big") % BN254_FR_MODULUS
    if nonzero and value == 0:
        return 1
    return value


def hash_point(domain: str, payload: Any) -> G1Point:
    """Derive an independent deterministic G1 generator.

    Multiplying the fixed curve generator by a public scalar would make every
    vector-generator relation known to the verifier (and would reduce a
    Pedersen vector commitment to a single-base commitment).  Hyrax needs
    independent bases, so use a domain-separated try-and-increment
    hash-to-curve map instead.  BN254's base field is 3 mod 4, which makes the
    square-root step a single exponentiation; the candidate is checked before
    accepting it and the digest parity fixes the sign.
    """
    encoded = json.dumps(
        {"domain": domain, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    for counter in range(256):
        digest = hashlib.sha256(encoded + counter.to_bytes(4, "big")).digest()
        x = int.from_bytes(digest, "big") % BN254_FQ_MODULUS
        rhs = (x * x * x + 3) % BN254_FQ_MODULUS
        y = pow(rhs, (BN254_FQ_MODULUS + 1) // 4, BN254_FQ_MODULUS)
        if (y * y) % BN254_FQ_MODULUS != rhs:
            continue
        if (y & 1) != (digest[-1] & 1):
            y = (-y) % BN254_FQ_MODULUS
        point = G1Point(x, y, False)
        if point.is_on_curve():
            return point
    raise RuntimeError("BN254 hash-to-G1 failed after 256 candidates")


def generator_vector(label: str, count: int, suffix: str = "G") -> List[G1Point]:
    return [hash_point("zkGPT/BN254/G1", {"label": label, "suffix": suffix, "index": index}) for index in range(count)]


def multi_scalar_mul(points: Sequence[G1Point], scalars: Iterable[int]) -> G1Point:
    """Compute a batch MSM with Jacobian Pippenger buckets.

    The baseline's Hyrax path performs many MSMs over the same BN254 G1
    vectors.  Repeated affine ``point + scalar * point`` operations are
    correct but spend one field inversion per addition; the bucket route
    keeps additions in Jacobian coordinates and converts only the result.
    It is the same native commitment arithmetic and leaves the serialized
    Hyrax protocol unchanged.
    """
    entries = [
        (point, int(scalar) % BN254_FR_MODULUS)
        for point, scalar in zip(points, scalars)
        if not point.infinity and int(scalar) % BN254_FR_MODULUS
    ]
    if not entries:
        return G1Point.zero()

    # The optional OpenSSL implementation uses the same short-Weierstrass
    # BN254 curve and returns the same affine G1 value.  Keep very short MSMs
    # on the local Jacobian path because ctypes setup dominates them; larger
    # Hyrax/IPA MSMs are the hot path for which the native implementation is
    # materially faster.  The environment switch is useful for the arithmetic
    # equivalence test and for hosts without a compatible libcrypto.
    if _OPENSSL_MSM_ACCELERATOR is not None and len(entries) >= 64:
        accelerated = _OPENSSL_MSM_ACCELERATOR.msm(entries)
        if accelerated is not None:
            return accelerated

    # Select the fallback Pippenger window from the number of nonzero terms.
    # Four-bit buckets are best for the short local vectors used by the
    # per-family proofs, while larger vectors benefit from a wider window.
    # This is only an arithmetic implementation choice: the same native G1
    # linear combination is returned and the serialized protocol is unchanged.
    entry_count = len(entries)
    if entry_count < 128:
        window = 4
    elif entry_count < 256:
        window = 5
    elif entry_count < 512:
        window = 6
    elif entry_count < 2048:
        window = 7
    else:
        window = 8
    mask = (1 << window) - 1
    max_bits = max(scalar.bit_length() for _, scalar in entries)
    max_shift = ((max_bits + window - 1) // window - 1) * window
    result = (0, 1, 0)
    for shift in range(max_shift, -1, -window):
        for _ in range(window):
            result = _jacobian_double(*result)
        buckets = [(0, 1, 0) for _ in range(1 << window)]
        for point, scalar in entries:
            digit = (scalar >> shift) & mask
            if digit:
                buckets[digit] = _jacobian_add(
                    *buckets[digit], point.x % BN254_FQ_MODULUS,
                    point.y % BN254_FQ_MODULUS, 1,
                )
        running = (0, 1, 0)
        for digit in range(mask, 0, -1):
            running = _jacobian_add(*running, *buckets[digit])
            result = _jacobian_add(*result, *running)
    return _jacobian_to_affine(*result)


def derive_nonzero_challenge(domain: str, payload: Any) -> int:
    return hash_scalar(domain, payload, True)


def _jacobian_double(x: int, y: int, z: int) -> tuple[int, int, int]:
    p = BN254_FQ_MODULUS
    if z == 0 or y == 0:
        return 0, 1, 0
    yy = (y * y) % p
    s = (4 * x * yy) % p
    m = (3 * x * x) % p
    nx = (m * m - 2 * s) % p
    ny = (m * (s - nx) - 8 * yy * yy) % p
    nz = (2 * y * z) % p
    return nx, ny, nz


def _jacobian_add(x1: int, y1: int, z1: int, x2: int, y2: int, z2: int) -> tuple[int, int, int]:
    p = BN254_FQ_MODULUS
    if z1 == 0:
        return x2 % p, y2 % p, z2 % p
    if z2 == 0:
        return x1 % p, y1 % p, z1 % p
    z1z1 = (z1 * z1) % p
    z2z2 = (z2 * z2) % p
    u1 = (x1 * z2z2) % p
    u2 = (x2 * z1z1) % p
    s1 = (y1 * z2 * z2z2) % p
    s2 = (y2 * z1 * z1z1) % p
    h = (u2 - u1) % p
    r = (s2 - s1) % p
    if h == 0:
        if r == 0:
            return _jacobian_double(x1, y1, z1)
        return 0, 1, 0
    hh = (h * h) % p
    hhh = (h * hh) % p
    nx = (r * r - hhh - 2 * u1 * hh) % p
    ny = (r * (u1 * hh - nx) - s1 * hhh) % p
    nz = (h * z1 * z2) % p
    return nx, ny, nz


def _jacobian_to_affine(x: int, y: int, z: int) -> G1Point:
    if z == 0:
        return G1Point.zero()
    p = BN254_FQ_MODULUS
    z_inv = pow(z, p - 2, p)
    z2 = (z_inv * z_inv) % p
    return G1Point((x * z2) % p, (y * z2 * z_inv) % p, False)
