"""Canonical Fiat--Shamir transcript primitives for the zkGPT baseline.

The native zkGPT proof code has several independent public-coin sub-protocols.
This module gives those protocols one small, strict wire-level transcript:
messages are length delimited, values have a canonical encoding, and a
challenge is derived only after every preceding message has been absorbed.
Challenges are absorbed back into the transcript, so later challenges cannot
be replayed with a different message order.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .bn254 import BN254_FQ_MODULUS, G1Point
from .field import BN254_FR_MODULUS


TRANSCRIPT_VERSION = b"zkGPT/transcript/sha256-fr-v1"
SCALAR_BYTES = 32
POINT_BYTES = 65
SHA256_BLOCK_BYTES = 64


class EncodingError(ValueError):
    """Raised when a value is not in the canonical transcript encoding."""


def _length_prefix(value: bytes) -> bytes:
    if len(value) >= 1 << 64:
        raise EncodingError("transcript item is too large")
    return len(value).to_bytes(8, "big") + value


def encode_scalar(value: int) -> bytes:
    """Encode an Fr scalar as exactly 32 big-endian bytes.

    Unlike the internal field helpers this function never silently reduces a
    value modulo ``p``.  A proof parser therefore cannot turn a non-canonical
    scalar into a different canonical scalar by using an oversized integer.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise EncodingError("scalar must be an integer")
    if value < 0 or value >= BN254_FR_MODULUS:
        raise EncodingError("scalar is outside the canonical Fr range")
    return value.to_bytes(SCALAR_BYTES, "big")


def decode_scalar(value: bytes) -> int:
    if not isinstance(value, (bytes, bytearray)) or len(value) != SCALAR_BYTES:
        raise EncodingError("scalar encoding must be exactly 32 bytes")
    encoded = bytes(value)
    result = int.from_bytes(encoded, "big")
    if result >= BN254_FR_MODULUS:
        raise EncodingError("non-canonical Fr scalar")
    # Re-encoding also rejects alternate representations should this function
    # ever be changed to accept a bytes-like subclass.
    if encode_scalar(result) != encoded:
        raise EncodingError("non-canonical scalar encoding")
    return result


def encode_point(point: G1Point) -> bytes:
    """Encode a BN254 G1 point with a strict infinity/affine wire format."""

    if not isinstance(point, G1Point):
        raise EncodingError("point must be a G1Point")
    if point.infinity:
        if point.x != 0 or point.y != 0:
            raise EncodingError("infinity must use the canonical zero coordinates")
        return b"\x00"
    if not point.is_on_curve():
        raise EncodingError("point is not on BN254 G1")
    if not (0 <= point.x < BN254_FQ_MODULUS and 0 <= point.y < BN254_FQ_MODULUS):
        raise EncodingError("point coordinate is not canonical")
    return b"\x01" + point.x.to_bytes(32, "big") + point.y.to_bytes(32, "big")


def decode_point(value: bytes) -> G1Point:
    if not isinstance(value, (bytes, bytearray)):
        raise EncodingError("point encoding must be bytes")
    encoded = bytes(value)
    if encoded == b"\x00":
        return G1Point.zero()
    if len(encoded) != POINT_BYTES or encoded[0] != 1:
        raise EncodingError("invalid canonical G1 point length/tag")
    x = int.from_bytes(encoded[1:33], "big")
    y = int.from_bytes(encoded[33:65], "big")
    if x >= BN254_FQ_MODULUS or y >= BN254_FQ_MODULUS:
        raise EncodingError("non-canonical G1 coordinate")
    point = G1Point(x, y, False)
    if not point.is_on_curve():
        raise EncodingError("point is not on BN254 G1")
    if encode_point(point) != encoded:
        raise EncodingError("non-canonical G1 point encoding")
    return point


def _encode_int(value: int) -> bytes:
    if isinstance(value, bool):
        return b"b" + (b"\x01" if value else b"\x00")
    if value == 0:
        return b"i\x00\x00"
    negative = value < 0
    magnitude = abs(value).to_bytes((abs(value).bit_length() + 7) // 8, "big")
    return b"i" + (b"\x01" if negative else b"\x00") + _length_prefix(magnitude)


def canonical_encode(value: Any) -> bytes:
    """Encode transcript values without JSON's multiple spellings.

    The supported types intentionally match proof messages: ``None``, bool,
    integers, bytes, UTF-8 strings, lists/tuples, dictionaries, and BN254 G1
    points.  Floats and arbitrary objects are rejected rather than assigned a
    potentially ambiguous representation.
    """

    if value is None:
        return b"n"
    if isinstance(value, bool):
        return _encode_int(value)
    if isinstance(value, int):
        return _encode_int(value)
    if isinstance(value, bytes):
        return b"y" + _length_prefix(value)
    if isinstance(value, bytearray):
        return b"y" + _length_prefix(bytes(value))
    if isinstance(value, str):
        return b"s" + _length_prefix(value.encode("utf-8"))
    if isinstance(value, G1Point):
        return b"p" + _length_prefix(encode_point(value))
    if isinstance(value, (list, tuple)):
        tag = b"l" if isinstance(value, list) else b"t"
        encoded_items = [canonical_encode(item) for item in value]
        return tag + len(encoded_items).to_bytes(8, "big") + b"".join(
            _length_prefix(item) for item in encoded_items
        )
    if isinstance(value, dict):
        pairs = [(canonical_encode(key), canonical_encode(item)) for key, item in value.items()]
        pairs.sort(key=lambda pair: pair[0])
        return b"d" + len(pairs).to_bytes(8, "big") + b"".join(
            _length_prefix(key) + _length_prefix(item) for key, item in pairs
        )
    raise EncodingError(f"unsupported transcript value type: {type(value).__name__}")


def _expand_message_xmd(message: bytes, domain: bytes, length: int) -> bytes:
    """RFC 9380-style SHA-256 XMD expansion.

    The transcript uses this standard expand construction as the byte stream
    for rejection sampling into Fr.  It deliberately does not reduce a
    digest directly modulo the field modulus, which would introduce modulo
    bias.
    """

    return _expand_message_xmd_parts((message,), domain, length)


def _expand_message_xmd_parts(
    message_parts: tuple[bytes | bytearray | memoryview, ...],
    domain: bytes,
    length: int,
) -> bytes:
    """Expand a sequence of message buffers without concatenating them.

    The RFC 9380 XMD input is the concatenation of ``message_parts``.  Keeping
    the parts separate is byte-for-byte equivalent, but avoids copying the
    complete (and growing) transcript into a temporary object for every
    challenge.  This is deliberately an implementation-only optimization;
    transcript labels, state bytes, and challenge outputs are unchanged.
    """

    if not isinstance(domain, bytes) or not isinstance(message_parts, tuple):
        raise EncodingError("XMD inputs must be bytes")
    if any(not isinstance(part, (bytes, bytearray, memoryview)) for part in message_parts):
        raise EncodingError("XMD message parts must be bytes-like")
    if length <= 0 or length >= 1 << 16:
        raise EncodingError("invalid XMD output length")
    if len(domain) > 255:
        domain = hashlib.sha256(b"H2C-OVERSIZE-DST-" + domain).digest()
    domain_prime = domain + bytes((len(domain),))
    ell = (length + hashlib.sha256().digest_size - 1) // hashlib.sha256().digest_size
    if ell > 255:
        raise EncodingError("XMD output is too long")
    length_bytes = length.to_bytes(2, "big")
    z_pad = b"\x00" * SHA256_BLOCK_BYTES
    b0_hash = hashlib.sha256()
    b0_hash.update(z_pad)
    for part in message_parts:
        b0_hash.update(part)
    b0_hash.update(length_bytes)
    b0_hash.update(b"\x00")
    b0_hash.update(domain_prime)
    b0 = b0_hash.digest()
    blocks = [hashlib.sha256(b0 + b"\x01" + domain_prime).digest()]
    for block_index in range(2, ell + 1):
        previous = bytes(left ^ right for left, right in zip(b0, blocks[-1]))
        blocks.append(hashlib.sha256(previous + bytes((block_index,)) + domain_prime).digest())
    return b"".join(blocks)[:length]


def _hash_to_fr(message: bytes, domain: bytes, *, nonzero: bool) -> int:
    """Uniformly map an arbitrary transcript message to BN254 Fr."""

    return _hash_to_fr_parts((message,), domain, nonzero=nonzero)


def _hash_to_fr_parts(
    message_parts: tuple[bytes | bytearray | memoryview, ...],
    domain: bytes,
    *,
    nonzero: bool,
) -> int:
    """Hash concatenated message buffers to Fr without materializing them."""

    modulus = BN254_FR_MODULUS
    sample_space = 1 << (8 * SCALAR_BYTES)
    rejection_limit = sample_space - (sample_space % modulus)
    counter = 0
    while True:
        expanded = _expand_message_xmd_parts(
            (*message_parts, counter.to_bytes(4, "big")), domain, SCALAR_BYTES
        )
        candidate = int.from_bytes(expanded, "big")
        if candidate < rejection_limit:
            result = candidate % modulus
            if result != 0 or not nonzero:
                return result
        counter += 1
        if counter >= 1 << 32:
            raise EncodingError("hash-to-Fr rejection sampling exhausted")


@dataclass
class Transcript:
    """Mutable SHA-256 transcript with hash-to-BN254-Fr challenges."""

    domain: str | bytes

    def __post_init__(self) -> None:
        if isinstance(self.domain, str):
            domain_bytes = self.domain.encode("utf-8")
        elif isinstance(self.domain, bytes):
            domain_bytes = self.domain
        else:
            raise EncodingError("transcript domain must be text or bytes")
        self._state = bytearray(TRANSCRIPT_VERSION)
        # ``digest`` is queried at every proof phase boundary.  Maintain an
        # incremental hash alongside the canonical byte state so those
        # queries do not repeatedly re-hash the complete transcript.  The
        # byte state remains authoritative for challenge derivation, hence
        # this does not alter any transcript or proof bytes.
        self._state_hash = hashlib.sha256(TRANSCRIPT_VERSION)
        self._challenge_index = 0
        self._append(b"D", domain_bytes)

    def _append(self, tag: bytes, payload: bytes) -> None:
        encoded = tag + _length_prefix(payload)
        self._state.extend(encoded)
        self._state_hash.update(encoded)

    def absorb(self, label: str, message: Any) -> "Transcript":
        if not isinstance(label, str) or not label:
            raise EncodingError("transcript message label must be non-empty text")
        label_bytes = label.encode("utf-8")
        payload = canonical_encode(message)
        self._append(b"M", _length_prefix(label_bytes) + _length_prefix(payload))
        return self

    def absorb_bytes(self, label: str, message: bytes) -> "Transcript":
        if not isinstance(message, bytes):
            raise EncodingError("raw transcript message must be bytes")
        return self.absorb(label, message)

    def challenge_scalar(self, label: str, *, nonzero: bool = True) -> int:
        """Derive and absorb the next domain-separated Fr challenge.

        The state snapshot is hashed before the challenge marker is appended.
        This makes the ordering rule explicit: all prover messages up to this
        point are already in the hash input.  Absorbing the challenge itself
        means a later challenge is bound to this one as well.
        """

        if not isinstance(label, str) or not label:
            raise EncodingError("challenge label must be non-empty text")
        label_bytes = label.encode("utf-8")
        index = self._challenge_index
        material_prefix = (
            TRANSCRIPT_VERSION
            + _length_prefix(label_bytes)
            + index.to_bytes(8, "big")
        )
        value = _hash_to_fr_parts(
            (material_prefix, self._state),
            TRANSCRIPT_VERSION + b"/challenge",
            nonzero=nonzero,
        )
        self.absorb(
            f"challenge/{index}",
            {"label": label, "value": encode_scalar(value)},
        )
        self._challenge_index += 1
        return value

    challenge = challenge_scalar

    @property
    def state(self) -> bytes:
        return bytes(self._state)

    @property
    def digest(self) -> bytes:
        return self._state_hash.copy().digest()

    def copy(self) -> "Transcript":
        other = object.__new__(Transcript)
        other.domain = self.domain
        other._state = bytearray(self._state)
        other._state_hash = self._state_hash.copy()
        other._challenge_index = self._challenge_index
        return other
