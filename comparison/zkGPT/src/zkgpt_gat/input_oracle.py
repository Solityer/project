"""Verifier-derived linear forms over the canonical X|W|Aux|L input.

Private tensor terminals are always opened from the unique segment
commitments.  A proof cannot provide offsets, shapes, padding, or weights:
those are reconstructed from the externally authenticated ``Layout_in``.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .field import BN254_FR_MODULUS, next_power_of_two, to_field_int


def equality_weights(point: Sequence[int]) -> list[int]:
    values = [1]
    for raw in point:
        challenge = to_field_int(raw)
        values = [
            item
            for value in values
            for item in (
                value * (1 - challenge) % BN254_FR_MODULUS,
                value * challenge % BN254_FR_MODULUS,
            )
        ]
    return values


def array_descriptor(layout: Mapping[str, Any], segment: str, array_name: str) -> dict[str, Any]:
    segments = [item for item in layout.get("segments", []) if item.get("name") == segment]
    if len(segments) != 1:
        raise ValueError(f"missing or duplicate segment in Layout_in: {segment}")
    arrays = [item for item in segments[0].get("arrays", []) if item.get("name") == array_name]
    if len(arrays) != 1:
        raise ValueError(f"missing or duplicate array in Layout_in: {segment}/{array_name}")
    return dict(arrays[0])


def array_mle_linear_form(
    layout: Mapping[str, Any],
    segment: str,
    array_name: str,
    point: Sequence[int],
) -> tuple[list[int], dict[str, Any]]:
    descriptor = array_descriptor(layout, segment, array_name)
    n_in = int(layout.get("N_in", 0))
    if n_in <= 0 or n_in & (n_in - 1):
        raise ValueError("Layout_in has a noncanonical N_in")
    length = int(descriptor.get("length", -1))
    offset = int(descriptor.get("offset", -1))
    if length <= 0 or offset < 0 or offset + length > n_in:
        raise ValueError("array slice is outside Layout_in")
    local_domain = next_power_of_two(length)
    if len(point) != local_domain.bit_length() - 1:
        raise ValueError("array MLE point has wrong dimension")
    local = equality_weights(point)
    weights = [0] * n_in
    weights[offset: offset + length] = local[:length]
    linear_form = {
        "kind": "layout-array-mle",
        "segment": segment,
        "array_name": array_name,
        "offset": offset,
        "logical_length": length,
        "local_domain": local_domain,
        "shape": [int(value) for value in descriptor.get("shape", [])],
        "point": [str(to_field_int(value)) for value in point],
        "padding_rule": "local power-of-two zero extension inside authenticated Layout_in slice",
    }
    return weights, linear_form


def public_array_mle(values: Sequence[int], point: Sequence[int]) -> int:
    logical = [to_field_int(value) for value in values]
    if not logical:
        raise ValueError("public array MLE cannot be empty")
    domain = next_power_of_two(len(logical))
    if len(point) != domain.bit_length() - 1:
        raise ValueError("public array MLE point has wrong dimension")
    padded = logical + [0] * (domain - len(logical))
    return sum(a * b for a, b in zip(padded, equality_weights(point))) % BN254_FR_MODULUS
