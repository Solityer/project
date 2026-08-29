"""Canonical four-segment input layout for the zkGPT GAT relation."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .field import next_power_of_two, to_field_int
from .util import digest_json

ArrayList = Sequence[Tuple[str, np.ndarray]]
ArrayShapeList = Sequence[Tuple[str, Sequence[int]]]


def field_values(array: np.ndarray) -> Iterable[int]:
    value = np.asarray(array)
    if not (value.dtype == object or np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.bool_)):
        raise TypeError(f"canonical input segment requires integer array, got {value.dtype}")
    for item in np.ravel(value):
        yield to_field_int(int(item))


def logical_segment(name: str, arrays: ArrayList) -> Tuple[Dict[str, Any], List[int]]:
    offset = 0
    layout_arrays: List[Dict[str, Any]] = []
    values: List[int] = []
    seen: set[str] = set()
    for array_name, array in arrays:
        if array_name in seen:
            raise ValueError(f"duplicate {name} array name: {array_name}")
        seen.add(array_name)
        current = list(field_values(array))
        value = np.asarray(array)
        layout_arrays.append(
            {
                "name": array_name,
                "offset": offset,
                "length": len(current),
                "shape": [int(dim) for dim in value.shape],
                "encoding": "canonical_bn254_fr_integer",
            }
        )
        values.extend(current)
        offset += len(current)
    return {
        "name": name,
        "logical_length": len(values),
        "arrays": layout_arrays,
    }, values


def build_input_layout(
    x_public: ArrayList,
    private_segments: Mapping[str, ArrayList],
    lasso_family_registry: Sequence[Mapping[str, Any]] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, List[int]]]:
    ordered = [
        ("X_pub", x_public),
        ("W", private_segments["com_W"]),
        ("Aux", private_segments["com_Aux"]),
        ("L_qry", private_segments["com_L"]),
    ]
    descriptors: List[Dict[str, Any]] = []
    logical: Dict[str, List[int]] = {}
    for name, arrays in ordered:
        descriptor, values = logical_segment(name, arrays)
        descriptors.append(descriptor)
        logical[name] = values
    n_raw = max((len(values) for values in logical.values()), default=0)
    n_in = next_power_of_two(max(1, n_raw))
    for descriptor in descriptors:
        descriptor["padded_length"] = n_in
        descriptor["padding_begin"] = int(descriptor["logical_length"])
        descriptor["padding_value"] = "0"
        descriptor["segment_selector"] = {
            "X_pub": "00",
            "W": "01",
            "Aux": "10",
            "L_qry": "11",
        }[str(descriptor["name"])]
        if descriptor["name"] == "L_qry":
            families = [dict(item) for item in (lasso_family_registry or [])]
            if [item.get("offset") for item in families] != [
                item.get("offset") for item in descriptor["arrays"]
            ]:
                raise ValueError("Lasso family offsets do not match canonical L_qry arrays")
            for family, array in zip(families, descriptor["arrays"]):
                padded_count = int(family.get("padded_count", -1))
                if padded_count < 2 or padded_count & (padded_count - 1):
                    raise ValueError("Lasso family padded_count must be a power of two >= 2")
                if family.get("family") != array.get("name") or padded_count != int(array.get("length", -2)):
                    raise ValueError("Lasso family registry does not match L_qry array layout")
            descriptor["lasso_families"] = families
    padded = {
        name: values + [0] * (n_in - len(values))
        for name, values in logical.items()
    }
    layout = {
        "schema": "zkGPT_gat_four_segment_layout_v1",
        "segment_order": ["X_pub", "W", "Aux", "L_qry"],
        "selector_encoding": {"00": "X_pub", "01": "W", "10": "Aux", "11": "L_qry"},
        "N_raw": n_raw,
        "N_in": n_in,
        "input_layer_length": 4 * n_in,
        "padding_rule": "logical prefix followed by canonical zero padding; family-internal Lasso padding is part of logical L_qry",
        "segments": descriptors,
    }
    layout["layout_digest"] = digest_json(layout)
    return layout, padded


def build_input_layout_from_shapes(
    x_public: ArrayShapeList,
    private_segments: Mapping[str, ArrayShapeList],
    lasso_family_registry: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build the canonical layout from authenticated array shapes only.

    Model registration commits only ``W`` and therefore does not need to
    materialize the full GAT trace.  The layout digest nevertheless has to be
    identical to the one produced by :func:`build_input_layout` during a
    proof.  This helper performs that descriptor-only calculation without
    allocating the complete public feature or advice vectors.

    It is intentionally separate from the value-bearing builder so the
    formal proving path cannot accidentally accept shape-only input.
    """
    ordered = [
        ("X_pub", x_public),
        ("W", private_segments["com_W"]),
        ("Aux", private_segments["com_Aux"]),
        ("L_qry", private_segments["com_L"]),
    ]
    descriptors: List[Dict[str, Any]] = []
    logical_lengths: Dict[str, int] = {}
    for name, arrays in ordered:
        offset = 0
        layout_arrays: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for array_name, raw_shape in arrays:
            if array_name in seen:
                raise ValueError(f"duplicate {name} array name: {array_name}")
            seen.add(array_name)
            shape = tuple(int(dim) for dim in raw_shape)
            if any(dim < 0 for dim in shape):
                raise ValueError(f"negative {name} array shape: {array_name}")
            length = 1
            for dim in shape:
                length *= dim
            layout_arrays.append(
                {
                    "name": array_name,
                    "offset": offset,
                    "length": length,
                    "shape": list(shape),
                    "encoding": "canonical_bn254_fr_integer",
                }
            )
            offset += length
        descriptors.append({"name": name, "logical_length": offset, "arrays": layout_arrays})
        logical_lengths[name] = offset

    n_raw = max(logical_lengths.values(), default=0)
    n_in = next_power_of_two(max(1, n_raw))
    selector_encoding = {"X_pub": "00", "W": "01", "Aux": "10", "L_qry": "11"}
    for descriptor in descriptors:
        name = str(descriptor["name"])
        descriptor["padded_length"] = n_in
        descriptor["padding_begin"] = int(descriptor["logical_length"])
        descriptor["padding_value"] = "0"
        descriptor["segment_selector"] = selector_encoding[name]
        if name == "L_qry":
            families = [dict(item) for item in (lasso_family_registry or [])]
            if [item.get("offset") for item in families] != [
                item.get("offset") for item in descriptor["arrays"]
            ]:
                raise ValueError("Lasso family offsets do not match canonical L_qry arrays")
            for family, array in zip(families, descriptor["arrays"]):
                padded_count = int(family.get("padded_count", -1))
                if padded_count < 2 or padded_count & (padded_count - 1):
                    raise ValueError("Lasso family padded_count must be a power of two >= 2")
                if family.get("family") != array.get("name") or padded_count != int(array.get("length", -2)):
                    raise ValueError("Lasso family registry does not match L_qry array layout")
            descriptor["lasso_families"] = families
    layout = {
        "schema": "zkGPT_gat_four_segment_layout_v1",
        "segment_order": ["X_pub", "W", "Aux", "L_qry"],
        "selector_encoding": {"00": "X_pub", "01": "W", "10": "Aux", "11": "L_qry"},
        "N_raw": n_raw,
        "N_in": n_in,
        "input_layer_length": 4 * n_in,
        "padding_rule": "logical prefix followed by canonical zero padding; family-internal Lasso padding is part of logical L_qry",
        "segments": descriptors,
    }
    layout["layout_digest"] = digest_json(layout)
    return layout


def segment_descriptor(layout: Dict[str, Any], name: str) -> Dict[str, Any]:
    matches = [item for item in layout.get("segments", []) if item.get("name") == name]
    if len(matches) != 1:
        raise KeyError(f"layout segment not found: {name}")
    return matches[0]
