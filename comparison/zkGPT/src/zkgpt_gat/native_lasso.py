"""Native Lasso memory checking for the formal zkGPT GAT adapter.

Each family owns a fresh, per-proof ``LC_f`` commitment.  The complete
ordered LC set is absorbed before the first family challenge.  Lasso-only
memory advice never enters the canonical ``X|W|Aux|L_qry`` commitment.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Mapping, MutableSequence, Sequence

from .field import BN254_FR_MODULUS, log2_power_of_two, next_power_of_two, to_field_int
from .gpu_execution import active_lasso_product, active_product_coefficients
from .input_oracle import array_descriptor, equality_weights
from .native_gkr import (
    LinearGateSpec, LinearTerm, OracleRef, ProductGateSpec,
    prove_linear_gate, prove_product_gate, verify_linear_gate, verify_product_gate,
)
from .native_hyrax import (
    NativeHyrax, _point, hyrax_value_params,
    verify_batched_linear_claim_payload, verify_open_payload,
)
from .transcript import Transcript
from .util import digest_json
from .value_commitment import (
    CrossParamsSameValueProof, MultiplicationProof, ValueCommitment,
    prove_cross_params_same_value, prove_multiplication,
    verify_cross_params_same_value, verify_multiplication,
)
from .zk_sumcheck import prove_committed_sumcheck, verify_committed_sumcheck


PROTOCOL_ID = "zkGPT/native-lasso-memory-v1"
LOCAL_COMBINE_ID = "zkGPT/native-lasso-local-claim-reduction-v1"
REGISTRATION_FIELDS = {"backend", "parameter_id", "label", "vector_length", "commitment"}


def _public_registration(registration: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in registration.items() if key != "blind"}


def _bits(value: int, width: int) -> list[int]:
    if value < 0 or value >= 1 << width:
        raise ValueError("Lasso integer does not fit its public bit width")
    return [(value >> bit) & 1 for bit in range(width)]


def _layout_segment(name: str, arrays: Sequence[tuple[str, int]]) -> tuple[dict[str, Any], int]:
    offset = 0
    descriptors = []
    for array_name, length in arrays:
        if length <= 0:
            raise ValueError("Lasso arrays must be non-empty")
        descriptors.append({"name": array_name, "offset": offset, "length": length})
        offset += length
    return {"name": name, "logical_length": offset, "arrays": descriptors}, offset


def _layout(family: Mapping[str, Any], table_size: int) -> dict[str, Any]:
    query_count = int(family["padded_count"])
    address_width = max(1, (table_size - 1).bit_length())
    timestamp_width = max(1, query_count.bit_length())
    product_domain = next_power_of_two(query_count + table_size)
    lc_arrays = [
        ("query", query_count), ("address", query_count),
        ("read_timestamp", query_count), ("write_timestamp", query_count),
        ("multiplicity", table_size),
        ("address_bits", query_count * address_width),
        ("address_slack_bits", query_count * address_width),
        ("read_timestamp_bits", query_count * timestamp_width),
        ("read_timestamp_slack_bits", query_count * timestamp_width),
        ("write_timestamp_bits", query_count * timestamp_width),
        ("write_timestamp_slack_bits", query_count * timestamp_width),
        ("multiplicity_bits", table_size * timestamp_width),
        ("multiplicity_slack_bits", table_size * timestamp_width),
    ]
    gp_arrays = [
        ("left_factor", product_domain), ("right_factor", product_domain),
        ("left_accumulator", product_domain + 1),
        ("right_accumulator", product_domain + 1),
    ]
    lc, lc_length = _layout_segment("LC", lc_arrays)
    gp, gp_length = _layout_segment("GP", gp_arrays)
    domain = next_power_of_two(max(lc_length, gp_length))
    result = {
        "schema": "zkGPT_lasso_family_lc_layout_v1",
        "family": str(family["family"]), "table_id": str(family["table_id"]),
        "query_count": query_count, "table_size": table_size,
        "address_width": address_width, "timestamp_width": timestamp_width,
        "address_base": table_size, "timestamp_base": query_count + 1,
        "product_domain": product_domain, "N_in": domain,
        "segments": [lc, gp],
    }
    result["layout_digest"] = digest_json(result)
    return result


def _array(layout: Mapping[str, Any], segment: str, name: str) -> Mapping[str, Any]:
    return array_descriptor(layout, segment, name)


def _oracle(layout: Mapping[str, Any], segment: str, name: str, indices: Sequence[int] | None = None) -> OracleRef:
    length = int(_array(layout, segment, name)["length"])
    return OracleRef(segment, name, tuple(range(length)) if indices is None else tuple(int(value) for value in indices))


def _flatten(layout: Mapping[str, Any], segment: str, arrays: Mapping[str, Sequence[int]]) -> list[int]:
    descriptor = next(item for item in layout["segments"] if item["name"] == segment)
    out: list[int] = []
    for item in descriptor["arrays"]:
        values = [to_field_int(value) for value in arrays[item["name"]]]
        if len(values) != int(item["length"]):
            raise ValueError("Lasso witness array has wrong length")
        out.extend(values)
    out.extend([0] * (int(layout["N_in"]) - len(out)))
    return out


def _lc_witness(query: Sequence[int], table: Sequence[int], layout: Mapping[str, Any]) -> dict[str, list[int]]:
    table_index: dict[int, int] = {}
    for index, raw in enumerate(table):
        value = to_field_int(raw)
        if value in table_index:
            raise ValueError("Lasso fixed table encoding is not injective")
        table_index[value] = index
    query_values = [to_field_int(value) for value in query]
    try:
        addresses = [table_index[value] for value in query_values]
    except KeyError as exc:
        raise ValueError("canonical lookup query is outside the authenticated table") from exc
    counts = [0] * len(table)
    read: list[int] = []
    for address in addresses:
        read.append(counts[address])
        counts[address] += 1
    write = [value + 1 for value in read]
    aw, tw = int(layout["address_width"]), int(layout["timestamp_width"])
    q = len(query_values)
    return {
        "query": query_values, "address": addresses,
        "read_timestamp": read, "write_timestamp": write, "multiplicity": counts,
        "address_bits": [bit for value in addresses for bit in _bits(value, aw)],
        "address_slack_bits": [bit for value in addresses for bit in _bits(len(table) - 1 - value, aw)],
        "read_timestamp_bits": [bit for value in read for bit in _bits(value, tw)],
        "read_timestamp_slack_bits": [bit for value in read for bit in _bits(q - value, tw)],
        "write_timestamp_bits": [bit for value in write for bit in _bits(value, tw)],
        "write_timestamp_slack_bits": [bit for value in write for bit in _bits(q - value, tw)],
        "multiplicity_bits": [bit for value in counts for bit in _bits(value, tw)],
        "multiplicity_slack_bits": [bit for value in counts for bit in _bits(q - value, tw)],
    }


def _gp_witness(lc: Mapping[str, Sequence[int]], table: Sequence[int], layout: Mapping[str, Any], beta: int) -> dict[str, list[int]]:
    q, t, domain = int(layout["query_count"]), int(layout["table_size"]), int(layout["product_domain"])
    ab, tb = int(layout["address_base"]), int(layout["timestamp_base"])
    beta = to_field_int(beta)

    def packed(address: int, timestamp: int, value: int) -> int:
        integer = address + ab * timestamp + ab * tb * int(value)
        if integer < 0 or integer >= BN254_FR_MODULUS:
            raise ValueError("Lasso injective tuple packing exceeds BN254 Fr")
        return integer

    initial = [(beta + packed(i, 0, value)) % BN254_FR_MODULUS for i, value in enumerate(table)]
    writes = [(beta + packed(int(lc["address"][i]), int(lc["write_timestamp"][i]), int(lc["query"][i]))) % BN254_FR_MODULUS for i in range(q)]
    reads = [(beta + packed(int(lc["address"][i]), int(lc["read_timestamp"][i]), int(lc["query"][i]))) % BN254_FR_MODULUS for i in range(q)]
    final = [(beta + packed(i, int(lc["multiplicity"][i]), value)) % BN254_FR_MODULUS for i, value in enumerate(table)]
    left_factor, right_factor = initial + writes, reads + final
    left_factor.extend([1] * (domain - len(left_factor)))
    right_factor.extend([1] * (domain - len(right_factor)))

    def accumulate(factors: Sequence[int]) -> list[int]:
        out = [1]
        for value in factors:
            out.append(out[-1] * to_field_int(value) % BN254_FR_MODULUS)
        return out
    left_accumulator, right_accumulator = accumulate(left_factor), accumulate(right_factor)
    family = str(layout["family"])
    gpu_left = active_lasso_product(left_factor, f"lasso/{family}/left-grand-product")
    gpu_right = active_lasso_product(right_factor, f"lasso/{family}/right-grand-product")
    if gpu_left is not None:
        if gpu_left != left_accumulator[-1] or gpu_right != right_accumulator[-1]:
            raise ValueError("GPU Lasso grand product differs from exact memory trace")
        # These GPU-authored values are the roots committed in GP_f.
        left_accumulator[-1], right_accumulator[-1] = gpu_left, gpu_right
    return {"left_factor": left_factor, "right_factor": right_factor,
            "left_accumulator": left_accumulator, "right_accumulator": right_accumulator}


def _bit_terms(segment: str, name: str, rows: int, width: int, sign: int = 1) -> tuple[LinearTerm, ...]:
    return tuple(LinearTerm(sign * (1 << bit), OracleRef(segment, name, tuple(row * width + bit for row in range(rows)))) for bit in range(width))


def _gate_specs(layout: Mapping[str, Any], table: Sequence[int], beta: int) -> list[tuple[str, LinearGateSpec | ProductGateSpec]]:
    q, t, d = int(layout["query_count"]), int(layout["table_size"]), int(layout["product_domain"])
    aw, tw = int(layout["address_width"]), int(layout["timestamp_width"])
    ab, tb = int(layout["address_base"]), int(layout["timestamp_base"])
    canonical_table = [int(value) for value in table]
    if (
        len(canonical_table) != t
        or len(set(canonical_table)) != t
        or any(value < 0 or value >= BN254_FR_MODULUS for value in canonical_table)
        or (t - 1) + ab * q + ab * tb * max(canonical_table, default=0) >= BN254_FR_MODULUS
    ):
        raise ValueError("Lasso fixed table does not admit injective no-wrap memory tuples")
    zero_q = (0,) * q
    specs: list[tuple[str, LinearGateSpec | ProductGateSpec]] = [
        ("linear", LinearGateSpec("memory/write-timestamp", _oracle(layout, "LC", "write_timestamp"), (LinearTerm(1, _oracle(layout, "LC", "read_timestamp")),), (1,) * q)),
    ]
    for name, rows in (("address_bits", q), ("address_slack_bits", q), ("read_timestamp_bits", q), ("read_timestamp_slack_bits", q), ("write_timestamp_bits", q), ("write_timestamp_slack_bits", q), ("multiplicity_bits", t), ("multiplicity_slack_bits", t)):
        bits = _oracle(layout, "LC", name)
        specs.append(("product", ProductGateSpec(f"memory/{name}/boolean", bits, bits, bits)))
    specs.extend([
        ("linear", LinearGateSpec("memory/address/reconstruct", _oracle(layout, "LC", "address"), _bit_terms("LC", "address_bits", q, aw), zero_q)),
        ("linear", LinearGateSpec("memory/address/upper-bound", _oracle(layout, "LC", "address"), _bit_terms("LC", "address_slack_bits", q, aw, -1), (t - 1,) * q)),
        ("linear", LinearGateSpec("memory/read-timestamp/reconstruct", _oracle(layout, "LC", "read_timestamp"), _bit_terms("LC", "read_timestamp_bits", q, tw), zero_q)),
        ("linear", LinearGateSpec("memory/read-timestamp/upper-bound", _oracle(layout, "LC", "read_timestamp"), _bit_terms("LC", "read_timestamp_slack_bits", q, tw, -1), (q,) * q)),
        ("linear", LinearGateSpec("memory/write-timestamp/reconstruct", _oracle(layout, "LC", "write_timestamp"), _bit_terms("LC", "write_timestamp_bits", q, tw), zero_q)),
        ("linear", LinearGateSpec("memory/write-timestamp/upper-bound", _oracle(layout, "LC", "write_timestamp"), _bit_terms("LC", "write_timestamp_slack_bits", q, tw, -1), (q,) * q)),
        ("linear", LinearGateSpec("memory/multiplicity/reconstruct", _oracle(layout, "LC", "multiplicity"), _bit_terms("LC", "multiplicity_bits", t, tw), (0,) * t)),
        ("linear", LinearGateSpec("memory/multiplicity/upper-bound", _oracle(layout, "LC", "multiplicity"), _bit_terms("LC", "multiplicity_slack_bits", t, tw, -1), (q,) * t)),
    ])
    constants = tuple((to_field_int(beta) + i + ab * tb * int(value)) % BN254_FR_MODULUS for i, value in enumerate(table))
    specs.extend([
        ("linear", LinearGateSpec("memory/left/initial-factors", _oracle(layout, "GP", "left_factor", range(t)), (), constants)),
        ("linear", LinearGateSpec("memory/left/write-factors", _oracle(layout, "GP", "left_factor", range(t, t + q)), (LinearTerm(1, _oracle(layout, "LC", "address")), LinearTerm(ab, _oracle(layout, "LC", "write_timestamp")), LinearTerm(ab * tb, _oracle(layout, "LC", "query"))), (to_field_int(beta),) * q)),
        ("linear", LinearGateSpec("memory/right/read-factors", _oracle(layout, "GP", "right_factor", range(q)), (LinearTerm(1, _oracle(layout, "LC", "address")), LinearTerm(ab, _oracle(layout, "LC", "read_timestamp")), LinearTerm(ab * tb, _oracle(layout, "LC", "query"))), (to_field_int(beta),) * q)),
        ("linear", LinearGateSpec("memory/right/final-factors", _oracle(layout, "GP", "right_factor", range(q, q + t)), (LinearTerm(ab, _oracle(layout, "LC", "multiplicity")),), constants)),
    ])
    used = q + t
    if used < d:
        padding = tuple(range(used, d))
        specs.extend([
            ("linear", LinearGateSpec("memory/left/padding-factors", _oracle(layout, "GP", "left_factor", padding), (), (1,) * len(padding))),
            ("linear", LinearGateSpec("memory/right/padding-factors", _oracle(layout, "GP", "right_factor", padding), (), (1,) * len(padding))),
        ])
    specs.extend([
        ("linear", LinearGateSpec("memory/left/accumulator-one", _oracle(layout, "GP", "left_accumulator", (0,)), (), (1,))),
        ("linear", LinearGateSpec("memory/right/accumulator-one", _oracle(layout, "GP", "right_accumulator", (0,)), (), (1,))),
        ("product", ProductGateSpec("memory/left/accumulator-recurrence", _oracle(layout, "GP", "left_accumulator", range(1, d + 1)), _oracle(layout, "GP", "left_accumulator", range(d)), _oracle(layout, "GP", "left_factor", range(d)))),
        ("product", ProductGateSpec("memory/right/accumulator-recurrence", _oracle(layout, "GP", "right_accumulator", range(1, d + 1)), _oracle(layout, "GP", "right_accumulator", range(d)), _oracle(layout, "GP", "right_factor", range(d)))),
        ("linear", LinearGateSpec("memory/grand-product-equality", _oracle(layout, "GP", "left_accumulator", (d,)), (LinearTerm(1, _oracle(layout, "GP", "right_accumulator", (d,))),), (0,))),
    ])
    return specs


class _ProductProvider:
    def __init__(self, left: Sequence[int], right: Sequence[int], label: str = "lasso-local-product") -> None:
        if len(left) != len(right) or not left or len(left) & (len(left) - 1):
            raise ValueError("Lasso combine requires equal power-of-two vectors")
        self.left = [to_field_int(value) for value in left]
        self.right = [to_field_int(value) for value in right]
        self.label = label

    def __call__(self, _round: int, current_claim: int, _transcript: Transcript) -> list[int]:
        c0 = c1 = c2 = 0
        for a0, a1, b0, b1 in zip(self.left[0::2], self.left[1::2], self.right[0::2], self.right[1::2]):
            da, db = (a1 - a0) % BN254_FR_MODULUS, (b1 - b0) % BN254_FR_MODULUS
            c0 = (c0 + a0 * b0) % BN254_FR_MODULUS
            c1 = (c1 + da * b0 + a0 * db) % BN254_FR_MODULUS
            c2 = (c2 + da * db) % BN254_FR_MODULUS
        cpu_coefficients = [c0, c1, c2]
        gpu_coefficients = active_product_coefficients(
            (self.left, self.right), f"{self.label}/round/{_round}"
        )
        coefficients = gpu_coefficients if gpu_coefficients is not None else cpu_coefficients
        if coefficients != cpu_coefficients:
            raise ValueError("GPU Lasso combine coefficients differ from exact arithmetic")
        if (2 * coefficients[0] + coefficients[1] + coefficients[2]) % BN254_FR_MODULUS != to_field_int(current_claim):
            raise ValueError("Lasso local claim reduction mismatch")
        return coefficients

    def advance(self, challenge: int) -> None:
        r = to_field_int(challenge)
        fold = lambda values: [(a + r * (b - a)) % BN254_FR_MODULUS for a, b in zip(values[0::2], values[1::2])]
        self.left, self.right = fold(self.left), fold(self.right)


def _local_combine_prove(label: str, hyrax: NativeHyrax, vectors: Mapping[str, Sequence[int]], registrations: Mapping[str, Mapping[str, Any]], transcript: Transcript, work_dir: pathlib.Path, context: Mapping[str, Any]) -> dict[str, Any]:
    sources = [item for item in hyrax.opening_log if item.get("kind") == "linear"]
    if not sources:
        raise ValueError("Lasso family has no bound local claims")
    names, domain = ("LC", "GP"), len(vectors["LC"])
    coefficients, unified = [0] * (2 * domain), list(vectors["LC"]) + list(vectors["GP"])
    initial_value = initial_blind = 0
    initial_point = None
    source_payloads = []
    registration_to_segment = {registrations[name]["label"]: name for name in names}
    for index, source in enumerate(sources):
        segment = registration_to_segment.get(source.get("registration_label"))
        if segment is None:
            raise ValueError("Lasso local claim uses an unknown registration")
        rho = transcript.challenge_scalar(f"{label}/local-combine/source/{index}", nonzero=True)
        base = names.index(segment) * domain
        items = source["weights"].items() if isinstance(source["weights"], Mapping) else enumerate(source["weights"])
        for position, coefficient in items:
            coefficients[base + int(position)] = (coefficients[base + int(position)] + rho * to_field_int(coefficient)) % BN254_FR_MODULUS
        state = source["state"]
        initial_value = (initial_value + rho * state.evaluation) % BN254_FR_MODULUS
        initial_blind = (initial_blind + rho * state.blind) % BN254_FR_MODULUS
        point = rho * state.evaluation_commitment.point
        initial_point = point if initial_point is None else initial_point + point
        source_payloads.append({"index": index, "label": source["label"], "segment": segment, "linear_form": source["linear_form"], "evaluation_commitment": state.evaluation_commitment.to_dict(), "weight": str(rho)})
    if initial_point is None or sum(a * b for a, b in zip(coefficients, unified)) % BN254_FR_MODULUS != initial_value:
        raise ValueError("Lasso local source aggregation is inconsistent")
    params, initial = hyrax_value_params(domain), ValueCommitment(initial_point)
    reduction = prove_committed_sumcheck(f"{label}/local-combine/sumcheck", initial, _ProductProvider(coefficients, unified, f"{label}/local-combine"), 2, variable_count=log2_power_of_two(2 * domain), params=params, transcript=transcript, initial_value=initial_value, initial_blind=initial_blind)
    full_point = tuple(reversed(reduction.challenges)); selector, point = full_point[:1], full_point[1:]
    states, opening_payloads = {}, {}
    for name in names:
        state = hyrax.open_registered(f"{label}/local-combine/open/{name}", vectors[name], point, registrations[name], work_dir, transcript=transcript, public_context=context)
        states[name], opening_payloads[name] = state, state.proof
    selectors = equality_weights(selector)
    combined_value = sum(selectors[i] * states[name].evaluation for i, name in enumerate(names)) % BN254_FR_MODULUS
    combined_blind = sum(selectors[i] * states[name].blind for i, name in enumerate(names)) % BN254_FR_MODULUS
    combined = ValueCommitment(selectors[0] * states["LC"].evaluation_commitment.point + selectors[1] * states["GP"].evaluation_commitment.point)
    coefficient_eval = sum(a * b for a, b in zip(coefficients, equality_weights(full_point))) % BN254_FR_MODULUS
    terminal = prove_multiplication(params, ValueCommitment(coefficient_eval * params.U), combined, reduction.terminal_commitment, coefficient_eval, combined_value, reduction.terminal_value, 0, combined_blind, reduction.blind, label=f"{label}/local-combine/terminal", transcript=transcript)
    return {"protocol": LOCAL_COMBINE_ID, "sources": source_payloads, "initial_claim_commitment": initial.to_dict(), "sumcheck": reduction.proof.to_dict(), "selector_point": [str(v) for v in selector], "point": [str(v) for v in point], "openings": opening_payloads, "combined_commitment": combined.to_dict(), "coefficient_evaluation": str(coefficient_eval), "terminal_product": terminal.to_dict(), "transcript_digest": transcript.digest.hex()}


def _local_combine_verify(label: str, payload: Mapping[str, Any], sources: Sequence[Mapping[str, Any]], registrations: Mapping[str, Mapping[str, Any]], domain: int, transcript: Transcript, context: Mapping[str, Any]) -> bool:
    try:
        names, source_payloads = ("LC", "GP"), payload.get("sources")
        if set(payload) != {"protocol", "sources", "initial_claim_commitment", "sumcheck", "selector_point", "point", "openings", "combined_commitment", "coefficient_evaluation", "terminal_product", "transcript_digest"} or payload.get("protocol") != LOCAL_COMBINE_ID or not sources or not isinstance(source_payloads, list) or len(source_payloads) != len(sources) or set(payload.get("openings", {})) != set(names):
            return False
        coefficients, initial_point = [0] * (2 * domain), None
        for index, (wire, source) in enumerate(zip(source_payloads, sources)):
            rho = transcript.challenge_scalar(f"{label}/local-combine/source/{index}", nonzero=True)
            segment = str(source["segment"])
            expected = {"index": index, "label": source["label"], "segment": segment, "linear_form": source["linear_form"], "evaluation_commitment": source["evaluation_commitment"].to_dict(), "weight": str(rho)}
            if wire != expected or segment not in names:
                return False
            base = names.index(segment) * domain
            items = source["weights"].items() if isinstance(source["weights"], Mapping) else enumerate(source["weights"])
            for position, coefficient in items:
                coefficients[base + int(position)] = (coefficients[base + int(position)] + rho * to_field_int(coefficient)) % BN254_FR_MODULUS
            point = rho * source["evaluation_commitment"].point
            initial_point = point if initial_point is None else initial_point + point
        initial = ValueCommitment(initial_point)
        if ValueCommitment.from_dict(payload.get("initial_claim_commitment")) != initial:
            return False
        params = hyrax_value_params(domain)
        reduction = verify_committed_sumcheck(f"{label}/local-combine/sumcheck", payload.get("sumcheck", {}), initial, log2_power_of_two(2 * domain), 2, params=params, transcript=transcript)
        full_point = tuple(reversed(reduction.challenges)); selector, point = full_point[:1], full_point[1:]
        if payload.get("selector_point") != [str(v) for v in selector] or payload.get("point") != [str(v) for v in point]:
            return False
        evaluations = {}
        for name in names:
            if verify_open_payload(f"{label}/local-combine/open/{name}", payload["openings"][name], point, registrations[name], domain, transcript=transcript, public_context=context).get("status") != "VERIFY_OK":
                return False
            evaluations[name] = ValueCommitment(_point(payload["openings"][name]["evaluation_commitment"]))
        selectors = equality_weights(selector)
        combined = ValueCommitment(selectors[0] * evaluations["LC"].point + selectors[1] * evaluations["GP"].point)
        coefficient_eval = sum(a * b for a, b in zip(coefficients, equality_weights(full_point))) % BN254_FR_MODULUS
        return ValueCommitment.from_dict(payload.get("combined_commitment")) == combined and payload.get("coefficient_evaluation") == str(coefficient_eval) and verify_multiplication(params, ValueCommitment(coefficient_eval * params.U), combined, reduction.terminal_commitment, MultiplicationProof.from_dict(payload.get("terminal_product")), label=f"{label}/local-combine/terminal", transcript=transcript) and payload.get("transcript_digest") == transcript.digest.hex()
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


@dataclass
class _FamilyState:
    family: Mapping[str, Any]
    table: list[int]
    layout: dict[str, Any]
    lc_arrays: dict[str, list[int]]
    lc_vector: list[int]
    lc_registration: dict[str, Any]


def _query_weights(layout: Mapping[str, Any], point: Sequence[int]) -> dict[int, int]:
    query = _array(layout, "LC", "query")
    return {int(query["offset"]) + index: value for index, value in enumerate(equality_weights(point)) if value}


def _canonical_query_weights(canonical_layout: Mapping[str, Any], family: Mapping[str, Any], point: Sequence[int]) -> dict[int, int]:
    descriptor = array_descriptor(canonical_layout, "L_qry", str(family["family"]))
    return {int(descriptor["offset"]) + index: value for index, value in enumerate(equality_weights(point)) if value}


def prove_lasso(metadata: Mapping[str, Any], table_rows: Mapping[str, Sequence[int]], canonical_hyrax: NativeHyrax, canonical_l_vector: Sequence[int], canonical_l_registration: Mapping[str, Any], *, zk_root: pathlib.Path, work_dir: pathlib.Path, transcript: Transcript) -> dict[str, Any]:
    canonical_layout = metadata["Layout_in"]
    families = next(item for item in canonical_layout["segments"] if item["name"] == "L_qry")["lasso_families"]
    states: list[_FamilyState] = []
    local_hyrax = NativeHyrax(zk_root)
    for index, family in enumerate(families):
        table = [to_field_int(value) for value in table_rows[str(family["table_id"])]]
        layout = _layout(family, len(table))
        descriptor = array_descriptor(canonical_layout, "L_qry", str(family["family"]))
        offset, length = int(descriptor["offset"]), int(descriptor["length"])
        lc_arrays = _lc_witness(canonical_l_vector[offset:offset + length], table, layout)
        lc_vector = _flatten(layout, "LC", lc_arrays)
        registration = local_hyrax.commit_vector(f"LC_f/{index}/{family['family']}", lc_vector, work_dir)
        states.append(_FamilyState(family, table, layout, lc_arrays, lc_vector, registration))
    header = [{"index": index, "family": state.family["family"], "layout": state.layout, "registration": _public_registration(state.lc_registration)} for index, state in enumerate(states)]
    transcript.absorb("lasso/ordered-lc-header", header)
    family_proofs = []
    for index, state in enumerate(states):
        label = f"lasso/family/{index}/{state.family['family']}"
        beta = transcript.challenge_scalar(f"{label}/beta", nonzero=True)
        gp_arrays = _gp_witness(state.lc_arrays, state.table, state.layout, beta)
        gp_vector = _flatten(state.layout, "GP", gp_arrays)
        gp_registration = local_hyrax.commit_vector(f"GP_f/{index}/{state.family['family']}", gp_vector, work_dir)
        gp_public = _public_registration(gp_registration)
        transcript.absorb(f"{label}/gp-registration", gp_public)
        context = {"metadata_digest": digest_json(dict(metadata)), "Layout_in_digest": metadata["Layout_in_digest"], "family": state.family, "lc_layout_digest": state.layout["layout_digest"]}
        point = [transcript.challenge_scalar(f"{label}/query-point/{bit}") for bit in range(log2_power_of_two(int(state.family["padded_count"])))]
        lc_weights, l_weights = _query_weights(state.layout, point), _canonical_query_weights(canonical_layout, state.family, point)
        lc_form = {"kind": "lasso-lc-query", "family": state.family["family"], "point": [str(v) for v in point]}
        l_form = {"kind": "lasso-canonical-query", "family": state.family["family"], "point": [str(v) for v in point]}
        lc_claim = local_hyrax.commit_linear_claim_registered(f"{label}/query/lc", state.lc_vector, lc_weights, state.lc_registration, work_dir, transcript=transcript, linear_form=lc_form, public_context=context)
        l_claim = canonical_hyrax.commit_linear_claim_registered(f"{label}/query/canonical", canonical_l_vector, l_weights, canonical_l_registration, work_dir, transcript=transcript, linear_form=l_form, public_context=context)
        if lc_claim.evaluation != l_claim.evaluation:
            raise ValueError("LC query copy differs from canonical L_qry")
        link = prove_cross_params_same_value(hyrax_value_params(int(state.layout["N_in"])), lc_claim.evaluation_commitment, hyrax_value_params(int(canonical_layout["N_in"])), l_claim.evaluation_commitment, lc_claim.evaluation, lc_claim.blind, l_claim.blind, label=f"{label}/query/link", transcript=transcript)
        vectors, registrations = {"LC": state.lc_vector, "GP": gp_vector}, {"LC": state.lc_registration, "GP": gp_registration}
        gate_payloads = []
        for kind, spec in _gate_specs(state.layout, state.table, beta):
            prover = prove_linear_gate if kind == "linear" else prove_product_gate
            gate_payloads.append({"kind": kind, "proof": prover(spec, layout=state.layout, vectors=vectors, registrations=registrations, params=hyrax_value_params(int(state.layout["N_in"])), hyrax=local_hyrax, work_dir=work_dir, transcript=transcript, context=context)})
        local_combine = _local_combine_prove(label, local_hyrax, vectors, registrations, transcript, work_dir, context)
        local_hyrax.opening_log.clear()
        family_proofs.append({
            "index": index, "family": state.family["family"],
            "descriptor": state.family, "table_id": state.family["table_id"],
            "query_count": int(state.family["padded_count"]), "beta": str(beta),
            "gp_registration": gp_public,
            "query_link": {"point": [str(v) for v in point], "lc_form": lc_form, "lc_claim": lc_claim.proof, "canonical_form": l_form, "canonical_claim": l_claim.proof, "same_value": link.to_dict()},
            "gates": gate_payloads, "local_combine": local_combine,
        })
    return {"protocol": PROTOCOL_ID, "fixed_table_registry_digest": metadata["static_table_registry_digest"], "lc_header": header, "families": family_proofs, "transcript_digest": transcript.digest.hex()}


def verify_lasso(proof: Mapping[str, Any], metadata: Mapping[str, Any], table_rows: Mapping[str, Sequence[int]], canonical_l_registration: Mapping[str, Any], *, transcript: Transcript, canonical_source_sink: MutableSequence[dict[str, Any]]) -> bool:
    try:
        canonical_layout = metadata["Layout_in"]
        families = next(item for item in canonical_layout["segments"] if item["name"] == "L_qry")["lasso_families"]
        header, wires = proof.get("lc_header"), proof.get("families")
        if set(proof) != {"protocol", "fixed_table_registry_digest", "lc_header", "families", "transcript_digest"} or proof.get("protocol") != PROTOCOL_ID or proof.get("fixed_table_registry_digest") != metadata["static_table_registry_digest"] or not isinstance(header, list) or not isinstance(wires, list) or len(header) != len(families) or len(wires) != len(families) or not families:
            return False
        expected_states = []
        for index, (family, item) in enumerate(zip(families, header)):
            table = [to_field_int(v) for v in table_rows[str(family["table_id"])]]
            layout, registration = _layout(family, len(table)), item.get("registration")
            if item != {"index": index, "family": family["family"], "layout": layout, "registration": registration} or not isinstance(registration, Mapping) or set(registration) != REGISTRATION_FIELDS or registration.get("label") != f"LC_f/{index}/{family['family']}" or int(registration.get("vector_length", -1)) != int(layout["N_in"]):
                return False
            expected_states.append((family, table, layout, registration))
        transcript.absorb("lasso/ordered-lc-header", header)
        for index, ((family, table, layout, lc_registration), wire) in enumerate(zip(expected_states, wires)):
            label = f"lasso/family/{index}/{family['family']}"
            beta = transcript.challenge_scalar(f"{label}/beta", nonzero=True)
            gp_registration = wire.get("gp_registration")
            if set(wire) != {"index", "family", "descriptor", "table_id", "query_count", "beta", "gp_registration", "query_link", "gates", "local_combine"} or wire.get("index") != index or wire.get("family") != family["family"] or wire.get("descriptor") != family or wire.get("table_id") != family["table_id"] or wire.get("query_count") != int(family["padded_count"]) or wire.get("beta") != str(beta) or not isinstance(gp_registration, Mapping) or set(gp_registration) != REGISTRATION_FIELDS or gp_registration.get("label") != f"GP_f/{index}/{family['family']}" or int(gp_registration.get("vector_length", -1)) != int(layout["N_in"]):
                return False
            transcript.absorb(f"{label}/gp-registration", gp_registration)
            context = {"metadata_digest": digest_json(dict(metadata)), "Layout_in_digest": metadata["Layout_in_digest"], "family": family, "lc_layout_digest": layout["layout_digest"]}
            link_payload = wire.get("query_link")
            point = [transcript.challenge_scalar(f"{label}/query-point/{bit}") for bit in range(log2_power_of_two(int(family["padded_count"])))]
            lc_weights, l_weights = _query_weights(layout, point), _canonical_query_weights(canonical_layout, family, point)
            lc_form = {"kind": "lasso-lc-query", "family": family["family"], "point": [str(v) for v in point]}
            l_form = {"kind": "lasso-canonical-query", "family": family["family"], "point": [str(v) for v in point]}
            if not isinstance(link_payload, Mapping) or set(link_payload) != {"point", "lc_form", "lc_claim", "canonical_form", "canonical_claim", "same_value"} or link_payload.get("point") != [str(v) for v in point] or link_payload.get("lc_form") != lc_form or link_payload.get("canonical_form") != l_form:
                return False
            if verify_batched_linear_claim_payload(f"{label}/query/lc", link_payload.get("lc_claim", {}), lc_weights, lc_registration, int(layout["N_in"]), transcript=transcript, linear_form=lc_form, public_context=context).get("status") != "VERIFY_OK" or verify_batched_linear_claim_payload(f"{label}/query/canonical", link_payload.get("canonical_claim", {}), l_weights, canonical_l_registration, int(canonical_layout["N_in"]), transcript=transcript, linear_form=l_form, public_context=context).get("status") != "VERIFY_OK":
                return False
            lc_commitment = ValueCommitment(_point(link_payload["lc_claim"]["evaluation_commitment"])); l_commitment = ValueCommitment(_point(link_payload["canonical_claim"]["evaluation_commitment"]))
            if not verify_cross_params_same_value(hyrax_value_params(int(layout["N_in"])), lc_commitment, hyrax_value_params(int(canonical_layout["N_in"])), l_commitment, CrossParamsSameValueProof.from_dict(link_payload.get("same_value")), label=f"{label}/query/link", transcript=transcript):
                return False
            local_sources: list[dict[str, Any]] = [{"label": f"{label}/query/lc", "segment": "LC", "linear_form": lc_form, "evaluation_commitment": lc_commitment, "weights": lc_weights}]
            canonical_source_sink.append({"label": f"{label}/query/canonical", "segment": "com_L", "linear_form": l_form, "evaluation_commitment": l_commitment, "weights": l_weights})
            registrations = {"LC": lc_registration, "GP": gp_registration}
            gate_wires, specs = wire.get("gates"), _gate_specs(layout, table, beta)
            if not isinstance(gate_wires, list) or len(gate_wires) != len(specs):
                return False
            params = hyrax_value_params(int(layout["N_in"]))
            for gate_wire, (kind, spec) in zip(gate_wires, specs):
                verifier = verify_linear_gate if kind == "linear" else verify_product_gate
                if not isinstance(gate_wire, Mapping) or set(gate_wire) != {"kind", "proof"} or gate_wire.get("kind") != kind or not verifier(gate_wire.get("proof", {}), spec, layout=layout, registrations=registrations, params=params, transcript=transcript, context=context, source_sink=local_sources):
                    return False
            if not _local_combine_verify(label, wire.get("local_combine", {}), local_sources, registrations, int(layout["N_in"]), transcript, context):
                return False
        return proof.get("transcript_digest") == transcript.digest.hex()
    except (KeyError, TypeError, ValueError, OverflowError, StopIteration):
        return False
