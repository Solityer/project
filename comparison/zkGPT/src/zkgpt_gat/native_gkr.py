"""Operand-bound linear gate layer for the native zkGPT GKR adapter."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .field import BN254_FR_MODULUS, next_power_of_two, to_field_int
from .gpu_execution import active_product_coefficients
from .input_oracle import equality_weights
from .native_hyrax import NativeHyrax, _point, verify_batched_linear_claim_payload
from .transcript import Transcript
from .util import digest_json
from .value_commitment import (
    MultiplicationProof,
    PedersenParams,
    RepresentationProof,
    ValueCommitment,
    commit_value,
    prove_multiplication,
    prove_representation,
    verify_multiplication,
    verify_representation,
)
from .zk_sumcheck import prove_committed_sumcheck, verify_committed_sumcheck


PROTOCOL_ID = "zkGPT/native-gkr-linear-gate-v1"
PRODUCT_PROTOCOL_ID = "zkGPT/native-gkr-product-gate-v1"


@dataclass(frozen=True)
class OracleRef:
    segment: str
    array_name: str
    index_map: tuple[int, ...]

    def descriptor(self) -> dict[str, Any]:
        return {
            "segment": self.segment,
            "array_name": self.array_name,
            "output_count": len(self.index_map),
            "index_map_digest": digest_json(list(self.index_map)),
        }


@dataclass(frozen=True)
class LinearTerm:
    coefficient: int
    oracle: OracleRef

    def descriptor(self) -> dict[str, Any]:
        return {"coefficient": str(to_field_int(self.coefficient)), "oracle": self.oracle.descriptor()}


@dataclass(frozen=True)
class LinearGateSpec:
    label: str
    target: OracleRef
    terms: tuple[LinearTerm, ...]
    public_constant: tuple[int, ...]

    def __post_init__(self) -> None:
        count = len(self.target.index_map)
        if count <= 0 or len(self.public_constant) != count:
            raise ValueError("linear GKR family has an invalid public shape")
        if any(len(term.oracle.index_map) != count for term in self.terms):
            raise ValueError("linear GKR oracle maps have different output domains")

    def descriptor(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "output_count": len(self.target.index_map),
            "target": self.target.descriptor(),
            "terms": [term.descriptor() for term in self.terms],
            "public_constant_digest": digest_json([str(to_field_int(value)) for value in self.public_constant]),
        }


@dataclass(frozen=True)
class ProductGateSpec:
    """A point-wise gate ``target = left * right`` on a fixed output domain."""

    label: str
    target: OracleRef
    left: OracleRef
    right: OracleRef

    def __post_init__(self) -> None:
        count = len(self.target.index_map)
        if count <= 0 or len(self.left.index_map) != count or len(self.right.index_map) != count:
            raise ValueError("product GKR oracle maps have different output domains")

    def descriptor(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "output_count": len(self.target.index_map),
            "target": self.target.descriptor(),
            "left": self.left.descriptor(),
            "right": self.right.descriptor(),
        }


class _TripleProductProvider:
    """Degree-three oracle ``eq(point, i) * left(i) * right(i)``."""

    def __init__(self, selector: Sequence[int], left: Sequence[int], right: Sequence[int], label: str = "product3") -> None:
        if len(selector) != len(left) or len(left) != len(right) or not left or len(left) & (len(left) - 1):
            raise ValueError("triple product oracle requires equal power-of-two vectors")
        self.selector = [to_field_int(value) for value in selector]
        self.left = [to_field_int(value) for value in left]
        self.right = [to_field_int(value) for value in right]
        self.label = label

    def __call__(self, _round: int, current_claim: int, _transcript: Transcript) -> list[int]:
        coefficients = [0, 0, 0, 0]
        for s0, s1, a0, a1, b0, b1 in zip(
            self.selector[0::2], self.selector[1::2],
            self.left[0::2], self.left[1::2], self.right[0::2], self.right[1::2],
        ):
            ds = (s1 - s0) % BN254_FR_MODULUS
            da = (a1 - a0) % BN254_FR_MODULUS
            db = (b1 - b0) % BN254_FR_MODULUS
            coefficients[0] = (coefficients[0] + s0 * a0 * b0) % BN254_FR_MODULUS
            coefficients[1] = (coefficients[1] + ds * a0 * b0 + s0 * da * b0 + s0 * a0 * db) % BN254_FR_MODULUS
            coefficients[2] = (coefficients[2] + ds * da * b0 + ds * a0 * db + s0 * da * db) % BN254_FR_MODULUS
            coefficients[3] = (coefficients[3] + ds * da * db) % BN254_FR_MODULUS
        # g(0)+g(1) = c0 + (c0+c1+c2+c3).
        gpu_coefficients = active_product_coefficients(
            (self.selector, self.left, self.right), f"{self.label}/round/{_round}"
        )
        if gpu_coefficients is not None:
            if gpu_coefficients != coefficients:
                raise ValueError("GPU product3 coefficients differ from exact CPU arithmetic")
            coefficients = gpu_coefficients
        if (2 * coefficients[0] + sum(coefficients[1:])) % BN254_FR_MODULUS != to_field_int(current_claim):
            raise ValueError("triple product oracle claim mismatch")
        return coefficients

    def advance(self, challenge: int) -> None:
        value = to_field_int(challenge)
        fold = lambda values: [
            (left + value * (right - left)) % BN254_FR_MODULUS
            for left, right in zip(values[0::2], values[1::2])
        ]
        self.selector = fold(self.selector)
        self.left = fold(self.left)
        self.right = fold(self.right)


def identity_oracle(segment: str, array_name: str, count: int) -> OracleRef:
    return OracleRef(segment, array_name, tuple(range(count)))


def _layout_array(layout: Mapping[str, Any], oracle: OracleRef) -> Mapping[str, Any]:
    segments = [item for item in layout.get("segments", []) if item.get("name") == oracle.segment]
    if len(segments) != 1:
        raise ValueError("GKR oracle segment is missing from Layout_in")
    arrays = [item for item in segments[0].get("arrays", []) if item.get("name") == oracle.array_name]
    if len(arrays) != 1:
        raise ValueError("GKR oracle array is missing from Layout_in")
    descriptor = arrays[0]
    length = int(descriptor["length"])
    # ``-1`` is a verifier-defined public-zero routing slot.  It is useful
    # for ragged public adjacency without committing a prover-chosen padding
    # value; every other negative index remains invalid.
    if any(index < -1 or index >= length for index in oracle.index_map):
        raise ValueError("GKR oracle index map leaves its registered tensor")
    return descriptor


def _weights_and_form(
    layout: Mapping[str, Any], oracle: OracleRef, output_weights: Sequence[int]
) -> tuple[dict[int, int], dict[str, Any]]:
    descriptor = _layout_array(layout, oracle)
    if len(output_weights) != len(oracle.index_map):
        raise ValueError("GKR output weights have wrong length")
    local: dict[int, int] = {}
    for weight, index in zip(output_weights, oracle.index_map):
        if index >= 0:
            local[index] = (local.get(index, 0) + to_field_int(weight)) % BN254_FR_MODULUS
    n_in = int(layout["N_in"])
    offset = int(descriptor["offset"])
    weights = {offset + index: value for index, value in local.items() if value}
    form = {
        "kind": "layout-index-map-linear-form",
        **oracle.descriptor(),
        "layout_offset": offset,
        "layout_length": len(local),
        "output_weights_digest": digest_json([str(value) for value in output_weights]),
    }
    return weights, form


def _open(
    label: str,
    oracle: OracleRef,
    output_weights: Sequence[int],
    *,
    layout: Mapping[str, Any],
    vectors: Mapping[str, Sequence[int]],
    registrations: Mapping[str, Mapping[str, Any]],
    hyrax: NativeHyrax,
    work_dir: pathlib.Path,
    transcript: Transcript,
    context: Mapping[str, Any],
):
    weights, form = _weights_and_form(layout, oracle, output_weights)
    state = hyrax.commit_linear_claim_registered(
        label,
        vectors[oracle.segment],
        weights,
        registrations[oracle.segment],
        work_dir,
        transcript=transcript,
        linear_form=form,
        public_context=context,
    )
    return state, form


def prove_linear_gate(
    spec: LinearGateSpec,
    *,
    layout: Mapping[str, Any],
    vectors: Mapping[str, Sequence[int]],
    registrations: Mapping[str, Mapping[str, Any]],
    params: PedersenParams,
    hyrax: NativeHyrax,
    work_dir: pathlib.Path,
    transcript: Transcript,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    count = len(spec.target.index_map)
    point = [transcript.challenge_scalar(f"{spec.label}/point/{index}") for index in range(next_power_of_two(count).bit_length() - 1)]
    output_weights = equality_weights(point)[:count]
    target, target_form = _open(
        f"{spec.label}/target", spec.target, output_weights,
        layout=layout, vectors=vectors, registrations=registrations,
        hyrax=hyrax, work_dir=work_dir, transcript=transcript, context=context,
    )
    term_payloads: list[dict[str, Any]] = []
    term_states = []
    for index, term in enumerate(spec.terms):
        state, form = _open(
            f"{spec.label}/term/{index}", term.oracle, output_weights,
            layout=layout, vectors=vectors, registrations=registrations,
            hyrax=hyrax, work_dir=work_dir, transcript=transcript, context=context,
        )
        term_states.append(state)
        term_payloads.append({"linear_form": form, "proof": state.proof})
    public_eval = sum(
        weight * to_field_int(value)
        for weight, value in zip(output_weights, spec.public_constant)
    ) % BN254_FR_MODULUS
    expected_value = (sum(
        to_field_int(term.coefficient) * state.evaluation
        for term, state in zip(spec.terms, term_states)
    ) + public_eval) % BN254_FR_MODULUS
    if target.evaluation != expected_value:
        raise ValueError(f"linear GKR relation failed: {spec.label}")
    difference_point = target.evaluation_commitment.point - public_eval * params.U
    difference_blind = target.blind
    for term, state in zip(spec.terms, term_states):
        coefficient = to_field_int(term.coefficient)
        difference_point = difference_point - coefficient * state.evaluation_commitment.point
        difference_blind = (difference_blind - coefficient * state.blind) % BN254_FR_MODULUS
    zero_commitment = ValueCommitment(difference_point)
    equality = prove_representation(
        params,
        zero_commitment,
        0,
        difference_blind,
        label=f"{spec.label}/linear-equality",
        transcript=transcript,
    )
    return {
        "protocol": PROTOCOL_ID,
        "spec": spec.descriptor(),
        "point": [str(value) for value in point],
        "target_opening": {"linear_form": target_form, "proof": target.proof},
        "term_openings": term_payloads,
        "linear_equality": equality.to_dict(),
    }


def _verify_opening(
    label: str,
    oracle: OracleRef,
    output_weights: Sequence[int],
    payload: Mapping[str, Any],
    *,
    layout: Mapping[str, Any],
    registrations: Mapping[str, Mapping[str, Any]],
    transcript: Transcript,
    context: Mapping[str, Any],
    source_sink: list[dict[str, Any]] | None = None,
) -> ValueCommitment | None:
    weights, form = _weights_and_form(layout, oracle, output_weights)
    if payload.get("linear_form") != form:
        return None
    result = verify_batched_linear_claim_payload(
        label, payload.get("proof", {}), weights, registrations[oracle.segment], int(layout["N_in"]),
        transcript=transcript, linear_form=form, public_context=context,
    )
    if result.get("status") != "VERIFY_OK":
        return None
    commitment = ValueCommitment(_point(payload["proof"]["evaluation_commitment"]))
    if source_sink is not None:
        source_sink.append({
            "label": label,
            "segment": {"W": "com_W", "Aux": "com_Aux", "L_qry": "com_L"}.get(
                oracle.segment, oracle.segment
            ),
            "linear_form": form,
            "evaluation_commitment": commitment,
            "weights": weights,
        })
    return commitment


def verify_linear_gate(
    proof: Mapping[str, Any],
    spec: LinearGateSpec,
    *,
    layout: Mapping[str, Any],
    registrations: Mapping[str, Mapping[str, Any]],
    params: PedersenParams,
    transcript: Transcript,
    context: Mapping[str, Any],
    source_sink: list[dict[str, Any]] | None = None,
) -> bool:
    try:
        if not isinstance(proof, Mapping) or set(proof) != {
            "protocol", "spec", "point", "target_opening", "term_openings", "linear_equality"
        }:
            return False
        if proof["protocol"] != PROTOCOL_ID or proof["spec"] != spec.descriptor():
            return False
        count = len(spec.target.index_map)
        point = [transcript.challenge_scalar(f"{spec.label}/point/{index}") for index in range(next_power_of_two(count).bit_length() - 1)]
        if proof["point"] != [str(value) for value in point]:
            return False
        output_weights = equality_weights(point)[:count]
        target = _verify_opening(
            f"{spec.label}/target", spec.target, output_weights, proof["target_opening"],
            layout=layout, registrations=registrations, transcript=transcript, context=context,
            source_sink=source_sink,
        )
        term_payloads = proof["term_openings"]
        if target is None or not isinstance(term_payloads, list) or len(term_payloads) != len(spec.terms):
            return False
        terms = [
            _verify_opening(
                f"{spec.label}/term/{index}", term.oracle, output_weights, term_payloads[index],
                layout=layout, registrations=registrations, transcript=transcript, context=context,
                source_sink=source_sink,
            )
            for index, term in enumerate(spec.terms)
        ]
        if any(item is None for item in terms):
            return False
        public_eval = sum(weight * to_field_int(value) for weight, value in zip(output_weights, spec.public_constant)) % BN254_FR_MODULUS
        difference = target.point - public_eval * params.U
        for term, commitment in zip(spec.terms, terms):
            difference = difference - to_field_int(term.coefficient) * commitment.point
        return verify_representation(
            params,
            ValueCommitment(difference),
            RepresentationProof.from_dict(proof["linear_equality"]),
            label=f"{spec.label}/linear-equality",
            transcript=transcript,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _oracle_values(
    layout: Mapping[str, Any], oracle: OracleRef, vectors: Mapping[str, Sequence[int]]
) -> list[int]:
    descriptor = _layout_array(layout, oracle)
    vector = vectors.get(oracle.segment)
    if vector is None or len(vector) != int(layout["N_in"]):
        raise ValueError("missing registered GKR segment witness")
    offset = int(descriptor["offset"])
    return [0 if index < 0 else to_field_int(vector[offset + index]) for index in oracle.index_map]


def prove_product_gate(
    spec: ProductGateSpec,
    *,
    layout: Mapping[str, Any],
    vectors: Mapping[str, Sequence[int]],
    registrations: Mapping[str, Mapping[str, Any]],
    params: PedersenParams,
    hyrax: NativeHyrax,
    work_dir: pathlib.Path,
    transcript: Transcript,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    count = len(spec.target.index_map)
    domain = next_power_of_two(count)
    variables = domain.bit_length() - 1
    output_point = [
        transcript.challenge_scalar(f"{spec.label}/output-point/{index}")
        for index in range(variables)
    ]
    output_weights = equality_weights(output_point)
    target, target_form = _open(
        f"{spec.label}/target", spec.target, output_weights[:count],
        layout=layout, vectors=vectors, registrations=registrations,
        hyrax=hyrax, work_dir=work_dir, transcript=transcript, context=context,
    )
    left_values = _oracle_values(layout, spec.left, vectors) + [0] * (domain - count)
    right_values = _oracle_values(layout, spec.right, vectors) + [0] * (domain - count)
    expected = sum(
        selector * left * right
        for selector, left, right in zip(output_weights, left_values, right_values)
    ) % BN254_FR_MODULUS
    if target.evaluation != expected:
        raise ValueError(f"product GKR relation failed: {spec.label}")
    sumcheck = prove_committed_sumcheck(
        f"{spec.label}/product-sumcheck",
        target.evaluation_commitment,
        _TripleProductProvider(output_weights, left_values, right_values, f"{spec.label}/gkr"),
        3,
        variable_count=variables,
        params=params,
        transcript=transcript,
        initial_value=target.evaluation,
        initial_blind=target.blind,
    )
    # Sumcheck folds adjacent entries first, i.e. the least-significant MLE
    # coordinate.  ``equality_weights`` enumerates that coordinate last.
    terminal_point = tuple(reversed(sumcheck.challenges))
    terminal_weights = equality_weights(terminal_point)[:count]
    left, left_form = _open(
        f"{spec.label}/left", spec.left, terminal_weights,
        layout=layout, vectors=vectors, registrations=registrations,
        hyrax=hyrax, work_dir=work_dir, transcript=transcript, context=context,
    )
    right, right_form = _open(
        f"{spec.label}/right", spec.right, terminal_weights,
        layout=layout, vectors=vectors, registrations=registrations,
        hyrax=hyrax, work_dir=work_dir, transcript=transcript, context=context,
    )
    selector_eval = sum(
        left_weight * right_weight
        for left_weight, right_weight in zip(output_weights, equality_weights(terminal_point))
    ) % BN254_FR_MODULUS
    product_value = left.evaluation * right.evaluation % BN254_FR_MODULUS
    if product_value * selector_eval % BN254_FR_MODULUS != sumcheck.terminal_value:
        raise ValueError(
            "product GKR terminal oracle is inconsistent: "
            f"{product_value}*{selector_eval}!={sumcheck.terminal_value}"
        )
    product_commitment, product_blind = commit_value(params, product_value)
    operand_product = prove_multiplication(
        params, left.evaluation_commitment, right.evaluation_commitment,
        product_commitment, left.evaluation, right.evaluation, product_value,
        left.blind, right.blind, product_blind,
        label=f"{spec.label}/terminal-operands", transcript=transcript,
    )
    selector_commitment = ValueCommitment(selector_eval * params.U)
    terminal_product = prove_multiplication(
        params, product_commitment, selector_commitment,
        sumcheck.terminal_commitment, product_value, selector_eval,
        sumcheck.terminal_value, product_blind, 0, sumcheck.blind,
        label=f"{spec.label}/terminal-selector", transcript=transcript,
    )
    return {
        "protocol": PRODUCT_PROTOCOL_ID,
        "spec": spec.descriptor(),
        "output_point": [str(value) for value in output_point],
        "target_opening": {"linear_form": target_form, "proof": target.proof},
        "sumcheck": sumcheck.proof.to_dict(),
        "left_opening": {"linear_form": left_form, "proof": left.proof},
        "right_opening": {"linear_form": right_form, "proof": right.proof},
        "product_commitment": product_commitment.to_dict(),
        "operand_product": operand_product.to_dict(),
        "terminal_product": terminal_product.to_dict(),
    }


def verify_product_gate(
    proof: Mapping[str, Any],
    spec: ProductGateSpec,
    *,
    layout: Mapping[str, Any],
    registrations: Mapping[str, Mapping[str, Any]],
    params: PedersenParams,
    transcript: Transcript,
    context: Mapping[str, Any],
    source_sink: list[dict[str, Any]] | None = None,
) -> bool:
    try:
        required = {
            "protocol", "spec", "output_point", "target_opening", "sumcheck",
            "left_opening", "right_opening", "product_commitment",
            "operand_product", "terminal_product",
        }
        if not isinstance(proof, Mapping) or set(proof) != required:
            return False
        if proof["protocol"] != PRODUCT_PROTOCOL_ID or proof["spec"] != spec.descriptor():
            return False
        count = len(spec.target.index_map)
        domain = next_power_of_two(count)
        variables = domain.bit_length() - 1
        output_point = [
            transcript.challenge_scalar(f"{spec.label}/output-point/{index}")
            for index in range(variables)
        ]
        if proof["output_point"] != [str(value) for value in output_point]:
            return False
        output_weights = equality_weights(output_point)
        target = _verify_opening(
            f"{spec.label}/target", spec.target, output_weights[:count], proof["target_opening"],
            layout=layout, registrations=registrations, transcript=transcript, context=context,
            source_sink=source_sink,
        )
        if target is None:
            return False
        sumcheck = verify_committed_sumcheck(
            f"{spec.label}/product-sumcheck", proof["sumcheck"], target,
            variables, 3, params=params, transcript=transcript,
        )
        terminal_point = tuple(reversed(sumcheck.challenges))
        terminal_weights = equality_weights(terminal_point)[:count]
        left = _verify_opening(
            f"{spec.label}/left", spec.left, terminal_weights, proof["left_opening"],
            layout=layout, registrations=registrations, transcript=transcript, context=context,
            source_sink=source_sink,
        )
        right = _verify_opening(
            f"{spec.label}/right", spec.right, terminal_weights, proof["right_opening"],
            layout=layout, registrations=registrations, transcript=transcript, context=context,
            source_sink=source_sink,
        )
        if left is None or right is None:
            return False
        selector_eval = sum(
            left_weight * right_weight
            for left_weight, right_weight in zip(output_weights, equality_weights(terminal_point))
        ) % BN254_FR_MODULUS
        product = ValueCommitment.from_dict(proof["product_commitment"])
        if not verify_multiplication(
            params, left, right, product,
            MultiplicationProof.from_dict(proof["operand_product"]),
            label=f"{spec.label}/terminal-operands", transcript=transcript,
        ):
            return False
        return verify_multiplication(
            params, product, ValueCommitment(selector_eval * params.U),
            sumcheck.terminal_commitment,
            MultiplicationProof.from_dict(proof["terminal_product"]),
            label=f"{spec.label}/terminal-selector", transcript=transcript,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
