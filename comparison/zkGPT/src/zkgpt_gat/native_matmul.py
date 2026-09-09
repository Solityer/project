"""Exact-integer MatMul/MatVec proof bound to registered input segments."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .field import BN254_FR_MODULUS, next_power_of_two, sumcheck_domain, to_field_int
from .gpu_execution import active_product_coefficients
from .input_oracle import equality_weights, public_array_mle
from .native_hyrax import (
    NativeHyrax,
    absorb_mask_header,
    build_mask_header,
    evaluation_commitment_from_payload,
    masked_evaluation_claim_from_payload,
    verify_batched_linear_claim_payload,
)
from .transcript import Transcript
from .value_commitment import (
    MultiplicationProof,
    PedersenParams,
    ValueCommitment,
    prove_multiplication,
    verify_multiplication,
)
from .zk_sumcheck import prove_committed_sumcheck, verify_committed_sumcheck


PROTOCOL_ID = "zkGPT/exact-integer-matmul-sumcheck-v1"


class _ProductProvider:
    """Degree-two product oracle advanced by committed Sumcheck challenges."""

    def __init__(self, left: Sequence[int], right: Sequence[int], label: str = "product2") -> None:
        if len(left) != len(right) or not left or len(left) & (len(left) - 1):
            raise ValueError("product oracle requires equal nonempty power-of-two vectors")
        self.left = [to_field_int(value) for value in left]
        self.right = [to_field_int(value) for value in right]
        self.label = label

    def __call__(self, _round: int, current_claim: int, _transcript: Transcript) -> list[int]:
        c0 = c1 = c2 = 0
        for a0, a1, b0, b1 in zip(self.left[0::2], self.left[1::2], self.right[0::2], self.right[1::2]):
            da = (a1 - a0) % BN254_FR_MODULUS
            db = (b1 - b0) % BN254_FR_MODULUS
            c0 = (c0 + a0 * b0) % BN254_FR_MODULUS
            c1 = (c1 + da * b0 + db * a0) % BN254_FR_MODULUS
            c2 = (c2 + da * db) % BN254_FR_MODULUS
        cpu_coefficients = [c0, c1, c2]
        gpu_coefficients = active_product_coefficients(
            (self.left, self.right), f"{self.label}/round/{_round}"
        )
        coefficients = gpu_coefficients if gpu_coefficients is not None else cpu_coefficients
        if coefficients != cpu_coefficients:
            raise ValueError("GPU product2 coefficients differ from exact CPU arithmetic")
        if (2 * coefficients[0] + coefficients[1] + coefficients[2]) % BN254_FR_MODULUS != to_field_int(current_claim):
            raise ValueError("product oracle claim mismatch")
        return coefficients

    def advance(self, challenge: int) -> None:
        value = to_field_int(challenge)
        self.left = [(a + value * (b - a)) % BN254_FR_MODULUS for a, b in zip(self.left[0::2], self.left[1::2])]
        self.right = [(a + value * (b - a)) % BN254_FR_MODULUS for a, b in zip(self.right[0::2], self.right[1::2])]


@dataclass(frozen=True)
class TensorSource:
    segment: str
    array_name: str

    def to_dict(self) -> dict[str, str]:
        return {"segment": self.segment, "array_name": self.array_name}


@dataclass(frozen=True)
class MatMulSpec:
    label: str
    shape: tuple[int, int, int]
    left: TensorSource
    right: TensorSource
    output: TensorSource
    residual: TensorSource
    output_scale: int
    residual_semantics: str = "exact_zero"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "shape": list(self.shape),
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "output": self.output.to_dict(),
            "residual": self.residual.to_dict(),
            "output_scale": self.output_scale,
            "residual_semantics": self.residual_semantics,
        }


def _tensor_weights(
    shape: Sequence[int],
    axes: Sequence[Sequence[int]],
    *,
    padded_axes: Sequence[bool] | None = None,
) -> list[int]:
    if len(shape) != len(axes):
        raise ValueError("tensor weight rank mismatch")
    if padded_axes is None:
        padded_axes = (False,) * len(shape)
    if len(shape) != len(padded_axes):
        raise ValueError("tensor padding rank mismatch")
    per_axis: list[list[int]] = []
    for length, point, padded in zip(shape, axes, padded_axes):
        length = int(length)
        if length <= 0:
            raise ValueError("tensor axis length must be positive")
        domain = sumcheck_domain(length) if padded else next_power_of_two(length)
        if len(point) != domain.bit_length() - 1:
            raise ValueError("tensor axis MLE point has wrong dimension")
        per_axis.append(equality_weights(point)[:length])
    out = [1]
    for weights in per_axis:
        out = [left * right % BN254_FR_MODULUS for left in out for right in weights]
    return out


def _segment_linear_weights(
    layout: Mapping[str, Any],
    source: TensorSource,
    shape: Sequence[int],
    axes: Sequence[Sequence[int]],
    *,
    padded_axes: Sequence[bool] | None = None,
) -> tuple[dict[int, int], dict[str, Any]]:
    segments = [item for item in layout.get("segments", []) if item.get("name") == source.segment]
    if len(segments) != 1:
        raise ValueError("MatMul source segment is absent from Layout_in")
    arrays = [item for item in segments[0].get("arrays", []) if item.get("name") == source.array_name]
    if len(arrays) != 1:
        raise ValueError("MatMul source array is absent from Layout_in")
    descriptor = arrays[0]
    expected_shape = [int(value) for value in shape]
    descriptor_shape = [int(value) for value in descriptor.get("shape", [])]
    descriptor_length = 1
    for value in descriptor_shape:
        descriptor_length *= value
    expected_length = 1
    for value in expected_shape:
        expected_length *= value
    if descriptor_length != expected_length:
        raise ValueError("MatMul source element count differs from Layout_in")
    local = _tensor_weights(expected_shape, axes, padded_axes=padded_axes)
    if len(local) != int(descriptor["length"]):
        raise ValueError("MatMul linear form does not cover the entire tensor")
    offset = int(descriptor["offset"])
    weights = {offset + index: value for index, value in enumerate(local) if value}
    form = {
        "kind": "layout-tensor-product-mle",
        "segment": source.segment,
        "array_name": source.array_name,
        "offset": offset,
        "layout_shape": descriptor_shape,
        "evaluation_shape": expected_shape,
        "axis_points": [[str(to_field_int(value)) for value in point] for point in axes],
    }
    return weights, form


def _fold_left(left: np.ndarray, row_weights: Sequence[int], inner_domain: int) -> list[int]:
    n, k = left.shape
    out = [0] * inner_domain
    for row in range(n):
        for inner in range(k):
            out[inner] = (out[inner] + row_weights[row] * to_field_int(int(left[row, inner]))) % BN254_FR_MODULUS
    return out


def _fold_right(right: np.ndarray, col_weights: Sequence[int], inner_domain: int) -> list[int]:
    k, m = right.shape
    out = [0] * inner_domain
    for inner in range(k):
        for col in range(m):
            out[inner] = (out[inner] + col_weights[col] * to_field_int(int(right[inner, col]))) % BN254_FR_MODULUS
    return out


def _public_tensor_eval(
    values: np.ndarray,
    shape: Sequence[int],
    axes: Sequence[Sequence[int]],
    *,
    padded_axes: Sequence[bool] | None = None,
) -> int:
    flat = [to_field_int(int(value)) for value in np.ravel(np.asarray(values, dtype=object))]
    weights = _tensor_weights(shape, axes, padded_axes=padded_axes)
    return sum(a * b for a, b in zip(flat, weights)) % BN254_FR_MODULUS


def _open_private(
    label: str,
    source: TensorSource,
    shape: Sequence[int],
    axes: Sequence[Sequence[int]],
    *,
    layout: Mapping[str, Any],
    vectors: Mapping[str, Sequence[int]],
    registrations: Mapping[str, Mapping[str, Any]],
    hyrax: NativeHyrax,
    work_dir: pathlib.Path,
    transcript: Transcript,
    context: Mapping[str, Any],
    padded_axes: Sequence[bool] | None = None,
):
    weights, form = _segment_linear_weights(layout, source, shape, axes, padded_axes=padded_axes)
    state = hyrax.commit_linear_claim_registered(
        label,
        vectors[source.segment],
        weights,
        registrations[source.segment],
        work_dir,
        transcript=transcript,
        linear_form=form,
        public_context=context,
    )
    return state, form


def prove_matmul(
    spec: MatMulSpec,
    left: np.ndarray,
    right: np.ndarray,
    output: np.ndarray,
    residual: np.ndarray,
    *,
    layout: Mapping[str, Any],
    public_arrays: Mapping[str, np.ndarray],
    vectors: Mapping[str, Sequence[int]],
    registrations: Mapping[str, Mapping[str, Any]],
    params: PedersenParams,
    hyrax: NativeHyrax,
    work_dir: pathlib.Path,
    transcript: Transcript,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    mask_header = build_mask_header(
        spec.label,
        context,
        ("output-evaluation", "residual-evaluation", "inner-product-sumcheck", "left-evaluation", "right-evaluation", "terminal-relation"),
    )
    absorb_mask_header(transcript, mask_header)
    n, k, m = spec.shape
    if left.shape != (n, k) or right.shape != (k, m) or output.shape != (n, m) or residual.shape != (n, m):
        raise ValueError(f"MatMul witness shape mismatch: {spec.label}")
    if spec.output_scale != 1 or spec.residual_semantics != "exact_zero":
        raise ValueError(f"unsupported native MatMul relation: {spec.label}")
    if any(int(value) != 0 for value in np.ravel(np.asarray(residual, dtype=object))):
        # Forward's projection and attention paths are exact integer
        # products.  SignedRoundDiv is only used by aggregation/output
        # alignment, so a nonzero matrix residual is not a rounding witness.
        raise ValueError(f"native MatMul residual must be exactly zero: {spec.label}")
    row_point = [transcript.challenge_scalar(f"{spec.label}/row/{i}") for i in range(next_power_of_two(n).bit_length() - 1)]
    col_point = [transcript.challenge_scalar(f"{spec.label}/col/{i}") for i in range(next_power_of_two(m).bit_length() - 1)]
    row_weights = equality_weights(row_point)[:n]
    col_weights = equality_weights(col_point)[:m]
    output_state, output_form = _open_private(
        f"{spec.label}/output", spec.output, (n, m), (row_point, col_point),
        layout=layout, vectors=vectors, registrations=registrations, hyrax=hyrax,
        work_dir=work_dir, transcript=transcript, context=context,
    )
    residual_state, residual_form = _open_private(
        f"{spec.label}/residual", spec.residual, (n, m), (row_point, col_point),
        layout=layout, vectors=vectors, registrations=registrations, hyrax=hyrax,
        work_dir=work_dir, transcript=transcript, context=context,
    )
    initial_value = (spec.output_scale * output_state.evaluation + residual_state.evaluation) % BN254_FR_MODULUS
    initial_blind = (spec.output_scale * output_state.blind + residual_state.blind) % BN254_FR_MODULUS
    initial_commitment = ValueCommitment(
        spec.output_scale * output_state.evaluation_commitment.point + residual_state.evaluation_commitment.point
    )
    inner_domain = sumcheck_domain(k)
    folded_left = _fold_left(left, row_weights, inner_domain)
    folded_right = _fold_right(right, col_weights, inner_domain)
    if sum(a * b for a, b in zip(folded_left, folded_right)) % BN254_FR_MODULUS != initial_value:
        raise ValueError(f"exact MatMul/rescale relation failed: {spec.label}")
    oracle = _ProductProvider(folded_left, folded_right, f"{spec.label}/matmul")
    sumcheck = prove_committed_sumcheck(
        f"{spec.label}/inner-product",
        initial_commitment,
        oracle,
        2,
        variable_count=inner_domain.bit_length() - 1,
        params=params,
        transcript=transcript,
        initial_value=initial_value,
        initial_blind=initial_blind,
    )
    # The provider folds adjacent entries (least-significant coordinate)
    # first, while tensor MLE points are stored most-significant first.
    inner_point = tuple(reversed(sumcheck.challenges))
    if spec.left.segment == "X_pub":
        public_left = public_arrays.get(spec.left.array_name)
        if public_left is None:
            raise ValueError("missing external public MatMul input")
        left_value = _public_tensor_eval(
            public_left, (n, k), (row_point, inner_point), padded_axes=(False, True)
        )
        left_commitment = ValueCommitment(left_value * params.U)
        left_blind = 0
        left_opening: dict[str, Any] = {"kind": "public", "source": spec.left.to_dict()}
    else:
        state, form = _open_private(
            f"{spec.label}/left", spec.left, (n, k), (row_point, inner_point),
            layout=layout, vectors=vectors, registrations=registrations, hyrax=hyrax,
            work_dir=work_dir, transcript=transcript, context=context,
            padded_axes=(False, True),
        )
        left_value, left_blind, left_commitment = state.evaluation, state.blind, state.evaluation_commitment
        left_opening = {"kind": "private", "linear_form": form, "proof": state.proof}
    right_state, right_form = _open_private(
        f"{spec.label}/right", spec.right, (k, m), (inner_point, col_point),
        layout=layout, vectors=vectors, registrations=registrations, hyrax=hyrax,
        work_dir=work_dir, transcript=transcript, context=context,
        padded_axes=(True, False),
    )
    multiplication = prove_multiplication(
        params,
        left_commitment,
        right_state.evaluation_commitment,
        sumcheck.terminal_commitment,
        left_value,
        right_state.evaluation,
        sumcheck.terminal_value,
        left_blind,
        right_state.blind,
        sumcheck.blind,
        label=f"{spec.label}/terminal-multiplication",
        transcript=transcript,
    )
    return {
        "protocol": PROTOCOL_ID,
        "spec": spec.to_dict(),
        "residual_binding": {
            "protocol": "zkGPT/native-matmul-exact-zero-residual-v1",
            "semantics": spec.residual_semantics,
            "source": spec.residual.to_dict(),
            "shape": [int(n), int(m)],
            "value": "0",
        },
        "row_point": [str(value) for value in row_point],
        "col_point": [str(value) for value in col_point],
        "output_opening": {"linear_form": output_form, "proof": output_state.proof},
        "residual_opening": {"linear_form": residual_form, "proof": residual_state.proof},
        "sumcheck": sumcheck.proof.to_dict(),
        "left_opening": left_opening,
        "right_opening": {"linear_form": right_form, "proof": right_state.proof},
        "terminal_multiplication": multiplication.to_dict(),
        "mask_header": mask_header,
    }


def _verify_private(
    label: str,
    source: TensorSource,
    shape: Sequence[int],
    axes: Sequence[Sequence[int]],
    payload: Mapping[str, Any],
    *,
    layout: Mapping[str, Any],
    registrations: Mapping[str, Mapping[str, Any]],
    transcript: Transcript,
    context: Mapping[str, Any],
    padded_axes: Sequence[bool] | None = None,
    source_sink: list[dict[str, Any]] | None = None,
) -> ValueCommitment | None:
    weights, form = _segment_linear_weights(layout, source, shape, axes, padded_axes=padded_axes)
    if payload.get("linear_form") != form:
        return None
    result = verify_batched_linear_claim_payload(
        label,
        payload.get("proof", {}),
        weights,
        registrations[source.segment],
        int(layout["N_in"]),
        transcript=transcript,
        linear_form=form,
        public_context=context,
    )
    if result.get("status") != "VERIFY_OK":
        return None
    commitment = evaluation_commitment_from_payload(payload["proof"])
    if source_sink is not None:
        source_sink.append({
            "label": label,
            "segment": {"W": "com_W", "Aux": "com_Aux", "L_qry": "com_L"}[source.segment],
            "linear_form": form,
            "evaluation_commitment": commitment,
            "evaluation_claim": masked_evaluation_claim_from_payload(payload["proof"]),
            "weights": weights,
        })
    return commitment


def verify_matmul(
    proof: Mapping[str, Any],
    expected_spec: MatMulSpec,
    *,
    layout: Mapping[str, Any],
    public_arrays: Mapping[str, np.ndarray],
    registrations: Mapping[str, Mapping[str, Any]],
    params: PedersenParams,
    transcript: Transcript,
    context: Mapping[str, Any],
    source_sink: list[dict[str, Any]] | None = None,
) -> bool:
    try:
        required = {
            "protocol", "spec", "row_point", "col_point", "output_opening", "residual_opening",
            "sumcheck", "left_opening", "right_opening", "terminal_multiplication", "residual_binding",
            "mask_header",
        }
        if not isinstance(proof, Mapping) or set(proof) != required:
            return False
        if proof["protocol"] != PROTOCOL_ID or proof["spec"] != expected_spec.to_dict():
            return False
        expected_header = build_mask_header(
            expected_spec.label,
            context,
            ("output-evaluation", "residual-evaluation", "inner-product-sumcheck", "left-evaluation", "right-evaluation", "terminal-relation"),
        )
        if proof.get("mask_header") != expected_header:
            return False
        absorb_mask_header(transcript, expected_header)
        n, k, m = expected_spec.shape
        expected_residual_binding = {
            "protocol": "zkGPT/native-matmul-exact-zero-residual-v1",
            "semantics": "exact_zero",
            "source": expected_spec.residual.to_dict(),
            "shape": [int(n), int(m)],
            "value": "0",
        }
        if proof.get("residual_binding") != expected_residual_binding:
            return False
        row_point = [transcript.challenge_scalar(f"{expected_spec.label}/row/{i}") for i in range(next_power_of_two(n).bit_length() - 1)]
        col_point = [transcript.challenge_scalar(f"{expected_spec.label}/col/{i}") for i in range(next_power_of_two(m).bit_length() - 1)]
        if proof["row_point"] != [str(value) for value in row_point] or proof["col_point"] != [str(value) for value in col_point]:
            return False
        output_commitment = _verify_private(
            f"{expected_spec.label}/output", expected_spec.output, (n, m), (row_point, col_point), proof["output_opening"],
            layout=layout, registrations=registrations, transcript=transcript, context=context,
            source_sink=source_sink,
        )
        residual_commitment = _verify_private(
            f"{expected_spec.label}/residual", expected_spec.residual, (n, m), (row_point, col_point), proof["residual_opening"],
            layout=layout, registrations=registrations, transcript=transcript, context=context,
            source_sink=source_sink,
        )
        if output_commitment is None or residual_commitment is None:
            return False
        initial = ValueCommitment(expected_spec.output_scale * output_commitment.point + residual_commitment.point)
        inner_domain = sumcheck_domain(k)
        sumcheck = verify_committed_sumcheck(
            f"{expected_spec.label}/inner-product",
            proof["sumcheck"],
            initial,
            inner_domain.bit_length() - 1,
            2,
            params=params,
            transcript=transcript,
        )
        inner_point = tuple(reversed(sumcheck.challenges))
        if expected_spec.left.segment == "X_pub":
            if proof["left_opening"] != {"kind": "public", "source": expected_spec.left.to_dict()}:
                return False
            public_left = public_arrays.get(expected_spec.left.array_name)
            if public_left is None:
                return False
            left_value = _public_tensor_eval(
                public_left, (n, k), (row_point, inner_point), padded_axes=(False, True)
            )
            left_commitment = ValueCommitment(left_value * params.U)
        else:
            if proof["left_opening"].get("kind") != "private":
                return False
            left_commitment = _verify_private(
                f"{expected_spec.label}/left", expected_spec.left, (n, k), (row_point, inner_point), proof["left_opening"],
                layout=layout, registrations=registrations, transcript=transcript, context=context,
                padded_axes=(False, True),
                source_sink=source_sink,
            )
            if left_commitment is None:
                return False
        right_commitment = _verify_private(
            f"{expected_spec.label}/right", expected_spec.right, (k, m), (inner_point, col_point), proof["right_opening"],
            layout=layout, registrations=registrations, transcript=transcript, context=context,
            padded_axes=(True, False),
            source_sink=source_sink,
        )
        if right_commitment is None:
            return False
        return verify_multiplication(
            params,
            left_commitment,
            right_commitment,
            sumcheck.terminal_commitment,
            MultiplicationProof.from_dict(proof["terminal_multiplication"]),
            label=f"{expected_spec.label}/terminal-multiplication",
            transcript=transcript,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
