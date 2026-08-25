"""Shared public integer-bound ledger derivation for producer and verifier."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .field import BN254_FR_MODULUS
from .rounding import division_bound_record, validate_rounding_mode


FIELD_HALF = (BN254_FR_MODULUS - 1) // 2


def _fixed_lrelu(value: int) -> int:
    value = int(value)
    return value if value >= 0 else -((-value + 2) // 5)


def _fixed_exp_zero() -> int:
    # ExpMap(0) is the maximum certified output for the nonnegative domain.
    return 16


def _int_values(value: Any):
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _int_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _int_values(item)


def _family_range_policy(bound: int, range_bits: int) -> dict[str, int | str]:
    """Return the canonical range-family geometry for one public bound.

    The family width is deliberately derived from the certified ledger bound,
    never from the largest witness value observed by a prover.  ``bit_width``
    is the number of bits needed for the inclusive magnitude bound and
    ``limb_count`` is consequently stable for producer and verifier.
    """

    bound = max(0, int(bound))
    range_bits = int(range_bits)
    if range_bits <= 0:
        raise ValueError("range_bits must be positive")
    bit_width = max(1, (bound + 1).bit_length())
    limb_count = max(1, (bit_width + range_bits - 1) // range_bits)
    return {
        "bound": bound,
        "bit_width": bit_width,
        "range_bits": range_bits,
        "limb_count": limb_count,
        "limb_count_rule": "ceil(bit_width/range_bits)",
        "bound_source": "authenticated_integer_bound_ledger",
    }
def _head_record(
    *,
    family: str,
    kind: str,
    layer: int | str,
    head: int,
    input_bound: int,
    input_dim: int,
    width: int,
    classes: int,
    degree: int,
    bound: Mapping[str, Any],
    mode: str,
    node_count: int,
    output: bool,
    output_input_scale: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    b_w = int(bound.get("B_W", -1))
    b_src = int(bound.get("B_a_src", -1))
    b_dst = int(bound.get("B_a_dst", -1))
    if min(b_w, b_src, b_dst) < 0:
        raise ValueError(f"missing nonnegative registered model bound: {family}")
    if output:
        b_bias = int(bound.get("B_b", -1))
        if b_bias < 0:
            raise ValueError(f"missing output bias bound: {family}")
    else:
        b_bias = 0
    h_prime = max(1, int(input_dim) * int(input_bound) * max(1, b_w))
    score = max(1, 2 * int(width if not output else classes) * h_prime * max(b_src, b_dst))
    z_bound = max(abs(_fixed_lrelu(score)), abs(_fixed_lrelu(-score)))
    delta_bound = 2 * z_bound
    exp_max = _fixed_exp_zero()
    sum_bound = max(1, int(degree)) * exp_max
    numerator_bound = sum_bound * h_prime
    h_prime_scale = int(output_input_scale) * 16
    dividend_bound = 16 * numerator_bound
    denominator_bound = h_prime_scale * sum_bound
    agg_id = f"{family}.SignedRoundDiv.aggregate"
    agg = division_bound_record(
        agg_id,
        mode,
        a_bound=dividend_bound,
        d_bound=denominator_bound,
        d_min=h_prime_scale,
        count=int(node_count) * int(width if not output else classes),
        source=family,
        numerator_source="H_or_Y_agg_num*aggregate_output_scale",
        denominator_source="Sum*h_prime_scale",
    )
    agg["B_x_independent"] = numerator_bound
    divisions = [agg]
    result: dict[str, Any] = {
        "kind": kind,
        "layer": layer,
        "head": int(head),
        "division_id": agg_id,
        "B_H": int(input_bound),
        "B_W": b_w,
        "B_H_prime" if not output else "B_Y_prime": h_prime,
        "B_a_src": b_src,
        "B_a_dst": b_dst,
        "B_b": b_bias,
        "B_S": score,
        "B_Z": z_bound,
        "B_delta_raw": delta_bound,
        "B_delta": delta_bound,
        "B_U": exp_max,
        "B_Sum": sum_bound,
        "B_H_num" if not output else "B_Y_num": numerator_bound,
        "S_H_prime" if not output else "S_Y_prime": h_prime_scale,
        "S_agg_pre": 16,
        "S_out_pre": 16 if output else None,
        "B_x_div": agg["B_a"],
        "B_d_div": agg["B_d"],
        "B_a_div": agg["B_a"],
        "B_q_div": agg["B_q"],
        "B_r_div": agg["B_r"],
        "B_delta_div": agg["B_delta"],
        "B_d": agg["B_d"],
        "B_2a_plus_d": agg["B_2a_plus_d"],
        "B_2dq_plus_r": agg["B_2dq_plus_r"],
        "B_2d_minus_1": agg["B_2d_minus_1"],
        "B_r_plus_delta": agg["B_r_plus_delta"],
        "rounding_mode": mode,
    }
    if output:
        align_id = f"{family}.SignedRoundDiv.scale_alignment"
        align_a = int(agg["B_q"]) * 16
        align_d = 16
        align = division_bound_record(
            align_id,
            mode,
            a_bound=align_a,
            d_bound=align_d,
            d_min=align_d,
            count=int(node_count) * int(classes),
            source=family,
            numerator_source="SignedRoundDiv.aggregate.output*output_head_scale",
            denominator_source="aggregate_output_scale",
        )
        align["B_x_independent"] = int(agg["B_q"])
        divisions.append(align)
        result.update(
            {
                "scale_alignment_division_id": align_id,
                "B_x_align": align["B_a"],
                "B_q_align": align["B_q"],
                "B_Y_head": int(align["B_q"]) + b_bias,
            }
        )
    return result, divisions, int(result.get("B_Y_head", result["B_q_div"]))


def build_integer_bound_ledger(
    *,
    mode: str,
    quant_cfg_id: str,
    authenticated_quant_scales: Mapping[str, int],
    registered_model_bounds: Mapping[str, Mapping[str, Any]],
    hidden_profile: Sequence[Mapping[str, Any]],
    d_in: Sequence[int],
    classes: int,
    k_out: int,
    max_degree: int,
    feature_bound: int,
    node_count: int,
    range_bits: int = 12,
) -> dict[str, Any]:
    mode = validate_rounding_mode(mode)
    hidden = list(hidden_profile)
    if not hidden or len(d_in) != len(hidden) + 1:
        raise ValueError("invalid hidden profile for integer-bound ledger")
    degree = max(1, int(max_degree))
    current_input_bound = max(1, int(feature_bound))
    input_scale = 1
    heads: list[dict[str, Any]] = []
    divisions: list[dict[str, Any]] = []
    for position, layer in enumerate(hidden, start=1):
        outputs: list[int] = []
        width = int(layer["head_dim"])
        for head_index in range(int(layer["head_count"])):
            family = f"hidden.l{position}.h{head_index}"
            record, divs, output_bound = _head_record(
                family=family, kind="hidden", layer=position, head=head_index,
                input_bound=current_input_bound, input_dim=int(d_in[position - 1]),
                width=width, classes=int(classes), degree=degree,
                bound=registered_model_bounds[family], mode=mode,
                node_count=int(node_count), output=False,
                output_input_scale=input_scale,
            )
            heads.append(record); divisions.extend(divs); outputs.append(output_bound)
        current_input_bound = max(outputs, default=1)
        input_scale = 16

    output_heads: list[dict[str, Any]] = []
    output_bounds: list[int] = []
    for head_index in range(int(k_out)):
        family = f"output.h{head_index}"
        record, divs, output_bound = _head_record(
            family=family, kind="output", layer="out", head=head_index,
            input_bound=current_input_bound, input_dim=int(d_in[-1]),
            width=int(classes), classes=int(classes), degree=degree,
            bound=registered_model_bounds[family], mode=mode,
            node_count=int(node_count), output=True,
            output_input_scale=16,
        )
        output_heads.append(record); divisions.extend(divs); output_bounds.append(output_bound)

    output_sum_bound = max(1, int(k_out)) * max(output_bounds, default=1)
    final_id = "output.final_average.SignedRoundDiv"
    final = division_bound_record(
        final_id,
        mode,
        a_bound=output_sum_bound,
        d_bound=max(1, int(k_out)),
        d_min=max(1, int(k_out)),
        count=int(node_count) * int(classes),
        source="output.final_average",
        numerator_source="output.Y_sum",
        denominator_source="K_out",
    )
    final["B_x_independent"] = max(output_bounds, default=1)
    divisions.append(final)

    all_head_bounds = heads + output_heads
    max_bound = max((abs(int(value)) for value in _int_values({"heads": all_head_bounds, "divisions": divisions})), default=1)
    if max_bound >= FIELD_HALF:
        raise ValueError("integer_bound_ledger exceeds BN254 signed field half")
    table_domains = {
        "lrelu": [-max((int(item["B_S"]) for item in all_head_bounds), default=1), max((int(item["B_S"]) for item in all_head_bounds), default=1)],
        "elu": [-max((int(item["B_q_div"]) for item in heads), default=1), max((int(item["B_q_div"]) for item in heads), default=1)],
        "exp": [0, max((int(item["B_delta_raw"]) for item in all_head_bounds), default=1)],
    }
    division_ids = [str(item["id"]) for item in divisions]
    division_by_id = {str(item["id"]): item for item in divisions}
    range_families: dict[str, dict[str, int | str]] = {}
    for item in all_head_bounds:
        tag = f"{item['kind']}.l{item['layer']}.h{item['head']}" if item["kind"] == "hidden" else f"output.h{item['head']}"
        range_families[f"{tag}.range_delta"] = _family_range_policy(item["B_delta"], range_bits)
        aggregate_id = str(item["division_id"])
        aggregate = division_by_id[aggregate_id]
        range_families[f"{aggregate_id}.range"] = _family_range_policy(
            max(int(aggregate[key]) for key in ("B_a", "B_q", "B_r", "B_delta", "B_d")),
            range_bits,
        )
        if item.get("scale_alignment_division_id"):
            align_id = str(item["scale_alignment_division_id"])
            align = division_by_id[align_id]
            range_families[f"{align_id}.range"] = _family_range_policy(
                max(int(align[key]) for key in ("B_a", "B_q", "B_r", "B_delta", "B_d")),
                range_bits,
            )
    final_policy = _family_range_policy(
        max(int(division_by_id[final_id][key]) for key in ("B_a", "B_q", "B_r", "B_delta", "B_d")),
        range_bits,
    )
    range_families[f"{final_id}.range"] = final_policy
    # Matrix residuals have one canonical signed family.  The shared public
    # ledger does not inspect witness values, so use the certified maximum
    # integer bound as its conservative local width.
    range_families["rescale_rounding_range"] = _family_range_policy(max_bound, range_bits)
    return {
        "schema": "zkGPT_gat_integer_bound_ledger_v2",
        "field": "BN254-Fr",
        "field_half_modulus": str(FIELD_HALF),
        "max_in_degree": degree,
        "feature_bound": int(feature_bound),
        "quant_cfg_id": str(quant_cfg_id),
        "rounding_mode": mode,
        "authenticated_quant_scales": dict(authenticated_quant_scales),
        "registered_model_bounds": {str(k): dict(v) for k, v in registered_model_bounds.items()},
        "bit_width": max(2, int(max_bound).bit_length()),
        "range_bits": int(range_bits),
        "heads": all_head_bounds,
        "signed_round_divisions": divisions,
        "division_instance_ids": division_ids,
        "division_cardinality": {str(item["id"]): int(item["count"]) for item in divisions},
        "table_domains": table_domains,
        "final_average": {
            "K_out": int(k_out), "B_Y_sum": output_sum_bound,
            "denominator": int(k_out), "division_id": final_id,
        },
        "scales": {
            "S_input": 1, "S_W_hidden": 16, "S_W_output": 16,
            "S_a_hidden": 16, "S_a_output": 1, "S_delta": 256,
            "S_exp": 16, "S_agg_pre": 16, "S_agg": 16,
            "S_out_pre": 16, "S_out_head": 16, "S_out": 16,
            "S_inv": "field_exact", "S_alpha": "field_exact",
        },
        "range_family_policy": {
            "bit_width": max(2, int(max_bound).bit_length()),
            "limb_count_rule": "ceil(bit_width/range_bits)",
            "query_padding_rule": "next_power_of_two(max(2,real_count))",
            "families": range_families,
        },
        "failures": [],
    }
