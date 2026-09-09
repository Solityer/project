"""Shared public integer-bound ledger derivation for producer and verifier."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .field import BN254_FR_MODULUS
from .quantization import (
    canonical_scale_profile,
    resolve_quant_scale_profile,
    scale_chain_from_profile,
)
from .rounding import division_bound_record, validate_rounding_mode


FIELD_HALF = (BN254_FR_MODULUS - 1) // 2


def _fixed_lrelu(value: int) -> int:
    value = int(value)
    return value if value >= 0 else -((-value + 2) // 5)


def _fixed_exp_zero(output_scale: int | None = None) -> int:
    # ExpMap(0) is the maximum certified output for the nonnegative domain.
    if output_scale is None:
        output_scale = canonical_scale_profile()["exp_output"]
    output_scale = int(output_scale)
    if output_scale <= 0:
        raise ValueError("ExpMap output scale must be positive")
    return output_scale


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


def _family_range_policy(
    bound: int,
    range_bits: int,
    *,
    encoding_offset: int = 0,
) -> dict[str, int | str]:
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
    encoding_offset = int(encoding_offset)
    if encoding_offset < 0:
        raise ValueError("range-family encoding offset must be nonnegative")
    encoded_bound = bound + encoding_offset
    bit_width = max(1, (encoded_bound + 1).bit_length())
    limb_count = max(1, (bit_width + range_bits - 1) // range_bits)
    result: dict[str, int | str] = {
        "bound": bound,
        "encoded_bound": encoded_bound,
        "bit_width": bit_width,
        "range_bits": range_bits,
        "limb_count": limb_count,
        "limb_count_rule": "ceil(bit_width/range_bits)",
        "bound_source": "authenticated_integer_bound_ledger",
    }
    if encoding_offset:
        result.update({"encoding": "offset", "encoding_offset": encoding_offset})
    else:
        result.update({"encoding": "magnitude", "encoding_offset": 0})
    return result


def _function_output_bound(
    table: str,
    domain: Sequence[int],
    *,
    scale_profile: Mapping[str, int] | None = None,
) -> int:
    """Return a certified bound over the complete integer table domain."""

    if len(domain) != 2:
        raise ValueError(f"invalid {table} certified domain")
    lo, hi = int(domain[0]), int(domain[1])
    if hi < lo:
        raise ValueError(f"invalid {table} certified domain ordering")
    profile = canonical_scale_profile() if scale_profile is None else scale_profile
    if table == "lrelu":
        values = (_fixed_lrelu(lo), _fixed_lrelu(hi), 0)
    elif table == "exp":
        values = (_fixed_exp_zero(int(profile["exp_output"])),)
    elif table == "elu":
        output_scale = int(profile["elu_output"])
        values = (hi, -output_scale, output_scale)
    else:
        raise ValueError(f"unknown certified function family: {table}")
    return max((abs(int(value)) for value in values), default=0)
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
    aggregate_output_scale: int,
    output_head_scale: int,
    exp_output_scale: int,
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
    exp_max = _fixed_exp_zero(exp_output_scale)
    sum_bound = max(1, int(degree)) * exp_max
    numerator_bound = sum_bound * h_prime
    h_prime_scale = int(output_input_scale)
    aggregate_output_scale = int(aggregate_output_scale)
    dividend_bound = aggregate_output_scale * numerator_bound
    denominator_bound = h_prime_scale * sum_bound
    # Every authenticated destination group contains at least one edge, and
    # the canonical max-subtraction makes at least one of its deltas exactly
    # zero.  ExpMap(0) is ``exp_output_scale``; consequently the actual
    # denominator is bounded below by one such term, not merely by the
    # h_prime scale.  This tightens only the public no-wrap quotient bound;
    # the SignedRoundDiv relation and all witness values are unchanged.
    denominator_min = h_prime_scale * int(exp_output_scale)
    agg_id = f"{family}.SignedRoundDiv.aggregate"
    agg = division_bound_record(
        agg_id,
        mode,
        a_bound=dividend_bound,
        d_bound=denominator_bound,
        d_min=denominator_min,
        count=int(node_count) * int(width if not output else classes),
        source=family,
        numerator_source="H_or_Y_agg_num*authenticated_aggregate_output_scale",
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
        "B_H_agg" if not output else "B_Y_agg": int(agg["B_q"]),
        "S_H_prime" if not output else "S_Y_prime": h_prime_scale,
        "S_agg_pre": aggregate_output_scale,
        "S_out_pre": aggregate_output_scale if output else None,
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
        output_head_scale = int(output_head_scale)
        align_a = int(agg["B_q"]) * output_head_scale
        align_d = aggregate_output_scale
        align = division_bound_record(
            align_id,
            mode,
            a_bound=align_a,
            d_bound=align_d,
            d_min=align_d,
            count=int(node_count) * int(classes),
            source=family,
            numerator_source="SignedRoundDiv.aggregate.output*output_head_scale",
            denominator_source="authenticated_aggregate_output_scale",
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
    scales = resolve_quant_scale_profile(quant_cfg_id, authenticated_quant_scales)
    scale_chain = scale_chain_from_profile(
        scales,
        hidden_profile=hidden,
        k_out=int(k_out),
        quant_cfg_id=str(quant_cfg_id),
    )
    current_input_bound = max(1, int(feature_bound))
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
                output_input_scale=int(scale_chain["s_h"][position - 1]),
                aggregate_output_scale=int(scales["aggregate_output"]),
                output_head_scale=int(scales["output_head"]),
                exp_output_scale=int(scales["exp_output"]),
            )
            heads.append(record); divisions.extend(divs); outputs.append(output_bound)
        current_input_bound = max(outputs, default=1)

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
            output_input_scale=int(scale_chain["s_h"][-1]),
            aggregate_output_scale=int(scales["aggregate_output"]),
            output_head_scale=int(scales["output_head"]),
            exp_output_scale=int(scales["exp_output"]),
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
    table_domains = {
        "lrelu": [-max((int(item["B_S"]) for item in all_head_bounds), default=1), max((int(item["B_S"]) for item in all_head_bounds), default=1)],
        "elu": [-max((int(item["B_q_div"]) for item in heads), default=1), max((int(item["B_q_div"]) for item in heads), default=1)],
        "exp": [0, max((int(item["B_delta_raw"]) for item in all_head_bounds), default=1)],
    }
    function_bounds = {
        "T_LReLU": {
            "input_bound": max(abs(int(value)) for value in table_domains["lrelu"]),
            "output_bound": _function_output_bound(
                "lrelu", table_domains["lrelu"], scale_profile=scales
            ),
        },
        "T_ELU": {
            "input_bound": max(abs(int(value)) for value in table_domains["elu"]),
            "output_bound": _function_output_bound(
                "elu", table_domains["elu"], scale_profile=scales
            ),
        },
        "T_exp": {
            "input_bound": max(abs(int(value)) for value in table_domains["exp"]),
            "output_bound": _function_output_bound(
                "exp", table_domains["exp"], scale_profile=scales
            ),
        },
    }
    for table in function_bounds.values():
        x_bound = int(table["input_bound"])
        y_bound = int(table["output_bound"])
        table["base"] = 2 * y_bound + 1
        table["max_packed"] = 2 * x_bound * int(table["base"]) + 2 * y_bound
        if int(table["max_packed"]) >= FIELD_HALF:
            raise ValueError("function tuple packing exceeds BN254 signed field half")
    division_ids = [str(item["id"]) for item in divisions]
    division_by_id = {str(item["id"]): item for item in divisions}
    range_families: dict[str, dict[str, int | str]] = {}
    for item in all_head_bounds:
        tag = f"{item['kind']}.l{item['layer']}.h{item['head']}" if item["kind"] == "hidden" else f"output.h{item['head']}"
        range_families[f"{tag}.range_delta"] = _family_range_policy(item["B_delta"], range_bits)
        range_families[f"{tag}.M_range"] = _family_range_policy(item["B_Z"], range_bits, encoding_offset=item["B_Z"])
        range_families[f"{tag}.LeakyReLU.x_range"] = _family_range_policy(item["B_S"], range_bits, encoding_offset=item["B_S"])
        range_families[f"{tag}.LeakyReLU.y_range"] = _family_range_policy(item["B_Z"], range_bits, encoding_offset=item["B_Z"])
        range_families[f"{tag}.ExpMap.x_range"] = _family_range_policy(item["B_delta"], range_bits)
        range_families[f"{tag}.ExpMap.y_range"] = _family_range_policy(item["B_U"], range_bits)
        if item["kind"] == "hidden":
            range_families[f"{tag}.ELU.x_range"] = _family_range_policy(item["B_q_div"], range_bits, encoding_offset=item["B_q_div"])
            range_families[f"{tag}.ELU.y_range"] = _family_range_policy(item["B_H_agg"], range_bits, encoding_offset=item["B_H_agg"])
        aggregate_id = str(item["division_id"])
        aggregate = division_by_id[aggregate_id]
        range_families[f"{aggregate_id}.range"] = _family_range_policy(
            max(int(aggregate[key]) for key in ("B_a", "B_q", "B_r", "B_delta", "B_d")),
            range_bits,
            encoding_offset=max(int(aggregate[key]) for key in ("B_a", "B_q", "B_r", "B_delta", "B_d")),
        )
        if item.get("scale_alignment_division_id"):
            align_id = str(item["scale_alignment_division_id"])
            align = division_by_id[align_id]
            range_families[f"{align_id}.range"] = _family_range_policy(
                max(int(align[key]) for key in ("B_a", "B_q", "B_r", "B_delta", "B_d")),
                range_bits,
                encoding_offset=max(int(align[key]) for key in ("B_a", "B_q", "B_r", "B_delta", "B_d")),
            )
    final_policy = _family_range_policy(
        max(int(division_by_id[final_id][key]) for key in ("B_a", "B_q", "B_r", "B_delta", "B_d")),
        range_bits,
        encoding_offset=max(int(division_by_id[final_id][key]) for key in ("B_a", "B_q", "B_r", "B_delta", "B_d")),
    )
    range_families[f"{final_id}.range"] = final_policy
    # Matrix residuals have one canonical signed family.  The shared public
    # ledger does not inspect witness values, so use the certified maximum
    # integer bound as its conservative local width.
    max_bound = max((abs(int(value)) for value in _int_values({"heads": all_head_bounds, "divisions": divisions})), default=1)
    if max_bound >= FIELD_HALF:
        raise ValueError("integer_bound_ledger exceeds BN254 signed field half")
    range_families["rescale_rounding_range"] = _family_range_policy(max_bound, range_bits)
    encoded_max = max(
        [max_bound]
        + [int(item.get("encoded_bound", item.get("bound", 0))) for item in range_families.values()]
        + [int(item["max_packed"]) for item in function_bounds.values()],
    )
    if encoded_max >= FIELD_HALF:
        raise ValueError("integer-bound ledger packed/range encoding exceeds BN254 signed field half")
    return {
        "schema": "zkGPT_gat_integer_bound_ledger_v2",
        "field": "BN254-Fr",
        "field_half_modulus": str(FIELD_HALF),
        "max_in_degree": degree,
        "feature_bound": int(feature_bound),
        "quant_cfg_id": str(quant_cfg_id),
        "rounding_mode": mode,
        "authenticated_quant_scales": scales,
        "registered_model_bounds": {str(k): dict(v) for k, v in registered_model_bounds.items()},
        "bit_width": max(2, int(max_bound).bit_length()),
        "range_bits": int(range_bits),
        "heads": all_head_bounds,
        "signed_round_divisions": divisions,
        "division_instance_ids": division_ids,
        "division_cardinality": {str(item["id"]): int(item["count"]) for item in divisions},
        "table_domains": table_domains,
        "function_bounds": function_bounds,
        "final_average": {
            "K_out": int(k_out), "B_Y_sum": output_sum_bound,
            "denominator": int(k_out), "division_id": final_id,
        },
        "scale_chain": scale_chain,
        "scales": {
            "S_input": int(scales["input"]), "S_W_hidden": int(scales["hidden_weight"]), "S_W_output": int(scales["output_weight"]),
            "S_a_hidden": int(scales["hidden_attention"]), "S_a_output": int(scales["output_attention"]), "S_delta": int(scales["lrelu_input"]),
            "S_exp": int(scales["exp_output"]), "S_agg_pre": int(scales["aggregate_output"]), "S_agg": int(scales["aggregate_output"]),
            "S_out_pre": int(scales["aggregate_output"]), "S_out_head": int(scales["output_head"]), "S_out": int(scales["output_head"]),
            "S_H_prime_hidden": [[int(value) for value in row] for row in scale_chain["s_h_prime_hidden"]],
            "S_Y_prime": [int(value) for value in scale_chain["s_y_prime"]],
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
