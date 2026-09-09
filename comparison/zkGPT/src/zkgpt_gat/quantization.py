"""Baseline-local copy of the registered GAT fixed-point scale chain.

The comparison adapter deliberately does not import the main experiment at
runtime.  Keeping this small, auditable descriptor local gives checkpoint
loading, public-Y generation, the prover, and the independent verifier one
canonical source for the numerical identity that is being compared.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


CANONICAL_QUANT_CFG_ID = "gat_l3_scale_chain"
CANONICAL_CHAIN_SCHEMA = "gat_l3_scale_chain"
ROW_NORMALIZE_INPUT_PREPROCESS_ID = "row_normalize"
OGBN_ARXIV_RAW_INPUT_PREPROCESS_ID = "ogbn_arxiv_raw_float32"

# These are the only flat scale names that may participate in the formal
# integer forward.  Keeping the vocabulary here (instead of letting each
# consumer pick a subset) makes a quantization identity auditable at every
# producer/consumer boundary.
SCALE_PROFILE_KEYS = (
    "input", "hidden_weight", "hidden_attention", "output_weight",
    "output_attention", "output_bias", "lrelu_input", "lrelu_output",
    "exp_input", "exp_output", "elu_input", "elu_output",
    "aggregate_output", "output_head",
)


def canonical_input_preprocess_id(dataset: str) -> str:
    normalized = str(dataset).strip().lower().replace("-", "_")
    if normalized == "ogbn_arxiv":
        return OGBN_ARXIV_RAW_INPUT_PREPROCESS_ID
    if normalized in {"cora", "citeseer", "pubmed", "ppi"}:
        return ROW_NORMALIZE_INPUT_PREPROCESS_ID
    raise ValueError(f"unsupported formal dataset for input preprocessing: {dataset!r}")


def canonical_scale_profile() -> dict[str, int]:
    """Return the flat profile consumed by the existing integer forward path."""
    return {
        "input": 16,
        "hidden_weight": 4,
        "hidden_attention": 4,
        "output_weight": 4,
        "output_attention": 4,
        "output_bias": 16,
        "lrelu_input": 256,
        "lrelu_output": 256,
        "exp_input": 256,
        "exp_output": 16,
        "elu_input": 16,
        "elu_output": 16,
        "aggregate_output": 16,
        "output_head": 16,
    }


def canonical_scale_chain_descriptor(
    *, hidden_profile: Sequence[Mapping[str, object]] | None = None, k_out: int = 2
) -> dict[str, object]:
    if hidden_profile is not None:
        shape = tuple((int(item["head_count"]), int(item["head_dim"])) for item in hidden_profile)
        if shape != ((8, 8), (8, 8)):
            raise ValueError("canonical scale chain requires hidden_profile=[8x8,8x8]")
    if int(k_out) != 2:
        raise ValueError("canonical scale chain requires K_out=2")
    profile = canonical_scale_profile()
    selected_hidden = list(hidden_profile or ({"head_count": 8, "head_dim": 8}, {"head_count": 8, "head_dim": 8}))
    return scale_chain_from_profile(
        profile,
        hidden_profile=selected_hidden,
        k_out=int(k_out),
        quant_cfg_id=CANONICAL_QUANT_CFG_ID,
    )


def validate_scale_profile(
    profile: Mapping[str, Any],
    *,
    quant_cfg_id: str | None = None,
    require_canonical: bool = False,
) -> dict[str, int]:
    """Validate and normalize one authenticated flat scale profile.

    The function is intentionally independent of ``model.py`` so checkpoint
    loading, ledger construction, proof generation, and external verification
    can all use exactly the same derivation.  Non-canonical profiles are only
    accepted when explicitly supplied by a caller (the production model
    loader still requires :data:`CANONICAL_QUANT_CFG_ID`).
    """

    if not isinstance(profile, Mapping):
        raise ValueError("authenticated quantization profile must be an object")
    missing = [key for key in SCALE_PROFILE_KEYS if key not in profile]
    if missing:
        raise ValueError(f"authenticated quantization profile is missing: {missing}")
    unknown = [key for key in profile if str(key) not in SCALE_PROFILE_KEYS]
    if unknown:
        raise ValueError(f"authenticated quantization profile has unknown fields: {unknown}")
    normalized: dict[str, int] = {}
    for key in SCALE_PROFILE_KEYS:
        value = profile[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"quantization scale {key} must be an integer")
        value = int(value)
        if value <= 0:
            raise ValueError(f"quantization scale {key} must be positive")
        normalized[key] = value

    # Function-table scales and the aggregate/output scales are not free
    # metadata.  They must agree with the arithmetic chain that creates their
    # inputs and outputs.  These checks catch a wrong denominator or a stale
    # table descriptor before any witness/proof material is produced.
    if normalized["lrelu_input"] != normalized["exp_input"]:
        raise ValueError("LeakyReLU and ExpMap input scales differ")
    if normalized["lrelu_output"] != normalized["lrelu_input"]:
        raise ValueError("LeakyReLU output scale differs from its input scale")
    if normalized["exp_output"] != normalized["elu_output"]:
        raise ValueError("ExpMap output and ELU output scales differ")
    if normalized["elu_input"] != normalized["aggregate_output"]:
        raise ValueError("ELU input and aggregate output scales differ")
    if normalized["elu_output"] != normalized["aggregate_output"]:
        raise ValueError("ELU output and aggregate output scales differ")
    if normalized["output_bias"] != normalized["output_head"]:
        raise ValueError("output bias and output-head scales differ")

    if quant_cfg_id == CANONICAL_QUANT_CFG_ID or require_canonical:
        expected = canonical_scale_profile()
        if normalized != expected:
            raise ValueError("quantization profile is not the registered canonical scale chain")
    return normalized


def resolve_quant_scale_profile(
    quant_cfg_id: str,
    supplied: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Resolve a scale profile from an authenticated identity and payload.

    Canonical production identity is immutable.  A non-canonical profile can
    only be used when the complete profile is explicitly supplied (for tiny
    synthetic protocol fixtures); there is no name-based alias or fallback.
    """

    quant_cfg_id = str(quant_cfg_id)
    if quant_cfg_id == CANONICAL_QUANT_CFG_ID:
        profile = canonical_scale_profile() if supplied is None else supplied
        return validate_scale_profile(profile, quant_cfg_id=quant_cfg_id, require_canonical=True)
    if supplied is None:
        raise ValueError(f"unknown quantization identity without authenticated profile: {quant_cfg_id}")
    return validate_scale_profile(supplied, quant_cfg_id=quant_cfg_id)


def scale_chain_from_profile(
    profile: Mapping[str, Any],
    *,
    hidden_profile: Sequence[Mapping[str, Any]],
    k_out: int,
    quant_cfg_id: str,
) -> dict[str, object]:
    """Derive the nested scale chain used by forward/GKR/Lasso metadata."""

    scales = resolve_quant_scale_profile(quant_cfg_id, profile)
    hidden = list(hidden_profile)
    if not hidden:
        raise ValueError("scale chain requires at least one hidden layer")
    k_out = int(k_out)
    if k_out <= 0:
        raise ValueError("scale chain requires a positive output-head count")
    s_h = [int(scales["input"])] + [int(scales["elu_output"])] * len(hidden)
    hidden_w = []
    hidden_h_prime = []
    hidden_a_src = []
    hidden_a_dst = []
    s_agg_pre = []
    s_agg = []
    s_cat = []
    for layer in hidden:
        heads = int(layer["head_count"])
        if heads <= 0:
            raise ValueError("scale chain requires positive hidden head count")
        h_prime = s_h[len(hidden_h_prime)] * int(scales["hidden_weight"])
        # The score is formed from H' and attention kernels.  A scale chain
        # that does not close this product to S_delta is not authenticated.
        if int(scales["hidden_attention"]) * h_prime != int(scales["lrelu_input"]):
            raise ValueError("hidden attention scale does not close H' to score scale")
        hidden_w.append([int(scales["hidden_weight"])] * heads)
        hidden_h_prime.append([h_prime] * heads)
        hidden_a_src.append([int(scales["hidden_attention"])] * heads)
        hidden_a_dst.append([int(scales["hidden_attention"])] * heads)
        s_agg_pre.append(int(scales["aggregate_output"]))
        s_agg.append(int(scales["aggregate_output"]))
        s_cat.append(int(scales["aggregate_output"]))
        hidden_h_prime_count = len(hidden_h_prime)
        if hidden_h_prime_count > len(hidden):
            raise AssertionError("internal scale-chain construction error")
    output_h_prime = s_h[-1] * int(scales["output_weight"])
    if int(scales["output_attention"]) * output_h_prime != int(scales["lrelu_input"]):
        raise ValueError("output attention scale does not close Y' to score scale")
    return {
        "schema": CANONICAL_CHAIN_SCHEMA,
        "quant_cfg_id": str(quant_cfg_id),
        "s_input": int(scales["input"]),
        "s_delta": int(scales["lrelu_input"]),
        "s_exp": int(scales["exp_output"]),
        "s_h": s_h,
        "s_w_hidden": hidden_w,
        "s_h_prime_hidden": hidden_h_prime,
        "s_a_src_hidden": hidden_a_src,
        "s_a_dst_hidden": hidden_a_dst,
        "s_agg_pre": s_agg_pre,
        "s_agg": s_agg,
        "s_cat": s_cat,
        "s_out_pre": int(scales["aggregate_output"]),
        "s_out_head": int(scales["output_head"]),
        "s_out": int(scales["output_head"]),
        "s_w_output": [int(scales["output_weight"])] * k_out,
        "s_y_prime": [output_h_prime] * k_out,
        "s_a_src_output": [int(scales["output_attention"])] * k_out,
        "s_a_dst_output": [int(scales["output_attention"])] * k_out,
        "s_b_output": [int(scales["output_bias"])] * k_out,
        "rounding": "nearest_ties_away",
        "rescale": "none",
        "output_average": "signed_round_div",
    }


def quantize_ties_away(values: np.ndarray, scale: int) -> np.ndarray:
    if int(scale) <= 0:
        raise ValueError("quantization scale must be positive")
    raw = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        raise ValueError("quantization rejects NaN/Inf")
    scaled = raw * float(scale)
    return np.asarray(
        [int(np.floor(float(value) + 0.5) if float(value) >= 0.0 else np.ceil(float(value) - 0.5)) for value in scaled.ravel()],
        dtype=np.int64,
    ).reshape(raw.shape)


__all__ = [
    "CANONICAL_QUANT_CFG_ID",
    "CANONICAL_CHAIN_SCHEMA",
    "ROW_NORMALIZE_INPUT_PREPROCESS_ID",
    "OGBN_ARXIV_RAW_INPUT_PREPROCESS_ID",
    "canonical_input_preprocess_id",
    "canonical_scale_profile",
    "canonical_scale_chain_descriptor",
    "SCALE_PROFILE_KEYS",
    "validate_scale_profile",
    "resolve_quant_scale_profile",
    "scale_chain_from_profile",
    "quantize_ties_away",
]
