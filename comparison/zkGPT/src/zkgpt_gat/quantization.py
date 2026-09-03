"""Baseline-local copy of the registered GAT fixed-point scale chain.

The comparison adapter deliberately does not import the main experiment at
runtime.  Keeping this small, auditable descriptor local gives checkpoint
loading, public-Y generation, the prover, and the independent verifier one
canonical source for the numerical identity that is being compared.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


CANONICAL_QUANT_CFG_ID = "gat_l3_scale_chain"
CANONICAL_CHAIN_SCHEMA = "gat_l3_scale_chain"
ROW_NORMALIZE_INPUT_PREPROCESS_ID = "row_normalize"
OGBN_ARXIV_RAW_INPUT_PREPROCESS_ID = "ogbn_arxiv_raw_float32"


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
    return {
        "schema": CANONICAL_CHAIN_SCHEMA,
        "quant_cfg_id": CANONICAL_QUANT_CFG_ID,
        "s_input": 16,
        "s_delta": 256,
        "s_exp": 16,
        "s_h": [16, 16, 16],
        "s_w_hidden": [[4] * 8, [4] * 8],
        "s_h_prime_hidden": [[64] * 8, [64] * 8],
        "s_a_src_hidden": [[4] * 8, [4] * 8],
        "s_a_dst_hidden": [[4] * 8, [4] * 8],
        "s_agg_pre": [16, 16],
        "s_agg": [16, 16],
        "s_cat": [16, 16],
        "s_out_pre": 16,
        "s_out_head": 16,
        "s_out": 16,
        "s_w_output": [4, 4],
        "s_y_prime": [64, 64],
        "s_a_src_output": [4, 4],
        "s_a_dst_output": [4, 4],
        "s_b_output": [16, 16],
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
    "quantize_ties_away",
]
