#!/usr/bin/env python3
"""Canonical fixed-point scale-chain source for the formal GAT pipeline.

The values are deliberately loaded from the same small, checked-in source
file consumed by the C++ formal runtime.  Training, checkpoint metadata and
bundle export must never carry a second scale table or a scale-only default.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


_SOURCE_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs" / "quantization_scale_chain_v1.cfg"
CANONICAL_QUANT_CFG_ID = "gat_l3_scale_chain_v1"
CANONICAL_CHAIN_SCHEMA = "gat_l3_scale_chain_v1"

# Input preprocessing is a public part of the formal instance.  The
# Planetoid/PPI caches retain their historical row-normalized semantics, while
# ogbn-arxiv deliberately keeps the released float32 node features unchanged
# until the registered fixed-point boundary.  Keep this mapping here so every
# Python producer (training, trace/reference replay and export validation)
# selects the same contract instead of carrying a dataset-local normalization
# switch.
ROW_NORMALIZE_INPUT_PREPROCESS_ID = "row_normalize_v1"
OGBN_ARXIV_RAW_INPUT_PREPROCESS_ID = "ogbn_arxiv_raw_float32_v1"


def canonical_input_preprocess_id(dataset: str) -> str:
    """Return the one registered input preprocessing identity for a dataset."""
    normalized_dataset = str(dataset).strip().lower().replace("-", "_")
    if normalized_dataset == "ogbn_arxiv":
        return OGBN_ARXIV_RAW_INPUT_PREPROCESS_ID
    if normalized_dataset in {"cora", "citeseer", "pubmed", "ppi"}:
        return ROW_NORMALIZE_INPUT_PREPROCESS_ID
    raise ValueError(f"unsupported formal dataset for input preprocessing: {dataset!r}")


def _read_source() -> dict[str, str]:
    if not _SOURCE_PATH.is_file():
        raise ValueError(f"canonical quantization scale-chain source is missing: {_SOURCE_PATH}")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        _SOURCE_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{_SOURCE_PATH}:{line_number}: expected key=value")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value or key in values:
            raise ValueError(f"{_SOURCE_PATH}:{line_number}: malformed or duplicate key {key!r}")
        values[key] = value
    return values


def _parse_int(values: Mapping[str, str], key: str) -> int:
    try:
        value = int(values[key], 10)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"canonical scale chain has invalid integer {key}") from error
    if value <= 0:
        raise ValueError(f"canonical scale chain requires positive {key}")
    return value


def _parse_vector(values: Mapping[str, str], key: str) -> tuple[int, ...]:
    try:
        raw = values[key]
    except KeyError as error:
        raise ValueError(f"canonical scale chain is missing {key}") from error
    try:
        parsed = tuple(int(item.strip(), 10) for item in raw.split(","))
    except ValueError as error:
        raise ValueError(f"canonical scale chain has invalid vector {key}") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"canonical scale chain requires a positive vector {key}")
    return parsed


@dataclass(frozen=True)
class QuantizationScaleChain:
    schema: str
    quant_cfg_id: str
    s_input: int
    s_delta: int
    s_exp: int
    s_h: tuple[int, ...]
    s_w_hidden: tuple[tuple[int, ...], ...]
    s_h_prime_hidden: tuple[tuple[int, ...], ...]
    s_a_src_hidden: tuple[tuple[int, ...], ...]
    s_a_dst_hidden: tuple[tuple[int, ...], ...]
    s_agg_pre: tuple[int, ...]
    s_agg: tuple[int, ...]
    s_cat: tuple[int, ...]
    s_out_pre: int
    s_out_head: int
    s_out: int
    s_w_output: tuple[int, ...]
    s_y_prime: tuple[int, ...]
    s_a_src_output: tuple[int, ...]
    s_a_dst_output: tuple[int, ...]
    s_b_output: tuple[int, ...]
    rounding: str
    rescale: str
    output_average: str

    def canonical_serialization(self) -> str:
        vector = lambda name, value: f"{name}=[{','.join(str(item) for item in value)}]"
        matrix = lambda name, value: (
            f"{name}=[{';'.join('[' + ','.join(str(item) for item in row) + ']' for row in value)}]"
        )
        return (
            f"schema={self.schema}"
            f"|S_input={self.s_input}"
            f"|S_Delta={self.s_delta}"
            f"|S_exp={self.s_exp}"
            f"|{vector('S_H', self.s_h)}"
            f"|{matrix('S_W_hidden', self.s_w_hidden)}"
            f"|{matrix('S_H_prime_hidden', self.s_h_prime_hidden)}"
            f"|{matrix('S_a_src_hidden', self.s_a_src_hidden)}"
            f"|{matrix('S_a_dst_hidden', self.s_a_dst_hidden)}"
            f"|{vector('S_agg_pre', self.s_agg_pre)}"
            f"|{vector('S_agg', self.s_agg)}"
            f"|{vector('S_cat', self.s_cat)}"
            f"|S_out_pre={self.s_out_pre}"
            f"|S_out_head={self.s_out_head}"
            f"|S_out={self.s_out}"
            f"|{vector('S_W_output', self.s_w_output)}"
            f"|{vector('S_Y_prime_output', self.s_y_prime)}"
            f"|{vector('S_a_src_output', self.s_a_src_output)}"
            f"|{vector('S_a_dst_output', self.s_a_dst_output)}"
            f"|{vector('S_b_output', self.s_b_output)}"
            f"|S_inv=field_exact|S_alpha=field_exact"
            f"|round={self.rounding}|rescale={self.rescale}"
            f"|output_average={self.output_average}"
        )

    def hidden_head(self, layer_index: int, head_index: int) -> dict[str, int]:
        return {
            "input": self.s_h[layer_index],
            "weight": self.s_w_hidden[layer_index][head_index],
            "h_prime": self.s_h_prime_hidden[layer_index][head_index],
            "a_src": self.s_a_src_hidden[layer_index][head_index],
            "a_dst": self.s_a_dst_hidden[layer_index][head_index],
            "delta": self.s_delta,
            "exp": self.s_exp,
            "agg_pre": self.s_agg_pre[layer_index],
            "agg": self.s_agg[layer_index],
        }

    def output_head(self, head_index: int) -> dict[str, int]:
        return {
            "input": self.s_h[-1],
            "weight": self.s_w_output[head_index],
            # AttentionHead consumes the projection boundary under the
            # layer-neutral name ``h_prime``; retain ``y_prime`` as the
            # formal output-head terminology in the serialized chain.
            "h_prime": self.s_y_prime[head_index],
            "y_prime": self.s_y_prime[head_index],
            "a_src": self.s_a_src_output[head_index],
            "a_dst": self.s_a_dst_output[head_index],
            "delta": self.s_delta,
            "exp": self.s_exp,
            "agg_pre": self.s_out_pre,
            "out": self.s_out_head,
            "bias": self.s_b_output[head_index],
        }


def canonical_quantization_scale_chain(
    quant_cfg_id: str = CANONICAL_QUANT_CFG_ID,
    hidden_profile: Sequence[Mapping[str, object]] | None = None,
    k_out: int = 2,
) -> QuantizationScaleChain:
    values = _read_source()
    if quant_cfg_id != values.get("quant_cfg_id") or quant_cfg_id != CANONICAL_QUANT_CFG_ID:
        raise ValueError(
            "unsupported quant_cfg_id; formal training requires the registered canonical scale chain"
        )
    if hidden_profile is not None:
        shape = tuple(
            (int(entry["head_count"]), int(entry["head_dim"]))
            for entry in hidden_profile
        )
        if shape != ((8, 8), (8, 8)):
            raise ValueError("canonical quantization scale chain requires hidden_profile=[8x8,8x8]")
    if int(k_out) != 2:
        raise ValueError("canonical quantization scale chain requires K_out=2")

    chain = QuantizationScaleChain(
        schema=values.get("schema", ""),
        quant_cfg_id=values["quant_cfg_id"],
        s_input=_parse_int(values, "s_input"),
        s_delta=_parse_int(values, "s_delta"),
        s_exp=_parse_int(values, "s_exp"),
        s_h=_parse_vector(values, "s_h"),
        s_w_hidden=tuple(_parse_vector(values, f"s_w_hidden_{layer}") for layer in range(2)),
        s_h_prime_hidden=tuple(_parse_vector(values, f"s_h_prime_hidden_{layer}") for layer in range(2)),
        s_a_src_hidden=tuple(_parse_vector(values, f"s_a_src_hidden_{layer}") for layer in range(2)),
        s_a_dst_hidden=tuple(_parse_vector(values, f"s_a_dst_hidden_{layer}") for layer in range(2)),
        s_agg_pre=_parse_vector(values, "s_agg_pre"),
        s_agg=_parse_vector(values, "s_agg"),
        s_cat=_parse_vector(values, "s_cat"),
        s_out_pre=_parse_int(values, "s_out_pre"),
        s_out_head=_parse_int(values, "s_out_head"),
        s_out=_parse_int(values, "s_out"),
        s_w_output=_parse_vector(values, "s_w_output"),
        s_y_prime=_parse_vector(values, "s_y_prime"),
        s_a_src_output=_parse_vector(values, "s_a_src_output"),
        s_a_dst_output=_parse_vector(values, "s_a_dst_output"),
        s_b_output=_parse_vector(values, "s_b_output"),
        rounding=values.get("round", ""),
        rescale=values.get("rescale", ""),
        output_average=values.get("output_average", ""),
    )
    if chain.schema != CANONICAL_CHAIN_SCHEMA:
        raise ValueError("canonical scale-chain source schema mismatch")
    if chain.s_h != (chain.s_input, chain.s_cat[0], chain.s_cat[1]):
        raise ValueError("canonical scale-chain H transition mismatch")
    if chain.s_cat != chain.s_agg or chain.s_out_pre != chain.s_out_head or chain.s_out != chain.s_out_head:
        raise ValueError("canonical scale-chain output/concat transition mismatch")
    for layer in range(2):
        if not (
            len(chain.s_w_hidden[layer]) == len(chain.s_h_prime_hidden[layer]) == 8
            and len(chain.s_a_src_hidden[layer]) == len(chain.s_a_dst_hidden[layer]) == 8
        ):
            raise ValueError("canonical scale-chain hidden head count mismatch")
        for head in range(8):
            if chain.s_h_prime_hidden[layer][head] != chain.s_h[layer] * chain.s_w_hidden[layer][head]:
                raise ValueError("canonical scale-chain hidden H/W equation mismatch")
            if chain.s_h_prime_hidden[layer][head] * chain.s_a_src_hidden[layer][head] != chain.s_delta:
                raise ValueError("canonical scale-chain hidden source attention equation mismatch")
            if chain.s_h_prime_hidden[layer][head] * chain.s_a_dst_hidden[layer][head] != chain.s_delta:
                raise ValueError("canonical scale-chain hidden destination attention equation mismatch")
    if not all(
        chain.s_y_prime[head] == chain.s_h[-1] * chain.s_w_output[head]
        and chain.s_y_prime[head] * chain.s_a_src_output[head] == chain.s_delta
        and chain.s_y_prime[head] * chain.s_a_dst_output[head] == chain.s_delta
        and chain.s_b_output[head] == chain.s_out_head
        for head in range(2)
    ):
        raise ValueError("canonical scale-chain output equation mismatch")
    if chain.rounding != "nearest_ties_away" or chain.rescale != "none" or chain.output_average != "signed_round_div":
        raise ValueError("canonical scale-chain rounding/rescale policy mismatch")
    return chain


def quantize_array_ties_away(values: np.ndarray, scale: int) -> np.ndarray:
    """Return canonical signed integer q=round_away(float64(values)*scale)."""
    if scale <= 0:
        raise ValueError("quantization scale must be positive")
    raw = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        raise ValueError("quantization rejects NaN/Inf")
    scaled = raw * float(scale)
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    if not np.all(np.isfinite(rounded)):
        raise ValueError("quantization result is not finite")
    i64 = np.iinfo(np.int64)
    if np.any(rounded > i64.max) or np.any(rounded < i64.min):
        raise ValueError("quantization result exceeds int64")
    return rounded.astype(np.int64)


def dequantize_array_ties_away(values: np.ndarray, scale: int) -> np.ndarray:
    return quantize_array_ties_away(values, scale).astype(np.float64) / float(scale)


def canonical_row_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize rows with the formal C++ double left-fold semantics."""
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError("row normalization expects a rank-2 matrix")
    if not np.all(np.isfinite(raw)):
        raise ValueError("row normalization rejects NaN/Inf")
    normalized = raw.copy()
    for row_index, row in enumerate(raw):
        # Do not replace this with np.sum: its reduction order is an
        # implementation detail and can disagree with the C++ scalar loop at
        # a half-away-from-zero quantization tie.
        rowsum = 0.0
        for value in row:
            rowsum += float(value)
        if not np.isfinite(rowsum):
            raise ValueError("row normalization sum is not finite")
        if rowsum != 0.0:
            for column_index, value in enumerate(row):
                normalized[row_index, column_index] = float(value) / rowsum
    if not np.all(np.isfinite(normalized)):
        raise ValueError("row normalization produced NaN/Inf")
    return normalized


def canonical_input_preprocess(
    values: np.ndarray,
    dataset: str,
    input_preprocess_id: str | None = None,
) -> np.ndarray:
    """Materialize the canonical floating-point input view.

    ``ogbn_arxiv_raw_float32_v1`` intentionally casts through float32 before
    returning float64 so a direct raw-array caller has exactly the same
    numerical source as the cached ``features.npy`` producer.  The formal
    quantizer is applied by the caller after this boundary.
    """
    expected_id = canonical_input_preprocess_id(dataset)
    actual_id = expected_id if input_preprocess_id is None else str(input_preprocess_id)
    if actual_id != expected_id:
        raise ValueError(
            f"input_preprocess_id {actual_id!r} does not match dataset {dataset!r}; "
            f"expected {expected_id!r}"
        )
    raw = np.asarray(values)
    if raw.ndim != 2:
        raise ValueError("canonical input preprocessing expects a rank-2 matrix")
    if actual_id == OGBN_ARXIV_RAW_INPUT_PREPROCESS_ID:
        raw_float32 = np.asarray(raw, dtype=np.float32)
        if not np.all(np.isfinite(raw_float32)):
            raise ValueError("raw float32 input preprocessing rejects NaN/Inf")
        return raw_float32.astype(np.float64, copy=False)
    return canonical_row_normalize(np.asarray(raw, dtype=np.float64))
