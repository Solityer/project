#!/usr/bin/env python3
"""Static contract for the internal t_edge descriptor integration."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVER = (ROOT / "src/protocol/prover.cpp").read_text(encoding="utf-8")
HELPER_H = (ROOT / "src/protocol/internal/t_edge_descriptor.hpp").read_text(
    encoding="utf-8"
)
HELPER = (ROOT / "src/protocol/internal/t_edge_descriptor.cpp").read_text(
    encoding="utf-8"
)
CUDA_HEADER = (ROOT / "include/gatzk/protocol/cuda_t_edge.hpp").read_text(
    encoding="utf-8"
)
CUDA_SOURCE = (ROOT / "src/protocol/cuda_t_edge.cu").read_text(encoding="utf-8")


# The mapping and recursive flattening implementation has one internal source
# of truth.  Prover retains only thin ownership-aware wrappers.
assert "internal/t_edge_descriptor.hpp" in PROVER
assert "struct TEdgeDerivedLinearDescriptor" not in PROVER
assert "enum class TEdgeDerivedLinearFlattenStatus" not in PROVER
assert 'suffix == "Table_L"' not in PROVER
assert 'suffix == "Query_dst"' not in PROVER
assert "internal::make_t_edge_derived_linear_descriptor(" in PROVER
assert "internal::flatten_t_edge_derived_linear_sources(" in PROVER
assert "target_zero_domain_blinder" in PROVER
assert "collision_predicate" in PROVER


# The internal helper is intentionally outside include/ and has no proving-
# context or proof API dependency in its public header.
for token in (
    "TEdgeDescriptorShape",
    "TEdgeDerivedLinearDescriptor",
    "TEdgeDerivedLinearFlattenStatus",
    "make_t_edge_derived_linear_descriptor",
    "flatten_t_edge_derived_linear_sources",
    "UnknownSource",
    "Cycle",
    "DepthExceeded",
):
    assert token in HELPER_H or token in HELPER, token
assert "gatzk/protocol/proof.hpp" not in HELPER_H
assert "ProtocolContext" not in HELPER_H
for token in (
    'suffix == "Table_L"',
    'suffix == "Query_src"',
    'suffix == "Query_dst"',
    'suffix == "w_psq"',
    'suffix == "T_psq_edge"',
    'suffix == "w"',
    'suffix == "T_edge"',
    '"eta_src_h"',
    '"eta_src_out"',
    '"lambda_psq_h"',
    '"lambda_out"',
    "kMaximumDepth",
    "positions",
    "merged += scale",
    "merged.is_zero()",
):
    assert token in HELPER, token


# The resolver remains cache -> derived -> direct/spill.  A derived target is
# attempted by resident-linear first and falls through to the host view.
coset = PROVER[PROVER.index("QuotientCosetValuesPtr coset_values_ptr(") :]
assert coset.index("coset_cache_.find") < coset.index(
    "t_edge_derived_linear_descriptor(requested_name)"
)
assert coset.index("t_edge_derived_linear_descriptor(requested_name)") < coset.index(
    "lookup_polynomial(requested_name)"
)
materialize = PROVER[PROVER.index("const auto materialize_label =") :]
assert materialize.index("t_edge_derived_linear_descriptor(label)") < materialize.index(
    "try_materialize_derived_linear_gpu_leaf("
)
assert materialize.index("try_materialize_derived_linear_gpu_leaf(") < materialize.index(
    "try_materialize_direct_gpu_leaf("
)
assert "host_derived_fallback" in materialize


# Coefficient leaves are accepted by the linear source path only up to the
# half-size paired-NTT input; the over-wide shape is not advertised.
shape = PROVER[PROVER.index("linear_direct_shape_is_supported") :]
assert "target_size_ / 2" in shape
assert "maximum_coefficient_count" in shape


# Capability and telemetry remain execution-only and do not add proof fields.
assert "cuda_t_edge_leaf_exact_ntt_supported" in CUDA_HEADER
assert "target_size / 2" in CUDA_SOURCE
assert "CudaTEdgeExecutorAdmissionShape" in CUDA_HEADER
assert "cuda_t_edge_executor_can_admit(" in CUDA_SOURCE
assert "CUDA t_edge large route was selected but admission " in PROVER
assert "CUDA t_edge large route was selected but " in PROVER
assert "direct leaf execution failed: target_size=" in PROVER
assert "linear leaf execution failed: target_size=" in PROVER
assert "CPUOnly or an unselected CUDA route" in CUDA_HEADER
for token in (
    "descriptor_requests",
    "descriptor_eligible",
    "linear_hits",
    "fallback_capacity",
    "fallback_unsupported",
    "fallback_api_declined",
    "host_derived_fallback",
    "host_leaf_materialization_required",
    "leaf_exact_ntt_supported",
):
    assert token in PROVER

print("t_edge descriptor/resident contract: PASS")
