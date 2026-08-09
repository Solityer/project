#pragma once
// 本文件声明 trace 构建入口，
// 负责把前向结果、lookup、route 与状态机对象物化为正式 witness 列。

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol {

std::vector<std::size_t> stable_dynamic_commitment_chunk_order(
    const std::vector<std::vector<std::string>>& produced_labels,
    const std::vector<std::vector<std::string>>& required_labels);

// Compute the exact integer attention numerators in canonical edge order.
// The destination groups come from the formal edge ordering and let the
// production trace avoid rescanning every edge for every node and column.
std::vector<std::int64_t> weighted_edge_aggregate_raw_by_destination(
    const std::vector<std::int64_t>& edge_weights,
    const std::vector<std::vector<std::int64_t>>& source_rows,
    const std::vector<std::size_t>& edge_src_indices,
    const std::vector<std::size_t>& edge_dst_indices,
    const std::vector<std::pair<std::size_t, std::size_t>>& destination_groups,
    std::size_t node_count,
    std::string_view label);

struct SignedRoundDivInverseStageProfile {
    std::size_t active_count = 0;
    std::size_t column_size = 0;
    std::size_t zero_magnitude_count = 0;
    std::size_t positive_numerator_count = 0;
    std::size_t negative_numerator_count = 0;
    std::vector<double> scalar_inverse_ms;
    std::vector<double> batch_inverse_ms;
};

// Same-binary execution benchmark for the exact inverse-column core used by
// the formal SignedRoundDiv witness builder. It checks complete output columns,
// including signed/zero numerators, heterogeneous positive denominators, and
// inactive zero padding.
SignedRoundDivInverseStageProfile
benchmark_signed_round_div_inverse_stage(
    std::size_t active_count,
    std::size_t repeats);

struct TaylorShiftSelfCheckStageProfile {
    std::size_t polynomial_size = 0;
    std::vector<double> full_self_check_ms;
    std::vector<double> default_path_ms;
    algebra::FieldElement retained_u_rel = algebra::FieldElement::zero();
};

// Same-binary execution benchmark for the optional prover-side Taylor-shift
// closure checks. Both arms retain the proof-visible shifted evaluation; only
// the debug arm repeats the source and terminal-relation evaluations.
TaylorShiftSelfCheckStageProfile
benchmark_taylor_shift_self_check_stage(
    std::size_t polynomial_size,
    std::size_t repeats);

struct RouteConstructionStageProfile {
    std::size_t node_domain_size = 0;
    std::size_t node_active_count = 0;
    std::size_t edge_domain_size = 0;
    std::size_t edge_active_count = 0;
    std::vector<double> legacy_route_ms;
    std::vector<double> from_padded_route_ms;
};

// Same-binary execution benchmark for the formal route construction core.
// The legacy arm pads twice and materializes selectors/locations; the formal
// candidate retains the committed padding and uses active-prefix inverses.
RouteConstructionStageProfile benchmark_route_construction_stage(
    std::size_t node_domain_size,
    std::size_t node_active_count,
    std::size_t edge_domain_size,
    std::size_t edge_active_count,
    std::size_t repeats);

struct OutputHashStageProfile {
    std::size_t element_count = 0;
    std::vector<double> buffered_three_pass_ms;
    std::vector<double> streamed_reuse_ms;
};

// Same-binary comparison of the former three buffered textual field-matrix
// hashes and the formal streaming/reuse route. Both arms must produce the
// exact same public/reference/secure hash triplet.
OutputHashStageProfile benchmark_output_hash_stage(
    std::size_t row_count,
    std::size_t column_count,
    std::size_t repeats);

// Resolve the exact prover-side evaluation polynomial registered for a trace
// commitment. Large columns may be file-backed after commitment, so consumers
// must use this registry-aware path instead of assuming the commitment keeps a
// resident opening polynomial.
std::shared_ptr<const algebra::Polynomial> resolve_trace_opening_polynomial(
    const TraceArtifacts& trace,
    const std::string& label,
    const std::shared_ptr<algebra::RootOfUnityDomain>& expected_domain);

// 构建 proving 主链需要的全部动态 trace 对象。
// 它会同时产出动态承诺顺序、opening 取值来源、外点值与 witness 标量，是 prove 的唯一上游。
// 这里若改变标签顺序，不仅会影响 prover / verifier，还会连带改变 proof_size_bytes 的统计对象。
TraceArtifacts build_trace(const ProtocolContext& context, RunMetrics* metrics = nullptr);

}  // namespace gatzk::protocol
