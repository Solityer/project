#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol
{

    std::vector<std::size_t> stable_dynamic_commitment_chunk_order(const std::vector<std::vector<std::string>> &produced_labels,
                                                                   const std::vector<std::vector<std::string>> &required_labels);

    bool quantized_reference_matches_secure(const model::Matrix &secure_trace_logits, const model::Matrix &reference_quant_logits);

    std::vector<std::int64_t> weighted_edge_aggregate_raw_by_destination(const std::vector<std::int64_t> &edge_weights,
                                                                         const std::vector<std::vector<std::int64_t>> &source_rows,
                                                                         const std::vector<std::size_t> &edge_src_indices,
                                                                         const std::vector<std::size_t> &edge_dst_indices,
                                                                         const std::vector<std::pair<std::size_t, std::size_t>> &destination_groups,
                                                                         std::size_t node_count, std::string_view label);

    std::shared_ptr<const algebra::Polynomial> resolve_trace_opening_polynomial(const TraceArtifacts &trace, const std::string &label,
                                                                                const std::shared_ptr<algebra::RootOfUnityDomain> &expected_domain);

    const std::vector<algebra::FieldElement> &resolve_trace_zero_domain_blinder(const TraceArtifacts &trace, const std::string &label,
                                                                                const std::shared_ptr<algebra::RootOfUnityDomain> &expected_domain);

    std::shared_ptr<const algebra::Polynomial> resolve_trace_coefficient_polynomial(const TraceArtifacts &trace, const std::string &label);

    void persist_completed_trace_coefficient(TraceArtifacts &trace, const std::string &label, bool keep_trace_payloads);

    TraceArtifacts build_trace(const ProtocolContext &context, RunMetrics *metrics = nullptr);
}
