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

    struct Layer0Head0MechanismWitness
    {
        std::size_t node_count = 0;
        std::size_t edge_count = 0;
        std::size_t width = 0;
        std::vector<std::size_t> node_ptr;
        std::vector<std::size_t> edge_ptr;
        std::vector<std::size_t> edge_src_indices;
        std::vector<std::size_t> edge_dst_indices;
        std::vector<std::vector<std::int64_t>> h_prime_raw;
        model::Matrix h_prime;
        std::vector<std::int64_t> e_src_raw;
        std::vector<std::int64_t> e_dst_raw;
        std::vector<algebra::FieldElement> e_src;
        std::vector<algebra::FieldElement> e_dst;
        std::vector<std::int64_t> u_raw;
        std::vector<std::int64_t> sum_raw;
        std::vector<algebra::FieldElement> u;
        std::vector<algebra::FieldElement> sum;
        std::vector<algebra::FieldElement> alpha;
        model::Matrix weighted_edge_features;
        model::Matrix aggregate_pre_activation;
        std::uint64_t s_input = 0;
        std::uint64_t s_w_hidden = 0;
        std::uint64_t s_h_prime = 0;
        std::uint64_t s_a_src_hidden = 0;
        std::uint64_t s_a_dst_hidden = 0;
        std::uint64_t s_delta = 0;
        std::uint64_t s_exp = 0;
        std::uint64_t s_agg_pre = 0;
    };

    // Extract only the canonical first hidden layer's first attention head.
    // `context` may be minimally populated: local, model, config.quantization,
    // config.range_bits, and tables.exp are required; proof domains, SRS,
    // public polynomials, and ELU are not used.
    Layer0Head0MechanismWitness extract_layer0_head0_mechanism_witness(const ProtocolContext &context);

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
