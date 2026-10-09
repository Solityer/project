#include "../zkrelation-proof.cuh"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>

using namespace zkdl_gat::relation;

namespace {

struct ReadCounts {
    std::uint64_t key_segments = 0U;
    std::uint64_t active_segments = 0U;
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

SegmentedTensorProvider counted_provider(const std::vector<FrWire>& values,
                                         std::uint32_t segment_size,
                                         const std::shared_ptr<ReadCounts>& counts,
                                         bool key_source) {
    const auto* source = &values;
    return SegmentedTensorProvider(
        static_cast<std::uint32_t>(values.size()), segment_size,
        [source, counts, key_source](std::uint64_t offset, std::uint32_t count,
                                     std::vector<FrWire>& output,
                                     std::string* error) {
            if (offset > source->size() || count > source->size() - offset) {
                if (error) *error = "fixture provider range is out of bounds";
                return false;
            }
            if (key_source) ++counts->key_segments;
            else ++counts->active_segments;
            output.assign(source->begin() + static_cast<std::ptrdiff_t>(offset),
                          source->begin() + static_cast<std::ptrdiff_t>(offset + count));
            return true;
        });
}

std::vector<FrWire> legacy_prefix(const SegmentedTensorProvider& keys,
                                  const SegmentedTensorProvider& active,
                                  const FrWire& beta, const FrWire& gamma,
                                  std::uint64_t offset, std::uint32_t count,
                                  std::string* error) {
    std::vector<FrWire> output(count, FrWire{});
    FrWire product = zkdl::fr_from_u32(1U);
    const std::uint64_t end = offset + count;
    for (std::uint32_t segment = 0U; segment < keys.segment_count(); ++segment) {
        std::vector<FrWire> key_block, active_block;
        if (!keys.read_segment(segment, key_block, error) ||
            !active.read_segment(segment, active_block, error)) return {};
        for (std::uint32_t local = 0U; local < keys.segment_size(); ++local) {
            const std::uint64_t global =
                static_cast<std::uint64_t>(segment) * keys.segment_size() + local;
            if (active_block[local] == zkdl::fr_from_u32(1U)) {
                if (global >= offset && global < end)
                    output[static_cast<std::size_t>(global - offset)] = product;
                product = zkdl::fr_mul(
                    product, zkdl::fr_add(zkdl::fr_mul(beta, key_block[local]), gamma));
            }
        }
    }
    return output;
}

std::vector<FrWire> oracle(const std::vector<FrWire>& keys,
                           const std::vector<FrWire>& active,
                           const FrWire& beta, const FrWire& gamma) {
    std::vector<FrWire> result(keys.size(), FrWire{});
    FrWire product = zkdl::fr_from_u32(1U);
    for (std::size_t index = 0U; index < keys.size(); ++index) {
        if (active[index] == zkdl::fr_from_u32(1U)) {
            result[index] = product;
            product = zkdl::fr_mul(
                product, zkdl::fr_add(zkdl::fr_mul(beta, keys[index]), gamma));
        }
    }
    return result;
}

bool same(const std::vector<FrWire>& left, const std::vector<FrWire>& right) {
    return left.size() == right.size() &&
        (left.empty() || std::memcmp(left.data(), right.data(),
                                     left.size() * sizeof(FrWire)) == 0);
}

void check_negative(const std::vector<FrWire>& input_keys,
                    const std::vector<FrWire>& input_active,
                    const FrWire& beta, const FrWire& gamma,
                    const char* expected) {
    auto counts = std::make_shared<ReadCounts>();
    const auto keys = counted_provider(input_keys, 8U, counts, true);
    const auto active = counted_provider(input_active, 8U, counts, false);
    SegmentedTensorProvider prefix;
    std::string error;
    const auto digest = zkdl::sha256(std::vector<zkdl::byte>{'n'});
    require(make_authenticated_sorted_lookup_prefix_product_provider(
                keys, active, beta, gamma, prefix, &error, nullptr, digest, 5U),
            "negative fixture could not create a lazy prefix provider");
    std::vector<FrWire> output;
    require(!prefix.read_segment(0U, output, &error),
            "malformed sorted-lookup prefix source was accepted");
    if (error.find(expected) == std::string::npos)
        throw std::runtime_error(std::string(
            "malformed sorted-lookup prefix source error mismatch expected=") +
            expected + " actual=" + error);
    const auto before_retry = counts->key_segments + counts->active_segments;
    error.clear();
    require(!prefix.read_segment(1U, output, &error),
            "failed sorted-lookup prefix cache silently retried");
    require(error.find(expected) != std::string::npos &&
                counts->key_segments + counts->active_segments == before_retry,
            "failed sorted-lookup prefix cache did not retain its failure state");
}

void check_square_product_alias(const std::vector<zkdl::CpuG1Point>& generators,
                                const zkdl::CpuG1Point& h,
                                const zkdl::CpuG1Point& u) {
    ConstraintDescriptor square;
    square.id = "tiny/square";
    square.degree = 2U;
    Monomial square_term;
    square_term.coefficient = zkdl::fr_from_u32(1U);
    square_term.refs = {0U, 0U};
    square.terms.push_back(square_term);
    std::vector<ZkTerminalProductPlan> square_plans;
    std::vector<std::uint32_t> square_monomials;
    std::string error;
    if (!zktr_build_product_plan(square, 1U, {0U}, square_plans,
                                 square_monomials, &error) ||
        square_plans.size() != 1U)
        throw std::runtime_error("square terminal product plan failed: " + error);
    std::uint32_t expected_alias = 0U;
    require(zktr_product_alias_index(0U, expected_alias, &error) &&
                square_plans[0U].left_commitment_index ==
                    zktr_terminal_evaluation_index(0U) &&
                square_plans[0U].right_alias_source_index ==
                    square_plans[0U].left_commitment_index &&
                square_plans[0U].right_commitment_index == expected_alias &&
                square_plans[0U].output_commitment_index == zktr_product_index(0U) &&
                square_plans[0U].left_commitment_index !=
                    square_plans[0U].right_commitment_index &&
                square_plans[0U].right_commitment_index !=
                    square_plans[0U].output_commitment_index,
            "square terminal product roles are not deterministically separated");

    ConstraintDescriptor product;
    product.id = "tiny/non-square-product";
    product.degree = 2U;
    Monomial product_term;
    product_term.coefficient = zkdl::fr_from_u32(1U);
    product_term.refs = {0U, 1U};
    product.terms.push_back(product_term);
    std::vector<ZkTerminalProductPlan> product_plans;
    std::vector<std::uint32_t> product_monomials;
    require(zktr_build_product_plan(product, 2U, {0U, 1U}, product_plans,
                                    product_monomials, &error) &&
                product_plans.size() == 1U &&
                product_plans[0U].right_alias_source_index ==
                    std::numeric_limits<std::uint32_t>::max() &&
                product_plans[0U].right_commitment_index ==
                    zktr_terminal_evaluation_index(1U),
            "non-square product plan changed its existing wire roles");

    RelationStatement statement;
    statement.statement_digest = zkdl::sha256(std::vector<zkdl::byte>{'s'});
    statement.model_digest = zkdl::sha256(std::vector<zkdl::byte>{'m'});
    statement.table_digest = zkdl::sha256(std::vector<zkdl::byte>{'t'});
    statement.ledger_digest = zkdl::sha256(std::vector<zkdl::byte>{'l'});
    statement.registry_digest = zkdl::sha256(std::vector<zkdl::byte>{'r'});
    statement.srs_digest = zkdl::sha256(std::vector<zkdl::byte>{'p'});
    const auto descriptor_digest = zkdl::sha256(std::vector<zkdl::byte>{'q'});
    const auto& plan = square_plans[0U];
    const auto context = zktr_make_product_context(
        statement, descriptor_digest, h, u, plan.left_commitment_index,
        plan.right_commitment_index, plan.output_commitment_index);
    for (const std::uint32_t input : {0U, 1U, 2U}) {
        const FrWire x = zkdl::fr_from_u32(input);
        const FrWire blinder = zkdl::fr_from_u32(127U);
        const FrWire output_blinder = zkdl::fr_from_u32(131U);
        const FrWire x_squared = zkdl::fr_mul(x, x);
        const auto commitment = zkdl::cpu_pedersen_scalar_commit(x, blinder, h, u);
        const auto output_commitment = zkdl::cpu_pedersen_scalar_commit(
            x_squared, output_blinder, h, u);
        const auto proof = zkdl::cpu_open_pedersen_product(
            x, x, blinder, blinder, output_blinder, commitment, commitment,
            output_commitment, h, u, context).proof;
        std::string verify_error;
        if (!zkdl::cpu_verify_pedersen_product(
                proof, commitment, commitment, output_commitment, h, u,
                context, &verify_error))
            throw std::runtime_error(
                "fresh square-product verifier rejected x=" +
                std::to_string(input) + ": " + verify_error);

        auto bad_response = proof;
        bad_response.z = zkdl::fr_value(zkdl::fr_add(
            zkdl::fr_wire(bad_response.z), zkdl::fr_from_u32(1U)));
        verify_error.clear();
        require(!zkdl::cpu_verify_pedersen_product(
                    bad_response, commitment, commitment, output_commitment,
                    h, u, context, &verify_error),
                "square-product verifier accepted a tampered x-squared response");
    }
}

void check_marker_hybrid_sumcheck(const std::vector<zkdl::CpuG1Point>& generators,
                                  const zkdl::CpuG1Point& h,
                                  const zkdl::CpuG1Point& u) {
    const std::uint32_t expected_count = 2U;
    const FrWire count_rho = zkdl::fr_from_u32(5U), padding_rho = zkdl::fr_from_u32(7U);
    const std::vector<FrWire> piece_rhos = {count_rho, padding_rho}, eq_rho = {zkdl::fr_from_u32(11U)};
    ConstraintDescriptor relation;
    relation.id = "authenticated_sorted_lookup/relation/3";
    relation.degree = 2U;
    relation.constant = zkdl::fr_neg(zkdl::fr_mul(count_rho, zkdl::fr_from_u32(expected_count)));
    relation.terms = {{count_rho, {0U}}, {padding_rho, {0U}}, {zkdl::fr_neg(padding_rho), {0U, 1U}}};
    std::vector<ConstraintDescriptor> pieces(2U);
    pieces[0U].id = "marker-count";
    pieces[0U].degree = 1U;
    pieces[0U].constant = zkdl::fr_neg(zkdl::fr_from_u32(expected_count));
    pieces[0U].terms = {{zkdl::fr_from_u32(1U), {0U}}};
    pieces[1U].id = "marker-padding";
    pieces[1U].degree = 2U;
    pieces[1U].terms = {{zkdl::fr_from_u32(1U), {0U}}, {zkdl::fr_neg(zkdl::fr_from_u32(1U)), {0U, 1U}}};
    std::string error;
    require(sorted_marker_pieces_valid(pieces, piece_rhos, expected_count, &error),
            "canonical marker hybrid pieces were rejected");
    const ConstraintDescriptor mixed = sorted_marker_mixed_sumcheck_descriptor(relation, pieces, piece_rhos, 2U);
    RelationStatement statement;
    statement.statement_digest = zkdl::sha256(std::vector<zkdl::byte>{'s'});
    statement.model_digest = zkdl::sha256(std::vector<zkdl::byte>{'m'});
    statement.table_digest = zkdl::sha256(std::vector<zkdl::byte>{'t'});
    statement.ledger_digest = zkdl::sha256(std::vector<zkdl::byte>{'l'});
    statement.registry_digest = zkdl::sha256(std::vector<zkdl::byte>{'r'});
    statement.srs_digest = zkdl::sha256(std::vector<zkdl::byte>{'p'});
    const auto run = [&](const std::vector<FrWire>& markers, const std::vector<FrWire>& active, bool expected_valid, const char* label) {
        SegmentedTensorProvider marker_provider, active_provider;
        require(SegmentedTensorProvider::from_values(markers, 1U, marker_provider, &error) && SegmentedTensorProvider::from_values(active, 1U, active_provider, &error), "marker hybrid fixture provider setup failed");
        std::vector<SegmentedTensorProvider> providers, inputs = {marker_provider, active_provider};
        require(pointwise_providers(relation, inputs, eq_rho, providers, &error), "marker hybrid public mask provider setup failed");
        const FrWire initial_claim = zkdl::fr_mul(count_rho, zkdl::fr_from_u32(expected_count));
        zkdl::Transcript prover_transcript("tiny-sorted-marker-hybrid-v1");
        ZkSumcheckProverResult prover;
        error.clear();
        const bool proved = prove_zk_sumcheck_segmented(mixed, providers, mixed.degree, 2U, generators, h, u, initial_claim, statement, label, prover_transcript, prover, &error, 1U);
        require(proved == expected_valid, expected_valid ? "valid marker hybrid Sumcheck was rejected" : "invalid marker count/padding passed native Sumcheck");
        if (!expected_valid) return;
        zkdl::Transcript verifier_transcript("tiny-sorted-marker-hybrid-v1");
        ZkSumcheckVerifierResult verifier;
        require(verify_zk_sumcheck(prover.proof, mixed, mixed.degree, 2U, generators, h, u, initial_claim, statement, label, verifier_transcript, verifier, &error, 1U), "fresh marker hybrid Sumcheck verifier rejected a valid proof");
    };
    run({zkdl::fr_from_u32(1U), zkdl::fr_from_u32(1U)}, {zkdl::fr_from_u32(1U), zkdl::fr_from_u32(1U)}, true, "tiny.marker.hybrid.valid");
    run({zkdl::fr_from_u32(1U), FrWire{}}, {zkdl::fr_from_u32(1U), zkdl::fr_from_u32(1U)}, false, "tiny.marker.hybrid.bad-count");
    run({zkdl::fr_from_u32(1U), zkdl::fr_from_u32(1U)}, {zkdl::fr_from_u32(1U), FrWire{}}, false, "tiny.marker.hybrid.bad-padding");
}

void check_tiny_sorted_lookup(const std::vector<std::uint32_t>& query_keys,
                              const std::vector<std::uint32_t>& merged_keys,
                              bool expect_valid,
                              const std::vector<zkdl::CpuG1Point>& generators,
                              const zkdl::CpuG1Point& h,
                              const zkdl::CpuG1Point& u) {
    std::map<std::string, std::vector<FrWire>> values;
    const auto put = [&](const std::string& name,
                         const std::vector<std::uint32_t>& input) {
        auto& output = values[name];
        output.reserve(input.size());
        for (const std::uint32_t value : input)
            output.push_back(zkdl::fr_from_u32(value));
    };
    const auto put_fields = [&](const std::string& name,
                                const std::vector<FrWire>& input) {
        values[name] = input;
    };
    const auto put_zero = [&](const std::string& name, std::uint32_t size) {
        values[name].assign(size, FrWire{});
    };
    put("table_keys", {0U, 3U, 4U, 0U});
    put("table_active", {1U, 1U, 1U, 0U});
    std::vector<std::uint32_t> query_x, query_y, query_scratch;
    query_x.reserve(query_keys.size());
    query_y.reserve(query_keys.size());
    query_scratch.reserve(query_keys.size());
    for (std::size_t index = 0U; index < query_keys.size(); ++index) {
        query_x.push_back(query_keys[index] >> 2U);
        query_y.push_back(query_keys[index] & 3U);
        query_scratch.push_back(static_cast<std::uint32_t>(index + 1U));
    }
    put("query_keys", query_keys);
    put("query_x", query_x);
    put("query_y", query_y);
    put("query_prefix", query_scratch);
    put("query_active", {1U, 1U});
    std::vector<std::uint32_t> padded_merge_keys(8U, 0U);
    std::copy(merged_keys.begin(), merged_keys.end(), padded_merge_keys.begin());
    std::vector<std::uint32_t> merge_delta(8U, 0U);
    std::vector<std::uint32_t> merge_marker(8U, 0U);
    std::vector<FrWire> merge_inverse(8U, FrWire{});
    std::vector<std::uint32_t> merge_scratch(8U, 0U);
    std::uint32_t distinct = 0U;
    for (std::size_t index = 0U; index < merged_keys.size(); ++index) {
        if (index != 0U) merge_delta[index] = merged_keys[index] - merged_keys[index - 1U];
        merge_marker[index] = index == 0U || merged_keys[index] != merged_keys[index - 1U];
        if (merge_marker[index] != 0U) ++distinct;
        merge_scratch[index] = distinct;
        if (merge_delta[index] != 0U) {
            const FrWire delta = zkdl::fr_from_u32(merge_delta[index]);
            merge_inverse[index] = zkdl::fr_pow(
                delta, 256U, zkdl::fr_modulus_minus_two());
        }
    }
    put("merge_keys", padded_merge_keys);
    put("merge_delta", merge_delta);
    put("merge_marker", merge_marker);
    put_fields("merge_inverse", merge_inverse);
    put("merge_prefix", merge_scratch);
    put("merge_active", {1U, 1U, 1U, 1U, 1U, 0U, 0U, 0U});
    put_zero("query_prefix_next", 2U);
    put_zero("merge_prefix_next", 8U);

    const std::vector<std::pair<std::string, std::vector<FrWire>>> identity_guard = {
        {"table_keys", values.at("table_keys")},
        {"table_active", values.at("table_active")},
        {"query_keys", values.at("query_keys")},
        {"query_active", values.at("query_active")},
        {"merge_keys", values.at("merge_keys")},
        {"merge_active", values.at("merge_active")}};
    const auto h_before_identity = h;
    require(zkdl::cpu_g1_point_valid(h_before_identity) && !h_before_identity.is_zero(),
            "tiny sorted-lookup blinding generator is invalid or infinity");
    const auto identity_plus_h = zkdl::cpu_g1_add(zkdl::CpuG1Point::zero(), h);
    require(zkdl::cpu_g1_point_valid(identity_plus_h) &&
                zkdl::cpu_g1_equal(identity_plus_h, h),
            "native G1 identity plus valid H is incorrect");
    require(std::memcmp(&h, &h_before_identity, sizeof(h)) == 0,
            "native G1 identity addition mutated its valid H input");
    for (const auto& guarded : identity_guard)
        require(same(values.at(guarded.first), guarded.second),
                "native G1 identity addition mutated a toy key/active vector");

    const auto provider = [&](const std::string& name,
                              SegmentedTensorProvider& output) {
        const auto found = values.find(name);
        return found != values.end() && SegmentedTensorProvider::from_values(
            found->second, 1U, output, nullptr);
    };
    AuthenticatedSortedLookupProviderSet providers;
    std::string error;
    const auto require_provider = [&](const std::string& name,
                                      SegmentedTensorProvider& output) {
        require(provider(name, output), "tiny sorted-lookup provider setup failed");
    };
    require_provider("table_keys", providers.table_keys);
    require_provider("query_keys", providers.query_keys);
    require_provider("query_x", providers.query_x);
    require_provider("query_y", providers.query_y);
    require_provider("query_prefix", providers.query_prefix);
    require_provider("query_prefix_next", providers.query_prefix_next);
    require_provider("merge_keys", providers.merge_keys);
    require_provider("table_active", providers.table_active);
    require_provider("merge_active", providers.merge_active);
    require_provider("merge_delta", providers.merge_delta);
    require_provider("merge_marker", providers.merge_marker);
    require_provider("merge_inverse", providers.merge_inverse);
    require_provider("merge_prefix", providers.merge_prefix);
    require_provider("merge_prefix_next", providers.merge_prefix_next);
    require_provider("query_active", providers.query_active);
    require(make_authenticated_sorted_lookup_selector_provider(
                2U, 1U, 0U, providers.query_first, &error) &&
                make_authenticated_sorted_lookup_selector_provider(
                    2U, 1U, 1U, providers.query_last, &error) &&
                make_authenticated_sorted_lookup_selector_provider(
                    8U, 1U, 0U, providers.merge_first, &error) &&
                make_authenticated_sorted_lookup_selector_provider(
                    8U, 1U, 4U, providers.merge_last, &error) &&
                make_authenticated_sorted_lookup_shifted_key_provider(
                    providers.merge_keys, providers.merge_active,
                    providers.merge_previous_key, &error) &&
                make_authenticated_sorted_lookup_nonzero_provider(
                    providers.merge_delta, providers.merge_nonzero, &error),
            "tiny sorted-lookup derived providers failed");
    for (std::uint32_t bit = 0U; bit < 3U; ++bit) {
        SegmentedTensorProvider key_bit, delta_bit, query_bit;
        require(make_authenticated_sorted_lookup_bit_provider(
                    providers.merge_keys, 3U, bit, key_bit, &error) &&
                    make_authenticated_sorted_lookup_bit_provider(
                        providers.merge_delta, 3U, bit, delta_bit, &error) &&
                    make_authenticated_sorted_lookup_bit_provider(
                        providers.query_keys, 3U, bit, query_bit, &error),
                "tiny sorted-lookup bit provider failed");
        providers.merge_bits.push_back(std::move(key_bit));
        providers.merge_delta_bits.push_back(std::move(delta_bit));
        providers.query_bits.push_back(std::move(query_bit));
    }

    RelationStatement statement;
    statement.statement_digest = zkdl::sha256(std::vector<zkdl::byte>{'s'});
    statement.model_digest = zkdl::sha256(std::vector<zkdl::byte>{'m'});
    statement.table_digest = zkdl::sha256(std::vector<zkdl::byte>{'t'});
    statement.ledger_digest = zkdl::sha256(std::vector<zkdl::byte>{'l'});
    statement.registry_digest = zkdl::sha256(std::vector<zkdl::byte>{'r'});
    statement.srs_digest = zkdl::sha256(std::vector<zkdl::byte>{'p'});
    statement.public_output_digest = zkdl::sha256(std::vector<zkdl::byte>{'o'});
    zkdl::AuthenticatedSortedLookupDescriptor descriptor;
    descriptor.operation_id = expect_valid ? "tiny/sorted_lookup" : "tiny/nonmember_lookup";
    descriptor.table_ref = "tiny/table";
    descriptor.query_ref = "tiny/query";
    descriptor.merge_ref = "tiny/merge";
    descriptor.table_digest = statement.table_digest;
    descriptor.descriptor_digest = zkdl::sha256(std::vector<zkdl::byte>{'d'});
    descriptor.table_domain = 4U;
    descriptor.query_domain = 2U;
    descriptor.merge_domain = 8U;
    descriptor.table_active_length = 3U;
    descriptor.query_active_length = 2U;
    descriptor.merge_active_length = 5U;
    descriptor.segment_size = 1U;
    descriptor.table_segment_count = 4U;
    descriptor.query_segment_count = 2U;
    descriptor.merge_segment_count = 8U;
    descriptor.x_bits = 1U;
    descriptor.y_bits = 2U;
    descriptor.key_max = zkdl::fr_from_u32(4U);
    descriptor.table_unique_rows = 3U;

    const auto require_provider_state = [&](const char* stage) {
        std::string descriptor_error;
        if (!descriptor.valid(&descriptor_error))
            throw std::runtime_error(std::string("tiny sorted-lookup descriptor invalid at ") +
                                     stage + ": " + descriptor_error);
        const auto check_provider_readback = [&](const char* kind,
                                                  const SegmentedTensorProvider& provider_source,
                                                  const char* value_name) {
            std::vector<FrWire> actual;
            actual.reserve(provider_source.logical_domain());
            std::string read_error;
            for (std::uint32_t segment = 0U;
                 segment < provider_source.segment_count(); ++segment) {
                std::vector<FrWire> block;
                if (!provider_source.read_segment(segment, block, &read_error))
                    throw std::runtime_error(std::string("tiny sorted-lookup ") + kind +
                        " provider read failed at " + stage + ": " + read_error);
                const std::uint64_t offset =
                    static_cast<std::uint64_t>(segment) * provider_source.segment_size();
                const std::uint32_t expected_count = static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(provider_source.segment_size(),
                        provider_source.logical_domain() - offset));
                if (block.size() != provider_source.segment_size())
                    throw std::runtime_error(std::string("tiny sorted-lookup ") + kind +
                        " provider block shape changed at " + stage);
                actual.insert(actual.end(), block.begin(),
                              block.begin() + static_cast<std::ptrdiff_t>(expected_count));
            }
            if (!same(actual, values.at(value_name)))
                throw std::runtime_error(std::string("tiny sorted-lookup ") + kind +
                    " provider bytes differ from fixture at " + stage);
        };
        check_provider_readback("table_keys", providers.table_keys, "table_keys");
        check_provider_readback("table_active", providers.table_active, "table_active");
        check_provider_readback("query_keys", providers.query_keys, "query_keys");
        check_provider_readback("query_active", providers.query_active, "query_active");
        check_provider_readback("merge_keys", providers.merge_keys, "merge_keys");
        check_provider_readback("merge_active", providers.merge_active, "merge_active");
        std::string provider_error;
        const bool valid = providers.valid(descriptor, &provider_error);
        check_provider_readback("table_keys", providers.table_keys, "table_keys");
        check_provider_readback("table_active", providers.table_active, "table_active");
        check_provider_readback("query_keys", providers.query_keys, "query_keys");
        check_provider_readback("query_active", providers.query_active, "query_active");
        check_provider_readback("merge_keys", providers.merge_keys, "merge_keys");
        check_provider_readback("merge_active", providers.merge_active, "merge_active");
        if (!valid)
            throw std::runtime_error(std::string("tiny sorted-lookup providers invalid at ") +
                                     stage + ": " + provider_error);
    };
    require_provider_state("before outer commitments");

    std::vector<zkdl::AuthenticatedSortedLookupCommitmentRef> refs;
    std::vector<zkdl::CpuG1Point> outer_commitments;
    std::vector<FrWire> outer_blinders;
    std::vector<SegmentCommitmentMaterial> outer_materials;
    std::vector<CommitmentWire> outer_wires;
    const auto add_outer = [&](std::uint32_t role, std::uint32_t column,
                               const std::string& tag,
                               const SegmentedTensorProvider& source) {
        const std::uint32_t index = static_cast<std::uint32_t>(refs.size());
        const FrWire blinder = zkdl::fr_from_u32(101U + index);
        CommitmentWire wire;
        SegmentCommitmentMaterial material;
        if (!zktr_build_segment_commitment_material(
                "tiny/sorted_lookup/" + tag, 3U, source, generators, h, u,
                blinder, statement, wire, material, &error, nullptr, 1U))
            return false;
        zkdl::AuthenticatedSortedLookupCommitmentRef ref;
        ref.role = role;
        ref.column_index = column;
        ref.commitment_index = index;
        ref.logical_domain = source.logical_domain();
        ref.segment_size = source.segment_size();
        ref.segment_count = source.segment_count();
        refs.push_back(ref);
        outer_commitments.push_back(zkdl::cpu_g1_from_native(wire.commitment));
        outer_blinders.push_back(blinder);
        outer_materials.push_back(std::move(material));
        outer_wires.push_back(std::move(wire));
        return true;
    };
    const bool outer_commitments_built =
        add_outer(zkdl::kSortedLookupCommitmentQueryX, 0U, "query_x", providers.query_x) &&
                add_outer(zkdl::kSortedLookupCommitmentQueryY, 0U, "query_y", providers.query_y) &&
                add_outer(zkdl::kSortedLookupCommitmentQueryKey, 0U, "query_key", providers.query_keys) &&
                add_outer(zkdl::kSortedLookupCommitmentQueryPrefix, 0U, "query_prefix", providers.query_prefix) &&
                add_outer(zkdl::kSortedLookupCommitmentMergeKey, 0U, "merge_key", providers.merge_keys) &&
                add_outer(zkdl::kSortedLookupCommitmentMergeDelta, 0U, "merge_delta", providers.merge_delta) &&
                add_outer(zkdl::kSortedLookupCommitmentMergeMarker, 0U, "merge_marker", providers.merge_marker) &&
                add_outer(zkdl::kSortedLookupCommitmentMergeInverse, 0U, "merge_inverse", providers.merge_inverse) &&
                add_outer(zkdl::kSortedLookupCommitmentMergePrefix, 0U, "merge_prefix", providers.merge_prefix);
    if (!outer_commitments_built)
        throw std::runtime_error("tiny sorted-lookup outer commitment setup failed: " + error);
    for (std::uint32_t bit = 0U; bit < providers.merge_bits.size(); ++bit)
        require(add_outer(zkdl::kSortedLookupCommitmentMergeBit, bit,
                          "merge_bit_" + std::to_string(bit), providers.merge_bits[bit]),
                "tiny sorted-lookup merge-bit commitment setup failed");
    statement.segment_commitments_digest =
        relation_commitment_vector_digest(statement, outer_wires);
    require_provider_state("after outer commitments");

    zkdl::AuthenticatedSortedLookupProof proof;
    const bool proved = prove_authenticated_sorted_lookup(
        descriptor, providers, refs, outer_commitments, outer_blinders,
        outer_materials, generators, h, u, statement, proof, &error);
    if (!expect_valid) {
        if (proved) {
            const auto encoded = proof.serialize();
            zkdl::AuthenticatedSortedLookupProof decoded;
            if (!zkdl::AuthenticatedSortedLookupProof::deserialize(
                    encoded, decoded, &error))
                throw std::runtime_error("non-member sorted-lookup proof wire decode failed: " + error);
            if (verify_authenticated_sorted_lookup(
                    descriptor, providers.table_keys, providers.table_active,
                    refs, outer_commitments, outer_wires, generators, h, u,
                    statement, decoded, &error))
                throw std::runtime_error("sorted-lookup accepted a query key absent from the public table");
        }
        return;
    }
    if (!proved)
        throw std::runtime_error("tiny sorted-lookup proof failed: " + error);
    std::vector<zkdl::byte> encoded;
    try {
        encoded = proof.serialize();
    } catch (const std::exception& exception) {
        throw std::runtime_error(std::string("tiny sorted-lookup proof serialization failed: ") +
                                 exception.what());
    }
    zkdl::AuthenticatedSortedLookupProof decoded;
    if (!zkdl::AuthenticatedSortedLookupProof::deserialize(encoded, decoded, &error))
        throw std::runtime_error("tiny sorted-lookup proof wire decode failed: " + error);
    require(decoded.relations.size() == 5U &&
                decoded.relations[0].provider_tags.size() > 4U &&
                decoded.relations[0].provider_tags[4U] == "query_product_prefix" &&
                decoded.relations[0].outer_commitment_indices[4U] ==
                    std::numeric_limits<std::uint32_t>::max() &&
                decoded.relations[1].provider_tags[0U] == "merge_product_prefix" &&
                decoded.relations[1].outer_commitment_indices[0U] ==
                    std::numeric_limits<std::uint32_t>::max(),
            "challenge products were not represented by internal relation commitments");
    require(std::any_of(refs.begin(), refs.end(), [](const auto& ref) {
                return ref.role == zkdl::kSortedLookupCommitmentQueryPrefix;
            }) &&
                std::any_of(refs.begin(), refs.end(), [](const auto& ref) {
                    return ref.role == zkdl::kSortedLookupCommitmentMergePrefix;
                }),
            "original external count-prefix roles were omitted from the header");
    if (!verify_authenticated_sorted_lookup(
            descriptor, providers.table_keys, providers.table_active,
            refs, outer_commitments, outer_wires, generators, h, u,
            statement, decoded, &error))
        throw std::runtime_error("fresh sorted-lookup verifier rejected valid proof: " + error);

    const auto& query_nodes = decoded.relations[0U].terminal_relation.product_nodes;
    std::size_t square_node_index = query_nodes.size();
    bool has_legacy_product_roles = false;
    for (std::size_t index = 0U; index < query_nodes.size(); ++index) {
        const auto& node = query_nodes[index];
        if (node.right_commitment_index >= kZkTerminalProductAliasIndexBase &&
            node.right_commitment_index < kZkTerminalResidualIndex) {
            square_node_index = index;
            break;
        }
        if (node.left_commitment_index != node.right_commitment_index)
            has_legacy_product_roles = true;
    }
    std::uint32_t expected_square_alias = 0U;
    require(square_node_index < query_nodes.size() &&
                zktr_product_alias_index(square_node_index, expected_square_alias,
                                         &error) &&
                query_nodes[square_node_index].left_commitment_index ==
                    zktr_terminal_evaluation_index(3U) &&
                query_nodes[square_node_index].right_commitment_index ==
                    expected_square_alias &&
                query_nodes[square_node_index].product_proof.context.commitment_b_index ==
                    expected_square_alias &&
                zkdl::cpu_g1_equal(
                    zkdl::cpu_g1_from_native(
                        query_nodes[square_node_index].product_proof.commitment_a),
                    zkdl::cpu_g1_from_native(
                        query_nodes[square_node_index].product_proof.commitment_b)) &&
                has_legacy_product_roles,
            "sorted-lookup square alias is not source-bound or changed non-square roles");
    const auto require_verifier_rejects = [&](const auto& candidate,
                                              const char* message) {
        std::string verify_error;
        if (verify_authenticated_sorted_lookup(
                descriptor, providers.table_keys, providers.table_active,
                refs, outer_commitments, outer_wires, generators, h, u,
                statement, candidate, &verify_error))
            throw std::runtime_error(std::string(message) + ": proof accepted");
    };

    auto bad_alias_id = decoded;
    auto& bad_alias_node = bad_alias_id.relations[0U].terminal_relation
                               .product_nodes[square_node_index];
    std::uint32_t wrong_alias = kZkTerminalProductAliasIndexBase +
        zkdl::ZkTerminalRelationProof::kMaxProductNodes - 1U;
    if (wrong_alias == expected_square_alias)
        wrong_alias = kZkTerminalProductAliasIndexBase;
    bad_alias_node.right_commitment_index = wrong_alias;
    bad_alias_node.product_proof.context.commitment_b_index = wrong_alias;
    require_verifier_rejects(bad_alias_id,
                             "sorted-lookup accepted a wrong square alias ID");

    auto bad_alias_point = decoded;
    auto& bad_point_node = bad_alias_point.relations.at(0U).terminal_relation
                               .product_nodes.at(square_node_index);
    const auto alias_source =
        zkdl::cpu_g1_from_native(bad_point_node.product_proof.commitment_b);
    require(zkdl::cpu_g1_point_valid(alias_source) && !alias_source.is_zero() &&
                !zkdl::cpu_g1_equal(alias_source, h),
            "square terminal alias source point is invalid or aliases its replacement");
    const auto changed_alias = zkdl::cpu_g1_add(alias_source, h);
    require(zkdl::cpu_g1_point_valid(changed_alias),
            "native G1 addition rejected valid square alias point plus H");
    bad_point_node.product_proof.commitment_b = zkdl::cpu_g1_to_native(changed_alias);
    require_verifier_rejects(bad_alias_point,
                             "sorted-lookup accepted a changed square alias point");

    auto bad_square_response = decoded;
    auto& bad_response_node = bad_square_response.relations.at(0U).terminal_relation
                                  .product_nodes.at(square_node_index);
    bad_response_node.product_proof.z = zkdl::fr_value(zkdl::fr_add(
        zkdl::fr_wire(bad_response_node.product_proof.z),
        zkdl::fr_from_u32(1U)));
    require_verifier_rejects(bad_square_response,
                             "sorted-lookup accepted a tampered x-squared relation");

    auto bad_square_claim = decoded;
    auto& bad_claim_node = bad_square_claim.relations.at(0U).terminal_relation
                               .product_nodes.at(square_node_index);
    const auto original_output_claim = zkdl::cpu_g1_from_native(bad_claim_node.output_commitment);
    require(zkdl::cpu_g1_point_valid(original_output_claim) &&
                !zkdl::cpu_g1_equal(original_output_claim, h),
            "square terminal output claim is invalid or aliases the replacement point");
    bad_claim_node.output_commitment = zkdl::cpu_g1_to_native(h);
    require_verifier_rejects(bad_square_claim,
                             "sorted-lookup accepted a changed square output claim");

    auto tampered = decoded;
    std::size_t product_prefix_relation = tampered.relations.size();
    std::size_t product_prefix_tensor = 0U;
    for (std::size_t relation_index = 0U;
         relation_index < tampered.relations.size(); ++relation_index) {
        const auto& tags = tampered.relations.at(relation_index).provider_tags;
        const auto tag = std::find(tags.begin(), tags.end(), "merge_product_prefix");
        if (tag == tags.end()) continue;
        require(product_prefix_relation == tampered.relations.size(),
                "merge product-prefix tag is ambiguous across relations");
        product_prefix_relation = relation_index;
        product_prefix_tensor = static_cast<std::size_t>(tag - tags.begin());
        require(std::find(tag + 1, tags.end(), "merge_product_prefix") == tags.end(),
                "merge product-prefix tag is duplicated within a relation");
    }
    require(product_prefix_relation < tampered.relations.size(),
            "internal merge product-prefix provider tag is absent");
    auto& prefix_relation = tampered.relations.at(product_prefix_relation);
    require(product_prefix_tensor < prefix_relation.terminal_relation
                                             .terminal_segment_openings.size() &&
                product_prefix_tensor < prefix_relation.outer_commitment_indices.size() &&
                prefix_relation.outer_commitment_indices.at(product_prefix_tensor) ==
                    std::numeric_limits<std::uint32_t>::max(),
            "merge product-prefix tag does not map to an internal terminal provider");
    auto& prefix_segments = prefix_relation.terminal_relation
                                .terminal_segment_openings.at(product_prefix_tensor);
    require(!prefix_segments.empty() && prefix_segments.at(0U).segment_index == 0U,
            "internal merge product-prefix segment zero opening is absent");
    const auto original_prefix_xi =
        zkdl::cpu_g1_from_native(prefix_segments.at(0U).xi);
    require(zkdl::cpu_g1_point_valid(original_prefix_xi),
            "internal product-prefix terminal xi is not a valid G1 point");
    const auto changed = zkdl::cpu_g1_add(original_prefix_xi, h);
    require(zkdl::cpu_g1_point_valid(changed),
            "native G1 addition rejected valid product-prefix xi plus H");
    prefix_segments.at(0U).xi = zkdl::cpu_g1_to_native(changed);
    error.clear();
    if (verify_authenticated_sorted_lookup(
            descriptor, providers.table_keys, providers.table_active,
            refs, outer_commitments, outer_wires, generators, h, u,
            statement, tampered, &error))
        throw std::runtime_error("sorted-lookup accepted a tampered internal product-prefix opening");

    values["query_prefix"][0U] = zkdl::fr_from_u32(2U);
    error.clear();
    require(!prove_authenticated_sorted_lookup(
                descriptor, providers, refs, outer_commitments, outer_blinders,
                outer_materials, generators, h, u, statement, proof, &error),
            "sorted-lookup accepted mutated external query-count prefix material");
    require(error.find("segment commitment material differs from provider witness") !=
                std::string::npos,
            "query-count prefix mutation did not fail at authenticated material replay");
    values["query_prefix"][0U] = zkdl::fr_from_u32(1U);
    values["merge_prefix"][0U] = zkdl::fr_from_u32(2U);
    error.clear();
    require(!prove_authenticated_sorted_lookup(
                descriptor, providers, refs, outer_commitments, outer_blinders,
                outer_materials, generators, h, u, statement, proof, &error),
            "sorted-lookup accepted mutated external merge-count prefix material");
    require(error.find("segment commitment material differs from provider witness") !=
                std::string::npos,
            "merge-count prefix mutation did not fail at authenticated material replay");
    auto bad_header_count = descriptor;
    --bad_header_count.table_unique_rows;
    error.clear();
    require(!prove_authenticated_sorted_lookup(
                bad_header_count, providers, refs, outer_commitments, outer_blinders,
                outer_materials, generators, h, u, statement, proof, &error) &&
                error.find("key encoding/bounds are not canonical") != std::string::npos,
            "sorted-lookup accepted a header marker-count mismatch");
    values["merge_marker"][0U] = FrWire{};
    error.clear();
    require(!prove_authenticated_sorted_lookup(
                descriptor, providers, refs, outer_commitments, outer_blinders,
                outer_materials, generators, h, u, statement, proof, &error) &&
                error.find("segment commitment material differs from provider witness") != std::string::npos,
            "sorted-lookup accepted a marker-count vector that differs from its outer commitment");
    values["merge_marker"][0U] = zkdl::fr_from_u32(1U);
    values["merge_marker"][5U] = zkdl::fr_from_u32(1U);
    error.clear();
    require(!prove_authenticated_sorted_lookup(
                descriptor, providers, refs, outer_commitments, outer_blinders,
                outer_materials, generators, h, u, statement, proof, &error) &&
                error.find("sorted-lookup padded auxiliary row is nonzero") != std::string::npos,
            "sorted-lookup accepted a nonzero marker in a padded row");
}

}  // namespace

int main() {
    try {
        const std::vector<zkdl::CpuG1Point> proof_generators{
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(2U)),
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(3U)),
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(5U)),
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(13U)),
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(17U)),
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(29U)),
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(31U)),
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(37U))};
        const auto proof_h = zkdl::cpu_g1_mul(
            zkdl::cpu_g1_generator(), zkdl::fr_from_u32(19U));
        const auto proof_u = zkdl::cpu_g1_mul(
            zkdl::cpu_g1_generator(), zkdl::fr_from_u32(23U));
        check_square_product_alias(proof_generators, proof_h, proof_u);
        check_marker_hybrid_sumcheck(proof_generators, proof_h, proof_u);
        check_tiny_sorted_lookup({3U, 4U}, {0U, 3U, 3U, 4U, 4U}, true,
                                 proof_generators, proof_h, proof_u);
        check_tiny_sorted_lookup({4U, 3U}, {0U, 3U, 3U, 4U, 4U}, true,
                                 proof_generators, proof_h, proof_u);
        check_tiny_sorted_lookup({2U, 4U}, {0U, 2U, 3U, 4U, 4U}, false,
                                 proof_generators, proof_h, proof_u);

        constexpr std::uint32_t domain = 1024U;
        constexpr std::uint32_t segment_size = 8U;
        constexpr std::uint32_t key_bits = 5U;
        constexpr std::uint32_t active_length = 13U;
        const FrWire beta = zkdl::fr_from_u32(2U);
        const FrWire gamma = zkdl::fr_neg(zkdl::fr_from_u32(14U));
        std::vector<FrWire> keys(domain, FrWire{}), active(domain, FrWire{});
        const std::uint32_t active_keys[] = {1U, 7U, 7U, 11U, 15U, 17U, 18U,
                                             19U, 20U, 21U, 22U, 23U, 24U};
        for (std::uint32_t index = 0U; index < active_length; ++index) {
            keys[index] = zkdl::fr_from_u32(active_keys[index]);
            active[index] = zkdl::fr_from_u32(1U);
        }
        const std::vector<FrWire> expected = oracle(keys, active, beta, gamma);
        const auto digest = zkdl::sha256(std::vector<zkdl::byte>{'p'});

        auto new_counts = std::make_shared<ReadCounts>();
        const auto new_keys = counted_provider(keys, segment_size, new_counts, true);
        const auto new_active = counted_provider(active, segment_size, new_counts, false);
        SegmentedTensorProvider cached_prefix;
        std::string error;
        require(make_authenticated_sorted_lookup_prefix_product_provider(
                    new_keys, new_active, beta, gamma, cached_prefix, &error,
                    nullptr, digest, key_bits),
                "could not create cached prefix provider");

        auto old_counts = std::make_shared<ReadCounts>();
        const auto old_keys = counted_provider(keys, segment_size, old_counts, true);
        const auto old_active = counted_provider(active, segment_size, old_counts, false);
        const SegmentedTensorProvider legacy_provider(
            domain, segment_size,
            [old_keys, old_active, beta, gamma](std::uint64_t offset,
                                                std::uint32_t count,
                                                std::vector<FrWire>& output,
                                                std::string* read_error) {
                output = legacy_prefix(old_keys, old_active, beta, gamma,
                                       offset, count, read_error);
                return output.size() == count;
            });

        const auto old_start = std::chrono::steady_clock::now();
        std::vector<FrWire> legacy_output;
        for (std::uint32_t segment = 0U; segment < legacy_provider.segment_count(); ++segment) {
            std::vector<FrWire> block;
            require(legacy_provider.read_segment(segment, block, &error),
                    "legacy full-prefix oracle read failed");
            legacy_output.insert(legacy_output.end(), block.begin(), block.end());
        }
        const auto old_elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - old_start).count();

        const auto new_start = std::chrono::steady_clock::now();
        std::vector<FrWire> cached_output;
        for (std::uint32_t segment = 0U; segment < cached_prefix.segment_count(); ++segment) {
            std::vector<FrWire> block;
            require(cached_prefix.read_segment(segment, block, &error),
                    "cached prefix segment read failed");
            cached_output.insert(cached_output.end(), block.begin(), block.end());
        }
        const auto new_elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - new_start).count();
        require(same(legacy_output, expected) && same(cached_output, expected),
                "legacy/cached prefix values differ from independent full-vector oracle");

        const std::uint64_t source_reads_before_ranges =
            new_counts->key_segments + new_counts->active_segments;
        std::vector<FrWire> range, singleton, cross_segment;
        require(sorted_lookup_read_range(cached_prefix, 5U, 14U, range, &error) &&
                    sorted_lookup_read_range(cached_prefix, 1023U, 1U, singleton, &error) &&
                    sorted_lookup_read_range(cached_prefix, 7U, 3U, cross_segment, &error),
                "cached prefix partial/cross-segment range read failed");
        require(std::equal(range.begin(), range.end(), expected.begin() + 5) &&
                    singleton.size() == 1U && singleton.front() == expected.back() &&
                    std::equal(cross_segment.begin(), cross_segment.end(), expected.begin() + 7),
                "cached prefix partial/cross-segment ranges differ from oracle");
        const std::uint64_t old_source_reads =
            old_counts->key_segments + old_counts->active_segments;
        const std::uint64_t new_source_reads =
            new_counts->key_segments + new_counts->active_segments;
        require(old_source_reads == 2ULL * 128ULL * 128ULL &&
                    source_reads_before_ranges == 4ULL * 128ULL &&
                    old_source_reads / source_reads_before_ranges == 64ULL,
                "cached prefix source-read count did not remove repeated full scans");

        auto invalid_keys = keys;
        invalid_keys[2U] = zkdl::fr_from_u32(32U);
        check_negative(invalid_keys, active, beta, gamma,
                       "sorted-lookup packed key exceeds declared bit width");
        auto noncanonical_keys = keys;
        noncanonical_keys[3U].fill(0xffffffffU);
        check_negative(noncanonical_keys, active, beta, gamma,
                       "segmented tensor reader returned a non-canonical field value");
        auto invalid_active = active;
        invalid_active[1U] = zkdl::fr_from_u32(2U);
        check_negative(keys, invalid_active, beta, gamma,
                       "sorted-lookup prefix active mask is not boolean");
        auto invalid_padding = active;
        auto nonzero_padding_keys = keys;
        nonzero_padding_keys[20U] = zkdl::fr_from_u32(1U);
        check_negative(nonzero_padding_keys, invalid_padding, beta, gamma,
                       "sorted-lookup prefix inactive key is nonzero");

        std::cout << "sorted_lookup_prefix_cache=PASS domain=" << domain
                  << " segment_size=" << segment_size
                  << " active_rows=" << active_length
                  << " duplicate_keys=2 zero_factor=1"
                  << " old_source_segment_reads=" << old_source_reads
                  << " cached_source_segment_reads=" << source_reads_before_ranges
                  << " source_read_reduction="
                  << old_source_reads / source_reads_before_ranges << "x"
                  << " old_ms=" << old_elapsed << " cached_ms=" << new_elapsed
                  << " differential_values=" << domain
                  << " sorted_proof=valid,permuted,nonmember-rejected"
                  << " negative_cases=4 sticky_failure=PASS\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "sorted_lookup_prefix_cache=FAIL error=" << exception.what() << '\n';
        return 1;
    }
}
