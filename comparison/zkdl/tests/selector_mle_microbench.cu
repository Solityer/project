#include "../zkrelation-proof.cuh"
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace zkdl_gat::relation;
static void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
static std::vector<FrWire> legacy_route_selector(const OperationDescriptor& descriptor, const std::vector<FrWire>& point, bool destination) {
    const auto edge_domain = descriptor.selector_edge_domain;
    const auto node_domain = descriptor.selector_node_domain;
    if (edge_domain != (std::uint32_t(1U) << point.size()) || descriptor.selector_src.size() != edge_domain) throw std::invalid_argument("legacy route selector shape mismatch");
    std::vector<FrWire> selector(node_domain);
    const FrWire one = zkdl::fr_from_u32(1U), zero{};
    for (std::uint32_t node = 0U; node < node_domain; ++node) {
        std::vector<FrWire> indicator(edge_domain, zero);
        for (std::uint32_t edge = 0U; edge < edge_domain; ++edge) if (descriptor.selector_active[edge] && (destination ? descriptor.selector_dst[edge] : descriptor.selector_src[edge]) == node) indicator[edge] = one;
        for (const auto& challenge : point) fold_values(indicator, challenge);
        selector[node] = indicator.front();
    }
    return selector;
}
static std::vector<FrWire> legacy_feature_selector(const OperationDescriptor& descriptor, const std::vector<FrWire>& point) {
    const auto output_domain = descriptor.selector_feature_domain, source_domain = descriptor.selector_source_feature_domain;
    if (output_domain != (std::uint32_t(1U) << point.size()) || descriptor.selector_feature_map.size() != output_domain) throw std::invalid_argument("legacy feature selector shape mismatch");
    const FrWire one = zkdl::fr_from_u32(1U), zero{};
    std::vector<FrWire> selector(source_domain, zero);
    for (std::uint32_t source = 0U; source < source_domain; ++source) {
        std::vector<FrWire> indicator(output_domain, zero);
        for (std::uint32_t output = 0U; output < output_domain; ++output) if (descriptor.selector_feature_map[output] == source) indicator[output] = one;
        for (const auto& challenge : point) fold_values(indicator, challenge);
        selector[source] = indicator.front();
    }
    return selector;
}
static std::vector<FrWire> legacy_sparse_selector(const OperationDescriptor& descriptor, const std::vector<FrWire>& edge_point, const std::vector<FrWire>& feature_point) {
    const auto outer = legacy_route_selector(descriptor, edge_point, descriptor.route_by_destination), feature = legacy_feature_selector(descriptor, feature_point);
    const auto source_size = std::size_t(descriptor.selector_node_domain) * descriptor.selector_source_feature_domain;
    std::vector<FrWire> selector(std::max<std::size_t>(2U, source_size));
    for (std::size_t node = 0U; node < outer.size(); ++node) for (std::size_t column = 0U; column < feature.size(); ++column) selector[node * feature.size() + column] = zkdl::fr_mul(outer[node], feature[column]);
    return selector;
}
static OperationDescriptor make_descriptor(std::uint32_t edges, std::uint32_t nodes, std::uint32_t output_features, std::uint32_t source_features) {
    OperationDescriptor descriptor;
    descriptor.selector_edge_domain = edges;
    descriptor.selector_node_domain = nodes;
    descriptor.selector_feature_domain = output_features;
    descriptor.selector_source_feature_domain = source_features;
    descriptor.selector_src.resize(edges);
    descriptor.selector_dst.resize(edges);
    descriptor.selector_active.resize(edges);
    for (std::uint32_t edge = 0U; edge < edges; ++edge) {
        const bool active = edge % 13U != 0U;
        descriptor.selector_active[edge] = active ? 1U : 0U;
        descriptor.selector_src[edge] = active ? (edge * 17U + edge / 3U) % nodes : 0U;
        descriptor.selector_dst[edge] = active ? (edge * 29U + edge / 5U) % nodes : 0U;
    }
    descriptor.selector_feature_map.resize(output_features);
    for (std::uint32_t feature = 0U; feature < output_features; ++feature) descriptor.selector_feature_map[feature] = (feature * 7U + feature / 3U) % source_features;
    return descriptor;
}
static std::uint32_t point_bits(std::uint32_t domain) { std::uint32_t bits = 0U; while (domain > 1U) { domain >>= 1U; ++bits; } return bits; }
static FrWire challenge(std::uint32_t index) {
    if (index % 7U == 0U) return FrWire{};
    if (index % 7U == 1U) return zkdl::fr_from_u32(1U);
    if (index % 7U == 2U) { FrWire value{}; value[0] = 0xabcdef01U; value[4] = 0x12345678U; value[7] = 2U; return value; }
    return zkdl::fr_from_u32(index * 17U + 3U);
}
static bool same_bytes(const std::vector<FrWire>& left, const std::vector<FrWire>& right) { return left.size() == right.size() && (left.empty() || std::memcmp(left.data(), right.data(), left.size() * sizeof(FrWire)) == 0); }
static std::vector<FrWire> read_provider(const SegmentedTensorProvider& provider) {
    std::vector<FrWire> values;
    for (std::uint32_t segment = 0U; segment < provider.segment_count(); ++segment) {
        std::vector<FrWire> block;
        require(provider.read_segment(segment, block), "virtual provider segment read failed");
        values.insert(values.end(), block.begin(), block.end());
    }
    return values;
}
static void run_validated_materialize_tests() {
    std::vector<FrWire> values(8U);
    for (std::uint32_t i = 0U; i < values.size(); ++i) values[i] = zkdl::fr_from_u32(i + 1U);
    SegmentedTensorProvider provider;
    require(SegmentedTensorProvider::from_values(values, 4U, provider), "test source provider construction failed");
    TensorWitness stored;
    stored.id = "source";
    stored.visibility = 3U;
    stored.domain_size = static_cast<std::uint32_t>(values.size());
    stored.stored_provider = provider;
    std::vector<FrWire> materialized;
    require(stored.metadata_valid() && stored.materialize_validated(materialized), "stored source validation/materialization failed");
    require(same_bytes(values, materialized), "stored source materialization changed values");
    values[3U] = zkdl::kScalarModulus;
    SegmentedTensorProvider invalid_provider;
    require(SegmentedTensorProvider::from_values(values, 4U, invalid_provider), "invalid-value test provider construction failed");
    stored.stored_provider = invalid_provider;
    require(!stored.materialize_validated(materialized), "noncanonical stored source was accepted");
    stored.stored_provider = SegmentedTensorProvider{};
    stored.values = values;
    require(!stored.materialize_validated(materialized), "noncanonical dense source was accepted");
}
static double elapsed_ms(const std::vector<FrWire>& value, const std::chrono::steady_clock::time_point& start) { volatile std::uint32_t sink = value.empty() ? 0U : value.back()[0]; (void)sink; return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); }
static std::uint32_t next_word(std::uint32_t& state) { state ^= state << 13U; state ^= state >> 17U; state ^= state << 5U; return state; }
static void check_folded_state_differential(std::uint32_t domain, std::size_t basis_size) {
    ConstraintDescriptor descriptor;
    descriptor.id = "fold-state-differential";
    descriptor.degree = 2U;
    descriptor.terms = {Monomial{zkdl::fr_from_u32(1U), {0U, 1U}}, Monomial{zkdl::fr_neg(zkdl::fr_from_u32(1U)), {2U, 3U}}};
    std::vector<std::vector<FrWire>> values(4U, std::vector<FrWire>(domain));
    for (std::size_t tensor = 0U; tensor < values.size(); ++tensor)
        for (std::uint32_t index = 0U; index < domain; ++index)
            values[tensor][index] = challenge(static_cast<std::uint32_t>(tensor * 11U + index + 2U));
    std::vector<SegmentedTensorProvider> providers;
    for (std::size_t tensor_index = 0U; tensor_index < values.size(); ++tensor_index) {
        SegmentedTensorProvider provider;
        require(SegmentedTensorProvider::from_values(values[tensor_index], basis_size, provider), "fold-state input provider construction failed");
        providers.push_back(std::move(provider));
    }
    const std::uint32_t rounds = point_bits(domain);
    std::vector<FrWire> prior;
    std::vector<std::vector<FrWire>> state;
    const std::vector<FrWire> fixed_challenges{challenge(0U), challenge(1U), challenge(2U), challenge(3U), challenge(4U)};
    for (std::uint32_t round = 0U; round < rounds; ++round) {
        std::vector<FrWire> old_evaluations, current_evaluations;
        require(zksc_segmented_round_evaluations(descriptor, providers, 2U, round, prior, old_evaluations), "old segmented round evaluation failed");
        if (round != 0U) {
            require(zksc_folded_round_evaluations(descriptor, state, 2U, current_evaluations), "cached folded round evaluation failed");
            require(same_bytes(old_evaluations, current_evaluations), "cached round evaluations differ from segmented oracle");
        }
        const FrWire next = fixed_challenges[round];
        if (round == 0U) require(zksc_segmented_fold_state_initialize(providers, next, state), "cached fold-state initialization failed");
        else for (auto& tensor : state) require(zksc_fold_values_in_place(tensor, next), "cached in-place fold failed");
        prior.push_back(next);
    }
    auto oracle_values = values;
    for (std::size_t tensor = 0U; tensor < oracle_values.size(); ++tensor)
        for (const FrWire& value : prior)
            fold_values(oracle_values[tensor], value);
    std::vector<FrWire> old_terminal;
    require(zksc_segmented_terminal_values(providers, prior, old_terminal), "old segmented terminal evaluation failed");
    require(state.size() == old_terminal.size(), "cached terminal tensor count mismatch");
    for (std::size_t tensor = 0U; tensor < state.size(); ++tensor) {
        require(oracle_values[tensor].size() == 1U, "full-vector terminal oracle did not collapse to one value");
        require(old_terminal[tensor] == oracle_values[tensor].front(), "segmented terminal differs from independent full-vector oracle");
        require(state[tensor].size() == 1U && state[tensor].front() == oracle_values[tensor].front(), "cached terminal differs from independent full-vector oracle");
    }
    auto malformed = providers;
    const auto segment_size = malformed.front().segment_size();
    malformed[0] = SegmentedTensorProvider(
        domain, segment_size,
        [](std::uint64_t, std::uint32_t count, std::vector<FrWire>& output, std::string*) {
            output.assign(count == 0U ? 0U : count - 1U, FrWire{});
            return true;
        });
    std::string error;
    require(!zksc_segmented_fold_state_initialize(malformed, challenge(1U), state, &error), "malformed short provider was accepted by fold-state initialization");
}
static void check_large_segmented_terminal_oracle() {
    constexpr std::uint32_t domain = 1U << 15U;
    constexpr std::size_t basis_size = 1U << 8U;
    constexpr std::size_t tensor_count = 4U;
    std::vector<std::vector<FrWire>> values(tensor_count, std::vector<FrWire>(domain));
    for (std::size_t tensor = 0U; tensor < tensor_count; ++tensor)
        for (std::uint32_t index = 0U; index < domain; ++index)
            values[tensor][index] = challenge(static_cast<std::uint32_t>(tensor * 37U + index + 5U));
    std::vector<SegmentedTensorProvider> providers;
    providers.reserve(tensor_count);
    for (const auto& tensor : values) {
        SegmentedTensorProvider provider;
        require(SegmentedTensorProvider::from_values(tensor, basis_size, provider), "large terminal provider construction failed");
        providers.push_back(std::move(provider));
    }
    std::vector<FrWire> challenges;
    challenges.reserve(point_bits(domain));
    for (std::uint32_t round = 0U; round < point_bits(domain); ++round)
        challenges.push_back(challenge(round + 19U));
    auto oracle_values = values;
    std::vector<std::vector<FrWire>> cache = values;
    for (std::size_t tensor = 0U; tensor < tensor_count; ++tensor)
        for (std::size_t round = 0U; round < challenges.size(); ++round) {
            fold_values(oracle_values[tensor], challenges[round]);
            if (round == 0U) require(zksc_fold_values_in_place(cache[tensor], challenges[round]), "large terminal cache initialization failed");
            else require(zksc_fold_values_in_place(cache[tensor], challenges[round]), "large terminal cache fold failed");
        }
    std::vector<FrWire> terminal;
    require(zksc_segmented_terminal_values(providers, challenges, terminal), "large segmented terminal evaluation failed");
    require(terminal.size() == tensor_count, "large segmented terminal tensor count mismatch");
    for (std::size_t tensor = 0U; tensor < tensor_count; ++tensor) {
        require(oracle_values[tensor].size() == 1U && cache[tensor].size() == 1U, "large full-vector terminal oracle did not collapse to one value");
        require(terminal[tensor] == oracle_values[tensor].front(), "large segmented terminal differs from independent full-vector oracle");
        require(cache[tensor].front() == oracle_values[tensor].front(), "large cached terminal differs from independent full-vector oracle");
    }
}
static void check_folded_state_benchmark() {
    constexpr std::uint32_t domain = 1U << 15U;
    constexpr std::size_t basis_size = 1U << 10U;
    ConstraintDescriptor descriptor;
    descriptor.id = "fold-state-microbench";
    descriptor.degree = 2U;
    descriptor.terms = {Monomial{zkdl::fr_from_u32(1U), {0U, 1U}}, Monomial{zkdl::fr_neg(zkdl::fr_from_u32(1U)), {2U, 3U}}};
    std::vector<std::vector<FrWire>> values(4U, std::vector<FrWire>(domain));
    std::uint32_t seed = 0x2f6e2b1dU;
    for (auto& tensor : values)
        for (auto& value : tensor) value = zkdl::fr_from_u32(next_word(seed));
    std::vector<SegmentedTensorProvider> providers;
    for (std::size_t tensor_index = 0U; tensor_index < values.size(); ++tensor_index) {
        SegmentedTensorProvider provider;
        require(SegmentedTensorProvider::from_values(values[tensor_index], basis_size, provider), "fold-state benchmark provider construction failed");
        providers.push_back(std::move(provider));
    }
    std::vector<FrWire> challenges;
    for (std::uint32_t round = 0U; round < point_bits(domain); ++round)
        challenges.push_back(challenge(round));

    std::vector<std::vector<FrWire>> old_rounds;
    std::vector<FrWire> old_terminal;
    const auto old_start = std::chrono::steady_clock::now();
    std::vector<FrWire> prior;
    for (std::uint32_t round = 0U; round < challenges.size(); ++round) {
        std::vector<FrWire> evaluations;
        require(zksc_segmented_round_evaluations(descriptor, providers, 2U, round, prior, evaluations), "old fold-state benchmark round failed");
        old_rounds.push_back(std::move(evaluations));
        prior.push_back(challenges[round]);
    }
    require(zksc_segmented_terminal_values(providers, prior, old_terminal), "old fold-state benchmark terminal failed");
    const double old_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - old_start).count();

    std::vector<std::vector<FrWire>> current_rounds;
    std::vector<std::vector<FrWire>> state;
    const auto current_start = std::chrono::steady_clock::now();
    for (std::uint32_t round = 0U; round < challenges.size(); ++round) {
        std::vector<FrWire> evaluations;
        if (round == 0U)
            require(zksc_segmented_round_evaluations(descriptor, providers, 2U, round, {}, evaluations), "cached fold-state benchmark initial round failed");
        else
            require(zksc_folded_round_evaluations(descriptor, state, 2U, evaluations), "cached fold-state benchmark round failed");
        current_rounds.push_back(std::move(evaluations));
        if (round == 0U)
            require(zksc_segmented_fold_state_initialize(providers, challenges[round], state), "cached fold-state benchmark initialization failed");
        else
            for (auto& tensor : state)
                require(zksc_fold_values_in_place(tensor, challenges[round]), "cached fold-state benchmark in-place fold failed");
    }
    const double current_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - current_start).count();
    require(old_rounds.size() == current_rounds.size(), "fold-state benchmark round count mismatch");
    for (std::size_t round = 0U; round < old_rounds.size(); ++round)
        require(same_bytes(old_rounds[round], current_rounds[round]), "fold-state benchmark round output mismatch");
    require(state.size() == old_terminal.size(), "fold-state benchmark terminal count mismatch");
    for (std::size_t tensor = 0U; tensor < state.size(); ++tensor)
        require(state[tensor].size() == 1U && state[tensor].front() == old_terminal[tensor], "fold-state benchmark terminal output mismatch");
    std::cout << "fold_state_case status=PASS domain=" << domain << " tensors=4 segment_size=" << basis_size
              << " rounds=" << challenges.size() << " old_ms=" << old_ms << " cached_ms=" << current_ms
              << " speedup=" << old_ms / current_ms << '\n';

    SegmentedTensorProvider singleton;
    const std::vector<FrWire> singleton_values{zkdl::fr_from_u32(7U)};
    require(SegmentedTensorProvider::from_values(singleton_values, 1U, singleton), "domain-one provider construction failed");
    std::vector<std::vector<FrWire>> singleton_state;
    std::string error;
    require(!zksc_segmented_fold_state_initialize({singleton}, challenge(1U), singleton_state, &error), "fold-state initializer unexpectedly accepted a zero-round domain");
}
static void run_scalar_differential_and_benchmark() {
    std::vector<FrWire> values(8U);
    values[0] = FrWire{};
    values[1] = zkdl::fr_from_u32(1U);
    values[2] = zkdl::fr_sub_raw(zkdl::kScalarModulus, zkdl::fr_from_u32(1U));
    values[3] = challenge(2U);
    values[4] = zkdl::kScalarModulus;
    values[5] = zkdl::kScalarModulus;
    ++values[5][0];
    values[6].fill(0xffffffffU);
    std::uint32_t state = 0x91e10da5U;
    for (std::uint32_t sample = 0U; sample < 32U; ++sample) {
        FrWire value{};
        for (auto& limb : value) limb = next_word(state);
        if (!zkdl::fr_canonical(value)) value = zkdl::fr_sub_raw(value, zkdl::kScalarModulus);
        values[7] = value;
        for (std::size_t left = 0U; left < values.size(); ++left)
            for (std::size_t right = 0U; right < values.size(); ++right)
                require(zkdl::fr_mul_limb_fallback(values[left], values[right]) == zkdl::fr_mul(values[left], values[right]), "BLST scalar multiplication differential mismatch");
    }
    constexpr std::size_t count = 8192U;
    constexpr std::size_t repetitions = 4U;
    std::vector<FrWire> left(count), right(count);
    state = 0x6d2b79f5U;
    for (std::size_t i = 0U; i < count; ++i) {
        for (auto& limb : left[i]) limb = next_word(state);
        for (auto& limb : right[i]) limb = next_word(state);
        if (!zkdl::fr_canonical(left[i])) left[i] = zkdl::fr_sub_raw(left[i], zkdl::kScalarModulus);
        if (!zkdl::fr_canonical(right[i])) right[i] = zkdl::fr_sub_raw(right[i], zkdl::kScalarModulus);
    }
    std::uint32_t old_checksum = 0U, current_checksum = 0U;
    const auto old_begin = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0U; repeat < repetitions; ++repeat)
        for (std::size_t i = 0U; i < count; ++i) {
            const auto product = zkdl::fr_mul_limb_fallback(left[i], right[i]);
            old_checksum = old_checksum * 1664525U + product[0] + product[4];
        }
    const double old_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - old_begin).count();
    const auto current_begin = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0U; repeat < repetitions; ++repeat)
        for (std::size_t i = 0U; i < count; ++i) {
            const auto product = zkdl::fr_mul(left[i], right[i]);
            current_checksum = current_checksum * 1664525U + product[0] + product[4];
        }
    const double current_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - current_begin).count();
    require(old_checksum == current_checksum, "BLST scalar benchmark checksums differ");
    std::cout << "scalar_case status=PASS canonical_pairs=" << count * repetitions << " old_ms=" << old_ms << " current_ms=" << current_ms << " speedup=" << old_ms / current_ms << " checksum=" << current_checksum << '\n';
}
static void check_cpu_g1_msm_benchmark() {
    constexpr std::size_t count = 1U << 18U;
    std::vector<zkdl::CpuG1Point> points;
    std::vector<FrWire> scalars;
    std::uint32_t seed = 0x7f4a7c15U;
    const auto setup_start = std::chrono::steady_clock::now();
    const auto generator = zkdl::cpu_g1_generator();
    points.assign(count, generator);
    scalars.resize(count);
    FrWire scalar_sum{};
    for (auto& scalar : scalars) {
        scalar = zkdl::fr_from_u32(next_word(seed));
        scalar_sum = zkdl::fr_add(scalar_sum, scalar);
    }
    const double setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - setup_start).count();
    const auto msm_start = std::chrono::steady_clock::now();
    const auto result = zkdl::cpu_g1_sum(points, scalars);
    const double msm_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - msm_start).count();
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto repeated_result = zkdl::cpu_g1_sum(points, scalars);
    const double repeat_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - repeat_start).count();
    const auto expected = zkdl::cpu_g1_mul(generator, scalar_sum);
    require(zkdl::cpu_g1_equal(result, expected), "CPU G1 MSM synthetic result mismatch");
    require(zkdl::cpu_g1_equal(repeated_result, expected), "CPU G1 MSM repeated result mismatch");
    std::cout << "cpu_g1_msm_case status=PASS points=" << count << " point_kind=repeated_standard_generator"
              << " setup_ms=" << setup_ms << " first_call_conversion_and_pippenger_ms=" << msm_ms
              << " same_input_repeat_conversion_and_pippenger_ms=" << repeat_ms
              << " workspace_bound_bytes=" << zkdl::cpu_g1_msm_workspace_bound_bytes() << '\n';
}
static void run_case(std::uint32_t edges, std::uint32_t nodes, std::uint32_t output_features, std::uint32_t source_features) {
    auto descriptor = make_descriptor(edges, nodes, output_features, source_features);
    std::vector<FrWire> edge_point(point_bits(edges)), feature_point(point_bits(output_features));
    for (std::uint32_t i = 0U; i < edge_point.size(); ++i) edge_point[i] = challenge(i + 1U);
    for (std::uint32_t i = 0U; i < feature_point.size(); ++i) feature_point[i] = challenge(i + 2U);
    for (const bool destination : {false, true}) {
        const auto expected = legacy_route_selector(descriptor, edge_point, destination);
        require(same_bytes(expected, public_route_selector_mle(descriptor, edge_point, destination)), "route selector differential mismatch");
        const auto max_test_bits = std::min<std::uint32_t>(3U, static_cast<std::uint32_t>(edge_point.size()));
        for (std::uint32_t block_bits = 0U; block_bits <= max_test_bits; ++block_bits)
            require(same_bytes(expected, public_route_selector_mle_blocked(descriptor, edge_point, destination, block_bits)), "blocked route selector differential mismatch");
    }
    const auto expected_feature = legacy_feature_selector(descriptor, feature_point);
    require(same_bytes(expected_feature, public_route_feature_selector_mle(descriptor, feature_point)), "feature selector differential mismatch");
    for (std::uint32_t block_bits = 0U; block_bits <= std::min<std::uint32_t>(3U, static_cast<std::uint32_t>(feature_point.size())); ++block_bits)
        require(same_bytes(expected_feature, public_route_feature_selector_mle(descriptor, feature_point, block_bits)), "blocked feature selector differential mismatch");
    for (const bool destination : {false, true}) {
        descriptor.route_by_destination = destination;
        require(same_bytes(legacy_sparse_selector(descriptor, edge_point, feature_point), sparse_route_public_selector(descriptor, edge_point, feature_point)), "flattened route selector differential mismatch");
        const auto outer = public_route_selector_mle(descriptor, edge_point, destination);
        const auto feature = public_route_feature_selector_mle(descriptor, feature_point);
        std::vector<FrWire> source_values(std::max<std::size_t>(2U, std::size_t(nodes) * source_features));
        for (std::size_t i = 0U; i < source_values.size(); ++i) source_values[i] = challenge(static_cast<std::uint32_t>(i + 19U));
        const auto output_value = challenge(113U);
        std::vector<SegmentedTensorProvider> providers;
        const std::uint32_t source_domain = static_cast<std::uint32_t>(source_values.size());
        const std::size_t basis_size = std::min<std::size_t>(8U, source_values.size());
        require(sparse_route_sumcheck_providers(source_values, outer, feature, output_value, source_domain, basis_size, providers, nullptr), "virtual route provider construction failed");
        require(providers.size() == 4U, "virtual route provider count mismatch");
        std::vector<FrWire> expected_selector = legacy_sparse_selector(descriptor, edge_point, feature_point);
        std::vector<FrWire> expected_zero(source_domain);
        expected_zero[0] = zkdl::fr_from_u32(1U);
        std::vector<FrWire> expected_output(source_domain, output_value);
        require(same_bytes(expected_selector, read_provider(providers[0])), "virtual selector provider values/order mismatch");
        require(same_bytes(source_values, read_provider(providers[1])), "virtual source provider values/order mismatch");
        require(same_bytes(expected_zero, read_provider(providers[2])), "virtual zero provider values/order mismatch");
        require(same_bytes(expected_output, read_provider(providers[3])), "virtual constant output provider values/order mismatch");
        std::vector<FrWire> invalid_block;
        require(!providers[0].read_segment(providers[0].segment_count(), invalid_block), "virtual provider out-of-range read was accepted");
    }
    descriptor.route_by_destination = false;
    const auto begin_old = std::chrono::steady_clock::now();
    const auto old_result = legacy_sparse_selector(descriptor, edge_point, feature_point);
    const double old_ms = elapsed_ms(old_result, begin_old);
    const auto begin_current = std::chrono::steady_clock::now();
    const auto current_result = sparse_route_public_selector(descriptor, edge_point, feature_point);
    const double current_ms = elapsed_ms(current_result, begin_current);
    require(same_bytes(old_result, current_result), "timed route selector outputs differ");
    std::cout << "selector_case status=PASS nodes=" << nodes << " edges=" << edges << " output_features=" << output_features << " source_features=" << source_features << " old_ms=" << old_ms << " current_ms=" << current_ms << " speedup=" << old_ms / current_ms << '\n';
}
int main() {
    try {
#if defined(ZKDL_FOLDED_STATE_ONLY)
        check_folded_state_differential(2U, 4U);
        check_folded_state_differential(32U, 8U);
        check_folded_state_differential(32U, 2U);
        check_large_segmented_terminal_oracle();
        std::cout << "folded_state_differential=PASS synthetic=1 full_datasets=0\n";
#else
        run_validated_materialize_tests();
        check_folded_state_differential(2U, 4U);
        check_folded_state_differential(32U, 8U);
        check_folded_state_differential(32U, 2U);
        check_folded_state_benchmark();
        run_scalar_differential_and_benchmark();
        check_cpu_g1_msm_benchmark();
        run_case(16U, 8U, 8U, 4U);
        run_case(256U, 16U, 32U, 16U);
        run_case(1024U, 64U, 64U, 32U);
        auto singleton = make_descriptor(8U, 1U, 1U, 1U);
        singleton.selector_active.assign(8U, 1U);
        singleton.selector_src.assign(8U, 0U);
        singleton.selector_dst.assign(8U, 0U);
        singleton.selector_feature_map.assign(1U, 0U);
        singleton.route_by_destination = false;
        const std::vector<FrWire> edge_point{challenge(1U), challenge(2U), challenge(3U)}, feature_point{};
        require(same_bytes(legacy_sparse_selector(singleton, edge_point, feature_point), sparse_route_public_selector(singleton, edge_point, feature_point)), "singleton padded selector differential mismatch");
        require(sparse_route_public_selector(singleton, edge_point, feature_point).size() == 2U, "singleton selector padded output size mismatch");
        bool rejected = false;
        singleton.selector_active.pop_back();
        try { (void)public_route_selector_mle(singleton, edge_point, false); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "malformed route selector was not rejected");
        std::cout << "selector_mle_microbench=PASS synthetic=1 full_datasets=0\n";
#endif
        return 0;
    } catch (const std::exception& error) { std::cerr << "selector_mle_microbench=FAIL reason=" << error.what() << '\n'; return 1; }
}
