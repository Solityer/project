#include "../zkrelation-proof.cuh"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace zkdl_gat::relation;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct OldEightBitEqmaskTable {
    struct Chunk {
        std::uint32_t bit_offset = 0U;
        std::uint32_t bit_count = 0U;
        std::uint32_t entry_count = 0U;
        std::size_t entry_offset = 0U;
    };

    std::vector<Chunk> chunks;
    std::vector<FrWire> weights;
    std::uint32_t bits = 0U;

    void build(const std::vector<FrWire>& rho) {
        bits = static_cast<std::uint32_t>(rho.size());
        std::size_t payload_entries = 0U;
        for (std::uint32_t offset = 0U; offset < bits; offset += 8U) {
            const std::uint32_t chunk_bits =
                std::min<std::uint32_t>(8U, bits - offset);
            const std::uint32_t entries = 1U << chunk_bits;
            chunks.push_back(Chunk{offset, chunk_bits, entries, payload_entries});
            payload_entries += entries;
        }
        weights.resize(payload_entries);
        const FrWire one = zkdl::fr_from_u32(1U);
        for (const Chunk& chunk : chunks) {
            FrWire* table = weights.data() + chunk.entry_offset;
            table[0U] = one;
            for (std::uint32_t local_bit = 0U;
                 local_bit < chunk.bit_count; ++local_bit) {
                const FrWire& challenge = rho[chunk.bit_offset + local_bit];
                const FrWire zero_weight = zkdl::fr_sub(one, challenge);
                const std::uint32_t half = 1U << local_bit;
                for (std::uint32_t prefix = 0U; prefix < half; ++prefix) {
                    const FrWire prior = table[prefix];
                    table[prefix] = zkdl::fr_mul(prior, zero_weight);
                    table[half + prefix] = zkdl::fr_mul(prior, challenge);
                }
            }
        }
    }

    FrWire evaluate(std::uint64_t index) const {
        FrWire value = zkdl::fr_from_u32(1U);
        for (const Chunk& chunk : chunks) {
            const std::uint64_t mask = chunk.entry_count - 1U;
            const std::uint64_t entry = (index >> chunk.bit_offset) & mask;
            value = zkdl::fr_mul(value, weights[chunk.entry_offset + entry]);
        }
        return value;
    }
};

std::vector<FrWire> make_rho(std::uint32_t bits, bool edge_values,
                             std::uint64_t& state) {
    std::vector<FrWire> rho(bits);
    for (std::uint32_t bit = 0U; bit < bits; ++bit) {
        if (edge_values) {
            rho[bit] = zkdl::fr_from_u32((bit & 1U) == 0U ? 0U : 1U);
        } else {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            rho[bit] = zkdl::fr_from_u32(
                static_cast<std::uint32_t>(2U + ((state >> 17U) % 65521U)));
        }
    }
    return rho;
}

std::vector<std::uint64_t> check_indices(std::uint32_t bits,
                                         std::uint64_t& state) {
    if (bits <= 16U) {
        std::vector<std::uint64_t> indices(std::uint64_t{1U} << bits);
        for (std::uint64_t index = 0U; index < indices.size(); ++index)
            indices[static_cast<std::size_t>(index)] = index;
        return indices;
    }
    const std::uint64_t domain = std::uint64_t{1U} << bits;
    const std::uint64_t mask = domain - 1U;
    std::vector<std::uint64_t> indices = {
        0U, 1U, 2U, domain / 2U, domain - 2U, domain - 1U,
        (std::uint64_t{1U} << (bits - 1U)) - 1U};
    for (std::uint32_t probe = 0U; probe < 4096U; ++probe) {
        state = state * 2862933555777941757ULL + 3037000493ULL;
        indices.push_back(state & mask);
    }
    return indices;
}

void compare_table_with_old_algorithm() {
    const std::uint32_t bit_counts[] = {0U, 1U, 7U, 8U, 9U, 13U, 14U,
                                        15U, 16U, 17U, 27U, 30U};
    std::uint64_t state = 0x5a17c9e3d2b4810fULL;
    for (const std::uint32_t bits : bit_counts) {
        for (const bool edge_values : {true, false}) {
            const auto rho = make_rho(bits, edge_values, state);
            OldEightBitEqmaskTable old_table;
            old_table.build(rho);
            PointwiseEqmaskWeightTable new_table;
            std::string error;
            require(new_table.build(rho, &error),
                    "new table rejected valid challenge vector: " + error);
            require(new_table.bits == bits,
                    "new table reported the wrong challenge count");
            if (bits == 16U)
                require(new_table.entry_count == 512U,
                        "16-bit compatibility table no longer uses 8-bit chunks");
            if (bits == 17U)
                require(new_table.entry_count == 768U,
                        "17-bit table did not switch to balanced 9+8 groups");
            std::uint64_t expected_peak = 0U;
            require(pointwise_eqmask_weight_table_peak_bytes(bits, expected_peak) &&
                        new_table.peak_live_bytes == expected_peak,
                    "new table peak-live-byte accounting disagrees with its bound");
            const auto indices = check_indices(bits, state);
            for (const std::uint64_t index : indices)
                require(new_table.evaluate(index) == old_table.evaluate(index),
                        "new table differs from the old 8-bit table at index " +
                            std::to_string(index) + " with " +
                            std::to_string(bits) + " challenge bits");
            bool rejected_domain_end = false;
            try {
                new_table.evaluate(std::uint64_t{1U} << bits);
            } catch (const std::out_of_range&) {
                rejected_domain_end = true;
            }
            require(rejected_domain_end,
                    "new table accepted an index outside the logical domain");
        }
    }

    PointwiseEqmaskWeightTable table27;
    std::string error;
    require(table27.build(make_rho(27U, false, state), &error),
            "27-bit table failed to build");
    require(table27.entry_count == 24576U,
            "27-bit table did not use the expected balanced 14+13 layout");
    PointwiseEqmaskWeightTable table30;
    require(table30.build(make_rho(30U, false, state), &error),
            "30-bit table failed to build");
    require(table30.entry_count == 65536U,
            "30-bit table did not use the expected balanced 15+15 layout");

    PointwiseEqmaskWeightTable invalid;
    require(!invalid.build(std::vector<FrWire>(31U, zkdl::fr_from_u32(1U)), &error),
            "table accepted more than the supported 30 challenge bits");
    FrWire noncanonical{};
    noncanonical.fill(std::numeric_limits<std::uint32_t>::max());
    require(!invalid.build({noncanonical}, &error),
            "table accepted a noncanonical challenge");
    std::uint64_t ignored_peak = 0U;
    require(!pointwise_eqmask_weight_table_peak_bytes(31U, ignored_peak),
            "memory-bound helper accepted more than the supported 30 bits");
}

void check_provider_order() {
    ConstraintDescriptor descriptor;
    descriptor.id = "check/eqmask-provider-order";
    descriptor.degree = 1U;
    descriptor.terms = {Monomial{zkdl::fr_from_u32(1U), {0U}}};
    descriptor.logical_shape = {3U, 3U};
    descriptor.padded_shape = {4U, 4U};

    std::vector<FrWire> source(16U);
    for (std::size_t index = 0U; index < source.size(); ++index)
        source[index] = zkdl::fr_from_u32(static_cast<std::uint32_t>(index + 1U));
    SegmentedTensorProvider source_provider;
    std::string error;
    require(SegmentedTensorProvider::from_values(source, 16U,
                                                  source_provider, &error),
            "failed to create provider-order fixture: " + error);
    const std::vector<SegmentedTensorProvider> input = {source_provider};
    const std::vector<FrWire> rho = {
        zkdl::fr_from_u32(0U), zkdl::fr_from_u32(1U),
        zkdl::fr_from_u32(5U), zkdl::fr_from_u32(9U)};
    OldEightBitEqmaskTable old_table;
    old_table.build(rho);
    PointwiseEqmaskWeightTable table;
    require(table.build(rho, &error), "provider-order table build failed");

    std::vector<SegmentedTensorProvider> output;
    require(pointwise_providers(descriptor, input, rho, output, &error,
                                &table),
            "pointwise provider construction failed: " + error);
    require(output.size() == 3U,
            "pointwise provider count/order changed");
    std::vector<FrWire> copied_source, weights, active_selector;
    require(output[0U].read_segment(0U, copied_source, &error) &&
                output[1U].read_segment(0U, weights, &error) &&
                output[2U].read_segment(0U, active_selector, &error),
            "pointwise provider range read failed: " + error);
    require(copied_source == source,
            "pointwise input provider moved from its original position");
    for (std::uint64_t index = 0U; index < 16U; ++index) {
        require(weights[static_cast<std::size_t>(index)] == old_table.evaluate(index),
                "eqmask provider output order changed at index " +
                    std::to_string(index));
        const FrWire expected_selector = zkdl::fr_from_u32(
            pointwise_active_slot(descriptor, index) ? 1U : 0U);
        require(active_selector[static_cast<std::size_t>(index)] == expected_selector,
                "active-slot selector provider order changed");
    }
}

template <typename Table>
std::pair<double, std::uint64_t> benchmark_table(
    Table& table, const std::function<void()>& build,
    std::uint64_t start, std::uint64_t count, std::uint64_t mask) {
    std::uint64_t checksum = 0U;
    const auto begin = std::chrono::steady_clock::now();
    build();
    for (std::uint64_t offset = 0U; offset < count; ++offset) {
        const FrWire value = table.evaluate((start + offset) & mask);
        checksum ^= static_cast<std::uint64_t>(value[0U]) + offset;
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    return {seconds, checksum};
}

void compare_fixed_index_microbenchmark() {
    constexpr std::uint32_t bits = 27U;
    constexpr std::uint64_t sample_count = std::uint64_t{1U} << 20U;
    constexpr std::uint64_t start_index = 0x1234567U;
    std::uint64_t state = 0x2dc1a4e79b6035f8ULL;
    const auto rho = make_rho(bits, false, state);
    OldEightBitEqmaskTable old_table;
    PointwiseEqmaskWeightTable new_table;
    std::string error;
    constexpr std::uint64_t mask = (std::uint64_t{1U} << bits) - 1U;
    const auto old_result = benchmark_table(
        old_table, [&]() { old_table.build(rho); }, start_index,
        sample_count, mask);
    const auto new_result = benchmark_table(
        new_table,
        [&]() {
            require(new_table.build(rho, &error),
                    "microbenchmark table build failed: " + error);
        },
        start_index, sample_count, mask);
    require(old_result.second == new_result.second,
            "old and new table microbenchmark outputs differ");
    std::cout << "eqmask_table_ab=PASS bits=" << bits
              << " fixed_indices=" << sample_count
              << " old_seconds=" << old_result.first
              << " balanced_seconds=" << new_result.first
              << " speedup=" << old_result.first / new_result.first << '\n';
}

std::pair<double, ZkSumcheckFoldedState> serial_fold_oracle(
    const SegmentedTensorProvider& provider, const FrWire& challenge) {
    std::vector<FrWire> block;
    std::string error;
    const auto started = std::chrono::steady_clock::now();
    require(provider.read_segment(0U, block, &error),
            "serial fold oracle provider read failed: " + error);
    ZkSumcheckFoldedState state(1U);
    state[0U].reserve(block.size() / 2U);
    const zkdl_blst::blst_fr challenge_montgomery =
        zkdl::cpu_ipa_frwire_to_montgomery_unchecked(challenge);
    for (std::size_t index = 0U; index < block.size() / 2U; ++index) {
        const auto left =
            zkdl::cpu_ipa_frwire_to_montgomery_unchecked(block[2U * index]);
        const auto right = zkdl::cpu_ipa_frwire_to_montgomery_unchecked(
            block[2U * index + 1U]);
        zkdl_blst::blst_fr difference{}, scaled{}, folded{};
        zkdl_blst::blst_fr_sub(&difference, &right, &left);
        zkdl_blst::blst_fr_mul(&scaled, &challenge_montgomery, &difference);
        zkdl_blst::blst_fr_add(&folded, &left, &scaled);
        state[0U].push_back(folded);
    }
    return {std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count(),
            std::move(state)};
}

void check_fold_parallel_oracle() {
    constexpr std::uint32_t domain = 1U << 18U;
    constexpr std::uint32_t segment_size = 1U << 18U;
    std::vector<FrWire> source(domain);
    for (std::uint32_t index = 0U; index < domain; ++index)
        source[index] = zkdl::fr_from_u32(1U + ((index * 17U + 9U) % 65521U));
    SegmentedTensorProvider provider;
    std::string error;
    require(SegmentedTensorProvider::from_values(source, segment_size,
                                                  provider, &error),
            "parallel fold provider setup failed: " + error);
    const std::vector<SegmentedTensorProvider> providers = {provider};
    const FrWire challenge = zkdl::fr_from_u32(37U);
    const auto serial = serial_fold_oracle(provider, challenge);
    ZkSumcheckFoldedState parallel;
    const auto parallel_started = std::chrono::steady_clock::now();
    require(zksc_segmented_fold_state_initialize(providers, challenge,
                                                  parallel, &error),
            "parallel fold-state initialization failed: " + error);
    const double parallel_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - parallel_started).count();
    require(parallel.size() == serial.second.size() &&
                parallel[0U].size() == domain / 2U,
            "parallel fold-state shape differs from serial oracle");
    for (std::size_t index = 0U; index < parallel[0U].size(); ++index)
        require(zksc_folded_scalar_to_wire(parallel[0U][index]) ==
                    zksc_folded_scalar_to_wire(serial.second[0U][index]),
                "parallel fold differs from serial oracle at pair " +
                    std::to_string(index));
    std::cout << "fold_init_ab=PASS domain=" << domain
              << " pairs=" << domain / 2U
              << " serial_seconds=" << serial.first
              << " parallel_seconds=" << parallel_seconds
              << " speedup=" << serial.first / parallel_seconds << '\n';
}

void check_folded_sumcheck_prove_verify(
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& h, const zkdl::CpuG1Point& u) {
    ConstraintDescriptor descriptor;
    descriptor.id = "check/eqmask-folded-sumcheck";
    descriptor.degree = 1U;
    descriptor.terms = {Monomial{zkdl::fr_from_u32(1U), {0U}}};
    descriptor.logical_shape = {3U};
    descriptor.padded_shape = {4U};
    const std::vector<FrWire> source = {
        zkdl::fr_from_u32(2U), zkdl::fr_from_u32(3U),
        zkdl::fr_from_u32(5U), zkdl::fr_from_u32(7U)};
    SegmentedTensorProvider source_provider;
    std::string error;
    require(SegmentedTensorProvider::from_values(source, 2U,
                                                  source_provider, &error),
            "tiny folded Sumcheck provider setup failed: " + error);
    const std::vector<SegmentedTensorProvider> inputs = {source_provider};
    const std::vector<FrWire> rho = {
        zkdl::fr_from_u32(11U), zkdl::fr_from_u32(13U)};
    PointwiseEqmaskWeightTable table;
    require(table.build(rho, &error),
            "tiny folded Sumcheck eqmask build failed: " + error);
    std::vector<SegmentedTensorProvider> weighted_providers;
    require(pointwise_providers(descriptor, inputs, rho, weighted_providers,
                                &error, &table),
            "tiny folded Sumcheck provider construction failed: " + error);
    const ConstraintDescriptor weighted =
        pointwise_sumcheck_descriptor(descriptor, inputs.size());
    FrWire initial_claim{};
    for (std::uint64_t index = 0U; index < source.size(); ++index) {
        const FrWire selector = zkdl::fr_from_u32(
            pointwise_active_slot(descriptor, index) ? 1U : 0U);
        const FrWire value = zkdl::fr_mul(
            zkdl::fr_mul(source[index], table.evaluate(index)), selector);
        initial_claim = zkdl::fr_add(initial_claim, value);
    }

    RelationStatement statement;
    statement.statement_digest = zkdl::sha256(std::vector<zkdl::byte>{'s'});
    statement.model_digest = zkdl::sha256(std::vector<zkdl::byte>{'m'});
    statement.table_digest = zkdl::sha256(std::vector<zkdl::byte>{'t'});
    statement.ledger_digest = zkdl::sha256(std::vector<zkdl::byte>{'l'});
    statement.registry_digest = zkdl::sha256(std::vector<zkdl::byte>{'r'});
    statement.srs_digest = zkdl::sha256(std::vector<zkdl::byte>{'p'});
    constexpr const char* operation = "tiny.eqmask.folded.sumcheck";
    zkdl::Transcript prover_transcript(operation);
    ZkSumcheckProverResult prover;
    require(prove_zk_sumcheck_segmented(
                weighted, weighted_providers, weighted.degree, 4U,
                generators, h, u, initial_claim, statement, operation,
                prover_transcript, prover, &error, 2U, true, nullptr, nullptr,
                nullptr, true, table.peak_live_bytes),
            "tiny folded Sumcheck prover rejected valid input: " + error);
    zkdl::Transcript verifier_transcript(operation);
    ZkSumcheckVerifierResult verifier;
    require(verify_zk_sumcheck(prover.proof, weighted, weighted.degree, 4U,
                               generators, h, u, initial_claim, statement,
                               operation, verifier_transcript, verifier, &error,
                               2U),
            "fresh tiny folded Sumcheck verifier rejected proof: " + error);
}

}  // namespace

int main() {
    try {
        compare_table_with_old_algorithm();
        check_provider_order();
        compare_fixed_index_microbenchmark();
        check_fold_parallel_oracle();
        const std::vector<zkdl::CpuG1Point> generators{
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(2U)),
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(3U)),
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(5U)),
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(13U)),
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(17U)),
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(29U)),
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(31U)),
            zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(37U))};
        const auto h = zkdl::cpu_g1_mul(
            zkdl::cpu_g1_generator(), zkdl::fr_from_u32(19U));
        const auto u = zkdl::cpu_g1_mul(
            zkdl::cpu_g1_generator(), zkdl::fr_from_u32(23U));
        check_folded_sumcheck_prove_verify(generators, h, u);
        std::cout << "pointwise_eqmask_table_check=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pointwise_eqmask_table_check=FAIL reason="
                  << error.what() << '\n';
        return 1;
    }
}
