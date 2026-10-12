#include "../zkrelation-proof.cuh"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace zkdl_gat::relation;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void report_stage(const char* stage) {
    std::cout << "sumcheck_round_oracle_check stage=" << stage << std::endl;
}

ConstraintDescriptor make_descriptor(std::uint32_t degree) {
    ConstraintDescriptor descriptor;
    descriptor.id = "sumcheck-round-oracle/degree-" + std::to_string(degree);
    descriptor.degree = degree;
    descriptor.constant = zkdl::fr_from_u32(7U);
    if (degree == 2U) {
        descriptor.terms = {
            Monomial{zkdl::fr_from_u32(3U), {0U, 1U}},
            Monomial{zkdl::fr_neg(zkdl::fr_from_u32(5U)), {2U}},
            Monomial{zkdl::fr_from_u32(11U), {3U, 4U}},
            Monomial{zkdl::fr_from_u32(13U), {0U, 0U}}};
    } else if (degree == 4U) {
        descriptor.terms = {
            Monomial{zkdl::fr_from_u32(2U), {0U, 1U, 2U, 3U}},
            Monomial{zkdl::fr_neg(zkdl::fr_from_u32(4U)), {4U, 0U}},
            Monomial{zkdl::fr_from_u32(9U), {1U, 1U, 2U, 3U}},
            Monomial{zkdl::fr_from_u32(17U), {4U}}};
    } else {
        throw std::invalid_argument("unsupported oracle-test degree");
    }
    return descriptor;
}

std::vector<std::vector<FrWire>> make_values(std::size_t row_count) {
    constexpr std::size_t kTensorCount = 5U;
    std::vector<std::vector<FrWire>> values(
        kTensorCount, std::vector<FrWire>(2U * row_count));
    for (std::size_t tensor = 0U; tensor < values.size(); ++tensor) {
        for (std::size_t row = 0U; row < row_count; ++row) {
            const std::uint32_t base = static_cast<std::uint32_t>(
                ((row % 257U) * 31U + tensor * 19U + 5U) % 101U);
            values[tensor][2U * row] = zkdl::fr_from_u32(base);
            values[tensor][2U * row + 1U] = zkdl::fr_from_u32(
                static_cast<std::uint32_t>((base * 7U + row % 43U + 3U) % 103U));
        }
    }
    return values;
}

std::vector<FrWire> old_row_oracle(
    const ConstraintDescriptor& descriptor,
    const std::vector<std::vector<FrWire>>& current,
    std::uint32_t degree, std::size_t row_begin, std::size_t row_end) {
    std::vector<FrWire> result(degree + 1U);
    for (std::size_t row = row_begin; row < row_end; ++row) {
        std::vector<FrWire> values;
        values.reserve(current.size());
        for (const auto& tensor : current) values.push_back(tensor[2U * row]);
        for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation) {
            const FrWire x = zkdl::fr_from_u32(evaluation);
            values.clear();
            for (const auto& tensor : current)
                values.push_back(zkdl::fr_add(
                    tensor[2U * row], zkdl::fr_mul(x, zkdl::fr_sub(
                        tensor[2U * row + 1U], tensor[2U * row]))));
            result[evaluation] = zkdl::fr_add(
                result[evaluation], eval_constraint(descriptor, values));
        }
    }
    return result;
}

void require_same_wire(const std::vector<FrWire>& actual,
                       const std::vector<FrWire>& expected,
                       const std::string& label) {
    require(actual.size() == expected.size(), label + ": evaluation count mismatch");
    for (std::size_t index = 0U; index < actual.size(); ++index)
        require(actual[index] == expected[index],
                label + ": FrWire mismatch at evaluation " + std::to_string(index));
}

void check_threshold_range(std::uint32_t degree, std::size_t row_count,
                           std::size_t row_begin) {
    const auto descriptor = make_descriptor(degree);
    const auto values = make_values(row_count + row_begin + 23U);
    const std::size_t row_end = row_begin + row_count;
    const std::size_t chunks = row_count < 4096U ? 1U : std::min(
        row_count, static_cast<std::size_t>(zkdl::cpu_parallel_worker_count()) + 1U);
    ZkSumcheckRoundEvaluationScratch scratch(
        chunks, values.size(), degree + 1U);
    std::vector<FrWire> actual(degree + 1U);
    std::string error;
    require(zksc_accumulate_round_evaluation_rows(
                descriptor, values, degree, row_begin, row_end,
                actual, scratch, &error),
            "range helper rejected valid row window: " + error);
    require_same_wire(actual,
                      old_row_oracle(descriptor, values, degree,
                                     row_begin, row_end),
                      "range oracle degree=" + std::to_string(degree) +
                      " rows=" + std::to_string(row_count));
}

void check_scratch_shape_failclosed() {
    constexpr std::uint32_t kDegree = 2U;
    const auto descriptor = make_descriptor(kDegree);
    const auto values = make_values(4U);
    std::vector<FrWire> evaluations(kDegree + 1U);
    std::string error;

    ZkSumcheckRoundEvaluationScratch short_values(
        1U, values.size(), evaluations.size());
    short_values.row_values[0U].pop_back();
    require(!zksc_accumulate_round_evaluation_rows(
                descriptor, values, kDegree, 0U, 1U, evaluations,
                short_values, &error) &&
                error == "segmented ZkSumcheck row scratch shape is invalid",
            "row evaluator accepted a short per-chunk values buffer");

    error.clear();
    ZkSumcheckRoundEvaluationScratch short_partial(
        1U, values.size(), evaluations.size());
    short_partial.chunk_evaluations[0U].pop_back();
    require(!zksc_accumulate_round_evaluation_rows(
                descriptor, values, kDegree, 0U, 1U, evaluations,
                short_partial, &error) &&
                error == "segmented ZkSumcheck row scratch shape is invalid",
            "row evaluator accepted a short per-chunk evaluation buffer");
}

void check_tiny_sumcheck_proof_roundtrip() {
    constexpr std::uint32_t kDomain = 8U;
    constexpr std::uint32_t kDegree = 2U;
    const auto descriptor = make_descriptor(kDegree);
    const auto values = make_values(kDomain / 2U);
    FrWire initial_claim{};
    for (std::uint32_t index = 0U; index < kDomain; ++index) {
        std::vector<FrWire> point_values;
        point_values.reserve(values.size());
        for (const auto& tensor : values) point_values.push_back(tensor[index]);
        initial_claim = zkdl::fr_add(
            initial_claim, eval_constraint(descriptor, point_values));
    }

    const auto base = zkdl::cpu_g1_generator();
    std::vector<zkdl::CpuG1Point> generators;
    for (std::uint32_t index = 0U; index < 4U; ++index)
        generators.push_back(zkdl::cpu_g1_mul(
            base, zkdl::fr_from_u32(index + 2U)));
    const auto h = zkdl::cpu_g1_mul(base, zkdl::fr_from_u32(29U));
    const auto u = zkdl::cpu_g1_mul(base, zkdl::fr_from_u32(31U));
    RelationStatement statement;
    statement.statement_digest = zkdl::sha256(std::vector<zkdl::byte>{'s'});
    statement.model_digest = zkdl::sha256(std::vector<zkdl::byte>{'m'});
    statement.table_digest = zkdl::sha256(std::vector<zkdl::byte>{'t'});
    statement.ledger_digest = zkdl::sha256(std::vector<zkdl::byte>{'l'});
    statement.registry_digest = zkdl::sha256(std::vector<zkdl::byte>{'r'});
    statement.srs_digest = zkdl::sha256(std::vector<zkdl::byte>{'p'});

    zkdl::Transcript prover_transcript("sumcheck-round-oracle-roundtrip-v1");
    ZkSumcheckProverResult prover;
    std::string error;
    report_stage("tiny_prove_start");
    require(prove_zk_sumcheck(
                descriptor, values, kDegree, kDomain, generators, h, u,
                initial_claim, statement, "oracle/tiny", prover_transcript,
                prover, &error),
            "tiny hidden Sumcheck prover failed: " + error);
    report_stage("tiny_prove_done");
    require(prover.proof.rounds.size() == 3U,
            "tiny hidden Sumcheck emitted the wrong round count");

    zkdl::Transcript verifier_transcript("sumcheck-round-oracle-roundtrip-v1");
    ZkSumcheckVerifierResult verifier;
    report_stage("tiny_verify_start");
    require(verify_zk_sumcheck(
                prover.proof, descriptor, kDegree, kDomain, generators, h, u,
                initial_claim, statement, "oracle/tiny", verifier_transcript,
                verifier, &error),
            "independent tiny hidden Sumcheck verifier failed: " + error);
    report_stage("tiny_verify_done");
    require(fr_compare_mixed(
                prover_transcript.challenge("sumcheck-round-oracle/followup"),
                verifier_transcript.challenge("sumcheck-round-oracle/followup")) == 0,
            "tiny Sumcheck prover/verifier transcript replay diverged");
    report_stage("tiny_transcript_replay_done");
}

void check_full_routes(std::uint32_t degree) {
    constexpr std::size_t kRows = 8192U;
    constexpr std::size_t kBasis = 8192U;
    const auto descriptor = make_descriptor(degree);
    const auto values = make_values(kRows);
    report_stage(degree == 2U ? "degree2_oracle_start" : "degree4_oracle_start");
    const auto expected = old_row_oracle(
        descriptor, values, degree, 0U, kRows);
    report_stage(degree == 2U ? "degree2_oracle_done" : "degree4_oracle_done");
    std::string error;

    std::vector<FrWire> direct(degree + 1U);
    report_stage(degree == 2U ? "degree2_direct_start" : "degree4_direct_start");
    require(zksc_accumulate_round_evaluations(
                descriptor, values, degree, direct, &error),
            "direct cached-row accumulator failed: " + error);
    report_stage(degree == 2U ? "degree2_direct_done" : "degree4_direct_done");
    require_same_wire(direct, expected, "direct cached-row oracle");

    std::vector<FrWire> folded;
    report_stage(degree == 2U ? "degree2_folded_start" : "degree4_folded_start");
    require(zksc_folded_round_evaluations(
                descriptor, values, degree, folded, &error),
            "folded-state accumulator failed: " + error);
    report_stage(degree == 2U ? "degree2_folded_done" : "degree4_folded_done");
    require_same_wire(folded, expected, "folded-state oracle");

    std::vector<SegmentedTensorProvider> providers;
    report_stage(degree == 2U ? "degree2_provider_setup_start" : "degree4_provider_setup_start");
    require(make_segmented_tensor_providers(values, kBasis, providers, &error),
            "provider fixture construction failed: " + error);
    require(providers.size() == values.size() &&
                providers.front().segment_size() == kBasis,
            "provider fixture did not retain the expected multi-provider segment shape");
    report_stage(degree == 2U ? "degree2_provider_eval_start" : "degree4_provider_eval_start");
    std::vector<FrWire> provider_round;
    require(zksc_segmented_round_evaluations(
                descriptor, providers, degree, 0U, {}, provider_round, &error),
            "provider-row accumulator failed: " + error);
    report_stage(degree == 2U ? "degree2_provider_eval_done" : "degree4_provider_eval_done");
    require_same_wire(provider_round, expected, "provider-row oracle");

    // Repeating the same fixed 0..degree evaluation points must remain byte exact.
    std::vector<FrWire> repeated;
    report_stage(degree == 2U ? "degree2_repeat_start" : "degree4_repeat_start");
    require(zksc_folded_round_evaluations(
                descriptor, values, degree, repeated, &error),
            "repeated cached-row accumulator failed: " + error);
    report_stage(degree == 2U ? "degree2_repeat_done" : "degree4_repeat_done");
    require_same_wire(repeated, folded, "repeated cached-row evaluation");

    // Shape and reference failures remain fail-closed.
    std::vector<FrWire> wrong_size(degree);
    require(!zksc_accumulate_round_evaluations(
                descriptor, values, degree, wrong_size, &error) &&
                error == "segmented ZkSumcheck current round shape is invalid",
            "round accumulator accepted a wrong evaluation vector shape");
    auto mismatched = values;
    mismatched.back().pop_back();
    std::vector<FrWire> ignored(degree + 1U);
    error.clear();
    require(!zksc_accumulate_round_evaluations(
                descriptor, mismatched, degree, ignored, &error) &&
                error == "segmented ZkSumcheck current tensor shapes differ",
            "round accumulator accepted mismatched provider tensor shapes");

    ConstraintDescriptor bad_reference = descriptor;
    bad_reference.terms = {Monomial{zkdl::fr_from_u32(1U),
                                    {static_cast<std::uint32_t>(values.size())}}};
    bool threw_out_of_range = false;
    try {
        std::vector<FrWire> should_not_succeed(degree + 1U);
        (void)zksc_accumulate_round_evaluations(
            bad_reference, values, degree, should_not_succeed, &error);
    } catch (const std::out_of_range&) {
        threw_out_of_range = true;
    }
    require(threw_out_of_range,
            "invalid monomial reference no longer fails closed through vector::at");
}

}  // namespace

int main() {
    try {
        report_stage("threshold_4095_start");
        check_threshold_range(2U, 4095U, 17U);
        report_stage("threshold_4095_done");
        report_stage("threshold_4096_start");
        check_threshold_range(4U, 4096U, 29U);
        report_stage("threshold_4096_done");
        report_stage("scratch_shape_start");
        check_scratch_shape_failclosed();
        report_stage("scratch_shape_done");
        check_tiny_sumcheck_proof_roundtrip();
        check_full_routes(2U);
        check_full_routes(4U);
        std::cout << "sumcheck_round_oracle_check=PASS threshold=4095/4096 "
                     "nonzero_row_begin=PASS degrees=2/4 "
                     "provider/cache_wire_equality=PASS tiny_proof_verify=PASS "
                     "scratch_shape_failclosed=PASS shape_failclosed=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sumcheck_round_oracle_check=FAIL error="
                  << error.what() << '\n';
        return 1;
    }
}
