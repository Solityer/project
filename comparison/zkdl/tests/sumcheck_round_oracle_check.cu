#include "../zkrelation-proof.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <stdlib.h>
#include <unistd.h>

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

ConstraintDescriptor make_factorized_descriptor(bool inverse_relation,
                                                 bool reverse_sign = false) {
    ConstraintDescriptor descriptor;
    descriptor.id = inverse_relation
        ? "sumcheck-round-oracle/inverse-edge"
        : "sumcheck-round-oracle/weighted-product";
    descriptor.degree = 4U;
    descriptor.constant = FrWire{};
    const FrWire one = zkdl::fr_from_u32(1U);
    const FrWire negative_one = zkdl::fr_neg(one);
    const FrWire residual_coefficient = reverse_sign ? negative_one : one;
    const FrWire product_coefficient = reverse_sign ? one : negative_one;
    if (inverse_relation) {
        descriptor.terms = {
            Monomial{residual_coefficient, {3U, 3U}},
            Monomial{product_coefficient, {0U, 0U, 3U, 3U}}};
    } else {
        descriptor.terms = {
            Monomial{residual_coefficient, {2U, 3U, 3U}},
            Monomial{product_coefficient, {0U, 0U, 3U, 3U}}};
    }
    return descriptor;
}

ConstraintDescriptor make_pointwise_weighted_descriptor(const std::string& kind) {
    ConstraintDescriptor descriptor;
    descriptor.id = "sumcheck-round-oracle/pointwise-weighted/" + kind;
    descriptor.degree = 4U;
    descriptor.constant = FrWire{};
    const FrWire rho = zkdl::fr_from_u32(17U);
    const FrWire gamma = zkdl::fr_from_u32(29U);
    if (kind == "recurrence") {
        descriptor.terms = {
            Monomial{rho, {0U, 4U, 5U}},
            Monomial{zkdl::fr_neg(rho), {1U, 4U, 5U}},
            Monomial{gamma, {2U, 4U, 5U}},
            Monomial{zkdl::fr_neg(zkdl::fr_from_u32(31U)), {2U, 3U, 4U, 5U}}};
    } else if (kind == "reset") {
        descriptor.terms = {
            Monomial{rho, {0U, 1U, 2U, 3U}}};
    } else if (kind == "terminal") {
        descriptor.terms = {
            Monomial{rho, {0U, 1U, 3U, 4U}},
            Monomial{zkdl::fr_neg(gamma), {1U, 2U, 3U, 4U}}};
    } else if (kind == "unit-repeated") {
        const FrWire one = zkdl::fr_from_u32(1U);
        descriptor.terms = {
            Monomial{one, {0U, 2U, 3U}},
            Monomial{zkdl::fr_neg(one), {0U, 0U, 2U, 3U}},
            Monomial{one, {1U, 2U, 3U}}};
    } else if (kind == "zero-arity") {
        descriptor.terms = {
            Monomial{rho, {1U, 2U}},
            Monomial{zkdl::fr_neg(gamma), {0U, 1U, 2U}}};
    } else {
        throw std::invalid_argument("unsupported pointwise weighted descriptor");
    }
    return descriptor;
}

ConstraintDescriptor make_degree_three_pointwise_descriptor() {
    ConstraintDescriptor descriptor;
    descriptor.id = "sumcheck-round-oracle/pointwise-degree-three";
    descriptor.degree = 3U;
    descriptor.constant = FrWire{};
    descriptor.terms = {
        Monomial{zkdl::fr_from_u32(7U), {0U, 2U, 3U}},
        Monomial{zkdl::fr_neg(zkdl::fr_from_u32(11U)), {1U, 2U, 3U}}};
    return descriptor;
}

std::vector<std::vector<FrWire>> make_values(std::size_t row_count,
                                             std::size_t tensor_count = 5U) {
    std::vector<std::vector<FrWire>> values(
        tensor_count, std::vector<FrWire>(2U * row_count));
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
    std::uint32_t degree, std::size_t row_begin, std::size_t row_end,
    const std::vector<FrWire>& evaluation_points = {}) {
    if (!evaluation_points.empty() && evaluation_points.size() != degree + 1U)
        throw std::invalid_argument("oracle evaluation point count mismatch");
    std::vector<FrWire> result(degree + 1U);
    for (std::size_t row = row_begin; row < row_end; ++row) {
        std::vector<FrWire> values;
        values.reserve(current.size());
        for (const auto& tensor : current) values.push_back(tensor[2U * row]);
        for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation) {
            const FrWire x = evaluation_points.empty()
                ? zkdl::fr_from_u32(evaluation)
                : evaluation_points[evaluation];
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
                       const std::string& label);

#if defined(ZKDL_USE_BLST_G1_CODEC)
void legacy_montgomery_wire_row_loop(
    const std::vector<std::vector<FrWire>>& current,
    const ZkSumcheckMontgomeryConstraintPlan& plan,
    std::vector<FrWire>& evaluations) {
    const std::size_t row_count = current.front().size() / 2U;
    const std::size_t chunk_limit = row_count < 4096U ? 1U : std::min(
        row_count, static_cast<std::size_t>(zkdl::cpu_parallel_worker_count()) + 1U);
    const std::size_t grain = row_count / chunk_limit +
        (row_count % chunk_limit != 0U ? 1U : 0U);
    const std::size_t active_chunks = row_count / grain +
        (row_count % grain != 0U ? 1U : 0U);
    std::vector<std::vector<zkdl_blst::blst_fr>> row_states(
        active_chunks, std::vector<zkdl_blst::blst_fr>(3U * current.size()));
    std::vector<std::vector<zkdl_blst::blst_fr>> partials(
        active_chunks, std::vector<zkdl_blst::blst_fr>(plan.degree + 1U));
    std::vector<unsigned char> valid(active_chunks, 1U);
    evaluations.assign(plan.degree + 1U, FrWire{});
    zkdl::cpu_parallel_for(row_count, grain, [&](std::size_t local_row) {
        const std::size_t chunk = local_row / grain;
        auto& state = row_states[chunk];
        auto& partial = partials[chunk];
        for (std::size_t tensor = 0U; tensor < current.size(); ++tensor) {
            const auto& left_wire = current[tensor][2U * local_row];
            const auto& right_wire = current[tensor][2U * local_row + 1U];
            if (!zkdl::fr_canonical(left_wire) || !zkdl::fr_canonical(right_wire)) {
                valid[chunk] = 0U;
                return;
            }
            const auto left = zkdl::cpu_ipa_frwire_to_montgomery_unchecked(left_wire);
            const auto right = zkdl::cpu_ipa_frwire_to_montgomery_unchecked(right_wire);
            state[tensor] = left;
            zkdl_blst::blst_fr_sub(&state[current.size() + tensor], &right, &left);
        }
        for (std::uint32_t evaluation = 0U; evaluation <= plan.degree; ++evaluation) {
            const auto& x = plan.evaluation_points[evaluation];
            for (std::size_t tensor = 0U; tensor < current.size(); ++tensor) {
                zkdl_blst::blst_fr scaled{};
                zkdl_blst::blst_fr_mul(
                    &scaled, &x, &state[current.size() + tensor]);
                zkdl_blst::blst_fr_add(
                    &state[2U * current.size() + tensor], &state[tensor], &scaled);
            }
            const auto value = zksc_eval_constraint_montgomery(
                plan, state.data() + 2U * current.size());
            zkdl_blst::blst_fr next{};
            zkdl_blst::blst_fr_add(&next, &partial[evaluation], &value);
            partial[evaluation] = next;
        }
    });
    require(std::all_of(valid.begin(), valid.end(),
                        [](unsigned char value) { return value != 0U; }),
            "legacy Mont benchmark received non-canonical input");
    for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk)
        for (std::uint32_t evaluation = 0U; evaluation <= plan.degree; ++evaluation)
            evaluations[evaluation] = zkdl::fr_add(
                evaluations[evaluation],
                zkdl::cpu_ipa_montgomery_to_frwire(partials[chunk][evaluation]));
}

void legacy_montgomery_cached_row_loop(
    const ZkSumcheckFoldedState& current,
    const ZkSumcheckMontgomeryConstraintPlan& plan,
    std::vector<FrWire>& evaluations) {
    const std::size_t row_count = current.front().size() / 2U;
    const std::size_t chunk_limit = row_count < 4096U ? 1U : std::min(
        row_count, static_cast<std::size_t>(zkdl::cpu_parallel_worker_count()) + 1U);
    const std::size_t grain = row_count / chunk_limit +
        (row_count % chunk_limit != 0U ? 1U : 0U);
    const std::size_t active_chunks = row_count / grain +
        (row_count % grain != 0U ? 1U : 0U);
    std::vector<std::vector<zkdl_blst::blst_fr>> row_states(
        active_chunks, std::vector<zkdl_blst::blst_fr>(3U * current.size()));
    std::vector<std::vector<zkdl_blst::blst_fr>> partials(
        active_chunks, std::vector<zkdl_blst::blst_fr>(plan.degree + 1U));
    evaluations.assign(plan.degree + 1U, FrWire{});
    zkdl::cpu_parallel_for(row_count, grain, [&](std::size_t local_row) {
        const std::size_t chunk = local_row / grain;
        auto& state = row_states[chunk];
        auto& partial = partials[chunk];
        for (std::size_t tensor = 0U; tensor < current.size(); ++tensor) {
            const auto& left = current[tensor][2U * local_row];
            const auto& right = current[tensor][2U * local_row + 1U];
            state[tensor] = left;
            zkdl_blst::blst_fr_sub(&state[current.size() + tensor], &right, &left);
        }
        for (std::uint32_t evaluation = 0U; evaluation <= plan.degree; ++evaluation) {
            const auto& x = plan.evaluation_points[evaluation];
            for (std::size_t tensor = 0U; tensor < current.size(); ++tensor) {
                zkdl_blst::blst_fr scaled{};
                zkdl_blst::blst_fr_mul(
                    &scaled, &x, &state[current.size() + tensor]);
                zkdl_blst::blst_fr_add(
                    &state[2U * current.size() + tensor], &state[tensor], &scaled);
            }
            const auto value = zksc_eval_constraint_montgomery(
                plan, state.data() + 2U * current.size());
            zkdl_blst::blst_fr next{};
            zkdl_blst::blst_fr_add(&next, &partial[evaluation], &value);
            partial[evaluation] = next;
        }
    });
    for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk)
        for (std::uint32_t evaluation = 0U; evaluation <= plan.degree; ++evaluation)
            evaluations[evaluation] = zkdl::fr_add(
                evaluations[evaluation],
                zkdl::cpu_ipa_montgomery_to_frwire(partials[chunk][evaluation]));
}

void check_factorized_timing(std::size_t row_count) {
    constexpr std::uint32_t kDegree = 4U;
    const auto descriptor = make_factorized_descriptor(false);
    const auto wire_values = make_values(row_count);
    ZkSumcheckMontgomeryConstraintPlan plan;
    std::string error;
    require(zksc_compile_montgomery_constraint_plan(
                descriptor, wire_values.size(), kDegree, plan, &error),
            "benchmark Montgomery plan compile failed: " + error);
    require(plan.fast_coefficients,
            "benchmark relation did not select the specialized coefficient path");

    ZkSumcheckFoldedState cached(wire_values.size());
    for (std::size_t tensor = 0U; tensor < wire_values.size(); ++tensor) {
        cached[tensor].reserve(wire_values[tensor].size());
        for (const auto& value : wire_values[tensor])
            cached[tensor].push_back(
                zkdl::cpu_ipa_frwire_to_montgomery_unchecked(value));
    }
    const std::size_t chunks = row_count < 4096U ? 1U : std::min(
        row_count, static_cast<std::size_t>(zkdl::cpu_parallel_worker_count()) + 1U);

    const auto run_new_wire = [&]() {
        ZkSumcheckRoundEvaluationScratch scratch(
            chunks, wire_values.size(), kDegree + 1U, true, true);
        std::vector<FrWire> output(kDegree + 1U);
        require(zksc_accumulate_round_evaluation_rows(
                    descriptor, wire_values, kDegree, 0U, row_count,
                    output, scratch, &error, &plan),
                "optimized provider-row benchmark call failed: " + error);
        return output;
    };
    const auto run_new_cached = [&]() {
        ZkSumcheckRoundEvaluationScratch scratch(
            chunks, wire_values.size(), kDegree + 1U, true, true);
        std::vector<FrWire> output(kDegree + 1U);
        require(zksc_accumulate_montgomery_folded_round_evaluation_rows(
                    descriptor, cached, kDegree, 0U, row_count,
                    output, scratch, &error, &plan),
                "optimized cached-row benchmark call failed: " + error);
        return output;
    };

    const auto old_wire_warm = [&]() {
        std::vector<FrWire> output;
        legacy_montgomery_wire_row_loop(wire_values, plan, output);
        return output;
    };
    const auto old_cached_warm = [&]() {
        std::vector<FrWire> output;
        legacy_montgomery_cached_row_loop(cached, plan, output);
        return output;
    };
    require_same_wire(old_wire_warm(), run_new_wire(),
                      "legacy/new provider-row warmup");
    require_same_wire(old_cached_warm(), run_new_cached(),
                      "legacy/new cached-row warmup");

    std::vector<FrWire> old_wire, new_wire, old_cached, new_cached;
    const auto old_wire_start = std::chrono::steady_clock::now();
    legacy_montgomery_wire_row_loop(wire_values, plan, old_wire);
    const auto old_wire_end = std::chrono::steady_clock::now();
    const auto new_wire_start = std::chrono::steady_clock::now();
    new_wire = run_new_wire();
    const auto new_wire_end = std::chrono::steady_clock::now();
    const auto old_cached_start = std::chrono::steady_clock::now();
    legacy_montgomery_cached_row_loop(cached, plan, old_cached);
    const auto old_cached_end = std::chrono::steady_clock::now();
    const auto new_cached_start = std::chrono::steady_clock::now();
    new_cached = run_new_cached();
    const auto new_cached_end = std::chrono::steady_clock::now();
    require_same_wire(old_wire, new_wire, "legacy/new provider-row timing equality");
    require_same_wire(old_cached, new_cached, "legacy/new cached-row timing equality");
    const auto milliseconds = [](auto elapsed) {
        return std::chrono::duration<double, std::milli>(elapsed).count();
    };
    const double old_wire_ms = milliseconds(old_wire_end - old_wire_start);
    const double new_wire_ms = milliseconds(new_wire_end - new_wire_start);
    const double old_cached_ms = milliseconds(old_cached_end - old_cached_start);
    const double new_cached_ms = milliseconds(new_cached_end - new_cached_start);
    std::cout << "sumcheck_coeff_timing rows=" << row_count
              << " provider_legacy_ms=" << old_wire_ms
              << " provider_fast_ms=" << new_wire_ms
              << " provider_speedup=" << old_wire_ms / new_wire_ms
              << " cached_legacy_ms=" << old_cached_ms
              << " cached_fast_ms=" << new_cached_ms
              << " cached_speedup=" << old_cached_ms / new_cached_ms << '\n';
}

void check_factorized_plan_case(bool inverse_relation, bool reverse_sign,
                                bool custom_evaluation_points) {
    constexpr std::uint32_t kDegree = 4U;
    constexpr std::size_t kRows = 8192U;
    const auto descriptor = make_factorized_descriptor(
        inverse_relation, reverse_sign);
    const auto values = make_values(kRows + 7U);
    ZkSumcheckMontgomeryConstraintPlan plan;
    std::string error;
    require(zksc_compile_montgomery_constraint_plan(
                descriptor, values.size(), kDegree, plan, &error),
            "factorized Montgomery plan compile failed: " + error);
    require(plan.fast_coefficients &&
                plan.fast_linear_is_constant == inverse_relation,
            "factorized compiler did not select the expected residual shape");

    std::vector<FrWire> oracle_points;
    if (custom_evaluation_points) {
        oracle_points = {zkdl::fr_from_u32(11U), zkdl::fr_from_u32(3U),
                         zkdl::fr_from_u32(19U), zkdl::fr_from_u32(7U),
                         zkdl::fr_from_u32(23U)};
        for (std::size_t index = 0U; index < oracle_points.size(); ++index)
            plan.evaluation_points[index] =
                zkdl::cpu_ipa_frwire_to_montgomery_checked(oracle_points[index]);
    }

    const std::size_t row_begin = 3U;
    const std::size_t row_end = row_begin + kRows;
    const std::size_t chunks = kRows < 4096U ? 1U : std::min(
        kRows, static_cast<std::size_t>(zkdl::cpu_parallel_worker_count()) + 1U);
    ZkSumcheckRoundEvaluationScratch scratch(
        chunks, values.size(), kDegree + 1U, true, true);
    std::vector<FrWire> actual(kDegree + 1U);
    require(zksc_accumulate_round_evaluation_rows(
                descriptor, values, kDegree, row_begin, row_end,
                actual, scratch, &error, &plan),
            "factorized Montgomery row evaluator failed: " + error);
    require_same_wire(actual, old_row_oracle(
                descriptor, values, kDegree, row_begin, row_end, oracle_points),
            std::string(inverse_relation ? "inverse" : "weighted") +
                (reverse_sign ? " reversed" : "") +
                (custom_evaluation_points ? " custom-points" : "") +
                " factorized oracle");
}

void check_factorized_fallback() {
    auto descriptor = make_factorized_descriptor(false);
    descriptor.constant = zkdl::fr_from_u32(1U);
    constexpr std::uint32_t kDegree = 4U;
    const auto values = make_values(512U);
    ZkSumcheckMontgomeryConstraintPlan plan;
    std::string error;
    require(zksc_compile_montgomery_constraint_plan(
                descriptor, values.size(), kDegree, plan, &error),
            "fallback Montgomery plan compile failed: " + error);
    require(!plan.fast_coefficients,
            "nonzero-constant relation entered the specialized fast path");
    ZkSumcheckRoundEvaluationScratch scratch(
        1U, values.size(), kDegree + 1U, true, false);
    std::vector<FrWire> actual(kDegree + 1U);
    require(zksc_accumulate_round_evaluation_rows(
                descriptor, values, kDegree, 0U, 256U,
                actual, scratch, &error, &plan),
            "generic Montgomery fallback rejected a valid relation: " + error);
    require_same_wire(actual, old_row_oracle(
                descriptor, values, kDegree, 0U, 256U),
            "nonzero-constant generic fallback oracle");
}

void check_auxiliary_factorized_case(const ConstraintDescriptor& descriptor,
                                     const std::string& label,
                                     std::size_t tensor_count,
                                     std::uint32_t expected_base_degree,
                                     bool benchmark = false) {
    constexpr std::size_t kRows = 2048U;
    constexpr std::size_t kRowBegin = 3U;
    const std::uint32_t degree = descriptor.degree;
    const auto values = make_values(kRows + 7U, tensor_count);
    ZkSumcheckMontgomeryConstraintPlan plan;
    std::string error;
    require(zksc_compile_montgomery_constraint_plan(
                descriptor, values.size(), degree, plan, &error),
            label + " Montgomery plan compile failed: " + error);
    require(plan.auxiliary_factorized_coefficients && !plan.fast_coefficients &&
                plan.auxiliary_base_degree == expected_base_degree &&
                zksc_montgomery_auxiliary_coefficients_worthwhile(plan, kRows),
            label + " did not select the generic auxiliary coefficient path");

    std::vector<FrWire> oracle_points;
    for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation) {
        const FrWire point = zkdl::fr_from_u32(11U + 3U * evaluation);
        oracle_points.push_back(point);
        plan.evaluation_points[evaluation] =
            zkdl::cpu_ipa_frwire_to_montgomery_checked(point);
    }
    const std::size_t row_end = kRowBegin + kRows;
    ZkSumcheckRoundEvaluationScratch wire_scratch(
        1U, values.size(), degree + 1U, true, true);
    std::vector<FrWire> wire_actual(degree + 1U);
    require(zksc_accumulate_round_evaluation_rows(
                descriptor, values, degree, kRowBegin, row_end,
                wire_actual, wire_scratch, &error, &plan),
            label + " wire-row coefficient evaluation failed: " + error);
    const auto expected = old_row_oracle(
        descriptor, values, degree, kRowBegin, row_end, oracle_points);
    require_same_wire(wire_actual, expected, label + " wire-row old point-value oracle");

    ZkSumcheckFoldedState folded(values.size());
    for (std::size_t tensor = 0U; tensor < values.size(); ++tensor) {
        folded[tensor].reserve(values[tensor].size());
        for (const auto& value : values[tensor])
            folded[tensor].push_back(
                zkdl::cpu_ipa_frwire_to_montgomery_unchecked(value));
    }
    ZkSumcheckRoundEvaluationScratch folded_scratch(
        1U, values.size(), degree + 1U, true, true);
    std::vector<FrWire> folded_actual(degree + 1U);
    require(zksc_accumulate_montgomery_folded_round_evaluation_rows(
                descriptor, folded, degree, kRowBegin, row_end,
                folded_actual, folded_scratch, &error, &plan),
            label + " folded-row coefficient evaluation failed: " + error);
    require_same_wire(folded_actual, expected,
                      label + " folded-row old point-value oracle");

    if (!benchmark) return;
    const auto benchmark_values = make_values(kRows, tensor_count);
    std::vector<FrWire> legacy, optimized;
    const auto run_optimized = [&]() {
        ZkSumcheckRoundEvaluationScratch scratch(
            1U, benchmark_values.size(), degree + 1U, true, true);
        std::vector<FrWire> output(degree + 1U);
        std::string local_error;
        require(zksc_accumulate_round_evaluation_rows(
                    descriptor, benchmark_values, degree, 0U, kRows,
                    output, scratch, &local_error, &plan),
                label + " optimized A/B call failed: " + local_error);
        return output;
    };
    legacy_montgomery_wire_row_loop(benchmark_values, plan, legacy);
    require_same_wire(legacy, run_optimized(), label + " A/B warmup equality");
    constexpr std::size_t kSamples = 7U;
    std::vector<double> legacy_samples, coefficient_samples;
    legacy_samples.reserve(kSamples);
    coefficient_samples.reserve(kSamples);
    for (std::size_t sample = 0U; sample < kSamples; ++sample) {
        auto start = std::chrono::steady_clock::now();
        legacy_montgomery_wire_row_loop(benchmark_values, plan, legacy);
        auto end = std::chrono::steady_clock::now();
        legacy_samples.push_back(std::chrono::duration<double, std::milli>(
            end - start).count());
        start = std::chrono::steady_clock::now();
        optimized = run_optimized();
        end = std::chrono::steady_clock::now();
        coefficient_samples.push_back(std::chrono::duration<double, std::milli>(
            end - start).count());
        require_same_wire(legacy, optimized, label + " repeated A/B equality");
    }
    require_same_wire(legacy, optimized, label + " A/B equality");
    std::sort(legacy_samples.begin(), legacy_samples.end());
    std::sort(coefficient_samples.begin(), coefficient_samples.end());
    const double old_ms = legacy_samples[kSamples / 2U];
    const double new_ms = coefficient_samples[kSamples / 2U];
    std::cout << "sumcheck_aux_factor_timing case=" << label
              << " rows=" << kRows << " workers=0 samples=" << kSamples
              << " statistic=median"
              << " legacy_muls_per_row=" << plan.legacy_multiplications_per_row
              << " coefficient_muls_per_row="
              << plan.auxiliary_multiplications_per_row
              << " legacy_ms=" << old_ms << " coefficient_ms=" << new_ms
              << " speedup=" << old_ms / new_ms << '\n';
}

void check_auxiliary_factorized_fallback() {
    constexpr std::uint32_t kDegree = 4U;
    auto nonzero_constant = make_pointwise_weighted_descriptor("reset");
    nonzero_constant.constant = zkdl::fr_from_u32(1U);
    auto mismatched_auxiliary_refs = make_pointwise_weighted_descriptor("terminal");
    mismatched_auxiliary_refs.terms.front().refs.back() = 2U;
    const std::array<std::pair<ConstraintDescriptor, std::size_t>, 2U> cases{{
        {nonzero_constant, 4U},
        {mismatched_auxiliary_refs, 5U},
    }};
    for (const auto& test_case : cases) {
        const auto& descriptor = test_case.first;
        const auto values = make_values(320U, test_case.second);
        ZkSumcheckMontgomeryConstraintPlan plan;
        std::string error;
        require(zksc_compile_montgomery_constraint_plan(
                    descriptor, values.size(), kDegree, plan, &error),
                "auxiliary fallback plan compile failed: " + error);
        require(!plan.auxiliary_factorized_coefficients,
                "unsupported auxiliary descriptor entered the coefficient path");
        ZkSumcheckRoundEvaluationScratch scratch(
            1U, values.size(), kDegree + 1U, true, false);
        std::vector<FrWire> actual(kDegree + 1U);
        require(zksc_accumulate_round_evaluation_rows(
                    descriptor, values, kDegree, 0U, 256U,
                    actual, scratch, &error, &plan),
                "generic auxiliary fallback rejected a valid relation: " + error);
        require_same_wire(actual, old_row_oracle(
                    descriptor, values, kDegree, 0U, 256U),
                "generic auxiliary fallback point-value oracle");
    }
}

void check_degree_three_auxiliary_factorization() {
    const auto descriptor = make_degree_three_pointwise_descriptor();
    constexpr std::uint32_t kDegree = 3U;
    constexpr std::size_t kRows = 2048U;
    const auto values = make_values(kRows + 3U, 4U);
    ZkSumcheckMontgomeryConstraintPlan plan;
    std::string error;
    require(zksc_compile_montgomery_constraint_plan(
                descriptor, values.size(), kDegree, plan, &error),
            "degree-three auxiliary plan compile failed: " + error);
    require(plan.auxiliary_factorized_coefficients &&
                plan.auxiliary_base_degree == 1U &&
                zksc_montgomery_auxiliary_coefficients_worthwhile(plan, kRows),
            "degree-three relation did not select auxiliary coefficient mode");
    ZkSumcheckRoundEvaluationScratch scratch(
        1U, values.size(), kDegree + 1U, true, true);
    std::vector<FrWire> actual(kDegree + 1U);
    require(zksc_accumulate_round_evaluation_rows(
                descriptor, values, kDegree, 0U, kRows,
                actual, scratch, &error, &plan),
            "degree-three auxiliary row evaluation failed: " + error);
    require_same_wire(actual, old_row_oracle(
                descriptor, values, kDegree, 0U, kRows),
            "degree-three auxiliary point-value oracle");
}
#endif

void require_same_wire(const std::vector<FrWire>& actual,
                       const std::vector<FrWire>& expected,
                       const std::string& label) {
    require(actual.size() == expected.size(), label + ": evaluation count mismatch");
    for (std::size_t index = 0U; index < actual.size(); ++index)
        require(actual[index] == expected[index],
                label + ": FrWire mismatch at evaluation " + std::to_string(index));
}

#if defined(ZKDL_USE_BLST_G1_CODEC)
std::vector<ZkSumcheckFoldedScalar> make_fold_values(std::size_t count) {
    std::array<ZkSumcheckFoldedScalar, 257U> palette{};
    for (std::size_t index = 0U; index < palette.size(); ++index) {
        const std::uint32_t scalar = static_cast<std::uint32_t>(
            (index * 7919U + 1237U) % 1000003U + 1U);
        palette[index] = zkdl::cpu_ipa_frwire_to_montgomery_unchecked(
            zkdl::fr_from_u32(scalar));
    }
    std::vector<ZkSumcheckFoldedScalar> values(count);
    for (std::size_t index = 0U; index < count; ++index)
        values[index] = palette[(index * 73U + 19U) % palette.size()];
    return values;
}

void legacy_serial_fold(std::vector<ZkSumcheckFoldedScalar>& values,
                        const FrWire& challenge,
                        std::uint32_t progress_round = 0U,
                        std::uint32_t tensor_index = 0U,
                        const ZkSumcheckProgressCallback* progress = nullptr) {
    const std::size_t folded_size = values.size() / 2U;
    constexpr std::size_t kProgressRows = 1U << 18U;
    const auto challenge_montgomery =
        zkdl::cpu_ipa_frwire_to_montgomery_unchecked(challenge);
    for (std::size_t begin = 0U; begin < folded_size;) {
        const std::size_t end = progress == nullptr
            ? folded_size : std::min(folded_size, begin + kProgressRows);
        for (std::size_t index = begin; index < end; ++index) {
            const auto left = values[2U * index];
            const auto right = values[2U * index + 1U];
            zkdl_blst::blst_fr difference{}, scaled{}, folded{};
            zkdl_blst::blst_fr_sub(&difference, &right, &left);
            zkdl_blst::blst_fr_mul(&scaled, &challenge_montgomery, &difference);
            zkdl_blst::blst_fr_add(&folded, &left, &scaled);
            values[index] = folded;
        }
        begin = end;
        if (progress != nullptr)
            (*progress)("fold_state_rows", progress_round, tensor_index,
                        end, folded_size);
    }
    values.resize(folded_size);
}

void require_same_folded_values(
    const std::vector<ZkSumcheckFoldedScalar>& actual,
    const std::vector<ZkSumcheckFoldedScalar>& expected,
    const std::string& label) {
    require(actual.size() == expected.size(), label + ": folded size mismatch");
    for (std::size_t index = 0U; index < actual.size(); ++index)
        require(zksc_folded_scalar_to_wire(actual[index]) ==
                    zksc_folded_scalar_to_wire(expected[index]),
                label + ": field mismatch at index " + std::to_string(index));
}

void check_fold_boundaries_and_challenges() {
    const std::array<std::size_t, 10U> folded_sizes = {
        1U, 4095U, 4096U, 4097U, 16383U, 16384U, 16385U,
        32767U, 32768U, 32769U};
    const std::array<FrWire, 3U> challenges = {
        zkdl::fr_from_u32(0U), zkdl::fr_from_u32(1U),
        zkdl::fr_from_u32(0x6a09e667U)};
    for (const auto folded_size : folded_sizes) {
        const auto input = make_fold_values(2U * folded_size);
        for (const auto& challenge : challenges) {
            auto expected = input;
            auto actual = input;
            legacy_serial_fold(expected, challenge);
            std::string error;
            require(zksc_fold_values_in_place(actual, challenge, &error),
                    "BLST fold rejected a valid boundary input: " + error);
            require_same_folded_values(
                actual, expected,
                "BLST fold boundary=" + std::to_string(folded_size));
        }
    }

    std::vector<ZkSumcheckFoldedScalar> state = make_fold_values(1U << 12U);
    const std::array<FrWire, 12U> round_challenges = {
        zkdl::fr_from_u32(0U), zkdl::fr_from_u32(1U),
        zkdl::fr_from_u32(0x6a09e667U), zkdl::fr_from_u32(0x3c6ef372U),
        zkdl::fr_from_u32(0xa54ff53aU), zkdl::fr_from_u32(0x510e527fU),
        zkdl::fr_from_u32(0x9b05688cU), zkdl::fr_from_u32(0x1f83d9abU),
        zkdl::fr_from_u32(0x5be0cd19U), zkdl::fr_from_u32(0x428a2f98U),
        zkdl::fr_from_u32(0x71374491U), zkdl::fr_from_u32(0xb5c0fbcfU)};
    auto oracle_state = state;
    for (std::size_t round = 0U; round < round_challenges.size(); ++round) {
        legacy_serial_fold(oracle_state, round_challenges[round]);
        std::string error;
        require(zksc_fold_values_in_place(
                    state, round_challenges[round], &error,
                    static_cast<std::uint32_t>(round + 1U)),
                "BLST multi-round fold failed: " + error);
        require_same_folded_values(
            state, oracle_state,
            "BLST multi-round state round=" + std::to_string(round));
    }
}

void check_fold_failclosed_and_progress() {
    const FrWire challenge = zkdl::fr_from_u32(17U);
    std::string error;
    for (const std::size_t count : {0U, 1U, 3U}) {
        std::vector<ZkSumcheckFoldedScalar> invalid(count);
        error.clear();
        require(!zksc_fold_values_in_place(invalid, challenge, &error) &&
                    error == "segmented ZkSumcheck in-place fold shape is invalid",
                "BLST fold changed its invalid-shape rejection");
    }
    FrWire noncanonical{};
    noncanonical.fill(std::numeric_limits<std::uint32_t>::max());
    auto invalid_challenge_values = make_fold_values(4U);
    error.clear();
    require(!zksc_fold_values_in_place(
                invalid_challenge_values, noncanonical, &error) &&
                error == "segmented ZkSumcheck in-place fold shape is invalid",
            "BLST fold changed its non-canonical challenge rejection");

    constexpr std::size_t kFoldedSize = (1U << 18U) + 3U;
    constexpr std::uint32_t kRound = 9U;
    constexpr std::uint32_t kTensor = 4U;
    const auto input = make_fold_values(2U * kFoldedSize);
    auto folded = input;
    struct ProgressEvent {
        std::string phase;
        std::uint32_t round;
        std::uint32_t tensor;
        std::uint64_t completed;
        std::uint64_t total;
    };
    std::vector<ProgressEvent> events;
    const auto caller = std::this_thread::get_id();
    bool callback_on_caller = true;
    const ZkSumcheckProgressCallback progress =
        [&](const char* phase, std::uint32_t round, std::uint32_t tensor,
            std::uint64_t completed, std::uint64_t total) {
            callback_on_caller &= std::this_thread::get_id() == caller;
            events.push_back({phase, round, tensor, completed, total});
        };
    error.clear();
    require(zksc_fold_values_in_place(
                folded, challenge, &error, kRound, kTensor, &progress),
            "BLST fold with progress callback failed: " + error);
    require(callback_on_caller && events.size() == 2U &&
                events[0U].phase == "fold_state_rows" &&
                events[0U].round == kRound && events[0U].tensor == kTensor &&
                events[0U].completed == (1U << 18U) &&
                events[0U].total == kFoldedSize &&
                events[1U].phase == "fold_state_rows" &&
                events[1U].round == kRound && events[1U].tensor == kTensor &&
                events[1U].completed == kFoldedSize &&
                events[1U].total == kFoldedSize,
            "BLST fold changed progress callback boundaries or thread");

    auto throwing_fold = input;
    std::size_t callback_count = 0U;
    const ZkSumcheckProgressCallback throwing_progress =
        [&](const char*, std::uint32_t, std::uint32_t,
            std::uint64_t, std::uint64_t) {
            ++callback_count;
            throw std::runtime_error("fold progress fixture exception");
        };
    bool callback_exception_propagated = false;
    try {
        (void)zksc_fold_values_in_place(
            throwing_fold, challenge, &error, kRound, kTensor,
            &throwing_progress);
    } catch (const std::runtime_error& exception) {
        callback_exception_propagated =
            std::string(exception.what()) == "fold progress fixture exception";
    }
    require(callback_exception_propagated && callback_count == 1U &&
                throwing_fold.size() == input.size(),
            "BLST fold changed callback exception propagation");
}

void check_fold_timing(std::size_t element_count) {
    require(element_count >= 32768U && (element_count & 1U) == 0U,
            "BLST fold benchmark requires a large even element count");
    const auto input = make_fold_values(element_count);
    const FrWire challenge = zkdl::fr_from_u32(0x6a09e667U);
    auto oracle = input;
    legacy_serial_fold(oracle, challenge);
    auto optimized = input;
    std::string error;
    require(zksc_fold_values_in_place(optimized, challenge, &error),
            "BLST fold benchmark warmup failed: " + error);
    require_same_folded_values(optimized, oracle,
                               "BLST fold benchmark warmup");

    std::vector<double> serial_ms;
    std::vector<double> wavefront_ms;
    serial_ms.reserve(5U);
    wavefront_ms.reserve(5U);
    for (std::size_t sample = 0U; sample < 5U; ++sample) {
        auto serial = input;
        auto start = std::chrono::steady_clock::now();
        legacy_serial_fold(serial, challenge);
        auto stop = std::chrono::steady_clock::now();
        serial_ms.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
        require_same_folded_values(serial, oracle,
                                   "BLST serial fold A/B sample");

        auto wavefront = input;
        start = std::chrono::steady_clock::now();
        error.clear();
        require(zksc_fold_values_in_place(wavefront, challenge, &error),
                "BLST wavefront fold A/B failed: " + error);
        stop = std::chrono::steady_clock::now();
        wavefront_ms.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
        require_same_folded_values(wavefront, oracle,
                                   "BLST wavefront fold A/B sample");
    }
    std::sort(serial_ms.begin(), serial_ms.end());
    std::sort(wavefront_ms.begin(), wavefront_ms.end());
    std::cout << "sumcheck_fold_timing elements=" << element_count
              << " serial_median_ms=" << serial_ms[2U]
              << " wavefront_median_ms=" << wavefront_ms[2U]
              << " speedup=" << serial_ms[2U] / wavefront_ms[2U]
              << " samples=5 cpu_affinity=external\n";
}
#endif

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

void check_pointwise_weighted_sumcheck_roundtrip() {
    constexpr std::uint32_t kDomain = 2048U;
    constexpr std::uint32_t kBasis = 2048U;
    constexpr std::uint32_t kDegree = 4U;
    const auto descriptor = make_pointwise_weighted_descriptor("recurrence");
    const auto values = make_values(kDomain / 2U, 6U);
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
    for (std::uint32_t index = 0U; index < kBasis; ++index)
        generators.push_back(zkdl::cpu_g1_mul(
            base, zkdl::fr_from_u32(index + 2U)));
    const auto h = zkdl::cpu_g1_mul(base, zkdl::fr_from_u32(37U));
    const auto u = zkdl::cpu_g1_mul(base, zkdl::fr_from_u32(41U));
    RelationStatement statement;
    statement.statement_digest = zkdl::sha256(std::vector<zkdl::byte>{'s'});
    statement.model_digest = zkdl::sha256(std::vector<zkdl::byte>{'m'});
    statement.table_digest = zkdl::sha256(std::vector<zkdl::byte>{'t'});
    statement.ledger_digest = zkdl::sha256(std::vector<zkdl::byte>{'l'});
    statement.registry_digest = zkdl::sha256(std::vector<zkdl::byte>{'r'});
    statement.srs_digest = zkdl::sha256(std::vector<zkdl::byte>{'p'});

    const std::string transcript_domain = "sumcheck-pointwise-weighted-v1";
    zkdl::Transcript prover_transcript(transcript_domain);
    ZkSumcheckProverResult prover;
    std::string error;
    require(prove_zk_sumcheck(
                descriptor, values, kDegree, kDomain, generators, h, u,
                initial_claim, statement, "oracle/pointwise-weighted",
                prover_transcript, prover, &error, nullptr, nullptr,
                true, true),
            "weighted degree-four Sumcheck prover failed: " + error);
    require(prover.proof.domain_size == kDomain &&
                prover.proof.segment_size == kBasis &&
                prover.proof.segment_count == 1U,
            "weighted degree-four Sumcheck did not use one full-domain segment");
    require(prover.proof.rounds.size() == 11U,
            "weighted degree-four Sumcheck emitted the wrong round count");

    zkdl::Transcript verifier_transcript(transcript_domain);
    ZkSumcheckVerifierResult verifier;
    require(verify_zk_sumcheck(
                prover.proof, descriptor, kDegree, kDomain, generators, h, u,
                initial_claim, statement, "oracle/pointwise-weighted",
                verifier_transcript, verifier, &error),
            "fresh weighted degree-four Sumcheck Verify failed: " + error);
    require(fr_compare_mixed(
                prover_transcript.challenge(
                    "sumcheck-pointwise-weighted/followup"),
                verifier_transcript.challenge(
                    "sumcheck-pointwise-weighted/followup")) == 0,
            "weighted Sumcheck prover/verifier transcript replay diverged");

    auto tampered = prover.proof;
    tampered.rounds[0U].evaluation_commitment = G1Jacobian_t{};
    zkdl::Transcript tampered_transcript(transcript_domain);
    ZkSumcheckVerifierResult rejected;
    error.clear();
    require(!verify_zk_sumcheck(
                tampered, descriptor, kDegree, kDomain, generators, h, u,
                initial_claim, statement, "oracle/pointwise-weighted",
                tampered_transcript, rejected, &error),
            "fresh weighted Sumcheck Verify accepted a tampered evaluation commitment");
}

void check_spilled_sumcheck_round_differential(std::uint32_t domain = 4096U) {
    require(domain >= 2U && (domain & (domain - 1U)) == 0U,
            "spill differential domain must be a power of two");
    const std::uint32_t kDomain = domain;
    const std::uint32_t kBasis = domain;
    constexpr std::uint32_t kDegree = 4U;
    struct PrivateTempDirectory {
        std::filesystem::path path;
        ~PrivateTempDirectory() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } temporary;
    std::string pattern =
        (std::filesystem::current_path() / "tests" / ".sumcheck-spill-XXXXXX").string();
    std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
    mutable_pattern.push_back('\0');
    const char* created = ::mkdtemp(mutable_pattern.data());
    require(created != nullptr, "could not create the private Sumcheck spill fixture directory");
    temporary.path = created;

    const auto descriptor = make_pointwise_weighted_descriptor("recurrence");
    const auto values = make_values(kDomain / 2U, 6U);
    std::string error;
    ZkSumcheckMontgomeryConstraintPlan montgomery_plan;
#if defined(ZKDL_USE_BLST_G1_CODEC)
    require(zksc_compile_montgomery_constraint_plan(
                descriptor, values.size(), kDegree, montgomery_plan, &error),
            "spill differential Montgomery plan compile failed: " + error);
    require(montgomery_plan.auxiliary_factorized_coefficients,
            "spill differential did not select the weighted auxiliary path");
#endif
    const ZkSumcheckMontgomeryConstraintPlan* montgomery_plan_ptr = nullptr;
#if defined(ZKDL_USE_BLST_G1_CODEC)
    montgomery_plan_ptr = &montgomery_plan;
#endif
    std::vector<SegmentedTensorProvider> providers;
    require(make_segmented_tensor_providers(values, kBasis, providers, &error),
            "could not create spill differential providers: " + error);

    ZkSumcheckFoldedState memory_state;
    ZkSumcheckFoldedSpillFile spilled_state;
    std::vector<FrWire> prior_challenges;
    zkdl::Transcript memory_transcript("sumcheck-forced-spill-differential-v1");
    zkdl::Transcript spill_transcript("sumcheck-forced-spill-differential-v1");
    const std::string operation = "sumcheck/forced-spill-differential";
    const std::uint32_t rounds = log2_exact(kDomain);
    for (std::uint32_t round = 0U; round < rounds; ++round) {
        std::vector<FrWire> memory_evaluations;
        std::vector<FrWire> spilled_evaluations;
        if (round == 0U) {
            require(zksc_segmented_round_evaluations(
                        descriptor, providers, kDegree, round, prior_challenges,
                        memory_evaluations, &error, nullptr, montgomery_plan_ptr),
                    "memory Sumcheck round-zero evaluation failed: " + error);
            spilled_evaluations = memory_evaluations;
        } else {
            require(zksc_folded_round_evaluations(
                        descriptor, memory_state, kDegree, memory_evaluations,
                        &error, round, nullptr, montgomery_plan_ptr),
                    "memory folded-state evaluation failed: " + error);
            require(zksc_spilled_round_evaluations(
                        descriptor, spilled_state, kDegree, spilled_evaluations,
                        &error, round, nullptr, montgomery_plan_ptr),
                    "disk folded-state evaluation failed: " + error);
        }
        require_same_wire(spilled_evaluations, memory_evaluations,
                          "forced spill evaluations round=" + std::to_string(round));

        std::vector<Fr_t> memory_values, spill_values;
        memory_values.reserve(memory_evaluations.size());
        spill_values.reserve(spilled_evaluations.size());
        for (const auto& value : memory_evaluations)
            memory_values.push_back(zkdl::fr_value(value));
        for (const auto& value : spilled_evaluations)
            spill_values.push_back(zkdl::fr_value(value));
        const std::string evaluation_label = "sumcheck.round." + std::to_string(round) + ".evaluations";
        const std::string challenge_label = "sumcheck.round." + std::to_string(round) + ".challenge";
        memory_transcript.absorb_fr(evaluation_label, memory_values);
        spill_transcript.absorb_fr(evaluation_label, spill_values);
        const FrWire memory_challenge = zkdl::fr_wire(memory_transcript.challenge(challenge_label));
        const FrWire spill_challenge = zkdl::fr_wire(spill_transcript.challenge(challenge_label));
        require(memory_challenge == spill_challenge,
                "forced spill transcript challenge differs at round " + std::to_string(round));

        if (round == 0U) {
            require(zksc_segmented_fold_state_initialize(
                        providers, memory_challenge, memory_state, &error,
                        round + 1U),
                    "memory folded-state initialization failed: " + error);
            require(zksc_spilled_fold_state_initialize(
                        providers, spill_challenge, operation, temporary.path.string(),
                        round + 1U, spilled_state, &error),
                    "disk folded-state initialization failed: " + error);
        } else {
            for (auto& tensor : memory_state)
                require(zksc_fold_values_in_place(tensor, memory_challenge, &error,
                                                   round + 1U),
                        "memory folded-state round failed: " + error);
            require(zksc_spilled_fold_state(
                        spill_challenge, spilled_state, round + 1U, &error),
                    "disk folded-state round failed: " + error);
        }
        prior_challenges.push_back(memory_challenge);
    }

    std::vector<FrWire> memory_terminal, spill_terminal, provider_terminal;
    memory_terminal.reserve(memory_state.size());
    for (const auto& tensor : memory_state) {
        require(tensor.size() == 1U, "memory terminal has the wrong folded shape");
        memory_terminal.push_back(zksc_folded_scalar_to_wire(tensor.front()));
    }
    require(zksc_spilled_terminal_values(
                spilled_state, rounds, spill_terminal, &error),
            "disk Sumcheck terminal read failed: " + error);
    require(zksc_segmented_terminal_values(
                providers, prior_challenges, provider_terminal, &error),
            "provider Sumcheck terminal evaluation failed: " + error);
    require_same_wire(spill_terminal, memory_terminal,
                      "forced spill terminal memory equality");
    require_same_wire(spill_terminal, provider_terminal,
                      "forced spill terminal provider equality");
    require(memory_transcript.digest() == spill_transcript.digest(),
            "forced spill final transcript bytes differ");

    ZkSumcheckFoldedSpillFile malformed;
    require(ZkSumcheckFoldedSpillFile::create(
                temporary.path.string(), "sumcheck/noncanonical-fixture", 4U,
                1U, 1U, 0U, malformed, &error),
            "could not create malformed spill rejection fixture: " + error);
    FrWire noncanonical{};
    noncanonical.fill(std::numeric_limits<std::uint32_t>::max());
    error.clear();
    require(malformed.append_values(&noncanonical, 1U, &error) &&
                malformed.finish(&error),
            "could not complete the malformed spill input fixture: " + error);
    FrWire decoded{};
    error.clear();
    require(!malformed.read_values(
                0U, 1U, &decoded, 1U, "sumcheck/noncanonical-fixture", 4U,
                1U, 1U, 0U, &error) && !error.empty(),
            "disk spill reader accepted a non-canonical FrWire value");
}

void check_full_routes(std::uint32_t degree) {
    constexpr std::size_t kRows = 8192U;
    constexpr std::size_t kBasis = 8192U;
    const auto descriptor = make_descriptor(degree);
    const auto values = make_values(kRows);
    ZkSumcheckFoldedState folded_input(values.size());
#if defined(ZKDL_USE_BLST_G1_CODEC)
    for (std::size_t tensor = 0U; tensor < values.size(); ++tensor) {
        folded_input[tensor].reserve(values[tensor].size());
        for (const auto& value : values[tensor])
            folded_input[tensor].push_back(
                zkdl::cpu_ipa_frwire_to_montgomery_unchecked(value));
    }
#else
    folded_input = values;
#endif
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
                descriptor, folded_input, degree, folded, &error),
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
                descriptor, folded_input, degree, repeated, &error),
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

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--pointwise-factor-only") {
#if defined(ZKDL_USE_BLST_G1_CODEC)
            report_stage("pointwise_factor_oracle_start");
            check_auxiliary_factorized_case(
                make_pointwise_weighted_descriptor("recurrence"),
                "recurrence", 6U, 2U, true);
            check_auxiliary_factorized_case(
                make_pointwise_weighted_descriptor("reset"),
                "reset", 4U, 2U, true);
            check_auxiliary_factorized_case(
                make_pointwise_weighted_descriptor("terminal"),
                "terminal", 5U, 2U, true);
            check_auxiliary_factorized_case(
                make_pointwise_weighted_descriptor("unit-repeated"),
                "unit-repeated", 4U, 2U);
            check_auxiliary_factorized_case(
                make_pointwise_weighted_descriptor("zero-arity"),
                "zero-arity", 3U, 1U);
            check_degree_three_auxiliary_factorization();
            check_auxiliary_factorized_fallback();
            check_pointwise_weighted_sumcheck_roundtrip();
            report_stage("fold_wavefront_oracle_start");
            check_fold_boundaries_and_challenges();
            check_fold_failclosed_and_progress();
            check_fold_timing(1U << 18U);
            check_fold_timing(1U << 20U);
            report_stage("fold_wavefront_oracle_done");
            check_spilled_sumcheck_round_differential(131072U);
            std::cout << "sumcheck_pointwise_factor_check=PASS "
                         "weighted_degree3_4=PASS wire_folded=PASS "
                         "arbitrary_coefficients=PASS repeated_refs=PASS "
                         "zero_arity_constant=PASS fallback=PASS "
                         "fresh_verify_tamper=PASS "
                         "fold_wavefront_oracle=PASS multi_round=PASS "
                         "progress_main_thread=PASS "
                         "ab_rows=2048 serial_median7=PASS "
                         "actual_domain=2048 segment=2048 basis=2048 "
                         "parallel_spill_domain=131072 spill=PASS\n";
            return 0;
#else
            throw std::runtime_error(
                "pointwise factor check requires the BLST Montgomery backend");
#endif
        }
        if (argc > 1)
            throw std::invalid_argument("unknown sumcheck oracle check option");
        report_stage("threshold_4095_start");
        check_threshold_range(2U, 4095U, 17U);
        report_stage("threshold_4095_done");
        report_stage("threshold_4096_start");
        check_threshold_range(4U, 4096U, 29U);
        report_stage("threshold_4096_done");
        report_stage("scratch_shape_start");
        check_scratch_shape_failclosed();
        report_stage("scratch_shape_done");
        report_stage("forced_spill_differential_start");
        check_spilled_sumcheck_round_differential();
        report_stage("forced_spill_differential_done");
#if defined(ZKDL_USE_BLST_G1_CODEC)
        report_stage("factorized_oracle_start");
        check_factorized_plan_case(false, false, false);
        check_factorized_plan_case(true, false, false);
        check_factorized_plan_case(true, true, false);
        check_factorized_plan_case(false, true, true);
        check_factorized_fallback();
        report_stage("factorized_oracle_done");
        report_stage("factorized_timing_start");
        check_factorized_timing(32768U);
        check_factorized_timing(262144U);
        report_stage("factorized_timing_done");
#endif
        check_tiny_sumcheck_proof_roundtrip();
        check_full_routes(2U);
        check_full_routes(4U);
        std::cout << "sumcheck_round_oracle_check=PASS threshold=4095/4096 "
                     "nonzero_row_begin=PASS degrees=2/4 "
                     "provider/cache_wire_equality=PASS tiny_proof_verify=PASS "
                     "forced_spill_round_transcript_terminal=PASS "
                     "factorized_weighted_inverse_signs_custom_points_fallback=PASS "
                     "factorized_timing=synthetic-only "
                     "scratch_shape_failclosed=PASS shape_failclosed=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sumcheck_round_oracle_check=FAIL error="
                  << error.what() << '\n';
        return 1;
    }
}
