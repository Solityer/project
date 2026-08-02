#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/protocol/proof.hpp"
#include "gatzk/protocol/quotients.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using gatzk::algebra::FieldElement;

double elapsed_ms(const Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
}

std::vector<mcl::Fr> make_input(std::size_t size) {
    std::vector<mcl::Fr> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = FieldElement(
            19 + ((i * 1315423911ULL) % 104729ULL)).native();
    }
    return values;
}

std::vector<mcl::Fr> split_intt_serial(
    const std::vector<mcl::Fr>& input) {
    if (input.size() < 2 || (input.size() & (input.size() - 1U)) != 0U) {
        throw std::runtime_error("split INTT requires a power-of-two size >= 2");
    }
    const auto half_size = input.size() / 2;
    std::vector<mcl::Fr> even(half_size);
    std::vector<mcl::Fr> odd(half_size);
    for (std::size_t i = 0; i < half_size; ++i) {
        even[i] = input[2 * i];
        odd[i] = input[2 * i + 1];
    }
    const auto plan = gatzk::algebra::ntt_plan(half_size);
    plan->intt(even.data());
    plan->intt(odd.data());

    std::vector<mcl::Fr> output(input.size());
    const auto omega_inverse =
        FieldElement::root_of_unity(input.size()).inv().native();
    const auto inverse_two = FieldElement(2).inv().native();
    mcl::Fr omega_power;
    omega_power = 1;
    for (std::size_t i = 0; i < half_size; ++i) {
        mcl::Fr odd_scaled;
        mcl::Fr sum;
        mcl::Fr difference;
        mcl::Fr::mul(odd_scaled, odd[i], omega_power);
        mcl::Fr::add(sum, even[i], odd_scaled);
        mcl::Fr::sub(difference, even[i], odd_scaled);
        mcl::Fr::mul(output[i], sum, inverse_two);
        mcl::Fr::mul(output[i + half_size], difference, inverse_two);
        mcl::Fr::mul(omega_power, omega_power, omega_inverse);
    }
    return output;
}

std::vector<FieldElement> run_fh_zero_division_stage(
    const gatzk::protocol::ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    std::size_t point_count,
    bool use_cached_inverse) {
    const std::array<FieldElement, 4> zeros = {
        FieldElement(7), FieldElement(11),
        FieldElement(13), FieldElement(17)};
    const std::array<FieldElement, 4> inverses = {
        zeros[0].inv(), zeros[1].inv(), zeros[2].inv(), zeros[3].inv()};
    std::vector<FieldElement> output(point_count);
    const auto workers = std::min<std::size_t>(
        32, std::max<std::size_t>(1, std::thread::hardware_concurrency()));
    const auto chunk = (point_count + workers - 1) / workers;
    std::vector<std::future<void>> futures;
    futures.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
        const auto begin = worker * chunk;
        const auto end = std::min(point_count, begin + chunk);
        if (begin >= end) {
            break;
        }
        futures.push_back(std::async(std::launch::async, [&, begin, end] {
            for (std::size_t index = begin; index < end; ++index) {
                const auto residue = index & 3U;
                const auto z = FieldElement(101 + index);
                const gatzk::protocol::QuotientBoundaryValues boundary{
                    .first = FieldElement(19),
                    .last = FieldElement(23),
                    .zero = zeros[residue],
                    .zero_inverse = use_cached_inverse
                        ? std::optional<FieldElement>(inverses[residue])
                        : std::nullopt,
                };
                output[index] = gatzk::protocol::evaluate_t_fh(
                    context,
                    challenges,
                    [](const std::string& name, const FieldElement& point) {
                        return point * FieldElement(name.size() + 3);
                    },
                    z,
                    nullptr,
                    &boundary,
                    true);
            }
        }));
    }
    for (auto& future : futures) {
        future.get();
    }
    return output;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto size = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1]))
            : std::size_t{131072};
        const auto repeats = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2]))
            : std::size_t{3};
        const auto batch_objects = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3]))
            : std::size_t{0};
        const auto batch_workers = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4]))
            : std::size_t{0};
        const std::string batch_mode = argc > 5 ? argv[5] : "production";
        if (argc > 6 || repeats == 0 || size < 2
            || ((batch_objects == 0) != (batch_workers == 0))
            || (batch_mode != "production" && batch_mode != "serial")
            || (size & (size - 1U)) != 0U) {
            throw std::runtime_error(
                "usage: gatzk_quotient_transform_stage_bench [power_of_two_size] [repeats] [batch_objects batch_workers [production|serial]]");
        }
        const auto input = make_input(size);
        auto second_input = make_input(size);
        if (!second_input.empty()) {
            mcl::Fr::add(
                second_input.front(),
                second_input.front(),
                FieldElement(37).native());
        }
        std::vector<double> serial_samples;
        std::vector<double> forward_pair_serial_samples;
        std::vector<double> forward_pair_parallel_samples;
        std::vector<double> split_serial_samples;
        std::vector<double> split_parallel_samples;
        std::vector<double> power_serial_samples;
        std::vector<double> power_parallel_samples;
        std::vector<double> rotation_miss_samples;
        std::vector<double> rotation_hit_samples;
        std::vector<double> coset_split_serial_samples;
        std::vector<double> coset_split_parallel_samples;
        std::vector<double> interleave_serial_samples;
        std::vector<double> interleave_parallel_samples;
        std::vector<double> fh_division_samples;
        std::vector<double> fh_inverse_multiply_samples;
        serial_samples.reserve(repeats);
        forward_pair_serial_samples.reserve(repeats);
        forward_pair_parallel_samples.reserve(repeats);
        split_serial_samples.reserve(repeats);
        split_parallel_samples.reserve(repeats);
        power_serial_samples.reserve(repeats);
        power_parallel_samples.reserve(repeats);
        rotation_miss_samples.reserve(repeats);
        rotation_hit_samples.reserve(repeats);
        coset_split_serial_samples.reserve(repeats);
        coset_split_parallel_samples.reserve(repeats);
        interleave_serial_samples.reserve(repeats);
        interleave_parallel_samples.reserve(repeats);
        fh_division_samples.reserve(repeats);
        fh_inverse_multiply_samples.reserve(repeats);
        const auto rotation_domain =
            gatzk::algebra::RootOfUnityDomain::create(
                "rotation_stage", size);
        gatzk::protocol::ProtocolContext fh_context;
        fh_context.domains.fh = gatzk::algebra::RootOfUnityDomain::create(
            "fh_division_stage", 64);
        const std::map<std::string, FieldElement> fh_challenges = {
            {"alpha_quot", FieldElement(29)},
            {"beta_feat", FieldElement(31)},
        };
        bool exact = true;
        std::vector<double> batch_samples;
        batch_samples.reserve(repeats);
        for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
            auto forward_first = input;
            auto forward_second = second_input;
            auto start = Clock::now();
            gatzk::algebra::ntt_plan(size)->ntt(forward_first.data());
            gatzk::algebra::ntt_plan(size)->ntt(forward_second.data());
            forward_pair_serial_samples.push_back(elapsed_ms(start));
            auto parallel_first = input;
            auto parallel_second = second_input;
            start = Clock::now();
            gatzk::algebra::ntt_pair_in_place(
                parallel_first.data(), parallel_second.data(), size);
            forward_pair_parallel_samples.push_back(elapsed_ms(start));
            exact = exact && forward_first == parallel_first
                && forward_second == parallel_second;

            auto serial = input;
            start = Clock::now();
            gatzk::algebra::ntt_plan(size)->intt(serial.data());
            serial_samples.push_back(elapsed_ms(start));

            start = Clock::now();
            const auto split_serial = split_intt_serial(input);
            split_serial_samples.push_back(elapsed_ms(start));
            auto split_parallel = input;
            start = Clock::now();
            gatzk::algebra::intt_in_place(
                split_parallel.data(), split_parallel.size());
            split_parallel_samples.push_back(elapsed_ms(start));
            exact = exact && serial == split_serial
                && serial == split_parallel;

            const FieldElement shift(7);
            std::vector<mcl::Fr> serial_powers(size);
            start = Clock::now();
            serial_powers.front() = 1;
            const auto shift_native = shift.native();
            for (std::size_t index = 1; index < size; ++index) {
                mcl::Fr::mul(
                    serial_powers[index],
                    serial_powers[index - 1],
                    shift_native);
            }
            power_serial_samples.push_back(elapsed_ms(start));
            std::vector<mcl::Fr> parallel_powers(size);
            start = Clock::now();
            gatzk::algebra::geometric_powers_native(
                parallel_powers.data(), parallel_powers.size(), shift);
            power_parallel_samples.push_back(elapsed_ms(start));
            exact = exact && serial_powers == parallel_powers;

            start = Clock::now();
            const auto rotation_miss = rotation_domain->rotation_shift(
                FieldElement::one(), shift);
            rotation_miss_samples.push_back(elapsed_ms(start));
            start = Clock::now();
            const auto rotation_hit = rotation_domain->rotation_shift(
                shift, shift * rotation_domain->omega);
            rotation_hit_samples.push_back(elapsed_ms(start));
            exact = exact && !rotation_miss.has_value()
                && rotation_hit == std::optional<std::size_t>(1);

            const auto odd_root = FieldElement::root_of_unity(size * 2);
            const auto odd_root_native = odd_root.native();
            std::vector<mcl::Fr> serial_even(size);
            std::vector<mcl::Fr> serial_odd(size);
            start = Clock::now();
            mcl::Fr odd_root_power;
            odd_root_power = 1;
            for (std::size_t index = 0; index < size; ++index) {
                mcl::Fr::mul(
                    serial_even[index], input[index], parallel_powers[index]);
                mcl::Fr odd_shift_power;
                mcl::Fr::mul(
                    odd_shift_power,
                    parallel_powers[index],
                    odd_root_power);
                mcl::Fr::mul(
                    serial_odd[index], input[index], odd_shift_power);
                mcl::Fr::mul(
                    odd_root_power, odd_root_power, odd_root_native);
            }
            coset_split_serial_samples.push_back(elapsed_ms(start));
            std::vector<mcl::Fr> parallel_even(size);
            std::vector<mcl::Fr> parallel_odd(size);
            start = Clock::now();
            gatzk::algebra::split_coset_coefficients_native(
                input.data(),
                parallel_powers.data(),
                size,
                odd_root,
                parallel_even.data(),
                parallel_odd.data());
            coset_split_parallel_samples.push_back(elapsed_ms(start));
            exact = exact && serial_even == parallel_even
                && serial_odd == parallel_odd;

            std::vector<mcl::Fr> serial_interleaved(size * 2);
            start = Clock::now();
            for (std::size_t index = 0; index < size; ++index) {
                serial_interleaved[2 * index] = serial_even[index];
                serial_interleaved[2 * index + 1] = serial_odd[index];
            }
            interleave_serial_samples.push_back(elapsed_ms(start));
            std::vector<mcl::Fr> parallel_interleaved(size * 2);
            start = Clock::now();
            gatzk::algebra::interleave_native_lanes(
                parallel_even.data(),
                parallel_odd.data(),
                size,
                parallel_interleaved.data());
            interleave_parallel_samples.push_back(elapsed_ms(start));
            exact = exact && serial_interleaved == parallel_interleaved;

            constexpr std::size_t kFhDivisionStagePoints = 1ULL << 20;
            start = Clock::now();
            const auto fh_division = run_fh_zero_division_stage(
                fh_context, fh_challenges, kFhDivisionStagePoints, false);
            fh_division_samples.push_back(elapsed_ms(start));
            start = Clock::now();
            const auto fh_inverse_multiply = run_fh_zero_division_stage(
                fh_context, fh_challenges, kFhDivisionStagePoints, true);
            fh_inverse_multiply_samples.push_back(elapsed_ms(start));
            exact = exact && fh_division == fh_inverse_multiply;

            if (batch_objects != 0) {
                std::vector<FieldElement> batch_outputs(
                    batch_objects, FieldElement::zero());
                std::atomic<std::size_t> next{0};
                const auto run_object = [&]() {
                    while (true) {
                        const auto object = next.fetch_add(1);
                        if (object >= batch_objects) {
                            return;
                        }
                        auto coefficients = input;
                        coefficients.front() = FieldElement(
                            41 + object).native();
                        if (batch_mode == "production") {
                            gatzk::algebra::intt_in_place(
                                coefficients.data(), coefficients.size());
                        } else {
                            gatzk::algebra::ntt_plan(size)->intt(
                                coefficients.data());
                        }
                        std::vector<mcl::Fr> first(size * 2);
                        std::vector<mcl::Fr> second(size * 2);
                        std::copy(
                            coefficients.begin(), coefficients.end(),
                            first.begin());
                        std::copy(
                            coefficients.begin(), coefficients.end(),
                            second.begin());
                        if (batch_mode == "production") {
                            gatzk::algebra::ntt_pair_in_place(
                                first.data(), second.data(), size * 2);
                        } else {
                            const auto forward_plan =
                                gatzk::algebra::ntt_plan(size * 2);
                            forward_plan->ntt(first.data());
                            forward_plan->ntt(second.data());
                        }
                        mcl::Fr sum;
                        mcl::Fr::add(sum, first.front(), second.back());
                        batch_outputs[object] =
                            FieldElement::from_native(sum);
                    }
                };
                start = Clock::now();
                std::vector<std::future<void>> workers;
                workers.reserve(batch_workers);
                for (std::size_t worker = 0; worker < batch_workers; ++worker) {
                    workers.push_back(std::async(
                        std::launch::async, run_object));
                }
                for (auto& worker : workers) {
                    worker.get();
                }
                batch_samples.push_back(elapsed_ms(start));
                exact = exact && std::none_of(
                    batch_outputs.begin(), batch_outputs.end(),
                    [](const FieldElement& value) { return value.is_zero(); });
            }
        }
        if (!exact) {
            throw std::runtime_error("split INTT output mismatch");
        }
        auto median = [](std::vector<double> samples) {
            std::sort(samples.begin(), samples.end());
            return samples[samples.size() / 2];
        };
        std::cout << std::fixed << std::setprecision(3)
                  << "size=" << size
                  << " repeats=" << repeats
                  << " serial_median_ms=" << median(serial_samples)
                  << " forward_pair_serial_median_ms="
                  << median(forward_pair_serial_samples)
                  << " forward_pair_parallel_median_ms="
                  << median(forward_pair_parallel_samples)
                  << " split_serial_median_ms=" << median(split_serial_samples)
                  << " split_parallel_median_ms=" << median(split_parallel_samples)
                  << " power_serial_median_ms=" << median(power_serial_samples)
                  << " power_parallel_median_ms=" << median(power_parallel_samples)
                  << " rotation_miss_median_ms=" << median(rotation_miss_samples)
                  << " rotation_hit_median_ms=" << median(rotation_hit_samples)
                  << " coset_split_serial_median_ms="
                  << median(coset_split_serial_samples)
                  << " coset_split_parallel_median_ms="
                  << median(coset_split_parallel_samples)
                  << " interleave_serial_median_ms="
                  << median(interleave_serial_samples)
                  << " interleave_parallel_median_ms="
                  << median(interleave_parallel_samples)
                  << " fh_division_median_ms="
                  << median(fh_division_samples)
                  << " fh_inverse_multiply_median_ms="
                  << median(fh_inverse_multiply_samples)
                  << " batch_objects=" << batch_objects
                  << " batch_workers=" << batch_workers
                  << " batch_mode=" << batch_mode;
        if (!batch_samples.empty()) {
            std::cout << " batch_median_ms=" << median(batch_samples);
        }
        std::cout
                  << " exact=true\n";
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
