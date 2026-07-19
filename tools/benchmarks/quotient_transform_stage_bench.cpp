#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/vector_ops.hpp"

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
        serial_samples.reserve(repeats);
        forward_pair_serial_samples.reserve(repeats);
        forward_pair_parallel_samples.reserve(repeats);
        split_serial_samples.reserve(repeats);
        split_parallel_samples.reserve(repeats);
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
