#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "gatzk/protocol/lookup.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using gatzk::algebra::FieldElement;

double elapsed_ms(const Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples.at(samples.size() / 2);
}

struct BaselineResult {
    std::vector<FieldElement> committed_tuples;
    gatzk::protocol::IndexedViewSideTrace witness;
};

BaselineResult run_selector_baseline(
    const std::vector<FieldElement>& bias_flat,
    std::size_t active_length,
    std::size_t class_count,
    const FieldElement& eta,
    const FieldElement& beta) {
    const auto domain_size = bias_flat.size();
    std::vector<FieldElement> indices(
        domain_size, FieldElement::zero());
    std::vector<FieldElement> values(
        domain_size, FieldElement::zero());
    std::vector<FieldElement> selector(
        domain_size, FieldElement::zero());
    std::vector<FieldElement> multiplicity(
        domain_size, FieldElement::zero());
    for (std::size_t index = 0; index < active_length; ++index) {
        indices[index] = FieldElement(index % class_count);
        values[index] = bias_flat[index];
        selector[index] = FieldElement::one();
        multiplicity[index] = FieldElement::one();
    }
    std::vector<FieldElement> committed_tuples(
        domain_size, FieldElement::zero());
    for (std::size_t index = 0; index < domain_size; ++index) {
        committed_tuples[index] = selector[index]
            * (indices[index] + eta * values[index]);
    }
    return {
        .committed_tuples = std::move(committed_tuples),
        .witness = gatzk::protocol::build_indexed_view_side(
            indices, values, selector, multiplicity, eta, beta),
    };
}

struct PrefixResult {
    std::vector<FieldElement> committed_tuples;
    gatzk::protocol::IndexedViewActivePrefixWitness witness;
};

PrefixResult run_active_prefix(
    const std::vector<FieldElement>& bias_flat,
    std::size_t active_length,
    std::size_t class_count,
    const FieldElement& eta,
    const FieldElement& beta) {
    std::vector<FieldElement> tuples(
        bias_flat.size(), FieldElement::zero());
    for (std::size_t index = 0; index < active_length; ++index) {
        tuples[index] = FieldElement(index % class_count)
            + eta * bias_flat[index];
    }
    auto witness =
        gatzk::protocol::build_indexed_view_active_prefix_witness(
            tuples, active_length, FieldElement::one(), beta);
    return {
        .committed_tuples = std::move(tuples),
        .witness = std::move(witness),
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto repeats = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1]))
            : std::size_t{5};
        const auto domain_size = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2]))
            : std::size_t{1} << 20;
        const auto active_length = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3]))
            : domain_size * 13 / 16;
        const auto class_count = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4]))
            : std::size_t{121};
        if (argc > 5 || repeats < 5 || domain_size < 2
            || active_length >= domain_size || class_count == 0) {
            throw std::runtime_error(
                "usage: gatzk_indexed_view_prefix_stage_bench "
                "[repeats domain_size active_length class_count]; "
                "repeats>=5 and active_length<domain_size");
        }

        std::vector<FieldElement> bias_flat(
            domain_size, FieldElement::zero());
        for (std::size_t index = 0; index < active_length; ++index) {
            bias_flat[index] = FieldElement(
                7 + (index * 1315423911ULL) % 65521ULL);
        }
        const FieldElement eta(31);
        const FieldElement beta(1000003);
        std::vector<double> baseline_samples;
        std::vector<double> prefix_samples;
        baseline_samples.reserve(repeats);
        prefix_samples.reserve(repeats);

        for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
            BaselineResult baseline;
            PrefixResult prefix;
            const auto run_baseline = [&]() {
                const auto start = Clock::now();
                baseline = run_selector_baseline(
                    bias_flat, active_length, class_count, eta, beta);
                baseline_samples.push_back(elapsed_ms(start));
            };
            const auto run_prefix = [&]() {
                const auto start = Clock::now();
                prefix = run_active_prefix(
                    bias_flat, active_length, class_count, eta, beta);
                prefix_samples.push_back(elapsed_ms(start));
            };
            if ((repeat & 1U) == 0U) {
                run_baseline();
                run_prefix();
            } else {
                run_prefix();
                run_baseline();
            }
            if (baseline.committed_tuples != prefix.committed_tuples
                || baseline.witness.inverses != prefix.witness.inverses
                || baseline.witness.accumulator
                    != prefix.witness.accumulator) {
                throw std::runtime_error(
                    "indexed-view active-prefix benchmark mismatch");
            }
        }

        const auto baseline_median = median(baseline_samples);
        const auto prefix_median = median(prefix_samples);
        std::cout << std::fixed << std::setprecision(3)
                  << "domain_size=" << domain_size
                  << " active_length=" << active_length
                  << " class_count=" << class_count
                  << " repeats=" << repeats
                  << " selector_median_ms=" << baseline_median
                  << " active_prefix_median_ms=" << prefix_median
                  << " saved_ms=" << baseline_median - prefix_median
                  << " speedup=" << baseline_median / prefix_median
                  << " exact=true\n";
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
