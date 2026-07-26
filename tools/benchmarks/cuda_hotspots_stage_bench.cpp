#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/algebra/eval_backend.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::protocol {
std::vector<std::size_t> lookup_histogram_indices_cuda(
    const std::vector<std::size_t>& indices,
    std::size_t domain_size);
}

namespace {

using Clock = std::chrono::steady_clock;
using gatzk::algebra::FieldElement;

double elapsed_ms(const Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void require_equal(const std::vector<mcl::Fr>& expected, const std::vector<mcl::Fr>& actual) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error("CUDA result size mismatch");
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            throw std::runtime_error("CUDA field result mismatch at index " + std::to_string(i));
        }
    }
}

void bench_barycentric(const std::size_t domain_size, const std::size_t repeats) {
    const auto domain = gatzk::algebra::RootOfUnityDomain::create("cuda_bench", domain_size);
    const FieldElement point(7);
    std::vector<double> cpu_samples;
    std::vector<mcl::Fr> expected;
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        expected = domain->barycentric_weights_native(point);
        cpu_samples.push_back(elapsed_ms(start));
    }
    const auto zero_eval = domain->zero_polynomial_eval(point);
    std::vector<mcl::Fr> actual(domain_size);
    const auto* omega = domain->omega.native().getUnit();
    const auto* x = point.native().getUnit();
    const auto* zero = zero_eval.native().getUnit();
    const auto* inv_n = domain->inv_size.native().getUnit();
    const std::uint64_t omega_limbs[4] = {omega[0], omega[1], omega[2], omega[3]};
    const std::uint64_t x_limbs[4] = {x[0], x[1], x[2], x[3]};
    const std::uint64_t zero_limbs[4] = {zero[0], zero[1], zero[2], zero[3]};
    const std::uint64_t inv_n_limbs[4] = {inv_n[0], inv_n[1], inv_n[2], inv_n[3]};
    std::vector<double> samples;
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        const bool ok = gatzk::algebra::barycentric_weights_gpu(
            omega_limbs,
            x_limbs,
            zero_limbs,
            inv_n_limbs,
            reinterpret_cast<std::uint64_t*>(actual.data()),
            domain_size);
        samples.push_back(elapsed_ms(start));
        if (!ok) {
            throw std::runtime_error("CUDA barycentric execution failed");
        }
        require_equal(expected, actual);
    }
    std::cout << "case=barycentric domain_size=" << domain_size
              << " rows=1 cpu_median_ms=" << median(cpu_samples)
              << " gpu_median_ms=" << median(samples) << " exact_match=true\n";
}

std::vector<mcl::Fr> weighted_reference(
    const std::vector<mcl::Fr>& weights,
    const std::vector<mcl::Fr>& interleaved,
    const std::size_t row_count) {
    std::vector<mcl::Fr> out(row_count);
    for (auto& value : out) value.clear();
    for (std::size_t column = 0; column < weights.size(); ++column) {
        for (std::size_t row = 0; row < row_count; ++row) {
            mcl::Fr term;
            mcl::Fr::mul(term, weights[column], interleaved[column * row_count + row]);
            mcl::Fr::add(out[row], out[row], term);
        }
    }
    return out;
}

void bench_weighted(const std::size_t domain_size, const std::size_t row_count, const std::size_t repeats) {
    std::vector<mcl::Fr> weights(domain_size);
    std::vector<mcl::Fr> interleaved(domain_size * row_count);
    std::vector<std::vector<mcl::Fr>> columns(row_count, std::vector<mcl::Fr>(domain_size));
    for (std::size_t column = 0; column < domain_size; ++column) {
        weights[column] = static_cast<long long>((column * 17 + 3) % 1009);
        for (std::size_t row = 0; row < row_count; ++row) {
            const auto scalar = static_cast<long long>((column * 29 + row * 11 + 5) % 1013);
            interleaved[column * row_count + row] = scalar;
            columns[row][column] = scalar;
        }
    }
    const auto expected = weighted_reference(weights, interleaved, row_count);
    std::vector<mcl::Fr> actual(row_count);
    std::vector<const std::uint64_t*> column_ptrs(row_count);
    for (std::size_t row = 0; row < row_count; ++row) {
        column_ptrs[row] = reinterpret_cast<const std::uint64_t*>(columns[row].data());
    }

    auto run = [&](const std::string& label, const bool columnar) {
        std::vector<double> samples;
        for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
            for (auto& value : actual) value.clear();
            const auto start = Clock::now();
            const bool ok = columnar
                ? gatzk::algebra::weighted_poly_eval_gpu_multi(
                    reinterpret_cast<const std::uint64_t*>(weights.data()),
                    column_ptrs.data(),
                    reinterpret_cast<std::uint64_t*>(actual.data()),
                    domain_size,
                    row_count)
                : gatzk::algebra::weighted_poly_eval_gpu(
                    reinterpret_cast<const std::uint64_t*>(weights.data()),
                    reinterpret_cast<const std::uint64_t*>(interleaved.data()),
                    reinterpret_cast<std::uint64_t*>(actual.data()),
                    domain_size,
                    row_count);
            samples.push_back(elapsed_ms(start));
            if (!ok) {
                throw std::runtime_error("CUDA weighted evaluation failed");
            }
            require_equal(expected, actual);
        }
        std::cout << "case=" << label << " domain_size=" << domain_size
                  << " rows=" << row_count << " median_ms=" << median(samples)
                  << " exact_match=true\n";
    };
    run("weighted_interleaved", false);
    run("weighted_columnar", true);
}

void bench_formal_weighted_route(
    const std::size_t domain_size,
    const std::size_t row_count,
    const std::size_t repeats,
    const bool padded_tail) {
    const auto domain = gatzk::algebra::RootOfUnityDomain::create("edge", domain_size);
    std::vector<gatzk::algebra::Polynomial> polynomials;
    polynomials.reserve(row_count);
    for (std::size_t row = 0; row < row_count; ++row) {
        std::vector<FieldElement> values(domain_size);
        for (std::size_t column = 0; column < domain_size; ++column) {
            values[column] = FieldElement(static_cast<std::uint64_t>(
                (column * 29 + row * 11 + 5) % 1013));
        }
        if (padded_tail && row == 0) values.resize(domain_size / 2);
        polynomials.push_back(gatzk::algebra::Polynomial::from_evaluations(
            "cuda_route_" + std::to_string(row), std::move(values), domain));
    }
    std::vector<std::pair<std::string, const gatzk::algebra::Polynomial*>> named;
    std::vector<std::string> labels;
    named.reserve(row_count);
    labels.reserve(row_count);
    for (const auto& polynomial : polynomials) {
        named.emplace_back(polynomial.name, &polynomial);
        labels.push_back(polynomial.name);
    }
    gatzk::algebra::PackedEvaluationBackend backend(domain, std::move(named));
    std::vector<mcl::Fr> weights(domain_size);
    for (std::size_t column = 0; column < domain_size; ++column) {
        weights[column] = static_cast<long long>((column * 17 + 3) % 1009);
    }

    gatzk::util::Route2Options cpu_route;
    gatzk::util::set_route2_options(cpu_route);
    std::vector<double> cpu_samples;
    std::vector<FieldElement> expected;
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        expected = backend.evaluate_with_native_weights(labels, weights);
        cpu_samples.push_back(elapsed_ms(start));
    }

    auto gpu_route = cpu_route;
    gpu_route.cuda_field_hotspots = true;
    gatzk::util::set_route2_options(gpu_route);
    if (!gatzk::algebra::should_use_cuda_weighted_evaluation(domain_size, row_count)) {
        throw std::runtime_error("representative weighted case did not enter formal CUDA route");
    }
    std::vector<double> gpu_samples;
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        const auto actual = backend.evaluate_with_native_weights(labels, weights);
        gpu_samples.push_back(elapsed_ms(start));
        if (actual != expected) {
            throw std::runtime_error("formal CUDA weighted route mismatch");
        }
    }
    std::cout << "case="
              << (padded_tail ? "formal_weighted_padded_route" : "formal_weighted_route")
              << " domain_size=" << domain_size
              << " rows=" << row_count << " cpu_median_ms=" << median(cpu_samples)
              << " gpu_median_ms=" << median(gpu_samples) << " exact_match=true\n";
}

void bench_formal_weighted_rotations(
    const std::size_t domain_size,
    const std::size_t row_count,
    const std::size_t repeats,
    const bool padded_tail) {
    const auto domain = gatzk::algebra::RootOfUnityDomain::create(
        "edge", domain_size);
    std::vector<gatzk::algebra::Polynomial> polynomials;
    polynomials.reserve(row_count);
    for (std::size_t row = 0; row < row_count; ++row) {
        std::vector<FieldElement> values(domain_size);
        for (std::size_t column = 0; column < domain_size; ++column) {
            values[column] = FieldElement(static_cast<std::uint64_t>(
                (column * 29 + row * 11 + 5) % 1013));
        }
        if (padded_tail && row == 0) values.resize(domain_size / 2);
        polynomials.push_back(gatzk::algebra::Polynomial::from_evaluations(
            "cuda_rotation_" + std::to_string(row),
            std::move(values),
            domain));
    }
    std::vector<std::pair<std::string, const gatzk::algebra::Polynomial*>> named;
    std::vector<std::string> labels;
    named.reserve(row_count);
    labels.reserve(row_count);
    for (const auto& polynomial : polynomials) {
        named.emplace_back(polynomial.name, &polynomial);
        labels.push_back(polynomial.name);
    }
    gatzk::algebra::PackedEvaluationBackend backend(domain, std::move(named));
    std::vector<mcl::Fr> weights(domain_size);
    for (std::size_t column = 0; column < domain_size; ++column) {
        weights[column] = static_cast<long long>((column * 17 + 3) % 1009);
    }
    const std::vector<std::size_t> rotations = {
        0,
        1,
        domain_size / 2,
    };

    gatzk::util::Route2Options cpu_route;
    gatzk::util::set_route2_options(cpu_route);
    std::vector<double> cpu_samples;
    std::vector<std::vector<FieldElement>> expected;
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        expected = backend.evaluate_with_native_weight_rotations(
            labels, weights, rotations);
        cpu_samples.push_back(elapsed_ms(start));
    }

    auto gpu_route = cpu_route;
    gpu_route.cuda_field_hotspots = true;
    gatzk::util::set_route2_options(gpu_route);
    if (!gatzk::algebra::should_use_cuda_weighted_evaluation(
            domain_size, row_count, rotations.size())) {
        throw std::runtime_error(
            "representative rotated weighted case did not enter formal CUDA route");
    }
    std::vector<double> gpu_samples;
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        const auto actual = backend.evaluate_with_native_weight_rotations(
            labels, weights, rotations);
        gpu_samples.push_back(elapsed_ms(start));
        if (actual != expected) {
            throw std::runtime_error("formal CUDA rotated weighted route mismatch");
        }
    }
    std::cout << "case="
              << (padded_tail
                      ? "formal_weighted_padded_rotations"
                      : "formal_weighted_rotations")
              << " domain_size=" << domain_size
              << " rows=" << row_count
              << " points=" << rotations.size()
              << " cpu_median_ms=" << median(cpu_samples)
              << " gpu_median_ms=" << median(gpu_samples)
              << " exact_match=true\n";
}

void bench_histogram(
    const std::size_t index_count,
    const std::size_t bucket_count,
    const std::size_t repeats) {
    std::vector<std::size_t> indices(index_count);
    for (std::size_t i = 0; i < index_count; ++i) {
        indices[i] = (i * 37 + i / 7) % bucket_count;
    }
    std::vector<double> cpu_samples;
    std::vector<FieldElement> expected;
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        std::vector<FieldElement> cpu_counts(bucket_count, FieldElement::zero());
        const auto start = Clock::now();
        for (const auto index : indices) cpu_counts[index] += FieldElement::one();
        cpu_samples.push_back(elapsed_ms(start));
        if (!expected.empty() && cpu_counts != expected) {
            throw std::runtime_error("CPU histogram result mismatch");
        }
        expected = std::move(cpu_counts);
    }
    std::vector<double> samples;
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        const auto counts = gatzk::protocol::lookup_histogram_indices_cuda(indices, bucket_count);
        std::vector<FieldElement> actual(bucket_count, FieldElement::zero());
        for (std::size_t bucket = 0; bucket < bucket_count; ++bucket) {
            actual[bucket] = FieldElement(counts[bucket]);
        }
        samples.push_back(elapsed_ms(start));
        if (actual != expected) {
            throw std::runtime_error("CUDA histogram result mismatch");
        }
    }
    std::cout << "case=lookup_histogram index_count=" << index_count
              << " buckets=" << bucket_count << " cpu_median_ms=" << median(cpu_samples)
              << " gpu_median_ms=" << median(samples)
              << " exact_match=true\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 1) {
            throw std::runtime_error("usage: gatzk_cuda_hotspots_stage_bench");
        }
        (void)argv;
        if (!gatzk::algebra::cuda_backend_build_enabled()
            || !gatzk::algebra::cuda_backend_available()) {
            throw std::runtime_error("CUDA-enabled build and runtime are required");
        }
        constexpr std::size_t kDomainSize = 1ULL << 16;
        constexpr std::size_t kRows = 32;
        constexpr std::size_t kRepeats = 5;
        static_assert(kDomainSize <= (1ULL << 20));
        std::cout << std::fixed << std::setprecision(3)
                  << "device={" << gatzk::algebra::cuda_device_summary() << "}\n";
        const auto kernels_before = gatzk::algebra::cuda_kernel_invocation_count();
        bench_barycentric(kDomainSize, kRepeats);
        bench_weighted(kDomainSize, kRows, kRepeats);
        bench_formal_weighted_route(kDomainSize, kRows, kRepeats, false);
        bench_formal_weighted_route(kDomainSize, kRows, kRepeats, true);
        bench_formal_weighted_rotations(kDomainSize, kRows, kRepeats, false);
        bench_formal_weighted_rotations(kDomainSize, kRows, kRepeats, true);
        constexpr std::size_t kShapeRepeats = 3;
        bench_formal_weighted_rotations(
            1ULL << 15, 64, kShapeRepeats, true);
        bench_formal_weighted_rotations(
            1ULL << 18, 8, kShapeRepeats, true);
        bench_formal_weighted_rotations(
            1ULL << 20, 2, kShapeRepeats, true);
        bench_formal_weighted_rotations(
            1ULL << 22, 1, kShapeRepeats, true);
        gatzk::util::Route2Options cuda_route;
        cuda_route.cuda_trace_hotspots = true;
        cuda_route.cuda_field_hotspots = true;
        gatzk::util::set_route2_options(cuda_route);
        if (gatzk::algebra::should_use_cuda_lookup_histogram(1ULL << 12, 1ULL << 10)
            || !gatzk::algebra::should_use_cuda_lookup_histogram(1ULL << 14, 1ULL << 10)) {
            throw std::runtime_error("CUDA histogram dispatch thresholds are inconsistent");
        }
        bench_histogram(1ULL << 12, 1ULL << 10, kRepeats);
        bench_histogram(1ULL << 14, 1ULL << 10, kRepeats);
        bench_histogram(kDomainSize, 1ULL << 12, kRepeats);
        bench_histogram(1ULL << 20, 1ULL << 12, kRepeats);
        bench_histogram(1ULL << 22, 1ULL << 16, kRepeats);
        std::cout << "kernel_invocations="
                  << (gatzk::algebra::cuda_kernel_invocation_count() - kernels_before)
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
