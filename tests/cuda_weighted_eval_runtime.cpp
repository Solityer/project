#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/vector_ops.hpp"

namespace
{
    using gatzk::algebra::FieldElement;

    constexpr std::size_t kDomainSize = 8;
    constexpr std::size_t kRowCount = 3;

    mcl::Fr make_value(std::uint64_t value)
    {
        return FieldElement(value).native();
    }

    std::vector<mcl::Fr> expected_results(
        const std::vector<mcl::Fr> &weights,
        const std::vector<std::vector<mcl::Fr>> &columns,
        const std::vector<std::size_t> &column_lengths)
    {
        std::vector<mcl::Fr> expected(columns.size());
        for (auto &value : expected)
        {
            value.clear();
        }

        for (std::size_t row = 0; row < columns.size(); ++row)
        {
            const auto length = std::min(kDomainSize, column_lengths[row]);
            for (std::size_t col = 0; col < length; ++col)
            {
                mcl::Fr product;
                mcl::Fr::mul(product, weights[col], columns[row][col]);
                mcl::Fr::add(expected[row], expected[row], product);
            }
        }
        return expected;
    }

    void require_equal(
        const std::vector<mcl::Fr> &actual,
        const std::vector<mcl::Fr> &expected,
        const char *label)
    {
        if (actual.size() != expected.size())
        {
            throw std::runtime_error(std::string(label) + " size mismatch");
        }
        for (std::size_t index = 0; index < actual.size(); ++index)
        {
            if (actual[index] != expected[index])
            {
                throw std::runtime_error(std::string(label) + " differs from CPU reference");
            }
        }
    }

    void require_unchanged(
        const std::array<std::uint64_t, 4> &actual,
        const std::array<std::uint64_t, 4> &before,
        const char *label)
    {
        if (actual != before)
        {
            throw std::runtime_error(std::string(label) + " wrote output before rejecting");
        }
    }

    void run_guard_cases()
    {
        constexpr std::size_t kMaxGpuValueBytes =
            2ULL * 1024ULL * 1024ULL * 1024ULL;
        constexpr std::size_t kGpuFieldBytes = 4 * sizeof(std::uint64_t);
        constexpr std::size_t kMaxRows = 65535;

        const auto bytes_per_column = kMaxRows * kGpuFieldBytes;
        const auto max_cols_per_chunk =
            kMaxGpuValueBytes / bytes_per_column;
        if (max_cols_per_chunk == 0 ||
            max_cols_per_chunk * bytes_per_column > kMaxGpuValueBytes)
        {
            throw std::runtime_error(
                "row_count=65535 chunk plan exceeds the hard CUDA byte cap");
        }

        const std::array<std::uint64_t, 4> before = {
            0x1111111111111111ULL,
            0x2222222222222222ULL,
            0x3333333333333333ULL,
            0x4444444444444444ULL,
        };

        auto interleaved_output = before;
        if (gatzk::algebra::weighted_poly_eval_gpu(
                nullptr,
                nullptr,
                interleaved_output.data(),
                1,
                65536))
        {
            throw std::runtime_error("interleaved weighted evaluation accepted row_count=65536");
        }
        require_unchanged(interleaved_output, before, "interleaved guard");

        auto columnar_output = before;
        const std::array<const std::uint64_t *, 1> no_columns = {
            nullptr,
        };
        if (gatzk::algebra::weighted_poly_eval_gpu_columnar(
                nullptr,
                no_columns.data(),
                columnar_output.data(),
                1,
                65536))
        {
            throw std::runtime_error("columnar weighted evaluation accepted row_count=65536");
        }
        require_unchanged(columnar_output, before, "columnar guard");

        auto single_column_output = before;
        const auto single_column_rows =
            kMaxGpuValueBytes / kGpuFieldBytes + 1;
        if (gatzk::algebra::weighted_poly_eval_gpu(
                nullptr,
                nullptr,
                single_column_output.data(),
                1,
                single_column_rows))
        {
            throw std::runtime_error(
                "interleaved weighted evaluation accepted an oversized single column");
        }
        require_unchanged(single_column_output, before, "single-column guard");

        auto overflow_output = before;
        if (gatzk::algebra::weighted_poly_eval_gpu(
                nullptr,
                nullptr,
                overflow_output.data(),
                std::numeric_limits<std::size_t>::max(),
                1))
        {
            throw std::runtime_error(
                "interleaved weighted evaluation accepted an overflowing domain size");
        }
        require_unchanged(overflow_output, before, "overflow guard");
    }

    void run_small_cases()
    {
        std::vector<mcl::Fr> weights(kDomainSize);
        std::vector<std::vector<mcl::Fr>> columns(
            kRowCount,
            std::vector<mcl::Fr>(kDomainSize));
        for (std::size_t col = 0; col < kDomainSize; ++col)
        {
            weights[col] = make_value(col + 2);
            for (std::size_t row = 0; row < kRowCount; ++row)
            {
                columns[row][col] = make_value((col + 3) * (row + 5));
            }
        }

        std::vector<mcl::Fr> interleaved(kDomainSize * kRowCount);
        for (std::size_t col = 0; col < kDomainSize; ++col)
        {
            for (std::size_t row = 0; row < kRowCount; ++row)
            {
                interleaved[col * kRowCount + row] = columns[row][col];
            }
        }

        const std::vector<std::size_t> full_lengths(kRowCount, kDomainSize);
        const auto full_expected = expected_results(weights, columns, full_lengths);

        std::vector<mcl::Fr> interleaved_output(kRowCount);
        if (!gatzk::algebra::weighted_poly_eval_gpu(
                reinterpret_cast<const std::uint64_t *>(weights.data()),
                reinterpret_cast<const std::uint64_t *>(interleaved.data()),
                reinterpret_cast<std::uint64_t *>(interleaved_output.data()),
                kDomainSize,
                kRowCount))
        {
            throw std::runtime_error("interleaved weighted evaluation declined the CUDA case");
        }
        require_equal(interleaved_output, full_expected, "interleaved weighted evaluation");

        std::vector<const std::uint64_t *> column_ptrs(kRowCount);
        for (std::size_t row = 0; row < kRowCount; ++row)
        {
            column_ptrs[row] = reinterpret_cast<const std::uint64_t *>(columns[row].data());
        }

        std::vector<mcl::Fr> multi_output(kRowCount);
        if (!gatzk::algebra::weighted_poly_eval_gpu_multi(
                reinterpret_cast<const std::uint64_t *>(weights.data()),
                column_ptrs.data(),
                reinterpret_cast<std::uint64_t *>(multi_output.data()),
                kDomainSize,
                kRowCount,
                full_lengths.data()))
        {
            throw std::runtime_error("multi weighted evaluation declined the CUDA case");
        }
        require_equal(multi_output, full_expected, "multi weighted evaluation");

        if (gatzk::algebra::cuda_backend_unified_addressing_available())
        {
            std::vector<mcl::Fr> columnar_output(kRowCount);
            if (!gatzk::algebra::weighted_poly_eval_gpu_columnar(
                    reinterpret_cast<const std::uint64_t *>(weights.data()),
                    column_ptrs.data(),
                    reinterpret_cast<std::uint64_t *>(columnar_output.data()),
                    kDomainSize,
                    kRowCount,
                    full_lengths.data()))
            {
                throw std::runtime_error("columnar weighted evaluation declined the CUDA case");
            }
            require_equal(columnar_output, full_expected, "columnar weighted evaluation");
        }
    }

    std::vector<mcl::Fr> expected_rotated_results(
        const std::vector<mcl::Fr> &weights,
        const std::vector<std::vector<mcl::Fr>> &columns,
        const std::vector<std::size_t> &column_lengths,
        const std::vector<std::size_t> &rotations)
    {
        const auto domain_size = weights.size();
        std::vector<mcl::Fr> expected(
            rotations.size() * columns.size());
        for (auto &value : expected)
        {
            value.clear();
        }

        for (std::size_t point = 0; point < rotations.size(); ++point)
        {
            const auto rotation = rotations[point] % domain_size;
            for (std::size_t row = 0; row < columns.size(); ++row)
            {
                const auto length = std::min(
                    domain_size,
                    column_lengths[row]);
                for (std::size_t col = 0; col < length; ++col)
                {
                    const auto weight_index =
                        col >= rotation
                            ? col - rotation
                            : domain_size - (rotation - col);
                    mcl::Fr product;
                    mcl::Fr::mul(
                        product,
                        weights[weight_index],
                        columns[row][col]);
                    mcl::Fr::add(
                        expected[point * columns.size() + row],
                        expected[point * columns.size() + row],
                        product);
                }
            }
        }
        return expected;
    }

    void require_rotated_equal(
        const std::vector<mcl::Fr> &actual,
        const std::vector<mcl::Fr> &expected,
        const char *label)
    {
        if (actual.size() != expected.size())
        {
            throw std::runtime_error(std::string(label) + " size mismatch");
        }
        for (std::size_t index = 0; index < actual.size(); ++index)
        {
            if (actual[index] != expected[index])
            {
                throw std::runtime_error(
                    std::string(label) + " differs from CPU reference at " +
                    std::to_string(index));
            }
        }
    }

    std::vector<mcl::Fr> build_rotated_weights(
        const std::vector<mcl::Fr> &weights,
        std::size_t rotation)
    {
        const auto domain_size = weights.size();
        std::vector<mcl::Fr> rotated(domain_size);
        const auto normalized = rotation % domain_size;
        for (std::size_t col = 0; col < domain_size; ++col)
        {
            rotated[(col + normalized) & (domain_size - 1)] =
                weights[col];
        }
        return rotated;
    }

    std::vector<mcl::Fr> run_old_rotated_loop(
        const std::vector<mcl::Fr> &weights,
        const std::vector<std::vector<mcl::Fr>> &columns,
        const std::vector<std::size_t> &rotations)
    {
        const auto domain_size = weights.size();
        const auto row_count = columns.size();
        std::vector<const std::uint64_t *> column_ptrs(row_count);
        for (std::size_t row = 0; row < row_count; ++row)
        {
            column_ptrs[row] = reinterpret_cast<const std::uint64_t *>(
                columns[row].data());
        }

        std::vector<mcl::Fr> output(rotations.size() * row_count);
        for (std::size_t point = 0; point < rotations.size(); ++point)
        {
            const auto rotated = build_rotated_weights(
                weights,
                rotations[point]);
            if (!gatzk::algebra::weighted_poly_eval_gpu_multi(
                    reinterpret_cast<const std::uint64_t *>(
                        rotated.data()),
                    column_ptrs.data(),
                    reinterpret_cast<std::uint64_t *>(
                        output.data() + point * row_count),
                    domain_size,
                    row_count))
            {
                throw std::runtime_error(
                    "old rotated loop declined the CUDA case");
            }
        }
        return output;
    }

    std::vector<mcl::Fr> run_new_rotated_batch(
        const std::vector<mcl::Fr> &weights,
        const std::vector<std::vector<mcl::Fr>> &columns,
        const std::vector<std::size_t> &rotations,
        const std::vector<std::size_t> &column_lengths)
    {
        const auto domain_size = weights.size();
        const auto row_count = columns.size();
        std::vector<const std::uint64_t *> column_ptrs(row_count);
        for (std::size_t row = 0; row < row_count; ++row)
        {
            column_ptrs[row] = reinterpret_cast<const std::uint64_t *>(
                columns[row].data());
        }
        std::vector<mcl::Fr> output(rotations.size() * row_count);
        if (!gatzk::algebra::weighted_poly_eval_gpu_multi_rotations(
                reinterpret_cast<const std::uint64_t *>(weights.data()),
                column_ptrs.data(),
                reinterpret_cast<std::uint64_t *>(output.data()),
                domain_size,
                row_count,
                rotations.data(),
                rotations.size(),
                column_lengths.data()))
        {
            throw std::runtime_error(
                "multi-rotation CUDA API declined the CUDA case");
        }
        return output;
    }

    std::uint64_t median_of(
        std::vector<std::uint64_t> values)
    {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    std::uint64_t variance_of(
        const std::vector<std::uint64_t> &values,
        std::uint64_t median)
    {
        unsigned __int128 sum = 0;
        for (const auto value : values)
        {
            const auto delta =
                value > median ? value - median : median - value;
            sum += static_cast<unsigned __int128>(delta) * delta;
        }
        return static_cast<std::uint64_t>(
            sum / std::max<std::size_t>(1, values.size()));
    }

    void run_rotated_cases()
    {
        const bool unified =
            gatzk::algebra::cuda_backend_unified_addressing_available();
        std::cout << "CUDA device: "
                  << gatzk::algebra::cuda_device_summary()
                  << "; unified_pageable="
                  << (unified ? "true" : "false") << '\n';

        for (const auto domain_size : {std::size_t{512}, std::size_t{1} << 14})
        {
            constexpr std::size_t row_count = 4;
            std::vector<mcl::Fr> weights(domain_size);
            std::vector<std::vector<mcl::Fr>> columns(
                row_count,
                std::vector<mcl::Fr>(domain_size));
            std::vector<std::size_t> column_lengths(row_count);
            for (std::size_t col = 0; col < domain_size; ++col)
            {
                weights[col] = make_value((col * 13 + 7) % 97 + 1);
                for (std::size_t row = 0; row < row_count; ++row)
                {
                    columns[row][col] = make_value(
                        (col * (row + 3) + row * 11 + 5) % 101 + 1);
                }
            }
            for (std::size_t row = 0; row < row_count; ++row)
            {
                column_lengths[row] = domain_size - row * 3;
            }

            for (const auto rotation_count : {
                     std::size_t{1},
                     std::size_t{2},
                     std::size_t{4}})
            {
                const std::vector<std::size_t> rotations =
                    rotation_count == 1
                        ? std::vector<std::size_t>{0}
                        : rotation_count == 2
                              ? std::vector<std::size_t>{1, domain_size - 3}
                              : std::vector<std::size_t>{
                                    0,
                                    1,
                                    domain_size / 2,
                                    domain_size - 1};
                const auto expected = expected_rotated_results(
                    weights,
                    columns,
                    column_lengths,
                    rotations);
                const auto actual = run_new_rotated_batch(
                    weights,
                    columns,
                    rotations,
                    column_lengths);
                require_rotated_equal(
                    actual,
                    expected,
                    "multi-rotation correctness");
            }
        }

        const std::array<std::uint64_t, 4> sentinel = {
            0xaaaaaaaaaaaaaaaaULL,
            0xbbbbbbbbbbbbbbbbULL,
            0xccccccccccccccccULL,
            0xddddddddddddddddULL,
        };
        std::array<std::uint64_t, 4> invalid_output = sentinel;
        const std::array<std::size_t, 1> one_rotation = {0};
        const std::array<const std::uint64_t *, 1> no_columns = {nullptr};
        if (gatzk::algebra::weighted_poly_eval_gpu_multi_rotations(
                nullptr,
                no_columns.data(),
                invalid_output.data(),
                8,
                1,
                one_rotation.data(),
                1))
        {
            throw std::runtime_error(
                "multi-rotation API accepted null input pointers");
        }
        require_unchanged(
            invalid_output,
            sentinel,
            "multi-rotation null guard");

        invalid_output = sentinel;
        std::vector<mcl::Fr> tiny_weights(8);
        std::vector<mcl::Fr> tiny_column(8);
        const std::array<const std::uint64_t *, 1> tiny_columns = {
            reinterpret_cast<const std::uint64_t *>(tiny_column.data()),
        };
        if (gatzk::algebra::weighted_poly_eval_gpu_multi_rotations(
                reinterpret_cast<const std::uint64_t *>(tiny_weights.data()),
                tiny_columns.data(),
                invalid_output.data(),
                8,
                1,
                one_rotation.data(),
                65536))
        {
            throw std::runtime_error(
                "multi-rotation API accepted rotation_count=65536");
        }
        require_unchanged(
            invalid_output,
            sentinel,
            "multi-rotation rotation-count guard");

        invalid_output = sentinel;
        if (gatzk::algebra::weighted_poly_eval_gpu_multi_rotations(
                reinterpret_cast<const std::uint64_t *>(tiny_weights.data()),
                tiny_columns.data(),
                invalid_output.data(),
                8,
                65536,
                one_rotation.data(),
                1))
        {
            throw std::runtime_error(
                "multi-rotation API accepted poly_count=65536");
        }
        require_unchanged(
            invalid_output,
            sentinel,
            "multi-rotation poly-count guard");

        const std::array<const std::uint64_t *, 1> zero_columns = {nullptr};
        const std::array<std::size_t, 1> zero_lengths = {0};
        std::vector<mcl::Fr> zero_output(1);
        zero_output[0] = make_value(7);
        if (!gatzk::algebra::weighted_poly_eval_gpu_multi_rotations(
                reinterpret_cast<const std::uint64_t *>(tiny_weights.data()),
                zero_columns.data(),
                reinterpret_cast<std::uint64_t *>(zero_output.data()),
                8,
                1,
                one_rotation.data(),
                1,
                zero_lengths.data()))
        {
            throw std::runtime_error(
                "multi-rotation API rejected a zero-length null column");
        }
        const std::vector<mcl::Fr> zero_expected = {make_value(0)};
        require_equal(
            zero_output,
            zero_expected,
            "multi-rotation zero-length column");

        zero_output[0] = make_value(7);
        if (!gatzk::algebra::weighted_poly_eval_gpu_multi(
                reinterpret_cast<const std::uint64_t *>(tiny_weights.data()),
                zero_columns.data(),
                reinterpret_cast<std::uint64_t *>(zero_output.data()),
                8,
                1,
                zero_lengths.data()))
        {
            throw std::runtime_error(
                "multi weighted API rejected a zero-length null column");
        }
        require_equal(
            zero_output,
            zero_expected,
            "multi zero-length column");

        invalid_output = sentinel;
        if (gatzk::algebra::weighted_poly_eval_gpu_multi_rotations(
                reinterpret_cast<const std::uint64_t *>(tiny_weights.data()),
                tiny_columns.data(),
                invalid_output.data(),
                std::numeric_limits<std::size_t>::max(),
                1,
                one_rotation.data(),
                1))
        {
            throw std::runtime_error(
                "multi-rotation API accepted an overflowing domain size");
        }
        require_unchanged(
            invalid_output,
            sentinel,
            "multi-rotation overflow guard");

        const std::size_t benchmark_domain = std::size_t{1} << 14;
        constexpr std::size_t benchmark_rows = 8;
        const std::vector<std::size_t> benchmark_rotations = {0, 1, 17, 257};
        std::vector<mcl::Fr> benchmark_weights(benchmark_domain);
        std::vector<std::vector<mcl::Fr>> benchmark_columns(
            benchmark_rows,
            std::vector<mcl::Fr>(benchmark_domain));
        std::vector<std::size_t> benchmark_lengths(
            benchmark_rows,
            benchmark_domain);
        for (std::size_t col = 0; col < benchmark_domain; ++col)
        {
            benchmark_weights[col] = make_value((col * 5 + 19) % 127 + 1);
            for (std::size_t row = 0; row < benchmark_rows; ++row)
            {
                benchmark_columns[row][col] = make_value(
                    (col * (row + 5) + row * 17 + 3) % 131 + 1);
            }
        }
        const auto benchmark_expected = expected_rotated_results(
            benchmark_weights,
            benchmark_columns,
            benchmark_lengths,
            benchmark_rotations);

        // Warm up both routes once, then collect short matched samples.  The
        // counters are deltas around each route, so unrelated CUDA work is
        // excluded from the comparison.
        (void)run_old_rotated_loop(
            benchmark_weights,
            benchmark_columns,
            benchmark_rotations);
        (void)run_new_rotated_batch(
            benchmark_weights,
            benchmark_columns,
            benchmark_rotations,
            benchmark_lengths);

        constexpr std::size_t kRepeats = 5;
        std::vector<std::uint64_t> old_ns;
        std::vector<std::uint64_t> new_ns;
        old_ns.reserve(kRepeats);
        new_ns.reserve(kRepeats);
        for (std::size_t repeat = 0; repeat < kRepeats; ++repeat)
        {
            const auto old_kernel_before =
                gatzk::algebra::cuda_kernel_invocation_count();
            const auto old_stats_before =
                gatzk::algebra::cuda_execution_stats_snapshot();
            const auto old_start = std::chrono::steady_clock::now();
            const auto old_output = run_old_rotated_loop(
                benchmark_weights,
                benchmark_columns,
                benchmark_rotations);
            const auto old_end = std::chrono::steady_clock::now();
            const auto old_stats_after =
                gatzk::algebra::cuda_execution_stats_snapshot();
            const auto old_kernel_after =
                gatzk::algebra::cuda_kernel_invocation_count();
            require_rotated_equal(
                old_output,
                benchmark_expected,
                "old rotated loop correctness");
            old_ns.push_back(static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    old_end - old_start)
                    .count()));

            const auto new_kernel_before =
                gatzk::algebra::cuda_kernel_invocation_count();
            const auto new_stats_before =
                gatzk::algebra::cuda_execution_stats_snapshot();
            const auto new_start = std::chrono::steady_clock::now();
            const auto new_output = run_new_rotated_batch(
                benchmark_weights,
                benchmark_columns,
                benchmark_rotations,
                benchmark_lengths);
            const auto new_end = std::chrono::steady_clock::now();
            const auto new_stats_after =
                gatzk::algebra::cuda_execution_stats_snapshot();
            const auto new_kernel_after =
                gatzk::algebra::cuda_kernel_invocation_count();
            require_rotated_equal(
                new_output,
                benchmark_expected,
                "new rotated batch correctness");
            new_ns.push_back(static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    new_end - new_start)
                    .count()));

            if (repeat == kRepeats - 1)
            {
                const auto old_h2d = old_stats_after.h2d_bytes -
                                     old_stats_before.h2d_bytes;
                const auto old_d2h = old_stats_after.d2h_bytes -
                                     old_stats_before.d2h_bytes;
                const auto old_sync = old_stats_after.stream_synchronizations -
                                      old_stats_before.stream_synchronizations;
                const auto new_h2d = new_stats_after.h2d_bytes -
                                     new_stats_before.h2d_bytes;
                const auto new_d2h = new_stats_after.d2h_bytes -
                                     new_stats_before.d2h_bytes;
                const auto new_sync = new_stats_after.stream_synchronizations -
                                      new_stats_before.stream_synchronizations;
                std::cout << "rotated A/B domain=" << benchmark_domain
                          << " rotations=" << benchmark_rotations.size()
                          << " old_kernel="
                          << (old_kernel_after - old_kernel_before)
                          << " new_kernel="
                          << (new_kernel_after - new_kernel_before)
                          << " old_h2d=" << old_h2d
                          << " new_h2d=" << new_h2d
                          << " old_d2h=" << old_d2h
                          << " new_d2h=" << new_d2h
                          << " old_sync=" << old_sync
                          << " new_sync=" << new_sync << '\n';
            }
        }

        const auto old_median = median_of(old_ns);
        const auto new_median = median_of(new_ns);
        std::cout << std::fixed << std::setprecision(3)
                  << "rotated A/B wall_ns old_median=" << old_median
                  << " old_variance=" << variance_of(old_ns, old_median)
                  << " new_median=" << new_median
                  << " new_variance=" << variance_of(new_ns, new_median)
                  << " ratio=" << (new_median == 0
                                         ? 0.0
                                         : static_cast<double>(new_median) /
                                               static_cast<double>(old_median))
                  << '\n';
    }
}

int main()
{
    if (!gatzk::algebra::cuda_backend_available())
    {
        return 77;
    }

    try
    {
        gatzk::algebra::ensure_mcl_field_ready();
        run_guard_cases();
        run_small_cases();
        run_rotated_cases();
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }

    std::cout << "CUDA weighted evaluation runtime checks passed\n";
    return 0;
}
