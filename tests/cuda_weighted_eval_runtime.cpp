#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }

    std::cout << "CUDA weighted evaluation runtime checks passed\n";
    return 0;
}
