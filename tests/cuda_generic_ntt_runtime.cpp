#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/vector_ops.hpp"

namespace
{
    using gatzk::algebra::FieldElement;

    std::size_t current_rss_bytes()
    {
        std::ifstream input("/proc/self/statm");
        std::size_t pages = 0;
        std::size_t resident_pages = 0;
        if (!(input >> pages >> resident_pages))
        {
            return 0;
        }
        (void)pages;
        const auto page_size = ::sysconf(_SC_PAGESIZE);
        return page_size > 0
                   ? resident_pages * static_cast<std::size_t>(page_size)
                   : 0;
    }

    std::size_t peak_rss_bytes()
    {
        std::ifstream input("/proc/self/status");
        std::string key;
        std::size_t value = 0;
        std::string unit;
        while (input >> key >> value >> unit)
        {
            if (key == "VmHWM:")
            {
                return value * 1024;
            }
        }
        return 0;
    }

    std::size_t parse_log2(int argc, char **argv)
    {
        if (argc == 1)
        {
            return 25;
        }
        if (argc != 3 || std::string(argv[1]) != "--log2")
        {
            throw std::runtime_error("usage: gatzk_cuda_generic_ntt_runtime [--log2 N]");
        }
        const auto value = std::stoull(argv[2]);
        if (value < 1 || value >= sizeof(std::size_t) * 8)
        {
            throw std::runtime_error("invalid --log2");
        }
        return static_cast<std::size_t>(value);
    }

    std::vector<mcl::Fr> make_values(std::size_t size, std::uint64_t offset)
    {
        std::vector<mcl::Fr> values(size);
        for (std::size_t index = 0; index < size; ++index)
        {
            const auto value =
                19ULL + offset +
                ((static_cast<std::uint64_t>(index) * 2654435761ULL) %
                 1000003ULL);
            values[index] = FieldElement(value).native();
        }
        return values;
    }

    bool exact_equal(
        const std::vector<mcl::Fr> &lhs,
        const std::vector<mcl::Fr> &rhs)
    {
        return lhs.size() == rhs.size() &&
               std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    void cpu_ntt(std::vector<mcl::Fr> &values)
    {
        gatzk::algebra::ntt_plan(values.size())->ntt(values.data());
    }

    void cpu_intt(std::vector<mcl::Fr> &values)
    {
        gatzk::algebra::ntt_plan(values.size())->intt(values.data());
    }

    void run_small_cpu_exact()
    {
        constexpr std::size_t size = 1ULL << 12;
        auto original = make_values(size, 0);
        auto transformed = original;
        cpu_ntt(transformed);
        auto roundtrip = transformed;
        cpu_intt(roundtrip);
        if (!exact_equal(roundtrip, original))
        {
            throw std::runtime_error("2^12 CPU NTT/INTT roundtrip mismatch");
        }
        std::cout << "small_cpu_log2=12 exact=1 roundtrip_exact=1\n";
    }

    int run_large_cuda_exact(std::size_t log2)
    {
        if (!gatzk::algebra::cuda_backend_available())
        {
            std::cout << "status=unavailable log2=" << log2 << '\n';
            return 77;
        }

        if (log2 > 26)
        {
            throw std::runtime_error("generic staged runtime is bounded at 2^26");
        }

        const auto size = std::size_t{1} << log2;
        auto original = make_values(size, 0);
        auto second_original = make_values(size, 7919);

        auto cpu_forward = original;
        cpu_ntt(cpu_forward);
        auto cpu_second_forward = second_original;
        cpu_ntt(cpu_second_forward);

        const auto stats_before =
            gatzk::algebra::cuda_execution_stats_snapshot();
        const auto kernels_before =
            gatzk::algebra::cuda_kernel_invocation_count();
        const auto rss_before = current_rss_bytes();

        auto cuda_inverse = cpu_forward;
        if (!gatzk::algebra::try_cuda_intt_in_place(
                cuda_inverse.data(),
                size))
        {
            gatzk::algebra::release_cuda_ntt_workspace();
            std::cout << "status=admission_declined log2=" << log2
                      << " operation=intt rss_before=" << rss_before
                      << " rss_after=" << current_rss_bytes()
                      << " rss_hwm=" << peak_rss_bytes() << '\n';
            return 77;
        }
        const bool intt_exact = exact_equal(cuda_inverse, original);

        auto cuda_forward_roundtrip = cuda_inverse;
        if (!gatzk::algebra::try_cuda_ntt_in_place(
                cuda_forward_roundtrip.data(),
                size))
        {
            gatzk::algebra::release_cuda_ntt_workspace();
            throw std::runtime_error("CUDA NTT was declined after exact INTT");
        }
        const bool ntt_exact =
            exact_equal(cuda_forward_roundtrip, cpu_forward);
        if (!gatzk::algebra::try_cuda_intt_in_place(
                cuda_forward_roundtrip.data(),
                size))
        {
            gatzk::algebra::release_cuda_ntt_workspace();
            throw std::runtime_error("CUDA INTT was declined after exact NTT");
        }
        const bool roundtrip_exact =
            exact_equal(cuda_forward_roundtrip, original);

        auto cuda_first_pair = original;
        auto cuda_second_pair = second_original;
        if (!gatzk::algebra::try_cuda_ntt_pair_in_place(
                cuda_first_pair.data(),
                cuda_second_pair.data(),
                size))
        {
            gatzk::algebra::release_cuda_ntt_workspace();
            throw std::runtime_error("CUDA paired NTT was declined");
        }
        const bool pair_exact =
            exact_equal(cuda_first_pair, cpu_forward) &&
            exact_equal(cuda_second_pair, cpu_second_forward);

        const auto stats_after =
            gatzk::algebra::cuda_execution_stats_snapshot();
        const auto kernels_after =
            gatzk::algebra::cuda_kernel_invocation_count();
        const auto rss_after = current_rss_bytes();
        const auto delta = [](std::uint64_t after, std::uint64_t before)
        {
            return after >= before ? after - before : 0;
        };

        const auto batch_bytes = size * 2 * sizeof(mcl::Fr);
        const auto staged_workspace_bytes = batch_bytes * 4;
        gatzk::algebra::release_cuda_ntt_workspace();

        std::cout << "status=ok"
                  << " log2=" << log2
                  << " size=" << size
                  << " route=generic_staged_cuda"
                  << " intt_exact=" << (intt_exact ? 1 : 0)
                  << " ntt_exact=" << (ntt_exact ? 1 : 0)
                  << " roundtrip_exact=" << (roundtrip_exact ? 1 : 0)
                  << " pair_exact=" << (pair_exact ? 1 : 0)
                  << " batch=2"
                  << " workspace_bytes=" << staged_workspace_bytes
                  << " kernel_invocations="
                  << (kernels_after - kernels_before)
                  << " ntt=" << delta(stats_after.ntt_calls, stats_before.ntt_calls)
                  << " intt=" << delta(stats_after.intt_calls, stats_before.intt_calls)
                  << " h2d_bytes=" << delta(stats_after.h2d_bytes, stats_before.h2d_bytes)
                  << " d2h_bytes=" << delta(stats_after.d2h_bytes, stats_before.d2h_bytes)
                  << " sync=" << delta(
                         stats_after.stream_synchronizations,
                         stats_before.stream_synchronizations)
                  << " rss_before=" << rss_before
                  << " rss_after=" << rss_after
                  << " rss_after_release=" << current_rss_bytes()
                  << " rss_hwm=" << peak_rss_bytes()
                  << '\n';

        if (!intt_exact || !ntt_exact || !roundtrip_exact || !pair_exact)
        {
            return 1;
        }
        return 0;
    }
}

int main(int argc, char **argv)
{
    try
    {
        run_small_cpu_exact();
        const auto log2 = parse_log2(argc, argv);
        if (log2 <= 12)
        {
            return 0;
        }
        return run_large_cuda_exact(log2);
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
