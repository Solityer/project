#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/vector_ops.hpp"

namespace
{
using Clock = std::chrono::steady_clock;

double elapsed_ms(
    Clock::time_point begin)
{
    return std::chrono::duration<double, std::milli>(
        Clock::now() - begin)
        .count();
}

std::vector<mcl::Fr> make_input(
    std::size_t size,
    std::uint64_t seed)
{
    std::vector<mcl::Fr> values(size);

    mcl::Fr current;
    current = seed;

    mcl::Fr step;
    step = 17;

    for (std::size_t i = 0; i < size; ++i)
    {
        values[i] = current;
        mcl::Fr::add(
            current,
            current,
            step);
    }

    return values;
}
}

int main(
    int argc,
    char** argv)
{
    try
    {
        gatzk::algebra::ensure_mcl_field_ready();

        const auto size =
            argc > 1
                ? static_cast<std::size_t>(
                      std::stoull(argv[1]))
                : (std::size_t{1} << 20);

        const auto repeats =
            argc > 2
                ? static_cast<std::size_t>(
                      std::stoull(argv[2]))
                : std::size_t{3};

        if (argc > 3
            || repeats == 0
            || size == 0
            || (size & (size - 1U)) != 0U)
        {
            throw std::runtime_error(
                "usage: gatzk_cuda_ntt_stage_bench "
                "[power_of_two_size] [repeats]");
        }

        // Initialize the persistent 2^24 ICICLE domain outside the repeated
        // samples, but report that one-time setup cost separately.
        auto warm_gpu =
            make_input(
                std::size_t{1} << 10,
                3);

        auto warm_cpu = warm_gpu;

        gatzk::algebra::ntt_plan(
            warm_cpu.size())
            ->intt(warm_cpu.data());

        auto start = Clock::now();

        if (!gatzk::algebra::
                try_cuda_intt_in_place(
                    warm_gpu.data(),
                    warm_gpu.size()))
        {
            throw std::runtime_error(
                "CUDA NTT backend unavailable "
                "during domain warmup");
        }

        const auto initialization_ms =
            elapsed_ms(start);

        if (warm_cpu != warm_gpu)
        {
            throw std::runtime_error(
                "CUDA NTT warmup INTT differs "
                "from MCL");
        }

        const auto first_input =
            make_input(size, 11);

        const auto second_input =
            make_input(size, 29);

        std::cout
            << std::fixed
            << std::setprecision(3)
            << "cuda_domain_init_ms="
            << initialization_ms
            << " warmup_exact=true\n";

        std::cout
            << "size,repeat,"
            << "cpu_pair_ms,gpu_pair_ms,pair_speedup,"
            << "cpu_intt_ms,gpu_intt_ms,intt_speedup,"
            << "exact\n";

        for (std::size_t repeat = 0;
             repeat < repeats;
             ++repeat)
        {
            double cpu_pair_ms = 0.0;
            double gpu_pair_ms = 0.0;
            bool pair_exact = false;

            {
                auto cpu_first = first_input;
                auto cpu_second = second_input;

                start = Clock::now();

                const auto plan =
                    gatzk::algebra::ntt_plan(size);

                plan->ntt(cpu_first.data());
                plan->ntt(cpu_second.data());

                cpu_pair_ms =
                    elapsed_ms(start);

                auto gpu_first = first_input;
                auto gpu_second = second_input;

                start = Clock::now();

                if (!gatzk::algebra::
                        try_cuda_ntt_pair_in_place(
                            gpu_first.data(),
                            gpu_second.data(),
                            size))
                {
                    throw std::runtime_error(
                        "CUDA paired NTT route "
                        "was not admitted");
                }

                gpu_pair_ms =
                    elapsed_ms(start);

                pair_exact =
                    cpu_first == gpu_first
                    && cpu_second == gpu_second;
            }

            double cpu_intt_ms = 0.0;
            double gpu_intt_ms = 0.0;
            bool intt_exact = false;

            {
                auto cpu_inverse =
                    first_input;

                start = Clock::now();

                gatzk::algebra::ntt_plan(size)
                    ->intt(cpu_inverse.data());

                cpu_intt_ms =
                    elapsed_ms(start);

                auto gpu_inverse =
                    first_input;

                start = Clock::now();

                if (!gatzk::algebra::
                        try_cuda_intt_in_place(
                            gpu_inverse.data(),
                            size))
                {
                    throw std::runtime_error(
                        "CUDA INTT route "
                        "was not admitted");
                }

                gpu_intt_ms =
                    elapsed_ms(start);

                intt_exact =
                    cpu_inverse == gpu_inverse;
            }

            const bool exact =
                pair_exact && intt_exact;

            std::cout
                << size << ','
                << repeat << ','
                << cpu_pair_ms << ','
                << gpu_pair_ms << ','
                << cpu_pair_ms / gpu_pair_ms << ','
                << cpu_intt_ms << ','
                << gpu_intt_ms << ','
                << cpu_intt_ms / gpu_intt_ms << ','
                << (exact ? "true" : "false")
                << '\n';

            if (!exact)
            {
                throw std::runtime_error(
                    "CUDA NTT output differs "
                    "from exact MCL output");
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "fatal: "
            << error.what()
            << '\n';

        return 1;
    }

    return 0;
}