#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/util/route2.hpp"

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
    char **argv)
{
    try
    {
        gatzk::algebra::ensure_mcl_field_ready();

        const auto intt_size =
            argc > 1
                ? static_cast<std::size_t>(
                      std::stoull(argv[1]))
                : (std::size_t{1} << 20);

        const auto pair_size =
            argc > 2
                ? static_cast<std::size_t>(
                      std::stoull(argv[2]))
                : intt_size;

        const auto repeats =
            argc > 3
                ? static_cast<std::size_t>(
                      std::stoull(argv[3]))
                : std::size_t{3};

        const auto valid_power_of_two =
            [](std::size_t value)
        {
            return value != 0 && (value & (value - 1U)) == 0U;
        };

        if (argc > 4 || repeats == 0 || !valid_power_of_two(intt_size) || !valid_power_of_two(pair_size))
        {
            throw std::runtime_error(
                "usage: gatzk_cuda_ntt_stage_bench "
                "[intt_power_of_two_size] "
                "[pair_power_of_two_size] "
                "[repeats]");
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

        const auto inverse_input =
            make_input(intt_size, 11);

        const auto first_input =
            make_input(pair_size, 11);

        const auto second_input =
            make_input(pair_size, 29);

        std::cout
            << std::fixed
            << std::setprecision(3)
            << "cuda_domain_init_ms="
            << initialization_ms
            << " warmup_exact=true\n";

        std::cout
            << "cuda_device_summary="
            << gatzk::algebra::cuda_device_summary()
            << '\n';

        std::cout
            << "intt_size,pair_size,repeat,"
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

                gatzk::algebra::ntt_pair_in_place(
                    cpu_first.data(),
                    cpu_second.data(),
                    pair_size);

                cpu_pair_ms =
                    elapsed_ms(start);

                auto gpu_first = first_input;
                auto gpu_second = second_input;

                start = Clock::now();

                if (!gatzk::algebra::
                        try_cuda_ntt_pair_in_place(
                            gpu_first.data(),
                            gpu_second.data(),
                            pair_size))
                {
                    throw std::runtime_error(
                        "CUDA paired NTT route "
                        "was not admitted");
                }

                gpu_pair_ms =
                    elapsed_ms(start);

                pair_exact =
                    cpu_first == gpu_first && cpu_second == gpu_second;
            }

            double cpu_intt_ms = 0.0;
            double gpu_intt_ms = 0.0;
            bool intt_exact = false;

            {
                auto cpu_inverse =
                    inverse_input;

                start = Clock::now();

                gatzk::algebra::intt_in_place(
                    cpu_inverse.data(),
                    intt_size);

                cpu_intt_ms =
                    elapsed_ms(start);

                auto gpu_inverse =
                    inverse_input;

                start = Clock::now();

                if (!gatzk::algebra::
                        try_cuda_intt_in_place(
                            gpu_inverse.data(),
                            intt_size))
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
                << intt_size << ','
                << pair_size << ','
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
        // Build the exact CPU reference before enabling the production CUDA
        // route.  This keeps the direct CPU/GPU measurements above unchanged.
        auto reference_first =
            first_input;

        auto reference_second =
            second_input;

        gatzk::algebra::ntt_pair_in_place(
            reference_first.data(),
            reference_second.data(),
            pair_size);

        auto production_options =
            gatzk::util::route2_options();

        production_options.cuda_field_hotspots =
            true;

        gatzk::util::set_route2_options(
            production_options);

        const bool production_uses_cuda =
            gatzk::algebra::should_use_cuda_ntt_pair(
                pair_size);

        auto production_first =
            first_input;

        auto production_second =
            second_input;

        start = Clock::now();

        gatzk::algebra::ntt_pair_in_place(
            production_first.data(),
            production_second.data(),
            pair_size);

        const auto production_pair_ms =
            elapsed_ms(start);

        const bool production_exact =
            production_first == reference_first &&
            production_second == reference_second;

        std::cout
            << "production_pair_size="
            << pair_size
            << " production_pair_route="
            << (production_uses_cuda ? "cuda" : "cpu")
            << " production_pair_ms="
            << production_pair_ms
            << " production_pair_exact="
            << (production_exact ? "true" : "false")
            << '\n';

        if (!production_exact)
        {
            throw std::runtime_error(
                "production paired NTT route differs "
                "from the exact CPU result");
        }
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "fatal: "
            << error.what()
            << '\n';

        return 1;
    }

    return 0;
}