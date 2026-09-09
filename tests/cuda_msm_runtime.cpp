#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/cuda_msm.hpp"
#include "gatzk/crypto/curve.hpp"
#include "gatzk/util/route2.hpp"

namespace
{
    using gatzk::algebra::FieldElement;
    using gatzk::crypto::G1Point;

    constexpr std::size_t kBasisCount = 16;

    std::array<G1Point, kBasisCount>
    build_basis_cycle()
    {
        gatzk::algebra::ensure_mcl_field_ready();

        std::array<G1Point, kBasisCount> bases;

        G1Point generator;
        generator.value =
            mcl::bn::getG1basePoint();

        bases[0] = generator;

        for (std::size_t index = 1;
             index < bases.size();
             ++index)
        {
            bases[index] =
                gatzk::crypto::g1_add(
                    bases[index - 1],
                    generator);
        }

        return bases;
    }

    void run_msm_case(
        std::size_t count,
        std::size_t batch_size)
    {
        const auto bases =
            build_basis_cycle();

        std::vector<G1Point>
            points(count);

        std::vector<
            std::vector<FieldElement>>
            scalar_vectors(
                batch_size,
                std::vector<FieldElement>(
                    count,
                    FieldElement::zero()));

        std::vector<
            std::array<
                FieldElement,
                kBasisCount>>
            grouped_sums(batch_size);

        for (std::size_t index = 0;
             index < count;
             ++index)
        {
            const auto basis_index =
                index % kBasisCount;

            points[index] =
                bases[basis_index];

            for (std::size_t batch = 0;
                 batch < batch_size;
                 ++batch)
            {
                const auto raw =
                    static_cast<std::uint64_t>(
                        (index % 104729U) + 1U) *
                        static_cast<std::uint64_t>(
                            batch + 3U) +
                    static_cast<std::uint64_t>(
                        17U + basis_index);

                const FieldElement scalar(raw);

                scalar_vectors[batch][index] =
                    scalar;

                grouped_sums[batch][basis_index] +=
                    scalar;
            }
        }

        std::vector<G1Point>
            expected(
                batch_size,
                gatzk::crypto::g1_zero());

        for (std::size_t batch = 0;
             batch < batch_size;
             ++batch)
        {
            for (std::size_t basis_index = 0;
                 basis_index < kBasisCount;
                 ++basis_index)
            {
                expected[batch] =
                    gatzk::crypto::g1_add(
                        expected[batch],
                        gatzk::crypto::g1_mul(
                            bases[basis_index],
                            grouped_sums[batch][basis_index]));
            }
        }

        std::vector<G1Point>
            actual(
                batch_size,
                gatzk::crypto::g1_zero());

        if (batch_size == 1)
        {
            if (!gatzk::crypto::try_cuda_g1_msm(
                    &actual[0],
                    points.data(),
                    scalar_vectors[0].data(),
                    count,
                    true,
                    count))
            {
                throw std::runtime_error(
                    "CUDA MSM regression unexpectedly declined the selected CUDA route");
            }
        }
        else
        {
            std::vector<G1Point *>
                outputs(batch_size);

            std::vector<
                const FieldElement *>
                scalar_inputs(batch_size);

            std::vector<std::size_t>
                counts(batch_size, count);

            for (std::size_t batch = 0;
                 batch < batch_size;
                 ++batch)
            {
                outputs[batch] =
                    &actual[batch];

                scalar_inputs[batch] =
                    scalar_vectors[batch].data();
            }

            if (!gatzk::crypto::try_cuda_g1_msm_same_bases_batch(
                    outputs.data(),
                    points.data(),
                    scalar_inputs.data(),
                    counts.data(),
                    count,
                    batch_size,
                    count))
            {
                throw std::runtime_error(
                    "CUDA MSM batch regression unexpectedly declined the selected CUDA route");
            }
        }

        for (std::size_t batch = 0;
             batch < batch_size;
             ++batch)
        {
            if (!(actual[batch] ==
                  expected[batch]))
            {
                throw std::runtime_error(
                    "CUDA MSM result differs from the independent grouped mcl result");
            }
        }

        gatzk::crypto::
            release_cuda_g1_msm_resident_cache();
    }

    void run_mixed_count_case()
    {
        constexpr std::size_t kBatchSize = 4;
        constexpr std::size_t kMaximumCount = 16384;
        const std::array<std::size_t, kBatchSize> counts{
            8193,
            12289,
            16384,
            10001};

        const auto bases = build_basis_cycle();
        std::vector<G1Point> points(kMaximumCount);
        for (std::size_t index = 0;
             index < points.size();
             ++index)
        {
            points[index] = bases[index % kBasisCount];
        }

        // Keep a non-zero guard tail after each logical scalar vector.  A
        // mixed-count implementation that accidentally copies max_count
        // scalars would therefore change the result; the expected MSM only
        // includes the request's declared count.
        std::vector<std::vector<FieldElement>> scalar_vectors;
        scalar_vectors.reserve(kBatchSize);
        std::vector<std::array<FieldElement, kBasisCount>> grouped_sums(
            kBatchSize);
        for (std::size_t batch = 0;
             batch < kBatchSize;
             ++batch)
        {
            auto &scalars = scalar_vectors.emplace_back(
                kMaximumCount,
                FieldElement(
                    0x5a0000ULL +
                    static_cast<std::uint64_t>(batch)));
            for (std::size_t index = 0;
                 index < counts[batch];
                 ++index)
            {
                const auto basis_index = index % kBasisCount;
                const auto raw =
                    (static_cast<std::uint64_t>(index % 104729U) + 1U) *
                        static_cast<std::uint64_t>(batch + 11U) +
                    static_cast<std::uint64_t>(29U + basis_index);
                scalars[index] = FieldElement(raw);
                grouped_sums[batch][basis_index] += scalars[index];
            }
        }

        std::vector<G1Point> expected(
            kBatchSize,
            gatzk::crypto::g1_zero());
        for (std::size_t batch = 0;
             batch < kBatchSize;
             ++batch)
        {
            for (std::size_t basis_index = 0;
                 basis_index < kBasisCount;
                 ++basis_index)
            {
                expected[batch] = gatzk::crypto::g1_add(
                    expected[batch],
                    gatzk::crypto::g1_mul(
                        bases[basis_index],
                        grouped_sums[batch][basis_index]));
            }
        }

        const std::array<
            gatzk::crypto::CudaG1MsmBatchTestRoute,
            3>
            routes{
                gatzk::crypto::CudaG1MsmBatchTestRoute::Resident,
                gatzk::crypto::CudaG1MsmBatchTestRoute::CachedChunked,
                gatzk::crypto::CudaG1MsmBatchTestRoute::Streamed};
        const std::array<const char *, 3> route_names{
            "resident",
            "cached_chunked",
            "streamed"};

        for (std::size_t route_index = 0;
             route_index < routes.size();
             ++route_index)
        {
            gatzk::crypto::release_cuda_g1_msm_resident_cache();
            std::vector<G1Point> actual(
                kBatchSize,
                gatzk::crypto::g1_zero());
            std::vector<G1Point *> outputs(kBatchSize);
            std::vector<const FieldElement *> scalar_inputs(kBatchSize);
            for (std::size_t batch = 0;
                 batch < kBatchSize;
                 ++batch)
            {
                outputs[batch] = &actual[batch];
                scalar_inputs[batch] = scalar_vectors[batch].data();
            }

            if (!gatzk::crypto::try_cuda_g1_msm_same_bases_batch_for_tests(
                    outputs.data(),
                    points.data(),
                    scalar_inputs.data(),
                    counts.data(),
                    kMaximumCount,
                    kBatchSize,
                    kMaximumCount,
                    routes[route_index]))
            {
                throw std::runtime_error(
                    std::string("CUDA mixed-count route declined: ") +
                    route_names[route_index]);
            }

            for (std::size_t batch = 0;
                 batch < kBatchSize;
                 ++batch)
            {
                if (!(actual[batch] == expected[batch]))
                {
                    throw std::runtime_error(
                        std::string("CUDA mixed-count ") +
                        route_names[route_index] +
                        " route differs from logical-count reference");
                }
            }

            std::cout << "mixed_count_route="
                      << route_names[route_index]
                      << " counts=8193,12289,16384,10001"
                      << " status=EXACT\n";
        }
        gatzk::crypto::release_cuda_g1_msm_resident_cache();
    }
}

int main()
{
    if (!gatzk::algebra::
            cuda_backend_available())
    {
        return 77;
    }

    auto route =
        gatzk::util::route2_options();

    route.cuda_field_hotspots = true;

    gatzk::util::
        set_route2_options(route);

    try
    {
        run_mixed_count_case();

        run_msm_case(
            8193,
            1);

        run_msm_case(
            1ULL << 20,
            2);
    }
    catch (const std::exception &error)
    {
        std::cerr
            << error.what()
            << '\n';

        return 1;
    }

    std::cout
        << "cuda msm regression checks passed\n";

    return 0;
}
