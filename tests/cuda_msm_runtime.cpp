#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
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