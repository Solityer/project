#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

#include <cuda_runtime_api.h>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/protocol/cuda_t_edge.hpp"
#include "gatzk/protocol/t_edge_program.hpp"

namespace
{
    using gatzk::algebra::FieldElement;
    using gatzk::protocol::CudaTEdgeExecutor;
    using gatzk::protocol::TEdgeBoundProgram;
    using gatzk::protocol::TEdgeProgramAccess;
    using gatzk::protocol::TEdgeProgramInstruction;
    using gatzk::protocol::TEdgeProgramOpcode;
    using gatzk::protocol::TEdgeProgramTerm;

    using Clock = std::chrono::steady_clock;

    struct SourceLeaf
    {
        std::vector<FieldElement> evaluations;
        std::vector<FieldElement> blinder;
        std::vector<FieldElement> semantic_coset;
        std::vector<FieldElement> blinded_coset;
    };

    struct Fixture
    {
        std::size_t domain_size = 0;
        std::size_t target_size = 0;
        std::size_t rotation_step = 0;
        FieldElement shift = FieldElement::one();
        std::vector<FieldElement> source_scales;
        FieldElement target_scale = FieldElement::one();
        std::vector<SourceLeaf> sources;
        std::vector<FieldElement> target_blinder;
        std::vector<FieldElement> target_coset;
        TEdgeBoundProgram program;
    };

    struct Sample
    {
        double host_materialization_ms = 0.0;
        double device_materialization_ms = 0.0;
        double upload_sync_ms = 0.0;
        double execute_readback_ms = 0.0;
        double total_ms = 0.0;
        std::uint64_t digest = 0;
        std::size_t rss_before = 0;
        std::size_t rss_after = 0;
        std::size_t vram_free_before = 0;
        std::size_t vram_free_after = 0;
        gatzk::algebra::CudaExecutionStats stats_before;
        gatzk::algebra::CudaExecutionStats stats_after;
        std::uint64_t kernel_count_before = 0;
        std::uint64_t kernel_count_after = 0;
    };

    [[noreturn]] void fail(const std::string &message)
    {
        throw std::runtime_error(message);
    }

    void check_cuda(cudaError_t status, const char *operation)
    {
        if (status != cudaSuccess)
        {
            fail(std::string(operation) + ": " + cudaGetErrorString(status));
        }
        gatzk::algebra::record_cuda_stream_sync();
    }

    std::size_t resident_bytes()
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
        if (page_size <= 0)
        {
            return 0;
        }
        return resident_pages * static_cast<std::size_t>(page_size);
    }

    std::size_t free_vram_bytes()
    {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
        {
            return 0;
        }
        return free_bytes;
    }

    std::uint64_t fixture_scalar(
        std::size_t leaf,
        std::size_t index,
        std::uint64_t salt)
    {
        // Keep the fixture deterministic while exercising non-zero field
        // values and independent source/target blinders.
        const auto value =
            (salt + (leaf + 1) * 1000003ULL + (index + 1) * 9176ULL) %
            1000000007ULL;
        return value == 0 ? 1 : value;
    }

    std::vector<FieldElement> make_blinder(
        std::size_t leaf,
        std::uint64_t salt)
    {
        std::vector<FieldElement> result;
        result.reserve(3);
        for (std::size_t index = 0; index < 3; ++index)
        {
            result.emplace_back(fixture_scalar(leaf, index, salt));
        }
        return result;
    }

    std::vector<FieldElement> materialize_cpu_coset(
        const std::vector<FieldElement> &evaluations,
        const std::vector<FieldElement> &blinder,
        std::size_t domain_size,
        std::size_t target_size,
        const FieldElement &shift)
    {
        if (evaluations.size() != domain_size ||
            domain_size == 0 ||
            target_size == 0 ||
            target_size % domain_size != 0 ||
            blinder.size() > target_size - domain_size)
        {
            fail("invalid CPU coset fixture shape");
        }

        std::vector<FieldElement> coefficients = evaluations;
        auto *native_coefficients =
            FieldElement::mutable_native_array_data(coefficients.data());
        gatzk::algebra::ntt_plan(domain_size)->intt(native_coefficients);

        coefficients.resize(target_size, FieldElement::zero());
        for (std::size_t index = 0; index < blinder.size(); ++index)
        {
            coefficients[index] -= blinder[index];
            coefficients[domain_size + index] += blinder[index];
        }

        auto power = FieldElement::one();
        for (auto &coefficient : coefficients)
        {
            coefficient *= power;
            power *= shift;
        }

        gatzk::algebra::ntt_plan(target_size)->ntt(
            FieldElement::mutable_native_array_data(coefficients.data()));
        return coefficients;
    }

    FieldElement evaluate_blinder(
        const std::vector<FieldElement> &blinder,
        const FieldElement &point)
    {
        FieldElement result = FieldElement::zero();
        for (auto it = blinder.rbegin(); it != blinder.rend(); ++it)
        {
            result *= point;
            result += *it;
        }
        return result;
    }

    std::vector<FieldElement> blinder_difference(
        const std::vector<FieldElement> &target,
        const std::vector<SourceLeaf> &sources,
        const std::vector<FieldElement> &scales)
    {
        if (sources.size() != scales.size())
        {
            fail("t_edge linear scale/source count mismatch");
        }
        std::vector<FieldElement> result(
            std::max<std::size_t>(3, target.size()),
            FieldElement::zero());
        for (std::size_t index = 0; index < target.size(); ++index)
        {
            result[index] += target[index];
        }
        for (std::size_t source_index = 0;
             source_index < sources.size();
             ++source_index)
        {
            const auto &source = sources[source_index];
            for (std::size_t index = 0; index < source.blinder.size(); ++index)
            {
                result[index] -= scales[source_index] * source.blinder[index];
            }
        }
        while (result.size() > 1 && result.back().is_zero())
        {
            result.pop_back();
        }
        return result;
    }

    std::vector<FieldElement> make_scaled_target_host(
        const Fixture &fixture,
        const std::vector<FieldElement> &correction)
    {
        std::vector<FieldElement> result(
            fixture.target_size,
            FieldElement::zero());

        const auto root = FieldElement::root_of_unity(fixture.target_size);
        const auto zero_step = root.pow(fixture.domain_size);
        const auto one = FieldElement::one();
        auto point = fixture.shift;
        auto zero_power = fixture.shift.pow(fixture.domain_size);

        for (std::size_t index = 0; index < fixture.target_size; ++index)
        {
            FieldElement value = FieldElement::zero();
            for (std::size_t source_index = 0;
                 source_index < fixture.sources.size();
                 ++source_index)
            {
                value += fixture.sources[source_index].blinded_coset[index] *
                         fixture.source_scales[source_index];
            }

            if (!correction.empty())
            {
                value += (zero_power - one) *
                         evaluate_blinder(correction, point);
            }
            result[index] = value;
            point *= root;
            zero_power *= zero_step;
        }
        return result;
    }

    std::vector<FieldElement> make_independent_target_reference(
        const Fixture &fixture)
    {
        std::vector<FieldElement> result(
            fixture.target_size,
            FieldElement::zero());
        const auto root = FieldElement::root_of_unity(fixture.target_size);
        const auto zero_step = root.pow(fixture.domain_size);
        const auto one = FieldElement::one();
        auto point = fixture.shift;
        auto zero_power = fixture.shift.pow(fixture.domain_size);

        for (std::size_t index = 0; index < fixture.target_size; ++index)
        {
            FieldElement value = FieldElement::zero();
            for (std::size_t source_index = 0;
                 source_index < fixture.sources.size();
                 ++source_index)
            {
                value += fixture.sources[source_index].semantic_coset[index] *
                         fixture.source_scales[source_index];
            }
            value += (zero_power - one) *
                     evaluate_blinder(fixture.target_blinder, point);
            result[index] = value;
            point *= root;
            zero_power *= zero_step;
        }
        return result;
    }

    TEdgeBoundProgram make_program(std::size_t source_count)
    {
        TEdgeBoundProgram program;
        program.accesses.reserve(source_count + 1);
        for (std::size_t index = 0; index < source_count; ++index)
        {
            program.accesses.push_back(TEdgeProgramAccess{
                .label = "source_" + std::to_string(index),
                .rotated = (index & 1U) != 0U,
            });
        }
        program.accesses.push_back(TEdgeProgramAccess{
            .label = "derived_target",
            .rotated = false,
        });

        const auto source_begin = program.instructions.size();
        program.instructions.push_back(TEdgeProgramInstruction{
            .opcode = TEdgeProgramOpcode::LoadAccess,
            .operand = 0,
        });
        for (std::size_t index = 1; index < source_count; ++index)
        {
            program.instructions.push_back(TEdgeProgramInstruction{
                .opcode = TEdgeProgramOpcode::LoadAccess,
                .operand = static_cast<std::uint32_t>(index),
            });
            program.instructions.push_back(TEdgeProgramInstruction{
                .opcode = TEdgeProgramOpcode::Add,
                .operand = 0,
            });
        }
        const auto source_end = program.instructions.size();

        const auto target_begin = program.instructions.size();
        program.instructions.push_back(TEdgeProgramInstruction{
            .opcode = TEdgeProgramOpcode::LoadAccess,
            .operand = static_cast<std::uint32_t>(source_count),
        });
        const auto target_end = program.instructions.size();

        program.terms.push_back(TEdgeProgramTerm{
            .instruction_begin = source_begin,
            .instruction_end = source_end,
            .alpha_scale = FieldElement(3),
            .required_accesses = [&]() {
                std::vector<std::uint32_t> result;
                result.reserve(source_count);
                for (std::size_t index = 0; index < source_count; ++index)
                {
                    result.push_back(static_cast<std::uint32_t>(index));
                }
                return result;
            }(),
            .maximum_stack_depth = 2,
        });
        program.terms.push_back(TEdgeProgramTerm{
            .instruction_begin = target_begin,
            .instruction_end = target_end,
            .alpha_scale = FieldElement(5),
            .required_accesses = {
                static_cast<std::uint32_t>(source_count),
            },
            .maximum_stack_depth = 1,
        });
        program.maximum_stack_depth = 2;
        return program;
    }

    Fixture make_fixture(
        std::size_t domain_size,
        std::size_t target_size,
        std::size_t source_count)
    {
        if (source_count < 2 || source_count > 6 || target_size != domain_size * 4)
        {
            fail("invalid t_edge linear benchmark shape");
        }

        Fixture fixture;
        fixture.domain_size = domain_size;
        fixture.target_size = target_size;
        fixture.rotation_step = target_size / domain_size;
        fixture.shift = FieldElement(7);
        fixture.source_scales.reserve(source_count);
        for (std::size_t source_index = 0;
             source_index < source_count;
             ++source_index)
        {
            fixture.source_scales.emplace_back(
                static_cast<std::uint64_t>(
                    2 + source_count + source_index * 3));
        }
        fixture.target_scale = FieldElement(5);
        fixture.target_blinder = make_blinder(source_count + 11, 0xA51CEULL);
        fixture.sources.resize(source_count);

        for (std::size_t leaf = 0; leaf < source_count; ++leaf)
        {
            auto &source = fixture.sources[leaf];
            source.evaluations.reserve(domain_size);
            for (std::size_t index = 0; index < domain_size; ++index)
            {
                source.evaluations.emplace_back(
                    fixture_scalar(leaf, index, 0xBEEFULL));
            }
            source.blinder = make_blinder(leaf, 0x123400ULL);
            source.semantic_coset = materialize_cpu_coset(
                source.evaluations,
                {},
                domain_size,
                target_size,
                fixture.shift);
            source.blinded_coset = materialize_cpu_coset(
                source.evaluations,
                source.blinder,
                domain_size,
                target_size,
                fixture.shift);
        }

        const auto correction = blinder_difference(
            fixture.target_blinder,
            fixture.sources,
            fixture.source_scales);
        fixture.target_coset = make_scaled_target_host(fixture, correction);
        const auto independent_target = make_independent_target_reference(fixture);
        if (fixture.target_coset != independent_target)
        {
            fail("CPU derived t_edge linear reference mismatch");
        }

        fixture.program = make_program(source_count);
        return fixture;
    }

    std::uint64_t digest_values(const std::vector<FieldElement> &values)
    {
        std::uint64_t digest = 1469598103934665603ULL;
        std::array<std::uint8_t, 64> bytes{};
        for (const auto &value : values)
        {
            bytes.fill(0);
            (void)value.write_little_endian(bytes.data(), bytes.size());
            for (const auto byte : bytes)
            {
                digest ^= byte;
                digest *= 1099511628211ULL;
            }
        }
        return digest;
    }

    void require_equal(
        const std::vector<FieldElement> &actual,
        const std::vector<FieldElement> &expected,
        std::string_view label)
    {
        if (actual.size() != expected.size())
        {
            fail(std::string(label) + " size mismatch");
        }
        for (std::size_t index = 0; index < actual.size(); ++index)
        {
            if (!(actual[index] == expected[index]))
            {
                fail(std::string(label) + " mismatch at index " +
                     std::to_string(index));
            }
        }
    }

    std::unique_ptr<CudaTEdgeExecutor> make_executor(const Fixture &fixture)
    {
        auto executor = gatzk::protocol::create_cuda_t_edge_executor(
            fixture.domain_size,
            fixture.target_size,
            fixture.shift,
            FieldElement::root_of_unity(fixture.domain_size),
            fixture.sources.size() + 1);
        if (executor == nullptr)
        {
            fail("CUDA t_edge executor factory returned null");
        }
        executor->prepare_program(fixture.program);
        return executor;
    }

    void materialize_resident_sources(
        CudaTEdgeExecutor &executor,
        const Fixture &fixture)
    {
        for (std::size_t index = 0; index < fixture.sources.size(); ++index)
        {
            const auto &source = fixture.sources[index];
            const auto *blinder = source.blinder.empty()
                                      ? nullptr
                                      : FieldElement::native_array_data(
                                            source.blinder.data());
            if (!executor.materialize_evaluation_leaf(
                    index,
                    FieldElement::native_array_data(source.evaluations.data()),
                    source.evaluations.size(),
                    blinder,
                    source.blinder.size()))
            {
                fail("CUDA resident source materialization declined");
            }
        }
    }

    std::vector<FieldElement> expected_term_output(const Fixture &fixture)
    {
        std::vector<FieldElement> expected(
            fixture.target_size,
            FieldElement::zero());
        const auto root = FieldElement::root_of_unity(fixture.target_size);
        const auto zero_step = root.pow(fixture.domain_size);
        const auto one = FieldElement::one();
        auto zero_power = fixture.shift.pow(fixture.domain_size);

        std::vector<std::size_t> rotated_indices(fixture.target_size);
        for (std::size_t index = 0; index < fixture.target_size; ++index)
        {
            rotated_indices[index] =
                (index + fixture.rotation_step) & (fixture.target_size - 1U);
        }

        std::array<FieldElement, 4> inverse_cycle{};
        auto cycle_power = zero_power;
        for (std::size_t index = 0; index < fixture.rotation_step; ++index)
        {
            inverse_cycle[index] = (cycle_power - one).inv();
            cycle_power *= zero_step;
        }

        for (std::size_t index = 0; index < fixture.target_size; ++index)
        {
            FieldElement source_sum = FieldElement::zero();
            for (std::size_t leaf = 0; leaf < fixture.sources.size(); ++leaf)
            {
                const auto source_index = (leaf & 1U) != 0U
                                              ? rotated_indices[index]
                                              : index;
                source_sum += fixture.sources[leaf].blinded_coset[source_index];
            }
            const auto numerator =
                source_sum * FieldElement(3) +
                fixture.target_coset[index] * fixture.target_scale;
            expected[index] = numerator * inverse_cycle[index % fixture.rotation_step];
        }
        return expected;
    }

    void prepare_sample_executor(
        std::unique_ptr<CudaTEdgeExecutor> &executor,
        const Fixture &fixture,
        std::vector<std::uint32_t> &access_slots,
        Sample &sample)
    {
        executor = make_executor(fixture);
        materialize_resident_sources(*executor, fixture);

        access_slots.resize(fixture.sources.size() + 1);
        std::iota(access_slots.begin(), access_slots.end(), 0U);

        sample.rss_before = resident_bytes();
        sample.vram_free_before = free_vram_bytes();
        sample.stats_before = gatzk::algebra::cuda_execution_stats_snapshot();
        sample.kernel_count_before =
            gatzk::algebra::cuda_kernel_invocation_count();
    }

    void finish_sample(
        Sample &sample,
        const std::vector<FieldElement> &actual,
        const Fixture &fixture,
        const char *label,
        Clock::time_point total_start,
        Clock::time_point materialization_start,
        Clock::time_point materialization_end,
        Clock::time_point upload_end,
        Clock::time_point execute_end)
    {
        const auto expected = expected_term_output(fixture);
        require_equal(actual, expected, label);
        const auto digest = digest_values(actual);
        if (digest != digest_values(expected))
        {
            fail(std::string(label) + " digest mismatch");
        }

        sample.host_materialization_ms = 0.0;
        sample.device_materialization_ms =
            std::chrono::duration<double, std::milli>(
                materialization_end - materialization_start)
                .count();
        sample.upload_sync_ms =
            std::chrono::duration<double, std::milli>(
                upload_end - materialization_end)
                .count();
        sample.execute_readback_ms =
            std::chrono::duration<double, std::milli>(
                execute_end - upload_end)
                .count();
        sample.total_ms =
            std::chrono::duration<double, std::milli>(
                execute_end - total_start)
                .count();
        sample.digest = digest;
        sample.rss_after = resident_bytes();
        sample.vram_free_after = free_vram_bytes();
        sample.stats_after = gatzk::algebra::cuda_execution_stats_snapshot();
        sample.kernel_count_after =
            gatzk::algebra::cuda_kernel_invocation_count();
    }

    Sample run_legacy_sample(const Fixture &fixture)
    {
        Sample sample;
        std::unique_ptr<CudaTEdgeExecutor> executor;
        std::vector<std::uint32_t> access_slots;
        prepare_sample_executor(
            executor,
            fixture,
            access_slots,
            sample);

        const auto total_start = Clock::now();
        const auto host_start = total_start;
        const auto correction = blinder_difference(
            fixture.target_blinder,
            fixture.sources,
            fixture.source_scales);
        const auto target_host = make_scaled_target_host(fixture, correction);
        require_equal(
            target_host,
            fixture.target_coset,
            "CPU scaled t_edge linear materialization");
        const auto host_end = Clock::now();

        const auto upload_start = host_end;
        executor->upload_leaf_chunk(
            fixture.sources.size(),
            0,
            FieldElement::native_array_data(target_host.data()),
            target_host.size());
        check_cuda(cudaDeviceSynchronize(), "baseline upload synchronization");
        const auto upload_end = Clock::now();

        const auto execute_start = upload_end;
        executor->accumulate_term(0, access_slots);
        executor->accumulate_term(1, access_slots);
        std::vector<FieldElement> actual(fixture.target_size, FieldElement::zero());
        executor->finalize(actual.data(), actual.size());
        const auto execute_end = Clock::now();
        finish_sample(
            sample,
            actual,
            fixture,
            "CUDA linear legacy output",
            total_start,
            host_start,
            host_end,
            upload_end,
            execute_end);
        sample.host_materialization_ms =
            std::chrono::duration<double, std::milli>(
                host_end - host_start)
                .count();
        sample.device_materialization_ms = 0.0;
        sample.upload_sync_ms =
            std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
        return sample;
    }

    Sample run_resident_sample(const Fixture &fixture)
    {
        Sample sample;
        std::unique_ptr<CudaTEdgeExecutor> executor;
        std::vector<std::uint32_t> access_slots;
        prepare_sample_executor(
            executor,
            fixture,
            access_slots,
            sample);

        const auto total_start = Clock::now();
        const auto materialization_start = total_start;

        std::vector<gatzk::protocol::CudaTEdgeLinearSource> sources;
        sources.reserve(fixture.sources.size());
        for (std::size_t index = 0;
             index < fixture.sources.size();
             ++index)
        {
            const auto &blinder = fixture.sources[index].blinder;
            sources.push_back(
                gatzk::protocol::CudaTEdgeLinearSource{
                    .slot = static_cast<std::uint32_t>(index),
                    .scale = fixture.source_scales[index],
                    .zero_domain_blinder =
                        FieldElement::native_array_data(blinder.data()),
                    .blinder_count = blinder.size(),
                });
        }

        if (!executor->materialize_linear_leaf(
                fixture.sources.size(),
                sources,
                FieldElement::native_array_data(
                    fixture.target_blinder.data()),
                fixture.target_blinder.size(),
                gatzk::protocol::CudaTEdgeLeafLayout::Split))
        {
            fail("CUDA resident linear materialization declined");
        }
        const auto materialization_end = Clock::now();

        const auto execute_start = materialization_end;
        executor->accumulate_term(0, access_slots);
        executor->accumulate_term(1, access_slots);
        std::vector<FieldElement> actual(
            fixture.target_size,
            FieldElement::zero());
        executor->finalize(actual.data(), actual.size());
        const auto execute_end = Clock::now();

        finish_sample(
            sample,
            actual,
            fixture,
            "CUDA resident linear output",
            total_start,
            materialization_start,
            materialization_end,
            materialization_end,
            execute_end);
        sample.execute_readback_ms =
            std::chrono::duration<double, std::milli>(
                execute_end - execute_start)
                .count();
        return sample;
    }

    double median(std::vector<double> values)
    {
        std::sort(values.begin(), values.end());
        if (values.empty())
        {
            return 0.0;
        }
        const auto middle = values.size() / 2;
        if ((values.size() & 1U) != 0U)
        {
            return values[middle];
        }
        return (values[middle - 1] + values[middle]) / 2.0;
    }

    double minimum(const std::vector<double> &values)
    {
        return *std::min_element(values.begin(), values.end());
    }

    void print_case(
        std::size_t domain_size,
        std::size_t target_size,
        std::size_t source_count,
        const std::vector<Sample> &legacy_samples,
        const std::vector<Sample> &resident_samples)
    {
        if (legacy_samples.empty() ||
            legacy_samples.size() != resident_samples.size())
        {
            fail("CUDA t_edge linear A/B repeat mismatch");
        }

        auto collect = [](const std::vector<Sample> &samples,
                          auto selector)
        {
            std::vector<double> result;
            result.reserve(samples.size());
            for (const auto &sample : samples)
            {
                result.push_back(selector(sample));
            }
            return result;
        };

        const auto legacy_host = collect(
            legacy_samples,
            [](const Sample &sample) { return sample.host_materialization_ms; });
        const auto legacy_upload = collect(
            legacy_samples,
            [](const Sample &sample) { return sample.upload_sync_ms; });
        const auto legacy_execute = collect(
            legacy_samples,
            [](const Sample &sample) { return sample.execute_readback_ms; });
        const auto legacy_total = collect(
            legacy_samples,
            [](const Sample &sample) { return sample.total_ms; });
        const auto resident_materialization = collect(
            resident_samples,
            [](const Sample &sample) {
                return sample.device_materialization_ms;
            });
        const auto resident_execute = collect(
            resident_samples,
            [](const Sample &sample) { return sample.execute_readback_ms; });
        const auto resident_total = collect(
            resident_samples,
            [](const Sample &sample) { return sample.total_ms; });

        const auto &legacy_first = legacy_samples.front();
        const auto &resident_first = resident_samples.front();
        const auto digest = legacy_first.digest;
        for (std::size_t index = 0;
             index < legacy_samples.size();
             ++index)
        {
            if (legacy_samples[index].digest != digest ||
                resident_samples[index].digest != digest)
            {
                fail("CUDA t_edge linear A/B digest mismatch");
            }
        }

        auto delta = [](std::uint64_t after, std::uint64_t before) {
            return after - before;
        };
        const auto legacy_h2d_bytes = delta(
            legacy_first.stats_after.h2d_bytes,
            legacy_first.stats_before.h2d_bytes);
        const auto resident_h2d_bytes = delta(
            resident_first.stats_after.h2d_bytes,
            resident_first.stats_before.h2d_bytes);
        const auto legacy_d2h_bytes = delta(
            legacy_first.stats_after.d2h_bytes,
            legacy_first.stats_before.d2h_bytes);
        const auto resident_d2h_bytes = delta(
            resident_first.stats_after.d2h_bytes,
            resident_first.stats_before.d2h_bytes);
        const auto legacy_syncs = delta(
            legacy_first.stats_after.stream_synchronizations,
            legacy_first.stats_before.stream_synchronizations);
        const auto resident_syncs = delta(
            resident_first.stats_after.stream_synchronizations,
            resident_first.stats_before.stream_synchronizations);
        const auto legacy_kernels =
            legacy_first.kernel_count_after -
            legacy_first.kernel_count_before;
        const auto resident_kernels =
            resident_first.kernel_count_after -
            resident_first.kernel_count_before;

        const auto legacy_median = median(legacy_total);
        const auto resident_median = median(resident_total);
        const auto speedup =
            resident_median == 0.0
                ? 0.0
                : legacy_median / resident_median;

        std::cout << std::fixed << std::setprecision(3)
                  << "CUDA_T_EDGE_LINEAR_AB"
                  << " domain=" << domain_size
                  << " target=" << target_size
                  << " source_leaves=" << source_count
                  << " repeats=" << legacy_samples.size()
                  << " legacy_host_materialization_ms_median="
                  << median(legacy_host)
                  << " legacy_host_materialization_ms_min="
                  << minimum(legacy_host)
                  << " legacy_upload_sync_ms_median="
                  << median(legacy_upload)
                  << " legacy_upload_sync_ms_min="
                  << minimum(legacy_upload)
                  << " legacy_execute_readback_ms_median="
                  << median(legacy_execute)
                  << " legacy_execute_readback_ms_min="
                  << minimum(legacy_execute)
                  << " legacy_total_ms_median=" << legacy_median
                  << " legacy_total_ms_min=" << minimum(legacy_total)
                  << " resident_materialization_ms_median="
                  << median(resident_materialization)
                  << " resident_materialization_ms_min="
                  << minimum(resident_materialization)
                  << " resident_execute_readback_ms_median="
                  << median(resident_execute)
                  << " resident_execute_readback_ms_min="
                  << minimum(resident_execute)
                  << " resident_total_ms_median=" << resident_median
                  << " resident_total_ms_min=" << minimum(resident_total)
                  << " total_speedup=" << speedup
                  << " digest=0x" << std::hex << digest << std::dec
                  << " legacy_h2d_bytes=" << legacy_h2d_bytes
                  << " resident_h2d_bytes=" << resident_h2d_bytes
                  << " legacy_d2h_bytes=" << legacy_d2h_bytes
                  << " resident_d2h_bytes=" << resident_d2h_bytes
                  << " legacy_stream_syncs=" << legacy_syncs
                  << " resident_stream_syncs=" << resident_syncs
                  << " legacy_kernel_calls=" << legacy_kernels
                  << " resident_kernel_calls=" << resident_kernels
                  << " legacy_rss_before_bytes=" << legacy_first.rss_before
                  << " resident_rss_before_bytes=" << resident_first.rss_before
                  << " legacy_vram_free_before_bytes="
                  << legacy_first.vram_free_before
                  << " resident_vram_free_before_bytes="
                  << resident_first.vram_free_before
                  << '\n';
    }
}

int main(int argc, char **argv)
{
    try
    {
        if (!gatzk::protocol::cuda_t_edge_executor_available())
        {
            std::cout << "SKIP cuda_t_edge linear baseline unavailable\n";
            return 77;
        }

        std::size_t repeats = 5;
        if (argc > 1)
        {
            repeats = std::stoull(argv[1]);
        }
        if (repeats < 5 || repeats > 20)
        {
            fail("repeats must be in [5, 20]");
        }

        std::cout << "CUDA_T_EDGE_LINEAR_AB_ENV "
                  << gatzk::algebra::cuda_device_summary()
                  << " vram_free_bytes=" << free_vram_bytes()
                  << " rss_bytes=" << resident_bytes()
                  << '\n';

        for (const auto target_size : {std::size_t{1} << 16, std::size_t{1} << 18})
        {
            const auto domain_size = target_size / 4;
            for (const auto source_count : {std::size_t{2}, std::size_t{4}, std::size_t{6}})
            {
                const auto fixture = make_fixture(
                    domain_size,
                    target_size,
                    source_count);
                std::vector<Sample> legacy_samples;
                std::vector<Sample> resident_samples;
                legacy_samples.reserve(repeats);
                resident_samples.reserve(repeats);
                for (std::size_t repeat = 0; repeat < repeats; ++repeat)
                {
                    legacy_samples.push_back(run_legacy_sample(fixture));
                    resident_samples.push_back(run_resident_sample(fixture));
                }
                print_case(
                    domain_size,
                    target_size,
                    source_count,
                    legacy_samples,
                    resident_samples);
            }
        }

        std::cout << "CUDA_T_EDGE_LINEAR_AB_OK\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "CUDA_T_EDGE_LINEAR_AB_FAIL " << error.what() << '\n';
        return 1;
    }
}
