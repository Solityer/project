#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/cuda_msm.hpp"
#include "gatzk/crypto/curve.hpp"
#include "gatzk/util/route2.hpp"

#include "curves/params/bls12_381.cuh"
#include "msm/msm.cuh"

extern "C" cudaError_t bls12_381_msm_cuda(
    const bls12_381::scalar_t *,
    const bls12_381::affine_t *,
    int,
    msm::MSMConfig &,
    bls12_381::projective_t *);

namespace
{
    using gatzk::algebra::FieldElement;
    using gatzk::crypto::G1Point;

    constexpr std::size_t kBasisCount = 16;
    constexpr std::size_t kDefaultRepeats = 2;
    constexpr std::size_t kDefaultWarmups = 1;

    static_assert(
        sizeof(bls12_381::scalar_t::limbs_storage.limbs) <= 64,
        "ICICLE scalar representation no longer fits production conversion");
    static_assert(
        sizeof(bls12_381::point_field_t::limbs_storage.limbs) <= 64,
        "ICICLE point representation no longer fits production conversion");
    static_assert(
        sizeof(G1Point) == sizeof(mcl::G1),
        "G1Point must wrap exactly one mcl::G1");

    struct MsmCase
    {
        std::size_t count;
        std::size_t batch_size;
    };

    struct Options
    {
        std::vector<MsmCase> cases{{8193, 1}, {8193, 2}};
        std::size_t repeats = kDefaultRepeats;
        std::size_t warmups = kDefaultWarmups;
    };

    [[noreturn]] void usage_error(const std::string &message)
    {
        throw std::runtime_error(
            message +
            "\nusage: gatzk_cuda_msm_big_triangle_bench "
            "[--case COUNT BATCH] [--repeats N] [--warmups N]");
    }

    std::size_t parse_size(const char *value, const char *label)
    {
        if (value == nullptr || *value == '\0' || *value == '-')
        {
            usage_error(std::string("invalid ") + label);
        }
        std::size_t consumed = 0;
        unsigned long long parsed = 0;
        try
        {
            parsed = std::stoull(value, &consumed, 0);
        }
        catch (const std::exception &)
        {
            usage_error(std::string("invalid ") + label + ": " + value);
        }
        if (consumed != std::strlen(value) || parsed == 0)
        {
            usage_error(std::string("invalid ") + label + ": " + value);
        }
        return static_cast<std::size_t>(parsed);
    }

    Options parse_options(int argc, char **argv)
    {
        Options options;
        bool selected_cases = false;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument(argv[index]);
            if (argument == "--case")
            {
                if (index + 2 >= argc)
                {
                    usage_error("--case requires COUNT and BATCH");
                }
                if (!selected_cases)
                {
                    options.cases.clear();
                    selected_cases = true;
                }
                options.cases.push_back({
                    parse_size(argv[++index], "case count"),
                    parse_size(argv[++index], "case batch"),
                });
            }
            else if (argument == "--repeats")
            {
                if (++index >= argc)
                {
                    usage_error("--repeats requires N");
                }
                options.repeats = parse_size(argv[index], "repeats");
            }
            else if (argument == "--warmups")
            {
                if (++index >= argc)
                {
                    usage_error("--warmups requires N");
                }
                options.warmups = parse_size(argv[index], "warmups");
            }
            else if (argument == "--help" || argument == "-h")
            {
                std::cout
                    << "usage: gatzk_cuda_msm_big_triangle_bench "
                       "[--case COUNT BATCH] [--repeats N] [--warmups N]\n";
                std::exit(0);
            }
            else
            {
                usage_error("unknown option: " + argument);
            }
        }
        if (options.cases.empty() || options.repeats == 0)
        {
            usage_error("at least one case and one repeat are required");
        }
        return options;
    }

    void check_cuda(cudaError_t status, const char *operation)
    {
        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string(operation) + " failed: " +
                cudaGetErrorString(status));
        }
    }

    struct ScopedEvent
    {
        ScopedEvent()
        {
            check_cuda(cudaEventCreate(&event), "cudaEventCreate");
        }

        ~ScopedEvent()
        {
            if (event != nullptr)
            {
                (void)cudaEventDestroy(event);
            }
        }

        ScopedEvent(const ScopedEvent &) = delete;
        ScopedEvent &operator=(const ScopedEvent &) = delete;

        cudaEvent_t event = nullptr;
    };

    struct ScopedDeviceBuffer
    {
        explicit ScopedDeviceBuffer(std::size_t bytes)
        {
            if (bytes != 0)
            {
                check_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
            }
        }

        ~ScopedDeviceBuffer()
        {
            if (pointer != nullptr)
            {
                (void)cudaFree(pointer);
            }
        }

        ScopedDeviceBuffer(const ScopedDeviceBuffer &) = delete;
        ScopedDeviceBuffer &operator=(const ScopedDeviceBuffer &) = delete;

        ScopedDeviceBuffer(ScopedDeviceBuffer &&other) noexcept
            : pointer(other.pointer)
        {
            other.pointer = nullptr;
        }

        ScopedDeviceBuffer &operator=(ScopedDeviceBuffer &&other) noexcept
        {
            if (this != &other)
            {
                if (pointer != nullptr)
                {
                    (void)cudaFree(pointer);
                }
                pointer = other.pointer;
                other.pointer = nullptr;
            }
            return *this;
        }

        template <class T>
        T *as()
        {
            return reinterpret_cast<T *>(pointer);
        }

        template <class T>
        const T *as() const
        {
            return reinterpret_cast<const T *>(pointer);
        }

        void *pointer = nullptr;
    };

    struct MemorySnapshot
    {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        long max_rss_kib = 0;
    };

    MemorySnapshot memory_snapshot()
    {
        MemorySnapshot result;
        check_cuda(
            cudaMemGetInfo(&result.free_bytes, &result.total_bytes),
            "cudaMemGetInfo");
        struct rusage usage
        {
        };
        if (getrusage(RUSAGE_SELF, &usage) == 0)
        {
            result.max_rss_kib = usage.ru_maxrss;
        }
        return result;
    }

    std::string case_name(const MsmCase &test_case)
    {
        return std::to_string(test_case.count) + "x" +
            std::to_string(test_case.batch_size);
    }

    std::array<G1Point, kBasisCount> build_basis_cycle()
    {
        gatzk::algebra::ensure_mcl_field_ready();
        std::array<G1Point, kBasisCount> bases;
        G1Point generator;
        generator.value = mcl::bn::getG1basePoint();
        bases[0] = generator;
        for (std::size_t index = 1; index < bases.size(); ++index)
        {
            bases[index] = gatzk::crypto::g1_add(
                bases[index - 1],
                generator);
        }
        return bases;
    }

    struct Workload
    {
        MsmCase test_case;
        std::vector<G1Point> points;
        std::vector<std::vector<FieldElement>> scalar_vectors;
        std::vector<G1Point> expected;
    };

    Workload build_workload(const MsmCase &test_case)
    {
        if (test_case.count == 0 ||
            test_case.count >
                static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            test_case.batch_size == 0 ||
            test_case.batch_size >
                static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::runtime_error("MSM case exceeds ICICLE API limits");
        }
        const auto bases = build_basis_cycle();
        Workload workload{
            test_case,
            std::vector<G1Point>(test_case.count),
            std::vector<std::vector<FieldElement>>(
                test_case.batch_size,
                std::vector<FieldElement>(
                    test_case.count,
                    FieldElement::zero())),
            std::vector<G1Point>(
                test_case.batch_size,
                gatzk::crypto::g1_zero()),
        };
        std::vector<std::array<FieldElement, kBasisCount>> grouped(
            test_case.batch_size);
        for (std::size_t index = 0; index < test_case.count; ++index)
        {
            const auto basis_index = index % kBasisCount;
            workload.points[index] = bases[basis_index];
            for (std::size_t batch = 0;
                 batch < test_case.batch_size;
                 ++batch)
            {
                const auto raw =
                    static_cast<std::uint64_t>(
                        (index % 104729U) + 1U) *
                        static_cast<std::uint64_t>(batch + 3U) +
                    static_cast<std::uint64_t>(17U + basis_index);
                const FieldElement scalar(raw);
                workload.scalar_vectors[batch][index] = scalar;
                grouped[batch][basis_index] += scalar;
            }
        }
        for (std::size_t batch = 0;
             batch < test_case.batch_size;
             ++batch)
        {
            for (std::size_t basis_index = 0;
                 basis_index < kBasisCount;
                 ++basis_index)
            {
                workload.expected[batch] = gatzk::crypto::g1_add(
                    workload.expected[batch],
                    gatzk::crypto::g1_mul(
                        bases[basis_index],
                        grouped[batch][basis_index]));
            }
        }
        return workload;
    }

    template <class IcicleField, class MclField>
    IcicleField to_icicle_field(const MclField &input)
    {
        std::array<std::uint8_t, 64> bytes{};
        if (input.getLittleEndian(bytes.data(), bytes.size()) == 0)
        {
            throw std::runtime_error(
                "canonical getLittleEndian conversion failed");
        }
        IcicleField output{};
        constexpr auto byte_count =
            sizeof(output.limbs_storage.limbs);
        static_assert(byte_count <= 64);
        std::memcpy(
            output.limbs_storage.limbs,
            bytes.data(),
            byte_count);
        return output;
    }

    template <class MclField, class IcicleField>
    MclField to_mcl_field(const IcicleField &input)
    {
        MclField output;
        output.setLittleEndianMod(
            reinterpret_cast<const std::uint8_t *>(
                input.limbs_storage.limbs),
            sizeof(input.limbs_storage.limbs));
        return output;
    }

    void convert_points(
        bls12_381::affine_t *output,
        const std::vector<G1Point> &points)
    {
        std::vector<mcl::G1> normalized(points.size());
        auto *native_points = const_cast<mcl::G1 *>(
            reinterpret_cast<const mcl::G1 *>(points.data()));
        mcl::G1::normalizeVec(
            normalized.data(),
            native_points,
            points.size());
        for (std::size_t index = 0; index < points.size(); ++index)
        {
            if (normalized[index].isZero())
            {
                output[index] = bls12_381::affine_t::zero();
            }
            else
            {
                output[index] = {
                    to_icicle_field<bls12_381::point_field_t>(
                        normalized[index].x),
                    to_icicle_field<bls12_381::point_field_t>(
                        normalized[index].y),
                };
            }
        }
    }

    G1Point convert_result(
        const bls12_381::projective_t &input,
        const MsmCase &test_case,
        bool is_big_triangle)
    {
        G1Point output;
        if (bls12_381::projective_t::is_zero(input))
        {
            output.value.clear();
            return output;
        }
        if (!bls12_381::projective_t::is_on_curve(input))
        {
            throw std::runtime_error(
                "ICICLE result is off curve for case " +
                case_name(test_case));
        }
        const auto affine = bls12_381::projective_t::to_affine(input);
        if (affine == bls12_381::affine_t::zero())
        {
            output.value.clear();
            return output;
        }
        output.value.x = to_mcl_field<mcl::Fp>(affine.x);
        output.value.y = to_mcl_field<mcl::Fp>(affine.y);
        output.value.z = 1;
        const bool mcl_valid = output.value.isValid();
        const bool subgroup_valid =
            mcl_valid && mcl::bn::isValidOrderBLS12(output.value);
        if (!mcl_valid || !subgroup_valid)
        {
            throw std::runtime_error(
                "ICICLE result failed MCL curve/subgroup validation for case " +
                case_name(test_case) + " flag=" +
                (is_big_triangle ? "true" : "false"));
        }
        return output;
    }

    struct ResidentWorkload
    {
        explicit ResidentWorkload(const Workload &workload)
            : device_points(workload.test_case.count * sizeof(bls12_381::affine_t)),
              host_points(workload.test_case.count),
              host_scalars(
                  workload.test_case.count * workload.test_case.batch_size)
        {
        }

        ScopedDeviceBuffer device_points;
        std::vector<bls12_381::affine_t> host_points;
        std::vector<bls12_381::scalar_t> host_scalars;
        double point_conversion_ms = 0.0;
        double scalar_conversion_ms = 0.0;
        double point_upload_ms = 0.0;
    };

    template <class Function>
    double wall_ms(Function &&function)
    {
        const auto start = std::chrono::steady_clock::now();
        function();
        const auto stop = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(stop - start).count();
    }

    ResidentWorkload prepare_resident(
        const Workload &workload)
    {
        ResidentWorkload resident(workload);
        resident.point_conversion_ms = wall_ms([&]() {
            convert_points(resident.host_points.data(), workload.points);
        });
        resident.point_upload_ms = wall_ms([&]() {
            check_cuda(
                cudaMemcpy(
                    resident.device_points.pointer,
                    resident.host_points.data(),
                    resident.host_points.size() *
                        sizeof(resident.host_points[0]),
                    cudaMemcpyHostToDevice),
                "point upload");
        });
        resident.scalar_conversion_ms = wall_ms([&]() {
            for (std::size_t batch = 0;
                 batch < workload.test_case.batch_size;
                 ++batch)
            {
                for (std::size_t index = 0;
                     index < workload.test_case.count;
                     ++index)
                {
                    resident.host_scalars[
                        batch * workload.test_case.count + index] =
                        to_icicle_field<bls12_381::scalar_t>(
                            workload.scalar_vectors[batch][index].native());
                }
            }
        });
        return resident;
    }

    msm::MSMConfig make_config(
        const MsmCase &test_case,
        bool is_big_triangle)
    {
        auto config = msm::default_msm_config();
        config.points_size = static_cast<int>(test_case.count);
        config.precompute_factor = 1;
        config.c = 0;
        config.bitsize = 0;
        config.large_bucket_factor = 10;
        config.batch_size = static_cast<int>(test_case.batch_size);
        config.are_scalars_on_device = false;
        config.are_scalars_montgomery_form = false;
        config.are_points_on_device = true;
        config.are_points_montgomery_form = false;
        config.are_results_on_device = false;
        config.is_big_triangle = is_big_triangle;
        config.is_async = false;
        return config;
    }

    bool differ_only_in_flag(
        const msm::MSMConfig &first,
        const msm::MSMConfig &second)
    {
        return first.ctx.stream == second.ctx.stream &&
            first.ctx.device_id == second.ctx.device_id &&
            first.ctx.mempool == second.ctx.mempool &&
            first.points_size == second.points_size &&
            first.precompute_factor == second.precompute_factor &&
            first.c == second.c &&
            first.bitsize == second.bitsize &&
            first.large_bucket_factor == second.large_bucket_factor &&
            first.batch_size == second.batch_size &&
            first.are_scalars_on_device == second.are_scalars_on_device &&
            first.are_scalars_montgomery_form ==
                second.are_scalars_montgomery_form &&
            first.are_points_on_device == second.are_points_on_device &&
            first.are_points_montgomery_form ==
                second.are_points_montgomery_form &&
            first.are_results_on_device == second.are_results_on_device &&
            first.is_big_triangle != second.is_big_triangle &&
            first.is_async == second.is_async;
    }

    struct Measurement
    {
        bool is_big_triangle = false;
        std::size_t repeat = 0;
        std::size_t sequence = 0;
        double wall_ms = 0.0;
        double event_ms = 0.0;
        MemorySnapshot before;
        MemorySnapshot after;
        std::size_t scalar_transfer_bytes = 0;
        std::size_t gatzk_kernel_delta = 0;
    };

    Measurement run_direct(
        const Workload &workload,
        const ResidentWorkload &resident,
        bool is_big_triangle,
        std::size_t repeat,
        std::size_t sequence)
    {
        Measurement measurement;
        measurement.is_big_triangle = is_big_triangle;
        measurement.repeat = repeat;
        measurement.sequence = sequence;
        measurement.before = memory_snapshot();
        const auto counter_before =
            gatzk::algebra::cuda_kernel_invocation_count();

        auto config = make_config(workload.test_case, is_big_triangle);
        const auto other_config = make_config(
            workload.test_case,
            !is_big_triangle);
        if (!differ_only_in_flag(config, other_config))
        {
            throw std::runtime_error(
                "benchmark configs differ beyond is_big_triangle");
        }

        std::vector<bls12_381::projective_t> results(
            workload.test_case.batch_size);
        ScopedEvent start_event;
        ScopedEvent stop_event;
        const auto start = std::chrono::steady_clock::now();
        check_cuda(
            cudaEventRecord(start_event.event, config.ctx.stream),
            "MSM start event");
        check_cuda(
            bls12_381_msm_cuda(
                resident.host_scalars.data(),
                resident.device_points.as<bls12_381::affine_t>(),
                static_cast<int>(workload.test_case.count),
                config,
                results.data()),
            "direct ICICLE MSM");
        check_cuda(
            cudaEventRecord(stop_event.event, config.ctx.stream),
            "MSM stop event");
        check_cuda(
            cudaEventSynchronize(stop_event.event),
            "MSM stop synchronize");
        const auto stop = std::chrono::steady_clock::now();
        float event_ms = 0.0F;
        check_cuda(
            cudaEventElapsedTime(
                &event_ms,
                start_event.event,
                stop_event.event),
            "MSM event elapsed time");
        measurement.wall_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
        measurement.event_ms = event_ms;
        measurement.scalar_transfer_bytes =
            workload.test_case.count *
            workload.test_case.batch_size *
            sizeof(bls12_381::scalar_t);
        measurement.after = memory_snapshot();
        measurement.gatzk_kernel_delta =
            gatzk::algebra::cuda_kernel_invocation_count() -
            counter_before;

        for (std::size_t batch = 0;
             batch < workload.test_case.batch_size;
             ++batch)
        {
            const auto actual = convert_result(
                results[batch],
                workload.test_case,
                is_big_triangle);
            if (!(actual == workload.expected[batch]))
            {
                throw std::runtime_error(
                    "direct ICICLE result differs from grouped MCL reference "
                    "for case " +
                    case_name(workload.test_case) + " flag=" +
                    (is_big_triangle ? "true" : "false") + " batch=" +
                    std::to_string(batch));
            }
        }
        return measurement;
    }

    struct WrapperMeasurement
    {
        double wall_ms = 0.0;
        std::size_t kernel_delta = 0;
        gatzk::algebra::CudaExecutionStats stats_delta{};
    };

    gatzk::algebra::CudaExecutionStats stats_delta(
        const gatzk::algebra::CudaExecutionStats &after,
        const gatzk::algebra::CudaExecutionStats &before)
    {
        return {
            after.h2d_bytes - before.h2d_bytes,
            after.d2h_bytes - before.d2h_bytes,
            after.h2d_operations - before.h2d_operations,
            after.d2h_operations - before.d2h_operations,
            after.stream_synchronizations - before.stream_synchronizations,
            after.ntt_calls - before.ntt_calls,
            after.intt_calls - before.intt_calls,
            after.quotient_calls - before.quotient_calls,
            after.t_edge_calls - before.t_edge_calls,
            after.opening_fold_calls - before.opening_fold_calls,
            after.msm_single_calls - before.msm_single_calls,
            after.msm_batch_calls - before.msm_batch_calls,
            after.resident_hits - before.resident_hits,
            after.resident_misses - before.resident_misses,
        };
    }

    WrapperMeasurement run_wrapper_false(
        const Workload &workload)
    {
        std::vector<G1Point> actual(
            workload.test_case.batch_size,
            gatzk::crypto::g1_zero());
        std::vector<G1Point *> outputs(workload.test_case.batch_size);
        std::vector<const FieldElement *> scalar_inputs(
            workload.test_case.batch_size);
        for (std::size_t batch = 0;
             batch < workload.test_case.batch_size;
             ++batch)
        {
            outputs[batch] = &actual[batch];
            scalar_inputs[batch] = workload.scalar_vectors[batch].data();
        }
        const auto before_stats =
            gatzk::algebra::cuda_execution_stats_snapshot();
        const auto before_counter =
            gatzk::algebra::cuda_kernel_invocation_count();
        const auto start = std::chrono::steady_clock::now();
        if (!gatzk::crypto::try_cuda_g1_msm_same_bases_batch(
                outputs.data(),
                workload.points.data(),
                scalar_inputs.data(),
                workload.test_case.count,
                workload.test_case.batch_size,
                workload.test_case.count))
        {
            throw std::runtime_error(
                "production CUDA MSM wrapper declined false comparison");
        }
        const auto stop = std::chrono::steady_clock::now();
        for (std::size_t batch = 0; batch < actual.size(); ++batch)
        {
            if (!(actual[batch] == workload.expected[batch]))
            {
                throw std::runtime_error(
                    "production CUDA MSM wrapper differs from grouped MCL "
                    "reference for case " +
                    case_name(workload.test_case));
            }
        }
        WrapperMeasurement measurement;
        measurement.wall_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
        measurement.kernel_delta =
            gatzk::algebra::cuda_kernel_invocation_count() - before_counter;
        measurement.stats_delta = stats_delta(
            gatzk::algebra::cuda_execution_stats_snapshot(),
            before_stats);
        return measurement;
    }

    void print_measurement(
        const Workload &workload,
        const ResidentWorkload &resident,
        const Measurement &measurement)
    {
        const auto vram_delta =
            static_cast<long long>(measurement.before.free_bytes) -
            static_cast<long long>(measurement.after.free_bytes);
        std::cout
            << "direct"
            << ",case=" << case_name(workload.test_case)
            << ",flag=" << (measurement.is_big_triangle ? "true" : "false")
            << ",repeat=" << measurement.repeat
            << ",sequence=" << measurement.sequence
            << ",wall_ms=" << std::fixed << std::setprecision(3)
            << measurement.wall_ms
            << ",icicle_event_ms=" << measurement.event_ms
            << ",point_conversion_ms=" << resident.point_conversion_ms
            << ",scalar_conversion_ms=" << resident.scalar_conversion_ms
            << ",point_upload_ms=" << resident.point_upload_ms
            << ",scalar_transfer_bytes=" << measurement.scalar_transfer_bytes
            << ",point_resident=true"
            << ",gatzk_kernel_counter_delta="
            << measurement.gatzk_kernel_delta
            << ",rss_before_kib=" << measurement.before.max_rss_kib
            << ",rss_after_kib=" << measurement.after.max_rss_kib
            << ",vram_free_before_bytes=" << measurement.before.free_bytes
            << ",vram_free_after_bytes=" << measurement.after.free_bytes
            << ",vram_used_delta_bytes=" << vram_delta
            << ",result=exact\n";
    }

    void print_wrapper(
        const Workload &workload,
        const WrapperMeasurement &measurement)
    {
        const auto &stats = measurement.stats_delta;
        std::cout
            << "wrapper_false"
            << ",case=" << case_name(workload.test_case)
            << ",wall_ms=" << std::fixed << std::setprecision(3)
            << measurement.wall_ms
            << ",gatzk_kernel_counter_delta=" << measurement.kernel_delta
            << ",h2d_bytes=" << stats.h2d_bytes
            << ",d2h_bytes=" << stats.d2h_bytes
            << ",h2d_operations=" << stats.h2d_operations
            << ",d2h_operations=" << stats.d2h_operations
            << ",stream_synchronizations=" << stats.stream_synchronizations
            << ",msm_single_calls=" << stats.msm_single_calls
            << ",msm_batch_calls=" << stats.msm_batch_calls
            << ",resident_hits=" << stats.resident_hits
            << ",resident_misses=" << stats.resident_misses
            << ",result=exact\n";
    }

    void run_case(
        const MsmCase &test_case,
        const Options &options,
        bool integrated_memory)
    {
        const auto workload = build_workload(test_case);
        auto resident = prepare_resident(workload);
        std::cout
            << "case_begin,count=" << test_case.count
            << ",batch=" << test_case.batch_size
            << ",repeats=" << options.repeats
            << ",warmups=" << options.warmups
            << ",integrated_memory="
            << (integrated_memory ? "true" : "false") << '\n';

        for (std::size_t warmup = 0;
             warmup < options.warmups;
             ++warmup)
        {
            (void)run_direct(
                workload,
                resident,
                false,
                warmup,
                warmup);
            (void)run_direct(
                workload,
                resident,
                true,
                warmup,
                warmup);
        }

        std::size_t sequence = 0;
        for (std::size_t repeat = 0;
             repeat < options.repeats;
             ++repeat)
        {
            const bool first = (repeat % 2) != 0;
            const bool second = !first;
            const auto first_measurement = run_direct(
                workload,
                resident,
                first,
                repeat,
                sequence++);
            print_measurement(workload, resident, first_measurement);
            const auto second_measurement = run_direct(
                workload,
                resident,
                second,
                repeat,
                sequence++);
            print_measurement(workload, resident, second_measurement);
        }

        print_wrapper(workload, run_wrapper_false(workload));
        check_cuda(cudaDeviceSynchronize(), "case cleanup synchronize");
        gatzk::crypto::release_cuda_g1_msm_resident_cache();
        std::cout
            << "case_end,count=" << test_case.count
            << ",batch=" << test_case.batch_size << '\n';
    }
}

int main(int argc, char **argv)
{
    try
    {
        if (!gatzk::algebra::cuda_backend_available())
        {
            return 77;
        }
        auto route = gatzk::util::route2_options();
        route.cuda_field_hotspots = true;
        gatzk::util::set_route2_options(route);
        int integrated = 0;
        check_cuda(
            cudaDeviceGetAttribute(
                &integrated,
                cudaDevAttrIntegrated,
                0),
            "integrated device query");
        const auto options = parse_options(argc, argv);
        for (const auto &test_case : options.cases)
        {
            run_case(test_case, options, integrated != 0);
        }
        std::cout << "cuda msm big triangle benchmark passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
