#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/algebra/cuda_ntt_device.hpp"
#include "gatzk/algebra/cuda_weighted_session.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include "gatzk/util/route2.hpp"

#if GATZK_ENABLE_CUDA_BACKEND
#include <cuda_runtime_api.h>
#endif

namespace gatzk::algebra
{
    // 本文件统一管理 algebra 层的执行路由，
    // 包括 CPU / CUDA 可用性探测、size-aware 策略与常见向量算子。
    // 它位于 proving / verifying 的公共热路径上，但任何路由切换都不能改变 formal 结果。
#if GATZK_ENABLE_CUDA_BACKEND
    bool cuda_backend_runtime_available();
    bool cuda_field_hotspots_available();
    void scaled_accumulate_cuda(
        uint64_t *target_limbs,
        const uint64_t *source_limbs,
        const uint64_t scale_limbs[4],
        std::size_t element_count);

    void scaled_accumulate_pair_cuda(
        uint64_t *first_target_limbs,
        uint64_t *second_target_limbs,
        const uint64_t *source_limbs,
        const uint64_t first_scale_limbs[4],
        const uint64_t second_scale_limbs[4],
        std::size_t element_count);

    bool cuda_intt_in_place_exact(
        mcl::Fr *values,
        std::size_t size);

    bool cuda_ntt_in_place_exact(
        mcl::Fr *values,
        std::size_t size);

    bool cuda_ntt_pair_in_place_exact(
        mcl::Fr *first,
        mcl::Fr *second,
        std::size_t size);
#endif

    namespace
    {
        std::atomic<std::size_t> &cuda_kernel_counter()
        {
            static std::atomic<std::size_t> counter{0};
            return counter;
        }

        struct CudaExecutionCounters
        {
            std::atomic<std::uint64_t> h2d_bytes{0};
            std::atomic<std::uint64_t> d2h_bytes{0};
            std::atomic<std::uint64_t> h2d_operations{0};
            std::atomic<std::uint64_t> d2h_operations{0};
            std::atomic<std::uint64_t> stream_synchronizations{0};
            std::atomic<std::uint64_t> ntt_calls{0};
            std::atomic<std::uint64_t> intt_calls{0};
            std::atomic<std::uint64_t> quotient_calls{0};
            std::atomic<std::uint64_t> t_edge_calls{0};
            std::atomic<std::uint64_t> opening_fold_calls{0};
            std::atomic<std::uint64_t> msm_single_calls{0};
            std::atomic<std::uint64_t> msm_batch_calls{0};
            std::atomic<std::uint64_t> resident_hits{0};
            std::atomic<std::uint64_t> resident_misses{0};
        };

        CudaExecutionCounters &
        cuda_execution_counters()
        {
            static CudaExecutionCounters counters;
            return counters;
        }

        bool cuda_field_route_enabled()
        {
            return util::route2_options().cuda_field_hotspots && cuda_field_hotspots_build_enabled() && cuda_field_hotspots_available();
        }

        bool cuda_trace_route_enabled()
        {
            return util::route2_options().cuda_trace_hotspots && cuda_backend_available();
        }

        unsigned __int128 multiply_u128(std::size_t lhs, std::size_t rhs)
        {
            return static_cast<unsigned __int128>(lhs) * static_cast<unsigned __int128>(rhs);
        }

        std::size_t bounded_worker_count(
            std::size_t max_worker_count)
        {
            const auto hardware_workers =
                std::max<std::size_t>(
                    1,
                    std::thread::hardware_concurrency());

            if (max_worker_count == 0)
            {
                return hardware_workers;
            }

            return std::max<std::size_t>(
                1,
                std::min<std::size_t>(
                    hardware_workers,
                    max_worker_count));
        }

        AlgebraBackend parse_backend_env()
        {
            const char *value = std::getenv("GATZK_ALGEBRA_BACKEND");
            if (value == nullptr)
            {
                // 默认正式后端仍是 CPU；GPU 只能经由 compute-backend/route2 显式启用。
                return AlgebraBackend::Cpu;
            }
            const std::string backend(value);

            if (backend == "cpu")
            {
                return AlgebraBackend::Cpu;
            }
            if (backend == "cuda")
            {
                throw std::runtime_error(
                    "GATZK_ALGEBRA_BACKEND=cuda is unsupported; use --compute-backend cuda_hotspots on a CUDA-enabled build");
            }
            throw std::runtime_error("unsupported GATZK_ALGEBRA_BACKEND value: " + backend);
        }

        // 串行计算两个域元素向量在区间[begin, end)上的点积（--parallel-fft on）
        FieldElement dot_product_range(
            const std::vector<FieldElement> &lhs,
            const std::vector<FieldElement> &rhs,
            std::size_t begin,
            std::size_t end)
        {
            mcl::Fr sum;
            sum.clear();
            for (std::size_t i = begin; i < end; ++i)
            {
                mcl::Fr term;
                mcl::Fr::mul(term, lhs[i].native(), rhs[i].native());
                mcl::Fr::add(sum, sum, term);
            }
            return FieldElement::from_native(sum);
        }

        // 串行计算域元素向量与mcl::Fr向量在区间[begin, end)上的点积
        FieldElement dot_product_native_range(
            const std::vector<FieldElement> &lhs,
            const std::vector<mcl::Fr> &rhs,
            std::size_t begin,
            std::size_t end)
        {
            mcl::Fr sum;
            sum.clear();
            for (std::size_t i = begin; i < end; ++i)
            {
                mcl::Fr term;
                mcl::Fr::mul(term, lhs[i].native(), rhs[i]);
                mcl::Fr::add(sum, sum, term);
            }
            return FieldElement::from_native(sum);
        }

    }

    // 返回当前进程生效的代数后端选择。
    AlgebraBackend configured_algebra_backend()
    {
        static const auto backend = parse_backend_env();
        return backend;
    }

    // 当前正式主线只暴露 `"cpu"` 名称；CUDA 只是其中的热点执行策略。
    std::string configured_algebra_backend_name()
    {
        (void)configured_algebra_backend();
        return "cpu";
    }

    std::shared_ptr<const mcl::Ntt<mcl::Fr>> ntt_plan(std::size_t size)
    {
        static std::mutex mutex;
        static std::unordered_map<std::size_t, std::shared_ptr<const mcl::Ntt<mcl::Fr>>> cache;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (const auto it = cache.find(size); it != cache.end())
            {
                return it->second;
            }
        }

        auto plan = std::make_shared<mcl::Ntt<mcl::Fr>>();
        const mcl::Fr generator = 5;
        const mcl::Vint root_exponent = (mcl::Fr::getOp().mp - 1) >> 32;
        mcl::Fr::pow(plan->root_, generator, root_exponent);
        if (!plan->init(size))
        {
            throw std::runtime_error("failed to initialize NTT plan for size " + std::to_string(size));
        }
        if (!(FieldElement::from_native(plan->getW()) == FieldElement::root_of_unity(size)))
        {
            throw std::runtime_error("NTT root does not match protocol domain root for size " + std::to_string(size));
        }
        std::lock_guard<std::mutex> lock(mutex);
        const auto [it, inserted] = cache.emplace(size, plan);
        (void)inserted;
        return it->second;
    }

    void geometric_powers_native(
        mcl::Fr *output,
        std::size_t size,
        const FieldElement &base,
        std::size_t max_worker_count)
    {
        if (size == 0)
        {
            return;
        }
        if (output == nullptr)
        {
            throw std::runtime_error(
                "geometric powers require a non-null output");
        }
        constexpr std::size_t kMinRowsPerTask = 1ULL << 12;
        const auto available_workers =
            bounded_worker_count(
                max_worker_count);
        const auto task_count = size < kMinRowsPerTask
                                    ? std::size_t{1}
                                    : std::min<std::size_t>(
                                          available_workers,
                                          std::max<std::size_t>(2, size / kMinRowsPerTask));
        const auto chunk_size = (size + task_count - 1) / task_count;
        const auto base_native = base.native();
        std::vector<std::future<void>> futures;
        futures.reserve(task_count);
        for (std::size_t task = 0; task < task_count; ++task)
        {
            const auto begin = task * chunk_size;
            const auto end = std::min(size, begin + chunk_size);
            if (begin >= end)
            {
                break;
            }
            futures.push_back(std::async(
                std::launch::async,
                [=]
                {
                    auto power = base.pow(
                                         static_cast<std::uint64_t>(begin))
                                     .native();
                    for (std::size_t index = begin; index < end; ++index)
                    {
                        output[index] = power;
                        mcl::Fr::mul(power, power, base_native);
                    }
                }));
        }
        for (auto &future : futures)
        {
            future.get();
        }
    }

    void scale_by_geometric_powers_native(
        mcl::Fr *values,
        std::size_t size,
        const FieldElement &base,
        std::size_t max_worker_count)
    {
        if (size == 0)
        {
            return;
        }

        if (values == nullptr)
        {
            throw std::runtime_error(
                "geometric in-place scaling requires a non-null buffer");
        }

        constexpr std::size_t kMinRowsPerWorker =
            std::size_t{1} << 18;

        const auto available_workers =
            bounded_worker_count(
                max_worker_count);

        const auto worker_count =
            std::min<std::size_t>({
                8,
                available_workers,
                std::max<std::size_t>(
                    1,
                    (size + kMinRowsPerWorker - 1) /
                        kMinRowsPerWorker),
            });

        const auto base_native =
            base.native();

        const auto scale_range =
            [&](std::size_t begin,
                std::size_t end)
        {
            auto power =
                base.pow(
                        static_cast<std::uint64_t>(
                            begin))
                    .native();

            for (std::size_t index = begin;
                 index < end;
                 ++index)
            {
                mcl::Fr::mul(
                    values[index],
                    values[index],
                    power);

                mcl::Fr::mul(
                    power,
                    power,
                    base_native);
            }
        };

        if (worker_count <= 1)
        {
            scale_range(0, size);
            return;
        }

        const auto chunk_size =
            (size + worker_count - 1) /
            worker_count;

        std::vector<std::future<void>>
            futures;

        futures.reserve(worker_count);

        for (std::size_t worker = 0;
             worker < worker_count;
             ++worker)
        {
            const auto begin =
                worker * chunk_size;

            const auto end =
                std::min(
                    size,
                    begin + chunk_size);

            if (begin >= end)
            {
                break;
            }

            futures.push_back(
                std::async(
                    std::launch::async,
                    [&, begin, end]()
                    {
                        scale_range(
                            begin,
                            end);
                    }));
        }

        for (auto &future : futures)
        {
            future.get();
        }
    }

    void copy_scaled_by_geometric_powers_native(
        const mcl::Fr *input,
        mcl::Fr *output,
        std::size_t size,
        const FieldElement &base,
        std::size_t max_worker_count)
    {
        if (size == 0)
        {
            return;
        }

        if (input == nullptr ||
            output == nullptr)
        {
            throw std::runtime_error(
                "geometric copy scaling requires non-null buffers");
        }

        constexpr std::size_t kMinRowsPerWorker =
            std::size_t{1} << 18;

        const auto available_workers =
            bounded_worker_count(
                max_worker_count);

        const auto worker_count =
            std::min<std::size_t>({
                8,
                available_workers,
                std::max<std::size_t>(
                    1,
                    (size + kMinRowsPerWorker - 1) /
                        kMinRowsPerWorker),
            });

        const auto base_native =
            base.native();

        const auto copy_range =
            [&](std::size_t begin,
                std::size_t end)
        {
            auto power =
                base.pow(
                        static_cast<std::uint64_t>(
                            begin))
                    .native();

            for (std::size_t index = begin;
                 index < end;
                 ++index)
            {
                mcl::Fr::mul(
                    output[index],
                    input[index],
                    power);

                mcl::Fr::mul(
                    power,
                    power,
                    base_native);
            }
        };

        if (worker_count <= 1)
        {
            copy_range(0, size);
            return;
        }

        const auto chunk_size =
            (size + worker_count - 1) /
            worker_count;

        std::vector<std::future<void>>
            futures;

        futures.reserve(worker_count);

        for (std::size_t worker = 0;
             worker < worker_count;
             ++worker)
        {
            const auto begin =
                worker * chunk_size;

            const auto end =
                std::min(
                    size,
                    begin + chunk_size);

            if (begin >= end)
            {
                break;
            }

            futures.push_back(
                std::async(
                    std::launch::async,
                    [&, begin, end]()
                    {
                        copy_range(
                            begin,
                            end);
                    }));
        }

        for (auto &future : futures)
        {
            future.get();
        }
    }

    void split_coset_coefficients_native(
        const mcl::Fr *coefficients,
        const mcl::Fr *shift_powers,
        std::size_t coefficient_count,
        const FieldElement &odd_root,
        mcl::Fr *even_output,
        mcl::Fr *odd_output,
        std::size_t max_worker_count)
    {
        if (coefficient_count == 0)
        {
            return;
        }

        if (coefficients == nullptr || shift_powers == nullptr || even_output == nullptr || odd_output == nullptr)
        {
            throw std::runtime_error(
                "coset split scaling requires non-null buffers");
        }

        constexpr std::size_t kMinRowsPerTask =
            1ULL << 18;

        const auto available_workers =
            bounded_worker_count(max_worker_count);

        const auto worker_count =
            std::min<std::size_t>({
                8,
                available_workers,
                std::max<std::size_t>(
                    1,
                    (coefficient_count + kMinRowsPerTask - 1) / kMinRowsPerTask),
            });

        const auto chunk_size =
            (coefficient_count + worker_count - 1) / worker_count;

        const auto root_native =
            odd_root.native();

        std::vector<std::future<void>> futures;
        futures.reserve(worker_count);

        for (std::size_t worker = 0;
             worker < worker_count;
             ++worker)
        {
            const auto begin =
                worker * chunk_size;
            const auto end =
                std::min(
                    coefficient_count,
                    begin + chunk_size);

            if (begin >= end)
            {
                break;
            }

            futures.push_back(
                std::async(
                    std::launch::async,
                    [=]
                    {
                        auto root_power =
                            odd_root.pow(begin).native();

                        for (std::size_t index = begin;
                             index < end;
                             ++index)
                        {
                            mcl::Fr::mul(
                                even_output[index],
                                coefficients[index],
                                shift_powers[index]);

                            mcl::Fr odd_shift_power;
                            mcl::Fr::mul(
                                odd_shift_power,
                                shift_powers[index],
                                root_power);

                            mcl::Fr::mul(
                                odd_output[index],
                                coefficients[index],
                                odd_shift_power);

                            mcl::Fr::mul(
                                root_power,
                                root_power,
                                root_native);
                        }
                    }));
        }

        for (auto &future : futures)
        {
            future.get();
        }
    }

    void interleave_native_lanes(
        const mcl::Fr *even_values,
        const mcl::Fr *odd_values,
        std::size_t lane_size,
        mcl::Fr *output,
        std::size_t max_worker_count)
    {
        if (lane_size == 0)
        {
            return;
        }

        if (even_values == nullptr || odd_values == nullptr || output == nullptr)
        {
            throw std::runtime_error(
                "native lane interleave requires non-null buffers");
        }

        constexpr std::size_t kMinRowsPerTask =
            1ULL << 18;

        const auto available_workers =
            bounded_worker_count(max_worker_count);

        const auto worker_count =
            std::min<std::size_t>({
                8,
                available_workers,
                std::max<std::size_t>(
                    1,
                    (lane_size + kMinRowsPerTask - 1) / kMinRowsPerTask),
            });

        const auto chunk_size =
            (lane_size + worker_count - 1) / worker_count;

        std::vector<std::future<void>> futures;
        futures.reserve(worker_count);

        for (std::size_t worker = 0;
             worker < worker_count;
             ++worker)
        {
            const auto begin =
                worker * chunk_size;
            const auto end =
                std::min(
                    lane_size,
                    begin + chunk_size);

            if (begin >= end)
            {
                break;
            }

            futures.push_back(
                std::async(
                    std::launch::async,
                    [=]
                    {
                        for (std::size_t index = begin;
                             index < end;
                             ++index)
                        {
                            output[2 * index] =
                                even_values[index];
                            output[2 * index + 1] =
                                odd_values[index];
                        }
                    }));
        }

        for (auto &future : futures)
        {
            future.get();
        }
    }

    namespace
    {
        void ntt_out_of_place_parallel_impl(
            mcl::Fr *source,
            mcl::Fr *output,
            std::size_t size,
            std::size_t transform_workers)
        {
            constexpr std::size_t kParallelSplitThreshold = 1ULL << 18;
            if (size < kParallelSplitThreshold || transform_workers < 2)
            {
                std::copy(source, source + size, output);
                ntt_plan(size)->ntt(output);
                return;
            }

            const auto half_size = size / 2;
            const auto lane_worker_count =
                std::min<std::size_t>({
                    8,
                    std::max<std::size_t>(
                        1,
                        transform_workers),
                    std::max<std::size_t>(
                        1,
                        (half_size + (1ULL << 18) - 1) / (1ULL << 18)),
                });

            const auto parallel_half_ranges =
                [&](const auto &range_fn)
            {
                if (lane_worker_count <= 1)
                {
                    range_fn(0, half_size);
                    return;
                }
                const auto chunk_size =
                    (half_size + lane_worker_count - 1) / lane_worker_count;
                std::vector<std::future<void>> futures;
                futures.reserve(lane_worker_count);
                for (std::size_t worker = 0;
                     worker < lane_worker_count; ++worker)
                {
                    const auto begin = worker * chunk_size;
                    const auto end = std::min(
                        half_size, begin + chunk_size);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(
                        std::launch::async,
                        [&, begin, end]
                        {
                            range_fn(begin, end);
                        }));
                }
                for (auto &future : futures)
                {
                    future.get();
                }
            };

            parallel_half_ranges([&](std::size_t begin, std::size_t end)
                                 {
                for (std::size_t index = begin; index < end; ++index)
                {
                    output[index] = source[2 * index];
                    output[index + half_size] = source[2 * index + 1];
                } });

            const auto even_workers = transform_workers / 2;
            const auto odd_workers = transform_workers - even_workers;
            auto even_task = std::async(
                std::launch::async,
                [&]
                {
                    ntt_out_of_place_parallel_impl(
                        output,
                        source,
                        half_size,
                        even_workers);
                });
            ntt_out_of_place_parallel_impl(
                output + half_size,
                source + half_size,
                half_size,
                odd_workers);
            even_task.get();

            const auto omega = FieldElement::root_of_unity(size).native();
            const auto omega_field = FieldElement::from_native(omega);
            parallel_half_ranges([&](std::size_t begin, std::size_t end)
                                 {
                auto omega_power = omega_field.pow(begin).native();
                for (std::size_t index = begin; index < end; ++index)
                {
                    mcl::Fr odd_scaled;
                    mcl::Fr::mul(
                        odd_scaled,
                        source[index + half_size],
                        omega_power);
                    mcl::Fr::add(
                        output[index], source[index], odd_scaled);
                    mcl::Fr::sub(
                        output[index + half_size],
                        source[index],
                        odd_scaled);
                    mcl::Fr::mul(
                        omega_power, omega_power, omega);
                } });
        }

        void intt_in_place_parallel_impl(
            mcl::Fr *values,
            std::size_t size,
            std::size_t transform_workers)
        {
            constexpr std::size_t kParallelSplitThreshold = 1ULL << 15;
            if (size < kParallelSplitThreshold || transform_workers < 2)
            {
                ntt_plan(size)->intt(values);
                return;
            }

            const auto half_size = size / 2;
            std::vector<mcl::Fr> even(half_size);
            std::vector<mcl::Fr> odd(half_size);

            const auto lane_worker_count =
                std::min<std::size_t>({
                    8,
                    std::max<std::size_t>(
                        1,
                        transform_workers),
                    std::max<std::size_t>(
                        1,
                        (half_size + (1ULL << 18) - 1) / (1ULL << 18)),
                });
            const auto parallel_half_ranges =
                [&](const auto &range_fn)
            {
                if (lane_worker_count <= 1)
                {
                    range_fn(0, half_size);
                    return;
                }
                const auto chunk_size =
                    (half_size + lane_worker_count - 1) / lane_worker_count;
                std::vector<std::future<void>> futures;
                futures.reserve(lane_worker_count);
                for (std::size_t worker = 0;
                     worker < lane_worker_count; ++worker)
                {
                    const auto begin = worker * chunk_size;
                    const auto end = std::min(
                        half_size, begin + chunk_size);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(
                        std::launch::async,
                        [&, begin, end]
                        {
                            range_fn(begin, end);
                        }));
                }
                for (auto &future : futures)
                {
                    future.get();
                }
            };
            parallel_half_ranges([&](std::size_t begin, std::size_t end)
                                 {
                for (std::size_t i = begin; i < end; ++i)
                {
                    even[i] = values[2 * i];
                    odd[i] = values[2 * i + 1];
                } });

            const auto even_workers = transform_workers / 2;
            const auto odd_workers = transform_workers - even_workers;
            auto even_task = std::async(
                std::launch::async,
                [&]
                {
                    intt_in_place_parallel_impl(
                        even.data(), half_size, even_workers);
                });
            intt_in_place_parallel_impl(
                odd.data(), half_size, odd_workers);
            even_task.get();

            const auto omega_inverse =
                FieldElement::root_of_unity(size).inv().native();
            const auto omega_inverse_field =
                FieldElement::from_native(omega_inverse);
            const auto inverse_two = FieldElement(2).inv().native();
            parallel_half_ranges([&](std::size_t begin, std::size_t end)
                                 {
                auto omega_power = omega_inverse_field.pow(begin).native();
                for (std::size_t i = begin; i < end; ++i)
                {
                    mcl::Fr odd_scaled;
                    mcl::Fr sum;
                    mcl::Fr difference;
                    mcl::Fr::mul(odd_scaled, odd[i], omega_power);
                    mcl::Fr::add(sum, even[i], odd_scaled);
                    mcl::Fr::sub(difference, even[i], odd_scaled);
                    mcl::Fr::mul(values[i], sum, inverse_two);
                    mcl::Fr::mul(
                        values[i + half_size], difference, inverse_two);
                    mcl::Fr::mul(
                        omega_power, omega_power, omega_inverse);
                } });
        }
    }

    void intt_in_place(
        mcl::Fr *values,
        std::size_t size,
        std::size_t max_worker_count)
    {
        if (values == nullptr || size == 0 || (size & (size - 1U)) != 0U)
        {
            throw std::runtime_error(
                "INTT requires a non-null power-of-two field vector");
        }

        if (should_use_cuda_intt(
                size))
        {
            if (!try_cuda_intt_in_place(
                    values,
                    size))
            {
                throw std::runtime_error(
                    "CUDA production INTT route "
                    "was selected but not admitted");
            }

            return;
        }

        const auto transform_workers =
            std::min<std::size_t>(
                32,
                bounded_worker_count(
                    max_worker_count));

        intt_in_place_parallel_impl(
            values,
            size,
            transform_workers);
    }

    void ntt_in_place(
        mcl::Fr *values,
        std::size_t size,
        std::size_t max_worker_count)
    {
        if (values == nullptr ||
            size == 0 ||
            (size & (size - 1U)) != 0U)
        {
            throw std::runtime_error(
                "NTT requires a non-null power-of-two field vector");
        }

        if (should_use_cuda_ntt(
                size))
        {
            if (!try_cuda_ntt_in_place(
                    values,
                    size))
            {
                throw std::runtime_error(
                    "CUDA production NTT route "
                    "was selected but not admitted");
            }

            return;
        }

        (void)max_worker_count;

        ntt_plan(size)->ntt(
            values);
    }

    void ntt_pair_in_place(
        mcl::Fr *first,
        mcl::Fr *second,
        std::size_t size,
        std::size_t max_worker_count)
    {
        if (first == nullptr || second == nullptr || first == second || size == 0 || (size & (size - 1U)) != 0U)
        {
            throw std::runtime_error(
                "paired NTT requires two distinct non-null power-of-two vectors");
        }

        if (should_use_cuda_ntt_pair(size))
        {
            if (!try_cuda_ntt_pair_in_place(
                    first,
                    second,
                    size))
            {
                throw std::runtime_error(
                    "CUDA production paired NTT route "
                    "was selected but not admitted");
            }

            return;
        }

        constexpr std::size_t kParallelPairThreshold =
            1ULL << 15;

        constexpr std::size_t kRecursiveParallelThreshold =
            1ULL << 18;

        constexpr std::size_t kRecursiveParallelMaximum =
            1ULL << 24;

        const auto available_workers =
            bounded_worker_count(
                max_worker_count);

        if (size < kParallelPairThreshold || available_workers < 2)
        {
            const auto plan =
                ntt_plan(size);

            plan->ntt(first);
            plan->ntt(second);

            return;
        }

        if (size >= kRecursiveParallelThreshold && size <= kRecursiveParallelMaximum && available_workers >= 4)
        {
            const auto per_transform_workers =
                std::max<std::size_t>(
                    1,
                    std::min<std::size_t>(
                        16,
                        available_workers / 2));

            auto run_transform =
                [=](mcl::Fr *values)
            {
                // Keep transform scratch scoped to this quotient transform.
                // A thread_local buffer could retain hundreds of
                // MiB after quotient materialization and carry that memory
                // pressure into KZG opening and verification.
                std::vector<mcl::Fr>
                    scratch(size);

                ntt_out_of_place_parallel_impl(
                    values,
                    scratch.data(),
                    size,
                    per_transform_workers);

                std::copy_n(
                    scratch.data(),
                    size,
                    values);
            };

            auto first_task =
                std::async(
                    std::launch::async,
                    [=]
                    {
                        run_transform(first);
                    });

            run_transform(second);
            first_task.get();

            return;
        }

        const auto plan =
            ntt_plan(size);

        auto first_task =
            std::async(
                std::launch::async,
                [=]
                {
                    plan->ntt(first);
                });

        plan->ntt(second);
        first_task.get();
    }

    bool try_cuda_intt_in_place(
        mcl::Fr *values,
        std::size_t size)
    {
#if GATZK_ENABLE_CUDA_BACKEND
        if (!cuda_field_hotspots_available())
        {
            return false;
        }

        return cuda_intt_in_place_exact(
            values,
            size);
#else
        (void)values;
        (void)size;
        return false;
#endif
    }

    bool try_cuda_ntt_in_place(
        mcl::Fr *values,
        std::size_t size)
    {
#if GATZK_ENABLE_CUDA_BACKEND
        if (!cuda_field_hotspots_available())
        {
            return false;
        }

        return cuda_ntt_in_place_exact(
            values,
            size);
#else
        (void)values;
        (void)size;
        return false;
#endif
    }

    bool try_cuda_ntt_pair_in_place(
        mcl::Fr *first,
        mcl::Fr *second,
        std::size_t size)
    {
#if GATZK_ENABLE_CUDA_BACKEND
        if (!cuda_field_hotspots_available())
        {
            return false;
        }

        return cuda_ntt_pair_in_place_exact(
            first,
            second,
            size);
#else
        (void)first;
        (void)second;
        (void)size;
        return false;
#endif
    }

    std::size_t cuda_kernel_invocation_count()
    {
        return cuda_kernel_counter().load(std::memory_order_relaxed);
    }

    void record_cuda_kernel_invocations(std::size_t count)
    {
        cuda_kernel_counter().fetch_add(count, std::memory_order_relaxed);
    }

    CudaExecutionStats cuda_execution_stats_snapshot()
    {
        const auto &counters =
            cuda_execution_counters();

        return CudaExecutionStats{
            .h2d_bytes =
                counters.h2d_bytes.load(
                    std::memory_order_relaxed),
            .d2h_bytes =
                counters.d2h_bytes.load(
                    std::memory_order_relaxed),
            .h2d_operations =
                counters.h2d_operations.load(
                    std::memory_order_relaxed),
            .d2h_operations =
                counters.d2h_operations.load(
                    std::memory_order_relaxed),
            .stream_synchronizations =
                counters.stream_synchronizations.load(
                    std::memory_order_relaxed),
            .ntt_calls =
                counters.ntt_calls.load(
                    std::memory_order_relaxed),
            .intt_calls =
                counters.intt_calls.load(
                    std::memory_order_relaxed),
            .quotient_calls =
                counters.quotient_calls.load(
                    std::memory_order_relaxed),
            .t_edge_calls =
                counters.t_edge_calls.load(
                    std::memory_order_relaxed),
            .opening_fold_calls =
                counters.opening_fold_calls.load(
                    std::memory_order_relaxed),
            .msm_single_calls =
                counters.msm_single_calls.load(
                    std::memory_order_relaxed),
            .msm_batch_calls =
                counters.msm_batch_calls.load(
                    std::memory_order_relaxed),
            .resident_hits =
                counters.resident_hits.load(
                    std::memory_order_relaxed),
            .resident_misses =
                counters.resident_misses.load(
                    std::memory_order_relaxed),
        };
    }

    void record_cuda_h2d(
        std::size_t bytes)
    {
        auto &counters =
            cuda_execution_counters();

        counters.h2d_bytes.fetch_add(
            bytes,
            std::memory_order_relaxed);

        counters.h2d_operations.fetch_add(
            1,
            std::memory_order_relaxed);
    }

    void record_cuda_d2h(
        std::size_t bytes)
    {
        auto &counters =
            cuda_execution_counters();

        counters.d2h_bytes.fetch_add(
            bytes,
            std::memory_order_relaxed);

        counters.d2h_operations.fetch_add(
            1,
            std::memory_order_relaxed);
    }

    void record_cuda_stream_sync()
    {
        cuda_execution_counters()
            .stream_synchronizations
            .fetch_add(
                1,
                std::memory_order_relaxed);
    }

    void record_cuda_ntt_call()
    {
        cuda_execution_counters()
            .ntt_calls
            .fetch_add(
                1,
                std::memory_order_relaxed);
    }

    void record_cuda_intt_call()
    {
        cuda_execution_counters()
            .intt_calls
            .fetch_add(
                1,
                std::memory_order_relaxed);
    }

    void record_cuda_quotient_call()
    {
        cuda_execution_counters()
            .quotient_calls
            .fetch_add(
                1,
                std::memory_order_relaxed);
    }

    void record_cuda_t_edge_call()
    {
        cuda_execution_counters()
            .t_edge_calls
            .fetch_add(
                1,
                std::memory_order_relaxed);
    }

    void record_cuda_opening_fold_call()
    {
        cuda_execution_counters()
            .opening_fold_calls
            .fetch_add(
                1,
                std::memory_order_relaxed);
    }
    void record_cuda_msm_single_call()
    {
        cuda_execution_counters()
            .msm_single_calls
            .fetch_add(
                1,
                std::memory_order_relaxed);
    }

    void record_cuda_msm_batch_call()
    {
        cuda_execution_counters()
            .msm_batch_calls
            .fetch_add(
                1,
                std::memory_order_relaxed);
    }

    void record_cuda_resident_hit()
    {
        cuda_execution_counters()
            .resident_hits
            .fetch_add(
                1,
                std::memory_order_relaxed);
    }

    void record_cuda_resident_miss()
    {
        cuda_execution_counters()
            .resident_misses
            .fetch_add(
                1,
                std::memory_order_relaxed);
    }
#if !GATZK_ENABLE_CUDA_BACKEND

    bool cuda_coset_evaluations_to_host_exact(
        const mcl::Fr *,
        std::size_t,
        std::size_t,
        const mcl::Fr *,
        std::size_t,
        const FieldElement &,
        mcl::Fr *,
        std::size_t,
        bool)
    {
        return false;
    }

    bool cuda_coset_coefficients_to_host_exact(
        const mcl::Fr *,
        std::size_t,
        const FieldElement &,
        mcl::Fr *,
        std::size_t,
        bool)
    {
        return false;
    }

    bool cuda_coset_host_to_coefficients_exact(
        const mcl::Fr *,
        std::size_t,
        const FieldElement &,
        mcl::Fr *)
    {
        return false;
    }

    void release_cuda_ntt_workspace()
    {
    }

    void release_cuda_scaled_accumulate_workspace()
    {
    }

    std::unique_ptr<CudaScaledAccumulateSession>
    create_cuda_scaled_accumulate_session(
        mcl::Fr *,
        std::size_t)
    {
        return nullptr;
    }

    std::unique_ptr<CudaWeightedDotSession>
    create_cuda_weighted_dot_session(
        const std::vector<
            const mcl::Fr *> &,
        std::size_t,
        std::size_t)
    {
        return nullptr;
    }

    void release_cuda_weighted_dot_cache()
    {
    }

#endif

    bool cuda_backend_build_enabled()
    {
#if GATZK_ENABLE_CUDA_BACKEND
        return true;
#else
        return false;
#endif
    }

    bool cuda_backend_available()
    {
#if GATZK_ENABLE_CUDA_BACKEND
        return cuda_backend_runtime_available();
#else
        return false;
#endif
    }

    bool cuda_backend_unified_addressing_available()
    {
#if GATZK_ENABLE_CUDA_BACKEND
        static const bool available = []()
        {
            if (!cuda_backend_runtime_available())
            {
                return false;
            }
            int device = 0;
            if (cudaGetDevice(&device) != cudaSuccess)
            {
                return false;
            }
            int unified = 0;
            if (cudaDeviceGetAttribute(&unified, cudaDevAttrUnifiedAddressing, device) != cudaSuccess)
            {
                return false;
            }
            int pageable_access = 0;
            if (cudaDeviceGetAttribute(
                    &pageable_access,
                    cudaDevAttrPageableMemoryAccess,
                    device) != cudaSuccess)
            {
                return false;
            }
            int host_page_tables = 0;
            if (cudaDeviceGetAttribute(
                    &host_page_tables,
                    cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                    device) != cudaSuccess)
            {
                return false;
            }
            // UVA alone only gives a common address range.  Likewise,
            // PageableMemoryAccess permits the runtime to migrate/copy ordinary
            // host pages, but does not make an arbitrary std::vector pointer a
            // valid kernel address.  Direct columnar kernels may consume those
            // pointers only when the device actually walks the host page tables.
            return unified != 0 && pageable_access != 0 && host_page_tables != 0;
        }();
        return available;
#else
        return false;
#endif
    }

    std::string cuda_device_summary()
    {
#if GATZK_ENABLE_CUDA_BACKEND
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
        {
            return "cuda_build=true; cuda_runtime=false; cuda_device=unavailable";
        }
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess)
        {
            device = 0;
        }
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, device) != cudaSuccess)
        {
            return "cuda_build=true; cuda_runtime=true; cuda_device=unknown; cuda_device_count=" + std::to_string(device_count);
        }
        int pageable_access = 0;
        int host_page_tables = 0;
        (void)cudaDeviceGetAttribute(
            &pageable_access,
            cudaDevAttrPageableMemoryAccess,
            device);
        (void)cudaDeviceGetAttribute(
            &host_page_tables,
            cudaDevAttrPageableMemoryAccessUsesHostPageTables,
            device);
        return "cuda_build=true; cuda_runtime=true; cuda_device=" +
               std::string(props.name) +
               "; cuda_device_count=" +
               std::to_string(device_count) +
               "; cuda_integrated=" +
               std::string(props.integrated ? "true" : "false") +
               "; cuda_unified_addressing=" +
               std::string(props.unifiedAddressing ? "true" : "false") +
               "; cuda_pageable_memory_access=" +
               std::string(pageable_access != 0 ? "true" : "false") +
               "; cuda_pageable_host_page_tables=" +
               std::string(host_page_tables != 0 ? "true" : "false");
#else
        return "cuda_build=false; cuda_runtime=false; cuda_device=unavailable";
#endif
    }

    bool cuda_field_hotspots_build_enabled()
    {
#if GATZK_ENABLE_CUDA_BACKEND
        return true;
#else
        return false;
#endif
    }

    bool cuda_field_hotspots_available()
    {
#if GATZK_ENABLE_CUDA_BACKEND
        return cuda_backend_runtime_available();
#else
        return false;
#endif
    }

    bool should_use_cuda_barycentric_weights(
        std::size_t domain_size,
        bool points_precomputed)
    {
        if (!cuda_field_route_enabled() || points_precomputed || domain_size == 0)
        {
            return false;
        }

        const auto min_domain_size =
            cuda_backend_unified_addressing_available()
                ? (1ULL << 16)
                : (1ULL << 18);
        return domain_size >= min_domain_size;
    }

    bool should_use_cuda_weighted_evaluation(
        std::size_t domain_size,
        std::size_t vector_count,
        std::size_t point_count)
    {
        if (!cuda_field_route_enabled() || domain_size == 0 || vector_count == 0 || point_count == 0)
        {
            return false;
        }

        const bool unified = cuda_backend_unified_addressing_available();
        const auto min_domain_size = unified ? (1ULL << 15) : (1ULL << 16);
        const auto min_work_items = unified ? 1ULL : 4ULL;
        const auto work_items = multiply_u128(vector_count, point_count);
        const auto total_terms = multiply_u128(domain_size, vector_count) * static_cast<unsigned __int128>(point_count);
        const auto min_total_terms =
            unified
                ? (static_cast<unsigned __int128>(1) << 21)
                : (static_cast<unsigned __int128>(1) << 24);

        return domain_size >= min_domain_size && work_items >= static_cast<unsigned __int128>(min_work_items) && total_terms >= min_total_terms;
    }

    bool should_use_cuda_lookup_histogram(
        std::size_t index_count,
        std::size_t bucket_count)
    {
        if (!cuda_trace_route_enabled() ||
            index_count == 0 ||
            bucket_count == 0)
        {
            return false;
        }

        const bool direct_pageable_host_access =
            cuda_backend_unified_addressing_available();

        const auto min_index_count =
            direct_pageable_host_access
                ? (1ULL << 20)
                : (1ULL << 22);

        const auto min_bucket_count =
            direct_pageable_host_access
                ? 1024ULL
                : 4096ULL;

        const auto min_indices_per_bucket =
            direct_pageable_host_access
                ? 256ULL
                : 1024ULL;

        // On pageable-host-page-table devices the current explicit
        // host conversion / allocation / transfer implementation is
        // empirically faster through 2^21 indices, but slower at 2^22
        // even with only 2^12 buckets. Do not extrapolate CUDA past
        // the largest measured-fast workload on that execution path.
        if (direct_pageable_host_access &&
            index_count > (1ULL << 21))
        {
            return false;
        }

        return index_count >= min_index_count &&
               bucket_count >= min_bucket_count &&
               index_count / bucket_count >=
                   min_indices_per_bucket;
    }

    bool should_use_cuda_intt(
        std::size_t size)
    {
        constexpr std::size_t
            kCudaInttMinimumSize =
                std::size_t{1} << 23;

        constexpr std::size_t
            kCudaInttMaximumSize =
                std::size_t{1} << 26;

        if (!util::route2_options()
                 .cuda_field_hotspots ||
            size < kCudaInttMinimumSize ||
            size > kCudaInttMaximumSize)
        {
            return false;
        }

#if GATZK_ENABLE_CUDA_BACKEND
        return cuda_field_hotspots_available();
#else
        return false;
#endif
    }

    bool should_use_cuda_ntt(
        std::size_t size)
    {
        constexpr std::size_t
            kCudaNttMinimumSize =
                std::size_t{1} << 23;

        constexpr std::size_t
            kCudaNttMaximumSize =
                std::size_t{1} << 26;

        if (!util::route2_options()
                 .cuda_field_hotspots ||
            size < kCudaNttMinimumSize ||
            size > kCudaNttMaximumSize)
        {
            return false;
        }

#if GATZK_ENABLE_CUDA_BACKEND
        return cuda_field_hotspots_available();
#else
        return false;
#endif
    }

    bool should_use_cuda_ntt_pair(
        std::size_t size)
    {
        constexpr std::size_t
            kCudaPairMinimumSize =
                std::size_t{1} << 23;

        constexpr std::size_t
            kCudaPairMaximumSize =
                std::size_t{1} << 26;

        if (!util::route2_options()
                 .cuda_field_hotspots ||
            size < kCudaPairMinimumSize ||
            size > kCudaPairMaximumSize)
        {
            return false;
        }

#if GATZK_ENABLE_CUDA_BACKEND
        return cuda_field_hotspots_available();
#else
        return false;
#endif
    }

    bool should_use_cuda_coset_transform(
        std::size_t target_size)
    {
        // Quotient cosets use two lanes of target_size / 2 elements.  The
        // bounded CUDA implementation supports the existing direct range and
        // staged radix-2/radix-4 extensions up to a 2^26 lane (2^27 target).
        // Keep the historical lower admission boundary so small transforms do
        // not pay a CUDA launch and pageable staging cost.
        constexpr std::size_t kMinimumTargetSize =
            std::size_t{1} << 24;
        constexpr std::size_t kMaximumTargetSize =
            std::size_t{1} << 27;

        if (!cuda_field_route_enabled() ||
            target_size < kMinimumTargetSize ||
            target_size > kMaximumTargetSize ||
            target_size < 2 ||
            (target_size & (target_size - 1U)) != 0U)
        {
            return false;
        }

        return true;
    }

    bool try_cuda_scaled_accumulate_native(
        mcl::Fr *target,
        const mcl::Fr *source,
        const mcl::Fr &scale,
        std::size_t element_count)
    {
        // Direct kernel access to std::vector storage is valid only when the
        // device walks the process host page tables.  Keep smaller vectors on
        // the existing CPU path so kernel launch and page-fault costs cannot
        // dominate the opening fold.
        const auto min_cuda_elements =
            cuda_backend_unified_addressing_available()
                ? (1ULL << 20)
                : (1ULL << 22);

        if (element_count < min_cuda_elements ||
            !cuda_field_route_enabled())
        {
            return false;
        }
        if (target == nullptr || source == nullptr)
        {
            throw std::runtime_error(
                "CUDA scaled accumulate received a null native buffer");
        }
#if GATZK_ENABLE_CUDA_BACKEND
        static_assert(
            sizeof(mcl::Fr) == 4 * sizeof(uint64_t),
            "CUDA scaled accumulate requires four-limb mcl::Fr storage");
        scaled_accumulate_cuda(
            reinterpret_cast<uint64_t *>(target),
            reinterpret_cast<const uint64_t *>(source),
            reinterpret_cast<const uint64_t *>(&scale),
            element_count);
        return true;
#else
        // cuda_field_route_enabled() is false in non-CUDA builds.
        (void)scale;
        return false;
#endif
    }

    bool try_cuda_scaled_accumulate_pair_native(
        mcl::Fr *first_target,
        mcl::Fr *second_target,
        const mcl::Fr *source,
        const mcl::Fr &first_scale,
        const mcl::Fr &second_scale,
        std::size_t element_count)
    {
        const auto min_cuda_elements =
            cuda_backend_unified_addressing_available()
                ? (1ULL << 20)
                : (1ULL << 22);

        if (element_count < min_cuda_elements ||
            !cuda_field_route_enabled())
        {
            return false;
        }

        if (first_target == nullptr ||
            second_target == nullptr ||
            source == nullptr)
        {
            throw std::runtime_error(
                "CUDA paired scaled accumulate received a null native buffer");
        }

        if (first_target == second_target ||
            first_target == source ||
            second_target == source)
        {
            throw std::runtime_error(
                "CUDA paired scaled accumulate requires distinct source and targets");
        }

#if GATZK_ENABLE_CUDA_BACKEND
        static_assert(
            sizeof(mcl::Fr) == 4 * sizeof(uint64_t),
            "CUDA paired scaled accumulate requires four-limb mcl::Fr storage");

        scaled_accumulate_pair_cuda(
            reinterpret_cast<uint64_t *>(first_target),
            reinterpret_cast<uint64_t *>(second_target),
            reinterpret_cast<const uint64_t *>(source),
            reinterpret_cast<const uint64_t *>(&first_scale),
            reinterpret_cast<const uint64_t *>(&second_scale),
            element_count);

        return true;
#else
        (void)first_scale;
        (void)second_scale;
        return false;
#endif
    }

#if !GATZK_ENABLE_CUDA_BACKEND
    // 非 CUDA 构建下保留同签名桩函数，让上层统一走回退逻辑。
    bool barycentric_weights_gpu(
        const uint64_t[4], const uint64_t[4], const uint64_t[4], const uint64_t[4],
        uint64_t *, std::size_t) { return false; }
    bool weighted_poly_eval_gpu(
        const uint64_t *, const uint64_t *, uint64_t *,
        std::size_t, std::size_t) { return false; }
    bool weighted_poly_eval_gpu_columnar(
        const uint64_t *, const uint64_t *const *, uint64_t *,
        std::size_t, std::size_t, const std::size_t *, std::size_t) { return false; }
    bool weighted_poly_eval_gpu_multi(
        const uint64_t *, const uint64_t *const *, uint64_t *,
        std::size_t, std::size_t, const std::size_t *, std::size_t) { return false; }
    bool weighted_poly_eval_gpu_multi_rotations(
        const uint64_t *, const uint64_t *const *, uint64_t *,
        std::size_t, std::size_t, const std::size_t *, std::size_t,
        const std::size_t *) { return false; }
    bool weighted_poly_eval_gpu_interleaved_multi_rotations(
        const uint64_t *, const uint64_t *, uint64_t *,
        std::size_t, std::size_t, const std::size_t *, std::size_t) { return false; }
    bool cuda_interleaved_multi_rotation_admissible(
        std::size_t, std::size_t, std::size_t, std::size_t) { return false; }
#endif

    // 点积是 polynomial / KZG / verifier 会反复调用的基础热路径。
    // 这里根据规模决定串行、并行或 CUDA 执行，但返回值必须完全一致。
    FieldElement dot_product(
        const std::vector<FieldElement> &lhs,
        const std::vector<FieldElement> &rhs)
    {
        if (lhs.size() != rhs.size())
        {
            throw std::runtime_error("dot product size mismatch");
        }
        if (lhs.empty())
        {
            return FieldElement::zero();
        }

        const auto &route2 = util::route2_options();
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        if (!route2.parallel_fft || lhs.size() < 1024 || cpu_count == 1)
        {
            return dot_product_range(lhs, rhs, 0, lhs.size());
        }

        // 决定并行任务数：最多cpu_count个，每个任务至少处理512个元素
        const auto task_count = std::min<std::size_t>(cpu_count, lhs.size() / 512);
        if (task_count <= 1)
        {
            return dot_product_range(lhs, rhs, 0, lhs.size());
        }

        std::vector<std::future<FieldElement>> futures;
        futures.reserve(task_count);
        const auto chunk_size = (lhs.size() + task_count - 1) / task_count;
        for (std::size_t task = 0; task < task_count; ++task)
        {
            const auto begin = task * chunk_size;
            const auto end = std::min(lhs.size(), begin + chunk_size);
            if (begin >= end)
            {
                break;
            }
            futures.push_back(std::async(
                std::launch::async,
                [&lhs, &rhs, begin, end]()
                {
                    return dot_product_range(lhs, rhs, begin, end);
                }));
        }

        FieldElement sum = FieldElement::zero();
        for (auto &future : futures)
        {
            sum += future.get();
        }
        return sum;
    }

    // 计算左向量（FieldElement）与右向量（mcl::Fr 原生权重）的点积（--parallel-fft on）
    FieldElement dot_product_native_weights(
        const std::vector<FieldElement> &lhs,
        const std::vector<mcl::Fr> &rhs)
    {
        if (lhs.size() != rhs.size())
        {
            throw std::runtime_error("dot product size mismatch");
        }
        if (lhs.empty())
        {
            return FieldElement::zero();
        }

        // CPU 路径：按 native 权重做同样的归约
        const auto &route2 = util::route2_options();
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        if (!route2.parallel_fft || lhs.size() < 1024 || cpu_count == 1)
        {
            return dot_product_native_range(lhs, rhs, 0, lhs.size());
        }

        const auto task_count = std::min<std::size_t>(cpu_count, lhs.size() / 512);
        if (task_count <= 1)
        {
            return dot_product_native_range(lhs, rhs, 0, lhs.size());
        }

        std::vector<std::future<FieldElement>> futures;
        futures.reserve(task_count);
        const auto chunk_size = (lhs.size() + task_count - 1) / task_count;
        for (std::size_t task = 0; task < task_count; ++task)
        {
            const auto begin = task * chunk_size;
            const auto end = std::min(lhs.size(), begin + chunk_size);
            if (begin >= end)
            {
                break;
            }
            futures.push_back(std::async(
                std::launch::async,
                [&lhs, &rhs, begin, end]()
                {
                    return dot_product_native_range(lhs, rhs, begin, end);
                }));
        }

        FieldElement sum = FieldElement::zero();
        for (auto &future : futures)
        {
            sum += future.get();
        }
        return sum;
    }
}
