#include "gatzk/algebra/vector_ops.hpp"

#include <algorithm>
#include <atomic>
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
#endif

    namespace
    {
        std::atomic<std::size_t>& cuda_kernel_counter()
        {
            static std::atomic<std::size_t> counter{0};
            return counter;
        }

        bool cuda_field_route_enabled()
        {
            return util::route2_options().cuda_field_hotspots
                && cuda_field_hotspots_build_enabled()
                && cuda_field_hotspots_available();
        }

        bool cuda_trace_route_enabled()
        {
            return util::route2_options().cuda_trace_hotspots
                && cuda_backend_available();
        }

        unsigned __int128 multiply_u128(std::size_t lhs, std::size_t rhs)
        {
            return static_cast<unsigned __int128>(lhs) * static_cast<unsigned __int128>(rhs);
        }

        AlgebraBackend parse_backend_env()
        {
            const char* value = std::getenv("GATZK_ALGEBRA_BACKEND");
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
                    "legacy GATZK_ALGEBRA_BACKEND=cuda path has been removed; use --compute-backend cuda_hotspots on a CUDA-enabled build");
            }
            throw std::runtime_error("unsupported GATZK_ALGEBRA_BACKEND value: " + backend);
        }

        // 串行计算两个域元素向量在区间[begin, end)上的点积（--parallel-fft on）
        FieldElement dot_product_range(
            const std::vector<FieldElement>& lhs,
            const std::vector<FieldElement>& rhs,
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
            const std::vector<FieldElement>& lhs,
            const std::vector<mcl::Fr>& rhs,
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
        mcl::Fr* output,
        std::size_t size,
        const FieldElement& base)
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
        const auto available_workers = std::max<std::size_t>(
            1, std::thread::hardware_concurrency());
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
                        static_cast<std::uint64_t>(begin)).native();
                    for (std::size_t index = begin; index < end; ++index)
                    {
                        output[index] = power;
                        mcl::Fr::mul(power, power, base_native);
                    }
                }));
        }
        for (auto& future : futures)
        {
            future.get();
        }
    }

    void split_coset_coefficients_native(
        const mcl::Fr* coefficients,
        const mcl::Fr* shift_powers,
        std::size_t coefficient_count,
        const FieldElement& odd_root,
        mcl::Fr* even_output,
        mcl::Fr* odd_output)
    {
        if (coefficient_count == 0)
        {
            return;
        }
        if (coefficients == nullptr || shift_powers == nullptr
            || even_output == nullptr || odd_output == nullptr)
        {
            throw std::runtime_error(
                "coset split scaling requires non-null buffers");
        }
        constexpr std::size_t kMinRowsPerTask = 1ULL << 18;
        const auto worker_count = std::min<std::size_t>({
            8,
            std::max<std::size_t>(
                1, std::thread::hardware_concurrency()),
            std::max<std::size_t>(
                1,
                (coefficient_count + kMinRowsPerTask - 1)
                    / kMinRowsPerTask),
        });
        const auto chunk_size =
            (coefficient_count + worker_count - 1) / worker_count;
        const auto root_native = odd_root.native();
        std::vector<std::future<void>> futures;
        futures.reserve(worker_count);
        for (std::size_t worker = 0; worker < worker_count; ++worker)
        {
            const auto begin = worker * chunk_size;
            const auto end = std::min(
                coefficient_count, begin + chunk_size);
            if (begin >= end)
            {
                break;
            }
            futures.push_back(std::async(
                std::launch::async,
                [=]
                {
                    auto root_power = odd_root.pow(begin).native();
                    for (std::size_t index = begin; index < end; ++index)
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
                            root_power, root_power, root_native);
                    }
                }));
        }
        for (auto& future : futures)
        {
            future.get();
        }
    }

    void interleave_native_lanes(
        const mcl::Fr* even_values,
        const mcl::Fr* odd_values,
        std::size_t lane_size,
        mcl::Fr* output)
    {
        if (lane_size == 0)
        {
            return;
        }
        if (even_values == nullptr || odd_values == nullptr
            || output == nullptr)
        {
            throw std::runtime_error(
                "native lane interleave requires non-null buffers");
        }
        constexpr std::size_t kMinRowsPerTask = 1ULL << 18;
        const auto worker_count = std::min<std::size_t>({
            8,
            std::max<std::size_t>(
                1, std::thread::hardware_concurrency()),
            std::max<std::size_t>(
                1,
                (lane_size + kMinRowsPerTask - 1)
                    / kMinRowsPerTask),
        });
        const auto chunk_size =
            (lane_size + worker_count - 1) / worker_count;
        std::vector<std::future<void>> futures;
        futures.reserve(worker_count);
        for (std::size_t worker = 0; worker < worker_count; ++worker)
        {
            const auto begin = worker * chunk_size;
            const auto end = std::min(lane_size, begin + chunk_size);
            if (begin >= end)
            {
                break;
            }
            futures.push_back(std::async(
                std::launch::async,
                [=]
                {
                    for (std::size_t index = begin; index < end; ++index)
                    {
                        output[2 * index] = even_values[index];
                        output[2 * index + 1] = odd_values[index];
                    }
                }));
        }
        for (auto& future : futures)
        {
            future.get();
        }
    }

    namespace
    {
        void ntt_out_of_place_parallel_impl(
            mcl::Fr* source,
            mcl::Fr* output,
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
            const auto hardware_workers = std::max<std::size_t>(
                1, std::thread::hardware_concurrency());
            const auto lane_worker_count = std::min<std::size_t>({
                8,
                hardware_workers,
                std::max<std::size_t>(
                    1, (half_size + (1ULL << 18) - 1) / (1ULL << 18)),
            });
            const auto parallel_half_ranges =
                [&](const auto& range_fn)
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
                    for (auto& future : futures)
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
                }
            });

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
                }
            });
        }

        void intt_in_place_parallel_impl(
            mcl::Fr* values,
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
            const auto hardware_workers = std::max<std::size_t>(
                1, std::thread::hardware_concurrency());
            const auto lane_worker_count = std::min<std::size_t>({
                8,
                hardware_workers,
                std::max<std::size_t>(
                    1, (half_size + (1ULL << 18) - 1) / (1ULL << 18)),
            });
            const auto parallel_half_ranges =
                [&](const auto& range_fn)
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
                    for (auto& future : futures)
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
                }
            });

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
                }
            });
        }
    }

    void intt_in_place(mcl::Fr* values, std::size_t size)
    {
        if (values == nullptr || size == 0
            || (size & (size - 1U)) != 0U)
        {
            throw std::runtime_error(
                "INTT requires a non-null power-of-two field vector");
        }
        const auto transform_workers = std::min<std::size_t>(
            32,
            std::max<std::size_t>(
                1, std::thread::hardware_concurrency()));
        intt_in_place_parallel_impl(values, size, transform_workers);
    }

    void ntt_pair_in_place(mcl::Fr* first, mcl::Fr* second, std::size_t size)
    {
        if (first == nullptr || second == nullptr || first == second
            || size == 0 || (size & (size - 1U)) != 0U)
        {
            throw std::runtime_error(
                "paired NTT requires two distinct non-null power-of-two vectors");
        }
        constexpr std::size_t kParallelPairThreshold = 1ULL << 15;
        constexpr std::size_t kRecursiveParallelThreshold = 1ULL << 18;
        constexpr std::size_t kRecursiveParallelMaximum = 1ULL << 24;
        if (size < kParallelPairThreshold
            || std::thread::hardware_concurrency() < 2)
        {
            const auto plan = ntt_plan(size);
            plan->ntt(first);
            plan->ntt(second);
            return;
        }
        const auto hardware_workers = std::max<std::size_t>(
            1, std::thread::hardware_concurrency());
        if (size >= kRecursiveParallelThreshold
            && size <= kRecursiveParallelMaximum
            && hardware_workers >= 4)
        {
            const auto per_transform_workers = std::min<std::size_t>(
                16, hardware_workers / 2);
            auto run_transform = [=](mcl::Fr* values)
            {
                std::vector<mcl::Fr> scratch(size);
                ntt_out_of_place_parallel_impl(
                    values,
                    scratch.data(),
                    size,
                    per_transform_workers);
                std::copy(scratch.begin(), scratch.end(), values);
            };
            auto first_task = std::async(
                std::launch::async,
                [=]
                {
                    run_transform(first);
                });
            run_transform(second);
            first_task.get();
            return;
        }
        const auto plan = ntt_plan(size);
        auto first_task = std::async(
            std::launch::async,
            [=]
            {
                plan->ntt(first);
            });
        plan->ntt(second);
        first_task.get();
    }

    std::size_t cuda_kernel_invocation_count()
    {
        return cuda_kernel_counter().load(std::memory_order_relaxed);
    }

    void record_cuda_kernel_invocations(std::size_t count)
    {
        cuda_kernel_counter().fetch_add(count, std::memory_order_relaxed);
    }

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
        static const bool available = []() {
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
            return unified != 0
                && pageable_access != 0
                && host_page_tables != 0;
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
            return "cuda_build=true; cuda_runtime=true; cuda_device=unknown; cuda_device_count="
                + std::to_string(device_count);
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
        return "cuda_build=true; cuda_runtime=true; cuda_device=" + std::string(props.name)
            + "; cuda_device_count=" + std::to_string(device_count)
            + "; cuda_unified_addressing=" + std::string(props.unifiedAddressing ? "true" : "false")
            + "; cuda_pageable_memory_access=" + std::string(pageable_access != 0 ? "true" : "false")
            + "; cuda_pageable_host_page_tables=" + std::string(host_page_tables != 0 ? "true" : "false");
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

        return domain_size >= min_domain_size
            && work_items >= static_cast<unsigned __int128>(min_work_items)
            && total_terms >= min_total_terms;
    }

    bool should_use_cuda_lookup_histogram(
        std::size_t index_count,
        std::size_t bucket_count)
    {
        if (!cuda_trace_route_enabled() || index_count == 0 || bucket_count == 0)
        {
            return false;
        }

        const bool unified = cuda_backend_unified_addressing_available();
        const auto min_index_count = unified ? (1ULL << 20) : (1ULL << 22);
        const auto min_bucket_count = unified ? 1024ULL : 4096ULL;
        return index_count >= min_index_count
            && bucket_count >= min_bucket_count;
    }

#if !GATZK_ENABLE_CUDA_BACKEND
    // 非 CUDA 构建下保留同签名桩函数，让上层统一走回退逻辑。
    bool barycentric_weights_gpu(
        const uint64_t[4], const uint64_t[4], const uint64_t[4], const uint64_t[4],
        uint64_t*, std::size_t) { return false; }
    bool weighted_poly_eval_gpu(
        const uint64_t*, const uint64_t*, uint64_t*,
        std::size_t, std::size_t) { return false; }
    bool weighted_poly_eval_gpu_columnar(
        const uint64_t*, const uint64_t* const*, uint64_t*,
        std::size_t, std::size_t, const std::size_t*, std::size_t) { return false; }
    bool weighted_poly_eval_gpu_multi(
        const uint64_t*, const uint64_t* const*, uint64_t*,
        std::size_t, std::size_t, const std::size_t*, std::size_t) { return false; }
    bool weighted_poly_eval_gpu_multi_rotations(
        const uint64_t*, const uint64_t* const*, uint64_t*,
        std::size_t, std::size_t, const std::size_t*, std::size_t,
        const std::size_t*) { return false; }
#endif

    // 点积是 polynomial / KZG / verifier 会反复调用的基础热路径。
    // 这里根据规模决定串行、并行或 CUDA 执行，但返回值必须完全一致。
    FieldElement dot_product(
        const std::vector<FieldElement>& lhs,
        const std::vector<FieldElement>& rhs)
    {
        if (lhs.size() != rhs.size())
        {
            throw std::runtime_error("dot product size mismatch");
        }
        if (lhs.empty())
        {
            return FieldElement::zero();
        }

        const auto& route2 = util::route2_options();
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
                [&lhs, &rhs, begin, end]() {
                    return dot_product_range(lhs, rhs, begin, end);
                }));
        }

        FieldElement sum = FieldElement::zero();
        for (auto& future : futures)
        {
            sum += future.get();
        }
        return sum;
    }

    // 计算左向量（FieldElement）与右向量（mcl::Fr 原生权重）的点积（--parallel-fft on）
    FieldElement dot_product_native_weights(
        const std::vector<FieldElement>& lhs,
        const std::vector<mcl::Fr>& rhs)
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
        const auto& route2 = util::route2_options();
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
                [&lhs, &rhs, begin, end]() {
                    return dot_product_native_range(lhs, rhs, begin, end);
                }));
        }

        FieldElement sum = FieldElement::zero();
        for (auto& future : futures)
        {
            sum += future.get();
        }
        return sum;
    }
}
