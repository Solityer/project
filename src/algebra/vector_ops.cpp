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

    void intt_in_place(mcl::Fr* values, std::size_t size)
    {
        if (values == nullptr || size == 0
            || (size & (size - 1U)) != 0U)
        {
            throw std::runtime_error(
                "INTT requires a non-null power-of-two field vector");
        }
        constexpr std::size_t kParallelSplitThreshold = 1ULL << 15;
        if (size < kParallelSplitThreshold
            || std::thread::hardware_concurrency() < 2)
        {
            ntt_plan(size)->intt(values);
            return;
        }

        const auto half_size = size / 2;
        std::vector<mcl::Fr> even(half_size);
        std::vector<mcl::Fr> odd(half_size);
        for (std::size_t i = 0; i < half_size; ++i)
        {
            even[i] = values[2 * i];
            odd[i] = values[2 * i + 1];
        }
        const auto half_plan = ntt_plan(half_size);
        auto even_task = std::async(std::launch::async, [&]
                                    { half_plan->intt(even.data()); });
        half_plan->intt(odd.data());
        even_task.get();

        const auto omega_inverse =
            FieldElement::root_of_unity(size).inv().native();
        const auto inverse_two = FieldElement(2).inv().native();
        mcl::Fr omega_power;
        omega_power = 1;
        for (std::size_t i = 0; i < half_size; ++i)
        {
            mcl::Fr odd_scaled;
            mcl::Fr sum;
            mcl::Fr difference;
            mcl::Fr::mul(odd_scaled, odd[i], omega_power);
            mcl::Fr::add(sum, even[i], odd_scaled);
            mcl::Fr::sub(difference, even[i], odd_scaled);
            mcl::Fr::mul(values[i], sum, inverse_two);
            mcl::Fr::mul(values[i + half_size], difference, inverse_two);
            mcl::Fr::mul(omega_power, omega_power, omega_inverse);
        }
    }

    void ntt_pair_in_place(mcl::Fr* first, mcl::Fr* second, std::size_t size)
    {
        if (first == nullptr || second == nullptr || first == second
            || size == 0 || (size & (size - 1U)) != 0U)
        {
            throw std::runtime_error(
                "paired NTT requires two distinct non-null power-of-two vectors");
        }
        const auto plan = ntt_plan(size);
        constexpr std::size_t kParallelPairThreshold = 1ULL << 15;
        if (size < kParallelPairThreshold
            || std::thread::hardware_concurrency() < 2)
        {
            plan->ntt(first);
            plan->ntt(second);
            return;
        }
        auto first_task = std::async(std::launch::async, [=]
                                     { plan->ntt(first); });
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
            return unified != 0 && pageable_access != 0;
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
        return "cuda_build=true; cuda_runtime=true; cuda_device=" + std::string(props.name)
            + "; cuda_device_count=" + std::to_string(device_count)
            + "; cuda_unified_addressing=" + std::string(props.unifiedAddressing ? "true" : "false");
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
        const auto min_index_count = 16384ULL;
        const auto min_bucket_count = unified ? 1024ULL : 4096ULL;
        return index_count >= min_index_count && bucket_count >= min_bucket_count;
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
