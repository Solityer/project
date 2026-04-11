#include "gatzk/algebra/vector_ops.hpp"

#include <algorithm>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>

#include "gatzk/util/route2.hpp"

namespace gatzk::algebra
{
#if GATZK_ENABLE_CUDA_BACKEND
    bool cuda_backend_runtime_available();
#endif

    namespace
    {

        AlgebraBackend parse_backend_env()
        {
            const char* value = std::getenv("GATZK_ALGEBRA_BACKEND");
            if (value == nullptr)
            {
                // 如果环境变量未设置，默认返回CPU后端
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

        // 串行计算两个域元素向量在区间[begin, end)上的点积
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

    // 返回当前配置的代数后端
    AlgebraBackend configured_algebra_backend()
    {
        static const auto backend = parse_backend_env();
        return backend;
    }

    // 返回当前后端名称的字符串："cuda" 或 "cpu"
    std::string configured_algebra_backend_name()
    {
        (void)configured_algebra_backend();
        return "cpu";
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

    // 计算两个域元素向量的点积（自动选择CPU串行、CPU并行或CUDA）
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

    // 计算左向量（FieldElement）与右向量（mcl::Fr 原生权重）的点积
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
