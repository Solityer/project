#include "gatzk/crypto/curve.hpp"

#include <algorithm>
#include <array>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "gatzk/util/route2.hpp"

namespace gatzk::crypto
{
    // 本文件实现 KZG 依赖的群运算、批量 MSM 与序列化，
    // 是 protocol 层访问曲线后端的唯一实现落点。
    // 它属于 crypto 执行层，不允许改写承诺含义或 pairing 方程。

    namespace
    {
        // 同基点批量乘法会反复命中相同 `(base, scalar)`，这里用字符串键复用结果。
        std::string same_base_cache_key(const G1Point& point, const algebra::FieldElement& scalar)
        {
            return point.to_string() + "|" + scalar.to_string();
        }

        // 仅缓存执行层群乘结果，不跨越协议对象边界。
        class SameBaseMulCache
        {
        public:
            bool lookup(const G1Point& point, const algebra::FieldElement& scalar, G1Point* out)
            {
                const auto key = same_base_cache_key(point, scalar);
                std::lock_guard<std::mutex> lock(mutex_);
                if (const auto it = entries_.find(key); it != entries_.end())
                {
                    if (out != nullptr)
                    {
                        *out = it->second;
                    }
                    return true;
                }
                return false;
            }

            void store(const G1Point& point, const algebra::FieldElement& scalar, const G1Point& value)
            {
                const auto key = same_base_cache_key(point, scalar);
                std::lock_guard<std::mutex> lock(mutex_);
                entries_.emplace(key, value);
            }

        private:
            std::mutex mutex_;
            std::unordered_map<std::string, G1Point> entries_;
        };

        // 进程级单例缓存，供 MSM / 承诺等热点复用。
        SameBaseMulCache& same_base_mul_cache()
        {
            static SameBaseMulCache cache;
            return cache;
        }

        // 序列化接口同时服务 transcript 吸收与 artifact 导出。
        template <class Point>
        std::vector<std::uint8_t> serialize_impl(const Point& point)
        {
            algebra::ensure_mcl_field_ready();
            std::array<std::uint8_t, 256> buffer{};
            const auto written = point.serialize(buffer.data(), buffer.size());
            if (written == 0)
            {
                throw std::runtime_error("failed to serialize mcl point");
            }
            return std::vector<std::uint8_t>(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(written));
        }

    }

    // 文本化输出主要服务日志、manifest 与调试，不参与协议运算。
    std::string G1Point::to_string() const
    {
        algebra::ensure_mcl_field_ready();
        return value.serializeToHexStr();
    }

    std::string G2Point::to_string() const
    {
        algebra::ensure_mcl_field_ready();
        return value.serializeToHexStr();
    }

    // backend 名称会进入 artifact，帮助 README 与 manifest 对齐当前实现。
    std::string backend_name()
    {
        return GATZK_CRYPTO_BACKEND_NAME;
    }

    G1Point g1_zero()
    {
        G1Point out;
        algebra::ensure_mcl_field_ready();
        out.value.clear();
        return out;
    }

    G2Point g2_zero()
    {
        G2Point out;
        algebra::ensure_mcl_field_ready();
        out.value.clear();
        return out;
    }

    G1Point g1_add(const G1Point& lhs, const G1Point& rhs)
    {
        G1Point out;
        algebra::ensure_mcl_field_ready();
        mcl::G1::add(out.value, lhs.value, rhs.value);
        return out;
    }

    G1Point g1_sub(const G1Point& lhs, const G1Point& rhs)
    {
        G1Point out;
        algebra::ensure_mcl_field_ready();
        mcl::G1::sub(out.value, lhs.value, rhs.value);
        return out;
    }

    G1Point g1_neg(const G1Point& point)
    {
        G1Point out;
        algebra::ensure_mcl_field_ready();
        mcl::G1::neg(out.value, point.value);
        return out;
    }

    G1Point g1_mul(const G1Point& point, const algebra::FieldElement& scalar)
    {
        G1Point out;
        algebra::ensure_mcl_field_ready();
        mcl::G1::mul(out.value, point.value, scalar.native());
        return out;
    }

    std::size_t msm_worker_count(std::size_t scalar_count, std::size_t cpu_count)
    {
        if (scalar_count < 4096 || cpu_count <= 1)
        {
            return 1;
        }
        std::size_t by_size = 1;
        if (scalar_count >= (1ULL << 18))
        {
            by_size = 16;
        }
        else if (scalar_count >= (1ULL << 16))
        {
            by_size = 8;
        }
        else if (scalar_count >= (1ULL << 14))
        {
            by_size = 4;
        }
        else
        {
            by_size = 2;
        }
        return std::max<std::size_t>(1, std::min(cpu_count, by_size));
    }

    void mul_vec_chunked(
        mcl::G1& out,
        mcl::G1* points,
        const mcl::Fr* scalars,
        std::size_t count,
        std::size_t worker_count)
    {
        if (worker_count <= 1 || count < 4096)
        {
            mcl::G1::mulVec(out, points, scalars, count);
            return;
        }
        const auto task_count = std::min<std::size_t>(worker_count, count);
        const auto chunk_size = (count + task_count - 1) / task_count;
        std::vector<std::future<mcl::G1>> futures;
        futures.reserve(task_count);
        for (std::size_t task = 0; task < task_count; ++task)
        {
            const auto begin = task * chunk_size;
            const auto end = std::min(count, begin + chunk_size);
            if (begin >= end)
            {
                break;
            }
            futures.push_back(std::async(
                std::launch::async,
                [points, scalars, begin, end]() {
                    mcl::G1 partial;
                    mcl::G1::mulVec(partial, points + begin, scalars + begin, end - begin);
                    return partial;
                }));
        }
        for (auto& future : futures)
        {
            mcl::G1 partial = future.get();
            mcl::G1::add(out, out, partial);
        }
    }

    G1Point g1_multi_scalar_mul_profiled(
        const G1Point* points,
        const algebra::FieldElement* scalars,
        std::size_t count,
        G1MSMProfile* profile,
        const G1MSMExecutionOptions* options)
    {
        if ((points == nullptr || scalars == nullptr) && count != 0)
        {
            throw std::runtime_error("G1 MSM received null input");
        }
        if (profile != nullptr)
        {
            profile->input_count = count;
            profile->effective_scalar_count = count;
            profile->nonzero_scalar_count = count;
            profile->zero_scalar_skipping = false;
            profile->sparse_compaction = false;
            profile->multithreaded = false;
        }
        G1Point out = g1_zero();
        if (count == 0)
        {
            if (profile != nullptr)
            {
                profile->nonzero_scalar_count = 0;
                profile->effective_scalar_count = 0;
            }
            return out;
        }
        algebra::ensure_mcl_field_ready();
        const auto zero_scalar = algebra::FieldElement::zero();
        std::size_t nonzero_count = 0;
        if (count >= 1024 || profile != nullptr)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                if (!(scalars[i] == zero_scalar))
                {
                    ++nonzero_count;
                }
            }
            if (profile != nullptr)
            {
                profile->nonzero_scalar_count = nonzero_count;
            }
            if (nonzero_count == 0)
            {
                if (profile != nullptr)
                {
                    profile->zero_scalar_skipping = true;
                    profile->sparse_compaction = true;
                    profile->effective_scalar_count = 0;
                }
                return out;
            }
            if (nonzero_count * 4 < count * 3)
            {
                if (profile != nullptr)
                {
                    profile->zero_scalar_skipping = true;
                    profile->sparse_compaction = true;
                    profile->effective_scalar_count = nonzero_count;
                }
                std::vector<G1Point> compact_points;
                std::vector<algebra::FieldElement> compact_scalars;
                compact_points.reserve(nonzero_count);
                compact_scalars.reserve(nonzero_count);
                static_assert(sizeof(G1Point) == sizeof(mcl::G1), "G1Point must wrap a single mcl::G1");
                for (std::size_t i = 0; i < count; ++i)
                {
                    if (!(scalars[i] == zero_scalar))
                    {
                        compact_points.push_back(points[i]);
                        compact_scalars.push_back(scalars[i]);
                    }
                }
                auto* native_compact_points = reinterpret_cast<mcl::G1*>(compact_points.data());
                const auto* native_compact_scalars = algebra::FieldElement::native_array_data(compact_scalars.data());
                const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
                auto worker_count = msm_worker_count(compact_scalars.size(), cpu_count);
                if (options != nullptr && options->max_worker_count != 0)
                {
                    worker_count = std::min(worker_count, options->max_worker_count);
                }
                if (worker_count > 1)
                {
                    if (profile != nullptr)
                    {
                        profile->multithreaded = true;
                    }
                    mul_vec_chunked(
                        out.value,
                        native_compact_points,
                        native_compact_scalars,
                        compact_scalars.size(),
                        worker_count);
                }
                else
                {
                    mcl::G1::mulVec(out.value, native_compact_points, native_compact_scalars, compact_scalars.size());
                }
                return out;
            }
        }
        const auto* native_scalars = algebra::FieldElement::native_array_data(scalars);
        static_assert(sizeof(G1Point) == sizeof(mcl::G1), "G1Point must wrap a single mcl::G1");
        auto* native_points = const_cast<mcl::G1*>(reinterpret_cast<const mcl::G1*>(points));
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        auto worker_count = msm_worker_count(count, cpu_count);
        if (options != nullptr && options->max_worker_count != 0)
        {
            worker_count = std::min(worker_count, options->max_worker_count);
        }
        if (worker_count > 1)
        {
            if (profile != nullptr)
            {
                profile->multithreaded = true;
            }
            mul_vec_chunked(out.value, native_points, native_scalars, count, worker_count);
        }
        else
        {
            mcl::G1::mulVec(out.value, native_points, native_scalars, count);
        }
        return out;
    }

    G1Point g1_multi_scalar_mul(const G1Point* points, const algebra::FieldElement* scalars, std::size_t count)
    {
        return g1_multi_scalar_mul_profiled(points, scalars, count, nullptr);
    }

    G1Point g1_multi_scalar_mul(const std::vector<G1Point>& points, const std::vector<algebra::FieldElement>& scalars)
    {
        if (points.size() != scalars.size())
        {
            throw std::runtime_error("G1 MSM size mismatch");
        }
        return g1_multi_scalar_mul(points.data(), scalars.data(), points.size());
    }

    // 同基点批量乘法是 KZG 承诺与 witness 构造的典型热点。（--fast-msm on）
    std::vector<G1Point> g1_mul_same_base_batch(const G1Point& point, const std::vector<algebra::FieldElement>& scalars)
    {
        algebra::ensure_mcl_field_ready();
        std::vector<G1Point> out(scalars.size());
        if (scalars.empty())
        {
            return out;
        }

        std::vector<std::size_t> miss_indices;
        std::vector<algebra::FieldElement> miss_scalars;
        miss_indices.reserve(scalars.size());
        miss_scalars.reserve(scalars.size());
        // 先把缓存命中的 `(base, scalar)` 拿出来，未命中的再交给后面的 mcl 批处理。
        for (std::size_t i = 0; i < scalars.size(); ++i)
        {
            if (!same_base_mul_cache().lookup(point, scalars[i], &out[i]))
            {
                miss_indices.push_back(i);
                miss_scalars.push_back(scalars[i]);
            }
        }
        if (miss_indices.empty())
        {
            return out;
        }

        if (!util::route2_options().fast_msm)
        {
            for (std::size_t i = 0; i < miss_scalars.size(); ++i)
            {
                auto value = g1_mul(point, miss_scalars[i]);
                out[miss_indices[i]] = value;
                same_base_mul_cache().store(point, miss_scalars[i], value);
            }
            return out;
        }

        // 快速MSM路径
        // 把未命中的标量收成一次 `mulEach` 输入，尽量让同基点批乘只过一遍曲线后端。
        std::vector<mcl::G1> points(miss_scalars.size(), point.value);
        std::vector<mcl::Fr> native_scalars;
        native_scalars.reserve(miss_scalars.size());
        for (const auto& scalar : miss_scalars)
        {
            native_scalars.push_back(scalar.native());
        }

        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const bool run_parallel = cpu_count > 1 && miss_scalars.size() >= 256;
        if (run_parallel)
        {
            const auto task_count = std::min<std::size_t>(cpu_count, miss_scalars.size() / 128);
            if (task_count > 1)
            {
                const auto chunk_size = (miss_scalars.size() + task_count - 1) / task_count;
                std::vector<std::future<std::vector<mcl::G1>>> futures;
                futures.reserve(task_count);
                for (std::size_t task = 0; task < task_count; ++task)
                {
                    const auto begin = task * chunk_size;
                    const auto end = std::min<std::size_t>(miss_scalars.size(), begin + chunk_size);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(
                        std::launch::async,
                        [&, begin, end]() {
                            std::vector<mcl::G1> chunk(end - begin, point.value);
                            mcl::G1::mulEach(chunk.data(), native_scalars.data() + begin, chunk.size());
                            return chunk;
                        }));
                }
                std::size_t begin = 0;
                for (auto& future : futures)
                {
                    auto chunk = future.get();
                    for (std::size_t i = 0; i < chunk.size(); ++i)
                    {
                        G1Point value;
                        value.value = chunk[i];
                        out[miss_indices[begin + i]] = value;
                        same_base_mul_cache().store(point, miss_scalars[begin + i], value);
                    }
                    begin += chunk.size();
                }
                return out;
            }
        }

        // 串行回退时也保持同一批量接口，避免逐项乘法和缓存写回交错得太碎。
        mcl::G1::mulEach(points.data(), native_scalars.data(), points.size());
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            G1Point value;
            value.value = points[i];
            out[miss_indices[i]] = value;
            same_base_mul_cache().store(point, miss_scalars[i], value);
        }
        return out;
    }

    G2Point g2_add(const G2Point& lhs, const G2Point& rhs)
    {
        G2Point out;
        algebra::ensure_mcl_field_ready();
        mcl::G2::add(out.value, lhs.value, rhs.value);
        return out;
    }

    G2Point g2_sub(const G2Point& lhs, const G2Point& rhs)
    {
        G2Point out;
        algebra::ensure_mcl_field_ready();
        mcl::G2::sub(out.value, lhs.value, rhs.value);
        return out;
    }

    G2Point g2_mul(const G2Point& point, const algebra::FieldElement& scalar)
    {
        G2Point out;
        algebra::ensure_mcl_field_ready();
        mcl::G2::mul(out.value, point.value, scalar.native());
        return out;
    }

    bool operator==(const G1Point& lhs, const G1Point& rhs)
    {
        algebra::ensure_mcl_field_ready();
        return lhs.value == rhs.value;
    }

    bool operator==(const G2Point& lhs, const G2Point& rhs)
    {
        algebra::ensure_mcl_field_ready();
        return lhs.value == rhs.value;
    }

    // 序列化G1点为字节向量
    std::vector<std::uint8_t> serialize(const G1Point& point)
    {
        return serialize_impl(point.value);
    }

    std::vector<std::uint8_t> serialize(const G2Point& point)
    {
        return serialize_impl(point.value);
    }

    G1Point deserialize_g1(const std::vector<std::uint8_t>& bytes)
    {
        G1Point out;
        algebra::ensure_mcl_field_ready();
        const auto read = out.value.deserialize(bytes.data(), bytes.size());
        if (read != bytes.size())
        {
            throw std::runtime_error("failed to deserialize G1 point");
        }
        return out;
    }

    G2Point deserialize_g2(const std::vector<std::uint8_t>& bytes)
    {
        G2Point out;
        algebra::ensure_mcl_field_ready();
        const auto read = out.value.deserialize(bytes.data(), bytes.size());
        if (read != bytes.size())
        {
            throw std::runtime_error("failed to deserialize G2 point");
        }
        return out;
    }

    std::size_t serialized_size(const G1Point& point)
    {
        return serialize(point).size();
    }

    // 返回 G1 点序列化后的字节长度
    std::size_t serialized_size(const G2Point& point)
    {
        return serialize(point).size();
    }

    // 预计算G2点的配对系数，用于加速多次配对运算
    PreparedG2 prepare_g2(const G2Point& point)
    {
        algebra::ensure_mcl_field_ready();
        PreparedG2 prepared;
        mcl::precomputeG2(prepared.coeffs, point.value);
        return prepared;
    }

    bool pairing_equal(
        const G1Point& lhs_a,
        const G2Point& rhs_a,
        const G1Point& lhs_b,
        const G2Point& rhs_b)
    {
        algebra::ensure_mcl_field_ready();
        mcl::GT eval_a;
        mcl::GT eval_b;
        mcl::pairing(eval_a, lhs_a.value, rhs_a.value);
        mcl::pairing(eval_b, lhs_b.value, rhs_b.value);
        return eval_a == eval_b;
    }

    bool pairing_equal_with_prepared_rhs_a(
        const G1Point& lhs_a,
        const PreparedG2& rhs_a_prepared,
        const G1Point& lhs_b,
        const G2Point& rhs_b)
    {
        algebra::ensure_mcl_field_ready();
        mcl::Fp12 eval_a;
        mcl::Fp12 eval_b;
        mcl::precomputedMillerLoop(eval_a, lhs_a.value, rhs_a_prepared.coeffs);
        mcl::millerLoop(eval_b, lhs_b.value, rhs_b.value);
        mcl::finalExp(eval_a, eval_a);
        mcl::finalExp(eval_b, eval_b);
        return eval_a == eval_b;
    }

    // 判断动态点对与预计算点对的配对乘积是否为 1
    bool pairing_product_is_one_mixed_prepared(
        const G1Point& lhs_dynamic,
        const G2Point& rhs_dynamic,
        const G1Point& lhs_prepared,
        const PreparedG2& rhs_prepared)
    {
        algebra::ensure_mcl_field_ready();
        mcl::Fp12 miller_product;
        mcl::precomputedMillerLoop2mixed(
            miller_product,
            lhs_dynamic.value,
            rhs_dynamic.value,
            lhs_prepared.value,
            rhs_prepared.coeffs);
        mcl::finalExp(miller_product, miller_product);
        return miller_product.isOne();
    }
}
