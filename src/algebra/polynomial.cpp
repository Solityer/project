#include "gatzk/algebra/polynomial.hpp"

#include <algorithm>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::algebra
{
    namespace
    {
        // 用于并行计算时拆分区间 [begin, end)
        FieldElement barycentric_sum_range(
            const std::vector<FieldElement>& values,
            const std::vector<FieldElement>& domain_points,
            const FieldElement& x,
            std::size_t begin,
            std::size_t end)
        {
            FieldElement sum = FieldElement::zero();
            for (std::size_t i = begin; i < end; ++i)
            {
                // 计算重心插值公式中的部分和
                sum += values[i] * domain_points[i] / (x - domain_points[i]);
            }
            return sum;
        }

    }

    // 创建或从缓存中获取指定名称和大小的单位根域
    std::shared_ptr<RootOfUnityDomain> RootOfUnityDomain::create(const std::string& name, std::size_t size)
    {
        // 检查size是否为非零且为2的幂
        if (size == 0 || (size & (size - 1U)) != 0U)
        {
            throw std::runtime_error("domain size must be a non-zero power of two");
        }

        // 域名+大小 -> 域对象，避免重复创建
        static std::mutex cache_mutex;
        static std::unordered_map<std::string, std::shared_ptr<RootOfUnityDomain>> cache;
        const auto cache_key = name + ":" + std::to_string(size);
        {
            // 先加锁查找缓存
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (const auto it = cache.find(cache_key); it != cache.end())
            {
                return it->second;  // 命中缓存直接返回
            }
        }

        // 未命中，创建新域对象
        auto domain = std::make_shared<RootOfUnityDomain>();
        domain->name = name;
        domain->size = size;
        domain->omega = FieldElement::root_of_unity(size);
        domain->inv_size = FieldElement(static_cast<std::uint64_t>(size)).inv();

        // 如果size小于 2^24，预计算所有点及其缩放版本
        domain->points_precomputed = size < (1ULL << 24);
        if (domain->points_precomputed)
        {
            domain->points.resize(size);
            domain->points_scaled_by_inv_size.resize(size);
            // 第一个点是 1
            domain->points[0] = FieldElement::one();
            for (std::size_t i = 1; i < size; ++i)
            {
                domain->points[i] = domain->points[i - 1] * domain->omega;
            }

            // 每个点乘以inv_size，用于逆FFT或重心插值
            for (std::size_t i = 0; i < size; ++i)
            {
                domain->points_scaled_by_inv_size[i] = domain->points[i] * domain->inv_size;
            }
        }

        // 存入缓存
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache.emplace(cache_key, domain);
        }
        return domain;
    }

    // 返回域中第 index 个点（omega^index）
    FieldElement RootOfUnityDomain::point_at(std::size_t index) const
    {
        if (points_precomputed)
        {
            return points.at(index);
        }
        // 未预计算，动态计算幂
        return omega.pow(static_cast<std::uint64_t>(index));
    }

    // 返回第 index 个点乘以 inv_size 的结果
    FieldElement RootOfUnityDomain::point_scaled_by_inv_size_at(std::size_t index) const
    {
        if (points_precomputed)
        {
            return points_scaled_by_inv_size.at(index);
        }
        // 未预计算：先求点，再乘以 inv_size
        return point_at(index) * inv_size;
    }

    // 计算零点多项式 Z(x)=x^size-1 在x处的值
    FieldElement RootOfUnityDomain::zero_polynomial_eval(const FieldElement& x) const
    {
        return x.pow(size) - FieldElement::one();
    }

    // 计算第index个拉格朗日基函数在x处的值
    FieldElement RootOfUnityDomain::lagrange_basis_eval(std::size_t index, const FieldElement& x) const
    {
        const FieldElement xi = point_at(index);
        const FieldElement numerator = zero_polynomial_eval(x) * xi;
        return numerator * inv_size / (x - xi);
    }

    // 计算重心插值所需的权重（前缀积 + 后缀积）
    // 对于大域（size >= 2^20），使用并行分段批量求逆算法以避免串行瓶颈。
    std::vector<mcl::Fr> RootOfUnityDomain::barycentric_weights_native(const FieldElement& x) const
    {
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const bool use_parallel =
            util::route2_options().parallel_fft && size >= (1ULL << 20) && cpu_count > 1;

        if (!use_parallel)
        {
            // --- 串行实现 ---
            std::vector<mcl::Fr> denominators(size);
            std::vector<mcl::Fr> prefixes(size);
            mcl::Fr accumulator = 1;
            FieldElement point = FieldElement::one();
            for (std::size_t i = 0; i < size; ++i)
            {
                const auto current_point = points_precomputed ? points[i] : point;
                mcl::Fr::sub(denominators[i], x.native(), current_point.native());
                prefixes[i] = accumulator;
                mcl::Fr::mul(accumulator, accumulator, denominators[i]);
                if (!points_precomputed)
                {
                    point *= omega;
                }
            }

            mcl::Fr suffix_inverse;
            mcl::Fr::inv(suffix_inverse, accumulator);
            std::vector<mcl::Fr> weights(size);
            const auto zero_eval = zero_polynomial_eval(x).native();

            point = points_precomputed ? FieldElement::zero() : omega.pow(static_cast<std::uint64_t>(size - 1));
            const auto omega_inv = points_precomputed ? FieldElement::one() : omega.inv();
            for (std::size_t i = size; i-- > 0;)
            {
                mcl::Fr inverse;
                mcl::Fr::mul(inverse, suffix_inverse, prefixes[i]);
                const auto scaled_point = points_precomputed ? points_scaled_by_inv_size[i] : (point * inv_size);
                mcl::Fr::mul(weights[i], zero_eval, scaled_point.native());
                mcl::Fr::mul(weights[i], weights[i], inverse);
                mcl::Fr::mul(suffix_inverse, suffix_inverse, denominators[i]);
                if (!points_precomputed && i > 0)
                {
                    point *= omega_inv;
                }
            }
            return weights;
        }

        // --- 并行实现：分段批量求逆 ---
        // 数学原理：inv(d[i]) = (1/chunk_product[c]) × prefix_within[i] × suffix_within[i]
        // 其中 chunk_product[c] 为第 c 段所有分母之积，仅需一次全局求逆。
        // 各段可完全并行执行，全局归约仅 O(task_count) 步（串行但极快）。

        const std::size_t task_count =
            std::min<std::size_t>(cpu_count, (size + (1ULL << 20) - 1) / (1ULL << 20));
        const std::size_t chunk_size = (size + task_count - 1) / task_count;

        std::vector<mcl::Fr> denominators(size);
        std::vector<mcl::Fr> prefix_within(size);   // 段内前缀积（不含 i 自身）
        std::vector<mcl::Fr> chunk_products(task_count);
        std::vector<mcl::Fr> weights(size);
        const auto zero_eval = zero_polynomial_eval(x).native();

        // 预先计算 omega 逆元（仅在 !points_precomputed 时使用）
        const auto omega_inv = points_precomputed ? FieldElement::one() : omega.inv();

        // --- 阶段一：并行 --- 计算分母及段内前缀积
        {
            std::vector<std::future<void>> futures;
            futures.reserve(task_count);
            for (std::size_t task = 0; task < task_count; ++task)
            {
                futures.push_back(std::async(
                    std::launch::async,
                    [&, task]()
                    {
                        const std::size_t begin = task * chunk_size;
                        const std::size_t end = std::min(size, begin + chunk_size);
                        FieldElement omega_power = points_precomputed
                            ? FieldElement::zero()
                            : omega.pow(static_cast<std::uint64_t>(begin));

                        mcl::Fr acc;
                        acc = mcl::Fr(1);
                        for (std::size_t i = begin; i < end; ++i)
                        {
                            const auto current_pt =
                                points_precomputed ? points[i].native() : omega_power.native();
                            mcl::Fr::sub(denominators[i], x.native(), current_pt);
                            prefix_within[i] = acc;
                            mcl::Fr::mul(acc, acc, denominators[i]);
                            if (!points_precomputed)
                            {
                                omega_power *= omega;
                            }
                        }
                        chunk_products[task] = acc;
                    }));
            }
            for (auto& f : futures)
            {
                f.get();
            }
        }

        // --- 阶段二：串行 O(task_count) --- 跨段前缀积、一次全局求逆、各段外积逆元
        // chunk_outer_inv[c] = 1 / (其余所有段分母之积) = global_inv × cross_prefix[c] × cross_suffix[c]
        std::vector<mcl::Fr> cross_prefix(task_count + 1);
        cross_prefix[0] = mcl::Fr(1);
        for (std::size_t c = 0; c < task_count; ++c)
        {
            mcl::Fr::mul(cross_prefix[c + 1], cross_prefix[c], chunk_products[c]);
        }
        mcl::Fr global_inv;
        mcl::Fr::inv(global_inv, cross_prefix[task_count]);  // 唯一一次全局求逆

        std::vector<mcl::Fr> chunk_outer_inv(task_count);
        {
            mcl::Fr cross_suffix;
            cross_suffix = mcl::Fr(1);
            for (std::size_t c = task_count; c-- > 0;)
            {
                // chunk_outer_inv[c] = global_inv × cross_prefix[c] × cross_suffix_of_c
                //                    = 1 / chunk_product[c]  （数学恒等式）
                mcl::Fr tmp;
                mcl::Fr::mul(tmp, global_inv, cross_prefix[c]);
                mcl::Fr::mul(chunk_outer_inv[c], tmp, cross_suffix);
                mcl::Fr::mul(cross_suffix, cross_suffix, chunk_products[c]);
            }
        }

        // --- 阶段三：并行 --- 段内倒序扫描，合并段内后缀积与外积逆元，输出权重
        {
            std::vector<std::future<void>> futures;
            futures.reserve(task_count);
            for (std::size_t task = 0; task < task_count; ++task)
            {
                futures.push_back(std::async(
                    std::launch::async,
                    [&, task]()
                    {
                        const std::size_t begin = task * chunk_size;
                        const std::size_t end = std::min(size, begin + chunk_size);
                        if (begin >= end)
                        {
                            return;
                        }
                        // 对于 !points_precomputed：从末尾倒退跟踪 omega^i
                        FieldElement omega_power = points_precomputed
                            ? FieldElement::zero()
                            : omega.pow(static_cast<std::uint64_t>(end - 1));

                        mcl::Fr running_suffix;
                        running_suffix = mcl::Fr(1);  // = ∏_{j=i+1}^{end-1} d[j]（段内后缀积）

                        for (std::size_t i = end; i-- > begin;)
                        {
                            // inv_d[i] = chunk_outer_inv × prefix_within[i] × running_suffix
                            mcl::Fr inv_d;
                            mcl::Fr::mul(inv_d, chunk_outer_inv[task], prefix_within[i]);
                            mcl::Fr::mul(inv_d, inv_d, running_suffix);

                            // weights[i] = Z_N(x) × (omega^i/N) × inv_d[i]
                            const auto scaled_pt = points_precomputed
                                ? points_scaled_by_inv_size[i].native()
                                : (omega_power * inv_size).native();
                            mcl::Fr::mul(weights[i], zero_eval, scaled_pt);
                            mcl::Fr::mul(weights[i], weights[i], inv_d);

                            // 更新后缀积及 omega 幂次
                            mcl::Fr::mul(running_suffix, running_suffix, denominators[i]);
                            if (!points_precomputed && i > begin)
                            {
                                omega_power *= omega_inv;
                            }
                        }
                    }));
            }
            for (auto& f : futures)
            {
                f.get();
            }
        }

        return weights;
    }

    std::vector<FieldElement> RootOfUnityDomain::barycentric_weights(const FieldElement& x) const
    {
        const auto native_weights = barycentric_weights_native(x);
        std::vector<FieldElement> weights(size, FieldElement::zero());
        for (std::size_t i = 0; i < size; ++i)
        {
            weights[i] = FieldElement::from_native(native_weights[i]);
        }
        return weights;
    }
    
    // 给定域中两点from和to，返回满足 to = from * omega^shift 的shift值
    std::optional<std::size_t> RootOfUnityDomain::rotation_shift(
        const FieldElement& from,
        const FieldElement& to) const
    {
        if (from.is_zero() || to.is_zero())
        {
            return from == to ? std::optional<std::size_t>(0) : std::nullopt;
        }

        // 比例 ratio = to / from，寻找shift使得 omega^shift == ratio
        const auto ratio = to / from;
        FieldElement power = FieldElement::one();
        for (std::size_t shift = 0; shift < size; ++shift)
        {
            if (power == ratio)
            {
                return shift;
            }
            power *= omega;
        }
        return std::nullopt;
    }

    // 从系数列表创建多项式（基为系数形式）
    Polynomial Polynomial::from_coefficients(const std::string& name, std::vector<FieldElement> coefficients)
    {
        Polynomial out;
        out.name = name;
        out.basis = PolynomialBasis::Coefficient;
        out.data = std::move(coefficients);
        return out;
    }

    // 从求值列表创建多项式（基为求值形式），必须提供匹配的单位根域
    Polynomial Polynomial::from_evaluations(
        const std::string& name,
        std::vector<FieldElement> evaluations,
        const std::shared_ptr<RootOfUnityDomain>& domain)
    {
        if (domain == nullptr || evaluations.size() != domain->size)
        {
            throw std::runtime_error("evaluation polynomial requires a matching domain");
        }
        Polynomial out;
        out.name = name;
        out.basis = PolynomialBasis::Evaluation;
        out.data = std::move(evaluations);
        out.domain = domain;
        return out;
    }

    // 计算多项式在 x 处的值
    // 根据基选择不同算法：系数形式用霍纳法；求值形式用重心插值
    FieldElement Polynomial::evaluate(const FieldElement& x) const
    {
        // 系数形式：直接霍纳法
        if (basis == PolynomialBasis::Coefficient)
        {
            return horner(data, x);
        }

        // 求值形式：必须有域
        if (domain == nullptr)
        {
            throw std::runtime_error("evaluation polynomial is missing its domain");
        }

        // 如果 x 恰好是域上的一个已知点，直接返回对应值
        if (domain->points_precomputed)
        {
            for (std::size_t i = 0; i < domain->points.size(); ++i)
            {
                if (domain->points[i] == x)
                {
                    return data[i];
                }
            }
        }
        else if (const auto shift = domain->rotation_shift(FieldElement::one(), x); shift.has_value())
        {
            if (*shift < data.size())
            {
                return data[*shift];
            }
        }

         // 一般情况：使用重心插值公式
        const auto& route2 = util::route2_options();

        const FieldElement zero_eval = domain->zero_polynomial_eval(x);
        FieldElement sum = FieldElement::zero();
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        if (domain->points_precomputed && route2.parallel_fft && data.size() >= 1024 && cpu_count > 1)
        {
            const auto task_count = std::min<std::size_t>(cpu_count, data.size() / 512);
            const auto chunk_size = task_count > 1 ? (data.size() + task_count - 1) / task_count : data.size();
            if (task_count > 1)
            {
                std::vector<std::future<FieldElement>> futures;
                futures.reserve(task_count);
                for (std::size_t task = 0; task < task_count; ++task)
                {
                    const auto begin = task * chunk_size;
                    const auto end = std::min(data.size(), begin + chunk_size);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(
                        std::launch::async,
                        [&, begin, end]() {
                            return barycentric_sum_range(data, domain->points, x, begin, end);
                        }));
                }
                for (auto& future : futures)
                {
                    sum += future.get();
                }
                return zero_eval * domain->inv_size * sum;
            }
        }
        FieldElement point = FieldElement::one();
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            const auto current_point = domain->points_precomputed ? domain->points[i] : point;
            sum += data[i] * current_point / (x - current_point);
            if (!domain->points_precomputed)
            {
                point *= domain->omega;
            }
        }
        return zero_eval * domain->inv_size * sum;
    }

    // 霍纳方法求多项式值：系数从高次到低次存储
    FieldElement horner(const std::vector<FieldElement>& coefficients, const FieldElement& x)
    {
        FieldElement out = FieldElement::zero();
        for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it)
        {
            out *= x;
            out += *it;
        }
        return out;
    }

    // 拉格朗日插值：给定点集points和对应函数值values，计算在x处的插值结果
    FieldElement interpolate_at(
        const std::vector<FieldElement>& points,
        const std::vector<FieldElement>& values,
        const FieldElement& x)
    {
        if (points.size() != values.size())
        {
            throw std::runtime_error("interpolation points/value mismatch");
        }
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            if (points[i] == x)
            {
                return values[i];
            }
        }
        FieldElement total = FieldElement::zero();
        // 标准拉格朗日插值公式
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            FieldElement basis = FieldElement::one();
            for (std::size_t j = 0; j < points.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }
                basis *= (x - points[j]) / (points[i] - points[j]);
            }
            total += basis * values[i];
        }
        return total;
    }

    // 将矩阵（二维向量）按行展平为一维向量
    std::vector<FieldElement> flatten_matrix_coefficients(const std::vector<std::vector<FieldElement>>& matrix)
    {
        std::vector<FieldElement> out;
        for (const auto& row : matrix)
        {
            out.insert(out.end(), row.begin(), row.end());
        }
        return out;
    }
}