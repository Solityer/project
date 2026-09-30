#include "gatzk/algebra/polynomial.hpp"

#include <algorithm>
#include <atomic>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/util/route2.hpp"
#include "gatzk/util/effective_memory.hpp"

namespace gatzk::algebra
{
    // 本文件实现工作域、多项式基变换与重心插值求值，
    // 是 trace / quotient / domain opening / verifier 共用的基础代数层。
    // 这里允许按 CPU / CUDA 路由切换执行策略，但不改变多项式本身的 formal 含义。

    namespace
    {
        // 用于并行计算时拆分区间 [begin, end)
        FieldElement barycentric_sum_range(const std::vector<FieldElement> &values, const std::vector<FieldElement> &domain_points, const FieldElement &x,
                                           std::size_t begin, std::size_t end)
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
    std::shared_ptr<RootOfUnityDomain> RootOfUnityDomain::create(const std::string &name, std::size_t size)
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
                return it->second; // 命中缓存直接返回
            }
        }

        // 未命中，创建新域对象
        auto domain = std::make_shared<RootOfUnityDomain>();
        domain->name = name;
        domain->size = size;
        domain->omega = FieldElement::root_of_unity(size);
        domain->inv_size = FieldElement(static_cast<std::uint64_t>(size)).inv();

        // 如果size小于 2^24，预计算所有点及其缩放版本
        domain->points_precomputed = size < (1ULL << 20);
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
    FieldElement RootOfUnityDomain::zero_polynomial_eval(const FieldElement &x) const
    {
        return x.pow(size) - FieldElement::one();
    }

    // 计算第index个拉格朗日基函数在x处的值
    FieldElement RootOfUnityDomain::lagrange_basis_eval(std::size_t index, const FieldElement &x) const
    {
        if (index >= size)
        {
            throw std::out_of_range("Lagrange basis index exceeds domain size");
        }
        if (const auto point_index = rotation_shift(FieldElement::one(), x); point_index.has_value())
        {
            return *point_index == index ? FieldElement::one() : FieldElement::zero();
        }
        const FieldElement xi = point_at(index);
        const FieldElement numerator = zero_polynomial_eval(x) * xi;
        return numerator * inv_size / (x - xi);
    }

    // 计算重心插值所需的权重（前缀积 + 后缀积）
    // 对于大域（size >= 2^20），使用并行分段批量求逆算法以避免串行瓶颈。
    // 当 CUDA field hotspots 可用且域足够大时，使用 GPU 加速。
    std::vector<mcl::Fr> RootOfUnityDomain::barycentric_weights_native(const FieldElement &x) const
    {
        const auto &route2 = util::route2_options();

        const bool try_gpu = should_use_cuda_barycentric_weights(size, points_precomputed);
        if (try_gpu)
        {
            const auto zero_eval = zero_polynomial_eval(x);
            std::vector<mcl::Fr> weights(size);
            const auto *ou = omega.native().getUnit();
            const auto *xu = x.native().getUnit();
            const auto *zu = zero_eval.native().getUnit();
            const auto *nu = inv_size.native().getUnit();
            uint64_t omega_limbs[4] = {ou[0], ou[1], ou[2], ou[3]};
            uint64_t x_limbs[4] = {xu[0], xu[1], xu[2], xu[3]};
            uint64_t zero_eval_limbs[4] = {zu[0], zu[1], zu[2], zu[3]};
            uint64_t inv_n_limbs[4] = {nu[0], nu[1], nu[2], nu[3]};
            const bool gpu_ok = barycentric_weights_gpu(omega_limbs, x_limbs, zero_eval_limbs, inv_n_limbs, reinterpret_cast<uint64_t *>(weights.data()), size);
            if (gpu_ok)
            {
                return weights;
            }
            throw std::runtime_error("CUDA barycentric weight kernel failed; refusing silent CPU fallback");
        }

        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const bool use_parallel = route2.parallel_fft && size >= (1ULL << 20) && cpu_count > 1;

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
        // 数学上仍然是在还原每个 `(x - omega^i)^{-1}`，只是把前后缀乘积拆到多线程执行。

        const std::size_t task_count = std::min<std::size_t>(cpu_count, (size + (1ULL << 20) - 1) / (1ULL << 20));
        const std::size_t chunk_size = (size + task_count - 1) / task_count;

        std::vector<mcl::Fr> denominators(size);
        std::vector<mcl::Fr> prefix_within(size); // 段内前缀积（不含 i 自身）
        std::vector<mcl::Fr> chunk_products(task_count);
        std::vector<mcl::Fr> weights(size);
        const auto zero_eval = zero_polynomial_eval(x).native();

        // 非预计算域需要显式倒退 `omega^i`，因此提前准备 `omega^{-1}`。
        const auto omega_inv = points_precomputed ? FieldElement::one() : omega.inv();

        // --- 阶段一：并行 --- 计算分母及段内前缀积
        {
            std::vector<std::future<void>> futures;
            futures.reserve(task_count);
            for (std::size_t task = 0; task < task_count; ++task)
            {
                futures.push_back(std::async(std::launch::async,
                                             [&, task]()
                                             {
                                                 const std::size_t begin = task * chunk_size;
                                                 const std::size_t end = std::min(size, begin + chunk_size);
                                                 FieldElement omega_power =
                                                     points_precomputed ? FieldElement::zero() : omega.pow(static_cast<std::uint64_t>(begin));

                                                 mcl::Fr acc;
                                                 acc = mcl::Fr(1);
                                                 for (std::size_t i = begin; i < end; ++i)
                                                 {
                                                     const auto current_pt = points_precomputed ? points[i].native() : omega_power.native();
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
            for (auto &f : futures)
            {
                f.get();
            }
        }

        // --- 阶段二：串行 O(task_count) ---
        // 先做跨段前缀积，再只做一次全局求逆，最后恢复每段自己的外积逆元。
        std::vector<mcl::Fr> cross_prefix(task_count + 1);
        cross_prefix[0] = mcl::Fr(1);
        for (std::size_t c = 0; c < task_count; ++c)
        {
            mcl::Fr::mul(cross_prefix[c + 1], cross_prefix[c], chunk_products[c]);
        }
        mcl::Fr global_inv;
        mcl::Fr::inv(global_inv, cross_prefix[task_count]); // 唯一一次全局求逆

        std::vector<mcl::Fr> chunk_outer_inv(task_count);
        {
            mcl::Fr cross_suffix;
            cross_suffix = mcl::Fr(1);
            for (std::size_t c = task_count; c-- > 0;)
            {
                // 数学上等价于 `1 / chunk_product[c]`，只是写成更便于增量计算的形式。
                mcl::Fr tmp;
                mcl::Fr::mul(tmp, global_inv, cross_prefix[c]);
                mcl::Fr::mul(chunk_outer_inv[c], tmp, cross_suffix);
                mcl::Fr::mul(cross_suffix, cross_suffix, chunk_products[c]);
            }
        }

        // --- 阶段三：并行 ---
        // 倒序补齐段内后缀积，并把外积逆元合并回每个位置的最终权重。
        {
            std::vector<std::future<void>> futures;
            futures.reserve(task_count);
            for (std::size_t task = 0; task < task_count; ++task)
            {
                futures.push_back(std::async(std::launch::async,
                                             [&, task]()
                                             {
                                                 const std::size_t begin = task * chunk_size;
                                                 const std::size_t end = std::min(size, begin + chunk_size);
                                                 if (begin >= end)
                                                 {
                                                     return;
                                                 }
                                                 // 非预计算域需要从段末尾开始倒推 `omega^i`。
                                                 FieldElement omega_power =
                                                     points_precomputed ? FieldElement::zero() : omega.pow(static_cast<std::uint64_t>(end - 1));

                                                 mcl::Fr running_suffix;
                                                 running_suffix = mcl::Fr(1); // 表示段内后缀积 `∏_{j=i+1}^{end-1} d[j]`

                                                 for (std::size_t i = end; i-- > begin;)
                                                 {
                                                     // 还原当前分母的逆元。
                                                     mcl::Fr inv_d;
                                                     mcl::Fr::mul(inv_d, chunk_outer_inv[task], prefix_within[i]);
                                                     mcl::Fr::mul(inv_d, inv_d, running_suffix);

                                                     // 写出与 formal 主线一致的 barycentric 权重。
                                                     const auto scaled_pt =
                                                         points_precomputed ? points_scaled_by_inv_size[i].native() : (omega_power * inv_size).native();
                                                     mcl::Fr::mul(weights[i], zero_eval, scaled_pt);
                                                     mcl::Fr::mul(weights[i], weights[i], inv_d);

                                                     // 更新段内后缀积和 `omega^i`。
                                                     mcl::Fr::mul(running_suffix, running_suffix, denominators[i]);
                                                     if (!points_precomputed && i > begin)
                                                     {
                                                         omega_power *= omega_inv;
                                                     }
                                                 }
                                             }));
            }
            for (auto &f : futures)
            {
                f.get();
            }
        }

        return weights;
    }

    std::vector<FieldElement> RootOfUnityDomain::barycentric_weights(const FieldElement &x) const
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
    std::optional<std::size_t> RootOfUnityDomain::rotation_shift(const FieldElement &from, const FieldElement &to) const
    {
        if (from.is_zero() || to.is_zero())
        {
            return from == to ? std::optional<std::size_t>(0) : std::nullopt;
        }

        // 比例 ratio = to / from，寻找shift使得 omega^shift == ratio
        const auto ratio = to / from;
        // The domain is exactly the size-th roots of unity.  Fiat--Shamir
        // challenge points are almost always outside that subgroup; reject
        // those in O(log size) before attempting the exact discrete-log scan.
        // In-domain rotations still use the canonical scan below, preserving
        // the returned shift for every accepted ratio.
        if (ratio.pow(size) != FieldElement::one())
        {
            return std::nullopt;
        }
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
    Polynomial Polynomial::from_coefficients(const std::string &name, std::vector<FieldElement> coefficients)
    {
        Polynomial out;
        out.name = name;
        out.basis = PolynomialBasis::Coefficient;
        out.data = std::move(coefficients);
        return out;
    }

    // 从求值列表创建多项式（基为求值形式），必须提供匹配的单位根域
    Polynomial Polynomial::from_evaluations(const std::string &name, std::vector<FieldElement> evaluations, const std::shared_ptr<RootOfUnityDomain> &domain)
    {
        if (domain == nullptr || evaluations.size() > domain->size)
        {
            throw std::runtime_error("evaluation polynomial requires a compatible domain");
        }
        Polynomial out;
        out.name = name;
        out.basis = PolynomialBasis::Evaluation;
        out.data = std::move(evaluations);
        out.domain = domain;
        return out;
    }

    Polynomial Polynomial::from_shared_evaluations(const std::string &name, std::shared_ptr<const std::vector<FieldElement>> evaluations,
                                                   const std::shared_ptr<RootOfUnityDomain> &domain)
    {
        if (domain == nullptr || evaluations == nullptr || evaluations->size() > domain->size)
        {
            throw std::runtime_error("evaluation polynomial requires a compatible shared domain");
        }
        Polynomial out;
        out.name = name;
        out.basis = PolynomialBasis::Evaluation;
        out.shared_data = std::move(evaluations);
        out.domain = domain;
        return out;
    }

    const std::vector<FieldElement> &Polynomial::values() const
    {
        return shared_data != nullptr ? *shared_data : data;
    }

    std::size_t Polynomial::size() const
    {
        return shared_data != nullptr ? shared_data->size() : data.size();
    }

    const FieldElement &Polynomial::value_at(std::size_t index) const
    {
        return shared_data != nullptr ? shared_data->at(index) : data.at(index);
    }

    // 计算多项式在 x 处的值
    // 根据基选择不同算法：系数形式用霍纳法；求值形式用重心插值
    FieldElement Polynomial::semantic_evaluate(const FieldElement &x) const
    {
        if (basis == PolynomialBasis::Coefficient)
            return horner(values(), x);
        if (domain == nullptr)
            throw std::runtime_error("evaluation polynomial is missing its domain");
        const auto &source = values();
        if (source.size() > domain->size)
            throw std::runtime_error("evaluation polynomial exceeds its domain");
        if (source.empty())
            return FieldElement::zero();
        if (const auto index = domain->rotation_shift(FieldElement::one(), x); index.has_value())
            return *index < source.size() ? source[*index] : FieldElement::zero();

        const auto scale = domain->zero_polynomial_eval(x) * domain->inv_size;
        if (domain->points_precomputed && source.size() < 32)
            return scale * barycentric_sum_range(source, domain->points, x, 0, source.size());

        // Evaluate the original barycentric formula with one inversion per
        // bounded block, not one inversion per scalar. No full-domain weights
        // vector is retained by this direct-evaluation path.
        constexpr std::size_t grain = std::size_t{1} << 14;
        const auto blocks = source.size() / grain + (source.size() % grain != 0);
        const auto hardware = std::max(1U, std::thread::hardware_concurrency());
        const auto workers = util::route2_options().parallel_fft ? std::min<std::size_t>(blocks, hardware) : std::size_t{1};
        const auto *input = FieldElement::native_array_data(source.data());
        const auto *points = domain->points_precomputed ? FieldElement::native_array_data(domain->points.data()) : nullptr;
        const auto omega_inverse = domain->omega.inv();
        std::vector<FieldElement> partial(blocks, FieldElement::zero());
        std::atomic<std::size_t> next{0};
        const auto worker = [&]
        {
            std::vector<mcl::Fr> prefix(std::min(grain, source.size()));
            for (;;)
            {
                const auto block = next.fetch_add(1, std::memory_order_relaxed);
                if (block >= blocks)
                    break;
                const auto begin = block * grain;
                const auto count = std::min(grain, source.size() - begin);
                auto point = domain->point_at(begin).native();
                mcl::Fr product = FieldElement::one().native();
                for (std::size_t offset = 0; offset < count; ++offset)
                {
                    prefix[offset] = product;
                    const auto &current = points != nullptr ? points[begin + offset] : point;
                    mcl::Fr denominator;
                    mcl::Fr::sub(denominator, x.native(), current);
                    mcl::Fr::mul(product, product, denominator);
                    if (points == nullptr)
                        mcl::Fr::mul(point, point, domain->omega.native());
                }
                mcl::Fr inverse;
                mcl::Fr::inv(inverse, product);
                if (points == nullptr)
                    mcl::Fr::mul(point, point, omega_inverse.native());
                mcl::Fr sum;
                sum.clear();
                for (std::size_t offset = count; offset-- > 0;)
                {
                    const auto &current = points != nullptr ? points[begin + offset] : point;
                    mcl::Fr denominator;
                    mcl::Fr reciprocal;
                    mcl::Fr term;
                    mcl::Fr::sub(denominator, x.native(), current);
                    mcl::Fr::mul(reciprocal, inverse, prefix[offset]);
                    mcl::Fr::mul(term, input[begin + offset], current);
                    mcl::Fr::mul(term, term, reciprocal);
                    mcl::Fr::add(sum, sum, term);
                    mcl::Fr::mul(inverse, inverse, denominator);
                    if (points == nullptr)
                        mcl::Fr::mul(point, point, omega_inverse.native());
                }
                partial[block] = FieldElement::from_native(sum);
            }
        };
        if (workers <= 1)
            worker();
        else
        {
            std::vector<std::future<void>> futures;
            futures.reserve(workers);
            for (std::size_t index = 0; index < workers; ++index)
                futures.push_back(std::async(std::launch::async, worker));
            for (auto &future : futures)
                future.get();
        }
        auto sum = FieldElement::zero();
        for (const auto &value : partial)
            sum += value;
        return scale * sum;
    }

    void Polynomial::set_zero_domain_blinder(std::vector<FieldElement> coefficients)
    {
        if (basis != PolynomialBasis::Evaluation || domain == nullptr)
        {
            throw std::runtime_error("zero-domain blinding requires an evaluation polynomial");
        }
        while (!coefficients.empty() && coefficients.back() == FieldElement::zero())
        {
            coefficients.pop_back();
        }
        zero_domain_blinder = std::move(coefficients);
    }

    bool Polynomial::is_zero_domain_blinded() const
    {
        return !zero_domain_blinder.empty();
    }

    FieldElement Polynomial::evaluate(const FieldElement &x) const
    {
        const auto semantic = semantic_evaluate(x);
        if (!is_zero_domain_blinded())
        {
            return semantic;
        }
        return semantic + domain->zero_polynomial_eval(x) * horner(zero_domain_blinder, x);
    }

    // 霍纳方法求多项式值：系数从高次到低次存储
    FieldElement horner(const std::vector<FieldElement> &coefficients, const FieldElement &x)
    {
        if (coefficients.empty())
        {
            return FieldElement::zero();
        }
        constexpr std::size_t kBlockElements = std::size_t{1} << 16;
        constexpr std::size_t kParallelElements = std::size_t{1} << 19;
        const auto hardware = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const auto *input = FieldElement::native_array_data(coefficients.data());
        const auto &point = x.native();

        const auto evaluate_block = [&](std::size_t begin, std::size_t end)
        {
            mcl::Fr value;
            value.clear();
            for (std::size_t index = end; index-- > begin;)
            {
                mcl::Fr::mul(value, value, point);
                mcl::Fr::add(value, value, input[index]);
            }
            return FieldElement::from_native(value);
        };

        if (!util::route2_options().parallel_fft || coefficients.size() < kParallelElements || hardware == 1)
        {
            return evaluate_block(0, coefficients.size());
        }

        const auto block_count = 1 + (coefficients.size() - 1) / kBlockElements;
        const auto worker_count = std::min(hardware, block_count);
        std::vector<FieldElement> partial(block_count, FieldElement::zero());
        std::vector<std::future<void>> futures;
        futures.reserve(worker_count);

        std::atomic<std::size_t> next_block{0};
        for (std::size_t worker = 0; worker < worker_count; ++worker)
        {
            futures.push_back(std::async(std::launch::async,
                                         [&]()
                                         {
                                             while (true)
                                             {
                                                 const auto block = next_block.fetch_add(1, std::memory_order_relaxed);
                                                 if (block >= block_count)
                                                 {
                                                     break;
                                                 }
                                                 const auto begin = block * kBlockElements;
                                                 const auto count = std::min(kBlockElements, coefficients.size() - begin);
                                                 partial[block] = evaluate_block(begin, begin + count);
                                             }
                                         }));
        }
        for (auto &future : futures)
        {
            future.get();
        }

        const auto stride = x.pow(kBlockElements);
        auto result = FieldElement::zero();
        for (std::size_t block = block_count; block-- > 0;)
        {
            result *= stride;
            result += partial[block];
        }
        return result;
    }

    // 拉格朗日插值：给定点集points和对应函数值values，计算在x处的插值结果
    FieldElement interpolate_at(const std::vector<FieldElement> &points, const std::vector<FieldElement> &values, const FieldElement &x)
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

    std::vector<FieldElement> materialize_blinded_coefficients(const Polynomial &polynomial, std::size_t configured_degree)
    {
        if (configured_degree == std::numeric_limits<std::size_t>::max())
        {
            throw std::runtime_error("configured polynomial degree overflows coefficient length");
        }
        const auto configured_length = configured_degree + 1;
        const auto &source = polynomial.values();

        if (polynomial.basis == PolynomialBasis::Coefficient)
        {
            if (source.size() > configured_length)
            {
                throw std::runtime_error("coefficient polynomial exceeds configured shift degree: " + polynomial.name);
            }
            if (!polynomial.zero_domain_blinder.empty())
            {
                throw std::runtime_error("coefficient polynomial cannot carry a domain-vanishing blinder: " + polynomial.name);
            }
            std::vector<FieldElement> out(configured_length, FieldElement::zero());
            std::copy(source.begin(), source.end(), out.begin());
            return out;
        }

        if (polynomial.domain == nullptr)
        {
            throw std::runtime_error("evaluation polynomial is missing its domain during coefficient materialization: " + polynomial.name);
        }
        const auto n = polynomial.domain->size;
        if (source.size() > n || n > configured_length)
        {
            throw std::runtime_error("evaluation polynomial exceeds configured shift degree: " + polynomial.name);
        }
        if (polynomial.zero_domain_blinder.size() > configured_length - n)
        {
            throw std::runtime_error("domain-vanishing blinder exceeds configured shift degree: " + polynomial.name);
        }

        // The final buffer owns both the interpolant and its registered tail.
        // Transform only the first n entries; never transform the blinder.
        std::vector<FieldElement> out(configured_length, FieldElement::zero());
        std::copy(source.begin(), source.end(), out.begin());
        intt_in_place(FieldElement::mutable_native_array_data(out.data()), n);

        for (std::size_t i = 0; i < polynomial.zero_domain_blinder.size(); ++i)
        {
            out[i] -= polynomial.zero_domain_blinder[i];
            out[n + i] += polynomial.zero_domain_blinder[i];
        }
        return out;
    }

    namespace
    {
        template <typename Fn> void taylor_parallel_ranges(std::size_t count, Fn &&fn)
        {
            constexpr std::size_t grain = std::size_t{1} << 15;
            const auto blocks = count / grain + (count % grain != 0);
            const auto workers =
                util::route2_options().parallel_fft ? std::min<std::size_t>(blocks, std::max(1U, std::thread::hardware_concurrency())) : std::size_t{1};
            if (workers <= 1)
            {
                fn(0, count);
                return;
            }
            std::atomic<std::size_t> next{0};
            std::vector<std::future<void>> futures;
            futures.reserve(workers);
            for (std::size_t worker = 0; worker < workers; ++worker)
            {
                futures.push_back(std::async(std::launch::async,
                                             [&]
                                             {
                                                 for (;;)
                                                 {
                                                     const auto block = next.fetch_add(1, std::memory_order_relaxed);
                                                     if (block >= blocks)
                                                         break;
                                                     const auto begin = block * grain;
                                                     fn(begin, begin + std::min(grain, count - begin));
                                                 }
                                             }));
            }
            for (auto &future : futures)
                future.get();
        }

        struct TaylorFactorials
        {
            std::vector<FieldElement> factorial;
            std::vector<FieldElement> inverse_factorial;
        };

        struct TaylorShiftPrecompute
        {
            std::size_t convolution_domain_size = 0;
            std::size_t wrapped_tail_length = 0;
            std::size_t tail_begin = 0;
            std::shared_ptr<const TaylorFactorials> factorials;
            std::vector<FieldElement> wrapped_kernel_tail;
            std::vector<FieldElement> shifted_inverse_factorial_ntt;
        };

        struct TaylorShiftCache
        {
            std::mutex mutex;
            std::unordered_map<std::size_t, std::shared_ptr<const TaylorFactorials>> factorials;
            std::unordered_map<std::string, std::shared_ptr<const TaylorShiftPrecompute>> kernels;
            std::unordered_map<std::string, std::uint64_t> last_use;
            std::uint64_t clock = 0;
        };

        TaylorShiftCache &taylor_shift_cache()
        {
            static TaylorShiftCache cache;
            return cache;
        }

        std::size_t taylor_cache_bytes(const TaylorShiftCache &cache)
        {
            std::size_t bytes = 0;
            const auto add = [&](std::size_t count)
            {
                if (count > (std::numeric_limits<std::size_t>::max() - bytes) / sizeof(FieldElement))
                    bytes = std::numeric_limits<std::size_t>::max();
                else
                    bytes += count * sizeof(FieldElement);
            };
            for (const auto &entry : cache.factorials)
            {
                add(entry.second->factorial.capacity());
                add(entry.second->inverse_factorial.capacity());
            }
            for (const auto &entry : cache.kernels)
            {
                add(entry.second->shifted_inverse_factorial_ntt.capacity());
                add(entry.second->wrapped_kernel_tail.capacity());
            }
            return bytes;
        }

        std::size_t taylor_cache_limit()
        {
            const auto memory = util::read_effective_memory_snapshot();
            constexpr std::size_t ceiling = std::size_t{8} << 30;
            if (memory.effective_total_known)
                return std::min(ceiling, memory.effective_total_bytes / 16);
            return std::size_t{512} << 20;
        }

        void trim_taylor_cache(TaylorShiftCache &cache, std::size_t limit, std::size_t preserve_factor_count)
        {
            const auto prune_factorials = [&]
            {
                for (auto it = cache.factorials.begin(); it != cache.factorials.end();)
                {
                    if (it->first != preserve_factor_count && it->second.use_count() == 1)
                        it = cache.factorials.erase(it);
                    else
                        ++it;
                }
            };
            prune_factorials();
            while (taylor_cache_bytes(cache) > limit)
            {
                auto victim = cache.kernels.end();
                auto oldest = std::numeric_limits<std::uint64_t>::max();
                for (auto it = cache.kernels.begin(); it != cache.kernels.end(); ++it)
                {
                    if (it->second.use_count() != 1)
                        continue;
                    const auto age = cache.last_use.at(it->first);
                    if (victim == cache.kernels.end() || age < oldest)
                    {
                        victim = it;
                        oldest = age;
                    }
                }
                if (victim == cache.kernels.end())
                    break;
                cache.last_use.erase(victim->first);
                cache.kernels.erase(victim);
                prune_factorials();
            }
        }

        void touch_taylor_cache(TaylorShiftCache &cache, const std::string &key)
        {
            if (++cache.clock == 0)
            {
                cache.clock = 1;
                for (auto &entry : cache.last_use)
                    entry.second = 0;
            }
            cache.last_use[key] = cache.clock;
        }

        std::shared_ptr<const TaylorFactorials> make_taylor_factorials(std::size_t count)
        {
            auto out = std::make_shared<TaylorFactorials>();
            out->factorial.assign(count, FieldElement::one());
            out->inverse_factorial.assign(count, FieldElement::one());
            constexpr std::size_t grain = std::size_t{1} << 15;
            const auto blocks = count / grain + (count % grain != 0);
            std::vector<FieldElement> products(blocks, FieldElement::one());

            // Each block first stores local prefix products.
            taylor_parallel_ranges(count,
                                   [&](std::size_t begin, std::size_t end)
                                   {
                                       for (std::size_t base = begin; base < end; base += grain)
                                       {
                                           const auto limit = base + std::min(grain, end - base);
                                           auto product = FieldElement::one();
                                           for (std::size_t i = base; i < limit; ++i)
                                           {
                                               if (i != 0)
                                                   product *= FieldElement(static_cast<std::uint64_t>(i));
                                               out->factorial[i] = product;
                                           }
                                           products[base / grain] = product;
                                       }
                                   });

            std::vector<FieldElement> prefixes(blocks, FieldElement::one());
            auto product = FieldElement::one();
            for (std::size_t block = 0; block < blocks; ++block)
            {
                prefixes[block] = product;
                product *= products[block];
            }
            std::vector<FieldElement> inverse_ends(blocks, FieldElement::one());
            auto inverse = product.inv();
            for (std::size_t block = blocks; block-- > 0;)
            {
                inverse_ends[block] = inverse;
                inverse *= products[block];
            }

            taylor_parallel_ranges(count,
                                   [&](std::size_t begin, std::size_t end)
                                   {
                                       for (std::size_t base = begin; base < end; base += grain)
                                       {
                                           const auto limit = base + std::min(grain, end - base);
                                           const auto block = base / grain;
                                           for (std::size_t i = base; i < limit; ++i)
                                               out->factorial[i] *= prefixes[block];
                                           auto inverse_value = inverse_ends[block];
                                           for (std::size_t i = limit; i-- > base;)
                                           {
                                               out->inverse_factorial[i] = inverse_value;
                                               if (i != 0)
                                                   inverse_value *= FieldElement(static_cast<std::uint64_t>(i));
                                           }
                                       }
                                   });
            return out;
        }

        std::shared_ptr<const TaylorShiftPrecompute> cached_taylor_shift_precompute(std::size_t configured_degree, const FieldElement &delta)
        {
            auto &cache = taylor_shift_cache();
            // Single flight: concurrent misses must not duplicate a large NTT.
            std::lock_guard<std::mutex> lock(cache.mutex);
            const auto key = std::to_string(configured_degree) + ":" + delta.to_string();
            if (const auto found = cache.kernels.find(key); found != cache.kernels.end())
            {
                touch_taylor_cache(cache, key);
                return found->second;
            }

            const auto count = configured_degree + 1;
            const auto full_length = configured_degree * 2 + 1;
            const auto cyclic_size = next_power_of_two(count);
            const auto wrapped = full_length > cyclic_size ? full_length - cyclic_size : 0;
            auto prepared = std::make_shared<TaylorShiftPrecompute>();
            if (wrapped <= 256)
            {
                prepared->convolution_domain_size = cyclic_size;
                prepared->wrapped_tail_length = wrapped;
            }
            else
            {
                prepared->convolution_domain_size = next_power_of_two(full_length);
            }
            const auto limit = taylor_cache_limit();
            const auto maximum = std::numeric_limits<std::size_t>::max();
            auto additional_count = prepared->convolution_domain_size;
            if (!cache.factorials.contains(count))
            {
                if (count > (maximum - additional_count) / 2)
                    throw std::runtime_error("Taylor cache size overflow");
                additional_count += 2 * count;
            }
            if (prepared->wrapped_tail_length > maximum - additional_count || additional_count + prepared->wrapped_tail_length > maximum / sizeof(FieldElement))
                throw std::runtime_error("Taylor cache byte size overflow");
            const auto additional_bytes = (additional_count + prepared->wrapped_tail_length) * sizeof(FieldElement);
            trim_taylor_cache(cache, limit > additional_bytes ? limit - additional_bytes : 0, count);
            auto found_factorials = cache.factorials.find(count);
            if (found_factorials == cache.factorials.end())
                found_factorials = cache.factorials.emplace(count, make_taylor_factorials(count)).first;
            prepared->factorials = found_factorials->second;

            auto &kernel = prepared->shifted_inverse_factorial_ntt;
            kernel.assign(prepared->convolution_domain_size, FieldElement::zero());
            taylor_parallel_ranges(count,
                                   [&](std::size_t begin, std::size_t end)
                                   {
                                       auto delta_power = delta.pow(static_cast<std::uint64_t>(begin));
                                       for (std::size_t i = begin; i < end; ++i)
                                       {
                                           kernel[i] = delta_power * prepared->factorials->inverse_factorial[i];
                                           delta_power *= delta;
                                       }
                                   });
            prepared->tail_begin = count - prepared->wrapped_tail_length;
            prepared->wrapped_kernel_tail.assign(kernel.begin() + static_cast<std::ptrdiff_t>(prepared->tail_begin),
                                                 kernel.begin() + static_cast<std::ptrdiff_t>(count));
            ntt_in_place(FieldElement::mutable_native_array_data(kernel.data()), prepared->convolution_domain_size);
            cache.kernels.emplace(key, prepared);
            touch_taylor_cache(cache, key);
            trim_taylor_cache(cache, limit, count);
            return prepared;
        }
    }

    void release_taylor_shift_cache()
    {
        auto &cache = taylor_shift_cache();
        decltype(cache.factorials) factorials;
        decltype(cache.kernels) kernels;
        {
            std::lock_guard<std::mutex> lock(cache.mutex);
            factorials.swap(cache.factorials);
            kernels.swap(cache.kernels);
            cache.last_use.clear();
            cache.clock = 0;
        }
    }

    std::vector<FieldElement> coefficient_taylor_shift(const std::vector<FieldElement> &coefficients, std::size_t configured_degree, const FieldElement &delta)
    {
        if (configured_degree == std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("configured Taylor-shift degree overflows coefficient length");
        const auto count = configured_degree + 1;
        if (coefficients.size() != count)
            throw std::runtime_error("Taylor shift requires the compiler-fixed coefficient vector length");
        if (configured_degree > (std::numeric_limits<std::size_t>::max() - 1) / 2)
            throw std::runtime_error("Taylor-shift convolution length overflow");

        // P(X + delta) == P(X) for a constant polynomial.  Avoid constructing
        // factorial/kernel cache state or invoking a one-point transform.
        if (configured_degree == 0)
        {
            return coefficients;
        }

        const auto prepared = cached_taylor_shift_precompute(configured_degree, delta);
        const auto &factorial = prepared->factorials->factorial;
        const auto &inverse_factorial = prepared->factorials->inverse_factorial;
        const auto size = prepared->convolution_domain_size;
        std::vector<FieldElement> a_rev(size, FieldElement::zero());
        taylor_parallel_ranges(count,
                               [&](std::size_t begin, std::size_t end)
                               {
                                   for (std::size_t i = begin; i < end; ++i)
                                   {
                                       const auto source = configured_degree - i;
                                       a_rev[i] = coefficients[source] * factorial[source];
                                   }
                               });

        // A false return means that the CUDA route was never selected.
        // A selected CUDA route reports every admission/execution failure.
        if (!try_cuda_taylor_convolution(FieldElement::mutable_native_array_data(a_rev.data()),
                                         FieldElement::native_array_data(prepared->shifted_inverse_factorial_ntt.data()), size))
        {
            ntt_in_place(FieldElement::mutable_native_array_data(a_rev.data()), size);
            taylor_parallel_ranges(size,
                                   [&](std::size_t begin, std::size_t end)
                                   {
                                       for (std::size_t i = begin; i < end; ++i)
                                           a_rev[i] *= prepared->shifted_inverse_factorial_ntt[i];
                                   });
            intt_in_place(FieldElement::mutable_native_array_data(a_rev.data()), size);
        }

        for (std::size_t wrapped_index = 0; wrapped_index < prepared->wrapped_tail_length; ++wrapped_index)
        {
            const auto linear_index = size + wrapped_index;
            auto wrapped = FieldElement::zero();
            for (std::size_t source = linear_index - configured_degree; source <= configured_degree; ++source)
            {
                const auto source_coefficient = configured_degree - source;
                const auto kernel_index = linear_index - source - prepared->tail_begin;
                wrapped += coefficients[source_coefficient] * factorial[source_coefficient] * prepared->wrapped_kernel_tail.at(kernel_index);
            }
            a_rev[wrapped_index] -= wrapped;
        }

        std::vector<FieldElement> shifted(count, FieldElement::zero());
        taylor_parallel_ranges(count,
                               [&](std::size_t begin, std::size_t end)
                               {
                                   for (std::size_t k = begin; k < end; ++k)
                                       shifted[k] = a_rev[configured_degree - k] * inverse_factorial[k];
                               });
        return shifted;
    }

    // 将矩阵（二维向量）按行展平为一维向量
    std::vector<FieldElement> flatten_matrix_coefficients(const std::vector<std::vector<FieldElement>> &matrix)
    {
        std::size_t count = 0;
        for (const auto &row : matrix)
        {
            if (row.size() > std::numeric_limits<std::size_t>::max() - count)
                throw std::runtime_error("flattened matrix coefficient count overflow");
            count += row.size();
        }
        std::vector<FieldElement> out;
        out.reserve(count);
        for (const auto &row : matrix)
            out.insert(out.end(), row.begin(), row.end());
        return out;
    }
}
