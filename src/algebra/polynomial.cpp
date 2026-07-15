#include "gatzk/algebra/polynomial.hpp"

#include <algorithm>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::algebra
{
    // 本文件实现工作域、多项式基变换与重心插值求值，
    // 是 trace / quotient / domain opening / verifier 共用的基础代数层。
    // 这里允许按 CPU / CUDA 路由切换执行策略，但不改变多项式本身的 formal 含义。

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
    // 当 CUDA field hotspots 可用且域足够大时，使用 GPU 加速。
    std::vector<mcl::Fr> RootOfUnityDomain::barycentric_weights_native(const FieldElement& x) const
    {
        const auto& route2 = util::route2_options();

        const bool try_gpu = should_use_cuda_barycentric_weights(size, points_precomputed);
        if (try_gpu)
        {
            const auto zero_eval = zero_polynomial_eval(x);
            std::vector<mcl::Fr> weights(size);
            const auto* ou = omega.native().getUnit();
            const auto* xu = x.native().getUnit();
            const auto* zu = zero_eval.native().getUnit();
            const auto* nu = inv_size.native().getUnit();
            uint64_t omega_limbs[4] = {ou[0], ou[1], ou[2], ou[3]};
            uint64_t x_limbs[4] = {xu[0], xu[1], xu[2], xu[3]};
            uint64_t zero_eval_limbs[4] = {zu[0], zu[1], zu[2], zu[3]};
            uint64_t inv_n_limbs[4] = {nu[0], nu[1], nu[2], nu[3]};
            const bool gpu_ok = barycentric_weights_gpu(
                omega_limbs, x_limbs, zero_eval_limbs, inv_n_limbs,
                reinterpret_cast<uint64_t*>(weights.data()),
                size);
            if (gpu_ok)
            {
                return weights;
            }
            throw std::runtime_error("CUDA barycentric weight kernel failed; refusing silent CPU fallback");
        }

        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const bool use_parallel =
            route2.parallel_fft && size >= (1ULL << 20) && cpu_count > 1;

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

        const std::size_t task_count =
            std::min<std::size_t>(cpu_count, (size + (1ULL << 20) - 1) / (1ULL << 20));
        const std::size_t chunk_size = (size + task_count - 1) / task_count;

        std::vector<mcl::Fr> denominators(size);
        std::vector<mcl::Fr> prefix_within(size);   // 段内前缀积（不含 i 自身）
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

        // --- 阶段二：串行 O(task_count) ---
        // 先做跨段前缀积，再只做一次全局求逆，最后恢复每段自己的外积逆元。
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
                        // 非预计算域需要从段末尾开始倒推 `omega^i`。
                        FieldElement omega_power = points_precomputed
                            ? FieldElement::zero()
                            : omega.pow(static_cast<std::uint64_t>(end - 1));

                        mcl::Fr running_suffix;
                        running_suffix = mcl::Fr(1);  // 表示段内后缀积 `∏_{j=i+1}^{end-1} d[j]`

                        for (std::size_t i = end; i-- > begin;)
                        {
                            // 还原当前分母的逆元。
                            mcl::Fr inv_d;
                            mcl::Fr::mul(inv_d, chunk_outer_inv[task], prefix_within[i]);
                            mcl::Fr::mul(inv_d, inv_d, running_suffix);

                            // 写出与 formal 主线一致的 barycentric 权重。
                            const auto scaled_pt = points_precomputed
                                ? points_scaled_by_inv_size[i].native()
                                : (omega_power * inv_size).native();
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

    Polynomial Polynomial::from_shared_evaluations(
        const std::string& name,
        std::shared_ptr<const std::vector<FieldElement>> evaluations,
        const std::shared_ptr<RootOfUnityDomain>& domain)
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

    const std::vector<FieldElement>& Polynomial::values() const
    {
        return shared_data != nullptr ? *shared_data : data;
    }

    std::size_t Polynomial::size() const
    {
        return shared_data != nullptr ? shared_data->size() : data.size();
    }

    const FieldElement& Polynomial::value_at(std::size_t index) const
    {
        return shared_data != nullptr ? shared_data->at(index) : data.at(index);
    }

    // 计算多项式在 x 处的值
    // 根据基选择不同算法：系数形式用霍纳法；求值形式用重心插值
    FieldElement Polynomial::semantic_evaluate(const FieldElement& x) const
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

        const auto& values_ref = values();

        // 如果 x 恰好是域上的一个已知点，直接返回对应值
        if (domain->points_precomputed)
        {
            for (std::size_t i = 0; i < domain->points.size(); ++i)
            {
                if (domain->points[i] == x)
                {
                    return i < values_ref.size() ? values_ref[i] : FieldElement::zero();
                }
            }
        }
        else if (const auto shift = domain->rotation_shift(FieldElement::one(), x); shift.has_value())
        {
            if (*shift < values_ref.size())
            {
                return values_ref[*shift];
            }
        }

         // 一般情况：使用重心插值公式
        const auto& route2 = util::route2_options();

        const FieldElement zero_eval = domain->zero_polynomial_eval(x);
        FieldElement sum = FieldElement::zero();
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        if (domain->points_precomputed && route2.parallel_fft && values_ref.size() >= 1024 && cpu_count > 1)
        {
            const auto task_count = std::min<std::size_t>(cpu_count, values_ref.size() / 512);
            const auto chunk_size = task_count > 1 ? (values_ref.size() + task_count - 1) / task_count : values_ref.size();
            if (task_count > 1)
            {
                std::vector<std::future<FieldElement>> futures;
                futures.reserve(task_count);
                for (std::size_t task = 0; task < task_count; ++task)
                {
                    const auto begin = task * chunk_size;
                    const auto end = std::min(values_ref.size(), begin + chunk_size);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(
                        std::launch::async,
                        [&, begin, end]() {
                            return barycentric_sum_range(values_ref, domain->points, x, begin, end);
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
        for (std::size_t i = 0; i < values_ref.size(); ++i)
        {
            const auto current_point = domain->points_precomputed ? domain->points[i] : point;
            sum += values_ref[i] * current_point / (x - current_point);
            if (!domain->points_precomputed)
            {
                point *= domain->omega;
            }
        }
        return zero_eval * domain->inv_size * sum;
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

    FieldElement Polynomial::evaluate(const FieldElement& x) const
    {
        const auto semantic = semantic_evaluate(x);
        if (!is_zero_domain_blinded())
        {
            return semantic;
        }
        return semantic + domain->zero_polynomial_eval(x) * horner(zero_domain_blinder, x);
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

    std::vector<FieldElement> materialize_blinded_coefficients(
        const Polynomial& polynomial,
        std::size_t configured_degree)
    {
        if (configured_degree == std::numeric_limits<std::size_t>::max())
        {
            throw std::runtime_error("configured polynomial degree overflows coefficient length");
        }
        const auto configured_length = configured_degree + 1;
        std::vector<FieldElement> out(configured_length, FieldElement::zero());
        if (polynomial.basis == PolynomialBasis::Coefficient)
        {
            if (polynomial.data.size() > configured_length)
            {
                throw std::runtime_error(
                    "coefficient polynomial exceeds configured shift degree: " + polynomial.name);
            }
            std::copy(polynomial.data.begin(), polynomial.data.end(), out.begin());
            if (!polynomial.zero_domain_blinder.empty())
            {
                throw std::runtime_error(
                    "coefficient polynomial cannot carry a domain-vanishing blinder: "
                    + polynomial.name);
            }
            return out;
        }
        if (polynomial.domain == nullptr)
        {
            throw std::runtime_error(
                "evaluation polynomial is missing its domain during coefficient materialization: "
                + polynomial.name);
        }
        if (polynomial.values().size() > polynomial.domain->size
            || polynomial.domain->size > configured_length)
        {
            throw std::runtime_error(
                "evaluation polynomial exceeds configured shift degree: " + polynomial.name);
        }

        std::vector<FieldElement> semantic(polynomial.domain->size, FieldElement::zero());
        std::copy(polynomial.values().begin(), polynomial.values().end(), semantic.begin());
        ntt_plan(semantic.size())->intt(
            FieldElement::mutable_native_array_data(semantic.data()));
        std::copy(semantic.begin(), semantic.end(), out.begin());

        const auto n = polynomial.domain->size;
        if (n + polynomial.zero_domain_blinder.size() > configured_length)
        {
            throw std::runtime_error(
                "domain-vanishing blinder exceeds configured shift degree: "
                + polynomial.name);
        }
        // Z_D(X)R(X)=(X^n-1)R(X). The blinder must be added explicitly;
        // domain evaluations alone do not contain this material.
        for (std::size_t i = 0; i < polynomial.zero_domain_blinder.size(); ++i)
        {
            out[i] -= polynomial.zero_domain_blinder[i];
            out[n + i] += polynomial.zero_domain_blinder[i];
        }
        return out;
    }

    std::vector<FieldElement> coefficient_taylor_shift(
        const std::vector<FieldElement>& coefficients,
        std::size_t configured_degree,
        const FieldElement& delta)
    {
        if (configured_degree == std::numeric_limits<std::size_t>::max())
        {
            throw std::runtime_error("configured Taylor-shift degree overflows coefficient length");
        }
        const auto configured_length = configured_degree + 1;
        if (coefficients.size() != configured_length)
        {
            throw std::runtime_error(
                "Taylor shift requires the compiler-fixed coefficient vector length");
        }
        if (configured_degree > (std::numeric_limits<std::size_t>::max() - 1) / 2)
        {
            throw std::runtime_error("Taylor-shift convolution length overflow");
        }
        const auto convolution_length = configured_degree * 2 + 1;
        const auto n_shift = next_power_of_two(convolution_length);

        std::vector<FieldElement> factorial(configured_length, FieldElement::one());
        for (std::size_t i = 1; i < configured_length; ++i)
        {
            factorial[i] = factorial[i - 1] * FieldElement(static_cast<std::uint64_t>(i));
        }
        std::vector<FieldElement> inverse_factorial(configured_length, FieldElement::one());
        inverse_factorial.back() = factorial.back().inv();
        for (std::size_t i = configured_degree; i > 0; --i)
        {
            inverse_factorial[i - 1]
                = inverse_factorial[i] * FieldElement(static_cast<std::uint64_t>(i));
        }

        std::vector<FieldElement> a_rev(n_shift, FieldElement::zero());
        std::vector<FieldElement> b(n_shift, FieldElement::zero());
        auto delta_power = FieldElement::one();
        for (std::size_t i = 0; i < configured_length; ++i)
        {
            const auto source_index = configured_degree - i;
            a_rev[i] = coefficients[source_index] * factorial[source_index];
            b[i] = delta_power * inverse_factorial[i];
            delta_power *= delta;
        }

        const auto plan = ntt_plan(n_shift);
        plan->ntt(FieldElement::mutable_native_array_data(a_rev.data()));
        plan->ntt(FieldElement::mutable_native_array_data(b.data()));
        for (std::size_t i = 0; i < n_shift; ++i) a_rev[i] *= b[i];
        plan->intt(FieldElement::mutable_native_array_data(a_rev.data()));

        std::vector<FieldElement> shifted(configured_length, FieldElement::zero());
        for (std::size_t k = 0; k < configured_length; ++k)
        {
            shifted[k] = a_rev[configured_degree - k] * inverse_factorial[k];
        }
        return shifted;
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
