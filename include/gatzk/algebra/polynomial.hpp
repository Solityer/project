#pragma once
// 本文件定义 formal 主线使用的工作域与多项式表示，
// 负责把 trace / table / quotient 列统一到 coefficient 或 evaluation 语义上。

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::algebra
{
    // 单位根工作域。prove / verify 的每个 bundle 都围绕这些域做插值、opening 和 quotient 检查。
    struct RootOfUnityDomain
    {
        std::string name;
        // 域的大小（点的个数，必须是2的幂）
        std::size_t size = 0;
        FieldElement omega = FieldElement::one();
        FieldElement inv_size = FieldElement::one();
        // 域上的所有点: omega^0, omega^1, ..., omega^(size-1)
        std::vector<FieldElement> points;
        std::vector<FieldElement> points_scaled_by_inv_size;
        bool points_precomputed = true;

        // 创建指定名称和大小的单位根域
        static std::shared_ptr<RootOfUnityDomain> create(const std::string &name, std::size_t size);

        // 返回第 index 个点（即 omega^index）
        FieldElement point_at(std::size_t index) const;

        FieldElement point_scaled_by_inv_size_at(std::size_t index) const;

        // 计算零点多项式 Z(x) = x^size - 1 在x处的值
        FieldElement zero_polynomial_eval(const FieldElement &x) const;

        // 计算第index个拉格朗日基函数L_index(x)在x处的值
        FieldElement lagrange_basis_eval(std::size_t index, const FieldElement &x) const;

        std::vector<mcl::Fr> barycentric_weights_native(const FieldElement &x) const;
        std::vector<FieldElement> barycentric_weights(const FieldElement &x) const;
        std::optional<std::size_t> rotation_shift(const FieldElement &from, const FieldElement &to) const;
    };

    class CudaHostTransferGuard
    {
    public:
        virtual ~CudaHostTransferGuard() = default;

        virtual void wait() const = 0;
    };

    struct CudaResidentCoefficientBuffer
    {
        std::shared_ptr<void> allocation;

        const std::uint64_t *device_montgomery_limbs = nullptr;

        std::size_t capacity = 0;

        std::size_t logical_count = 0;

        std::size_t nonzero_count = 0;

        bool nonzero_count_known = true;

        int device = -1;

        std::shared_ptr<const CudaHostTransferGuard> host_transfer;
    };

    enum class PolynomialBasis
    {
        Coefficient, // 系数形式
        Evaluation   // 求值形式
    };

    struct Polynomial
    {
        std::string name;

        // 当前多项式持有的表示基。
        PolynomialBasis basis = PolynomialBasis::Coefficient;

        // 系数形式时是系数列，求值形式时是工作域上的评估值。
        std::vector<FieldElement> data;

        // 对于共享 evaluation 列，真实数据可挂在共享缓冲区上，避免复制大列。
        std::shared_ptr<const std::vector<FieldElement>> shared_data;

        // evaluation 形式必须绑定具体工作域，否则无法解释旋转与 opening 点。
        std::shared_ptr<RootOfUnityDomain> domain;

        // Private evaluation traces commit to P + Z_domain * R. `data` keeps
        // the canonical values P(omega^i); this coefficient vector stores R.
        std::vector<FieldElement> zero_domain_blinder;

        // Execution-only CUDA coefficient ownership. Keep this member after
        // host polynomial storage so its transfer guard is destroyed first.
        // This sidecar is excluded from serialization, transcript construction
        // and polynomial semantics.
        std::shared_ptr<const CudaResidentCoefficientBuffer> cuda_resident_coefficients;

        static Polynomial from_coefficients(const std::string &name, std::vector<FieldElement> coefficients);
        static Polynomial from_evaluations(const std::string &name, std::vector<FieldElement> evaluations, const std::shared_ptr<RootOfUnityDomain> &domain);
        static Polynomial from_shared_evaluations(const std::string &name, std::shared_ptr<const std::vector<FieldElement>> evaluations,
                                                  const std::shared_ptr<RootOfUnityDomain> &domain);

        const std::vector<FieldElement> &values() const;
        std::size_t size() const;
        const FieldElement &value_at(std::size_t index) const;

        void set_zero_domain_blinder(std::vector<FieldElement> coefficients);
        bool is_zero_domain_blinded() const;
        FieldElement semantic_evaluate(const FieldElement &x) const;

        // 按当前基在点 `x` 上求值，是 transcript 重放、quotient 与 verifier 复算的统一入口。
        FieldElement evaluate(const FieldElement &x) const;
    };

    FieldElement horner(const std::vector<FieldElement> &coefficients, const FieldElement &x);
    FieldElement interpolate_at(const std::vector<FieldElement> &points, const std::vector<FieldElement> &values, const FieldElement &x);

    // Materialize the exact blinded coefficient vector used by a KZG
    // commitment. The result has configured_degree + 1 entries and is never
    // trimmed according to witness-dependent high zero coefficients.
    std::vector<FieldElement> materialize_blinded_coefficients(const Polynomial &polynomial, std::size_t configured_degree);

    // Canonical coefficient Taylor shift fixed by the protocol compiler.
    // Computes F(X + delta) with the registered radix-2 convolution size while
    // preserving the public configured coefficient length.
    std::vector<FieldElement> coefficient_taylor_shift(const std::vector<FieldElement> &coefficients, std::size_t configured_degree, const FieldElement &delta);

    // Execution-only cache; live callers retain their immutable shared owners.
    void release_taylor_shift_cache();

    // 将矩阵按 formal 约定的行主序拍平成多项式系数列。
    std::vector<FieldElement> flatten_matrix_coefficients(const std::vector<std::vector<FieldElement>> &matrix);

}
