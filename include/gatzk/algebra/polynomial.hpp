#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::algebra
{
    // 单位根域结构体，包含FFT的旋转因子和域上的点
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
        static std::shared_ptr<RootOfUnityDomain> create(const std::string& name, std::size_t size);

        // 返回第 index 个点（即 omega^index）
        FieldElement point_at(std::size_t index) const;

        FieldElement point_scaled_by_inv_size_at(std::size_t index) const;

        // 计算零点多项式 Z(x) = x^size - 1 在x处的值
        FieldElement zero_polynomial_eval(const FieldElement& x) const;

        // 计算第index个拉格朗日基函数L_index(x)在x处的值
        FieldElement lagrange_basis_eval(std::size_t index, const FieldElement& x) const;

        std::vector<mcl::Fr> barycentric_weights_native(const FieldElement& x) const;
        std::vector<FieldElement> barycentric_weights(const FieldElement& x) const;
        std::optional<std::size_t> rotation_shift(const FieldElement& from, const FieldElement& to) const;
    };

    enum class PolynomialBasis
    {
        Coefficient, // 系数形式
        Evaluation // 求值形式
    };

    struct Polynomial
    {
        std::string name;

        // 当前使用的基（系数或求值）
        PolynomialBasis basis = PolynomialBasis::Coefficient;

        // 多项式数据：系数形式时是系数列表，求值形式时是域上的点值列表
        std::vector<FieldElement> data;

        // 如果基是求值形式，记录使用的单位根域
        std::shared_ptr<RootOfUnityDomain> domain;

        static Polynomial from_coefficients(const std::string& name, std::vector<FieldElement> coefficients);
        static Polynomial from_evaluations(
            const std::string& name,
            std::vector<FieldElement> evaluations,
            const std::shared_ptr<RootOfUnityDomain>& domain);

        // 计算多项式在 x 处的值（自动根据当前基选择算法：系数形式用霍纳法，求值形式用拉格朗日插值）
        FieldElement evaluate(const FieldElement& x) const;
    };

    FieldElement horner(const std::vector<FieldElement>& coefficients, const FieldElement& x);
    FieldElement interpolate_at(
        const std::vector<FieldElement>& points,
        const std::vector<FieldElement>& values,
        const FieldElement& x);

    // 将矩阵（二维向量）展开为一维向量（按行拼接）
    std::vector<FieldElement> flatten_matrix_coefficients(const std::vector<std::vector<FieldElement>>& matrix);

}