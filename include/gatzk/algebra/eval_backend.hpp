#pragma once
// 本文件定义单位根域上的批量求值后端，负责把多列 evaluation 形式
// 多项式压成共享布局，服务 quotient、opening 与 verifier 复算。

#include <cstdint>
#include <mcl/bn.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gatzk/algebra/polynomial.hpp"

// 为多个多项式在单位根域上的求值提供高效计算
namespace gatzk::algebra
{
    // 将同域多项式打包为共享求值后端，避免 prove / verify 热路径
    // 对同一组列逐列重复做域点访问与权重乘法。
    class PackedEvaluationBackend
    {
    public:
        // `polynomials` 中每个标签必须唯一，并且都属于同一工作域。
        // 后端只保留当前 prove/verify 生命周期内的只读指针；值布局按实际
        // 使用的标签子集惰性物化，避免复制完整公开列注册表。
        PackedEvaluationBackend(
            std::shared_ptr<RootOfUnityDomain> domain,
            // polynomials: 多个多项式的标签-指针列表，每个多项式将在此域上求值
            std::vector<std::pair<std::string, const Polynomial*>> polynomials);

        const std::shared_ptr<RootOfUnityDomain>& domain() const
        {
            return domain_;
        }

        const std::vector<std::pair<std::string, const Polynomial*>>& polynomials() const
        {
            return polynomials_;
        }

        // 将协议层标签映射到打包布局中的行号，供后续共享扫描复用。
        std::vector<std::size_t> resolve_row_indices(const std::vector<std::string>& labels) const;
        std::vector<FieldElement> values_at_direct_index(const std::vector<std::string>& labels, std::size_t index) const;

        // 按域权重对指定标签子集做线性组合，常用于 domain opening 与 quotient 依赖求值。
        std::vector<FieldElement> evaluate_with_weights(
            const std::vector<std::string>& labels,
            const std::vector<FieldElement>& weights) const;

        // 与 `evaluate_with_weights` 等价，但直接消费原生 `mcl::Fr` 权重以减少转换成本。
        std::vector<FieldElement> evaluate_with_native_weights(
            const std::vector<std::string>& labels,
            const std::vector<mcl::Fr>& weights) const;

        std::vector<std::vector<FieldElement>> evaluate_with_weight_rotations(
            const std::vector<std::string>& labels,
            const std::vector<FieldElement>& representative_weights,
            const std::vector<std::size_t>& rotations) const;
  
        std::vector<std::vector<FieldElement>> evaluate_with_native_weight_rotations(
            const std::vector<std::string>& labels,
            const std::vector<mcl::Fr>& representative_weights,
            const std::vector<std::size_t>& rotations) const;

        // 为重复出现的标签子集建立缓存视图，减少 proving 热路径中的子集重建。
        const std::vector<std::size_t>& subset_row_indices(const std::vector<std::string>& labels) const;
        const std::vector<std::uint32_t>& subset_row_indices_u32(const std::vector<std::string>& labels) const;

    private:
        struct SubsetView;
        struct SubsetCacheState;

        const SubsetView& subset_for(const std::vector<std::string>& labels) const;
        const mcl::Fr& native_value_at(
            std::size_t polynomial_index,
            std::size_t column) const;

        std::shared_ptr<RootOfUnityDomain> domain_;
        std::vector<std::pair<std::string, const Polynomial*>> polynomials_;
        std::unordered_map<std::string, std::size_t> label_to_index_;
        mutable std::shared_ptr<SubsetCacheState> subset_cache_state_;
    };

}
