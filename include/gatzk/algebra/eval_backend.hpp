#pragma once

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
    // 预计算多个多项式在单位根域上的值
    class PackedEvaluationBackend
    {
    public:
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

        const std::vector<mcl::Fr>& interleaved_values_native() const
        {
            return interleaved_values_;
        }

        // 根据标签列表，解析出这些标签对应的行索引
        std::vector<std::size_t> resolve_row_indices(const std::vector<std::string>& labels) const;
        std::vector<FieldElement> values_at_direct_index(const std::vector<std::string>& labels, std::size_t index) const;
        
        // 使用普通权重向量计算加权和
        std::vector<FieldElement> evaluate_with_weights(
            const std::vector<std::string>& labels,
            const std::vector<FieldElement>& weights) const;

        // 使用原生权重向量（mcl::Fr）计算加权和
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

        // 返回指定标签子集的行索引
        const std::vector<std::size_t>& subset_row_indices(const std::vector<std::string>& labels) const;
        const std::vector<std::uint32_t>& subset_row_indices_u32(const std::vector<std::string>& labels) const;

    private:
        struct SubsetView;
        struct SubsetCacheState;

        const SubsetView& subset_for(const std::vector<std::string>& labels) const;

        std::shared_ptr<RootOfUnityDomain> domain_;
        std::vector<std::pair<std::string, const Polynomial*>> polynomials_;
        std::unordered_map<std::string, std::size_t> label_to_index_;
        std::vector<mcl::Fr> interleaved_values_;
        mutable std::shared_ptr<SubsetCacheState> subset_cache_state_;
    };

}
