#pragma once

#include <cstdint>
#include <mcl/bn.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gatzk/algebra/polynomial.hpp"

namespace gatzk::algebra
{
    enum class EvaluationExecutionPolicy
    {
        Automatic,
        CpuOnly,
    };

    class PackedEvaluationBackend
    {
    public:
        PackedEvaluationBackend(std::shared_ptr<RootOfUnityDomain> domain, std::vector<std::pair<std::string, const Polynomial *>> polynomials);

        const std::shared_ptr<RootOfUnityDomain> &domain() const { return domain_; }

        const std::vector<std::pair<std::string, const Polynomial *>> &polynomials() const { return polynomials_; }

        std::vector<std::size_t> resolve_row_indices(const std::vector<std::string> &labels) const;
        std::vector<FieldElement> values_at_direct_index(const std::vector<std::string> &labels, std::size_t index) const;

        std::vector<FieldElement> evaluate_with_weights(const std::vector<std::string> &labels, const std::vector<FieldElement> &weights) const;

        std::vector<FieldElement> evaluate_with_native_weights(const std::vector<std::string> &labels, const std::vector<mcl::Fr> &weights) const;

        std::vector<std::vector<FieldElement>> evaluate_with_weight_rotations(const std::vector<std::string> &labels,
                                                                              const std::vector<FieldElement> &representative_weights,
                                                                              const std::vector<std::size_t> &rotations) const;

        std::vector<std::vector<FieldElement>>
        evaluate_with_native_weight_rotations(const std::vector<std::string> &labels, const std::vector<mcl::Fr> &representative_weights,
                                              const std::vector<std::size_t> &rotations,
                                              EvaluationExecutionPolicy execution_policy = EvaluationExecutionPolicy::Automatic) const;

        const std::vector<std::size_t> &subset_row_indices(const std::vector<std::string> &labels) const;
        const std::vector<std::uint32_t> &subset_row_indices_u32(const std::vector<std::string> &labels) const;

    private:
        struct SubsetView;
        struct SubsetCacheState;

        SubsetView build_subset_view(const std::vector<std::string> &labels, bool materialize_interleaved) const;
        const SubsetView &subset_for(const std::vector<std::string> &labels) const;
        const mcl::Fr &native_value_at(std::size_t polynomial_index, std::size_t column) const;

        std::shared_ptr<RootOfUnityDomain> domain_;
        std::vector<std::pair<std::string, const Polynomial *>> polynomials_;
        std::unordered_map<std::string, std::size_t> label_to_index_;
        mutable std::shared_ptr<SubsetCacheState> subset_cache_state_;
    };
}
