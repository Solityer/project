#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mcl/bn.hpp>

#include "gatzk/algebra/packed_field.hpp"
#include "gatzk/algebra/polynomial.hpp"

namespace gatzk::algebra {

struct DeviceFieldBufferLease {
    std::shared_ptr<void> storage;
    std::size_t count = 0;

    bool empty() const { return storage == nullptr || count == 0; }
};

struct PackedEvaluationDeviceResult {
    DeviceFieldBufferLease buffer;
    std::size_t row_count = 0;
    std::size_t point_count = 1;

    bool empty() const { return buffer.empty() || row_count == 0 || point_count == 0; }
};

class PackedEvaluationBackend {
  public:
    PackedEvaluationBackend(
        std::shared_ptr<RootOfUnityDomain> domain,
        std::vector<std::pair<std::string, const Polynomial*>> polynomials);

    const std::shared_ptr<RootOfUnityDomain>& domain() const { return domain_; }
    const std::vector<std::pair<std::string, const Polynomial*>>& polynomials() const { return polynomials_; }
    const std::vector<mcl::Fr>& packed_values_native() const { return packed_values_; }
    std::vector<std::size_t> resolve_row_indices(const std::vector<std::string>& labels) const;
    std::vector<FieldElement> values_at_direct_index(const std::vector<std::string>& labels, std::size_t index) const;
    std::vector<FieldElement> evaluate_with_weights(
        const std::vector<std::string>& labels,
        const std::vector<FieldElement>& weights) const;
    std::vector<FieldElement> evaluate_with_native_weights(
        const std::vector<std::string>& labels,
        const std::vector<mcl::Fr>& weights) const;
    std::vector<FieldElement> evaluate_with_packed_native_weights(
        const std::vector<std::string>& labels,
        const std::vector<mcl::Fr>& weights,
        const PackedFieldBuffer& packed_weights) const;
    PackedEvaluationDeviceResult evaluate_device_with_packed_native_weights(
        const std::vector<std::string>& labels,
        const std::vector<mcl::Fr>& weights,
        const PackedFieldBuffer& packed_weights) const;
    std::vector<std::vector<FieldElement>> evaluate_with_weight_rotations(
        const std::vector<std::string>& labels,
        const std::vector<FieldElement>& representative_weights,
        const std::vector<std::size_t>& rotations) const;
    std::vector<std::vector<FieldElement>> evaluate_with_native_weight_rotations(
        const std::vector<std::string>& labels,
        const std::vector<mcl::Fr>& representative_weights,
        const std::vector<std::size_t>& rotations) const;
    std::vector<std::vector<FieldElement>> evaluate_with_packed_native_weight_rotations(
        const std::vector<std::string>& labels,
        const std::vector<mcl::Fr>& representative_weights,
        const PackedFieldBuffer& representative_weights_packed,
        const std::vector<std::size_t>& rotations) const;
    PackedEvaluationDeviceResult evaluate_device_with_packed_native_weight_rotations(
        const std::vector<std::string>& labels,
        const std::vector<mcl::Fr>& representative_weights,
        const PackedFieldBuffer& representative_weights_packed,
        const std::vector<std::size_t>& rotations) const;
    std::vector<FieldElement> materialize_device_result(const PackedEvaluationDeviceResult& result) const;
    std::vector<std::vector<FieldElement>> materialize_device_rotation_result(
        const PackedEvaluationDeviceResult& result) const;
    const PackedFieldBuffer& packed_values_device_ready() const;
    const std::vector<std::size_t>& subset_row_indices(const std::vector<std::string>& labels) const;
    const std::vector<std::uint32_t>& subset_row_indices_u32(const std::vector<std::string>& labels) const;
    const PackedFieldBuffer* subset_packed_values_device_ready(const std::vector<std::string>& labels) const;

  private:
    struct SubsetView;
    struct SubsetCacheState;

    const SubsetView& subset_for(const std::vector<std::string>& labels) const;

    std::shared_ptr<RootOfUnityDomain> domain_;
    std::vector<std::pair<std::string, const Polynomial*>> polynomials_;
    std::unordered_map<std::string, std::size_t> label_to_index_;
    std::vector<mcl::Fr> packed_values_;
    mutable PackedFieldBuffer packed_values_packed_;
    mutable std::shared_ptr<SubsetCacheState> subset_cache_state_;
};

}  // namespace gatzk::algebra
