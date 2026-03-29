#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <mcl/bn.hpp>

#include "gatzk/algebra/field.hpp"

namespace gatzk::algebra {

struct alignas(32) PackedFieldElement {
    std::uint64_t limbs[4] = {0, 0, 0, 0};
};

class PackedFieldBuffer {
  public:
    PackedFieldBuffer() = default;
    explicit PackedFieldBuffer(std::size_t count)
        : elements_(count) {}

    bool empty() const { return elements_.empty(); }
    std::size_t size() const { return elements_.size(); }
    void resize(std::size_t count) { elements_.resize(count); }

    const PackedFieldElement* data() const { return elements_.data(); }
    PackedFieldElement* data() { return elements_.data(); }

    const PackedFieldElement& operator[](std::size_t index) const { return elements_[index]; }
    PackedFieldElement& operator[](std::size_t index) { return elements_[index]; }

    const std::vector<PackedFieldElement>& elements() const { return elements_; }
    std::vector<PackedFieldElement>& elements() { return elements_; }

  private:
    std::vector<PackedFieldElement> elements_;
};

PackedFieldBuffer pack_native_field_elements(const std::vector<mcl::Fr>& values);
PackedFieldBuffer pack_field_elements(const std::vector<FieldElement>& values);

void pack_native_field_elements_into(
    const std::vector<mcl::Fr>& values,
    PackedFieldBuffer* out);
void pack_field_elements_into(
    const std::vector<FieldElement>& values,
    PackedFieldBuffer* out);

mcl::Fr unpack_native_field_element(const PackedFieldElement& value);
FieldElement unpack_field_element(const PackedFieldElement& value);
std::vector<FieldElement> unpack_field_elements(const PackedFieldBuffer& values);

}  // namespace gatzk::algebra
