#include "gatzk/algebra/packed_field.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace gatzk::algebra {
namespace {

std::uint64_t load_u64_le(const std::uint8_t* bytes) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
    }
    return value;
}

void store_u64_le(std::uint8_t* bytes, std::uint64_t value) {
    for (std::size_t i = 0; i < 8; ++i) {
        bytes[i] = static_cast<std::uint8_t>((value >> (8U * i)) & 0xffU);
    }
}

PackedFieldElement pack_native_field_element(const mcl::Fr& value) {
    ensure_mcl_field_ready();
    std::array<std::uint8_t, 32> bytes{};
    const auto written = value.getLittleEndian(bytes.data(), bytes.size());
    if (written > bytes.size()) {
        throw std::runtime_error("mcl::Fr little-endian serialization overflow");
    }

    PackedFieldElement out;
    for (std::size_t limb = 0; limb < 4; ++limb) {
        out.limbs[limb] = load_u64_le(bytes.data() + limb * 8U);
    }
    return out;
}

std::array<std::uint8_t, 32> unpack_field_bytes(const PackedFieldElement& value) {
    std::array<std::uint8_t, 32> bytes{};
    for (std::size_t limb = 0; limb < 4; ++limb) {
        store_u64_le(bytes.data() + limb * 8U, value.limbs[limb]);
    }
    return bytes;
}

}  // namespace

PackedFieldBuffer pack_native_field_elements(const std::vector<mcl::Fr>& values) {
    PackedFieldBuffer out(values.size());
    pack_native_field_elements_into(values, &out);
    return out;
}

PackedFieldBuffer pack_field_elements(const std::vector<FieldElement>& values) {
    PackedFieldBuffer out(values.size());
    pack_field_elements_into(values, &out);
    return out;
}

void pack_native_field_elements_into(
    const std::vector<mcl::Fr>& values,
    PackedFieldBuffer* out) {
    if (out == nullptr) {
        throw std::runtime_error("pack_native_field_elements_into requires an output buffer");
    }
    out->resize(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        (*out)[i] = pack_native_field_element(values[i]);
    }
}

void pack_field_elements_into(
    const std::vector<FieldElement>& values,
    PackedFieldBuffer* out) {
    if (out == nullptr) {
        throw std::runtime_error("pack_field_elements_into requires an output buffer");
    }
    out->resize(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        (*out)[i] = pack_native_field_element(values[i].native());
    }
}

mcl::Fr unpack_native_field_element(const PackedFieldElement& value) {
    ensure_mcl_field_ready();
    const auto bytes = unpack_field_bytes(value);
    mcl::Fr out;
    out.setLittleEndianMod(bytes.data(), bytes.size());
    return out;
}

FieldElement unpack_field_element(const PackedFieldElement& value) {
    return FieldElement::from_native(unpack_native_field_element(value));
}

std::vector<FieldElement> unpack_field_elements(const PackedFieldBuffer& values) {
    std::vector<FieldElement> out;
    out.reserve(values.size());
    for (const auto& value : values.elements()) {
        out.push_back(unpack_field_element(value));
    }
    return out;
}

}  // namespace gatzk::algebra
