#include "gatzk/algebra/field_io.hpp"

#include <algorithm>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace gatzk::algebra {

void write_field_elements_binary(
    std::ostream& stream,
    std::span<const FieldElement> values) {
    constexpr std::size_t kFieldBytes = 32;
    constexpr std::size_t kFieldsPerBuffer = (1ULL << 20) / kFieldBytes;
    std::vector<std::uint8_t> buffer(kFieldsPerBuffer * kFieldBytes);
    for (std::size_t offset = 0; offset < values.size();
         offset += kFieldsPerBuffer) {
        const auto count = std::min(
            kFieldsPerBuffer, values.size() - offset);
        for (std::size_t i = 0; i < count; ++i) {
            values[offset + i].write_little_endian(
                buffer.data() + i * kFieldBytes,
                kFieldBytes);
        }
        stream.write(
            reinterpret_cast<const char*>(buffer.data()),
            static_cast<std::streamsize>(count * kFieldBytes));
        if (!stream) {
            throw std::runtime_error(
                "failed to write field-element spill data");
        }
    }
}

std::vector<FieldElement> read_field_elements_binary(
    std::istream& stream,
    std::size_t value_count) {
    constexpr std::size_t kFieldBytes = 32;
    constexpr std::size_t kFieldsPerBuffer = (1ULL << 20) / kFieldBytes;
    std::vector<std::uint8_t> buffer(kFieldsPerBuffer * kFieldBytes);
    std::vector<FieldElement> values(value_count, FieldElement::zero());
    for (std::size_t offset = 0; offset < value_count;
         offset += kFieldsPerBuffer) {
        const auto count = std::min(
            kFieldsPerBuffer, value_count - offset);
        stream.read(
            reinterpret_cast<char*>(buffer.data()),
            static_cast<std::streamsize>(count * kFieldBytes));
        if (!stream) {
            throw std::runtime_error(
                "failed to read field-element spill data");
        }
        for (std::size_t i = 0; i < count; ++i) {
            values[offset + i] = FieldElement::from_little_endian_mod(
                buffer.data() + i * kFieldBytes,
                kFieldBytes);
        }
    }
    return values;
}

}  // namespace gatzk::algebra
