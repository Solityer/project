#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mcl/bn.hpp>

namespace gatzk::algebra {

inline void ensure_mcl_field_ready() {
    static std::once_flag once;
    std::call_once(once, []() {
        mcl::bn::initPairing(mcl::BLS12_381);
    });
}

class FieldElement {
  public:
    FieldElement() {
        ensure_mcl_field_ready();
        value_.clear();
    }

    explicit FieldElement(std::uint64_t raw) {
        ensure_mcl_field_ready();
        if (raw <= static_cast<std::uint64_t>(std::numeric_limits<long long>::max())) {
            value_ = static_cast<long long>(raw);
            return;
        }
        value_.setStr(std::to_string(raw), 10);
    }

    static FieldElement zero() { return FieldElement(0); }
    static FieldElement one() { return FieldElement(1); }
    static FieldElement from_native(const mcl::Fr& value) {
        FieldElement out;
        out.value_ = value;
        return out;
    }

    static FieldElement from_signed(std::int64_t value) {
        FieldElement out;
        out.value_ = static_cast<long long>(value);
        return out;
    }

    static FieldElement from_little_endian_mod(const std::uint8_t* bytes, std::size_t size) {
        FieldElement out;
        out.value_.setLittleEndianMod(bytes, size);
        return out;
    }

    static FieldElement root_of_unity(std::size_t size) {
        if (size == 0 || (size & (size - 1U)) != 0U) {
            throw std::runtime_error("domain size must be a non-zero power of two");
        }
        ensure_mcl_field_ready();
        const mcl::Vint exponent = (mcl::Fr::getOp().mp - 1) / static_cast<int>(size);
        const mcl::Fr generator = 5;
        FieldElement out;
        mcl::Fr::pow(out.value_, generator, exponent);
        return out;
    }

    std::uint64_t value() const {
        // Trace construction uses many field elements as small range-table or
        // node-index values. Reading them through mcl's native integer path
        // avoids repeated decimal string round-trips without changing any field
        // semantics.
        return value_.getUint64();
    }
    bool is_zero() const { return value_.isZero(); }

    FieldElement operator+(const FieldElement& other) const {
        FieldElement out;
        mcl::Fr::add(out.value_, value_, other.value_);
        return out;
    }

    FieldElement operator-(const FieldElement& other) const {
        FieldElement out;
        mcl::Fr::sub(out.value_, value_, other.value_);
        return out;
    }

    FieldElement operator*(const FieldElement& other) const {
        FieldElement out;
        mcl::Fr::mul(out.value_, value_, other.value_);
        return out;
    }

    FieldElement operator/(const FieldElement& other) const { return *this * other.inv(); }

    FieldElement& operator+=(const FieldElement& other) {
        mcl::Fr::add(value_, value_, other.value_);
        return *this;
    }

    FieldElement& operator-=(const FieldElement& other) {
        mcl::Fr::sub(value_, value_, other.value_);
        return *this;
    }

    FieldElement& operator*=(const FieldElement& other) {
        mcl::Fr::mul(value_, value_, other.value_);
        return *this;
    }

    bool operator==(const FieldElement& other) const { return value_ == other.value_; }

    FieldElement pow(std::uint64_t exponent) const {
        FieldElement out;
        mcl::Fr::pow(out.value_, value_, exponent);
        return out;
    }

    FieldElement inv() const {
        if (is_zero()) {
            throw std::runtime_error("attempted to invert zero in FieldElement");
        }
        FieldElement out;
        mcl::Fr::inv(out.value_, value_);
        return out;
    }

    std::string to_string() const { return value_.getStr(10); }
    const mcl::Fr& native() const { return value_; }

    std::size_t write_little_endian(std::uint8_t* bytes, std::size_t size) const {
        if (bytes == nullptr) {
            throw std::runtime_error("FieldElement::write_little_endian received null output buffer");
        }
        std::memset(bytes, 0, size);
        return value_.getLittleEndian(bytes, size);
    }

  private:
    mcl::Fr value_;
};

inline FieldElement operator-(const FieldElement& value) {
    return FieldElement::zero() - value;
}

inline std::ostream& operator<<(std::ostream& stream, const FieldElement& value) {
    stream << value.to_string();
    return stream;
}

inline std::vector<FieldElement> batch_invert(const std::vector<FieldElement>& values) {
    std::vector<FieldElement> out(values.size(), FieldElement::zero());
    if (values.empty()) {
        return out;
    }

    std::vector<mcl::Fr> prefixes(values.size());
    mcl::Fr accumulator = 1;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i].is_zero()) {
            throw std::runtime_error("batch_invert encountered zero");
        }
        prefixes[i] = accumulator;
        mcl::Fr::mul(accumulator, accumulator, values[i].native());
    }

    mcl::Fr suffix_inverse;
    mcl::Fr::inv(suffix_inverse, accumulator);
    for (std::size_t i = values.size(); i-- > 0;) {
        mcl::Fr inverse;
        mcl::Fr::mul(inverse, suffix_inverse, prefixes[i]);
        out[i] = FieldElement::from_native(inverse);
        mcl::Fr::mul(suffix_inverse, suffix_inverse, values[i].native());
    }
    return out;
}

inline std::size_t next_power_of_two(std::size_t value) {
    std::size_t out = 1;
    while (out < value) {
        out <<= 1U;
    }
    return out;
}

}  // namespace gatzk::algebra
