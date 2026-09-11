#ifndef ZKDL_G1_WIRE_CUH
#define ZKDL_G1_WIRE_CUH

/*
 * Host-side canonical BLS12-381 G1 codec used by the baseline proof wire.
 *
 * The restored CUDA implementation stores Jacobian coordinates in Montgomery
 * form.  A proof wire must not expose that implementation detail: points are
 * encoded as the IETF 48-byte compressed affine form and are checked for
 * canonical field representation, curve membership, and the prime-order
 * subgroup before being admitted.  This header deliberately uses only the
 * local CPU arithmetic below; it has no dependency on a third-party runtime.
 */

#include "g1-tensor.cuh"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace zkdl_g1_wire {

using byte = std::uint8_t;
using Fp = std::array<std::uint32_t, blstrs__fp__Fp_LIMBS>;
using Scalar = std::array<std::uint32_t, blstrs__scalar__Scalar_LIMBS>;

static const Fp kModulus = {{
    4294945451U, 3120496639U, 2975072255U, 514588670U,
    4138792484U, 1731252896U, 4085584575U, 1685539716U,
    1129032919U, 1260103606U, 964683418U, 436277738U
}};

// Host code must not read CUDA __constant__ objects directly.  Keep the
// canonical Montgomery representation of one local to this codec instead of
// relying on a device symbol during proof serialization.
static const Fp kMontgomeryOne = {{
    196605U, 1980301312U, 3289120770U, 3958636555U,
    1405573306U, 1598593111U, 1884444485U, 2010011731U,
    2723605613U, 1543969431U, 4202751123U, 368467651U
}};

inline blstrs__fp__Fp native_from_raw(const Fp& raw) {
    blstrs__fp__Fp out{};
    for (std::size_t i = 0; i < raw.size(); ++i) out.val[i] = raw[i];
    return out;
}

static const Scalar kScalarModulus = {{
    1U, 4294967295U, 4294859774U, 1404937218U,
    161601541U, 859428872U, 698187080U, 1944954707U
}};

inline int compare(const Fp& a, const Fp& b) {
    for (int i = static_cast<int>(a.size()) - 1; i >= 0; --i) {
        if (a[static_cast<std::size_t>(i)] < b[static_cast<std::size_t>(i)]) return -1;
        if (a[static_cast<std::size_t>(i)] > b[static_cast<std::size_t>(i)]) return 1;
    }
    return 0;
}

inline Fp sub_raw(const Fp& a, const Fp& b) {
    Fp out{};
    std::uint64_t borrow = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const std::uint64_t bi = static_cast<std::uint64_t>(b[i]) + borrow;
        out[i] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(a[i]) - bi);
        borrow = static_cast<std::uint64_t>(a[i]) < bi ? 1U : 0U;
    }
    return out;
}

inline Fp add(const Fp& a, const Fp& b) {
    Fp out{};
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const std::uint64_t sum = static_cast<std::uint64_t>(a[i]) + b[i] + carry;
        out[i] = static_cast<std::uint32_t>(sum);
        carry = sum >> 32U;
    }
    if (carry != 0U || compare(out, kModulus) >= 0) out = sub_raw(out, kModulus);
    return out;
}

inline Fp sub(const Fp& a, const Fp& b) {
    if (compare(a, b) >= 0) return sub_raw(a, b);
    return sub_raw(kModulus, sub_raw(b, a));
}

inline Fp reduce(const std::array<std::uint32_t, 2U * blstrs__fp__Fp_LIMBS>& product) {
    std::array<std::uint32_t, blstrs__fp__Fp_LIMBS + 1U> remainder{};
    std::array<std::uint32_t, blstrs__fp__Fp_LIMBS + 1U> modulus{};
    for (std::size_t i = 0; i < kModulus.size(); ++i) modulus[i] = kModulus[i];
    for (int bit = static_cast<int>(product.size() * 32U) - 1; bit >= 0; --bit) {
        std::uint64_t carry =
            (product[static_cast<std::size_t>(bit / 32)] >> (bit % 32)) & 1U;
        for (std::size_t i = 0; i < remainder.size(); ++i) {
            const std::uint64_t shifted = static_cast<std::uint64_t>(remainder[i]) * 2U + carry;
            remainder[i] = static_cast<std::uint32_t>(shifted);
            carry = shifted >> 32U;
        }
        int ge = remainder.back() == 0U ? 0 : 1;
        if (ge == 0) {
            for (int i = static_cast<int>(kModulus.size()) - 1; i >= 0; --i) {
                if (remainder[static_cast<std::size_t>(i)] != modulus[static_cast<std::size_t>(i)]) {
                    ge = remainder[static_cast<std::size_t>(i)] > modulus[static_cast<std::size_t>(i)] ? 1 : -1;
                    break;
                }
            }
        }
        if (ge >= 0) {
            std::uint64_t borrow = 0;
            for (std::size_t i = 0; i < remainder.size(); ++i) {
                const std::uint64_t subtrahend = static_cast<std::uint64_t>(modulus[i]) + borrow;
                const std::uint64_t value = remainder[i];
                remainder[i] = static_cast<std::uint32_t>(value - subtrahend);
                borrow = value < subtrahend ? 1U : 0U;
            }
        }
    }
    Fp out{};
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = remainder[i];
    return out;
}

inline Fp mul(const Fp& a, const Fp& b) {
    std::array<std::uint32_t, 2U * blstrs__fp__Fp_LIMBS> product{};
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::uint64_t carry = 0;
        for (std::size_t j = 0; j < b.size(); ++j) {
            const std::size_t at = i + j;
            const unsigned __int128 value =
                static_cast<unsigned __int128>(a[i]) * b[j] + product[at] + carry;
            product[at] = static_cast<std::uint32_t>(value);
            carry = static_cast<std::uint64_t>(value >> 32U);
        }
        std::size_t at = i + b.size();
        while (carry != 0U && at < product.size()) {
            const std::uint64_t value = static_cast<std::uint64_t>(product[at]) + carry;
            product[at] = static_cast<std::uint32_t>(value);
            carry = value >> 32U;
            ++at;
        }
    }
    return reduce(product);
}

inline Fp pow(Fp base, const Fp& exponent) {
    Fp result{};
    result[0] = 1U;
    for (int bit = static_cast<int>(blstrs__fp__Fp_LIMBS * 32U) - 1; bit >= 0; --bit) {
        result = mul(result, result);
        if ((exponent[static_cast<std::size_t>(bit / 32)] >> (bit % 32)) & 1U)
            result = mul(result, base);
    }
    return result;
}

inline Fp modulus_minus_two() {
    Fp out = kModulus;
    std::uint64_t borrow = 2U;
    for (std::size_t i = 0; i < out.size() && borrow != 0U; ++i) {
        const std::uint64_t value = out[i];
        out[i] = static_cast<std::uint32_t>(value - borrow);
        borrow = value < borrow ? 1U : 0U;
    }
    return out;
}

inline Fp inv(const Fp& value) {
    Fp zero{};
    if (compare(value, zero) == 0) throw std::invalid_argument("G1 codec inverse of zero");
    return pow(value, modulus_minus_two());
}

inline Fp native_raw(const blstrs__fp__Fp& value) {
    Fp out{};
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = value.val[i];
    return out;
}

inline Fp mont_factor() {
    Fp out{};
    out[0] = 1U;
    for (std::size_t i = 0; i < kModulus.size() * 32U; ++i) out = add(out, out);
    return out;
}

inline const Fp& mont_inverse() {
    static const Fp value = inv(kMontgomeryOne);
    return value;
}

inline Fp from_mont(const blstrs__fp__Fp& value) { return mul(native_raw(value), mont_inverse()); }
inline Fp to_mont(const Fp& value) { return mul(value, mont_factor()); }

inline blstrs__fp__Fp native_from_ordinary(const Fp& value) {
    blstrs__fp__Fp out{};
    const Fp montgomery = to_mont(value);
    for (std::size_t i = 0; i < montgomery.size(); ++i) out.val[i] = montgomery[i];
    return out;
}

struct Point {
    Fp x;
    Fp y;
    Fp z;

    Point() : x{}, y{}, z{} {}
    bool is_zero() const { Fp zero{}; return compare(z, zero) == 0; }
};

inline Point point_double(const Point& input) {
    if (input.is_zero()) return input;
    const Fp a = mul(input.x, input.x);
    const Fp b = mul(input.y, input.y);
    Fp c = mul(b, b);
    Fp d = add(input.x, b);
    d = mul(d, d); d = sub(sub(d, a), c); d = add(d, d);
    const Fp e = add(add(a, a), a);
    const Fp f = mul(e, e);
    Point out;
    out.z = add(input.y, input.z);
    out.z = sub(sub(mul(out.z, out.z), mul(input.y, input.y)), mul(input.z, input.z));
    out.x = sub(sub(f, d), d);
    c = add(add(c, c), add(c, c)); c = add(c, c);
    out.y = sub(mul(sub(d, out.x), e), c);
    return out;
}

inline Point point_add(const Point& left, const Point& right) {
    if (left.is_zero()) return right;
    if (right.is_zero()) return left;
    const Fp z1z1 = mul(left.z, left.z);
    const Fp z2z2 = mul(right.z, right.z);
    const Fp u1 = mul(left.x, z2z2);
    const Fp u2 = mul(right.x, z1z1);
    const Fp s1 = mul(mul(left.y, right.z), z2z2);
    const Fp s2 = mul(mul(right.y, left.z), z1z1);
    if (compare(u1, u2) == 0) {
        if (compare(s1, s2) == 0) return point_double(left);
        return Point();
    }
    const Fp h = sub(u2, u1);
    Fp i = add(h, h); i = mul(i, i);
    const Fp j = mul(h, i);
    Fp r = sub(s2, s1); r = add(r, r);
    const Fp v = mul(u1, i);
    Point out;
    out.x = sub(sub(mul(r, r), j), add(v, v));
    out.y = mul(sub(v, out.x), r);
    const Fp sj = add(mul(s1, j), mul(s1, j));
    out.y = sub(out.y, sj);
    out.z = add(left.z, right.z);
    out.z = mul(out.z, out.z);
    out.z = sub(sub(out.z, z1z1), z2z2);
    out.z = mul(out.z, h);
    return out;
}

inline Point point_mul(const Point& input, const Scalar& scalar) {
    Point result;
    Point base = input;
    int highest = static_cast<int>(scalar.size() * 32U) - 1;
    while (highest >= 0 && ((scalar[static_cast<std::size_t>(highest / 32)] >>
                             (highest % 32)) & 1U) == 0U) --highest;
    for (int bit = 0; bit <= highest; ++bit) {
        if ((scalar[static_cast<std::size_t>(bit / 32)] >> (bit % 32)) & 1U)
            result = point_add(result, base);
        base = point_double(base);
    }
    return result;
}

inline Point from_native(const G1Jacobian_t& point) {
    Point out;
    out.x = from_mont(point.x);
    out.y = from_mont(point.y);
    out.z = from_mont(point.z);
    return out;
}

inline G1Jacobian_t to_native(const Point& point) {
    G1Jacobian_t out{};
    if (point.is_zero()) {
        out.x = native_from_raw(Fp{});
        out.y = native_from_raw(kMontgomeryOne);
        out.z = native_from_raw(Fp{});
        return out;
    }
    out.x = blstrs__fp__Fp{};
    out.y = blstrs__fp__Fp{};
    out.z = native_from_raw(kMontgomeryOne);
    const Fp z_inv = inv(point.z);
    const Fp z2 = mul(z_inv, z_inv);
    const Fp z3 = mul(z2, z_inv);
    const Fp x = mul(point.x, z2);
    const Fp y = mul(point.y, z3);
    const Fp x_mont = to_mont(x);
    const Fp y_mont = to_mont(y);
    for (std::size_t i = 0; i < x_mont.size(); ++i) {
        out.x.val[i] = x_mont[i];
        out.y.val[i] = y_mont[i];
    }
    return out;
}

inline bool is_on_curve_affine(const Point& point) {
    if (point.is_zero()) return true;
    Fp z_inv = inv(point.z);
    Fp z2 = mul(z_inv, z_inv);
    Fp z3 = mul(z2, z_inv);
    const Fp x = mul(point.x, z2);
    const Fp y = mul(point.y, z3);
    const Fp lhs = mul(y, y);
    const Fp rhs = add(mul(mul(x, x), x), [] {
        Fp four{}; four[0] = 4U; return four;
    }());
    return compare(lhs, rhs) == 0;
}

inline bool is_subgroup(const Point& point) {
    return point.is_zero() || point_mul(point, kScalarModulus).is_zero();
}

inline bool validate_native(const G1Jacobian_t& native, std::string* error = nullptr) {
    if (compare(native_raw(native.x), kModulus) >= 0 ||
        compare(native_raw(native.y), kModulus) >= 0 ||
        compare(native_raw(native.z), kModulus) >= 0) {
        if (error) *error = "G1 coordinate is not a canonical field encoding";
        return false;
    }
    const Point point = from_native(native);
    if (point.is_zero()) {
        const Fp zero{};
        Fp one{};
        one[0] = 1U;
        if (compare(point.x, zero) != 0 || compare(point.y, one) != 0) {
            if (error) *error = "non-canonical G1 infinity";
            return false;
        }
        return true;
    }
    Fp one{};
    one[0] = 1U;
    if (compare(point.z, one) != 0) {
        if (error) *error = "non-canonical G1 Jacobian z (expected affine one)";
        return false;
    }
    if (!is_on_curve_affine(point)) {
        if (error) *error = "G1 point is not on curve";
        return false;
    }
    if (!is_subgroup(point)) {
        if (error) *error = "G1 point is not in the prime-order subgroup";
        return false;
    }
    return true;
}

inline void to_big_endian(const Fp& value, std::array<byte, 48U>& out) {
    for (std::size_t i = 0; i < out.size(); ++i) {
        const std::size_t bit = (out.size() - 1U - i) * 8U;
        const std::size_t limb = bit / 32U;
        const std::size_t shift = bit % 32U;
        std::uint64_t chunk = static_cast<std::uint64_t>(value[limb]) >> shift;
        if (shift > 24U && limb + 1U < value.size()) chunk |=
            static_cast<std::uint64_t>(value[limb + 1U]) << (32U - shift);
        out[i] = static_cast<byte>(chunk & 0xffU);
    }
}

inline Fp from_big_endian(const byte* bytes) {
    Fp out{};
    for (std::size_t i = 0; i < 48U; ++i) {
        const std::size_t shift = (47U - i) * 8U;
        const std::size_t limb = shift / 32U;
        const std::size_t within = shift % 32U;
        out[limb] |= static_cast<std::uint32_t>(bytes[i]) << within;
    }
    return out;
}

inline bool lexicographically_largest(const Fp& y) {
    Fp neg{};
    if (compare(y, neg) != 0) neg = sub(kModulus, y);
    return compare(y, neg) > 0;
}

inline Fp sqrt_exponent() {
    Fp out = kModulus;
    std::uint64_t carry = 1U;
    for (std::size_t i = 0; i < out.size() && carry != 0U; ++i) {
        const std::uint64_t value = static_cast<std::uint64_t>(out[i]) + carry;
        out[i] = static_cast<std::uint32_t>(value);
        carry = value >> 32U;
    }
    std::uint32_t carry_low = 0U;
    for (int i = static_cast<int>(out.size()) - 1; i >= 0; --i) {
        const std::uint64_t cur = (static_cast<std::uint64_t>(carry_low) << 32U) +
                                  out[static_cast<std::size_t>(i)];
        out[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(cur / 4U);
        carry_low = static_cast<std::uint32_t>(cur % 4U);
    }
    return out;
}

inline bool encode(const G1Jacobian_t& native, std::array<byte, 48U>& out,
                   std::string* error = nullptr) {
    if (compare(native_raw(native.x), kModulus) >= 0 ||
        compare(native_raw(native.y), kModulus) >= 0 ||
        compare(native_raw(native.z), kModulus) >= 0) {
        if (error) *error = "cannot encode a non-canonical G1 field coordinate";
        return false;
    }
    const Point projective = from_native(native);
    if (projective.is_zero()) {
        if (compare(native_raw(native.x), Fp{}) != 0 ||
            native_raw(native.y) != kMontgomeryOne) {
            if (error) *error = "cannot encode non-canonical G1 infinity";
            return false;
        }
        out.fill(0U);
        out[0] = 0xc0U;
        return true;
    }
    if (!is_on_curve_affine(projective) || !is_subgroup(projective)) {
        if (error) *error = "cannot encode invalid G1 point";
        return false;
    }
    const Fp z_inv = inv(projective.z);
    const Fp z2 = mul(z_inv, z_inv);
    const Fp z3 = mul(z2, z_inv);
    const Fp x = mul(projective.x, z2);
    const Fp y = mul(projective.y, z3);
    to_big_endian(x, out);
    out[0] = static_cast<byte>(out[0] & 0x1fU);
    out[0] = static_cast<byte>(out[0] | 0x80U |
                               (lexicographically_largest(y) ? 0x20U : 0U));
    return true;
}

inline bool decode(const std::array<byte, 48U>& encoded, G1Jacobian_t& native,
                   std::string* error = nullptr) {
    const byte flags = encoded[0];
    /* Ordinary compressed encodings legitimately use the low five flag
     * bits for the most-significant x-coordinate bits.  Only the
     * compression bit is mandatory here; infinity-specific zero payload
     * checks remain below. */
    if ((flags & 0x80U) == 0U) {
        if (error) *error = "G1 encoding is not canonical compressed form";
        return false;
    }
    const bool infinity = (flags & 0x40U) != 0U;
    const bool sort = (flags & 0x20U) != 0U;
    if (infinity) {
        if (sort || (flags & 0x1fU) != 0U) {
            if (error) *error = "G1 infinity has non-canonical flags";
            return false;
        }
        for (std::size_t i = 1; i < encoded.size(); ++i) if (encoded[i] != 0U) {
            if (error) *error = "G1 infinity payload is nonzero";
            return false;
        }
        native = G1Jacobian_t{native_from_raw(Fp{}), native_from_raw(kMontgomeryOne),
                              native_from_raw(Fp{})};
        std::array<byte, 48U> roundtrip{};
        if (!encode(native, roundtrip, error) || roundtrip != encoded) {
            if (error) *error = "G1 infinity failed canonical decode/encode roundtrip";
            return false;
        }
        return true;
    }
    std::array<byte, 48U> x_encoding = encoded;
    x_encoding[0] = static_cast<byte>(x_encoding[0] & 0x1fU);
    Fp x = from_big_endian(x_encoding.data());
    if (compare(x, kModulus) >= 0) {
        if (error) *error = "G1 x coordinate is not canonical";
        return false;
    }
    Fp rhs = mul(mul(x, x), x);
    Fp four{}; four[0] = 4U;
    rhs = add(rhs, four);
    Fp y = pow(rhs, sqrt_exponent());
    if (compare(mul(y, y), rhs) != 0) {
        if (error) *error = "G1 compressed y is not a quadratic residue";
        return false;
    }
    if (lexicographically_largest(y) != sort) y = sub(kModulus, y);
    Point point;
    point.x = x; point.y = y; point.z = [] {
        Fp one{}; one[0] = 1U; return one;
    }();
    if (!is_on_curve_affine(point) || !is_subgroup(point)) {
        if (error) *error = "G1 decoded point failed curve/subgroup validation";
        return false;
    }
    native.x = native_from_ordinary(x);
    native.y = native_from_ordinary(y);
    native.z = native_from_raw(kMontgomeryOne);
    std::array<byte, 48U> roundtrip{};
    if (!encode(native, roundtrip, error) || roundtrip != encoded) {
        if (error) *error = "G1 point failed canonical decode/encode roundtrip";
        return false;
    }
    return true;
}

}  // namespace zkdl_g1_wire

#endif
