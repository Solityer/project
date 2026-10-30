#ifndef ZKDL_CPU_BACKEND_CUH
#define ZKDL_CPU_BACKEND_CUH

/*
 * Portable host implementation of the same BLS12-381 scalar/ G1 Jacobian
 * equations used by the native CUDA tensors.  This header intentionally has
 * no CUDA calls in its execution path: a CPU runner can use the same
 * ProofEnvelope, Transcript and EvaluationOpeningProof wire objects while
 * evaluating commitment/opening equations on the host.
 */

#include "proof.cuh"
#include "gat_adapter/hyrax_gpu_fold_api.hpp"

#if defined(__linux__)
#include <cerrno>
#include <sys/random.h>
#endif

#include <mutex>
#include <memory>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <map>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>

namespace zkdl {

/* All IPA openings are bound to the public relation descriptor before the
 * recursive transcript is started.  Callers must populate every field; an
 * opening may not acquire an index/view after proving. */
struct CpuIpaOpeningMetadata {
    std::uint32_t commitment_index = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t evaluation_commitment_index = std::numeric_limits<std::uint32_t>::max();
    std::vector<FrWire> outer_point;
    std::array<byte, 32U> descriptor_digest{};
    std::array<byte, 32U> registry_digest{};
    std::array<byte, 32U> srs_digest{};
    // Optional segmented-Hyrax binding.  Zero logical_domain denotes the
    // legacy/single-domain opening used by small tensors and non-GAT helpers.
    std::uint64_t logical_domain = 0U;
    std::uint32_t segment_index = 0U;
    std::uint64_t logical_offset = 0U;
    std::uint32_t actual_length = 0U;
    std::uint32_t padded_length = 0U;
};

inline bool cpu_ipa_metadata_valid(const CpuIpaOpeningMetadata& metadata) {
    return metadata.commitment_index != std::numeric_limits<std::uint32_t>::max() &&
           metadata.descriptor_digest != std::array<byte, 32U>{} &&
           metadata.registry_digest != std::array<byte, 32U>{} &&
           metadata.srs_digest != std::array<byte, 32U>{};
}

inline bool cpu_ipa_v3_metadata_valid(const CpuIpaOpeningMetadata& metadata) {
    return cpu_ipa_metadata_valid(metadata) &&
           metadata.evaluation_commitment_index != std::numeric_limits<std::uint32_t>::max() &&
           metadata.evaluation_commitment_index != metadata.commitment_index;
}

using CpuFp = std::array<std::uint32_t, blstrs__fp__Fp_LIMBS>;

static const CpuFp kFpModulus = {{
    4294945451U, 3120496639U, 2975072255U, 514588670U,
    4138792484U, 1731252896U, 4085584575U, 1685539716U,
    1129032919U, 1260103606U, 964683418U, 436277738U
}};

inline int cpu_fp_compare(const CpuFp& a, const CpuFp& b) {
    for (int i = blstrs__fp__Fp_LIMBS - 1; i >= 0; --i) {
        if (a[static_cast<size_t>(i)] < b[static_cast<size_t>(i)]) return -1;
        if (a[static_cast<size_t>(i)] > b[static_cast<size_t>(i)]) return 1;
    }
    return 0;
}

inline CpuFp cpu_fp_sub_raw(const CpuFp& a, const CpuFp& b) {
    CpuFp out{};
    std::uint64_t borrow = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const std::uint64_t ai = a[i];
        const std::uint64_t bi = static_cast<std::uint64_t>(b[i]) + borrow;
        out[i] = static_cast<std::uint32_t>(ai - bi);
        borrow = ai < bi ? 1U : 0U;
    }
    return out;
}

inline CpuFp cpu_fp_add(const CpuFp& a, const CpuFp& b) {
    CpuFp out{};
    std::uint64_t carry = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const std::uint64_t sum = static_cast<std::uint64_t>(a[i]) + b[i] + carry;
        out[i] = static_cast<std::uint32_t>(sum);
        carry = sum >> 32;
    }
    if (carry != 0 || cpu_fp_compare(out, kFpModulus) >= 0) out = cpu_fp_sub_raw(out, kFpModulus);
    return out;
}

inline CpuFp cpu_fp_sub(const CpuFp& a, const CpuFp& b) {
    if (cpu_fp_compare(a, b) >= 0) return cpu_fp_sub_raw(a, b);
    return cpu_fp_sub_raw(kFpModulus, cpu_fp_sub_raw(b, a));
}

inline CpuFp cpu_fp_reduce(const std::array<std::uint32_t, 2 * blstrs__fp__Fp_LIMBS>& product) {
    std::array<std::uint32_t, blstrs__fp__Fp_LIMBS + 1> remainder{};
    std::array<std::uint32_t, blstrs__fp__Fp_LIMBS + 1> modulus{};
    for (size_t i = 0; i < blstrs__fp__Fp_LIMBS; ++i) modulus[i] = kFpModulus[i];
    for (int bit = static_cast<int>(product.size() * 32) - 1; bit >= 0; --bit) {
        std::uint64_t carry = (product[static_cast<size_t>(bit / 32)] >> (bit % 32)) & 1U;
        for (size_t i = 0; i < remainder.size(); ++i) {
            const std::uint64_t shifted = static_cast<std::uint64_t>(remainder[i]) * 2U + carry;
            remainder[i] = static_cast<std::uint32_t>(shifted);
            carry = shifted >> 32;
        }
        int comparison = remainder.back() == 0 ? 0 : 1;
        if (comparison == 0) {
            for (int i = blstrs__fp__Fp_LIMBS - 1; i >= 0; --i) {
                if (remainder[static_cast<size_t>(i)] != modulus[static_cast<size_t>(i)]) {
                    comparison = remainder[static_cast<size_t>(i)] > modulus[static_cast<size_t>(i)] ? 1 : -1;
                    break;
                }
            }
        }
        if (comparison >= 0) {
            std::uint64_t borrow = 0;
            for (size_t i = 0; i < remainder.size(); ++i) {
                const std::uint64_t subtrahend = static_cast<std::uint64_t>(modulus[i]) + borrow;
                const std::uint64_t value = remainder[i];
                remainder[i] = static_cast<std::uint32_t>(value - subtrahend);
                borrow = value < subtrahend ? 1U : 0U;
            }
        }
    }
    CpuFp out{};
    for (size_t i = 0; i < out.size(); ++i) out[i] = remainder[i];
    return out;
}

inline CpuFp cpu_fp_mul(const CpuFp& a, const CpuFp& b) {
    std::array<std::uint32_t, 2 * blstrs__fp__Fp_LIMBS> product{};
    for (size_t i = 0; i < a.size(); ++i) {
        std::uint64_t carry = 0;
        for (size_t j = 0; j < b.size(); ++j) {
            const size_t at = i + j;
            const unsigned __int128 cur =
                static_cast<unsigned __int128>(a[i]) * b[j] + product[at] + carry;
            product[at] = static_cast<std::uint32_t>(cur);
            carry = static_cast<std::uint64_t>(cur >> 32);
        }
        size_t at = i + b.size();
        while (carry != 0 && at < product.size()) {
            const std::uint64_t cur = static_cast<std::uint64_t>(product[at]) + carry;
            product[at] = static_cast<std::uint32_t>(cur);
            carry = cur >> 32;
            ++at;
        }
    }
    return cpu_fp_reduce(product);
}

inline CpuFp cpu_fp_pow(CpuFp base, const CpuFp& exponent) {
    CpuFp result{};
    result[0] = 1U;
    for (int bit = blstrs__fp__Fp_LIMBS * 32 - 1; bit >= 0; --bit) {
        result = cpu_fp_mul(result, result);
        if ((exponent[static_cast<size_t>(bit / 32)] >> (bit % 32)) & 1U)
            result = cpu_fp_mul(result, base);
    }
    return result;
}

inline CpuFp cpu_fp_modulus_minus_two() {
    CpuFp out = kFpModulus;
    std::uint64_t borrow = 2U;
    for (size_t i = 0; i < out.size() && borrow != 0; ++i) {
        const std::uint64_t value = out[i];
        out[i] = static_cast<std::uint32_t>(value - borrow);
        borrow = value < borrow ? 1U : 0U;
    }
    return out;
}

inline CpuFp cpu_fp_inv(const CpuFp& value) {
    CpuFp zero{};
    if (cpu_fp_compare(value, zero) == 0) throw std::invalid_argument("inverse of zero");
    return cpu_fp_pow(value, cpu_fp_modulus_minus_two());
}

inline CpuFp cpu_fp_from_native(const Fp_t& value) {
    CpuFp out{};
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS; ++i) out[i] = value.val[i];
    return out;
}

inline Fp_t cpu_fp_to_native(const CpuFp& value) {
    Fp_t out{};
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS; ++i) out.val[i] = value[i];
    return out;
}

inline CpuFp cpu_fp_one() { CpuFp out{}; out[0] = 1U; return out; }

inline CpuFp cpu_fp_mont_factor() {
    CpuFp out = cpu_fp_one();
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS * 32U; ++i) out = cpu_fp_add(out, out);
    return out;
}

inline CpuFp cpu_fp_from_mont(const CpuFp& value) {
#if defined(ZKDL_USE_BLST_G1_CODEC)
    zkdl_blst::blst_fp mont{};
    for (size_t i = 0U; i < 6U; ++i) {
        mont.l[i] = static_cast<std::uint64_t>(value[2U * i]) |
                    (static_cast<std::uint64_t>(value[2U * i + 1U]) << 32U);
    }
    CpuFp out{};
    zkdl_blst::blst_uint32_from_fp(out.data(), &mont);
    return out;
#else
    // Native Fp coordinates are Montgomery residues (Fp_ONE is R mod p).
    // The host implementation stores ordinary residues, so multiply by
    // R^{-1} rather than treating a Montgomery coordinate as canonical.
    static const CpuFp r_inverse = cpu_fp_inv(cpu_fp_from_native(Fp_t{ {
        196605U, 1980301312U, 3289120770U, 3958636555U,
        1405573306U, 1598593111U, 1884444485U, 2010011731U,
        2723605613U, 1543969431U, 4202751123U, 368467651U
    } }));
    return cpu_fp_mul(value, r_inverse);
#endif
}

inline CpuFp cpu_fp_to_mont(const CpuFp& value) {
#if defined(ZKDL_USE_BLST_G1_CODEC)
    if (cpu_fp_compare(value, kFpModulus) >= 0) {
        throw std::invalid_argument("non-canonical G1 Fp coordinate");
    }
    zkdl_blst::blst_fp mont{};
    zkdl_blst::blst_fp_from_uint32(&mont, value.data());
    CpuFp out{};
    for (size_t i = 0U; i < 6U; ++i) {
        out[2U * i] = static_cast<std::uint32_t>(mont.l[i]);
        out[2U * i + 1U] = static_cast<std::uint32_t>(mont.l[i] >> 32U);
    }
    return out;
#else
    return cpu_fp_mul(value, cpu_fp_mont_factor());
#endif
}

struct CpuG1Point {
    CpuFp x;
    CpuFp y;
    CpuFp z;

    CpuG1Point() : x{}, y{}, z{} {}
    static CpuG1Point zero() { return CpuG1Point(); }
    bool is_zero() const { CpuFp zero{}; return cpu_fp_compare(z, zero) == 0; }
};

inline CpuG1Point cpu_g1_generator() {
    CpuG1Point out;
    out.x = cpu_fp_from_mont(cpu_fp_from_native(G1_generator_x_mont));
    out.y = cpu_fp_from_mont(cpu_fp_from_native(G1_generator_y_mont));
    out.z = cpu_fp_one();
    return out;
}

inline CpuG1Point cpu_g1_from_native(const G1Jacobian_t& point) {
    CpuG1Point out;
    out.x = cpu_fp_from_mont(cpu_fp_from_native(point.x));
    out.y = cpu_fp_from_mont(cpu_fp_from_native(point.y));
    out.z = cpu_fp_from_mont(cpu_fp_from_native(point.z));
    return out;
}

inline G1Jacobian_t cpu_g1_to_native(const CpuG1Point& point) {
    G1Jacobian_t out{};
    /* CpuG1Point::zero() is the arithmetic identity with an all-zero
     * coordinate tuple.  The proof wire uses the unique Jacobian infinity
     * (x=0,y=1,z=0), so normalize the identity before strict G1 encoding. */
    if (point.is_zero()) {
        out.x = cpu_fp_to_native(CpuFp{});
        out.y = cpu_fp_to_native(cpu_fp_mont_factor());
        out.z = cpu_fp_to_native(CpuFp{});
        return out;
    }
    out.x = cpu_fp_to_native(cpu_fp_to_mont(point.x));
    out.y = cpu_fp_to_native(cpu_fp_to_mont(point.y));
    out.z = cpu_fp_to_native(cpu_fp_to_mont(point.z));
    return out;
}

inline CpuG1Point cpu_g1_double_portable(const CpuG1Point& input) {
    if (input.is_zero()) return input;
    const CpuFp a = cpu_fp_mul(input.x, input.x);
    const CpuFp b = cpu_fp_mul(input.y, input.y);
    CpuFp c = cpu_fp_mul(b, b);
    CpuFp d = cpu_fp_add(input.x, b);
    d = cpu_fp_mul(d, d); d = cpu_fp_sub(cpu_fp_sub(d, a), c); d = cpu_fp_add(d, d);
    const CpuFp e = cpu_fp_add(cpu_fp_add(a, a), a);
    const CpuFp f = cpu_fp_mul(e, e);
    CpuG1Point out;
    out.z = cpu_fp_add(input.y, input.z); out.z = cpu_fp_sub(cpu_fp_sub(cpu_fp_mul(out.z, out.z), cpu_fp_mul(input.y, input.y)), cpu_fp_mul(input.z, input.z));
    out.x = cpu_fp_sub(cpu_fp_sub(f, d), d);
    c = cpu_fp_add(cpu_fp_add(c, c), cpu_fp_add(c, c)); c = cpu_fp_add(c, c);
    out.y = cpu_fp_sub(cpu_fp_mul(cpu_fp_sub(d, out.x), e), c);
    return out;
}

inline CpuG1Point cpu_g1_add_portable(const CpuG1Point& left, const CpuG1Point& right) {
    if (left.is_zero()) return right;
    if (right.is_zero()) return left;
    const CpuFp z1z1 = cpu_fp_mul(left.z, left.z);
    const CpuFp z2z2 = cpu_fp_mul(right.z, right.z);
    const CpuFp u1 = cpu_fp_mul(left.x, z2z2);
    const CpuFp u2 = cpu_fp_mul(right.x, z1z1);
    CpuFp s1 = cpu_fp_mul(cpu_fp_mul(left.y, right.z), z2z2);
    const CpuFp s2 = cpu_fp_mul(cpu_fp_mul(right.y, left.z), z1z1);
    if (cpu_fp_compare(u1, u2) == 0 && cpu_fp_compare(s1, s2) == 0) return cpu_g1_double_portable(left);
    const CpuFp h = cpu_fp_sub(u2, u1);
    CpuFp i = cpu_fp_add(h, h); i = cpu_fp_mul(i, i);
    const CpuFp j = cpu_fp_mul(h, i);
    CpuFp r = cpu_fp_sub(s2, s1); r = cpu_fp_add(r, r);
    const CpuFp v = cpu_fp_mul(u1, i);
    CpuG1Point out;
    out.x = cpu_fp_sub(cpu_fp_sub(cpu_fp_mul(r, r), j), cpu_fp_add(v, v));
    out.y = cpu_fp_mul(cpu_fp_sub(v, out.x), r);
    s1 = cpu_fp_add(cpu_fp_mul(s1, j), cpu_fp_mul(s1, j));
    out.y = cpu_fp_sub(out.y, s1);
    out.z = cpu_fp_add(left.z, right.z); out.z = cpu_fp_mul(out.z, out.z);
    out.z = cpu_fp_sub(cpu_fp_sub(out.z, z1z1), z2z2); out.z = cpu_fp_mul(out.z, h);
    return out;
}

#if defined(ZKDL_USE_BLST_G1_CODEC)
inline zkdl_blst::blst_fp cpu_fp_to_blst(const CpuFp& value) {
    if (cpu_fp_compare(value, kFpModulus) >= 0) {
        throw std::invalid_argument("non-canonical G1 Fp coordinate");
    }
    zkdl_blst::blst_fp out{};
    zkdl_blst::blst_fp_from_uint32(&out, value.data());
    return out;
}

inline CpuFp cpu_fp_from_blst(const zkdl_blst::blst_fp& value) {
    CpuFp out{};
    zkdl_blst::blst_uint32_from_fp(out.data(), &value);
    return out;
}

inline zkdl_blst::blst_p1 cpu_g1_to_blst(const CpuG1Point& point) {
    if (point.is_zero()) {
        const CpuFp one = cpu_fp_one();
        return zkdl_blst::blst_p1{cpu_fp_to_blst(CpuFp{}),
                                  cpu_fp_to_blst(one),
                                  cpu_fp_to_blst(CpuFp{})};
    }
    zkdl_blst::blst_p1 out{cpu_fp_to_blst(point.x),
                           cpu_fp_to_blst(point.y),
                           cpu_fp_to_blst(point.z)};
    if (!zkdl_blst::blst_p1_on_curve(&out)) {
        throw std::invalid_argument("G1 arithmetic input is not on the curve");
    }
    return out;
}

inline bool cpu_g1_point_valid(const CpuG1Point& point, bool require_subgroup = true) {
    try {
        const zkdl_blst::blst_p1 native = cpu_g1_to_blst(point);
        return zkdl_blst::blst_p1_on_curve(&native) &&
               (!require_subgroup || zkdl_blst::blst_p1_in_g1(&native));
    } catch (const std::exception&) {
        return false;
    }
}

inline CpuG1Point cpu_g1_from_blst(const zkdl_blst::blst_p1& point) {
    if (zkdl_blst::blst_p1_is_inf(&point)) return CpuG1Point::zero();
    CpuG1Point out;
    out.x = cpu_fp_from_blst(point.x);
    out.y = cpu_fp_from_blst(point.y);
    out.z = cpu_fp_from_blst(point.z);
    return out;
}
#endif

inline CpuG1Point cpu_g1_double(const CpuG1Point& input) {
#if defined(ZKDL_USE_BLST_G1_CODEC)
    const zkdl_blst::blst_p1 native = cpu_g1_to_blst(input);
    zkdl_blst::blst_p1 doubled{};
    zkdl_blst::blst_p1_double(&doubled, &native);
    return cpu_g1_from_blst(doubled);
#else
    return cpu_g1_double_portable(input);
#endif
}

inline CpuG1Point cpu_g1_add(const CpuG1Point& left, const CpuG1Point& right) {
#if defined(ZKDL_USE_BLST_G1_CODEC)
    const zkdl_blst::blst_p1 native_left = cpu_g1_to_blst(left);
    const zkdl_blst::blst_p1 native_right = cpu_g1_to_blst(right);
    zkdl_blst::blst_p1 sum{};
    zkdl_blst::blst_p1_add_or_double(&sum, &native_left, &native_right);
    return cpu_g1_from_blst(sum);
#else
    return cpu_g1_add_portable(left, right);
#endif
}

inline CpuG1Point cpu_g1_neg(const CpuG1Point& input) {
    if (input.is_zero()) return input;
    CpuG1Point out = input;
    CpuFp zero{};
    out.y = cpu_fp_sub(zero, out.y);
    return out;
}

inline CpuG1Point cpu_g1_mul_portable(const CpuG1Point& input, const FrWire& scalar) {
    CpuG1Point result = CpuG1Point::zero();
    CpuG1Point base = input;
    int highest = blstrs__scalar__Scalar_LIMBS * 32 - 1;
    while (highest >= 0 && ((scalar[static_cast<size_t>(highest / 32)] >>
                             (highest % 32)) & 1U) == 0U) --highest;
    for (int bit = 0; bit <= highest; ++bit) {
        if ((scalar[bit / 32U] >> (bit % 32U)) & 1U) result = cpu_g1_add_portable(result, base);
        base = cpu_g1_double_portable(base);
    }
    return result;
}

inline CpuG1Point cpu_g1_mul_integer_bits(const CpuG1Point& input, const FrWire& scalar) {
#if defined(ZKDL_USE_BLST_G1_CODEC)
    const zkdl_blst::blst_p1 native = cpu_g1_to_blst(input);
    zkdl_blst::blst_p1 product{};
    zkdl_blst::blst_p1_mult(
        &product, &native, reinterpret_cast<const zkdl_blst::byte*>(scalar.data()),
        sizeof(FrWire) * 8U);
    return cpu_g1_from_blst(product);
#else
    return cpu_g1_mul_portable(input, scalar);
#endif
}

inline CpuG1Point cpu_g1_mul(const CpuG1Point& input, const FrWire& scalar) {
    return cpu_g1_mul_integer_bits(input, scalar);
}

inline CpuG1Point cpu_g1_mul_canonical(const CpuG1Point& input, const FrWire& scalar) {
    if (!fr_canonical(scalar)) {
        throw std::invalid_argument("non-canonical G1 scalar");
    }
    return cpu_g1_mul_integer_bits(input, scalar);
}

inline bool cpu_g1_equal(const CpuG1Point& left, const CpuG1Point& right) {
#if defined(ZKDL_USE_BLST_G1_CODEC)
    const zkdl_blst::blst_p1 native_left = cpu_g1_to_blst(left);
    const zkdl_blst::blst_p1 native_right = cpu_g1_to_blst(right);
    return zkdl_blst::blst_p1_is_equal(&native_left, &native_right);
#else
    if (left.is_zero() || right.is_zero()) return left.is_zero() && right.is_zero();
    const CpuFp z1z1 = cpu_fp_mul(left.z, left.z);
    const CpuFp z2z2 = cpu_fp_mul(right.z, right.z);
    const CpuFp x1 = cpu_fp_mul(left.x, z2z2);
    const CpuFp x2 = cpu_fp_mul(right.x, z1z1);
    if (cpu_fp_compare(x1, x2) != 0) return false;
    const CpuFp y1 = cpu_fp_mul(left.y, cpu_fp_mul(right.z, z2z2));
    const CpuFp y2 = cpu_fp_mul(right.y, cpu_fp_mul(left.z, z1z1));
    return cpu_fp_compare(y1, y2) == 0;
#endif
}

inline bool cpu_g1_equal_portable(const CpuG1Point& left, const CpuG1Point& right) {
    if (left.is_zero() || right.is_zero()) return left.is_zero() && right.is_zero();
    const CpuFp z1z1 = cpu_fp_mul(left.z, left.z);
    const CpuFp z2z2 = cpu_fp_mul(right.z, right.z);
    const CpuFp x1 = cpu_fp_mul(left.x, z2z2);
    const CpuFp x2 = cpu_fp_mul(right.x, z1z1);
    if (cpu_fp_compare(x1, x2) != 0) return false;
    const CpuFp y1 = cpu_fp_mul(left.y, cpu_fp_mul(right.z, z2z2));
    const CpuFp y2 = cpu_fp_mul(right.y, cpu_fp_mul(left.z, z1z1));
    return cpu_fp_compare(y1, y2) == 0;
}

inline CpuG1Point cpu_g1_sum_portable(const std::vector<CpuG1Point>& points,
                                     const std::vector<FrWire>& scalars) {
    if (points.size() != scalars.size()) throw std::invalid_argument("G1 MSM shape mismatch");
    CpuG1Point result = CpuG1Point::zero();
    for (size_t i = 0; i < points.size(); ++i) {
        result = cpu_g1_add_portable(result, cpu_g1_mul_portable(points[i], scalars[i]));
    }
    return result;
}

static constexpr size_t kCpuG1MsmChunkPoints = 1U << 14U;
static constexpr size_t kCpuG1MsmProfileBytes = 16U * 1024U * 1024U;
static constexpr size_t kCpuG1MsmMaxConcurrentChunks = 20U;
static constexpr size_t kCpuG1MsmParallelThresholdChunks = 4U;

inline unsigned cpu_parallel_worker_count() {
    const unsigned hardware = std::thread::hardware_concurrency();
    if (hardware < 2U) return 0U;
    // Keep one core for the calling thread and cap the pool for predictable
    // host-memory and scheduling costs on larger machines.
    return std::min(19U, hardware - 1U);
}

class CpuBoundedThreadPool {
public:
    CpuBoundedThreadPool() {
        const unsigned count = cpu_parallel_worker_count();
        workers_.reserve(count);
        try {
            for (unsigned index = 0U; index < count; ++index)
                workers_.emplace_back([this] { worker_loop(); });
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
            }
            available_.notify_all();
            for (auto& worker : workers_)
                if (worker.joinable()) worker.join();
            workers_.clear();
            stopping_ = false;
        }
    }

    CpuBoundedThreadPool(const CpuBoundedThreadPool&) = delete;
    CpuBoundedThreadPool& operator=(const CpuBoundedThreadPool&) = delete;

    ~CpuBoundedThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        available_.notify_all();
        for (auto& worker : workers_)
            if (worker.joinable()) worker.join();
    }

    std::size_t size() const noexcept { return workers_.size(); }

    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) throw std::runtime_error("CPU proof worker pool is stopping");
            tasks_.push(std::move(task));
        }
        available_.notify_one();
    }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                available_.wait(lock, [&] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            try { task(); } catch (...) { /* task wrappers own exception reporting */ }
        }
    }

    std::mutex mutex_;
    std::condition_variable available_;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

inline CpuBoundedThreadPool& cpu_bounded_thread_pool() {
    static CpuBoundedThreadPool pool;
    return pool;
}

inline bool& cpu_in_parallel_worker() {
    static thread_local bool in_worker = false;
    return in_worker;
}

template <typename Function>
inline void cpu_parallel_for_with_threshold(std::size_t count,
                                            std::size_t grain,
                                            std::size_t parallel_threshold,
                                            Function&& function) {
    if (count < parallel_threshold || grain == 0U || cpu_in_parallel_worker()) {
        for (std::size_t index = 0U; index < count; ++index) function(index);
        return;
    }
    CpuBoundedThreadPool& pool = cpu_bounded_thread_pool();
    const std::size_t background_workers = pool.size();
    if (background_workers == 0U) {
        for (std::size_t index = 0U; index < count; ++index) function(index);
        return;
    }
    struct Group {
        std::atomic<std::size_t> next{0U};
        std::atomic<std::size_t> remaining{0U};
        std::atomic<bool> cancelled{false};
        std::mutex mutex;
        std::condition_variable completed;
        std::exception_ptr first_error;
    };
    auto group = std::make_shared<Group>();
    auto callable = std::make_shared<typename std::decay<Function>::type>(
        std::forward<Function>(function));
    group->remaining.store(background_workers + 1U, std::memory_order_relaxed);
    const auto run = [group, callable, count, grain] {
        const bool previous_worker_state = cpu_in_parallel_worker();
        cpu_in_parallel_worker() = true;
        try {
            while (!group->cancelled.load(std::memory_order_relaxed)) {
                const std::size_t begin = group->next.fetch_add(grain, std::memory_order_relaxed);
                if (begin >= count) break;
                const std::size_t end = std::min(count, begin + grain);
                for (std::size_t index = begin; index < end; ++index) {
                    if (group->cancelled.load(std::memory_order_relaxed)) break;
                    (*callable)(index);
                }
            }
        } catch (...) {
            group->cancelled.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(group->mutex);
            if (!group->first_error) group->first_error = std::current_exception();
        }
        cpu_in_parallel_worker() = previous_worker_state;
        if (group->remaining.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
            std::lock_guard<std::mutex> lock(group->mutex);
            group->completed.notify_one();
        }
    };

    std::size_t submitted = 0U;
    try {
        for (; submitted < background_workers; ++submitted) pool.submit(run);
    } catch (...) {
        group->cancelled.store(true, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(group->mutex);
            if (!group->first_error) group->first_error = std::current_exception();
        }
        const std::size_t not_submitted = background_workers - submitted;
        group->remaining.fetch_sub(not_submitted, std::memory_order_acq_rel);
    }
    run();
    {
        std::unique_lock<std::mutex> lock(group->mutex);
        group->completed.wait(lock, [&] {
            return group->remaining.load(std::memory_order_acquire) == 0U;
        });
        if (group->first_error) std::rethrow_exception(group->first_error);
    }
}

template <typename Function>
inline void cpu_parallel_for(std::size_t count, std::size_t grain,
                             Function&& function) {
    cpu_parallel_for_with_threshold(count, grain, 4096U,
                                    std::forward<Function>(function));
}

// Explicit opt-in for a small number of individually expensive independent
// tasks.  Existing callers retain cpu_parallel_for's 4096-item threshold.
template <typename Function>
inline void cpu_parallel_for_expensive_tasks(std::size_t count,
                                             std::size_t grain,
                                             std::size_t minimum_tasks,
                                             Function&& function) {
    cpu_parallel_for_with_threshold(count, grain, minimum_tasks,
                                    std::forward<Function>(function));
}

inline size_t cpu_g1_msm_workspace_bound_bytes(size_t point_count) {
#if defined(ZKDL_USE_BLST_G1_CODEC)
    if (point_count == 0U || point_count > kCpuG1MsmChunkPoints) return 0U;
    const size_t scratch_bytes = point_count < 32U ? 0U :
        zkdl_blst::blst_p1s_mult_pippenger_scratch_sizeof(point_count);
    const size_t signed_scalar_bytes =
        point_count * sizeof(std::array<zkdl_blst::byte, 8U>);
    const size_t pointer_and_affine_bytes = point_count *
        (sizeof(zkdl_blst::blst_p1_affine) +
         2U * sizeof(const zkdl_blst::byte*));
    if (signed_scalar_bytes > std::numeric_limits<size_t>::max() -
                                  pointer_and_affine_bytes)
        throw std::overflow_error("BLST G1 MSM workspace size overflow");
    const size_t point_storage_bytes =
        signed_scalar_bytes + pointer_and_affine_bytes;
    if (scratch_bytes > std::numeric_limits<size_t>::max() -
                            point_storage_bytes) {
        throw std::overflow_error("BLST G1 MSM workspace size overflow");
    }
    return scratch_bytes + point_storage_bytes;
#else
    (void)point_count;
    return 0U;
#endif
}

inline size_t cpu_g1_msm_workspace_bound_bytes() {
    return cpu_g1_msm_workspace_bound_bytes(kCpuG1MsmChunkPoints);
}

inline size_t cpu_g1_msm_or_nbits(const FrWire& scalar_or) noexcept {
    for (size_t limb = scalar_or.size(); limb > 0U; --limb) {
        const std::uint32_t word = scalar_or[limb - 1U];
        if (word != 0U) {
            return (limb - 1U) * 32U +
                   32U - static_cast<size_t>(__builtin_clz(word));
        }
    }
    return 0U;
}

inline bool cpu_g1_msm_try_decode_signed64(
    const FrWire& scalar, std::uint64_t& magnitude, bool& negative) noexcept {
    bool positive_small = true;
    for (std::size_t limb = 2U; limb < scalar.size(); ++limb)
        if (scalar[limb] != 0U) {
            positive_small = false;
            break;
        }
    if (positive_small && scalar[1U] <= 0x7fffffffU) {
        magnitude = (static_cast<std::uint64_t>(scalar[1U]) << 32U) |
                    static_cast<std::uint64_t>(scalar[0U]);
        negative = false;
        return true;
    }

    const FrWire negative_magnitude = fr_sub_raw(kScalarModulus, scalar);
    for (std::size_t limb = 2U; limb < negative_magnitude.size(); ++limb)
        if (negative_magnitude[limb] != 0U) return false;
    if (negative_magnitude[1U] > 0x80000000U ||
        (negative_magnitude[1U] == 0x80000000U &&
         negative_magnitude[0U] != 0U)) return false;
    magnitude = (static_cast<std::uint64_t>(negative_magnitude[1U]) << 32U) |
                static_cast<std::uint64_t>(negative_magnitude[0U]);
    if (magnitude == 0U || magnitude > (std::uint64_t{1U} << 63U))
        return false;
    negative = true;
    return true;
}

inline size_t cpu_g1_msm_u64_nbits(std::uint64_t value) noexcept {
    return value == 0U ? 0U :
        64U - static_cast<size_t>(__builtin_clzll(value));
}

inline CpuG1Point cpu_g1_sum(const CpuG1Point* points, const FrWire* scalars,
                             size_t count) {
    if (count != 0U && (points == nullptr || scalars == nullptr))
        throw std::invalid_argument("G1 MSM has null input");
#if defined(ZKDL_USE_BLST_G1_CODEC)
    if (count == 0U) return CpuG1Point::zero();
    const size_t workspace_points = std::min(count, kCpuG1MsmChunkPoints);
    if (cpu_g1_msm_workspace_bound_bytes(workspace_points) >
        kCpuG1MsmProfileBytes) {
        throw std::runtime_error("BLST G1 MSM exceeds its source-bound workspace profile");
    }
    CpuG1Point result = CpuG1Point::zero();
    std::unique_ptr<zkdl_blst::blst_p1_affine[]> affine(
        new zkdl_blst::blst_p1_affine[workspace_points]);
    std::unique_ptr<const zkdl_blst::byte*[]> scalar_ptrs(
        new const zkdl_blst::byte*[workspace_points]);
    std::unique_ptr<zkdl_blst::limb_t[]> scratch;
    size_t scratch_bytes = 0U;
    const zkdl_blst::blst_p1_affine* affine_sources[2] = {affine.get(), nullptr};
    for (size_t offset = 0U; offset < count;) {
        const size_t chunk_count = std::min(kCpuG1MsmChunkPoints, count - offset);
        FrWire scalar_or{};
        for (size_t i = 0U; i < chunk_count; ++i) {
            if (!fr_canonical(scalars[offset + i])) {
                throw std::invalid_argument("non-canonical G1 MSM scalar");
            }
            for (size_t limb = 0U; limb < scalar_or.size(); ++limb)
                scalar_or[limb] |= scalars[offset + i][limb];
            const zkdl_blst::blst_p1 projective = cpu_g1_to_blst(points[offset + i]);
            zkdl_blst::blst_p1_to_affine(&affine[i], &projective);
            scalar_ptrs[i] = reinterpret_cast<const zkdl_blst::byte*>(scalars[offset + i].data());
        }
        const size_t nbits = cpu_g1_msm_or_nbits(scalar_or);
        if (nbits == 0U) {
            offset += chunk_count;
            continue;
        }
        if (chunk_count >= 32U && !scratch) {
            scratch_bytes = zkdl_blst::blst_p1s_mult_pippenger_scratch_sizeof(
                workspace_points);
            if (scratch_bytes == 0U ||
                scratch_bytes % sizeof(zkdl_blst::limb_t) != 0U)
                throw std::runtime_error("BLST returned an invalid Pippenger scratch bound");
            scratch.reset(new zkdl_blst::limb_t[
                scratch_bytes / sizeof(zkdl_blst::limb_t)]);
        }
        zkdl_blst::blst_p1 partial{};
        zkdl_blst::blst_p1s_mult_pippenger(
            &partial, affine_sources, chunk_count, scalar_ptrs.get(), nbits,
            scratch.get());
        result = cpu_g1_add(result, cpu_g1_from_blst(partial));
        offset += chunk_count;
    }
    return result;
#else
    CpuG1Point result = CpuG1Point::zero();
    for (size_t i = 0; i < count; ++i) {
        if (!fr_canonical(scalars[i])) throw std::invalid_argument("non-canonical G1 MSM scalar");
        result = cpu_g1_add(result, cpu_g1_mul_portable(points[i], scalars[i]));
    }
    return result;
#endif
}

inline CpuG1Point cpu_g1_sum(const std::vector<CpuG1Point>& points,
                             const std::vector<FrWire>& scalars) {
    if (points.size() != scalars.size()) throw std::invalid_argument("G1 MSM shape mismatch");
    return cpu_g1_sum(points.data(), scalars.data(), points.size());
}

inline CpuG1Point cpu_opening_commitment_eval(const std::vector<CpuG1Point>& generators,
                                              const std::vector<FrWire>& values) {
    return cpu_g1_sum(generators, values);
}

/* Obtain a fresh scalar from the host OS entropy source.  This helper is only
 * used by the private/model hiding-commitment constructor; formal callers
 * must still provide the resulting hiding record through the authenticated
 * resource seam.  There is deliberately no deterministic or generator-based
 * fallback. */
inline FrWire cpu_fresh_csprng_scalar() {
#if !defined(__linux__)
    throw std::runtime_error("BLOCKED: Linux getrandom is required for IPA blinding");
#else
    constexpr std::size_t kMaxAttempts = 128U;
    for (std::size_t attempt = 0U; attempt < kMaxAttempts; ++attempt) {
        FrWire result{};
        std::size_t received = 0U;
        while (received < sizeof(result)) {
            const ssize_t count = ::getrandom(
                reinterpret_cast<unsigned char*>(result.data()) + received,
                sizeof(result) - received, 0);
            if (count < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("BLOCKED: getrandom failed for IPA blinding");
            }
            if (count == 0) throw std::runtime_error("BLOCKED: getrandom returned no entropy");
            received += static_cast<std::size_t>(count);
        }
        if (fr_canonical(result) && fr_compare(result, FrWire{}) != 0) return result;
    }
    throw std::runtime_error("BLOCKED: canonical nonzero IPA blinder sampling exhausted");
#endif
}

inline FrWire cpu_fr_inv(const FrWire& value) {
    FrWire zero{};
    if (fr_compare(value, zero) == 0) throw std::invalid_argument("IPA inverse of zero");
    return fr_pow(value, blstrs__scalar__Scalar_LIMBS * 32U - 1U,
                  fr_modulus_minus_two());
}

inline std::uint32_t cpu_ipa_padded_size(std::uint32_t generator_size) {
    if (generator_size == 0U || generator_size > (1U << 19))
        throw std::invalid_argument("IPA generator domain exceeds baseline cap");
    std::uint32_t padded = 1U;
    while (padded < generator_size + 1U) {
        if (padded > (1U << 19)) throw std::invalid_argument("IPA padded domain exceeds baseline cap");
        padded <<= 1U;
    }
    return padded;
}

/*
 * Strict G1 decoding is intentionally expensive: the wire codec checks the
 * curve and the prime-order subgroup for every point.  Formal proving and
 * independent verification repeatedly pass the same authenticated basis
 * through that boundary, so retain only a small positive cache.  The cache
 * key is a versioned, fixed-width host representation of every coordinate,
 * together with the complete basis shape and domain.  It is hashed for the
 * lookup index and compared byte-for-byte before a hit is accepted.  Thus a
 * digest collision cannot turn an unvalidated or different basis into a hit.
 * Entries are added only after all strict checks have succeeded; failures are
 * never cached.  Large bases are deliberately not retained to keep the
 * process-wide cache bounded in both entry count and memory.
 */
struct CpuPedersenBasisCacheEntry {
    bool occupied = false;
    std::array<byte, 32U> key_digest{};
    std::vector<byte> key;
    std::array<byte, 32U> canonical_digest{};
    std::uint64_t last_used = 0U;
};

struct CpuPedersenBasisCache {
    static constexpr std::size_t kMaxEntries = 8U;
    static constexpr std::size_t kMaxKeyBytes = 1U << 20U;
    std::mutex mutex;
    std::array<CpuPedersenBasisCacheEntry, kMaxEntries> entries{};
    std::uint64_t clock = 0U;
};

inline CpuPedersenBasisCache& cpu_pedersen_basis_cache() {
    static CpuPedersenBasisCache cache;
    return cache;
}

inline void cpu_pedersen_basis_cache_append_point(
    std::vector<byte>& key, const CpuG1Point& point) {
    for (const auto limb : point.x) append_u32(key, limb);
    for (const auto limb : point.y) append_u32(key, limb);
    for (const auto limb : point.z) append_u32(key, limb);
}

inline std::vector<byte> cpu_pedersen_basis_cache_key(
    const std::vector<CpuG1Point>& generators, const CpuG1Point& h,
    const CpuG1Point& u, std::uint32_t padded_size) {
    const std::size_t point_bytes = 3U * CpuFp{}.size() * sizeof(std::uint32_t);
    if (generators.size() >
        (CpuPedersenBasisCache::kMaxKeyBytes - 128U) / point_bytes)
        return {};
    static constexpr char kDomain[] = "zkdl.cpu-pedersen-basis-cache.v1";
    std::vector<byte> key;
    key.reserve(sizeof(kDomain) - 1U + 4U + 8U + 4U +
                (generators.size() + 2U) * point_bytes);
    append_u32(key, static_cast<std::uint32_t>(sizeof(kDomain) - 1U));
    key.insert(key.end(), kDomain, kDomain + sizeof(kDomain) - 1U);
    append_u64(key, static_cast<std::uint64_t>(generators.size()));
    append_u32(key, padded_size);
    for (const auto& generator : generators)
        cpu_pedersen_basis_cache_append_point(key, generator);
    cpu_pedersen_basis_cache_append_point(key, h);
    cpu_pedersen_basis_cache_append_point(key, u);
    return key;
}

inline bool cpu_pedersen_basis_cache_lookup(
    const std::vector<byte>& key, std::array<byte, 32U>& canonical_digest) {
    if (key.empty() || key.size() > CpuPedersenBasisCache::kMaxKeyBytes)
        return false;
    const std::array<byte, 32U> key_digest = sha256(key);
    CpuPedersenBasisCache& cache = cpu_pedersen_basis_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    for (auto& entry : cache.entries) {
        if (entry.occupied && entry.key_digest == key_digest && entry.key == key) {
            canonical_digest = entry.canonical_digest;
            entry.last_used = ++cache.clock;
            return true;
        }
    }
    return false;
}

inline void cpu_pedersen_basis_cache_store(
    const std::vector<byte>& key, const std::array<byte, 32U>& canonical_digest) {
    if (key.empty() || key.size() > CpuPedersenBasisCache::kMaxKeyBytes) return;
    const std::array<byte, 32U> key_digest = sha256(key);
    CpuPedersenBasisCache& cache = cpu_pedersen_basis_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    CpuPedersenBasisCacheEntry* selected = nullptr;
    for (auto& entry : cache.entries) {
        if (entry.occupied && entry.key_digest == key_digest && entry.key == key) {
            selected = &entry;
            break;
        }
        if (!entry.occupied && selected == nullptr) selected = &entry;
    }
    if (selected == nullptr) {
        selected = &cache.entries.front();
        for (auto& entry : cache.entries)
            if (entry.last_used < selected->last_used) selected = &entry;
    }
    selected->occupied = true;
    selected->key_digest = key_digest;
    selected->key = key;
    selected->canonical_digest = canonical_digest;
    selected->last_used = ++cache.clock;
}

inline std::array<byte, 32U> cpu_ipa_basis_digest(
    const std::vector<CpuG1Point>& generators, const CpuG1Point& h,
    const CpuG1Point& u, std::uint32_t padded_size) {
    const std::vector<byte> cache_key = cpu_pedersen_basis_cache_key(
        generators, h, u, padded_size);
    std::array<byte, 32U> cached_digest{};
    if (cpu_pedersen_basis_cache_lookup(cache_key, cached_digest)) return cached_digest;
    std::vector<byte> encoded;
    append_u32(encoded, static_cast<std::uint32_t>(generators.size()));
    append_u32(encoded, padded_size);
    for (const auto& generator : generators) append_g1(encoded, cpu_g1_to_native(generator));
    append_g1(encoded, cpu_g1_to_native(h));
    append_g1(encoded, cpu_g1_to_native(u));
    const std::array<byte, 32U> digest = sha256(encoded);
    cpu_pedersen_basis_cache_store(cache_key, digest);
    return digest;
}

#if defined(ZKDL_USE_BLST_G1_CODEC)
inline zkdl_blst::blst_fr cpu_ipa_frwire_to_montgomery_unchecked(
    const FrWire& value);
inline FrWire cpu_ipa_montgomery_to_frwire(
    const zkdl_blst::blst_fr& value);
#endif

inline FrWire cpu_ipa_inner_product(const FrWire* left, const FrWire* right,
                                    std::size_t count) {
    if (count != 0U && (left == nullptr || right == nullptr))
        throw std::invalid_argument("IPA inner-product has null input");
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(FrWire))
        throw std::length_error("IPA inner-product byte range overflows");
    const std::size_t byte_count = count * sizeof(FrWire);
    const std::uintptr_t left_address = reinterpret_cast<std::uintptr_t>(left);
    const std::uintptr_t right_address = reinterpret_cast<std::uintptr_t>(right);
    if ((count != 0U && byte_count >
             std::numeric_limits<std::uintptr_t>::max() - left_address) ||
        (count != 0U && byte_count >
             std::numeric_limits<std::uintptr_t>::max() - right_address))
        throw std::length_error("IPA inner-product pointer range overflows");
    for (std::size_t index = 0U; index < count; ++index)
        if (!fr_canonical(left[index]) || !fr_canonical(right[index]))
            throw std::invalid_argument("non-canonical IPA inner-product input");
#if defined(ZKDL_USE_BLST_G1_CODEC)
    if (count == 0U) return FrWire{};
    constexpr std::size_t kChunkElements = 1U << 12U;
    const std::size_t chunk_count = (count - 1U) / kChunkElements + 1U;
    std::vector<zkdl_blst::blst_fr> partials(chunk_count);
    cpu_parallel_for_expensive_tasks(chunk_count, 1U, 4U,
        [&](std::size_t chunk) {
            const std::size_t begin = chunk * kChunkElements;
            const std::size_t end = std::min(count, begin + kChunkElements);
            zkdl_blst::blst_fr partial{};
            for (std::size_t index = begin; index < end; ++index) {
                // The complete serial canonical scan above still validates
                // both operands before dispatch; only then may a zero pair
                // skip Montgomery conversion and multiplication.
                if (fr_compare(left[index], FrWire{}) == 0 ||
                    fr_compare(right[index], FrWire{}) == 0)
                    continue;
                const zkdl_blst::blst_fr left_montgomery =
                    cpu_ipa_frwire_to_montgomery_unchecked(left[index]);
                const zkdl_blst::blst_fr right_montgomery =
                    cpu_ipa_frwire_to_montgomery_unchecked(right[index]);
                zkdl_blst::blst_fr product{}, next_partial{};
                zkdl_blst::blst_fr_mul(
                    &product, &left_montgomery, &right_montgomery);
                zkdl_blst::blst_fr_add(&next_partial, &partial, &product);
                partial = next_partial;
            }
            partials[chunk] = partial;
        });
    zkdl_blst::blst_fr total{};
    for (const zkdl_blst::blst_fr& partial : partials) {
        zkdl_blst::blst_fr next_total{};
        zkdl_blst::blst_fr_add(&next_total, &total, &partial);
        total = next_total;
    }
    const FrWire result = cpu_ipa_montgomery_to_frwire(total);
    if (!fr_canonical(result))
        throw std::runtime_error("IPA inner-product result is not canonical");
    return result;
#else
    FrWire result{};
    for (std::size_t index = 0U; index < count; ++index)
        result = fr_add(result, fr_mul(left[index], right[index]));
    return result;
#endif
}

inline FrWire cpu_ipa_inner_product(
    const std::vector<FrWire>& left, std::size_t left_offset,
    const std::vector<FrWire>& right, std::size_t right_offset,
    std::size_t count) {
    if (left_offset > left.size() || count > left.size() - left_offset ||
        right_offset > right.size() || count > right.size() - right_offset)
        throw std::invalid_argument("IPA inner-product slice is out of range");
    const FrWire* left_data = count == 0U ? nullptr : left.data() + left_offset;
    const FrWire* right_data = count == 0U ? nullptr : right.data() + right_offset;
    return cpu_ipa_inner_product(left_data, right_data, count);
}

inline FrWire cpu_ipa_inner_product(const std::vector<FrWire>& left,
                                    const std::vector<FrWire>& right) {
    if (left.size() != right.size())
        throw std::invalid_argument("IPA inner-product shape mismatch");
    return cpu_ipa_inner_product(left, 0U, right, 0U, left.size());
}

inline std::vector<FrWire> cpu_ipa_mle_weights(const std::vector<FrWire>& point) {
    if (point.size() >= 31U) throw std::invalid_argument("IPA point dimension exceeds baseline cap");
    std::vector<FrWire> weights(size_t(1) << point.size());
    weights[0] = fr_from_u32(1U);
    const FrWire one = fr_from_u32(1U);
    for (size_t bit = 0; bit < point.size(); ++bit) {
        const size_t old_size = size_t(1) << bit;
        const FrWire u = point[bit];
        const FrWire one_minus = fr_sub(one, u);
        for (size_t i = old_size; i-- > 0U;) {
            const FrWire parent = weights[i];
            weights[2U * i] = fr_mul(parent, one_minus);
            weights[2U * i + 1U] = fr_mul(parent, u);
        }
    }
    return weights;
}

// The padded vector's zero upper half makes round zero contribute only inverse[0].
inline FrWire cpu_ipa_terminal_mle_weight(const std::vector<FrWire>& point, const std::vector<FrWire>& fold_challenges, const std::vector<FrWire>& fold_inverses) {
    FrWire weight = fold_inverses[0U];
    const FrWire one = fr_from_u32(1U);
    for (std::size_t coordinate = 0U; coordinate < point.size(); ++coordinate) {
        const FrWire& x = point[coordinate];
        const FrWire factor = fr_add(fr_mul(fr_sub(one, x), fold_inverses[coordinate + 1U]), fr_mul(x, fold_challenges[coordinate + 1U]));
        weight = fr_mul(weight, factor);
    }
    return weight;
}

inline FrWire cpu_ipa_nonzero_challenge(Transcript& transcript, const std::string& label) {
    FrWire challenge{};
    do {
        challenge = fr_wire(transcript.challenge(label));
    } while (fr_compare(challenge, FrWire{}) == 0);
    return challenge;
}

inline bool cpu_pedersen_strict_g1(
    const CpuG1Point& point, std::string* error = nullptr,
    std::array<byte, 48U>* encoded_out = nullptr) {
    std::array<byte, 48U> encoded{};
    std::string point_error;
    if (zkdl_g1_wire::encode(cpu_g1_to_native(point), encoded, &point_error)) {
        if (encoded_out != nullptr) *encoded_out = encoded;
        return true;
    }
    if (error) *error = point_error.empty() ? "invalid Pedersen G1 point" : point_error;
    return false;
}

inline bool cpu_pedersen_strict_wire_g1(const G1Jacobian_t& point,
                                        std::string* error = nullptr) {
    return pedersen_wire_g1_valid(point, error);
}

inline bool cpu_pedersen_basis_valid(const std::vector<CpuG1Point>& generators,
                                     const CpuG1Point& h, const CpuG1Point& u,
                                     std::uint32_t expected_size,
                                     std::string* error = nullptr) {
    if (generators.size() != expected_size || generators.empty() || h.is_zero() || u.is_zero() ||
        cpu_g1_equal(h, u)) {
        if (error) *error = "Pedersen basis shape or H/U independence is invalid";
        return false;
    }
    const std::vector<byte> cache_key = cpu_pedersen_basis_cache_key(
        generators, h, u, expected_size);
    std::array<byte, 32U> cached_digest{};
    if (cpu_pedersen_basis_cache_lookup(cache_key, cached_digest)) return true;
    std::array<byte, 48U> encoded_h{};
    std::array<byte, 48U> encoded_u{};
    if (!cpu_pedersen_strict_g1(h, error, &encoded_h) ||
        !cpu_pedersen_strict_g1(u, error, &encoded_u)) return false;
    std::vector<byte> canonical_encoding;
    append_u32(canonical_encoding, static_cast<std::uint32_t>(generators.size()));
    append_u32(canonical_encoding, expected_size);
    std::array<byte, 48U> encoded_point{};
    for (const auto& generator : generators) {
        if (generator.is_zero() || !cpu_pedersen_strict_g1(generator, error, &encoded_point)) {
            if (error && error->empty()) *error = "Pedersen basis contains an invalid point";
            return false;
        }
        if (cpu_g1_equal(generator, h) || cpu_g1_equal(generator, u)) {
            if (error) *error = "Pedersen basis reuses H/U";
            return false;
        }
        canonical_encoding.insert(canonical_encoding.end(), encoded_point.begin(), encoded_point.end());
    }
    canonical_encoding.insert(canonical_encoding.end(), encoded_h.begin(), encoded_h.end());
    canonical_encoding.insert(canonical_encoding.end(), encoded_u.begin(), encoded_u.end());
    /* The digest layout is kept byte-for-byte identical to
     * cpu_ipa_basis_digest's authenticated basis digest.  The cache entry is
     * published only after the complete strict validation above. */
    cpu_pedersen_basis_cache_store(cache_key, sha256(canonical_encoding));
    return true;
}

class CpuIpaBasisContext {
private:
    struct Data {
        std::vector<CpuG1Point> generators;
        CpuG1Point h;
        CpuG1Point u;
        std::uint32_t padded_size = 0U;
        std::array<byte, 32U> full_digest{};
        mutable std::mutex digest_mutex;
        mutable std::map<std::pair<std::uint32_t, std::uint32_t>,
                         std::array<byte, 32U>> prefix_digests;
    };

    explicit CpuIpaBasisContext(std::shared_ptr<const Data> data)
        : data_(std::move(data)) {}

public:
    CpuIpaBasisContext() = default;

    // Only this strict factory can create a live receipt. Callers transfer
    // ownership of the decoded request basis; the receipt then holds the
    // immutable vectors and H/U used by all proof or verification openings.
    static bool authenticate(std::vector<CpuG1Point>&& generators,
                             const CpuG1Point& h, const CpuG1Point& u,
                             CpuIpaBasisContext& result,
                             std::string* error = nullptr) {
        result = CpuIpaBasisContext{};
        if (generators.empty() || generators.size() > (1U << 19U) ||
            h.is_zero() || u.is_zero() || cpu_g1_equal(h, u)) {
            if (error) *error = "IPA basis context shape or H/U independence is invalid";
            return false;
        }
        std::uint32_t padded_size = 0U;
        try {
            padded_size = cpu_ipa_padded_size(
                static_cast<std::uint32_t>(generators.size()));
        } catch (const std::exception& exception) {
            if (error) *error = exception.what();
            return false;
        }
        auto data = std::make_shared<Data>();
        data->generators = std::move(generators);
        data->h = h;
        data->u = u;
        data->padded_size = padded_size;
        std::array<byte, 48U> encoded_h{};
        std::array<byte, 48U> encoded_u{};
        if (!cpu_pedersen_strict_g1(data->h, error, &encoded_h) ||
            !cpu_pedersen_strict_g1(data->u, error, &encoded_u)) return false;
        std::vector<byte> canonical_encoding;
        canonical_encoding.reserve(8U + data->generators.size() * 48U + 96U);
        append_u32(canonical_encoding,
                   static_cast<std::uint32_t>(data->generators.size()));
        append_u32(canonical_encoding, data->padded_size);
        std::array<byte, 48U> encoded_point{};
        for (const auto& generator : data->generators) {
            if (generator.is_zero() ||
                !cpu_pedersen_strict_g1(generator, error, &encoded_point)) {
                if (error && error->empty())
                    *error = "IPA basis context contains an invalid point";
                return false;
            }
            if (cpu_g1_equal(generator, data->h) ||
                cpu_g1_equal(generator, data->u)) {
                if (error) *error = "IPA basis context reuses H/U";
                return false;
            }
            canonical_encoding.insert(canonical_encoding.end(),
                                      encoded_point.begin(), encoded_point.end());
        }
        canonical_encoding.insert(canonical_encoding.end(), encoded_h.begin(),
                                  encoded_h.end());
        canonical_encoding.insert(canonical_encoding.end(), encoded_u.begin(),
                                  encoded_u.end());
        data->full_digest = sha256(canonical_encoding);
        data->prefix_digests.emplace(
            std::make_pair(static_cast<std::uint32_t>(data->generators.size()),
                           data->padded_size), data->full_digest);
        result = CpuIpaBasisContext(std::move(data));
        return true;
    }

    bool valid() const noexcept { return static_cast<bool>(data_); }
    std::size_t size() const noexcept { return data_ ? data_->generators.size() : 0U; }
    bool empty() const noexcept { return size() == 0U; }
    const std::vector<CpuG1Point>& generators() const noexcept {
        static const std::vector<CpuG1Point> empty_generators;
        return data_ ? data_->generators : empty_generators;
    }
    const CpuG1Point& h() const noexcept {
        static const CpuG1Point zero = CpuG1Point::zero();
        return data_ ? data_->h : zero;
    }
    const CpuG1Point& u() const noexcept {
        static const CpuG1Point zero = CpuG1Point::zero();
        return data_ ? data_->u : zero;
    }
    std::vector<CpuG1Point>::const_iterator begin() const noexcept {
        return generators().cbegin();
    }
    std::vector<CpuG1Point>::const_iterator end() const noexcept {
        return generators().cend();
    }
    const CpuG1Point& operator[](std::size_t index) const {
        return generators().at(index);
    }
    operator const std::vector<CpuG1Point>&() const noexcept { return generators(); }

    bool owns_exact_generators(const std::vector<CpuG1Point>& generators,
                               const CpuG1Point& h,
                               const CpuG1Point& u) const noexcept {
        return data_ && &generators == &data_->generators &&
               cpu_g1_equal(h, data_->h) && cpu_g1_equal(u, data_->u);
    }

    bool validates_prefix(std::uint32_t expected_size, const CpuG1Point& h,
                          const CpuG1Point& u) const noexcept {
        return data_ && expected_size != 0U && expected_size <= data_->generators.size() &&
               cpu_g1_equal(h, data_->h) && cpu_g1_equal(u, data_->u);
    }

    std::array<byte, 32U> digest_for(std::uint32_t generator_size,
                                     std::uint32_t padded_size) const {
        if (!data_ || generator_size == 0U || generator_size > data_->generators.size() ||
            cpu_ipa_padded_size(generator_size) != padded_size)
            throw std::invalid_argument("IPA basis context digest shape mismatch");
        const auto key = std::make_pair(generator_size, padded_size);
        {
            std::lock_guard<std::mutex> lock(data_->digest_mutex);
            const auto found = data_->prefix_digests.find(key);
            if (found != data_->prefix_digests.end()) return found->second;
        }
        std::vector<byte> encoded;
        encoded.reserve(8U + static_cast<std::size_t>(generator_size) * 48U + 96U);
        append_u32(encoded, generator_size);
        append_u32(encoded, padded_size);
        for (std::uint32_t index = 0U; index < generator_size; ++index)
            append_g1(encoded, cpu_g1_to_native(data_->generators[index]));
        append_g1(encoded, cpu_g1_to_native(data_->h));
        append_g1(encoded, cpu_g1_to_native(data_->u));
        const std::array<byte, 32U> digest = sha256(encoded);
        std::lock_guard<std::mutex> lock(data_->digest_mutex);
        return data_->prefix_digests.emplace(key, digest).first->second;
    }

private:
    std::shared_ptr<const Data> data_;
};

/* A proof-local immutable affine view of an authenticated IPA basis.  The
 * projective points remain owned by CpuIpaBasisContext; this object only
 * caches their BLST affine conversion for repeated segmented MSMs. */
class CpuG1PreparedMsmBasis {
public:
    explicit CpuG1PreparedMsmBasis(const CpuIpaBasisContext& authenticated_basis)
        : authenticated_basis_(authenticated_basis) {
        if (!authenticated_basis_.valid())
            throw std::invalid_argument("prepared G1 MSM requires an authenticated IPA basis");
#if defined(ZKDL_USE_BLST_G1_CODEC)
        const auto& generators = authenticated_basis_.generators();
        affine_.resize(generators.size());
        for (std::size_t index = 0U; index < generators.size(); ++index) {
            const zkdl_blst::blst_p1 projective = cpu_g1_to_blst(generators[index]);
            zkdl_blst::blst_p1_to_affine(&affine_[index], &projective);
        }
#endif
    }

    const CpuIpaBasisContext& authenticated_basis() const noexcept {
        return authenticated_basis_;
    }
    std::size_t size() const noexcept {
#if defined(ZKDL_USE_BLST_G1_CODEC)
        return affine_.size();
#else
        return authenticated_basis_.size();
#endif
    }
    bool owns_exact_generators(const CpuIpaBasisContext& basis,
                               const std::vector<CpuG1Point>& generators,
                               const CpuG1Point& h,
                               const CpuG1Point& u) const noexcept {
        return authenticated_basis_.owns_exact_generators(generators, h, u) &&
               basis.owns_exact_generators(generators, h, u);
    }
#if defined(ZKDL_USE_BLST_G1_CODEC)
    const zkdl_blst::blst_p1_affine* affine_points() const noexcept {
        return affine_.data();
    }
#endif

private:
    CpuIpaBasisContext authenticated_basis_;
#if defined(ZKDL_USE_BLST_G1_CODEC)
    std::vector<zkdl_blst::blst_p1_affine> affine_;
#endif
};

/* Scratch is caller-owned and reused by every MSM performed through one
 * CpuG1MsmSession.  It is deliberately not shared globally or across proof
 * threads. */
class CpuG1MsmWorkspace {
public:
    CpuG1MsmWorkspace() {
#if defined(ZKDL_USE_BLST_G1_CODEC)
        if (cpu_g1_msm_workspace_bound_bytes() > kCpuG1MsmProfileBytes)
            throw std::runtime_error("BLST G1 MSM exceeds its source-bound workspace profile");
        scratch_bytes_ =
            zkdl_blst::blst_p1s_mult_pippenger_scratch_sizeof(kCpuG1MsmChunkPoints);
        if (scratch_bytes_ == 0U ||
            scratch_bytes_ % sizeof(zkdl_blst::limb_t) != 0U)
            throw std::runtime_error("BLST returned an invalid Pippenger scratch bound");
        scalar_ptrs_.reset(new const zkdl_blst::byte*[kCpuG1MsmChunkPoints]);
        scratch_.reset(new zkdl_blst::limb_t[
            scratch_bytes_ / sizeof(zkdl_blst::limb_t)]);
#endif
    }

    CpuG1MsmWorkspace(const CpuG1MsmWorkspace&) = delete;
    CpuG1MsmWorkspace& operator=(const CpuG1MsmWorkspace&) = delete;
    CpuG1MsmWorkspace(CpuG1MsmWorkspace&&) noexcept = default;
    CpuG1MsmWorkspace& operator=(CpuG1MsmWorkspace&&) noexcept = default;

#if defined(ZKDL_USE_BLST_G1_CODEC)
    const zkdl_blst::byte** scalar_ptrs() noexcept { return scalar_ptrs_.get(); }
    std::array<zkdl_blst::byte, 8U>* signed_scalar_scratch(
        std::size_t count) {
        if (signed_scalar_capacity_ < count) {
            signed_scalar_scratch_.reset();
            signed_scalar_capacity_ = 0U;
            signed_scalar_scratch_.reset(
                new std::array<zkdl_blst::byte, 8U>[count]);
            signed_scalar_capacity_ = count;
        }
        return signed_scalar_scratch_.get();
    }
    zkdl_blst::blst_p1_affine* signed_affine_scratch(
        std::size_t count) {
        if (signed_affine_capacity_ < count) {
            signed_affine_scratch_.reset();
            signed_affine_capacity_ = 0U;
            signed_affine_scratch_.reset(
                new zkdl_blst::blst_p1_affine[count]);
            signed_affine_capacity_ = count;
        }
        return signed_affine_scratch_.get();
    }
    const zkdl_blst::blst_p1_affine** indexed_point_ptrs() {
        if (!indexed_point_ptrs_)
            indexed_point_ptrs_.reset(
                new const zkdl_blst::blst_p1_affine*[kCpuG1MsmChunkPoints]);
        return indexed_point_ptrs_.get();
    }
    zkdl_blst::limb_t* scratch() noexcept { return scratch_.get(); }
    std::size_t resident_bytes() const noexcept {
        return scratch_bytes_ + kCpuG1MsmChunkPoints *
            (sizeof(const zkdl_blst::byte*) +
             (indexed_point_ptrs_ ? sizeof(const zkdl_blst::blst_p1_affine*) : 0U)) +
            signed_scalar_capacity_ *
                sizeof(std::array<zkdl_blst::byte, 8U>) +
            signed_affine_capacity_ * sizeof(zkdl_blst::blst_p1_affine);
    }
#endif

private:
#if defined(ZKDL_USE_BLST_G1_CODEC)
    std::unique_ptr<const zkdl_blst::byte*[]> scalar_ptrs_;
    std::unique_ptr<const zkdl_blst::blst_p1_affine*[]> indexed_point_ptrs_;
    std::unique_ptr<zkdl_blst::limb_t[]> scratch_;
    std::unique_ptr<std::array<zkdl_blst::byte, 8U>[]> signed_scalar_scratch_;
    std::unique_ptr<zkdl_blst::blst_p1_affine[]> signed_affine_scratch_;
    std::size_t signed_scalar_capacity_ = 0U;
    std::size_t signed_affine_capacity_ = 0U;
    size_t scratch_bytes_ = 0U;
#endif
};

struct CpuHyraxGpuFoldStats {
    std::uint64_t kernel_count = 0U;
    std::uint64_t peak_allocated_bytes = 0U;
    int device_ordinal = -1;
    std::string device_name;
    std::string device_uuid;
};

struct CpuHyraxGpuFoldRound {
    CpuG1Point left;
    CpuG1Point right;
    FrWire cross_left{};
    FrWire cross_right{};
};

struct CpuHyraxGpuFoldFinalState {
    CpuG1Point generator;
    FrWire a{};
    FrWire b{};
};

/* CUDA is an explicit, proof-local offload for authenticated Hyrax G/a/b
 * folding and cross terms. Transcript control, relation evaluation, and
 * verification remain on the CPU. */
class CpuHyraxGpuFoldSession {
public:
    virtual ~CpuHyraxGpuFoldSession() = default;
    virtual bool owns_exact_generators(
        const CpuIpaBasisContext& basis,
        const std::vector<CpuG1Point>& generators,
        const CpuG1Point& h, const CpuG1Point& u) const noexcept = 0;
    virtual void reset(std::size_t generator_count) = 0;
    virtual void initialize_scalars(const std::vector<FrWire>& current_a,
                                    const std::vector<FrWire>& current_b) = 0;
    virtual CpuHyraxGpuFoldRound compute_lr(std::size_t current_size) = 0;
    virtual void fold(const FrWire& challenge, const FrWire& inverse,
                      std::size_t half) = 0;
    virtual CpuHyraxGpuFoldFinalState final_state() = 0;
    virtual CpuHyraxGpuFoldStats stats() const = 0;
};

std::unique_ptr<CpuHyraxGpuFoldSession> make_cpu_hyrax_gpu_fold_session(
    const CpuIpaBasisContext& basis, const CpuG1Point& h,
    const CpuG1Point& u);

class CpuG1MsmSession {
public:
    explicit CpuG1MsmSession(const CpuIpaBasisContext& authenticated_basis)
        : prepared_basis_(authenticated_basis) {}

    CpuG1Point sum(const CpuIpaBasisContext& authenticated_basis,
                   const std::vector<CpuG1Point>& generators,
                   const CpuG1Point& h, const CpuG1Point& u,
                   const FrWire* scalars, size_t count) {
        if (!prepared_basis_.owns_exact_generators(
                authenticated_basis, generators, h, u))
            throw std::invalid_argument(
                "prepared G1 MSM basis does not own the supplied generators/H/U");
        if (count != 0U && scalars == nullptr)
            throw std::invalid_argument("prepared G1 MSM has null scalar input");
        if (count > prepared_basis_.size())
            throw std::invalid_argument("prepared G1 MSM exceeds authenticated basis size");
        std::lock_guard<std::mutex> workspace_lock(workspace_mutex_);
#if defined(ZKDL_USE_BLST_G1_CODEC)
        if (count == 0U) return CpuG1Point::zero();
        const std::size_t chunk_count =
            (count + kCpuG1MsmChunkPoints - 1U) / kCpuG1MsmChunkPoints;
        std::vector<std::size_t> chunk_nbits(chunk_count, 0U);
        bool any_nonzero = false;
        for (std::size_t chunk = 0U; chunk < chunk_count; ++chunk) {
            const std::size_t offset = chunk * kCpuG1MsmChunkPoints;
            const std::size_t points = std::min(kCpuG1MsmChunkPoints, count - offset);
            FrWire scalar_or{};
            for (std::size_t index = 0U; index < points; ++index) {
                const FrWire& scalar = scalars[offset + index];
                if (!fr_canonical(scalar))
                    throw std::invalid_argument("non-canonical G1 MSM scalar");
                for (std::size_t limb = 0U; limb < scalar_or.size(); ++limb)
                    scalar_or[limb] |= scalar[limb];
            }
            chunk_nbits[chunk] = cpu_g1_msm_or_nbits(scalar_or);
            any_nonzero = any_nonzero || chunk_nbits[chunk] != 0U;
        }
        if (!any_nonzero) return CpuG1Point::zero();
        std::size_t workspace_count = 1U;
        bool parallel_chunks = false;
        if (chunk_count >= kCpuG1MsmParallelThresholdChunks &&
            !cpu_in_parallel_worker()) {
            const std::size_t lane_limit = std::min(
                kCpuG1MsmMaxConcurrentChunks,
                static_cast<std::size_t>(cpu_parallel_worker_count()) + 1U);
            workspace_count = std::min(chunk_count, lane_limit);
            parallel_chunks = workspace_count >= kCpuG1MsmParallelThresholdChunks;
        }
        if (!parallel_chunks) workspace_count = 1U;
        if (workspaces_.size() < workspace_count)
            workspaces_.resize(workspace_count);

        const zkdl_blst::blst_p1_affine* const affine_base =
            prepared_basis_.affine_points();
        std::vector<CpuG1Point> partials(chunk_count, CpuG1Point::zero());
        std::vector<std::size_t> active_signed_nbits(workspace_count, 0U);
        std::vector<std::uint8_t> active_signed_has_negative(
            workspace_count, 0U);
        const auto execute_chunk = [&](std::size_t chunk_index,
                                       std::size_t workspace_index) {
            const std::size_t offset = chunk_index * kCpuG1MsmChunkPoints;
            const std::size_t points = std::min(kCpuG1MsmChunkPoints, count - offset);
            if (chunk_nbits[chunk_index] == 0U) return;
            CpuG1MsmWorkspace& workspace = workspaces_[workspace_index];
            const std::size_t pippenger_nbits =
                active_signed_nbits[workspace_index] != 0U
                    ? active_signed_nbits[workspace_index]
                    : chunk_nbits[chunk_index];
            const zkdl_blst::blst_p1_affine* chunk_affine = affine_base + offset;
            if (pippenger_nbits == chunk_nbits[chunk_index]) {
                for (std::size_t index = 0U; index < points; ++index)
                    workspace.scalar_ptrs()[index] =
                        reinterpret_cast<const zkdl_blst::byte*>(
                            scalars[offset + index].data());
            } else if (active_signed_has_negative[workspace_index] != 0U) {
                chunk_affine = workspace.signed_affine_scratch(points);
            }
            const zkdl_blst::blst_p1_affine* affine_sources[2] = {
                chunk_affine, nullptr};
            zkdl_blst::blst_p1 partial{};
            zkdl_blst::blst_p1s_mult_pippenger(
                &partial, affine_sources, points,
                workspace.scalar_ptrs(), pippenger_nbits,
                workspace.scratch());
            partials[chunk_index] = cpu_g1_from_blst(partial);
        };
        for (std::size_t base = 0U; base < chunk_count; base += workspace_count) {
            const std::size_t active_chunks =
                std::min(workspace_count, chunk_count - base);
            std::fill(active_signed_nbits.begin(), active_signed_nbits.end(), 0U);
            std::fill(active_signed_has_negative.begin(),
                      active_signed_has_negative.end(), 0U);
            for (std::size_t lane = 0U; lane < active_chunks; ++lane) {
                const std::size_t chunk_index = base + lane;
                if (chunk_nbits[chunk_index] <= 64U) continue;
                const std::size_t offset = chunk_index * kCpuG1MsmChunkPoints;
                const std::size_t points =
                    std::min(kCpuG1MsmChunkPoints, count - offset);
                std::size_t signed_nbits = 0U;
                bool signed_small = true;
                bool has_negative = false;
                for (std::size_t index = 0U; index < points; ++index) {
                    std::uint64_t magnitude = 0U;
                    bool negative = false;
                    if (!cpu_g1_msm_try_decode_signed64(
                            scalars[offset + index], magnitude, negative)) {
                        signed_small = false;
                        break;
                    }
                    signed_nbits = std::max(
                        signed_nbits, cpu_g1_msm_u64_nbits(magnitude));
                    has_negative = has_negative || negative;
                }
                if (!signed_small || signed_nbits == 0U) continue;

                CpuG1MsmWorkspace& workspace = workspaces_[lane];
                auto* scalar_scratch = workspace.signed_scalar_scratch(points);
                auto* affine_scratch = has_negative
                    ? workspace.signed_affine_scratch(points)
                    : nullptr;
                if (has_negative)
                    std::copy_n(affine_base + offset, points, affine_scratch);
                for (std::size_t index = 0U; index < points; ++index) {
                    std::uint64_t magnitude = 0U;
                    bool negative = false;
                    if (!cpu_g1_msm_try_decode_signed64(
                            scalars[offset + index], magnitude, negative))
                        throw std::logic_error(
                            "signed-small G1 MSM scalar changed during chunk preparation");
                    for (std::size_t byte_index = 0U; byte_index < 8U; ++byte_index)
                        scalar_scratch[index][byte_index] =
                            static_cast<zkdl_blst::byte>(
                                magnitude >> (8U * byte_index));
                    workspace.scalar_ptrs()[index] = scalar_scratch[index].data();
                    if (negative) {
                        zkdl_blst::blst_fp negative_y{};
                        zkdl_blst::blst_fp_cneg(
                            &negative_y, &affine_scratch[index].y, true);
                        affine_scratch[index].y = negative_y;
                    }
                }
                if (has_negative) {
                    active_signed_has_negative[lane] = 1U;
                }
                active_signed_nbits[lane] = signed_nbits;
            }
            if (parallel_chunks &&
                active_chunks >= kCpuG1MsmParallelThresholdChunks) {
                cpu_parallel_for_expensive_tasks(
                    active_chunks, 1U, kCpuG1MsmParallelThresholdChunks,
                    [&](std::size_t lane) { execute_chunk(base + lane, lane); });
            } else {
                for (std::size_t lane = 0U; lane < active_chunks; ++lane)
                    execute_chunk(base + lane, lane);
            }
        }
        CpuG1Point result = CpuG1Point::zero();
        for (const auto& partial : partials)
            result = cpu_g1_add(result, partial);
        return result;
#else
        if (count != 0U && generators.size() < count)
            throw std::invalid_argument("prepared G1 MSM generator shape mismatch");
        CpuG1Point result = CpuG1Point::zero();
        for (size_t index = 0U; index < count; ++index) {
            if (!fr_canonical(scalars[index]))
                throw std::invalid_argument("non-canonical G1 MSM scalar");
            result = cpu_g1_add(result,
                                cpu_g1_mul_portable(generators[index], scalars[index]));
        }
        return result;
#endif
    }

    CpuG1Point sum_indexed(const CpuIpaBasisContext& authenticated_basis,
                           const std::vector<CpuG1Point>& generators,
                           const CpuG1Point& h, const CpuG1Point& u,
                           std::size_t authenticated_prefix,
                           const std::uint32_t* indices, const FrWire* scalars,
                           std::size_t count) {
        if (!prepared_basis_.owns_exact_generators(
                authenticated_basis, generators, h, u))
            throw std::invalid_argument(
                "prepared indexed G1 MSM basis does not own supplied generators/H/U");
        if (authenticated_prefix == 0U || authenticated_prefix > prepared_basis_.size())
            throw std::invalid_argument(
                "prepared indexed G1 MSM prefix exceeds authenticated basis");
        if (count != 0U && (indices == nullptr || scalars == nullptr))
            throw std::invalid_argument("prepared indexed G1 MSM has null input");
        std::lock_guard<std::mutex> workspace_lock(workspace_mutex_);
        for (std::size_t index = 0U; index < count; ++index) {
            if (indices[index] >= authenticated_prefix)
                throw std::invalid_argument(
                    "prepared indexed G1 MSM index exceeds authenticated prefix");
            if (!fr_canonical(scalars[index]))
                throw std::invalid_argument("non-canonical indexed G1 MSM scalar");
        }
#if defined(ZKDL_USE_BLST_G1_CODEC)
        if (count == 0U) return CpuG1Point::zero();
        const std::size_t chunk_count =
            (count + kCpuG1MsmChunkPoints - 1U) / kCpuG1MsmChunkPoints;
        std::vector<FrWire> chunk_scalar_or(chunk_count);
        for (std::size_t index = 0U; index < count; ++index) {
            FrWire& scalar_or = chunk_scalar_or[index / kCpuG1MsmChunkPoints];
            for (std::size_t limb = 0U; limb < scalar_or.size(); ++limb)
                scalar_or[limb] |= scalars[index][limb];
        }
        std::vector<std::size_t> chunk_nbits(chunk_count, 0U);
        bool any_nonzero = false;
        for (std::size_t chunk = 0U; chunk < chunk_count; ++chunk) {
            chunk_nbits[chunk] = cpu_g1_msm_or_nbits(chunk_scalar_or[chunk]);
            any_nonzero = any_nonzero || chunk_nbits[chunk] != 0U;
        }
        if (!any_nonzero) return CpuG1Point::zero();
        std::size_t workspace_count = 1U;
        bool parallel_chunks = false;
        if (chunk_count >= kCpuG1MsmParallelThresholdChunks &&
            !cpu_in_parallel_worker()) {
            const std::size_t lane_limit = std::min(
                kCpuG1MsmMaxConcurrentChunks,
                static_cast<std::size_t>(cpu_parallel_worker_count()) + 1U);
            workspace_count = std::min(chunk_count, lane_limit);
            parallel_chunks = workspace_count >= kCpuG1MsmParallelThresholdChunks;
        }
        if (!parallel_chunks) workspace_count = 1U;
        if (workspaces_.size() < workspace_count)
            workspaces_.resize(workspace_count);

        const zkdl_blst::blst_p1_affine* const affine_base =
            prepared_basis_.affine_points();
        std::vector<CpuG1Point> partials(chunk_count, CpuG1Point::zero());
        const auto execute_chunk = [&](std::size_t chunk_index,
                                       std::size_t workspace_index) {
            const std::size_t offset = chunk_index * kCpuG1MsmChunkPoints;
            const std::size_t points = std::min(kCpuG1MsmChunkPoints, count - offset);
            if (chunk_nbits[chunk_index] == 0U) return;
            CpuG1MsmWorkspace& workspace = workspaces_[workspace_index];
            auto point_ptrs = workspace.indexed_point_ptrs();
            for (std::size_t index = 0U; index < points; ++index) {
                point_ptrs[index] = affine_base + indices[offset + index];
                workspace.scalar_ptrs()[index] = reinterpret_cast<const zkdl_blst::byte*>(
                    scalars[offset + index].data());
            }
            zkdl_blst::blst_p1 partial{};
            zkdl_blst::blst_p1s_mult_pippenger(
                &partial, point_ptrs, points, workspace.scalar_ptrs(),
                chunk_nbits[chunk_index], workspace.scratch());
            partials[chunk_index] = cpu_g1_from_blst(partial);
        };
        for (std::size_t base = 0U; base < chunk_count; base += workspace_count) {
            const std::size_t active_chunks =
                std::min(workspace_count, chunk_count - base);
            if (parallel_chunks &&
                active_chunks >= kCpuG1MsmParallelThresholdChunks) {
                cpu_parallel_for_expensive_tasks(
                    active_chunks, 1U, kCpuG1MsmParallelThresholdChunks,
                    [&](std::size_t lane) { execute_chunk(base + lane, lane); });
            } else {
                for (std::size_t lane = 0U; lane < active_chunks; ++lane)
                    execute_chunk(base + lane, lane);
            }
        }
        CpuG1Point result = CpuG1Point::zero();
        for (const auto& partial : partials)
            result = cpu_g1_add(result, partial);
        return result;
#else
        CpuG1Point result = CpuG1Point::zero();
        for (std::size_t index = 0U; index < count; ++index)
            result = cpu_g1_add(
                result, cpu_g1_mul_portable(generators[indices[index]], scalars[index]));
        return result;
#endif
    }

    std::size_t prepared_basis_bytes() const noexcept {
#if defined(ZKDL_USE_BLST_G1_CODEC)
        return prepared_basis_.size() * sizeof(zkdl_blst::blst_p1_affine);
#else
        return 0U;
#endif
    }

    std::size_t workspace_bytes() const {
#if defined(ZKDL_USE_BLST_G1_CODEC)
        std::lock_guard<std::mutex> lock(workspace_mutex_);
        std::size_t total = 0U;
        for (const auto& workspace : workspaces_)
            total += workspace.resident_bytes();
        return total;
#else
        return 0U;
#endif
    }

    std::size_t workspace_count() const {
        std::lock_guard<std::mutex> lock(workspace_mutex_);
        return workspaces_.size();
    }

    void set_hyrax_gpu_fold_session(
        CpuHyraxGpuFoldSession* session) noexcept {
        hyrax_gpu_fold_session_ = session;
    }

    bool owns_exact_generators(const CpuIpaBasisContext& basis,
                               const std::vector<CpuG1Point>& generators,
                               const CpuG1Point& h,
                               const CpuG1Point& u) const noexcept {
        return prepared_basis_.owns_exact_generators(basis, generators, h, u);
    }

    CpuHyraxGpuFoldSession* hyrax_gpu_fold_session() const noexcept {
        return hyrax_gpu_fold_session_;
    }

private:
    CpuG1PreparedMsmBasis prepared_basis_;
    mutable std::mutex workspace_mutex_;
    std::vector<CpuG1MsmWorkspace> workspaces_;
    CpuHyraxGpuFoldSession* hyrax_gpu_fold_session_ = nullptr;
};

inline bool cpu_pedersen_basis_valid(const CpuIpaBasisContext& basis,
                                     const CpuG1Point& h, const CpuG1Point& u,
                                     std::uint32_t expected_size,
                                     std::string* error = nullptr) {
    if (!basis.validates_prefix(expected_size, h, u)) {
        if (error) *error = "authenticated IPA basis context does not match requested prefix/H/U";
        return false;
    }
    return true;
}

inline std::array<byte, 32U> cpu_ipa_basis_digest(
    const CpuIpaBasisContext& basis, std::uint32_t generator_size,
    std::uint32_t padded_size) {
    return basis.digest_for(generator_size, padded_size);
}

inline bool cpu_pedersen_context_equal(const PedersenProofContext& left,
                                       const PedersenProofContext& right) {
    return left.commitment_index == right.commitment_index && left.operation == right.operation &&
           left.commitment_a_index == right.commitment_a_index &&
           left.commitment_b_index == right.commitment_b_index &&
           left.commitment_c_index == right.commitment_c_index &&
           left.evaluation_commitment_index == right.evaluation_commitment_index &&
           left.degree == right.degree && left.basis_size == right.basis_size &&
           left.basis_digest == right.basis_digest &&
           left.descriptor_digest == right.descriptor_digest &&
           left.registry_digest == right.registry_digest && left.srs_digest == right.srs_digest;
}

inline void cpu_pedersen_absorb_context(std::vector<byte>& message,
                                        const PedersenProofContext& context,
                                        const CpuG1Point& h, const CpuG1Point& u) {
    append_pedersen_context(message, context);
    append_g1(message, cpu_g1_to_native(h));
    append_g1(message, cpu_g1_to_native(u));
}

inline CpuG1Point cpu_pedersen_scalar_commit(const FrWire& value, const FrWire& randomness,
                                             const CpuG1Point& h, const CpuG1Point& u) {
    return cpu_g1_add(cpu_g1_mul(u, value), cpu_g1_mul(h, randomness));
}

inline bool cpu_pedersen_context_valid_for(const PedersenProofContext& context,
                                           std::uint32_t operation,
                                           std::uint32_t degree,
                                           std::uint32_t basis_size,
                                           std::string* error = nullptr) {
    if (!pedersen_context_shape_valid(context, error) || context.operation != operation ||
        context.degree != degree || context.basis_size != basis_size) {
        if (error && error->empty()) *error = "Pedersen operation context mismatch";
        return false;
    }
    return true;
}

inline bool cpu_pedersen_basis_digest_matches(
    const PedersenProofContext& context, const std::vector<CpuG1Point>& generators,
    const CpuG1Point& h, const CpuG1Point& u, std::string* error = nullptr) {
    if (generators.size() != context.basis_size) {
        if (error) *error = "Pedersen basis size does not match context";
        return false;
    }
    const std::array<byte, 32U> expected = cpu_ipa_basis_digest(
        generators, h, u, context.basis_size);
    if (expected != context.basis_digest) {
        if (error) *error = "Pedersen basis digest mismatch";
        return false;
    }
    return true;
}

inline FrWire cpu_pedersen_challenge(const std::string& domain,
                                     const std::string& label,
                                     const std::vector<byte>& message) {
    Transcript transcript(domain);
    transcript.absorb(label, message);
    return cpu_ipa_nonzero_challenge(transcript, label + ".challenge");
}

struct PedersenScalarValueProverResult {
    PedersenScalarValueProof proof;
};

inline PedersenScalarValueProverResult cpu_open_pedersen_scalar_value_public(
    const FrWire& value, const FrWire& randomness, const CpuG1Point& expected_commitment,
    const CpuG1Point& h, const CpuG1Point& u, const PedersenProofContext& context) {
    std::string error;
    if (!fr_canonical(value) || !fr_canonical(randomness) ||
        !cpu_pedersen_context_valid_for(context, PedersenProofContext::kOperationScalarValue,
                                        1U, 0U, &error) ||
        !cpu_pedersen_strict_g1(h, &error) || !cpu_pedersen_strict_g1(u, &error) ||
        h.is_zero() || u.is_zero() || cpu_g1_equal(h, u) ||
        !cpu_pedersen_strict_g1(expected_commitment, &error))
        throw std::invalid_argument(error.empty() ? "invalid Pedersen scalar value inputs" : error);
    const CpuG1Point commitment = cpu_pedersen_scalar_commit(value, randomness, h, u);
    if (!cpu_pedersen_basis_digest_matches(context, std::vector<CpuG1Point>{}, h, u, &error) ||
        !cpu_g1_equal(commitment, expected_commitment))
        throw std::invalid_argument(error.empty() ? "Pedersen scalar commitment mismatch" : error);

    PedersenScalarValueProverResult result;
    PedersenScalarValueProof& proof = result.proof;
    proof.context = context;
    proof.commitment = cpu_g1_to_native(commitment);
    const FrWire nonce = cpu_fresh_csprng_scalar();
    proof.schnorr_R = cpu_g1_to_native(cpu_g1_mul(h, nonce));
    std::vector<byte> message;
    cpu_pedersen_absorb_context(message, context, h, u);
    append_g1(message, proof.commitment);
    append_fr(message, fr_value(value));
    append_g1(message, proof.schnorr_R);
    Transcript transcript("zkdl-pedersen-scalar-v1");
    transcript.absorb("pedersen.scalar.value", message);
    const FrWire challenge = cpu_ipa_nonzero_challenge(
        transcript, "pedersen.scalar.value.challenge");
    proof.schnorr_s = fr_value(fr_add(nonce, fr_mul(challenge, randomness)));
    return result;
}

/* Scalar value openings are explicitly public-claim proofs.  The public
 * convenience name is the only API that permits a zero commitment blinder. */
inline PedersenScalarValueProverResult cpu_open_pedersen_scalar_value(
    const FrWire& value, const FrWire& randomness, const CpuG1Point& expected_commitment,
    const CpuG1Point& h, const CpuG1Point& u, const PedersenProofContext& context) {
    if (fr_compare(randomness, FrWire{}) == 0)
        throw std::invalid_argument("private scalar value blinder must be nonzero");
    return cpu_open_pedersen_scalar_value_public(value, randomness, expected_commitment,
                                                 h, u, context);
}

inline bool cpu_verify_pedersen_scalar_value(
    const PedersenScalarValueProof& proof, const FrWire& expected_value,
    const CpuG1Point& expected_commitment, const CpuG1Point& h, const CpuG1Point& u,
    const PedersenProofContext& expected_context, std::string* error = nullptr) {
    if (!pedersen_scalar_value_shape_valid(proof, error) || !fr_canonical(expected_value) ||
        !cpu_pedersen_strict_g1(h, error) || !cpu_pedersen_strict_g1(u, error) ||
        h.is_zero() || u.is_zero() || cpu_g1_equal(h, u) ||
        !cpu_pedersen_strict_g1(expected_commitment, error) ||
        !cpu_pedersen_basis_digest_matches(expected_context, std::vector<CpuG1Point>{}, h, u,
                                           error) ||
        !cpu_pedersen_strict_wire_g1(proof.commitment, error) ||
        !cpu_pedersen_strict_wire_g1(proof.schnorr_R, error) ||
        !cpu_pedersen_context_equal(proof.context, expected_context) ||
        !cpu_g1_equal(cpu_g1_from_native(proof.commitment), expected_commitment)) {
        if (error && error->empty()) *error = "Pedersen scalar value statement mismatch";
        return false;
    }
    std::vector<byte> message;
    cpu_pedersen_absorb_context(message, expected_context, h, u);
    append_g1(message, proof.commitment);
    append_fr(message, fr_value(expected_value));
    append_g1(message, proof.schnorr_R);
    Transcript transcript("zkdl-pedersen-scalar-v1");
    transcript.absorb("pedersen.scalar.value", message);
    const FrWire challenge = cpu_ipa_nonzero_challenge(
        transcript, "pedersen.scalar.value.challenge");
    const CpuG1Point lhs = cpu_g1_mul(h, fr_wire(proof.schnorr_s));
    const CpuG1Point rhs = cpu_g1_add(
        cpu_g1_from_native(proof.schnorr_R),
        cpu_g1_mul(cpu_g1_add(cpu_g1_from_native(proof.commitment),
                              cpu_g1_neg(cpu_g1_mul(u, expected_value))), challenge));
    if (!cpu_g1_equal(lhs, rhs)) {
        if (error) *error = "Pedersen scalar value Schnorr equation mismatch";
        return false;
    }
    return true;
}

struct PedersenScalarEqualityProverResult {
    PedersenScalarEqualityProof proof;
};

inline PedersenScalarEqualityProverResult cpu_open_pedersen_scalar_equality(
    const FrWire& value, const FrWire& randomness_1, const FrWire& randomness_2,
    const CpuG1Point& expected_commitment_1, const CpuG1Point& expected_commitment_2,
    const CpuG1Point& h, const CpuG1Point& u, const PedersenProofContext& context) {
    std::string error;
    if (!fr_canonical(value) || !fr_canonical(randomness_1) || !fr_canonical(randomness_2) ||
        fr_compare(randomness_1, FrWire{}) == 0 || fr_compare(randomness_2, FrWire{}) == 0 ||
        !cpu_pedersen_context_valid_for(context, PedersenProofContext::kOperationScalarEquality,
                                        1U, 0U, &error) ||
        !cpu_pedersen_strict_g1(h, &error) || !cpu_pedersen_strict_g1(u, &error) ||
        h.is_zero() || u.is_zero() || cpu_g1_equal(h, u) ||
        !cpu_pedersen_strict_g1(expected_commitment_1, &error) ||
        !cpu_pedersen_strict_g1(expected_commitment_2, &error))
        throw std::invalid_argument(error.empty() ? "invalid Pedersen equality inputs" : error);
    const CpuG1Point commitment_1 = cpu_pedersen_scalar_commit(value, randomness_1, h, u);
    const CpuG1Point commitment_2 = cpu_pedersen_scalar_commit(value, randomness_2, h, u);
    if (!cpu_pedersen_basis_digest_matches(context, std::vector<CpuG1Point>{}, h, u, &error) ||
        !cpu_g1_equal(commitment_1, expected_commitment_1) ||
        !cpu_g1_equal(commitment_2, expected_commitment_2))
        throw std::invalid_argument(error.empty() ? "Pedersen equality commitment mismatch" : error);
    PedersenScalarEqualityProverResult result;
    PedersenScalarEqualityProof& proof = result.proof;
    proof.context = context;
    proof.commitment_1 = cpu_g1_to_native(commitment_1);
    proof.commitment_2 = cpu_g1_to_native(commitment_2);
    const FrWire nonce = cpu_fresh_csprng_scalar();
    proof.schnorr_R = cpu_g1_to_native(cpu_g1_mul(h, nonce));
    std::vector<byte> message;
    cpu_pedersen_absorb_context(message, context, h, u);
    append_g1(message, proof.commitment_1);
    append_g1(message, proof.commitment_2);
    append_g1(message, proof.schnorr_R);
    Transcript transcript("zkdl-pedersen-equality-v1");
    transcript.absorb("pedersen.scalar.equality", message);
    const FrWire challenge = cpu_ipa_nonzero_challenge(
        transcript, "pedersen.scalar.equality.challenge");
    proof.schnorr_s = fr_value(fr_add(
        nonce, fr_mul(challenge, fr_sub(randomness_1, randomness_2))));
    return result;
}

inline bool cpu_verify_pedersen_scalar_equality(
    const PedersenScalarEqualityProof& proof, const CpuG1Point& expected_commitment_1,
    const CpuG1Point& expected_commitment_2, const CpuG1Point& h, const CpuG1Point& u,
    const PedersenProofContext& expected_context, std::string* error = nullptr) {
    if (!pedersen_scalar_equality_shape_valid(proof, error) ||
        !cpu_pedersen_strict_g1(h, error) || !cpu_pedersen_strict_g1(u, error) ||
        h.is_zero() || u.is_zero() || cpu_g1_equal(h, u) ||
        !cpu_pedersen_basis_digest_matches(expected_context, std::vector<CpuG1Point>{}, h, u,
                                           error) ||
        !cpu_pedersen_strict_g1(expected_commitment_1, error) ||
        !cpu_pedersen_strict_g1(expected_commitment_2, error) ||
        !cpu_pedersen_strict_wire_g1(proof.commitment_1, error) ||
        !cpu_pedersen_strict_wire_g1(proof.commitment_2, error) ||
        !cpu_pedersen_strict_wire_g1(proof.schnorr_R, error) ||
        !cpu_pedersen_context_equal(proof.context, expected_context) ||
        !cpu_g1_equal(cpu_g1_from_native(proof.commitment_1), expected_commitment_1) ||
        !cpu_g1_equal(cpu_g1_from_native(proof.commitment_2), expected_commitment_2)) {
        if (error && error->empty()) *error = "Pedersen equality statement mismatch";
        return false;
    }
    std::vector<byte> message;
    cpu_pedersen_absorb_context(message, expected_context, h, u);
    append_g1(message, proof.commitment_1);
    append_g1(message, proof.commitment_2);
    append_g1(message, proof.schnorr_R);
    Transcript transcript("zkdl-pedersen-equality-v1");
    transcript.absorb("pedersen.scalar.equality", message);
    const FrWire challenge = cpu_ipa_nonzero_challenge(
        transcript, "pedersen.scalar.equality.challenge");
    const CpuG1Point commitment_delta = cpu_g1_add(
        cpu_g1_from_native(proof.commitment_1),
        cpu_g1_neg(cpu_g1_from_native(proof.commitment_2)));
    const CpuG1Point rhs = cpu_g1_add(
        cpu_g1_from_native(proof.schnorr_R), cpu_g1_mul(commitment_delta, challenge));
    if (!cpu_g1_equal(cpu_g1_mul(h, fr_wire(proof.schnorr_s)), rhs)) {
        if (error) *error = "Pedersen equality Schnorr equation mismatch";
        return false;
    }
    return true;
}

struct PedersenProductProverResult {
    PedersenProductProof proof;
};

inline PedersenProductProverResult cpu_open_pedersen_product(
    const FrWire& a, const FrWire& b, const FrWire& randomness_a,
    const FrWire& randomness_b, const FrWire& randomness_c,
    const CpuG1Point& expected_commitment_a, const CpuG1Point& expected_commitment_b,
    const CpuG1Point& expected_commitment_c, const CpuG1Point& h, const CpuG1Point& u,
    const PedersenProofContext& context) {
    std::string error;
    if (!fr_canonical(a) || !fr_canonical(b) || !fr_canonical(randomness_a) ||
        !fr_canonical(randomness_b) || !fr_canonical(randomness_c) ||
        fr_compare(randomness_a, FrWire{}) == 0 || fr_compare(randomness_b, FrWire{}) == 0 ||
        fr_compare(randomness_c, FrWire{}) == 0 ||
        !cpu_pedersen_context_valid_for(context, PedersenProofContext::kOperationProduct,
                                        2U, 0U, &error) ||
        !cpu_pedersen_strict_g1(h, &error) || !cpu_pedersen_strict_g1(u, &error) ||
        h.is_zero() || u.is_zero() || cpu_g1_equal(h, u) ||
        !cpu_pedersen_strict_g1(expected_commitment_a, &error) ||
        !cpu_pedersen_strict_g1(expected_commitment_b, &error) ||
        !cpu_pedersen_strict_g1(expected_commitment_c, &error))
        throw std::invalid_argument(error.empty() ? "invalid Pedersen product inputs" : error);
    const CpuG1Point commitment_a = cpu_pedersen_scalar_commit(a, randomness_a, h, u);
    const CpuG1Point commitment_b = cpu_pedersen_scalar_commit(b, randomness_b, h, u);
    PedersenProductProverResult result;
    PedersenProductProof& proof = result.proof;
    proof.context = context;
    const FrWire product = fr_mul(a, b);
    const CpuG1Point commitment_c = cpu_pedersen_scalar_commit(product, randomness_c, h, u);
    if (!cpu_pedersen_basis_digest_matches(context, std::vector<CpuG1Point>{}, h, u, &error) ||
        !cpu_g1_equal(commitment_a, expected_commitment_a) ||
        !cpu_g1_equal(commitment_b, expected_commitment_b) ||
        !cpu_g1_equal(commitment_c, expected_commitment_c))
        throw std::invalid_argument(error.empty() ? "Pedersen product commitment mismatch" : error);
    proof.commitment_a = cpu_g1_to_native(commitment_a);
    proof.commitment_b = cpu_g1_to_native(commitment_b);
    proof.commitment_c = cpu_g1_to_native(commitment_c);
    const FrWire d = cpu_fresh_csprng_scalar();
    const FrWire s = cpu_fresh_csprng_scalar();
    const FrWire t = cpu_fresh_csprng_scalar();
    proof.T1 = cpu_g1_to_native(cpu_g1_add(cpu_g1_mul(u, d), cpu_g1_mul(h, s)));
    proof.T2 = cpu_g1_to_native(cpu_g1_add(
        cpu_g1_mul(cpu_g1_from_native(proof.commitment_b), d), cpu_g1_mul(h, t)));
    std::vector<byte> message;
    cpu_pedersen_absorb_context(message, context, h, u);
    append_g1(message, proof.commitment_a);
    append_g1(message, proof.commitment_b);
    append_g1(message, proof.commitment_c);
    append_g1(message, proof.T1);
    append_g1(message, proof.T2);
    Transcript transcript("zkdl-pedersen-product-v1");
    transcript.absorb("pedersen.scalar.product", message);
    const FrWire challenge = cpu_ipa_nonzero_challenge(
        transcript, "pedersen.scalar.product.challenge");
    proof.z = fr_value(fr_add(d, fr_mul(challenge, a)));
    proof.z_s = fr_value(fr_add(s, fr_mul(challenge, randomness_a)));
    proof.z_t = fr_value(fr_add(
        t, fr_mul(challenge, fr_sub(randomness_c, fr_mul(a, randomness_b)))));
    return result;
}

inline bool cpu_verify_pedersen_product(
    const PedersenProductProof& proof, const CpuG1Point& expected_commitment_a,
    const CpuG1Point& expected_commitment_b, const CpuG1Point& expected_commitment_c,
    const CpuG1Point& h, const CpuG1Point& u, const PedersenProofContext& expected_context,
    std::string* error = nullptr) {
    if (!pedersen_product_shape_valid(proof, error) || !cpu_pedersen_strict_g1(h, error) ||
        !cpu_pedersen_strict_g1(u, error) || h.is_zero() || u.is_zero() || cpu_g1_equal(h, u) ||
        !cpu_pedersen_basis_digest_matches(expected_context, std::vector<CpuG1Point>{}, h, u,
                                           error) ||
        !cpu_pedersen_strict_g1(expected_commitment_a, error) ||
        !cpu_pedersen_strict_g1(expected_commitment_b, error) ||
        !cpu_pedersen_strict_g1(expected_commitment_c, error) ||
        !cpu_pedersen_strict_wire_g1(proof.commitment_a, error) ||
        !cpu_pedersen_strict_wire_g1(proof.commitment_b, error) ||
        !cpu_pedersen_strict_wire_g1(proof.commitment_c, error) ||
        !cpu_pedersen_strict_wire_g1(proof.T1, error) ||
        !cpu_pedersen_strict_wire_g1(proof.T2, error) ||
        !cpu_pedersen_context_equal(proof.context, expected_context) ||
        !cpu_g1_equal(cpu_g1_from_native(proof.commitment_a), expected_commitment_a) ||
        !cpu_g1_equal(cpu_g1_from_native(proof.commitment_b), expected_commitment_b) ||
        !cpu_g1_equal(cpu_g1_from_native(proof.commitment_c), expected_commitment_c)) {
        if (error && error->empty()) *error = "Pedersen product statement mismatch";
        return false;
    }
    std::vector<byte> message;
    cpu_pedersen_absorb_context(message, expected_context, h, u);
    append_g1(message, proof.commitment_a);
    append_g1(message, proof.commitment_b);
    append_g1(message, proof.commitment_c);
    append_g1(message, proof.T1);
    append_g1(message, proof.T2);
    Transcript transcript("zkdl-pedersen-product-v1");
    transcript.absorb("pedersen.scalar.product", message);
    const FrWire challenge = cpu_ipa_nonzero_challenge(
        transcript, "pedersen.scalar.product.challenge");
    const CpuG1Point lhs_1 = cpu_g1_add(
        cpu_g1_mul(u, fr_wire(proof.z)), cpu_g1_mul(h, fr_wire(proof.z_s)));
    const CpuG1Point rhs_1 = cpu_g1_add(
        cpu_g1_from_native(proof.T1), cpu_g1_mul(cpu_g1_from_native(proof.commitment_a), challenge));
    const CpuG1Point lhs_2 = cpu_g1_add(
        cpu_g1_mul(cpu_g1_from_native(proof.commitment_b), fr_wire(proof.z)),
        cpu_g1_mul(h, fr_wire(proof.z_t)));
    const CpuG1Point rhs_2 = cpu_g1_add(
        cpu_g1_from_native(proof.T2), cpu_g1_mul(cpu_g1_from_native(proof.commitment_c), challenge));
    if (!cpu_g1_equal(lhs_1, rhs_1) || !cpu_g1_equal(lhs_2, rhs_2)) {
        if (error) *error = "Pedersen product Sigma equations mismatch";
        return false;
    }
    return true;
}

struct PedersenVectorLinearProverResult {
    PedersenVectorLinearProof proof;
};

inline PedersenVectorLinearProverResult cpu_open_pedersen_vector_linear(
    const std::vector<CpuG1Point>& generators, const std::vector<FrWire>& values,
    const FrWire& randomness, const std::vector<FrWire>& weights,
    const FrWire& evaluation_randomness, const CpuG1Point& expected_commitment,
    const CpuG1Point& expected_evaluation_commitment, const CpuG1Point& h,
    const CpuG1Point& u, const PedersenProofContext& context) {
    std::string error;
    if (!cpu_pedersen_context_valid_for(context, PedersenProofContext::kOperationVectorLinear,
                                        context.degree, static_cast<std::uint32_t>(generators.size()),
                                        &error) ||
        !cpu_pedersen_basis_valid(generators, h, u, context.basis_size, &error) ||
        weights.size() != generators.size() || values.size() != generators.size() ||
        weights.empty() || weights.size() > PedersenProofContext::kMaxBasisSize ||
        !fr_canonical(randomness) || !fr_canonical(evaluation_randomness) ||
        fr_compare(randomness, FrWire{}) == 0 ||
        fr_compare(evaluation_randomness, FrWire{}) == 0 ||
        !cpu_pedersen_strict_g1(expected_commitment, &error) ||
        !cpu_pedersen_strict_g1(expected_evaluation_commitment, &error))
        throw std::invalid_argument(error.empty() ? "invalid Pedersen vector inputs" : error);
    for (const auto& value : values)
        if (!fr_canonical(value)) throw std::invalid_argument("non-canonical vector value");
    for (const auto& weight : weights)
        if (!fr_canonical(weight)) throw std::invalid_argument("non-canonical vector weight");

    PedersenVectorLinearProverResult result;
    PedersenVectorLinearProof& proof = result.proof;
    proof.context = context;
    for (const auto& weight : weights) proof.weights.push_back(fr_value(weight));
    const CpuG1Point commitment = cpu_g1_add(cpu_g1_sum(generators, values),
                                             cpu_g1_mul(h, randomness));
    const FrWire evaluation = cpu_ipa_inner_product(values, weights);
    const CpuG1Point evaluation_commitment = cpu_pedersen_scalar_commit(
        evaluation, evaluation_randomness, h, u);
    if (!cpu_pedersen_basis_digest_matches(context, generators, h, u, &error) ||
        !cpu_g1_equal(commitment, expected_commitment) ||
        !cpu_g1_equal(evaluation_commitment, expected_evaluation_commitment))
        throw std::invalid_argument(error.empty() ? "Pedersen vector commitment mismatch" : error);
    proof.commitment = cpu_g1_to_native(commitment);
    proof.evaluation_commitment = cpu_g1_to_native(evaluation_commitment);
    std::vector<FrWire> d(values.size());
    for (auto& value : d) value = cpu_fresh_csprng_scalar();
    const FrWire s = cpu_fresh_csprng_scalar();
    const FrWire t = cpu_fresh_csprng_scalar();
    proof.A = cpu_g1_to_native(cpu_g1_add(cpu_g1_sum(generators, d), cpu_g1_mul(h, s)));
    const FrWire d_evaluation = cpu_ipa_inner_product(d, weights);
    proof.B = cpu_g1_to_native(cpu_g1_add(
        cpu_g1_mul(u, d_evaluation), cpu_g1_mul(h, t)));
    std::vector<byte> message;
    cpu_pedersen_absorb_context(message, context, h, u);
    for (const auto& generator : generators) append_g1(message, cpu_g1_to_native(generator));
    append_u64(message, static_cast<std::uint64_t>(weights.size()));
    for (const auto& weight : weights) append_fr(message, fr_value(weight));
    append_g1(message, proof.commitment);
    append_g1(message, proof.evaluation_commitment);
    append_g1(message, proof.A);
    append_g1(message, proof.B);
    Transcript transcript("zkdl-pedersen-vector-v1");
    transcript.absorb("pedersen.vector.linear", message);
    const FrWire challenge = cpu_ipa_nonzero_challenge(
        transcript, "pedersen.vector.linear.challenge");
    proof.z.reserve(values.size());
    for (std::size_t i = 0U; i < values.size(); ++i)
        proof.z.push_back(fr_value(fr_add(d[i], fr_mul(challenge, values[i]))));
    proof.z_r = fr_value(fr_add(s, fr_mul(challenge, randomness)));
    proof.z_t = fr_value(fr_add(t, fr_mul(challenge, evaluation_randomness)));
    return result;
}

inline bool cpu_verify_pedersen_vector_linear(
    const PedersenVectorLinearProof& proof, const CpuG1Point& expected_commitment,
    const CpuG1Point& expected_evaluation_commitment,
    const std::vector<CpuG1Point>& generators, const std::vector<FrWire>& expected_weights,
    const CpuG1Point& h, const CpuG1Point& u, const PedersenProofContext& expected_context,
    std::string* error = nullptr) {
    if (!pedersen_vector_linear_shape_valid(proof, error) ||
        !pedersen_context_shape_valid(expected_context, error) ||
        !cpu_pedersen_basis_valid(generators, h, u, expected_context.basis_size, error) ||
        !cpu_pedersen_basis_digest_matches(expected_context, generators, h, u, error) ||
        expected_weights.size() != generators.size() ||
        expected_weights.size() > PedersenProofContext::kMaxBasisSize ||
        proof.weights.size() != expected_weights.size() ||
        !cpu_pedersen_strict_g1(expected_commitment, error) ||
        !cpu_pedersen_strict_g1(expected_evaluation_commitment, error) ||
        !cpu_pedersen_strict_wire_g1(proof.commitment, error) ||
        !cpu_pedersen_strict_wire_g1(proof.evaluation_commitment, error) ||
        !cpu_pedersen_strict_wire_g1(proof.A, error) ||
        !cpu_pedersen_strict_wire_g1(proof.B, error) ||
        !cpu_pedersen_context_equal(proof.context, expected_context) ||
        !cpu_g1_equal(cpu_g1_from_native(proof.commitment), expected_commitment) ||
        !cpu_g1_equal(cpu_g1_from_native(proof.evaluation_commitment),
                      expected_evaluation_commitment)) {
        if (error && error->empty()) *error = "Pedersen vector statement mismatch";
        return false;
    }
    for (std::size_t i = 0U; i < expected_weights.size(); ++i) {
        if (!fr_canonical(expected_weights[i]) ||
            fr_compare(fr_wire(proof.weights[i]), expected_weights[i]) != 0) {
            if (error) *error = "Pedersen vector weight mismatch";
            return false;
        }
    }
    std::vector<byte> message;
    cpu_pedersen_absorb_context(message, expected_context, h, u);
    for (const auto& generator : generators) append_g1(message, cpu_g1_to_native(generator));
    append_u64(message, static_cast<std::uint64_t>(expected_weights.size()));
    for (const auto& weight : expected_weights) append_fr(message, fr_value(weight));
    append_g1(message, proof.commitment);
    append_g1(message, proof.evaluation_commitment);
    append_g1(message, proof.A);
    append_g1(message, proof.B);
    Transcript transcript("zkdl-pedersen-vector-v1");
    transcript.absorb("pedersen.vector.linear", message);
    const FrWire challenge = cpu_ipa_nonzero_challenge(
        transcript, "pedersen.vector.linear.challenge");
    std::vector<FrWire> responses;
    responses.reserve(proof.z.size());
    for (const auto& response : proof.z) responses.push_back(fr_wire(response));
    const CpuG1Point rhs_a = cpu_g1_add(cpu_g1_sum(generators, responses),
                                        cpu_g1_mul(h, fr_wire(proof.z_r)));
    const CpuG1Point lhs_a = cpu_g1_add(
        cpu_g1_from_native(proof.A), cpu_g1_mul(cpu_g1_from_native(proof.commitment), challenge));
    const FrWire response_evaluation = cpu_ipa_inner_product(responses, expected_weights);
    const CpuG1Point rhs_b = cpu_g1_add(
        cpu_g1_mul(u, response_evaluation), cpu_g1_mul(h, fr_wire(proof.z_t)));
    const CpuG1Point lhs_b = cpu_g1_add(
        cpu_g1_from_native(proof.B),
        cpu_g1_mul(cpu_g1_from_native(proof.evaluation_commitment), challenge));
    if (!cpu_g1_equal(lhs_a, rhs_a) || !cpu_g1_equal(lhs_b, rhs_b)) {
        if (error) *error = "Pedersen vector linear equations mismatch";
        return false;
    }
    return true;
}

/* Appendix-A.3/Fennel v3 opening helpers.  The v3 transcript deliberately
 * has a different domain from the historical IPA-v2 wire: xi/tau are
 * absorbed before any round challenge, each blinded L/R pair is absorbed
 * before its challenge, and delta/beta are absorbed before the terminal
 * challenge. */
inline void cpu_hyrax_v3_absorb_header(Transcript& transcript,
                                       const HyraxOpeningProofV3& opening) {
    std::vector<byte> header;
    append_u32(header, opening.version);
    append_u32(header, opening.commitment_index);
    append_u32(header, opening.evaluation_commitment_index);
    append_u32(header, opening.generator_size);
    append_u32(header, opening.padded_generator_size);
    append_u64(header, opening.logical_domain);
    append_u32(header, opening.segment_index);
    append_u64(header, opening.logical_offset);
    append_u32(header, opening.actual_length);
    append_u32(header, opening.padded_length);
    append_u64(header, static_cast<std::uint64_t>(opening.inner_point.size()));
    for (const auto& coordinate : opening.inner_point) append_fr(header, coordinate);
    append_u64(header, static_cast<std::uint64_t>(opening.outer_point.size()));
    for (const auto& coordinate : opening.outer_point) append_fr(header, coordinate);
    append_g1(header, opening.xi);
    append_g1(header, opening.tau);
    append_g1(header, opening.blinding_generator);
    append_g1(header, opening.u_generator);
    header.insert(header.end(), opening.basis_digest.begin(), opening.basis_digest.end());
    header.insert(header.end(), opening.descriptor_digest.begin(), opening.descriptor_digest.end());
    header.insert(header.end(), opening.registry_digest.begin(), opening.registry_digest.end());
    header.insert(header.end(), opening.srs_digest.begin(), opening.srs_digest.end());
    transcript.absorb("hyrax.v3.header", header);
}

inline void cpu_hyrax_v3_absorb_public_evaluation(Transcript& transcript,
                                                  const FrWire& evaluation) {
    std::vector<byte> encoded;
    append_fr(encoded, fr_value(evaluation));
    transcript.absorb("hyrax.v3.public.evaluation", encoded);
}

inline void cpu_hyrax_v3_absorb_round(Transcript& transcript,
                                      const G1Jacobian_t& left,
                                      const G1Jacobian_t& right) {
    std::vector<byte> round;
    append_g1(round, left);
    append_g1(round, right);
    transcript.absorb("hyrax.v3.round", round);
}

inline void cpu_hyrax_v3_absorb_terminal(Transcript& transcript,
                                         const G1Jacobian_t& delta,
                                         const G1Jacobian_t& beta) {
    std::vector<byte> terminal;
    append_g1(terminal, delta);
    append_g1(terminal, beta);
    transcript.absorb("hyrax.v3.terminal", terminal);
}

struct HyraxOpeningProverResult {
    // Only this object retains prover-local scalar state.  None of these
    // fields are consumed by the v3 serializer; the wire has no y/final_a/
    // final_b/randomness field.
    HyraxOpeningProofV3 proof;
    FrWire tau_blinder{};
    FrWire folded_blinder{};
};

#if defined(ZKDL_USE_BLST_G1_CODEC)
struct CpuIpaPreparedScalar {
    std::array<byte, 32U> little_endian{};
    zkdl_blst::blst_fr montgomery{};
};

inline CpuIpaPreparedScalar cpu_ipa_prepare_frwire_checked(
    const FrWire& value) {
    if (!fr_canonical(value))
        throw std::invalid_argument("non-canonical Hyrax fold scalar");
    CpuIpaPreparedScalar prepared;
    std::array<byte, 32U> little_endian{};
    for (std::size_t limb = 0U; limb < value.size(); ++limb)
        for (std::size_t part = 0U; part < 4U; ++part)
            little_endian[4U * limb + part] = static_cast<byte>(
                value[limb] >> (8U * part));
    prepared.little_endian = little_endian;
    zkdl_blst::blst_scalar scalar{};
    zkdl_blst::blst_scalar_from_lendian(&scalar, little_endian.data());
    zkdl_blst::blst_fr_from_scalar(&prepared.montgomery, &scalar);
    return prepared;
}

inline zkdl_blst::blst_fr cpu_ipa_frwire_to_montgomery_unchecked(
    const FrWire& value) {
    std::array<byte, 32U> little_endian{};
    for (std::size_t limb = 0U; limb < value.size(); ++limb)
        for (std::size_t part = 0U; part < 4U; ++part)
            little_endian[4U * limb + part] = static_cast<byte>(
                value[limb] >> (8U * part));
    zkdl_blst::blst_scalar scalar{};
    zkdl_blst::blst_scalar_from_lendian(&scalar, little_endian.data());
    zkdl_blst::blst_fr montgomery{};
    zkdl_blst::blst_fr_from_scalar(&montgomery, &scalar);
    return montgomery;
}

inline zkdl_blst::blst_fr cpu_ipa_frwire_to_montgomery_checked(
    const FrWire& value) {
    return cpu_ipa_prepare_frwire_checked(value).montgomery;
}

inline FrWire cpu_ipa_montgomery_to_frwire(
    const zkdl_blst::blst_fr& value) {
    zkdl_blst::blst_scalar scalar{};
    zkdl_blst::blst_scalar_from_fr(&scalar, &value);
    std::array<byte, 32U> little_endian{};
    zkdl_blst::blst_lendian_from_scalar(little_endian.data(), &scalar);
    FrWire wire{};
    for (std::size_t limb = 0U; limb < wire.size(); ++limb)
        for (std::size_t part = 0U; part < 4U; ++part)
            wire[limb] |= static_cast<std::uint32_t>(
                little_endian[4U * limb + part]) << (8U * part);
    return wire;
}

inline FrWire cpu_ipa_montgomery_product_to_frwire(
    const zkdl_blst::blst_fr& left,
    const zkdl_blst::blst_fr& right) {
    zkdl_blst::blst_fr product{};
    zkdl_blst::blst_fr_mul(&product, &left, &right);
    return cpu_ipa_montgomery_to_frwire(product);
}
#endif

inline std::vector<FrWire> cpu_ipa_next_generator_weights(
    const std::vector<FrWire>& weights, const FrWire& challenge,
    const FrWire& inverse) {
    if (weights.size() > std::numeric_limits<std::size_t>::max() / 2U)
        throw std::overflow_error("Hyrax folded generator coefficient count overflow");
    std::vector<FrWire> next(2U * weights.size());
    for (std::size_t index = 0U; index < weights.size(); ++index) {
        // Contiguous-half folds interleave the original-basis coefficient
        // blocks: [v*w0,c*w0,v*w1,c*w1,...].
        next[2U * index] = fr_mul(inverse, weights[index]);
        next[2U * index + 1U] = fr_mul(challenge, weights[index]);
    }
    return next;
}

inline std::vector<FrWire> cpu_ipa_next_generator_weights_fast(
    const std::vector<FrWire>& weights, const FrWire& challenge,
    const FrWire& inverse) {
#if defined(ZKDL_USE_BLST_G1_CODEC)
    if (weights.size() > std::numeric_limits<std::size_t>::max() / 2U)
        throw std::overflow_error("Hyrax folded generator coefficient count overflow");
    const zkdl_blst::blst_fr challenge_montgomery =
        cpu_ipa_frwire_to_montgomery_checked(challenge);
    const zkdl_blst::blst_fr inverse_montgomery =
        cpu_ipa_frwire_to_montgomery_checked(inverse);
    std::vector<zkdl_blst::blst_fr> weights_montgomery;
    weights_montgomery.reserve(weights.size());
    for (const FrWire& weight : weights)
        weights_montgomery.push_back(
            cpu_ipa_frwire_to_montgomery_checked(weight));
    std::vector<FrWire> next(2U * weights.size());
    for (std::size_t index = 0U; index < weights.size(); ++index) {
        next[2U * index] = cpu_ipa_montgomery_product_to_frwire(
            inverse_montgomery, weights_montgomery[index]);
        next[2U * index + 1U] = cpu_ipa_montgomery_product_to_frwire(
            challenge_montgomery, weights_montgomery[index]);
        if (!fr_canonical(next[2U * index]) ||
            !fr_canonical(next[2U * index + 1U]))
            throw std::runtime_error(
                "Hyrax folded generator coefficient is not canonical");
    }
    return next;
#else
    return cpu_ipa_next_generator_weights(weights, challenge, inverse);
#endif
}

inline std::vector<FrWire> cpu_ipa_hyrax_verifier_generator_weights(
    const std::vector<FrWire>& fold_challenges,
    const std::vector<FrWire>& fold_inverses,
    std::size_t generator_count, std::size_t padded_size) {
    if (generator_count == 0U ||
        generator_count > std::numeric_limits<std::size_t>::max() / 2U ||
        padded_size != 2U * generator_count ||
        fold_challenges.size() != fold_inverses.size())
        throw std::invalid_argument(
            "Hyrax verifier generator coefficient shape mismatch");
    std::size_t expected_rounds = 0U;
    for (std::size_t remaining = padded_size; remaining > 1U; remaining >>= 1U)
        ++expected_rounds;
    if (fold_challenges.size() != expected_rounds)
        throw std::invalid_argument(
            "Hyrax verifier generator coefficient round mismatch");

#if defined(ZKDL_USE_BLST_G1_CODEC)
    std::vector<zkdl_blst::blst_fr> challenge_montgomery;
    std::vector<zkdl_blst::blst_fr> inverse_montgomery;
    challenge_montgomery.reserve(expected_rounds);
    inverse_montgomery.reserve(expected_rounds);
    for (std::size_t round = 0U; round < expected_rounds; ++round) {
        challenge_montgomery.push_back(
            cpu_ipa_frwire_to_montgomery_checked(fold_challenges[round]));
        inverse_montgomery.push_back(
            cpu_ipa_frwire_to_montgomery_checked(fold_inverses[round]));
    }

    // Verify's MSM consumes only the original generator prefix.  Because the
    // padded basis is [G_0..G_(n-1), 0..0], these are exactly the low half
    // of the legacy full expansion: round zero contributes only inverse[0].
    std::vector<zkdl_blst::blst_fr> weights_montgomery(1U);
    weights_montgomery[0U] = cpu_ipa_frwire_to_montgomery_checked(
        fr_from_u32(1U));
    for (std::size_t round = expected_rounds; round-- > 1U;) {
        const std::size_t half = weights_montgomery.size();
        std::vector<zkdl_blst::blst_fr> expanded(2U * half);
        for (std::size_t index = 0U; index < half; ++index) {
            zkdl_blst::blst_fr_mul(
                &expanded[index], &inverse_montgomery[round],
                &weights_montgomery[index]);
            zkdl_blst::blst_fr_mul(
                &expanded[half + index], &challenge_montgomery[round],
                &weights_montgomery[index]);
        }
        weights_montgomery.swap(expanded);
    }
    if (weights_montgomery.size() != generator_count)
        throw std::logic_error(
            "Hyrax verifier generator coefficient prefix has the wrong size");

    std::vector<FrWire> weights(generator_count);
    for (std::size_t index = 0U; index < generator_count; ++index) {
        zkdl_blst::blst_fr coefficient{};
        zkdl_blst::blst_fr_mul(
            &coefficient, &inverse_montgomery[0U],
            &weights_montgomery[index]);
        weights[index] = cpu_ipa_montgomery_to_frwire(coefficient);
        if (!fr_canonical(weights[index]))
            throw std::runtime_error(
                "Hyrax verifier generator coefficient is not canonical");
    }
    return weights;
#else
    std::vector<FrWire> weights(1U, fr_from_u32(1U));
    for (std::size_t round = expected_rounds; round-- > 0U;) {
        const std::size_t half = weights.size();
        std::vector<FrWire> expanded(2U * half);
        for (std::size_t index = 0U; index < half; ++index) {
            expanded[index] = fr_mul(fold_inverses[round], weights[index]);
            expanded[half + index] =
                fr_mul(fold_challenges[round], weights[index]);
        }
        weights.swap(expanded);
    }
    if (weights.size() != padded_size)
        throw std::logic_error(
            "Hyrax verifier generator coefficient vector has the wrong size");
    weights.resize(generator_count);
    return weights;
#endif
}

inline void cpu_ipa_fold_projective_vectors_parallel(
    const std::vector<CpuG1Point>& current_generators,
    const std::vector<FrWire>& current_a,
    const std::vector<FrWire>& current_b,
    const FrWire& challenge, const FrWire& inverse,
    std::vector<CpuG1Point>& next_generators,
    std::vector<FrWire>& next_a, std::vector<FrWire>& next_b,
    bool known_zero_high_half = false) {
    const std::size_t current_size = current_a.size();
    if (current_size == 0U || (current_size & 1U) != 0U ||
        current_generators.size() != current_size ||
        current_b.size() != current_size)
        throw std::invalid_argument("Hyrax projective fold vector shape mismatch");
    if (!fr_canonical(challenge) || !fr_canonical(inverse))
        throw std::invalid_argument("non-canonical Hyrax fold challenge");
    for (const FrWire& value : current_a)
        if (!fr_canonical(value))
            throw std::invalid_argument("non-canonical Hyrax a-fold input");
    for (const FrWire& value : current_b)
        if (!fr_canonical(value))
            throw std::invalid_argument("non-canonical Hyrax b-fold input");

    const std::size_t half = current_size / 2U;
    next_generators.resize(half);
    next_a.resize(half);
    next_b.resize(half);
#if defined(ZKDL_USE_BLST_G1_CODEC)
    const CpuIpaPreparedScalar challenge_scalar =
        cpu_ipa_prepare_frwire_checked(challenge);
    const CpuIpaPreparedScalar inverse_scalar =
        cpu_ipa_prepare_frwire_checked(inverse);
    if (known_zero_high_half) {
        // The initial padded state is [G, 0], [a, 0], [b, 0].  The caller
        // derives this flag only from that construction, so fold the known
        // nonzero half directly without multiplying or converting zeros.
        cpu_parallel_for(half, 128U, [&](std::size_t index) {
            const zkdl_blst::blst_p1 low_generator =
                cpu_g1_to_blst(current_generators[index]);
            zkdl_blst::blst_p1 folded_generator{};
            zkdl_blst::blst_p1_mult(
                &folded_generator, &low_generator,
                inverse_scalar.little_endian.data(), sizeof(FrWire) * 8U);
            next_generators[index] = cpu_g1_from_blst(folded_generator);

            const zkdl_blst::blst_fr a_low =
                cpu_ipa_frwire_to_montgomery_unchecked(current_a[index]);
            const zkdl_blst::blst_fr b_low =
                cpu_ipa_frwire_to_montgomery_unchecked(current_b[index]);
            zkdl_blst::blst_fr a_folded{}, b_folded{};
            zkdl_blst::blst_fr_mul(
                &a_folded, &a_low, &challenge_scalar.montgomery);
            zkdl_blst::blst_fr_mul(
                &b_folded, &b_low, &inverse_scalar.montgomery);
            next_a[index] = cpu_ipa_montgomery_to_frwire(a_folded);
            next_b[index] = cpu_ipa_montgomery_to_frwire(b_folded);
            if (!fr_canonical(next_a[index]) || !fr_canonical(next_b[index]))
                throw std::runtime_error("Hyrax folded scalar is not canonical");
        });
    } else {
        cpu_parallel_for(half, 128U, [&](std::size_t index) {
            const zkdl_blst::blst_p1 low_generator =
                cpu_g1_to_blst(current_generators[index]);
            const zkdl_blst::blst_p1 high_generator =
                cpu_g1_to_blst(current_generators[half + index]);
            zkdl_blst::blst_p1 low_scaled{};
            zkdl_blst::blst_p1 high_scaled{};
            zkdl_blst::blst_p1 folded_generator{};
            zkdl_blst::blst_p1_mult(
                &low_scaled, &low_generator, inverse_scalar.little_endian.data(),
                sizeof(FrWire) * 8U);
            zkdl_blst::blst_p1_mult(
                &high_scaled, &high_generator, challenge_scalar.little_endian.data(),
                sizeof(FrWire) * 8U);
            zkdl_blst::blst_p1_add_or_double(
                &folded_generator, &low_scaled, &high_scaled);
            next_generators[index] = cpu_g1_from_blst(folded_generator);

            const zkdl_blst::blst_fr a_low =
                cpu_ipa_frwire_to_montgomery_unchecked(current_a[index]);
            const zkdl_blst::blst_fr a_high =
                cpu_ipa_frwire_to_montgomery_unchecked(current_a[half + index]);
            const zkdl_blst::blst_fr b_low =
                cpu_ipa_frwire_to_montgomery_unchecked(current_b[index]);
            const zkdl_blst::blst_fr b_high =
                cpu_ipa_frwire_to_montgomery_unchecked(current_b[half + index]);
            zkdl_blst::blst_fr a_low_product{}, a_high_product{}, a_folded{};
            zkdl_blst::blst_fr b_low_product{}, b_high_product{}, b_folded{};
            zkdl_blst::blst_fr_mul(&a_low_product, &a_low,
                                   &challenge_scalar.montgomery);
            zkdl_blst::blst_fr_mul(&a_high_product, &a_high,
                                   &inverse_scalar.montgomery);
            zkdl_blst::blst_fr_add(&a_folded, &a_low_product, &a_high_product);
            zkdl_blst::blst_fr_mul(&b_low_product, &b_low,
                                   &inverse_scalar.montgomery);
            zkdl_blst::blst_fr_mul(&b_high_product, &b_high,
                                   &challenge_scalar.montgomery);
            zkdl_blst::blst_fr_add(&b_folded, &b_low_product, &b_high_product);
            next_a[index] = cpu_ipa_montgomery_to_frwire(a_folded);
            next_b[index] = cpu_ipa_montgomery_to_frwire(b_folded);
            if (!fr_canonical(next_a[index]) || !fr_canonical(next_b[index]))
                throw std::runtime_error("Hyrax folded scalar is not canonical");
        });
    }
#else
    if (known_zero_high_half) {
        cpu_parallel_for(half, 128U, [&](std::size_t index) {
            next_generators[index] =
                cpu_g1_mul(current_generators[index], inverse);
            next_a[index] = fr_mul(current_a[index], challenge);
            next_b[index] = fr_mul(current_b[index], inverse);
        });
    } else {
        cpu_parallel_for(half, 128U, [&](std::size_t index) {
            next_generators[index] = cpu_g1_add(
                cpu_g1_mul(current_generators[index], inverse),
                cpu_g1_mul(current_generators[half + index], challenge));
            next_a[index] = fr_add(
                fr_mul(current_a[index], challenge),
                fr_mul(current_a[half + index], inverse));
            next_b[index] = fr_add(
                fr_mul(current_b[index], inverse),
                fr_mul(current_b[half + index], challenge));
        });
    }
#endif
}

inline void cpu_ipa_fold_scalar_vectors_parallel(
    const std::vector<FrWire>& current_a,
    const std::vector<FrWire>& current_b,
    const FrWire& challenge, const FrWire& inverse,
    std::vector<FrWire>& next_a, std::vector<FrWire>& next_b) {
    const std::size_t current_size = current_a.size();
    if (current_size == 0U || (current_size & 1U) != 0U ||
        current_b.size() != current_size)
        throw std::invalid_argument("Hyrax scalar fold vector shape mismatch");
    if (!fr_canonical(challenge) || !fr_canonical(inverse))
        throw std::invalid_argument("non-canonical Hyrax fold challenge");
    for (const FrWire& value : current_a)
        if (!fr_canonical(value))
            throw std::invalid_argument("non-canonical Hyrax a-fold input");
    for (const FrWire& value : current_b)
        if (!fr_canonical(value))
            throw std::invalid_argument("non-canonical Hyrax b-fold input");

    const std::size_t half = current_size / 2U;
    next_a.resize(half);
    next_b.resize(half);
#if defined(ZKDL_USE_BLST_G1_CODEC)
    const CpuIpaPreparedScalar challenge_scalar =
        cpu_ipa_prepare_frwire_checked(challenge);
    const CpuIpaPreparedScalar inverse_scalar =
        cpu_ipa_prepare_frwire_checked(inverse);
    std::vector<zkdl_blst::blst_fr> current_a_montgomery;
    current_a_montgomery.reserve(current_a.size());
    for (const FrWire& value : current_a)
        current_a_montgomery.push_back(
            cpu_ipa_frwire_to_montgomery_unchecked(value));
    std::vector<zkdl_blst::blst_fr> current_b_montgomery;
    current_b_montgomery.reserve(current_b.size());
    for (const FrWire& value : current_b)
        current_b_montgomery.push_back(
            cpu_ipa_frwire_to_montgomery_unchecked(value));
    cpu_parallel_for(half, 128U, [&](std::size_t index) {
        const zkdl_blst::blst_fr& a_low = current_a_montgomery[index];
        const zkdl_blst::blst_fr& a_high = current_a_montgomery[half + index];
        const zkdl_blst::blst_fr& b_low = current_b_montgomery[index];
        const zkdl_blst::blst_fr& b_high = current_b_montgomery[half + index];
        zkdl_blst::blst_fr a_low_product{}, a_high_product{}, a_folded{};
        zkdl_blst::blst_fr b_low_product{}, b_high_product{}, b_folded{};
        zkdl_blst::blst_fr_mul(&a_low_product, &a_low,
                               &challenge_scalar.montgomery);
        zkdl_blst::blst_fr_mul(&a_high_product, &a_high,
                               &inverse_scalar.montgomery);
        zkdl_blst::blst_fr_add(&a_folded, &a_low_product, &a_high_product);
        zkdl_blst::blst_fr_mul(&b_low_product, &b_low,
                               &inverse_scalar.montgomery);
        zkdl_blst::blst_fr_mul(&b_high_product, &b_high,
                               &challenge_scalar.montgomery);
        zkdl_blst::blst_fr_add(&b_folded, &b_low_product, &b_high_product);
        next_a[index] = cpu_ipa_montgomery_to_frwire(a_folded);
        next_b[index] = cpu_ipa_montgomery_to_frwire(b_folded);
        if (!fr_canonical(next_a[index]) || !fr_canonical(next_b[index]))
            throw std::runtime_error("Hyrax folded scalar is not canonical");
    });
#else
    cpu_parallel_for(half, 128U, [&](std::size_t index) {
        next_a[index] = fr_add(
            fr_mul(current_a[index], challenge),
            fr_mul(current_a[half + index], inverse));
        next_b[index] = fr_add(
            fr_mul(current_b[index], inverse),
            fr_mul(current_b[half + index], challenge));
    });
#endif
}

inline void cpu_ipa_fold_generator_lrs_indexed(
    CpuG1MsmSession& msm_session, const CpuIpaBasisContext& basis,
    const std::vector<CpuG1Point>& generators, std::size_t generator_count,
    const CpuG1Point& blinding_generator, const CpuG1Point& u_generator,
    const std::vector<FrWire>& current_weights,
    const std::vector<FrWire>& current_a,
    std::vector<std::uint32_t>& indices, std::vector<FrWire>& scalars,
    CpuG1Point& left, CpuG1Point& right) {
    const std::size_t current_size = current_a.size();
    if (current_size == 0U || (current_size & 1U) != 0U ||
        current_weights.empty() ||
        current_weights.size() > std::numeric_limits<std::size_t>::max() / current_size)
        throw std::invalid_argument("Hyrax indexed generator-fold shape is invalid");
    const std::size_t padded_size = cpu_ipa_padded_size(
        static_cast<std::uint32_t>(generator_count));
    if (current_size > padded_size || padded_size % current_size != 0U ||
        current_weights.size() != padded_size / current_size)
        throw std::invalid_argument("Hyrax indexed generator-fold coefficient shape mismatch");
    const std::size_t half = current_size / 2U;
    indices.clear();
    scalars.clear();
#if defined(ZKDL_USE_BLST_G1_CODEC)
    std::vector<zkdl_blst::blst_fr> weights_montgomery;
    weights_montgomery.reserve(current_weights.size());
    for (const FrWire& weight : current_weights)
        weights_montgomery.push_back(
            cpu_ipa_frwire_to_montgomery_checked(weight));
    std::vector<zkdl_blst::blst_fr> current_a_montgomery;
    current_a_montgomery.reserve(current_a.size());
    for (const FrWire& value : current_a)
        current_a_montgomery.push_back(
            cpu_ipa_frwire_to_montgomery_checked(value));
#else
    for (const FrWire& weight : current_weights)
        if (!fr_canonical(weight))
            throw std::invalid_argument(
                "non-canonical Hyrax generator-fold coefficient");
    for (const FrWire& value : current_a)
        if (!fr_canonical(value))
            throw std::invalid_argument(
                "non-canonical Hyrax generator-fold scalar");
#endif
    for (std::size_t block = 0U; block < current_weights.size(); ++block) {
        const std::size_t base = block * current_size;
        for (std::size_t index = 0U; index < half; ++index) {
            const std::size_t basis_index = base + half + index;
            if (basis_index >= generator_count) continue;
#if defined(ZKDL_USE_BLST_G1_CODEC)
            const FrWire scalar = cpu_ipa_montgomery_product_to_frwire(
                weights_montgomery[block], current_a_montgomery[index]);
#else
            const FrWire scalar = fr_mul(current_weights[block], current_a[index]);
#endif
            if (fr_compare(scalar, FrWire{}) != 0) {
                indices.push_back(static_cast<std::uint32_t>(basis_index));
                scalars.push_back(scalar);
            }
        }
    }
    const std::size_t left_count = indices.size();
    for (std::size_t block = 0U; block < current_weights.size(); ++block) {
        const std::size_t base = block * current_size;
        for (std::size_t index = 0U; index < half; ++index) {
            const std::size_t basis_index = base + index;
            if (basis_index >= generator_count) continue;
#if defined(ZKDL_USE_BLST_G1_CODEC)
            const FrWire scalar = cpu_ipa_montgomery_product_to_frwire(
                weights_montgomery[block],
                current_a_montgomery[half + index]);
#else
            const FrWire scalar = fr_mul(
                current_weights[block], current_a[half + index]);
#endif
            if (fr_compare(scalar, FrWire{}) != 0) {
                indices.push_back(static_cast<std::uint32_t>(basis_index));
                scalars.push_back(scalar);
            }
        }
    }
    const std::size_t right_count = indices.size() - left_count;
    const std::uint32_t* index_data = indices.empty() ? nullptr : indices.data();
    const FrWire* scalar_data = scalars.empty() ? nullptr : scalars.data();
    left = msm_session.sum_indexed(
        basis, generators, blinding_generator, u_generator, generator_count,
        index_data, scalar_data, left_count);
    const std::uint32_t* right_indices = right_count == 0U
        ? nullptr : index_data + left_count;
    const FrWire* right_scalars = right_count == 0U
        ? nullptr : scalar_data + left_count;
    right = msm_session.sum_indexed(
        basis, generators, blinding_generator, u_generator, generator_count,
        right_indices, right_scalars, right_count);
}

inline HyraxOpeningProverResult cpu_open_hyrax_v3_impl(
    const std::vector<CpuG1Point>& generators, const std::vector<FrWire>& values,
    const std::vector<FrWire>& point, const CpuG1Point& blinding_generator,
    const CpuG1Point& u_generator, const FrWire& randomness,
    const FrWire& tau_blinder, const CpuG1Point& expected_tau,
    const CpuIpaOpeningMetadata& metadata, bool public_evaluation,
    const CpuG1Point* checked_xi = nullptr,
    const CpuIpaBasisContext* authenticated_basis = nullptr,
    std::uint32_t requested_generator_size = 0U,
    CpuG1MsmSession* msm_session = nullptr) {
    const std::size_t generator_count = requested_generator_size == 0U
        ? generators.size() : requested_generator_size;
    if (point.size() >= 31U || generators.empty() || generator_count == 0U ||
        generator_count > generators.size() || generator_count != values.size() ||
        generator_count != (size_t(1U) << point.size()) ||
        !fr_canonical(randomness) || !fr_canonical(tau_blinder) ||
        !cpu_ipa_v3_metadata_valid(metadata))
        throw std::invalid_argument("Hyrax v3 opening shape/metadata is invalid");
    if (authenticated_basis != nullptr &&
        (!authenticated_basis->owns_exact_generators(generators, blinding_generator,
                                                       u_generator) ||
         !authenticated_basis->validates_prefix(
             static_cast<std::uint32_t>(generator_count), blinding_generator,
             u_generator)))
        throw std::invalid_argument("Hyrax v3 authenticated basis context mismatch");
    if (authenticated_basis == nullptr && generator_count != generators.size())
        throw std::invalid_argument("Hyrax v3 raw basis API cannot use an unauthenticated prefix");
    for (const auto& value : values)
        if (!fr_canonical(value)) throw std::invalid_argument("Hyrax v3 value is not canonical");
    for (const auto& coordinate : point)
        if (!fr_canonical(coordinate)) throw std::invalid_argument("Hyrax v3 point is not canonical");
    for (const auto& coordinate : metadata.outer_point)
        if (!fr_canonical(coordinate)) throw std::invalid_argument("Hyrax v3 outer point is not canonical");
    if (blinding_generator.is_zero() || u_generator.is_zero() ||
        cpu_g1_equal(blinding_generator, u_generator))
        throw std::invalid_argument("Hyrax v3 requires independent nonzero H/U");
    if (!public_evaluation && fr_compare(tau_blinder, FrWire{}) == 0)
        throw std::invalid_argument("Hyrax v3 private evaluation requires a nonzero tau blinder");
    if (!cpu_pedersen_strict_g1(expected_tau))
        throw std::invalid_argument("Hyrax v3 expected tau is not a valid G1 point");

    HyraxOpeningProverResult result;
    HyraxOpeningProofV3& opening = result.proof;
    const std::uint32_t generator_size = static_cast<std::uint32_t>(generator_count);
    const std::uint32_t padded_size = cpu_ipa_padded_size(generator_size);
    std::size_t fold_count = 0U;
    for (std::uint32_t remaining = padded_size; remaining > 1U; remaining >>= 1U)
        ++fold_count;
    opening.rounds.reserve(2U * fold_count);
    opening.inner_point.reserve(point.size());
    opening.outer_point.reserve(metadata.outer_point.size());
    opening.commitment_index = metadata.commitment_index;
    opening.evaluation_commitment_index = metadata.evaluation_commitment_index;
    opening.generator_size = generator_size;
    opening.padded_generator_size = padded_size;
    opening.logical_domain = metadata.logical_domain;
    opening.segment_index = metadata.segment_index;
    opening.logical_offset = metadata.logical_offset;
    opening.actual_length = metadata.actual_length;
    opening.padded_length = metadata.padded_length;
    for (const auto& coordinate : point) opening.inner_point.push_back(fr_value(coordinate));
    for (const auto& coordinate : metadata.outer_point) opening.outer_point.push_back(fr_value(coordinate));
    opening.blinding_generator = cpu_g1_to_native(blinding_generator);
    opening.u_generator = cpu_g1_to_native(u_generator);
    opening.basis_digest = authenticated_basis != nullptr
        ? cpu_ipa_basis_digest(*authenticated_basis, generator_size, padded_size)
        : cpu_ipa_basis_digest(generators, blinding_generator,
                               u_generator, padded_size);
    opening.descriptor_digest = metadata.descriptor_digest;
    opening.registry_digest = metadata.registry_digest;
    opening.srs_digest = metadata.srs_digest;

    const CpuG1Point xi = checked_xi != nullptr
        ? *checked_xi
        : cpu_g1_add(cpu_g1_sum(generators.data(), values.data(), generator_count),
                     cpu_g1_mul(blinding_generator, randomness));
    std::vector<FrWire> current_b = cpu_ipa_mle_weights(point);
    current_b.resize(padded_size, FrWire{});
    const FrWire evaluation = cpu_ipa_inner_product(
        values, 0U, current_b, 0U, generator_count);
    result.tau_blinder = public_evaluation ? FrWire{} : tau_blinder;
    const CpuG1Point tau = cpu_g1_add(
        cpu_g1_mul(u_generator, evaluation),
        cpu_g1_mul(blinding_generator, result.tau_blinder));
    if (!cpu_g1_equal(tau, expected_tau))
        throw std::invalid_argument("Hyrax v3 computed tau does not match expected tau");
    opening.xi = cpu_g1_to_native(xi);
    opening.tau = cpu_g1_to_native(tau);

    const bool indexed_generator_fold =
        authenticated_basis != nullptr && msm_session != nullptr;
    CpuHyraxGpuFoldSession* gpu_fold_session = indexed_generator_fold
        ? msm_session->hyrax_gpu_fold_session() : nullptr;
#if defined(ZKDL_TEST_FORCE_GPU_HYRAX_FOLD)
    constexpr std::size_t kGpuHyraxFoldMinimumGenerators = 1U;
#else
    constexpr std::size_t kGpuHyraxFoldMinimumGenerators = 4096U;
#endif
    const bool gpu_generator_fold = gpu_fold_session != nullptr &&
        generator_count >= kGpuHyraxFoldMinimumGenerators;
    if (gpu_generator_fold &&
        (!gpu_fold_session->owns_exact_generators(
             *authenticated_basis, generators, blinding_generator, u_generator) ||
         !msm_session->owns_exact_generators(
             *authenticated_basis, generators, blinding_generator, u_generator)))
        throw std::invalid_argument(
            "Hyrax CUDA fold session does not own the authenticated G/H/U basis");
    std::vector<CpuG1Point> current_generators;
    std::vector<FrWire> current_weights;
    std::vector<std::uint32_t> indexed_indices;
    std::vector<FrWire> indexed_scalars;
    if (gpu_generator_fold) {
        gpu_fold_session->reset(generator_count);
    } else if (indexed_generator_fold) {
        current_weights.push_back(fr_from_u32(1U));
        indexed_indices.reserve(generator_count);
        indexed_scalars.reserve(generator_count);
    } else {
        current_generators.assign(padded_size, CpuG1Point::zero());
    }
    std::vector<FrWire> current_a(padded_size);
    for (std::size_t i = 0U; i < generator_count; ++i) {
        if (!indexed_generator_fold) current_generators[i] = generators[i];
        current_a[i] = values[i];
    }
    if (gpu_generator_fold) {
        gpu_fold_session->initialize_scalars(current_a, current_b);
        std::vector<FrWire>().swap(current_a);
        std::vector<FrWire>().swap(current_b);
    }
    CpuG1Point current_upsilon = cpu_g1_add(xi, tau);
    FrWire current_blinder = fr_add(randomness, result.tau_blinder);

    Transcript transcript("hyrax-appendix-a3-v3");
    if (public_evaluation) cpu_hyrax_v3_absorb_public_evaluation(transcript, evaluation);
    cpu_hyrax_v3_absorb_header(transcript, opening);
    std::size_t current_size = padded_size;
    while (current_size > 1U) {
        const std::size_t half = current_size / 2U;
        const bool initial_zero_padding_round = current_size == padded_size;
        CpuG1Point left = CpuG1Point::zero();
        CpuG1Point right = CpuG1Point::zero();
        if (gpu_generator_fold) {
            if (!initial_zero_padding_round) {
                const CpuHyraxGpuFoldRound lr = gpu_fold_session->compute_lr(current_size);
                left = lr.left;
                right = lr.right;
                left = cpu_g1_add(left, cpu_g1_mul(u_generator, lr.cross_left));
                right = cpu_g1_add(right, cpu_g1_mul(u_generator, lr.cross_right));
            }
        } else if (!initial_zero_padding_round && indexed_generator_fold) {
            cpu_ipa_fold_generator_lrs_indexed(
                *msm_session, *authenticated_basis, generators, generator_count,
                blinding_generator, u_generator, current_weights, current_a,
                indexed_indices, indexed_scalars, left, right);
        } else if (!initial_zero_padding_round) {
            left = cpu_g1_sum(current_generators.data() + half,
                              current_a.data(), half);
            right = cpu_g1_sum(current_generators.data(),
                               current_a.data() + half, half);
        }
        if (!gpu_generator_fold && !initial_zero_padding_round) {
            left = cpu_g1_add(left, cpu_g1_mul(
                u_generator, cpu_ipa_inner_product(
                    current_a, 0U, current_b, half, half)));
            right = cpu_g1_add(right, cpu_g1_mul(
                u_generator, cpu_ipa_inner_product(
                    current_a, half, current_b, 0U, half)));
        }
        const FrWire r_left = cpu_fresh_csprng_scalar();
        const FrWire r_right = cpu_fresh_csprng_scalar();
        left = cpu_g1_add(left, cpu_g1_mul(blinding_generator, r_left));
        right = cpu_g1_add(right, cpu_g1_mul(blinding_generator, r_right));
        const G1Jacobian_t left_native = cpu_g1_to_native(left);
        const G1Jacobian_t right_native = cpu_g1_to_native(right);
        opening.rounds.push_back(left_native);
        opening.rounds.push_back(right_native);
        cpu_hyrax_v3_absorb_round(transcript, left_native, right_native);
        const FrWire challenge = cpu_ipa_nonzero_challenge(transcript, "hyrax.v3.fold.challenge");
        const FrWire inverse = cpu_fr_inv(challenge);
        const FrWire challenge_sq = fr_mul(challenge, challenge);
        const FrWire inverse_sq = fr_mul(inverse, inverse);
        current_upsilon = cpu_g1_add(current_upsilon, cpu_g1_mul(left, challenge_sq));
        current_upsilon = cpu_g1_add(current_upsilon, cpu_g1_mul(right, inverse_sq));
        current_blinder = fr_add(current_blinder,
                                 fr_add(fr_mul(challenge_sq, r_left),
                                        fr_mul(inverse_sq, r_right)));
        if (gpu_generator_fold) {
            gpu_fold_session->fold(challenge, inverse, half);
        } else {
            std::vector<FrWire> next_a(half), next_b(half);
            if (indexed_generator_fold) {
                cpu_ipa_fold_scalar_vectors_parallel(
                    current_a, current_b, challenge, inverse, next_a, next_b);
                current_weights = cpu_ipa_next_generator_weights_fast(
                    current_weights, challenge, inverse);
            } else {
                std::vector<CpuG1Point> next_generators;
                cpu_ipa_fold_projective_vectors_parallel(
                    current_generators, current_a, current_b, challenge, inverse,
                    next_generators, next_a, next_b,
                    initial_zero_padding_round);
                current_generators.swap(next_generators);
            }
            current_a.swap(next_a);
            current_b.swap(next_b);
        }
        current_size = half;
    }

    // Fennel's short terminal Sigma equation proves the folded scalar while
    // keeping a_hat and the accumulated blinder hidden.  The public folded
    // weight b_hat is verifier-derived and must be nonzero for this form.
    CpuHyraxGpuFoldFinalState gpu_final_state;
    if (gpu_generator_fold) gpu_final_state = gpu_fold_session->final_state();
    const FrWire a_hat = gpu_generator_fold ? gpu_final_state.a : current_a[0];
    const FrWire b_hat = gpu_generator_fold ? gpu_final_state.b : current_b[0];
    if (fr_compare(b_hat, FrWire{}) == 0)
        throw std::runtime_error("BLOCKED: Hyrax v3 terminal weight is zero");
    const CpuG1Point g_hat = gpu_generator_fold
        ? gpu_final_state.generator
        : indexed_generator_fold
        ? msm_session->sum(*authenticated_basis, generators,
                           blinding_generator, u_generator,
                           current_weights.data(), generator_count)
        : current_generators[0];
    const FrWire d = cpu_fresh_csprng_scalar();
    const FrWire r_delta = cpu_fresh_csprng_scalar();
    const FrWire r_beta = cpu_fresh_csprng_scalar();
    const CpuG1Point delta = cpu_g1_add(cpu_g1_mul(g_hat, d),
                                        cpu_g1_mul(blinding_generator, r_delta));
    const CpuG1Point beta = cpu_g1_add(cpu_g1_mul(u_generator, d),
                                       cpu_g1_mul(blinding_generator, r_beta));
    opening.delta = cpu_g1_to_native(delta);
    opening.beta = cpu_g1_to_native(beta);
    cpu_hyrax_v3_absorb_terminal(transcript, opening.delta, opening.beta);
    const FrWire final_challenge = cpu_ipa_nonzero_challenge(
        transcript, "hyrax.v3.terminal.challenge");
    const FrWire z1 = fr_add(fr_mul(fr_mul(final_challenge, a_hat), b_hat), d);
    const FrWire z2 = fr_add(
        fr_mul(b_hat, fr_add(fr_mul(final_challenge, current_blinder), r_beta)), r_delta);
    opening.z1 = fr_value(z1);
    opening.z2 = fr_value(z2);
    result.folded_blinder = current_blinder;

    CpuG1Point lhs = cpu_g1_mul(
        cpu_g1_add(cpu_g1_mul(current_upsilon, final_challenge), beta), b_hat);
    lhs = cpu_g1_add(lhs, delta);
    const CpuG1Point rhs = cpu_g1_add(
        cpu_g1_mul(cpu_g1_add(g_hat, cpu_g1_mul(u_generator, b_hat)), z1),
        cpu_g1_mul(blinding_generator, z2));
    if (!cpu_g1_equal(lhs, rhs))
        throw std::runtime_error("Hyrax v3 terminal Sigma equation failed");
    return result;
}

inline HyraxOpeningProverResult cpu_open_hyrax_v3(
    const std::vector<CpuG1Point>& generators, const std::vector<FrWire>& values,
    const std::vector<FrWire>& point, const CpuG1Point& blinding_generator,
    const CpuG1Point& u_generator, const FrWire& randomness,
    const FrWire& tau_blinder, const CpuG1Point& expected_tau,
    const CpuIpaOpeningMetadata& metadata) {
    return cpu_open_hyrax_v3_impl(generators, values, point, blinding_generator,
                                  u_generator, randomness, tau_blinder, expected_tau,
                                  metadata, false);
}

inline HyraxOpeningProverResult cpu_open_hyrax_v3_with_checked_xi(
    const std::vector<CpuG1Point>& generators, const std::vector<FrWire>& values,
    const std::vector<FrWire>& point, const CpuG1Point& blinding_generator,
    const CpuG1Point& u_generator, const FrWire& randomness,
    const FrWire& tau_blinder, const CpuG1Point& expected_tau,
    const CpuIpaOpeningMetadata& metadata, const CpuG1Point& checked_xi) {
    return cpu_open_hyrax_v3_impl(generators, values, point, blinding_generator,
                                  u_generator, randomness, tau_blinder, expected_tau,
                                  metadata, false, &checked_xi);
}

inline HyraxOpeningProverResult cpu_open_hyrax_v3_authenticated(
    const CpuIpaBasisContext& basis, const std::vector<FrWire>& values,
    const std::vector<FrWire>& point, const FrWire& randomness,
    const FrWire& tau_blinder, const CpuG1Point& expected_tau,
    const CpuIpaOpeningMetadata& metadata, const CpuG1Point& checked_xi,
    CpuG1MsmSession* msm_session = nullptr) {
    if (!basis.valid() || values.size() > basis.size() ||
        values.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Hyrax v3 authenticated basis context is unavailable");
    const auto& generators = basis.generators();
    const auto open_with_session = [&](CpuG1MsmSession* session) {
        return cpu_open_hyrax_v3_impl(
            generators, values, point, basis.h(), basis.u(), randomness,
            tau_blinder, expected_tau, metadata, false, &checked_xi, &basis,
            static_cast<std::uint32_t>(values.size()), session);
    };
    if (msm_session != nullptr) return open_with_session(msm_session);
    CpuG1MsmSession local_session(basis);
    return open_with_session(&local_session);
}

/* The pre-F3a private signature is retained only as a compile-time seam for
 * callers that have not yet supplied the authenticated tau statement.  It
 * cannot create a v3 proof or generate a hidden evaluation blinder. */
inline HyraxOpeningProverResult cpu_open_hyrax_v3(
    const std::vector<CpuG1Point>& generators, const std::vector<FrWire>& values,
    const std::vector<FrWire>& point, const CpuG1Point& blinding_generator,
    const CpuG1Point& u_generator, const FrWire& randomness,
    const CpuIpaOpeningMetadata& metadata) {
    (void)generators; (void)values; (void)point; (void)blinding_generator;
    (void)u_generator; (void)randomness; (void)metadata;
    throw std::invalid_argument(
        "Hyrax v3 private API requires explicit tau blinder and expected tau");
}

inline HyraxOpeningProverResult cpu_open_hyrax_v3_public(
    const std::vector<CpuG1Point>& generators, const std::vector<FrWire>& values,
    const std::vector<FrWire>& point, const CpuG1Point& blinding_generator,
    const CpuG1Point& u_generator, const FrWire& randomness,
    const CpuIpaOpeningMetadata& metadata) {
    if (!fr_canonical(randomness))
        throw std::invalid_argument("Hyrax v3 public xi blinder is not canonical");
    const FrWire evaluation = cpu_ipa_inner_product(
        values, cpu_ipa_mle_weights(point));
    const CpuG1Point expected_tau = cpu_g1_mul(u_generator, evaluation);
    return cpu_open_hyrax_v3_impl(generators, values, point, blinding_generator,
                                  u_generator, randomness, FrWire{}, expected_tau,
                                  metadata, true);
}

inline HyraxOpeningProverResult hyrax_opening_v3(
    const std::vector<CpuG1Point>& generators, const std::vector<FrWire>& values,
    const std::vector<FrWire>& point, const CpuG1Point& blinding_generator,
    const CpuG1Point& u_generator, const FrWire& randomness,
    const CpuIpaOpeningMetadata& metadata) {
    return cpu_open_hyrax_v3(generators, values, point, blinding_generator,
                             u_generator, randomness, metadata);
}

inline HyraxOpeningProverResult hyrax_opening_v3(
    const std::vector<CpuG1Point>& generators, const std::vector<FrWire>& values,
    const std::vector<FrWire>& point, const CpuG1Point& blinding_generator,
    const CpuG1Point& u_generator, const FrWire& randomness,
    const FrWire& tau_blinder, const CpuG1Point& expected_tau,
    const CpuIpaOpeningMetadata& metadata) {
    return cpu_open_hyrax_v3(generators, values, point, blinding_generator,
                             u_generator, randomness, tau_blinder, expected_tau,
                             metadata);
}

inline bool cpu_verify_hyrax_v3_impl(
    const HyraxOpeningProofV3& opening, const CpuG1Point& expected_xi,
    const CpuG1Point& expected_tau,
    const std::vector<CpuG1Point>& generators, const std::vector<FrWire>& expected_point,
    const CpuG1Point& blinding_generator, const CpuG1Point& u_generator,
    const CpuIpaOpeningMetadata& expected_metadata, const FrWire* public_evaluation,
    std::string* error,
    const CpuIpaBasisContext* authenticated_basis = nullptr,
    std::uint32_t requested_generator_size = 0U,
    CpuG1MsmSession* msm_session = nullptr) {
    if (!hyrax_opening_v3_shape_valid(opening, error)) return false;
    if (public_evaluation != nullptr && !fr_canonical(*public_evaluation)) {
        if (error) *error = "Hyrax v3 public evaluation is not canonical";
        return false;
    }
    const auto strict_point = [&](const G1Jacobian_t& point, const char* label) {
        std::array<byte, 48U> encoded{};
        std::string point_error;
        if (zkdl_g1_wire::encode(point, encoded, &point_error)) return true;
        if (error) *error = "invalid Hyrax v3 " + std::string(label) + " point";
        return false;
    };
    if (!strict_point(opening.xi, "xi") || !strict_point(opening.tau, "tau") ||
        !strict_point(opening.blinding_generator, "H") ||
        !strict_point(opening.u_generator, "U") || !strict_point(opening.delta, "delta") ||
        !strict_point(opening.beta, "beta")) return false;
    for (const auto& round : opening.rounds)
        if (!strict_point(round, "round")) return false;
    if (public_evaluation != nullptr &&
        !cpu_g1_equal(expected_tau, cpu_g1_mul(u_generator, *public_evaluation))) {
        if (error) *error = "Hyrax v3 public evaluation/tau mismatch";
        return false;
    }
    const std::size_t generator_count = requested_generator_size == 0U
        ? generators.size() : requested_generator_size;
    if (expected_point.size() >= 31U || generators.empty() ||
        generator_count == 0U || generator_count > generators.size() ||
        generator_count != (size_t(1U) << expected_point.size()) ||
        opening.generator_size != generator_count ||
        opening.inner_point.size() != expected_point.size() ||
        !cpu_g1_equal(cpu_g1_from_native(opening.xi), expected_xi) ||
        !cpu_g1_equal(cpu_g1_from_native(opening.tau), expected_tau) ||
        !cpu_g1_equal(cpu_g1_from_native(opening.blinding_generator), blinding_generator) ||
        !cpu_g1_equal(cpu_g1_from_native(opening.u_generator), u_generator) ||
        !cpu_ipa_v3_metadata_valid(expected_metadata) ||
        opening.commitment_index != expected_metadata.commitment_index ||
        opening.evaluation_commitment_index != expected_metadata.evaluation_commitment_index ||
        opening.outer_point.size() != expected_metadata.outer_point.size() ||
        opening.descriptor_digest != expected_metadata.descriptor_digest ||
        opening.registry_digest != expected_metadata.registry_digest ||
        opening.srs_digest != expected_metadata.srs_digest ||
        opening.logical_domain != expected_metadata.logical_domain ||
        opening.segment_index != expected_metadata.segment_index ||
        opening.logical_offset != expected_metadata.logical_offset ||
        opening.actual_length != expected_metadata.actual_length ||
        opening.padded_length != expected_metadata.padded_length) {
        if (error) *error = "Hyrax v3 statement/basis metadata mismatch";
        return false;
    }
    if (authenticated_basis != nullptr &&
        (!authenticated_basis->owns_exact_generators(generators, blinding_generator,
                                                       u_generator) ||
         !authenticated_basis->validates_prefix(
             static_cast<std::uint32_t>(generator_count), blinding_generator,
             u_generator))) {
        if (error) *error = "Hyrax v3 authenticated basis context mismatch";
        return false;
    }
    if (authenticated_basis == nullptr && generator_count != generators.size()) {
        if (error) *error = "Hyrax v3 raw basis API cannot use an unauthenticated prefix";
        return false;
    }
    for (std::size_t i = 0U; i < expected_point.size(); ++i)
        if (fr_compare(fr_wire(opening.inner_point[i]), expected_point[i]) != 0) {
            if (error) *error = "Hyrax v3 inner point metadata mismatch";
            return false;
        }
    for (std::size_t i = 0U; i < expected_metadata.outer_point.size(); ++i)
        if (fr_compare(fr_wire(opening.outer_point[i]), expected_metadata.outer_point[i]) != 0) {
            if (error) *error = "Hyrax v3 outer point metadata mismatch";
            return false;
        }
    const std::uint32_t padded_size = cpu_ipa_padded_size(opening.generator_size);
    const std::array<byte, 32U> expected_basis_digest = authenticated_basis != nullptr
        ? cpu_ipa_basis_digest(*authenticated_basis,
                               static_cast<std::uint32_t>(generator_count), padded_size)
        : cpu_ipa_basis_digest(generators, blinding_generator,
                               u_generator, padded_size);
    if (opening.padded_generator_size != padded_size ||
        opening.basis_digest != expected_basis_digest) {
        if (error) *error = "Hyrax v3 generator basis is not authenticated";
        return false;
    }
    CpuG1Point current_upsilon = cpu_g1_add(
        cpu_g1_from_native(opening.xi), cpu_g1_from_native(opening.tau));
    Transcript transcript("hyrax-appendix-a3-v3");
    if (public_evaluation != nullptr)
        cpu_hyrax_v3_absorb_public_evaluation(transcript, *public_evaluation);
    cpu_hyrax_v3_absorb_header(transcript, opening);
    const std::size_t fold_count = opening.rounds.size() / 2U;
    std::size_t expected_fold_count = 0U;
    for (std::uint32_t remaining = opening.padded_generator_size;
         remaining > 1U; remaining >>= 1U) {
        ++expected_fold_count;
    }
    if (fold_count != expected_fold_count) {
        if (error) *error = "Hyrax v3 round count does not match padded basis";
        return false;
    }
    std::vector<FrWire> fold_challenges;
    std::vector<FrWire> fold_inverses;
    fold_challenges.reserve(fold_count);
    fold_inverses.reserve(fold_count);
    for (std::size_t round = 0U; round < fold_count; ++round) {
        const G1Jacobian_t& left_native = opening.rounds[2U * round];
        const G1Jacobian_t& right_native = opening.rounds[2U * round + 1U];
        cpu_hyrax_v3_absorb_round(transcript, left_native, right_native);
        const FrWire challenge = cpu_ipa_nonzero_challenge(transcript, "hyrax.v3.fold.challenge");
        const FrWire inverse = cpu_fr_inv(challenge);
        fold_challenges.push_back(challenge);
        fold_inverses.push_back(inverse);
        const FrWire challenge_sq = fr_mul(challenge, challenge);
        const FrWire inverse_sq = fr_mul(inverse, inverse);
        current_upsilon = cpu_g1_add(current_upsilon,
                                     cpu_g1_mul(cpu_g1_from_native(left_native), challenge_sq));
        current_upsilon = cpu_g1_add(current_upsilon,
                                     cpu_g1_mul(cpu_g1_from_native(right_native), inverse_sq));
    }
    const FrWire b_hat = cpu_ipa_terminal_mle_weight(
        expected_point, fold_challenges, fold_inverses);
    if (generator_count > std::numeric_limits<std::size_t>::max() / 2U ||
        padded_size != 2U * generator_count) {
        if (error) *error = "Hyrax v3 padded generator basis shape mismatch";
        return false;
    }
    const std::vector<FrWire> generator_weights =
        cpu_ipa_hyrax_verifier_generator_weights(
            fold_challenges, fold_inverses, generator_count, padded_size);
    if (generator_weights.size() != generator_count) {
        if (error) *error = "Hyrax v3 folded generator coefficient shape mismatch";
        return false;
    }
    const CpuG1Point g_hat = msm_session != nullptr && authenticated_basis != nullptr
        ? msm_session->sum(*authenticated_basis, generators, blinding_generator,
                           u_generator, generator_weights.data(), generator_count)
        : cpu_g1_sum(generators.data(), generator_weights.data(), generator_count);
    if (fr_compare(b_hat, FrWire{}) == 0) {
        if (error) *error = "BLOCKED: Hyrax v3 terminal weight is zero";
        return false;
    }
    cpu_hyrax_v3_absorb_terminal(transcript, opening.delta, opening.beta);
    const FrWire final_challenge = cpu_ipa_nonzero_challenge(
        transcript, "hyrax.v3.terminal.challenge");
    // Fennel short terminal: b_hat*(c*Upsilon_hat + beta)+delta equals
    // z1*(G_hat+b_hat*U)+z2*H.  The verifier derives b_hat/G_hat and never
    // accepts a prover-supplied folded scalar or blinder.
    CpuG1Point lhs = cpu_g1_mul(
        cpu_g1_add(cpu_g1_mul(current_upsilon, final_challenge),
                   cpu_g1_from_native(opening.beta)), b_hat);
    lhs = cpu_g1_add(lhs, cpu_g1_from_native(opening.delta));
    const CpuG1Point rhs = cpu_g1_add(
        cpu_g1_mul(cpu_g1_add(g_hat, cpu_g1_mul(u_generator, b_hat)),
                   fr_wire(opening.z1)),
        cpu_g1_mul(blinding_generator, fr_wire(opening.z2)));
    if (!cpu_g1_equal(lhs, rhs)) {
        if (error) *error = "Hyrax v3 terminal Sigma equation mismatch";
        return false;
    }
    return true;
}

inline bool cpu_verify_hyrax_v3(
    const HyraxOpeningProofV3& opening, const CpuG1Point& expected_xi,
    const CpuG1Point& expected_tau, const std::vector<CpuG1Point>& generators,
    const std::vector<FrWire>& expected_point, const CpuG1Point& blinding_generator,
    const CpuG1Point& u_generator, const CpuIpaOpeningMetadata& expected_metadata,
    std::string* error = nullptr) {
    return cpu_verify_hyrax_v3_impl(opening, expected_xi, expected_tau, generators,
                                    expected_point, blinding_generator, u_generator,
                                    expected_metadata, nullptr, error);
}

inline bool cpu_verify_hyrax_v3_authenticated(
    const HyraxOpeningProofV3& opening, const CpuG1Point& expected_xi,
    const CpuG1Point& expected_tau, const CpuIpaBasisContext& basis,
    const std::vector<FrWire>& expected_point,
    const CpuIpaOpeningMetadata& expected_metadata,
    std::string* error = nullptr, CpuG1MsmSession* msm_session = nullptr) {
    if (!basis.valid()) {
        if (error) *error = "Hyrax v3 authenticated basis context is unavailable";
        return false;
    }
    const auto verify_with_session = [&](CpuG1MsmSession* session) {
        return cpu_verify_hyrax_v3_impl(
            opening, expected_xi, expected_tau, basis.generators(), expected_point,
            basis.h(), basis.u(), expected_metadata, nullptr, error, &basis,
            opening.generator_size, session);
    };
    if (msm_session != nullptr) return verify_with_session(msm_session);
    CpuG1MsmSession local_session(basis);
    return verify_with_session(&local_session);
}

inline bool cpu_verify_hyrax_v3_public(
    const HyraxOpeningProofV3& opening, const CpuG1Point& expected_xi,
    const FrWire& expected_evaluation, const std::vector<CpuG1Point>& generators,
    const std::vector<FrWire>& expected_point, const CpuG1Point& blinding_generator,
    const CpuG1Point& u_generator, const CpuIpaOpeningMetadata& expected_metadata,
    std::string* error = nullptr) {
    if (!fr_canonical(expected_evaluation)) {
        if (error) *error = "Hyrax v3 public evaluation is not canonical";
        return false;
    }
    const CpuG1Point expected_tau = cpu_g1_mul(u_generator, expected_evaluation);
    return cpu_verify_hyrax_v3_impl(opening, expected_xi, expected_tau, generators,
                                    expected_point, blinding_generator, u_generator,
                                    expected_metadata, &expected_evaluation, error);
}

inline bool hyrax_verify_v3(
    const HyraxOpeningProofV3& opening, const CpuG1Point& expected_xi,
    const CpuG1Point& expected_tau, const std::vector<CpuG1Point>& generators,
    const std::vector<FrWire>& expected_point, const CpuG1Point& blinding_generator,
    const CpuG1Point& u_generator, const CpuIpaOpeningMetadata& expected_metadata,
    std::string* error = nullptr) {
    return cpu_verify_hyrax_v3(opening, expected_xi, expected_tau, generators,
                               expected_point, blinding_generator, u_generator,
                               expected_metadata, error);
}

inline void cpu_ipa_absorb_header(Transcript& transcript,
                                  const EvaluationOpeningProof& opening) {
    std::vector<byte> header;
    append_u32(header, opening.generator_size);
    append_u32(header, opening.padded_generator_size);
    append_u64(header, static_cast<std::uint64_t>(opening.inner_point.size()));
    for (const auto& point : opening.inner_point) append_fr(header, point);
    append_u32(header, opening.commitment_index);
    append_u64(header, static_cast<std::uint64_t>(opening.outer_point.size()));
    for (const auto& point : opening.outer_point) append_fr(header, point);
    append_fr(header, opening.value);
    append_g1(header, opening.commitment_eval);
    append_g1(header, opening.blinding_generator);
    append_g1(header, opening.u_generator);
    header.insert(header.end(), opening.basis_digest.begin(), opening.basis_digest.end());
    header.insert(header.end(), opening.descriptor_digest.begin(), opening.descriptor_digest.end());
    header.insert(header.end(), opening.registry_digest.begin(), opening.registry_digest.end());
    header.insert(header.end(), opening.srs_digest.begin(), opening.srs_digest.end());
    transcript.absorb("zkdl.ipa.header", header);
}

inline EvaluationOpeningProof cpu_open_ipa(
    const std::vector<CpuG1Point>& generators, const std::vector<FrWire>& values,
    const std::vector<FrWire>& point, const CpuG1Point& blinding_generator,
    const CpuG1Point& u_generator, const FrWire& randomness,
    const CpuIpaOpeningMetadata& metadata) {
    if (point.size() >= 31U || generators.empty() || generators.size() != values.size() ||
        generators.size() != (size_t(1) << point.size()))
        throw std::invalid_argument("IPA opening shape/basis mismatch");
    for (const auto& value : values) if (!fr_canonical(value))
        throw std::invalid_argument("IPA opening value is non-canonical");
    if (!cpu_ipa_metadata_valid(metadata))
        throw std::invalid_argument("IPA opening descriptor/resource metadata is incomplete");
    for (const auto& coordinate : point) if (!fr_canonical(coordinate))
        throw std::invalid_argument("IPA opening point is non-canonical");
    for (const auto& coordinate : metadata.outer_point) if (!fr_canonical(coordinate))
        throw std::invalid_argument("IPA opening outer point is non-canonical");
    const std::uint32_t generator_size = static_cast<std::uint32_t>(generators.size());
    const std::uint32_t padded_size = cpu_ipa_padded_size(generator_size);
    if (blinding_generator.is_zero() || u_generator.is_zero() ||
        cpu_g1_equal(blinding_generator, u_generator))
        throw std::invalid_argument("IPA requires independent nonzero H and U");

    EvaluationOpeningProof opening;
    opening.generator_size = generator_size;
    opening.padded_generator_size = padded_size;
    opening.commitment_index = metadata.commitment_index;
    for (const auto& coordinate : point) opening.inner_point.push_back(fr_value(coordinate));
    for (const auto& coordinate : metadata.outer_point) opening.outer_point.push_back(fr_value(coordinate));
    opening.blinding_generator = cpu_g1_to_native(blinding_generator);
    opening.u_generator = cpu_g1_to_native(u_generator);
    opening.basis_digest = cpu_ipa_basis_digest(generators, blinding_generator,
                                                u_generator, padded_size);
    opening.descriptor_digest = metadata.descriptor_digest;
    opening.registry_digest = metadata.registry_digest;
    opening.srs_digest = metadata.srs_digest;

    std::vector<CpuG1Point> current_generators(padded_size, CpuG1Point::zero());
    std::vector<FrWire> current_a(padded_size), current_b(padded_size);
    for (std::size_t i = 0; i < generators.size(); ++i) {
        current_generators[i] = generators[i];
        current_a[i] = values[i];
    }
    current_a[generators.size()] = randomness;
    current_generators[generators.size()] = blinding_generator;
    const std::vector<FrWire> mle_weights = cpu_ipa_mle_weights(point);
    if (mle_weights.size() > current_b.size())
        throw std::invalid_argument("IPA MLE basis exceeds padded domain");
    std::copy(mle_weights.begin(), mle_weights.end(), current_b.begin());

    CpuG1Point commitment = cpu_g1_sum(generators, values);
    commitment = cpu_g1_add(commitment, cpu_g1_mul(blinding_generator, randomness));
    opening.commitment_eval = cpu_g1_to_native(commitment);
    std::vector<FrWire> original_b(current_b.begin(),
                                    current_b.begin() + generators.size());
    opening.value = fr_value(cpu_ipa_inner_product(values, original_b));
    if (!fr_canonical(randomness))
        throw std::invalid_argument("IPA blinding scalar is non-canonical");

    Transcript transcript("zkdl-ipa-opening-v2");
    cpu_ipa_absorb_header(transcript, opening);
    CpuG1Point current_p = cpu_g1_add(commitment,
                                      cpu_g1_mul(u_generator, fr_wire(opening.value)));
    while (current_generators.size() > 1U) {
        const std::size_t half = current_generators.size() / 2U;
        std::vector<CpuG1Point> next_generators(half);
        std::vector<FrWire> next_a(half), next_b(half);
        CpuG1Point left = CpuG1Point::zero();
        CpuG1Point right = CpuG1Point::zero();
        for (std::size_t i = 0; i < half; ++i) {
            left = cpu_g1_add(left, cpu_g1_mul(current_generators[half + i], current_a[i]));
            left = cpu_g1_add(left, cpu_g1_mul(u_generator,
                                               fr_mul(current_a[i], current_b[half + i])));
            right = cpu_g1_add(right, cpu_g1_mul(current_generators[i], current_a[half + i]));
            right = cpu_g1_add(right, cpu_g1_mul(u_generator,
                                                 fr_mul(current_a[half + i], current_b[i])));
        }
        opening.rounds.push_back(cpu_g1_to_native(left));
        opening.rounds.push_back(cpu_g1_to_native(right));
        std::vector<byte> round_bytes;
        append_g1(round_bytes, opening.rounds[opening.rounds.size() - 2U]);
        append_g1(round_bytes, opening.rounds.back());
        transcript.absorb("zkdl.ipa.round", round_bytes);
        const FrWire x = cpu_ipa_nonzero_challenge(transcript, "zkdl.ipa.challenge");
        const FrWire x_inv = cpu_fr_inv(x);
        const FrWire x2 = fr_mul(x, x);
        const FrWire x_inv2 = fr_mul(x_inv, x_inv);
        current_p = cpu_g1_add(current_p, cpu_g1_mul(left, x2));
        current_p = cpu_g1_add(current_p, cpu_g1_mul(right, x_inv2));
        for (std::size_t i = 0; i < half; ++i) {
            next_a[i] = fr_add(fr_mul(current_a[i], x),
                               fr_mul(current_a[half + i], x_inv));
            next_b[i] = fr_add(fr_mul(current_b[i], x_inv),
                               fr_mul(current_b[half + i], x));
            next_generators[i] = cpu_g1_add(
                cpu_g1_mul(current_generators[i], x_inv),
                cpu_g1_mul(current_generators[half + i], x));
        }
        current_a.swap(next_a);
        current_b.swap(next_b);
        current_generators.swap(next_generators);
    }
    opening.final_generator = cpu_g1_to_native(current_generators[0]);
    opening.final_a = fr_value(current_a[0]);
    opening.final_b = fr_value(current_b[0]);
    const CpuG1Point expected = cpu_g1_add(
        cpu_g1_mul(current_generators[0], current_a[0]),
        cpu_g1_mul(u_generator, fr_mul(current_a[0], current_b[0])));
    if (!cpu_g1_equal(current_p, expected))
        throw std::runtime_error("IPA prover terminal equation failed");
    return opening;
}

inline EvaluationOpeningProof cpu_open(const std::vector<CpuG1Point>& generators,
                                       const std::vector<FrWire>& values,
                                       const std::vector<FrWire>& point) {
    throw std::invalid_argument("authenticated IPA H/U basis is required");
}

inline EvaluationOpeningProof cpu_open(const std::vector<CpuG1Point>& generators,
                                       const std::vector<FrWire>& values,
                                       const std::vector<FrWire>& point,
                                       const CpuG1Point& blinding_generator,
                                       const CpuG1Point& u_generator,
                                       const FrWire& randomness,
                                       const CpuIpaOpeningMetadata& metadata) {
    return cpu_open_ipa(generators, values, point, blinding_generator,
                        u_generator, randomness, metadata);
}

inline bool cpu_verify_opening(const EvaluationOpeningProof& opening,
                               const CpuG1Point& expected_commitment_eval,
                               const std::vector<CpuG1Point>& generators,
                               const CpuG1Point& blinding_generator,
                               const CpuG1Point& u_generator,
                               const CpuIpaOpeningMetadata& expected_metadata,
                               std::string* error = nullptr) {
    if (!opening_shape_valid(opening, error)) return false;
    const auto strict_wire_point_valid = [&](const G1Jacobian_t& point,
                                             const char* label) {
        std::array<byte, 48U> encoded{};
        std::string point_error;
        if (zkdl_g1_wire::encode(point, encoded, &point_error)) return true;
        if (error) {
            *error = "invalid IPA " + std::string(label) + " point";
            if (!point_error.empty()) *error += ": " + point_error;
        }
        return false;
    };
    if (!strict_wire_point_valid(opening.commitment_eval, "commitment") ||
        !strict_wire_point_valid(opening.blinding_generator, "H") ||
        !strict_wire_point_valid(opening.u_generator, "U") ||
        !strict_wire_point_valid(opening.final_generator, "final")) return false;
    for (const auto& round : opening.rounds)
        if (!strict_wire_point_valid(round, "round")) return false;
    if (generators.size() != opening.generator_size ||
        !cpu_g1_equal(cpu_g1_from_native(opening.commitment_eval), expected_commitment_eval) ||
        !cpu_g1_equal(cpu_g1_from_native(opening.blinding_generator), blinding_generator) ||
        !cpu_g1_equal(cpu_g1_from_native(opening.u_generator), u_generator) ||
        blinding_generator.is_zero() || u_generator.is_zero() ||
        cpu_g1_equal(blinding_generator, u_generator) ||
        !cpu_ipa_metadata_valid(expected_metadata) ||
        opening.commitment_index != expected_metadata.commitment_index ||
        opening.outer_point.size() != expected_metadata.outer_point.size() ||
        opening.descriptor_digest != expected_metadata.descriptor_digest ||
        opening.registry_digest != expected_metadata.registry_digest ||
        opening.srs_digest != expected_metadata.srs_digest) {
        if (error) *error = "IPA opening statement/basis mismatch";
        return false;
    }
    for (std::size_t i = 0U; i < opening.outer_point.size(); ++i)
        if (fr_compare(fr_wire(opening.outer_point[i]), expected_metadata.outer_point[i]) != 0) {
            if (error) *error = "IPA opening outer point metadata mismatch";
            return false;
        }
    const std::uint32_t padded_size = cpu_ipa_padded_size(opening.generator_size);
    if (opening.padded_generator_size != padded_size ||
        opening.basis_digest != cpu_ipa_basis_digest(generators, blinding_generator,
                                                     u_generator, padded_size)) {
        if (error) *error = "IPA generator basis is not authenticated";
        return false;
    }
    std::vector<CpuG1Point> current_generators(padded_size, CpuG1Point::zero());
    std::vector<FrWire> point;
    point.reserve(opening.inner_point.size());
    for (const auto& coordinate : opening.inner_point) point.push_back(fr_wire(coordinate));
    std::vector<FrWire> current_b(padded_size);
    const std::vector<FrWire> mle_weights = cpu_ipa_mle_weights(point);
    if (mle_weights.size() > current_b.size()) {
        if (error) *error = "IPA MLE basis exceeds padded domain";
        return false;
    }
    std::copy(mle_weights.begin(), mle_weights.end(), current_b.begin());
    for (std::size_t i = 0; i < generators.size(); ++i) current_generators[i] = generators[i];
    current_generators[generators.size()] = blinding_generator;
    CpuG1Point current_p = cpu_g1_add(
        cpu_g1_from_native(opening.commitment_eval),
        cpu_g1_mul(u_generator, fr_wire(opening.value)));
    Transcript transcript("zkdl-ipa-opening-v2");
    cpu_ipa_absorb_header(transcript, opening);
    std::size_t folding_rounds = 0U;
    for (std::uint32_t remaining = padded_size; remaining > 1U; remaining >>= 1U)
        ++folding_rounds;
    const std::size_t expected_rounds = 2U * folding_rounds;
    if (opening.rounds.size() != expected_rounds) {
        if (error) *error = "IPA opening round count mismatch";
        return false;
    }
    for (std::size_t round = 0; round < folding_rounds; ++round) {
        const G1Jacobian_t& left_native = opening.rounds[2U * round];
        const G1Jacobian_t& right_native = opening.rounds[2U * round + 1U];
        const CpuG1Point left = cpu_g1_from_native(left_native);
        const CpuG1Point right = cpu_g1_from_native(right_native);
        std::vector<byte> round_bytes;
        append_g1(round_bytes, left_native);
        append_g1(round_bytes, right_native);
        transcript.absorb("zkdl.ipa.round", round_bytes);
        const FrWire x = cpu_ipa_nonzero_challenge(transcript, "zkdl.ipa.challenge");
        const FrWire x_inv = cpu_fr_inv(x);
        current_p = cpu_g1_add(current_p, cpu_g1_mul(left, fr_mul(x, x)));
        current_p = cpu_g1_add(current_p, cpu_g1_mul(right, fr_mul(x_inv, x_inv)));
        const std::size_t half = current_generators.size() / 2U;
        std::vector<CpuG1Point> next_generators(half);
        std::vector<FrWire> next_b(half);
        for (std::size_t i = 0; i < half; ++i) {
            next_b[i] = fr_add(fr_mul(current_b[i], x_inv),
                               fr_mul(current_b[half + i], x));
            next_generators[i] = cpu_g1_add(
                cpu_g1_mul(current_generators[i], x_inv),
                cpu_g1_mul(current_generators[half + i], x));
        }
        current_b.swap(next_b);
        current_generators.swap(next_generators);
    }
    const FrWire final_a = fr_wire(opening.final_a);
    const FrWire final_b = fr_wire(opening.final_b);
    if (fr_compare(final_b, current_b[0]) != 0) {
        if (error) *error = "IPA final b does not match verifier folded weight";
        return false;
    }
    const CpuG1Point expected = cpu_g1_add(
        cpu_g1_mul(current_generators[0], final_a),
        cpu_g1_mul(u_generator, fr_mul(final_a, final_b)));
    const bool final_generator_matches = cpu_g1_equal(
        cpu_g1_from_native(opening.final_generator), current_generators[0]);
    const bool terminal_point_matches = cpu_g1_equal(current_p, expected);
    if (!final_generator_matches || !terminal_point_matches) {
        if (error) {
            if (!final_generator_matches) *error = "IPA final generator mismatch";
            else *error = "IPA terminal P equation mismatch";
        }
        return false;
    }
    return true;
}

inline bool cpu_verify_opening(const EvaluationOpeningProof& opening,
                               const CpuG1Point& expected_commitment_eval,
                               std::string* error = nullptr) {
    if (error) *error = "authenticated IPA G/H/U basis is required";
    return false;
}

/* Explicit hiding commitment metadata.  The secret randomness is retained
 * only by the prover object; the serialized CommitmentRecord contains the
 * value commitment, blinding generator, full commitment and a Schnorr proof
 * of knowledge of the blinder. */
class CpuHidingCommitment {
public:
    std::string label;
    std::uint32_t visibility;
    std::vector<CpuG1Point> generators;
    CpuG1Point blinding_generator;
    std::vector<FrWire> values;
    FrWire randomness;
    FrWire schnorr_nonce;
    CpuG1Point value_commitment;
    CpuG1Point commitment;
    CpuG1Point schnorr_R;
    FrWire schnorr_response;

    CpuHidingCommitment(const std::string& label_, std::uint32_t visibility_,
                       const std::vector<CpuG1Point>& generators_,
                       const CpuG1Point& blinding_generator_,
                       const std::vector<FrWire>& values_, const FrWire& randomness_,
                       const FrWire& schnorr_nonce_)
        : label(label_), visibility(visibility_), generators(generators_),
          blinding_generator(blinding_generator_), values(values_), randomness(randomness_),
          schnorr_nonce(schnorr_nonce_), value_commitment(cpu_opening_commitment_eval(generators_, values_)),
          commitment(), schnorr_R(cpu_g1_mul(blinding_generator_, schnorr_nonce_)), schnorr_response{} {
        if (label.empty() || (visibility != 1U && visibility != 2U) ||
            generators.empty() || blinding_generator.is_zero() ||
            !fr_canonical(randomness) || fr_compare(randomness, FrWire{}) == 0 ||
            !fr_canonical(schnorr_nonce) || fr_compare(schnorr_nonce, FrWire{}) == 0)
            throw std::invalid_argument("invalid private hiding commitment inputs");
        commitment = cpu_g1_add(value_commitment, cpu_g1_mul(blinding_generator, randomness));
        const FrWire challenge = hiding_challenge(label, visibility, value_commitment,
                                                  blinding_generator, commitment, schnorr_R);
        schnorr_response = fr_add(schnorr_nonce, fr_mul(challenge, randomness));
    }

    static CpuHidingCommitment fresh(
        const std::string& label_, std::uint32_t visibility_,
        const std::vector<CpuG1Point>& generators_,
        const CpuG1Point& blinding_generator_,
        const std::vector<FrWire>& values_) {
        return CpuHidingCommitment(label_, visibility_, generators_,
                                   blinding_generator_, values_,
                                   cpu_fresh_csprng_scalar(),
                                   cpu_fresh_csprng_scalar());
    }

    static FrWire hiding_challenge(const std::string& label_, std::uint32_t visibility_,
                                   const CpuG1Point& value_commitment_,
                                   const CpuG1Point& blinding_generator_,
                                   const CpuG1Point& commitment_, const CpuG1Point& schnorr_R_) {
        Transcript transcript("zkdl-hiding-commitment-v1");
        std::vector<byte> message;
        append_u32(message, visibility_);
        append_g1(message, cpu_g1_to_native(value_commitment_));
        append_g1(message, cpu_g1_to_native(blinding_generator_));
        append_g1(message, cpu_g1_to_native(commitment_));
        append_g1(message, cpu_g1_to_native(schnorr_R_));
        transcript.absorb(label_, message);
        return fr_wire(transcript.challenge(label_ + ".schnorr"));
    }

    CommitmentRecord record() const {
        CommitmentRecord out;
        out.label = label;
        out.visibility = visibility;
        out.points.reserve(generators.size());
        for (size_t i = 0; i < generators.size(); ++i) out.points.push_back(cpu_g1_to_native(generators[i]));
        out.blinding_generator = cpu_g1_to_native(blinding_generator);
        out.commitment = cpu_g1_to_native(commitment);
        return out;
    }

    static bool verify_record(const CommitmentRecord& record, std::string* error = nullptr) {
        if (record.label.empty() || (record.visibility != 1U && record.visibility != 2U) ||
            record.points.empty()) {
            if (error) *error = "invalid hiding commitment metadata";
            return false;
        }
        const CpuG1Point h = cpu_g1_from_native(record.blinding_generator);
        const CpuG1Point commitment = cpu_g1_from_native(record.commitment);
        if (h.is_zero() || commitment.is_zero()) {
            if (error) *error = "zero hiding generator or commitment";
            return false;
        }
        std::array<byte, 48U> encoded{};
        std::string point_error;
        if (!zkdl_g1_wire::encode(record.blinding_generator, encoded, &point_error) ||
            !zkdl_g1_wire::encode(record.commitment, encoded, &point_error)) {
            if (error) *error = point_error.empty() ? "invalid hiding commitment point" : point_error;
            return false;
        }
        for (const auto& point : record.points)
            if (!zkdl_g1_wire::encode(point, encoded, &point_error)) {
                if (error) *error = point_error.empty() ? "invalid hiding basis point" : point_error;
                return false;
            }
        return true;
    }
};

inline bool validate_hiding_commitments(const std::vector<CommitmentRecord>& records,
                                        std::string* error = nullptr) {
    std::vector<std::string> labels;
    bool have_model = false, have_dynamic = false;
    for (size_t i = 0; i < records.size(); ++i) {
        if (std::find(labels.begin(), labels.end(), records[i].label) != labels.end()) {
            if (error) *error = "duplicate commitment label";
            return false;
        }
        labels.push_back(records[i].label);
        if (!CpuHidingCommitment::verify_record(records[i], error)) return false;
        have_model = have_model || records[i].visibility == 1U;
        have_dynamic = have_dynamic || records[i].visibility == 2U;
    }
    (void)have_model; (void)have_dynamic;
    return true;
}

} // namespace zkdl

#endif
