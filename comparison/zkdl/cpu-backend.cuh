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

#if defined(__linux__)
#include <cerrno>
#include <sys/random.h>
#endif

#include <mutex>
#include <memory>

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

inline size_t cpu_g1_msm_workspace_bound_bytes() {
#if defined(ZKDL_USE_BLST_G1_CODEC)
    const size_t scratch_bytes =
        zkdl_blst::blst_p1s_mult_pippenger_scratch_sizeof(kCpuG1MsmChunkPoints);
    if (scratch_bytes > std::numeric_limits<size_t>::max() -
                            kCpuG1MsmChunkPoints *
                                (sizeof(zkdl_blst::blst_p1_affine) + sizeof(const zkdl_blst::byte*))) {
        throw std::overflow_error("BLST G1 MSM workspace size overflow");
    }
    return scratch_bytes + kCpuG1MsmChunkPoints *
                               (sizeof(zkdl_blst::blst_p1_affine) + sizeof(const zkdl_blst::byte*));
#else
    return 0U;
#endif
}

inline CpuG1Point cpu_g1_sum(const std::vector<CpuG1Point>& points,
                             const std::vector<FrWire>& scalars) {
    if (points.size() != scalars.size()) throw std::invalid_argument("G1 MSM shape mismatch");
#if defined(ZKDL_USE_BLST_G1_CODEC)
    if (points.empty()) return CpuG1Point::zero();
    if (cpu_g1_msm_workspace_bound_bytes() > kCpuG1MsmProfileBytes) {
        throw std::runtime_error("BLST G1 MSM exceeds its source-bound workspace profile");
    }
    CpuG1Point result = CpuG1Point::zero();
    std::unique_ptr<zkdl_blst::blst_p1_affine[]> affine(
        new zkdl_blst::blst_p1_affine[kCpuG1MsmChunkPoints]);
    std::unique_ptr<const zkdl_blst::byte*[]> scalar_ptrs(
        new const zkdl_blst::byte*[kCpuG1MsmChunkPoints]);
    const size_t scratch_bytes =
        zkdl_blst::blst_p1s_mult_pippenger_scratch_sizeof(kCpuG1MsmChunkPoints);
    if (scratch_bytes == 0U || scratch_bytes % sizeof(zkdl_blst::limb_t) != 0U) {
        throw std::runtime_error("BLST returned an invalid Pippenger scratch bound");
    }
    std::unique_ptr<zkdl_blst::limb_t[]> scratch(
        new zkdl_blst::limb_t[scratch_bytes / sizeof(zkdl_blst::limb_t)]);
    const zkdl_blst::blst_p1_affine* affine_sources[2] = {affine.get(), nullptr};
    for (size_t offset = 0U; offset < points.size();) {
        const size_t count = std::min(kCpuG1MsmChunkPoints, points.size() - offset);
        for (size_t i = 0U; i < count; ++i) {
            if (!fr_canonical(scalars[offset + i])) {
                throw std::invalid_argument("non-canonical G1 MSM scalar");
            }
            const zkdl_blst::blst_p1 projective = cpu_g1_to_blst(points[offset + i]);
            zkdl_blst::blst_p1_to_affine(&affine[i], &projective);
            scalar_ptrs[i] = reinterpret_cast<const zkdl_blst::byte*>(scalars[offset + i].data());
        }
        zkdl_blst::blst_p1 partial{};
        zkdl_blst::blst_p1s_mult_pippenger(
            &partial, affine_sources, count, scalar_ptrs.get(), sizeof(FrWire) * 8U,
            scratch.get());
        result = cpu_g1_add(result, cpu_g1_from_blst(partial));
        offset += count;
    }
    return result;
#else
    CpuG1Point result = CpuG1Point::zero();
    for (size_t i = 0; i < points.size(); ++i) {
        if (!fr_canonical(scalars[i])) throw std::invalid_argument("non-canonical G1 MSM scalar");
        result = cpu_g1_add(result, cpu_g1_mul_portable(points[i], scalars[i]));
    }
    return result;
#endif
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

inline FrWire cpu_ipa_inner_product(const std::vector<FrWire>& left,
                                    const std::vector<FrWire>& right) {
    if (left.size() != right.size()) throw std::invalid_argument("IPA inner-product shape mismatch");
    FrWire result{};
    for (size_t i = 0; i < left.size(); ++i)
        result = fr_add(result, fr_mul(left[i], right[i]));
    return result;
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

inline HyraxOpeningProverResult cpu_open_hyrax_v3_impl(
    const std::vector<CpuG1Point>& generators, const std::vector<FrWire>& values,
    const std::vector<FrWire>& point, const CpuG1Point& blinding_generator,
    const CpuG1Point& u_generator, const FrWire& randomness,
    const FrWire& tau_blinder, const CpuG1Point& expected_tau,
    const CpuIpaOpeningMetadata& metadata, bool public_evaluation) {
    if (point.size() >= 31U || generators.empty() || generators.size() != values.size() ||
        generators.size() != (size_t(1U) << point.size()) ||
        !fr_canonical(randomness) || !fr_canonical(tau_blinder) ||
        !cpu_ipa_v3_metadata_valid(metadata))
        throw std::invalid_argument("Hyrax v3 opening shape/metadata is invalid");
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
    const std::uint32_t generator_size = static_cast<std::uint32_t>(generators.size());
    const std::uint32_t padded_size = cpu_ipa_padded_size(generator_size);
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
    opening.basis_digest = cpu_ipa_basis_digest(generators, blinding_generator,
                                                u_generator, padded_size);
    opening.descriptor_digest = metadata.descriptor_digest;
    opening.registry_digest = metadata.registry_digest;
    opening.srs_digest = metadata.srs_digest;

    const CpuG1Point xi = cpu_g1_add(cpu_g1_sum(generators, values),
                                     cpu_g1_mul(blinding_generator, randomness));
    const FrWire evaluation = cpu_ipa_inner_product(values, cpu_ipa_mle_weights(point));
    result.tau_blinder = public_evaluation ? FrWire{} : tau_blinder;
    const CpuG1Point tau = cpu_g1_add(
        cpu_g1_mul(u_generator, evaluation),
        cpu_g1_mul(blinding_generator, result.tau_blinder));
    if (!cpu_g1_equal(tau, expected_tau))
        throw std::invalid_argument("Hyrax v3 computed tau does not match expected tau");
    opening.xi = cpu_g1_to_native(xi);
    opening.tau = cpu_g1_to_native(tau);

    std::vector<CpuG1Point> current_generators(padded_size, CpuG1Point::zero());
    std::vector<FrWire> current_a(padded_size), current_b(padded_size);
    for (std::size_t i = 0U; i < generators.size(); ++i) {
        current_generators[i] = generators[i];
        current_a[i] = values[i];
    }
    const std::vector<FrWire> weights = cpu_ipa_mle_weights(point);
    std::copy(weights.begin(), weights.end(), current_b.begin());
    CpuG1Point current_upsilon = cpu_g1_add(xi, tau);
    FrWire current_blinder = fr_add(randomness, result.tau_blinder);

    Transcript transcript("hyrax-appendix-a3-v3");
    if (public_evaluation) cpu_hyrax_v3_absorb_public_evaluation(transcript, evaluation);
    cpu_hyrax_v3_absorb_header(transcript, opening);
    while (current_generators.size() > 1U) {
        const std::size_t half = current_generators.size() / 2U;
        CpuG1Point left = CpuG1Point::zero();
        CpuG1Point right = CpuG1Point::zero();
        for (std::size_t i = 0U; i < half; ++i) {
            left = cpu_g1_add(left, cpu_g1_mul(current_generators[half + i], current_a[i]));
            left = cpu_g1_add(left, cpu_g1_mul(u_generator,
                                               fr_mul(current_a[i], current_b[half + i])));
            right = cpu_g1_add(right, cpu_g1_mul(current_generators[i], current_a[half + i]));
            right = cpu_g1_add(right, cpu_g1_mul(u_generator,
                                                 fr_mul(current_a[half + i], current_b[i])));
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
        std::vector<CpuG1Point> next_generators(half);
        std::vector<FrWire> next_a(half), next_b(half);
        for (std::size_t i = 0U; i < half; ++i) {
            next_a[i] = fr_add(fr_mul(current_a[i], challenge),
                               fr_mul(current_a[half + i], inverse));
            next_b[i] = fr_add(fr_mul(current_b[i], inverse),
                               fr_mul(current_b[half + i], challenge));
            next_generators[i] = cpu_g1_add(
                cpu_g1_mul(current_generators[i], inverse),
                cpu_g1_mul(current_generators[half + i], challenge));
        }
        current_a.swap(next_a);
        current_b.swap(next_b);
        current_generators.swap(next_generators);
    }

    // Fennel's short terminal Sigma equation proves the folded scalar while
    // keeping a_hat and the accumulated blinder hidden.  The public folded
    // weight b_hat is verifier-derived and must be nonzero for this form.
    const FrWire a_hat = current_a[0];
    const FrWire b_hat = current_b[0];
    if (fr_compare(b_hat, FrWire{}) == 0)
        throw std::runtime_error("BLOCKED: Hyrax v3 terminal weight is zero");
    const CpuG1Point g_hat = current_generators[0];
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
    std::string* error) {
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
    if (expected_point.size() >= 31U || generators.empty() ||
        generators.size() != (size_t(1U) << expected_point.size()) ||
        opening.generator_size != generators.size() ||
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
    if (opening.padded_generator_size != padded_size ||
        opening.basis_digest != cpu_ipa_basis_digest(generators, blinding_generator,
                                                     u_generator, padded_size)) {
        if (error) *error = "Hyrax v3 generator basis is not authenticated";
        return false;
    }
    std::vector<CpuG1Point> current_generators(padded_size, CpuG1Point::zero());
    for (std::size_t i = 0U; i < generators.size(); ++i) current_generators[i] = generators[i];
    std::vector<FrWire> current_b(padded_size);
    const std::vector<FrWire> weights = cpu_ipa_mle_weights(expected_point);
    std::copy(weights.begin(), weights.end(), current_b.begin());
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
    for (std::size_t round = 0U; round < fold_count; ++round) {
        const G1Jacobian_t& left_native = opening.rounds[2U * round];
        const G1Jacobian_t& right_native = opening.rounds[2U * round + 1U];
        cpu_hyrax_v3_absorb_round(transcript, left_native, right_native);
        const FrWire challenge = cpu_ipa_nonzero_challenge(transcript, "hyrax.v3.fold.challenge");
        const FrWire inverse = cpu_fr_inv(challenge);
        const FrWire challenge_sq = fr_mul(challenge, challenge);
        const FrWire inverse_sq = fr_mul(inverse, inverse);
        current_upsilon = cpu_g1_add(current_upsilon,
                                     cpu_g1_mul(cpu_g1_from_native(left_native), challenge_sq));
        current_upsilon = cpu_g1_add(current_upsilon,
                                     cpu_g1_mul(cpu_g1_from_native(right_native), inverse_sq));
        const std::size_t half = current_generators.size() / 2U;
        std::vector<CpuG1Point> next_generators(half);
        std::vector<FrWire> next_b(half);
        for (std::size_t i = 0U; i < half; ++i) {
            next_b[i] = fr_add(fr_mul(current_b[i], inverse),
                               fr_mul(current_b[half + i], challenge));
            next_generators[i] = cpu_g1_add(
                cpu_g1_mul(current_generators[i], inverse),
                cpu_g1_mul(current_generators[half + i], challenge));
        }
        current_b.swap(next_b);
        current_generators.swap(next_generators);
    }
    const CpuG1Point g_hat = current_generators[0];
    const FrWire b_hat = current_b[0];
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
