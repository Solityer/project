#ifndef PROOF_CUH
#define PROOF_CUH

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "g1-wire.cuh"

#include <vector>
#include <random>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <algorithm>

/*
 * The original zkDL example kept Sumcheck values in an ephemeral vector and
 * discarded both the round messages and the opening messages.  The GAT
 * adaptation needs an actual verifier-facing object, so the small wire
 * protocol below deliberately lives next to the native Sumcheck routines.
 * It is only a serialization/verifier layer around the existing native
 * primitives; it is not a second proving system.
 */
namespace zkdl {

using byte = std::uint8_t;
using FrWire = std::array<std::uint32_t, blstrs__scalar__Scalar_LIMBS>;

static const FrWire kScalarModulus = {{
    1U, 4294967295U, 4294859774U, 1404937218U,
    161601541U, 859428872U, 698187080U, 1944954707U
}};

inline FrWire fr_wire(const Fr_t& value) {
    FrWire out;
    for (uint i = 0; i < blstrs__scalar__Scalar_LIMBS; ++i) out[i] = value.val[i];
    return out;
}

inline Fr_t fr_value(const FrWire& value) {
    Fr_t out{};
    for (uint i = 0; i < blstrs__scalar__Scalar_LIMBS; ++i) out.val[i] = value[i];
    return out;
}

inline int fr_compare(const FrWire& a, const FrWire& b) {
    for (int i = blstrs__scalar__Scalar_LIMBS - 1; i >= 0; --i) {
        if (a[static_cast<size_t>(i)] < b[static_cast<size_t>(i)]) return -1;
        if (a[static_cast<size_t>(i)] > b[static_cast<size_t>(i)]) return 1;
    }
    return 0;
}

inline bool fr_canonical(const FrWire& a) { return fr_compare(a, kScalarModulus) < 0; }

inline FrWire fr_sub_raw(const FrWire& a, const FrWire& b) {
    FrWire out{};
    std::uint64_t borrow = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const std::uint64_t ai = a[i];
        const std::uint64_t bi = static_cast<std::uint64_t>(b[i]) + borrow;
        out[i] = static_cast<std::uint32_t>(ai - bi);
        borrow = ai < bi ? 1U : 0U;
    }
    return out;
}

inline FrWire fr_add(const FrWire& a, const FrWire& b) {
    FrWire out{};
    std::uint64_t carry = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const std::uint64_t sum = static_cast<std::uint64_t>(a[i]) + b[i] + carry;
        out[i] = static_cast<std::uint32_t>(sum);
        carry = sum >> 32;
    }
    if (carry != 0 || fr_compare(out, kScalarModulus) >= 0) out = fr_sub_raw(out, kScalarModulus);
    return out;
}

inline FrWire fr_sub(const FrWire& a, const FrWire& b) {
    if (fr_compare(a, b) >= 0) return fr_sub_raw(a, b);
    FrWire delta = fr_sub_raw(b, a);
    return fr_sub_raw(kScalarModulus, delta);
}

inline FrWire fr_neg(const FrWire& a) {
    FrWire zero{};
    return fr_canonical(a) && fr_compare(a, zero) != 0 ? fr_sub_raw(kScalarModulus, a) : zero;
}

inline FrWire fr_reduce(const std::array<std::uint32_t, 2 * blstrs__scalar__Scalar_LIMBS>& product) {
    // Keep one extra limb while shifting.  A 256-bit remainder can overflow
    // its low limbs during the bit-by-bit reduction even though the reduced
    // value is below p; subtracting p from the truncated low limbs loses the
    // 2^256 carry and produces an invalid field element.
    std::array<std::uint32_t, blstrs__scalar__Scalar_LIMBS + 1> remainder{};
    std::array<std::uint32_t, blstrs__scalar__Scalar_LIMBS + 1> modulus{};
    for (size_t i = 0; i < blstrs__scalar__Scalar_LIMBS; ++i) modulus[i] = kScalarModulus[i];
    for (int bit = static_cast<int>(product.size() * 32) - 1; bit >= 0; --bit) {
        std::uint64_t carry = (product[static_cast<size_t>(bit / 32)] >> (bit % 32)) & 1U;
        for (size_t i = 0; i < remainder.size(); ++i) {
            const std::uint64_t shifted = static_cast<std::uint64_t>(remainder[i]) * 2U + carry;
            remainder[i] = static_cast<std::uint32_t>(shifted);
            carry = shifted >> 32;
        }
        int comparison = remainder.back() == 0 ? 0 : 1;
        if (comparison == 0) {
            for (int i = blstrs__scalar__Scalar_LIMBS - 1; i >= 0; --i) {
                if (remainder[static_cast<size_t>(i)] != modulus[static_cast<size_t>(i)]) {
                    comparison = remainder[static_cast<size_t>(i)] > modulus[static_cast<size_t>(i)] ? 1 : -1;
                    break;
                }
            }
        }
        const bool ge_modulus = comparison >= 0;
        if (ge_modulus) {
            std::uint64_t borrow = 0;
            for (size_t i = 0; i < remainder.size(); ++i) {
                const std::uint64_t subtrahend = static_cast<std::uint64_t>(modulus[i]) + borrow;
                const std::uint64_t value = remainder[i];
                remainder[i] = static_cast<std::uint32_t>(value - subtrahend);
                borrow = value < subtrahend ? 1U : 0U;
            }
        }
    }
    FrWire out{};
    for (size_t i = 0; i < out.size(); ++i) out[i] = remainder[i];
    return out;
}

inline FrWire fr_mul(const FrWire& a, const FrWire& b) {
    std::array<std::uint32_t, 2 * blstrs__scalar__Scalar_LIMBS> product{};
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
    return fr_reduce(product);
}

inline FrWire fr_pow(FrWire base, std::uint32_t exponent_bit_count,
                     const FrWire& exponent) {
    FrWire result{};
    result[0] = 1U;
    for (int bit = static_cast<int>(exponent_bit_count) - 1; bit >= 0; --bit) {
        result = fr_mul(result, result);
        if ((exponent[static_cast<size_t>(bit / 32)] >> (bit % 32)) & 1U)
            result = fr_mul(result, base);
    }
    return result;
}

inline FrWire fr_from_u32(std::uint32_t value) {
    FrWire out{};
    out[0] = value;
    return out;
}

inline FrWire fr_modulus_minus_two() {
    FrWire out = kScalarModulus;
    std::uint64_t borrow = 2U;
    for (size_t i = 0; i < out.size() && borrow != 0; ++i) {
        const std::uint64_t value = out[i];
        out[i] = static_cast<std::uint32_t>(value - borrow);
        borrow = value < borrow ? 1U : 0U;
    }
    return out;
}

inline void append_u32(std::vector<byte>& out, std::uint32_t value) {
    out.push_back(static_cast<byte>(value));
    out.push_back(static_cast<byte>(value >> 8));
    out.push_back(static_cast<byte>(value >> 16));
    out.push_back(static_cast<byte>(value >> 24));
}

inline void append_u64(std::vector<byte>& out, std::uint64_t value) {
    append_u32(out, static_cast<std::uint32_t>(value));
    append_u32(out, static_cast<std::uint32_t>(value >> 32));
}

inline bool read_u32(const std::vector<byte>& in, size_t& offset, std::uint32_t& value) {
    if (offset > in.size() || in.size() - offset < 4) return false;
    value = static_cast<std::uint32_t>(in[offset]) |
            (static_cast<std::uint32_t>(in[offset + 1]) << 8) |
            (static_cast<std::uint32_t>(in[offset + 2]) << 16) |
            (static_cast<std::uint32_t>(in[offset + 3]) << 24);
    offset += 4;
    return true;
}

inline bool read_u64(const std::vector<byte>& in, size_t& offset, std::uint64_t& value) {
    std::uint32_t lo = 0, hi = 0;
    if (!read_u32(in, offset, lo) || !read_u32(in, offset, hi)) return false;
    value = static_cast<std::uint64_t>(lo) | (static_cast<std::uint64_t>(hi) << 32);
    return true;
}

inline void append_fr(std::vector<byte>& out, const Fr_t& value) {
    const FrWire wire = fr_wire(value);
    for (size_t i = 0; i < wire.size(); ++i) append_u32(out, wire[i]);
}

inline bool read_fr(const std::vector<byte>& in, size_t& offset, Fr_t& value,
                    bool require_canonical = true) {
    FrWire wire{};
    for (size_t i = 0; i < wire.size(); ++i) {
        if (!read_u32(in, offset, wire[i])) return false;
    }
    if (require_canonical && !fr_canonical(wire)) return false;
    value = fr_value(wire);
    return true;
}

/* A compact SHA-256 implementation keeps Fiat--Shamir independent of an
 * ambient OpenSSL installation and gives every build the same transcript. */
class Sha256 {
    std::array<std::uint32_t, 8> state_;
    std::array<byte, 64> block_;
    std::uint64_t bit_count_;
    size_t block_size_;

    static std::uint32_t rotr(std::uint32_t x, std::uint32_t n) {
        return (x >> n) | (x << (32U - n));
    }
    static std::uint32_t ch(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return (x & y) ^ (~x & z);
    }
    static std::uint32_t maj(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return (x & y) ^ (x & z) ^ (y & z);
    }
    static std::uint32_t big0(std::uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
    static std::uint32_t big1(std::uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
    static std::uint32_t small0(std::uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
    static std::uint32_t small1(std::uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

    void transform(const byte* data) {
        static const std::uint32_t k[64] = {
            0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
            0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
            0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
            0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
            0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
            0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
            0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
            0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
        };
        std::array<std::uint32_t, 64> w{};
        for (size_t i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(data[4 * i]) << 24) |
                   (static_cast<std::uint32_t>(data[4 * i + 1]) << 16) |
                   (static_cast<std::uint32_t>(data[4 * i + 2]) << 8) |
                   static_cast<std::uint32_t>(data[4 * i + 3]);
        }
        for (size_t i = 16; i < 64; ++i)
            w[i] = small1(w[i - 2]) + w[i - 7] + small0(w[i - 15]) + w[i - 16];
        std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (size_t i = 0; i < 64; ++i) {
            const std::uint32_t t1 = h + big1(e) + ch(e, f, g) + k[i] + w[i];
            const std::uint32_t t2 = big0(a) + maj(a, b, c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

public:
    Sha256()
        : state_{{0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
                  0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U}},
          block_{}, bit_count_(0), block_size_(0) {}

    void update(const byte* data, size_t len) {
        bit_count_ += static_cast<std::uint64_t>(len) * 8U;
        while (len != 0) {
            const size_t take = std::min(len, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, data, take);
            block_size_ += take; data += take; len -= take;
            if (block_size_ == block_.size()) { transform(block_.data()); block_size_ = 0; }
        }
    }
    std::array<byte, 32> final() {
        block_[block_size_++] = 0x80U;
        if (block_size_ > 56) {
            while (block_size_ < 64) block_[block_size_++] = 0;
            transform(block_.data()); block_size_ = 0;
        }
        while (block_size_ < 56) block_[block_size_++] = 0;
        for (int i = 7; i >= 0; --i) block_[block_size_++] = static_cast<byte>(bit_count_ >> (8 * i));
        transform(block_.data());
        std::array<byte, 32> out{};
        for (size_t i = 0; i < state_.size(); ++i) {
            out[4 * i] = static_cast<byte>(state_[i] >> 24);
            out[4 * i + 1] = static_cast<byte>(state_[i] >> 16);
            out[4 * i + 2] = static_cast<byte>(state_[i] >> 8);
            out[4 * i + 3] = static_cast<byte>(state_[i]);
        }
        return out;
    }
};

inline std::array<byte, 32> sha256(const std::vector<byte>& bytes) {
    Sha256 hash;
    if (!bytes.empty()) hash.update(bytes.data(), bytes.size());
    return hash.final();
}

class Transcript {
    std::vector<byte> state_;
    std::uint64_t challenge_counter_;

    void absorb_tag(const char* tag, size_t len) {
        append_u32(state_, static_cast<std::uint32_t>(len));
        state_.insert(state_.end(), tag, tag + len);
    }

public:
    explicit Transcript(const std::string& domain = "zkdl-gat-transcript-v1")
        : state_(), challenge_counter_(0) {
        absorb_tag(domain.data(), domain.size());
    }

    void absorb(const std::string& label, const std::vector<byte>& message) {
        absorb_tag(label.data(), label.size());
        append_u64(state_, static_cast<std::uint64_t>(message.size()));
        state_.insert(state_.end(), message.begin(), message.end());
    }

    void absorb_fr(const std::string& label, const std::vector<Fr_t>& values) {
        std::vector<byte> message;
        append_u64(message, static_cast<std::uint64_t>(values.size()));
        for (size_t i = 0; i < values.size(); ++i) append_fr(message, values[i]);
        absorb(label, message);
    }

    Fr_t challenge(const std::string& label) {
        std::vector<byte> input = state_;
        absorb_tag(label.data(), label.size());
        append_u64(input, challenge_counter_++);
        input.insert(input.end(), label.begin(), label.end());
        const std::array<byte, 32> digest = sha256(input);
        FrWire wire{};
        for (size_t i = 0; i < wire.size(); ++i)
            wire[i] = static_cast<std::uint32_t>(digest[4 * i]) |
                      (static_cast<std::uint32_t>(digest[4 * i + 1]) << 8) |
                      (static_cast<std::uint32_t>(digest[4 * i + 2]) << 16) |
                      (static_cast<std::uint32_t>(digest[4 * i + 3]) << 24);
        while (!fr_canonical(wire)) wire = fr_sub_raw(wire, kScalarModulus);
        const Fr_t result = fr_value(wire);
        append_fr(state_, result);
        return result;
    }

    std::array<byte, 32> digest() const { return sha256(state_); }
    const std::vector<byte>& bytes() const { return state_; }
};

struct SumcheckRound {
    std::vector<Fr_t> evaluations; // g(0), g(1), ..., g(degree)
};

struct SumcheckProof {
    std::string label;
    std::uint32_t degree;
    std::vector<SumcheckRound> rounds;
    std::vector<Fr_t> terminal_claims;
    Fr_t final_claim;

    SumcheckProof() : label(), degree(0), rounds(), terminal_claims(), final_claim{} {}
};

inline void append_string(std::vector<byte>& out, const std::string& value) {
    append_u64(out, static_cast<std::uint64_t>(value.size()));
    out.insert(out.end(), value.begin(), value.end());
}

inline bool read_string(const std::vector<byte>& in, size_t& offset, std::string& value,
                        size_t max_size = (1U << 20)) {
    std::uint64_t length = 0;
    if (!read_u64(in, offset, length) || length > max_size ||
        length > static_cast<std::uint64_t>(in.size() - offset)) return false;
    value.assign(reinterpret_cast<const char*>(in.data() + offset), static_cast<size_t>(length));
    offset += static_cast<size_t>(length);
    return true;
}

inline void serialize_sumcheck(const SumcheckProof& proof, std::vector<byte>& out) {
    append_string(out, proof.label);
    append_u32(out, proof.degree);
    append_u64(out, static_cast<std::uint64_t>(proof.rounds.size()));
    for (size_t i = 0; i < proof.rounds.size(); ++i) {
        append_u64(out, static_cast<std::uint64_t>(proof.rounds[i].evaluations.size()));
        for (size_t j = 0; j < proof.rounds[i].evaluations.size(); ++j)
            append_fr(out, proof.rounds[i].evaluations[j]);
    }
    append_u64(out, static_cast<std::uint64_t>(proof.terminal_claims.size()));
    for (size_t i = 0; i < proof.terminal_claims.size(); ++i) append_fr(out, proof.terminal_claims[i]);
    append_fr(out, proof.final_claim);
}

inline bool deserialize_sumcheck(const std::vector<byte>& in, size_t& offset,
                                 SumcheckProof& proof,
                                 size_t max_rounds = (1U << 20)) {
    std::uint32_t degree = 0;
    std::uint64_t rounds = 0;
    SumcheckProof parsed;
    if (!read_string(in, offset, parsed.label) || !read_u32(in, offset, degree) ||
        !read_u64(in, offset, rounds) || degree == 0 || degree > 16 || rounds > max_rounds)
        return false;
    parsed.degree = degree;
    parsed.rounds.reserve(static_cast<size_t>(rounds));
    for (std::uint64_t i = 0; i < rounds; ++i) {
        std::uint64_t values = 0;
        SumcheckRound round;
        if (!read_u64(in, offset, values) || values != static_cast<std::uint64_t>(degree + 1)) return false;
        round.evaluations.resize(static_cast<size_t>(values));
        for (size_t j = 0; j < round.evaluations.size(); ++j)
            if (!read_fr(in, offset, round.evaluations[j])) return false;
        parsed.rounds.push_back(round);
    }
    std::uint64_t terminals = 0;
    // Native relation families may open more than 64 canonical views (for
    // example a bounded SignedRoundDiv range stack), but the wire remains
    // explicitly capped rather than accepting an unbounded allocation.
    if (!read_u64(in, offset, terminals) || terminals > 4096U) return false;
    parsed.terminal_claims.resize(static_cast<size_t>(terminals));
    for (size_t i = 0; i < parsed.terminal_claims.size(); ++i)
        if (!read_fr(in, offset, parsed.terminal_claims[i])) return false;
    if (!read_fr(in, offset, parsed.final_claim)) return false;
    proof = parsed;
    return true;
}

inline void append_g1(std::vector<byte>& out, const G1Jacobian_t& point) {
    std::array<byte, 48U> encoded{};
    std::string error;
    if (!zkdl_g1_wire::encode(point, encoded, &error))
        throw std::invalid_argument(error.empty() ? "invalid G1 point" : error);
    out.insert(out.end(), encoded.begin(), encoded.end());
}

inline bool read_g1(const std::vector<byte>& in, size_t& offset, G1Jacobian_t& point) {
    if (offset > in.size() || in.size() - offset < 48U) return false;
    std::array<byte, 48U> encoded{};
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + encoded.size()),
              encoded.begin());
    offset += encoded.size();
    return zkdl_g1_wire::decode(encoded, point, nullptr);
}

/* Versioned inner-product opening.  The old P/A/B folding object is not part
 * of the formal wire anymore: deserializers require this IPA magic/version
 * and reject legacy bytes before any point is consumed.  `commitment_eval` is
 * the blinded C=<a,G>+rH itself; there is no serialized bare C0. */
struct EvaluationOpeningProof {
    static const std::uint32_t kVersion = 2U;
    std::uint32_t version;
    std::uint32_t commitment_index;
    std::uint32_t generator_size;
    std::uint32_t padded_generator_size;
    std::vector<Fr_t> inner_point;
    std::vector<Fr_t> outer_point;
    Fr_t value;
    G1Jacobian_t commitment_eval;
    G1Jacobian_t blinding_generator; // authenticated independent H
    G1Jacobian_t u_generator;        // authenticated independent U
    std::array<byte, 32U> basis_digest;
    std::array<byte, 32U> descriptor_digest;
    std::array<byte, 32U> registry_digest;
    std::array<byte, 32U> srs_digest;
    std::vector<G1Jacobian_t> rounds; // L,R for every recursive fold
    G1Jacobian_t final_generator;
    Fr_t final_a;
    Fr_t final_b;

    EvaluationOpeningProof()
        : version(kVersion), commitment_index(std::numeric_limits<std::uint32_t>::max()), generator_size(0),
          padded_generator_size(0), inner_point(), outer_point(), value{}, commitment_eval{},
          blinding_generator{}, u_generator{}, basis_digest{{}}, descriptor_digest{{}},
          registry_digest{{}}, srs_digest{{}}, rounds(), final_generator{},
          final_a{}, final_b{} {}
};

// The public name of the version-2 opening protocol.  Keep the historical
// type spelling as an ABI-compatible alias for the native zkMatMul/zkProd
// wrappers while making it explicit that no legacy P/A/B opening is accepted.
using IpaOpeningProof = EvaluationOpeningProof;

inline void serialize_opening(const EvaluationOpeningProof& opening, std::vector<byte>& out) {
    out.push_back('I'); out.push_back('P'); out.push_back('A'); out.push_back('1');
    append_u32(out, EvaluationOpeningProof::kVersion);
    append_u32(out, opening.commitment_index);
    append_u32(out, opening.generator_size);
    append_u32(out, opening.padded_generator_size);
    append_u64(out, static_cast<std::uint64_t>(opening.inner_point.size()));
    for (size_t i = 0; i < opening.inner_point.size(); ++i) append_fr(out, opening.inner_point[i]);
    append_u64(out, static_cast<std::uint64_t>(opening.outer_point.size()));
    for (size_t i = 0; i < opening.outer_point.size(); ++i) append_fr(out, opening.outer_point[i]);
    append_fr(out, opening.value);
    append_g1(out, opening.commitment_eval);
    append_g1(out, opening.blinding_generator);
    append_g1(out, opening.u_generator);
    out.insert(out.end(), opening.basis_digest.begin(), opening.basis_digest.end());
    out.insert(out.end(), opening.descriptor_digest.begin(), opening.descriptor_digest.end());
    out.insert(out.end(), opening.registry_digest.begin(), opening.registry_digest.end());
    out.insert(out.end(), opening.srs_digest.begin(), opening.srs_digest.end());
    append_u64(out, static_cast<std::uint64_t>(opening.rounds.size()));
    for (size_t i = 0; i < opening.rounds.size(); ++i) append_g1(out, opening.rounds[i]);
    append_g1(out, opening.final_generator);
    append_fr(out, opening.final_a);
    append_fr(out, opening.final_b);
}

inline bool deserialize_opening(const std::vector<byte>& in, size_t& offset,
                                EvaluationOpeningProof& opening,
                                size_t max_point = 4096, size_t max_round_points = (1U << 20)) {
    EvaluationOpeningProof parsed;
    std::uint64_t points = 0;
    if (offset > in.size() || in.size() - offset < 4U ||
        in[offset] != 'I' || in[offset + 1U] != 'P' ||
        in[offset + 2U] != 'A' || in[offset + 3U] != '1') return false;
    offset += 4U;
    std::uint32_t version = 0U;
    if (!read_u32(in, offset, version) || version != EvaluationOpeningProof::kVersion ||
        !read_u32(in, offset, parsed.commitment_index) ||
        !read_u32(in, offset, parsed.generator_size) ||
        !read_u32(in, offset, parsed.padded_generator_size) ||
        !read_u64(in, offset, points) || points > max_point) return false;
    parsed.version = version;
    parsed.inner_point.resize(static_cast<size_t>(points));
    for (size_t i = 0; i < parsed.inner_point.size(); ++i)
        if (!read_fr(in, offset, parsed.inner_point[i])) return false;
    if (!read_u64(in, offset, points) || points > max_point) return false;
    parsed.outer_point.resize(static_cast<size_t>(points));
    for (size_t i = 0; i < parsed.outer_point.size(); ++i)
        if (!read_fr(in, offset, parsed.outer_point[i])) return false;
    if (!read_fr(in, offset, parsed.value) || !read_g1(in, offset, parsed.commitment_eval) ||
        !read_g1(in, offset, parsed.blinding_generator) ||
        !read_g1(in, offset, parsed.u_generator) ||
        in.size() - offset < parsed.basis_digest.size()) return false;
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + parsed.basis_digest.size()),
              parsed.basis_digest.begin());
    offset += parsed.basis_digest.size();
    if (in.size() - offset < parsed.descriptor_digest.size() +
        parsed.registry_digest.size() + parsed.srs_digest.size()) return false;
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + parsed.descriptor_digest.size()),
              parsed.descriptor_digest.begin());
    offset += parsed.descriptor_digest.size();
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + parsed.registry_digest.size()),
              parsed.registry_digest.begin());
    offset += parsed.registry_digest.size();
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + parsed.srs_digest.size()),
              parsed.srs_digest.begin());
    offset += parsed.srs_digest.size();
    if (!read_u64(in, offset, points) || points > max_round_points ||
        (points & 1U) != 0U) return false;
    parsed.rounds.resize(static_cast<size_t>(points));
    for (size_t i = 0; i < parsed.rounds.size(); ++i)
        if (!read_g1(in, offset, parsed.rounds[i])) return false;
    if (!read_g1(in, offset, parsed.final_generator) ||
        !read_fr(in, offset, parsed.final_a) || !read_fr(in, offset, parsed.final_b)) return false;
    opening = parsed;
    return true;
}

inline bool opening_shape_valid(const EvaluationOpeningProof& opening, std::string* error = nullptr) {
    if (opening.rounds.size() % 2U != 0U || opening.generator_size == 0U ||
        opening.padded_generator_size == 0U ||
        (opening.padded_generator_size & (opening.padded_generator_size - 1U)) != 0U ||
        opening.padded_generator_size < opening.generator_size + 1U ||
        opening.padded_generator_size > (1U << 20) ||
        opening.inner_point.size() >= 31U ||
        opening.padded_generator_size != (std::size_t(1) << opening.rounds.size() / 2U) ||
        opening.generator_size != (std::size_t(1) << opening.inner_point.size())) {
        if (error) *error = "IPA opening fold/domain shape is invalid";
        return false;
    }
    if (opening.version != EvaluationOpeningProof::kVersion) {
        if (error) *error = "legacy opening version is not accepted";
        return false;
    }
    if (opening.commitment_index == std::numeric_limits<std::uint32_t>::max() ||
        opening.descriptor_digest == std::array<byte, 32U>{} ||
        opening.registry_digest == std::array<byte, 32U>{} ||
        opening.srs_digest == std::array<byte, 32U>{}) {
        if (error) *error = "IPA opening descriptor/resource binding is incomplete";
        return false;
    }
    for (size_t i = 0; i < opening.inner_point.size(); ++i)
        if (!fr_canonical(fr_wire(opening.inner_point[i]))) {
            if (error) *error = "opening point is not canonical";
            return false;
        }
    for (size_t i = 0; i < opening.outer_point.size(); ++i)
        if (!fr_canonical(fr_wire(opening.outer_point[i]))) {
            if (error) *error = "opening outer point is not canonical";
            return false;
        }
    if (!fr_canonical(fr_wire(opening.value)) || !fr_canonical(fr_wire(opening.final_a)) ||
        !fr_canonical(fr_wire(opening.final_b))) {
        if (error) *error = "IPA opening scalar is not canonical";
        return false;
    }
    return true;
}

/*
 * Version-3 Hyrax Appendix-A.3-style logarithmic dot-product opening.
 *
 * This wire is intentionally separate from EvaluationOpeningProof: v2
 * exposes the terminal value/final scalars and is retained only for the
 * not-yet-migrated native wrappers.  v3 carries the blinded commitment pair
 * (xi,tau), blinded L/R fold points, and the final Sigma responses only.  In
 * particular, no y, final_a, final_b, or commitment blinder is serialized.
 */
struct HyraxOpeningProofV3 {
    // Wire revision 4 adds the explicit evaluation commitment role.  Older
    // v3 bytes are intentionally rejected instead of being parsed with an
    // ambiguous role.
    static const std::uint32_t kVersion = 4U;
    std::uint32_t version;
    std::uint32_t commitment_index;
    std::uint32_t evaluation_commitment_index;
    std::uint32_t generator_size;
    std::uint32_t padded_generator_size;
    std::vector<Fr_t> inner_point;
    std::vector<Fr_t> outer_point;
    G1Jacobian_t xi;                 // C=<a,G>+rH
    G1Jacobian_t tau;                // yU+r_tau H
    G1Jacobian_t blinding_generator; // authenticated H
    G1Jacobian_t u_generator;        // authenticated U
    std::array<byte, 32U> basis_digest;
    std::array<byte, 32U> descriptor_digest;
    std::array<byte, 32U> registry_digest;
    std::array<byte, 32U> srs_digest;
    // Alternating blinded L,R points for each recursive fold.
    std::vector<G1Jacobian_t> rounds;
    G1Jacobian_t delta;               // final Sigma commitment
    G1Jacobian_t beta;                // independent final Q-mask commitment
    Fr_t z1;                          // response for the folded vector
    Fr_t z2;                          // response for the aggregate blinder

    HyraxOpeningProofV3()
        : version(kVersion), commitment_index(std::numeric_limits<std::uint32_t>::max()),
          evaluation_commitment_index(std::numeric_limits<std::uint32_t>::max()),
          generator_size(0U), padded_generator_size(0U), inner_point(), outer_point(),
          xi{}, tau{}, blinding_generator{}, u_generator{}, basis_digest{{}},
          descriptor_digest{{}}, registry_digest{{}}, srs_digest{{}}, rounds(),
          delta{}, beta{}, z1{}, z2{} {}
};

inline bool hyrax_opening_v3_shape_valid(const HyraxOpeningProofV3& opening,
                                         std::string* error = nullptr);

inline void serialize_hyrax_opening_v3(const HyraxOpeningProofV3& opening,
                                       std::vector<byte>& out) {
    std::string shape_error;
    if (!hyrax_opening_v3_shape_valid(opening, &shape_error))
        throw std::invalid_argument(shape_error.empty() ? "invalid Hyrax v3 opening" : shape_error);
    out.push_back('H'); out.push_back('Y'); out.push_back('X'); out.push_back('3');
    append_u32(out, HyraxOpeningProofV3::kVersion);
    append_u32(out, opening.commitment_index);
    append_u32(out, opening.evaluation_commitment_index);
    append_u32(out, opening.generator_size);
    append_u32(out, opening.padded_generator_size);
    append_u64(out, static_cast<std::uint64_t>(opening.inner_point.size()));
    for (const auto& coordinate : opening.inner_point) append_fr(out, coordinate);
    append_u64(out, static_cast<std::uint64_t>(opening.outer_point.size()));
    for (const auto& coordinate : opening.outer_point) append_fr(out, coordinate);
    append_g1(out, opening.xi);
    append_g1(out, opening.tau);
    append_g1(out, opening.blinding_generator);
    append_g1(out, opening.u_generator);
    out.insert(out.end(), opening.basis_digest.begin(), opening.basis_digest.end());
    out.insert(out.end(), opening.descriptor_digest.begin(), opening.descriptor_digest.end());
    out.insert(out.end(), opening.registry_digest.begin(), opening.registry_digest.end());
    out.insert(out.end(), opening.srs_digest.begin(), opening.srs_digest.end());
    append_u64(out, static_cast<std::uint64_t>(opening.rounds.size()));
    for (const auto& point : opening.rounds) append_g1(out, point);
    append_g1(out, opening.delta);
    append_g1(out, opening.beta);
    append_fr(out, opening.z1);
    append_fr(out, opening.z2);
}

inline bool deserialize_hyrax_opening_v3(const std::vector<byte>& in, size_t& offset,
                                         HyraxOpeningProofV3& opening,
                                         size_t max_point = 4096U,
                                         size_t max_round_points = 64U) {
    HyraxOpeningProofV3 parsed;
    std::uint64_t points = 0U;
    if (offset > in.size() || in.size() - offset < 4U ||
        in[offset] != 'H' || in[offset + 1U] != 'Y' ||
        in[offset + 2U] != 'X' || in[offset + 3U] != '3') return false;
    offset += 4U;
    std::uint32_t version = 0U;
    if (!read_u32(in, offset, version) || version != HyraxOpeningProofV3::kVersion ||
        !read_u32(in, offset, parsed.commitment_index) ||
        !read_u32(in, offset, parsed.evaluation_commitment_index) ||
        !read_u32(in, offset, parsed.generator_size) ||
        !read_u32(in, offset, parsed.padded_generator_size) ||
        !read_u64(in, offset, points) || points > max_point) return false;
    parsed.version = version;
    parsed.inner_point.resize(static_cast<size_t>(points));
    for (auto& coordinate : parsed.inner_point)
        if (!read_fr(in, offset, coordinate)) return false;
    if (!read_u64(in, offset, points) || points > max_point) return false;
    parsed.outer_point.resize(static_cast<size_t>(points));
    for (auto& coordinate : parsed.outer_point)
        if (!read_fr(in, offset, coordinate)) return false;
    if (!read_g1(in, offset, parsed.xi) || !read_g1(in, offset, parsed.tau) ||
        !read_g1(in, offset, parsed.blinding_generator) ||
        !read_g1(in, offset, parsed.u_generator) ||
        in.size() - offset < parsed.basis_digest.size() + parsed.descriptor_digest.size() +
                              parsed.registry_digest.size() + parsed.srs_digest.size()) return false;
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + parsed.basis_digest.size()),
              parsed.basis_digest.begin());
    offset += parsed.basis_digest.size();
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + parsed.descriptor_digest.size()),
              parsed.descriptor_digest.begin());
    offset += parsed.descriptor_digest.size();
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + parsed.registry_digest.size()),
              parsed.registry_digest.begin());
    offset += parsed.registry_digest.size();
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + parsed.srs_digest.size()),
              parsed.srs_digest.begin());
    offset += parsed.srs_digest.size();
    if (!read_u64(in, offset, points) || points > max_round_points || (points & 1U) != 0U)
        return false;
    parsed.rounds.resize(static_cast<size_t>(points));
    for (auto& point : parsed.rounds)
        if (!read_g1(in, offset, point)) return false;
    if (!read_g1(in, offset, parsed.delta) || !read_g1(in, offset, parsed.beta) ||
        !read_fr(in, offset, parsed.z1) || !read_fr(in, offset, parsed.z2)) return false;
    opening = parsed;
    return true;
}

inline bool hyrax_opening_v3_shape_valid(const HyraxOpeningProofV3& opening,
                                         std::string* error) {
    const auto fail_shape = [&](const char* message) {
        if (error) *error = message;
        return false;
    };
    if (opening.version != HyraxOpeningProofV3::kVersion || opening.generator_size == 0U ||
        opening.inner_point.size() >= 31U || opening.rounds.size() % 2U != 0U ||
        opening.commitment_index == std::numeric_limits<std::uint32_t>::max() ||
        opening.evaluation_commitment_index == std::numeric_limits<std::uint32_t>::max() ||
        opening.evaluation_commitment_index == opening.commitment_index ||
        opening.descriptor_digest == std::array<byte, 32U>{} ||
        opening.registry_digest == std::array<byte, 32U>{} ||
        opening.srs_digest == std::array<byte, 32U>{}) {
        return fail_shape("Hyrax v3 metadata/shape is invalid");
    }
    if (opening.generator_size != (std::size_t(1U) << opening.inner_point.size()) ||
        opening.padded_generator_size == 0U ||
        (opening.padded_generator_size & (opening.padded_generator_size - 1U)) != 0U ||
        opening.padded_generator_size < opening.generator_size + 1U ||
        opening.padded_generator_size > (1U << 19U) ||
        opening.padded_generator_size != (std::size_t(1U) << (opening.rounds.size() / 2U))) {
        return fail_shape("Hyrax v3 padded generator/fold shape is invalid");
    }
    if (!fr_canonical(fr_wire(opening.z1)) || !fr_canonical(fr_wire(opening.z2)))
        return fail_shape("Hyrax v3 scalar response is not canonical");
    for (const auto& coordinate : opening.inner_point)
        if (!fr_canonical(fr_wire(coordinate))) return fail_shape("Hyrax v3 inner point is not canonical");
    for (const auto& coordinate : opening.outer_point)
        if (!fr_canonical(fr_wire(coordinate))) return fail_shape("Hyrax v3 outer point is not canonical");
    return true;
}

/* Versioned Pedersen/Fennel proof wires.  These objects carry only public
 * statement data, commitments, and responses; values, blinders, and nonce
 * scalars remain prover-local.  H/U and the generator basis are supplied by
 * the authenticated verifier context and are deliberately not trusted from
 * the wire. */
struct PedersenProofContext {
    static const std::uint32_t kMaxBasisSize = 4096U;
    static const std::uint32_t kMaxDegree = 64U;
    static const std::uint32_t kUnsetIndex = std::numeric_limits<std::uint32_t>::max();
    static const std::uint32_t kOperationScalarValue = 1U;
    static const std::uint32_t kOperationScalarEquality = 2U;
    static const std::uint32_t kOperationProduct = 3U;
    static const std::uint32_t kOperationVectorLinear = 4U;
    std::uint32_t commitment_index;
    std::uint32_t operation;
    std::uint32_t commitment_a_index;
    std::uint32_t commitment_b_index;
    std::uint32_t commitment_c_index;
    std::uint32_t evaluation_commitment_index;
    std::uint32_t degree;
    std::uint32_t basis_size;
    std::array<byte, 32U> basis_digest;
    std::array<byte, 32U> descriptor_digest;
    std::array<byte, 32U> registry_digest;
    std::array<byte, 32U> srs_digest;

    PedersenProofContext()
        : commitment_index(kUnsetIndex), operation(0U), commitment_a_index(kUnsetIndex),
          commitment_b_index(kUnsetIndex), commitment_c_index(kUnsetIndex),
          evaluation_commitment_index(kUnsetIndex), degree(0U), basis_size(0U),
          basis_digest{{}}, descriptor_digest{{}}, registry_digest{{}}, srs_digest{{}} {}
};

inline bool pedersen_context_shape_valid(const PedersenProofContext& context,
                                         std::string* error = nullptr) {
    const auto fail = [&](const char* message) {
        if (error) *error = message;
        return false;
    };
    if (context.commitment_index == PedersenProofContext::kUnsetIndex ||
        context.operation < PedersenProofContext::kOperationScalarValue ||
        context.operation > PedersenProofContext::kOperationVectorLinear ||
        context.degree == 0U || context.degree > PedersenProofContext::kMaxDegree ||
        context.basis_size > PedersenProofContext::kMaxBasisSize ||
        context.descriptor_digest == std::array<byte, 32U>{} ||
        context.registry_digest == std::array<byte, 32U>{} ||
        context.srs_digest == std::array<byte, 32U>{})
        return fail("Pedersen proof context is incomplete or out of bounds");
    switch (context.operation) {
        case PedersenProofContext::kOperationScalarValue:
            if (context.commitment_a_index != context.commitment_index ||
                context.commitment_b_index != PedersenProofContext::kUnsetIndex ||
                context.commitment_c_index != PedersenProofContext::kUnsetIndex ||
                context.evaluation_commitment_index != PedersenProofContext::kUnsetIndex)
                return fail("Pedersen scalar value roles are invalid");
            break;
        case PedersenProofContext::kOperationScalarEquality:
            if (context.commitment_a_index != context.commitment_index ||
                context.commitment_b_index == PedersenProofContext::kUnsetIndex ||
                context.commitment_b_index == context.commitment_a_index ||
                context.commitment_c_index != PedersenProofContext::kUnsetIndex ||
                context.evaluation_commitment_index != PedersenProofContext::kUnsetIndex)
                return fail("Pedersen equality roles are invalid");
            break;
        case PedersenProofContext::kOperationProduct:
            if (context.commitment_a_index != context.commitment_index ||
                context.commitment_b_index == PedersenProofContext::kUnsetIndex ||
                context.commitment_c_index == PedersenProofContext::kUnsetIndex ||
                context.commitment_b_index == context.commitment_a_index ||
                context.commitment_c_index == context.commitment_a_index ||
                context.commitment_c_index == context.commitment_b_index ||
                context.evaluation_commitment_index != PedersenProofContext::kUnsetIndex)
                return fail("Pedersen product roles are invalid");
            break;
        case PedersenProofContext::kOperationVectorLinear:
            if (context.commitment_a_index != context.commitment_index ||
                context.commitment_b_index != PedersenProofContext::kUnsetIndex ||
                context.commitment_c_index != PedersenProofContext::kUnsetIndex ||
                context.evaluation_commitment_index == PedersenProofContext::kUnsetIndex ||
                context.evaluation_commitment_index == context.commitment_a_index)
                return fail("Pedersen vector roles are invalid");
            break;
        default:
            return fail("unknown Pedersen operation");
    }
    return true;
}

inline void append_pedersen_context(std::vector<byte>& out,
                                    const PedersenProofContext& context) {
    append_u32(out, context.commitment_index);
    append_u32(out, context.operation);
    append_u32(out, context.commitment_a_index);
    append_u32(out, context.commitment_b_index);
    append_u32(out, context.commitment_c_index);
    append_u32(out, context.evaluation_commitment_index);
    append_u32(out, context.degree);
    append_u32(out, context.basis_size);
    out.insert(out.end(), context.basis_digest.begin(), context.basis_digest.end());
    out.insert(out.end(), context.descriptor_digest.begin(), context.descriptor_digest.end());
    out.insert(out.end(), context.registry_digest.begin(), context.registry_digest.end());
    out.insert(out.end(), context.srs_digest.begin(), context.srs_digest.end());
}

inline bool read_pedersen_context(const std::vector<byte>& in, size_t& offset,
                                  PedersenProofContext& context) {
    if (!read_u32(in, offset, context.commitment_index) ||
        !read_u32(in, offset, context.operation) ||
        !read_u32(in, offset, context.commitment_a_index) ||
        !read_u32(in, offset, context.commitment_b_index) ||
        !read_u32(in, offset, context.commitment_c_index) ||
        !read_u32(in, offset, context.evaluation_commitment_index) ||
        !read_u32(in, offset, context.degree) || !read_u32(in, offset, context.basis_size))
        return false;
    const std::size_t digest_bytes = context.basis_digest.size() +
                                     context.descriptor_digest.size() +
                                     context.registry_digest.size() + context.srs_digest.size();
    if (offset > in.size() || in.size() - offset < digest_bytes) return false;
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + context.basis_digest.size()),
              context.basis_digest.begin());
    offset += context.basis_digest.size();
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + context.descriptor_digest.size()),
              context.descriptor_digest.begin());
    offset += context.descriptor_digest.size();
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + context.registry_digest.size()),
              context.registry_digest.begin());
    offset += context.registry_digest.size();
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + context.srs_digest.size()),
              context.srs_digest.begin());
    offset += context.srs_digest.size();
    return pedersen_context_shape_valid(context);
}

struct PedersenScalarValueProof {
    static const std::uint32_t kVersion = 1U;
    std::uint32_t version;
    PedersenProofContext context;
    G1Jacobian_t commitment; // C=vU+rH
    G1Jacobian_t schnorr_R;  // kH
    Fr_t schnorr_s;          // k+e*r

    PedersenScalarValueProof()
        : version(kVersion), context(), commitment{}, schnorr_R{}, schnorr_s{} {}
};

struct PedersenScalarEqualityProof {
    static const std::uint32_t kVersion = 1U;
    std::uint32_t version;
    PedersenProofContext context;
    G1Jacobian_t commitment_1;
    G1Jacobian_t commitment_2;
    G1Jacobian_t schnorr_R;
    Fr_t schnorr_s;

    PedersenScalarEqualityProof()
        : version(kVersion), context(), commitment_1{}, commitment_2{}, schnorr_R{},
          schnorr_s{} {}
};

struct PedersenProductProof {
    static const std::uint32_t kVersion = 1U;
    std::uint32_t version;
    PedersenProofContext context;
    G1Jacobian_t commitment_a;
    G1Jacobian_t commitment_b;
    G1Jacobian_t commitment_c;
    G1Jacobian_t T1;
    G1Jacobian_t T2;
    Fr_t z;
    Fr_t z_s;
    Fr_t z_t;

    PedersenProductProof()
        : version(kVersion), context(), commitment_a{}, commitment_b{}, commitment_c{},
          T1{}, T2{}, z{}, z_s{}, z_t{} {}
};

struct PedersenVectorLinearProof {
    static const std::uint32_t kVersion = 1U;
    std::uint32_t version;
    PedersenProofContext context;
    std::vector<Fr_t> weights;
    G1Jacobian_t commitment;
    G1Jacobian_t evaluation_commitment;
    G1Jacobian_t A;
    G1Jacobian_t B;
    std::vector<Fr_t> z;
    Fr_t z_r;
    Fr_t z_t;

    PedersenVectorLinearProof()
        : version(kVersion), context(), weights(), commitment{}, evaluation_commitment{},
          A{}, B{}, z{}, z_r{}, z_t{} {}
};

inline bool pedersen_scalar_value_shape_valid(const PedersenScalarValueProof& proof,
                                              std::string* error = nullptr) {
    if (proof.version != PedersenScalarValueProof::kVersion ||
        !pedersen_context_shape_valid(proof.context, error) || proof.context.degree != 1U ||
        proof.context.basis_size != 0U) {
        if (error && error->empty()) *error = "invalid Pedersen scalar value shape";
        return false;
    }
    return fr_canonical(fr_wire(proof.schnorr_s));
}

inline bool pedersen_scalar_equality_shape_valid(const PedersenScalarEqualityProof& proof,
                                                 std::string* error = nullptr) {
    if (proof.version != PedersenScalarEqualityProof::kVersion ||
        !pedersen_context_shape_valid(proof.context, error) || proof.context.degree != 1U ||
        proof.context.basis_size != 0U) {
        if (error && error->empty()) *error = "invalid Pedersen scalar equality shape";
        return false;
    }
    return fr_canonical(fr_wire(proof.schnorr_s));
}

inline bool pedersen_product_shape_valid(const PedersenProductProof& proof,
                                         std::string* error = nullptr) {
    if (proof.version != PedersenProductProof::kVersion ||
        !pedersen_context_shape_valid(proof.context, error) || proof.context.degree != 2U ||
        proof.context.basis_size != 0U) {
        if (error && error->empty()) *error = "invalid Pedersen product shape";
        return false;
    }
    return fr_canonical(fr_wire(proof.z)) && fr_canonical(fr_wire(proof.z_s)) &&
           fr_canonical(fr_wire(proof.z_t));
}

inline bool pedersen_vector_linear_shape_valid(const PedersenVectorLinearProof& proof,
                                               std::string* error = nullptr) {
    if (proof.version != PedersenVectorLinearProof::kVersion ||
        !pedersen_context_shape_valid(proof.context, error) || proof.context.basis_size == 0U ||
        proof.weights.size() != proof.context.basis_size ||
        proof.z.size() != proof.context.basis_size ||
        proof.weights.size() > PedersenProofContext::kMaxBasisSize) {
        if (error && error->empty()) *error = "invalid Pedersen vector linear shape";
        return false;
    }
    for (const auto& weight : proof.weights)
        if (!fr_canonical(fr_wire(weight))) return false;
    for (const auto& response : proof.z)
        if (!fr_canonical(fr_wire(response))) return false;
    return fr_canonical(fr_wire(proof.z_r)) && fr_canonical(fr_wire(proof.z_t));
}

inline bool pedersen_wire_g1_valid(const G1Jacobian_t& point, std::string* error = nullptr) {
    std::array<byte, 48U> encoded{};
    std::string point_error;
    if (zkdl_g1_wire::encode(point, encoded, &point_error)) return true;
    if (error) *error = point_error.empty() ? "invalid Pedersen G1 point" : point_error;
    return false;
}

inline void serialize_pedersen_scalar_value(const PedersenScalarValueProof& proof,
                                            std::vector<byte>& out) {
    std::string error;
    if (!pedersen_scalar_value_shape_valid(proof, &error) ||
        !pedersen_wire_g1_valid(proof.commitment, &error) ||
        !pedersen_wire_g1_valid(proof.schnorr_R, &error))
        throw std::invalid_argument(error.empty() ? "invalid Pedersen scalar value proof" : error);
    out.insert(out.end(), {'P', 'S', 'C', '1'});
    append_u32(out, proof.version);
    append_pedersen_context(out, proof.context);
    append_g1(out, proof.commitment);
    append_g1(out, proof.schnorr_R);
    append_fr(out, proof.schnorr_s);
}

inline bool deserialize_pedersen_scalar_value(const std::vector<byte>& in,
                                              PedersenScalarValueProof& proof,
                                              std::string* error = nullptr) {
    size_t offset = 0U;
    PedersenScalarValueProof parsed;
    std::uint32_t version = 0U;
    if (in.size() < 4U || in[0] != 'P' || in[1] != 'S' || in[2] != 'C' || in[3] != '1')
        return false;
    offset = 4U;
    if (!read_u32(in, offset, version) || version != PedersenScalarValueProof::kVersion ||
        !read_pedersen_context(in, offset, parsed.context) || parsed.context.degree != 1U ||
        parsed.context.basis_size != 0U || !read_g1(in, offset, parsed.commitment) ||
        !read_g1(in, offset, parsed.schnorr_R) || !read_fr(in, offset, parsed.schnorr_s) ||
        offset != in.size()) {
        if (error) *error = "invalid or trailing Pedersen scalar value proof";
        return false;
    }
    parsed.version = version;
    if (!pedersen_scalar_value_shape_valid(parsed, error)) return false;
    proof = parsed;
    return true;
}

inline void serialize_pedersen_scalar_equality(const PedersenScalarEqualityProof& proof,
                                               std::vector<byte>& out) {
    std::string error;
    if (!pedersen_scalar_equality_shape_valid(proof, &error) ||
        !pedersen_wire_g1_valid(proof.commitment_1, &error) ||
        !pedersen_wire_g1_valid(proof.commitment_2, &error) ||
        !pedersen_wire_g1_valid(proof.schnorr_R, &error))
        throw std::invalid_argument(error.empty() ? "invalid Pedersen equality proof" : error);
    out.insert(out.end(), {'P', 'S', 'E', '1'});
    append_u32(out, proof.version);
    append_pedersen_context(out, proof.context);
    append_g1(out, proof.commitment_1);
    append_g1(out, proof.commitment_2);
    append_g1(out, proof.schnorr_R);
    append_fr(out, proof.schnorr_s);
}

inline bool deserialize_pedersen_scalar_equality(const std::vector<byte>& in,
                                                 PedersenScalarEqualityProof& proof,
                                                 std::string* error = nullptr) {
    size_t offset = 0U;
    PedersenScalarEqualityProof parsed;
    std::uint32_t version = 0U;
    if (in.size() < 4U || in[0] != 'P' || in[1] != 'S' || in[2] != 'E' || in[3] != '1')
        return false;
    offset = 4U;
    if (!read_u32(in, offset, version) || version != PedersenScalarEqualityProof::kVersion ||
        !read_pedersen_context(in, offset, parsed.context) || parsed.context.degree != 1U ||
        parsed.context.basis_size != 0U || !read_g1(in, offset, parsed.commitment_1) ||
        !read_g1(in, offset, parsed.commitment_2) || !read_g1(in, offset, parsed.schnorr_R) ||
        !read_fr(in, offset, parsed.schnorr_s) || offset != in.size()) {
        if (error) *error = "invalid or trailing Pedersen equality proof";
        return false;
    }
    parsed.version = version;
    if (!pedersen_scalar_equality_shape_valid(parsed, error)) return false;
    proof = parsed;
    return true;
}

inline void serialize_pedersen_product(const PedersenProductProof& proof,
                                       std::vector<byte>& out) {
    std::string error;
    if (!pedersen_product_shape_valid(proof, &error) ||
        !pedersen_wire_g1_valid(proof.commitment_a, &error) ||
        !pedersen_wire_g1_valid(proof.commitment_b, &error) ||
        !pedersen_wire_g1_valid(proof.commitment_c, &error) ||
        !pedersen_wire_g1_valid(proof.T1, &error) || !pedersen_wire_g1_valid(proof.T2, &error))
        throw std::invalid_argument(error.empty() ? "invalid Pedersen product proof" : error);
    out.insert(out.end(), {'P', 'P', 'R', '1'});
    append_u32(out, proof.version);
    append_pedersen_context(out, proof.context);
    append_g1(out, proof.commitment_a);
    append_g1(out, proof.commitment_b);
    append_g1(out, proof.commitment_c);
    append_g1(out, proof.T1);
    append_g1(out, proof.T2);
    append_fr(out, proof.z);
    append_fr(out, proof.z_s);
    append_fr(out, proof.z_t);
}

inline bool deserialize_pedersen_product(const std::vector<byte>& in,
                                         PedersenProductProof& proof,
                                         std::string* error = nullptr) {
    size_t offset = 0U;
    PedersenProductProof parsed;
    std::uint32_t version = 0U;
    if (in.size() < 4U || in[0] != 'P' || in[1] != 'P' || in[2] != 'R' || in[3] != '1')
        return false;
    offset = 4U;
    if (!read_u32(in, offset, version) || version != PedersenProductProof::kVersion ||
        !read_pedersen_context(in, offset, parsed.context) || parsed.context.degree != 2U ||
        parsed.context.basis_size != 0U || !read_g1(in, offset, parsed.commitment_a) ||
        !read_g1(in, offset, parsed.commitment_b) || !read_g1(in, offset, parsed.commitment_c) ||
        !read_g1(in, offset, parsed.T1) || !read_g1(in, offset, parsed.T2) ||
        !read_fr(in, offset, parsed.z) || !read_fr(in, offset, parsed.z_s) ||
        !read_fr(in, offset, parsed.z_t) || offset != in.size()) {
        if (error) *error = "invalid or trailing Pedersen product proof";
        return false;
    }
    parsed.version = version;
    if (!pedersen_product_shape_valid(parsed, error)) return false;
    proof = parsed;
    return true;
}

inline void serialize_pedersen_vector_linear(const PedersenVectorLinearProof& proof,
                                             std::vector<byte>& out) {
    std::string error;
    if (!pedersen_vector_linear_shape_valid(proof, &error) ||
        !pedersen_wire_g1_valid(proof.commitment, &error) ||
        !pedersen_wire_g1_valid(proof.evaluation_commitment, &error) ||
        !pedersen_wire_g1_valid(proof.A, &error) || !pedersen_wire_g1_valid(proof.B, &error))
        throw std::invalid_argument(error.empty() ? "invalid Pedersen vector proof" : error);
    out.insert(out.end(), {'P', 'V', 'L', '1'});
    append_u32(out, proof.version);
    append_pedersen_context(out, proof.context);
    append_u64(out, static_cast<std::uint64_t>(proof.weights.size()));
    for (const auto& weight : proof.weights) append_fr(out, weight);
    append_g1(out, proof.commitment);
    append_g1(out, proof.evaluation_commitment);
    append_g1(out, proof.A);
    append_g1(out, proof.B);
    append_u64(out, static_cast<std::uint64_t>(proof.z.size()));
    for (const auto& response : proof.z) append_fr(out, response);
    append_fr(out, proof.z_r);
    append_fr(out, proof.z_t);
}

inline bool deserialize_pedersen_vector_linear(const std::vector<byte>& in,
                                               PedersenVectorLinearProof& proof,
                                               std::string* error = nullptr) {
    size_t offset = 0U;
    std::uint32_t version = 0U;
    std::uint64_t count = 0U;
    PedersenVectorLinearProof parsed;
    if (in.size() < 4U || in[0] != 'P' || in[1] != 'V' || in[2] != 'L' || in[3] != '1')
        return false;
    offset = 4U;
    if (!read_u32(in, offset, version) || version != PedersenVectorLinearProof::kVersion ||
        !read_pedersen_context(in, offset, parsed.context) ||
        !read_u64(in, offset, count) || count == 0U ||
        count > PedersenProofContext::kMaxBasisSize ||
        count != parsed.context.basis_size) {
        if (error) *error = "invalid Pedersen vector weight shape";
        return false;
    }
    parsed.weights.resize(static_cast<size_t>(count));
    for (auto& weight : parsed.weights)
        if (!read_fr(in, offset, weight)) return false;
    if (!read_g1(in, offset, parsed.commitment) ||
        !read_g1(in, offset, parsed.evaluation_commitment) || !read_g1(in, offset, parsed.A) ||
        !read_g1(in, offset, parsed.B) || !read_u64(in, offset, count) ||
        count != parsed.context.basis_size || count > PedersenProofContext::kMaxBasisSize) {
        if (error) *error = "invalid Pedersen vector response shape";
        return false;
    }
    parsed.z.resize(static_cast<size_t>(count));
    for (auto& response : parsed.z)
        if (!read_fr(in, offset, response)) return false;
    if (!read_fr(in, offset, parsed.z_r) || !read_fr(in, offset, parsed.z_t) ||
        offset != in.size()) {
        if (error) *error = "invalid or trailing Pedersen vector proof";
        return false;
    }
    parsed.version = version;
    if (!pedersen_vector_linear_shape_valid(parsed, error)) return false;
    proof = parsed;
    return true;
}

enum class ZkTerminalClaimLinkKind : std::uint32_t {
    ExternalEquality = 1U,
    PublicValue = 2U
};

/* A product-DAG node contains only role/index metadata, the output
 * commitment, and the existing Pedersen product Sigma proof.  Factor values
 * and all blinders remain prover-local. */
struct ZkProductNodeProof {
    static const std::uint32_t kVersion = 1U;
    static const std::uint32_t kMaxNestedProofBytes = 1U << 20U;
    std::uint32_t version = kVersion;
    std::uint32_t monomial_index = PedersenProofContext::kUnsetIndex;
    std::uint32_t factor_step = PedersenProofContext::kUnsetIndex;
    std::uint32_t left_commitment_index = PedersenProofContext::kUnsetIndex;
    std::uint32_t right_commitment_index = PedersenProofContext::kUnsetIndex;
    std::uint32_t output_commitment_index = PedersenProofContext::kUnsetIndex;
    G1Jacobian_t output_commitment{};
    PedersenProductProof product_proof;
};

struct ZkTerminalRelationProof {
    static const std::uint32_t kVersion = 1U;
    static const std::uint32_t kMaxTerminalCommitments = 4096U;
    static const std::uint32_t kMaxProductNodes = 4096U;
    static const std::uint32_t kMaxNestedProofBytes = 1U << 22U;
    std::uint32_t version = kVersion;
    // Scalar evaluation commitments T_i = eval_i*U + r_i*H.  The matching
    // canonical tensor commitment is carried by each Hyrax opening's xi and
    // is checked against the verifier's authenticated tensor map.
    std::vector<G1Jacobian_t> terminal_commitments;
    std::vector<HyraxOpeningProofV3> terminal_openings;
    std::vector<ZkProductNodeProof> product_nodes;
    G1Jacobian_t residual_commitment{};
    ZkTerminalClaimLinkKind claim_link_kind =
        ZkTerminalClaimLinkKind::ExternalEquality;
    G1Jacobian_t external_claim_commitment{};
    PedersenScalarEqualityProof external_equality_binding;
    PedersenScalarValueProof public_value_binding;
    PedersenProofContext context;
};

inline bool zk_terminal_g1_default(const G1Jacobian_t& point) {
    for (const auto value : point.x.val) if (value != 0U) return false;
    for (const auto value : point.y.val) if (value != 0U) return false;
    for (const auto value : point.z.val) if (value != 0U) return false;
    return true;
}

inline bool zk_terminal_fr_default(const Fr_t& value) {
    for (const auto limb : value.val) if (limb != 0U) return false;
    return true;
}

inline bool zk_terminal_context_default(const PedersenProofContext& context) {
    return context.commitment_index == PedersenProofContext::kUnsetIndex &&
           context.operation == 0U &&
           context.commitment_a_index == PedersenProofContext::kUnsetIndex &&
           context.commitment_b_index == PedersenProofContext::kUnsetIndex &&
           context.commitment_c_index == PedersenProofContext::kUnsetIndex &&
           context.evaluation_commitment_index == PedersenProofContext::kUnsetIndex &&
           context.degree == 0U && context.basis_size == 0U &&
           context.basis_digest == std::array<byte, 32U>{{}} &&
           context.descriptor_digest == std::array<byte, 32U>{{}} &&
           context.registry_digest == std::array<byte, 32U>{{}} &&
           context.srs_digest == std::array<byte, 32U>{{}};
}

inline bool zk_terminal_scalar_value_default(const PedersenScalarValueProof& proof) {
    return proof.version == PedersenScalarValueProof::kVersion &&
           zk_terminal_context_default(proof.context) &&
           zk_terminal_g1_default(proof.commitment) &&
           zk_terminal_g1_default(proof.schnorr_R) &&
           zk_terminal_fr_default(proof.schnorr_s);
}

inline bool zk_terminal_scalar_equality_default(const PedersenScalarEqualityProof& proof) {
    return proof.version == PedersenScalarEqualityProof::kVersion &&
           zk_terminal_context_default(proof.context) &&
           zk_terminal_g1_default(proof.commitment_1) &&
           zk_terminal_g1_default(proof.commitment_2) &&
           zk_terminal_g1_default(proof.schnorr_R) &&
           zk_terminal_fr_default(proof.schnorr_s);
}

inline bool zk_terminal_g1_nonzero_valid(const G1Jacobian_t& point,
                                         std::string* error = nullptr) {
    std::array<byte, 48U> encoded{};
    std::string point_error;
    if (!zkdl_g1_wire::encode(point, encoded, &point_error)) {
        if (error) *error = point_error.empty() ? "invalid terminal G1 point" : point_error;
        return false;
    }
    if (zkdl_g1_wire::from_native(point).is_zero()) {
        if (error) *error = "terminal commitment cannot be infinity";
        return false;
    }
    return true;
}

inline bool zk_terminal_g1_equal(const G1Jacobian_t& left, const G1Jacobian_t& right) {
    std::array<byte, 48U> left_encoded{};
    std::array<byte, 48U> right_encoded{};
    std::string error;
    return zkdl_g1_wire::encode(left, left_encoded, &error) &&
           zkdl_g1_wire::encode(right, right_encoded, &error) &&
           left_encoded == right_encoded;
}

inline bool zk_product_node_shape_valid(const ZkProductNodeProof& node,
                                        std::string* error = nullptr) {
    const auto fail = [&](const char* message) {
        if (error) *error = message;
        return false;
    };
    if (node.version != ZkProductNodeProof::kVersion ||
        node.monomial_index == PedersenProofContext::kUnsetIndex ||
        node.factor_step == PedersenProofContext::kUnsetIndex ||
        node.left_commitment_index == PedersenProofContext::kUnsetIndex ||
        node.right_commitment_index == PedersenProofContext::kUnsetIndex ||
        node.output_commitment_index == PedersenProofContext::kUnsetIndex ||
        node.left_commitment_index == node.right_commitment_index ||
        node.left_commitment_index == node.output_commitment_index ||
        node.right_commitment_index == node.output_commitment_index ||
        !zk_terminal_g1_nonzero_valid(node.output_commitment, error) ||
        !pedersen_product_shape_valid(node.product_proof, error) ||
        !pedersen_wire_g1_valid(node.product_proof.commitment_a, error) ||
        !pedersen_wire_g1_valid(node.product_proof.commitment_b, error) ||
        !pedersen_wire_g1_valid(node.product_proof.commitment_c, error))
        return fail("invalid Zk product node shape");
    const auto& context = node.product_proof.context;
    if (context.operation != PedersenProofContext::kOperationProduct ||
        context.commitment_index != node.left_commitment_index ||
        context.commitment_a_index != node.left_commitment_index ||
        context.commitment_b_index != node.right_commitment_index ||
        context.commitment_c_index != node.output_commitment_index ||
        !zk_terminal_g1_equal(node.product_proof.commitment_c, node.output_commitment))
        return fail("Zk product node roles or output commitment are not bound");
    return true;
}

inline bool zk_terminal_read_nested(const std::vector<byte>& in, size_t& offset,
                                    std::vector<byte>& nested,
                                    std::uint32_t max_bytes) {
    std::uint64_t bytes = 0U;
    if (!read_u64(in, offset, bytes) || bytes == 0U || bytes > max_bytes ||
        offset > in.size() || bytes > in.size() - offset)
        return false;
    nested.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                  in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    offset += static_cast<std::size_t>(bytes);
    return true;
}

inline bool zk_terminal_read_optional_nested(const std::vector<byte>& in, size_t& offset,
                                              std::vector<byte>& nested,
                                              std::uint32_t max_bytes) {
    std::uint64_t bytes = 0U;
    if (!read_u64(in, offset, bytes) || bytes > max_bytes ||
        offset > in.size() || bytes > in.size() - offset)
        return false;
    nested.clear();
    if (bytes == 0U) return true;
    nested.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                  in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    offset += static_cast<std::size_t>(bytes);
    return true;
}

inline void zk_terminal_append_nested(std::vector<byte>& out,
                                      const std::vector<byte>& nested) {
    append_u64(out, static_cast<std::uint64_t>(nested.size()));
    out.insert(out.end(), nested.begin(), nested.end());
}

inline void serialize_zk_product_node(const ZkProductNodeProof& node,
                                      std::vector<byte>& out) {
    std::string error;
    if (!zk_product_node_shape_valid(node, &error))
        throw std::invalid_argument(error.empty() ? "invalid Zk product node" : error);
    out.insert(out.end(), {'Z', 'P', 'N', '1'});
    append_u32(out, node.version);
    append_u32(out, node.monomial_index);
    append_u32(out, node.factor_step);
    append_u32(out, node.left_commitment_index);
    append_u32(out, node.right_commitment_index);
    append_u32(out, node.output_commitment_index);
    append_g1(out, node.output_commitment);
    std::vector<byte> product;
    serialize_pedersen_product(node.product_proof, product);
    zk_terminal_append_nested(out, product);
}

inline bool deserialize_zk_product_node(const std::vector<byte>& in, size_t& offset,
                                        ZkProductNodeProof& node,
                                        std::string* error = nullptr) {
    ZkProductNodeProof parsed;
    std::vector<byte> nested;
    if (offset > in.size() || in.size() - offset < 4U ||
        in[offset] != 'Z' || in[offset + 1U] != 'P' ||
        in[offset + 2U] != 'N' || in[offset + 3U] != '1') {
        if (error) *error = "invalid Zk product node magic";
        return false;
    }
    offset += 4U;
    if (!read_u32(in, offset, parsed.version) ||
        !read_u32(in, offset, parsed.monomial_index) ||
        !read_u32(in, offset, parsed.factor_step) ||
        !read_u32(in, offset, parsed.left_commitment_index) ||
        !read_u32(in, offset, parsed.right_commitment_index) ||
        !read_u32(in, offset, parsed.output_commitment_index) ||
        !read_g1(in, offset, parsed.output_commitment) ||
        !zk_terminal_read_nested(in, offset, nested,
                                 ZkProductNodeProof::kMaxNestedProofBytes)) {
        if (error) *error = "truncated Zk product node";
        return false;
    }
    if (!deserialize_pedersen_product(nested, parsed.product_proof, error)) {
        if (error && error->empty()) *error = "invalid nested Zk product proof";
        return false;
    }
    if (!zk_product_node_shape_valid(parsed, error)) return false;
    node = parsed;
    return true;
}

inline bool zk_terminal_relation_shape_valid(const ZkTerminalRelationProof& proof,
                                             std::string* error = nullptr) {
    const auto fail = [&](const char* message) {
        if (error) *error = message;
        return false;
    };
    if (proof.version != ZkTerminalRelationProof::kVersion ||
        proof.terminal_commitments.empty() ||
        proof.terminal_commitments.size() > ZkTerminalRelationProof::kMaxTerminalCommitments ||
        proof.terminal_openings.size() != proof.terminal_commitments.size() ||
        proof.product_nodes.size() > ZkTerminalRelationProof::kMaxProductNodes ||
        !pedersen_context_shape_valid(proof.context, error) ||
        proof.context.basis_digest == std::array<byte, 32U>{{}} ||
        !zk_terminal_g1_nonzero_valid(proof.residual_commitment, error) ||
        !zk_terminal_g1_nonzero_valid(proof.external_claim_commitment, error))
        return fail("invalid Zk terminal relation header");
    std::vector<std::uint32_t> opening_indices;
    opening_indices.reserve(proof.terminal_openings.size());
    for (std::size_t index = 0U; index < proof.terminal_commitments.size(); ++index) {
        if (!zk_terminal_g1_nonzero_valid(proof.terminal_commitments[index], error) ||
            !hyrax_opening_v3_shape_valid(proof.terminal_openings[index], error) ||
            !zk_terminal_g1_equal(proof.terminal_openings[index].tau,
                                  proof.terminal_commitments[index]) ||
            proof.terminal_openings[index].descriptor_digest != proof.context.descriptor_digest ||
            proof.terminal_openings[index].registry_digest != proof.context.registry_digest ||
            proof.terminal_openings[index].srs_digest != proof.context.srs_digest)
            return fail("terminal commitment/opening binding is invalid");
        const std::uint32_t opening_index = proof.terminal_openings[index].commitment_index;
        if (std::find(opening_indices.begin(), opening_indices.end(), opening_index) !=
            opening_indices.end())
            return fail("duplicate terminal opening commitment index");
        opening_indices.push_back(opening_index);
    }
    for (const auto& node : proof.product_nodes)
        if (!zk_product_node_shape_valid(node, error) ||
            node.product_proof.context.descriptor_digest != proof.context.descriptor_digest ||
            node.product_proof.context.registry_digest != proof.context.registry_digest ||
            node.product_proof.context.srs_digest != proof.context.srs_digest)
            return fail("terminal product node is not relation-bound");
    if (proof.claim_link_kind == ZkTerminalClaimLinkKind::ExternalEquality) {
        if (!pedersen_scalar_equality_shape_valid(proof.external_equality_binding, error) ||
            !zk_terminal_scalar_value_default(proof.public_value_binding) ||
            proof.external_equality_binding.context.operation !=
                PedersenProofContext::kOperationScalarEquality ||
            !zk_terminal_g1_equal(proof.external_equality_binding.commitment_1,
                                  proof.residual_commitment) ||
            !zk_terminal_g1_equal(proof.external_equality_binding.commitment_2,
                                  proof.external_claim_commitment))
            return fail("invalid terminal external equality binding");
    } else if (proof.claim_link_kind == ZkTerminalClaimLinkKind::PublicValue) {
        if (!pedersen_scalar_value_shape_valid(proof.public_value_binding, error) ||
            !zk_terminal_scalar_equality_default(proof.external_equality_binding) ||
            proof.public_value_binding.context.operation !=
                PedersenProofContext::kOperationScalarValue ||
            !zk_terminal_g1_equal(proof.public_value_binding.commitment,
                                  proof.external_claim_commitment) ||
            !zk_terminal_g1_equal(proof.external_claim_commitment,
                                  proof.residual_commitment))
            return fail("invalid terminal public-value binding");
    } else {
        return fail("unknown terminal claim link kind");
    }
    return true;
}

inline void serialize_zk_terminal_relation(const ZkTerminalRelationProof& proof,
                                           std::vector<byte>& out) {
    std::string error;
    if (!zk_terminal_relation_shape_valid(proof, &error))
        throw std::invalid_argument(error.empty() ? "invalid Zk terminal relation proof" : error);
    out.insert(out.end(), {'Z', 'T', 'R', '1'});
    append_u32(out, proof.version);
    append_pedersen_context(out, proof.context);
    append_u64(out, static_cast<std::uint64_t>(proof.terminal_commitments.size()));
    for (std::size_t index = 0U; index < proof.terminal_commitments.size(); ++index) {
        append_g1(out, proof.terminal_commitments[index]);
        std::vector<byte> opening;
        serialize_hyrax_opening_v3(proof.terminal_openings[index], opening);
        zk_terminal_append_nested(out, opening);
    }
    append_u64(out, static_cast<std::uint64_t>(proof.product_nodes.size()));
    for (const auto& node : proof.product_nodes) {
        std::vector<byte> encoded;
        serialize_zk_product_node(node, encoded);
        zk_terminal_append_nested(out, encoded);
    }
    append_g1(out, proof.residual_commitment);
    append_u32(out, static_cast<std::uint32_t>(proof.claim_link_kind));
    append_g1(out, proof.external_claim_commitment);
    if (proof.claim_link_kind == ZkTerminalClaimLinkKind::ExternalEquality) {
        std::vector<byte> equality;
        serialize_pedersen_scalar_equality(proof.external_equality_binding, equality);
        zk_terminal_append_nested(out, equality);
        append_u64(out, 0U);
    } else {
        append_u64(out, 0U);
        std::vector<byte> value;
        serialize_pedersen_scalar_value(proof.public_value_binding, value);
        zk_terminal_append_nested(out, value);
    }
}

inline bool deserialize_zk_terminal_relation(const std::vector<byte>& in,
                                             ZkTerminalRelationProof& proof,
                                             std::string* error = nullptr) {
    ZkTerminalRelationProof parsed;
    size_t offset = 0U;
    std::uint64_t count = 0U;
    if (in.size() < 4U || in[0] != 'Z' || in[1] != 'T' ||
        in[2] != 'R' || in[3] != '1') {
        if (error) *error = "invalid Zk terminal relation magic";
        return false;
    }
    offset = 4U;
    if (!read_u32(in, offset, parsed.version) ||
        !read_pedersen_context(in, offset, parsed.context) ||
        !read_u64(in, offset, count) || count == 0U ||
        count > ZkTerminalRelationProof::kMaxTerminalCommitments) {
        if (error) *error = "invalid Zk terminal relation header";
        return false;
    }
    parsed.terminal_commitments.resize(static_cast<std::size_t>(count));
    parsed.terminal_openings.resize(static_cast<std::size_t>(count));
    std::vector<byte> nested;
    for (std::size_t index = 0U; index < parsed.terminal_commitments.size(); ++index) {
        if (!read_g1(in, offset, parsed.terminal_commitments[index]) ||
            !zk_terminal_read_nested(in, offset, nested,
                                     ZkTerminalRelationProof::kMaxNestedProofBytes)) {
            if (error) *error = "truncated terminal opening";
            return false;
        }
        size_t nested_offset = 0U;
        if (!deserialize_hyrax_opening_v3(nested, nested_offset,
                                           parsed.terminal_openings[index]) ||
            nested_offset != nested.size()) {
            if (error) *error = "invalid nested Hyrax v3 terminal opening";
            return false;
        }
    }
    if (!read_u64(in, offset, count) || count > ZkTerminalRelationProof::kMaxProductNodes) {
        if (error) *error = "invalid terminal product-node count";
        return false;
    }
    parsed.product_nodes.resize(static_cast<std::size_t>(count));
    for (auto& node : parsed.product_nodes) {
        if (!zk_terminal_read_nested(in, offset, nested,
                                     ZkTerminalRelationProof::kMaxNestedProofBytes)) {
            if (error) *error = "truncated terminal product node";
            return false;
        }
        size_t nested_offset = 0U;
        if (!deserialize_zk_product_node(nested, nested_offset, node, error) ||
            nested_offset != nested.size()) {
            if (error && error->empty()) *error = "invalid nested terminal product node";
            return false;
        }
    }
    std::uint32_t kind = 0U;
    std::vector<byte> equality;
    std::vector<byte> value;
    if (!read_g1(in, offset, parsed.residual_commitment) ||
        !read_u32(in, offset, kind) ||
        (kind != static_cast<std::uint32_t>(ZkTerminalClaimLinkKind::ExternalEquality) &&
         kind != static_cast<std::uint32_t>(ZkTerminalClaimLinkKind::PublicValue)) ||
        !read_g1(in, offset, parsed.external_claim_commitment) ||
        !zk_terminal_read_optional_nested(in, offset, equality,
                                          ZkTerminalRelationProof::kMaxNestedProofBytes) ||
        !zk_terminal_read_optional_nested(in, offset, value,
                                          ZkTerminalRelationProof::kMaxNestedProofBytes) ||
        offset != in.size()) {
        if (error) *error = "truncated or trailing terminal relation proof";
        return false;
    }
    parsed.claim_link_kind = static_cast<ZkTerminalClaimLinkKind>(kind);
    if (parsed.claim_link_kind == ZkTerminalClaimLinkKind::ExternalEquality) {
        if (equality.empty() || !value.empty() ||
            !deserialize_pedersen_scalar_equality(equality,
                                                  parsed.external_equality_binding, error) ||
            !zk_terminal_scalar_value_default(parsed.public_value_binding)) {
            if (error && error->empty()) *error = "invalid terminal equality branch";
            return false;
        }
    } else {
        if (!equality.empty() || value.empty() ||
            !zk_terminal_scalar_equality_default(parsed.external_equality_binding) ||
            !deserialize_pedersen_scalar_value(value, parsed.public_value_binding, error)) {
            if (error && error->empty()) *error = "invalid terminal public-value branch";
            return false;
        }
    }
    if (!zk_terminal_relation_shape_valid(parsed, error)) return false;
    proof = parsed;
    return true;
}

/*
 * Formal GAT Sumcheck wire.
 *
 * The legacy SumcheckProof wire exposes every round evaluation and every
 * terminal scalar.  That representation is intentionally not usable by the
 * formal relation path: a prover could replace a hidden claim without
 * supplying a commitment/opening witness.  ZkSumcheckProof carries only the
 * versioned Pedersen statements and their Sigma responses.  Round challenges
 * are replayed by the relation verifier after it has absorbed the canonical
 * round proof bytes.
 */
enum class ZkSumcheckLinkKind : std::uint32_t {
    InitialPublicClaim = 1U,
    PreviousEvaluation = 2U
};

struct ZkSumcheckRoundProof {
    static const std::uint32_t kVersion = 1U;
    std::uint32_t version = kVersion;
    std::uint32_t round_index = 0U;
    G1Jacobian_t coefficient_commitment{};
    G1Jacobian_t endpoint_commitment{};
    PedersenVectorLinearProof endpoint_linear;
    ZkSumcheckLinkKind link_kind = ZkSumcheckLinkKind::InitialPublicClaim;
    // Exactly one of these two branches is encoded.  The inactive branch is
    // required to remain its default value and is serialized as zero bytes.
    PedersenScalarValueProof initial_claim_binding;
    PedersenScalarEqualityProof previous_claim_binding;
    G1Jacobian_t evaluation_commitment{};
    PedersenVectorLinearProof evaluation_linear;
};

struct ZkSumcheckProof {
    static const std::uint32_t kVersion = 1U;
    static const std::uint32_t kMaxRounds = 30U;
    static const std::uint64_t kMaxNestedProofBytes = 1U << 22U;

    std::uint32_t version = kVersion;
    std::string label;
    std::uint32_t degree = 0U;
    std::uint32_t domain_size = 0U;
    // Public initial-claim binding.  The context is a scalar-value context;
    // its H/U and role/index fields are checked against the proof-level H/U
    // and the first round by the formal relation verifier.
    PedersenProofContext initial_claim_context;
    std::vector<ZkSumcheckRoundProof> rounds;
    G1Jacobian_t final_claim_commitment{};
    std::array<byte, 32U> descriptor_digest{{}};
    std::array<byte, 32U> registry_digest{{}};
    std::array<byte, 32U> srs_digest{{}};
    std::array<byte, 32U> basis_digest{{}};
    G1Jacobian_t blinding_generator{}; // authenticated H; never selected by prover
    G1Jacobian_t u_generator{};        // authenticated U; never selected by prover
};

inline bool zk_sumcheck_g1_is_default(const G1Jacobian_t& point) {
    for (const auto value : point.x.val) if (value != 0U) return false;
    for (const auto value : point.y.val) if (value != 0U) return false;
    for (const auto value : point.z.val) if (value != 0U) return false;
    return true;
}

inline bool zk_sumcheck_fr_is_default(const Fr_t& value) {
    for (const auto limb : value.val) if (limb != 0U) return false;
    return true;
}

inline bool zk_sumcheck_context_is_default(const PedersenProofContext& context) {
    return context.commitment_index == PedersenProofContext::kUnsetIndex &&
           context.operation == 0U &&
           context.commitment_a_index == PedersenProofContext::kUnsetIndex &&
           context.commitment_b_index == PedersenProofContext::kUnsetIndex &&
           context.commitment_c_index == PedersenProofContext::kUnsetIndex &&
           context.evaluation_commitment_index == PedersenProofContext::kUnsetIndex &&
           context.degree == 0U && context.basis_size == 0U &&
           context.basis_digest == std::array<byte, 32U>{{}} &&
           context.descriptor_digest == std::array<byte, 32U>{{}} &&
           context.registry_digest == std::array<byte, 32U>{{}} &&
           context.srs_digest == std::array<byte, 32U>{{}};
}

inline bool zk_sumcheck_g1_equal(const G1Jacobian_t& left, const G1Jacobian_t& right) {
    std::array<byte, 48U> left_encoded{};
    std::array<byte, 48U> right_encoded{};
    std::string error;
    return zkdl_g1_wire::encode(left, left_encoded, &error) &&
           zkdl_g1_wire::encode(right, right_encoded, &error) && left_encoded == right_encoded;
}

inline bool zk_sumcheck_g1_nonzero_valid(const G1Jacobian_t& point,
                                         std::string* error = nullptr) {
    if (!pedersen_wire_g1_valid(point, error) || zk_sumcheck_g1_is_default(point)) {
        if (error && error->empty()) *error = "ZkSumcheck G1 point is zero or invalid";
        return false;
    }
    const zkdl_g1_wire::Point decoded = zkdl_g1_wire::from_native(point);
    if (decoded.is_zero()) {
        if (error) *error = "ZkSumcheck infinity is not allowed for a commitment point";
        return false;
    }
    return true;
}

inline bool zk_sumcheck_digest_nonzero(const std::array<byte, 32U>& digest) {
    return digest != std::array<byte, 32U>{{}};
}

inline bool zk_sumcheck_scalar_value_default(const PedersenScalarValueProof& proof) {
    return proof.version == PedersenScalarValueProof::kVersion &&
           zk_sumcheck_context_is_default(proof.context) &&
           zk_sumcheck_g1_is_default(proof.commitment) &&
           zk_sumcheck_g1_is_default(proof.schnorr_R) &&
           zk_sumcheck_fr_is_default(proof.schnorr_s);
}

inline bool zk_sumcheck_scalar_equality_default(const PedersenScalarEqualityProof& proof) {
    return proof.version == PedersenScalarEqualityProof::kVersion &&
           zk_sumcheck_context_is_default(proof.context) &&
           zk_sumcheck_g1_is_default(proof.commitment_1) &&
           zk_sumcheck_g1_is_default(proof.commitment_2) &&
           zk_sumcheck_g1_is_default(proof.schnorr_R) &&
           zk_sumcheck_fr_is_default(proof.schnorr_s);
}

inline bool zk_sumcheck_context_for_vector(const PedersenProofContext& context,
                                           std::uint32_t degree,
                                           std::string* error = nullptr) {
    return pedersen_context_shape_valid(context, error) &&
           context.operation == PedersenProofContext::kOperationVectorLinear &&
           context.degree == degree && context.basis_size == degree + 1U;
}

inline bool zk_sumcheck_round_shape_valid(const ZkSumcheckRoundProof& round,
                                          std::uint32_t degree,
                                          std::string* error = nullptr) {
    const auto fail = [&](const char* message) {
        if (error) *error = message;
        return false;
    };
    if (round.version != ZkSumcheckRoundProof::kVersion ||
        !zk_sumcheck_g1_nonzero_valid(round.coefficient_commitment, error) ||
        !zk_sumcheck_g1_nonzero_valid(round.endpoint_commitment, error) ||
        !zk_sumcheck_g1_nonzero_valid(round.evaluation_commitment, error) ||
        !zk_sumcheck_context_for_vector(round.endpoint_linear.context, degree, error) ||
        !zk_sumcheck_context_for_vector(round.evaluation_linear.context, degree, error) ||
        !pedersen_vector_linear_shape_valid(round.endpoint_linear, error) ||
        !pedersen_vector_linear_shape_valid(round.evaluation_linear, error))
        return fail("invalid ZkSumcheck round commitment/proof shape");
    if (!zk_sumcheck_g1_equal(round.endpoint_linear.commitment,
                              round.coefficient_commitment) ||
        !zk_sumcheck_g1_equal(round.evaluation_linear.commitment,
                              round.coefficient_commitment) ||
        !zk_sumcheck_g1_equal(round.endpoint_linear.evaluation_commitment,
                              round.endpoint_commitment) ||
        !zk_sumcheck_g1_equal(round.evaluation_linear.evaluation_commitment,
                              round.evaluation_commitment))
        return fail("ZkSumcheck round commitments are not bound to linear proofs");
    if (round.endpoint_linear.weights.size() != static_cast<std::size_t>(degree + 1U))
        return fail("ZkSumcheck endpoint weight arity mismatch");
    if (round.link_kind == ZkSumcheckLinkKind::InitialPublicClaim) {
        if (!pedersen_scalar_value_shape_valid(round.initial_claim_binding, error) ||
            !zk_sumcheck_scalar_equality_default(round.previous_claim_binding) ||
            !zk_sumcheck_g1_equal(round.initial_claim_binding.commitment,
                                  round.endpoint_commitment))
            return fail("ZkSumcheck initial-claim branch is malformed");
    } else if (round.link_kind == ZkSumcheckLinkKind::PreviousEvaluation) {
        if (!pedersen_scalar_equality_shape_valid(round.previous_claim_binding, error) ||
            !zk_sumcheck_scalar_value_default(round.initial_claim_binding) ||
            !zk_sumcheck_g1_equal(round.previous_claim_binding.commitment_1,
                                  round.endpoint_commitment))
            return fail("ZkSumcheck previous-claim branch is malformed");
    } else {
        return fail("unknown ZkSumcheck link branch");
    }
    return true;
}

inline bool zk_sumcheck_shape_valid(const ZkSumcheckProof& proof,
                                    std::string* error = nullptr) {
    const auto fail = [&](const char* message) {
        if (error) *error = message;
        return false;
    };
    if (proof.version != ZkSumcheckProof::kVersion || proof.label.empty() ||
        proof.label.size() > 4096U || proof.degree == 0U || proof.degree > 8U ||
        proof.domain_size == 0U ||
        (proof.domain_size & (proof.domain_size - 1U)) != 0U ||
        proof.domain_size > (1U << ZkSumcheckProof::kMaxRounds) ||
        proof.rounds.size() > ZkSumcheckProof::kMaxRounds ||
        proof.rounds.size() != static_cast<std::size_t>(
            [&]() { std::uint32_t bits = 0U, value = proof.domain_size;
                    while (value > 1U) { value >>= 1U; ++bits; }
                    return bits; }()))
        return fail("invalid ZkSumcheck proof version/domain/round count");
    if (!pedersen_context_shape_valid(proof.initial_claim_context, error) ||
        proof.initial_claim_context.operation != PedersenProofContext::kOperationScalarValue ||
        proof.initial_claim_context.degree != 1U || proof.initial_claim_context.basis_size != 0U ||
        !zk_sumcheck_digest_nonzero(proof.descriptor_digest) ||
        !zk_sumcheck_digest_nonzero(proof.registry_digest) ||
        !zk_sumcheck_digest_nonzero(proof.srs_digest) ||
        !zk_sumcheck_digest_nonzero(proof.basis_digest) ||
        proof.initial_claim_context.descriptor_digest != proof.descriptor_digest ||
        proof.initial_claim_context.registry_digest != proof.registry_digest ||
        proof.initial_claim_context.srs_digest != proof.srs_digest ||
        !zk_sumcheck_g1_nonzero_valid(proof.blinding_generator, error) ||
        !zk_sumcheck_g1_nonzero_valid(proof.u_generator, error) ||
        zk_sumcheck_g1_equal(proof.blinding_generator, proof.u_generator) ||
        !zk_sumcheck_g1_nonzero_valid(proof.final_claim_commitment, error))
        return fail("invalid ZkSumcheck proof context/basis/role binding");
    if (proof.rounds.empty()) return fail("ZkSumcheck requires at least one round");
    for (std::size_t index = 0U; index < proof.rounds.size(); ++index) {
        const auto& round = proof.rounds[index];
        if (round.round_index != index || !zk_sumcheck_round_shape_valid(round, proof.degree, error))
            return fail("ZkSumcheck round order or shape is not canonical");
        if (index == 0U && round.link_kind != ZkSumcheckLinkKind::InitialPublicClaim)
            return fail("ZkSumcheck first round lacks initial public claim binding");
        if (index != 0U && round.link_kind != ZkSumcheckLinkKind::PreviousEvaluation)
            return fail("ZkSumcheck non-first round has the wrong claim branch");
        if (index != 0U && !zk_sumcheck_g1_equal(
                round.previous_claim_binding.commitment_2,
                proof.rounds[index - 1U].evaluation_commitment))
            return fail("ZkSumcheck round claim chain is not contiguous");
    }
    if (!zk_sumcheck_g1_equal(proof.final_claim_commitment,
                              proof.rounds.back().evaluation_commitment))
        return fail("ZkSumcheck final claim commitment is not the last evaluation");
    return true;
}

inline void zk_sumcheck_append_nested(std::vector<byte>& out,
                                      const std::vector<byte>& nested) {
    append_u64(out, static_cast<std::uint64_t>(nested.size()));
    out.insert(out.end(), nested.begin(), nested.end());
}

inline bool zk_sumcheck_read_nested(const std::vector<byte>& in, size_t& offset,
                                    std::vector<byte>& nested) {
    std::uint64_t bytes = 0U;
    if (!read_u64(in, offset, bytes) || bytes == 0U ||
        bytes > ZkSumcheckProof::kMaxNestedProofBytes || bytes > in.size() - offset)
        return false;
    nested.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                  in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    offset += static_cast<std::size_t>(bytes);
    return true;
}

inline bool zk_sumcheck_read_optional_nested(const std::vector<byte>& in, size_t& offset,
                                             std::vector<byte>& nested) {
    std::uint64_t bytes = 0U;
    if (!read_u64(in, offset, bytes) || bytes > ZkSumcheckProof::kMaxNestedProofBytes ||
        bytes > in.size() - offset) return false;
    nested.clear();
    if (bytes != 0U) {
        nested.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                      in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
        offset += static_cast<std::size_t>(bytes);
    }
    return true;
}

inline void serialize_zk_sumcheck_round(const ZkSumcheckRoundProof& round,
                                        std::vector<byte>& out) {
    std::string error;
    if (!zk_sumcheck_round_shape_valid(round, round.endpoint_linear.context.degree, &error))
        throw std::invalid_argument(error.empty() ? "invalid ZkSumcheck round" : error);
    append_u32(out, round.version);
    append_u32(out, round.round_index);
    append_g1(out, round.coefficient_commitment);
    append_g1(out, round.endpoint_commitment);
    std::vector<byte> endpoint;
    serialize_pedersen_vector_linear(round.endpoint_linear, endpoint);
    zk_sumcheck_append_nested(out, endpoint);
    append_u32(out, static_cast<std::uint32_t>(round.link_kind));
    if (round.link_kind == ZkSumcheckLinkKind::InitialPublicClaim) {
        std::vector<byte> initial;
        serialize_pedersen_scalar_value(round.initial_claim_binding, initial);
        zk_sumcheck_append_nested(out, initial);
        append_u64(out, 0U);
    } else {
        append_u64(out, 0U);
        std::vector<byte> previous;
        serialize_pedersen_scalar_equality(round.previous_claim_binding, previous);
        zk_sumcheck_append_nested(out, previous);
    }
    append_g1(out, round.evaluation_commitment);
    std::vector<byte> evaluation;
    serialize_pedersen_vector_linear(round.evaluation_linear, evaluation);
    zk_sumcheck_append_nested(out, evaluation);
}

inline bool deserialize_zk_sumcheck_round(const std::vector<byte>& in, size_t& offset,
                                          ZkSumcheckRoundProof& round,
                                          std::uint32_t degree,
                                          std::string* error = nullptr) {
    ZkSumcheckRoundProof parsed;
    std::vector<byte> nested;
    std::uint32_t link = 0U;
    if (!read_u32(in, offset, parsed.version) || parsed.version != ZkSumcheckRoundProof::kVersion ||
        !read_u32(in, offset, parsed.round_index) || !read_g1(in, offset, parsed.coefficient_commitment) ||
        !read_g1(in, offset, parsed.endpoint_commitment) ||
        !zk_sumcheck_read_nested(in, offset, nested) ||
        !deserialize_pedersen_vector_linear(nested, parsed.endpoint_linear, error) ||
        !read_u32(in, offset, link) ||
        (link != static_cast<std::uint32_t>(ZkSumcheckLinkKind::InitialPublicClaim) &&
         link != static_cast<std::uint32_t>(ZkSumcheckLinkKind::PreviousEvaluation))) {
        if (error && error->empty()) *error = "invalid ZkSumcheck round header";
        return false;
    }
    parsed.link_kind = static_cast<ZkSumcheckLinkKind>(link);
    std::vector<byte> initial;
    std::vector<byte> previous;
    if (!zk_sumcheck_read_optional_nested(in, offset, initial) ||
        !zk_sumcheck_read_optional_nested(in, offset, previous)) {
        if (error) *error = "invalid ZkSumcheck claim branch framing";
        return false;
    }
    if (parsed.link_kind == ZkSumcheckLinkKind::InitialPublicClaim) {
        if (initial.empty() || !previous.empty() ||
            !deserialize_pedersen_scalar_value(initial, parsed.initial_claim_binding, error) ||
            !zk_sumcheck_scalar_equality_default(parsed.previous_claim_binding)) {
            if (error && error->empty()) *error = "invalid ZkSumcheck initial claim branch";
            return false;
        }
    } else {
        if (!initial.empty() || previous.empty() ||
            !zk_sumcheck_scalar_value_default(parsed.initial_claim_binding) ||
            !deserialize_pedersen_scalar_equality(previous,
                                                  parsed.previous_claim_binding, error)) {
            if (error && error->empty()) *error = "invalid ZkSumcheck previous claim branch";
            return false;
        }
    }
    if (!read_g1(in, offset, parsed.evaluation_commitment) ||
        !zk_sumcheck_read_nested(in, offset, nested) ||
        !deserialize_pedersen_vector_linear(nested, parsed.evaluation_linear, error) ||
        !zk_sumcheck_round_shape_valid(parsed, degree, error)) {
        if (error && error->empty()) *error = "invalid ZkSumcheck round proof";
        return false;
    }
    round = parsed;
    return true;
}

inline std::uint32_t zk_sumcheck_round_count(std::uint32_t domain_size) {
    std::uint32_t rounds = 0U;
    while (domain_size > 1U) {
        domain_size >>= 1U;
        ++rounds;
    }
    return rounds;
}

inline bool zk_sumcheck_read_digest(const std::vector<byte>& in, size_t& offset,
                                    std::array<byte, 32U>& digest) {
    if (offset > in.size() || in.size() - offset < digest.size()) return false;
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + digest.size()),
              digest.begin());
    offset += digest.size();
    return true;
}

inline void serialize_zk_sumcheck(const ZkSumcheckProof& proof,
                                  std::vector<byte>& out) {
    std::string error;
    if (!zk_sumcheck_shape_valid(proof, &error))
        throw std::invalid_argument(error.empty() ? "invalid ZkSumcheck proof" : error);
    out.insert(out.end(), {'Z', 'S', 'C', '1'});
    append_u32(out, proof.version);
    append_string(out, proof.label);
    append_u32(out, proof.degree);
    append_u32(out, proof.domain_size);
    append_pedersen_context(out, proof.initial_claim_context);
    append_u64(out, static_cast<std::uint64_t>(proof.rounds.size()));
    for (const auto& round : proof.rounds) serialize_zk_sumcheck_round(round, out);
    append_g1(out, proof.final_claim_commitment);
    out.insert(out.end(), proof.descriptor_digest.begin(), proof.descriptor_digest.end());
    out.insert(out.end(), proof.registry_digest.begin(), proof.registry_digest.end());
    out.insert(out.end(), proof.srs_digest.begin(), proof.srs_digest.end());
    out.insert(out.end(), proof.basis_digest.begin(), proof.basis_digest.end());
    append_g1(out, proof.blinding_generator);
    append_g1(out, proof.u_generator);
}

inline bool deserialize_zk_sumcheck(const std::vector<byte>& in,
                                    ZkSumcheckProof& proof,
                                    std::string* error = nullptr) {
    ZkSumcheckProof parsed;
    std::uint64_t rounds = 0U;
    size_t offset = 0U;
    if (in.size() < 4U || in[0] != 'Z' || in[1] != 'S' || in[2] != 'C' || in[3] != '1') {
        if (error) *error = "invalid ZkSumcheck proof magic";
        return false;
    }
    offset = 4U;
    if (!read_u32(in, offset, parsed.version) || parsed.version != ZkSumcheckProof::kVersion ||
        !read_string(in, offset, parsed.label, 4096U) ||
        !read_u32(in, offset, parsed.degree) || !read_u32(in, offset, parsed.domain_size) ||
        !read_pedersen_context(in, offset, parsed.initial_claim_context) ||
        !read_u64(in, offset, rounds) || rounds > ZkSumcheckProof::kMaxRounds ||
        rounds != zk_sumcheck_round_count(parsed.domain_size)) {
        if (error) *error = "invalid ZkSumcheck proof header or round count";
        return false;
    }
    parsed.rounds.resize(static_cast<std::size_t>(rounds));
    for (auto& round : parsed.rounds)
        if (!deserialize_zk_sumcheck_round(in, offset, round, parsed.degree, error)) return false;
    if (!read_g1(in, offset, parsed.final_claim_commitment) ||
        !zk_sumcheck_read_digest(in, offset, parsed.descriptor_digest) ||
        !zk_sumcheck_read_digest(in, offset, parsed.registry_digest) ||
        !zk_sumcheck_read_digest(in, offset, parsed.srs_digest) ||
        !zk_sumcheck_read_digest(in, offset, parsed.basis_digest) ||
        !read_g1(in, offset, parsed.blinding_generator) ||
        !read_g1(in, offset, parsed.u_generator) || offset != in.size()) {
        if (error) *error = "truncated or trailing ZkSumcheck proof";
        return false;
    }
    if (!zk_sumcheck_shape_valid(parsed, error)) return false;
    proof = parsed;
    return true;
}

struct CommitmentRecord {
    std::string label;
    // 1 = registered model, 2 = per-inference dynamic tensor.
    std::uint32_t visibility;
    std::vector<G1Jacobian_t> points;
    // The canonical proof wire exposes only the blinded commitment and its
    // independent H.  A bare C0/value commitment is intentionally not a
    // sidecar field: the IPA opening itself proves the committed tensor
    // against C=<a,G>+rH while the prover retains r privately.
    G1Jacobian_t blinding_generator;
    G1Jacobian_t commitment;

    CommitmentRecord()
        : label(), visibility(0), points(), blinding_generator{}, commitment{} {}
};

/* A binding block is deliberately generic: GAT adapter code supplies the
 * operation label and terminal claims while native zkMatMul/zkProd fill the
 * same Sumcheck wire format. */
struct BindingBlock {
    std::string label;
    std::uint32_t operation_kind;
    std::vector<Fr_t> input_claims;
    std::vector<Fr_t> output_claims;
    SumcheckProof sumcheck;

    BindingBlock() : label(), operation_kind(0), input_claims(), output_claims(), sumcheck() {}
};

enum : std::uint32_t {
    kOperationMatMul = 1U,
    kOperationProduct = 2U,
    kOperationOutputAverage = 3U
};

inline bool verify_sumcheck(const SumcheckProof& proof, const Fr_t& initial_claim,
                            Transcript& transcript, std::vector<Fr_t>* challenges,
                            std::string* error);

/* The operation-specific polynomial is still supplied by the native
 * prover.  This common check makes the output claim explicit and prevents a
 * caller from treating a Sumcheck terminal value as an unbound intermediate.
 * The caller supplies the independently reconstructed initial claim. */
inline bool verify_output_binding(const BindingBlock& block, const Fr_t& initial_claim,
                                  const Fr_t& expected_output, Transcript& transcript,
                                  std::string* error = nullptr) {
    if (block.output_claims.size() != 1U) {
        if (error) *error = "operation does not carry exactly one output claim";
        return false;
    }
    if (!verify_sumcheck(block.sumcheck, initial_claim, transcript, nullptr, error)) return false;
    if (fr_compare(fr_wire(block.sumcheck.final_claim), fr_wire(block.output_claims[0])) != 0 ||
        fr_compare(fr_wire(block.output_claims[0]), fr_wire(expected_output)) != 0) {
        if (error) *error = "operation output claim is not bound to expected output";
        return false;
    }
    return true;
}

struct ProofEnvelope {
    static const std::uint32_t kVersion = 2;
    std::uint32_t version;
    std::string protocol_id;
    std::array<byte, 32> statement_digest;
    std::vector<byte> public_output;
    std::vector<CommitmentRecord> commitments;
    std::vector<BindingBlock> bindings;
    std::vector<EvaluationOpeningProof> openings;
    std::vector<SumcheckProof> sumchecks;

    ProofEnvelope()
        : version(kVersion), protocol_id("zkdl-gat-proof-v2"), statement_digest{{}},
          public_output(), commitments(), bindings(), openings(), sumchecks() {}

    std::vector<byte> serialize() const {
        std::vector<byte> out;
        out.reserve(256);
        out.push_back('Z'); out.push_back('K'); out.push_back('D'); out.push_back('L');
        append_u32(out, version);
        append_string(out, protocol_id);
        out.insert(out.end(), statement_digest.begin(), statement_digest.end());
        append_u64(out, static_cast<std::uint64_t>(public_output.size()));
        out.insert(out.end(), public_output.begin(), public_output.end());
        append_u64(out, static_cast<std::uint64_t>(commitments.size()));
        for (size_t i = 0; i < commitments.size(); ++i) {
            append_string(out, commitments[i].label);
            append_u32(out, commitments[i].visibility);
            append_u64(out, static_cast<std::uint64_t>(commitments[i].points.size()));
            for (size_t j = 0; j < commitments[i].points.size(); ++j) append_g1(out, commitments[i].points[j]);
            append_g1(out, commitments[i].blinding_generator);
            append_g1(out, commitments[i].commitment);
        }
        append_u64(out, static_cast<std::uint64_t>(bindings.size()));
        for (size_t i = 0; i < bindings.size(); ++i) {
            append_string(out, bindings[i].label);
            append_u32(out, bindings[i].operation_kind);
            append_u64(out, static_cast<std::uint64_t>(bindings[i].input_claims.size()));
            for (size_t j = 0; j < bindings[i].input_claims.size(); ++j) append_fr(out, bindings[i].input_claims[j]);
            append_u64(out, static_cast<std::uint64_t>(bindings[i].output_claims.size()));
            for (size_t j = 0; j < bindings[i].output_claims.size(); ++j) append_fr(out, bindings[i].output_claims[j]);
            std::vector<byte> encoded;
            serialize_sumcheck(bindings[i].sumcheck, encoded);
            append_u64(out, static_cast<std::uint64_t>(encoded.size()));
            out.insert(out.end(), encoded.begin(), encoded.end());
        }
        append_u64(out, static_cast<std::uint64_t>(openings.size()));
        for (size_t i = 0; i < openings.size(); ++i) serialize_opening(openings[i], out);
        append_u64(out, static_cast<std::uint64_t>(sumchecks.size()));
        for (size_t i = 0; i < sumchecks.size(); ++i) {
            std::vector<byte> encoded;
            serialize_sumcheck(sumchecks[i], encoded);
            append_u64(out, static_cast<std::uint64_t>(encoded.size()));
            out.insert(out.end(), encoded.begin(), encoded.end());
        }
        return out;
    }

    static bool deserialize(const std::vector<byte>& in, ProofEnvelope& envelope,
                            std::string* error = nullptr) {
        const size_t max_items = (1U << 20);
        size_t offset = 0;
        ProofEnvelope parsed;
        std::uint32_t version = 0;
        std::uint64_t count = 0, bytes = 0;
        if (in.size() < 4 || in[0] != 'Z' || in[1] != 'K' || in[2] != 'D' || in[3] != 'L') {
            if (error) *error = "invalid zkDL proof magic"; return false;
        }
        offset = 4;
        if (!read_u32(in, offset, version) || version != kVersion ||
            !read_string(in, offset, parsed.protocol_id) ||
            parsed.protocol_id != "zkdl-gat-proof-v2" ||
            in.size() - offset < parsed.statement_digest.size()) {
            if (error) *error = "unsupported zkDL proof version"; return false;
        }
        parsed.version = version;
        std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
                  in.begin() + static_cast<std::ptrdiff_t>(offset + parsed.statement_digest.size()),
                  parsed.statement_digest.begin());
        offset += parsed.statement_digest.size();
        if (!read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
        parsed.public_output.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                                    in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
        offset += static_cast<size_t>(bytes);
        if (!read_u64(in, offset, count) || count > max_items) return false;
        parsed.commitments.reserve(static_cast<size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            CommitmentRecord record;
            if (!read_string(in, offset, record.label) || !read_u32(in, offset, record.visibility) ||
                !read_u64(in, offset, bytes) || bytes > max_items) return false;
            record.points.resize(static_cast<size_t>(bytes));
            for (size_t j = 0; j < record.points.size(); ++j) if (!read_g1(in, offset, record.points[j])) return false;
            if (!read_g1(in, offset, record.blinding_generator) ||
                !read_g1(in, offset, record.commitment)) return false;
            parsed.commitments.push_back(record);
        }
        if (!read_u64(in, offset, count) || count > max_items) return false;
        parsed.bindings.reserve(static_cast<size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            BindingBlock binding;
            if (!read_string(in, offset, binding.label) || !read_u32(in, offset, binding.operation_kind) ||
                !read_u64(in, offset, bytes) || bytes > max_items) return false;
            binding.input_claims.resize(static_cast<size_t>(bytes));
            for (size_t j = 0; j < binding.input_claims.size(); ++j) if (!read_fr(in, offset, binding.input_claims[j])) return false;
            if (!read_u64(in, offset, bytes) || bytes > max_items) return false;
            binding.output_claims.resize(static_cast<size_t>(bytes));
            for (size_t j = 0; j < binding.output_claims.size(); ++j) if (!read_fr(in, offset, binding.output_claims[j])) return false;
            if (!read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
            std::vector<byte> encoded(in.begin() + static_cast<std::ptrdiff_t>(offset),
                                      in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
            offset += static_cast<size_t>(bytes);
            size_t inner = 0;
            if (!deserialize_sumcheck(encoded, inner, binding.sumcheck) || inner != encoded.size()) return false;
            parsed.bindings.push_back(binding);
        }
        if (!read_u64(in, offset, count) || count > max_items) return false;
        parsed.openings.resize(static_cast<size_t>(count));
        for (size_t i = 0; i < parsed.openings.size(); ++i)
            if (!deserialize_opening(in, offset, parsed.openings[i])) return false;
        if (!read_u64(in, offset, count) || count > max_items) return false;
        parsed.sumchecks.resize(static_cast<size_t>(count));
        for (size_t i = 0; i < parsed.sumchecks.size(); ++i) {
            if (!read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
            std::vector<byte> encoded(in.begin() + static_cast<std::ptrdiff_t>(offset),
                                      in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
            offset += static_cast<size_t>(bytes);
            size_t inner = 0;
            if (!deserialize_sumcheck(encoded, inner, parsed.sumchecks[i]) || inner != encoded.size()) return false;
        }
        if (offset != in.size()) { if (error) *error = "trailing proof bytes"; return false; }
        envelope = parsed;
        return true;
    }

    bool verify_public_binding(const std::array<byte, 32>& expected_statement,
                               const std::vector<byte>& expected_output,
                               std::string* error = nullptr) const {
        if (statement_digest != expected_statement) {
            if (error) *error = "proof statement digest mismatch";
            return false;
        }
        if (public_output != expected_output) {
            if (error) *error = "proof output is not the verifier supplied output";
            return false;
        }
        return true;
    }
};

inline FrWire interpolate_at(const std::vector<Fr_t>& values, const FrWire& x) {
    if (values.empty()) throw std::invalid_argument("sumcheck round has no evaluations");
    FrWire result{};
    for (size_t i = 0; i < values.size(); ++i) {
        FrWire term = fr_wire(values[i]);
        const FrWire xi = fr_from_u32(static_cast<std::uint32_t>(i));
        for (size_t j = 0; j < values.size(); ++j) if (i != j) {
            const FrWire xj = fr_from_u32(static_cast<std::uint32_t>(j));
            term = fr_mul(term, fr_sub(x, xj));
            term = fr_mul(term, fr_pow(fr_sub(xi, xj), 255, fr_modulus_minus_two()));
        }
        result = fr_add(result, term);
    }
    return result;
}

inline bool verify_sumcheck(const SumcheckProof& proof, const Fr_t& initial_claim,
                            Transcript& transcript, std::vector<Fr_t>* challenges = nullptr,
                            std::string* error = nullptr) {
    if (proof.degree == 0 || proof.degree > 16) {
        if (error) *error = "invalid sumcheck degree";
        return false;
    }
    FrWire claim = fr_wire(initial_claim);
    if (!fr_canonical(claim)) { if (error) *error = "non-canonical initial claim"; return false; }
    if (challenges) challenges->clear();
    for (size_t round = 0; round < proof.rounds.size(); ++round) {
        const SumcheckRound& message = proof.rounds[round];
        if (message.evaluations.size() != static_cast<size_t>(proof.degree + 1)) {
            if (error) *error = "wrong sumcheck round arity";
            return false;
        }
        FrWire sum01 = fr_add(fr_wire(message.evaluations[0]), fr_wire(message.evaluations[1]));
        if (fr_compare(sum01, claim) != 0) {
            if (error) *error = "sumcheck boolean endpoint equation failed";
            return false;
        }
        transcript.absorb_fr(proof.label + ".round", message.evaluations);
        const Fr_t challenge = transcript.challenge(proof.label + ".challenge");
        if (challenges) challenges->push_back(challenge);
        claim = interpolate_at(message.evaluations, fr_wire(challenge));
    }
    if (fr_compare(claim, fr_wire(proof.final_claim)) != 0) {
        if (error) *error = "sumcheck terminal claim mismatch";
        return false;
    }
    return true;
}

} // namespace zkdl


vector<Fr_t> random_vec(uint len)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<unsigned int> dist(0, UINT_MAX);
    vector<Fr_t> out(len);
    for (uint i = 0; i < len; ++ i) out[i] = {dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt) % 1944954707};
    return out;
}

uint ceilLog2(uint num) {
    if (num == 0) return 0;
    
    // Decrease num to handle the case where num is already a power of 2
    num--;

    uint result = 0;
    
    // Keep shifting the number to the right until it becomes zero. 
    // Each shift means the number is halved, which corresponds to 
    // a division by 2 in logarithmic terms.
    while (num > 0) {
        num >>= 1;
        result++;
    }

    return result;
}

template<typename T>
std::vector<T> concatenate(const std::vector<std::vector<T>>& vecs) {
    // First, compute the total size for the result vector.
    size_t totalSize = 0;
    for (const auto& v : vecs) {
        totalSize += v.size();
    }

    // Allocate space for the result vector.
    std::vector<T> result;
    result.reserve(totalSize);

    // Append each vector's contents to the result vector.
    for (const auto& v : vecs) {
        result.insert(result.end(), v.begin(), v.end());
    }

    return result;
}


KERNEL void Fr_ip_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;
    Fr_t a0 = (gid0 < in_size) ? a[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t b0 = (gid0 < in_size) ? b[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (gid1 < in_size) ? a[gid1] : blstrs__scalar__Scalar_ZERO;
    Fr_t b1 = (gid1 < in_size) ? b[gid1] : blstrs__scalar__Scalar_ZERO;
    out0[gid] = Fr_field_mul(a0, b0);
    out1[gid] = blstrs__scalar__Scalar_add(Fr_field_mul(a0, blstrs__scalar__Scalar_sub(b1, b0)),
        Fr_field_mul(b0, blstrs__scalar__Scalar_sub(a1, a0)));
    out2[gid] = Fr_field_mul(blstrs__scalar__Scalar_sub(a1, a0), blstrs__scalar__Scalar_sub(b1, b0));
}

void Fr_ip_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<Fr_t>& proof)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (begin >= end) {
        proof.push_back(a(0));
        proof.push_back(b(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_ip_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, b.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    cudaDeviceSynchronize();
    proof.push_back(out0.sum());
    proof.push_back(out1.sum());
    proof.push_back(out2.sum());

    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(b.gpu_data, b_new.gpu_data, *begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_ip_sc(a_new, b_new, begin + 1, end, proof);
}

vector<Fr_t> inner_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u)
{
    vector<Fr_t> proof;
    uint log_size = u.size();
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (a.size <= (1 << (log_size))/2) throw std::runtime_error("Incompatible dimensions");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions");

    Fr_ip_sc(a, b, u.begin(), u.end(), proof);
    return proof;
}

void Fr_hp_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions 5");
    if (v_end - v_begin != u_end - u_begin) throw std::runtime_error("Incompatible dimensions 6");
    if (v_begin >= v_end) {
        proof.push_back(a(0));
        proof.push_back(b(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_ip_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, b.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    cudaDeviceSynchronize();
    vector<Fr_t> u_(u_begin + 1, u_end);
    //std::cout << u_.size() << "\t" << out0.size << "\t" << out1.size << "\t" << out2.size << std::endl;
    proof.push_back(out0(u_));
    proof.push_back(out1(u_));
    proof.push_back(out2(u_));

    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *v_begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(b.gpu_data, b_new.gpu_data, *v_begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_hp_sc(a_new, b_new, u_begin + 1, u_end, v_begin + 1, v_end, proof);
}

vector<Fr_t> hadamard_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u, vector<Fr_t> v)
{
    vector<Fr_t> proof;
    if (u.size() != v.size()) throw std::runtime_error("Incompatible dimensions 1");
    uint log_size = u.size();
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions 2");
    if (a.size <= (1 << (log_size - 1))) throw std::runtime_error("Incompatible dimensions 3");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions 4");

    Fr_hp_sc(a, b, u.begin(), u.end(), v.begin(), v.end(), proof);
    return proof;
}

KERNEL void Fr_bin_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    Fr_t a0 = (2 * gid < in_size) ? a[2 * gid] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (2 * gid + 1 < in_size) ? a[2 * gid + 1] : blstrs__scalar__Scalar_ZERO;
    out0[gid] = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_mul(a0, a0), a0);
    Fr_t diff = blstrs__scalar__Scalar_sub(a1, a0);
    out1[gid] = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_double(a0), diff), diff);
    out2[gid] = blstrs__scalar__Scalar_sqr(diff);
}

void Fr_bin_sc(const FrTensor& a, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof)
{
    if (v_end - v_begin != u_end - u_begin) throw std::runtime_error("Incompatible dimensions 6");
    if (v_begin >= v_end) {
        proof.push_back(a(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_bin_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    cudaDeviceSynchronize();
    vector<Fr_t> u_(u_begin + 1, u_end);
    //std::cout << u_.size() << "\t" << out0.size << "\t" << out1.size << "\t" << out2.size << std::endl;
    proof.push_back(out0(u_));
    proof.push_back(out1(u_));
    proof.push_back(out2(u_));

    FrTensor a_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *v_begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_bin_sc(a_new, u_begin + 1, u_end, v_begin + 1, v_end, proof);
}

vector<Fr_t> binary_sumcheck(const FrTensor& a, vector<Fr_t> u, vector<Fr_t> v)
{
    vector<Fr_t> proof;
    if (u.size() != v.size()) throw std::runtime_error("Incompatible dimensions");
    uint log_size = u.size();
    if (a.size <= (1 << (log_size))/2) throw std::runtime_error("Incompatible dimensions");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions");

    Fr_bin_sc(a, u.begin(), u.end(), v.begin(), v.end(), proof);
    return proof;
}

#endif
