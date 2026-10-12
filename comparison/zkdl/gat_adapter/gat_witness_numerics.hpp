#pragma once

#include "gat_adapter.hpp"

#include <openssl/bn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace zkdl_gat::witness_numerics {

using i64 = std::int64_t;

[[noreturn]] inline void fail(const std::string &message) { throw std::runtime_error("zkdl-gat: " + message); }

class BigNum {
  public:
    BigNum() : value_(BN_new()) {
        if (!value_) fail("BN_new failed");
        BN_zero(value_);
    }
    BigNum(const BigNum &) = delete;
    BigNum &operator=(const BigNum &) = delete;
    BigNum(BigNum &&other) noexcept : value_(other.value_) { other.value_ = nullptr; }
    BigNum &operator=(BigNum &&other) noexcept {
        if (this != &other) {
            if (value_) BN_free(value_);
            value_ = other.value_;
            other.value_ = nullptr;
        }
        return *this;
    }
    ~BigNum() {
        if (value_) BN_free(value_);
    }
    BIGNUM *get() { return value_; }
    const BIGNUM *get() const { return value_; }
    BIGNUM **address() { return &value_; }

  private:
    BIGNUM *value_;
};

inline const BIGNUM *scalar_modulus() {
    static BIGNUM *modulus = [] {
        BIGNUM *result = nullptr;
        if (BN_hex2bn(&result, "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001") == 0)
            fail("BN_hex2bn failed for scalar modulus");
        return result;
    }();
    return modulus;
}

inline std::string scalar_modulus_decimal() {
    char *decimal = BN_bn2dec(scalar_modulus());
    if (!decimal) fail("BN_bn2dec failed for scalar modulus");
    const std::string result(decimal);
    OPENSSL_free(decimal);
    return result;
}

inline BigNum field_from_i64(i64 value, BN_CTX *context) {
    (void)context;
    BigNum result;
    const std::uint64_t magnitude = value < 0 ? static_cast<std::uint64_t>(-(value + 1)) + 1U : static_cast<std::uint64_t>(value);
    std::array<std::uint8_t, sizeof(magnitude)> little_endian{};
    for (std::size_t index = 0U; index < little_endian.size(); ++index) little_endian[index] = static_cast<std::uint8_t>(magnitude >> (8U * index));
    if (!BN_lebin2bn(little_endian.data(), static_cast<int>(little_endian.size()), result.get())) fail("BN_lebin2bn failed for signed field value");
    if (value < 0 && !BN_sub(result.get(), scalar_modulus(), result.get())) fail("BN_sub failed for signed field value");
    return result;
}

inline BigNum field_from_value(const FieldValue &value) {
    BigNum result;
    if (!BN_lebin2bn(value.bytes.data(), static_cast<int>(value.bytes.size()), result.get()) || BN_cmp(result.get(), scalar_modulus()) >= 0)
        fail("non-canonical field witness encoding");
    return result;
}

inline FieldValue field_encode(const BIGNUM *input) {
    if (BN_num_bytes(input) > 32) fail("field encoding exceeds 256 bits");
    std::array<std::uint8_t, 32> big_endian{};
    const int size = BN_bn2binpad(input, big_endian.data(), static_cast<int>(big_endian.size()));
    if (size != static_cast<int>(big_endian.size())) fail("BN_bn2binpad failed");
    FieldValue output;
    for (std::size_t i = 0; i < output.bytes.size(); ++i) output.bytes[i] = big_endian[output.bytes.size() - i - 1];
    return output;
}

inline BigNum field_inverse(i64 value, BN_CTX *context) {
    BigNum base = field_from_i64(value, context);
    if (BN_is_zero(base.get())) fail("zero denominator in attention relation");
    BigNum result;
    if (!BN_mod_inverse(result.get(), base.get(), scalar_modulus(), context)) fail("field inverse failed");
    return result;
}

constexpr std::size_t kFieldInverseBatchSize = 4096U;

template <typename Store>
inline void batch_field_inverses(const std::vector<i64> &values, BN_CTX *context, const std::string &name, Store store) {
    if (!context) fail("null BN_CTX for batch field inversion in " + name);
    for (std::size_t begin = 0U; begin < values.size(); begin += std::min(kFieldInverseBatchSize, values.size() - begin)) {
        const std::size_t count = std::min(kFieldInverseBatchSize, values.size() - begin);
        BN_CTX_start(context);
        try {
            BIGNUM *product = BN_CTX_get(context);
            BIGNUM *inverse = BN_CTX_get(context);
            BIGNUM *output = BN_CTX_get(context);
            BIGNUM *next_inverse = BN_CTX_get(context);
            BIGNUM *one = BN_CTX_get(context);
            if (!product || !inverse || !output || !next_inverse || !one) fail("BN_CTX_get failed for batch field inversion in " + name);
            if (!BN_one(product) || !BN_one(one)) fail("BN_one failed for batch field inversion in " + name);
            std::vector<std::size_t> indices;
            std::vector<BIGNUM *> inputs;
            std::vector<BIGNUM *> prefixes;
            indices.reserve(count);
            inputs.reserve(count);
            prefixes.reserve(count);
            for (std::size_t offset = 0U; offset < count; ++offset) {
                const std::size_t index = begin + offset;
                if (values[index] == 0) {
                    store(index, static_cast<const BIGNUM *>(nullptr));
                    continue;
                }
                BIGNUM *input = BN_CTX_get(context);
                BIGNUM *prefix = BN_CTX_get(context);
                if (!input || !prefix) fail("BN_CTX_get failed for batch field input in " + name);
                BigNum encoded = field_from_i64(values[index], context);
                if (!BN_copy(input, encoded.get()) || !BN_mod_mul(prefix, product, input, scalar_modulus(), context) || !BN_copy(product, prefix))
                    fail("batch field prefix multiplication failed in " + name);
                indices.push_back(index);
                inputs.push_back(input);
                prefixes.push_back(prefix);
            }
            if (!indices.empty()) {
                if (!BN_mod_inverse(inverse, product, scalar_modulus(), context)) fail("batch field inverse failed in " + name);
                for (std::size_t item = indices.size(); item-- > 0U;) {
                    const BIGNUM *prefix_before = item == 0U ? one : prefixes[item - 1U];
                    if (!BN_mod_mul(output, inverse, prefix_before, scalar_modulus(), context)) fail("batch field inverse output failed in " + name);
                    store(indices[item], static_cast<const BIGNUM *>(output));
                    if (!BN_mod_mul(next_inverse, inverse, inputs[item], scalar_modulus(), context) || !BN_copy(inverse, next_inverse))
                        fail("batch field inverse back-substitution failed in " + name);
                }
            }
        } catch (...) {
            BN_CTX_end(context);
            throw;
        }
        BN_CTX_end(context);
    }
}

struct FieldInverseCache {
    explicit FieldInverseCache(std::size_t nodes) : by_node(nodes) {}
    std::vector<std::optional<BigNum>> by_node;
};

template <typename Head>
inline FieldInverseCache fill_field_attention(Head &head, const Graph &graph, std::uint64_t edge_begin, std::uint64_t edge_end,
                                               const std::vector<i64> &denominator_by_node) {
    FieldInverseCache inverses(static_cast<std::size_t>(graph.nodes));
    BN_CTX *context = BN_CTX_new();
    if (!context) fail("BN_CTX_new failed");
    if (denominator_by_node.size() != graph.nodes) {
        BN_CTX_free(context);
        fail("attention denominator node shape mismatch");
    }
    head.inverse_field.resize(static_cast<std::size_t>(graph.edges));
    head.alpha_field.resize(static_cast<std::size_t>(graph.edges));
    const std::string head_kind = std::is_same<Head, HiddenHeadTrace>::value ? "hidden" : "output";
    try {
        batch_field_inverses(denominator_by_node, context, head_kind + " attention denominator", [&](std::size_t node, const BIGNUM *inverse) {
            if (!inverse) return;
            BigNum cached;
            if (!BN_copy(cached.get(), inverse)) fail("cannot cache " + head_kind + " attention inverse");
            inverses.by_node[node].emplace(std::move(cached));
        });
        for (std::uint64_t edge = edge_begin; edge < edge_end; ++edge) {
            const std::uint64_t node = graph.dst.at(static_cast<std::size_t>(edge));
            if (node >= inverses.by_node.size() || !inverses.by_node[node]) fail("zero denominator in attention relation");
            const BigNum &inverse = *inverses.by_node[node];
            BigNum exponent = field_from_i64(head.exp_weight.at(static_cast<std::size_t>(edge)), context);
            BigNum alpha;
            if (!BN_mod_mul(alpha.get(), exponent.get(), inverse.get(), scalar_modulus(), context)) fail("field alpha multiplication failed");
            head.inverse_field[static_cast<std::size_t>(edge)] = field_encode(inverse.get());
            head.alpha_field[static_cast<std::size_t>(edge)] = field_encode(alpha.get());
        }
    } catch (...) {
        BN_CTX_free(context);
        throw;
    }
    BN_CTX_free(context);
    return inverses;
}

inline i64 checked_i128(__int128 value, const std::string &what) {
    if (value < static_cast<__int128>(std::numeric_limits<i64>::min()) || value > static_cast<__int128>(std::numeric_limits<i64>::max()))
        fail("signed 64-bit overflow in " + what);
    return static_cast<i64>(value);
}

inline i64 round_div(i64 numerator, i64 denominator, const std::string &what) {
    if (denominator <= 0) fail("non-positive divisor in " + what);
    const __int128 n = numerator;
    const __int128 d = denominator;
    if (n >= 0) return checked_i128((n + d / 2) / d, what);
    return checked_i128(-((-n + d / 2) / d), what);
}

inline FieldValue aggregate_consistency_residual(const BIGNUM *aggregate, i64 denominator, i64 aggregate_num, BN_CTX *context,
                                                 const std::string &name) {
    if (!context) fail("null BN_CTX for aggregate consistency in " + name);
    BigNum denominator_value = field_from_i64(denominator, context);
    BigNum integer_value = field_from_i64(aggregate_num, context);
    BigNum rhs;
    BigNum residual;
    if (!BN_mod_mul(rhs.get(), denominator_value.get(), aggregate, scalar_modulus(), context) ||
        !BN_mod_sub(residual.get(), rhs.get(), integer_value.get(), scalar_modulus(), context))
        fail("aggregate consistency field arithmetic failed in " + name);
    return field_encode(residual.get());
}

inline SignedRoundDivWitness make_round_div_witness(const Matrix &numerator, const std::vector<i64> &denominator, std::uint64_t numerator_scale,
                                                    std::uint64_t denominator_scale, const Matrix &expected, const std::string &name) {
    if (numerator.rows != expected.rows || numerator.cols != expected.cols || denominator.size() != numerator.rows || numerator_scale == 0 ||
        denominator_scale == 0)
        fail("SignedRoundDiv shape/scale mismatch in " + name);
    const auto auxiliary_start = std::chrono::steady_clock::now();
    SignedRoundDivWitness witness;
    witness.rows = numerator.rows;
    witness.cols = numerator.cols;
    const std::size_t count = numerator.values.size();
    witness.sigma.assign(count, 0);
    witness.z.assign(count, 0);
    witness.a.assign(count, 0);
    witness.q.assign(count, 0);
    witness.r.assign(count, 0);
    witness.delta.assign(count, 0);
    witness.divisor.assign(count, 0);
    witness.a_inverse.assign(count, FieldValue{});
    witness.divisor_inverse.assign(count, FieldValue{});
    std::vector<i64> divisor_by_row(static_cast<std::size_t>(numerator.rows));
    std::vector<std::size_t> divisor_slot_by_row(static_cast<std::size_t>(numerator.rows));
    std::vector<i64> unique_divisors;
    std::map<i64, std::size_t> unique_divisor_slots;
    unique_divisors.reserve(static_cast<std::size_t>(numerator.rows));
    for (std::uint64_t row = 0; row < numerator.rows; ++row) {
        const i64 divisor = checked_i128(static_cast<__int128>(denominator[row]) * denominator_scale, name + ".divisor");
        if (divisor <= 0) fail("SignedRoundDiv divisor is not positive in " + name);
        divisor_by_row[static_cast<std::size_t>(row)] = divisor;
        const auto slot = unique_divisor_slots.find(divisor);
        if (slot == unique_divisor_slots.end()) {
            const std::size_t index = unique_divisors.size();
            unique_divisors.push_back(divisor);
            unique_divisor_slots.emplace(divisor, index);
            divisor_slot_by_row[static_cast<std::size_t>(row)] = index;
        } else {
            divisor_slot_by_row[static_cast<std::size_t>(row)] = slot->second;
        }
    }
    BN_CTX *context = BN_CTX_new();
    if (!context) fail("BN_CTX_new failed for " + name);
    try {
        std::vector<FieldValue> divisor_inverse_by_slot(unique_divisors.size());
        batch_field_inverses(unique_divisors, context, name + ".divisor_inverse", [&](std::size_t index, const BIGNUM *inverse) {
            if (!inverse) fail("SignedRoundDiv divisor inverse is zero in " + name);
            divisor_inverse_by_slot[index] = field_encode(inverse);
        });
        for (std::uint64_t row = 0; row < numerator.rows; ++row) {
            const i64 divisor = divisor_by_row[static_cast<std::size_t>(row)];
            for (std::uint64_t col = 0; col < numerator.cols; ++col) {
                const std::size_t index = static_cast<std::size_t>(row * numerator.cols + col);
                const i64 x = checked_i128(static_cast<__int128>(numerator.at(row, col)) * numerator_scale, name + ".dividend");
                const i64 y = expected.at(row, col);
                const __int128 abs_x = x < 0 ? -static_cast<__int128>(x) : static_cast<__int128>(x);
                const __int128 abs_y = y < 0 ? -static_cast<__int128>(y) : static_cast<__int128>(y);
                const __int128 remainder = 2 * abs_x + divisor - 2 * static_cast<__int128>(divisor) * abs_y;
                const __int128 slack = 2 * static_cast<__int128>(divisor) - 1 - remainder;
                if (remainder < 0 || slack < 0) fail("invalid nearest-ties-away witness in " + name);
                witness.sigma[index] = static_cast<std::uint8_t>(x < 0 ? 1U : 0U);
                witness.z[index] = static_cast<std::uint8_t>(x == 0 ? 1U : 0U);
                witness.a[index] = checked_i128(abs_x, name + ".a");
                witness.q[index] = checked_i128(abs_y, name + ".q");
                witness.r[index] = checked_i128(remainder, name + ".r");
                witness.delta[index] = checked_i128(slack, name + ".delta");
                witness.divisor[index] = divisor;
                witness.divisor_inverse[index] = divisor_inverse_by_slot[divisor_slot_by_row[static_cast<std::size_t>(row)]];
            }
        }
        batch_field_inverses(witness.a, context, name + ".a_inverse", [&](std::size_t index, const BIGNUM *inverse) {
            if (inverse) witness.a_inverse[index] = field_encode(inverse);
        });
    } catch (...) {
        BN_CTX_free(context);
        throw;
    }
    BN_CTX_free(context);
    witness.proof_auxiliary_construction_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - auxiliary_start).count();
    return witness;
}

inline void aggregate_field_products(const Graph &graph, const Matrix &source, const std::vector<i64> &denominators,
                                    const std::vector<FieldValue> &alpha_field, const Matrix &aggregate_num, FieldMatrix &products,
                                    FieldMatrix &aggregates, FieldMatrix &residuals, BN_CTX *context, const std::string &name) {
    const auto elements = [&](std::uint64_t rows, std::uint64_t cols) {
        if (rows != 0U && cols > std::numeric_limits<std::size_t>::max() / rows) fail("attention aggregate field shape overflow in " + name);
        return static_cast<std::size_t>(rows * cols);
    };
    if (!context || alpha_field.size() != graph.edges || source.rows != graph.nodes || products.rows != graph.edges || products.cols != source.cols ||
        products.values.size() != elements(graph.edges, source.cols) || denominators.size() != graph.nodes || aggregate_num.rows != graph.nodes ||
        aggregate_num.cols != source.cols || aggregate_num.values.size() != elements(graph.nodes, source.cols) || aggregates.rows != graph.nodes ||
        aggregates.cols != source.cols || aggregates.values.size() != elements(graph.nodes, source.cols) || residuals.rows != graph.nodes ||
        residuals.cols != source.cols || residuals.values.size() != elements(graph.nodes, source.cols) || graph.node_edge_ptr.size() != graph.nodes + 1U ||
        graph.src.size() != graph.edges || graph.dst.size() != graph.edges)
        fail("attention aggregate field shape mismatch in " + name);
    for (std::uint64_t node = 0; node < graph.nodes; ++node) {
        std::vector<BigNum> values(static_cast<std::size_t>(source.cols));
        for (auto &value : values) BN_zero(value.get());
        for (std::uint64_t edge = graph.node_edge_ptr[node]; edge < graph.node_edge_ptr[node + 1U]; ++edge) {
            if (edge >= graph.edges || graph.dst[edge] != node || graph.src[edge] >= source.rows) fail("edge pointer/destination mismatch in " + name);
            const BigNum alpha = field_from_value(alpha_field[static_cast<std::size_t>(edge)]);
            for (std::uint64_t dim = 0; dim < source.cols; ++dim) {
                BigNum scalar = field_from_i64(source.at(graph.src[edge], dim), context);
                BigNum product;
                BigNum &aggregate = values[static_cast<std::size_t>(dim)];
                if (!BN_mod_mul(product.get(), alpha.get(), scalar.get(), scalar_modulus(), context) ||
                    !BN_mod_add(aggregate.get(), aggregate.get(), product.get(), scalar_modulus(), context))
                    fail("aggregate field arithmetic failed in " + name);
                products.values[static_cast<std::size_t>(edge * source.cols + dim)] = field_encode(product.get());
            }
        }
        for (std::uint64_t dim = 0; dim < source.cols; ++dim) {
            const std::size_t index = static_cast<std::size_t>(node * source.cols + dim);
            const BIGNUM *aggregate = values[static_cast<std::size_t>(dim)].get();
            aggregates.values[index] = field_encode(aggregate);
            residuals.values[index] = aggregate_consistency_residual(aggregate, denominators[node], aggregate_num.at(node, dim), context, name);
        }
    }
}

inline Matrix concat_head_aggregates(const std::vector<HiddenHeadTrace> &heads, std::uint64_t rows, std::uint64_t cols_per_head) {
    if (heads.empty() || cols_per_head > std::numeric_limits<std::uint64_t>::max() / heads.size()) fail("hidden layer concatenation shape overflow");
    const std::uint64_t cols = static_cast<std::uint64_t>(heads.size()) * cols_per_head;
    if (rows != 0U && cols > std::numeric_limits<std::size_t>::max() / rows) fail("matrix size overflow");
    Matrix output;
    output.rows = rows;
    output.cols = cols;
    output.values.assign(static_cast<std::size_t>(rows * cols), 0);
    for (std::size_t head = 0; head < heads.size(); ++head) {
        const Matrix &aggregate = heads[head].aggregate;
        if (aggregate.rows != rows || aggregate.cols != cols_per_head || aggregate.values.size() != rows * cols_per_head)
            fail("hidden head aggregate shape mismatch");
        for (std::uint64_t row = 0; row < rows; ++row)
            std::copy(aggregate.values.begin() + static_cast<std::ptrdiff_t>(row * cols_per_head),
                      aggregate.values.begin() + static_cast<std::ptrdiff_t>((row + 1U) * cols_per_head),
                      output.values.begin() + static_cast<std::ptrdiff_t>(row * cols + head * cols_per_head));
    }
    return output;
}

} // namespace zkdl_gat::witness_numerics
