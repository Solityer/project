#include "../gat_adapter/gat_witness_numerics.hpp"
#include "../gat_adapter/signed_round_bounds.hpp"

#include <openssl/bn.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace zkdl_gat;
using namespace zkdl_gat::witness_numerics;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

BigNum legacy_field_from_i64(std::int64_t value, BN_CTX *context) {
    const std::string decimal = std::to_string(value);
    BIGNUM *raw = nullptr;
    if (BN_dec2bn(&raw, decimal.c_str()) == 0 || !raw) throw std::runtime_error("legacy BN_dec2bn failed");
    std::unique_ptr<BIGNUM, decltype(&BN_free)> raw_guard(raw, BN_free);
    BigNum output;
    if (!BN_nnmod(output.get(), raw, scalar_modulus(), context)) throw std::runtime_error("legacy BN_nnmod failed");
    return output;
}

FieldValue legacy_field_encode(const BIGNUM *value) {
    std::array<std::uint8_t, 32> big_endian{};
    if (BN_bn2binpad(value, big_endian.data(), static_cast<int>(big_endian.size())) != static_cast<int>(big_endian.size()))
        throw std::runtime_error("legacy BN_bn2binpad failed");
    FieldValue encoded;
    for (std::size_t index = 0; index < encoded.bytes.size(); ++index) encoded.bytes[index] = big_endian[encoded.bytes.size() - index - 1U];
    return encoded;
}

BigNum legacy_inverse(std::int64_t value, BN_CTX *context) {
    BigNum input = legacy_field_from_i64(value, context);
    if (BN_is_zero(input.get())) throw std::runtime_error("legacy oracle cannot invert zero");
    BigNum inverse;
    if (!BN_mod_inverse(inverse.get(), input.get(), scalar_modulus(), context)) throw std::runtime_error("legacy BN_mod_inverse failed");
    return inverse;
}

bool same_field(const FieldValue &left, const FieldValue &right) {
    return std::memcmp(left.bytes.data(), right.bytes.data(), left.bytes.size()) == 0;
}

void append_u64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64U; shift += 8U) bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_i64(std::vector<std::uint8_t> &bytes, std::int64_t value) { append_u64(bytes, static_cast<std::uint64_t>(value)); }

std::vector<std::uint8_t> serialize_witness(const SignedRoundDivWitness &witness) {
    std::vector<std::uint8_t> bytes;
    append_u64(bytes, witness.rows);
    append_u64(bytes, witness.cols);
    bytes.insert(bytes.end(), witness.sigma.begin(), witness.sigma.end());
    bytes.insert(bytes.end(), witness.z.begin(), witness.z.end());
    for (const auto value : witness.a) append_i64(bytes, value);
    for (const auto value : witness.q) append_i64(bytes, value);
    for (const auto value : witness.r) append_i64(bytes, value);
    for (const auto value : witness.delta) append_i64(bytes, value);
    for (const auto value : witness.divisor) append_i64(bytes, value);
    for (const auto &value : witness.a_inverse) bytes.insert(bytes.end(), value.bytes.begin(), value.bytes.end());
    for (const auto &value : witness.divisor_inverse) bytes.insert(bytes.end(), value.bytes.begin(), value.bytes.end());
    return bytes;
}

std::int64_t legacy_round_div(std::int64_t numerator, std::int64_t denominator) {
    if (denominator <= 0) throw std::runtime_error("legacy round divisor must be positive");
    const __int128 n = numerator;
    const __int128 d = denominator;
    const __int128 result = n >= 0 ? (n + d / 2) / d : -((-n + d / 2) / d);
    if (result < std::numeric_limits<std::int64_t>::min() || result > std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error("legacy round result overflow");
    return static_cast<std::int64_t>(result);
}

SignedRoundDivWitness legacy_round_div_witness(const Matrix &numerator, const std::vector<std::int64_t> &denominator, std::uint64_t numerator_scale,
                                               std::uint64_t denominator_scale, const Matrix &expected, BN_CTX *context) {
    SignedRoundDivWitness witness;
    witness.rows = numerator.rows;
    witness.cols = numerator.cols;
    const std::size_t count = numerator.values.size();
    witness.sigma.assign(count, 0U);
    witness.z.assign(count, 0U);
    witness.a.assign(count, 0);
    witness.q.assign(count, 0);
    witness.r.assign(count, 0);
    witness.delta.assign(count, 0);
    witness.divisor.assign(count, 0);
    witness.a_inverse.assign(count, FieldValue{});
    witness.divisor_inverse.assign(count, FieldValue{});
    for (std::uint64_t row = 0; row < numerator.rows; ++row) {
        const __int128 d128 = static_cast<__int128>(denominator.at(static_cast<std::size_t>(row))) * denominator_scale;
        if (d128 <= 0 || d128 > std::numeric_limits<std::int64_t>::max()) throw std::runtime_error("legacy divisor out of range");
        const auto divisor = static_cast<std::int64_t>(d128);
        for (std::uint64_t col = 0; col < numerator.cols; ++col) {
            const std::size_t index = static_cast<std::size_t>(row * numerator.cols + col);
            const __int128 x128 = static_cast<__int128>(numerator.at(row, col)) * numerator_scale;
            if (x128 < std::numeric_limits<std::int64_t>::min() || x128 > std::numeric_limits<std::int64_t>::max())
                throw std::runtime_error("legacy dividend out of range");
            const auto x = static_cast<std::int64_t>(x128);
            const auto y = expected.at(row, col);
            const __int128 abs_x = x < 0 ? -static_cast<__int128>(x) : static_cast<__int128>(x);
            const __int128 abs_y = y < 0 ? -static_cast<__int128>(y) : static_cast<__int128>(y);
            const __int128 remainder = 2 * abs_x + divisor - 2 * static_cast<__int128>(divisor) * abs_y;
            const __int128 slack = 2 * static_cast<__int128>(divisor) - 1 - remainder;
            if (abs_x > std::numeric_limits<std::int64_t>::max() || abs_y > std::numeric_limits<std::int64_t>::max() || remainder < 0 ||
                remainder > std::numeric_limits<std::int64_t>::max() || slack < 0 || slack > std::numeric_limits<std::int64_t>::max())
                throw std::runtime_error("legacy SignedRoundDiv auxiliary out of range");
            witness.sigma[index] = static_cast<std::uint8_t>(x < 0 ? 1U : 0U);
            witness.z[index] = static_cast<std::uint8_t>(x == 0 ? 1U : 0U);
            witness.a[index] = static_cast<std::int64_t>(abs_x);
            witness.q[index] = static_cast<std::int64_t>(abs_y);
            witness.r[index] = static_cast<std::int64_t>(remainder);
            witness.delta[index] = static_cast<std::int64_t>(slack);
            witness.divisor[index] = divisor;
            const FieldValue divisor_inverse = legacy_field_encode(legacy_inverse(divisor, context).get());
            witness.divisor_inverse[index] = divisor_inverse;
            if (witness.a[index] != 0) witness.a_inverse[index] = legacy_field_encode(legacy_inverse(witness.a[index], context).get());
        }
    }
    return witness;
}

void test_signed_field_encoding(BN_CTX *context) {
    const std::vector<std::int64_t> values = {0, 1, -1, std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max(),
                                              0x0102030405060708LL, -0x0102030405060708LL};
    for (const auto value : values) {
        BigNum actual_bn = field_from_i64(value, context);
        BigNum expected_bn = legacy_field_from_i64(value, context);
        const FieldValue actual = field_encode(actual_bn.get());
        const FieldValue expected = legacy_field_encode(expected_bn.get());
        require(same_field(actual, expected), "signed field encoding differs from legacy BN_nnmod for " + std::to_string(value));
        BigNum decoded = field_from_value(actual);
        require(BN_cmp(decoded.get(), expected_bn.get()) == 0, "canonical field decode mismatch for " + std::to_string(value));
    }
}

void test_scalar_round_div() {
    const std::array<std::int64_t, 13> numerators = {
        std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::min() + 1, -9, -5, -3, -1, 0,
        1, 3, 5, 9, std::numeric_limits<std::int64_t>::max() - 1, std::numeric_limits<std::int64_t>::max()};
    const std::array<std::int64_t, 7> denominators = {1, 2, 3, 4, 5, 7, 11};
    for (std::size_t index = 0; index < numerators.size(); ++index) {
        const std::int64_t denominator = denominators[index % denominators.size()];
        require(round_div(numerators[index], denominator, "numerical_test") == legacy_round_div(numerators[index], denominator),
                "production scalar round_div differs from legacy ties-away result");
    }
}

void test_batch_inverse(BN_CTX *context) {
    const std::size_t count = kFieldInverseBatchSize * 2U + 19U;
    std::vector<std::int64_t> values(count, 0);
    const std::array<std::int64_t, 8> pattern = {1, -1, 7, -7, 17, -17, 1234567, -1234567};
    for (std::size_t index = 8U; index < values.size(); ++index) values[index] = pattern[index % pattern.size()];
    values[4095U] = 13;
    values[4096U] = -13;
    values[8191U] = 0;
    values[8192U] = 13;
    values[7U] = std::numeric_limits<std::int64_t>::min();
    values[6U] = std::numeric_limits<std::int64_t>::max();
    std::vector<FieldValue> actual(values.size());
    std::vector<bool> visited(values.size(), false);
    batch_field_inverses(values, context, "numerical test", [&](std::size_t index, const BIGNUM *inverse) {
        visited[index] = true;
        if (inverse) actual[index] = field_encode(inverse);
    });
    for (std::size_t index = 0; index < values.size(); ++index) {
        require(visited[index], "batch inverse did not store index " + std::to_string(index));
        if (values[index] == 0) {
            require(actual[index].bytes == FieldValue{}.bytes, "zero inverse slot is not canonical field zero");
            continue;
        }
        const FieldValue expected = legacy_field_encode(legacy_inverse(values[index], context).get());
        require(same_field(actual[index], expected), "batch inverse differs from scalar BN_mod_inverse at index " + std::to_string(index));
        BigNum input = legacy_field_from_i64(values[index], context);
        BigNum inverse = field_from_value(actual[index]);
        BigNum product;
        require(BN_mod_mul(product.get(), input.get(), inverse.get(), scalar_modulus(), context) && BN_is_one(product.get()),
                "batch inverse product is not one at index " + std::to_string(index));
    }
    std::vector<std::int64_t> zeros(kFieldInverseBatchSize * 2U + 1U, 0);
    std::size_t zero_stores = 0U;
    batch_field_inverses(zeros, context, "all-zero numerical test", [&](std::size_t, const BIGNUM *inverse) {
        require(inverse == nullptr, "all-zero batch emitted a nonzero inverse");
        ++zero_stores;
    });
    require(zero_stores == zeros.size(), "all-zero batch omitted an output slot");
}

void test_round_div_witness(BN_CTX *context) {
    constexpr std::uint64_t rows = 9001U;
    Matrix numerator;
    numerator.rows = rows;
    numerator.cols = 1U;
    numerator.values.resize(static_cast<std::size_t>(rows));
    Matrix expected;
    expected.rows = rows;
    expected.cols = 1U;
    expected.values.resize(static_cast<std::size_t>(rows));
    std::vector<std::int64_t> denominator(static_cast<std::size_t>(rows));
    const std::array<std::int64_t, 11> samples = {-37, -9, -1, 0, 1, 9, 37, std::numeric_limits<std::int64_t>::min() + 100000,
                                                  std::numeric_limits<std::int64_t>::max() - 100000, -1000001, 1000001};
    const std::array<std::int64_t, 4> divisors = {3, 5, 7, 3};
    for (std::uint64_t row = 0; row < rows; ++row) {
        const auto value = samples[static_cast<std::size_t>(row % samples.size())];
        const auto divisor = divisors[static_cast<std::size_t>(row % divisors.size())];
        numerator.at(row, 0U) = value;
        denominator[static_cast<std::size_t>(row)] = divisor;
        expected.at(row, 0U) = legacy_round_div(value, divisor);
    }
    const SignedRoundDivWitness actual = make_round_div_witness(numerator, denominator, 1U, 1U, expected, "numerical_test");
    const SignedRoundDivWitness legacy = legacy_round_div_witness(numerator, denominator, 1U, 1U, expected, context);
    require(serialize_witness(actual) == serialize_witness(legacy), "complete SignedRoundDiv witness bytes differ from scalar legacy oracle");
    for (const std::size_t index : {4095U, 4096U, 8191U, 8192U})
        require(same_field(actual.a_inverse[index], legacy.a_inverse[index]) &&
                    same_field(actual.divisor_inverse[index], legacy.divisor_inverse[index]),
                "SignedRoundDiv inverse bytes differ at batch boundary index " + std::to_string(index));
}

IntegerBoundFamily scale_test_family(const std::string &id,
                                    std::uint64_t layer,
                                    std::uint64_t head, bool output) {
    IntegerBoundFamily family;
    family.family_id = id;
    family.layer = layer;
    family.head = head;
    family.output = output;
    family.max_dividend_abs = 1000000000000000000LL;
    family.max_aux_abs = 1000000000000000000LL;
    family.max_divisor = 32;
    return family;
}

void check_signed_numerator_scale_case(
    const signed_round_bounds::Widths &widths, const std::string &label,
    const std::vector<std::int64_t> &denominators, BN_CTX *context) {
    constexpr std::uint64_t rows = 4U;
    constexpr std::uint64_t cols = 2U;
    Matrix numerator;
    numerator.rows = rows;
    numerator.cols = cols;
    numerator.values = {3, -5, (std::int64_t(1) << 54U) + 7,
                        -((std::int64_t(1) << 54U) + 11), -17, 19, 0, 0};
    Matrix expected;
    expected.rows = rows;
    expected.cols = cols;
    expected.values.resize(numerator.values.size());
    require(denominators.size() == rows && widths.numerator_scale > 0 &&
                widths.denominator_scale > 0,
            label + ": invalid test scales or denominator shape");
    for (std::uint64_t row = 0U; row < rows; ++row) {
        const auto divisor = checked_i128(
            static_cast<__int128>(denominators.at(static_cast<std::size_t>(row))) *
                widths.denominator_scale,
            label + ".divisor");
        for (std::uint64_t column = 0U; column < cols; ++column) {
            const auto index = static_cast<std::size_t>(row * cols + column);
            const auto scaled_numerator = checked_i128(
                static_cast<__int128>(numerator.values.at(index)) *
                    widths.numerator_scale,
                label + ".scaled_numerator");
            expected.values[index] = round_div(scaled_numerator, divisor,
                                                label + ".expected");
        }
    }
    const auto witness = make_round_div_witness(
        numerator, denominators,
        static_cast<std::uint64_t>(widths.numerator_scale),
        static_cast<std::uint64_t>(widths.denominator_scale), expected,
        label);
    const auto legacy = legacy_round_div_witness(
        numerator, denominators,
        static_cast<std::uint64_t>(widths.numerator_scale),
        static_cast<std::uint64_t>(widths.denominator_scale), expected,
        context);
    require(serialize_witness(witness) == serialize_witness(legacy),
            label + ": scaled round witness changed");

    const BigNum scale = field_from_i64(widths.numerator_scale, context);
    bool old_positive_mismatch = false;
    bool old_negative_mismatch = false;
    for (std::size_t index = 0U; index < numerator.values.size(); ++index) {
        const std::int64_t raw = numerator.values.at(index);
        const std::int64_t writer_raw = numerator.values.at(index);
        const auto scaled_raw = checked_i128(
            static_cast<__int128>(raw) * widths.numerator_scale,
            label + ".old_writer_scaled_value");
        const BigNum canonical = field_from_i64(raw, context);
        const BigNum raw_view = field_from_i64(writer_raw, context);
        const BigNum old_scaled_view = field_from_i64(scaled_raw, context);
        BigNum new_identity_residual;
        BigNum old_identity_residual;
        require(BN_mod_sub(new_identity_residual.get(), raw_view.get(),
                           canonical.get(), scalar_modulus(), context) &&
                    BN_mod_sub(old_identity_residual.get(),
                               old_scaled_view.get(), canonical.get(),
                               scalar_modulus(), context),
                label + ": independent FAC residual arithmetic failed");
        require(BN_is_zero(new_identity_residual.get()),
                label + ": raw numerator view failed identity FAC");
        if (widths.numerator_scale == 1 || raw == 0) {
            require(BN_is_zero(old_identity_residual.get()),
                    label + ": old writer unexpectedly changed an S=1/zero value");
        } else {
            require(!BN_is_zero(old_identity_residual.get()),
                    label + ": old scaled writer failed to reproduce identity FAC mismatch");
            old_positive_mismatch = old_positive_mismatch || raw > 0;
            old_negative_mismatch = old_negative_mismatch || raw < 0;
        }

        BigNum scaled_input;
        BigNum absolute;
        BigNum signed_absolute;
        BigNum signed_residual;
        require(BN_mod_mul(scaled_input.get(), canonical.get(), scale.get(),
                           scalar_modulus(), context),
                label + ": signed numerator scale multiplication failed");
        BigNum witness_absolute = field_from_i64(witness.a.at(index), context);
        BigNum witness_sign = field_from_i64(witness.sigma.at(index), context);
        require(BN_mod_mul(absolute.get(), witness_absolute.get(),
                           witness_sign.get(), scalar_modulus(), context) &&
                    BN_mod_lshift1(signed_absolute.get(), absolute.get(),
                                   scalar_modulus(), context) &&
                    BN_mod_sub(signed_residual.get(), scaled_input.get(),
                               witness_absolute.get(), scalar_modulus(), context) &&
                    BN_mod_add(signed_residual.get(), signed_residual.get(),
                               signed_absolute.get(), scalar_modulus(), context),
                label + ": signed-numerator constraint arithmetic failed");
        require(BN_is_zero(signed_residual.get()),
                label + ": S*numerator - a + 2*sigma*a is nonzero");
        if (index >= 6U)
            require(raw == 0 && witness.a.at(index) == 0 &&
                        witness.sigma.at(index) == 0U,
                    label + ": padded node did not remain zero");
    }
    if (widths.numerator_scale > 1)
        require(old_positive_mismatch && old_negative_mismatch,
                label + ": old writer mismatch was not observed for both signs");
}

void test_signed_numerator_scale_identity(BN_CTX *context) {
    Config config;
    config.architecture.hidden_dims = {2U};
    config.architecture.output_heads = 2U;
    config.quantization.hidden_aggregate_pre = {16U};
    config.quantization.hidden_h_prime = {{64U}};
    config.quantization.output_y_prime = {8U, 16U};
    config.quantization.output_pre = 16U;
    config.quantization.output_head = 8U;

    IntegerBoundLedger ledger;
    ledger.authenticated = true;
    ledger.families = {
        scale_test_family("hidden/layer0/head0", 0U, 0U, false),
        scale_test_family("output/head0", 1U, 0U, true),
        scale_test_family("output/head1", 1U, 1U, true),
    };

    const auto hidden = signed_round_bounds::derive(
        config, ledger, "hidden/layer0/head0/signed_round_div");
    const auto output = signed_round_bounds::derive(
        config, ledger, "output/head0/signed_round_div");
    const auto alignment = signed_round_bounds::derive(
        config, ledger, "output/head1/output_scale_alignment");
    const auto average = signed_round_bounds::derive(
        config, ledger, "final_average");
    require(hidden.numerator_scale == 16 && output.numerator_scale == 16 &&
                alignment.numerator_scale == 8 && average.numerator_scale == 1,
            "signed-round authoritative numerator scales changed");

    check_signed_numerator_scale_case(hidden, "hidden-S16", {2, 3, 5, 1},
                                      context);
    check_signed_numerator_scale_case(output, "output-S16", {2, 3, 5, 1},
                                      context);
    check_signed_numerator_scale_case(alignment, "alignment-S8",
                                      {16, 16, 16, 16}, context);
    check_signed_numerator_scale_case(average, "final-average-S1",
                                      {2, 2, 2, 2}, context);
}

template <typename Head>
void check_attention_alpha(Head &head, const Graph &graph, const std::vector<std::int64_t> &denominators, BN_CTX *context) {
    const FieldInverseCache cache = fill_field_attention(head, graph, 0U, graph.edges, denominators);
    require(cache.by_node.size() == graph.nodes, "attention inverse cache has wrong node count");
    for (std::uint64_t edge = 0; edge < graph.edges; ++edge) {
        const auto node = graph.dst[static_cast<std::size_t>(edge)];
        const FieldValue expected_inverse = legacy_field_encode(legacy_inverse(denominators[node], context).get());
        BigNum weight = legacy_field_from_i64(head.exp_weight[static_cast<std::size_t>(edge)], context);
        BigNum inverse = legacy_inverse(denominators[node], context);
        BigNum expected_alpha_bn;
        require(BN_mod_mul(expected_alpha_bn.get(), weight.get(), inverse.get(), scalar_modulus(), context), "legacy alpha multiplication failed");
        require(same_field(head.inverse_field[static_cast<std::size_t>(edge)], expected_inverse), "attention inverse field bytes changed");
        require(same_field(head.alpha_field[static_cast<std::size_t>(edge)], legacy_field_encode(expected_alpha_bn.get())), "attention alpha field bytes changed");
    }
}

void test_attention_and_aggregate(BN_CTX *context) {
    Graph graph;
    graph.nodes = 3U;
    graph.edges = 7U;
    graph.src = {0U, 1U, 2U, 0U, 2U, 1U, 2U};
    graph.dst = {0U, 0U, 0U, 1U, 1U, 2U, 2U};
    graph.node_edge_ptr = {0U, 3U, 5U, 7U};
    const std::vector<std::int64_t> denominators = {11, 13, 17};
    OutputHeadTrace output_head;
    output_head.exp_weight = {1, 3, 7, 5, 9, 11, 13};
    check_attention_alpha(output_head, graph, denominators, context);
    HiddenHeadTrace hidden_head;
    hidden_head.exp_weight = output_head.exp_weight;
    check_attention_alpha(hidden_head, graph, denominators, context);

    Matrix source;
    source.rows = graph.nodes;
    source.cols = 4U;
    source.values = {-19, 0, 7, 31, 5, -11, 0, 17, 23, -29, 37, 0};
    Matrix aggregate_num;
    aggregate_num.rows = graph.nodes;
    aggregate_num.cols = source.cols;
    aggregate_num.values.assign(static_cast<std::size_t>(graph.nodes * source.cols), 0);
    for (std::uint64_t edge = 0; edge < graph.edges; ++edge) {
        const auto node = graph.dst[static_cast<std::size_t>(edge)];
        const auto weight = output_head.exp_weight[static_cast<std::size_t>(edge)];
        for (std::uint64_t dim = 0; dim < source.cols; ++dim) {
            const __int128 value = static_cast<__int128>(aggregate_num.at(node, dim)) +
                                   static_cast<__int128>(source.at(graph.src[static_cast<std::size_t>(edge)], dim)) * weight;
            require(value >= std::numeric_limits<std::int64_t>::min() && value <= std::numeric_limits<std::int64_t>::max(), "aggregate_num fixture overflow");
            aggregate_num.at(node, dim) = static_cast<std::int64_t>(value);
        }
    }
    FieldMatrix products, aggregates, residuals;
    products.rows = graph.edges;
    products.cols = source.cols;
    products.values.assign(static_cast<std::size_t>(graph.edges * source.cols), FieldValue{});
    aggregates.rows = graph.nodes;
    aggregates.cols = source.cols;
    aggregates.values.assign(static_cast<std::size_t>(graph.nodes * source.cols), FieldValue{});
    residuals.rows = graph.nodes;
    residuals.cols = source.cols;
    residuals.values.assign(static_cast<std::size_t>(graph.nodes * source.cols), FieldValue{});
    aggregate_field_products(graph, source, denominators, output_head.alpha_field, aggregate_num, products, aggregates, residuals, context, "numerical_test");

    std::vector<BigNum> old_inverses;
    old_inverses.reserve(denominators.size());
    for (const auto denominator : denominators) old_inverses.push_back(legacy_inverse(denominator, context));
    std::vector<BigNum> old_aggregates;
    old_aggregates.reserve(static_cast<std::size_t>(graph.nodes * source.cols));
    for (std::uint64_t index = 0; index < graph.nodes * source.cols; ++index) {
        old_aggregates.emplace_back();
        BN_zero(old_aggregates.back().get());
    }
    std::vector<BigNum> old_integer_sums;
    old_integer_sums.reserve(static_cast<std::size_t>(graph.nodes * source.cols));
    for (std::uint64_t index = 0; index < graph.nodes * source.cols; ++index) {
        old_integer_sums.emplace_back();
        BN_zero(old_integer_sums.back().get());
    }
    std::vector<FieldValue> old_products(static_cast<std::size_t>(graph.edges * source.cols));
    for (std::uint64_t edge = 0; edge < graph.edges; ++edge) {
        const auto node = graph.dst[static_cast<std::size_t>(edge)];
        BigNum weight = legacy_field_from_i64(output_head.exp_weight[static_cast<std::size_t>(edge)], context);
        BigNum old_alpha;
        require(BN_mod_mul(old_alpha.get(), weight.get(), old_inverses[node].get(), scalar_modulus(), context), "legacy repeated weight*inverse failed");
        for (std::uint64_t dim = 0; dim < source.cols; ++dim) {
            BigNum scalar = legacy_field_from_i64(source.at(graph.src[static_cast<std::size_t>(edge)], dim), context);
            BigNum product;
            require(BN_mod_mul(product.get(), old_alpha.get(), scalar.get(), scalar_modulus(), context), "legacy aggregate product failed");
            const std::size_t edge_index = static_cast<std::size_t>(edge * source.cols + dim);
            old_products[edge_index] = legacy_field_encode(product.get());
            BigNum &sum = old_aggregates[static_cast<std::size_t>(node * source.cols + dim)];
            require(BN_mod_add(sum.get(), sum.get(), product.get(), scalar_modulus(), context), "legacy aggregate sum failed");
            BigNum integer_term;
            require(BN_mod_mul(integer_term.get(), weight.get(), scalar.get(), scalar_modulus(), context),
                    "legacy integer aggregate product failed");
            BigNum &integer_sum = old_integer_sums[static_cast<std::size_t>(node * source.cols + dim)];
            require(BN_mod_add(integer_sum.get(), integer_sum.get(), integer_term.get(), scalar_modulus(), context),
                    "legacy integer aggregate sum failed");
        }
    }
    for (std::uint64_t node = 0; node < graph.nodes; ++node) {
        for (std::uint64_t dim = 0; dim < source.cols; ++dim) {
            const std::size_t index = static_cast<std::size_t>(node * source.cols + dim);
            const FieldValue old_aggregate = legacy_field_encode(old_aggregates[index].get());
            BigNum denominator = legacy_field_from_i64(denominators[node], context);
            BigNum rhs, old_residual;
            require(BN_mod_mul(rhs.get(), denominator.get(), old_aggregates[index].get(), scalar_modulus(), context) &&
                        BN_mod_sub(old_residual.get(), rhs.get(), old_integer_sums[index].get(), scalar_modulus(), context),
                    "legacy aggregate residual failed");
            BigNum encoded_aggregate_num = legacy_field_from_i64(aggregate_num.at(node, dim), context);
            require(BN_cmp(encoded_aggregate_num.get(), old_integer_sums[index].get()) == 0,
                    "checked integer aggregate differs from independent legacy BN edge sum");
            require(same_field(aggregates.values[index], old_aggregate), "aggregate field bytes differ from old field sum");
            require(same_field(residuals.values[index], legacy_field_encode(old_residual.get())), "aggregate residual bytes differ from old field sum");
        }
    }
    for (std::size_t index = 0; index < products.values.size(); ++index)
        require(same_field(products.values[index], old_products[index]), "per-edge aggregate field bytes differ from old weight*inverse path");

    Graph boundary_graph;
    boundary_graph.nodes = 3U;
    boundary_graph.edges = 4U;
    boundary_graph.src = {0U, 1U, 0U, 1U};
    boundary_graph.dst = {0U, 1U, 2U, 2U};
    boundary_graph.node_edge_ptr = {0U, 1U, 2U, 4U};
    Matrix boundary_source;
    boundary_source.rows = boundary_graph.nodes;
    boundary_source.cols = 1U;
    boundary_source.values = {std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max(), 0};
    const std::vector<std::int64_t> boundary_denominators(boundary_graph.nodes, 1);
    OutputHeadTrace boundary_head;
    boundary_head.exp_weight.assign(static_cast<std::size_t>(boundary_graph.edges), 1);
    check_attention_alpha(boundary_head, boundary_graph, boundary_denominators, context);
    Matrix boundary_integer_sum;
    boundary_integer_sum.rows = boundary_graph.nodes;
    boundary_integer_sum.cols = 1U;
    boundary_integer_sum.values = {std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max(), -1};
    FieldMatrix boundary_products, boundary_aggregates, boundary_residuals;
    boundary_products.rows = boundary_graph.edges;
    boundary_products.cols = 1U;
    boundary_products.values.assign(static_cast<std::size_t>(boundary_graph.edges), FieldValue{});
    boundary_aggregates.rows = boundary_graph.nodes;
    boundary_aggregates.cols = 1U;
    boundary_aggregates.values.assign(static_cast<std::size_t>(boundary_graph.nodes), FieldValue{});
    boundary_residuals.rows = boundary_graph.nodes;
    boundary_residuals.cols = 1U;
    boundary_residuals.values.assign(static_cast<std::size_t>(boundary_graph.nodes), FieldValue{});
    aggregate_field_products(boundary_graph, boundary_source, boundary_denominators, boundary_head.alpha_field,
                             boundary_integer_sum, boundary_products, boundary_aggregates, boundary_residuals,
                             context, "boundary_numerical_test");

    std::vector<BigNum> boundary_old_sums;
    boundary_old_sums.reserve(static_cast<std::size_t>(boundary_graph.nodes));
    std::vector<BigNum> boundary_old_integer_sums;
    boundary_old_integer_sums.reserve(static_cast<std::size_t>(boundary_graph.nodes));
    std::vector<BigNum> boundary_old_inverses;
    boundary_old_inverses.reserve(boundary_denominators.size());
    for (std::uint64_t node = 0; node < boundary_graph.nodes; ++node) {
        boundary_old_sums.emplace_back();
        BN_zero(boundary_old_sums.back().get());
        boundary_old_integer_sums.emplace_back();
        BN_zero(boundary_old_integer_sums.back().get());
        boundary_old_inverses.push_back(legacy_inverse(boundary_denominators[static_cast<std::size_t>(node)], context));
    }
    for (std::uint64_t edge = 0; edge < boundary_graph.edges; ++edge) {
        const std::uint64_t source_node = boundary_graph.src[static_cast<std::size_t>(edge)];
        const std::uint64_t destination = boundary_graph.dst[static_cast<std::size_t>(edge)];
        BigNum weight = legacy_field_from_i64(boundary_head.exp_weight[static_cast<std::size_t>(edge)], context);
        BigNum old_alpha;
        require(BN_mod_mul(old_alpha.get(), weight.get(), boundary_old_inverses[static_cast<std::size_t>(destination)].get(),
                           scalar_modulus(), context),
                "legacy boundary alpha multiplication failed");
        BigNum old_source = legacy_field_from_i64(boundary_source.at(source_node, 0U), context);
        BigNum old_product;
        require(BN_mod_mul(old_product.get(), old_alpha.get(), old_source.get(), scalar_modulus(), context),
                "legacy boundary weighted aggregate product failed");
        BigNum &old_sum = boundary_old_sums[static_cast<std::size_t>(destination)];
        require(BN_mod_add(old_sum.get(), old_sum.get(), old_product.get(), scalar_modulus(), context),
                "legacy boundary aggregate sum failed");
        require(same_field(boundary_products.values[static_cast<std::size_t>(edge)], legacy_field_encode(old_product.get())),
                "boundary per-edge aggregate field bytes differ from old BN path");
        BigNum old_integer_product;
        require(BN_mod_mul(old_integer_product.get(), weight.get(), old_source.get(), scalar_modulus(), context),
                "legacy boundary integer product failed");
        BigNum &old_integer_sum = boundary_old_integer_sums[static_cast<std::size_t>(destination)];
        require(BN_mod_add(old_integer_sum.get(), old_integer_sum.get(), old_integer_product.get(), scalar_modulus(), context),
                "legacy boundary integer aggregate sum failed");
    }
    for (std::uint64_t node = 0; node < boundary_graph.nodes; ++node) {
        const std::size_t index = static_cast<std::size_t>(node);
        require(same_field(boundary_aggregates.values[index], legacy_field_encode(boundary_old_sums[index].get())),
                "boundary aggregate field bytes differ from old BN sum");
        BigNum old_integer = legacy_field_from_i64(boundary_integer_sum.values[index], context);
        require(BN_cmp(old_integer.get(), boundary_old_integer_sums[index].get()) == 0,
                "boundary integer aggregate differs from independent legacy BN edge sum");
        BigNum denominator = legacy_field_from_i64(boundary_denominators[index], context);
        BigNum rhs;
        BigNum old_residual;
        require(BN_mod_mul(rhs.get(), denominator.get(), boundary_old_sums[index].get(), scalar_modulus(), context) &&
                    BN_mod_sub(old_residual.get(), rhs.get(), boundary_old_integer_sums[index].get(), scalar_modulus(), context),
                "legacy boundary aggregate residual failed");
        require(same_field(boundary_residuals.values[index], legacy_field_encode(old_residual.get())),
                "boundary aggregate residual bytes differ from old BN path");
    }
}

void test_concat_layout() {
    HiddenHeadTrace first, second;
    first.aggregate.rows = second.aggregate.rows = 2U;
    first.aggregate.cols = second.aggregate.cols = 2U;
    first.aggregate.values = {1, 2, 3, 4};
    second.aggregate.values = {5, 6, 7, 8};
    const Matrix concatenated = concat_head_aggregates({first, second}, 2U, 2U);
    require(concatenated.rows == 2U && concatenated.cols == 4U && concatenated.values == std::vector<std::int64_t>({1, 2, 5, 6, 3, 4, 7, 8}),
            "head concat row/head layout changed");
}

} // namespace

int main() {
    try {
        BN_CTX *context = BN_CTX_new();
        if (!context) throw std::runtime_error("BN_CTX_new failed");
        try {
            test_signed_field_encoding(context);
            test_scalar_round_div();
            test_batch_inverse(context);
            test_round_div_witness(context);
            test_signed_numerator_scale_identity(context);
            test_attention_and_aggregate(context);
            test_concat_layout();
        } catch (...) {
            BN_CTX_free(context);
            throw;
        }
        BN_CTX_free(context);
        std::cout << "gat_witness_numerics=PASS field=batch-inverse round-div=byte-equal signed-numerator=identity-repaired-S1-S16-positive-negative-padding attention=byte-equal aggregate=byte-equal concat=layout-equal\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
