#pragma once

#include "gat_adapter.hpp"
#include "checkpoint_manifest_json.hpp"

#include <array>
#include <charconv>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include <openssl/evp.h>

namespace zkdl_gat::integer_bound_ledger {

namespace json = checkpoint_manifest_json;

constexpr std::size_t kMaxFamilies = 4096U;

inline std::string span_text(const std::string &text, json::Span span) {
    if (span.begin > span.end || span.end > text.size()) throw std::runtime_error("invalid integer-bound ledger JSON span");
    return text.substr(span.begin, span.end - span.begin);
}

inline std::map<std::string, json::Span> object(const std::string &text, json::Span span, const std::string &what) {
    return json::object_members(text, span, what);
}

inline json::Span document(const std::string &text) {
    const std::size_t end = json::scan_value(text, 0U);
    std::size_t trailing = end;
    json::skip_space(text, trailing);
    if (trailing != text.size()) throw std::runtime_error("integer-bound ledger has trailing JSON bytes");
    return {0U, end};
}

inline void exact_keys(const std::map<std::string, json::Span> &fields, const std::set<std::string> &expected, const std::string &what) {
    std::set<std::string> actual;
    for (const auto &field : fields) actual.insert(field.first);
    if (actual != expected) throw std::runtime_error(what + " fields do not match the canonical generator schema");
}

inline json::Span required(const std::map<std::string, json::Span> &fields, const std::string &key, const std::string &what) {
    const auto found = fields.find(key);
    if (found == fields.end()) throw std::runtime_error(what + " is missing " + key);
    return found->second;
}

inline std::string string_value(const std::string &text, json::Span span, const std::string &what) {
    std::size_t position = span.begin;
    const std::string value = json::read_string(text, position);
    if (position != span.end) throw std::runtime_error(what + " must be a JSON string");
    return value;
}

inline std::uint64_t unsigned_value(const std::string &text, json::Span span, const std::string &what) {
    return json::parse_u64(span_text(text, span), what);
}

inline std::int64_t signed_value(const std::string &text, json::Span span, const std::string &what) {
    const std::string token = span_text(text, span);
    if (token.empty()) throw std::runtime_error(what + " is not a JSON integer");
    std::size_t digits = token.front() == '-' ? 1U : 0U;
    if (digits == token.size() || (token[digits] == '0' && token.size() - digits != 1U))
        throw std::runtime_error(what + " is not a canonical signed JSON integer");
    for (std::size_t index = digits; index < token.size(); ++index) {
        if (token[index] < '0' || token[index] > '9') throw std::runtime_error(what + " is not a JSON integer");
    }
    if (token == "-0") throw std::runtime_error(what + " uses noncanonical negative zero");
    std::int64_t value = 0;
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value, 10);
    if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size()) throw std::runtime_error(what + " exceeds signed 64-bit range");
    return value;
}

inline std::vector<json::Span> array_values(const std::string &text, json::Span array, const std::string &what) {
    std::size_t position = array.begin;
    json::skip_space(text, position);
    if (position >= text.size() || text[position++] != '[') throw std::runtime_error(what + " is not a JSON array");
    std::vector<json::Span> values;
    json::skip_space(text, position);
    if (position < text.size() && text[position] == ']') ++position;
    else {
        for (;;) {
            json::skip_space(text, position);
            const std::size_t begin = position;
            const std::size_t end = json::scan_value(text, position);
            values.push_back({begin, end});
            position = end;
            json::skip_space(text, position);
            if (position >= text.size()) throw std::runtime_error("unterminated " + what);
            if (text[position] == ']') { ++position; break; }
            if (text[position++] != ',') throw std::runtime_error("invalid separator in " + what);
        }
    }
    if (position != array.end) throw std::runtime_error(what + " has trailing JSON bytes");
    return values;
}

inline std::string required_string(const std::string &text, const std::map<std::string, json::Span> &fields, const std::string &key,
                                   const std::string &what) {
    return string_value(text, required(fields, key, what), what + "." + key);
}

inline std::int64_t required_i64(const std::string &text, const std::map<std::string, json::Span> &fields, const std::string &key,
                                 const std::string &what) {
    return signed_value(text, required(fields, key, what), what + "." + key);
}

inline std::uint64_t required_u64(const std::string &text, const std::map<std::string, json::Span> &fields, const std::string &key,
                                  const std::string &what) {
    return unsigned_value(text, required(fields, key, what), what + "." + key);
}

inline IntegerBoundFamily parse_family(const std::string &text, json::Span span, std::size_t index) {
    const std::string what = "integer-bound ledger family " + std::to_string(index);
    const auto fields = object(text, span, what);
    exact_keys(fields, {"family_id", "layer_index", "head_index", "output", "max_input_abs", "max_weight_abs", "max_attention_abs", "max_bias_abs",
                        "max_score_abs", "max_table_output_abs", "max_delta", "max_exp", "max_sum", "max_aggregate_num", "max_dividend_abs",
                        "max_aux_abs", "max_divisor", "max_field_product_abs"}, what);
    IntegerBoundFamily family;
    family.family_id = required_string(text, fields, "family_id", what);
    family.layer = required_u64(text, fields, "layer_index", what);
    family.head = required_u64(text, fields, "head_index", what);
    const auto output = required_u64(text, fields, "output", what);
    if (output > 1U) throw std::runtime_error(what + ".output must be integer 0 or 1");
    family.output = output == 1U;
    const std::string expected_id = family.output ? "output/head" + std::to_string(family.head)
                                                  : "hidden/layer" + std::to_string(family.layer) + "/head" + std::to_string(family.head);
    if (family.family_id != expected_id) throw std::runtime_error(what + ".family_id disagrees with its canonical coordinates");
    family.max_input_abs = required_i64(text, fields, "max_input_abs", what);
    family.max_weight_abs = required_i64(text, fields, "max_weight_abs", what);
    family.max_attention_abs = required_i64(text, fields, "max_attention_abs", what);
    family.max_bias_abs = required_i64(text, fields, "max_bias_abs", what);
    family.max_score_abs = required_i64(text, fields, "max_score_abs", what);
    family.max_table_output_abs = required_i64(text, fields, "max_table_output_abs", what);
    family.max_delta = required_i64(text, fields, "max_delta", what);
    family.max_exp = required_i64(text, fields, "max_exp", what);
    family.max_sum = required_i64(text, fields, "max_sum", what);
    family.max_aggregate_num = required_i64(text, fields, "max_aggregate_num", what);
    family.max_dividend_abs = required_i64(text, fields, "max_dividend_abs", what);
    family.max_aux_abs = required_i64(text, fields, "max_aux_abs", what);
    family.max_divisor = required_i64(text, fields, "max_divisor", what);
    family.max_field_product_abs = required_i64(text, fields, "max_field_product_abs", what);
    return family;
}

inline IntegerBoundLedger parse(const std::string &text) {
    const auto root = document(text);
    const auto fields = object(text, root, "integer-bound ledger");
    exact_keys(fields, {"schema", "ledger_id", "degree_bound_id", "quant_cfg_id", "static_table_id", "field_modulus", "scale_chain_digest",
                        "static_table_descriptor_digest", "digest", "max_score_abs", "max_table_output_abs", "max_delta", "max_exp", "max_sum",
                        "max_aggregate_num", "max_dividend_abs", "max_aux_abs", "max_divisor", "families"}, "integer-bound ledger");
    IntegerBoundLedger ledger;
    ledger.schema = required_string(text, fields, "schema", "integer-bound ledger");
    if (ledger.schema != "zkDL_gat_integer_bound_ledger_v1") throw std::runtime_error("integer-bound ledger schema mismatch");
    ledger.ledger_id = required_string(text, fields, "ledger_id", "integer-bound ledger");
    ledger.degree_bound_id = required_string(text, fields, "degree_bound_id", "integer-bound ledger");
    ledger.quant_cfg_id = required_string(text, fields, "quant_cfg_id", "integer-bound ledger");
    ledger.static_table_id = required_string(text, fields, "static_table_id", "integer-bound ledger");
    ledger.field_modulus = required_string(text, fields, "field_modulus", "integer-bound ledger");
    ledger.scale_chain_digest = required_string(text, fields, "scale_chain_digest", "integer-bound ledger");
    ledger.static_table_descriptor_digest = required_string(text, fields, "static_table_descriptor_digest", "integer-bound ledger");
    ledger.digest = required_string(text, fields, "digest", "integer-bound ledger");
    ledger.max_score_abs = required_i64(text, fields, "max_score_abs", "integer-bound ledger");
    ledger.max_table_output_abs = required_i64(text, fields, "max_table_output_abs", "integer-bound ledger");
    ledger.max_delta = required_i64(text, fields, "max_delta", "integer-bound ledger");
    ledger.max_exp = required_i64(text, fields, "max_exp", "integer-bound ledger");
    ledger.max_sum = required_i64(text, fields, "max_sum", "integer-bound ledger");
    ledger.max_aggregate_num = required_i64(text, fields, "max_aggregate_num", "integer-bound ledger");
    ledger.max_dividend_abs = required_i64(text, fields, "max_dividend_abs", "integer-bound ledger");
    ledger.max_aux_abs = required_i64(text, fields, "max_aux_abs", "integer-bound ledger");
    ledger.max_divisor = required_i64(text, fields, "max_divisor", "integer-bound ledger");
    const auto families = array_values(text, required(fields, "families", "integer-bound ledger"), "integer-bound ledger families");
    if (families.empty() || families.size() > kMaxFamilies) throw std::runtime_error("integer-bound ledger family count is outside native bounds");
    ledger.families.reserve(families.size());
    for (std::size_t index = 0; index < families.size(); ++index) ledger.families.push_back(parse_family(text, families[index], index));
    return ledger;
}

class CanonicalBytes {
  public:
    void u64(std::uint64_t value) {
        for (unsigned shift = 0; shift < 64U; shift += 8U) bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
    }
    void string(const std::string &value) {
        u64(static_cast<std::uint64_t>(value.size()));
        bytes_.insert(bytes_.end(), value.begin(), value.end());
    }
    std::string finish() const {
        std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
        unsigned digest_size = 0U;
        if (EVP_Digest(bytes_.data(), bytes_.size(), digest.data(), &digest_size, EVP_sha256(), nullptr) != 1 || digest_size != 32U)
            throw std::runtime_error("native SHA-256 failed for integer-bound ledger");
        static constexpr char hex[] = "0123456789abcdef";
        std::string result;
        result.resize(64U);
        for (std::size_t index = 0; index < 32U; ++index) {
            result[index * 2U] = hex[digest[index] >> 4U];
            result[index * 2U + 1U] = hex[digest[index] & 0x0fU];
        }
        return result;
    }

  private:
    std::vector<std::uint8_t> bytes_;
};

inline std::string digest(const IntegerBoundLedger &ledger) {
    CanonicalBytes bytes;
    bytes.string("zkDL_gat_integer_bound_ledger_v1");
    for (const auto *value : {&ledger.ledger_id, &ledger.quant_cfg_id, &ledger.static_table_id, &ledger.field_modulus, &ledger.scale_chain_digest,
                              &ledger.static_table_descriptor_digest}) bytes.string(*value);
    for (const auto value : {ledger.max_score_abs, ledger.max_table_output_abs, ledger.max_delta, ledger.max_exp, ledger.max_sum, ledger.max_aggregate_num,
                             ledger.max_dividend_abs, ledger.max_aux_abs, ledger.max_divisor}) bytes.u64(static_cast<std::uint64_t>(value));
    bytes.u64(static_cast<std::uint64_t>(ledger.families.size()));
    for (const auto &family : ledger.families) {
        bytes.string(family.family_id);
        bytes.u64(family.layer);
        bytes.u64(family.head);
        bytes.u64(family.output ? 1U : 0U);
        for (const auto value : {family.max_input_abs, family.max_weight_abs, family.max_attention_abs, family.max_bias_abs, family.max_score_abs,
                                 family.max_table_output_abs, family.max_delta, family.max_exp, family.max_sum, family.max_aggregate_num,
                                 family.max_dividend_abs, family.max_aux_abs, family.max_divisor, family.max_field_product_abs})
            bytes.u64(static_cast<std::uint64_t>(value));
    }
    return bytes.finish();
}

} // namespace zkdl_gat::integer_bound_ledger
