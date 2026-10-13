
#include "gat_native_bridge.hpp"
#include "gat_native_resources.hpp"
#include "output_stream.hpp"
#include "checkpoint_manifest_json.hpp"
#include "integer_bound_ledger.hpp"
#include "lookup_table_index_bits.hpp"
#include "signed_round_bounds.hpp"
#include "static_table_bounds.hpp"
#include "gat_witness_numerics.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <openssl/bn.h>

namespace zkdl_gat {
namespace {

using u64 = std::uint64_t;
using i64 = std::int64_t;
using witness_numerics::BigNum;
using witness_numerics::FieldInverseCache;
using witness_numerics::aggregate_field_products;
using witness_numerics::concat_head_aggregates;
using witness_numerics::field_encode;
using witness_numerics::field_from_i64;
using witness_numerics::field_inverse;
using witness_numerics::fill_field_attention;
using witness_numerics::make_round_div_witness;
using witness_numerics::scalar_modulus;
using witness_numerics::scalar_modulus_decimal;

constexpr u64 kMaxStaticTableRows = 1ULL << 27;
constexpr u64 kMaxLedgerFamilies = 4096;
constexpr u64 kMaxNativeDomain = 1ULL << 30;

[[noreturn]] void fail(const std::string &message) { throw std::runtime_error("zkdl-gat: " + message); }

void finish_output(std::ofstream &output, const std::filesystem::path &path, const std::string &label) {
    try {
        finish_output_stream(output, path, label);
    } catch (const std::exception &exception) {
        fail(exception.what());
    }
}

std::uint32_t checked_native_domain_u32(u64 value, const std::string &label, bool allow_zero = false) {
    if ((!allow_zero && value == 0U) || value > kMaxNativeDomain || value > std::numeric_limits<std::uint32_t>::max())
        fail("BLOCKED: " + label + " exceeds the native u32 domain bound");
    return static_cast<std::uint32_t>(value);
}

bool checked_multiply_u64(u64 left, u64 right, u64 &product) {
    if (right != 0U && left > std::numeric_limits<u64>::max() / right) return false;
    product = left * right;
    return true;
}

u64 checked_shape_product(const std::vector<u64> &shape, const std::string &label) {
    if (shape.empty()) fail("empty relation shape: " + label);
    u64 product = 1U;
    for (const u64 dimension : shape) {
        if (dimension == 0U || product > kMaxNativeDomain / dimension)
            fail("relation shape product exceeds native coordinate bound: " + label);
        product *= dimension;
    }
    return product;
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::map<std::string, std::string> read_kv(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) fail("cannot open configuration " + path.string());
    std::map<std::string, std::string> result;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        line = trim(line);
        if (line.empty()) continue;
        const auto equal = line.find('=');
        if (equal == std::string::npos) { fail("malformed key=value line " + std::to_string(line_no) + " in " + path.string()); }
        const std::string key = trim(line.substr(0, equal));
        const std::string value = trim(line.substr(equal + 1));
        if (key.empty() || value.empty()) fail("empty key/value in " + path.string());
        if (!result.emplace(key, value).second) fail("duplicate key " + key + " in " + path.string());
    }
    return result;
}

u64 parse_decimal_u64(const std::string &text, const std::string &what);

u64 parse_u64(const std::string &text, const std::string &what) {
    return parse_decimal_u64(text, what);
}

u64 parse_decimal_u64(const std::string &text, const std::string &what) {
    const std::string value_text = trim(text);
    if (value_text.empty() || (value_text.size() > 1U && value_text.front() == '0'))
        fail("invalid canonical decimal integer for " + what + ": " + text);
    u64 value = 0U;
    for (const char digit_char : value_text) {
        if (digit_char < '0' || digit_char > '9') fail("invalid decimal integer for " + what + ": " + text);
        const u64 digit = static_cast<u64>(digit_char - '0');
        if (value > (std::numeric_limits<u64>::max() - digit) / 10U) fail("decimal integer overflow for " + what);
        value = value * 10U + digit;
    }
    return value;
}

bool canonical_decimal_float(const std::string &text) {
    static const std::regex decimal_float(R"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)");
    return std::regex_match(text, decimal_float);
}

i64 parse_i64(const std::string &text, const std::string &what) {
    const std::string value_text = trim(text);
    if (value_text.empty()) fail("invalid signed integer for " + what + ": " + text);
    std::size_t used = 0;
    long long value = 0;
    try {
        value = std::stoll(value_text, &used, 0);
    } catch (...) { fail("invalid signed integer for " + what + ": " + text); }
    if (used != value_text.size()) fail("invalid trailing characters for " + what + ": " + text);
    return static_cast<i64>(value);
}

bool parse_bool(const std::string &text, const std::string &what) {
    const std::string value = trim(text);
    if (value == "true" || value == "1") return true;
    if (value == "false" || value == "0") return false;
    fail("invalid boolean for " + what + ": " + text);
}

double parse_double(const std::string &text, const std::string &what) {
    const std::string value_text = trim(text);
    std::size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(value_text, &used);
    } catch (...) { fail("invalid floating-point value for " + what + ": " + text); }
    if (used != value_text.size() || !std::isfinite(value) || value < 0.0) { fail("invalid non-negative floating-point value for " + what + ": " + text); }
    return value;
}

std::vector<u64> parse_list(const std::string &text, const std::string &what) {
    std::string value = trim(text);
    if (!value.empty() && value.front() == '[') {
        if (value.back() != ']') fail("unterminated list for " + what);
        value = value.substr(1, value.size() - 2);
    }
    std::vector<u64> output;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim(item);
        if (item.empty()) continue;
        output.push_back(parse_u64(item, what));
    }
    if (output.empty()) fail("empty list for " + what);
    return output;
}

std::vector<std::pair<u64, u64>> parse_profile(const std::string &text) {
    std::string value = trim(text);
    if (value.size() < 2 || value.front() != '[' || value.back() != ']') { fail("invalid hidden_profile: " + text); }
    value = value.substr(1, value.size() - 2);
    std::vector<std::pair<u64, u64>> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim(item);
        const auto x = item.find('x');
        if (x == std::string::npos) fail("invalid hidden_profile item: " + item);
        result.emplace_back(parse_u64(item.substr(0, x), "hidden head count"), parse_u64(item.substr(x + 1), "hidden head dimension"));
    }
    if (result.empty()) fail("empty hidden_profile");
    return result;
}

std::filesystem::path local_path(const std::filesystem::path &baseline, const std::string &raw, const std::string &field) {
    const std::filesystem::path candidate(raw);
    if (candidate.is_absolute()) fail(field + " must be baseline-local: " + raw);
    for (const auto &component : candidate) {
        if (component == "..") fail(field + " escapes baseline root: " + raw);
    }
    const auto normalized = (baseline / candidate).lexically_normal();
    const auto base = baseline.lexically_normal();
    const std::string base_text = base.generic_string();
    const std::string path_text = normalized.generic_string();
    if (path_text != base_text && path_text.rfind(base_text + "/", 0) != 0) { fail(field + " escapes baseline root: " + raw); }
    return normalized;
}

std::string file_text(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) fail("cannot open " + path.string());
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

class Sha256 {
  public:
    Sha256() : bit_count_(0), buffered_(0) {
        state_[0] = 0x6a09e667U;
        state_[1] = 0xbb67ae85U;
        state_[2] = 0x3c6ef372U;
        state_[3] = 0xa54ff53aU;
        state_[4] = 0x510e527fU;
        state_[5] = 0x9b05688cU;
        state_[6] = 0x1f83d9abU;
        state_[7] = 0x5be0cd19U;
    }

    void update(const void *data, std::size_t length) {
        const auto *bytes = static_cast<const std::uint8_t *>(data);
        bit_count_ += static_cast<std::uint64_t>(length) * 8U;
        while (length != 0) {
            const std::size_t take = std::min<std::size_t>(length, 64U - buffered_);
            std::memcpy(buffer_.data() + buffered_, bytes, take);
            buffered_ += take;
            bytes += take;
            length -= take;
            if (buffered_ == 64U) {
                transform(buffer_.data());
                buffered_ = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> finish() {
        const std::uint64_t original_bits = bit_count_;
        const std::uint8_t one = 0x80U;
        update(&one, 1);
        const std::uint8_t zero = 0;
        while (buffered_ != 56U) update(&zero, 1);
        std::array<std::uint8_t, 8> length{};
        for (int i = 0; i < 8; ++i) { length[7 - i] = static_cast<std::uint8_t>(original_bits >> (i * 8)); }
        update(length.data(), length.size());
        std::array<std::uint8_t, 32> result{};
        for (int i = 0; i < 8; ++i) {
            result[i * 4] = static_cast<std::uint8_t>(state_[i] >> 24);
            result[i * 4 + 1] = static_cast<std::uint8_t>(state_[i] >> 16);
            result[i * 4 + 2] = static_cast<std::uint8_t>(state_[i] >> 8);
            result[i * 4 + 3] = static_cast<std::uint8_t>(state_[i]);
        }
        return result;
    }

  private:
    static constexpr std::array<std::uint32_t, 64> constants_ = {0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
                                                                 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
                                                                 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
                                                                 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
                                                                 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
                                                                 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
                                                                 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
                                                                 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
                                                                 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
                                                                 0xc67178f2U};

    static std::uint32_t rotr(std::uint32_t value, unsigned amount) { return (value >> amount) | (value << (32U - amount)); }
    static std::uint32_t choose(std::uint32_t x, std::uint32_t y, std::uint32_t z) { return (x & y) ^ (~x & z); }
    static std::uint32_t majority(std::uint32_t x, std::uint32_t y, std::uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
    static std::uint32_t big_sigma0(std::uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
    static std::uint32_t big_sigma1(std::uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
    static std::uint32_t small_sigma0(std::uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
    static std::uint32_t small_sigma1(std::uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

    void transform(const std::uint8_t *block) {
        std::array<std::uint32_t, 64> words{};
        for (int i = 0; i < 16; ++i) {
            words[i] = (static_cast<std::uint32_t>(block[4 * i]) << 24) | (static_cast<std::uint32_t>(block[4 * i + 1]) << 16) |
                       (static_cast<std::uint32_t>(block[4 * i + 2]) << 8) | static_cast<std::uint32_t>(block[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) { words[i] = small_sigma1(words[i - 2]) + words[i - 7] + small_sigma0(words[i - 15]) + words[i - 16]; }
        std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t t1 = h + big_sigma1(e) + choose(e, f, g) + constants_[i] + words[i];
            const std::uint32_t t2 = big_sigma0(a) + majority(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> buffer_{};
    std::uint64_t bit_count_;
    std::size_t buffered_;
};

std::string hex(const std::array<std::uint8_t, 32> &digest) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const auto byte : digest) out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

std::string sha256_bytes(const void *data, std::size_t size) {
    Sha256 hash;
    hash.update(data, size);
    return hex(hash.finish());
}

std::string sha256_text(const std::string &text) { return sha256_bytes(text.data(), text.size()); }

std::string json_quote_config(const std::string &value) {
    std::string result;
    result.reserve(value.size() + 2U);
    result.push_back('"');
    for (const unsigned char byte : value) {
        if (byte == '"' || byte == '\\') {
            result.push_back('\\');
            result.push_back(static_cast<char>(byte));
        } else if (byte == '\n') {
            result += "\\n";
        } else if (byte == '\r') {
            result += "\\r";
        } else if (byte == '\t') {
            result += "\\t";
        } else {
            result.push_back(static_cast<char>(byte));
        }
    }
    result.push_back('"');
    return result;
}

void append_native_shape_counts_json(
    std::ostream &output, const NativeRelationPlanMetrics &metrics) {
    output << ",\"native_constraint_opening_segment_count\":"
           << metrics.constraint_opening_segment_count
           << ",\"native_padding_opening_segment_count\":"
           << metrics.padding_opening_segment_count
           << ",\"native_sorted_lookup_opening_segment_count\":"
           << metrics.sorted_lookup_opening_segment_count
           << ",\"native_sparse_route_opening_count\":"
           << metrics.sparse_route_opening_count
           << ",\"native_fac_selector_opening_count\":"
           << metrics.fac_selector_opening_count
           << ",\"native_route_aggregate_input_segment_count\":"
           << metrics.route_aggregate_input_segment_count
           << ",\"native_public_output_opening_segment_count\":"
           << metrics.public_output_opening_segment_count
           << ",\"native_computed_hyrax_opening_count\":"
           << metrics.computed_hyrax_opening_count
           << ",\"native_retained_hyrax_opening_object_count\":"
           << metrics.retained_hyrax_opening_object_count
           << ",\"native_retained_hyrax_terminal_mirror_copy_count\":"
           << metrics.retained_hyrax_terminal_mirror_copy_count
           << ",\"native_total_hyrax_opening_segment_count\":"
           << metrics.total_hyrax_opening_segment_count
           << ",\"native_hyrax_round_vector_capacity_bytes_upper_bound\":"
           << metrics.hyrax_round_vector_capacity_bytes_upper_bound
           << ",\"native_hyrax_point_vector_capacity_bytes_upper_bound\":"
           << metrics.hyrax_point_vector_capacity_bytes_upper_bound
           << ",\"native_hyrax_opening_object_header_bytes_upper_bound\":"
           << metrics.hyrax_opening_object_header_bytes_upper_bound
           << ",\"native_hyrax_maximum_single_opening_vector_bytes_upper_bound\":"
           << metrics.hyrax_maximum_single_opening_vector_bytes_upper_bound
           << ",\"native_hyrax_opening_segment_domain_histogram\":[";
    for (std::size_t index = 0U;
         index < metrics.hyrax_opening_segment_domain_histogram.size(); ++index) {
        if (index != 0U) output << ',';
        const auto &bucket = metrics.hyrax_opening_segment_domain_histogram[index];
        output << "{\"logical_domain\":" << bucket.logical_domain
               << ",\"segment_size\":" << bucket.segment_size
               << ",\"padded_generator_size\":" << bucket.padded_generator_size
               << ",\"computed_opening_count\":" << bucket.computed_opening_count
               << ",\"retained_object_count\":" << bucket.retained_object_count
               << ",\"terminal_mirror_copy_count\":" << bucket.terminal_mirror_copy_count
               << ",\"retained_round_vector_bytes_upper_bound\":"
               << bucket.retained_round_vector_bytes_upper_bound
               << ",\"retained_point_vector_bytes_upper_bound\":"
               << bucket.retained_point_vector_bytes_upper_bound << '}';
    }
    output << "],\"native_sumcheck_metrics_scope\":\"segmented_calls_only\""
           << ",\"native_hidden_matmul_unsegmented_sumcheck_call_count\":"
           << metrics.primitive_count
           << ",\"native_sumcheck_folded_state_budget_bytes\":"
           << metrics.sumcheck_folded_state_budget_bytes
           << ",\"native_sumcheck_cached_call_count\":"
           << metrics.sumcheck_cached_call_count
           << ",\"native_sumcheck_fallback_call_count\":"
           << metrics.sumcheck_fallback_call_count
           << ",\"native_sumcheck_spill_call_count\":"
           << metrics.sumcheck_spill_call_count
           << ",\"native_sumcheck_forced_cached_call_count\":"
           << metrics.sumcheck_forced_cached_call_count
           << ",\"native_sumcheck_cached_provider_value_slots_upper_bound\":"
           << metrics.sumcheck_cached_provider_value_slots_upper_bound
           << ",\"native_sumcheck_fallback_provider_value_slots_upper_bound\":"
           << metrics.sumcheck_fallback_provider_value_slots_upper_bound
           << ",\"native_sumcheck_cached_provider_scan_bytes_upper_bound\":"
           << metrics.sumcheck_cached_provider_scan_bytes_upper_bound
           << ",\"native_sumcheck_fallback_provider_scan_bytes_upper_bound\":"
           << metrics.sumcheck_fallback_provider_scan_bytes_upper_bound
           << ",\"native_sumcheck_spill_provider_value_slots_upper_bound\":"
           << metrics.sumcheck_spill_provider_value_slots_upper_bound
           << ",\"native_sumcheck_spill_provider_scan_bytes_upper_bound\":"
           << metrics.sumcheck_spill_provider_scan_bytes_upper_bound
           << ",\"native_sumcheck_spill_provider_source_bytes_upper_bound\":"
           << metrics.sumcheck_spill_provider_source_bytes_upper_bound
           << ",\"native_sumcheck_spill_storage_io_bytes_upper_bound\":"
           << metrics.sumcheck_spill_storage_io_bytes_upper_bound
           << ",\"native_sumcheck_max_spill_disk_peak_bytes_upper_bound\":"
           << metrics.sumcheck_max_spill_disk_peak_bytes_upper_bound
           << ",\"native_sumcheck_max_spill_ram_peak_bytes_upper_bound\":"
           << metrics.sumcheck_max_spill_ram_peak_bytes_upper_bound
           << ",\"native_sumcheck_max_folded_state_peak_bytes_upper_bound\":"
           << metrics.sumcheck_max_folded_state_peak_bytes_upper_bound
           << ",\"native_sumcheck_max_streaming_peak_bytes_upper_bound\":"
           << metrics.sumcheck_max_streaming_peak_bytes_upper_bound
           << ",\"native_sumcheck_shape_histogram\":[";
    for (std::size_t index = 0U; index < metrics.sumcheck_shape_histogram.size(); ++index) {
        if (index != 0U) output << ',';
        const auto &bucket = metrics.sumcheck_shape_histogram[index];
        output << "{\"logical_domain\":" << bucket.logical_domain
               << ",\"segment_size\":" << bucket.segment_size
               << ",\"provider_count\":" << bucket.provider_count
               << ",\"cached_call_count\":" << bucket.cached_call_count
               << ",\"fallback_call_count\":" << bucket.fallback_call_count
               << ",\"spill_call_count\":" << bucket.spill_call_count
               << ",\"forced_cached_call_count\":" << bucket.forced_cached_call_count
               << ",\"cached_provider_value_slots_upper_bound\":"
               << bucket.cached_provider_value_slots_upper_bound
               << ",\"fallback_provider_value_slots_upper_bound\":"
               << bucket.fallback_provider_value_slots_upper_bound
               << ",\"spill_provider_value_slots_upper_bound\":"
               << bucket.spill_provider_value_slots_upper_bound << '}';
    }
    output << ']';
}

std::string config_identity_digest(const std::map<std::string, std::string> &values) {
    std::string canonical = "{";
    bool first = true;
    for (const auto &[key, value] : values) {
        if (key == "verifier_registry_digest" || key == "model_blinder_sidecar_digest") continue;
        if (!first) canonical.push_back(',');
        first = false;
        canonical += json_quote_config(key) + ":" + json_quote_config(value);
    }
    canonical.push_back('}');
    return sha256_text(canonical);
}

std::string config_binding_identity_digest(const std::map<std::string, std::string> &values) {
    std::string canonical = "{";
    bool first = true;
    for (const auto &[key, value] : values) {
        if (key == "srs_manifest_digest" || key == "verifier_registry_digest" || key == "model_blinder_sidecar_digest") continue;
        if (!first) canonical.push_back(',');
        first = false;
        canonical += json_quote_config(key) + ":" + json_quote_config(value);
    }
    canonical.push_back('}');
    return sha256_text(canonical);
}

std::string sha256_file(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) fail("cannot open for hashing " + path.string());
    Sha256 hash;
    std::array<char, 1 << 16> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) hash.update(buffer.data(), static_cast<std::size_t>(count));
    }
    return hex(hash.finish());
}

void hash_u64(Sha256 &hash, u64 value) {
    std::array<std::uint8_t, 8> bytes{};
    for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<std::uint8_t>(value >> (8U * i));
    hash.update(bytes.data(), bytes.size());
}

void hash_i64(Sha256 &hash, i64 value) { hash_u64(hash, static_cast<u64>(value)); }

std::string digest_matrix(const Matrix &matrix) {
    Sha256 hash;
    hash_u64(hash, matrix.rows);
    hash_u64(hash, matrix.cols);
    for (const auto value : matrix.values) hash_i64(hash, value);
    return hex(hash.finish());
}

std::string digest_graph(const Graph &graph) {
    Sha256 hash;
    hash_u64(hash, graph.total_nodes);
    hash_u64(hash, graph.nodes);
    hash_u64(hash, graph.edges);
    hash_u64(hash, graph.graphs);
    hash_u64(hash, graph.features);
    hash_u64(hash, graph.absolute_ids.size());
    for (const auto value : graph.absolute_ids) hash_u64(hash, value);
    hash_u64(hash, graph.node_ptr.size());
    for (const auto value : graph.node_ptr) hash_u64(hash, value);
    hash_u64(hash, graph.edge_ptr.size());
    for (const auto value : graph.edge_ptr) hash_u64(hash, value);
    hash_u64(hash, graph.node_edge_ptr.size());
    for (const auto value : graph.node_edge_ptr) hash_u64(hash, value);
    hash_u64(hash, graph.src.size());
    for (std::size_t i = 0; i < graph.src.size(); ++i) {
        hash_u64(hash, graph.src[i]);
        hash_u64(hash, graph.dst[i]);
    }
    return hex(hash.finish());
}

std::string digest_canonical_graph(const Graph &graph) {
    Sha256 hash;
    constexpr char tag[] = "comparison-canonical-graph-v1";
    hash.update(reinterpret_cast<const std::uint8_t *>(tag), sizeof(tag));
    hash_u64(hash, graph.nodes);
    hash_u64(hash, graph.edges);
    hash_u64(hash, graph.graphs);
    const auto append = [&](const auto &values) {
        hash_u64(hash, values.size());
        for (const auto value : values) hash_u64(hash, value);
    };
    append(graph.src);
    append(graph.dst);
    append(graph.node_ptr);
    append(graph.edge_ptr);
    return hex(hash.finish());
}

std::string digest_graph_boundaries(const Graph &graph) {
    Sha256 hash;
    const std::string tag = "zkDL_gat_graph_boundaries_v1";
    hash_u64(hash, tag.size());
    hash.update(tag.data(), tag.size());
    const auto append = [&](const std::vector<u64> &values) {
        hash_u64(hash, values.size());
        for (const auto value : values) hash_u64(hash, value);
    };
    append(graph.node_ptr);
    append(graph.edge_ptr);
    append(graph.node_edge_ptr);
    return hex(hash.finish());
}

std::string digest_tables(const StaticTables &tables) {
    Sha256 hash;
    const auto hash_string = [&](const std::string &value) {
        hash_u64(hash, static_cast<u64>(value.size()));
        hash.update(value.data(), value.size());
    };
    hash_string("zkDL_gat_static_tables_v1");
    hash_string(tables.generator_id);
    hash_string(tables.version);
    hash_string(tables.scale_id);
    hash_string(tables.domain_id);
    hash_string(tables.row_order);
    hash_string(tables.spec_hash);
    hash_string(tables.padding_rule);
    hash_u64(hash, tables.lrelu_padded_domain);
    hash_u64(hash, tables.elu_padded_domain);
    hash_u64(hash, tables.exp_padded_domain);
    hash_i64(hash, tables.lrelu_min);
    hash_i64(hash, tables.lrelu_max);
    hash_i64(hash, tables.elu_min);
    hash_i64(hash, tables.elu_max);
    hash_u64(hash, tables.exp_max);
    hash_u64(hash, tables.lrelu.size());
    for (const auto value : tables.lrelu) hash_i64(hash, value);
    hash_u64(hash, tables.elu.size());
    for (const auto value : tables.elu) hash_i64(hash, value);
    hash_u64(hash, tables.exp.size());
    for (const auto value : tables.exp) hash_i64(hash, value);
    return hex(hash.finish());
}

std::string digest_ledger(const IntegerBoundLedger &ledger) {
    return integer_bound_ledger::digest(ledger);
}

std::string json_fragment(const std::string &text, const std::string &key, char open, char close) {
    const std::string marker = "\"" + key + "\"";
    const auto marker_pos = text.find(marker);
    if (marker_pos == std::string::npos) fail("JSON field missing: " + key);
    const auto open_pos = text.find(open, marker_pos + marker.size());
    if (open_pos == std::string::npos) fail("JSON field has no value: " + key);
    bool quoted = false;
    bool escaped = false;
    int depth = 0;
    for (std::size_t index = open_pos; index < text.size(); ++index) {
        const char current = text[index];
        if (quoted) {
            if (escaped)
                escaped = false;
            else if (current == '\\')
                escaped = true;
            else if (current == '\"')
                quoted = false;
            continue;
        }
        if (current == '\"') {
            quoted = true;
            continue;
        }
        if (current == open) ++depth;
        if (current == close && --depth == 0) return text.substr(open_pos, index - open_pos + 1);
    }
    fail("unterminated JSON field: " + key);
}

std::vector<std::string> json_object_array(const std::string &text, const std::string &key,
                                           std::size_t maximum_values = 1U << 16U) {
    const std::string fragment = json_fragment(text, key, '[', ']');
    std::vector<std::string> objects;
    std::size_t index = 1U;
    const auto skip_space = [&]() {
        while (index < fragment.size() && std::isspace(static_cast<unsigned char>(fragment[index]))) ++index;
    };
    skip_space();
    while (index + 1U < fragment.size() && fragment[index] != ']') {
        if (fragment[index] != '{') fail("analysis-only planned JSON array contains a non-object: " + key);
        const std::size_t begin = index;
        bool quoted = false, escaped = false;
        std::size_t depth = 0U;
        for (; index < fragment.size(); ++index) {
            const char current = fragment[index];
            if (quoted) {
                if (escaped) escaped = false;
                else if (current == '\\') escaped = true;
                else if (current == '"') quoted = false;
                continue;
            }
            if (current == '"') { quoted = true; continue; }
            if (current == '{') ++depth;
            else if (current == '}') {
                if (depth == 0U) fail("analysis-only planned JSON object depth underflow: " + key);
                if (--depth == 0U) {
                    ++index;
                    objects.push_back(fragment.substr(begin, index - begin));
                    break;
                }
            }
        }
        if (depth != 0U || objects.size() > maximum_values)
            fail("analysis-only planned JSON object array is truncated or exceeds its bound: " + key);
        skip_space();
        if (index < fragment.size() && fragment[index] == ',') { ++index; skip_space(); }
        else if (index < fragment.size() && fragment[index] != ']')
            fail("analysis-only planned JSON object array separator is invalid: " + key);
    }
    if (objects.empty()) fail("analysis-only planned JSON object array is empty: " + key);
    return objects;
}

std::map<std::string, u64> json_u64_object(const std::string &text, const std::string &key,
                                           std::size_t maximum_values = 1U << 20U) {
    const std::string fragment = json_fragment(text, key, '{', '}');
    std::map<std::string, u64> result;
    std::size_t index = 1U;
    const auto skip_space = [&]() {
        while (index < fragment.size() && std::isspace(static_cast<unsigned char>(fragment[index]))) ++index;
    };
    skip_space();
    while (index + 1U < fragment.size() && fragment[index] != '}') {
        if (fragment[index] != '"') fail("analysis-only planned numeric object key is invalid: " + key);
        const std::size_t key_begin = ++index;
        const auto key_end = fragment.find('"', key_begin);
        if (key_end == std::string::npos) fail("analysis-only planned numeric object key is unterminated: " + key);
        const std::string item_key = fragment.substr(key_begin, key_end - key_begin);
        index = key_end + 1U;
        skip_space();
        if (index >= fragment.size() || fragment[index++] != ':')
            fail("analysis-only planned numeric object has no key separator: " + key);
        skip_space();
        const std::size_t value_begin = index;
        while (index < fragment.size() && fragment[index] >= '0' && fragment[index] <= '9') ++index;
        if (value_begin == index) fail("analysis-only planned numeric object value is not unsigned: " + key);
        const u64 value = parse_u64(fragment.substr(value_begin, index - value_begin), key + "/" + item_key);
        if (!result.emplace(item_key, value).second || result.size() > maximum_values)
            fail("analysis-only planned numeric object has duplicate keys or exceeds its bound: " + key);
        skip_space();
        if (index < fragment.size() && fragment[index] == ',') { ++index; skip_space(); }
        else if (index < fragment.size() && fragment[index] != '}')
            fail("analysis-only planned numeric object separator is invalid: " + key);
    }
    if (result.empty()) fail("analysis-only planned numeric object is empty: " + key);
    return result;
}

std::vector<u64> json_u64_array(const std::string &text, const std::string &key,
                                std::size_t maximum_values = 16U) {
    const std::string fragment = json_fragment(text, key, '[', ']');
    std::vector<u64> values;
    std::stringstream stream(fragment.substr(1U, fragment.size() - 2U));
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim(item);
        if (item.empty()) continue;
        if (values.size() >= maximum_values) fail("analysis-only planned integer array exceeds its bound: " + key);
        values.push_back(parse_u64(item, "analysis-only planned " + key));
    }
    if (values.empty()) fail("analysis-only planned integer array is empty: " + key);
    return values;
}

std::string json_scalar(const std::string &text, const std::string &key) {
    const std::string marker = "\"" + key + "\"";
    const auto marker_pos = text.find(marker);
    if (marker_pos == std::string::npos) fail("JSON field missing: " + key);
    const auto colon = text.find(':', marker_pos + marker.size());
    if (colon == std::string::npos) fail("JSON field has no separator: " + key);
    std::size_t begin = colon + 1;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
    if (begin >= text.size()) fail("JSON field has no value: " + key);
    if (text[begin] == '\"') {
        const auto end = text.find('\"', begin + 1);
        if (end == std::string::npos) fail("unterminated JSON string: " + key);
        return text.substr(begin + 1, end - begin - 1);
    }
    std::size_t end = begin;
    while (end < text.size() && text[end] != ',' && text[end] != '}' && !std::isspace(static_cast<unsigned char>(text[end]))) ++end;
    return text.substr(begin, end - begin);
}

std::vector<i64> json_i64_array(const std::string &text, const std::string &key, u64 maximum_values = kMaxStaticTableRows) {
    const auto fragment = json_fragment(text, key, '[', ']');
    std::vector<i64> values;
    std::stringstream stream(fragment.substr(1, fragment.size() - 2));
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            if (values.size() >= maximum_values) fail("BLOCKED: JSON array exceeds bounded size: " + key);
            values.push_back(parse_i64(item, "JSON " + key));
        }
    }
    if (values.empty()) fail("JSON array is empty: " + key);
    return values;
}

struct StaticTableDescriptorShape {
    std::map<std::string, std::string> scalars;
    std::set<std::string> arrays;
    std::map<std::string, u64> array_element_counts;
};

class JsonShapeReader {
  public:
    explicit JsonShapeReader(const std::filesystem::path &path) : input_(path, std::ios::binary) {
        if (!input_) fail("cannot open static table descriptor for shape scan: " + path.string());
    }

    StaticTableDescriptorShape read_static_table_descriptor() {
        StaticTableDescriptorShape result;
        expect('{');
        skip_space();
        if (peek() == '}') fail("static table descriptor shape has no fields");
        for (;;) {
            const std::string key = read_string();
            expect(':');
            if (key == "lrelu" || key == "elu" || key == "exp") {
                const u64 element_count = read_integer_array_count();
                if (!result.arrays.insert(key).second)
                    fail("duplicate static table array in shape scan: " + key);
                result.array_element_counts.emplace(key, element_count);
            } else {
                std::string value;
                if (peek() == '"') value = read_string();
                else value = read_integer_token();
                if (!result.scalars.emplace(key, std::move(value)).second)
                    fail("duplicate static table scalar in shape scan: " + key);
            }
            skip_space();
            const int separator = get();
            if (separator == '}') break;
            if (separator != ',') fail("malformed static table descriptor object");
            skip_space();
        }
        skip_space();
        if (peek() != std::char_traits<char>::eof())
            fail("static table descriptor has trailing bytes after its JSON object");
        return result;
    }

  private:
    int peek() { return input_.peek(); }
    int get() { return input_.get(); }

    void skip_space() {
        while (peek() != std::char_traits<char>::eof() &&
               std::isspace(static_cast<unsigned char>(peek()))) get();
    }

    void expect(char expected) {
        skip_space();
        if (get() != expected) fail(std::string("malformed static table descriptor: expected '") + expected + "'");
    }

    void skip_value() {
        skip_space();
        const int first = peek();
        if (first == std::char_traits<char>::eof()) fail("static table descriptor value is truncated");
        if (first == '"') {
            (void)read_string();
            return;
        }
        if (first == '[' || first == '{') {
            std::vector<char> closers;
            bool quoted = false;
            bool escaped = false;
            for (;;) {
                const int next = get();
                if (next == std::char_traits<char>::eof()) fail("unterminated static table descriptor array/object");
                if (quoted) {
                    if (escaped) escaped = false;
                    else if (next == '\\') escaped = true;
                    else if (next == '"') quoted = false;
                    continue;
                }
                if (next == '"') quoted = true;
                else if (next == '[') closers.push_back(']');
                else if (next == '{') closers.push_back('}');
                else if (next == ']' || next == '}') {
                    if (closers.empty() || closers.back() != next)
                        fail("mismatched static table descriptor array/object delimiter");
                    closers.pop_back();
                    if (closers.empty()) return;
                }
                if (closers.size() > 32U) fail("static table descriptor nesting exceeds its shape bound");
            }
        }
        (void)read_integer_token();
    }

    u64 read_integer_array_count() {
        expect('[');
        skip_space();
        if (peek() == ']') {
            get();
            return 0U;
        }
        u64 count = 0U;
        for (;;) {
            (void)read_integer_token();
            if (count == std::numeric_limits<u64>::max())
                fail("static table descriptor array element count overflows");
            ++count;
            skip_space();
            const int separator = get();
            if (separator == ']') return count;
            if (separator != ',') fail("malformed integer array in static table descriptor");
        }
    }

    std::string read_string() {
        expect('"');
        std::string value;
        for (;;) {
            const int next = get();
            if (next == std::char_traits<char>::eof()) fail("unterminated static table descriptor string");
            if (next == '"') return value;
            if (next == '\\') {
                const int escaped = get();
                if (escaped == '"' || escaped == '\\' || escaped == '/') value.push_back(static_cast<char>(escaped));
                else if (escaped == 'b') value.push_back('\b');
                else if (escaped == 'f') value.push_back('\f');
                else if (escaped == 'n') value.push_back('\n');
                else if (escaped == 'r') value.push_back('\r');
                else if (escaped == 't') value.push_back('\t');
                else fail("unsupported escape in static table descriptor identity string");
            } else {
                if (static_cast<unsigned char>(next) < 0x20U) fail("control byte in static table descriptor string");
                value.push_back(static_cast<char>(next));
            }
            if (value.size() > 4096U) fail("static table descriptor identity string exceeds its bound");
        }
    }

    std::string read_integer_token() {
        skip_space();
        std::string token;
        if (peek() == '-') token.push_back(static_cast<char>(get()));
        while (peek() >= '0' && peek() <= '9') {
            token.push_back(static_cast<char>(get()));
            if (token.size() > 32U) fail("static table descriptor integer token is too long");
        }
        if (token.empty() || token == "-") fail("static table descriptor contains a non-integer scalar");
        if (peek() != ',' && peek() != '}' && peek() != ']' &&
            peek() != std::char_traits<char>::eof() &&
            !std::isspace(static_cast<unsigned char>(peek())))
            fail("static table descriptor integer has a non-canonical suffix");
        (void)parse_i64(token, "static table descriptor integer");
        return token;
    }

    std::ifstream input_;
};

std::vector<std::string> json_objects(const std::string &text, const std::string &key, u64 maximum_objects = kMaxLedgerFamilies) {
    const auto fragment = json_fragment(text, key, '[', ']');
    std::vector<std::string> objects;
    bool quoted = false;
    bool escaped = false;
    int depth = 0;
    std::size_t begin = 0;
    for (std::size_t index = 1; index + 1 < fragment.size(); ++index) {
        const char current = fragment[index];
        if (quoted) {
            if (escaped)
                escaped = false;
            else if (current == '\\')
                escaped = true;
            else if (current == '\"')
                quoted = false;
            continue;
        }
        if (current == '\"') {
            quoted = true;
            continue;
        }
        if (current == '{') {
            if (depth++ == 0) begin = index;
        } else if (current == '}' && depth > 0 && --depth == 0) {
            if (objects.size() >= maximum_objects) fail("BLOCKED: JSON object array exceeds bounded size: " + key);
            objects.push_back(fragment.substr(begin, index - begin + 1));
        }
    }
    if (objects.empty()) fail("JSON object array is empty: " + key);
    return objects;
}

i64 json_i64_value(const std::string &text, const std::string &key) { return parse_i64(json_scalar(text, key), "JSON " + key); }

u64 json_u64_value(const std::string &text, const std::string &key) { return parse_u64(json_scalar(text, key), "JSON " + key); }

bool json_bool_value(const std::string &text, const std::string &key) { return parse_bool(json_scalar(text, key), "JSON " + key); }

i64 checked_i128(__int128 value, const std::string &what) {
    if (value < static_cast<__int128>(std::numeric_limits<i64>::min()) || value > static_cast<__int128>(std::numeric_limits<i64>::max())) {
        fail("signed 64-bit overflow in " + what);
    }
    return static_cast<i64>(value);
}

u64 checked_domain_rows(i64 minimum, i64 maximum, const std::string &name) {
    if (minimum > maximum) fail("invalid domain bounds in " + name);
    const __int128 rows = static_cast<__int128>(maximum) - static_cast<__int128>(minimum) + 1;
    if (rows <= 0 || rows > static_cast<__int128>(kMaxStaticTableRows) || rows > static_cast<__int128>(std::numeric_limits<std::size_t>::max())) {
        fail("BLOCKED: table domain is too large in " + name);
    }
    return static_cast<u64>(rows);
}

i64 round_div(i64 numerator, i64 denominator, const std::string &what) {
    if (denominator <= 0) fail("non-positive divisor in " + what);
    const __int128 n = numerator;
    const __int128 d = denominator;
    if (n >= 0) return checked_i128((n + d / 2) / d, what);
    return checked_i128(-((-n + d / 2) / d), what);
}

i64 rounded_float(double value, u64 scale, const std::string &what) {
    if (!std::isfinite(value)) fail("non-finite model value in " + what);
    const long double scaled = static_cast<long double>(value) * static_cast<long double>(scale);
    const long double rounded = scaled >= 0 ? std::floor(scaled + 0.5L) : std::ceil(scaled - 0.5L);
    const long double signed_i64_limit = std::ldexp(1.0L, 63);
    if (rounded < -signed_i64_limit || rounded >= signed_i64_limit) {
        fail("quantization overflow in " + what);
    }
    return static_cast<i64>(rounded);
}

Matrix make_matrix(u64 rows, u64 cols) {
    if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows) fail("matrix size overflow");
    Matrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.values.assign(static_cast<std::size_t>(rows * cols), 0);
    return matrix;
}

const ModelTensor &model_tensor(const ModelBundle &model, const std::string &name) {
    const auto iterator = model.tensors.find(name);
    if (iterator == model.tensors.end()) fail("checkpoint tensor missing: " + name);
    return iterator->second;
}

Matrix quantized_tensor_matrix(const ModelBundle &model, const std::string &name, u64 scale) {
    const auto &tensor = model_tensor(model, name);
    if (tensor.shape.size() < 2) fail("tensor is not matrix-shaped: " + name);
    const u64 rows = tensor.shape[tensor.shape.size() - 2];
    const u64 cols = tensor.shape.back();
    if (rows != 0U && cols > std::numeric_limits<u64>::max() / rows) fail("tensor matrix product overflows native width: " + name);
    const u64 tensor_count = rows * cols;
    if (tensor.values.size() != tensor_count || tensor_count > kMaxNativeDomain) fail("tensor count mismatch or exceeds native domain: " + name);
    Matrix output = make_matrix(rows, cols);
    for (u64 row = 0; row < rows; ++row) {
        for (u64 col = 0; col < cols; ++col) { output.at(row, col) = rounded_float(tensor.values[static_cast<std::size_t>(row * cols + col)], scale, name); }
    }
    return output;
}

std::vector<i64> quantized_tensor_vector(const ModelBundle &model, const std::string &name, u64 scale) {
    const auto &tensor = model_tensor(model, name);
    std::vector<i64> output;
    output.reserve(tensor.values.size());
    for (const auto value : tensor.values) output.push_back(rounded_float(value, scale, name));
    return output;
}

std::int64_t table_lookup(const std::vector<i64> &values, i64 minimum, i64 maximum, i64 index, const std::string &name) {
    if (index < minimum || index > maximum) { fail(name + " lookup outside committed table domain"); }
    const __int128 offset = static_cast<__int128>(index) - static_cast<__int128>(minimum);
    if (offset < 0 || offset >= static_cast<__int128>(values.size())) { fail(name + " lookup offset exceeds committed table"); }
    return values.at(static_cast<std::size_t>(offset));
}

Matrix matmul_cpu(const Matrix &left, const Matrix &right, const std::string &name) {
    if (left.cols != right.rows) fail("matmul dimension mismatch in " + name);
    Matrix output = make_matrix(left.rows, right.cols);
    for (u64 row = 0; row < left.rows; ++row) {
        for (u64 col = 0; col < right.cols; ++col) {
            __int128 sum = 0;
            for (u64 inner = 0; inner < left.cols; ++inner) { sum += static_cast<__int128>(left.at(row, inner)) * right.at(inner, col); }
            output.at(row, col) = checked_i128(sum, name);
        }
    }
    return output;
}

__global__ void matmul_kernel(const long long *left, const long long *right, long long *output, unsigned long long rows, unsigned long long inner,
                              unsigned long long cols) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * cols) return;
    const unsigned long long row = index / cols;
    const unsigned long long col = index % cols;
    long long sum = 0;
    for (unsigned long long k = 0; k < inner; ++k) sum += left[row * inner + k] * right[k * cols + col];
    output[index] = sum;
}

void cuda_check(cudaError_t error, const std::string &what) {
    if (error != cudaSuccess) fail(what + ": " + cudaGetErrorString(error));
}

Matrix matmul_gpu(const Matrix &left, const Matrix &right, const std::string &name) {
    if (left.cols != right.rows) fail("matmul dimension mismatch in " + name);
    const std::size_t left_bytes = left.values.size() * sizeof(i64);
    const std::size_t right_bytes = right.values.size() * sizeof(i64);
    const std::size_t output_bytes = static_cast<std::size_t>(left.rows * right.cols) * sizeof(i64);
    long long *left_device = nullptr;
    long long *right_device = nullptr;
    long long *output_device = nullptr;
    try {
        cuda_check(cudaMalloc(&left_device, left_bytes), name + " cudaMalloc(left)");
        cuda_check(cudaMalloc(&right_device, right_bytes), name + " cudaMalloc(right)");
        cuda_check(cudaMalloc(&output_device, output_bytes), name + " cudaMalloc(output)");
        cuda_check(cudaMemcpy(left_device, left.values.data(), left_bytes, cudaMemcpyHostToDevice), name + " H2D(left)");
        cuda_check(cudaMemcpy(right_device, right.values.data(), right_bytes, cudaMemcpyHostToDevice), name + " H2D(right)");
        const auto count = left.rows * right.cols;
        const unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
        matmul_kernel<<<blocks, threads>>>(left_device, right_device, output_device, left.rows, left.cols, right.cols);
        cuda_check(cudaGetLastError(), name + " kernel launch");
        cuda_check(cudaDeviceSynchronize(), name + " kernel synchronize");
        Matrix output = make_matrix(left.rows, right.cols);
        cuda_check(cudaMemcpy(output.values.data(), output_device, output_bytes, cudaMemcpyDeviceToHost), name + " D2H(output)");
        cudaFree(left_device);
        cudaFree(right_device);
        cudaFree(output_device);
        return output;
    } catch (...) {
        if (left_device) cudaFree(left_device);
        if (right_device) cudaFree(right_device);
        if (output_device) cudaFree(output_device);
        throw;
    }
}

std::vector<i64> attention_scores(const Matrix &projected, const Matrix &src_kernel, const Matrix &dst_kernel, const Graph &graph, std::vector<i64> *e_src,
                                  std::vector<i64> *e_dst, std::vector<i64> *score, const std::string &name) {
    if (src_kernel.rows != projected.cols || dst_kernel.rows != projected.cols || src_kernel.cols != 1 || dst_kernel.cols != 1)
        fail("attention kernel shape mismatch in " + name);
    e_src->assign(graph.edges, 0);
    e_dst->assign(graph.edges, 0);
    score->assign(graph.edges, 0);
    for (u64 edge = 0; edge < graph.edges; ++edge) {
        __int128 source = 0, destination = 0;
        for (u64 dim = 0; dim < projected.cols; ++dim) {
            source += static_cast<__int128>(projected.at(graph.src[edge], dim)) * src_kernel.at(dim, 0);
            destination += static_cast<__int128>(projected.at(graph.dst[edge], dim)) * dst_kernel.at(dim, 0);
        }
        (*e_src)[edge] = checked_i128(source, name + ".e_src");
        (*e_dst)[edge] = checked_i128(destination, name + ".e_dst");
        (*score)[edge] = checked_i128(source + destination, name + ".score");
    }
    return *score;
}

void aggregate_head(const Config &config, const Graph &graph, const StaticTables &tables, const Matrix &projected, const std::vector<i64> &score,
                    u64 h_prime_scale, u64 aggregate_pre_scale, HiddenHeadTrace *output, const std::string &name) {
    output->h_prime = projected;
    output->leaky_relu.resize(graph.edges);
    output->maximum.assign(graph.nodes, std::numeric_limits<i64>::min());
    output->delta.resize(graph.edges);
    output->exp_weight.resize(graph.edges);
    output->denominator.resize(graph.edges);
    output->routed_h_prime = make_matrix(graph.edges, projected.cols);
    output->product_u_h_prime = make_matrix(graph.edges, projected.cols);
    output->product_alpha_h_prime_field.rows = graph.edges;
    output->product_alpha_h_prime_field.cols = projected.cols;
    output->product_alpha_h_prime_field.values.assign(static_cast<std::size_t>(graph.edges * projected.cols), FieldValue{});
    output->aggregate_pre = make_matrix(graph.nodes, projected.cols);
    output->aggregate_pre_field.rows = graph.nodes;
    output->aggregate_pre_field.cols = projected.cols;
    output->aggregate_pre_field.values.assign(static_cast<std::size_t>(graph.nodes * projected.cols), FieldValue{});
    output->consistency_residual_field.rows = graph.nodes;
    output->consistency_residual_field.cols = projected.cols;
    output->consistency_residual_field.values.assign(static_cast<std::size_t>(graph.nodes * projected.cols), FieldValue{});
    output->aggregate_num = make_matrix(graph.nodes, projected.cols);
    output->elu_input = make_matrix(graph.nodes, projected.cols);
    output->aggregate = make_matrix(graph.nodes, projected.cols);
    for (u64 edge = 0; edge < graph.edges; ++edge) {
        output->leaky_relu[edge] = table_lookup(tables.lrelu, tables.lrelu_min, tables.lrelu_max, score[edge], name + ".lrelu");
        output->maximum[graph.dst[edge]] = std::max(output->maximum[graph.dst[edge]], output->leaky_relu[edge]);
    }
    std::vector<i64> denominator_by_node(graph.nodes, 0);
    for (u64 edge = 0; edge < graph.edges; ++edge) {
        const u64 node = graph.dst[edge];
        output->delta[edge] = checked_i128(static_cast<__int128>(output->maximum[node]) - output->leaky_relu[edge], name + ".delta");
        output->exp_weight[edge] = table_lookup(tables.exp, 0, static_cast<i64>(tables.exp_max), output->delta[edge], name + ".exp");
        denominator_by_node[node] = checked_i128(static_cast<__int128>(denominator_by_node[node]) + output->exp_weight[edge], name + ".denominator");
    }
    for (u64 edge = 0; edge < graph.edges; ++edge) output->denominator[edge] = denominator_by_node[graph.dst[edge]];
    if (output->denominator.empty()) fail(name + " has no edges");
    const FieldInverseCache inverse_cache = fill_field_attention(*output, graph, 0, graph.edges, denominator_by_node);

    for (u64 edge = 0; edge < graph.edges; ++edge) {
        const u64 node = graph.dst[edge];
        for (u64 dim = 0; dim < projected.cols; ++dim) {
            const i64 value = projected.at(graph.src[edge], dim);
            output->routed_h_prime.at(edge, dim) = value;
            output->product_u_h_prime.at(edge, dim) = checked_i128(static_cast<__int128>(value) * output->exp_weight[edge], name + ".u_h_prime");
            output->aggregate_num.at(node,
                                     dim) = checked_i128(static_cast<__int128>(output->aggregate_num.at(node, dim)) + output->product_u_h_prime.at(edge, dim),
                                                         name + ".aggregate_num");
            output->aggregate_pre.at(node, dim) = 0;
            output->elu_input.at(node, dim) = 0;
        }
    }

    BN_CTX *field_context = BN_CTX_new();
    if (!field_context) fail("BN_CTX_new failed for aggregate field");
    try {
        aggregate_field_products(graph, projected, denominator_by_node, output->alpha_field, output->aggregate_num,
                                output->product_alpha_h_prime_field, output->aggregate_pre_field,
                                output->consistency_residual_field, field_context, name);
    } catch (...) {
        BN_CTX_free(field_context);
        throw;
    }
    for (u64 node = 0; node < graph.nodes; ++node) {
        if (node + 1 >= graph.node_edge_ptr.size()) fail("edge pointer does not cover destination node");
        if (!inverse_cache.by_node[node]) fail("zero denominator in attention relation");
        for (u64 dim = 0; dim < projected.cols; ++dim) {
            const i64 denom = denominator_by_node[node];
            output->aggregate_pre.at(node, dim) = round_div(checked_i128(static_cast<__int128>(output->aggregate_num.at(node, dim)) * aggregate_pre_scale,
                                                                         name + ".aggregate_scaled"),
                                                            checked_i128(static_cast<__int128>(denom) * h_prime_scale, name + ".aggregate_denominator"),
                                                            name + ".aggregate_pre");
            output->elu_input.at(node, dim) = output->aggregate_pre.at(node, dim);
            output->aggregate.at(node, dim) = table_lookup(tables.elu, tables.elu_min, tables.elu_max, output->elu_input.at(node, dim), name + ".elu");
        }
    }
    BN_CTX_free(field_context);
    output->aggregate_round_div = make_round_div_witness(output->aggregate_num, denominator_by_node, aggregate_pre_scale, h_prime_scale, output->aggregate_pre,
                                                         name + ".signed_round_div");
    (void)config;
}

void aggregate_output_head(const Config &config, const Graph &graph, const StaticTables &tables, const Matrix &projected, const std::vector<i64> &score,
                           u64 h_prime_scale, u64 output_pre_scale, OutputHeadTrace *output, const std::string &name) {
    output->y_prime = projected;
    output->leaky_relu.resize(graph.edges);
    output->maximum.assign(graph.nodes, std::numeric_limits<i64>::min());
    output->delta.resize(graph.edges);
    output->exp_weight.resize(graph.edges);
    output->denominator.resize(graph.edges);
    output->routed_y_prime = make_matrix(graph.edges, projected.cols);
    output->product_u_y_prime = make_matrix(graph.edges, projected.cols);
    output->product_alpha_y_prime_field.rows = graph.edges;
    output->product_alpha_y_prime_field.cols = projected.cols;
    output->product_alpha_y_prime_field.values.assign(static_cast<std::size_t>(graph.edges * projected.cols), FieldValue{});
    output->aggregate_pre = make_matrix(graph.nodes, projected.cols);
    output->aggregate_pre_field.rows = graph.nodes;
    output->aggregate_pre_field.cols = projected.cols;
    output->aggregate_pre_field.values.assign(static_cast<std::size_t>(graph.nodes * projected.cols), FieldValue{});
    output->consistency_residual_field.rows = graph.nodes;
    output->consistency_residual_field.cols = projected.cols;
    output->consistency_residual_field.values.assign(static_cast<std::size_t>(graph.nodes * projected.cols), FieldValue{});
    output->aggregate_num = make_matrix(graph.nodes, projected.cols);
    output->y_quantized = make_matrix(graph.nodes, projected.cols);
    output->aligned = make_matrix(graph.nodes, projected.cols);
    output->bias.assign(projected.cols, 0);
    output->output = make_matrix(graph.nodes, projected.cols);
    for (u64 edge = 0; edge < graph.edges; ++edge) {
        output->leaky_relu[edge] = table_lookup(tables.lrelu, tables.lrelu_min, tables.lrelu_max, score[edge], name + ".lrelu");
        output->maximum[graph.dst[edge]] = std::max(output->maximum[graph.dst[edge]], output->leaky_relu[edge]);
    }
    std::vector<i64> denominator_by_node(graph.nodes, 0);
    for (u64 edge = 0; edge < graph.edges; ++edge) {
        output->delta[edge] = checked_i128(static_cast<__int128>(output->maximum[graph.dst[edge]]) - output->leaky_relu[edge], name + ".delta");
        output->exp_weight[edge] = table_lookup(tables.exp, 0, static_cast<i64>(tables.exp_max), output->delta[edge], name + ".exp");
        denominator_by_node[graph.dst[edge]] = checked_i128(static_cast<__int128>(denominator_by_node[graph.dst[edge]]) + output->exp_weight[edge],
                                                            name + ".denominator");
    }
    for (u64 edge = 0; edge < graph.edges; ++edge) output->denominator[edge] = denominator_by_node[graph.dst[edge]];
    const FieldInverseCache inverse_cache = fill_field_attention(*output, graph, 0, graph.edges, denominator_by_node);
    for (u64 edge = 0; edge < graph.edges; ++edge) {
        const u64 node = graph.dst[edge];
        for (u64 dim = 0; dim < projected.cols; ++dim) {
            const i64 value = projected.at(graph.src[edge], dim);
            output->routed_y_prime.at(edge, dim) = value;
            output->product_u_y_prime.at(edge, dim) = checked_i128(static_cast<__int128>(value) * output->exp_weight[edge], name + ".u_y_prime");
            output->aggregate_num.at(node,
                                     dim) = checked_i128(static_cast<__int128>(output->aggregate_num.at(node, dim)) + output->product_u_y_prime.at(edge, dim),
                                                         name + ".aggregate_num");
        }
    }

    BN_CTX *field_context = BN_CTX_new();
    if (!field_context) fail("BN_CTX_new failed for output aggregate field");
    try {
        aggregate_field_products(graph, projected, denominator_by_node, output->alpha_field, output->aggregate_num,
                                output->product_alpha_y_prime_field, output->aggregate_pre_field,
                                output->consistency_residual_field, field_context, name);
    } catch (...) {
        BN_CTX_free(field_context);
        throw;
    }
    for (u64 node = 0; node < graph.nodes; ++node) {
        if (node + 1 >= graph.node_edge_ptr.size()) fail("edge pointer does not cover output destination node");
        if (!inverse_cache.by_node[node]) fail("zero denominator in attention relation");
        for (u64 dim = 0; dim < projected.cols; ++dim) {
            output->aggregate_pre.at(node, dim) = round_div(checked_i128(static_cast<__int128>(output->aggregate_num.at(node, dim)) * output_pre_scale,
                                                                         name + ".aggregate_scaled"),
                                                            checked_i128(static_cast<__int128>(denominator_by_node[node]) * h_prime_scale,
                                                                         name + ".aggregate_denominator"),
                                                            name + ".aggregate_pre");
            output->y_quantized.at(node, dim) = output->aggregate_pre.at(node, dim);
            output->aligned.at(node, dim) = output->y_quantized.at(node, dim);
        }
    }
    BN_CTX_free(field_context);
    std::vector<i64> denominators(graph.nodes, 0);
    for (u64 node = 0; node < graph.nodes; ++node) denominators[node] = denominator_by_node[node];
    output->aggregate_round_div = make_round_div_witness(output->aggregate_num, denominators, output_pre_scale, h_prime_scale, output->aggregate_pre,
                                                         name + ".signed_round_div");
    (void)config;
}

GatTrace run_forward(const Config &config, const Graph &graph, const ModelBundle &model, const StaticTables &tables, bool gpu_projection) {
    GatTrace trace;
    // These vectors are retained for the whole forward.  Reserving their
    // authenticated architecture counts makes the live-shape profile below
    // match the actual vector storage instead of an unspecified growth path.
    trace.hidden_layers.reserve(config.architecture.hidden_dims.size());
    trace.output_heads.reserve(static_cast<std::size_t>(config.architecture.output_heads));
    const Matrix *current = &graph.feature_values;
    for (u64 layer = 0; layer < config.architecture.hidden_dims.size(); ++layer) {
        HiddenLayerTrace layer_trace;
        layer_trace.heads.reserve(static_cast<std::size_t>(config.architecture.hidden_heads.at(layer)));
        const Matrix &current_input = *current;
        layer_trace.input = current_input;
        const u64 head_count = config.architecture.hidden_heads.at(layer);
        const u64 head_dim = config.architecture.hidden_dims.at(layer);
        const u64 in_dim = current_input.cols;
        for (u64 head = 0; head < head_count; ++head) {
            const std::string prefix = "hidden/layer" + std::to_string(layer) + "/head" + std::to_string(head);
            const Matrix weight = quantized_tensor_matrix(model, prefix + "/seq/kernel", config.quantization.hidden_weight.at(layer).at(head));
            const Matrix projected = gpu_projection ? matmul_gpu(current_input, weight, prefix + ".projection") : matmul_cpu(current_input, weight, prefix + ".projection");
            const Matrix src_kernel = quantized_tensor_matrix(model, prefix + "/attn_src/kernel", config.quantization.hidden_attention_src.at(layer).at(head));
            const Matrix dst_kernel = quantized_tensor_matrix(model, prefix + "/attn_dst/kernel", config.quantization.hidden_attention_dst.at(layer).at(head));
            HiddenHeadTrace head_trace;
            attention_scores(projected, src_kernel, dst_kernel, graph, &head_trace.e_src, &head_trace.e_dst, &head_trace.score, prefix + ".attention");
            aggregate_head(config, graph, tables, projected, head_trace.score, config.quantization.hidden_h_prime.at(layer).at(head),
                           config.quantization.hidden_aggregate_pre.at(layer), &head_trace, prefix);
            layer_trace.heads.push_back(std::move(head_trace));
        }
        layer_trace.concat = concat_head_aggregates(layer_trace.heads, graph.nodes, head_dim);
        if (layer_trace.concat.cols != head_count * head_dim || in_dim == 0) fail("hidden layer shape invariant failed");
        trace.hidden_layers.push_back(std::move(layer_trace));
        current = &trace.hidden_layers.back().concat;
    }
    for (u64 head = 0; head < config.architecture.output_heads; ++head) {
        const std::string prefix = "output/head" + std::to_string(head);
        const Matrix weight = quantized_tensor_matrix(model, prefix + "/seq/kernel", config.quantization.output_weight.at(head));
        const Matrix projected = gpu_projection ? matmul_gpu(*current, weight, prefix + ".projection") : matmul_cpu(*current, weight, prefix + ".projection");
        const Matrix src_kernel = quantized_tensor_matrix(model, prefix + "/attn_src/kernel", config.quantization.output_attention_src.at(head));
        const Matrix dst_kernel = quantized_tensor_matrix(model, prefix + "/attn_dst/kernel", config.quantization.output_attention_dst.at(head));
        OutputHeadTrace head_trace;
        attention_scores(projected, src_kernel, dst_kernel, graph, &head_trace.e_src, &head_trace.e_dst, &head_trace.score, prefix + ".attention");
        aggregate_output_head(config, graph, tables, projected, head_trace.score, config.quantization.output_y_prime.at(head), config.quantization.output_pre,
                              &head_trace, prefix);
        const auto bias = quantized_tensor_vector(model, prefix + "/output_bias", config.quantization.output_bias.at(head));
        if (bias.size() != projected.cols || bias.size() != config.architecture.classes) fail(prefix + " bias shape mismatch");
        head_trace.bias = bias;
        for (u64 node = 0; node < graph.nodes; ++node) {
            for (u64 cls = 0; cls < projected.cols; ++cls) {
                head_trace.output.at(node, cls) = checked_i128(static_cast<__int128>(head_trace.aligned.at(node, cls)) + bias[cls], prefix + ".bias");
            }
        }
        head_trace.alignment_round_div = make_round_div_witness(head_trace.y_quantized,
                                                                std::vector<i64>(graph.nodes, static_cast<i64>(config.quantization.output_pre)),
                                                                config.quantization.output_head, 1, head_trace.aligned, prefix + ".output_scale_alignment");
        trace.output_heads.push_back(std::move(head_trace));
    }
    if (trace.output_heads.empty()) fail("no output heads");
    trace.output_sum = make_matrix(graph.nodes, config.architecture.classes);
    for (const auto &head : trace.output_heads) {
        if (head.output.cols != trace.output_sum.cols) fail("output head class shape mismatch");
        for (std::size_t index = 0; index < trace.output_sum.values.size(); ++index) {
            trace.output_sum.values[index] = checked_i128(static_cast<__int128>(trace.output_sum.values[index]) + head.output.values[index], "output head sum");
        }
    }
    trace.logits = make_matrix(graph.nodes, config.architecture.classes);
    for (std::size_t index = 0; index < trace.logits.values.size(); ++index) {
        trace.logits.values[index] = round_div(trace.output_sum.values[index], static_cast<i64>(trace.output_heads.size()), "final average");
    }
    trace.final_average_round_div = make_round_div_witness(trace.output_sum, std::vector<i64>(graph.nodes, static_cast<i64>(trace.output_heads.size())), 1, 1,
                                                           trace.logits, "final_average");
    return trace;
}

std::string json_string_field(const std::string &text, const std::string &key) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) fail("manifest field missing: " + key);
    return match[1].str();
}

std::string json_string_field_or(const std::string &text, const std::string &key, const std::string &fallback) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    return std::regex_search(text, match, pattern) ? match[1].str() : fallback;
}

u64 json_u64_field(const std::string &text, const std::string &key) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) fail("manifest field missing: " + key);
    return parse_u64(match[1].str(), "manifest " + key);
}

bool is_digest(const std::string &value) { return std::regex_match(value, std::regex("[0-9a-fA-F]{64}")); }

std::map<std::string, std::vector<u64>> manifest_tensor_shapes(const std::string &manifest) {
    try { return checkpoint_manifest_json::tensor_shapes(manifest); }
    catch (const std::exception &error) { fail(error.what()); }
}

void expect_manifest(const std::string &manifest, const Config &config) {
    if (json_string_field(manifest, "family_schema") != "multi_layer_multi_head") { fail("checkpoint family_schema is not canonical"); }
    if (json_string_field(manifest, "dataset") != config.dataset) fail("checkpoint dataset mismatch");
    if (json_string_field(manifest, "model_arch_id") != config.model_arch_id) fail("checkpoint model_arch_id mismatch");
    if (json_string_field(manifest, "model_param_id") != config.model_param_id) fail("checkpoint model_param_id mismatch");
    if (json_string_field(manifest, "quant_cfg_id") != config.quant_cfg_id) fail("checkpoint quant_cfg_id mismatch");
    if (json_string_field(manifest, "static_table_id") != config.static_table_id) fail("checkpoint static_table_id mismatch");
    if (json_string_field(manifest, "input_preprocess_id") != config.input_preprocess_id) fail("checkpoint input_preprocess_id mismatch");
    if (json_string_field(manifest, "degree_bound_id") != config.degree_bound_id) fail("checkpoint degree_bound_id mismatch");
    if (json_string_field(manifest, "edge_sort_rule") != config.edge_sort_rule) fail("checkpoint edge_sort_rule mismatch");
    if (json_string_field(manifest, "batching_rule") != config.batching_rule) fail("checkpoint batching_rule mismatch");
    const auto checkpoint_prefix = json_string_field(manifest, "checkpoint_prefix");
    const auto expected_prefix = std::filesystem::relative(config.checkpoint_bundle / "best_model.pt", config.baseline_root).generic_string();
    if (checkpoint_prefix != expected_prefix) fail("checkpoint_prefix is not baseline-local canonical path");
    if (json_u64_field(manifest, "L") != config.architecture.layers) fail("checkpoint L mismatch");
    if (json_u64_field(manifest, "K_out") != config.architecture.output_heads) fail("checkpoint K_out mismatch");
    if (json_u64_field(manifest, "C") != config.architecture.classes) fail("checkpoint class count mismatch");
    if (json_u64_field(manifest, "tensor_count") == 0) fail("checkpoint tensor_count is zero");
}

struct RawEdge {
    u64 graph = 0;
    u64 src = 0;
    u64 dst = 0;
    u64 ordinal = 0;
};

std::vector<float> load_npy_float32(const std::filesystem::path &path, u64 expected_rows, u64 expected_cols) {
    std::ifstream input(path, std::ios::binary);
    if (!input) fail("cannot open numpy feature file " + path.string());
    std::array<unsigned char, 8> preamble{};
    input.read(reinterpret_cast<char *>(preamble.data()), static_cast<std::streamsize>(preamble.size()));
    if (input.gcount() != static_cast<std::streamsize>(preamble.size()) || preamble[0] != 0x93 || preamble[1] != 'N' || preamble[2] != 'U' ||
        preamble[3] != 'M' || preamble[4] != 'P' || preamble[5] != 'Y') {
        fail("invalid numpy magic in " + path.string());
    }
    const unsigned version_major = preamble[6];
    const unsigned version_minor = preamble[7];
    if (version_minor != 0 || (version_major != 1 && version_major != 2)) {
        fail("unsupported numpy format version in " + path.string());
    }
    const std::size_t length_size = version_major == 1 ? 2 : 4;
    std::array<unsigned char, 4> length_bytes{};
    input.read(reinterpret_cast<char *>(length_bytes.data()), static_cast<std::streamsize>(length_size));
    if (input.gcount() != static_cast<std::streamsize>(length_size)) fail("truncated numpy header length");
    std::uint32_t header_size = 0;
    for (std::size_t i = 0; i < length_size; ++i) header_size |= static_cast<std::uint32_t>(length_bytes[i]) << (8U * i);
    if (header_size == 0 || header_size > 65536U) fail("numpy header size is outside the supported bound");
    std::string header(header_size, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (input.gcount() != static_cast<std::streamsize>(header.size())) fail("truncated numpy header");
    const auto key_occurrences = [&header](const std::string &key) {
        std::size_t count = 0U, position = 0U;
        while ((position = header.find(key, position)) != std::string::npos) { ++count; position += key.size(); }
        return count;
    };
    if (key_occurrences("'descr'") + key_occurrences("\"descr\"") != 1U ||
        key_occurrences("'fortran_order'") + key_occurrences("\"fortran_order\"") != 1U ||
        key_occurrences("'shape'") + key_occurrences("\"shape\"") != 1U) {
        fail("numpy feature header contains duplicate/missing formal keys");
    }
    if (header.find("'descr': '<f4'") == std::string::npos && header.find("\"descr\": \"<f4\"") == std::string::npos)
        fail("numpy feature dtype is not little-endian float32");
    if (header.find("fortran_order': False") == std::string::npos && header.find("fortran_order\": False") == std::string::npos) {
        fail("numpy feature array must be C-contiguous");
    }
    const auto shape_pos = header.find("shape");
    if (shape_pos == std::string::npos) fail("numpy shape missing");
    const auto left = header.find('(', shape_pos);
    const auto right = header.find(')', left);
    if (left == std::string::npos || right == std::string::npos) fail("numpy shape malformed");
    std::string shape = header.substr(left + 1, right - left - 1);
    std::stringstream shape_stream(shape);
    std::string row_text, col_text;
    if (!std::getline(shape_stream, row_text, ',') || !std::getline(shape_stream, col_text, ',')) fail("numpy rank is not two");
    const u64 rows = parse_decimal_u64(trim(row_text), "numpy rows");
    const u64 cols = parse_decimal_u64(trim(col_text), "numpy cols");
    if (rows != expected_rows || cols != expected_cols) fail("numpy feature shape mismatch");
    if (rows != 0U && cols > static_cast<u64>(std::numeric_limits<std::size_t>::max()) / rows) fail("numpy feature element count overflows address space");
    const u64 value_count = rows * cols;
    if (value_count > static_cast<u64>(std::numeric_limits<std::size_t>::max() / sizeof(float)) ||
        value_count > static_cast<u64>(std::numeric_limits<std::streamsize>::max() / static_cast<std::streamsize>(sizeof(float)))) {
        fail("numpy feature payload exceeds native read bounds");
    }
    std::vector<float> values(static_cast<std::size_t>(value_count));
    input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (input.gcount() != static_cast<std::streamsize>(values.size() * sizeof(float))) fail("truncated numpy feature payload");
    if (input.peek() != std::char_traits<char>::eof()) fail("numpy feature payload has trailing bytes");
    return values;
}

std::string authenticated_asset_digest(const Config &config, const std::filesystem::path &path) {
    const auto assets = load_asset_manifest(config);
    const auto relative = std::filesystem::relative(path, config.baseline_root).generic_string();
    for (const auto &file : assets.files) {
        if (file.path == relative) return file.sha256;
    }
    fail("asset manifest does not authenticate relation resource: " + relative);
}

void require_authentication_file(const Config &config, const std::filesystem::path &path, const std::string &label) {
    if (path.empty() || std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path)) {
        fail("BLOCKED: authenticated " + label + " is missing or not a regular baseline-local file: " + path.string());
    }
    (void)authenticated_asset_digest(config, path);
}

void validate_field_bound(i64 value, const std::string &label) {
    if (value < 0) fail("negative integer bound in " + label);
    BN_CTX *context = BN_CTX_new();
    if (!context) fail("BN_CTX_new failed for " + label);
    BigNum half;
    if (!BN_copy(half.get(), scalar_modulus()) || !BN_sub_word(half.get(), 1) || !BN_rshift1(half.get(), half.get())) {
        BN_CTX_free(context);
        fail("failed to construct field half for " + label);
    }
    BigNum candidate = field_from_i64(value, context);
    const bool fits = BN_cmp(candidate.get(), half.get()) < 0;
    BN_CTX_free(context);
    if (!fits) fail("integer bound is not strictly below backend field half in " + label);
}

void validate_static_tables(const Config &config, StaticTables *tables) {
    if (!tables || !tables->authenticated) fail("BLOCKED: static table descriptor is not authenticated");
    if (tables->quant_cfg_id != config.quant_cfg_id || tables->scale_id.empty() || tables->domain_id.empty() ||
        tables->row_order != "ascending_index_min_to_max" || tables->padding_rule != "next_power_of_two_zero_v1" || tables->generator_id.empty() ||
        tables->version.empty()) {
        fail("authenticated static table descriptor identity mismatch");
    }
    const auto lrelu_rows = checked_domain_rows(tables->lrelu_min, tables->lrelu_max, "LeakyReLU");
    const auto elu_rows = checked_domain_rows(tables->elu_min, tables->elu_max, "ELU");
    if (tables->exp_max >= kMaxStaticTableRows) { fail("BLOCKED: ExpMap table domain is too large"); }
    const auto exp_rows = tables->exp_max + 1;
    const auto next_power_of_two = [](std::uint64_t value) {
        if (value == 0U) return std::uint64_t{0U};
        std::uint64_t result = 1U;
        while (result < value) {
            if (result > std::numeric_limits<std::uint64_t>::max() / 2U) fail("authenticated static table padded domain overflows");
            result <<= 1U;
        }
        return result;
    };
    if (tables->lrelu_padded_domain != next_power_of_two(lrelu_rows) || tables->elu_padded_domain != next_power_of_two(elu_rows) ||
        tables->exp_padded_domain != next_power_of_two(exp_rows)) {
        fail("authenticated static table padded domain is not canonical");
    }
    if (tables->lrelu.size() != lrelu_rows || tables->elu.size() != elu_rows || tables->exp.size() != exp_rows || tables->lrelu.empty() ||
        tables->elu.empty() || tables->exp.empty()) {
        fail("authenticated static table row count does not match its descriptor domain");
    }
    for (std::size_t index = 0; index < tables->lrelu.size(); ++index) {
        const i64 input = tables->lrelu_min + static_cast<i64>(index);
        const i64 output = tables->lrelu[index];
        if ((input >= 0 && output != input) || (input < 0 && (output < input || output > 0))) {
            fail("authenticated LeakyReLU table violates monotone canonical range");
        }
        if (index != 0 && output < tables->lrelu[index - 1]) fail("authenticated LeakyReLU table is not monotone");
    }
    for (std::size_t index = 0; index < tables->elu.size(); ++index) {
        const i64 input = tables->elu_min + static_cast<i64>(index);
        const i64 output = tables->elu[index];
        if ((input >= 0 && output != input) || (input < 0 && output >= 0)) { fail("authenticated ELU table violates canonical range"); }
        if (index != 0 && output < tables->elu[index - 1]) fail("authenticated ELU table is not monotone");
    }
    for (std::size_t index = 0; index < tables->exp.size(); ++index) {
        if (tables->exp[index] <= 0 || (index != 0 && tables->exp[index] > tables->exp[index - 1])) {
            fail("authenticated ExpMap table is not positive/non-increasing");
        }
    }
    const auto expected_digest = digest_tables(*tables);
    if (tables->digest != expected_digest) fail("authenticated static table digest mismatch");
    std::ostringstream spec;
    spec << config.static_table_id << "\n"
         << tables->generator_id << "\n"
         << tables->version << "\n"
         << tables->scale_id << "\n"
         << tables->domain_id << "\n"
         << tables->row_order << "\n"
         << tables->padding_rule << "\n"
         << config.range_bits << "\n"
         << tables->lrelu_min << "\n"
         << tables->lrelu_max << "\n"
         << tables->exp_max << "\n"
         << tables->lrelu_padded_domain << "\n"
         << tables->elu_padded_domain << "\n"
         << tables->exp_padded_domain << "\n";
    if (tables->spec_hash != sha256_text(spec.str())) fail("authenticated static table specification hash mismatch");
}

StaticTables load_static_table_descriptor(const Config &config) {
    require_authentication_file(config, config.static_table_descriptor_path, "static table descriptor");
    const auto text = file_text(config.static_table_descriptor_path);
    if (trim(text).empty() || trim(text).front() != '{') { fail("BLOCKED: static table descriptor must use the canonical JSON resource format"); }
    StaticTables tables;
    if (json_scalar(text, "schema") != "zkDL_gat_static_table_descriptor_v1") { fail("static table descriptor schema mismatch"); }
    tables.quant_cfg_id = json_scalar(text, "quant_cfg_id");
    tables.generator_id = json_scalar(text, "generator_id");
    tables.version = json_scalar(text, "version");
    tables.scale_id = json_scalar(text, "scale_id");
    tables.domain_id = json_scalar(text, "domain_id");
    tables.row_order = json_scalar(text, "row_order");
    tables.padding_rule = json_scalar(text, "padding_rule");
    tables.spec_hash = json_scalar(text, "spec_hash");
    tables.digest = json_scalar(text, "digest");
    if (json_scalar(text, "static_table_id") != config.static_table_id) fail("static table descriptor id mismatch");
    tables.lrelu_min = json_i64_value(text, "lrelu_min");
    tables.lrelu_max = json_i64_value(text, "lrelu_max");
    tables.elu_min = json_i64_value(text, "elu_min");
    tables.elu_max = json_i64_value(text, "elu_max");
    tables.exp_max = json_u64_value(text, "exp_max");
    tables.lrelu_padded_domain = json_u64_value(text, "lrelu_padded_domain");
    tables.elu_padded_domain = json_u64_value(text, "elu_padded_domain");
    tables.exp_padded_domain = json_u64_value(text, "exp_padded_domain");
    tables.lrelu = json_i64_array(text, "lrelu");
    tables.elu = json_i64_array(text, "elu");
    tables.exp = json_i64_array(text, "exp");
    tables.lrelu_rows = tables.lrelu.size();
    tables.elu_rows = tables.elu.size();
    tables.exp_rows = tables.exp.size();
    tables.lrelu_value_min = tables.lrelu.front();
    tables.lrelu_value_max = tables.lrelu.back();
    tables.elu_value_min = tables.elu.front();
    tables.elu_value_max = tables.elu.back();
    tables.exp_value_min = tables.exp.back();
    tables.exp_value_max = tables.exp.front();
    tables.metadata_only = false;
    tables.descriptor_digest = authenticated_asset_digest(config, config.static_table_descriptor_path);
    require_authentication_file(config, config.sorted_lookup_descriptor_path, "sorted lookup descriptor");
    const auto lookup_text = file_text(config.sorted_lookup_descriptor_path);
    if (json_scalar(lookup_text, "schema") != "zkDL_gat_sorted_lookup_descriptor_v1" || json_u64_value(lookup_text, "version") != 1U ||
        json_scalar(lookup_text, "key_encoding") != "packed_pair_u128" || json_scalar(lookup_text, "source_root") != "baseline-local") {
        fail("sorted lookup descriptor schema/encoding identity mismatch");
    }
    if (lookup_text.find("/chi/") != std::string::npos || lookup_text.find("expanded-chi") != std::string::npos) {
        fail("BLOCKED: sorted lookup descriptor retains a legacy expanded-chi dependency");
    }
    if (json_scalar(lookup_text, "static_table_digest") != tables.descriptor_digest) {
        fail("sorted lookup descriptor is not bound to the authenticated static table file");
    }
    tables.sorted_lookup_descriptor = std::filesystem::relative(config.sorted_lookup_descriptor_path, config.baseline_root).generic_string();
    tables.sorted_lookup_descriptor_digest = authenticated_asset_digest(config, config.sorted_lookup_descriptor_path);
    tables.authenticated = true;
    validate_static_tables(config, &tables);
    return tables;
}

StaticTables load_static_table_descriptor_shape(const Config &config) {
    require_authentication_file(config, config.static_table_descriptor_path,
                                "static table descriptor");
    const auto parsed = JsonShapeReader(config.static_table_descriptor_path)
                            .read_static_table_descriptor();
    const auto scalar = [&](const std::string &key) -> const std::string & {
        const auto found = parsed.scalars.find(key);
        if (found == parsed.scalars.end())
            fail("analysis-only static table descriptor scalar is missing: " + key);
        return found->second;
    };
    const std::set<std::string> expected_arrays = {"elu", "exp", "lrelu"};
    if (parsed.arrays != expected_arrays)
        fail("analysis-only static table descriptor arrays do not match the canonical schema");

    StaticTables tables;
    tables.quant_cfg_id = scalar("quant_cfg_id");
    tables.generator_id = scalar("generator_id");
    tables.version = scalar("version");
    tables.scale_id = scalar("scale_id");
    tables.domain_id = scalar("domain_id");
    tables.row_order = scalar("row_order");
    tables.padding_rule = scalar("padding_rule");
    tables.spec_hash = scalar("spec_hash");
    tables.digest = scalar("digest");
    tables.lrelu_min = parse_i64(scalar("lrelu_min"), "static table lrelu_min");
    tables.lrelu_max = parse_i64(scalar("lrelu_max"), "static table lrelu_max");
    tables.elu_min = parse_i64(scalar("elu_min"), "static table elu_min");
    tables.elu_max = parse_i64(scalar("elu_max"), "static table elu_max");
    tables.exp_max = parse_decimal_u64(scalar("exp_max"), "static table exp_max");
    tables.lrelu_padded_domain = parse_decimal_u64(scalar("lrelu_padded_domain"), "lrelu padded domain");
    tables.elu_padded_domain = parse_decimal_u64(scalar("elu_padded_domain"), "elu padded domain");
    tables.exp_padded_domain = parse_decimal_u64(scalar("exp_padded_domain"), "exp padded domain");

    const std::set<std::string> expected_scalars = {
        "digest", "domain_id", "elu_max", "elu_min", "elu_padded_domain", "exp_max",
        "exp_padded_domain", "generator_id", "lrelu_max", "lrelu_min",
        "lrelu_padded_domain", "padding_rule", "quant_cfg_id", "row_order", "scale_id",
        "schema", "spec_hash", "static_table_id", "version"};
    if (parsed.scalars.size() != expected_scalars.size())
        fail("analysis-only static table descriptor has unknown metadata fields");
    for (const auto &key : expected_scalars)
        if (parsed.scalars.find(key) == parsed.scalars.end())
            fail("analysis-only static table descriptor scalar is missing: " + key);
    if (scalar("schema") != "zkDL_gat_static_table_descriptor_v1" ||
        scalar("static_table_id") != config.static_table_id ||
        tables.quant_cfg_id != config.quant_cfg_id ||
        tables.generator_id != "zkDL_baseline_local_static_table_generator_v1" ||
        tables.version != "v1" || tables.row_order != "ascending_index_min_to_max" ||
        tables.padding_rule != "next_power_of_two_zero_v1" || tables.scale_id.empty() ||
        tables.domain_id != "range_bits:" + std::to_string(config.range_bits) ||
        !is_digest(tables.spec_hash) || !is_digest(tables.digest))
        fail("analysis-only static table descriptor identity is unsupported or unauthenticated");

    const u64 lrelu_rows = checked_domain_rows(tables.lrelu_min, tables.lrelu_max, "LeakyReLU");
    const u64 elu_rows = checked_domain_rows(tables.elu_min, tables.elu_max, "ELU");
    if (tables.exp_max >= kMaxStaticTableRows)
        fail("BLOCKED: analysis-only ExpMap table domain is too large");
    const u64 exp_rows = tables.exp_max + 1U;
    if (parsed.array_element_counts.at("lrelu") != lrelu_rows ||
        parsed.array_element_counts.at("elu") != elu_rows ||
        parsed.array_element_counts.at("exp") != exp_rows)
        fail("analysis-only static table streamed element counts do not match canonical row counts");
    if (config.range_bits == 0U || config.range_bits > 25U ||
        config.range_bits >= 63U)
        fail("BLOCKED: analysis-only static table range_bits is outside the canonical generator bound");
    const i64 input_bound = static_cast<i64>(1ULL << config.range_bits);
    const i64 expected_lrelu_min = -input_bound;
    const i64 expected_exp_max = checked_i128(static_cast<__int128>(input_bound) -
        static_cast<__int128>(round_div(-input_bound, 5, "analysis-only LReLU output minimum")),
        "analysis-only ExpMap input maximum");
    const auto next_power_of_two = [](u64 value) {
        if (value == 0U || value > kMaxStaticTableRows)
            fail("analysis-only static table row count is out of bounds");
        u64 padded = 1U;
        while (padded < value) padded <<= 1U;
        return padded;
    };
    if (tables.lrelu_min != expected_lrelu_min || tables.lrelu_max != input_bound ||
        tables.elu_min != expected_lrelu_min || tables.elu_max != input_bound ||
        tables.exp_max != static_cast<u64>(expected_exp_max) ||
        lrelu_rows > kMaxStaticTableRows || elu_rows > kMaxStaticTableRows ||
        exp_rows == 0U || exp_rows > kMaxStaticTableRows ||
        tables.lrelu_padded_domain != next_power_of_two(lrelu_rows) ||
        tables.elu_padded_domain != next_power_of_two(elu_rows) ||
        tables.exp_padded_domain != next_power_of_two(exp_rows))
        fail("analysis-only static table dimensions do not match the canonical generator shape");

    std::ostringstream spec;
    spec << config.static_table_id << "\n"
         << tables.generator_id << "\n"
         << tables.version << "\n"
         << tables.scale_id << "\n"
         << tables.domain_id << "\n"
         << tables.row_order << "\n"
         << tables.padding_rule << "\n"
         << config.range_bits << "\n"
         << tables.lrelu_min << "\n"
         << tables.lrelu_max << "\n"
         << tables.exp_max << "\n"
         << tables.lrelu_padded_domain << "\n"
         << tables.elu_padded_domain << "\n"
         << tables.exp_padded_domain << "\n";
    if (tables.spec_hash != sha256_text(spec.str()))
        fail("analysis-only static table specification hash mismatch");

    tables.lrelu_rows = lrelu_rows;
    tables.elu_rows = elu_rows;
    tables.exp_rows = exp_rows;
    // The input-domain endpoints safely contain the monotone LeakyReLU and
    // ELU outputs.  The authenticated generator bounds ExpMap outputs to
    // [1, 16]. These intervals only overestimate lookup key widths.
    tables.lrelu_value_min = tables.lrelu_min;
    tables.lrelu_value_max = tables.lrelu_max;
    tables.elu_value_min = tables.elu_min;
    tables.elu_value_max = tables.elu_max;
    tables.exp_value_min = 1;
    tables.exp_value_max = 16;
    tables.metadata_only = true;
    tables.descriptor_digest = authenticated_asset_digest(config,
                                                           config.static_table_descriptor_path);
    if (sha256_file(config.static_table_descriptor_path) != tables.descriptor_digest)
        fail("analysis-only static table descriptor differs from its authenticated asset digest");

    require_authentication_file(config, config.sorted_lookup_descriptor_path,
                                "sorted lookup descriptor");
    const auto lookup_text = file_text(config.sorted_lookup_descriptor_path);
    if (json_scalar(lookup_text, "schema") != "zkDL_gat_sorted_lookup_descriptor_v1" ||
        json_u64_value(lookup_text, "version") != 1U ||
        json_scalar(lookup_text, "key_encoding") != "packed_pair_u128" ||
        json_scalar(lookup_text, "source_root") != "baseline-local" ||
        lookup_text.find("/chi/") != std::string::npos ||
        lookup_text.find("expanded-chi") != std::string::npos ||
        json_scalar(lookup_text, "static_table_digest") != tables.descriptor_digest)
        fail("analysis-only sorted lookup descriptor identity is unsupported");
    tables.sorted_lookup_descriptor =
        std::filesystem::relative(config.sorted_lookup_descriptor_path, config.baseline_root).generic_string();
    tables.sorted_lookup_descriptor_digest =
        authenticated_asset_digest(config, config.sorted_lookup_descriptor_path);
    if (sha256_file(config.sorted_lookup_descriptor_path) != tables.sorted_lookup_descriptor_digest)
        fail("analysis-only sorted lookup descriptor differs from its authenticated asset digest");
    tables.authenticated = true;
    return tables;
}

void validate_ledger_identity(const Config &config, const StaticTables &tables, const IntegerBoundLedger &ledger) {
    if (!ledger.authenticated || ledger.schema != "zkDL_gat_integer_bound_ledger_v1" ||
        ledger.ledger_id.empty() || ledger.degree_bound_id.empty() ||
        ledger.quant_cfg_id != config.quant_cfg_id || ledger.static_table_id != config.static_table_id || ledger.field_modulus != scalar_modulus_decimal() ||
        ledger.scale_chain_digest != config.scale_chain_digest || ledger.static_table_descriptor_digest != tables.descriptor_digest) {
        fail("authenticated integer-bound ledger identity mismatch");
    }
    if (config.degree_bound_id == "auto" || config.degree_bound_id != ledger.degree_bound_id) {
        fail("BLOCKED: secure config does not bind the ledger's authenticated degree bound id");
    }
    for (const auto value : {ledger.max_score_abs, ledger.max_table_output_abs, ledger.max_delta, ledger.max_exp, ledger.max_sum, ledger.max_aggregate_num,
                             ledger.max_dividend_abs, ledger.max_aux_abs, ledger.max_divisor}) {
        validate_field_bound(value, "ledger global bound");
    }
    if (ledger.max_divisor <= 0 || ledger.families.empty() || ledger.families.size() > kMaxLedgerFamilies) {
        fail("BLOCKED: integer-bound ledger has an unbounded family registry");
    }
    const u64 hidden_layers = config.architecture.hidden_heads.size();
    if (config.architecture.output_heads > kMaxLedgerFamilies) { fail("BLOCKED: architecture has an unbounded output family registry"); }
    u64 expected_families = config.architecture.output_heads;
    for (const auto heads : config.architecture.hidden_heads) {
        if (heads > kMaxLedgerFamilies || expected_families > kMaxLedgerFamilies - heads) {
            fail("BLOCKED: architecture has an unbounded integer-bound family registry");
        }
        expected_families += heads;
    }
    if (ledger.families.size() != expected_families) { fail("integer-bound ledger family count does not match canonical architecture"); }
    std::set<std::string> family_ids;
    for (std::size_t index = 0; index < ledger.families.size(); ++index) {
        const auto &family = ledger.families[index];
        const bool identity_in_range = family.output ? family.layer == hidden_layers && family.head < config.architecture.output_heads :
                                                       family.layer < hidden_layers && family.head < config.architecture.hidden_heads.at(family.layer);
        if (!identity_in_range ||
            !family_ids.insert(std::to_string(family.layer) + ":" + std::to_string(family.head) + ":" + (family.output ? "out" : "hidden")).second) {
            fail("integer-bound ledger contains an out-of-range or duplicate family identity");
        }
        for (const auto value : {family.max_input_abs, family.max_weight_abs, family.max_attention_abs, family.max_bias_abs, family.max_score_abs,
                                 family.max_table_output_abs, family.max_delta, family.max_exp, family.max_sum, family.max_aggregate_num,
                                 family.max_dividend_abs, family.max_aux_abs, family.max_divisor, family.max_field_product_abs}) {
            validate_field_bound(value, "ledger family bound " + std::to_string(index));
        }
        if (family.max_divisor <= 0 || family.max_score_abs > ledger.max_score_abs || family.max_table_output_abs > ledger.max_table_output_abs ||
            family.max_delta > ledger.max_delta || family.max_exp > ledger.max_exp || family.max_sum > ledger.max_sum ||
            family.max_aggregate_num > ledger.max_aggregate_num || family.max_dividend_abs > ledger.max_dividend_abs ||
            family.max_aux_abs > ledger.max_aux_abs || family.max_divisor > ledger.max_divisor) {
            fail("integer-bound ledger family exceeds its global bound");
        }
    }
    if (ledger.digest != digest_ledger(ledger)) fail("authenticated integer-bound ledger digest mismatch");
}

IntegerBoundLedger load_integer_bound_ledger(const Config &config, const StaticTables &tables) {
    require_authentication_file(config, config.integer_bound_ledger_path, "integer-bound ledger");
    const auto text = file_text(config.integer_bound_ledger_path);
    if (trim(text).empty() || trim(text).front() != '{') { fail("BLOCKED: integer-bound ledger must use the canonical JSON resource format"); }
    IntegerBoundLedger ledger;
    try { ledger = integer_bound_ledger::parse(text); }
    catch (const std::exception &error) { fail(error.what()); }
    ledger.authenticated = true;
    ledger.descriptor_digest = authenticated_asset_digest(config, config.integer_bound_ledger_path);
    validate_ledger_identity(config, tables, ledger);
    return ledger;
}

const IntegerBoundFamily &ledger_family(const IntegerBoundLedger &ledger, u64 layer, u64 head, bool output) {
    const IntegerBoundFamily *found = nullptr;
    for (const auto &family : ledger.families) {
        if (family.layer == layer && family.head == head && family.output == output) {
            if (found) fail("integer-bound ledger contains duplicate family identity");
            found = &family;
        }
    }
    if (!found) fail("integer-bound ledger is missing a required layer/head family");
    return *found;
}

i64 max_abs_matrix(const Matrix &matrix, const std::string &name) {
    __int128 maximum = 0;
    for (const auto value : matrix.values) {
        const __int128 absolute = value < 0 ? -static_cast<__int128>(value) : static_cast<__int128>(value);
        if (absolute > maximum) maximum = absolute;
    }
    return checked_i128(maximum, name);
}

i64 max_abs_quantized_tensor(const ModelBundle &model, const std::string &name, u64 scale) {
    const auto values = quantized_tensor_vector(model, name, scale);
    __int128 maximum = 0;
    for (const auto value : values) {
        const __int128 absolute = value < 0 ? -static_cast<__int128>(value) : static_cast<__int128>(value);
        if (absolute > maximum) maximum = absolute;
    }
    return checked_i128(maximum, name + ".bound");
}

i64 ceil_positive_div(i64 numerator, i64 denominator, const std::string &name) {
    if (numerator < 0 || denominator <= 0) fail("invalid positive division bound in " + name);
    return checked_i128((static_cast<__int128>(numerator) + denominator - 1) / denominator, name);
}

void require_ledger_bound(const IntegerBoundFamily &family, i64 required, i64 registered, const std::string &name) {
    if (required < 0 || registered < 0 || required > registered) { fail("BLOCKED: integer-bound ledger admission failed for " + name); }
    validate_field_bound(required, name);
}

void require_round_div_no_wrap(i64 dividend_abs, i64 divisor, i64 quotient_abs, const IntegerBoundFamily &family, const std::string &name) {
    if (dividend_abs < 0 || divisor <= 0 || quotient_abs < 0) fail("invalid SignedRoundDiv bound in " + name);
    const i64 two_a_plus_d = checked_i128(static_cast<__int128>(2) * dividend_abs + divisor, name + ".2a_plus_d");
    const i64 two_dq_plus_r = checked_i128(static_cast<__int128>(2) * divisor * quotient_abs + (2 * divisor - 1), name + ".2dq_plus_r");
    const i64 two_d_minus_one = checked_i128(static_cast<__int128>(2) * divisor - 1, name + ".2d_minus_one");
    for (const auto value : {dividend_abs, divisor, quotient_abs, two_a_plus_d, two_dq_plus_r, two_d_minus_one}) {
        validate_field_bound(value, name + ".field_fit");
    }
    const i64 required_aux = std::max({dividend_abs, divisor, quotient_abs, two_a_plus_d, two_dq_plus_r, two_d_minus_one});
    require_ledger_bound(family, required_aux, family.max_aux_abs, name + ".aux");
}

void admit_integer_bound_ledger(const Config &config, const Graph &graph, const ModelBundle &model, const StaticTables &tables,
                                const IntegerBoundLedger &ledger) {
    if (!tables.authenticated || !ledger.authenticated) fail("BLOCKED: relation resources are not authenticated");
    u64 max_degree = 0;
    u64 min_degree = std::numeric_limits<u64>::max();
    for (u64 node = 0; node < graph.nodes; ++node) {
        const u64 degree = graph.node_edge_ptr.at(node + 1) - graph.node_edge_ptr.at(node);
        max_degree = std::max(max_degree, degree);
        min_degree = std::min(min_degree, degree);
    }
    if (max_degree == 0 || min_degree == 0) fail("BLOCKED: graph has an empty destination group for ledger admission");
    const i64 input_bound = max_abs_matrix(graph.feature_values, "input_bound");
    i64 current_bound = input_bound;
    const auto table_output_bound = [&](const std::vector<i64> &values, i64 minimum, i64 maximum, i64 input_abs, const std::string &name) {
        if (input_abs < 0 || -static_cast<__int128>(input_abs) < minimum || input_abs > maximum) {
            fail("BLOCKED: " + name + " input bound is outside its authenticated table domain");
        }
        const u64 first = static_cast<u64>(static_cast<__int128>(-input_abs) - minimum);
        const u64 last = static_cast<u64>(static_cast<__int128>(input_abs) - minimum);
        if (last >= values.size() || first > last) fail("invalid authenticated table bound in " + name);
        const auto abs_value = [](i64 value) { return value < 0 ? -static_cast<__int128>(value) : static_cast<__int128>(value); };

        const __int128 maximum_abs = std::max(abs_value(values.at(static_cast<std::size_t>(first))), abs_value(values.at(static_cast<std::size_t>(last))));
        return checked_i128(maximum_abs, name + ".output_bound");
    };
    const auto require_global = [&](i64 required, i64 registered, const std::string &name) {
        if (required < 0 || registered < 0 || required > registered) fail("BLOCKED: global ledger admission failed for " + name);
        validate_field_bound(required, name);
    };
    require_global(input_bound, ledger.max_score_abs, "input_bound");
    for (u64 layer = 0; layer < config.architecture.hidden_dims.size(); ++layer) {
        const u64 in_dim = layer == 0 ? graph.features : config.architecture.hidden_heads[layer - 1] * config.architecture.hidden_dims[layer - 1];
        const u64 head_dim = config.architecture.hidden_dims[layer];
        i64 next_bound = 0;
        for (u64 head = 0; head < config.architecture.hidden_heads[layer]; ++head) {
            const auto &family = ledger_family(ledger, layer, head, false);
            const std::string prefix = "hidden/layer" + std::to_string(layer) + "/head" + std::to_string(head);
            const i64 weight_bound = max_abs_quantized_tensor(model, prefix + "/seq/kernel", config.quantization.hidden_weight.at(layer).at(head));
            const i64 attention_src_bound = max_abs_quantized_tensor(model, prefix + "/attn_src/kernel",
                                                                     config.quantization.hidden_attention_src.at(layer).at(head));
            const i64 attention_dst_bound = max_abs_quantized_tensor(model, prefix + "/attn_dst/kernel",
                                                                     config.quantization.hidden_attention_dst.at(layer).at(head));
            const i64 h_prime_bound = checked_i128(static_cast<__int128>(in_dim) * current_bound * weight_bound, prefix + ".h_prime_bound");
            const i64 score_src_bound = checked_i128(static_cast<__int128>(head_dim) * h_prime_bound * attention_src_bound, prefix + ".score_src_bound");
            const i64 score_dst_bound = checked_i128(static_cast<__int128>(head_dim) * h_prime_bound * attention_dst_bound, prefix + ".score_dst_bound");
            const i64 score_bound = checked_i128(static_cast<__int128>(score_src_bound) + score_dst_bound, prefix + ".score_bound");
            const i64 table_bound = table_output_bound(tables.lrelu, tables.lrelu_min, tables.lrelu_max, score_bound, prefix + ".lrelu");
            i64 lrelu_output_min = 0, lrelu_output_max = 0;
            if (!static_table_bounds::symmetric_output_interval(tables.lrelu, tables.lrelu_min, tables.lrelu_max, score_bound,
                                                               lrelu_output_min, lrelu_output_max))
                fail("BLOCKED: authenticated LReLU output span cannot be derived for " + prefix);
            const i64 delta_bound = checked_i128(static_cast<__int128>(lrelu_output_max) - lrelu_output_min, prefix + ".delta_bound");
            if (!static_table_bounds::exp_domain_covers_delta(delta_bound, tables.exp_max))
                fail("BLOCKED: ExpMap table domain admission failed for " + prefix + " required_delta_span=" + std::to_string(delta_bound) +
                     " exp_max=" + std::to_string(tables.exp_max));
            const i64 exp_bound = tables.exp.front();
            const i64 sum_bound = checked_i128(static_cast<__int128>(max_degree) * exp_bound, prefix + ".sum_bound");
            const i64 aggregate_num_bound = checked_i128(static_cast<__int128>(sum_bound) * h_prime_bound, prefix + ".aggregate_num_bound");
            const i64 dividend_bound = checked_i128(static_cast<__int128>(config.quantization.hidden_aggregate_pre.at(layer)) * aggregate_num_bound,
                                                    prefix + ".dividend_bound");
            const i64 divisor_bound = checked_i128(static_cast<__int128>(config.quantization.hidden_h_prime.at(layer).at(head)) * sum_bound,
                                                   prefix + ".divisor_bound");
            const i64 min_divisor = static_cast<i64>(config.quantization.hidden_h_prime.at(layer).at(head));
            const i64 quotient_bound = ceil_positive_div(dividend_bound, min_divisor, prefix + ".quotient_bound");
            const i64 field_product_bound = checked_i128(static_cast<__int128>(exp_bound) * h_prime_bound, prefix + ".field_product_bound");
            require_ledger_bound(family, current_bound, family.max_input_abs, prefix + ".input");
            require_ledger_bound(family, weight_bound, family.max_weight_abs, prefix + ".weight");
            require_ledger_bound(family, std::max(attention_src_bound, attention_dst_bound), family.max_attention_abs, prefix + ".attention");
            require_ledger_bound(family, score_bound, family.max_score_abs, prefix + ".score");
            require_ledger_bound(family, table_bound, family.max_table_output_abs, prefix + ".table_output");
            require_ledger_bound(family, delta_bound, family.max_delta, prefix + ".delta");
            require_ledger_bound(family, exp_bound, family.max_exp, prefix + ".exp");
            require_ledger_bound(family, sum_bound, family.max_sum, prefix + ".sum");
            require_ledger_bound(family, aggregate_num_bound, family.max_aggregate_num, prefix + ".aggregate_num");
            require_ledger_bound(family, dividend_bound, family.max_dividend_abs, prefix + ".dividend");
            require_ledger_bound(family, divisor_bound, family.max_divisor, prefix + ".divisor");
            require_ledger_bound(family, field_product_bound, family.max_field_product_abs, prefix + ".field_product");
            require_round_div_no_wrap(dividend_bound, min_divisor, quotient_bound, family, prefix + ".signed_round_div");
            i64 elu_query_bound = 0;
            if (!static_table_bounds::positive_weighted_round_abs_bound(
                    tables.exp, min_degree, h_prime_bound, config.quantization.hidden_aggregate_pre.at(layer),
                    config.quantization.hidden_h_prime.at(layer).at(head), elu_query_bound))
                fail("BLOCKED: cannot derive a positive-weighted aggregate bound for " + prefix + ".elu");
            validate_field_bound(elu_query_bound, prefix + ".elu_query_bound");
            const i64 elu_bound = table_output_bound(tables.elu, tables.elu_min, tables.elu_max, elu_query_bound, prefix + ".elu");
            next_bound = std::max(next_bound, elu_bound);
            for (const auto value : {score_bound, delta_bound, sum_bound, aggregate_num_bound, dividend_bound, divisor_bound, field_product_bound})
                require_global(value, ledger.max_dividend_abs, prefix + ".global");
        }
        current_bound = next_bound;
    }
    const u64 output_layer = config.architecture.hidden_dims.size();
    const u64 output_in_dim = config.architecture.hidden_heads.back() * config.architecture.hidden_dims.back();
    i64 output_head_bound = 0;
    for (u64 head = 0; head < config.architecture.output_heads; ++head) {
        const auto &family = ledger_family(ledger, output_layer, head, true);
        const std::string prefix = "output/head" + std::to_string(head);
        const i64 weight_bound = max_abs_quantized_tensor(model, prefix + "/seq/kernel", config.quantization.output_weight.at(head));
        const i64 attention_src_bound = max_abs_quantized_tensor(model, prefix + "/attn_src/kernel", config.quantization.output_attention_src.at(head));
        const i64 attention_dst_bound = max_abs_quantized_tensor(model, prefix + "/attn_dst/kernel", config.quantization.output_attention_dst.at(head));
        const i64 bias_bound = max_abs_quantized_tensor(model, prefix + "/output_bias", config.quantization.output_bias.at(head));
        const i64 y_prime_bound = checked_i128(static_cast<__int128>(output_in_dim) * current_bound * weight_bound, prefix + ".y_prime_bound");
        const i64 score_src_bound = checked_i128(static_cast<__int128>(config.architecture.classes) * y_prime_bound * attention_src_bound,
                                                 prefix + ".score_src_bound");
        const i64 score_dst_bound = checked_i128(static_cast<__int128>(config.architecture.classes) * y_prime_bound * attention_dst_bound,
                                                 prefix + ".score_dst_bound");
        const i64 score_bound = checked_i128(static_cast<__int128>(score_src_bound) + score_dst_bound, prefix + ".score_bound");
        const i64 table_bound = table_output_bound(tables.lrelu, tables.lrelu_min, tables.lrelu_max, score_bound, prefix + ".lrelu");
        i64 lrelu_output_min = 0, lrelu_output_max = 0;
        if (!static_table_bounds::symmetric_output_interval(tables.lrelu, tables.lrelu_min, tables.lrelu_max, score_bound,
                                                           lrelu_output_min, lrelu_output_max))
            fail("BLOCKED: authenticated LReLU output span cannot be derived for " + prefix);
        const i64 delta_bound = checked_i128(static_cast<__int128>(lrelu_output_max) - lrelu_output_min, prefix + ".delta_bound");
        if (!static_table_bounds::exp_domain_covers_delta(delta_bound, tables.exp_max))
            fail("BLOCKED: ExpMap table domain admission failed for " + prefix + " required_delta_span=" + std::to_string(delta_bound) +
                 " exp_max=" + std::to_string(tables.exp_max));
        const i64 exp_bound = tables.exp.front();
        const i64 sum_bound = checked_i128(static_cast<__int128>(max_degree) * exp_bound, prefix + ".sum_bound");
        const i64 aggregate_num_bound = checked_i128(static_cast<__int128>(sum_bound) * y_prime_bound, prefix + ".aggregate_num_bound");
        const i64 dividend_bound = checked_i128(static_cast<__int128>(config.quantization.output_pre) * aggregate_num_bound, prefix + ".dividend_bound");
        const i64 divisor_bound = checked_i128(static_cast<__int128>(config.quantization.output_y_prime.at(head)) * sum_bound, prefix + ".divisor_bound");
        const i64 quotient_bound = ceil_positive_div(dividend_bound, static_cast<i64>(config.quantization.output_y_prime.at(head)), prefix + ".quotient_bound");
        const i64 aligned_bound = ceil_positive_div(static_cast<i64>(config.quantization.output_head) * quotient_bound,
                                                    static_cast<i64>(config.quantization.output_pre), prefix + ".aligned_bound");
        const i64 head_bound = checked_i128(static_cast<__int128>(aligned_bound) + bias_bound, prefix + ".head_bound");
        const i64 field_product_bound = checked_i128(static_cast<__int128>(exp_bound) * y_prime_bound, prefix + ".field_product_bound");
        require_ledger_bound(family, current_bound, family.max_input_abs, prefix + ".input");
        require_ledger_bound(family, weight_bound, family.max_weight_abs, prefix + ".weight");
        require_ledger_bound(family, std::max(attention_src_bound, attention_dst_bound), family.max_attention_abs, prefix + ".attention");
        require_ledger_bound(family, bias_bound, family.max_bias_abs, prefix + ".bias");
        require_ledger_bound(family, score_bound, family.max_score_abs, prefix + ".score");
        require_ledger_bound(family, table_bound, family.max_table_output_abs, prefix + ".table_output");
        require_ledger_bound(family, delta_bound, family.max_delta, prefix + ".delta");
        require_ledger_bound(family, exp_bound, family.max_exp, prefix + ".exp");
        require_ledger_bound(family, sum_bound, family.max_sum, prefix + ".sum");
        require_ledger_bound(family, aggregate_num_bound, family.max_aggregate_num, prefix + ".aggregate_num");
        require_ledger_bound(family,
                             std::max({dividend_bound, static_cast<i64>(config.quantization.output_head) * quotient_bound,
                                       static_cast<i64>(config.architecture.output_heads) * head_bound}),
                             family.max_dividend_abs, prefix + ".dividend");
        require_ledger_bound(family, divisor_bound, family.max_divisor, prefix + ".divisor");
        require_ledger_bound(family, field_product_bound, family.max_field_product_abs, prefix + ".field_product");
        require_round_div_no_wrap(dividend_bound, static_cast<i64>(config.quantization.output_y_prime.at(head)), quotient_bound, family,
                                  prefix + ".signed_round_div");
        require_round_div_no_wrap(static_cast<i64>(config.quantization.output_head) * quotient_bound, static_cast<i64>(config.quantization.output_pre),
                                  aligned_bound, family, prefix + ".output_alignment");
        output_head_bound = std::max(output_head_bound, head_bound);
    }
    const i64 output_sum_bound = checked_i128(static_cast<__int128>(config.architecture.output_heads) * output_head_bound, "final_average.sum_bound");
    require_global(output_sum_bound, ledger.max_dividend_abs, "final_average.sum");
    const i64 final_quotient_bound = ceil_positive_div(output_sum_bound, static_cast<i64>(config.architecture.output_heads), "final_average.quotient_bound");
    require_global(final_quotient_bound, ledger.max_dividend_abs, "final_average.quotient");
    const auto &final_family = ledger_family(ledger, output_layer, 0, true);
    require_round_div_no_wrap(output_sum_bound, static_cast<i64>(config.architecture.output_heads), final_quotient_bound, final_family, "final_average");
}

}

AssetManifest load_asset_manifest(const Config &config) {
    if (config.asset_manifest_path.empty()) fail("asset manifest path is empty");
    if (std::filesystem::is_symlink(config.asset_manifest_path) || !std::filesystem::is_regular_file(config.asset_manifest_path)) {
        fail("baseline-local asset_manifest is missing or not a regular file: " + config.asset_manifest_path.string());
    }
    const auto manifest = file_text(config.asset_manifest_path);
    AssetManifest result;
    result.schema = json_string_field(manifest, "schema");
    if (result.schema != "zkDL_gat_asset_manifest_v1") fail("asset manifest schema mismatch");
    result.dataset = json_string_field(manifest, "dataset");
    if (result.dataset != config.dataset) fail("asset manifest dataset mismatch");
    result.quant_cfg_id = json_string_field(manifest, "quant_cfg_id");
    if (result.quant_cfg_id != config.quant_cfg_id) fail("asset manifest quant_cfg_id mismatch");
    result.config = json_string_field(manifest, "config");
    result.scale_chain_digest = json_string_field(manifest, "scale_chain_digest");
    result.static_table_descriptor = json_string_field(manifest, "static_table_descriptor");
    result.static_table_descriptor_digest = json_string_field(manifest, "static_table_descriptor_digest");
    result.sorted_lookup_descriptor = json_string_field(manifest, "sorted_lookup_descriptor");
    result.sorted_lookup_descriptor_digest = json_string_field(manifest, "sorted_lookup_descriptor_digest");
    result.integer_bound_ledger = json_string_field(manifest, "integer_bound_ledger");
    result.integer_bound_ledger_descriptor_digest = json_string_field(manifest, "integer_bound_ledger_descriptor_digest");
    result.srs_path = json_string_field_or(manifest, "srs_path", "UNAVAILABLE");
    result.srs_manifest_digest = json_string_field_or(manifest, "srs_manifest_digest", "UNAVAILABLE");
    result.model_registry_path = json_string_field_or(manifest, "model_registry_path", "UNAVAILABLE");
    result.model_registry_digest = json_string_field_or(manifest, "model_registry_digest", "UNAVAILABLE");
    result.model_blinder_sidecar_path = json_string_field_or(manifest, "model_blinder_sidecar_path", "UNAVAILABLE");
    result.model_blinder_sidecar_digest = json_string_field_or(manifest, "model_blinder_sidecar_digest", "UNAVAILABLE");
    if (!is_digest(result.scale_chain_digest) || !is_digest(result.static_table_descriptor_digest) || !is_digest(result.sorted_lookup_descriptor_digest) ||
        !is_digest(result.integer_bound_ledger_descriptor_digest) || result.scale_chain_digest != config.scale_chain_digest ||
        result.static_table_descriptor != std::filesystem::relative(config.static_table_descriptor_path, config.baseline_root).generic_string() ||
        result.sorted_lookup_descriptor != std::filesystem::relative(config.sorted_lookup_descriptor_path, config.baseline_root).generic_string() ||
        result.integer_bound_ledger != std::filesystem::relative(config.integer_bound_ledger_path, config.baseline_root).generic_string()) {
        fail("asset manifest relation-resource identity mismatch");
    }
    const auto expected_config = std::filesystem::relative(config.config_path, config.baseline_root).generic_string();
    if (result.config != expected_config) fail("asset manifest config identity mismatch");
    const auto expected_srs_path = std::filesystem::relative(std::filesystem::path(config.srs_path), config.baseline_root).generic_string();
    const auto expected_registry_path = std::filesystem::relative(config.model_registry_path, config.baseline_root).generic_string();
    if (result.srs_path != expected_srs_path || result.srs_manifest_digest != config.srs_manifest_digest ||
        result.model_registry_path != expected_registry_path || result.model_registry_digest != config.verifier_registry_digest ||
        result.model_blinder_sidecar_path != "UNAVAILABLE" || result.model_blinder_sidecar_digest != "UNAVAILABLE") {
        fail("asset manifest authenticated native resource identity mismatch");
    }
    result.digest = sha256_file(config.asset_manifest_path);

    const std::regex entry("\\{\\s*\"bytes\"\\s*:\\s*([0-9]+)\\s*,\\s*\"path\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"sha256\"\\s*:\\s*\"([0-9a-fA-F]{64})\"\\s*\\}");
    std::set<std::string> paths;
    const std::string checkpoint_root = std::filesystem::relative(config.checkpoint_bundle, config.baseline_root).generic_string();
    const std::string checkpoint_prefix = checkpoint_root.empty() ? std::string() : checkpoint_root + "/";
    for (std::sregex_iterator iterator(manifest.begin(), manifest.end(), entry), end; iterator != end; ++iterator) {
        AssetManifestEntry file;
        file.bytes = parse_u64((*iterator)[1].str(), "asset manifest bytes");
        file.path = (*iterator)[2].str();
        file.sha256 = (*iterator)[3].str();
        if (file.path.empty() || std::filesystem::path(file.path).is_absolute()) { fail("asset manifest file path is not baseline-local: " + file.path); }
        for (const auto &component : std::filesystem::path(file.path)) {
            if (component == "..") fail("asset manifest file path escapes baseline root: " + file.path);
        }
        if (!paths.insert(file.path).second) fail("duplicate asset manifest file: " + file.path);
        if (file.path.rfind("artifacts/model_blinders/", 0U) == 0U || file.path.find("/model_blinders/") != std::string::npos || file.path.find("/.private_native/") != std::string::npos || std::filesystem::path(file.path).filename() == "private_asset_manifest.json" || (!checkpoint_prefix.empty() && file.path.rfind(checkpoint_prefix, 0U) == 0U && file.path != checkpoint_prefix + "manifest.json")) fail("public asset manifest contains private prover material");
        const auto resolved = local_path(config.baseline_root, file.path, "asset manifest file");
        if (std::filesystem::is_symlink(resolved) || !std::filesystem::is_regular_file(resolved)) {
            fail("asset manifest file is missing or symlinked: " + file.path);
        }
        if (std::filesystem::file_size(resolved) != file.bytes) fail("asset manifest byte size mismatch: " + file.path);
        if (sha256_file(resolved) != file.sha256) fail("asset manifest digest mismatch: " + file.path);
        result.files.push_back(std::move(file));
    }
    if (result.files.empty()) fail("asset manifest has no files");
    const auto require_entry = [&](const std::string &relative) {
        if (!paths.count(relative)) fail("asset manifest missing required file: " + relative);
    };
    require_entry(result.config);
    require_entry("configs/quantization_scale_chain.cfg");
    require_entry((std::filesystem::relative(config.checkpoint_bundle, config.baseline_root) / "manifest.json").generic_string());
    if (!is_digest(json_string_field(manifest, "checkpoint_tensors_txt_digest")) || !is_digest(json_string_field(manifest, "checkpoint_tensors_npz_digest"))) fail("public asset manifest checkpoint identities are not SHA-256 digests");
    require_entry((std::filesystem::relative(config.cache_root / config.dataset, config.baseline_root) / "meta.cfg").generic_string());
    require_entry(std::filesystem::relative(config.static_table_descriptor_path, config.baseline_root).generic_string());
    require_entry(std::filesystem::relative(config.sorted_lookup_descriptor_path, config.baseline_root).generic_string());
    require_entry(std::filesystem::relative(config.integer_bound_ledger_path, config.baseline_root).generic_string());
    const auto manifest_digest_for = [&](const std::string &relative) -> const std::string & {
        for (const auto &file : result.files)
            if (file.path == relative) return file.sha256;
        fail("asset manifest relation resource entry is missing: " + relative);
    };
    if (manifest_digest_for(result.static_table_descriptor) != result.static_table_descriptor_digest ||
        manifest_digest_for(result.sorted_lookup_descriptor) != result.sorted_lookup_descriptor_digest ||
        manifest_digest_for(result.integer_bound_ledger) != result.integer_bound_ledger_descriptor_digest ||
        manifest_digest_for("configs/quantization_scale_chain.cfg") != result.scale_chain_digest) {
        fail("asset manifest relation-resource digest mismatch");
    }
    return result;
}

const char *backend_name(Backend backend) noexcept { return backend == Backend::Cpu ? "cpu" : "gpu"; }

Backend parse_backend(const std::string &value) {
    if (value == "cpu") return Backend::Cpu;
    if (value == "gpu") return Backend::Gpu;
    fail("backend must be cpu or gpu");
}

Config load_config(const std::filesystem::path &path, bool admission_only,
                   bool allow_analysis_only_planned) {
    const auto absolute = std::filesystem::absolute(path).lexically_normal();
    if (absolute.parent_path().filename() != "configs") fail("config must be in baseline-local configs/");
    const auto values = read_kv(absolute);
    Config config;
    config.config_path = absolute;
    config.baseline_root = absolute.parent_path().parent_path();
    const std::set<std::string> canonical_keys = {"dataset",
                                                  "data_root",
                                                  "cache_root",
                                                  "export_dir",
                                                  "checkpoint_bundle",
                                                  "task_type",
                                                  "report_unit",
                                                  "batching_rule",
                                                  "self_loop_rule",
                                                  "edge_sort_rule",
                                                  "chunking_rule",
                                                  "L",
                                                  "hidden_profile",
                                                  "d_in_profile",
                                                  "K_out",
                                                  "num_classes",
                                                  "range_bits",
                                                  "seed",
                                                  "dump_trace",
                                                  "auto_prepare_dataset",
                                                  "model_arch_id",
                                                  "model_param_id",
                                                  "quant_cfg_id",
                                                  "input_preprocess_id",
                                                  "static_table_id",
                                                  "degree_bound_id",
                                                  "symmetrize_edges",
                                                  "deduplicate_edges",
                                                  "batch_graphs",
                                                  "srs_type",
                                                  "srs_id",
                                                  "srs_path",
                                                  "srs_manifest_digest",
                                                  "model_registry_path",
                                                  "model_blinder_sidecar_path",
                                                  "model_blinder_sidecar_digest",
                                                  "privacy_policy",
                                                  "input_privacy",
                                                  "output_mode",
                                                  "opening_mode",
                                                  "binding_mode",
                                                  "proof_task",
                                                  "security_profile_id",
                                                  "verifier_registry_digest",
                                                  "static_table_descriptor",
                                                  "sorted_lookup_descriptor",
                                                  "integer_bound_ledger"};
    for (const auto &item : values) {
        if (!canonical_keys.count(item.first)) fail("unknown non-canonical configuration key: " + item.first);
    }
    const auto required = [&](const std::string &key) -> std::string {
        const auto iterator = values.find(key);
        if (iterator == values.end()) fail("configuration key missing: " + key);
        return iterator->second;
    };
    config.dataset = required("dataset");
    static const std::set<std::string> datasets = {"cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv"};
    if (!datasets.count(config.dataset)) fail("dataset is outside canonical five-dataset set");
    config.data_root = local_path(config.baseline_root, required("data_root"), "data_root");
    config.cache_root = local_path(config.baseline_root, required("cache_root"), "cache_root");
    config.checkpoint_bundle = local_path(config.baseline_root, required("checkpoint_bundle"), "checkpoint_bundle");
    config.run_root = local_path(config.baseline_root, required("export_dir"), "export_dir");
    config.asset_manifest_path = config.run_root / "asset_manifest.json";
    config.static_table_descriptor_path = local_path(config.baseline_root, required("static_table_descriptor"), "static_table_descriptor");
    config.sorted_lookup_descriptor_path = local_path(config.baseline_root, required("sorted_lookup_descriptor"), "sorted_lookup_descriptor");
    config.integer_bound_ledger_path = local_path(config.baseline_root, required("integer_bound_ledger"), "integer_bound_ledger");
    config.model_arch_id = required("model_arch_id");
    config.model_param_id = required("model_param_id");
    config.quant_cfg_id = required("quant_cfg_id");
    config.static_table_id = required("static_table_id");
    config.input_preprocess_id = required("input_preprocess_id");
    config.proof_task = required("proof_task");
    config.privacy_policy = required("privacy_policy");
    config.input_privacy = required("input_privacy");
    config.output_mode = required("output_mode");
    config.batching_rule = required("batching_rule");
    config.self_loop_rule = required("self_loop_rule");
    config.edge_sort_rule = required("edge_sort_rule");
    config.task_type = required("task_type");
    config.report_unit = required("report_unit");
    config.chunking_rule = required("chunking_rule");
    config.degree_bound_id = required("degree_bound_id");
    config.srs_type = required("srs_type");
    config.srs_id = required("srs_id");
    config.srs_path = local_path(config.baseline_root, required("srs_path"), "srs_path").string();
    config.srs_manifest_digest = required("srs_manifest_digest");
    if (config.srs_manifest_digest != "UNAVAILABLE" && !is_digest(config.srs_manifest_digest) &&
        !(admission_only && allow_analysis_only_planned && config.srs_manifest_digest == "CEREMONY_REQUIRED"))
        fail("srs_manifest_digest must be SHA-256 or explicit UNAVAILABLE");
    config.model_registry_path = local_path(config.baseline_root, required("model_registry_path"), "model_registry_path");
    config.model_blinder_sidecar_path = local_path(config.baseline_root, required("model_blinder_sidecar_path"), "model_blinder_sidecar_path");
    config.model_blinder_sidecar_digest = required("model_blinder_sidecar_digest");
    if (config.model_blinder_sidecar_digest != "UNAVAILABLE" && !is_digest(config.model_blinder_sidecar_digest) &&
        !(admission_only && allow_analysis_only_planned && config.model_blinder_sidecar_digest == "CEREMONY_REQUIRED"))
        fail("model_blinder_sidecar_digest must be SHA-256 or explicit UNAVAILABLE");
    config.opening_mode = required("opening_mode");
    config.binding_mode = required("binding_mode");
    config.security_profile_id = required("security_profile_id");
    config.verifier_registry_digest = required("verifier_registry_digest");
    if (!is_digest(config.verifier_registry_digest) &&
        !(admission_only && config.verifier_registry_digest == "UNAVAILABLE") &&
        !(admission_only && allow_analysis_only_planned && config.verifier_registry_digest == "CEREMONY_REQUIRED"))
        fail("verifier_registry_digest must be SHA-256");
    config.range_bits = parse_u64(required("range_bits"), "range_bits");
    config.batch_graphs = values.count("batch_graphs") ? parse_u64(values.at("batch_graphs"), "batch_graphs") : 1;
    config.seed = parse_u64(required("seed"), "seed");
    config.dump_trace = parse_bool(required("dump_trace"), "dump_trace");
    config.symmetrize_edges = values.count("symmetrize_edges") && parse_bool(values.at("symmetrize_edges"), "symmetrize_edges");
    config.deduplicate_edges = !values.count("deduplicate_edges") || parse_bool(values.at("deduplicate_edges"), "deduplicate_edges");
    config.auto_prepare_dataset = parse_bool(required("auto_prepare_dataset"), "auto_prepare_dataset");
    config.architecture.layers = parse_u64(required("L"), "L");
    const auto profile = parse_profile(required("hidden_profile"));
    for (const auto item : profile) {
        config.architecture.hidden_heads.push_back(item.first);
        config.architecture.hidden_dims.push_back(item.second);
    }
    config.architecture.input_dims = parse_list(required("d_in_profile"), "d_in_profile");
    config.architecture.output_heads = parse_u64(required("K_out"), "K_out");
    config.architecture.classes = parse_u64(required("num_classes"), "num_classes");
    if (config.architecture.layers != config.architecture.hidden_dims.size() + 1 ||
        config.architecture.input_dims.size() != config.architecture.hidden_dims.size())
        fail("L/profile dimension mismatch");
    if (config.architecture.layers < 2 || config.architecture.output_heads == 0 || config.architecture.classes == 0 || config.batch_graphs == 0) {
        fail("canonical architecture/batching dimensions are invalid");
    }

    const auto quant_path = config.baseline_root / "configs" / "quantization_scale_chain.cfg";
    if (std::filesystem::is_symlink(quant_path) || !std::filesystem::is_regular_file(quant_path)) {
        fail("BLOCKED: quantization scale-chain file is missing or not a regular baseline-local file");
    }
    config.scale_chain_digest = sha256_file(quant_path);
    const auto quant = read_kv(quant_path);
    if (quant.at("schema") != "gat_l3_scale_chain" || quant.at("quant_cfg_id") != config.quant_cfg_id) { fail("quantization scale chain schema/id mismatch"); }
    const auto scale_required = [&](const std::string &key) -> const std::string & {
        const auto iterator = quant.find(key);
        if (iterator == quant.end()) fail("quantization scale-chain key missing: " + key);
        return iterator->second;
    };
    const auto parse_layer_scales = [&](const std::string &prefix) {
        std::vector<std::vector<u64>> result;
        result.reserve(config.architecture.hidden_dims.size());
        for (u64 layer = 0; layer < config.architecture.hidden_dims.size(); ++layer) {
            result.push_back(parse_list(scale_required(prefix + std::to_string(layer)), prefix + std::to_string(layer)));
        }
        return result;
    };
    config.quantization.input = parse_u64(scale_required("s_input"), "s_input");
    config.quantization.delta = parse_u64(scale_required("s_delta"), "s_delta");
    config.quantization.exp = parse_u64(scale_required("s_exp"), "s_exp");
    config.quantization.hidden_input = parse_list(scale_required("s_h"), "s_h");
    config.quantization.hidden_weight = parse_layer_scales("s_w_hidden_");
    config.quantization.hidden_h_prime = parse_layer_scales("s_h_prime_hidden_");
    config.quantization.hidden_attention_src = parse_layer_scales("s_a_src_hidden_");
    config.quantization.hidden_attention_dst = parse_layer_scales("s_a_dst_hidden_");
    config.quantization.hidden_aggregate_pre = parse_list(quant.at("s_agg_pre"), "s_agg_pre");
    config.quantization.hidden_aggregate = parse_list(quant.at("s_agg"), "s_agg");
    config.quantization.hidden_concat = parse_list(scale_required("s_cat"), "s_cat");
    config.quantization.output_weight = parse_list(quant.at("s_w_output"), "s_w_output");
    config.quantization.output_y_prime = parse_list(quant.at("s_y_prime"), "s_y_prime");
    config.quantization.output_attention_src = parse_list(quant.at("s_a_src_output"), "s_a_src_output");
    config.quantization.output_attention_dst = parse_list(quant.at("s_a_dst_output"), "s_a_dst_output");
    config.quantization.output_bias = parse_list(quant.at("s_b_output"), "s_b_output");
    config.quantization.output_pre = parse_u64(quant.at("s_out_pre"), "s_out_pre");
    config.quantization.output_head = parse_u64(quant.at("s_out_head"), "s_out_head");
    config.quantization.output = parse_u64(quant.at("s_out"), "s_out");
    config.quantization.rounding = quant.at("round");
    config.quantization.rescale = quant.at("rescale");
    config.quantization.output_average = quant.at("output_average");
    if (config.quantization.rounding != "nearest_ties_away" || config.quantization.rescale != "none" ||
        config.quantization.output_average != "signed_round_div")
        fail("unsupported non-canonical quantization chain");
    const u64 hidden_layers = config.architecture.hidden_dims.size();
    if (config.quantization.hidden_input.size() != hidden_layers + 1 || config.quantization.hidden_aggregate_pre.size() != hidden_layers ||
        config.quantization.hidden_aggregate.size() != hidden_layers || config.quantization.hidden_concat.size() != hidden_layers ||
        config.quantization.output_weight.size() != config.architecture.output_heads ||
        config.quantization.output_y_prime.size() != config.architecture.output_heads ||
        config.quantization.output_attention_src.size() != config.architecture.output_heads ||
        config.quantization.output_attention_dst.size() != config.architecture.output_heads ||
        config.quantization.output_bias.size() != config.architecture.output_heads) {
        fail("quantization scale-chain vector lengths do not match canonical architecture");
    }
    const auto positive = [](u64 value, const std::string &name) {
        if (value == 0) fail("quantization scale must be positive: " + name);
    };
    positive(config.quantization.input, "s_input");
    positive(config.quantization.delta, "s_delta");
    positive(config.quantization.exp, "s_exp");
    positive(config.quantization.output_pre, "s_out_pre");
    positive(config.quantization.output_head, "s_out_head");
    positive(config.quantization.output, "s_out");
    if (config.quantization.hidden_input.front() != config.quantization.input) { fail("s_h[0] must equal s_input"); }
    const auto product_scale = [](u64 left, u64 right, const std::string &name) {
        if (left != 0 && right > std::numeric_limits<u64>::max() / left) fail("scale-chain overflow in " + name);
        return left * right;
    };
    for (u64 layer = 0; layer < hidden_layers; ++layer) {
        const auto expected_heads = config.architecture.hidden_heads.at(layer);
        for (u64 head = 0; head < expected_heads; ++head) {
            if (config.quantization.hidden_weight.at(layer).size() != expected_heads || config.quantization.hidden_h_prime.at(layer).size() != expected_heads ||
                config.quantization.hidden_attention_src.at(layer).size() != expected_heads ||
                config.quantization.hidden_attention_dst.at(layer).size() != expected_heads) {
                fail("hidden scale-chain head count mismatch at layer " + std::to_string(layer));
            }
            const auto h_prime = product_scale(config.quantization.hidden_input.at(layer), config.quantization.hidden_weight.at(layer).at(head), "S_Hprime");
            if (h_prime != config.quantization.hidden_h_prime.at(layer).at(head) ||
                product_scale(h_prime, config.quantization.hidden_attention_src.at(layer).at(head), "S_delta_src") != config.quantization.delta ||
                product_scale(h_prime, config.quantization.hidden_attention_dst.at(layer).at(head), "S_delta_dst") != config.quantization.delta) {
                fail("hidden projection/attention scale-chain mismatch at layer " + std::to_string(layer) + " head " + std::to_string(head));
            }
            positive(config.quantization.hidden_weight.at(layer).at(head), "s_w_hidden");
            positive(config.quantization.hidden_h_prime.at(layer).at(head), "s_h_prime_hidden");
            positive(config.quantization.hidden_attention_src.at(layer).at(head), "s_a_src_hidden");
            positive(config.quantization.hidden_attention_dst.at(layer).at(head), "s_a_dst_hidden");
        }
        if (config.quantization.hidden_aggregate.at(layer) != config.quantization.hidden_concat.at(layer) ||
            config.quantization.hidden_concat.at(layer) != config.quantization.hidden_input.at(layer + 1)) {
            fail("hidden aggregate/concat/input scale-chain mismatch at layer " + std::to_string(layer));
        }
        positive(config.quantization.hidden_aggregate_pre.at(layer), "s_agg_pre");
        positive(config.quantization.hidden_aggregate.at(layer), "s_agg");
        positive(config.quantization.hidden_concat.at(layer), "s_cat");
    }
    if (config.quantization.output_pre == 0 || config.quantization.output_head == 0 || config.quantization.output != config.quantization.output_head) {
        fail("output scale-chain requires S_out == S_out_head");
    }
    const u64 output_input_scale = config.quantization.hidden_input.back();
    const u64 output_input_dim = config.architecture.hidden_heads.back() * config.architecture.hidden_dims.back();
    (void)output_input_dim;
    for (u64 head = 0; head < config.architecture.output_heads; ++head) {
        const auto y_prime = product_scale(output_input_scale, config.quantization.output_weight.at(head), "S_Yprime");
        if (y_prime != config.quantization.output_y_prime.at(head) ||
            product_scale(y_prime, config.quantization.output_attention_src.at(head), "S_delta_output_src") != config.quantization.delta ||
            product_scale(y_prime, config.quantization.output_attention_dst.at(head), "S_delta_output_dst") != config.quantization.delta ||
            config.quantization.output_bias.at(head) != config.quantization.output_head) {
            fail("output projection/attention/bias scale-chain mismatch at head " + std::to_string(head));
        }
        positive(config.quantization.output_weight.at(head), "s_w_output");
        positive(config.quantization.output_y_prime.at(head), "s_y_prime");
        positive(config.quantization.output_attention_src.at(head), "s_a_src_output");
        positive(config.quantization.output_attention_dst.at(head), "s_a_dst_output");
        positive(config.quantization.output_bias.at(head), "s_b_output");
    }

    config.config_digest = config_identity_digest(values);
    config.config_binding_digest = config_binding_identity_digest(values);
    return config;
}

ModelBundle load_model_bundle(const Config &config) {
    const auto assets = load_asset_manifest(config);
    const auto private_manifest_path = config.run_root / "private_asset_manifest.json";
    if (std::filesystem::is_symlink(private_manifest_path) || !std::filesystem::is_regular_file(private_manifest_path)) fail("private checkpoint asset manifest is missing or symlinked");
    const auto private_permissions = std::filesystem::status(private_manifest_path).permissions();
    if ((private_permissions & (std::filesystem::perms::group_all | std::filesystem::perms::others_all)) != std::filesystem::perms::none) fail("private checkpoint asset manifest must be owner-only");
    const auto private_manifest = file_text(private_manifest_path);
    if (json_string_field(private_manifest, "schema") != "zkDL_gat_private_asset_manifest_v1" || json_string_field(private_manifest, "dataset") != config.dataset || json_string_field(private_manifest, "registry_digest") != config.verifier_registry_digest) fail("private checkpoint asset manifest identity mismatch");
    const auto public_manifest = file_text(config.asset_manifest_path);
    if (!std::filesystem::is_directory(config.checkpoint_bundle)) fail("checkpoint bundle is not a directory");
    const auto manifest_path = config.checkpoint_bundle / "manifest.json";
    const auto manifest = file_text(manifest_path);
    expect_manifest(manifest, config);
    const auto manifest_shapes = manifest_tensor_shapes(manifest);
    ModelBundle model;
    model.root = config.checkpoint_bundle;
    model.dataset = config.dataset;
    model.model_arch_id = config.model_arch_id;
    model.model_param_id = config.model_param_id;
    model.quant_cfg_id = config.quant_cfg_id;
    model.static_table_id = config.static_table_id;
    model.manifest_digest = sha256_file(manifest_path);
    model.tensor_count_manifest = json_u64_field(manifest, "tensor_count");
    const auto asset_digest = [&](const std::filesystem::path &path) -> std::string {
        const auto relative = std::filesystem::relative(path, config.baseline_root).generic_string();
        const std::regex entry("\\{\\s*\"bytes\"\\s*:\\s*([0-9]+)\\s*,\\s*\"path\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"sha256\"\\s*:\\s*\"([0-9a-fA-F]{64})\"\\s*\\}");
        std::string expected;
        u64 expected_bytes = 0U;
        for (std::sregex_iterator iterator(private_manifest.begin(), private_manifest.end(), entry), end; iterator != end; ++iterator) {
            if ((*iterator)[2].str() != relative) continue;
            if (!expected.empty()) fail("duplicate private checkpoint asset: " + relative);
            expected = (*iterator)[3].str();
            expected_bytes = parse_u64((*iterator)[1].str(), "private checkpoint bytes");
        }
        if (expected.empty() || std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) != expected_bytes || sha256_file(path) != expected) fail("private asset manifest does not authenticate checkpoint file: " + relative);
        if (path.filename() == "tensors.txt" && expected != json_string_field(public_manifest, "checkpoint_tensors_txt_digest")) fail("private checkpoint text identity does not match public statement resources");
        if (path.filename() == "tensors.npz" && expected != json_string_field(public_manifest, "checkpoint_tensors_npz_digest")) fail("private checkpoint archive identity does not match public statement resources");
        return expected;
    };
    model.tensors_text_digest = asset_digest(config.checkpoint_bundle / "tensors.txt");
    model.tensors_npz_digest = asset_digest(config.checkpoint_bundle / "tensors.npz");
    const auto best_model = config.checkpoint_bundle / "best_model.pt";
    if (std::filesystem::is_regular_file(best_model)) model.best_model_digest = asset_digest(best_model);
    std::ifstream npz(config.checkpoint_bundle / "tensors.npz", std::ios::binary);
    std::array<unsigned char, 4> npz_magic{};
    npz.read(reinterpret_cast<char *>(npz_magic.data()), static_cast<std::streamsize>(npz_magic.size()));
    if (npz.gcount() != static_cast<std::streamsize>(npz_magic.size()) || npz_magic[0] != 'P' || npz_magic[1] != 'K' || npz_magic[2] != 3 ||
        npz_magic[3] != 4) {
        fail("checkpoint tensors.npz is not a valid ZIP archive");
    }
    std::ifstream input(config.checkpoint_bundle / "tensors.txt");
    if (!input) fail("cannot open checkpoint tensor export");
    std::string marker;
    while (input >> marker) {
        if (marker != "TENSOR") fail("unexpected checkpoint token " + marker);
        ModelTensor tensor;
        u64 rank = 0, count = 0;
        input >> tensor.name >> rank;
        if (!input || rank == 0 || rank > 8) fail("malformed checkpoint tensor header");
        tensor.shape.resize(rank);
        u64 product = 1;
        for (u64 &dimension : tensor.shape) {
            input >> dimension;
            if (!input || dimension == 0 || product > std::numeric_limits<u64>::max() / dimension) fail("invalid tensor shape");
            product *= dimension;
        }
        input >> count;
        if (!input || count != product) fail("checkpoint tensor count mismatch: " + tensor.name);
        tensor.values.resize(static_cast<std::size_t>(count));
        for (double &value : tensor.values) {
            input >> value;
            if (!input || !std::isfinite(value)) fail("invalid checkpoint tensor value: " + tensor.name);
        }
        if (!model.tensors.emplace(tensor.name, std::move(tensor)).second) fail("duplicate checkpoint tensor");
    }
    if (model.tensors.empty()) fail("checkpoint contains no tensors");
    if (model.tensors.size() != model.tensor_count_manifest || manifest_shapes.size() != model.tensor_count_manifest) {
        fail("checkpoint tensor_count does not match manifest tensor_index");
    }
    for (const auto &item : model.tensors) {
        const auto shape = manifest_shapes.find(item.first);
        if (shape == manifest_shapes.end()) fail("checkpoint tensor is absent from manifest tensor_index: " + item.first);
        if (shape->second != item.second.shape) fail("checkpoint tensor shape disagrees with manifest: " + item.first);
    }
    return model;
}

ModelBundle load_model_metadata(const Config &config) {
    (void)load_asset_manifest(config);
    if (!std::filesystem::is_directory(config.checkpoint_bundle)) fail("checkpoint bundle is not a directory");
    const auto manifest_path = config.checkpoint_bundle / "manifest.json";
    if (std::filesystem::is_symlink(manifest_path) || !std::filesystem::is_regular_file(manifest_path))
        fail("BLOCKED: public checkpoint manifest is missing or not a regular file");
    const auto manifest = file_text(manifest_path);
    expect_manifest(manifest, config);
    (void)manifest_tensor_shapes(manifest);

    ModelBundle model;
    model.root = config.checkpoint_bundle;
    model.dataset = config.dataset;
    model.model_arch_id = config.model_arch_id;
    model.model_param_id = config.model_param_id;
    model.quant_cfg_id = config.quant_cfg_id;
    model.static_table_id = config.static_table_id;
    model.manifest_digest = sha256_file(manifest_path);
    model.tensor_count_manifest = json_u64_field(manifest, "tensor_count");
    const auto public_manifest = file_text(config.asset_manifest_path);
    model.tensors_text_digest = json_string_field(public_manifest, "checkpoint_tensors_txt_digest");
    model.tensors_npz_digest = json_string_field(public_manifest, "checkpoint_tensors_npz_digest");
    return model;
}

Graph load_graph(const Config &config) {
    const auto directory = config.cache_root / config.dataset;
    const auto metadata = read_kv(directory / "meta.cfg");
    const auto metadata_value = [&](const std::string &key) -> std::string {
        const auto iterator = metadata.find(key);
        if (iterator == metadata.end()) fail("cache metadata key missing: " + key);
        return iterator->second;
    };
    if (metadata_value("name") != config.dataset) fail("cache dataset name mismatch");
    const u64 nodes = parse_decimal_u64(metadata_value("num_nodes"), "num_nodes");
    const u64 features = parse_decimal_u64(metadata_value("num_features"), "num_features");
    if (metadata_value("input_preprocess_id") != config.input_preprocess_id) fail("cache input_preprocess_id mismatch");
    if (metadata.count("num_classes") && parse_decimal_u64(metadata.at("num_classes"), "num_classes") != config.architecture.classes) {
        fail("cache class count mismatch");
    }
    if (metadata.count("task_type") && metadata.at("task_type") != config.task_type) fail("cache task_type mismatch");
    if (metadata.count("report_unit") && metadata.at("report_unit") != config.report_unit) fail("cache report_unit mismatch");
    if (config.architecture.input_dims.front() != features) fail("feature dimension mismatch");
    Graph graph;
    graph.total_nodes = nodes;
    graph.nodes = nodes;
    graph.features = features;
    graph.graphs = metadata.count("graph_count") ? parse_decimal_u64(metadata.at("graph_count"), "graph_count") : 1;
    if (nodes == 0U || nodes > kMaxNativeDomain || features == 0U || features > kMaxNativeDomain || graph.graphs == 0U || graph.graphs > kMaxNativeDomain) fail("graph node/feature/count exceeds native domain bounds");
    const auto padded_axis = [](u64 value, const std::string &label) {
        if (value == 0U || value > kMaxNativeDomain) fail(label + " exceeds native domain bounds");
        u64 result = 1U;
        while (result < value) result <<= 1U;
        return result;
    };
    const u64 input_domain_rows = padded_axis(nodes, "graph node count");
    const u64 input_domain_features = padded_axis(features, "graph feature count");
    if (input_domain_rows > kMaxNativeDomain / input_domain_features) fail("input feature relation domain exceeds native coordinate bound");
    if (config.dataset != "ppi" && graph.graphs != 1) { fail("non-PPI canonical graph must contain exactly one graph"); }
    graph.absolute_ids.resize(nodes);
    std::iota(graph.absolute_ids.begin(), graph.absolute_ids.end(), 0);
    graph.node_ptr = {0, nodes};
    if (config.dataset == "ppi") {
        graph.node_ptr.clear();
        std::ifstream ptr(directory / "node_ptr.txt");
        if (!ptr) fail("missing PPI node_ptr");
        std::string token;
        while (ptr >> token) graph.node_ptr.push_back(parse_decimal_u64(token, "PPI node_ptr"));
        if (graph.node_ptr.size() != graph.graphs + 1 || graph.node_ptr.empty() || graph.node_ptr.front() != 0 || graph.node_ptr.back() != nodes)
            fail("PPI node_ptr mismatch");
        for (std::size_t index = 1; index < graph.node_ptr.size(); ++index) {
            if (graph.node_ptr[index] <= graph.node_ptr[index - 1] || graph.node_ptr[index] > nodes) {
                fail("PPI node_ptr is not strictly increasing and bounded");
            }
        }
    }

    Matrix features_matrix = make_matrix(nodes, features);
    if (config.dataset == "ppi" || config.dataset == "ogbn_arxiv") {
        const auto raw = load_npy_float32(directory / "features.npy", nodes, features);
        for (u64 row = 0; row < nodes; ++row) {
            long double sum = 0;
            if (config.input_preprocess_id == "row_normalize") {
                for (u64 col = 0; col < features; ++col) sum += raw[static_cast<std::size_t>(row * features + col)];
            }
            for (u64 col = 0; col < features; ++col) {
                const double source = raw[static_cast<std::size_t>(row * features + col)];
                const double value = config.input_preprocess_id == "row_normalize" && sum != 0 ? source / static_cast<double>(sum) : source;
                features_matrix.at(row, col) = rounded_float(value, config.quantization.input, "feature");
            }
        }
    } else {
        std::ifstream input(directory / "features.txt");
        if (!input) fail("missing sparse feature file");
        std::string shape_line;
        if (!std::getline(input, shape_line)) fail("missing sparse feature shape line");
        std::stringstream shape_tokens(shape_line);
        std::string rows_text, cols_text, trailing_shape_token;
        if (!(shape_tokens >> rows_text >> cols_text) || (shape_tokens >> trailing_shape_token))
            fail("sparse feature shape line must contain exactly two decimal axes");
        const u64 rows = parse_decimal_u64(rows_text, "sparse feature rows");
        const u64 cols = parse_decimal_u64(cols_text, "sparse feature columns");
        if (rows != nodes || cols != features) fail("sparse feature shape mismatch");
        std::string line;
        for (u64 row = 0; row < nodes; ++row) {
            if (!std::getline(input, line)) fail("truncated sparse feature file");
            std::stringstream tokens(line);
            std::string token;
            long double sum = 0;
            std::vector<std::pair<u64, double>> entries;
            std::set<u64> columns;
            while (tokens >> token) {
                const auto colon = token.find(':');
                if (colon == std::string::npos) fail("malformed sparse feature token");
                const u64 column = parse_decimal_u64(token.substr(0, colon), "sparse feature index");
                if (column >= features) fail("sparse feature index out of range");
                if (!columns.insert(column).second) fail("duplicate sparse feature column");
                const std::string raw_value = token.substr(colon + 1);
                if (!canonical_decimal_float(raw_value)) fail("sparse feature value is not a complete decimal float");
                std::size_t consumed = 0U;
                const double value = std::stod(raw_value, &consumed);
                if (consumed != raw_value.size()) fail("sparse feature value has trailing characters");
                if (!std::isfinite(value)) fail("non-finite sparse feature");
                entries.emplace_back(column, value);
                sum += value;
            }
            for (const auto entry : entries) {
                const double value = config.input_preprocess_id == "row_normalize" && sum != 0 ? entry.second / static_cast<double>(sum) : entry.second;
                features_matrix.at(row, entry.first) = rounded_float(value, config.quantization.input, "feature");
            }
        }
    }
    graph.feature_values = std::move(features_matrix);

    std::ifstream edges_file(directory / "edges.txt");
    if (!edges_file) fail("missing edge file");
    std::vector<RawEdge> edges;
    std::set<std::tuple<u64, u64, u64>> seen;
    std::string line;
    u64 ordinal = 0;
    const auto append_edge = [&](u64 gid, u64 src, u64 dst) {
        if (config.deduplicate_edges && !seen.emplace(gid, src, dst).second) return false;
        if (edges.size() >= static_cast<std::size_t>(kMaxNativeDomain / 2U)) fail("normalized edge count exceeds native sparse-route domain bound");
        edges.push_back({gid, src, dst, ordinal++});
        return true;
    };
    while (std::getline(edges_file, line)) {
        line = trim(line);
        if (line.empty()) continue;
        std::stringstream stream(line);
        std::vector<u64> values;
        std::string token;
        while (stream >> token) values.push_back(parse_decimal_u64(token, "edge id/endpoint"));
        if ((config.dataset == "ppi" && values.size() != 3) || (config.dataset != "ppi" && values.size() != 2)) fail("malformed edge row");
        const u64 gid = config.dataset == "ppi" ? values[0] : 0;
        if (config.dataset == "ppi" && gid >= graph.graphs) fail("PPI edge graph id out of range");
        if (config.dataset == "ppi") {
            const u64 graph_begin = graph.node_ptr.at(gid);
            const u64 graph_end = graph.node_ptr.at(gid + 1U);
            if (values[1] < graph_begin || values[1] >= graph_end || values[2] < graph_begin || values[2] >= graph_end)
                fail("PPI global edge endpoint ownership mismatch");
        }
        const u64 source = config.dataset == "ppi" ? values[1] : values[0];
        const u64 destination = config.dataset == "ppi" ? values[2] : values[1];
        if (source >= nodes || destination >= nodes) fail("edge endpoint out of range");
        append_edge(gid, source, destination);
        if (config.symmetrize_edges) append_edge(gid, destination, source);
    }
    for (u64 node = 0; node < nodes; ++node) {
        u64 gid = 0;
        if (config.dataset == "ppi") gid = static_cast<u64>(std::upper_bound(graph.node_ptr.begin(), graph.node_ptr.end(), node) - graph.node_ptr.begin() - 1);
        append_edge(gid, node, node);
    }
    if (edges.size() > static_cast<std::size_t>(kMaxNativeDomain / 2U)) fail("normalized edge count exceeds native sparse-route domain bound");
    std::stable_sort(edges.begin(), edges.end(), [](const RawEdge &left, const RawEdge &right) {
        if (left.graph != right.graph) return left.graph < right.graph;
        if (left.dst != right.dst) return left.dst < right.dst;
        return left.ordinal < right.ordinal;
    });
    graph.src.reserve(edges.size());
    graph.dst.reserve(edges.size());
    graph.node_edge_ptr.assign(nodes + 1, 0);
    const auto checked_node_index = [nodes](u64 index) {
        if (index >= nodes || index >= kMaxNativeDomain) fail("canonical flat node index exceeds native wire bound");
        return checked_native_domain_u32(index, "canonical flat node index", true);
    };
    for (const auto &edge : edges) {
        graph.src.push_back(checked_node_index(edge.src));
        graph.dst.push_back(checked_node_index(edge.dst));
        ++graph.node_edge_ptr[edge.dst + 1];
    }
    for (u64 node = 1; node <= nodes; ++node) graph.node_edge_ptr[node] += graph.node_edge_ptr[node - 1];
    graph.edges = graph.src.size();
    graph.edge_ptr.assign(graph.graphs + 1, 0);
    for (const auto &edge : edges) ++graph.edge_ptr[edge.graph + 1];
    for (u64 gid = 1; gid <= graph.graphs; ++gid) graph.edge_ptr[gid] += graph.edge_ptr[gid - 1];
    if (graph.edge_ptr.size() != graph.graphs + 1 || graph.edge_ptr.front() != 0 || graph.edge_ptr.back() != graph.edges) {
        fail("graph edge_ptr does not cover the canonical edge list");
    }
    for (u64 gid = 0; gid < graph.graphs; ++gid) {
        if (graph.edge_ptr[gid] >= graph.edge_ptr[gid + 1]) fail("graph edge_ptr contains an empty graph block");
    }
    if (graph.node_edge_ptr.size() != nodes + 1 || graph.node_edge_ptr.front() != 0 || graph.node_edge_ptr.back() != graph.edges) {
        fail("node edge_ptr does not cover the canonical edge list");
    }
    for (u64 node = 0; node < nodes; ++node) {
        if (graph.node_edge_ptr[node] > graph.node_edge_ptr[node + 1]) fail("node edge_ptr is not monotone");
        if (graph.node_edge_ptr[node] == graph.node_edge_ptr[node + 1]) fail("node has no incoming edge after self-loop preparation");
        for (u64 edge = graph.node_edge_ptr[node]; edge < graph.node_edge_ptr[node + 1]; ++edge) {
            if (graph.dst.at(edge) != node || graph.src.at(edge) >= nodes) fail("edge_ptr/destination boundary mismatch");
            if (config.dataset == "ppi") {
                const u64 source_graph = static_cast<u64>(std::upper_bound(graph.node_ptr.begin(), graph.node_ptr.end(), graph.src[edge]) -
                                                          graph.node_ptr.begin() - 1);
                const u64 destination_graph = static_cast<u64>(std::upper_bound(graph.node_ptr.begin(), graph.node_ptr.end(), graph.dst[edge]) -
                                                               graph.node_ptr.begin() - 1);
                if (source_graph != destination_graph) fail("PPI edge crosses graph boundary");
            }
        }
    }
    return graph;
}

Graph load_graph_shape_for_resource_plan(const Config &config, u64 *raw_edge_records,
                                         u64 *csr_max_in_degree) {
    const auto directory = config.cache_root / config.dataset;
    const auto metadata_path = directory / "meta.cfg";
    if (std::filesystem::is_symlink(metadata_path) || !std::filesystem::is_regular_file(metadata_path))
        fail("BLOCKED: analysis-only resource plan metadata is missing or not a regular file");
    const auto metadata = read_kv(metadata_path);
    const auto metadata_value = [&](const std::string &key) -> std::string {
        const auto iterator = metadata.find(key);
        if (iterator == metadata.end()) fail("analysis-only resource metadata key missing: " + key);
        return iterator->second;
    };
    if (metadata_value("name") != config.dataset) fail("analysis-only resource metadata dataset mismatch");
    Graph graph;
    graph.total_nodes = parse_decimal_u64(metadata_value("num_nodes"), "num_nodes");
    graph.nodes = graph.total_nodes;
    graph.features = parse_decimal_u64(metadata_value("num_features"), "num_features");
    graph.graphs = metadata.count("graph_count") ? parse_decimal_u64(metadata.at("graph_count"), "graph_count") : 1U;
    if (metadata_value("input_preprocess_id") != config.input_preprocess_id ||
        (metadata.count("num_classes") && parse_decimal_u64(metadata.at("num_classes"), "num_classes") != config.architecture.classes) ||
        (metadata.count("task_type") && metadata.at("task_type") != config.task_type) ||
        (metadata.count("report_unit") && metadata.at("report_unit") != config.report_unit))
        fail("analysis-only resource metadata does not match the checked config");
    if (graph.nodes == 0U || graph.nodes > kMaxNativeDomain || graph.features == 0U ||
        graph.features > kMaxNativeDomain || graph.graphs == 0U || graph.graphs > kMaxNativeDomain ||
        config.architecture.input_dims.front() != graph.features ||
        (config.dataset != "ppi" && graph.graphs != 1U))
        fail("BLOCKED: analysis-only resource graph dimensions exceed the supported config");

    graph.node_ptr = {0U, graph.nodes};
    if (config.dataset == "ppi") {
        graph.node_ptr.clear();
        std::ifstream node_ptr_file(directory / "node_ptr.txt");
        if (!node_ptr_file) fail("BLOCKED: analysis-only resource plan is missing PPI node_ptr");
        std::string token;
        while (node_ptr_file >> token)
            graph.node_ptr.push_back(parse_decimal_u64(token, "PPI node_ptr"));
        if (!node_ptr_file.eof() || graph.node_ptr.size() != graph.graphs + 1U ||
            graph.node_ptr.empty() || graph.node_ptr.front() != 0U || graph.node_ptr.back() != graph.nodes)
            fail("BLOCKED: analysis-only resource PPI node_ptr shape is invalid");
        for (std::size_t index = 1U; index < graph.node_ptr.size(); ++index)
            if (graph.node_ptr[index] <= graph.node_ptr[index - 1U] || graph.node_ptr[index] > graph.nodes)
                fail("BLOCKED: analysis-only resource PPI node_ptr is not strictly increasing");
    }

    const auto edge_path = directory / "edges.txt";
    if (std::filesystem::is_symlink(edge_path) || !std::filesystem::is_regular_file(edge_path))
        fail("BLOCKED: analysis-only resource edge file is missing or not a regular file");
    std::ifstream edges(edge_path);
    if (!edges) fail("cannot open edge file for analysis-only resource count");
    u64 raw_count = 0U;
    u64 normalized_count = 0U;
    struct EdgeKey {
        u64 graph;
        u64 source;
        u64 destination;
        bool operator==(const EdgeKey &other) const noexcept {
            return graph == other.graph && source == other.source && destination == other.destination;
        }
    };
    struct EdgeKeyHash {
        std::size_t operator()(const EdgeKey &key) const noexcept {
            const auto mix = [](u64 value) {
                value ^= value >> 30U;
                value *= 0xbf58476d1ce4e5b9ULL;
                value ^= value >> 27U;
                value *= 0x94d049bb133111ebULL;
                return value ^ (value >> 31U);
            };
            u64 value = mix(key.graph + 0x9e3779b97f4a7c15ULL);
            value ^= mix(key.source + 0x243f6a8885a308d3ULL) + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
            value ^= mix(key.destination + 0x13198a2e03707344ULL) + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
            return static_cast<std::size_t>(value);
        }
    };
    std::unordered_set<EdgeKey, EdgeKeyHash> seen;
    if (config.deduplicate_edges) seen.reserve(1U << 20U);
    std::vector<std::uint32_t> normalized_in_degree(static_cast<std::size_t>(graph.nodes), 0U);
    const auto append_shape_edge = [&](u64 gid, u64 source, u64 destination) {
        if (config.deduplicate_edges && !seen.insert(EdgeKey{gid, source, destination}).second) return;
        if (normalized_count >= kMaxNativeDomain / 2U)
            fail("BLOCKED: analysis-only normalized edge count exceeds native sparse-route domain bound");
        auto &degree = normalized_in_degree.at(static_cast<std::size_t>(destination));
        if (degree == std::numeric_limits<std::uint32_t>::max())
            fail("BLOCKED: analysis-only normalized CSR degree exceeds u32 bound");
        ++degree;
        ++normalized_count;
    };
    std::string line;
    while (std::getline(edges, line)) {
        line = trim(line);
        if (line.empty()) continue;
        std::stringstream row(line);
        std::string token;
        u64 columns = 0U;
        u64 first = 0U, second = 0U, third = 0U;
        while (row >> token) {
            const u64 value = parse_decimal_u64(token, "edge id/endpoint");
            if (columns == 0U) first = value;
            else if (columns == 1U) second = value;
            else if (columns == 2U) third = value;
            if (columns == std::numeric_limits<u64>::max()) fail("edge row has too many columns");
            ++columns;
        }
        const u64 expected_columns = config.dataset == "ppi" ? 3U : 2U;
        if (columns != expected_columns ||
            (config.dataset == "ppi" && (first >= graph.graphs || second >= graph.nodes || third >= graph.nodes)) ||
            (config.dataset != "ppi" && (first >= graph.nodes || second >= graph.nodes)))
            fail("malformed edge row in analysis-only resource count");
        const u64 gid = config.dataset == "ppi" ? first : 0U;
        const u64 source = config.dataset == "ppi" ? second : first;
        const u64 destination = config.dataset == "ppi" ? third : second;
        if (config.dataset == "ppi" &&
            (source < graph.node_ptr.at(gid) || source >= graph.node_ptr.at(gid + 1U) ||
             destination < graph.node_ptr.at(gid) || destination >= graph.node_ptr.at(gid + 1U)))
            fail("PPI global edge endpoint ownership mismatch in analysis-only resource count");
        if (raw_count == std::numeric_limits<u64>::max()) fail("raw edge record count overflows");
        ++raw_count;
        append_shape_edge(gid, source, destination);
        if (config.symmetrize_edges) append_shape_edge(gid, destination, source);
    }
    if (!edges.eof()) fail("I/O failure while counting raw edge records");
    for (u64 node = 0U; node < graph.nodes; ++node) {
        u64 gid = 0U;
        if (config.dataset == "ppi")
            gid = static_cast<u64>(std::upper_bound(graph.node_ptr.begin(), graph.node_ptr.end(), node) -
                                   graph.node_ptr.begin() - 1);
        append_shape_edge(gid, node, node);
    }
    graph.edges = normalized_count;
    if (graph.edges == 0U || graph.edges > kMaxNativeDomain / 2U)
        fail("BLOCKED: analysis-only normalized edge count exceeds the native sparse-route domain");
    const u64 max_in_degree = static_cast<u64>(*std::max_element(
        normalized_in_degree.begin(), normalized_in_degree.end()));

    // RelationPlan consumes only these public vector lengths. Their contents,
    // graph endpoints, feature rows, normalized graph, and witness are neither
    // constructed nor made available to a proof/admission path here.
    graph.node_ptr.resize(static_cast<std::size_t>(graph.graphs + 1U));
    graph.edge_ptr.resize(static_cast<std::size_t>(graph.graphs + 1U));
    graph.node_edge_ptr.resize(static_cast<std::size_t>(graph.nodes + 1U));
    if (raw_edge_records) *raw_edge_records = raw_count;
    if (csr_max_in_degree) *csr_max_in_degree = max_in_degree;
    return graph;
}

StaticTables build_static_tables(const Config &config, const Graph &graph) {
    (void)graph;
    return load_static_table_descriptor(config);
}

static RelationPlan build_relation_plan_internal(const Config &config, const Graph &graph,
                                                 const StaticTables &tables,
                                                 const IntegerBoundLedger *planned_ledger) {
    RelationPlan plan;
    const bool planned_analysis_only = planned_ledger != nullptr;
    if ((!planned_analysis_only && !tables.authenticated) ||
        (planned_analysis_only && (tables.authenticated || !tables.metadata_only ||
                                   planned_ledger->authenticated || !planned_ledger->analysis_only_planned)))
        fail("BLOCKED: relation plan inputs do not match the selected authenticated/planned mode");
    plan.static_table_descriptor_digest = tables.descriptor_digest;
    plan.integer_bound_ledger = planned_analysis_only ? *planned_ledger : load_integer_bound_ledger(config, tables);
    plan.scale_chain_digest = config.scale_chain_digest;
    plan.integer_bound_ledger_digest = planned_analysis_only ? "ANALYSIS_ONLY_PLANNED" : plan.integer_bound_ledger.digest;
    plan.integer_bound_ledger_descriptor_digest = planned_analysis_only ? "ANALYSIS_ONLY_PLANNED" : plan.integer_bound_ledger.descriptor_digest;
    const auto power_of_two = [](u64 value) {
        if (value == 0U || value > kMaxNativeDomain) fail("relation plan logical dimension exceeds native domain bound");
        u64 padded = 1;
        while (padded < value) {
            if (padded > kMaxNativeDomain / 2U) fail("relation plan padded dimension exceeds native domain bound");
            padded <<= 1;
        }
        return padded;
    };
    plan.sorted_lookup.schema = planned_analysis_only ? "analysis_only_planned_sorted_lookup_shape_v1" : "authenticated_sorted_lookup_plan_v1";
    plan.sorted_lookup.version = 1U;
    plan.sorted_lookup.descriptor_path = planned_analysis_only ? std::string() : tables.sorted_lookup_descriptor;
    plan.sorted_lookup.descriptor_digest = tables.sorted_lookup_descriptor_digest;
    plan.sorted_lookup.table_digest = tables.descriptor_digest;
    plan.sorted_lookup.segment_size = 1ULL << 18U;
    const auto u128_decimal = [](unsigned __int128 value) {
        if (value == 0U) return std::string("0");
        std::string result;
        while (value != 0U) {
            result.push_back(static_cast<char>('0' + static_cast<unsigned>(value % 10U)));
            value /= 10U;
        }
        std::reverse(result.begin(), result.end());
        return result;
    };
    const auto bit_width = [](u64 value) -> std::uint32_t {
        std::uint32_t width = 0U;
        do {
            ++width;
            value >>= 1U;
        } while (value != 0U);
        return width;
    };
    const auto shape_product = [](const std::vector<u64> &shape, const std::string &label) {
        if (shape.empty()) fail("sorted lookup query shape is empty: " + label);
        u64 product = 1U;
        for (const auto dimension : shape) {
            if (dimension == 0U || product > kMaxNativeDomain / dimension) fail("sorted lookup query shape exceeds native coordinate bound: " + label);
            product *= dimension;
        }
        return product;
    };
    const auto add_sorted_lookup_query = [&](const std::string &operation, const std::string &table_ref,
                                             u64 table_rows, i64 table_input_min,
                                             i64 table_output_min, i64 table_output_max,
                                             u64 query_rows) {
        if (table_rows == 0U || query_rows == 0U || table_output_min > table_output_max)
            fail("sorted lookup has empty logical/output domain: " + operation);
        const u64 x_range = table_rows - 1U;
        const u64 y_range = static_cast<u64>(static_cast<__int128>(table_output_max) - table_output_min);
        const std::uint32_t x_bits = bit_width(x_range);
        const std::uint32_t y_bits = bit_width(y_range);
        if (x_bits == 0U || y_bits == 0U || x_bits > 127U || y_bits > 127U || x_bits > 127U - y_bits)
            fail("sorted lookup packed key exceeds u128 bound: " + operation);
        const unsigned __int128 key_max = (static_cast<unsigned __int128>(x_range) << y_bits) | y_range;
        SortedLookupQueryPlan query;
        query.operation_id = operation;
        query.table_ref = table_ref;
        query.query_ref = operation + "/query";
        query.query_x_ref = operation + "/query/x";
        query.query_y_ref = operation + "/query/y";
        query.query_key_ref = operation + "/query/key";
        query.merge_ref = operation + "/merge";
        query.merge_key_ref = operation + "/merge/key";
        query.merge_delta_ref = operation + "/merge/delta";
        query.merge_marker_ref = operation + "/merge/marker";
        query.merge_inverse_ref = operation + "/merge/inverse";
        query.merge_prefix_ref = operation + "/merge/prefix";
        query.query_prefix_ref = operation + "/query/prefix";
        query.commitment_role_map = "table_key,query_x,query_y,query_key,query_prefix,merge_key,merge_delta,"
                                    "merge_marker,merge_inverse,merge_prefix,merge_bits,merge_delta_bits,query_bits";
        query.table_logical_rows = table_rows;
        query.table_domain = power_of_two(query.table_logical_rows);
        query.query_logical_rows = query_rows;
        query.query_domain = power_of_two(query_rows);
        if (query.table_logical_rows > kMaxNativeDomain || query.query_logical_rows > kMaxNativeDomain - query.table_logical_rows)
            fail("sorted lookup merge logical domain exceeds native coordinate bound: " + operation);
        query.merge_logical_rows = query.table_logical_rows + query.query_logical_rows;
        query.merge_domain = power_of_two(query.merge_logical_rows);
        query.segment_size = std::min<u64>(1ULL << 18U, std::min({query.table_domain, query.query_domain, query.merge_domain}));
        if (query.segment_size == 0U || query.table_domain % query.segment_size != 0U || query.query_domain % query.segment_size != 0U ||
            query.merge_domain % query.segment_size != 0U)
            fail("sorted lookup segment layout is not power-of-two aligned: " + operation);
        query.table_segment_count = query.table_domain / query.segment_size;
        query.query_segment_count = query.query_domain / query.segment_size;
        query.merge_segment_count = query.merge_domain / query.segment_size;
        if (query.table_domain > std::numeric_limits<u64>::max() - query.query_domain) fail("sorted lookup domain capacity overflow: " + operation);
        const u64 total_domains = query.table_domain + query.query_domain;
        if (total_domains > std::numeric_limits<u64>::max() - query.merge_domain) fail("sorted lookup domain capacity overflow: " + operation);
        const u64 all_domains = total_domains + query.merge_domain;

        constexpr u64 kScratchRecordBytes = 107U;
        if (all_domains > std::numeric_limits<u64>::max() / kScratchRecordBytes) fail("sorted lookup scratch capacity overflow: " + operation);
        query.x_bits = x_bits;
        query.y_bits = y_bits;
        query.key_max_u128 = u128_decimal(key_max);
        std::ostringstream layout;
        layout << operation << '|' << table_ref << '|' << table_input_min << '|' << query.table_logical_rows << '|' << query.table_domain << '|'
               << query.query_logical_rows << '|' << query.query_domain << '|' << query.merge_logical_rows << '|' << query.merge_domain << '|'
               << query.segment_size << '|' << query.x_bits << '|' << query.y_bits << '|' << query.key_max_u128 << '|' << query.commitment_role_map;
        query.layout_digest = sha256_text(layout.str());
        plan.sorted_lookup.queries.push_back(std::move(query));
    };
    std::set<std::string> tensor_ids;
    std::set<std::string> operation_ids;
    const auto stage_order = [](u64 layer, u64 head) { return "layer=" + std::to_string(layer) + ";head=" + std::to_string(head) + ";node-edge-canonical"; };
    const auto add_tensor = [&](std::string id, std::string visibility, std::string order, std::string axis, std::vector<u64> logical,
                                std::vector<std::string> instances, std::string instance_order, std::string mapping, u64 degree) {
        if (id.empty() || !tensor_ids.insert(id).second) fail("duplicate relation tensor family: " + id);
        if (logical.empty() || instances.empty() || mapping.empty() || degree == 0) { fail("relation tensor has incomplete canonical metadata: " + id); }
        if (visibility != "public" && visibility != "witness") fail("invalid tensor visibility: " + id);
        TensorFamily family;
        family.id = std::move(id);
        family.visibility = std::move(visibility);
        family.mle_order = std::move(order);
        family.stack_axis = std::move(axis);
        family.logical_shape = std::move(logical);
        family.padded_shape.reserve(family.logical_shape.size());
        for (const auto dimension : family.logical_shape) family.padded_shape.push_back(power_of_two(dimension));
        (void)shape_product(family.logical_shape, family.id + ".logical");
        (void)shape_product(family.padded_shape, family.id + ".padded");
        family.instances = std::move(instances);
        family.instance_order = std::move(instance_order);
        family.mapping = std::move(mapping);
        family.zero_selector = "Qpad_v1(" + family.id + ")=1[i<logical_shape],0[otherwise]";
        family.selector_id = "selector:padding:" + family.id;
        family.padding_rule = "all-padding-values-zero;masked-from-relation";
        family.terminal_point = "FS:terminal:" + family.id;
        family.terminal_value = "MLE(" + family.id + "," + family.terminal_point + ")";
        family.terminal_claim = "terminal_claim_v1|family=" + family.id + "|point=" + family.terminal_point + "|value=" + family.terminal_value;
        family.degree_bound = degree;
        plan.tensor_families.push_back(std::move(family));
    };
    const auto add_op = [&](std::string id, std::string kind, u64 degree, std::vector<std::string> inputs, std::vector<std::string> outputs,
                            std::vector<std::string> views, bool reindex, bool public_route, std::vector<std::string> auxiliary, std::string selector) {
        if (id.empty() || !operation_ids.insert(id).second) fail("duplicate relation operation family: " + id);
        if (degree == 0 || kind.empty() || inputs.empty() || outputs.empty() || views.empty() || selector.empty()) {
            fail("relation operation has incomplete canonical metadata: " + id);
        }
        OperationFamily operation;
        operation.id = std::move(id);
        operation.kind = std::move(kind);
        operation.degree = degree;
        operation.degree_bound = "zkDL_degree_v1:" + std::to_string(degree);
        operation.inputs = std::move(inputs);
        operation.outputs = std::move(outputs);
        operation.views = std::move(views);
        operation.requires_reindex = reindex;
        operation.public_route = public_route;
        operation.instance_order = "layer-major/head-major/node-major/edge-major";
        operation.mapping = "operation_view_v1->canonical_mle:" + operation.id;
        operation.auxiliary = std::move(auxiliary);
        operation.selector_id = "selector:operation:" + operation.id + ":" + std::move(selector);
        operation.terminal_point = "FS:operation:" + operation.id;
        operation.terminal_value = "claim(" + operation.id + "," + operation.terminal_point + ")";
        operation.terminal_claim = "operation_terminal_claim_v1|operation=" + operation.id + "|point=" + operation.terminal_point +
                                   "|value=" + operation.terminal_value;
        plan.operation_families.push_back(std::move(operation));
    };
    const auto add_stage_tensors = [&](const std::string &stage, u64 layer, u64 head, u64 in_dim, u64 out_dim, bool output) {
        const std::string order = stage_order(layer, head);
        const std::string input_id = stage + "/trace/input";
        const std::string trace_visibility = layer == 0 && !output ? "public" : "witness";
        add_tensor(stage + "/model/seq_kernel", "witness", "model-layer-head-row-major", "model", {1, in_dim, out_dim}, {"model"}, order,
                   "model_param_registry", 1);
        add_tensor(stage + "/model/attn_src", "witness", "model-layer-head-row-major", "model", {out_dim}, {"model"}, order, "model_param_registry", 1);
        add_tensor(stage + "/model/attn_dst", "witness", "model-layer-head-row-major", "model", {out_dim}, {"model"}, order, "model_param_registry", 1);
        if (output)
            add_tensor(stage + "/model/output_bias", "witness", "model-layer-head-row-major", "model", {out_dim}, {"model"}, order, "model_param_registry", 1);
        add_tensor(input_id, trace_visibility, "node-major-little-endian", "node", {graph.nodes, in_dim}, {"input"}, order, "layer_input_route", 1);
        add_tensor(stage + "/trace/" + (output ? "y_prime" : "h_prime"), "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"projection"},
                   order, "projection_output", 2);
        for (const auto &name :
             {std::string("e_src"), std::string("e_dst"), std::string("score"), std::string("leaky_relu"), std::string("delta"), std::string("exp_weight")}) {
            add_tensor(stage + "/trace/" + name, "witness", "edge-major-little-endian", "edge", {graph.edges}, {name}, order, "edge_trace", 1);
        }
        add_tensor(stage + "/trace/inverse_field", "witness", "edge-major-field-little-endian", "edge", {graph.edges}, {"inverse"}, order,
                   "field_exact:bls12-381-fr", 2);
        add_tensor(stage + "/trace/alpha_field", "witness", "edge-major-field-little-endian", "edge", {graph.edges}, {"alpha"}, order,
                   "field_exact:bls12-381-fr", 2);
        add_tensor(stage + "/trace/maximum", "witness", "node-major-little-endian", "node", {graph.nodes}, {"maximum"}, order, "destination_group_max", 1);
        add_tensor(stage + "/trace/denominator", "witness", "node-major-little-endian", "node", {graph.nodes}, {"sum"}, order, "destination_group_sum", 1);
        add_tensor(stage + "/trace/max_selector", "witness", "edge-major-little-endian", "edge", {graph.edges}, {"max_selector"}, order, "max_arg_selector", 1);
        add_tensor(stage + "/trace/range_bits", "witness", "edge-major-little-endian", "bit", {graph.edges, table_index_bit_width(tables.exp_row_count())}, {"range"}, order,
                   "range_bit_decomposition", 1);
        for (const auto &name : {std::string("lrelu_idx"), std::string("exp_idx")}) {
            add_tensor(stage + "/trace/" + name, "witness", "edge-or-node-major-little-endian", "lookup", std::vector<u64>{graph.edges}, {name}, order,
                       "indexed_public_table", 1);
        }
        add_tensor(stage + "/trace/lrelu_bits", "witness", "lookup-query-bit-major", "bit",
                   {graph.edges, table_index_bit_width(tables.lrelu_row_count())}, {"lrelu_bits"}, order, "lookup_index_bits", 1);
        add_tensor(stage + "/trace/exp_bits", "witness", "lookup-query-bit-major", "bit",
                   {graph.edges, table_index_bit_width(tables.exp_row_count())}, {"exp_bits"}, order, "lookup_index_bits", 1);
        if (!output) {
            add_tensor(stage + "/trace/elu_idx", "witness", "node-major-little-endian", "lookup", {graph.nodes, out_dim}, {"elu_idx"}, order,
                       "indexed_public_table", 1);
            add_tensor(stage + "/trace/elu_bits", "witness", "lookup-query-bit-major", "bit",
                       {graph.nodes, out_dim, table_index_bit_width(tables.elu_row_count())}, {"elu_bits"}, order,
                       "lookup_index_bits", 1);
        }
        const std::string value = stage + "/trace/" + (output ? "y_prime" : "h_prime");
        add_tensor(stage + "/trace/routed_" + (output ? "y_prime" : "h_prime"), "witness", "edge-major-little-endian", "edge", {graph.edges, out_dim},
                   {"route"}, order, "public_sparse_route", 1);
        add_tensor(stage + "/trace/product_u_" + (output ? "y_prime" : "h_prime"), "witness", "edge-major-little-endian", "edge", {graph.edges, out_dim},
                   {"u_product"}, order, "integer_product", 2);
        add_tensor(stage + "/trace/product_alpha_" + (output ? "y_prime" : "h_prime") + "_field", "witness", "edge-major-field-little-endian", "edge",
                   {graph.edges, out_dim}, {"alpha_product"}, order, "field_product:bls12-381-fr", 2);
        add_tensor(stage + "/trace/aggregate_num", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"aggregate_num"}, order,
                   "integer_group_sum", 1);
        add_tensor(stage + "/trace/aggregate_pre_field", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"aggregate_pre_field"}, order,
                   "field_group_sum", 2);
        add_tensor(stage + "/trace/consistency_residual_field", "witness", "node-major-field-little-endian", "node", {graph.nodes, out_dim}, {"consistency"},
                   order, "field_integer_consistency:bls12-381-fr", 2);

        add_tensor(stage + "/trace/rounddiv_aux", "witness", "node-major-column-channel-little-endian", "channel", {graph.nodes, out_dim, 8}, {"division"},
                   order, "rounddiv_auxiliary:node-column-channel", 1);
        add_tensor(stage + "/trace/rounddiv_a_inverse", "witness", "node-major-field-little-endian", "aux", {graph.nodes, out_dim}, {"a_inverse"}, order,
                   "field_exact:bls12-381-fr", 1);
        add_tensor(stage + "/trace/rounddiv_divisor_inverse", "witness", "node-major-field-little-endian", "aux", {graph.nodes, out_dim}, {"divisor_inverse"},
                   order, "field_exact:bls12-381-fr", 1);
        add_tensor(stage + "/trace/aggregate_pre", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"aggregate_pre"}, order,
                   "signed_round_div", 1);
        if (output) {
            add_tensor(stage + "/trace/y_quantized", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"y_quantized"}, order,
                       "output_quantization", 1);
            add_tensor(stage + "/trace/aligned", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"aligned"}, order,
                       "output_scale_alignment", 1);
            add_tensor(stage + "/trace/output", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"output"}, order,
                       "output_bias_addition", 1);
        } else {
            add_tensor(stage + "/trace/elu_input", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"elu_input"}, order, "elu_domain",
                       1);
            add_tensor(stage + "/trace/aggregate", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"aggregate"}, order, "elu_table", 1);
        }
        (void)value;
    };
    add_tensor("input_features", "public", "node-major-little-endian", "node", {graph.nodes, graph.features}, {"H_1"}, "graph-major/node-major",
               "public_feature_index", 1);
    add_tensor("sparse_route", "public", "edge-major-little-endian", "edge", {graph.edges, 2}, {"src", "dst"}, "graph-major/destination-major",
               "public_edge_route", 1);
    add_tensor("group_selectors", "public", "edge-major-little-endian", "edge", {graph.edges, 2}, {"group_new", "group_end"}, "graph-major/destination-major",
               "public_group_selectors", 1);
    add_tensor("node_ptr", "public", "graph-major-little-endian", "graph", {graph.node_ptr.size()}, {"node_ptr"}, "graph-major", "ppi_graph_boundary", 1);
    add_tensor("edge_ptr", "public", "graph-major-little-endian", "graph", {graph.edge_ptr.size()}, {"edge_ptr"}, "graph-major", "ppi_edge_boundary", 1);
    add_tensor("node_edge_ptr", "public", "node-major-little-endian", "node", {graph.node_edge_ptr.size()}, {"node_edge_ptr"}, "node-major",
               "destination_segment_boundary", 1);
    add_tensor("static_lrelu", "public", "table-index-major", "table", {tables.lrelu_row_count()}, {"T_LReLU"}, "canonical-table-row-order",
               "static_table:lrelu", 1);
    add_tensor("static_elu", "public", "table-index-major", "table", {tables.elu_row_count()}, {"T_ELU"}, "canonical-table-row-order",
               "static_table:elu", 1);
    add_tensor("static_exp", "public", "table-index-major", "table", {tables.exp_row_count()}, {"T_exp"}, "canonical-table-row-order",
               "static_table:exp", 1);
    add_tensor("output_head_count", "public", "scalar", "constant", {1}, {"K_out"}, "canonical-constant", "public_scalar", 1);

    for (u64 layer = 0; layer < config.architecture.hidden_dims.size(); ++layer) {
        const u64 in_dim = layer == 0 ? graph.features : config.architecture.hidden_heads[layer - 1] * config.architecture.hidden_dims[layer - 1];
        const u64 head_dim = config.architecture.hidden_dims[layer];
        std::vector<std::string> aggregate_inputs;
        for (u64 head = 0; head < config.architecture.hidden_heads[layer]; ++head) {
            const std::string stage = "hidden/layer" + std::to_string(layer) + "/head" + std::to_string(head);
            add_stage_tensors(stage, layer, head, in_dim, head_dim, false);
            const std::string input = layer == 0 ? "input_features" : "hidden/layer" + std::to_string(layer - 1) + "/concat";
            const std::string hp = stage + "/trace/h_prime";
            const std::string routed = stage + "/trace/routed_h_prime";
            const std::string pu = stage + "/trace/product_u_h_prime";
            const std::string pa = stage + "/trace/product_alpha_h_prime_field";
            add_op(stage + "/projection", "tensor_matmul_sumcheck", 2, {input, stage + "/model/seq_kernel"}, {hp}, {"node", "inner", "feature"}, true, false,
                   {"selector:" + stage + "/projection"}, "projection_padding:" + stage);
            add_op(stage + "/attention_inner_product", "inner_product_sumcheck", 2, {hp, stage + "/model/attn_src", stage + "/model/attn_dst"},
                   {stage + "/trace/e_src", stage + "/trace/e_dst"}, {"edge", "inner", "head"}, true, false, {"src", "dst"}, "route_selector:" + stage);
            add_op(stage + "/sparse_route", "public_selector_reindex", 1, {"sparse_route", hp}, {routed}, {"edge", "node", "feature"}, true, true,
                   {"src", "dst", "node_edge_ptr"}, "public_route:" + stage);
            add_op(stage + "/score_add", "field_addition", 1, {stage + "/trace/e_src", stage + "/trace/e_dst"}, {stage + "/trace/score"}, {"edge", "head"},
                   true, false, {}, "score_add:" + stage);
            add_op(stage + "/leaky_relu", "indexed_public_table_sumcheck", 1,
                   {stage + "/trace/score", "static_lrelu", stage + "/trace/lrelu_idx", stage + "/trace/lrelu_bits"}, {stage + "/trace/leaky_relu"},
                   {"query", "table"}, true, false, {"tau_lrelu", "omega_lrelu"}, "lookup_lrelu:" + stage);
            add_op(stage + "/max_range", "segmented_max_range_sumcheck", 1, {stage + "/trace/leaky_relu", "sparse_route", "group_selectors"},
                   {stage + "/trace/maximum", stage + "/trace/max_selector", stage + "/trace/delta"}, {"node", "edge", "group"}, true, true,
                   {"range_bits", "group_new", "group_end"}, "public_route:max_range:" + stage);
            add_op(stage + "/exp_map", "indexed_public_table_sumcheck", 1,
                   {stage + "/trace/delta", "static_exp", stage + "/trace/exp_idx", stage + "/trace/exp_bits"}, {stage + "/trace/exp_weight"},
                   {"query", "table"}, true, false, {"tau_exp", "omega_exp"}, "lookup_exp:" + stage);
            add_op(stage + "/u_h_prime", "tensor_product_sumcheck", 2, {stage + "/trace/exp_weight", routed}, {pu}, {"edge", "feature"}, true, false, {},
                   "product_u:" + stage);
            add_op(stage + "/group_sum", "segmented_sumcheck", 1, {stage + "/trace/exp_weight", pu, "sparse_route", "edge_ptr", "node_edge_ptr"},
                   {stage + "/trace/denominator", stage + "/trace/aggregate_num"}, {"graph", "node", "edge", "feature"}, true, true,
                   {"group_new", "group_end", "zero_padding"}, "public_route:group_sum:" + stage);
            add_op(stage + "/inverse_alpha", "field_inverse_product_sumcheck", 2, {stage + "/trace/denominator", stage + "/trace/exp_weight", "node_edge_ptr"},
                   {stage + "/trace/inverse_field", stage + "/trace/alpha_field"}, {"node", "edge"}, true, false,
                   {"inverse_field", "alpha_field", "nonzero_denominator"}, "inverse_alpha:" + stage);
            add_op(stage + "/alpha_h_prime", "tensor_product_sumcheck", 2, {stage + "/trace/alpha_field", routed}, {pa}, {"edge", "feature"}, true, false, {},
                   "product_alpha:" + stage);
            add_op(stage + "/field_aggregate", "segmented_sumcheck", 1, {pa, "sparse_route", "node_edge_ptr"}, {stage + "/trace/aggregate_pre_field"},
                   {"node", "edge", "feature"}, true, true, {"group_new", "group_end"}, "public_route:field_aggregate:" + stage);
            add_op(stage + "/integer_numerator", "segmented_sumcheck", 1, {pu, "sparse_route", "node_edge_ptr"}, {stage + "/trace/aggregate_num"},
                   {"node", "edge", "feature"}, true, true, {"group_new", "group_end"}, "public_route:integer_numerator:" + stage);
            add_op(stage + "/consistency", "tensor_product_sumcheck", 2,
                   {stage + "/trace/denominator", stage + "/trace/aggregate_pre_field", stage + "/trace/aggregate_num"},
                   {stage + "/trace/consistency_residual_field"}, {"node", "feature"}, true, false, {}, "consistency:" + stage);
            add_op(stage + "/signed_round_div", "signed_round_div_sumcheck", 1, {stage + "/trace/aggregate_num", stage + "/trace/denominator"},
                   {stage + "/trace/aggregate_pre", stage + "/trace/rounddiv_aux"}, {"node", "feature"}, true, false,
                   {"sigma", "z", "a", "a_inv", "d_inv", "q", "r", "delta", "range_bits"}, "round_div:" + stage);
            add_op(stage + "/elu", "indexed_public_table_sumcheck", 1,
                   {stage + "/trace/aggregate_pre", "static_elu", stage + "/trace/elu_idx", stage + "/trace/elu_bits"}, {stage + "/trace/aggregate"},
                   {"query", "table"}, true, false, {"tau_elu", "omega_elu"}, "lookup_elu:" + stage);
            aggregate_inputs.push_back(stage + "/trace/aggregate");
        }
        const std::string concat = "hidden/layer" + std::to_string(layer) + "/concat";
        add_tensor(concat, "witness", "node-major-head-major-feature", "head", {graph.nodes, config.architecture.hidden_heads[layer], head_dim}, {"heads"},
                   "layer-major/head-major/feature-major", "head_concat_reindex", 1);
        add_op(concat, "tensor_reindex_sumcheck", 1, aggregate_inputs, {concat}, {"node", "head", "feature"}, true, false, {"head_offsets", "zero_padding"},
               "concat:" + std::to_string(layer));
    }
    const u64 output_in = config.architecture.hidden_heads.back() * config.architecture.hidden_dims.back();
    std::vector<std::string> output_values;
    for (u64 head = 0; head < config.architecture.output_heads; ++head) {
        const std::string stage = "output/head" + std::to_string(head);
        add_stage_tensors(stage, config.architecture.hidden_dims.size(), head, output_in, config.architecture.classes, true);
        const std::string hp = stage + "/trace/y_prime";
        const std::string routed = stage + "/trace/routed_y_prime";
        const std::string pu = stage + "/trace/product_u_y_prime";
        const std::string pa = stage + "/trace/product_alpha_y_prime_field";
        const std::string input = "hidden/layer" + std::to_string(config.architecture.hidden_dims.size() - 1) + "/concat";
        add_op(stage + "/output_projection", "tensor_matmul_sumcheck", 2, {input, stage + "/model/seq_kernel"}, {hp}, {"node", "inner", "class"}, true, false,
               {"zero_padding"}, "output_projection:" + stage);
        add_op(stage + "/output_attention_inner_product", "inner_product_sumcheck", 2, {hp, stage + "/model/attn_src", stage + "/model/attn_dst"},
               {stage + "/trace/e_src", stage + "/trace/e_dst"}, {"edge", "inner", "class"}, true, false, {"src", "dst"}, "output_route_selector:" + stage);
        add_op(stage + "/sparse_route", "public_selector_reindex", 1, {"sparse_route", hp}, {routed}, {"edge", "node", "class"}, true, true,
               {"src", "dst", "node_edge_ptr"}, "public_route:" + stage);
        add_op(stage + "/score_add", "field_addition", 1, {stage + "/trace/e_src", stage + "/trace/e_dst"}, {stage + "/trace/score"}, {"edge", "class"}, true,
               false, {}, "score_add:" + stage);
        add_op(stage + "/leaky_relu", "indexed_public_table_sumcheck", 1,
               {stage + "/trace/score", "static_lrelu", stage + "/trace/lrelu_idx", stage + "/trace/lrelu_bits"}, {stage + "/trace/leaky_relu"},
               {"query", "table"}, true, false, {"tau_lrelu", "omega_lrelu"}, "lookup_lrelu:" + stage);
        add_op(stage + "/max_range", "segmented_max_range_sumcheck", 1, {stage + "/trace/leaky_relu", "sparse_route", "group_selectors"},
               {stage + "/trace/maximum", stage + "/trace/max_selector", stage + "/trace/delta"}, {"node", "edge", "group"}, true, true,
               {"range_bits", "group_new", "group_end"}, "public_route:max_range:" + stage);
        add_op(stage + "/exp_map", "indexed_public_table_sumcheck", 1,
               {stage + "/trace/delta", "static_exp", stage + "/trace/exp_idx", stage + "/trace/exp_bits"}, {stage + "/trace/exp_weight"}, {"query", "table"},
               true, false, {"tau_exp", "omega_exp"}, "lookup_exp:" + stage);
        add_op(stage + "/u_y_prime", "tensor_product_sumcheck", 2, {stage + "/trace/exp_weight", routed}, {pu}, {"edge", "class"}, true, false, {},
               "product_u:" + stage);
        add_op(stage + "/group_sum", "segmented_sumcheck", 1, {stage + "/trace/exp_weight", pu, "sparse_route", "edge_ptr", "node_edge_ptr"},
               {stage + "/trace/denominator", stage + "/trace/aggregate_num"}, {"graph", "node", "edge", "class"}, true, true,
               {"group_new", "group_end", "zero_padding"}, "public_route:group_sum:" + stage);
        add_op(stage + "/inverse_alpha", "field_inverse_product_sumcheck", 2, {stage + "/trace/denominator", stage + "/trace/exp_weight", "node_edge_ptr"},
               {stage + "/trace/inverse_field", stage + "/trace/alpha_field"}, {"node", "edge"}, true, false,
               {"inverse_field", "alpha_field", "nonzero_denominator"}, "inverse_alpha:" + stage);
        add_op(stage + "/alpha_y_prime", "tensor_product_sumcheck", 2, {stage + "/trace/alpha_field", routed}, {pa}, {"edge", "class"}, true, false, {},
               "product_alpha:" + stage);
        add_op(stage + "/field_aggregate", "segmented_sumcheck", 1, {pa, "sparse_route", "node_edge_ptr"}, {stage + "/trace/aggregate_pre_field"},
               {"node", "edge", "class"}, true, true, {"group_new", "group_end"}, "public_route:field_aggregate:" + stage);
        add_op(stage + "/integer_numerator", "segmented_sumcheck", 1, {pu, "sparse_route", "node_edge_ptr"}, {stage + "/trace/aggregate_num"},
               {"node", "edge", "class"}, true, true, {"group_new", "group_end"}, "public_route:integer_numerator:" + stage);
        add_op(stage + "/consistency", "tensor_product_sumcheck", 2,
               {stage + "/trace/denominator", stage + "/trace/aggregate_pre_field", stage + "/trace/aggregate_num"},
               {stage + "/trace/consistency_residual_field"}, {"node", "class"}, true, false, {}, "consistency:" + stage);
        add_op(stage + "/signed_round_div", "signed_round_div_sumcheck", 1, {stage + "/trace/aggregate_num", stage + "/trace/denominator"},
               {stage + "/trace/y_quantized", stage + "/trace/rounddiv_aux"}, {"node", "class"}, true, false,
               {"sigma", "z", "a", "a_inv", "d_inv", "q", "r", "delta", "range_bits"}, "round_div:" + stage);
        add_op(stage + "/output_scale_alignment", "signed_round_div_sumcheck", 1, {stage + "/trace/y_quantized"}, {stage + "/trace/aligned"}, {"node", "class"},
               true, false, {"scale_out_head", "scale_out_pre"}, "scale_alignment:" + stage);
        add_op(stage + "/output_bias", "field_addition", 1, {stage + "/trace/aligned", stage + "/model/output_bias"}, {stage + "/trace/output"},
               {"node", "class"}, true, false, {"bias"}, "output_bias:" + stage);
        output_values.push_back(stage + "/trace/output");
    }
    add_tensor("output_heads", "witness", "node-major-head-major-class", "head", {graph.nodes, config.architecture.output_heads, config.architecture.classes},
               {"heads"}, "head-major/node-major/class-major", "output_head_stack", 1);
    add_tensor("output_sum", "witness", "node-major-class-major", "class", {graph.nodes, config.architecture.classes}, {"sum"}, "node-major/class-major",
               "head_sum", 1);
    add_tensor("public_Y", "public", "node-major-class-major", "class", {graph.nodes, config.architecture.classes}, {"logits"}, "node-major/class-major",
               "public_output", 1);
    add_op("head_sum", "tensor_sumcheck", 1, output_values, {"output_sum"}, {"node", "head", "class"}, true, false, {"head_offsets", "zero_padding"},
           "head_sum");
    add_op("final_average", "signed_round_div_sumcheck", 1, {"output_sum", "output_head_count"}, {"public_Y"}, {"node", "class"}, true, true,
           {"rounddiv_aux", "public_output"}, "final_average");
    add_op("terminal_opening", "hyrax_terminal_evaluation", 1, {"public_Y"}, {"public_Y"}, {"public_mle_point"}, false, true,
           {"opening_point", "terminal_claim"}, "public_output_terminal");

    const auto append_view = [&](const std::string &id, const std::vector<u64> &logical, const std::string &mapping, const std::string &axis) {
        if (id.empty() || logical.empty() || !tensor_ids.insert(id).second) fail("duplicate/empty relation operation view family: " + id);
        TensorFamily family;
        family.id = id;
        const auto public_boundary = [&](const std::string &suffix) { return id.size() >= suffix.size() && id.compare(id.size() - suffix.size(), suffix.size(), suffix) == 0U; };
        family.visibility = public_boundary("/view/group_new") || public_boundary("/view/group_end") ? "public" : "witness";
        family.mle_order = "operation-view-row-major";
        family.stack_axis = axis;
        family.logical_shape = logical;
        family.padded_shape.reserve(logical.size());
        for (const auto dimension : logical) family.padded_shape.push_back(power_of_two(dimension));
        family.instances = {"derived"};
        family.instance_order = "operation-major/canonical-view";
        family.mapping = mapping;
        family.zero_selector = "Qpad_v1(" + id + ")=1[i<logical_shape],0[otherwise]";
        family.selector_id = "selector:padding:" + id;
        family.padding_rule = "all-padding-values-zero;masked-from-relation";
        family.terminal_point = "FS:terminal:" + id;
        family.terminal_value = "MLE(" + id + "," + family.terminal_point + ")";
        family.terminal_claim = "terminal_claim_v1|family=" + id + "|point=" + family.terminal_point + "|value=" + family.terminal_value;
        family.degree_bound = 1U;
        plan.tensor_families.push_back(std::move(family));
    };
    if (config.range_bits == 0U || config.range_bits >= 31U) fail("relation operation views require a bounded range_bits width");
    const u64 delta_range_bits = table_index_bit_width(tables.exp_row_count());
    const auto edge_shape = [&](u64 width = 1U) { return width == 1U ? std::vector<u64>{graph.edges} : std::vector<u64>{graph.edges, width}; };
    const auto node_shape = [&](u64 width = 1U) { return width == 1U ? std::vector<u64>{graph.nodes} : std::vector<u64>{graph.nodes, width}; };
    const auto append_round_views = [&](const std::string &operation, u64 width) {
        const auto shape = node_shape(width);
        const auto padded_feature_shape = node_shape(power_of_two(width));
        const auto range_widths = planned_analysis_only
            ? signed_round_bounds::derive_planned(config, plan.integer_bound_ledger, operation)
            : signed_round_bounds::derive(config, plan.integer_bound_ledger, operation);
        for (const auto &role : {"signed_numerator", "signed_quotient", "a", "q_abs", "r", "delta", "sigma", "z", "a_inverse", "divisor_inverse"})
            append_view(operation + "/view/" + std::string(role), shape, "signed_round_div:" + std::string(role), "node");
        // Main aggregation divisions carry both the source's unscaled node
        // denominator and a separate scaled divisor view bound to rounddiv_aux.
        // Each view is logically active across Fpad and explicitly FAC-bound;
        // relation constraints use the canonical numerator's [N,F] active mask.
        append_view(operation + "/view/divisor", padded_feature_shape,
                    range_widths.binds_raw_denominator ? "signed_round_div:raw_divisor:full_padded_feature_broadcast" :
                                                         "signed_round_div:divisor:full_padded_feature_broadcast",
                    "node");
        if (range_widths.binds_raw_denominator)
            append_view(operation + "/view/divisor_aux", padded_feature_shape,
                        "signed_round_div:divisor_aux:full_padded_feature_broadcast", "node");
        const std::vector<std::pair<std::string, u64>> bit_widths = {
            {"a", range_widths.a_bits}, {"q_abs", range_widths.q_bits}, {"r", range_widths.r_bits},
            {"delta", range_widths.delta_bits}, {"divisor", range_widths.divisor_bits},
        };
        for (const auto &value : bit_widths)
            for (u64 bit = 0U; bit < value.second; ++bit)
                append_view(operation + "/view/range/" + value.first + "/" + std::to_string(bit), shape,
                            "signed_round_div:range:" + value.first, "node");
    };
    const auto append_table_views = [&](const std::string &operation, const std::string &table_ref,
                                        bool edge_domain, u64 feature_width, u64 table_rows,
                                        i64 table_input_min, i64 table_output_min,
                                        i64 table_output_max) {
        const auto shape = edge_domain ? edge_shape(feature_width) : node_shape(feature_width);
        add_sorted_lookup_query(operation, table_ref, table_rows, table_input_min,
                                table_output_min, table_output_max,
                                shape_product(shape, operation + ".query"));
        append_view(operation + "/view/query", shape, "indexed_public_table:query", "lookup");
        append_view(operation + "/view/index", shape, "indexed_public_table:index", "lookup");
        append_view(operation + "/view/output0", shape, "indexed_public_table:output", "lookup");
        const u64 depth = table_index_bit_width(table_rows);
        for (u64 bit = 0U; bit < depth; ++bit) append_view(operation + "/view/bit/" + std::to_string(bit), shape, "indexed_public_table:bit", "lookup");
    };
    const auto append_stage_views = [&](const std::string &stage, u64 width, bool output) {
        const auto edge = edge_shape();
        const auto edge_feature = edge_shape(width);
        const auto edge_feature_broadcast = edge_shape(power_of_two(width));
        const auto node = node_shape();
        const auto node_feature = node_shape(width);
        const auto node_feature_broadcast = node_shape(power_of_two(width));
        const std::string inner = stage + (output ? "/output_attention_inner_product" : "/attention_inner_product");
        append_view(inner + "/view/src_matrix", edge_feature, "attention_route:src", "edge");
        append_view(inner + "/view/dst_matrix", edge_feature, "attention_route:dst", "edge");
        append_view(inner + "/view/src_output", edge, "attention_route:e_src", "edge");
        append_view(inner + "/view/dst_output", edge, "attention_route:e_dst", "edge");

        const std::string max = stage + "/max_range";
        for (const auto &role : {"value", "maximum_broadcast", "selector", "delta", "group_new", "group_end", "selector_prefix", "selector_prev"})
            append_view(max + "/view/" + std::string(role), edge, "max_range:" + std::string(role), "edge");
        for (u64 bit = 0U; bit < delta_range_bits; ++bit) append_view(max + "/view/range_bit/" + std::to_string(bit), edge, "max_range:range_bit", "edge");

        const std::string group = stage + "/group_sum";
        for (const auto &role : {"group_new", "group_end"})
            append_view(group + "/view/" + std::string(role), edge_feature_broadcast, "group_sum:" + std::string(role), "edge");
        for (const auto &role : {"input0", "output0", "prefix0", "prefix_prev0"})
            append_view(group + "/view/" + std::string(role), edge_feature_broadcast, "group_sum:" + std::string(role), "edge");
        for (const auto &role : {"input1", "output1", "prefix1", "prefix_prev1"})
            append_view(group + "/view/" + std::string(role), edge_feature, "group_sum:" + std::string(role), "edge");

        const std::string alpha_product = stage + (output ? "/alpha_y_prime" : "/alpha_h_prime");
        append_view(alpha_product + "/view/input0", edge_feature_broadcast,
                    "product_alpha:edge-scalar-feature-broadcast", "edge");

        const std::string inverse = stage + "/inverse_alpha";
        for (const auto &role : {"input0", "input1", "output0", "output1"})
            append_view(inverse + "/view/" + std::string(role), edge, "inverse_alpha:" + std::string(role), "edge");

        for (const auto &operation : {stage + "/field_aggregate", stage + "/integer_numerator"}) {
            append_view(operation + "/view/group_new", edge_feature_broadcast, "segmented_sum:group_new", "edge");
            append_view(operation + "/view/group_end", edge_feature_broadcast, "segmented_sum:group_end", "edge");
            for (const auto &role : {"input0", "output0", "prefix0", "prefix_prev0"})
                append_view(operation + "/view/" + std::string(role), edge_feature, "segmented_sum:" + std::string(role), "edge");
        }

        const std::string consistency = stage + "/consistency";
        append_view(consistency + "/view/input0", node_feature_broadcast,
                    "consistency:denominator-feature-broadcast", "node");
        for (const auto &role : {"input1", "input2", "output0"})
            append_view(consistency + "/view/" + std::string(role), node_feature, "consistency:" + std::string(role), "node");

        append_round_views(stage + "/signed_round_div", width);
        const std::string table_lrelu = stage + "/leaky_relu";
        append_table_views(table_lrelu, "static_lrelu", true, 1U,
                           tables.lrelu_row_count(), tables.lrelu_min,
                           tables.lrelu_value_min, tables.lrelu_value_max);
        const std::string table_exp = stage + "/exp_map";
        append_table_views(table_exp, "static_exp", true, 1U,
                           tables.exp_row_count(), 0,
                           tables.exp_value_min, tables.exp_value_max);
        if (!output)
            append_table_views(stage + "/elu", "static_elu", false, width,
                               tables.elu_row_count(), tables.elu_min,
                               tables.elu_value_min, tables.elu_value_max);
        if (output) {
            append_round_views(stage + "/output_scale_alignment", width);
            const std::string bias = stage + "/output_bias";
            for (const auto &role : {"input0", "input1", "output0"})
                append_view(bias + "/view/" + std::string(role), node_feature, "output_bias:" + std::string(role), "node");
        }
    };
    for (u64 layer = 0U; layer < config.architecture.hidden_dims.size(); ++layer) {
        const u64 width = config.architecture.hidden_dims[layer];
        for (u64 head = 0U; head < config.architecture.hidden_heads[layer]; ++head)
            append_stage_views("hidden/layer" + std::to_string(layer) + "/head" + std::to_string(head), width, false);
        const std::string concat = "hidden/layer" + std::to_string(layer) + "/concat";
        const auto concat_shape = std::vector<u64>{graph.nodes, config.architecture.hidden_heads[layer], width};
        for (u64 head = 0U; head < config.architecture.hidden_heads[layer]; ++head)
            append_view(concat + "/view/input" + std::to_string(head), concat_shape, "head_concat:head" + std::to_string(head), "head");
        append_view(concat + "/view/output", concat_shape, "head_concat:output", "head");
    }
    for (u64 head = 0U; head < config.architecture.output_heads; ++head)
        append_stage_views("output/head" + std::to_string(head), config.architecture.classes, true);
    append_round_views("final_average", config.architecture.classes);

    const auto append_list = [](std::ostringstream &stream, const std::vector<std::string> &values) {
        stream << values.size() << ':';
        for (const auto &value : values) stream << value.size() << ':' << value << ';';
    };
    std::ostringstream sorted_lookup;
    sorted_lookup << "schema=" << plan.sorted_lookup.schema << "\nversion=" << plan.sorted_lookup.version
                  << "\ndescriptor_path=" << plan.sorted_lookup.descriptor_path << "\ndescriptor_digest=" << plan.sorted_lookup.descriptor_digest
                  << "\ntable_digest=" << plan.sorted_lookup.table_digest << "\nkey_encoding=" << plan.sorted_lookup.key_encoding
                  << "\nactive_mask_rule=" << plan.sorted_lookup.active_mask_rule << "\nsorted_delta_rule=" << plan.sorted_lookup.sorted_delta_rule
                  << "\nmarker_rule=" << plan.sorted_lookup.marker_rule << "\ninverse_rule=" << plan.sorted_lookup.inverse_rule
                  << "\nsegment_size=" << plan.sorted_lookup.segment_size << "\n";
    for (const auto &query : plan.sorted_lookup.queries) {
        sorted_lookup << "Q|" << query.operation_id << '|' << query.table_ref << '|' << query.query_ref << '|' << query.query_x_ref << '|' << query.query_y_ref
                      << '|' << query.query_key_ref << '|' << query.merge_ref << '|' << query.merge_key_ref << '|' << query.merge_delta_ref << '|'
                      << query.merge_marker_ref << '|' << query.merge_inverse_ref << '|' << query.merge_prefix_ref << '|' << query.query_prefix_ref << '|'
                      << query.commitment_role_map << '|' << query.table_logical_rows << '|' << query.table_domain << '|' << query.query_logical_rows << '|'
                      << query.query_domain << '|' << query.merge_logical_rows << '|' << query.merge_domain << '|' << query.segment_size << '|'
                      << query.table_segment_count << '|' << query.query_segment_count << '|' << query.merge_segment_count << '|' << query.x_bits << '|'
                      << query.y_bits << '|' << query.key_max_u128 << '|' << query.layout_digest << '\n';
    }
    sorted_lookup << "config_digest=" << config.config_digest << "\nconfig_binding_digest=" << config.config_binding_digest << "\n";
    plan.sorted_lookup.plan_digest = sha256_text(sorted_lookup.str());
    std::ostringstream registry;
    registry << "schema=" << plan.schema_id << "\nproof_scope=" << plan.proof_scope << "\nbinding=" << plan.binding_rule << "\n"
             << "static_table_descriptor_digest=" << plan.static_table_descriptor_digest << "\n"
             << "scale_chain_digest=" << plan.scale_chain_digest << "\n"
             << "integer_bound_ledger_digest=" << plan.integer_bound_ledger_digest << "\n"
             << "integer_bound_ledger_descriptor_digest=" << plan.integer_bound_ledger_descriptor_digest << "\n"
             << "sorted_lookup_plan_digest=" << plan.sorted_lookup.plan_digest << "\n";
    for (const auto &family : plan.tensor_families) {
        registry << "T|" << family.id << '|' << family.visibility << '|' << family.mle_order << '|' << family.stack_axis << '|';
        append_list(registry, family.instances);
        registry << family.logical_shape.size() << ':';
        for (const auto value : family.logical_shape) registry << value << ',';
        registry << '|' << family.padded_shape.size() << ':';
        for (const auto value : family.padded_shape) registry << value << ',';
        registry << '|' << family.instance_order << '|' << family.mapping << '|' << family.zero_selector << '|' << family.selector_id << '|'
                 << family.padding_rule << '|' << family.terminal_point << '|' << family.terminal_value << '|' << family.terminal_claim << '|'
                 << family.degree_bound << '\n';
    }
    for (const auto &operation : plan.operation_families) {
        registry << "O|" << operation.id << '|' << operation.kind << '|' << operation.degree_bound << '|' << operation.degree << '|';
        append_list(registry, operation.inputs);
        registry << '|';
        append_list(registry, operation.outputs);
        registry << '|';
        append_list(registry, operation.views);
        registry << '|' << operation.instance_order << '|' << operation.mapping << '|';
        append_list(registry, operation.auxiliary);
        registry << '|' << operation.selector_id << '|' << operation.terminal_point << '|' << operation.terminal_value << '|' << operation.terminal_claim << '|'
                 << (operation.requires_reindex ? 1 : 0) << '|' << (operation.public_route ? 1 : 0) << '\n';
    }
    plan.registry_digest = sha256_text(registry.str());
    std::ostringstream claims;
    std::ostringstream mappings;
    std::ostringstream selectors;
    std::ostringstream degrees;
    for (const auto &family : plan.tensor_families) {
        claims << family.id << '=' << family.terminal_claim << '\n';
        mappings << family.id << '=' << family.mapping << '\n';
        selectors << family.id << '=' << family.selector_id << '|' << family.zero_selector << '|' << family.padding_rule << '\n';
        degrees << family.id << '=' << family.degree_bound << '\n';
    }
    for (const auto &operation : plan.operation_families) {
        claims << operation.id << '=' << operation.terminal_claim << '\n';
        mappings << operation.id << '=' << operation.mapping << '\n';
        selectors << operation.id << '=' << operation.selector_id << '\n';
        degrees << operation.id << '=' << operation.degree_bound << '\n';
    }
    plan.terminal_claims_digest = sha256_text(claims.str());
    plan.mapping_digest = sha256_text(mappings.str());
    plan.selector_digest = sha256_text(selectors.str());
    plan.degree_digest = sha256_text(degrees.str());
    claims << "sorted_lookup=" << plan.sorted_lookup.plan_digest << '\n';
    mappings << "sorted_lookup=" << plan.sorted_lookup.descriptor_digest << '|' << plan.sorted_lookup.table_digest << '\n';
    selectors << "sorted_lookup=" << plan.sorted_lookup.key_encoding << '|' << plan.sorted_lookup.active_mask_rule << '|'
              << plan.sorted_lookup.sorted_delta_rule << '|' << plan.sorted_lookup.marker_rule << '|' << plan.sorted_lookup.inverse_rule << '|'
              << plan.sorted_lookup.plan_digest << '\n';
    degrees << "sorted_lookup=" << plan.sorted_lookup.segment_size << '\n';
    plan.terminal_claims_digest = sha256_text(claims.str());
    plan.mapping_digest = sha256_text(mappings.str());
    plan.selector_digest = sha256_text(selectors.str());
    plan.degree_digest = sha256_text(degrees.str());
    plan.schema_digest = sha256_text(plan.schema_id + "\n" + plan.proof_scope + "\n" + plan.binding_rule + "\n" + plan.sorted_lookup.plan_digest + "\n" +
                                     config.config_digest + "\n" + config.config_binding_digest + "\n");
    return plan;
}

RelationPlan build_relation_plan(const Config &config, const Graph &graph, const StaticTables &tables) {
    return build_relation_plan_internal(config, graph, tables, nullptr);
}

static RelationPlan build_relation_plan_planned(const Config &config, const Graph &graph,
                                                const StaticTables &tables,
                                                const IntegerBoundLedger &ledger) {
    return build_relation_plan_internal(config, graph, tables, &ledger);
}

bool relation_plan_complete(const RelationPlan &relation, std::string *error) {
    const auto reject = [&](const std::string &message) {
        if (error) *error = message;
        return false;
    };
    const auto bounded_power_of_two = [](u64 value) -> u64 {
        if (value == 0U || value > kMaxNativeDomain) return 0U;
        u64 padded = 1U;
        while (padded < value) {
            if (padded > kMaxNativeDomain / 2U) return 0U;
            padded <<= 1U;
        }
        return padded;
    };
    const auto bounded_shape_product = [](const std::vector<u64> &shape) -> u64 {
        if (shape.empty()) return 0U;
        u64 product = 1U;
        for (const u64 axis : shape) {
            if (axis == 0U || product > kMaxNativeDomain / axis) return 0U;
            product *= axis;
        }
        return product;
    };
    if (relation.schema_id != "zkDL_gat_inference_v1" || relation.proof_scope != "full_gat_forward" || relation.binding_rule != "fac4dnn_reindex_v1") {
        return reject("relation plan schema/proof/binding identity mismatch");
    }
    for (const auto &digest : {relation.registry_digest, relation.terminal_claims_digest, relation.schema_digest, relation.mapping_digest,
                               relation.selector_digest, relation.degree_digest, relation.static_table_descriptor_digest, relation.scale_chain_digest,
                               relation.integer_bound_ledger_digest, relation.integer_bound_ledger_descriptor_digest, relation.sorted_lookup.descriptor_digest,
                               relation.sorted_lookup.table_digest, relation.sorted_lookup.plan_digest}) {
        if (!is_digest(digest)) return reject("relation plan digest is not a canonical SHA-256");
    }
    if (relation.sorted_lookup.schema != "authenticated_sorted_lookup_plan_v1" || relation.sorted_lookup.version != 1U ||
        relation.sorted_lookup.descriptor_path.empty() || relation.sorted_lookup.key_encoding != "packed_pair_u128" ||
        relation.sorted_lookup.active_mask_rule != "prefix_one_then_zero_v1" || relation.sorted_lookup.sorted_delta_rule != "nonnegative_bounded_delta_v1" ||
        relation.sorted_lookup.marker_rule != "group_first_v1" || relation.sorted_lookup.inverse_rule != "delta_inverse_or_zero_v1" ||
        relation.sorted_lookup.segment_size == 0U || relation.sorted_lookup.segment_size > (1ULL << 18U) || relation.sorted_lookup.queries.empty()) {
        return reject("relation plan sorted-lookup descriptor/layout is not canonical");
    }
    for (const auto &query : relation.sorted_lookup.queries) {
        if (query.operation_id.empty() || query.table_ref.empty() || query.query_ref.empty() || query.merge_ref.empty() || query.commitment_role_map.empty() ||
        query.table_logical_rows == 0U || query.query_logical_rows == 0U || query.table_logical_rows > kMaxNativeDomain ||
            query.query_logical_rows > kMaxNativeDomain - std::min(query.table_logical_rows, kMaxNativeDomain) ||
            query.merge_logical_rows != query.table_logical_rows + query.query_logical_rows ||
            query.merge_logical_rows > kMaxNativeDomain || query.table_domain != bounded_power_of_two(query.table_logical_rows) ||
            query.query_domain != bounded_power_of_two(query.query_logical_rows) || query.merge_domain != bounded_power_of_two(query.merge_logical_rows) ||
            query.segment_size == 0U || query.segment_size > relation.sorted_lookup.segment_size || bounded_power_of_two(query.segment_size) != query.segment_size ||
            query.table_domain % query.segment_size != 0U || query.query_domain % query.segment_size != 0U || query.merge_domain % query.segment_size != 0U ||
            query.table_segment_count != query.table_domain / query.segment_size || query.query_segment_count != query.query_domain / query.segment_size ||
            query.merge_segment_count != query.merge_domain / query.segment_size || query.x_bits == 0U || query.y_bits == 0U ||
            query.x_bits > 127U || query.y_bits > 127U || query.x_bits > 127U - query.y_bits || query.key_max_u128.empty() || !is_digest(query.layout_digest)) {
            return reject("relation plan sorted-lookup query layout is invalid: " + query.operation_id);
        }
    }
    if (!relation.integer_bound_ledger.authenticated || relation.integer_bound_ledger.digest != relation.integer_bound_ledger_digest ||
        relation.integer_bound_ledger.descriptor_digest != relation.integer_bound_ledger_descriptor_digest ||
        relation.integer_bound_ledger.static_table_descriptor_digest != relation.static_table_descriptor_digest ||
        relation.scale_chain_digest != relation.integer_bound_ledger.scale_chain_digest ||
        relation.sorted_lookup.table_digest != relation.static_table_descriptor_digest) {
        return reject("relation plan integer-bound ledger identity is not authenticated");
    }
    std::set<std::string> tensor_ids;
    for (const auto &family : relation.tensor_families) {
        if (family.id.empty() || !tensor_ids.insert(family.id).second) { return reject("relation plan has duplicate/empty tensor family id"); }
        const auto trace_marker = family.id.rfind("/trace/");
        const auto trace_name = trace_marker == std::string::npos ? std::string{} : family.id.substr(trace_marker + 7);
        if (trace_name == "inverse" || trace_name == "alpha" || trace_name == "product_alpha_h_prime" || trace_name == "product_alpha_y_prime") {
            return reject("relation plan contains retired integer inverse/alpha/product tensor: " + family.id);
        }
        if (family.visibility != "public" && family.visibility != "witness" || family.logical_shape.empty() ||
            family.logical_shape.size() != family.padded_shape.size() || family.instances.empty() || family.mle_order.empty() || family.stack_axis.empty() ||
            family.instance_order.empty() || family.mapping.empty() || family.selector_id.empty() || family.padding_rule.empty() ||
            family.terminal_point.empty() || family.terminal_value.empty() || family.terminal_claim.empty() || family.degree_bound == 0) {
            return reject("relation tensor family has incomplete canonical metadata: " + family.id);
        }
        for (std::size_t index = 0; index < family.logical_shape.size(); ++index) {
            const u64 logical = family.logical_shape[index];
            const u64 padded = family.padded_shape[index];
            if (logical == 0U || padded != bounded_power_of_two(logical)) {
                return reject("relation tensor family has non-power-of-two padding: " + family.id);
            }
        }
        if (bounded_shape_product(family.logical_shape) == 0U || bounded_shape_product(family.padded_shape) == 0U) {
            return reject("relation tensor family logical/padded product exceeds native coordinate boundary: " + family.id);
        }
        const std::string expected_selector = "selector:padding:" + family.id;
        if (family.selector_id != expected_selector || family.zero_selector.find("Qpad_v1(" + family.id + ")") != 0 ||
            family.padding_rule.find("all-padding-values-zero") == std::string::npos) {
            return reject("relation tensor family padding selector is not canonical: " + family.id);
        }
        if (family.terminal_claim.find("terminal_claim_v1|family=" + family.id + "|") != 0 ||
            family.terminal_claim.find("|point=" + family.terminal_point + "|") == std::string::npos ||
            family.terminal_claim.find("|value=" + family.terminal_value) == std::string::npos) {
            return reject("relation tensor terminal claim is not bound to its family: " + family.id);
        }
    }
    const auto has_tensor = [&](const std::string &id) { return tensor_ids.count(id) != 0; };
    if (!has_tensor("input_features") || !has_tensor("sparse_route") || !has_tensor("group_selectors") || !has_tensor("node_ptr") || !has_tensor("edge_ptr") ||
        !has_tensor("node_edge_ptr") || !has_tensor("static_lrelu") || !has_tensor("static_elu") || !has_tensor("static_exp") ||
        !has_tensor("output_head_count") || !has_tensor("output_heads") || !has_tensor("output_sum") || !has_tensor("public_Y")) {
        return reject("relation plan is missing a canonical public/aggregate tensor family");
    }
    std::set<std::string> operation_ids;
    std::map<std::string, const OperationFamily *> operations;
    for (const auto &operation : relation.operation_families) {
        if (operation.id.empty() || !operation_ids.insert(operation.id).second) { return reject("relation plan has duplicate/empty operation family id"); }
        if (operation.kind.empty() || operation.degree == 0 || operation.degree_bound != "zkDL_degree_v1:" + std::to_string(operation.degree) ||
            operation.inputs.empty() || operation.outputs.empty() || operation.views.empty() || operation.instance_order.empty() || operation.mapping.empty() ||
            operation.selector_id.empty() || operation.terminal_point.empty() || operation.terminal_value.empty() || operation.terminal_claim.empty()) {
            return reject("relation operation has incomplete canonical metadata: " + operation.id);
        }
        for (const auto &input : operation.inputs)
            if (!has_tensor(input)) return reject("relation operation input is unregistered: " + operation.id + " -> " + input);
        for (const auto &output : operation.outputs)
            if (!has_tensor(output)) return reject("relation operation output is unregistered: " + operation.id + " -> " + output);
        if (operation.mapping.find("operation_view_v1->canonical_mle:" + operation.id) != 0)
            return reject("relation operation mapping is not canonical: " + operation.id);
        if (operation.terminal_claim.find("operation_terminal_claim_v1|operation=" + operation.id + "|") != 0 ||
            operation.terminal_claim.find("|point=" + operation.terminal_point + "|") == std::string::npos ||
            operation.terminal_claim.find("|value=" + operation.terminal_value) == std::string::npos) {
            return reject("relation operation terminal claim is not bound to its operation: " + operation.id);
        }
        if (operation.id != "terminal_opening" && !operation.requires_reindex)
            return reject("non-terminal operation omitted FAC4DNN re-index binding: " + operation.id);
        if (operation.selector_id.find("selector:operation:" + operation.id + ":") != 0)
            return reject("operation selector namespace is not canonical: " + operation.id);
        if (operation.public_route && operation.selector_id.find("public_route:") == std::string::npos && operation.id != "final_average" &&
            operation.id != "terminal_opening")
            return reject("public route operation has wrong selector: " + operation.id);
        if (operation.kind.find("indexed_public_table") != std::string::npos &&
            (operation.auxiliary.size() < 2 || operation.auxiliary[0].find("tau_") != 0 || operation.auxiliary[1].find("omega_") != 0)) {
            return reject("indexed table operation lacks independent challenges: " + operation.id);
        }
        if (operation.kind.find("signed_round_div") != std::string::npos && operation.auxiliary.empty())
            return reject("round-div operation lacks auxiliary witness registry: " + operation.id);
        operations.emplace(operation.id, &operation);
    }
    const auto require_op = [&](const std::string &id) -> bool {
        if (!operations.count(id)) return reject("relation plan is missing mandatory operation family: " + id);
        return true;
    };
    for (const auto &family : relation.tensor_families) {
        if (family.id.find("/model/seq_kernel") == std::string::npos) continue;
        const auto stage = family.id.substr(0, family.id.find("/model/"));
        if (stage.rfind("hidden/layer", 0) == 0) {
            for (const auto &suffix :
                 {"/projection", "/attention_inner_product", "/sparse_route", "/score_add", "/leaky_relu", "/max_range", "/exp_map", "/u_h_prime", "/group_sum",
                  "/inverse_alpha", "/alpha_h_prime", "/field_aggregate", "/integer_numerator", "/consistency", "/signed_round_div", "/elu"}) {
                if (!require_op(stage + suffix)) return false;
            }
        } else if (stage.rfind("output/head", 0) == 0) {
            for (const auto &suffix : {"/output_projection", "/output_attention_inner_product", "/sparse_route", "/score_add", "/leaky_relu", "/max_range",
                                       "/exp_map", "/u_y_prime", "/group_sum", "/inverse_alpha", "/alpha_y_prime", "/field_aggregate", "/integer_numerator",
                                       "/consistency", "/signed_round_div", "/output_scale_alignment", "/output_bias"}) {
                if (!require_op(stage + suffix)) return false;
            }
        } else {
            return reject("model tensor family is not layer/head canonical: " + family.id);
        }
    }
    for (const auto &id : {std::string("head_sum"), std::string("final_average"), std::string("terminal_opening")}) {
        if (!require_op(id)) return false;
    }
    const auto terminal = operations.at("terminal_opening");
    if (terminal->inputs != std::vector<std::string>{"public_Y"} || terminal->outputs != std::vector<std::string>{"public_Y"} || !terminal->public_route) {
        return reject("terminal opening does not bind the canonical public output family");
    }
    return true;
}

using NativeFrWire = NativeFieldWire;

const NativeFrWire &native_scalar_modulus() {
    static const NativeFrWire modulus = {{1U, 4294967295U, 4294859774U, 1404937218U, 161601541U, 859428872U, 698187080U, 1944954707U}};
    return modulus;
}

int native_fr_compare(const NativeFrWire &left, const NativeFrWire &right) {
    for (std::size_t index = left.size(); index-- > 0U;) {
        if (left[index] < right[index]) return -1;
        if (left[index] > right[index]) return 1;
    }
    return 0;
}

bool native_fr_canonical(const NativeFrWire &value) { return native_fr_compare(value, native_scalar_modulus()) < 0; }

NativeFrWire native_fr_sub_raw(const NativeFrWire &left, const NativeFrWire &right) {
    NativeFrWire output{};
    std::uint64_t borrow = 0U;
    for (std::size_t index = 0U; index < output.size(); ++index) {
        const std::uint64_t a = left[index];
        const std::uint64_t b = static_cast<std::uint64_t>(right[index]) + borrow;
        output[index] = static_cast<std::uint32_t>(a - b);
        borrow = a < b ? 1U : 0U;
    }
    return output;
}

NativeFrWire native_fr_add(const NativeFrWire &left, const NativeFrWire &right) {
    NativeFrWire output{};
    std::uint64_t carry = 0U;
    for (std::size_t index = 0U; index < output.size(); ++index) {
        const std::uint64_t sum = static_cast<std::uint64_t>(left[index]) + right[index] + carry;
        output[index] = static_cast<std::uint32_t>(sum);
        carry = sum >> 32U;
    }
    if (carry != 0U || native_fr_compare(output, native_scalar_modulus()) >= 0) output = native_fr_sub_raw(output, native_scalar_modulus());
    return output;
}

NativeFrWire native_fr_neg(const NativeFrWire &value) {
    NativeFrWire zero{};
    return native_fr_canonical(value) && native_fr_compare(value, zero) != 0 ? native_fr_sub_raw(native_scalar_modulus(), value) : zero;
}

NativeFrWire native_fr_from_signed_i64(i64 value) {
    if (value >= 0) {
        NativeFrWire output{};
        output[0] = static_cast<std::uint32_t>(value);
        output[1] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(value) >> 32U);
        return output;
    }
    const std::uint64_t magnitude = static_cast<std::uint64_t>(-(value + 1)) + 1U;
    NativeFrWire output{};
    output[0] = static_cast<std::uint32_t>(magnitude);
    output[1] = static_cast<std::uint32_t>(magnitude >> 32U);
    return native_fr_neg(output);
}

NativeFrWire native_wire(i64 value) { return native_fr_from_signed_i64(value); }

NativeFrWire native_wire(const FieldValue &value) {
    NativeFrWire wire{};
    for (std::size_t limb = 0U; limb < wire.size(); ++limb) {
        const std::size_t offset = limb * sizeof(std::uint32_t);
        wire[limb] = static_cast<std::uint32_t>(value.bytes[offset]) | (static_cast<std::uint32_t>(value.bytes[offset + 1U]) << 8U) |
                     (static_cast<std::uint32_t>(value.bytes[offset + 2U]) << 16U) | (static_cast<std::uint32_t>(value.bytes[offset + 3U]) << 24U);
    }
    if (!native_fr_canonical(wire)) fail("non-canonical field witness encoding");
    return wire;
}

u64 native_shape_product(const std::vector<u64> &shape, const std::string &label) {
    return checked_shape_product(shape, label);
}

std::vector<NativeFrWire> native_embed_zero_padded(const TensorFamily &family, const std::vector<NativeFrWire> &logical) {
    const u64 logical_size = native_shape_product(family.logical_shape, family.id + ".logical");
    const u64 padded_size = native_shape_product(family.padded_shape, family.id + ".padded");
    if (logical.size() != logical_size || padded_size > kMaxNativeDomain)
        fail("canonical tensor value count/field domain mismatch: " + family.id);
    std::vector<NativeFrWire> result(static_cast<std::size_t>(padded_size));
    std::vector<u64> coordinate(family.logical_shape.size(), 0U);
    for (u64 linear = 0U; linear < logical_size; ++linear) {
        u64 remainder = linear;
        u64 padded_index = 0U;
        for (std::size_t index = family.logical_shape.size(); index-- > 0U;) {
            coordinate[index] = remainder % family.logical_shape[index];
            remainder /= family.logical_shape[index];
        }
        for (std::size_t index = 0U; index < family.padded_shape.size(); ++index) {
            padded_index = padded_index * family.padded_shape[index] + coordinate[index];
        }
        result.at(static_cast<std::size_t>(padded_index)) = logical.at(static_cast<std::size_t>(linear));
    }
    return result;
}

std::vector<NativeFrWire> native_i64_values(const std::vector<i64> &values) {
    std::vector<NativeFrWire> result;
    result.reserve(values.size());
    for (const auto value : values) result.push_back(native_wire(value));
    return result;
}

void populate_authenticated_resource_wire(const Config &config, NativeRelationWitnessInput &output, bool prover_sidecar) {
    output.registry_path = config.model_registry_path.string();

    AuthenticatedSrsResource srs;
    std::string resource_error;
    if (!load_authenticated_srs(config, &srs, &resource_error)) return;

    AuthenticatedModelRegistry registry;
    if (!load_authenticated_model_registry(config, srs, &registry, &resource_error)) return;

    AuthenticatedModelBlinders blinders;
    if (prover_sidecar && !load_authenticated_model_blinders(config, registry, &blinders, &resource_error)) return;

    output.srs_digest = srs.manifest_digest;
    output.srs_authenticated = srs.authenticated;
    output.registry_path = registry.path;
    output.registry_digest = registry.digest;
    output.registry_authenticated = registry.authenticated;
    output.generators = srs.generators;
    output.ipa_generators = srs.generators;
    output.ipa_blinding_generator = srs.blinding_generator;
    output.ipa_u_generator = srs.u_generator;
    output.registered_model_commitments = registry.model_commitments;

    if (!prover_sidecar) return;
    output.model_blinder_digest = blinders.digest;
    output.model_blinders = blinders.entries;
    output.model_blinders_authenticated = blinders.authenticated;
}

std::vector<NativeFrWire> native_matrix_values(const Matrix &matrix) { return native_i64_values(matrix.values); }

std::vector<i64> native_node_values_from_edges(const Graph &graph, const std::vector<i64> &edge_values) {
    if (edge_values.size() != graph.edges || graph.node_edge_ptr.size() != graph.nodes + 1U) fail("edge-to-node trace shape mismatch");
    std::vector<i64> result(graph.nodes, 0);
    for (u64 node = 0U; node < graph.nodes; ++node) {
        const u64 begin = graph.node_edge_ptr.at(node);
        const u64 end = graph.node_edge_ptr.at(node + 1U);
        if (begin > end || end > graph.edges) fail("destination group boundary is outside edge domain");
        if (begin != end) result[node] = edge_values.at(static_cast<std::size_t>(begin));
    }
    return result;
}

std::vector<std::uint8_t> native_group_new(const Graph &graph) {
    if (graph.dst.size() != graph.edges) fail("graph group selector shape mismatch");
    std::vector<std::uint8_t> result(graph.edges, 0U);
    for (u64 edge = 0U; edge < graph.edges; ++edge) {
        if (graph.dst[edge] >= graph.nodes) fail("graph destination is outside node domain");
        result[edge] = edge == 0U || graph.dst[edge] != graph.dst[edge - 1U] ? 1U : 0U;
    }
    return result;
}

std::vector<std::uint8_t> native_group_end(const Graph &graph) {
    if (graph.dst.size() != graph.edges) fail("graph group selector shape mismatch");
    std::vector<std::uint8_t> result(graph.edges, 0U);
    for (u64 edge = 0U; edge < graph.edges; ++edge) {
        if (graph.dst[edge] >= graph.nodes) fail("graph destination is outside node domain");
        result[edge] = edge + 1U == graph.edges || graph.dst[edge] != graph.dst[edge + 1U] ? 1U : 0U;
    }
    return result;
}

NativeRelationWitnessInput build_native_relation_witness(const Config &config, const Graph &graph, const ModelBundle &model, const StaticTables &tables,
                                                         const RelationPlan &plan, const GatTrace &trace) {
    std::map<std::string, const TensorFamily *> families;
    for (const auto &family : plan.tensor_families) {
        if (!families.emplace(family.id, &family).second) fail("duplicate canonical witness family: " + family.id);
    }
    std::map<std::string, std::shared_ptr<const NativeFamilyStorage>> logical;
    const auto storage_directory = config.run_root / ".private_native";
    if (std::filesystem::is_symlink(storage_directory)) fail("private native storage directory is symlinked");
    std::filesystem::create_directories(storage_directory);
    std::filesystem::permissions(storage_directory, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace);
    std::vector<NativeSortedLookupProviderInput> sorted_lookup_providers;
    double sorted_lookup_protocol_construction_seconds = 0.0;
    double proof_auxiliary_construction_seconds = 0.0;
    const auto family_for = [&](const std::string &id) -> const TensorFamily & {
        const auto found = families.find(id);
        if (found == families.end()) fail("canonical witness family is not in RelationPlan: " + id);
        return *found->second;
    };
    struct NativeValues {
        std::uint64_t count;
        std::uint32_t encoding;
        std::function<NativeFrWire(std::uint64_t)> reader;
        std::uint64_t size() const { return count; }
        NativeFrWire operator[](std::uint64_t index) const { if (index >= count) fail("streamed witness read exceeds logical domain"); return reader(index); }
    };
    const auto view = [](std::uint64_t count, std::uint32_t encoding, auto reader) { return NativeValues{count, encoding, reader}; };
    const auto native_i64_values = [&](const auto &values) { return view(values.size(), 2U, [&values](std::uint64_t index) { return native_wire(static_cast<i64>(values.at(index))); }); };
    const auto native_field_values = [&](const std::vector<FieldValue> &values) { return view(values.size(), 3U, [&values](std::uint64_t index) { return native_wire(values.at(index)); }); };
    const auto native_matrix_values = [&](const Matrix &matrix) { return native_i64_values(matrix.values); };
    const auto native_field_matrix_values = [&](const FieldMatrix &matrix) { return native_field_values(matrix.values); };
    const auto native_bit_column = [&](const std::vector<i64> &values, u64 bit, const std::string &label) { return view(values.size(), 1U, [&values, bit, label](std::uint64_t index) { const auto value = values.at(index); if (value < 0 || bit >= 64U) fail("invalid bounded bit source: " + label); return native_wire(static_cast<i64>((static_cast<std::uint64_t>(value) >> bit) & 1U)); }); };
    const auto native_repeat_edge_bits = [&](const std::vector<std::uint8_t> &values, u64 width) { return view(values.size() * width, 1U, [&values, width](std::uint64_t index) { return native_wire(static_cast<i64>(values.at(index / width))); }); };
    const auto native_edge_feature_from_scalar = [&](const std::vector<i64> &values, u64 width) { return view(values.size() * width, 2U, [&values, width](std::uint64_t index) { return native_wire(values.at(index / width)); }); };
    const auto native_edge_feature_from_field_scalar = [&](const std::vector<FieldValue> &values, u64 width) { return view(values.size() * width, 3U, [&values, width](std::uint64_t index) { return native_wire(values.at(index / width)); }); };
    const auto native_broadcast_node_vector = [&](const Graph &g, const std::vector<i64> &values) { if (values.size() != g.nodes) fail("node-to-edge broadcast shape mismatch"); return view(g.edges, 2U, [&g, &values](std::uint64_t index) { return native_wire(values.at(g.dst.at(index))); }); };
    const auto native_broadcast_node_scalar_features = [&](const Graph &g, const std::vector<i64> &values, u64 width) {
        if (values.size() != g.nodes || width == 0U) fail("node-to-feature broadcast shape mismatch");
        return view(g.nodes * width, 2U, [&values, width](std::uint64_t index) { return native_wire(values.at(index / width)); });
    };
    const auto native_broadcast_node_matrix = [&](const Graph &g, const Matrix &matrix) { if (matrix.rows != g.nodes) fail("node-to-edge matrix broadcast shape mismatch"); return view(g.edges * matrix.cols, 2U, [&g, &matrix](std::uint64_t index) { return native_wire(matrix.at(g.dst.at(index / matrix.cols), index % matrix.cols)); }); };
    const auto native_broadcast_node_field_matrix = [&](const Graph &g, const FieldMatrix &matrix) { if (matrix.rows != g.nodes) fail("node-to-edge field broadcast shape mismatch"); return view(g.edges * matrix.cols, 3U, [&g, &matrix](std::uint64_t index) { return native_wire(matrix.values.at(g.dst.at(index / matrix.cols) * matrix.cols + index % matrix.cols)); }); };
    const auto native_route_matrix = [&](const Graph &g, const Matrix &matrix, bool source) { if (matrix.rows != g.nodes) fail("route matrix shape mismatch"); return view(g.edges * matrix.cols, 2U, [&g, &matrix, source](std::uint64_t index) { return native_wire(matrix.at((source ? g.src : g.dst).at(index / matrix.cols), index % matrix.cols)); }); };
    const auto native_prefix = [&](NativeValues input, const std::vector<std::uint8_t> &group_bits, u64 width, bool previous) {
        if (input.size() != group_bits.size() * width || width == 0U) fail("segmented prefix input shape mismatch");
        return view(input.size(), 3U, [input, &group_bits, width, previous, running = std::vector<NativeFrWire>(width)](std::uint64_t index) mutable { const auto edge = index / width, column = index % width; if (edge == 0U || group_bits.at(edge) != 0U) running.at(column) = NativeFrWire{}; const auto prior = running.at(column); running.at(column) = native_fr_add(prior, input[index]); return previous ? prior : running.at(column); });
    };
    const auto put = [&](const std::string &id, const auto &values, double *reader_compute_seconds = nullptr) {
        const auto &family = family_for(id);
        if (values.size() != native_shape_product(family.logical_shape, id + ".logical")) fail("canonical witness logical value count mismatch: " + id);
        std::uint32_t encoding = 0U;
        if constexpr (std::is_same_v<std::decay_t<decltype(values)>, NativeValues>) encoding = values.encoding;
        if (!logical.emplace(id, NativeFamilyStorage::persist_generated(storage_directory, values.size(), encoding, [&](std::uint64_t index) { return values[index]; },
                                                                         reader_compute_seconds)).second) fail("duplicate canonical witness values: " + id);
    };
    const auto put_i64 = [&](const std::string &id, const auto &values) { put(id, native_i64_values(values)); };
    const auto put_u64 = [&](const std::string &id, const std::vector<u64> &values) { put(id, view(values.size(), 2U, [&values, id](std::uint64_t index) { if (values.at(index) > static_cast<u64>(std::numeric_limits<i64>::max())) fail("unsigned graph boundary exceeds field wire signed range: " + id); return native_wire(static_cast<i64>(values.at(index))); })); };
    const auto put_matrix = [&](const std::string &id, const Matrix &matrix) { put(id, native_matrix_values(matrix)); };
    const auto put_field = [&](const std::string &id, const std::vector<FieldValue> &values) { put(id, native_field_values(values)); };
    const auto put_field_matrix = [&](const std::string &id, const FieldMatrix &matrix) { put(id, native_field_matrix_values(matrix)); };
    const auto bits_for = [&](const std::vector<i64> &values, u64 width, const std::string &label) {
        if (width == 0U || width > 64U) fail("invalid bounded bit width: " + label);
        return view(values.size() * width, 1U, [&values, width, label](std::uint64_t index) { const auto value = values.at(index / width); if (value < 0 || (width < 63U && (static_cast<std::uint64_t>(value) >> width) != 0U)) fail("value exceeds bounded bit decomposition: " + label); return native_wire(static_cast<i64>((static_cast<std::uint64_t>(value) >> (index % width)) & 1U)); });
    };
    const auto group_new = native_group_new(graph);
    const auto group_end = native_group_end(graph);
    const auto edge_group_bits = [&](u64 width) { return native_repeat_edge_bits(group_new, width); };
    const auto edge_group_end_bits = [&](u64 width) { return native_repeat_edge_bits(group_end, width); };
    const auto node_vector_from_edges = [&](const std::vector<i64> &values) { return native_node_values_from_edges(graph, values); };

    put_matrix("input_features", graph.feature_values);
    put("sparse_route", view(graph.edges * 2U, 2U, [&](std::uint64_t index) { return native_wire(static_cast<i64>((index % 2U == 0U ? graph.src : graph.dst).at(index / 2U))); }));
    put("group_selectors", view(graph.edges * 2U, 1U, [&](std::uint64_t index) { return native_wire(static_cast<i64>((index % 2U == 0U ? group_new : group_end).at(index / 2U))); }));
    put_u64("node_ptr", graph.node_ptr);
    put_u64("edge_ptr", graph.edge_ptr);
    put_u64("node_edge_ptr", graph.node_edge_ptr);
    put_i64("static_lrelu", tables.lrelu);
    put_i64("static_elu", tables.elu);
    put_i64("static_exp", tables.exp);
    put_i64("output_head_count", std::vector<i64>{static_cast<i64>(config.architecture.output_heads)});

    const auto round_aux = [&](const SignedRoundDivWitness &witness) {
        if (witness.rows == 0U || witness.cols == 0U || witness.rows != graph.nodes) {
            fail("SignedRoundDiv auxiliary rows do not match canonical node domain");
        }
        const std::size_t count = static_cast<std::size_t>(witness.rows * witness.cols);
        if (witness.sigma.size() != count || witness.z.size() != count || witness.a.size() != count || witness.q.size() != count || witness.r.size() != count ||
            witness.delta.size() != count || witness.divisor.size() != count) {
            fail("SignedRoundDiv auxiliary witness is incomplete");
        }
        return view(count * 8U, 2U, [&witness](std::uint64_t index) { const auto cell = index / 8U; switch (index % 8U) { case 0U: return native_wire(static_cast<i64>(witness.sigma.at(cell))); case 1U: return native_wire(static_cast<i64>(witness.z.at(cell))); case 2U: return native_wire(witness.a.at(cell)); case 3U: return native_wire(witness.q.at(cell)); case 4U: return native_wire(witness.r.at(cell)); case 5U: return native_wire(witness.delta.at(cell)); case 6U: return native_wire(witness.divisor.at(cell)); default: return NativeFrWire{}; } });
    };
    const auto lookup_bits = [&](const std::vector<i64> &query, i64 minimum, i64 maximum, u64 width, const std::string &label) {
        if (width == 0U || width > 64U) fail("invalid lookup bit width: " + label);
        return view(query.size() * width, 1U, [&query, minimum, maximum, width, label](std::uint64_t index) { const auto value = query.at(index / width); if (value < minimum || value > maximum) fail("lookup query is outside authenticated table domain: " + label); const auto offset = static_cast<std::uint64_t>(checked_i128(static_cast<__int128>(value) - minimum, label + ".offset")); if (width < 63U && (offset >> width) != 0U) fail("lookup offset exceeds bounded bit decomposition: " + label); return native_wire(static_cast<i64>((offset >> (index % width)) & 1U)); });
    };

    const auto write_sorted_lookup_provider = [&](const std::string &operation, const std::vector<i64> &query, i64 input_min,
                                                  const std::vector<i64> &output_values, const std::vector<i64> &table) {
        const auto protocol_start = std::chrono::steady_clock::now();
        double explicit_file_io_seconds = 0.0;
        const auto plan_query = std::find_if(plan.sorted_lookup.queries.begin(), plan.sorted_lookup.queries.end(),
                                             [&](const SortedLookupQueryPlan &item) { return item.operation_id == operation; });
        if (plan_query == plan.sorted_lookup.queries.end()) fail("sorted lookup operation is absent from RelationPlan: " + operation);
        if (query.size() != output_values.size() || table.empty()) fail("sorted lookup Q/T row shape mismatch: " + operation);
        const i64 output_min = *std::min_element(table.begin(), table.end());
        const i64 output_max = *std::max_element(table.begin(), table.end());
        using U128 = unsigned __int128;
        struct LookupRow {
            U128 key = 0U;
            U128 x = 0U;
            U128 y = 0U;
            std::uint64_t table_count = 0U;
            std::uint64_t query_count = 0U;
        };
        std::map<U128, LookupRow> ordered;
        const U128 y_base = static_cast<U128>(1U) << plan_query->y_bits;
        const U128 key_limit = (static_cast<U128>(1U) << (plan_query->x_bits + plan_query->y_bits));
        const auto add_row = [&](i64 raw_x, i64 raw_y, std::uint8_t source) {
            if (raw_x < 0 || raw_y < 0) fail("sorted lookup packed coordinates are negative: " + operation);
            const U128 x = static_cast<U128>(raw_x);
            const U128 y = static_cast<U128>(raw_y);
            if ((x >= (static_cast<U128>(1U) << plan_query->x_bits)) || (y >= y_base) || ((x << plan_query->y_bits) | y) >= key_limit)
                fail("sorted lookup packed coordinate exceeds descriptor bounds: " + operation);
            const U128 key = (x << plan_query->y_bits) | y;
            auto &row = ordered[key];
            row.key = key; row.x = x; row.y = y;
            if (source == 1U) ++row.table_count; else ++row.query_count;
        };
        std::set<U128> table_keys;
        for (std::size_t index = 0; index < table.size(); ++index) {
            const i64 y = checked_i128(static_cast<__int128>(table[index]) - output_min, operation + ".table_y");
            const U128 x = static_cast<U128>(index);
            const U128 key = (x << plan_query->y_bits) | static_cast<U128>(y);
            if (!table_keys.insert(key).second) fail("authenticated sorted lookup table keys are not unique: " + operation);
            add_row(static_cast<i64>(index), y, 1U);
        }
        for (std::size_t index = 0; index < query.size(); ++index) {
            if (query[index] < input_min || static_cast<__int128>(query[index]) > static_cast<__int128>(input_min) + static_cast<__int128>(table.size()) - 1) {
                fail("sorted lookup query is outside authenticated table: " + operation);
            }
            const i64 x = checked_i128(static_cast<__int128>(query[index]) - input_min, operation + ".query_x");
            const i64 expected = table.at(static_cast<std::size_t>(x));
            if (output_values[index] != expected) fail("sorted lookup query/output is not table-bound: " + operation);
            const i64 y = checked_i128(static_cast<__int128>(output_values[index]) - output_min, operation + ".query_y");
            add_row(x, y, 2U);
        }
        if (ordered.empty() || plan_query->table_logical_rows != table.size() || plan_query->query_logical_rows != query.size())
            fail("sorted lookup provider logical lengths do not match RelationPlan: " + operation);

        const auto u128_decimal_local = [](U128 value) {
            if (value == 0U) return std::string("0");
            std::string result;
            while (value != 0U) {
                result.push_back(static_cast<char>('0' + static_cast<unsigned>(value % 10U)));
                value /= 10U;
            }
            std::reverse(result.begin(), result.end());
            return result;
        };
        constexpr std::size_t kScratchWriteBufferBytes = 8U * 1024U * 1024U;
        std::vector<char> output_buffer;
        output_buffer.reserve(kScratchWriteBufferBytes);
        const auto flush_buffer = [&](std::ofstream &stream) {
            if (output_buffer.empty()) return;
            const auto io_start = std::chrono::steady_clock::now();
            stream.write(output_buffer.data(), static_cast<std::streamsize>(output_buffer.size()));
            explicit_file_io_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - io_start).count();
            output_buffer.clear();
        };
        const auto append_bytes = [&](std::ofstream &stream, const char *bytes, std::size_t count) {
            while (count != 0U) {
                const std::size_t room = kScratchWriteBufferBytes - output_buffer.size();
                const std::size_t take = std::min(room, count);
                output_buffer.insert(output_buffer.end(), bytes, bytes + take);
                bytes += take;
                count -= take;
                if (output_buffer.size() == kScratchWriteBufferBytes) flush_buffer(stream);
            }
        };
        const auto write_u64 = [&](std::ofstream &stream, u64 value) {
            std::array<char, 8U> encoded{};
            for (unsigned bit = 0U; bit < encoded.size(); ++bit) encoded[bit] = static_cast<char>(value >> (8U * bit));
            append_bytes(stream, encoded.data(), encoded.size());
        };
        const auto write_u32 = [&](std::ofstream &stream, std::uint32_t value) {
            std::array<char, 4U> encoded{};
            for (unsigned bit = 0U; bit < encoded.size(); ++bit) encoded[bit] = static_cast<char>(value >> (8U * bit));
            append_bytes(stream, encoded.data(), encoded.size());
        };
        const auto write_u128 = [&](std::ofstream &stream, U128 value) {
            std::array<char, 16U> encoded{};
            for (unsigned bit = 0U; bit < encoded.size(); ++bit) encoded[bit] = static_cast<char>(value >> (8U * bit));
            append_bytes(stream, encoded.data(), encoded.size());
        };
        const auto write_string = [&](std::ofstream &stream, const std::string &value) {
            write_u64(stream, value.size());
            append_bytes(stream, value.data(), value.size());
        };
        const auto write_field = [&](std::ofstream &stream, const FieldValue &value) {
            append_bytes(stream, reinterpret_cast<const char *>(value.bytes.data()), value.bytes.size());
        };
        constexpr std::size_t kInverseCacheCapacity = 4096U;
        using InverseCacheEntry = std::pair<FieldValue, std::list<U128>::iterator>;
        std::map<U128, InverseCacheEntry> inverse_cache;
        std::list<U128> inverse_lru;
        const auto inverse_for = [&](U128 delta, BN_CTX *context) {
            if (delta == 0U) return FieldValue{};
            if (delta > static_cast<U128>(std::numeric_limits<i64>::max()))
                fail("sorted lookup delta exceeds native inverse input bound: " + operation);
            const auto cached = inverse_cache.find(delta);
            if (cached != inverse_cache.end()) {
                inverse_lru.splice(inverse_lru.begin(), inverse_lru, cached->second.second);
                return cached->second.first;
            }
            const FieldValue result = field_encode(field_inverse(static_cast<i64>(delta), context).get());
            inverse_lru.push_front(delta);
            inverse_cache.emplace(delta, InverseCacheEntry{result, inverse_lru.begin()});
            if (inverse_cache.size() > kInverseCacheCapacity) {
                const U128 evicted = inverse_lru.back();
                inverse_lru.pop_back();
                inverse_cache.erase(evicted);
            }
            return result;
        };
        const auto write_record = [&](std::ofstream &stream, U128 key, U128 delta, U128 x, U128 y, u64 prefix, std::uint8_t active, std::uint8_t marker,
                                      std::uint8_t source, BN_CTX *context) {
            write_u128(stream, key);
            write_u128(stream, delta);
            write_field(stream, inverse_for(delta, context));
            write_u128(stream, x);
            write_u128(stream, y);
            write_u64(stream, prefix);
            const std::array<char, 3U> flags{{
                static_cast<char>(active), static_cast<char>(marker),
                static_cast<char>(source)}};
            append_bytes(stream, flags.data(), flags.size());
        };
        const auto write_zero = [&](std::ofstream &stream, BN_CTX *context) { write_record(stream, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, context); };
        const auto scratch_dir = config.run_root / "scratch" / "sorted_lookup";
        const auto directory_start = std::chrono::steady_clock::now();
        std::filesystem::create_directories(scratch_dir);
        explicit_file_io_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - directory_start).count();
        const auto basename = "lookup_" + sha256_text(operation) + ".bin";
        const auto final_path = scratch_dir / basename;
        const auto temp_path = scratch_dir /
                               ("." + basename + ".tmp-" +
                                std::to_string(static_cast<unsigned long long>(std::chrono::high_resolution_clock::now().time_since_epoch().count())));
        const auto open_start = std::chrono::steady_clock::now();
        std::ofstream stream(temp_path, std::ios::binary | std::ios::trunc);
        explicit_file_io_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - open_start).count();
        if (!stream) fail("cannot create sorted lookup scratch: " + operation);
        BN_CTX *context = BN_CTX_new();
        if (!context) fail("BN_CTX_new failed for sorted lookup scratch");
        try {
            append_bytes(stream, "ZKDLQTS1", 8U);
            write_u32(stream, 1U);
            write_string(stream, operation);
            write_string(stream, plan_query->commitment_role_map);
            write_u64(stream, plan_query->table_domain);
            write_u64(stream, plan_query->query_domain);
            write_u64(stream, plan_query->merge_domain);
            write_u64(stream, plan_query->segment_size);
            write_u64(stream, plan_query->table_logical_rows);
            write_u64(stream, plan_query->query_logical_rows);
            write_u64(stream, plan_query->merge_logical_rows);
            write_u32(stream, plan_query->x_bits);
            write_u32(stream, plan_query->y_bits);
            write_string(stream, plan_query->key_max_u128);
            write_u64(stream, table.size());
            for (std::size_t index = 0; index < table.size(); ++index) {
                const U128 x = static_cast<U128>(index);
                const U128 y = static_cast<U128>(table[index] - output_min);
                write_record(stream, (x << plan_query->y_bits) | y, 0U, x, y, static_cast<u64>(index + 1U), 1U, 1U, 1U, context);
            }
            for (u64 index = table.size(); index < plan_query->table_domain; ++index) write_zero(stream, context);
            write_u64(stream, query.size());
            for (std::size_t index = 0; index < query.size(); ++index) {
                const U128 x = static_cast<U128>(query[index] - input_min);
                const U128 y = static_cast<U128>(output_values[index] - output_min);
                write_record(stream, (x << plan_query->y_bits) | y, 0U, x, y, static_cast<u64>(index + 1U), 1U, 1U, 2U, context);
            }
            for (u64 index = query.size(); index < plan_query->query_domain; ++index) write_zero(stream, context);
            write_u64(stream, plan_query->merge_logical_rows);
            bool previous = false;
            U128 previous_key = 0U;
            u64 prefix = 0U;
            u64 emitted = 0U;
            for (const auto &[key, row] : ordered) {
                for (u64 index = 0U; index < row.table_count + row.query_count; ++index) {
                    const U128 delta = previous ? key - previous_key : 0U;
                    const std::uint8_t marker = (!previous || key != previous_key) ? 1U : 0U;
                    if (marker) ++prefix;
                    write_record(stream, key, delta, row.x, row.y, prefix, 1U, marker, index < row.table_count ? 1U : 2U, context);
                    previous = true;
                    previous_key = key;
                    ++emitted;
                }
            }
            if (emitted != plan_query->merge_logical_rows) fail("sorted lookup merge stream length mismatch: " + operation);
            for (u64 index = emitted; index < plan_query->merge_domain; ++index) write_zero(stream, context);
            flush_buffer(stream);
            const auto flush_start = std::chrono::steady_clock::now();
            stream.flush();
            explicit_file_io_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - flush_start).count();
            if (!stream) fail("sorted lookup scratch write failed: " + operation);
            const auto close_start = std::chrono::steady_clock::now();
            stream.close();
            explicit_file_io_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - close_start).count();
            if (!stream) fail("sorted lookup scratch close failed: " + operation);
            BN_CTX_free(context);
            context = nullptr;
            std::error_code remove_error;
            const auto publish_start = std::chrono::steady_clock::now();
            std::filesystem::remove(final_path, remove_error);
            std::filesystem::rename(temp_path, final_path);
            std::filesystem::permissions(final_path, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                         std::filesystem::perm_options::replace);
            explicit_file_io_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - publish_start).count();
        } catch (...) {
            if (context) BN_CTX_free(context);
            stream.close();
            std::error_code cleanup_error;
            std::filesystem::remove(temp_path, cleanup_error);
            throw;
        }
        NativeSortedLookupProviderInput provider;
        provider.operation_id = operation;
        provider.table_ref = plan_query->table_ref;
        provider.query_ref = plan_query->query_ref;
        provider.merge_ref = plan_query->merge_ref;
        provider.scratch_path = std::filesystem::relative(final_path, config.baseline_root).generic_string();
        const auto digest_start = std::chrono::steady_clock::now();
        provider.scratch_digest = sha256_file(final_path);
        explicit_file_io_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - digest_start).count();
        provider.layout_digest = plan_query->layout_digest;
        std::stringstream role_stream(plan_query->commitment_role_map);
        for (std::string role; std::getline(role_stream, role, ',');) provider.commitment_roles.push_back(role);
        provider.table_logical_rows = plan_query->table_logical_rows;
        provider.table_domain = plan_query->table_domain;
        provider.query_logical_rows = plan_query->query_logical_rows;
        provider.query_domain = plan_query->query_domain;
        provider.merge_logical_rows = plan_query->merge_logical_rows;
        provider.merge_domain = plan_query->merge_domain;
        provider.segment_size = plan_query->segment_size;
        provider.x_bits = plan_query->x_bits;
        provider.y_bits = plan_query->y_bits;
        provider.key_max_u128 = plan_query->key_max_u128;
        sorted_lookup_providers.push_back(std::move(provider));
        const double protocol_wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - protocol_start).count();
        if (explicit_file_io_seconds < 0.0 || explicit_file_io_seconds > protocol_wall_seconds)
            fail("sorted lookup file I/O timing exceeded protocol auxiliary construction interval: " + operation);
        sorted_lookup_protocol_construction_seconds += protocol_wall_seconds - explicit_file_io_seconds;
    };

    const auto put_stage_base = [&](const std::string &stage, u64 layer, u64 head, bool output) {
        const u64 width = output ? config.architecture.classes : config.architecture.hidden_dims.at(layer);
        const Matrix &input = output ? trace.hidden_layers.back().concat : trace.hidden_layers.at(layer).input;
        put_matrix(stage + "/trace/input", input);
        const Matrix weight = quantized_tensor_matrix(model, stage + "/seq/kernel",
                                                      output ? config.quantization.output_weight.at(head) :
                                                               config.quantization.hidden_weight.at(layer).at(head));
        const std::vector<i64> src_kernel = quantized_tensor_vector(model, stage + "/attn_src/kernel",
                                                                    output ? config.quantization.output_attention_src.at(head) :
                                                                             config.quantization.hidden_attention_src.at(layer).at(head));
        const std::vector<i64> dst_kernel = quantized_tensor_vector(model, stage + "/attn_dst/kernel",
                                                                    output ? config.quantization.output_attention_dst.at(head) :
                                                                             config.quantization.hidden_attention_dst.at(layer).at(head));
        put_matrix(stage + "/model/seq_kernel", weight);
        put_i64(stage + "/model/attn_src", src_kernel);
        put_i64(stage + "/model/attn_dst", dst_kernel);
        if (output) { put_i64(stage + "/model/output_bias", quantized_tensor_vector(model, stage + "/output_bias", config.quantization.output_bias.at(head))); }

        const auto put_common = [&](const auto &head_trace, const Matrix &projected, const std::string &name) {
            put_matrix(stage + "/trace/" + name, projected);
            put_i64(stage + "/trace/e_src", head_trace.e_src);
            put_i64(stage + "/trace/e_dst", head_trace.e_dst);
            put_i64(stage + "/trace/score", head_trace.score);
            put_i64(stage + "/trace/leaky_relu", head_trace.leaky_relu);
            put_i64(stage + "/trace/delta", head_trace.delta);
            put_i64(stage + "/trace/exp_weight", head_trace.exp_weight);
            put_field(stage + "/trace/inverse_field", head_trace.inverse_field);
            put_field(stage + "/trace/alpha_field", head_trace.alpha_field);
            put_i64(stage + "/trace/maximum", head_trace.maximum);
            put_i64(stage + "/trace/denominator", node_vector_from_edges(head_trace.denominator));
            std::vector<i64> max_selector(graph.edges, 0);
            for (u64 node = 0U; node < graph.nodes; ++node) {
                bool selected = false;
                const u64 begin = graph.node_edge_ptr.at(static_cast<std::size_t>(node));
                const u64 end = graph.node_edge_ptr.at(static_cast<std::size_t>(node + 1U));
                for (u64 edge = begin; edge < end; ++edge) {
                    if (!selected && head_trace.leaky_relu.at(static_cast<std::size_t>(edge)) == head_trace.maximum.at(static_cast<std::size_t>(node))) {
                        max_selector[edge] = 1;
                        selected = true;
                    }
                }
                if (!selected) fail("max selector has no selected edge for destination group");
            }
            put_i64(stage + "/trace/max_selector", max_selector);
            put(stage + "/trace/range_bits", bits_for(head_trace.delta, table_index_bit_width(tables.exp.size()), stage + ".delta"));
            put_i64(stage + "/trace/lrelu_idx", head_trace.score);
            put_i64(stage + "/trace/exp_idx", head_trace.delta);
            put(stage + "/trace/lrelu_bits", lookup_bits(head_trace.score, tables.lrelu_min, tables.lrelu_max,
                                                           table_index_bit_width(tables.lrelu.size()), stage + ".lrelu"));
            put(stage + "/trace/exp_bits", lookup_bits(head_trace.delta, 0, static_cast<i64>(tables.exp_max),
                                                        table_index_bit_width(tables.exp.size()), stage + ".exp"));
            put(stage + "/trace/rounddiv_aux", round_aux(head_trace.aggregate_round_div));
            put_field(stage + "/trace/rounddiv_a_inverse", head_trace.aggregate_round_div.a_inverse);
            put_field(stage + "/trace/rounddiv_divisor_inverse", head_trace.aggregate_round_div.divisor_inverse);
            put_matrix(stage + "/trace/aggregate_num", head_trace.aggregate_num);
            put_field_matrix(stage + "/trace/aggregate_pre_field", head_trace.aggregate_pre_field);
            put_field_matrix(stage + "/trace/consistency_residual_field", head_trace.consistency_residual_field);
        };

        if (!output) {
            const auto &head_trace = trace.hidden_layers.at(layer).heads.at(head);
            put_common(head_trace, head_trace.h_prime, "h_prime");
            put_matrix(stage + "/trace/routed_h_prime", head_trace.routed_h_prime);
            put_matrix(stage + "/trace/product_u_h_prime", head_trace.product_u_h_prime);
            put_field_matrix(stage + "/trace/product_alpha_h_prime_field", head_trace.product_alpha_h_prime_field);
            put_matrix(stage + "/trace/aggregate_pre", head_trace.aggregate_pre);
            put_matrix(stage + "/trace/elu_input", head_trace.elu_input);
            put_matrix(stage + "/trace/aggregate", head_trace.aggregate);
            put_i64(stage + "/trace/elu_idx", head_trace.aggregate_pre.values.empty() ? std::vector<i64>(graph.nodes, 0) : head_trace.aggregate_pre.values);
            put(stage + "/trace/elu_bits", lookup_bits(head_trace.aggregate_pre.values, tables.elu_min, tables.elu_max,
                                                         table_index_bit_width(tables.elu.size()), stage + ".elu"));
        } else {
            const auto &head_trace = trace.output_heads.at(head);
            put_common(head_trace, head_trace.y_prime, "y_prime");
            put_matrix(stage + "/trace/routed_y_prime", head_trace.routed_y_prime);
            put_matrix(stage + "/trace/product_u_y_prime", head_trace.product_u_y_prime);
            put_field_matrix(stage + "/trace/product_alpha_y_prime_field", head_trace.product_alpha_y_prime_field);
            put_matrix(stage + "/trace/aggregate_pre", head_trace.aggregate_pre);
            put_matrix(stage + "/trace/y_quantized", head_trace.y_quantized);
            put_matrix(stage + "/trace/aligned", head_trace.aligned);
            put_matrix(stage + "/trace/output", head_trace.output);
        }
        (void)width;
    };

    for (u64 layer = 0U; layer < config.architecture.hidden_dims.size(); ++layer) {
        for (u64 head = 0U; head < config.architecture.hidden_heads.at(layer); ++head)
            put_stage_base("hidden/layer" + std::to_string(layer) + "/head" + std::to_string(head), layer, head, false);
        const auto &concat = trace.hidden_layers.at(layer).concat;
        const u64 head_count = config.architecture.hidden_heads.at(layer);
        const u64 width = config.architecture.hidden_dims.at(layer);
        put("hidden/layer" + std::to_string(layer) + "/concat", native_matrix_values(concat));
    }
    for (u64 head = 0U; head < config.architecture.output_heads; ++head)
        put_stage_base("output/head" + std::to_string(head), config.architecture.hidden_dims.size(), head, true);
    put("output_heads", view(graph.nodes * config.architecture.output_heads * config.architecture.classes, 2U, [&](std::uint64_t index) { const auto cls = index % config.architecture.classes, row = index / config.architecture.classes, head = row % config.architecture.output_heads, node = row / config.architecture.output_heads; return native_wire(trace.output_heads.at(head).output.at(node, cls)); }));
    put_matrix("output_sum", trace.output_sum);
    put_matrix("public_Y", trace.logits);

    const auto put_round_views = [&](const std::string &operation, const SignedRoundDivWitness &witness, const Matrix &numerator, u64 numerator_scale,
                                     const Matrix &quotient, const std::vector<i64> *raw_denominator_nodes) {
        proof_auxiliary_construction_seconds += witness.proof_auxiliary_construction_seconds;
        const auto validation_start = std::chrono::steady_clock::now();
        if (witness.rows != numerator.rows || witness.cols != numerator.cols || quotient.rows != numerator.rows || quotient.cols != numerator.cols) {
            fail("SignedRoundDiv view shape does not match its canonical matrices: " + operation);
        }
        const std::size_t count = numerator.values.size();
        if (witness.sigma.size() != count || witness.z.size() != count || witness.a.size() != count || witness.q.size() != count || witness.r.size() != count ||
            witness.delta.size() != count || witness.divisor.size() != count || witness.a_inverse.size() != count || witness.divisor_inverse.size() != count) {
            fail("SignedRoundDiv view witness is incomplete: " + operation);
        }
        const auto &divisor_family = family_for(operation + "/view/divisor");
        const u64 padded_width = divisor_family.padded_shape.size() == 1U ? 1U : divisor_family.padded_shape[1];
        const auto bounds = signed_round_bounds::derive(config, plan.integer_bound_ledger, operation);
        if ((divisor_family.logical_shape.size() != 1U && divisor_family.logical_shape.size() != 2U) ||
            divisor_family.logical_shape[0] != witness.rows ||
            (divisor_family.logical_shape.size() == 2U && divisor_family.logical_shape[1] != padded_width) ||
            padded_width < witness.cols || witness.rows > std::numeric_limits<u64>::max() / padded_width) {
            fail("SignedRoundDiv full-feature divisor view has an invalid authenticated shape: " + operation);
        }
        for (u64 row = 0U; row < witness.rows; ++row) {
            const i64 row_divisor = witness.divisor.at(static_cast<std::size_t>(row * witness.cols));
            for (u64 feature = 1U; feature < witness.cols; ++feature)
                if (witness.divisor.at(static_cast<std::size_t>(row * witness.cols + feature)) != row_divisor)
                    fail("SignedRoundDiv divisor is not a row scalar before padded broadcast: " + operation);
        }
        const std::vector<std::tuple<std::string, const std::vector<i64> *, u64, i64>> range_roles = {
            {"a", &witness.a, bounds.a_bits, bounds.a_max}, {"q_abs", &witness.q, bounds.q_bits, bounds.q_max},
            {"r", &witness.r, bounds.r_bits, bounds.r_max}, {"delta", &witness.delta, bounds.delta_bits, bounds.delta_max},
            {"divisor", &witness.divisor, bounds.divisor_bits, bounds.divisor_max},
        };
        for (const auto &role : range_roles) {
            const u64 width = std::get<2>(role);
            const i64 maximum = std::get<3>(role);
            for (const i64 value : *std::get<1>(role))
                if (value < 0 || value > maximum || (width < 63U && (static_cast<u64>(value) >> width) != 0U))
                    fail("SignedRoundDiv witness exceeds authenticated role bound: " + operation + "/" + std::get<0>(role));
        }
        if (bounds.binds_raw_denominator) {
            if (raw_denominator_nodes == nullptr || raw_denominator_nodes->size() != witness.rows ||
                witness.rows > std::numeric_limits<u64>::max() / padded_width)
                fail("SignedRoundDiv raw denominator source is missing or malformed: " + operation);
            for (u64 row = 0U; row < witness.rows; ++row) {
                const i64 raw = raw_denominator_nodes->at(static_cast<std::size_t>(row));
                const i64 scaled = checked_i128(static_cast<__int128>(raw) * bounds.denominator_scale, operation + ".scaled_divisor");
                if (raw <= 0 || witness.divisor.at(static_cast<std::size_t>(row * witness.cols)) != scaled)
                    fail("SignedRoundDiv raw/scaled divisor witness mismatch: " + operation);
            }
            const auto &scaled_family = family_for(operation + "/view/divisor_aux");
            if (scaled_family.logical_shape.size() != 2U || scaled_family.logical_shape[0] != witness.rows ||
                scaled_family.logical_shape[1] != padded_width)
                fail("SignedRoundDiv scaled divisor auxiliary view has an invalid authenticated shape: " + operation);
        } else if (raw_denominator_nodes != nullptr) {
            fail("SignedRoundDiv unexpectedly received a raw denominator for a public-divisor operation: " + operation);
        }
        proof_auxiliary_construction_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - validation_start).count();
        put(operation + "/view/signed_numerator", view(count, 2U, [&numerator, numerator_scale, operation](std::uint64_t index) { return native_wire(checked_i128(static_cast<__int128>(numerator.values.at(index)) * numerator_scale, operation + ".signed_numerator")); }), &proof_auxiliary_construction_seconds);
        if (bounds.binds_raw_denominator) {
            put(operation + "/view/divisor", view(witness.rows * padded_width, 2U,
                 [raw_denominator_nodes, padded_width](std::uint64_t index) {
                     return native_wire(raw_denominator_nodes->at(static_cast<std::size_t>(index / padded_width)));
                 }), &proof_auxiliary_construction_seconds);
            put(operation + "/view/divisor_aux", view(witness.rows * padded_width, 2U,
                 [&witness, padded_width](std::uint64_t index) {
                     const u64 row = index / padded_width;
                     return native_wire(witness.divisor.at(static_cast<std::size_t>(row * witness.cols)));
                 }), &proof_auxiliary_construction_seconds);
        } else {
            put(operation + "/view/divisor", view(native_shape_product(divisor_family.logical_shape, operation + ".divisor.logical"), 2U,
                 [&witness, padded_width](std::uint64_t index) {
                     const u64 row = index / padded_width;
                     return native_wire(witness.divisor.at(static_cast<std::size_t>(row * witness.cols)));
                 }), &proof_auxiliary_construction_seconds);
        }
        put(operation + "/view/signed_quotient", native_matrix_values(quotient), &proof_auxiliary_construction_seconds);
        put(operation + "/view/a", native_i64_values(witness.a), &proof_auxiliary_construction_seconds);
        put(operation + "/view/q_abs", native_i64_values(witness.q), &proof_auxiliary_construction_seconds);
        put(operation + "/view/r", native_i64_values(witness.r), &proof_auxiliary_construction_seconds);
        put(operation + "/view/delta", native_i64_values(witness.delta), &proof_auxiliary_construction_seconds);
        put(operation + "/view/sigma", native_i64_values(witness.sigma), &proof_auxiliary_construction_seconds);
        put(operation + "/view/z", native_i64_values(witness.z), &proof_auxiliary_construction_seconds);
        put(operation + "/view/a_inverse", native_field_values(witness.a_inverse), &proof_auxiliary_construction_seconds);
        put(operation + "/view/divisor_inverse", native_field_values(witness.divisor_inverse), &proof_auxiliary_construction_seconds);
        for (const auto &role : range_roles)
            for (u64 bit = 0U; bit < std::get<2>(role); ++bit)
                put(operation + "/view/range/" + std::get<0>(role) + "/" + std::to_string(bit),
                    native_bit_column(*std::get<1>(role), bit, operation + "." + std::get<0>(role)), &proof_auxiliary_construction_seconds);
    };

    const auto put_group_views = [&](const std::string &stage, u64 width, bool output, u64 aggregate_scale, const auto &head_trace) {
        using HeadTrace = std::decay_t<decltype(head_trace)>;
        constexpr bool is_hidden = std::is_same_v<HeadTrace, HiddenHeadTrace>;
        u64 padded_width = 1U;
        while (padded_width < width) padded_width <<= 1U;
        const auto edge_width = native_edge_feature_from_scalar(head_trace.exp_weight, padded_width);
        const auto pu = [&]() {
            if constexpr (is_hidden) {
                return native_matrix_values(head_trace.product_u_h_prime);
            } else {
                return native_matrix_values(head_trace.product_u_y_prime);
            }
        }();
        const auto group_new_width = edge_group_bits(padded_width);
        const auto group_end_width = edge_group_end_bits(padded_width);
        const auto denominator_nodes = node_vector_from_edges(head_trace.denominator);
        const auto denominator_edges = native_broadcast_node_vector(graph, denominator_nodes);
        const auto denominator_broadcast = view(graph.edges * padded_width, 2U, [denominator_edges, padded_width](std::uint64_t index) { return denominator_edges[index / padded_width]; });
        const std::string group = stage + "/group_sum";
        put(group + "/view/group_new", group_new_width);
        put(group + "/view/group_end", group_end_width);
        put(group + "/view/input0", edge_width);
        put(group + "/view/input1", pu);
        put(group + "/view/output0", denominator_broadcast);
        put(group + "/view/output1", native_broadcast_node_matrix(graph, head_trace.aggregate_num));
        put(group + "/view/prefix0", native_prefix(edge_width, group_new, padded_width, false));
        put(group + "/view/prefix_prev0", native_prefix(edge_width, group_new, padded_width, true));
        put(group + "/view/prefix1", native_prefix(pu, group_new, width, false));
        put(group + "/view/prefix_prev1", native_prefix(pu, group_new, width, true));

        const std::string inverse = stage + "/inverse_alpha";
        put(inverse + "/view/input0", denominator_edges);
        put(inverse + "/view/input1", native_i64_values(head_trace.exp_weight));
        put(inverse + "/view/output0", native_field_values(head_trace.inverse_field));
        put(inverse + "/view/output1", native_field_values(head_trace.alpha_field));

        const auto put_segmented = [&](const std::string &operation, NativeValues input, NativeValues output_values) {
            put(operation + "/view/group_new", group_new_width);
            put(operation + "/view/group_end", group_end_width);
            put(operation + "/view/input0", input);
            put(operation + "/view/output0", output_values);
            put(operation + "/view/prefix0", native_prefix(input, group_new, width, false));
            put(operation + "/view/prefix_prev0", native_prefix(input, group_new, width, true));
        };
        const auto field_product = [&]() {
            if constexpr (is_hidden) {
                return native_field_matrix_values(head_trace.product_alpha_h_prime_field);
            } else {
                return native_field_matrix_values(head_trace.product_alpha_y_prime_field);
            }
        }();
        put_segmented(stage + "/field_aggregate", field_product, native_broadcast_node_field_matrix(graph, head_trace.aggregate_pre_field));
        put_segmented(stage + "/integer_numerator", pu, native_broadcast_node_matrix(graph, head_trace.aggregate_num));

        const std::string consistency = stage + "/consistency";
        put(consistency + "/view/input0", native_broadcast_node_scalar_features(graph, denominator_nodes, padded_width));
        put(consistency + "/view/input1", native_field_matrix_values(head_trace.aggregate_pre_field));
        put(consistency + "/view/input2", native_matrix_values(head_trace.aggregate_num));
        put(consistency + "/view/output0", native_field_matrix_values(head_trace.consistency_residual_field));
        const std::string alpha_product = stage + (output ? "/alpha_y_prime" : "/alpha_h_prime");
        put(alpha_product + "/view/input0", native_edge_feature_from_field_scalar(head_trace.alpha_field, padded_width));

        const auto put_max_views = [&]() {
            const std::string max = stage + "/max_range";
            put_i64(max + "/view/value", head_trace.leaky_relu);
            put(max + "/view/maximum_broadcast", native_broadcast_node_vector(graph, head_trace.maximum));
            std::vector<i64> max_selector(graph.edges, 0);
            for (u64 node = 0U; node < graph.nodes; ++node) {
                bool selected = false;
                for (u64 edge = graph.node_edge_ptr.at(static_cast<std::size_t>(node)); edge < graph.node_edge_ptr.at(static_cast<std::size_t>(node + 1U));
                     ++edge) {
                    if (!selected && head_trace.leaky_relu.at(static_cast<std::size_t>(edge)) == head_trace.maximum.at(static_cast<std::size_t>(node))) {
                        max_selector[edge] = 1;
                        selected = true;
                    }
                }
            }
            put_i64(max + "/view/selector", max_selector);
            put(max + "/view/delta", native_i64_values(head_trace.delta));
            put(max + "/view/group_new", native_i64_values(group_new));
            put(max + "/view/group_end", native_i64_values(group_end));
            const auto selector_prefix = [&](bool previous) { return view(graph.edges, 2U, [&max_selector, &group_new, previous, running = i64{0}](std::uint64_t index) mutable { if (index == 0U || group_new.at(index) != 0U) running = 0; const auto prior = running; running += max_selector.at(index); return native_wire(previous ? prior : running); }); };
            put(max + "/view/selector_prefix", selector_prefix(false));
            put(max + "/view/selector_prev", selector_prefix(true));
            for (u64 bit = 0U; bit < table_index_bit_width(tables.exp.size()); ++bit)
                put(max + "/view/range_bit/" + std::to_string(bit), native_bit_column(head_trace.delta, bit, max + ".range" + std::to_string(bit)));
        };
        put_max_views();

        const auto put_table = [&](const std::string &operation, const std::vector<i64> &query, i64 minimum, i64 maximum, const std::vector<i64> &output_values,
                                   const std::vector<i64> &table) {
            put_i64(operation + "/view/query", query);
            put_i64(operation + "/view/index", query);
            put_i64(operation + "/view/output0", output_values);
            write_sorted_lookup_provider(operation, query, minimum, output_values, table);
            u64 depth = 0U;
            std::size_t table_width = 1U;
            while (table_width < table.size()) {
                table_width <<= 1U;
                ++depth;
            }
            for (u64 bit = 0U; bit < depth; ++bit) put(operation + "/view/bit/" + std::to_string(bit), view(query.size(), 1U, [&query, minimum, maximum, bit, operation](std::uint64_t index) { const auto value = query.at(index); if (value < minimum || value > maximum) fail("lookup view query outside authenticated table: " + operation); const auto offset = static_cast<std::uint64_t>(checked_i128(static_cast<__int128>(value) - minimum, operation + ".offset")); return native_wire(static_cast<i64>((offset >> bit) & 1U)); }));
        };
        put_table(stage + "/leaky_relu", head_trace.score, tables.lrelu_min, tables.lrelu_max, head_trace.leaky_relu, tables.lrelu);
        put_table(stage + "/exp_map", head_trace.delta, 0, static_cast<i64>(tables.exp_max), head_trace.exp_weight, tables.exp);
        if constexpr (is_hidden)
            put_table(stage + "/elu", head_trace.aggregate_pre.values, tables.elu_min, tables.elu_max, head_trace.aggregate.values, tables.elu);

        const std::string inner = stage + (output ? "/output_attention_inner_product" : "/attention_inner_product");
        const Matrix &projected = [&]() -> const Matrix & {
            if constexpr (is_hidden) {
                return head_trace.h_prime;
            } else {
                return head_trace.y_prime;
            }
        }();
        put(inner + "/view/src_matrix", native_route_matrix(graph, projected, true));
        put(inner + "/view/dst_matrix", native_route_matrix(graph, projected, false));
        put_i64(inner + "/view/src_output", head_trace.e_src);
        put_i64(inner + "/view/dst_output", head_trace.e_dst);

        const std::string round = stage + "/signed_round_div";
        const Matrix &rounded = [&]() -> const Matrix & {
            if constexpr (is_hidden) {
                return head_trace.aggregate_pre;
            } else {
                return head_trace.y_quantized;
            }
        }();
        put_round_views(round, head_trace.aggregate_round_div, head_trace.aggregate_num, aggregate_scale, rounded, &denominator_nodes);
        if constexpr (!is_hidden) {
            put_round_views(stage + "/output_scale_alignment", head_trace.alignment_round_div, head_trace.y_quantized, config.quantization.output_head,
                            head_trace.aligned, nullptr);
            const std::string bias = stage + "/output_bias";
            put(bias + "/view/input0", native_matrix_values(head_trace.aligned));
            put(bias + "/view/input1", view(graph.nodes * width, 2U, [&head_trace, width](std::uint64_t index) { return native_wire(head_trace.bias.at(index % width)); }));
            put(bias + "/view/output0", native_matrix_values(head_trace.output));
        }
    };

    for (u64 layer = 0U; layer < config.architecture.hidden_dims.size(); ++layer) {
        const u64 width = config.architecture.hidden_dims.at(layer);
        const u64 aggregate_scale = config.quantization.hidden_aggregate_pre.at(layer);
        for (u64 head = 0U; head < config.architecture.hidden_heads.at(layer); ++head) {
            const std::string stage = "hidden/layer" + std::to_string(layer) + "/head" + std::to_string(head);
            put_group_views(stage, width, false, aggregate_scale, trace.hidden_layers.at(layer).heads.at(head));
        }
        const std::string concat = "hidden/layer" + std::to_string(layer) + "/concat";
        const u64 head_count = config.architecture.hidden_heads.at(layer);
        const Matrix &concat_matrix = trace.hidden_layers.at(layer).concat;
        for (u64 head = 0U; head < head_count; ++head) {
            const Matrix &source = trace.hidden_layers.at(layer).heads.at(head).aggregate;
            if (source.rows != graph.nodes || source.cols != width) fail("hidden concat source shape mismatch");
            put(concat + "/view/input" + std::to_string(head), view(graph.nodes * head_count * width, 2U, [&source, head, head_count, width](std::uint64_t index) { const auto column = index % width, row = index / width; return row % head_count == head ? native_wire(source.at(row / head_count, column)) : NativeFrWire{}; }));
        }
        put(concat + "/view/output", native_matrix_values(concat_matrix));
    }
    for (u64 head = 0U; head < config.architecture.output_heads; ++head) {
        const std::string stage = "output/head" + std::to_string(head);
        put_group_views(stage, config.architecture.classes, true, config.quantization.output_pre, trace.output_heads.at(head));
    }
    put_round_views("final_average", trace.final_average_round_div, trace.output_sum, 1U, trace.logits, nullptr);

    NativeRelationWitnessInput result;
    result.tensors.reserve(plan.tensor_families.size());
    result.tensor_shapes.reserve(plan.tensor_families.size());
    for (const auto &family : plan.tensor_families) {
        const auto found = logical.find(family.id);
        if (found == logical.end()) fail("canonical native witness is missing tensor values: " + family.id);
        NativeTensorWitnessInput tensor;
        tensor.id = family.id;
        tensor.visibility = family.visibility == "public" ? 1U : (family.id.find("/model/") != std::string::npos ? 2U : 3U);
        tensor.domain_size = checked_native_domain_u32(native_shape_product(family.padded_shape, family.id + ".domain"), family.id + ".domain");
        if (family.id == "public_Y") {
            const std::uint64_t maximum_size_t_elements =
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(NativeFrWire));
            if (tensor.domain_size > maximum_size_t_elements || tensor.domain_size > tensor.values.max_size())
                fail("BLOCKED: public_Y padded witness does not fit addressable vector capacity");
            tensor.values.reserve(static_cast<std::size_t>(tensor.domain_size));
            std::string storage_error;
            for (std::uint64_t offset = 0U; offset < tensor.domain_size; offset += (1U << 18U)) { std::vector<NativeFrWire> block; const auto count = checked_native_domain_u32(std::min<std::uint64_t>(1U << 18U, tensor.domain_size - offset), family.id + ".public_Y.block"); if (!found->second->read_padded(family.logical_shape, family.padded_shape, offset, count, block, &storage_error)) fail(storage_error); tensor.values.insert(tensor.values.end(), block.begin(), block.end()); }
        } else { tensor.storage = found->second; tensor.logical_shape = family.logical_shape; tensor.padded_shape = family.padded_shape; }
        result.tensors.push_back(std::move(tensor));

        NativeTensorShapeInput shape;
        shape.id = family.id;
        shape.domain_size = checked_native_domain_u32(native_shape_product(family.padded_shape, family.id + ".domain"), family.id + ".domain");
        shape.visibility = result.tensors.back().visibility;
        shape.logical_shape = family.logical_shape;
        shape.padded_shape = family.padded_shape;
        shape.mle_order = family.mle_order;
        shape.stack_axis = family.stack_axis;
        shape.mapping = family.mapping;
        shape.padding_rule = family.padding_rule;
        if (shape.visibility == 2U) shape.registry_id = "model_registry:" + family.id;
        result.tensor_shapes.push_back(std::move(shape));
    }

    if (sorted_lookup_providers.size() != plan.sorted_lookup.queries.size()) fail("native witness sorted-lookup provider count does not match RelationPlan");
    result.sorted_lookup_protocol_construction_seconds = sorted_lookup_protocol_construction_seconds;
    result.proof_auxiliary_construction_seconds = proof_auxiliary_construction_seconds;
    result.sorted_lookup_providers = std::move(sorted_lookup_providers);

    populate_authenticated_resource_wire(config, result, true);
    return result;
}

NativeRelationWitnessInput build_native_public_context(const Config &config, const RelationPlan &plan, const Matrix &public_output) {
    NativeRelationWitnessInput result;
    result.tensor_shapes.reserve(plan.tensor_families.size());
    for (const auto &family : plan.tensor_families) {
        NativeTensorShapeInput shape;
        shape.id = family.id;
        shape.domain_size = checked_native_domain_u32(native_shape_product(family.padded_shape, family.id + ".domain"), family.id + ".domain");
        shape.visibility = family.visibility == "public" ? 1U : (family.id.find("/model/") != std::string::npos ? 2U : 3U);
        shape.logical_shape = family.logical_shape;
        shape.padded_shape = family.padded_shape;
        shape.mle_order = family.mle_order;
        shape.stack_axis = family.stack_axis;
        shape.mapping = family.mapping;
        shape.padding_rule = family.padding_rule;
        if (shape.visibility == 2U) shape.registry_id = "model_registry:" + family.id;
        result.tensor_shapes.push_back(std::move(shape));
    }
    const auto public_family = std::find_if(plan.tensor_families.begin(), plan.tensor_families.end(),
                                            [](const TensorFamily &family) { return family.id == "public_Y"; });
    if (public_family == plan.tensor_families.end()) fail("RelationPlan has no public_Y family");
    if (public_output.rows != public_family->logical_shape.at(0) || public_output.cols != public_family->logical_shape.at(1))
        fail("independent verifier public_Y shape does not match RelationPlan");
    NativeTensorWitnessInput tensor;
    tensor.id = "public_Y";
    tensor.visibility = 1U;
    tensor.domain_size = checked_native_domain_u32(native_shape_product(public_family->padded_shape, "public_Y.domain"), "public_Y.domain");
    tensor.values = native_embed_zero_padded(*public_family, native_matrix_values(public_output));
    result.tensors.push_back(std::move(tensor));
    populate_authenticated_resource_wire(config, result, false);
    return result;
}

GatTrace execute_cpu(const Config &config, const Graph &graph, const ModelBundle &model, const StaticTables &tables) {
    return run_forward(config, graph, model, tables, false);
}

GatTrace execute_gpu(const Config &config, const Graph &graph, const ModelBundle &model, const StaticTables &tables, bool *kernel_executed) {
    if (!kernel_executed) fail("execute_gpu requires kernel status output");
    *kernel_executed = false;
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) fail("GPU backend requested but no CUDA device is available");
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    GatTrace trace = run_forward(config, graph, model, tables, true);
    *kernel_executed = true;
    return trace;
}

std::string statement_canonical_payload(const PublicStatement &statement) {
    std::ostringstream canonical;
    canonical << "schema=zkDL_gat_statement_v1\n"
              << "dataset=" << statement.dataset << "\n"
              << "model_arch_id=" << statement.model_arch_id << "\n"
              << "model_param_id=" << statement.model_param_id << "\n"
              << "quant_cfg_id=" << statement.quant_cfg_id << "\n"
              << "static_table_id=" << statement.static_table_id << "\n"
              << "input_preprocess_id=" << statement.input_preprocess_id << "\n"
              << "topology_id=" << statement.topology_id << "\n"
              << "feature_id=" << statement.feature_id << "\n"
              << "output_id=" << statement.output_id << "\n"
              << "public_output_path=" << statement.public_output_path << "\npublic_output_digest=" << statement.public_output_digest << "\n"
              << "relation_registry_digest=" << statement.relation_registry_digest << "\n"
              << "relation_terminal_claims_digest=" << statement.relation_terminal_claims_digest << "\n"
              << "relation_mapping_digest=" << statement.relation_mapping_digest << "\n"
              << "relation_selector_digest=" << statement.relation_selector_digest << "\n"
              << "relation_degree_digest=" << statement.relation_degree_digest << "\n"
              << "sorted_lookup_descriptor_digest=" << statement.sorted_lookup_descriptor_digest << "\n"
              << "sorted_lookup_plan_digest=" << statement.sorted_lookup_plan_digest << "\n"
              << "sorted_lookup_query_count=" << statement.sorted_lookup_query_count << "\n"
              << "static_table_generator_id=" << statement.static_table_generator_id << "\n"
              << "static_table_version=" << statement.static_table_version << "\n"
              << "static_table_scale_id=" << statement.static_table_scale_id << "\n"
              << "static_table_domain_id=" << statement.static_table_domain_id << "\n"
              << "static_table_row_order=" << statement.static_table_row_order << "\n"
              << "static_table_spec_hash=" << statement.static_table_spec_hash << "\n"
              << "static_table_digest=" << statement.static_table_digest << "\n"
              << "static_table_descriptor_digest=" << statement.static_table_descriptor_digest << "\n"
              << "scale_chain_digest=" << statement.scale_chain_digest << "\n"
              << "integer_bound_ledger_id=" << statement.integer_bound_ledger_id << "\n"
              << "integer_bound_ledger_digest=" << statement.integer_bound_ledger_digest << "\n"
              << "integer_bound_ledger_descriptor_digest=" << statement.integer_bound_ledger_descriptor_digest << "\n"
              << "config_digest=" << statement.config_digest << "\nasset_manifest_digest=" << statement.asset_manifest_digest << "\n"
              << "checkpoint_manifest_digest=" << statement.checkpoint_manifest_digest << "\n"
              << "checkpoint_tensors_txt_digest=" << statement.checkpoint_tensors_txt_digest << "\n"
              << "checkpoint_tensors_npz_digest=" << statement.checkpoint_tensors_npz_digest << "\n"
              << "verifier_registry_digest=" << statement.verifier_registry_digest << "\n"
              << "topology_boundary_digest=" << statement.topology_boundary_digest << "\n"
              << "total_nodes=" << statement.total_nodes << "\n"
              << "nodes=" << statement.nodes << "\n"
              << "edges=" << statement.edges << "\n"
              << "graphs=" << statement.graphs << "\n"
              << "classes=" << statement.classes << "\n"
              << "hidden_layer_count=" << statement.hidden_layer_count << "\nhidden_head_count=" << statement.hidden_head_count << "\n"
              << "output_head_count=" << statement.output_head_count << "\n"
              << "proof_scope=" << statement.proof_scope << "\n"
              << "verifier_reads_witness=0\n"
              << "privacy_policy=" << statement.privacy_policy << "\n";
    return canonical.str();
}

std::string statement_identity(const PublicStatement &statement) { return sha256_text(statement_canonical_payload(statement)); }

std::vector<std::uint8_t> native_encode_public_field_vector(const std::vector<NativeFrWire> &values) {
    std::vector<std::uint8_t> encoded;
    encoded.reserve(8U + values.size() * NativeFrWire{}.size() * sizeof(std::uint32_t));
    const auto append_u64 = [&](std::uint64_t value) {
        for (unsigned index = 0U; index < 8U; ++index) encoded.push_back(static_cast<std::uint8_t>(value >> (8U * index)));
    };
    const auto append_u32 = [&](std::uint32_t value) {
        for (unsigned index = 0U; index < 4U; ++index) encoded.push_back(static_cast<std::uint8_t>(value >> (8U * index)));
    };
    append_u64(static_cast<std::uint64_t>(values.size()));
    for (const auto &value : values)
        for (const auto limb : value) append_u32(limb);
    return encoded;
}

std::string canonical_public_field_digest(const Matrix &matrix) {
    const auto encoded = native_encode_public_field_vector(native_matrix_values(matrix));
    return sha256_bytes(encoded.data(), encoded.size());
}

std::string canonical_public_field_digest(const Matrix &matrix, const TensorFamily &family) {
    const auto padded = native_embed_zero_padded(family, native_matrix_values(matrix));
    const auto encoded = native_encode_public_field_vector(padded);
    return sha256_bytes(encoded.data(), encoded.size());
}

PublicStatement make_statement(const Config &config, const Graph &graph, const ModelBundle &model, const StaticTables &tables, const RelationPlan &relation,
                               const GatTrace &trace) {
    PublicStatement statement;
    statement.dataset = config.dataset;
    statement.model_arch_id = config.model_arch_id;
    statement.model_param_id = config.model_param_id;
    statement.quant_cfg_id = config.quant_cfg_id;
    statement.static_table_id = config.static_table_id;
    statement.input_preprocess_id = config.input_preprocess_id;
    statement.topology_id = digest_graph(graph);
    statement.feature_id = digest_matrix(graph.feature_values);
    statement.output_id = digest_matrix(trace.logits);
    const auto public_output_family = std::find_if(relation.tensor_families.begin(), relation.tensor_families.end(),
                                                   [](const TensorFamily &family) { return family.id == "public_Y"; });
    if (public_output_family == relation.tensor_families.end()) fail("RelationPlan has no public_Y family");
    statement.public_output_digest = trace.logits.values.empty()
                                         ? canonical_public_field_digest(trace.logits)
                                         : canonical_public_field_digest(trace.logits, *public_output_family);
    statement.relation_registry_digest = relation.registry_digest;
    statement.relation_terminal_claims_digest = relation.terminal_claims_digest;
    statement.relation_mapping_digest = relation.mapping_digest;
    statement.relation_selector_digest = relation.selector_digest;
    statement.relation_degree_digest = relation.degree_digest;
    statement.sorted_lookup_descriptor_digest = relation.sorted_lookup.descriptor_digest;
    statement.sorted_lookup_plan_digest = relation.sorted_lookup.plan_digest;
    statement.sorted_lookup_query_count = relation.sorted_lookup.queries.size();
    statement.static_table_generator_id = tables.generator_id;
    statement.static_table_version = tables.version;
    statement.static_table_scale_id = tables.scale_id;
    statement.static_table_domain_id = tables.domain_id;
    statement.static_table_row_order = tables.row_order;
    statement.static_table_spec_hash = tables.spec_hash;
    statement.static_table_digest = tables.digest;
    statement.static_table_descriptor_digest = tables.descriptor_digest;
    statement.scale_chain_digest = config.scale_chain_digest;
    statement.integer_bound_ledger_id = relation.integer_bound_ledger.ledger_id;
    statement.integer_bound_ledger_digest = relation.integer_bound_ledger.digest;
    statement.integer_bound_ledger_descriptor_digest = relation.integer_bound_ledger.descriptor_digest;
    statement.config_digest = config.config_digest;
    statement.asset_manifest_digest = sha256_file(config.asset_manifest_path);
    statement.checkpoint_manifest_digest = model.manifest_digest;
    statement.checkpoint_tensors_txt_digest = model.tensors_text_digest;
    statement.checkpoint_tensors_npz_digest = model.tensors_npz_digest;
    statement.verifier_registry_digest = config.verifier_registry_digest;
    statement.topology_boundary_digest = digest_graph_boundaries(graph);
    statement.total_nodes = graph.total_nodes;
    statement.nodes = graph.nodes;
    statement.edges = graph.edges;
    statement.graphs = graph.graphs;
    statement.classes = config.architecture.classes;
    statement.hidden_layer_count = config.architecture.hidden_dims.size();
    statement.hidden_head_count = std::accumulate(config.architecture.hidden_heads.begin(), config.architecture.hidden_heads.end(), u64{0});
    statement.output_head_count = config.architecture.output_heads;
    statement.privacy_policy = config.privacy_policy;
    statement.verifier_reads_witness = false;
    statement.statement_id = statement_identity(statement);
    return statement;
}

void write_statement(const std::filesystem::path &path, const PublicStatement &statement) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) fail("cannot write statement " + path.string());
    output << "schema=zkDL_gat_statement_v1\nstatement_id=" << statement.statement_id << "\n"
           << "dataset=" << statement.dataset << "\nmodel_arch_id=" << statement.model_arch_id << "\n"
           << "model_param_id=" << statement.model_param_id << "\nquant_cfg_id=" << statement.quant_cfg_id << "\n"
           << "static_table_id=" << statement.static_table_id << "\ninput_preprocess_id=" << statement.input_preprocess_id << "\n"
           << "topology_id=" << statement.topology_id << "\nfeature_id=" << statement.feature_id << "\noutput_id=" << statement.output_id << "\n"
           << "public_output_path=" << statement.public_output_path << "\npublic_output_digest=" << statement.public_output_digest << "\n"
           << "relation_registry_digest=" << statement.relation_registry_digest << "\n"
           << "relation_terminal_claims_digest=" << statement.relation_terminal_claims_digest << "\n"
           << "relation_mapping_digest=" << statement.relation_mapping_digest << "\n"
           << "relation_selector_digest=" << statement.relation_selector_digest << "\n"
           << "relation_degree_digest=" << statement.relation_degree_digest << "\n"
           << "sorted_lookup_descriptor_digest=" << statement.sorted_lookup_descriptor_digest << "\n"
           << "sorted_lookup_plan_digest=" << statement.sorted_lookup_plan_digest << "\n"
           << "sorted_lookup_query_count=" << statement.sorted_lookup_query_count << "\n"
           << "static_table_generator_id=" << statement.static_table_generator_id << "\nstatic_table_version=" << statement.static_table_version << "\n"
           << "static_table_scale_id=" << statement.static_table_scale_id << "\nstatic_table_domain_id=" << statement.static_table_domain_id << "\n"
           << "static_table_row_order=" << statement.static_table_row_order << "\nstatic_table_spec_hash=" << statement.static_table_spec_hash << "\n"
           << "static_table_digest=" << statement.static_table_digest << "\nstatic_table_descriptor_digest=" << statement.static_table_descriptor_digest << "\n"
           << "scale_chain_digest=" << statement.scale_chain_digest << "\n"
           << "integer_bound_ledger_id=" << statement.integer_bound_ledger_id << "\ninteger_bound_ledger_digest=" << statement.integer_bound_ledger_digest
           << "\n"
           << "integer_bound_ledger_descriptor_digest=" << statement.integer_bound_ledger_descriptor_digest << "\nconfig_digest=" << statement.config_digest
           << "\n"
           << "asset_manifest_digest=" << statement.asset_manifest_digest << "\ncheckpoint_manifest_digest=" << statement.checkpoint_manifest_digest << "\n"
           << "checkpoint_tensors_txt_digest=" << statement.checkpoint_tensors_txt_digest
           << "\ncheckpoint_tensors_npz_digest=" << statement.checkpoint_tensors_npz_digest << "\n"
           << "verifier_registry_digest=" << statement.verifier_registry_digest << "\ntopology_boundary_digest=" << statement.topology_boundary_digest << "\n"
           << "total_nodes=" << statement.total_nodes << "\nnodes=" << statement.nodes << "\nedges=" << statement.edges << "\n"
           << "graphs=" << statement.graphs << "\nclasses=" << statement.classes << "\nproof_scope=" << statement.proof_scope << "\n"
           << "hidden_layer_count=" << statement.hidden_layer_count << "\nhidden_head_count=" << statement.hidden_head_count << "\n"
           << "output_head_count=" << statement.output_head_count << "\n"
           << "privacy_policy=" << statement.privacy_policy << "\nverifier_reads_witness=" << (statement.verifier_reads_witness ? 1 : 0) << "\n";
    finish_output(output, path, "statement");
}

bool read_statement(const std::filesystem::path &path, PublicStatement *statement, std::string *error) {
    try {
        if (!statement) fail("read_statement received null output");
        const auto values = read_kv(path);
        const auto get = [&](const std::string &key) -> std::string {
            const auto iterator = values.find(key);
            if (iterator == values.end()) fail("statement field missing: " + key);
            return iterator->second;
        };
        if (get("schema") != "zkDL_gat_statement_v1") fail("statement schema mismatch");
        statement->statement_id = get("statement_id");
        statement->dataset = get("dataset");
        statement->model_arch_id = get("model_arch_id");
        statement->model_param_id = get("model_param_id");
        statement->quant_cfg_id = get("quant_cfg_id");
        statement->static_table_id = get("static_table_id");
        statement->input_preprocess_id = get("input_preprocess_id");
        statement->topology_id = get("topology_id");
        statement->feature_id = get("feature_id");
        statement->output_id = get("output_id");
        statement->public_output_path = get("public_output_path");
        statement->public_output_digest = get("public_output_digest");
        statement->relation_registry_digest = get("relation_registry_digest");
        statement->relation_terminal_claims_digest = get("relation_terminal_claims_digest");
        statement->relation_mapping_digest = get("relation_mapping_digest");
        statement->relation_selector_digest = get("relation_selector_digest");
        statement->relation_degree_digest = get("relation_degree_digest");
        statement->sorted_lookup_descriptor_digest = get("sorted_lookup_descriptor_digest");
        statement->sorted_lookup_plan_digest = get("sorted_lookup_plan_digest");
        statement->sorted_lookup_query_count = parse_u64(get("sorted_lookup_query_count"), "statement sorted_lookup_query_count");
        statement->static_table_generator_id = get("static_table_generator_id");
        statement->static_table_version = get("static_table_version");
        statement->static_table_scale_id = get("static_table_scale_id");
        statement->static_table_domain_id = get("static_table_domain_id");
        statement->static_table_row_order = get("static_table_row_order");
        statement->static_table_spec_hash = get("static_table_spec_hash");
        statement->static_table_digest = get("static_table_digest");
        statement->static_table_descriptor_digest = get("static_table_descriptor_digest");
        statement->scale_chain_digest = get("scale_chain_digest");
        statement->integer_bound_ledger_id = get("integer_bound_ledger_id");
        statement->integer_bound_ledger_digest = get("integer_bound_ledger_digest");
        statement->integer_bound_ledger_descriptor_digest = get("integer_bound_ledger_descriptor_digest");
        statement->config_digest = get("config_digest");
        statement->asset_manifest_digest = get("asset_manifest_digest");
        statement->checkpoint_manifest_digest = get("checkpoint_manifest_digest");
        statement->checkpoint_tensors_txt_digest = get("checkpoint_tensors_txt_digest");
        statement->checkpoint_tensors_npz_digest = get("checkpoint_tensors_npz_digest");
        statement->verifier_registry_digest = get("verifier_registry_digest");
        statement->topology_boundary_digest = get("topology_boundary_digest");
        statement->total_nodes = parse_u64(get("total_nodes"), "statement total_nodes");
        statement->nodes = parse_u64(get("nodes"), "statement nodes");
        statement->edges = parse_u64(get("edges"), "statement edges");
        statement->graphs = parse_u64(get("graphs"), "statement graphs");
        statement->classes = parse_u64(get("classes"), "statement classes");
        statement->hidden_layer_count = parse_u64(get("hidden_layer_count"), "statement hidden_layer_count");
        statement->hidden_head_count = parse_u64(get("hidden_head_count"), "statement hidden_head_count");
        statement->output_head_count = parse_u64(get("output_head_count"), "statement output_head_count");
        statement->proof_scope = get("proof_scope");
        statement->privacy_policy = get("privacy_policy");
        statement->verifier_reads_witness = parse_bool(get("verifier_reads_witness"), "statement verifier_reads_witness");
        if (statement->verifier_reads_witness) fail("statement violates witness-independent verifier boundary");
        if (statement->statement_id != statement_identity(*statement)) fail("statement identity mismatch");
        return true;
    } catch (const std::exception &exception) {
        if (error) *error = exception.what();
        return false;
    }
}

void write_run_record(const std::filesystem::path &path, const RunRecord &record) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) fail("cannot write run record " + path.string());
    output << "schema=zkDL_gat_run_record_v2\ndataset=" << record.dataset << "\nbackend=" << record.backend << "\n"
           << "statement_id=" << record.statement_id << "\nproof_id=" << record.proof_id << "\nverification_status=" << record.verification_status << "\n"
           << "proof_scope=" << record.proof_scope << "\nstatic_table_descriptor_digest=" << record.static_table_descriptor_digest << "\n"
           << "scale_chain_digest=" << record.scale_chain_digest << "\ninteger_bound_ledger_id=" << record.integer_bound_ledger_id << "\n"
           << "integer_bound_ledger_digest=" << record.integer_bound_ledger_digest << "\n"
           << "integer_bound_ledger_descriptor_digest=" << record.integer_bound_ledger_descriptor_digest << "\n"
           << "proof_size_bytes=" << record.proof_size_bytes << "\ncommitment_size_bytes=" << record.commitment_size_bytes << "\n"
           << "commitment_time_semantics=" << record.commitment_time_semantics << "\n"
           << "commitment_time_seconds=" << std::setprecision(17) << record.commitment_time_seconds << "\nprove_time_seconds=" << record.prove_time_seconds
           << "\n"
           << "verify_time_seconds=" << record.verify_time_seconds << "\nverifier_reads_witness=" << (record.verifier_reads_witness ? 1 : 0) << "\n"
           << "gpu_kernel_executed=" << (record.gpu_kernel_executed ? 1 : 0) << "\nsilent_fallback=" << (record.silent_fallback ? 1 : 0) << "\n"
           << "output_bound=" << (record.output_bound ? 1 : 0) << "\nerror=" << record.error << "\n";
    finish_output(output, path, "run record");
}

namespace {

struct CliOptions {
    std::string command;
    std::filesystem::path config;
    std::filesystem::path planned_shape;
    std::filesystem::path run_dir;
    std::filesystem::path statement;
    std::filesystem::path proof;
    std::filesystem::path envelope;
    Backend backend = Backend::Cpu;
    bool has_backend = false;
};

void usage() {
    std::cerr << "usage: gat_runner <inspect|admit|resource-plan|prepare|run|prove|verify> --config FILE "
                 "[--backend cpu|gpu] [--run-dir DIR] [--planned-shape ANALYSIS_ONLY_JSON]\n";
}

CliOptions parse_cli(int argc, char **argv) {
    if (argc < 3) {
        usage();
        fail("missing command/config");
    }
    CliOptions options;
    options.command = argv[1];
    const std::set<std::string> commands = {"inspect", "admit", "resource-plan", "prepare", "run", "prove", "verify"};
    if (!commands.count(options.command)) fail("unknown command: " + options.command);
    for (int i = 2; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--config" || argument == "--planned-shape" || argument == "--run-dir" || argument == "--statement" || argument == "--proof" || argument == "--envelope" ||
            argument == "--backend") {
            if (++i >= argc) fail("missing value for " + argument);
            const std::string value = argv[i];
            if (argument == "--config")
                options.config = value;
            else if (argument == "--planned-shape")
                options.planned_shape = value;
            else if (argument == "--run-dir")
                options.run_dir = value;
            else if (argument == "--statement")
                options.statement = value;
            else if (argument == "--proof")
                options.proof = value;
            else if (argument == "--envelope")
                options.envelope = value;
            else {
                options.backend = parse_backend(value);
                options.has_backend = true;
            }
        } else {
            fail("unknown argument: " + argument);
        }
    }
    if (options.config.empty()) fail("--config is required");
    if (!options.planned_shape.empty() && options.command != "resource-plan")
        fail("--planned-shape is allowed only for resource-plan");
    if ((options.command == "run" || options.command == "prove") && !options.has_backend) { fail("--backend is required for " + options.command); }
    return options;
}

std::filesystem::path run_path(const Config &config, const std::filesystem::path &raw) {
    if (raw.empty()) return config.run_root;
    if (raw.is_absolute()) {
        const auto base = config.baseline_root.lexically_normal().generic_string();
        const auto candidate = raw.lexically_normal().generic_string();
        if (candidate != base && candidate.rfind(base + "/", 0) != 0) fail("run directory escapes baseline root");
        return raw.lexically_normal();
    }
    return local_path(config.baseline_root, raw.generic_string(), "run_dir");
}

Config config_for_run(const Config &config, const std::filesystem::path &directory) {
    Config result = config;
    result.run_root = directory;
    result.asset_manifest_path = directory / "asset_manifest.json";
    return result;
}

std::filesystem::path run_file_path(const std::filesystem::path &directory, const std::filesystem::path &raw, const std::string &name) {
    const auto expected = (directory / name).lexically_normal();
    const auto candidate = raw.empty() ? expected : (raw.is_absolute() ? raw.lexically_normal() : local_path(directory, raw.generic_string(), name));
    if (candidate != expected) { fail(name + " must be the run-local " + expected.string()); }
    return candidate;
}

void write_matrix_text(const std::filesystem::path &path, const Matrix &matrix) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) fail("cannot write matrix " + path.string());
    output << matrix.rows << " " << matrix.cols << "\n";
    for (u64 row = 0; row < matrix.rows; ++row) {
        for (u64 col = 0; col < matrix.cols; ++col) {
            if (col) output << ' ';
            output << matrix.at(row, col);
        }
        output << '\n';
    }
    finish_output(output, path, "matrix");
}

Matrix read_public_output(const std::filesystem::path &path, std::uint64_t expected_rows, std::uint64_t expected_cols) {
    if (path.empty() || std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path)) {
        fail("public output is missing or not a regular baseline-local file: " + path.string());
    }
    std::ifstream input(path);
    if (!input) fail("cannot open public output " + path.string());
    u64 rows = 0, cols = 0;
    input >> rows >> cols;
    if (!input || rows != expected_rows || cols != expected_cols) fail("public output shape mismatch");
    Matrix output = make_matrix(rows, cols);
    for (auto &value : output.values) {
        input >> value;
        if (!input) fail("truncated public output");
    }
    std::string trailing;
    if (input >> trailing) fail("public output contains trailing data");
    return output;
}

std::string serialize_plan_text(const RelationPlan &plan, const StaticTables &tables) {
    std::ostringstream output;
    output << "schema_id=" << plan.schema_id << "\nproof_scope=" << plan.proof_scope << "\nbinding_rule=" << plan.binding_rule << "\n"
           << "registry_digest=" << plan.registry_digest << "\nterminal_claims_digest=" << plan.terminal_claims_digest << "\n"
           << "schema_digest=" << plan.schema_digest << "\nmapping_digest=" << plan.mapping_digest << "\n"
           << "selector_digest=" << plan.selector_digest << "\ndegree_digest=" << plan.degree_digest << "\n"
           << "static_table_descriptor_digest=" << plan.static_table_descriptor_digest << "\n"
           << "scale_chain_digest=" << plan.scale_chain_digest << "\n"
           << "integer_bound_ledger_digest=" << plan.integer_bound_ledger_digest << "\n"
           << "integer_bound_ledger_descriptor_digest=" << plan.integer_bound_ledger_descriptor_digest << "\n"
           << "integer_bound_ledger_id=" << plan.integer_bound_ledger.ledger_id << "\n"
           << "table_generator_id=" << tables.generator_id << "\ntable_spec_hash=" << tables.spec_hash << "\ntable_digest=" << tables.digest << "\n"
           << "table_domain_id=" << tables.domain_id << "\ntable_row_order=" << tables.row_order << "\n"
           << "sorted_lookup_descriptor=" << plan.sorted_lookup.descriptor_path << "\n"
           << "sorted_lookup_descriptor_digest=" << plan.sorted_lookup.descriptor_digest << "\n"
           << "sorted_lookup_table_digest=" << plan.sorted_lookup.table_digest << "\n"
           << "sorted_lookup_plan_digest=" << plan.sorted_lookup.plan_digest << "\n"
           << "sorted_lookup_query_count=" << plan.sorted_lookup.queries.size() << "\n"
           << "sorted_lookup_segment_size=" << plan.sorted_lookup.segment_size << "\n";
    for (const auto &query : plan.sorted_lookup.queries) {
        output << "sorted_lookup_query=" << query.operation_id << '|' << query.table_ref << '|' << query.query_ref << '|' << query.query_x_ref << '|'
               << query.query_y_ref << '|' << query.query_key_ref << '|' << query.merge_ref << '|' << query.merge_key_ref << '|' << query.merge_delta_ref << '|'
               << query.merge_marker_ref << '|' << query.merge_inverse_ref << '|' << query.merge_prefix_ref << '|' << query.query_prefix_ref << '|'
               << query.commitment_role_map << '|' << query.table_logical_rows << '|' << query.table_domain << '|' << query.query_logical_rows << '|'
               << query.query_domain << '|' << query.merge_logical_rows << '|' << query.merge_domain << '|' << query.segment_size << '|'
               << query.table_segment_count << '|' << query.query_segment_count << '|' << query.merge_segment_count << '|' << query.x_bits << '|'
               << query.y_bits << '|' << query.key_max_u128 << '|' << query.layout_digest << '\n';
    }
    for (const auto &family : plan.tensor_families) {
        output << "tensor=" << family.id << "|" << family.visibility << "|" << family.mle_order << "|" << family.stack_axis << "|";
        for (std::size_t i = 0; i < family.logical_shape.size(); ++i) {
            if (i) output << ',';
            output << family.logical_shape[i];
        }
        output << '|';
        for (std::size_t i = 0; i < family.padded_shape.size(); ++i) {
            if (i) output << ',';
            output << family.padded_shape[i];
        }
        output << '|';
        for (std::size_t i = 0; i < family.instances.size(); ++i) {
            if (i) output << ',';
            output << family.instances[i];
        }
        output << '|' << family.instance_order << '|' << family.mapping << '|' << family.zero_selector << '|' << family.selector_id << '|'
               << family.padding_rule << '|' << family.terminal_point << '|' << family.terminal_value << '|' << family.terminal_claim
               << "|degree=" << family.degree_bound << '\n';
    }
    for (const auto &operation : plan.operation_families) {
        output << "operation=" << operation.id << '|' << operation.kind << '|' << operation.degree_bound << "|degree=" << operation.degree << '|'
               << (operation.requires_reindex ? 1 : 0) << '|' << (operation.public_route ? 1 : 0) << '|';
        for (std::size_t i = 0; i < operation.inputs.size(); ++i) {
            if (i) output << ',';
            output << operation.inputs[i];
        }
        output << '|';
        for (std::size_t i = 0; i < operation.outputs.size(); ++i) {
            if (i) output << ',';
            output << operation.outputs[i];
        }
        output << '|';
        for (std::size_t i = 0; i < operation.views.size(); ++i) {
            if (i) output << ',';
            output << operation.views[i];
        }
        output << '|' << operation.instance_order << '|' << operation.mapping << '|';
        for (std::size_t i = 0; i < operation.auxiliary.size(); ++i) {
            if (i) output << ',';
            output << operation.auxiliary[i];
        }
        output << '|' << operation.selector_id << '|' << operation.terminal_point << '|' << operation.terminal_value << '|' << operation.terminal_claim << '\n';
    }
    return output.str();
}

void write_plan(const std::filesystem::path &path, const RelationPlan &plan, const StaticTables &tables) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) fail("cannot write relation plan " + path.string());
    const std::string serialized = serialize_plan_text(plan, tables);
    if (!serialized.empty()) output.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
    finish_output(output, path, "relation plan");
}

void write_bytes(const std::filesystem::path &path, const std::vector<std::uint8_t> &bytes) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) fail("cannot write proof bytes " + path.string());
    if (!bytes.empty()) output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    finish_output(output, path, "proof bytes");
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) fail("cannot read proof bytes " + path.string());
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (end < 0) fail("cannot size proof bytes");
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    if (!bytes.empty()) input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input && !input.eof()) fail("truncated proof bytes");
    return bytes;
}

void write_envelope(const std::filesystem::path &path, const ProofEnvelope &envelope) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) fail("cannot write proof envelope " + path.string());
    output << "schema_id=" << envelope.schema_id << "\nstatement_id=" << envelope.statement_id << "\nproof_id=" << envelope.proof_id << "\n"
           << "backend=" << envelope.backend << "\nproof_scope=" << envelope.proof_scope << "\nnative_core_api=" << envelope.native_core_api << "\n"
           << "status=" << envelope.status << "\nstatic_table_descriptor_digest=" << envelope.static_table_descriptor_digest << "\n"
           << "scale_chain_digest=" << envelope.scale_chain_digest << "\ninteger_bound_ledger_id=" << envelope.integer_bound_ledger_id << "\n"
           << "integer_bound_ledger_digest=" << envelope.integer_bound_ledger_digest << "\n"
           << "integer_bound_ledger_descriptor_digest=" << envelope.integer_bound_ledger_descriptor_digest << "\n"
           << "proof_size_bytes=" << envelope.proof_size_bytes << "\ncommitment_size_bytes=" << envelope.commitment_size_bytes << "\n"
           << "commitment_time_semantics=" << envelope.commitment_time_semantics << "\n"
           << "commitment_time_seconds=" << std::setprecision(17) << envelope.commitment_time_seconds << "\nprove_time_seconds=" << envelope.prove_time_seconds
           << "verify_time_seconds=" << envelope.verify_time_seconds << "\nverifier_reads_witness=" << (envelope.verifier_reads_witness ? 1 : 0) << "\n"
           << "gpu_kernel_executed=" << (envelope.gpu_kernel_executed ? 1 : 0) << "\nsilent_fallback=" << (envelope.silent_fallback ? 1 : 0) << "\n"
           << "output_bound=" << (envelope.output_bound ? 1 : 0) << "\nerror=" << envelope.error << "\n";
    finish_output(output, path, "proof envelope");
}

void write_proof_timing(const std::filesystem::path &path, const ProofEnvelope &envelope) {
    constexpr const char *kProofAuxiliaryTimingScope =
        "signed_round_aux_witness_and_relation_views;steady_clock_wall;exclude_family_storage_write_hash_and_explicit_file_io";
    const std::vector<double> components = {
        envelope.proof_auxiliary_construction_seconds, envelope.sorted_lookup_protocol_construction_seconds,
        envelope.relation_context_prove_seconds, envelope.relation_prove_wall_seconds,
        envelope.relation_commitment_time_seconds, envelope.prove_provider_file_io_excluded_seconds,
        envelope.proof_serialization_seconds,
    };
    if (envelope.status != "PROVED" || envelope.proof_auxiliary_construction_scope != kProofAuxiliaryTimingScope ||
        std::any_of(components.begin(), components.end(), [](double value) { return !std::isfinite(value) || value < 0.0; }) ||
        envelope.relation_commitment_time_seconds > envelope.relation_prove_wall_seconds ||
        envelope.prove_provider_file_io_excluded_seconds > envelope.relation_prove_wall_seconds - envelope.relation_commitment_time_seconds) {
        fail("raw proof timing sidecar has invalid status/scope/components");
    }
    const double recomposed_prove = envelope.proof_auxiliary_construction_seconds + envelope.sorted_lookup_protocol_construction_seconds +
        envelope.relation_context_prove_seconds + envelope.relation_prove_wall_seconds - envelope.relation_commitment_time_seconds -
        envelope.prove_provider_file_io_excluded_seconds + envelope.proof_serialization_seconds;
    if (!std::isfinite(recomposed_prove) || !std::isfinite(envelope.prove_time_seconds) ||
        std::abs(recomposed_prove - envelope.prove_time_seconds) >
            2e-12 * std::max(1.0, std::max(std::abs(recomposed_prove), std::abs(envelope.prove_time_seconds)))) {
        fail("raw proof timing sidecar components do not sum to Prove Time");
    }
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) fail("cannot write proof timing diagnostics " + path.string());
    output << "schema_id=zkDL_gat_proof_timing_v1\nstatus=PROVED\n"
           << "statement_id=" << envelope.statement_id << "\nproof_id=" << envelope.proof_id << "\n"
           << "commitment_time_seconds=" << std::setprecision(17) << envelope.commitment_time_seconds << "\n"
           << "prove_time_seconds=" << envelope.prove_time_seconds << "\n"
           << "proof_auxiliary_construction_seconds=" << envelope.proof_auxiliary_construction_seconds << "\n"
           << "proof_auxiliary_construction_scope=" << envelope.proof_auxiliary_construction_scope << "\n"
           << "sorted_lookup_protocol_construction_seconds=" << envelope.sorted_lookup_protocol_construction_seconds << "\n"
           << "relation_context_prove_seconds=" << envelope.relation_context_prove_seconds << "\n"
           << "relation_prove_wall_seconds=" << envelope.relation_prove_wall_seconds << "\n"
           << "relation_commitment_time_seconds=" << envelope.relation_commitment_time_seconds << "\n"
           << "prove_provider_file_io_excluded_seconds=" << envelope.prove_provider_file_io_excluded_seconds << "\n"
           << "proof_serialization_seconds=" << envelope.proof_serialization_seconds << "\n";
    finish_output(output, path, "proof timing diagnostics");
}

ProofEnvelope read_envelope(const std::filesystem::path &path) {
    const auto values = read_kv(path);
    const auto get = [&](const std::string &key) -> std::string {
        const auto iterator = values.find(key);
        if (iterator == values.end()) fail("proof envelope field missing: " + key);
        return iterator->second;
    };
    ProofEnvelope envelope;
    envelope.schema_id = get("schema_id");
    envelope.statement_id = get("statement_id");
    envelope.proof_id = get("proof_id");
    envelope.backend = get("backend");
    envelope.proof_scope = get("proof_scope");
    envelope.native_core_api = get("native_core_api");
    envelope.status = get("status");
    envelope.proof_size_bytes = parse_u64(get("proof_size_bytes"), "proof_size_bytes");
    envelope.static_table_descriptor_digest = get("static_table_descriptor_digest");
    envelope.scale_chain_digest = get("scale_chain_digest");
    envelope.integer_bound_ledger_id = get("integer_bound_ledger_id");
    envelope.integer_bound_ledger_digest = get("integer_bound_ledger_digest");
    envelope.integer_bound_ledger_descriptor_digest = get("integer_bound_ledger_descriptor_digest");
    envelope.commitment_size_bytes = parse_u64(get("commitment_size_bytes"), "commitment_size_bytes");
    envelope.commitment_time_semantics = get("commitment_time_semantics");
    envelope.commitment_time_seconds = parse_double(get("commitment_time_seconds"), "commitment_time_seconds");
    envelope.prove_time_seconds = parse_double(get("prove_time_seconds"), "prove_time_seconds");
    envelope.verify_time_seconds = parse_double(get("verify_time_seconds"), "verify_time_seconds");
    envelope.verifier_reads_witness = parse_bool(get("verifier_reads_witness"), "verifier_reads_witness");
    envelope.gpu_kernel_executed = parse_bool(get("gpu_kernel_executed"), "gpu_kernel_executed");
    envelope.silent_fallback = parse_bool(get("silent_fallback"), "silent_fallback");
    envelope.output_bound = parse_bool(get("output_bound"), "output_bound");
    envelope.error = get("error");
    return envelope;
}

RunRecord rejected_record(const Config &config, Backend backend, const PublicStatement *statement, const std::string &message, bool kernel_executed) {
    RunRecord record;
    record.dataset = config.dataset;
    record.backend = backend_name(backend);
    record.proof_scope = "full_gat_forward";
    record.verification_status = "BLOCKED";
    record.error = message;
    record.gpu_kernel_executed = kernel_executed;
    record.statement_id = statement ? statement->statement_id : std::string(64, '0');
    record.static_table_descriptor_digest = statement ? statement->static_table_descriptor_digest : "UNAVAILABLE";
    record.scale_chain_digest = statement ? statement->scale_chain_digest : config.scale_chain_digest;
    record.integer_bound_ledger_id = statement ? statement->integer_bound_ledger_id : "UNAVAILABLE";
    record.integer_bound_ledger_digest = statement ? statement->integer_bound_ledger_digest : "UNAVAILABLE";
    record.integer_bound_ledger_descriptor_digest = statement ? statement->integer_bound_ledger_descriptor_digest : "UNAVAILABLE";
    record.silent_fallback = false;
    record.verifier_reads_witness = false;
    record.output_bound = false;
    return record;
}

void bind_envelope_identity(ProofEnvelope *envelope, const PublicStatement &statement) {
    if (!envelope) fail("proof envelope pointer is null");
    const auto check_or_bind = [](std::string *actual, const std::string &expected, const std::string &label) {
        if (!actual->empty() && *actual != expected) fail("native proof envelope " + label + " mismatch");
        *actual = expected;
    };
    check_or_bind(&envelope->static_table_descriptor_digest, statement.static_table_descriptor_digest, "static table descriptor identity");
    check_or_bind(&envelope->scale_chain_digest, statement.scale_chain_digest, "scale-chain identity");
    check_or_bind(&envelope->integer_bound_ledger_id, statement.integer_bound_ledger_id, "integer-bound ledger id");
    check_or_bind(&envelope->integer_bound_ledger_digest, statement.integer_bound_ledger_digest, "integer-bound ledger digest");
    check_or_bind(&envelope->integer_bound_ledger_descriptor_digest, statement.integer_bound_ledger_descriptor_digest,
                  "integer-bound ledger descriptor identity");
}

RunRecord record_from_envelope(const Config &config, const PublicStatement &statement, const ProofEnvelope &envelope, const std::string &status,
                               const std::string &error = {}) {
    RunRecord record;
    record.dataset = config.dataset;
    record.backend = envelope.backend;
    record.statement_id = statement.statement_id;
    record.proof_id = envelope.proof_id;
    record.verification_status = status;
    record.proof_scope = statement.proof_scope;
    record.static_table_descriptor_digest = statement.static_table_descriptor_digest;
    record.scale_chain_digest = statement.scale_chain_digest;
    record.integer_bound_ledger_id = statement.integer_bound_ledger_id;
    record.integer_bound_ledger_digest = statement.integer_bound_ledger_digest;
    record.integer_bound_ledger_descriptor_digest = statement.integer_bound_ledger_descriptor_digest;
    record.proof_size_bytes = envelope.proof_size_bytes;
    record.commitment_size_bytes = envelope.commitment_size_bytes;
    record.commitment_time_semantics = envelope.commitment_time_semantics;
    record.commitment_time_seconds = envelope.commitment_time_seconds;
    record.prove_time_seconds = envelope.prove_time_seconds;
    record.verify_time_seconds = envelope.verify_time_seconds;
    record.verifier_reads_witness = envelope.verifier_reads_witness;
    record.gpu_kernel_executed = envelope.gpu_kernel_executed;
    record.silent_fallback = envelope.silent_fallback;
    record.output_bound = envelope.output_bound;
    record.error = error.empty() ? envelope.error : error;
    return record;
}

int produce(const Config &config, const CliOptions &options) {
    const auto directory = run_path(config, options.run_dir);
    const auto run_config = config_for_run(config, directory);
    const auto statement_path = directory / "statement.txt";
    const auto envelope_path = directory / "proof.meta";
    const auto proof_path = directory / "proof.bin";
    const auto record_path = directory / "run.record";
    bool kernel_executed = false;
    try {
        const auto model = load_model_bundle(run_config);
        const auto graph = load_graph(run_config);
        const auto tables = build_static_tables(run_config, graph);
        const auto relation = build_relation_plan(run_config, graph, tables);
        std::string relation_error;
        if (!relation_plan_complete(relation, &relation_error)) fail(relation_error);

        admit_integer_bound_ledger(run_config, graph, model, tables, relation.integer_bound_ledger);
        GatTrace trace;
        try {
            trace = options.backend == Backend::Cpu ? execute_cpu(run_config, graph, model, tables) :
                                                      execute_gpu(run_config, graph, model, tables, &kernel_executed);
        } catch (const std::exception &exception) {
            const auto statement = make_statement(run_config, graph, model, tables, relation, GatTrace{});
            write_statement(statement_path, statement);
            ProofEnvelope envelope;
            envelope.backend = backend_name(options.backend);
            envelope.statement_id = statement.statement_id;
            envelope.status = "BLOCKED";
            envelope.gpu_kernel_executed = kernel_executed;
            envelope.error = exception.what();
            bind_envelope_identity(&envelope, statement);
            write_envelope(envelope_path, envelope);
            write_run_record(record_path, rejected_record(run_config, options.backend, &statement, exception.what(), kernel_executed));
            return 2;
        }
        const auto statement = make_statement(run_config, graph, model, tables, relation, trace);
        write_statement(statement_path, statement);
        write_matrix_text(directory / "public_Y.txt", trace.logits);
        write_plan(directory / "relation.plan", relation, tables);

        const auto witness = build_native_relation_witness(run_config, graph, model, tables, relation, trace);
        // Every private trace family above has been synchronously persisted
        // into authenticated NativeFamilyStorage.  The native proof core only
        // requires this non-null request token; it consumes those storage
        // readers and the already-built public statement, never a lazy view of
        // GatTrace.  Release the full forward trace before constructing the
        // prover context and proof so it cannot overlap the much larger native
        // relation phases.
        trace = GatTrace{};
        NativeProveRequest request;
        request.config = &run_config;
        request.graph = &graph;
        request.model = &model;
        request.tables = &tables;
        request.relation = &relation;
        request.trace = &trace;
        request.statement = &statement;
        request.backend = options.backend;
        request.witness = &witness;
        NativeProofResult result;
        std::string error;
        if (!zkdl_gat_native_prove(request, &result, &error)) {
            bind_envelope_identity(&result.envelope, statement);
            write_envelope(envelope_path, result.envelope);
            const std::string message = error.empty() ? "native prove rejected" : error;
            auto record = rejected_record(run_config, options.backend, &statement, message, kernel_executed);
            record.proof_id = result.envelope.proof_id;
            record.proof_size_bytes = result.envelope.proof_size_bytes;
            write_run_record(record_path, record);
            return 4;
        }

        if (result.envelope.statement_id != statement.statement_id || result.envelope.backend != backend_name(options.backend) ||
            result.envelope.proof_scope != statement.proof_scope || result.envelope.proof_size_bytes != result.proof_bytes.size() ||
            result.envelope.gpu_kernel_executed != kernel_executed) {
            const std::string message = "native proof envelope does not bind statement/backend/proof provenance";
            write_envelope(envelope_path, result.envelope);
            write_run_record(record_path, rejected_record(run_config, options.backend, &statement, message, kernel_executed));
            return 5;
        }
        bind_envelope_identity(&result.envelope, statement);
        if (result.envelope.status != "PROVED" || result.envelope.proof_id.empty() || result.proof_bytes.empty() ||
            result.envelope.commitment_time_semantics != "native_tensor_commitments" ||
            result.envelope.native_core_api != "zkdl_native_gat_bridge_v1") {
            const std::string message = "native proof envelope is not a complete proved proof artifact";
            write_envelope(envelope_path, result.envelope);
            write_run_record(record_path, rejected_record(run_config, options.backend, &statement, message, kernel_executed));
            return 6;
        }
        write_bytes(proof_path, result.proof_bytes);
        write_envelope(envelope_path, result.envelope);
        write_proof_timing(directory / "proof.timing", result.envelope);
        const auto record = record_from_envelope(run_config, statement, result.envelope, "NOT_RUN");
        write_run_record(record_path, record);
        return 0;
    } catch (const std::exception &exception) {
        ProofEnvelope envelope;
        envelope.backend = backend_name(options.backend);
        envelope.status = "BLOCKED";
        envelope.gpu_kernel_executed = kernel_executed;
        envelope.error = exception.what();
        write_envelope(envelope_path, envelope);
        write_run_record(record_path, rejected_record(run_config, options.backend, nullptr, exception.what(), kernel_executed));
        return 2;
    }
}

int prepare(const Config &config, const CliOptions &options) {
    const auto directory = run_path(config, options.run_dir);
    const auto run_config = config_for_run(config, directory);
    const auto model = load_model_bundle(run_config);
    const auto graph = load_graph(run_config);
    const auto tables = build_static_tables(run_config, graph);
    const auto plan = build_relation_plan(run_config, graph, tables);
    std::string relation_error;
    if (!relation_plan_complete(plan, &relation_error)) fail(relation_error);
    write_plan(directory / "relation.plan", plan, tables);
    std::ofstream identity(directory / "identity.cfg");
    if (!identity) fail("cannot write identity.cfg");
    identity << "dataset=" << config.dataset << "\nmodel_manifest_sha256=" << model.manifest_digest << "\n"
             << "topology_sha256=" << digest_graph(graph) << "\nfeature_sha256=" << digest_matrix(graph.feature_values) << "\n"
             << "table_sha256=" << tables.digest << "\nstatic_table_descriptor_digest=" << tables.descriptor_digest << "\n"
             << "scale_chain_digest=" << config.scale_chain_digest << "\ninteger_bound_ledger_id=" << plan.integer_bound_ledger.ledger_id << "\n"
             << "integer_bound_ledger_digest=" << plan.integer_bound_ledger.digest << "\n"
             << "integer_bound_ledger_descriptor_digest=" << plan.integer_bound_ledger.descriptor_digest << "\n"
             << "proof_scope=" << plan.proof_scope << "\n";
    finish_output(identity, directory / "identity.cfg", "prepare identity");
    std::cout << "prepared dataset=" << config.dataset << " nodes=" << graph.nodes << " edges=" << graph.edges
              << " relation_operations=" << plan.operation_families.size() << "\n";
    return 0;
}

void validate_statement_binding(const Config &config, const Graph &graph, const ModelBundle &model, const StaticTables &tables, const RelationPlan &relation,
                                const PublicStatement &statement) {
    const auto require_equal = [](const std::string &actual, const std::string &expected, const std::string &label) {
        if (actual != expected) fail("public statement " + label + " does not match authenticated run inputs");
    };
    require_equal(statement.static_table_descriptor_digest, tables.descriptor_digest, "static table descriptor digest");
    require_equal(statement.static_table_digest, tables.digest, "static table digest");
    require_equal(statement.scale_chain_digest, config.scale_chain_digest, "scale-chain digest");
    require_equal(statement.integer_bound_ledger_id, relation.integer_bound_ledger.ledger_id, "integer-bound ledger id");
    require_equal(statement.integer_bound_ledger_digest, relation.integer_bound_ledger.digest, "integer-bound ledger digest");
    require_equal(statement.integer_bound_ledger_descriptor_digest, relation.integer_bound_ledger.descriptor_digest, "integer-bound ledger descriptor digest");
    require_equal(statement.sorted_lookup_descriptor_digest, relation.sorted_lookup.descriptor_digest, "sorted-lookup descriptor digest");
    require_equal(statement.sorted_lookup_plan_digest, relation.sorted_lookup.plan_digest, "sorted-lookup plan digest");
    if (statement.sorted_lookup_query_count != relation.sorted_lookup.queries.size()) fail("public statement sorted-lookup query count mismatch");
    if (!is_digest(statement.static_table_descriptor_digest) || !is_digest(statement.scale_chain_digest) || !is_digest(statement.integer_bound_ledger_digest) ||
        !is_digest(statement.integer_bound_ledger_descriptor_digest) || !is_digest(statement.sorted_lookup_descriptor_digest) ||
        !is_digest(statement.sorted_lookup_plan_digest)) {
        fail("public statement relation identities are not canonical SHA-256 digests");
    }
    if (statement.verifier_reads_witness || statement.proof_scope != "full_gat_forward") { fail("public statement violates the canonical verifier boundary"); }

    auto expected = make_statement(config, graph, model, tables, relation, GatTrace{});
    expected.output_id = statement.output_id;
    expected.public_output_digest = statement.public_output_digest;
    expected.statement_id = statement_identity(expected);
    if (statement.statement_id != expected.statement_id) { fail("public statement identity is not bound to authenticated run inputs"); }
    if (statement.asset_manifest_digest != sha256_file(config.asset_manifest_path)) {
        fail("public statement asset manifest digest is stale or unauthenticated");
    }
}

int admit(const Config &config) {
    const auto model = load_model_bundle(config);
    const auto graph = load_graph(config);
    const auto tables = build_static_tables(config, graph);
    const auto relation = build_relation_plan(config, graph, tables);
    std::string relation_error;
    if (!relation_plan_complete(relation, &relation_error)) fail(relation_error);
    admit_integer_bound_ledger(config, graph, model, tables, relation.integer_bound_ledger);
    NativeRelationPlanMetrics plan_metrics;
    if (!zkdl_gat_native_relation_plan_metrics(config, graph, tables, relation, &plan_metrics, &relation_error))
        fail(relation_error.empty() ? "native relation descriptor admission failed" : relation_error);
    const std::uint64_t relation_plan_text_bytes =
        static_cast<std::uint64_t>(serialize_plan_text(relation, tables).size());

    std::uint64_t logical_family_elements = 0U;
    std::uint64_t private_family_storage_upper_bytes = 0U;
    std::uint64_t maximum_padded_family_domain = 0U;
    for (const auto &family : relation.tensor_families) {
        std::uint64_t logical_elements = 1U, padded_domain = 1U;
        if (family.logical_shape.empty() || family.logical_shape.size() != family.padded_shape.size())
            fail("native admission found a malformed relation-family shape: " + family.id);
        for (std::size_t axis = 0U; axis < family.logical_shape.size(); ++axis) {
            if (family.logical_shape[axis] == 0U || family.padded_shape[axis] < family.logical_shape[axis] ||
                !checked_multiply_u64(logical_elements, family.logical_shape[axis], logical_elements) ||
                !checked_multiply_u64(padded_domain, family.padded_shape[axis], padded_domain))
                fail("native admission relation-family shape overflows: " + family.id);
        }
        if (logical_family_elements > std::numeric_limits<std::uint64_t>::max() - logical_elements)
            fail("native admission logical family element total overflows");
        logical_family_elements += logical_elements;
        maximum_padded_family_domain = std::max(maximum_padded_family_domain, padded_domain);
        if (family.visibility != "public") {
            if (logical_elements > (std::numeric_limits<std::uint64_t>::max() - 24U) / 32U ||
                private_family_storage_upper_bytes > std::numeric_limits<std::uint64_t>::max() - (logical_elements * 32U + 24U))
                fail("native admission private-family storage upper bound overflows");
            private_family_storage_upper_bytes += logical_elements * 32U + 24U;
        }
    }
    const auto admission_padded_axis = [](u64 value) {
        if (value == 0U || value > kMaxNativeDomain)
            fail("native admission padded shape axis is outside the canonical domain");
        u64 padded = 1U;
        while (padded < value) padded <<= 1U;
        return padded;
    };
    const auto public_y_family = std::find_if(relation.tensor_families.begin(), relation.tensor_families.end(),
                                              [](const TensorFamily &family) { return family.id == "public_Y"; });
    if (public_y_family == relation.tensor_families.end() || public_y_family->visibility != "public" ||
        public_y_family->logical_shape != std::vector<u64>{graph.nodes, config.architecture.classes} ||
        public_y_family->padded_shape.size() != 2U ||
        public_y_family->padded_shape[0] != admission_padded_axis(graph.nodes) ||
        public_y_family->padded_shape[1] != admission_padded_axis(config.architecture.classes))
        fail("native admission public_Y padded shape differs from the authenticated graph/output shape");
    const std::uint64_t native_public_y_padded_domain_elements =
        checked_shape_product(public_y_family->padded_shape, "public_Y.padded");

    std::uint64_t native_static_table_raw_i64_capacity_elements = 0U;
    std::uint64_t native_static_table_padded_field_elements = 0U;
    const auto add_static_table_capacity = [&](const std::string &id, const std::vector<i64> &raw,
                                               u64 padded_domain) {
        const auto family = std::find_if(relation.tensor_families.begin(), relation.tensor_families.end(),
                                         [&](const TensorFamily &candidate) { return candidate.id == id; });
        if (family == relation.tensor_families.end() || family->visibility != "public" || raw.empty() ||
            raw.capacity() < raw.size() || family->logical_shape != std::vector<u64>{raw.size()} ||
            family->padded_shape != std::vector<u64>{padded_domain} || padded_domain == 0U ||
            padded_domain != admission_padded_axis(static_cast<u64>(raw.size())))
            fail("native admission authenticated static-table shape/capacity mismatch: " + id);
        const auto raw_capacity = static_cast<u64>(raw.capacity());
        if (raw_capacity > std::numeric_limits<u64>::max() - native_static_table_raw_i64_capacity_elements ||
            padded_domain > std::numeric_limits<u64>::max() - native_static_table_padded_field_elements)
            fail("native admission static-table resident element count overflows");
        native_static_table_raw_i64_capacity_elements += raw_capacity;
        native_static_table_padded_field_elements += padded_domain;
    };
    add_static_table_capacity("static_lrelu", tables.lrelu, tables.lrelu_padded_domain);
    add_static_table_capacity("static_elu", tables.elu, tables.elu_padded_domain);
    add_static_table_capacity("static_exp", tables.exp, tables.exp_padded_domain);
    constexpr std::string_view sorted_roles = "table_key,query_x,query_y,query_key,query_prefix,merge_key,merge_delta,merge_marker,merge_inverse,merge_prefix,merge_bits,merge_delta_bits,query_bits";
    std::uint64_t sorted_lookup_scratch_upper_bytes = 0U;
    std::uint64_t maximum_sorted_lookup_scratch_bytes = 0U;
    std::uint64_t maximum_sorted_lookup_merge_logical_rows = 0U;
    std::uint64_t sorted_lookup_outer_commitment_count = 0U;
    std::uint64_t sorted_lookup_outer_segment_count = 0U;
    for (const auto &query : relation.sorted_lookup.queries) {
        const std::uint64_t record_domains = query.table_domain + query.query_domain + query.merge_domain;
        if (record_domains > (std::numeric_limits<std::uint64_t>::max() - 124U) / 107U)
            fail("native admission sorted-lookup scratch bound overflows");
        const std::uint64_t header_bytes = 124U + query.operation_id.size() + sorted_roles.size() + query.key_max_u128.size();
        const std::uint64_t scratch_bytes = record_domains * 107U + header_bytes;
        if (sorted_lookup_scratch_upper_bytes > std::numeric_limits<std::uint64_t>::max() - scratch_bytes)
            fail("native admission total sorted-lookup scratch bound overflows");
        sorted_lookup_scratch_upper_bytes += scratch_bytes;
        maximum_sorted_lookup_scratch_bytes = std::max(maximum_sorted_lookup_scratch_bytes, scratch_bytes);
        maximum_sorted_lookup_merge_logical_rows = std::max(maximum_sorted_lookup_merge_logical_rows, query.merge_logical_rows);
        const std::uint64_t query_segments = (query.query_domain + (1ULL << 18U) - 1U) / (1ULL << 18U);
        const std::uint64_t merge_segments = (query.merge_domain + (1ULL << 18U) - 1U) / (1ULL << 18U);
        const std::uint64_t outer_count = 9U + query.x_bits + query.y_bits;
        const std::uint64_t query_outer_segments = query_segments * 4U;
        const std::uint64_t merge_outer_segments = merge_segments * (5U + query.x_bits + query.y_bits);
        if (sorted_lookup_outer_commitment_count > std::numeric_limits<std::uint64_t>::max() - outer_count ||
            query_outer_segments > std::numeric_limits<std::uint64_t>::max() - merge_outer_segments ||
            sorted_lookup_outer_segment_count > std::numeric_limits<std::uint64_t>::max() - query_outer_segments - merge_outer_segments) {
            fail("native admission sorted-lookup outer commitment profile overflows");
        }
        sorted_lookup_outer_commitment_count += outer_count;
        sorted_lookup_outer_segment_count += query_outer_segments + merge_outer_segments;
    }
    std::uint64_t model_parameter_elements = 0U;
    std::uint64_t model_commitment_segment_count = 0U;
    for (const auto &item : model.tensors) {
        if (item.second.values.size() > std::numeric_limits<std::uint64_t>::max() - model_parameter_elements)
            fail("native admission model parameter element total overflows");
        model_parameter_elements += item.second.values.size();
        const auto &shape = item.second.shape;
        if (shape.empty()) fail("native admission found a model tensor with empty shape");
        std::uint64_t domain = 1U;
        const std::size_t begin = shape.size() == 3U && shape.front() == 1U ? 1U : 0U;
        for (std::size_t axis = begin; axis < shape.size(); ++axis) {
            if (shape[axis] == 0U || shape[axis] > (1ULL << 30U)) fail("native admission model tensor axis is invalid");
            std::uint64_t padded_axis = 1U;
            while (padded_axis < shape[axis]) {
                if (padded_axis > (1ULL << 29U)) fail("native admission model tensor padded axis overflows");
                padded_axis <<= 1U;
            }
            if (!checked_multiply_u64(domain, padded_axis, domain) || domain > (1ULL << 30U))
                fail("native admission model tensor padded domain overflows");
        }
        const std::uint64_t segments = (domain + (1ULL << 18U) - 1U) / (1ULL << 18U);
        if (model_commitment_segment_count > std::numeric_limits<std::uint64_t>::max() - segments)
            fail("native admission model commitment segment total overflows");
        model_commitment_segment_count += segments;
    }
    std::cout << "{\"schema\":\"zkdl_native_relation_admission_v2\",\"status\":\"PASS\",\"dataset\":\"" << config.dataset
              << "\",\"input_preprocess_id\":" << json_quote_config(config.input_preprocess_id)
              << ",\"batching_rule\":" << json_quote_config(config.batching_rule)
              << ",\"nodes\":" << graph.nodes << ",\"edges\":" << graph.edges << ",\"graphs\":" << graph.graphs
              << ",\"checkpoint_tensor_count\":" << model.tensors.size()
              << ",\"model_parameter_elements\":" << model_parameter_elements
              << ",\"model_commitment_segment_count\":" << model_commitment_segment_count
              << ",\"tensor_family_count\":" << relation.tensor_families.size()
              << ",\"operation_count\":" << relation.operation_families.size()
              << ",\"native_primitive_count\":" << plan_metrics.primitive_count
              << ",\"native_constraint_count\":" << plan_metrics.constraint_count
              << ",\"native_route_count\":" << plan_metrics.route_count
              << ",\"native_fac_binding_count\":" << plan_metrics.fac_binding_count
              << ",\"native_padding_family_count\":" << plan_metrics.padding_family_count
              << ",\"native_family_segment_count\":" << plan_metrics.family_segment_count
              << ",\"native_primitive_opening_segment_count\":" << plan_metrics.primitive_opening_segment_count
              << ",\"native_operation_descriptor_bytes\":" << plan_metrics.operation_descriptor_bytes
              << ",\"native_operation_descriptor_max_bytes\":" << plan_metrics.operation_descriptor_max_bytes
              << ",\"native_sumcheck_body_max_bytes\":" << plan_metrics.sumcheck_body_max_bytes
              << ",\"native_terminal_relation_body_max_bytes\":" << plan_metrics.terminal_relation_body_max_bytes
              << ",\"native_constraint_body_max_bytes\":" << plan_metrics.constraint_body_max_bytes
              << ",\"native_padding_record_body_max_bytes\":" << plan_metrics.padding_record_body_max_bytes
              << ",\"native_hidden_matmul_body_max_bytes\":" << plan_metrics.hidden_matmul_body_max_bytes
              << ",\"native_sparse_route_body_max_bytes\":" << plan_metrics.sparse_route_body_max_bytes
              << ",\"native_fac_line8_body_max_bytes\":" << plan_metrics.fac_line8_body_max_bytes
              << ",\"native_sorted_lookup_relation_body_max_bytes\":" << plan_metrics.sorted_lookup_relation_body_max_bytes
              << ",\"native_sorted_lookup_proof_body_max_bytes\":" << plan_metrics.sorted_lookup_proof_body_max_bytes
              << ",\"native_relation_proof_sections_wire_bytes_upper_bound\":" << plan_metrics.relation_proof_sections_wire_bytes_upper_bound
              << ",\"native_proof_wire_bytes_upper_bound\":" << plan_metrics.proof_wire_bytes_upper_bound
              << ",\"native_public_y_padded_domain_elements\":" << native_public_y_padded_domain_elements
              << ",\"native_static_table_raw_i64_capacity_elements\":" << native_static_table_raw_i64_capacity_elements
              << ",\"native_static_table_padded_field_elements\":" << native_static_table_padded_field_elements
              << ",\"native_commitment_section_wire_bytes_upper_bound\":" << plan_metrics.commitment_section_wire_bytes_upper_bound
              << ",\"native_operation_descriptor_resident_bytes_upper_bound\":" << plan_metrics.operation_descriptor_resident_bytes_upper_bound
              << ",\"relation_plan_text_bytes\":" << relation_plan_text_bytes
              << ",\"native_trace_i64_payload_bytes\":" << plan_metrics.trace_i64_payload_bytes
              << ",\"native_trace_field_payload_bytes\":" << plan_metrics.trace_field_payload_bytes
              << ",\"native_trace_u8_payload_bytes\":" << plan_metrics.trace_u8_payload_bytes
              << ",\"native_trace_object_metadata_bytes\":" << plan_metrics.trace_object_metadata_bytes
              << ",\"native_trace_forward_live_peak_bytes_upper_bound\":" << plan_metrics.trace_forward_live_peak_bytes_upper_bound
              << ",\"native_maximum_route_source_domain\":" << plan_metrics.maximum_route_source_domain
              << ",\"native_maximum_route_edge_domain\":" << plan_metrics.maximum_route_edge_domain
              << ",\"native_maximum_route_selector_factor_elements\":" << plan_metrics.maximum_route_selector_factor_elements
              << ",\"native_route_sumcheck_live_peak_bytes_upper_bound\":" << plan_metrics.route_sumcheck_live_peak_bytes_upper_bound
              << ",\"native_segmented_sumcheck_live_peak_bytes_upper_bound\":" << plan_metrics.segmented_sumcheck_live_peak_bytes_upper_bound;
    append_native_shape_counts_json(std::cout, plan_metrics);
    std::cout << ",\"sorted_lookup_query_count\":" << relation.sorted_lookup.queries.size()
              << ",\"maximum_sorted_lookup_merge_logical_rows\":" << maximum_sorted_lookup_merge_logical_rows
              << ",\"sorted_lookup_outer_commitment_count\":" << sorted_lookup_outer_commitment_count
              << ",\"sorted_lookup_outer_segment_count\":" << sorted_lookup_outer_segment_count
              << ",\"logical_family_elements\":" << logical_family_elements
              << ",\"maximum_padded_family_domain\":" << maximum_padded_family_domain
              << ",\"private_family_storage_upper_bound_bytes\":" << private_family_storage_upper_bytes
              << ",\"sorted_lookup_scratch_upper_bound_bytes\":" << sorted_lookup_scratch_upper_bytes
              << ",\"maximum_sorted_lookup_scratch_bytes\":" << maximum_sorted_lookup_scratch_bytes
              << ",\"integer_bound_ledger_digest\":\"" << relation.integer_bound_ledger_digest
              << "\",\"relation_registry_digest\":\"" << relation.registry_digest
              << "\",\"canonical_graph_digest\":\"" << digest_canonical_graph(graph)
              << "\",\"canonical_graph_nodes\":" << graph.nodes
              << ",\"canonical_graph_edges\":" << graph.edges
              << ",\"canonical_graph_count\":" << graph.graphs
              << ",\"canonical_graph_node_ptr\":[";
    for (std::size_t index = 0U; index < graph.node_ptr.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << graph.node_ptr[index];
    }
    std::cout << "],\"canonical_graph_edge_ptr\":[";
    for (std::size_t index = 0U; index < graph.edge_ptr.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << graph.edge_ptr[index];
    }
    std::cout << "]}\n";
    return 0;
}

struct AnalysisPlannedShape {
    StaticTables tables;
    IntegerBoundLedger ledger;
    std::string document;
    std::string sha256;
    std::map<std::string, u64> family_domains;
    u64 family_domain_count = 0U;
    std::string family_domain_sha256;
    std::string metadata_sha256;
    std::string edge_file_sha256;
    std::string node_ptr_file_sha256;
    std::string checkpoint_manifest_sha256;
    std::string checkpoint_tensor_export_sha256;
    u64 raw_edge_rows = 0U;
    u64 max_quantized_model_abs = 0U;
    u64 checkpoint_tensor_count = 0U;
};

AnalysisPlannedShape load_analysis_planned_shape(const Config &config,
                                                const Graph &graph,
                                                u64 raw_edge_records,
                                                u64 csr_max_in_degree,
                                                const std::filesystem::path &path) {
    if (path.empty() || std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path))
        fail("BLOCKED: analysis-only planned-shape sidecar is missing, symlinked, or not regular");
    const u64 sidecar_bytes = std::filesystem::file_size(path);
    if (sidecar_bytes == 0U || sidecar_bytes > (16U << 20U))
        fail("BLOCKED: analysis-only planned-shape sidecar exceeds its bounded metadata size");
    AnalysisPlannedShape result;
    result.document = file_text(path);
    result.sha256 = sha256_file(path);
    const auto &document = result.document;
    if (json_string_field(document, "schema") != "zkdl_analysis_only_planned_shape_v1" ||
        json_string_field(document, "status") != "ANALYSIS_ONLY_PLANNED" ||
        json_string_field(document, "dataset") != config.dataset ||
        json_string_field(document, "config_sha256") != sha256_file(config.config_path) ||
        json_string_field(document, "scale_chain_sha256") != config.scale_chain_digest ||
        json_scalar(document, "prepared_resources_claimed") != "false" ||
        json_scalar(document, "authenticated_resource_digests_present") != "false")
        fail("BLOCKED: planned-shape sidecar identity/status does not bind this analysis config");
    for (const auto *forbidden : {"static_table_descriptor_digest", "sorted_lookup_descriptor_digest",
                                  "integer_bound_ledger_digest", "srs_manifest_digest",
                                  "verifier_registry_digest", "model_blinder_sidecar_digest"}) {
        if (document.find(std::string("\"") + forbidden + "\"") != std::string::npos)
            fail("BLOCKED: planned-shape sidecar contains an authenticated resource identity field");
    }

    const auto input_identity = json_fragment(document, "input_identity", '{', '}');
    const auto dataset_dir = config.cache_root / config.dataset;
    const auto metadata_path = dataset_dir / "meta.cfg";
    const auto edge_path = dataset_dir / "edges.txt";
    const auto manifest_path = config.checkpoint_bundle / "manifest.json";
    const auto tensor_export_path = config.checkpoint_bundle / "tensors.txt";
    result.metadata_sha256 = sha256_file(metadata_path);
    result.edge_file_sha256 = sha256_file(edge_path);
    result.node_ptr_file_sha256 = config.dataset == "ppi" ? sha256_file(dataset_dir / "node_ptr.txt") : std::string();
    result.checkpoint_manifest_sha256 = sha256_file(manifest_path);
    result.checkpoint_tensor_export_sha256 = sha256_file(tensor_export_path);
    if (json_string_field(input_identity, "metadata_sha256") != result.metadata_sha256 ||
        json_string_field(input_identity, "edge_file_sha256") != result.edge_file_sha256 ||
        json_scalar(input_identity, "node_ptr_file_sha256") !=
            (config.dataset == "ppi" ? result.node_ptr_file_sha256 : std::string("null")) ||
        json_string_field(input_identity, "checkpoint_manifest_sha256") != result.checkpoint_manifest_sha256 ||
        json_string_field(input_identity, "checkpoint_tensor_export_sha256") != result.checkpoint_tensor_export_sha256)
        fail("BLOCKED: planned-shape source hashes do not match current graph/checkpoint files");

    const auto source_identity = json_fragment(document, "analysis_source_identity", '{', '}');
    const auto require_source_hash = [&](const std::string &field, const std::filesystem::path &source) {
        if (json_string_field(source_identity, field) != sha256_file(source))
            fail("BLOCKED: planned-shape source identity is stale: " + field);
    };
    require_source_hash("planner_sha256", config.baseline_root / "scripts/prepare_form_resources.py");
    require_source_hash("resource_report_sha256", config.baseline_root / "scripts/analysis_only_resource_report.py");
    require_source_hash("formal_phase_sha256", config.baseline_root / "scripts/formal_phase.py");
    require_source_hash("formal_runner_sha256", config.baseline_root / "scripts/formal_runner.py");

    const auto graph_shape = json_fragment(document, "graph", '{', '}');
    if (json_u64_field(graph_shape, "nodes") != graph.nodes ||
        json_u64_field(graph_shape, "normalized_edges") != graph.edges ||
        json_u64_field(graph_shape, "graphs") != graph.graphs ||
        json_u64_field(graph_shape, "csr_max_in_degree") != csr_max_in_degree ||
        json_u64_field(graph_shape, "features") != graph.features ||
        json_u64_field(graph_shape, "classes") != config.architecture.classes ||
        json_u64_field(graph_shape, "raw_edge_rows") != raw_edge_records)
        fail("BLOCKED: planned graph shape differs from exact native graph normalization");
    result.raw_edge_rows = raw_edge_records;

    const auto checkpoint = json_fragment(document, "checkpoint", '{', '}');
    const auto checkpoint_manifest = file_text(manifest_path);
    expect_manifest(checkpoint_manifest, config);
    result.checkpoint_tensor_count = json_u64_field(checkpoint_manifest, "tensor_count");
    result.max_quantized_model_abs = json_u64_field(checkpoint, "max_quantized_model_abs");
    if (result.checkpoint_tensor_count == 0U ||
        json_u64_field(checkpoint, "tensor_count") != result.checkpoint_tensor_count ||
        result.max_quantized_model_abs == 0U || result.max_quantized_model_abs > static_cast<u64>(std::numeric_limits<i64>::max()))
        fail("BLOCKED: planned checkpoint count or quantized bound is invalid");
    const auto checkpoint_shapes = manifest_tensor_shapes(checkpoint_manifest);
    if (checkpoint_shapes.size() != result.checkpoint_tensor_count)
        fail("BLOCKED: current checkpoint manifest shape index is incomplete");

    const auto table = json_fragment(document, "table_plan", '{', '}');
    auto &tables = result.tables;
    tables.quant_cfg_id = json_string_field(table, "quant_cfg_id");
    tables.generator_id = json_string_field(table, "generator_id");
    tables.version = json_string_field(table, "version");
    tables.scale_id = json_string_field(table, "scale_id");
    tables.domain_id = json_string_field(table, "domain_id");
    tables.row_order = json_string_field(table, "row_order");
    tables.padding_rule = json_string_field(table, "padding_rule");
    tables.spec_hash = json_string_field(table, "spec_hash");
    tables.lrelu_min = json_i64_value(table, "lrelu_min");
    tables.lrelu_max = json_i64_value(table, "lrelu_max");
    tables.elu_min = json_i64_value(table, "elu_min");
    tables.elu_max = json_i64_value(table, "elu_max");
    tables.exp_max = json_u64_value(table, "exp_max");
    tables.lrelu_rows = json_u64_field(table, "lrelu_rows");
    tables.elu_rows = json_u64_field(table, "elu_rows");
    tables.exp_rows = json_u64_field(table, "exp_rows");
    tables.lrelu_padded_domain = json_u64_field(table, "lrelu_padded_domain");
    tables.elu_padded_domain = json_u64_field(table, "elu_padded_domain");
    tables.exp_padded_domain = json_u64_field(table, "exp_padded_domain");
    tables.lrelu_value_min = json_i64_value(table, "lrelu_value_min");
    tables.lrelu_value_max = json_i64_value(table, "lrelu_value_max");
    tables.elu_value_min = json_i64_value(table, "elu_value_min");
    tables.elu_value_max = json_i64_value(table, "elu_value_max");
    tables.exp_value_min = json_i64_value(table, "exp_value_min");
    tables.exp_value_max = json_i64_value(table, "exp_value_max");
    tables.metadata_only = true;
    tables.authenticated = false;
    tables.descriptor_digest.clear();
    tables.digest.clear();
    tables.sorted_lookup_descriptor.clear();
    tables.sorted_lookup_descriptor_digest.clear();
    if (json_string_field(table, "schema") != "zkDL_gat_static_table_descriptor_v1" ||
        json_string_field(table, "static_table_id") != config.static_table_id ||
        tables.quant_cfg_id != config.quant_cfg_id ||
        tables.generator_id != "zkDL_baseline_local_static_table_generator_v1" ||
        tables.version != "v1" ||
        tables.scale_id != "scale_chain:s_delta=256,s_agg_pre=16,s_agg=16,s_exp=16" ||
        tables.domain_id != "range_bits:" + std::to_string(config.range_bits) ||
        tables.row_order != "ascending_index_min_to_max" ||
        tables.padding_rule != "next_power_of_two_zero_v1" || !is_digest(tables.spec_hash))
        fail("BLOCKED: planned static-table metadata has unknown source shape or identity");
    if (config.range_bits == 0U || config.range_bits > 25U || config.range_bits >= 63U)
        fail("BLOCKED: planned static-table range_bits is outside native bounds");
    const i64 input_bound = static_cast<i64>(1ULL << config.range_bits);
    const i64 expected_min = -input_bound;
    const u64 expected_exp_max = static_cast<u64>(checked_i128(
        static_cast<__int128>(input_bound) - round_div(-input_bound, 5, "planned LReLU minimum"),
        "planned ExpMap maximum"));
    const auto next_power = [](u64 value) {
        if (value == 0U || value > kMaxStaticTableRows) fail("planned static-table rows exceed native bounds");
        u64 padded = 1U;
        while (padded < value) padded <<= 1U;
        return padded;
    };
    if (tables.lrelu_min != expected_min || tables.lrelu_max != input_bound ||
        tables.elu_min != expected_min || tables.elu_max != input_bound ||
        tables.exp_max != expected_exp_max ||
        tables.lrelu_rows != checked_domain_rows(tables.lrelu_min, tables.lrelu_max, "planned LReLU") ||
        tables.elu_rows != checked_domain_rows(tables.elu_min, tables.elu_max, "planned ELU") ||
        tables.exp_rows != tables.exp_max + 1U ||
        tables.lrelu_padded_domain != next_power(tables.lrelu_rows) ||
        tables.elu_padded_domain != next_power(tables.elu_rows) ||
        tables.exp_padded_domain != next_power(tables.exp_rows) ||
        tables.lrelu_value_min != static_cast<i64>(std::round(-static_cast<double>(input_bound) * 0.2)) ||
        tables.lrelu_value_max != tables.lrelu_max ||
        tables.elu_value_min != static_cast<i64>(std::round((std::exp(-static_cast<double>(input_bound) / 16.0) - 1.0) * 16.0)) ||
        tables.elu_value_max != tables.elu_max ||
        tables.exp_value_min != std::max<i64>(1, static_cast<i64>(std::round(
            std::exp(-static_cast<double>(tables.exp_max) / 256.0) * 16.0))) || tables.exp_value_max != 16)
        fail("BLOCKED: planned table rows/ranges/padding differ from the canonical generator shape");
    std::ostringstream spec;
    spec << config.static_table_id << "\n" << tables.generator_id << "\n" << tables.version << "\n"
         << tables.scale_id << "\n" << tables.domain_id << "\n" << tables.row_order << "\n"
         << tables.padding_rule << "\n" << config.range_bits << "\n" << tables.lrelu_min << "\n"
         << tables.lrelu_max << "\n" << tables.exp_max << "\n" << tables.lrelu_padded_domain << "\n"
         << tables.elu_padded_domain << "\n" << tables.exp_padded_domain << "\n";
    if (tables.spec_hash != sha256_text(spec.str()))
        fail("BLOCKED: planned static-table spec hash does not match canonical table dimensions");

    const auto ledger_plan = json_fragment(document, "ledger_plan", '{', '}');
    auto &ledger = result.ledger;
    ledger.ledger_id = json_string_field(ledger_plan, "ledger_id");
    ledger.degree_bound_id = json_string_field(ledger_plan, "degree_bound_id");
    ledger.quant_cfg_id = json_string_field(ledger_plan, "quant_cfg_id");
    ledger.static_table_id = json_string_field(ledger_plan, "static_table_id");
    ledger.field_modulus = json_string_field(ledger_plan, "field_modulus");
    ledger.scale_chain_digest = json_string_field(ledger_plan, "scale_chain_sha256");
    ledger.static_table_descriptor_digest.clear();
    ledger.digest.clear();
    ledger.descriptor_digest.clear();
    ledger.authenticated = false;
    ledger.analysis_only_planned = true;
    if (json_string_field(ledger_plan, "schema") != "zkDL_gat_integer_bound_ledger_v1" ||
        ledger.ledger_id.empty() || ledger.degree_bound_id.empty() ||
        ledger.quant_cfg_id != config.quant_cfg_id || ledger.static_table_id != config.static_table_id ||
        ledger.field_modulus != scalar_modulus_decimal() || ledger.scale_chain_digest != config.scale_chain_digest)
        fail("BLOCKED: planned integer-bound ledger metadata does not bind current config");
    const auto family_objects = json_object_array(ledger_plan, "families", kMaxLedgerFamilies);
    const auto set_family_i64 = [&](const std::string &object, const std::string &key) {
        const i64 value = json_i64_value(object, key);
        validate_field_bound(value, "planned ledger " + key);
        return value;
    };
    std::vector<std::string> expected_family_ids;
    for (std::size_t layer = 0U; layer < config.architecture.hidden_heads.size(); ++layer)
        for (u64 head = 0U; head < config.architecture.hidden_heads[layer]; ++head)
            expected_family_ids.push_back("hidden/layer" + std::to_string(layer) + "/head" + std::to_string(head));
    for (u64 head = 0U; head < config.architecture.output_heads; ++head)
        expected_family_ids.push_back("output/head" + std::to_string(head));
    if (family_objects.size() != expected_family_ids.size())
        fail("BLOCKED: planned ledger family count differs from the configured architecture");
    for (std::size_t index = 0U; index < family_objects.size(); ++index) {
        const auto &object = family_objects[index];
        IntegerBoundFamily family;
        family.family_id = json_string_field(object, "family_id");
        family.layer = json_u64_field(object, "layer_index");
        family.head = json_u64_field(object, "head_index");
        family.output = json_u64_field(object, "output") != 0U;
        if (family.family_id != expected_family_ids[index] ||
            family.output != (family.family_id.rfind("output/", 0U) == 0U))
            fail("BLOCKED: planned ledger family identity/order is not canonical");
        family.max_input_abs = set_family_i64(object, "max_input_abs");
        family.max_weight_abs = set_family_i64(object, "max_weight_abs");
        family.max_attention_abs = set_family_i64(object, "max_attention_abs");
        family.max_bias_abs = set_family_i64(object, "max_bias_abs");
        family.max_score_abs = set_family_i64(object, "max_score_abs");
        family.max_table_output_abs = set_family_i64(object, "max_table_output_abs");
        family.max_delta = set_family_i64(object, "max_delta");
        family.max_exp = set_family_i64(object, "max_exp");
        family.max_sum = set_family_i64(object, "max_sum");
        family.max_aggregate_num = set_family_i64(object, "max_aggregate_num");
        family.max_dividend_abs = set_family_i64(object, "max_dividend_abs");
        family.max_aux_abs = set_family_i64(object, "max_aux_abs");
        family.max_divisor = set_family_i64(object, "max_divisor");
        family.max_field_product_abs = set_family_i64(object, "max_field_product_abs");
        if (family.max_weight_abs <= 0 || family.max_attention_abs <= 0 ||
            family.max_divisor <= 0 || family.layer >= config.architecture.hidden_dims.size() + 1U ||
            family.head >= (family.output ? config.architecture.output_heads : config.architecture.hidden_heads.at(family.layer)))
            fail("BLOCKED: planned ledger family bounds/coordinates are outside canonical ranges");
        ledger.families.push_back(std::move(family));
    }
    const auto maxima = json_fragment(ledger_plan, "maxima", '{', '}');
    ledger.max_score_abs = json_i64_value(maxima, "max_score_abs");
    ledger.max_table_output_abs = json_i64_value(maxima, "max_table_output_abs");
    ledger.max_delta = json_i64_value(maxima, "max_delta");
    ledger.max_exp = json_i64_value(maxima, "max_exp");
    ledger.max_sum = json_i64_value(maxima, "max_sum");
    ledger.max_aggregate_num = json_i64_value(maxima, "max_aggregate_num");
    ledger.max_dividend_abs = json_i64_value(maxima, "max_dividend_abs");
    ledger.max_aux_abs = json_i64_value(maxima, "max_aux_abs");
    ledger.max_divisor = json_i64_value(maxima, "max_divisor");
    for (const auto &family : ledger.families)
        if (family.max_score_abs > ledger.max_score_abs || family.max_table_output_abs > ledger.max_table_output_abs ||
            family.max_delta > ledger.max_delta || family.max_exp > ledger.max_exp || family.max_sum > ledger.max_sum ||
            family.max_aggregate_num > ledger.max_aggregate_num || family.max_dividend_abs > ledger.max_dividend_abs ||
            family.max_aux_abs > ledger.max_aux_abs || family.max_divisor > ledger.max_divisor)
            fail("BLOCKED: planned ledger family exceeds a declared global maximum");

    result.family_domains = json_u64_object(document, "family_domains");
    result.family_domain_count = json_u64_field(document, "family_domain_count");
    result.family_domain_sha256 = json_string_field(document, "family_domain_sha256");
    if (result.family_domain_count != result.family_domains.size() || !is_digest(result.family_domain_sha256))
        fail("BLOCKED: planned family domain map/count/hash is malformed");
    std::ostringstream family_domain_json;
    family_domain_json << '{';
    bool first = true;
    for (const auto &entry : result.family_domains) {
        if (!first) family_domain_json << ',';
        first = false;
        family_domain_json << json_quote_config(entry.first) << ':' << entry.second;
    }
    family_domain_json << '}';
    if (sha256_text(family_domain_json.str()) != result.family_domain_sha256)
        fail("BLOCKED: planned family domain map hash mismatch");
    return result;
}

int analysis_only_resource_plan(const Config &config,
                                const std::filesystem::path &planned_shape_path) {
    u64 raw_edge_records = 0U;
    u64 csr_max_in_degree = 0U;
    const Graph graph = load_graph_shape_for_resource_plan(config, &raw_edge_records,
                                                           &csr_max_in_degree);
    const bool planned_mode = !planned_shape_path.empty();
    AnalysisPlannedShape planned_shape;
    StaticTables tables;
    RelationPlan plan;
    if (planned_mode) {
        planned_shape = load_analysis_planned_shape(config, graph, raw_edge_records,
                                                    csr_max_in_degree, planned_shape_path);
        tables = planned_shape.tables;
        plan = build_relation_plan_planned(config, graph, tables, planned_shape.ledger);
    } else {
        tables = load_static_table_descriptor_shape(config);
        plan = build_relation_plan(config, graph, tables);
    }
    std::string relation_error;
    if (!planned_mode && !relation_plan_complete(plan, &relation_error))
        fail("BLOCKED: analysis-only resource plan metadata is incomplete: " + relation_error);
    if (planned_mode) {
        if (plan.schema_id != "zkDL_gat_inference_v1" || plan.proof_scope != "full_gat_forward" ||
            plan.binding_rule != "fac4dnn_reindex_v1" || plan.operation_families.empty() ||
            plan.tensor_families.empty() || plan.sorted_lookup.queries.empty() ||
            plan.integer_bound_ledger.authenticated || !plan.integer_bound_ledger.analysis_only_planned ||
            tables.authenticated || !tables.metadata_only)
            fail("BLOCKED: planned relation shape unexpectedly entered an authenticated resource path");
        const auto planned_lookup = json_fragment(planned_shape.document, "sorted_lookup_plan", '{', '}');
        if (json_u64_field(planned_lookup, "segment_size") != (1ULL << 18U))
            fail("BLOCKED: analytic sorted-lookup basis segment size differs from native profile");
        const auto planned_queries = json_object_array(planned_lookup, "query_families", 1U << 16U);
        if (planned_queries.size() != plan.sorted_lookup.queries.size())
            fail("BLOCKED: planned sorted-lookup query count differs from native shape plan");
        const auto planned_table_families = json_object_array(planned_lookup, "table_families", 16U);
        if (planned_table_families.size() != 3U)
            fail("BLOCKED: analytic sorted-lookup plan does not contain its three canonical tables");
        std::map<std::string, std::string> queries_by_id;
        std::map<std::string, const SortedLookupQueryPlan *> native_queries_by_table;
        u64 query_rows_total = 0U;
        u64 table_domain_total = 0U;
        u64 query_domain_total = 0U;
        u64 merge_domain_total = 0U;
        u64 scratch_record_payload_total = 0U;
        u64 scratch_bytes_total = 0U;
        u64 scratch_bytes_max_single_file = 0U;
        constexpr std::string_view sorted_roles =
            "table_key,query_x,query_y,query_key,query_prefix,merge_key,merge_delta,merge_marker,merge_inverse,merge_prefix,merge_bits,merge_delta_bits,query_bits";
        const auto add_plan_total = [&](u64 &total, u64 value, const std::string &label) {
            if (value > std::numeric_limits<u64>::max() - total)
                fail("BLOCKED: analytic sorted-lookup total overflows: " + label);
            total += value;
        };
        for (const auto &object : planned_queries) {
            const std::string id = json_string_field(object, "operation_id");
            if (id.empty() || !queries_by_id.emplace(id, object).second)
                fail("BLOCKED: planned sorted-lookup operation identifiers are empty or duplicated");
        }
        for (const auto &query : plan.sorted_lookup.queries) {
            const auto found = queries_by_id.find(query.operation_id);
            if (found == queries_by_id.end())
                fail("BLOCKED: native sorted-lookup operation is absent from analytic plan: " + query.operation_id);
            const auto &object = found->second;
            const auto query_shape = json_u64_array(object, "query_shape");
            u64 query_shape_product = 1U;
            for (const u64 axis : query_shape) {
                if (axis == 0U || query_shape_product > kMaxNativeDomain / axis)
                    fail("BLOCKED: planned sorted-lookup query shape overflows native bounds");
                query_shape_product *= axis;
            }
            if (json_string_field(object, "table_ref") != query.table_ref ||
                json_string_field(object, "operation") !=
                    (query.table_ref == "static_lrelu" ? "leaky_relu" :
                     query.table_ref == "static_exp" ? "exp_map" :
                     query.table_ref == "static_elu" ? "elu" : "") ||
                json_u64_field(object, "table_rows") != query.table_logical_rows ||
                json_u64_field(object, "query_rows") != query.query_logical_rows ||
                query_shape_product != query.query_logical_rows ||
                json_u64_field(object, "table_domain") != query.table_domain ||
                json_u64_field(object, "query_domain") != query.query_domain ||
                json_u64_field(object, "merge_rows") != query.merge_logical_rows ||
                json_u64_field(object, "merge_domain") != query.merge_domain ||
                json_u64_field(object, "segment_size") != query.segment_size ||
                json_u64_field(object, "table_segment_count") != query.table_segment_count ||
                json_u64_field(object, "query_segment_count") != query.query_segment_count ||
                json_u64_field(object, "merge_segment_count") != query.merge_segment_count ||
                json_u64_field(object, "x_bits") != query.x_bits ||
                json_u64_field(object, "y_bits") != query.y_bits ||
                json_string_field(object, "key_max_u128") != query.key_max_u128)
                fail("BLOCKED: native sorted-lookup dimensions/key widths differ from analytic plan: " + query.operation_id);
            const auto record_domains = query.table_domain + query.query_domain + query.merge_domain;
            u64 record_payload_bytes = 0U;
            if (!checked_multiply_u64(record_domains, 107U, record_payload_bytes))
                fail("BLOCKED: analytic sorted-lookup record payload overflows: " + query.operation_id);
            const u64 record_header_bytes = 124U + query.operation_id.size() +
                static_cast<u64>(sorted_roles.size()) + query.key_max_u128.size();
            if (record_payload_bytes > std::numeric_limits<u64>::max() - record_header_bytes)
                fail("BLOCKED: analytic sorted-lookup record file size overflows: " + query.operation_id);
            const u64 record_file_bytes = record_payload_bytes + record_header_bytes;
            if (json_u64_field(object, "scratch_header_bytes") != record_header_bytes ||
                json_u64_field(object, "scratch_record_payload_bytes") != record_payload_bytes ||
                json_u64_field(object, "scratch_bytes_upper_bound") != record_file_bytes)
                fail("BLOCKED: sorted-lookup scratch byte fields differ from native record sizing: " + query.operation_id);
            add_plan_total(query_rows_total, query.query_logical_rows, "query rows");
            add_plan_total(table_domain_total, query.table_domain, "table domains");
            add_plan_total(query_domain_total, query.query_domain, "query domains");
            add_plan_total(merge_domain_total, query.merge_domain, "merge domains");
            add_plan_total(scratch_record_payload_total, record_payload_bytes, "scratch payload");
            add_plan_total(scratch_bytes_total, record_file_bytes, "scratch bytes");
            scratch_bytes_max_single_file = std::max(scratch_bytes_max_single_file, record_file_bytes);
            native_queries_by_table.emplace(query.table_ref, &query);
        }
        if (json_u64_field(planned_lookup, "query_count") != planned_queries.size() ||
            json_u64_field(planned_lookup, "query_rows_total") != query_rows_total ||
            json_u64_field(planned_lookup, "table_domain_total") != table_domain_total ||
            json_u64_field(planned_lookup, "query_domain_total") != query_domain_total ||
            json_u64_field(planned_lookup, "merge_domain_total") != merge_domain_total ||
            json_u64_field(planned_lookup, "scratch_record_payload_bytes_total") != scratch_record_payload_total ||
            json_u64_field(planned_lookup, "scratch_bytes_total") != scratch_bytes_total ||
            json_u64_field(planned_lookup, "scratch_bytes_max_single_file") != scratch_bytes_max_single_file)
            fail("BLOCKED: sorted-lookup aggregate scratch/profile totals differ from native query shapes");
        std::set<std::string> seen_table_refs;
        for (const auto &object : planned_table_families) {
            const std::string table_ref = json_string_field(object, "table_ref");
            const auto query_it = native_queries_by_table.find(table_ref);
            if (query_it == native_queries_by_table.end() || !seen_table_refs.insert(table_ref).second)
                fail("BLOCKED: analytic sorted-lookup table family is unknown or duplicated: " + table_ref);
            const auto &query = *query_it->second;
            i64 input_min = 0, input_max = 0, output_min = 0, output_max = 0;
            std::string operation;
            if (table_ref == "static_lrelu") {
                input_min = tables.lrelu_min; input_max = tables.lrelu_max;
                output_min = tables.lrelu_value_min; output_max = tables.lrelu_value_max;
                operation = "leaky_relu";
            } else if (table_ref == "static_elu") {
                input_min = tables.elu_min; input_max = tables.elu_max;
                output_min = tables.elu_value_min; output_max = tables.elu_value_max;
                operation = "elu";
            } else if (table_ref == "static_exp") {
                input_min = 0; input_max = static_cast<i64>(tables.exp_max);
                output_min = tables.exp_value_min; output_max = tables.exp_value_max;
                operation = "exp_map";
            } else {
                fail("BLOCKED: analytic sorted-lookup table family is outside the native table set");
            }
            if (json_string_field(object, "operation") != operation ||
                json_u64_field(object, "logical_rows") != query.table_logical_rows ||
                json_u64_field(object, "padded_domain") != query.table_domain ||
                !json_bool_value(object, "sorted_unique") ||
                json_i64_value(object, "input_min") != input_min ||
                json_i64_value(object, "input_max") != input_max ||
                json_i64_value(object, "output_min") != output_min ||
                json_i64_value(object, "output_max") != output_max ||
                json_u64_field(object, "x_bits") != query.x_bits ||
                json_u64_field(object, "y_bits") != query.y_bits ||
                json_string_field(object, "key_encoding") != "packed_pair_u128" ||
                json_string_field(object, "key_max_u128") != query.key_max_u128 ||
                json_string_field(object, "padding_rule") != "zero_after_logical_rows_v1")
                fail("BLOCKED: analytic sorted-lookup table endpoints/layout differ from native table metadata: " + table_ref);
        }
        if (seen_table_refs.size() != 3U)
            fail("BLOCKED: analytic sorted-lookup plan omitted a canonical table family");
        if (planned_shape.family_domains.size() != plan.tensor_families.size() + 1U ||
            planned_shape.family_domains.find("*") == planned_shape.family_domains.end())
            fail("BLOCKED: planned SRS family map does not match native relation family inventory");
        u64 maximum_relation_domain = 1U;
        for (const auto &family : plan.tensor_families) {
            u64 domain = 1U;
            for (const auto axis : family.padded_shape) {
                if (!checked_multiply_u64(domain, axis, domain) || domain > kMaxNativeDomain)
                    fail("BLOCKED: native planned family domain overflows: " + family.id);
            }
            const auto expected = planned_shape.family_domains.find(family.id);
            if (expected == planned_shape.family_domains.end() || expected->second != domain)
                fail("BLOCKED: native relation family domain differs from analytic plan: " + family.id);
            maximum_relation_domain = std::max(maximum_relation_domain, domain);
        }
        if (planned_shape.family_domains.at("*") < maximum_relation_domain)
            fail("BLOCKED: planned wildcard SRS domain is below a relation family domain");
    }

    // Resource sizing uses public checkpoint shape metadata only.  Validate
    // every declared model tensor against the current config architecture so
    // model/SRS counts cannot silently omit a tensor or fall back to zero.
    const auto checkpoint_manifest_path = config.checkpoint_bundle / "manifest.json";
    if (std::filesystem::is_symlink(checkpoint_manifest_path) ||
        !std::filesystem::is_regular_file(checkpoint_manifest_path))
        fail("BLOCKED: analysis-only checkpoint manifest is missing or symlinked");
    const auto checkpoint_manifest = file_text(checkpoint_manifest_path);
    expect_manifest(checkpoint_manifest, config);
    const auto checkpoint_shapes = manifest_tensor_shapes(checkpoint_manifest);
    const u64 checkpoint_manifest_tensor_count = json_u64_field(checkpoint_manifest, "tensor_count");
    if (checkpoint_manifest_tensor_count == 0U ||
        checkpoint_shapes.size() != checkpoint_manifest_tensor_count)
        fail("BLOCKED: analysis-only checkpoint tensor index/count mismatch");
    if (planned_mode && checkpoint_manifest_tensor_count != planned_shape.checkpoint_tensor_count)
        fail("BLOCKED: native checkpoint manifest count differs from planned checkpoint identity");

    NativeDescriptorShapeFootprint footprint;
    if (!zkdl_gat_native_descriptor_shape_footprint(config, graph, tables, plan,
                                                    &footprint, &relation_error))
        fail(relation_error.empty() ? "analysis-only descriptor shape footprint failed" : relation_error);
    NativeRelationPlanMetrics native_metrics;
    if (!zkdl_gat_native_relation_plan_metrics(config, graph, tables, plan,
                                               &native_metrics, &relation_error, true))
        fail(relation_error.empty() ? "analysis-only typed resource metrics failed" : relation_error);

    const auto checked_sum = [](u64 left, u64 right, const std::string &label) {
        if (right > std::numeric_limits<u64>::max() - left)
            fail("BLOCKED: analysis-only resource count overflows: " + label);
        return left + right;
    };
    const auto checked_product = [](u64 left, u64 right, const std::string &label) {
        u64 product = 0U;
        if (!checked_multiply_u64(left, right, product))
            fail("BLOCKED: analysis-only resource count overflows: " + label);
        return product;
    };
    const auto next_power_of_two = [](u64 value, const std::string &label) {
        if (value == 0U || value > kMaxNativeDomain)
            fail("BLOCKED: analysis-only resource domain is invalid: " + label);
        u64 padded = 1U;
        while (padded < value) {
            if (padded > kMaxNativeDomain / 2U)
                fail("BLOCKED: analysis-only resource padded domain overflows: " + label);
            padded <<= 1U;
        }
        return padded;
    };

    constexpr u64 srs_basis_domain = 1ULL << 18U;
    u64 logical_family_elements = 0U;
    u64 private_family_storage_bytes = 0U;
    u64 maximum_padded_family_domain = 0U;
    u64 public_y_padded_domain = 0U;
    u64 static_table_padded_elements = 0U;
    for (const auto &family : plan.tensor_families) {
        u64 logical_elements = 1U, padded_domain = 1U;
        if (family.logical_shape.empty() || family.logical_shape.size() != family.padded_shape.size())
            fail("BLOCKED: analysis-only resource plan encountered malformed family axes: " + family.id);
        for (std::size_t axis = 0U; axis < family.logical_shape.size(); ++axis) {
            if (family.logical_shape[axis] == 0U || family.padded_shape[axis] < family.logical_shape[axis])
                fail("BLOCKED: analysis-only resource plan encountered invalid family axis: " + family.id);
            logical_elements = checked_product(logical_elements, family.logical_shape[axis], family.id + " logical elements");
            padded_domain = checked_product(padded_domain, family.padded_shape[axis], family.id + " padded domain");
        }
        logical_family_elements = checked_sum(logical_family_elements, logical_elements, "logical family elements");
        maximum_padded_family_domain = std::max(maximum_padded_family_domain, padded_domain);
        if (family.visibility != "public") {
            const u64 encoded = checked_sum(checked_product(logical_elements, 32U, "private family rows"),
                                            24U, "private family header");
            private_family_storage_bytes = checked_sum(private_family_storage_bytes, encoded,
                                                       "private family storage");
        }
        if (family.id == "public_Y") {
            if (family.visibility != "public")
                fail("BLOCKED: analysis-only public_Y is not declared public");
            public_y_padded_domain = padded_domain;
        }
    }
    if (public_y_padded_domain == 0U)
        fail("BLOCKED: analysis-only resource plan lacks public_Y family");
    for (const u64 domain : {tables.lrelu_padded_domain, tables.elu_padded_domain,
                             tables.exp_padded_domain})
        static_table_padded_elements = checked_sum(static_table_padded_elements, domain,
                                                   "static-table padded elements");
    const u64 static_table_rows = checked_sum(
        checked_sum(tables.lrelu_row_count(), tables.elu_row_count(), "static-table rows"),
        tables.exp_row_count(), "static-table rows");
    // json_i64_array uses repeated vector push_back.  The pinned libstdc++
    // growth rule reserves at most the next geometric capacity, bounded here
    // by 2x the authenticated streamed row count without allocating the rows.
    const u64 static_table_raw_capacity_upper =
        checked_product(static_table_rows, 2U, "static-table raw capacity upper bound");

    u64 sorted_scratch_bytes = 0U;
    u64 maximum_sorted_scratch_bytes = 0U;
    u64 maximum_sorted_merge_rows = 0U;
    constexpr std::string_view sorted_roles =
        "table_key,query_x,query_y,query_key,query_prefix,merge_key,merge_delta,merge_marker,merge_inverse,merge_prefix,merge_bits,merge_delta_bits,query_bits";
    for (const auto &query : plan.sorted_lookup.queries) {
        const u64 record_domains = checked_sum(
            checked_sum(query.table_domain, query.query_domain, "sorted lookup domains"),
            query.merge_domain, "sorted lookup domains");
        const u64 domain_bytes = checked_product(record_domains, 107U, "sorted lookup scratch domains");
        const u64 header = checked_sum(124U, checked_sum(query.operation_id.size(),
            checked_sum(sorted_roles.size(), query.key_max_u128.size(), "sorted lookup header"),
            "sorted lookup header"), "sorted lookup header");
        const u64 scratch = checked_sum(domain_bytes, header, "sorted lookup record scratch");
        sorted_scratch_bytes = checked_sum(sorted_scratch_bytes, scratch, "sorted lookup scratch total");
        maximum_sorted_scratch_bytes = std::max(maximum_sorted_scratch_bytes, scratch);
        maximum_sorted_merge_rows = std::max(maximum_sorted_merge_rows, query.merge_logical_rows);
    }

    u64 relation_srs_segment_count = 0U;
    u64 maximum_relation_srs_domain = 1U;
    for (const auto &family : plan.tensor_families) {
        u64 domain = 1U;
        for (const u64 axis : family.padded_shape)
            domain = checked_product(domain, axis, family.id + " SRS domain");
        maximum_relation_srs_domain = std::max(maximum_relation_srs_domain, domain);
        const u64 segments = (domain + srs_basis_domain - 1U) / srs_basis_domain;
        relation_srs_segment_count = checked_sum(relation_srs_segment_count, segments,
                                                 "relation family SRS segment count");
    }

    u64 model_parameter_elements = 0U;
    u64 model_tensor_count = 0U;
    u64 model_commitment_segment_count = 0U;
    u64 maximum_model_padded_domain = 1U;
    std::map<std::string, std::vector<u64>> expected_model_shapes;
    std::vector<std::pair<std::string, u64>> model_tensor_domains;
    const auto add_model_tensor = [&](const std::string &name,
                                      std::initializer_list<u64> axes,
                                      const std::string &label) {
        u64 logical = 1U, domain = 1U;
        std::size_t rank = 0U;
        std::vector<u64> logical_shape;
        logical_shape.reserve(axes.size());
        for (const auto axis : axes) {
            if (axis == 0U) fail("BLOCKED: analysis-only model tensor has an empty axis: " + label);
            logical_shape.push_back(axis);
            logical = checked_product(logical, axis, label + " element count");
            if (!(axes.size() == 3U && rank == 0U && axis == 1U))
                domain = checked_product(domain, next_power_of_two(axis, label), label + " padded domain");
            ++rank;
        }
        if (!expected_model_shapes.emplace(name, logical_shape).second)
            fail("BLOCKED: analysis-only architecture contains a duplicate model tensor: " + name);
        const auto manifest_shape = checkpoint_shapes.find(name);
        if (manifest_shape == checkpoint_shapes.end() || manifest_shape->second != logical_shape)
            fail("BLOCKED: checkpoint tensor shape/index differs from canonical architecture: " + name);
        model_parameter_elements = checked_sum(model_parameter_elements, logical,
                                               "model parameter elements");
        model_tensor_count = checked_sum(model_tensor_count, 1U, "model tensor count");
        maximum_model_padded_domain = std::max(maximum_model_padded_domain, domain);
        model_tensor_domains.emplace_back(name, domain);
        model_commitment_segment_count = checked_sum(model_commitment_segment_count,
            (domain + srs_basis_domain - 1U) / srs_basis_domain,
            "model commitment segment count");
    };
    u64 model_input_width = graph.features;
    for (std::size_t layer = 0U; layer < config.architecture.hidden_dims.size(); ++layer) {
        const u64 dimension = config.architecture.hidden_dims.at(layer);
        const u64 heads = config.architecture.hidden_heads.at(layer);
        for (u64 head = 0U; head < heads; ++head) {
            const std::string prefix = "hidden/layer" + std::to_string(layer) + "/head" + std::to_string(head) + "/";
            add_model_tensor(prefix + "seq/kernel", {1U, model_input_width, dimension}, "hidden projection");
            add_model_tensor(prefix + "attn_src/kernel", {1U, dimension, 1U}, "hidden source attention");
            add_model_tensor(prefix + "attn_dst/kernel", {1U, dimension, 1U}, "hidden destination attention");
        }
        model_input_width = checked_product(heads, dimension, "hidden concatenated width");
    }
    const u64 classes = config.architecture.classes;
    for (u64 head = 0U; head < config.architecture.output_heads; ++head) {
        const std::string prefix = "output/head" + std::to_string(head) + "/";
        add_model_tensor(prefix + "seq/kernel", {1U, model_input_width, classes}, "output projection");
        add_model_tensor(prefix + "attn_src/kernel", {1U, classes, 1U}, "output source attention");
        add_model_tensor(prefix + "attn_dst/kernel", {1U, classes, 1U}, "output destination attention");
        add_model_tensor(prefix + "output_bias", {classes}, "output bias");
    }
    if (expected_model_shapes.size() != checkpoint_shapes.size() ||
        expected_model_shapes.size() != checkpoint_manifest_tensor_count)
        fail("BLOCKED: checkpoint tensor index has tensors outside the canonical architecture");

    const u64 wildcard_srs_domain = std::max(maximum_relation_srs_domain,
                                             maximum_model_padded_domain);
    if (planned_mode) {
        if (planned_shape.family_domains.at("*") != wildcard_srs_domain)
            fail("BLOCKED: planned wildcard SRS domain does not match relation/model family domains");
        const auto srs_plan = json_fragment(planned_shape.document, "srs_plan", '{', '}');
        constexpr u64 expected_srs_basis_domain = 1ULL << 18U;
        const u64 expected_srs_frame_bytes = checked_sum(
            40U, checked_product(expected_srs_basis_domain, 48U,
                                 "planned SRS basis frame"),
            "planned SRS frame");
        if (json_u64_field(srs_plan, "basis_domain") != expected_srs_basis_domain ||
            json_u64_field(srs_plan, "frame_bytes") != expected_srs_frame_bytes)
            fail("BLOCKED: planned SRS basis/frame differs from the native source profile");
        const auto srs_families = json_fragment(srs_plan, "families", '{', '}');
        std::size_t srs_family_records = 0U;
        for (std::size_t position = 0U;
             (position = srs_families.find("\"logical_domain\"", position)) != std::string::npos;
             position += sizeof("\"logical_domain\"") - 1U)
            ++srs_family_records;
        if (srs_family_records != planned_shape.family_domains.size())
            fail("BLOCKED: planned SRS family count differs from the exact family-domain map");
        u64 expected_max_segment_count = 0U;
        for (const auto &entry : planned_shape.family_domains) {
            const auto planned_family = json_fragment(srs_families, entry.first, '{', '}');
            const u64 segment_count = (entry.second + expected_srs_basis_domain - 1U) /
                                      expected_srs_basis_domain;
            if (json_u64_field(planned_family, "logical_domain") != entry.second ||
                json_u64_field(planned_family, "segment_size") != expected_srs_basis_domain ||
                json_u64_field(planned_family, "segment_count") != segment_count)
                fail("BLOCKED: planned SRS domain/segment count differs for family " + entry.first);
            expected_max_segment_count = std::max(expected_max_segment_count, segment_count);
        }
        const u64 expected_preflight_srs_bytes = checked_sum(
            checked_sum(expected_srs_frame_bytes, 2U * (40U + 48U),
                        "planned SRS auxiliary frames"),
            checked_product(static_cast<u64>(planned_shape.family_domains.size()), 4096U,
                            "planned SRS family manifest"),
            "planned SRS materialized bytes");
        if (json_u64_field(srs_plan, "max_segment_count") != expected_max_segment_count ||
            json_u64_field(srs_plan, "estimated_materialized_bytes") != expected_preflight_srs_bytes)
            fail("BLOCKED: planned SRS totals differ from native family-domain sizing");
    }
    const u64 srs_domain_count = checked_sum(
        checked_sum(plan.tensor_families.size(), model_tensor_count, "SRS family count"),
        1U, "SRS wildcard family count");
    const u64 srs_segment_count = checked_sum(
        checked_sum(relation_srs_segment_count, model_commitment_segment_count,
                    "SRS concrete family segments"),
        (wildcard_srs_domain + srs_basis_domain - 1U) / srs_basis_domain,
        "SRS wildcard segments");
    const u64 srs_frame_bytes = checked_sum(40U,
        checked_product(srs_basis_domain, 48U, "SRS basis frame"), "SRS frame");
    const u64 srs_materialized_bytes = checked_sum(
        checked_sum(srs_frame_bytes, 2U * (40U + 48U), "SRS auxiliary frames"),
        checked_product(srs_domain_count, 4096U, "SRS family manifest reservation"),
        "SRS materialized bytes");

    u64 cache_file_count = 0U;
    u64 checkpoint_file_count = 0U;
    const auto count_regular_files = [&](const std::filesystem::path &root, u64 &count,
                                         u64 &total_bytes, const std::string &label) {
        if (std::filesystem::is_symlink(root) || !std::filesystem::is_directory(root))
            fail("BLOCKED: analysis-only resource file inventory is missing: " + root.string());
        for (const auto &entry : std::filesystem::recursive_directory_iterator(root)) {
            if (entry.is_symlink()) fail("BLOCKED: analysis-only resource inventory contains a symlink: " + entry.path().string());
            if (entry.is_regular_file()) {
                count = checked_sum(count, 1U, label + " count");
                total_bytes = checked_sum(total_bytes, entry.file_size(), label + " bytes");
            }
            else if (!entry.is_directory()) fail("BLOCKED: analysis-only resource inventory contains a special file");
        }
    };
    u64 cache_file_bytes = 0U, checkpoint_file_bytes = 0U;
    count_regular_files(config.cache_root / config.dataset, cache_file_count,
                        cache_file_bytes, "cache inventory");
    count_regular_files(config.checkpoint_bundle, checkpoint_file_count,
                        checkpoint_file_bytes, "checkpoint inventory");
    const u64 relation_plan_text_base_bytes = serialize_plan_text(plan, tables).size();
    u64 planned_identifier_text_reserve_bytes = 0U;
    if (planned_mode) {
        const auto digest_text_reserve = [&](const std::string &value) -> u64 {
            return value.size() < 64U ? static_cast<u64>(64U - value.size()) : 0U;
        };
        planned_identifier_text_reserve_bytes = checked_sum(
            checked_sum(digest_text_reserve(plan.static_table_descriptor_digest),
                        digest_text_reserve(plan.integer_bound_ledger_digest),
                        "planned relation identifier text reserve"),
            checked_sum(
                checked_sum(digest_text_reserve(plan.integer_bound_ledger_descriptor_digest),
                            digest_text_reserve(plan.sorted_lookup.descriptor_digest),
                            "planned relation identifier text reserve"),
                checked_sum(digest_text_reserve(plan.sorted_lookup.table_digest),
                            digest_text_reserve(tables.digest),
                            "planned relation identifier text reserve"),
                "planned relation identifier text reserve"),
            "planned relation identifier text reserve");
        const auto formal_sorted_descriptor =
            std::filesystem::relative(config.sorted_lookup_descriptor_path, config.baseline_root).generic_string();
        planned_identifier_text_reserve_bytes = checked_sum(
            planned_identifier_text_reserve_bytes,
            static_cast<u64>(formal_sorted_descriptor.size()),
            "planned sorted-descriptor path text reserve");
    }
    const u64 relation_plan_text_bytes = checked_sum(
        relation_plan_text_base_bytes, planned_identifier_text_reserve_bytes,
        "relation plan text capacity");
    const u64 native_static_table_raw_i64_bytes_upper =
        checked_product(static_table_raw_capacity_upper, sizeof(i64), "static-table raw bytes");
    const u64 native_static_table_padded_field_bytes =
        checked_product(static_table_padded_elements, 32U, "static-table padded field bytes");

    u64 padded_edge_domain = 1U;
    while (padded_edge_domain < graph.edges) {
        if (padded_edge_domain > kMaxNativeDomain / 2U)
            fail("BLOCKED: analysis-only padded edge domain exceeds native bound");
        padded_edge_domain <<= 1U;
    }
    const auto source_path = [&](const std::string &relative) {
        const auto path = config.baseline_root / relative;
        if (std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path))
            fail("BLOCKED: analysis-only source provenance file is missing: " + path.string());
        return sha256_file(path);
    };
    const auto metadata_path = config.cache_root / config.dataset / "meta.cfg";
    const auto edge_path = config.cache_root / config.dataset / "edges.txt";
    const auto node_ptr_path = config.cache_root / config.dataset / "node_ptr.txt";
    const auto checkpoint_tensor_export_path = config.checkpoint_bundle / "tensors.txt";
    const std::string config_file_digest = sha256_file(config.config_path);
    const std::string metadata_digest = sha256_file(metadata_path);
    const std::string edge_file_digest = sha256_file(edge_path);
    const std::string node_ptr_file_digest = config.dataset == "ppi" ? sha256_file(node_ptr_path) : std::string();
    const std::string checkpoint_tensor_export_digest = sha256_file(checkpoint_tensor_export_path);
    const std::string planner_source_digest = source_path("scripts/prepare_form_resources.py");
    const std::string runner_source_digest = source_path("gat_adapter/gat_runner.cu");
    const std::string resource_report_digest = source_path("scripts/analysis_only_resource_report.py");
    const std::string formal_phase_digest = source_path("scripts/formal_phase.py");
    const std::string formal_runner_digest = source_path("scripts/formal_runner.py");
    const std::string bridge_header_digest = source_path("gat_adapter/gat_native_bridge.hpp");
    const std::string native_source_digest = source_path("gat_adapter/gat_native_core.cu");
    const std::string native_header_digest = source_path("gat_adapter/gat_native_core.cuh");
    const std::string proof_header_digest = source_path("zkrelation-proof.cuh");

    std::cout << "{\"schema\":\"zkdl_analysis_only_resource_plan_v1\",\"status\":"
              << json_quote_config(planned_mode ? "ANALYSIS_ONLY_PLANNED" : "ANALYSIS_ONLY")
              << ",\"dataset\":" << json_quote_config(config.dataset)
              << ",\"config_sha256\":" << json_quote_config(config_file_digest)
              << ",\"metadata_sha256\":" << json_quote_config(metadata_digest)
              << ",\"raw_edge_file_sha256\":" << json_quote_config(edge_file_digest)
              << ",\"node_ptr_file_sha256\":"
              << (config.dataset == "ppi" ? json_quote_config(node_ptr_file_digest) : "null")
              << ",\"checkpoint_tensor_export_sha256\":" << json_quote_config(checkpoint_tensor_export_digest)
              << ",\"scale_chain_sha256\":" << json_quote_config(config.scale_chain_digest)
              << ",\"planner_source_sha256\":" << json_quote_config(planner_source_digest)
              << ",\"gat_runner_source_sha256\":" << json_quote_config(runner_source_digest)
              << ",\"analysis_resource_report_sha256\":" << json_quote_config(resource_report_digest)
              << ",\"formal_phase_source_sha256\":" << json_quote_config(formal_phase_digest)
              << ",\"formal_runner_source_sha256\":" << json_quote_config(formal_runner_digest)
              << ",\"native_bridge_header_sha256\":" << json_quote_config(bridge_header_digest)
              << ",\"native_core_source_sha256\":" << json_quote_config(native_source_digest)
              << ",\"native_core_header_sha256\":" << json_quote_config(native_header_digest)
              << ",\"relation_proof_header_sha256\":" << json_quote_config(proof_header_digest);
    if (planned_mode) {
        std::cout << ",\"planned_shape_sha256\":" << json_quote_config(planned_shape.sha256)
                  << ",\"planned_metadata_status\":\"ANALYSIS_ONLY_PLANNED\""
                  << ",\"planned_identifier_text_reserve_bytes\":" << planned_identifier_text_reserve_bytes
                  << ",\"prepared_resources_claimed\":false"
                  << ",\"authenticated_resource_digests_present\":false";
    } else {
        std::cout << ",\"static_table_descriptor_digest\":" << json_quote_config(tables.descriptor_digest)
                  << ",\"sorted_lookup_descriptor_digest\":" << json_quote_config(tables.sorted_lookup_descriptor_digest)
                  << ",\"integer_bound_ledger_digest\":" << json_quote_config(plan.integer_bound_ledger_digest)
                  << ",\"relation_registry_digest\":" << json_quote_config(plan.registry_digest);
    }
    std::cout << ",\"edge_count_basis\":\"exact_load_graph_normalization_with_configured_symmetry_dedup_and_self_loops\""
              << ",\"raw_edge_record_count\":" << raw_edge_records
              << ",\"nodes\":" << graph.nodes << ",\"features\":" << graph.features
              << ",\"classes\":" << config.architecture.classes
              << ",\"graphs\":" << graph.graphs << ",\"normalized_edge_count\":" << graph.edges
              << ",\"csr_max_in_degree\":" << csr_max_in_degree
              << ",\"padded_edge_domain\":" << padded_edge_domain
              << ",\"tensor_family_count\":" << plan.tensor_families.size()
              << ",\"operation_count\":" << footprint.operation_count
              << ",\"fac_binding_count\":" << footprint.fac_binding_count
              << ",\"route_count\":" << footprint.route_count
              << ",\"expected_unique_outer_map_bytes\":" << footprint.expected_unique_outer_map_bytes
              << ",\"expected_unique_active_mask_bytes\":" << footprint.expected_unique_active_mask_bytes
              << ",\"expected_feature_map_bytes_upper_bound\":" << footprint.expected_feature_map_bytes
              << ",\"received_selector_array_bytes_upper_bound\":" << footprint.received_selector_array_bytes
              << ",\"ordinary_descriptor_capacity_bytes_both_trees\":" << footprint.ordinary_descriptor_capacity_bytes
              << ",\"descriptor_object_header_bytes\":" << footprint.descriptor_object_header_bytes
              << ",\"operation_descriptor_wire_bytes\":" << footprint.operation_descriptor_wire_bytes
              << ",\"max_operation_descriptor_wire_bytes\":" << footprint.operation_descriptor_max_bytes
              << ",\"descriptor_compare_buffer_bytes\":" << footprint.descriptor_compare_buffer_bytes
              << ",\"operation_descriptor_resident_bytes_upper_bound\":"
              << footprint.operation_descriptor_resident_bytes_upper_bound
              << ",\"model_parameter_elements\":" << model_parameter_elements
              << ",\"model_commitment_segment_count\":" << model_commitment_segment_count
              << ",\"sorted_lookup_query_count\":" << plan.sorted_lookup.queries.size()
              << ",\"logical_family_elements\":" << logical_family_elements
              << ",\"maximum_padded_family_domain\":" << maximum_padded_family_domain
              << ",\"private_family_storage_upper_bound_bytes\":" << private_family_storage_bytes
              << ",\"sorted_lookup_scratch_upper_bound_bytes\":" << sorted_scratch_bytes
              << ",\"maximum_sorted_lookup_scratch_bytes\":" << maximum_sorted_scratch_bytes
              << ",\"maximum_sorted_lookup_merge_logical_rows\":" << maximum_sorted_merge_rows
              << ",\"native_primitive_count\":" << native_metrics.primitive_count
              << ",\"native_constraint_count\":" << native_metrics.constraint_count
              << ",\"native_route_count\":" << native_metrics.route_count
              << ",\"native_fac_binding_count\":" << native_metrics.fac_binding_count
              << ",\"native_padding_family_count\":" << native_metrics.padding_family_count
              << ",\"native_family_segment_count\":" << native_metrics.family_segment_count
              << ",\"native_primitive_opening_segment_count\":" << native_metrics.primitive_opening_segment_count
              << ",\"native_operation_descriptor_bytes\":" << native_metrics.operation_descriptor_bytes
              << ",\"native_operation_descriptor_max_bytes\":" << native_metrics.operation_descriptor_max_bytes
              << ",\"native_sumcheck_body_max_bytes\":" << native_metrics.sumcheck_body_max_bytes
              << ",\"native_terminal_relation_body_max_bytes\":" << native_metrics.terminal_relation_body_max_bytes
              << ",\"native_constraint_body_max_bytes\":" << native_metrics.constraint_body_max_bytes
              << ",\"native_padding_record_body_max_bytes\":" << native_metrics.padding_record_body_max_bytes
              << ",\"native_hidden_matmul_body_max_bytes\":" << native_metrics.hidden_matmul_body_max_bytes
              << ",\"native_sparse_route_body_max_bytes\":" << native_metrics.sparse_route_body_max_bytes
              << ",\"native_fac_line8_body_max_bytes\":" << native_metrics.fac_line8_body_max_bytes
              << ",\"native_sorted_lookup_relation_body_max_bytes\":" << native_metrics.sorted_lookup_relation_body_max_bytes
              << ",\"native_sorted_lookup_proof_body_max_bytes\":" << native_metrics.sorted_lookup_proof_body_max_bytes
              << ",\"native_relation_proof_sections_wire_bytes_upper_bound\":" << native_metrics.relation_proof_sections_wire_bytes_upper_bound
              << ",\"native_proof_wire_bytes_upper_bound\":" << native_metrics.proof_wire_bytes_upper_bound
              << ",\"native_public_y_padded_domain_elements\":" << public_y_padded_domain
              << ",\"native_static_table_raw_i64_capacity_elements_upper_bound\":" << static_table_raw_capacity_upper
              << ",\"native_static_table_raw_i64_capacity_elements\":" << static_table_raw_capacity_upper
              << ",\"native_static_table_padded_field_elements\":" << static_table_padded_elements
              << ",\"native_static_table_raw_i64_bytes_upper_bound\":" << native_static_table_raw_i64_bytes_upper
              << ",\"native_static_table_padded_field_bytes\":" << native_static_table_padded_field_bytes
              << ",\"native_commitment_section_wire_bytes_upper_bound\":" << native_metrics.commitment_section_wire_bytes_upper_bound
              << ",\"native_operation_descriptor_resident_bytes_upper_bound\":" << footprint.operation_descriptor_resident_bytes_upper_bound
              << ",\"native_trace_i64_payload_bytes\":" << native_metrics.trace_i64_payload_bytes
              << ",\"native_trace_field_payload_bytes\":" << native_metrics.trace_field_payload_bytes
              << ",\"native_trace_u8_payload_bytes\":" << native_metrics.trace_u8_payload_bytes
              << ",\"native_trace_object_metadata_bytes\":" << native_metrics.trace_object_metadata_bytes
              << ",\"native_trace_forward_live_peak_bytes_upper_bound\":" << native_metrics.trace_forward_live_peak_bytes_upper_bound
              << ",\"native_maximum_route_source_domain\":" << native_metrics.maximum_route_source_domain
              << ",\"native_maximum_route_edge_domain\":" << native_metrics.maximum_route_edge_domain
              << ",\"native_maximum_route_selector_factor_elements\":" << native_metrics.maximum_route_selector_factor_elements
              << ",\"native_route_sumcheck_live_peak_bytes_upper_bound\":" << native_metrics.route_sumcheck_live_peak_bytes_upper_bound
              << ",\"native_segmented_sumcheck_live_peak_bytes_upper_bound\":" << native_metrics.segmented_sumcheck_live_peak_bytes_upper_bound;
    append_native_shape_counts_json(std::cout, native_metrics);
    std::cout << ",\"relation_plan_text_bytes\":" << relation_plan_text_bytes
              << ",\"checkpoint_tensor_count\":" << checkpoint_manifest_tensor_count
              << ",\"srs_basis_domain\":" << srs_basis_domain
              << ",\"srs_frame_bytes\":" << srs_frame_bytes
              << ",\"srs_family_count\":" << srs_domain_count
              << ",\"srs_segment_count\":" << srs_segment_count
              << ",\"srs_materialized_bytes_upper_bound\":" << srs_materialized_bytes
              << ",\"model_tensor_count\":" << model_tensor_count
              << ",\"cache_file_count\":" << cache_file_count
              << ",\"cache_file_bytes\":" << cache_file_bytes
              << ",\"checkpoint_file_count\":" << checkpoint_file_count
              << ",\"checkpoint_file_bytes\":" << checkpoint_file_bytes
              << ",\"checkpoint_manifest_sha256\":" << json_quote_config(sha256_file(checkpoint_manifest_path))
              << ",\"srs_family_domains\":["
              << "{\"id\":\"*\",\"logical_domain\":" << wildcard_srs_domain
              << ",\"segment_count\":" << ((wildcard_srs_domain + srs_basis_domain - 1U) / srs_basis_domain) << "}";
    for (const auto &family : plan.tensor_families) {
        u64 domain = 1U;
        for (const u64 axis : family.padded_shape) domain = checked_product(domain, axis, family.id + " output domain");
        std::cout << ",{\"id\":" << json_quote_config(family.id)
                  << ",\"logical_domain\":" << domain
                  << ",\"segment_count\":" << ((domain + srs_basis_domain - 1U) / srs_basis_domain) << "}";
    }
    for (const auto &model_tensor : model_tensor_domains) {
        std::cout << ",{\"id\":" << json_quote_config(model_tensor.first)
                  << ",\"logical_domain\":" << model_tensor.second
                  << ",\"segment_count\":" << ((model_tensor.second + srs_basis_domain - 1U) / srs_basis_domain) << "}";
    }
    std::cout << "],\"proof_artifact_created\":false,\"dataset_prepared\":false,\"formal_admission_run\":false"
              << ",\"independent_verification_run\":false,\"result_collection_run\":false}\n";
    return 0;
}

int verify_only(const Config &config, const CliOptions &options) {
    const auto directory = run_path(config, options.run_dir);
    auto run_config = config_for_run(config, directory);
    run_config.public_verifier_mode = true;
    const auto statement_path = run_file_path(directory, options.statement, "statement.txt");
    const auto envelope_path = run_file_path(directory, options.envelope, "proof.meta");
    const auto proof_path = run_file_path(directory, options.proof, "proof.bin");
    PublicStatement statement;
    ProofEnvelope envelope;
    Backend backend = options.has_backend ? options.backend : Backend::Cpu;
    bool statement_loaded = false;
    bool envelope_loaded = false;
    std::string error;
    try {
        if (!read_statement(statement_path, &statement, &error)) fail(error);
        statement_loaded = true;
        envelope = read_envelope(envelope_path);
        envelope_loaded = true;
        if (envelope.schema_id != "zkDL_gat_proof_envelope_v1" ||
            envelope.commitment_time_semantics != "native_tensor_commitments" || envelope.status != "PROVED" ||
            envelope.native_core_api != "zkdl_native_gat_bridge_v1" ||
            envelope.proof_id.empty() || envelope.statement_id != statement.statement_id || envelope.proof_scope != statement.proof_scope ||
            envelope.verifier_reads_witness || envelope.silent_fallback || !envelope.output_bound || envelope.proof_size_bytes == 0) {
            fail("BLOCKED: proof envelope is not a complete independent-verification artifact");
        }
        backend = parse_backend(envelope.backend);
        if (options.has_backend && options.backend != backend) fail("requested backend does not match proof envelope");
        if (envelope.static_table_descriptor_digest != statement.static_table_descriptor_digest ||
            envelope.scale_chain_digest != statement.scale_chain_digest || envelope.integer_bound_ledger_id != statement.integer_bound_ledger_id ||
            envelope.integer_bound_ledger_digest != statement.integer_bound_ledger_digest ||
            envelope.integer_bound_ledger_descriptor_digest != statement.integer_bound_ledger_descriptor_digest) {
            fail("BLOCKED: proof envelope relation identities do not bind the public statement");
        }

        const auto model = load_model_metadata(run_config);
        const auto graph = load_graph(run_config);
        const auto tables = build_static_tables(run_config, graph);
        const auto relation = build_relation_plan(run_config, graph, tables);
        std::string relation_error;
        if (!relation_plan_complete(relation, &relation_error)) fail(relation_error);
        validate_statement_binding(run_config, graph, model, tables, relation, statement);

        const auto public_output_path = local_path(directory, statement.public_output_path, "public_output_path");
        const auto public_output = read_public_output(public_output_path, statement.nodes, statement.classes);
        const auto public_output_family = std::find_if(relation.tensor_families.begin(), relation.tensor_families.end(),
                                                       [](const TensorFamily &family) { return family.id == "public_Y"; });
        if (public_output_family == relation.tensor_families.end()) fail("RelationPlan has no public_Y family");
        if (statement.output_id != digest_matrix(public_output) ||
            canonical_public_field_digest(public_output, *public_output_family) != statement.public_output_digest) {
            fail("public output shape or digest does not match the public statement");
        }
        if (backend == Backend::Gpu && !envelope.gpu_kernel_executed) { fail("BLOCKED: GPU proof envelope does not attest CUDA execution"); }
        if (backend == Backend::Cpu && envelope.gpu_kernel_executed) { fail("BLOCKED: CPU proof envelope claims CUDA execution"); }
        const auto proof = read_bytes(proof_path);
        if (proof.size() != envelope.proof_size_bytes) fail("proof byte size does not match proof envelope");
        const auto public_context = build_native_public_context(run_config, relation, public_output);
        NativeVerifyRequest request;
        request.config = &run_config;
        request.statement = &statement;
        request.envelope = &envelope;
        request.proof_bytes = &proof;
        request.backend = backend;
        request.graph = &graph;
        request.public_context = &public_context;
        request.relation = &relation;
        request.tables = &tables;
        if (!zkdl_gat_native_verify(request, &error)) fail(error.empty() ? "native verifier rejected proof" : error);
        write_envelope(envelope_path, envelope);
        std::ofstream result(directory / "verification.result");
        if (!result) fail("cannot write verification.result");
        result << "status=VERIFIED\nstatement_id=" << statement.statement_id << "\nproof_id=" << envelope.proof_id << "\n";
        finish_output(result, directory / "verification.result", "verification result");
        write_run_record(directory / "run.record", record_from_envelope(run_config, statement, envelope, "VERIFIED"));
        std::cout << "verification=VERIFIED statement_id=" << statement.statement_id << "\n";
        return 0;
    } catch (const std::exception &exception) {
        const std::string message = std::string("BLOCKED: independent verification failed: ") + exception.what();
        try {
            const auto record = rejected_record(run_config, backend, statement_loaded ? &statement : nullptr, message,
                                                envelope_loaded && envelope.gpu_kernel_executed);
            write_run_record(directory / "run.record", record);
            std::ofstream result(directory / "verification.result");
            if (result) {
                result << "status=BLOCKED\nstatement_id=" << (statement_loaded ? statement.statement_id : "UNAVAILABLE")
                       << "\nproof_id=" << (envelope_loaded ? envelope.proof_id : "UNAVAILABLE") << "\nerror=" << message << "\n";
                finish_output(result, directory / "verification.result", "blocked verification result");
            }
        } catch (...) {}
        std::cerr << "verification=BLOCKED " << message << '\n';
        return 2;
    }
}

}

}

int main(int argc, char **argv) {
    try {
        const auto options = zkdl_gat::parse_cli(argc, argv);
        const bool planned_shape_mode = options.command == "resource-plan" && !options.planned_shape.empty();
        const auto config = zkdl_gat::load_config(options.config,
                                                  options.command == "admit" || options.command == "resource-plan",
                                                  planned_shape_mode);
        if (options.command == "inspect") {
            std::cout << "dataset=" << config.dataset << " baseline_root=" << config.baseline_root.string()
                      << " checkpoint_bundle=" << config.checkpoint_bundle.string()
                      << " backend=" << (options.has_backend ? zkdl_gat::backend_name(options.backend) : "unspecified") << "\n";
            return 0;
        }
        if (options.command == "admit") return zkdl_gat::admit(config);
        if (options.command == "resource-plan")
            return zkdl_gat::analysis_only_resource_plan(config, options.planned_shape);
        if (options.command == "prepare") return zkdl_gat::prepare(config, options);
        if (options.command == "run" || options.command == "prove") return zkdl_gat::produce(config, options);
        return zkdl_gat::verify_only(config, options);
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
