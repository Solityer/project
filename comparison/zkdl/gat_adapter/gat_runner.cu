// Baseline-local GAT relation adapter for the official zkDL CUDA tree.
//
// This translation unit owns only canonical input/model decoding, the fixed
// point GAT witness relation, and the production command line.  Proof bytes
// are produced and checked exclusively through gat_native_bridge.hpp; the
// weak hook is deliberately fail-closed when the official core bridge is not
// linked.

#include "gat_native_bridge.hpp"

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
#include <limits>
#include <map>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <openssl/bn.h>

namespace zkdl_gat {
namespace {

using u64 = std::uint64_t;
using i64 = std::int64_t;

// Relation resources are authenticated before materialization, but their
// declared dimensions are still untrusted input.  Keep every table/family
// container and every verifier-side range scan explicitly bounded.
constexpr u64 kMaxStaticTableRows = 1ULL << 27;
constexpr u64 kMaxLedgerFamilies = 4096;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error("zkdl-gat: " + message);
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::map<std::string, std::string> read_kv(const std::filesystem::path& path) {
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
        if (equal == std::string::npos) {
            fail("malformed key=value line " + std::to_string(line_no) + " in " + path.string());
        }
        const std::string key = trim(line.substr(0, equal));
        const std::string value = trim(line.substr(equal + 1));
        if (key.empty() || value.empty()) fail("empty key/value in " + path.string());
        if (!result.emplace(key, value).second) fail("duplicate key " + key + " in " + path.string());
    }
    return result;
}

u64 parse_u64(const std::string& text, const std::string& what) {
    const std::string value_text = trim(text);
    if (value_text.empty() || value_text.front() == '-') {
        fail("invalid unsigned integer for " + what + ": " + text);
    }
    std::size_t used = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(value_text, &used, 0);
    } catch (...) {
        fail("invalid unsigned integer for " + what + ": " + text);
    }
    if (used != value_text.size()) fail("invalid trailing characters for " + what + ": " + text);
    return static_cast<u64>(value);
}

i64 parse_i64(const std::string& text, const std::string& what) {
    const std::string value_text = trim(text);
    if (value_text.empty()) fail("invalid signed integer for " + what + ": " + text);
    std::size_t used = 0;
    long long value = 0;
    try {
        value = std::stoll(value_text, &used, 0);
    } catch (...) {
        fail("invalid signed integer for " + what + ": " + text);
    }
    if (used != value_text.size()) fail("invalid trailing characters for " + what + ": " + text);
    return static_cast<i64>(value);
}

bool parse_bool(const std::string& text, const std::string& what) {
    const std::string value = trim(text);
    if (value == "true" || value == "1") return true;
    if (value == "false" || value == "0") return false;
    fail("invalid boolean for " + what + ": " + text);
}

double parse_double(const std::string& text, const std::string& what) {
    const std::string value_text = trim(text);
    std::size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(value_text, &used);
    } catch (...) {
        fail("invalid floating-point value for " + what + ": " + text);
    }
    if (used != value_text.size() || !std::isfinite(value) || value < 0.0) {
        fail("invalid non-negative floating-point value for " + what + ": " + text);
    }
    return value;
}

std::vector<u64> parse_list(const std::string& text, const std::string& what) {
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

std::vector<std::pair<u64, u64>> parse_profile(const std::string& text) {
    std::string value = trim(text);
    if (value.size() < 2 || value.front() != '[' || value.back() != ']') {
        fail("invalid hidden_profile: " + text);
    }
    value = value.substr(1, value.size() - 2);
    std::vector<std::pair<u64, u64>> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim(item);
        const auto x = item.find('x');
        if (x == std::string::npos) fail("invalid hidden_profile item: " + item);
        result.emplace_back(parse_u64(item.substr(0, x), "hidden head count"),
                            parse_u64(item.substr(x + 1), "hidden head dimension"));
    }
    if (result.empty()) fail("empty hidden_profile");
    return result;
}

std::filesystem::path local_path(const std::filesystem::path& baseline,
                                 const std::string& raw, const std::string& field) {
    const std::filesystem::path candidate(raw);
    if (candidate.is_absolute()) fail(field + " must be baseline-local: " + raw);
    for (const auto& component : candidate) {
        if (component == "..") fail(field + " escapes baseline root: " + raw);
    }
    const auto normalized = (baseline / candidate).lexically_normal();
    const auto base = baseline.lexically_normal();
    const std::string base_text = base.generic_string();
    const std::string path_text = normalized.generic_string();
    if (path_text != base_text && path_text.rfind(base_text + "/", 0) != 0) {
        fail(field + " escapes baseline root: " + raw);
    }
    return normalized;
}

std::string file_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) fail("cannot open " + path.string());
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

// Small self-contained SHA-256 implementation.  The adapter does not depend
// on the host OpenSSL installation, so a copied baseline remains reproducible
// under the official nvcc build.
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

    void update(const void* data, std::size_t length) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
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
        for (int i = 0; i < 8; ++i) {
            length[7 - i] = static_cast<std::uint8_t>(original_bits >> (i * 8));
        }
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
    static constexpr std::array<std::uint32_t, 64> constants_ = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

    static std::uint32_t rotr(std::uint32_t value, unsigned amount) {
        return (value >> amount) | (value << (32U - amount));
    }
    static std::uint32_t choose(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return (x & y) ^ (~x & z);
    }
    static std::uint32_t majority(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return (x & y) ^ (x & z) ^ (y & z);
    }
    static std::uint32_t big_sigma0(std::uint32_t x) {
        return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
    }
    static std::uint32_t big_sigma1(std::uint32_t x) {
        return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
    }
    static std::uint32_t small_sigma0(std::uint32_t x) {
        return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
    }
    static std::uint32_t small_sigma1(std::uint32_t x) {
        return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
    }

    void transform(const std::uint8_t* block) {
        std::array<std::uint32_t, 64> words{};
        for (int i = 0; i < 16; ++i) {
            words[i] = (static_cast<std::uint32_t>(block[4 * i]) << 24) |
                       (static_cast<std::uint32_t>(block[4 * i + 1]) << 16) |
                       (static_cast<std::uint32_t>(block[4 * i + 2]) << 8) |
                       static_cast<std::uint32_t>(block[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            words[i] = small_sigma1(words[i - 2]) + words[i - 7] +
                       small_sigma0(words[i - 15]) + words[i - 16];
        }
        std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t t1 = h + big_sigma1(e) + choose(e, f, g) + constants_[i] + words[i];
            const std::uint32_t t2 = big_sigma0(a) + majority(a, b, c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> buffer_{};
    std::uint64_t bit_count_;
    std::size_t buffered_;
};

std::string hex(const std::array<std::uint8_t, 32>& digest) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const auto byte : digest) out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

std::string sha256_bytes(const void* data, std::size_t size) {
    Sha256 hash;
    hash.update(data, size);
    return hex(hash.finish());
}

std::string sha256_text(const std::string& text) {
    return sha256_bytes(text.data(), text.size());
}

std::string sha256_file(const std::filesystem::path& path) {
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

void hash_u64(Sha256& hash, u64 value) {
    std::array<std::uint8_t, 8> bytes{};
    for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<std::uint8_t>(value >> (8U * i));
    hash.update(bytes.data(), bytes.size());
}

void hash_i64(Sha256& hash, i64 value) { hash_u64(hash, static_cast<u64>(value)); }

std::string digest_matrix(const Matrix& matrix) {
    Sha256 hash;
    hash_u64(hash, matrix.rows); hash_u64(hash, matrix.cols);
    for (const auto value : matrix.values) hash_i64(hash, value);
    return hex(hash.finish());
}

std::string digest_graph(const Graph& graph) {
    Sha256 hash;
    hash_u64(hash, graph.total_nodes); hash_u64(hash, graph.nodes); hash_u64(hash, graph.edges);
    hash_u64(hash, graph.graphs); hash_u64(hash, graph.features);
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
        hash_u64(hash, graph.src[i]); hash_u64(hash, graph.dst[i]);
    }
    return hex(hash.finish());
}

std::string digest_graph_boundaries(const Graph& graph) {
    Sha256 hash;
    const std::string tag = "zkDL_gat_graph_boundaries_v1";
    hash_u64(hash, tag.size());
    hash.update(tag.data(), tag.size());
    const auto append = [&](const std::vector<u64>& values) {
        hash_u64(hash, values.size());
        for (const auto value : values) hash_u64(hash, value);
    };
    append(graph.node_ptr);
    append(graph.edge_ptr);
    append(graph.node_edge_ptr);
    return hex(hash.finish());
}

std::string digest_tables(const StaticTables& tables) {
    Sha256 hash;
    const auto hash_string = [&](const std::string& value) {
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
    hash_i64(hash, tables.lrelu_min); hash_i64(hash, tables.lrelu_max);
    hash_i64(hash, tables.elu_min); hash_i64(hash, tables.elu_max);
    hash_u64(hash, tables.exp_max);
    hash_u64(hash, tables.lrelu.size());
    for (const auto value : tables.lrelu) hash_i64(hash, value);
    hash_u64(hash, tables.elu.size());
    for (const auto value : tables.elu) hash_i64(hash, value);
    hash_u64(hash, tables.exp.size());
    for (const auto value : tables.exp) hash_i64(hash, value);
    return hex(hash.finish());
}

std::string digest_ledger(const IntegerBoundLedger& ledger) {
    Sha256 hash;
    const auto hash_string = [&](const std::string& value) {
        hash_u64(hash, static_cast<u64>(value.size()));
        hash.update(value.data(), value.size());
    };
    const auto hash_family = [&](const IntegerBoundFamily& family) {
        hash_u64(hash, family.layer);
        hash_u64(hash, family.head);
        hash_u64(hash, family.output ? 1U : 0U);
        for (const auto value : {family.max_input_abs, family.max_weight_abs,
                                 family.max_attention_abs, family.max_bias_abs,
                                 family.max_score_abs, family.max_table_output_abs,
                                 family.max_delta, family.max_exp, family.max_sum,
                                 family.max_aggregate_num, family.max_dividend_abs,
                                 family.max_aux_abs, family.max_divisor,
                                 family.max_field_product_abs}) {
            hash_i64(hash, value);
        }
    };
    hash_string("zkDL_gat_integer_bound_ledger_v1");
    hash_string(ledger.ledger_id);
    hash_string(ledger.quant_cfg_id);
    hash_string(ledger.static_table_id);
    hash_string(ledger.field_modulus);
    hash_string(ledger.scale_chain_digest);
    hash_string(ledger.static_table_descriptor_digest);
    for (const auto value : {ledger.max_score_abs, ledger.max_table_output_abs,
                             ledger.max_delta, ledger.max_exp, ledger.max_sum,
                             ledger.max_aggregate_num, ledger.max_dividend_abs,
                             ledger.max_aux_abs, ledger.max_divisor}) {
        hash_i64(hash, value);
    }
    hash_u64(hash, ledger.families.size());
    for (const auto& family : ledger.families) hash_family(family);
    return hex(hash.finish());
}

std::string json_fragment(const std::string& text, const std::string& key, char open, char close) {
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
            if (escaped) escaped = false;
            else if (current == '\\') escaped = true;
            else if (current == '\"') quoted = false;
            continue;
        }
        if (current == '\"') { quoted = true; continue; }
        if (current == open) ++depth;
        if (current == close && --depth == 0) return text.substr(open_pos, index - open_pos + 1);
    }
    fail("unterminated JSON field: " + key);
}

std::string json_scalar(const std::string& text, const std::string& key) {
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

std::vector<i64> json_i64_array(const std::string& text, const std::string& key,
                                u64 maximum_values = kMaxStaticTableRows) {
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

std::vector<std::string> json_objects(const std::string& text, const std::string& key,
                                      u64 maximum_objects = kMaxLedgerFamilies) {
    const auto fragment = json_fragment(text, key, '[', ']');
    std::vector<std::string> objects;
    bool quoted = false;
    bool escaped = false;
    int depth = 0;
    std::size_t begin = 0;
    for (std::size_t index = 1; index + 1 < fragment.size(); ++index) {
        const char current = fragment[index];
        if (quoted) {
            if (escaped) escaped = false;
            else if (current == '\\') escaped = true;
            else if (current == '\"') quoted = false;
            continue;
        }
        if (current == '\"') { quoted = true; continue; }
        if (current == '{') { if (depth++ == 0) begin = index; }
        else if (current == '}' && depth > 0 && --depth == 0) {
            if (objects.size() >= maximum_objects) fail("BLOCKED: JSON object array exceeds bounded size: " + key);
            objects.push_back(fragment.substr(begin, index - begin + 1));
        }
    }
    if (objects.empty()) fail("JSON object array is empty: " + key);
    return objects;
}

i64 json_i64_value(const std::string& text, const std::string& key) {
    return parse_i64(json_scalar(text, key), "JSON " + key);
}

u64 json_u64_value(const std::string& text, const std::string& key) {
    return parse_u64(json_scalar(text, key), "JSON " + key);
}

bool json_bool_value(const std::string& text, const std::string& key) {
    return parse_bool(json_scalar(text, key), "JSON " + key);
}

i64 checked_i128(__int128 value, const std::string& what) {
    if (value < static_cast<__int128>(std::numeric_limits<i64>::min()) ||
        value > static_cast<__int128>(std::numeric_limits<i64>::max())) {
        fail("signed 64-bit overflow in " + what);
    }
    return static_cast<i64>(value);
}

u64 checked_domain_rows(i64 minimum, i64 maximum, const std::string& name) {
    if (minimum > maximum) fail("invalid domain bounds in " + name);
    const __int128 rows = static_cast<__int128>(maximum) -
                          static_cast<__int128>(minimum) + 1;
    if (rows <= 0 || rows > static_cast<__int128>(kMaxStaticTableRows) ||
        rows > static_cast<__int128>(std::numeric_limits<std::size_t>::max())) {
        fail("BLOCKED: table domain is too large in " + name);
    }
    return static_cast<u64>(rows);
}

i64 round_div(i64 numerator, i64 denominator, const std::string& what) {
    if (denominator <= 0) fail("non-positive divisor in " + what);
    const __int128 n = numerator;
    const __int128 d = denominator;
    if (n >= 0) return checked_i128((n + d / 2) / d, what);
    return checked_i128(-((-n + d / 2) / d), what);
}

i64 rounded_float(double value, u64 scale, const std::string& what) {
    if (!std::isfinite(value)) fail("non-finite model value in " + what);
    const long double scaled = static_cast<long double>(value) * static_cast<long double>(scale);
    const long double rounded = scaled >= 0 ? std::floor(scaled + 0.5L) : std::ceil(scaled - 0.5L);
    if (rounded < static_cast<long double>(std::numeric_limits<i64>::min()) ||
        rounded > static_cast<long double>(std::numeric_limits<i64>::max())) {
        fail("quantization overflow in " + what);
    }
    return static_cast<i64>(rounded);
}

Matrix make_matrix(u64 rows, u64 cols) {
    if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows) fail("matrix size overflow");
    Matrix matrix;
    matrix.rows = rows; matrix.cols = cols;
    matrix.values.assign(static_cast<std::size_t>(rows * cols), 0);
    return matrix;
}

Matrix concat_columns(const std::vector<Matrix>& matrices) {
    if (matrices.empty()) fail("cannot concatenate empty matrix family");
    const u64 rows = matrices.front().rows;
    u64 cols = 0;
    for (const auto& matrix : matrices) {
        if (matrix.rows != rows) fail("concat row mismatch");
        cols += matrix.cols;
    }
    Matrix output = make_matrix(rows, cols);
    u64 offset = 0;
    for (const auto& matrix : matrices) {
        for (u64 row = 0; row < rows; ++row) {
            std::copy(matrix.values.begin() + static_cast<std::ptrdiff_t>(row * matrix.cols),
                      matrix.values.begin() + static_cast<std::ptrdiff_t>((row + 1) * matrix.cols),
                      output.values.begin() + static_cast<std::ptrdiff_t>(row * cols + offset));
        }
        offset += matrix.cols;
    }
    return output;
}

const ModelTensor& model_tensor(const ModelBundle& model, const std::string& name) {
    const auto iterator = model.tensors.find(name);
    if (iterator == model.tensors.end()) fail("checkpoint tensor missing: " + name);
    return iterator->second;
}

Matrix quantized_tensor_matrix(const ModelBundle& model, const std::string& name,
                               u64 scale) {
    const auto& tensor = model_tensor(model, name);
    if (tensor.shape.size() < 2) fail("tensor is not matrix-shaped: " + name);
    const u64 rows = tensor.shape[tensor.shape.size() - 2];
    const u64 cols = tensor.shape.back();
    if (tensor.values.size() != rows * cols) fail("tensor count mismatch: " + name);
    Matrix output = make_matrix(rows, cols);
    for (u64 row = 0; row < rows; ++row) {
        for (u64 col = 0; col < cols; ++col) {
            output.at(row, col) = rounded_float(tensor.values[static_cast<std::size_t>(row * cols + col)],
                                                scale, name);
        }
    }
    return output;
}

std::vector<i64> quantized_tensor_vector(const ModelBundle& model, const std::string& name,
                                         u64 scale) {
    const auto& tensor = model_tensor(model, name);
    std::vector<i64> output;
    output.reserve(tensor.values.size());
    for (const auto value : tensor.values) output.push_back(rounded_float(value, scale, name));
    return output;
}

std::int64_t table_lookup(const std::vector<i64>& values, i64 minimum, i64 maximum,
                          i64 index, const std::string& name) {
    if (index < minimum || index > maximum) {
        fail(name + " lookup outside committed table domain");
    }
    const __int128 offset = static_cast<__int128>(index) -
                            static_cast<__int128>(minimum);
    if (offset < 0 || offset >= static_cast<__int128>(values.size())) {
        fail(name + " lookup offset exceeds committed table");
    }
    return values.at(static_cast<std::size_t>(offset));
}

Matrix matmul_cpu(const Matrix& left, const Matrix& right, const std::string& name) {
    if (left.cols != right.rows) fail("matmul dimension mismatch in " + name);
    Matrix output = make_matrix(left.rows, right.cols);
    for (u64 row = 0; row < left.rows; ++row) {
        for (u64 col = 0; col < right.cols; ++col) {
            __int128 sum = 0;
            for (u64 inner = 0; inner < left.cols; ++inner) {
                sum += static_cast<__int128>(left.at(row, inner)) * right.at(inner, col);
            }
            output.at(row, col) = checked_i128(sum, name);
        }
    }
    return output;
}

__global__ void matmul_kernel(const long long* left, const long long* right, long long* output,
                              unsigned long long rows, unsigned long long inner,
                              unsigned long long cols) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * cols) return;
    const unsigned long long row = index / cols;
    const unsigned long long col = index % cols;
    long long sum = 0;
    for (unsigned long long k = 0; k < inner; ++k) sum += left[row * inner + k] * right[k * cols + col];
    output[index] = sum;
}

void cuda_check(cudaError_t error, const std::string& what) {
    if (error != cudaSuccess) fail(what + ": " + cudaGetErrorString(error));
}

Matrix matmul_gpu(const Matrix& left, const Matrix& right, const std::string& name) {
    if (left.cols != right.rows) fail("matmul dimension mismatch in " + name);
    const std::size_t left_bytes = left.values.size() * sizeof(i64);
    const std::size_t right_bytes = right.values.size() * sizeof(i64);
    const std::size_t output_bytes = static_cast<std::size_t>(left.rows * right.cols) * sizeof(i64);
    long long* left_device = nullptr;
    long long* right_device = nullptr;
    long long* output_device = nullptr;
    try {
        cuda_check(cudaMalloc(&left_device, left_bytes), name + " cudaMalloc(left)");
        cuda_check(cudaMalloc(&right_device, right_bytes), name + " cudaMalloc(right)");
        cuda_check(cudaMalloc(&output_device, output_bytes), name + " cudaMalloc(output)");
        cuda_check(cudaMemcpy(left_device, left.values.data(), left_bytes, cudaMemcpyHostToDevice), name + " H2D(left)");
        cuda_check(cudaMemcpy(right_device, right.values.data(), right_bytes, cudaMemcpyHostToDevice), name + " H2D(right)");
        const auto count = left.rows * right.cols;
        const unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
        matmul_kernel<<<blocks, threads>>>(left_device, right_device, output_device,
                                            left.rows, left.cols, right.cols);
        cuda_check(cudaGetLastError(), name + " kernel launch");
        cuda_check(cudaDeviceSynchronize(), name + " kernel synchronize");
        Matrix output = make_matrix(left.rows, right.cols);
        cuda_check(cudaMemcpy(output.values.data(), output_device, output_bytes, cudaMemcpyDeviceToHost),
                   name + " D2H(output)");
        cudaFree(left_device); cudaFree(right_device); cudaFree(output_device);
        return output;
    } catch (...) {
        if (left_device) cudaFree(left_device);
        if (right_device) cudaFree(right_device);
        if (output_device) cudaFree(output_device);
        throw;
    }
}

class BigNum {
public:
    BigNum() : value_(BN_new()) {
        if (!value_) fail("BN_new failed");
        BN_zero(value_);
    }
    BigNum(const BigNum&) = delete;
    BigNum& operator=(const BigNum&) = delete;
    BigNum(BigNum&& other) noexcept : value_(other.value_) { other.value_ = nullptr; }
    BigNum& operator=(BigNum&& other) noexcept {
        if (this != &other) { if (value_) BN_free(value_); value_ = other.value_; other.value_ = nullptr; }
        return *this;
    }
    ~BigNum() { if (value_) BN_free(value_); }
    BIGNUM* get() { return value_; }
    const BIGNUM* get() const { return value_; }
    BIGNUM** address() { return &value_; }
private:
    BIGNUM* value_;
};

const BIGNUM* scalar_modulus() {
    static BIGNUM* modulus = [] {
        BIGNUM* result = nullptr;
        if (BN_hex2bn(&result, "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001") == 0) {
            fail("BN_hex2bn failed for scalar modulus");
        }
        return result;
    }();
    return modulus;
}

std::string scalar_modulus_decimal() {
    char* decimal = BN_bn2dec(scalar_modulus());
    if (!decimal) fail("BN_bn2dec failed for scalar modulus");
    const std::string result(decimal);
    OPENSSL_free(decimal);
    return result;
}

BigNum field_from_i64(i64 value, BN_CTX* context) {
    BigNum result;
    const std::string decimal = std::to_string(value);
    if (BN_dec2bn(result.address(), decimal.c_str()) == 0) fail("BN_dec2bn failed");
    if (!BN_nnmod(result.get(), result.get(), scalar_modulus(), context)) fail("BN_nnmod failed");
    return result;
}

FieldValue field_encode(const BIGNUM* input) {
    if (BN_num_bytes(input) > 32) fail("field encoding exceeds 256 bits");
    std::array<std::uint8_t, 32> big_endian{};
    const int size = BN_bn2binpad(input, big_endian.data(), static_cast<int>(big_endian.size()));
    if (size != static_cast<int>(big_endian.size())) fail("BN_bn2binpad failed");
    FieldValue output;
    for (std::size_t i = 0; i < output.bytes.size(); ++i) output.bytes[i] = big_endian[output.bytes.size() - i - 1];
    return output;
}

BigNum field_inverse(i64 value, BN_CTX* context) {
    BigNum base = field_from_i64(value, context);
    if (BN_is_zero(base.get())) fail("zero denominator in attention relation");
    BigNum result;
    if (!BN_mod_inverse(result.get(), base.get(), scalar_modulus(), context)) fail("field inverse failed");
    return result;
}

void fill_field_attention(HiddenHeadTrace& head, const Graph& graph, u64 edge_begin,
                          u64 edge_end, const std::vector<i64>& denominator_by_node) {
    BN_CTX* context = BN_CTX_new();
    if (!context) fail("BN_CTX_new failed");
    head.inverse_field.resize(graph.edges);
    head.alpha_field.resize(graph.edges);
    for (u64 edge = edge_begin; edge < edge_end; ++edge) {
        const u64 node = graph.dst[edge];
        BigNum inverse = field_inverse(denominator_by_node[node], context);
        BigNum exponent = field_from_i64(head.exp_weight[edge], context);
        BigNum alpha;
        if (!BN_mod_mul(alpha.get(), exponent.get(), inverse.get(), scalar_modulus(), context)) fail("field alpha multiplication failed");
        head.inverse_field[edge] = field_encode(inverse.get());
        head.alpha_field[edge] = field_encode(alpha.get());
    }
    BN_CTX_free(context);
}

void fill_field_attention(OutputHeadTrace& head, const Graph& graph, u64 edge_begin,
                          u64 edge_end, const std::vector<i64>& denominator_by_node) {
    BN_CTX* context = BN_CTX_new();
    if (!context) fail("BN_CTX_new failed");
    head.inverse_field.resize(graph.edges);
    head.alpha_field.resize(graph.edges);
    for (u64 edge = edge_begin; edge < edge_end; ++edge) {
        const u64 node = graph.dst[edge];
        BigNum inverse = field_inverse(denominator_by_node[node], context);
        BigNum exponent = field_from_i64(head.exp_weight[edge], context);
        BigNum alpha;
        if (!BN_mod_mul(alpha.get(), exponent.get(), inverse.get(), scalar_modulus(), context)) fail("field alpha multiplication failed");
        head.inverse_field[edge] = field_encode(inverse.get());
        head.alpha_field[edge] = field_encode(alpha.get());
    }
    BN_CTX_free(context);
}

SignedRoundDivWitness make_round_div_witness(const Matrix& numerator,
                                             const std::vector<i64>& denominator,
                                             u64 numerator_scale,
                                             u64 denominator_scale,
                                             const Matrix& expected,
                                             const std::string& name) {
    if (numerator.rows != expected.rows || numerator.cols != expected.cols ||
        denominator.size() != numerator.rows || numerator_scale == 0 || denominator_scale == 0) {
        fail("SignedRoundDiv shape/scale mismatch in " + name);
    }
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
    BN_CTX* context = BN_CTX_new();
    if (!context) fail("BN_CTX_new failed for " + name);
    try {
        for (u64 row = 0; row < numerator.rows; ++row) {
            const i64 d = checked_i128(static_cast<__int128>(denominator[row]) * denominator_scale,
                                       name + ".divisor");
            if (d <= 0) fail("SignedRoundDiv divisor is not positive in " + name);
            BigNum divisor_inverse = field_inverse(d, context);
            for (u64 col = 0; col < numerator.cols; ++col) {
                const std::size_t index = static_cast<std::size_t>(row * numerator.cols + col);
                const i64 x = checked_i128(static_cast<__int128>(numerator.at(row, col)) * numerator_scale,
                                           name + ".dividend");
                const i64 y = expected.at(row, col);
                const __int128 abs_x = x < 0 ? -static_cast<__int128>(x) : static_cast<__int128>(x);
                const __int128 abs_y = y < 0 ? -static_cast<__int128>(y) : static_cast<__int128>(y);
                const __int128 remainder = 2 * abs_x + d - 2 * static_cast<__int128>(d) * abs_y;
                const __int128 slack = 2 * static_cast<__int128>(d) - 1 - remainder;
                if (remainder < 0 || slack < 0) fail("invalid nearest-ties-away witness in " + name);
                witness.sigma[index] = static_cast<std::uint8_t>(x < 0 ? 1U : 0U);
                witness.z[index] = static_cast<std::uint8_t>(x == 0 ? 1U : 0U);
                witness.a[index] = checked_i128(abs_x, name + ".a");
                witness.q[index] = checked_i128(abs_y, name + ".q");
                witness.r[index] = checked_i128(remainder, name + ".r");
                witness.delta[index] = checked_i128(slack, name + ".delta");
                witness.divisor[index] = d;
                witness.divisor_inverse[index] = field_encode(divisor_inverse.get());
                if (witness.a[index] != 0) {
                    BigNum inverse = field_inverse(witness.a[index], context);
                    witness.a_inverse[index] = field_encode(inverse.get());
                }
            }
        }
    } catch (...) {
        BN_CTX_free(context);
        throw;
    }
    BN_CTX_free(context);
    return witness;
}

std::vector<i64> attention_scores(const Matrix& projected, const Matrix& src_kernel,
                                  const Matrix& dst_kernel, const Graph& graph,
                                  std::vector<i64>* e_src, std::vector<i64>* e_dst,
                                  std::vector<i64>* score, const std::string& name) {
    if (src_kernel.rows != projected.cols || dst_kernel.rows != projected.cols ||
        src_kernel.cols != 1 || dst_kernel.cols != 1) fail("attention kernel shape mismatch in " + name);
    e_src->assign(graph.edges, 0); e_dst->assign(graph.edges, 0); score->assign(graph.edges, 0);
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

void aggregate_head(const Config& config, const Graph& graph, const StaticTables& tables,
                    const Matrix& projected, const std::vector<i64>& score,
                    u64 h_prime_scale, u64 aggregate_pre_scale,
                    HiddenHeadTrace* output, const std::string& name) {
    output->h_prime = projected;
    output->leaky_relu.resize(graph.edges);
    output->maximum.assign(graph.nodes, std::numeric_limits<i64>::min());
    output->delta.resize(graph.edges); output->exp_weight.resize(graph.edges);
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
        output->leaky_relu[edge] = table_lookup(tables.lrelu, tables.lrelu_min, tables.lrelu_max,
                                                score[edge], name + ".lrelu");
        output->maximum[graph.dst[edge]] = std::max(output->maximum[graph.dst[edge]], output->leaky_relu[edge]);
    }
    std::vector<i64> denominator_by_node(graph.nodes, 0);
    for (u64 edge = 0; edge < graph.edges; ++edge) {
        const u64 node = graph.dst[edge];
        output->delta[edge] = checked_i128(static_cast<__int128>(output->maximum[node]) - output->leaky_relu[edge],
                                           name + ".delta");
        output->exp_weight[edge] = table_lookup(tables.exp, 0, static_cast<i64>(tables.exp_max),
                                                output->delta[edge], name + ".exp");
        denominator_by_node[node] = checked_i128(static_cast<__int128>(denominator_by_node[node]) + output->exp_weight[edge],
                                                  name + ".denominator");
    }
    for (u64 edge = 0; edge < graph.edges; ++edge) output->denominator[edge] = denominator_by_node[graph.dst[edge]];
    if (output->denominator.empty()) fail(name + " has no edges");
    fill_field_attention(*output, graph, 0, graph.edges, denominator_by_node);
    // Accumulate the complete numerator before applying SignedRoundDiv.  The
    // previous implementation rounded every prefix of a destination group,
    // which is not the relation in the protocol and is observably wrong for
    // groups containing more than one edge.
    for (u64 edge = 0; edge < graph.edges; ++edge) {
        const u64 node = graph.dst[edge];
        for (u64 dim = 0; dim < projected.cols; ++dim) {
            const i64 value = projected.at(graph.src[edge], dim);
            output->routed_h_prime.at(edge, dim) = value;
            output->product_u_h_prime.at(edge, dim) = checked_i128(
                static_cast<__int128>(value) * output->exp_weight[edge], name + ".u_h_prime");
            output->aggregate_num.at(node, dim) = checked_i128(
                static_cast<__int128>(output->aggregate_num.at(node, dim)) +
                    output->product_u_h_prime.at(edge, dim), name + ".aggregate_num");
            output->aggregate_pre.at(node, dim) = 0;
            output->elu_input.at(node, dim) = 0;
        }
    }

    // Aggregate pre-field is the actual native field relation, while the
    // integer view above is the fixed-point projection used by ELU.  The
    // field view must use alpha=U*inv_dst (not the integer U numerator):
    // this is the relation that makes aggregate_num == Sum*aggregate_pre in
    // the scalar field.  Keep the accumulation on BIGNUMs so no host-side
    // second field arithmetic implementation is introduced.
    BN_CTX* field_context = BN_CTX_new();
    if (!field_context) fail("BN_CTX_new failed for aggregate field");
    for (u64 node = 0; node < graph.nodes; ++node) {
        for (u64 dim = 0; dim < projected.cols; ++dim) {
            BigNum value;
            BN_zero(value.get());
            BigNum integer_value;
            BN_zero(integer_value.get());
            if (node + 1 >= graph.node_edge_ptr.size()) {
                BN_CTX_free(field_context);
                fail("edge pointer does not cover destination node");
            }
            BigNum inverse = field_inverse(denominator_by_node[node], field_context);
            for (u64 edge = graph.node_edge_ptr[node]; edge < graph.node_edge_ptr[node + 1]; ++edge) {
                if (graph.dst[edge] != node) {
                    BN_CTX_free(field_context);
                    fail("edge pointer/destination mismatch");
                }
                BigNum weight = field_from_i64(output->exp_weight[edge], field_context);
                BigNum source = field_from_i64(projected.at(graph.src[edge], dim), field_context);
                BigNum alpha;
                BigNum product;
                BigNum integer_product;
                if (!BN_mod_mul(alpha.get(), weight.get(), inverse.get(), scalar_modulus(), field_context) ||
                    !BN_mod_mul(product.get(), alpha.get(), source.get(), scalar_modulus(), field_context) ||
                    !BN_mod_mul(integer_product.get(), weight.get(), source.get(), scalar_modulus(), field_context) ||
                    !BN_mod_add(value.get(), value.get(), product.get(), scalar_modulus(), field_context) ||
                    !BN_mod_add(integer_value.get(), integer_value.get(), integer_product.get(), scalar_modulus(), field_context)) {
                    BN_CTX_free(field_context);
                    fail("aggregate field arithmetic failed");
                }
                output->product_alpha_h_prime_field.values[static_cast<std::size_t>(edge * projected.cols + dim)] =
                    field_encode(product.get());
            }
            output->aggregate_pre_field.values[static_cast<std::size_t>(node * projected.cols + dim)] =
                field_encode(value.get());
            BigNum rhs;
            BigNum residual;
            if (!BN_mod_mul(rhs.get(), field_from_i64(denominator_by_node[node], field_context).get(), value.get(),
                            scalar_modulus(), field_context) ||
                !BN_mod_sub(residual.get(), integer_value.get(), rhs.get(), scalar_modulus(), field_context)) {
                BN_CTX_free(field_context);
                fail("aggregate consistency field arithmetic failed");
            }
            output->consistency_residual_field.values[static_cast<std::size_t>(node * projected.cols + dim)] =
                field_encode(residual.get());
            const i64 denom = denominator_by_node[node];
            output->aggregate_pre.at(node, dim) = round_div(
                checked_i128(static_cast<__int128>(output->aggregate_num.at(node, dim)) * aggregate_pre_scale,
                             name + ".aggregate_scaled"),
                checked_i128(static_cast<__int128>(denom) * h_prime_scale,
                             name + ".aggregate_denominator"), name + ".aggregate_pre");
            output->elu_input.at(node, dim) = output->aggregate_pre.at(node, dim);
            output->aggregate.at(node, dim) = table_lookup(
                tables.elu, tables.elu_min, tables.elu_max, output->elu_input.at(node, dim), name + ".elu");
        }
    }
    BN_CTX_free(field_context);
    output->aggregate_round_div = make_round_div_witness(
        output->aggregate_num, denominator_by_node, aggregate_pre_scale, h_prime_scale,
        output->aggregate_pre, name + ".signed_round_div");
    (void)config;
}

void aggregate_output_head(const Config& config, const Graph& graph, const StaticTables& tables,
                           const Matrix& projected, const std::vector<i64>& score,
                           u64 h_prime_scale, u64 output_pre_scale,
                           OutputHeadTrace* output, const std::string& name) {
    output->y_prime = projected;
    output->leaky_relu.resize(graph.edges);
    output->maximum.assign(graph.nodes, std::numeric_limits<i64>::min());
    output->delta.resize(graph.edges); output->exp_weight.resize(graph.edges);
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
        output->leaky_relu[edge] = table_lookup(tables.lrelu, tables.lrelu_min, tables.lrelu_max,
                                                score[edge], name + ".lrelu");
        output->maximum[graph.dst[edge]] = std::max(output->maximum[graph.dst[edge]], output->leaky_relu[edge]);
    }
    std::vector<i64> denominator_by_node(graph.nodes, 0);
    for (u64 edge = 0; edge < graph.edges; ++edge) {
        output->delta[edge] = checked_i128(static_cast<__int128>(output->maximum[graph.dst[edge]]) - output->leaky_relu[edge],
                                           name + ".delta");
        output->exp_weight[edge] = table_lookup(tables.exp, 0, static_cast<i64>(tables.exp_max),
                                                output->delta[edge], name + ".exp");
        denominator_by_node[graph.dst[edge]] = checked_i128(static_cast<__int128>(denominator_by_node[graph.dst[edge]]) +
                                                            output->exp_weight[edge], name + ".denominator");
    }
    for (u64 edge = 0; edge < graph.edges; ++edge) output->denominator[edge] = denominator_by_node[graph.dst[edge]];
    fill_field_attention(*output, graph, 0, graph.edges, denominator_by_node);
    for (u64 edge = 0; edge < graph.edges; ++edge) {
        const u64 node = graph.dst[edge];
        for (u64 dim = 0; dim < projected.cols; ++dim) {
            const i64 value = projected.at(graph.src[edge], dim);
            output->routed_y_prime.at(edge, dim) = value;
            output->product_u_y_prime.at(edge, dim) = checked_i128(static_cast<__int128>(value) * output->exp_weight[edge],
                                                                   name + ".u_y_prime");
            output->aggregate_num.at(node, dim) = checked_i128(static_cast<__int128>(output->aggregate_num.at(node, dim)) +
                                                               output->product_u_y_prime.at(edge, dim), name + ".aggregate_num");
        }
    }
    // Preserve the same canonical alpha aggregate used by the hidden layers.
    // The integer y_quantized/aligned views below remain the fixed-point
    // SignedRoundDiv path and are intentionally not substituted by this
    // field value.
    BN_CTX* field_context = BN_CTX_new();
    if (!field_context) fail("BN_CTX_new failed for output aggregate field");
    for (u64 node = 0; node < graph.nodes; ++node) {
        if (node + 1 >= graph.node_edge_ptr.size()) {
            BN_CTX_free(field_context);
            fail("edge pointer does not cover output destination node");
        }
        BigNum inverse = field_inverse(denominator_by_node[node], field_context);
        for (u64 dim = 0; dim < projected.cols; ++dim) {
            BigNum value;
            BN_zero(value.get());
            BigNum integer_value;
            BN_zero(integer_value.get());
            for (u64 edge = graph.node_edge_ptr[node]; edge < graph.node_edge_ptr[node + 1]; ++edge) {
                if (graph.dst[edge] != node) {
                    BN_CTX_free(field_context);
                    fail("output edge pointer/destination mismatch");
                }
                BigNum weight = field_from_i64(output->exp_weight[edge], field_context);
                BigNum source = field_from_i64(projected.at(graph.src[edge], dim), field_context);
                BigNum alpha;
                BigNum product;
                BigNum integer_product;
                if (!BN_mod_mul(alpha.get(), weight.get(), inverse.get(), scalar_modulus(), field_context) ||
                    !BN_mod_mul(product.get(), alpha.get(), source.get(), scalar_modulus(), field_context) ||
                    !BN_mod_mul(integer_product.get(), weight.get(), source.get(), scalar_modulus(), field_context) ||
                    !BN_mod_add(value.get(), value.get(), product.get(), scalar_modulus(), field_context) ||
                    !BN_mod_add(integer_value.get(), integer_value.get(), integer_product.get(), scalar_modulus(), field_context)) {
                    BN_CTX_free(field_context);
                    fail("output aggregate field arithmetic failed");
                }
                output->product_alpha_y_prime_field.values[static_cast<std::size_t>(edge * projected.cols + dim)] =
                    field_encode(product.get());
            }
            output->aggregate_pre_field.values[static_cast<std::size_t>(node * projected.cols + dim)] =
                field_encode(value.get());
            BigNum rhs;
            BigNum residual;
            BigNum denominator = field_from_i64(denominator_by_node[node], field_context);
            if (!BN_mod_mul(rhs.get(), denominator.get(), value.get(), scalar_modulus(), field_context) ||
                !BN_mod_sub(residual.get(), integer_value.get(), rhs.get(), scalar_modulus(), field_context)) {
                BN_CTX_free(field_context);
                fail("output aggregate consistency field arithmetic failed");
            }
            output->consistency_residual_field.values[static_cast<std::size_t>(node * projected.cols + dim)] =
                field_encode(residual.get());
            output->aggregate_pre.at(node, dim) = round_div(
                checked_i128(static_cast<__int128>(output->aggregate_num.at(node, dim)) * output_pre_scale,
                             name + ".aggregate_scaled"),
                checked_i128(static_cast<__int128>(denominator_by_node[node]) * h_prime_scale,
                             name + ".aggregate_denominator"), name + ".aggregate_pre");
            output->y_quantized.at(node, dim) = output->aggregate_pre.at(node, dim);
            output->aligned.at(node, dim) = output->y_quantized.at(node, dim);
        }
    }
    BN_CTX_free(field_context);
    std::vector<i64> denominators(graph.nodes, 0);
    for (u64 node = 0; node < graph.nodes; ++node) denominators[node] = denominator_by_node[node];
    output->aggregate_round_div = make_round_div_witness(
        output->aggregate_num, denominators, output_pre_scale, h_prime_scale,
        output->aggregate_pre, name + ".signed_round_div");
    (void)config;
}

GatTrace run_forward(const Config& config, const Graph& graph, const ModelBundle& model,
                     const StaticTables& tables, bool gpu_projection) {
    GatTrace trace;
    trace.first_input = graph.feature_values;
    Matrix current = graph.feature_values;
    for (u64 layer = 0; layer < config.architecture.hidden_dims.size(); ++layer) {
        HiddenLayerTrace layer_trace;
        layer_trace.input = current;
        std::vector<Matrix> head_outputs;
        const u64 head_count = config.architecture.hidden_heads.at(layer);
        const u64 head_dim = config.architecture.hidden_dims.at(layer);
        const u64 in_dim = current.cols;
        for (u64 head = 0; head < head_count; ++head) {
            const std::string prefix = "hidden/layer" + std::to_string(layer) + "/head" + std::to_string(head);
            const Matrix weight = quantized_tensor_matrix(model, prefix + "/seq/kernel",
                                                          config.quantization.hidden_weight.at(layer).at(head));
            const Matrix projected = gpu_projection ? matmul_gpu(current, weight, prefix + ".projection")
                                                     : matmul_cpu(current, weight, prefix + ".projection");
            const Matrix src_kernel = quantized_tensor_matrix(model, prefix + "/attn_src/kernel",
                                                              config.quantization.hidden_attention_src.at(layer).at(head));
            const Matrix dst_kernel = quantized_tensor_matrix(model, prefix + "/attn_dst/kernel",
                                                              config.quantization.hidden_attention_dst.at(layer).at(head));
            HiddenHeadTrace head_trace;
            attention_scores(projected, src_kernel, dst_kernel, graph, &head_trace.e_src, &head_trace.e_dst,
                             &head_trace.score, prefix + ".attention");
            aggregate_head(config, graph, tables, projected, head_trace.score,
                           config.quantization.hidden_h_prime.at(layer).at(head),
                           config.quantization.hidden_aggregate_pre.at(layer),
                           &head_trace, prefix);
            head_outputs.push_back(head_trace.aggregate);
            layer_trace.heads.push_back(std::move(head_trace));
        }
        layer_trace.concat = concat_columns(head_outputs);
        current = layer_trace.concat;
        if (current.cols != head_count * head_dim || in_dim == 0) fail("hidden layer shape invariant failed");
        trace.hidden_layers.push_back(std::move(layer_trace));
    }
    for (u64 head = 0; head < config.architecture.output_heads; ++head) {
        const std::string prefix = "output/head" + std::to_string(head);
        const Matrix weight = quantized_tensor_matrix(model, prefix + "/seq/kernel",
                                                      config.quantization.output_weight.at(head));
        const Matrix projected = gpu_projection ? matmul_gpu(current, weight, prefix + ".projection")
                                                 : matmul_cpu(current, weight, prefix + ".projection");
        const Matrix src_kernel = quantized_tensor_matrix(model, prefix + "/attn_src/kernel",
                                                          config.quantization.output_attention_src.at(head));
        const Matrix dst_kernel = quantized_tensor_matrix(model, prefix + "/attn_dst/kernel",
                                                          config.quantization.output_attention_dst.at(head));
        OutputHeadTrace head_trace;
        attention_scores(projected, src_kernel, dst_kernel, graph, &head_trace.e_src, &head_trace.e_dst,
                         &head_trace.score, prefix + ".attention");
        aggregate_output_head(config, graph, tables, projected, head_trace.score,
                              config.quantization.output_y_prime.at(head),
                              config.quantization.output_pre,
                              &head_trace, prefix);
        const auto bias = quantized_tensor_vector(model, prefix + "/output_bias",
                                                  config.quantization.output_bias.at(head));
        if (bias.size() != projected.cols || bias.size() != config.architecture.classes) fail(prefix + " bias shape mismatch");
        head_trace.bias = bias;
        for (u64 node = 0; node < graph.nodes; ++node) {
            for (u64 cls = 0; cls < projected.cols; ++cls) {
                head_trace.output.at(node, cls) = checked_i128(static_cast<__int128>(head_trace.aligned.at(node, cls)) + bias[cls],
                                                               prefix + ".bias");
            }
        }
        head_trace.alignment_round_div = make_round_div_witness(
            head_trace.y_quantized,
            std::vector<i64>(graph.nodes, static_cast<i64>(config.quantization.output_pre)),
            config.quantization.output_head,
            1,
            head_trace.aligned,
            prefix + ".output_scale_alignment");
        trace.output_heads.push_back(std::move(head_trace));
    }
    if (trace.output_heads.empty()) fail("no output heads");
    trace.output_sum = make_matrix(graph.nodes, config.architecture.classes);
    for (const auto& head : trace.output_heads) {
        if (head.output.cols != trace.output_sum.cols) fail("output head class shape mismatch");
        for (std::size_t index = 0; index < trace.output_sum.values.size(); ++index) {
            trace.output_sum.values[index] = checked_i128(static_cast<__int128>(trace.output_sum.values[index]) + head.output.values[index],
                                                          "output head sum");
        }
    }
    trace.logits = make_matrix(graph.nodes, config.architecture.classes);
    for (std::size_t index = 0; index < trace.logits.values.size(); ++index) {
        trace.logits.values[index] = round_div(trace.output_sum.values[index],
                                               static_cast<i64>(trace.output_heads.size()), "final average");
    }
    trace.final_average_round_div = make_round_div_witness(
        trace.output_sum,
        std::vector<i64>(graph.nodes, static_cast<i64>(trace.output_heads.size())),
        1,
        1,
        trace.logits,
        "final_average");
    return trace;
}

std::string json_string_field(const std::string& text, const std::string& key) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) fail("manifest field missing: " + key);
    return match[1].str();
}

u64 json_u64_field(const std::string& text, const std::string& key) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) fail("manifest field missing: " + key);
    return parse_u64(match[1].str(), "manifest " + key);
}

bool is_digest(const std::string& value) {
    return std::regex_match(value, std::regex("[0-9a-fA-F]{64}"));
}

std::vector<std::uint64_t> json_shape(const std::string& text, const std::string& what) {
    return parse_list(text, what);
}

std::map<std::string, std::vector<u64>> manifest_tensor_shapes(const std::string& manifest) {
    const auto marker = manifest.find("\"tensor_index\"");
    if (marker == std::string::npos) fail("checkpoint manifest tensor_index missing");
    const std::string index = manifest.substr(marker);
    const std::regex entry(
        "\\\"([^\\\"]+)\\\"\\s*:\\s*\\{[\\s\\S]*?\\\"shape\\\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::map<std::string, std::vector<u64>> shapes;
    for (std::sregex_iterator iterator(index.begin(), index.end(), entry), end; iterator != end; ++iterator) {
        const std::string name = (*iterator)[1].str();
        const auto shape = json_shape((*iterator)[2].str(), "manifest tensor shape " + name);
        if (!shapes.emplace(name, shape).second) fail("duplicate checkpoint manifest tensor index: " + name);
    }
    if (shapes.empty()) fail("checkpoint manifest tensor_index is empty");
    return shapes;
}

void expect_manifest(const std::string& manifest, const Config& config) {
    if (json_string_field(manifest, "family_schema") != "multi_layer_multi_head") {
        fail("checkpoint family_schema is not canonical");
    }
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
    const auto expected_prefix = std::filesystem::relative(
        config.checkpoint_bundle / "best_model.pt", config.baseline_root).generic_string();
    if (checkpoint_prefix != expected_prefix) fail("checkpoint_prefix is not baseline-local canonical path");
    if (json_u64_field(manifest, "L") != config.architecture.layers) fail("checkpoint L mismatch");
    if (json_u64_field(manifest, "K_out") != config.architecture.output_heads) fail("checkpoint K_out mismatch");
    if (json_u64_field(manifest, "C") != config.architecture.classes) fail("checkpoint class count mismatch");
    if (json_u64_field(manifest, "tensor_count") == 0) fail("checkpoint tensor_count is zero");
}

struct RawEdge { u64 graph = 0; u64 src = 0; u64 dst = 0; u64 ordinal = 0; };

std::vector<float> load_npy_float32(const std::filesystem::path& path, u64 expected_rows,
                                    u64 expected_cols) {
    std::ifstream input(path, std::ios::binary);
    if (!input) fail("cannot open numpy feature file " + path.string());
    std::array<unsigned char, 10> preamble{};
    input.read(reinterpret_cast<char*>(preamble.data()), static_cast<std::streamsize>(preamble.size()));
    if (input.gcount() != static_cast<std::streamsize>(preamble.size()) || preamble[0] != 0x93 ||
        preamble[1] != 'N' || preamble[2] != 'U' || preamble[3] != 'M' || preamble[4] != 'P' || preamble[5] != 'Y') {
        fail("invalid numpy magic in " + path.string());
    }
    const unsigned version_major = preamble[6];
    const std::size_t length_size = version_major == 1 ? 2 : 4;
    std::uint32_t header_size = 0;
    for (std::size_t i = 0; i < length_size; ++i) header_size |= static_cast<std::uint32_t>(preamble[8 + i]) << (8U * i);
    std::string header(header_size, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (input.gcount() != static_cast<std::streamsize>(header.size())) fail("truncated numpy header");
    if (header.find("'descr': '<f4'") == std::string::npos && header.find("\"descr\": \"<f4\"") == std::string::npos &&
        header.find("'descr': '|f4'") == std::string::npos) fail("numpy feature dtype is not little-endian float32");
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
    const u64 rows = parse_u64(trim(row_text), "numpy rows");
    const u64 cols = parse_u64(trim(col_text), "numpy cols");
    if (rows != expected_rows || cols != expected_cols) fail("numpy feature shape mismatch");
    std::vector<float> values(static_cast<std::size_t>(rows * cols));
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (input.gcount() != static_cast<std::streamsize>(values.size() * sizeof(float))) fail("truncated numpy feature payload");
    return values;
}

std::string authenticated_asset_digest(const Config& config, const std::filesystem::path& path) {
    const auto assets = load_asset_manifest(config);
    const auto relative = std::filesystem::relative(path, config.baseline_root).generic_string();
    for (const auto& file : assets.files) {
        if (file.path == relative) return file.sha256;
    }
    fail("asset manifest does not authenticate relation resource: " + relative);
}

void require_authentication_file(const Config& config, const std::filesystem::path& path,
                                 const std::string& label) {
    if (path.empty() || std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path)) {
        fail("BLOCKED: authenticated " + label + " is missing or not a regular baseline-local file: " + path.string());
    }
    (void)authenticated_asset_digest(config, path);
}

void validate_field_bound(i64 value, const std::string& label) {
    if (value < 0) fail("negative integer bound in " + label);
    BN_CTX* context = BN_CTX_new();
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

void validate_static_tables(const Config& config, StaticTables* tables) {
    if (!tables || !tables->authenticated) fail("BLOCKED: static table descriptor is not authenticated");
    if (tables->quant_cfg_id != config.quant_cfg_id || tables->scale_id.empty() || tables->domain_id.empty() ||
        tables->row_order != "ascending_index_min_to_max" || tables->generator_id.empty() || tables->version.empty()) {
        fail("authenticated static table descriptor identity mismatch");
    }
    const auto lrelu_rows = checked_domain_rows(tables->lrelu_min, tables->lrelu_max, "LeakyReLU");
    const auto elu_rows = checked_domain_rows(tables->elu_min, tables->elu_max, "ELU");
    if (tables->exp_max >= kMaxStaticTableRows) {
        fail("BLOCKED: ExpMap table domain is too large");
    }
    const auto exp_rows = tables->exp_max + 1;
    if (tables->lrelu.size() != lrelu_rows || tables->elu.size() != elu_rows ||
        tables->exp.size() != exp_rows || tables->lrelu.empty() || tables->elu.empty() || tables->exp.empty()) {
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
        if ((input >= 0 && output != input) || (input < 0 && output >= 0)) {
            fail("authenticated ELU table violates canonical range");
        }
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
    spec << config.static_table_id << "\n" << tables->generator_id << "\n" << tables->version << "\n"
         << tables->scale_id << "\n" << tables->domain_id << "\n" << tables->row_order << "\n"
         << config.range_bits << "\n" << tables->lrelu_min << "\n" << tables->lrelu_max << "\n" << tables->exp_max;
    if (tables->spec_hash != sha256_text(spec.str())) fail("authenticated static table specification hash mismatch");
}

StaticTables load_static_table_descriptor(const Config& config) {
    require_authentication_file(config, config.static_table_descriptor_path, "static table descriptor");
    const auto text = file_text(config.static_table_descriptor_path);
    if (trim(text).empty() || trim(text).front() != '{') {
        fail("BLOCKED: static table descriptor must use the canonical JSON resource format");
    }
    StaticTables tables;
    if (json_scalar(text, "schema") != "zkDL_gat_static_table_descriptor_v1") {
        fail("static table descriptor schema mismatch");
    }
    tables.quant_cfg_id = json_scalar(text, "quant_cfg_id");
    tables.generator_id = json_scalar(text, "generator_id");
    tables.version = json_scalar(text, "version");
    tables.scale_id = json_scalar(text, "scale_id");
    tables.domain_id = json_scalar(text, "domain_id");
    tables.row_order = json_scalar(text, "row_order");
    tables.spec_hash = json_scalar(text, "spec_hash");
    tables.digest = json_scalar(text, "digest");
    if (json_scalar(text, "static_table_id") != config.static_table_id) fail("static table descriptor id mismatch");
    tables.lrelu_min = json_i64_value(text, "lrelu_min");
    tables.lrelu_max = json_i64_value(text, "lrelu_max");
    tables.elu_min = json_i64_value(text, "elu_min");
    tables.elu_max = json_i64_value(text, "elu_max");
    tables.exp_max = json_u64_value(text, "exp_max");
    tables.lrelu = json_i64_array(text, "lrelu");
    tables.elu = json_i64_array(text, "elu");
    tables.exp = json_i64_array(text, "exp");
    tables.descriptor_digest = authenticated_asset_digest(config, config.static_table_descriptor_path);
    tables.authenticated = true;
    validate_static_tables(config, &tables);
    return tables;
}

IntegerBoundFamily parse_ledger_family(const std::string& object) {
    IntegerBoundFamily family;
    family.layer = json_u64_value(object, "layer");
    family.head = json_u64_value(object, "head");
    family.output = json_bool_value(object, "output");
    family.max_input_abs = json_i64_value(object, "max_input_abs");
    family.max_weight_abs = json_i64_value(object, "max_weight_abs");
    family.max_attention_abs = json_i64_value(object, "max_attention_abs");
    family.max_bias_abs = json_i64_value(object, "max_bias_abs");
    family.max_score_abs = json_i64_value(object, "max_score_abs");
    family.max_table_output_abs = json_i64_value(object, "max_table_output_abs");
    family.max_delta = json_i64_value(object, "max_delta");
    family.max_exp = json_i64_value(object, "max_exp");
    family.max_sum = json_i64_value(object, "max_sum");
    family.max_aggregate_num = json_i64_value(object, "max_aggregate_num");
    family.max_dividend_abs = json_i64_value(object, "max_dividend_abs");
    family.max_aux_abs = json_i64_value(object, "max_aux_abs");
    family.max_divisor = json_i64_value(object, "max_divisor");
    family.max_field_product_abs = json_i64_value(object, "max_field_product_abs");
    return family;
}

void validate_ledger_identity(const Config& config, const StaticTables& tables,
                              const IntegerBoundLedger& ledger) {
    if (!ledger.authenticated || ledger.schema != "zkDL_gat_integer_bound_ledger_v1" ||
        ledger.ledger_id.empty() || ledger.quant_cfg_id != config.quant_cfg_id ||
        ledger.static_table_id != config.static_table_id || ledger.field_modulus != scalar_modulus_decimal() ||
        ledger.scale_chain_digest != config.scale_chain_digest ||
        ledger.static_table_descriptor_digest != tables.descriptor_digest) {
        fail("authenticated integer-bound ledger identity mismatch");
    }
    if (config.degree_bound_id == "auto" || config.degree_bound_id != ledger.ledger_id) {
        fail("BLOCKED: secure config does not bind the authenticated integer-bound ledger id");
    }
    for (const auto value : {ledger.max_score_abs, ledger.max_table_output_abs, ledger.max_delta,
                             ledger.max_exp, ledger.max_sum, ledger.max_aggregate_num,
                             ledger.max_dividend_abs, ledger.max_aux_abs, ledger.max_divisor}) {
        validate_field_bound(value, "ledger global bound");
    }
    if (ledger.max_divisor <= 0 || ledger.families.empty() || ledger.families.size() > kMaxLedgerFamilies) {
        fail("BLOCKED: integer-bound ledger has an unbounded family registry");
    }
    const u64 hidden_layers = config.architecture.hidden_heads.size();
    if (config.architecture.output_heads > kMaxLedgerFamilies) {
        fail("BLOCKED: architecture has an unbounded output family registry");
    }
    u64 expected_families = config.architecture.output_heads;
    for (const auto heads : config.architecture.hidden_heads) {
        if (heads > kMaxLedgerFamilies || expected_families > kMaxLedgerFamilies - heads) {
            fail("BLOCKED: architecture has an unbounded integer-bound family registry");
        }
        expected_families += heads;
    }
    if (ledger.families.size() != expected_families) {
        fail("integer-bound ledger family count does not match canonical architecture");
    }
    std::set<std::string> family_ids;
    for (std::size_t index = 0; index < ledger.families.size(); ++index) {
        const auto& family = ledger.families[index];
        const bool identity_in_range = family.output
            ? family.layer == hidden_layers && family.head < config.architecture.output_heads
            : family.layer < hidden_layers && family.head < config.architecture.hidden_heads.at(family.layer);
        if (!identity_in_range || !family_ids.insert(std::to_string(family.layer) + ":" +
                                                      std::to_string(family.head) + ":" +
                                                      (family.output ? "out" : "hidden")).second) {
            fail("integer-bound ledger contains an out-of-range or duplicate family identity");
        }
        for (const auto value : {family.max_input_abs, family.max_weight_abs, family.max_attention_abs,
                                 family.max_bias_abs, family.max_score_abs, family.max_table_output_abs,
                                 family.max_delta, family.max_exp, family.max_sum, family.max_aggregate_num,
                                 family.max_dividend_abs, family.max_aux_abs, family.max_divisor,
                                 family.max_field_product_abs}) {
            validate_field_bound(value, "ledger family bound " + std::to_string(index));
        }
        if (family.max_divisor <= 0 || family.max_score_abs > ledger.max_score_abs ||
            family.max_table_output_abs > ledger.max_table_output_abs || family.max_delta > ledger.max_delta ||
            family.max_exp > ledger.max_exp || family.max_sum > ledger.max_sum ||
            family.max_aggregate_num > ledger.max_aggregate_num || family.max_dividend_abs > ledger.max_dividend_abs ||
            family.max_aux_abs > ledger.max_aux_abs || family.max_divisor > ledger.max_divisor) {
            fail("integer-bound ledger family exceeds its global bound");
        }
    }
    if (ledger.digest != digest_ledger(ledger)) fail("authenticated integer-bound ledger digest mismatch");
}

IntegerBoundLedger load_integer_bound_ledger(const Config& config, const StaticTables& tables) {
    require_authentication_file(config, config.integer_bound_ledger_path, "integer-bound ledger");
    const auto text = file_text(config.integer_bound_ledger_path);
    if (trim(text).empty() || trim(text).front() != '{') {
        fail("BLOCKED: integer-bound ledger must use the canonical JSON resource format");
    }
    IntegerBoundLedger ledger;
    if (json_scalar(text, "schema") != "zkDL_gat_integer_bound_ledger_v1") fail("integer-bound ledger schema mismatch");
    ledger.schema = json_scalar(text, "schema");
    ledger.ledger_id = json_scalar(text, "ledger_id");
    ledger.quant_cfg_id = json_scalar(text, "quant_cfg_id");
    ledger.static_table_id = json_scalar(text, "static_table_id");
    ledger.field_modulus = json_scalar(text, "field_modulus");
    ledger.scale_chain_digest = json_scalar(text, "scale_chain_digest");
    ledger.static_table_descriptor_digest = json_scalar(text, "static_table_descriptor_digest");
    ledger.digest = json_scalar(text, "digest");
    ledger.max_score_abs = json_i64_value(text, "max_score_abs");
    ledger.max_table_output_abs = json_i64_value(text, "max_table_output_abs");
    ledger.max_delta = json_i64_value(text, "max_delta");
    ledger.max_exp = json_i64_value(text, "max_exp");
    ledger.max_sum = json_i64_value(text, "max_sum");
    ledger.max_aggregate_num = json_i64_value(text, "max_aggregate_num");
    ledger.max_dividend_abs = json_i64_value(text, "max_dividend_abs");
    ledger.max_aux_abs = json_i64_value(text, "max_aux_abs");
    ledger.max_divisor = json_i64_value(text, "max_divisor");
    for (const auto& object : json_objects(text, "families")) ledger.families.push_back(parse_ledger_family(object));
    ledger.authenticated = true;
    ledger.descriptor_digest = authenticated_asset_digest(config, config.integer_bound_ledger_path);
    validate_ledger_identity(config, tables, ledger);
    return ledger;
}

const IntegerBoundFamily& ledger_family(const IntegerBoundLedger& ledger, u64 layer, u64 head, bool output) {
    const IntegerBoundFamily* found = nullptr;
    for (const auto& family : ledger.families) {
        if (family.layer == layer && family.head == head && family.output == output) {
            if (found) fail("integer-bound ledger contains duplicate family identity");
            found = &family;
        }
    }
    if (!found) fail("integer-bound ledger is missing a required layer/head family");
    return *found;
}

i64 max_abs_matrix(const Matrix& matrix, const std::string& name) {
    __int128 maximum = 0;
    for (const auto value : matrix.values) {
        const __int128 absolute = value < 0 ? -static_cast<__int128>(value) : static_cast<__int128>(value);
        if (absolute > maximum) maximum = absolute;
    }
    return checked_i128(maximum, name);
}

i64 max_abs_quantized_tensor(const ModelBundle& model, const std::string& name, u64 scale) {
    const auto values = quantized_tensor_vector(model, name, scale);
    __int128 maximum = 0;
    for (const auto value : values) {
        const __int128 absolute = value < 0 ? -static_cast<__int128>(value) : static_cast<__int128>(value);
        if (absolute > maximum) maximum = absolute;
    }
    return checked_i128(maximum, name + ".bound");
}

i64 ceil_positive_div(i64 numerator, i64 denominator, const std::string& name) {
    if (numerator < 0 || denominator <= 0) fail("invalid positive division bound in " + name);
    return checked_i128((static_cast<__int128>(numerator) + denominator - 1) / denominator, name);
}

void require_ledger_bound(const IntegerBoundFamily& family, i64 required, i64 registered,
                          const std::string& name) {
    if (required < 0 || registered < 0 || required > registered) {
        fail("BLOCKED: integer-bound ledger admission failed for " + name);
    }
    validate_field_bound(required, name);
}

void require_round_div_no_wrap(i64 dividend_abs, i64 divisor, i64 quotient_abs,
                               const IntegerBoundFamily& family, const std::string& name) {
    if (dividend_abs < 0 || divisor <= 0 || quotient_abs < 0) fail("invalid SignedRoundDiv bound in " + name);
    const i64 two_a_plus_d = checked_i128(static_cast<__int128>(2) * dividend_abs + divisor,
                                          name + ".2a_plus_d");
    const i64 two_dq_plus_r = checked_i128(static_cast<__int128>(2) * divisor * quotient_abs +
                                               (2 * divisor - 1), name + ".2dq_plus_r");
    const i64 two_d_minus_one = checked_i128(static_cast<__int128>(2) * divisor - 1,
                                             name + ".2d_minus_one");
    for (const auto value : {dividend_abs, divisor, quotient_abs, two_a_plus_d,
                             two_dq_plus_r, two_d_minus_one}) {
        validate_field_bound(value, name + ".field_fit");
    }
    const i64 required_aux = std::max({dividend_abs, divisor, quotient_abs, two_a_plus_d,
                                       two_dq_plus_r, two_d_minus_one});
    require_ledger_bound(family, required_aux, family.max_aux_abs, name + ".aux");
}

void admit_integer_bound_ledger(const Config& config, const Graph& graph,
                                const ModelBundle& model, const StaticTables& tables,
                                const IntegerBoundLedger& ledger) {
    if (!tables.authenticated || !ledger.authenticated) fail("BLOCKED: relation resources are not authenticated");
    u64 max_degree = 0;
    for (u64 node = 0; node < graph.nodes; ++node) {
        max_degree = std::max(max_degree, graph.node_edge_ptr.at(node + 1) - graph.node_edge_ptr.at(node));
    }
    if (max_degree == 0) fail("BLOCKED: graph has no destination group for ledger admission");
    const i64 input_bound = max_abs_matrix(graph.feature_values, "input_bound");
    i64 current_bound = input_bound;
    const auto table_output_bound = [&](const std::vector<i64>& values, i64 minimum, i64 maximum,
                                        i64 input_abs, const std::string& name) {
        if (input_abs < 0 || -static_cast<__int128>(input_abs) < minimum || input_abs > maximum) {
            fail("BLOCKED: " + name + " input bound is outside its authenticated table domain");
        }
        const u64 first = static_cast<u64>(static_cast<__int128>(-input_abs) - minimum);
        const u64 last = static_cast<u64>(static_cast<__int128>(input_abs) - minimum);
        if (last >= values.size() || first > last) fail("invalid authenticated table bound in " + name);
        const auto abs_value = [](i64 value) {
            return value < 0 ? -static_cast<__int128>(value) : static_cast<__int128>(value);
        };
        // All authenticated lookup tables are monotone.  Therefore the
        // maximum absolute output on a contiguous query interval is attained
        // at one of its endpoints; do not let a hostile interval create an
        // unbounded verifier-side scan.
        const __int128 maximum_abs = std::max(abs_value(values.at(static_cast<std::size_t>(first))),
                                              abs_value(values.at(static_cast<std::size_t>(last))));
        return checked_i128(maximum_abs, name + ".output_bound");
    };
    const auto require_global = [&](i64 required, i64 registered, const std::string& name) {
        if (required < 0 || registered < 0 || required > registered) fail("BLOCKED: global ledger admission failed for " + name);
        validate_field_bound(required, name);
    };
    require_global(input_bound, ledger.max_score_abs, "input_bound");
    for (u64 layer = 0; layer < config.architecture.hidden_dims.size(); ++layer) {
        const u64 in_dim = layer == 0 ? graph.features : config.architecture.hidden_heads[layer - 1] * config.architecture.hidden_dims[layer - 1];
        const u64 head_dim = config.architecture.hidden_dims[layer];
        i64 next_bound = 0;
        for (u64 head = 0; head < config.architecture.hidden_heads[layer]; ++head) {
            const auto& family = ledger_family(ledger, layer, head, false);
            const std::string prefix = "hidden/layer" + std::to_string(layer) + "/head" + std::to_string(head);
            const i64 weight_bound = max_abs_quantized_tensor(model, prefix + "/seq/kernel",
                                                               config.quantization.hidden_weight.at(layer).at(head));
            const i64 attention_src_bound = max_abs_quantized_tensor(model, prefix + "/attn_src/kernel",
                                                                       config.quantization.hidden_attention_src.at(layer).at(head));
            const i64 attention_dst_bound = max_abs_quantized_tensor(model, prefix + "/attn_dst/kernel",
                                                                       config.quantization.hidden_attention_dst.at(layer).at(head));
            const i64 h_prime_bound = checked_i128(static_cast<__int128>(in_dim) * current_bound * weight_bound,
                                                   prefix + ".h_prime_bound");
            const i64 score_src_bound = checked_i128(static_cast<__int128>(head_dim) * h_prime_bound * attention_src_bound,
                                                     prefix + ".score_src_bound");
            const i64 score_dst_bound = checked_i128(static_cast<__int128>(head_dim) * h_prime_bound * attention_dst_bound,
                                                     prefix + ".score_dst_bound");
            const i64 score_bound = checked_i128(static_cast<__int128>(score_src_bound) + score_dst_bound,
                                                 prefix + ".score_bound");
            const i64 table_bound = table_output_bound(tables.lrelu, tables.lrelu_min, tables.lrelu_max,
                                                       score_bound, prefix + ".lrelu");
            const i64 delta_bound = checked_i128(static_cast<__int128>(2) * table_bound,
                                                 prefix + ".delta_bound");
            if (delta_bound > static_cast<i64>(tables.exp_max)) fail("BLOCKED: ExpMap table domain admission failed for " + prefix);
            const i64 exp_bound = tables.exp.front();
            const i64 sum_bound = checked_i128(static_cast<__int128>(max_degree) * exp_bound, prefix + ".sum_bound");
            const i64 aggregate_num_bound = checked_i128(static_cast<__int128>(sum_bound) * h_prime_bound,
                                                         prefix + ".aggregate_num_bound");
            const i64 dividend_bound = checked_i128(static_cast<__int128>(config.quantization.hidden_aggregate_pre.at(layer)) * aggregate_num_bound,
                                                    prefix + ".dividend_bound");
            const i64 divisor_bound = checked_i128(static_cast<__int128>(config.quantization.hidden_h_prime.at(layer).at(head)) * sum_bound,
                                                   prefix + ".divisor_bound");
            const i64 min_divisor = static_cast<i64>(config.quantization.hidden_h_prime.at(layer).at(head));
            const i64 quotient_bound = ceil_positive_div(dividend_bound, min_divisor, prefix + ".quotient_bound");
            const i64 field_product_bound = checked_i128(static_cast<__int128>(exp_bound) * h_prime_bound,
                                                         prefix + ".field_product_bound");
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
            const i64 elu_bound = table_output_bound(tables.elu, tables.elu_min, tables.elu_max,
                                                     quotient_bound, prefix + ".elu");
            next_bound = std::max(next_bound, elu_bound);
            for (const auto value : {score_bound, delta_bound, sum_bound, aggregate_num_bound,
                                     dividend_bound, divisor_bound, field_product_bound}) require_global(value, ledger.max_dividend_abs, prefix + ".global");
        }
        current_bound = next_bound;
    }
    const u64 output_layer = config.architecture.hidden_dims.size();
    const u64 output_in_dim = config.architecture.hidden_heads.back() * config.architecture.hidden_dims.back();
    i64 output_head_bound = 0;
    for (u64 head = 0; head < config.architecture.output_heads; ++head) {
        const auto& family = ledger_family(ledger, output_layer, head, true);
        const std::string prefix = "output/head" + std::to_string(head);
        const i64 weight_bound = max_abs_quantized_tensor(model, prefix + "/seq/kernel", config.quantization.output_weight.at(head));
        const i64 attention_src_bound = max_abs_quantized_tensor(model, prefix + "/attn_src/kernel", config.quantization.output_attention_src.at(head));
        const i64 attention_dst_bound = max_abs_quantized_tensor(model, prefix + "/attn_dst/kernel", config.quantization.output_attention_dst.at(head));
        const i64 bias_bound = max_abs_quantized_tensor(model, prefix + "/output_bias", config.quantization.output_bias.at(head));
        const i64 y_prime_bound = checked_i128(static_cast<__int128>(output_in_dim) * current_bound * weight_bound,
                                               prefix + ".y_prime_bound");
        const i64 score_src_bound = checked_i128(static_cast<__int128>(config.architecture.classes) * y_prime_bound * attention_src_bound,
                                                 prefix + ".score_src_bound");
        const i64 score_dst_bound = checked_i128(static_cast<__int128>(config.architecture.classes) * y_prime_bound * attention_dst_bound,
                                                 prefix + ".score_dst_bound");
        const i64 score_bound = checked_i128(static_cast<__int128>(score_src_bound) + score_dst_bound, prefix + ".score_bound");
        const i64 table_bound = table_output_bound(tables.lrelu, tables.lrelu_min, tables.lrelu_max, score_bound, prefix + ".lrelu");
        const i64 delta_bound = checked_i128(static_cast<__int128>(2) * table_bound, prefix + ".delta_bound");
        if (delta_bound > static_cast<i64>(tables.exp_max)) fail("BLOCKED: ExpMap table domain admission failed for " + prefix);
        const i64 exp_bound = tables.exp.front();
        const i64 sum_bound = checked_i128(static_cast<__int128>(max_degree) * exp_bound, prefix + ".sum_bound");
        const i64 aggregate_num_bound = checked_i128(static_cast<__int128>(sum_bound) * y_prime_bound, prefix + ".aggregate_num_bound");
        const i64 dividend_bound = checked_i128(static_cast<__int128>(config.quantization.output_pre) * aggregate_num_bound,
                                                prefix + ".dividend_bound");
        const i64 divisor_bound = checked_i128(static_cast<__int128>(config.quantization.output_y_prime.at(head)) * sum_bound,
                                               prefix + ".divisor_bound");
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
        require_ledger_bound(family, std::max({dividend_bound, static_cast<i64>(config.quantization.output_head) * quotient_bound,
                                               static_cast<i64>(config.architecture.output_heads) * head_bound}),
                             family.max_dividend_abs, prefix + ".dividend");
        require_ledger_bound(family, divisor_bound, family.max_divisor, prefix + ".divisor");
        require_ledger_bound(family, field_product_bound, family.max_field_product_abs, prefix + ".field_product");
        require_round_div_no_wrap(dividend_bound, static_cast<i64>(config.quantization.output_y_prime.at(head)), quotient_bound,
                                  family, prefix + ".signed_round_div");
        require_round_div_no_wrap(static_cast<i64>(config.quantization.output_head) * quotient_bound,
                                  static_cast<i64>(config.quantization.output_pre), aligned_bound,
                                  family, prefix + ".output_alignment");
        output_head_bound = std::max(output_head_bound, head_bound);
    }
    const i64 output_sum_bound = checked_i128(static_cast<__int128>(config.architecture.output_heads) * output_head_bound,
                                              "final_average.sum_bound");
    require_global(output_sum_bound, ledger.max_dividend_abs, "final_average.sum");
    const i64 final_quotient_bound = ceil_positive_div(output_sum_bound,
                                                       static_cast<i64>(config.architecture.output_heads),
                                                       "final_average.quotient_bound");
    require_global(final_quotient_bound, ledger.max_dividend_abs, "final_average.quotient");
    const auto& final_family = ledger_family(ledger, output_layer, 0, true);
    require_round_div_no_wrap(output_sum_bound, static_cast<i64>(config.architecture.output_heads),
                              final_quotient_bound, final_family, "final_average");
}

}  // namespace

AssetManifest load_asset_manifest(const Config& config) {
    if (config.asset_manifest_path.empty()) fail("asset manifest path is empty");
    if (std::filesystem::is_symlink(config.asset_manifest_path) ||
        !std::filesystem::is_regular_file(config.asset_manifest_path)) {
        fail("baseline-local asset_manifest is missing or not a regular file: " +
             config.asset_manifest_path.string());
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
    result.integer_bound_ledger = json_string_field(manifest, "integer_bound_ledger");
    result.integer_bound_ledger_descriptor_digest = json_string_field(manifest, "integer_bound_ledger_descriptor_digest");
    if (!is_digest(result.scale_chain_digest) || !is_digest(result.static_table_descriptor_digest) ||
        !is_digest(result.integer_bound_ledger_descriptor_digest) ||
        result.scale_chain_digest != config.scale_chain_digest ||
        result.static_table_descriptor != std::filesystem::relative(
            config.static_table_descriptor_path, config.baseline_root).generic_string() ||
        result.integer_bound_ledger != std::filesystem::relative(
            config.integer_bound_ledger_path, config.baseline_root).generic_string()) {
        fail("asset manifest relation-resource identity mismatch");
    }
    const auto expected_config = std::filesystem::relative(config.config_path, config.baseline_root).generic_string();
    if (result.config != expected_config) fail("asset manifest config identity mismatch");
    result.digest = sha256_file(config.asset_manifest_path);

    const std::regex entry(
        "\\{\\s*\"bytes\"\\s*:\\s*([0-9]+)\\s*,\\s*\"path\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"sha256\"\\s*:\\s*\"([0-9a-fA-F]{64})\"\\s*\\}");
    std::set<std::string> paths;
    const std::string checkpoint_root = std::filesystem::relative(
        config.checkpoint_bundle, config.baseline_root).generic_string();
    const std::string checkpoint_prefix = checkpoint_root.empty() ? std::string() : checkpoint_root + "/";
    for (std::sregex_iterator iterator(manifest.begin(), manifest.end(), entry), end; iterator != end; ++iterator) {
        AssetManifestEntry file;
        file.bytes = parse_u64((*iterator)[1].str(), "asset manifest bytes");
        file.path = (*iterator)[2].str();
        file.sha256 = (*iterator)[3].str();
        if (file.path.empty() || std::filesystem::path(file.path).is_absolute()) {
            fail("asset manifest file path is not baseline-local: " + file.path);
        }
        for (const auto& component : std::filesystem::path(file.path)) {
            if (component == "..") fail("asset manifest file path escapes baseline root: " + file.path);
        }
        if (!paths.insert(file.path).second) fail("duplicate asset manifest file: " + file.path);
        const auto resolved = local_path(config.baseline_root, file.path, "asset manifest file");
        const bool private_checkpoint_asset = config.public_verifier_mode &&
            !checkpoint_prefix.empty() && file.path.rfind(checkpoint_prefix, 0U) == 0U &&
            file.path != checkpoint_prefix + "manifest.json";
        if (private_checkpoint_asset) {
            // The digest recorded in the public asset manifest is enough for
            // statement binding.  Do not open/hash checkpoint tensor or model
            // payloads in an independent verifier process.
            result.files.push_back(std::move(file));
            continue;
        }
        if (std::filesystem::is_symlink(resolved) || !std::filesystem::is_regular_file(resolved)) {
            fail("asset manifest file is missing or symlinked: " + file.path);
        }
        if (std::filesystem::file_size(resolved) != file.bytes) fail("asset manifest byte size mismatch: " + file.path);
        if (sha256_file(resolved) != file.sha256) fail("asset manifest digest mismatch: " + file.path);
        result.files.push_back(std::move(file));
    }
    if (result.files.empty()) fail("asset manifest has no files");
    const auto require_entry = [&](const std::string& relative) {
        if (!paths.count(relative)) fail("asset manifest missing required file: " + relative);
    };
    require_entry(result.config);
    require_entry("configs/quantization_scale_chain.cfg");
    require_entry((std::filesystem::relative(config.checkpoint_bundle, config.baseline_root) / "manifest.json").generic_string());
    require_entry((std::filesystem::relative(config.checkpoint_bundle, config.baseline_root) / "tensors.txt").generic_string());
    require_entry((std::filesystem::relative(config.checkpoint_bundle, config.baseline_root) / "tensors.npz").generic_string());
    require_entry((std::filesystem::relative(config.cache_root / config.dataset, config.baseline_root) / "meta.cfg").generic_string());
    require_entry(std::filesystem::relative(config.static_table_descriptor_path, config.baseline_root).generic_string());
    require_entry(std::filesystem::relative(config.integer_bound_ledger_path, config.baseline_root).generic_string());
    const auto manifest_digest_for = [&](const std::string& relative) -> const std::string& {
        for (const auto& file : result.files) if (file.path == relative) return file.sha256;
        fail("asset manifest relation resource entry is missing: " + relative);
    };
    if (manifest_digest_for(result.static_table_descriptor) != result.static_table_descriptor_digest ||
        manifest_digest_for(result.integer_bound_ledger) != result.integer_bound_ledger_descriptor_digest ||
        manifest_digest_for("configs/quantization_scale_chain.cfg") != result.scale_chain_digest) {
        fail("asset manifest relation-resource digest mismatch");
    }
    return result;
}

const char* backend_name(Backend backend) noexcept { return backend == Backend::Cpu ? "cpu" : "gpu"; }

Backend parse_backend(const std::string& value) {
    if (value == "cpu") return Backend::Cpu;
    if (value == "gpu") return Backend::Gpu;
    fail("backend must be cpu or gpu");
}

Config load_config(const std::filesystem::path& path) {
    const auto absolute = std::filesystem::absolute(path).lexically_normal();
    if (absolute.parent_path().filename() != "configs") fail("config must be in baseline-local configs/");
    const auto values = read_kv(absolute);
    Config config;
    config.config_path = absolute;
    config.baseline_root = absolute.parent_path().parent_path();
    const std::set<std::string> canonical_keys = {
        "dataset", "data_root", "cache_root", "export_dir", "checkpoint_bundle", "task_type", "report_unit",
        "batching_rule", "self_loop_rule", "edge_sort_rule", "chunking_rule", "L", "hidden_profile", "d_in_profile",
        "K_out", "num_classes", "range_bits", "seed", "dump_trace", "auto_prepare_dataset", "model_arch_id",
        "model_param_id", "quant_cfg_id", "input_preprocess_id", "static_table_id", "degree_bound_id", "symmetrize_edges",
        "deduplicate_edges", "batch_graphs", "srs_type", "srs_id", "srs_path", "privacy_policy", "input_privacy",
        "output_mode", "opening_mode", "binding_mode", "proof_task", "security_profile_id", "verifier_registry_digest",
        "static_table_descriptor", "integer_bound_ledger"
    };
    for (const auto& item : values) {
        if (!canonical_keys.count(item.first)) fail("unknown non-canonical configuration key: " + item.first);
    }
    const auto required = [&](const std::string& key) -> std::string {
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
    config.static_table_descriptor_path = local_path(
        config.baseline_root, required("static_table_descriptor"), "static_table_descriptor");
    config.integer_bound_ledger_path = local_path(
        config.baseline_root, required("integer_bound_ledger"), "integer_bound_ledger");
    config.model_arch_id = required("model_arch_id"); config.model_param_id = required("model_param_id");
    config.quant_cfg_id = required("quant_cfg_id"); config.static_table_id = required("static_table_id");
    config.input_preprocess_id = required("input_preprocess_id"); config.proof_task = required("proof_task");
    config.privacy_policy = required("privacy_policy"); config.input_privacy = required("input_privacy");
    config.output_mode = required("output_mode"); config.batching_rule = required("batching_rule");
    config.self_loop_rule = required("self_loop_rule"); config.edge_sort_rule = required("edge_sort_rule");
    config.task_type = required("task_type"); config.report_unit = required("report_unit");
    config.chunking_rule = required("chunking_rule"); config.degree_bound_id = required("degree_bound_id");
    config.srs_type = required("srs_type"); config.srs_id = required("srs_id");
    config.srs_path = local_path(config.baseline_root, required("srs_path"), "srs_path").string();
    config.opening_mode = required("opening_mode"); config.binding_mode = required("binding_mode");
    config.security_profile_id = required("security_profile_id"); config.verifier_registry_digest = required("verifier_registry_digest");
    if (!is_digest(config.verifier_registry_digest)) fail("verifier_registry_digest must be SHA-256");
    config.range_bits = parse_u64(required("range_bits"), "range_bits");
    config.batch_graphs = values.count("batch_graphs") ? parse_u64(values.at("batch_graphs"), "batch_graphs") : 1;
    config.seed = parse_u64(required("seed"), "seed");
    config.dump_trace = parse_bool(required("dump_trace"), "dump_trace");
    config.symmetrize_edges = values.count("symmetrize_edges") && parse_bool(values.at("symmetrize_edges"), "symmetrize_edges");
    config.deduplicate_edges = !values.count("deduplicate_edges") || parse_bool(values.at("deduplicate_edges"), "deduplicate_edges");
    config.auto_prepare_dataset = parse_bool(required("auto_prepare_dataset"), "auto_prepare_dataset");
    config.architecture.layers = parse_u64(required("L"), "L");
    const auto profile = parse_profile(required("hidden_profile"));
    for (const auto item : profile) { config.architecture.hidden_heads.push_back(item.first); config.architecture.hidden_dims.push_back(item.second); }
    config.architecture.input_dims = parse_list(required("d_in_profile"), "d_in_profile");
    config.architecture.output_heads = parse_u64(required("K_out"), "K_out");
    config.architecture.classes = parse_u64(required("num_classes"), "num_classes");
    if (config.architecture.layers != config.architecture.hidden_dims.size() + 1 ||
        config.architecture.input_dims.size() != config.architecture.hidden_dims.size()) fail("L/profile dimension mismatch");
    if (config.architecture.layers < 2 || config.architecture.output_heads == 0 || config.architecture.classes == 0 || config.batch_graphs == 0) {
        fail("canonical architecture/batching dimensions are invalid");
    }

    const auto quant_path = config.baseline_root / "configs" / "quantization_scale_chain.cfg";
    if (std::filesystem::is_symlink(quant_path) || !std::filesystem::is_regular_file(quant_path)) {
        fail("BLOCKED: quantization scale-chain file is missing or not a regular baseline-local file");
    }
    config.scale_chain_digest = sha256_file(quant_path);
    const auto quant = read_kv(quant_path);
    if (quant.at("schema") != "gat_l3_scale_chain" || quant.at("quant_cfg_id") != config.quant_cfg_id) {
        fail("quantization scale chain schema/id mismatch");
    }
    const auto scale_required = [&](const std::string& key) -> const std::string& {
        const auto iterator = quant.find(key);
        if (iterator == quant.end()) fail("quantization scale-chain key missing: " + key);
        return iterator->second;
    };
    const auto parse_layer_scales = [&](const std::string& prefix) {
        std::vector<std::vector<u64>> result;
        result.reserve(config.architecture.hidden_dims.size());
        for (u64 layer = 0; layer < config.architecture.hidden_dims.size(); ++layer) {
            result.push_back(parse_list(scale_required(prefix + std::to_string(layer)),
                                        prefix + std::to_string(layer)));
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
    config.quantization.rounding = quant.at("round"); config.quantization.rescale = quant.at("rescale");
    config.quantization.output_average = quant.at("output_average");
    if (config.quantization.rounding != "nearest_ties_away" || config.quantization.rescale != "none" ||
        config.quantization.output_average != "signed_round_div") fail("unsupported non-canonical quantization chain");
    const u64 hidden_layers = config.architecture.hidden_dims.size();
    if (config.quantization.hidden_input.size() != hidden_layers + 1 ||
        config.quantization.hidden_aggregate_pre.size() != hidden_layers ||
        config.quantization.hidden_aggregate.size() != hidden_layers ||
        config.quantization.hidden_concat.size() != hidden_layers ||
        config.quantization.output_weight.size() != config.architecture.output_heads ||
        config.quantization.output_y_prime.size() != config.architecture.output_heads ||
        config.quantization.output_attention_src.size() != config.architecture.output_heads ||
        config.quantization.output_attention_dst.size() != config.architecture.output_heads ||
        config.quantization.output_bias.size() != config.architecture.output_heads) {
        fail("quantization scale-chain vector lengths do not match canonical architecture");
    }
    const auto positive = [](u64 value, const std::string& name) {
        if (value == 0) fail("quantization scale must be positive: " + name);
    };
    positive(config.quantization.input, "s_input"); positive(config.quantization.delta, "s_delta");
    positive(config.quantization.exp, "s_exp"); positive(config.quantization.output_pre, "s_out_pre");
    positive(config.quantization.output_head, "s_out_head"); positive(config.quantization.output, "s_out");
    if (config.quantization.hidden_input.front() != config.quantization.input) {
        fail("s_h[0] must equal s_input");
    }
    const auto product_scale = [](u64 left, u64 right, const std::string& name) {
        if (left != 0 && right > std::numeric_limits<u64>::max() / left) fail("scale-chain overflow in " + name);
        return left * right;
    };
    for (u64 layer = 0; layer < hidden_layers; ++layer) {
        const auto expected_heads = config.architecture.hidden_heads.at(layer);
        for (u64 head = 0; head < expected_heads; ++head) {
            if (config.quantization.hidden_weight.at(layer).size() != expected_heads ||
                config.quantization.hidden_h_prime.at(layer).size() != expected_heads ||
                config.quantization.hidden_attention_src.at(layer).size() != expected_heads ||
                config.quantization.hidden_attention_dst.at(layer).size() != expected_heads) {
                fail("hidden scale-chain head count mismatch at layer " + std::to_string(layer));
            }
            const auto h_prime = product_scale(config.quantization.hidden_input.at(layer),
                                               config.quantization.hidden_weight.at(layer).at(head), "S_Hprime");
            if (h_prime != config.quantization.hidden_h_prime.at(layer).at(head) ||
                product_scale(h_prime, config.quantization.hidden_attention_src.at(layer).at(head), "S_delta_src") != config.quantization.delta ||
                product_scale(h_prime, config.quantization.hidden_attention_dst.at(layer).at(head), "S_delta_dst") != config.quantization.delta) {
                fail("hidden projection/attention scale-chain mismatch at layer " + std::to_string(layer) +
                     " head " + std::to_string(head));
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
    if (config.quantization.output_pre == 0 || config.quantization.output_head == 0 ||
        config.quantization.output != config.quantization.output_head) {
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
    config.config_digest = sha256_file(absolute);
    return config;
}

ModelBundle load_model_bundle(const Config& config) {
    const auto assets = load_asset_manifest(config);
    if (!std::filesystem::is_directory(config.checkpoint_bundle)) fail("checkpoint bundle is not a directory");
    const auto manifest_path = config.checkpoint_bundle / "manifest.json";
    const auto manifest = file_text(manifest_path);
    expect_manifest(manifest, config);
    const auto manifest_shapes = manifest_tensor_shapes(manifest);
    ModelBundle model;
    model.root = config.checkpoint_bundle;
    model.dataset = config.dataset; model.model_arch_id = config.model_arch_id;
    model.model_param_id = config.model_param_id; model.quant_cfg_id = config.quant_cfg_id;
    model.static_table_id = config.static_table_id; model.manifest_digest = sha256_file(manifest_path);
    model.tensor_count_manifest = json_u64_field(manifest, "tensor_count");
    const auto asset_digest = [&](const std::filesystem::path& path) -> std::string {
        const auto relative = std::filesystem::relative(path, config.baseline_root).generic_string();
        for (const auto& file : assets.files) if (file.path == relative) return file.sha256;
        fail("asset manifest does not authenticate checkpoint file: " + relative);
    };
    model.tensors_text_digest = asset_digest(config.checkpoint_bundle / "tensors.txt");
    model.tensors_npz_digest = asset_digest(config.checkpoint_bundle / "tensors.npz");
    const auto best_model = config.checkpoint_bundle / "best_model.pt";
    if (std::filesystem::is_regular_file(best_model)) model.best_model_digest = asset_digest(best_model);
    std::ifstream npz(config.checkpoint_bundle / "tensors.npz", std::ios::binary);
    std::array<unsigned char, 4> npz_magic{};
    npz.read(reinterpret_cast<char*>(npz_magic.data()), static_cast<std::streamsize>(npz_magic.size()));
    if (npz.gcount() != static_cast<std::streamsize>(npz_magic.size()) || npz_magic[0] != 'P' ||
        npz_magic[1] != 'K' || npz_magic[2] != 3 || npz_magic[3] != 4) {
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
        for (u64& dimension : tensor.shape) {
            input >> dimension;
            if (!input || dimension == 0 || product > std::numeric_limits<u64>::max() / dimension) fail("invalid tensor shape");
            product *= dimension;
        }
        input >> count;
        if (!input || count != product) fail("checkpoint tensor count mismatch: " + tensor.name);
        tensor.values.resize(static_cast<std::size_t>(count));
        for (double& value : tensor.values) {
            input >> value;
            if (!input || !std::isfinite(value)) fail("invalid checkpoint tensor value: " + tensor.name);
        }
        if (!model.tensors.emplace(tensor.name, std::move(tensor)).second) fail("duplicate checkpoint tensor");
    }
    if (model.tensors.empty()) fail("checkpoint contains no tensors");
    if (model.tensors.size() != model.tensor_count_manifest || manifest_shapes.size() != model.tensor_count_manifest) {
        fail("checkpoint tensor_count does not match manifest tensor_index");
    }
    for (const auto& item : model.tensors) {
        const auto shape = manifest_shapes.find(item.first);
        if (shape == manifest_shapes.end()) fail("checkpoint tensor is absent from manifest tensor_index: " + item.first);
        if (shape->second != item.second.shape) fail("checkpoint tensor shape disagrees with manifest: " + item.first);
    }
    return model;
}

ModelBundle load_model_metadata(const Config& config) {
    // This path intentionally authenticates only public checkpoint metadata.
    // load_asset_manifest honors public_verifier_mode and records, but does
    // not open/hash, private checkpoint payloads.
    const auto assets = load_asset_manifest(config);
    if (!std::filesystem::is_directory(config.checkpoint_bundle))
        fail("checkpoint bundle is not a directory");
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
    const auto recorded_digest = [&](const std::filesystem::path& path) -> std::string {
        const auto relative = std::filesystem::relative(path, config.baseline_root).generic_string();
        for (const auto& file : assets.files)
            if (file.path == relative) return file.sha256;
        fail("asset manifest does not authenticate checkpoint metadata: " + relative);
    };
    // These are recorded identities only.  No private payload is opened or
    // parsed by this metadata-only loader.
    model.tensors_text_digest = recorded_digest(config.checkpoint_bundle / "tensors.txt");
    model.tensors_npz_digest = recorded_digest(config.checkpoint_bundle / "tensors.npz");
    const auto best_model = config.checkpoint_bundle / "best_model.pt";
    const auto best_relative = std::filesystem::relative(best_model, config.baseline_root).generic_string();
    for (const auto& file : assets.files)
        if (file.path == best_relative) {
            model.best_model_digest = file.sha256;
            break;
        }
    return model;
}

Graph load_graph(const Config& config) {
    const auto directory = config.cache_root / config.dataset;
    const auto metadata = read_kv(directory / "meta.cfg");
    const auto metadata_value = [&](const std::string& key) -> std::string {
        const auto iterator = metadata.find(key);
        if (iterator == metadata.end()) fail("cache metadata key missing: " + key);
        return iterator->second;
    };
    if (metadata_value("name") != config.dataset) fail("cache dataset name mismatch");
    const u64 nodes = parse_u64(metadata_value("num_nodes"), "num_nodes");
    const u64 features = parse_u64(metadata_value("num_features"), "num_features");
    if (metadata_value("input_preprocess_id") != config.input_preprocess_id) fail("cache input_preprocess_id mismatch");
    if (metadata.count("num_classes") && parse_u64(metadata.at("num_classes"), "num_classes") != config.architecture.classes) {
        fail("cache class count mismatch");
    }
    if (metadata.count("task_type") && metadata.at("task_type") != config.task_type) fail("cache task_type mismatch");
    if (metadata.count("report_unit") && metadata.at("report_unit") != config.report_unit) fail("cache report_unit mismatch");
    if (config.architecture.input_dims.front() != features) fail("feature dimension mismatch");
    Graph graph;
    graph.total_nodes = nodes; graph.nodes = nodes; graph.features = features;
    graph.graphs = metadata.count("graph_count") ? parse_u64(metadata.at("graph_count"), "graph_count") : 1;
    if (graph.graphs == 0) fail("graph_count must be positive");
    if (config.dataset != "ppi" && graph.graphs != 1) {
        fail("non-PPI canonical graph must contain exactly one graph");
    }
    graph.absolute_ids.resize(nodes); std::iota(graph.absolute_ids.begin(), graph.absolute_ids.end(), 0);
    graph.node_ptr = {0, nodes};
    if (config.dataset == "ppi") {
        graph.node_ptr.clear();
        std::ifstream ptr(directory / "node_ptr.txt");
        if (!ptr) fail("missing PPI node_ptr");
        u64 value = 0;
        while (ptr >> value) graph.node_ptr.push_back(value);
        if (graph.node_ptr.size() != graph.graphs + 1 || graph.node_ptr.empty() || graph.node_ptr.front() != 0 ||
            graph.node_ptr.back() != nodes) fail("PPI node_ptr mismatch");
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
        u64 rows = 0, cols = 0;
        input >> rows >> cols;
        if (rows != nodes || cols != features) fail("sparse feature shape mismatch");
        std::string line;
        std::getline(input, line);
        for (u64 row = 0; row < nodes; ++row) {
            if (!std::getline(input, line)) fail("truncated sparse feature file");
            std::stringstream tokens(line);
            std::string token;
            long double sum = 0;
            std::vector<std::pair<u64, double>> entries;
            while (tokens >> token) {
                const auto colon = token.find(':');
                if (colon == std::string::npos) fail("malformed sparse feature token");
                const u64 column = parse_u64(token.substr(0, colon), "sparse feature index");
                if (column >= features) fail("sparse feature index out of range");
                const double value = std::stod(token.substr(colon + 1));
                if (!std::isfinite(value)) fail("non-finite sparse feature");
                entries.emplace_back(column, value); sum += value;
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
    while (std::getline(edges_file, line)) {
        line = trim(line); if (line.empty()) continue;
        std::stringstream stream(line);
        std::vector<u64> values; u64 value = 0;
        while (stream >> value) values.push_back(value);
        if ((config.dataset == "ppi" && values.size() != 3) || (config.dataset != "ppi" && values.size() != 2)) fail("malformed edge row");
        const u64 gid = config.dataset == "ppi" ? values[0] : 0;
        if (config.dataset == "ppi" && gid >= graph.graphs) fail("PPI edge graph id out of range");
        if (config.dataset == "ppi") {
            const u64 graph_nodes = graph.node_ptr.at(gid + 1) - graph.node_ptr.at(gid);
            if (values[1] >= graph_nodes || values[2] >= graph_nodes) fail("PPI edge local endpoint out of range");
        }
        const u64 source = config.dataset == "ppi" ? graph.node_ptr.at(gid) + values[1] : values[0];
        const u64 destination = config.dataset == "ppi" ? graph.node_ptr.at(gid) + values[2] : values[1];
        if (source >= nodes || destination >= nodes) fail("edge endpoint out of range");
        const auto add = [&](u64 src, u64 dst) {
            if (config.deduplicate_edges && !seen.emplace(gid, src, dst).second) return;
            edges.push_back({gid, src, dst, ordinal++});
        };
        add(source, destination);
        if (config.symmetrize_edges) add(destination, source);
    }
    for (u64 node = 0; node < nodes; ++node) {
        u64 gid = 0;
        if (config.dataset == "ppi") gid = static_cast<u64>(std::upper_bound(graph.node_ptr.begin(), graph.node_ptr.end(), node) - graph.node_ptr.begin() - 1);
        if (!config.deduplicate_edges || seen.emplace(gid, node, node).second) edges.push_back({gid, node, node, ordinal++});
    }
    std::stable_sort(edges.begin(), edges.end(), [](const RawEdge& left, const RawEdge& right) {
        if (left.graph != right.graph) return left.graph < right.graph;
        if (left.dst != right.dst) return left.dst < right.dst;
        return left.ordinal < right.ordinal;
    });
    graph.src.reserve(edges.size()); graph.dst.reserve(edges.size());
    graph.node_edge_ptr.assign(nodes + 1, 0);
    for (const auto& edge : edges) {
        graph.src.push_back(static_cast<std::uint32_t>(edge.src)); graph.dst.push_back(static_cast<std::uint32_t>(edge.dst));
        ++graph.node_edge_ptr[edge.dst + 1];
    }
    for (u64 node = 1; node <= nodes; ++node) graph.node_edge_ptr[node] += graph.node_edge_ptr[node - 1];
    graph.edges = graph.src.size();
    graph.edge_ptr.assign(graph.graphs + 1, 0);
    for (const auto& edge : edges) ++graph.edge_ptr[edge.graph + 1];
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
                const u64 source_graph = static_cast<u64>(std::upper_bound(graph.node_ptr.begin(), graph.node_ptr.end(), graph.src[edge]) - graph.node_ptr.begin() - 1);
                const u64 destination_graph = static_cast<u64>(std::upper_bound(graph.node_ptr.begin(), graph.node_ptr.end(), graph.dst[edge]) - graph.node_ptr.begin() - 1);
                if (source_graph != destination_graph) fail("PPI edge crosses graph boundary");
            }
        }
    }
    return graph;
}

StaticTables build_static_tables(const Config& config, const Graph& graph) {
    (void)graph;
    return load_static_table_descriptor(config);
}

RelationPlan build_relation_plan(const Config& config, const Graph& graph, const StaticTables& tables) {
    RelationPlan plan;
    if (!tables.authenticated) fail("BLOCKED: relation plan cannot use unauthenticated static tables");
    plan.static_table_descriptor_digest = tables.descriptor_digest;
    plan.integer_bound_ledger = load_integer_bound_ledger(config, tables);
    plan.scale_chain_digest = config.scale_chain_digest;
    plan.integer_bound_ledger_digest = plan.integer_bound_ledger.digest;
    plan.integer_bound_ledger_descriptor_digest = plan.integer_bound_ledger.descriptor_digest;
    const auto power_of_two = [](u64 value) {
        if (value == 0) fail("relation plan contains a zero logical dimension");
        u64 padded = 1;
        while (padded < value) {
            if (padded > std::numeric_limits<u64>::max() / 2) fail("relation plan padded dimension overflow");
            padded <<= 1;
        }
        return padded;
    };
    std::set<std::string> tensor_ids;
    std::set<std::string> operation_ids;
    const auto stage_order = [](u64 layer, u64 head) {
        return "layer=" + std::to_string(layer) + ";head=" + std::to_string(head) + ";node-edge-canonical";
    };
    const auto add_tensor = [&](std::string id, std::string visibility, std::string order,
                                std::string axis, std::vector<u64> logical,
                                std::vector<std::string> instances, std::string instance_order,
                                std::string mapping, u64 degree) {
        if (id.empty() || !tensor_ids.insert(id).second) fail("duplicate relation tensor family: " + id);
        if (logical.empty() || instances.empty() || mapping.empty() || degree == 0) {
            fail("relation tensor has incomplete canonical metadata: " + id);
        }
        if (visibility != "public" && visibility != "witness") fail("invalid tensor visibility: " + id);
        TensorFamily family;
        family.id = std::move(id); family.visibility = std::move(visibility);
        family.mle_order = std::move(order); family.stack_axis = std::move(axis);
        family.logical_shape = std::move(logical); family.padded_shape.reserve(family.logical_shape.size());
        for (const auto dimension : family.logical_shape) family.padded_shape.push_back(power_of_two(dimension));
        family.instances = std::move(instances); family.instance_order = std::move(instance_order);
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
    const auto add_op = [&](std::string id, std::string kind, u64 degree,
                            std::vector<std::string> inputs, std::vector<std::string> outputs,
                            std::vector<std::string> views, bool reindex, bool public_route,
                            std::vector<std::string> auxiliary, std::string selector) {
        if (id.empty() || !operation_ids.insert(id).second) fail("duplicate relation operation family: " + id);
        if (degree == 0 || kind.empty() || inputs.empty() || outputs.empty() || views.empty() || selector.empty()) {
            fail("relation operation has incomplete canonical metadata: " + id);
        }
        OperationFamily operation;
        operation.id = std::move(id); operation.kind = std::move(kind);
        operation.degree = degree; operation.degree_bound = "zkDL_degree_v1:" + std::to_string(degree);
        operation.inputs = std::move(inputs); operation.outputs = std::move(outputs); operation.views = std::move(views);
        operation.requires_reindex = reindex; operation.public_route = public_route;
        operation.instance_order = "layer-major/head-major/node-major/edge-major";
        operation.mapping = "operation_view_v1->canonical_mle:" + operation.id;
        operation.auxiliary = std::move(auxiliary);
        operation.selector_id = "selector:operation:" + operation.id + ":" + std::move(selector);
        operation.terminal_point = "FS:operation:" + operation.id;
        operation.terminal_value = "claim(" + operation.id + "," + operation.terminal_point + ")";
        operation.terminal_claim = "operation_terminal_claim_v1|operation=" + operation.id + "|point=" + operation.terminal_point + "|value=" + operation.terminal_value;
        plan.operation_families.push_back(std::move(operation));
    };
    const auto add_stage_tensors = [&](const std::string& stage, u64 layer, u64 head,
                                       u64 in_dim, u64 out_dim, bool output) {
        const std::string order = stage_order(layer, head);
        const std::string input_id = stage + "/trace/input";
        const std::string trace_visibility = layer == 0 && !output ? "public" : "witness";
        add_tensor(stage + "/model/seq_kernel", "witness", "model-layer-head-row-major", "model", {1, in_dim, out_dim}, {"model"}, order, "model_param_registry", 1);
        add_tensor(stage + "/model/attn_src", "witness", "model-layer-head-row-major", "model", {out_dim}, {"model"}, order, "model_param_registry", 1);
        add_tensor(stage + "/model/attn_dst", "witness", "model-layer-head-row-major", "model", {out_dim}, {"model"}, order, "model_param_registry", 1);
        if (output) add_tensor(stage + "/model/output_bias", "witness", "model-layer-head-row-major", "model", {out_dim}, {"model"}, order, "model_param_registry", 1);
        add_tensor(input_id, trace_visibility, "node-major-little-endian", "node", {graph.nodes, in_dim}, {"input"}, order, "layer_input_route", 1);
        add_tensor(stage + "/trace/" + (output ? "y_prime" : "h_prime"), "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"projection"}, order, "projection_output", 2);
        for (const auto& name : {std::string("e_src"), std::string("e_dst"), std::string("score"), std::string("leaky_relu"), std::string("delta"), std::string("exp_weight")}) {
            add_tensor(stage + "/trace/" + name, "witness", "edge-major-little-endian", "edge", {graph.edges}, {name}, order, "edge_trace", 1);
        }
        add_tensor(stage + "/trace/inverse_field", "witness", "edge-major-field-little-endian", "edge", {graph.edges}, {"inverse"}, order, "field_exact:bls12-381-fr", 2);
        add_tensor(stage + "/trace/alpha_field", "witness", "edge-major-field-little-endian", "edge", {graph.edges}, {"alpha"}, order, "field_exact:bls12-381-fr", 2);
        add_tensor(stage + "/trace/maximum", "witness", "node-major-little-endian", "node", {graph.nodes}, {"maximum"}, order, "destination_group_max", 1);
        add_tensor(stage + "/trace/denominator", "witness", "node-major-little-endian", "node", {graph.nodes}, {"sum"}, order, "destination_group_sum", 1);
        add_tensor(stage + "/trace/max_selector", "witness", "edge-major-little-endian", "edge", {graph.edges}, {"max_selector"}, order, "max_arg_selector", 1);
        add_tensor(stage + "/trace/range_bits", "witness", "edge-major-little-endian", "bit", {graph.edges, config.range_bits}, {"range"}, order, "range_bit_decomposition", 1);
        for (const auto& name : {std::string("lrelu_idx"), std::string("exp_idx")}) {
            add_tensor(stage + "/trace/" + name, "witness", "edge-or-node-major-little-endian", "lookup",
                       std::vector<u64>{graph.edges},
                       {name}, order, "indexed_public_table", 1);
        }
        for (const auto& name : {std::string("lrelu_bits"), std::string("exp_bits")}) {
            add_tensor(stage + "/trace/" + name, "witness", "lookup-query-bit-major", "bit",
                       std::vector<u64>{graph.edges, config.range_bits},
                       {name}, order, "lookup_index_bits", 1);
        }
        if (!output) {
            add_tensor(stage + "/trace/elu_idx", "witness", "node-major-little-endian", "lookup",
                       {graph.nodes, out_dim}, {"elu_idx"}, order, "indexed_public_table", 1);
            add_tensor(stage + "/trace/elu_bits", "witness", "lookup-query-bit-major", "bit",
                       {graph.nodes, out_dim, config.range_bits}, {"elu_bits"}, order, "lookup_index_bits", 1);
        }
        const std::string value = stage + "/trace/" + (output ? "y_prime" : "h_prime");
        add_tensor(stage + "/trace/routed_" + (output ? "y_prime" : "h_prime"), "witness", "edge-major-little-endian", "edge", {graph.edges, out_dim}, {"route"}, order, "public_sparse_route", 1);
        add_tensor(stage + "/trace/product_u_" + (output ? "y_prime" : "h_prime"), "witness", "edge-major-little-endian", "edge", {graph.edges, out_dim}, {"u_product"}, order, "integer_product", 2);
        add_tensor(stage + "/trace/product_alpha_" + (output ? "y_prime" : "h_prime") + "_field", "witness", "edge-major-field-little-endian", "edge", {graph.edges, out_dim}, {"alpha_product"}, order, "field_product:bls12-381-fr", 2);
        add_tensor(stage + "/trace/aggregate_num", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"aggregate_num"}, order, "integer_group_sum", 1);
        add_tensor(stage + "/trace/aggregate_pre_field", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"aggregate_pre_field"}, order, "field_group_sum", 2);
        add_tensor(stage + "/trace/consistency_residual_field", "witness", "node-major-field-little-endian", "node", {graph.nodes, out_dim}, {"consistency"}, order, "field_integer_consistency:bls12-381-fr", 2);
        add_tensor(stage + "/trace/rounddiv_aux", "witness", "node-major-little-endian", "aux", {graph.nodes, 8, out_dim}, {"division"}, order, "signed_round_div_auxiliary", 1);
        add_tensor(stage + "/trace/rounddiv_a_inverse", "witness", "node-major-field-little-endian", "aux", {graph.nodes, out_dim}, {"a_inverse"}, order, "field_exact:bls12-381-fr", 1);
        add_tensor(stage + "/trace/rounddiv_divisor_inverse", "witness", "node-major-field-little-endian", "aux", {graph.nodes, out_dim}, {"divisor_inverse"}, order, "field_exact:bls12-381-fr", 1);
        add_tensor(stage + "/trace/aggregate_pre", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"aggregate_pre"}, order, "signed_round_div", 1);
        if (output) {
            add_tensor(stage + "/trace/y_quantized", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"y_quantized"}, order, "output_quantization", 1);
            add_tensor(stage + "/trace/aligned", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"aligned"}, order, "output_scale_alignment", 1);
            add_tensor(stage + "/trace/output", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"output"}, order, "output_bias_addition", 1);
        } else {
            add_tensor(stage + "/trace/elu_input", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"elu_input"}, order, "elu_domain", 1);
            add_tensor(stage + "/trace/aggregate", "witness", "node-major-little-endian", "node", {graph.nodes, out_dim}, {"aggregate"}, order, "elu_table", 1);
        }
        (void)value;
    };
    add_tensor("input_features", "public", "node-major-little-endian", "node", {graph.nodes, graph.features}, {"H_1"}, "graph-major/node-major", "public_feature_index", 1);
    add_tensor("sparse_route", "public", "edge-major-little-endian", "edge", {graph.edges, 2}, {"src", "dst"}, "graph-major/destination-major", "public_edge_route", 1);
    add_tensor("group_selectors", "public", "edge-major-little-endian", "edge", {graph.edges, 2}, {"group_new", "group_end"}, "graph-major/destination-major", "public_group_selectors", 1);
    add_tensor("node_ptr", "public", "graph-major-little-endian", "graph", {graph.node_ptr.size()}, {"node_ptr"}, "graph-major", "ppi_graph_boundary", 1);
    add_tensor("edge_ptr", "public", "graph-major-little-endian", "graph", {graph.edge_ptr.size()}, {"edge_ptr"}, "graph-major", "ppi_edge_boundary", 1);
    add_tensor("node_edge_ptr", "public", "node-major-little-endian", "node", {graph.node_edge_ptr.size()}, {"node_edge_ptr"}, "node-major", "destination_segment_boundary", 1);
    add_tensor("static_lrelu", "public", "table-index-major", "table", {static_cast<u64>(tables.lrelu.size())}, {"T_LReLU"}, "canonical-table-row-order", "static_table:lrelu", 1);
    add_tensor("static_elu", "public", "table-index-major", "table", {static_cast<u64>(tables.elu.size())}, {"T_ELU"}, "canonical-table-row-order", "static_table:elu", 1);
    add_tensor("static_exp", "public", "table-index-major", "table", {static_cast<u64>(tables.exp.size())}, {"T_exp"}, "canonical-table-row-order", "static_table:exp", 1);
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
            add_op(stage + "/projection", "tensor_matmul_sumcheck", 2, {input, stage + "/model/seq_kernel"}, {hp}, {"node", "inner", "feature"}, true, false, {"selector:" + stage + "/projection"}, "projection_padding:" + stage);
            add_op(stage + "/attention_inner_product", "inner_product_sumcheck", 2, {hp, stage + "/model/attn_src", stage + "/model/attn_dst"}, {stage + "/trace/e_src", stage + "/trace/e_dst"}, {"edge", "inner", "head"}, true, false, {"src", "dst"}, "route_selector:" + stage);
            add_op(stage + "/sparse_route", "public_selector_reindex", 1, {"sparse_route", hp}, {routed}, {"edge", "node", "feature"}, true, true, {"src", "dst", "node_edge_ptr"}, "public_route:" + stage);
            add_op(stage + "/score_add", "field_addition", 1, {stage + "/trace/e_src", stage + "/trace/e_dst"}, {stage + "/trace/score"}, {"edge", "head"}, true, false, {}, "score_add:" + stage);
            add_op(stage + "/leaky_relu", "indexed_public_table_sumcheck", 1, {stage + "/trace/score", "static_lrelu", stage + "/trace/lrelu_idx", stage + "/trace/lrelu_bits"}, {stage + "/trace/leaky_relu"}, {"query", "table"}, true, false, {"tau_lrelu", "omega_lrelu"}, "lookup_lrelu:" + stage);
            add_op(stage + "/max_range", "segmented_max_range_sumcheck", 1, {stage + "/trace/leaky_relu", "sparse_route", "group_selectors"}, {stage + "/trace/maximum", stage + "/trace/max_selector", stage + "/trace/delta"}, {"node", "edge", "group"}, true, true, {"range_bits", "group_new", "group_end"}, "max_range:" + stage);
            add_op(stage + "/exp_map", "indexed_public_table_sumcheck", 1, {stage + "/trace/delta", "static_exp", stage + "/trace/exp_idx", stage + "/trace/exp_bits"}, {stage + "/trace/exp_weight"}, {"query", "table"}, true, false, {"tau_exp", "omega_exp"}, "lookup_exp:" + stage);
            add_op(stage + "/u_h_prime", "tensor_product_sumcheck", 2, {stage + "/trace/exp_weight", routed}, {pu}, {"edge", "feature"}, true, false, {}, "product_u:" + stage);
            add_op(stage + "/group_sum", "segmented_sumcheck", 1, {stage + "/trace/exp_weight", pu, "sparse_route", "edge_ptr", "node_edge_ptr"}, {stage + "/trace/denominator", stage + "/trace/aggregate_num"}, {"graph", "node", "edge", "feature"}, true, true, {"group_new", "group_end", "zero_padding"}, "group_sum:" + stage);
            add_op(stage + "/inverse_alpha", "field_inverse_product_sumcheck", 2, {stage + "/trace/denominator", stage + "/trace/exp_weight", "node_edge_ptr"}, {stage + "/trace/inverse_field", stage + "/trace/alpha_field"}, {"node", "edge"}, true, false, {"inverse_field", "alpha_field", "nonzero_denominator"}, "inverse_alpha:" + stage);
            add_op(stage + "/alpha_h_prime", "tensor_product_sumcheck", 2, {stage + "/trace/alpha_field", routed}, {pa}, {"edge", "feature"}, true, false, {}, "product_alpha:" + stage);
            add_op(stage + "/field_aggregate", "segmented_sumcheck", 1, {pa, "sparse_route", "node_edge_ptr"}, {stage + "/trace/aggregate_pre_field"}, {"node", "edge", "feature"}, true, true, {"group_new", "group_end"}, "field_aggregate:" + stage);
            add_op(stage + "/integer_numerator", "segmented_sumcheck", 1, {pu, "sparse_route", "node_edge_ptr"}, {stage + "/trace/aggregate_num"}, {"node", "edge", "feature"}, true, true, {"group_new", "group_end"}, "integer_numerator:" + stage);
            add_op(stage + "/consistency", "tensor_product_sumcheck", 2, {stage + "/trace/denominator", stage + "/trace/aggregate_pre_field", stage + "/trace/aggregate_num"}, {stage + "/trace/consistency_residual_field"}, {"node", "feature"}, true, false, {}, "consistency:" + stage);
            add_op(stage + "/signed_round_div", "signed_round_div_sumcheck", 1, {stage + "/trace/aggregate_num", stage + "/trace/denominator"}, {stage + "/trace/aggregate_pre", stage + "/trace/rounddiv_aux"}, {"node", "feature"}, true, false, {"sigma", "z", "a", "a_inv", "d_inv", "q", "r", "delta", "range_bits"}, "round_div:" + stage);
            add_op(stage + "/elu", "indexed_public_table_sumcheck", 1, {stage + "/trace/aggregate_pre", "static_elu", stage + "/trace/elu_idx", stage + "/trace/elu_bits"}, {stage + "/trace/aggregate"}, {"query", "table"}, true, false, {"tau_elu", "omega_elu"}, "lookup_elu:" + stage);
            aggregate_inputs.push_back(stage + "/trace/aggregate");
        }
        const std::string concat = "hidden/layer" + std::to_string(layer) + "/concat";
        add_tensor(concat, "witness", "node-major-head-major-feature", "head", {graph.nodes, config.architecture.hidden_heads[layer], head_dim}, {"heads"}, "layer-major/head-major/feature-major", "head_concat_reindex", 1);
        add_op(concat, "tensor_reindex_sumcheck", 1, aggregate_inputs, {concat}, {"node", "head", "feature"}, true, false, {"head_offsets", "zero_padding"}, "concat:" + std::to_string(layer));
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
        add_op(stage + "/output_projection", "tensor_matmul_sumcheck", 2, {input, stage + "/model/seq_kernel"}, {hp}, {"node", "inner", "class"}, true, false, {"zero_padding"}, "output_projection:" + stage);
        add_op(stage + "/output_attention_inner_product", "inner_product_sumcheck", 2, {hp, stage + "/model/attn_src", stage + "/model/attn_dst"}, {stage + "/trace/e_src", stage + "/trace/e_dst"}, {"edge", "inner", "class"}, true, false, {"src", "dst"}, "output_route_selector:" + stage);
        add_op(stage + "/sparse_route", "public_selector_reindex", 1, {"sparse_route", hp}, {routed}, {"edge", "node", "class"}, true, true, {"src", "dst", "node_edge_ptr"}, "public_route:" + stage);
        add_op(stage + "/score_add", "field_addition", 1, {stage + "/trace/e_src", stage + "/trace/e_dst"}, {stage + "/trace/score"}, {"edge", "class"}, true, false, {}, "score_add:" + stage);
        add_op(stage + "/leaky_relu", "indexed_public_table_sumcheck", 1, {stage + "/trace/score", "static_lrelu", stage + "/trace/lrelu_idx", stage + "/trace/lrelu_bits"}, {stage + "/trace/leaky_relu"}, {"query", "table"}, true, false, {"tau_lrelu", "omega_lrelu"}, "lookup_lrelu:" + stage);
        add_op(stage + "/max_range", "segmented_max_range_sumcheck", 1, {stage + "/trace/leaky_relu", "sparse_route", "group_selectors"}, {stage + "/trace/maximum", stage + "/trace/max_selector", stage + "/trace/delta"}, {"node", "edge", "group"}, true, true, {"range_bits", "group_new", "group_end"}, "max_range:" + stage);
        add_op(stage + "/exp_map", "indexed_public_table_sumcheck", 1, {stage + "/trace/delta", "static_exp", stage + "/trace/exp_idx", stage + "/trace/exp_bits"}, {stage + "/trace/exp_weight"}, {"query", "table"}, true, false, {"tau_exp", "omega_exp"}, "lookup_exp:" + stage);
        add_op(stage + "/u_y_prime", "tensor_product_sumcheck", 2, {stage + "/trace/exp_weight", routed}, {pu}, {"edge", "class"}, true, false, {}, "product_u:" + stage);
        add_op(stage + "/group_sum", "segmented_sumcheck", 1, {stage + "/trace/exp_weight", pu, "sparse_route", "edge_ptr", "node_edge_ptr"}, {stage + "/trace/denominator", stage + "/trace/aggregate_num"}, {"graph", "node", "edge", "class"}, true, true, {"group_new", "group_end", "zero_padding"}, "group_sum:" + stage);
        add_op(stage + "/inverse_alpha", "field_inverse_product_sumcheck", 2, {stage + "/trace/denominator", stage + "/trace/exp_weight", "node_edge_ptr"}, {stage + "/trace/inverse_field", stage + "/trace/alpha_field"}, {"node", "edge"}, true, false, {"inverse_field", "alpha_field", "nonzero_denominator"}, "inverse_alpha:" + stage);
        add_op(stage + "/alpha_y_prime", "tensor_product_sumcheck", 2, {stage + "/trace/alpha_field", routed}, {pa}, {"edge", "class"}, true, false, {}, "product_alpha:" + stage);
        add_op(stage + "/field_aggregate", "segmented_sumcheck", 1, {pa, "sparse_route", "node_edge_ptr"}, {stage + "/trace/aggregate_pre_field"}, {"node", "edge", "class"}, true, true, {"group_new", "group_end"}, "field_aggregate:" + stage);
        add_op(stage + "/integer_numerator", "segmented_sumcheck", 1, {pu, "sparse_route", "node_edge_ptr"}, {stage + "/trace/aggregate_num"}, {"node", "edge", "class"}, true, true, {"group_new", "group_end"}, "integer_numerator:" + stage);
        add_op(stage + "/consistency", "tensor_product_sumcheck", 2, {stage + "/trace/denominator", stage + "/trace/aggregate_pre_field", stage + "/trace/aggregate_num"}, {stage + "/trace/consistency_residual_field"}, {"node", "class"}, true, false, {}, "consistency:" + stage);
        add_op(stage + "/signed_round_div", "signed_round_div_sumcheck", 1, {stage + "/trace/aggregate_num", stage + "/trace/denominator"}, {stage + "/trace/y_quantized", stage + "/trace/rounddiv_aux"}, {"node", "class"}, true, false, {"sigma", "z", "a", "a_inv", "d_inv", "q", "r", "delta", "range_bits"}, "round_div:" + stage);
        add_op(stage + "/output_scale_alignment", "signed_round_div_sumcheck", 1, {stage + "/trace/y_quantized"}, {stage + "/trace/aligned"}, {"node", "class"}, true, false, {"scale_out_head", "scale_out_pre"}, "scale_alignment:" + stage);
        add_op(stage + "/output_bias", "field_addition", 1, {stage + "/trace/aligned", stage + "/model/output_bias"}, {stage + "/trace/output"}, {"node", "class"}, true, false, {"bias"}, "output_bias:" + stage);
        output_values.push_back(stage + "/trace/output");
    }
    add_tensor("output_heads", "witness", "node-major-head-major-class", "head", {graph.nodes, config.architecture.output_heads, config.architecture.classes}, {"heads"}, "head-major/node-major/class-major", "output_head_stack", 1);
    add_tensor("output_sum", "witness", "node-major-class-major", "class", {graph.nodes, config.architecture.classes}, {"sum"}, "node-major/class-major", "head_sum", 1);
    add_tensor("public_Y", "public", "node-major-class-major", "class", {graph.nodes, config.architecture.classes}, {"logits"}, "node-major/class-major", "public_output", 1);
    add_op("head_sum", "tensor_sumcheck", 1, output_values, {"output_sum"}, {"node", "head", "class"}, true, false, {"head_offsets", "zero_padding"}, "head_sum");
    add_op("final_average", "signed_round_div_sumcheck", 1, {"output_sum", "output_head_count"}, {"public_Y"}, {"node", "class"}, true, true, {"rounddiv_aux", "public_output"}, "final_average");
    add_op("terminal_opening", "hyrax_terminal_evaluation", 1, {"public_Y"}, {"public_Y"}, {"public_mle_point"}, false, true, {"opening_point", "terminal_claim"}, "public_output_terminal");

    // Operation descriptors consume explicit, same-domain MLE views.  Keep
    // those views in the authenticated plan (rather than smuggling their
    // domains through a prover-only side map) so the independent verifier can
    // reconstruct the exact route, broadcast, prefix, lookup-tree, and
    // SignedRoundDiv shapes from the same registry.  Values are populated by
    // build_native_relation_witness below; these families only describe their
    // canonical logical/padded domains and mapping identity.
    const auto append_view = [&](const std::string& id, const std::vector<u64>& logical,
                                 const std::string& mapping, const std::string& axis) {
        if (id.empty() || logical.empty() || !tensor_ids.insert(id).second)
            fail("duplicate/empty relation operation view family: " + id);
        TensorFamily family;
        family.id = id;
        family.visibility = "witness";
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
        family.terminal_claim = "terminal_claim_v1|family=" + id + "|point=" + family.terminal_point +
                                "|value=" + family.terminal_value;
        family.degree_bound = 1U;
        plan.tensor_families.push_back(std::move(family));
    };
    if (config.range_bits == 0U || config.range_bits >= 31U)
        fail("relation operation views require a bounded range_bits width");
    const auto edge_shape = [&](u64 width = 1U) {
        return width == 1U ? std::vector<u64>{graph.edges}
                           : std::vector<u64>{graph.edges, width};
    };
    const auto node_shape = [&](u64 width = 1U) {
        return width == 1U ? std::vector<u64>{graph.nodes}
                           : std::vector<u64>{graph.nodes, width};
    };
    const auto table_depth = [](std::size_t size) {
        u64 depth = 0U;
        std::size_t width = 1U;
        while (width < size) { width <<= 1U; ++depth; }
        return depth;
    };
    const auto append_round_views = [&](const std::string& operation, u64 width) {
        const auto shape = node_shape(width);
        for (const auto& role : {"signed_numerator", "divisor", "signed_quotient", "a", "q_abs",
                                 "r", "delta", "sigma", "z", "a_inverse", "divisor_inverse"})
            append_view(operation + "/view/" + std::string(role), shape,
                        "signed_round_div:" + std::string(role), "node");
        for (const auto& value : {"a", "q_abs", "r", "delta", "divisor"})
            for (u64 bit = 0U; bit < config.range_bits; ++bit)
                append_view(operation + "/view/range/" + std::string(value) + "/" + std::to_string(bit),
                            shape, "signed_round_div:range:" + std::string(value), "node");
    };
    const auto append_table_views = [&](const std::string& operation, bool edge_domain,
                                        u64 feature_width, std::size_t table_size) {
        const auto shape = edge_domain ? edge_shape(feature_width) : node_shape(feature_width);
        append_view(operation + "/view/query", shape, "indexed_public_table:query", "lookup");
        append_view(operation + "/view/index", shape, "indexed_public_table:index", "lookup");
        append_view(operation + "/view/output0", shape, "indexed_public_table:output", "lookup");
        const u64 depth = table_depth(table_size);
        for (u64 bit = 0U; bit < depth; ++bit)
            append_view(operation + "/view/bit/" + std::to_string(bit), shape,
                        "indexed_public_table:bit", "lookup");
        for (u64 level = 0U; level <= depth; ++level) {
            const u64 level_width = 1ULL << level;
            for (u64 node = 0U; node < level_width; ++node)
                append_view(operation + "/view/chi/" + std::to_string(level) + "/" + std::to_string(node),
                            shape, "indexed_public_table:chi", "lookup");
        }
    };
    const auto append_stage_views = [&](const std::string& stage, u64 width, bool output) {
        const auto edge = edge_shape();
        const auto edge_feature = edge_shape(width);
        const auto node = node_shape();
        const auto node_feature = node_shape(width);
        const std::string inner = stage + (output ? "/output_attention_inner_product" : "/attention_inner_product");
        append_view(inner + "/view/src_matrix", edge_feature, "attention_route:src", "edge");
        append_view(inner + "/view/dst_matrix", edge_feature, "attention_route:dst", "edge");
        append_view(inner + "/view/src_output", edge, "attention_route:e_src", "edge");
        append_view(inner + "/view/dst_output", edge, "attention_route:e_dst", "edge");

        const std::string max = stage + "/max_range";
        for (const auto& role : {"value", "maximum_broadcast", "selector", "delta", "group_new",
                                 "group_end", "selector_prefix", "selector_prev"})
            append_view(max + "/view/" + std::string(role), edge,
                        "max_range:" + std::string(role), "edge");
        for (u64 bit = 0U; bit < config.range_bits; ++bit)
            append_view(max + "/view/range_bit/" + std::to_string(bit), edge,
                        "max_range:range_bit", "edge");

        const std::string group = stage + "/group_sum";
        for (const auto& role : {"group_new", "group_end"})
            append_view(group + "/view/" + std::string(role), edge_feature,
                        "group_sum:" + std::string(role), "edge");
        for (const auto& role : {"input0", "input1", "output0", "output1", "prefix0", "prefix_prev0",
                                 "prefix1", "prefix_prev1"})
            append_view(group + "/view/" + std::string(role), edge_feature,
                        "group_sum:" + std::string(role), "edge");

        const std::string inverse = stage + "/inverse_alpha";
        for (const auto& role : {"input0", "input1", "output0", "output1"})
            append_view(inverse + "/view/" + std::string(role), edge,
                        "inverse_alpha:" + std::string(role), "edge");

        for (const auto& operation : {stage + "/field_aggregate", stage + "/integer_numerator"}) {
            append_view(operation + "/view/group_new", edge_feature, "segmented_sum:group_new", "edge");
            append_view(operation + "/view/group_end", edge_feature, "segmented_sum:group_end", "edge");
            for (const auto& role : {"input0", "output0", "prefix0", "prefix_prev0"})
                append_view(operation + "/view/" + std::string(role), edge_feature,
                            "segmented_sum:" + std::string(role), "edge");
        }

        const std::string consistency = stage + "/consistency";
        for (const auto& role : {"input0", "input1", "input2", "output0"})
            append_view(consistency + "/view/" + std::string(role), node_feature,
                        "consistency:" + std::string(role), "node");

        append_round_views(stage + "/signed_round_div", width);
        const std::string table_lrelu = stage + "/leaky_relu";
        append_table_views(table_lrelu, true, 1U, tables.lrelu.size());
        const std::string table_exp = stage + "/exp_map";
        append_table_views(table_exp, true, 1U, tables.exp.size());
        if (!output) append_table_views(stage + "/elu", false, width, tables.elu.size());
        if (output) {
            append_round_views(stage + "/output_scale_alignment", width);
            const std::string bias = stage + "/output_bias";
            for (const auto& role : {"input0", "input1", "output0"})
                append_view(bias + "/view/" + std::string(role), node_feature,
                            "output_bias:" + std::string(role), "node");
        }
    };
    for (u64 layer = 0U; layer < config.architecture.hidden_dims.size(); ++layer) {
        const u64 width = config.architecture.hidden_dims[layer];
        for (u64 head = 0U; head < config.architecture.hidden_heads[layer]; ++head)
            append_stage_views("hidden/layer" + std::to_string(layer) + "/head" + std::to_string(head), width, false);
        const std::string concat = "hidden/layer" + std::to_string(layer) + "/concat";
        const auto concat_shape = std::vector<u64>{graph.nodes, config.architecture.hidden_heads[layer], width};
        for (u64 head = 0U; head < config.architecture.hidden_heads[layer]; ++head)
            append_view(concat + "/view/input" + std::to_string(head), concat_shape,
                        "head_concat:head" + std::to_string(head), "head");
        append_view(concat + "/view/output", concat_shape, "head_concat:output", "head");
    }
    for (u64 head = 0U; head < config.architecture.output_heads; ++head)
        append_stage_views("output/head" + std::to_string(head), config.architecture.classes, true);
    append_round_views("final_average", config.architecture.classes);

    const auto append_list = [](std::ostringstream& stream, const std::vector<std::string>& values) {
        stream << values.size() << ':';
        for (const auto& value : values) stream << value.size() << ':' << value << ';';
    };
    std::ostringstream registry;
    registry << "schema=" << plan.schema_id << "\nproof_scope=" << plan.proof_scope << "\nbinding=" << plan.binding_rule << "\n"
             << "static_table_descriptor_digest=" << plan.static_table_descriptor_digest << "\n"
             << "scale_chain_digest=" << plan.scale_chain_digest << "\n"
             << "integer_bound_ledger_digest=" << plan.integer_bound_ledger_digest << "\n"
             << "integer_bound_ledger_descriptor_digest=" << plan.integer_bound_ledger_descriptor_digest << "\n";
    for (const auto& family : plan.tensor_families) {
        registry << "T|" << family.id << '|' << family.visibility << '|' << family.mle_order << '|' << family.stack_axis << '|';
        append_list(registry, family.instances);
        registry << family.logical_shape.size() << ':'; for (const auto value : family.logical_shape) registry << value << ',';
        registry << '|' << family.padded_shape.size() << ':'; for (const auto value : family.padded_shape) registry << value << ',';
        registry << '|' << family.instance_order << '|' << family.mapping << '|' << family.zero_selector << '|'
                 << family.selector_id << '|' << family.padding_rule << '|' << family.terminal_point << '|'
                 << family.terminal_value << '|' << family.terminal_claim << '|' << family.degree_bound << '\n';
    }
    for (const auto& operation : plan.operation_families) {
        registry << "O|" << operation.id << '|' << operation.kind << '|' << operation.degree_bound << '|' << operation.degree << '|';
        append_list(registry, operation.inputs); registry << '|'; append_list(registry, operation.outputs); registry << '|'; append_list(registry, operation.views);
        registry << '|' << operation.instance_order << '|' << operation.mapping << '|'; append_list(registry, operation.auxiliary);
        registry << '|' << operation.selector_id << '|' << operation.terminal_point << '|' << operation.terminal_value << '|'
                 << operation.terminal_claim << '|' << (operation.requires_reindex ? 1 : 0) << '|' << (operation.public_route ? 1 : 0) << '\n';
    }
    plan.registry_digest = sha256_text(registry.str());
    std::ostringstream claims;
    std::ostringstream mappings;
    std::ostringstream selectors;
    std::ostringstream degrees;
    for (const auto& family : plan.tensor_families) {
        claims << family.id << '=' << family.terminal_claim << '\n';
        mappings << family.id << '=' << family.mapping << '\n';
        selectors << family.id << '=' << family.selector_id << '|' << family.zero_selector << '|' << family.padding_rule << '\n';
        degrees << family.id << '=' << family.degree_bound << '\n';
    }
    for (const auto& operation : plan.operation_families) {
        claims << operation.id << '=' << operation.terminal_claim << '\n';
        mappings << operation.id << '=' << operation.mapping << '\n';
        selectors << operation.id << '=' << operation.selector_id << '\n';
        degrees << operation.id << '=' << operation.degree_bound << '\n';
    }
    plan.terminal_claims_digest = sha256_text(claims.str());
    plan.mapping_digest = sha256_text(mappings.str());
    plan.selector_digest = sha256_text(selectors.str());
    plan.degree_digest = sha256_text(degrees.str());
    plan.schema_digest = sha256_text(plan.schema_id + "\n" + plan.proof_scope + "\n" + plan.binding_rule + "\n");
    return plan;
}

bool relation_plan_complete(const RelationPlan& relation, std::string* error) {
    const auto reject = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    if (relation.schema_id != "zkDL_gat_inference_v1" || relation.proof_scope != "full_gat_forward" ||
        relation.binding_rule != "fac4dnn_reindex_v1") {
        return reject("relation plan schema/proof/binding identity mismatch");
    }
    for (const auto& digest : {relation.registry_digest, relation.terminal_claims_digest,
                               relation.schema_digest, relation.mapping_digest,
                               relation.selector_digest, relation.degree_digest,
                               relation.static_table_descriptor_digest,
                               relation.scale_chain_digest,
                               relation.integer_bound_ledger_digest,
                               relation.integer_bound_ledger_descriptor_digest}) {
        if (!is_digest(digest)) return reject("relation plan digest is not a canonical SHA-256");
    }
    if (!relation.integer_bound_ledger.authenticated ||
        relation.integer_bound_ledger.digest != relation.integer_bound_ledger_digest ||
        relation.integer_bound_ledger.descriptor_digest != relation.integer_bound_ledger_descriptor_digest ||
        relation.integer_bound_ledger.static_table_descriptor_digest != relation.static_table_descriptor_digest ||
        relation.scale_chain_digest != relation.integer_bound_ledger.scale_chain_digest) {
        return reject("relation plan integer-bound ledger identity is not authenticated");
    }
    std::set<std::string> tensor_ids;
    for (const auto& family : relation.tensor_families) {
        if (family.id.empty() || !tensor_ids.insert(family.id).second) {
            return reject("relation plan has duplicate/empty tensor family id");
        }
        const auto trace_marker = family.id.rfind("/trace/");
        const auto trace_name = trace_marker == std::string::npos
            ? std::string{} : family.id.substr(trace_marker + 7);
        if (trace_name == "inverse" || trace_name == "alpha" ||
            trace_name == "product_alpha_h_prime" || trace_name == "product_alpha_y_prime") {
            return reject("relation plan contains retired integer inverse/alpha/product tensor: " + family.id);
        }
        if (family.visibility != "public" && family.visibility != "witness" ||
            family.logical_shape.empty() || family.logical_shape.size() != family.padded_shape.size() ||
            family.instances.empty() || family.mle_order.empty() || family.stack_axis.empty() ||
            family.instance_order.empty() || family.mapping.empty() || family.selector_id.empty() ||
            family.padding_rule.empty() || family.terminal_point.empty() || family.terminal_value.empty() ||
            family.terminal_claim.empty() || family.degree_bound == 0) {
            return reject("relation tensor family has incomplete canonical metadata: " + family.id);
        }
        for (std::size_t index = 0; index < family.logical_shape.size(); ++index) {
            const u64 logical = family.logical_shape[index];
            const u64 padded = family.padded_shape[index];
            if (logical == 0 || padded < logical || (padded & (padded - 1U)) != 0) {
                return reject("relation tensor family has non-power-of-two padding: " + family.id);
            }
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
    const auto has_tensor = [&](const std::string& id) { return tensor_ids.count(id) != 0; };
    if (!has_tensor("input_features") || !has_tensor("sparse_route") || !has_tensor("group_selectors") ||
        !has_tensor("node_ptr") || !has_tensor("edge_ptr") || !has_tensor("node_edge_ptr") ||
        !has_tensor("static_lrelu") || !has_tensor("static_elu") || !has_tensor("static_exp") ||
        !has_tensor("output_head_count") || !has_tensor("output_heads") || !has_tensor("output_sum") ||
        !has_tensor("public_Y")) {
        return reject("relation plan is missing a canonical public/aggregate tensor family");
    }
    std::set<std::string> operation_ids;
    std::map<std::string, const OperationFamily*> operations;
    for (const auto& operation : relation.operation_families) {
        if (operation.id.empty() || !operation_ids.insert(operation.id).second) {
            return reject("relation plan has duplicate/empty operation family id");
        }
        if (operation.kind.empty() || operation.degree == 0 || operation.degree_bound != "zkDL_degree_v1:" + std::to_string(operation.degree) ||
            operation.inputs.empty() || operation.outputs.empty() || operation.views.empty() || operation.instance_order.empty() ||
            operation.mapping.empty() || operation.selector_id.empty() || operation.terminal_point.empty() ||
            operation.terminal_value.empty() || operation.terminal_claim.empty()) {
            return reject("relation operation has incomplete canonical metadata: " + operation.id);
        }
        for (const auto& input : operation.inputs) if (!has_tensor(input)) return reject("relation operation input is unregistered: " + operation.id + " -> " + input);
        for (const auto& output : operation.outputs) if (!has_tensor(output)) return reject("relation operation output is unregistered: " + operation.id + " -> " + output);
        if (operation.mapping.find("operation_view_v1->canonical_mle:" + operation.id) != 0) return reject("relation operation mapping is not canonical: " + operation.id);
        if (operation.terminal_claim.find("operation_terminal_claim_v1|operation=" + operation.id + "|") != 0 ||
            operation.terminal_claim.find("|point=" + operation.terminal_point + "|") == std::string::npos ||
            operation.terminal_claim.find("|value=" + operation.terminal_value) == std::string::npos) {
            return reject("relation operation terminal claim is not bound to its operation: " + operation.id);
        }
        if (operation.id != "terminal_opening" && !operation.requires_reindex) return reject("non-terminal operation omitted FAC4DNN re-index binding: " + operation.id);
        if (operation.selector_id.find("selector:operation:" + operation.id + ":") != 0) return reject("operation selector namespace is not canonical: " + operation.id);
        if (operation.public_route && operation.selector_id.find("public_route:") == std::string::npos && operation.id != "final_average" && operation.id != "terminal_opening") return reject("public route operation has wrong selector: " + operation.id);
        if (operation.kind.find("indexed_public_table") != std::string::npos &&
            (operation.auxiliary.size() < 2 || operation.auxiliary[0].find("tau_") != 0 || operation.auxiliary[1].find("omega_") != 0)) {
            return reject("indexed table operation lacks independent challenges: " + operation.id);
        }
        if (operation.kind.find("signed_round_div") != std::string::npos && operation.auxiliary.empty()) return reject("round-div operation lacks auxiliary witness registry: " + operation.id);
        operations.emplace(operation.id, &operation);
    }
    const auto require_op = [&](const std::string& id) -> bool {
        if (!operations.count(id)) return reject("relation plan is missing mandatory operation family: " + id);
        return true;
    };
    for (const auto& family : relation.tensor_families) {
        if (family.id.find("/model/seq_kernel") == std::string::npos) continue;
        const auto stage = family.id.substr(0, family.id.find("/model/"));
        if (stage.rfind("hidden/layer", 0) == 0) {
            for (const auto& suffix : {"/projection", "/attention_inner_product", "/sparse_route", "/score_add", "/leaky_relu", "/max_range", "/exp_map", "/u_h_prime", "/group_sum", "/inverse_alpha", "/alpha_h_prime", "/field_aggregate", "/integer_numerator", "/consistency", "/signed_round_div", "/elu"}) {
                if (!require_op(stage + suffix)) return false;
            }
        } else if (stage.rfind("output/head", 0) == 0) {
            for (const auto& suffix : {"/output_projection", "/output_attention_inner_product", "/sparse_route", "/score_add", "/leaky_relu", "/max_range", "/exp_map", "/u_y_prime", "/group_sum", "/inverse_alpha", "/alpha_y_prime", "/field_aggregate", "/integer_numerator", "/consistency", "/signed_round_div", "/output_scale_alignment", "/output_bias"}) {
                if (!require_op(stage + suffix)) return false;
            }
        } else {
            return reject("model tensor family is not layer/head canonical: " + family.id);
        }
    }
    for (const auto& id : {std::string("head_sum"), std::string("final_average"), std::string("terminal_opening")}) {
        if (!require_op(id)) return false;
    }
    const auto terminal = operations.at("terminal_opening");
    if (terminal->inputs != std::vector<std::string>{"public_Y"} || terminal->outputs != std::vector<std::string>{"public_Y"} || !terminal->public_route) {
        return reject("terminal opening does not bind the canonical public output family");
    }
    return true;
}

using NativeFrWire = NativeFieldWire;

const NativeFrWire& native_scalar_modulus() {
    static const NativeFrWire modulus = {{
        1U, 4294967295U, 4294859774U, 1404937218U,
        161601541U, 859428872U, 698187080U, 1944954707U}};
    return modulus;
}

int native_fr_compare(const NativeFrWire& left, const NativeFrWire& right) {
    for (std::size_t index = left.size(); index-- > 0U;) {
        if (left[index] < right[index]) return -1;
        if (left[index] > right[index]) return 1;
    }
    return 0;
}

bool native_fr_canonical(const NativeFrWire& value) {
    return native_fr_compare(value, native_scalar_modulus()) < 0;
}

NativeFrWire native_fr_sub_raw(const NativeFrWire& left, const NativeFrWire& right) {
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

NativeFrWire native_fr_add(const NativeFrWire& left, const NativeFrWire& right) {
    NativeFrWire output{};
    std::uint64_t carry = 0U;
    for (std::size_t index = 0U; index < output.size(); ++index) {
        const std::uint64_t sum = static_cast<std::uint64_t>(left[index]) + right[index] + carry;
        output[index] = static_cast<std::uint32_t>(sum);
        carry = sum >> 32U;
    }
    if (carry != 0U || native_fr_compare(output, native_scalar_modulus()) >= 0)
        output = native_fr_sub_raw(output, native_scalar_modulus());
    return output;
}

NativeFrWire native_fr_neg(const NativeFrWire& value) {
    NativeFrWire zero{};
    return native_fr_canonical(value) && native_fr_compare(value, zero) != 0
        ? native_fr_sub_raw(native_scalar_modulus(), value) : zero;
}

NativeFrWire native_fr_from_signed_i64(i64 value) {
    if (value >= 0) {
        NativeFrWire output{};
        output[0] = static_cast<std::uint32_t>(value);
        return output;
    }
    const std::uint64_t magnitude = static_cast<std::uint64_t>(-(value + 1)) + 1U;
    NativeFrWire output{};
    output[0] = static_cast<std::uint32_t>(magnitude);
    output[1] = static_cast<std::uint32_t>(magnitude >> 32U);
    return native_fr_neg(output);
}

NativeFrWire native_wire(i64 value) {
    return native_fr_from_signed_i64(value);
}

NativeFrWire native_wire(const FieldValue& value) {
    NativeFrWire wire{};
    for (std::size_t limb = 0U; limb < wire.size(); ++limb) {
        const std::size_t offset = limb * sizeof(std::uint32_t);
        wire[limb] = static_cast<std::uint32_t>(value.bytes[offset]) |
                     (static_cast<std::uint32_t>(value.bytes[offset + 1U]) << 8U) |
                     (static_cast<std::uint32_t>(value.bytes[offset + 2U]) << 16U) |
                     (static_cast<std::uint32_t>(value.bytes[offset + 3U]) << 24U);
    }
    if (!native_fr_canonical(wire)) fail("non-canonical field witness encoding");
    return wire;
}

u64 native_shape_product(const std::vector<u64>& shape, const std::string& label) {
    if (shape.empty()) fail("empty canonical tensor shape: " + label);
    u64 product = 1U;
    for (const auto dimension : shape) {
        if (dimension == 0U || product > std::numeric_limits<u64>::max() / dimension)
            fail("canonical tensor shape overflow: " + label);
        product *= dimension;
    }
    return product;
}

std::vector<NativeFrWire> native_embed_zero_padded(
    const TensorFamily& family, const std::vector<NativeFrWire>& logical) {
    const u64 logical_size = native_shape_product(family.logical_shape, family.id + ".logical");
    const u64 padded_size = native_shape_product(family.padded_shape, family.id + ".padded");
    if (logical.size() != logical_size || padded_size > std::numeric_limits<std::uint32_t>::max())
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

std::vector<NativeFrWire> native_i64_values(const std::vector<i64>& values) {
    std::vector<NativeFrWire> result;
    result.reserve(values.size());
    for (const auto value : values) result.push_back(native_wire(value));
    return result;
}

std::vector<NativeFrWire> native_u64_values(const std::vector<u64>& values,
                                            const std::string& label) {
    std::vector<NativeFrWire> result;
    result.reserve(values.size());
    for (const auto value : values) {
        if (value > static_cast<u64>(std::numeric_limits<i64>::max()))
            fail("unsigned graph boundary exceeds field wire signed range: " + label);
        result.push_back(native_wire(static_cast<i64>(value)));
    }
    return result;
}

std::vector<NativeFrWire> native_field_values(const std::vector<FieldValue>& values) {
    std::vector<NativeFrWire> result;
    result.reserve(values.size());
    for (const auto& value : values) result.push_back(native_wire(value));
    return result;
}

std::vector<NativeFrWire> native_matrix_values(const Matrix& matrix) {
    return native_i64_values(matrix.values);
}

std::vector<NativeFrWire> native_field_matrix_values(const FieldMatrix& matrix) {
    return native_field_values(matrix.values);
}

std::vector<NativeFrWire> native_broadcast_node_vector(
    const Graph& graph, const std::vector<i64>& node_values) {
    if (node_values.size() != graph.nodes || graph.dst.size() != graph.edges)
        fail("node-to-edge broadcast shape mismatch");
    std::vector<i64> values(graph.edges, 0);
    for (u64 edge = 0U; edge < graph.edges; ++edge) {
        if (graph.dst[edge] >= graph.nodes) fail("graph destination is outside node domain");
        values[edge] = node_values.at(graph.dst[edge]);
    }
    return native_i64_values(values);
}

std::vector<NativeFrWire> native_broadcast_node_matrix(
    const Graph& graph, const Matrix& matrix) {
    if (matrix.rows != graph.nodes || graph.dst.size() != graph.edges)
        fail("node-to-edge matrix broadcast shape mismatch");
    std::vector<i64> values;
    values.reserve(static_cast<std::size_t>(graph.edges * matrix.cols));
    for (u64 edge = 0U; edge < graph.edges; ++edge) {
        if (graph.dst[edge] >= graph.nodes) fail("graph destination is outside node domain");
        for (u64 column = 0U; column < matrix.cols; ++column)
            values.push_back(matrix.at(graph.dst[edge], column));
    }
    return native_i64_values(values);
}

std::vector<NativeFrWire> native_broadcast_node_field_matrix(
    const Graph& graph, const FieldMatrix& matrix) {
    if (matrix.rows != graph.nodes || graph.dst.size() != graph.edges)
        fail("node-to-edge field matrix broadcast shape mismatch");
    std::vector<NativeFrWire> values;
    values.reserve(static_cast<std::size_t>(graph.edges * matrix.cols));
    for (u64 edge = 0U; edge < graph.edges; ++edge) {
        if (graph.dst[edge] >= graph.nodes) fail("graph destination is outside node domain");
        for (u64 column = 0U; column < matrix.cols; ++column)
            values.push_back(native_wire(matrix.values.at(static_cast<std::size_t>(graph.dst[edge] * matrix.cols + column))));
    }
    return values;
}

std::vector<NativeFrWire> native_route_matrix(
    const Graph& graph, const Matrix& matrix, bool source) {
    if (matrix.rows != graph.nodes || graph.src.size() != graph.edges || graph.dst.size() != graph.edges)
        fail("route matrix shape mismatch");
    std::vector<i64> values;
    values.reserve(static_cast<std::size_t>(graph.edges * matrix.cols));
    for (u64 edge = 0U; edge < graph.edges; ++edge) {
        const u64 node = source ? graph.src[edge] : graph.dst[edge];
        if (node >= graph.nodes) fail("graph route index is outside node domain");
        for (u64 column = 0U; column < matrix.cols; ++column)
            values.push_back(matrix.at(node, column));
    }
    return native_i64_values(values);
}

std::vector<i64> native_node_values_from_edges(const Graph& graph,
                                               const std::vector<i64>& edge_values) {
    if (edge_values.size() != graph.edges || graph.node_edge_ptr.size() != graph.nodes + 1U)
        fail("edge-to-node trace shape mismatch");
    std::vector<i64> result(graph.nodes, 0);
    for (u64 node = 0U; node < graph.nodes; ++node) {
        const u64 begin = graph.node_edge_ptr.at(node);
        const u64 end = graph.node_edge_ptr.at(node + 1U);
        if (begin > end || end > graph.edges) fail("destination group boundary is outside edge domain");
        if (begin != end) result[node] = edge_values.at(static_cast<std::size_t>(begin));
    }
    return result;
}

std::vector<std::uint8_t> native_group_new(const Graph& graph) {
    if (graph.dst.size() != graph.edges) fail("graph group selector shape mismatch");
    std::vector<std::uint8_t> result(graph.edges, 0U);
    for (u64 edge = 0U; edge < graph.edges; ++edge) {
        if (graph.dst[edge] >= graph.nodes) fail("graph destination is outside node domain");
        result[edge] = edge == 0U || graph.dst[edge] != graph.dst[edge - 1U] ? 1U : 0U;
    }
    return result;
}

std::vector<std::uint8_t> native_group_end(const Graph& graph) {
    if (graph.dst.size() != graph.edges) fail("graph group selector shape mismatch");
    std::vector<std::uint8_t> result(graph.edges, 0U);
    for (u64 edge = 0U; edge < graph.edges; ++edge) {
        if (graph.dst[edge] >= graph.nodes) fail("graph destination is outside node domain");
        result[edge] = edge + 1U == graph.edges || graph.dst[edge] != graph.dst[edge + 1U] ? 1U : 0U;
    }
    return result;
}

std::vector<NativeFrWire> native_repeat_edge_bits(
    const std::vector<std::uint8_t>& edge_values, u64 width) {
    std::vector<NativeFrWire> result;
    result.reserve(edge_values.size() * static_cast<std::size_t>(width));
    for (const auto value : edge_values)
        for (u64 index = 0U; index < width; ++index) result.push_back(native_wire(value));
    return result;
}

std::vector<NativeFrWire> native_repeat_node_bits(
    const std::vector<std::uint8_t>& node_values, u64 width) {
    std::vector<NativeFrWire> result;
    result.reserve(node_values.size() * static_cast<std::size_t>(width));
    for (const auto value : node_values)
        for (u64 index = 0U; index < width; ++index) result.push_back(native_wire(value));
    return result;
}

std::vector<NativeFrWire> native_edge_feature_from_scalar(
    const std::vector<i64>& values, u64 width) {
    std::vector<NativeFrWire> result;
    result.reserve(values.size() * static_cast<std::size_t>(width));
    for (const auto value : values)
        for (u64 index = 0U; index < width; ++index) result.push_back(native_wire(value));
    return result;
}

std::vector<NativeFrWire> native_node_feature_from_scalar(
    const std::vector<i64>& values, u64 width) {
    std::vector<NativeFrWire> result;
    result.reserve(values.size() * static_cast<std::size_t>(width));
    for (const auto value : values)
        for (u64 index = 0U; index < width; ++index) result.push_back(native_wire(value));
    return result;
}

std::vector<NativeFrWire> native_bit_column(const std::vector<i64>& values,
                                            u64 bit, const std::string& label) {
    std::vector<NativeFrWire> result;
    result.reserve(values.size());
    for (const auto value : values) {
        if (value < 0) fail("invalid bounded bit source: " + label);
        result.push_back(native_wire(static_cast<i64>((static_cast<std::uint64_t>(value) >> bit) & 1U)));
    }
    return result;
}

std::vector<NativeFrWire> native_edge_feature_from_node_matrix(
    const Graph& graph, const Matrix& matrix) {
    return native_broadcast_node_matrix(graph, matrix);
}

std::vector<NativeFrWire> native_edge_feature_from_matrix(
    const Matrix& matrix) {
    return native_matrix_values(matrix);
}

std::vector<NativeFrWire> native_prefix(const std::vector<NativeFrWire>& input,
                                        const std::vector<std::uint8_t>& group_new,
                                        u64 width, bool previous) {
    if (input.size() != group_new.size() * static_cast<std::size_t>(width))
        fail("segmented prefix input shape mismatch");
    std::vector<NativeFrWire> result(input.size());
    for (u64 edge = 0U; edge < group_new.size(); ++edge) {
        for (u64 column = 0U; column < width; ++column) {
            const std::size_t index = static_cast<std::size_t>(edge * width + column);
            if (previous && (edge == 0U || group_new[edge] != 0U)) {
                result[index] = NativeFrWire{};
            } else if (previous) {
                result[index] = result.at(static_cast<std::size_t>((edge - 1U) * width + column));
            } else {
                const NativeFrWire prior = (edge == 0U || group_new[edge] != 0U)
                    ? NativeFrWire{}
                    : result.at(static_cast<std::size_t>((edge - 1U) * width + column));
                result[index] = native_fr_add(prior, input[index]);
            }
        }
    }
    return result;
}

std::vector<NativeFrWire> native_field_prefix(const std::vector<NativeFrWire>& input,
                                              const std::vector<std::uint8_t>& group_new,
                                              u64 width, bool previous) {
    return native_prefix(input, group_new, width, previous);
}

NativeRelationWitnessInput build_native_relation_witness(
    const Config& config, const Graph& graph, const ModelBundle& model,
    const StaticTables& tables, const RelationPlan& plan, const GatTrace& trace) {
    std::map<std::string, const TensorFamily*> families;
    for (const auto& family : plan.tensor_families) {
        if (!families.emplace(family.id, &family).second)
            fail("duplicate canonical witness family: " + family.id);
    }
    std::map<std::string, std::vector<NativeFrWire>> logical;
    const auto family_for = [&](const std::string& id) -> const TensorFamily& {
        const auto found = families.find(id);
        if (found == families.end()) fail("canonical witness family is not in RelationPlan: " + id);
        return *found->second;
    };
    const auto put = [&](const std::string& id, std::vector<NativeFrWire> values) {
        const auto& family = family_for(id);
        if (values.size() != native_shape_product(family.logical_shape, id + ".logical"))
            fail("canonical witness logical value count mismatch: " + id);
        if (!logical.emplace(id, std::move(values)).second)
            fail("duplicate canonical witness values: " + id);
    };
    const auto put_i64 = [&](const std::string& id, const std::vector<i64>& values) {
        put(id, native_i64_values(values));
    };
    const auto put_u64 = [&](const std::string& id, const std::vector<u64>& values) {
        put(id, native_u64_values(values, id));
    };
    const auto put_matrix = [&](const std::string& id, const Matrix& matrix) {
        put(id, native_matrix_values(matrix));
    };
    const auto put_field = [&](const std::string& id, const std::vector<FieldValue>& values) {
        put(id, native_field_values(values));
    };
    const auto put_field_matrix = [&](const std::string& id, const FieldMatrix& matrix) {
        put(id, native_field_matrix_values(matrix));
    };
    const auto scalar_matrix = [&](const std::vector<i64>& values, u64 width) {
        Matrix matrix = make_matrix(values.size(), width);
        for (u64 row = 0U; row < values.size(); ++row)
            for (u64 column = 0U; column < width; ++column) matrix.at(row, column) = values[row];
        return matrix;
    };
    const auto bits_for = [&](const std::vector<i64>& values, u64 width,
                              const std::string& label) {
        std::vector<NativeFrWire> result;
        result.reserve(values.size() * static_cast<std::size_t>(width));
        for (const auto value : values) {
            if (value < 0) fail("negative value cannot be decomposed into canonical bits: " + label);
            const auto unsigned_value = static_cast<std::uint64_t>(value);
            if (width < 63U && (unsigned_value >> width) != 0U)
                fail("value exceeds bounded bit decomposition: " + label);
            for (u64 bit = 0U; bit < width; ++bit)
                result.push_back(native_wire(static_cast<i64>((unsigned_value >> bit) & 1U)));
        }
        return result;
    };
    const auto group_new = native_group_new(graph);
    const auto group_end = native_group_end(graph);
    const auto edge_group_bits = [&](u64 width) {
        return native_repeat_edge_bits(group_new, width);
    };
    const auto edge_group_end_bits = [&](u64 width) {
        return native_repeat_edge_bits(group_end, width);
    };
    const auto node_vector_from_edges = [&](const std::vector<i64>& values) {
        return native_node_values_from_edges(graph, values);
    };

    std::vector<i64> route;
    route.reserve(static_cast<std::size_t>(graph.edges * 2U));
    for (u64 edge = 0U; edge < graph.edges; ++edge) {
        route.push_back(static_cast<i64>(graph.src.at(static_cast<std::size_t>(edge))));
        route.push_back(static_cast<i64>(graph.dst.at(static_cast<std::size_t>(edge))));
    }
    std::vector<i64> selectors;
    selectors.reserve(static_cast<std::size_t>(graph.edges * 2U));
    for (u64 edge = 0U; edge < graph.edges; ++edge) {
        selectors.push_back(group_new.at(static_cast<std::size_t>(edge)));
        selectors.push_back(group_end.at(static_cast<std::size_t>(edge)));
    }
    put_matrix("input_features", graph.feature_values);
    put_i64("sparse_route", route);
    put_i64("group_selectors", selectors);
    put_u64("node_ptr", graph.node_ptr);
    put_u64("edge_ptr", graph.edge_ptr);
    put_u64("node_edge_ptr", graph.node_edge_ptr);
    put_i64("static_lrelu", tables.lrelu);
    put_i64("static_elu", tables.elu);
    put_i64("static_exp", tables.exp);
    put_i64("output_head_count", {static_cast<i64>(config.architecture.output_heads)});

    const auto round_aux = [&](const SignedRoundDivWitness& witness) {
        if (witness.rows == 0U || witness.cols == 0U || witness.rows != graph.nodes) {
            fail("SignedRoundDiv auxiliary rows do not match canonical node domain");
        }
        const std::size_t count = static_cast<std::size_t>(witness.rows * witness.cols);
        if (witness.sigma.size() != count || witness.z.size() != count || witness.a.size() != count ||
            witness.q.size() != count || witness.r.size() != count || witness.delta.size() != count ||
            witness.divisor.size() != count) {
            fail("SignedRoundDiv auxiliary witness is incomplete");
        }
        std::vector<NativeFrWire> values;
        values.reserve(count * 8U);
        for (u64 row = 0U; row < witness.rows; ++row) {
            for (u64 column = 0U; column < witness.cols; ++column) {
                const std::size_t index = static_cast<std::size_t>(row * witness.cols + column);
                values.push_back(native_wire(static_cast<i64>(witness.sigma[index])));
                values.push_back(native_wire(static_cast<i64>(witness.z[index])));
                values.push_back(native_wire(witness.a[index]));
                values.push_back(native_wire(witness.q[index]));
                values.push_back(native_wire(witness.r[index]));
                values.push_back(native_wire(witness.delta[index]));
                values.push_back(native_wire(witness.divisor[index]));
                values.push_back(NativeFrWire{});  // reserved zero padding channel
            }
        }
        return values;
    };
    const auto lookup_bits = [&](const std::vector<i64>& query, i64 minimum,
                                 i64 maximum, u64 width, const std::string& label) {
        std::vector<i64> offsets;
        offsets.reserve(query.size());
        for (const auto value : query) {
            if (value < minimum || value > maximum)
                fail("lookup query is outside authenticated table domain: " + label);
            offsets.push_back(checked_i128(static_cast<__int128>(value) - minimum, label + ".offset"));
        }
        return bits_for(offsets, width, label + ".bits");
    };

    const auto put_stage_base = [&](const std::string& stage, u64 layer, u64 head, bool output) {
        const u64 width = output ? config.architecture.classes : config.architecture.hidden_dims.at(layer);
        const Matrix& input = output ? trace.hidden_layers.back().concat : trace.hidden_layers.at(layer).input;
        put_matrix(stage + "/trace/input", input);
        const Matrix weight = quantized_tensor_matrix(
            model, stage + "/seq/kernel",
            output ? config.quantization.output_weight.at(head) : config.quantization.hidden_weight.at(layer).at(head));
        const std::vector<i64> src_kernel = quantized_tensor_vector(
            model, stage + "/attn_src/kernel",
            output ? config.quantization.output_attention_src.at(head) : config.quantization.hidden_attention_src.at(layer).at(head));
        const std::vector<i64> dst_kernel = quantized_tensor_vector(
            model, stage + "/attn_dst/kernel",
            output ? config.quantization.output_attention_dst.at(head) : config.quantization.hidden_attention_dst.at(layer).at(head));
        put_matrix(stage + "/model/seq_kernel", weight);
        put_i64(stage + "/model/attn_src", src_kernel);
        put_i64(stage + "/model/attn_dst", dst_kernel);
        if (output) {
            put_i64(stage + "/model/output_bias",
                    quantized_tensor_vector(model, stage + "/output_bias", config.quantization.output_bias.at(head)));
        }

        const auto put_common = [&](const auto& head_trace, const Matrix& projected,
                                    const std::string& name) {
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
                    if (!selected && head_trace.leaky_relu.at(static_cast<std::size_t>(edge)) ==
                                       head_trace.maximum.at(static_cast<std::size_t>(node))) {
                        max_selector[edge] = 1;
                        selected = true;
                    }
                }
                if (!selected) fail("max selector has no selected edge for destination group");
            }
            put_i64(stage + "/trace/max_selector", max_selector);
            put(stage + "/trace/range_bits", bits_for(head_trace.delta, config.range_bits, stage + ".delta"));
            put_i64(stage + "/trace/lrelu_idx", head_trace.score);
            put_i64(stage + "/trace/exp_idx", head_trace.delta);
            put(stage + "/trace/lrelu_bits", lookup_bits(head_trace.score, tables.lrelu_min,
                                                            tables.lrelu_max, config.range_bits, stage + ".lrelu"));
            put(stage + "/trace/exp_bits", lookup_bits(head_trace.delta, 0,
                                                         static_cast<i64>(tables.exp_max), config.range_bits, stage + ".exp"));
            put(stage + "/trace/rounddiv_aux", round_aux(head_trace.aggregate_round_div));
            put_field(stage + "/trace/rounddiv_a_inverse", head_trace.aggregate_round_div.a_inverse);
            put_field(stage + "/trace/rounddiv_divisor_inverse", head_trace.aggregate_round_div.divisor_inverse);
            put_matrix(stage + "/trace/aggregate_num", head_trace.aggregate_num);
            put_field_matrix(stage + "/trace/aggregate_pre_field", head_trace.aggregate_pre_field);
            put_field_matrix(stage + "/trace/consistency_residual_field", head_trace.consistency_residual_field);
        };

        if (!output) {
            const auto& head_trace = trace.hidden_layers.at(layer).heads.at(head);
            put_common(head_trace, head_trace.h_prime, "h_prime");
            put_matrix(stage + "/trace/routed_h_prime", head_trace.routed_h_prime);
            put_matrix(stage + "/trace/product_u_h_prime", head_trace.product_u_h_prime);
            put_field_matrix(stage + "/trace/product_alpha_h_prime_field", head_trace.product_alpha_h_prime_field);
            put_matrix(stage + "/trace/aggregate_pre", head_trace.aggregate_pre);
            put_matrix(stage + "/trace/elu_input", head_trace.elu_input);
            put_matrix(stage + "/trace/aggregate", head_trace.aggregate);
            put_i64(stage + "/trace/elu_idx", head_trace.aggregate_pre.values.empty()
                    ? std::vector<i64>(graph.nodes, 0)
                    : head_trace.aggregate_pre.values);
            put(stage + "/trace/elu_bits", lookup_bits(head_trace.aggregate_pre.values, tables.elu_min,
                                                         tables.elu_max, config.range_bits, stage + ".elu"));
        } else {
            const auto& head_trace = trace.output_heads.at(head);
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
        const auto& concat = trace.hidden_layers.at(layer).concat;
        const u64 head_count = config.architecture.hidden_heads.at(layer);
        const u64 width = config.architecture.hidden_dims.at(layer);
        std::vector<NativeFrWire> values;
        values.reserve(static_cast<std::size_t>(graph.nodes * head_count * width));
        for (u64 node = 0U; node < graph.nodes; ++node)
            for (u64 head = 0U; head < head_count; ++head)
                for (u64 column = 0U; column < width; ++column)
                    values.push_back(native_wire(concat.at(node, head * width + column)));
        put("hidden/layer" + std::to_string(layer) + "/concat", std::move(values));
    }
    for (u64 head = 0U; head < config.architecture.output_heads; ++head)
        put_stage_base("output/head" + std::to_string(head), config.architecture.hidden_dims.size(), head, true);
    std::vector<NativeFrWire> output_heads;
    output_heads.reserve(static_cast<std::size_t>(graph.nodes * config.architecture.output_heads * config.architecture.classes));
    for (u64 node = 0U; node < graph.nodes; ++node)
        for (const auto& head : trace.output_heads)
            for (u64 cls = 0U; cls < config.architecture.classes; ++cls)
                output_heads.push_back(native_wire(head.output.at(node, cls)));
    put("output_heads", std::move(output_heads));
    put_matrix("output_sum", trace.output_sum);
    put_matrix("public_Y", trace.logits);

    const auto put_round_views = [&](const std::string& operation,
                                     const SignedRoundDivWitness& witness,
                                     const Matrix& numerator, u64 numerator_scale,
                                     const Matrix& quotient) {
        if (witness.rows != numerator.rows || witness.cols != numerator.cols ||
            quotient.rows != numerator.rows || quotient.cols != numerator.cols) {
            fail("SignedRoundDiv view shape does not match its canonical matrices: " + operation);
        }
        const std::size_t count = numerator.values.size();
        if (witness.sigma.size() != count || witness.z.size() != count || witness.a.size() != count ||
            witness.q.size() != count || witness.r.size() != count || witness.delta.size() != count ||
            witness.divisor.size() != count || witness.a_inverse.size() != count ||
            witness.divisor_inverse.size() != count) {
            fail("SignedRoundDiv view witness is incomplete: " + operation);
        }
        std::vector<i64> signed_numerator;
        signed_numerator.reserve(count);
        for (const auto value : numerator.values)
            signed_numerator.push_back(checked_i128(static_cast<__int128>(value) * numerator_scale,
                                                    operation + ".signed_numerator"));
        put_i64(operation + "/view/signed_numerator", signed_numerator);
        put_i64(operation + "/view/divisor", witness.divisor);
        put_matrix(operation + "/view/signed_quotient", quotient);
        put_i64(operation + "/view/a", witness.a);
        put_i64(operation + "/view/q_abs", witness.q);
        put_i64(operation + "/view/r", witness.r);
        put_i64(operation + "/view/delta", witness.delta);
        put_i64(operation + "/view/sigma", std::vector<i64>(witness.sigma.begin(), witness.sigma.end()));
        put_i64(operation + "/view/z", std::vector<i64>(witness.z.begin(), witness.z.end()));
        put_field(operation + "/view/a_inverse", witness.a_inverse);
        put_field(operation + "/view/divisor_inverse", witness.divisor_inverse);
        for (u64 bit = 0U; bit < config.range_bits; ++bit) {
            put(operation + "/view/range/a/" + std::to_string(bit), native_bit_column(witness.a, bit, operation + ".a"));
            put(operation + "/view/range/q_abs/" + std::to_string(bit), native_bit_column(witness.q, bit, operation + ".q"));
            put(operation + "/view/range/r/" + std::to_string(bit), native_bit_column(witness.r, bit, operation + ".r"));
            put(operation + "/view/range/delta/" + std::to_string(bit), native_bit_column(witness.delta, bit, operation + ".delta"));
            put(operation + "/view/range/divisor/" + std::to_string(bit), native_bit_column(witness.divisor, bit, operation + ".divisor"));
        }
    };

    const auto put_group_views = [&](const std::string& stage, u64 width, bool output,
                                     u64 aggregate_scale, const auto& head_trace) {
        using HeadTrace = std::decay_t<decltype(head_trace)>;
        constexpr bool is_hidden = std::is_same_v<HeadTrace, HiddenHeadTrace>;
        const auto edge_width = native_edge_feature_from_scalar(head_trace.exp_weight, width);
        const auto pu = [&]() {
            if constexpr (is_hidden) {
                return native_matrix_values(head_trace.product_u_h_prime);
            } else {
                return native_matrix_values(head_trace.product_u_y_prime);
            }
        }();
        const auto group_new_width = edge_group_bits(width);
        const auto group_end_width = edge_group_end_bits(width);
        const auto denominator_nodes = node_vector_from_edges(head_trace.denominator);
        const auto denominator_edges = native_broadcast_node_vector(graph, denominator_nodes);
        const auto denominator_broadcast = [&]() {
            std::vector<NativeFrWire> result;
            result.reserve(static_cast<std::size_t>(graph.edges * width));
            for (const auto value : denominator_edges)
                for (u64 column = 0U; column < width; ++column) result.push_back(value);
            return result;
        }();
        const std::string group = stage + "/group_sum";
        put(group + "/view/group_new", group_new_width);
        put(group + "/view/group_end", group_end_width);
        put(group + "/view/input0", edge_width);
        put(group + "/view/input1", pu);
        put(group + "/view/output0", denominator_broadcast);
        put(group + "/view/output1", native_broadcast_node_matrix(graph, head_trace.aggregate_num));
        put(group + "/view/prefix0", native_prefix(edge_width, group_new, width, false));
        put(group + "/view/prefix_prev0", native_prefix(edge_width, group_new, width, true));
        put(group + "/view/prefix1", native_prefix(pu, group_new, width, false));
        put(group + "/view/prefix_prev1", native_prefix(pu, group_new, width, true));

        const std::string inverse = stage + "/inverse_alpha";
        put(inverse + "/view/input0", denominator_edges);
        put(inverse + "/view/input1", native_i64_values(head_trace.exp_weight));
        put(inverse + "/view/output0", native_field_values(head_trace.inverse_field));
        put(inverse + "/view/output1", native_field_values(head_trace.alpha_field));

        const auto put_segmented = [&](const std::string& operation, const std::vector<NativeFrWire>& input,
                                       const std::vector<NativeFrWire>& output_values) {
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
        put_segmented(stage + "/field_aggregate", field_product,
                      native_broadcast_node_field_matrix(graph, head_trace.aggregate_pre_field));
        put_segmented(stage + "/integer_numerator", pu,
                      native_broadcast_node_matrix(graph, head_trace.aggregate_num));

        const std::string consistency = stage + "/consistency";
        const auto denominator_matrix = scalar_matrix(denominator_nodes, width);
        put(consistency + "/view/input0", native_matrix_values(denominator_matrix));
        put(consistency + "/view/input1", native_field_matrix_values(head_trace.aggregate_pre_field));
        put(consistency + "/view/input2", native_matrix_values(head_trace.aggregate_num));
        put(consistency + "/view/output0", native_field_matrix_values(head_trace.consistency_residual_field));

        const auto put_max_views = [&]() {
            const std::string max = stage + "/max_range";
            put_i64(max + "/view/value", head_trace.leaky_relu);
            put(max + "/view/maximum_broadcast", native_broadcast_node_vector(graph, head_trace.maximum));
            std::vector<i64> max_selector(graph.edges, 0);
            for (u64 node = 0U; node < graph.nodes; ++node) {
                bool selected = false;
                for (u64 edge = graph.node_edge_ptr.at(static_cast<std::size_t>(node));
                     edge < graph.node_edge_ptr.at(static_cast<std::size_t>(node + 1U)); ++edge) {
                    if (!selected && head_trace.leaky_relu.at(static_cast<std::size_t>(edge)) ==
                                       head_trace.maximum.at(static_cast<std::size_t>(node))) {
                        max_selector[edge] = 1; selected = true;
                    }
                }
            }
            put_i64(max + "/view/selector", max_selector);
            put(max + "/view/delta", native_i64_values(head_trace.delta));
            put(max + "/view/group_new", native_i64_values(std::vector<i64>(group_new.begin(), group_new.end())));
            put(max + "/view/group_end", native_i64_values(std::vector<i64>(group_end.begin(), group_end.end())));
            put_i64(max + "/view/selector_prefix", [&]() {
                std::vector<i64> prefix(graph.edges, 0);
                i64 running = 0;
                for (u64 edge = 0U; edge < graph.edges; ++edge) {
                    if (group_new[edge]) running = 0;
                    running += max_selector[edge]; prefix[edge] = running;
                }
                return prefix;
            }());
            put_i64(max + "/view/selector_prev", [&]() {
                std::vector<i64> previous(graph.edges, 0);
                i64 running = 0;
                for (u64 edge = 0U; edge < graph.edges; ++edge) {
                    previous[edge] = group_new[edge] ? 0 : running;
                    running = previous[edge] + max_selector[edge];
                }
                return previous;
            }());
            for (u64 bit = 0U; bit < config.range_bits; ++bit)
                put(max + "/view/range_bit/" + std::to_string(bit),
                    native_bit_column(head_trace.delta, bit, max + ".range" + std::to_string(bit)));
        };
        put_max_views();

        const auto put_table = [&](const std::string& operation, const std::vector<i64>& query,
                                   i64 minimum, i64 maximum, const std::vector<i64>& output_values,
                                   const std::vector<i64>& table) {
            put_i64(operation + "/view/query", query);
            put_i64(operation + "/view/index", query);
            put_i64(operation + "/view/output0", output_values);
            u64 depth = 0U;
            std::size_t table_width = 1U;
            while (table_width < table.size()) { table_width <<= 1U; ++depth; }
            std::vector<i64> offsets;
            offsets.reserve(query.size());
            for (const auto value : query) {
                if (value < minimum || value > maximum)
                    fail("lookup view query outside authenticated table: " + operation);
                offsets.push_back(checked_i128(static_cast<__int128>(value) - minimum,
                                               operation + ".offset"));
            }
            for (u64 bit = 0U; bit < depth; ++bit)
                put(operation + "/view/bit/" + std::to_string(bit),
                    native_bit_column(offsets, bit, operation + ".bit"));
            for (u64 level = 0U; level <= depth; ++level) {
                const u64 level_width = 1ULL << level;
                for (u64 node = 0U; node < level_width; ++node) {
                    std::vector<NativeFrWire> chi;
                    chi.reserve(offsets.size());
                    const u64 mask = level == 0U ? 0U : ((1ULL << level) - 1U);
                    for (const auto offset : offsets)
                        chi.push_back(native_wire(level == 0U || (static_cast<u64>(offset) & mask) == node ? 1 : 0));
                    put(operation + "/view/chi/" + std::to_string(level) + "/" + std::to_string(node),
                        std::move(chi));
                }
            }
        };
        put_table(stage + "/leaky_relu", head_trace.score, tables.lrelu_min, tables.lrelu_max,
                  head_trace.leaky_relu, tables.lrelu);
        put_table(stage + "/exp_map", head_trace.delta, 0, static_cast<i64>(tables.exp_max),
                  head_trace.exp_weight, tables.exp);
        if constexpr (is_hidden)
            put_table(stage + "/elu", head_trace.aggregate_pre.values, tables.elu_min, tables.elu_max,
                      head_trace.aggregate.values, tables.elu);

        const std::string inner = stage + (output ? "/output_attention_inner_product" : "/attention_inner_product");
        const Matrix& projected = [&]() -> const Matrix& {
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
        const Matrix& rounded = [&]() -> const Matrix& {
            if constexpr (is_hidden) {
                return head_trace.aggregate_pre;
            } else {
                return head_trace.y_quantized;
            }
        }();
        put_round_views(round, head_trace.aggregate_round_div, head_trace.aggregate_num,
                        aggregate_scale, rounded);
        if constexpr (!is_hidden) {
            put_round_views(stage + "/output_scale_alignment", head_trace.alignment_round_div,
                            head_trace.y_quantized, config.quantization.output_head, head_trace.aligned);
            const std::string bias = stage + "/output_bias";
            put(bias + "/view/input0", native_matrix_values(head_trace.aligned));
            std::vector<NativeFrWire> bias_values;
            bias_values.reserve(static_cast<std::size_t>(graph.nodes * width));
            for (u64 node = 0U; node < graph.nodes; ++node)
                for (u64 column = 0U; column < width; ++column)
                    bias_values.push_back(native_wire(head_trace.bias.at(static_cast<std::size_t>(column))));
            put(bias + "/view/input1", std::move(bias_values));
            put(bias + "/view/output0", native_matrix_values(head_trace.output));
        }
    };

    for (u64 layer = 0U; layer < config.architecture.hidden_dims.size(); ++layer) {
        const u64 width = config.architecture.hidden_dims.at(layer);
        const u64 aggregate_scale = config.quantization.hidden_aggregate_pre.at(layer);
        for (u64 head = 0U; head < config.architecture.hidden_heads.at(layer); ++head) {
            const std::string stage = "hidden/layer" + std::to_string(layer) + "/head" + std::to_string(head);
            put_group_views(stage, width, false, aggregate_scale,
                            trace.hidden_layers.at(layer).heads.at(head));
        }
        const std::string concat = "hidden/layer" + std::to_string(layer) + "/concat";
        const u64 head_count = config.architecture.hidden_heads.at(layer);
        const Matrix& concat_matrix = trace.hidden_layers.at(layer).concat;
        std::vector<NativeFrWire> concat_output = native_matrix_values(concat_matrix);
        std::vector<NativeFrWire> expected_concat(
            static_cast<std::size_t>(graph.nodes * head_count * width));
        for (u64 head = 0U; head < head_count; ++head) {
            std::vector<NativeFrWire> input(expected_concat.size());
            const Matrix& source = trace.hidden_layers.at(layer).heads.at(head).aggregate;
            if (source.rows != graph.nodes || source.cols != width)
                fail("hidden concat source shape mismatch");
            for (u64 node = 0U; node < graph.nodes; ++node)
                for (u64 column = 0U; column < width; ++column)
                    input.at(static_cast<std::size_t>((node * head_count + head) * width + column)) =
                        native_wire(source.at(node, column));
            put(concat + "/view/input" + std::to_string(head), std::move(input));
        }
        put(concat + "/view/output", std::move(concat_output));
    }
    for (u64 head = 0U; head < config.architecture.output_heads; ++head) {
        const std::string stage = "output/head" + std::to_string(head);
        put_group_views(stage, config.architecture.classes, true, config.quantization.output_pre,
                        trace.output_heads.at(head));
    }
    put_round_views("final_average", trace.final_average_round_div,
                    trace.output_sum, 1U, trace.logits);

    NativeRelationWitnessInput result;
    result.tensors.reserve(plan.tensor_families.size());
    result.tensor_shapes.reserve(plan.tensor_families.size());
    for (const auto& family : plan.tensor_families) {
        const auto found = logical.find(family.id);
        if (found == logical.end())
            fail("canonical native witness is missing tensor values: " + family.id);
        NativeTensorWitnessInput tensor;
        tensor.id = family.id;
        tensor.visibility = family.visibility == "public" ? 1U :
            (family.id.find("/model/") != std::string::npos ? 2U : 3U);
        tensor.domain_size = static_cast<std::uint32_t>(native_shape_product(family.padded_shape, family.id + ".domain"));
        tensor.values = native_embed_zero_padded(family, found->second);
        result.tensors.push_back(std::move(tensor));

        NativeTensorShapeInput shape;
        shape.id = family.id;
        shape.domain_size = static_cast<std::uint32_t>(native_shape_product(family.padded_shape, family.id + ".domain"));
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

    // Resource identity is metadata only at this seam.  A registry is marked
    // authenticated only after the real baseline-local file is found and its
    // SHA-256 equals the secure configuration; no placeholder registry,
    // generator vector, VK, or SRS data is ever manufactured here.
    result.registry_path = (config.baseline_root / "artifacts" / "verifier_registry.json").string();
    const auto decode_digest = [](const std::string& text, std::array<std::uint8_t, 32>& output) {
        if (!is_digest(text)) return false;
        const auto nibble = [](char value) -> int {
            if (value >= '0' && value <= '9') return value - '0';
            if (value >= 'a' && value <= 'f') return value - 'a' + 10;
            if (value >= 'A' && value <= 'F') return value - 'A' + 10;
            return -1;
        };
        for (std::size_t index = 0U; index < output.size(); ++index) {
            const int high = nibble(text[index * 2U]);
            const int low = nibble(text[index * 2U + 1U]);
            if (high < 0 || low < 0) return false;
            output[index] = static_cast<std::uint8_t>((high << 4) | low);
        }
        return true;
    };
    if (!decode_digest(config.verifier_registry_digest, result.registry_digest))
        fail("secure verifier registry identity is not a canonical SHA-256");
    const std::filesystem::path registry(result.registry_path);
    if (!std::filesystem::is_symlink(registry) && std::filesystem::is_regular_file(registry) &&
        sha256_file(registry) == config.verifier_registry_digest) {
        result.registry_authenticated = true;
    }
    const std::filesystem::path srs_manifest = std::filesystem::path(config.srs_path) / "srs_manifest.json";
    if (!std::filesystem::is_symlink(srs_manifest) && std::filesystem::is_regular_file(srs_manifest)) {
        decode_digest(sha256_file(srs_manifest), result.srs_digest);
    }
    // The restored tree has no canonical compressed-G1 loader.  Keep the
    // generator domain empty and authentication false; native resource
    // admission will return the explicit SRS BLOCKED status after this
    // witness has been fully materialized.
    result.srs_authenticated = false;
    result.generators.clear();
    result.registered_model_commitments.clear();
    result.operations.clear();
    return result;
}

NativeRelationWitnessInput build_native_public_context(
    const Config& config, const RelationPlan& plan, const Matrix& public_output) {
    NativeRelationWitnessInput result;
    result.tensor_shapes.reserve(plan.tensor_families.size());
    const auto decode_digest = [](const std::string& text, std::array<std::uint8_t, 32>& output) {
        if (!is_digest(text)) return false;
        const auto nibble = [](char value) -> int {
            if (value >= '0' && value <= '9') return value - '0';
            if (value >= 'a' && value <= 'f') return value - 'a' + 10;
            if (value >= 'A' && value <= 'F') return value - 'A' + 10;
            return -1;
        };
        for (std::size_t index = 0U; index < output.size(); ++index) {
            const int high = nibble(text[index * 2U]);
            const int low = nibble(text[index * 2U + 1U]);
            if (high < 0 || low < 0) return false;
            output[index] = static_cast<std::uint8_t>((high << 4) | low);
        }
        return true;
    };
    for (const auto& family : plan.tensor_families) {
        NativeTensorShapeInput shape;
        shape.id = family.id;
        shape.domain_size = static_cast<std::uint32_t>(native_shape_product(family.padded_shape, family.id + ".domain"));
        shape.visibility = family.visibility == "public" ? 1U :
            (family.id.find("/model/") != std::string::npos ? 2U : 3U);
        shape.logical_shape = family.logical_shape;
        shape.padded_shape = family.padded_shape;
        shape.mle_order = family.mle_order;
        shape.stack_axis = family.stack_axis;
        shape.mapping = family.mapping;
        shape.padding_rule = family.padding_rule;
        if (shape.visibility == 2U) shape.registry_id = "model_registry:" + family.id;
        result.tensor_shapes.push_back(std::move(shape));
    }
    const auto public_family = std::find_if(
        plan.tensor_families.begin(), plan.tensor_families.end(),
        [](const TensorFamily& family) { return family.id == "public_Y"; });
    if (public_family == plan.tensor_families.end()) fail("RelationPlan has no public_Y family");
    if (public_output.rows != public_family->logical_shape.at(0) ||
        public_output.cols != public_family->logical_shape.at(1))
        fail("independent verifier public_Y shape does not match RelationPlan");
    NativeTensorWitnessInput tensor;
    tensor.id = "public_Y";
    tensor.visibility = 1U;
    tensor.domain_size = static_cast<std::uint32_t>(native_shape_product(
        public_family->padded_shape, "public_Y.domain"));
    tensor.values = native_embed_zero_padded(*public_family, native_matrix_values(public_output));
    result.tensors.push_back(std::move(tensor));
    result.registry_path = (config.baseline_root / "artifacts" / "verifier_registry.json").string();
    if (!decode_digest(config.verifier_registry_digest, result.registry_digest))
        fail("secure verifier registry identity is not a canonical SHA-256");
    const std::filesystem::path registry(result.registry_path);
    if (!std::filesystem::is_symlink(registry) && std::filesystem::is_regular_file(registry) &&
        sha256_file(registry) == config.verifier_registry_digest) {
        result.registry_authenticated = true;
    }
    result.srs_authenticated = false;
    result.generators.clear();
    result.registered_model_commitments.clear();
    result.operations.clear();
    return result;
}

GatTrace execute_cpu(const Config& config, const Graph& graph, const ModelBundle& model, const StaticTables& tables) {
    return run_forward(config, graph, model, tables, false);
}

GatTrace execute_gpu(const Config& config, const Graph& graph, const ModelBundle& model,
                     const StaticTables& tables, bool* kernel_executed) {
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

std::string statement_canonical_payload(const PublicStatement& statement) {
    std::ostringstream canonical;
    canonical << "schema=zkDL_gat_statement_v1\n" << "dataset=" << statement.dataset << "\n"
              << "model_arch_id=" << statement.model_arch_id << "\n" << "model_param_id=" << statement.model_param_id << "\n"
              << "quant_cfg_id=" << statement.quant_cfg_id << "\n" << "static_table_id=" << statement.static_table_id << "\n"
              << "input_preprocess_id=" << statement.input_preprocess_id << "\n" << "topology_id=" << statement.topology_id << "\n"
              << "feature_id=" << statement.feature_id << "\n" << "output_id=" << statement.output_id << "\n"
              << "public_output_path=" << statement.public_output_path << "\npublic_output_digest=" << statement.public_output_digest << "\n"
              << "relation_registry_digest=" << statement.relation_registry_digest << "\n"
              << "relation_terminal_claims_digest=" << statement.relation_terminal_claims_digest << "\n"
              << "relation_mapping_digest=" << statement.relation_mapping_digest << "\n"
              << "relation_selector_digest=" << statement.relation_selector_digest << "\n"
              << "relation_degree_digest=" << statement.relation_degree_digest << "\n"
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
              << "total_nodes=" << statement.total_nodes << "\n" << "nodes=" << statement.nodes << "\n"
              << "edges=" << statement.edges << "\n" << "graphs=" << statement.graphs << "\n"
              << "classes=" << statement.classes << "\n"
              << "hidden_layer_count=" << statement.hidden_layer_count << "\nhidden_head_count=" << statement.hidden_head_count << "\n"
              << "output_head_count=" << statement.output_head_count << "\n"
              << "proof_scope=" << statement.proof_scope << "\n" << "verifier_reads_witness=0\n"
              << "privacy_policy=" << statement.privacy_policy << "\n";
    return canonical.str();
}

std::string statement_identity(const PublicStatement& statement) {
    return sha256_text(statement_canonical_payload(statement));
}

std::vector<std::uint8_t> native_encode_public_field_vector(
    const std::vector<NativeFrWire>& values) {
    std::vector<std::uint8_t> encoded;
    encoded.reserve(8U + values.size() * NativeFrWire{}.size() * sizeof(std::uint32_t));
    const auto append_u64 = [&](std::uint64_t value) {
        for (unsigned index = 0U; index < 8U; ++index)
            encoded.push_back(static_cast<std::uint8_t>(value >> (8U * index)));
    };
    const auto append_u32 = [&](std::uint32_t value) {
        for (unsigned index = 0U; index < 4U; ++index)
            encoded.push_back(static_cast<std::uint8_t>(value >> (8U * index)));
    };
    append_u64(static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values)
        for (const auto limb : value) append_u32(limb);
    return encoded;
}

std::string canonical_public_field_digest(const Matrix& matrix) {
    const auto encoded = native_encode_public_field_vector(native_matrix_values(matrix));
    return sha256_bytes(encoded.data(), encoded.size());
}

PublicStatement make_statement(const Config& config, const Graph& graph, const ModelBundle& model,
                               const StaticTables& tables, const RelationPlan& relation,
                               const GatTrace& trace) {
    PublicStatement statement;
    statement.dataset = config.dataset; statement.model_arch_id = config.model_arch_id;
    statement.model_param_id = config.model_param_id; statement.quant_cfg_id = config.quant_cfg_id;
    statement.static_table_id = config.static_table_id; statement.input_preprocess_id = config.input_preprocess_id;
    statement.topology_id = digest_graph(graph); statement.feature_id = digest_matrix(graph.feature_values);
    statement.output_id = digest_matrix(trace.logits);
    statement.public_output_digest = canonical_public_field_digest(trace.logits);
    statement.relation_registry_digest = relation.registry_digest;
    statement.relation_terminal_claims_digest = relation.terminal_claims_digest;
    statement.relation_mapping_digest = relation.mapping_digest;
    statement.relation_selector_digest = relation.selector_digest;
    statement.relation_degree_digest = relation.degree_digest;
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
    statement.total_nodes = graph.total_nodes; statement.nodes = graph.nodes; statement.edges = graph.edges;
    statement.graphs = graph.graphs; statement.classes = config.architecture.classes;
    statement.hidden_layer_count = config.architecture.hidden_dims.size();
    statement.hidden_head_count = std::accumulate(config.architecture.hidden_heads.begin(), config.architecture.hidden_heads.end(), u64{0});
    statement.output_head_count = config.architecture.output_heads;
    statement.privacy_policy = config.privacy_policy; statement.verifier_reads_witness = false;
    statement.statement_id = statement_identity(statement);
    return statement;
}

void write_statement(const std::filesystem::path& path, const PublicStatement& statement) {
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
           << "static_table_generator_id=" << statement.static_table_generator_id << "\nstatic_table_version=" << statement.static_table_version << "\n"
           << "static_table_scale_id=" << statement.static_table_scale_id << "\nstatic_table_domain_id=" << statement.static_table_domain_id << "\n"
           << "static_table_row_order=" << statement.static_table_row_order << "\nstatic_table_spec_hash=" << statement.static_table_spec_hash << "\n"
           << "static_table_digest=" << statement.static_table_digest << "\nstatic_table_descriptor_digest=" << statement.static_table_descriptor_digest << "\n"
           << "scale_chain_digest=" << statement.scale_chain_digest << "\n"
           << "integer_bound_ledger_id=" << statement.integer_bound_ledger_id << "\ninteger_bound_ledger_digest=" << statement.integer_bound_ledger_digest << "\n"
           << "integer_bound_ledger_descriptor_digest=" << statement.integer_bound_ledger_descriptor_digest << "\nconfig_digest=" << statement.config_digest << "\n"
           << "asset_manifest_digest=" << statement.asset_manifest_digest << "\ncheckpoint_manifest_digest=" << statement.checkpoint_manifest_digest << "\n"
           << "checkpoint_tensors_txt_digest=" << statement.checkpoint_tensors_txt_digest << "\ncheckpoint_tensors_npz_digest=" << statement.checkpoint_tensors_npz_digest << "\n"
           << "verifier_registry_digest=" << statement.verifier_registry_digest << "\ntopology_boundary_digest=" << statement.topology_boundary_digest << "\n"
           << "total_nodes=" << statement.total_nodes << "\nnodes=" << statement.nodes << "\nedges=" << statement.edges << "\n"
           << "graphs=" << statement.graphs << "\nclasses=" << statement.classes << "\nproof_scope=" << statement.proof_scope << "\n"
           << "hidden_layer_count=" << statement.hidden_layer_count << "\nhidden_head_count=" << statement.hidden_head_count << "\n"
           << "output_head_count=" << statement.output_head_count << "\n"
           << "privacy_policy=" << statement.privacy_policy << "\nverifier_reads_witness=" << (statement.verifier_reads_witness ? 1 : 0) << "\n";
}

bool read_statement(const std::filesystem::path& path, PublicStatement* statement, std::string* error) {
    try {
        if (!statement) fail("read_statement received null output");
        const auto values = read_kv(path);
        const auto get = [&](const std::string& key) -> std::string {
            const auto iterator = values.find(key); if (iterator == values.end()) fail("statement field missing: " + key); return iterator->second;
        };
        if (get("schema") != "zkDL_gat_statement_v1") fail("statement schema mismatch");
        statement->statement_id = get("statement_id"); statement->dataset = get("dataset");
        statement->model_arch_id = get("model_arch_id"); statement->model_param_id = get("model_param_id");
        statement->quant_cfg_id = get("quant_cfg_id"); statement->static_table_id = get("static_table_id");
        statement->input_preprocess_id = get("input_preprocess_id"); statement->topology_id = get("topology_id");
        statement->feature_id = get("feature_id"); statement->output_id = get("output_id");
        statement->public_output_path = get("public_output_path");
        statement->public_output_digest = get("public_output_digest");
        statement->relation_registry_digest = get("relation_registry_digest");
        statement->relation_terminal_claims_digest = get("relation_terminal_claims_digest");
        statement->relation_mapping_digest = get("relation_mapping_digest");
        statement->relation_selector_digest = get("relation_selector_digest");
        statement->relation_degree_digest = get("relation_degree_digest");
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
        statement->nodes = parse_u64(get("nodes"), "statement nodes"); statement->edges = parse_u64(get("edges"), "statement edges");
        statement->graphs = parse_u64(get("graphs"), "statement graphs"); statement->classes = parse_u64(get("classes"), "statement classes");
        statement->hidden_layer_count = parse_u64(get("hidden_layer_count"), "statement hidden_layer_count");
        statement->hidden_head_count = parse_u64(get("hidden_head_count"), "statement hidden_head_count");
        statement->output_head_count = parse_u64(get("output_head_count"), "statement output_head_count");
        statement->proof_scope = get("proof_scope"); statement->privacy_policy = get("privacy_policy");
        statement->verifier_reads_witness = parse_bool(get("verifier_reads_witness"), "statement verifier_reads_witness");
        if (statement->verifier_reads_witness) fail("statement violates witness-independent verifier boundary");
        if (statement->statement_id != statement_identity(*statement)) fail("statement identity mismatch");
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

void write_run_record(const std::filesystem::path& path, const RunRecord& record) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) fail("cannot write run record " + path.string());
    output << "schema=zkDL_gat_run_record_v1\ndataset=" << record.dataset << "\nbackend=" << record.backend << "\n"
           << "statement_id=" << record.statement_id << "\nproof_id=" << record.proof_id << "\nverification_status=" << record.verification_status << "\n"
           << "proof_scope=" << record.proof_scope << "\nstatic_table_descriptor_digest=" << record.static_table_descriptor_digest << "\n"
           << "scale_chain_digest=" << record.scale_chain_digest << "\ninteger_bound_ledger_id=" << record.integer_bound_ledger_id << "\n"
           << "integer_bound_ledger_digest=" << record.integer_bound_ledger_digest << "\n"
           << "integer_bound_ledger_descriptor_digest=" << record.integer_bound_ledger_descriptor_digest << "\n"
           << "proof_size_bytes=" << record.proof_size_bytes << "\ncommitment_size_bytes=" << record.commitment_size_bytes << "\n"
           << "commitment_time_seconds=" << std::setprecision(17) << record.commitment_time_seconds << "\nprove_time_seconds=" << record.prove_time_seconds << "\n"
           << "verify_time_seconds=" << record.verify_time_seconds << "\nverifier_reads_witness=" << (record.verifier_reads_witness ? 1 : 0) << "\n"
           << "gpu_kernel_executed=" << (record.gpu_kernel_executed ? 1 : 0) << "\nsilent_fallback=" << (record.silent_fallback ? 1 : 0) << "\n"
           << "output_bound=" << (record.output_bound ? 1 : 0) << "\nerror=" << record.error << "\n";
}

namespace {

struct CliOptions {
    std::string command;
    std::filesystem::path config;
    std::filesystem::path run_dir;
    std::filesystem::path statement;
    std::filesystem::path proof;
    std::filesystem::path envelope;
    Backend backend = Backend::Cpu;
    bool has_backend = false;
};

void usage() {
    std::cerr << "usage: gat_runner <inspect|prepare|run|prove|verify> --config FILE "
                 "[--backend cpu|gpu] [--run-dir DIR]\n";
}

CliOptions parse_cli(int argc, char** argv) {
    if (argc < 3) { usage(); fail("missing command/config"); }
    CliOptions options;
    options.command = argv[1];
    const std::set<std::string> commands = {"inspect", "prepare", "run", "prove", "verify"};
    if (!commands.count(options.command)) fail("unknown command: " + options.command);
    for (int i = 2; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--config" || argument == "--run-dir" || argument == "--statement" ||
            argument == "--proof" || argument == "--envelope" || argument == "--backend") {
            if (++i >= argc) fail("missing value for " + argument);
            const std::string value = argv[i];
            if (argument == "--config") options.config = value;
            else if (argument == "--run-dir") options.run_dir = value;
            else if (argument == "--statement") options.statement = value;
            else if (argument == "--proof") options.proof = value;
            else if (argument == "--envelope") options.envelope = value;
            else { options.backend = parse_backend(value); options.has_backend = true; }
        } else {
            fail("unknown argument: " + argument);
        }
    }
    if (options.config.empty()) fail("--config is required");
    if ((options.command == "run" || options.command == "prove") && !options.has_backend) {
        fail("--backend is required for " + options.command);
    }
    return options;
}

std::filesystem::path run_path(const Config& config, const std::filesystem::path& raw) {
    if (raw.empty()) return config.run_root;
    if (raw.is_absolute()) {
        const auto base = config.baseline_root.lexically_normal().generic_string();
        const auto candidate = raw.lexically_normal().generic_string();
        if (candidate != base && candidate.rfind(base + "/", 0) != 0) fail("run directory escapes baseline root");
        return raw.lexically_normal();
    }
    return local_path(config.baseline_root, raw.generic_string(), "run_dir");
}

Config config_for_run(const Config& config, const std::filesystem::path& directory) {
    Config result = config;
    result.run_root = directory;
    result.asset_manifest_path = directory / "asset_manifest.json";
    return result;
}

std::filesystem::path run_file_path(const std::filesystem::path& directory,
                                    const std::filesystem::path& raw,
                                    const std::string& name) {
    const auto expected = (directory / name).lexically_normal();
    const auto candidate = raw.empty()
        ? expected
        : (raw.is_absolute() ? raw.lexically_normal()
                             : local_path(directory, raw.generic_string(), name));
    if (candidate != expected) {
        fail(name + " must be the run-local " + expected.string());
    }
    return candidate;
}

void write_matrix_text(const std::filesystem::path& path, const Matrix& matrix) {
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
}

Matrix read_public_output(const std::filesystem::path& path, std::uint64_t expected_rows,
                          std::uint64_t expected_cols) {
    if (path.empty() || std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path)) {
        fail("public output is missing or not a regular baseline-local file: " + path.string());
    }
    std::ifstream input(path);
    if (!input) fail("cannot open public output " + path.string());
    u64 rows = 0, cols = 0;
    input >> rows >> cols;
    if (!input || rows != expected_rows || cols != expected_cols) fail("public output shape mismatch");
    Matrix output = make_matrix(rows, cols);
    for (auto& value : output.values) {
        input >> value;
        if (!input) fail("truncated public output");
    }
    std::string trailing;
    if (input >> trailing) fail("public output contains trailing data");
    return output;
}

void write_plan(const std::filesystem::path& path, const RelationPlan& plan, const StaticTables& tables) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) fail("cannot write relation plan " + path.string());
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
           << "table_domain_id=" << tables.domain_id << "\ntable_row_order=" << tables.row_order << "\n";
    for (const auto& family : plan.tensor_families) {
        output << "tensor=" << family.id << "|" << family.visibility << "|" << family.mle_order << "|" << family.stack_axis << "|";
        for (std::size_t i = 0; i < family.logical_shape.size(); ++i) { if (i) output << ','; output << family.logical_shape[i]; }
        output << '|';
        for (std::size_t i = 0; i < family.padded_shape.size(); ++i) { if (i) output << ','; output << family.padded_shape[i]; }
        output << '|';
        for (std::size_t i = 0; i < family.instances.size(); ++i) { if (i) output << ','; output << family.instances[i]; }
        output << '|' << family.instance_order << '|' << family.mapping << '|' << family.zero_selector << '|'
               << family.selector_id << '|' << family.padding_rule << '|' << family.terminal_point << '|'
               << family.terminal_value << '|' << family.terminal_claim << "|degree=" << family.degree_bound << '\n';
    }
    for (const auto& operation : plan.operation_families) {
        output << "operation=" << operation.id << '|' << operation.kind << '|' << operation.degree_bound << "|degree=" << operation.degree << '|'
               << (operation.requires_reindex ? 1 : 0) << '|' << (operation.public_route ? 1 : 0) << '|';
        for (std::size_t i = 0; i < operation.inputs.size(); ++i) { if (i) output << ','; output << operation.inputs[i]; }
        output << '|';
        for (std::size_t i = 0; i < operation.outputs.size(); ++i) { if (i) output << ','; output << operation.outputs[i]; }
        output << '|';
        for (std::size_t i = 0; i < operation.views.size(); ++i) { if (i) output << ','; output << operation.views[i]; }
        output << '|' << operation.instance_order << '|' << operation.mapping << '|';
        for (std::size_t i = 0; i < operation.auxiliary.size(); ++i) { if (i) output << ','; output << operation.auxiliary[i]; }
        output << '|' << operation.selector_id << '|' << operation.terminal_point << '|' << operation.terminal_value
               << '|' << operation.terminal_claim << '\n';
    }
}

void write_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) fail("cannot write proof bytes " + path.string());
    if (!bytes.empty()) output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) fail("cannot read proof bytes " + path.string());
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (end < 0) fail("cannot size proof bytes");
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    if (!bytes.empty()) input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input && !input.eof()) fail("truncated proof bytes");
    return bytes;
}

void write_envelope(const std::filesystem::path& path, const ProofEnvelope& envelope) {
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
           << "commitment_time_seconds=" << std::setprecision(17) << envelope.commitment_time_seconds << "\nprove_time_seconds=" << envelope.prove_time_seconds << "\n"
           << "verify_time_seconds=" << envelope.verify_time_seconds << "\nverifier_reads_witness=" << (envelope.verifier_reads_witness ? 1 : 0) << "\n"
           << "gpu_kernel_executed=" << (envelope.gpu_kernel_executed ? 1 : 0) << "\nsilent_fallback=" << (envelope.silent_fallback ? 1 : 0) << "\n"
           << "output_bound=" << (envelope.output_bound ? 1 : 0) << "\nerror=" << envelope.error << "\n";
}

ProofEnvelope read_envelope(const std::filesystem::path& path) {
    const auto values = read_kv(path);
    const auto get = [&](const std::string& key) -> std::string {
        const auto iterator = values.find(key); if (iterator == values.end()) fail("proof envelope field missing: " + key); return iterator->second;
    };
    ProofEnvelope envelope;
    envelope.schema_id = get("schema_id"); envelope.statement_id = get("statement_id"); envelope.proof_id = get("proof_id");
    envelope.backend = get("backend"); envelope.proof_scope = get("proof_scope"); envelope.native_core_api = get("native_core_api");
    envelope.status = get("status"); envelope.proof_size_bytes = parse_u64(get("proof_size_bytes"), "proof_size_bytes");
    envelope.static_table_descriptor_digest = get("static_table_descriptor_digest");
    envelope.scale_chain_digest = get("scale_chain_digest");
    envelope.integer_bound_ledger_id = get("integer_bound_ledger_id");
    envelope.integer_bound_ledger_digest = get("integer_bound_ledger_digest");
    envelope.integer_bound_ledger_descriptor_digest = get("integer_bound_ledger_descriptor_digest");
    envelope.commitment_size_bytes = parse_u64(get("commitment_size_bytes"), "commitment_size_bytes");
    envelope.commitment_time_seconds = parse_double(get("commitment_time_seconds"), "commitment_time_seconds");
    envelope.prove_time_seconds = parse_double(get("prove_time_seconds"), "prove_time_seconds");
    envelope.verify_time_seconds = parse_double(get("verify_time_seconds"), "verify_time_seconds");
    envelope.verifier_reads_witness = parse_bool(get("verifier_reads_witness"), "verifier_reads_witness");
    envelope.gpu_kernel_executed = parse_bool(get("gpu_kernel_executed"), "gpu_kernel_executed");
    envelope.silent_fallback = parse_bool(get("silent_fallback"), "silent_fallback");
    envelope.output_bound = parse_bool(get("output_bound"), "output_bound"); envelope.error = get("error");
    return envelope;
}

RunRecord rejected_record(const Config& config, Backend backend, const PublicStatement* statement,
                         const std::string& message, bool kernel_executed) {
    RunRecord record;
    record.dataset = config.dataset; record.backend = backend_name(backend); record.proof_scope = "full_gat_forward";
    record.verification_status = "BLOCKED"; record.error = message; record.gpu_kernel_executed = kernel_executed;
    record.statement_id = statement ? statement->statement_id : std::string(64, '0');
    record.static_table_descriptor_digest = statement ? statement->static_table_descriptor_digest : "UNAVAILABLE";
    record.scale_chain_digest = statement ? statement->scale_chain_digest : config.scale_chain_digest;
    record.integer_bound_ledger_id = statement ? statement->integer_bound_ledger_id : "UNAVAILABLE";
    record.integer_bound_ledger_digest = statement ? statement->integer_bound_ledger_digest : "UNAVAILABLE";
    record.integer_bound_ledger_descriptor_digest = statement ? statement->integer_bound_ledger_descriptor_digest : "UNAVAILABLE";
    record.silent_fallback = false; record.verifier_reads_witness = false; record.output_bound = false;
    return record;
}

void bind_envelope_identity(ProofEnvelope* envelope, const PublicStatement& statement) {
    if (!envelope) fail("proof envelope pointer is null");
    const auto check_or_bind = [](std::string* actual, const std::string& expected,
                                  const std::string& label) {
        if (!actual->empty() && *actual != expected) fail("native proof envelope " + label + " mismatch");
        *actual = expected;
    };
    check_or_bind(&envelope->static_table_descriptor_digest, statement.static_table_descriptor_digest,
                  "static table descriptor identity");
    check_or_bind(&envelope->scale_chain_digest, statement.scale_chain_digest, "scale-chain identity");
    check_or_bind(&envelope->integer_bound_ledger_id, statement.integer_bound_ledger_id,
                  "integer-bound ledger id");
    check_or_bind(&envelope->integer_bound_ledger_digest, statement.integer_bound_ledger_digest,
                  "integer-bound ledger digest");
    check_or_bind(&envelope->integer_bound_ledger_descriptor_digest,
                  statement.integer_bound_ledger_descriptor_digest,
                  "integer-bound ledger descriptor identity");
}

RunRecord record_from_envelope(const Config& config, const PublicStatement& statement,
                              const ProofEnvelope& envelope, const std::string& status,
                              const std::string& error = {}) {
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

int produce(const Config& config, const CliOptions& options) {
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
        // This admission is intentionally before execute_gpu: a rejected
        // integer/field bound must not allocate or launch any CUDA kernel.
        admit_integer_bound_ledger(run_config, graph, model, tables, relation.integer_bound_ledger);
        GatTrace trace;
        try {
            trace = options.backend == Backend::Cpu ? execute_cpu(run_config, graph, model, tables)
                                                    : execute_gpu(run_config, graph, model, tables, &kernel_executed);
        } catch (const std::exception& exception) {
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
        // Materialize every canonical FrWire tensor and operation view before
        // entering the native bridge.  Resource admission (SRS, registry,
        // model VK) remains the native core's responsibility; this runner
        // must not turn an absent seam/resource into a host-only pseudo-proof.
        const auto witness = build_native_relation_witness(
            run_config, graph, model, tables, relation, trace);
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
        // Native code owns all envelope identity and timing fields.  The
        // adapter accepts only an exact statement/backend/provenance binding.
        if (result.envelope.statement_id != statement.statement_id ||
            result.envelope.backend != backend_name(options.backend) ||
            result.envelope.proof_scope != statement.proof_scope ||
            result.envelope.proof_size_bytes != result.proof_bytes.size() ||
            result.envelope.gpu_kernel_executed != kernel_executed) {
            const std::string message = "native proof envelope does not bind statement/backend/proof provenance";
            write_envelope(envelope_path, result.envelope);
            write_run_record(record_path, rejected_record(run_config, options.backend, &statement, message, kernel_executed));
            return 5;
        }
        bind_envelope_identity(&result.envelope, statement);
        if (result.envelope.status != "VERIFIED" || result.envelope.proof_id.empty() ||
            result.proof_bytes.empty() || result.envelope.native_core_api != "zkdl_native_gat_bridge_v1") {
            const std::string message = "native proof envelope is not a complete verified proof artifact";
            write_envelope(envelope_path, result.envelope);
            write_run_record(record_path, rejected_record(run_config, options.backend, &statement, message, kernel_executed));
            return 6;
        }
        write_bytes(proof_path, result.proof_bytes);
        write_envelope(envelope_path, result.envelope);
        const auto record = record_from_envelope(run_config, statement, result.envelope, "NOT_RUN");
        write_run_record(record_path, record);
        return 0;
    } catch (const std::exception& exception) {
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

int prepare(const Config& config, const CliOptions& options) {
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
    std::cout << "prepared dataset=" << config.dataset << " nodes=" << graph.nodes << " edges=" << graph.edges
              << " relation_operations=" << plan.operation_families.size() << "\n";
    return 0;
}

void validate_statement_binding(const Config& config, const Graph& graph, const ModelBundle& model,
                               const StaticTables& tables, const RelationPlan& relation,
                               const PublicStatement& statement) {
    const auto require_equal = [](const std::string& actual, const std::string& expected,
                                  const std::string& label) {
        if (actual != expected) fail("public statement " + label + " does not match authenticated run inputs");
    };
    require_equal(statement.static_table_descriptor_digest, tables.descriptor_digest,
                  "static table descriptor digest");
    require_equal(statement.static_table_digest, tables.digest, "static table digest");
    require_equal(statement.scale_chain_digest, config.scale_chain_digest, "scale-chain digest");
    require_equal(statement.integer_bound_ledger_id, relation.integer_bound_ledger.ledger_id,
                  "integer-bound ledger id");
    require_equal(statement.integer_bound_ledger_digest, relation.integer_bound_ledger.digest,
                  "integer-bound ledger digest");
    require_equal(statement.integer_bound_ledger_descriptor_digest,
                  relation.integer_bound_ledger.descriptor_digest,
                  "integer-bound ledger descriptor digest");
    if (!is_digest(statement.static_table_descriptor_digest) || !is_digest(statement.scale_chain_digest) ||
        !is_digest(statement.integer_bound_ledger_digest) ||
        !is_digest(statement.integer_bound_ledger_descriptor_digest)) {
        fail("public statement relation identities are not canonical SHA-256 digests");
    }
    if (statement.verifier_reads_witness || statement.proof_scope != "full_gat_forward") {
        fail("public statement violates the canonical verifier boundary");
    }

    // Reconstruct every statement field from the authenticated resources.  The
    // output is the only field supplied by the prover and is checked separately
    // against public_Y.txt below; all other fields must have the same identity.
    auto expected = make_statement(config, graph, model, tables, relation, GatTrace{});
    expected.output_id = statement.output_id;
    expected.public_output_digest = statement.public_output_digest;
    expected.statement_id = statement_identity(expected);
    if (statement.statement_id != expected.statement_id) {
        fail("public statement identity is not bound to authenticated run inputs");
    }
    if (statement.asset_manifest_digest != sha256_file(config.asset_manifest_path)) {
        fail("public statement asset manifest digest is stale or unauthenticated");
    }
}

int verify_only(const Config& config, const CliOptions& options) {
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
        if (envelope.schema_id != "zkDL_gat_proof_envelope_v1" || envelope.status != "VERIFIED" ||
            envelope.native_core_api != "zkdl_native_gat_bridge_v1" || envelope.proof_id.empty() ||
            envelope.statement_id != statement.statement_id || envelope.proof_scope != statement.proof_scope ||
            envelope.verifier_reads_witness || envelope.silent_fallback || !envelope.output_bound ||
            envelope.proof_size_bytes == 0) {
            fail("BLOCKED: proof envelope is not a complete independent-verification artifact");
        }
        backend = parse_backend(envelope.backend);
        if (options.has_backend && options.backend != backend) fail("requested backend does not match proof envelope");
        if (envelope.static_table_descriptor_digest != statement.static_table_descriptor_digest ||
            envelope.scale_chain_digest != statement.scale_chain_digest ||
            envelope.integer_bound_ledger_id != statement.integer_bound_ledger_id ||
            envelope.integer_bound_ledger_digest != statement.integer_bound_ledger_digest ||
            envelope.integer_bound_ledger_descriptor_digest != statement.integer_bound_ledger_descriptor_digest) {
            fail("BLOCKED: proof envelope relation identities do not bind the public statement");
        }

        // Reload the manifest, checkpoint, graph, descriptor, ledger, and scale
        // chain in this verifier process.  No producer-side precheck is trusted.
        const auto model = load_model_metadata(run_config);
        const auto graph = load_graph(run_config);
        const auto tables = build_static_tables(run_config, graph);
        const auto relation = build_relation_plan(run_config, graph, tables);
        std::string relation_error;
        if (!relation_plan_complete(relation, &relation_error)) fail(relation_error);
        validate_statement_binding(run_config, graph, model, tables, relation, statement);

        const auto public_output_path = local_path(directory, statement.public_output_path, "public_output_path");
        const auto public_output = read_public_output(public_output_path, statement.nodes, statement.classes);
        if (statement.output_id != digest_matrix(public_output) ||
            canonical_public_field_digest(public_output) != statement.public_output_digest) {
            fail("public output shape or digest does not match the public statement");
        }
        if (backend == Backend::Gpu && !envelope.gpu_kernel_executed) {
            fail("BLOCKED: GPU proof envelope does not attest CUDA execution");
        }
        if (backend == Backend::Cpu && envelope.gpu_kernel_executed) {
            fail("BLOCKED: CPU proof envelope claims CUDA execution");
        }
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
        std::ofstream result(directory / "verification.result");
        if (!result) fail("cannot write verification.result");
        result << "status=VERIFIED\nstatement_id=" << statement.statement_id << "\nproof_id=" << envelope.proof_id << "\n";
        write_run_record(directory / "run.record", record_from_envelope(run_config, statement, envelope, "VERIFIED"));
        std::cout << "verification=VERIFIED statement_id=" << statement.statement_id << "\n";
        return 0;
    } catch (const std::exception& exception) {
        const std::string message = std::string("BLOCKED: independent verification failed: ") + exception.what();
        try {
            const auto record = rejected_record(run_config, backend, statement_loaded ? &statement : nullptr,
                                                message, envelope_loaded && envelope.gpu_kernel_executed);
            write_run_record(directory / "run.record", record);
            std::ofstream result(directory / "verification.result");
            if (result) {
                result << "status=BLOCKED\nstatement_id=" << (statement_loaded ? statement.statement_id : "UNAVAILABLE")
                       << "\nproof_id=" << (envelope_loaded ? envelope.proof_id : "UNAVAILABLE") << "\nerror=" << message << "\n";
            }
        } catch (...) {
            // Preserve the original verification failure.  A missing run
            // directory must not turn a clear BLOCKED result into a parser crash.
        }
        std::cerr << "verification=BLOCKED " << message << '\n';
        return 2;
    }
}

}  // namespace

}  // namespace zkdl_gat

int main(int argc, char** argv) {
    try {
        const auto options = zkdl_gat::parse_cli(argc, argv);
        const auto config = zkdl_gat::load_config(options.config);
        if (options.command == "inspect") {
            std::cout << "dataset=" << config.dataset << " baseline_root=" << config.baseline_root.string()
                      << " checkpoint_bundle=" << config.checkpoint_bundle.string() << " backend="
                      << (options.has_backend ? zkdl_gat::backend_name(options.backend) : "unspecified") << "\n";
            return 0;
        }
        if (options.command == "prepare") return zkdl_gat::prepare(config, options);
        if (options.command == "run" || options.command == "prove") return zkdl_gat::produce(config, options);
        return zkdl_gat::verify_only(config, options);
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
