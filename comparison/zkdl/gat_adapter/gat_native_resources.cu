#include "gat_native_resources.hpp"

#include "../proof.cuh"
#include "../g1-wire.cuh"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace zkdl_gat {
namespace {

using Byte = std::uint8_t;
using Digest = std::array<Byte, 32U>;
constexpr std::size_t kG1Size = 48U;
constexpr std::uint32_t kFrameVersion = 1U;
// The proof context consumes one selected prefix chunk.  There is therefore
// no protocol-wide 2^19 cap: a larger relation-family domain is represented
// by its own authenticated frame (or by a larger selected chunk), while
// unrelated families are never concatenated into one global domain.  The
// byte bound is an operational allocation guard, not a domain truncation.
constexpr std::uint64_t kMaxSrsFrameBytes = 1ULL << 32U;
constexpr std::size_t kMaxJsonResourceBytes = 1U << 26U;
constexpr std::uint64_t kMaxSrsFramePoints = kMaxSrsFrameBytes / kG1Size;
constexpr const char* kUnavailable = "UNAVAILABLE";

void set_error(std::string* error, const std::string& message) {
    if (error) *error = message;
}

bool read_bytes(const std::filesystem::path& path, std::vector<Byte>& bytes,
                std::string* error, std::uint64_t max_bytes = kMaxJsonResourceBytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        set_error(error, "BLOCKED: cannot open authenticated resource: " + path.string());
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) > max_bytes) {
        set_error(error, "BLOCKED: authenticated resource has invalid size: " + path.string());
        return false;
    }
    input.seekg(0, std::ios::beg);
    bytes.resize(static_cast<std::size_t>(size));
    if (!bytes.empty()) input.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!input && !bytes.empty()) {
        set_error(error, "BLOCKED: cannot read authenticated resource: " + path.string());
        return false;
    }
    return true;
}

bool resource_file_digest(const std::filesystem::path& path, Digest& digest,
                          std::string* error) {
    std::vector<Byte> bytes;
    if (!read_bytes(path, bytes, error)) return false;
    digest = zkdl::sha256(bytes);
    return true;
}

int hex_nibble(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

bool resource_parse_digest(const std::string& text, Digest& digest) {
    if (text.size() != 64U) return false;
    for (std::size_t i = 0U; i < digest.size(); ++i) {
        const int high = hex_nibble(text[2U * i]);
        const int low = hex_nibble(text[2U * i + 1U]);
        if (high < 0 || low < 0) return false;
        digest[i] = static_cast<Byte>((high << 4U) | low);
    }
    return true;
}

bool canonical_digest_text(const std::string& text) {
    if (text.size() != 64U) return false;
    return std::all_of(text.begin(), text.end(), [](char value) {
        return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
    });
}

std::string resource_digest_hex(const Digest& digest) {
    static const char* digits = "0123456789abcdef";
    std::string result;
    result.reserve(64U);
    for (const Byte value : digest) {
        result.push_back(digits[(value >> 4U) & 0x0fU]);
        result.push_back(digits[value & 0x0fU]);
    }
    return result;
}

std::string basis_digest_hex(const std::string& g_digest,
                             const std::string& h_digest,
                             const std::string& u_digest) {
    const std::string canonical =
        "{\"G\":\"" + g_digest + "\",\"H\":\"" + h_digest +
        "\",\"U\":\"" + u_digest +
        "\",\"schema\":\"zkDL_gat_basis_digest_v1\"}";
    const std::vector<Byte> bytes(canonical.begin(), canonical.end());
    return resource_digest_hex(zkdl::sha256(bytes));
}

void skip_space(const std::string& text, std::size_t& offset) {
    while (offset < text.size() && std::isspace(static_cast<unsigned char>(text[offset]))) ++offset;
}

bool json_string_at(const std::string& text, std::size_t offset,
                    std::string& value, std::size_t* end = nullptr) {
    skip_space(text, offset);
    if (offset >= text.size() || text[offset] != '"') return false;
    const std::size_t first = ++offset;
    while (offset < text.size() && text[offset] != '"') {
        // Canonical resource JSON does not permit escapes.  Rejecting them
        // avoids accepting two textual identities for one path/digest.
        if (text[offset] == '\\') return false;
        ++offset;
    }
    if (offset >= text.size() || offset == first) return false;
    value = text.substr(first, offset - first);
    if (end) *end = offset + 1U;
    return true;
}

bool json_key_value_offset(const std::string& text, const std::string& key,
                           std::size_t& value_offset) {
    const std::string marker = "\"" + key + "\"";
    const std::size_t key_pos = text.find(marker);
    if (key_pos == std::string::npos) return false;
    // A canonical object cannot contain a duplicate key.  This also stops a
    // nested object from silently overriding its top-level identity field.
    if (text.find(marker, key_pos + marker.size()) != std::string::npos) return false;
    std::size_t colon = key_pos + marker.size();
    skip_space(text, colon);
    if (colon >= text.size() || text[colon] != ':') return false;
    value_offset = colon + 1U;
    skip_space(text, value_offset);
    return value_offset < text.size();
}

bool json_string_field(const std::string& text, const std::string& key,
                       std::string& value) {
    std::size_t offset = 0U;
    if (!json_key_value_offset(text, key, offset)) return false;
    return json_string_at(text, offset, value);
}

bool json_u64_field(const std::string& text, const std::string& key,
                    std::uint64_t& value) {
    std::size_t offset = 0U;
    if (!json_key_value_offset(text, key, offset) ||
        !std::isdigit(static_cast<unsigned char>(text[offset]))) return false;
    const std::size_t first = offset;
    while (offset < text.size() && std::isdigit(static_cast<unsigned char>(text[offset]))) ++offset;
    const std::string digits = text.substr(first, offset - first);
    try {
        std::size_t consumed = 0U;
        value = std::stoull(digits, &consumed, 10);
        if (consumed != digits.size()) return false;
    } catch (...) {
        return false;
    }
    if (offset < text.size() && !std::isspace(static_cast<unsigned char>(text[offset])) &&
        text[offset] != ',' && text[offset] != '}' && text[offset] != ']') return false;
    return true;
}

bool json_array_bounds(const std::string& text, const std::string& key,
                       std::size_t& begin, std::size_t& end) {
    if (!json_key_value_offset(text, key, begin) || text[begin] != '[') return false;
    const std::size_t open = begin;
    bool quoted = false;
    bool escaped = false;
    std::size_t depth = 0U;
    for (std::size_t i = open; i < text.size(); ++i) {
        const char value = text[i];
        if (quoted) {
            if (escaped) escaped = false;
            else if (value == '\\') escaped = true;
            else if (value == '"') quoted = false;
            continue;
        }
        if (value == '"') { quoted = true; continue; }
        if (value == '[') ++depth;
        else if (value == ']') {
            if (depth == 0U) return false;
            --depth;
            if (depth == 0U) { end = i; return true; }
        }
    }
    return false;
}

bool json_u64_array(const std::string& text, const std::string& key,
                    std::vector<std::uint64_t>& values) {
    std::size_t begin = 0U, end = 0U;
    if (!json_array_bounds(text, key, begin, end)) return false;
    ++begin;
    skip_space(text, begin);
    if (begin == end) return false;
    while (begin < end) {
        if (!std::isdigit(static_cast<unsigned char>(text[begin]))) return false;
        const std::size_t first = begin;
        while (begin < end && std::isdigit(static_cast<unsigned char>(text[begin]))) ++begin;
        try {
            std::size_t consumed = 0U;
            const std::string digits = text.substr(first, begin - first);
            const std::uint64_t value = std::stoull(digits, &consumed, 10);
            if (consumed != digits.size() || value == 0U) return false;
            values.push_back(value);
        } catch (...) {
            return false;
        }
        skip_space(text, begin);
        if (begin == end) break;
        if (text[begin] != ',') return false;
        ++begin;
        skip_space(text, begin);
    }
    return !values.empty();
}

bool matching_delimiter(const std::string& text, std::size_t open,
                        char opening, char closing, std::size_t& close) {
    if (open >= text.size() || text[open] != opening) return false;
    std::size_t depth = 0U;
    bool quoted = false;
    bool escaped = false;
    for (std::size_t i = open; i < text.size(); ++i) {
        const char value = text[i];
        if (quoted) {
            if (escaped) escaped = false;
            else if (value == '\\') escaped = true;
            else if (value == '"') quoted = false;
            continue;
        }
        if (value == '"') { quoted = true; continue; }
        if (value == opening) ++depth;
        else if (value == closing) {
            if (depth == 0U) return false;
            --depth;
            if (depth == 0U) { close = i; return true; }
        }
    }
    return false;
}

bool json_object_array(const std::string& text, const std::string& key,
                       std::vector<std::string>& objects) {
    std::size_t begin = 0U, end = 0U;
    if (!json_array_bounds(text, key, begin, end)) return false;
    ++begin;
    skip_space(text, begin);
    if (begin == end) return false;
    while (begin < end) {
        if (text[begin] != '{') return false;
        std::size_t object_end = 0U;
        if (!matching_delimiter(text, begin, '{', '}', object_end) || object_end >= end) return false;
        objects.push_back(text.substr(begin, object_end - begin + 1U));
        begin = object_end + 1U;
        skip_space(text, begin);
        if (begin == end) break;
        if (text[begin] != ',') return false;
        ++begin;
        skip_space(text, begin);
    }
    return !objects.empty();
}

bool resource_field_string(const std::string& object, const std::string& label,
                           const std::string& key, std::string& value) {
    std::size_t object_begin = 0U;
    const std::string marker = "\"" + label + "\"";
    const std::size_t label_pos = object.find(marker);
    if (label_pos != std::string::npos) {
        std::size_t colon = label_pos + marker.size();
        skip_space(object, colon);
        if (colon < object.size() && object[colon] == ':') {
            object_begin = colon + 1U;
            skip_space(object, object_begin);
            if (object_begin < object.size() && object[object_begin] == '{') {
                std::size_t object_end = 0U;
                if (!matching_delimiter(object, object_begin, '{', '}', object_end)) return false;
                return json_string_field(object.substr(object_begin, object_end - object_begin + 1U), key, value);
            }
        }
    }
    return json_string_field(object, label + "_" + key, value);
}

bool resource_field_u64(const std::string& object, const std::string& label,
                        const std::string& key, std::uint64_t& value) {
    const std::string marker = "\"" + label + "\"";
    const std::size_t label_pos = object.find(marker);
    if (label_pos != std::string::npos) {
        std::size_t colon = label_pos + marker.size();
        skip_space(object, colon);
        if (colon < object.size() && object[colon] == ':') {
            std::size_t object_begin = colon + 1U;
            skip_space(object, object_begin);
            if (object_begin < object.size() && object[object_begin] == '{') {
                std::size_t object_end = 0U;
                if (!matching_delimiter(object, object_begin, '{', '}', object_end)) return false;
                return json_u64_field(object.substr(object_begin, object_end - object_begin + 1U), key, value);
            }
        }
    }
    return json_u64_field(object, label + "_" + key, value);
}

bool relative_resource_path(const std::filesystem::path& root,
                            const std::string& raw, std::filesystem::path& path,
                            std::string* error) {
    const std::filesystem::path relative(raw);
    if (raw.empty() || relative.is_absolute()) {
        set_error(error, "BLOCKED: authenticated resource path is not relative: " + raw);
        return false;
    }
    for (const auto& component : relative) {
        if (component == "..") {
            set_error(error, "BLOCKED: authenticated resource path escapes SRS root: " + raw);
            return false;
        }
    }
    path = (root / relative).lexically_normal();
    if (std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path)) {
        set_error(error, "BLOCKED: authenticated resource file is missing or symlinked: " + path.string());
        return false;
    }
    return true;
}

struct PointFrame {
    std::uint32_t version = 0U;
    std::uint32_t point_size = 0U;
    std::uint64_t count = 0U;
    std::uint64_t domain = 0U;
    std::uint64_t degree = 0U;
    std::vector<std::vector<Byte>> points;
};

bool frame_u32(const std::vector<Byte>& bytes, std::size_t& offset, std::uint32_t& value) {
    if (offset > bytes.size() || bytes.size() - offset < 4U) return false;
    value = static_cast<std::uint32_t>(bytes[offset]) |
            (static_cast<std::uint32_t>(bytes[offset + 1U]) << 8U) |
            (static_cast<std::uint32_t>(bytes[offset + 2U]) << 16U) |
            (static_cast<std::uint32_t>(bytes[offset + 3U]) << 24U);
    offset += 4U;
    return true;
}

bool frame_u64(const std::vector<Byte>& bytes, std::size_t& offset, std::uint64_t& value) {
    std::uint32_t low = 0U, high = 0U;
    if (!frame_u32(bytes, offset, low) || !frame_u32(bytes, offset, high)) return false;
    value = static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32U);
    return true;
}

bool decode_frame(const std::vector<Byte>& bytes, PointFrame& frame,
                  std::string* error) {
    constexpr std::array<Byte, 8U> magic = {{'G', 'A', 'T', 'Z', 'K', 'G', '1', 'C'}};
    constexpr std::size_t header_size = 8U + 4U + 4U + 8U + 8U + 8U;
    if (bytes.size() < header_size || !std::equal(magic.begin(), magic.end(), bytes.begin())) {
        set_error(error, "BLOCKED: SRS point file is not a GATZKG1C frame");
        return false;
    }
    std::size_t offset = 8U;
    if (!frame_u32(bytes, offset, frame.version) ||
        !frame_u32(bytes, offset, frame.point_size) ||
        !frame_u64(bytes, offset, frame.count) ||
        !frame_u64(bytes, offset, frame.domain) ||
        !frame_u64(bytes, offset, frame.degree) ||
        frame.version != kFrameVersion || frame.point_size != kG1Size ||
        frame.count == 0U ||
        frame.domain == 0U || frame.count > std::numeric_limits<std::size_t>::max() / kG1Size) {
        set_error(error, "BLOCKED: GATZKG1C frame header is invalid");
        return false;
    }
    std::uint64_t expected = 0U;
    if (frame.count > kMaxSrsFramePoints ||
        frame.count > std::numeric_limits<std::uint64_t>::max() / frame.point_size) {
        set_error(error, "BLOCKED: GATZKG1C frame payload overflows");
        return false;
    }
    const std::uint64_t payload_bytes = frame.count * frame.point_size;
    if (payload_bytes > std::numeric_limits<std::uint64_t>::max() -
                           static_cast<std::uint64_t>(offset)) {
        set_error(error, "BLOCKED: GATZKG1C frame payload offset overflows");
        return false;
    }
    expected = static_cast<std::uint64_t>(offset) + payload_bytes;
    if (expected != bytes.size()) {
        set_error(error, "BLOCKED: GATZKG1C frame has truncated/trailing payload");
        return false;
    }
    frame.points.reserve(static_cast<std::size_t>(frame.count));
    for (std::uint64_t index = 0U; index < frame.count; ++index) {
        std::vector<Byte> encoded(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                                  bytes.begin() + static_cast<std::ptrdiff_t>(offset + frame.point_size));
        offset += frame.point_size;
        std::size_t point_offset = 0U;
        G1Jacobian_t native{};
        if (!zkdl::read_g1(encoded, point_offset, native) || point_offset != encoded.size()) {
            set_error(error, "BLOCKED: GATZKG1C point failed strict G1 decode");
            return false;
        }
        std::array<Byte, kG1Size> canonical{};
        std::string codec_error;
        if (!zkdl_g1_wire::encode(native, canonical, &codec_error) ||
            !std::equal(canonical.begin(), canonical.end(), encoded.begin(), encoded.end())) {
            set_error(error, "BLOCKED: GATZKG1C point failed canonical roundtrip");
            return false;
        }
        if (zkdl_g1_wire::from_native(native).is_zero()) {
            set_error(error, "BLOCKED: authenticated SRS point cannot be identity");
            return false;
        }
        frame.points.push_back(std::move(encoded));
    }
    return true;
}

bool power_of_two(std::uint64_t value) {
    return value != 0U && (value & (value - 1U)) == 0U;
}

bool canonical_point_set(const std::vector<std::vector<Byte>>& points) {
    std::set<std::vector<Byte>> unique(points.begin(), points.end());
    return unique.size() == points.size();
}

bool canonical_path(const Config& config, const std::filesystem::path& path,
                    std::filesystem::path& normalized, std::string* error) {
    const std::filesystem::path root = std::filesystem::absolute(config.baseline_root).lexically_normal();
    normalized = std::filesystem::absolute(path).lexically_normal();
    const auto relative = normalized.lexically_relative(root);
    if (relative.empty() || relative.string().find("..") == 0U ||
        std::filesystem::is_symlink(normalized) || !std::filesystem::is_regular_file(normalized)) {
        set_error(error, "BLOCKED: authenticated model registry path is missing or escapes baseline root: " + normalized.string());
        return false;
    }
    return true;
}

bool parse_hex_bytes(const std::string& text, std::size_t expected,
                     std::vector<Byte>& bytes) {
    if (text.size() != expected * 2U) return false;
    bytes.resize(expected);
    for (std::size_t i = 0U; i < expected; ++i) {
        const int high = hex_nibble(text[2U * i]);
        const int low = hex_nibble(text[2U * i + 1U]);
        if (high < 0 || low < 0) return false;
        bytes[i] = static_cast<Byte>((high << 4U) | low);
    }
    return true;
}

bool parse_native_field(const std::string& text, NativeFieldWire& output) {
    std::vector<Byte> bytes;
    if (!parse_hex_bytes(text, 32U, bytes)) return false;
    for (std::size_t i = 0U; i < output.size(); ++i) {
        output[i] = static_cast<std::uint32_t>(bytes[4U * i]) |
                    (static_cast<std::uint32_t>(bytes[4U * i + 1U]) << 8U) |
                    (static_cast<std::uint32_t>(bytes[4U * i + 2U]) << 16U) |
                    (static_cast<std::uint32_t>(bytes[4U * i + 3U]) << 24U);
    }
    return zkdl::fr_canonical(output) && zkdl::fr_compare(output, NativeFieldWire{}) != 0;
}

bool product_equals(const std::vector<std::uint64_t>& shape,
                    std::uint64_t expected) {
    std::uint64_t product = 1U;
    for (const auto value : shape) {
        if (value == 0U || product > std::numeric_limits<std::uint64_t>::max() / value) return false;
        product *= value;
    }
    return product == expected;
}

bool model_commitment_valid(const std::vector<Byte>& encoded) {
    if (encoded.size() != kG1Size) return false;
    std::size_t offset = 0U;
    G1Jacobian_t native{};
    if (!zkdl::read_g1(encoded, offset, native) || offset != encoded.size() ||
        zkdl_g1_wire::from_native(native).is_zero()) return false;
    std::array<Byte, kG1Size> roundtrip{};
    return zkdl_g1_wire::encode(native, roundtrip, nullptr) &&
           std::equal(roundtrip.begin(), roundtrip.end(), encoded.begin(), encoded.end());
}

bool permissions_owner_only(const std::filesystem::path& path) {
    const auto permissions = std::filesystem::status(path).permissions();
    using P = std::filesystem::perms;
    const P forbidden = P::group_read | P::group_write | P::group_exec |
                        P::others_read | P::others_write | P::others_exec;
    const P required = P::owner_read | P::owner_write;
    return (permissions & forbidden) == P::none && (permissions & required) == required;
}

bool manifest_resource_digest(const std::string& manifest, const std::string& label,
                              std::string& path, std::string& sha256,
                              std::uint64_t& count) {
    return resource_field_string(manifest, label, "path", path) &&
           resource_field_string(manifest, label, "sha256", sha256) &&
           resource_field_u64(manifest, label, "count", count);
}

bool manifest_resource_frame(const std::string& manifest, const std::string& label,
                             const PointFrame& frame, std::uint64_t expected_count,
                             std::uint64_t domain, std::uint64_t degree,
                             std::string* error) {
    std::string path, digest;
    std::uint64_t count = 0U;
    if (!manifest_resource_digest(manifest, label, path, digest, count) ||
        !canonical_digest_text(digest) || count != expected_count ||
        frame.count != count || frame.domain != domain || frame.degree != degree) {
        set_error(error, "BLOCKED: " + label + " GATZKG1C metadata is not manifest-bound");
        return false;
    }
    return true;
}

bool required_registry_string(const std::string& text, const std::string& key,
                              std::string& output, std::string* error) {
    if (!json_string_field(text, key, output) || output.empty()) {
        set_error(error, "BLOCKED: model registry is missing canonical field: " + key);
        return false;
    }
    return true;
}

bool registry_digest_field(const std::string& text, const std::string& key,
                           std::string& output, std::string* error) {
    if (!required_registry_string(text, key, output, error) ||
        output != kUnavailable && !canonical_digest_text(output)) {
        set_error(error, "BLOCKED: model registry digest is not canonical: " + key);
        return false;
    }
    return true;
}

bool expected_file_digest(const std::filesystem::path& path, std::string& output,
                          std::string* error) {
    if (std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path)) {
        set_error(error, "BLOCKED: model registry identity resource is missing: " + path.string());
        return false;
    }
    Digest digest{};
    if (!resource_file_digest(path, digest, error)) return false;
    output = resource_digest_hex(digest);
    return true;
}

bool optional_json_u64(const std::string& text, const char* key,
                       std::uint64_t& value, bool& present) {
    const std::string marker = std::string("\"") + key + "\"";
    if (text.find(marker) == std::string::npos) {
        present = false;
        return true;
    }
    present = true;
    return json_u64_field(text, key, value);
}

bool first_json_string(const std::string& text,
                       std::initializer_list<const char*> keys,
                       std::string& value, bool& present) {
    present = false;
    for (const char* key : keys) {
        const std::string marker = std::string("\"") + key + "\"";
        if (text.find(marker) == std::string::npos) continue;
        present = true;
        return json_string_field(text, key, value);
    }
    return true;
}

bool first_string_occurrence(const std::string& text, const char* key,
                             std::string& value) {
    const std::string marker = std::string("\"") + key + "\"";
    const std::size_t key_pos = text.find(marker);
    if (key_pos == std::string::npos) return false;
    std::size_t colon = key_pos + marker.size();
    skip_space(text, colon);
    if (colon >= text.size() || text[colon] != ':') return false;
    ++colon;
    return json_string_at(text, colon, value);
}

bool generator_id_field(const std::string& object, std::string& value,
                        bool& present) {
    if (!first_json_string(object, {"generator_id", "generator"}, value, present)) return false;
    return true;
}

bool parse_chunk_frame_ref(const std::string& object, const char* label,
                           const std::filesystem::path& root, std::string& path,
                           std::string& digest, std::uint64_t& count,
                           std::string* error) {
    std::string raw_path;
    if (!resource_field_string(object, label, "path", raw_path) ||
        !resource_field_string(object, label, "sha256", digest) ||
        !resource_field_u64(object, label, "count", count) ||
        count == 0U || !canonical_digest_text(digest)) {
        set_error(error, std::string("BLOCKED: SRS chunk omits canonical ") + label +
                  " path/digest/count");
        return false;
    }
    std::filesystem::path checked;
    if (!relative_resource_path(root, raw_path, checked, error)) return false;
    path = checked.lexically_relative(root).generic_string();
    return !path.empty();
}

bool parse_srs_chunk(const std::string& object, const std::string& family_id,
                     std::uint64_t ordinal, const std::filesystem::path& root,
                     const std::string& manifest_generator,
                     const std::string& manifest_basis,
                     AuthenticatedSrsResource::Chunk& chunk,
                     std::string* error) {
    chunk = AuthenticatedSrsResource::Chunk{};
    chunk.family_id = family_id;
    if (chunk.family_id.empty()) {
        set_error(error, "BLOCKED: SRS chunk has empty relation family id");
        return false;
    }
    bool present = false;
    if (!first_json_string(object, {"chunk_id", "id"}, chunk.chunk_id, present)) {
        set_error(error, "BLOCKED: SRS chunk id is malformed");
        return false;
    }
    if (!present) chunk.chunk_id = chunk.family_id + "/chunk" + std::to_string(ordinal);
    if (chunk.chunk_id.empty()) {
        set_error(error, "BLOCKED: SRS chunk has empty chunk id");
        return false;
    }
    if (!optional_json_u64(object, "chunk_index", chunk.chunk_index, present)) {
        set_error(error, "BLOCKED: SRS chunk_index is malformed");
        return false;
    }
    if (!present) chunk.chunk_index = ordinal;
    if (!optional_json_u64(object, "point_offset", chunk.point_offset, present)) {
        set_error(error, "BLOCKED: SRS point_offset is malformed");
        return false;
    }
    std::uint64_t point_size = 0U;
    if (!json_u64_field(object, "point_count", chunk.point_count) ||
        !json_u64_field(object, "domain", chunk.domain) ||
        !json_u64_field(object, "degree", chunk.degree) ||
        !json_u64_field(object, "point_size", point_size) ||
        chunk.point_count == 0U || chunk.domain == 0U ||
        chunk.point_count != chunk.domain || point_size != kG1Size ||
        !power_of_two(chunk.domain) || chunk.domain > kMaxSrsFramePoints ||
        chunk.degree < (chunk.domain == 1U ? 0U : chunk.domain - 1U) ||
        chunk.point_offset > std::numeric_limits<std::uint64_t>::max() - chunk.point_count) {
        set_error(error, "BLOCKED: SRS chunk point_count/domain/degree/point_size is invalid");
        return false;
    }
    chunk.point_size = static_cast<std::uint32_t>(point_size);
    // The native zkDL proof consumes a prefix basis.  A nonzero offset is
    // still parsed and authenticated as metadata, but cannot be silently
    // reinterpreted as the first generator of a different basis.
    if (chunk.point_offset != 0U) {
        set_error(error, "BLOCKED: native zkDL SRS chunk point_offset must be zero for prefix basis");
        return false;
    }
    bool has_generator = false;
    if (!generator_id_field(object, chunk.generator_id, has_generator)) {
        set_error(error, "BLOCKED: SRS chunk generator_id is malformed");
        return false;
    }
    if (!has_generator) chunk.generator_id = manifest_generator;
    if (chunk.generator_id.empty()) {
        set_error(error, "BLOCKED: SRS chunk has no authenticated generator_id");
        return false;
    }
    bool has_basis = false;
    if (!first_json_string(object, {"basis_digest"}, chunk.basis_digest, has_basis) ||
        !has_basis || !canonical_digest_text(chunk.basis_digest) ||
        chunk.basis_digest != manifest_basis) {
        set_error(error, "BLOCKED: SRS chunk basis_digest is not manifest-bound");
        return false;
    }
    std::uint64_t g_count = 0U, h_count = 0U, u_count = 0U;
    if (!parse_chunk_frame_ref(object, "G", root, chunk.g_path, chunk.g_digest, g_count, error) ||
        !parse_chunk_frame_ref(object, "H", root, chunk.h_path, chunk.h_digest, h_count, error) ||
        !parse_chunk_frame_ref(object, "U", root, chunk.u_path, chunk.u_digest, u_count, error) ||
        g_count != chunk.point_count || h_count != 1U || u_count != 1U) {
        set_error(error, "BLOCKED: SRS chunk G/H/U point counts are not domain-bound");
        return false;
    }
    bool has_generator_digest = false;
    std::string declared_generator_digest;
    if (!first_json_string(object, {"generator_digest", "g_digest"},
                           declared_generator_digest, has_generator_digest)) {
        set_error(error, "BLOCKED: SRS chunk generator digest is malformed");
        return false;
    }
    if (has_generator_digest) {
        if (!canonical_digest_text(declared_generator_digest) ||
            declared_generator_digest != chunk.g_digest) {
            set_error(error, "BLOCKED: SRS chunk generator_digest does not bind G frame");
            return false;
        }
    }
    chunk.generator_digest = chunk.g_digest;
    return true;
}

bool materialize_srs_chunk(const std::filesystem::path& root,
                           AuthenticatedSrsResource::Chunk& chunk,
                           std::string* error) {
    if (chunk.authenticated && chunk.materialized) return true;
    std::filesystem::path g_path, h_path, u_path;
    if (!relative_resource_path(root, chunk.g_path, g_path, error) ||
        !relative_resource_path(root, chunk.h_path, h_path, error) ||
        !relative_resource_path(root, chunk.u_path, u_path, error)) return false;
    std::vector<Byte> g_bytes, h_bytes, u_bytes;
    if (!read_bytes(g_path, g_bytes, error, kMaxSrsFrameBytes) ||
        !read_bytes(h_path, h_bytes, error, kMaxSrsFrameBytes) ||
        !read_bytes(u_path, u_bytes, error, kMaxSrsFrameBytes)) return false;
    const auto check_digest = [&](const std::vector<Byte>& bytes,
                                  const std::string& expected, const char* label) {
        if (resource_digest_hex(zkdl::sha256(bytes)) != expected) {
            set_error(error, std::string("BLOCKED: SRS chunk ") + label +
                      " frame digest mismatch");
            return false;
        }
        return true;
    };
    if (!check_digest(g_bytes, chunk.g_digest, "G") ||
        !check_digest(h_bytes, chunk.h_digest, "H") ||
        !check_digest(u_bytes, chunk.u_digest, "U")) return false;
    if (basis_digest_hex(chunk.g_digest, chunk.h_digest, chunk.u_digest) != chunk.basis_digest) {
        set_error(error, "BLOCKED: SRS chunk basis_digest does not bind G/H/U frames");
        return false;
    }
    PointFrame g_frame, h_frame, u_frame;
    if (!decode_frame(g_bytes, g_frame, error) ||
        !decode_frame(h_bytes, h_frame, error) ||
        !decode_frame(u_bytes, u_frame, error) ||
        g_frame.count != chunk.point_count || h_frame.count != 1U ||
        u_frame.count != 1U || g_frame.domain != chunk.domain ||
        h_frame.domain != chunk.domain || u_frame.domain != chunk.domain ||
        g_frame.degree != chunk.degree || h_frame.degree != chunk.degree ||
        u_frame.degree != chunk.degree || !canonical_point_set(g_frame.points)) {
        set_error(error, "BLOCKED: SRS chunk frame header is not manifest-bound");
        return false;
    }
    if (g_frame.points.end() !=
            std::find(g_frame.points.begin(), g_frame.points.end(), h_frame.points.front()) ||
        g_frame.points.end() !=
            std::find(g_frame.points.begin(), g_frame.points.end(), u_frame.points.front()) ||
        h_frame.points.front() == u_frame.points.front()) {
        set_error(error, "BLOCKED: SRS chunk G/H/U roles are not independent");
        return false;
    }
    chunk.generators = std::move(g_frame.points);
    chunk.blinding_generator = std::move(h_frame.points.front());
    chunk.u_generator = std::move(u_frame.points.front());
    chunk.materialized = true;
    chunk.authenticated = true;
    return true;
}

bool choose_default_chunk(const std::string& manifest,
                          const std::vector<AuthenticatedSrsResource::Chunk>& chunks,
                          std::size_t& selected, std::string* error) {
    std::string default_family, default_chunk;
    bool has_family = false, has_chunk = false;
    if (!first_json_string(manifest, {"default_family_id", "default_family"},
                           default_family, has_family) ||
        !first_json_string(manifest, {"default_chunk_id", "default_chunk"},
                           default_chunk, has_chunk)) {
        set_error(error, "BLOCKED: SRS default chunk identity is malformed");
        return false;
    }
    if (has_family || has_chunk) {
        bool found = false;
        for (std::size_t i = 0U; i < chunks.size(); ++i) {
            if ((!has_family || chunks[i].family_id == default_family) &&
                (!has_chunk || chunks[i].chunk_id == default_chunk)) {
                if (found) {
                    set_error(error, "BLOCKED: SRS manifest has duplicate default chunk identity");
                    return false;
                }
                selected = i;
                found = true;
            }
        }
        if (!found) {
            set_error(error, "BLOCKED: SRS manifest default chunk is not present");
            return false;
        }
        return true;
    }
    selected = 0U;
    for (std::size_t i = 1U; i < chunks.size(); ++i) {
        const auto& candidate = chunks[i];
        const auto& current = chunks[selected];
        if (candidate.domain > current.domain ||
            (candidate.domain == current.domain &&
             std::tie(candidate.family_id, candidate.chunk_id, candidate.chunk_index) <
                 std::tie(current.family_id, current.chunk_id, current.chunk_index))) {
            selected = i;
        }
    }
    return true;
}

}  // namespace

bool load_authenticated_srs(const Config& config,
                             AuthenticatedSrsResource* resource,
                             std::string* error) {
    if (!resource) {
        set_error(error, "BLOCKED: SRS loader output is null");
        return false;
    }
    *resource = AuthenticatedSrsResource{};
    const std::filesystem::path root(config.srs_path);
    if (std::filesystem::is_symlink(root) || !std::filesystem::is_directory(root)) {
        set_error(error, "BLOCKED: canonical SRS directory is missing: " + root.string());
        return false;
    }
    if (config.srs_manifest_digest.empty() || config.srs_manifest_digest == kUnavailable ||
        !canonical_digest_text(config.srs_manifest_digest)) {
        set_error(error, "BLOCKED: secure config lacks a pre-authenticated srs_manifest_digest");
        return false;
    }
    Digest expected_manifest{};
    if (!resource_parse_digest(config.srs_manifest_digest, expected_manifest)) {
        set_error(error, "BLOCKED: srs_manifest_digest is not a canonical SHA-256");
        return false;
    }
    const std::filesystem::path manifest_path = root / "srs_manifest.json";
    if (std::filesystem::is_symlink(manifest_path) || !std::filesystem::is_regular_file(manifest_path)) {
        set_error(error, "BLOCKED: canonical SRS manifest is missing: " + manifest_path.string());
        return false;
    }
    std::vector<Byte> manifest_bytes;
    if (!read_bytes(manifest_path, manifest_bytes, error)) return false;
    const Digest actual_manifest = zkdl::sha256(manifest_bytes);
    if (actual_manifest != expected_manifest) {
        set_error(error, "BLOCKED: srs_manifest.json digest does not match secure config");
        return false;
    }
    const std::string manifest(manifest_bytes.begin(), manifest_bytes.end());
    std::string identity;
    if (!required_registry_string(manifest, "manifest_format", identity, error) ||
        identity != "zkgat_srs_manifest" ||
        !required_registry_string(manifest, "curve_id", identity, error) ||
        identity != "bls12_381" ||
        !required_registry_string(manifest, "encoding", identity, error) ||
        identity != "GATZKG1C" ||
        !required_registry_string(manifest, "srs_id", identity, error) ||
        identity != config.srs_id) {
        set_error(error, "BLOCKED: canonical SRS manifest identity/encoding mismatch");
        return false;
    }
    if (config.srs_type != "native_local_ephemeral_random") {
        set_error(error, "BLOCKED: secure config srs_type is not native_local_ephemeral_random");
        return false;
    }
    std::string setup_schema, setup_mode, trust_status, setup_algorithm, setup_spec;
    std::string setup_curve, setup_config_binding, setup_basis;
    const bool setup_ok =
        resource_field_string(manifest, "setup", "schema", setup_schema) &&
        setup_schema == "zkDL_gat_local_random_setup_v1" &&
        resource_field_string(manifest, "setup", "mode", setup_mode) &&
        setup_mode == "native_local_ephemeral_random" &&
        resource_field_string(manifest, "setup", "trust_status", trust_status) &&
        trust_status == "native_local_ephemeral_random" &&
        resource_field_string(manifest, "setup", "algorithm", setup_algorithm) &&
        setup_algorithm == "os_csprng_nonzero_scalar_mul_bls12_381_g1" &&
        resource_field_string(manifest, "setup", "spec", setup_spec) &&
        setup_spec == "zkDL_Hyrax_IPA_Pedersen_independent_g1_basis_v1" &&
        resource_field_string(manifest, "setup", "curve_id", setup_curve) &&
        setup_curve == "bls12_381" &&
        resource_field_string(manifest, "setup", "config_binding_digest", setup_config_binding) &&
        canonical_digest_text(setup_config_binding) &&
        setup_config_binding == config.config_binding_digest &&
        resource_field_string(manifest, "setup", "basis_digest", setup_basis) &&
        canonical_digest_text(setup_basis);
    std::string manifest_basis;
    if (!setup_ok || !first_string_occurrence(manifest, "basis_digest", manifest_basis) ||
        manifest_basis != setup_basis) {
        set_error(error, "BLOCKED: SRS manifest local random setup metadata is invalid");
        return false;
    }
    std::uint64_t setup_domain = 0U, setup_generator_count = 0U;
    std::uint64_t setup_h_count = 0U, setup_u_count = 0U;
    if (!resource_field_u64(manifest, "setup", "domain", setup_domain) ||
        !resource_field_u64(manifest, "setup", "generator_count", setup_generator_count) ||
        !resource_field_u64(manifest, "setup", "h_count", setup_h_count) ||
        !resource_field_u64(manifest, "setup", "u_count", setup_u_count) ||
        !power_of_two(setup_domain) || setup_generator_count != setup_domain ||
        setup_h_count != 1U || setup_u_count != 1U) {
        set_error(error, "BLOCKED: SRS local random setup domain/count metadata is invalid");
        return false;
    }
    std::string manifest_generator;
    if (!first_string_occurrence(manifest, "generator_id", manifest_generator) ||
        manifest_generator != "zkDL_native_local_csprng_g1_v1") {
        set_error(error, "BLOCKED: local random SRS generator_id is invalid");
        return false;
    }
    std::uint64_t manifest_version = 0U;
    bool has_version = false;
    if (!optional_json_u64(manifest, "manifest_version", manifest_version, has_version) ||
        !has_version || manifest_version != 2U) {
        set_error(error, "BLOCKED: local random SRS requires manifest_version=2");
        return false;
    }
    std::string chunking_rule;
    bool has_chunking_rule = false;
    if (!first_json_string(manifest, {"chunking_rule", "srs_chunking_rule"},
                           chunking_rule, has_chunking_rule)) {
        set_error(error, "BLOCKED: SRS manifest chunking_rule is malformed");
        return false;
    }

    std::vector<AuthenticatedSrsResource::Chunk> chunks;
    const bool has_families = manifest.find("\"families\"") != std::string::npos;
    if (!has_families) {
        set_error(error, "BLOCKED: local random SRS requires the canonical families layout");
        return false;
    }
    if (has_families) {
        std::vector<std::string> families;
        if (!json_object_array(manifest, "families", families)) {
            set_error(error, "BLOCKED: SRS families array is malformed");
            return false;
        }
        for (const auto& family_object : families) {
            std::string family_id;
            bool family_present = false;
            if (!first_json_string(family_object, {"family_id", "id"}, family_id,
                                   family_present) || !family_present || family_id.empty()) {
                set_error(error, "BLOCKED: SRS family has no canonical family_id");
                return false;
            }
            std::vector<std::string> family_chunks;
            if (!json_object_array(family_object, "chunks", family_chunks) || family_chunks.empty()) {
                set_error(error, "BLOCKED: SRS family has no canonical chunks");
                return false;
            }
            for (std::size_t index = 0U; index < family_chunks.size(); ++index) {
                AuthenticatedSrsResource::Chunk chunk;
                if (!parse_srs_chunk(family_chunks[index], family_id, index, root,
                                     manifest_generator, manifest_basis, chunk, error)) return false;
                chunks.push_back(std::move(chunk));
            }
        }
    }
    if (chunks.empty()) {
        set_error(error, "BLOCKED: authenticated SRS manifest has no chunks");
        return false;
    }
    if (!has_chunking_rule && chunking_rule.empty())
        chunking_rule = "per_family_prefix_v1";
    if (!config.chunking_rule.empty() && config.chunking_rule != "none" &&
        config.chunking_rule != chunking_rule) {
        set_error(error, "BLOCKED: SRS manifest chunking_rule differs from secure config");
        return false;
    }
    std::set<std::string> identities;
    std::set<std::pair<std::string, std::uint64_t>> indexes;
    for (const auto& chunk : chunks) {
        const std::string identity_key = chunk.family_id + "\n" + chunk.chunk_id;
        if (!identities.insert(identity_key).second ||
            !indexes.insert({chunk.family_id, chunk.chunk_index}).second) {
            set_error(error, "BLOCKED: SRS manifest has duplicate family/chunk identity");
            return false;
        }
        if (!manifest_generator.empty() && chunk.generator_id != manifest_generator) {
            set_error(error, "BLOCKED: SRS chunk generator_id differs from manifest");
            return false;
        }
        if (chunk.basis_digest != manifest_basis) {
            set_error(error, "BLOCKED: SRS chunk basis digest differs from manifest");
            return false;
        }
    }
    std::size_t selected = 0U;
    if (!choose_default_chunk(manifest, chunks, selected, error)) return false;
    if (!materialize_srs_chunk(root, chunks[selected], error)) return false;
    const auto& selected_chunk = chunks[selected];
    Digest selected_generator_digest{};
    if (!resource_parse_digest(selected_chunk.generator_digest, selected_generator_digest)) {
        set_error(error, "BLOCKED: selected SRS generator digest is not canonical");
        return false;
    }
    resource->manifest_path = manifest_path.string();
    resource->manifest_digest = actual_manifest;
    resource->generator_digest = selected_generator_digest;
    resource->chunking_rule = chunking_rule;
    resource->generator_id = selected_chunk.generator_id;
    resource->basis_digest = selected_chunk.basis_digest;
    const std::string selected_family_id = selected_chunk.family_id;
    const std::string selected_chunk_id = selected_chunk.chunk_id;
    // Keep the actual vector position, not the manifest's logical
    // chunk_index.  Selection and common-frame reuse index `chunks` directly;
    // chunk_index is only an authenticated per-family ordering field and is
    // not required to be globally contiguous.
    const std::uint64_t selected_chunk_vector_index = static_cast<std::uint64_t>(selected);
    const std::uint64_t selected_count = selected_chunk.point_count;
    const std::uint64_t selected_domain = selected_chunk.domain;
    const std::uint64_t selected_degree = selected_chunk.degree;
    const std::uint32_t selected_point_size = selected_chunk.point_size;
    resource->chunks = std::move(chunks);
    resource->selected_family_id = selected_family_id;
    resource->selected_chunk_id = selected_chunk_id;
    resource->selected_chunk_index = selected_chunk_vector_index;
    resource->count = selected_count;
    resource->domain = selected_domain;
    resource->degree = selected_degree;
    resource->point_size = selected_point_size;
    resource->generators = resource->chunks[selected].generators;
    resource->blinding_generator = resource->chunks[selected].blinding_generator;
    resource->u_generator = resource->chunks[selected].u_generator;
    resource->authenticated = true;
    return true;
}

bool select_authenticated_srs_chunk(AuthenticatedSrsResource* resource,
                                     const std::string& family_id,
                                     std::uint64_t required_domain,
                                     const AuthenticatedSrsResource::Chunk** chunk,
                                     std::string* error) {
    if (!resource || !chunk) {
        set_error(error, "BLOCKED: SRS chunk selector output is null");
        return false;
    }
    *chunk = nullptr;
    if (!resource->authenticated || resource->manifest_path.empty() ||
        !power_of_two(required_domain) || required_domain == 0U) {
        set_error(error, "BLOCKED: SRS chunk selector has invalid authenticated request");
        return false;
    }
    std::size_t selected = resource->chunks.size();
    for (std::size_t i = 0U; i < resource->chunks.size(); ++i) {
        const auto& candidate = resource->chunks[i];
        if (candidate.family_id != family_id && candidate.family_id != "*") continue;
        if (candidate.point_count < required_domain || candidate.domain < required_domain) continue;
        if (selected == resource->chunks.size() ||
            std::tie(candidate.domain, candidate.chunk_index, candidate.chunk_id) <
                std::tie(resource->chunks[selected].domain,
                         resource->chunks[selected].chunk_index,
                         resource->chunks[selected].chunk_id)) selected = i;
    }
    if (selected == resource->chunks.size()) {
        set_error(error, "BLOCKED: authenticated SRS manifest has no chunk for relation family/domain: " +
                  family_id + "/" + std::to_string(required_domain));
        return false;
    }
    const std::filesystem::path root = std::filesystem::path(resource->manifest_path).parent_path();
    // The preparation tool may retain one family entry per relation family
    // while authenticating all of them against the same common prefix frame.
    // Reuse the already authenticated default material in that case instead
    // of rereading and retaining another copy of a potentially multi-GB G
    // frame.  The path and digest tuple is part of the authenticated chunk
    // metadata; a partial/path-only match is never treated as shared.
    if (resource->selected_chunk_index >= resource->chunks.size()) {
        set_error(error, "BLOCKED: authenticated SRS default chunk index is out of range");
        return false;
    }
    auto& common = resource->chunks[resource->selected_chunk_index];
    auto& candidate = resource->chunks[selected];
    const bool same_authenticated_frames =
        selected != resource->selected_chunk_index && common.materialized &&
        candidate.g_path == common.g_path && candidate.h_path == common.h_path &&
        candidate.u_path == common.u_path && candidate.g_digest == common.g_digest &&
        candidate.h_digest == common.h_digest && candidate.u_digest == common.u_digest;
    if (same_authenticated_frames) {
        if (common.domain < required_domain || common.generators.size() < required_domain ||
            common.blinding_generator.empty() || common.u_generator.empty()) {
            set_error(error, "BLOCKED: authenticated common SRS frame does not cover relation domain");
            return false;
        }
        *chunk = &common;
        return true;
    }
    if (!materialize_srs_chunk(root, resource->chunks[selected], error)) return false;
    const auto& selected_chunk = resource->chunks[selected];
    if (selected_chunk.domain < required_domain ||
        selected_chunk.generators.size() < required_domain ||
        selected_chunk.blinding_generator.empty() || selected_chunk.u_generator.empty()) {
        set_error(error, "BLOCKED: authenticated SRS selected chunk does not cover relation domain");
        return false;
    }
    *chunk = &selected_chunk;
    return true;
}

bool load_authenticated_model_registry(const Config& config,
                                       const AuthenticatedSrsResource& srs,
                                       AuthenticatedModelRegistry* resource,
                                       std::string* error) {
    if (!resource) {
        set_error(error, "BLOCKED: model registry loader output is null");
        return false;
    }
    *resource = AuthenticatedModelRegistry{};
    if (!srs.authenticated || srs.manifest_digest == Digest{} || srs.generator_digest == Digest{}) {
        set_error(error, "BLOCKED: model registry requires an authenticated SRS resource");
        return false;
    }
    std::filesystem::path path;
    if (!canonical_path(config, config.model_registry_path, path, error)) return false;
    Digest expected_digest{};
    if (!resource_parse_digest(config.verifier_registry_digest, expected_digest)) {
        set_error(error, "BLOCKED: verifier_registry_digest is not canonical");
        return false;
    }
    Digest actual_digest{};
    std::vector<Byte> bytes;
    if (!read_bytes(path, bytes, error)) return false;
    actual_digest = zkdl::sha256(bytes);
    if (actual_digest != expected_digest) {
        set_error(error, "BLOCKED: model registry digest does not match secure config");
        return false;
    }
    const std::string text(bytes.begin(), bytes.end());
    std::string value;
    if (!required_registry_string(text, "schema", value, error) || value != "zkDL_gat_model_registry_v2") {
        set_error(error, "BLOCKED: model registry schema is not canonical v2");
        return false;
    }
    auto require_match = [&](const char* key, const std::string& expected) {
        std::string actual;
        return required_registry_string(text, key, actual, error) && actual == expected;
    };
    if (!require_match("dataset", config.dataset) ||
        !require_match("model_arch_id", config.model_arch_id) ||
        !require_match("model_param_id", config.model_param_id) ||
        !require_match("quant_cfg_id", config.quant_cfg_id) ||
        !require_match("static_table_id", config.static_table_id) ||
        !require_match("srs_id", config.srs_id) ||
        !require_match("source_root", "baseline-local")) {
        set_error(error, "BLOCKED: model registry config identity mismatch");
        return false;
    }
    std::string registry_identity_digest;
    if (!registry_digest_field(text, "srs_manifest_digest", value, error) ||
        value != resource_digest_hex(srs.manifest_digest) ||
        !registry_digest_field(text, "generator_digest", value, error) ||
        value != resource_digest_hex(srs.generator_digest) ||
        !registry_digest_field(text, "config_digest", value, error) ||
        value != config.config_digest ||
        !registry_digest_field(text, "registry_identity_digest", registry_identity_digest, error) ||
        !registry_digest_field(text, "verifier_registry_digest", value, error) ||
        value != registry_identity_digest ||
        !registry_digest_field(text, "scale_chain_digest", value, error) ||
        value != config.scale_chain_digest) {
        set_error(error, "BLOCKED: model registry SRS/config/scale identity mismatch");
        return false;
    }
    std::string static_table_digest;
    std::string integer_bound_ledger_digest;
    std::string model_manifest_digest;
    const auto required_resource_digest = [&](const char* key,
                                              const std::filesystem::path& path_to_check,
                                              std::string& captured) {
        std::string declared;
        std::string actual;
        if (!registry_digest_field(text, key, declared, error) ||
            !expected_file_digest(path_to_check, actual, error) || declared != actual) return false;
        captured = std::move(declared);
        return true;
    };
    if (!required_resource_digest("static_table_digest", config.static_table_descriptor_path,
                                 static_table_digest) ||
        !required_resource_digest("integer_bound_ledger_digest", config.integer_bound_ledger_path,
                                 integer_bound_ledger_digest) ||
        !required_resource_digest("model_manifest_digest", config.checkpoint_bundle / "manifest.json",
                                 model_manifest_digest)) return false;
    std::vector<std::string> tensor_objects;
    if (!json_object_array(text, "model_tensors", tensor_objects) || tensor_objects.empty()) {
        set_error(error, "BLOCKED: model registry has no canonical model_tensors entries");
        return false;
    }
    std::set<std::string> ids;
    for (const auto& object : tensor_objects) {
        NativeCommitmentInput commitment;
        std::string commitment_text;
        std::uint64_t domain = 0U;
        std::vector<std::uint64_t> logical_shape, padded_shape;
        if (!required_registry_string(object, "tensor_id", commitment.tensor_id, error) ||
            !ids.insert(commitment.tensor_id).second ||
            !json_u64_field(object, "domain_size", domain) || domain == 0U ||
            domain > std::numeric_limits<std::uint32_t>::max() || !power_of_two(domain) ||
            !json_u64_array(object, "logical_shape", logical_shape) ||
            !json_u64_array(object, "padded_shape", padded_shape) ||
            !product_equals(padded_shape, domain) ||
            !(json_string_field(object, "commitment", commitment_text) ||
              resource_field_string(object, "commitment", "hex", commitment_text)) ||
            !parse_hex_bytes(commitment_text, kG1Size, commitment.commitment) ||
            !model_commitment_valid(commitment.commitment)) {
            set_error(error, "BLOCKED: model registry tensor entry is malformed: " + commitment.tensor_id);
            return false;
        }
        bool shape_valid = logical_shape.size() == padded_shape.size();
        for (std::size_t index = 0U; shape_valid && index < logical_shape.size(); ++index)
            shape_valid = logical_shape[index] != 0U && logical_shape[index] <= padded_shape[index];
        if (!shape_valid || domain > srs.domain || commitment_text.size() != kG1Size * 2U ||
            !std::all_of(commitment_text.begin(), commitment_text.end(), [](char c) {
                return std::isdigit(static_cast<unsigned char>(c)) || (c >= 'a' && c <= 'f');
            })) {
            set_error(error, "BLOCKED: model registry tensor commitment is not lowercase canonical hex");
            return false;
        }
        commitment.visibility = 2U;
        commitment.domain_size = static_cast<std::uint32_t>(domain);
        resource->model_commitments.push_back(std::move(commitment));
    }
    resource->path = path.string();
    resource->digest = actual_digest;
    resource->schema = "zkDL_gat_model_registry_v2";
    resource->dataset = config.dataset;
    resource->model_arch_id = config.model_arch_id;
    resource->model_param_id = config.model_param_id;
    resource->quant_cfg_id = config.quant_cfg_id;
    resource->static_table_id = config.static_table_id;
    resource->srs_id = config.srs_id;
    resource->srs_manifest_digest = resource_digest_hex(srs.manifest_digest);
    resource->generator_digest = resource_digest_hex(srs.generator_digest);
    resource->config_digest = config.config_digest;
    resource->verifier_registry_digest = config.verifier_registry_digest;
    resource->scale_chain_digest = config.scale_chain_digest;
    resource->static_table_digest = std::move(static_table_digest);
    resource->integer_bound_ledger_digest = std::move(integer_bound_ledger_digest);
    resource->model_manifest_digest = std::move(model_manifest_digest);
    resource->authenticated = true;
    return true;
}

bool load_authenticated_model_blinders(const Config& config,
                                       const AuthenticatedModelRegistry& registry,
                                       AuthenticatedModelBlinders* resource,
                                       std::string* error) {
    if (!resource) {
        set_error(error, "BLOCKED: model blinder loader output is null");
        return false;
    }
    *resource = AuthenticatedModelBlinders{};
    if (!registry.authenticated || registry.digest == Digest{}) {
        set_error(error, "BLOCKED: model blinder sidecar requires authenticated model registry");
        return false;
    }
    if (config.model_blinder_sidecar_digest.empty() ||
        config.model_blinder_sidecar_digest == kUnavailable ||
        !canonical_digest_text(config.model_blinder_sidecar_digest)) {
        set_error(error, "BLOCKED: secure config lacks a pre-authenticated model blinder sidecar digest");
        return false;
    }
    std::filesystem::path path;
    if (!canonical_path(config, config.model_blinder_sidecar_path, path, error) ||
        !permissions_owner_only(path)) {
        set_error(error, "BLOCKED: model blinder sidecar must be a private owner-only file");
        return false;
    }
    Digest expected_digest{};
    if (!resource_parse_digest(config.model_blinder_sidecar_digest, expected_digest)) {
        set_error(error, "BLOCKED: model blinder sidecar digest is not canonical");
        return false;
    }
    std::vector<Byte> bytes;
    if (!read_bytes(path, bytes, error)) return false;
    const Digest actual_digest = zkdl::sha256(bytes);
    if (actual_digest != expected_digest) {
        set_error(error, "BLOCKED: model blinder sidecar digest does not match secure config");
        return false;
    }
    const std::string text(bytes.begin(), bytes.end());
    std::string value;
    if (!required_registry_string(text, "schema", value, error) ||
        value != "zkDL_gat_model_blinder_sidecar_v1" ||
        !required_registry_string(text, "registry_digest", value, error) ||
        value != resource_digest_hex(registry.digest)) {
        set_error(error, "BLOCKED: model blinder sidecar registry identity mismatch");
        return false;
    }
    std::vector<std::string> objects;
    if (!json_object_array(text, "entries", objects) || objects.size() != registry.model_commitments.size()) {
        set_error(error, "BLOCKED: model blinder sidecar entry count does not match model registry");
        return false;
    }
    std::set<std::string> ids;
    for (const auto& object : objects) {
        NativeModelBlinderInput entry;
        std::string blinder_text;
        if (!required_registry_string(object, "tensor_id", entry.tensor_id, error) ||
            !ids.insert(entry.tensor_id).second ||
            !required_registry_string(object, "blinder", blinder_text, error) ||
            !parse_native_field(blinder_text, entry.blinder)) {
            set_error(error, "BLOCKED: model blinder sidecar entry is malformed: " + entry.tensor_id);
            return false;
        }
        entry.registry_digest = registry.digest;
        const auto found = std::find_if(registry.model_commitments.begin(), registry.model_commitments.end(),
                                        [&](const NativeCommitmentInput& commitment) {
                                            return commitment.tensor_id == entry.tensor_id;
                                        });
        if (found == registry.model_commitments.end()) {
            set_error(error, "BLOCKED: model blinder sidecar has an unregistered tensor: " + entry.tensor_id);
            return false;
        }
        resource->entries.push_back(std::move(entry));
    }
    resource->path = path.string();
    resource->digest = actual_digest;
    resource->registry_digest = registry.digest;
    resource->authenticated = true;
    return true;
}

}  // namespace zkdl_gat
