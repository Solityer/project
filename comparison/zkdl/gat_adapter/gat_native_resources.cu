#include "gat_native_resources.hpp"

#include "../proof.cuh"
#include "../g1-wire.cuh"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace zkdl_gat {
namespace {

using Byte = std::uint8_t;
using Digest = std::array<Byte, 32U>;
constexpr std::size_t kG1Size = 48U;
constexpr std::uint32_t kFrameVersion = 1U;
constexpr std::uint64_t kMaxSrsPoints = 1ULL << 19U;
constexpr std::size_t kMaxJsonResourceBytes = 1U << 26U;
constexpr const char* kUnavailable = "UNAVAILABLE";

void set_error(std::string* error, const std::string& message) {
    if (error) *error = message;
}

bool read_bytes(const std::filesystem::path& path, std::vector<Byte>& bytes,
                std::string* error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        set_error(error, "BLOCKED: cannot open authenticated resource: " + path.string());
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) > kMaxJsonResourceBytes) {
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
        frame.count == 0U || frame.count > kMaxSrsPoints ||
        frame.domain == 0U || frame.count > std::numeric_limits<std::size_t>::max() / kG1Size) {
        set_error(error, "BLOCKED: GATZKG1C frame header is invalid");
        return false;
    }
    std::uint64_t expected = 0U;
    if (frame.count > std::numeric_limits<std::uint64_t>::max() / frame.point_size) {
        set_error(error, "BLOCKED: GATZKG1C frame payload overflows");
        return false;
    }
    expected = static_cast<std::uint64_t>(offset) + frame.count * frame.point_size;
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
    std::uint64_t domain = 0U, degree = 0U, point_size = 0U;
    if (!json_u64_field(manifest, "domain", domain) ||
        !json_u64_field(manifest, "degree", degree) ||
        !json_u64_field(manifest, "point_size", point_size) ||
        !power_of_two(domain) || domain > kMaxSrsPoints ||
        (degree == 0U && domain != 1U) ||
        (domain > 1U && degree < domain - 1U) ||
        point_size != kG1Size) {
        set_error(error, "BLOCKED: canonical SRS manifest domain/degree/point_size is invalid");
        return false;
    }
    const auto frame_path = [&](const char* label, std::filesystem::path& path) {
        std::string raw;
        if (!resource_field_string(manifest, label, "path", raw)) {
            set_error(error, std::string("BLOCKED: SRS manifest omits ") + label + " path");
            return false;
        }
        if (!relative_resource_path(root, raw, path, error)) return false;
        return true;
    };
    std::filesystem::path g_path, h_path, u_path;
    if (!frame_path("G", g_path) || !frame_path("H", h_path) || !frame_path("U", u_path)) return false;
    std::vector<Byte> g_bytes, h_bytes, u_bytes;
    if (!read_bytes(g_path, g_bytes, error) || !read_bytes(h_path, h_bytes, error) ||
        !read_bytes(u_path, u_bytes, error)) return false;
    std::string g_digest_text, h_digest_text, u_digest_text;
    std::uint64_t g_count = 0U, h_count = 0U, u_count = 0U;
    if (!manifest_resource_digest(manifest, "G", identity, g_digest_text, g_count) ||
        !manifest_resource_digest(manifest, "H", identity, h_digest_text, h_count) ||
        !manifest_resource_digest(manifest, "U", identity, u_digest_text, u_count) ||
        !canonical_digest_text(g_digest_text) || !canonical_digest_text(h_digest_text) ||
        !canonical_digest_text(u_digest_text)) {
        set_error(error, "BLOCKED: canonical SRS manifest point metadata is malformed");
        return false;
    }
    const auto check_file_hash = [&](const std::vector<Byte>& bytes, const std::string& expected,
                                     const char* label) {
        if (resource_digest_hex(zkdl::sha256(bytes)) != expected) {
            set_error(error, std::string("BLOCKED: ") + label + " file digest mismatch");
            return false;
        }
        return true;
    };
    if (!check_file_hash(g_bytes, g_digest_text, "G") ||
        !check_file_hash(h_bytes, h_digest_text, "H") ||
        !check_file_hash(u_bytes, u_digest_text, "U")) return false;
    PointFrame g_frame, h_frame, u_frame;
    if (!decode_frame(g_bytes, g_frame, error) || !decode_frame(h_bytes, h_frame, error) ||
        !decode_frame(u_bytes, u_frame, error)) return false;
    if (!manifest_resource_frame(manifest, "G", g_frame, domain, domain, degree, error) ||
        !manifest_resource_frame(manifest, "H", h_frame, 1U, domain, degree, error) ||
        !manifest_resource_frame(manifest, "U", u_frame, 1U, domain, degree, error) ||
        !canonical_point_set(g_frame.points)) return false;
    if (g_frame.points.size() != g_count || h_frame.points.size() != h_count ||
        u_frame.points.size() != u_count || h_frame.points.size() != 1U || u_frame.points.size() != 1U) {
        set_error(error, "BLOCKED: SRS frame count does not match manifest");
        return false;
    }
    if (g_frame.points.end() != std::find(g_frame.points.begin(), g_frame.points.end(), h_frame.points.front()) ||
        g_frame.points.end() != std::find(g_frame.points.begin(), g_frame.points.end(), u_frame.points.front()) ||
        h_frame.points.front() == u_frame.points.front()) {
        set_error(error, "BLOCKED: SRS G/H/U roles are not independent");
        return false;
    }
    resource->manifest_path = manifest_path.string();
    resource->manifest_digest = actual_manifest;
    resource->generator_digest = zkdl::sha256(g_bytes);
    resource->count = g_frame.count;
    resource->domain = domain;
    resource->degree = degree;
    resource->point_size = static_cast<std::uint32_t>(point_size);
    resource->generators = std::move(g_frame.points);
    resource->blinding_generator = std::move(h_frame.points.front());
    resource->u_generator = std::move(u_frame.points.front());
    resource->authenticated = true;
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
    if (!registry_digest_field(text, "srs_manifest_digest", value, error) ||
        value != resource_digest_hex(srs.manifest_digest) ||
        !registry_digest_field(text, "generator_digest", value, error) ||
        value != resource_digest_hex(srs.generator_digest) ||
        !registry_digest_field(text, "config_digest", value, error) ||
        value != config.config_digest ||
        !registry_digest_field(text, "verifier_registry_digest", value, error) ||
        value != config.verifier_registry_digest ||
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
            domain > kMaxSrsPoints || !power_of_two(domain) ||
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
