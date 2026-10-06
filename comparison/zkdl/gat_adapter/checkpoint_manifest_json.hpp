#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace zkdl_gat::checkpoint_manifest_json {

struct Span { std::size_t begin = 0U; std::size_t end = 0U; };

[[noreturn]] inline void reject(const std::string &message) { throw std::runtime_error(message); }

inline void skip_space(const std::string &text, std::size_t &position) {
    while (position < text.size() && (text[position] == ' ' || text[position] == '\t' || text[position] == '\r' || text[position] == '\n')) ++position;
}

inline std::uint64_t hex4(const std::string &text, std::size_t &position) {
    if (position + 4U > text.size()) reject("truncated JSON unicode escape in checkpoint manifest");
    std::uint64_t value = 0U;
    for (unsigned digit_index = 0U; digit_index < 4U; ++digit_index) {
        const char digit = text[position++];
        value <<= 4U;
        if (digit >= '0' && digit <= '9') value |= static_cast<std::uint64_t>(digit - '0');
        else if (digit >= 'a' && digit <= 'f') value |= static_cast<std::uint64_t>(digit - 'a' + 10);
        else if (digit >= 'A' && digit <= 'F') value |= static_cast<std::uint64_t>(digit - 'A' + 10);
        else reject("invalid JSON unicode escape in checkpoint manifest");
    }
    return value;
}

inline void append_utf8(std::string &output, std::uint64_t value) {
    if (value <= 0x7fU) output.push_back(static_cast<char>(value));
    else if (value <= 0x7ffU) { output.push_back(static_cast<char>(0xc0U | (value >> 6U))); output.push_back(static_cast<char>(0x80U | (value & 0x3fU))); }
    else if (value <= 0xffffU) { output.push_back(static_cast<char>(0xe0U | (value >> 12U))); output.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3fU))); output.push_back(static_cast<char>(0x80U | (value & 0x3fU))); }
    else { output.push_back(static_cast<char>(0xf0U | (value >> 18U))); output.push_back(static_cast<char>(0x80U | ((value >> 12U) & 0x3fU))); output.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3fU))); output.push_back(static_cast<char>(0x80U | (value & 0x3fU))); }
}

inline std::string read_string(const std::string &text, std::size_t &position) {
    if (position >= text.size() || text[position++] != '"') reject("expected JSON string in checkpoint manifest");
    std::string output;
    while (position < text.size()) {
        const unsigned char value = static_cast<unsigned char>(text[position++]);
        if (value == '"') return output;
        if (value < 0x20U) reject("control byte in JSON string in checkpoint manifest");
        if (value != '\\') { output.push_back(static_cast<char>(value)); continue; }
        if (position >= text.size()) reject("truncated JSON escape in checkpoint manifest");
        const char escape = text[position++];
        if (escape == '"' || escape == '\\' || escape == '/') output.push_back(escape);
        else if (escape == 'b') output.push_back('\b');
        else if (escape == 'f') output.push_back('\f');
        else if (escape == 'n') output.push_back('\n');
        else if (escape == 'r') output.push_back('\r');
        else if (escape == 't') output.push_back('\t');
        else if (escape == 'u') {
            std::uint64_t codepoint = hex4(text, position);
            if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
                if (position + 2U > text.size() || text[position++] != '\\' || text[position++] != 'u') reject("unpaired JSON high surrogate in checkpoint manifest");
                const std::uint64_t low = hex4(text, position);
                if (low < 0xdc00U || low > 0xdfffU) reject("invalid JSON surrogate pair in checkpoint manifest");
                codepoint = 0x10000U + ((codepoint - 0xd800U) << 10U) + (low - 0xdc00U);
            } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) reject("unpaired JSON low surrogate in checkpoint manifest");
            append_utf8(output, codepoint);
        } else reject("invalid JSON escape in checkpoint manifest");
    }
    reject("unterminated JSON string in checkpoint manifest");
}

inline std::size_t scan_value(const std::string &text, std::size_t position, unsigned depth = 0U) {
    if (depth > 128U) reject("checkpoint manifest JSON nesting exceeds the parser bound");
    skip_space(text, position);
    if (position >= text.size()) reject("missing JSON value in checkpoint manifest");
    if (text[position] == '"') { (void)read_string(text, position); return position; }
    if (text[position] == '{') {
        ++position;
        skip_space(text, position);
        if (position < text.size() && text[position] == '}') return position + 1U;
        std::set<std::string> keys;
        for (;;) {
            skip_space(text, position);
            const auto key = read_string(text, position);
            if (!keys.insert(key).second) reject("duplicate JSON object key in checkpoint manifest: " + key);
            skip_space(text, position);
            if (position >= text.size() || text[position++] != ':') reject("missing JSON object colon in checkpoint manifest");
            position = scan_value(text, position, depth + 1U);
            skip_space(text, position);
            if (position >= text.size()) reject("unterminated JSON object in checkpoint manifest");
            if (text[position] == '}') return position + 1U;
            if (text[position++] != ',') reject("invalid JSON object separator in checkpoint manifest");
        }
    }
    if (text[position] == '[') {
        ++position;
        skip_space(text, position);
        if (position < text.size() && text[position] == ']') return position + 1U;
        for (;;) {
            position = scan_value(text, position, depth + 1U);
            skip_space(text, position);
            if (position >= text.size()) reject("unterminated JSON array in checkpoint manifest");
            if (text[position] == ']') return position + 1U;
            if (text[position++] != ',') reject("invalid JSON array separator in checkpoint manifest");
        }
    }
    if (text.compare(position, 4U, "true") == 0) return position + 4U;
    if (text.compare(position, 5U, "false") == 0) return position + 5U;
    if (text.compare(position, 4U, "null") == 0) return position + 4U;
    const std::size_t begin = position;
    if (text[position] == '-') ++position;
    if (position >= text.size()) reject("malformed JSON number in checkpoint manifest");
    if (text[position] == '0') ++position;
    else if (text[position] >= '1' && text[position] <= '9') { do { ++position; } while (position < text.size() && text[position] >= '0' && text[position] <= '9'); }
    else reject("malformed JSON number in checkpoint manifest");
    if (position < text.size() && text[position] == '.') { ++position; const auto fraction = position; while (position < text.size() && text[position] >= '0' && text[position] <= '9') ++position; if (position == fraction) reject("malformed JSON fraction in checkpoint manifest"); }
    if (position < text.size() && (text[position] == 'e' || text[position] == 'E')) { ++position; if (position < text.size() && (text[position] == '+' || text[position] == '-')) ++position; const auto exponent = position; while (position < text.size() && text[position] >= '0' && text[position] <= '9') ++position; if (position == exponent) reject("malformed JSON exponent in checkpoint manifest"); }
    if (position == begin) reject("invalid JSON value in checkpoint manifest");
    return position;
}

inline std::map<std::string, Span> object_members(const std::string &text, Span object, const std::string &what) {
    std::size_t position = object.begin;
    skip_space(text, position);
    if (position >= text.size() || text[position++] != '{') reject(what + " is not a JSON object");
    std::map<std::string, Span> members;
    skip_space(text, position);
    if (position < text.size() && text[position] == '}') ++position;
    else {
        for (;;) {
            skip_space(text, position);
            const std::string key = read_string(text, position);
            skip_space(text, position);
            if (position >= text.size() || text[position++] != ':') reject("missing JSON object colon in " + what);
            skip_space(text, position);
            const std::size_t begin = position;
            const std::size_t end = scan_value(text, position);
            if (!members.emplace(key, Span{begin, end}).second) reject("duplicate JSON object key in " + what + ": " + key);
            position = end;
            skip_space(text, position);
            if (position >= text.size()) reject("unterminated JSON object in " + what);
            if (text[position] == '}') { ++position; break; }
            if (text[position++] != ',') reject("invalid JSON object separator in " + what);
        }
    }
    if (position != object.end) reject("unexpected JSON bytes after " + what);
    return members;
}

inline std::uint64_t parse_u64(const std::string &text, const std::string &what) {
    if (text.empty() || text.size() > 20U || (text.size() > 1U && text.front() == '0')) reject(what + " is not a canonical unsigned integer");
    std::uint64_t value = 0U;
    for (const char digit : text) {
        if (digit < '0' || digit > '9') reject(what + " contains a non-unsigned integer");
        const std::uint64_t next = static_cast<std::uint64_t>(digit - '0');
        if (value > (UINT64_MAX - next) / 10U) reject(what + " exceeds u64");
        value = value * 10U + next;
    }
    return value;
}

inline std::vector<std::uint64_t> u64_array(const std::string &text, Span array, const std::string &what) {
    std::size_t position = array.begin;
    skip_space(text, position);
    if (position >= text.size() || text[position++] != '[') reject(what + " is not a JSON array");
    std::vector<std::uint64_t> values;
    skip_space(text, position);
    if (position < text.size() && text[position] == ']') ++position;
    else {
        for (;;) {
            skip_space(text, position);
            const auto begin = position;
            position = scan_value(text, position);
            const auto value = parse_u64(text.substr(begin, position - begin), what);
            if (value == 0U) reject(what + " contains a non-positive dimension");
            values.push_back(value);
            skip_space(text, position);
            if (position >= text.size()) reject("unterminated " + what);
            if (text[position] == ']') { ++position; break; }
            if (text[position++] != ',') reject("invalid separator in " + what);
        }
    }
    if (position != array.end || values.empty()) reject(what + " is empty or has trailing JSON bytes");
    return values;
}

inline std::map<std::string, std::vector<std::uint64_t>> tensor_shapes(const std::string &manifest) {
    const std::size_t root_end = scan_value(manifest, 0U);
    std::size_t trailing = root_end;
    skip_space(manifest, trailing);
    if (trailing != manifest.size()) reject("checkpoint manifest has trailing JSON bytes");
    const auto root = object_members(manifest, Span{0U, root_end}, "checkpoint manifest");
    const auto index_it = root.find("tensor_index");
    if (index_it == root.end()) reject("checkpoint manifest tensor_index missing");
    const auto index = object_members(manifest, index_it->second, "checkpoint manifest tensor_index");
    std::map<std::string, std::vector<std::uint64_t>> shapes;
    for (const auto &entry : index) {
        const auto descriptor = object_members(manifest, entry.second, "checkpoint manifest tensor descriptor " + entry.first);
        const auto shape_it = descriptor.find("shape");
        if (shape_it == descriptor.end()) reject("checkpoint manifest tensor shape missing: " + entry.first);
        if (!shapes.emplace(entry.first, u64_array(manifest, shape_it->second, "manifest tensor shape " + entry.first)).second)
            reject("duplicate checkpoint manifest tensor index: " + entry.first);
    }
    if (shapes.empty()) reject("checkpoint manifest tensor_index is empty");
    return shapes;
}

} // namespace zkdl_gat::checkpoint_manifest_json
