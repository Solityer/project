#include "gat_proof.hpp"
#include "gat_formal_internal.hpp"
#include "gat_types.hpp"
#include "native_storage.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace {

using namespace zkgpt::gat;
using Clock = std::chrono::steady_clock;

struct Options {
    std::string dataset;
    std::string backend;
    std::string baseline_root = ".";
    std::string config;
    std::string registration;
    std::string com_w_secret;
    std::string formal_session_id;
    std::string run_nonce;
    std::string session_manifest_sha256;
    std::string formal_run_directory;
    bool help = false;
    bool register_only = false;
    bool capacity_only = false;
    bool capacity_profile_only = false;
    bool defer_self_verify = false;
};

struct Record {
    std::string schema = "zkgpt-gat-result-v2";
    std::string paper_timing_semantics = "zkgpt-gat-paper-timing-v4-native-commitment-wire-io-excluded";
    std::string dataset;
    std::string backend;
    std::string capability_status = "FORMAL_CPU_MASKED_FULL";
    std::string status = "FAILED";
    std::string verification_status = "FAILED";
    std::string proof_scope = "full_gat";
    std::string privacy_policy = "model_and_trace_hidden";
    std::string privacy_profile = "masked_full";
    std::string input_privacy = "public_graph_public_features";
    std::string output_mode = "public_logits";
    std::string native_backend;
    std::string failure_reason;
    std::string statement_digest;
    std::string circuit_id;
    std::string plan_digest;
    std::string run_dir;
    std::string session_id;
    std::string run_nonce;
    std::string session_manifest_sha256;
    std::string statement_path;
    std::string public_logits_path;
    std::string verifier_context_path;
    std::string registration_path;
    std::string prover_secret_path;
    std::string registration_digest;
    std::string registration_binding_digest;
    std::string registration_layout_digest;
    std::string artifact_path;
    std::string verifier_report_path;
    std::string proof_digest;
    std::string artifact_sha256;
    std::string full_dataset_identity;
    std::string checkpoint_identity;
    std::string cache_identity;
    std::string com_w_point;
    bool proof_generated = false;
    bool verifier_called = false;
    bool verifier_ok = false;
    bool verifier_reads_witness = false;
    bool verifier_reads_checkpoint = false;
    bool cpu_fallback = false;
    bool formal_backend_supported = true;
    bool gpu_compiled = false;
    bool gpu_device_available = false;
    bool gpu_kernel_executed = false;
    std::uint64_t gpu_kernel_invocations = 0;
    std::uint64_t proof_message_bytes = 0;
    std::uint64_t proof_size_bytes = 0;
    std::uint64_t artifact_size_bytes = 0;
    double load_ms = 0.0;
    double registration_ms = 0.0;
    double witness_ms = 0.0;
    double preparation_ms = 0.0;
    double preparation_net_ms = 0.0;
    double aux_commitment_ms = 0.0;
    double lasso_query_commitment_ms = 0.0;
    double lasso_family_generator_derive_ms = 0.0;
    double lasso_family_commitment_ms = 0.0;
    double lasso_family_selected_row_mapping_ms = 0.0;
    double commitment_ms = 0.0;
    double prove_total_ms = 0.0;
    double self_verify_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    double initial_commitment_blinding_prepare_ms = 0.0;
    double serialize_ms = 0.0;
    double artifact_wire_self_check_ms = 0.0;
    double artifact_write_ms = 0.0;
    double artifact_readback_ms = 0.0;
};

struct CapacityResourceSnapshot {
    std::uint64_t host_headroom_bytes = 0;
    std::uint64_t scratch_available_bytes = 0;
    std::uint64_t artifact_available_bytes = 0;
    std::uint64_t shared_available_bytes = 0;
    bool same_filesystem = false;
};

std::uint64_t parse_u64_text(const std::string& text, const std::string& what) {
    if (text.empty()) throw std::runtime_error("empty numeric resource value: " + what);
    std::uint64_t value = 0;
    for (const unsigned char ch : text) {
        if (ch < '0' || ch > '9') throw std::runtime_error("invalid numeric resource value: " + what);
        const std::uint64_t digit = static_cast<std::uint64_t>(ch - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10)
            throw std::overflow_error("resource value exceeds u64: " + what);
        value = value * 10 + digit;
    }
    return value;
}

std::optional<std::uint64_t> read_u64_file(const std::filesystem::path& path, bool allow_max) {
    std::ifstream input(path);
    if (!input) return std::nullopt;
    std::string value;
    input >> value;
    if (value == "max" && allow_max) return std::nullopt;
    return parse_u64_text(value, path.string());
}

struct HostMemoryInfo {
    std::uint64_t available_bytes = 0;
    std::uint64_t total_bytes = 0;
};

HostMemoryInfo host_memory_info() {
    HostMemoryInfo result;
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    std::uint64_t available_kib = 0;
    std::uint64_t total_kib = 0;
    std::string unit;
    while (meminfo >> key) {
        std::uint64_t value = 0;
        if (!(meminfo >> value >> unit)) break;
        std::string rest;
        std::getline(meminfo, rest);
        if (unit != "kB") continue;
        if (key == "MemAvailable:") available_kib = value;
        if (key == "MemTotal:") total_kib = value;
    }
    const auto kib_to_bytes = [](std::uint64_t value, const char* what) {
        if (value > std::numeric_limits<std::uint64_t>::max() / 1024)
            throw std::overflow_error(std::string("host memory value exceeds u64: ") + what);
        return value * 1024;
    };
    result.total_bytes = kib_to_bytes(total_kib, "MemTotal");
    if (available_kib != 0) {
        result.available_bytes = kib_to_bytes(available_kib, "MemAvailable");
    } else {
        const long pages = ::sysconf(_SC_AVPHYS_PAGES);
        const long page_size = ::sysconf(_SC_PAGESIZE);
        if (pages <= 0 || page_size <= 0 || static_cast<unsigned long long>(pages) > std::numeric_limits<std::uint64_t>::max() / static_cast<unsigned long long>(page_size))
            throw std::runtime_error("cannot determine available physical memory");
        result.available_bytes = static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page_size);
    }
    if (result.total_bytes == 0) {
        const long pages = ::sysconf(_SC_PHYS_PAGES);
        const long page_size = ::sysconf(_SC_PAGESIZE);
        if (pages > 0 && page_size > 0 && static_cast<unsigned long long>(pages) <= std::numeric_limits<std::uint64_t>::max() / static_cast<unsigned long long>(page_size))
            result.total_bytes = static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page_size);
    }
    if (result.total_bytes == 0 || result.available_bytes == 0)
        throw std::runtime_error("cannot establish available host memory headroom");
    return result;
}

std::optional<std::uint64_t> cgroup_memory_headroom(const std::uint64_t physical_total_bytes) {
    std::ifstream cgroup("/proc/self/cgroup");
    std::string line;
    std::optional<std::filesystem::path> v2_path;
    std::optional<std::filesystem::path> v1_memory_path;
    while (std::getline(cgroup, line)) {
        const std::size_t first = line.find(':');
        const std::size_t second = first == std::string::npos ? std::string::npos : line.find(':', first + 1);
        if (second == std::string::npos) continue;
        const std::string controllers = line.substr(first + 1, second - first - 1);
        std::string group_path = line.substr(second + 1);
        if (group_path.find("..") != std::string::npos)
            throw std::runtime_error("cgroup path contains parent traversal");
        while (!group_path.empty() && group_path.front() == '/') group_path.erase(group_path.begin());
        if (controllers.empty()) v2_path = std::filesystem::path("/sys/fs/cgroup") / group_path;
        else if (controllers.find("memory") != std::string::npos) v1_memory_path = std::filesystem::path("/sys/fs/cgroup/memory") / group_path;
    }
    const auto measure = [physical_total_bytes](const std::filesystem::path& start, const std::filesystem::path& mount, bool v2) -> std::optional<std::uint64_t> {
        std::optional<std::uint64_t> headroom;
        std::filesystem::path current = start.lexically_normal();
        const std::filesystem::path root = mount.lexically_normal();
        for (;;) {
            const auto current_usage = read_u64_file(current / (v2 ? "memory.current" : "memory.usage_in_bytes"), false);
            const auto apply_limit = [&](const char* filename) {
                const auto limit = read_u64_file(current / filename, v2);
                if (!limit || !current_usage) return;
                if (!v2 && *limit >= physical_total_bytes) return;
                const std::uint64_t remaining = *limit > *current_usage ? *limit - *current_usage : 0;
                headroom = headroom ? std::min(*headroom, remaining) : remaining;
            };
            apply_limit(v2 ? "memory.max" : "memory.limit_in_bytes");
            if (v2) apply_limit("memory.high");
            else apply_limit("memory.soft_limit_in_bytes");
            if (current == root || current.empty()) break;
            const auto parent = current.parent_path();
            if (parent == current || parent.empty()) break;
            current = parent;
        }
        return headroom;
    };
    if (v2_path) {
        const auto headroom = measure(*v2_path, "/sys/fs/cgroup", true);
        if (headroom) return headroom;
    }
    if (v1_memory_path) {
        const auto headroom = measure(*v1_memory_path, "/sys/fs/cgroup/memory", false);
        if (headroom) return headroom;
    }
    return std::nullopt;
}

std::uint64_t effective_host_headroom_bytes() {
    const HostMemoryInfo memory = host_memory_info();
    const auto cgroup_headroom = cgroup_memory_headroom(memory.total_bytes);
    return cgroup_headroom ? std::min(memory.available_bytes, *cgroup_headroom) : memory.available_bytes;
}

CapacityResourceSnapshot capacity_resource_snapshot(const std::string& run_dir) {
    CapacityResourceSnapshot snapshot;
    snapshot.host_headroom_bytes = effective_host_headroom_bytes();
    const auto scratch_dir = zkgpt::native_storage::scratch_directory();
    snapshot.scratch_available_bytes = zkgpt::native_storage::scratch_available_bytes();
    std::error_code space_error;
    const std::filesystem::space_info artifact_space = std::filesystem::space(run_dir, space_error);
    if (space_error || artifact_space.available > std::numeric_limits<std::uint64_t>::max())
        throw std::runtime_error("cannot establish artifact filesystem headroom: " + run_dir);
    snapshot.artifact_available_bytes = static_cast<std::uint64_t>(artifact_space.available);
    struct stat scratch_stat{};
    struct stat artifact_stat{};
    if (::stat(scratch_dir.c_str(), &scratch_stat) != 0 || ::stat(run_dir.c_str(), &artifact_stat) != 0)
        throw std::runtime_error("cannot determine scratch/artifact filesystem identity");
    snapshot.same_filesystem = scratch_stat.st_dev == artifact_stat.st_dev;
    if (snapshot.same_filesystem)
        snapshot.shared_available_bytes = std::min(snapshot.scratch_available_bytes, snapshot.artifact_available_bytes);
    return snapshot;
}

double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
}

bool absolute_path(const std::string& path) { return !path.empty() && (path[0] == '/' || path[0] == '\\'); }

std::string without_trailing_slash(std::string path) {
    while (path.size() > 1 && (path.back() == '/' || path.back() == '\\')) path.pop_back();
    return path;
}

std::string join_path(const std::string& left, const std::string& right) {
    if (right.empty()) return left;
    if (left.empty() || left == ".") return right;
    return without_trailing_slash(left) + "/" + right;
}

std::string local_path(const std::string& root, const std::string& relative, const char* what) {
    if (relative.empty() || absolute_path(relative)) { throw std::runtime_error(std::string("baseline-local ") + what + " must be a relative path"); }
    std::size_t begin = 0;
    while (begin <= relative.size()) {
        const std::size_t slash = relative.find('/', begin);
        const std::string part = relative.substr(begin, slash == std::string::npos ? std::string::npos : slash - begin);
        if (part == "..") throw std::runtime_error(std::string("baseline-local ") + what + " escapes the baseline root");
        if (slash == std::string::npos) break;
        begin = slash + 1;
    }
    return join_path(root, relative);
}

void make_directory(const std::string& path) {
    if (path.empty() || path == ".") return;
    std::string current;
    if (path[0] == '/') current = "/";
    std::size_t begin = path[0] == '/' ? 1 : 0;
    while (begin <= path.size()) {
        const std::size_t slash = path.find('/', begin);
        const std::string part = path.substr(begin, slash == std::string::npos ? std::string::npos : slash - begin);
        if (!part.empty() && part != ".") {
            if (!current.empty() && current.back() != '/') current.push_back('/');
            current += part;
            if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) { throw std::runtime_error("cannot create baseline output directory: " + current); }
        }
        if (slash == std::string::npos) break;
        begin = slash + 1;
    }
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (ch < 0x20)
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(ch) << std::dec;
            else
                out << static_cast<char>(ch);
        }
    }
    return out.str();
}

void write_text(const std::string& path, const std::string& content) {
    const std::string temporary = path + ".partial";
    {
        std::ofstream out(temporary.c_str(), std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open output file: " + path);
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
        if (!out.good()) throw std::runtime_error("cannot write output file: " + path);
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::remove(temporary.c_str());
        throw std::runtime_error("cannot finalize output file: " + path);
    }
}

std::string read_regular_text(const std::string& path, const char* description, std::size_t limit = 4U * 1024U * 1024U) {
    struct stat info{};
    if (::lstat(path.c_str(), &info) != 0 || S_ISLNK(info.st_mode) || !S_ISREG(info.st_mode) || info.st_nlink != 1) {
        throw std::runtime_error(std::string("invalid ") + description + ": " + path);
    }
    if (info.st_size < 0 || static_cast<std::uint64_t>(info.st_size) > limit) {
        throw std::runtime_error(std::string(description) + " exceeds the input limit: " + path);
    }
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + description + ": " + path);
    std::ostringstream out;
    out << in.rdbuf();
    if (!in.good() && !in.eof()) throw std::runtime_error(std::string("cannot read ") + description + ": " + path);
    return out.str();
}

bool json_string_field(const std::string& json, const std::string& key, std::string& value) {
    const std::string needle = "\"" + key + "\"";
    std::size_t cursor = 0;
    while ((cursor = json.find(needle, cursor)) != std::string::npos) {
        std::size_t colon = cursor + needle.size();
        while (colon < json.size() && (json[colon] == ' ' || json[colon] == '\n' || json[colon] == '\r' || json[colon] == '\t')) ++colon;
        if (colon >= json.size() || json[colon] != ':') {
            cursor += needle.size();
            continue;
        }
        ++colon;
        while (colon < json.size() && (json[colon] == ' ' || json[colon] == '\n' || json[colon] == '\r' || json[colon] == '\t')) ++colon;
        if (colon >= json.size() || json[colon] != '"') return false;
        ++colon;
        std::ostringstream decoded;
        bool escaped = false;
        for (; colon < json.size(); ++colon) {
            const char ch = json[colon];
            if (escaped) {
                switch (ch) {
                case '"':
                    decoded << '"';
                    break;
                case '\\':
                    decoded << '\\';
                    break;
                case '/':
                    decoded << '/';
                    break;
                case 'n':
                    decoded << '\n';
                    break;
                case 'r':
                    decoded << '\r';
                    break;
                case 't':
                    decoded << '\t';
                    break;
                default:
                    return false;
                }
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                value = decoded.str();
                return true;
            } else {
                decoded << ch;
            }
        }
        return false;
    }
    return false;
}

bool json_size_field(const std::string& json, const std::string& key, std::size_t& value, std::size_t search_from = 0) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t cursor = json.find(needle, search_from);
    if (cursor == std::string::npos) return false;
    std::size_t colon = cursor + needle.size();
    while (colon < json.size() && (json[colon] == ' ' || json[colon] == '\n' || json[colon] == '\r' || json[colon] == '\t')) ++colon;
    if (colon >= json.size() || json[colon] != ':') return false;
    ++colon;
    while (colon < json.size() && (json[colon] == ' ' || json[colon] == '\n' || json[colon] == '\r' || json[colon] == '\t')) ++colon;
    if (colon >= json.size() || json[colon] < '0' || json[colon] > '9') return false;
    const std::size_t begin = colon;
    while (colon < json.size() && json[colon] >= '0' && json[colon] <= '9') ++colon;
    if (colon < json.size() && json[colon] != ',' && json[colon] != '}' && json[colon] != ' ' && json[colon] != '\n' && json[colon] != '\r' &&
        json[colon] != '\t')
        return false;
    try {
        const unsigned long long parsed = std::stoull(json.substr(begin, colon - begin));
        if (parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) return false;
        value = static_cast<std::size_t>(parsed);
        return true;
    } catch (...) { return false; }
}

ModelRegistration read_registration(const std::string& path, const std::string& dataset, bool allow_legacy = false) {
    const std::string json = read_regular_text(path, "model registration");
    ModelRegistration registration;
    std::string schema;
    std::string commitment_scheme;
    if (!json_string_field(json, "schema", schema) || (schema != "zkgpt-gat-model-registration-v2" && !(allow_legacy && schema == "zkgpt-gat-model-registration-v1")) ||
        !json_string_field(json, "dataset", registration.dataset) || !json_string_field(json, "status", registration.status) ||
        !json_string_field(json, "model_arch_id", registration.model_arch_id) || !json_string_field(json, "quant_cfg_id", registration.quant_cfg_id) ||
        !json_string_field(json, "static_tables_id", registration.static_tables_id) || !json_string_field(json, "field_id", registration.field_id) ||
        !json_string_field(json, "field_modulus", registration.field_modulus) || !json_string_field(json, "manifest_digest", registration.manifest_digest) ||
        !json_string_field(json, "model_digest", registration.model_digest) || !json_string_field(json, "bounds_digest", registration.bounds_digest) ||
        !json_string_field(json, "quant_digest", registration.quant_digest) || !json_string_field(json, "layout_digest", registration.layout_digest) ||
        !json_string_field(json, "selector", registration.selector) || !json_string_field(json, "opening_policy", registration.opening_policy) ||
        !json_string_field(json, "binding_digest", registration.binding_digest) ||
        !json_string_field(json, "registration_digest", registration.registration_digest) || !json_string_field(json, "commitment_scheme", commitment_scheme) ||
        !json_string_field(json, "basis_id", registration.com_W.basis_id) || !json_string_field(json, "point_hex", registration.com_W.point_hex)) {
        throw std::runtime_error("model registration artifact is incomplete");
    }
    registration.com_W.scheme = commitment_scheme;
    if (registration.dataset != dataset) { throw std::runtime_error("model registration dataset mismatch"); }
    if (!json_size_field(json, "n_raw", registration.n_raw) || !json_size_field(json, "n_in", registration.n_in) ||
        !json_size_field(json, "logical_size", registration.com_W.logical_size) || !json_size_field(json, "padded_size", registration.com_W.padded_size)) {
        throw std::runtime_error("model registration numeric fields are incomplete");
    }
    const std::string segment_marker = "\"input_segments\"";
    std::size_t cursor = json.find(segment_marker);
    cursor = cursor == std::string::npos ? cursor : json.find('[', cursor + segment_marker.size());
    if (cursor == std::string::npos) throw std::runtime_error("model registration input segments are missing");
    const char* const names[] = {"X_pub", "W", "Aux", "L_qry"};
    for (const char* name : names) {
        InputSegment segment;
        const std::size_t name_key = json.find("\"name\"", cursor);
        if (name_key == std::string::npos || !json_string_field(json.substr(name_key), "name", segment.name) || segment.name != name ||
            !json_size_field(json, "offset", segment.offset, cursor) || !json_size_field(json, "logical_length", segment.logical_length, cursor) ||
            !json_size_field(json, "padded_length", segment.padded_length, cursor)) {
            throw std::runtime_error("model registration input segment is malformed");
        }
        registration.input_segments.push_back(segment);
        const std::size_t object_end = json.find('}', cursor);
        if (object_end == std::string::npos) throw std::runtime_error("model registration input segment is truncated");
        cursor = object_end + 1;
    }
    registration.has_private_padding_mask = false;
    if (registration.status != "registered_long_lived_com_W" || registration.input_segments.size() != 4 || !registration.com_W.valid()) {
        throw std::runtime_error("model registration wire failed canonical validation");
    }
    return registration;
}

void require_owner_secret_file(const std::string& path) {
    struct stat info{};
    if (::lstat(path.c_str(), &info) != 0 || S_ISLNK(info.st_mode) || !S_ISREG(info.st_mode) || info.st_nlink != 1 ||
        (info.st_mode & 07777) != (S_IRUSR | S_IWUSR) || info.st_uid != ::geteuid()) {
        throw std::runtime_error("com_W prover secret must be an owner-only 0600 regular file");
    }
}

bool path_exists(const std::string& path) {
    struct stat info{};
    return ::lstat(path.c_str(), &info) == 0;
}

std::string rel_display(const std::string& root, const std::string& path) {
    const std::string clean_root = without_trailing_slash(root);
    if (clean_root == ".") return path;
    const std::string prefix = clean_root + "/";
    return path.compare(0, prefix.size(), prefix) == 0 ? path.substr(prefix.size()) : path;
}

std::string bool_json(bool value) { return value ? "true" : "false"; }

void json_string(std::ostringstream& out, const char* key, const std::string& value, bool comma = true) {
    out << "  \"" << key << "\": \"" << json_escape(value) << "\"" << (comma ? ",\n" : "\n");
}

void json_bool(std::ostringstream& out, const char* key, bool value, bool comma = true) {
    out << "  \"" << key << "\": " << bool_json(value) << (comma ? ",\n" : "\n");
}

void json_size_array(std::ostringstream& out, const char* key, const std::vector<std::size_t>& values, const char* indent = "    ") {
    out << indent << "\"" << key << "\": [";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) out << ',';
        out << values[index];
    }
    out << ']';
}

void json_number(std::ostringstream& out, const char* key, double value, bool comma = true) {
    out << "  \"" << key << "\": " << std::setprecision(std::numeric_limits<double>::max_digits10)
        << value << (comma ? ",\n" : "\n");
}

void write_result(const std::string& path, const Record& record) {
    std::ostringstream out;
    out << "{\n";
    json_string(out, "schema", record.schema);
    json_string(out, "paper_timing_semantics", record.paper_timing_semantics);
    json_string(out, "dataset", record.dataset);
    json_string(out, "backend", record.backend);
    json_string(out, "capability_status", record.capability_status);
    json_string(out, "status", record.status);
    json_string(out, "verification_status", record.verification_status);
    json_string(out, "proof_scope", record.proof_scope);
    json_string(out, "privacy_policy", record.privacy_policy);
    json_string(out, "privacy_profile", record.privacy_profile);
    json_string(out, "input_privacy", record.input_privacy);
    json_string(out, "output_mode", record.output_mode);
    json_string(out, "native_backend", record.native_backend);
    json_string(out, "failure_reason", record.failure_reason);
    json_string(out, "statement_digest", record.statement_digest);
    json_string(out, "proof_digest", record.proof_digest);
    json_string(out, "circuit_id", record.circuit_id);
    json_string(out, "plan_digest", record.plan_digest);
    json_string(out, "full_dataset_identity", record.full_dataset_identity);
    json_string(out, "checkpoint_identity", record.checkpoint_identity);
    json_string(out, "cache_identity", record.cache_identity);
    json_string(out, "run_dir", record.run_dir);
    json_string(out, "session_id", record.session_id);
    json_string(out, "run_nonce", record.run_nonce);
    json_string(out, "session_manifest_sha256", record.session_manifest_sha256);
    json_string(out, "statement_path", record.statement_path);
    json_string(out, "public_logits_path", record.public_logits_path);
    json_string(out, "verifier_context_path", record.verifier_context_path);
    json_string(out, "registration_path", record.registration_path);
    json_string(out, "prover_secret_path", record.prover_secret_path);
    json_string(out, "registration_digest", record.registration_digest);
    json_string(out, "registration_binding_digest", record.registration_binding_digest);
    json_string(out, "registration_layout_digest", record.registration_layout_digest);
    json_string(out, "artifact_path", record.artifact_path);
    json_string(out, "verifier_report_path", record.verifier_report_path);
    json_string(out, "artifact_sha256", record.artifact_sha256);
    json_string(out, "com_W_point", record.com_w_point);
    json_bool(out, "proof_generated", record.proof_generated);
    json_bool(out, "verifier_called", record.verifier_called);
    json_bool(out, "verifier_ok", record.verifier_ok);
    json_bool(out, "verifier_reads_witness", record.verifier_reads_witness);
    json_bool(out, "verifier_reads_checkpoint", record.verifier_reads_checkpoint);
    json_bool(out, "cpu_fallback", record.cpu_fallback);
    json_bool(out, "formal_backend_supported", record.formal_backend_supported);
    json_bool(out, "gpu_compiled", record.gpu_compiled);
    json_bool(out, "gpu_device_available", record.gpu_device_available);
    json_bool(out, "gpu_kernel_executed", record.gpu_kernel_executed);
    out << "  \"gpu_kernel_invocations\": " << record.gpu_kernel_invocations << ",\n";
    out << "  \"proof_message_bytes\": " << record.proof_message_bytes << ",\n";
    out << "  \"proof_size_bytes\": " << record.proof_size_bytes << ",\n";
    out << "  \"artifact_size_bytes\": " << record.artifact_size_bytes << ",\n";
    out << "  \"timing_ms\": {\n";
    out << "    \"load\": " << std::setprecision(17) << record.load_ms << ",\n";
    out << "    \"registration\": " << record.registration_ms << ",\n";
    out << "    \"witness\": " << record.witness_ms << ",\n";
    out << "    \"preparation\": " << record.preparation_ms << ",\n";
    out << "    \"preparation_net\": " << record.preparation_net_ms << ",\n";
    out << "    \"aux_commitment\": " << record.aux_commitment_ms << ",\n";
    out << "    \"lasso_query_commitment\": " << record.lasso_query_commitment_ms << ",\n";
    out << "    \"lasso_family_generator_derive\": " << record.lasso_family_generator_derive_ms << ",\n";
    out << "    \"lasso_family_commitments\": " << record.lasso_family_commitment_ms << ",\n";
    out << "    \"lasso_family_selected_row_mapping\": " << record.lasso_family_selected_row_mapping_ms << ",\n";
    out << "    \"commitment\": " << record.commitment_ms << ",\n";
    out << "    \"prove_total\": " << record.prove_total_ms << ",\n";
    out << "    \"self_verify\": " << record.self_verify_ms << ",\n";
    out << "    \"prove\": " << record.prove_ms << ",\n";
    out << "    \"verify\": " << record.verify_ms << ",\n";
    out << "    \"initial_commitment_blinding_prepare\": " << record.initial_commitment_blinding_prepare_ms << ",\n";
    out << "    \"serialize\": " << record.serialize_ms << ",\n";
    out << "    \"artifact_wire_self_check\": " << record.artifact_wire_self_check_ms << ",\n";
    out << "    \"artifact_write\": " << record.artifact_write_ms << ",\n";
    out << "    \"artifact_readback\": " << record.artifact_readback_ms << "\n";
    out << "  }\n";
    out << "}\n";
    write_text(path, out.str());
}

void write_logits(const std::string& path, const Tensor2D& logits) {
    std::ostringstream out;
    out << logits.rows << ' ' << logits.cols << '\n';
    for (std::size_t row = 0; row < logits.rows; ++row) {
        for (std::size_t col = 0; col < logits.cols; ++col) {
            if (col != 0) out << ' ';
            out << logits.at(row, col);
        }
        out << '\n';
    }
    write_text(path, out.str());
}

void write_registration(const std::string& path, const ModelRegistration& registration, const std::string& root, const std::string& dataset) {
    std::ostringstream out;
    out << "{\n";
    json_string(out, "schema", "zkgpt-gat-model-registration-v2");
    json_string(out, "plan_recipe", "canonical_plan_recipe_v3");
    json_string(out, "dataset", dataset);
    json_string(out, "status", registration.status);

    json_bool(out, "registration_only", true);
    json_bool(out, "forward_executed", false);
    json_bool(out, "plan_materialized", false);
    json_string(out, "layout_source", "canonical_metadata");
    json_string(out, "model_arch_id", registration.model_arch_id);
    json_string(out, "quant_cfg_id", registration.quant_cfg_id);
    json_string(out, "static_tables_id", registration.static_tables_id);
    json_string(out, "field_id", registration.field_id);
    json_string(out, "field_modulus", registration.field_modulus);
    json_string(out, "manifest_digest", registration.manifest_digest);
    json_string(out, "model_digest", registration.model_digest);
    json_string(out, "bounds_digest", registration.bounds_digest);
    json_string(out, "quant_digest", registration.quant_digest);
    json_string(out, "layout_digest", registration.layout_digest);
    json_string(out, "selector", registration.selector);
    json_string(out, "opening_policy", registration.opening_policy);
    json_string(out, "binding_digest", registration.binding_digest);
    json_string(out, "registration_digest", registration.registration_digest);
    out << "  \"n_raw\": " << registration.n_raw << ",\n";
    out << "  \"n_in\": " << registration.n_in << ",\n";
    out << "  \"input_segments\": [\n";
    for (std::size_t index = 0; index < registration.input_segments.size(); ++index) {
        const InputSegment& segment = registration.input_segments[index];
        out << "    {\n";
        json_string(out, "name", segment.name);
        out << "      \"offset\": " << segment.offset << ",\n";
        out << "      \"logical_length\": " << segment.logical_length << ",\n";
        out << "      \"padded_length\": " << segment.padded_length << "\n";
        out << "    }" << (index + 1 == registration.input_segments.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    json_string(out, "commitment_scheme", registration.com_W.scheme);
    json_string(out, "basis_id", registration.com_W.basis_id);
    out << "  \"logical_size\": " << registration.com_W.logical_size << ",\n";
    out << "  \"padded_size\": " << registration.com_W.padded_size << ",\n";
    json_string(out, "point_hex", registration.com_W.point_hex, false);
    out << "}\n";
    (void)root;
    write_text(path, out.str());
}

void write_capacity_report(const GatCapacityReport& report, const std::string& dataset, const std::string& backend, const std::string& quant_cfg_id,
                           const std::string& cache_identity, const std::string& checkpoint_identity,
                           const Graph* graph = nullptr, const CanonicalConfig* config = nullptr,
                           bool analysis_only = false) {
    std::ostringstream out;
    out << "{\n";
    json_string(out, "schema", report.schema);
    json_string(out, "profile_kind", analysis_only ? "analysis_only" : "formal_admission");
    json_string(out, "profile_binary_sha256", sha256_file_digest("/proc/self/exe"));
    json_string(out, "dataset", dataset);
    json_string(out, "backend", backend);
    json_string(out, "quant_cfg_id", quant_cfg_id);
    json_string(out, "cache_identity", cache_identity);
    json_string(out, "checkpoint_identity", checkpoint_identity);
    json_string(out, "memory_accounting_kind", report.memory_accounting_kind);
    out << "  \"device_peak_bytes\": " << report.device_peak_bytes << ",\n";
    const auto number = [&out](const char* key, const auto value, bool comma = true) { out << "  \"" << key << "\": " << value << (comma ? ",\n" : "\n"); };
    number("num_nodes", report.num_nodes);
    number("num_edges", report.num_edges);
    number("num_features", report.num_features);
    number("num_classes", report.num_classes);
    number("graph_count", report.graph_count);
    json_string(out, "model_arch_id", report.model_shape.model_arch_id);
    number("model_layers", report.model_shape.layers);
    number("model_output_heads", report.model_shape.output_heads);
    number("model_classes", report.model_shape.classes);
    number("model_input_dim", report.model_shape.input_dim);
    number("model_hidden_dim", report.model_shape.hidden_dim);
    number("n_raw", report.input_layout.n_raw);
    number("n_in", report.input_layout.n_in);
    out << "  \"input_segments\": [";
    for (std::size_t index = 0; index < report.input_layout.segments.size(); ++index) {
        const InputSegment& segment = report.input_layout.segments[index];
        if (index != 0) out << ',';
        out << "{\"name\":\"" << json_escape(segment.name)
            << "\",\"offset\":" << segment.offset
            << ",\"logical_length\":" << segment.logical_length
            << ",\"padded_length\":" << segment.padded_length << '}';
    }
    out << "],\n";
    number("explicit_gate_count", report.explicit_gate_count);
    number("max_materialized_gate_bytes", report.max_materialized_gate_bytes);
    number("max_materialized_vector_bytes", report.max_materialized_vector_bytes);
    number("context_recipe_bytes_bound", report.context_recipe_bytes_bound);
    number("context_payload_bytes_bound", report.context_payload_bytes_bound);
    json_bool(out, "ppi_node_ptr_boundary", report.ppi_node_ptr_boundary);
    json_bool(out, "ppi_edge_ptr_boundary", report.ppi_edge_ptr_boundary);
    json_bool(out, "symbolic_analysis_profile_complete", report.symbolic_analysis_profile_complete);
    json_bool(out, "gkr_point_shape_exact", report.gkr_point_shape_exact);
    json_bool(out, "gkr_point_domain_upper_bound_valid", report.gkr_point_domain_upper_bound_valid);
    number("lasso_logical_domain_entries", report.lasso_logical_domain_entries);
    number("lasso_private_selected_row_workspace_peak_bytes", report.lasso_private_selected_row_workspace_peak_bytes);
    number("lasso_selected_row_cache_bytes_upper_bound", report.lasso_selected_row_cache_bytes_upper_bound);
    number("lasso_selected_row_cache_heap_bytes_upper_bound", report.lasso_selected_row_cache_heap_bytes_upper_bound);
    number("lasso_selected_row_cache_mapped_bytes_upper_bound", report.lasso_selected_row_cache_mapped_bytes_upper_bound);
    number("lasso_sparse_matrix_workspace_bytes_upper_bound", report.lasso_sparse_matrix_workspace_bytes_upper_bound);
    number("lasso_heap_workspace_peak_bytes", report.lasso_heap_workspace_peak_bytes);
    number("hyrax_active_domain_workspace_entries", report.hyrax_active_domain_workspace_entries);
    number("hyrax_active_domain_workspace_bytes", report.hyrax_active_domain_workspace_bytes);
    number("hyrax_l_qry_tensor_columns", report.hyrax_l_qry_tensor_columns);
    number("hyrax_l_qry_tensor_workspace_bytes", report.hyrax_l_qry_tensor_workspace_bytes);
    number("hyrax_l_qry_binary_lut_workspace_bytes", report.hyrax_l_qry_binary_lut_workspace_bytes);
    number("hyrax_aux_tensor_workspace_bytes", report.hyrax_aux_tensor_workspace_bytes);
    number("hyrax_folded_columns_workspace_bytes", report.hyrax_folded_columns_workspace_bytes);
    number("hyrax_tensor_retained_workspace_bytes", report.hyrax_tensor_retained_workspace_bytes);
    number("hyrax_tensor_opening_transient_peak_workspace_bytes", report.hyrax_tensor_opening_transient_peak_workspace_bytes);
    number("hyrax_tensor_combined_peak_workspace_bytes", report.hyrax_tensor_combined_peak_workspace_bytes);
    number("hyrax_w_mle_workspace_bytes", report.hyrax_w_mle_workspace_bytes);
    number("borrowed_trace_bytes_upper_bound", report.borrowed_trace_bytes_upper_bound);
    number("retained_relation_witness_entries", report.retained_relation_witness_entries);
    number("retained_relation_witness_bytes_upper_bound", report.retained_relation_witness_bytes_upper_bound);
    number("relation_public_output_entries", report.relation_public_output_entries);
    number("relation_public_output_bytes_upper_bound", report.relation_public_output_bytes_upper_bound);
    number("relation_residual_entries", report.relation_residual_entries);
    number("relation_residual_bytes_upper_bound", report.relation_residual_bytes_upper_bound);
    number("retained_layer_witness_entries_upper_bound", report.retained_layer_witness_entries_upper_bound);
    number("retained_layer_witness_bytes_upper_bound", report.retained_layer_witness_bytes_upper_bound);
    number("evaluated_layer_witness_entries_upper_bound", report.evaluated_layer_witness_entries_upper_bound);
    number("evaluated_layer_witness_bytes_upper_bound", report.evaluated_layer_witness_bytes_upper_bound);
    number("native_gkr_witness_heap_bytes_upper_bound", report.native_gkr_witness_heap_bytes_upper_bound);
    number("native_gkr_witness_mapped_bytes_upper_bound", report.native_gkr_witness_mapped_bytes_upper_bound);
    number("relation_gate_unary_entries", report.relation_gate_unary_entries);
    number("relation_gate_binary_entries", report.relation_gate_binary_entries);
    number("relation_gate_output_entries", report.relation_gate_output_entries);
    number("max_single_gate_array_bytes_upper_bound", report.max_single_gate_array_bytes_upper_bound);
    number("explicit_unary_gate_buffer_entry_bytes", report.explicit_unary_gate_buffer_entry_bytes);
    number("explicit_unary_gate_buffer_entries_upper_bound", report.explicit_unary_gate_buffer_entries_upper_bound);
    number("max_explicit_unary_gate_buffer_bytes_upper_bound", report.max_explicit_unary_gate_buffer_bytes_upper_bound);
    number("explicit_unary_gate_buffer_compact_coexist_bytes_upper_bound",
           report.explicit_unary_gate_buffer_compact_coexist_bytes_upper_bound);
    number("max_single_relation_builder_array_bytes_upper_bound", report.max_single_relation_builder_array_bytes_upper_bound);
    number("gkr_subset_source_map_bytes_upper_bound", report.gkr_subset_source_map_bytes_upper_bound);
    number("gkr_subset_sort_scratch_bytes_upper_bound", report.gkr_subset_sort_scratch_bytes_upper_bound);
    number("gkr_subset_bitmap_scratch_bytes_upper_bound", report.gkr_subset_bitmap_scratch_bytes_upper_bound);
    number("gkr_subset_rank_scratch_bytes_upper_bound", report.gkr_subset_rank_scratch_bytes_upper_bound);
    number("gkr_input_fold_cache_late_rounds", report.gkr_input_fold_cache_late_rounds);
    number("gkr_input_fold_cache_scalar_bytes", report.gkr_input_fold_cache_scalar_bytes);
    number("gkr_input_fold_cache_raw_bytes_upper_bound", report.gkr_input_fold_cache_raw_bytes_upper_bound);
    number("gkr_input_fold_cache_mapped_bytes_upper_bound", report.gkr_input_fold_cache_mapped_bytes_upper_bound);
    number("gkr_input_fold_cache_heap_bytes_upper_bound", report.gkr_input_fold_cache_heap_bytes_upper_bound);
    number("gkr_input_fold_cache_page_size_bytes", report.gkr_input_fold_cache_page_size_bytes);
    number("gkr_point_shape_bitmap_scratch_bytes", report.gkr_point_shape_bitmap_scratch_bytes);
    number("gkr_point_shape_touched_index_scratch_bytes_upper_bound",
           report.gkr_point_shape_touched_index_scratch_bytes_upper_bound);
    number("gkr_point_shape_layer_count", report.gkr_point_shape_layer_count);
    number("gkr_point_shape_fconn_count", report.gkr_point_shape_fconn_count);
    json_string(out, "gkr_point_shape_digest", report.gkr_point_shape_digest);
    json_bool(out, "gkr_point_shape_ready", report.gkr_point_shape_ready);
    number("gkr_subset_source_support_entries_upper_bound", report.gkr_subset_source_support_entries_upper_bound);
    number("gkr_layer0_source_domain_entries_upper_bound", report.gkr_layer0_source_domain_entries_upper_bound);
    number("gkr_layer0_lqry_padded_block_entries_upper_bound", report.gkr_layer0_lqry_padded_block_entries_upper_bound);
    number("gkr_layer0_lqry_zero_gap_entries", report.gkr_layer0_lqry_zero_gap_entries);
    number("native_transcript_heap_bytes_upper_bound", report.native_transcript_heap_bytes_upper_bound);
    number("transcript_content_bytes_upper_bound", report.transcript_content_bytes_upper_bound);
    number("transcript_wire_bytes_upper_bound", report.transcript_wire_bytes_upper_bound);
    number("transcript_prefix_bytes_upper_bound", report.transcript_prefix_bytes_upper_bound);
    number("maximum_challenge_suffix_bytes_upper_bound", report.maximum_challenge_suffix_bytes_upper_bound);
    number("transcript_metadata_bytes_upper_bound", report.transcript_metadata_bytes_upper_bound);
    number("zero_tail_certificate_count_upper_bound", report.zero_tail_certificate_count_upper_bound);
    number("l_query_zero_interval_message_count_upper_bound", report.l_query_zero_interval_message_count_upper_bound);
    number("lasso_table_and_index_bytes_upper_bound", report.lasso_table_and_index_bytes_upper_bound);
    number("gkr_layer_count_upper_bound", report.gkr_layer_count_upper_bound);
    number("gkr_fconn_layer_count_upper_bound", report.gkr_fconn_layer_count_upper_bound);
    json_bool(out, "gkr_work_counts_exact", report.gkr_work_counts_exact);
    number("gkr_phase1_rounds_upper_bound", report.gkr_phase1_rounds_upper_bound);
    number("gkr_phase2_rounds_upper_bound", report.gkr_phase2_rounds_upper_bound);
    number("gkr_phase1_first_round_pair_visits_upper_bound", report.gkr_phase1_first_round_pair_visits_upper_bound);
    number("gkr_phase1_followup_pair_visits_upper_bound", report.gkr_phase1_followup_pair_visits_upper_bound);
    number("gkr_phase1_total_pair_visits_upper_bound", report.gkr_phase1_total_pair_visits_upper_bound);
    number("gkr_phase1_extent_visits_upper_bound", report.gkr_phase1_extent_visits_upper_bound);
    number("gkr_phase1_followup_extent_visits_upper_bound", report.gkr_phase1_followup_extent_visits_upper_bound);
    number("gkr_phase1_first_round_field_adds_upper_bound", report.gkr_phase1_first_round_field_adds_upper_bound);
    number("gkr_phase1_first_round_field_subs_upper_bound", report.gkr_phase1_first_round_field_subs_upper_bound);
    number("gkr_phase1_first_round_field_muls_upper_bound", report.gkr_phase1_first_round_field_muls_upper_bound);
    number("gkr_phase1_followup_field_adds_upper_bound", report.gkr_phase1_followup_field_adds_upper_bound);
    number("gkr_phase1_followup_field_subs_upper_bound", report.gkr_phase1_followup_field_subs_upper_bound);
    number("gkr_phase1_followup_field_muls_upper_bound", report.gkr_phase1_followup_field_muls_upper_bound);
    number("gkr_phase1_init_active_scalar_assignments", report.gkr_phase1_init_field_writes);
    number("gkr_phase2_first_round_pair_visits_upper_bound", report.gkr_phase2_first_round_pair_visits_upper_bound);
    number("gkr_phase2_followup_pair_visits_upper_bound", report.gkr_phase2_followup_pair_visits_upper_bound);
    number("gkr_phase2_total_pair_visits_upper_bound", report.gkr_phase2_total_pair_visits_upper_bound);
    number("gkr_phase2_extent_visits_upper_bound", report.gkr_phase2_extent_visits_upper_bound);
    number("gkr_phase2_followup_extent_visits_upper_bound", report.gkr_phase2_followup_extent_visits_upper_bound);
    number("gkr_phase2_first_round_field_adds_upper_bound", report.gkr_phase2_first_round_field_adds_upper_bound);
    number("gkr_phase2_first_round_field_subs_upper_bound", report.gkr_phase2_first_round_field_subs_upper_bound);
    number("gkr_phase2_first_round_field_muls_upper_bound", report.gkr_phase2_first_round_field_muls_upper_bound);
    number("gkr_phase2_followup_field_adds_upper_bound", report.gkr_phase2_followup_field_adds_upper_bound);
    number("gkr_phase2_followup_field_subs_upper_bound", report.gkr_phase2_followup_field_subs_upper_bound);
    number("gkr_phase2_followup_field_muls_upper_bound", report.gkr_phase2_followup_field_muls_upper_bound);
    number("gkr_phase2_init_active_scalar_assignments", report.gkr_phase2_init_field_writes);
    number("fconn_scalar_multiply_terms", report.fconn_scalar_multiply_terms);
    number("fconn_output_coordinates", report.fconn_output_coordinates);
    number("lasso_total_rows", report.lasso_total_rows);
    number("lasso_total_row_commitment_chunks", report.lasso_total_row_commitment_chunks);
    number("lasso_real_query_entries", report.lasso_real_query_entries);
    number("lasso_padded_query_entries", report.lasso_padded_query_entries);
    number("lasso_family_zero_pad_entries", report.lasso_family_zero_pad_entries);
    number("hyrax_matrix_masked_openings", report.hyrax_matrix_masked_openings);
    number("hyrax_l_qry_tensor_openings", report.hyrax_l_qry_tensor_openings);
    number("gkr_input_source_count_upper_bound", report.gkr_input_source_count_upper_bound);
    number("gkr_message_count_upper_bound", report.gkr_message_count_upper_bound);
    number("gkr_wire_bytes_upper_bound", report.gkr_wire_bytes_upper_bound);
    number("gkr_prefix_workspace_bytes_upper_bound", report.gkr_prefix_workspace_bytes_upper_bound);
    number("gkr_prefix_filesystem_allocation_unit_bytes", report.gkr_prefix_filesystem_allocation_unit_bytes);
    number("l_query_zero_interval_count_upper_bound", report.l_query_zero_interval_count_upper_bound);
    number("estimated_peak_host_memory_bytes", report.estimated_peak_host_memory_bytes);
    number("forward_admission_peak_host_memory_bytes_upper_bound", report.forward_admission_peak_host_memory_bytes_upper_bound);
    json_bool(out, "forward_value_admission_ready", report.forward_value_admission_ready);
    json_string(out, "forward_value_admission_mode", report.forward_value_admission_mode);
    number("estimated_native_scratch_bytes", report.estimated_native_scratch_bytes);
    number("estimated_native_heap_bytes_upper_bound", report.estimated_native_heap_bytes_upper_bound);
    number("estimated_native_mapped_bytes_upper_bound", report.estimated_native_mapped_bytes_upper_bound);
    number("max_materialized_fr_bytes", report.max_materialized_fr_bytes);
    number("max_materialized_g1_bytes", report.max_materialized_g1_bytes);
    number("flattened_input_domain_entries", report.flattened_input_domain_entries);
    number("flattened_input_domain_limit_u64", report.flattened_input_domain_limit_u64);
    number("disk_requirement_bytes_upper_bound", report.disk_requirement_bytes_upper_bound);
    number("proof_message_count_upper_bound", report.proof_message_count_upper_bound);
    number("maximum_transcript_field_entries_upper_bound", report.maximum_transcript_field_entries_upper_bound);
    number("total_transcript_field_entries_upper_bound", report.total_transcript_field_entries_upper_bound);
    number("maximum_message_string_bytes_upper_bound", report.maximum_message_string_bytes_upper_bound);
    number("proof_artifact_wire_bytes_upper_bound", report.proof_artifact_wire_bytes_upper_bound);
    number("artifact_disk_bytes_upper_bound", report.artifact_disk_bytes_upper_bound);
    number("host_memory_limit_bytes", report.host_memory_limit_bytes);
    number("native_scratch_available_bytes", report.native_scratch_available_bytes);
    number("coordinate_domain_limit_u64", report.coordinate_domain_limit_u64);
    number("proof_message_count_limit", report.proof_message_count_limit);
    number("maximum_message_string_bytes_limit", report.maximum_message_string_bytes_limit);
    number("proof_artifact_wire_limit_bytes", report.proof_artifact_wire_limit_bytes);
    number("artifact_disk_available_bytes", report.artifact_disk_available_bytes);
    json_bool(out, "scratch_artifact_same_filesystem", report.scratch_artifact_same_filesystem);
    number("shared_scratch_artifact_available_bytes", report.shared_scratch_artifact_available_bytes);
    out << "  \"native_mapped_phase_breakdown\": {";
    for (std::size_t index = 0; index < report.native_mapped_phase_breakdown.size(); ++index) {
        const auto& entry = report.native_mapped_phase_breakdown[index];
        if (index != 0) out << ',';
        out << "\n    \"" << json_escape(entry.first) << "\": " << entry.second;
    }
    if (!report.native_mapped_phase_breakdown.empty()) out << '\n';
    out << "  },\n";
    out << "  \"lasso_workspace_profile\": [";
    for (std::size_t index = 0; index < report.lasso_workspace_profile.size(); ++index) {
        const GatLassoWorkspaceProfile& family = report.lasso_workspace_profile[index];
        const NativeLassoCapacityProfile& workspace = family.workspace;
        if (index != 0) out << ',';
        out << "{\"family_id\":\"" << json_escape(family.family_id) << "\",\"workspace\":{";
        bool first = true;
        const auto workspace_number = [&out, &first](const char* key, const std::uint64_t value) {
            if (!first) out << ',';
            first = false;
            out << '\"' << key << "\":" << value;
        };
        workspace_number("padded_queries", workspace.padded_queries);
        workspace_number("table_domain", workspace.table_domain);
        workspace_number("full_domain", workspace.full_domain);
        workspace_number("rows", workspace.rows);
        workspace_number("columns", workspace.columns);
        workspace_number("row_commitment_chunk_count", workspace.row_commitment_chunk_count);
        workspace_number("booleanity_rounds", workspace.booleanity_rounds);
        workspace_number("onehot_rounds", workspace.onehot_rounds);
        workspace_number("lookup_rounds", workspace.lookup_rounds);
        workspace_number("hyrax_matrix_masked_openings", workspace.hyrax_matrix_masked_openings);
        workspace_number("hyrax_l_qry_tensor_openings", workspace.hyrax_l_qry_tensor_openings);
        if (!first) out << ',';
        first = false;
        out << "\"t2_booleanity\":" << bool_json(workspace.t2_booleanity);
        workspace_number("private_witness_entries", workspace.private_witness_entries);
        workspace_number("private_witness_element_bytes", workspace.private_witness_element_bytes);
        workspace_number("private_witness_bytes", workspace.private_witness_bytes);
        workspace_number("row_blinding_bytes", workspace.row_blinding_bytes);
        workspace_number("row_commitment_bytes", workspace.row_commitment_bytes);
        workspace_number("column_generator_bytes", workspace.column_generator_bytes);
        workspace_number("row_fold_weights_bytes", workspace.row_fold_weights_bytes);
        workspace_number("row_fold_values_bytes", workspace.row_fold_values_bytes);
        workspace_number("onehot_query_bytes", workspace.onehot_query_bytes);
        workspace_number("t2_column_one_bytes", workspace.t2_column_one_bytes);
        workspace_number("onehot_heap_peak_bytes", workspace.onehot_heap_peak_bytes);
        workspace_number("masked_opening_generator_bytes", workspace.masked_opening_generator_bytes);
        workspace_number("masked_opening_coefficient_bytes", workspace.masked_opening_coefficient_bytes);
        workspace_number("masked_opening_response_bytes", workspace.masked_opening_response_bytes);
        workspace_number("row_ledger_retained_bytes", workspace.row_ledger_retained_bytes);
        workspace_number("commit_peak_bytes", workspace.commit_peak_bytes);
        workspace_number("booleanity_peak_bytes", workspace.booleanity_peak_bytes);
        workspace_number("row_fold_peak_bytes", workspace.row_fold_peak_bytes);
        workspace_number("opening_peak_bytes", workspace.opening_peak_bytes);
        workspace_number("family_peak_bytes", workspace.family_peak_bytes);
        out << "},\"logical_domain_entries\":" << family.logical_domain_entries
            << ",\"row_commitment_chunk_count\":" << family.row_commitment_chunk_count
            << ",\"row_commitment_chunk_entries\":" << family.row_commitment_chunk_entries
            << ",\"booleanity_rounds\":" << family.booleanity_rounds
            << ",\"a_query_challenges\":" << family.a_query_challenges
            << ",\"onehot_rounds\":" << family.onehot_rounds
            << ",\"lookup_rounds\":" << family.lookup_rounds
            << ",\"hyrax_matrix_masked_openings\":" << family.hyrax_matrix_masked_openings
            << ",\"hyrax_l_qry_tensor_openings\":" << family.hyrax_l_qry_tensor_openings
            << ",\"row_commitment_wire_bytes_upper_bound\":" << family.row_commitment_wire_bytes_upper_bound
            << ",\"family_message_count_upper_bound\":" << family.family_message_count_upper_bound
            << ",\"family_wire_bytes_upper_bound\":" << family.family_wire_bytes_upper_bound << '}';
    }
    out << "],\n";
    out << "  \"gkr_layer_work_profile\": [";
    for (std::size_t index = 0; index < report.gkr_layer_work_profile.size(); ++index) {
        const GatGkrLayerShapeProfile& layer = report.gkr_layer_work_profile[index];
        if (index != 0) out << ',';
        out << "{\"layer_type\":" << layer.layer_type
            << ",\"size\":" << layer.size
            << ",\"size_u0\":" << layer.size_u0
            << ",\"size_u1\":" << layer.size_u1
            << ",\"size_v0\":" << layer.size_v0
            << ",\"size_v1\":" << layer.size_v1
            << ",\"unary_gates\":" << layer.unary_gates
            << ",\"binary_gates\":" << layer.binary_gates
            << ",\"native_rows\":" << layer.native_rows
            << ",\"native_inner\":" << layer.native_inner
            << ",\"native_cols\":" << layer.native_cols
            << ",\"bit_length\":" << layer.bit_length
            << ",\"max_bl_u\":" << layer.max_bl_u
            << ",\"max_bl_v\":" << layer.max_bl_v
            << ",\"need_phase2\":" << bool_json(layer.need_phase2)
            << ",\"source_domains_exact\":" << bool_json(layer.source_domains_exact)
            << ",\"phase1_rounds\":" << layer.phase1_rounds
            << ",\"phase1_first_round_pair_visits\":" << layer.phase1_first_round_pair_visits
            << ",\"phase1_followup_pair_visits\":" << layer.phase1_followup_pair_visits
            << ",\"phase1_total_pair_visits\":" << layer.phase1_total_pair_visits
            << ",\"phase1_extent_visits\":" << layer.phase1_extent_visits
            << ",\"phase1_followup_extent_visits\":" << layer.phase1_followup_extent_visits
            << ",\"phase1_first_round_field_adds_upper_bound\":" << layer.phase1_first_round_field_adds_upper_bound
            << ",\"phase1_first_round_field_subs_upper_bound\":" << layer.phase1_first_round_field_subs_upper_bound
            << ",\"phase1_first_round_field_muls_upper_bound\":" << layer.phase1_first_round_field_muls_upper_bound
            << ",\"phase1_followup_field_adds_upper_bound\":" << layer.phase1_followup_field_adds_upper_bound
            << ",\"phase1_followup_field_subs_upper_bound\":" << layer.phase1_followup_field_subs_upper_bound
            << ",\"phase1_followup_field_muls_upper_bound\":" << layer.phase1_followup_field_muls_upper_bound
            << ",\"phase1_init_active_scalar_assignments\":" << layer.phase1_init_field_writes
            << ",\"phase2_rounds\":" << layer.phase2_rounds
            << ",\"phase2_first_round_pair_visits\":" << layer.phase2_first_round_pair_visits
            << ",\"phase2_followup_pair_visits\":" << layer.phase2_followup_pair_visits
            << ",\"phase2_total_pair_visits\":" << layer.phase2_total_pair_visits
            << ",\"phase2_extent_visits\":" << layer.phase2_extent_visits
            << ",\"phase2_followup_extent_visits\":" << layer.phase2_followup_extent_visits
            << ",\"phase2_first_round_field_adds_upper_bound\":" << layer.phase2_first_round_field_adds_upper_bound
            << ",\"phase2_first_round_field_subs_upper_bound\":" << layer.phase2_first_round_field_subs_upper_bound
            << ",\"phase2_first_round_field_muls_upper_bound\":" << layer.phase2_first_round_field_muls_upper_bound
            << ",\"phase2_followup_field_adds_upper_bound\":" << layer.phase2_followup_field_adds_upper_bound
            << ",\"phase2_followup_field_subs_upper_bound\":" << layer.phase2_followup_field_subs_upper_bound
            << ",\"phase2_followup_field_muls_upper_bound\":" << layer.phase2_followup_field_muls_upper_bound
            << ",\"phase2_init_active_scalar_assignments\":" << layer.phase2_init_field_writes << '}';
    }
    out << "],\n";
    json_string(out, "proof_path", report.proof_path);
    json_bool(out, "full_relation_bound", report.full_relation_bound);
    json_bool(out, "ready", report.ready);
    json_string(out, "status", analysis_only ? "ANALYSIS_ONLY" : (report.ready ? "READY" : "BLOCKED"));
    json_string(out, "reason", report.reason, graph != nullptr);
    if (graph != nullptr) {
        if (config == nullptr) throw std::invalid_argument("canonical graph identity requires its consumed config");
        const std::string input_preprocess_id = config->get("input_preprocess_id");
        const std::string batching_rule = config->get("batching_rule");
        if (input_preprocess_id.empty() || batching_rule.empty())
            throw std::invalid_argument("canonical graph identity config is missing preprocessing or batching semantics");
        out << "  \"canonical_graph_identity\": {\n";
        out << "    \"digest\": \"" << comparison_canonical_graph_digest(*graph) << "\",\n";
        out << "    \"nodes\": " << graph->num_nodes << ",\n";
        out << "    \"edges\": " << graph->edges() << ",\n";
        out << "    \"graphs\": " << graph->graph_count << ",\n";
        json_size_array(out, "node_ptr", graph->node_ptr);
        out << ",\n";
        json_size_array(out, "edge_ptr", graph->edge_ptr);
        out << ",\n";
        out << "    \"input_preprocess_id\": \"" << json_escape(input_preprocess_id) << "\",\n";
        out << "    \"batching_rule\": \"" << json_escape(batching_rule) << "\"\n";
        out << "  }\n";
    }
    out << "}\n";
    std::cout << out.str();
}

void write_public_statement(const std::string& path, const Record& record, const Graph& graph, const Tensor2D& features, const QuantConfig& quant,
                            const StaticTables& tables, const GatCircuitPlan& plan) {
    std::ostringstream out;
    out << "{\n";
    json_string(out, "schema", "zkgpt-gat-public-statement-v1");
    json_string(out, "relation_id", "R_GAT-zkGPT");
    json_string(out, "dataset", record.dataset);
    json_string(out, "backend", record.backend);
    json_string(out, "proof_task", "node_classification_full_gat");
    json_string(out, "proof_scope", "full_gat");
    json_string(out, "privacy_policy", "model_and_trace_hidden");
    json_string(out, "privacy_extension_profile_id", "masked_full");
    json_string(out, "input_privacy", "public_graph_public_features");
    json_string(out, "output_mode", "public_logits");
    json_string(out, "quant_cfg_id", quant.quant_cfg_id);
    json_string(out, "static_tables_id", tables.table_id);
    json_string(out, "static_tables_digest", tables.digest);
    json_string(out, "registration_digest", record.registration_digest);
    json_string(out, "registration_layout_digest", record.registration_layout_digest);
    json_string(out, "registration_binding_digest", record.registration_binding_digest);
    json_string(out, "circuit_id", plan.circuit_id);
    json_string(out, "plan_digest", plan.plan_digest);
    json_string(out, "statement_digest", record.statement_digest);
    out << "  \"num_nodes\": " << graph.num_nodes << ",\n";
    out << "  \"num_edges\": " << graph.edges() << ",\n";
    out << "  \"graph_count\": " << graph.graph_count << ",\n";
    out << "  \"num_features\": " << features.cols << ",\n";
    out << "  \"num_classes\": " << graph.num_classes << ",\n";
    json_string(out, "com_W_point", record.com_w_point, false);
    out << "}\n";
    write_text(path, out.str());
}

void write_formal_verifier_context(const std::string& path, const detail::ValidatedFormalInputs& inputs, const VerifierContext& context) {
    const auto write_begin = Clock::now();
    std::cerr << "gat_adapter_prover_stage: formal-context-write begin" << std::endl;
    std::string reason;
    if (!detail::write_verifier_context_formal(inputs, context, path, &reason)) {
        throw std::runtime_error("cannot write verifier context v4: " + (reason.empty() ? std::string("unknown error") : reason));
    }
    std::cerr << "gat_adapter_prover_stage: formal-context-write complete elapsed_ms="
              << elapsed_ms(write_begin, Clock::now()) << std::endl;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            options.help = true;
        } else if (arg == "--dataset" && i + 1 < argc) {
            options.dataset = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            options.backend = argv[++i];
        } else if (arg == "--baseline-root" && i + 1 < argc) {
            options.baseline_root = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            options.config = argv[++i];
        } else if (arg == "--registration" && i + 1 < argc) {
            options.registration = argv[++i];
        } else if (arg == "--com-w-secret" && i + 1 < argc) {
            options.com_w_secret = argv[++i];
        } else if (arg == "--formal-session-id" && i + 1 < argc) {
            options.formal_session_id = argv[++i];
        } else if (arg == "--run-nonce" && i + 1 < argc) {
            options.run_nonce = argv[++i];
        } else if (arg == "--session-manifest-sha256" && i + 1 < argc) {
            options.session_manifest_sha256 = argv[++i];
        } else if (arg == "--formal-run-directory" && i + 1 < argc) {
            options.formal_run_directory = argv[++i];
        } else if (arg == "--register-only") {
            options.register_only = true;
        } else if (arg == "--capacity-only") {
            options.capacity_only = true;
        } else if (arg == "--capacity-profile-only") {
            options.capacity_profile_only = true;
        } else if (arg == "--defer-self-verify") {
            options.defer_self_verify = true;
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (options.help) return options;
    const std::vector<std::string> datasets = {"cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv"};
    bool known_dataset = false;
    for (const std::string& dataset : datasets)
        if (options.dataset == dataset) known_dataset = true;
    if (!known_dataset) throw std::runtime_error("--dataset must be one of cora,citeseer,pubmed,ppi,ogbn_arxiv");
    if (options.backend != "cpu" && options.backend != "gpu") throw std::runtime_error("--backend must be cpu or gpu");
    if (options.baseline_root.empty()) throw std::runtime_error("--baseline-root is empty");
    const bool any_session_binding = !options.formal_session_id.empty() || !options.run_nonce.empty() ||
        !options.session_manifest_sha256.empty() || !options.formal_run_directory.empty();
    if (options.capacity_only && options.capacity_profile_only)
        throw std::runtime_error("formal capacity admission and analysis-only capacity profile are separate modes");
    const bool capacity_mode = options.capacity_only || options.capacity_profile_only;
    if (capacity_mode && any_session_binding)
        throw std::runtime_error("capacity modes do not accept formal session outputs");
    if (options.defer_self_verify && (!any_session_binding || options.register_only || capacity_mode))
        throw std::runtime_error("deferred native verification is only available for a bound formal proof run");
    if (!capacity_mode && !any_session_binding)
        throw std::runtime_error("formal proof/registration requires an explicit session binding");
    if (any_session_binding) {
        const auto is_hex = [](const std::string& value, std::size_t expected) {
            return value.size() == expected && std::all_of(value.begin(), value.end(), [](unsigned char ch) {
                return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
            });
        };
        if (!is_hex(options.formal_session_id, 32) || !is_hex(options.run_nonce, 32) ||
            !is_hex(options.session_manifest_sha256, 64) ||
            options.formal_run_directory != "baseline-local/runs/formal/" + options.formal_session_id + "/" + options.dataset) {
            throw std::runtime_error("formal session binding is incomplete or noncanonical");
        }
    }
    if (options.config.empty()) options.config = "baseline-local/configs/" + options.dataset + "_secure.cfg";
    return options;
}

void print_help() {
    std::cout << "Usage: gat_cli --dataset <cora|citeseer|pubmed|ppi|ogbn_arxiv> "
                 "--backend <cpu|gpu> [--baseline-root <zkGPT-root>] [--config <baseline-relative.cfg>] "
                 "[--registration <baseline-relative.json>] [--com-w-secret <baseline-relative.secret>]\n"
                 "       gat_cli ... --register-only\n"
                 "       gat_cli --capacity-only --dataset <...> --backend cpu [--baseline-root <zkGPT-root>] [--config <...>]\n"
                 "       gat_cli --capacity-profile-only --dataset <...> --backend cpu [--baseline-root <zkGPT-root>] [--config <...>]\n"
                 "       --defer-self-verify leaves a pending result for the formal runner's independent verifier.\n"
                 "The normal command emits one baseline-local run record and exits nonzero on FAILED or UNSUPPORTED.\n"
                 "--register-only writes the long-lived public model registration and its prover-only 0600 secret.\n";
}

}

int main(int argc, char** argv) {
    Record record;
    std::string result_file;
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            print_help();
            return 0;
        }
        record.dataset = options.dataset;
        record.backend = options.backend;
        record.session_id = options.formal_session_id;
        record.run_nonce = options.run_nonce;
        record.session_manifest_sha256 = options.session_manifest_sha256;
        record.native_backend = options.backend == "gpu" ? "cuda_formal_masked_full" : "cpu_native_zkgpt";
        if (options.backend == "gpu") {
            record.capability_status = "UNSUPPORTED_FORMAL_MASKED_FULL_GPU";
            record.formal_backend_supported = false;
#if defined(ZKGPT_HAS_CUDA) && ZKGPT_HAS_CUDA
            record.gpu_compiled = true;
#endif
        }
        const std::string root = without_trailing_slash(options.baseline_root);
        const bool session_bound_run = !options.formal_session_id.empty();
        const std::string fallback_run_rel = "baseline-local/runs/" + options.dataset + "_secure/" + options.backend;
        std::string run_rel = fallback_run_rel;
        if (session_bound_run) run_rel = options.formal_run_directory;
        std::string run_dir = local_path(root, run_rel, "run directory");
        make_directory(run_dir);
        const auto load_begin = Clock::now();

        const std::string config_path = local_path(root, options.config, "config");
        const CanonicalConfig config = CanonicalConfig::load(config_path);
        if (config.get("dataset") != options.dataset) throw std::runtime_error("dataset/config identity mismatch");
        if (config.get("privacy_policy") != record.privacy_policy || config.get("input_privacy") != record.input_privacy ||
            config.get("output_mode") != record.output_mode || config.get("proof_task") != "node_classification_full_gat") {
            throw std::runtime_error("config does not declare the strict full-GAT privacy/proof boundary");
        }
        const std::string configured_run = config.get("export_dir");
        if (!session_bound_run && !configured_run.empty()) {
            run_rel = configured_run + "/" + options.backend;
            run_dir = local_path(root, run_rel, "run directory");
            make_directory(run_dir);
        }
        record.run_dir = run_rel;
        result_file = run_dir + "/result.json";
        const std::string registration_rel = options.registration.empty() ? run_rel + "/model_registration.json" : options.registration;
        const std::string com_w_secret_rel = options.com_w_secret.empty() ? run_rel + "/com_W.prover.secret" : options.com_w_secret;
        const std::string registration_file = local_path(root, registration_rel, "model registration");
        const std::string com_w_secret_file = local_path(root, com_w_secret_rel, "com_W prover secret");
        if (registration_file != run_dir + "/model_registration.json" || com_w_secret_file != run_dir + "/com_W.prover.secret") {
            throw std::runtime_error("registration and com_W secret must remain in the formal run directory");
        }

        if (options.backend == "gpu" && !options.register_only) {
            record.status = "UNSUPPORTED";
            record.verification_status = "UNSUPPORTED";
            record.failure_reason = "formal masked_full GPU path is unsupported; no CUDA evidence or CPU proof fallback is permitted";
            write_result(result_file, record);
            std::cerr << record.status << ": " << record.failure_reason << '\n';
            return 2;
        }
        const std::string cache_rel = config.get("cache_root");
        const std::string bundle_rel = config.get("checkpoint_bundle");
        if (cache_rel.empty() || bundle_rel.empty()) throw std::runtime_error("config is missing baseline-local cache/checkpoint paths");
        const std::string cache_dir = local_path(root, cache_rel + "/" + options.dataset, "cache");
        const std::string bundle_dir = local_path(root, bundle_rel, "checkpoint bundle");
        const std::string quant_path = local_path(root, "baseline-local/configs/quantization_scale_chain.cfg", "quantization config");
        QuantConfig quant = QuantConfig::load(quant_path);
        if (quant.quant_cfg_id != config.get("quant_cfg_id")) throw std::runtime_error("quantization config identity mismatch");

        if (options.capacity_only || options.capacity_profile_only) {
            if (options.backend != "cpu") {
                write_capacity_report(GatCapacityReport{}, options.dataset, options.backend, quant.quant_cfg_id,
                                      rel_display(root, cache_dir + "/integrity.sha256"), rel_display(root, bundle_dir + "/integrity.sha256"),
                                      nullptr, nullptr, options.capacity_profile_only);
                return 1;
            }

            std::cerr << "CAPACITY_STAGE_BEGIN dataset=" << options.dataset << " stage=resource_snapshot\n" << std::flush;
            const CapacityResourceSnapshot resource_snapshot = capacity_resource_snapshot(run_dir);
            std::cerr << "CAPACITY_STAGE_END dataset=" << options.dataset << " stage=resource_snapshot host_headroom_bytes="
                      << resource_snapshot.host_headroom_bytes << " shared_disk_available_bytes="
                      << resource_snapshot.shared_available_bytes << " same_filesystem="
                      << (resource_snapshot.same_filesystem ? "true" : "false") << '\n' << std::flush;
            std::cerr << "CAPACITY_STAGE_BEGIN dataset=" << options.dataset << " stage=canonical_graph_load\n" << std::flush;
            const Graph graph = load_graph(cache_dir, config);
            std::cerr << "CAPACITY_STAGE_END dataset=" << options.dataset << " stage=canonical_graph_load nodes="
                      << graph.num_nodes << " edges=" << graph.edges() << " features=" << graph.num_features
                      << " graphs=" << graph.graph_count << '\n' << std::flush;
            const std::filesystem::path feature_txt = std::filesystem::path(cache_dir) / "features.txt";
            const std::filesystem::path feature_npy = std::filesystem::path(cache_dir) / "features.npy";
            const std::filesystem::path feature_source = std::filesystem::is_regular_file(feature_txt) ? feature_txt : feature_npy;
            if (!std::filesystem::is_regular_file(feature_source)) throw std::runtime_error("canonical feature source is missing");
            const std::uintmax_t feature_file_size = std::filesystem::file_size(feature_source);
            if (feature_file_size == 0 || feature_file_size > std::numeric_limits<std::uint64_t>::max())
                throw std::runtime_error("canonical feature source has an invalid file size");
            std::cerr << "CAPACITY_STAGE_BEGIN dataset=" << options.dataset << " stage=shape_estimate\n" << std::flush;
            Model shape_model = Model::load_shape_only(bundle_dir);
            shape_model.validate_identity(config);
            GatCapacityReport report = inspect_gat_capacity(graph, shape_model, quant, static_cast<std::uint64_t>(feature_file_size), options.capacity_profile_only);
            std::cerr << "CAPACITY_STAGE_END dataset=" << options.dataset << " stage=shape_estimate host_peak_upper_bytes="
                      << report.estimated_peak_host_memory_bytes << " disk_peak_upper_bytes="
                      << report.disk_requirement_bytes_upper_bound << " artifact_bytes_upper="
                      << report.artifact_disk_bytes_upper_bound << '\n' << std::flush;
            std::cerr << "CAPACITY_STAGE_BEGIN dataset=" << options.dataset << " stage=gkr_point_shape\n" << std::flush;
            const bool point_shape_ready = report.gkr_point_shape_ready;
            const std::uint64_t point_shape_bitmap_bytes = report.gkr_point_shape_bitmap_scratch_bytes;
            std::cerr << "CAPACITY_STAGE_END dataset=" << options.dataset << " stage=gkr_point_shape status="
                      << (options.capacity_profile_only
                              ? (report.gkr_point_domain_upper_bound_valid ? "ANALYSIS_UPPER_BOUND" : "ANALYSIS_UNCERTAIN")
                              : (point_shape_ready ? "PASS" : "BLOCKED"))
                      << " exact=" << (report.gkr_point_shape_exact ? "true" : "false") << " layers=" << report.gkr_point_shape_layer_count
                      << " fconn=" << report.gkr_point_shape_fconn_count << " bitmap_bytes=" << point_shape_bitmap_bytes
                      << " touched_index_bytes_upper="
                      << report.gkr_point_shape_touched_index_scratch_bytes_upper_bound;
            if (!point_shape_ready) std::cerr << " reason=" << report.reason;
            std::cerr << '\n' << std::flush;
            if (!point_shape_ready) {
                report.ready = false;
                report.reason = "native GKR point-domain shape admission failed: " + report.reason;
            } else {
                const std::uint64_t heap_bitmap_bytes =
                    point_shape_bitmap_bytes <= zkgpt::native_storage::heap_limit
                        ? point_shape_bitmap_bytes : 0;
                const std::uint64_t mapped_bitmap_bytes =
                    point_shape_bitmap_bytes > zkgpt::native_storage::heap_limit
                        ? point_shape_bitmap_bytes : 0;
                const std::uint64_t touched_index_bytes =
                    report.gkr_point_shape_touched_index_scratch_bytes_upper_bound;
                const std::uint64_t heap_touched_index_bytes =
                    touched_index_bytes <= zkgpt::native_storage::heap_limit
                        ? touched_index_bytes : 0;
                const std::uint64_t mapped_touched_index_bytes =
                    touched_index_bytes > zkgpt::native_storage::heap_limit
                        ? touched_index_bytes : 0;
                if (heap_touched_index_bytes > std::numeric_limits<std::uint64_t>::max() -
                                                   heap_bitmap_bytes)
                    throw std::overflow_error("GKR shape-admission heap capacity bound overflows u64");
                const std::uint64_t shape_heap_bytes = heap_bitmap_bytes + heap_touched_index_bytes;
                if (shape_heap_bytes > std::numeric_limits<std::uint64_t>::max() -
                                           report.estimated_native_heap_bytes_upper_bound)
                    throw std::overflow_error("GKR shape-admission capacity bound overflows u64");
                if (mapped_touched_index_bytes > std::numeric_limits<std::uint64_t>::max() -
                                                     mapped_bitmap_bytes)
                    throw std::overflow_error("GKR shape-admission mapped capacity bound overflows u64");
                const std::uint64_t shape_mapped_bytes = mapped_bitmap_bytes + mapped_touched_index_bytes;
                report.estimated_native_heap_bytes_upper_bound += shape_heap_bytes;
                // inspect_gat_capacity has already bounded shape admission as
                // its own host phase and combined it with proof/forward peaks
                // using max. These mapped files are not retained into later phases.
                report.native_mapped_phase_breakdown.emplace_back(
                    "shape_admission", shape_mapped_bytes);
                report.native_mapped_phase_breakdown.emplace_back(
                    "component:shape_admission_bitmap", mapped_bitmap_bytes);
                report.native_mapped_phase_breakdown.emplace_back(
                    "component:shape_admission_touched_indices", mapped_touched_index_bytes);
                report.estimated_native_mapped_bytes_upper_bound = std::max(
                    report.estimated_native_mapped_bytes_upper_bound,
                    shape_mapped_bytes);
                report.estimated_native_scratch_bytes = std::max(
                    report.estimated_native_scratch_bytes,
                    shape_mapped_bytes);
            }
            shape_model = Model{};
            if (options.capacity_profile_only) {
                report.host_memory_limit_bytes = resource_snapshot.host_headroom_bytes;
                report.native_scratch_available_bytes = resource_snapshot.scratch_available_bytes;
                report.artifact_disk_available_bytes = resource_snapshot.artifact_available_bytes;
                report.scratch_artifact_same_filesystem = resource_snapshot.same_filesystem;
                report.shared_scratch_artifact_available_bytes = resource_snapshot.shared_available_bytes;
                finalize_gat_capacity_report(report);
                report.ready = false;
                report.reason = report.gkr_point_domain_upper_bound_valid
                    ? "analysis-only symbolic profile: per-gate point source census, canonical all-value forward admission, SRS setup, proving, and verification were intentionally not run"
                    : "analysis-only symbolic profile has an unvalidated conservative point-domain upper bound; no formal readiness inference is permitted";
                std::cerr << "CAPACITY_PROFILE_ONLY dataset=" << options.dataset
                          << " status=ANALYSIS_ONLY symbolic_profile_complete="
                          << (report.symbolic_analysis_profile_complete ? "true" : "false")
                          << " native_point_shape=NOT_ENUMERATED forward_admission=NOT_RUN host_peak_upper_bytes="
                          << report.estimated_peak_host_memory_bytes << " disk_peak_upper_bytes="
                          << report.disk_requirement_bytes_upper_bound << '\n' << std::flush;
                write_capacity_report(report, options.dataset, options.backend,
                                      quant.quant_cfg_id,
                                      rel_display(root, cache_dir + "/integrity.sha256"),
                                      rel_display(root, bundle_dir + "/integrity.sha256"),
                                      &graph, &config, true);
                return report.symbolic_analysis_profile_complete ? 0 : 1;
            }
            if (point_shape_ready && report.forward_admission_peak_host_memory_bytes_upper_bound <= resource_snapshot.host_headroom_bytes) {
                report.forward_value_admission_ready = false;
                report.forward_value_admission_mode = "deferred_to_owned_forward_v1";
                std::cerr << "CAPACITY_STAGE_END dataset=" << options.dataset
                          << " stage=canonical_all_value_forward status=DEFERRED_TO_OWNED_FORWARD\n" << std::flush;
            } else if (point_shape_ready) {
                report.ready = false;
                report.reason = "canonical forward-admission host bound exceeds current available memory";
            }
            report.host_memory_limit_bytes = resource_snapshot.host_headroom_bytes;
            report.native_scratch_available_bytes = resource_snapshot.scratch_available_bytes;
            report.artifact_disk_available_bytes = resource_snapshot.artifact_available_bytes;
            report.scratch_artifact_same_filesystem = resource_snapshot.same_filesystem;
            report.shared_scratch_artifact_available_bytes = resource_snapshot.shared_available_bytes;
            finalize_gat_capacity_report(report);
            write_capacity_report(report, options.dataset, options.backend, quant.quant_cfg_id, rel_display(root, cache_dir + "/integrity.sha256"),
                                  rel_display(root, bundle_dir + "/integrity.sha256"), &graph, &config);
            return report.ready ? 0 : 1;
        }

        if (options.register_only) {
            if (path_exists(registration_file) != path_exists(com_w_secret_file)) {
                throw std::runtime_error("incomplete model registration pair; refusing to replace an unpaired artifact");
            }
            const auto registration_begin = Clock::now();
            const Graph graph = load_graph(cache_dir, config);
            const Model shape_model = Model::load_shape_only(bundle_dir);
            shape_model.validate_identity(config);
            const InputLayout layout = build_gat_input_layout(graph, shape_model, quant);

            const Model model = Model::load(bundle_dir);
            model.validate_identity(config);
            if (path_exists(registration_file)) {
                const ModelRegistration previous = read_registration(registration_file, options.dataset, true);
                require_owner_secret_file(com_w_secret_file);
                Fr previous_blinder;
                std::string previous_reason;
                if (!read_prover_commitment_secret(previous, com_w_secret_file, previous_blinder, &previous_reason))
                    throw std::runtime_error("existing registration secret is invalid: " + previous_reason);
                std::string previous_schema;
                json_string_field(read_regular_text(registration_file, "model registration"), "schema", previous_schema);
                if (previous_schema == "zkgpt-gat-model-registration-v2" && registration_matches_model(model, quant, layout, previous)) {
                    std::cout << "REGISTERED dataset=" << options.dataset << " backend=" << options.backend << " registration=" << registration_rel << " reused=true\n";
                    return 0;
                }
            }
            const ModelRegistration registration = register_model(model, quant, layout, model.manifest.model_arch_id);
            record.registration_ms = elapsed_ms(registration_begin, Clock::now());
            record.com_w_point = registration.com_W.point_hex;
            record.registration_digest = registration.registration_digest;
            record.registration_binding_digest = registration.binding_digest;
            record.registration_layout_digest = registration.layout_digest;
            record.registration_path = run_rel + "/model_registration.json";
            record.checkpoint_identity = "baseline-local/checkpoints/" + options.dataset + "_gat/integrity.sha256";
            const std::string staged_registration = registration_file + ".staged." + std::to_string(::getpid());
            const std::string staged_secret = com_w_secret_file + ".staged." + std::to_string(::getpid());
            std::string secret_reason;
            try {
                write_registration(staged_registration, registration, root, options.dataset);
                if (!write_prover_commitment_secret(registration, staged_secret, &secret_reason))
                    throw std::runtime_error("cannot stage com_W prover secret: " + secret_reason);
                require_owner_secret_file(staged_secret);
                const ModelRegistration staged = read_registration(staged_registration, options.dataset);
                Fr staged_blinder;
                if (!registration_matches_model(model, quant, layout, staged) ||
                    !read_prover_commitment_secret(staged, staged_secret, staged_blinder, &secret_reason))
                    throw std::runtime_error("staged model registration validation failed: " + secret_reason);
                if (::rename(staged_secret.c_str(), com_w_secret_file.c_str()) != 0 ||
                    ::rename(staged_registration.c_str(), registration_file.c_str()) != 0)
                    throw std::runtime_error("cannot atomically install staged registration files");
            } catch (...) {
                ::unlink(staged_registration.c_str());
                ::unlink(staged_secret.c_str());
                throw;
            }
            std::cout << "REGISTERED dataset=" << options.dataset << " backend=" << options.backend << " registration=" << record.registration_path << '\n';
            return 0;
        }

        Graph loaded_graph = load_graph(cache_dir, config);
        Tensor2D loaded_features = load_features(cache_dir, config, quant, loaded_graph);
        (void)load_labels(cache_dir, loaded_graph);
        Model model = Model::load(bundle_dir);
        model.validate_identity(config);
        const StaticTables tables = StaticTables::build(quant);
        if (!tables.authenticate()) throw std::runtime_error("canonical static table authentication failed");
        record.full_dataset_identity = "canonical:" + options.dataset;
        record.checkpoint_identity = "baseline-local/checkpoints/" + options.dataset + "_gat/integrity.sha256";
        record.cache_identity = "baseline-local/data/cache/" + options.dataset + "/integrity.sha256";
        record.load_ms = elapsed_ms(load_begin, Clock::now());

        const auto witness_begin = Clock::now();
        detail::ValidatedFormalInputs owned_inputs =
            detail::ValidatedFormalInputs::compute_and_own(
                std::move(loaded_graph), std::move(loaded_features),
                std::move(model), std::move(quant));
        const Graph& graph = owned_inputs.graph();
        const Tensor2D& features = owned_inputs.features();
        const QuantConfig& owner_quant = owned_inputs.quant();
        const ForwardResult& forward = owned_inputs.forward();
        const auto plan_owner = owned_inputs.plan_owner();
        const GatCircuitPlan& plan = *plan_owner;
        if (!plan.official_layered || plan.has_placeholder) throw std::runtime_error("official layered GAT circuit plan is unavailable");
        if (owned_inputs.canonical_forward_generation_calls() != 1U ||
            owned_inputs.canonical_forward_value_admission_calls() != 1U)
            throw std::logic_error("formal inputs did not complete one fresh canonical forward and value-admission pass");
        std::cerr << "[zkGPT][forward] canonical_forward_generation_calls=1 canonical_forward_generation_ms="
                  << owned_inputs.canonical_forward_generation_ms() << '\n' << std::flush;
        std::cerr << "[zkGPT][forward] canonical_forward_value_admission_calls=1 canonical_forward_value_admission_ms="
                  << owned_inputs.canonical_forward_value_admission_ms() << '\n' << std::flush;
        record.witness_ms = elapsed_ms(witness_begin, Clock::now());

        const auto registration_begin = Clock::now();
        ModelRegistration registration = read_registration(registration_file, options.dataset);
        require_owner_secret_file(com_w_secret_file);
        Fr com_w_blinder;
        std::string secret_reason;
        if (!read_prover_commitment_secret(registration, com_w_secret_file, com_w_blinder, &secret_reason)) {
            throw std::runtime_error("cannot load com_W prover secret: " + secret_reason);
        }
        registration.com_W_padding_mask = com_w_blinder;
        registration.has_private_padding_mask = true;
        record.registration_ms = elapsed_ms(registration_begin, Clock::now());
        record.com_w_point = registration.com_W.point_hex;
        record.registration_digest = registration.registration_digest;
        record.registration_binding_digest = registration.binding_digest;
        record.registration_layout_digest = registration.layout_digest;
        record.registration_path = run_rel + "/model_registration.json";
        record.prover_secret_path = run_rel + "/com_W.prover.secret";

        VerifierContext context = make_verifier_context(graph, features, owner_quant, tables, forward.logits, registration, plan_owner,
                                                        record.privacy_profile);
        std::string artifact_limit_reason;
        const std::uint64_t artifact_wire_limit =
            gat_proof_artifact_wire_limit(context, &artifact_limit_reason);
        if (artifact_wire_limit == 0)
            throw std::runtime_error("trusted proof artifact capacity is unavailable: " + artifact_limit_reason);

        context.backend = options.backend;
        context.full_dataset_identity = record.full_dataset_identity;
        context.checkpoint_identity = record.checkpoint_identity;
        context.cache_identity = record.cache_identity;
        record.statement_digest = verifier_context_digest(context);
        record.circuit_id = plan.circuit_id;
        record.plan_digest = plan.plan_digest;
        record.statement_path = run_rel + "/public_statement.json";
        record.public_logits_path = run_rel + "/public_logits.txt";
        record.verifier_context_path = run_rel + "/verifier_context.bin";
        write_logits(run_dir + "/public_logits.txt", forward.logits);
        write_public_statement(run_dir + "/public_statement.json", record, graph, features, owner_quant, tables, plan);
        write_formal_verifier_context(run_dir + "/verifier_context.bin", owned_inputs, context);

        const auto prove_begin = Clock::now();
        std::cerr << "gat_adapter_prover_stage: native-formal-proof begin" << std::endl;
        ProveResult proved = detail::prove_gat_formal(
            owned_inputs, registration, context, record.privacy_profile);
        std::cerr << "gat_adapter_prover_stage: native-formal-proof complete elapsed_ms="
                  << elapsed_ms(prove_begin, Clock::now()) << " status=" << prove_status_name(proved.status) << std::endl;
        record.prove_total_ms = elapsed_ms(prove_begin, Clock::now());
        record.preparation_ms = proved.preparation_ms;
        record.aux_commitment_ms = proved.aux_commitment_ms;
        record.lasso_query_commitment_ms = proved.lasso_query_commitment_ms;
        record.lasso_family_generator_derive_ms = proved.lasso_family_generator_derive_ms;
        record.lasso_family_commitment_ms = proved.lasso_family_commitment_ms;
        record.lasso_family_selected_row_mapping_ms = proved.lasso_family_selected_row_mapping_ms;
        record.initial_commitment_blinding_prepare_ms = proved.initial_commitment_blinding_prepare_ms;
        record.commitment_ms = proved.commitment_ms;
        record.self_verify_ms = proved.self_verify_ms;
        const double timing_components[] = {record.prove_total_ms, record.preparation_ms,
            record.aux_commitment_ms, record.lasso_query_commitment_ms,
            record.lasso_family_generator_derive_ms,
            record.lasso_family_commitment_ms,
            record.initial_commitment_blinding_prepare_ms,
            record.lasso_family_selected_row_mapping_ms,
            record.commitment_ms, record.self_verify_ms};
        for (double component : timing_components) {
            if (!std::isfinite(component) || component < 0.0)
                throw std::runtime_error("native prover returned a non-finite or negative timing component");
        }
        if (record.preparation_ms < record.lasso_query_commitment_ms)
            throw std::runtime_error("native preparation timing is shorter than its L_qry commitment substage");
        record.preparation_net_ms = record.preparation_ms - record.lasso_query_commitment_ms;
        const double expected_total_commitment = record.aux_commitment_ms +
            record.lasso_query_commitment_ms +
            record.lasso_family_generator_derive_ms +
            record.lasso_family_commitment_ms;
        if (!std::isfinite(expected_total_commitment) || expected_total_commitment != record.commitment_ms)
            throw std::runtime_error("native commitment timing does not equal its measured commitment substages");
        const double paper_prove_without_artifact_write = record.prove_total_ms -
            record.initial_commitment_blinding_prepare_ms - record.commitment_ms -
            record.self_verify_ms;
        if (!std::isfinite(paper_prove_without_artifact_write) || paper_prove_without_artifact_write < 0.0)
            throw std::runtime_error("native prove timing decomposition is invalid");
        record.prove_ms = paper_prove_without_artifact_write;
        if (proved.status != ProveStatus::GENERATED || !proved.has_artifact) {
            record.status = proved.status == ProveStatus::UNSUPPORTED_PRIVACY_PROFILE ? "UNSUPPORTED" : "FAILED";
            record.verification_status = record.status;
            record.failure_reason = proved.diagnostic.empty() ? "native prover did not generate a proof artifact" : proved.diagnostic;
            write_result(result_file, record);
            std::cerr << record.status << ": " << record.failure_reason << '\n';
            return record.status == "UNSUPPORTED" ? 2 : 1;
        }

        const std::string artifact_file = run_dir + "/proof.artifact";
        std::string artifact_reason;
        if (!options.defer_self_verify) {
            const auto artifact_wire_self_check_begin = Clock::now();
            if (!proved.artifact.valid_wire(&artifact_reason))
                throw std::runtime_error("proof artifact failed internal wire self-check: " + artifact_reason);
            record.artifact_wire_self_check_ms = elapsed_ms(artifact_wire_self_check_begin, Clock::now());
        }
        std::string artifact_wire;
        const auto artifact_serialize_begin = Clock::now();
        std::cerr << "[zkGPT][artifact] serialize_begin wire_limit_bytes=" << artifact_wire_limit << '\n' << std::flush;
        // Deferred formal runs keep this artifact pending until the independent verifier accepts its serialized bytes.
        if (!proved.artifact.serialize_after_wire_validation(artifact_wire, artifact_wire_limit, &artifact_reason))
            throw std::runtime_error("proof artifact in-memory serialization failed: " + artifact_reason);
        record.serialize_ms = elapsed_ms(artifact_serialize_begin, Clock::now());
        std::cerr << "[zkGPT][artifact] serialize_complete bytes=" << artifact_wire.size()
                  << " elapsed_ms=" << record.serialize_ms << '\n' << std::flush;
        record.prove_ms += record.serialize_ms;
        if (artifact_wire.empty() || artifact_wire.size() > artifact_wire_limit)
            throw std::runtime_error("serialized proof artifact has invalid exact byte length");
        record.proof_size_bytes = static_cast<std::uint64_t>(artifact_wire.size());
        record.artifact_size_bytes = record.proof_size_bytes;
        record.artifact_sha256 = sha256_bytes_digest(artifact_wire);
        const auto artifact_write_begin = Clock::now();
        std::ofstream artifact_out(artifact_file.c_str(), std::ios::binary | std::ios::trunc);
        if (!artifact_out)
            throw std::runtime_error("cannot open proof artifact for writing");
        artifact_out.write(artifact_wire.data(), static_cast<std::streamsize>(artifact_wire.size()));
        artifact_out.flush();
        if (!artifact_out.good()) {
            artifact_out.close();
            std::remove(artifact_file.c_str());
            throw std::runtime_error("proof artifact write failed");
        }
        artifact_out.close();
        if (artifact_out.fail()) {
            std::remove(artifact_file.c_str());
            throw std::runtime_error("proof artifact close failed");
        }
        record.artifact_write_ms = elapsed_ms(artifact_write_begin, Clock::now());
        std::string().swap(artifact_wire);
        proved.artifact = GatProofArtifact{};
        GatProofArtifact artifact;
        std::string artifact_readback_wire;
        std::cerr << "[zkGPT][artifact] readback_begin\n" << std::flush;
        const auto artifact_readback_begin = Clock::now();
        bool artifact_readback_ok = false;
        if (options.defer_self_verify) {
            artifact_readback_ok = GatProofArtifact::read_wire_bytes(artifact_file, artifact_readback_wire, artifact_wire_limit, &artifact_reason);
            if (artifact_readback_ok &&
                (artifact_readback_wire.size() != record.proof_size_bytes ||
                 sha256_bytes_digest(artifact_readback_wire) != record.artifact_sha256)) {
                artifact_readback_ok = false;
                artifact_reason = "proof artifact readback length or SHA-256 differs from serialized bytes";
            }
            if (artifact_readback_ok)
                artifact_readback_ok = GatProofArtifact::parse_wire(artifact_readback_wire, artifact, artifact_wire_limit, &artifact_reason);
            std::string().swap(artifact_readback_wire);
        } else {
            artifact_readback_ok = GatProofArtifact::read(artifact_file, artifact, artifact_wire_limit, &artifact_reason);
        }
        record.artifact_readback_ms = elapsed_ms(artifact_readback_begin, Clock::now());
        std::cerr << "[zkGPT][artifact] readback_complete elapsed_ms=" << record.artifact_readback_ms << " ok=" << (artifact_readback_ok ? "true" : "false") << '\n' << std::flush;
        if (!artifact_readback_ok)
            throw std::runtime_error("serialized proof artifact cannot be read: " + artifact_reason);
        if (!std::isfinite(record.prove_ms) || !std::isfinite(record.serialize_ms) ||
            !std::isfinite(record.artifact_wire_self_check_ms) ||
            record.artifact_write_ms < 0.0 || record.artifact_readback_ms < 0.0)
            throw std::runtime_error("artifact timing decomposition is invalid");
        record.artifact_path = run_rel + "/proof.artifact";
        record.proof_generated = true;
        record.proof_digest = canonical_digest64(artifact.transcript_digest);
        for (const ProofSection* section : {&artifact.Pi_Lasso, &artifact.Pi_MatMul, &artifact.Pi_GKR, &artifact.Pi_combine, &artifact.Pi_open}) {
            for (const TranscriptMessage& message : section->messages) {
                record.proof_message_bytes += static_cast<std::uint64_t>(message.label.size() + message.payload.size());
            }
        }
        struct stat artifact_stat;
        if (::stat(artifact_file.c_str(), &artifact_stat) != 0 || artifact_stat.st_size < 0 ||
            static_cast<std::uint64_t>(artifact_stat.st_size) != record.proof_size_bytes)
            throw std::runtime_error("proof artifact file size differs from its in-memory canonical wire");
        record.verifier_report_path = run_rel + "/verifier_report.json";

        if (options.defer_self_verify) {
            record.status = "PROOF_GENERATED_PENDING_VERIFY";
            record.verification_status = "PROOF_GENERATED_PENDING_VERIFY";
            record.verifier_called = false;
            record.verifier_ok = false;
            record.verify_ms = 0.0;
            write_result(result_file, record);
            std::cout << record.status << " dataset=" << record.dataset << " backend=" << record.backend << " result=" << record.run_dir + "/result.json" << '\n';
            return 0;
        }

        const auto verify_begin = Clock::now();
        std::string verify_reason;
        record.verifier_called = true;
        record.verifier_ok = verify_gat(context, artifact, &verify_reason);
        record.verify_ms = elapsed_ms(verify_begin, Clock::now());
        if (!record.verifier_ok) {
            record.status = "FAILED";
            record.verification_status = "FAILED";
            record.failure_reason = verify_reason.empty() ? "independent verifier rejected proof artifact" : verify_reason;
        } else {
            record.status = "VERIFY_OK";
            record.verification_status = "VERIFY_OK";
        }
        write_result(result_file, record);
        std::cout << record.status << " dataset=" << record.dataset << " backend=" << record.backend << " result=" << record.run_dir + "/result.json" << '\n';
        return record.status == "VERIFY_OK" ? 0 : 1;
    } catch (const std::exception& error) {
        record.status = record.status == "UNSUPPORTED" ? "UNSUPPORTED" : "FAILED";
        record.verification_status = record.status;
        record.failure_reason = error.what();
        if (!result_file.empty()) {
            try {
                write_result(result_file, record);
            } catch (...) {}
        }
        std::cerr << record.status << ": " << record.failure_reason << '\n';
        return record.status == "UNSUPPORTED" ? 2 : 1;
    }
}
