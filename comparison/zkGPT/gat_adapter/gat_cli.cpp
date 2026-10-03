#include "gat_proof.hpp"
#include "gat_types.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
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
    bool help = false;
    bool register_only = false;
    bool capacity_only = false;
};

struct Record {
    std::string schema = "zkgpt-gat-result-v1";
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
    double commitment_ms = 0.0;
    double prove_total_ms = 0.0;
    double self_verify_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    double serialize_ms = 0.0;
};

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

void json_number(std::ostringstream& out, const char* key, double value, bool comma = true) {
    out << "  \"" << key << "\": " << std::fixed << std::setprecision(3) << value << (comma ? ",\n" : "\n");
}

void write_result(const std::string& path, const Record& record) {
    std::ostringstream out;
    out << "{\n";
    json_string(out, "schema", record.schema);
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
    out << "    \"load\": " << std::fixed << std::setprecision(3) << record.load_ms << ",\n";
    out << "    \"registration\": " << record.registration_ms << ",\n";
    out << "    \"witness\": " << record.witness_ms << ",\n";
    out << "    \"commitment\": " << record.commitment_ms << ",\n";
    out << "    \"prove_total\": " << record.prove_total_ms << ",\n";
    out << "    \"self_verify\": " << record.self_verify_ms << ",\n";
    out << "    \"prove\": " << record.prove_ms << ",\n";
    out << "    \"verify\": " << record.verify_ms << ",\n";
    out << "    \"serialize\": " << record.serialize_ms << "\n";
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
                           const std::string& cache_identity, const std::string& checkpoint_identity) {
    std::ostringstream out;
    out << "{\n";
    json_string(out, "schema", report.schema);
    json_string(out, "dataset", dataset);
    json_string(out, "backend", backend);
    json_string(out, "quant_cfg_id", quant_cfg_id);
    json_string(out, "cache_identity", cache_identity);
    json_string(out, "checkpoint_identity", checkpoint_identity);
    out << "  \"num_nodes\": " << report.num_nodes << ",\n";
    out << "  \"num_edges\": " << report.num_edges << ",\n";
    out << "  \"num_features\": " << report.num_features << ",\n";
    out << "  \"num_classes\": " << report.num_classes << ",\n";
    out << "  \"graph_count\": " << report.graph_count << ",\n";
    json_string(out, "model_arch_id", report.model_shape.model_arch_id);
    out << "  \"model_layers\": " << report.model_shape.layers << ",\n";
    out << "  \"model_output_heads\": " << report.model_shape.output_heads << ",\n";
    out << "  \"model_classes\": " << report.model_shape.classes << ",\n";
    out << "  \"model_input_dim\": " << report.model_shape.input_dim << ",\n";
    out << "  \"model_hidden_dim\": " << report.model_shape.hidden_dim << ",\n";
    out << "  \"n_raw\": " << report.input_layout.n_raw << ",\n";
    out << "  \"n_in\": " << report.input_layout.n_in << ",\n";
    out << "  \"fconn_count\": " << report.fconn_count << ",\n";
    json_bool(out, "explicit_matmul_gate_expansion", report.explicit_matmul_gate_expansion);
    out << "  \"max_materialized_gate\": " << report.max_materialized_gate << ",\n";
    out << "  \"max_materialized_vector\": " << report.max_materialized_vector << ",\n";
    out << "  \"attention_relation_count\": " << report.attention_relation_count << ",\n";
    out << "  \"division_relation_count\": " << report.division_relation_count << ",\n";
    out << "  \"context_recipe_bytes_bound\": " << report.context_recipe_bytes_bound << ",\n";
    out << "  \"context_payload_bytes_bound\": " << report.context_payload_bytes_bound << ",\n";
    json_bool(out, "ppi_node_ptr_boundary", report.ppi_node_ptr_boundary);
    json_bool(out, "ppi_edge_ptr_boundary", report.ppi_edge_ptr_boundary);
    out << "  \"masked_lasso_matrix_entries\": " << report.masked_lasso_matrix_entries << ",\n";
    out << "  \"masked_response_entries_lower_bound\": " << report.masked_response_entries_lower_bound << ",\n";
    out << "  \"masked_proof_bytes_lower_bound\": " << report.masked_proof_bytes_lower_bound << ",\n";
    out << "  \"maximum_masked_response_entries\": " << report.maximum_masked_response_entries << ",\n";
    out << "  \"native_response_vector_limit\": " << (1ULL << 20) << ",\n";
    out << "  \"lasso_family_domains\": [";
    for (std::size_t index = 0; index < report.lasso_family_domains.size(); ++index) { if (index != 0) out << ','; out << '[' << report.lasso_family_domains[index][0] << ',' << report.lasso_family_domains[index][1] << ']'; }
    out << "],\n";
    json_string(out, "proof_path", report.proof_path);
    json_bool(out, "full_relation_bound", report.full_relation_bound);
    json_bool(out, "ready", report.ready);
    json_string(out, "status", report.ready ? "READY" : "BLOCKED");
    out << "  \"native_message_string_limit_bytes\": " << (1ULL << 26) << ",\n";
    out << "  \"native_artifact_wire_limit_bytes\": " << (1ULL << 30) << ",\n";
    json_string(out, "reason", report.reason, false);
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

void write_verifier_context(const std::string& path, const VerifierContext& context) {
    std::string reason;
    if (!zkgpt::gat::write_verifier_context(context, path, &reason)) {
        throw std::runtime_error("cannot write verifier context v4: " + (reason.empty() ? std::string("unknown error") : reason));
    }
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
        } else if (arg == "--register-only") {
            options.register_only = true;
        } else if (arg == "--capacity-only") {
            options.capacity_only = true;
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
    if (options.config.empty()) options.config = "baseline-local/configs/" + options.dataset + "_secure.cfg";
    return options;
}

void print_help() {
    std::cout << "Usage: gat_cli --dataset <cora|citeseer|pubmed|ppi|ogbn_arxiv> "
                 "--backend <cpu|gpu> [--baseline-root <zkGPT-root>] [--config <baseline-relative.cfg>] "
                 "[--registration <baseline-relative.json>] [--com-w-secret <baseline-relative.secret>]\n"
                 "       gat_cli ... --register-only\n"
                 "       gat_cli --capacity-only --dataset <...> --backend cpu [--baseline-root <zkGPT-root>] [--config <...>]\n"
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
        record.native_backend = options.backend == "gpu" ? "cuda_formal_masked_full" : "cpu_native_zkgpt";
        if (options.backend == "gpu") {
            record.capability_status = "UNSUPPORTED_FORMAL_MASKED_FULL_GPU";
            record.formal_backend_supported = false;
#if defined(ZKGPT_HAS_CUDA) && ZKGPT_HAS_CUDA
            record.gpu_compiled = true;
#endif
        }
        const std::string root = without_trailing_slash(options.baseline_root);
        const std::string fallback_run_rel = "baseline-local/runs/" + options.dataset + "_secure/" + options.backend;
        std::string run_rel = fallback_run_rel;
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
        if (!configured_run.empty()) {
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
        const QuantConfig quant = QuantConfig::load(quant_path);
        if (quant.quant_cfg_id != config.get("quant_cfg_id")) throw std::runtime_error("quantization config identity mismatch");

        if (options.capacity_only) {
            if (options.backend != "cpu") {
                write_capacity_report(GatCapacityReport{}, options.dataset, options.backend, quant.quant_cfg_id,
                                      rel_display(root, cache_dir + "/integrity.sha256"), rel_display(root, bundle_dir + "/integrity.sha256"));
                return 1;
            }

            const Graph graph = load_graph(cache_dir, config);
            const Model shape_model = Model::load_shape_only(bundle_dir);
            shape_model.validate_identity(config);
            const GatCapacityReport report = inspect_gat_capacity(graph, shape_model, quant);
            write_capacity_report(report, options.dataset, options.backend, quant.quant_cfg_id, rel_display(root, cache_dir + "/integrity.sha256"),
                                  rel_display(root, bundle_dir + "/integrity.sha256"));
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

        const Graph graph = load_graph(cache_dir, config);
        const Tensor2D features = load_features(cache_dir, config, quant, graph);
        (void)load_labels(cache_dir, graph);
        const Model model = Model::load(bundle_dir);
        model.validate_identity(config);
        const StaticTables tables = StaticTables::build(quant);
        if (!tables.authenticate()) throw std::runtime_error("canonical static table authentication failed");
        record.full_dataset_identity = "canonical:" + options.dataset;
        record.checkpoint_identity = "baseline-local/checkpoints/" + options.dataset + "_gat/integrity.sha256";
        record.cache_identity = "baseline-local/data/cache/" + options.dataset + "/integrity.sha256";
        record.load_ms = elapsed_ms(load_begin, Clock::now());

        const auto witness_begin = Clock::now();
        const ForwardResult forward = run_forward(graph, features, model, quant);
        if (!validate_relation(graph, features, model, quant, forward)) throw std::runtime_error("canonical GAT relation validation failed");
        const GatCircuitPlan plan = build_gat_circuit_plan(graph, model, quant, forward);
        if (!plan.official_layered || plan.has_placeholder) throw std::runtime_error("official layered GAT circuit plan is unavailable");
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

        VerifierContext context = make_verifier_context(graph, features, quant, tables, forward.logits, registration, plan, record.privacy_profile);
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
        write_public_statement(run_dir + "/public_statement.json", record, graph, features, quant, tables, plan);
        write_verifier_context(run_dir + "/verifier_context.bin", context);

        const auto prove_begin = Clock::now();
        const ProveResult proved = prove_gat(plan, graph, features, model, quant, forward, registration, record.privacy_profile);
        record.prove_total_ms = elapsed_ms(prove_begin, Clock::now());
        record.commitment_ms = proved.commitment_ms;
        record.self_verify_ms = proved.self_verify_ms;
        record.prove_ms = std::max(0.0, record.prove_total_ms - record.commitment_ms - record.self_verify_ms);
        if (proved.status != ProveStatus::GENERATED || !proved.has_artifact) {
            record.status = proved.status == ProveStatus::UNSUPPORTED_PRIVACY_PROFILE ? "UNSUPPORTED" : "FAILED";
            record.verification_status = record.status;
            record.failure_reason = proved.diagnostic.empty() ? "native prover did not generate a proof artifact" : proved.diagnostic;
            write_result(result_file, record);
            std::cerr << record.status << ": " << record.failure_reason << '\n';
            return record.status == "UNSUPPORTED" ? 2 : 1;
        }

        const std::string artifact_file = run_dir + "/proof.artifact";
        const auto serialize_begin = Clock::now();
        std::string artifact_reason;
        if (!proved.artifact.write(artifact_file, artifact_wire_limit,
                                   &artifact_reason))
            throw std::runtime_error("proof artifact serialization failed: " + artifact_reason);
        GatProofArtifact artifact;
        if (!GatProofArtifact::read(artifact_file, artifact, artifact_wire_limit,
                                    &artifact_reason))
            throw std::runtime_error("serialized proof artifact cannot be read: " + artifact_reason);
        record.serialize_ms = elapsed_ms(serialize_begin, Clock::now());
        record.artifact_path = run_rel + "/proof.artifact";
        record.proof_generated = true;
        record.proof_digest = canonical_digest64(artifact.transcript_digest);
        for (const ProofSection* section : {&artifact.Pi_Lasso, &artifact.Pi_MatMul, &artifact.Pi_GKR, &artifact.Pi_combine, &artifact.Pi_open}) {
            for (const TranscriptMessage& message : section->messages) {
                record.proof_message_bytes += static_cast<std::uint64_t>(message.label.size() + message.payload.size());
            }
        }
        struct stat artifact_stat;
        if (::stat(artifact_file.c_str(), &artifact_stat) != 0 || artifact_stat.st_size < 0) throw std::runtime_error("cannot stat proof artifact");
        record.proof_size_bytes = static_cast<std::uint64_t>(artifact_stat.st_size);
        record.artifact_size_bytes = record.proof_size_bytes;
        record.artifact_sha256 = sha256_file_digest(artifact_file);
        record.verifier_report_path = run_rel + "/verifier_report.json";

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
