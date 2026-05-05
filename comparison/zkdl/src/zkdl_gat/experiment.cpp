#include "zkdl_gat/experiment.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace zkdl_gat {
namespace {

std::string trim(std::string value) {
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), value.end());
    return value;
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }
    std::ostringstream stream;
    stream << input.rdbuf();
    return stream.str();
}

std::string get_required(const DatasetConfig& config, const std::string& key) {
    const auto it = config.values.find(key);
    if (it == config.values.end() || it->second.empty()) {
        throw std::runtime_error("missing required config key: " + key);
    }
    return it->second;
}

std::string get_optional(const DatasetConfig& config, const std::string& key, const std::string& fallback = "") {
    const auto it = config.values.find(key);
    if (it == config.values.end()) {
        return fallback;
    }
    return it->second;
}

std::filesystem::path resolve_path(const std::string& experiment_root, const std::string& value) {
    const std::filesystem::path path(value);
    if (path.is_absolute()) {
        return path;
    }
    return std::filesystem::weakly_canonical(std::filesystem::path(experiment_root) / path);
}

bool file_exists(const std::string& experiment_root, const std::string& value) {
    if (value.empty()) {
        return false;
    }
    return std::filesystem::exists(resolve_path(experiment_root, value));
}

std::uintmax_t file_size_or_zero(const std::string& experiment_root, const std::string& value) {
    if (!file_exists(experiment_root, value)) {
        return 0;
    }
    const auto path = resolve_path(experiment_root, value);
    if (!std::filesystem::is_regular_file(path)) {
        return 0;
    }
    return std::filesystem::file_size(path);
}

std::size_t count_lines_or_zero(const std::string& experiment_root, const std::string& value) {
    if (!file_exists(experiment_root, value)) {
        return 0;
    }
    std::ifstream input(resolve_path(experiment_root, value));
    std::size_t count = 0;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            ++count;
        }
    }
    return count;
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

void write_json_string(std::ostream& out, const std::string& key, const std::string& value, bool comma = true) {
    out << "  \"" << json_escape(key) << "\": \"" << json_escape(value) << "\"";
    if (comma) {
        out << ',';
    }
    out << '\n';
}

void write_json_bool(std::ostream& out, const std::string& key, bool value, bool comma = true) {
    out << "  \"" << json_escape(key) << "\": " << (value ? "true" : "false");
    if (comma) {
        out << ',';
    }
    out << '\n';
}

void write_json_int(std::ostream& out, const std::string& key, long long value, bool comma = true) {
    out << "  \"" << json_escape(key) << "\": " << value;
    if (comma) {
        out << ',';
    }
    out << '\n';
}

void write_json_optional_double(std::ostream& out, const std::string& key, const std::optional<double>& value) {
    out << "  \"" << json_escape(key) << "\": ";
    if (value) {
        out << *value;
    } else {
        out << "null";
    }
    out << ",\n";
}

void write_json_optional_size(std::ostream& out, const std::string& key, const std::optional<unsigned long long>& value) {
    out << "  \"" << json_escape(key) << "\": ";
    if (value) {
        out << *value;
    } else {
        out << "null";
    }
    out << ",\n";
}

void write_string_map(std::ostream& out, const std::string& key, const std::map<std::string, std::string>& values) {
    out << "  \"" << json_escape(key) << "\": {\n";
    std::size_t index = 0;
    for (const auto& [map_key, value] : values) {
        out << "    \"" << json_escape(map_key) << "\": \"" << json_escape(value) << "\"";
        if (++index != values.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  },\n";
}

std::string extract_json_string(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (std::regex_search(text, match, pattern)) {
        return match[1].str();
    }
    return "";
}

bool extract_json_bool(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (std::regex_search(text, match, pattern)) {
        return match[1].str() == "true";
    }
    return false;
}

int extract_json_int(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (std::regex_search(text, match, pattern)) {
        return std::stoi(match[1].str());
    }
    return 0;
}

std::vector<std::string> common_gat_capability_gaps(bool include_cpu_gap) {
    std::vector<std::string> gaps = {
        "full_gat_attention_statement",
        "dst_group_softmax_max_constraint",
        "inverse_constraint",
        "exp_lookup_argument",
        "elu_lookup_argument",
        "range_argument",
        "rescale_argument",
        "fac4dnn_aggregation",
        "final_logits_public_binding",
        "verifier_acceptance_api"
    };
    if (include_cpu_gap) {
        gaps.insert(gaps.begin(), "cpu_tensor_commitment_backend");
    }
    return gaps;
}

}  // namespace

DatasetConfig load_dataset_config(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open dataset config: " + path);
    }
    DatasetConfig config;
    config.path = path;
    std::string line;
    while (std::getline(input, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error("invalid config line: " + line);
        }
        config.values[trim(line.substr(0, equals))] = trim(line.substr(equals + 1));
    }
    return config;
}

GpuProbe load_gpu_probe(const std::string& path) {
    GpuProbe probe;
    if (path.empty() || !std::filesystem::exists(path)) {
        return probe;
    }
    const auto text = read_text(path);
    probe.loaded = true;
    probe.gpu_backend_active = extract_json_bool(text, "gpu_backend_active");
    probe.cpu_fallback_used = extract_json_bool(text, "cpu_fallback_used");
    probe.gpu_device_count = extract_json_int(text, "gpu_device_count");
    probe.gpu_device_name = extract_json_string(text, "gpu_device_name");
    probe.gpu_backend_name = extract_json_string(text, "gpu_backend_name");
    probe.primitive_probe_status = extract_json_string(text, "primitive_probe_status");
    return probe;
}

Manifest build_manifest(
    const DatasetConfig& config,
    const std::string& route,
    const std::string& experiment_root,
    const GpuProbe& gpu_probe) {
    Manifest manifest;
    manifest.dataset = get_required(config, "dataset_id");
    manifest.backend = route;
    manifest.route_id = route == "gpu" ? "zkdl_gat_full_forward_gpu" : "zkdl_gat_full_forward_cpu";
    manifest.status = "blocked";
    manifest.strict_public_input_equivalent = true;
    manifest.full_gat_forward_in_zkdl_proof = false;
    manifest.final_logits_derivation_complete = false;
    manifest.no_precomputed_logits_bypass = true;
    manifest.capability_gaps = common_gat_capability_gaps(route != "gpu");

    const std::vector<std::string> source_keys = {
        "source_config",
        "cache_meta",
        "cache_edges",
        "cache_features",
        "cache_labels",
        "cache_node_ptr",
        "checkpoint_manifest",
        "checkpoint_tensors"
    };
    bool sources_ok = true;
    for (const auto& key : source_keys) {
        const auto value = get_optional(config, key);
        if (value.empty()) {
            continue;
        }
        manifest.source_files[key] = resolve_path(experiment_root, value).string();
        const auto hash_key = key + "_sha256";
        manifest.source_sha256[hash_key] = get_optional(config, hash_key);
        sources_ok = sources_ok && file_exists(experiment_root, value);
    }

    manifest.dataset_stats["node_count"] = get_optional(config, "node_count");
    manifest.dataset_stats["raw_edge_count"] = get_optional(config, "raw_edge_count");
    manifest.dataset_stats["graph_count"] = get_optional(config, "graph_count", "1");
    manifest.dataset_stats["feature_dim"] = get_optional(config, "feature_dim");
    manifest.dataset_stats["class_count"] = get_optional(config, "class_count");
    manifest.dataset_stats["layer_count"] = get_optional(config, "L");
    manifest.dataset_stats["hidden_profile"] = get_optional(config, "hidden_profile");
    manifest.dataset_stats["K_out"] = get_optional(config, "K_out");
    manifest.dataset_stats["self_loop_rule"] = get_optional(config, "self_loop_rule");
    manifest.dataset_stats["edge_sort_rule"] = get_optional(config, "edge_sort_rule");
    manifest.dataset_stats["edge_rows_seen"] = std::to_string(count_lines_or_zero(experiment_root, get_optional(config, "cache_edges")));
    manifest.dataset_stats["checkpoint_tensor_bytes"] =
        std::to_string(file_size_or_zero(experiment_root, get_optional(config, "checkpoint_tensors")));

    if (!sources_ok) {
        manifest.strict_public_input_equivalent = false;
        manifest.failure_step = "source_manifest_check";
        manifest.block_reason = "public_input_encoding_mismatch";
        return manifest;
    }

    if (route == "gpu") {
        manifest.gpu_backend_active = gpu_probe.gpu_backend_active;
        manifest.gpu_device_name = gpu_probe.gpu_device_name;
        manifest.gpu_device_count = gpu_probe.gpu_device_count;
        manifest.gpu_backend_name = gpu_probe.gpu_backend_name.empty() ? "zkDL CUDA primitive backend" : gpu_probe.gpu_backend_name;
        manifest.cpu_fallback_used = gpu_probe.cpu_fallback_used;
        if (!gpu_probe.loaded || gpu_probe.primitive_probe_status != "ok") {
            manifest.failure_step = "zkdl_cuda_primitive_probe";
            manifest.block_reason = gpu_probe.gpu_device_count == 0 ? "cuda_device_not_found" : "zkdl_gpu_backend_unavailable";
            return manifest;
        }
        manifest.failure_step = "zkdl_gat_full_forward_capability_check";
        manifest.block_reason = "gat_forward_constraint_not_representable_in_current_zkdl";
        return manifest;
    }

    manifest.failure_step = "zkdl_cpu_backend_capability_check";
    manifest.block_reason = "zkdl_cpu_backend_unavailable";
    return manifest;
}

void write_manifest(const Manifest& manifest, const std::string& output_root) {
    const auto output_dir = std::filesystem::path(output_root) / manifest.dataset / manifest.backend;
    std::filesystem::create_directories(output_dir);
    const auto output_path = output_dir / "run_manifest.json";
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("failed to write " + output_path.string());
    }
    out << "{\n";
    write_json_string(out, "dataset", manifest.dataset);
    write_json_string(out, "route_id", manifest.route_id);
    write_json_string(out, "backend", manifest.backend);
    write_json_string(out, "status", manifest.status);
    write_json_bool(out, "strict_public_input_equivalent", manifest.strict_public_input_equivalent);
    write_json_bool(out, "full_gat_forward_in_zkdl_proof", manifest.full_gat_forward_in_zkdl_proof);
    write_json_bool(out, "final_logits_derivation_complete", manifest.final_logits_derivation_complete);
    write_json_bool(out, "no_precomputed_logits_bypass", manifest.no_precomputed_logits_bypass);
    write_json_optional_double(out, "commitment_time_ms", manifest.commitment_time_ms);
    write_json_optional_double(out, "prove_time_ms", manifest.prove_time_ms);
    write_json_optional_double(out, "verify_time_ms", manifest.verify_time_ms);
    write_json_optional_size(out, "proof_size_bytes", manifest.proof_size_bytes);
    write_json_string(out, "block_reason", manifest.block_reason);
    write_json_string(out, "failure_step", manifest.failure_step);
    write_json_bool(out, "gpu_backend_active", manifest.gpu_backend_active);
    write_json_string(out, "gpu_device_name", manifest.gpu_device_name);
    write_json_int(out, "gpu_device_count", manifest.gpu_device_count);
    write_json_string(out, "gpu_backend_name", manifest.gpu_backend_name);
    write_json_bool(out, "cpu_fallback_used", manifest.cpu_fallback_used);
    out << "  \"capability_gaps\": [";
    for (std::size_t i = 0; i < manifest.capability_gaps.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << "\"" << json_escape(manifest.capability_gaps[i]) << "\"";
    }
    out << "],\n";
    write_string_map(out, "source_files", manifest.source_files);
    write_string_map(out, "source_sha256", manifest.source_sha256);
    out << "  \"dataset_stats\": {\n";
    std::size_t index = 0;
    for (const auto& [key, value] : manifest.dataset_stats) {
        out << "    \"" << json_escape(key) << "\": \"" << json_escape(value) << "\"";
        if (++index != manifest.dataset_stats.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  }\n";
    out << "}\n";
}

int run_cli(int argc, char** argv) {
    std::string route;
    std::string config_path;
    std::string output_root = "runs";
    std::string experiment_root = ".";
    std::string gpu_probe_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + flag);
            }
            return argv[++i];
        };
        if (arg == "--route") {
            route = require_value(arg);
        } else if (arg == "--config") {
            config_path = require_value(arg);
        } else if (arg == "--output-root") {
            output_root = require_value(arg);
        } else if (arg == "--experiment-root") {
            experiment_root = require_value(arg);
        } else if (arg == "--gpu-probe") {
            gpu_probe_path = require_value(arg);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (route != "cpu" && route != "gpu") {
        throw std::runtime_error("--route must be cpu or gpu");
    }
    if (config_path.empty()) {
        throw std::runtime_error("--config is required");
    }
    const auto config = load_dataset_config(config_path);
    const auto gpu_probe = route == "gpu" ? load_gpu_probe(gpu_probe_path) : GpuProbe{};
    const auto manifest = build_manifest(config, route, experiment_root, gpu_probe);
    write_manifest(manifest, output_root);
    std::cout
        << "dataset=" << manifest.dataset
        << " route_id=" << manifest.route_id
        << " backend=" << manifest.backend
        << " status=" << manifest.status
        << " failure_step=" << manifest.failure_step
        << " block_reason=" << manifest.block_reason
        << '\n';
    return 0;
}

}  // namespace zkdl_gat
