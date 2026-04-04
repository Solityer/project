#include "gatzk/util/config.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace gatzk::util {
namespace {

std::string trim(std::string value) {
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), value.end());
    return value;
}

bool parse_bool(const std::string& value) {
    return value == "true" || value == "1" || value == "yes";
}

std::string strip_wrappers(std::string value) {
    value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) || ch == '[' || ch == ']' || ch == '(' || ch == ')';
    }), value.end());
    return value;
}

std::vector<std::string> split_tokens(const std::string& value) {
    std::vector<std::string> out;
    std::string current;
    for (const auto ch : value) {
        if (ch == ',' || ch == ';') {
            if (!current.empty()) {
                out.push_back(current);
                current.clear();
            }
            continue;
        }
        current.push_back(ch);
    }
    if (!current.empty()) {
        out.push_back(current);
    }
    return out;
}

std::vector<std::size_t> parse_size_list(const std::string& value) {
    const auto compact = strip_wrappers(value);
    if (compact.empty()) {
        return {};
    }
    std::vector<std::size_t> out;
    for (const auto& token : split_tokens(compact)) {
        out.push_back(std::stoull(token));
    }
    return out;
}

std::vector<HiddenLayerProfile> parse_hidden_profile(const std::string& value) {
    const auto compact = strip_wrappers(value);
    if (compact.empty()) {
        return {};
    }
    std::vector<HiddenLayerProfile> out;
    for (const auto& token : split_tokens(compact)) {
        const auto sep = token.find_first_of("x:X:");
        if (sep == std::string::npos) {
            throw std::runtime_error("invalid hidden_profile token: " + token);
        }
        out.push_back(HiddenLayerProfile{
            .head_count = std::stoull(token.substr(0, sep)),
            .head_dim = std::stoull(token.substr(sep + 1)),
        });
    }
    return out;
}

void fail_if_present(
    const std::unordered_map<std::string, std::string>& entries,
    const std::set<std::string>& keys) {
    for (const auto& key : keys) {
        if (entries.contains(key)) {
            throw std::runtime_error("unsupported legacy config field: " + key);
        }
    }
}

void validate_config(const AppConfig& config) {
    if (config.hidden_profile.empty()) {
        throw std::runtime_error("hidden_profile must be set explicitly");
    }
    if (config.layer_count == 0) {
        throw std::runtime_error("L must be set explicitly and be >= 2");
    }
    if (config.layer_count != config.hidden_profile.size() + 1) {
        throw std::runtime_error(
            "L=" + std::to_string(config.layer_count)
            + " conflicts with hidden_profile size=" + std::to_string(config.hidden_profile.size()));
    }
    if (!config.d_in_profile.empty() && config.d_in_profile.size() != config.hidden_profile.size()) {
        throw std::runtime_error(
            "d_in_profile size=" + std::to_string(config.d_in_profile.size())
            + " conflicts with hidden_profile size=" + std::to_string(config.hidden_profile.size()));
    }
    if (!config.d_in_profile.empty() && config.d_in_profile.front() == 0) {
        throw std::runtime_error("d_in_profile entries must be non-zero");
    }
    if (config.K_out == 0) {
        throw std::runtime_error("K_out must be >= 1");
    }
    if (config.num_classes == 0) {
        throw std::runtime_error("C/num_classes must be >= 1");
    }
    for (std::size_t layer_index = 0; layer_index < config.hidden_profile.size(); ++layer_index) {
        const auto& layer = config.hidden_profile[layer_index];
        if (layer.head_count == 0 || layer.head_dim == 0) {
            throw std::runtime_error("hidden_profile entries must have non-zero head_count and head_dim");
        }
        if (layer_index + 1 < config.hidden_profile.size() && !config.d_in_profile.empty()) {
            const auto expected = layer.head_count * layer.head_dim;
            const auto actual = config.d_in_profile[layer_index + 1];
            if (expected != actual) {
                throw std::runtime_error(
                    "d_in_profile[" + std::to_string(layer_index + 1) + "]="
                    + std::to_string(actual)
                    + " conflicts with previous hidden concat width="
                    + std::to_string(expected));
            }
        }
    }
}

}  // namespace

AppConfig load_config(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open config: " + path);
    }

    std::unordered_map<std::string, std::string> entries;
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
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error("invalid config line: " + line);
        }
        const auto key = trim(line.substr(0, eq));
        const auto value = trim(line.substr(eq + 1));
        entries[key] = value;
    }

    AppConfig config;
    fail_if_present(entries, {"hidden_heads", "checkpoint_dir"});
    const auto absolute = std::filesystem::absolute(path);
    config.config_dir = absolute.parent_path().string();
    config.project_root = absolute.parent_path().parent_path().string();
    if (entries.contains("dataset")) config.dataset = entries["dataset"];
    if (entries.contains("data_root")) config.data_root = entries["data_root"];
    if (entries.contains("cache_root")) config.cache_root = entries["cache_root"];
    if (entries.contains("export_dir")) config.export_dir = entries["export_dir"];
    if (entries.contains("checkpoint_bundle")) config.checkpoint_bundle = entries["checkpoint_bundle"];
    if (entries.contains("reference_output_dir")) config.reference_output_dir = entries["reference_output_dir"];
    if (entries.contains("task_type")) config.task_type = entries["task_type"];
    if (entries.contains("report_unit")) config.report_unit = entries["report_unit"];
    if (entries.contains("batching_rule")) config.batching_rule = entries["batching_rule"];
    if (entries.contains("subgraph_rule")) config.subgraph_rule = entries["subgraph_rule"];
    if (entries.contains("self_loop_rule")) config.self_loop_rule = entries["self_loop_rule"];
    if (entries.contains("edge_sort_rule")) config.edge_sort_rule = entries["edge_sort_rule"];
    if (entries.contains("chunking_rule")) config.chunking_rule = entries["chunking_rule"];
    if (entries.contains("quant_cfg_id")) config.quant_cfg_id = entries["quant_cfg_id"];
    if (entries.contains("model_arch_id")) config.model_arch_id = entries["model_arch_id"];
    if (entries.contains("model_param_id")) config.model_param_id = entries["model_param_id"];
    if (entries.contains("static_table_id")) config.static_table_id = entries["static_table_id"];
    if (entries.contains("degree_bound_id")) config.degree_bound_id = entries["degree_bound_id"];
    if (entries.contains("hidden_dim")) config.hidden_dim = std::stoull(entries["hidden_dim"]);
    if (entries.contains("num_classes")) config.num_classes = std::stoull(entries["num_classes"]);
    if (entries.contains("range_bits")) config.range_bits = std::stoull(entries["range_bits"]);
    if (entries.contains("seed")) config.seed = std::stoull(entries["seed"]);
    if (entries.contains("local_nodes")) config.local_nodes = std::stoull(entries["local_nodes"]);
    if (entries.contains("center_node")) config.center_node = std::stoull(entries["center_node"]);
    if (entries.contains("L")) config.layer_count = std::stoull(entries["L"]);
    if (entries.contains("K_out")) config.K_out = std::stoull(entries["K_out"]);
    if (entries.contains("batch_graphs")) config.batch_graphs = std::stoull(entries["batch_graphs"]);
    if (entries.contains("allow_synthetic_model")) config.allow_synthetic_model = parse_bool(entries["allow_synthetic_model"]);
    if (entries.contains("dump_trace")) config.dump_trace = parse_bool(entries["dump_trace"]);
    if (entries.contains("auto_prepare_dataset")) config.auto_prepare_dataset = parse_bool(entries["auto_prepare_dataset"]);
    if (entries.contains("prove_enabled")) config.prove_enabled = parse_bool(entries["prove_enabled"]);
    if (entries.contains("symmetrize_edges")) config.symmetrize_edges = parse_bool(entries["symmetrize_edges"]);
    if (entries.contains("deduplicate_edges")) config.deduplicate_edges = parse_bool(entries["deduplicate_edges"]);
    if (entries.contains("hidden_profile")) config.hidden_profile = parse_hidden_profile(entries["hidden_profile"]);
    if (entries.contains("d_in_profile")) config.d_in_profile = parse_size_list(entries["d_in_profile"]);
    if (config.hidden_profile.empty() && config.hidden_dim != 0) {
        config.hidden_profile = {{1, config.hidden_dim}};
    }
    if (config.layer_count == 0 && !config.hidden_profile.empty()) {
        config.layer_count = config.hidden_profile.size() + 1;
    }
    if (config.quant_cfg_id.empty()) {
        config.quant_cfg_id = "range_bits=" + std::to_string(config.range_bits);
    }
    if (config.static_table_id.empty()) {
        config.static_table_id = "tables:lrelu+elu+exp+range";
    }
    if (config.degree_bound_id.empty()) {
        config.degree_bound_id = "auto";
    }
    validate_config(config);
    return config;
}

}  // namespace gatzk::util
