#include "gatzk/util/config.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
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
    const auto absolute = std::filesystem::absolute(path);
    config.config_dir = absolute.parent_path().string();
    config.project_root = absolute.parent_path().parent_path().string();
    if (entries.contains("dataset")) config.dataset = entries["dataset"];
    if (entries.contains("data_root")) config.data_root = entries["data_root"];
    if (entries.contains("cache_root")) config.cache_root = entries["cache_root"];
    if (entries.contains("export_dir")) config.export_dir = entries["export_dir"];
    if (entries.contains("checkpoint_bundle")) config.checkpoint_bundle = entries["checkpoint_bundle"];
    if (entries.contains("reference_output_dir")) config.reference_output_dir = entries["reference_output_dir"];
    if (entries.contains("hidden_dim")) config.hidden_dim = std::stoull(entries["hidden_dim"]);
    if (entries.contains("num_classes")) config.num_classes = std::stoull(entries["num_classes"]);
    if (entries.contains("range_bits")) config.range_bits = std::stoull(entries["range_bits"]);
    if (entries.contains("seed")) config.seed = std::stoull(entries["seed"]);
    if (entries.contains("local_nodes")) config.local_nodes = std::stoull(entries["local_nodes"]);
    if (entries.contains("center_node")) config.center_node = std::stoull(entries["center_node"]);
    if (entries.contains("allow_synthetic_model")) config.allow_synthetic_model = parse_bool(entries["allow_synthetic_model"]);
    if (entries.contains("dump_trace")) config.dump_trace = parse_bool(entries["dump_trace"]);
    if (entries.contains("auto_prepare_dataset")) config.auto_prepare_dataset = parse_bool(entries["auto_prepare_dataset"]);
    if (entries.contains("prove_enabled")) config.prove_enabled = parse_bool(entries["prove_enabled"]);
    return config;
}

}  // namespace gatzk::util
