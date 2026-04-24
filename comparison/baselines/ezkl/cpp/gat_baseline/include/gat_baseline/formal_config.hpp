#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gat_baseline {

struct HiddenLayerProfile {
    std::size_t head_count = 0;
    std::size_t head_dim = 0;
};

struct FormalConfig {
    std::string dataset = "toy";
    std::filesystem::path data_root = "data";
    std::filesystem::path cache_root = "data/cache";
    std::filesystem::path export_dir = "runs/default";
    std::filesystem::path config_path;
    std::filesystem::path config_dir = ".";
    std::filesystem::path project_root = ".";
    std::filesystem::path checkpoint_bundle;
    std::filesystem::path reference_output_dir = "runs/reference";
    std::string task_type = "transductive_node_classification";
    std::string report_unit = "node";
    std::string batching_rule = "whole_graph_single";
    std::string subgraph_rule = "whole_graph";
    std::string self_loop_rule = "per_node";
    std::string edge_sort_rule = "edge_gid_then_dst_stable";
    std::string chunking_rule = "none";
    std::string quant_cfg_id;
    std::string model_arch_id;
    std::string model_param_id;
    std::string static_table_id;
    std::string degree_bound_id = "auto";
    std::size_t hidden_dim = 4;
    std::size_t num_classes = 2;
    std::size_t range_bits = 10;
    std::uint64_t seed = 7;
    std::size_t layer_count = 0;
    std::size_t output_head_count = 1;
    std::size_t batch_graphs = 1;
    bool allow_synthetic_model = false;
    bool dump_trace = true;
    bool auto_prepare_dataset = false;
    bool deduplicate_edges = true;
    bool symmetrize_edges = false;
    std::vector<HiddenLayerProfile> hidden_profile;
    std::vector<std::size_t> d_in_profile;
};

FormalConfig load_formal_config(const std::filesystem::path& path);
std::filesystem::path resolve_project_path(const FormalConfig& config, const std::filesystem::path& raw);
std::filesystem::path dataset_cache_path(const FormalConfig& config);

}  // namespace gat_baseline
