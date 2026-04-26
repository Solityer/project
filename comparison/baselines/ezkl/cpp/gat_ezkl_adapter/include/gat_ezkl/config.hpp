#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gat_ezkl {

struct HiddenLayerProfile {
    std::size_t head_count = 0;
    std::size_t head_dim = 0;
};

struct AppConfig {
    std::string dataset;
    std::string data_root = "data";
    std::string cache_root = "data/cache";
    std::string export_dir;
    std::string checkpoint_bundle;
    std::string reference_output_dir;
    std::string task_type;
    std::string report_unit;
    std::string batching_rule = "whole_graph_single";
    std::string subgraph_rule = "whole_graph";
    std::string self_loop_rule = "per_node";
    std::string edge_sort_rule = "edge_gid_then_dst_stable";
    std::string chunking_rule = "none";
    std::string quant_cfg_id;
    std::string model_arch_id;
    std::string model_param_id;
    std::string static_table_id;
    std::string degree_bound_id;
    std::size_t hidden_dim = 0;
    std::size_t num_classes = 0;
    std::size_t range_bits = 0;
    std::uint64_t seed = 0;
    std::size_t layer_count = 0;
    std::size_t K_out = 1;
    std::size_t batch_graphs = 1;
    bool allow_synthetic_model = false;
    bool dump_trace = false;
    bool auto_prepare_dataset = false;
    bool symmetrize_edges = false;
    bool deduplicate_edges = true;
    std::vector<HiddenLayerProfile> hidden_profile;
    std::vector<std::size_t> d_in_profile;
};

struct DatasetSpec {
    std::string run_id;
    std::string config_file;
    std::string main_dataset_name;
    std::string display_name;
};

AppConfig load_config_file(const std::filesystem::path& path);
DatasetSpec official_dataset_spec(const std::string& run_id);
std::vector<DatasetSpec> official_dataset_specs();
std::filesystem::path config_path_for_dataset(
    const std::filesystem::path& project_root,
    const std::string& run_id);
void validate_full_dataset_contract(const AppConfig& config, const DatasetSpec& spec);

}  // namespace gat_ezkl
