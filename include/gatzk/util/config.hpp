#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace gatzk::util {

struct HiddenLayerProfile {
    std::size_t head_count = 0;
    std::size_t head_dim = 0;
};

struct AppConfig {
    std::string dataset = "toy";
    std::string data_root = "data";
    std::string cache_root = "data/cache";
    std::string export_dir = "runs/default";
    std::string config_dir = ".";
    std::string project_root = ".";
    std::string checkpoint_bundle;
    std::string reference_output_dir = "runs/reference";
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
    std::string degree_bound_id;
    std::size_t hidden_dim = 4;
    std::size_t num_classes = 2;
    std::size_t range_bits = 10;
    std::uint64_t seed = 7;
    std::size_t layer_count = 0;
    std::size_t K_out = 1;
    std::size_t batch_graphs = 1;
    bool allow_synthetic_model = false;
    bool dump_trace = true;
    bool auto_prepare_dataset = false;
    bool symmetrize_edges = false;
    bool deduplicate_edges = true;
    std::vector<HiddenLayerProfile> hidden_profile;
    std::vector<std::size_t> d_in_profile;
};

AppConfig load_config(const std::string& path);

}  // namespace gatzk::util
