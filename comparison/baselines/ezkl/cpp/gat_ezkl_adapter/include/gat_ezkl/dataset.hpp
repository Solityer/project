#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "gat_ezkl/config.hpp"

namespace gat_ezkl {

struct Edge {
    std::size_t src = 0;
    std::size_t dst = 0;
    std::size_t graph_id = 0;
    std::size_t stable_index = 0;
};

struct GraphDataset {
    std::string name;
    std::size_t num_nodes = 0;
    std::size_t num_features = 0;
    std::size_t num_classes = 0;
    std::vector<std::vector<double>> features_fp;
    std::vector<int> labels;
    std::vector<Edge> edges;
    std::size_t graph_count = 1;
    std::vector<std::size_t> node_ptr;
    std::vector<std::size_t> edge_ptr;
    std::string task_type = "transductive_node_classification";
    std::string report_unit = "node";
};

struct NormalizedGraph {
    std::string name;
    std::size_t n_total = 0;
    std::size_t node_count = 0;
    std::size_t edge_count = 0;
    std::size_t graph_count = 1;
    std::size_t feature_dim = 0;
    std::size_t class_count = 0;
    std::vector<std::size_t> absolute_ids;
    std::vector<std::vector<double>> features_fp;
    std::vector<int> labels;
    std::vector<Edge> edges;
    std::vector<std::size_t> src;
    std::vector<std::size_t> dst;
    std::vector<std::size_t> node_ptr;
    std::vector<std::size_t> edge_ptr;
};

GraphDataset load_dataset_from_cache(
    const std::filesystem::path& project_root,
    const AppConfig& config);
NormalizedGraph normalize_full_graph(const GraphDataset& dataset, const AppConfig& config);
std::size_t public_input_element_count(const NormalizedGraph& graph);
std::size_t public_output_element_count(const NormalizedGraph& graph);

}  // namespace gat_ezkl
