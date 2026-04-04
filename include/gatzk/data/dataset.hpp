#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::data {

struct Edge {
    std::size_t src = 0;
    std::size_t dst = 0;
    std::size_t graph_id = 0;
    std::size_t stable_index = 0;
};

struct NormalizedGraphInput {
    std::size_t N_total = 0;
    std::size_t N = 0;
    std::size_t E = 0;
    std::size_t G_batch = 1;
    std::vector<std::size_t> I;
    std::vector<std::size_t> src;
    std::vector<std::size_t> dst;
    std::vector<std::size_t> node_ptr;
    std::vector<std::size_t> edge_ptr;
};

struct GraphDataset {
    std::string name;
    std::size_t num_nodes = 0;
    std::size_t num_features = 0;
    std::size_t num_classes = 0;
    std::vector<std::vector<double>> features_fp;
    std::vector<std::vector<algebra::FieldElement>> features;
    std::vector<int> labels;
    std::vector<Edge> edges;
    std::size_t graph_count = 1;
    std::vector<std::size_t> node_ptr;
    std::vector<std::size_t> edge_ptr;
    std::string task_type = "transductive_node_classification";
    std::string report_unit = "node";
};

struct LocalGraph {
    std::string name;
    std::size_t num_nodes = 0;
    std::size_t num_features = 0;
    std::size_t num_classes = 0;
    std::vector<std::size_t> absolute_ids;
    std::vector<std::vector<double>> features_fp;
    std::vector<std::vector<algebra::FieldElement>> features;
    std::vector<int> labels;
    std::vector<Edge> edges;
    NormalizedGraphInput public_input;
    std::size_t graph_count = 1;
    std::vector<std::size_t> node_ptr;
    std::vector<std::size_t> edge_ptr;
    std::string task_type = "transductive_node_classification";
    std::string report_unit = "node";
    std::string batching_rule = "whole_graph_single";
    std::string subgraph_rule = "whole_graph";
    std::string self_loop_rule = "per_node";
    std::string edge_sort_rule = "edge_gid_then_dst_stable";
    std::string chunking_rule = "none";
};

}  // namespace gatzk::data
