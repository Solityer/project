#pragma once
// 本文件定义数据层进入 formal 主线后的对象模型，
// 区分原始数据集、规范化后的局部图，以及需要进入公开元数据的图结构信息。

#include <cstddef>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::data {

struct Edge {
    // `graph_id` / `stable_index` 会进入 batch 规范化与稳定排序规则。
    std::size_t src = 0;
    std::size_t dst = 0;
    std::size_t graph_id = 0;
    std::size_t stable_index = 0;
};

// 公开输入层面的图结构摘要，会被 context / metadata / verifier 一致消费。
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

// 原始数据集对象，保存完整图数据与量化前后的特征视图。
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

// 规范化后送入 formal prove / verify 主线的局部图对象。
// 这里同时持有 witness 侧特征、公开输入摘要和 batch 切分信息。
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
