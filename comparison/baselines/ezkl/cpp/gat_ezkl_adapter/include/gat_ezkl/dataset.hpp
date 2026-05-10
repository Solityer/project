#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "gat_ezkl/config.hpp"

namespace gat_ezkl
{
    struct Edge
    {
        std::size_t src = 0;
        std::size_t dst = 0;
        std::size_t graph_id = 0;  // 多图情况下，标识属于哪个子图
        std::size_t stable_index = 0; // 去重/排序后的全局边索引，用于确定性序列化
    };

    struct GraphDataset
    {
        std::string name;
        std::size_t num_nodes = 0;
        std::size_t num_features = 0;
        std::size_t num_classes = 0;
        std::vector<std::vector<double>> features_fp; // 浮点特征矩阵 [node][feature]
        std::vector<int> labels;
        std::vector<Edge> edges;
        std::size_t graph_count = 1;       // 子图数量（单图为1）
        std::vector<std::size_t> node_ptr; // CSR 格式：每个子图的节点起始偏移
        std::vector<std::size_t> edge_ptr; // CSR 格式：每个子图的边起始偏移
        std::string task_type = "transductive_node_classification";
        std::string report_unit = "node";
    };

    struct NormalizedGraph
    {
        std::string name;
        std::size_t n_total = 0; // 所有子图节点总数（等于node_count，保留兼容）
        std::size_t node_count = 0;
        std::size_t edge_count = 0;
        std::size_t graph_count = 1;
        std::size_t feature_dim = 0;
        std::size_t class_count = 0;
        std::vector<std::size_t> absolute_ids; // 归一化前的原始节点 ID 映射
        std::vector<std::vector<double>> features_fp;
        std::vector<int> labels;
        std::vector<Edge> edges;
        std::vector<std::size_t> src;
        std::vector<std::size_t> dst;
        std::vector<std::size_t> node_ptr; // CSR 格式节点指针
        std::vector<std::size_t> edge_ptr;
    };

    // 从项目缓存目录加载数据集（如果不存在则可能触发下载/预处理）
    GraphDataset load_dataset_from_cache(const std::filesystem::path &project_root, const AppConfig &config);

    // 核心预处理：去重、排序、CSR 构建、特征/标签对齐
    NormalizedGraph normalize_full_graph(const GraphDataset &dataset, const AppConfig &config);

    // 公共输入元素数 = 特征维度 × 节点数，用于 ZK 电路中分配见证
    std::size_t public_input_element_count(const NormalizedGraph &graph);

    // 公共输出元素数 = 类别数 × 节点数，用于校验证明的输出维度
    std::size_t public_output_element_count(const NormalizedGraph &graph);
}