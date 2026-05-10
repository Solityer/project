#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gat_ezkl/checkpoint.hpp"
#include "gat_ezkl/config.hpp"
#include "gat_ezkl/dataset.hpp"

namespace gat_ezkl 
{
    // FNV-1a 哈希，对原始字节序列计算哈希，seed 用于防碰撞或分桶
    std::uint64_t fnv1a64_bytes(const void* data, std::size_t len, std::uint64_t seed = 1469598103934665603ULL);
    std::uint64_t fnv1a64_string(const std::string& text, std::uint64_t seed = 1469598103934665603ULL);

    // 将 uint64 转为 16 进制字符串
    std::string hex_u64(std::uint64_t value);
    std::string hash_file_hex(const std::filesystem::path& path);
    std::string hash_files_hex(const std::vector<std::filesystem::path>& paths);

    // 对 AppConfig 做确定性哈希，用于唯一标识一个配置
    std::string hash_config_hex(const AppConfig& config);
    std::string hash_graph_hex(const NormalizedGraph& graph);
    std::string hash_shape_hex(const NormalizedGraph& graph, const CheckpointBundle& bundle);
    std::string hash_metadata_hex(const std::string& metadata_json);
    std::string hash_output_shape_hex(const NormalizedGraph& graph);
}