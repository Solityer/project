#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "gat_ezkl/config.hpp"

namespace gat_ezkl {

struct TensorInfo {
    std::vector<std::size_t> shape;
    std::size_t value_count = 0;
};

struct CheckpointHeadSpec {
    std::size_t layer_index = 0;
    std::size_t local_head_index = 0;
    std::size_t global_head_index = 0;
    std::string seq_kernel;
    std::string attn_src_kernel;
    std::string attn_dst_kernel;
    std::string attn_src_bias;
    std::string attn_dst_bias;
    std::string output_bias;
};

struct CheckpointOutputHeadSpec {
    std::size_t head_index = 0;
    std::string seq_kernel;
    std::string attn_src_kernel;
    std::string attn_dst_kernel;
    std::string attn_src_bias;
    std::string attn_dst_bias;
    std::string output_bias;
};

struct CheckpointBundle {
    std::filesystem::path root;
    std::string family_schema_version;
    std::string output_average_rule;
    std::string model_arch_id;
    std::string model_param_id;
    std::string quant_cfg_id;
    std::string static_table_id;
    std::string degree_bound_id;
    std::size_t layer_count = 0;
    std::size_t K_out = 1;
    std::size_t class_count = 0;
    std::vector<std::size_t> d_in_profile;
    std::vector<HiddenLayerProfile> hidden_profile;
    std::vector<CheckpointHeadSpec> hidden_heads;
    std::vector<CheckpointOutputHeadSpec> output_heads;
    std::unordered_map<std::string, TensorInfo> tensors;
};

CheckpointBundle load_checkpoint_bundle(const std::filesystem::path& bundle_root);
void validate_checkpoint_against_config(const CheckpointBundle& bundle, AppConfig* config);

}  // namespace gat_ezkl
