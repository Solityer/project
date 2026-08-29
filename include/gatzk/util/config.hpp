#pragma once
// 本文件定义 formal 主线的运行配置对象，
// 所有 dataset、artifact、batching 与 proof contract 相关参数都从这里进入系统。

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <string>
#include <utility>
#include <vector>

namespace gatzk::util
{

    // 隐层族形状配置，决定每层头数和单头维度。
    struct HiddenLayerProfile
    {
        std::size_t head_count = 0;
        std::size_t head_dim = 0;
    };

    // Canonical fixed-point scale chain selected solely by quant_cfg_id.  The
    // vectors retain layer/head coordinates so no consumer can accidentally
    // reuse a scale from another layer or head.
    struct QuantizationScaleChain
    {
        std::uint64_t s_input = 0;
        std::uint64_t s_delta = 0;
        std::uint64_t s_exp = 0;
        std::vector<std::uint64_t> s_h;
        std::vector<std::vector<std::uint64_t>> s_w_hidden;
        std::vector<std::vector<std::uint64_t>> s_h_prime;
        std::vector<std::vector<std::uint64_t>> s_a_src_hidden;
        std::vector<std::vector<std::uint64_t>> s_a_dst_hidden;
        std::vector<std::uint64_t> s_agg_pre;
        std::vector<std::uint64_t> s_agg;
        std::vector<std::uint64_t> s_cat;
        std::uint64_t s_out_pre = 0;
        std::uint64_t s_out_head = 0;
        std::uint64_t s_out = 0;
        std::vector<std::uint64_t> s_w_output;
        std::vector<std::uint64_t> s_y_prime;
        std::vector<std::uint64_t> s_a_src_output;
        std::vector<std::uint64_t> s_a_dst_output;
        std::vector<std::uint64_t> s_b_output;

        // Deterministic, complete serialization used in PublicMetadata.
        std::string canonical_serialization() const;
    };

    // Resolve and validate the one registered scale chain for a formal ID.
    // Unknown IDs and shape mismatches fail closed.
    QuantizationScaleChain canonical_quantization_scale_chain(
        std::string_view quant_cfg_id,
        const std::vector<HiddenLayerProfile> &hidden_profile,
        std::size_t K_out);

    // 一次正式运行的统一配置。这里的字段会被 context、metadata 与正式 artifact 共同消费。
    struct AppConfig
    {
        std::string dataset;
        std::string data_root = "data";
        std::string cache_root = "data/cache";
        std::string export_dir = "runs/default";
        std::string config_dir = ".";
        std::string project_root = ".";
        std::string checkpoint_bundle;
        std::string task_type = "transductive_node_classification";
        std::string report_unit = "node";
        std::string batching_rule = "whole_graph_single";
        std::string subgraph_rule = "whole_graph";
        std::string self_loop_rule = "per_node";
        std::string edge_sort_rule = "edge_gid_then_dst_stable";
        std::string chunking_rule = "none";
        std::string quant_cfg_id;
        QuantizationScaleChain quantization;
        std::string model_arch_id;
        std::string model_param_id;
        std::string static_table_id;
        std::string degree_bound_id;
        std::string srs_type = "trusted_srs";
        std::string srs_id;
        std::string srs_path;
        std::string privacy_policy = "model_and_trace_hidden";
        std::string input_privacy = "public_graph_public_features";
        std::string output_mode = "public_logits";
        std::string opening_mode = "point_grouped_batch_opening";
        std::string binding_mode = "zk_subproof_for_crpc";
        std::string proof_task = "node_classification_full_gat";
        std::string security_profile_id = "zkgat_secure_profile_v1";
        std::size_t hidden_dim = 4;
        std::size_t num_classes = 2;
        std::size_t range_bits = 10;
        std::uint64_t seed = 7;
        std::size_t layer_count = 0;
        std::size_t K_out = 1;
        std::size_t batch_graphs = 1;
        bool dump_trace = true;
        bool auto_prepare_dataset = false;
        bool symmetrize_edges = false;
        bool deduplicate_edges = true;
        std::vector<HiddenLayerProfile> hidden_profile;
        std::vector<std::size_t> d_in_profile;
    };

    // 从 `.cfg` 文件解析正式运行配置，并执行字段级约束检查。
    AppConfig load_config(const std::string &path);

} // namespace gatzk::util
