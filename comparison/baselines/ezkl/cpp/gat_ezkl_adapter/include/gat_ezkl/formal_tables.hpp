#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gat_ezkl/config.hpp"

namespace gat_ezkl 
{
    // 量化逆运算：控制 1/x 查表法的精度、缩放、舍入行为
    struct InverseQuantPolicy 
    {
        std::string inverse_policy = "quantized_inverse_lookup";
        std::int64_t sum_scale = 16;
        std::int64_t inv_scale = 16;
        std::int64_t alpha_scale = 16;  // 注意力系数缩放因子
        std::int64_t inverse_scale_product = 256;
        std::int64_t inverse_band = 0;
        std::string rounding_rule = "half_away_from_zero";
        std::string alpha_rescale_policy = "rounded_product_div_scale_with_quantization_band";
        std::int64_t alpha_rescale_divisor = 16;
        std::int64_t alpha_band = 2;
        std::string policy_hash;
    };

    // 逆运算查表条目：量化后的 sum -> 量化后的 1/sum
    struct InverseTableEntry 
    {
        std::int64_t sum_q = 0;
        std::int64_t inv_q = 0;
    };

    // 查找表的哈希集合，全部存 hex 字符串便于比对
    struct FormalTableHashes 
    {
        std::string range_hash;
        std::string lrelu_hash;
        std::string elu_hash;
        std::string exp_hash;
        std::string inv_hash;
        std::string inverse_policy_hash;  // 逆运算策略哈希
        std::string table_hash;
        std::string bin_hash;  // 二进制文件的哈希
    };

    // 查找表摘要
    struct FormalTablesSummary 
    {
        std::string source = "formal_quant_cfg_derived";
        std::string table_policy = "derived_from_main_formal_static_tables_v1";
        std::string inverse_table_policy = "degree_bound_exp_bound_sum_scale_v1";
        std::string quant_cfg_id;
        std::string static_table_id;
        std::string degree_bound_id;  // 度数界 ID（图的最大度）
        std::size_t range_bits = 0;
        std::size_t range_entries = 0;
        std::size_t lrelu_entries = 0;
        std::size_t elu_entries = 0;
        std::size_t exp_entries = 0;
        std::size_t inv_entries = 0;  // 逆运算表条目数
        std::int64_t signed_bound = 0;
        std::size_t lrelu_bound = 0;
        std::size_t elu_bound = 0;
        std::size_t range_size = 0;
        std::size_t degree_bound = 0;
        std::int64_t exp_bound = 0;
        std::int64_t derived_sum_bound = 0;
        std::size_t inverse_table_range_old = 0;
        std::size_t inverse_table_range_size = 0;
        std::size_t elu_band = 1;
        std::size_t exp_band = 2;
        std::string scale_policy = "trace_scalar_scale_16";
        InverseQuantPolicy inverse;
        std::string inverse_policy = "quantized_inverse_lookup";
        FormalTableHashes hashes;
    };

    InverseQuantPolicy derive_inverse_quant_policy(const AppConfig& config);

    // 生成完整的逆运算查找表
    std::vector<InverseTableEntry> generate_inverse_table(
        const InverseQuantPolicy& policy,
        std::size_t range_size);

    // 单次查表：给定量化后的 sum，返回量化后的 1/sum
    std::int64_t quantized_inverse_for_sum(
        std::int64_t sum_q,
        const InverseQuantPolicy& policy);

    // 验证电路见证中的 (sum_q, inv_q) 对是否符合逆运算策略
    bool validate_inverse_witness_against_policy(
        std::int64_t sum_q,
        std::int64_t inv_q,
        const InverseQuantPolicy& policy,
        std::int64_t* expected_inv_q = nullptr);
    
    // 验证 alpha 重缩放的正确性
    bool validate_alpha_rescale(
        std::int64_t u_q,
        std::int64_t inv_q,
        std::int64_t alpha_q,
        const InverseQuantPolicy& policy,
        std::int64_t* expected_alpha_q = nullptr);

    FormalTablesSummary write_formal_table_artifacts(
        const std::filesystem::path& output_dir,
        const AppConfig& config,
        std::size_t degree_bound);

    std::string formal_tables_json(const FormalTablesSummary& summary);
    std::string formal_table_hashes_json(const FormalTablesSummary& summary);
    std::string formal_table_binding_json(const FormalTablesSummary& summary);
}