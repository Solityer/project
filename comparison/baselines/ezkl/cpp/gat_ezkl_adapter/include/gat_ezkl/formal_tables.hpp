#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gat_ezkl/config.hpp"

namespace gat_ezkl {

struct InverseQuantPolicy {
    std::string inverse_policy = "quantized_inverse_lookup";
    std::int64_t sum_scale = 16;
    std::int64_t inv_scale = 16;
    std::int64_t alpha_scale = 16;
    std::int64_t inverse_scale_product = 256;
    std::int64_t inverse_band = 0;
    std::string rounding_rule = "half_away_from_zero";
    std::string alpha_rescale_policy = "rounded_product_div_scale_with_quantization_band";
    std::int64_t alpha_rescale_divisor = 16;
    std::int64_t alpha_band = 2;
    std::string policy_hash;
};

struct InverseTableEntry {
    std::int64_t sum_q = 0;
    std::int64_t inv_q = 0;
};

struct FormalTableHashes {
    std::string range_hash;
    std::string lrelu_hash;
    std::string elu_hash;
    std::string exp_hash;
    std::string inv_hash;
    std::string inverse_policy_hash;
    std::string table_hash;
    std::string bin_hash;
};

struct FormalTablesSummary {
    std::string source = "formal_quant_cfg_derived";
    std::string table_policy = "derived_from_main_formal_static_tables_v1";
    std::string quant_cfg_id;
    std::string static_table_id;
    std::size_t range_bits = 0;
    std::size_t range_entries = 0;
    std::size_t lrelu_entries = 0;
    std::size_t elu_entries = 0;
    std::size_t exp_entries = 0;
    std::size_t inv_entries = 0;
    std::int64_t signed_bound = 0;
    std::size_t lrelu_bound = 0;
    std::size_t elu_bound = 0;
    std::size_t range_size = 0;
    std::size_t elu_band = 1;
    std::size_t exp_band = 2;
    std::string scale_policy = "trace_scalar_scale_16";
    InverseQuantPolicy inverse;
    std::string inverse_policy = "quantized_inverse_lookup";
    FormalTableHashes hashes;
};

InverseQuantPolicy derive_inverse_quant_policy(const AppConfig& config);
std::vector<InverseTableEntry> generate_inverse_table(
    const InverseQuantPolicy& policy,
    std::size_t range_size);
std::int64_t quantized_inverse_for_sum(
    std::int64_t sum_q,
    const InverseQuantPolicy& policy);
bool validate_inverse_witness_against_policy(
    std::int64_t sum_q,
    std::int64_t inv_q,
    const InverseQuantPolicy& policy,
    std::int64_t* expected_inv_q = nullptr);
bool validate_alpha_rescale(
    std::int64_t u_q,
    std::int64_t inv_q,
    std::int64_t alpha_q,
    const InverseQuantPolicy& policy,
    std::int64_t* expected_alpha_q = nullptr);

FormalTablesSummary write_formal_table_artifacts(
    const std::filesystem::path& output_dir,
    const AppConfig& config);

std::string formal_tables_json(const FormalTablesSummary& summary);
std::string formal_table_hashes_json(const FormalTablesSummary& summary);
std::string formal_table_binding_json(const FormalTablesSummary& summary);

}  // namespace gat_ezkl
