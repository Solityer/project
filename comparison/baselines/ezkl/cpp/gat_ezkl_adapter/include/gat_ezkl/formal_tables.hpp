#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "gat_ezkl/config.hpp"

namespace gat_ezkl {

struct FormalTableHashes {
    std::string range_hash;
    std::string lrelu_hash;
    std::string elu_hash;
    std::string exp_hash;
    std::string inv_hash;
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
    std::string inverse_policy = "exact_sum_times_inv_gate_no_inverse_table";
    FormalTableHashes hashes;
};

FormalTablesSummary write_formal_table_artifacts(
    const std::filesystem::path& output_dir,
    const AppConfig& config);

std::string formal_tables_json(const FormalTablesSummary& summary);
std::string formal_table_hashes_json(const FormalTablesSummary& summary);
std::string formal_table_binding_json(const FormalTablesSummary& summary);

}  // namespace gat_ezkl
