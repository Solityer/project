#pragma once

#include <filesystem>
#include <string>

#include "gat_ezkl/checkpoint.hpp"
#include "gat_ezkl/config.hpp"
#include "gat_ezkl/dataset.hpp"
#include "gat_ezkl/gat.hpp"
#include "gat_ezkl/quant.hpp"

namespace gat_ezkl {

std::string json_escape(const std::string& value);
std::string now_utc_iso8601();
void write_text_file(const std::filesystem::path& path, const std::string& text);

std::string build_metadata_json(
    const std::string& run_id,
    const AppConfig& config,
    const NormalizedGraph& graph,
    const CheckpointBundle& bundle,
    const QuantConfig& quant,
    const GatSemanticSummary& gat_summary,
    const std::string& graph_hash,
    const std::string& checkpoint_hash,
    const std::string& shape_hash);

std::string build_model_info_json(
    const AppConfig& config,
    const CheckpointBundle& bundle,
    const GatSemanticSummary& gat_summary,
    const std::string& checkpoint_hash,
    const std::string& shape_hash);

std::string build_input_summary_json(
    const NormalizedGraph& graph,
    const QuantConfig& quant,
    const std::string& graph_hash);

}  // namespace gat_ezkl
