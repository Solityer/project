#pragma once

#include <filesystem>
#include <string>

#include "gat_ezkl/checkpoint.hpp"
#include "gat_ezkl/config.hpp"
#include "gat_ezkl/dataset.hpp"

namespace gat_ezkl {

void write_formal_csr_graph_artifacts(
    const std::filesystem::path& output_dir,
    const std::string& route_id,
    const AppConfig& config,
    const NormalizedGraph& graph,
    const CheckpointBundle& bundle,
    const std::string& graph_hash,
    const std::string& checkpoint_hash,
    const std::string& metadata_hash,
    const std::string& shape_hash,
    const std::string& formal_quant_cfg_id,
    const std::string& formal_static_tables_id,
    const std::string& backend_aux_tables_id,
    const std::string& security_profile_id);

}  // namespace gat_ezkl
