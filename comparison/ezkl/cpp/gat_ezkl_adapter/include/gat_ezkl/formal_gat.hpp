#pragma once

#include <filesystem>

#include "gat_ezkl/checkpoint.hpp"
#include "gat_ezkl/config.hpp"
#include "gat_ezkl/dataset.hpp"

namespace gat_ezkl {

struct FormalProfileIds {
    std::string quant_cfg_id;
    std::string static_tables_id;
    std::string backend_aux_tables_id;
};

// Emits the canonical integer inputs, exact-table generator specification,
// registered model bounds, integer no-wrap ledger, and an independently
// computed complete fixed-point GAT forward result consumed by the Halo2 path.
FormalProfileIds write_formal_gat_artifacts(
    const std::filesystem::path& output_dir,
    const AppConfig& config,
    const NormalizedGraph& graph,
    const CheckpointBundle& checkpoint,
    std::size_t max_in_degree);

}  // namespace gat_ezkl
