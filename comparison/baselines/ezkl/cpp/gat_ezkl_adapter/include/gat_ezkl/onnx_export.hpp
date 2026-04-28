#pragma once

#include <filesystem>
#include <string>

#include "gat_ezkl/checkpoint.hpp"
#include "gat_ezkl/config.hpp"
#include "gat_ezkl/dataset.hpp"
#include "gat_ezkl/gat.hpp"
#include "gat_ezkl/segment_softmax.hpp"

namespace gat_ezkl {

struct ExportDecision {
    std::string route_id = "ezkl_gat_native_compressed_csr_attention_full";
    std::string status = "ready";
    std::string block_reason;
    std::string limitation;
    bool strict_public_input_equivalent = false;
    bool full_gat_forward_in_ezkl_proof = false;
    bool network_onnx_is_blocked_artifact = true;
};

ExportDecision export_native_csr_segment_attention_gat(
    const std::filesystem::path& output_dir,
    const std::string& route_id,
    const AppConfig& config,
    const NormalizedGraph& graph,
    const CheckpointBundle& bundle,
    const GatSemanticSummary& summary);

}  // namespace gat_ezkl
