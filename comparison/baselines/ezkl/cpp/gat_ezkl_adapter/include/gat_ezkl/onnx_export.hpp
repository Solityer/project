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
    std::string route_id = "strict_gat_full_onnx_v1";
    std::string status = "blocked";
    std::string block_reason = "ezkl_cannot_represent_grouped_softmax";
    std::string limitation;
    bool strict_public_input_equivalent = false;
    bool full_gat_forward_in_ezkl_proof = false;
    bool network_onnx_is_placeholder = true;
};

ExportDecision export_or_block_strict_gat_onnx(
    const std::filesystem::path& output_dir,
    const AppConfig& config,
    const NormalizedGraph& graph,
    const CheckpointBundle& bundle,
    const GatSemanticSummary& summary);

ExportDecision export_or_block_strict_segment_gat_onnx(
    const std::filesystem::path& output_dir,
    const AppConfig& config,
    const NormalizedGraph& graph,
    const CheckpointBundle& bundle,
    const GatSemanticSummary& summary,
    const SegmentAnalysisReport& segment_report);

ExportDecision export_native_csr_segment_attention_gat(
    const std::filesystem::path& output_dir,
    const AppConfig& config,
    const NormalizedGraph& graph,
    const CheckpointBundle& bundle,
    const GatSemanticSummary& summary);

}  // namespace gat_ezkl
