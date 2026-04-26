#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gat_ezkl/bounds.hpp"
#include "gat_ezkl/checkpoint.hpp"
#include "gat_ezkl/config.hpp"
#include "gat_ezkl/dataset.hpp"
#include "gat_ezkl/gat.hpp"

namespace gat_ezkl {

struct SegmentPackedLayout {
    std::vector<std::size_t> segment_ptr;
    std::vector<std::size_t> edge_slot_index;
    std::vector<std::size_t> packed_src;
    std::vector<std::size_t> packed_edge_index;
    std::vector<std::uint8_t> valid_mask;
    std::size_t max_in_degree = 0;
    std::size_t total_packed_slots = 0;
    std::size_t padding_slots = 0;
    double padding_ratio = 0.0;
};

struct SegmentLayoutHashes {
    std::string valid_mask_hash;
    std::string segment_ptr_hash;
    std::string packed_src_hash;
    std::string packed_edge_index_hash;
};

struct CsrLayoutHashes {
    std::string src_hash;
    std::string dst_hash;
    std::string segment_ptr_hash;
    std::string edge_ptr_hash;
    std::string node_ptr_hash;
    std::string segment_graph_id_hash;
};

struct CsrSegmentLayout {
    std::vector<std::size_t> src;
    std::vector<std::size_t> dst;
    std::vector<std::size_t> segment_ptr;
    std::vector<std::size_t> edge_ptr;
    std::vector<std::size_t> node_ptr;
    std::vector<std::size_t> segment_graph_id;
    std::size_t max_in_degree = 0;
    CsrLayoutHashes hashes;
};

struct SegmentCostEstimate {
    std::size_t estimated_attention_tensor_elements_per_head = 0;
    std::size_t hidden_layer_count = 0;
    std::size_t hidden_head_count_total = 0;
    std::size_t output_head_count = 0;
    std::size_t estimated_total_attention_elements = 0;
    std::size_t estimated_onnx_nodes = 0;
    std::size_t estimated_ezkl_rows = 0;
};

struct SegmentAnalysisReport {
    std::string dataset;
    std::size_t node_count = 0;
    std::size_t edge_count = 0;
    std::size_t graph_count = 0;
    std::size_t max_in_degree = 0;
    std::size_t total_packed_slots = 0;
    std::size_t padding_slots = 0;
    double padding_ratio = 0.0;
    SegmentCostEstimate estimate;
    bool route_feasible_small = false;
    bool route_feasible_full = false;
    std::string blocker_if_any;
    bool segment_lowering = true;
    std::string segment_layout = "dst_segment_packed";
    std::string edge_order_rule;
    std::string self_loop_rule;
    std::int64_t sentinel_value = 0;
    bool sentinel_bound_verified = false;
    std::string sentinel_bound_explanation;
    SegmentLayoutHashes hashes;
    std::size_t max_packed_slots_limit = 0;
};

bool verify_edge_gid_then_dst_stable_for_segments(
    const NormalizedGraph& graph,
    std::string* reason);
SegmentPackedLayout build_dst_segments(
    const NormalizedGraph& graph,
    std::string* reason);
SegmentPackedLayout build_segment_packed_layout(
    const NormalizedGraph& graph,
    std::string* reason);
CsrSegmentLayout build_csr_segment_layout(
    const NormalizedGraph& graph,
    std::string* reason);
SegmentLayoutHashes hash_segment_layout(const SegmentPackedLayout& layout);
CsrLayoutHashes hash_csr_segment_layout(const CsrSegmentLayout& layout);
SegmentCostEstimate estimate_segment_lowering_cost(
    const SegmentPackedLayout& layout,
    const CheckpointBundle& bundle,
    const GatSemanticSummary& summary);
SegmentAnalysisReport analyze_segment_softmax(
    const std::string& dataset,
    const AppConfig& config,
    const NormalizedGraph& graph,
    const CheckpointBundle& bundle,
    const GatSemanticSummary& summary,
    const ScoreBounds& bounds,
    std::size_t max_packed_slots_limit);
void write_segment_analysis_json(
    const std::filesystem::path& path,
    const SegmentAnalysisReport& report);
void write_segment_debug_files(
    const std::filesystem::path& output_dir,
    const SegmentPackedLayout& layout);
void write_native_csr_graph_artifacts(
    const std::filesystem::path& output_dir,
    const std::string& route_id,
    const AppConfig& config,
    const NormalizedGraph& graph,
    const CheckpointBundle& bundle,
    const GatSemanticSummary& summary,
    const std::string& graph_hash,
    const std::string& checkpoint_hash,
    const std::string& metadata_hash,
    const std::string& shape_hash);

std::string segment_analysis_json(const SegmentAnalysisReport& report);
std::string segment_metadata_json(const SegmentAnalysisReport& report);

}  // namespace gat_ezkl
