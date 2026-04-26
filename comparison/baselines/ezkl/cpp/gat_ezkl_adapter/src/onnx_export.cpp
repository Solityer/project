#include "gat_ezkl/onnx_export.hpp"

#include <filesystem>
#include <sstream>

#include "gat_ezkl/metadata.hpp"

namespace gat_ezkl {

ExportDecision export_or_block_strict_gat_onnx(
    const std::filesystem::path& output_dir,
    const AppConfig& config,
    const NormalizedGraph& graph,
    const CheckpointBundle& bundle,
    const GatSemanticSummary& summary) {
    (void)config;
    (void)bundle;
    ExportDecision decision;
    decision.route_id = "strict_gat_full_onnx_v1";
    decision.status = "blocked";
    decision.block_reason = "ezkl_cannot_represent_grouped_softmax";
    decision.strict_public_input_equivalent = false;
    decision.full_gat_forward_in_ezkl_proof = false;
    decision.network_onnx_is_placeholder = true;

    std::ostringstream limitation;
    limitation
        << "Strict full-dataset GAT needs exact grouped softmax over variable-degree incoming edges. "
        << "The safe ONNX lowering would need scatter/gather plus exact per-dst normalization or a dense N x N attention matrix. "
        << "For this full dataset N=" << graph.node_count
        << ", E=" << graph.edge_count
        << ", heads=" << summary.total_attention_head_count
        << ", dense_attention_elements_all_heads=" << summary.dense_attention_elements_all_heads
        << ". A finite dense mask would change non-edge semantics, and the exact dense/scatter lowering is outside the resource envelope for a formal full-dataset ezkl run. "
        << "No identity, digest, folding, or precomputed-logits route was used.";
    decision.limitation = limitation.str();

    std::ostringstream placeholder;
    placeholder << "{\n"
        << "  \"artifact\": \"network.onnx\",\n"
        << "  \"placeholder\": true,\n"
        << "  \"route_id\": \"" << decision.route_id << "\",\n"
        << "  \"status\": \"blocked\",\n"
        << "  \"block_reason\": \"" << decision.block_reason << "\",\n"
        << "  \"limitation\": \"" << json_escape(decision.limitation) << "\"\n"
        << "}\n";
    write_text_file(output_dir / "network.onnx", placeholder.str());
    write_text_file(output_dir / "network.onnx.blocked.json", placeholder.str());
    return decision;
}

ExportDecision export_or_block_strict_segment_gat_onnx(
    const std::filesystem::path& output_dir,
    const AppConfig& config,
    const NormalizedGraph& graph,
    const CheckpointBundle& bundle,
    const GatSemanticSummary& summary,
    const SegmentAnalysisReport& segment_report) {
    (void)bundle;
    ExportDecision decision;
    decision.route_id = "ezkl_gat_strict_segment_softmax_full";
    decision.status = "blocked";
    decision.strict_public_input_equivalent = false;
    decision.full_gat_forward_in_ezkl_proof = false;
    decision.network_onnx_is_placeholder = true;

    if (!segment_report.blocker_if_any.empty()) {
        decision.block_reason = segment_report.blocker_if_any;
        std::ostringstream limitation;
        limitation
            << "Segment-packed grouped softmax analysis did not reach an exportable fixed-shape layout. "
            << "blocker=" << segment_report.blocker_if_any
            << ", N=" << graph.node_count
            << ", E=" << graph.edge_count
            << ", max_in_degree=" << segment_report.max_in_degree
            << ", padded_edge_slots=" << segment_report.total_packed_slots
            << ", max_packed_slots_limit=" << segment_report.max_packed_slots_limit
            << ". No fallback route was used.";
        decision.limitation = limitation.str();
    } else {
        decision.block_reason = "ezkl_inverse_semantics_unavailable";
        std::ostringstream limitation;
        limitation
            << "C++ adapter constructed an exact dst-segment-packed layout for grouped softmax "
            << "(N=" << graph.node_count
            << ", E=" << graph.edge_count
            << ", Dmax=" << segment_report.max_in_degree
            << ", padded_edge_slots=" << segment_report.total_packed_slots
            << ", padding_ratio=" << segment_report.padding_ratio
            << ") and verified the invalid-slot ReduceMax sentinel bound. "
            << "The remaining ONNX lowering requires alpha_k = rescale(U_k * Inv_dst(k)) with the quant_cfg_id-bound scale-16 inverse lookup policy. "
            << "This ezkl revision exposes ONNX Div only for constant denominators, and its Recip path is an approximate gadget governed by settings epsilon rather than the deterministic T_inv table emitted for the native route. "
            << "ReduceMax/ReduceSum/Gather are not the decisive blocker; the blocker is preserving the quantized inverse lookup and alpha rescale semantics in ordinary ONNX. "
            << "No precomputed logits, dense NxN attention, output binding, identity, digest, folding, subset, shard, or sample route was used. "
            << "heads=" << summary.total_attention_head_count
            << ", estimated_total_attention_elements=" << segment_report.estimate.estimated_total_attention_elements
            << ".";
        decision.limitation = limitation.str();
    }

    std::ostringstream placeholder;
    placeholder << "{\n"
        << "  \"artifact\": \"network.onnx\",\n"
        << "  \"placeholder\": true,\n"
        << "  \"route_id\": \"" << decision.route_id << "\",\n"
        << "  \"status\": \"blocked\",\n"
        << "  \"block_reason\": \"" << decision.block_reason << "\",\n"
        << "  \"segment_lowering\": true,\n"
        << "  \"segment_layout\": \"dst_segment_packed\",\n"
        << "  \"max_in_degree\": " << segment_report.max_in_degree << ",\n"
        << "  \"padded_edge_slots\": " << segment_report.total_packed_slots << ",\n"
        << "  \"original_edge_count\": " << segment_report.edge_count << ",\n"
        << "  \"sentinel_bound_verified\": "
        << (segment_report.sentinel_bound_verified ? "true" : "false") << ",\n"
        << "  \"limitation\": \"" << json_escape(decision.limitation) << "\"\n"
        << "}\n";
    write_text_file(output_dir / "network.onnx", placeholder.str());
    write_text_file(output_dir / "network.onnx.blocked.json", placeholder.str());
    return decision;
}

ExportDecision export_native_csr_segment_attention_gat(
    const std::filesystem::path& output_dir,
    const std::string& route_id,
    const AppConfig& config,
    const NormalizedGraph& graph,
    const CheckpointBundle& bundle,
    const GatSemanticSummary& summary) {
    (void)config;
    (void)bundle;
    const bool compressed_route = route_id == "ezkl_gat_native_compressed_csr_attention_full";
    ExportDecision decision;
    decision.route_id = route_id;
    decision.status = "ready";
    decision.block_reason = "";
    decision.strict_public_input_equivalent = false;
    decision.full_gat_forward_in_ezkl_proof = false;
    decision.network_onnx_is_placeholder = false;

    std::ostringstream limitation;
    limitation
        << "C++ adapter exported full-dataset CSR/ragged graph attention artifacts for ezkl native "
        << (compressed_route ? "GraphSegmentAttentionCompressedCSR" : "GraphSegmentAttention")
        << ". The route avoids dense NxN attention and padded N*Dmax as the "
        << "primary representation. The adapter emits final_logits_binding.json and output-layer metadata, "
        << "while strict=true is left to the Rust native backend only after complete output-head derivation is constrained. "
        << "Proof success still depends on native ezkl compile/setup/prove/verify. "
        << "N=" << graph.node_count
        << ", E=" << graph.edge_count
        << ", G=" << graph.graph_count
        << ", heads=" << summary.total_attention_head_count
        << ".";
    decision.limitation = limitation.str();

    std::ostringstream marker;
    marker << "{\n"
        << "  \"artifact\": \"network.onnx\",\n"
        << "  \"placeholder\": false,\n"
        << "  \"model_type\": \""
        << (compressed_route ? "graph-segment-attention-compressed-native-csr" : "graph-segment-attention-native-csr")
        << "\",\n"
        << "  \"route_id\": \"" << decision.route_id << "\",\n"
        << "  \"status\": \"ready\",\n"
        << "  \"note\": \"native graph route uses graph_attention_spec.json, csr_segment_layout.bin, and quant_cfg-derived inverse/alpha policy metadata, not ordinary ONNX ops\"\n"
        << "}\n";
    write_text_file(output_dir / "network.onnx", marker.str());
    write_text_file(output_dir / "network.native.json", marker.str());
    return decision;
}

}  // namespace gat_ezkl
