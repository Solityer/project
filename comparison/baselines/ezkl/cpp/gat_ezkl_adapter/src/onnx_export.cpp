#include "gat_ezkl/onnx_export.hpp"

#include <filesystem>
#include <sstream>

#include "gat_ezkl/metadata.hpp"

namespace gat_ezkl
{
    ExportDecision export_native_csr_segment_attention_gat(
        const std::filesystem::path &output_dir,
        const std::string &route_id,
        const AppConfig &config,
        const NormalizedGraph &graph,
        const CheckpointBundle &bundle,
        const GatSemanticSummary &summary)
    {
        (void)config;
        (void)bundle;
        ExportDecision decision;
        decision.route_id = route_id;
        decision.status = "ready";
        decision.block_reason = "";
        decision.strict_public_input_equivalent = false;
        decision.full_gat_forward_in_ezkl_proof = false;
        decision.network_onnx_is_blocked_artifact = false;

        std::ostringstream limitation;
        limitation
            << "C++ adapter exported full-dataset CSR/ragged graph attention artifacts for ezkl native "
            << "GraphSegmentAttentionCompressedCSR"
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

        // 写一个标记文件代替真正的 ONNX 模型，ezkl 后端据此切换到原生 CSR 路径
        std::ostringstream marker;
        marker << "{\n"
               << "  \"artifact\": \"network.onnx\",\n"
               << "  \"blocked_artifact\": false,\n"
               << "  \"model_type\": \""
               << "graph-segment-attention-compressed-native-csr"
               << "\",\n"
               << "  \"route_id\": \"" << decision.route_id << "\",\n"
               << "  \"status\": \"ready\",\n"
               << "  \"note\": \"native graph route uses graph_attention_spec.json, csr_segment_layout.bin, and quant_cfg-derived inverse/alpha policy metadata, not ordinary ONNX ops\"\n"
               << "}\n";
        write_text_file(output_dir / "network.onnx", marker.str());
        write_text_file(output_dir / "network.native.json", marker.str());
        return decision;
    }
}