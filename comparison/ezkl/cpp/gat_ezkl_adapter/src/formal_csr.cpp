#include "gat_ezkl/formal_csr.hpp"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "gat_ezkl/formal_io.hpp"

namespace gat_ezkl {
namespace {

constexpr const char* kFormalRoute = "ezkl_gat_native_compressed_csr_attention_full";

void write_u64(std::ofstream& output, std::uint64_t value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_vector(std::ofstream& output, const std::vector<std::size_t>& values) {
    write_u64(output, values.size());
    for (const auto value : values) write_u64(output, value);
}

std::string size_vector_json(const std::vector<std::size_t>& values) {
    std::ostringstream output;
    output << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ",";
        output << values[index];
    }
    output << "]";
    return output.str();
}

std::string hidden_profile_json(const std::vector<HiddenLayerProfile>& profile) {
    std::ostringstream output;
    output << "[";
    for (std::size_t index = 0; index < profile.size(); ++index) {
        if (index) output << ",";
        output << "{\"head_count\":" << profile[index].head_count
               << ",\"head_dim\":" << profile[index].head_dim << "}";
    }
    output << "]";
    return output.str();
}

std::string hidden_heads_json(const CheckpointBundle& bundle) {
    std::ostringstream output;
    output << "[";
    for (std::size_t index = 0; index < bundle.hidden_heads.size(); ++index) {
        const auto& head = bundle.hidden_heads[index];
        if (index) output << ",";
        output << "{\"layer_index\":" << head.layer_index
               << ",\"local_head_index\":" << head.local_head_index
               << ",\"head_index\":" << head.global_head_index
               << ",\"seq_kernel\":\"" << json_escape(head.seq_kernel)
               << "\",\"attn_src_kernel\":\"" << json_escape(head.attn_src_kernel)
               << "\",\"attn_dst_kernel\":\"" << json_escape(head.attn_dst_kernel)
               << "\"}";
    }
    output << "]";
    return output.str();
}

std::string output_heads_json(const CheckpointBundle& bundle) {
    std::ostringstream output;
    output << "[";
    for (std::size_t index = 0; index < bundle.output_heads.size(); ++index) {
        const auto& head = bundle.output_heads[index];
        if (index) output << ",";
        output << "{\"head_index\":" << head.head_index
               << ",\"seq_kernel\":\"" << json_escape(head.seq_kernel)
               << "\",\"attn_src_kernel\":\"" << json_escape(head.attn_src_kernel)
               << "\",\"attn_dst_kernel\":\"" << json_escape(head.attn_dst_kernel)
               << "\",\"output_bias\":\"" << json_escape(head.output_bias) << "\"}";
    }
    output << "]";
    return output.str();
}

}  // namespace

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
    const std::string& security_profile_id) {
    if (route_id != kFormalRoute) {
        throw std::runtime_error("formal CSR exporter received an unregistered GAT route");
    }
    if (graph.absolute_ids.size() != graph.node_count) {
        throw std::runtime_error("formal CSR is missing canonical absolute node identities");
    }
    for (std::size_t index = 0; index < graph.absolute_ids.size(); ++index) {
        if (graph.absolute_ids[index] != index) {
            throw std::runtime_error("full-dataset formal route requires I=[0,N)");
        }
    }
    std::vector<std::size_t> segment_ptr(graph.node_count + 1, 0);
    std::vector<std::size_t> segment_graph_id(graph.node_count, 0);
    std::size_t cursor = 0;
    std::size_t maximum_degree = 0;
    for (std::size_t node = 0; node < graph.node_count; ++node) {
        segment_ptr[node] = cursor;
        while (cursor < graph.edges.size() && graph.edges[cursor].dst == node) ++cursor;
        const auto degree = cursor - segment_ptr[node];
        if (degree == 0) throw std::runtime_error("formal CSR requires a self-loop/incoming edge");
        maximum_degree = std::max(maximum_degree, degree);
        while (segment_graph_id[node] + 1 < graph.node_ptr.size()
               && node >= graph.node_ptr[segment_graph_id[node] + 1]) {
            ++segment_graph_id[node];
        }
    }
    segment_ptr[graph.node_count] = cursor;
    if (cursor != graph.edges.size()) throw std::runtime_error("formal CSR edge order mismatch");

    std::filesystem::create_directories(output_dir);
    std::ofstream binary(output_dir / "csr_segment_layout.bin", std::ios::binary | std::ios::trunc);
    if (!binary) throw std::runtime_error("failed to write formal CSR payload");
    binary.write("EZKLGATCSR2\n", 12);
    write_vector(binary, graph.src);
    write_vector(binary, graph.dst);
    write_vector(binary, segment_ptr);
    write_vector(binary, graph.edge_ptr);
    write_vector(binary, graph.node_ptr);
    write_vector(binary, segment_graph_id);
    write_vector(binary, graph.absolute_ids);
    write_u64(binary, maximum_degree);
    if (!binary) throw std::runtime_error("failed to complete formal CSR payload");

    std::ostringstream layout;
    layout << "{\"layout_kind\":\"csr_dst_segment_ragged_v2\",\"payload\":\"csr_segment_layout.bin\""
           << ",\"node_count\":" << graph.node_count << ",\"edge_count\":" << graph.edge_count
           << ",\"graph_count\":" << graph.graph_count << ",\"max_in_degree\":" << maximum_degree
           << ",\"absolute_ids_in_payload\":true}\n";
    write_text_file(output_dir / "csr_segment_layout.json", layout.str());

    std::ostringstream spec;
    spec << "{\n"
         << "  \"route_id\": \"" << json_escape(route_id) << "\",\n"
         << "  \"dataset\": \"" << json_escape(config.dataset) << "\",\n"
         << "  \"input_preprocess_id\": \"" << json_escape(config.input_preprocess_id) << "\",\n"
         << "  \"batching_rule\": \"" << json_escape(config.batching_rule) << "\",\n"
         << "  \"node_count\": " << graph.node_count << ",\n"
         << "  \"edge_count\": " << graph.edge_count << ",\n"
         << "  \"graph_count\": " << graph.graph_count << ",\n"
         << "  \"feature_dim\": " << graph.feature_dim << ",\n"
         << "  \"class_count\": " << graph.class_count << ",\n"
         << "  \"L\": " << bundle.layer_count << ",\n"
         << "  \"K_out\": " << bundle.K_out << ",\n"
         << "  \"hidden_profile\": " << hidden_profile_json(bundle.hidden_profile) << ",\n"
         << "  \"d_in_profile\": " << size_vector_json(bundle.d_in_profile) << ",\n"
         << "  \"hidden_heads\": " << hidden_heads_json(bundle) << ",\n"
         << "  \"output_heads\": " << output_heads_json(bundle) << ",\n"
         << "  \"model_arch_id\": \"" << json_escape(config.model_arch_id) << "\",\n"
         << "  \"model_param_id\": \"" << json_escape(config.model_param_id) << "\",\n"
         << "  \"quant_cfg_id\": \"" << formal_quant_cfg_id << "\",\n"
         << "  \"static_tables_id\": \"" << formal_static_tables_id << "\",\n"
         << "  \"backend_aux_tables_id\": \"" << backend_aux_tables_id << "\",\n"
         << "  \"source_quant_cfg_id\": \"" << json_escape(config.quant_cfg_id) << "\",\n"
         << "  \"source_static_table_id\": \"" << json_escape(config.static_table_id) << "\",\n"
         << "  \"security_profile_id\": \"" << security_profile_id << "\",\n"
         << "  \"graph_hash\": \"" << graph_hash << "\",\n"
         << "  \"checkpoint_hash\": \"" << checkpoint_hash << "\",\n"
         << "  \"metadata_hash\": \"" << metadata_hash << "\",\n"
         << "  \"shape_hash\": \"" << shape_hash << "\"\n"
         << "}\n";
    write_text_file(output_dir / "graph_attention_spec.json", spec.str());
}

}  // namespace gat_ezkl
