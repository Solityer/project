#include "gat_ezkl/metadata.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace gat_ezkl {
namespace {

std::string bool_json(bool value) {
    return value ? "true" : "false";
}

std::string hidden_profile_json(const std::vector<HiddenLayerProfile>& hidden_profile) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < hidden_profile.size(); ++i) {
        if (i != 0) out << ",";
        out << "{\"head_count\":" << hidden_profile[i].head_count
            << ",\"head_dim\":" << hidden_profile[i].head_dim << "}";
    }
    out << "]";
    return out.str();
}

std::string size_vector_json(const std::vector<std::size_t>& values) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ",";
        out << values[i];
    }
    out << "]";
    return out.str();
}

}  // namespace

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const auto ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

std::string now_utc_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&tt, &tm);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

void write_text_file(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to write file: " + path.string());
    }
    output << text;
}

std::string build_metadata_json(
    const std::string& run_id,
    const AppConfig& config,
    const NormalizedGraph& graph,
    const CheckpointBundle& bundle,
    const QuantConfig& quant,
    const GatSemanticSummary& gat_summary,
    const std::string& graph_hash,
    const std::string& checkpoint_hash,
    const std::string& shape_hash,
    const FormalTablesSummary* formal_tables) {
    std::ostringstream out;
    out << "{\n"
        << "  \"protocol_id\": \"gat_zkml_ezkl_baseline_v1\",\n"
        << "  \"run_id\": \"" << json_escape(run_id) << "\",\n"
        << "  \"dataset_id\": \"" << json_escape(config.dataset) << "\",\n"
        << "  \"task_type\": \"" << json_escape(config.task_type) << "\",\n"
        << "  \"report_unit\": \"" << json_escape(config.report_unit) << "\",\n"
        << "  \"batching_rule\": \"" << json_escape(config.batching_rule) << "\",\n"
        << "  \"subgraph_rule\": \"whole_graph\",\n"
        << "  \"self_loop_rule\": \"" << json_escape(config.self_loop_rule) << "\",\n"
        << "  \"edge_sort_rule\": \"" << json_escape(config.edge_sort_rule) << "\",\n"
        << "  \"chunking_rule\": \"" << json_escape(config.chunking_rule) << "\",\n"
        << "  \"model_arch_id\": \"" << json_escape(config.model_arch_id) << "\",\n"
        << "  \"model_param_id\": \"" << json_escape(config.model_param_id) << "\",\n"
        << "  \"quant_cfg_id\": \"" << json_escape(quant.quant_cfg_id) << "\",\n"
        << "  \"checkpoint_quant_cfg_id\": \"" << json_escape(bundle.quant_cfg_id) << "\",\n"
        << "  \"output_average_rule\": \"" << json_escape(bundle.output_average_rule) << "\",\n"
        << "  \"visibility_cfg\": \"public_input_public_logits_constant_weight_hash\",\n"
        << "  \"input_visibility\": \"public_full_dataset_tensor_summary\",\n"
        << "  \"output_visibility\": \"public_logits_when_verified\",\n"
        << "  \"param_visibility\": \"weight_constants_bound_by_checkpoint_hash\",\n"
        << "  \"graph_hash\": \"" << graph_hash << "\",\n"
        << "  \"checkpoint_hash\": \"" << checkpoint_hash << "\",\n"
        << "  \"shape_hash\": \"" << shape_hash << "\",\n"
        << "  \"hash_algorithm\": \"fnv1a64_adapter\",\n"
        << "  \"dim_cfg\": {\"N\":" << graph.node_count
        << ",\"E\":" << graph.edge_count
        << ",\"G_batch\":" << graph.graph_count
        << ",\"d_in\":" << graph.feature_dim
        << ",\"num_classes\":" << graph.class_count
        << ",\"L\":" << bundle.layer_count
        << ",\"K_out\":" << bundle.K_out
        << ",\"hidden_profile\":" << hidden_profile_json(bundle.hidden_profile) << "},\n"
        << "  \"quantization\": {\"input_scale\":" << quant.input_scale
        << ",\"weight_scale\":" << quant.weight_scale
        << ",\"rounding\":\"" << json_escape(quant.rounding) << "\"},\n"
        << "  \"static_table_id\": \"" << json_escape(config.static_table_id) << "\",\n";
    if (formal_tables != nullptr) {
        out << "  \"formal_tables\": " << formal_table_binding_json(*formal_tables) << ",\n";
    }
    out
        << "  \"gat_semantics\": {\"implemented_in_cpp\":" << bool_json(gat_summary.implemented_in_cpp)
        << ",\"hidden_activation\":\"" << gat_summary.hidden_activation
        << "\",\"attention_activation\":\"" << gat_summary.attention_activation
        << "\",\"output_activation\":\"" << gat_summary.output_activation
        << "\",\"hidden_attention_head_count\":" << gat_summary.hidden_attention_head_count
        << ",\"output_attention_head_count\":" << gat_summary.output_attention_head_count
        << ",\"total_attention_head_count\":" << gat_summary.total_attention_head_count
        << ",\"dense_attention_elements_per_head\":" << gat_summary.dense_attention_elements_per_head
        << ",\"dense_attention_elements_all_heads\":" << gat_summary.dense_attention_elements_all_heads << "}\n"
        << "}\n";
    return out.str();
}

std::string build_model_info_json(
    const AppConfig& config,
    const CheckpointBundle& bundle,
    const GatSemanticSummary& gat_summary,
    const std::string& checkpoint_hash,
    const std::string& shape_hash,
    const FormalTablesSummary* formal_tables) {
    std::ostringstream out;
    out << "{\n"
        << "  \"model_arch_id\": \"" << json_escape(config.model_arch_id) << "\",\n"
        << "  \"model_param_id\": \"" << json_escape(config.model_param_id) << "\",\n"
        << "  \"checkpoint_root\": \"" << json_escape(bundle.root.string()) << "\",\n"
        << "  \"checkpoint_hash\": \"" << checkpoint_hash << "\",\n"
        << "  \"shape_hash\": \"" << shape_hash << "\",\n"
        << "  \"L\": " << bundle.layer_count << ",\n"
        << "  \"hidden_profile\": " << hidden_profile_json(bundle.hidden_profile) << ",\n"
        << "  \"d_in_profile\": " << size_vector_json(bundle.d_in_profile) << ",\n"
        << "  \"K_out\": " << bundle.K_out << ",\n"
        << "  \"num_classes\": " << bundle.class_count << ",\n"
        << "  \"hidden_head_count\": " << bundle.hidden_heads.size() << ",\n"
        << "  \"output_head_count\": " << bundle.output_heads.size() << ",\n"
        << "  \"tensor_count_loaded_from_tensors_txt\": " << bundle.tensors.size() << ",\n";
    if (formal_tables != nullptr) {
        out << "  \"formal_tables\": " << formal_table_binding_json(*formal_tables) << ",\n";
    }
    out
        << "  \"gat_forward_semantics_cpp_available\": " << bool_json(gat_summary.implemented_in_cpp) << "\n"
        << "}\n";
    return out.str();
}

std::string build_input_summary_json(
    const NormalizedGraph& graph,
    const QuantConfig& quant,
    const std::string& graph_hash) {
    std::ostringstream out;
    out << "{\n"
        << "  \"input_json_kind\": \"full_dataset_public_input_summary\",\n"
        << "  \"contains_full_tensor_values\": false,\n"
        << "  \"reason_full_tensor_values_omitted\": \"strict route blocked before ezkl witness generation; hashes bind the C++ normalized full input\",\n"
        << "  \"N_total\": " << graph.n_total << ",\n"
        << "  \"N\": " << graph.node_count << ",\n"
        << "  \"E\": " << graph.edge_count << ",\n"
        << "  \"G_batch\": " << graph.graph_count << ",\n"
        << "  \"feature_dim\": " << graph.feature_dim << ",\n"
        << "  \"num_classes\": " << graph.class_count << ",\n"
        << "  \"node_ptr\": " << size_vector_json(graph.node_ptr) << ",\n"
        << "  \"edge_ptr\": " << size_vector_json(graph.edge_ptr) << ",\n"
        << "  \"feature_quant_scale\": " << quant.input_scale << ",\n"
        << "  \"graph_hash\": \"" << graph_hash << "\"\n"
        << "}\n";
    return out.str();
}

}  // namespace gat_ezkl
