#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gat_ezkl/bounds.hpp"
#include "gat_ezkl/checkpoint.hpp"
#include "gat_ezkl/config.hpp"
#include "gat_ezkl/dataset.hpp"
#include "gat_ezkl/formal_tables.hpp"
#include "gat_ezkl/gat.hpp"
#include "gat_ezkl/graph.hpp"
#include "gat_ezkl/hash.hpp"
#include "gat_ezkl/metadata.hpp"
#include "gat_ezkl/onnx_export.hpp"
#include "gat_ezkl/quant.hpp"
#include "gat_ezkl/segment_softmax.hpp"

namespace {

std::map<std::string, std::string> parse_args(int argc, char** argv) {
    std::map<std::string, std::string> out;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            throw std::runtime_error("expected --key value argument, got: " + key);
        }
        const auto normalized = key.substr(2);
        if (i + 1 >= argc) {
            throw std::runtime_error("expected value for argument: " + key);
        }
        const std::string value = argv[++i];
        if (value.rfind("--", 0) == 0) {
            throw std::runtime_error("expected value for argument: " + key);
        }
        out[normalized] = value;
    }
    return out;
}

std::filesystem::path require_path_arg(
    const std::map<std::string, std::string>& args,
    const std::string& key) {
    const auto it = args.find(key);
    if (it == args.end() || it->second.empty()) {
        throw std::runtime_error("missing required argument --" + key);
    }
    return std::filesystem::path(it->second);
}

std::string require_string_arg(
    const std::map<std::string, std::string>& args,
    const std::string& key) {
    const auto it = args.find(key);
    if (it == args.end() || it->second.empty()) {
        throw std::runtime_error("missing required argument --" + key);
    }
    return it->second;
}

std::string optional_string_arg(
    const std::map<std::string, std::string>& args,
    const std::string& key,
    const std::string& fallback) {
    const auto it = args.find(key);
    return it == args.end() ? fallback : it->second;
}

bool is_native_graph_route(const std::string& route) {
    return route == "ezkl_gat_native_compressed_csr_attention_full";
}

std::string native_primitive_for_route(const std::string& route) {
    (void)route;
    return "GraphSegmentAttentionCompressedCSR";
}

std::string native_primitive_version_for_route(const std::string& route) {
    (void)route;
    return "csr_segment_attention_compressed_v1";
}

std::size_t public_degree_bound_from_graph(const gat_ezkl::NormalizedGraph& graph) {
    std::vector<std::size_t> degrees(graph.node_count, 0);
    std::size_t max_degree = 0;
    for (const auto& edge : graph.edges) {
        if (edge.dst >= degrees.size()) {
            throw std::runtime_error("edge dst out of range while deriving inverse table degree bound");
        }
        auto& degree = degrees[edge.dst];
        ++degree;
        max_degree = std::max(max_degree, degree);
    }
    if (max_degree == 0) {
        throw std::runtime_error("cannot derive inverse table degree bound from an edgeless graph");
    }
    return max_degree;
}

bool path_is_within(const std::filesystem::path& child, const std::filesystem::path& parent) {
    const auto abs_child = std::filesystem::weakly_canonical(child);
    const auto abs_parent = std::filesystem::weakly_canonical(parent);
    auto child_it = abs_child.begin();
    for (auto parent_it = abs_parent.begin(); parent_it != abs_parent.end(); ++parent_it, ++child_it) {
        if (child_it == abs_child.end() || *child_it != *parent_it) {
            return false;
        }
    }
    return true;
}

std::string null_metric_json() {
    return "null";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        const auto run_id = require_string_arg(args, "dataset");
        const auto project_root = std::filesystem::weakly_canonical(require_path_arg(args, "project-root"));
        const auto ezkl_root = std::filesystem::weakly_canonical(require_path_arg(args, "ezkl-root"));
        const auto output_dir = require_path_arg(args, "output-dir");
        const auto route = optional_string_arg(args, "route", "ezkl_gat_native_compressed_csr_attention_full");
        const auto analysis_only = optional_string_arg(args, "analysis-only", "");

        if (!path_is_within(output_dir, ezkl_root)) {
            throw std::runtime_error("output-dir must be inside ezkl-root");
        }
        if (project_root != std::filesystem::weakly_canonical("/home/pzh/project")) {
            throw std::runtime_error("project-root must be /home/pzh/project for this baseline");
        }

        std::filesystem::create_directories(output_dir);
        const auto spec = gat_ezkl::official_dataset_spec(run_id);
        auto config = gat_ezkl::load_config_file(gat_ezkl::config_path_for_dataset(project_root, run_id));
        gat_ezkl::validate_full_dataset_contract(config, spec);

        const auto checkpoint_root = project_root / config.checkpoint_bundle;
        auto checkpoint = gat_ezkl::load_checkpoint_bundle(checkpoint_root);
        gat_ezkl::validate_checkpoint_against_config(checkpoint, &config);

        const auto dataset = gat_ezkl::load_dataset_from_cache(project_root, config);
        if (dataset.name != spec.display_name && dataset.name != spec.main_dataset_name && !(run_id == "ogbn_arxiv_full" && dataset.name == "ogbn-arxiv")) {
            throw std::runtime_error("dataset cache name does not match official dataset spec");
        }
        if (dataset.num_features != (config.d_in_profile.empty() ? dataset.num_features : config.d_in_profile.front())) {
            throw std::runtime_error("dataset feature dimension does not match config d_in_profile");
        }
        if (dataset.num_classes != config.num_classes) {
            throw std::runtime_error("dataset class count does not match config");
        }
        const auto graph = gat_ezkl::normalize_full_graph(dataset, config);
        if (graph.node_count != dataset.num_nodes) {
            throw std::runtime_error("normalized graph is not full dataset");
        }
        std::string graph_reason;
        if (!gat_ezkl::graph_blocks_are_valid(graph, &graph_reason)) {
            throw std::runtime_error("normalized graph block validation failed: " + graph_reason);
        }
        if (!gat_ezkl::edge_order_is_gid_then_dst_stable(graph, &graph_reason)) {
            throw std::runtime_error("normalized graph sort validation failed: " + graph_reason);
        }

        const auto quant = gat_ezkl::quant_config_from_id(config.quant_cfg_id, config.range_bits);
        const auto gat_summary = gat_ezkl::summarize_gat_semantics(graph, checkpoint);
        if (!analysis_only.empty()) {
            throw std::runtime_error("analysis-only routes are not part of the final strict ezkl comparison");
        }
        if (!is_native_graph_route(route)) {
            throw std::runtime_error("only the strict CPU compressed CSR route is allowed for the final comparison");
        }
        const bool wants_native_csr_route = true;
        const auto graph_hash = gat_ezkl::hash_graph_hex(graph);
        const auto checkpoint_hash = gat_ezkl::hash_files_hex({checkpoint_root / "manifest.json", checkpoint_root / "tensors.txt"});
        const auto shape_hash = gat_ezkl::hash_shape_hex(graph, checkpoint);
        const auto output_hash = gat_ezkl::hash_output_shape_hex(graph);

        std::optional<gat_ezkl::FormalTablesSummary> formal_tables;
        if (wants_native_csr_route) {
            formal_tables = gat_ezkl::write_formal_table_artifacts(
                output_dir,
                config,
                public_degree_bound_from_graph(graph));
        }

        const auto metadata_json = gat_ezkl::build_metadata_json(
            run_id,
            config,
            graph,
            checkpoint,
            quant,
            gat_summary,
            graph_hash,
            checkpoint_hash,
            shape_hash,
            formal_tables ? &*formal_tables : nullptr);
        const auto metadata_hash = gat_ezkl::hash_metadata_hex(metadata_json);
        const auto model_info_json = gat_ezkl::build_model_info_json(
            config,
            graph,
            checkpoint,
            gat_summary,
            checkpoint_hash,
            shape_hash,
            formal_tables ? &*formal_tables : nullptr);
        const auto input_json = gat_ezkl::build_input_summary_json(graph, quant, graph_hash);
        if (wants_native_csr_route) {
            gat_ezkl::write_native_csr_graph_artifacts(
                output_dir,
                route,
                config,
                graph,
                checkpoint,
                gat_summary,
                graph_hash,
                checkpoint_hash,
                metadata_hash,
                shape_hash,
                formal_tables ? &*formal_tables : nullptr);
        }
        const auto decision = gat_ezkl::export_native_csr_segment_attention_gat(
            output_dir,
            route,
            config,
            graph,
            checkpoint,
            gat_summary);

        gat_ezkl::write_text_file(output_dir / "metadata.json", metadata_json);
        gat_ezkl::write_text_file(output_dir / "model_info.json", model_info_json);
        gat_ezkl::write_text_file(output_dir / "input.json", input_json);
        gat_ezkl::write_text_file(output_dir / "shape_hash.txt", shape_hash + "\n");
        gat_ezkl::write_text_file(output_dir / "graph_hash.txt", graph_hash + "\n");
        gat_ezkl::write_text_file(output_dir / "checkpoint_hash.txt", checkpoint_hash + "\n");

        std::ostringstream settings;
        settings << "{\n"
            << "  \"settings_kind\": \"blocked_strict_route_preflight\",\n"
            << "  \"route_id\": \"" << decision.route_id << "\",\n"
            << "  \"status\": \"" << decision.status << "\",\n"
            << "  \"check_mode\": \"not_started\",\n"
            << "  \"visibility\": {\n"
            << "    \"input_visibility\": \"public_full_dataset_tensor_summary\",\n"
            << "    \"output_visibility\": \"public_logits_when_verified\",\n"
            << "    \"param_visibility\": \"weight_constants_bound_by_checkpoint_hash\"\n"
            << "  },\n"
            << "  \"quant_cfg_id\": \"" << gat_ezkl::json_escape(quant.quant_cfg_id) << "\",\n"
            << "  \"static_table_id\": \"" << gat_ezkl::json_escape(config.static_table_id) << "\"";
        if (formal_tables) {
            settings << ",\n  \"formal_tables\": " << gat_ezkl::formal_table_binding_json(*formal_tables) << ",\n"
                     << "  \"inverse_policy\": \"" << gat_ezkl::json_escape(formal_tables->inverse_policy) << "\",\n"
                     << "  \"inverse_policy_hash\": \"" << formal_tables->hashes.inverse_policy_hash << "\"\n";
        } else {
            settings << "\n";
        }
        settings << "}\n";
        gat_ezkl::write_text_file(output_dir / "settings.json", settings.str());

        std::ostringstream export_status;
        export_status << "{\n"
            << "  \"route_id\": \"" << decision.route_id << "\",\n"
            << "  \"status\": \"" << decision.status << "\",\n"
            << "  \"block_reason\": \"" << decision.block_reason << "\",\n"
            << "  \"limitation\": \"" << gat_ezkl::json_escape(decision.limitation) << "\",\n"
            << "  \"strict_public_input_equivalent\": " << (decision.strict_public_input_equivalent ? "true" : "false") << ",\n"
            << "  \"full_gat_forward_in_ezkl_proof\": " << (decision.full_gat_forward_in_ezkl_proof ? "true" : "false") << ",\n"
            << "  \"network_onnx_is_blocked_artifact\": " << (decision.network_onnx_is_blocked_artifact ? "true" : "false") << "\n"
            << "}\n";
        gat_ezkl::write_text_file(output_dir / "export_status.json", export_status.str());

        constexpr std::size_t kPublicInstanceChunkSize = 1u << 22;
        const auto public_instance_length = graph.node_count * graph.class_count;
        const auto public_instance_chunk_size = std::min<std::size_t>(
            kPublicInstanceChunkSize,
            std::max<std::size_t>(public_instance_length, 1));
        const auto public_instance_column_count =
            (public_instance_length + public_instance_chunk_size - 1) / public_instance_chunk_size;

        std::ostringstream manifest;
        manifest << "{\n"
            << "  \"generated_at\": \"" << gat_ezkl::now_utc_iso8601() << "\",\n"
            << "  \"dataset\": \"" << run_id << "\",\n"
            << "  \"main_dataset\": \"" << spec.display_name << "\",\n"
            << "  \"route_id\": \"" << decision.route_id << "\",\n"
            << "  \"status\": \"" << decision.status << "\",\n"
            << "  \"block_reason\": \"" << decision.block_reason << "\",\n"
            << "  \"limitation\": \"" << gat_ezkl::json_escape(decision.limitation) << "\",\n"
            << "  \"strict_public_input_equivalent\": " << (decision.strict_public_input_equivalent ? "true" : "false") << ",\n"
            << "  \"full_gat_forward_in_ezkl_proof\": " << (decision.full_gat_forward_in_ezkl_proof ? "true" : "false") << ",\n"
            << "  \"full_dataset\": true,\n"
            << "  \"input_visibility\": \"public_full_dataset_tensor_summary\",\n"
            << "  \"output_visibility\": \"public_logits_when_verified\",\n"
            << "  \"param_visibility\": \"weight_constants_bound_by_checkpoint_hash\",\n"
            << "  \"native_primitive\": \"" << (wants_native_csr_route ? native_primitive_for_route(route) : "") << "\",\n"
            << "  \"native_primitive_version\": \"" << (wants_native_csr_route ? native_primitive_version_for_route(route) : "") << "\",\n"
            << "  \"csr_layout\": \"" << (wants_native_csr_route ? "csr_dst_segment_ragged" : "") << "\",\n"
            << "  \"model_arch_id\": \"" << gat_ezkl::json_escape(config.model_arch_id) << "\",\n"
            << "  \"model_param_id\": \"" << gat_ezkl::json_escape(config.model_param_id) << "\",\n"
            << "  \"quant_cfg_id\": \"" << gat_ezkl::json_escape(quant.quant_cfg_id) << "\",\n"
            << "  \"static_table_id\": \"" << gat_ezkl::json_escape(config.static_table_id) << "\",\n";
        if (formal_tables) {
            manifest
                << "  \"formal_tables\": " << gat_ezkl::formal_table_binding_json(*formal_tables) << ",\n"
                << "  \"formal_table_hash\": \"" << formal_tables->hashes.table_hash << "\",\n";
            manifest
                << "  \"inverse_policy\": \"" << gat_ezkl::json_escape(formal_tables->inverse_policy) << "\",\n"
                << "  \"inverse_scale_product\": " << formal_tables->inverse.inverse_scale_product << ",\n"
                << "  \"inverse_policy_hash\": \"" << formal_tables->hashes.inverse_policy_hash << "\",\n";
        }
        manifest
            << "  \"preprocess_id\": \"full_dataset_cpp_normalize_self_loop_dedup_sort_v1\",\n"
            << "  \"batching_rule\": \"" << gat_ezkl::json_escape(config.batching_rule) << "\",\n"
            << "  \"subgraph_rule\": \"" << gat_ezkl::json_escape(config.subgraph_rule) << "\",\n"
            << "  \"self_loop_rule\": \"" << gat_ezkl::json_escape(config.self_loop_rule) << "\",\n"
            << "  \"edge_sort_rule\": \"" << gat_ezkl::json_escape(config.edge_sort_rule) << "\",\n"
            << "  \"chunking_rule\": \"" << gat_ezkl::json_escape(config.chunking_rule) << "\",\n"
            << "  \"commitment_time_ms\": " << null_metric_json() << ",\n"
            << "  \"prove_time_ms\": " << null_metric_json() << ",\n"
            << "  \"verify_time_ms\": " << null_metric_json() << ",\n"
            << "  \"proof_size_bytes\": " << null_metric_json() << ",\n"
            << "  \"setup_ms\": " << null_metric_json() << ",\n"
            << "  \"gen_witness_ms\": " << null_metric_json() << ",\n"
            << "  \"compile_ms\": " << null_metric_json() << ",\n"
            << "  \"gen_settings_ms\": " << null_metric_json() << ",\n"
            << "  \"check_mode\": \"not_started\",\n"
            << "  \"public_input_element_count\": " << gat_ezkl::public_input_element_count(graph) << ",\n"
            << "  \"public_output_element_count\": " << gat_ezkl::public_output_element_count(graph) << ",\n"
            << "  \"N\": " << graph.node_count << ",\n"
            << "  \"E\": " << graph.edge_count << ",\n"
            << "  \"G_batch\": " << graph.graph_count << ",\n"
            << "  \"node_count\": " << graph.node_count << ",\n"
            << "  \"edge_count\": " << graph.edge_count << ",\n"
            << "  \"graph_count\": " << graph.graph_count << ",\n"
            << "  \"graph_hash\": \"" << graph_hash << "\",\n"
            << "  \"checkpoint_hash\": \"" << checkpoint_hash << "\",\n"
            << "  \"metadata_hash\": \"" << metadata_hash << "\",\n"
            << "  \"shape_hash\": \"" << shape_hash << "\",\n"
            << "  \"output_hash\": \"" << output_hash << "\",\n"
            << "  \"final_logits_binding_path\": \"final_logits_binding.json\",\n"
            << "  \"final_logits_binding_status\": \"public_instance_bound_to_derived_final_logits\",\n"
            << "  \"final_logits_shape\": [" << graph.node_count << "," << graph.class_count << "],\n"
            << "  \"public_instance_order\": \"final_logits_row_major_node_then_class_chunked_columns_column_then_row\",\n"
            << "  \"public_instance_layout\": \"chunked_columns\",\n"
            << "  \"public_instance_chunk_size\": " << public_instance_chunk_size << ",\n"
            << "  \"public_instance_column_count\": " << public_instance_column_count << ",\n"
            << "  \"public_instance_total_length\": " << public_instance_length << ",\n"
            << "  \"public_instance_length\": " << public_instance_length << ",\n"
            << "  \"output_average_policy\": \"" << gat_ezkl::json_escape(checkpoint.output_average_rule) << "\",\n"
            << "  \"output_rounding_policy\": \"integer_exact_no_post_average_rescale\",\n"
            << "  \"final_logits_derivation_complete\": true,\n"
            << "  \"output_derivation_required\": true,\n"
            << "  \"output_derivation_complete_candidate\": true,\n"
            << "  \"onnx_model_id\": \"blocked-artifact:" << shape_hash << "\",\n"
            << "  \"settings_id\": \"blocked-preflight:" << metadata_hash << "\",\n"
            << "  \"compiled_circuit_id\": null,\n"
            << "  \"vk_id\": null,\n"
            << "  \"pk_id\": null,\n"
            << "  \"forbidden_substitute_used\": false,\n"
            << "  \"precomputed_logits_binding_used\": false,\n"
            << "  \"verify_passed\": false,\n"
            << "  \"failure_step\": null,\n"
            << "  \"stdout_tail\": \"\",\n"
            << "  \"stderr_tail\": \"\"\n"
            << "}\n";
        gat_ezkl::write_text_file(output_dir / "run_manifest.json", manifest.str());

        std::cout << "dataset=" << run_id
                  << " route_id=" << decision.route_id
                  << " status=" << decision.status
                  << " block_reason=" << decision.block_reason
                  << " N=" << graph.node_count
                  << " E=" << graph.edge_count
                  << " G=" << graph.graph_count
                  << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gat_ezkl_export error: " << error.what() << std::endl;
        return 2;
    }
}
