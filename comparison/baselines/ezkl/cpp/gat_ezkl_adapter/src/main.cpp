#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

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
        if (normalized == "fail-on-padding-bound-unverified"
            || normalized == "emit-segment-debug") {
            out[normalized] = "true";
            continue;
        }
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

bool optional_bool_arg(
    const std::map<std::string, std::string>& args,
    const std::string& key) {
    const auto it = args.find(key);
    return it != args.end() && (it->second == "true" || it->second == "1" || it->second == "yes");
}

bool is_native_graph_route(const std::string& route) {
    return route == "ezkl_gat_native_csr_segment_attention_full"
        || route == "ezkl_gat_native_compressed_csr_attention_full";
}

bool is_compressed_native_graph_route(const std::string& route) {
    return route == "ezkl_gat_native_compressed_csr_attention_full";
}

std::string native_primitive_for_route(const std::string& route) {
    return is_compressed_native_graph_route(route)
        ? "GraphSegmentAttentionCompressedCSR"
        : "GraphSegmentAttention";
}

std::string native_primitive_version_for_route(const std::string& route) {
    return is_compressed_native_graph_route(route)
        ? "csr_segment_attention_compressed_v1"
        : "csr_segment_attention_v1";
}

std::size_t optional_size_arg(
    const std::map<std::string, std::string>& args,
    const std::string& key,
    std::size_t fallback) {
    const auto it = args.find(key);
    if (it == args.end() || it->second.empty()) {
        return fallback;
    }
    return static_cast<std::size_t>(std::stoull(it->second));
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

std::string manifest_segment_fields(const gat_ezkl::SegmentAnalysisReport* report) {
    if (report == nullptr) {
        return "";
    }
    std::ostringstream out;
    out << "  \"segment_lowering\": true,\n"
        << "  \"segment_layout\": \"" << gat_ezkl::json_escape(report->segment_layout) << "\",\n"
        << "  \"max_in_degree\": " << report->max_in_degree << ",\n"
        << "  \"padded_edge_slots\": " << report->total_packed_slots << ",\n"
        << "  \"original_edge_count\": " << report->edge_count << ",\n"
        << "  \"padding_ratio\": " << report->padding_ratio << ",\n"
        << "  \"sentinel_value\": " << report->sentinel_value << ",\n"
        << "  \"sentinel_bound_verified\": " << (report->sentinel_bound_verified ? "true" : "false") << ",\n"
        << "  \"valid_mask_hash\": \"" << report->hashes.valid_mask_hash << "\",\n"
        << "  \"segment_ptr_hash\": \"" << report->hashes.segment_ptr_hash << "\",\n"
        << "  \"packed_src_hash\": \"" << report->hashes.packed_src_hash << "\",\n"
        << "  \"packed_edge_index_hash\": \"" << report->hashes.packed_edge_index_hash << "\",\n";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        const auto run_id = require_string_arg(args, "dataset");
        const auto project_root = std::filesystem::weakly_canonical(require_path_arg(args, "project-root"));
        const auto ezkl_root = std::filesystem::weakly_canonical(require_path_arg(args, "ezkl-root"));
        const auto output_dir = require_path_arg(args, "output-dir");
        const auto route = optional_string_arg(args, "route", "strict_gat_full_onnx_v1");
        const auto analysis_only = optional_string_arg(args, "analysis-only", "");
        const auto max_packed_slots_limit = optional_size_arg(args, "max-packed-slots-limit", 16000000);
        const auto emit_segment_debug = optional_bool_arg(args, "emit-segment-debug");

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
        std::optional<gat_ezkl::SegmentAnalysisReport> segment_report;
        const bool wants_native_csr_route = is_native_graph_route(route);
        const bool wants_segment_route = route == "ezkl_gat_strict_segment_softmax_full"
            || analysis_only == "segment-softmax";
        if (wants_segment_route) {
            const auto bounds = gat_ezkl::choose_invalid_score_sentinel(
                gat_ezkl::infer_score_bounds_from_quant_cfg(quant, config.range_bits));
            segment_report = gat_ezkl::analyze_segment_softmax(
                run_id,
                config,
                graph,
                checkpoint,
                gat_summary,
                bounds,
                max_packed_slots_limit);
            gat_ezkl::write_segment_analysis_json(
                output_dir / "segment_softmax_analysis.json",
                *segment_report);
            gat_ezkl::write_text_file(
                output_dir / "segment_metadata.json",
                gat_ezkl::segment_metadata_json(*segment_report));
            if (emit_segment_debug) {
                std::string reason;
                const auto layout = gat_ezkl::build_segment_packed_layout(graph, &reason);
                if (reason.empty()) {
                    gat_ezkl::write_segment_debug_files(output_dir, layout);
                }
            }
            if (analysis_only == "segment-softmax") {
                std::cout << "dataset=" << run_id
                          << " route_id=" << route
                          << " analysis=segment-softmax"
                          << " N=" << graph.node_count
                          << " E=" << graph.edge_count
                          << " Dmax=" << segment_report->max_in_degree
                          << " packed_slots=" << segment_report->total_packed_slots
                          << " feasible=" << (segment_report->route_feasible_full ? "true" : "false")
                          << " blocker=" << segment_report->blocker_if_any
                          << std::endl;
                return 0;
            }
        }
        const auto graph_hash = gat_ezkl::hash_graph_hex(graph);
        const auto checkpoint_hash = gat_ezkl::hash_files_hex({checkpoint_root / "manifest.json", checkpoint_root / "tensors.txt"});
        const auto shape_hash = gat_ezkl::hash_shape_hex(graph, checkpoint);
        const auto output_hash = gat_ezkl::hash_output_shape_hex(graph);

        std::optional<gat_ezkl::FormalTablesSummary> formal_tables;
        if (wants_native_csr_route) {
            formal_tables = gat_ezkl::write_formal_table_artifacts(output_dir, config);
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
        const auto decision = wants_native_csr_route
            ? gat_ezkl::export_native_csr_segment_attention_gat(
                output_dir,
                route,
                config,
                graph,
                checkpoint,
                gat_summary)
            : route == "ezkl_gat_strict_segment_softmax_full"
            ? gat_ezkl::export_or_block_strict_segment_gat_onnx(
                output_dir,
                config,
                graph,
                checkpoint,
                gat_summary,
                *segment_report)
            : gat_ezkl::export_or_block_strict_gat_onnx(
                output_dir,
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
            settings << ",\n  \"formal_tables\": " << gat_ezkl::formal_table_binding_json(*formal_tables) << "\n";
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
            << manifest_segment_fields(segment_report ? &*segment_report : nullptr)
            << "  \"network_onnx_is_placeholder\": " << (decision.network_onnx_is_placeholder ? "true" : "false") << "\n"
            << "}\n";
        gat_ezkl::write_text_file(output_dir / "export_status.json", export_status.str());

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
            << manifest_segment_fields(segment_report ? &*segment_report : nullptr)
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
            << "  \"onnx_model_id\": \"blocked-placeholder:" << shape_hash << "\",\n"
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
