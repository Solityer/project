#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gat_ezkl/checkpoint.hpp"
#include "gat_ezkl/config.hpp"
#include "gat_ezkl/dataset.hpp"
#include "gat_ezkl/formal_csr.hpp"
#include "gat_ezkl/formal_gat.hpp"
#include "gat_ezkl/formal_io.hpp"
#include "gat_ezkl/graph.hpp"
#include "gat_ezkl/formal_hash.hpp"

namespace {

constexpr const char* kFormalRoute = "ezkl_gat_native_compressed_csr_attention_full";

std::map<std::string, std::string> parse_args(int argc, char** argv) {
    std::map<std::string, std::string> out;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
            throw std::runtime_error("expected --key value arguments");
        }
        out[key.substr(2)] = argv[++i];
    }
    return out;
}

std::string required(const std::map<std::string, std::string>& args, const std::string& key) {
    const auto it = args.find(key);
    if (it == args.end() || it->second.empty()) {
        throw std::runtime_error("missing required argument --" + key);
    }
    return it->second;
}

bool path_is_within(const std::filesystem::path& child, const std::filesystem::path& parent) {
    const auto canonical_child = std::filesystem::weakly_canonical(child);
    const auto canonical_parent = std::filesystem::weakly_canonical(parent);
    auto child_it = canonical_child.begin();
    for (auto parent_it = canonical_parent.begin(); parent_it != canonical_parent.end();
         ++parent_it, ++child_it) {
        if (child_it == canonical_child.end() || *child_it != *parent_it) return false;
    }
    return true;
}

std::size_t max_in_degree(const gat_ezkl::NormalizedGraph& graph) {
    std::vector<std::size_t> degree(graph.node_count, 0);
    for (const auto& edge : graph.edges) {
        if (edge.dst >= degree.size()) throw std::runtime_error("edge endpoint outside graph");
        ++degree[edge.dst];
    }
    const auto value = *std::max_element(degree.begin(), degree.end());
    if (value == 0) throw std::runtime_error("formal graph must contain incoming edges");
    if (std::any_of(degree.begin(), degree.end(), [](std::size_t item) { return item == 0; })) {
        throw std::runtime_error("formal graph is missing required normalized self-loops");
    }
    return value;
}

void print_help() {
    std::cout
        << "Usage: gat_ezkl_export --dataset ID --config FILE --dataset-root DIR "
           "--checkpoint DIR --security-profile FILE --ezkl-root DIR --output-dir DIR\n"
        << "Emits canonical integer inputs and formal GAT circuit contracts.\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            print_help();
            return 0;
        }
        const auto args = parse_args(argc, argv);
        const auto dataset_id = required(args, "dataset");
        const auto config_path = std::filesystem::weakly_canonical(required(args, "config"));
        const auto dataset_root = std::filesystem::weakly_canonical(required(args, "dataset-root"));
        const auto checkpoint_root = std::filesystem::weakly_canonical(required(args, "checkpoint"));
        const auto ezkl_root = std::filesystem::weakly_canonical(required(args, "ezkl-root"));
        const auto output_dir = std::filesystem::weakly_canonical(required(args, "output-dir"));
        const auto security_profile = std::filesystem::weakly_canonical(required(args, "security-profile"));
        const auto route = args.count("route") ? args.at("route") : kFormalRoute;
        if (route != kFormalRoute) throw std::runtime_error("unregistered formal GAT route");
        if (!path_is_within(output_dir, ezkl_root)) {
            throw std::runtime_error("output-dir must be within the baseline-local ezkl root");
        }
        std::filesystem::create_directories(output_dir);

        const auto official = gat_ezkl::official_dataset_spec(dataset_id);
        auto config = gat_ezkl::load_config_file(config_path);
        gat_ezkl::validate_full_dataset_contract(config, official);
        if (dataset_root.filename() != config.dataset) {
            throw std::runtime_error("dataset-root basename must equal the configured dataset id");
        }
        config.cache_root = dataset_root.parent_path().string();
        const auto dataset = gat_ezkl::load_dataset_from_cache({}, config);
        auto checkpoint = gat_ezkl::load_checkpoint_bundle(checkpoint_root);
        gat_ezkl::validate_checkpoint_against_config(checkpoint, &config);
        if (config.d_in_profile.empty() || dataset.num_features != config.d_in_profile.front()
            || dataset.num_classes != config.num_classes) {
            throw std::runtime_error("dataset shape does not match formal configuration");
        }
        const auto graph = gat_ezkl::normalize_full_graph(dataset, config);
        if (graph.node_count != dataset.num_nodes) {
            throw std::runtime_error("formal adapter refuses sampled/subgraph inputs");
        }
        std::string graph_reason;
        if (!gat_ezkl::graph_blocks_are_valid(graph, &graph_reason)
            || !gat_ezkl::edge_order_is_gid_then_dst_stable(graph, &graph_reason)) {
            throw std::runtime_error("canonical topology validation failed: " + graph_reason);
        }

        const auto graph_hash = gat_ezkl::formal_hash_graph(graph);
        const auto checkpoint_hash = gat_ezkl::formal_hash_checkpoint(
            checkpoint_root / "manifest.json", checkpoint_root / "tensors.txt");
        const auto shape_hash = gat_ezkl::formal_hash_shape(graph, checkpoint);
        std::ostringstream metadata;
        metadata << "{\"kind\":\"ezkl_formal_gat_adapter_v1\",\"dataset\":\""
                 << gat_ezkl::json_escape(dataset_id) << "\",\"route_id\":\"" << route
                 << "\",\"full_dataset\":true,\"canonical_integer_semantics\":true}\n";
        const auto metadata_text = metadata.str();
        const auto metadata_hash = gat_ezkl::formal_hash_metadata(metadata_text);
        std::ifstream security_input(security_profile, std::ios::binary);
        if (!security_input) throw std::runtime_error("failed to read security profile");
        std::string security_text((std::istreambuf_iterator<char>(security_input)), {});
        while (!security_text.empty() && (security_text.back() == '\n' || security_text.back() == '\r'))
            security_text.pop_back();
        if (security_text.find("\"ceremony_descriptor_kind\":\"ezkl_trusted_ceremony_srs_v1\"") == std::string::npos
            || security_text.find("\"source_srs_sha256\":\"sha256:") == std::string::npos
            || security_text.find("\"source_srs_max_k\":") == std::string::npos) {
            throw std::runtime_error(
                "security profile is unresolved; materialize it from the trusted ceremony descriptor and SRS first");
        }
        const auto security_profile_id = gat_ezkl::formal_sha256_text(
            "ezkl_security_profile", "canonical_json", security_text);
        std::filesystem::copy_file(security_profile, output_dir / "security_profile.json",
                                   std::filesystem::copy_options::overwrite_existing);

        const auto profile_ids = gat_ezkl::write_formal_gat_artifacts(
            output_dir, config, graph, checkpoint, max_in_degree(graph));
        gat_ezkl::write_formal_csr_graph_artifacts(
            output_dir, route, config, graph, checkpoint, graph_hash, checkpoint_hash,
            metadata_hash, shape_hash, profile_ids.quant_cfg_id, profile_ids.static_tables_id,
            profile_ids.backend_aux_tables_id,
            security_profile_id);
        gat_ezkl::write_text_file(output_dir / "metadata.json", metadata_text);

        std::ostringstream manifest;
        manifest << "{\n"
                 << "  \"manifest_kind\": \"ezkl_formal_adapter_manifest_v1\",\n"
                 << "  \"status\": \"canonical_artifacts_ready\",\n"
                 << "  \"dataset\": \"" << gat_ezkl::json_escape(dataset_id) << "\",\n"
                 << "  \"route_id\": \"" << route << "\",\n"
                 << "  \"full_dataset\": true,\n"
                 << "  \"public_inputs\": [\"node_features.i64.bin\",\"csr_segment_layout.bin\"],\n"
                 << "  \"private_inputs\": [\"model_parameters.i64.txt\"],\n"
                 << "  \"formal_contracts\": [\"formal_quant_config.json\",\"formal_table_spec.json\",\"formal_main_tables.i64.bin\",\"backend_aux_table_spec.json\",\"backend_aux_tables.i64.bin\",\"integer_bound_ledger.json\",\"model_registration.json\",\"security_profile.json\"],\n"
                 << "  \"static_tables_id\": \"" << profile_ids.static_tables_id << "\",\n"
                 << "  \"backend_aux_tables_id\": \"" << profile_ids.backend_aux_tables_id << "\",\n"
                 << "  \"ezkl_security_profile_id\": \"" << security_profile_id << "\",\n"
                 << "  \"witness_generation\": \"ezkl graph-gat-gen-witness reconstructs the full constrained trace\",\n"
                 << "  \"attention_bias_used\": false,\n"
                 << "  \"precomputed_logits_used\": false\n"
                 << "}\n";
        gat_ezkl::write_text_file(output_dir / "adapter_manifest.json", manifest.str());
        gat_ezkl::write_text_file(output_dir / "graph_hash.txt", graph_hash + "\n");
        gat_ezkl::write_text_file(output_dir / "checkpoint_hash.txt", checkpoint_hash + "\n");
        gat_ezkl::write_text_file(output_dir / "shape_hash.txt", shape_hash + "\n");

        std::cout << "dataset=" << dataset_id << " status=canonical_artifacts_ready"
                  << " N=" << graph.node_count << " E=" << graph.edge_count
                  << " G=" << graph.graph_count << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gat_ezkl_export error: " << error.what() << std::endl;
        return 2;
    }
}
