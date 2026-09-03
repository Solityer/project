#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
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
    static const std::set<std::string> allowed = {
        "dataset", "config", "dataset-root", "checkpoint", "security-profile",
        "project-root", "output-dir", "route",
    };
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
            throw std::runtime_error("expected --key value arguments");
        }
        const auto name = key.substr(2);
        if (!allowed.count(name) || out.count(name)) {
            throw std::runtime_error("unsupported or duplicate option: " + key);
        }
        const auto value = std::string(argv[++i]);
        if (value.empty() || value.rfind("--", 0) == 0) {
            throw std::runtime_error("missing value for --" + name);
        }
        out.emplace(name, value);
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

std::filesystem::path absolute_lexical(const std::filesystem::path& path) {
    const auto absolute = path.is_absolute() ? path : std::filesystem::absolute(path);
    return absolute.lexically_normal();
}

bool path_is_within(const std::filesystem::path& child, const std::filesystem::path& parent) {
    const auto relative = absolute_lexical(child).lexically_relative(absolute_lexical(parent));
    if (relative.empty()) return true;
    for (const auto& component : relative) {
        if (component == std::filesystem::path("..")
            || component == std::filesystem::path(".")) {
            return false;
        }
    }
    return true;
}

void require_regular_file(const std::filesystem::path& path, const std::string& label) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || std::filesystem::is_symlink(status) || !std::filesystem::is_regular_file(status)) {
        throw std::runtime_error(label + " must be a regular non-symlink file: " + path.string());
    }
    const auto links = std::filesystem::hard_link_count(path, error);
    if (error || links != 1) {
        throw std::runtime_error(label + " must have exactly one hard link: " + path.string());
    }
}

void require_directory(const std::filesystem::path& path, const std::string& label) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || std::filesystem::is_symlink(status) || !std::filesystem::is_directory(status)) {
        throw std::runtime_error(label + " must be a regular non-symlink directory: " + path.string());
    }
}

void validate_path_tree(
    const std::filesystem::path& project_root,
    const std::filesystem::path& path,
    const std::string& label,
    bool final_may_be_missing) {
    const auto root = absolute_lexical(project_root);
    const auto candidate = absolute_lexical(path);
    if (!path_is_within(candidate, root) || candidate == root) {
        throw std::runtime_error(label + " escaped the baseline project root: " + candidate.string());
    }
    auto current = root;
    const auto relative = candidate.lexically_relative(root);
    std::vector<std::filesystem::path> components;
    for (const auto& component : relative) {
        if (component == std::filesystem::path("..")
            || component == std::filesystem::path(".")) {
            throw std::runtime_error(label + " has non-canonical path components");
        }
        components.push_back(component);
    }
    for (std::size_t index = 0; index < components.size(); ++index) {
        current /= components[index];
        std::error_code error;
        const auto status = std::filesystem::symlink_status(current, error);
        if (error || (!std::filesystem::exists(status) && index + 1 != components.size())) {
            throw std::runtime_error(label + " has an inaccessible parent component: " + current.string());
        }
        if (!std::filesystem::exists(status)) {
            if (index + 1 == components.size() && final_may_be_missing) return;
            throw std::runtime_error(label + " does not exist: " + current.string());
        }
        if (std::filesystem::is_symlink(status)) {
            throw std::runtime_error(label + " contains a symlink: " + current.string());
        }
        if (index + 1 != components.size() && !std::filesystem::is_directory(status)) {
            throw std::runtime_error(label + " contains a non-directory parent: " + current.string());
        }
    }
}

std::filesystem::path require_under(
    const std::filesystem::path& project_root,
    const std::filesystem::path& value,
    const std::vector<std::filesystem::path>& allowed_roots,
    const std::string& label,
    bool final_may_be_missing) {
    const auto path = absolute_lexical(value);
    bool allowed = false;
    for (const auto& root : allowed_roots) {
        if (path_is_within(path, absolute_lexical(root))) {
            allowed = true;
            break;
        }
    }
    if (!allowed) {
        throw std::runtime_error(label + " is outside its baseline-local allowed directory: " + path.string());
    }
    validate_path_tree(project_root, path, label, final_may_be_missing);
    return path;
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
        const auto project_root = absolute_lexical(required(args, "project-root"));
        require_directory(project_root, "project-root");
        if (project_root.filename() != "ezkl") {
            throw std::runtime_error("project-root must be the baseline-local comparison/ezkl directory");
        }
        const auto config_path = require_under(
            project_root, required(args, "config"), {project_root / "configs"}, "config", false);
        const auto dataset_root = require_under(
            project_root, required(args, "dataset-root"), {project_root / "data" / "cache"},
            "dataset-root", false);
        const auto checkpoint_root = require_under(
            project_root, required(args, "checkpoint"),
            {project_root / "artifacts" / "checkpoints"}, "checkpoint", false);
        const auto security_profile = require_under(
            project_root, required(args, "security-profile"),
            {project_root / "configs", project_root / "results" / "runs_tmp"},
            "security-profile", false);
        const auto output_dir = require_under(
            project_root, required(args, "output-dir"),
            {project_root / "results" / "runs", project_root / "results" / "runs_tmp"},
            "output-dir", true);
        const auto route = args.count("route") ? args.at("route") : kFormalRoute;
        if (route != kFormalRoute) throw std::runtime_error("unregistered formal GAT route");
        require_regular_file(config_path, "config");
        require_directory(dataset_root, "dataset-root");
        require_directory(checkpoint_root, "checkpoint");
        require_regular_file(security_profile, "security-profile");
        require_regular_file(dataset_root / "meta.cfg", "dataset meta.cfg");
        require_regular_file(dataset_root / "edges.txt", "dataset edges.txt");
        if (std::filesystem::exists(dataset_root / "features.txt")) {
            require_regular_file(dataset_root / "features.txt", "dataset features.txt");
        } else {
            require_regular_file(dataset_root / "features.npy", "dataset features.npy");
        }
        for (const auto& optional : {std::string("labels.txt"), std::string("labels_multi.txt"), std::string("node_ptr.txt")}) {
            if (std::filesystem::exists(dataset_root / optional)) {
                require_regular_file(dataset_root / optional, "dataset " + optional);
            }
        }
        require_regular_file(checkpoint_root / "manifest.json", "checkpoint manifest.json");
        require_regular_file(checkpoint_root / "tensors.npz", "checkpoint tensors.npz");
        require_regular_file(checkpoint_root / "tensors.txt", "checkpoint tensors.txt");
        if (std::filesystem::exists(output_dir)) require_directory(output_dir, "output-dir");
        std::filesystem::create_directories(output_dir);
        require_directory(output_dir, "output-dir");

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
            checkpoint_root / "manifest.json", checkpoint_root / "tensors.txt",
            checkpoint_root / "tensors.npz");
        const auto shape_hash = gat_ezkl::formal_hash_shape(graph, checkpoint);
        std::ostringstream metadata;
        metadata << "{\"kind\":\"ezkl_formal_gat_adapter_v1\",\"dataset\":\""
                 << gat_ezkl::json_escape(dataset_id) << "\",\"route_id\":\"" << route
                 << "\",\"input_preprocess_id\":\""
                 << gat_ezkl::json_escape(config.input_preprocess_id)
                 << "\",\"quant_cfg_id\":\""
                 << gat_ezkl::json_escape(config.quant_cfg_id)
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
            || security_text.find("\"source_srs_max_k\":28") == std::string::npos
            || security_text.find("\"semantic_version\":\"ezkl_formal_gat_semantics_v2\"") == std::string::npos
            || security_text.find("\"operation_registry_id\":\"sha256:") == std::string::npos
            || security_text.find("\"semantic_source_id\":\"sha256:") == std::string::npos) {
            throw std::runtime_error(
                "security profile must bind trusted ceremony source_srs_max_k exactly 28");
        }
        const auto security_profile_id = gat_ezkl::formal_sha256_text(
            "ezkl_security_profile", "canonical_json", security_text);
        std::filesystem::copy_file(security_profile, output_dir / "security_profile.json",
                                   std::filesystem::copy_options::overwrite_existing);

        const auto profile_ids = gat_ezkl::write_formal_gat_artifacts(
            output_dir, config, graph, checkpoint, shape_hash, max_in_degree(graph));
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
                 << "  \"input_preprocess_id\": \""
                 << gat_ezkl::json_escape(config.input_preprocess_id) << "\",\n"
                 << "  \"full_dataset\": true,\n"
                 << "  \"public_inputs\": [\"node_features.i64.bin\",\"csr_segment_layout.bin\"],\n"
                 << "  \"private_inputs\": [\"model_parameters.i64.txt\"],\n"
                 << "  \"formal_contracts\": [\"formal_quant_config.json\",\"formal_table_spec.json\",\"formal_main_tables.i64.bin\",\"backend_aux_table_spec.json\",\"backend_aux_tables.i64.bin\",\"integer_bound_ledger.json\",\"model_registration.json\",\"security_profile.json\"],\n"
                 << "  \"model_schema\": \"" << gat_ezkl::canonical_model_schema() << "\",\n"
                 << "  \"model_order\": \"" << gat_ezkl::json_escape(gat_ezkl::canonical_model_order()) << "\",\n"
                 << "  \"model_order_digest\": \"" << gat_ezkl::canonical_model_order_digest() << "\",\n"
                 << "  \"static_tables_id\": \"" << profile_ids.static_tables_id << "\",\n"
                 << "  \"backend_aux_tables_id\": \"" << profile_ids.backend_aux_tables_id << "\",\n"
                 << "  \"shape_hash\": \"" << shape_hash << "\",\n"
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
