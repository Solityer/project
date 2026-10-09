#include "gat_proof.hpp"
#include "gat_types.hpp"

#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace zkgpt::gat;

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("expected absolute zkGPT baseline root");
        const fs::path root = fs::canonical(argv[1]);
        const fs::path config_path = root / "baseline-local/configs/cora_secure.cfg";
        const CanonicalConfig config = CanonicalConfig::load(config_path.string());
        if (config.get("dataset") != "cora") throw std::invalid_argument("Cora config identity mismatch");
        const fs::path cache_dir = root / config.get("cache_root") / "cora";
        const fs::path bundle_dir = root / config.get("checkpoint_bundle");
        const fs::path quant_path = root / "baseline-local/configs/quantization_scale_chain.cfg";
        const Graph graph = load_graph(cache_dir.string(), config);
        const Model model = Model::load_shape_only(bundle_dir.string());
        model.validate_identity(config);
        const QuantConfig quant = QuantConfig::load(quant_path.string());
        const fs::path text_features = cache_dir / "features.txt";
        const fs::path npy_features = cache_dir / "features.npy";
        const fs::path feature_path = fs::is_regular_file(text_features) ? text_features : npy_features;
        if (!fs::is_regular_file(feature_path)) throw std::runtime_error("canonical Cora feature file is missing");
        const std::uintmax_t feature_bytes = fs::file_size(feature_path);
        if (feature_bytes == 0 || feature_bytes > std::numeric_limits<std::uint64_t>::max())
            throw std::runtime_error("canonical Cora feature file size is invalid");

        initialize_gat_field();
        // This bounded diagnostic performs only the existing exact shape census.
        // inspect_gat_capacity does not run forward admission, SRS setup, proving,
        // or verification; these remain owned by the formal run.
        const GatCapacityReport report = inspect_gat_capacity(
            graph, model, quant, static_cast<std::uint64_t>(feature_bytes), false);
        if (report.gkr_layer_work_profile.size() != 113 ||
            report.gkr_point_shape_fconn_count != 54 ||
            !report.gkr_point_shape_exact || !report.gkr_work_counts_exact)
            throw std::runtime_error("exact Cora shape census did not return the canonical 113/54 profile");

        std::cout << "{\n"
                  << "  \"profile_kind\": \"exact_shape_census_only\",\n"
                  << "  \"formal_readiness\": false,\n"
                  << "  \"dataset\": \"cora\",\n"
                  << "  \"num_nodes\": " << report.num_nodes << ",\n"
                  << "  \"num_edges\": " << report.num_edges << ",\n"
                  << "  \"num_features\": " << report.num_features << ",\n"
                  << "  \"num_classes\": " << report.num_classes << ",\n"
                  << "  \"exact_source_support_entries\": " << report.gkr_subset_source_support_entries_upper_bound << ",\n"
                  << "  \"exact_max_layer0_lane_entries\": " << report.gkr_layer0_source_domain_entries_upper_bound << ",\n"
                  << "  \"exact_source_map_bytes\": " << report.gkr_subset_source_map_bytes_upper_bound << ",\n"
                  << "  \"exact_prefix_peak_bytes\": " << report.gkr_prefix_workspace_bytes_upper_bound << ",\n"
                  << "  \"exact_layer_count\": " << report.gkr_point_shape_layer_count << ",\n"
                  << "  \"exact_fconn_count\": " << report.gkr_point_shape_fconn_count << ",\n"
                  << "  \"gkr_point_shape_digest\": \"" << report.gkr_point_shape_digest << "\",\n"
                  << "  \"layers\": [\n";
        for (std::size_t index = 0; index < report.gkr_layer_work_profile.size(); ++index) {
            const GatGkrLayerShapeProfile& layer = report.gkr_layer_work_profile[index];
            std::cout << "    {\"index\":" << index
                      << ",\"layer_type\":" << layer.layer_type
                      << ",\"size\":" << layer.size
                      << ",\"size_u0\":" << layer.size_u0
                      << ",\"size_v0\":" << layer.size_v0
                      << ",\"size_u1\":" << layer.size_u1
                      << ",\"size_v1\":" << layer.size_v1
                      << ",\"unary_gates\":" << layer.unary_gates
                      << ",\"binary_gates\":" << layer.binary_gates
                      << ",\"max_bl_u\":" << layer.max_bl_u
                      << ",\"max_bl_v\":" << layer.max_bl_v
                      << ",\"need_phase2\":" << (layer.need_phase2 ? "true" : "false") << "}"
                      << (index + 1 == report.gkr_layer_work_profile.size() ? "\n" : ",\n");
        }
        std::cout << "  ]\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "EXACT_CORA_CAPACITY_PROBE_FAILED: " << error.what() << '\n';
        return 1;
    }
}
