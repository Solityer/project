#include "gat_ezkl/hash.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace gat_ezkl 
{
    // FNV-1a 循环：逐字节异或 + 乘大质数
    std::uint64_t fnv1a64_bytes(const void* data, std::size_t len, std::uint64_t seed) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        std::uint64_t hash = seed;
        for (std::size_t i = 0; i < len; ++i) {
            hash ^= static_cast<std::uint64_t>(bytes[i]);
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    std::uint64_t fnv1a64_string(const std::string& text, std::uint64_t seed) {
        return fnv1a64_bytes(text.data(), text.size(), seed);
    }

    std::string hex_u64(std::uint64_t value) {
        std::ostringstream out;
        out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << value;
        return out.str();
    }

    std::string hash_file_hex(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("failed to hash file: " + path.string());
        }
        std::uint64_t state = 1469598103934665603ULL;
        char buffer[8192];
        while (input) {
            input.read(buffer, sizeof(buffer));
            const auto got = input.gcount();
            if (got > 0) {
                state = fnv1a64_bytes(buffer, static_cast<std::size_t>(got), state);
            }
        }
        return hex_u64(state);
    }

    std::string hash_files_hex(const std::vector<std::filesystem::path>& paths) {
        std::uint64_t state = 1469598103934665603ULL;
        for (const auto& path : paths) {
            const auto h = hash_file_hex(path);
            state = fnv1a64_string(path.string(), state);
            state = fnv1a64_string(h, state);
        }
        return hex_u64(state);
    }

    std::string hash_config_hex(const AppConfig& config) {
        std::ostringstream out;
        out << config.dataset << '\n'
            << config.task_type << '\n'
            << config.report_unit << '\n'
            << config.batching_rule << '\n'
            << config.self_loop_rule << '\n'
            << config.edge_sort_rule << '\n'
            << config.chunking_rule << '\n'
            << config.layer_count << '\n'
            << config.K_out << '\n'
            << config.num_classes << '\n'
            << config.quant_cfg_id << '\n'
            << config.model_arch_id << '\n'
            << config.model_param_id << '\n';
        for (const auto& layer : config.hidden_profile) {
            out << layer.head_count << 'x' << layer.head_dim << ';';
        }
        out << '\n';
        for (const auto dim : config.d_in_profile) {
            out << dim << ',';
        }
        return hex_u64(fnv1a64_string(out.str()));
    }

    std::string hash_graph_hex(const NormalizedGraph& graph) {
        std::uint64_t state = 1469598103934665603ULL;
        auto mix_size = [&](std::size_t value) {
            state = fnv1a64_bytes(&value, sizeof(value), state);
        };
        mix_size(graph.n_total);
        mix_size(graph.node_count);
        mix_size(graph.edge_count);
        mix_size(graph.graph_count);
        mix_size(graph.feature_dim);
        mix_size(graph.class_count);
        for (const auto value : graph.absolute_ids) mix_size(value);
        for (const auto value : graph.node_ptr) mix_size(value);
        for (const auto value : graph.edge_ptr) mix_size(value);
        for (const auto value : graph.src) mix_size(value);
        for (const auto value : graph.dst) mix_size(value);
        for (const auto& row : graph.features_fp) {
            for (const auto value : row) {
                state = fnv1a64_bytes(&value, sizeof(value), state);
            }
        }
        return hex_u64(state);
    }

    std::string hash_shape_hex(const NormalizedGraph& graph, const CheckpointBundle& bundle) {
        std::ostringstream out;
        out << "N=" << graph.node_count
            << ";E=" << graph.edge_count
            << ";G=" << graph.graph_count
            << ";D=" << graph.feature_dim
            << ";C=" << graph.class_count
            << ";L=" << bundle.layer_count
            << ";K_out=" << bundle.K_out;
        for (const auto& layer : bundle.hidden_profile) {
            out << ";hid=" << layer.head_count << "x" << layer.head_dim;
        }
        out << ";d_in=";
        for (const auto dim : bundle.d_in_profile) {
            out << dim << ',';
        }
        return hex_u64(fnv1a64_string(out.str()));
    }

    std::string hash_metadata_hex(const std::string& metadata_json) {
        return hex_u64(fnv1a64_string(metadata_json));
    }

    std::string hash_output_shape_hex(const NormalizedGraph& graph) {
        const auto text = std::to_string(graph.node_count) + "x" + std::to_string(graph.class_count);
        return hex_u64(fnv1a64_string(text));
    }
}
