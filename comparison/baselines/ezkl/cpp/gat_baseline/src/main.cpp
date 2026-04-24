#include "gat_baseline/formal_config.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct Edge {
    std::size_t graph_id = 0;
    std::size_t src = 0;
    std::size_t dst = 0;
    std::size_t stable_index = 0;
};

struct DenseTensor {
    std::vector<std::size_t> shape;
    std::vector<float> values;
};

struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> values;

    Matrix() = default;

    Matrix(std::size_t row_count, std::size_t col_count)
        : rows(row_count), cols(col_count), values(row_count * col_count, 0.0f) {}

    float& at(std::size_t row, std::size_t col) {
        return values[row * cols + col];
    }

    float at(std::size_t row, std::size_t col) const {
        return values[row * cols + col];
    }
};

struct GraphDataset {
    std::size_t num_nodes = 0;
    std::size_t num_features = 0;
    std::size_t num_classes = 0;
    std::size_t graph_count = 1;
    Matrix features;
    std::vector<std::int64_t> labels;
    std::vector<Edge> edges;
    std::vector<std::size_t> node_ptr;
};

struct GraphBundle {
    Matrix features;
    std::vector<std::int64_t> labels;
    std::vector<std::int64_t> src;
    std::vector<std::int64_t> dst;
    std::vector<std::int64_t> node_ptr;
    std::vector<std::int64_t> edge_ptr;
    std::vector<std::int64_t> absolute_ids;
    std::size_t graph_count = 1;
    std::size_t max_in_degree = 0;
};

struct HeadSpec {
    std::size_t layer_index = 0;
    std::size_t local_head_index = 0;
    std::size_t global_head_index = 0;
    std::string seq_kernel;
    std::string attn_dst_kernel;
    std::string attn_dst_bias;
    std::string attn_src_kernel;
    std::string attn_src_bias;
    std::string output_bias;
};

struct OutputHeadSpec {
    std::size_t head_index = 0;
    std::string seq_kernel;
    std::string attn_dst_kernel;
    std::string attn_dst_bias;
    std::string attn_src_kernel;
    std::string attn_src_bias;
    std::string output_bias;
};

struct LayerInfo {
    std::size_t layer_index = 0;
    std::size_t input_dim = 0;
    std::size_t head_count = 0;
    std::size_t head_dim = 0;
};

struct BundleInfo {
    std::string model_arch_id;
    std::string model_param_id;
    std::string quant_cfg_id;
    std::string output_average_rule;
    std::size_t num_classes = 0;
    std::vector<LayerInfo> hidden_layers;
    std::vector<HeadSpec> hidden_head_specs;
    std::vector<OutputHeadSpec> output_head_specs;
};

struct HeadWeights {
    Matrix seq_kernel;
    std::vector<float> attn_src_kernel;
    float attn_src_bias = 0.0f;
    std::vector<float> attn_dst_kernel;
    float attn_dst_bias = 0.0f;
    std::vector<float> output_bias;
};

struct LayerWeights {
    LayerInfo info;
    std::vector<HeadWeights> heads;
};

struct Checkpoint {
    BundleInfo info;
    std::vector<LayerWeights> hidden_layers;
    std::vector<HeadWeights> output_heads;
};

struct Args {
    std::string dataset;
    std::string task_type;
    std::string report_unit;
    fs::path config_path;
    fs::path cache_root;
    fs::path checkpoint_root;
    fs::path output_root;
    std::size_t batch_graphs = 1;
    bool deduplicate_edges = true;
    bool symmetrize_edges = false;
};

std::uint32_t ceil_log2_u64(std::uint64_t value) {
    if (value <= 1) {
        return 0;
    }

    value -= 1;
    std::uint32_t logrows = 0;
    while (value > 0) {
        value >>= 1;
        ++logrows;
    }
    return logrows;
}

std::string bool_string(bool value) {
    return value ? "true" : "false";
}

fs::path normalized_absolute(const fs::path& path) {
    return fs::absolute(path).lexically_normal();
}

void require_match(const std::string& name, const std::string& expected, const std::string& actual) {
    if (expected != actual) {
        throw std::runtime_error(
            "config mismatch for " + name + ": expected '" + expected + "', got '" + actual + "'");
    }
}

void require_match(const std::string& name, std::size_t expected, std::size_t actual) {
    if (expected != actual) {
        throw std::runtime_error(
            "config mismatch for " + name + ": expected " + std::to_string(expected)
            + ", got " + std::to_string(actual));
    }
}

void require_match(const std::string& name, bool expected, bool actual) {
    if (expected != actual) {
        throw std::runtime_error(
            "config mismatch for " + name + ": expected " + bool_string(expected)
            + ", got " + bool_string(actual));
    }
}

void require_path_match(const std::string& name, const fs::path& expected, const fs::path& actual) {
    const auto normalized_expected = normalized_absolute(expected);
    const auto normalized_actual = normalized_absolute(actual);
    if (normalized_expected != normalized_actual) {
        throw std::runtime_error(
            "config mismatch for " + name + ": expected " + normalized_expected.string()
            + ", got " + normalized_actual.string());
    }
}

void validate_formal_config(const gat_baseline::FormalConfig& config, const Args& args) {
    require_match("dataset", config.dataset, args.dataset);
    require_match("task_type", config.task_type, args.task_type);
    require_match("report_unit", config.report_unit, args.report_unit);
    require_match("batch_graphs", config.batch_graphs, args.batch_graphs);
    require_match("deduplicate_edges", config.deduplicate_edges, args.deduplicate_edges);
    require_match("symmetrize_edges", config.symmetrize_edges, args.symmetrize_edges);
    require_path_match("cache_root", gat_baseline::dataset_cache_path(config), args.cache_root);
    require_path_match(
        "checkpoint_bundle",
        gat_baseline::resolve_project_path(config, config.checkpoint_bundle),
        args.checkpoint_root);
    require_path_match(
        "export_dir",
        gat_baseline::resolve_project_path(config, config.export_dir),
        args.output_root.parent_path());
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const auto ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                out << ch;
                break;
        }
    }
    return out.str();
}

std::string quote(const std::string& value) {
    return "\"" + json_escape(value) + "\"";
}

std::unordered_map<std::string, std::string> read_meta(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open meta file: " + path.string());
    }
    std::unordered_map<std::string, std::string> out;
    std::string line;
    while (std::getline(input, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        auto key = line.substr(0, eq);
        auto value = line.substr(eq + 1);
        key.erase(std::remove_if(key.begin(), key.end(), [](unsigned char ch) { return std::isspace(ch) != 0; }), key.end());
        value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) { return std::isspace(ch) == 0; }));
        value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) { return std::isspace(ch) == 0; }).base(), value.end());
        if (!key.empty()) {
            out[key] = value;
        }
    }
    return out;
}

struct NpyArray {
    std::vector<std::size_t> shape;
    std::vector<float> values;
};

NpyArray load_npy_float(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open npy file: " + path.string());
    }

    char magic[6] = {};
    input.read(magic, sizeof(magic));
    if (std::string(magic, sizeof(magic)) != "\x93NUMPY") {
        throw std::runtime_error("invalid npy magic: " + path.string());
    }

    std::uint8_t major = 0;
    std::uint8_t minor = 0;
    input.read(reinterpret_cast<char*>(&major), 1);
    input.read(reinterpret_cast<char*>(&minor), 1);
    (void)minor;

    std::uint32_t header_len = 0;
    if (major == 1) {
        std::uint16_t short_len = 0;
        input.read(reinterpret_cast<char*>(&short_len), sizeof(short_len));
        header_len = short_len;
    } else {
        input.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    }

    std::string header(header_len, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));

    const auto descr_pos = header.find("'descr'");
    if (descr_pos == std::string::npos) {
        throw std::runtime_error("missing descr in npy header: " + path.string());
    }
    const auto descr_quote = header.find('\'', header.find(':', descr_pos));
    const auto descr_end = header.find('\'', descr_quote + 1);
    const auto descr = header.substr(descr_quote + 1, descr_end - descr_quote - 1);
    const bool is_float64 = descr == "<f8" || descr == "|f8";
    const bool is_float32 = descr == "<f4" || descr == "|f4";
    if (!is_float64 && !is_float32) {
        throw std::runtime_error("expected float32/float64 npy payload: " + path.string());
    }

    const auto shape_pos = header.find('(', header.find("shape"));
    const auto shape_end = header.find(')', shape_pos);
    if (shape_pos == std::string::npos || shape_end == std::string::npos) {
        throw std::runtime_error("missing shape in npy header: " + path.string());
    }

    NpyArray out;
    std::stringstream shape_stream(header.substr(shape_pos + 1, shape_end - shape_pos - 1));
    std::string token;
    while (std::getline(shape_stream, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char ch) { return std::isspace(ch) != 0; }), token.end());
        if (!token.empty()) {
            out.shape.push_back(std::stoull(token));
        }
    }
    if (out.shape.empty()) {
        throw std::runtime_error("empty shape in npy header: " + path.string());
    }

    std::size_t count = 1;
    for (const auto dim : out.shape) {
        count *= dim;
    }

    out.values.resize(count, 0.0f);
    if (is_float64) {
        std::vector<double> buffer(count, 0.0);
        input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(count * sizeof(double)));
        for (std::size_t index = 0; index < count; ++index) {
            out.values[index] = static_cast<float>(buffer[index]);
        }
    } else {
        input.read(reinterpret_cast<char*>(out.values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    }
    return out;
}

Matrix load_features(const fs::path& root, std::size_t rows, std::size_t cols) {
    const auto txt_path = root / "features.txt";
    const auto npy_path = root / "features.npy";
    Matrix out(rows, cols);
    if (fs::exists(txt_path)) {
        std::ifstream input(txt_path);
        if (!input) {
            throw std::runtime_error("failed to open features file: " + txt_path.string());
        }
        std::size_t file_rows = 0;
        std::size_t file_cols = 0;
        input >> file_rows >> file_cols;
        std::string line;
        std::getline(input, line);
        if (file_rows != rows || file_cols != cols) {
            throw std::runtime_error("feature metadata mismatch in " + txt_path.string());
        }
        for (std::size_t row = 0; row < rows; ++row) {
            if (!std::getline(input, line)) {
                throw std::runtime_error("unexpected EOF in " + txt_path.string());
            }
            std::istringstream line_stream(line);
            std::string entry;
            while (line_stream >> entry) {
                const auto colon = entry.find(':');
                if (colon == std::string::npos) {
                    continue;
                }
                const auto index = static_cast<std::size_t>(std::stoull(entry.substr(0, colon)));
                const auto value = std::stof(entry.substr(colon + 1));
                out.at(row, index) = value;
            }
        }
    } else {
        if (!fs::exists(npy_path)) {
            throw std::runtime_error("missing feature file under " + root.string());
        }
        const auto array = load_npy_float(npy_path);
        if (array.shape.size() != 2 || array.shape[0] != rows || array.shape[1] != cols) {
            throw std::runtime_error("feature npy shape mismatch in " + npy_path.string());
        }
        out.values = array.values;
    }

    for (std::size_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            sum += static_cast<double>(out.at(row, col));
        }
        if (sum == 0.0) {
            continue;
        }
        const auto scale = static_cast<float>(1.0 / sum);
        for (std::size_t col = 0; col < cols; ++col) {
            out.at(row, col) *= scale;
        }
    }
    return out;
}

std::vector<std::int64_t> load_labels(const fs::path& root, std::size_t rows) {
    const auto path = root / "labels.txt";
    if (!fs::exists(path)) {
        return std::vector<std::int64_t>(rows, 0);
    }
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open labels file: " + path.string());
    }
    std::vector<std::int64_t> out;
    out.reserve(rows);
    std::int64_t value = 0;
    while (input >> value) {
        out.push_back(value);
    }
    if (out.size() != rows) {
        throw std::runtime_error("label count mismatch in " + path.string());
    }
    return out;
}

std::vector<std::size_t> load_size_vector(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open vector file: " + path.string());
    }
    std::vector<std::size_t> out;
    std::size_t value = 0;
    while (input >> value) {
        out.push_back(value);
    }
    return out;
}

std::vector<Edge> load_edges(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open edges file: " + path.string());
    }
    std::vector<Edge> out;
    std::string line;
    std::size_t stable_index = 0;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        std::vector<std::size_t> parts;
        std::size_t value = 0;
        while (iss >> value) {
            parts.push_back(value);
        }
        if (parts.empty()) {
            continue;
        }
        Edge edge;
        edge.stable_index = stable_index++;
        if (parts.size() == 2) {
            edge.graph_id = 0;
            edge.src = parts[0];
            edge.dst = parts[1];
        } else if (parts.size() == 3) {
            edge.graph_id = parts[0];
            edge.src = parts[1];
            edge.dst = parts[2];
        } else {
            throw std::runtime_error("malformed edge line in " + path.string());
        }
        out.push_back(edge);
    }
    return out;
}

GraphDataset load_dataset(const fs::path& cache_root) {
    const auto meta = read_meta(cache_root / "meta.cfg");
    GraphDataset dataset;
    dataset.num_nodes = std::stoull(meta.at("num_nodes"));
    dataset.num_features = std::stoull(meta.at("num_features"));
    dataset.num_classes = std::stoull(meta.at("num_classes"));
    dataset.features = load_features(cache_root, dataset.num_nodes, dataset.num_features);
    dataset.labels = load_labels(cache_root, dataset.num_nodes);
    if (fs::exists(cache_root / "node_ptr.txt")) {
        dataset.node_ptr = load_size_vector(cache_root / "node_ptr.txt");
    } else {
        dataset.node_ptr = {0, dataset.num_nodes};
    }
    if (dataset.node_ptr.size() < 2) {
        dataset.node_ptr = {0, dataset.num_nodes};
    }
    dataset.graph_count = dataset.node_ptr.size() - 1;
    dataset.edges = load_edges(cache_root / "edges.txt");
    return dataset;
}

std::uint64_t edge_key(std::size_t src, std::size_t dst, std::size_t graph_id) {
    return (static_cast<std::uint64_t>(graph_id) << 48U)
        ^ (static_cast<std::uint64_t>(src) << 24U)
        ^ static_cast<std::uint64_t>(dst);
}

GraphBundle normalize_graph(const GraphDataset& dataset, const Args& args) {
    GraphBundle bundle;
    const auto selected_count = std::min(std::max<std::size_t>(1, args.batch_graphs), dataset.graph_count);
    std::vector<std::size_t> batch_node_ptr = {0};
    std::vector<std::size_t> batch_edge_ptr = {0};
    std::vector<Edge> batch_edges;

    for (std::size_t batch_graph_id = 0; batch_graph_id < selected_count; ++batch_graph_id) {
        const auto dataset_graph_id = batch_graph_id;
        const auto node_begin = dataset.node_ptr.at(dataset_graph_id);
        const auto node_end = dataset.node_ptr.at(dataset_graph_id + 1);
        std::vector<std::size_t> selected_nodes(node_end - node_begin, 0);
        std::iota(selected_nodes.begin(), selected_nodes.end(), node_begin);
        std::sort(selected_nodes.begin(), selected_nodes.end());

        const auto local_offset = batch_node_ptr.back();
        batch_node_ptr.push_back(local_offset + selected_nodes.size());

        std::unordered_map<std::size_t, std::size_t> global_to_local;
        global_to_local.reserve(selected_nodes.size());
        for (std::size_t index = 0; index < selected_nodes.size(); ++index) {
            global_to_local[selected_nodes[index]] = local_offset + index;
            bundle.absolute_ids.push_back(static_cast<std::int64_t>(selected_nodes[index]));
            bundle.labels.push_back(dataset.labels.at(selected_nodes[index]));
        }

        std::vector<Edge> graph_edges;
        std::unordered_set<std::uint64_t> seen;
        std::size_t stable_index = 0;
        auto append_edge = [&](std::size_t src_global, std::size_t dst_global) {
            const auto src_local = global_to_local.at(src_global);
            const auto dst_local = global_to_local.at(dst_global);
            const auto key = edge_key(src_local, dst_local, batch_graph_id);
            if (args.deduplicate_edges && !seen.insert(key).second) {
                return;
            }
            if (!args.deduplicate_edges) {
                seen.insert(key);
            }
            graph_edges.push_back(Edge{batch_graph_id, src_local, dst_local, stable_index++});
        };

        for (const auto& edge : dataset.edges) {
            if (edge.graph_id != dataset_graph_id) {
                continue;
            }
            if (!global_to_local.contains(edge.src) || !global_to_local.contains(edge.dst)) {
                continue;
            }
            append_edge(edge.src, edge.dst);
            if (args.symmetrize_edges && edge.src != edge.dst) {
                append_edge(edge.dst, edge.src);
            }
        }

        for (const auto node_id : selected_nodes) {
            append_edge(node_id, node_id);
        }

        std::stable_sort(graph_edges.begin(), graph_edges.end(), [](const Edge& lhs, const Edge& rhs) {
            if (lhs.graph_id != rhs.graph_id) {
                return lhs.graph_id < rhs.graph_id;
            }
            if (lhs.dst != rhs.dst) {
                return lhs.dst < rhs.dst;
            }
            return lhs.stable_index < rhs.stable_index;
        });

        batch_edge_ptr.push_back(batch_edge_ptr.back() + graph_edges.size());
        batch_edges.insert(batch_edges.end(), graph_edges.begin(), graph_edges.end());
    }

    bundle.graph_count = selected_count;
    bundle.features = Matrix(bundle.absolute_ids.size(), dataset.num_features);
    for (std::size_t row = 0; row < bundle.absolute_ids.size(); ++row) {
        const auto absolute_id = static_cast<std::size_t>(bundle.absolute_ids[row]);
        for (std::size_t col = 0; col < dataset.num_features; ++col) {
            bundle.features.at(row, col) = dataset.features.at(absolute_id, col);
        }
    }
    bundle.src.reserve(batch_edges.size());
    bundle.dst.reserve(batch_edges.size());
    for (const auto& edge : batch_edges) {
        bundle.src.push_back(static_cast<std::int64_t>(edge.src));
        bundle.dst.push_back(static_cast<std::int64_t>(edge.dst));
    }
    bundle.node_ptr.reserve(batch_node_ptr.size());
    bundle.edge_ptr.reserve(batch_edge_ptr.size());
    for (const auto value : batch_node_ptr) {
        bundle.node_ptr.push_back(static_cast<std::int64_t>(value));
    }
    for (const auto value : batch_edge_ptr) {
        bundle.edge_ptr.push_back(static_cast<std::int64_t>(value));
    }

    std::vector<std::size_t> incoming(bundle.features.rows, 0);
    for (const auto value : bundle.dst) {
        ++incoming[static_cast<std::size_t>(value)];
    }
    bundle.max_in_degree = *std::max_element(incoming.begin(), incoming.end());
    return bundle;
}

std::string load_text_file(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open file: " + path.string());
    }
    std::ostringstream stream;
    stream << input.rdbuf();
    return stream.str();
}

std::size_t parse_required_size_t(const std::string& text, const std::string& key) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing numeric key in checkpoint manifest: " + key);
    }
    return static_cast<std::size_t>(std::stoull(match[1].str()));
}

std::string parse_required_string(const std::string& text, const std::string& key) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing string key in checkpoint manifest: " + key);
    }
    return match[1].str();
}

std::string extract_array_body(const std::string& text, const std::string& key) {
    const auto key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing array key in checkpoint manifest: " + key);
    }
    const auto array_begin = text.find('[', key_pos);
    if (array_begin == std::string::npos) {
        throw std::runtime_error("malformed array in checkpoint manifest: " + key);
    }
    std::size_t depth = 0;
    for (std::size_t index = array_begin; index < text.size(); ++index) {
        if (text[index] == '[') {
            ++depth;
        } else if (text[index] == ']') {
            --depth;
            if (depth == 0) {
                return text.substr(array_begin + 1, index - array_begin - 1);
            }
        }
    }
    throw std::runtime_error("unterminated array in checkpoint manifest: " + key);
}

std::vector<std::string> parse_object_array_entries(const std::string& text, const std::string& key) {
    const auto body = extract_array_body(text, key);
    std::vector<std::string> entries;
    std::size_t depth = 0;
    std::size_t object_begin = std::string::npos;
    for (std::size_t index = 0; index < body.size(); ++index) {
        if (body[index] == '{') {
            if (depth == 0) {
                object_begin = index;
            }
            ++depth;
        } else if (body[index] == '}') {
            if (depth == 0) {
                throw std::runtime_error("malformed object array in checkpoint manifest: " + key);
            }
            --depth;
            if (depth == 0 && object_begin != std::string::npos) {
                entries.push_back(body.substr(object_begin, index - object_begin + 1));
                object_begin = std::string::npos;
            }
        }
    }
    return entries;
}

LayerInfo parse_layer_info(const std::string& entry) {
    LayerInfo out;
    out.layer_index = parse_required_size_t(entry, "layer_index");
    out.input_dim = parse_required_size_t(entry, "input_dim");
    out.head_count = parse_required_size_t(entry, "head_count");
    out.head_dim = parse_required_size_t(entry, "head_dim");
    return out;
}

HeadSpec parse_head_spec(const std::string& entry) {
    HeadSpec out;
    out.layer_index = parse_required_size_t(entry, "layer_index");
    out.local_head_index = parse_required_size_t(entry, "local_head_index");
    out.global_head_index = parse_required_size_t(entry, "global_head_index");
    out.seq_kernel = parse_required_string(entry, "seq_kernel");
    out.attn_dst_kernel = parse_required_string(entry, "attn_dst_kernel");
    out.attn_dst_bias = parse_required_string(entry, "attn_dst_bias");
    out.attn_src_kernel = parse_required_string(entry, "attn_src_kernel");
    out.attn_src_bias = parse_required_string(entry, "attn_src_bias");
    out.output_bias = parse_required_string(entry, "output_bias");
    return out;
}

OutputHeadSpec parse_output_head_spec(const std::string& entry) {
    OutputHeadSpec out;
    out.head_index = parse_required_size_t(entry, "head_index");
    out.seq_kernel = parse_required_string(entry, "seq_kernel");
    out.attn_dst_kernel = parse_required_string(entry, "attn_dst_kernel");
    out.attn_dst_bias = parse_required_string(entry, "attn_dst_bias");
    out.attn_src_kernel = parse_required_string(entry, "attn_src_kernel");
    out.attn_src_bias = parse_required_string(entry, "attn_src_bias");
    out.output_bias = parse_required_string(entry, "output_bias");
    return out;
}

std::unordered_map<std::string, DenseTensor> load_text_tensors(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open checkpoint tensor dump: " + path.string());
    }

    std::unordered_map<std::string, DenseTensor> out;
    std::string marker;
    while (input >> marker) {
        if (marker != "TENSOR") {
            throw std::runtime_error("malformed tensor dump marker in " + path.string());
        }
        std::string name;
        std::size_t rank = 0;
        input >> name >> rank;
        DenseTensor tensor;
        tensor.shape.resize(rank);
        for (std::size_t index = 0; index < rank; ++index) {
            input >> tensor.shape[index];
        }
        std::size_t count = 0;
        input >> count;
        tensor.values.resize(count, 0.0f);
        double value = 0.0;
        for (std::size_t index = 0; index < count; ++index) {
            input >> value;
            tensor.values[index] = static_cast<float>(value);
        }
        out.emplace(name, std::move(tensor));
    }
    return out;
}

const DenseTensor& require_tensor(const std::unordered_map<std::string, DenseTensor>& tensors, const std::string& name) {
    const auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::runtime_error("missing tensor in checkpoint dump: " + name);
    }
    return it->second;
}

Matrix reshape_seq_kernel(const DenseTensor& tensor) {
    if (tensor.shape.size() != 3 || tensor.shape[0] != 1) {
        throw std::runtime_error("expected seq kernel tensor shape [1, in, out]");
    }
    Matrix out(tensor.shape[1], tensor.shape[2]);
    out.values = tensor.values;
    return out;
}

std::vector<float> reshape_attention_kernel(const DenseTensor& tensor) {
    if (tensor.shape.size() != 3 || tensor.shape[0] != 1 || tensor.shape[2] != 1) {
        throw std::runtime_error("expected attention kernel tensor shape [1, width, 1]");
    }
    return tensor.values;
}

float reshape_scalar_bias(const DenseTensor& tensor) {
    if (tensor.shape.size() != 1 || tensor.shape[0] != 1 || tensor.values.size() != 1) {
        throw std::runtime_error("expected scalar bias tensor shape [1]");
    }
    return tensor.values.front();
}

std::vector<float> reshape_output_bias(const DenseTensor& tensor) {
    if (tensor.shape.size() != 1) {
        throw std::runtime_error("expected output bias tensor shape [width]");
    }
    return tensor.values;
}

HeadWeights load_head_weights(const std::unordered_map<std::string, DenseTensor>& tensors, const HeadSpec& spec) {
    return HeadWeights{
        reshape_seq_kernel(require_tensor(tensors, spec.seq_kernel)),
        reshape_attention_kernel(require_tensor(tensors, spec.attn_src_kernel)),
        reshape_scalar_bias(require_tensor(tensors, spec.attn_src_bias)),
        reshape_attention_kernel(require_tensor(tensors, spec.attn_dst_kernel)),
        reshape_scalar_bias(require_tensor(tensors, spec.attn_dst_bias)),
        reshape_output_bias(require_tensor(tensors, spec.output_bias)),
    };
}

HeadWeights load_output_head_weights(const std::unordered_map<std::string, DenseTensor>& tensors, const OutputHeadSpec& spec) {
    return HeadWeights{
        reshape_seq_kernel(require_tensor(tensors, spec.seq_kernel)),
        reshape_attention_kernel(require_tensor(tensors, spec.attn_src_kernel)),
        reshape_scalar_bias(require_tensor(tensors, spec.attn_src_bias)),
        reshape_attention_kernel(require_tensor(tensors, spec.attn_dst_kernel)),
        reshape_scalar_bias(require_tensor(tensors, spec.attn_dst_bias)),
        reshape_output_bias(require_tensor(tensors, spec.output_bias)),
    };
}

Checkpoint load_checkpoint(const fs::path& checkpoint_root) {
    const auto manifest_text = load_text_file(checkpoint_root / "manifest.json");
    BundleInfo info;
    info.model_arch_id = parse_required_string(manifest_text, "model_arch_id");
    info.model_param_id = parse_required_string(manifest_text, "model_param_id");
    info.quant_cfg_id = parse_required_string(manifest_text, "quant_cfg_id");
    info.output_average_rule = parse_required_string(manifest_text, "output_average_rule");
    info.num_classes = parse_required_size_t(manifest_text, "C");
    for (const auto& entry : parse_object_array_entries(manifest_text, "hidden_layers")) {
        info.hidden_layers.push_back(parse_layer_info(entry));
    }
    for (const auto& entry : parse_object_array_entries(manifest_text, "hidden_head_specs")) {
        info.hidden_head_specs.push_back(parse_head_spec(entry));
    }
    for (const auto& entry : parse_object_array_entries(manifest_text, "output_head_specs")) {
        info.output_head_specs.push_back(parse_output_head_spec(entry));
    }

    const auto tensors = load_text_tensors(checkpoint_root / "tensors.txt");
    Checkpoint checkpoint;
    checkpoint.info = info;
    checkpoint.hidden_layers.reserve(info.hidden_layers.size());
    for (const auto& layer_info : info.hidden_layers) {
        LayerWeights layer;
        layer.info = layer_info;
        for (const auto& head_spec : info.hidden_head_specs) {
            if (head_spec.layer_index == layer_info.layer_index) {
                layer.heads.push_back(load_head_weights(tensors, head_spec));
            }
        }
        if (layer.heads.size() != layer_info.head_count) {
            throw std::runtime_error("hidden head count mismatch while loading checkpoint bundle");
        }
        checkpoint.hidden_layers.push_back(std::move(layer));
    }
    checkpoint.output_heads.reserve(info.output_head_specs.size());
    for (const auto& head_spec : info.output_head_specs) {
        checkpoint.output_heads.push_back(load_output_head_weights(tensors, head_spec));
    }
    return checkpoint;
}

std::string archive_key(const std::string& name) {
    std::string out = name;
    std::size_t pos = 0;
    while ((pos = out.find('/', pos)) != std::string::npos) {
        out.replace(pos, 1, "__");
        pos += 2;
    }
    pos = 0;
    while ((pos = out.find(':', pos)) != std::string::npos) {
        out.replace(pos, 1, "__");
        pos += 2;
    }
    return out;
}

template <typename T>
void write_npy(const fs::path& path, const std::string& descr, const std::vector<std::size_t>& shape, const std::vector<T>& values) {
    std::size_t count = 1;
    for (const auto dim : shape) {
        count *= dim;
    }
    if (count != values.size()) {
        throw std::runtime_error("npy shape/value mismatch for " + path.string());
    }
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open output npy: " + path.string());
    }

    std::ostringstream shape_stream;
    shape_stream << '(';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            shape_stream << ", ";
        }
        shape_stream << shape[index];
    }
    if (shape.size() == 1) {
        shape_stream << ',';
    }
    shape_stream << ')';

    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape_stream.str() + ", }";
    const std::size_t preamble = 10;
    std::size_t header_size = header.size() + 1;
    const auto remainder = (preamble + header_size) % 16;
    if (remainder != 0) {
        header.append(16 - remainder, ' ');
        header_size = header.size() + 1;
    }
    header.push_back('\n');

    const std::array<unsigned char, 6> magic = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    output.write(reinterpret_cast<const char*>(magic.data()), magic.size());
    const std::array<unsigned char, 2> version = {1, 0};
    output.write(reinterpret_cast<const char*>(version.data()), version.size());
    const std::uint16_t encoded_header_size = static_cast<std::uint16_t>(header.size());
    output.write(reinterpret_cast<const char*>(&encoded_header_size), sizeof(encoded_header_size));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
}

Matrix matmul(const Matrix& left, const Matrix& right) {
    if (left.cols != right.rows) {
        throw std::runtime_error("matrix multiply dimension mismatch");
    }
    Matrix out(left.rows, right.cols);
    for (std::size_t row = 0; row < left.rows; ++row) {
        for (std::size_t shared = 0; shared < left.cols; ++shared) {
            const auto lhs = left.at(row, shared);
            for (std::size_t col = 0; col < right.cols; ++col) {
                out.at(row, col) += lhs * right.at(shared, col);
            }
        }
    }
    return out;
}

std::vector<float> project_scores(const Matrix& matrix, const std::vector<float>& weights, float bias) {
    if (matrix.cols != weights.size()) {
        throw std::runtime_error("score projection dimension mismatch");
    }
    std::vector<float> out(matrix.rows, bias);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            out[row] += matrix.at(row, col) * weights[col];
        }
    }
    return out;
}

float leaky_relu(float value) {
    return value >= 0.0f ? value : value * 0.2f;
}

float elu(float value) {
    return value >= 0.0f ? value : std::expm1(value);
}

std::vector<std::vector<std::size_t>> incoming_sources(const GraphBundle& graph) {
    std::vector<std::vector<std::size_t>> incoming(graph.features.rows);
    for (std::size_t edge_index = 0; edge_index < graph.src.size(); ++edge_index) {
        incoming[static_cast<std::size_t>(graph.dst[edge_index])].push_back(static_cast<std::size_t>(graph.src[edge_index]));
    }
    for (std::size_t node = 0; node < incoming.size(); ++node) {
        if (incoming[node].empty()) {
            throw std::runtime_error("node has no incoming edges after normalization");
        }
    }
    return incoming;
}

Matrix run_head(const Matrix& input, const GraphBundle& graph, const std::vector<std::vector<std::size_t>>& incoming, const HeadWeights& head, bool apply_output_bias, bool apply_activation) {
    auto h_prime = matmul(input, head.seq_kernel);
    const auto e_src = project_scores(h_prime, head.attn_src_kernel, head.attn_src_bias);
    const auto e_dst = project_scores(h_prime, head.attn_dst_kernel, head.attn_dst_bias);
    Matrix out(graph.features.rows, h_prime.cols);

    for (std::size_t node = 0; node < graph.features.rows; ++node) {
        float max_score = -std::numeric_limits<float>::infinity();
        for (const auto src : incoming[node]) {
            max_score = std::max(max_score, leaky_relu(e_src[src] + e_dst[node]));
        }
        std::vector<float> exp_scores;
        exp_scores.reserve(incoming[node].size());
        float denom = 0.0f;
        for (const auto src : incoming[node]) {
            const auto exp_score = static_cast<float>(std::exp(leaky_relu(e_src[src] + e_dst[node]) - max_score));
            exp_scores.push_back(exp_score);
            denom += exp_score;
        }
        for (std::size_t edge_index = 0; edge_index < incoming[node].size(); ++edge_index) {
            const auto src = incoming[node][edge_index];
            const auto alpha = exp_scores[edge_index] / denom;
            for (std::size_t col = 0; col < h_prime.cols; ++col) {
                out.at(node, col) += alpha * h_prime.at(src, col);
            }
        }
        for (std::size_t col = 0; col < h_prime.cols; ++col) {
            auto value = out.at(node, col);
            if (apply_output_bias && col < head.output_bias.size()) {
                value += head.output_bias[col];
            }
            if (apply_activation) {
                value = elu(value);
            }
            out.at(node, col) = value;
        }
    }

    return out;
}

Matrix concat_columns(const std::vector<Matrix>& inputs) {
    if (inputs.empty()) {
        return {};
    }
    const auto rows = inputs.front().rows;
    std::size_t cols = 0;
    for (const auto& matrix : inputs) {
        if (matrix.rows != rows) {
            throw std::runtime_error("concat row mismatch");
        }
        cols += matrix.cols;
    }
    Matrix out(rows, cols);
    std::size_t offset = 0;
    for (const auto& matrix : inputs) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < matrix.cols; ++col) {
                out.at(row, offset + col) = matrix.at(row, col);
            }
        }
        offset += matrix.cols;
    }
    return out;
}

Matrix average_matrices(const std::vector<Matrix>& inputs) {
    if (inputs.empty()) {
        return {};
    }
    Matrix out = inputs.front();
    for (std::size_t index = 1; index < inputs.size(); ++index) {
        if (inputs[index].rows != out.rows || inputs[index].cols != out.cols) {
            throw std::runtime_error("average matrix shape mismatch");
        }
        for (std::size_t offset = 0; offset < out.values.size(); ++offset) {
            out.values[offset] += inputs[index].values[offset];
        }
    }
    const auto scale = static_cast<float>(1.0 / static_cast<double>(inputs.size()));
    for (auto& value : out.values) {
        value *= scale;
    }
    return out;
}

Matrix run_reference_forward(const Checkpoint& checkpoint, const GraphBundle& graph) {
    const auto incoming = incoming_sources(graph);
    Matrix hidden = graph.features;
    for (const auto& layer : checkpoint.hidden_layers) {
        std::vector<Matrix> head_outputs;
        head_outputs.reserve(layer.heads.size());
        for (const auto& head : layer.heads) {
            head_outputs.push_back(run_head(hidden, graph, incoming, head, false, true));
        }
        hidden = concat_columns(head_outputs);
    }

    std::vector<Matrix> output_heads;
    output_heads.reserve(checkpoint.output_heads.size());
    for (const auto& head : checkpoint.output_heads) {
        output_heads.push_back(run_head(hidden, graph, incoming, head, true, false));
    }
    return average_matrices(output_heads);
}

Args parse_args(int argc, char** argv) {
    Args args;
    std::unordered_map<std::string, std::string> values;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (!key.starts_with("--")) {
            throw std::runtime_error("unexpected positional argument: " + key);
        }
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for argument: " + key);
        }
        values[key.substr(2)] = argv[++index];
    }

    auto require = [&](const std::string& key) -> std::string {
        const auto it = values.find(key);
        if (it == values.end() || it->second.empty()) {
            throw std::runtime_error("missing required argument --" + key);
        }
        return it->second;
    };

    args.dataset = require("dataset");
    args.task_type = require("task-type");
    args.report_unit = require("report-unit");
    args.config_path = require("config-path");
    args.cache_root = require("cache-root");
    args.checkpoint_root = require("checkpoint-root");
    args.output_root = require("output-root");
    args.batch_graphs = static_cast<std::size_t>(std::stoull(require("batch-graphs")));
    args.deduplicate_edges = require("deduplicate-edges") == "1";
    args.symmetrize_edges = require("symmetrize-edges") == "1";
    return args;
}

void write_stage_manifest(
    const fs::path& path,
    const Args& args,
    const gat_baseline::FormalConfig& config,
    const GraphBundle& graph,
    const Matrix& logits,
    const Checkpoint& checkpoint,
    const std::unordered_map<std::string, std::string>& tensor_files) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open stage manifest for write: " + path.string());
    }

    const auto public_feature_values = static_cast<std::uint64_t>(graph.features.rows) * static_cast<std::uint64_t>(graph.features.cols);
    const auto public_input_count =
        public_feature_values
        + static_cast<std::uint64_t>(graph.src.size())
        + static_cast<std::uint64_t>(graph.dst.size())
        + static_cast<std::uint64_t>(graph.node_ptr.size())
        + static_cast<std::uint64_t>(graph.edge_ptr.size());
    const auto public_output_count =
        (static_cast<std::uint64_t>(logits.rows) * static_cast<std::uint64_t>(logits.cols))
        + 4U;
    constexpr std::uint64_t kPublicSafetyMargin = 64U;
    const auto total_public_instances = public_input_count + public_output_count + kPublicSafetyMargin;
    const auto minimum_public_logrows = std::max<std::uint32_t>(17U, ceil_log2_u64(total_public_instances));

    output << "{\n";
    output << "  \"dataset\": " << quote(args.dataset) << ",\n";
    output << "  \"task_type\": " << quote(args.task_type) << ",\n";
    output << "  \"report_unit\": " << quote(args.report_unit) << ",\n";
    output << "  \"node_count\": " << graph.features.rows << ",\n";
    output << "  \"edge_count\": " << graph.src.size() << ",\n";
    output << "  \"graph_count\": " << graph.graph_count << ",\n";
    output << "  \"max_in_degree\": " << graph.max_in_degree << ",\n";
    output << "  \"num_classes\": " << checkpoint.info.num_classes << ",\n";
    output << "  \"model_arch_id\": " << quote(checkpoint.info.model_arch_id) << ",\n";
    output << "  \"model_param_id\": " << quote(checkpoint.info.model_param_id) << ",\n";
    output << "  \"quant_cfg_id\": " << quote(checkpoint.info.quant_cfg_id) << ",\n";
    output << "  \"output_average_rule\": " << quote(checkpoint.info.output_average_rule) << ",\n";
    output << "  \"config\": {\n";
    output << "    \"config_path\": " << quote(normalized_absolute(config.config_path).string()) << ",\n";
    output << "    \"batching_rule\": " << quote(config.batching_rule) << ",\n";
    output << "    \"subgraph_rule\": " << quote(config.subgraph_rule) << ",\n";
    output << "    \"self_loop_rule\": " << quote(config.self_loop_rule) << ",\n";
    output << "    \"edge_sort_rule\": " << quote(config.edge_sort_rule) << ",\n";
    output << "    \"chunking_rule\": " << quote(config.chunking_rule) << ",\n";
    output << "    \"static_table_id\": " << quote(config.static_table_id) << ",\n";
    output << "    \"degree_bound_id\": " << quote(config.degree_bound_id) << ",\n";
    output << "    \"layer_count\": " << config.layer_count << ",\n";
    output << "    \"output_head_count\": " << config.output_head_count << ",\n";
    output << "    \"batch_graphs\": " << config.batch_graphs << ",\n";
    output << "    \"deduplicate_edges\": " << (config.deduplicate_edges ? "true" : "false") << ",\n";
    output << "    \"symmetrize_edges\": " << (config.symmetrize_edges ? "true" : "false") << "\n";
    output << "  },\n";
    output << "  \"public_io\": {\n";
    output << "    \"input\": {\n";
    output << "      \"features\": " << public_feature_values << ",\n";
    output << "      \"src\": " << graph.src.size() << ",\n";
    output << "      \"dst\": " << graph.dst.size() << ",\n";
    output << "      \"node_ptr\": " << graph.node_ptr.size() << ",\n";
    output << "      \"edge_ptr\": " << graph.edge_ptr.size() << "\n";
    output << "    },\n";
    output << "    \"output\": {\n";
    output << "      \"logits\": " << (static_cast<std::uint64_t>(logits.rows) * static_cast<std::uint64_t>(logits.cols)) << ",\n";
    output << "      \"graph_checks\": 4\n";
    output << "    },\n";
    output << "    \"public_input_count\": " << public_input_count << ",\n";
    output << "    \"public_output_count\": " << public_output_count << ",\n";
    output << "    \"safety_margin\": " << kPublicSafetyMargin << ",\n";
    output << "    \"total_public_instances\": " << total_public_instances << ",\n";
    output << "    \"minimum_public_logrows\": " << minimum_public_logrows << "\n";
    output << "  },\n";
    output << "  \"graph_tensors\": {\n";
    output << "    \"features\": \"features.npy\",\n";
    output << "    \"labels\": \"labels.npy\",\n";
    output << "    \"src\": \"src.npy\",\n";
    output << "    \"dst\": \"dst.npy\",\n";
    output << "    \"node_ptr\": \"node_ptr.npy\",\n";
    output << "    \"edge_ptr\": \"edge_ptr.npy\",\n";
    output << "    \"absolute_ids\": \"absolute_ids.npy\",\n";
    output << "    \"reference_logits\": \"reference_logits.npy\",\n";
    output << "    \"graph_checks\": \"graph_checks.npy\"\n";
    output << "  },\n";
    output << "  \"hidden_layers\": [\n";
    for (std::size_t index = 0; index < checkpoint.info.hidden_layers.size(); ++index) {
        const auto& layer = checkpoint.info.hidden_layers[index];
        output << "    {\"layer_index\": " << layer.layer_index
               << ", \"input_dim\": " << layer.input_dim
               << ", \"head_count\": " << layer.head_count
               << ", \"head_dim\": " << layer.head_dim << "}";
        output << (index + 1 == checkpoint.info.hidden_layers.size() ? "\n" : ",\n");
    }
    output << "  ],\n";
    output << "  \"hidden_head_specs\": [\n";
    for (std::size_t index = 0; index < checkpoint.info.hidden_head_specs.size(); ++index) {
        const auto& head = checkpoint.info.hidden_head_specs[index];
        output << "    {"
               << "\"layer_index\": " << head.layer_index
               << ", \"local_head_index\": " << head.local_head_index
               << ", \"global_head_index\": " << head.global_head_index
               << ", \"seq_kernel\": " << quote(head.seq_kernel)
               << ", \"attn_dst_kernel\": " << quote(head.attn_dst_kernel)
               << ", \"attn_dst_bias\": " << quote(head.attn_dst_bias)
               << ", \"attn_src_kernel\": " << quote(head.attn_src_kernel)
               << ", \"attn_src_bias\": " << quote(head.attn_src_bias)
               << ", \"output_bias\": " << quote(head.output_bias)
               << "}";
        output << (index + 1 == checkpoint.info.hidden_head_specs.size() ? "\n" : ",\n");
    }
    output << "  ],\n";
    output << "  \"output_head_specs\": [\n";
    for (std::size_t index = 0; index < checkpoint.info.output_head_specs.size(); ++index) {
        const auto& head = checkpoint.info.output_head_specs[index];
        output << "    {"
               << "\"head_index\": " << head.head_index
               << ", \"seq_kernel\": " << quote(head.seq_kernel)
               << ", \"attn_dst_kernel\": " << quote(head.attn_dst_kernel)
               << ", \"attn_dst_bias\": " << quote(head.attn_dst_bias)
               << ", \"attn_src_kernel\": " << quote(head.attn_src_kernel)
               << ", \"attn_src_bias\": " << quote(head.attn_src_bias)
               << ", \"output_bias\": " << quote(head.output_bias)
               << "}";
        output << (index + 1 == checkpoint.info.output_head_specs.size() ? "\n" : ",\n");
    }
    output << "  ],\n";
    output << "  \"tensor_files\": {\n";
    std::size_t index = 0;
    for (const auto& [name, relpath] : tensor_files) {
        output << "    " << quote(name) << ": " << quote(relpath);
        output << (++index == tensor_files.size() ? "\n" : ",\n");
    }
    output << "  }\n";
    output << "}\n";
}

int run(const Args& args) {
    fs::create_directories(args.output_root);

    const auto config = gat_baseline::load_formal_config(args.config_path);
    validate_formal_config(config, args);
    const auto dataset = load_dataset(args.cache_root);
    const auto graph = normalize_graph(dataset, args);
    const auto checkpoint = load_checkpoint(args.checkpoint_root);
    const auto logits = run_reference_forward(checkpoint, graph);

    write_npy(args.output_root / "features.npy", "<f4", {graph.features.rows, graph.features.cols}, graph.features.values);
    write_npy(args.output_root / "labels.npy", "<i8", {graph.labels.size()}, graph.labels);
    write_npy(args.output_root / "src.npy", "<i8", {graph.src.size()}, graph.src);
    write_npy(args.output_root / "dst.npy", "<i8", {graph.dst.size()}, graph.dst);
    write_npy(args.output_root / "node_ptr.npy", "<i8", {graph.node_ptr.size()}, graph.node_ptr);
    write_npy(args.output_root / "edge_ptr.npy", "<i8", {graph.edge_ptr.size()}, graph.edge_ptr);
    write_npy(args.output_root / "absolute_ids.npy", "<i8", {graph.absolute_ids.size()}, graph.absolute_ids);
    write_npy(args.output_root / "reference_logits.npy", "<f4", {logits.rows, logits.cols}, logits.values);
    write_npy(args.output_root / "graph_checks.npy", "<f4", {4}, std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f});

    const auto tensors = load_text_tensors(args.checkpoint_root / "tensors.txt");
    std::unordered_map<std::string, std::string> tensor_files;
    for (const auto& [name, tensor] : tensors) {
        const auto relpath = fs::path("weights") / (archive_key(name) + ".npy");
        tensor_files.emplace(name, relpath.generic_string());
        write_npy(args.output_root / relpath, "<f4", tensor.shape, tensor.values);
    }

    write_stage_manifest(args.output_root / "stage_manifest.json", args, config, graph, logits, checkpoint, tensor_files);

    std::cout << "dataset=" << args.dataset
              << " nodes=" << graph.features.rows
              << " edges=" << graph.src.size()
              << " graphs=" << graph.graph_count
              << " max_in_degree=" << graph.max_in_degree
              << " model_arch_id=" << checkpoint.info.model_arch_id
              << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }
}
