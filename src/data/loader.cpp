#include "gatzk/data/loader.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace gatzk::data {
namespace {

using algebra::FieldElement;

std::vector<std::vector<double>> row_normalize(
    const std::vector<std::vector<double>>& raw_features) {
    std::vector<std::vector<double>> normalized = raw_features;
    for (auto& row : normalized) {
        double sum = 0.0;
        for (const auto value : row) {
            sum += value;
        }
        if (sum == 0.0) {
            continue;
        }
        for (auto& value : row) {
            value /= sum;
        }
    }
    return normalized;
}

std::unordered_map<std::string, std::string> read_meta(const std::filesystem::path& path) {
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
        key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
        value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), value.end());
        out[key] = value;
    }
    return out;
}

FieldElement quantize(double value) {
    return FieldElement::from_signed(static_cast<std::int64_t>(value >= 0.0 ? value * 16.0 + 0.5 : value * 16.0 - 0.5));
}

struct NpyArray {
    std::vector<std::size_t> shape;
    std::vector<double> values;
};

NpyArray load_npy_double(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.good()) {
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
    } else if (major == 2 || major == 3) {
        input.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    } else {
        throw std::runtime_error("unsupported npy version: " + path.string());
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

    const auto fortran_pos = header.find("fortran_order");
    if (fortran_pos == std::string::npos || header.find("False", fortran_pos) == std::string::npos) {
        throw std::runtime_error("fortran-order arrays are not supported: " + path.string());
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
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
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
    out.values.resize(count, 0.0);
    if (is_float64) {
        input.read(reinterpret_cast<char*>(out.values.data()), static_cast<std::streamsize>(count * sizeof(double)));
    } else {
        std::vector<float> float_values(count, 0.0f);
        input.read(reinterpret_cast<char*>(float_values.data()), static_cast<std::streamsize>(count * sizeof(float)));
        for (std::size_t i = 0; i < count; ++i) {
            out.values[i] = static_cast<double>(float_values[i]);
        }
    }
    return out;
}

std::vector<std::vector<double>> load_features_matrix(
    const std::filesystem::path& root,
    std::size_t rows,
    std::size_t cols) {
    if (std::filesystem::exists(root / "features.txt")) {
        std::vector<std::vector<double>> raw_features(rows, std::vector<double>(cols, 0.0));
        std::ifstream input(root / "features.txt");
        if (!input) {
            throw std::runtime_error("failed to open features file");
        }
        std::size_t file_rows = 0;
        std::size_t file_cols = 0;
        input >> file_rows >> file_cols;
        std::string line;
        std::getline(input, line);
        if (file_rows != rows || file_cols != cols) {
            throw std::runtime_error("feature metadata mismatch");
        }
        for (std::size_t row = 0; row < file_rows; ++row) {
            std::getline(input, line);
            std::istringstream iss(line);
            std::string entry;
            while (iss >> entry) {
                const auto colon = entry.find(':');
                const auto index = static_cast<std::size_t>(std::stoull(entry.substr(0, colon)));
                const auto value = std::stod(entry.substr(colon + 1));
                raw_features[row][index] = value;
            }
        }
        return raw_features;
    }

    const auto array = load_npy_double(root / "features.npy");
    if (array.shape.size() != 2 || array.shape[0] != rows || array.shape[1] != cols) {
        throw std::runtime_error("feature npy shape mismatch");
    }
    std::vector<std::vector<double>> out(rows, std::vector<double>(cols, 0.0));
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            out[row][col] = array.values[row * cols + col];
        }
    }
    return out;
}

std::vector<int> load_labels(const std::filesystem::path& root, std::size_t rows) {
    std::vector<int> labels;
    if (!std::filesystem::exists(root / "labels.txt")) {
        labels.assign(rows, 0);
        return labels;
    }
    std::ifstream input(root / "labels.txt");
    if (!input) {
        throw std::runtime_error("failed to open labels file");
    }
    labels.reserve(rows);
    int label = 0;
    while (input >> label) {
        labels.push_back(label);
    }
    if (labels.size() != rows) {
        throw std::runtime_error("label count mismatch");
    }
    return labels;
}

std::vector<std::size_t> load_size_vector(const std::filesystem::path& path) {
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

GraphDataset load_cached_dataset(const std::filesystem::path& root) {
    const auto meta = read_meta(root / "meta.cfg");
    GraphDataset dataset;
    dataset.name = meta.at("name");
    dataset.num_nodes = std::stoull(meta.at("num_nodes"));
    dataset.num_features = std::stoull(meta.at("num_features"));
    dataset.num_classes = std::stoull(meta.at("num_classes"));
    if (meta.contains("graph_count")) {
        dataset.graph_count = std::stoull(meta.at("graph_count"));
    }
    if (meta.contains("task_type")) {
        dataset.task_type = meta.at("task_type");
    }
    if (meta.contains("report_unit")) {
        dataset.report_unit = meta.at("report_unit");
    }

    const auto raw_features = load_features_matrix(root, dataset.num_nodes, dataset.num_features);
    dataset.features_fp = row_normalize(raw_features);
    dataset.features.assign(dataset.num_nodes, std::vector<FieldElement>(dataset.num_features, FieldElement::zero()));
    for (std::size_t row = 0; row < dataset.num_nodes; ++row) {
        for (std::size_t col = 0; col < dataset.num_features; ++col) {
            dataset.features[row][col] = quantize(dataset.features_fp[row][col]);
        }
    }
    dataset.labels = load_labels(root, dataset.num_nodes);

    if (std::filesystem::exists(root / "node_ptr.txt")) {
        dataset.node_ptr = load_size_vector(root / "node_ptr.txt");
    } else {
        dataset.node_ptr = {0, dataset.num_nodes};
    }
    if (dataset.node_ptr.size() < 2) {
        dataset.node_ptr = {0, dataset.num_nodes};
    }
    dataset.graph_count = dataset.node_ptr.size() - 1;

    {
        std::ifstream input(root / "edges.txt");
        if (!input) {
            throw std::runtime_error("failed to open edges file");
        }
        std::string line;
        std::size_t stable_index = 0;
        while (std::getline(input, line)) {
            if (line.empty()) {
                continue;
            }
            std::istringstream iss(line);
            Edge edge;
            edge.graph_id = 0;
            edge.stable_index = stable_index++;
            std::vector<std::size_t> parts;
            std::size_t value = 0;
            while (iss >> value) {
                parts.push_back(value);
            }
            if (parts.size() == 2) {
                edge.src = parts[0];
                edge.dst = parts[1];
            } else if (parts.size() == 3) {
                edge.graph_id = parts[0];
                edge.src = parts[1];
                edge.dst = parts[2];
            } else {
                throw std::runtime_error("malformed edge line in " + (root / "edges.txt").string());
            }
            dataset.edges.push_back(edge);
        }
    }
    dataset.edge_ptr.assign(dataset.graph_count + 1, 0);
    for (const auto& edge : dataset.edges) {
        if (edge.graph_id >= dataset.graph_count) {
            throw std::runtime_error("edge graph_id out of range");
        }
        ++dataset.edge_ptr[edge.graph_id + 1];
    }
    std::partial_sum(dataset.edge_ptr.begin(), dataset.edge_ptr.end(), dataset.edge_ptr.begin());
    return dataset;
}

void maybe_prepare_planetoid_cache(const util::AppConfig& config, const std::filesystem::path& cache_root) {
    if (!config.auto_prepare_dataset || std::filesystem::exists(cache_root / "meta.cfg")) {
        return;
    }
    const std::filesystem::path project_root(config.project_root);
    const auto python = std::filesystem::exists(project_root / ".venv/bin/python")
        ? (project_root / ".venv/bin/python").string()
        : std::string("python3");
    const auto script = (project_root / "scripts/prepare_planetoid.py").string();
    const auto command = python + " " + script +
        " --data-root " + (project_root / config.data_root).string() +
        " --dataset " + config.dataset +
        " --cache-root " + (project_root / config.cache_root).string();
    if (std::system(command.c_str()) != 0) {
        throw std::runtime_error("failed to prepare planetoid cache for dataset " + config.dataset);
    }
}

void maybe_prepare_ppi_cache(const util::AppConfig& config, const std::filesystem::path& cache_root) {
    if (!config.auto_prepare_dataset || std::filesystem::exists(cache_root / "meta.cfg")) {
        return;
    }
    const std::filesystem::path project_root(config.project_root);
    const auto python = std::filesystem::exists(project_root / ".venv/bin/python")
        ? (project_root / ".venv/bin/python").string()
        : std::string("python3");
    const auto script = (project_root / "scripts/prepare_ppi.py").string();
    const auto command = python + " " + script +
        " --data-root " + (project_root / config.data_root / "ppi").string() +
        " --output-root " + (project_root / config.cache_root / "ppi").string();
    if (std::system(command.c_str()) != 0) {
        throw std::runtime_error("failed to prepare PPI cache");
    }
}

void maybe_prepare_ogbn_arxiv_cache(const util::AppConfig& config, const std::filesystem::path& cache_root) {
    if (!config.auto_prepare_dataset || std::filesystem::exists(cache_root / "meta.cfg")) {
        return;
    }
    const std::filesystem::path project_root(config.project_root);
    const auto python = std::filesystem::exists(project_root / ".venv/bin/python")
        ? (project_root / ".venv/bin/python").string()
        : std::string("python3");
    const auto script = (project_root / "scripts/prepare_ogbn_arxiv.py").string();
    const auto command = python + " " + script +
        " --project-root " + project_root.string() +
        " --data-root " + config.data_root;
    if (std::system(command.c_str()) != 0) {
        throw std::runtime_error("failed to prepare ogbn-arxiv cache");
    }
}

std::vector<std::size_t> selected_graph_ids(const GraphDataset& dataset, const util::AppConfig& config) {
    const auto batch_graphs = std::max<std::size_t>(1, config.batch_graphs);
    const auto count = std::min(batch_graphs, dataset.graph_count);
    std::vector<std::size_t> out(count, 0);
    std::iota(out.begin(), out.end(), 0);
    return out;
}

std::vector<std::size_t> bfs_pick_nodes(
    const std::unordered_map<std::size_t, std::vector<std::size_t>>& adjacency,
    std::size_t center_node,
    std::size_t target_count,
    const std::vector<std::size_t>& candidates) {
    std::vector<std::size_t> selected;
    std::unordered_set<std::size_t> visited;
    std::queue<std::size_t> queue;
    const auto start = std::find(candidates.begin(), candidates.end(), center_node) != candidates.end()
        ? center_node
        : candidates.front();
    queue.push(start);
    visited.insert(start);

    while (!queue.empty() && selected.size() < target_count) {
        const auto node = queue.front();
        queue.pop();
        selected.push_back(node);
        if (const auto it = adjacency.find(node); it != adjacency.end()) {
            for (const auto neighbor : it->second) {
                if (!visited.contains(neighbor)) {
                    visited.insert(neighbor);
                    queue.push(neighbor);
                }
            }
        }
    }

    for (const auto node : candidates) {
        if (selected.size() >= target_count) {
            break;
        }
        if (!visited.contains(node)) {
            visited.insert(node);
            selected.push_back(node);
        }
    }
    return selected;
}

void fill_local_features(
    const GraphDataset& dataset,
    const std::vector<std::size_t>& absolute_ids,
    LocalGraph* local) {
    local->features_fp.reserve(local->features_fp.size() + absolute_ids.size());
    local->features.reserve(local->features.size() + absolute_ids.size());
    local->labels.reserve(local->labels.size() + absolute_ids.size());
    for (const auto absolute_id : absolute_ids) {
        local->absolute_ids.push_back(absolute_id);
        local->features_fp.push_back(dataset.features_fp.at(absolute_id));
        local->features.push_back(dataset.features.at(absolute_id));
        local->labels.push_back(dataset.labels.at(absolute_id));
    }
}

std::uint64_t edge_key(std::size_t src, std::size_t dst, std::size_t graph_id) {
    return (static_cast<std::uint64_t>(graph_id) << 48U)
        ^ (static_cast<std::uint64_t>(src) << 24U)
        ^ static_cast<std::uint64_t>(dst);
}

}  // namespace

GraphDataset load_dataset(const util::AppConfig& config) {
    const std::filesystem::path project_root(config.project_root);
    if (config.dataset == "toy") {
        return load_cached_dataset(project_root / "data/toy");
    }

    const auto cache_root = project_root / config.cache_root / config.dataset;
    if (config.dataset == "ppi") {
        maybe_prepare_ppi_cache(config, cache_root);
    } else if (config.dataset == "ogbn_arxiv") {
        maybe_prepare_ogbn_arxiv_cache(config, cache_root);
    } else {
        maybe_prepare_planetoid_cache(config, cache_root);
    }
    return load_cached_dataset(cache_root);
}

LocalGraph normalize_graph_input(const GraphDataset& dataset, const util::AppConfig& config) {
    if (dataset.num_nodes == 0) {
        throw std::runtime_error("cannot normalize an empty dataset");
    }

    LocalGraph local;
    local.name = dataset.name + "_normalized";
    local.num_features = dataset.num_features;
    local.num_classes = dataset.num_classes;
    local.task_type = config.task_type.empty() ? dataset.task_type : config.task_type;
    local.report_unit = config.report_unit.empty() ? dataset.report_unit : config.report_unit;
    local.batching_rule = config.batching_rule;
    local.subgraph_rule = config.subgraph_rule;
    local.self_loop_rule = config.self_loop_rule;
    local.edge_sort_rule = config.edge_sort_rule;
    local.chunking_rule = config.chunking_rule;
    local.graph_count = 0;
    local.public_input.N_total = dataset.num_nodes;

    const auto graph_ids = selected_graph_ids(dataset, config);
    const auto total_batch_graphs = graph_ids.empty() ? 1 : graph_ids.size();

    std::unordered_map<std::size_t, std::vector<std::size_t>> adjacency;
    adjacency.reserve(dataset.num_nodes);
    for (const auto& edge : dataset.edges) {
        adjacency[edge.src].push_back(edge.dst);
        adjacency[edge.dst].push_back(edge.src);
    }

    std::vector<std::size_t> batch_node_ptr = {0};
    std::vector<std::size_t> batch_edge_ptr = {0};
    std::vector<Edge> batch_edges;
    batch_edges.reserve(dataset.edges.size() + dataset.num_nodes);

    for (std::size_t batch_graph_id = 0; batch_graph_id < total_batch_graphs; ++batch_graph_id) {
        const auto dataset_graph_id = graph_ids[batch_graph_id];
        const auto node_begin = dataset.node_ptr.at(dataset_graph_id);
        const auto node_end = dataset.node_ptr.at(dataset_graph_id + 1);
        std::vector<std::size_t> graph_nodes(node_end - node_begin, 0);
        std::iota(graph_nodes.begin(), graph_nodes.end(), node_begin);

        std::vector<std::size_t> selected_nodes;
        if (total_batch_graphs == 1
            && config.subgraph_rule != "whole_graph"
            && config.local_nodes < graph_nodes.size()) {
            selected_nodes = bfs_pick_nodes(adjacency, config.center_node, config.local_nodes, graph_nodes);
        } else {
            selected_nodes = graph_nodes;
        }
        std::sort(selected_nodes.begin(), selected_nodes.end());
        fill_local_features(dataset, selected_nodes, &local);
        const auto local_offset = batch_node_ptr.back();
        batch_node_ptr.push_back(local_offset + selected_nodes.size());

        std::unordered_map<std::size_t, std::size_t> global_to_local;
        global_to_local.reserve(selected_nodes.size());
        for (std::size_t i = 0; i < selected_nodes.size(); ++i) {
            global_to_local[selected_nodes[i]] = local_offset + i;
        }

        std::vector<Edge> graph_edges;
        std::unordered_set<std::uint64_t> seen;
        std::size_t stable_index = 0;
        for (const auto& edge : dataset.edges) {
            if (edge.graph_id != dataset_graph_id) {
                continue;
            }
            if (!global_to_local.contains(edge.src) || !global_to_local.contains(edge.dst)) {
                continue;
            }
            auto append_edge = [&](std::size_t src_global, std::size_t dst_global) {
                const auto src_local = global_to_local.at(src_global);
                const auto dst_local = global_to_local.at(dst_global);
                const auto key = edge_key(src_local, dst_local, batch_graph_id);
                if (config.deduplicate_edges && !seen.insert(key).second) {
                    return;
                }
                graph_edges.push_back(Edge{
                    .src = src_local,
                    .dst = dst_local,
                    .graph_id = batch_graph_id,
                    .stable_index = stable_index++,
                });
            };
            append_edge(edge.src, edge.dst);
            if (config.symmetrize_edges && edge.src != edge.dst) {
                append_edge(edge.dst, edge.src);
            }
        }
        for (std::size_t i = 0; i < selected_nodes.size(); ++i) {
            const auto node_local = local_offset + i;
            const auto key = edge_key(node_local, node_local, batch_graph_id);
            if (!seen.insert(key).second) {
                continue;
            }
            graph_edges.push_back(Edge{
                .src = node_local,
                .dst = node_local,
                .graph_id = batch_graph_id,
                .stable_index = stable_index++,
            });
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

    local.num_nodes = local.absolute_ids.size();
    local.edges = std::move(batch_edges);
    local.node_ptr = std::move(batch_node_ptr);
    local.edge_ptr = std::move(batch_edge_ptr);
    local.graph_count = local.node_ptr.size() - 1;
    local.public_input.N_total = dataset.num_nodes;
    local.public_input.N = local.num_nodes;
    local.public_input.E = local.edges.size();
    local.public_input.G_batch = local.graph_count;
    local.public_input.I = local.absolute_ids;
    local.public_input.node_ptr = local.node_ptr;
    local.public_input.edge_ptr = local.edge_ptr;
    local.public_input.src.reserve(local.edges.size());
    local.public_input.dst.reserve(local.edges.size());
    for (const auto& edge : local.edges) {
        local.public_input.src.push_back(edge.src);
        local.public_input.dst.push_back(edge.dst);
    }
    local.public_input.src.shrink_to_fit();
    local.public_input.dst.shrink_to_fit();
    return local;
}

LocalGraph extract_local_subgraph(const GraphDataset& dataset, std::size_t center_node, std::size_t local_nodes) {
    util::AppConfig config;
    config.dataset = dataset.name;
    config.task_type = dataset.task_type;
    config.report_unit = dataset.report_unit;
    config.subgraph_rule = local_nodes >= dataset.num_nodes ? "whole_graph" : "sampled_subgraph";
    config.local_nodes = local_nodes;
    config.center_node = center_node;
    config.batch_graphs = 1;
    config.symmetrize_edges = false;
    config.deduplicate_edges = true;
    return normalize_graph_input(dataset, config);
}

}  // namespace gatzk::data
