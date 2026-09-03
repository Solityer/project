#include "gat_ezkl/dataset.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <set>
#include <tuple>
#include <unordered_map>

namespace gat_ezkl 
{
    namespace 
    {
        std::string trim(std::string value) 
        {
            value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
                return !std::isspace(ch);
            }));
            value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
            }).base(), value.end());
            return value;
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
                out[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
            }
            return out;
        }

        struct NpyArray {
            std::vector<std::size_t> shape;
            std::vector<double> values;
            bool source_float32 = false;
        };

        NpyArray load_npy_double(const std::filesystem::path& path) {
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
            } else if (major == 2 || major == 3) {
                input.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
            } else {
                throw std::runtime_error("unsupported npy version: " + path.string());
            }
            std::string header(header_len, '\0');
            input.read(header.data(), static_cast<std::streamsize>(header.size()));
            const auto descr_pos = header.find("'descr'");
            const auto descr_quote = header.find('\'', header.find(':', descr_pos));
            const auto descr_end = header.find('\'', descr_quote + 1);
            if (descr_pos == std::string::npos || descr_quote == std::string::npos || descr_end == std::string::npos) {
                throw std::runtime_error("missing npy dtype descriptor: " + path.string());
            }
            const auto descr = header.substr(descr_quote + 1, descr_end - descr_quote - 1);
            const bool is_float64 = descr == "<f8" || descr == "|f8";
            const bool is_float32 = descr == "<f4" || descr == "|f4";
            if (!is_float64 && !is_float32) {
                throw std::runtime_error("expected float32/float64 npy payload: " + path.string());
            }
            const auto fortran_pos = header.find("fortran_order");
            if (fortran_pos == std::string::npos || header.find("False", fortran_pos) == std::string::npos) {
                throw std::runtime_error("fortran-order npy arrays are not supported: " + path.string());
            }
            const auto shape_pos = header.find('(', header.find("shape"));
            const auto shape_end = header.find(')', shape_pos);
            if (shape_pos == std::string::npos || shape_end == std::string::npos) {
                throw std::runtime_error("missing npy shape: " + path.string());
            }
            NpyArray out;
            out.source_float32 = is_float32;
            std::stringstream shape_stream(header.substr(shape_pos + 1, shape_end - shape_pos - 1));
            std::string token;
            while (std::getline(shape_stream, token, ',')) {
                token = trim(token);
                if (!token.empty()) {
                    out.shape.push_back(static_cast<std::size_t>(std::stoull(token)));
                }
            }
            std::size_t count = 1;
            for (const auto dim : out.shape) {
                if (dim != 0 && count > std::numeric_limits<std::size_t>::max() / dim) {
                    throw std::runtime_error("npy shape product overflow: " + path.string());
                }
                count *= dim;
            }
            out.values.resize(count, 0.0);
            if (is_float64) {
                input.read(reinterpret_cast<char*>(out.values.data()), static_cast<std::streamsize>(count * sizeof(double)));
            } else {
                std::vector<float> tmp(count, 0.0f);
                input.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(count * sizeof(float)));
                for (std::size_t i = 0; i < count; ++i) {
                    out.values[i] = static_cast<double>(tmp[i]);
                }
            }
            if (!input || std::any_of(out.values.begin(), out.values.end(),
                                     [](double value) { return !std::isfinite(value); })) {
                throw std::runtime_error("npy payload is truncated or non-finite: " + path.string());
            }
            return out;
        }

        // 行归一化：每行除以该行总和（跳过全零行）
        std::vector<std::vector<double>> row_normalize(std::vector<std::vector<double>> values) {
            for (auto& row : values) {
                double sum = 0.0;
                for (const auto value : row) {
                    if (!std::isfinite(value)) {
                        throw std::runtime_error("dataset feature contains a non-finite value");
                    }
                    sum += value;
                }
                if (!std::isfinite(sum)) {
                    throw std::runtime_error("dataset feature row sum is non-finite");
                }
                if (sum == 0.0) {
                    continue;
                }
                for (auto& value : row) {
                    value /= sum;
                    if (!std::isfinite(value)) {
                        throw std::runtime_error("dataset row normalization produced a non-finite value");
                    }
                }
            }
            return values;
        }

        // 加载特征矩阵：优先稀疏文本格式 features.txt，否则 features.npy
        std::vector<std::vector<double>> load_features_matrix(
            const std::filesystem::path& root,
            std::size_t rows,
            std::size_t cols,
            bool* source_float32) {
            if (source_float32 == nullptr) {
                throw std::runtime_error("feature source dtype output is null");
            }
            *source_float32 = false;
            if (std::filesystem::exists(root / "features.txt")) {
                std::vector<std::vector<double>> raw(rows, std::vector<double>(cols, 0.0));
                std::ifstream input(root / "features.txt");
                if (!input) {
                    throw std::runtime_error("failed to open features.txt");
                }
                std::size_t file_rows = 0;
                std::size_t file_cols = 0;
                input >> file_rows >> file_cols;
                if (file_rows != rows || file_cols != cols) {
                    throw std::runtime_error("features.txt shape mismatch");
                }
                std::string line;
                std::getline(input, line);
                for (std::size_t row = 0; row < rows; ++row) {
                    if (!std::getline(input, line)) {
                        throw std::runtime_error("features.txt has fewer rows than declared");
                    }
                    std::istringstream iss(line);
                    std::string entry;
                    std::set<std::size_t> seen_columns;
                    while (iss >> entry) {
                        const auto colon = entry.find(':');
                        if (colon == std::string::npos) {
                            throw std::runtime_error("malformed sparse feature entry");
                        }
                        const auto col = static_cast<std::size_t>(std::stoull(entry.substr(0, colon)));
                        if (col >= cols) {
                            throw std::runtime_error("feature column out of range");
                        }
                        if (!seen_columns.insert(col).second) {
                            throw std::runtime_error("duplicate sparse feature column");
                        }
                        std::size_t consumed = 0;
                        const auto parsed = std::stod(entry.substr(colon + 1), &consumed);
                        if (consumed != entry.size() - colon - 1 || !std::isfinite(parsed)) {
                            throw std::runtime_error("non-finite or malformed sparse feature value");
                        }
                        raw[row][col] = parsed;
                    }
                }
                return raw;
            }
            const auto npy = load_npy_double(root / "features.npy");
            *source_float32 = npy.source_float32;
            if (npy.shape.size() != 2 || npy.shape[0] != rows || npy.shape[1] != cols) {
                throw std::runtime_error("features.npy shape mismatch");
            }
            std::vector<std::vector<double>> out(rows, std::vector<double>(cols, 0.0));
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t col = 0; col < cols; ++col) {
                    out[row][col] = npy.values[row * cols + col];
                }
            }
            return out;
        }

        // Load single-label datasets.  Formal runs do not permit a missing
        // label file to silently become an all-zero task.
        std::vector<int> load_labels(const std::filesystem::path& root, std::size_t rows) {
            std::vector<int> labels;
            std::ifstream input(root / "labels.txt");
            if (!input) {
                throw std::runtime_error("formal dataset is missing labels.txt");
            }
            int label = 0;
            while (input >> label) {
                labels.push_back(label);
            }
            if (!input.eof()) {
                throw std::runtime_error("malformed labels.txt");
            }
            if (labels.size() != rows) {
                throw std::runtime_error("label count mismatch");
            }
            return labels;
        }

        std::vector<std::vector<int>> load_multi_labels(
            const std::filesystem::path& root, std::size_t rows, std::size_t cols) {
            std::ifstream input(root / "labels_multi.txt");
            if (!input) {
                throw std::runtime_error("formal multi-label dataset is missing labels_multi.txt");
            }
            std::size_t file_rows = 0;
            std::size_t file_cols = 0;
            if (!(input >> file_rows >> file_cols) || file_rows != rows || file_cols != cols) {
                throw std::runtime_error("labels_multi.txt shape mismatch");
            }
            std::vector<std::vector<int>> labels(rows, std::vector<int>(cols, 0));
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t col = 0; col < cols; ++col) {
                    if (!(input >> labels[row][col])
                        || (labels[row][col] != 0 && labels[row][col] != 1)) {
                        throw std::runtime_error("labels_multi.txt contains a non-binary entry");
                    }
                }
            }
            std::string extra;
            if (input >> extra) {
                throw std::runtime_error("labels_multi.txt contains trailing entries");
            }
            return labels;
        }

        // 加载纯数字序列文件（每行一个整数）
        std::vector<std::size_t> load_size_vector(const std::filesystem::path& path) {
            std::ifstream input(path);
            if (!input) {
                throw std::runtime_error("failed to open vector file: " + path.string());
            }
            std::vector<std::size_t> out;
            std::string token;
            while (input >> token) {
                std::size_t consumed = 0;
                const auto value = std::stoull(token, &consumed);
                if (consumed != token.size()
                    || value > std::numeric_limits<std::size_t>::max()) {
                    throw std::runtime_error("malformed size vector entry: " + path.string());
                }
                out.push_back(static_cast<std::size_t>(value));
            }
            if (!input.eof()) throw std::runtime_error("malformed size vector: " + path.string());
            return out;
        }

    } // namespace

    // 从项目缓存加载数据集
    GraphDataset load_dataset_from_cache(
        const std::filesystem::path& project_root,
        const AppConfig& config) {
        const auto cache_root = project_root / config.cache_root / config.dataset;
        const auto meta = read_meta(cache_root / "meta.cfg");

        GraphDataset dataset;
        const auto metadata_name = meta.at("name");
        const auto expected_display_name = config.dataset == "ogbn_arxiv"
            ? "ogbn-arxiv" : config.dataset;
        if (metadata_name != config.dataset && metadata_name != expected_display_name) {
            throw std::runtime_error("dataset meta name does not match canonical dataset identity");
        }
        dataset.name = config.dataset;
        dataset.input_preprocess_id = config.input_preprocess_id;
        dataset.num_nodes = static_cast<std::size_t>(std::stoull(meta.at("num_nodes")));
        dataset.num_features = static_cast<std::size_t>(std::stoull(meta.at("num_features")));
        dataset.num_classes = static_cast<std::size_t>(std::stoull(meta.at("num_classes")));
        const bool has_declared_graph_count = meta.count("graph_count") != 0;
        if (has_declared_graph_count) {
            dataset.graph_count = static_cast<std::size_t>(std::stoull(meta.at("graph_count")));
        }
        if (meta.count("task_type")) {
            dataset.task_type = meta.at("task_type");
        }
        if (meta.count("report_unit")) {
            dataset.report_unit = meta.at("report_unit");
        }
        if (meta.count("label_format")) {
            dataset.label_format = meta.at("label_format");
        }
        bool source_float32 = false;
        auto features = load_features_matrix(
            cache_root, dataset.num_nodes, dataset.num_features, &source_float32);
        if (config.dataset == "ogbn_arxiv") {
            if (config.input_preprocess_id != "ogbn_arxiv_raw_float32" || !source_float32) {
                throw std::runtime_error(
                    "OGBN-Arxiv formal input must use the original float32 feature payload");
            }
            dataset.features_fp = std::move(features);
        } else {
            if (config.input_preprocess_id != "row_normalize") {
                throw std::runtime_error("non-OGBN formal input requires row_normalize");
            }
            dataset.features_fp = row_normalize(std::move(features));
        }
        if (dataset.label_format == "multi_hot") {
            dataset.labels_multi = load_multi_labels(
                cache_root, dataset.num_nodes, dataset.num_classes);
            dataset.labels.assign(dataset.num_nodes, 0);
        } else {
            dataset.labels = load_labels(cache_root, dataset.num_nodes);
        }

        if (std::filesystem::exists(cache_root / "node_ptr.txt")) {
            dataset.node_ptr = load_size_vector(cache_root / "node_ptr.txt");
        } else {
            dataset.node_ptr = {0, dataset.num_nodes};
        }
        if (dataset.node_ptr.size() < 2 || dataset.node_ptr.front() != 0 || dataset.node_ptr.back() != dataset.num_nodes) {
            throw std::runtime_error("invalid node_ptr for dataset cache");
        }
        const auto declared_graph_count = dataset.graph_count;
        if (dataset.node_ptr.size() < 2 || dataset.node_ptr.front() != 0
            || dataset.node_ptr.back() != dataset.num_nodes
            || std::any_of(dataset.node_ptr.begin(), dataset.node_ptr.end() - 1,
                           [&](std::size_t value) { return value >= dataset.num_nodes && value != 0; })
            || std::any_of(dataset.node_ptr.begin(), dataset.node_ptr.end() - 1,
                           [](std::size_t value) { return value == std::numeric_limits<std::size_t>::max(); })) {
            throw std::runtime_error("invalid node_ptr for dataset cache");
        }
        if (std::adjacent_find(dataset.node_ptr.begin(), dataset.node_ptr.end(),
                               std::greater_equal<std::size_t>()) != dataset.node_ptr.end()) {
            throw std::runtime_error("node_ptr must be strictly increasing");
        }
        dataset.graph_count = dataset.node_ptr.size() - 1;
        if (has_declared_graph_count && declared_graph_count != dataset.graph_count) {
            throw std::runtime_error("meta graph_count disagrees with node_ptr");
        }

        std::ifstream edge_input(cache_root / "edges.txt");
        if (!edge_input) {
            throw std::runtime_error("failed to open edges.txt");
        }
        std::string line;
        std::size_t stable_index = 0;
        while (std::getline(edge_input, line)) {
            line = trim(line);
            if (line.empty()) {
                continue;
            }
            std::istringstream iss(line);
            std::vector<std::size_t> parts;
            std::string token;
            while (iss >> token) {
                std::size_t consumed = 0;
                const auto parsed = std::stoull(token, &consumed);
                if (consumed != token.size()
                    || parsed > std::numeric_limits<std::size_t>::max()) {
                    throw std::runtime_error("malformed edge line in edges.txt");
                }
                parts.push_back(static_cast<std::size_t>(parsed));
            }
            if (!iss.eof()) throw std::runtime_error("malformed edge line in edges.txt");
            Edge edge;
            edge.stable_index = stable_index++;
            if (parts.size() == 2) {
                edge.src = parts[0];
                edge.dst = parts[1];
                edge.graph_id = 0;
            } else if (parts.size() == 3) {
                edge.graph_id = parts[0];
                edge.src = parts[1];
                edge.dst = parts[2];
            } else {
                throw std::runtime_error("malformed edge line in edges.txt");
            }
            if (dataset.graph_count > 1 && parts.size() != 3) {
                throw std::runtime_error("multi-graph edges must carry an explicit graph_id");
            }
            if (edge.graph_id >= dataset.graph_count || edge.src >= dataset.num_nodes
                || edge.dst >= dataset.num_nodes) {
                throw std::runtime_error("edge graph_id out of range");
            }
            if (edge.src < dataset.node_ptr[edge.graph_id]
                || edge.src >= dataset.node_ptr[edge.graph_id + 1]
                || edge.dst < dataset.node_ptr[edge.graph_id]
                || edge.dst >= dataset.node_ptr[edge.graph_id + 1]) {
                throw std::runtime_error("edge endpoint crosses graph block");
            }
            dataset.edges.push_back(edge);
        }
        dataset.edge_ptr.assign(dataset.graph_count + 1, 0);
        for (const auto& edge : dataset.edges) {
            ++dataset.edge_ptr[edge.graph_id + 1];
        }
        std::partial_sum(dataset.edge_ptr.begin(), dataset.edge_ptr.end(), dataset.edge_ptr.begin());
        return dataset;
    }

    // 将原始数据集归一化：节点重编号、边去重/对称化、自环添加、CSR 构建
    NormalizedGraph normalize_full_graph(const GraphDataset& dataset, const AppConfig& config) {
        if (dataset.num_nodes == 0) {
            throw std::runtime_error("cannot normalize empty dataset");
        }
        if (dataset.input_preprocess_id != config.input_preprocess_id) {
            throw std::runtime_error("dataset preprocessing identity disagrees with config");
        }
        if (config.dataset == "ppi" && dataset.label_format != "multi_hot") {
            throw std::runtime_error("PPI formal task requires its canonical multi-hot labels");
        }
        if (config.dataset != "ppi" && dataset.label_format == "multi_hot") {
            throw std::runtime_error("multi-hot labels are only valid for the canonical PPI task");
        }
        if (dataset.graph_count == 0 || dataset.node_ptr.size() != dataset.graph_count + 1
            || dataset.node_ptr.front() != 0 || dataset.node_ptr.back() != dataset.num_nodes
            || std::adjacent_find(dataset.node_ptr.begin(), dataset.node_ptr.end(),
                                  std::greater_equal<std::size_t>()) != dataset.node_ptr.end()) {
            throw std::runtime_error("dataset node_ptr is not a strict canonical partition");
        }
        if (dataset.features_fp.size() != dataset.num_nodes
            || (dataset.label_format == "multi_hot"
                ? dataset.labels_multi.size() != dataset.num_nodes
                : dataset.labels.size() != dataset.num_nodes)) {
            throw std::runtime_error("dataset feature/label rows do not match num_nodes");
        }
        for (const auto& row : dataset.features_fp) {
            if (row.size() != dataset.num_features
                || std::any_of(row.begin(), row.end(), [](double value) { return !std::isfinite(value); })) {
                throw std::runtime_error("dataset feature shape or finiteness mismatch");
            }
        }
        if (dataset.label_format == "multi_hot") {
            for (const auto& row : dataset.labels_multi) {
                if (row.size() != dataset.num_classes
                    || std::any_of(row.begin(), row.end(), [](int label) {
                        return label != 0 && label != 1;
                    })) {
                    throw std::runtime_error("dataset multi-label shape/value mismatch");
                }
            }
        } else {
            for (const auto label : dataset.labels) {
                if (label < 0 || static_cast<std::size_t>(label) >= dataset.num_classes) {
                    throw std::runtime_error("dataset label is outside the configured class range");
                }
            }
        }
        if (config.batching_rule == "multi_graph_batch") {
            if (config.batch_graphs != dataset.graph_count) {
                throw std::runtime_error("PPI/full multi-graph batch must not silently clip batch_graphs");
            }
        } else if (dataset.graph_count != 1) {
            throw std::runtime_error("whole_graph_single expected exactly one graph");
        }

        std::vector<std::vector<Edge>> edges_by_graph(dataset.graph_count);
        std::set<std::tuple<std::size_t, std::size_t, std::size_t>> input_edges;
        for (const auto& edge : dataset.edges) {
            if (edge.graph_id >= dataset.graph_count || edge.src >= dataset.num_nodes
                || edge.dst >= dataset.num_nodes
                || edge.src < dataset.node_ptr[edge.graph_id]
                || edge.src >= dataset.node_ptr[edge.graph_id + 1]
                || edge.dst < dataset.node_ptr[edge.graph_id]
                || edge.dst >= dataset.node_ptr[edge.graph_id + 1]) {
                throw std::runtime_error("dataset edge is out of range or crosses graph block");
            }
            if (!input_edges.emplace(edge.graph_id, edge.src, edge.dst).second) {
                throw std::runtime_error("dataset contains a duplicate directed edge");
            }
            edges_by_graph.at(edge.graph_id).push_back(edge);
        }

        NormalizedGraph local;
        local.name = dataset.name + "_full_normalized";
        local.input_preprocess_id = dataset.input_preprocess_id;
        local.input_preprocess_id = dataset.input_preprocess_id;
        local.n_total = dataset.num_nodes;
        local.feature_dim = dataset.num_features;
        local.class_count = dataset.num_classes;
        local.graph_count = dataset.graph_count;
        local.node_ptr = {0};
        local.edge_ptr = {0};

        for (std::size_t graph_id = 0; graph_id < dataset.graph_count; ++graph_id) {
            const auto node_begin = dataset.node_ptr.at(graph_id);
            const auto node_end = dataset.node_ptr.at(graph_id + 1);
            const auto local_offset = local.node_ptr.back();
            std::unordered_map<std::size_t, std::size_t> global_to_local;
            global_to_local.reserve(node_end - node_begin);
            for (std::size_t node = node_begin; node < node_end; ++node) {
                const auto local_id = local_offset + (node - node_begin);
                global_to_local[node] = local_id;
                local.absolute_ids.push_back(node);
                local.features_fp.push_back(dataset.features_fp.at(node));
                local.labels.push_back(dataset.labels.at(node));
            }
            local.node_ptr.push_back(local.absolute_ids.size());

            std::vector<Edge> graph_edges;
            std::set<std::pair<std::size_t, std::size_t>> seen;
            std::size_t stable_index = 0;
            auto append_edge = [&](std::size_t src_global, std::size_t dst_global) {
                const auto src_it = global_to_local.find(src_global);
                const auto dst_it = global_to_local.find(dst_global);
                if (src_it == global_to_local.end() || dst_it == global_to_local.end()) {
                    throw std::runtime_error("edge endpoint is not in its declared graph");
                }
                const auto src_local = src_it->second;
                const auto dst_local = dst_it->second;
                if (!seen.emplace(src_local, dst_local).second) {
                    throw std::runtime_error("normalized graph contains a duplicate directed edge");
                }
                graph_edges.push_back(Edge{src_local, dst_local, graph_id, stable_index++});
            };
            for (const auto& edge : edges_by_graph.at(graph_id)) {
                append_edge(edge.src, edge.dst);
                if (config.symmetrize_edges && edge.src != edge.dst) {
                    append_edge(edge.dst, edge.src);
                }
            }
            if (config.self_loop_rule != "per_node") {
                throw std::runtime_error("self_loop_rule must be per_node");
            }
            for (std::size_t node = node_begin; node < node_end; ++node) {
                append_edge(node, node);
            }
            std::stable_sort(graph_edges.begin(), graph_edges.end(), [](const Edge& lhs, const Edge& rhs) {
                if (lhs.graph_id != rhs.graph_id) return lhs.graph_id < rhs.graph_id;
                if (lhs.dst != rhs.dst) return lhs.dst < rhs.dst;
                if (lhs.src != rhs.src) return lhs.src < rhs.src;
                return lhs.stable_index < rhs.stable_index;
            });
            local.edge_ptr.push_back(local.edge_ptr.back() + graph_edges.size());
            local.edges.insert(local.edges.end(), graph_edges.begin(), graph_edges.end());
        }

        local.node_count = local.absolute_ids.size();
        local.edge_count = local.edges.size();
        local.src.reserve(local.edge_count);
        local.dst.reserve(local.edge_count);
        for (const auto& edge : local.edges) {
            local.src.push_back(edge.src);
            local.dst.push_back(edge.dst);
        }
        return local;
    }

    // ZK 公共输入元素数 = 特征 + 边 src/dst + CSR 指针，全部公开给电路
    std::size_t public_input_element_count(const NormalizedGraph& graph) {
        return graph.node_count * graph.feature_dim
            + graph.src.size()
            + graph.dst.size()
            + graph.node_ptr.size()
            + graph.edge_ptr.size();
    }

    // ZK 公共输出元素数 = 节点数 × 类别数（通常是 softmax 前的分数）
    std::size_t public_output_element_count(const NormalizedGraph& graph) {
        return graph.node_count * graph.class_count;
    }
}
