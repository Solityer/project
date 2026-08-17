#pragma once

// 五个正式数据集 cache、图结构与公开拓扑加载。

// 单个 official dataset 的公开运行配置；CPU/GPU 路径共用同一份 spec。
struct DatasetRunSpec {
    std::string dataset;
    std::string cache_name;
    std::string checkpoint_name;
    std::string config_name;
    std::string task_type;
    std::string report_unit;
    std::string batching_rule;
    std::size_t batch_graphs = 1;
    std::size_t range_bits = 12;
};

// 公开边记录；stable_index 用于协议要求的图内 dst 稳定排序。
struct Edge {
    std::size_t src = 0;
    std::size_t dst = 0;
    std::size_t graph_id = 0;
    std::size_t stable_index = 0;
};

// 从主项目 cache 读取的全量图数据；只作为 witness/public topology 生成输入。
struct GraphDataset {
    std::string name;
    std::size_t num_nodes = 0;
    std::size_t num_features = 0;
    std::size_t num_classes = 0;
    std::size_t graph_count = 1;
    std::string task_type;
    std::string report_unit;
    std::vector<double> features; // row-major, row-normalized
    std::vector<int> labels;
    std::vector<Edge> edges;
    std::vector<std::size_t> node_ptr;
    std::vector<std::size_t> edge_ptr;
};

// full-chain 证明实例使用的局部图；当前 official 要求覆盖 full dataset。
struct LocalGraph {
    std::size_t num_nodes = 0;
    std::size_t num_features = 0;
    std::size_t num_classes = 0;
    std::size_t graph_count = 1;
    std::vector<double> features;
    std::vector<int> labels;
    std::vector<Edge> edges;
    std::vector<std::size_t> absolute_ids;
    std::vector<std::size_t> node_ptr;
    std::vector<std::size_t> edge_ptr;
};

// 读取 cache meta.cfg；输入为主项目 cache 路径，输出公开 dataset 元数据。
std::unordered_map<std::string, std::string> read_meta(const fs::path& path) {
    std::unordered_map<std::string, std::string> out;
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open meta: " + path.string());
    }
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

// 简单 NPY 浮点数组载体；用于兼容 features.npy。
struct NpyArray {
    std::vector<std::size_t> shape;
    std::vector<double> values;
};

// 读取 f4/f8 C-order NPY；输出 double 向量，不写任何 artifact。
NpyArray load_npy_double(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open npy: " + path.string());
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
    const auto descr = header.substr(descr_quote + 1, descr_end - descr_quote - 1);
    const bool is_float64 = descr == "<f8" || descr == "|f8";
    const bool is_float32 = descr == "<f4" || descr == "|f4";
    if (!is_float64 && !is_float32) {
        throw std::runtime_error("expected f4/f8 npy payload: " + path.string());
    }
    if (header.find("fortran_order") == std::string::npos || header.find("False") == std::string::npos) {
        throw std::runtime_error("fortran-order npy arrays are not supported: " + path.string());
    }
    const auto shape_pos = header.find('(', header.find("shape"));
    const auto shape_end = header.find(')', shape_pos);
    std::stringstream shape_stream(header.substr(shape_pos + 1, shape_end - shape_pos - 1));
    std::string token;
    NpyArray out;
    while (std::getline(shape_stream, token, ',')) {
        token = trim(token);
        if (!token.empty()) {
            out.shape.push_back(std::stoull(token));
        }
    }
    std::size_t count = 1;
    for (const auto dim : out.shape) {
        count *= dim;
    }
    out.values.resize(count, 0.0);
    if (is_float64) {
        input.read(reinterpret_cast<char*>(out.values.data()), static_cast<std::streamsize>(count * sizeof(double)));
    } else {
        std::vector<float> values(count, 0.0f);
        input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
        for (std::size_t i = 0; i < count; ++i) {
            out.values[i] = static_cast<double>(values[i]);
        }
    }
    return out;
}

// 加载并按行归一化特征；输出用于 GAT forward 和特征检索证明。
std::vector<double> load_features(const fs::path& root, std::size_t rows, std::size_t cols) {
    std::vector<double> out(rows * cols, 0.0);
    if (fs::exists(root / "features.txt")) {
        std::ifstream input(root / "features.txt");
        if (!input) {
            throw std::runtime_error("failed to open features.txt");
        }
        std::size_t file_rows = 0;
        std::size_t file_cols = 0;
        input >> file_rows >> file_cols;
        if (file_rows != rows || file_cols != cols) {
            throw std::runtime_error("feature metadata mismatch");
        }
        std::string line;
        std::getline(input, line);
        for (std::size_t row = 0; row < rows; ++row) {
            std::getline(input, line);
            std::istringstream entries(line);
            std::string entry;
            while (entries >> entry) {
                const auto colon = entry.find(':');
                if (colon == std::string::npos) {
                    continue;
                }
                const auto col = static_cast<std::size_t>(std::stoull(entry.substr(0, colon)));
                out[row * cols + col] = std::stod(entry.substr(colon + 1));
            }
        }
    } else {
        const auto array = load_npy_double(root / "features.npy");
        if (array.shape.size() != 2 || array.shape[0] != rows || array.shape[1] != cols) {
            throw std::runtime_error("features.npy shape mismatch");
        }
        out = array.values;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            sum += out[row * cols + col];
        }
        if (sum != 0.0) {
            for (std::size_t col = 0; col < cols; ++col) {
                out[row * cols + col] /= sum;
            }
        }
    }
    return out;
}

// 加载标签；标签不进入 proof_size，仅用于完整 dataset trace 的元数据一致性。
std::vector<int> load_labels(const fs::path& root, std::size_t rows) {
    std::vector<int> labels;
    std::ifstream input(root / "labels.txt");
    if (!input) {
        labels.assign(rows, 0);
        return labels;
    }
    int value = 0;
    while (input >> value) {
        labels.push_back(value);
    }
    if (labels.size() != rows) {
        throw std::runtime_error("label count mismatch in " + root.string());
    }
    return labels;
}

// 读取 node_ptr/edge_ptr 这类 size_t 文本向量；输出绑定 batch 分组公开语义。
std::vector<std::size_t> load_size_vector(const fs::path& path) {
    std::vector<std::size_t> out;
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }
    std::size_t value = 0;
    while (input >> value) {
        out.push_back(value);
    }
    return out;
}

// 从主项目 data/cache 加载全量数据集；只读上游数据，不修改主实验。
GraphDataset load_cached_dataset(const fs::path& root) {
    const auto meta = read_meta(root / "meta.cfg");
    GraphDataset dataset;
    dataset.name = meta.at("name");
    dataset.num_nodes = std::stoull(meta.at("num_nodes"));
    dataset.num_features = std::stoull(meta.at("num_features"));
    dataset.num_classes = std::stoull(meta.at("num_classes"));
    dataset.graph_count = meta.count("graph_count") ? std::stoull(meta.at("graph_count")) : 1;
    dataset.task_type = meta.count("task_type") ? meta.at("task_type") : "transductive_node_classification";
    dataset.report_unit = meta.count("report_unit") ? meta.at("report_unit") : "node";
    dataset.features = load_features(root, dataset.num_nodes, dataset.num_features);
    dataset.labels = load_labels(root, dataset.num_nodes);
    dataset.node_ptr = fs::exists(root / "node_ptr.txt") ? load_size_vector(root / "node_ptr.txt") : std::vector<std::size_t>{0, dataset.num_nodes};
    if (dataset.node_ptr.size() < 2) {
        dataset.node_ptr = {0, dataset.num_nodes};
    }
    dataset.graph_count = dataset.node_ptr.size() - 1;

    std::ifstream edges_input(root / "edges.txt");
    if (!edges_input) {
        throw std::runtime_error("failed to open edges.txt in " + root.string());
    }
    std::string line;
    std::size_t stable = 0;
    while (std::getline(edges_input, line)) {
        if (trim(line).empty()) {
            continue;
        }
        std::istringstream stream(line);
        std::vector<std::size_t> parts;
        std::size_t value = 0;
        while (stream >> value) {
            parts.push_back(value);
        }
        Edge edge;
        edge.stable_index = stable++;
        if (parts.size() == 2) {
            edge.src = parts[0];
            edge.dst = parts[1];
            edge.graph_id = 0;
        } else if (parts.size() == 3) {
            edge.graph_id = parts[0];
            edge.src = parts[1];
            edge.dst = parts[2];
        } else {
            throw std::runtime_error("malformed edge line in " + root.string());
        }
        dataset.edges.push_back(edge);
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

// 生成协议公开拓扑：选择 full graph、补自环并按 graph_id/dst 稳定排序。
LocalGraph normalize_graph(const GraphDataset& dataset, const DatasetRunSpec& spec) {
    LocalGraph local;
    local.num_features = dataset.num_features;
    local.num_classes = dataset.num_classes;
    const std::size_t selected_graphs = std::min<std::size_t>(std::max<std::size_t>(1, spec.batch_graphs), dataset.graph_count);
    local.graph_count = selected_graphs;
    local.node_ptr = {0};
    local.edge_ptr = {0};
    std::vector<Edge> all_edges;

    for (std::size_t batch_graph_id = 0; batch_graph_id < selected_graphs; ++batch_graph_id) {
        const std::size_t node_begin = dataset.node_ptr.at(batch_graph_id);
        const std::size_t node_end = dataset.node_ptr.at(batch_graph_id + 1);
        const std::size_t local_offset = local.node_ptr.back();
        std::vector<std::size_t> selected_nodes(node_end - node_begin, 0);
        std::iota(selected_nodes.begin(), selected_nodes.end(), node_begin);
        local.node_ptr.push_back(local_offset + selected_nodes.size());
        for (const auto absolute_id : selected_nodes) {
            local.absolute_ids.push_back(absolute_id);
            local.labels.push_back(dataset.labels.at(absolute_id));
            const auto begin = dataset.features.begin() + static_cast<std::ptrdiff_t>(absolute_id * dataset.num_features);
            local.features.insert(local.features.end(), begin, begin + static_cast<std::ptrdiff_t>(dataset.num_features));
        }

        std::unordered_map<std::size_t, std::size_t> global_to_local;
        global_to_local.reserve(selected_nodes.size());
        for (std::size_t i = 0; i < selected_nodes.size(); ++i) {
            global_to_local[selected_nodes[i]] = local_offset + i;
        }
        std::unordered_set<std::uint64_t> seen;
        std::vector<Edge> graph_edges;
        std::size_t stable = 0;
        for (const auto& edge : dataset.edges) {
            if (edge.graph_id != batch_graph_id) {
                continue;
            }
            if (!global_to_local.count(edge.src) || !global_to_local.count(edge.dst)) {
                continue;
            }
            const auto src = global_to_local.at(edge.src);
            const auto dst = global_to_local.at(edge.dst);
            const auto key = edge_key(src, dst, batch_graph_id);
            if (!seen.insert(key).second) {
                continue;
            }
            graph_edges.push_back(Edge{src, dst, batch_graph_id, stable++});
        }
        for (std::size_t i = 0; i < selected_nodes.size(); ++i) {
            const auto node = local_offset + i;
            const auto key = edge_key(node, node, batch_graph_id);
            if (!seen.insert(key).second) {
                continue;
            }
            graph_edges.push_back(Edge{node, node, batch_graph_id, stable++});
        }
        std::stable_sort(graph_edges.begin(), graph_edges.end(), [](const Edge& lhs, const Edge& rhs) {
            if (lhs.graph_id != rhs.graph_id) return lhs.graph_id < rhs.graph_id;
            if (lhs.dst != rhs.dst) return lhs.dst < rhs.dst;
            return lhs.stable_index < rhs.stable_index;
        });
        all_edges.insert(all_edges.end(), graph_edges.begin(), graph_edges.end());
        local.edge_ptr.push_back(all_edges.size());
    }
    local.num_nodes = local.absolute_ids.size();
    local.edges = std::move(all_edges);
    return local;
}
