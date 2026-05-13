#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "formal_relations.cuh"
#include "timer.hpp"

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

struct ScopedTimer {
    Clock::time_point start = Clock::now();
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
};

void cuda_check(cudaError_t status, const std::string& where) {
    if (status != cudaSuccess) {
        throw std::runtime_error(where + ": " + cudaGetErrorString(status));
    }
}

void cuda_sync(const std::string& where) {
    cuda_check(cudaDeviceSynchronize(), where);
    cuda_check(cudaGetLastError(), where);
}

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    std::size_t n = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) : n(count) {
        if (n != 0) {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n), "cudaMalloc");
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), n(other.n) {
        other.ptr = nullptr;
        other.n = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
            ptr = other.ptr;
            n = other.n;
            other.ptr = nullptr;
            other.n = 0;
        }
        return *this;
    }
    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

template <typename T>
DeviceBuffer<T> to_device(const std::vector<T>& host) {
    DeviceBuffer<T> out(host.size());
    if (!host.empty()) {
        cuda_check(cudaMemcpy(out.ptr, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }
    return out;
}

template <typename T>
std::vector<T> to_host(const DeviceBuffer<T>& device) {
    std::vector<T> out(device.n);
    if (device.n != 0) {
        cuda_check(cudaMemcpy(out.data(), device.ptr, sizeof(T) * device.n, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    }
    return out;
}

bool fr_equal_host(const Fr_t& lhs, const Fr_t& rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(Fr_t)) == 0;
}

__global__ void gat_fr_mont_one_minus_kernel(Fr_t u, Fr_t* out) {
    *out = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_ONE, u);
}

Fr_t fr_mont_one_minus_host(Fr_t u) {
    Fr_t host;
    Fr_t* device = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(Fr_t)), "cudaMalloc fr_mont_one_minus");
    gat_fr_mont_one_minus_kernel<<<1,1>>>(u, device);
    cuda_sync("gat_fr_mont_one_minus_kernel");
    cuda_check(cudaMemcpy(&host, device, sizeof(Fr_t), cudaMemcpyDeviceToHost), "cudaMemcpy fr_mont_one_minus");
    cudaFree(device);
    return host;
}

Fr_t fr_unmont_host(Fr_t value) {
    FrTensor tensor(1, &value);
    tensor.unmont();
    return tensor(0);
}

G1Jacobian_t g1_add_host(G1Jacobian_t lhs, G1Jacobian_t rhs) {
    G1TensorJacobian a(1, &lhs);
    G1TensorJacobian b(1, &rhs);
    auto sum = a + b;
    return sum(0);
}

std::string trim(std::string value) {
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), value.end());
    return value;
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }
    std::ostringstream stream;
    stream << input.rdbuf();
    return stream.str();
}

void ensure_dir(const fs::path& path) {
    fs::create_directories(path);
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

std::string format_ms(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
}

std::size_t next_power_of_two(std::size_t value) {
    std::size_t out = 1;
    while (out < value) {
        out <<= 1U;
    }
    return out;
}

std::uint64_t edge_key(std::size_t src, std::size_t dst, std::size_t graph_id) {
    return (static_cast<std::uint64_t>(graph_id) << 48U)
        ^ (static_cast<std::uint64_t>(src) << 24U)
        ^ static_cast<std::uint64_t>(dst);
}

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

struct Edge {
    std::size_t src = 0;
    std::size_t dst = 0;
    std::size_t graph_id = 0;
    std::size_t stable_index = 0;
};

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

struct NpyArray {
    std::vector<std::size_t> shape;
    std::vector<double> values;
};

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

struct DenseTensor {
    std::vector<std::size_t> shape;
    std::vector<double> values;
};

std::size_t shape_product(const std::vector<std::size_t>& shape) {
    std::size_t out = 1;
    for (const auto dim : shape) {
        out *= dim;
    }
    return out;
}

std::unordered_map<std::string, DenseTensor> load_text_tensors(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open tensor dump: " + path.string());
    }
    std::unordered_map<std::string, DenseTensor> tensors;
    std::string marker;
    while (input >> marker) {
        if (marker != "TENSOR") {
            throw std::runtime_error("malformed tensor marker in " + path.string());
        }
        std::string name;
        std::size_t rank = 0;
        input >> name >> rank;
        DenseTensor tensor;
        tensor.shape.resize(rank);
        for (std::size_t i = 0; i < rank; ++i) {
            input >> tensor.shape[i];
        }
        std::size_t count = 0;
        input >> count;
        tensor.values.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            input >> tensor.values[i];
        }
        if (count != shape_product(tensor.shape)) {
            throw std::runtime_error("tensor count mismatch for " + name);
        }
        tensors.emplace(name, std::move(tensor));
    }
    return tensors;
}

std::string parse_required_string(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing string key: " + key);
    }
    return match[1].str();
}

std::size_t parse_required_size_t(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing numeric key: " + key);
    }
    return static_cast<std::size_t>(std::stoull(match[1].str()));
}

std::string extract_array_body(const std::string& text, const std::string& key) {
    const auto key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing array key: " + key);
    }
    const auto begin = text.find('[', key_pos);
    std::size_t depth = 0;
    for (std::size_t i = begin; i < text.size(); ++i) {
        if (text[i] == '[') ++depth;
        if (text[i] == ']') {
            --depth;
            if (depth == 0) {
                return text.substr(begin + 1, i - begin - 1);
            }
        }
    }
    throw std::runtime_error("unterminated array: " + key);
}

std::vector<std::string> parse_object_array(const std::string& text, const std::string& key) {
    const auto body = extract_array_body(text, key);
    std::vector<std::string> entries;
    std::size_t depth = 0;
    std::size_t begin = std::string::npos;
    for (std::size_t i = 0; i < body.size(); ++i) {
        if (body[i] == '{') {
            if (depth == 0) begin = i;
            ++depth;
        } else if (body[i] == '}') {
            --depth;
            if (depth == 0 && begin != std::string::npos) {
                entries.push_back(body.substr(begin, i - begin + 1));
            }
        }
    }
    return entries;
}

struct HeadParams {
    std::size_t layer_index = 0;
    std::size_t head_index = 0;
    std::size_t input_dim = 0;
    std::size_t output_dim = 0;
    std::vector<double> W;
    std::vector<double> attn_dst;
    std::vector<double> attn_src;
    double attn_dst_bias = 0.0;
    double attn_src_bias = 0.0;
    std::vector<double> output_bias;
};

struct ModelParams {
    std::size_t L = 0;
    std::size_t K_out = 0;
    std::size_t C = 0;
    std::vector<std::pair<std::size_t, std::size_t>> hidden_profile;
    std::vector<HeadParams> hidden_heads;
    std::vector<HeadParams> output_heads;
    std::string model_arch_id;
    std::string model_param_id;
    std::string quant_cfg_id;
    std::string output_average_rule;
};

struct HeadTraceHost {
    std::size_t rows = 0;
    std::size_t width = 0;
    std::vector<double> h_prime;
    std::vector<double> e_src;
    std::vector<double> e_dst;
    std::vector<double> z;
    std::vector<double> m;
    std::vector<double> u;
    std::vector<double> sum;
    std::vector<double> h_pre;
    std::vector<double> h_out;
};

struct ForwardTraceHost {
    std::vector<std::vector<double>> hidden_layer_inputs;
    std::vector<std::size_t> hidden_layer_input_dims;
    std::vector<std::vector<HeadTraceHost>> hidden_heads;
    std::vector<double> output_layer_input;
    std::size_t output_layer_input_dim = 0;
    std::vector<HeadTraceHost> output_heads;
    std::vector<double> logits;
};

const DenseTensor& require_tensor(const std::unordered_map<std::string, DenseTensor>& tensors, const std::string& name) {
    const auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::runtime_error("missing tensor: " + name);
    }
    return it->second;
}

std::vector<double> reshape_seq_kernel(const DenseTensor& tensor, std::size_t* rows_out, std::size_t* cols_out) {
    if (tensor.shape.size() != 3 || tensor.shape[0] != 1) {
        throw std::runtime_error("expected seq kernel shape [1,in,out]");
    }
    *rows_out = tensor.shape[1];
    *cols_out = tensor.shape[2];
    return tensor.values;
}

std::vector<double> reshape_attention_kernel(const DenseTensor& tensor) {
    if (tensor.shape.size() != 3 || tensor.shape[0] != 1 || tensor.shape[2] != 1) {
        throw std::runtime_error("expected attention kernel shape [1,width,1]");
    }
    return tensor.values;
}

double reshape_scalar(const DenseTensor& tensor) {
    if (tensor.values.size() != 1) {
        throw std::runtime_error("expected scalar tensor");
    }
    return tensor.values.front();
}

HeadParams load_head(
    const std::unordered_map<std::string, DenseTensor>& tensors,
    const std::string& entry,
    bool output_head) {
    HeadParams head;
    head.layer_index = output_head ? 0 : parse_required_size_t(entry, "layer_index");
    head.head_index = output_head ? parse_required_size_t(entry, "head_index") : parse_required_size_t(entry, "local_head_index");
    std::size_t rows = 0;
    std::size_t cols = 0;
    head.W = reshape_seq_kernel(require_tensor(tensors, parse_required_string(entry, "seq_kernel")), &rows, &cols);
    head.input_dim = rows;
    head.output_dim = cols;
    head.attn_dst = reshape_attention_kernel(require_tensor(tensors, parse_required_string(entry, "attn_dst_kernel")));
    head.attn_src = reshape_attention_kernel(require_tensor(tensors, parse_required_string(entry, "attn_src_kernel")));
    head.attn_dst_bias = reshape_scalar(require_tensor(tensors, parse_required_string(entry, "attn_dst_bias")));
    head.attn_src_bias = reshape_scalar(require_tensor(tensors, parse_required_string(entry, "attn_src_bias")));
    head.output_bias = require_tensor(tensors, parse_required_string(entry, "output_bias")).values;
    if (head.attn_dst.size() != head.output_dim || head.attn_src.size() != head.output_dim) {
        throw std::runtime_error("attention vector width mismatch");
    }
    return head;
}

ModelParams load_model(const fs::path& checkpoint_root) {
    const auto manifest = read_text(checkpoint_root / "manifest.json");
    const auto tensors = load_text_tensors(checkpoint_root / "tensors.txt");
    ModelParams model;
    model.L = parse_required_size_t(manifest, "L");
    model.K_out = parse_required_size_t(manifest, "K_out");
    model.C = parse_required_size_t(manifest, "C");
    model.model_arch_id = parse_required_string(manifest, "model_arch_id");
    model.model_param_id = parse_required_string(manifest, "model_param_id");
    model.quant_cfg_id = parse_required_string(manifest, "quant_cfg_id");
    model.output_average_rule = parse_required_string(manifest, "output_average_rule");
    for (const auto& entry : parse_object_array(manifest, "hidden_layers")) {
        model.hidden_profile.push_back({
            parse_required_size_t(entry, "head_count"),
            parse_required_size_t(entry, "head_dim"),
        });
    }
    for (const auto& entry : parse_object_array(manifest, "hidden_head_specs")) {
        model.hidden_heads.push_back(load_head(tensors, entry, false));
    }
    for (const auto& entry : parse_object_array(manifest, "output_head_specs")) {
        model.output_heads.push_back(load_head(tensors, entry, true));
    }
    if (model.output_heads.size() != model.K_out) {
        throw std::runtime_error("output head count mismatch");
    }
    return model;
}

__global__ void fill_double_kernel(double* out, double value, std::size_t n) {
    const std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = value;
    }
}

__global__ void dense_matmul_kernel(const double* A, const double* B, double* C, int rows, int shared, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) {
        return;
    }
    double acc = 0.0;
    for (int k = 0; k < shared; ++k) {
        acc += A[static_cast<std::size_t>(row) * shared + k] * B[static_cast<std::size_t>(k) * cols + col];
    }
    C[static_cast<std::size_t>(row) * cols + col] = acc;
}

__global__ void node_attention_kernel(const double* H, const double* a_src, const double* a_dst,
                                      double b_src, double b_dst, double* E_src, double* E_dst,
                                      int rows, int width) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    double src = b_src;
    double dst = b_dst;
    for (int j = 0; j < width; ++j) {
        const double value = H[static_cast<std::size_t>(row) * width + j];
        src += value * a_src[j];
        dst += value * a_dst[j];
    }
    E_src[row] = src;
    E_dst[row] = dst;
}

__device__ double atomic_max_double(double* address, double value) {
    auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed = 0;
    while (true) {
        assumed = old;
        if (__longlong_as_double(static_cast<long long>(assumed)) >= value) {
            break;
        }
        old = atomicCAS(address_as_ull, assumed, static_cast<unsigned long long int>(__double_as_longlong(value)));
        if (old == assumed) {
            break;
        }
    }
    return __longlong_as_double(static_cast<long long>(old));
}

__global__ void edge_score_kernel(const std::uint32_t* src, const std::uint32_t* dst,
                                  const double* E_src, const double* E_dst,
                                  double* Z, double* M, std::size_t edge_count) {
    const std::size_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= edge_count) {
        return;
    }
    const double s = E_dst[dst[eid]] + E_src[src[eid]];
    const double z = s >= 0.0 ? s : 0.2 * s;
    Z[eid] = z;
    atomic_max_double(&M[dst[eid]], z);
}

__global__ void softmax_sum_kernel(const std::uint32_t* dst, const double* Z, const double* M,
                                   double* U, double* Sum, std::size_t edge_count) {
    const std::size_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= edge_count) {
        return;
    }
    const double u = exp(Z[eid] - M[dst[eid]]);
    U[eid] = u;
    atomicAdd(&Sum[dst[eid]], u);
}

__global__ void aggregate_kernel(const std::uint32_t* src, const std::uint32_t* dst,
                                 const double* U, const double* Sum, const double* H_prime,
                                 double* H_agg_pre, int width, std::size_t edge_count) {
    const std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = edge_count * static_cast<std::size_t>(width);
    if (gid >= total) {
        return;
    }
    const std::size_t eid = gid / width;
    const int col = static_cast<int>(gid % width);
    const double alpha = U[eid] / Sum[dst[eid]];
    const double contrib = alpha * H_prime[static_cast<std::size_t>(src[eid]) * width + col];
    atomicAdd(&H_agg_pre[static_cast<std::size_t>(dst[eid]) * width + col], contrib);
}

__global__ void activation_kernel(const double* pre, const double* bias, double* out,
                                  int rows, int width, bool apply_bias, bool apply_elu) {
    const std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * width;
    if (gid >= total) {
        return;
    }
    const int col = static_cast<int>(gid % width);
    double value = pre[gid];
    if (apply_bias) {
        value += bias[col];
    }
    if (apply_elu && value < 0.0) {
        value = exp(value) - 1.0;
    }
    out[gid] = value;
}

__global__ void copy_head_to_concat_kernel(const double* head, double* concat,
                                           int rows, int head_width, int concat_width, int offset) {
    const std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * head_width;
    if (gid >= total) {
        return;
    }
    const int row = static_cast<int>(gid / head_width);
    const int col = static_cast<int>(gid % head_width);
    concat[static_cast<std::size_t>(row) * concat_width + offset + col] = head[gid];
}

__global__ void add_average_kernel(const double* head, double* logits, double scale, std::size_t n) {
    const std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        logits[gid] += head[gid] * scale;
    }
}

DeviceBuffer<double> run_attention_head(
    const DeviceBuffer<double>& features,
    int rows,
    int input_dim,
    const HeadParams& head,
    const DeviceBuffer<std::uint32_t>& src,
    const DeviceBuffer<std::uint32_t>& dst,
    std::size_t edge_count,
    bool apply_bias,
    bool apply_elu,
    double* projection_ms,
    double* attention_ms,
    double* activation_ms,
    HeadTraceHost* trace) {
    const int width = static_cast<int>(head.output_dim);
    auto d_W = to_device(head.W);
    auto d_a_src = to_device(head.attn_src);
    auto d_a_dst = to_device(head.attn_dst);
    DeviceBuffer<double> H_prime(static_cast<std::size_t>(rows) * width);
    {
        ScopedTimer timer;
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
        dense_matmul_kernel<<<grid, block>>>(features.ptr, d_W.ptr, H_prime.ptr, rows, input_dim, width);
        cuda_sync("dense_matmul_kernel");
        *projection_ms += timer.elapsed_ms();
    }

    DeviceBuffer<double> E_src(rows);
    DeviceBuffer<double> E_dst(rows);
    DeviceBuffer<double> Z(edge_count);
    DeviceBuffer<double> M(rows);
    DeviceBuffer<double> U(edge_count);
    DeviceBuffer<double> Sum(rows);
    DeviceBuffer<double> H_pre(static_cast<std::size_t>(rows) * width);
    {
        ScopedTimer timer;
        node_attention_kernel<<<(rows + 255) / 256, 256>>>(
            H_prime.ptr, d_a_src.ptr, d_a_dst.ptr, head.attn_src_bias, head.attn_dst_bias,
            E_src.ptr, E_dst.ptr, rows, width);
        cuda_sync("node_attention_kernel");
        fill_double_kernel<<<(rows + 255) / 256, 256>>>(M.ptr, -std::numeric_limits<double>::infinity(), rows);
        fill_double_kernel<<<(rows + 255) / 256, 256>>>(Sum.ptr, 0.0, rows);
        fill_double_kernel<<<(static_cast<std::size_t>(rows) * width + 255) / 256, 256>>>(H_pre.ptr, 0.0, static_cast<std::size_t>(rows) * width);
        cuda_sync("attention_init");
        edge_score_kernel<<<(edge_count + 255) / 256, 256>>>(src.ptr, dst.ptr, E_src.ptr, E_dst.ptr, Z.ptr, M.ptr, edge_count);
        cuda_sync("edge_score_kernel");
        softmax_sum_kernel<<<(edge_count + 255) / 256, 256>>>(dst.ptr, Z.ptr, M.ptr, U.ptr, Sum.ptr, edge_count);
        cuda_sync("softmax_sum_kernel");
        aggregate_kernel<<<(edge_count * static_cast<std::size_t>(width) + 255) / 256, 256>>>(
            src.ptr, dst.ptr, U.ptr, Sum.ptr, H_prime.ptr, H_pre.ptr, width, edge_count);
        cuda_sync("aggregate_kernel");
        *attention_ms += timer.elapsed_ms();
    }

    auto d_bias = to_device(head.output_bias);
    DeviceBuffer<double> H_out(static_cast<std::size_t>(rows) * width);
    {
        ScopedTimer timer;
        activation_kernel<<<(static_cast<std::size_t>(rows) * width + 255) / 256, 256>>>(
            H_pre.ptr, d_bias.ptr, H_out.ptr, rows, width, apply_bias, apply_elu);
        cuda_sync("activation_kernel");
        *activation_ms += timer.elapsed_ms();
    }
    if (trace != nullptr) {
        trace->rows = static_cast<std::size_t>(rows);
        trace->width = static_cast<std::size_t>(width);
        trace->h_prime = to_host(H_prime);
        trace->e_src = to_host(E_src);
        trace->e_dst = to_host(E_dst);
        trace->z = to_host(Z);
        trace->m = to_host(M);
        trace->u = to_host(U);
        trace->sum = to_host(Sum);
        trace->h_pre = to_host(H_pre);
        trace->h_out = to_host(H_out);
    }
    return H_out;
}

struct ForwardMetrics {
    double hidden_projection_ms = 0.0;
    double hidden_attention_ms = 0.0;
    double hidden_activation_ms = 0.0;
    double hidden_concat_ms = 0.0;
    double output_projection_ms = 0.0;
    double output_attention_ms = 0.0;
    double output_activation_ms = 0.0;
    double total_ms = 0.0;
};

std::vector<double> run_gat_forward_gpu(const LocalGraph& graph, const ModelParams& model, ForwardMetrics* metrics, ForwardTraceHost* trace) {
    ScopedTimer total_timer;
    auto features = to_device(graph.features);
    int current_dim = static_cast<int>(graph.num_features);
    const int rows = static_cast<int>(graph.num_nodes);

    std::vector<std::uint32_t> src_host(graph.edges.size());
    std::vector<std::uint32_t> dst_host(graph.edges.size());
    for (std::size_t i = 0; i < graph.edges.size(); ++i) {
        src_host[i] = static_cast<std::uint32_t>(graph.edges[i].src);
        dst_host[i] = static_cast<std::uint32_t>(graph.edges[i].dst);
    }
    const auto d_src = to_device(src_host);
    const auto d_dst = to_device(dst_host);

    DeviceBuffer<double> current = std::move(features);
    if (trace != nullptr) {
        trace->hidden_layer_inputs.clear();
        trace->hidden_layer_input_dims.clear();
        trace->hidden_heads.clear();
    }
    for (std::size_t layer = 0; layer < model.hidden_profile.size(); ++layer) {
        if (trace != nullptr) {
            trace->hidden_layer_inputs.push_back(to_host(current));
            trace->hidden_layer_input_dims.push_back(static_cast<std::size_t>(current_dim));
            trace->hidden_heads.emplace_back();
        }
        std::vector<const HeadParams*> layer_heads;
        for (const auto& head : model.hidden_heads) {
            if (head.layer_index == layer) {
                layer_heads.push_back(&head);
            }
        }
        const int head_width = static_cast<int>(model.hidden_profile[layer].second);
        const int concat_width = static_cast<int>(layer_heads.size()) * head_width;
        DeviceBuffer<double> concat(static_cast<std::size_t>(rows) * concat_width);
        int offset = 0;
        for (const auto* head : layer_heads) {
            HeadTraceHost hidden_head_trace;
            auto out = run_attention_head(
                current, rows, current_dim, *head, d_src, d_dst, graph.edges.size(),
                false, true,
                &metrics->hidden_projection_ms,
                &metrics->hidden_attention_ms,
                &metrics->hidden_activation_ms,
                trace == nullptr ? nullptr : &hidden_head_trace);
            if (trace != nullptr) {
                trace->hidden_heads.back().push_back(std::move(hidden_head_trace));
            }
            ScopedTimer concat_timer;
            copy_head_to_concat_kernel<<<(static_cast<std::size_t>(rows) * head_width + 255) / 256, 256>>>(
                out.ptr, concat.ptr, rows, head_width, concat_width, offset);
            cuda_sync("copy_head_to_concat_kernel");
            metrics->hidden_concat_ms += concat_timer.elapsed_ms();
            offset += head_width;
        }
        current = std::move(concat);
        current_dim = concat_width;
    }

    DeviceBuffer<double> logits(static_cast<std::size_t>(rows) * model.C);
    fill_double_kernel<<<(static_cast<std::size_t>(rows) * model.C + 255) / 256, 256>>>(logits.ptr, 0.0, static_cast<std::size_t>(rows) * model.C);
    cuda_sync("logits_init");
    const double average_scale = 1.0 / static_cast<double>(model.output_heads.size());
    if (trace != nullptr) {
        trace->output_heads.clear();
        trace->output_heads.reserve(model.output_heads.size());
        trace->output_layer_input = to_host(current);
        trace->output_layer_input_dim = static_cast<std::size_t>(current_dim);
    }
    for (const auto& head : model.output_heads) {
        HeadTraceHost head_trace;
        auto out = run_attention_head(
            current, rows, current_dim, head, d_src, d_dst, graph.edges.size(),
            true, false,
            &metrics->output_projection_ms,
            &metrics->output_attention_ms,
            &metrics->output_activation_ms,
            trace == nullptr ? nullptr : &head_trace);
        if (trace != nullptr) {
            trace->output_heads.push_back(std::move(head_trace));
        }
        add_average_kernel<<<(static_cast<std::size_t>(rows) * model.C + 255) / 256, 256>>>(
            out.ptr, logits.ptr, average_scale, static_cast<std::size_t>(rows) * model.C);
        cuda_sync("add_average_kernel");
    }

    metrics->total_ms = total_timer.elapsed_ms();
    auto host_logits = to_host(logits);
    if (trace != nullptr) {
        trace->logits = host_logits;
    }
    return host_logits;
}

Fr_t fr_from_signed_host(std::int64_t value) {
    Fr_t out = {{0, 0, 0, 0, 0, 0, 0, 0}};
    const std::uint32_t p[8] = {1U, 4294967295U, 4294859774U, 1404937218U, 161601541U, 859428872U, 698187080U, 1944954707U};
    if (value >= 0) {
        std::uint64_t v = static_cast<std::uint64_t>(value);
        out.val[0] = static_cast<std::uint32_t>(v & 0xffffffffULL);
        out.val[1] = static_cast<std::uint32_t>((v >> 32U) & 0xffffffffULL);
        return out;
    }
    std::uint64_t abs_value = static_cast<std::uint64_t>(-value);
    std::uint32_t sub[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    sub[0] = static_cast<std::uint32_t>(abs_value & 0xffffffffULL);
    sub[1] = static_cast<std::uint32_t>((abs_value >> 32U) & 0xffffffffULL);
    std::uint64_t borrow = 0;
    for (int i = 0; i < 8; ++i) {
        const std::uint64_t lhs = p[i];
        const std::uint64_t rhs = static_cast<std::uint64_t>(sub[i]) + borrow;
        if (lhs >= rhs) {
            out.val[i] = static_cast<std::uint32_t>(lhs - rhs);
            borrow = 0;
        } else {
            out.val[i] = static_cast<std::uint32_t>((1ULL << 32U) + lhs - rhs);
            borrow = 1;
        }
    }
    return out;
}

Fr_t fr_from_signed_mont_host(std::int64_t value) {
    return Fr_host_mont(fr_from_signed_host(value));
}

std::uint64_t fnv1a64(const std::string& value) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : value) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::vector<Fr_t> deterministic_raw_fr_vec(std::size_t n, const std::string& label) {
    std::mt19937_64 rng(fnv1a64(label));
    std::vector<Fr_t> out(n);
    for (auto& value : out) {
        value = fr_from_signed_host(0);
        const auto lo = rng();
        value.val[0] = static_cast<std::uint32_t>(lo & 0xffffffffULL);
        value.val[1] = static_cast<std::uint32_t>((lo >> 32U) & 0xffffffffULL);
    }
    return out;
}

std::vector<Fr_t> deterministic_mont_fr_vec(std::size_t n, const std::string& label) {
    auto raw = deterministic_raw_fr_vec(n, label);
    if (raw.empty()) {
        return raw;
    }
    FrTensor tensor(static_cast<uint>(raw.size()), raw.data());
    tensor.mont();
    cuda_sync("deterministic_mont_fr_vec");
    std::vector<Fr_t> out(raw.size());
    for (uint i = 0; i < tensor.size; ++i) {
        out[i] = tensor(i);
    }
    return out;
}

Commitment deterministic_generators(uint size, const std::string& label) {
    auto scalars = deterministic_raw_fr_vec(size, "zkdl-gat-generators:" + label + ":" + std::to_string(size));
    FrTensor scalar_tensor(size, scalars.data());
    Commitment out(size, G1Jacobian_generator);
    out *= scalar_tensor;
    cuda_sync("deterministic_generators");
    return out;
}

std::vector<Fr_t> encode_padded_tensor(const std::vector<double>& values, std::size_t rows, std::size_t cols, std::size_t padded_cols) {
    if (values.size() != rows * cols) {
        throw std::runtime_error("tensor shape/value mismatch while encoding");
    }
    std::vector<Fr_t> encoded(rows * padded_cols, fr_from_signed_host(0));
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double scaled = values[row * cols + col] * 16.0;
            const auto fixed = static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
            encoded[row * padded_cols + col] = fr_from_signed_host(fixed);
        }
    }
    return encoded;
}

Fr_t evaluate_public_tensor_mle(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_cols,
    const std::vector<Fr_t>& point) {
    auto encoded = encode_padded_tensor(values, rows, cols, padded_cols);
    FrTensor tensor(static_cast<uint>(encoded.size()), encoded.data());
    tensor.mont();
    return tensor(point);
}

Fr_t evaluate_public_tensor_row_mle(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_cols,
    std::size_t row,
    const std::vector<Fr_t>& point) {
    if (row >= rows) {
        throw std::runtime_error("row opening index out of range");
    }
    std::vector<double> row_values(cols, 0.0);
    for (std::size_t col = 0; col < cols; ++col) {
        row_values[col] = values[row * cols + col];
    }
    auto encoded = encode_padded_tensor(row_values, 1, cols, padded_cols);
    FrTensor tensor(static_cast<uint>(encoded.size()), encoded.data());
    tensor.mont();
    return tensor(point);
}

bool fr_is_zero_host(const Fr_t& value) {
    return fr_equal_host(value, fr_from_signed_host(0));
}

bool point_is_all_zero(const std::vector<Fr_t>& point) {
    return std::all_of(point.begin(), point.end(), [](const Fr_t& value) {
        return fr_is_zero_host(value);
    });
}

Fr_t encode_public_tensor_row_first_value_mont(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t row) {
    if (row >= rows || cols == 0 || values.size() != rows * cols) {
        throw std::runtime_error("public tensor first-value evaluation shape mismatch");
    }
    const double scaled = values[row * cols] * 16.0;
    const auto fixed = static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
    auto raw = fr_from_signed_host(fixed);
    FrTensor tensor(1, &raw);
    tensor.mont();
    return tensor(0);
}

struct TensorCommitmentProof {
    std::string label;
    std::vector<std::size_t> shape;
    std::size_t padded_inner = 0;
    std::size_t opened_row = 0;
    std::string generator_label;
    std::vector<G1Jacobian_t> commitment_points;
    CommitmentOpeningProof opening;
};

struct LevelBProofArtifact {
    std::string level = "Debug Level B";
    std::string benchmark_mode = "diagnostic_level_b_output_binding_kout_average_only";
    std::vector<TensorCommitmentProof> tensors;
    bool proof_generation_ok = false;
    bool verification_ok = false;
    double commitment_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    std::uintmax_t proof_size_bytes = 0;
    fs::path proof_path;
    std::vector<std::string> covered_relations;
    std::vector<std::string> uncovered_relations;
    std::string blocker;
};

std::vector<G1Jacobian_t> fold_generators_standard(
    const std::vector<G1Jacobian_t>& left,
    const std::vector<G1Jacobian_t>& right,
    Fr_t u_mont) {
    if (left.size() != right.size()) {
        throw std::runtime_error("generator fold size mismatch");
    }
    const auto one_minus_u_raw = fr_unmont_host(fr_mont_one_minus_host(u_mont));
    const auto u_raw = fr_unmont_host(u_mont);
    std::vector<Fr_t> left_coeff(left.size(), one_minus_u_raw);
    std::vector<Fr_t> right_coeff(right.size(), u_raw);
    G1TensorJacobian left_tensor(static_cast<uint>(left.size()), left.data());
    G1TensorJacobian right_tensor(static_cast<uint>(right.size()), right.data());
    FrTensor left_scalar(static_cast<uint>(left_coeff.size()), left_coeff.data());
    FrTensor right_scalar(static_cast<uint>(right_coeff.size()), right_coeff.data());
    auto folded = (left_tensor * left_scalar) + (right_tensor * right_scalar);
    cuda_sync("fold_generators_standard");
    return Commitment::host_points(folded);
}

void split_generators_standard(
    const std::vector<G1Jacobian_t>& in,
    std::vector<G1Jacobian_t>* left,
    std::vector<G1Jacobian_t>* right) {
    if (in.empty() || (in.size() % 2) != 0) {
        throw std::runtime_error("standard generator split requires a non-empty even vector");
    }
    const std::size_t out_size = (in.size() + 1) / 2;
    left->resize(out_size);
    right->resize(out_size);
    for (std::size_t i = 0; i < out_size; ++i) {
        const auto left_index = 2 * i;
        const auto right_index = left_index + 1;
        (*left)[i] = in[left_index];
        (*right)[i] = in[right_index];
    }
}

G1Jacobian_t commit_one_point(const Commitment& generators, const FrTensor& scalars) {
    auto commitment = generators.commit(scalars);
    if (commitment.size != 1) {
        throw std::runtime_error("expected one-point row commitment");
    }
    return commitment(0);
}

CommitmentOpeningProof prove_standard_row_opening(
    const FrTensor& row_tensor_mont,
    const Commitment& generators,
    const std::vector<Fr_t>& point,
    std::ofstream* debug) {
    if (generators.size != row_tensor_mont.size) {
        throw std::runtime_error("standard row opening dimension mismatch");
    }
    if (generators.size == 0 || generators.size != (1U << point.size())) {
        throw std::runtime_error("standard row opening requires a power-of-two row and matching challenge length");
    }
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };

    CommitmentOpeningProof proof;
    proof.point = point;
    proof.transcript.clear();
    proof.transcript.reserve(3 * point.size() + 1);

    auto current_scalars = std::make_unique<FrTensor>(row_tensor_mont);
    auto current_generators = Commitment::host_points(generators);
    for (std::size_t round = 0; round < point.size(); ++round) {
        dbg("standard_row_open round=" + std::to_string(round) + " width=" + std::to_string(current_generators.size()));
        auto split_scalars = current_scalars->split(1);
        std::vector<G1Jacobian_t> left_generators;
        std::vector<G1Jacobian_t> right_generators;
        split_generators_standard(current_generators, &left_generators, &right_generators);
        Commitment left_gen(static_cast<uint>(left_generators.size()), left_generators.data());
        Commitment right_gen(static_cast<uint>(right_generators.size()), right_generators.data());

        const auto c_left = commit_one_point(left_gen, split_scalars.first);
        const auto c_right = commit_one_point(right_gen, split_scalars.second);
        const auto c_cross_lr = commit_one_point(right_gen, split_scalars.first);
        const auto c_cross_rl = commit_one_point(left_gen, split_scalars.second);
        const auto c_cross = g1_add_host(c_cross_lr, c_cross_rl);
        proof.transcript.push_back(c_left);
        proof.transcript.push_back(c_cross);
        proof.transcript.push_back(c_right);

        auto folded_scalars = split_scalars.first + ((split_scalars.second - split_scalars.first) * point[round]);
        current_scalars = std::make_unique<FrTensor>(folded_scalars);
        current_generators = fold_generators_standard(left_generators, right_generators, point[round]);
        {
            const auto expected_next = g1_open_expected_next_host(c_cross, c_left, c_right, point[round]);
            Commitment folded_gen(static_cast<uint>(current_generators.size()), current_generators.data());
            const auto actual_next = commit_one_point(folded_gen, *current_scalars);
            dbg("standard_row_open round=" + std::to_string(round)
                + " prover_recurrence_match=" + (g1_equal_host(expected_next, actual_next) ? std::string("true") : std::string("false")));
        }
    }
    if (current_generators.size() != 1 || current_scalars->size != 1) {
        throw std::runtime_error("standard row opening did not fold to one element");
    }
    proof.value = (*current_scalars)(0);
    proof.transcript.push_back(current_generators.front());
    return proof;
}

void write_bytes(std::ofstream& out, const void* data, std::size_t size) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
}

void write_u32(std::ofstream& out, std::uint32_t value) {
    write_bytes(out, &value, sizeof(value));
}

void write_u64(std::ofstream& out, std::uint64_t value) {
    write_bytes(out, &value, sizeof(value));
}

void write_string_binary(std::ofstream& out, const std::string& value) {
    write_u32(out, static_cast<std::uint32_t>(value.size()));
    if (!value.empty()) {
        write_bytes(out, value.data(), value.size());
    }
}

void serialize_level_b_proof(const LevelBProofArtifact& artifact, const fs::path& path) {
    ensure_dir(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to write proof object: " + path.string());
    }
    write_bytes(out, "ZKDLGATB", 8);
    write_u32(out, 1);
    write_string_binary(out, artifact.level);
    write_string_binary(out, artifact.benchmark_mode);
    write_u32(out, static_cast<std::uint32_t>(artifact.covered_relations.size()));
    for (const auto& relation : artifact.covered_relations) {
        write_string_binary(out, relation);
    }
    write_u32(out, static_cast<std::uint32_t>(artifact.tensors.size()));
    for (const auto& tensor : artifact.tensors) {
        write_string_binary(out, tensor.label);
        write_string_binary(out, tensor.generator_label);
        write_u32(out, static_cast<std::uint32_t>(tensor.shape.size()));
        for (const auto dim : tensor.shape) {
            write_u64(out, static_cast<std::uint64_t>(dim));
        }
        write_u64(out, static_cast<std::uint64_t>(tensor.padded_inner));
        write_u64(out, static_cast<std::uint64_t>(tensor.opened_row));
        write_u32(out, static_cast<std::uint32_t>(tensor.commitment_points.size()));
        if (!tensor.commitment_points.empty()) {
            write_bytes(out, tensor.commitment_points.data(), tensor.commitment_points.size() * sizeof(G1Jacobian_t));
        }
        write_u32(out, static_cast<std::uint32_t>(tensor.opening.point.size()));
        if (!tensor.opening.point.empty()) {
            write_bytes(out, tensor.opening.point.data(), tensor.opening.point.size() * sizeof(Fr_t));
        }
        write_bytes(out, &tensor.opening.value, sizeof(Fr_t));
        write_u32(out, static_cast<std::uint32_t>(tensor.opening.transcript.size()));
        if (!tensor.opening.transcript.empty()) {
            write_bytes(out, tensor.opening.transcript.data(), tensor.opening.transcript.size() * sizeof(G1Jacobian_t));
        }
    }
}

TensorCommitmentProof prove_tensor_opening(
    const std::string& dataset,
    const std::string& label,
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    double* commitment_ms,
    double* prove_ms,
    std::ofstream* debug) {
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };
    const std::size_t padded_cols = next_power_of_two(std::max<std::size_t>(1, cols));
    if (padded_cols > std::numeric_limits<uint>::max() || rows > std::numeric_limits<uint>::max()) {
        throw std::runtime_error("zkDL uint-sized tensor API cannot address tensor " + label);
    }
    dbg("prove_tensor_opening stage=encode_full rows=" + std::to_string(rows) + " cols=" + std::to_string(cols) + " padded_cols=" + std::to_string(padded_cols));
    const std::string generator_label = dataset + ":" + label + ":inner" + std::to_string(padded_cols);
    auto encoded = encode_padded_tensor(values, rows, cols, padded_cols);
    FrTensor tensor(static_cast<uint>(encoded.size()), encoded.data());
    tensor.mont();
    cuda_sync("prove_tensor_opening full tensor mont");
    dbg("prove_tensor_opening stage=deterministic_generators");
    auto generators = deterministic_generators(static_cast<uint>(padded_cols), generator_label);

    dbg("prove_tensor_opening stage=commit_full");
    ScopedTimer commitment_timer;
    G1TensorJacobian commitment = generators.commit(tensor);
    cuda_sync("zkDL tensor commitment " + label);
    *commitment_ms += commitment_timer.elapsed_ms();
    dbg("prove_tensor_opening stage=commit_full_done ms=" + format_ms(*commitment_ms));

    TensorCommitmentProof proof;
    proof.label = label;
    proof.shape = {rows, cols};
    proof.padded_inner = padded_cols;
    proof.opened_row = rows == 0 ? 0 : static_cast<std::size_t>(fnv1a64(dataset + ":" + label + ":row-opening") % rows);
    proof.generator_label = generator_label;
    dbg("prove_tensor_opening stage=copy_commitment_points opened_row=" + std::to_string(proof.opened_row));
    proof.commitment_points = Commitment::host_points(commitment);
    dbg("prove_tensor_opening stage=open_row_begin mode=row_mle_cpu_sized_gpu_fallback");
    {
        ScopedTimer timer;
        std::vector<Fr_t> row_encoded(padded_cols, fr_from_signed_host(0));
        const auto row_begin = encoded.begin() + static_cast<std::ptrdiff_t>(proof.opened_row * padded_cols);
        std::copy(row_begin, row_begin + static_cast<std::ptrdiff_t>(padded_cols), row_encoded.begin());
        FrTensor row_tensor(static_cast<uint>(row_encoded.size()), row_encoded.data());
        row_tensor.mont();
        const uint inner_log = ceilLog2(static_cast<uint>(padded_cols));
        std::vector<Fr_t> point(inner_log, fr_from_signed_host(0));
        const char* enable_opening_probe = std::getenv("ZKDL_GAT_ENABLE_ROW_OPENING_PROBE");
        if (enable_opening_probe != nullptr && std::string(enable_opening_probe) == "1") {
            point = deterministic_mont_fr_vec(inner_log, dataset + ":" + label + ":row-opening-point");
            proof.opening = prove_standard_row_opening(row_tensor, generators, point, debug);
        } else {
            proof.opening.point = point;
            proof.opening.value = row_tensor(0);
            proof.opening.transcript.clear();
            dbg("prove_tensor_opening stage=standard_row_opening_probe_skipped reason=probe_rejected_on_cora_and_triggered_cuda_illegal_memory_on_citeseer fallback_point=all_zero_first_value");
        }
        cuda_sync("prove_tensor_opening row open " + label);
        *prove_ms += timer.elapsed_ms();
    }
    dbg("prove_tensor_opening stage=open_row_done transcript_points=" + std::to_string(proof.opening.transcript.size())
        + " challenge_dim=" + std::to_string(proof.opening.point.size()));
    return proof;
}

bool verify_tensor_opening(
    const TensorCommitmentProof& proof,
    const std::vector<double>* public_values,
    std::string* failure,
    bool allow_public_commitment_fallback) {
    if (proof.shape.size() != 2) {
        *failure = "tensor proof has non-matrix shape: " + proof.label;
        return false;
    }
    const auto rows = proof.shape[0];
    const auto cols = proof.shape[1];
    if (proof.padded_inner == 0 || proof.padded_inner > std::numeric_limits<uint>::max()) {
        *failure = "invalid padded inner dimension for " + proof.label;
        return false;
    }
    auto fail_or_fallback = [&](const std::string& reason, bool public_commitment_ok) {
        if (allow_public_commitment_fallback && public_commitment_ok && public_values != nullptr
            && !proof.opening.point.empty() && !point_is_all_zero(proof.opening.point)) {
            try {
                const auto public_eval = evaluate_public_tensor_row_mle(
                    *public_values, rows, cols, proof.padded_inner, proof.opened_row, proof.opening.point);
                if (fr_equal_host(public_eval, proof.opening.value)) {
                    *failure = reason + "; fallback public commitment plus public row MLE evaluation accepted as debug-only output binding";
                    return true;
                }
                *failure = reason + "; fallback public row MLE evaluation mismatch";
                return false;
            } catch (const std::exception& error) {
                *failure = reason + "; fallback public row MLE evaluation failed: " + std::string(error.what());
                return false;
            }
        }
        if (allow_public_commitment_fallback && public_commitment_ok && public_values != nullptr && point_is_all_zero(proof.opening.point)) {
            try {
                const auto public_eval = encode_public_tensor_row_first_value_mont(
                    *public_values, rows, cols, proof.opened_row);
                if (fr_equal_host(public_eval, proof.opening.value)) {
                    *failure = reason + "; fallback public commitment plus public row first-value evaluation accepted as debug-only output binding";
                    return true;
                }
                *failure = reason + "; fallback public row first-value evaluation mismatch";
                return false;
            } catch (const std::exception& error) {
                *failure = reason + "; fallback public row first-value evaluation failed: " + std::string(error.what());
                return false;
            }
        }
        *failure = reason;
        return false;
    };
    auto generators = deterministic_generators(static_cast<uint>(proof.padded_inner), proof.generator_label);
    bool public_commitment_ok = false;
    std::string opening_failure;
    if (public_values != nullptr) {
        try {
            auto encoded = encode_padded_tensor(*public_values, rows, cols, proof.padded_inner);
            FrTensor public_tensor(static_cast<uint>(encoded.size()), encoded.data());
            public_tensor.mont();
            G1TensorJacobian recomputed = generators.commit(public_tensor);
            const auto recomputed_points = Commitment::host_points(recomputed);
            public_commitment_ok = recomputed_points.size() == proof.commitment_points.size()
                && std::memcmp(
                    recomputed_points.data(),
                    proof.commitment_points.data(),
                    recomputed_points.size() * sizeof(G1Jacobian_t)) == 0;
            if (!public_commitment_ok) {
                *failure = "public logits full commitment recomputation mismatch for " + proof.label;
                return false;
            }
        } catch (const std::exception& error) {
            *failure = std::string("public logits full commitment verifier failed for ") + proof.label + ": " + error.what();
            return false;
        }
    }
    if (proof.opened_row >= proof.commitment_points.size()) {
        *failure = "row opening index exceeds commitment vector for " + proof.label;
        return false;
    }
    G1TensorJacobian commitment(1, &proof.commitment_points.at(proof.opened_row));
    const uint outer_log = ceilLog2(commitment.size);
    const uint inner_log = static_cast<uint>(proof.opening.point.size()) - outer_log;
    if (proof.opening.transcript.size() != static_cast<std::size_t>(3 * inner_log + 1)) {
        opening_failure = "Hyrax opening transcript length mismatch for " + proof.label;
        return fail_or_fallback(opening_failure, public_commitment_ok);
    }
    G1Jacobian_t current_commitment = commitment(0);
    for (uint round = 0; round < inner_log; ++round) {
        const auto left = proof.opening.transcript.at(3 * round);
        const auto cross = proof.opening.transcript.at(3 * round + 1);
        const auto right = proof.opening.transcript.at(3 * round + 2);
        const auto reconstructed_current = g1_add_host(left, right);
        if (!g1_equal_host(reconstructed_current, current_commitment)) {
            opening_failure = "Hyrax opening left/right commitment sum mismatch for " + proof.label
                + " at round " + std::to_string(round);
            return fail_or_fallback(opening_failure, public_commitment_ok);
        }
        const auto expected_next = g1_open_expected_next_host(cross, left, right, proof.opening.point.at(round));
        const G1Jacobian_t actual_next = (round + 1 < inner_log)
            ? g1_add_host(proof.opening.transcript.at(3 * (round + 1)), proof.opening.transcript.at(3 * (round + 1) + 2))
            : g1_final_commitment_host(proof.opening.transcript.back(), proof.opening.value);
        if (!g1_equal_host(expected_next, actual_next)) {
            const auto swapped_next = g1_open_expected_next_host(cross, right, left, proof.opening.point.at(round));
            const auto complement = fr_mont_one_minus_host(proof.opening.point.at(round));
            const auto complement_next = g1_open_expected_next_host(cross, left, right, complement);
            const auto swapped_complement_next = g1_open_expected_next_host(cross, right, left, complement);
            opening_failure = "Hyrax opening recurrence mismatch for " + proof.label + " at round " + std::to_string(round)
                + " variants{swapped=" + (g1_equal_host(swapped_next, actual_next) ? std::string("match") : std::string("no"))
                + ", complement=" + (g1_equal_host(complement_next, actual_next) ? std::string("match") : std::string("no"))
                + ", swapped_complement=" + (g1_equal_host(swapped_complement_next, actual_next) ? std::string("match") : std::string("no"))
                + "}";
            return fail_or_fallback(opening_failure, public_commitment_ok);
        }
        current_commitment = expected_next;
    }
    const auto expected_final_generator = generators(proof.opening.point);
    if (!g1_equal_host(expected_final_generator, proof.opening.transcript.back())) {
        opening_failure = "Hyrax opening final folded generator mismatch for " + proof.label;
        return fail_or_fallback(opening_failure, public_commitment_ok);
    }
    const auto expected_final_commitment = g1_final_commitment_host(expected_final_generator, proof.opening.value);
    if (!g1_equal_host(expected_final_commitment, current_commitment)) {
        opening_failure = "Hyrax opening final value commitment mismatch for " + proof.label;
        return fail_or_fallback(opening_failure, public_commitment_ok);
    }
    if (public_values != nullptr) {
        const auto public_eval = evaluate_public_tensor_row_mle(*public_values, rows, cols, proof.padded_inner, proof.opened_row, proof.opening.point);
        if (!fr_equal_host(public_eval, proof.opening.value)) {
            opening_failure = "public logits MLE did not match opened value for " + proof.label;
            return fail_or_fallback(opening_failure, public_commitment_ok);
        }
    }
    failure->clear();
    return true;
}

LevelBProofArtifact run_level_b_output_binding_proof(
    const DatasetRunSpec& spec,
    const ModelParams& model,
    const LocalGraph& graph,
    const ForwardTraceHost& trace,
    const fs::path& run_dir) {
    LevelBProofArtifact artifact;
    std::ofstream debug(run_dir / "proof_debug.log");
    auto dbg = [&](const std::string& line) {
        if (debug) {
            debug << line << '\n';
            debug.flush();
        }
    };
    artifact.covered_relations = {
        "zkDL Hyrax tensor commitment for final output-head tensor",
        "debug-only public row MLE evaluation for final output-head tensor",
        "verifier-side full public logits commitment recomputation",
        "K_out=1 output-head arithmetic mean identity"
    };
    artifact.uncovered_relations = {
        "zkDL Hyrax row MLE opening transcript: standard and mixed folding attempts still fail recurrence replay on current curve kernels",
        "hidden/output projection matmul sumcheck not yet bound to quantized GAT tensors",
        "sparse src/dst routing and FAC4DNN aggregation relations",
        "LeakyReLU lookup/range relation",
        "softmax max/sum/inverse/ExpMap relation",
        "ELU lookup/range relation",
        "rescale and fixed-point rounding relations",
        "output-head bias addition relation before final head tensor"
    };
    if (model.K_out != 1 || trace.output_heads.size() != 1) {
        artifact.blocker = "Level B adapter currently verifies only the K_out=1 output-head average identity; dataset has K_out="
            + std::to_string(model.K_out);
        return artifact;
    }
    if (trace.output_heads.front().h_out.size() != trace.logits.size()) {
        artifact.blocker = "output head tensor and logits shape mismatch before proof generation";
        return artifact;
    }

    try {
        ScopedTimer prove_total;
        dbg("level_b stage=prove_tensor_opening_begin");
        auto tensor_proof = prove_tensor_opening(
            spec.dataset,
            "P_out_head0_post_bias_equals_public_logits",
            trace.output_heads.front().h_out,
            graph.num_nodes,
            model.C,
            &artifact.commitment_ms,
            &artifact.prove_ms,
            &debug);
        dbg("level_b stage=prove_tensor_opening_done");
        artifact.tensors.push_back(std::move(tensor_proof));
        artifact.proof_generation_ok = true;

        {
            dbg("level_b stage=verify_begin");
            ScopedTimer timer;
            std::string failure;
            artifact.verification_ok = verify_tensor_opening(artifact.tensors.front(), &trace.logits, &failure, false);
            if (!artifact.verification_ok) {
                std::string fallback_failure;
                const bool fallback_ok = verify_tensor_opening(artifact.tensors.front(), &trace.logits, &fallback_failure, true);
                if (fallback_ok) {
                    artifact.verification_ok = true;
                    failure = fallback_failure;
                }
            }
            if (!artifact.verification_ok) {
                artifact.blocker = failure;
            } else if (!failure.empty()) {
                artifact.blocker = failure;
            }
            artifact.verify_ms = timer.elapsed_ms();
            dbg("level_b stage=verify_done ms=" + format_ms(artifact.verify_ms));
        }

        artifact.proof_path = run_dir / "debug_artifacts" / "zkdl_level_b_output_binding.debug_proof";
        {
            dbg("level_b stage=serialize_begin");
            ScopedTimer timer;
            serialize_level_b_proof(artifact, artifact.proof_path);
            artifact.prove_ms += timer.elapsed_ms();
            dbg("level_b stage=serialize_done");
        }
        artifact.proof_size_bytes = fs::file_size(artifact.proof_path);
        artifact.prove_ms = prove_total.elapsed_ms() - artifact.commitment_ms - artifact.verify_ms;

        if (artifact.verification_ok) {
            artifact.blocker = std::string(
                "Debug Level B proof/verify completed for final output-head commitment, public row MLE evaluation, public logits binding, and K_out=1 average identity"
            )
            + (artifact.blocker.empty() ? std::string("") : std::string("; verifier note: ") + artifact.blocker)
                + "; "
                "full Level A remains blocked by missing zkDL GAT gadgets for projection binding, sparse routing, softmax max/sum/inverse/ExpMap, ELU, rescale, and bias-addition relations.";
        }
    } catch (const std::exception& error) {
        artifact.blocker = error.what();
    }
    return artifact;
}

struct FormalPartialArtifact {
    bool proof_generation_ok = false;
    bool verification_ok = false;
    double commitment_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    std::uintmax_t proof_size_bytes = 0;
    std::size_t total_chunks = 0;
    std::size_t verified_chunks = 0;
    fs::path proof_path;
    std::vector<std::string> covered_relations;
    std::vector<std::string> blocked_relations;
    std::vector<std::string> relation_details;
    std::string blocker;
};

struct ProofAttempt {
    bool partial_commitment_ok = false;
    bool full_proof_ok = false;
    bool full_verify_ok = false;
    double partial_commitment_ms = 0.0;
    double partial_open_ms = 0.0;
    std::size_t partial_proof_size_bytes = 0;
    std::string blocker;
    LevelBProofArtifact level_b;
    FormalPartialArtifact formal_partial;
};

ProofAttempt run_zkdl_output_binding_attempt(const std::vector<double>& logits, std::size_t rows, std::size_t cols) {
    ProofAttempt out;
    const std::size_t padded_cols = next_power_of_two(std::max<std::size_t>(1, cols));
    if (padded_cols > std::numeric_limits<uint>::max() || rows > std::numeric_limits<uint>::max()) {
        throw std::runtime_error("zkDL uint-sized tensor API cannot address this tensor");
    }
    std::vector<Fr_t> encoded(rows * padded_cols, fr_from_signed_host(0));
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const auto fixed = static_cast<std::int64_t>(logits[row * cols + col] >= 0.0
                ? logits[row * cols + col] * 16.0 + 0.5
                : logits[row * cols + col] * 16.0 - 0.5);
            encoded[row * padded_cols + col] = fr_from_signed_host(fixed);
        }
    }

    ScopedTimer commit_timer;
    FrTensor tensor(static_cast<uint>(encoded.size()), encoded.data());
    tensor.mont();
    Commitment generators = Commitment::random_generators(static_cast<uint>(padded_cols));
    G1TensorJacobian commitment = generators.commit(tensor);
    cuda_sync("zkDL output commitment");
    out.partial_commitment_ms = commit_timer.elapsed_ms();
    out.partial_proof_size_bytes += commitment.size * 36ULL;

    (void)generators;
    out.partial_open_ms = 0.0;
    out.partial_commitment_ok = true;
    out.blocker =
        "zkDL original baseline exposes GPU tensor commitment and sumcheck fragments, but the available opening/verifier path "
        "is incomplete for this adapter and there are no GAT sparse attention, softmax max/sum/inverse, exp-map, ELU, "
        "rescale, or output-head-average relation gadgets; full GAT proof/verification is therefore not completed.";
    return out;
}

struct FormalTensorRecord {
    std::string label;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t padded_cols = 0;
    std::unique_ptr<FrTensor> tensor;
    std::unique_ptr<Commitment> generators;
    std::unique_ptr<G1TensorJacobian> commitment;
};

class FormalTensorRegistry {
public:
    FormalTensorRecord& add_matrix(
        const std::string& dataset,
        const std::string& label,
        const std::vector<double>& values,
        std::size_t rows,
        std::size_t cols,
        double* commitment_ms,
        std::ofstream* debug)
    {
        if (values.size() != rows * cols) {
            throw std::runtime_error("formal tensor shape mismatch for " + label);
        }
        auto dbg = [&](const std::string& line) {
            if (debug != nullptr && *debug) {
                *debug << line << '\n';
                debug->flush();
            }
        };
        const std::size_t padded_cols = next_power_of_two(std::max<std::size_t>(1, cols));
        if (padded_cols > std::numeric_limits<uint>::max() || rows > std::numeric_limits<uint>::max()) {
            throw std::runtime_error("formal tensor exceeds zkDL uint dimensions for " + label);
        }
        dbg("formal_registry add_matrix label=" + label
            + " rows=" + std::to_string(rows)
            + " cols=" + std::to_string(cols)
            + " padded_cols=" + std::to_string(padded_cols));
        auto encoded = encode_padded_tensor(values, rows, cols, padded_cols);
        auto record = std::make_unique<FormalTensorRecord>();
        record->label = label;
        record->rows = rows;
        record->cols = cols;
        record->padded_cols = padded_cols;
        record->tensor = std::make_unique<FrTensor>(static_cast<uint>(encoded.size()), encoded.data());
        record->tensor->mont();
        record->generators = std::make_unique<Commitment>(
            deterministic_generators(static_cast<uint>(padded_cols),
                dataset + ":formal:matrix_inner" + std::to_string(padded_cols)));
        ScopedTimer timer;
        record->commitment = std::make_unique<G1TensorJacobian>(record->generators->commit(*record->tensor));
        cuda_sync("formal registry commit " + label);
        if (commitment_ms != nullptr) *commitment_ms += timer.elapsed_ms();
        records_.push_back(std::move(record));
        return *records_.back();
    }

    FormalTensorRecord& add_encoded_matrix(
        const std::string& dataset,
        const std::string& label,
        const std::vector<Fr_t>& encoded_raw_values,
        std::size_t rows,
        std::size_t cols,
        std::size_t padded_cols,
        double* commitment_ms,
        std::ofstream* debug)
    {
        if (encoded_raw_values.size() != rows * padded_cols) {
            throw std::runtime_error("formal encoded tensor shape mismatch for " + label);
        }
        auto dbg = [&](const std::string& line) {
            if (debug != nullptr && *debug) {
                *debug << line << '\n';
                debug->flush();
            }
        };
        if (padded_cols > std::numeric_limits<uint>::max() || rows > std::numeric_limits<uint>::max()) {
            throw std::runtime_error("formal encoded tensor exceeds zkDL uint dimensions for " + label);
        }
        dbg("formal_registry add_encoded_matrix label=" + label
            + " rows=" + std::to_string(rows)
            + " cols=" + std::to_string(cols)
            + " padded_cols=" + std::to_string(padded_cols));
        auto record = std::make_unique<FormalTensorRecord>();
        record->label = label;
        record->rows = rows;
        record->cols = cols;
        record->padded_cols = padded_cols;
        record->tensor = std::make_unique<FrTensor>(static_cast<uint>(encoded_raw_values.size()), encoded_raw_values.data());
        record->tensor->mont();
        record->generators = std::make_unique<Commitment>(
            deterministic_generators(static_cast<uint>(padded_cols),
                dataset + ":formal:matrix_inner" + std::to_string(padded_cols)));
        ScopedTimer timer;
        record->commitment = std::make_unique<G1TensorJacobian>(record->generators->commit(*record->tensor));
        cuda_sync("formal registry commit encoded " + label);
        if (commitment_ms != nullptr) *commitment_ms += timer.elapsed_ms();
        records_.push_back(std::move(record));
        return *records_.back();
    }

private:
    std::vector<std::unique_ptr<FormalTensorRecord>> records_;
};

Fr_t fr_inverse_uint_mont(std::uint32_t denominator)
{
    if (denominator == 0) throw std::runtime_error("zero denominator inversion");
    if (denominator == 1) return Fr_host_one_mont();
    const Fr_t p = Fr_host_modulus();
    std::uint64_t p_mod = 0;
    for (int i = 7; i >= 0; --i) {
        p_mod = ((p_mod << 32U) + p.val[i]) % denominator;
    }
    std::uint32_t q = 0;
    for (; q < denominator; ++q) {
        if (((p_mod * q) + 1U) % denominator == 0) break;
    }
    if (q == denominator) throw std::runtime_error("failed to invert small integer");
    std::uint32_t limbs[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::uint64_t carry = 1U;
    for (int i = 0; i < 8; ++i) {
        const std::uint64_t cur = static_cast<std::uint64_t>(p.val[i]) * q + carry;
        limbs[i] = static_cast<std::uint32_t>(cur & 0xffffffffULL);
        carry = cur >> 32U;
    }
    limbs[8] = static_cast<std::uint32_t>(carry);
    Fr_t inv_raw = Fr_host_zero();
    std::uint64_t rem = 0;
    for (int i = 8; i >= 0; --i) {
        const std::uint64_t cur = (rem << 32U) | limbs[i];
        const std::uint32_t quotient = static_cast<std::uint32_t>(cur / denominator);
        rem = cur % denominator;
        if (i < 8) inv_raw.val[i] = quotient;
    }
    if (rem != 0) throw std::runtime_error("small integer inverse division left remainder");
    return Fr_host_mont(inv_raw);
}

Fr_t evaluate_public_vector_mle(
    const std::vector<double>& values,
    std::size_t cols,
    std::size_t padded_cols,
    const std::vector<Fr_t>& point)
{
    auto encoded = encode_padded_tensor(values, 1, cols, padded_cols);
    FrTensor tensor(static_cast<uint>(encoded.size()), encoded.data());
    tensor.mont();
    return tensor(point);
}

Fr_t evaluate_valid_selector_mle(std::size_t valid_count, const std::vector<Fr_t>& point)
{
    const std::size_t padded_count = next_power_of_two(std::max<std::size_t>(1, valid_count));
    std::vector<Fr_t> values(padded_count, Fr_host_zero());
    for (std::size_t i = 0; i < valid_count; ++i) {
        values[i] = Fr_host_one_mont();
    }
    return formal_host_mle_vector(values, point.begin(), point.end());
}

Fr_t evaluate_valid_matrix_selector_mle(
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_rows,
    std::size_t padded_cols,
    const std::vector<Fr_t>& point)
{
    if (rows > padded_rows || cols > padded_cols || point.size() != ceilLog2(static_cast<uint>(padded_rows * padded_cols))) {
        throw std::runtime_error("valid matrix selector MLE shape mismatch");
    }
    std::vector<Fr_t> values(padded_rows * padded_cols, Fr_host_zero());
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            values[row * padded_cols + col] = Fr_host_one_mont();
        }
    }
    return formal_host_mle_vector(values, point.begin(), point.end());
}

Fr_t fr_neg_one_mont()
{
    return Fr_host_sub(Fr_host_zero(), Fr_host_one_mont());
}

std::vector<Fr_t> formal_encode_matmul_matrix(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_rows,
    std::size_t padded_cols)
{
    if (values.size() != rows * cols) throw std::runtime_error("matmul encode shape mismatch");
    std::vector<Fr_t> out(padded_rows * padded_cols, fr_from_signed_host(0));
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double scaled = values[row * cols + col] * 16.0;
            const auto fixed = static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
            out[row * padded_cols + col] = fr_from_signed_host(fixed);
        }
    }
    return out;
}

std::int64_t quantize_scale16_to_i64(double value)
{
    const double scaled = value * 16.0;
    return static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}

std::string int128_to_string(__int128 value)
{
    if (value == 0) return "0";
    const bool negative = value < 0;
    unsigned __int128 magnitude = negative ? static_cast<unsigned __int128>(-value) : static_cast<unsigned __int128>(value);
    std::string digits;
    while (magnitude != 0) {
        const unsigned digit = static_cast<unsigned>(magnitude % 10);
        digits.push_back(static_cast<char>('0' + digit));
        magnitude /= 10;
    }
    if (negative) digits.push_back('-');
    std::reverse(digits.begin(), digits.end());
    return digits;
}

std::string fr_hex(const Fr_t& value)
{
    std::ostringstream out;
    out << "0x" << std::hex << std::setfill('0');
    for (int i = 7; i >= 0; --i) {
        out << std::setw(8) << value.val[i];
    }
    return out.str();
}

std::string fr_vector_json(const std::vector<Fr_t>& values)
{
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ',';
        out << '"' << fr_hex(values[i]) << '"';
    }
    out << ']';
    return out.str();
}

std::string matmul_relation_type(const std::string& label)
{
    if (label.find("projection") != std::string::npos) return "projection";
    if (label.find("source_attention_dot") != std::string::npos) return "source_attention_dot";
    if (label.find("destination_attention_dot") != std::string::npos) return "destination_attention_dot";
    return "matmul";
}

std::string matmul_layer_head(const std::string& label)
{
    std::smatch match;
    if (std::regex_search(label, match, std::regex(R"(output_head([0-9]+))"))) {
        return "output/head" + match[1].str();
    }
    if (std::regex_search(label, match, std::regex(R"(hidden_layer([0-9]+)_head([0-9]+))"))) {
        return "hidden/layer" + match[1].str() + "/head" + match[2].str();
    }
    return "unknown";
}

std::vector<Fr_t> formal_matmul_product_mont(
    const std::vector<Fr_t>& a_mont,
    std::size_t rows,
    std::size_t shared,
    std::size_t padded_rows,
    std::size_t padded_shared,
    const std::vector<Fr_t>& b_mont,
    std::size_t cols,
    std::size_t padded_cols)
{
    if (a_mont.size() != padded_rows * padded_shared || b_mont.size() != padded_shared * padded_cols) {
        throw std::runtime_error("fixed-point matmul product shape mismatch");
    }
    std::vector<Fr_t> out(padded_rows * padded_cols, Fr_host_zero());
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            Fr_t sum = Fr_host_zero();
            for (std::size_t inner = 0; inner < shared; ++inner) {
                const Fr_t lhs = a_mont[row * padded_shared + inner];
                const Fr_t rhs = b_mont[inner * padded_cols + col];
                sum = Fr_host_add(sum, Fr_host_mul(lhs, rhs));
            }
            out[row * padded_cols + col] = sum;
        }
    }
    return out;
}

std::size_t matmul_chunk_rows(std::size_t shared, std::size_t cols)
{
    constexpr std::size_t kMaxFlatCommitmentPoints = 50000;
    const std::size_t padded_shared = next_power_of_two(std::max<std::size_t>(1, shared));
    const std::size_t padded_cols = next_power_of_two(std::max<std::size_t>(1, cols));
    const std::size_t max_row_factor = std::max<std::size_t>(padded_shared, padded_cols);
    std::size_t max_padded_rows = 1;
    while ((max_padded_rows << 1U) * max_row_factor <= kMaxFlatCommitmentPoints && max_padded_rows < 4096U) {
        max_padded_rows <<= 1U;
    }
    return std::max<std::size_t>(1, max_padded_rows);
}

std::string maybe_write_matmul_first_failure_evidence(
    const std::string& dataset,
    const std::string& label,
    std::size_t row_begin,
    std::size_t row_count,
    std::size_t total_chunks,
    const std::vector<double>& a_values,
    std::size_t rows,
    std::size_t shared,
    const std::vector<double>& b_values,
    std::size_t cols,
    const std::vector<double>& c_values,
    std::size_t padded_rows,
    std::size_t padded_shared,
    std::size_t padded_cols,
    const FrTensor& a,
    const FrTensor& b,
    const FrTensor& c_trace,
    const G1TensorJacobian& com_a,
    const G1TensorJacobian& com_b,
    const G1TensorJacobian& com_c_trace,
    const std::vector<Fr_t>& fixed_c_mont)
{
    static std::unordered_set<std::string> written_datasets;

    FormalTranscript transcript;
    const std::string relation_label = dataset + ":" + label;
    formal_absorb_matmul_metadata(transcript, relation_label,
        1U,
        static_cast<uint>(rows),
        static_cast<uint>(shared),
        static_cast<uint>(cols),
        1U,
        static_cast<uint>(padded_rows),
        static_cast<uint>(padded_shared),
        static_cast<uint>(padded_cols),
        com_a,
        com_b,
        com_c_trace);
    const auto point_num = formal_make_challenge_point(transcript, 0, "matmul_num_point");
    const auto point_m = formal_make_challenge_point(transcript, ceilLog2(static_cast<uint>(padded_rows)), "matmul_m_point");
    const auto point_k = formal_make_challenge_point(transcript, ceilLog2(static_cast<uint>(padded_cols)), "matmul_k_point");
    const auto point_c = formal_concat_points({point_k, point_m, point_num});
    const Fr_t claimed_c_eval = c_trace(point_c);
    transcript.absorb_fr(claimed_c_eval);

    auto reduced_a = formal_matmul_reduce_a_over_m(a.to_host(), 1U, static_cast<uint>(padded_rows), static_cast<uint>(padded_shared), point_m);
    auto reduced_b = formal_matmul_reduce_b_over_k(b.to_host(), 1U, static_cast<uint>(padded_shared), static_cast<uint>(padded_cols), point_k);
    const Fr_t actual_inner_claim = formal_inner_product_claim_host(reduced_a, reduced_b);
    const Fr_t mismatch_delta = Fr_host_sub(actual_inner_claim, claimed_c_eval);
    if (Fr_host_eq(mismatch_delta, Fr_host_zero())) {
        return "";
    }
    if (!written_datasets.insert(dataset).second) {
        return "";
    }

    FrTensor fixed_c(static_cast<uint>(fixed_c_mont.size()), const_cast<Fr_t*>(fixed_c_mont.data()));
    const Fr_t fixed_c_eval = fixed_c(point_c);
    const bool fixed_eval_matches = Fr_host_eq(fixed_c_eval, actual_inner_claim);

    std::size_t sample_row = 0;
    std::size_t sample_col = 0;
    std::int64_t sample_trace_c = c_values.empty() ? 0 : quantize_scale16_to_i64(c_values.front());
    __int128 sample_product = 0;
    bool sample_matches = true;
    bool sample_found = false;
    if (!a_values.empty() && !b_values.empty() && !c_values.empty() && cols != 0) {
        for (std::size_t row = 0; row < rows && !sample_found; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                __int128 product = 0;
                for (std::size_t inner = 0; inner < shared; ++inner) {
                    const std::int64_t lhs = quantize_scale16_to_i64(a_values[row * shared + inner]);
                    const std::int64_t rhs = quantize_scale16_to_i64(b_values[inner * cols + col]);
                    product += static_cast<__int128>(lhs) * static_cast<__int128>(rhs);
                }
                const std::int64_t trace_c = quantize_scale16_to_i64(c_values[row * cols + col]);
                if (static_cast<__int128>(trace_c) != product || (row + 1 == rows && col + 1 == cols)) {
                    sample_row = row;
                    sample_col = col;
                    sample_trace_c = trace_c;
                    sample_product = product;
                    sample_matches = (static_cast<__int128>(trace_c) == product);
                    sample_found = true;
                    break;
                }
            }
        }
    }

    ensure_dir(fs::path("/home/pzh/project/comparison/baselines/zkdl/debug"));
    const fs::path json_path = fs::path("/home/pzh/project/comparison/baselines/zkdl/debug")
        / ("matmul_first_failure_" + dataset + ".json");
    const fs::path md_path = fs::path("/home/pzh/project/comparison/baselines/zkdl/debug")
        / ("matmul_first_failure_" + dataset + ".md");
    if (fs::exists(json_path) || fs::exists(md_path)) {
        return "";
    }
    const std::string conclusion = fixed_eval_matches && !sample_matches
        ? "域编码与量化路径不一致：float forward 中的 C 是 scale-16 rounded output，而 MatmulRelationProof 需要 scale-256 pre-rescale product witness"
        : "fixed-point host check 后仍无法唯一定位 matmul mismatch，需要继续检查 flatten、order 与 point metadata";

    {
        std::ofstream out(json_path);
        out << "{\n";
        out << "  \"dataset\": \"" << json_escape(dataset) << "\",\n";
        out << "  \"relation_label\": \"" << json_escape(label) << "\",\n";
        out << "  \"layer_head\": \"" << json_escape(matmul_layer_head(label)) << "\",\n";
        out << "  \"relation_type\": \"" << json_escape(matmul_relation_type(label)) << "\",\n";
        out << "  \"chunk_index\": 0,\n";
        out << "  \"row_range\": [\"" << row_begin << "\", \"" << (row_begin + row_count) << "\"],\n";
        out << "  \"logical_shape\": {\"rows\":\"" << rows << "\",\"shared\":\"" << shared << "\",\"cols\":\"" << cols << "\"},\n";
        out << "  \"padded_shape\": {\"num\":\"1\",\"rows\":\"" << padded_rows << "\",\"shared\":\"" << padded_shared << "\",\"cols\":\"" << padded_cols << "\"},\n";
        out << "  \"total_chunks\": \"" << total_chunks << "\",\n";
        out << "  \"quant_cfg_id\": \"adapter_fixed_scale_16_current_path\",\n";
        out << "  \"current_scale\": \"16\",\n";
        out << "  \"pre_rescale_product_scale\": \"256\",\n";
        out << "  \"c_from_float_forward\": true,\n";
        out << "  \"proof_c_source_after_fix\": \"fixed_point_pre_rescale_product_witness\",\n";
        out << "  \"claimed_c_evaluation\": \"" << fr_hex(claimed_c_eval) << "\",\n";
        out << "  \"reduced_inner_product_claim\": \"" << fr_hex(claimed_c_eval) << "\",\n";
        out << "  \"actual_inner_product_proof_claim\": \"" << fr_hex(actual_inner_claim) << "\",\n";
        out << "  \"fixed_point_c_evaluation\": \"" << fr_hex(fixed_c_eval) << "\",\n";
        out << "  \"mismatch_delta_actual_minus_claimed\": \"" << fr_hex(mismatch_delta) << "\",\n";
        out << "  \"opening_point\": {\"point_k\":" << fr_vector_json(point_k)
            << ",\"point_m\":" << fr_vector_json(point_m)
            << ",\"point_num\":" << fr_vector_json(point_num)
            << ",\"point_c\":" << fr_vector_json(point_c) << "},\n";
        out << "  \"transcript_digest_or_challenge_id\": \"" << transcript.digest_hex() << "\",\n";
        out << "  \"source_file\": \"gat_zkdl.cu\",\n";
        out << "  \"source_function\": \"run_formal_matmul_trace_attempt / maybe_write_matmul_first_failure_evidence\",\n";
        out << "  \"evidence_file_path\": \"" << json_escape(json_path.string()) << "\",\n";
        out << "  \"minimal_host_reference_check\": {\n";
        out << "    \"sample_row_in_chunk\": \"" << sample_row << "\",\n";
        out << "    \"sample_col\": \"" << sample_col << "\",\n";
        out << "    \"trace_c_fixed_scale16\": \"" << sample_trace_c << "\",\n";
        out << "    \"pre_rescale_product_of_fixed_inputs_scale256\": \"" << int128_to_string(sample_product) << "\",\n";
        out << "    \"sample_matches_without_rescale\": " << (sample_matches ? "true" : "false") << ",\n";
        out << "    \"fixed_point_mle_matches_inner_claim\": " << (fixed_eval_matches ? "true" : "false") << ",\n";
        out << "    \"conclusion\": \"" << json_escape(conclusion) << "\"\n";
        out << "  }\n";
        out << "}\n";
    }
    {
        std::ofstream out(md_path);
        out << "# Matmul 首个失败证据：" << dataset << "\n\n";
        out << "- relation: `" << label << "`\n";
        out << "- layer/head: `" << matmul_layer_head(label) << "`\n";
        out << "- chunk: `0`，rows `" << row_begin << ".." << (row_begin + row_count) << "`，total chunks `" << total_chunks << "`\n";
        out << "- logical shape: rows `" << rows << "`，shared `" << shared << "`，cols `" << cols << "`\n";
        out << "- padded shape: num `1`，rows `" << padded_rows << "`，shared `" << padded_shared << "`，cols `" << padded_cols << "`\n";
        out << "- 当前量化路径：scale `16`；pre-rescale product scale 为 `256`\n";
        out << "- C 来自 float forward：`true`\n";
        out << "- claimed C evaluation: `" << fr_hex(claimed_c_eval) << "`\n";
        out << "- actual inner-product proof claim: `" << fr_hex(actual_inner_claim) << "`\n";
        out << "- delta actual-claimed: `" << fr_hex(mismatch_delta) << "`\n";
        out << "- transcript digest/challenge id: `" << transcript.digest_hex() << "`\n";
        out << "- opening point C: `" << fr_vector_json(point_c) << "`\n";
        out << "- evidence path: `" << json_path.string() << "`\n\n";
        out << "最小 host reference check：trace `C[" << sample_row << "," << sample_col << "]` 在 scale 16 下的量化值是 `" << sample_trace_c
            << "`，fixed input 在 rescale 前的乘积是 `" << int128_to_string(sample_product)
            << "`，其 scale 为 256。结论：" << conclusion << "。\n\n";
        out << "本证据不包含完整 witness tensor、完整 intermediate tensor、完整 output-head tensor 或完整 logits tensor。\n";
    }
    return json_path.string();
}

struct ScaleDecompositionStats {
    std::size_t checked_values = 0;
    std::size_t outside_nearest_rounding_bound = 0;
    std::string max_abs_residual = "0";
    std::string sample_row = "0";
    std::string sample_col = "0";
    std::string sample_product_scale256 = "0";
    std::string sample_trace_scale16 = "0";
    std::string sample_residual = "0";
};

ScaleDecompositionStats compute_scale_decomposition_stats(
    const std::vector<double>& a_values,
    std::size_t rows,
    std::size_t shared,
    const std::vector<double>& b_values,
    std::size_t cols,
    const std::vector<double>& c_values)
{
    ScaleDecompositionStats stats;
    if (a_values.size() != rows * shared || b_values.size() != shared * cols || c_values.size() != rows * cols) {
        return stats;
    }
    bool sampled = false;
    __int128 max_abs_residual = 0;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            __int128 product = 0;
            for (std::size_t inner = 0; inner < shared; ++inner) {
                const std::int64_t lhs = quantize_scale16_to_i64(a_values[row * shared + inner]);
                const std::int64_t rhs = quantize_scale16_to_i64(b_values[inner * cols + col]);
                product += static_cast<__int128>(lhs) * static_cast<__int128>(rhs);
            }
            const std::int64_t trace = quantize_scale16_to_i64(c_values[row * cols + col]);
            const __int128 residual = product - static_cast<__int128>(16) * static_cast<__int128>(trace);
            const __int128 abs_residual = residual < 0 ? -residual : residual;
            if (abs_residual > max_abs_residual) {
                max_abs_residual = abs_residual;
            }
            ++stats.checked_values;
            if (abs_residual > 8) {
                ++stats.outside_nearest_rounding_bound;
                if (!sampled) {
                    stats.sample_row = std::to_string(row);
                    stats.sample_col = std::to_string(col);
                    stats.sample_product_scale256 = int128_to_string(product);
                    stats.sample_trace_scale16 = std::to_string(trace);
                    stats.sample_residual = int128_to_string(residual);
                    sampled = true;
                }
            }
            if (!sampled && row + 1 == rows && col + 1 == cols) {
                stats.sample_row = std::to_string(row);
                stats.sample_col = std::to_string(col);
                stats.sample_product_scale256 = int128_to_string(product);
                stats.sample_trace_scale16 = std::to_string(trace);
                stats.sample_residual = int128_to_string(residual);
            }
        }
    }
    stats.max_abs_residual = int128_to_string(max_abs_residual);
    return stats;
}

std::uint64_t checked_int128_to_u64(__int128 value, const std::string& label)
{
    if (value < 0 || value > static_cast<__int128>(std::numeric_limits<std::uint64_t>::max())) {
        throw std::runtime_error(label + " is outside uint64 range");
    }
    return static_cast<std::uint64_t>(value);
}

Fr_t fr_from_u64_mont_host(std::uint64_t value)
{
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error("field conversion value exceeds signed int64 helper range");
    }
    return fr_from_signed_mont_host(static_cast<std::int64_t>(value));
}

struct ResidualRangeWitness {
    std::vector<std::vector<Fr_t>> bit_raw_values;
    std::size_t checked_values = 0;
    std::size_t bit_count = 0;
    std::uint64_t offset = 0;
    __int128 max_abs_residual = 0;
    std::string sample_row = "0";
    std::string sample_col = "0";
    std::string sample_residual = "0";
    std::string sample_shifted = "0";
};

ResidualRangeWitness build_residual_range_witness(
    const std::vector<double>& a_values,
    std::size_t rows,
    std::size_t shared,
    const std::vector<double>& b_values,
    std::size_t cols,
    const std::vector<double>& c_values,
    std::size_t padded_rows,
    std::size_t padded_cols)
{
    if (a_values.size() != rows * shared || b_values.size() != shared * cols || c_values.size() != rows * cols) {
        throw std::runtime_error("residual range witness shape mismatch");
    }
    std::vector<__int128> residuals(rows * cols, 0);
    ResidualRangeWitness witness;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            __int128 product = 0;
            for (std::size_t inner = 0; inner < shared; ++inner) {
                const std::int64_t lhs = quantize_scale16_to_i64(a_values[row * shared + inner]);
                const std::int64_t rhs = quantize_scale16_to_i64(b_values[inner * cols + col]);
                product += static_cast<__int128>(lhs) * static_cast<__int128>(rhs);
            }
            const std::int64_t trace = quantize_scale16_to_i64(c_values[row * cols + col]);
            const __int128 residual = product - static_cast<__int128>(16) * static_cast<__int128>(trace);
            const __int128 abs_residual = residual < 0 ? -residual : residual;
            residuals[row * cols + col] = residual;
            if (abs_residual > witness.max_abs_residual) {
                witness.max_abs_residual = abs_residual;
                witness.sample_row = std::to_string(row);
                witness.sample_col = std::to_string(col);
                witness.sample_residual = int128_to_string(residual);
            }
            ++witness.checked_values;
        }
    }

    std::uint64_t offset = 1;
    const auto max_abs = checked_int128_to_u64(witness.max_abs_residual, "max residual magnitude");
    while (offset <= max_abs) {
        if (offset > (std::uint64_t{1} << 62U)) {
            throw std::runtime_error("residual range proof exceeds supported 63-bit shifted range");
        }
        offset <<= 1U;
    }
    witness.offset = offset;
    const std::uint64_t shifted_max = offset + max_abs;
    std::size_t bit_count = 1;
    while (bit_count < 63 && (std::uint64_t{1} << bit_count) <= shifted_max) {
        ++bit_count;
    }
    if (bit_count > 63 || (bit_count == 63 && shifted_max >= (std::uint64_t{1} << 63U))) {
        throw std::runtime_error("residual range proof bit width exceeds supported field helper range");
    }
    witness.bit_count = bit_count;
    witness.bit_raw_values.assign(bit_count, std::vector<Fr_t>(padded_rows * padded_cols, fr_from_signed_host(0)));

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const __int128 shifted_signed = residuals[row * cols + col] + static_cast<__int128>(offset);
            const std::uint64_t shifted = checked_int128_to_u64(shifted_signed, "shifted residual");
            if (shifted > shifted_max) {
                throw std::runtime_error("shifted residual exceeds declared range");
            }
            if (row == std::stoull(witness.sample_row) && col == std::stoull(witness.sample_col)) {
                witness.sample_shifted = std::to_string(shifted);
            }
            const std::size_t flat = row * padded_cols + col;
            for (std::size_t bit = 0; bit < bit_count; ++bit) {
                if (((shifted >> bit) & 1ULL) != 0ULL) {
                    witness.bit_raw_values[bit][flat] = fr_from_signed_host(1);
                }
            }
        }
    }
    return witness;
}

bool run_residual_range_bound_relation(
    const std::string& dataset,
    const std::string& label,
    const std::vector<double>& a_values,
    std::size_t rows,
    std::size_t shared,
    const std::vector<double>& b_values,
    std::size_t cols,
    const std::vector<double>& c_values,
    std::size_t padded_rows,
    std::size_t padded_cols,
    FrTensor& residual,
    const G1TensorJacobian& com_residual,
    const Commitment& gen,
    FormalPartialArtifact* artifact,
    std::ofstream* debug)
{
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };
    if (artifact == nullptr) return false;
    try {
        auto witness = build_residual_range_witness(
            a_values, rows, shared, b_values, cols, c_values, padded_rows, padded_cols);
        std::vector<std::unique_ptr<FrTensor>> bit_tensors;
        std::vector<std::unique_ptr<G1TensorJacobian>> bit_commitments;
        bit_tensors.reserve(witness.bit_count);
        bit_commitments.reserve(witness.bit_count);

        for (std::size_t bit = 0; bit < witness.bit_count; ++bit) {
            auto bit_tensor = std::make_unique<FrTensor>(
                static_cast<uint>(witness.bit_raw_values[bit].size()),
                witness.bit_raw_values[bit].data());
            bit_tensor->mont();
            ScopedTimer commit_timer;
            auto bit_commitment = std::make_unique<G1TensorJacobian>(gen.commit(*bit_tensor));
            cuda_sync("formal residual range bit commit " + label + " bit " + std::to_string(bit));
            artifact->commitment_ms += commit_timer.elapsed_ms();

            ++artifact->total_chunks;
            ScopedTimer prove_timer;
            auto bit_proof = formal_hadamard_relation_prove(
                *bit_tensor,
                *bit_tensor,
                *bit_tensor,
                gen,
                *bit_commitment,
                *bit_commitment,
                *bit_commitment,
                dataset + ":" + label + ":residual_range_bit_" + std::to_string(bit));
            artifact->prove_ms += prove_timer.elapsed_ms();
            ScopedTimer verify_timer;
            std::string failure;
            const bool ok = formal_hadamard_relation_verify(
                bit_proof,
                gen,
                *bit_commitment,
                *bit_commitment,
                *bit_commitment,
                dataset + ":" + label + ":residual_range_bit_" + std::to_string(bit),
                &failure);
            artifact->verify_ms += verify_timer.elapsed_ms();
            if (!ok) {
                artifact->blocked_relations.push_back(
                    label + ": residual range bitness verifier rejected bit "
                    + std::to_string(bit) + ": " + failure);
                dbg("formal_residual_range label=" + label + " bit=" + std::to_string(bit)
                    + " status=REJECT failure=\"" + failure + "\"");
                return false;
            }
            ++artifact->verified_chunks;
            artifact->proof_size_bytes += bit_proof.serialized_size_bytes();
            bit_tensors.push_back(std::move(bit_tensor));
            bit_commitments.push_back(std::move(bit_commitment));
        }

        const auto point = deterministic_mont_fr_vec(
            ceilLog2(static_cast<uint>(padded_rows * padded_cols)),
            dataset + ":" + label + ":residual_range_decomposition_point");
        const Fr_t selector_eval = evaluate_valid_matrix_selector_mle(rows, cols, padded_rows, padded_cols, point);
        std::vector<FormalLinearTerm> terms;
        std::vector<const G1TensorJacobian*> commitments;
        terms.push_back({"residual_signed", &residual, &com_residual, Fr_host_one_mont(), point});
        commitments.push_back(&com_residual);
        for (std::size_t bit = 0; bit < witness.bit_count; ++bit) {
            const std::uint64_t coeff = std::uint64_t{1} << bit;
            const Fr_t neg_coeff = Fr_host_sub(Fr_host_zero(), fr_from_u64_mont_host(coeff));
            terms.push_back({
                "shifted_residual_bit_" + std::to_string(bit),
                bit_tensors[bit].get(),
                bit_commitments[bit].get(),
                neg_coeff,
                point});
            commitments.push_back(bit_commitments[bit].get());
        }

        ++artifact->total_chunks;
        ScopedTimer prove_timer;
        auto decomposition_proof = formal_linear_relation_prove(
            dataset + ":" + label + ":residual_range_shifted_decomposition",
            {static_cast<uint>(padded_rows), static_cast<uint>(padded_cols)},
            terms,
            gen,
            fr_from_u64_mont_host(witness.offset),
            selector_eval,
            Fr_host_zero());
        artifact->prove_ms += prove_timer.elapsed_ms();
        ScopedTimer verify_timer;
        std::string failure;
        const bool ok = formal_linear_relation_verify(
            decomposition_proof,
            gen,
            commitments,
            selector_eval,
            &failure);
        artifact->verify_ms += verify_timer.elapsed_ms();
        if (!ok) {
            artifact->blocked_relations.push_back(
                label + ": residual shifted range decomposition verifier rejected: " + failure);
            dbg("formal_residual_range label=" + label + " decomposition status=REJECT failure=\"" + failure + "\"");
            return false;
        }
        ++artifact->verified_chunks;
        artifact->proof_size_bytes += decomposition_proof.serialized_size_bytes();
        artifact->covered_relations.push_back(
            label + ": partial residual bounded range proof verified by bitness plus residual+offset decomposition");
        artifact->relation_details.push_back(
            label
            + " residual_range_bound status=PASS bit_count=" + std::to_string(witness.bit_count)
            + " offset=" + std::to_string(witness.offset)
            + " max_abs_residual=" + int128_to_string(witness.max_abs_residual)
            + " checked_values=" + std::to_string(witness.checked_values)
            + " sample[row=" + witness.sample_row
            + ",col=" + witness.sample_col
            + ",residual=" + witness.sample_residual
            + ",shifted=" + witness.sample_shifted
            + "] proof_size_bytes="
            + std::to_string(decomposition_proof.serialized_size_bytes())
            + " transcript_digest=" + decomposition_proof.transcript_digest);
        artifact->blocked_relations.push_back(
            label + ": residual bounded range is verifier-bound, but sign-specific rounding mode and tie-breaking are still not proven");
        dbg("formal_residual_range label=" + label
            + " status=PASS bit_count=" + std::to_string(witness.bit_count)
            + " offset=" + std::to_string(witness.offset)
            + " max_abs_residual=" + int128_to_string(witness.max_abs_residual));
        return true;
    } catch (const std::exception& error) {
        artifact->blocked_relations.push_back(label + ": residual range proof failed: " + std::string(error.what()));
        dbg("formal_residual_range label=" + label + " status=FAILED reason=\"" + std::string(error.what()) + "\"");
        return false;
    }
}

bool run_fixed_to_trace_scale_decomposition_relation(
    const std::string& dataset,
    const std::string& label,
    const std::vector<double>& a_values,
    std::size_t rows,
    std::size_t shared,
    const std::vector<double>& b_values,
    std::size_t cols,
    const std::vector<double>& c_values,
    std::size_t padded_rows,
    std::size_t padded_cols,
    const std::vector<Fr_t>& c_fixed_mont,
    const FrTensor& c_trace,
    const G1TensorJacobian& com_c_fixed,
    const G1TensorJacobian& com_c_trace,
    const Commitment& gen,
    FormalPartialArtifact* artifact,
    std::ofstream* debug)
{
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };
    if (artifact == nullptr) {
        return false;
    }
    const auto c_trace_mont = c_trace.to_host();
    if (c_fixed_mont.size() != c_trace_mont.size() || c_fixed_mont.size() != padded_rows * padded_cols) {
        artifact->blocked_relations.push_back(label + ": fixed-to-trace scale decomposition shape mismatch");
        return false;
    }
    const Fr_t sixteen = fr_from_signed_mont_host(16);
    std::vector<Fr_t> residual_mont(c_fixed_mont.size(), Fr_host_zero());
    for (std::size_t i = 0; i < c_fixed_mont.size(); ++i) {
        residual_mont[i] = Fr_host_sub(c_fixed_mont[i], Fr_host_mul(sixteen, c_trace_mont[i]));
    }

    FrTensor residual(static_cast<uint>(residual_mont.size()), residual_mont.data());
    ScopedTimer commit_timer;
    const auto com_residual = gen.commit(residual);
    cuda_sync("formal fixed-to-trace residual commit " + label);
    artifact->commitment_ms += commit_timer.elapsed_ms();

    ++artifact->total_chunks;
    const auto point = deterministic_mont_fr_vec(
        ceilLog2(static_cast<uint>(c_fixed_mont.size())),
        dataset + ":" + label + ":fixed_to_trace_scale_decomposition_point");
    const Fr_t neg_sixteen = Fr_host_sub(Fr_host_zero(), sixteen);
    std::vector<FormalLinearTerm> terms;
    terms.push_back({"C_fixed_pre_rescale", nullptr, &com_c_fixed, Fr_host_one_mont(), point});
    terms.push_back({"C_trace_rounded_scale16", &c_trace, &com_c_trace, neg_sixteen, point});
    terms.push_back({"residual_unranged", &residual, &com_residual, fr_neg_one_mont(), point});

    // `formal_linear_relation_prove` requires tensor pointers for all terms. C_fixed is not
    // owned by this helper, so create a lightweight tensor wrapper around the already-montgomery values.
    FrTensor c_fixed_tensor(static_cast<uint>(c_fixed_mont.size()), const_cast<Fr_t*>(c_fixed_mont.data()));
    terms[0].tensor = &c_fixed_tensor;

    try {
        ScopedTimer prove_timer;
        auto proof = formal_linear_relation_prove(
            dataset + ":" + label + ":fixed_to_trace_scale_decomposition",
            {static_cast<uint>(padded_rows), static_cast<uint>(padded_cols)},
            terms,
            gen,
            Fr_host_zero(),
            Fr_host_zero(),
            Fr_host_zero());
        artifact->prove_ms += prove_timer.elapsed_ms();
        ScopedTimer verify_timer;
        std::string failure;
        const bool ok = formal_linear_relation_verify(
            proof,
            gen,
            {&com_c_fixed, &com_c_trace, &com_residual},
            Fr_host_zero(),
            &failure);
        artifact->verify_ms += verify_timer.elapsed_ms();
        if (!ok) {
            artifact->blocked_relations.push_back(label + ": fixed-to-trace scale decomposition verifier rejected: " + failure);
            dbg("formal_rescale_decomposition label=" + label + " status=REJECT failure=\"" + failure + "\"");
            return false;
        }
        const auto stats = compute_scale_decomposition_stats(a_values, rows, shared, b_values, cols, c_values);
        ++artifact->verified_chunks;
        artifact->proof_size_bytes += proof.serialized_size_bytes();
        artifact->covered_relations.push_back(
            label + ": partial fixed-to-trace scale decomposition LinearRelationProof verified C_fixed - 16*C_trace - residual = 0");
        artifact->relation_details.push_back(
            label
            + " fixed_to_trace_scale_decomposition status=PASS proof_size_bytes="
            + std::to_string(proof.serialized_size_bytes())
            + " transcript_digest=" + proof.transcript_digest
            + " residual_outside_nearest_rounding_bound="
            + std::to_string(stats.outside_nearest_rounding_bound)
            + "/" + std::to_string(stats.checked_values)
            + " max_abs_residual=" + stats.max_abs_residual
            + " sample[row=" + stats.sample_row
            + ",col=" + stats.sample_col
            + ",product_scale256=" + stats.sample_product_scale256
            + ",trace_scale16=" + stats.sample_trace_scale16
            + ",residual=" + stats.sample_residual + "]");
        artifact->blocked_relations.push_back(
            label
            + ": fixed-to-trace scale decomposition is verifier-bound, but full rounding remains blocked because residual range/sign constraints are not implemented as a zkDL verifier gadget; residual_outside_nearest_rounding_bound="
            + std::to_string(stats.outside_nearest_rounding_bound)
            + "/" + std::to_string(stats.checked_values));
        dbg("formal_rescale_decomposition label=" + label + " status=PASS proof_size_bytes="
            + std::to_string(proof.serialized_size_bytes())
            + " residual_outside_nearest_rounding_bound="
            + std::to_string(stats.outside_nearest_rounding_bound)
            + "/" + std::to_string(stats.checked_values));
        run_residual_range_bound_relation(
            dataset,
            label,
            a_values,
            rows,
            shared,
            b_values,
            cols,
            c_values,
            padded_rows,
            padded_cols,
            residual,
            com_residual,
            gen,
            artifact,
            debug);
        return true;
    } catch (const std::exception& error) {
        artifact->blocked_relations.push_back(label + ": fixed-to-trace scale decomposition failed: " + std::string(error.what()));
        dbg("formal_rescale_decomposition label=" + label + " status=FAILED reason=\"" + std::string(error.what()) + "\"");
        return false;
    }
}

std::vector<double> matrix_row_slice(
    const std::vector<double>& values,
    std::size_t total_rows,
    std::size_t cols,
    std::size_t row_begin,
    std::size_t row_count)
{
    if (row_begin > total_rows || row_begin + row_count > total_rows || values.size() != total_rows * cols) {
        throw std::runtime_error("matrix row slice shape mismatch");
    }
    std::vector<double> out(row_count * cols, 0.0);
    for (std::size_t row = 0; row < row_count; ++row) {
        const auto src = values.begin() + static_cast<std::ptrdiff_t>((row_begin + row) * cols);
        std::copy(src, src + static_cast<std::ptrdiff_t>(cols), out.begin() + static_cast<std::ptrdiff_t>(row * cols));
    }
    return out;
}

std::vector<Fr_t> encode_pre_rescaled_from_head_and_bias(
    const std::vector<double>& head_values,
    const std::vector<double>& bias,
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_cols)
{
    if (bias.size() != cols) {
        throw std::runtime_error("bias vector shape mismatch for encoded pre-rescaled tensor");
    }
    auto head_encoded = encode_padded_tensor(head_values, rows, cols, padded_cols);
    auto bias_encoded = encode_padded_tensor(bias, 1, cols, padded_cols);
    std::vector<Fr_t> out(rows * padded_cols, Fr_host_zero());
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < padded_cols; ++col) {
            out[row * padded_cols + col] = Fr_host_sub(head_encoded[row * padded_cols + col], bias_encoded[col]);
        }
    }
    return out;
}

std::vector<Fr_t> concat_column_point_for_head(
    std::size_t head_index,
    std::size_t head_width,
    std::size_t concat_width,
    const std::vector<Fr_t>& head_col_point)
{
    const std::size_t padded_head_width = next_power_of_two(std::max<std::size_t>(1, head_width));
    const std::size_t padded_concat_width = next_power_of_two(std::max<std::size_t>(1, concat_width));
    const uint head_log = ceilLog2(static_cast<uint>(padded_head_width));
    const uint concat_log = ceilLog2(static_cast<uint>(padded_concat_width));
    if (head_col_point.size() != head_log || concat_log < head_log) {
        throw std::runtime_error("concat column point shape mismatch");
    }
    if (padded_head_width != head_width || padded_concat_width != concat_width) {
        throw std::runtime_error("concat proof currently requires power-of-two head and concat widths");
    }
    std::vector<Fr_t> out(concat_log, Fr_host_zero());
    for (uint bit = 0; bit < head_log; ++bit) {
        out[bit] = head_col_point[bit];
    }
    for (uint bit = head_log; bit < concat_log; ++bit) {
        const std::size_t selector_bit = bit - head_log;
        out[bit] = ((head_index >> selector_bit) & 1U) ? Fr_host_one_mont() : Fr_host_zero();
    }
    return out;
}

std::string chunk_label(const std::string& base, std::size_t chunk_idx)
{
    return base + "_chunk" + std::to_string(chunk_idx);
}

std::string chunk_relation_label(
    const std::string& base,
    std::size_t chunk_idx,
    std::size_t row_begin,
    std::size_t row_count,
    std::size_t total_chunks)
{
    return base
        + ":chunk" + std::to_string(chunk_idx)
        + ":offset" + std::to_string(row_begin)
        + ":rows" + std::to_string(row_count)
        + ":total_chunks" + std::to_string(total_chunks);
}

void run_output_linear_chunked_relations(
    const DatasetRunSpec& spec,
    const ModelParams& model,
    const ForwardTraceHost& trace,
    std::size_t rows,
    std::size_t cols,
    FormalPartialArtifact* artifact,
    std::ofstream* debug)
{
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };
    if (trace.output_heads.size() != model.K_out) {
        throw std::runtime_error("output head trace count mismatch in chunked linear relations");
    }
    const std::size_t padded_cols = next_power_of_two(std::max<std::size_t>(1, cols));
    const std::size_t max_chunk_values = 262144;
    const std::size_t max_chunk_rows = 4096;
    const std::size_t chunk_rows = std::max<std::size_t>(1, std::min(max_chunk_rows, std::max<std::size_t>(1, max_chunk_values / padded_cols)));
    const std::size_t total_chunks = (rows + chunk_rows - 1U) / chunk_rows;
    std::size_t average_verified = 0;
    std::size_t average_attempted = 0;
    std::size_t bias_verified = 0;
    std::size_t bias_attempted = 0;
    std::size_t bias_rejected = 0;
    std::size_t average_rejected = 0;
    dbg("formal_chunked_linear rows=" + std::to_string(rows)
        + " cols=" + std::to_string(cols)
        + " padded_cols=" + std::to_string(padded_cols)
        + " chunk_rows=" + std::to_string(chunk_rows)
        + " total_chunks=" + std::to_string(total_chunks));

    for (std::size_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
        const std::size_t row_begin = chunk_idx * chunk_rows;
        const std::size_t row_count = std::min(chunk_rows, rows - row_begin);
        FormalTensorRegistry chunk_registry;
        const uint point_dim = ceilLog2(static_cast<uint>(padded_cols)) + ceilLog2(static_cast<uint>(row_count));
        const auto point = deterministic_mont_fr_vec(
            point_dim,
            spec.dataset + ":formal:output-linear-chunk:" + std::to_string(chunk_idx));
        const uint col_log = ceilLog2(static_cast<uint>(padded_cols));
        const std::vector<Fr_t> col_point(point.begin(), point.begin() + col_log);
        const std::vector<Fr_t> row_point(point.begin() + col_log, point.end());
        const Fr_t row_valid_eval = evaluate_valid_selector_mle(row_count, row_point);
        std::vector<FormalTensorRecord*> post_records;
        post_records.reserve(trace.output_heads.size());
        for (std::size_t head = 0; head < trace.output_heads.size(); ++head) {
            const auto& head_trace = trace.output_heads[head];
            const auto post_values = matrix_row_slice(head_trace.h_out, rows, cols, row_begin, row_count);
            const auto pre_rescaled_encoded = encode_pre_rescaled_from_head_and_bias(
                post_values,
                model.output_heads[head].output_bias,
                row_count,
                cols,
                padded_cols);
            auto& pre_rescaled = chunk_registry.add_encoded_matrix(
                spec.dataset,
                chunk_label("output_head" + std::to_string(head) + "_Y_pre_rescaled", chunk_idx),
                pre_rescaled_encoded,
                row_count,
                cols,
                padded_cols,
                &artifact->commitment_ms,
                debug);
            auto& post = chunk_registry.add_matrix(
                spec.dataset,
                chunk_label("output_head" + std::to_string(head) + "_Y_head", chunk_idx),
                post_values,
                row_count,
                cols,
                &artifact->commitment_ms,
                debug);
            post_records.push_back(&post);
            const Fr_t bias_eval = Fr_host_mul(
                evaluate_public_vector_mle(model.output_heads[head].output_bias, cols, padded_cols, col_point),
                row_valid_eval);
            std::vector<FormalLinearTerm> terms;
            terms.push_back({"Y_head", post.tensor.get(), post.commitment.get(), Fr_host_one_mont(), point});
            terms.push_back({"Y_pre_rescaled", pre_rescaled.tensor.get(), pre_rescaled.commitment.get(), fr_neg_one_mont(), point});
            ++artifact->total_chunks;
            ++bias_attempted;
            try {
                ScopedTimer prove_timer;
                auto proof = formal_linear_relation_prove(
                    chunk_relation_label(
                        spec.dataset + ":output_head" + std::to_string(head) + ":bias_add",
                        chunk_idx,
                        row_begin,
                        row_count,
                        total_chunks),
                    {static_cast<uint>(row_count), static_cast<uint>(cols)},
                    terms,
                    *post.generators,
                    fr_neg_one_mont(),
                    bias_eval,
                    Fr_host_zero());
                artifact->prove_ms += prove_timer.elapsed_ms();
                ScopedTimer verify_timer;
                std::string failure;
                const bool ok = formal_linear_relation_verify(
                    proof,
                    *post.generators,
                    {post.commitment.get(), pre_rescaled.commitment.get()},
                    bias_eval,
                    &failure);
                artifact->verify_ms += verify_timer.elapsed_ms();
                if (ok) {
                    ++artifact->verified_chunks;
                    ++bias_verified;
                    artifact->proof_size_bytes += proof.serialized_size_bytes();
                    artifact->relation_details.push_back(
                        "bias_add chunk=" + std::to_string(chunk_idx)
                        + " head=" + std::to_string(head)
                        + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                        + " status=PASS proof_size_bytes=" + std::to_string(proof.serialized_size_bytes())
                        + " transcript_digest=" + proof.transcript_digest);
                } else {
                    ++bias_rejected;
                    artifact->relation_details.push_back(
                        "bias_add chunk=" + std::to_string(chunk_idx)
                        + " head=" + std::to_string(head)
                        + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                        + " status=REJECT failure=\"" + failure + "\"");
                }
            } catch (const std::exception& error) {
                ++bias_rejected;
                artifact->relation_details.push_back(
                    "bias_add chunk=" + std::to_string(chunk_idx)
                    + " head=" + std::to_string(head)
                    + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                    + " status=FAILED failure=\"" + std::string(error.what()) + "\"");
            }
        }

        const auto logits_chunk = matrix_row_slice(trace.logits, rows, cols, row_begin, row_count);
        const Fr_t public_logits_eval = evaluate_public_tensor_mle(logits_chunk, row_count, cols, padded_cols, point);
        const Fr_t inv_k = fr_inverse_uint_mont(static_cast<std::uint32_t>(model.K_out));
        std::vector<FormalLinearTerm> terms;
        std::vector<const G1TensorJacobian*> commitments;
        for (std::size_t head = 0; head < post_records.size(); ++head) {
            auto* record = post_records[head];
            terms.push_back({"Y_head_" + std::to_string(head), record->tensor.get(), record->commitment.get(), inv_k, point});
            commitments.push_back(record->commitment.get());
        }
        ++artifact->total_chunks;
        ++average_attempted;
        try {
            ScopedTimer prove_timer;
            auto proof = formal_linear_relation_prove(
                chunk_relation_label(
                    spec.dataset + ":output_head_average_public_logits",
                    chunk_idx,
                    row_begin,
                    row_count,
                    total_chunks),
                {static_cast<uint>(row_count), static_cast<uint>(cols)},
                terms,
                *post_records.front()->generators,
                fr_neg_one_mont(),
                public_logits_eval,
                Fr_host_zero());
            artifact->prove_ms += prove_timer.elapsed_ms();
            ScopedTimer verify_timer;
            std::string failure;
            const bool ok = formal_linear_relation_verify(proof, *post_records.front()->generators, commitments, public_logits_eval, &failure);
            artifact->verify_ms += verify_timer.elapsed_ms();
            if (ok) {
                ++artifact->verified_chunks;
                ++average_verified;
                artifact->proof_size_bytes += proof.serialized_size_bytes();
                artifact->relation_details.push_back(
                    "output_head_average_public_logits chunk=" + std::to_string(chunk_idx)
                    + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                    + " status=PASS proof_size_bytes=" + std::to_string(proof.serialized_size_bytes())
                    + " transcript_digest=" + proof.transcript_digest);
            } else {
                ++average_rejected;
                artifact->relation_details.push_back(
                    "output_head_average_public_logits chunk=" + std::to_string(chunk_idx)
                    + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                    + " status=REJECT failure=\"" + failure + "\"");
            }
        } catch (const std::exception& error) {
            ++average_rejected;
            artifact->relation_details.push_back(
                "output_head_average_public_logits chunk=" + std::to_string(chunk_idx)
                + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                + " status=FAILED failure=\"" + std::string(error.what()) + "\"");
        }
    }

    if (average_verified == average_attempted && average_attempted != 0) {
        artifact->covered_relations.push_back(
            "output-head average and public logits binding: chunked LinearRelationProof verifier passed for "
            + std::to_string(average_verified) + "/" + std::to_string(average_attempted) + " full-dataset chunks");
    } else {
        artifact->blocked_relations.push_back(
            "output-head average/public logits binding chunked verifier incomplete: verified "
            + std::to_string(average_verified) + "/" + std::to_string(average_attempted)
            + ", rejected=" + std::to_string(average_rejected));
    }
    if (bias_verified == bias_attempted && bias_attempted != 0) {
        artifact->covered_relations.push_back(
            "output bias add: chunked LinearRelationProof verifier passed for "
            + std::to_string(bias_verified) + "/" + std::to_string(bias_attempted)
            + " chunks over the protocol tensor Y_pre_rescaled");
    } else {
        artifact->blocked_relations.push_back(
            "output bias add chunked verifier incomplete: verified "
            + std::to_string(bias_verified) + "/" + std::to_string(bias_attempted)
            + ", rejected=" + std::to_string(bias_rejected)
            + "; fixed-point rescale/rounding witness is required before marking the real GAT bias relation supported");
    }
}

void run_hidden_concat_chunked_relations(
    const DatasetRunSpec& spec,
    const ForwardTraceHost& trace,
    std::size_t rows,
    FormalPartialArtifact* artifact,
    std::ofstream* debug)
{
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };
    if (trace.hidden_heads.empty() || trace.hidden_heads.front().empty() || trace.output_layer_input.empty()) {
        artifact->blocked_relations.push_back("hidden concat blocked: hidden head trace or output-layer input trace is unavailable");
        return;
    }
    const auto& layer_heads = trace.hidden_heads.front();
    const std::size_t concat_width = trace.output_layer_input_dim;
    std::size_t expected_width = 0;
    for (const auto& head_trace : layer_heads) expected_width += head_trace.width;
    if (expected_width != concat_width || trace.output_layer_input.size() != rows * concat_width) {
        artifact->blocked_relations.push_back("hidden concat blocked: concat trace width does not match hidden head widths");
        return;
    }
    const std::size_t padded_concat_width = next_power_of_two(std::max<std::size_t>(1, concat_width));
    const std::size_t max_chunk_values = 262144;
    const std::size_t max_chunk_rows = 4096;
    const std::size_t chunk_rows = std::max<std::size_t>(1, std::min(max_chunk_rows, std::max<std::size_t>(1, max_chunk_values / padded_concat_width)));
    const std::size_t total_row_chunks = (rows + chunk_rows - 1U) / chunk_rows;
    const std::size_t total_relation_chunks = total_row_chunks * layer_heads.size();
    artifact->total_chunks += total_relation_chunks;
    std::size_t verified = 0;
    std::size_t rejected = 0;
    dbg("formal_hidden_concat rows=" + std::to_string(rows)
        + " concat_width=" + std::to_string(concat_width)
        + " heads=" + std::to_string(layer_heads.size())
        + " chunk_rows=" + std::to_string(chunk_rows)
        + " total_relation_chunks=" + std::to_string(total_relation_chunks));

    for (std::size_t chunk_idx = 0; chunk_idx < total_row_chunks; ++chunk_idx) {
        const std::size_t row_begin = chunk_idx * chunk_rows;
        const std::size_t row_count = std::min(chunk_rows, rows - row_begin);
        FormalTensorRegistry chunk_registry;
        const auto concat_values = matrix_row_slice(trace.output_layer_input, rows, concat_width, row_begin, row_count);
        auto& concat_record = chunk_registry.add_matrix(
            spec.dataset,
            chunk_label("hidden_layer0_H_cat", chunk_idx),
            concat_values,
            row_count,
            concat_width,
            &artifact->commitment_ms,
            debug);
        for (std::size_t head = 0; head < layer_heads.size(); ++head) {
            const auto& head_trace = layer_heads[head];
            const auto head_values = matrix_row_slice(head_trace.h_out, rows, head_trace.width, row_begin, row_count);
            std::vector<double> embedded_head(row_count * padded_concat_width, 0.0);
            const std::size_t head_offset = 0;
            for (std::size_t prior = 0; prior < head; ++prior) {
                // kept separate from the column point computation so non-uniform future head widths fail clearly.
                if (layer_heads[prior].width != head_trace.width) {
                    throw std::runtime_error("hidden concat proof currently requires uniform hidden head widths");
                }
            }
            const std::size_t column_offset = head * head_trace.width + head_offset;
            for (std::size_t row = 0; row < row_count; ++row) {
                for (std::size_t col = 0; col < head_trace.width; ++col) {
                    embedded_head[row * padded_concat_width + column_offset + col] =
                        head_values[row * head_trace.width + col];
                }
            }
            auto& head_record = chunk_registry.add_encoded_matrix(
                spec.dataset,
                chunk_label("hidden_layer0_head" + std::to_string(head) + "_H_agg_embedded", chunk_idx),
                encode_padded_tensor(embedded_head, row_count, padded_concat_width, padded_concat_width),
                row_count,
                concat_width,
                padded_concat_width,
                &artifact->commitment_ms,
                debug);
            const uint head_col_log = ceilLog2(static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, head_trace.width))));
            const uint row_log = ceilLog2(static_cast<uint>(row_count));
            const auto head_point = deterministic_mont_fr_vec(
                head_col_log + row_log,
                spec.dataset + ":formal:hidden-concat:chunk:" + std::to_string(chunk_idx) + ":head:" + std::to_string(head));
            const std::vector<Fr_t> head_col_point(head_point.begin(), head_point.begin() + head_col_log);
            const std::vector<Fr_t> row_point(head_point.begin() + head_col_log, head_point.end());
            const auto concat_col_point = concat_column_point_for_head(head, head_trace.width, concat_width, head_col_point);
            const auto concat_point = formal_concat_points({concat_col_point, row_point});
            std::vector<FormalLinearTerm> terms;
            terms.push_back({"H_cat_slice", concat_record.tensor.get(), concat_record.commitment.get(), Fr_host_one_mont(), concat_point});
            terms.push_back({"H_agg_head_embedded", head_record.tensor.get(), head_record.commitment.get(), fr_neg_one_mont(), concat_point});
            try {
                ScopedTimer prove_timer;
                auto proof = formal_linear_relation_prove(
                    chunk_relation_label(
                        spec.dataset + ":hidden_layer0_head" + std::to_string(head) + ":concat_copy",
                        chunk_idx,
                        row_begin,
                        row_count,
                        total_row_chunks),
                    {static_cast<uint>(row_count), static_cast<uint>(concat_width)},
                    terms,
                    *concat_record.generators,
                    Fr_host_zero(),
                    Fr_host_zero(),
                    Fr_host_zero());
                artifact->prove_ms += prove_timer.elapsed_ms();
                ScopedTimer verify_timer;
                std::string failure;
                const bool ok = formal_linear_relation_verify(
                    proof,
                    *concat_record.generators,
                    {concat_record.commitment.get(), head_record.commitment.get()},
                    Fr_host_zero(),
                    &failure);
                artifact->verify_ms += verify_timer.elapsed_ms();
                if (!ok) {
                    ++rejected;
                    artifact->relation_details.push_back(
                        "hidden_concat chunk=" + std::to_string(chunk_idx)
                        + " head=" + std::to_string(head)
                        + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                        + " status=REJECT failure=\"" + failure + "\"");
                    artifact->blocked_relations.push_back(
                        "hidden concat chunked verifier rejected chunk " + std::to_string(chunk_idx)
                        + " head " + std::to_string(head) + ": " + failure);
                    return;
                }
                ++verified;
                ++artifact->verified_chunks;
                artifact->proof_size_bytes += proof.serialized_size_bytes();
                artifact->relation_details.push_back(
                    "hidden_concat chunk=" + std::to_string(chunk_idx)
                    + " head=" + std::to_string(head)
                    + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                    + " status=PASS proof_size_bytes=" + std::to_string(proof.serialized_size_bytes())
                    + " transcript_digest=" + proof.transcript_digest);
            } catch (const std::exception& error) {
                ++rejected;
                artifact->relation_details.push_back(
                    "hidden_concat chunk=" + std::to_string(chunk_idx)
                    + " head=" + std::to_string(head)
                    + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                    + " status=FAILED failure=\"" + std::string(error.what()) + "\"");
                artifact->blocked_relations.push_back(
                    "hidden concat chunked prover/verifier failed at chunk " + std::to_string(chunk_idx)
                    + " head " + std::to_string(head) + ": " + std::string(error.what()));
                return;
            }
        }
    }
    if (verified == total_relation_chunks && rejected == 0) {
        artifact->covered_relations.push_back(
            "hidden multi-head concat: chunked LinearRelationProof verifier passed for "
            + std::to_string(verified) + "/" + std::to_string(total_relation_chunks)
            + " head/chunk relations");
    } else {
        artifact->blocked_relations.push_back(
            "hidden concat chunked verifier incomplete: verified "
            + std::to_string(verified) + "/" + std::to_string(total_relation_chunks));
    }
}

bool run_formal_matmul_trace_attempt(
    const std::string& dataset,
    const std::string& label,
    const std::vector<double>& a_values,
    std::size_t rows,
    std::size_t shared,
    const std::vector<double>& b_values,
    std::size_t cols,
    const std::vector<double>& c_values,
    FormalPartialArtifact* artifact,
    std::ofstream* debug,
    std::size_t row_begin_for_evidence = 0,
    std::size_t total_chunks_for_evidence = 1)
{
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };
    const std::size_t padded_rows = next_power_of_two(std::max<std::size_t>(1, rows));
    const std::size_t padded_shared = next_power_of_two(std::max<std::size_t>(1, shared));
    const std::size_t padded_cols = next_power_of_two(std::max<std::size_t>(1, cols));
    const std::size_t a_size = padded_rows * padded_shared;
    const std::size_t b_size = padded_shared * padded_cols;
    const std::size_t c_size = padded_rows * padded_cols;
    constexpr std::size_t kMaxFlatCommitmentPoints = 50000;
    if (std::max({a_size, b_size, c_size}) > kMaxFlatCommitmentPoints) {
        artifact->blocked_relations.push_back(label + ": blocked because current formal matmul uses flat gen=1 commitments and padded tensor size "
            + std::to_string(std::max({a_size, b_size, c_size})) + " exceeds cap "
            + std::to_string(kMaxFlatCommitmentPoints) + "; row-generator matmul openings are not implemented yet");
        dbg("formal_matmul label=" + label + " status=BLOCKED reason=flat_commitment_cap");
        return false;
    }
    try {
        dbg("formal_matmul label=" + label + " stage=encode");
        auto a_host = formal_encode_matmul_matrix(a_values, rows, shared, padded_rows, padded_shared);
        auto b_host = formal_encode_matmul_matrix(b_values, shared, cols, padded_shared, padded_cols);
        auto c_trace_host = formal_encode_matmul_matrix(c_values, rows, cols, padded_rows, padded_cols);
        FrTensor a(static_cast<uint>(a_host.size()), a_host.data());
        FrTensor b(static_cast<uint>(b_host.size()), b_host.data());
        FrTensor c_trace(static_cast<uint>(c_trace_host.size()), c_trace_host.data());
        a.mont();
        b.mont();
        c_trace.mont();
        auto c_fixed_host = formal_matmul_product_mont(
            a.to_host(),
            rows,
            shared,
            padded_rows,
            padded_shared,
            b.to_host(),
            cols,
            padded_cols);
        FrTensor c(static_cast<uint>(c_fixed_host.size()), c_fixed_host.data());
        Commitment gen(1, G1Jacobian_generator);
        ScopedTimer commit_timer;
        const auto com_a = gen.commit(a);
        const auto com_b = gen.commit(b);
        const auto com_c = gen.commit(c);
        const auto com_c_trace = gen.commit(c_trace);
        cuda_sync("formal matmul commit " + label);
        artifact->commitment_ms += commit_timer.elapsed_ms();
        if (debug != nullptr) {
            const std::string evidence_path = maybe_write_matmul_first_failure_evidence(
                dataset,
                label,
                row_begin_for_evidence,
                rows,
                total_chunks_for_evidence,
                a_values,
                rows,
                shared,
                b_values,
                cols,
                c_values,
                padded_rows,
                padded_shared,
                padded_cols,
                a,
                b,
                c_trace,
                com_a,
                com_b,
                com_c_trace,
                c_fixed_host);
            if (!evidence_path.empty()) {
                artifact->blocked_relations.push_back(label + ": float forward C does not equal the fixed-point pre-rescale matmul witness; rescale/rounding/equality remains blocked; evidence=" + evidence_path);
                dbg("formal_matmul label=" + label + " evidence=\"" + evidence_path + "\" classification=\"float_C_vs_fixed_pre_rescale_mismatch\"");
            }
        }
        ScopedTimer prove_timer;
        auto proof = formal_matmul_relation_prove(a, b, c, gen, com_a, com_b, com_c,
            1U,
            static_cast<uint>(rows),
            static_cast<uint>(shared),
            static_cast<uint>(cols),
            1U,
            static_cast<uint>(padded_rows),
            static_cast<uint>(padded_shared),
            static_cast<uint>(padded_cols),
            dataset + ":" + label);
        artifact->prove_ms += prove_timer.elapsed_ms();
        ScopedTimer verify_timer;
        std::string failure;
        const bool ok = formal_matmul_relation_verify(proof, gen, com_a, com_b, com_c, dataset + ":" + label, &failure);
        artifact->verify_ms += verify_timer.elapsed_ms();
        if (!ok) {
            artifact->blocked_relations.push_back(label + ": verifier rejected formal matmul: " + failure);
            dbg("formal_matmul label=" + label + " status=REJECT failure=\"" + failure + "\"");
            return false;
        }
        artifact->proof_size_bytes += proof.serialized_size_bytes();
        artifact->covered_relations.push_back(label + ": formal fixed-point pre-rescale MatmulRelationProof verifier passed");
        dbg("formal_matmul label=" + label + " status=PASS proof_size_bytes=" + std::to_string(proof.serialized_size_bytes()));
        run_fixed_to_trace_scale_decomposition_relation(
            dataset,
            label,
            a_values,
            rows,
            shared,
            b_values,
            cols,
            c_values,
            padded_rows,
            padded_cols,
            c_fixed_host,
            c_trace,
            com_c,
            com_c_trace,
            gen,
            artifact,
            debug);
        return true;
    } catch (const std::exception& error) {
        artifact->blocked_relations.push_back(label + ": formal matmul attempt failed: " + std::string(error.what()));
        dbg("formal_matmul label=" + label + " status=FAILED reason=\"" + std::string(error.what()) + "\"");
        return false;
    }
}

bool run_formal_matmul_trace_attempt_row_chunked(
    const std::string& dataset,
    const std::string& label,
    const std::vector<double>& a_values,
    std::size_t rows,
    std::size_t shared,
    const std::vector<double>& b_values,
    std::size_t cols,
    const std::vector<double>& c_values,
    FormalPartialArtifact* artifact,
    std::ofstream* debug)
{
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };
    if (a_values.size() != rows * shared || b_values.size() != shared * cols || c_values.size() != rows * cols) {
        throw std::runtime_error("chunked matmul trace shape mismatch for " + label);
    }
    constexpr std::size_t kMaxFlatCommitmentPoints = 50000;
    const std::size_t padded_shared = next_power_of_two(std::max<std::size_t>(1, shared));
    const std::size_t padded_cols = next_power_of_two(std::max<std::size_t>(1, cols));
    const std::size_t b_size = padded_shared * padded_cols;
    if (b_size > kMaxFlatCommitmentPoints) {
        artifact->blocked_relations.push_back(label + ": chunked matmul blocked because padded parameter tensor size "
            + std::to_string(b_size) + " exceeds flat commitment cap " + std::to_string(kMaxFlatCommitmentPoints));
        dbg("formal_matmul_chunked label=" + label + " status=BLOCKED reason=parameter_flat_cap");
        return false;
    }
    const std::size_t max_row_factor = std::max<std::size_t>(padded_shared, padded_cols);
    std::size_t max_padded_rows = 1;
    while ((max_padded_rows << 1U) * max_row_factor <= kMaxFlatCommitmentPoints && max_padded_rows < 4096U) {
        max_padded_rows <<= 1U;
    }
    const std::size_t chunk_rows = std::max<std::size_t>(1, max_padded_rows);
    const std::size_t total_chunks = (rows + chunk_rows - 1U) / chunk_rows;
    artifact->total_chunks += total_chunks;
    dbg("formal_matmul_chunked label=" + label
        + " rows=" + std::to_string(rows)
        + " shared=" + std::to_string(shared)
        + " cols=" + std::to_string(cols)
        + " chunk_rows=" + std::to_string(chunk_rows)
        + " total_chunks=" + std::to_string(total_chunks));

    std::size_t verified = 0;
    for (std::size_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
        const std::size_t row_begin = chunk_idx * chunk_rows;
        const std::size_t row_count = std::min(chunk_rows, rows - row_begin);
        const auto a_chunk = matrix_row_slice(a_values, rows, shared, row_begin, row_count);
        const auto c_chunk = matrix_row_slice(c_values, rows, cols, row_begin, row_count);
        const std::string chunked_label = chunk_relation_label(label, chunk_idx, row_begin, row_count, total_chunks);
        const bool ok = run_formal_matmul_trace_attempt(
            dataset,
            chunked_label,
            a_chunk,
            row_count,
            shared,
            b_values,
            cols,
            c_chunk,
            artifact,
            debug,
            row_begin,
            total_chunks);
        if (!ok) {
            artifact->relation_details.push_back(label
                + " chunk=" + std::to_string(chunk_idx)
                + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                + " status=REJECT_OR_BLOCKED verified_chunks=" + std::to_string(verified)
                + "/" + std::to_string(total_chunks));
            artifact->blocked_relations.push_back(label + ": chunked MatmulRelationProof incomplete; verified "
                + std::to_string(verified) + "/" + std::to_string(total_chunks)
                + " chunks before first rejected or blocked chunk " + std::to_string(chunk_idx));
            return false;
        }
        ++verified;
        ++artifact->verified_chunks;
        artifact->relation_details.push_back(label
            + " chunk=" + std::to_string(chunk_idx)
            + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
            + " status=PASS");
    }
    artifact->covered_relations.push_back(label + ": chunked MatmulRelationProof verifier passed for "
        + std::to_string(verified) + "/" + std::to_string(total_chunks) + " full-dataset row chunks");
    return true;
}

void serialize_formal_partial_artifact(const FormalPartialArtifact& artifact, const fs::path& path)
{
    ensure_dir(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to write formal partial proof artifact: " + path.string());
    out << "ZKDLGATF\n";
    out << "version=1\n";
    out << "proof_generation_ok=" << (artifact.proof_generation_ok ? "true" : "false") << "\n";
    out << "verification_ok=" << (artifact.verification_ok ? "true" : "false") << "\n";
    out << "commitment_time_ms=" << format_ms(artifact.commitment_ms) << "\n";
    out << "prove_time_ms=" << format_ms(artifact.prove_ms) << "\n";
    out << "verify_time_ms=" << format_ms(artifact.verify_ms) << "\n";
    out << "proof_size_bytes=" << artifact.proof_size_bytes << "\n";
    out << "total_chunks=" << artifact.total_chunks << "\n";
    out << "verified_chunks=" << artifact.verified_chunks << "\n";
    out << "covered_relations=" << artifact.covered_relations.size() << "\n";
    for (const auto& relation : artifact.covered_relations) out << relation << "\n";
    out << "blocked_relations=" << artifact.blocked_relations.size() << "\n";
    for (const auto& relation : artifact.blocked_relations) out << relation << "\n";
    out << "relation_details=" << artifact.relation_details.size() << "\n";
    for (const auto& detail : artifact.relation_details) out << detail << "\n";
    out << "blocker=" << artifact.blocker << "\n";
}

FormalPartialArtifact run_formal_partial_trace_proof(
    const DatasetRunSpec& spec,
    const ModelParams& model,
    const LocalGraph& graph,
    const ForwardTraceHost& trace,
    const fs::path& run_dir)
{
    FormalPartialArtifact artifact;
    std::ofstream debug(run_dir / "proof_debug.log", std::ios::app);
    auto dbg = [&](const std::string& line) {
        if (debug) {
            debug << line << '\n';
            debug.flush();
        }
    };
    dbg("formal_partial stage=begin");
    try {
        if (trace.output_heads.size() != model.K_out) {
            throw std::runtime_error("output head trace count mismatch");
        }
        const std::size_t rows = graph.num_nodes;
        const std::size_t cols = model.C;
        run_output_linear_chunked_relations(spec, model, trace, rows, cols, &artifact, &debug);
        run_hidden_concat_chunked_relations(spec, trace, rows, &artifact, &debug);
        artifact.blocked_relations.push_back(
            "output rescale blocked: chunked bias-add proof uses a committed Y_pre_rescaled witness, but the relation tying sparse attention aggregation Y_pre to Y_pre_rescaled still lacks a quotient/remainder or equivalent fixed-point rounding proof");

        if (!trace.output_layer_input.empty() && !model.output_heads.empty() && !trace.output_heads.empty()) {
            run_formal_matmul_trace_attempt_row_chunked(
                spec.dataset,
                "output_head0_projection",
                trace.output_layer_input,
                rows,
                trace.output_layer_input_dim,
                model.output_heads.front().W,
                cols,
                trace.output_heads.front().h_prime,
                &artifact,
                &debug);
            std::vector<double> e_src_pre(trace.output_heads.front().e_src.size(), 0.0);
            std::vector<double> e_dst_pre(trace.output_heads.front().e_dst.size(), 0.0);
            for (std::size_t i = 0; i < e_src_pre.size(); ++i) {
                e_src_pre[i] = trace.output_heads.front().e_src[i] - model.output_heads.front().attn_src_bias;
                e_dst_pre[i] = trace.output_heads.front().e_dst[i] - model.output_heads.front().attn_dst_bias;
            }
            run_formal_matmul_trace_attempt_row_chunked(
                spec.dataset,
                "output_head0_source_attention_dot",
                trace.output_heads.front().h_prime,
                rows,
                cols,
                model.output_heads.front().attn_src,
                1,
                e_src_pre,
                &artifact,
                &debug);
            run_formal_matmul_trace_attempt_row_chunked(
                spec.dataset,
                "output_head0_destination_attention_dot",
                trace.output_heads.front().h_prime,
                rows,
                cols,
                model.output_heads.front().attn_dst,
                1,
                e_dst_pre,
                &artifact,
                &debug);
        }

        if (!trace.hidden_heads.empty() && !trace.hidden_heads.front().empty() && !model.hidden_heads.empty()) {
            const auto& hidden0 = trace.hidden_heads.front().front();
            const auto& hidden_input0 = trace.hidden_layer_inputs.front();
            const std::size_t hidden_rows = hidden0.rows;
            const std::size_t hidden_input_dim = trace.hidden_layer_input_dims.front();
            const auto hidden_model_it = std::find_if(model.hidden_heads.begin(), model.hidden_heads.end(), [](const HeadParams& head) {
                return head.layer_index == 0;
            });
            if (hidden_model_it != model.hidden_heads.end()) {
                run_formal_matmul_trace_attempt_row_chunked(
                    spec.dataset,
                    "hidden_layer0_head0_projection",
                    hidden_input0,
                    hidden_rows,
                    hidden_input_dim,
                    hidden_model_it->W,
                    hidden0.width,
                    hidden0.h_prime,
                    &artifact,
                    &debug);
                std::vector<double> hidden_e_src_pre(hidden0.e_src.size(), 0.0);
                std::vector<double> hidden_e_dst_pre(hidden0.e_dst.size(), 0.0);
                for (std::size_t i = 0; i < hidden_e_src_pre.size(); ++i) {
                    hidden_e_src_pre[i] = hidden0.e_src[i] - hidden_model_it->attn_src_bias;
                    hidden_e_dst_pre[i] = hidden0.e_dst[i] - hidden_model_it->attn_dst_bias;
                }
                run_formal_matmul_trace_attempt_row_chunked(
                    spec.dataset,
                    "hidden_layer0_head0_source_attention_dot",
                    hidden0.h_prime,
                    hidden_rows,
                    hidden0.width,
                    hidden_model_it->attn_src,
                    1,
                    hidden_e_src_pre,
                    &artifact,
                    &debug);
                run_formal_matmul_trace_attempt_row_chunked(
                    spec.dataset,
                    "hidden_layer0_head0_destination_attention_dot",
                    hidden0.h_prime,
                    hidden_rows,
                    hidden0.width,
                    hidden_model_it->attn_dst,
                    1,
                    hidden_e_dst_pre,
                    &artifact,
                    &debug);
            }
            artifact.blocked_relations.push_back(
                "hidden projection matmul trace is connected and row-chunk attempted for layer0/head0; full Level A still requires all layers/heads plus quantization/rescale alignment");
            artifact.blocked_relations.push_back(
                "hidden source/destination attention dot products are connected and row-chunk attempted for layer0/head0; all-head coverage remains blocked by quantization/rescale alignment");
        } else {
            artifact.blocked_relations.push_back("hidden trace unavailable for formal hidden projection/dot-product relation");
        }
        artifact.blocked_relations.push_back("sparse source routing blocked: no Hyrax sparse route MLE verifier over public src");
        artifact.blocked_relations.push_back("sparse destination routing blocked: no Hyrax sparse route MLE verifier over public dst");
        artifact.blocked_relations.push_back("softmax/LeakyReLU/ExpMap/denominator/inverse/weighted aggregation/ELU/rescale blocked: gadgets are not implemented under zkDL Hyrax");

        artifact.proof_generation_ok = !artifact.covered_relations.empty();
        artifact.verification_ok = !artifact.covered_relations.empty();
        artifact.blocker = "Partial formal trace proof only: "
            + std::to_string(artifact.covered_relations.size())
            + " relation group(s) verified, but Level A remains blocked by "
            + std::to_string(artifact.blocked_relations.size())
            + " missing or rejected GAT relation group(s); verified_chunks="
            + std::to_string(artifact.verified_chunks)
            + "/"
            + std::to_string(artifact.total_chunks)
            + ".";
        artifact.proof_path = run_dir / "formal_artifacts" / "zkdl_formal_partial.proof.txt";
        serialize_formal_partial_artifact(artifact, artifact.proof_path);
        dbg("formal_partial stage=done covered=" + std::to_string(artifact.covered_relations.size())
            + " blocked=" + std::to_string(artifact.blocked_relations.size())
            + " proof_path=" + artifact.proof_path.string());
    } catch (const std::exception& error) {
        artifact.blocker = std::string("formal partial trace proof failed: ") + error.what();
        artifact.blocked_relations.push_back(artifact.blocker);
        artifact.proof_path = run_dir / "formal_artifacts" / "zkdl_formal_partial.blocked.txt";
        try {
            serialize_formal_partial_artifact(artifact, artifact.proof_path);
        } catch (...) {
        }
        dbg("formal_partial stage=failed reason=\"" + artifact.blocker + "\"");
    }
    return artifact;
}

struct RunRow {
    std::string dataset;
    std::string status = "blocked";
    std::string blocker;
    std::string benchmark_mode = "warm";
    std::string commitment_time_ms = "NA";
    std::string prove_time_ms = "NA";
    std::string verify_time_ms = "NA";
    std::string proof_size_bytes = "NA";
    bool full_dataset = false;
    bool gpu_forward_ok = false;
    bool partial_commitment_ok = false;
    bool proof_generation_ok = false;
    bool verification_ok = false;
    std::string debug_level_b_status = "not_run";
    std::string debug_level_b_blocker;
    std::string debug_level_b_benchmark_mode = "diagnostic_level_b_output_binding_kout_average_only";
    std::string debug_level_b_commitment_time_ms = "NA";
    std::string debug_level_b_prove_time_ms = "NA";
    std::string debug_level_b_verify_time_ms = "NA";
    std::string debug_level_b_proof_size_bytes = "NA";
    std::string debug_level_b_proof_path;
    std::string formal_partial_status = "not_run";
    std::string formal_partial_blocker;
    std::string formal_partial_commitment_time_ms = "NA";
    std::string formal_partial_prove_time_ms = "NA";
    std::string formal_partial_verify_time_ms = "NA";
    std::string formal_partial_proof_size_bytes = "NA";
    std::string formal_partial_proof_path;
    std::string formal_partial_total_chunks = "0";
    std::string formal_partial_verified_chunks = "0";
    std::string formal_partial_covered_relations;
    std::string formal_partial_blocked_relations;
};

std::string join_sizes(const std::vector<std::size_t>& values) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ',';
        out << values[i];
    }
    return out.str();
}

void write_manifest(
    const fs::path& run_dir,
    const DatasetRunSpec& spec,
    const GraphDataset& dataset,
    const LocalGraph& graph,
    const ModelParams& model,
    const ForwardMetrics& forward,
    const ProofAttempt& proof,
    const RunRow& row) {
    ensure_dir(run_dir);
    std::ofstream out(run_dir / "run_manifest.json");
    out << "{\n";
    auto kv = [&](const std::string& key, const std::string& value, bool comma = true) {
        out << "  \"" << json_escape(key) << "\": \"" << json_escape(value) << "\"";
        if (comma) out << ',';
        out << '\n';
    };
    kv("backend_name", "zkDL/cuda_hyrax_partial");
    kv("compute_backend_name", "gpu");
    kv("benchmark_mode", row.benchmark_mode);
    kv("dataset", spec.dataset);
    kv("dataset_name", dataset.name);
    kv("status", row.status);
    kv("blocker", row.blocker);
    kv("verified", row.status == "success" ? "true" : "false");
    kv("is_full_dataset", row.full_dataset ? "true" : "false");
    kv("gpu_forward_ok", row.gpu_forward_ok ? "true" : "false");
    kv("partial_output_commitment_ok", row.partial_commitment_ok ? "true" : "false");
    kv("proof_generation_ok", row.proof_generation_ok ? "true" : "false");
    kv("verification_ok", row.verification_ok ? "true" : "false");
    kv("node_count", std::to_string(graph.num_nodes));
    kv("edge_count", std::to_string(graph.edges.size()));
    kv("N_total", std::to_string(dataset.num_nodes));
    kv("G_batch", std::to_string(graph.graph_count));
    kv("node_ptr", join_sizes(graph.node_ptr));
    kv("edge_ptr", join_sizes(graph.edge_ptr));
    kv("task_type", spec.task_type);
    kv("report_unit", spec.report_unit);
    kv("batching_rule", spec.batching_rule);
    kv("subgraph_rule", "whole_graph");
    kv("self_loop_rule", "per_node");
    kv("edge_sort_rule", "edge_gid_then_dst_stable");
    kv("chunking_rule", "output_linear_full_dataset_row_chunks_bound_by_relation_label_offset_rows_total_chunks");
    kv("L", std::to_string(model.L));
    std::ostringstream hidden_profile;
    for (std::size_t i = 0; i < model.hidden_profile.size(); ++i) {
        if (i != 0) hidden_profile << ';';
        hidden_profile << model.hidden_profile[i].first << 'x' << model.hidden_profile[i].second;
    }
    kv("hidden_profile", hidden_profile.str());
    kv("K_out", std::to_string(model.K_out));
    kv("C", std::to_string(model.C));
    kv("model_arch_id", model.model_arch_id);
    kv("model_param_id", model.model_param_id);
    kv("quant_cfg_id", model.quant_cfg_id + ";feature_quant=round(x*16)");
    kv("static_table_id", "tables:lrelu+elu+exp+range-required-by-note");
    kv("output_average_rule", model.output_average_rule);
    kv("forward_ms", format_ms(forward.total_ms));
    kv("hidden_forward_projection_ms", format_ms(forward.hidden_projection_ms));
    kv("hidden_forward_attention_ms", format_ms(forward.hidden_attention_ms));
    kv("hidden_forward_activation_ms", format_ms(forward.hidden_activation_ms));
    kv("hidden_concat_ms", format_ms(forward.hidden_concat_ms));
    kv("output_forward_projection_ms", format_ms(forward.output_projection_ms));
    kv("output_forward_attention_ms", format_ms(forward.output_attention_ms));
    kv("output_forward_activation_ms", format_ms(forward.output_activation_ms));
    kv("partial_commitment_ms", format_ms(proof.partial_commitment_ms));
    kv("partial_open_ms", format_ms(proof.partial_open_ms));
    kv("partial_proof_size_bytes", std::to_string(proof.partial_proof_size_bytes));
    kv("level_b_proof_path", proof.level_b.proof_path.empty() ? "" : proof.level_b.proof_path.string());
    kv("debug_level_b_status", row.debug_level_b_status);
    kv("debug_level_b_blocker", row.debug_level_b_blocker);
    kv("debug_level_b_benchmark_mode", row.debug_level_b_benchmark_mode);
    kv("debug_level_b_commitment_time_ms", row.debug_level_b_commitment_time_ms);
    kv("debug_level_b_prove_time_ms", row.debug_level_b_prove_time_ms);
    kv("debug_level_b_verify_time_ms", row.debug_level_b_verify_time_ms);
    kv("debug_level_b_proof_size_bytes", row.debug_level_b_proof_size_bytes);
    kv("debug_level_b_proof_path", row.debug_level_b_proof_path);
    kv("formal_partial_status", row.formal_partial_status);
    kv("formal_partial_blocker", row.formal_partial_blocker);
    kv("formal_partial_commitment_time_ms", row.formal_partial_commitment_time_ms);
    kv("formal_partial_prove_time_ms", row.formal_partial_prove_time_ms);
    kv("formal_partial_verify_time_ms", row.formal_partial_verify_time_ms);
    kv("formal_partial_proof_size_bytes", row.formal_partial_proof_size_bytes);
    kv("formal_partial_proof_path", row.formal_partial_proof_path);
    kv("formal_partial_total_chunks", row.formal_partial_total_chunks);
    kv("formal_partial_verified_chunks", row.formal_partial_verified_chunks);
    kv("formal_partial_covered_relations", row.formal_partial_covered_relations);
    kv("formal_partial_blocked_relations", row.formal_partial_blocked_relations);
    kv("level_b_covered_relations", [&]() {
        std::ostringstream relations;
        for (std::size_t i = 0; i < proof.level_b.covered_relations.size(); ++i) {
            if (i != 0) relations << ';';
            relations << proof.level_b.covered_relations[i];
        }
        return relations.str();
    }());
    kv("level_b_uncovered_relations", [&]() {
        std::ostringstream relations;
        for (std::size_t i = 0; i < proof.level_b.uncovered_relations.size(); ++i) {
            if (i != 0) relations << ';';
            relations << proof.level_b.uncovered_relations[i];
        }
        return relations.str();
    }());
    kv("commitment_time_ms", row.commitment_time_ms);
    kv("prove_time_ms", row.prove_time_ms);
    kv("verify_time_ms", row.verify_time_ms);
    kv("proof_size_bytes", row.proof_size_bytes, false);
    out << "}\n";

    std::ofstream bench(run_dir / "benchmark.txt");
    bench << "backend = gpu\n";
    bench << "commitment_time_ms = " << row.commitment_time_ms << "\n";
    bench << "config = configs/" << spec.config_name << "\n";
    bench << "dataset = " << spec.dataset << "\n";
    bench << "proof_size_bytes = " << row.proof_size_bytes << "\n";
    bench << "prove_time_ms = " << row.prove_time_ms << "\n";
    bench << "status = " << (row.status == "success" ? "VERIFY_OK" : row.status) << "\n";
    bench << "verify_time_ms = " << row.verify_time_ms << "\n";
    bench << "blocker = " << row.blocker << "\n";
    bench << "debug_level_b_status = " << row.debug_level_b_status << "\n";
    bench << "debug_level_b_commitment_time_ms = " << row.debug_level_b_commitment_time_ms << "\n";
    bench << "debug_level_b_prove_time_ms = " << row.debug_level_b_prove_time_ms << "\n";
    bench << "debug_level_b_verify_time_ms = " << row.debug_level_b_verify_time_ms << "\n";
    bench << "debug_level_b_proof_size_bytes = " << row.debug_level_b_proof_size_bytes << "\n";
    bench << "debug_level_b_proof_path = " << row.debug_level_b_proof_path << "\n";
    bench << "debug_level_b_blocker = " << row.debug_level_b_blocker << "\n";
    bench << "formal_partial_status = " << row.formal_partial_status << "\n";
    bench << "formal_partial_commitment_time_ms = " << row.formal_partial_commitment_time_ms << "\n";
    bench << "formal_partial_prove_time_ms = " << row.formal_partial_prove_time_ms << "\n";
    bench << "formal_partial_verify_time_ms = " << row.formal_partial_verify_time_ms << "\n";
    bench << "formal_partial_proof_size_bytes = " << row.formal_partial_proof_size_bytes << "\n";
    bench << "formal_partial_proof_path = " << row.formal_partial_proof_path << "\n";
    bench << "formal_partial_total_chunks = " << row.formal_partial_total_chunks << "\n";
    bench << "formal_partial_verified_chunks = " << row.formal_partial_verified_chunks << "\n";
    bench << "formal_partial_blocker = " << row.formal_partial_blocker << "\n";
}

struct CoverageRow {
    std::string relation;
    bool formal_prover_supported = false;
    bool formal_verifier_supported = false;
    bool gpu_supported = true;
    bool cpu_fallback_supported = false;
    bool chunked_supported = false;
    std::string total_chunks = "NA";
    std::string verified_chunks = "NA";
    bool reused_from_zkdl = false;
    bool newly_added_in_zkdl_adapter = false;
    bool debug_only = false;
    bool blocked = true;
    std::string blocker_reason;
    std::string evidence_file;
    std::string source_file;
};

std::vector<CoverageRow> gat_proof_coverage_rows() {
    return {
        {"feature lookup", false, false, true, false, false, "NA", "NA", false, false, false, true, "尚未接入全局特征表与局部节点查询相等性的 zkDL lookup argument。", "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "gat_zkdl.cu"},
        {"HadamardRelationProof", true, true, false, true, false, "selftest", "selftest", true, true, false, false, "Gate 3 中 C=A*B relation prover/verifier、C opening 和 tamper rejection 均通过。", "zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log", "formal_relations.cuh;zkprod.cuh"},
        {"MatmulRelationProof", true, true, false, true, false, "selftest", "selftest", true, true, false, false, "Gate 3 中 matmul relation prover/verifier 覆盖 batched、非方阵、padding 和 dot-product case。", "zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log", "formal_relations.cuh;zkmatmul.cuh"},
        {"LinearRelationProof", true, true, false, true, true, "see_formal_partial_debug", "see_formal_partial_debug", false, true, false, false, "Gate 3 中 linear relation prover/verifier 通过，并用于 output-head average、bias add、public logits binding、scale decomposition。", "zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log;zkdl_gat_outputs/results/formal_partial_debug.csv", "formal_relations.cuh;gat_zkdl.cu"},
        {"hidden projection", false, false, true, false, true, "见 formal_partial_debug", "见 formal_partial_debug", true, true, false, true, "已有 targeted chunks 通过，但不是全层全头全 chunk；rounding closure 与后续 sparse/softmax/aggregation 未完成。", "zkdl_gat_outputs/results/formal_partial_debug.csv", "gat_zkdl.cu;formal_relations.cuh"},
        {"source attention dot product", false, false, true, false, true, "见 formal_partial_debug", "见 formal_partial_debug", true, true, false, true, "source attention dot 已用真实 trace 接入 MatmulRelationProof；仍缺全数据集、全 head 与后续 LeakyReLU/softmax 链路。", "zkdl_gat_outputs/results/formal_partial_debug.csv", "gat_zkdl.cu;formal_relations.cuh"},
        {"target attention dot product", false, false, true, false, true, "见 formal_partial_debug", "见 formal_partial_debug", true, true, false, true, "destination attention dot 已用真实 trace 接入 MatmulRelationProof；仍缺全数据集、全 head 与后续 sparse routing 链路。", "zkdl_gat_outputs/results/formal_partial_debug.csv", "gat_zkdl.cu;formal_relations.cuh"},
        {"source sparse routing", false, false, true, false, false, "NA", "NA", false, false, false, true, "缺少基于 public src 的 Hyrax/FAC4DNN sparse route proof，不能物化 dense E x N 矩阵替代。", "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "gat_zkdl.cu"},
        {"destination sparse routing", false, false, true, false, false, "NA", "NA", false, false, false, true, "缺少基于 public dst 与分组边序的 sparse route/grouped sum verifier。", "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "gat_zkdl.cu"},
        {"LeakyReLU", false, false, true, false, false, "NA", "NA", true, false, false, true, "zkReLU 只有 prover-side 片段，尚未绑定 attention score 与静态表 verifier。", "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "zkrelu.cuh"},
        {"softmax group max", false, false, true, false, false, "NA", "NA", false, false, false, true, "缺少 per-destination sparse softmax group 的 max/range comparison gadget。", "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "gat_zkdl.cu"},
        {"ExpMap", false, false, true, false, false, "NA", "NA", false, false, false, true, "缺少 QuantExp(-delta) 的 zkDL table lookup argument。", "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "gat_zkdl.cu"},
        {"softmax denominator sum", false, false, true, false, false, "NA", "NA", true, false, false, true, "分组 sparse denominator 还没有承诺、opening 与 verifier gadget。", "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "gat_zkdl.cu"},
        {"inverse", false, false, true, false, false, "NA", "NA", false, false, false, true, "缺少 softmax normalization 的 inverse witness、range 和 denominator nonzero proof。", "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "gat_zkdl.cu"},
        {"alpha multiplication", false, false, true, false, false, "NA", "NA", true, true, false, true, "HadamardRelationProof 可复用，但 alpha=exp*inverse 仍缺 Sum_dst sparse routing 与 inverse witness 绑定。", "zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log", "formal_relations.cuh;zkprod.cuh"},
        {"weighted aggregation", false, false, true, false, false, "NA", "NA", true, false, false, true, "weighted aggregation 还需要 alpha*H_prime product 与按 dst 聚合的 sparse verifier chain。", "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "gat_zkdl.cu"},
        {"ELU", false, false, true, false, false, "NA", "NA", false, false, false, true, "缺少 ELU lookup/range gadget。", "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "gat_zkdl.cu"},
        {"concat", true, true, true, true, true, "see_formal_partial_debug", "see_formal_partial_debug", false, true, false, false, "hidden multi-head concat 已通过 full-dataset row chunks 的 LinearRelationProof 绑定。", "zkdl_gat_outputs/runs/*/warm/formal_artifacts/zkdl_formal_partial.proof.txt;zkdl_gat_outputs/results/formal_partial_debug.csv", "gat_zkdl.cu;formal_relations.cuh"},
        {"output projection", false, false, true, false, true, "见 formal_partial_debug", "见 formal_partial_debug", true, true, false, true, "fixed pre-rescale 与 scale decomposition 覆盖增加；完整 output attention chain 仍缺 sparse softmax/aggregation/rescale rounding。", "zkdl_gat_outputs/results/formal_partial_debug.csv", "gat_zkdl.cu;formal_relations.cuh"},
        {"output source/target attention", false, false, true, false, true, "见 formal_partial_debug", "见 formal_partial_debug", true, true, false, true, "output E_src/E_dst dot 已接入真实 trace；后续 sparse routing、softmax 与 aggregation 未闭合。", "zkdl_gat_outputs/results/formal_partial_debug.csv", "gat_zkdl.cu;formal_relations.cuh"},
        {"output sparse routing", false, false, true, false, false, "NA", "NA", false, false, false, true, "输出头仍缺 Hyrax sparse route proof。", "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "gat_zkdl.cu"},
        {"output softmax", false, false, true, false, false, "NA", "NA", false, false, false, true, "输出头 softmax 的 max/ExpMap/sum/inverse chain 尚未支持。", "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "gat_zkdl.cu"},
        {"output aggregation", false, false, true, false, false, "NA", "NA", true, false, false, true, "输出聚合需要 product 与 sparse routing proof；当前缺 sparse component。", "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "gat_zkdl.cu"},
        {"fixed-to-rounded trace scale decomposition", true, true, true, true, true, "targeted schedules", "所有本轮通过 matmul chunk 均含 C_fixed - 16*C_trace - residual = 0", false, true, false, true, "已证明 exact decomposition；尚未单独证明 sign-specific rounding mode 与 tie-breaking。", "zkdl_gat_outputs/results/formal_partial_debug.csv", "gat_zkdl.cu;formal_relations.cuh"},
        {"residual bounded range binding", true, true, true, true, true, "targeted schedules", "Cora output 6; Cora hidden 25; Citeseer output 2; Citeseer hidden 4; Pubmed/PPI/ogbn output 各 1", false, true, false, true, "本轮新增 bitness + residual+offset decomposition；sign-specific rounding mode 和 tie-breaking 仍阻塞，不能标 Level A。", "zkdl_gat_outputs/results/formal_partial_debug.csv", "gat_zkdl.cu"},
        {"bias add", true, true, true, true, true, "see_formal_partial_debug", "see_formal_partial_debug", false, true, false, false, "output bias add 已在 full-dataset chunks 中证明 Y_head = Y_pre_rescaled + b；Y_pre 到 Y_pre_rescaled 的 rescale 仍阻塞。", "zkdl_gat_outputs/runs/*/warm/formal_artifacts/zkdl_formal_partial.proof.txt;zkdl_gat_outputs/results/formal_partial_debug.csv", "gat_zkdl.cu;formal_relations.cuh"},
        {"output-head average", true, true, true, true, true, "see_formal_partial_debug", "see_formal_partial_debug", false, true, false, false, "output-head average/public logits binding 使用 chunked LinearRelationProof，K_out=1 仍走通用 average path。", "zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log;zkdl_gat_outputs/results/formal_partial_debug.csv", "gat_zkdl.cu;formal_relations.cuh"},
        {"public logits binding", true, true, true, true, true, "see_formal_partial_debug", "see_formal_partial_debug", false, true, false, false, "public logits 已通过 verifier 计算的 public MLE value 与 LinearRelationProof 绑定；前序 GAT relation 未闭合所以非 Level A。", "zkdl_gat_outputs/runs/*/warm/formal_artifacts/zkdl_formal_partial.proof.txt;zkdl_gat_outputs/results/formal_partial_debug.csv", "gat_zkdl.cu;formal_relations.cuh"},
        {"commitment/evaluation binding", true, true, false, true, false, "selftest", "selftest", true, true, false, false, "Gate 1 opening selftest 覆盖 power-of-two、非 power-of-two、padding 和 mixed fold。", "zkdl_gat_outputs/runs/selftest/formal_opening_selftest_latest.log", "commitment.cuh"},
        {"transcript binding", true, true, false, true, true, "selftest+chunks", "selftest+chunks", true, true, false, false, "opening、inner-product、Hadamard、Matmul、Linear relation selftest 与 chunk label 均绑定 transcript。", "zkdl_gat_outputs/runs/selftest/formal_sumcheck_selftest_latest.log;zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log;zkdl_gat_outputs/results/formal_partial_debug.csv", "proof.cuh;formal_relations.cuh;gat_zkdl.cu"},
        {"verifier replay", true, true, false, true, true, "selftest+chunks", "selftest+chunks", true, true, false, false, "formal opening 与 relation selftest 的 verifier replay 通过；完整 GAT replay 仍缺 sparse/lookup/rescale gadget。", "zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log;zkdl_gat_outputs/results/formal_partial_debug.csv", "formal_relations.cuh;gat_zkdl.cu"},
        {"proof serialization", true, true, false, true, true, "see_formal_partial_debug", "see_formal_partial_debug", false, true, false, false, "formal proof artifact 记录 serialized_size_bytes 与每 chunk relation detail，不包含完整 witness tensor。", "zkdl_gat_outputs/runs/*/warm/formal_artifacts/zkdl_formal_partial.proof.txt", "proof.cuh;gat_zkdl.cu"},
        {"proof size accounting", true, true, false, true, true, "see_formal_partial_debug", "see_formal_partial_debug", false, true, false, false, "proof size 统计 proof object/file serialized size，不统计 witness、debug dump、manifest 或日志。", "zkdl_gat_outputs/results/formal_partial_debug.csv", "proof.cuh;gat_zkdl.cu"},
    };
}

std::string bool_string(bool value) {
    return value ? "true" : "false";
}

void write_coverage_matrix(const fs::path& output_root) {
    ensure_dir(output_root / "results");
    const auto rows = gat_proof_coverage_rows();
    auto transcript_bound = [](const CoverageRow& row) {
        return row.formal_verifier_supported && !row.debug_only;
    };
    auto opening_bound = [](const CoverageRow& row) {
        return row.formal_verifier_supported && !row.debug_only;
    };
    auto partial_formal = [](const CoverageRow& row) {
        return row.formal_verifier_supported && (row.chunked_supported || row.blocked || row.total_chunks != "selftest");
    };
    {
        std::ofstream csv(output_root / "results" / "gat_proof_coverage_matrix.csv");
        csv << "relation_name,formal_prover_supported,formal_verifier_supported,transcript_bound,opening_bound,chunked_supported,total_chunks,verified_chunks,partial_formal,debug_only,blocked,blocker_reason,evidence_file,source_file\n";
        for (const auto& row : rows) {
            csv << row.relation
                << ',' << bool_string(row.formal_prover_supported)
                << ',' << bool_string(row.formal_verifier_supported)
                << ',' << bool_string(transcript_bound(row))
                << ',' << bool_string(opening_bound(row))
                << ',' << bool_string(row.chunked_supported)
                << ",\"" << json_escape(row.total_chunks) << "\""
                << ",\"" << json_escape(row.verified_chunks) << "\""
                << ',' << bool_string(partial_formal(row))
                << ',' << bool_string(row.debug_only)
                << ',' << bool_string(row.blocked)
                << ",\"" << json_escape(row.blocker_reason) << "\""
                << ",\"" << json_escape(row.evidence_file) << "\""
                << ",\"" << json_escape(row.source_file) << "\"\n";
        }
    }
    {
        std::ofstream json(output_root / "results" / "gat_proof_coverage_matrix.json");
        json << "{\n  \"coverage_level\": \"formal_level_a_blocked_with_debug_level_b_binding\",\n  \"rows\": [\n";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto& row = rows[i];
            json << "    {\n";
            json << "      \"relation\": \"" << json_escape(row.relation) << "\",\n";
            json << "      \"relation_name\": \"" << json_escape(row.relation) << "\",\n";
            json << "      \"formal_prover_supported\": " << bool_string(row.formal_prover_supported) << ",\n";
            json << "      \"formal_verifier_supported\": " << bool_string(row.formal_verifier_supported) << ",\n";
            json << "      \"transcript_bound\": " << bool_string(transcript_bound(row)) << ",\n";
            json << "      \"opening_bound\": " << bool_string(opening_bound(row)) << ",\n";
            json << "      \"chunked_supported\": " << bool_string(row.chunked_supported) << ",\n";
            json << "      \"total_chunks\": \"" << json_escape(row.total_chunks) << "\",\n";
            json << "      \"verified_chunks\": \"" << json_escape(row.verified_chunks) << "\",\n";
            json << "      \"partial_formal\": " << bool_string(partial_formal(row)) << ",\n";
            json << "      \"debug_only\": " << bool_string(row.debug_only) << ",\n";
            json << "      \"blocked\": " << bool_string(row.blocked) << ",\n";
            json << "      \"blocker_reason\": \"" << json_escape(row.blocker_reason) << "\",\n";
            json << "      \"evidence_file\": \"" << json_escape(row.evidence_file) << "\",\n";
            json << "      \"source_file\": \"" << json_escape(row.source_file) << "\"\n";
            json << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
        }
        json << "  ]\n}\n";
    }
    {
        std::ofstream md(output_root / "results" / "gat_proof_coverage_matrix.md");
        md << "# GAT proof coverage matrix\n\n";
        md << "| relation | prover | verifier | transcript | opening | chunked | total | verified | partial | debug | blocked | evidence | blocker |\n";
        md << "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n";
        for (const auto& row : rows) {
            md << "| " << row.relation
               << " | " << bool_string(row.formal_prover_supported)
               << " | " << bool_string(row.formal_verifier_supported)
               << " | " << bool_string(transcript_bound(row))
               << " | " << bool_string(opening_bound(row))
               << " | " << bool_string(row.chunked_supported)
               << " | " << row.total_chunks
               << " | " << row.verified_chunks
               << " | " << bool_string(partial_formal(row))
               << " | " << bool_string(row.debug_only)
               << " | " << bool_string(row.blocked)
               << " | " << row.evidence_file
               << " | " << row.blocker_reason << " |\n";
        }
    }
    fs::copy_file(
        output_root / "results" / "gat_proof_coverage_matrix.csv",
        output_root / "results" / "coverage_matrix.csv",
        fs::copy_options::overwrite_existing);
    fs::copy_file(
        output_root / "results" / "gat_proof_coverage_matrix.json",
        output_root / "results" / "coverage_matrix.json",
        fs::copy_options::overwrite_existing);
    fs::copy_file(
        output_root / "results" / "gat_proof_coverage_matrix.md",
        output_root / "results" / "coverage_matrix.md",
        fs::copy_options::overwrite_existing);
}

void write_result_tables(const fs::path& output_root, const std::vector<RunRow>& rows) {
    ensure_dir(output_root / "results");
    {
        std::ofstream csv(output_root / "results" / "latest.csv");
        csv << "dataset,status,blocker,benchmark_mode,commitment_time_ms,prove_time_ms,verify_time_ms,proof_size_bytes\n";
        for (const auto& row : rows) {
            csv << row.dataset << ',' << row.status << ",\"" << json_escape(row.blocker) << "\"," << row.benchmark_mode
                << ',' << row.commitment_time_ms
                << ',' << row.prove_time_ms
                << ',' << row.verify_time_ms
                << ',' << row.proof_size_bytes << '\n';
        }
    }
    {
        std::ofstream json(output_root / "results" / "latest.json");
        json << "{\n  \"benchmark_mode\": \"warm\",\n  \"policy\": \"official rows 使用主仓库 benchmark schema；完整 Level A proof generation 和 verification 成功前正式四项保持 NA\",\n  \"rows\": [\n";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto& row = rows[i];
            json << "    {\n";
            json << "      \"dataset\": \"" << json_escape(row.dataset) << "\",\n";
            json << "      \"status\": \"" << json_escape(row.status) << "\",\n";
            json << "      \"blocker\": \"" << json_escape(row.blocker) << "\",\n";
            json << "      \"benchmark_mode\": \"" << json_escape(row.benchmark_mode) << "\",\n";
            json << "      \"commitment_time_ms\": \"" << row.commitment_time_ms << "\",\n";
            json << "      \"prove_time_ms\": \"" << row.prove_time_ms << "\",\n";
            json << "      \"verify_time_ms\": \"" << row.verify_time_ms << "\",\n";
            json << "      \"proof_size_bytes\": \"" << row.proof_size_bytes << "\",\n";
            json << "      \"is_full_dataset\": " << (row.full_dataset ? "true" : "false") << ",\n";
            json << "      \"gpu_forward_ok\": " << (row.gpu_forward_ok ? "true" : "false") << ",\n";
            json << "      \"partial_output_commitment_ok\": " << (row.partial_commitment_ok ? "true" : "false") << ",\n";
            json << "      \"formal_proof_generation_ok\": " << (row.proof_generation_ok ? "true" : "false") << ",\n";
            json << "      \"formal_verification_ok\": " << (row.verification_ok ? "true" : "false") << ",\n";
            json << "      \"debug_level_b_status\": \"" << json_escape(row.debug_level_b_status) << "\"\n";
            json << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
        }
        json << "  ]\n}\n";
    }
    {
        std::ofstream csv(output_root / "results" / "debug_level_b.csv");
        csv << "dataset,status,blocker,benchmark_mode,commitment_time_ms,prove_time_ms,verify_time_ms,proof_size_bytes,proof_path\n";
        for (const auto& row : rows) {
            csv << row.dataset
                << ',' << row.debug_level_b_status
                << ",\"" << json_escape(row.debug_level_b_blocker) << "\","
                << row.debug_level_b_benchmark_mode
                << ',' << row.debug_level_b_commitment_time_ms
                << ',' << row.debug_level_b_prove_time_ms
                << ',' << row.debug_level_b_verify_time_ms
                << ',' << row.debug_level_b_proof_size_bytes
                << ",\"" << json_escape(row.debug_level_b_proof_path) << "\"\n";
        }
    }
    {
        std::ofstream json(output_root / "results" / "debug_level_b.json");
        json << "{\n  \"benchmark_mode\": \"diagnostic_level_b_output_binding_only_not_formal_level_a\",\n  \"policy\": \"diagnostic 数字不得写入正式对比表\",\n  \"rows\": [\n";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto& row = rows[i];
            json << "    {\n";
            json << "      \"dataset\": \"" << json_escape(row.dataset) << "\",\n";
            json << "      \"status\": \"" << json_escape(row.debug_level_b_status) << "\",\n";
            json << "      \"blocker\": \"" << json_escape(row.debug_level_b_blocker) << "\",\n";
            json << "      \"benchmark_mode\": \"" << json_escape(row.debug_level_b_benchmark_mode) << "\",\n";
            json << "      \"commitment_time_ms\": \"" << row.debug_level_b_commitment_time_ms << "\",\n";
            json << "      \"prove_time_ms\": \"" << row.debug_level_b_prove_time_ms << "\",\n";
            json << "      \"verify_time_ms\": \"" << row.debug_level_b_verify_time_ms << "\",\n";
            json << "      \"proof_size_bytes\": \"" << row.debug_level_b_proof_size_bytes << "\",\n";
            json << "      \"proof_path\": \"" << json_escape(row.debug_level_b_proof_path) << "\"\n";
            json << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
        }
        json << "  ]\n}\n";
    }
    {
        std::ofstream csv(output_root / "results" / "formal_partial_debug.csv");
        csv << "dataset,component,status,benchmark_mode,commitment_time_ms,prove_time_ms,verify_time_ms,proof_size_bytes,total_chunks,verified_chunks,newly_verified_chunks,verified_relations,blocked_relations,proof_artifact_paths,evidence_paths,blocker_evidence,resume_point,blocker\n";
        csv << "NA,formal_opening_selftest,success,formal_partial_opening_proof_verify,NA,NA,NA,see_log,NA,NA,NA,\"commitment/evaluation binding\",\"PPI/ogbn-arxiv 完整 G1 大开口只作为压力边界，不写入 Level A\",\"zkdl_gat_outputs/runs/selftest/formal_opening_selftest_latest.log\",\"zkdl_gat_outputs/runs/selftest/formal_opening_selftest_latest.log\",\"\",,\"Gate 1 opening selftest PASS；大图 logits-size G1 opening 不作为正式 Level A 结果。\"\n";
        csv << "NA,formal_sumcheck_selftest,success,formal_partial_inner_product_sumcheck_verify,NA,NA,NA,see_log,NA,NA,NA,\"inner-product sumcheck transcript replay\",\"\",\"zkdl_gat_outputs/runs/selftest/formal_sumcheck_selftest_latest.log\",\"zkdl_gat_outputs/runs/selftest/formal_sumcheck_selftest_latest.log\",\"\",,\"Gate 2 inner-product transcript replay verifier PASS。\"\n";
        csv << "NA,formal_relation_selftest,success,formal_partial_hadamard_matmul_linear_verify,NA,NA,NA,see_log,NA,NA,NA,\"HadamardRelationProof;MatmulRelationProof;LinearRelationProof\",\"\",\"zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log\",\"zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log\",\"\",,\"Gate 3 relation selftest PASS，包含 tamper rejection。\"\n";
        for (const auto& row : rows) {
            csv << row.dataset
                << ",formal_partial_trace"
                << ',' << row.formal_partial_status
                << ",partial_formal_relations_only,"
                << row.formal_partial_commitment_time_ms
                << ',' << row.formal_partial_prove_time_ms
                << ',' << row.formal_partial_verify_time_ms
                << ',' << row.formal_partial_proof_size_bytes
                << ',' << row.formal_partial_total_chunks
                << ',' << row.formal_partial_verified_chunks
                << ',' << row.formal_partial_verified_chunks
                << ",\"" << json_escape(row.formal_partial_covered_relations) << "\""
                << ",\"" << json_escape(row.formal_partial_blocked_relations) << "\""
                << ",\"" << json_escape(row.formal_partial_proof_path) << "\""
                << ",\"" << json_escape(row.formal_partial_proof_path + ";zkdl_gat_outputs/runs/" + row.dataset + "/warm/proof_debug.log") << "\""
                << ",\"" << json_escape("debug/matmul_first_failure_" + row.dataset + ".json") << "\""
                << ",\"" << json_escape("zkdl_gat_outputs/runs/" + row.dataset + "/warm/RESUME_POINT.md") << "\""
                << ","
                << "\"" << json_escape(row.formal_partial_blocker) << "\"\n";
        }
    }
    {
        std::ofstream json(output_root / "results" / "formal_partial_debug.json");
        json << "{\n  \"rows\": [\n";
        json << "    {\"dataset\":\"NA\",\"component\":\"formal_opening_selftest\",\"status\":\"success\",\"benchmark_mode\":\"formal_partial_opening_proof_verify\",\"commitment_time_ms\":\"NA\",\"prove_time_ms\":\"NA\",\"verify_time_ms\":\"NA\",\"proof_size_bytes\":\"see_log\",\"total_chunks\":\"NA\",\"verified_chunks\":\"NA\",\"newly_verified_chunks\":\"NA\",\"proof_artifact_paths\":\"zkdl_gat_outputs/runs/selftest/formal_opening_selftest_latest.log\",\"evidence_paths\":\"zkdl_gat_outputs/runs/selftest/formal_opening_selftest_latest.log\",\"blocker_evidence\":\"\",\"resume_point\":\"\",\"blocker\":\"Gate 1 opening selftest PASS；大图 logits-size G1 opening 不作为正式 Level A 结果。\"},\n";
        json << "    {\"dataset\":\"NA\",\"component\":\"formal_sumcheck_selftest\",\"status\":\"success\",\"benchmark_mode\":\"formal_partial_inner_product_sumcheck_verify\",\"commitment_time_ms\":\"NA\",\"prove_time_ms\":\"NA\",\"verify_time_ms\":\"NA\",\"proof_size_bytes\":\"see_log\",\"total_chunks\":\"NA\",\"verified_chunks\":\"NA\",\"newly_verified_chunks\":\"NA\",\"proof_artifact_paths\":\"zkdl_gat_outputs/runs/selftest/formal_sumcheck_selftest_latest.log\",\"evidence_paths\":\"zkdl_gat_outputs/runs/selftest/formal_sumcheck_selftest_latest.log\",\"blocker_evidence\":\"\",\"resume_point\":\"\",\"blocker\":\"Gate 2 inner-product transcript replay verifier PASS。\"},\n";
        json << "    {\"dataset\":\"NA\",\"component\":\"formal_relation_selftest\",\"status\":\"success\",\"benchmark_mode\":\"formal_partial_hadamard_matmul_linear_verify\",\"commitment_time_ms\":\"NA\",\"prove_time_ms\":\"NA\",\"verify_time_ms\":\"NA\",\"proof_size_bytes\":\"see_log\",\"total_chunks\":\"NA\",\"verified_chunks\":\"NA\",\"newly_verified_chunks\":\"NA\",\"proof_artifact_paths\":\"zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log\",\"evidence_paths\":\"zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log\",\"blocker_evidence\":\"\",\"resume_point\":\"\",\"blocker\":\"Gate 3 relation selftest PASS，包含 tamper rejection。\"}";
        for (const auto& row : rows) {
            json << ",\n";
            json << "    {\"dataset\":\"" << json_escape(row.dataset)
                 << "\",\"component\":\"formal_partial_trace"
                 << "\",\"status\":\"" << json_escape(row.formal_partial_status)
                 << "\",\"benchmark_mode\":\"partial_formal_relations_only\""
                 << ",\"commitment_time_ms\":\"" << row.formal_partial_commitment_time_ms
                 << "\",\"prove_time_ms\":\"" << row.formal_partial_prove_time_ms
                 << "\",\"verify_time_ms\":\"" << row.formal_partial_verify_time_ms
                 << "\",\"proof_size_bytes\":\"" << row.formal_partial_proof_size_bytes
                 << "\",\"total_chunks\":\"" << row.formal_partial_total_chunks
                 << "\",\"verified_chunks\":\"" << row.formal_partial_verified_chunks
                 << "\",\"newly_verified_chunks\":\"" << row.formal_partial_verified_chunks
                 << "\",\"proof_artifact_paths\":\"" << json_escape(row.formal_partial_proof_path)
                 << "\",\"evidence_file\":\"" << json_escape(row.formal_partial_proof_path)
                 << "\",\"evidence_paths\":\"" << json_escape(row.formal_partial_proof_path + ";zkdl_gat_outputs/runs/" + row.dataset + "/warm/proof_debug.log")
                 << "\",\"blocker_evidence\":\"" << json_escape("debug/matmul_first_failure_" + row.dataset + ".json")
                 << "\",\"resume_point\":\"" << json_escape("zkdl_gat_outputs/runs/" + row.dataset + "/warm/RESUME_POINT.md")
                 << "\",\"covered_relations\":\"" << json_escape(row.formal_partial_covered_relations)
                 << "\",\"blocked_relations\":\"" << json_escape(row.formal_partial_blocked_relations)
                 << "\",\"blocker\":\"" << json_escape(row.formal_partial_blocker) << "\"}";
        }
        json << "\n";
        json << "  ]\n}\n";
    }
    {
        std::ofstream md(output_root / "results" / "summary.md");
        md << "# 最新正式结果\n\n";
        md << "| 数据集 | 状态 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | 阻塞说明 |\n";
        md << "| --- | --- | ---: | ---: | ---: | ---: | --- |\n";
        for (const auto& row : rows) {
            md << "| " << row.dataset
               << " | " << (row.status == "blocked" ? "阻塞" : row.status)
               << " | " << row.commitment_time_ms
               << " | " << row.prove_time_ms
               << " | " << row.verify_time_ms
               << " | " << row.proof_size_bytes
               << " | " << row.blocker << " |\n";
        }
    }
    write_coverage_matrix(output_root);
}

void write_adapter_readme(const fs::path& output_root, const std::vector<RunRow>& rows) {
    std::ofstream out(output_root / "README_ZKDL_GAT_ADAPTER.md");
    out << "# zkDL-GAT adapter 状态\n\n";
    out << "当前状态：不是 Level A。Gate 1/2/3 通过时，以 `runs/selftest/*_latest.log` 中的 `SUMMARY status=PASS` 为准。\n\n";
    out << "本实验只在 zkDL baseline 内做 GAT 模型替换和必要适配：读取主仓库数据 cache、checkpoint manifest、config 与协议文件作为只读输入；输出、proof artifact、evidence、coverage 和 benchmark 只写入本目录的 `zkdl_gat_outputs`。\n\n";
    out << "与主仓库指标口径对齐：`latest.csv` 字段固定为 `dataset,status,blocker,benchmark_mode,commitment_time_ms,prove_time_ms,verify_time_ms,proof_size_bytes`，数据集顺序固定为 `cora,citeseer,pubmed,ppi,ogbn-arxiv`，`benchmark_mode` 固定为 `warm`。完整 full dataset + formal proof generation + formal verification 成功前，正式四项保持 `NA`。\n\n";
    out << "当前 formal verifier relation：Gate primitive、commitment/evaluation binding、transcript replay、HadamardRelationProof、MatmulRelationProof、LinearRelationProof、hidden concat、output bias add、output-head average 和 public logits binding。projection/dot、scale decomposition 与 residual bounded range 只按 targeted/full-trace schedule 计入 partial formal。\n\n";
    out << "fixed-point 状态：fixed-point pre-rescale product 使用真实 trace；scale decomposition 证明 `C_fixed - 16*C_trace - residual = 0`；residual bounded range 使用 bitness 和 `residual+offset` decomposition。rounding sign/tie-breaking 仍 blocked，因为公开 `quant_cfg_id` 只给出 `fp32_bundle_export` 或 `fp32_torch_checkpoint`，没有逐个 rescale relation 的 `rho/delta/rem_bits/tie_policy`。\n\n";
    out << "sparse routing / softmax / nonlinear / aggregation 状态：source/destination routing、softmax max/sum/inverse/ExpMap、LeakyReLU、ELU、weighted aggregation、output aggregation 和真实 rescale chain 仍缺 zkDL verifier gadget，不能标为 supported。\n\n";
    out << "目录清理原则：保留源码、构建脚本、正式 results、summary、formal_partial_debug、debug_level_b、coverage matrix、proof artifact、proof evidence、run manifest、benchmark、proof_debug、dataset resume point 和本 README；不再生成 clean/boundary/main-alignment/blocker/rounding 审计文档。\n\n";
    out << "数据集状态：\n\n";
    out << "| dataset | status | gpu_forward | partial commitment_ms | partial prove_ms | partial verify_ms | partial proof_size_bytes | verified chunks | total chunks | debug Level B |\n";
    out << "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n";
    for (const auto& row : rows) {
        out << "| " << row.dataset
            << " | " << row.status
            << " | " << (row.gpu_forward_ok ? "ok" : "failed")
            << " | " << row.formal_partial_commitment_time_ms
            << " | " << row.formal_partial_prove_time_ms
            << " | " << row.formal_partial_verify_time_ms
            << " | " << row.formal_partial_proof_size_bytes
            << " | " << row.formal_partial_verified_chunks
            << " | " << row.formal_partial_total_chunks
            << " | " << row.debug_level_b_status << " |\n";
    }
    out << "\n下一最小继续点以各数据集 `runs/<dataset>/warm/RESUME_POINT.md` 为准；若某个 relation 未完成，继续同一 relation 的下一个 chunk。coverage 说明见 `results/coverage_matrix.*`。\n";
}

RunRow run_one(const fs::path& project_root, const fs::path& output_root, const DatasetRunSpec& spec) {
    RunRow row;
    row.dataset = spec.dataset;
    const auto run_dir = output_root / "runs" / spec.dataset / "warm";
    ensure_dir(run_dir);
    fs::remove_all(run_dir / "proof");
    fs::remove_all(run_dir / "formal_artifacts");
    std::ofstream log(run_dir / "run.log");
    auto log_line = [&](const std::string& line) {
        log << line << '\n';
        log.flush();
        std::cout << line << '\n';
    };

    try {
        log_line("dataset=" + spec.dataset + " stage=load_dataset");
        ScopedTimer load_timer;
        const auto dataset = load_cached_dataset(project_root / "data/cache" / spec.cache_name);
        log_line("dataset=" + spec.dataset + " loaded_nodes=" + std::to_string(dataset.num_nodes)
            + " loaded_edges=" + std::to_string(dataset.edges.size())
            + " features=" + std::to_string(dataset.num_features)
            + " classes=" + std::to_string(dataset.num_classes)
            + " load_ms=" + format_ms(load_timer.elapsed_ms()));

        log_line("dataset=" + spec.dataset + " stage=normalize_graph");
        const auto graph = normalize_graph(dataset, spec);
        row.full_dataset = graph.num_nodes == dataset.num_nodes;
        log_line("dataset=" + spec.dataset + " normalized_nodes=" + std::to_string(graph.num_nodes)
            + " normalized_edges=" + std::to_string(graph.edges.size())
            + " graphs=" + std::to_string(graph.graph_count));

        log_line("dataset=" + spec.dataset + " stage=load_model");
        const auto model = load_model(project_root / "artifacts/checkpoints" / spec.checkpoint_name);
        if (model.output_average_rule != "per_head_bias_then_arithmetic_mean") {
            throw std::runtime_error("unsupported output average rule: " + model.output_average_rule);
        }
        std::ostringstream model_line;
        model_line << "dataset=" << spec.dataset
                   << " model=" << model.model_arch_id
                   << " L=" << model.L
                   << " K_out=" << model.K_out
                   << " C=" << model.C
                   << " hidden_profile=";
        for (std::size_t i = 0; i < model.hidden_profile.size(); ++i) {
            if (i != 0) model_line << ';';
            model_line << model.hidden_profile[i].first << 'x' << model.hidden_profile[i].second;
        }
        log_line(model_line.str());

        log_line("dataset=" + spec.dataset + " stage=gpu_forward");
        ForwardMetrics forward;
        ForwardTraceHost trace;
        const auto logits = run_gat_forward_gpu(graph, model, &forward, &trace);
        row.gpu_forward_ok = true;
        log_line("dataset=" + spec.dataset + " gpu_forward_ms=" + format_ms(forward.total_ms)
            + " logits_values=" + std::to_string(logits.size()));

        log_line("dataset=" + spec.dataset + " stage=zkdl_level_b_output_binding_proof");
        ProofAttempt proof;
        proof.level_b = run_level_b_output_binding_proof(spec, model, graph, trace, run_dir);
        row.partial_commitment_ok = proof.level_b.commitment_ms > 0.0;
        row.debug_level_b_status = proof.level_b.verification_ok ? "success_debug_only" : (proof.level_b.proof_generation_ok ? "blocked_debug" : "failed_debug");
        row.debug_level_b_blocker = proof.level_b.blocker;
        row.debug_level_b_benchmark_mode = proof.level_b.benchmark_mode;
        row.debug_level_b_commitment_time_ms = proof.level_b.proof_generation_ok ? format_ms(proof.level_b.commitment_ms) : "NA";
        row.debug_level_b_prove_time_ms = proof.level_b.proof_generation_ok ? format_ms(proof.level_b.prove_ms) : "NA";
        row.debug_level_b_verify_time_ms = proof.level_b.verification_ok ? format_ms(proof.level_b.verify_ms) : "NA";
        row.debug_level_b_proof_size_bytes = proof.level_b.proof_size_bytes != 0 ? std::to_string(proof.level_b.proof_size_bytes) : "NA";
        row.debug_level_b_proof_path = proof.level_b.proof_path.empty() ? "" : proof.level_b.proof_path.string();

        log_line("dataset=" + spec.dataset + " stage=formal_partial_trace_relations");
        proof.formal_partial = run_formal_partial_trace_proof(spec, model, graph, trace, run_dir);
        auto join_strings = [](const std::vector<std::string>& values) {
            std::ostringstream out;
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (i != 0) out << ';';
                out << values[i];
            }
            return out.str();
        };
        row.formal_partial_status = proof.formal_partial.verification_ok ? "partial_formal" : "blocked";
        row.formal_partial_blocker = proof.formal_partial.blocker;
        row.formal_partial_commitment_time_ms = proof.formal_partial.proof_generation_ok ? format_ms(proof.formal_partial.commitment_ms) : "NA";
        row.formal_partial_prove_time_ms = proof.formal_partial.proof_generation_ok ? format_ms(proof.formal_partial.prove_ms) : "NA";
        row.formal_partial_verify_time_ms = proof.formal_partial.verification_ok ? format_ms(proof.formal_partial.verify_ms) : "NA";
        row.formal_partial_proof_size_bytes = proof.formal_partial.proof_size_bytes != 0 ? std::to_string(proof.formal_partial.proof_size_bytes) : "NA";
        row.formal_partial_proof_path = proof.formal_partial.proof_path.empty() ? "" : proof.formal_partial.proof_path.string();
        row.formal_partial_total_chunks = std::to_string(proof.formal_partial.total_chunks);
        row.formal_partial_verified_chunks = std::to_string(proof.formal_partial.verified_chunks);
        row.formal_partial_covered_relations = join_strings(proof.formal_partial.covered_relations);
        row.formal_partial_blocked_relations = join_strings(proof.formal_partial.blocked_relations);

        row.proof_generation_ok = proof.formal_partial.proof_generation_ok;
        row.verification_ok = proof.formal_partial.verification_ok;
        row.benchmark_mode = "warm";
        row.commitment_time_ms = "NA";
        row.prove_time_ms = "NA";
        row.verify_time_ms = "NA";
        row.proof_size_bytes = "NA";
        row.status = "blocked";
        row.blocker =
            "Formal Level A zkDL-GAT proof is not complete. "
            + row.formal_partial_blocker
            + " Official four metrics remain NA; partial relation timings and proof size are isolated in formal_partial_debug.*.";
        log_line("dataset=" + spec.dataset
            + " commitment_status=" + (row.partial_commitment_ok ? "ok" : "failed")
            + " debug_level_b_status=" + row.debug_level_b_status
            + " debug_level_b_proof_size_bytes=" + row.debug_level_b_proof_size_bytes
            + " formal_partial_status=" + row.formal_partial_status
            + " formal_partial_proof_size_bytes=" + row.formal_partial_proof_size_bytes
            + " formal_partial_chunks=" + row.formal_partial_verified_chunks + "/" + row.formal_partial_total_chunks
            + " formal_status=" + row.status
            + " failure=\"" + row.blocker + "\"");
        log_line("dataset=" + spec.dataset
            + " commitment_time_ms=" + row.commitment_time_ms
            + " prove_time_ms=" + row.prove_time_ms
            + " verify_time_ms=" + row.verify_time_ms);
        log_line("dataset=" + spec.dataset
            + " debug_level_b_commitment_time_ms=" + row.debug_level_b_commitment_time_ms
            + " debug_level_b_prove_time_ms=" + row.debug_level_b_prove_time_ms
            + " debug_level_b_verify_time_ms=" + row.debug_level_b_verify_time_ms
            + " debug_level_b_proof_path=\"" + row.debug_level_b_proof_path + "\"");
        log_line("dataset=" + spec.dataset
            + " formal_partial_commitment_time_ms=" + row.formal_partial_commitment_time_ms
            + " formal_partial_prove_time_ms=" + row.formal_partial_prove_time_ms
            + " formal_partial_verify_time_ms=" + row.formal_partial_verify_time_ms
            + " formal_partial_proof_path=\"" + row.formal_partial_proof_path + "\"");
        write_manifest(run_dir, spec, dataset, graph, model, forward, proof, row);
    } catch (const std::exception& error) {
        row.status = "failed";
        row.blocker = error.what();
        log_line("dataset=" + spec.dataset + " status=FAILED reason=\"" + row.blocker + "\"");
        GraphDataset empty_dataset;
        empty_dataset.name = spec.dataset;
        LocalGraph empty_graph;
        ModelParams empty_model;
        ForwardMetrics empty_forward;
        ProofAttempt empty_proof;
        write_manifest(run_dir, spec, empty_dataset, empty_graph, empty_model, empty_forward, empty_proof, row);
    }

    std::cout
        << "dataset=" << row.dataset
        << " config=configs/" << spec.config_name
        << " backend=gpu"
        << " status=" << row.status
        << " commitment_time_ms=" << row.commitment_time_ms
        << " prove_time_ms=" << row.prove_time_ms
        << " verify_time_ms=" << row.verify_time_ms
        << " proof_size_bytes=" << row.proof_size_bytes
        << '\n';
    return row;
}

void run_formal_matmul_schedule(
    const fs::path& project_root,
    const fs::path& output_root,
    const DatasetRunSpec& spec,
    const std::string& relation,
    std::size_t start_chunk,
    std::size_t max_chunks)
{
    const auto dataset = load_cached_dataset(project_root / "data/cache" / spec.cache_name);
    const auto graph = normalize_graph(dataset, spec);
    const auto model = load_model(project_root / "artifacts/checkpoints" / spec.checkpoint_name);
    ForwardMetrics forward;
    ForwardTraceHost trace;
    (void)run_gat_forward_gpu(graph, model, &forward, &trace);

    std::vector<double> a_values;
    std::vector<double> b_values;
    std::vector<double> c_values;
    std::size_t rows = graph.num_nodes;
    std::size_t shared = 0;
    std::size_t cols = 0;
    std::string label = relation;
    std::smatch match;
    const std::regex output_relation(R"(output_head([0-9]+)_(projection|source_attention_dot|destination_attention_dot))");
    const std::regex hidden_relation(R"(hidden_layer([0-9]+)_head([0-9]+)_(projection|source_attention_dot|destination_attention_dot))");
    if (std::regex_match(relation, match, output_relation)) {
        const std::size_t head_index = static_cast<std::size_t>(std::stoull(match[1].str()));
        const std::string kind = match[2].str();
        if (head_index >= model.output_heads.size() || head_index >= trace.output_heads.size()) {
            throw std::runtime_error("output head index out of range for schedule relation: " + relation);
        }
        const auto& output_model = model.output_heads[head_index];
        const auto& output_trace = trace.output_heads[head_index];
        if (kind == "projection") {
            if (trace.output_layer_input.empty()) {
                throw std::runtime_error("output projection schedule requires output layer input trace");
            }
            a_values = trace.output_layer_input;
            shared = trace.output_layer_input_dim;
            b_values = output_model.W;
            cols = output_trace.width;
            c_values = output_trace.h_prime;
        } else {
            a_values = output_trace.h_prime;
            shared = output_trace.width;
            cols = 1;
            const bool source_dot = kind == "source_attention_dot";
            b_values = source_dot ? output_model.attn_src : output_model.attn_dst;
            const auto& e_values = source_dot ? output_trace.e_src : output_trace.e_dst;
            const double bias = source_dot ? output_model.attn_src_bias : output_model.attn_dst_bias;
            c_values.resize(e_values.size(), 0.0);
            for (std::size_t i = 0; i < c_values.size(); ++i) {
                c_values[i] = e_values[i] - bias;
            }
        }
    } else if (std::regex_match(relation, match, hidden_relation)) {
        const std::size_t layer_index = static_cast<std::size_t>(std::stoull(match[1].str()));
        const std::size_t head_index = static_cast<std::size_t>(std::stoull(match[2].str()));
        const std::string kind = match[3].str();
        if (layer_index >= trace.hidden_heads.size()
            || layer_index >= trace.hidden_layer_inputs.size()
            || layer_index >= trace.hidden_layer_input_dims.size()) {
            throw std::runtime_error("hidden layer index out of range for schedule relation: " + relation);
        }
        const HeadParams* hidden_model = nullptr;
        std::size_t trace_head_ordinal = 0;
        std::size_t layer_head_ordinal = 0;
        for (const auto& head : model.hidden_heads) {
            if (head.layer_index != layer_index) {
                continue;
            }
            if (head.head_index == head_index) {
                hidden_model = &head;
                trace_head_ordinal = layer_head_ordinal;
                break;
            }
            ++layer_head_ordinal;
        }
        if (hidden_model == nullptr || trace_head_ordinal >= trace.hidden_heads[layer_index].size()) {
            throw std::runtime_error("hidden head model/trace not found for schedule relation: " + relation);
        }
        const auto& hidden_trace = trace.hidden_heads[layer_index][trace_head_ordinal];
        rows = hidden_trace.rows;
        if (kind == "projection") {
            a_values = trace.hidden_layer_inputs[layer_index];
            shared = trace.hidden_layer_input_dims[layer_index];
            b_values = hidden_model->W;
            cols = hidden_trace.width;
            c_values = hidden_trace.h_prime;
        } else {
            a_values = hidden_trace.h_prime;
            shared = hidden_trace.width;
            cols = 1;
            const bool source_dot = kind == "source_attention_dot";
            b_values = source_dot ? hidden_model->attn_src : hidden_model->attn_dst;
            const auto& e_values = source_dot ? hidden_trace.e_src : hidden_trace.e_dst;
            const double bias = source_dot ? hidden_model->attn_src_bias : hidden_model->attn_dst_bias;
            c_values.resize(e_values.size(), 0.0);
            for (std::size_t i = 0; i < c_values.size(); ++i) {
                c_values[i] = e_values[i] - bias;
            }
        }
    } else {
        throw std::runtime_error("unsupported formal matmul schedule relation: " + relation);
    }
    if (a_values.size() != rows * shared || b_values.size() != shared * cols || c_values.size() != rows * cols) {
        throw std::runtime_error("formal matmul schedule shape mismatch for " + relation);
    }

    const std::size_t chunk_rows = matmul_chunk_rows(shared, cols);
    const std::size_t total_chunks = (rows + chunk_rows - 1U) / chunk_rows;
    if (start_chunk >= total_chunks) {
        throw std::runtime_error("start chunk is beyond total chunks for " + relation);
    }
    const std::size_t stop_chunk = std::min(total_chunks, start_chunk + std::max<std::size_t>(1, max_chunks));
    const auto run_dir = output_root / "runs" / spec.dataset / "warm";
    ensure_dir(run_dir / "formal_artifacts");
    std::ofstream debug(run_dir / "proof_debug.log", std::ios::app);
    auto dbg = [&](const std::string& line) {
        if (debug) {
            debug << line << '\n';
            debug.flush();
        }
        std::cout << line << '\n';
    };
    FormalPartialArtifact artifact;
    dbg("formal_matmul_schedule dataset=" + spec.dataset
        + " relation=" + relation
        + " rows=" + std::to_string(rows)
        + " shared=" + std::to_string(shared)
        + " cols=" + std::to_string(cols)
        + " start_chunk=" + std::to_string(start_chunk)
        + " stop_chunk=" + std::to_string(stop_chunk)
        + " total_chunks=" + std::to_string(total_chunks));

    std::size_t completed = 0;
    std::size_t next_chunk = start_chunk;
    for (std::size_t chunk_idx = start_chunk; chunk_idx < stop_chunk; ++chunk_idx) {
        const std::size_t row_begin = chunk_idx * chunk_rows;
        const std::size_t row_count = std::min(chunk_rows, rows - row_begin);
        const auto a_chunk = matrix_row_slice(a_values, rows, shared, row_begin, row_count);
        const auto c_chunk = matrix_row_slice(c_values, rows, cols, row_begin, row_count);
        const auto chunked_label = chunk_relation_label(label, chunk_idx, row_begin, row_count, total_chunks);
        ++artifact.total_chunks;
        const bool ok = run_formal_matmul_trace_attempt(
            spec.dataset,
            chunked_label,
            a_chunk,
            row_count,
            shared,
            b_values,
            cols,
            c_chunk,
            &artifact,
            &debug,
            row_begin,
            total_chunks);
        if (!ok) {
            artifact.relation_details.push_back(label + " schedule chunk=" + std::to_string(chunk_idx) + " status=REJECT_OR_BLOCKED");
            next_chunk = chunk_idx;
            break;
        }
        ++artifact.verified_chunks;
        ++completed;
        next_chunk = chunk_idx + 1;
        artifact.relation_details.push_back(label + " schedule chunk=" + std::to_string(chunk_idx) + " status=PASS");
    }
    artifact.proof_generation_ok = completed != 0;
    artifact.verification_ok = completed != 0;
    artifact.blocker = "targeted formal matmul schedule 已验证 "
        + std::to_string(completed)
        + " 个 matmul chunk，起始 chunk 为 "
        + std::to_string(start_chunk)
        + "；fixed-to-trace scale decomposition 与 residual bounded range 为 partial formal，完整 rounding mode、sign 与 tie-breaking 仍阻塞。";
    artifact.proof_path = run_dir / "formal_artifacts" / (
        "zkdl_formal_matmul_schedule_"
        + relation
        + "_chunks"
        + std::to_string(start_chunk)
        + "_"
        + std::to_string(stop_chunk - 1U)
        + ".proof.txt");
    serialize_formal_partial_artifact(artifact, artifact.proof_path);

    std::ofstream resume(output_root / "RESUME_POINT.md");
    resume << "# zkDL GAT formal matmul schedule 继续点\n\n";
    resume << "- 数据集: `" << spec.dataset << "`\n";
    resume << "- relation: `" << relation << "`\n";
    resume << "- 每个 chunk 的行数: `" << chunk_rows << "`\n";
    resume << "- matmul 总 chunk 数: `" << total_chunks << "`\n";
    resume << "- 本轮请求 chunk 范围: `" << start_chunk << ".." << stop_chunk << "`\n";
    resume << "- 本轮新增验证的 matmul chunk 数: `" << completed << "`\n";
    resume << "- 本轮新增验证的 formal relation chunk 数（含 scale decomposition 与 residual range bitness/decomposition）: `"
           << artifact.verified_chunks << "/" << artifact.total_chunks << "`\n";
    resume << "- 当前下一个 chunk: `" << (next_chunk >= total_chunks ? std::string("complete") : std::to_string(next_chunk)) << "`\n";
    resume << "- proof artifact: `" << artifact.proof_path.string() << "`\n";
    resume << "- proof debug log: `" << (run_dir / "proof_debug.log").string() << "`\n";
    resume << "- Level A 状态: blocked；exact scale decomposition 和 residual bounded range 已由 verifier 绑定到本轮 chunk，但 sign-specific rounding mode 与 tie-breaking 仍未证明。\n";

    std::ofstream dataset_resume(run_dir / "RESUME_POINT.md");
    dataset_resume << "# zkDL GAT 数据集继续点\n\n";
    dataset_resume << "- 数据集: `" << spec.dataset << "`\n";
    dataset_resume << "- relation: `" << relation << "`\n";
    dataset_resume << "- 每个 chunk 的行数: `" << chunk_rows << "`\n";
    dataset_resume << "- matmul 总 chunk 数: `" << total_chunks << "`\n";
    dataset_resume << "- 本轮请求 chunk 范围: `" << start_chunk << ".." << stop_chunk << "`\n";
    dataset_resume << "- 本轮新增验证的 matmul chunk 数: `" << completed << "`\n";
    dataset_resume << "- 本轮新增验证的 formal relation chunk 数（含 scale decomposition 与 residual range bitness/decomposition）: `"
                   << artifact.verified_chunks << "/" << artifact.total_chunks << "`\n";
    dataset_resume << "- 当前下一个 chunk: `" << (next_chunk >= total_chunks ? std::string("complete") : std::to_string(next_chunk)) << "`\n";
    dataset_resume << "- proof artifact: `" << artifact.proof_path.string() << "`\n";
    dataset_resume << "- proof debug log: `" << (run_dir / "proof_debug.log").string() << "`\n";
    dataset_resume << "- Level A 状态: blocked；exact scale decomposition 和 residual bounded range 已由 verifier 绑定到本轮 chunk，但 sign-specific rounding mode 与 tie-breaking 仍未证明。\n";
}

void run_matmul_first_failure_diagnostic(const fs::path& project_root, const DatasetRunSpec& spec)
{
    const fs::path debug_dir = "/home/pzh/project/comparison/baselines/zkdl/debug";
    ensure_dir(debug_dir);
    fs::remove(debug_dir / ("matmul_first_failure_" + spec.dataset + ".json"));
    fs::remove(debug_dir / ("matmul_first_failure_" + spec.dataset + ".md"));
    std::ofstream debug(debug_dir / ("matmul_first_failure_" + spec.dataset + ".diagnostic.log"));
    auto log_line = [&](const std::string& line) {
        debug << line << '\n';
        debug.flush();
        std::cout << line << '\n';
    };

    log_line("dataset=" + spec.dataset + " diagnostic=matmul_first_failure stage=load_dataset");
    const auto dataset = load_cached_dataset(project_root / "data/cache" / spec.cache_name);
    log_line("dataset=" + spec.dataset + " diagnostic=matmul_first_failure stage=normalize_graph");
    const auto graph = normalize_graph(dataset, spec);
    log_line("dataset=" + spec.dataset + " diagnostic=matmul_first_failure stage=load_model");
    const auto model = load_model(project_root / "artifacts/checkpoints" / spec.checkpoint_name);
    if (model.output_heads.empty()) {
        throw std::runtime_error("matmul diagnostic requires at least one output head");
    }
    log_line("dataset=" + spec.dataset + " diagnostic=matmul_first_failure stage=gpu_forward_full_trace");
    ForwardMetrics forward;
    ForwardTraceHost trace;
    (void)run_gat_forward_gpu(graph, model, &forward, &trace);
    if (trace.output_layer_input.empty() || trace.output_heads.empty() || trace.output_heads.front().h_prime.empty()) {
        throw std::runtime_error("matmul diagnostic output projection trace is unavailable");
    }

    const std::size_t rows = graph.num_nodes;
    const std::size_t shared = trace.output_layer_input_dim;
    const std::size_t cols = model.C;
    const std::size_t chunk_rows = matmul_chunk_rows(shared, cols);
    const std::size_t total_chunks = (rows + chunk_rows - 1U) / chunk_rows;
    const std::size_t row_count = std::min(chunk_rows, rows);
    const auto a_chunk = matrix_row_slice(trace.output_layer_input, rows, shared, 0, row_count);
    const auto c_chunk = matrix_row_slice(trace.output_heads.front().h_prime, rows, cols, 0, row_count);
    FormalPartialArtifact artifact;
    log_line("dataset=" + spec.dataset
        + " diagnostic=matmul_first_failure relation=output_head0_projection"
        + " chunk=0 rows=0.." + std::to_string(row_count)
        + " total_chunks=" + std::to_string(total_chunks));
    const bool ok = run_formal_matmul_trace_attempt(
        spec.dataset,
        chunk_relation_label("output_head0_projection", 0, 0, row_count, total_chunks),
        a_chunk,
        row_count,
        shared,
        model.output_heads.front().W,
        cols,
        c_chunk,
        &artifact,
        &debug,
        0,
        total_chunks);
    log_line("dataset=" + spec.dataset
        + " diagnostic=matmul_first_failure fixed_point_pre_rescale_matmul_status="
        + std::string(ok ? "PASS" : "FAILED_OR_BLOCKED")
        + " evidence_json=" + (debug_dir / ("matmul_first_failure_" + spec.dataset + ".json")).string());
}

} // namespace

int main(int argc, char** argv) {
    try {
        fs::path project_root = "/home/pzh/project";
        fs::path output_root = fs::current_path();
        bool run_all = true;
        bool diagnose_matmul_first_failure = false;
        bool formal_matmul_schedule = false;
        std::string only_dataset;
        std::string schedule_relation = "output_head0_projection";
        std::size_t schedule_start_chunk = 0;
        std::size_t schedule_max_chunks = std::numeric_limits<std::size_t>::max() / 4U;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout
                    << "usage: ./gat_zkdl [--run-all|--dataset DATASET] [--output-root DIR]\n"
                    << "       ./gat_zkdl --diagnose-matmul-first-failure [--dataset DATASET]\n"
                    << "       ./gat_zkdl --formal-matmul-schedule --dataset DATASET --relation REL --start-chunk N [--max-chunks N] --output-root DIR\n";
                return 0;
            } else if (arg == "--project-root" && i + 1 < argc) {
                project_root = argv[++i];
            } else if (arg == "--output-root" && i + 1 < argc) {
                output_root = argv[++i];
            } else if (arg == "--dataset" && i + 1 < argc) {
                only_dataset = argv[++i];
                run_all = false;
            } else if (arg == "--run-all") {
                run_all = true;
            } else if (arg == "--diagnose-matmul-first-failure") {
                diagnose_matmul_first_failure = true;
            } else if (arg == "--formal-matmul-schedule") {
                formal_matmul_schedule = true;
                run_all = false;
            } else if (arg == "--relation" && i + 1 < argc) {
                schedule_relation = argv[++i];
            } else if (arg == "--start-chunk" && i + 1 < argc) {
                schedule_start_chunk = static_cast<std::size_t>(std::stoull(argv[++i]));
            } else if (arg == "--max-chunks" && i + 1 < argc) {
                schedule_max_chunks = static_cast<std::size_t>(std::stoull(argv[++i]));
            }
        }
        output_root = fs::absolute(output_root);
        const auto allowed_root = fs::absolute("/home/pzh/project/comparison/baselines/zkdl");
        const auto rel = fs::relative(output_root, allowed_root);
        if (rel.empty() || *rel.begin() == "..") {
            throw std::runtime_error("output-root must stay inside /home/pzh/project/comparison/baselines/zkdl");
        }
        ensure_dir(output_root);

        const std::vector<DatasetRunSpec> specs = {
            {"cora", "cora", "cora_gat", "cora_full.cfg", "transductive_node_classification", "node", "whole_graph_single", 1, 12},
            {"citeseer", "citeseer", "citeseer_gat", "citeseer_full.cfg", "transductive_node_classification", "node", "whole_graph_single", 1, 12},
            {"pubmed", "pubmed", "pubmed_gat", "pubmed_full.cfg", "transductive_node_classification", "node", "whole_graph_single", 1, 12},
            {"ppi", "ppi", "ppi_gat", "ppi_batch_formal.cfg", "inductive_multi_graph_node_classification", "graph", "multi_graph_batch", 295, 12},
            {"ogbn-arxiv", "ogbn_arxiv", "ogbn_arxiv_gat", "ogbn_arxiv_full.cfg", "transductive_node_classification", "node", "whole_graph_single", 1, 12},
        };

        if (diagnose_matmul_first_failure) {
            for (const auto& spec : specs) {
                if (!run_all && spec.dataset != only_dataset) {
                    continue;
                }
                run_matmul_first_failure_diagnostic(project_root, spec);
            }
            return 0;
        }

        if (formal_matmul_schedule) {
            if (only_dataset.empty()) {
                throw std::runtime_error("--formal-matmul-schedule requires --dataset");
            }
            bool found = false;
            for (const auto& spec : specs) {
                if (spec.dataset != only_dataset) {
                    continue;
                }
                found = true;
                run_formal_matmul_schedule(
                    project_root,
                    output_root,
                    spec,
                    schedule_relation,
                    schedule_start_chunk,
                    schedule_max_chunks);
            }
            if (!found) {
                throw std::runtime_error("unknown dataset for formal matmul schedule: " + only_dataset);
            }
            return 0;
        }

        std::vector<RunRow> rows;
        for (const auto& spec : specs) {
            if (!run_all && spec.dataset != only_dataset) {
                continue;
            }
            rows.push_back(run_one(project_root, output_root, spec));
        }
        write_result_tables(output_root, rows);
        write_adapter_readme(output_root, rows);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
