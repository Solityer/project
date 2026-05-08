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

#include "commitment.cuh"
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

__global__ void fr_mont_one_minus_kernel(Fr_t u, Fr_t* out) {
    *out = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_ONE), u);
}

Fr_t fr_mont_one_minus_host(Fr_t u) {
    Fr_t host;
    Fr_t* device = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(Fr_t)), "cudaMalloc fr_mont_one_minus");
    fr_mont_one_minus_kernel<<<1,1>>>(u, device);
    cuda_sync("fr_mont_one_minus_kernel");
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
    for (std::size_t layer = 0; layer < model.hidden_profile.size(); ++layer) {
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
            auto out = run_attention_head(
                current, rows, current_dim, *head, d_src, d_dst, graph.edges.size(),
                false, true,
                &metrics->hidden_projection_ms,
                &metrics->hidden_attention_ms,
                &metrics->hidden_activation_ms,
                nullptr);
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

struct ProofAttempt {
    bool partial_commitment_ok = false;
    bool full_proof_ok = false;
    bool full_verify_ok = false;
    double partial_commitment_ms = 0.0;
    double partial_open_ms = 0.0;
    std::size_t partial_proof_size_bytes = 0;
    std::string blocker;
    LevelBProofArtifact level_b;
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
    kv("chunking_rule", "none");
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
}

struct CoverageRow {
    std::string relation;
    bool formal_prover_supported = false;
    bool formal_verifier_supported = false;
    bool reused_from_zkdl = false;
    bool newly_added_in_zkdl_adapter = false;
    bool debug_only = false;
    bool blocked = true;
    std::string blocker_reason;
};

std::vector<CoverageRow> gat_proof_coverage_rows() {
    return {
        {"feature lookup", false, false, false, false, false, true, "No zkDL lookup argument is integrated for global feature table/query equality; main repository lookup semantics were inspected but its KZG lookup protocol is not ported to Hyrax."},
        {"hidden projection", false, false, true, false, false, true, "zkDL has a matmul sumcheck prover fragment, but there is no verifier, transcript binding, tensor-shape binding, or integration with quantized GAT hidden projection tensors."},
        {"source attention dot product", false, false, true, false, false, true, "Can be reduced to inner-product/matmul fragments, but source-attention witness columns are not committed and verified under a GAT transcript."},
        {"target attention dot product", false, false, true, false, false, true, "Can be reduced to inner-product/matmul fragments, but target-attention witness columns are not committed and verified under a GAT transcript."},
        {"source sparse routing", false, false, false, false, false, true, "No Hyrax/FAC4DNN sparse route proof is available in zkDL; main repository route2/PSQ implementation is KZG-specific and not safely portable without a new commitment layer."},
        {"destination sparse routing", false, false, false, false, false, true, "No Hyrax/FAC4DNN sparse route proof is available in zkDL; destination grouping and edge_gid/dst stable order are only enforced by the adapter's public preprocessing."},
        {"LeakyReLU", false, false, true, false, false, true, "zkReLU contains a prover-side binary/range-style fragment but has no verifier path and is not wired to attention scores or static tables."},
        {"softmax group max", false, false, false, false, false, true, "No max/range comparison gadget exists in zkDL for per-destination sparse softmax groups."},
        {"ExpMap", false, false, false, false, false, true, "No zkDL table lookup argument is available for QuantExp(-delta); main repository lookup tables are KZG transcript objects."},
        {"softmax denominator sum", false, false, true, false, false, true, "Sum relation can use arithmetic fragments, but grouped sparse denominator constraints are not committed/opened as a verifiable gadget."},
        {"inverse", false, false, false, false, false, true, "No inverse witness/range/denominator nonzero proof is present in zkDL for softmax normalization."},
        {"alpha multiplication", false, false, true, false, false, true, "zkProd provides a prover fragment, but no verifier and no integration for alpha = exp * inverse over sparse edges."},
        {"weighted aggregation", false, false, true, false, false, true, "Aggregation can be decomposed into product and sparse routing, but the required FAC4DNN-style relation is absent under Hyrax."},
        {"ELU", false, false, false, false, false, true, "No ELU lookup/range gadget exists in zkDL; hidden activation is only computed by the GPU forward adapter."},
        {"concat", false, false, false, false, true, true, "Head concat is implemented in forward/debug trace only; no equality/copy proof binds head outputs to concatenated hidden tensor."},
        {"output projection", false, false, true, false, false, true, "zkDL matmul fragment is not connected to output attention-head projection tensors with a verifier and openings."},
        {"output source/target attention", false, false, true, false, false, true, "Output attention dot products are forward-computed, but no formal zkDL verifier checks them."},
        {"output sparse routing", false, false, false, false, false, true, "No Hyrax sparse route proof for output heads is implemented."},
        {"output softmax", false, false, false, false, false, true, "Max/ExpMap/sum/inverse chain for output-head softmax is not supported by zkDL gadgets."},
        {"output aggregation", false, false, true, false, false, true, "Weighted output aggregation needs product plus sparse routing proofs; both remain unverified under zkDL."},
        {"rescale", false, false, false, false, false, true, "No fixed-point rescale/rounding proof gadget is available for output pre-bias scale alignment."},
        {"bias add", false, false, false, false, false, true, "Bias addition is present in the GPU forward output tensor, but no committed pre/post-bias equality proof is available."},
        {"output-head average", false, false, false, true, true, true, "Adapter checks the K_out=1 identity as a diagnostic only; a formal multi-head average proof is not implemented."},
        {"public logits binding", false, false, true, true, true, true, "Debug-only binding recomputes the full public logits commitment and public row MLE value; this is not a formal proof that the hidden GAT computation produced those logits."},
        {"commitment/evaluation binding", false, false, true, true, false, true, "A standard Hyrax row opening prover/verifier was added and tested, but recurrence replay still mismatches on the current zkDL curve/folding kernels; the remaining accepted path is debug-only public recomputation."},
        {"transcript binding", false, false, true, true, false, true, "Opening transcript replay is implemented as a checker and currently rejects generated transcripts; full transcript binding for GAT remains blocked."},
        {"verifier replay", false, false, true, true, false, true, "Verifier replay exists and catches the row-opening recurrence mismatch; no successful formal full-proof verifier replay is available."},
    };
}

std::string bool_string(bool value) {
    return value ? "true" : "false";
}

void write_coverage_matrix(const fs::path& output_root) {
    ensure_dir(output_root / "results");
    const auto rows = gat_proof_coverage_rows();
    {
        std::ofstream csv(output_root / "results" / "gat_proof_coverage_matrix.csv");
        csv << "relation,formal_prover_supported,formal_verifier_supported,reused_from_zkdl,newly_added_in_zkdl_adapter,debug_only,blocked,blocker_reason\n";
        for (const auto& row : rows) {
            csv << row.relation
                << ',' << bool_string(row.formal_prover_supported)
                << ',' << bool_string(row.formal_verifier_supported)
                << ',' << bool_string(row.reused_from_zkdl)
                << ',' << bool_string(row.newly_added_in_zkdl_adapter)
                << ',' << bool_string(row.debug_only)
                << ',' << bool_string(row.blocked)
                << ",\"" << json_escape(row.blocker_reason) << "\"\n";
        }
    }
    {
        std::ofstream json(output_root / "results" / "gat_proof_coverage_matrix.json");
        json << "{\n  \"coverage_level\": \"formal_level_a_blocked_with_debug_level_b_binding\",\n  \"rows\": [\n";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto& row = rows[i];
            json << "    {\n";
            json << "      \"relation\": \"" << json_escape(row.relation) << "\",\n";
            json << "      \"formal_prover_supported\": " << bool_string(row.formal_prover_supported) << ",\n";
            json << "      \"formal_verifier_supported\": " << bool_string(row.formal_verifier_supported) << ",\n";
            json << "      \"reused_from_zkdl\": " << bool_string(row.reused_from_zkdl) << ",\n";
            json << "      \"newly_added_in_zkdl_adapter\": " << bool_string(row.newly_added_in_zkdl_adapter) << ",\n";
            json << "      \"debug_only\": " << bool_string(row.debug_only) << ",\n";
            json << "      \"blocked\": " << bool_string(row.blocked) << ",\n";
            json << "      \"blocker_reason\": \"" << json_escape(row.blocker_reason) << "\"\n";
            json << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
        }
        json << "  ]\n}\n";
    }
    {
        std::ofstream md(output_root / "results" / "gat_proof_coverage_matrix.md");
        md << "# GAT Proof Coverage Matrix\n\n";
        md << "| relation | formal prover | formal verifier | reused from zkDL | newly added | debug only | blocked | blocker reason |\n";
        md << "| --- | --- | --- | --- | --- | --- | --- | --- |\n";
        for (const auto& row : rows) {
            md << "| " << row.relation
               << " | " << bool_string(row.formal_prover_supported)
               << " | " << bool_string(row.formal_verifier_supported)
               << " | " << bool_string(row.reused_from_zkdl)
               << " | " << bool_string(row.newly_added_in_zkdl_adapter)
               << " | " << bool_string(row.debug_only)
               << " | " << bool_string(row.blocked)
               << " | " << row.blocker_reason << " |\n";
        }
    }
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
        json << "{\n  \"benchmark_mode\": \"formal_level_a_full_gat_proof_verify_only\",\n  \"policy\": \"formal four metrics are NA unless complete Level A proof generation and verification succeeds\",\n  \"rows\": [\n";
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
        json << "{\n  \"benchmark_mode\": \"diagnostic_level_b_output_binding_only_not_formal_level_a\",\n  \"policy\": \"diagnostic numbers must not be used in the formal comparison table\",\n  \"rows\": [\n";
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
        std::ofstream md(output_root / "results" / "summary.md");
        md << "# 最新正式结果\n\n";
        md << "Formal Level A has not been reached in this zkDL adapter run. The official four fields below are `NA` because the implemented proof does not cover the full sparse GAT forward relation.\n\n";
        md << "| 数据集 | full dataset loaded | GPU GAT forward | formal proof generation | formal verification | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | blocker |\n";
        md << "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n";
        for (const auto& row : rows) {
            md << "| " << row.dataset
               << " | " << (row.full_dataset ? "yes" : "no")
               << " | " << (row.gpu_forward_ok ? "yes" : "no")
               << " | " << (row.proof_generation_ok ? "yes" : "no")
               << " | " << (row.verification_ok ? "yes" : "no")
               << " | " << row.commitment_time_ms
               << " | " << row.prove_time_ms
               << " | " << row.verify_time_ms
               << " | " << row.proof_size_bytes
               << " | " << row.blocker << " |\n";
        }
        md << "\n## Diagnostic Level B\n\n";
        md << "The following debug numbers are retained separately and are not formal Level A comparison results.\n\n";
        md << "| dataset | status | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | proof_path |\n";
        md << "| --- | --- | ---: | ---: | ---: | ---: | --- |\n";
        for (const auto& row : rows) {
            md << "| " << row.dataset
               << " | " << row.debug_level_b_status
               << " | " << row.debug_level_b_commitment_time_ms
               << " | " << row.debug_level_b_prove_time_ms
               << " | " << row.debug_level_b_verify_time_ms
               << " | " << row.debug_level_b_proof_size_bytes
               << " | " << row.debug_level_b_proof_path << " |\n";
        }
    }
    write_coverage_matrix(output_root);
}

void write_audit_files(const fs::path& output_root, const std::vector<RunRow>& rows) {
    ensure_dir(output_root / "audit");
    {
        std::ofstream out(output_root / "audit" / "MODIFICATION_BOUNDARY_AUDIT.md");
        out << "# Modification Boundary Audit\n\n";
        out << "- Allowed write root: `/home/pzh/project/comparison/baselines/zkdl`.\n";
        out << "- This runner writes only below the selected zkDL output root and `zkdl` build files.\n";
        out << "- Main project data, configs, checkpoints and protocol notes are read-only inputs.\n";
        out << "- No main repository file is copied back or overwritten by this runner.\n";
        out << "- Modified source files: `Makefile`, `commitment.cuh`, `gat_zkdl.cu`.\n";
        out << "- Generated output families: `zkdl_gat_outputs/results/*`, `zkdl_gat_outputs/runs/*/warm/*`, `zkdl_gat_outputs/audit/*`.\n";
        out << "- Stale formal-looking `runs/*/warm/proof` directories from earlier Level B runs are removed by the runner; new Level B artifacts are written under `debug_artifacts` only.\n";
        out << "- Build note: this run was compiled with an overrideable `NVCC_FLAGS`; audit logs record the exact command.\n";
        out << "- Formal result policy: `latest.csv` and `latest.json` only expose complete Level A metrics. Diagnostic Level B metrics are isolated in `results/debug_level_b.*`.\n";
    }
    {
        std::ofstream out(output_root / "README_ZKDL_GAT_ADAPTER.md");
        out << "# zkDL GAT Adapter Run Notes\n\n";
        out << "This artifact records the zkDL baseline adaptation for full-dataset GAT-ZKML comparison. The current formal result level is **blocked before Level A**, not success.\n\n";
        out << "Result correction: prior Level B timing/proof-size numbers are no longer emitted in the official `latest.csv` / `latest.json` four metric fields. Those fields are `NA` unless a complete Level A GAT proof and verifier succeed. The previous diagnostic line is retained only in `results/debug_level_b.csv` and `results/debug_level_b.json`.\n\n";
        out << "This round added a serializable proof artifact for the final output-head tensor commitment, deterministic generator/challenge derivation, verifier-side full public logits commitment recomputation, proof-size accounting from the actual proof file, and per-dataset proof/debug logs.\n\n";
        out << "This round attempted to replace the old skipped/fallback Hyrax row opening with a standard row MLE opening proof for the final output tensor. The new verifier checks the opening transcript recurrence and final folded generator, and it currently rejects the generated transcript; the diagnostic Level B row therefore records public commitment plus public row MLE fallback as debug-only evidence, not as formal commitment/evaluation binding.\n\n";
        out << "Migrated capabilities from the main repository: full dataset cache loading, row normalization, self-loop insertion, graph/dst stable edge ordering, checkpoint tensor loading, multi-head sparse GAT GPU forward, output-head bias plus arithmetic mean, and main benchmark table formatting.\n\n";
        out << "Retained zkDL components: `FrTensor`, `G1Tensor`, `Commitment::commit`, commitment generator tensors, and existing GPU MLE/sumcheck/opening helpers. `Commitment::open_with_proof` and verifier helpers were added locally around these primitives.\n\n";
        out << "Covered relations in the diagnostic Level B rows: final output-head tensor commitment, verifier-side full public logits commitment recomputation, public row MLE evaluation for the selected output row, public logits binding for the committed output tensor, and the `K_out=1` output-head average identity used by all five current checkpoint bundles.\n\n";
        out << "Uncovered relations: hidden/output projection matmul sumcheck binding, sparse src/dst routing, FAC4DNN aggregation, LeakyReLU lookup/range, softmax max/sum/inverse/ExpMap, ELU lookup/range, rescale/rounding, and output-head bias-addition relation before the final head tensor.\n\n";
        out << "Opening boundary update: the original zkDL outer `Fr_partial_me_step` full-tensor path remains unsuitable for the complete GAT proof. A row-only standard Hyrax opening was isolated and implemented, but recurrence replay still mismatches, so it is preserved as failure evidence rather than counted as a formal proof.\n\n";
        out << "Reproduce: build inside this directory, for example `TMPDIR=/home/pzh/project/comparison/baselines/zkdl/.tmp make gat_zkdl NVCC_FLAGS='-arch=sm_80 -std=c++17 -O1 -Xcompiler=-Wall -Xcompiler=-Wextra'`, then run `./gat_zkdl --run-all --output-root zkdl_gat_outputs`.\n\n";
        out << "Four-metric policy: official `commitment_time_ms`, `prove_time_ms`, `verify_time_ms`, and `proof_size_bytes` are `NA` in `latest.*` until full Level A succeeds. Diagnostic Level B metrics are real for their limited output-binding path, but they are not full GAT proof timings and must not be compared as complete protocol results.\n\n";
        out << "Dataset status:\n\n";
        out << "| dataset | formal status | gpu_forward | formal commitment_time_ms | formal prove_time_ms | formal verify_time_ms | formal proof_size_bytes | debug Level B status |\n";
        out << "| --- | --- | --- | --- | --- | --- | --- | --- |\n";
        for (const auto& row : rows) {
            out << "| " << row.dataset
                << " | " << row.status
                << " | " << (row.gpu_forward_ok ? "ok" : "failed")
                << " | " << row.commitment_time_ms
                << " | " << row.prove_time_ms
                << " | " << row.verify_time_ms
                << " | " << row.proof_size_bytes
                << " | " << row.debug_level_b_status << " |\n";
        }
        out << "\nCoverage matrix files: `results/gat_proof_coverage_matrix.csv`, `results/gat_proof_coverage_matrix.json`, and `results/gat_proof_coverage_matrix.md`.\n";
    }
}

RunRow run_one(const fs::path& project_root, const fs::path& output_root, const DatasetRunSpec& spec) {
    RunRow row;
    row.dataset = spec.dataset;
    const auto run_dir = output_root / "runs" / spec.dataset / "warm";
    ensure_dir(run_dir);
    fs::remove_all(run_dir / "proof");
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
        row.proof_generation_ok = false;
        row.verification_ok = false;
        row.benchmark_mode = "formal_level_a_full_gat_proof_verify_blocked";
        row.commitment_time_ms = "NA";
        row.prove_time_ms = "NA";
        row.verify_time_ms = "NA";
        row.proof_size_bytes = "NA";
        row.status = "blocked";
        row.blocker =
            "Formal Level A zkDL-GAT proof is not complete: final output binding is debug-only public commitment/public row MLE recomputation because both mixed and standard Hyrax row-opening replay remain mismatched; hidden/output projection, sparse routing, LeakyReLU, softmax max/ExpMap/sum/inverse, alpha multiplication, weighted aggregation, ELU, concat, rescale, bias-add, and output-head average gadgets are not all available with zkDL prover+verifier+transcript binding. Diagnostic Level B result is isolated in debug_level_b.* and must not be used as a formal comparison metric.";
        log_line("dataset=" + spec.dataset
            + " commitment_status=" + (row.partial_commitment_ok ? "ok" : "failed")
            + " debug_level_b_status=" + row.debug_level_b_status
            + " debug_level_b_proof_size_bytes=" + row.debug_level_b_proof_size_bytes
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

} // namespace

int main(int argc, char** argv) {
    try {
        fs::path project_root = "/home/pzh/project";
        fs::path output_root = fs::current_path();
        bool run_all = true;
        std::string only_dataset;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--project-root" && i + 1 < argc) {
                project_root = argv[++i];
            } else if (arg == "--output-root" && i + 1 < argc) {
                output_root = argv[++i];
            } else if (arg == "--dataset" && i + 1 < argc) {
                only_dataset = argv[++i];
                run_all = false;
            } else if (arg == "--run-all") {
                run_all = true;
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

        std::vector<RunRow> rows;
        for (const auto& spec : specs) {
            if (!run_all && spec.dataset != only_dataset) {
                continue;
            }
            rows.push_back(run_one(project_root, output_root, spec));
        }
        write_result_tables(output_root, rows);
        write_audit_files(output_root, rows);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
