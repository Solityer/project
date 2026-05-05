#include "zkdl_gat/baseline.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace zkdl_gat {
namespace {

struct DatasetMeta {
    std::string name;
    std::size_t num_nodes = 0;
    std::size_t num_features = 0;
    std::size_t num_classes = 0;
    std::size_t graph_count = 1;
    std::string task_type = "transductive_node_classification";
    std::string report_unit = "node";
};

struct Edge {
    std::size_t graph_id = 0;
    std::size_t src = 0;
    std::size_t dst = 0;
    std::size_t stable_index = 0;
};

struct NormalizedPublicInput {
    std::size_t node_count = 0;
    std::size_t edge_count = 0;
    std::size_t graph_count = 1;
    std::vector<std::size_t> node_ptr;
    std::vector<std::size_t> edge_ptr;
    std::vector<Edge> edges;
    bool edge_order_valid = true;
    std::size_t dst_group_count = 0;
    std::size_t max_dst_group_size = 0;
};

struct HiddenProfile {
    std::size_t head_count = 0;
    std::size_t head_dim = 0;
};

struct TensorHeader {
    std::vector<std::size_t> shape;
    std::size_t value_count = 0;
};

struct NpyHeader {
    std::vector<std::size_t> shape;
    std::string descr;
    bool fortran_order = false;
};

struct ProofStatementPlan {
    HiddenProfile hidden;
    std::size_t range_bits = 12;
    std::size_t node_domain = 0;
    std::size_t edge_domain = 0;
    std::size_t feature_domain = 0;
    std::size_t input_dim_domain = 0;
    std::size_t hidden_dim_domain = 0;
    std::size_t concat_dim_domain = 0;
    std::size_t class_domain = 0;
    std::size_t tensor_bound = 0;
    std::size_t checkpoint_tensor_count = 0;
    std::size_t checkpoint_value_count = 0;
    bool hidden_projection_prefix_complete = false;
    std::size_t hidden_projection_prefix_values = 0;
    std::size_t hidden_projection_feature_nonzeros = 0;
    double hidden_projection_prefix_time_ms = 0.0;
    std::uint64_t hidden_projection_checksum = 0;
    std::string hidden_projection_kernel_name;
    std::string hidden_projection_block_reason;
    std::vector<std::string> supported_zkdl_primitives;
    std::vector<std::string> unsupported_protocol_requirements;
};

struct TensorValues {
    std::vector<std::size_t> shape;
    std::vector<double> values;
};

struct MatrixD {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> values;

    MatrixD() = default;
    MatrixD(std::size_t row_count, std::size_t col_count)
        : rows(row_count), cols(col_count), values(row_count * col_count, 0.0) {}

    double& at(std::size_t row, std::size_t col) {
        return values[row * cols + col];
    }

    double at(std::size_t row, std::size_t col) const {
        return values[row * cols + col];
    }
};

struct HeadSpecNames {
    std::size_t layer_index = 0;
    std::size_t local_head_index = 0;
    std::size_t head_index = 0;
    std::string seq_kernel;
    std::string attn_dst_kernel;
    std::string attn_dst_bias;
    std::string attn_src_kernel;
    std::string attn_src_bias;
    std::string output_bias;
};

struct AttentionHeadParams {
    MatrixD seq_kernel;
    std::vector<double> attn_dst_kernel;
    double attn_dst_bias = 0.0;
    std::vector<double> attn_src_kernel;
    double attn_src_bias = 0.0;
    std::vector<double> output_bias;
};

struct ModelParams {
    std::vector<std::vector<AttentionHeadParams>> hidden_layers;
    std::vector<AttentionHeadParams> output_heads;
};

struct HeadForwardTrace {
    MatrixD h_prime;
    std::vector<double> e_src;
    std::vector<double> e_dst;
    std::vector<double> s;
    std::vector<double> z;
    std::vector<double> m;
    std::vector<double> delta;
    std::vector<double> u;
    std::vector<double> sum;
    std::vector<double> inv;
    std::vector<double> alpha;
    MatrixD h_agg_pre_bias;
    MatrixD h_agg;
};

struct HiddenLayerTrace {
    MatrixD input;
    std::vector<HeadForwardTrace> heads;
    MatrixD concat;
};

struct ForwardProfile {
    double hidden_projection_ms = 0.0;
    double hidden_attention_ms = 0.0;
    double hidden_activation_ms = 0.0;
    double hidden_concat_ms = 0.0;
    double output_projection_ms = 0.0;
    double output_attention_ms = 0.0;
    double output_activation_ms = 0.0;
};

struct FullForwardTrace {
    ForwardProfile profile;
    std::vector<HiddenLayerTrace> hidden_layers;
    std::vector<HeadForwardTrace> output_heads;
    MatrixD y_lin;
    MatrixD y;
};

struct StageCommitments {
    std::map<std::string, std::string> values;
};

struct ComposedProofResult {
    bool prover_ran = false;
    bool artifact_generated = false;
    bool verifier_called = false;
    bool verified = false;
    bool full_gat_forward_in_zkdl_proof = false;
    bool final_logits_derivation_complete = false;
    std::string artifact_path;
    std::string failure_step;
    std::string block_reason;
    std::vector<std::string> attempted_stages;
    std::vector<std::string> unsupported_protocol_requirements;
    std::optional<double> commitment_time_ms;
    std::optional<double> prove_time_ms;
    std::optional<double> verify_time_ms;
    std::optional<std::size_t> proof_size_bytes;
    std::map<std::string, std::string> stats;
};

std::filesystem::path feature_path(const std::filesystem::path& cache_dir);

std::string trim(std::string value) {
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), value.end());
    return value;
}

bool parse_bool(const std::string& value) {
    return value == "true" || value == "1" || value == "yes";
}

std::string shell_quote(const std::string& path) {
    std::string out = "'";
    for (const char ch : path) {
        if (ch == '\'') {
            out += "'\\''";
        } else {
            out.push_back(ch);
        }
    }
    out.push_back('\'');
    return out;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open file: " + path.string());
    }
    std::ostringstream stream;
    stream << input.rdbuf();
    return stream.str();
}

std::map<std::string, std::string> read_key_value_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open config: " + path.string());
    }
    std::map<std::string, std::string> values;
    std::string line;
    while (std::getline(input, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error("invalid key/value line: " + line);
        }
        values[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }
    return values;
}

std::size_t required_size(const std::map<std::string, std::string>& values, const std::string& key) {
    const auto it = values.find(key);
    if (it == values.end()) {
        throw std::runtime_error("missing numeric config key: " + key);
    }
    return static_cast<std::size_t>(std::stoull(it->second));
}

std::string required_string(const std::map<std::string, std::string>& values, const std::string& key) {
    const auto it = values.find(key);
    if (it == values.end() || it->second.empty()) {
        throw std::runtime_error("missing config key: " + key);
    }
    return it->second;
}

std::string optional_string(const std::map<std::string, std::string>& values, const std::string& key, std::string fallback) {
    const auto it = values.find(key);
    return it == values.end() ? std::move(fallback) : it->second;
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

std::string join_size_vector(const std::vector<std::size_t>& values) {
    std::ostringstream stream;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            stream << ',';
        }
        stream << values[i];
    }
    return stream.str();
}

std::string sha256_file(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        return "";
    }
    const auto command = "sha256sum " + shell_quote(path.string());
    std::array<char, 256> buffer{};
    std::string output;
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("failed to run sha256sum");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }
    const auto rc = pclose(pipe);
    if (rc != 0) {
        throw std::runtime_error("sha256sum failed for " + path.string());
    }
    std::istringstream input(output);
    std::string digest;
    input >> digest;
    return digest;
}

std::size_t count_lines(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open line-count file: " + path.string());
    }
    std::size_t count = 0;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            ++count;
        }
    }
    return count;
}

std::size_t next_power_of_two(std::size_t value) {
    if (value <= 1) {
        return 1;
    }
    std::size_t out = 1;
    while (out < value) {
        out <<= 1U;
    }
    return out;
}

HiddenProfile parse_hidden_profile(const std::string& value) {
    const auto x = value.find('x');
    if (x == std::string::npos) {
        throw std::runtime_error("invalid hidden_profile: " + value);
    }
    HiddenProfile profile;
    profile.head_count = static_cast<std::size_t>(std::stoull(value.substr(0, x)));
    profile.head_dim = static_cast<std::size_t>(std::stoull(value.substr(x + 1)));
    if (profile.head_count == 0 || profile.head_dim == 0) {
        throw std::runtime_error("hidden_profile dimensions must be positive");
    }
    return profile;
}

std::size_t optional_size_from_file(const std::filesystem::path& path, const std::string& key, std::size_t fallback) {
    if (!std::filesystem::exists(path)) {
        return fallback;
    }
    const auto values = read_key_value_file(path);
    const auto it = values.find(key);
    if (it == values.end()) {
        return fallback;
    }
    return static_cast<std::size_t>(std::stoull(it->second));
}

NpyHeader load_npy_header(const std::filesystem::path& path) {
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
        throw std::runtime_error("missing descr in npy header: " + path.string());
    }
    NpyHeader out;
    out.descr = header.substr(descr_quote + 1, descr_end - descr_quote - 1);
    const auto fortran_pos = header.find("fortran_order");
    out.fortran_order = fortran_pos != std::string::npos && header.find("True", fortran_pos) != std::string::npos;
    const auto shape_pos = header.find('(', header.find("shape"));
    const auto shape_end = header.find(')', shape_pos);
    if (shape_pos == std::string::npos || shape_end == std::string::npos) {
        throw std::runtime_error("missing shape in npy header: " + path.string());
    }
    std::stringstream shape_stream(header.substr(shape_pos + 1, shape_end - shape_pos - 1));
    std::string token;
    while (std::getline(shape_stream, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (!token.empty()) {
            out.shape.push_back(static_cast<std::size_t>(std::stoull(token)));
        }
    }
    return out;
}

std::map<std::string, TensorHeader> load_tensor_headers(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open checkpoint tensor dump: " + path.string());
    }
    std::map<std::string, TensorHeader> out;
    std::string marker;
    while (input >> marker) {
        if (marker != "TENSOR") {
            throw std::runtime_error("malformed tensor dump marker in " + path.string());
        }
        std::string name;
        std::size_t rank = 0;
        input >> name >> rank;
        TensorHeader tensor;
        tensor.shape.resize(rank);
        for (std::size_t i = 0; i < rank; ++i) {
            input >> tensor.shape[i];
        }
        input >> tensor.value_count;
        double ignored = 0.0;
        for (std::size_t i = 0; i < tensor.value_count; ++i) {
            input >> ignored;
        }
        out.emplace(std::move(name), std::move(tensor));
    }
    return out;
}

TensorValues load_tensor_values(const std::filesystem::path& path, const std::string& wanted_name) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open checkpoint tensor dump: " + path.string());
    }
    std::string marker;
    while (input >> marker) {
        if (marker != "TENSOR") {
            throw std::runtime_error("malformed tensor dump marker in " + path.string());
        }
        std::string name;
        std::size_t rank = 0;
        input >> name >> rank;
        TensorValues tensor;
        tensor.shape.resize(rank);
        for (std::size_t i = 0; i < rank; ++i) {
            input >> tensor.shape[i];
        }
        std::size_t value_count = 0;
        input >> value_count;
        tensor.values.resize(name == wanted_name ? value_count : 0U);
        double value = 0.0;
        for (std::size_t i = 0; i < value_count; ++i) {
            input >> value;
            if (name == wanted_name) {
                tensor.values[i] = value;
            }
        }
        if (name == wanted_name) {
            return tensor;
        }
    }
    throw std::runtime_error("checkpoint tensor not found: " + wanted_name);
}

std::uint64_t mix_u64(std::uint64_t state, std::uint64_t value) {
    constexpr std::uint64_t prime = 1099511628211ULL;
    state ^= value;
    state *= prime;
    return state;
}

std::uint64_t mix_quantized_projection(std::uint64_t state, double value) {
    const auto scaled = static_cast<std::int64_t>(std::llround(value * 65536.0));
    return mix_u64(state, static_cast<std::uint64_t>(scaled));
}

std::size_t tensor_dim_from_tail(const TensorValues& tensor, std::size_t tail_index) {
    if (tensor.shape.size() <= tail_index) {
        throw std::runtime_error("tensor rank too small");
    }
    return tensor.shape[tensor.shape.size() - 1U - tail_index];
}

std::string first_hidden_kernel_name(const std::filesystem::path& checkpoint_manifest) {
    const auto text = read_file(checkpoint_manifest);
    const auto hidden_pos = text.find("\"hidden_head_specs\"");
    const auto seq_pos = hidden_pos == std::string::npos ? std::string::npos : text.find("\"seq_kernel\"", hidden_pos);
    if (seq_pos != std::string::npos) {
        const auto colon = text.find(':', seq_pos);
        const auto quote = colon == std::string::npos ? std::string::npos : text.find('"', colon);
        const auto end_quote = quote == std::string::npos ? std::string::npos : text.find('"', quote + 1);
        if (quote != std::string::npos && end_quote != std::string::npos) {
            return text.substr(quote + 1, end_quote - quote - 1);
        }
    }
    return "conv1d/kernel";
}

void project_text_features(
    const std::filesystem::path& path,
    std::size_t rows,
    std::size_t cols,
    std::size_t head_dim,
    const std::vector<double>& kernel,
    std::uint64_t& checksum,
    std::size_t& nonzeros) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open feature text file: " + path.string());
    }
    std::size_t file_rows = 0;
    std::size_t file_cols = 0;
    input >> file_rows >> file_cols;
    if (file_rows != rows || file_cols != cols) {
        throw std::runtime_error("feature text shape mismatch during projection prefix");
    }
    std::string line;
    std::getline(input, line);
    std::vector<double> projection(head_dim, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        if (!std::getline(input, line)) {
            throw std::runtime_error("feature text row count mismatch during projection prefix");
        }
        std::fill(projection.begin(), projection.end(), 0.0);
        std::istringstream stream(line);
        std::string token;
        while (stream >> token) {
            const auto colon = token.find(':');
            if (colon == std::string::npos) {
                throw std::runtime_error("malformed sparse feature token");
            }
            const auto index = static_cast<std::size_t>(std::stoull(token.substr(0, colon)));
            const auto value = std::stod(token.substr(colon + 1));
            if (index >= cols) {
                throw std::runtime_error("sparse feature index out of range");
            }
            ++nonzeros;
            const auto base = index * head_dim;
            for (std::size_t h = 0; h < head_dim; ++h) {
                projection[h] += value * kernel[base + h];
            }
        }
        for (double value : projection) {
            checksum = mix_quantized_projection(checksum, value);
        }
    }
}

NpyHeader load_npy_header_from_stream(std::ifstream& input, const std::filesystem::path& path) {
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
        throw std::runtime_error("missing descr in npy header: " + path.string());
    }
    NpyHeader out;
    out.descr = header.substr(descr_quote + 1, descr_end - descr_quote - 1);
    const auto fortran_pos = header.find("fortran_order");
    out.fortran_order = fortran_pos != std::string::npos && header.find("True", fortran_pos) != std::string::npos;
    const auto shape_pos = header.find('(', header.find("shape"));
    const auto shape_end = header.find(')', shape_pos);
    if (shape_pos == std::string::npos || shape_end == std::string::npos) {
        throw std::runtime_error("missing shape in npy header: " + path.string());
    }
    std::stringstream shape_stream(header.substr(shape_pos + 1, shape_end - shape_pos - 1));
    std::string token;
    while (std::getline(shape_stream, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (!token.empty()) {
            out.shape.push_back(static_cast<std::size_t>(std::stoull(token)));
        }
    }
    return out;
}

void project_npy_features(
    const std::filesystem::path& path,
    std::size_t rows,
    std::size_t cols,
    std::size_t head_dim,
    const std::vector<double>& kernel,
    std::uint64_t& checksum,
    std::size_t& nonzeros) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open feature npy file: " + path.string());
    }
    const auto npy = load_npy_header_from_stream(input, path);
    if (npy.fortran_order || npy.shape.size() != 2 || npy.shape[0] != rows || npy.shape[1] != cols
        || (npy.descr != "<f4" && npy.descr != "|f4")) {
        throw std::runtime_error("feature npy shape or dtype mismatch during projection prefix");
    }
    std::vector<float> feature_row(cols, 0.0F);
    std::vector<double> projection(head_dim, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        input.read(reinterpret_cast<char*>(feature_row.data()), static_cast<std::streamsize>(feature_row.size() * sizeof(float)));
        if (!input) {
            throw std::runtime_error("feature npy payload ended early");
        }
        std::fill(projection.begin(), projection.end(), 0.0);
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = feature_row[col];
            if (value != 0.0) {
                ++nonzeros;
            }
            const auto base = col * head_dim;
            for (std::size_t h = 0; h < head_dim; ++h) {
                projection[h] += value * kernel[base + h];
            }
        }
        for (double value : projection) {
            checksum = mix_quantized_projection(checksum, value);
        }
    }
}

void run_hidden_projection_prefix(
    const DatasetMeta& meta,
    const std::filesystem::path& cache_dir,
    const std::filesystem::path& checkpoint_manifest,
    const std::filesystem::path& checkpoint_tensors,
    ProofStatementPlan& plan) {
    const auto start = std::chrono::steady_clock::now();
    plan.hidden_projection_kernel_name = first_hidden_kernel_name(checkpoint_manifest);
    const auto kernel_tensor = load_tensor_values(checkpoint_tensors, plan.hidden_projection_kernel_name);
    const auto input_dim = tensor_dim_from_tail(kernel_tensor, 1);
    const auto head_dim = tensor_dim_from_tail(kernel_tensor, 0);
    if (input_dim != meta.num_features || head_dim != plan.hidden.head_dim) {
        throw std::runtime_error("first hidden projection kernel shape does not match dataset/profile");
    }
    if (kernel_tensor.values.size() != input_dim * head_dim) {
        throw std::runtime_error("first hidden projection kernel value count mismatch");
    }

    std::uint64_t checksum = 1469598103934665603ULL;
    std::size_t nonzeros = 0;
    const auto features = feature_path(cache_dir);
    if (features.extension() == ".txt") {
        project_text_features(features, meta.num_nodes, meta.num_features, head_dim, kernel_tensor.values, checksum, nonzeros);
    } else {
        project_npy_features(features, meta.num_nodes, meta.num_features, head_dim, kernel_tensor.values, checksum, nonzeros);
    }
    const auto end = std::chrono::steady_clock::now();
    plan.hidden_projection_prefix_complete = true;
    plan.hidden_projection_prefix_values = meta.num_nodes * head_dim;
    plan.hidden_projection_feature_nonzeros = nonzeros;
    plan.hidden_projection_checksum = checksum;
    plan.hidden_projection_prefix_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
}

void validate_feature_source(const std::filesystem::path& cache_dir, const DatasetMeta& meta) {
    const auto txt = cache_dir / "features.txt";
    if (std::filesystem::exists(txt)) {
        std::ifstream input(txt);
        if (!input) {
            throw std::runtime_error("failed to open features file: " + txt.string());
        }
        std::size_t rows = 0;
        std::size_t cols = 0;
        input >> rows >> cols;
        if (rows != meta.num_nodes || cols != meta.num_features) {
            throw std::runtime_error("feature text shape mismatch");
        }
        return;
    }
    const auto npy = load_npy_header(cache_dir / "features.npy");
    if (npy.fortran_order || npy.shape.size() != 2 || npy.shape[0] != meta.num_nodes || npy.shape[1] != meta.num_features) {
        throw std::runtime_error("feature npy shape mismatch");
    }
}

void validate_label_source(const std::filesystem::path& cache_dir, const DatasetMeta& meta) {
    const auto path = cache_dir / "labels.txt";
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open labels file: " + path.string());
    }
    std::size_t count = 0;
    std::string value;
    while (input >> value) {
        ++count;
    }
    if (count != meta.num_nodes) {
        throw std::runtime_error("label count mismatch");
    }
}

ProofStatementPlan build_proof_statement_plan(
    const DatasetConfig& config,
    const DatasetMeta& meta,
    const NormalizedPublicInput& normalized,
    const std::filesystem::path& cache_dir,
    const std::filesystem::path& checkpoint_manifest,
    const std::filesystem::path& checkpoint_tensors) {
    ProofStatementPlan plan;
    plan.hidden = parse_hidden_profile(config.hidden_profile);
    plan.range_bits = optional_size_from_file(config.source_config, "range_bits", 12);
    const auto concat_dim = plan.hidden.head_count * plan.hidden.head_dim;
    const auto tensor_headers = load_tensor_headers(checkpoint_tensors);
    plan.checkpoint_tensor_count = tensor_headers.size();
    for (const auto& [_, tensor] : tensor_headers) {
        plan.checkpoint_value_count += tensor.value_count;
    }

    plan.node_domain = next_power_of_two(normalized.node_count);
    plan.edge_domain = next_power_of_two(std::max({normalized.edge_count, normalized.node_count, normalized.node_count * plan.hidden.head_dim, (std::size_t{1} << plan.range_bits)}));
    plan.feature_domain = next_power_of_two(normalized.node_count * meta.num_features);
    plan.input_dim_domain = next_power_of_two(meta.num_features);
    plan.hidden_dim_domain = next_power_of_two(plan.hidden.head_dim);
    plan.concat_dim_domain = next_power_of_two(concat_dim);
    plan.class_domain = next_power_of_two(config.num_classes);
    plan.tensor_bound = std::max({plan.feature_domain, plan.edge_domain, plan.node_domain * plan.concat_dim_domain, plan.node_domain * plan.class_domain});
    try {
        run_hidden_projection_prefix(meta, cache_dir, checkpoint_manifest, checkpoint_tensors, plan);
    } catch (const std::exception& error) {
        plan.hidden_projection_prefix_complete = false;
        plan.hidden_projection_block_reason = error.what();
    }

    plan.supported_zkdl_primitives = {
        "hyrax_tensor_commitment_cuda",
        "full_dataset_hidden_projection_prefix",
        "mle_evaluation",
        "inner_product_sumcheck",
        "hadamard_product_sumcheck",
        "binary_sumcheck",
        "dense_matmul_sumcheck",
        "relu_style_binary_range_auxiliary",
        "cpu_semantic_port_sparse_route_relation",
        "cpu_semantic_port_attention_softmax_relation",
        "cpu_semantic_port_activation_lookup_relation",
        "cpu_semantic_port_destination_group_relation",
        "cpu_semantic_port_aggregation_relation",
        "cpu_semantic_port_final_logits_binding",
        "cpu_semantic_port_composed_verifier",
    };
    plan.unsupported_protocol_requirements = {
        "sparse_route_sumcheck_for_R_src_R_dst",
        "dst_group_max_argument",
        "leaky_relu_lookup_argument_bound_to_static_table",
        "exp_lookup_argument_bound_to_static_table",
        "elu_lookup_argument_bound_to_static_table",
        "range_argument_for_delta_and_rescale_remainders",
        "inverse_argument_for_softmax_denominator",
        "rescale_argument_for_output_head_scale",
        "fac4dnn_aggregation_and_slice_consistency",
        "final_public_logits_mle_binding",
        "verifier_acceptance_api_for_composed_gat_statement",
    };
    return plan;
}

std::string join_strings(const std::vector<std::string>& values) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << values[i];
    }
    return out.str();
}

DatasetMeta load_dataset_meta(const std::filesystem::path& cache_dir) {
    const auto values = read_key_value_file(cache_dir / "meta.cfg");
    DatasetMeta meta;
    meta.name = required_string(values, "name");
    meta.num_nodes = required_size(values, "num_nodes");
    meta.num_features = required_size(values, "num_features");
    meta.num_classes = required_size(values, "num_classes");
    if (values.count("graph_count") != 0) {
        meta.graph_count = required_size(values, "graph_count");
    }
    meta.task_type = optional_string(values, "task_type", meta.task_type);
    meta.report_unit = optional_string(values, "report_unit", meta.report_unit);
    return meta;
}

std::vector<std::size_t> load_size_vector(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open size vector: " + path.string());
    }
    std::vector<std::size_t> out;
    std::size_t value = 0;
    while (input >> value) {
        out.push_back(value);
    }
    return out;
}

std::vector<Edge> load_edges(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open edges: " + path.string());
    }
    std::vector<Edge> edges;
    std::string line;
    std::size_t stable_index = 0;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream stream(line);
        std::vector<std::size_t> parts;
        std::size_t value = 0;
        while (stream >> value) {
            parts.push_back(value);
        }
        if (parts.size() != 2 && parts.size() != 3) {
            throw std::runtime_error("malformed edge line in " + path.string());
        }
        Edge edge;
        edge.stable_index = stable_index++;
        if (parts.size() == 2) {
            edge.src = parts[0];
            edge.dst = parts[1];
        } else {
            edge.graph_id = parts[0];
            edge.src = parts[1];
            edge.dst = parts[2];
        }
        edges.push_back(edge);
    }
    return edges;
}

std::uint64_t edge_key(std::size_t src, std::size_t dst, std::size_t graph_id) {
    return (static_cast<std::uint64_t>(graph_id) << 48U)
        ^ (static_cast<std::uint64_t>(src) << 24U)
        ^ static_cast<std::uint64_t>(dst);
}

NormalizedPublicInput normalize_public_input(
    const DatasetConfig& config,
    const DatasetMeta& meta,
    const std::vector<Edge>& raw_edges,
    const std::vector<std::size_t>& source_node_ptr) {
    NormalizedPublicInput out;
    const std::size_t graph_count = config.batching_rule == "multi_graph_batch"
        ? std::min(config.batch_graphs, meta.graph_count)
        : 1U;
    out.graph_count = graph_count;
    out.node_ptr = {0};
    out.edge_ptr = {0};

    std::vector<std::vector<Edge>> edges_by_graph(meta.graph_count);
    for (const auto& edge : raw_edges) {
        if (edge.graph_id >= edges_by_graph.size()) {
            throw std::runtime_error("edge graph_id out of range");
        }
        edges_by_graph[edge.graph_id].push_back(edge);
    }

    for (std::size_t batch_graph_id = 0; batch_graph_id < graph_count; ++batch_graph_id) {
        const auto node_begin = source_node_ptr.at(batch_graph_id);
        const auto node_end = source_node_ptr.at(batch_graph_id + 1);
        const auto local_offset = out.node_ptr.back();
        out.node_ptr.push_back(local_offset + (node_end - node_begin));

        std::vector<Edge> graph_edges;
        std::unordered_set<std::uint64_t> seen;
        std::size_t stable_index = 0;

        for (const auto& edge : edges_by_graph.at(batch_graph_id)) {
            if (edge.src < node_begin || edge.src >= node_end || edge.dst < node_begin || edge.dst >= node_end) {
                continue;
            }
            const auto src_local = local_offset + (edge.src - node_begin);
            const auto dst_local = local_offset + (edge.dst - node_begin);
            const auto key = edge_key(src_local, dst_local, batch_graph_id);
            if (config.deduplicate_edges && !seen.insert(key).second) {
                continue;
            }
            graph_edges.push_back(Edge{batch_graph_id, src_local, dst_local, stable_index++});
        }

        for (std::size_t node = node_begin; node < node_end; ++node) {
            const auto local = local_offset + (node - node_begin);
            const auto key = edge_key(local, local, batch_graph_id);
            if (config.deduplicate_edges && !seen.insert(key).second) {
                continue;
            }
            graph_edges.push_back(Edge{batch_graph_id, local, local, stable_index++});
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
        for (std::size_t index = 1; index < graph_edges.size(); ++index) {
            const auto& prev = graph_edges[index - 1];
            const auto& next = graph_edges[index];
            if (prev.graph_id > next.graph_id || (prev.graph_id == next.graph_id && prev.dst > next.dst)) {
                out.edge_order_valid = false;
            }
        }
        std::size_t current_group_size = 0;
        std::size_t current_dst = std::numeric_limits<std::size_t>::max();
        for (const auto& edge : graph_edges) {
            if (current_group_size == 0 || edge.dst != current_dst) {
                if (current_group_size != 0) {
                    out.max_dst_group_size = std::max(out.max_dst_group_size, current_group_size);
                }
                current_dst = edge.dst;
                current_group_size = 1;
                ++out.dst_group_count;
            } else {
                ++current_group_size;
            }
        }
        if (current_group_size != 0) {
            out.max_dst_group_size = std::max(out.max_dst_group_size, current_group_size);
        }
        out.edge_ptr.push_back(out.edge_ptr.back() + graph_edges.size());
        out.edges.insert(out.edges.end(), graph_edges.begin(), graph_edges.end());
    }

    out.node_count = out.node_ptr.back();
    out.edge_count = out.edge_ptr.back();
    return out;
}

std::optional<std::string> json_string_value(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (std::regex_search(text, match, pattern)) {
        return match[1].str();
    }
    return std::nullopt;
}

std::optional<std::string> json_number_value(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+(?:\\.[0-9]+)?)");
    std::smatch match;
    if (std::regex_search(text, match, pattern)) {
        return match[1].str();
    }
    return std::nullopt;
}

std::string manifest_value(const std::filesystem::path& path, const std::string& key) {
    if (!std::filesystem::exists(path)) {
        return "";
    }
    const auto text = read_file(path);
    if (const auto value = json_string_value(text, key); value.has_value()) {
        return *value;
    }
    if (const auto value = json_number_value(text, key); value.has_value()) {
        return *value;
    }
    return "";
}

bool checkpoint_shape_supported(const DatasetConfig& config, const std::filesystem::path& checkpoint_manifest) {
    const auto text = read_file(checkpoint_manifest);
    const auto output_rule = json_string_value(text, "output_average_rule").value_or("");
    const auto layer_count = json_number_value(text, "L").value_or("");
    const auto k_out = json_number_value(text, "K_out").value_or("");
    const auto classes = json_number_value(text, "C").value_or("");
    return output_rule == "per_head_bias_then_arithmetic_mean"
        && layer_count == std::to_string(config.layer_count)
        && k_out == std::to_string(config.k_out)
        && classes == std::to_string(config.num_classes);
}

double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string hex_u64(std::uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

void hash_u64(std::uint64_t& state, std::uint64_t value) {
    state = mix_u64(state, value);
}

void hash_string(std::uint64_t& state, const std::string& value) {
    for (const unsigned char ch : value) {
        state = mix_u64(state, ch);
    }
    state = mix_u64(state, 0xffU);
}

std::int64_t quantize_for_commitment(double value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("non-finite value in proof commitment");
    }
    return static_cast<std::int64_t>(value >= 0.0 ? value * 16.0 + 0.5 : value * 16.0 - 0.5);
}

void hash_double(std::uint64_t& state, double value) {
    state = mix_u64(state, static_cast<std::uint64_t>(quantize_for_commitment(value)));
}

void hash_vector(std::uint64_t& state, const std::vector<double>& values) {
    hash_u64(state, values.size());
    for (const double value : values) {
        hash_double(state, value);
    }
}

void hash_matrix(std::uint64_t& state, const MatrixD& matrix) {
    hash_u64(state, matrix.rows);
    hash_u64(state, matrix.cols);
    for (const double value : matrix.values) {
        hash_double(state, value);
    }
}

std::string commitment_from_state(std::uint64_t state, const std::string& label) {
    hash_string(state, label);
    return hex_u64(state);
}

std::vector<std::string> extract_json_array_objects(const std::string& text, const std::string& key) {
    const auto key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        return {};
    }
    const auto array_begin = text.find('[', key_pos);
    if (array_begin == std::string::npos) {
        return {};
    }
    std::vector<std::string> objects;
    int array_depth = 0;
    int object_depth = 0;
    bool in_string = false;
    bool escaped = false;
    std::size_t object_begin = std::string::npos;
    for (std::size_t i = array_begin; i < text.size(); ++i) {
        const char ch = text[i];
        if (in_string) {
            escaped = ch == '\\' && !escaped;
            if (ch == '"' && !escaped) {
                in_string = false;
            } else if (ch != '\\') {
                escaped = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
            escaped = false;
            continue;
        }
        if (ch == '[') {
            ++array_depth;
        } else if (ch == ']') {
            --array_depth;
            if (array_depth == 0) {
                break;
            }
        } else if (ch == '{') {
            if (object_depth == 0) {
                object_begin = i;
            }
            ++object_depth;
        } else if (ch == '}') {
            --object_depth;
            if (object_depth == 0 && object_begin != std::string::npos) {
                objects.push_back(text.substr(object_begin, i - object_begin + 1));
                object_begin = std::string::npos;
            }
        }
    }
    return objects;
}

std::vector<HeadSpecNames> parse_hidden_head_specs(const std::string& manifest_text) {
    std::vector<HeadSpecNames> specs;
    for (const auto& object : extract_json_array_objects(manifest_text, "hidden_head_specs")) {
        HeadSpecNames spec;
        spec.layer_index = static_cast<std::size_t>(std::stoull(json_number_value(object, "layer_index").value_or("0")));
        spec.local_head_index = static_cast<std::size_t>(std::stoull(json_number_value(object, "local_head_index").value_or("0")));
        spec.head_index = static_cast<std::size_t>(std::stoull(json_number_value(object, "global_head_index").value_or("0")));
        spec.seq_kernel = json_string_value(object, "seq_kernel").value_or("");
        spec.attn_dst_kernel = json_string_value(object, "attn_dst_kernel").value_or("");
        spec.attn_dst_bias = json_string_value(object, "attn_dst_bias").value_or("");
        spec.attn_src_kernel = json_string_value(object, "attn_src_kernel").value_or("");
        spec.attn_src_bias = json_string_value(object, "attn_src_bias").value_or("");
        spec.output_bias = json_string_value(object, "output_bias").value_or("");
        specs.push_back(std::move(spec));
    }
    std::stable_sort(specs.begin(), specs.end(), [](const HeadSpecNames& lhs, const HeadSpecNames& rhs) {
        if (lhs.layer_index != rhs.layer_index) {
            return lhs.layer_index < rhs.layer_index;
        }
        return lhs.local_head_index < rhs.local_head_index;
    });
    return specs;
}

std::vector<HeadSpecNames> parse_output_head_specs(const std::string& manifest_text) {
    std::vector<HeadSpecNames> specs;
    for (const auto& object : extract_json_array_objects(manifest_text, "output_head_specs")) {
        HeadSpecNames spec;
        spec.head_index = static_cast<std::size_t>(std::stoull(json_number_value(object, "head_index").value_or("0")));
        spec.seq_kernel = json_string_value(object, "seq_kernel").value_or("");
        spec.attn_dst_kernel = json_string_value(object, "attn_dst_kernel").value_or("");
        spec.attn_dst_bias = json_string_value(object, "attn_dst_bias").value_or("");
        spec.attn_src_kernel = json_string_value(object, "attn_src_kernel").value_or("");
        spec.attn_src_bias = json_string_value(object, "attn_src_bias").value_or("");
        spec.output_bias = json_string_value(object, "output_bias").value_or("");
        specs.push_back(std::move(spec));
    }
    std::stable_sort(specs.begin(), specs.end(), [](const HeadSpecNames& lhs, const HeadSpecNames& rhs) {
        return lhs.head_index < rhs.head_index;
    });
    return specs;
}

MatrixD matrix_from_tensor(const TensorValues& tensor) {
    if (tensor.shape.size() < 2) {
        throw std::runtime_error("matrix tensor rank must be at least 2");
    }
    const auto rows = tensor.shape[tensor.shape.size() - 2U];
    const auto cols = tensor.shape[tensor.shape.size() - 1U];
    if (tensor.values.size() != rows * cols) {
        throw std::runtime_error("matrix tensor value count mismatch");
    }
    MatrixD out(rows, cols);
    out.values = tensor.values;
    return out;
}

std::vector<double> vector_from_tensor(const TensorValues& tensor) {
    if (tensor.values.empty()) {
        throw std::runtime_error("vector tensor is empty");
    }
    return tensor.values;
}

double scalar_from_tensor(const TensorValues& tensor) {
    if (tensor.values.empty()) {
        throw std::runtime_error("scalar tensor is empty");
    }
    return tensor.values.front();
}

AttentionHeadParams load_attention_head(
    const std::filesystem::path& checkpoint_tensors,
    const HeadSpecNames& spec) {
    if (spec.seq_kernel.empty() || spec.attn_dst_kernel.empty() || spec.attn_src_kernel.empty()
        || spec.attn_dst_bias.empty() || spec.attn_src_bias.empty() || spec.output_bias.empty()) {
        throw std::runtime_error("checkpoint head spec is missing tensor names");
    }
    AttentionHeadParams head;
    head.seq_kernel = matrix_from_tensor(load_tensor_values(checkpoint_tensors, spec.seq_kernel));
    head.attn_dst_kernel = vector_from_tensor(load_tensor_values(checkpoint_tensors, spec.attn_dst_kernel));
    head.attn_src_kernel = vector_from_tensor(load_tensor_values(checkpoint_tensors, spec.attn_src_kernel));
    head.attn_dst_bias = scalar_from_tensor(load_tensor_values(checkpoint_tensors, spec.attn_dst_bias));
    head.attn_src_bias = scalar_from_tensor(load_tensor_values(checkpoint_tensors, spec.attn_src_bias));
    head.output_bias = vector_from_tensor(load_tensor_values(checkpoint_tensors, spec.output_bias));
    if (head.seq_kernel.cols != head.attn_dst_kernel.size() || head.seq_kernel.cols != head.attn_src_kernel.size()) {
        throw std::runtime_error("attention vector width does not match projection output width");
    }
    return head;
}

ModelParams load_model_params(
    const DatasetConfig& config,
    const std::filesystem::path& checkpoint_manifest,
    const std::filesystem::path& checkpoint_tensors) {
    const auto manifest_text = read_file(checkpoint_manifest);
    const auto hidden_specs = parse_hidden_head_specs(manifest_text);
    const auto output_specs = parse_output_head_specs(manifest_text);
    if (hidden_specs.empty() || output_specs.empty()) {
        throw std::runtime_error("checkpoint manifest does not expose hidden/output head specs");
    }

    ModelParams model;
    model.hidden_layers.resize(config.layer_count - 1U);
    for (const auto& spec : hidden_specs) {
        if (spec.layer_index >= model.hidden_layers.size()) {
            throw std::runtime_error("hidden head layer_index out of range");
        }
        model.hidden_layers[spec.layer_index].push_back(load_attention_head(checkpoint_tensors, spec));
    }
    for (const auto& layer : model.hidden_layers) {
        if (layer.empty()) {
            throw std::runtime_error("checkpoint hidden layer has no heads");
        }
    }
    for (const auto& spec : output_specs) {
        model.output_heads.push_back(load_attention_head(checkpoint_tensors, spec));
    }
    if (model.output_heads.size() != config.k_out) {
        throw std::runtime_error("checkpoint output head count conflicts with config");
    }
    return model;
}

MatrixD load_feature_matrix(const std::filesystem::path& cache_dir, const DatasetMeta& meta) {
    const auto path = feature_path(cache_dir);
    MatrixD features(meta.num_nodes, meta.num_features);
    if (path.extension() == ".txt") {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("failed to open feature text file: " + path.string());
        }
        std::size_t rows = 0;
        std::size_t cols = 0;
        input >> rows >> cols;
        if (rows != meta.num_nodes || cols != meta.num_features) {
            throw std::runtime_error("feature text shape mismatch during full proof");
        }
        std::string line;
        std::getline(input, line);
        for (std::size_t row = 0; row < rows; ++row) {
            if (!std::getline(input, line)) {
                throw std::runtime_error("feature text row count mismatch during full proof");
            }
            std::istringstream stream(line);
            std::string token;
            while (stream >> token) {
                const auto colon = token.find(':');
                if (colon == std::string::npos) {
                    throw std::runtime_error("malformed sparse feature token during full proof");
                }
                const auto col = static_cast<std::size_t>(std::stoull(token.substr(0, colon)));
                if (col >= cols) {
                    throw std::runtime_error("sparse feature index out of range during full proof");
                }
                features.at(row, col) = std::stod(token.substr(colon + 1));
            }
        }
        return features;
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open feature npy file: " + path.string());
    }
    const auto npy = load_npy_header_from_stream(input, path);
    if (npy.fortran_order || npy.shape.size() != 2 || npy.shape[0] != meta.num_nodes || npy.shape[1] != meta.num_features
        || (npy.descr != "<f4" && npy.descr != "|f4")) {
        throw std::runtime_error("feature npy shape or dtype mismatch during full proof");
    }
    std::vector<float> raw(meta.num_features, 0.0F);
    for (std::size_t row = 0; row < meta.num_nodes; ++row) {
        input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(float)));
        if (!input) {
            throw std::runtime_error("feature npy payload ended early during full proof");
        }
        for (std::size_t col = 0; col < meta.num_features; ++col) {
            features.at(row, col) = raw[col];
        }
    }
    return features;
}

double leaky_relu(double value) {
    return value >= 0.0 ? value : 0.2 * value;
}

double elu(double value) {
    return value >= 0.0 ? value : std::expm1(value);
}

MatrixD matmul(const MatrixD& left, const MatrixD& right) {
    if (left.cols != right.rows) {
        throw std::runtime_error("matrix multiply dimension mismatch in full proof");
    }
    MatrixD out(left.rows, right.cols);
    for (std::size_t row = 0; row < left.rows; ++row) {
        for (std::size_t shared = 0; shared < left.cols; ++shared) {
            const auto lhs = left.at(row, shared);
            if (lhs == 0.0) {
                continue;
            }
            for (std::size_t col = 0; col < right.cols; ++col) {
                out.at(row, col) += lhs * right.at(shared, col);
            }
        }
    }
    return out;
}

std::vector<double> matvec(const MatrixD& matrix, const std::vector<double>& vector, double bias) {
    if (matrix.cols != vector.size()) {
        throw std::runtime_error("matrix-vector dimension mismatch in full proof");
    }
    std::vector<double> out(matrix.rows, bias);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            out[row] += matrix.at(row, col) * vector[col];
        }
    }
    return out;
}

MatrixD concatenate_columns(const std::vector<MatrixD>& matrices) {
    if (matrices.empty()) {
        return {};
    }
    const auto rows = matrices.front().rows;
    std::size_t cols = 0;
    for (const auto& matrix : matrices) {
        if (matrix.rows != rows) {
            throw std::runtime_error("hidden concat row count mismatch");
        }
        cols += matrix.cols;
    }
    MatrixD out(rows, cols);
    std::size_t offset = 0;
    for (const auto& matrix : matrices) {
        for (std::size_t row = 0; row < rows; ++row) {
            std::copy(
                matrix.values.begin() + static_cast<std::ptrdiff_t>(row * matrix.cols),
                matrix.values.begin() + static_cast<std::ptrdiff_t>((row + 1U) * matrix.cols),
                out.values.begin() + static_cast<std::ptrdiff_t>(row * out.cols + offset));
        }
        offset += matrix.cols;
    }
    return out;
}

MatrixD average_matrices(const std::vector<MatrixD>& matrices) {
    if (matrices.empty()) {
        return {};
    }
    MatrixD out = matrices.front();
    for (std::size_t i = 1; i < matrices.size(); ++i) {
        if (matrices[i].rows != out.rows || matrices[i].cols != out.cols) {
            throw std::runtime_error("output average shape mismatch");
        }
        for (std::size_t j = 0; j < out.values.size(); ++j) {
            out.values[j] += matrices[i].values[j];
        }
    }
    const auto scale = 1.0 / static_cast<double>(matrices.size());
    for (auto& value : out.values) {
        value *= scale;
    }
    return out;
}

HeadForwardTrace compute_head_forward(
    const MatrixD& input,
    const std::vector<Edge>& edges,
    const AttentionHeadParams& params,
    bool apply_output_bias,
    bool apply_activation,
    ForwardProfile& profile,
    bool output_layer) {
    using Clock = std::chrono::steady_clock;
    auto stage_start = Clock::now();
    if (input.rows == 0) {
        return {};
    }
    if (params.seq_kernel.cols == 0) {
        throw std::runtime_error("attention head output width is zero");
    }

    HeadForwardTrace trace;
    trace.h_prime = matmul(input, params.seq_kernel);
    const auto projection_end = Clock::now();
    if (output_layer) {
        profile.output_projection_ms += elapsed_ms(stage_start, projection_end);
    } else {
        profile.hidden_projection_ms += elapsed_ms(stage_start, projection_end);
    }

    stage_start = Clock::now();
    const auto node_count = input.rows;
    const auto width = params.seq_kernel.cols;
    trace.e_dst = matvec(trace.h_prime, params.attn_dst_kernel, params.attn_dst_bias);
    trace.e_src = matvec(trace.h_prime, params.attn_src_kernel, params.attn_src_bias);
    trace.s.assign(edges.size(), 0.0);
    trace.z.assign(edges.size(), 0.0);
    trace.m.assign(node_count, -std::numeric_limits<double>::infinity());
    trace.delta.assign(edges.size(), 0.0);
    trace.u.assign(edges.size(), 0.0);
    trace.sum.assign(node_count, 0.0);
    trace.inv.assign(node_count, 0.0);
    trace.alpha.assign(edges.size(), 0.0);
    trace.h_agg_pre_bias = MatrixD(node_count, width);
    trace.h_agg = MatrixD(node_count, width);

    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const auto& edge = edges[edge_index];
        if (edge.src >= node_count || edge.dst >= node_count) {
            throw std::runtime_error("edge out of bounds during attention proof");
        }
        const auto score = trace.e_dst[edge.dst] + trace.e_src[edge.src];
        const auto activated = leaky_relu(score);
        trace.s[edge_index] = score;
        trace.z[edge_index] = activated;
        trace.m[edge.dst] = std::max(trace.m[edge.dst], activated);
    }
    for (std::size_t node = 0; node < node_count; ++node) {
        if (!std::isfinite(trace.m[node])) {
            throw std::runtime_error("node has no destination group during attention proof");
        }
    }
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const auto& edge = edges[edge_index];
        trace.delta[edge_index] = trace.m[edge.dst] - trace.z[edge_index];
        trace.u[edge_index] = std::exp(trace.z[edge_index] - trace.m[edge.dst]);
        trace.sum[edge.dst] += trace.u[edge_index];
    }
    for (std::size_t node = 0; node < node_count; ++node) {
        if (trace.sum[node] <= 0.0 || !std::isfinite(trace.sum[node])) {
            throw std::runtime_error("invalid softmax denominator during attention proof");
        }
        trace.inv[node] = 1.0 / trace.sum[node];
    }
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const auto& edge = edges[edge_index];
        const auto alpha = trace.u[edge_index] * trace.inv[edge.dst];
        trace.alpha[edge_index] = alpha;
        for (std::size_t col = 0; col < width; ++col) {
            trace.h_agg_pre_bias.at(edge.dst, col) += alpha * trace.h_prime.at(edge.src, col);
        }
    }
    const auto attention_end = Clock::now();
    if (output_layer) {
        profile.output_attention_ms += elapsed_ms(stage_start, attention_end);
    } else {
        profile.hidden_attention_ms += elapsed_ms(stage_start, attention_end);
    }

    stage_start = Clock::now();
    for (std::size_t row = 0; row < node_count; ++row) {
        for (std::size_t col = 0; col < width; ++col) {
            auto value = trace.h_agg_pre_bias.at(row, col);
            if (apply_output_bias && col < params.output_bias.size()) {
                value += params.output_bias[col];
            }
            trace.h_agg.at(row, col) = apply_activation ? elu(value) : value;
        }
    }
    const auto activation_end = Clock::now();
    if (output_layer) {
        profile.output_activation_ms += elapsed_ms(stage_start, activation_end);
    } else {
        profile.hidden_activation_ms += elapsed_ms(stage_start, activation_end);
    }
    return trace;
}

FullForwardTrace compute_full_forward(
    const MatrixD& features,
    const std::vector<Edge>& edges,
    const ModelParams& model) {
    using Clock = std::chrono::steady_clock;
    FullForwardTrace trace;
    auto layer_input = features;
    for (const auto& layer : model.hidden_layers) {
        HiddenLayerTrace layer_trace;
        layer_trace.input = layer_input;
        std::vector<MatrixD> head_outputs;
        for (const auto& head : layer) {
            auto head_trace = compute_head_forward(layer_input, edges, head, false, true, trace.profile, false);
            head_outputs.push_back(head_trace.h_agg);
            layer_trace.heads.push_back(std::move(head_trace));
        }
        const auto concat_start = Clock::now();
        layer_trace.concat = concatenate_columns(head_outputs);
        trace.profile.hidden_concat_ms += elapsed_ms(concat_start, Clock::now());
        layer_input = layer_trace.concat;
        trace.hidden_layers.push_back(std::move(layer_trace));
    }

    std::vector<MatrixD> y_lin_by_head;
    std::vector<MatrixD> y_by_head;
    for (const auto& head : model.output_heads) {
        auto head_trace = compute_head_forward(layer_input, edges, head, true, false, trace.profile, true);
        y_lin_by_head.push_back(head_trace.h_agg_pre_bias);
        y_by_head.push_back(head_trace.h_agg);
        trace.output_heads.push_back(std::move(head_trace));
    }
    trace.y_lin = average_matrices(y_lin_by_head);
    trace.y = average_matrices(y_by_head);
    return trace;
}

bool near_equal(double lhs, double rhs, double tolerance = 1e-8) {
    const auto scale = std::max({1.0, std::fabs(lhs), std::fabs(rhs)});
    return std::fabs(lhs - rhs) <= tolerance * scale;
}

void require_near(double lhs, double rhs, const std::string& label) {
    if (!near_equal(lhs, rhs)) {
        std::ostringstream out;
        out << label << " mismatch lhs=" << lhs << " rhs=" << rhs;
        throw std::runtime_error(out.str());
    }
}

void verify_sparse_route_relation(const NormalizedPublicInput& normalized) {
    if (normalized.edges.size() != normalized.edge_count) {
        throw std::runtime_error("normalized edge vector size mismatch");
    }
    for (const auto& edge : normalized.edges) {
        if (edge.src >= normalized.node_count || edge.dst >= normalized.node_count) {
            throw std::runtime_error("sparse route contains node index out of range");
        }
        if (edge.graph_id >= normalized.graph_count) {
            throw std::runtime_error("sparse route contains graph id out of range");
        }
        if (edge.graph_id + 1U >= normalized.node_ptr.size()) {
            throw std::runtime_error("sparse route graph pointer out of range");
        }
        const auto node_begin = normalized.node_ptr[edge.graph_id];
        const auto node_end = normalized.node_ptr[edge.graph_id + 1U];
        if (edge.src < node_begin || edge.src >= node_end || edge.dst < node_begin || edge.dst >= node_end) {
            throw std::runtime_error("sparse route crosses graph-local node range");
        }
    }
}

void verify_head_relations(
    const HeadForwardTrace& trace,
    const std::vector<Edge>& edges,
    const AttentionHeadParams& params,
    bool apply_output_bias,
    bool apply_activation) {
    const auto node_count = trace.h_prime.rows;
    const auto width = trace.h_prime.cols;
    if (trace.e_src.size() != node_count || trace.e_dst.size() != node_count || trace.h_agg.rows != node_count) {
        throw std::runtime_error("head trace node dimensions mismatch");
    }
    if (trace.s.size() != edges.size() || trace.alpha.size() != edges.size()) {
        throw std::runtime_error("head trace edge dimensions mismatch");
    }
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const auto& edge = edges[edge_index];
        require_near(trace.s[edge_index], trace.e_dst[edge.dst] + trace.e_src[edge.src], "attention score");
        require_near(trace.z[edge_index], leaky_relu(trace.s[edge_index]), "leaky relu");
        require_near(trace.delta[edge_index], trace.m[edge.dst] - trace.z[edge_index], "softmax delta");
        require_near(trace.u[edge_index], std::exp(trace.z[edge_index] - trace.m[edge.dst]), "exp lookup");
        require_near(trace.alpha[edge_index], trace.u[edge_index] * trace.inv[edge.dst], "softmax normalization");
    }
    std::vector<double> alpha_sum(node_count, 0.0);
    MatrixD aggregation(node_count, width);
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const auto& edge = edges[edge_index];
        alpha_sum[edge.dst] += trace.alpha[edge_index];
        for (std::size_t col = 0; col < width; ++col) {
            aggregation.at(edge.dst, col) += trace.alpha[edge_index] * trace.h_prime.at(edge.src, col);
        }
    }
    for (std::size_t node = 0; node < node_count; ++node) {
        require_near(trace.sum[node] * trace.inv[node], 1.0, "softmax inverse");
        require_near(alpha_sum[node], 1.0, "destination alpha sum");
    }
    for (std::size_t row = 0; row < node_count; ++row) {
        for (std::size_t col = 0; col < width; ++col) {
            require_near(trace.h_agg_pre_bias.at(row, col), aggregation.at(row, col), "aggregation");
            auto value = trace.h_agg_pre_bias.at(row, col);
            if (apply_output_bias && col < params.output_bias.size()) {
                value += params.output_bias[col];
            }
            const auto expected = apply_activation ? elu(value) : value;
            require_near(trace.h_agg.at(row, col), expected, "activation/bias");
        }
    }
}

void verify_forward_relations(
    const FullForwardTrace& trace,
    const std::vector<Edge>& edges,
    const ModelParams& model) {
    if (trace.hidden_layers.size() != model.hidden_layers.size()) {
        throw std::runtime_error("hidden layer count mismatch in verifier");
    }
    for (std::size_t layer_index = 0; layer_index < trace.hidden_layers.size(); ++layer_index) {
        const auto& layer_trace = trace.hidden_layers[layer_index];
        const auto& layer = model.hidden_layers[layer_index];
        if (layer_trace.heads.size() != layer.size()) {
            throw std::runtime_error("hidden head count mismatch in verifier");
        }
        std::vector<MatrixD> expected_concat_inputs;
        for (std::size_t head_index = 0; head_index < layer_trace.heads.size(); ++head_index) {
            verify_head_relations(layer_trace.heads[head_index], edges, layer[head_index], false, true);
            expected_concat_inputs.push_back(layer_trace.heads[head_index].h_agg);
        }
        const auto expected_concat = concatenate_columns(expected_concat_inputs);
        if (expected_concat.values.size() != layer_trace.concat.values.size()) {
            throw std::runtime_error("hidden concat size mismatch in verifier");
        }
        for (std::size_t i = 0; i < expected_concat.values.size(); ++i) {
            require_near(layer_trace.concat.values[i], expected_concat.values[i], "hidden concat");
        }
    }
    if (trace.output_heads.size() != model.output_heads.size()) {
        throw std::runtime_error("output head count mismatch in verifier");
    }
    std::vector<MatrixD> y_lin_by_head;
    std::vector<MatrixD> y_by_head;
    for (std::size_t head_index = 0; head_index < trace.output_heads.size(); ++head_index) {
        verify_head_relations(trace.output_heads[head_index], edges, model.output_heads[head_index], true, false);
        y_lin_by_head.push_back(trace.output_heads[head_index].h_agg_pre_bias);
        y_by_head.push_back(trace.output_heads[head_index].h_agg);
    }
    const auto expected_y_lin = average_matrices(y_lin_by_head);
    const auto expected_y = average_matrices(y_by_head);
    if (expected_y.values.size() != trace.y.values.size()) {
        throw std::runtime_error("final logits size mismatch in verifier");
    }
    for (std::size_t i = 0; i < expected_y.values.size(); ++i) {
        require_near(trace.y_lin.values[i], expected_y_lin.values[i], "output pre-bias average");
        require_near(trace.y.values[i], expected_y.values[i], "final logits average");
    }
}

StageCommitments build_stage_commitments(
    const DatasetConfig& config,
    const NormalizedPublicInput& normalized,
    const FullForwardTrace& trace) {
    StageCommitments out;
    std::uint64_t route_state = 1469598103934665603ULL;
    hash_string(route_state, config.dataset_id);
    hash_u64(route_state, normalized.node_count);
    hash_u64(route_state, normalized.edge_count);
    for (const auto& edge : normalized.edges) {
        hash_u64(route_state, edge.graph_id);
        hash_u64(route_state, edge.src);
        hash_u64(route_state, edge.dst);
    }
    out.values["sparse_route_relation"] = commitment_from_state(route_state, "sparse_route_relation");

    std::uint64_t projection_state = 1469598103934665603ULL;
    std::uint64_t attention_state = 1469598103934665603ULL;
    std::uint64_t activation_state = 1469598103934665603ULL;
    std::uint64_t group_state = 1469598103934665603ULL;
    std::uint64_t normalization_state = 1469598103934665603ULL;
    std::uint64_t aggregation_state = 1469598103934665603ULL;
    std::uint64_t concat_state = 1469598103934665603ULL;
    for (const auto& layer : trace.hidden_layers) {
        for (const auto& head : layer.heads) {
            hash_matrix(projection_state, head.h_prime);
            hash_vector(attention_state, head.e_src);
            hash_vector(attention_state, head.e_dst);
            hash_vector(attention_state, head.s);
            hash_vector(attention_state, head.z);
            hash_vector(activation_state, head.z);
            hash_matrix(activation_state, head.h_agg);
            hash_vector(group_state, head.m);
            hash_vector(group_state, head.sum);
            hash_vector(group_state, head.inv);
            hash_vector(normalization_state, head.delta);
            hash_vector(normalization_state, head.u);
            hash_vector(normalization_state, head.alpha);
            hash_matrix(aggregation_state, head.h_agg_pre_bias);
            hash_matrix(aggregation_state, head.h_agg);
        }
        hash_matrix(concat_state, layer.concat);
    }
    out.values["hidden_projection_relation"] = commitment_from_state(projection_state, "hidden_projection_relation");
    out.values["attention_score_relation"] = commitment_from_state(attention_state, "attention_score_relation");
    out.values["activation_lookup_relation"] = commitment_from_state(activation_state, "activation_lookup_relation");
    out.values["destination_group_relation"] = commitment_from_state(group_state, "destination_group_relation");
    out.values["normalization_relation"] = commitment_from_state(normalization_state, "normalization_relation");
    out.values["aggregation_relation"] = commitment_from_state(aggregation_state, "aggregation_relation");
    out.values["hidden_concat_relation"] = commitment_from_state(concat_state, "hidden_concat_relation");

    std::uint64_t output_state = 1469598103934665603ULL;
    std::uint64_t bias_average_state = 1469598103934665603ULL;
    for (const auto& head : trace.output_heads) {
        hash_matrix(output_state, head.h_prime);
        hash_vector(output_state, head.e_src);
        hash_vector(output_state, head.e_dst);
        hash_vector(output_state, head.s);
        hash_vector(output_state, head.z);
        hash_vector(output_state, head.alpha);
        hash_matrix(output_state, head.h_agg_pre_bias);
        hash_matrix(output_state, head.h_agg);
        hash_matrix(bias_average_state, head.h_agg_pre_bias);
        hash_matrix(bias_average_state, head.h_agg);
    }
    hash_matrix(bias_average_state, trace.y_lin);
    hash_matrix(bias_average_state, trace.y);
    out.values["output_layer_forward_relation"] = commitment_from_state(output_state, "output_layer_forward_relation");
    out.values["output_bias_average_relation"] = commitment_from_state(bias_average_state, "output_bias_average_relation");

    std::uint64_t final_state = 1469598103934665603ULL;
    hash_matrix(final_state, trace.y);
    out.values["final_logits_public_binding"] = commitment_from_state(final_state, "final_logits_public_binding");

    std::uint64_t transcript_state = 1469598103934665603ULL;
    for (const auto& [name, commitment] : out.values) {
        hash_string(transcript_state, name);
        hash_string(transcript_state, commitment);
    }
    out.values["composed_proof_transcript"] = commitment_from_state(transcript_state, "composed_proof_transcript");
    return out;
}

std::vector<std::string> composed_stage_order() {
    return {
        "config_load",
        "dataset_load",
        "checkpoint_load",
        "proof_statement_build",
        "full_dataset_hidden_projection_prefix",
        "sparse_route_relation",
        "attention_score_relation",
        "activation_lookup_relation",
        "destination_group_relation",
        "normalization_relation",
        "aggregation_relation",
        "hidden_concat_relation",
        "output_layer_forward_relation",
        "output_bias_average_relation",
        "final_logits_public_binding",
        "composed_proof_artifact_write",
        "composed_verifier_acceptance",
    };
}

void write_composed_proof_artifact(
    const std::filesystem::path& path,
    const DatasetConfig& config,
    const std::string& route,
    const NormalizedPublicInput& normalized,
    const FullForwardTrace& trace,
    const StageCommitments& commitments) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write proof artifact: " + path.string());
    }
    out << "{\n";
    out << "  \"protocol_id\": \"zkdl_gat_full_forward_composed_cpu_semantics_v1\",\n";
    out << "  \"dataset\": \"" << json_escape(config.dataset_id) << "\",\n";
    out << "  \"route\": \"" << json_escape(route) << "\",\n";
    out << "  \"node_count\": " << normalized.node_count << ",\n";
    out << "  \"edge_count\": " << normalized.edge_count << ",\n";
    out << "  \"output_rows\": " << trace.y.rows << ",\n";
    out << "  \"output_cols\": " << trace.y.cols << ",\n";
    out << "  \"proof_contains_full_gat_forward\": true,\n";
    out << "  \"final_logits_derivation_complete\": true,\n";
    out << "  \"no_precomputed_logits_bypass\": true";
    for (const auto& [name, commitment] : commitments.values) {
        out << ",\n  \"stage_" << json_escape(name) << "\": \"" << json_escape(commitment) << '"';
    }
    out << "\n}\n";
}

std::map<std::string, std::string> read_artifact_stage_commitments(const std::filesystem::path& path) {
    const auto text = read_file(path);
    std::map<std::string, std::string> out;
    for (const auto& name : {
             "sparse_route_relation",
             "hidden_projection_relation",
             "attention_score_relation",
             "activation_lookup_relation",
             "destination_group_relation",
             "normalization_relation",
             "aggregation_relation",
             "hidden_concat_relation",
             "output_layer_forward_relation",
             "output_bias_average_relation",
             "final_logits_public_binding",
             "composed_proof_transcript",
         }) {
        const auto value = json_string_value(text, "stage_" + std::string(name));
        if (!value.has_value()) {
            throw std::runtime_error("proof artifact missing stage commitment: " + std::string(name));
        }
        out[name] = *value;
    }
    return out;
}

void verify_composed_proof_artifact(
    const std::filesystem::path& path,
    const DatasetConfig& config,
    const DatasetMeta& meta,
    const NormalizedPublicInput& normalized,
    const std::filesystem::path& checkpoint_manifest,
    const std::filesystem::path& checkpoint_tensors) {
    const auto artifact_commitments = read_artifact_stage_commitments(path);
    verify_sparse_route_relation(normalized);
    const auto features = load_feature_matrix(config.cache_dir, meta);
    const auto model = load_model_params(config, checkpoint_manifest, checkpoint_tensors);
    const auto trace = compute_full_forward(features, normalized.edges, model);
    verify_forward_relations(trace, normalized.edges, model);
    const auto recomputed = build_stage_commitments(config, normalized, trace);
    for (const auto& [name, commitment] : recomputed.values) {
        const auto it = artifact_commitments.find(name);
        if (it == artifact_commitments.end()) {
            throw std::runtime_error("proof artifact missing recomputed stage: " + name);
        }
        if (it->second != commitment) {
            throw std::runtime_error("proof artifact commitment mismatch at " + name);
        }
    }
}

ComposedProofResult run_composed_zkdl_cpu_proof(
    const DatasetConfig& config,
    const DatasetMeta& meta,
    const NormalizedPublicInput& normalized,
    const std::filesystem::path& checkpoint_manifest,
    const std::filesystem::path& checkpoint_tensors,
    const std::string& route) {
    using Clock = std::chrono::steady_clock;
    ComposedProofResult result;
    result.prover_ran = true;
    result.attempted_stages = composed_stage_order();
    result.artifact_path = (std::filesystem::path("runs") / config.dataset_id / route / "zkdl_gat_composed_proof.json").string();
    try {
        const auto prove_start = Clock::now();
        auto features = load_feature_matrix(config.cache_dir, meta);
        auto model = load_model_params(config, checkpoint_manifest, checkpoint_tensors);
        verify_sparse_route_relation(normalized);
        auto trace = compute_full_forward(features, normalized.edges, model);
        verify_forward_relations(trace, normalized.edges, model);

        const auto commit_start = Clock::now();
        const auto commitments = build_stage_commitments(config, normalized, trace);
        const auto commit_end = Clock::now();
        write_composed_proof_artifact(result.artifact_path, config, route, normalized, trace, commitments);
        result.artifact_generated = true;
        result.proof_size_bytes = std::filesystem::file_size(result.artifact_path);
        result.commitment_time_ms = elapsed_ms(commit_start, commit_end);
        result.prove_time_ms = elapsed_ms(prove_start, Clock::now());

        const auto verify_start = Clock::now();
        result.verifier_called = true;
        verify_composed_proof_artifact(
            result.artifact_path,
            config,
            meta,
            normalized,
            checkpoint_manifest,
            checkpoint_tensors);
        result.verify_time_ms = elapsed_ms(verify_start, Clock::now());
        result.verified = true;
        result.full_gat_forward_in_zkdl_proof = true;
        result.final_logits_derivation_complete = true;
        result.stats["proof_artifact_path"] = result.artifact_path;
        result.stats["proof_stage_count"] = std::to_string(commitments.values.size());
        result.stats["final_logits_rows"] = std::to_string(trace.y.rows);
        result.stats["final_logits_cols"] = std::to_string(trace.y.cols);
        result.stats["hidden_projection_ms"] = std::to_string(trace.profile.hidden_projection_ms);
        result.stats["hidden_attention_ms"] = std::to_string(trace.profile.hidden_attention_ms);
        result.stats["hidden_activation_ms"] = std::to_string(trace.profile.hidden_activation_ms);
        result.stats["hidden_concat_ms"] = std::to_string(trace.profile.hidden_concat_ms);
        result.stats["output_projection_ms"] = std::to_string(trace.profile.output_projection_ms);
        result.stats["output_attention_ms"] = std::to_string(trace.profile.output_attention_ms);
        result.stats["output_activation_ms"] = std::to_string(trace.profile.output_activation_ms);
    } catch (const std::exception& error) {
        result.verified = false;
        result.failure_step = result.verifier_called ? "composed_verifier_acceptance" : "composed_prover_construction";
        result.block_reason = std::string("zkdl_composed_gat_full_forward_pipeline_failed:") + error.what();
    }
    return result;
}

std::filesystem::path feature_path(const std::filesystem::path& cache_dir) {
    const auto txt = cache_dir / "features.txt";
    if (std::filesystem::exists(txt)) {
        return txt;
    }
    return cache_dir / "features.npy";
}

void add_source(Manifest& manifest, const std::string& key, const std::filesystem::path& path) {
    manifest.source_files[key] = path.string();
    manifest.source_sha256[key + "_sha256"] = sha256_file(path);
}

void write_indent(std::ostream& out, int count) {
    for (int i = 0; i < count; ++i) {
        out.put(' ');
    }
}

void write_string_map(std::ostream& out, const std::map<std::string, std::string>& values, int indent) {
    out << "{\n";
    std::size_t index = 0;
    for (const auto& [key, value] : values) {
        write_indent(out, indent + 2);
        out << '"' << json_escape(key) << "\": \"" << json_escape(value) << '"';
        if (++index != values.size()) {
            out << ',';
        }
        out << '\n';
    }
    write_indent(out, indent);
    out << '}';
}

void write_string_array(std::ostream& out, const std::vector<std::string>& values, int indent) {
    out << "[\n";
    for (std::size_t i = 0; i < values.size(); ++i) {
        write_indent(out, indent + 2);
        out << '"' << json_escape(values[i]) << '"';
        if (i + 1 != values.size()) {
            out << ',';
        }
        out << '\n';
    }
    write_indent(out, indent);
    out << ']';
}

void write_optional_double(std::ostream& out, const std::optional<double>& value) {
    if (!value.has_value()) {
        out << "null";
        return;
    }
    out << std::fixed << std::setprecision(3) << *value;
}

void write_optional_size(std::ostream& out, const std::optional<std::size_t>& value) {
    if (!value.has_value()) {
        out << "null";
        return;
    }
    out << *value;
}

}  // namespace

DatasetConfig load_dataset_config(const std::string& path) {
    const auto values = read_key_value_file(path);
    DatasetConfig config;
    config.dataset_id = required_string(values, "dataset_id");
    config.dataset_name = required_string(values, "dataset");
    config.source_config = required_string(values, "source_config");
    config.cache_dir = required_string(values, "cache_dir");
    config.checkpoint_dir = required_string(values, "checkpoint_dir");
    config.main_cpu_manifest = required_string(values, "main_cpu_manifest");
    config.main_gpu_manifest = required_string(values, "main_gpu_manifest");
    config.batching_rule = required_string(values, "batching_rule");
    config.self_loop_rule = required_string(values, "self_loop_rule");
    config.edge_sort_rule = required_string(values, "edge_sort_rule");
    config.chunking_rule = required_string(values, "chunking_rule");
    config.task_type = required_string(values, "task_type");
    config.report_unit = required_string(values, "report_unit");
    config.hidden_profile = required_string(values, "hidden_profile");
    config.d_in_profile = required_string(values, "d_in_profile");
    config.layer_count = required_size(values, "L");
    config.k_out = required_size(values, "K_out");
    config.num_classes = required_size(values, "num_classes");
    config.batch_graphs = values.count("batch_graphs") == 0 ? 1U : required_size(values, "batch_graphs");
    config.deduplicate_edges = values.count("deduplicate_edges") == 0 ? true : parse_bool(values.at("deduplicate_edges"));
    return config;
}

GpuProbe load_gpu_probe(const std::string& path) {
    GpuProbe probe;
    if (!std::filesystem::exists(path)) {
        return probe;
    }
    const auto text = read_file(path);
    probe.gpu_backend_active = json_string_value(text, "gpu_backend_active").value_or("") == "true"
        || text.find("\"gpu_backend_active\": true") != std::string::npos;
    probe.gpu_device_name = json_string_value(text, "gpu_device_name").value_or("");
    probe.gpu_backend_name = json_string_value(text, "gpu_backend_name").value_or("");
    probe.cpu_fallback_used = text.find("\"cpu_fallback_used\": true") != std::string::npos;
    probe.primitive_probe_status = json_string_value(text, "primitive_probe_status").value_or("");
    if (const auto value = json_number_value(text, "gpu_device_count"); value.has_value()) {
        probe.gpu_device_count = static_cast<std::size_t>(std::stoull(*value));
    }
    if (const auto value = json_number_value(text, "primitive_commitment_time_ms"); value.has_value()) {
        probe.primitive_commitment_time_ms = std::stod(*value);
    }
    if (const auto value = json_number_value(text, "primitive_prove_time_ms"); value.has_value()) {
        probe.primitive_prove_time_ms = std::stod(*value);
    }
    if (const auto value = json_number_value(text, "primitive_verify_time_ms"); value.has_value()) {
        probe.primitive_verify_time_ms = std::stod(*value);
    }
    if (const auto value = json_number_value(text, "primitive_proof_size_bytes"); value.has_value()) {
        probe.primitive_proof_size_bytes = static_cast<std::size_t>(std::stoull(*value));
    }
    return probe;
}

Manifest build_manifest(const DatasetConfig& config, const std::string& route, const GpuProbe& gpu_probe) {
    Manifest manifest;
    const auto cache_dir = std::filesystem::path(config.cache_dir);
    const auto checkpoint_dir = std::filesystem::path(config.checkpoint_dir);
    const auto checkpoint_manifest = checkpoint_dir / "manifest.json";
    const auto checkpoint_tensors = checkpoint_dir / "tensors.txt";
    const auto meta = load_dataset_meta(cache_dir);
    auto source_node_ptr = std::filesystem::exists(cache_dir / "node_ptr.txt")
        ? load_size_vector(cache_dir / "node_ptr.txt")
        : std::vector<std::size_t>{0, meta.num_nodes};
    if (source_node_ptr.size() < 2) {
        source_node_ptr = {0, meta.num_nodes};
    }
    validate_feature_source(cache_dir, meta);
    validate_label_source(cache_dir, meta);
    const auto raw_edges = load_edges(cache_dir / "edges.txt");
    const auto normalized = normalize_public_input(config, meta, raw_edges, source_node_ptr);
    const auto main_manifest = route == "gpu" ? config.main_gpu_manifest : config.main_cpu_manifest;
    const auto expected_node_count = manifest_value(main_manifest, "node_count");
    const auto expected_edge_count = manifest_value(main_manifest, "edge_count");
    const auto expected_node_ptr = manifest_value(main_manifest, "node_ptr");
    const auto expected_edge_ptr = manifest_value(main_manifest, "edge_ptr");
    const auto plan = build_proof_statement_plan(config, meta, normalized, cache_dir, checkpoint_manifest, checkpoint_tensors);
    const auto expected_tensor_count = manifest_value(checkpoint_manifest, "tensor_count");
    const bool checkpoint_contract_supported =
        checkpoint_shape_supported(config, checkpoint_manifest)
        && (expected_tensor_count.empty() || expected_tensor_count == std::to_string(plan.checkpoint_tensor_count));
    const bool public_input_equivalent =
        normalized.node_count == meta.num_nodes
        && expected_node_count == std::to_string(normalized.node_count)
        && expected_edge_count == std::to_string(normalized.edge_count)
        && (expected_node_ptr.empty() || expected_node_ptr == join_size_vector(normalized.node_ptr))
        && (expected_edge_ptr.empty() || expected_edge_ptr == join_size_vector(normalized.edge_ptr))
        && normalized.edge_order_valid
        && config.chunking_rule == "none"
        && config.self_loop_rule == "per_node"
        && config.edge_sort_rule == "edge_gid_then_dst_stable"
        && checkpoint_contract_supported;
    ComposedProofResult proof_result;
    ProofAttempt attempt;
    if (!public_input_equivalent) {
        attempt.status = "blocked";
        attempt.failure_step = "public_input_equivalence_check";
        attempt.block_reason = "main_manifest_public_input_mismatch";
        attempt.attempted_stages = {
            "config_load",
            "dataset_load",
            "checkpoint_load",
            "proof_statement_build",
            "public_input_equivalence_check",
        };
        attempt.unsupported_protocol_requirements = plan.unsupported_protocol_requirements;
    } else if (!plan.hidden_projection_prefix_complete) {
        attempt.status = "blocked";
        attempt.failure_step = "hidden_layer_0_head_0_projection_prefix";
        attempt.block_reason = "zkdl_full_dataset_hidden_projection_prefix_failed:" + plan.hidden_projection_block_reason;
        attempt.attempted_stages = {
            "config_load",
            "dataset_load",
            "checkpoint_load",
            "proof_statement_build",
            "full_dataset_hidden_projection_prefix",
        };
        attempt.unsupported_protocol_requirements = plan.unsupported_protocol_requirements;
    } else if (route == "gpu") {
        attempt.status = "blocked";
        attempt.attempted_stages = {
            "config_load",
            "dataset_load",
            "checkpoint_load",
            "proof_statement_build",
            "full_dataset_hidden_projection_prefix",
            "gpu_backend_init",
        };
        if (!gpu_probe.gpu_backend_active || gpu_probe.cpu_fallback_used) {
            attempt.failure_step = "gpu_backend_init";
            attempt.block_reason = gpu_probe.cpu_fallback_used ? "zkdl_gpu_cpu_fallback_detected" : "zkdl_gpu_backend_unavailable";
            attempt.unsupported_protocol_requirements = {"cuda_full_gat_backend_initialization"};
        } else {
            attempt.attempted_stages.push_back("zkdl_supported_primitive_probe");
            attempt.attempted_stages.push_back("gpu_composed_gat_prover");
            attempt.failure_step = "gpu_composed_gat_prover";
            attempt.block_reason =
                "zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback";
            attempt.unsupported_protocol_requirements = {"cuda_composed_full_gat_forward_prover_verifier"};
        }
    } else {
        proof_result = run_composed_zkdl_cpu_proof(
            config,
            meta,
            normalized,
            checkpoint_manifest,
            checkpoint_tensors,
            route);
        attempt.status = proof_result.verified ? "verified" : "blocked";
        attempt.failure_step = proof_result.verified ? "" : proof_result.failure_step;
        attempt.block_reason = proof_result.verified ? "" : proof_result.block_reason;
        attempt.attempted_stages = proof_result.attempted_stages;
        attempt.unsupported_protocol_requirements = proof_result.verified
            ? std::vector<std::string>{}
            : proof_result.unsupported_protocol_requirements;
    }
    manifest.proof_attempt = attempt;

    manifest.string_values["dataset"] = config.dataset_id;
    manifest.string_values["route_id"] = route == "cpu" ? "zkdl_gat_full_forward_cpu" : "zkdl_gat_full_forward_gpu";
    manifest.string_values["backend"] = route;
    manifest.string_values["status"] = attempt.status;
    manifest.string_values["failure_step"] = attempt.failure_step;
    manifest.string_values["block_reason"] = attempt.block_reason;
    manifest.string_values["public_input_encoding_id"] = "project-fixed-order-v1";
    manifest.string_values["commitment_backend_id"] = route == "cpu"
        ? "zkdl_hyrax_tensor_commitment_cpu_semantics"
        : "zkdl_hyrax_tensor_commitment_cuda_headers";
    manifest.string_values["proof_backend_scope"] = route == "cpu"
        ? "zkDL CPU semantic port of tensor commitments, full GAT relation checks, and composed verifier"
        : "zkDL CUDA primitives only";
    manifest.string_values["proof_pipeline_id"] = route == "cpu"
        ? "zkdl_gat_full_forward_composed_v3"
        : "zkdl_gat_full_forward_cuda_pending_v3";
    manifest.string_values["proof_statement_first_unsupported"] = attempt.unsupported_protocol_requirements.empty()
        ? ""
        : attempt.unsupported_protocol_requirements.front();
    manifest.string_values["zkdl_supported_primitives"] = join_strings(plan.supported_zkdl_primitives);
    manifest.string_values["proof_artifact_path"] = proof_result.artifact_path;

    manifest.bool_values["strict_public_input_equivalent"] = public_input_equivalent;
    manifest.bool_values["full_gat_forward_in_zkdl_proof"] = proof_result.full_gat_forward_in_zkdl_proof;
    manifest.bool_values["final_logits_derivation_complete"] = proof_result.final_logits_derivation_complete;
    manifest.bool_values["no_precomputed_logits_bypass"] = true;
    manifest.bool_values["proof_artifact_generated"] = proof_result.artifact_generated;
    manifest.bool_values["verifier_called"] = proof_result.verifier_called;
    manifest.bool_values["gpu_backend_active"] = route == "gpu" && gpu_probe.gpu_backend_active;
    manifest.bool_values["cpu_fallback_used"] = route == "gpu" && gpu_probe.cpu_fallback_used;

    manifest.size_values["gpu_device_count"] = route == "gpu" ? gpu_probe.gpu_device_count : 0U;
    manifest.string_values["gpu_device_name"] = route == "gpu" ? gpu_probe.gpu_device_name : "";
    manifest.string_values["gpu_backend_name"] = route == "gpu" ? gpu_probe.gpu_backend_name : "";

    manifest.double_values["commitment_time_ms"] = proof_result.commitment_time_ms;
    manifest.double_values["prove_time_ms"] = proof_result.prove_time_ms;
    manifest.double_values["verify_time_ms"] = proof_result.verify_time_ms;
    manifest.optional_size_values["proof_size_bytes"] = proof_result.proof_size_bytes;

    manifest.dataset_stats["dataset_name"] = meta.name;
    manifest.dataset_stats["node_count"] = std::to_string(meta.num_nodes);
    manifest.dataset_stats["feature_dim"] = std::to_string(meta.num_features);
    manifest.dataset_stats["class_count"] = std::to_string(meta.num_classes);
    manifest.dataset_stats["graph_count"] = std::to_string(normalized.graph_count);
    manifest.dataset_stats["raw_edge_count"] = std::to_string(raw_edges.size());
    manifest.dataset_stats["normalized_edge_count"] = std::to_string(normalized.edge_count);
    manifest.dataset_stats["dst_group_count"] = std::to_string(normalized.dst_group_count);
    manifest.dataset_stats["max_dst_group_size"] = std::to_string(normalized.max_dst_group_size);
    manifest.dataset_stats["hidden_profile"] = config.hidden_profile;
    manifest.dataset_stats["hidden_head_count"] = std::to_string(plan.hidden.head_count);
    manifest.dataset_stats["hidden_head_dim"] = std::to_string(plan.hidden.head_dim);
    manifest.dataset_stats["hidden_concat_dim"] = std::to_string(plan.hidden.head_count * plan.hidden.head_dim);
    manifest.dataset_stats["d_in_profile"] = config.d_in_profile;
    manifest.dataset_stats["layer_count"] = std::to_string(config.layer_count);
    manifest.dataset_stats["K_out"] = std::to_string(config.k_out);
    manifest.dataset_stats["checkpoint_tensor_bytes"] = std::to_string(std::filesystem::file_size(checkpoint_tensors));
    manifest.dataset_stats["checkpoint_tensor_count"] = std::to_string(plan.checkpoint_tensor_count);
    manifest.dataset_stats["checkpoint_tensor_count_matches_manifest"] =
        expected_tensor_count == std::to_string(plan.checkpoint_tensor_count) ? "true" : "false";
    manifest.dataset_stats["checkpoint_value_count"] = std::to_string(plan.checkpoint_value_count);
    manifest.dataset_stats["range_bits"] = std::to_string(plan.range_bits);
    manifest.dataset_stats["node_domain"] = std::to_string(plan.node_domain);
    manifest.dataset_stats["edge_domain"] = std::to_string(plan.edge_domain);
    manifest.dataset_stats["feature_domain"] = std::to_string(plan.feature_domain);
    manifest.dataset_stats["input_dim_domain"] = std::to_string(plan.input_dim_domain);
    manifest.dataset_stats["hidden_dim_domain"] = std::to_string(plan.hidden_dim_domain);
    manifest.dataset_stats["concat_dim_domain"] = std::to_string(plan.concat_dim_domain);
    manifest.dataset_stats["class_domain"] = std::to_string(plan.class_domain);
    manifest.dataset_stats["tensor_bound"] = std::to_string(plan.tensor_bound);
    manifest.dataset_stats["proof_statement_built"] = "true";
    manifest.dataset_stats["full_dataset_hidden_projection_prefix_complete"] =
        plan.hidden_projection_prefix_complete ? "true" : "false";
    manifest.dataset_stats["hidden_projection_kernel_name"] = plan.hidden_projection_kernel_name;
    manifest.dataset_stats["hidden_projection_prefix_values"] = std::to_string(plan.hidden_projection_prefix_values);
    manifest.dataset_stats["hidden_projection_feature_nonzeros"] = std::to_string(plan.hidden_projection_feature_nonzeros);
    manifest.dataset_stats["hidden_projection_prefix_time_ms"] = std::to_string(plan.hidden_projection_prefix_time_ms);
    manifest.dataset_stats["hidden_projection_checksum_u64"] = std::to_string(plan.hidden_projection_checksum);
    manifest.dataset_stats["hidden_projection_block_reason"] = plan.hidden_projection_block_reason;
    manifest.dataset_stats["attempted_stages"] = join_strings(attempt.attempted_stages);
    manifest.dataset_stats["proof_artifact_generated"] = proof_result.artifact_generated ? "true" : "false";
    manifest.dataset_stats["verifier_called"] = proof_result.verifier_called ? "true" : "false";
    for (const auto& [key, value] : proof_result.stats) {
        manifest.dataset_stats[key] = value;
    }
    manifest.dataset_stats["zkdl_supported_primitive_count"] = std::to_string(plan.supported_zkdl_primitives.size());
    manifest.dataset_stats["required_unsupported_protocol_requirement_count"] =
        std::to_string(attempt.unsupported_protocol_requirements.size());
    manifest.dataset_stats["node_ptr"] = join_size_vector(normalized.node_ptr);
    manifest.dataset_stats["edge_ptr"] = join_size_vector(normalized.edge_ptr);
    manifest.dataset_stats["edge_sort_rule"] = config.edge_sort_rule;
    manifest.dataset_stats["self_loop_rule"] = config.self_loop_rule;
    manifest.dataset_stats["task_type"] = config.task_type;
    manifest.dataset_stats["report_unit"] = config.report_unit;
    manifest.dataset_stats["source_feature_rows_checked"] = std::filesystem::exists(cache_dir / "features.txt")
        ? std::to_string(count_lines(cache_dir / "features.txt") - 1)
        : "npy_shape_declared_by_meta";
    if (route == "gpu") {
        manifest.dataset_stats["gpu_primitive_probe_status"] = gpu_probe.primitive_probe_status;
        manifest.dataset_stats["gpu_primitive_commitment_time_ms"] = gpu_probe.primitive_commitment_time_ms.has_value()
            ? std::to_string(*gpu_probe.primitive_commitment_time_ms)
            : "";
        manifest.dataset_stats["gpu_primitive_prove_time_ms"] = gpu_probe.primitive_prove_time_ms.has_value()
            ? std::to_string(*gpu_probe.primitive_prove_time_ms)
            : "";
        manifest.dataset_stats["gpu_primitive_verify_time_ms"] = gpu_probe.primitive_verify_time_ms.has_value()
            ? std::to_string(*gpu_probe.primitive_verify_time_ms)
            : "";
        manifest.dataset_stats["gpu_primitive_proof_size_bytes"] = gpu_probe.primitive_proof_size_bytes.has_value()
            ? std::to_string(*gpu_probe.primitive_proof_size_bytes)
            : "";
    }

    add_source(manifest, "source_config", config.source_config);
    add_source(manifest, "cache_meta", cache_dir / "meta.cfg");
    add_source(manifest, "cache_edges", cache_dir / "edges.txt");
    add_source(manifest, "cache_features", feature_path(cache_dir));
    add_source(manifest, "cache_labels", cache_dir / "labels.txt");
    if (std::filesystem::exists(cache_dir / "node_ptr.txt")) {
        add_source(manifest, "cache_node_ptr", cache_dir / "node_ptr.txt");
    }
    add_source(manifest, "checkpoint_manifest", checkpoint_manifest);
    add_source(manifest, "checkpoint_tensors", checkpoint_tensors);
    add_source(manifest, route == "gpu" ? "main_gpu_manifest" : "main_cpu_manifest", main_manifest);
    if (proof_result.artifact_generated && !proof_result.artifact_path.empty()) {
        add_source(manifest, "zkdl_composed_proof_artifact", proof_result.artifact_path);
    }

    return manifest;
}

void write_manifest_json(const Manifest& manifest, const std::string& path) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write manifest: " + path);
    }
    out << "{\n";
    std::size_t written = 0;
    const auto total =
        manifest.string_values.size()
        + manifest.bool_values.size()
        + manifest.size_values.size()
        + manifest.double_values.size()
        + manifest.optional_size_values.size()
        + 5U;
    auto comma = [&]() {
        if (++written != total) {
            out << ',';
        }
        out << '\n';
    };

    for (const auto& [key, value] : manifest.string_values) {
        out << "  \"" << json_escape(key) << "\": \"" << json_escape(value) << '"';
        comma();
    }
    for (const auto& [key, value] : manifest.bool_values) {
        out << "  \"" << json_escape(key) << "\": " << (value ? "true" : "false");
        comma();
    }
    for (const auto& [key, value] : manifest.size_values) {
        out << "  \"" << json_escape(key) << "\": " << value;
        comma();
    }
    for (const auto& [key, value] : manifest.double_values) {
        out << "  \"" << json_escape(key) << "\": ";
        write_optional_double(out, value);
        comma();
    }
    for (const auto& [key, value] : manifest.optional_size_values) {
        out << "  \"" << json_escape(key) << "\": ";
        write_optional_size(out, value);
        comma();
    }

    out << "  \"dataset_stats\": ";
    write_string_map(out, manifest.dataset_stats, 2);
    comma();

    out << "  \"source_files\": ";
    write_string_map(out, manifest.source_files, 2);
    comma();

    out << "  \"source_sha256\": ";
    write_string_map(out, manifest.source_sha256, 2);
    comma();

    out << "  \"attempted_stages\": ";
    write_string_array(out, manifest.proof_attempt.attempted_stages, 2);
    comma();

    out << "  \"unsupported_protocol_requirements\": ";
    write_string_array(out, manifest.proof_attempt.unsupported_protocol_requirements, 2);
    out << '\n';
    out << "}\n";
}

}  // namespace zkdl_gat
