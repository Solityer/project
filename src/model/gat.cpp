#include "gatzk/model/gat.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace gatzk::model {
namespace {

std::string load_text_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open checkpoint manifest: " + path.string());
    }
    std::ostringstream stream;
    stream << input.rdbuf();
    return stream.str();
}

std::size_t parse_required_size_t(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing numeric key in checkpoint manifest: " + key);
    }
    return static_cast<std::size_t>(std::stoull(match[1].str()));
}

}  // namespace

ModelParameters build_model_parameters(
    std::size_t input_dim,
    std::size_t hidden_dim,
    std::size_t num_classes,
    std::uint64_t seed) {
    ModelParameters params;
    params.W.assign(input_dim, std::vector<algebra::FieldElement>(hidden_dim, algebra::FieldElement::zero()));
    for (std::size_t row = 0; row < input_dim; ++row) {
        bool has_non_zero = false;
        for (std::size_t col = 0; col < hidden_dim; ++col) {
            const auto enabled = ((row * 17 + col * 31 + seed) % 7U) == 0U;
            params.W[row][col] = enabled ? algebra::FieldElement::one() : algebra::FieldElement::zero();
            has_non_zero = has_non_zero || enabled;
        }
        if (!has_non_zero) {
            params.W[row][row % hidden_dim] = algebra::FieldElement::one();
        }
    }

    params.a_src.resize(hidden_dim, algebra::FieldElement::one());
    params.a_dst.resize(hidden_dim, algebra::FieldElement::one());
    for (std::size_t i = 0; i < hidden_dim; ++i) {
        params.a_src[i] = algebra::FieldElement::from_signed(static_cast<std::int64_t>((i + seed) % 2U + 1U));
        params.a_dst[i] = algebra::FieldElement::from_signed(static_cast<std::int64_t>((i + seed + 1U) % 2U + 1U));
    }

    params.W_out.assign(hidden_dim, std::vector<algebra::FieldElement>(num_classes, algebra::FieldElement::zero()));
    for (std::size_t row = 0; row < hidden_dim; ++row) {
        bool has_non_zero = false;
        for (std::size_t col = 0; col < num_classes; ++col) {
            const auto enabled = ((row * 13 + col * 19 + seed) % 5U) == 0U;
            params.W_out[row][col] = enabled ? algebra::FieldElement::one() : algebra::FieldElement::zero();
            has_non_zero = has_non_zero || enabled;
        }
        if (!has_non_zero) {
            params.W_out[row][row % num_classes] = algebra::FieldElement::one();
        }
    }

    params.b.resize(num_classes, algebra::FieldElement::zero());
    for (std::size_t i = 0; i < num_classes; ++i) {
        params.b[i] = algebra::FieldElement::from_signed(static_cast<std::int64_t>(i % 3U));
    }
    return params;
}

CheckpointBundleInfo inspect_checkpoint_bundle(const std::string& bundle_root) {
    const auto manifest_path = std::filesystem::path(bundle_root) / "manifest.json";
    const auto manifest = load_text_file(manifest_path);

    CheckpointBundleInfo info;
    info.bundle_root = bundle_root;
    info.hidden_head_count = parse_required_size_t(manifest, "hidden_head_count");
    info.has_output_attention_head = manifest.find("\"output_head\"") != std::string::npos;
    return info;
}

bool checkpoint_bundle_matches_single_head_protocol(
    const CheckpointBundleInfo& info,
    std::string* reason) {
    std::vector<std::string> failures;
    if (info.hidden_head_count != 1) {
        failures.push_back(
            "hidden_head_count=" + std::to_string(info.hidden_head_count)
            + " but the current protocol model expects exactly 1 hidden attention head");
    }
    if (info.has_output_attention_head) {
        failures.push_back(
            "the exported bundle contains an output attention head, while the current protocol model expects affine output parameters W_out/b");
    }
    if (failures.empty()) {
        return true;
    }
    if (reason != nullptr) {
        std::ostringstream stream;
        for (std::size_t i = 0; i < failures.size(); ++i) {
            if (i != 0) {
                stream << "; ";
            }
            stream << failures[i];
        }
        *reason = stream.str();
    }
    return false;
}

Matrix project_features(const Matrix& left, const Matrix& right) {
    if (left.empty() || right.empty()) {
        return {};
    }
    const std::size_t shared = left.front().size();
    if (right.size() != shared) {
        throw std::runtime_error("matrix multiply dimension mismatch");
    }
    const std::size_t rows = left.size();
    const std::size_t cols = right.front().size();
    Matrix out(rows, std::vector<algebra::FieldElement>(cols, algebra::FieldElement::zero()));
    const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const auto task_count = rows >= 64 && cpu_count > 1 ? std::min<std::size_t>(cpu_count, rows) : 1;
    const auto chunk_size = (rows + task_count - 1) / task_count;

    std::vector<std::future<void>> futures;
    futures.reserve(task_count);
    for (std::size_t task = 0; task < task_count; ++task) {
        const auto begin = task * chunk_size;
        const auto end = std::min(rows, begin + chunk_size);
        if (begin >= end) {
            break;
        }
        futures.push_back(std::async(std::launch::async, [&, begin, end]() {
            for (std::size_t i = begin; i < end; ++i) {
                std::vector<mcl::Fr> native_row(cols);
                for (auto& value : native_row) {
                    value.clear();
                }
                for (std::size_t k = 0; k < shared; ++k) {
                    const auto& lhs = left[i][k].native();
                    for (std::size_t j = 0; j < cols; ++j) {
                        mcl::Fr term;
                        mcl::Fr::mul(term, lhs, right[k][j].native());
                        mcl::Fr::add(native_row[j], native_row[j], term);
                    }
                }
                for (std::size_t j = 0; j < cols; ++j) {
                    out[i][j] = algebra::FieldElement::from_native(native_row[j]);
                }
            }
        }));
    }
    for (auto& future : futures) {
        future.get();
    }
    return out;
}

std::vector<algebra::FieldElement> matvec_projection(const Matrix& matrix, const std::vector<algebra::FieldElement>& vector) {
    std::vector<algebra::FieldElement> out(matrix.size(), algebra::FieldElement::zero());
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        mcl::Fr sum;
        sum.clear();
        for (std::size_t j = 0; j < vector.size(); ++j) {
            mcl::Fr term;
            mcl::Fr::mul(term, matrix[i][j].native(), vector[j].native());
            mcl::Fr::add(sum, sum, term);
        }
        out[i] = algebra::FieldElement::from_native(sum);
    }
    return out;
}

std::vector<algebra::FieldElement> compress_rows(const Matrix& matrix, const algebra::FieldElement& challenge) {
    std::vector<algebra::FieldElement> out(matrix.size(), algebra::FieldElement::zero());
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        mcl::Fr power = 1;
        mcl::Fr sum;
        sum.clear();
        for (const auto& value : matrix[i]) {
            mcl::Fr term;
            mcl::Fr::mul(term, value.native(), power);
            mcl::Fr::add(sum, sum, term);
            mcl::Fr::mul(power, power, challenge.native());
        }
        out[i] = algebra::FieldElement::from_native(sum);
    }
    return out;
}

Matrix aggregate_by_edges(
    const Matrix& h_prime,
    const std::vector<algebra::FieldElement>& alpha,
    const std::vector<data::Edge>& edges,
    std::size_t num_nodes) {
    const std::size_t width = h_prime.front().size();
    Matrix out(num_nodes, std::vector<algebra::FieldElement>(width, algebra::FieldElement::zero()));
    std::vector<std::vector<mcl::Fr>> native_out(num_nodes, std::vector<mcl::Fr>(width));
    for (auto& row : native_out) {
        for (auto& value : row) {
            value.clear();
        }
    }
    for (std::size_t k = 0; k < edges.size(); ++k) {
        const auto& alpha_k = alpha[k].native();
        for (std::size_t j = 0; j < width; ++j) {
            mcl::Fr term;
            mcl::Fr::mul(term, alpha_k, h_prime[edges[k].src][j].native());
            mcl::Fr::add(native_out[edges[k].dst][j], native_out[edges[k].dst][j], term);
        }
    }
    for (std::size_t i = 0; i < num_nodes; ++i) {
        for (std::size_t j = 0; j < width; ++j) {
            out[i][j] = algebra::FieldElement::from_native(native_out[i][j]);
        }
    }
    return out;
}

Matrix output_projection(
    const Matrix& h_agg,
    const Matrix& w_out,
    const std::vector<algebra::FieldElement>& bias,
    Matrix* linear_part) {
    auto linear = project_features(h_agg, w_out);
    Matrix out = linear;
    for (auto& row : out) {
        for (std::size_t j = 0; j < bias.size(); ++j) {
            row[j] += bias[j];
        }
    }
    if (linear_part != nullptr) {
        *linear_part = linear;
    }
    return out;
}

}
