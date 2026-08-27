#include "gat_ezkl/formal_gat.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gat_ezkl/formal_io.hpp"
#include "gat_ezkl/formal_hash.hpp"

namespace gat_ezkl {
namespace {

using Integer = std::int64_t;
using Matrix = std::vector<std::vector<Integer>>;

struct RoundDivCertificate {
    __int128 a = 0, q = 0, r = 0, delta = 0, d_min = 1, d = 1;
    __int128 lhs = 0, mul_dq = 0, rhs = 0, two_d_minus_one = 0;
};

struct FamilyCertificate {
    std::string label;
    std::size_t input_dim = 0, output_dim = 0;
    __int128 h = 0, w = 0, a_src = 0, a_dst = 0, projection = 0;
    __int128 e_src = 0, e_dst = 0, score = 0, z = 0, m = 0;
    __int128 delta_raw = 0, delta = 0, exp = 0, sum = 0, aggregate_num = 0;
    __int128 projection_mul = 0, projection_acc = 0;
    __int128 attention_mul = 0, attention_acc = 0;
    __int128 aggregate_mul = 0, aggregate_acc = 0;
    __int128 scalar_h = 0, scalar_projection = 0, scalar_e_src = 0;
    __int128 scalar_e_dst = 0, scalar_score = 0, scalar_z = 0, scalar_delta = 0;
    RoundDivCertificate recovery;
    bool has_alignment = false;
    RoundDivCertificate alignment;
    __int128 output_bound = 0, bias = 0;
};

__int128 wide_add(__int128 left, __int128 right, const char* label) {
    __int128 value = 0;
    if (__builtin_add_overflow(left, right, &value) || value < 0)
        throw std::runtime_error(std::string("integer-bound add overflow: ") + label);
    return value;
}

__int128 wide_mul(__int128 left, __int128 right, const char* label) {
    __int128 value = 0;
    if (left < 0 || right < 0 || __builtin_mul_overflow(left, right, &value) || value < 0)
        throw std::runtime_error(std::string("integer-bound multiply overflow: ") + label);
    return value;
}

__int128 signed_add(__int128 left, __int128 right, const char* label) {
    __int128 value = 0;
    if (__builtin_add_overflow(left, right, &value))
        throw std::runtime_error(std::string("signed integer-bound add overflow: ") + label);
    return value;
}

__int128 signed_sub(__int128 left, __int128 right, const char* label) {
    __int128 value = 0;
    if (__builtin_sub_overflow(left, right, &value))
        throw std::runtime_error(std::string("signed integer-bound subtract overflow: ") + label);
    return value;
}

__int128 signed_mul(__int128 left, __int128 right, const char* label) {
    __int128 value = 0;
    if (__builtin_mul_overflow(left, right, &value))
        throw std::runtime_error(std::string("signed integer-bound multiply overflow: ") + label);
    return value;
}

__int128 signed_neg(__int128 value, const char* label) {
    __int128 result = 0;
    if (__builtin_sub_overflow(static_cast<__int128>(0), value, &result))
        throw std::runtime_error(std::string("signed integer-bound negate overflow: ") + label);
    return result;
}

__int128 signed_abs(__int128 value, const char* label) {
    return value < 0 ? signed_neg(value, label) : value;
}

std::string decimal(__int128 value) {
    if (value < 0) throw std::runtime_error("negative bound cannot be serialized");
    if (value == 0) return "0";
    std::string output;
    while (value != 0) {
        output.push_back(static_cast<char>('0' + value % 10));
        value /= 10;
    }
    std::reverse(output.begin(), output.end());
    return output;
}

__int128 nearest_q_bound(__int128 a, __int128 d_min) {
    return wide_add(wide_mul(2, a, "round q"), d_min, "round q") /
           wide_mul(2, d_min, "round q");
}

RoundDivCertificate round_certificate(__int128 a, __int128 d_min, __int128 d) {
    if (d_min <= 0 || d < d_min) throw std::runtime_error("invalid division denominator bounds");
    RoundDivCertificate out;
    out.a = a;
    out.d_min = d_min;
    out.d = d;
    out.q = nearest_q_bound(a, d_min);
    out.r = out.delta = wide_add(wide_mul(2, d, "round remainder"), -1, "round remainder");
    out.lhs = wide_add(wide_mul(2, out.a, "round lhs"), out.d, "round lhs");
    out.mul_dq = wide_mul(out.d, out.q, "round dq");
    out.rhs = wide_add(wide_mul(2, out.mul_dq, "round rhs"), out.r, "round rhs");
    out.two_d_minus_one = wide_add(wide_mul(2, out.d, "round 2d"), -1, "round 2d");
    return out;
}

std::string round_json(const RoundDivCertificate& value) {
    std::ostringstream output;
    output << "{\"a\":" << decimal(value.a) << ",\"q\":" << decimal(value.q)
           << ",\"r\":" << decimal(value.r) << ",\"delta\":" << decimal(value.delta)
           << ",\"d_min\":" << decimal(value.d_min) << ",\"d\":" << decimal(value.d)
           << ",\"lhs_2a_plus_d\":" << decimal(value.lhs)
           << ",\"mul_dq\":" << decimal(value.mul_dq)
           << ",\"rhs_2dq_plus_r\":" << decimal(value.rhs)
           << ",\"two_d_minus_one\":" << decimal(value.two_d_minus_one) << "}";
    return output.str();
}

void write_u64_le(std::ofstream& output, std::uint64_t value) {
    std::array<char, 8> bytes{};
    for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<char>(value >> (8 * i));
    output.write(bytes.data(), bytes.size());
}

void write_table_row(std::ofstream& output, std::uint64_t kind, Integer x, Integer y) {
    write_u64_le(output, kind);
    write_u64_le(output, static_cast<std::uint64_t>(x));
    write_u64_le(output, static_cast<std::uint64_t>(y));
}

std::string family_json(const FamilyCertificate& value) {
    std::ostringstream output;
    output << "{\"label\":\"" << json_escape(value.label) << "\""
           << ",\"input_dim\":" << value.input_dim << ",\"output_dim\":" << value.output_dim
           << ",\"h\":" << decimal(value.h) << ",\"w\":" << decimal(value.w)
           << ",\"a_src\":" << decimal(value.a_src) << ",\"a_dst\":" << decimal(value.a_dst)
           << ",\"projection\":" << decimal(value.projection)
           << ",\"e_src\":" << decimal(value.e_src) << ",\"e_dst\":" << decimal(value.e_dst)
           << ",\"score\":" << decimal(value.score) << ",\"z\":" << decimal(value.z)
           << ",\"m\":" << decimal(value.m) << ",\"delta_raw\":" << decimal(value.delta_raw)
           << ",\"delta\":" << decimal(value.delta) << ",\"exp\":" << decimal(value.exp)
           << ",\"sum\":" << decimal(value.sum)
           << ",\"aggregate_num\":" << decimal(value.aggregate_num)
           << ",\"projection_mul\":" << decimal(value.projection_mul)
           << ",\"projection_acc\":" << decimal(value.projection_acc)
           << ",\"attention_mul\":" << decimal(value.attention_mul)
           << ",\"attention_acc\":" << decimal(value.attention_acc)
           << ",\"aggregate_mul\":" << decimal(value.aggregate_mul)
           << ",\"aggregate_acc\":" << decimal(value.aggregate_acc)
           << ",\"scalar_h\":" << decimal(value.scalar_h)
           << ",\"scalar_projection\":" << decimal(value.scalar_projection)
           << ",\"scalar_e_src\":" << decimal(value.scalar_e_src)
           << ",\"scalar_e_dst\":" << decimal(value.scalar_e_dst)
           << ",\"scalar_score\":" << decimal(value.scalar_score)
           << ",\"scalar_z\":" << decimal(value.scalar_z)
           << ",\"scalar_delta\":" << decimal(value.scalar_delta)
           << ",\"recovery\":" << round_json(value.recovery)
           << ",\"alignment\":" << (value.has_alignment ? round_json(value.alignment) : "null")
           << ",\"output_bound\":" << decimal(value.output_bound)
           << ",\"bias\":" << decimal(value.bias) << "}";
    return output.str();
}

Integer checked(__int128 value, const char* label) {
    if (value < std::numeric_limits<Integer>::min() || value > std::numeric_limits<Integer>::max()) {
        throw std::runtime_error(std::string("canonical integer overflow: ") + label);
    }
    return static_cast<Integer>(value);
}

Integer round_nearest_away(double value) {
    if (!std::isfinite(value)) throw std::runtime_error("non-finite canonical quantization input");
    const auto rounded = value >= 0.0
        ? std::floor(static_cast<long double>(value) + 0.5L)
        : std::ceil(static_cast<long double>(value) - 0.5L);
    if (!std::isfinite(rounded)
        || rounded < static_cast<long double>(std::numeric_limits<Integer>::min())
        || rounded > static_cast<long double>(std::numeric_limits<Integer>::max())) {
        throw std::runtime_error("canonical quantization exceeds signed i64");
    }
    return static_cast<Integer>(rounded);
}

Integer signed_round_div(__int128 numerator, __int128 denominator) {
    if (denominator <= 0) throw std::runtime_error("SignedRoundDiv denominator must be positive");
    const bool negative = numerator < 0;
    const auto magnitude = signed_abs(numerator, "SignedRoundDiv magnitude");
    const auto twice_magnitude = signed_mul(magnitude, 2, "SignedRoundDiv 2a");
    const auto rounded_numerator = signed_add(twice_magnitude, denominator, "SignedRoundDiv numerator");
    const auto rounded_denominator = signed_mul(denominator, 2, "SignedRoundDiv denominator");
    const auto quotient = rounded_numerator / rounded_denominator;
    const auto signed_quotient = negative ? signed_neg(quotient, "SignedRoundDiv result") : quotient;
    return checked(signed_quotient, "SignedRoundDiv");
}

Integer lrelu(Integer value) {
    if (value >= 0) return value;
    const auto magnitude = signed_abs(static_cast<__int128>(value), "LReLU input");
    const auto result = signed_round_div(magnitude, 5);
    return checked(signed_neg(static_cast<__int128>(result), "LReLU result"), "LReLU");
}

Integer exp_neg(Integer delta) {
    if (delta < 0) throw std::runtime_error("ExpMap input must be nonnegative");
    const auto input = static_cast<double>(delta) / 256.0;
    return std::max<Integer>(0, round_nearest_away(16.0 * std::exp(-input)));
}

Integer elu(Integer value) {
    if (value >= 0) return value;
    const auto input = static_cast<double>(value) / 16.0;
    return round_nearest_away(16.0 * (std::exp(input) - 1.0));
}

Integer abs_max(const std::vector<Integer>& values) {
    Integer bound = 0;
    for (const auto value : values) bound = std::max(bound, checked(signed_abs(value, "absolute bound"), "absolute bound"));
    return bound;
}

const TensorInfo& tensor(const CheckpointBundle& bundle, const std::string& name) {
    const auto it = bundle.tensors.find(name);
    if (it == bundle.tensors.end()) throw std::runtime_error("missing canonical tensor: " + name);
    return it->second;
}

std::vector<Integer> quantized_tensor(
    const CheckpointBundle& bundle,
    const std::string& name,
    Integer scale) {
    std::vector<Integer> out;
    if (scale <= 0) throw std::runtime_error("canonical tensor scale must be positive");
    for (const auto value : tensor(bundle, name).values) {
        if (!std::isfinite(value)) throw std::runtime_error("canonical tensor value is non-finite");
        const auto scaled = static_cast<long double>(value) * static_cast<long double>(scale);
        if (!std::isfinite(scaled)
            || scaled < static_cast<long double>(std::numeric_limits<Integer>::min()) - 0.5L
            || scaled > static_cast<long double>(std::numeric_limits<Integer>::max()) + 0.5L) {
            throw std::runtime_error("canonical tensor value exceeds signed i64 range");
        }
        out.push_back(round_nearest_away(static_cast<double>(scaled)));
    }
    return out;
}

Matrix reshape(const std::vector<Integer>& values, std::size_t rows, std::size_t columns) {
    if (columns != 0 && rows > std::numeric_limits<std::size_t>::max() / columns
        || values.size() != rows * columns) throw std::runtime_error("canonical tensor shape mismatch");
    Matrix out(rows, std::vector<Integer>(columns));
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            out[row][column] = values[row * columns + column];
    return out;
}

Matrix matmul(const Matrix& left, const Matrix& right) {
    if (left.empty() || right.empty() || left.front().size() != right.size())
        throw std::runtime_error("canonical matmul shape mismatch");
    Matrix out(left.size(), std::vector<Integer>(right.front().size(), 0));
    for (std::size_t row = 0; row < left.size(); ++row)
        for (std::size_t shared = 0; shared < right.size(); ++shared)
            for (std::size_t column = 0; column < right.front().size(); ++column)
                out[row][column] = checked(__int128{out[row][column]} +
                    __int128{left[row][shared]} * right[shared][column], "projection");
    return out;
}

std::vector<Integer> matvec(const Matrix& matrix, const std::vector<Integer>& vector) {
    std::vector<Integer> out(matrix.size(), 0);
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        if (matrix[row].size() != vector.size()) throw std::runtime_error("canonical matvec shape mismatch");
        __int128 sum = 0;
        for (std::size_t column = 0; column < vector.size(); ++column)
            sum = signed_add(sum, signed_mul(__int128{matrix[row][column]}, vector[column], "attention dot product"),
                             "attention dot accumulator");
        out[row] = checked(sum, "attention dot");
    }
    return out;
}

struct IntegerHead {
    Matrix projection;
    std::vector<Integer> a_src;
    std::vector<Integer> a_dst;
    std::vector<Integer> bias;
};

Matrix forward_head(
    const Matrix& input,
    const std::vector<Edge>& edges,
    const IntegerHead& head,
    bool hidden,
    FamilyCertificate* certificate) {
    if (certificate == nullptr) throw std::runtime_error("missing formal family certificate sink");
    certificate->input_dim = input.front().size();
    certificate->output_dim = head.projection.front().size();
    for (const auto& row : input) certificate->h = std::max<__int128>(certificate->h, abs_max(row));
    for (const auto& row : head.projection) certificate->w = std::max<__int128>(certificate->w, abs_max(row));
    certificate->a_src = abs_max(head.a_src);
    certificate->a_dst = abs_max(head.a_dst);
    for (const auto& row : input) for (std::size_t column = 0; column < head.projection.front().size(); ++column) {
        __int128 accumulator = 0;
        for (std::size_t shared = 0; shared < row.size(); ++shared) {
            const auto product = __int128{row[shared]} * head.projection[shared][column];
            certificate->projection_mul = std::max(certificate->projection_mul,
                                                    signed_abs(product, "projection product"));
            accumulator = signed_add(accumulator, product, "projection accumulator");
            certificate->projection_acc = std::max(certificate->projection_acc,
                                                    signed_abs(accumulator, "projection accumulator"));
        }
    }
    const auto projected = matmul(input, head.projection);
    for (const auto& row : projected) certificate->projection = std::max<__int128>(certificate->projection, abs_max(row));
    const auto source_score = matvec(projected, head.a_src);
    const auto target_score = matvec(projected, head.a_dst);
    for (std::size_t row = 0; row < projected.size(); ++row) {
        __int128 src_acc = 0, dst_acc = 0;
        for (std::size_t column = 0; column < projected[row].size(); ++column) {
            const auto src_product = __int128{projected[row][column]} * head.a_src[column];
            const auto dst_product = __int128{projected[row][column]} * head.a_dst[column];
            certificate->attention_mul = std::max({certificate->attention_mul,
                signed_abs(src_product, "source attention product"),
                signed_abs(dst_product, "destination attention product")});
            src_acc = signed_add(src_acc, src_product, "source attention accumulator");
            dst_acc = signed_add(dst_acc, dst_product, "destination attention accumulator");
            certificate->attention_acc = std::max({certificate->attention_acc,
                signed_abs(src_acc, "source attention accumulator"),
                signed_abs(dst_acc, "destination attention accumulator")});
        }
        certificate->e_src = std::max<__int128>(certificate->e_src,
                                                signed_abs(source_score[row], "source attention score"));
        certificate->e_dst = std::max<__int128>(certificate->e_dst,
                                                signed_abs(target_score[row], "destination attention score"));
    }
    const auto node_count = input.size();
    const auto width = projected.front().size();
    std::vector<Integer> score(edges.size()), activated(edges.size()), maximum(
        node_count, std::numeric_limits<Integer>::min());
    std::vector<std::vector<std::size_t>> incoming(node_count);
    for (std::size_t edge = 0; edge < edges.size(); ++edge) {
        score[edge] = checked(__int128{source_score.at(edges[edge].src)} +
            target_score.at(edges[edge].dst), "edge score");
        activated[edge] = lrelu(score[edge]);
        certificate->score = std::max<__int128>(certificate->score,
                                                signed_abs(score[edge], "edge score"));
        certificate->z = std::max<__int128>(certificate->z,
                                            signed_abs(activated[edge], "activated edge score"));
        incoming.at(edges[edge].dst).push_back(edge);
        maximum.at(edges[edge].dst) = std::max(maximum.at(edges[edge].dst), activated[edge]);
    }
    std::vector<Integer> exponential(edges.size()), denominator(node_count, 0);
    for (std::size_t edge = 0; edge < edges.size(); ++edge) {
        const auto dst = edges[edge].dst;
        const auto delta = checked(signed_sub(maximum[dst], activated[edge], "softmax delta"),
                                   "softmax delta");
        exponential[edge] = exp_neg(delta);
        certificate->m = std::max<__int128>(certificate->m,
                                            signed_abs(maximum[dst], "segment maximum"));
        certificate->delta_raw = certificate->delta = std::max<__int128>(
            certificate->delta, delta);
        certificate->exp = std::max<__int128>(certificate->exp, exponential[edge]);
        denominator[dst] = checked(__int128{denominator[dst]} + exponential[edge], "softmax Sum");
        certificate->sum = std::max<__int128>(certificate->sum, denominator[dst]);
    }
    Matrix numerator(node_count, std::vector<Integer>(width, 0));
    for (std::size_t edge = 0; edge < edges.size(); ++edge) {
        for (std::size_t column = 0; column < width; ++column) {
            const auto term = __int128{exponential[edge]} * projected[edges[edge].src][column];
            certificate->aggregate_mul = std::max(certificate->aggregate_mul,
                                                   signed_abs(term, "aggregation product"));
            numerator[edges[edge].dst][column] = checked(
                __int128{numerator[edges[edge].dst][column]} + term,
                "integer aggregation numerator");
            certificate->aggregate_acc = certificate->aggregate_num = std::max<__int128>(
                certificate->aggregate_acc,
                signed_abs(numerator[edges[edge].dst][column], "aggregation accumulator"));
        }
    }
    Matrix output(node_count, std::vector<Integer>(width, 0));
    __int128 recovery_a = 0, recovery_d_min = std::numeric_limits<Integer>::max(), recovery_d = 0;
    for (std::size_t node = 0; node < node_count; ++node) {
        if (denominator[node] <= 0) throw std::runtime_error("nonpositive Softmax denominator");
        for (std::size_t column = 0; column < width; ++column) {
            const auto recovered = hidden
                ? signed_round_div(numerator[node][column], __int128{16} * denominator[node])
                : signed_round_div(__int128{16} * numerator[node][column],
                                   __int128{256} * denominator[node]);
            output[node][column] = hidden ? elu(recovered) : recovered;
            recovery_a = std::max<__int128>(recovery_a, hidden
                ? signed_abs(numerator[node][column], "hidden recovery numerator")
                : signed_mul(16, signed_abs(numerator[node][column], "output recovery numerator"),
                             "output recovery numerator"));
            const auto actual_d = hidden ? signed_mul(16, denominator[node], "hidden recovery denominator")
                                         : signed_mul(256, denominator[node], "output recovery denominator");
            recovery_d_min = std::min(recovery_d_min, actual_d);
            recovery_d = std::max(recovery_d, actual_d);
            certificate->output_bound = std::max<__int128>(certificate->output_bound,
                                                            signed_abs(output[node][column], "forward output"));
        }
    }
    certificate->recovery = round_certificate(recovery_a, recovery_d_min, recovery_d);
    return output;
}

Matrix concatenate(const std::vector<Matrix>& heads) {
    if (heads.empty()) throw std::runtime_error("hidden layer has no heads");
    Matrix out(heads.front().size());
    for (const auto& head : heads)
        for (std::size_t node = 0; node < out.size(); ++node)
            out[node].insert(out[node].end(), head[node].begin(), head[node].end());
    return out;
}

void write_i64_payload(const std::filesystem::path& path, const Matrix& matrix) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t rows = matrix.size();
    const std::uint64_t columns = matrix.empty() ? 0 : matrix.front().size();
    output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
    for (const auto& row : matrix)
        for (const auto value : row) output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output) throw std::runtime_error("failed to write canonical integer features");
}

void write_tensor(std::ofstream& output, const std::string& name,
                  const TensorInfo& source, const std::vector<Integer>& values) {
    output << "TENSOR " << name << " " << source.shape.size();
    for (const auto dimension : source.shape) output << " " << dimension;
    output << " " << values.size();
    for (const auto value : values) output << " " << value;
    output << "\n";
}

}  // namespace

FormalProfileIds write_formal_gat_artifacts(
    const std::filesystem::path& output_dir,
    const AppConfig& config,
    const NormalizedGraph& graph,
    const CheckpointBundle& checkpoint,
    std::size_t max_in_degree) {
    Matrix features(graph.node_count, std::vector<Integer>(graph.feature_dim));
    Integer feature_bound = 0;
    for (std::size_t node = 0; node < graph.node_count; ++node)
        for (std::size_t column = 0; column < graph.feature_dim; ++column) {
            features[node][column] = round_nearest_away(graph.features_fp[node][column]);
            feature_bound = std::max(feature_bound, static_cast<Integer>(std::llabs(features[node][column])));
        }
    write_i64_payload(output_dir / "node_features.i64.bin", features);

    std::ofstream model(output_dir / "model_parameters.i64.txt", std::ios::trunc);
    if (!model) throw std::runtime_error("failed to write canonical model parameters");
    auto hidden_specs = checkpoint.hidden_heads;
    std::sort(hidden_specs.begin(), hidden_specs.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.layer_index, lhs.local_head_index, lhs.global_head_index)
            < std::tie(rhs.layer_index, rhs.local_head_index, rhs.global_head_index);
    });
    auto output_specs = checkpoint.output_heads;
    std::sort(output_specs.begin(), output_specs.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.head_index < rhs.head_index;
    });
    std::map<std::string, std::vector<Integer>> canonical;
    Integer weight_bound = 0, hidden_attention_bound = 0, output_attention_bound = 0, bias_bound = 0;
    for (const auto& head : hidden_specs) {
        for (const auto& name : {head.seq_kernel, head.attn_src_kernel, head.attn_dst_kernel}) {
            const auto values = quantized_tensor(checkpoint, name, 16);
            write_tensor(model, name, tensor(checkpoint, name), values);
            canonical.emplace(name, values);
        }
        weight_bound = std::max(weight_bound, abs_max(canonical.at(head.seq_kernel)));
        hidden_attention_bound = std::max({hidden_attention_bound,
            abs_max(canonical.at(head.attn_src_kernel)), abs_max(canonical.at(head.attn_dst_kernel))});
    }
    for (const auto& head : output_specs) {
        const auto weights = quantized_tensor(checkpoint, head.seq_kernel, 16);
        const auto a_src = quantized_tensor(checkpoint, head.attn_src_kernel, 1);
        const auto a_dst = quantized_tensor(checkpoint, head.attn_dst_kernel, 1);
        const auto bias = quantized_tensor(checkpoint, head.output_bias, 16);
        for (const auto& pair : {std::pair<std::string, std::vector<Integer>>(head.seq_kernel, weights),
                                 {head.attn_src_kernel, a_src}, {head.attn_dst_kernel, a_dst},
                                 {head.output_bias, bias}}) {
            write_tensor(model, pair.first, tensor(checkpoint, pair.first), pair.second);
            canonical.emplace(pair.first, pair.second);
        }
        weight_bound = std::max(weight_bound, abs_max(weights));
        output_attention_bound = std::max({output_attention_bound, abs_max(a_src), abs_max(a_dst)});
        bias_bound = std::max(bias_bound, abs_max(bias));
    }
    model.close();

    Matrix layer_input = features;
    std::vector<Integer> final_logits;
    std::vector<FamilyCertificate> families;
    for (std::size_t layer = 0; layer < checkpoint.hidden_profile.size(); ++layer) {
        std::vector<Matrix> heads;
        for (const auto& spec : hidden_specs) if (spec.layer_index == layer) {
            FamilyCertificate family;
            family.label = "hidden.layer_" + std::to_string(layer) + ".head_" +
                           std::to_string(spec.local_head_index);
            IntegerHead head{reshape(canonical.at(spec.seq_kernel), layer_input.front().size(),
                                     checkpoint.hidden_profile[layer].head_dim),
                             canonical.at(spec.attn_src_kernel), canonical.at(spec.attn_dst_kernel), {}};
            heads.push_back(forward_head(layer_input, graph.edges, head, true, &family));
            families.push_back(family);
        }
        layer_input = concatenate(heads);
    }
    std::vector<Matrix> output_heads;
    for (const auto& spec : output_specs) {
        FamilyCertificate family;
        family.label = "output.head_" + std::to_string(spec.head_index);
        IntegerHead head{reshape(canonical.at(spec.seq_kernel), layer_input.front().size(), graph.class_count),
                         canonical.at(spec.attn_src_kernel), canonical.at(spec.attn_dst_kernel),
                         canonical.at(spec.output_bias)};
            if (head.bias.size() != graph.class_count) {
                throw std::runtime_error("output bias length does not match class count");
            }
            auto output = forward_head(layer_input, graph.edges, head, false, &family);
        family.bias = abs_max(head.bias);
        __int128 alignment_a = 0, head_bound = 0;
        for (std::size_t node = 0; node < graph.node_count; ++node)
            for (std::size_t column = 0; column < graph.class_count; ++column) {
                alignment_a = std::max<__int128>(
                    alignment_a, signed_mul(16, signed_abs(output[node][column], "output alignment"),
                                            "output alignment"));
                output[node][column] = checked(
                    signed_add(output[node][column], head.bias.at(column), "output bias"), "output bias");
                head_bound = std::max<__int128>(head_bound,
                                                signed_abs(output[node][column], "output head bound"));
            }
        family.has_alignment = true;
        family.alignment = round_certificate(alignment_a, 16, 16);
        family.output_bound = head_bound;
        families.push_back(family);
        output_heads.push_back(std::move(output));
    }
    for (std::size_t node = 0; node < graph.node_count; ++node)
        for (std::size_t column = 0; column < graph.class_count; ++column) {
            __int128 sum = 0;
            for (const auto& head : output_heads) {
                sum = signed_add(sum, head[node][column], "final output head sum");
            }
            final_logits.push_back(signed_round_div(sum, checkpoint.K_out));
        }

    // Freeze the main experiment's quantization profile.  The checkpoint's
    // source IDs are validated separately; these collision-resistant IDs bind
    // the actual scale semantics and exact integer rows used by this backend.
    std::ostringstream quant_core;
    quant_core << "{\"rounding_mode\":\"nearest_ties_away\",\"limb_bits\":" << config.range_bits
          << ",\"feature_scale\":1,\"hidden_weight_scale\":16,\"hidden_attention_scale\":16"
          << ",\"hidden_projection_scale\":16,\"delta_scale\":256,\"exp_scale\":16"
          << ",\"hidden_recovery_num_scale\":1,\"hidden_recovery_den_scale\":16"
          << ",\"elu_input_scale\":16,\"elu_output_scale\":16"
          << ",\"agg_scale\":16,\"output_weight_scale\":16"
          << ",\"output_attention_scale\":1,\"output_projection_scale\":256"
          << ",\"output_pre_scale\":16,\"output_head_scale\":16,\"output_bias_scale\":16"
          << ",\"output_scale\":16}";
    const auto quant_id = formal_sha256_text(
        "formal_quant_profile", "canonical_json", quant_core.str());
    const auto quant_text = quant_core.str().substr(0, quant_core.str().size() - 1)
        + ",\"source_quant_cfg_id\":\"" + json_escape(config.quant_cfg_id)
        + "\",\"canonical_digest\":\"" + quant_id + "\"}\n";
    write_text_file(output_dir / "formal_quant_config.json", quant_text);

    const std::uint64_t configured_range_size = std::uint64_t{1} << config.range_bits;
    // The table domain is part of the canonical proof contract.  It must be
    // large enough for the conservative scalar family ledger, rather than a
    // fixed range_bits-sized window that can reject a valid registered model.
    // The exact domain is finalized below after the scalar family recurrence
    // has been evaluated; the configured limb range remains unchanged.
    Integer lrelu_min = -checked(configured_range_size, "lrelu minimum");
    Integer lrelu_max = checked(configured_range_size, "lrelu maximum");
    Integer elu_bound = checked(configured_range_size, "ELU table bound");
    Integer exp_max = checked(configured_range_size - 1, "ExpMap table bound");

    __int128 scalar_hidden_h = feature_bound;
    for (std::size_t layer = 0; layer < checkpoint.hidden_profile.size(); ++layer) {
        __int128 next_h = 0;
        for (auto& family : families) {
            const auto prefix = "hidden.layer_" + std::to_string(layer) + ".";
            if (family.label.rfind(prefix, 0) != 0) continue;
            family.scalar_h = scalar_hidden_h;
            family.scalar_projection = wide_mul(family.input_dim,
                wide_mul(family.scalar_h, family.w, "scalar projection"), "scalar projection");
            family.scalar_e_src = wide_mul(family.output_dim,
                wide_mul(family.scalar_projection, family.a_src, "scalar Esrc"), "scalar Esrc");
            family.scalar_e_dst = wide_mul(family.output_dim,
                wide_mul(family.scalar_projection, family.a_dst, "scalar Edst"), "scalar Edst");
            family.scalar_score = wide_add(family.scalar_e_src, family.scalar_e_dst, "scalar score");
            family.scalar_z = std::max<__int128>(
                std::llabs(lrelu(checked(-family.scalar_score, "scalar score"))),
                std::llabs(lrelu(checked(family.scalar_score, "scalar score"))));
            family.scalar_delta = wide_mul(2, family.scalar_z, "scalar delta");
            const auto scalar_sum = wide_mul(max_in_degree, 16, "scalar Sum");
            const auto scalar_num = wide_mul(scalar_sum, family.scalar_projection, "scalar aggregate");
            family.h = family.scalar_h;
            family.projection_mul = wide_mul(family.h, family.w, "scalar projection term");
            family.projection_acc = family.projection = family.scalar_projection;
            family.attention_mul = std::max(
                wide_mul(family.projection, family.a_src, "scalar attention term"),
                wide_mul(family.projection, family.a_dst, "scalar attention term"));
            family.attention_acc = std::max(family.scalar_e_src, family.scalar_e_dst);
            family.e_src = family.scalar_e_src;
            family.e_dst = family.scalar_e_dst;
            family.score = family.scalar_score;
            family.z = family.m = family.scalar_z;
            family.delta_raw = family.delta = family.scalar_delta;
            family.exp = 16;
            family.sum = scalar_sum;
            family.aggregate_mul = wide_mul(16, family.projection, "scalar aggregate term");
            family.aggregate_acc = family.aggregate_num = scalar_num;
            family.recovery = round_certificate(scalar_num, 256, wide_mul(16, scalar_sum, "scalar recovery d"));
            family.has_alignment = false;
            family.bias = 0;
            family.output_bound = std::max<__int128>(family.recovery.q, 16);
            next_h = std::max(next_h, family.output_bound);
        }
        scalar_hidden_h = next_h;
    }
    for (auto& family : families) {
        if (family.label.rfind("output.", 0) != 0) continue;
        family.scalar_h = scalar_hidden_h;
        family.scalar_projection = wide_mul(family.input_dim,
            wide_mul(family.scalar_h, family.w, "scalar projection"), "scalar projection");
        family.scalar_e_src = wide_mul(family.output_dim,
            wide_mul(family.scalar_projection, family.a_src, "scalar Esrc"), "scalar Esrc");
        family.scalar_e_dst = wide_mul(family.output_dim,
            wide_mul(family.scalar_projection, family.a_dst, "scalar Edst"), "scalar Edst");
        family.scalar_score = wide_add(family.scalar_e_src, family.scalar_e_dst, "scalar score");
        family.scalar_z = std::max<__int128>(
            std::llabs(lrelu(checked(-family.scalar_score, "scalar score"))),
            std::llabs(lrelu(checked(family.scalar_score, "scalar score"))));
        family.scalar_delta = wide_mul(2, family.scalar_z, "scalar delta");
        const auto scalar_sum = wide_mul(max_in_degree, 16, "scalar Sum");
        const auto scalar_num = wide_mul(scalar_sum, family.scalar_projection, "scalar aggregate");
        family.h = family.scalar_h;
        family.projection_mul = wide_mul(family.h, family.w, "scalar projection term");
        family.projection_acc = family.projection = family.scalar_projection;
        family.attention_mul = std::max(
            wide_mul(family.projection, family.a_src, "scalar attention term"),
            wide_mul(family.projection, family.a_dst, "scalar attention term"));
        family.attention_acc = std::max(family.scalar_e_src, family.scalar_e_dst);
        family.e_src = family.scalar_e_src;
        family.e_dst = family.scalar_e_dst;
        family.score = family.scalar_score;
        family.z = family.m = family.scalar_z;
        family.delta_raw = family.delta = family.scalar_delta;
        family.exp = 16;
        family.sum = scalar_sum;
        family.aggregate_mul = wide_mul(16, family.projection, "scalar aggregate term");
        family.aggregate_acc = family.aggregate_num = scalar_num;
        family.recovery = round_certificate(
            wide_mul(16, scalar_num, "scalar output recovery a"), 4096,
            wide_mul(256, scalar_sum, "scalar output recovery d"));
        family.has_alignment = true;
        family.alignment = round_certificate(wide_mul(16, family.recovery.q, "scalar alignment a"), 16, 16);
        family.output_bound = wide_add(family.alignment.q, family.bias, "scalar output+bias");
    }

    __int128 maximum_delta = 0;
    __int128 maximum_scalar_score = 0;
    __int128 maximum_scalar_delta = 0;
    __int128 maximum_scalar_elu = configured_range_size;
    __int128 maximum_output_head = 0;
    for (const auto& family : families) {
        maximum_scalar_score = std::max(maximum_scalar_score, family.scalar_score);
        maximum_scalar_delta = std::max(maximum_scalar_delta, family.scalar_delta);
        if (family.label.rfind("hidden.", 0) == 0)
            maximum_scalar_elu = std::max(maximum_scalar_elu, family.recovery.q);
        maximum_delta = std::max(maximum_delta, family.delta);
        if (family.label.rfind("output.", 0) == 0)
            maximum_output_head = std::max(maximum_output_head, family.output_bound);
    }
    lrelu_min = checked(-std::max<__int128>(maximum_scalar_score, configured_range_size),
                        "derived lrelu minimum");
    lrelu_max = checked(std::max<__int128>(maximum_scalar_score, configured_range_size),
                        "derived lrelu maximum");
    elu_bound = checked(maximum_scalar_elu, "derived ELU table bound");
    exp_max = checked(std::max<__int128>(maximum_scalar_delta, configured_range_size - 1),
                      "derived ExpMap table bound");
    // The scalar certificate is the mandatory gate.  The actual trace must
    // also fit the same frozen rows, but widening the rows is a compile-time
    // contract change, not a witness-dependent fallback.
    for (const auto& family : families) {
        if (-family.scalar_score < lrelu_min || family.scalar_score > lrelu_max
            || family.scalar_delta > exp_max)
            throw std::runtime_error("derived scalar family/table gate failed for " + family.label);
        if (family.score > lrelu_max || -family.score < lrelu_min || family.delta > exp_max)
            throw std::runtime_error("derived static table domain is too small for " + family.label);
        if (family.label.rfind("hidden.", 0) == 0 && family.recovery.q > elu_bound)
            throw std::runtime_error("derived ELU table domain is too small for " + family.label);
    }
    const auto output_sum_bound = wide_mul(checkpoint.K_out, maximum_output_head, "final head sum");
    const auto final_average = round_certificate(output_sum_bound, checkpoint.K_out, checkpoint.K_out);
    const auto max_relu_bound = checked(std::max<__int128>(
        wide_add(maximum_delta, 1, "MaxReLU table"),
        max_in_degree > 0 ? max_in_degree - 1 : 0), "MaxReLU table");

    const auto main_table_path = output_dir / "formal_main_tables.i64.bin";
    std::ofstream table_rows(main_table_path, std::ios::binary | std::ios::trunc);
    table_rows.write("EZKLGATTABLE1\n", 14);
    for (Integer value = lrelu_min;; ++value) {
        write_table_row(table_rows, 2, value, lrelu(value)); if (value == lrelu_max) break;
    }
    for (Integer value = -elu_bound;; ++value) {
        write_table_row(table_rows, 3, value, elu(value)); if (value == elu_bound) break;
    }
    for (Integer value = 0;; ++value) {
        write_table_row(table_rows, 4, value, exp_neg(value)); if (value == exp_max) break;
    }
    if (!table_rows) throw std::runtime_error("failed to write canonical main table rows");
    table_rows.close();
    const auto main_rows_id = formal_sha256_file(
        "formal_static_table_rows", "canonical_rows", main_table_path);
    const auto main_spec_json = [&](const std::string& digest) {
        std::ostringstream tables;
        tables << "{\"lrelu_min\":" << lrelu_min << ",\"lrelu_max\":" << lrelu_max
               << ",\"elu_min\":" << -elu_bound << ",\"elu_max\":" << elu_bound
               << ",\"exp_max\":" << exp_max
               << ",\"generator_id\":\"main_gatzk_quantized_tables_v1\""
               << ",\"profile_version\":\"main_gatzk_table_profile_v1\""
               << ",\"rounding_mode\":\"nearest_ties_away\""
               << ",\"clipping_rule\":\"exp_nonnegative_clamp_zero\""
               << ",\"canonical_order\":\"kind_then_signed_input_ascending\""
               << ",\"lrelu_scale\":256,\"elu_input_scale\":16,\"elu_output_scale\":16"
               << ",\"exp_input_scale\":256,\"exp_output_scale\":16"
               << ",\"source_static_table_id\":\"" << json_escape(config.static_table_id) << "\""
               << ",\"rows_path\":\"formal_main_tables.i64.bin\",\"rows_digest\":\""
               << main_rows_id << "\",\"canonical_digest\":\"" << digest << "\"}";
        return tables.str();
    };
    const auto table_id = formal_sha256_text(
        "formal_static_table_profile", "canonical_json", main_spec_json(""));
    write_text_file(output_dir / "formal_table_spec.json", main_spec_json(table_id) + "\n");

    const auto aux_table_path = output_dir / "backend_aux_tables.i64.bin";
    std::ofstream aux_rows(aux_table_path, std::ios::binary | std::ios::trunc);
    aux_rows.write("EZKLGATTABLE1\n", 14);
    for (Integer value = 0; value < checked(configured_range_size, "limb rows"); ++value)
        write_table_row(aux_rows, 1, value, value);
    for (Integer value = -max_relu_bound;; ++value) {
        write_table_row(aux_rows, 5, value, std::max<Integer>(value, 0));
        if (value == max_relu_bound) break;
    }
    if (!aux_rows) throw std::runtime_error("failed to write canonical backend auxiliary rows");
    aux_rows.close();
    const auto aux_rows_id = formal_sha256_file(
        "ezkl_backend_aux_table_rows", "canonical_rows", aux_table_path);
    const auto aux_spec_json = [&](const std::string& digest) {
        std::ostringstream aux;
        aux << "{\"max_relu_min\":" << -max_relu_bound
            << ",\"max_relu_max\":" << max_relu_bound
            << ",\"limb_max\":" << (configured_range_size - 1)
            << ",\"generator_id\":\"ezkl_backend_aux_tables_v1\""
            << ",\"profile_version\":\"ezkl_backend_aux_table_profile_v1\""
            << ",\"canonical_order\":\"kind_then_signed_input_ascending\""
            << ",\"rows_path\":\"backend_aux_tables.i64.bin\",\"rows_digest\":\""
            << aux_rows_id << "\",\"canonical_digest\":\"" << digest << "\"}";
        return aux.str();
    };
    const auto aux_table_id = formal_sha256_text(
        "ezkl_backend_aux_table_profile", "canonical_json", aux_spec_json(""));
    write_text_file(output_dir / "backend_aux_table_spec.json", aux_spec_json(aux_table_id) + "\n");

    std::ostringstream bounds;
    bounds << "{\"feature\":" << feature_bound << ",\"weight\":" << weight_bound
           << ",\"hidden_attention\":" << hidden_attention_bound
           << ",\"output_attention\":" << output_attention_bound << ",\"output_bias\":" << bias_bound << "}";
    std::ostringstream registration;
    registration << "{\"registration_kind\":\"ezkl_formal_model_registration_v1\""
                 << ",\"model_schema\":\"" << canonical_model_schema() << "\""
                 << ",\"model_order\":\"" << canonical_model_order() << "\""
                 << ",\"model_order_digest\":\"" << canonical_model_order_digest() << "\""
                 << ",\"model_arch_id\":\"" << json_escape(checkpoint.model_arch_id) << "\""
                 << ",\"model_param_id\":\"" << json_escape(checkpoint.model_param_id) << "\""
                 << ",\"quant_cfg_id\":\"" << quant_id << "\""
                 << ",\"canonical_parameter_count\":0,\"canonical_parameter_digest\":\"\""
                 << ",\"parameter_bounds\":" << bounds.str()
                 << ",\"model_bounds_digest\":\"\",\"registered_model_hash\":\"\"}\n";
    write_text_file(output_dir / "model_registration.json", registration.str());
    std::ostringstream ledger;
    ledger << "{\"ledger_kind\":\"ezkl_integer_bound_ledger_v2\""
           << ",\"field_modulus\":\"bn254_fr\",\"max_in_degree\":" << max_in_degree
           << ",\"model_bounds_digest\":\"\",\"families\":[";
    for (std::size_t index = 0; index < families.size(); ++index) {
        if (index) ledger << ",";
        ledger << family_json(families[index]);
    }
    ledger << "],\"output_head_bound\":" << decimal(maximum_output_head)
           << ",\"output_sum_bound\":" << decimal(output_sum_bound)
           << ",\"final_average\":" << round_json(final_average)
           << ",\"no_wrap_verified\":true}\n";
    write_text_file(output_dir / "integer_bound_ledger.json", ledger.str());

    std::ostringstream forward;
    forward << "{\"forward_kind\":\"canonical_integer_full_gat_v1\",\"complete\":true"
            << ",\"attention_bias_used\":false,\"rounding_mode\":\"nearest_ties_away\""
            << ",\"logits\":[";
    for (std::size_t index = 0; index < final_logits.size(); ++index) {
        if (index) forward << ",";
        forward << final_logits[index];
    }
    forward << "]}\n";
    write_text_file(output_dir / "canonical_forward.json", forward.str());
    return FormalProfileIds{quant_id, table_id, aux_table_id};
}

}  // namespace gat_ezkl
