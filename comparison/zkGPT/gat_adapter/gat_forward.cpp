#include "gat_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace zkgpt {
namespace gat {
namespace {

std::once_flag field_once;

void ensure_field() {
    std::call_once(field_once, []() { mcl::bn::initPairing(mcl::BN254); });
}

Int checked_int(Wide value, const char* what) {
    if (value > static_cast<Wide>(std::numeric_limits<Int>::max()) || value < static_cast<Wide>(std::numeric_limits<Int>::min())) {
        throw std::overflow_error(std::string("GAT integer overflow: ") + what);
    }
    return static_cast<Int>(value);
}

Wide abs_wide(Int value) { return value < 0 ? -static_cast<Wide>(value) : static_cast<Wide>(value); }

const char* bn254_modulus_string() { return "16798108731015832284940804142231733909759579603404752749028378864165570215949"; }

const char* bn254_half_modulus_string() { return "8399054365507916142470402071115866954879789801702376374514189432082785107974"; }

std::string wide_decimal(Wide value) {
    if (value == 0) return "0";
    const bool negative = value < 0;
    Wide magnitude = negative ? -value : value;
    std::string digits;
    while (magnitude != 0) {
        digits.push_back(static_cast<char>('0' + magnitude % 10));
        magnitude /= 10;
    }
    if (negative) digits.push_back('-');
    std::reverse(digits.begin(), digits.end());
    return digits;
}

bool decimal_nonnegative_leq(Wide value, const char* limit) {
    if (value < 0) return false;
    const std::string lhs = wide_decimal(value);
    const std::string rhs(limit);
    if (lhs.size() != rhs.size()) return lhs.size() < rhs.size();
    return lhs <= rhs;
}

bool actual_bn254_field() {
    ensure_field();
    return Field::getModulo() == bn254_modulus_string();
}

Wide checked_wide_product(Wide left, Wide right, const char* what) {
    if (left < 0 || right < 0) throw std::invalid_argument(std::string("negative bound in ") + what);
    const Wide max_wide = (static_cast<Wide>(1) << 126) - 1;
    if (left != 0 && right > max_wide / left) throw std::overflow_error(std::string("bound overflow in ") + what);
    return left * right;
}

Int quantize_model_value(double value, Int scale) {
    if (scale <= 0 || !std::isfinite(value)) throw std::invalid_argument("invalid model quantization input");
    const long double scaled = static_cast<long double>(value) * static_cast<long double>(scale);
    const long double rounded = scaled >= 0 ? std::floor(scaled + 0.5L) : std::ceil(scaled - 0.5L);
    if (rounded > static_cast<long double>(std::numeric_limits<Int>::max()) || rounded < static_cast<long double>(std::numeric_limits<Int>::min())) {
        throw std::overflow_error("model quantization overflow");
    }
    return static_cast<Int>(rounded);
}

Int tensor_value(const Tensor& tensor, std::size_t row, std::size_t col, Int scale) {
    if (tensor.shape.dims.size() != 3 || tensor.shape.dims[0] != 1 || row >= tensor.shape.dims[1] || col >= tensor.shape.dims[2]) {
        throw std::invalid_argument("tensor shape does not match GAT access");
    }
    return quantize_model_value(tensor.values[row * tensor.shape.dims[2] + col], scale);
}

Int vector_tensor_value(const Tensor& tensor, std::size_t index, Int scale) {
    if (tensor.shape.dims.size() != 3 || tensor.shape.dims[0] != 1 || tensor.shape.dims[2] != 1 || index >= tensor.shape.dims[1]) {
        throw std::invalid_argument("attention tensor shape does not match GAT access");
    }
    return quantize_model_value(tensor.values[index], scale);
}

std::string field_decimal(const Field& value) { return value.getStr(10); }

void append_table_digest(std::ostringstream& out, const StaticTables& tables) {
    out << tables.table_id << '\n'
        << tables.generator_id << '\n'
        << tables.quant_cfg_id << '\n'
        << tables.rounding << '\n'
        << tables.rescale << '\n'
        << tables.output_average << '\n'
        << tables.row_order << '\n'
        << tables.exp_approx_id << '\n'
        << tables.elu_approx_id << '\n'
        << tables.lrelu_slope_num << ':' << tables.lrelu_slope_den << '\n'
        << tables.s_delta << ':' << tables.s_exp << ':' << tables.s_elu_in << ':' << tables.s_elu_out << '\n'
        << tables.lrelu_min << ':' << tables.lrelu_max << ':' << tables.exp_min << ':' << tables.exp_max << ':' << tables.elu_min << ':' << tables.elu_max
        << '\n';
    out << "lrelu\n";
    for (const TablePoint& point : tables.lrelu) out << point.x << ',' << point.y << '\n';
    out << "exp\n";
    for (const TablePoint& point : tables.exp) out << point.x << ',' << point.y << '\n';
    out << "elu\n";
    for (const TablePoint& point : tables.elu) out << point.x << ',' << point.y << '\n';
}

std::string digest_tables(const StaticTables& tables) {
    ensure_field();
    std::ostringstream encoded;
    append_table_digest(encoded, tables);
    Field hash;
    const std::string bytes = encoded.str();
    hash.setHashOf(bytes.data(), bytes.size());
    return field_decimal(hash);
}

Int table_lrelu_value(Int x, Int slope_num, Int slope_den) {
    if (x >= 0) return x;
    return signed_round_div(checked_int(static_cast<Wide>(x) * slope_num, "LeakyReLU numerator"), slope_den, RoundingMode::NEAREST_TIES_AWAY).value;
}

Int table_exp_value(Int delta, Int s_delta, Int s_exp) {
    if (delta < 0 || s_delta <= 0 || s_exp <= 0) throw std::invalid_argument("invalid ExpMap table parameters");
    const long double exponent = -static_cast<long double>(delta) / static_cast<long double>(s_delta);
    const long double value = static_cast<long double>(s_exp) * std::exp(exponent);
    const long double rounded = std::floor(value + 0.5L);
    if (rounded > static_cast<long double>(std::numeric_limits<Int>::max())) throw std::overflow_error("ExpMap table overflow");
    return static_cast<Int>(rounded);
}

Int table_elu_value(Int x, Int s_in, Int s_out) {
    if (s_in <= 0 || s_out <= 0) throw std::invalid_argument("invalid ELU table parameters");
    if (x >= 0) return checked_int(static_cast<Wide>(x) * s_out / s_in, "ELU positive branch");
    const long double value = static_cast<long double>(s_out) * (std::exp(static_cast<long double>(x) / s_in) - 1.0L);
    const long double rounded = std::ceil(value - 0.5L);
    if (rounded > static_cast<long double>(std::numeric_limits<Int>::max()) || rounded < static_cast<long double>(std::numeric_limits<Int>::min())) {
        throw std::overflow_error("ELU table overflow");
    }
    return static_cast<Int>(rounded);
}

std::size_t checked_size(Wide value, const char* what) {
    if (value < 0 || value > static_cast<Wide>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(std::string("GAT size overflow: ") + what);
    }
    return static_cast<std::size_t>(value);
}

void validate_tensor2d(const Tensor2D& tensor, std::size_t rows, std::size_t cols, const char* what) {
    if (tensor.rows != rows || tensor.cols != cols || tensor.values.size() != rows * cols) {
        throw std::invalid_argument(std::string("invalid ") + what + " shape");
    }
}

std::vector<std::pair<std::size_t, std::size_t>> incoming_ranges(const Graph& graph) {
    std::vector<std::pair<std::size_t, std::size_t>> ranges(graph.num_nodes, std::make_pair(graph.edges(), graph.edges()));
    for (std::size_t edge = 0; edge < graph.edges(); ++edge) {
        const std::size_t node = static_cast<std::size_t>(graph.dst[edge]);
        if (ranges[node].first == graph.edges()) ranges[node].first = edge;
        ranges[node].second = edge + 1;
    }
    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        if (ranges[node].first == graph.edges()) throw std::invalid_argument("GAT node has no incoming edge");
    }
    return ranges;
}

struct AttentionScales {
    Int projection = 0;
    Int attention_src = 0;
    Int attention_dst = 0;
    Int aggregate_pre = 0;
    Int aggregate_output = 0;
};

std::string hidden_family(std::size_t layer, std::size_t head) {
    std::ostringstream out;
    out << "hidden/layer" << layer << "/head" << head;
    return out.str();
}

std::string output_family(std::size_t head) {
    std::ostringstream out;
    out << "output/head" << head;
    return out.str();
}

void set_field_aggregation(AttentionTrace& trace, const Graph& graph, std::size_t output_dim) {
    const std::size_t edges = graph.edges();
    trace.field_numerator.assign(trace.integer_numerator.size(), Field(0));
    trace.field_aggregate.assign(trace.integer_numerator.size(), Field(0));
    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        const Int sum = trace.group_sum[node];
        if (sum <= 0) throw std::invalid_argument("GAT softmax denominator is not positive");
        const Field denominator = Field(sum);
        for (std::size_t col = 0; col < output_dim; ++col) {
            const std::size_t index = node * output_dim + col;
            trace.field_numerator[index] = Field(trace.integer_numerator[index]);
            trace.field_aggregate[index] = trace.field_numerator[index] / denominator;
        }
    }
    (void)edges;
}

AttentionTrace compute_attention(const Graph& graph, const Tensor2D& input, const Model& model, const QuantConfig& quant, const StaticTables& tables,
                                 std::size_t layer, std::size_t head, bool output_head, const AttentionScales& scales) {
    const std::size_t input_dim = input.cols;
    const Tensor* seq = nullptr;
    const Tensor* attn_src = nullptr;
    const Tensor* attn_dst = nullptr;
    if (output_head) {
        seq = &model.tensor(output_family(head) + "/seq/kernel");
        attn_src = &model.tensor(output_family(head) + "/attn_src/kernel");
        attn_dst = &model.tensor(output_family(head) + "/attn_dst/kernel");
    } else {
        seq = &model.tensor(hidden_family(layer, head) + "/seq/kernel");
        attn_src = &model.tensor(hidden_family(layer, head) + "/attn_src/kernel");
        attn_dst = &model.tensor(hidden_family(layer, head) + "/attn_dst/kernel");
    }
    if (seq->shape.dims.size() != 3 || seq->shape.dims[0] != 1 || seq->shape.dims[1] != input_dim || seq->shape.dims[2] == 0) {
        throw std::invalid_argument("GAT projection shape mismatch");
    }
    const std::size_t output_dim = seq->shape.dims[2];
    if (attn_src->shape.dims.size() != 3 || attn_dst->shape.dims.size() != 3 || attn_src->shape.dims[1] != output_dim ||
        attn_dst->shape.dims[1] != output_dim || attn_src->shape.dims[2] != 1 || attn_dst->shape.dims[2] != 1) {
        throw std::invalid_argument("GAT attention vector shape mismatch");
    }

    AttentionTrace trace;
    trace.family_id = output_head ? output_family(head) : hidden_family(layer, head);
    trace.input_dim = input_dim;
    trace.output_dim = output_dim;
    trace.projection.assign(graph.num_nodes * output_dim, 0);
    trace.source_score.assign(graph.num_nodes, 0);
    trace.destination_score.assign(graph.num_nodes, 0);
    trace.raw_score.assign(graph.edges(), 0);
    trace.leaky_score.assign(graph.edges(), 0);
    trace.group_max.assign(graph.num_nodes, std::numeric_limits<Int>::min());
    trace.delta.assign(graph.edges(), 0);
    trace.exp.assign(graph.edges(), 0);
    trace.group_sum.assign(graph.num_nodes, 0);
    trace.group_inverse.assign(graph.num_nodes, Field(0));
    trace.max_selector.assign(graph.edges(), 0);
    trace.alpha.assign(graph.edges(), Field(0));
    trace.integer_numerator.assign(graph.num_nodes * output_dim, 0);
    trace.recovered.assign(graph.num_nodes * output_dim, 0);
    trace.relation_ok = true;
    const std::vector<std::pair<std::size_t, std::size_t>> ranges = incoming_ranges(graph);

    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        for (std::size_t col = 0; col < output_dim; ++col) {
            Wide value = 0;
            for (std::size_t inner = 0; inner < input_dim; ++inner) {
                const Int weight = tensor_value(*seq, inner, col, output_head ? quant.s_w_output[head] : quant.s_w_hidden[layer][head]);
                value += static_cast<Wide>(input.at(node, inner)) * weight;
            }
            trace.projection[node * output_dim + col] = checked_int(value, "GAT projection");
        }
        Wide source = 0;
        Wide destination = 0;
        for (std::size_t col = 0; col < output_dim; ++col) {
            const Int projected = trace.projection[node * output_dim + col];
            const Int src_weight = vector_tensor_value(*attn_src, col, output_head ? quant.s_a_src_output[head] : quant.s_a_src_hidden[layer][head]);
            const Int dst_weight = vector_tensor_value(*attn_dst, col, output_head ? quant.s_a_dst_output[head] : quant.s_a_dst_hidden[layer][head]);
            source += static_cast<Wide>(projected) * src_weight;
            destination += static_cast<Wide>(projected) * dst_weight;
        }
        trace.source_score[node] = checked_int(source, "GAT source score");
        trace.destination_score[node] = checked_int(destination, "GAT destination score");
    }

    for (std::size_t edge = 0; edge < graph.edges(); ++edge) {
        const std::size_t source = static_cast<std::size_t>(graph.src[edge]);
        const std::size_t destination = static_cast<std::size_t>(graph.dst[edge]);
        const Int score = checked_int(static_cast<Wide>(trace.source_score[source]) + trace.destination_score[destination], "GAT raw score");
        trace.raw_score[edge] = score;
        trace.leaky_score[edge] = tables.lookup_lrelu(score);
        trace.group_max[destination] = std::max(trace.group_max[destination], trace.leaky_score[edge]);
    }
    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        std::size_t chosen = ranges[node].first;
        for (std::size_t edge = ranges[node].first; edge < ranges[node].second; ++edge) {
            if (trace.leaky_score[edge] > trace.leaky_score[chosen]) chosen = edge;
        }
        trace.max_selector[chosen] = 1;
        for (std::size_t edge = ranges[node].first; edge < ranges[node].second; ++edge) {
            const Int delta = checked_int(static_cast<Wide>(trace.group_max[node]) - trace.leaky_score[edge], "GAT softmax delta");
            trace.delta[edge] = delta;
            trace.exp[edge] = tables.lookup_exp(delta);
            trace.group_sum[node] = checked_int(static_cast<Wide>(trace.group_sum[node]) + trace.exp[edge], "GAT softmax denominator");
        }
        if (trace.group_sum[node] <= 0) throw std::invalid_argument("GAT softmax denominator is not positive");
        ensure_field();
        trace.group_inverse[node] = Field(1) / Field(trace.group_sum[node]);
    }
    for (std::size_t edge = 0; edge < graph.edges(); ++edge) {
        const std::size_t destination = static_cast<std::size_t>(graph.dst[edge]);
        trace.alpha[edge] = Field(trace.exp[edge]) * trace.group_inverse[destination];
        if (!(trace.alpha[edge] * Field(trace.group_sum[destination]) - Field(trace.exp[edge])).isZero()) { trace.relation_ok = false; }
    }

    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        for (std::size_t col = 0; col < output_dim; ++col) {
            Wide numerator = 0;
            for (std::size_t edge = ranges[node].first; edge < ranges[node].second; ++edge) {
                const std::size_t source = static_cast<std::size_t>(graph.src[edge]);
                numerator += static_cast<Wide>(trace.exp[edge]) * trace.projection[source * output_dim + col];
            }
            const std::size_t index = node * output_dim + col;
            trace.integer_numerator[index] = checked_int(numerator, "GAT integer aggregation numerator");
            const Wide dividend = static_cast<Wide>(scales.aggregate_pre) * trace.integer_numerator[index];
            const Wide denominator = static_cast<Wide>(scales.projection) * trace.group_sum[node];
            const Int dividend_i = checked_int(dividend, "GAT aggregation dividend");
            const Int denominator_i = checked_int(denominator, "GAT aggregation denominator");
            const DivisionResult recovered = signed_round_div(dividend_i, denominator_i);
            trace.divisions.push_back(recovered.witness);
            trace.recovered[index] = recovered.value;
        }
    }
    set_field_aggregation(trace, graph, output_dim);
    trace.activated.assign(trace.recovered.size(), 0);
    if (!output_head) {
        for (std::size_t i = 0; i < trace.recovered.size(); ++i) trace.activated[i] = tables.lookup_elu(trace.recovered[i]);
    } else trace.activated = trace.recovered;

    trace.field_integer_consistent = true;
    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        const Field inverse_sum = Field(1) / Field(trace.group_sum[node]);
        if (!(Field(trace.group_sum[node]) * trace.group_inverse[node] - Field(1)).isZero()) { trace.field_integer_consistent = false; }
        for (std::size_t col = 0; col < output_dim; ++col) {
            const std::size_t index = node * output_dim + col;
            const Field expected = Field(trace.integer_numerator[index]) * inverse_sum;
            if (!(trace.field_numerator[index] - Field(trace.integer_numerator[index])).isZero() || !(trace.field_aggregate[index] - expected).isZero() ||
                !(trace.field_aggregate[index] * Field(trace.group_sum[node]) - trace.field_numerator[index]).isZero()) {
                trace.field_integer_consistent = false;
            }
        }
    }
    trace.relation_ok = trace.relation_ok && trace.field_integer_consistent;
    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        std::size_t selected = 0;
        for (std::size_t edge = ranges[node].first; edge < ranges[node].second; ++edge) {
            selected += static_cast<std::size_t>(trace.max_selector[edge]);
            if (trace.max_selector[edge] != 0 && trace.delta[edge] != 0) trace.relation_ok = false;
        }
        if (selected != 1) trace.relation_ok = false;
    }
    (void)scales.attention_src;
    (void)scales.attention_dst;
    (void)scales.aggregate_output;
    return trace;
}

Wide max_abs_values(const std::vector<Int>& values) {
    Wide maximum = 0;
    for (Int value : values) maximum = std::max(maximum, abs_wide(value));
    return maximum;
}

void add_tensor_bound(const Tensor& tensor, Int scale, Wide& maximum) {
    for (double value : tensor.values) { maximum = std::max(maximum, abs_wide(quantize_model_value(value, scale))); }
}

void family_indices(const std::string& id, bool& output, std::size_t& layer, std::size_t& head) {
    output = false;
    layer = head = 0;
    if (id.compare(0, std::string("hidden/layer").size(), "hidden/layer") == 0) {
        const std::size_t layer_start = std::string("hidden/layer").size();
        const std::size_t head_marker = id.find("/head", layer_start);
        if (head_marker == std::string::npos) throw std::invalid_argument("malformed hidden bound family id");
        layer = static_cast<std::size_t>(std::stoul(id.substr(layer_start, head_marker - layer_start)));
        head = static_cast<std::size_t>(std::stoul(id.substr(head_marker + 5)));
    } else if (id.compare(0, std::string("output/head").size(), "output/head") == 0) {
        output = true;
        head = static_cast<std::size_t>(std::stoul(id.substr(std::string("output/head").size())));
    } else {
        throw std::invalid_argument("unknown bound family id: " + id);
    }
    if ((!output && (layer >= 2 || head >= 8)) || (output && head >= 2)) { throw std::invalid_argument("bound family index is outside the canonical model"); }
}

BoundFamily make_bound_family(const std::string& id, const Graph& graph, Wide input_abs, const Model& model, const QuantConfig& quant, const StaticTables& tables, const AttentionTrace& trace, const AttentionScales& scales) {
    BoundFamily family;
    family.id = id;
    family.input_dim = static_cast<Wide>(trace.input_dim);
    family.output_dim = static_cast<Wide>(trace.output_dim);
    family.max_in_degree = 0;
    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        Wide degree = 0;
        for (std::size_t edge = 0; edge < graph.edges(); ++edge)
            if (static_cast<std::size_t>(graph.dst[edge]) == node) ++degree;
        family.max_in_degree = std::max(family.max_in_degree, degree);
    }
    bool output = false;
    std::size_t layer = 0, head = 0;
    family_indices(id, output, layer, head);
    Wide model_abs = 0;
    Wide weight_abs = 0;
    Wide attention_abs = 0;
    if (output) {
        add_tensor_bound(model.tensor(output_family(head) + "/seq/kernel"), quant.s_w_output[head], weight_abs);
        add_tensor_bound(model.tensor(output_family(head) + "/attn_src/kernel"), quant.s_a_src_output[head], attention_abs);
        add_tensor_bound(model.tensor(output_family(head) + "/attn_dst/kernel"), quant.s_a_dst_output[head], attention_abs);
        add_tensor_bound(model.tensor(output_family(head) + "/output_bias"), quant.s_b_output[head], model_abs);
    } else {
        add_tensor_bound(model.tensor(hidden_family(layer, head) + "/seq/kernel"), quant.s_w_hidden[layer][head], weight_abs);
        add_tensor_bound(model.tensor(hidden_family(layer, head) + "/attn_src/kernel"), quant.s_a_src_hidden[layer][head], attention_abs);
        add_tensor_bound(model.tensor(hidden_family(layer, head) + "/attn_dst/kernel"), quant.s_a_dst_hidden[layer][head], attention_abs);
    }
    model_abs = std::max(model_abs, std::max(weight_abs, attention_abs));
    for (std::size_t i = 0; i < trace.projection.size(); ++i) { family.projection_abs = std::max(family.projection_abs, abs_wide(trace.projection[i])); }
    weight_abs = std::max<Wide>(1, model_abs);
    attention_abs = weight_abs;
    family.model_abs = model_abs;
    family.input_abs = input_abs;
    family.weight_abs = weight_abs;
    family.attention_abs = attention_abs;
    family.source_score_abs = std::max(max_abs_values(trace.source_score), max_abs_values(trace.destination_score));
    family.destination_score_abs = max_abs_values(trace.destination_score);
    family.score_abs = std::max(max_abs_values(trace.raw_score), max_abs_values(trace.leaky_score));
    family.table_input_abs =
        std::max(std::max(abs_wide(tables.lrelu_min), abs_wide(tables.lrelu_max)), std::max(abs_wide(tables.exp_min), abs_wide(tables.exp_max)));
    family.table_output_abs = 0;
    for (const TablePoint& point : tables.lrelu) family.table_output_abs = std::max(family.table_output_abs, abs_wide(point.y));
    for (const TablePoint& point : tables.exp) family.table_output_abs = std::max(family.table_output_abs, abs_wide(point.y));
    for (const TablePoint& point : tables.elu) family.table_output_abs = std::max(family.table_output_abs, abs_wide(point.y));
    family.delta_abs = max_abs_values(trace.delta);
    family.exp_abs = max_abs_values(trace.exp);
    family.sum_abs = max_abs_values(trace.group_sum);
    family.numerator_abs = max_abs_values(trace.integer_numerator);
    family.division_x_abs = 0;
    family.division_d_abs = 0;
    family.q_abs = family.r_abs = family.delta_aux_abs = 0;
    family.elu_input_abs = max_abs_values(trace.recovered);
    family.product_abs = 0;
    for (const DivisionWitness& division : trace.divisions) {
        family.division_x_abs = std::max(family.division_x_abs, abs_wide(division.dividend));
        family.division_d_abs = std::max(family.division_d_abs, abs_wide(division.denominator));
        family.q_abs = std::max(family.q_abs, abs_wide(division.q));
        family.r_abs = std::max(family.r_abs, abs_wide(division.r));
        family.delta_aux_abs = std::max(family.delta_aux_abs, abs_wide(division.delta));
    }
    for (std::size_t edge = 0; edge < graph.edges(); ++edge) {
        const std::size_t source = static_cast<std::size_t>(graph.src[edge]);
        for (std::size_t col = 0; col < trace.output_dim; ++col) {
            family.product_abs =
                std::max(family.product_abs,
                         checked_wide_product(abs_wide(trace.exp[edge]), abs_wide(trace.projection[source * trace.output_dim + col]), "attention product"));
        }
    }
    family.packed_tuple_abs = checked_wide_product(family.table_input_abs, 2 * family.table_output_abs + 1, "packed lookup tuple");
    family.field_modulus = Field::getModulo();
    family.field_half_modulus = bn254_half_modulus_string();
    family.inverse_field_valid = actual_bn254_field();
    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        if (trace.group_sum[node] <= 0 || !(Field(trace.group_sum[node]) * trace.group_inverse[node] - Field(1)).isZero()) {
            family.inverse_field_valid = false;
        }
    }
    family.no_wrap = true;
    const Wide* bounds[] = {&family.model_abs,
                            &family.input_abs,
                            &family.weight_abs,
                            &family.attention_abs,
                            &family.score_abs,
                            &family.table_input_abs,
                            &family.table_output_abs,
                            &family.delta_abs,
                            &family.exp_abs,
                            &family.sum_abs,
                            &family.numerator_abs,
                            &family.division_x_abs,
                            &family.division_d_abs,
                            &family.packed_tuple_abs,
                            &family.q_abs,
                            &family.r_abs,
                            &family.delta_aux_abs,
                            &family.product_abs,
                            &family.projection_abs,
                            &family.source_score_abs,
                            &family.destination_score_abs,
                            &family.elu_input_abs};
    for (const Wide* bound : bounds)
        if (!decimal_nonnegative_leq(*bound, bn254_half_modulus_string())) family.no_wrap = false;
    (void)scales;
    return family;
}

void compare_division(const DivisionWitness& witness, Int value, RoundingMode mode, bool& ok) {
    if (!witness.valid(mode)) ok = false;
    const DivisionResult expected = signed_round_div(witness.dividend, witness.denominator, mode);
    if (expected.value != value) ok = false;
}

bool validate_attention(const Graph& graph, const AttentionTrace& trace, bool output_head) {
    if (trace.projection.size() != graph.num_nodes * trace.output_dim || trace.source_score.size() != graph.num_nodes ||
        trace.destination_score.size() != graph.num_nodes || trace.raw_score.size() != graph.edges() || trace.leaky_score.size() != graph.edges() ||
        trace.group_max.size() != graph.num_nodes || trace.delta.size() != graph.edges() || trace.exp.size() != graph.edges() ||
        trace.group_sum.size() != graph.num_nodes || trace.group_inverse.size() != graph.num_nodes || trace.max_selector.size() != graph.edges() ||
        trace.alpha.size() != graph.edges() || trace.integer_numerator.size() != graph.num_nodes * trace.output_dim ||
        trace.field_numerator.size() != trace.integer_numerator.size() || trace.field_aggregate.size() != trace.integer_numerator.size() ||
        trace.recovered.size() != trace.integer_numerator.size() || trace.activated.size() != trace.integer_numerator.size() ||
        trace.divisions.size() != trace.integer_numerator.size() || !trace.relation_ok)
        return false;
    if (output_head && trace.activated != trace.recovered) return false;
    const std::vector<std::pair<std::size_t, std::size_t>> ranges = incoming_ranges(graph);
    bool ok = trace.field_integer_consistent;
    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        if (trace.group_sum[node] <= 0) ok = false;
        std::size_t selected = 0;
        for (std::size_t edge = ranges[node].first; edge < ranges[node].second; ++edge) {
            if (trace.max_selector[edge] != 0 && trace.delta[edge] != 0) ok = false;
            selected += static_cast<std::size_t>(trace.max_selector[edge]);
        }
        if (selected != 1) ok = false;
        for (std::size_t col = 0; col < trace.output_dim; ++col) {
            const std::size_t index = node * trace.output_dim + col;
            if (!(trace.field_numerator[index] - Field(trace.integer_numerator[index])).isZero()) ok = false;
            if (!(trace.field_aggregate[index] * Field(trace.group_sum[node]) - Field(trace.integer_numerator[index])).isZero()) ok = false;
            compare_division(trace.divisions[index], trace.recovered[index], RoundingMode::NEAREST_TIES_AWAY, ok);
        }
    }
    return ok;
}

bool equal_field(const Field& left, const Field& right) { return (left - right).isZero(); }

bool equal_field_vector(const std::vector<Field>& left, const std::vector<Field>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t i = 0; i < left.size(); ++i)
        if (!equal_field(left[i], right[i])) return false;
    return true;
}

bool equal_division(const DivisionWitness& left, const DivisionWitness& right) {
    return left.sigma == right.sigma && left.z == right.z && left.a == right.a && left.q == right.q && left.r == right.r && left.delta == right.delta &&
           left.dividend == right.dividend && left.denominator == right.denominator && equal_field(left.a_inverse, right.a_inverse) &&
           equal_field(left.d_inverse, right.d_inverse);
}

bool equal_divisions(const std::vector<DivisionWitness>& left, const std::vector<DivisionWitness>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t i = 0; i < left.size(); ++i)
        if (!equal_division(left[i], right[i])) return false;
    return true;
}

bool equal_attention(const AttentionTrace& actual, const AttentionTrace& expected) {
    return actual.family_id == expected.family_id && actual.input_dim == expected.input_dim && actual.output_dim == expected.output_dim &&
           actual.projection == expected.projection && actual.source_score == expected.source_score && actual.destination_score == expected.destination_score &&
           actual.raw_score == expected.raw_score && actual.leaky_score == expected.leaky_score && actual.group_max == expected.group_max &&
           actual.delta == expected.delta && actual.exp == expected.exp && actual.group_sum == expected.group_sum &&
           equal_field_vector(actual.group_inverse, expected.group_inverse) && actual.max_selector == expected.max_selector &&
           equal_field_vector(actual.alpha, expected.alpha) && actual.integer_numerator == expected.integer_numerator &&
           equal_field_vector(actual.field_numerator, expected.field_numerator) && equal_field_vector(actual.field_aggregate, expected.field_aggregate) &&
           actual.recovered == expected.recovered && actual.activated == expected.activated && equal_divisions(actual.divisions, expected.divisions) &&
           actual.field_integer_consistent == expected.field_integer_consistent && actual.relation_ok == expected.relation_ok;
}

bool equal_ledger(const IntegerBoundLedger& actual, const IntegerBoundLedger& expected) {
    if (actual.backend != expected.backend || actual.table_id != expected.table_id || actual.field_modulus != expected.field_modulus ||
        actual.field_half_modulus != expected.field_half_modulus || actual.lrelu_min != expected.lrelu_min || actual.lrelu_max != expected.lrelu_max ||
        actual.exp_min != expected.exp_min || actual.exp_max != expected.exp_max || actual.elu_min != expected.elu_min || actual.elu_max != expected.elu_max ||
        actual.families.size() != expected.families.size())
        return false;
    const auto equal_family = [](const BoundFamily& left, const BoundFamily& right) {
        return left.id == right.id && left.model_abs == right.model_abs && left.input_abs == right.input_abs && left.weight_abs == right.weight_abs &&
               left.attention_abs == right.attention_abs && left.score_abs == right.score_abs && left.table_input_abs == right.table_input_abs &&
               left.table_output_abs == right.table_output_abs && left.delta_abs == right.delta_abs && left.exp_abs == right.exp_abs &&
               left.sum_abs == right.sum_abs && left.numerator_abs == right.numerator_abs && left.division_x_abs == right.division_x_abs &&
               left.division_d_abs == right.division_d_abs && left.packed_tuple_abs == right.packed_tuple_abs && left.q_abs == right.q_abs &&
               left.r_abs == right.r_abs && left.delta_aux_abs == right.delta_aux_abs && left.product_abs == right.product_abs &&
               left.input_dim == right.input_dim && left.output_dim == right.output_dim && left.max_in_degree == right.max_in_degree &&
               left.projection_abs == right.projection_abs && left.source_score_abs == right.source_score_abs &&
               left.destination_score_abs == right.destination_score_abs && left.alignment_x_abs == right.alignment_x_abs &&
               left.average_x_abs == right.average_x_abs && left.elu_input_abs == right.elu_input_abs && left.field_modulus == right.field_modulus &&
               left.field_half_modulus == right.field_half_modulus && left.inverse_field_valid == right.inverse_field_valid && left.no_wrap == right.no_wrap;
    };
    for (std::size_t i = 0; i < actual.families.size(); ++i)
        if (!equal_family(actual.families[i], expected.families[i])) return false;
    return true;
}

std::string admission_ledger_mismatch(const IntegerBoundLedger& actual, const IntegerBoundLedger& expected) {
    if (actual.families.size() != expected.families.size()) return "family_count";
    for (std::size_t index = 0; index < actual.families.size(); ++index) {
        const BoundFamily& a = actual.families[index];
        const BoundFamily& b = expected.families[index];
        if (a.id != b.id) return "family_id";
#define ZKGPT_ADMISSION_COMPARE_FIELD(name) if (a.name != b.name) return std::string(a.id) + ":" #name
        ZKGPT_ADMISSION_COMPARE_FIELD(model_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(input_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(weight_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(attention_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(score_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(table_input_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(table_output_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(delta_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(exp_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(sum_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(numerator_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(division_x_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(division_d_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(packed_tuple_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(q_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(r_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(delta_aux_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(product_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(input_dim);
        ZKGPT_ADMISSION_COMPARE_FIELD(output_dim);
        ZKGPT_ADMISSION_COMPARE_FIELD(max_in_degree);
        ZKGPT_ADMISSION_COMPARE_FIELD(projection_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(source_score_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(destination_score_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(alignment_x_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(average_x_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(elu_input_abs);
        ZKGPT_ADMISSION_COMPARE_FIELD(field_modulus);
        ZKGPT_ADMISSION_COMPARE_FIELD(field_half_modulus);
        ZKGPT_ADMISSION_COMPARE_FIELD(inverse_field_valid);
        ZKGPT_ADMISSION_COMPARE_FIELD(no_wrap);
#undef ZKGPT_ADMISSION_COMPARE_FIELD
    }
    return "unknown";
}

unsigned admission_bit_width(Wide bound) {
    if (bound < 0) throw std::invalid_argument("negative authenticated admission range bound");
    unsigned bits = 1;
    while (bits < 63 && bound >= (static_cast<Wide>(1) << bits)) ++bits;
    return bits;
}

void require_admission_range_value(Wide value, unsigned bits, const char* family) {
    if (bits == 0 || bits > 63 || value < 0 || value > static_cast<Wide>(std::numeric_limits<Int>::max()) ||
        value >= (static_cast<Wide>(1) << bits))
        throw std::out_of_range(std::string("canonical ") + family + " range query exceeds authenticated bit domain");
}

std::pair<Wide, Wide> admission_table_abs_bounds(const std::vector<TablePoint>& points, const char* family) {
    if (points.empty()) throw std::invalid_argument(std::string("authenticated ") + family + " table is empty");
    Wide input = 0, output = 0;
    for (const TablePoint& point : points) {
        input = std::max(input, abs_wide(point.x));
        output = std::max(output, abs_wide(point.y));
    }
    return {input, output};
}

struct AdmissionTableAbsBounds {
    std::pair<Wide, Wide> lrelu;
    std::pair<Wide, Wide> exp;
    std::pair<Wide, Wide> elu;
};

Wide admission_tensor_abs_bound(const Tensor2D& input) {
    Wide bound = 0;
    for (Int value : input.values) bound = std::max(bound, abs_wide(value));
    return bound;
}

void require_admission_tuple(Int x, Int y, const std::pair<Wide, Wide>& bounds, const char* family) {
    const unsigned x_bits = admission_bit_width(2 * bounds.first);
    const unsigned y_bits = admission_bit_width(2 * bounds.second);
    require_admission_range_value(static_cast<Wide>(x) + bounds.first, x_bits, family);
    require_admission_range_value(bounds.first - static_cast<Wide>(x), x_bits, family);
    require_admission_range_value(static_cast<Wide>(y) + bounds.second, y_bits, family);
    require_admission_range_value(bounds.second - static_cast<Wide>(y), y_bits, family);
}

void require_admission_division(const DivisionWitness& witness, const std::array<unsigned, 5>& profile, const char* family) {
    if (!witness.valid(RoundingMode::NEAREST_TIES_AWAY))
        throw std::invalid_argument(std::string("canonical ") + family + " division witness is invalid");
    const std::array<Int, 5> ranged{{witness.a, witness.q, witness.r, witness.delta, witness.denominator}};
    for (std::size_t index = 0; index < ranged.size(); ++index)
        require_admission_range_value(ranged[index], profile[index], family);
}

BoundFamily admission_family_base(const std::string& id, const Tensor2D& input, const Model& model, const QuantConfig& quant, const StaticTables& tables, std::size_t layer, std::size_t head, bool output_head, std::size_t output_dim, std::size_t maximum_degree, Wide family_input_abs) {
    BoundFamily family;
    family.id = id;
    family.input_dim = static_cast<Wide>(input.cols);
    family.output_dim = static_cast<Wide>(output_dim);
    family.max_in_degree = static_cast<Wide>(maximum_degree);
    Wide model_abs = 0, weight_abs = 0, attention_abs = 0;
    if (output_head) {
        add_tensor_bound(model.tensor(output_family(head) + "/seq/kernel"), quant.s_w_output[head], weight_abs);
        add_tensor_bound(model.tensor(output_family(head) + "/attn_src/kernel"), quant.s_a_src_output[head], attention_abs);
        add_tensor_bound(model.tensor(output_family(head) + "/attn_dst/kernel"), quant.s_a_dst_output[head], attention_abs);
        add_tensor_bound(model.tensor(output_family(head) + "/output_bias"), quant.s_b_output[head], model_abs);
    } else {
        add_tensor_bound(model.tensor(hidden_family(layer, head) + "/seq/kernel"), quant.s_w_hidden[layer][head], weight_abs);
        add_tensor_bound(model.tensor(hidden_family(layer, head) + "/attn_src/kernel"), quant.s_a_src_hidden[layer][head], attention_abs);
        add_tensor_bound(model.tensor(hidden_family(layer, head) + "/attn_dst/kernel"), quant.s_a_dst_hidden[layer][head], attention_abs);
    }
    model_abs = std::max(model_abs, std::max(weight_abs, attention_abs));
    family.model_abs = model_abs;
    family.weight_abs = std::max<Wide>(1, model_abs);
    family.attention_abs = family.weight_abs;
    family.input_abs = family_input_abs;
    family.table_input_abs = std::max({abs_wide(tables.lrelu_min), abs_wide(tables.lrelu_max), abs_wide(tables.exp_min), abs_wide(tables.exp_max)});
    for (const TablePoint& point : tables.lrelu) family.table_output_abs = std::max(family.table_output_abs, abs_wide(point.y));
    for (const TablePoint& point : tables.exp) family.table_output_abs = std::max(family.table_output_abs, abs_wide(point.y));
    for (const TablePoint& point : tables.elu) family.table_output_abs = std::max(family.table_output_abs, abs_wide(point.y));
    family.packed_tuple_abs = checked_wide_product(family.table_input_abs, 2 * family.table_output_abs + 1, "packed lookup tuple");
    family.field_modulus = Field::getModulo();
    family.field_half_modulus = bn254_half_modulus_string();
    family.inverse_field_valid = actual_bn254_field();
    return family;
}

void finalize_admission_family(BoundFamily& family) {
    family.no_wrap = true;
    const Wide* bounds[] = {&family.model_abs, &family.input_abs, &family.weight_abs, &family.attention_abs,
                            &family.score_abs, &family.table_input_abs, &family.table_output_abs, &family.delta_abs,
                            &family.exp_abs, &family.sum_abs, &family.numerator_abs, &family.division_x_abs,
                            &family.division_d_abs, &family.packed_tuple_abs, &family.q_abs, &family.r_abs,
                            &family.delta_aux_abs, &family.product_abs, &family.input_dim, &family.output_dim,
                            &family.max_in_degree, &family.projection_abs, &family.source_score_abs,
                            &family.destination_score_abs, &family.alignment_x_abs, &family.average_x_abs,
                            &family.elu_input_abs};
    for (const Wide* bound : bounds)
        if (!decimal_nonnegative_leq(*bound, bn254_half_modulus_string())) family.no_wrap = false;
}

void observe_admission_division(BoundFamily& family, const DivisionWitness& division) {
    family.division_x_abs = std::max(family.division_x_abs, abs_wide(division.dividend));
    family.division_d_abs = std::max(family.division_d_abs, abs_wide(division.denominator));
    family.q_abs = std::max(family.q_abs, abs_wide(division.q));
    family.r_abs = std::max(family.r_abs, abs_wide(division.r));
    family.delta_aux_abs = std::max(family.delta_aux_abs, abs_wide(division.delta));
}

BoundFamily compute_attention_admission(const Graph& graph, const Tensor2D& input, const Tensor2D& family_input, const Model& model, const QuantConfig& quant, const StaticTables& tables, const std::vector<std::pair<std::size_t, std::size_t>>& ranges, std::size_t maximum_degree, const AdmissionTableAbsBounds& table_bounds, Wide family_input_abs, std::size_t layer, std::size_t head, bool output_head, const AttentionScales& scales, const std::vector<std::array<unsigned, 5>>& division_profiles, std::size_t profile_index, Tensor2D* concatenated, std::size_t concat_offset, std::vector<Int>* output_recovered) {
    const std::size_t input_dim = input.cols;
    const Tensor* seq = nullptr;
    const Tensor* attn_src = nullptr;
    const Tensor* attn_dst = nullptr;
    if (output_head) {
        seq = &model.tensor(output_family(head) + "/seq/kernel");
        attn_src = &model.tensor(output_family(head) + "/attn_src/kernel");
        attn_dst = &model.tensor(output_family(head) + "/attn_dst/kernel");
    } else {
        seq = &model.tensor(hidden_family(layer, head) + "/seq/kernel");
        attn_src = &model.tensor(hidden_family(layer, head) + "/attn_src/kernel");
        attn_dst = &model.tensor(hidden_family(layer, head) + "/attn_dst/kernel");
    }
    if (seq->shape.dims.size() != 3 || seq->shape.dims[0] != 1 || seq->shape.dims[1] != input_dim || seq->shape.dims[2] == 0)
        throw std::invalid_argument("GAT admission projection shape mismatch");
    const std::size_t output_dim = seq->shape.dims[2];
    if (attn_src->shape.dims.size() != 3 || attn_dst->shape.dims.size() != 3 || attn_src->shape.dims[1] != output_dim ||
        attn_dst->shape.dims[1] != output_dim || attn_src->shape.dims[2] != 1 || attn_dst->shape.dims[2] != 1)
        throw std::invalid_argument("GAT admission attention vector shape mismatch");
    if (profile_index >= division_profiles.size()) throw std::invalid_argument("GAT admission division profile is incomplete");
    if (output_head) {
        if (concatenated != nullptr || output_recovered == nullptr) throw std::invalid_argument("GAT admission output sink is malformed");
        output_recovered->assign(graph.num_nodes * output_dim, 0);
    } else {
        if (concatenated == nullptr || output_recovered != nullptr || concatenated->rows != graph.num_nodes ||
            concat_offset + output_dim > concatenated->cols)
            throw std::invalid_argument("GAT admission hidden-layer sink is malformed");
    }

    if (maximum_degree == 0) throw std::invalid_argument("canonical GAT admission has an empty destination group");
    std::vector<Int> projection(graph.num_nodes * output_dim, 0);
    std::vector<Int> source_score(graph.num_nodes, 0);
    std::vector<Int> destination_score(graph.num_nodes, 0);
    std::vector<Int> group_max(graph.num_nodes, std::numeric_limits<Int>::min());
    std::vector<Int> group_sum(graph.num_nodes, 0);
    std::vector<Int> edge_workspace(maximum_degree, 0);
    const std::string id = output_head ? output_family(head) : hidden_family(layer, head);
    BoundFamily family = admission_family_base(id, family_input, model, quant, tables, layer, head, output_head, output_dim, maximum_degree, family_input_abs);
    family.input_dim = static_cast<Wide>(input_dim);
    bool relation_ok = true;
    const auto& profile = division_profiles[profile_index];
    const auto observe_profile = [&](const DivisionWitness& witness, const char* family_name) {
        require_admission_division(witness, profile, family_name);
        observe_admission_division(family, witness);
    };

    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        for (std::size_t col = 0; col < output_dim; ++col) {
            Wide value = 0;
            for (std::size_t inner = 0; inner < input_dim; ++inner) {
                const Int weight = tensor_value(*seq, inner, col, output_head ? quant.s_w_output[head] : quant.s_w_hidden[layer][head]);
                value += static_cast<Wide>(input.at(node, inner)) * weight;
            }
            projection[node * output_dim + col] = checked_int(value, "GAT admission projection");
            family.projection_abs = std::max(family.projection_abs, abs_wide(projection[node * output_dim + col]));
        }
        Wide source = 0, destination = 0;
        for (std::size_t col = 0; col < output_dim; ++col) {
            const Int projected = projection[node * output_dim + col];
            const Int src_weight = vector_tensor_value(*attn_src, col, output_head ? quant.s_a_src_output[head] : quant.s_a_src_hidden[layer][head]);
            const Int dst_weight = vector_tensor_value(*attn_dst, col, output_head ? quant.s_a_dst_output[head] : quant.s_a_dst_hidden[layer][head]);
            source += static_cast<Wide>(projected) * src_weight;
            destination += static_cast<Wide>(projected) * dst_weight;
        }
        source_score[node] = checked_int(source, "GAT admission source score");
        destination_score[node] = checked_int(destination, "GAT admission destination score");
        family.source_score_abs = std::max(family.source_score_abs, abs_wide(source_score[node]));
        family.source_score_abs = std::max(family.source_score_abs, abs_wide(destination_score[node]));
        family.destination_score_abs = std::max(family.destination_score_abs, abs_wide(destination_score[node]));
    }

    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        const auto range = ranges[node];
        std::size_t chosen = 0;
        Int maximum = std::numeric_limits<Int>::min();
        for (std::size_t edge = range.first; edge < range.second; ++edge) {
            const std::size_t local = edge - range.first;
            const std::size_t source = static_cast<std::size_t>(graph.src[edge]);
            const std::size_t destination = static_cast<std::size_t>(graph.dst[edge]);
            const Int raw = checked_int(static_cast<Wide>(source_score[source]) + destination_score[destination], "GAT admission raw score");
            const Int leaky = tables.lookup_lrelu(raw);
            require_admission_tuple(raw, leaky, table_bounds.lrelu, "LeakyReLU");
            family.score_abs = std::max(family.score_abs, abs_wide(raw));
            family.score_abs = std::max(family.score_abs, abs_wide(leaky));
            edge_workspace[local] = leaky;
            if (local == 0 || leaky > maximum) {
                maximum = leaky;
                chosen = local;
            }
        }
        group_max[node] = maximum;
        Int sum = 0;
        for (std::size_t edge = range.first; edge < range.second; ++edge) {
            const std::size_t local = edge - range.first;
            const Int delta = checked_int(static_cast<Wide>(group_max[node]) - edge_workspace[local], "GAT admission softmax delta");
            if (delta < 0) relation_ok = false;
            require_admission_range_value(delta, admission_bit_width(2 * table_bounds.lrelu.second), "softmax delta");
            family.delta_abs = std::max(family.delta_abs, abs_wide(delta));
            const Int exp = tables.lookup_exp(delta);
            require_admission_tuple(delta, exp, table_bounds.exp, "ExpMap");
            family.exp_abs = std::max(family.exp_abs, abs_wide(exp));
            edge_workspace[local] = exp;
            sum = checked_int(static_cast<Wide>(sum) + exp, "GAT admission softmax denominator");
            if (local == chosen && delta != 0) relation_ok = false;
        }
        group_sum[node] = sum;
        family.sum_abs = std::max(family.sum_abs, abs_wide(sum));
        if (sum <= 0) throw std::invalid_argument("GAT admission softmax denominator is not positive");
        ensure_field();
        const Field group_inverse = Field(1) / Field(sum);
        if (!(Field(sum) * group_inverse - Field(1)).isZero()) family.inverse_field_valid = false;
        for (std::size_t edge = range.first; edge < range.second; ++edge) {
            const Field alpha = Field(edge_workspace[edge - range.first]) * group_inverse;
            if (!(alpha * Field(sum) - Field(edge_workspace[edge - range.first])).isZero()) relation_ok = false;
        }

        const Field inverse_sum = Field(1) / Field(sum);
        for (std::size_t col = 0; col < output_dim; ++col) {
            Wide numerator = 0;
            for (std::size_t edge = range.first; edge < range.second; ++edge) {
                const std::size_t source = static_cast<std::size_t>(graph.src[edge]);
                const Int exp = edge_workspace[edge - range.first];
                const Int projected = projection[source * output_dim + col];
                const Wide product = checked_wide_product(abs_wide(exp), abs_wide(projected), "admission attention product");
                family.product_abs = std::max(family.product_abs, product);
                numerator += static_cast<Wide>(exp) * projected;
            }
            const Int integer_numerator = checked_int(numerator, "GAT admission integer aggregation numerator");
            family.numerator_abs = std::max(family.numerator_abs, abs_wide(integer_numerator));
            const Int dividend = checked_int(static_cast<Wide>(scales.aggregate_pre) * integer_numerator, "GAT admission aggregation dividend");
            const Int denominator = checked_int(static_cast<Wide>(scales.projection) * sum, "GAT admission aggregation denominator");
            const DivisionResult recovered = signed_round_div(dividend, denominator);
            observe_profile(recovered.witness, "attention");
            const Field field_numerator(integer_numerator);
            const Field field_aggregate = field_numerator / Field(sum);
            const Field expected = Field(integer_numerator) * inverse_sum;
            if (!(field_numerator - Field(integer_numerator)).isZero() || !(field_aggregate - expected).isZero() ||
                !(field_aggregate * Field(sum) - field_numerator).isZero())
                relation_ok = false;
            const Int activated = output_head ? recovered.value : tables.lookup_elu(recovered.value);
            if (!output_head) require_admission_tuple(recovered.value, activated, table_bounds.elu, "ELU");
            family.elu_input_abs = std::max(family.elu_input_abs, abs_wide(recovered.value));
            if (output_head) (*output_recovered)[node * output_dim + col] = activated;
            else concatenated->at(node, concat_offset + col) = activated;
        }
    }
    finalize_admission_family(family);
    if (!relation_ok) throw std::invalid_argument("canonical GAT admission relation check failed for " + id);
    return family;
}

bool run_forward_admission_impl(const Graph& graph, const Tensor2D& features, const Model& model, const QuantConfig& quant,
                                const std::vector<std::array<unsigned, 5>>& division_profiles,
                                const ForwardResult* retained_reference,
                                ForwardAdmissionProgress progress) {
    ensure_field();
    graph.validate();
    model.validate_schema();
    quant.validate();
    validate_tensor2d(features, graph.num_nodes, graph.num_features, "public features");
    if (features.cols != model.input_dim()) throw std::invalid_argument("GAT feature dimension does not match model");
    const std::size_t expected_profiles = 2 * model.hidden_heads() + 2 * model.manifest.output_heads + 1;
    if (division_profiles.size() != expected_profiles) throw std::invalid_argument("canonical GAT division range profile coverage is incomplete");

    IntegerBoundLedger ledger;
    const StaticTables tables = StaticTables::build(quant);
    if (!tables.authenticate()) throw std::invalid_argument("canonical GAT static table authentication failed");
    const AdmissionTableAbsBounds table_bounds{admission_table_abs_bounds(tables.lrelu, "LeakyReLU"), admission_table_abs_bounds(tables.exp, "ExpMap"), admission_table_abs_bounds(tables.elu, "ELU")};
    const auto ranges = incoming_ranges(graph);
    std::size_t maximum_degree = 0;
    for (const auto& range : ranges) maximum_degree = std::max(maximum_degree, range.second - range.first);
    if (maximum_degree == 0) throw std::invalid_argument("canonical GAT admission has an empty destination group");
    ledger.backend = "BN254";
    ledger.table_id = tables.table_id;
    ledger.field_modulus = Field::getModulo();
    ledger.field_half_modulus = bn254_half_modulus_string();
    ledger.lrelu_min = tables.lrelu_min;
    ledger.lrelu_max = tables.lrelu_max;
    ledger.exp_min = tables.exp_min;
    ledger.exp_max = tables.exp_max;
    ledger.elu_min = tables.elu_min;
    ledger.elu_max = tables.elu_max;

    const Tensor2D* current = &features;
    std::unique_ptr<Tensor2D> current_owned;
    std::size_t profile_index = 0;
    for (std::size_t layer = 0; layer < 2; ++layer) {
        const std::size_t output_dim = model.tensor(hidden_family(layer, 0) + "/seq/kernel").shape.dims.at(2);
        const std::size_t hidden_heads = model.hidden_heads();
        Tensor2D next(graph.num_nodes, output_dim * hidden_heads);
        const Wide layer_input_abs = admission_tensor_abs_bound(*current);
        for (std::size_t head = 0; head < hidden_heads; ++head) {
            const AttentionScales scales{quant.s_hprime_hidden[layer][head], quant.s_a_src_hidden[layer][head],
                                         quant.s_a_dst_hidden[layer][head], quant.s_agg_pre[layer], quant.s_agg[layer]};
            if (progress != nullptr) progress("hidden_head", layer, head, false);
            ledger.add(compute_attention_admission(graph, *current, *current, model, quant, tables, ranges, maximum_degree, table_bounds, layer_input_abs, layer, head, false, scales, division_profiles, profile_index++, &next, head * output_dim, nullptr));
            if (progress != nullptr) progress("hidden_head", layer, head, true);
        }
        current_owned = std::make_unique<Tensor2D>(std::move(next));
        current = current_owned.get();
    }

    const std::size_t classes = model.classes();
    const std::size_t output_coordinates = graph.num_nodes * classes;
    const Wide output_family_input_abs = admission_tensor_abs_bound(*current);
    std::vector<Int> head_sum(output_coordinates, 0);
    for (std::size_t head = 0; head < model.manifest.output_heads; ++head) {
        const AttentionScales scales{quant.s_yprime_output[head], quant.s_a_src_output[head], quant.s_a_dst_output[head],
                                     quant.s_out_pre, quant.s_out_head};
        std::vector<Int> recovered;
        if (progress != nullptr) progress("output_head", 0, head, false);
        ledger.add(compute_attention_admission(graph, *current, *current, model, quant, tables, ranges, maximum_degree, table_bounds, output_family_input_abs, 0, head, true, scales, division_profiles, profile_index++, nullptr, 0, &recovered));
        const Tensor& bias = model.tensor(output_family(head) + "/output_bias");
        if (bias.shape.dims.size() != 1 || bias.shape.dims[0] != classes || bias.values.size() != classes)
            throw std::invalid_argument("output bias shape mismatch during admission");
        for (std::size_t index = 0; index < output_coordinates; ++index) {
            const Int dividend = checked_int(static_cast<Wide>(quant.s_out_head) * recovered[index], "admission output alignment dividend");
            const DivisionResult aligned = signed_round_div(dividend, quant.s_out_pre);
            require_admission_division(aligned.witness, division_profiles.at(profile_index), "output alignment");
            const Int bias_value = quantize_model_value(bias.values[index % classes], quant.s_b_output[head]);
            const Int logit = checked_int(static_cast<Wide>(aligned.value) + bias_value, "admission output bias addition");
            head_sum[index] = checked_int(static_cast<Wide>(head_sum[index]) + logit, "admission output head sum");
        }
        if (progress != nullptr) progress("output_head", 0, head, true);
        ++profile_index;
    }
    std::vector<Int> logits(output_coordinates, 0);
    const auto& average_profile = division_profiles.at(profile_index++);
    if (progress != nullptr) progress("output_average", 0, 0, false);
    for (std::size_t index = 0; index < output_coordinates; ++index) {
        const DivisionResult average = signed_round_div(head_sum[index], static_cast<Int>(model.manifest.output_heads));
        require_admission_division(average.witness, average_profile, "final output average");
        logits[index] = average.value;
    }
    if (progress != nullptr) progress("output_average", 0, 0, true);
    if (profile_index != division_profiles.size()) throw std::logic_error("canonical GAT admission did not consume all division profiles");

    if (!ledger.validate()) throw std::invalid_argument("canonical GAT admission integer-bound ledger is invalid");
    if (retained_reference != nullptr) {
        if (!retained_reference->trace.relation_ok || !validate_relation(graph, features, model, quant, *retained_reference))
            throw std::invalid_argument("retained canonical forward reference is invalid");
        if (retained_reference->logits.values != logits)
            throw std::invalid_argument("streamed admission logits differ from retained canonical forward result");
        if (!equal_ledger(ledger, retained_reference->trace.ledger))
            throw std::invalid_argument("streamed admission family ledger differs from retained canonical forward result at " +
                                        admission_ledger_mismatch(ledger, retained_reference->trace.ledger));
    }
    return true;
}

}

bool run_forward_admission(const Graph& graph, const Tensor2D& features, const Model& model, const QuantConfig& quant,
                           const std::vector<std::array<unsigned, 5>>& division_range_profiles, std::string* reason,
                           const ForwardResult* retained_reference,
                           ForwardAdmissionProgress progress) {
    try {
        run_forward_admission_impl(graph, features, model, quant, division_range_profiles, retained_reference, progress);
        if (reason != nullptr) reason->clear();
        return true;
    } catch (const std::exception& error) {
        if (reason != nullptr) *reason = error.what();
        return false;
    } catch (...) {
        if (reason != nullptr) *reason = "unknown canonical GAT admission failure";
        return false;
    }
}

DivisionResult signed_round_div(Int dividend, Int denominator, RoundingMode mode) {
    ensure_field();
    if (denominator <= 0) throw std::invalid_argument("SignedRoundDiv denominator must be positive");
    const Wide absolute = abs_wide(dividend);
    const Wide divisor = static_cast<Wide>(denominator);
    const Int sigma = dividend < 0 ? 1 : 0;
    const Int z = absolute == 0 ? 1 : 0;
    Wide quotient = 0;
    Wide remainder = 0;
    Wide delta = 0;
    if (mode == RoundingMode::TOWARD_ZERO) {
        quotient = absolute / divisor;
        remainder = absolute % divisor;
        delta = divisor - 1 - remainder;
    } else {
        quotient = (absolute + divisor / 2) / divisor;
        remainder = 2 * absolute + divisor - 2 * divisor * quotient;
        delta = 2 * divisor - 1 - remainder;
    }
    DivisionResult result;
    result.value = checked_int((sigma == 0 ? quotient : -quotient), "SignedRoundDiv quotient");
    result.witness.sigma = sigma;
    result.witness.z = z;
    result.witness.a = checked_int(absolute, "SignedRoundDiv absolute dividend");
    result.witness.q = checked_int(quotient, "SignedRoundDiv absolute quotient");
    result.witness.r = checked_int(remainder, "SignedRoundDiv remainder");
    result.witness.delta = checked_int(delta, "SignedRoundDiv delta");
    result.witness.dividend = dividend;
    result.witness.denominator = denominator;
    result.witness.a_inverse = z == 1 ? Field(0) : Field(1) / Field(result.witness.a);
    result.witness.d_inverse = Field(1) / Field(denominator);
    if (!result.witness.valid(mode)) throw std::logic_error("internal SignedRoundDiv witness is invalid");
    return result;
}

bool DivisionWitness::valid(RoundingMode mode) const {
    ensure_field();
    if (denominator <= 0 || (sigma != 0 && sigma != 1) || (z != 0 && z != 1) || a < 0 || q < 0 || r < 0 || delta < 0 || sigma != (dividend < 0 ? 1 : 0) ||
        z != (a == 0 ? 1 : 0))
        return false;
    const Wide signed_a = sigma == 0 ? static_cast<Wide>(a) : -static_cast<Wide>(a);
    const Wide signed_q = sigma == 0 ? static_cast<Wide>(q) : -static_cast<Wide>(q);
    if (signed_a != static_cast<Wide>(dividend) || (a == 0 ? !a_inverse.isZero() : !(Field(a) * a_inverse - Field(1)).isZero()) ||
        !(Field(denominator) * d_inverse - Field(1)).isZero())
        return false;
    const Wide d = denominator;
    if (mode == RoundingMode::TOWARD_ZERO) {
        if (static_cast<Wide>(a) != d * q + r || static_cast<Wide>(r) + delta != d - 1) return false;
    } else {
        if (2 * static_cast<Wide>(a) + d != 2 * d * q + r || static_cast<Wide>(r) + delta != 2 * d - 1) return false;
    }
    const Wide expected_abs_q = mode == RoundingMode::TOWARD_ZERO ? static_cast<Wide>(a) / d : (static_cast<Wide>(a) + d / 2) / d;
    return expected_abs_q == q && signed_q == static_cast<Wide>(sigma == 0 ? q : -q) && (z == 1 ? (a == 0 && a_inverse.isZero()) : !a_inverse.isZero());
}

StaticTables StaticTables::build(const QuantConfig& quant) {
    quant.validate();
    ensure_field();
    StaticTables tables;
    tables.table_id = quant.static_tables_id;
    tables.generator_id = quant.static_table_generator_id;
    tables.quant_cfg_id = quant.quant_cfg_id;
    tables.rounding = quant.rounding;
    tables.rescale = quant.rescale;
    tables.output_average = quant.output_average;
    tables.row_order = quant.static_row_order;
    tables.exp_approx_id = quant.exp_approx_id;
    tables.elu_approx_id = quant.elu_approx_id;
    tables.lrelu_slope_num = quant.lrelu_slope_num;
    tables.lrelu_slope_den = quant.lrelu_slope_den;
    tables.s_delta = quant.s_delta;
    tables.s_exp = quant.s_exp;
    tables.s_elu_in = quant.s_agg_pre[0];
    tables.s_elu_out = quant.s_agg[0];
    tables.lrelu_min = quant.static_lrelu_min;
    tables.lrelu_max = quant.static_lrelu_max;
    tables.exp_min = quant.static_exp_min;
    tables.exp_max = quant.static_exp_max;
    tables.elu_min = quant.static_elu_min;
    tables.elu_max = quant.static_elu_max;
    for (Int x = tables.lrelu_min;; ++x) {
        tables.lrelu.push_back(TablePoint{x, table_lrelu_value(x, tables.lrelu_slope_num, tables.lrelu_slope_den)});
        if (x == tables.lrelu_max) break;
    }
    for (Int x = tables.exp_min;; ++x) {
        tables.exp.push_back(TablePoint{x, table_exp_value(x, tables.s_delta, tables.s_exp)});
        if (x == tables.exp_max) break;
    }
    for (Int x = tables.elu_min;; ++x) {
        tables.elu.push_back(TablePoint{x, table_elu_value(x, tables.s_elu_in, tables.s_elu_out)});
        if (x == tables.elu_max) break;
    }
    tables.digest = digest_tables(tables);
    if (!tables.authenticate()) throw std::logic_error("generated static tables failed authentication");
    return tables;
}

StaticTables StaticTables::build(const QuantConfig& quant, Int observed_score_abs, Int observed_delta_max, Int observed_elu_abs) {
    (void)observed_score_abs;
    (void)observed_delta_max;
    (void)observed_elu_abs;
    return build(quant);
}

Int StaticTables::lookup_lrelu(Int x) const {
    if (x < lrelu_min || x > lrelu_max || lrelu.empty()) throw std::out_of_range("LeakyReLU query outside authenticated table");
    return lrelu[static_cast<std::size_t>(x - lrelu_min)].y;
}

Int StaticTables::lookup_exp(Int delta) const {
    if (delta < exp_min || delta > exp_max || exp.empty()) throw std::out_of_range("ExpMap query outside authenticated table");
    return exp[static_cast<std::size_t>(delta - exp_min)].y;
}

Int StaticTables::lookup_elu(Int x) const {
    if (x < elu_min || x > elu_max || elu.empty()) throw std::out_of_range("ELU query outside authenticated table");
    return elu[static_cast<std::size_t>(x - elu_min)].y;
}

bool StaticTables::authenticate() const {
    try {
        if (table_id != "tables:lrelu+elu+exp+range" || generator_id != "zkgpt-gat-static-tables-v2" || quant_cfg_id != "gat_l3_scale_chain" ||
            rounding != "nearest_ties_away" || rescale != "none" || output_average != "signed_round_div" || row_order != "ascending_input" ||
            exp_approx_id != "exp_neg_quantized" || elu_approx_id != "elu_expm1_quantized" || lrelu_slope_num <= 0 || lrelu_slope_den <= 0 ||
            lrelu_slope_num >= lrelu_slope_den || s_delta <= 0 || s_exp <= 0 || s_elu_in <= 0 || s_elu_out <= 0 || lrelu_min >= lrelu_max || exp_min != 0 ||
            exp_max < exp_min || elu_min >= elu_max)
            return false;
        if (lrelu.size() != static_cast<std::size_t>(static_cast<Wide>(lrelu_max) - lrelu_min + 1) ||
            exp.size() != static_cast<std::size_t>(static_cast<Wide>(exp_max) - exp_min + 1) ||
            elu.size() != static_cast<std::size_t>(static_cast<Wide>(elu_max) - elu_min + 1))
            return false;
        for (std::size_t i = 0; i < lrelu.size(); ++i) {
            const Int x = lrelu_min + static_cast<Int>(i);
            if (lrelu[i].x != x || lrelu[i].y != table_lrelu_value(x, lrelu_slope_num, lrelu_slope_den)) return false;
        }
        for (std::size_t i = 0; i < exp.size(); ++i) {
            const Int x = exp_min + static_cast<Int>(i);
            if (exp[i].x != x || exp[i].y != table_exp_value(x, s_delta, s_exp)) return false;
        }
        for (std::size_t i = 0; i < elu.size(); ++i) {
            const Int x = elu_min + static_cast<Int>(i);
            if (elu[i].x != x || elu[i].y != table_elu_value(x, s_elu_in, s_elu_out)) return false;
        }
        return digest == digest_tables(*this);
    } catch (const std::exception&) { return false; }
}

void IntegerBoundLedger::add(const BoundFamily& family) {
    if (family.id.empty()) throw std::invalid_argument("integer-bound family id is empty");
    families.push_back(family);
}

bool IntegerBoundLedger::validate() const {
    if (!actual_bn254_field() || backend != "BN254" || table_id.empty() || field_modulus != bn254_modulus_string() ||
        field_half_modulus != bn254_half_modulus_string() || lrelu_min >= lrelu_max || exp_min < 0 || exp_max < exp_min || elu_min >= elu_max ||
        families.empty())
        return false;
    if (lrelu_min > 0 || lrelu_max < 0 || exp_min != 0 || elu_min > 0 || elu_max < 0) return false;
    for (const BoundFamily& family : families) {
        if (family.id.empty() || family.input_dim <= 0 || family.output_dim <= 0 || family.max_in_degree <= 0 || family.table_input_abs < 0 ||
            family.table_output_abs < 0 || family.delta_abs < 0 || family.exp_abs < 0 || family.sum_abs < 0 || family.numerator_abs < 0 ||
            family.division_x_abs < 0 || family.division_d_abs <= 0 || family.q_abs < 0 || family.r_abs < 0 || family.delta_aux_abs < 0 ||
            family.product_abs < 0 || family.field_modulus != field_modulus || family.field_half_modulus != field_half_modulus || !family.inverse_field_valid ||
            !family.no_wrap)
            return false;
        const Wide* bounds[] = {&family.model_abs,
                                &family.input_abs,
                                &family.weight_abs,
                                &family.attention_abs,
                                &family.score_abs,
                                &family.table_input_abs,
                                &family.table_output_abs,
                                &family.delta_abs,
                                &family.exp_abs,
                                &family.sum_abs,
                                &family.numerator_abs,
                                &family.division_x_abs,
                                &family.division_d_abs,
                                &family.packed_tuple_abs,
                                &family.q_abs,
                                &family.r_abs,
                                &family.delta_aux_abs,
                                &family.product_abs,
                                &family.input_dim,
                                &family.output_dim,
                                &family.max_in_degree,
                                &family.projection_abs,
                                &family.source_score_abs,
                                &family.destination_score_abs,
                                &family.alignment_x_abs,
                                &family.average_x_abs,
                                &family.elu_input_abs};
        for (const Wide* bound : bounds)
            if (!decimal_nonnegative_leq(*bound, field_half_modulus.c_str())) return false;
    }
    return true;
}

ForwardResult run_forward(const Graph& graph, const Tensor2D& features, const Model& model, const QuantConfig& quant) {
    ensure_field();
    graph.validate();
    model.validate_schema();
    quant.validate();
    validate_tensor2d(features, graph.num_nodes, graph.num_features, "public features");
    if (features.cols != model.input_dim()) throw std::invalid_argument("GAT feature dimension does not match model");
    ForwardResult result;
    result.trace.input = features;
    result.trace.tables = StaticTables::build(quant);
    Tensor2D current = features;
    std::vector<Wide> ledger_seen;
    for (std::size_t layer = 0; layer < 2; ++layer) {
        HiddenLayerTrace hidden;
        hidden.layer = layer;
        const std::size_t output_dim = model.tensor(hidden_family(layer, 0) + "/seq/kernel").shape.dims.at(2);
        const std::size_t hidden_heads = model.hidden_heads();
        hidden.heads.reserve(hidden_heads);
        for (std::size_t head = 0; head < hidden_heads; ++head) {
            AttentionScales scales;
            scales.projection = quant.s_hprime_hidden[layer][head];
            scales.attention_src = quant.s_a_src_hidden[layer][head];
            scales.attention_dst = quant.s_a_dst_hidden[layer][head];
            scales.aggregate_pre = quant.s_agg_pre[layer];
            scales.aggregate_output = quant.s_agg[layer];
            AttentionTrace trace = compute_attention(graph, current, model, quant, result.trace.tables, layer, head, false, scales);
            if (trace.output_dim != output_dim || !trace.relation_ok) throw std::runtime_error("hidden GAT relation failed");
            hidden.heads.push_back(std::move(trace));
        }
        hidden.concatenated = Tensor2D(graph.num_nodes, output_dim * hidden_heads);
        for (std::size_t node = 0; node < graph.num_nodes; ++node) {
            for (std::size_t head = 0; head < hidden_heads; ++head) {
                for (std::size_t col = 0; col < output_dim; ++col) {
                    hidden.concatenated.at(node, head * output_dim + col) = hidden.heads[head].activated[node * output_dim + col];
                }
            }
        }
        current = hidden.concatenated;
        result.trace.hidden_layers.push_back(std::move(hidden));
    }

    result.trace.output_heads.reserve(model.manifest.output_heads);
    const std::size_t classes = model.classes();
    for (std::size_t head = 0; head < model.manifest.output_heads; ++head) {
        OutputHeadTrace output;
        output.head = head;
        AttentionScales scales;
        scales.projection = quant.s_yprime_output[head];
        scales.attention_src = quant.s_a_src_output[head];
        scales.attention_dst = quant.s_a_dst_output[head];
        scales.aggregate_pre = quant.s_out_pre;
        scales.aggregate_output = quant.s_out_head;
        output.attention = compute_attention(graph, current, model, quant, result.trace.tables, 0, head, true, scales);
        output.aggregate = output.attention.recovered;
        output.aligned.assign(output.aggregate.size(), 0);
        output.bias.assign(classes, 0);
        const Tensor& bias = model.tensor(output_family(head) + "/output_bias");
        if (bias.shape.dims.size() != 1 || bias.shape.dims[0] != classes || bias.values.size() != classes)
            throw std::invalid_argument("output bias shape mismatch");
        for (std::size_t c = 0; c < classes; ++c) { output.bias[c] = quantize_model_value(bias.values[c], quant.s_b_output[head]); }
        output.logits.assign(output.aggregate.size(), 0);
        output.alignment_divisions.reserve(output.aggregate.size());
        for (std::size_t i = 0; i < output.aggregate.size(); ++i) {
            const Int dividend = checked_int(static_cast<Wide>(quant.s_out_head) * output.aggregate[i], "output alignment dividend");
            const DivisionResult aligned = signed_round_div(dividend, quant.s_out_pre);
            output.alignment_divisions.push_back(aligned.witness);
            output.aligned[i] = aligned.value;
            output.logits[i] = checked_int(static_cast<Wide>(aligned.value) + output.bias[i % classes], "output bias addition");
        }
        if (!output.attention.relation_ok) throw std::runtime_error("output GAT relation failed");
        result.trace.output_heads.push_back(std::move(output));
    }
    result.trace.head_sum.assign(graph.num_nodes * classes, 0);
    for (std::size_t i = 0; i < result.trace.head_sum.size(); ++i) {
        Wide sum = 0;
        for (const OutputHeadTrace& head : result.trace.output_heads) sum += head.logits[i];
        result.trace.head_sum[i] = checked_int(sum, "output head sum");
    }
    result.trace.logits = Tensor2D(graph.num_nodes, classes);
    result.trace.final_average_divisions.reserve(result.trace.head_sum.size());
    for (std::size_t i = 0; i < result.trace.head_sum.size(); ++i) {
        const DivisionResult average = signed_round_div(result.trace.head_sum[i], static_cast<Int>(model.manifest.output_heads));
        result.trace.final_average_divisions.push_back(average.witness);
        result.trace.logits.values[i] = average.value;
    }
    result.logits = result.trace.logits;

    result.trace.ledger.backend = "BN254";
    result.trace.ledger.table_id = result.trace.tables.table_id;
    result.trace.ledger.field_modulus = Field::getModulo();
    result.trace.ledger.field_half_modulus = bn254_half_modulus_string();
    result.trace.ledger.lrelu_min = result.trace.tables.lrelu_min;
    result.trace.ledger.lrelu_max = result.trace.tables.lrelu_max;
    result.trace.ledger.exp_min = result.trace.tables.exp_min;
    result.trace.ledger.exp_max = result.trace.tables.exp_max;
    result.trace.ledger.elu_min = result.trace.tables.elu_min;
    result.trace.ledger.elu_max = result.trace.tables.elu_max;
    const Tensor2D* layer_input = &features;
    for (const HiddenLayerTrace& hidden : result.trace.hidden_layers) {
        const Wide layer_input_abs = admission_tensor_abs_bound(*layer_input);
        for (std::size_t head = 0; head < hidden.heads.size(); ++head) {
            const AttentionScales scales{quant.s_hprime_hidden[hidden.layer][head], quant.s_a_src_hidden[hidden.layer][head],
                                         quant.s_a_dst_hidden[hidden.layer][head], quant.s_agg_pre[hidden.layer], quant.s_agg[hidden.layer]};
            result.trace.ledger.add(make_bound_family(hidden_family(hidden.layer, head), graph, layer_input_abs, model, quant, result.trace.tables, hidden.heads[head], scales));
        }
        layer_input = &hidden.concatenated;
    }
    const Wide output_input_abs = admission_tensor_abs_bound(current);
    for (std::size_t head = 0; head < result.trace.output_heads.size(); ++head) {
        const AttentionScales scales{quant.s_yprime_output[head], quant.s_a_src_output[head], quant.s_a_dst_output[head], quant.s_out_pre, quant.s_out_head};
        result.trace.ledger.add(make_bound_family(output_family(head), graph, output_input_abs, model, quant, result.trace.tables, result.trace.output_heads[head].attention, scales));
    }
    result.trace.relation_ok = result.trace.tables.authenticate() && result.trace.ledger.validate();
    for (const HiddenLayerTrace& hidden : result.trace.hidden_layers)
        for (const AttentionTrace& head : hidden.heads) result.trace.relation_ok = result.trace.relation_ok && head.relation_ok;
    for (const OutputHeadTrace& head : result.trace.output_heads) result.trace.relation_ok = result.trace.relation_ok && head.attention.relation_ok;
    result.trace.relation_ok = result.trace.relation_ok && validate_relation(graph, features, model, quant, result);
    return result;
}

bool validate_relation(const Graph& graph, const Tensor2D& features, const Model& model, const QuantConfig& quant, const ForwardResult& result) {
    try {
        graph.validate();
        model.validate_schema();
        quant.validate();
        validate_tensor2d(features, graph.num_nodes, graph.num_features, "public features");

        ForwardResult expected;
        expected.trace.input = features;
        expected.trace.tables = StaticTables::build(quant);
        Tensor2D current = features;
        for (std::size_t layer = 0; layer < 2; ++layer) {
            HiddenLayerTrace hidden;
            hidden.layer = layer;
            const std::size_t output_dim = model.tensor(hidden_family(layer, 0) + "/seq/kernel").shape.dims.at(2);
            const std::size_t hidden_heads = model.hidden_heads();
            hidden.heads.reserve(hidden_heads);
            for (std::size_t head = 0; head < hidden_heads; ++head) {
                const AttentionScales scales{quant.s_hprime_hidden[layer][head], quant.s_a_src_hidden[layer][head], quant.s_a_dst_hidden[layer][head],
                                             quant.s_agg_pre[layer], quant.s_agg[layer]};
                AttentionTrace trace = compute_attention(graph, current, model, quant, expected.trace.tables, layer, head, false, scales);
                if (trace.output_dim != output_dim) return false;
                hidden.heads.push_back(std::move(trace));
            }
            hidden.concatenated = Tensor2D(graph.num_nodes, output_dim * hidden_heads);
            for (std::size_t node = 0; node < graph.num_nodes; ++node) {
                for (std::size_t head = 0; head < hidden_heads; ++head) {
                    for (std::size_t col = 0; col < output_dim; ++col) {
                        hidden.concatenated.at(node, head * output_dim + col) = hidden.heads[head].activated[node * output_dim + col];
                    }
                }
            }
            current = hidden.concatenated;
            expected.trace.hidden_layers.push_back(std::move(hidden));
        }

        const std::size_t classes = model.classes();
        expected.trace.output_heads.reserve(model.manifest.output_heads);
        for (std::size_t head = 0; head < model.manifest.output_heads; ++head) {
            OutputHeadTrace output;
            output.head = head;
            const AttentionScales scales{quant.s_yprime_output[head], quant.s_a_src_output[head], quant.s_a_dst_output[head], quant.s_out_pre,
                                         quant.s_out_head};
            output.attention = compute_attention(graph, current, model, quant, expected.trace.tables, 0, head, true, scales);
            output.aggregate = output.attention.recovered;
            output.aligned.assign(output.aggregate.size(), 0);
            output.bias.assign(classes, 0);
            const Tensor& bias = model.tensor(output_family(head) + "/output_bias");
            if (bias.shape.dims.size() != 1 || bias.shape.dims[0] != classes || bias.values.size() != classes) return false;
            for (std::size_t c = 0; c < classes; ++c) { output.bias[c] = quantize_model_value(bias.values[c], quant.s_b_output[head]); }
            output.logits.assign(output.aggregate.size(), 0);
            output.alignment_divisions.reserve(output.aggregate.size());
            for (std::size_t i = 0; i < output.aggregate.size(); ++i) {
                const Int dividend = checked_int(static_cast<Wide>(quant.s_out_head) * output.aggregate[i], "validation output alignment dividend");
                const DivisionResult aligned = signed_round_div(dividend, quant.s_out_pre);
                output.alignment_divisions.push_back(aligned.witness);
                output.aligned[i] = aligned.value;
                output.logits[i] = checked_int(static_cast<Wide>(aligned.value) + output.bias[i % classes], "validation output bias addition");
            }
            expected.trace.output_heads.push_back(std::move(output));
        }
        expected.trace.head_sum.assign(graph.num_nodes * classes, 0);
        for (std::size_t i = 0; i < expected.trace.head_sum.size(); ++i) {
            Wide sum = 0;
            for (const OutputHeadTrace& head : expected.trace.output_heads) sum += head.logits[i];
            expected.trace.head_sum[i] = checked_int(sum, "validation output head sum");
        }
        expected.trace.final_average_divisions.reserve(expected.trace.head_sum.size());
        expected.trace.logits = Tensor2D(graph.num_nodes, classes);
        for (std::size_t i = 0; i < expected.trace.head_sum.size(); ++i) {
            const DivisionResult average = signed_round_div(expected.trace.head_sum[i], static_cast<Int>(model.manifest.output_heads));
            expected.trace.final_average_divisions.push_back(average.witness);
            expected.trace.logits.values[i] = average.value;
        }
        expected.logits = expected.trace.logits;

        expected.trace.ledger.backend = "BN254";
        expected.trace.ledger.table_id = expected.trace.tables.table_id;
        expected.trace.ledger.field_modulus = Field::getModulo();
        expected.trace.ledger.field_half_modulus = bn254_half_modulus_string();
        expected.trace.ledger.lrelu_min = expected.trace.tables.lrelu_min;
        expected.trace.ledger.lrelu_max = expected.trace.tables.lrelu_max;
        expected.trace.ledger.exp_min = expected.trace.tables.exp_min;
        expected.trace.ledger.exp_max = expected.trace.tables.exp_max;
        expected.trace.ledger.elu_min = expected.trace.tables.elu_min;
        expected.trace.ledger.elu_max = expected.trace.tables.elu_max;
        const Tensor2D* layer_input = &features;
        for (const HiddenLayerTrace& hidden : expected.trace.hidden_layers) {
            const Wide layer_input_abs = admission_tensor_abs_bound(*layer_input);
            for (std::size_t head = 0; head < hidden.heads.size(); ++head) {
                const AttentionScales scales{quant.s_hprime_hidden[hidden.layer][head], quant.s_a_src_hidden[hidden.layer][head],
                                             quant.s_a_dst_hidden[hidden.layer][head], quant.s_agg_pre[hidden.layer], quant.s_agg[hidden.layer]};
                expected.trace.ledger.add(make_bound_family(hidden_family(hidden.layer, head), graph, layer_input_abs, model, quant, expected.trace.tables, hidden.heads[head], scales));
            }
            layer_input = &hidden.concatenated;
        }
        const Wide output_input_abs = admission_tensor_abs_bound(current);
        for (std::size_t head = 0; head < expected.trace.output_heads.size(); ++head) {
            const AttentionScales scales{quant.s_yprime_output[head], quant.s_a_src_output[head], quant.s_a_dst_output[head], quant.s_out_pre,
                                         quant.s_out_head};
            expected.trace.ledger.add(make_bound_family(output_family(head), graph, output_input_abs, model, quant, expected.trace.tables, expected.trace.output_heads[head].attention, scales));
        }
        expected.trace.relation_ok = expected.trace.tables.authenticate() && expected.trace.ledger.validate();
        for (const HiddenLayerTrace& hidden : expected.trace.hidden_layers) {
            for (const AttentionTrace& head : hidden.heads) expected.trace.relation_ok = expected.trace.relation_ok && head.relation_ok;
        }
        for (const OutputHeadTrace& head : expected.trace.output_heads) expected.trace.relation_ok = expected.trace.relation_ok && head.attention.relation_ok;
        if (!expected.trace.relation_ok) return false;

        if (result.logits.rows != graph.num_nodes || result.logits.cols != graph.num_classes ||
            result.logits.values.size() != graph.num_nodes * graph.num_classes || result.trace.input.rows != graph.num_nodes ||
            result.trace.input.cols != graph.num_features || result.trace.input.values != features.values || result.logits.values != expected.logits.values ||
            result.trace.logits.values != expected.trace.logits.values || result.trace.hidden_layers.size() != expected.trace.hidden_layers.size() ||
            result.trace.output_heads.size() != expected.trace.output_heads.size() || result.trace.head_sum != expected.trace.head_sum ||
            !equal_divisions(result.trace.final_average_divisions, expected.trace.final_average_divisions) ||
            !equal_ledger(result.trace.ledger, expected.trace.ledger))
            return false;
        if (result.trace.tables.table_id != expected.trace.tables.table_id || result.trace.tables.generator_id != expected.trace.tables.generator_id ||
            result.trace.tables.digest != expected.trace.tables.digest || !result.trace.tables.authenticate())
            return false;
        for (std::size_t layer = 0; layer < expected.trace.hidden_layers.size(); ++layer) {
            const HiddenLayerTrace& actual_layer = result.trace.hidden_layers[layer];
            const HiddenLayerTrace& expected_layer = expected.trace.hidden_layers[layer];
            if (actual_layer.layer != expected_layer.layer || actual_layer.heads.size() != expected_layer.heads.size() ||
                actual_layer.concatenated.rows != expected_layer.concatenated.rows || actual_layer.concatenated.cols != expected_layer.concatenated.cols ||
                actual_layer.concatenated.values != expected_layer.concatenated.values)
                return false;
            for (std::size_t head = 0; head < expected_layer.heads.size(); ++head) {
                if (!validate_attention(graph, actual_layer.heads[head], false) || !equal_attention(actual_layer.heads[head], expected_layer.heads[head]))
                    return false;
            }
        }
        for (std::size_t head = 0; head < expected.trace.output_heads.size(); ++head) {
            const OutputHeadTrace& actual = result.trace.output_heads[head];
            const OutputHeadTrace& reference = expected.trace.output_heads[head];
            if (actual.head != reference.head || !validate_attention(graph, actual.attention, true) ||
                !equal_attention(actual.attention, reference.attention) || actual.bias != reference.bias || actual.aggregate != reference.aggregate ||
                actual.aligned != reference.aligned || actual.logits != reference.logits ||
                !equal_divisions(actual.alignment_divisions, reference.alignment_divisions))
                return false;
        }
        return result.trace.relation_ok;
    } catch (const std::exception&) { return false; }
}

}
}
