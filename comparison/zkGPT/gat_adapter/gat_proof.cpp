#include "gat_proof.hpp"

#include "hyrax_util.hpp"
#include "utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <set>
#include <tuple>

// hyrax.hpp declares the integer/pippenger overload but the official
// hyrax.cpp also exports this Fr overload for the same Pedersen primitive.
// Keep the adapter declaration local so the upstream source remains
// untouched while still binding the real symbol at link time.
mcl::bn::G1 perdersen_commit(mcl::bn::G1 *, mcl::bn::Fr *, int);

namespace zkgpt {
namespace gat {
namespace {

using namespace mcl::bn;

std::once_flag field_once;

void ensure_field() {
    std::call_once(field_once, []() { mcl::bn::initPairing(mcl::BN254); });
}

void append_u64(std::string &out, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
        out.push_back(static_cast<char>((value >> (i * 8)) & 0xffU));
    }
}

void append_u32(std::string &out, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        out.push_back(static_cast<char>((value >> (i * 8)) & 0xffU));
    }
}

void append_string(std::string &out, const std::string &value) {
    append_u64(out, static_cast<std::uint64_t>(value.size()));
    out.append(value);
}

void append_int(std::string &out, Int value) {
    const bool negative = value < 0;
    const Wide absolute = negative ? -static_cast<Wide>(value) : static_cast<Wide>(value);
    out.push_back(static_cast<char>(negative ? 1 : 0));
    append_u64(out, static_cast<std::uint64_t>(absolute));
}

void append_shape(std::string &out, const Shape &shape) {
    append_u64(out, static_cast<std::uint64_t>(shape.dims.size()));
    for (std::size_t dimension : shape.dims) append_u64(out, static_cast<std::uint64_t>(dimension));
}

std::string field_hex(const Field &value) {
    std::string result = value.getStr(16);
    if (result.empty()) result = "0";
    return result;
}

std::string hash_hex(const std::string &bytes) {
    ensure_field();
    Field value;
    value.setHashOf(bytes.data(), bytes.size());
    return field_hex(value);
}

std::string point_hex(const G1 &point) {
    return point.serializeToHexStr();
}

std::size_t next_power_of_two(std::size_t value) {
    std::size_t result = 1;
    while (result < std::max<std::size_t>(1, value)) {
        if (result > std::numeric_limits<std::size_t>::max() / 2) throw std::overflow_error("zkGPT input layout overflow");
        result <<= 1;
    }
    return result;
}

std::size_t checked_add(std::size_t left, std::size_t right, const char *what) {
    if (right > std::numeric_limits<std::size_t>::max() - left) throw std::overflow_error(std::string("zkGPT size overflow: ") + what);
    return left + right;
}

std::size_t checked_mul(std::size_t left, std::size_t right, const char *what) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) throw std::overflow_error(std::string("zkGPT size overflow: ") + what);
    return left * right;
}

std::uint32_t as_u32(std::size_t value, const char *what) {
    if (value > std::numeric_limits<std::uint32_t>::max()) throw std::overflow_error(std::string("zkGPT circuit index overflow: ") + what);
    return static_cast<std::uint32_t>(value);
}

char bit_length(std::size_t value) {
    std::size_t n = std::max<std::size_t>(1, value);
    char bits = 0;
    while (n > 1) {
        n >>= 1;
        ++bits;
    }
    return bits;
}

std::string hidden_tensor_name(std::size_t layer, std::size_t head, const char *kind) {
    std::ostringstream out;
    out << "hidden/layer" << layer << "/head" << head << "/" << kind;
    return out.str();
}

std::string output_tensor_name(std::size_t head, const char *kind) {
    std::ostringstream out;
    out << "output/head" << head << "/" << kind;
    return out.str();
}

// Keep the proof-plan translation unit self-contained.  The forward
// implementation has equivalent private helpers, but those intentionally do
// not cross the adapter source boundary.
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

Int quantize_model_value(double value, Int scale) {
    if (scale <= 0) throw std::invalid_argument("model quantization scale must be positive");
    const long double scaled = static_cast<long double>(value) * static_cast<long double>(scale);
    const long double rounded = scaled >= 0 ? std::floor(scaled + 0.5L) : std::ceil(scaled - 0.5L);
    if (rounded > static_cast<long double>(std::numeric_limits<Int>::max()) || rounded < static_cast<long double>(std::numeric_limits<Int>::min())) throw std::overflow_error("model quantization overflow");
    return static_cast<Int>(rounded);
}

std::vector<Int> flatten_model(const Model &model, const QuantConfig &quant) {
    model.validate_schema();
    quant.validate();
    std::vector<Int> result;
    const auto append_tensor = [&result](const Tensor &tensor, Int scale) {
        for (double value : tensor.values) result.push_back(quantize_model_value(value, scale));
    };
    for (std::size_t layer = 0; layer < 2; ++layer) {
        for (std::size_t head = 0; head < 8; ++head) {
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "seq/kernel")), quant.s_w_hidden[layer][head]);
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "attn_src/kernel")), quant.s_a_src_hidden[layer][head]);
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "attn_dst/kernel")), quant.s_a_dst_hidden[layer][head]);
        }
    }
    for (std::size_t head = 0; head < 2; ++head) {
        append_tensor(model.tensor(output_tensor_name(head, "seq/kernel")), quant.s_w_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "attn_src/kernel")), quant.s_a_src_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "attn_dst/kernel")), quant.s_a_dst_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "output_bias")), quant.s_b_output[head]);
    }
    return result;
}

const char *canonical_selector() {
    return "00:X_pub,01:W,10:Aux,11:L_qry";
}

const char *canonical_opening_policy() {
    return "same_r_three_openings_v1";
}

void append_double_bits(std::string &out, double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected double width");
    std::memcpy(&bits, &value, sizeof(bits));
    append_u64(out, bits);
}

void append_int_vector(std::string &out, const std::vector<Int> &values) {
    append_u64(out, static_cast<std::uint64_t>(values.size()));
    for (Int value : values) append_int(out, value);
}

void append_manifest_encoding(std::string &out, const ModelManifest &manifest) {
    append_string(out, "zkgpt-gat-canonical-manifest-v1");
    append_string(out, manifest.dataset);
    append_string(out, manifest.model_arch_id);
    append_string(out, manifest.quant_cfg_id);
    append_u64(out, static_cast<std::uint64_t>(manifest.tensor_count));
    append_u64(out, static_cast<std::uint64_t>(manifest.layers));
    append_u64(out, static_cast<std::uint64_t>(manifest.output_heads));
    append_u64(out, static_cast<std::uint64_t>(manifest.classes));
    append_string(out, manifest.model_param_id);
    append_string(out, manifest.static_table_id);
    append_string(out, manifest.input_preprocess_id);
    append_string(out, manifest.batching_rule);
    append_string(out, manifest.edge_sort_rule);
    append_string(out, manifest.self_loop_rule);
    append_string(out, manifest.label_format);
    append_string(out, manifest.output_average_rule);
    append_string(out, manifest.quantization_scale_chain);
}

void append_quant_encoding(std::string &out, const QuantConfig &quant) {
    append_string(out, "zkgpt-gat-canonical-quant-v1");
    append_string(out, quant.schema);
    append_string(out, quant.quant_cfg_id);
    append_int(out, quant.s_input);
    append_int(out, quant.s_delta);
    append_int(out, quant.s_exp);
    for (Int value : quant.s_h) append_int(out, value);
    for (std::size_t layer = 0; layer < quant.s_w_hidden.size(); ++layer) {
        append_int_vector(out, quant.s_w_hidden[layer]);
        append_int_vector(out, quant.s_hprime_hidden[layer]);
        append_int_vector(out, quant.s_a_src_hidden[layer]);
        append_int_vector(out, quant.s_a_dst_hidden[layer]);
    }
    for (Int value : quant.s_agg_pre) append_int(out, value);
    for (Int value : quant.s_agg) append_int(out, value);
    for (Int value : quant.s_cat) append_int(out, value);
    append_int(out, quant.s_out_pre);
    append_int(out, quant.s_out_head);
    append_int(out, quant.s_out);
    for (Int value : quant.s_w_output) append_int(out, value);
    for (Int value : quant.s_yprime_output) append_int(out, value);
    for (Int value : quant.s_a_src_output) append_int(out, value);
    for (Int value : quant.s_a_dst_output) append_int(out, value);
    for (Int value : quant.s_b_output) append_int(out, value);
    append_string(out, quant.rounding);
    append_string(out, quant.rescale);
    append_string(out, quant.output_average);
    append_int(out, quant.range_bits);
    append_string(out, quant.static_tables_id);
    append_string(out, quant.static_table_generator_id);
    append_string(out, quant.static_row_order);
    append_string(out, quant.exp_approx_id);
    append_string(out, quant.elu_approx_id);
    append_int(out, quant.static_lrelu_min);
    append_int(out, quant.static_lrelu_max);
    append_int(out, quant.static_exp_min);
    append_int(out, quant.static_exp_max);
    append_int(out, quant.static_elu_min);
    append_int(out, quant.static_elu_max);
    append_int(out, quant.lrelu_slope_num);
    append_int(out, quant.lrelu_slope_den);
}

void append_layout_encoding(std::string &out, const InputLayout &layout) {
    append_string(out, "zkgpt-gat-four-segment-layout-v1");
    append_u64(out, static_cast<std::uint64_t>(layout.n_raw));
    append_u64(out, static_cast<std::uint64_t>(layout.n_in));
    append_string(out, layout.selector);
    append_u64(out, static_cast<std::uint64_t>(layout.segments.size()));
    for (const InputSegment &segment : layout.segments) {
        append_string(out, segment.name);
        append_u64(out, static_cast<std::uint64_t>(segment.offset));
        append_u64(out, static_cast<std::uint64_t>(segment.logical_length));
        append_u64(out, static_cast<std::uint64_t>(segment.padded_length));
    }
}

bool canonical_layout_valid(const InputLayout &layout) {
    if (layout.n_in == 0 || layout.selector != canonical_selector() || layout.segments.size() != 4) return false;
    const char *names[] = {"X_pub", "W", "Aux", "L_qry"};
    std::size_t max_logical = 1;
    for (std::size_t index = 0; index < layout.segments.size(); ++index) {
        const InputSegment &segment = layout.segments[index];
        if (segment.name != names[index] || segment.padded_length != layout.n_in ||
            segment.logical_length > layout.n_in ||
            segment.offset != checked_mul(layout.n_in, index, "canonical segment offset")) return false;
        max_logical = std::max(max_logical, segment.logical_length);
    }
    if (layout.n_raw != max_logical || next_power_of_two(layout.n_raw) != layout.n_in) return false;
    return true;
}

std::string canonical_manifest_digest(const Model &model) {
    std::string encoded;
    append_manifest_encoding(encoded, model.manifest);
    return hash_hex(encoded);
}

void append_canonical_model_encoding(std::string &out, const Model &model,
                                     const QuantConfig &quant, bool include_values) {
    append_string(out, "zkgpt-gat-canonical-model-v1");
    append_u64(out, static_cast<std::uint64_t>(model.tensors.size()));
    const auto append_tensor = [&out, include_values](const Tensor &tensor, Int scale) {
        append_string(out, tensor.name);
        append_shape(out, tensor.shape);
        append_int(out, scale);
        append_u64(out, static_cast<std::uint64_t>(tensor.values.size()));
        if (include_values) {
            for (double value : tensor.values) append_double_bits(out, value);
        }
    };
    for (std::size_t layer = 0; layer < 2; ++layer) {
        for (std::size_t head = 0; head < 8; ++head) {
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "seq/kernel")), quant.s_w_hidden[layer][head]);
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "attn_src/kernel")), quant.s_a_src_hidden[layer][head]);
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "attn_dst/kernel")), quant.s_a_dst_hidden[layer][head]);
        }
    }
    for (std::size_t head = 0; head < 2; ++head) {
        append_tensor(model.tensor(output_tensor_name(head, "seq/kernel")), quant.s_w_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "attn_src/kernel")), quant.s_a_src_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "attn_dst/kernel")), quant.s_a_dst_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "output_bias")), quant.s_b_output[head]);
    }
}

std::string canonical_model_digest(const Model &model, const QuantConfig &quant) {
    std::string encoded;
    append_canonical_model_encoding(encoded, model, quant, true);
    return hash_hex(encoded);
}

std::string canonical_bounds_digest(const Model &model, const QuantConfig &quant) {
    std::string encoded;
    append_string(encoded, "zkgpt-gat-per-tensor-bounds-v1");
    const auto append_tensor = [&encoded](const Tensor &tensor, Int scale) {
        append_string(encoded, tensor.name);
        append_shape(encoded, tensor.shape);
        append_int(encoded, scale);
        Int max_abs = 0;
        for (double value : tensor.values) {
            const Int quantized = quantize_model_value(value, scale);
            const Wide absolute = quantized < 0 ? -static_cast<Wide>(quantized) : static_cast<Wide>(quantized);
            if (absolute > static_cast<Wide>(max_abs)) max_abs = static_cast<Int>(absolute);
        }
        append_int(encoded, max_abs);
    };
    for (std::size_t layer = 0; layer < 2; ++layer) {
        for (std::size_t head = 0; head < 8; ++head) {
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "seq/kernel")), quant.s_w_hidden[layer][head]);
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "attn_src/kernel")), quant.s_a_src_hidden[layer][head]);
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "attn_dst/kernel")), quant.s_a_dst_hidden[layer][head]);
        }
    }
    for (std::size_t head = 0; head < 2; ++head) {
        append_tensor(model.tensor(output_tensor_name(head, "seq/kernel")), quant.s_w_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "attn_src/kernel")), quant.s_a_src_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "attn_dst/kernel")), quant.s_a_dst_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "output_bias")), quant.s_b_output[head]);
    }
    return hash_hex(encoded);
}

std::string canonical_quant_digest(const QuantConfig &quant) {
    std::string encoded;
    append_quant_encoding(encoded, quant);
    return hash_hex(encoded);
}

std::string commitment_encoding(const CommitmentWire &commitment);

std::string registration_material_encoding(const Model &model, const QuantConfig &quant,
                                            const InputLayout &layout,
                                            const std::string &model_arch_id) {
    std::string encoded;
    append_string(encoded, "zkgpt-gat-registration-material-v2");
    append_string(encoded, model.manifest.dataset);
    append_string(encoded, model_arch_id);
    append_string(encoded, quant.quant_cfg_id);
    append_string(encoded, StaticTables::build(quant).table_id);
    append_string(encoded, "BN254");
    append_string(encoded, Field::getModulo());
    append_string(encoded, canonical_manifest_digest(model));
    append_string(encoded, canonical_model_digest(model, quant));
    append_string(encoded, canonical_bounds_digest(model, quant));
    append_string(encoded, canonical_quant_digest(quant));
    append_layout_encoding(encoded, layout);
    append_string(encoded, canonical_opening_policy());
    return encoded;
}

std::string registration_basis_label(const Model &model, const QuantConfig &quant,
                                     const InputLayout &layout,
                                     const std::string &model_arch_id) {
    return "zkGPT/model/v2/" + hash_hex(registration_material_encoding(model, quant, layout, model_arch_id));
}

std::string registration_binding_encoding(const ModelRegistration &registration) {
    std::string encoded;
    append_string(encoded, "zkgpt-gat-registration-binding-v2");
    append_string(encoded, registration.dataset);
    append_string(encoded, registration.model_arch_id);
    append_string(encoded, registration.quant_cfg_id);
    append_string(encoded, registration.static_tables_id);
    append_string(encoded, registration.field_id);
    append_string(encoded, registration.field_modulus);
    append_string(encoded, registration.manifest_digest);
    append_string(encoded, registration.model_digest);
    append_string(encoded, registration.bounds_digest);
    append_string(encoded, registration.quant_digest);
    append_string(encoded, registration.layout_digest);
    append_string(encoded, registration.selector);
    append_u64(encoded, static_cast<std::uint64_t>(registration.n_raw));
    append_u64(encoded, static_cast<std::uint64_t>(registration.n_in));
    append_string(encoded, registration.opening_policy);
    append_string(encoded, commitment_encoding(registration.com_W));
    return encoded;
}

void populate_registration_metadata(ModelRegistration &registration,
                                    const Model &model, const QuantConfig &quant,
                                    const InputLayout &layout,
                                    const std::string &model_arch_id) {
    ensure_field();
    if (!canonical_layout_valid(layout)) throw std::invalid_argument("model registration requires the canonical four-segment layout");
    if (model_arch_id.empty() || model.manifest.model_arch_id != model_arch_id) throw std::invalid_argument("model registration architecture identity mismatch");
    const StaticTables tables = StaticTables::build(quant);
    if (!tables.authenticate()) throw std::invalid_argument("model registration static table authentication failed");
    registration.dataset = model.manifest.dataset;
    registration.model_arch_id = model_arch_id;
    registration.quant_cfg_id = quant.quant_cfg_id;
    registration.static_tables_id = tables.table_id;
    registration.field_id = "BN254";
    registration.field_modulus = Field::getModulo();
    registration.manifest_digest = canonical_manifest_digest(model);
    registration.model_digest = canonical_model_digest(model, quant);
    registration.bounds_digest = canonical_bounds_digest(model, quant);
    registration.quant_digest = canonical_quant_digest(quant);
    std::string layout_encoded;
    append_layout_encoding(layout_encoded, layout);
    registration.layout_digest = hash_hex(layout_encoded);
    registration.selector = layout.selector;
    registration.n_raw = layout.n_raw;
    registration.n_in = layout.n_in;
    registration.input_segments = layout.segments;
    registration.opening_policy = canonical_opening_policy();
}

bool registration_metadata_matches(const ModelRegistration &actual,
                                    const ModelRegistration &expected) {
    if (actual.input_segments.size() != expected.input_segments.size()) return false;
    for (std::size_t index = 0; index < actual.input_segments.size(); ++index) {
        const InputSegment &left = actual.input_segments[index];
        const InputSegment &right = expected.input_segments[index];
        if (left.name != right.name || left.offset != right.offset ||
            left.logical_length != right.logical_length || left.padded_length != right.padded_length) return false;
    }
    return actual.dataset == expected.dataset &&
           actual.model_arch_id == expected.model_arch_id &&
           actual.quant_cfg_id == expected.quant_cfg_id &&
           actual.static_tables_id == expected.static_tables_id &&
           actual.field_id == expected.field_id &&
           actual.field_modulus == expected.field_modulus &&
           actual.manifest_digest == expected.manifest_digest &&
           actual.model_digest == expected.model_digest &&
           actual.bounds_digest == expected.bounds_digest &&
           actual.quant_digest == expected.quant_digest &&
           actual.layout_digest == expected.layout_digest &&
           actual.selector == expected.selector &&
           actual.n_raw == expected.n_raw && actual.n_in == expected.n_in &&
           actual.opening_policy == expected.opening_policy;
}

InputLayout registration_layout(const ModelRegistration &registration) {
    InputLayout layout;
    layout.n_raw = registration.n_raw;
    layout.n_in = registration.n_in;
    layout.selector = registration.selector;
    layout.segments = registration.input_segments;
    return layout;
}

bool registration_wire_valid(const ModelRegistration &registration) {
    ensure_field();
    const InputLayout layout = registration_layout(registration);
    if (!canonical_layout_valid(layout) || registration.status != "registered_long_lived_com_W" ||
        registration.dataset.empty() || registration.model_arch_id.empty() ||
        registration.quant_cfg_id.empty() || registration.static_tables_id.empty() ||
        registration.field_id != "BN254" || registration.field_modulus != Field::getModulo() ||
        registration.manifest_digest.empty() || registration.model_digest.empty() ||
        registration.bounds_digest.empty() || registration.quant_digest.empty() ||
        registration.opening_policy != canonical_opening_policy() ||
        !registration.com_W.valid() ||
        registration.com_W.logical_size != layout.segments.at(1).logical_length ||
        registration.com_W.padded_size != layout.n_in) return false;
    std::string layout_encoded;
    append_layout_encoding(layout_encoded, layout);
    if (registration.layout_digest != hash_hex(layout_encoded)) return false;
    if (registration.binding_digest != hash_hex(registration_binding_encoding(registration))) return false;
    return registration.registration_digest ==
           hash_hex("zkgpt-gat-registration-v2/" + registration.binding_digest + "/" + commitment_encoding(registration.com_W));
}

std::size_t attention_aux_size(const AttentionTrace &trace) {
    std::size_t result = 0;
    const auto add = [&result](std::size_t value, const char *what) { result = checked_add(result, value, what); };
    add(trace.projection.size(), "projection");
    add(trace.source_score.size(), "source score");
    add(trace.destination_score.size(), "destination score");
    add(trace.raw_score.size(), "raw score");
    add(trace.leaky_score.size(), "leaky score");
    add(trace.group_max.size(), "group max");
    add(trace.delta.size(), "delta");
    add(trace.exp.size(), "exp");
    add(trace.group_sum.size(), "group sum");
    add(trace.group_inverse.size(), "group inverse");
    add(trace.max_selector.size(), "max selector");
    add(trace.alpha.size(), "alpha");
    add(trace.integer_numerator.size(), "integer numerator");
    add(trace.field_numerator.size(), "field numerator");
    add(trace.field_aggregate.size(), "field aggregate");
    add(trace.recovered.size(), "recovered");
    add(trace.activated.size(), "activated");
    add(checked_mul(trace.divisions.size(), 9, "division witness"), "division witness");
    return result;
}

std::size_t trace_aux_size(const GatTrace *trace) {
    if (trace == nullptr) return 1;
    std::size_t result = 0;
    const auto add = [&result](std::size_t value, const char *what) { result = checked_add(result, value, what); };
    add(trace->input.values.size(), "input trace");
    for (const HiddenLayerTrace &layer : trace->hidden_layers) {
        add(layer.concatenated.values.size(), "concat trace");
        for (const AttentionTrace &head : layer.heads) add(attention_aux_size(head), "hidden attention trace");
    }
    for (const OutputHeadTrace &head : trace->output_heads) {
        add(attention_aux_size(head.attention), "output attention trace");
        add(head.bias.size(), "output bias");
        add(head.aggregate.size(), "output aggregate");
        add(head.aligned.size(), "output aligned");
        add(head.logits.size(), "output logits");
        add(checked_mul(head.alignment_divisions.size(), 9, "alignment division witness"), "alignment division witness");
    }
    add(trace->head_sum.size(), "head sum");
    add(checked_mul(trace->final_average_divisions.size(), 9, "average division witness"), "average division witness");
    add(trace->logits.values.size(), "logits trace");
    return result;
}

std::size_t trace_query_size(const GatTrace *trace) {
    if (trace == nullptr) return 1;
    std::size_t lrelu_queries = 0;
    std::size_t exp_queries = 0;
    std::size_t elu_queries = 0;
    std::size_t range_queries = 0;
    const auto add_attention = [&lrelu_queries, &exp_queries, &elu_queries, &range_queries](const AttentionTrace &head, bool elu) {
        lrelu_queries = checked_add(lrelu_queries, head.raw_score.size(), "LeakyReLU query count");
        exp_queries = checked_add(exp_queries, head.delta.size(), "ExpMap query count");
        if (elu) elu_queries = checked_add(elu_queries, head.recovered.size(), "ELU query count");
        range_queries = checked_add(range_queries, checked_mul(head.recovered.size(), 4, "range query count"), "range query count");
        range_queries = checked_add(range_queries, checked_mul(head.divisions.size(), 4, "division query count"), "division query count");
    };
    for (const HiddenLayerTrace &layer : trace->hidden_layers) for (const AttentionTrace &head : layer.heads) add_attention(head, true);
    for (const OutputHeadTrace &head : trace->output_heads) {
        add_attention(head.attention, false);
        range_queries = checked_add(range_queries, checked_mul(head.alignment_divisions.size(), 4, "alignment query count"), "alignment query count");
    }
    range_queries = checked_add(range_queries, checked_mul(trace->final_average_divisions.size(), 4, "average query count"), "average query count");
    return checked_add(checked_add(checked_add(next_power_of_two(lrelu_queries), next_power_of_two(exp_queries), "padded query count"),
                                  next_power_of_two(elu_queries), "padded query count"),
                       next_power_of_two(range_queries), "padded query count");
}

std::size_t canonical_attention_aux_size(const Graph &graph, std::size_t output_dim) {
    const std::size_t nodes = graph.num_nodes;
    const std::size_t edges = graph.edges();
    const std::size_t coordinates = checked_mul(nodes, output_dim, "canonical attention coordinates");
    std::size_t result = 0;
    const auto add = [&result](std::size_t value, const char *what) { result = checked_add(result, value, what); };
    add(coordinates, "projection coordinates");
    add(nodes, "source score coordinates");
    add(nodes, "destination score coordinates");
    add(edges, "raw score coordinates");
    add(edges, "LeakyReLU coordinates");
    add(nodes, "group max coordinates");
    add(edges, "delta coordinates");
    add(edges, "exp coordinates");
    add(nodes, "group sum coordinates");
    add(nodes, "group inverse coordinates");
    add(edges, "max selector coordinates");
    add(edges, "alpha coordinates");
    add(coordinates, "integer numerator coordinates");
    add(coordinates, "field numerator coordinates");
    add(coordinates, "field aggregate coordinates");
    add(coordinates, "recovered coordinates");
    add(coordinates, "activated coordinates");
    add(checked_mul(coordinates, 9, "division witness coordinates"), "division witness coordinates");
    return result;
}

std::size_t canonical_aux_size(const Graph &graph, const Model &model) {
    const std::size_t hidden0 = model.tensor(hidden_tensor_name(0, 0, "seq/kernel")).shape.dims.at(2);
    const std::size_t hidden1 = model.tensor(hidden_tensor_name(1, 0, "seq/kernel")).shape.dims.at(2);
    const std::size_t classes = model.classes();
    std::size_t result = checked_mul(graph.num_nodes, model.input_dim(), "canonical input trace coordinates");
    for (std::size_t output_dim : {hidden0, hidden1}) {
        result = checked_add(result, checked_mul(graph.num_nodes, checked_mul(output_dim, 8, "canonical concat width"), "canonical concat coordinates"), "canonical concat coordinates");
        for (std::size_t head = 0; head < 8; ++head) result = checked_add(result, canonical_attention_aux_size(graph, output_dim), "canonical hidden attention coordinates");
    }
    for (std::size_t head = 0; head < 2; ++head) {
        result = checked_add(result, canonical_attention_aux_size(graph, classes), "canonical output attention coordinates");
        result = checked_add(result, classes, "canonical output bias coordinates");
        const std::size_t output_coordinates = checked_mul(graph.num_nodes, classes, "canonical output coordinates");
        result = checked_add(result, output_coordinates, "canonical output aggregate coordinates");
        result = checked_add(result, output_coordinates, "canonical output aligned coordinates");
        result = checked_add(result, output_coordinates, "canonical output logits coordinates");
        result = checked_add(result, checked_mul(output_coordinates, 9, "canonical alignment witness coordinates"), "canonical alignment witness coordinates");
    }
    const std::size_t output_coordinates = checked_mul(graph.num_nodes, classes, "canonical final coordinates");
    result = checked_add(result, output_coordinates, "canonical head sum coordinates");
    result = checked_add(result, checked_mul(output_coordinates, 9, "canonical average witness coordinates"), "canonical average witness coordinates");
    result = checked_add(result, output_coordinates, "canonical final logits coordinates");
    return result;
}

struct QueryCounts {
    std::size_t lrelu = 0;
    std::size_t exp = 0;
    std::size_t elu = 0;
    std::size_t range = 0;
};

QueryCounts canonical_query_counts(const Graph &graph, const Model &model) {
    QueryCounts result;
    const std::size_t hidden0 = model.tensor(hidden_tensor_name(0, 0, "seq/kernel")).shape.dims.at(2);
    const std::size_t hidden1 = model.tensor(hidden_tensor_name(1, 0, "seq/kernel")).shape.dims.at(2);
    const std::size_t classes = model.classes();
    const auto add_attention = [&result, &graph](std::size_t output_dim, bool elu) {
        const std::size_t coordinates = checked_mul(graph.num_nodes, output_dim, "canonical query coordinates");
        result.lrelu = checked_add(result.lrelu, graph.edges(), "canonical LeakyReLU queries");
        result.exp = checked_add(result.exp, graph.edges(), "canonical ExpMap queries");
        if (elu) result.elu = checked_add(result.elu, coordinates, "canonical ELU queries");
        result.range = checked_add(result.range, checked_mul(coordinates, 4, "canonical range queries"), "canonical range queries");
        result.range = checked_add(result.range, checked_mul(coordinates, 4, "canonical division queries"), "canonical division queries");
    };
    for (std::size_t output_dim : {hidden0, hidden1}) for (std::size_t head = 0; head < 8; ++head) add_attention(output_dim, true);
    for (std::size_t head = 0; head < 2; ++head) {
        add_attention(classes, false);
        const std::size_t coordinates = checked_mul(graph.num_nodes, classes, "canonical output query coordinates");
        result.range = checked_add(result.range, checked_mul(coordinates, 4, "canonical alignment queries"), "canonical alignment queries");
    }
    const std::size_t coordinates = checked_mul(graph.num_nodes, classes, "canonical final query coordinates");
    result.range = checked_add(result.range, checked_mul(coordinates, 4, "canonical average queries"), "canonical average queries");
    return result;
}

std::size_t canonical_query_size(const Graph &graph, const Model &model) {
    const QueryCounts counts = canonical_query_counts(graph, model);
    std::size_t result = next_power_of_two(counts.lrelu);
    result = checked_add(result, next_power_of_two(counts.exp), "canonical padded query size");
    result = checked_add(result, next_power_of_two(counts.elu), "canonical padded query size");
    return checked_add(result, next_power_of_two(counts.range), "canonical padded query size");
}

InputLayout make_layout(std::size_t public_size, std::size_t model_size,
                        std::size_t aux_size, std::size_t query_size) {
    InputLayout layout;
    layout.n_raw = std::max<std::size_t>(1, std::max(std::max(public_size, model_size), std::max(aux_size, query_size)));
    layout.n_in = next_power_of_two(layout.n_raw);
    layout.segments = {
        {"X_pub", 0, public_size, layout.n_in},
        {"W", layout.n_in, model_size, layout.n_in},
        {"Aux", checked_mul(layout.n_in, 2, "Aux segment offset"), aux_size, layout.n_in},
        {"L_qry", checked_mul(layout.n_in, 3, "Lasso segment offset"), query_size, layout.n_in}
    };
    return layout;
}

struct AttentionCoordinates {
    std::size_t projection = 0;
    std::size_t source_score = 0;
    std::size_t destination_score = 0;
    std::size_t raw_score = 0;
    std::size_t leaky_score = 0;
    std::size_t group_max = 0;
    std::size_t delta = 0;
    std::size_t exp = 0;
    std::size_t group_sum = 0;
    std::size_t group_inverse = 0;
    std::size_t max_selector = 0;
    std::size_t alpha = 0;
    std::size_t integer_numerator = 0;
    std::size_t field_numerator = 0;
    std::size_t field_aggregate = 0;
    std::size_t recovered = 0;
    std::size_t activated = 0;
    std::size_t divisions = 0;
    std::size_t input_dim = 0;
    std::size_t output_dim = 0;
};

struct TraceCoordinates {
    std::vector<std::vector<AttentionCoordinates>> hidden;
    std::vector<std::size_t> hidden_concat;
    std::vector<AttentionCoordinates> output;
    std::vector<std::size_t> output_bias;
    std::vector<std::size_t> output_aggregate;
    std::vector<std::size_t> output_aligned;
    std::vector<std::size_t> output_logits;
    std::vector<std::size_t> output_alignment_divisions;
    std::size_t head_sum = 0;
    std::size_t final_average_divisions = 0;
    std::size_t final_logits = 0;
    std::size_t end = 0;
};

AttentionCoordinates assign_attention_coordinates(std::size_t &cursor,
                                                  const Graph &graph,
                                                  std::size_t input_dim,
                                                  std::size_t output_dim) {
    AttentionCoordinates result;
    result.input_dim = input_dim;
    result.output_dim = output_dim;
    const std::size_t nodes = graph.num_nodes;
    const std::size_t edges = graph.edges();
    const std::size_t coordinates = checked_mul(nodes, output_dim, "attention projection coordinates");
    const auto take = [&cursor](std::size_t count, const char *what) {
        const std::size_t offset = cursor;
        cursor = checked_add(cursor, count, what);
        return offset;
    };
    result.projection = take(coordinates, "projection coordinates");
    result.source_score = take(nodes, "source score coordinates");
    result.destination_score = take(nodes, "destination score coordinates");
    result.raw_score = take(edges, "raw score coordinates");
    result.leaky_score = take(edges, "LeakyReLU coordinates");
    result.group_max = take(nodes, "group max coordinates");
    result.delta = take(edges, "delta coordinates");
    result.exp = take(edges, "ExpMap coordinates");
    result.group_sum = take(nodes, "group sum coordinates");
    result.group_inverse = take(nodes, "group inverse coordinates");
    result.max_selector = take(edges, "max selector coordinates");
    result.alpha = take(edges, "alpha coordinates");
    result.integer_numerator = take(coordinates, "integer numerator coordinates");
    result.field_numerator = take(coordinates, "field numerator coordinates");
    result.field_aggregate = take(coordinates, "field aggregate coordinates");
    result.recovered = take(coordinates, "recovered coordinates");
    result.activated = take(coordinates, "activated coordinates");
    result.divisions = take(checked_mul(coordinates, 9, "division witness coordinates"), "division witness coordinates");
    return result;
}

TraceCoordinates make_trace_coordinates(const Graph &graph, const Model &model,
                                        const InputLayout &layout) {
    if (layout.segments.size() != 4) throw std::invalid_argument("trace coordinates require four input segments");
    TraceCoordinates result;
    std::size_t cursor = checked_add(layout.segments.at(2).offset,
                                     checked_mul(graph.num_nodes, model.input_dim(), "input trace coordinates"),
                                     "trace coordinate cursor");
    result.hidden.resize(2);
    result.hidden_concat.resize(2);
    for (std::size_t layer = 0; layer < 2; ++layer) {
        const std::size_t output_dim = model.tensor(hidden_tensor_name(layer, 0, "seq/kernel")).shape.dims.at(2);
        result.hidden_concat[layer] = cursor;
        cursor = checked_add(cursor, checked_mul(graph.num_nodes, checked_mul(output_dim, 8, "concat width"), "concat coordinates"), "trace coordinate cursor");
        result.hidden[layer].reserve(8);
        const std::size_t input_dim = model.tensor(hidden_tensor_name(layer, 0, "seq/kernel")).shape.dims.at(1);
        for (std::size_t head = 0; head < 8; ++head) result.hidden[layer].push_back(assign_attention_coordinates(cursor, graph, input_dim, output_dim));
    }
    result.output.resize(2);
    result.output_bias.resize(2);
    result.output_aggregate.resize(2);
    result.output_aligned.resize(2);
    result.output_logits.resize(2);
    result.output_alignment_divisions.resize(2);
    const std::size_t classes = model.classes();
    const std::size_t output_coordinates = checked_mul(graph.num_nodes, classes, "output coordinates");
    for (std::size_t head = 0; head < 2; ++head) {
        const std::size_t input_dim = model.tensor(output_tensor_name(head, "seq/kernel")).shape.dims.at(1);
        result.output[head] = assign_attention_coordinates(cursor, graph, input_dim, classes);
        result.output_bias[head] = cursor;
        cursor = checked_add(cursor, classes, "output bias coordinates");
        result.output_aggregate[head] = cursor;
        cursor = checked_add(cursor, output_coordinates, "output aggregate coordinates");
        result.output_aligned[head] = cursor;
        cursor = checked_add(cursor, output_coordinates, "output aligned coordinates");
        result.output_logits[head] = cursor;
        cursor = checked_add(cursor, output_coordinates, "output logits coordinates");
        result.output_alignment_divisions[head] = cursor;
        cursor = checked_add(cursor, checked_mul(output_coordinates, 9, "alignment division coordinates"), "alignment division coordinates");
    }
    result.head_sum = cursor;
    cursor = checked_add(cursor, output_coordinates, "head sum coordinates");
    result.final_average_divisions = cursor;
    cursor = checked_add(cursor, checked_mul(output_coordinates, 9, "final average division coordinates"), "final average division coordinates");
    result.final_logits = cursor;
    cursor = checked_add(cursor, output_coordinates, "final logits coordinates");
    result.end = cursor;
    const std::size_t expected_end = checked_add(layout.segments.at(2).offset,
                                                layout.segments.at(2).logical_length,
                                                "Aux segment end");
    if (cursor != expected_end) throw std::invalid_argument("canonical trace coordinates do not cover the Aux segment");
    return result;
}

struct QueryCoordinates {
    QueryCounts counts;
    std::size_t lrelu = 0;
    std::size_t exp = 0;
    std::size_t elu = 0;
    std::size_t range = 0;
    std::size_t end = 0;
};

QueryCoordinates make_query_coordinates(const Graph &graph, const Model &model,
                                        const InputLayout &layout) {
    QueryCoordinates result;
    result.counts = canonical_query_counts(graph, model);
    std::size_t cursor = layout.segments.at(3).offset;
    result.lrelu = cursor;
    cursor = checked_add(cursor, next_power_of_two(result.counts.lrelu), "LReLU query block");
    result.exp = cursor;
    cursor = checked_add(cursor, next_power_of_two(result.counts.exp), "ExpMap query block");
    result.elu = cursor;
    cursor = checked_add(cursor, next_power_of_two(result.counts.elu), "ELU query block");
    result.range = cursor;
    cursor = checked_add(cursor, next_power_of_two(result.counts.range), "range query block");
    result.end = cursor;
    const std::size_t expected_end = checked_add(layout.segments.at(3).offset,
                                                layout.segments.at(3).logical_length,
                                                "L_qry segment end");
    if (cursor != expected_end) throw std::invalid_argument("canonical query coordinates do not cover L_qry");
    return result;
}

std::string family_name_from_index(std::size_t index) {
    std::ostringstream out;
    out << "family/" << index;
    return out.str();
}

void append_checkpoint(GatCircuitPlan &plan, const std::string &id,
                       const std::string &relation, const std::string &segment,
                       std::size_t coordinates, std::size_t coordinate_offset = 0,
                       const std::string &coordinate_rule = "canonical_row_major") {
    CheckpointDescriptor checkpoint;
    checkpoint.id = id;
    checkpoint.relation_id = relation;
    checkpoint.segment = segment;
    checkpoint.coordinates = coordinates;
    checkpoint.coordinate_offset = coordinate_offset;
    checkpoint.coordinate_rule = coordinate_rule;
    plan.checkpoints.push_back(std::move(checkpoint));
}

void require_input_region(const GatCircuitPlan &plan, std::size_t offset,
                          std::size_t length, const char *what) {
    const std::size_t input_size = plan.circuit.circuit.front().size;
    if (offset > input_size || length > input_size - offset) {
        throw std::invalid_argument(std::string("canonical circuit coordinate exceeds input layer: ") + what);
    }
}

void append_unary_layer(GatCircuitPlan &plan, layerType type, std::size_t output_size,
                        std::size_t input_offset, std::size_t input_stride = 1) {
    if (plan.circuit.circuit.size() >= std::numeric_limits<u8>::max()) throw std::overflow_error("zkGPT circuit has too many layers");
    layer current;
    current.ty = type;
    current.size = as_u32(std::max<std::size_t>(1, output_size), "layer size");
    current.zero_start_id = current.size;
    current.bit_length = bit_length(current.size);
    current.need_phase2 = false;
    const std::size_t count = std::max<std::size_t>(1, output_size);
    if (output_size == 0) require_input_region(plan, input_offset, 1, "empty unary layer");
    else require_input_region(plan, input_offset, checked_add(checked_mul(output_size - 1, input_stride, "unary coordinate stride"), 1, "unary coordinate span"), "unary layer");
    for (std::size_t g = 0; g < count; ++g) {
        const std::size_t source = checked_add(input_offset, checked_mul(g, input_stride, "unary coordinate"), "unary coordinate");
        current.uni_gates.emplace_back(as_u32(g, "unary gate output"), as_u32(source, "unary gate input"), 0, 1);
    }
    plan.circuit.circuit.push_back(std::move(current));
}

void append_binary_layer(GatCircuitPlan &plan, layerType type, std::size_t output_size,
                         std::size_t lhs_offset, std::size_t rhs_offset,
                         std::size_t lhs_stride = 1, std::size_t rhs_stride = 1,
                         std::uint32_t l = 0, ll sc = 1) {
    if (plan.circuit.circuit.size() >= std::numeric_limits<u8>::max()) throw std::overflow_error("zkGPT circuit has too many layers");
    layer current;
    current.ty = type;
    current.size = as_u32(std::max<std::size_t>(1, output_size), "binary layer size");
    current.zero_start_id = current.size;
    current.bit_length = bit_length(current.size);
    current.need_phase2 = false;
    const std::size_t count = std::max<std::size_t>(1, output_size);
    const std::size_t span = output_size == 0 ? 1 : checked_add(checked_mul(output_size - 1, lhs_stride, "binary lhs stride"), 1, "binary lhs span");
    require_input_region(plan, lhs_offset, span, "binary lhs");
    const std::size_t rhs_span = output_size == 0 ? 1 : checked_add(checked_mul(output_size - 1, rhs_stride, "binary rhs stride"), 1, "binary rhs span");
    require_input_region(plan, rhs_offset, rhs_span, "binary rhs");
    for (std::size_t g = 0; g < count; ++g) {
        const std::size_t lhs = checked_add(lhs_offset, checked_mul(g, lhs_stride, "binary lhs coordinate"), "binary lhs coordinate");
        const std::size_t rhs = checked_add(rhs_offset, checked_mul(g, rhs_stride, "binary rhs coordinate"), "binary rhs coordinate");
        current.bin_gates.emplace_back(as_u32(g, "binary gate output"), as_u32(lhs, "binary gate lhs"), as_u32(rhs, "binary gate rhs"), sc, static_cast<u8>(l));
    }
    plan.circuit.circuit.push_back(std::move(current));
}

struct ExplicitUnaryGate {
    std::size_t g = 0;
    std::size_t u = 0;
    u8 lu = 0;
    ll sc = 1;
};

struct ExplicitBinaryGate {
    std::size_t g = 0;
    std::size_t u = 0;
    std::size_t v = 0;
    u8 l = 0;
    ll sc = 1;
};

void append_explicit_layer(GatCircuitPlan &plan, layerType type,
                           std::size_t output_size,
                           const std::vector<ExplicitUnaryGate> &unary,
                           const std::vector<ExplicitBinaryGate> &binary,
                           bool need_phase2 = false) {
    if (plan.circuit.circuit.size() >= std::numeric_limits<u8>::max()) throw std::overflow_error("zkGPT circuit has too many layers");
    layer current;
    current.ty = type;
    current.size = as_u32(std::max<std::size_t>(1, output_size), "explicit layer size");
    current.zero_start_id = current.size;
    current.bit_length = bit_length(current.size);
    current.need_phase2 = need_phase2;
    const std::size_t input_size = plan.circuit.circuit.front().size;
    for (const ExplicitUnaryGate &gate : unary) {
        if (gate.g >= output_size || gate.u >= input_size) throw std::invalid_argument("explicit unary gate coordinate is out of bounds");
        current.uni_gates.emplace_back(as_u32(gate.g, "explicit unary output"), as_u32(gate.u, "explicit unary input"), gate.lu, gate.sc);
    }
    for (const ExplicitBinaryGate &gate : binary) {
        if (gate.g >= output_size || gate.u >= input_size || gate.v >= input_size) throw std::invalid_argument("explicit binary gate coordinate is out of bounds");
        current.bin_gates.emplace_back(as_u32(gate.g, "explicit binary output"), as_u32(gate.u, "explicit binary lhs"), as_u32(gate.v, "explicit binary rhs"), gate.sc, gate.l);
    }
    if (current.uni_gates.empty() && current.bin_gates.empty()) throw std::invalid_argument("explicit layered relation has no gates");
    plan.circuit.circuit.push_back(std::move(current));
}

void append_projection_layer(GatCircuitPlan &plan, const MatMulDescriptor &descriptor) {
    layer current;
    current.ty = layerType::FCONN;
    current.size = as_u32(std::max<std::size_t>(1, descriptor.output_coordinates), "projection layer size");
    current.zero_start_id = current.size;
    current.bit_length = bit_length(current.size);
    current.need_phase2 = true;
    if (descriptor.rows == 0 || descriptor.inner == 0 || descriptor.cols == 0 ||
        descriptor.output_coordinates != checked_mul(descriptor.rows, descriptor.cols, "projection coordinates")) {
        throw std::invalid_argument("canonical projection descriptor has incompatible dimensions");
    }
    require_input_region(plan, descriptor.lhs_offset, checked_mul(descriptor.rows, descriptor.inner, "projection lhs region"), "projection lhs");
    require_input_region(plan, descriptor.rhs_offset, checked_mul(descriptor.inner, descriptor.cols, "projection rhs region"), "projection rhs");
    require_input_region(plan, descriptor.output_offset, descriptor.output_coordinates, "projection output");
    for (std::size_t row = 0; row < descriptor.rows; ++row) {
        for (std::size_t col = 0; col < descriptor.cols; ++col) {
            const std::size_t output_index = checked_add(checked_mul(row, descriptor.cols, "projection output index"), col, "projection output index");
            for (std::size_t term = 0; term < descriptor.inner; ++term) {
                const std::size_t lhs = descriptor.lhs_offset + row * descriptor.inner + term;
                const std::size_t rhs = descriptor.rhs_offset + term * descriptor.cols + col;
                current.bin_gates.emplace_back(as_u32(output_index, "projection output"), as_u32(lhs, "projection lhs"), as_u32(rhs, "projection rhs"), 1, 0);
            }
        }
    }
    plan.circuit.circuit.push_back(std::move(current));
}

std::string plan_encoding(const GatCircuitPlan &plan) {
    std::string encoded;
    append_string(encoded, plan.registry.registry_id);
    append_string(encoded, plan.input_layout.selector);
    append_string(encoded, plan.opening_policy);
    append_u64(encoded, static_cast<std::uint64_t>(plan.terminal_openings.size()));
    for (const std::string &opening : plan.terminal_openings) append_string(encoded, opening);
    append_u64(encoded, plan.input_layout.n_raw);
    append_u64(encoded, plan.input_layout.n_in);
    for (const InputSegment &segment : plan.input_layout.segments) {
        append_string(encoded, segment.name);
        append_u64(encoded, segment.offset);
        append_u64(encoded, segment.logical_length);
        append_u64(encoded, segment.padded_length);
    }
    for (const MatMulDescriptor &descriptor : plan.matmuls) {
        append_string(encoded, descriptor.relation_id);
        append_string(encoded, descriptor.family_id);
        append_string(encoded, descriptor.lhs);
        append_string(encoded, descriptor.rhs);
        append_string(encoded, descriptor.output);
        append_u64(encoded, descriptor.rows);
        append_u64(encoded, descriptor.inner);
        append_u64(encoded, descriptor.cols);
        append_u64(encoded, descriptor.output_coordinates);
        append_u64(encoded, descriptor.lhs_offset);
        append_u64(encoded, descriptor.rhs_offset);
        append_u64(encoded, descriptor.output_offset);
        append_string(encoded, descriptor.coordinate_rule);
        append_string(encoded, descriptor.scale_rule);
    }
    for (const SparseGkrDescriptor &descriptor : plan.sparse_gkr) {
        append_string(encoded, descriptor.relation_id);
        append_string(encoded, descriptor.family_id);
        append_u64(encoded, descriptor.edge_count);
        append_u64(encoded, descriptor.value_width);
        append_u64(encoded, descriptor.input_offset);
        append_u64(encoded, descriptor.source_offset);
        append_u64(encoded, descriptor.destination_offset);
        append_u64(encoded, descriptor.output_offset);
        append_string(encoded, descriptor.coordinate_rule);
        append_string(encoded, descriptor.group_rule);
        for (const std::string &constraint : descriptor.constraints) append_string(encoded, constraint);
    }
    for (const LassoFamilyDescriptor &descriptor : plan.lasso) {
        append_string(encoded, descriptor.relation_id);
        append_string(encoded, descriptor.family_id);
        append_string(encoded, descriptor.table_id);
        append_u64(encoded, descriptor.real_queries);
        append_u64(encoded, descriptor.padded_queries);
        append_u64(encoded, descriptor.table_domain);
        append_int(encoded, descriptor.padding_value);
        append_u64(encoded, descriptor.query_offset);
        append_string(encoded, descriptor.incidence_rule);
        append_string(encoded, descriptor.challenge_policy);
        append_string(encoded, descriptor.query_encoding);
    }
    for (const CheckpointDescriptor &checkpoint : plan.checkpoints) {
        append_string(encoded, checkpoint.id);
        append_string(encoded, checkpoint.relation_id);
        append_string(encoded, checkpoint.segment);
        append_u64(encoded, checkpoint.coordinates);
        append_u64(encoded, checkpoint.coordinate_offset);
        append_string(encoded, checkpoint.coordinate_rule);
    }
    for (const layer &current : plan.circuit.circuit) {
        append_u32(encoded, static_cast<std::uint32_t>(current.ty));
        append_u64(encoded, current.size);
        append_u64(encoded, current.uni_gates.size());
        append_u64(encoded, current.bin_gates.size());
        for (const uniGate &gate : current.uni_gates) {
            append_u32(encoded, gate.g);
            append_u32(encoded, gate.u);
            append_u32(encoded, gate.lu);
            append_int(encoded, gate.sc);
        }
        for (const binGate &gate : current.bin_gates) {
            append_u32(encoded, gate.g);
            append_u32(encoded, gate.u);
            append_u32(encoded, gate.v);
            append_u32(encoded, gate.l);
            append_int(encoded, gate.sc);
        }
    }
    return encoded;
}

bool validate_gat_circuit_plan_impl(const GatCircuitPlan &plan,
                                    std::string *reason);

GatCircuitPlan build_plan_impl(const Graph &graph, const Model &model,
                               const QuantConfig &quant, const ForwardResult *forward) {
    graph.validate();
    model.validate_schema();
    quant.validate();
    if (forward != nullptr && !validate_relation(graph, forward->trace.input, model, quant, *forward)) throw std::invalid_argument("cannot build proof plan from an invalid GAT trace");

    const std::size_t model_size = flatten_model(model, quant).size();
    const std::size_t public_size = checked_mul(graph.num_nodes, model.input_dim(), "public input size");
    const std::size_t aux_size = canonical_aux_size(graph, model);
    const std::size_t query_size = canonical_query_size(graph, model);
    if (forward != nullptr && trace_aux_size(&forward->trace) != aux_size) {
        throw std::invalid_argument("forward trace Aux shape does not match canonical plan coordinates");
    }
    if (forward != nullptr && trace_query_size(&forward->trace) != query_size) {
        throw std::invalid_argument("forward trace lookup shape does not match canonical L_qry coordinates");
    }

    GatCircuitPlan plan;
    plan.registry = RelationRegistry::canonical();
    plan.opening_policy = "same_r_three_terminal_openings_v1";
    plan.terminal_openings = {"W@r", "Aux@r", "L_qry@r"};
    plan.input_layout = make_layout(public_size, model_size, aux_size, query_size);
    const std::size_t circuit_input_size = checked_mul(plan.input_layout.n_in, 4, "four-segment input layer");
    if (circuit_input_size > std::numeric_limits<std::uint32_t>::max()) throw std::overflow_error("zkGPT input layer exceeds official u32 circuit size");
    plan.circuit.init(0, 1);
    plan.circuit.circuit[0].ty = layerType::INPUT;
    plan.circuit.circuit[0].size = as_u32(circuit_input_size, "input layer size");
    plan.circuit.circuit[0].zero_start_id = plan.circuit.circuit[0].size;
    plan.circuit.circuit[0].bit_length = bit_length(circuit_input_size);

    const std::size_t hidden0_dim = model.tensor(hidden_tensor_name(0, 0, "seq/kernel")).shape.dims.at(2);
    const std::size_t hidden1_dim = model.tensor(hidden_tensor_name(1, 0, "seq/kernel")).shape.dims.at(2);
    const std::size_t classes = model.classes();
    const std::size_t first_hidden_projection = checked_mul(graph.num_nodes, hidden0_dim, "hidden projection coordinates");
    const std::size_t second_hidden_projection = checked_mul(graph.num_nodes, hidden1_dim, "hidden projection coordinates");

    const TraceCoordinates coordinates = make_trace_coordinates(graph, model, plan.input_layout);
    const QueryCoordinates query_coordinates = make_query_coordinates(graph, model, plan.input_layout);
    std::size_t w_cursor = plan.input_layout.segments.at(1).offset;
    std::size_t lrelu_query_cursor = query_coordinates.lrelu;
    std::size_t exp_query_cursor = query_coordinates.exp;
    std::size_t elu_query_cursor = query_coordinates.elu;
    std::size_t range_query_cursor = query_coordinates.range;
    const auto add_attention_descriptors = [&plan, &graph](const std::string &family,
                                                            const AttentionCoordinates &a,
                                                            std::size_t output_dim,
                                                            std::size_t seq_w,
                                                            std::size_t src_w,
                                                            std::size_t dst_w,
                                                            std::size_t input_dim,
                                                            std::size_t lhs_offset,
                                                            const std::string &projection_scale,
                                                            const std::string &attention_scale) {
        const std::size_t projection_coordinates = checked_mul(graph.num_nodes, output_dim, "projection coordinates");
        const auto add_matmul = [&plan, &graph](const std::string &relation,
                                                 const std::string &lhs,
                                                 const std::string &rhs,
                                                 const std::string &output,
                                                 std::size_t rows,
                                                 std::size_t inner,
                                                 std::size_t cols,
                                                 std::size_t lhs_offset,
                                                 std::size_t rhs_offset,
                                                 std::size_t output_offset,
                                                 const std::string &scale_rule) {
            MatMulDescriptor descriptor;
            descriptor.relation_id = relation;
            descriptor.family_id = relation;
            descriptor.lhs = lhs;
            descriptor.rhs = rhs;
            descriptor.output = output;
            descriptor.rows = rows;
            descriptor.inner = inner;
            descriptor.cols = cols;
            descriptor.output_coordinates = checked_mul(rows, cols, "MatMul output coordinates");
            descriptor.lhs_offset = lhs_offset;
            descriptor.rhs_offset = rhs_offset;
            descriptor.output_offset = output_offset;
            descriptor.coordinate_rule = "row_major_exact[row,term,col]";
            descriptor.scale_rule = scale_rule;
            plan.matmuls.push_back(descriptor);
            append_projection_layer(plan, descriptor);
        };
        add_matmul(family + "/projection", family + ".input", family + ".seq/kernel", family + ".H_prime",
                   graph.num_nodes, input_dim, output_dim, lhs_offset, seq_w, a.projection, projection_scale);
        add_matmul(family + "/attention/source_dot", family + ".H_prime", family + ".attn_src/kernel", family + ".E_src",
                   graph.num_nodes, output_dim, 1, a.projection, src_w, a.source_score, attention_scale);
        add_matmul(family + "/attention/destination_dot", family + ".H_prime", family + ".attn_dst/kernel", family + ".E_dst",
                   graph.num_nodes, output_dim, 1, a.projection, dst_w, a.destination_score, attention_scale);
    };
    const auto add_score_softmax_layers = [&plan, &graph](const AttentionCoordinates &a,
                                                           std::size_t lrelu_offset,
                                                           std::size_t exp_offset) {
        const std::size_t edges = graph.edges();
        const std::size_t nodes = graph.num_nodes;
        std::vector<ExplicitBinaryGate> score_gates;
        std::vector<ExplicitUnaryGate> score_queries;
        score_gates.reserve(edges);
        score_queries.reserve(edges);
        for (std::size_t edge = 0; edge < edges; ++edge) {
            const std::size_t source = static_cast<std::size_t>(graph.src[edge]);
            const std::size_t destination = static_cast<std::size_t>(graph.dst[edge]);
            score_gates.push_back({edge, a.source_score + source, a.destination_score + destination, 0, 1});
            score_queries.push_back({edge, a.raw_score + edge, 0, 1});
            score_queries.push_back({edge, lrelu_offset + edge, 0, 1});
        }
        append_explicit_layer(plan, layerType::ADD_BIAS, std::max<std::size_t>(1, edges),
                              score_queries, score_gates, false);

        std::vector<ExplicitUnaryGate> softmax_unary;
        std::vector<ExplicitBinaryGate> softmax_binary;
        softmax_unary.reserve(edges * 5 + nodes * 3);
        softmax_binary.reserve(edges);
        for (std::size_t edge = 0; edge < edges; ++edge) {
            const std::size_t destination = static_cast<std::size_t>(graph.dst[edge]);
            softmax_binary.push_back({edge, a.leaky_score + edge, a.group_max + destination, 0, 1});
            softmax_unary.push_back({edge, exp_offset + edge, 0, 1});
            softmax_unary.push_back({edge, a.exp + edge, 0, 1});
            softmax_unary.push_back({edge, a.alpha + edge, 0, 1});
            softmax_unary.push_back({edge, a.max_selector + edge, 0, 1});
        }
        for (std::size_t node = 0; node < nodes; ++node) {
            softmax_unary.push_back({node, a.group_max + node, 0, 1});
            softmax_unary.push_back({node, a.group_sum + node, 0, 1});
            softmax_unary.push_back({node, a.group_inverse + node, 0, 1});
        }
        append_explicit_layer(plan, layerType::SOFTMAX_1,
                              std::max<std::size_t>(1, std::max(edges, nodes)),
                              softmax_unary, softmax_binary, true);
    };
    const auto add_aggregate_layer = [&plan, &graph](const AttentionCoordinates &a,
                                                      bool hidden) {
        const std::size_t edges = graph.edges();
        const std::size_t nodes = graph.num_nodes;
        const std::size_t output_dim = a.output_dim;
        const std::size_t coordinates_count = checked_mul(nodes, output_dim, "aggregate coordinates");
        std::vector<ExplicitUnaryGate> unary;
        std::vector<ExplicitBinaryGate> binary;
        unary.reserve(coordinates_count * 5 + edges * output_dim);
        binary.reserve(edges * output_dim);
        for (std::size_t edge = 0; edge < edges; ++edge) {
            const std::size_t source = static_cast<std::size_t>(graph.src[edge]);
            const std::size_t destination = static_cast<std::size_t>(graph.dst[edge]);
            for (std::size_t col = 0; col < output_dim; ++col) {
                const std::size_t destination_coordinate = destination * output_dim + col;
                const std::size_t source_coordinate = source * output_dim + col;
                binary.push_back({destination_coordinate, a.alpha + edge,
                                  a.projection + source_coordinate, 0, 1});
            }
        }
        for (std::size_t index = 0; index < coordinates_count; ++index) {
            unary.push_back({index, a.integer_numerator + index, 0, 1});
            unary.push_back({index, a.field_numerator + index, 0, 1});
            unary.push_back({index, a.field_aggregate + index, 0, 1});
            unary.push_back({index, a.recovered + index, 0, 1});
            if (hidden) unary.push_back({index, a.activated + index, 0, 1});
        }
        append_explicit_layer(plan, layerType::ADD_BIAS,
                              std::max<std::size_t>(1, coordinates_count), unary, binary, true);
    };
    for (std::size_t layer_id = 0; layer_id < 2; ++layer_id) {
        const std::size_t output_dim = layer_id == 0 ? hidden0_dim : hidden1_dim;
        const std::size_t input_dim = model.tensor(hidden_tensor_name(layer_id, 0, "seq/kernel")).shape.dims.at(1);
        const std::size_t projection_coordinates = checked_mul(graph.num_nodes, output_dim, "hidden projection coordinates");
        for (std::size_t head = 0; head < 8; ++head) {
            const AttentionCoordinates &a = coordinates.hidden[layer_id][head];
            const std::string family = hidden_family(layer_id, head);
            const std::size_t seq_w = w_cursor;
            w_cursor = checked_add(w_cursor, checked_mul(input_dim, output_dim, "hidden sequence weights"), "hidden weight cursor");
            const std::size_t src_w = w_cursor;
            w_cursor = checked_add(w_cursor, output_dim, "hidden source weights");
            const std::size_t dst_w = w_cursor;
            w_cursor = checked_add(w_cursor, output_dim, "hidden destination weights");
            add_attention_descriptors(family, a, output_dim, seq_w, src_w, dst_w, input_dim,
                                      layer_id == 0 ? plan.input_layout.segments.at(0).offset : coordinates.hidden_concat[layer_id - 1],
                                      "s_input*s_w_hidden[layer][head]->s_hprime_hidden[layer][head]",
                                      "s_hprime_hidden[layer][head]*s_a_{src,dst}->s_h");
            append_checkpoint(plan, family + "/projection", family + "/projection", "Aux",
                              projection_coordinates, a.projection, "Aux[row_major_exact[node,col]]");
            append_checkpoint(plan, family + "/source_score", family + "/attention/source_dot", "Aux",
                              graph.num_nodes, a.source_score, "Aux[row_major_exact[node]]");
            append_checkpoint(plan, family + "/destination_score", family + "/attention/destination_dot", "Aux",
                              graph.num_nodes, a.destination_score, "Aux[row_major_exact[node]]");

            SparseGkrDescriptor score;
            score.relation_id = "attention.score/" + family;
            score.family_id = family;
            score.edge_count = graph.edges();
            score.value_width = output_dim;
            score.input_offset = a.raw_score;
            score.source_offset = a.source_score;
            score.destination_offset = a.destination_score;
            score.output_offset = a.raw_score;
            score.coordinate_rule = "edge_gid_then_dst_stable[e].src=src[e],dst=dst[e]";
            score.group_rule = "Q_grp_new[e] at group start; Q_grp_end[e] at group end";
            score.constraints = {
                "E_src[node]=dot(H_prime[node,row],a_src[row])",
                "E_dst[node]=dot(H_prime[node,row],a_dst[row])",
                "raw_score[e]=E_src[src[e]]+E_dst[dst[e]]",
                "Q_grp_new[e]=(e==edge_ptr[gid] or dst[e]!=dst[e-1])",
                "Q_grp_end[e]=(e+1==edge_ptr[gid+1] or dst[e+1]!=dst[e])",
                "leaky_score[e]=T_LReLU(raw_score[e])"
            };
            plan.sparse_gkr.push_back(score);
            const std::size_t lrelu_offset = lrelu_query_cursor;
            lrelu_query_cursor = checked_add(lrelu_query_cursor, graph.edges(), "LeakyReLU query cursor");
            add_score_softmax_layers(a, lrelu_offset, exp_query_cursor);
            append_checkpoint(plan, score.relation_id + ".raw_score", score.relation_id, "Aux", graph.edges(),
                              a.raw_score, "Aux[row_major_exact[edge]]");

            SparseGkrDescriptor softmax;
            softmax.relation_id = "attention.softmax/" + family;
            softmax.family_id = family;
            softmax.edge_count = graph.edges();
            softmax.value_width = output_dim;
            softmax.input_offset = a.leaky_score;
            softmax.source_offset = a.group_max;
            softmax.destination_offset = a.group_sum;
            softmax.output_offset = a.alpha;
            softmax.coordinate_rule = "edge_gid_then_dst_stable[e].destination_group=dst[e]";
            softmax.group_rule = "Q_grp_new at destination-group start; Q_grp_end at destination-group end";
            softmax.constraints = {
                "group_max[dst[e]]=max(leaky_score[group])",
                "delta[e]=group_max[dst[e]]-leaky_score[e]",
                "exp[e]=T_exp(delta[e])",
                "group_sum[node]=sum(exp[e]) over incoming canonical edges",
                "group_inverse[node]*group_sum[node]=1 in BN254",
                "max_selector[e] in {0,1} and sum_group(max_selector)=1",
                "alpha[e]=exp[e]*group_inverse[dst[e]]",
                "Q_grp_new and Q_grp_end recurrence is explicit"
            };
            plan.sparse_gkr.push_back(softmax);
            const std::size_t exp_offset = exp_query_cursor;
            exp_query_cursor = checked_add(exp_query_cursor, graph.edges(), "ExpMap query cursor");
            append_checkpoint(plan, softmax.relation_id + ".delta", softmax.relation_id, "Aux", graph.edges(),
                              a.delta, "Aux[row_major_exact[edge]]");
            append_checkpoint(plan, softmax.relation_id + ".group_inverse", softmax.relation_id, "Aux", graph.num_nodes,
                              a.group_inverse, "Aux[row_major_exact[node]]");
            append_checkpoint(plan, softmax.relation_id + ".alpha", softmax.relation_id, "Aux", graph.edges(),
                              a.alpha, "Aux[row_major_exact[edge]]");

            SparseGkrDescriptor aggregate;
            aggregate.relation_id = "aggregation.hidden/" + family;
            aggregate.family_id = family;
            aggregate.edge_count = graph.edges();
            aggregate.value_width = output_dim;
            aggregate.input_offset = a.alpha;
            aggregate.source_offset = a.projection;
            aggregate.destination_offset = a.integer_numerator;
            aggregate.output_offset = a.recovered;
            aggregate.coordinate_rule = "edge_gid_then_dst_stable[e].src=src[e],dst=dst[e],col-major-inner";
            aggregate.group_rule = "Q_grp_new/Q_grp_end prefix recurrence with destination broadcast";
            aggregate.constraints = {
                "Q_grp_new and Q_grp_end drive destination group recurrence",
                "integer_numerator[node,col]=sum(exp[e]*H_prime[src[e],col])",
                "field_numerator=integer_numerator embedded in BN254",
                "field_aggregate[node,col]*group_sum[node]=field_numerator[node,col]",
                "field_aggregate and integer_numerator have dual aggregate consistency",
                "recovered=SignedRoundDiv(s_agg_pre*integer_numerator,s_hprime*group_sum)",
                "destination gather uses canonical src[e],dst[e] coordinates"
            };
            plan.sparse_gkr.push_back(aggregate);
            add_aggregate_layer(a, true);
            append_checkpoint(plan, aggregate.relation_id + ".integer_numerator", aggregate.relation_id, "Aux", projection_coordinates,
                              a.integer_numerator, "Aux[row_major_exact[node,col]]");
            append_checkpoint(plan, aggregate.relation_id + ".field_aggregate", aggregate.relation_id, "Aux", projection_coordinates,
                              a.field_aggregate, "Aux[row_major_exact[node,col]]");
            append_checkpoint(plan, aggregate.relation_id + ".recovered", aggregate.relation_id, "Aux", projection_coordinates,
                              a.recovered, "Aux[row_major_exact[node,col]]");

            std::vector<ExplicitUnaryGate> round_unary;
            const std::size_t round_queries = checked_mul(projection_coordinates, 8, "hidden range query wiring");
            round_unary.reserve(round_queries + projection_coordinates * 3);
            for (std::size_t index = 0; index < projection_coordinates; ++index) {
                round_unary.push_back({index, a.recovered + index, 0, 1});
                round_unary.push_back({index, a.integer_numerator + index, 0, 1});
                round_unary.push_back({index, a.divisions + index * 9, 0, 1});
            }
            for (std::size_t index = 0; index < round_queries; ++index) round_unary.push_back({index, range_query_cursor + index, 0, 1});
            append_explicit_layer(plan, layerType::RELU, std::max<std::size_t>(1, round_queries), round_unary, {}, true);
            range_query_cursor = checked_add(range_query_cursor, round_queries, "hidden range query cursor");
            append_checkpoint(plan, "rounding/hidden/" + family, "rounding", "Aux", checked_mul(projection_coordinates, 9, "hidden division coordinates"),
                              a.divisions, "Aux[row_major_exact[node,col,witness_component]]");

            std::vector<ExplicitUnaryGate> elu_unary;
            elu_unary.reserve(projection_coordinates * 2);
            for (std::size_t index = 0; index < projection_coordinates; ++index) {
                elu_unary.push_back({index, a.recovered + index, 0, 1});
                elu_unary.push_back({index, elu_query_cursor + index, 0, 1});
            }
            append_explicit_layer(plan, layerType::RELU, std::max<std::size_t>(1, projection_coordinates), elu_unary, {}, false);
            elu_query_cursor = checked_add(elu_query_cursor, projection_coordinates, "ELU query cursor");
            append_checkpoint(plan, "activation.hidden/" + family, "activation.hidden", "Aux", projection_coordinates,
                              a.activated, "Aux[row_major_exact[node,col]]");
        }
        const std::size_t concat_coordinates = checked_mul(graph.num_nodes, checked_mul(output_dim, 8, "concat width"), "concat coordinates");
        std::vector<ExplicitUnaryGate> concat_unary;
        concat_unary.reserve(concat_coordinates);
        for (std::size_t node = 0; node < graph.num_nodes; ++node) {
            for (std::size_t head = 0; head < 8; ++head) {
                const AttentionCoordinates &a = coordinates.hidden[layer_id][head];
                for (std::size_t col = 0; col < output_dim; ++col) {
                    const std::size_t output_index = node * output_dim * 8 + head * output_dim + col;
                    concat_unary.push_back({output_index, a.activated + node * output_dim + col, 0, 1});
                }
            }
        }
        append_explicit_layer(plan, layerType::ADD_BIAS, std::max<std::size_t>(1, concat_coordinates), concat_unary, {}, false);
        append_checkpoint(plan, "concat/hidden/layer" + std::to_string(layer_id), "concat", "Aux", concat_coordinates,
                          coordinates.hidden_concat[layer_id], "Aux[row_major_exact[node,head,col]]");
    }

    for (std::size_t head = 0; head < 2; ++head) {
        const std::string family = output_family(head);
        const AttentionCoordinates &a = coordinates.output[head];
        const std::size_t projection_coordinates = checked_mul(graph.num_nodes, classes, "output projection coordinates");
        const std::size_t input_dim = model.tensor(output_tensor_name(head, "seq/kernel")).shape.dims.at(1);
        const std::size_t seq_w = w_cursor;
        w_cursor = checked_add(w_cursor, checked_mul(input_dim, classes, "output sequence weights"), "output weight cursor");
        const std::size_t src_w = w_cursor;
        w_cursor = checked_add(w_cursor, classes, "output source weights");
        const std::size_t dst_w = w_cursor;
        w_cursor = checked_add(w_cursor, classes, "output destination weights");
        const std::size_t bias_w = w_cursor;
        w_cursor = checked_add(w_cursor, classes, "output bias weights");
        MatMulDescriptor projection;
        projection.relation_id = "projection.output/head" + std::to_string(head);
        projection.family_id = family;
        projection.lhs = "Aux.H_cat";
        projection.rhs = family + ".seq/kernel";
        projection.output = family + ".Y_prime";
        projection.rows = graph.num_nodes;
        projection.inner = input_dim;
        projection.cols = classes;
        projection.output_coordinates = projection_coordinates;
        projection.lhs_offset = coordinates.hidden_concat[1];
        projection.rhs_offset = seq_w;
        projection.output_offset = a.projection;
        projection.coordinate_rule = "row_major_exact[row,term,col]";
        projection.scale_rule = "s_cat*s_w_output[head]->s_yprime_output[head]";
        plan.matmuls.push_back(projection);
        append_projection_layer(plan, projection);
        append_checkpoint(plan, projection.relation_id + ".Y_prime", projection.relation_id, "Aux", projection_coordinates,
                          a.projection, "Aux[row_major_exact[node,col]]");
        const std::string attention_scale = "s_yprime_output[head]*s_a_{src,dst}->s_out";
        const auto add_dot = [&plan, &graph, &family](const std::string &suffix,
                                                       std::size_t lhs_offset,
                                                       const std::string &rhs,
                                                       const std::string &output,
                                                       std::size_t rhs_offset,
                                                       std::size_t output_offset,
                                                       std::size_t input_dim,
                                                       const std::string &scale) {
            MatMulDescriptor descriptor;
            descriptor.relation_id = family + "/attention/" + suffix;
            descriptor.family_id = family;
            descriptor.lhs = family + ".Y_prime";
            descriptor.rhs = rhs;
            descriptor.output = output;
            descriptor.rows = graph.num_nodes;
            descriptor.inner = input_dim;
            descriptor.cols = 1;
            descriptor.output_coordinates = graph.num_nodes;
            descriptor.lhs_offset = lhs_offset;
            descriptor.rhs_offset = rhs_offset;
            descriptor.output_offset = output_offset;
            descriptor.coordinate_rule = "row_major_exact[row,term,col]";
            descriptor.scale_rule = scale;
            plan.matmuls.push_back(descriptor);
            append_projection_layer(plan, descriptor);
            append_checkpoint(plan, descriptor.relation_id, descriptor.relation_id, "Aux", graph.num_nodes,
                              output_offset, "Aux[row_major_exact[node]]");
        };
        add_dot("source_dot", a.projection, family + ".attn_src/kernel", family + ".E_src", src_w,
                a.source_score, classes, attention_scale);
        add_dot("destination_dot", a.projection, family + ".attn_dst/kernel", family + ".E_dst", dst_w,
                a.destination_score, classes, attention_scale);
        SparseGkrDescriptor score;
        score.relation_id = "attention.score/" + family;
        score.family_id = family;
        score.edge_count = graph.edges();
        score.value_width = classes;
        score.input_offset = a.raw_score;
        score.source_offset = a.source_score;
        score.destination_offset = a.destination_score;
        score.output_offset = a.raw_score;
        score.coordinate_rule = "edge_gid_then_dst_stable[e].src=src[e],dst=dst[e]";
        score.group_rule = "Q_grp_new[e] at group start; Q_grp_end[e] at group end";
        score.constraints = {"raw_score[e]=E_src[src[e]]+E_dst[dst[e]]", "Q_grp_new/Q_grp_end canonical recurrence", "leaky_score[e]=T_LReLU(raw_score[e])"};
        plan.sparse_gkr.push_back(score);
        const std::size_t lrelu_offset = lrelu_query_cursor;
        lrelu_query_cursor = checked_add(lrelu_query_cursor, graph.edges(), "output LeakyReLU query cursor");
        add_score_softmax_layers(a, lrelu_offset, exp_query_cursor);
        exp_query_cursor = checked_add(exp_query_cursor, graph.edges(), "output ExpMap query cursor");
        SparseGkrDescriptor softmax;
        softmax.relation_id = "attention.softmax/" + family;
        softmax.family_id = family;
        softmax.edge_count = graph.edges();
        softmax.value_width = classes;
        softmax.input_offset = a.leaky_score;
        softmax.source_offset = a.group_max;
        softmax.destination_offset = a.group_sum;
        softmax.output_offset = a.alpha;
        softmax.coordinate_rule = "edge_gid_then_dst_stable[e].destination_group=dst[e]";
        softmax.group_rule = "Q_grp_new/Q_grp_end destination-group recurrence";
        softmax.constraints = {"group_max=max(leaky_score) by dst", "delta=group_max-leaky_score", "exp=T_exp(delta)", "group_sum=sum(exp)", "group_inverse*group_sum=1", "max_selector one-hot by group", "alpha=exp*group_inverse[dst]", "Q_grp_new/Q_grp_end destination-group recurrence"};
        plan.sparse_gkr.push_back(softmax);
        append_checkpoint(plan, softmax.relation_id + ".delta", softmax.relation_id, "Aux", graph.edges(), a.delta,
                          "Aux[row_major_exact[edge]]");
        append_checkpoint(plan, softmax.relation_id + ".alpha", softmax.relation_id, "Aux", graph.edges(), a.alpha,
                          "Aux[row_major_exact[edge]]");
        SparseGkrDescriptor aggregate;
        aggregate.relation_id = "aggregation.output/" + family;
        aggregate.family_id = family;
        aggregate.edge_count = graph.edges();
        aggregate.value_width = classes;
        aggregate.input_offset = a.alpha;
        aggregate.source_offset = a.projection;
        aggregate.destination_offset = a.integer_numerator;
        aggregate.output_offset = a.recovered;
        aggregate.coordinate_rule = "edge_gid_then_dst_stable[e].src=src[e],dst=dst[e],col-major-inner";
        aggregate.group_rule = "Q_grp_new/Q_grp_end prefix recurrence with destination broadcast";
        aggregate.constraints = {"integer numerator=sum(exp*Y_prime[src])", "field_numerator=integer_numerator", "field_aggregate*group_sum=field_numerator", "recovered=SignedRoundDiv(s_out_pre*integer_numerator,s_yprime*group_sum)", "dual integer/field aggregate consistency"};
        plan.sparse_gkr.push_back(aggregate);
        add_aggregate_layer(a, false);
        append_checkpoint(plan, aggregate.relation_id + ".recovered", aggregate.relation_id, "Aux", projection_coordinates,
                          a.recovered, "Aux[row_major_exact[node,col]]");
        std::vector<ExplicitUnaryGate> output_unary;
        std::vector<ExplicitBinaryGate> output_binary;
        output_unary.reserve(projection_coordinates * 5);
        output_binary.reserve(projection_coordinates);
        for (std::size_t node = 0; node < graph.num_nodes; ++node) {
            for (std::size_t col = 0; col < classes; ++col) {
                const std::size_t index = node * classes + col;
                output_unary.push_back({index, a.recovered + index, 0, 1});
                output_unary.push_back({index, a.divisions + index * 9, 0, 1});
                output_unary.push_back({index, coordinates.output_alignment_divisions[head] + index * 9, 0, 1});
                output_unary.push_back({index, coordinates.output_aggregate[head] + index, 0, 1});
                output_unary.push_back({index, coordinates.output_aligned[head] + index, 0, 1});
                output_unary.push_back({index, coordinates.output_logits[head] + index, 0, 1});
                const std::size_t bias_index = col;
                output_binary.push_back({index, coordinates.output_aligned[head] + index, bias_w + bias_index, 0, 1});
            }
        }
        const std::size_t output_range_queries = checked_mul(projection_coordinates, 12, "output range query wiring");
        for (std::size_t index = 0; index < output_range_queries; ++index) output_unary.push_back({index, range_query_cursor + index, 0, 1});
        // Range/division query wires are independent GKR outputs.  Their
        // coordinates must be part of the native layer domain even when they
        // outnumber the public-logit coordinates.
        append_explicit_layer(plan, layerType::ADD_BIAS,
                              std::max<std::size_t>(projection_coordinates, output_range_queries),
                              output_unary, output_binary, true);
        range_query_cursor = checked_add(range_query_cursor, output_range_queries, "output range query cursor");
        append_checkpoint(plan, "rounding/output/" + family, "rounding", "Aux", checked_mul(projection_coordinates, 9, "output division coordinates"),
                          a.divisions, "Aux[row_major_exact[node,col,witness_component]]");
        append_checkpoint(plan, "output/aggregate/" + family, "aggregation.output", "Aux", projection_coordinates,
                          coordinates.output_aggregate[head], "Aux[row_major_exact[node,col]]");
        append_checkpoint(plan, "output/aligned/" + family, "rounding", "Aux", projection_coordinates,
                          coordinates.output_aligned[head], "Aux[row_major_exact[node,col]]");
        append_checkpoint(plan, "output/bias/" + family, "output.public_logits", "W", classes, bias_w,
                          "W[row_major_exact[class]]");
        append_checkpoint(plan, "output/logits/" + family, "output.public_logits", "Aux", projection_coordinates,
                          coordinates.output_logits[head], "Aux[row_major_exact[node,col]]");
    }
    const std::size_t output_coordinates = checked_mul(graph.num_nodes, classes, "final output coordinates");
    std::vector<ExplicitBinaryGate> head_sum_binary;
    head_sum_binary.reserve(output_coordinates);
    for (std::size_t index = 0; index < output_coordinates; ++index) {
        head_sum_binary.push_back({index, coordinates.output_logits[0] + index,
                                   coordinates.output_logits[1] + index, 0, 1});
    }
    append_explicit_layer(plan, layerType::ADD_BIAS, std::max<std::size_t>(1, output_coordinates), {}, head_sum_binary, true);
    append_checkpoint(plan, "output/head_sum", "output.public_logits", "Aux", output_coordinates,
                      coordinates.head_sum, "Aux[row_major_exact[node,class]]");
    std::vector<ExplicitUnaryGate> average_unary;
    average_unary.reserve(output_coordinates * 3);
    const std::size_t average_range_queries = checked_mul(output_coordinates, 4, "average range query wiring");
    for (std::size_t index = 0; index < output_coordinates; ++index) {
        average_unary.push_back({index, coordinates.head_sum + index, 0, 1});
        average_unary.push_back({index, coordinates.final_average_divisions + index * 9, 0, 1});
    }
    for (std::size_t index = 0; index < average_range_queries; ++index) average_unary.push_back({index, range_query_cursor + index, 0, 1});
    append_explicit_layer(plan, layerType::RELU, std::max<std::size_t>(1, average_range_queries), average_unary, {}, true);
    range_query_cursor = checked_add(range_query_cursor, average_range_queries, "average range query cursor");
    append_checkpoint(plan, "output/final_average", "rounding", "Aux", checked_mul(output_coordinates, 9, "average division coordinates"),
                      coordinates.final_average_divisions, "Aux[row_major_exact[node,class,witness_component]]");
    std::vector<ExplicitUnaryGate> final_unary;
    final_unary.reserve(output_coordinates);
    for (std::size_t index = 0; index < output_coordinates; ++index) final_unary.push_back({index, coordinates.final_logits + index, 0, 1});
    append_explicit_layer(plan, layerType::ADD_BIAS, std::max<std::size_t>(1, output_coordinates), final_unary, {}, false);
    append_checkpoint(plan, "output/public_logits", "output.public_logits", "Aux", output_coordinates,
                      coordinates.final_logits, "Aux[row_major_exact[node,class]]");

    const GatTrace *trace = forward == nullptr ? nullptr : &forward->trace;
    const StaticTables canonical_tables = StaticTables::build(quant);
    const QueryCounts query_counts = query_coordinates.counts;
    const auto add_lasso = [&plan](const std::string &family, const std::string &table,
                                   std::size_t real, std::size_t offset,
                                   std::size_t domain, Int padding) {
        LassoFamilyDescriptor descriptor;
        descriptor.relation_id = "lasso/" + family;
        descriptor.family_id = family;
        descriptor.table_id = table;
        descriptor.real_queries = real;
        descriptor.padded_queries = next_power_of_two(real);
        descriptor.table_domain = next_power_of_two(domain);
        descriptor.padding_value = padding;
        descriptor.query_offset = offset;
        descriptor.incidence_rule = "L_qry/" + family + "/canonical_operation_index_row_major";
        descriptor.challenge_policy = "LC_before_challenge";
        descriptor.query_encoding = "lookup_tuple_encoding_id:packed_signed_pair_v1";
        plan.lasso.push_back(descriptor);
        append_checkpoint(plan, descriptor.relation_id, descriptor.relation_id, "L_qry", descriptor.padded_queries,
                          offset, "L_qry[row_major_exact[query]]");
    };
    add_lasso("function/lrelu", "T_LReLU", query_counts.lrelu, query_coordinates.lrelu,
              canonical_tables.lrelu.size(), canonical_tables.lrelu.front().x);
    add_lasso("function/exp", "T_exp", query_counts.exp, query_coordinates.exp,
              canonical_tables.exp.size(), canonical_tables.exp.front().x);
    add_lasso("function/elu", "T_ELU", query_counts.elu, query_coordinates.elu,
              canonical_tables.elu.size(), canonical_tables.elu.front().x);
    add_lasso("range/all", "T_range", query_counts.range, query_coordinates.range, 2, 0);
    if (trace != nullptr && trace->tables.digest != canonical_tables.digest) throw std::invalid_argument("forward trace static table digest does not match canonical plan tables");
    if (w_cursor != checked_add(plan.input_layout.segments.at(1).offset,
                                plan.input_layout.segments.at(1).logical_length, "model weight end")) {
        throw std::invalid_argument("canonical model weight coordinates do not cover W segment");
    }
    if (lrelu_query_cursor != checked_add(query_coordinates.lrelu, query_counts.lrelu, "LeakyReLU query end") ||
        exp_query_cursor != checked_add(query_coordinates.exp, query_counts.exp, "ExpMap query end") ||
        elu_query_cursor != checked_add(query_coordinates.elu, query_counts.elu, "ELU query end") ||
        range_query_cursor != checked_add(query_coordinates.range, query_counts.range, "range query end")) {
        throw std::invalid_argument("canonical L_qry wiring does not cover all query coordinates");
    }

    plan.native_api = {
        "layeredCircuit::initSubset",
        "prover::sumcheckInitAll / sumcheckInit / sumcheckInitPhase1 / sumcheckUpdate1 / sumcheckFinalize1",
        "prover::sumcheckInitPhase2 / sumcheckUpdate2 / sumcheckFinalize2",
        "range_prover::sumcheck_deg1 / range_prover::sumcheck_deg3",
        "perdersen_commit / hyrax::open",
        "verifier::verifyGKR / verifier::verifyLasso / verifier::openCommit"
    };
    plan.circuit.size = static_cast<u8>(plan.circuit.circuit.size());
    // Every gate uses a canonical input-layer coordinate, so the official
    // subset preparation is safe and produces the real layered metadata.
    plan.circuit.initSubset();
    plan.circuit_id = "R_GAT-zkGPT/" + hash_hex(plan_encoding(plan));
    plan.plan_digest = hash_hex(plan_encoding(plan));
    plan.official_layered = true;
    plan.has_placeholder = false;
    if (plan.registry.descriptors.empty() || !plan.registry.covers_all_segments()) plan.official_layered = false;
    std::string plan_reason;
    if (!validate_gat_circuit_plan_impl(plan, &plan_reason)) throw std::invalid_argument("canonical GAT proof plan validation failed: " + plan_reason);
    return plan;
}

bool validate_gat_circuit_plan_impl(const GatCircuitPlan &plan, std::string *reason) {
    const auto fail = [reason](const std::string &message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    try {
        if (!plan.official_layered || plan.has_placeholder) return fail("plan is not an official layered circuit");
        if (plan.registry.descriptors.empty() || !plan.registry.covers_all_segments()) return fail("relation registry does not cover the canonical relations");
        if (!canonical_layout_valid(plan.input_layout)) return fail("input layout is not the canonical four-segment layout");
        if (plan.opening_policy != "same_r_three_terminal_openings_v1" ||
            plan.terminal_openings != std::vector<std::string>{"W@r", "Aux@r", "L_qry@r"}) {
            return fail("terminal opening policy is not same-r W/Aux/L_qry");
        }
        if (plan.circuit.circuit.size() <= 1 || plan.circuit.size != plan.circuit.circuit.size() ||
            plan.circuit.circuit.front().ty != layerType::INPUT || plan.circuit.circuit.front().size == 0) {
            return fail("layered circuit has an invalid input or layer count");
        }
        const std::size_t input_size = plan.circuit.circuit.front().size;
        for (std::size_t layer_id = 1; layer_id < plan.circuit.circuit.size(); ++layer_id) {
            const layer &current = plan.circuit.circuit[layer_id];
            if (current.ty == layerType::PLACE_HOLDER || current.size == 0) return fail("placeholder or empty circuit layer");
            for (const uniGate &gate : current.uni_gates) {
                if (gate.g >= current.size || gate.u >= input_size) return fail("unary gate coordinate is out of bounds");
            }
            for (const binGate &gate : current.bin_gates) {
                if (gate.g >= current.size || gate.u >= input_size || gate.v >= input_size) return fail("binary gate coordinate is out of bounds");
            }
            if (current.uni_gates.empty() && current.bin_gates.empty()) return fail("empty canonical circuit layer");
        }
        const auto require_span = [input_size](std::size_t offset, std::size_t length) {
            return offset <= input_size && length <= input_size - offset;
        };
        for (const MatMulDescriptor &descriptor : plan.matmuls) {
            if (descriptor.relation_id.empty() || descriptor.rows == 0 || descriptor.inner == 0 || descriptor.cols == 0 ||
                descriptor.output_coordinates != checked_mul(descriptor.rows, descriptor.cols, "plan MatMul output") ||
                !require_span(descriptor.lhs_offset, checked_mul(descriptor.rows, descriptor.inner, "plan MatMul lhs")) ||
                !require_span(descriptor.rhs_offset, checked_mul(descriptor.inner, descriptor.cols, "plan MatMul rhs")) ||
                !require_span(descriptor.output_offset, descriptor.output_coordinates) ||
                descriptor.coordinate_rule.find("row_major_exact") == std::string::npos ||
                descriptor.scale_rule.empty() || descriptor.native_path.empty()) {
                return fail("MatMul descriptor shape, coordinate, or scale binding is invalid");
            }
        }
        if (plan.matmuls.empty()) return fail("canonical plan has no MatMul descriptors");
        for (const SparseGkrDescriptor &descriptor : plan.sparse_gkr) {
            if (descriptor.relation_id.empty() || descriptor.edge_count == 0 || descriptor.value_width == 0 ||
                !require_span(descriptor.input_offset, 1) || !require_span(descriptor.source_offset, 1) ||
                !require_span(descriptor.destination_offset, 1) || !require_span(descriptor.output_offset, 1) ||
                descriptor.coordinate_rule.find("edge_gid_then_dst_stable") == std::string::npos ||
                descriptor.group_rule.find("Q_grp_new") == std::string::npos ||
                descriptor.group_rule.find("Q_grp_end") == std::string::npos || descriptor.constraints.empty()) {
                return fail("sparse GKR descriptor is not canonically bound");
            }
            bool has_group_recurrence = false;
            bool has_field_relation = false;
            bool has_round_relation = false;
            for (const std::string &constraint : descriptor.constraints) {
                if (constraint.find("Q_grp_new") != std::string::npos && constraint.find("Q_grp_end") != std::string::npos) has_group_recurrence = true;
                if (constraint.find("field") != std::string::npos) has_field_relation = true;
                if (constraint.find("SignedRoundDiv") != std::string::npos) has_round_relation = true;
                if (constraint.find("modulo") != std::string::npos || constraint.find("neighbor") != std::string::npos) return fail("sparse descriptor contains a placeholder coordinate rule");
            }
            if (descriptor.relation_id.find("softmax") != std::string::npos && !has_group_recurrence) return fail("softmax descriptor lacks group recurrence");
            if (descriptor.relation_id.find("aggregation") != std::string::npos && (!has_field_relation || !has_round_relation)) return fail("aggregation descriptor lacks field or SignedRoundDiv relation");
        }
        if (plan.sparse_gkr.empty()) return fail("canonical plan has no sparse GKR descriptors");
        const InputSegment &query_segment = plan.input_layout.segments.at(3);
        std::size_t query_cursor = query_segment.offset;
        if (plan.lasso.size() != 4) return fail("canonical plan must contain four Lasso families");
        for (const LassoFamilyDescriptor &descriptor : plan.lasso) {
            if (descriptor.relation_id.empty() || descriptor.real_queries == 0 ||
                descriptor.padded_queries != next_power_of_two(descriptor.real_queries) ||
                descriptor.table_domain == 0 || (descriptor.table_domain & (descriptor.table_domain - 1)) != 0 ||
                descriptor.query_offset != query_cursor ||
                !require_span(descriptor.query_offset, descriptor.padded_queries) ||
                descriptor.query_offset + descriptor.padded_queries > query_segment.offset + query_segment.logical_length ||
                descriptor.incidence_rule.find("L_qry/") != 0 ||
                descriptor.incidence_rule.find("canonical_operation_index_row_major") == std::string::npos ||
                descriptor.challenge_policy != "LC_before_challenge" || descriptor.query_encoding.empty()) {
                return fail("Lasso query block is not canonical or not contiguous");
            }
            query_cursor = checked_add(query_cursor, descriptor.padded_queries, "validated Lasso query cursor");
        }
        if (query_cursor != query_segment.offset + query_segment.logical_length) return fail("Lasso blocks do not cover L_qry logical length");
        std::set<std::string> checkpoint_ids;
        const auto find_segment = [&plan](const std::string &name) -> const InputSegment * {
            for (const InputSegment &segment : plan.input_layout.segments) if (segment.name == name) return &segment;
            return nullptr;
        };
        if (plan.checkpoints.empty()) return fail("canonical plan has no checkpoints");
        for (const CheckpointDescriptor &checkpoint : plan.checkpoints) {
            if (checkpoint.id.empty() || !checkpoint_ids.insert(checkpoint.id).second || checkpoint.coordinates == 0 ||
                checkpoint.coordinate_rule.find("row_major_exact") == std::string::npos) return fail("checkpoint identity or coordinate rule is invalid");
            const InputSegment *segment = find_segment(checkpoint.segment);
            if (segment == nullptr || checkpoint.coordinate_offset < segment->offset ||
                checkpoint.coordinates > segment->logical_length ||
                checkpoint.coordinate_offset - segment->offset > segment->logical_length - checkpoint.coordinates ||
                checkpoint.coordinate_offset + checkpoint.coordinates > segment->offset + segment->logical_length) return fail("checkpoint exceeds its canonical segment");
        }
        const std::string digest = hash_hex(plan_encoding(plan));
        if (plan.plan_digest != digest || plan.circuit_id != "R_GAT-zkGPT/" + digest) return fail("plan digest does not bind canonical descriptors and wiring");
        return true;
    } catch (const std::exception &error) {
        return fail(error.what());
    }
}

std::string commitment_encoding(const CommitmentWire &commitment) {
    std::string encoded;
    append_string(encoded, commitment.scheme);
    append_string(encoded, commitment.basis_id);
    append_u64(encoded, commitment.logical_size);
    append_u64(encoded, commitment.padded_size);
    append_string(encoded, commitment.point_hex);
    return encoded;
}

void append_registration_encoding(std::string &encoded,
                                  const ModelRegistration &registration) {
    append_string(encoded, registration.dataset);
    append_string(encoded, registration.model_arch_id);
    append_string(encoded, registration.quant_cfg_id);
    append_string(encoded, registration.static_tables_id);
    append_string(encoded, registration.field_id);
    append_string(encoded, registration.field_modulus);
    append_string(encoded, registration.manifest_digest);
    append_string(encoded, registration.model_digest);
    append_string(encoded, registration.bounds_digest);
    append_string(encoded, registration.quant_digest);
    append_string(encoded, registration.layout_digest);
    append_string(encoded, registration.selector);
    append_u64(encoded, static_cast<std::uint64_t>(registration.n_raw));
    append_u64(encoded, static_cast<std::uint64_t>(registration.n_in));
    append_u64(encoded, static_cast<std::uint64_t>(registration.input_segments.size()));
    for (const InputSegment &segment : registration.input_segments) {
        append_string(encoded, segment.name);
        append_u64(encoded, static_cast<std::uint64_t>(segment.offset));
        append_u64(encoded, static_cast<std::uint64_t>(segment.logical_length));
        append_u64(encoded, static_cast<std::uint64_t>(segment.padded_length));
    }
    append_string(encoded, registration.opening_policy);
    append_string(encoded, registration.binding_digest);
    append_string(encoded, registration.registration_digest);
    append_string(encoded, registration.status);
    append_string(encoded, commitment_encoding(registration.com_W));
}

std::string section_encoding(const ProofSection &section) {
    std::string encoded;
    append_string(encoded, section.protocol);
    append_u64(encoded, section.messages.size());
    for (const TranscriptMessage &message : section.messages) {
        append_string(encoded, message.label);
        append_string(encoded, message.payload);
    }
    return encoded;
}

std::string artifact_transcript_digest(const GatProofArtifact &artifact) {
    FiatShamirTranscript transcript("zkgpt-gat-fs-v1");
    transcript.absorb("statement", artifact.statement_digest);
    transcript.absorb("com_aux", commitment_encoding(artifact.com_aux));
    transcript.absorb("com_L", commitment_encoding(artifact.com_L));
    transcript.absorb("Pi_Lasso", section_encoding(artifact.Pi_Lasso));
    transcript.absorb("Pi_MatMul", section_encoding(artifact.Pi_MatMul));
    transcript.absorb("Pi_GKR", section_encoding(artifact.Pi_GKR));
    transcript.absorb("Pi_combine", section_encoding(artifact.Pi_combine));
    transcript.absorb("Pi_open", section_encoding(artifact.Pi_open));
    (void)transcript.challenge("lasso");
    (void)transcript.challenge("matmul");
    (void)transcript.challenge("gkr");
    (void)transcript.challenge("combine");
    (void)transcript.challenge("open");
    return transcript.digest();
}

std::string context_encoding(const VerifierContext &context) {
    std::string encoded;
    append_string(encoded, context.relation_id);
    append_string(encoded, context.model_arch_id);
    append_string(encoded, context.privacy_extension_profile_id);
    append_string(encoded, context.quant_cfg_id);
    append_string(encoded, context.static_tables_id);
    append_registration_encoding(encoded, context.registration);
    append_string(encoded, commitment_encoding(context.com_W));
    append_string(encoded, context.plan.circuit_id);
    append_string(encoded, context.plan.plan_digest);
    append_u64(encoded, context.graph.num_nodes);
    append_u64(encoded, context.graph.num_features);
    append_u64(encoded, context.graph.graph_count);
    append_u64(encoded, context.graph.num_classes);
    for (Int value : context.graph.src) append_int(encoded, value);
    for (Int value : context.graph.dst) append_int(encoded, value);
    for (Int value : context.graph.edge_gid) append_int(encoded, value);
    for (std::size_t value : context.graph.node_ptr) append_u64(encoded, value);
    for (std::size_t value : context.graph.edge_ptr) append_u64(encoded, value);
    append_u64(encoded, context.public_features.rows);
    append_u64(encoded, context.public_features.cols);
    for (Int value : context.public_features.values) append_int(encoded, value);
    append_u64(encoded, context.public_logits.rows);
    append_u64(encoded, context.public_logits.cols);
    for (Int value : context.public_logits.values) append_int(encoded, value);
    append_string(encoded, context.tables.table_id);
    append_string(encoded, context.tables.generator_id);
    append_string(encoded, context.tables.digest);
    append_int(encoded, context.tables.lrelu_min);
    append_int(encoded, context.tables.lrelu_max);
    append_int(encoded, context.tables.exp_min);
    append_int(encoded, context.tables.exp_max);
    append_int(encoded, context.tables.elu_min);
    append_int(encoded, context.tables.elu_max);
    append_string(encoded, context.quant.quant_cfg_id);
    append_string(encoded, context.quant.schema);
    append_int(encoded, context.quant.s_input);
    append_int(encoded, context.quant.s_delta);
    append_int(encoded, context.quant.s_exp);
    for (Int value : context.quant.s_h) append_int(encoded, value);
    const auto append_vector = [&encoded](const std::vector<Int> &values) {
        append_u64(encoded, values.size());
        for (Int value : values) append_int(encoded, value);
    };
    for (std::size_t layer = 0; layer < context.quant.s_w_hidden.size(); ++layer) {
        append_vector(context.quant.s_w_hidden[layer]);
        append_vector(context.quant.s_hprime_hidden[layer]);
        append_vector(context.quant.s_a_src_hidden[layer]);
        append_vector(context.quant.s_a_dst_hidden[layer]);
    }
    for (Int value : context.quant.s_agg_pre) append_int(encoded, value);
    for (Int value : context.quant.s_agg) append_int(encoded, value);
    for (Int value : context.quant.s_cat) append_int(encoded, value);
    append_int(encoded, context.quant.s_out_pre);
    append_int(encoded, context.quant.s_out_head);
    append_int(encoded, context.quant.s_out);
    for (Int value : context.quant.s_w_output) append_int(encoded, value);
    for (Int value : context.quant.s_yprime_output) append_int(encoded, value);
    for (Int value : context.quant.s_a_src_output) append_int(encoded, value);
    for (Int value : context.quant.s_a_dst_output) append_int(encoded, value);
    for (Int value : context.quant.s_b_output) append_int(encoded, value);
    append_string(encoded, context.quant.rounding);
    append_string(encoded, context.quant.rescale);
    append_string(encoded, context.quant.output_average);
    append_string(encoded, context.quant.static_tables_id);
    append_string(encoded, context.quant.static_table_generator_id);
    append_string(encoded, context.quant.static_row_order);
    append_string(encoded, context.quant.exp_approx_id);
    append_string(encoded, context.quant.elu_approx_id);
    append_int(encoded, context.quant.static_lrelu_min);
    append_int(encoded, context.quant.static_lrelu_max);
    append_int(encoded, context.quant.static_exp_min);
    append_int(encoded, context.quant.static_exp_max);
    append_int(encoded, context.quant.static_elu_min);
    append_int(encoded, context.quant.static_elu_max);
    append_int(encoded, context.quant.lrelu_slope_num);
    append_int(encoded, context.quant.lrelu_slope_den);
    return encoded;
}

void write_u32(std::ofstream &out, std::uint32_t value) {
    char bytes[4];
    for (unsigned i = 0; i < 4; ++i) bytes[i] = static_cast<char>((value >> (i * 8)) & 0xffU);
    out.write(bytes, sizeof(bytes));
}

void write_u64(std::ofstream &out, std::uint64_t value) {
    char bytes[8];
    for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<char>((value >> (i * 8)) & 0xffU);
    out.write(bytes, sizeof(bytes));
}

void write_string(std::ofstream &out, const std::string &value) {
    write_u64(out, static_cast<std::uint64_t>(value.size()));
    if (!value.empty()) out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

bool read_exact(std::ifstream &in, char *buffer, std::size_t size) {
    if (size == 0) return true;
    in.read(buffer, static_cast<std::streamsize>(size));
    return in.good() && static_cast<std::size_t>(in.gcount()) == size;
}

bool read_u32(std::ifstream &in, std::uint32_t &value) {
    char bytes[4];
    if (!read_exact(in, bytes, sizeof(bytes))) return false;
    value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[i])) << (i * 8);
    return true;
}

bool read_u64(std::ifstream &in, std::uint64_t &value) {
    char bytes[8];
    if (!read_exact(in, bytes, sizeof(bytes))) return false;
    value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i])) << (i * 8);
    return true;
}

bool read_string(std::ifstream &in, std::string &value) {
    std::uint64_t size = 0;
    if (!read_u64(in, size) || size > (1ULL << 26)) return false;
    value.assign(static_cast<std::size_t>(size), '\0');
    return read_exact(in, value.empty() ? nullptr : &value[0], value.size());
}

void write_commitment(std::ofstream &out, const CommitmentWire &commitment) {
    write_string(out, commitment.scheme);
    write_string(out, commitment.basis_id);
    write_u64(out, commitment.logical_size);
    write_u64(out, commitment.padded_size);
    write_string(out, commitment.point_hex);
}

bool read_commitment(std::ifstream &in, CommitmentWire &commitment) {
    std::uint64_t logical_size = 0;
    std::uint64_t padded_size = 0;
    if (!read_string(in, commitment.scheme) || !read_string(in, commitment.basis_id) ||
        !read_u64(in, logical_size) || !read_u64(in, padded_size) ||
        !read_string(in, commitment.point_hex)) return false;
    if (logical_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        padded_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) return false;
    commitment.logical_size = static_cast<std::size_t>(logical_size);
    commitment.padded_size = static_cast<std::size_t>(padded_size);
    return true;
}

void write_section(std::ofstream &out, const ProofSection &section) {
    write_string(out, section.protocol);
    write_u64(out, static_cast<std::uint64_t>(section.messages.size()));
    for (const TranscriptMessage &message : section.messages) {
        write_string(out, message.label);
        write_string(out, message.payload);
    }
}

bool read_section(std::ifstream &in, ProofSection &section) {
    std::uint64_t count = 0;
    if (!read_string(in, section.protocol) || !read_u64(in, count) || count > (1ULL << 20)) return false;
    section.messages.clear();
    section.messages.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        TranscriptMessage message;
        if (!read_string(in, message.label) || !read_string(in, message.payload)) return false;
        section.messages.push_back(std::move(message));
    }
    return true;
}

void set_reason(std::string *reason, const std::string &value) {
    if (reason != nullptr) *reason = value;
}

bool forbidden_payload(const std::string &value) {
    static const char *forbidden[] = {"WITNESS", "witness", "PRIVATE_TRACE", "private_trace", "raw_model", "raw_aux", "raw_lookup"};
    for (const char *needle : forbidden) if (value.find(needle) != std::string::npos) return true;
    return false;
}

} // namespace

bool validate_gat_circuit_plan(const GatCircuitPlan &plan, std::string *reason) {
    return validate_gat_circuit_plan_impl(plan, reason);
}

GatCircuitPlan build_gat_circuit_plan(const Graph &graph, const Model &model,
                                      const QuantConfig &quant,
                                      const ForwardResult &forward) {
    return build_plan_impl(graph, model, quant, &forward);
}

GatCircuitPlan build_gat_circuit_plan(const Graph &graph, const Model &model,
                                      const QuantConfig &quant) {
    return build_plan_impl(graph, model, quant, nullptr);
}

bool CommitmentWire::valid() const {
    if (scheme != "hyrax-pedersen-bn254" || basis_id.empty() || point_hex.empty() || logical_size == 0 || padded_size == 0 || (padded_size & (padded_size - 1)) != 0 || logical_size > padded_size) return false;
    return point_hex.size() >= 2;
}

CommitmentWire commit_private_segment(const std::vector<Int> &values,
                                      const std::string &basis_label) {
    if (values.empty()) throw std::invalid_argument("cannot commit an empty zkGPT segment");
    return commit_private_segment(values, basis_label, next_power_of_two(values.size()));
}

CommitmentWire commit_private_segment(const std::vector<Int> &values,
                                      const std::string &basis_label,
                                      std::size_t padded_domain) {
    ensure_field();
    if (basis_label.empty()) throw std::invalid_argument("Hyrax commitment basis label is empty");
    if (values.empty()) throw std::invalid_argument("cannot commit an empty zkGPT segment");
    if (padded_domain == 0 || (padded_domain & (padded_domain - 1)) != 0 || values.size() > padded_domain) {
        throw std::invalid_argument("Hyrax commitment domain is not the canonical shared N_in");
    }
    const std::size_t padded = padded_domain;
    std::vector<Fr> scalars(padded, Fr(0));
    for (std::size_t i = 0; i < values.size(); ++i) scalars[i] = Fr(values[i]);
    // A fresh blind is inserted in a dedicated padding coordinate whenever
    // possible.  For an exactly full domain the last coordinate is blinded;
    // the opening protocol must account for this coordinate in its native
    // masking extension before a formal artifact can be emitted.
    Fr blind;
    blind.setByCSPRNG();
    scalars[padded - 1] += blind;
    std::vector<G1> generators(padded);
    for (std::size_t i = 0; i < padded; ++i) {
        const std::string label = basis_label + "/coordinate/" + std::to_string(i);
        hashAndMapToG1(generators[i], label);
    }
    // This is the official Hyrax Pedersen commitment primitive, not an
    // adapter-side replacement.  No scalar or blind is retained in the wire.
    const G1 point = perdersen_commit(generators.data(), scalars.data(), static_cast<int>(padded));
    CommitmentWire result;
    result.basis_id = hash_hex(basis_label);
    result.logical_size = values.size();
    result.padded_size = padded;
    result.point_hex = point_hex(point);
    return result;
}

ModelRegistration register_model(const Model &model, const QuantConfig &quant,
                                  const std::string &model_arch_id) {
    (void)model;
    (void)quant;
    (void)model_arch_id;
    throw std::invalid_argument("canonical four-segment InputLayout is required for model registration");
}

ModelRegistration register_model(const Model &model, const QuantConfig &quant,
                                  const InputLayout &layout,
                                  const std::string &model_arch_id) {
    model.validate_schema();
    quant.validate();
    if (model_arch_id.empty()) throw std::invalid_argument("model architecture id is empty");
    const std::vector<Int> flattened = flatten_model(model, quant);
    ModelRegistration registration;
    populate_registration_metadata(registration, model, quant, layout, model_arch_id);
    const std::string basis_label = registration_basis_label(model, quant, layout, model_arch_id);
    registration.com_W = commit_private_segment(flattened, basis_label, layout.n_in);
    if (registration.com_W.logical_size != layout.segments.at(1).logical_length ||
        registration.com_W.padded_size != layout.n_in) {
        throw std::invalid_argument("model commitment does not use the canonical W/N_in domain");
    }
    registration.binding_digest = hash_hex(registration_binding_encoding(registration));
    registration.registration_digest = hash_hex("zkgpt-gat-registration-v2/" + registration.binding_digest + "/" + commitment_encoding(registration.com_W));
    registration.status = "registered_long_lived_com_W";
    return registration;
}

bool registration_matches_model(const Model &model, const QuantConfig &quant,
                                const InputLayout &layout,
                                const ModelRegistration &registration) {
    try {
        model.validate_schema();
        quant.validate();
        ensure_field();
        if (!canonical_layout_valid(layout)) return false;
        if (registration.status != "registered_long_lived_com_W" || !registration.com_W.valid()) return false;
        ModelRegistration expected;
        populate_registration_metadata(expected, model, quant, layout, registration.model_arch_id);
        if (!registration_metadata_matches(registration, expected)) return false;
        const std::string basis_label = registration_basis_label(model, quant, layout, registration.model_arch_id);
        if (registration.com_W.scheme != "hyrax-pedersen-bn254" ||
            registration.com_W.basis_id != hash_hex(basis_label) ||
            registration.com_W.logical_size != layout.segments.at(1).logical_length ||
            registration.com_W.padded_size != layout.n_in) return false;
        const std::string expected_binding = hash_hex(registration_binding_encoding(registration));
        if (registration.binding_digest != expected_binding) return false;
        const std::string expected_registration = hash_hex("zkgpt-gat-registration-v2/" + registration.binding_digest + "/" + commitment_encoding(registration.com_W));
        return registration.registration_digest == expected_registration;
    } catch (const std::exception &) {
        return false;
    }
}

FiatShamirTranscript::FiatShamirTranscript(const std::string &domain)
    : domain_(domain) {
    if (domain_.empty()) throw std::invalid_argument("Fiat-Shamir domain is empty");
}

void FiatShamirTranscript::absorb(const std::string &label, const std::string &payload) {
    if (label.empty()) throw std::invalid_argument("Fiat-Shamir message label is empty");
    messages_.push_back(TranscriptMessage{label, payload});
}

Field FiatShamirTranscript::challenge(const std::string &label) {
    if (label.empty()) throw std::invalid_argument("Fiat-Shamir challenge label is empty");
    std::string encoded;
    append_string(encoded, domain_);
    append_u64(encoded, static_cast<std::uint64_t>(messages_.size()));
    for (const TranscriptMessage &message : messages_) {
        append_string(encoded, message.label);
        append_string(encoded, message.payload);
    }
    append_string(encoded, label);
    Field result;
    result.setHashOf(encoded.data(), encoded.size());
    absorb("challenge/" + label, field_hex(result));
    return result;
}

std::string FiatShamirTranscript::digest() const {
    std::string encoded;
    append_string(encoded, domain_);
    append_u64(encoded, static_cast<std::uint64_t>(messages_.size()));
    for (const TranscriptMessage &message : messages_) {
        append_string(encoded, message.label);
        append_string(encoded, message.payload);
    }
    return hash_hex(encoded);
}

bool GatProofArtifact::has_private_payload() const {
    if (forbidden_payload(privacy_profile) || forbidden_payload(statement_digest) || forbidden_payload(transcript_digest)) return true;
    const ProofSection *sections[] = {&Pi_Lasso, &Pi_MatMul, &Pi_GKR, &Pi_combine, &Pi_open};
    for (const ProofSection *section : sections) {
        if (forbidden_payload(section->protocol)) return true;
        for (const TranscriptMessage &message : section->messages) if (forbidden_payload(message.label) || forbidden_payload(message.payload)) return true;
    }
    return false;
}

bool GatProofArtifact::valid_wire(std::string *reason) const {
    if (version != 1 || schema != "zkgpt-gat-proof-v1") { set_reason(reason, "unsupported proof wire version"); return false; }
    if (relation_id != "R_GAT-zkGPT") { set_reason(reason, "proof relation id mismatch"); return false; }
    if (privacy_profile != "masked_full") { set_reason(reason, "proof does not declare masked_full"); return false; }
    if (statement_digest.empty() || transcript_digest.empty()) { set_reason(reason, "proof transcript or statement digest is missing"); return false; }
    if (!com_aux.valid() || !com_L.valid()) { set_reason(reason, "dynamic Hyrax commitment wire is invalid"); return false; }
    const ProofSection *sections[] = {&Pi_Lasso, &Pi_MatMul, &Pi_GKR, &Pi_combine, &Pi_open};
    const char *names[] = {"Pi_Lasso", "Pi_MatMul", "Pi_GKR", "Pi_combine", "Pi_open"};
    for (unsigned i = 0; i < 5; ++i) {
        if (sections[i]->protocol != names[i] || sections[i]->messages.empty()) { set_reason(reason, std::string("missing ") + names[i] + " messages"); return false; }
        if (sections[i]->messages.size() > (1ULL << 20)) { set_reason(reason, "proof section has too many messages"); return false; }
        for (const TranscriptMessage &message : sections[i]->messages) {
            if (message.label.empty() || message.label.size() > (1ULL << 20) || message.payload.size() > (1ULL << 26)) { set_reason(reason, "proof message exceeds wire limits"); return false; }
        }
    }
    if (has_private_payload()) { set_reason(reason, "proof wire contains a private payload marker"); return false; }
    if (artifact_transcript_digest(*this) != transcript_digest) { set_reason(reason, "Fiat-Shamir transcript digest mismatch"); return false; }
    return true;
}

bool GatProofArtifact::write(const std::string &path, std::string *reason) const {
    if (!valid_wire(reason)) return false;
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!out) { set_reason(reason, "cannot open proof artifact for writing"); return false; }
    const char magic[] = "ZKGPTGAT";
    out.write(magic, sizeof(magic) - 1);
    write_u32(out, version);
    write_string(out, schema);
    write_string(out, relation_id);
    write_string(out, privacy_profile);
    write_string(out, statement_digest);
    write_commitment(out, com_aux);
    write_commitment(out, com_L);
    write_section(out, Pi_Lasso);
    write_section(out, Pi_MatMul);
    write_section(out, Pi_GKR);
    write_section(out, Pi_combine);
    write_section(out, Pi_open);
    write_string(out, transcript_digest);
    if (!out.good()) { set_reason(reason, "proof artifact write failed"); return false; }
    return true;
}

bool GatProofArtifact::read(const std::string &path, GatProofArtifact &out,
                            std::string *reason) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) { set_reason(reason, "cannot open proof artifact"); return false; }
    char magic[8];
    if (!read_exact(in, magic, sizeof(magic)) || std::memcmp(magic, "ZKGPTGAT", sizeof(magic)) != 0) { set_reason(reason, "invalid proof artifact magic"); return false; }
    std::uint32_t version = 0;
    if (!read_u32(in, version) || !read_string(in, out.schema) || !read_string(in, out.relation_id) || !read_string(in, out.privacy_profile) || !read_string(in, out.statement_digest) || !read_commitment(in, out.com_aux) || !read_commitment(in, out.com_L) || !read_section(in, out.Pi_Lasso) || !read_section(in, out.Pi_MatMul) || !read_section(in, out.Pi_GKR) || !read_section(in, out.Pi_combine) || !read_section(in, out.Pi_open) || !read_string(in, out.transcript_digest)) { set_reason(reason, "truncated proof artifact"); return false; }
    out.version = version;
    char trailing = 0;
    if (in.read(&trailing, 1)) { set_reason(reason, "proof artifact has trailing bytes"); return false; }
    return out.valid_wire(reason);
}

VerifierContext make_verifier_context(const Graph &graph,
                                      const Tensor2D &public_features,
                                      const QuantConfig &quant,
                                      const StaticTables &tables,
                                      const Tensor2D &public_logits,
                                      const CommitmentWire &com_W,
                                      const GatCircuitPlan &plan,
                                      const std::string &model_arch_id,
                                      const std::string &privacy_profile) {
    ModelRegistration legacy;
    legacy.com_W = com_W;
    legacy.model_arch_id = model_arch_id;
    legacy.quant_cfg_id = quant.quant_cfg_id;
    legacy.static_tables_id = tables.table_id;
    legacy.status = "legacy_unbound";
    (void)graph;
    (void)public_features;
    (void)public_logits;
    (void)plan;
    (void)privacy_profile;
    throw std::invalid_argument("verifier context requires a bound model registration artifact");
}

VerifierContext make_verifier_context(const Graph &graph,
                                      const Tensor2D &public_features,
                                      const QuantConfig &quant,
                                      const StaticTables &tables,
                                      const Tensor2D &public_logits,
                                      const ModelRegistration &registration,
                                      const GatCircuitPlan &plan,
                                      const std::string &privacy_profile) {
    graph.validate();
    quant.validate();
    if (!tables.authenticate()) throw std::invalid_argument("unauthenticated static tables in verifier context");
    if (public_features.rows != graph.num_nodes || public_features.cols != graph.num_features ||
        public_features.cols == 0 || public_features.values.size() != public_features.rows * public_features.cols) throw std::invalid_argument("invalid public feature shape in verifier context");
    if (public_logits.rows != graph.num_nodes || public_logits.cols != graph.num_classes ||
        public_logits.cols == 0 || public_logits.values.size() != public_logits.rows * public_logits.cols) throw std::invalid_argument("invalid public logit shape in verifier context");
    if (registration.model_arch_id.empty() || privacy_profile.empty()) throw std::invalid_argument("verifier context identity is incomplete");
    if (tables.quant_cfg_id != quant.quant_cfg_id || tables.table_id.empty()) throw std::invalid_argument("static table identity does not match quantization config");
    if (!registration_wire_valid(registration)) throw std::invalid_argument("invalid bound model registration artifact");
    if (!plan.official_layered || plan.has_placeholder || plan.circuit_id.empty() || plan.plan_digest.empty()) throw std::invalid_argument("verifier context requires an official layered plan");
    if (!canonical_layout_valid(plan.input_layout) ||
        registration.layout_digest != hash_hex([&]() {
            std::string encoded;
            append_layout_encoding(encoded, plan.input_layout);
            return encoded;
        }()) ||
        registration.n_in != plan.input_layout.n_in ||
        registration.input_segments.size() != plan.input_layout.segments.size()) {
        throw std::invalid_argument("bound model registration does not match the canonical four-segment plan layout");
    }
    for (std::size_t index = 0; index < plan.input_layout.segments.size(); ++index) {
        const InputSegment &left = registration.input_segments[index];
        const InputSegment &right = plan.input_layout.segments[index];
        if (left.name != right.name || left.offset != right.offset ||
            left.logical_length != right.logical_length || left.padded_length != right.padded_length) {
            throw std::invalid_argument("bound model registration segment does not match plan layout");
        }
    }
    if (registration.quant_digest != canonical_quant_digest(quant) ||
        registration.static_tables_id != tables.table_id || registration.field_id != "BN254" ||
        registration.field_modulus != Field::getModulo()) throw std::invalid_argument("bound model registration policy identity mismatch");
    VerifierContext context;
    context.graph = graph;
    context.public_features = public_features;
    context.quant = quant;
    context.tables = tables;
    context.public_logits = public_logits;
    context.com_W = registration.com_W;
    context.registration = registration;
    context.plan = plan;
    context.model_arch_id = registration.model_arch_id;
    context.privacy_extension_profile_id = privacy_profile;
    context.quant_cfg_id = quant.quant_cfg_id;
    context.static_tables_id = tables.table_id;
    return context;
}

std::string verifier_context_digest(const VerifierContext &context) {
    return hash_hex(context_encoding(context));
}

const char *prove_status_name(ProveStatus status) {
    switch (status) {
        case ProveStatus::GENERATED: return "generated";
        case ProveStatus::UNSUPPORTED_PRIVACY_PROFILE: return "unsupported_privacy_profile";
        case ProveStatus::REJECTED: return "rejected";
    }
    return "rejected";
}

ProveResult prove_gat(const GatCircuitPlan &plan, const Graph &graph,
                      const Tensor2D &public_features, const Model &model,
                      const QuantConfig &quant, const ForwardResult &forward,
                      const CommitmentWire &com_W,
                      const std::string &privacy_profile) {
    (void)plan;
    (void)graph;
    (void)public_features;
    (void)model;
    (void)quant;
    (void)forward;
    (void)com_W;
    (void)privacy_profile;
    ProveResult result;
    result.status = ProveStatus::REJECTED;
    result.diagnostic = "bound model registration is required by the production prover";
    return result;
}

ProveResult prove_gat(const GatCircuitPlan &plan, const Graph &graph,
                      const Tensor2D &public_features, const Model &model,
                      const QuantConfig &quant, const ForwardResult &forward,
                      const ModelRegistration &registration,
                      const std::string &privacy_profile) {
    ProveResult result;
    try {
        graph.validate();
        model.validate_schema();
        quant.validate();
        if (!plan.official_layered || plan.has_placeholder) throw std::invalid_argument("official layered circuit plan is incomplete");
        if (!registration_matches_model(model, quant, plan.input_layout, registration)) throw std::invalid_argument("private model does not match the bound model registration");
        if (!validate_relation(graph, public_features, model, quant, forward)) throw std::invalid_argument("canonical GAT relation is invalid");
        if (privacy_profile != "masked_full") {
            result.status = ProveStatus::UNSUPPORTED_PRIVACY_PROFILE;
            result.diagnostic = "unsupported_privacy_profile: zkGPT native verifier exposes no complete masking transcript";
            return result;
        }
        const MaskingSession masking = begin_masking_session(privacy_profile, plan.plan_digest);
        (void)masking;
        result.status = ProveStatus::UNSUPPORTED_PRIVACY_PROFILE;
        result.diagnostic = "unsupported_privacy_profile: masked_full requires composable fresh masking for GKR, MatMul, Lasso and W/Aux/L openings; official source does not expose it";
        result.has_artifact = false;
        return result;
    } catch (const std::exception &error) {
        result.status = ProveStatus::REJECTED;
        result.diagnostic = error.what();
        result.has_artifact = false;
        return result;
    }
}

bool verify_gat(const VerifierContext &context, const GatProofArtifact &artifact,
                std::string *reason) {
    try {
        context.graph.validate();
        context.quant.validate();
        if (!context.tables.authenticate()) { set_reason(reason, "static tables are not authenticated"); return false; }
        if (context.public_features.rows != context.graph.num_nodes || context.public_features.cols != context.graph.num_features || context.public_features.cols == 0 || context.public_features.values.size() != context.public_features.rows * context.public_features.cols) { set_reason(reason, "public feature dimensions are invalid"); return false; }
        if (context.public_logits.rows != context.graph.num_nodes || context.public_logits.cols != context.graph.num_classes || context.public_logits.cols == 0 || context.public_logits.values.size() != context.public_logits.rows * context.public_logits.cols) { set_reason(reason, "public logit dimensions are invalid"); return false; }
        if (!registration_wire_valid(context.registration)) { set_reason(reason, "bound model registration is invalid"); return false; }
        if (context.com_W.scheme != context.registration.com_W.scheme ||
            context.com_W.basis_id != context.registration.com_W.basis_id ||
            context.com_W.logical_size != context.registration.com_W.logical_size ||
            context.com_W.padded_size != context.registration.com_W.padded_size ||
            context.com_W.point_hex != context.registration.com_W.point_hex) {
            set_reason(reason, "verifier context com_W does not match the registration artifact");
            return false;
        }
        if (!context.plan.official_layered || context.plan.has_placeholder) { set_reason(reason, "verifier plan is not an official layered circuit"); return false; }
        if (!canonical_layout_valid(context.plan.input_layout) ||
            context.registration.n_in != context.plan.input_layout.n_in ||
            context.registration.input_segments.size() != context.plan.input_layout.segments.size()) {
            set_reason(reason, "registration layout does not match the canonical four-segment plan");
            return false;
        }
        for (std::size_t index = 0; index < context.plan.input_layout.segments.size(); ++index) {
            const InputSegment &left = context.registration.input_segments[index];
            const InputSegment &right = context.plan.input_layout.segments[index];
            if (left.name != right.name || left.offset != right.offset ||
                left.logical_length != right.logical_length || left.padded_length != right.padded_length) {
                set_reason(reason, "registration segment does not match the canonical plan");
                return false;
            }
        }
        if (context.tables.quant_cfg_id != context.quant.quant_cfg_id || context.tables.table_id.empty()) { set_reason(reason, "static table identity mismatch"); return false; }
        if (context.relation_id != "R_GAT-zkGPT" || context.quant_cfg_id != context.quant.quant_cfg_id || context.static_tables_id != context.tables.table_id ||
            context.model_arch_id != context.registration.model_arch_id) { set_reason(reason, "public verifier context identity mismatch"); return false; }
        if (context.privacy_extension_profile_id != "masked_full") { set_reason(reason, "unsupported_privacy_profile"); return false; }
        if (!artifact.valid_wire(reason)) return false;
        VerifierContext canonical = context;
        const std::string expected_statement = hash_hex(context_encoding(canonical));
        if (artifact.statement_digest != expected_statement) { set_reason(reason, "proof statement digest does not match independent public context"); return false; }
        // The official verifier's GKR/Lasso methods are private and its
        // interactive randomness is not a serializable Fiat-Shamir transcript.
        // Refuse acceptance until the native masked extension is connected;
        // accepting the wire here would turn a digest check into a fake proof.
        set_reason(reason, "unsupported_privacy_profile: native masked_full verifier transcript is unavailable");
        return false;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}

MaskingSession begin_masking_session(const std::string &profile_id,
                                     const std::string &statement_digest) {
    ensure_field();
    if (profile_id.empty() || statement_digest.empty()) throw std::invalid_argument("masking session identity is incomplete");
    Field nonce;
    nonce.setByCSPRNG();
    MaskingSession session;
    session.profile_id = profile_id;
    session.session_nonce = hash_hex(profile_id + "/" + statement_digest + "/" + field_hex(nonce));
    session.complete = false;
    return session;
}

NativePrimitiveEvidence exercise_official_primitives(const std::string &label) {
    ensure_field();
    if (label.empty()) throw std::invalid_argument("native primitive evidence label is empty");
    NativePrimitiveEvidence evidence;
    prover native_prover;
    native_prover.C.init(0, 2);
    native_prover.C.circuit[0].ty = layerType::INPUT;
    native_prover.C.circuit[0].size = 2;
    native_prover.C.circuit[0].zero_start_id = 2;
    native_prover.C.circuit[0].bit_length = 1;
    native_prover.C.circuit[1].ty = layerType::RELU;
    native_prover.C.circuit[1].size = 1;
    native_prover.C.circuit[1].zero_start_id = 1;
    native_prover.C.circuit[1].bit_length = 0;
    // Two terms force both input coordinates into the official subset map;
    // the second term has zero coefficient and does not alter x*w.
    native_prover.C.circuit[1].bin_gates.emplace_back(0, 0, 1, 1, 0);
    native_prover.C.circuit[1].bin_gates.emplace_back(0, 1, 0, 0, 0);
    native_prover.C.initSubset();
    native_prover.val.resize(2);
    native_prover.val[0] = {Field(3), Field(5)};
    native_prover.val[1] = {Field(15)};
    native_prover.init();
    std::vector<Field> output_point;
    output_point.push_back(Field(0));
    native_prover.sumcheckInitAll(output_point.begin());
    native_prover.sumcheckInit(Field::one(), Field(0));
    native_prover.sumcheckInitPhase1(Field::one());
    const Field challenge = [&]() {
        Field value;
        value.setHashOf(label.data(), label.size());
        return value;
    }();
    const quadratic_poly polynomial = native_prover.sumcheckUpdate1(Field(0));
    Field claim0, claim1;
    native_prover.sumcheckFinalize1(challenge, claim0, claim1);
    evidence.sumcheck_called = true;
    evidence.gkr_path_bound = true;
    evidence.messages.push_back("official::prover::sumcheckUpdate1=" + field_hex(polynomial.a) + "," + field_hex(polynomial.b) + "," + field_hex(polynomial.c));
    evidence.messages.push_back("official::prover::sumcheckFinalize1=" + field_hex(claim0) + "," + field_hex(claim1));

    range_prover native_range;
    Field range_values[2] = {Field(0), Field(1)};
    const range_prover::SC_Return range_result = native_range.sumcheck_deg1(1, range_values, Field(1));
    evidence.lasso_range_called = true;
    evidence.messages.push_back("official::range_prover::sumcheck_deg1=" + field_hex(range_result.claim_f));
    delete[] range_result.random;

    G1 generators[2];
    hashAndMapToG1(generators[0], label + "/hyrax/0");
    hashAndMapToG1(generators[1], label + "/hyrax/1");
    Field scalars[2] = {Field(3), Field(5)};
    const G1 commitment = perdersen_commit(generators, scalars, 2);
    evidence.hyrax_commit_called = true;
    evidence.messages.push_back("official::hyrax::perdersen_commit=" + point_hex(commitment));
    evidence.messages.push_back("official::lasso::sumcheckLassoUpdate=unavailable_in_upstream");
    return evidence;
}

} // namespace gat
} // namespace zkgpt
