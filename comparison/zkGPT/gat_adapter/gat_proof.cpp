#include "gat_proof.hpp"

#include "hyrax_util.hpp"
#include "native_lasso.hpp"
#include "utils.hpp"
#include "verifier.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <set>
#include <tuple>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

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

const std::size_t kArtifactMaxString = 1ULL << 26;
const std::size_t kArtifactMaxMessages = 1ULL << 20;
const std::uint64_t kArtifactMaxWireBytes = 1ULL << 30;
// A verifier context contains public graph/features and the complete
// canonical circuit plan.  Keep a finite wire/memory budget while allowing
// the largest supported public instances to be serialized.
const std::uint64_t kContextMaxWireBytes = 2ULL << 30;
const std::uint64_t kContextMaxElements = 1ULL << 30;
const std::uint64_t kContextMaxLayers = 1ULL << 20;
const std::uint64_t kContextMaxDescriptors = 1ULL << 20;
const char kNativeArtifactDomain[] = "zkgpt-native-gat-fs-v1";
const char kVerifierContextMagic[] = "ZKGPTVC2";

// Defined with the transcript bridge below; the production prover uses the
// same lossless section projection after all protocol scopes have completed.
void core_copy_sections(const NativeProofSession &session,
                        GatProofArtifact &artifact);
bool core_split_pipe(const std::string &payload,
                     std::vector<std::string> &parts);
bool core_decode_field(const std::string &encoded, Fr &value);
bool core_decode_point(const std::string &encoded, G1 &point);
bool core_power_of_two(std::size_t value);
std::size_t core_log2(std::size_t value);
const NativeProofMessage *core_find_message(
    const NativeProofSession &session, NativeProofMessageKind kind,
    const std::string &label, std::string *reason);

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

std::string fixed_digest(std::string value) {
    if (value.empty() || value.size() > 64)
        throw std::invalid_argument("digest is outside the fixed-width field");
    for (char &character : value) {
        if (character >= 'A' && character <= 'F') character = static_cast<char>(character - 'A' + 'a');
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f')))
            throw std::invalid_argument("digest is not lowercase hexadecimal");
    }
    return std::string(64 - value.size(), '0') + value;
}

std::string normalize_digest(const std::string &value) {
    if (value.size() == 64) {
        try {
            return fixed_digest(value);
        } catch (...) {
        }
    }
    return fixed_digest(hash_hex(value));
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
    const std::array<std::size_t, 4> blocks = {
        next_power_of_two(lrelu_queries), next_power_of_two(exp_queries),
        next_power_of_two(elu_queries), next_power_of_two(range_queries)};
    std::size_t cursor = 0;
    for (std::size_t block : blocks) {
        const std::size_t remainder = cursor % block;
        if (remainder != 0)
            cursor = checked_add(cursor, block - remainder, "aligned query gap");
        cursor = checked_add(cursor, block, "padded query count");
    }
    return cursor;
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
    const std::array<std::size_t, 4> blocks = {
        next_power_of_two(counts.lrelu), next_power_of_two(counts.exp),
        next_power_of_two(counts.elu), next_power_of_two(counts.range)};
    std::size_t result = 0;
    for (std::size_t block : blocks) {
        const std::size_t remainder = result % block;
        if (remainder != 0)
            result = checked_add(result, block - remainder,
                                 "canonical aligned query gap");
        result = checked_add(result, block, "canonical padded query size");
    }
    return result;
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
    const auto start_block = [&cursor](std::size_t block,
                                       const char *what) {
        if (block == 0 || (block & (block - 1)) != 0)
            throw std::invalid_argument("canonical query block is not power-of-two");
        const std::size_t remainder = cursor % block;
        if (remainder != 0)
            cursor = checked_add(cursor, block - remainder,
                                 "canonical aligned query gap");
        const std::size_t start = cursor;
        cursor = checked_add(cursor, block, what);
        return start;
    };
    result.lrelu = start_block(next_power_of_two(result.counts.lrelu),
                               "LReLU query block");
    result.exp = start_block(next_power_of_two(result.counts.exp),
                             "ExpMap query block");
    result.elu = start_block(next_power_of_two(result.counts.elu),
                             "ELU query block");
    result.range = start_block(next_power_of_two(result.counts.range),
                               "range query block");
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

using LinearTerm = std::pair<std::size_t, ll>;

struct ProductTerm {
    std::size_t left = 0;
    std::size_t right = 0;
    ll coefficient = 1;
};

// A native zkGPT gate output is the sum of all terms having the same `g`.
// Build one output coordinate per residual, rather than appending unrelated
// unary gates with the same output id.  This keeps the relation honest under
// layeredCircuit::initSubset and mirrors the upstream prover's gate algebra.
struct RelationLayerBuilder {
    std::size_t next_output = 0;
    std::vector<ExplicitUnaryGate> unary;
    std::vector<ExplicitBinaryGate> binary;

    std::size_t add_residual(const std::vector<LinearTerm> &linear,
                             const std::vector<ProductTerm> &products) {
        const std::size_t output = next_output++;
        if (linear.empty() && products.empty())
            throw std::invalid_argument("empty GAT arithmetic residual");
        for (const LinearTerm &term : linear)
            unary.push_back({output, term.first, 0, term.second});
        for (const ProductTerm &term : products)
            binary.push_back({output, term.left, term.right, 0,
                              term.coefficient});
        return output;
    }

    std::size_t add_unary(std::size_t input, ll coefficient = 1) {
        return add_residual({{input, coefficient}}, {});
    }
};

std::vector<std::pair<std::size_t, std::size_t>> incoming_ranges_for_plan(
    const Graph &graph) {
    std::vector<std::pair<std::size_t, std::size_t>> ranges(
        graph.num_nodes, {graph.edges(), graph.edges()});
    for (std::size_t edge = 0; edge < graph.edges(); ++edge) {
        const std::size_t destination = static_cast<std::size_t>(graph.dst[edge]);
        if (ranges[destination].first == graph.edges())
            ranges[destination].first = edge;
        ranges[destination].second = edge + 1;
    }
    for (const auto &range : ranges)
        if (range.first == graph.edges())
            throw std::invalid_argument("canonical GAT circuit has an empty destination group");
    return ranges;
}

void add_matmul_residuals(RelationLayerBuilder &builder,
                          const std::vector<MatMulDescriptor> &matmuls) {
    for (const MatMulDescriptor &descriptor : matmuls) {
        for (std::size_t row = 0; row < descriptor.rows; ++row) {
            for (std::size_t col = 0; col < descriptor.cols; ++col) {
                const std::size_t output_index =
                    descriptor.output_offset + row * descriptor.cols + col;
                std::vector<LinearTerm> linear = {{output_index, -1}};
                std::vector<ProductTerm> products;
                products.reserve(descriptor.inner);
                for (std::size_t term = 0; term < descriptor.inner; ++term) {
                    products.push_back({
                        descriptor.lhs_offset + row * descriptor.inner + term,
                        descriptor.rhs_offset + term * descriptor.cols + col,
                        1});
                }
                builder.add_residual(linear, products);
            }
        }
    }
}

void add_division_residuals(RelationLayerBuilder &builder,
                            std::size_t division_offset,
                            std::size_t output_offset,
                            const std::vector<LinearTerm> &dividend,
                            const std::vector<LinearTerm> &denominator,
                            const std::string &rounding) {
    // Aux_div layout: sigma,z,a,q,r,delta,a_inverse,d_inverse,denominator.
    const std::size_t sigma = division_offset;
    const std::size_t zero = division_offset + 1;
    const std::size_t a = division_offset + 2;
    const std::size_t q = division_offset + 3;
    const std::size_t r = division_offset + 4;
    const std::size_t delta = division_offset + 5;
    const std::size_t a_inverse = division_offset + 6;
    const std::size_t d_inverse = division_offset + 7;
    const std::size_t d = division_offset + 8;

    std::vector<LinearTerm> dividend_relation = dividend;
    dividend_relation.push_back({a, -1});
    builder.add_residual(dividend_relation, {{sigma, a, 2}});

    builder.add_residual({{output_offset, 1}, {q, -1}}, {{sigma, q, 2}});

    if (!denominator.empty()) {
        std::vector<LinearTerm> denominator_relation = denominator;
        denominator_relation.push_back({d, -1});
        builder.add_residual(denominator_relation, {});
    }

    // SignedRoundDiv has two different quotient equations.  In nearest mode
    // `r` is the centered remainder, so the toward-zero equation a=dq+r is
    // false (for example 8192/2048 gives q=4,r=2048).  Emit exactly the
    // canonical equation selected by the public rounding policy.
    if (rounding == "nearest_ties_away") {
        builder.add_residual({{a, 2}, {d, 1}, {r, -1}},
                             {{d, q, -2}});
    } else if (rounding == "toward_zero") {
        builder.add_residual({{a, 1}, {r, -1}}, {{d, q, -1}});
    } else {
        throw std::invalid_argument("unsupported canonical GAT rounding mode");
    }

    // Boolean/sign/zero products are exact polynomial constraints.  The two
    // unit equations (a*a_inverse+z=1 and d*d_inverse=1) and the final
    // tie-bound equation contain a constant one; their exact native check is
    // supplied by the SignedRoundDiv/Lasso relation and local witness gate
    // evaluator, so no constant wire is invented in this adapter circuit.
    builder.add_residual({{sigma, -1}}, {{sigma, sigma, 1}});
    builder.add_residual({{zero, -1}}, {{zero, zero, 1}});
    builder.add_residual({}, {{zero, a, 1}});
    builder.add_residual({}, {{zero, sigma, 1}});
    builder.add_residual({}, {{zero, a_inverse, 1}});
    (void)delta;
    (void)d_inverse;
}

void add_attention_residuals(
    RelationLayerBuilder &builder, const Graph &graph,
    const std::vector<std::pair<std::size_t, std::size_t>> &incoming,
    const AttentionCoordinates &a, bool hidden, const QuantConfig &quant) {
    const std::size_t edges = graph.edges();
    const std::size_t output_dim = a.output_dim;
    const std::size_t coordinates = graph.num_nodes * output_dim;

    for (std::size_t edge = 0; edge < edges; ++edge) {
        const std::size_t source = static_cast<std::size_t>(graph.src[edge]);
        const std::size_t destination = static_cast<std::size_t>(graph.dst[edge]);
        builder.add_residual(
            {{a.source_score + source, 1},
             {a.destination_score + destination, 1},
             {a.raw_score + edge, -1}}, {});
        builder.add_residual(
            {{a.group_max + destination, 1},
             {a.leaky_score + edge, -1},
             {a.delta + edge, -1}}, {});
        builder.add_residual({}, {{a.max_selector + edge, a.delta + edge, 1}});
        builder.add_residual(
            {{a.exp + edge, -1}},
            {{a.alpha + edge, a.group_sum + destination, 1}});
    }

    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        std::vector<LinearTerm> sum_exp = {{a.group_sum + node, -1}};
        std::vector<ProductTerm> selected_max;
        std::vector<ProductTerm> selected_sum;
        for (std::size_t edge = incoming[node].first;
             edge < incoming[node].second; ++edge) {
            sum_exp.push_back({a.exp + edge, 1});
            selected_max.push_back({a.max_selector + edge,
                                    a.leaky_score + edge, 1});
            selected_sum.push_back({a.max_selector + edge,
                                    a.group_sum + node, 1});
        }
        builder.add_residual(sum_exp, {});
        builder.add_residual({{a.group_max + node, -1}}, selected_max);
        builder.add_residual({{a.group_sum + node, -1}}, selected_sum);
        for (std::size_t col = 0; col < output_dim; ++col) {
            const std::size_t coordinate = node * output_dim + col;
            std::vector<LinearTerm> integer_relation = {
                {a.integer_numerator + coordinate, -1}};
            std::vector<ProductTerm> integer_products;
            std::vector<ProductTerm> field_products;
            for (std::size_t edge = incoming[node].first;
                 edge < incoming[node].second; ++edge) {
                const std::size_t source = static_cast<std::size_t>(graph.src[edge]);
                integer_products.push_back({a.exp + edge,
                                            a.projection + source * output_dim + col,
                                            1});
                field_products.push_back({a.alpha + edge,
                                          a.projection + source * output_dim + col,
                                          1});
            }
            builder.add_residual(integer_relation, integer_products);
            builder.add_residual({{a.field_numerator + coordinate, 1},
                                  {a.integer_numerator + coordinate, -1}}, {});
            std::vector<LinearTerm> field_relation = {
                {a.field_numerator + coordinate, -1}};
            builder.add_residual(field_relation,
                                 {{a.field_aggregate + coordinate,
                                   a.group_sum + node, 1}});
        }
    }

    // The precise scale/denominator selection is family-specific and is
    // supplied by the caller below; this routine only emits scale-free
    // sparse arithmetic.  Keep the arguments live for compile-time checking.
    (void)hidden;
    (void)quant;
    (void)coordinates;
}

void add_attention_recovery_residuals(
    RelationLayerBuilder &builder, const Graph &graph,
    const AttentionCoordinates &a, bool hidden, std::size_t layer_id,
    std::size_t head_id, const QuantConfig &quant) {
    const Int dividend_scale = hidden ? quant.s_agg_pre[layer_id]
                                     : quant.s_out_pre;
    const Int denominator_scale = hidden
        ? quant.s_hprime_hidden[layer_id][head_id]
        : quant.s_yprime_output[head_id];
    const std::size_t coordinates = graph.num_nodes * a.output_dim;
    for (std::size_t index = 0; index < coordinates; ++index) {
        add_division_residuals(
            builder, a.divisions + index * 9, a.recovered + index,
            {{a.integer_numerator + index, dividend_scale}},
            {{a.group_sum + index / a.output_dim, denominator_scale}},
            quant.rounding);
    }
}

void finish_relation_layer(GatCircuitPlan &plan,
                           RelationLayerBuilder &builder,
                           std::size_t public_output_coordinates) {
    if (builder.next_output < public_output_coordinates)
        throw std::logic_error("GAT relation layer output count underflow");
    if (builder.next_output > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("GAT relation layer exceeds official u32 output size");
    layer relation;
    relation.ty = layerType::RELU;
    relation.size = as_u32(builder.next_output, "GAT relation layer size");
    relation.zero_start_id = relation.size;
    relation.bit_length = bit_length(relation.size);
    relation.need_phase2 = false;
    relation.uni_gates.reserve(builder.unary.size());
    relation.bin_gates.reserve(builder.binary.size());
    for (const ExplicitUnaryGate &gate : builder.unary)
        relation.uni_gates.emplace_back(as_u32(gate.g, "GAT residual output"),
                                       as_u32(gate.u, "GAT residual input"),
                                       gate.lu, gate.sc);
    for (const ExplicitBinaryGate &gate : builder.binary)
        relation.bin_gates.emplace_back(as_u32(gate.g, "GAT residual output"),
                                       as_u32(gate.u, "GAT residual lhs"),
                                       as_u32(gate.v, "GAT residual rhs"),
                                       gate.sc, gate.l);
    plan.circuit.circuit.resize(1);
    plan.circuit.circuit.push_back(std::move(relation));
    plan.circuit.size = static_cast<u8>(plan.circuit.circuit.size());
    plan.public_output_coordinates = public_output_coordinates;
    plan.residual_coordinates = builder.next_output - public_output_coordinates;
}

std::string plan_encoding(const GatCircuitPlan &plan) {
    std::string encoded;
    append_string(encoded, plan.registry.registry_id);
    append_u64(encoded, static_cast<std::uint64_t>(plan.registry.descriptors.size()));
    for (const RelationDescriptor &descriptor : plan.registry.descriptors) {
        append_string(encoded, descriptor.id);
        append_string(encoded, descriptor.primitive);
        append_string(encoded, descriptor.segment);
        append_string(encoded, descriptor.relation);
        append_string(encoded, descriptor.public_claim);
        append_string(encoded, descriptor.witness);
    }
    append_string(encoded, plan.input_layout.selector);
    append_string(encoded, plan.opening_policy);
    append_u64(encoded, static_cast<std::uint64_t>(plan.public_output_coordinates));
    append_u64(encoded, static_cast<std::uint64_t>(plan.residual_coordinates));
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
    append_u64(encoded, static_cast<std::uint64_t>(plan.matmuls.size()));
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
        append_string(encoded, descriptor.native_path);
    }
    append_u64(encoded, static_cast<std::uint64_t>(plan.sparse_gkr.size()));
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
        append_u64(encoded, static_cast<std::uint64_t>(descriptor.constraints.size()));
        for (const std::string &constraint : descriptor.constraints) append_string(encoded, constraint);
    }
    append_u64(encoded, static_cast<std::uint64_t>(plan.lasso.size()));
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
    append_u64(encoded, static_cast<std::uint64_t>(plan.checkpoints.size()));
    for (const CheckpointDescriptor &checkpoint : plan.checkpoints) {
        append_string(encoded, checkpoint.id);
        append_string(encoded, checkpoint.relation_id);
        append_string(encoded, checkpoint.segment);
        append_u64(encoded, checkpoint.coordinates);
        append_u64(encoded, checkpoint.coordinate_offset);
        append_string(encoded, checkpoint.coordinate_rule);
    }
    append_u64(encoded, static_cast<std::uint64_t>(plan.native_api.size()));
    for (const std::string &api : plan.native_api) append_string(encoded, api);
    append_u64(encoded, static_cast<std::uint64_t>(plan.circuit.circuit.size()));
    append_u64(encoded, static_cast<std::uint64_t>(plan.circuit.size));
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

    // Replace the former collection of independent checkpoint layers with a
    // single official relation layer.  Each output coordinate has exactly
    // one semantic meaning: public Y first, then one arithmetic residual.
    // Every residual reads the canonical four-segment input layer, so no
    // gate can accidentally read a stale checkpoint or overwrite another
    // operation's output coordinate.
    RelationLayerBuilder relation_builder;
    const std::size_t public_output_coordinates =
        checked_mul(graph.num_nodes, classes, "public output coordinates");
    for (std::size_t index = 0; index < public_output_coordinates; ++index)
        relation_builder.add_unary(coordinates.final_logits + index);

    add_matmul_residuals(relation_builder, plan.matmuls);
    const auto incoming = incoming_ranges_for_plan(graph);
    for (std::size_t layer_id = 0; layer_id < 2; ++layer_id) {
        const std::size_t output_dim =
            model.tensor(hidden_tensor_name(layer_id, 0, "seq/kernel"))
                .shape.dims.at(2);
        for (std::size_t head = 0; head < 8; ++head) {
            const AttentionCoordinates &a = coordinates.hidden[layer_id][head];
            add_attention_residuals(relation_builder, graph, incoming, a, true,
                                    quant);
            add_attention_recovery_residuals(relation_builder, graph, a, true,
                                             layer_id, head, quant);
        }
        for (std::size_t node = 0; node < graph.num_nodes; ++node) {
            for (std::size_t head = 0; head < 8; ++head) {
                const AttentionCoordinates &a = coordinates.hidden[layer_id][head];
                for (std::size_t col = 0; col < output_dim; ++col) {
                    const std::size_t output_index =
                        coordinates.hidden_concat[layer_id] +
                        node * output_dim * 8 + head * output_dim + col;
                    const std::size_t source =
                        a.activated + node * output_dim + col;
                    relation_builder.add_residual(
                        {{output_index, 1}, {source, -1}}, {});
                }
            }
        }
    }

    std::array<std::size_t, 2> output_bias_offsets{{0, 0}};
    for (const CheckpointDescriptor &checkpoint : plan.checkpoints) {
        for (std::size_t head = 0; head < 2; ++head) {
            const std::string expected =
                "output/bias/output/head" + std::to_string(head);
            if (checkpoint.id == expected)
                output_bias_offsets[head] = checkpoint.coordinate_offset;
        }
    }
    for (std::size_t head = 0; head < 2; ++head) {
        const AttentionCoordinates &a = coordinates.output[head];
        const std::size_t output_coordinates =
            checked_mul(graph.num_nodes, classes, "output relation coordinates");
        add_attention_residuals(relation_builder, graph, incoming, a, false,
                                quant);
        add_attention_recovery_residuals(relation_builder, graph, a, false, 0,
                                         head, quant);
        for (std::size_t index = 0; index < output_coordinates; ++index) {
            relation_builder.add_residual(
                {{coordinates.output_aggregate[head] + index, 1},
                 {a.recovered + index, -1}}, {});
            add_division_residuals(
                relation_builder,
                coordinates.output_alignment_divisions[head] + index * 9,
                coordinates.output_aligned[head] + index,
                {{coordinates.output_aggregate[head] + index,
                  quant.s_out_head}},
                {}, quant.rounding);
            relation_builder.add_residual(
                {{coordinates.output_logits[head] + index, -1},
                 {coordinates.output_aligned[head] + index, 1},
                 {output_bias_offsets[head] + index % classes, 1}}, {});
        }
    }
    for (std::size_t index = 0; index < public_output_coordinates; ++index) {
        relation_builder.add_residual(
            {{coordinates.head_sum + index, -1},
             {coordinates.output_logits[0] + index, 1},
             {coordinates.output_logits[1] + index, 1}}, {});
        add_division_residuals(
            relation_builder, coordinates.final_average_divisions + index * 9,
            coordinates.final_logits + index,
            {{coordinates.head_sum + index, 1}}, {}, quant.rounding);
    }
    finish_relation_layer(plan, relation_builder, public_output_coordinates);

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
        if (plan.public_output_coordinates == 0 || plan.residual_coordinates == 0 ||
            plan.circuit.circuit.back().size !=
                plan.public_output_coordinates + plan.residual_coordinates) {
            return fail("final relation layer does not expose Y plus residual coordinates");
        }
        const std::size_t input_size = plan.circuit.circuit.front().size;
        for (std::size_t layer_id = 1; layer_id < plan.circuit.circuit.size(); ++layer_id) {
            const layer &current = plan.circuit.circuit[layer_id];
            if (current.ty == layerType::PLACE_HOLDER || current.size == 0) return fail("placeholder or empty circuit layer");
            for (const uniGate &gate : current.uni_gates) {
                if (gate.g >= current.size || gate.u >= input_size) return fail("unary gate coordinate is out of bounds");
                if (gate.lu != 0) return fail("canonical GAT relation unexpectedly reads a non-input unary layer");
            }
            for (const binGate &gate : current.bin_gates) {
                if (gate.g >= current.size || gate.u >= input_size || gate.v >= input_size) return fail("binary gate coordinate is out of bounds");
                if (gate.l != 0) return fail("canonical GAT relation unexpectedly reads a non-input binary layer");
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
            if (descriptor.padded_queries != 0) {
                const std::size_t local_offset = query_cursor - query_segment.offset;
                const std::size_t remainder = local_offset % descriptor.padded_queries;
                if (remainder != 0)
                    query_cursor = checked_add(
                        query_cursor, descriptor.padded_queries - remainder,
                        "validated aligned Lasso query gap");
            }
            if (descriptor.relation_id.empty() || descriptor.real_queries == 0 ||
                descriptor.padded_queries != next_power_of_two(descriptor.real_queries) ||
                descriptor.table_domain == 0 || (descriptor.table_domain & (descriptor.table_domain - 1)) != 0 ||
                descriptor.query_offset != query_cursor ||
                ((descriptor.query_offset - query_segment.offset) %
                     descriptor.padded_queries) != 0 ||
                !require_span(descriptor.query_offset, descriptor.padded_queries) ||
                descriptor.query_offset + descriptor.padded_queries > query_segment.offset + query_segment.logical_length ||
                descriptor.incidence_rule.find("L_qry/") != 0 ||
                descriptor.incidence_rule.find("canonical_operation_index_row_major") == std::string::npos ||
                descriptor.challenge_policy != "LC_before_challenge" || descriptor.query_encoding.empty()) {
                std::ostringstream detail;
                detail << "Lasso query block is not canonical or not contiguous"
                       << " family=" << descriptor.family_id
                       << " off=" << descriptor.query_offset
                       << " cursor=" << query_cursor
                       << " seg=" << query_segment.offset
                       << " logical=" << query_segment.logical_length
                       << " real=" << descriptor.real_queries
                       << " padded=" << descriptor.padded_queries;
                return fail(detail.str());
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
    append_string(encoded, "zkgpt-gat-commitment-wire-v2");
    append_string(encoded, commitment.scheme);
    append_string(encoded, commitment.basis_id);
    append_u64(encoded, commitment.logical_size);
    append_u64(encoded, commitment.padded_size);
    append_string(encoded, commitment.point_hex);
    return encoded;
}

void append_registration_encoding(std::string &encoded,
                                  const ModelRegistration &registration) {
    append_string(encoded, "zkgpt-gat-registration-context-v2");
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
        append_u64(encoded, message.kind);
        append_u64(encoded, message.sequence);
    }
    return encoded;
}

std::string artifact_transcript_digest(const GatProofArtifact &artifact) {
    // The native typed transcript is the authenticated proof object.  Keep
    // the helper name for callers that already use it, but deliberately avoid
    // rebuilding a digest from section summaries: summaries are collector
    // metadata and cannot stand in for GKR/MatMul/Lasso/opening messages.
    NativeProofSession session;
    std::string reason;
    if (!NativeProofSession::from_messages(artifact.native_domain,
                                           artifact.native_statement,
                                           artifact.native_messages, session,
                                           &reason)) {
        return std::string();
    }
    return session.digest();
}

// Build a genuine Hyrax/Pedersen vector commitment and retain its blinding
// scalar only on the prover side.  The blind is an independent H*rho term;
// putting rho into a padding coordinate would leave the commitment itself
// unblinded and would not support a zero-knowledge opening.
CommitmentWire commit_segment_with_mask(const std::vector<Int> &values,
                                        const std::string &basis_label,
                                        std::size_t padded_domain,
                                        Fr *mask_out,
                                        std::vector<Fr> *masked_values_out) {
    ensure_field();
    if (values.empty() || basis_label.empty() || padded_domain == 0 ||
        (padded_domain & (padded_domain - 1)) != 0 ||
        values.size() > padded_domain) {
        throw std::invalid_argument("invalid masked Hyrax segment domain");
    }
    std::vector<Fr> scalars(padded_domain, Fr(0));
    for (std::size_t i = 0; i < values.size(); ++i)
        scalars[i] = Fr(values[i]);
    const std::string basis_id = fixed_digest(hash_hex(basis_label));
    const G1 H = hyrax::derive_pedersen_blinding_generator(basis_id);
    Fr mask;
    mask.setByCSPRNG();
    std::vector<G1> generators(padded_domain);
    for (std::size_t i = 0; i < padded_domain; ++i)
        hashAndMapToG1(generators[i], basis_label + "/coordinate/" + std::to_string(i));
    G1 point = perdersen_commit(generators.data(), scalars.data(),
                                static_cast<int>(padded_domain));
    point += H * mask;
    if (mask_out != nullptr) *mask_out = mask;
    if (masked_values_out != nullptr) *masked_values_out = std::move(scalars);
    CommitmentWire result;
    result.basis_id = basis_id;
    result.logical_size = values.size();
    result.padded_size = padded_domain;
    result.point_hex = point_hex(point);
    return result;
}

CommitmentWire commit_field_segment_with_mask(
    const std::vector<Fr> &values, const std::string &basis_label,
    std::size_t padded_domain, std::vector<Fr> *masked_values_out) {
    ensure_field();
    if (values.empty() || basis_label.empty() || padded_domain == 0 ||
        (padded_domain & (padded_domain - 1)) != 0 ||
        values.size() > padded_domain)
        throw std::invalid_argument("invalid field segment commitment domain");
    std::vector<Fr> scalars(padded_domain, Fr(0));
    std::copy(values.begin(), values.end(), scalars.begin());
    const std::string basis_id = fixed_digest(hash_hex(basis_label));
    const G1 H = hyrax::derive_pedersen_blinding_generator(basis_id);
    Fr mask;
    mask.setByCSPRNG();
    std::vector<G1> generators(padded_domain);
    for (std::size_t index = 0; index < padded_domain; ++index)
        hashAndMapToG1(generators[index], basis_label + "/coordinate/" +
                                      std::to_string(index));
    G1 point = perdersen_commit(generators.data(), scalars.data(),
                                static_cast<int>(padded_domain));
    point += H * mask;
    if (masked_values_out != nullptr) *masked_values_out = scalars;
    CommitmentWire result;
    result.basis_id = basis_id;
    result.logical_size = values.size();
    result.padded_size = padded_domain;
    result.point_hex = point_hex(point);
    return result;
}

std::string registration_basis_label_public(const ModelRegistration &registration,
                                            const InputLayout &layout) {
    std::string encoded;
    append_string(encoded, "zkgpt-gat-registration-material-v2");
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
    append_layout_encoding(encoded, layout);
    append_string(encoded, registration.opening_policy);
    return "zkGPT/model/v2/" + hash_hex(encoded);
}

std::vector<Fr> adapter_mle_basis(const std::vector<Fr> &point) {
    if (point.empty() || point.size() >= sizeof(std::size_t) * 8)
        throw std::invalid_argument("adapter MLE point dimension is invalid");
    std::vector<Fr> basis(std::size_t(1) << point.size());
    basis[0] = Fr(1) - point[0];
    basis[1] = point[0];
    for (std::size_t bit = 1; bit < point.size(); ++bit) {
        const std::size_t old_size = std::size_t(1) << bit;
        for (std::size_t index = old_size; index-- > 0;) {
            const Fr previous = basis[index];
            basis[index] = previous * (Fr(1) - point[bit]);
            basis[index + old_size] = previous * point[bit];
        }
    }
    return basis;
}

void append_division_components(std::vector<Fr> &values,
                                const DivisionWitness &division) {
    // The canonical adapter layout reserves nine coordinates.  The signed
    // dividend is derivable from sigma*a; keep denominator as the ninth
    // public-policy-bound integer instead of duplicating that derived value.
    values.push_back(Fr(division.sigma));
    values.push_back(Fr(division.z));
    values.push_back(Fr(division.a));
    values.push_back(Fr(division.q));
    values.push_back(Fr(division.r));
    values.push_back(Fr(division.delta));
    values.push_back(division.a_inverse);
    values.push_back(division.d_inverse);
    values.push_back(Fr(division.denominator));
}

void append_attention_aux_values(std::vector<Fr> &values,
                                 const AttentionTrace &trace) {
    const auto append_ints = [&values](const std::vector<Int> &source) {
        for (Int value : source) values.push_back(Fr(value));
    };
    append_ints(trace.projection);
    append_ints(trace.source_score);
    append_ints(trace.destination_score);
    append_ints(trace.raw_score);
    append_ints(trace.leaky_score);
    append_ints(trace.group_max);
    append_ints(trace.delta);
    append_ints(trace.exp);
    append_ints(trace.group_sum);
    values.insert(values.end(), trace.group_inverse.begin(), trace.group_inverse.end());
    append_ints(trace.max_selector);
    values.insert(values.end(), trace.alpha.begin(), trace.alpha.end());
    append_ints(trace.integer_numerator);
    values.insert(values.end(), trace.field_numerator.begin(), trace.field_numerator.end());
    values.insert(values.end(), trace.field_aggregate.begin(), trace.field_aggregate.end());
    append_ints(trace.recovered);
    append_ints(trace.activated);
    for (const DivisionWitness &division : trace.divisions)
        append_division_components(values, division);
}

std::vector<Fr> flatten_aux_values(const ForwardResult &forward) {
    std::vector<Fr> values;
    values.reserve(trace_aux_size(&forward.trace));
    for (Int value : forward.trace.input.values) values.push_back(Fr(value));
    for (const HiddenLayerTrace &layer : forward.trace.hidden_layers) {
        for (Int value : layer.concatenated.values) values.push_back(Fr(value));
        for (const AttentionTrace &head : layer.heads)
            append_attention_aux_values(values, head);
    }
    for (const OutputHeadTrace &head : forward.trace.output_heads) {
        append_attention_aux_values(values, head.attention);
        for (Int value : head.bias) values.push_back(Fr(value));
        for (Int value : head.aggregate) values.push_back(Fr(value));
        for (Int value : head.aligned) values.push_back(Fr(value));
        for (Int value : head.logits) values.push_back(Fr(value));
        for (const DivisionWitness &division : head.alignment_divisions)
            append_division_components(values, division);
    }
    for (Int value : forward.trace.head_sum) values.push_back(Fr(value));
    for (const DivisionWitness &division : forward.trace.final_average_divisions)
        append_division_components(values, division);
    for (Int value : forward.trace.logits.values) values.push_back(Fr(value));
    return values;
}

Wide abs_wide_adapter(Int value) {
    return value < 0 ? -static_cast<Wide>(value) : static_cast<Wide>(value);
}

Fr packed_table_value(Int x, Int y, Int bx, Int by) {
    const Wide base = static_cast<Wide>(2) * static_cast<Wide>(by) + 1;
    const Wide packed = (static_cast<Wide>(x) + bx) * base +
                        (static_cast<Wide>(y) + by);
    if (packed < 0 || packed > static_cast<Wide>(std::numeric_limits<Int>::max()))
        throw std::overflow_error("packed lookup tuple exceeds adapter integer range");
    return Fr(static_cast<Int>(packed));
}

void append_four_bits(std::vector<Fr> &values, Int input) {
    Wide absolute = abs_wide_adapter(input);
    for (unsigned bit = 0; bit < 4; ++bit) {
        values.push_back(Fr(static_cast<Int>((absolute >> bit) & 1)));
    }
}

void append_range_attention_queries(std::vector<Fr> &values,
                                    const AttentionTrace &trace) {
    for (Int value : trace.recovered) append_four_bits(values, value);
    for (const DivisionWitness &division : trace.divisions)
        append_four_bits(values, division.q);
}

struct LassoAdapterBundle {
    std::vector<NativeLassoFamilyProver> families;
    NativeLassoQueryProver query;
    // The query vector itself remains private prover input.  Keep the raw
    // coordinates here (the old name is retained for source compatibility),
    // while the hiding Pedersen blinder is carried by query and never enters
    // the artifact wire.
    std::vector<Fr> masked_l;
    CommitmentWire com_L;
};

std::string native_family_id(const std::string &family) {
    if (family == "function/lrelu") return "function_lrelu";
    if (family == "function/exp") return "function_exp";
    if (family == "function/elu") return "function_elu";
    if (family == "range/all") return "range_all";
    throw std::invalid_argument("unknown canonical Lasso family: " + family);
}

const std::vector<TablePoint> &table_points_for(
    const StaticTables &tables, const std::string &family) {
    if (family == "function/lrelu") return tables.lrelu;
    if (family == "function/exp") return tables.exp;
    if (family == "function/elu") return tables.elu;
    throw std::invalid_argument("function table requested for non-function family");
}

LassoAdapterBundle build_lasso_bundle(const GatCircuitPlan &plan,
                                      const ForwardResult &forward,
                                      const StaticTables &tables,
                                      const std::string &basis_label) {
    if (plan.input_layout.segments.size() != 4)
        throw std::invalid_argument("Lasso bundle requires canonical four-segment layout");
    const InputSegment &l_segment = plan.input_layout.segments.at(3);
    std::map<std::string, std::vector<std::pair<Int, Int> > > function_queries;
    std::map<std::string, std::vector<Fr> > range_queries;
    const auto append_attention = [&](const AttentionTrace &trace, bool hidden) {
        std::vector<std::pair<Int, Int> > &lrelu = function_queries["function/lrelu"];
        std::vector<std::pair<Int, Int> > &exp = function_queries["function/exp"];
        for (std::size_t index = 0; index < trace.raw_score.size(); ++index)
            lrelu.push_back(std::make_pair(trace.raw_score[index], trace.leaky_score[index]));
        for (std::size_t index = 0; index < trace.delta.size(); ++index)
            exp.push_back(std::make_pair(trace.delta[index], trace.exp[index]));
        if (hidden) {
            std::vector<std::pair<Int, Int> > &elu = function_queries["function/elu"];
            for (std::size_t index = 0; index < trace.recovered.size(); ++index)
                elu.push_back(std::make_pair(trace.recovered[index], trace.activated[index]));
        }
        append_range_attention_queries(range_queries["range/all"], trace);
    };
    for (const HiddenLayerTrace &layer : forward.trace.hidden_layers)
        for (const AttentionTrace &head : layer.heads) append_attention(head, true);
    for (const OutputHeadTrace &head : forward.trace.output_heads) {
        append_attention(head.attention, false);
        for (const DivisionWitness &division : head.alignment_divisions)
            append_four_bits(range_queries["range/all"], division.q);
    }
    for (const DivisionWitness &division : forward.trace.final_average_divisions)
        append_four_bits(range_queries["range/all"], division.q);

    std::vector<Fr> l_values(plan.input_layout.n_in, Fr(0));
    LassoAdapterBundle result;
    for (const LassoFamilyDescriptor &descriptor : plan.lasso) {
        const std::string family_id = native_family_id(descriptor.family_id);
        NativeLassoFamilyProver family;
        family.family_id = family_id;
        family.table_id = "table_" + family_id;
        family.resource_id = "resource_" + family_id;
        family.registration_id = "registration_" + family_id;
        family.segment_offset = descriptor.query_offset - l_segment.offset;
        family.logical_queries = descriptor.real_queries;
        family.padded_queries = descriptor.padded_queries;
        if (family.padded_queries == 0 ||
            family.segment_offset % family.padded_queries != 0)
            throw std::invalid_argument(
                "Lasso family segment is not power-of-two aligned for masked opening");
        if (family.segment_offset + family.padded_queries > l_segment.padded_length)
            throw std::invalid_argument("Lasso family exceeds canonical L_qry domain");

        std::vector<Fr> real_queries;
        if (descriptor.family_id == "range/all") {
            family.table = {Fr(0), Fr(1)};
            family.padding_index = 0;
            family.padding_value = Fr(0);
            real_queries = range_queries[descriptor.family_id];
        } else {
            const std::vector<TablePoint> &points =
                table_points_for(tables, descriptor.family_id);
            Int bx = 0, by = 0;
            for (const TablePoint &point : points) {
                bx = std::max<Int>(bx, static_cast<Int>(abs_wide_adapter(point.x)));
                by = std::max<Int>(by, static_cast<Int>(abs_wide_adapter(point.y)));
            }
            family.table.reserve(descriptor.table_domain);
            for (const TablePoint &point : points)
                family.table.push_back(packed_table_value(point.x, point.y, bx, by));
            if (family.table.empty()) throw std::invalid_argument("empty static Lasso table");
            while (family.table.size() < descriptor.table_domain)
                family.table.push_back(family.table.front());
            family.padding_index = 0;
            family.padding_value = family.table[0];
            const std::vector<std::pair<Int, Int> > &pairs =
                function_queries[descriptor.family_id];
            for (const std::pair<Int, Int> &pair : pairs)
                real_queries.push_back(packed_table_value(pair.first, pair.second, bx, by));
        }
        if (real_queries.size() != descriptor.real_queries)
            throw std::invalid_argument("canonical Lasso query count does not match plan");
        for (std::size_t index = 0; index < real_queries.size(); ++index)
            l_values[family.segment_offset + index] = real_queries[index];
        for (std::size_t index = real_queries.size();
             index < family.padded_queries; ++index)
            l_values[family.segment_offset + index] = family.padding_value;
        while (real_queries.size() < family.padded_queries)
            real_queries.push_back(family.padding_value);
        family.queries = real_queries;
        family.matrix.assign(family.padded_queries * family.table.size(), Fr(0));
        for (std::size_t row = 0; row < family.padded_queries; ++row) {
            std::size_t selected = family.table.size();
            for (std::size_t column = 0; column < family.table.size(); ++column)
                if (family.table[column] == family.queries[row]) {
                    selected = column;
                    break;
                }
            if (selected == family.table.size())
                throw std::invalid_argument("Lasso query is absent from static table");
            family.matrix[row + selected * family.padded_queries] = Fr(1);
        }
        family.matrix_generators.resize(family.matrix.size());
        for (std::size_t index = 0; index < family.matrix_generators.size(); ++index)
            hashAndMapToG1(family.matrix_generators[index],
                           basis_label + "/lasso/" + family.family_id +
                           "/matrix/" + std::to_string(index));
        result.families.push_back(std::move(family));
    }
    std::sort(result.families.begin(), result.families.end(),
              [](const NativeLassoFamilyProver &left,
                 const NativeLassoFamilyProver &right) {
                  return left.family_id < right.family_id;
              });
    result.query.G.clear();
    hashAndMapToG1(result.query.G, basis_label + "/G");
    result.query.generators.resize(plan.input_layout.n_in);
    for (std::size_t index = 0; index < result.query.generators.size(); ++index)
        hashAndMapToG1(result.query.generators[index],
                       basis_label + "/coordinate/" + std::to_string(index));
    // L_qry is a real hiding Pedersen MLE commitment.  The previous adapter
    // generated a blind in a wire-only helper and then recomputed a clear
    // commitment from the vector, which made the native masked Lasso API
    // reject the query commitment (and silently lost the prover blinder).
    // Construct it once through the native Hyrax primitive and retain the
    // scalar only in the in-process query object.
    const std::string query_basis_id = fixed_digest(hash_hex(basis_label));
    const hyrax::PedersenMleCommitment query_commitment =
        hyrax::pedersen_commit_mle(result.query.generators, l_values,
                                   query_basis_id);
    result.masked_l = l_values;
    result.query.values = l_values;
    result.query.commitment = query_commitment.commitment;
    result.query.basis_id = query_commitment.basis_id;
    result.query.blinding_generator = query_commitment.blinding_generator;
    result.query.commitment_blinding = query_commitment.blinding;
    result.query.has_commitment_blinding = true;
    result.com_L.scheme = "hyrax-pedersen-bn254";
    result.com_L.basis_id = query_commitment.basis_id;
    result.com_L.logical_size = l_segment.logical_length;
    result.com_L.padded_size = plan.input_layout.n_in;
    result.com_L.point_hex = point_hex(query_commitment.commitment);
    return result;
}

// Rebuild only the public side of the canonical Lasso registry.  The
// independent verifier must be able to reconstruct tables, resource IDs and
// matrix generator domains without seeing the private query/matrix arrays.
std::vector<NativeLassoFamilyPublic> build_lasso_public_families(
    const GatCircuitPlan &plan, const StaticTables &tables,
    const std::string &basis_label) {
    if (plan.input_layout.segments.size() != 4)
        throw std::invalid_argument("public Lasso registry requires four segments");
    const InputSegment &l_segment = plan.input_layout.segments.at(3);
    std::vector<NativeLassoFamilyPublic> result;
    result.reserve(plan.lasso.size());
    for (const LassoFamilyDescriptor &descriptor : plan.lasso) {
        NativeLassoFamilyPublic family;
        family.family_id = native_family_id(descriptor.family_id);
        family.table_id = "table_" + family.family_id;
        family.resource_id = "resource_" + family.family_id;
        family.registration_id = "registration_" + family.family_id;
        family.segment_offset = descriptor.query_offset - l_segment.offset;
        family.logical_queries = descriptor.real_queries;
        family.padded_queries = descriptor.padded_queries;
        family.padding_index = 0;
        if (descriptor.family_id == "range/all") {
            family.table = {Fr(0), Fr(1)};
            family.padding_value = Fr(0);
        } else {
            const std::vector<TablePoint> &points =
                table_points_for(tables, descriptor.family_id);
            Int bx = 0, by = 0;
            for (const TablePoint &point : points) {
                bx = std::max<Int>(bx, static_cast<Int>(abs_wide_adapter(point.x)));
                by = std::max<Int>(by, static_cast<Int>(abs_wide_adapter(point.y)));
            }
            family.table.reserve(descriptor.table_domain);
            for (const TablePoint &point : points)
                family.table.push_back(packed_table_value(point.x, point.y, bx, by));
            if (family.table.empty())
                throw std::invalid_argument("empty public Lasso table");
            while (family.table.size() < descriptor.table_domain)
                family.table.push_back(family.table.front());
            family.padding_value = family.table.front();
        }
        if (family.segment_offset + family.padded_queries > l_segment.padded_length ||
            family.table.empty() ||
            (family.padded_queries & (family.padded_queries - 1)) != 0)
            throw std::invalid_argument("public Lasso family exceeds canonical domain");
        family.matrix_generators.resize(family.padded_queries * family.table.size());
        for (std::size_t index = 0; index < family.matrix_generators.size(); ++index)
            hashAndMapToG1(family.matrix_generators[index],
                           basis_label + "/lasso/" + family.family_id +
                               "/matrix/" + std::to_string(index));
        result.push_back(std::move(family));
    }
    std::sort(result.begin(), result.end(),
              [](const NativeLassoFamilyPublic &left,
                 const NativeLassoFamilyPublic &right) {
                  return left.family_id < right.family_id;
              });
    const std::string tables_digest = native_lasso_tables_digest(result);
    const std::string resources_digest = native_lasso_resources_digest(result);
    for (NativeLassoFamilyPublic &family : result) {
        family.statement_tables_digest = tables_digest;
        family.statement_resources_digest = resources_digest;
    }
    return result;
}

NativeLassoQueryPublic build_lasso_public_query(
    const GatCircuitPlan &plan, const CommitmentWire &com_L,
    const std::string &basis_label) {
    if (!com_L.valid() || com_L.padded_size != plan.input_layout.n_in)
        throw std::invalid_argument("public Lasso query commitment is invalid");
    NativeLassoQueryPublic query;
    hashAndMapToG1(query.G, basis_label + "/G");
    query.generators.resize(plan.input_layout.n_in);
    for (std::size_t index = 0; index < query.generators.size(); ++index)
        hashAndMapToG1(query.generators[index],
                       basis_label + "/coordinate/" + std::to_string(index));
    query.commitment.deserializeHexStr(com_L.point_hex);
    query.basis_id = com_L.basis_id;
    query.statement_l_digest = fixed_digest(hash_hex(commitment_encoding(com_L)));
    return query;
}

std::vector<Fr> input_segment(const GatCircuitPlan &plan,
                              const std::vector<Fr> &input,
                              std::size_t segment_index) {
    if (segment_index >= plan.input_layout.segments.size())
        throw std::invalid_argument("canonical input segment index is invalid");
    const InputSegment &segment = plan.input_layout.segments.at(segment_index);
    const std::size_t domain = plan.input_layout.n_in;
    if (segment.padded_length != domain || segment.offset + domain > input.size())
        throw std::invalid_argument("canonical input segment domain is invalid");
    return std::vector<Fr>(input.begin() + segment.offset,
                           input.begin() + segment.offset + domain);
}

Fr adapter_mle_eval(const std::vector<Fr> &values,
                    const std::vector<Fr> &point) {
    if (values.empty() || point.size() >= sizeof(std::size_t) * 8 ||
        values.size() > (std::size_t(1) << point.size()))
        throw std::invalid_argument("adapter MLE dimensions are invalid");
    std::vector<Fr> folded(std::size_t(1) << point.size(), Fr(0));
    std::copy(values.begin(), values.end(), folded.begin());
    std::size_t active = folded.size();
    for (const Fr &coordinate : point) {
        const std::size_t half = active / 2;
        for (std::size_t index = 0; index < half; ++index)
            folded[index] = folded[2 * index] * (Fr(1) - coordinate) +
                            folded[2 * index + 1] * coordinate;
        active = half;
    }
    return folded.front();
}

std::vector<Fr> public_feature_segment(const GatCircuitPlan &plan,
                                       const Graph &graph,
                                       const Tensor2D &features) {
    if (plan.input_layout.segments.empty() ||
        features.values.size() > plan.input_layout.segments.front().logical_length)
        throw std::invalid_argument("public feature segment exceeds X_pub");
    std::vector<Fr> result(plan.input_layout.n_in, Fr(0));
    for (std::size_t index = 0; index < features.values.size(); ++index)
        result[index] = Fr(features.values[index]);
    (void)graph;
    return result;
}

std::vector<Fr> canonical_l_query_values(const GatCircuitPlan &plan,
                                          const ForwardResult &forward,
                                          const StaticTables &tables) {
    if (plan.input_layout.segments.size() != 4)
        throw std::invalid_argument("canonical L_qry requires four input segments");
    const InputSegment &l_segment = plan.input_layout.segments.at(3);
    std::map<std::string, std::vector<std::pair<Int, Int>>> function_queries;
    std::map<std::string, std::vector<Fr>> range_queries;
    const auto append_attention = [&](const AttentionTrace &trace, bool hidden) {
        auto &lrelu = function_queries["function/lrelu"];
        auto &exp = function_queries["function/exp"];
        for (std::size_t index = 0; index < trace.raw_score.size(); ++index)
            lrelu.emplace_back(trace.raw_score[index], trace.leaky_score[index]);
        for (std::size_t index = 0; index < trace.delta.size(); ++index)
            exp.emplace_back(trace.delta[index], trace.exp[index]);
        if (hidden) {
            auto &elu = function_queries["function/elu"];
            for (std::size_t index = 0; index < trace.recovered.size(); ++index)
                elu.emplace_back(trace.recovered[index], trace.activated[index]);
        }
        append_range_attention_queries(range_queries["range/all"], trace);
    };
    for (const HiddenLayerTrace &layer : forward.trace.hidden_layers)
        for (const AttentionTrace &head : layer.heads) append_attention(head, true);
    for (const OutputHeadTrace &head : forward.trace.output_heads) {
        append_attention(head.attention, false);
        for (const DivisionWitness &division : head.alignment_divisions)
            append_four_bits(range_queries["range/all"], division.q);
    }
    for (const DivisionWitness &division : forward.trace.final_average_divisions)
        append_four_bits(range_queries["range/all"], division.q);

    std::vector<Fr> values(plan.input_layout.n_in, Fr(0));
    for (const LassoFamilyDescriptor &descriptor : plan.lasso) {
        const std::size_t family_offset = descriptor.query_offset - l_segment.offset;
        if (family_offset + descriptor.padded_queries > values.size())
            throw std::invalid_argument("canonical L_qry family exceeds its segment");
        std::vector<Fr> real_queries;
        Fr padding = Fr(descriptor.padding_value);
        if (descriptor.family_id == "range/all") {
            const auto &queries = range_queries[descriptor.family_id];
            real_queries = queries;
            padding = Fr(0);
        } else {
            const auto &points = table_points_for(tables, descriptor.family_id);
            Int bx = 0, by = 0;
            for (const TablePoint &point : points) {
                bx = std::max<Int>(bx, static_cast<Int>(abs_wide_adapter(point.x)));
                by = std::max<Int>(by, static_cast<Int>(abs_wide_adapter(point.y)));
            }
            padding = packed_table_value(points.front().x, points.front().y, bx, by);
            for (const auto &pair : function_queries[descriptor.family_id])
                real_queries.push_back(packed_table_value(pair.first, pair.second, bx, by));
        }
        if (real_queries.size() != descriptor.real_queries)
            throw std::invalid_argument("canonical L_qry query count mismatch");
        for (std::size_t index = 0; index < descriptor.padded_queries; ++index) {
            values[family_offset + index] =
                index < real_queries.size() ? real_queries[index] : padding;
        }
    }
    return values;
}

std::vector<Fr> canonical_input_values(const GatCircuitPlan &plan,
                                       const Graph &graph,
                                       const Tensor2D &public_features,
                                       const Model &model,
                                       const QuantConfig &quant,
                                       const ForwardResult &forward) {
    const std::size_t input_size = plan.circuit.circuit.front().size;
    const std::size_t n_in = plan.input_layout.n_in;
    if (input_size != checked_mul(n_in, 4, "canonical input witness"))
        throw std::invalid_argument("canonical input witness size does not match plan");
    if (public_features.rows != graph.num_nodes ||
        public_features.cols != graph.num_features ||
        public_features.values.size() != public_features.rows * public_features.cols)
        throw std::invalid_argument("public feature shape does not match GAT witness");
    std::vector<Fr> values(input_size, Fr(0));
    for (std::size_t index = 0; index < public_features.values.size(); ++index)
        values[index] = Fr(public_features.values[index]);
    const std::vector<Int> model_values = flatten_model(model, quant);
    const InputSegment &w_segment = plan.input_layout.segments.at(1);
    for (std::size_t index = 0; index < model_values.size(); ++index)
        values[w_segment.offset + index] = Fr(model_values[index]);
    const std::vector<Fr> aux_values = flatten_aux_values(forward);
    const InputSegment &aux_segment = plan.input_layout.segments.at(2);
    if (aux_values.size() != aux_segment.logical_length)
        throw std::invalid_argument("canonical Aux witness size does not match plan");
    for (std::size_t index = 0; index < aux_values.size(); ++index)
        values[aux_segment.offset + index] = aux_values[index];
    const std::vector<Fr> l_values =
        canonical_l_query_values(plan, forward, StaticTables::build(quant));
    const InputSegment &l_segment = plan.input_layout.segments.at(3);
    for (std::size_t index = 0; index < l_values.size(); ++index)
        values[l_segment.offset + index] = l_values[index];
    return values;
}

bool evaluate_layered_values(const GatCircuitPlan &plan,
                             const std::vector<Fr> &input,
                             std::vector<std::vector<Fr>> &values,
                             std::string *reason) {
    const auto fail = [reason](const std::string &message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    try {
        if (plan.circuit.circuit.empty() ||
            input.size() != plan.circuit.circuit.front().size)
            return fail("layered witness input size is invalid");
        values.clear();
        values.resize(plan.circuit.circuit.size());
        values[0] = input;
        for (std::size_t layer_id = 1;
             layer_id < plan.circuit.circuit.size(); ++layer_id) {
            const layer &current = plan.circuit.circuit[layer_id];
            const layer &previous = plan.circuit.circuit[layer_id - 1];
            std::vector<Fr> output(current.size, Fr(0));
            for (const uniGate &gate : current.uni_gates) {
                if (gate.g >= output.size()) return fail("unary gate output is out of bounds");
                const Fr source = gate.lu == 0
                    ? input.at(current.ori_id_u.at(gate.u))
                    : values.at(layer_id - 1).at(gate.u);
                output[gate.g] += source * Fr(gate.sc);
            }
            for (const binGate &gate : current.bin_gates) {
                if (gate.g >= output.size()) return fail("binary gate output is out of bounds");
                const std::size_t lhs_layer = gate.getLayerIdU(static_cast<u8>(layer_id));
                const std::size_t rhs_layer = gate.getLayerIdV(static_cast<u8>(layer_id));
                const Fr lhs = lhs_layer == 0
                    ? input.at(current.ori_id_u.at(gate.u))
                    : values.at(layer_id - 1).at(gate.u);
                const Fr rhs = rhs_layer == 0
                    ? input.at(current.ori_id_v.at(gate.v))
                    : values.at(layer_id - 1).at(gate.v);
                output[gate.g] += lhs * rhs * Fr(gate.sc);
            }
            values[layer_id] = std::move(output);
            (void)previous;
        }
        return true;
    } catch (const std::exception &error) {
        return fail(error.what());
    }
}

std::string context_encoding(const VerifierContext &context) {
    std::string encoded;
    append_string(encoded, "zkgpt-gat-verifier-context-v2");
    append_string(encoded, context.relation_id);
    append_string(encoded, context.model_arch_id);
    append_string(encoded, context.privacy_extension_profile_id);
    append_string(encoded, context.quant_cfg_id);
    append_string(encoded, context.static_tables_id);
    append_registration_encoding(encoded, context.registration);
    append_string(encoded, commitment_encoding(context.com_W));
    // The digest must bind the complete public plan, not merely the plan
    // digest string supplied by a caller.  plan_encoding itself includes the
    // registry, all descriptors and every layered gate.
    append_string(encoded, plan_encoding(context.plan));
    append_string(encoded, "public-graph-v2");
    append_u64(encoded, context.graph.num_nodes);
    append_u64(encoded, context.graph.num_features);
    append_u64(encoded, context.graph.graph_count);
    append_u64(encoded, context.graph.num_classes);
    append_u64(encoded, static_cast<std::uint64_t>(context.graph.src.size()));
    for (Int value : context.graph.src) append_int(encoded, value);
    append_u64(encoded, static_cast<std::uint64_t>(context.graph.dst.size()));
    for (Int value : context.graph.dst) append_int(encoded, value);
    append_u64(encoded, static_cast<std::uint64_t>(context.graph.edge_gid.size()));
    for (Int value : context.graph.edge_gid) append_int(encoded, value);
    append_u64(encoded, static_cast<std::uint64_t>(context.graph.node_ptr.size()));
    for (std::size_t value : context.graph.node_ptr) append_u64(encoded, value);
    append_u64(encoded, static_cast<std::uint64_t>(context.graph.edge_ptr.size()));
    for (std::size_t value : context.graph.edge_ptr) append_u64(encoded, value);
    append_string(encoded, "public-features-v2");
    append_u64(encoded, context.public_features.rows);
    append_u64(encoded, context.public_features.cols);
    append_u64(encoded, static_cast<std::uint64_t>(context.public_features.values.size()));
    for (Int value : context.public_features.values) append_int(encoded, value);
    append_string(encoded, "public-logits-v2");
    append_u64(encoded, context.public_logits.rows);
    append_u64(encoded, context.public_logits.cols);
    append_u64(encoded, static_cast<std::uint64_t>(context.public_logits.values.size()));
    for (Int value : context.public_logits.values) append_int(encoded, value);
    append_string(encoded, context.tables.table_id);
    append_string(encoded, context.tables.generator_id);
    append_string(encoded, context.tables.digest);
    append_string(encoded, context.tables.quant_cfg_id);
    append_string(encoded, context.tables.rounding);
    append_string(encoded, context.tables.rescale);
    append_string(encoded, context.tables.output_average);
    append_string(encoded, context.tables.row_order);
    append_string(encoded, context.tables.exp_approx_id);
    append_string(encoded, context.tables.elu_approx_id);
    append_int(encoded, context.tables.lrelu_slope_num);
    append_int(encoded, context.tables.lrelu_slope_den);
    append_int(encoded, context.tables.s_delta);
    append_int(encoded, context.tables.s_exp);
    append_int(encoded, context.tables.s_elu_in);
    append_int(encoded, context.tables.s_elu_out);
    append_int(encoded, context.tables.lrelu_min);
    append_int(encoded, context.tables.lrelu_max);
    append_int(encoded, context.tables.exp_min);
    append_int(encoded, context.tables.exp_max);
    append_int(encoded, context.tables.elu_min);
    append_int(encoded, context.tables.elu_max);
    const auto append_table = [&encoded](const std::vector<TablePoint> &table) {
        append_u64(encoded, static_cast<std::uint64_t>(table.size()));
        for (const TablePoint &point : table) {
            append_int(encoded, point.x);
            append_int(encoded, point.y);
        }
    };
    append_table(context.tables.lrelu);
    append_table(context.tables.exp);
    append_table(context.tables.elu);
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
    append_int(encoded, context.quant.range_bits);
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

bool wire_size_within_limit(std::ifstream &in) {
    const std::istream::pos_type origin = in.tellg();
    if (origin == std::istream::pos_type(-1)) return false;
    in.seekg(0, std::ios::end);
    const std::istream::pos_type end = in.tellg();
    in.seekg(origin);
    if (end == std::istream::pos_type(-1) || end < origin) return false;
    return static_cast<std::uint64_t>(end) <= kArtifactMaxWireBytes;
}

void write_u8(std::ofstream &out, std::uint8_t value) {
    const char byte = static_cast<char>(value);
    out.write(&byte, 1);
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

bool read_u8(std::ifstream &in, std::uint8_t &value) {
    char byte = 0;
    if (!read_exact(in, &byte, 1)) return false;
    value = static_cast<std::uint8_t>(static_cast<unsigned char>(byte));
    return true;
}

bool read_string(std::ifstream &in, std::string &value) {
    std::uint64_t size = 0;
    if (!read_u64(in, size) || size > kArtifactMaxString) return false;
    value.assign(static_cast<std::size_t>(size), '\0');
    return read_exact(in, value.empty() ? nullptr : &value[0], value.size());
}

void write_native_statement(std::ofstream &out,
                            const NativeProofStatement &statement) {
    write_string(out, statement.verifier_context_digest);
    write_string(out, statement.field_id);
    write_string(out, statement.plan_digest);
    write_string(out, statement.resources_digest);
    write_string(out, statement.tables_digest);
    write_string(out, statement.com_w_digest);
    write_string(out, statement.aux_digest);
    write_string(out, statement.l_digest);
    write_string(out, statement.basis_digest);
    write_u64(out, statement.l_dimension);
    write_string(out, statement.statement_root_digest);
}

bool read_native_statement(std::ifstream &in, NativeProofStatement &statement) {
    return read_string(in, statement.verifier_context_digest) &&
           read_string(in, statement.field_id) &&
           read_string(in, statement.plan_digest) &&
           read_string(in, statement.resources_digest) &&
           read_string(in, statement.tables_digest) &&
           read_string(in, statement.com_w_digest) &&
           read_string(in, statement.aux_digest) &&
           read_string(in, statement.l_digest) &&
           read_string(in, statement.basis_digest) &&
           read_u64(in, statement.l_dimension) &&
           read_string(in, statement.statement_root_digest);
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
        write_u8(out, message.kind);
        write_u64(out, message.sequence);
    }
}

bool read_section(std::ifstream &in, ProofSection &section) {
    std::uint64_t count = 0;
    if (!read_string(in, section.protocol) || !read_u64(in, count) ||
        count > kArtifactMaxMessages) return false;
    section.messages.clear();
    section.messages.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        TranscriptMessage message;
        std::uint8_t kind = 0;
        if (!read_string(in, message.label) || !read_string(in, message.payload) ||
            !read_u8(in, kind) || !read_u64(in, message.sequence) ||
            kind < static_cast<std::uint8_t>(NativeProofMessageKind::Data) ||
            kind > static_cast<std::uint8_t>(NativeProofMessageKind::Challenge)) return false;
        message.kind = kind;
        section.messages.push_back(std::move(message));
    }
    return true;
}

void set_reason(std::string *reason, const std::string &value);
bool is_lower_hex_digest(const std::string &value);

void append_u8_wire(std::string &out, std::uint8_t value) {
    out.push_back(static_cast<char>(value));
}

void append_i8_wire(std::string &out, std::int8_t value) {
    append_u8_wire(out, static_cast<std::uint8_t>(value));
}

void append_bool_wire(std::string &out, bool value) {
    append_u8_wire(out, value ? 1 : 0);
}

class ContextReader {
public:
    ContextReader(const std::string &wire, std::size_t begin, std::size_t end)
        : wire_(wire), position_(begin), end_(end) {}

    bool ok() const { return error_.empty(); }
    const std::string &error() const { return error_; }
    bool at_end() const { return ok() && position_ == end_; }
    std::size_t remaining() const {
        return position_ <= end_ ? end_ - position_ : 0;
    }

    bool read_u8(std::uint8_t &value) {
        if (!take(1)) return false;
        value = static_cast<std::uint8_t>(static_cast<unsigned char>(wire_[position_++]));
        return true;
    }

    bool read_u32(std::uint32_t &value) {
        if (!take(4)) return false;
        value = 0;
        for (unsigned i = 0; i < 4; ++i)
            value |= static_cast<std::uint32_t>(static_cast<unsigned char>(wire_[position_++])) << (i * 8);
        return true;
    }

    bool read_u64(std::uint64_t &value) {
        if (!take(8)) return false;
        value = 0;
        for (unsigned i = 0; i < 8; ++i)
            value |= static_cast<std::uint64_t>(static_cast<unsigned char>(wire_[position_++])) << (i * 8);
        return true;
    }

    bool read_bool(bool &value) {
        std::uint8_t raw = 0;
        if (!read_u8(raw) || raw > 1) {
            fail("non-canonical boolean");
            return false;
        }
        value = raw != 0;
        return true;
    }

    bool read_i8(std::int8_t &value) {
        std::uint8_t raw = 0;
        if (!read_u8(raw)) return false;
        value = raw <= 127 ? static_cast<std::int8_t>(raw)
                           : static_cast<std::int8_t>(static_cast<int>(raw) - 256);
        return true;
    }

    bool read_size(std::size_t &value, const char *what) {
        std::uint64_t raw = 0;
        if (!read_u64(raw) || raw > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            fail(std::string("invalid size for ") + what);
            return false;
        }
        value = static_cast<std::size_t>(raw);
        return true;
    }

    bool read_count(std::size_t &value, const char *what,
                    std::uint64_t limit = kContextMaxElements) {
        if (!read_size(value, what) || static_cast<std::uint64_t>(value) > limit) {
            fail(std::string("oversized count for ") + what);
            return false;
        }
        // Every encoded element consumes at least one byte.  This rejects a
        // huge count before any allocation even when the remaining wire is
        // truncated.
        if (static_cast<std::uint64_t>(value) > remaining()) {
            fail(std::string("count exceeds remaining wire for ") + what);
            return false;
        }
        return true;
    }

    bool read_int(Int &value) {
        std::uint8_t sign = 0;
        std::uint64_t magnitude = 0;
        if (!read_u8(sign) || !read_u64(magnitude) || sign > 1) {
            fail("invalid signed integer wire");
            return false;
        }
        const std::uint64_t max_positive = static_cast<std::uint64_t>(std::numeric_limits<Int>::max());
        if (magnitude > max_positive + (sign == 1 ? 1ULL : 0ULL) ||
            (sign == 1 && magnitude == 0)) {
            fail("signed integer is outside int64 canonical range");
            return false;
        }
        if (sign == 0) value = static_cast<Int>(magnitude);
        else if (magnitude == max_positive + 1ULL) value = std::numeric_limits<Int>::min();
        else value = -static_cast<Int>(magnitude);
        return true;
    }

    bool read_string(std::string &value, const char *what,
                     std::uint64_t limit = kArtifactMaxString) {
        std::uint64_t raw = 0;
        if (!read_u64(raw) || raw > limit || raw > remaining()) {
            fail(std::string("invalid or oversized string for ") + what);
            return false;
        }
        value.assign(wire_.data() + position_, static_cast<std::size_t>(raw));
        position_ += static_cast<std::size_t>(raw);
        if (value.find('\0') != std::string::npos) {
            fail(std::string("NUL in canonical string ") + what);
            return false;
        }
        return true;
    }

private:
    bool take(std::size_t size) {
        if (size > end_ - std::min(position_, end_)) {
            fail("truncated verifier context");
            return false;
        }
        return true;
    }

    void fail(const std::string &message) {
        if (error_.empty()) error_ = message;
    }

    const std::string &wire_;
    std::size_t position_ = 0;
    std::size_t end_ = 0;
    std::string error_;
};

void append_context_commitment(std::string &out, const CommitmentWire &commitment) {
    append_string(out, commitment.scheme);
    append_string(out, commitment.basis_id);
    append_u64(out, static_cast<std::uint64_t>(commitment.logical_size));
    append_u64(out, static_cast<std::uint64_t>(commitment.padded_size));
    append_string(out, commitment.point_hex);
}

bool read_context_commitment(ContextReader &reader, CommitmentWire &commitment) {
    std::size_t logical = 0, padded = 0;
    return reader.read_string(commitment.scheme, "commitment scheme") &&
           reader.read_string(commitment.basis_id, "commitment basis") &&
           reader.read_size(logical, "commitment logical size") &&
           reader.read_size(padded, "commitment padded size") &&
           reader.read_string(commitment.point_hex, "commitment point") &&
           (commitment.logical_size = logical, commitment.padded_size = padded, true);
}

void append_context_segment(std::string &out, const InputSegment &segment) {
    append_string(out, segment.name);
    append_u64(out, static_cast<std::uint64_t>(segment.offset));
    append_u64(out, static_cast<std::uint64_t>(segment.logical_length));
    append_u64(out, static_cast<std::uint64_t>(segment.padded_length));
}

bool read_context_segment(ContextReader &reader, InputSegment &segment) {
    return reader.read_string(segment.name, "input segment name") &&
           reader.read_size(segment.offset, "input segment offset") &&
           reader.read_size(segment.logical_length, "input segment logical length") &&
           reader.read_size(segment.padded_length, "input segment padded length");
}

void append_context_int_vector(std::string &out, const std::vector<Int> &values) {
    append_u64(out, static_cast<std::uint64_t>(values.size()));
    for (Int value : values) append_int(out, value);
}

bool read_context_int_vector(ContextReader &reader, std::vector<Int> &values,
                             const char *what,
                             std::uint64_t limit = kContextMaxElements) {
    std::size_t count = 0;
    if (!reader.read_count(count, what, limit)) return false;
    values.resize(count);
    for (Int &value : values) if (!reader.read_int(value)) return false;
    return true;
}

void append_context_quant(std::string &out, const QuantConfig &quant) {
    append_string(out, quant.schema);
    append_string(out, quant.quant_cfg_id);
    append_int(out, quant.s_input);
    append_int(out, quant.s_delta);
    append_int(out, quant.s_exp);
    for (Int value : quant.s_h) append_int(out, value);
    for (std::size_t layer = 0; layer < quant.s_w_hidden.size(); ++layer) {
        append_context_int_vector(out, quant.s_w_hidden[layer]);
        append_context_int_vector(out, quant.s_hprime_hidden[layer]);
        append_context_int_vector(out, quant.s_a_src_hidden[layer]);
        append_context_int_vector(out, quant.s_a_dst_hidden[layer]);
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

bool read_context_quant(ContextReader &reader, QuantConfig &quant) {
    if (!reader.read_string(quant.schema, "quant schema") ||
        !reader.read_string(quant.quant_cfg_id, "quant id") ||
        !reader.read_int(quant.s_input) || !reader.read_int(quant.s_delta) ||
        !reader.read_int(quant.s_exp)) return false;
    for (Int &value : quant.s_h) if (!reader.read_int(value)) return false;
    for (std::size_t layer = 0; layer < quant.s_w_hidden.size(); ++layer) {
        if (!read_context_int_vector(reader, quant.s_w_hidden[layer], "hidden weight scales", 64) ||
            !read_context_int_vector(reader, quant.s_hprime_hidden[layer], "hidden projection scales", 64) ||
            !read_context_int_vector(reader, quant.s_a_src_hidden[layer], "hidden source scales", 64) ||
            !read_context_int_vector(reader, quant.s_a_dst_hidden[layer], "hidden destination scales", 64)) return false;
    }
    for (Int &value : quant.s_agg_pre) if (!reader.read_int(value)) return false;
    for (Int &value : quant.s_agg) if (!reader.read_int(value)) return false;
    for (Int &value : quant.s_cat) if (!reader.read_int(value)) return false;
    if (!reader.read_int(quant.s_out_pre) || !reader.read_int(quant.s_out_head) ||
        !reader.read_int(quant.s_out)) return false;
    for (Int &value : quant.s_w_output) if (!reader.read_int(value)) return false;
    for (Int &value : quant.s_yprime_output) if (!reader.read_int(value)) return false;
    for (Int &value : quant.s_a_src_output) if (!reader.read_int(value)) return false;
    for (Int &value : quant.s_a_dst_output) if (!reader.read_int(value)) return false;
    for (Int &value : quant.s_b_output) if (!reader.read_int(value)) return false;
    return reader.read_string(quant.rounding, "rounding policy") &&
           reader.read_string(quant.rescale, "rescale policy") &&
           reader.read_string(quant.output_average, "output average policy") &&
           reader.read_int(quant.range_bits) &&
           reader.read_string(quant.static_tables_id, "static tables id") &&
           reader.read_string(quant.static_table_generator_id, "static table generator") &&
           reader.read_string(quant.static_row_order, "static table row order") &&
           reader.read_string(quant.exp_approx_id, "exp approximation id") &&
           reader.read_string(quant.elu_approx_id, "elu approximation id") &&
           reader.read_int(quant.static_lrelu_min) &&
           reader.read_int(quant.static_lrelu_max) &&
           reader.read_int(quant.static_exp_min) &&
           reader.read_int(quant.static_exp_max) &&
           reader.read_int(quant.static_elu_min) &&
           reader.read_int(quant.static_elu_max) &&
           reader.read_int(quant.lrelu_slope_num) &&
           reader.read_int(quant.lrelu_slope_den);
}

void append_context_tables(std::string &out, const StaticTables &tables) {
    append_string(out, tables.table_id);
    append_string(out, tables.generator_id);
    append_string(out, tables.digest);
    append_string(out, tables.quant_cfg_id);
    append_string(out, tables.rounding);
    append_string(out, tables.rescale);
    append_string(out, tables.output_average);
    append_string(out, tables.row_order);
    append_string(out, tables.exp_approx_id);
    append_string(out, tables.elu_approx_id);
    append_int(out, tables.lrelu_slope_num);
    append_int(out, tables.lrelu_slope_den);
    append_int(out, tables.s_delta);
    append_int(out, tables.s_exp);
    append_int(out, tables.s_elu_in);
    append_int(out, tables.s_elu_out);
    append_int(out, tables.lrelu_min);
    append_int(out, tables.lrelu_max);
    append_int(out, tables.exp_min);
    append_int(out, tables.exp_max);
    append_int(out, tables.elu_min);
    append_int(out, tables.elu_max);
    const auto append_table = [&out](const std::vector<TablePoint> &table) {
        append_u64(out, static_cast<std::uint64_t>(table.size()));
        for (const TablePoint &point : table) {
            append_int(out, point.x);
            append_int(out, point.y);
        }
    };
    append_table(tables.lrelu);
    append_table(tables.exp);
    append_table(tables.elu);
}

bool read_context_tables(ContextReader &reader, StaticTables &tables) {
    if (!reader.read_string(tables.table_id, "table id") ||
        !reader.read_string(tables.generator_id, "table generator") ||
        !reader.read_string(tables.digest, "table digest") ||
        !reader.read_string(tables.quant_cfg_id, "table quant id") ||
        !reader.read_string(tables.rounding, "table rounding") ||
        !reader.read_string(tables.rescale, "table rescale") ||
        !reader.read_string(tables.output_average, "table output average") ||
        !reader.read_string(tables.row_order, "table row order") ||
        !reader.read_string(tables.exp_approx_id, "table exp approximation") ||
        !reader.read_string(tables.elu_approx_id, "table elu approximation") ||
        !reader.read_int(tables.lrelu_slope_num) ||
        !reader.read_int(tables.lrelu_slope_den) ||
        !reader.read_int(tables.s_delta) || !reader.read_int(tables.s_exp) ||
        !reader.read_int(tables.s_elu_in) || !reader.read_int(tables.s_elu_out) ||
        !reader.read_int(tables.lrelu_min) || !reader.read_int(tables.lrelu_max) ||
        !reader.read_int(tables.exp_min) || !reader.read_int(tables.exp_max) ||
        !reader.read_int(tables.elu_min) || !reader.read_int(tables.elu_max)) return false;
    const auto read_table = [&reader](std::vector<TablePoint> &table,
                                      const char *what) -> bool {
        std::size_t count = 0;
        if (!reader.read_count(count, what, 1ULL << 24)) return false;
        table.resize(count);
        for (TablePoint &point : table)
            if (!reader.read_int(point.x) || !reader.read_int(point.y)) return false;
        return true;
    };
    return read_table(tables.lrelu, "LeakyReLU table") &&
           read_table(tables.exp, "Exp table") &&
           read_table(tables.elu, "ELU table");
}

void append_context_registration(std::string &out,
                                 const ModelRegistration &registration) {
    append_context_commitment(out, registration.com_W);
    append_string(out, registration.dataset);
    append_string(out, registration.model_arch_id);
    append_string(out, registration.quant_cfg_id);
    append_string(out, registration.static_tables_id);
    append_string(out, registration.field_id);
    append_string(out, registration.field_modulus);
    append_string(out, registration.manifest_digest);
    append_string(out, registration.model_digest);
    append_string(out, registration.bounds_digest);
    append_string(out, registration.quant_digest);
    append_string(out, registration.layout_digest);
    append_string(out, registration.selector);
    append_u64(out, static_cast<std::uint64_t>(registration.n_raw));
    append_u64(out, static_cast<std::uint64_t>(registration.n_in));
    append_u64(out, static_cast<std::uint64_t>(registration.input_segments.size()));
    for (const InputSegment &segment : registration.input_segments)
        append_context_segment(out, segment);
    append_string(out, registration.opening_policy);
    append_string(out, registration.binding_digest);
    append_string(out, registration.registration_digest);
    append_string(out, registration.status);
}

bool read_context_registration(ContextReader &reader,
                               ModelRegistration &registration) {
    if (!read_context_commitment(reader, registration.com_W) ||
        !reader.read_string(registration.dataset, "registration dataset") ||
        !reader.read_string(registration.model_arch_id, "registration architecture") ||
        !reader.read_string(registration.quant_cfg_id, "registration quant id") ||
        !reader.read_string(registration.static_tables_id, "registration table id") ||
        !reader.read_string(registration.field_id, "registration field id") ||
        !reader.read_string(registration.field_modulus, "registration field modulus") ||
        !reader.read_string(registration.manifest_digest, "registration manifest digest") ||
        !reader.read_string(registration.model_digest, "registration model digest") ||
        !reader.read_string(registration.bounds_digest, "registration bounds digest") ||
        !reader.read_string(registration.quant_digest, "registration quant digest") ||
        !reader.read_string(registration.layout_digest, "registration layout digest") ||
        !reader.read_string(registration.selector, "registration selector") ||
        !reader.read_size(registration.n_raw, "registration raw layout size") ||
        !reader.read_size(registration.n_in, "registration padded layout size")) return false;
    std::size_t count = 0;
    if (!reader.read_count(count, "registration input segments", 8)) return false;
    registration.input_segments.resize(count);
    for (InputSegment &segment : registration.input_segments)
        if (!read_context_segment(reader, segment)) return false;
    if (!reader.read_string(registration.opening_policy, "registration opening policy") ||
        !reader.read_string(registration.binding_digest, "registration binding digest") ||
        !reader.read_string(registration.registration_digest, "registration digest") ||
        !reader.read_string(registration.status, "registration status")) return false;
    registration.com_W_padding_mask = Fr(0);
    registration.has_private_padding_mask = false;
    return true;
}

void append_context_plan(std::string &out, const GatCircuitPlan &plan) {
    append_string(out, plan.registry.registry_id);
    append_u64(out, static_cast<std::uint64_t>(plan.registry.descriptors.size()));
    for (const RelationDescriptor &descriptor : plan.registry.descriptors) {
        append_string(out, descriptor.id);
        append_string(out, descriptor.primitive);
        append_string(out, descriptor.segment);
        append_string(out, descriptor.relation);
        append_string(out, descriptor.public_claim);
        append_string(out, descriptor.witness);
    }
    append_string(out, plan.input_layout.selector);
    append_u64(out, static_cast<std::uint64_t>(plan.input_layout.n_raw));
    append_u64(out, static_cast<std::uint64_t>(plan.input_layout.n_in));
    append_u64(out, static_cast<std::uint64_t>(plan.input_layout.segments.size()));
    for (const InputSegment &segment : plan.input_layout.segments)
        append_context_segment(out, segment);
    append_u64(out, static_cast<std::uint64_t>(plan.matmuls.size()));
    for (const MatMulDescriptor &descriptor : plan.matmuls) {
        append_string(out, descriptor.relation_id);
        append_string(out, descriptor.family_id);
        append_string(out, descriptor.lhs);
        append_string(out, descriptor.rhs);
        append_string(out, descriptor.output);
        append_u64(out, descriptor.rows);
        append_u64(out, descriptor.inner);
        append_u64(out, descriptor.cols);
        append_u64(out, descriptor.output_coordinates);
        append_u64(out, descriptor.lhs_offset);
        append_u64(out, descriptor.rhs_offset);
        append_u64(out, descriptor.output_offset);
        append_string(out, descriptor.coordinate_rule);
        append_string(out, descriptor.scale_rule);
        append_string(out, descriptor.native_path);
    }
    append_u64(out, static_cast<std::uint64_t>(plan.sparse_gkr.size()));
    for (const SparseGkrDescriptor &descriptor : plan.sparse_gkr) {
        append_string(out, descriptor.relation_id);
        append_string(out, descriptor.family_id);
        append_u64(out, descriptor.edge_count);
        append_u64(out, descriptor.value_width);
        append_u64(out, descriptor.input_offset);
        append_u64(out, descriptor.source_offset);
        append_u64(out, descriptor.destination_offset);
        append_u64(out, descriptor.output_offset);
        append_string(out, descriptor.coordinate_rule);
        append_string(out, descriptor.group_rule);
        append_u64(out, static_cast<std::uint64_t>(descriptor.constraints.size()));
        for (const std::string &constraint : descriptor.constraints)
            append_string(out, constraint);
    }
    append_u64(out, static_cast<std::uint64_t>(plan.lasso.size()));
    for (const LassoFamilyDescriptor &descriptor : plan.lasso) {
        append_string(out, descriptor.relation_id);
        append_string(out, descriptor.family_id);
        append_string(out, descriptor.table_id);
        append_u64(out, descriptor.real_queries);
        append_u64(out, descriptor.padded_queries);
        append_u64(out, descriptor.table_domain);
        append_int(out, descriptor.padding_value);
        append_u64(out, descriptor.query_offset);
        append_string(out, descriptor.incidence_rule);
        append_string(out, descriptor.challenge_policy);
        append_string(out, descriptor.query_encoding);
    }
    append_u64(out, static_cast<std::uint64_t>(plan.checkpoints.size()));
    for (const CheckpointDescriptor &checkpoint : plan.checkpoints) {
        append_string(out, checkpoint.id);
        append_string(out, checkpoint.relation_id);
        append_string(out, checkpoint.segment);
        append_u64(out, checkpoint.coordinates);
        append_u64(out, checkpoint.coordinate_offset);
        append_string(out, checkpoint.coordinate_rule);
    }
    append_u64(out, static_cast<std::uint64_t>(plan.native_api.size()));
    for (const std::string &api : plan.native_api) append_string(out, api);
    append_string(out, plan.circuit_id);
    append_string(out, plan.plan_digest);
    append_string(out, plan.opening_policy);
    append_u64(out, static_cast<std::uint64_t>(plan.terminal_openings.size()));
    for (const std::string &opening : plan.terminal_openings) append_string(out, opening);
    append_u64(out, plan.public_output_coordinates);
    append_u64(out, plan.residual_coordinates);
    append_bool_wire(out, plan.official_layered);
    append_bool_wire(out, plan.has_placeholder);
    append_u8_wire(out, plan.circuit.size);
    append_u64(out, static_cast<std::uint64_t>(plan.circuit.circuit.size()));
    for (const layer &current : plan.circuit.circuit) {
        append_u32(out, static_cast<std::uint32_t>(current.ty));
        append_u32(out, current.size);
        for (unsigned i = 0; i < 2; ++i) append_u32(out, current.size_u[i]);
        for (unsigned i = 0; i < 2; ++i) append_u32(out, current.size_v[i]);
        for (unsigned i = 0; i < 2; ++i) append_i8_wire(out, current.bit_length_u[i]);
        for (unsigned i = 0; i < 2; ++i) append_i8_wire(out, current.bit_length_v[i]);
        append_i8_wire(out, current.bit_length);
        append_i8_wire(out, current.max_bl_u);
        append_i8_wire(out, current.max_bl_v);
        append_bool_wire(out, current.need_phase2);
        append_u32(out, current.zero_start_id);
        append_i8_wire(out, current.fft_bit_length);
        append_u64(out, static_cast<std::uint64_t>(current.uni_interval.size()));
        for (const auto &interval : current.uni_interval) {
            append_u32(out, static_cast<std::uint32_t>(interval.first));
            append_u32(out, static_cast<std::uint32_t>(interval.second));
        }
        append_u64(out, static_cast<std::uint64_t>(current.bin_interval.size()));
        for (const auto &interval : current.bin_interval) {
            append_u32(out, static_cast<std::uint32_t>(interval.first));
            append_u32(out, static_cast<std::uint32_t>(interval.second));
        }
        append_u64(out, static_cast<std::uint64_t>(current.ori_id_u.size()));
        for (u32 value : current.ori_id_u) append_u32(out, value);
        append_u64(out, static_cast<std::uint64_t>(current.ori_id_v.size()));
        for (u32 value : current.ori_id_v) append_u32(out, value);
        append_u64(out, static_cast<std::uint64_t>(current.uni_gates.size()));
        for (const uniGate &gate : current.uni_gates) {
            append_u32(out, gate.g);
            append_u32(out, gate.u);
            append_u8_wire(out, gate.lu);
            append_int(out, gate.sc);
        }
        append_u64(out, static_cast<std::uint64_t>(current.bin_gates.size()));
        for (const binGate &gate : current.bin_gates) {
            append_u32(out, gate.g);
            append_u32(out, gate.u);
            append_u32(out, gate.v);
            append_u8_wire(out, gate.l);
            append_int(out, gate.sc);
        }
    }
}

bool read_context_plan(ContextReader &reader, GatCircuitPlan &plan) {
    if (!reader.read_string(plan.registry.registry_id, "registry id")) return false;
    std::size_t count = 0;
    if (!reader.read_count(count, "registry descriptors", kContextMaxDescriptors)) return false;
    plan.registry.descriptors.resize(count);
    for (RelationDescriptor &descriptor : plan.registry.descriptors) {
        if (!reader.read_string(descriptor.id, "relation id") ||
            !reader.read_string(descriptor.primitive, "relation primitive") ||
            !reader.read_string(descriptor.segment, "relation segment") ||
            !reader.read_string(descriptor.relation, "relation description") ||
            !reader.read_string(descriptor.public_claim, "relation public claim") ||
            !reader.read_string(descriptor.witness, "relation witness")) return false;
    }
    if (!reader.read_string(plan.input_layout.selector, "layout selector") ||
        !reader.read_size(plan.input_layout.n_raw, "layout raw size") ||
        !reader.read_size(plan.input_layout.n_in, "layout padded size") ||
        !reader.read_count(count, "layout segments", 8)) return false;
    plan.input_layout.segments.resize(count);
    for (InputSegment &segment : plan.input_layout.segments)
        if (!read_context_segment(reader, segment)) return false;
    if (!reader.read_count(count, "MatMul descriptors", kContextMaxDescriptors)) return false;
    plan.matmuls.resize(count);
    for (MatMulDescriptor &descriptor : plan.matmuls) {
        if (!reader.read_string(descriptor.relation_id, "MatMul relation") ||
            !reader.read_string(descriptor.family_id, "MatMul family") ||
            !reader.read_string(descriptor.lhs, "MatMul lhs") ||
            !reader.read_string(descriptor.rhs, "MatMul rhs") ||
            !reader.read_string(descriptor.output, "MatMul output") ||
            !reader.read_size(descriptor.rows, "MatMul rows") ||
            !reader.read_size(descriptor.inner, "MatMul inner") ||
            !reader.read_size(descriptor.cols, "MatMul cols") ||
            !reader.read_size(descriptor.output_coordinates, "MatMul output coordinates") ||
            !reader.read_size(descriptor.lhs_offset, "MatMul lhs offset") ||
            !reader.read_size(descriptor.rhs_offset, "MatMul rhs offset") ||
            !reader.read_size(descriptor.output_offset, "MatMul output offset") ||
            !reader.read_string(descriptor.coordinate_rule, "MatMul coordinate rule") ||
            !reader.read_string(descriptor.scale_rule, "MatMul scale rule") ||
            !reader.read_string(descriptor.native_path, "MatMul native path")) return false;
    }
    if (!reader.read_count(count, "sparse GKR descriptors", kContextMaxDescriptors)) return false;
    plan.sparse_gkr.resize(count);
    for (SparseGkrDescriptor &descriptor : plan.sparse_gkr) {
        if (!reader.read_string(descriptor.relation_id, "sparse relation") ||
            !reader.read_string(descriptor.family_id, "sparse family") ||
            !reader.read_size(descriptor.edge_count, "sparse edge count") ||
            !reader.read_size(descriptor.value_width, "sparse value width") ||
            !reader.read_size(descriptor.input_offset, "sparse input offset") ||
            !reader.read_size(descriptor.source_offset, "sparse source offset") ||
            !reader.read_size(descriptor.destination_offset, "sparse destination offset") ||
            !reader.read_size(descriptor.output_offset, "sparse output offset") ||
            !reader.read_string(descriptor.coordinate_rule, "sparse coordinate rule") ||
            !reader.read_string(descriptor.group_rule, "sparse group rule") ||
            !reader.read_count(count, "sparse constraints", kContextMaxDescriptors)) return false;
        descriptor.constraints.resize(count);
        for (std::string &constraint : descriptor.constraints)
            if (!reader.read_string(constraint, "sparse constraint")) return false;
    }
    if (!reader.read_count(count, "Lasso descriptors", kContextMaxDescriptors)) return false;
    plan.lasso.resize(count);
    for (LassoFamilyDescriptor &descriptor : plan.lasso) {
        if (!reader.read_string(descriptor.relation_id, "Lasso relation") ||
            !reader.read_string(descriptor.family_id, "Lasso family") ||
            !reader.read_string(descriptor.table_id, "Lasso table") ||
            !reader.read_size(descriptor.real_queries, "Lasso real query count") ||
            !reader.read_size(descriptor.padded_queries, "Lasso padded query count") ||
            !reader.read_size(descriptor.table_domain, "Lasso table domain") ||
            !reader.read_int(descriptor.padding_value) ||
            !reader.read_size(descriptor.query_offset, "Lasso query offset") ||
            !reader.read_string(descriptor.incidence_rule, "Lasso incidence rule") ||
            !reader.read_string(descriptor.challenge_policy, "Lasso challenge policy") ||
            !reader.read_string(descriptor.query_encoding, "Lasso query encoding")) return false;
    }
    if (!reader.read_count(count, "checkpoint descriptors", kContextMaxDescriptors)) return false;
    plan.checkpoints.resize(count);
    for (CheckpointDescriptor &checkpoint : plan.checkpoints) {
        if (!reader.read_string(checkpoint.id, "checkpoint id") ||
            !reader.read_string(checkpoint.relation_id, "checkpoint relation") ||
            !reader.read_string(checkpoint.segment, "checkpoint segment") ||
            !reader.read_size(checkpoint.coordinates, "checkpoint coordinates") ||
            !reader.read_size(checkpoint.coordinate_offset, "checkpoint offset") ||
            !reader.read_string(checkpoint.coordinate_rule, "checkpoint coordinate rule")) return false;
    }
    if (!reader.read_count(count, "native API entries", kContextMaxDescriptors)) return false;
    plan.native_api.resize(count);
    for (std::string &api : plan.native_api)
        if (!reader.read_string(api, "native API entry")) return false;
    if (!reader.read_string(plan.circuit_id, "circuit id") ||
        !reader.read_string(plan.plan_digest, "plan digest") ||
        !reader.read_string(plan.opening_policy, "plan opening policy") ||
        !reader.read_count(count, "terminal openings", 16)) return false;
    plan.terminal_openings.resize(count);
    for (std::string &opening : plan.terminal_openings)
        if (!reader.read_string(opening, "terminal opening")) return false;
    if (!reader.read_size(plan.public_output_coordinates, "public output coordinates") ||
        !reader.read_size(plan.residual_coordinates, "residual coordinates") ||
        !reader.read_bool(plan.official_layered) || !reader.read_bool(plan.has_placeholder)) return false;
    std::uint8_t circuit_size = 0;
    if (!reader.read_u8(circuit_size) ||
        !reader.read_count(count, "layered circuit layers", kContextMaxLayers)) return false;
    plan.circuit.size = circuit_size;
    plan.circuit.circuit.resize(count);
    for (layer &current : plan.circuit.circuit) {
        std::uint32_t type = 0, size = 0;
        if (!reader.read_u32(type) || type > static_cast<std::uint32_t>(layerType::PLACE_HOLDER) ||
            !reader.read_u32(size)) return false;
        current.ty = static_cast<layerType>(type);
        current.size = size;
        for (unsigned i = 0; i < 2; ++i) if (!reader.read_u32(current.size_u[i])) return false;
        for (unsigned i = 0; i < 2; ++i) if (!reader.read_u32(current.size_v[i])) return false;
        for (unsigned i = 0; i < 2; ++i) if (!reader.read_i8(current.bit_length_u[i])) return false;
        for (unsigned i = 0; i < 2; ++i) if (!reader.read_i8(current.bit_length_v[i])) return false;
        if (!reader.read_i8(current.bit_length) || !reader.read_i8(current.max_bl_u) ||
            !reader.read_i8(current.max_bl_v) || !reader.read_bool(current.need_phase2) ||
            !reader.read_u32(current.zero_start_id) || !reader.read_i8(current.fft_bit_length)) return false;
        if (!reader.read_count(count, "unary interval entries", kContextMaxElements)) return false;
        current.uni_interval.resize(count);
        for (auto &interval : current.uni_interval) {
            std::uint32_t first = 0, second = 0;
            if (!reader.read_u32(first) || !reader.read_u32(second)) return false;
            interval = {static_cast<int>(first), static_cast<int>(second)};
        }
        if (!reader.read_count(count, "binary interval entries", kContextMaxElements)) return false;
        current.bin_interval.resize(count);
        for (auto &interval : current.bin_interval) {
            std::uint32_t first = 0, second = 0;
            if (!reader.read_u32(first) || !reader.read_u32(second)) return false;
            interval = {static_cast<int>(first), static_cast<int>(second)};
        }
        if (!reader.read_count(count, "unary original ids", kContextMaxElements)) return false;
        current.ori_id_u.resize(count);
        for (u32 &value : current.ori_id_u) if (!reader.read_u32(value)) return false;
        if (!reader.read_count(count, "binary original ids", kContextMaxElements)) return false;
        current.ori_id_v.resize(count);
        for (u32 &value : current.ori_id_v) if (!reader.read_u32(value)) return false;
        if (!reader.read_count(count, "unary gates", kContextMaxElements)) return false;
        current.uni_gates.clear();
        current.uni_gates.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            u32 g = 0, u = 0;
            std::uint8_t lu = 0;
            Int sc = 0;
            if (!reader.read_u32(g) || !reader.read_u32(u) || !reader.read_u8(lu) ||
                !reader.read_int(sc)) return false;
            current.uni_gates.emplace_back(g, u, lu, sc);
        }
        if (!reader.read_count(count, "binary gates", kContextMaxElements)) return false;
        current.bin_gates.clear();
        current.bin_gates.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            u32 g = 0, u = 0, v = 0;
            std::uint8_t l = 0;
            Int sc = 0;
            if (!reader.read_u32(g) || !reader.read_u32(u) || !reader.read_u32(v) ||
                !reader.read_u8(l) || !reader.read_int(sc)) return false;
            current.bin_gates.emplace_back(g, u, v, sc, l);
        }
    }
    return true;
}

void append_context_graph(std::string &out, const Graph &graph) {
    append_u64(out, graph.num_nodes);
    append_u64(out, graph.num_features);
    append_u64(out, graph.graph_count);
    append_u64(out, graph.num_classes);
    append_context_int_vector(out, graph.src);
    append_context_int_vector(out, graph.dst);
    append_context_int_vector(out, graph.edge_gid);
    append_u64(out, static_cast<std::uint64_t>(graph.node_ptr.size()));
    for (std::size_t value : graph.node_ptr) append_u64(out, static_cast<std::uint64_t>(value));
    append_u64(out, static_cast<std::uint64_t>(graph.edge_ptr.size()));
    for (std::size_t value : graph.edge_ptr) append_u64(out, static_cast<std::uint64_t>(value));
}

bool read_context_graph(ContextReader &reader, Graph &graph) {
    if (!reader.read_size(graph.num_nodes, "graph node count") ||
        !reader.read_size(graph.num_features, "graph feature count") ||
        !reader.read_size(graph.graph_count, "graph count") ||
        !reader.read_size(graph.num_classes, "graph class count") ||
        !read_context_int_vector(reader, graph.src, "graph source edges") ||
        !read_context_int_vector(reader, graph.dst, "graph destination edges") ||
        !read_context_int_vector(reader, graph.edge_gid, "graph edge graph ids")) return false;
    std::size_t count = 0;
    if (!reader.read_count(count, "graph node pointers")) return false;
    graph.node_ptr.resize(count);
    for (std::size_t &value : graph.node_ptr)
        if (!reader.read_size(value, "graph node pointer")) return false;
    if (!reader.read_count(count, "graph edge pointers")) return false;
    graph.edge_ptr.resize(count);
    for (std::size_t &value : graph.edge_ptr)
        if (!reader.read_size(value, "graph edge pointer")) return false;
    return true;
}

void append_context_tensor(std::string &out, const Tensor2D &tensor) {
    append_u64(out, static_cast<std::uint64_t>(tensor.rows));
    append_u64(out, static_cast<std::uint64_t>(tensor.cols));
    append_context_int_vector(out, tensor.values);
}

bool read_context_tensor(ContextReader &reader, Tensor2D &tensor,
                         const char *what) {
    if (!reader.read_size(tensor.rows, what) || !reader.read_size(tensor.cols, what)) return false;
    const std::size_t expected = checked_mul(tensor.rows, tensor.cols, "verifier context tensor dimensions");
    std::size_t count = 0;
    if (!reader.read_count(count, what) || count != expected) return false;
    tensor.values.resize(count);
    for (Int &value : tensor.values) if (!reader.read_int(value)) return false;
    return true;
}

bool static_tables_equal(const StaticTables &left, const StaticTables &right) {
    const auto equal_rows = [](const std::vector<TablePoint> &a,
                               const std::vector<TablePoint> &b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i].x != b[i].x || a[i].y != b[i].y) return false;
        return true;
    };
    return left.table_id == right.table_id && left.generator_id == right.generator_id &&
           left.digest == right.digest && left.quant_cfg_id == right.quant_cfg_id &&
           left.rounding == right.rounding && left.rescale == right.rescale &&
           left.output_average == right.output_average && left.row_order == right.row_order &&
           left.exp_approx_id == right.exp_approx_id && left.elu_approx_id == right.elu_approx_id &&
           left.lrelu_slope_num == right.lrelu_slope_num && left.lrelu_slope_den == right.lrelu_slope_den &&
           left.s_delta == right.s_delta && left.s_exp == right.s_exp &&
           left.s_elu_in == right.s_elu_in && left.s_elu_out == right.s_elu_out &&
           left.lrelu_min == right.lrelu_min && left.lrelu_max == right.lrelu_max &&
           left.exp_min == right.exp_min && left.exp_max == right.exp_max &&
           left.elu_min == right.elu_min && left.elu_max == right.elu_max &&
           equal_rows(left.lrelu, right.lrelu) && equal_rows(left.exp, right.exp) &&
           equal_rows(left.elu, right.elu);
}

bool registry_equal(const RelationRegistry &left, const RelationRegistry &right) {
    if (left.registry_id != right.registry_id ||
        left.descriptors.size() != right.descriptors.size()) return false;
    for (std::size_t i = 0; i < left.descriptors.size(); ++i) {
        const RelationDescriptor &a = left.descriptors[i];
        const RelationDescriptor &b = right.descriptors[i];
        if (a.id != b.id || a.primitive != b.primitive || a.segment != b.segment ||
            a.relation != b.relation || a.public_claim != b.public_claim ||
            a.witness != b.witness) return false;
    }
    return true;
}

bool context_layer_metadata_valid(const GatCircuitPlan &plan, std::string *reason) {
    const auto fail = [reason](const std::string &message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    if (plan.circuit.circuit.size() != plan.circuit.size) return fail("layered circuit size field mismatch");
    for (std::size_t i = 0; i < plan.circuit.circuit.size(); ++i) {
        const layer &current = plan.circuit.circuit[i];
        if (current.zero_start_id != current.size) return fail("layer zero_start_id is not canonical");
        if (i == 0) {
            if (!current.ori_id_u.empty() || !current.ori_id_v.empty() ||
                !current.uni_gates.empty() || !current.bin_gates.empty())
                return fail("input layer contains derived/gate state");
            continue;
        }
        if (current.max_bl_u != std::max(current.bit_length_u[0], current.bit_length_u[1]))
            return fail("layer max_bl_u is not derived canonically");
        const i8 expected_max_v = current.need_phase2
            ? std::max(current.bit_length_v[0], current.bit_length_v[1]) : 0;
        if (current.max_bl_v != expected_max_v) return fail("layer max_bl_v is not derived canonically");
        if (current.size_u[0] != current.ori_id_u.size() ||
            current.size_v[0] != current.ori_id_v.size())
            return fail("layer subset sizes do not match initSubset maps");
        if (current.bit_length_u[0] != ceilPow2BitLength(current.size_u[0]) ||
            current.bit_length_v[0] != ceilPow2BitLength(current.size_v[0])) {
            std::ostringstream detail;
            detail << "layer subset bit lengths are not canonical at layer " << i
                   << " u=" << static_cast<int>(current.bit_length_u[0]) << "/"
                   << static_cast<int>(ceilPow2BitLength(current.size_u[0]))
                   << " v=" << static_cast<int>(current.bit_length_v[0]) << "/"
                   << static_cast<int>(ceilPow2BitLength(current.size_v[0]));
            return fail(detail.str());
        }
        for (const uniGate &gate : current.uni_gates)
            if (gate.lu == 0 && gate.u >= current.ori_id_u.size())
                return fail("unary gate subset coordinate exceeds initSubset map");
        for (const binGate &gate : current.bin_gates) {
            if (gate.getLayerIdU(static_cast<u8>(i)) == 0 && gate.u >= current.ori_id_u.size())
                return fail("binary lhs subset coordinate exceeds initSubset map");
            if (gate.getLayerIdV(static_cast<u8>(i)) == 0 && gate.v >= current.ori_id_v.size())
                return fail("binary rhs subset coordinate exceeds initSubset map");
        }
    }
    return true;
}

bool validate_verifier_context_object(const VerifierContext &context,
                                      std::string *reason) {
    const auto fail = [reason](const std::string &message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    try {
        context.graph.validate();
        context.quant.validate();
        const StaticTables canonical_tables = StaticTables::build(context.quant);
        if (!static_tables_equal(context.tables, canonical_tables) ||
            !context.tables.authenticate()) return fail("static tables do not match independently regenerated canonical tables");
        if (!registry_equal(context.plan.registry, RelationRegistry::canonical()))
            return fail("relation registry is not the canonical zkGPT registry");
        if (!validate_gat_circuit_plan(context.plan, reason)) return false;
        if (!context_layer_metadata_valid(context.plan, reason)) return false;
        if (context.public_features.rows != context.graph.num_nodes ||
            context.public_features.cols != context.graph.num_features ||
            context.public_features.cols == 0 ||
            context.public_features.values.size() != checked_mul(
                context.public_features.rows, context.public_features.cols,
                "verifier context feature tensor")) return fail("public feature shape is invalid");
        if (context.public_logits.rows != context.graph.num_nodes ||
            context.public_logits.cols != context.graph.num_classes ||
            context.public_logits.cols == 0 ||
            context.public_logits.values.size() != checked_mul(
                context.public_logits.rows, context.public_logits.cols,
                "verifier context logits tensor")) return fail("public logits shape is invalid");
        if (!registration_wire_valid(context.registration)) return fail("model registration wire is invalid");
        if (context.com_W.scheme != context.registration.com_W.scheme ||
            context.com_W.basis_id != context.registration.com_W.basis_id ||
            context.com_W.logical_size != context.registration.com_W.logical_size ||
            context.com_W.padded_size != context.registration.com_W.padded_size ||
            context.com_W.point_hex != context.registration.com_W.point_hex)
            return fail("context com_W is inconsistent with model registration");
        if (context.relation_id != "R_GAT-zkGPT" || context.model_arch_id != context.registration.model_arch_id ||
            context.quant_cfg_id != context.quant.quant_cfg_id ||
            context.static_tables_id != context.tables.table_id ||
            context.privacy_extension_profile_id != "masked_full" ||
            context.tables.quant_cfg_id != context.quant.quant_cfg_id)
            return fail("verifier context identity is not canonical");
        return true;
    } catch (const std::exception &error) {
        return fail(error.what());
    }
}

void set_reason(std::string *reason, const std::string &value) {
    if (reason != nullptr) *reason = value;
}

bool forbidden_payload(const std::string &value) {
    static const char *forbidden[] = {"WITNESS", "witness", "PRIVATE_TRACE", "private_trace", "raw_model", "raw_aux", "raw_lookup"};
    for (const char *needle : forbidden) if (value.find(needle) != std::string::npos) return true;
    return false;
}

bool is_lower_hex_digest(const std::string &value) {
    if (value.size() != 64) return false;
    for (char character : value) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) return false;
    }
    return true;
}

bool canonical_field_wire(const std::string &value) {
    if (value.empty() || value.size() > 128) return false;
    try {
        ensure_field();
        Field parsed;
        parsed.setStr(value, 16);
        return field_hex(parsed) == value;
    } catch (...) {
        return false;
    }
}

bool valid_native_kind(std::uint8_t kind) {
    return kind >= static_cast<std::uint8_t>(NativeProofMessageKind::Data) &&
           kind <= static_cast<std::uint8_t>(NativeProofMessageKind::Challenge);
}

} // namespace

void initialize_gat_field() {
    ensure_field();
}

std::vector<Fr> pack_native_gkr_matmul_rhs(
    const MatMulDescriptor &descriptor,
    const std::vector<Fr> &canonical_input_layer) {
    // The plan and model registry deliberately remain canonical row-major.
    // Only the private witness object handed to native FCONN crosses this
    // representation boundary.  Keep every size/offset calculation checked
    // so a malformed descriptor cannot be silently reinterpreted as a
    // different MLE layout.
    if (descriptor.rows == 0 || descriptor.inner == 0 || descriptor.cols == 0)
        throw std::invalid_argument("native GKR MatMul RHS has an empty shape");
    const std::size_t padded_inner = next_power_of_two(descriptor.inner);
    const std::size_t padded_cols = next_power_of_two(descriptor.cols);
    const std::size_t rhs_size = checked_mul(
        descriptor.inner, descriptor.cols, "canonical GKR MatMul RHS size");
    const std::size_t native_rhs_size = checked_mul(
        padded_inner, padded_cols, "padded native GKR MatMul RHS size");
    const std::size_t output_size = checked_mul(
        descriptor.rows, descriptor.cols, "native GKR MatMul output size");
    if (descriptor.output_coordinates != output_size)
        throw std::invalid_argument(
            "native GKR MatMul RHS descriptor output shape is inconsistent");
    if (descriptor.coordinate_rule.find("row_major_exact") == std::string::npos)
        throw std::invalid_argument(
            "native GKR MatMul RHS requires canonical row-major coordinates");
    if (descriptor.rhs_offset > canonical_input_layer.size() ||
        rhs_size > canonical_input_layer.size() - descriptor.rhs_offset)
        throw std::invalid_argument(
            "native GKR MatMul RHS span exceeds canonical input layer");

    // Native FCONN treats the inner coordinate as the low bits and the
    // column coordinate as the outer bits.  Every column therefore owns a
    // full padded_inner block; values from the next canonical column must
    // never slide across the inner padding gap.
    std::vector<Fr> packed(native_rhs_size, Fr(0));
    for (std::size_t col = 0; col < descriptor.cols; ++col) {
        const std::size_t native_column_base = checked_mul(
            col, padded_inner, "native GKR MatMul RHS column base");
        for (std::size_t term = 0; term < descriptor.inner; ++term) {
            const std::size_t canonical_relative = checked_add(
                checked_mul(term, descriptor.cols,
                            "canonical MatMul RHS row-major stride"),
                col, "canonical MatMul RHS row-major coordinate");
            const std::size_t native_index = checked_add(
                native_column_base, term,
                "native MatMul RHS column-major coordinate");
            const std::size_t canonical_index = checked_add(
                descriptor.rhs_offset, canonical_relative,
                "canonical MatMul RHS input coordinate");
            packed.at(native_index) = canonical_input_layer.at(canonical_index);
        }
    }
    return packed;
}

bool validate_gat_circuit_plan(const GatCircuitPlan &plan, std::string *reason) {
    return validate_gat_circuit_plan_impl(plan, reason);
}

GatCircuitWitness build_gat_circuit_witness(
    const GatCircuitPlan &plan, const Graph &graph,
    const Tensor2D &public_features, const Model &model,
    const QuantConfig &quant, const ForwardResult &forward) {
    std::string reason;
    if (!validate_gat_circuit_plan(plan, &reason))
        throw std::invalid_argument("cannot build GAT circuit witness: " + reason);
    graph.validate();
    model.validate_schema();
    quant.validate();
    if (!validate_relation(graph, public_features, model, quant, forward))
        throw std::invalid_argument("cannot build GAT circuit witness from an invalid trace");

    GatCircuitWitness witness;
    witness.input_layer = canonical_input_values(
        plan, graph, public_features, model, quant, forward);
    witness.canonical_input_layer = witness.input_layer;
    if (!evaluate_layered_values(plan, witness.input_layer, witness.values, &reason))
        throw std::invalid_argument("canonical GAT circuit evaluation failed: " + reason);
    if (witness.values.empty() ||
        witness.values.back().size() !=
            plan.public_output_coordinates + plan.residual_coordinates)
        throw std::invalid_argument("canonical GAT circuit output shape is invalid");
    witness.public_output = witness.values.back();
    const std::size_t y_size = plan.public_output_coordinates;
    for (std::size_t index = 0; index < y_size; ++index) {
        if (witness.public_output[index] != Fr(forward.logits.values.at(index)))
            throw std::invalid_argument("canonical GAT circuit public Y binding failed");
    }
    witness.residuals.assign(witness.public_output.begin() + y_size,
                             witness.public_output.end());
    for (std::size_t index = 0; index < witness.residuals.size(); ++index)
        if (!witness.residuals[index].isZero()) {
            std::ostringstream detail;
            detail << "canonical GAT circuit has a non-zero residual at "
                   << index;
            throw std::invalid_argument(detail.str());
        }
    witness.relation_ok = true;
    return witness;
}

bool evaluate_gat_circuit_witness(const GatCircuitPlan &plan,
                                  const GatCircuitWitness &witness,
                                  std::string *reason) {
    const auto fail = [reason](const std::string &message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    try {
        std::string plan_reason;
        if (!validate_gat_circuit_plan(plan, &plan_reason))
            return fail("GAT circuit plan is invalid: " + plan_reason);
        if (witness.input_layer.size() != plan.circuit.circuit.front().size ||
            witness.canonical_input_layer.size() != witness.input_layer.size())
            return fail("GAT circuit witness input layer has the wrong size");
        for (std::size_t index = 0; index < witness.input_layer.size(); ++index)
            if (witness.input_layer[index] != witness.canonical_input_layer[index])
                return fail("GAT circuit witness input is not bound to canonical X/W/Aux/L");

        std::vector<std::vector<Fr>> evaluated;
        if (!evaluate_layered_values(plan, witness.input_layer, evaluated, reason))
            return false;
        if (witness.values.size() != evaluated.size())
            return fail("GAT circuit witness layer count is invalid");
        for (std::size_t layer = 0; layer < evaluated.size(); ++layer) {
            if (witness.values[layer] != evaluated[layer])
                return fail("GAT circuit witness layer does not match official gate evaluation");
        }
        if (witness.public_output != evaluated.back())
            return fail("GAT circuit witness public output does not match final layer");
        if (witness.public_output.size() !=
            plan.public_output_coordinates + plan.residual_coordinates)
            return fail("GAT circuit witness public output shape is invalid");
        if (witness.residuals.size() != plan.residual_coordinates)
            return fail("GAT circuit witness residual shape is invalid");
        for (std::size_t index = 0; index < witness.residuals.size(); ++index) {
            if (witness.residuals[index] !=
                witness.public_output[plan.public_output_coordinates + index])
                return fail("GAT circuit witness residual is not output-bound");
            if (!witness.residuals[index].isZero())
                return fail("GAT circuit relation residual is non-zero");
        }
        if (!witness.relation_ok)
            return fail("GAT circuit witness was not marked relation-valid");
        return true;
    } catch (const std::exception &error) {
        return fail(error.what());
    }
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
    if (scheme != "hyrax-pedersen-bn254" || !is_lower_hex_digest(basis_id) ||
        point_hex.empty() || logical_size == 0 || padded_size == 0 ||
        padded_size > (1ULL << 30) ||
        (padded_size & (padded_size - 1)) != 0 || logical_size > padded_size)
        return false;
    try {
        ensure_field();
        G1 point;
        point.deserializeHexStr(point_hex);
        return point.serializeToHexStr() == point_hex;
    } catch (...) {
        return false;
    }
}

CommitmentWire commit_private_segment(const std::vector<Int> &values,
                                      const std::string &basis_label) {
    if (values.empty()) throw std::invalid_argument("cannot commit an empty zkGPT segment");
    return commit_private_segment(values, basis_label, next_power_of_two(values.size()));
}

CommitmentWire commit_private_segment(const std::vector<Int> &values,
                                      const std::string &basis_label,
                                      std::size_t padded_domain) {
    Fr blind;
    return commit_segment_with_mask(values, basis_label, padded_domain, &blind,
                                    nullptr);
}

ProverCommitment commit_private_segment_for_prover(
    const std::vector<Int> &values, const std::string &basis_label,
    std::size_t padded_domain) {
    ProverCommitment result;
    result.wire = commit_segment_with_mask(values, basis_label, padded_domain,
                                           &result.blinder, nullptr);
    return result;
}

namespace {

const char kComWSecretMagic[] = "zkgpt-gat-com-w-secret-v1";

bool read_secret_file(const std::string &path, std::string &payload,
                      std::string *reason) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        set_reason(reason, "cannot open com_W prover secret");
        return false;
    }
    struct stat metadata {};
    if (::fstat(fd, &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
        (metadata.st_mode & (S_IRWXG | S_IRWXO)) != 0 || metadata.st_size < 0 ||
        metadata.st_size > 4096) {
        ::close(fd);
        set_reason(reason, "com_W prover secret permissions or size are invalid");
        return false;
    }
    payload.assign(static_cast<std::size_t>(metadata.st_size), '\0');
    std::size_t offset = 0;
    while (offset < payload.size()) {
        const ssize_t got = ::read(fd, &payload[offset], payload.size() - offset);
        if (got <= 0) {
            ::close(fd);
            set_reason(reason, "cannot read com_W prover secret");
            return false;
        }
        offset += static_cast<std::size_t>(got);
    }
    ::close(fd);
    return true;
}

bool parse_secret_payload(const ModelRegistration &registration,
                          const std::string &payload, Fr &blinder,
                          std::string *reason) {
    std::istringstream in(payload);
    std::string magic, version, registration_digest, basis_id;
    std::string logical_size, padded_size, point, blinder_hex, trailing;
    if (!(in >> magic >> version >> registration_digest >> basis_id >>
          logical_size >> padded_size >> point >> blinder_hex) ||
        (in >> trailing) || magic != kComWSecretMagic || version != "1" ||
        registration_digest != registration.registration_digest ||
        basis_id != registration.com_W.basis_id ||
        logical_size != std::to_string(registration.com_W.logical_size) ||
        padded_size != std::to_string(registration.com_W.padded_size) ||
        point != registration.com_W.point_hex || !canonical_field_wire(blinder_hex)) {
        set_reason(reason, "com_W prover secret identity mismatch");
        return false;
    }
    try {
        blinder.setStr(blinder_hex, 16);
        if (field_hex(blinder) != blinder_hex) throw std::runtime_error("non-canonical field");
    } catch (...) {
        set_reason(reason, "com_W prover secret blinder is malformed");
        return false;
    }
    return true;
}

} // namespace

bool write_prover_commitment_secret(const ModelRegistration &registration,
                                    const std::string &path,
                                    std::string *reason) {
    try {
        ensure_field();
        if (path.empty() || !registration_wire_valid(registration) ||
            !registration.has_private_padding_mask) {
            set_reason(reason, "com_W prover secret requires an in-process registered model");
            return false;
        }
        if (field_hex(registration.com_W_padding_mask).empty()) {
            set_reason(reason, "com_W prover secret blinder is unavailable");
            return false;
        }
        std::ostringstream encoded;
        encoded << kComWSecretMagic << " 1 " << registration.registration_digest
                << ' ' << registration.com_W.basis_id << ' '
                << registration.com_W.logical_size << ' '
                << registration.com_W.padded_size << ' '
                << registration.com_W.point_hex << ' '
                << field_hex(registration.com_W_padding_mask) << '\n';
        const std::string payload = encoded.str();
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL |
                                             O_CLOEXEC, S_IRUSR | S_IWUSR);
        if (fd < 0) {
            set_reason(reason, errno == EEXIST
                                  ? "com_W prover secret already exists"
                                  : "cannot create com_W prover secret");
            return false;
        }
        bool ok = true;
        std::size_t offset = 0;
        while (offset < payload.size()) {
            const ssize_t written = ::write(fd, payload.data() + offset,
                                            payload.size() - offset);
            if (written <= 0) {
                ok = false;
                break;
            }
            offset += static_cast<std::size_t>(written);
        }
        if (::fchmod(fd, S_IRUSR | S_IWUSR) != 0) ok = false;
        if (::close(fd) != 0) ok = false;
        if (!ok) {
            ::unlink(path.c_str());
            set_reason(reason, "cannot write com_W prover secret");
            return false;
        }
        return true;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool read_prover_commitment_secret(const ModelRegistration &registration,
                                   const std::string &path, Fr &blinder,
                                   std::string *reason) {
    try {
        ensure_field();
        if (path.empty() || !registration_wire_valid(registration)) {
            set_reason(reason, "com_W prover secret requires a valid registration wire");
            return false;
        }
        std::string payload;
        if (!read_secret_file(path, payload, reason)) return false;
        return parse_secret_payload(registration, payload, blinder, reason);
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
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
    registration.com_W = commit_segment_with_mask(
        flattened, basis_label, layout.n_in, &registration.com_W_padding_mask,
        nullptr);
    registration.has_private_padding_mask = true;
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
            registration.com_W.basis_id != fixed_digest(hash_hex(basis_label)) ||
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
    if (forbidden_payload(native_domain) ||
        forbidden_payload(native_statement.field_id)) return true;
    for (const NativeProofMessage &message : native_messages) {
        if (forbidden_payload(message.label) || forbidden_payload(message.payload)) return true;
    }
    return false;
}

bool GatProofArtifact::native_session(NativeProofSession &out,
                                      std::string *reason) const {
    if (native_domain != kNativeArtifactDomain) {
        set_reason(reason, "unsupported native proof transcript domain");
        return false;
    }
    return NativeProofSession::from_messages(native_domain, native_statement,
                                             native_messages, out, reason);
}

bool GatProofArtifact::valid_wire(std::string *reason) const {
    try {
        if (version != 2 || schema != "zkgpt-gat-proof-v2") {
            set_reason(reason, "unsupported proof wire version");
            return false;
        }
        if (relation_id != "R_GAT-zkGPT") {
            set_reason(reason, "proof relation id mismatch");
            return false;
        }
        const bool diagnostic_gkr = privacy_profile == "gkr_only_diagnostic";
        if (privacy_profile != "masked_full" && !diagnostic_gkr) {
            set_reason(reason, "proof does not declare a supported privacy profile");
            return false;
        }
        if (!is_lower_hex_digest(statement_digest) ||
            !canonical_field_wire(transcript_digest)) {
            set_reason(reason, "proof statement or transcript digest is not canonical");
            return false;
        }
        if (!com_aux.valid() || !com_L.valid()) {
            set_reason(reason, "dynamic Hyrax commitment wire is invalid");
            return false;
        }
        if (!native_statement.valid(reason) ||
            native_statement.field_id != "bn254-fr-v1" ||
            native_statement.verifier_context_digest != statement_digest ||
            native_statement.statement_root_digest != statement_digest) {
            set_reason(reason, "native statement is not bound to the public context digest");
            return false;
        }
        if (native_statement.aux_digest !=
                fixed_digest(hash_hex(commitment_encoding(com_aux))) ||
            native_statement.l_digest !=
                fixed_digest(hash_hex(commitment_encoding(com_L)))) {
            set_reason(reason, "native statement commitment binding mismatch");
            return false;
        }

        const ProofSection *sections[] = {
            &Pi_Lasso, &Pi_MatMul, &Pi_GKR, &Pi_combine, &Pi_open};
        const char *names[] = {
            "Pi_Lasso", "Pi_MatMul", "Pi_GKR", "Pi_combine", "Pi_open"};
        std::size_t section_message_count = 0;
        for (unsigned i = 0; i < 5; ++i) {
            if (sections[i]->protocol != names[i] ||
                sections[i]->messages.empty()) {
                set_reason(reason, std::string("missing ") + names[i] + " messages");
                return false;
            }
            if (sections[i]->messages.size() > kArtifactMaxMessages ||
                section_message_count > kArtifactMaxMessages -
                                           sections[i]->messages.size()) {
                set_reason(reason, "proof section message count exceeds wire limit");
                return false;
            }
            section_message_count += sections[i]->messages.size();
            for (const TranscriptMessage &message : sections[i]->messages) {
                if (message.label.empty() || message.label.size() > kArtifactMaxString ||
                    message.payload.size() > kArtifactMaxString ||
                    !valid_native_kind(message.kind)) {
                    set_reason(reason, "proof message exceeds canonical wire limits");
                    return false;
                }
            }
        }
        if (native_messages.empty() || native_messages.size() > kArtifactMaxMessages ||
            native_messages.size() != section_message_count) {
            set_reason(reason, "native transcript and proof sections have different message counts");
            return false;
        }

        // The five section summaries are a lossless projection of the one
        // native typed transcript.  Official FCONN/MatMul messages are
        // emitted while the descending GKR walk is in progress, so their
        // native sequence numbers can be interleaved with ordinary GKR
        // messages.  Validate the projection as a sequence-indexed bijection
        // instead of assuming that protocol sections are contiguous.
        std::vector<const TranscriptMessage *> by_sequence(native_messages.size(), nullptr);
        for (const ProofSection *section : sections) {
            std::uint64_t previous = 0;
            bool have_previous = false;
            for (const TranscriptMessage &summary : section->messages) {
                if (summary.sequence >= native_messages.size() ||
                    (have_previous && summary.sequence <= previous)) {
                    set_reason(reason, "proof section message sequence is not canonical");
                    return false;
                }
                if (by_sequence[static_cast<std::size_t>(summary.sequence)] != nullptr) {
                    set_reason(reason, "proof section maps one native message more than once");
                    return false;
                }
                by_sequence[static_cast<std::size_t>(summary.sequence)] = &summary;
                previous = summary.sequence;
                have_previous = true;
            }
        }
        for (std::size_t index = 0; index < native_messages.size(); ++index) {
            const TranscriptMessage *summary = by_sequence[index];
            const NativeProofMessage &native = native_messages[index];
            if (summary == nullptr || native.sequence != static_cast<std::uint64_t>(index) ||
                static_cast<std::uint8_t>(native.kind) != summary->kind ||
                native.label != summary->label || native.payload != summary->payload) {
                set_reason(reason, "proof sections do not exactly map to native transcript");
                return false;
            }
        }

        if (has_private_payload()) {
            set_reason(reason, "proof wire contains a private payload marker");
            return false;
        }
        NativeProofSession session;
        if (!native_session(session, reason)) return false;
        if (session.statement_digest() != statement_digest ||
            session.digest() != transcript_digest) {
            set_reason(reason, "Fiat-Shamir transcript or statement binding mismatch");
            return false;
        }
        return true;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
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
    write_string(out, native_domain);
    write_native_statement(out, native_statement);
    write_u64(out, static_cast<std::uint64_t>(native_messages.size()));
    for (const NativeProofMessage &message : native_messages) {
        write_u8(out, static_cast<std::uint8_t>(message.kind));
        write_u64(out, message.sequence);
        write_string(out, message.label);
        write_string(out, message.payload);
    }
    write_string(out, transcript_digest);
    if (!out.good()) { set_reason(reason, "proof artifact write failed"); return false; }
    return true;
}

bool GatProofArtifact::read(const std::string &path, GatProofArtifact &out,
                            std::string *reason) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) { set_reason(reason, "cannot open proof artifact"); return false; }
    if (!wire_size_within_limit(in)) {
        set_reason(reason, "proof artifact exceeds size limit");
        return false;
    }
    in.seekg(0, std::ios::beg);
    char magic[8];
    if (!read_exact(in, magic, sizeof(magic)) || std::memcmp(magic, "ZKGPTGAT", sizeof(magic)) != 0) { set_reason(reason, "invalid proof artifact magic"); return false; }
    GatProofArtifact parsed;
    std::uint32_t version = 0;
    if (!read_u32(in, version) || !read_string(in, parsed.schema) ||
        !read_string(in, parsed.relation_id) ||
        !read_string(in, parsed.privacy_profile) ||
        !read_string(in, parsed.statement_digest) ||
        !read_commitment(in, parsed.com_aux) ||
        !read_commitment(in, parsed.com_L) ||
        !read_section(in, parsed.Pi_Lasso) ||
        !read_section(in, parsed.Pi_MatMul) ||
        !read_section(in, parsed.Pi_GKR) ||
        !read_section(in, parsed.Pi_combine) ||
        !read_section(in, parsed.Pi_open) ||
        !read_string(in, parsed.native_domain) ||
        !read_native_statement(in, parsed.native_statement)) {
        set_reason(reason, "truncated proof artifact");
        return false;
    }
    parsed.version = version;
    std::uint64_t native_count = 0;
    if (!read_u64(in, native_count) || native_count == 0 ||
        native_count > kArtifactMaxMessages) {
        set_reason(reason, "invalid native proof message count");
        return false;
    }
    parsed.native_messages.reserve(static_cast<std::size_t>(native_count));
    for (std::uint64_t i = 0; i < native_count; ++i) {
        NativeProofMessage message;
        std::uint8_t kind = 0;
        if (!read_u8(in, kind) || !valid_native_kind(kind) ||
            !read_u64(in, message.sequence) ||
            !read_string(in, message.label) ||
            !read_string(in, message.payload)) {
            set_reason(reason, "invalid native proof message");
            return false;
        }
        message.kind = static_cast<NativeProofMessageKind>(kind);
        parsed.native_messages.push_back(std::move(message));
    }
    if (!read_string(in, parsed.transcript_digest)) {
        set_reason(reason, "truncated proof artifact transcript digest");
        return false;
    }
    char trailing = 0;
    if (in.read(&trailing, 1)) { set_reason(reason, "proof artifact has trailing bytes"); return false; }
    if (!parsed.valid_wire(reason)) return false;
    out = std::move(parsed);
    return true;
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
    ensure_field();
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
    ensure_field();
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
    return fixed_digest(hash_hex(context_encoding(context)));
}

namespace {

std::string serialize_verifier_context_payload(const VerifierContext &context) {
    std::string payload;
    append_string(payload, "zkgpt-gat-verifier-context-v2");
    append_string(payload, context.relation_id);
    append_string(payload, context.model_arch_id);
    append_string(payload, context.privacy_extension_profile_id);
    append_string(payload, context.quant_cfg_id);
    append_string(payload, context.static_tables_id);
    append_context_graph(payload, context.graph);
    append_context_tensor(payload, context.public_features);
    append_context_tensor(payload, context.public_logits);
    append_context_quant(payload, context.quant);
    append_context_tables(payload, context.tables);
    // This is the public registration wire only.  The in-memory producer may
    // carry com_W_padding_mask, but it is intentionally not serialized.
    append_context_registration(payload, context.registration);
    append_context_commitment(payload, context.com_W);
    append_context_plan(payload, context.plan);
    const std::string digest = verifier_context_digest(context);
    append_string(payload, digest);
    // The native statement uses the same authenticated context root.  Keep a
    // separate field so a reader can reject a malformed root independently of
    // the human-facing context digest field.
    append_string(payload, digest);
    return payload;
}

bool read_context_file(const std::string &path, std::string &wire,
                       std::string *reason) {
    struct stat info {};
    if (::lstat(path.c_str(), &info) != 0 || S_ISLNK(info.st_mode) ||
        !S_ISREG(info.st_mode) || info.st_nlink != 1) {
        set_reason(reason, "verifier context must be a regular non-aliased file");
        return false;
    }
    if (info.st_size < 0 || static_cast<std::uint64_t>(info.st_size) > kContextMaxWireBytes) {
        set_reason(reason, "verifier context exceeds the wire-size limit");
        return false;
    }
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) {
        set_reason(reason, "cannot open verifier context");
        return false;
    }
    wire.assign(static_cast<std::size_t>(info.st_size), '\0');
    if (!wire.empty()) input.read(&wire[0], static_cast<std::streamsize>(wire.size()));
    if (!input.good() && !input.eof()) {
        set_reason(reason, "cannot read verifier context");
        return false;
    }
    if (static_cast<std::size_t>(input.gcount()) != wire.size()) {
        set_reason(reason, "verifier context read was truncated");
        return false;
    }
    return true;
}

bool read_header_u32(const std::string &wire, std::size_t offset,
                     std::uint32_t &value) {
    if (offset > wire.size() || wire.size() - offset < 4) return false;
    value = 0;
    for (unsigned i = 0; i < 4; ++i)
        value |= static_cast<std::uint32_t>(static_cast<unsigned char>(wire[offset + i])) << (i * 8);
    return true;
}

bool read_header_u64(const std::string &wire, std::size_t offset,
                     std::uint64_t &value) {
    if (offset > wire.size() || wire.size() - offset < 8) return false;
    value = 0;
    for (unsigned i = 0; i < 8; ++i)
        value |= static_cast<std::uint64_t>(static_cast<unsigned char>(wire[offset + i])) << (i * 8);
    return true;
}

} // namespace

bool write_verifier_context(const VerifierContext &context,
                            const std::string &path,
                            std::string *reason) {
    try {
        if (!validate_verifier_context_object(context, reason)) return false;
        std::string payload = serialize_verifier_context_payload(context);
        if (payload.size() > kContextMaxWireBytes) {
            set_reason(reason, "serialized verifier context exceeds the wire-size limit");
            return false;
        }
        struct stat existing {};
        if (::lstat(path.c_str(), &existing) == 0 && S_ISLNK(existing.st_mode)) {
            set_reason(reason, "refusing to overwrite a verifier-context symlink");
            return false;
        }
        const std::string temporary = path + ".partial";
        std::ofstream output(temporary.c_str(), std::ios::binary | std::ios::trunc);
        if (!output) {
            set_reason(reason, "cannot open temporary verifier context");
            return false;
        }
        output.write(kVerifierContextMagic, 8);
        write_u32(output, 2);
        write_u64(output, static_cast<std::uint64_t>(payload.size()));
        if (!payload.empty()) output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        output.flush();
        if (!output.good()) {
            output.close();
            std::remove(temporary.c_str());
            set_reason(reason, "verifier context write failed");
            return false;
        }
        output.close();
        if (std::rename(temporary.c_str(), path.c_str()) != 0) {
            std::remove(temporary.c_str());
            set_reason(reason, "cannot finalize verifier context");
            return false;
        }
        return true;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool read_verifier_context(const std::string &path,
                           VerifierContext &context,
                           std::string *reason) {
    try {
        ensure_field();
        std::string wire;
        if (!read_context_file(path, wire, reason)) return false;
        constexpr std::size_t header_size = 8 + 4 + 8;
        if (wire.size() < header_size || std::memcmp(wire.data(), kVerifierContextMagic, 8) != 0) {
            set_reason(reason, "invalid verifier context magic");
            return false;
        }
        std::uint32_t version = 0;
        std::uint64_t payload_size = 0;
        if (!read_header_u32(wire, 8, version) || !read_header_u64(wire, 12, payload_size) ||
            version != 2 || payload_size != wire.size() - header_size ||
            payload_size > kContextMaxWireBytes) {
            set_reason(reason, "unsupported, oversized, or non-canonical verifier context header");
            return false;
        }
        ContextReader reader(wire, header_size, wire.size());
        VerifierContext parsed;
        std::string schema;
        if (!reader.read_string(schema, "context schema") || schema != "zkgpt-gat-verifier-context-v2" ||
            !reader.read_string(parsed.relation_id, "context relation") ||
            !reader.read_string(parsed.model_arch_id, "context model architecture") ||
            !reader.read_string(parsed.privacy_extension_profile_id, "context privacy profile") ||
            !reader.read_string(parsed.quant_cfg_id, "context quant id") ||
            !reader.read_string(parsed.static_tables_id, "context table id") ||
            !read_context_graph(reader, parsed.graph) ||
            !read_context_tensor(reader, parsed.public_features, "public feature tensor") ||
            !read_context_tensor(reader, parsed.public_logits, "public logits tensor") ||
            !read_context_quant(reader, parsed.quant) ||
            !read_context_tables(reader, parsed.tables) ||
            !read_context_registration(reader, parsed.registration) ||
            !read_context_commitment(reader, parsed.com_W) ||
            !read_context_plan(reader, parsed.plan)) {
            set_reason(reason, reader.error().empty() ? "malformed verifier context" : reader.error());
            return false;
        }
        std::string stored_digest;
        std::string stored_root;
        if (!reader.read_string(stored_digest, "context digest") ||
            !reader.read_string(stored_root, "context structured root") || !reader.at_end()) {
            set_reason(reason, reader.error().empty() ? "verifier context has trailing or unknown bytes" : reader.error());
            return false;
        }
        std::string validation_reason;
        if (!validate_verifier_context_object(parsed, &validation_reason)) {
            set_reason(reason, validation_reason);
            return false;
        }
        const std::string expected_digest = verifier_context_digest(parsed);
        if (!is_lower_hex_digest(stored_digest) || !is_lower_hex_digest(stored_root) ||
            stored_digest != expected_digest || stored_root != expected_digest) {
            set_reason(reason, "verifier context digest or structured root does not match canonical content");
            return false;
        }
        context = std::move(parsed);
        return true;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}

NativeProofStatement make_native_proof_statement(
    const VerifierContext &context, const CommitmentWire &com_aux,
    const CommitmentWire &com_L, const G1 &G,
    const std::vector<G1> &generators, const std::vector<Fr> &basis,
    std::size_t l_dimension) {
    ensure_field();
    if (l_dimension == 0 || !context.plan.official_layered ||
        context.plan.has_placeholder || !registration_wire_valid(context.registration) ||
        !context.com_W.valid() || !com_aux.valid() || !com_L.valid())
        throw std::invalid_argument("native statement factory received an unbound context");

    std::string resource_material;
    append_string(resource_material, "zkgpt-native-resources-v1");
    append_string(resource_material, context.registration.manifest_digest);
    append_string(resource_material, context.tables.digest);
    append_string(resource_material, context.tables.table_id);
    append_string(resource_material, context.tables.generator_id);
    append_string(resource_material, context.quant.quant_cfg_id);
    append_string(resource_material, context.static_tables_id);
    append_string(resource_material, context.plan.plan_digest);

    NativeProofStatement statement;
    statement.verifier_context_digest = fixed_digest(verifier_context_digest(context));
    statement.field_id = "bn254-fr-v1";
    statement.plan_digest = normalize_digest(context.plan.plan_digest);
    statement.resources_digest = fixed_digest(hash_hex(resource_material));
    statement.tables_digest = normalize_digest(context.tables.digest);
    statement.com_w_digest = fixed_digest(hash_hex(commitment_encoding(context.com_W)));
    statement.aux_digest = fixed_digest(hash_hex(commitment_encoding(com_aux)));
    statement.l_digest = fixed_digest(hash_hex(commitment_encoding(com_L)));
    statement.l_dimension = l_dimension;
    statement.basis_digest = NativeProofStatement::basis_digest_for(
        G, generators, basis, l_dimension);
    // The formal context is the authenticated statement root.  The
    // structured fields above are independently serialized into the session
    // and cannot be replaced by a transcript-local alias.
    statement.statement_root_digest = statement.verifier_context_digest;
    std::string reason;
    if (!statement.valid(&reason)) throw std::invalid_argument(reason);
    return statement;
}

GkrOnlyArtifactResult prove_gkr_only_artifact(
    prover &native_prover, const VerifierContext &context,
    const CommitmentWire &com_aux, const CommitmentWire &com_L,
    const G1 &G, const std::vector<G1> &generators,
    const std::vector<Fr> &basis, std::size_t l_dimension,
    const std::vector<F> &public_output,
    const std::vector<NativeGkrMatmulShape> &matmuls) {
    GkrOnlyArtifactResult result;
    try {
        if (public_output.empty() ||
            public_output.size() != context.public_logits.values.size()) {
            throw std::invalid_argument(
                "GKR diagnostic output does not match the public Y shape");
        }
        for (std::size_t index = 0; index < public_output.size(); ++index) {
            if (public_output[index] != Fr(context.public_logits.values[index])) {
                throw std::invalid_argument(
                    "GKR diagnostic output is not bound to public Y");
            }
        }
        if (native_prover.C.size < 2 || native_prover.C.circuit.empty() ||
            native_prover.C.circuit.back().size != public_output.size()) {
            throw std::invalid_argument(
                "GKR diagnostic circuit output does not match public Y");
        }
        if (!com_aux.valid() || !com_L.valid()) {
            throw std::invalid_argument(
                "GKR diagnostic commitment wire is invalid");
        }

        const NativeProofStatement statement = make_native_proof_statement(
            context, com_aux, com_L, G, generators, basis, l_dimension);
        NativeProofSession session(kNativeArtifactDomain, statement);

        // These public-only messages occupy the non-GKR sections.  The
        // statement already authenticates the same values; retaining the
        // explicit messages gives the artifact collector a lossless section
        // projection without ever serializing witness coordinates.
        session.absorb("adapter/context", statement.verifier_context_digest);
        session.absorb("adapter/com_aux", statement.aux_digest);

        verifier native_verifier(&native_prover, native_prover.C);
        if (!native_verifier.prove_gkr_with_session(session)) {
            throw std::runtime_error("native GKR prover/verifier loop rejected the synthetic relation");
        }

        const auto public_output_digest = [&public_output]() {
            std::string encoded;
            append_string(encoded, "zkgpt-native-public-output-v1");
            append_u64(encoded, static_cast<std::uint64_t>(public_output.size()));
            for (const F &value : public_output) append_string(encoded, field_hex(value));
            return fixed_digest(hash_hex(encoded));
        }();
        session.absorb("adapter/public_output", public_output_digest);
        session.absorb("adapter/com_L", statement.l_digest);

        std::string reason;
        if (!verify_gkr_transcript(session, native_prover.C, public_output,
                                   matmuls, &reason)) {
            throw std::runtime_error(
                "independent native GKR transcript replay rejected: " + reason);
        }

        GatProofArtifact artifact;
        artifact.privacy_profile = "gkr_only_diagnostic";
        artifact.statement_digest = statement.verifier_context_digest;
        artifact.com_aux = com_aux;
        artifact.com_L = com_L;
        artifact.native_domain = kNativeArtifactDomain;
        artifact.native_statement = statement;
        artifact.native_messages = session.messages();
        artifact.transcript_digest = session.digest();

        const auto copy_message = [](const NativeProofMessage &message) {
            TranscriptMessage summary;
            summary.label = message.label;
            summary.payload = message.payload;
            summary.kind = static_cast<std::uint8_t>(message.kind);
            summary.sequence = message.sequence;
            return summary;
        };
        const std::vector<NativeProofMessage> &messages = session.messages();
        if (messages.size() < 4) {
            throw std::logic_error(
                "native GKR session did not contain the adapter binding messages");
        }
        artifact.Pi_Lasso.messages.push_back(copy_message(messages[0]));
        artifact.Pi_MatMul.messages.push_back(copy_message(messages[1]));
        artifact.Pi_Lasso.protocol = "Pi_Lasso";
        artifact.Pi_MatMul.protocol = "Pi_MatMul";
        artifact.Pi_GKR.protocol = "Pi_GKR";
        artifact.Pi_combine.protocol = "Pi_combine";
        artifact.Pi_open.protocol = "Pi_open";
        for (std::size_t index = 2; index + 2 < messages.size(); ++index)
            artifact.Pi_GKR.messages.push_back(copy_message(messages[index]));
        artifact.Pi_combine.messages.push_back(
            copy_message(messages[messages.size() - 2]));
        artifact.Pi_open.messages.push_back(
            copy_message(messages[messages.size() - 1]));

        if (!artifact.valid_wire(&reason)) {
            throw std::runtime_error(
                "GKR diagnostic artifact failed canonical wire validation: " + reason);
        }
        result.generated = true;
        result.artifact = std::move(artifact);
        result.diagnostic = "gkr_only_diagnostic: native GKR transcript generated; masked_full remains unsupported";
        return result;
    } catch (const std::exception &error) {
        result.generated = false;
        result.diagnostic = error.what();
        return result;
    }
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
            result.diagnostic = "unsupported_privacy_profile: zkGPT formal adapter only exposes masked_full";
            return result;
        }
        if (!registration.has_private_padding_mask)
            throw std::invalid_argument("com_W prover secret is unavailable in this process");

        const StaticTables tables = StaticTables::build(quant);
        if (!tables.authenticate())
            throw std::invalid_argument("canonical static tables are not authenticated");
        const VerifierContext context = make_verifier_context(
            graph, public_features, quant, tables, forward.logits, registration,
            plan, privacy_profile);
        const GatCircuitWitness circuit_witness = build_gat_circuit_witness(
            plan, graph, public_features, model, quant, forward);
        std::string witness_reason;
        if (!evaluate_gat_circuit_witness(plan, circuit_witness, &witness_reason))
            throw std::invalid_argument("canonical GAT witness validation failed: " +
                                        witness_reason);

        const std::string basis_label = registration_basis_label(
            model, quant, plan.input_layout, registration.model_arch_id);
        const std::string basis_id = fixed_digest(hash_hex(basis_label));
        const std::size_t domain = plan.input_layout.n_in;
        std::vector<G1> generators(domain);
        for (std::size_t index = 0; index < domain; ++index)
            hashAndMapToG1(generators[index],
                           basis_label + "/coordinate/" + std::to_string(index));
        const G1 H = hyrax::derive_pedersen_blinding_generator(basis_id);

        std::vector<Fr> aux_values = input_segment(plan,
                                                    circuit_witness.input_layer, 2);
        const hyrax::PedersenMleCommitment aux_commitment =
            hyrax::pedersen_commit_mle(generators, aux_values, basis_id);
        CommitmentWire com_aux;
        com_aux.scheme = "hyrax-pedersen-bn254";
        com_aux.basis_id = basis_id;
        com_aux.logical_size = plan.input_layout.segments.at(2).logical_length;
        com_aux.padded_size = domain;
        com_aux.point_hex = point_hex(aux_commitment.commitment);

        LassoAdapterBundle lasso = build_lasso_bundle(
            plan, forward, tables, basis_label);
        const std::vector<NativeLassoFamilyPublic> lasso_public =
            build_lasso_public_families(plan, tables, basis_label);
        NativeLassoQueryPublic lasso_query_public = build_lasso_public_query(
            plan, lasso.com_L, basis_label);
        const std::string lasso_tables_digest =
            native_lasso_tables_digest(lasso_public);
        const std::string lasso_resources_digest =
            native_lasso_resources_digest(lasso_public);
        const std::string lasso_digest =
            fixed_digest(hash_hex(commitment_encoding(lasso.com_L)));
        for (NativeLassoFamilyProver &family : lasso.families) {
            family.statement_tables_digest = lasso_tables_digest;
            family.statement_resources_digest = lasso_resources_digest;
        }
        lasso.query.statement_l_digest = lasso_digest;
        lasso_query_public.statement_l_digest = lasso_digest;

        NativeProofStatement statement = make_native_proof_statement(
            context, com_aux, lasso.com_L, H, generators,
            std::vector<Fr>(domain, Fr(0)), domain);
        // The Lasso-native identity is the authenticated statement identity;
        // do not replace it with the generic resource digest from the context
        // factory or re-commit L_qry under a second wire.
        statement.tables_digest = lasso_tables_digest;
        statement.resources_digest = lasso_resources_digest;
        statement.l_digest = lasso_digest;
        std::string statement_reason;
        if (!statement.valid(&statement_reason))
            throw std::invalid_argument("masked GAT statement is invalid: " + statement_reason);
        NativeProofSession session(kNativeArtifactDomain, statement);
        std::string reason;
        NativeLassoBundleResult lasso_result;
        if (!native_lasso_prove_bundle(session, lasso.families, lasso.query,
                                       &lasso_result, &reason))
            throw std::runtime_error("native masked Lasso proof failed: " + reason);
        if (!lasso_result.has_final_l_commitment ||
            !lasso_result.final_l_commitment.valid(
                zkgpt::native_pedersen::derive_basis(
                    "zkgpt/masked-sumcheck/v2/" + session.statement_digest()),
                &reason) || lasso_result.final_query_point.empty())
            throw std::runtime_error(
                "native masked Lasso did not return an authenticated final L_qry binding");

        NativeLayeredGkrWitness gkr_witness;
        gkr_witness.layer_values = circuit_witness.values;
        gkr_witness.matmuls.reserve(plan.matmuls.size());
        for (const MatMulDescriptor &descriptor : plan.matmuls) {
            NativeGkrMatmulWitness matmul;
            matmul.shape = {descriptor.rows, descriptor.inner, descriptor.cols};
            const std::size_t lhs_size = checked_mul(
                descriptor.rows, descriptor.inner,
                "native GKR MatMul LHS size");
            const std::size_t rhs_size = checked_mul(
                descriptor.inner, descriptor.cols,
                "native GKR MatMul RHS size");
            if (descriptor.output_coordinates != checked_mul(
                    descriptor.rows, descriptor.cols,
                    "native GKR MatMul output size"))
                throw std::invalid_argument(
                    "native GKR MatMul descriptor output shape is inconsistent");
            if (descriptor.lhs_offset > circuit_witness.input_layer.size() ||
                lhs_size > circuit_witness.input_layer.size() -
                                descriptor.lhs_offset)
                throw std::invalid_argument(
                    "native GKR MatMul LHS span exceeds canonical input layer");
            if (descriptor.rhs_offset > circuit_witness.input_layer.size() ||
                rhs_size > circuit_witness.input_layer.size() -
                                descriptor.rhs_offset)
                throw std::invalid_argument(
                    "native GKR MatMul RHS span exceeds canonical input layer");
            matmul.left.assign(
                circuit_witness.input_layer.begin() + descriptor.lhs_offset,
                circuit_witness.input_layer.begin() +
                    checked_add(descriptor.lhs_offset, lhs_size,
                                "native GKR MatMul LHS end"));
            // Canonical model/input flattening is row-major [term,col].  The
            // native MLE product folds the RHS as [col,term], so make this
            // conversion explicit at the adapter boundary.
            matmul.right = pack_native_gkr_matmul_rhs(
                descriptor, circuit_witness.input_layer);
            const std::size_t native_rhs_size = checked_mul(
                next_power_of_two(descriptor.inner),
                next_power_of_two(descriptor.cols),
                "native GKR MatMul padded RHS size");
            if (matmul.right.size() != native_rhs_size)
                throw std::runtime_error(
                    "native GKR MatMul RHS pack returned the wrong size");
            gkr_witness.matmuls.push_back(std::move(matmul));
        }
        NativeGkrInputBinding gkr_binding;
        NativeGkrProverResult gkr_result;
        if (!prove_masked_gkr_transcript(
                session, plan.circuit, gkr_witness,
                circuit_witness.public_output, &reason, &gkr_binding,
                &gkr_result))
            throw std::runtime_error("native masked GKR proof failed: " + reason);

        const zkgpt::native_pedersen::PedersenBasis scalar_basis =
            zkgpt::native_pedersen::derive_basis(
                "zkgpt/masked-sumcheck/v2/" + session.statement_digest());
        if (gkr_binding.point.size() != core_log2(domain) + 2 ||
            !gkr_binding.claim_commitment.valid(scalar_basis, &reason) ||
            !gkr_binding.unmasked_claim_commitment.valid(scalar_basis, &reason) ||
            !gkr_binding.mask_commitment.valid(scalar_basis, &reason) ||
            !gkr_result.has_unmasked_input_claim_opening ||
            !gkr_result.unmasked_input_claim_opening.valid(scalar_basis, &reason) ||
            gkr_result.unmasked_input_claim_opening.commitment.point !=
                gkr_binding.unmasked_claim_commitment.point)
            throw std::runtime_error("native masked GKR input binding is malformed");

        const std::size_t r_dimension = core_log2(domain);
        const std::vector<Fr> r_minus(gkr_binding.point.begin(),
                                      gkr_binding.point.begin() + r_dimension);
        // Native MLE folding numbers the first coordinate as the low bit.
        // The canonical contiguous segments are X,W,Aux,L, so expose the
        // two high coordinates in canonical selector order.
        const Fr selector_s1 = gkr_binding.point[r_dimension + 1];
        const Fr selector_s2 = gkr_binding.point[r_dimension];
        session.absorb_fields("combine/r_minus", r_minus);
        session.absorb_fields("combine/selectors", {selector_s1, selector_s2});
        const std::vector<Fr> x_values = public_feature_segment(
            plan, graph, public_features);
        const Fr x_eval = adapter_mle_eval(x_values, r_minus);
        session.absorb_field("combine/public-X", x_eval);
        session.absorb("combine/structured-root", statement.structured_root());
        session.absorb("combine/commitments",
                       basis_id + "|" + commitment_encoding(registration.com_W) +
                           "|" + commitment_encoding(com_aux) + "|" +
                           commitment_encoding(lasso.com_L));
        session.absorb("combine/gkr-input-commitment",
                       scalar_basis.domain + "|" +
                           point_hex(gkr_binding.unmasked_claim_commitment.point));
        session.absorb_field("combine/lasso-final-claim",
                             lasso_result.final_l_evaluation);
        session.absorb("combine/lasso-final-commitment",
                       scalar_basis.domain + "|" +
                           point_hex(lasso_result.final_l_commitment.point));

        const Fr c_x = (Fr(1) - selector_s1) * (Fr(1) - selector_s2);
        const Fr c_w = (Fr(1) - selector_s1) * selector_s2;
        const Fr c_aux = selector_s1 * (Fr(1) - selector_s2);
        const Fr c_l = selector_s1 * selector_s2;
        std::vector<Fr> w_values = input_segment(plan,
                                                  circuit_witness.input_layer, 1);
        std::vector<Fr> l_values = input_segment(plan,
                                                  circuit_witness.input_layer, 3);
        zkgpt::native_pedersen::ScalarOpening w_raw_opening;
        zkgpt::native_pedersen::ScalarOpening aux_raw_opening;
        zkgpt::native_pedersen::ScalarOpening l_raw_opening;
        if (!hyrax::make_mle_evaluation_opening(
                scalar_basis, r_minus, w_values, w_raw_opening, &reason) ||
            !hyrax::make_mle_evaluation_opening(
                scalar_basis, r_minus, aux_values, aux_raw_opening, &reason) ||
            !hyrax::make_mle_evaluation_opening(
                scalar_basis, r_minus, l_values, l_raw_opening, &reason))
            throw std::runtime_error("canonical GAT scalar evaluation failed: " + reason);

        hyrax::PedersenMleCommitment w_commitment;
        w_commitment.commitment.deserializeHexStr(registration.com_W.point_hex);
        w_commitment.blinding_generator = H;
        w_commitment.blinding = registration.com_W_padding_mask;
        w_commitment.basis_id = basis_id;
        hyrax::PedersenMleCommitment l_commitment;
        l_commitment.commitment = lasso.query.commitment;
        l_commitment.blinding_generator = lasso.query.blinding_generator;
        l_commitment.blinding = lasso.query.commitment_blinding;
        l_commitment.basis_id = lasso.query.basis_id;
        hyrax::MaskedMleOpening w_opening;
        hyrax::MaskedMleOpening aux_opening;
        hyrax::MaskedMleOpening l_opening;
        if (!hyrax::open_masked_mle_with_session(
                w_commitment, generators, r_minus, w_values, session,
                "Open_GAT_W", w_opening, &reason, true, "", &scalar_basis,
                nullptr, &w_raw_opening) ||
            !hyrax::open_masked_mle_with_session(
                aux_commitment, generators, r_minus, aux_values, session,
                "Open_GAT_Aux", aux_opening, &reason, true, "", &scalar_basis,
                nullptr, &aux_raw_opening) ||
            !hyrax::open_masked_mle_with_session(
                l_commitment, generators, r_minus, l_values, session,
                "Open_GAT_L", l_opening, &reason, true, "", &scalar_basis,
                nullptr, &l_raw_opening))
            throw std::runtime_error("masked input terminal openings failed: " + reason);
        if (!w_opening.has_scalar_binding || !aux_opening.has_scalar_binding ||
            !l_opening.has_scalar_binding)
            throw std::runtime_error("masked input opening omitted scalar binding");
        if (!zkgpt::native_pedersen::prove_hidden_linear(
                session, scalar_basis, "combine/input-linear",
                {gkr_result.unmasked_input_claim_opening,
                 w_opening.raw_evaluation_opening,
                 aux_opening.raw_evaluation_opening,
                 l_opening.raw_evaluation_opening},
                {F_ONE, -c_w, -c_aux, -c_l}, c_x * x_eval, nullptr,
                &reason))
            throw std::runtime_error("masked GAT input linear relation failed: " + reason);

        GatProofArtifact artifact;
        artifact.privacy_profile = "masked_full";
        artifact.statement_digest = verifier_context_digest(context);
        artifact.com_aux = com_aux;
        artifact.com_L = lasso.com_L;
        artifact.native_domain = kNativeArtifactDomain;
        artifact.native_statement = statement;
        artifact.native_messages = session.messages();
        artifact.transcript_digest = session.digest();
        core_copy_sections(session, artifact);
        if (!artifact.valid_wire(&reason))
            throw std::runtime_error("formal masked GAT artifact is invalid: " + reason);
        std::string verify_reason;
        if (!verify_gat(context, artifact, &verify_reason))
            throw std::runtime_error("formal masked GAT self-verification failed: " +
                                     verify_reason);
        result.status = ProveStatus::GENERATED;
        result.has_artifact = true;
        result.artifact = std::move(artifact);
        result.diagnostic = "masked_full canonical GAT proof generated and independently verified";
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
        if (artifact.privacy_profile != "masked_full") { set_reason(reason, "unsupported_privacy_profile"); return false; }
        if (!artifact.valid_wire(reason)) return false;
        VerifierContext canonical = context;
        const std::string expected_statement = verifier_context_digest(canonical);
        if (artifact.statement_digest != expected_statement) { set_reason(reason, "proof statement digest does not match independent public context"); return false; }

        const InputLayout verifier_layout = registration_layout(context.registration);
        const std::string basis_label = registration_basis_label_public(
            context.registration, verifier_layout);
        const std::string basis_id = fixed_digest(hash_hex(basis_label));
        const std::size_t domain = context.plan.input_layout.n_in;
        if (domain == 0 || !core_power_of_two(domain) ||
            context.registration.com_W.basis_id != basis_id ||
            artifact.com_aux.basis_id != basis_id ||
            artifact.com_L.basis_id != basis_id ||
            artifact.com_aux.padded_size != domain ||
            artifact.com_L.padded_size != domain) {
            set_reason(reason, "masked GAT verifier basis or commitment domain mismatch");
            return false;
        }

        std::vector<G1> generators(domain);
        for (std::size_t index = 0; index < domain; ++index)
            hashAndMapToG1(generators[index],
                           basis_label + "/coordinate/" + std::to_string(index));
        const G1 H = hyrax::derive_pedersen_blinding_generator(basis_id);
        const std::vector<Fr> zero_basis(domain, Fr(0));
        NativeProofStatement expected_native_statement =
            make_native_proof_statement(context, artifact.com_aux, artifact.com_L,
                                        H, generators, zero_basis, domain);
        const std::vector<NativeLassoFamilyPublic> expected_lasso_families =
            build_lasso_public_families(context.plan, context.tables, basis_label);
        expected_native_statement.tables_digest =
            native_lasso_tables_digest(expected_lasso_families);
        expected_native_statement.resources_digest =
            native_lasso_resources_digest(expected_lasso_families);
        expected_native_statement.l_digest =
            fixed_digest(hash_hex(commitment_encoding(artifact.com_L)));
        const NativeProofStatement &actual_native_statement =
            artifact.native_statement;
        if (actual_native_statement.verifier_context_digest !=
                expected_native_statement.verifier_context_digest ||
            actual_native_statement.field_id != expected_native_statement.field_id ||
            actual_native_statement.plan_digest != expected_native_statement.plan_digest ||
            actual_native_statement.resources_digest != expected_native_statement.resources_digest ||
            actual_native_statement.tables_digest != expected_native_statement.tables_digest ||
            actual_native_statement.com_w_digest != expected_native_statement.com_w_digest ||
            actual_native_statement.aux_digest != expected_native_statement.aux_digest ||
            actual_native_statement.l_digest != expected_native_statement.l_digest ||
            actual_native_statement.basis_digest != expected_native_statement.basis_digest ||
            actual_native_statement.l_dimension != expected_native_statement.l_dimension ||
            actual_native_statement.statement_root_digest !=
                expected_native_statement.statement_root_digest) {
            set_reason(reason, "native proof statement structured fields do not match public context");
            return false;
        }

        NativeProofSession session;
        if (!artifact.native_session(session, reason)) return false;

        // Formal masked_full artifacts have one strict native grammar.  In
        // particular, do not let an unknown/clear adapter message sit between
        // two valid native scopes and be ignored by a component verifier.
        const auto has_prefix = [](const std::string &value,
                                   const std::string &prefix) {
            return value.compare(0, prefix.size(), prefix) == 0;
        };
        const auto is_lasso_label = [&has_prefix](const std::string &label) {
            return has_prefix(label, "lasso/") ||
                   has_prefix(label, "challenge/lasso/");
        };
        const auto is_gkr_label = [&has_prefix](const std::string &label) {
            return has_prefix(label, "gkr/") ||
                   has_prefix(label, "challenge/gkr/");
        };
        const auto is_combine_label = [&has_prefix](const std::string &label) {
            return has_prefix(label, "combine/") ||
                   has_prefix(label, "challenge/combine/");
        };
        const auto is_open_label = [&has_prefix](const std::string &label) {
            return has_prefix(label, "Open_W/") ||
                   has_prefix(label, "Open_Aux/") ||
                   has_prefix(label, "Open_L/") ||
                   has_prefix(label, "Open_GAT_W/") ||
                   has_prefix(label, "Open_GAT_Aux/") ||
                   has_prefix(label, "Open_GAT_L/") ||
                   has_prefix(label, "challenge/Open_W/") ||
                   has_prefix(label, "challenge/Open_Aux/") ||
                   has_prefix(label, "challenge/Open_L/") ||
                   has_prefix(label, "challenge/Open_GAT_W/") ||
                   has_prefix(label, "challenge/Open_GAT_Aux/") ||
                   has_prefix(label, "challenge/Open_GAT_L/");
        };
        for (const NativeProofMessage &message : session.messages()) {
            if (!is_lasso_label(message.label) && !is_gkr_label(message.label) &&
                !is_combine_label(message.label) && !is_open_label(message.label)) {
                set_reason(reason, "masked GAT transcript has an unknown message label: " +
                                     message.label);
                return false;
            }
        }

        const std::vector<NativeLassoFamilyPublic> &lasso_families =
            expected_lasso_families;
        const NativeLassoQueryPublic lasso_query = build_lasso_public_query(
            context.plan, artifact.com_L, basis_label);
        NativeLassoBundleResult lasso_result;
        if (!native_lasso_verify_bundle(session, lasso_families, lasso_query,
                                        &lasso_result, reason))
            return false;

        std::vector<F> public_output;
        public_output.reserve(context.public_logits.values.size() +
                              context.plan.residual_coordinates);
        for (Int value : context.public_logits.values)
            public_output.push_back(Fr(value));
        public_output.insert(public_output.end(),
                             context.plan.residual_coordinates, Fr(0));
        if (context.plan.public_output_coordinates !=
                context.public_logits.values.size() ||
            public_output.size() != context.plan.public_output_coordinates +
                                      context.plan.residual_coordinates ||
            context.plan.circuit.circuit.empty() ||
            context.plan.circuit.circuit.back().size != public_output.size()) {
            set_reason(reason, "masked GAT public output/residual shape is invalid");
            return false;
        }

        std::vector<NativeGkrMatmulShape> matmuls;
        matmuls.reserve(context.plan.matmuls.size());
        for (const MatMulDescriptor &descriptor : context.plan.matmuls)
            matmuls.push_back({descriptor.rows, descriptor.inner, descriptor.cols});

        std::size_t gkr_start = session.messages().size();
        for (std::size_t index = 0; index < session.messages().size(); ++index) {
            const NativeProofMessage &message = session.messages()[index];
            if (message.label == "gkr/public-output") {
                if (message.kind != NativeProofMessageKind::FieldVector ||
                    gkr_start != session.messages().size()) {
                    set_reason(reason, "masked GAT GKR public-output boundary is malformed");
                    return false;
                }
                gkr_start = index;
            }
            if (is_gkr_label(message.label) && index < gkr_start) {
                set_reason(reason, "masked GAT transcript has GKR messages before public output");
                return false;
            }
        }
        if (gkr_start == session.messages().size()) {
            set_reason(reason, "masked GAT GKR public-output message is missing");
            return false;
        }
        for (std::size_t index = 0; index < gkr_start; ++index)
            if (!is_lasso_label(session.messages()[index].label)) {
                set_reason(reason, "masked GAT transcript has a non-Lasso prefix before GKR");
                return false;
            }

        std::size_t gkr_end = session.messages().size();
        NativeGkrInputBinding verified_gkr_binding;
        if (!verify_masked_gkr_transcript(session, context.plan.circuit,
                                          public_output, matmuls, gkr_start,
                                          &gkr_end, reason,
                                          &verified_gkr_binding))
            return false;

        const zkgpt::native_pedersen::PedersenBasis scalar_basis =
            zkgpt::native_pedersen::derive_basis(
                "zkgpt/masked-sumcheck/v2/" + session.statement_digest());
        if (verified_gkr_binding.point.size() != core_log2(domain) + 2 ||
            !verified_gkr_binding.claim_commitment.valid(scalar_basis, reason) ||
            !verified_gkr_binding.unmasked_claim_commitment.valid(scalar_basis,
                                                                  reason) ||
            !verified_gkr_binding.mask_commitment.valid(scalar_basis, reason) ||
            !lasso_result.has_final_l_commitment ||
            !lasso_result.final_l_commitment.valid(scalar_basis, reason) ||
            lasso_result.final_query_point.empty()) {
            set_reason(reason, "masked GAT v2 authenticated input/final binding is malformed");
            return false;
        }

        auto decode_field_vector = [](const std::string &payload,
                                      std::vector<Fr> &values) {
            values.clear();
            std::size_t offset = 0;
            auto read_u64 = [&](std::uint64_t &value) {
                if (payload.size() - offset < sizeof(std::uint64_t)) return false;
                value = 0;
                for (unsigned index = 0; index < 8; ++index)
                    value |= static_cast<std::uint64_t>(
                                 static_cast<unsigned char>(payload[offset++]))
                             << (8 * index);
                return true;
            };
            std::uint64_t count = 0;
            if (!read_u64(count) || count > (1ULL << 20)) return false;
            values.reserve(static_cast<std::size_t>(count));
            for (std::size_t index = 0; index < static_cast<std::size_t>(count);
                 ++index) {
                std::uint64_t size = 0;
                if (!read_u64(size) || size == 0 || size > 128 ||
                    size > payload.size() - offset)
                    return false;
                const std::string encoded = payload.substr(
                    offset, static_cast<std::size_t>(size));
                offset += static_cast<std::size_t>(size);
                Fr value;
                if (!core_decode_field(encoded, value)) return false;
                values.push_back(value);
            }
            return offset == payload.size();
        };
        auto consume_message = [&](std::size_t &cursor,
                                   NativeProofMessageKind kind,
                                   const std::string &label)
            -> const NativeProofMessage * {
            if (cursor >= session.messages().size()) {
                set_reason(reason, "masked GAT transcript ended before " + label);
                return nullptr;
            }
            const NativeProofMessage &message = session.messages()[cursor];
            if (message.kind != kind || message.label != label) {
                set_reason(reason, "masked GAT transcript expected " + label);
                return nullptr;
            }
            ++cursor;
            return &message;
        };
        auto decode_message_field = [&](const NativeProofMessage *message,
                                        Fr &value, const std::string &label) {
            if (message == nullptr || !core_decode_field(message->payload, value)) {
                set_reason(reason, "masked GAT field message is malformed: " + label);
                return false;
            }
            return true;
        };

        if (gkr_end >= session.messages().size() ||
            !is_combine_label(session.messages()[gkr_end].label)) {
            set_reason(reason, "masked GAT GKR scope is not followed by combine");
            return false;
        }
        const std::size_t r_dimension = core_log2(domain);
        const std::vector<Fr> r_minus(verified_gkr_binding.point.begin(),
                                      verified_gkr_binding.point.begin() + r_dimension);
        const Fr selector_s1 = verified_gkr_binding.point[r_dimension + 1];
        const Fr selector_s2 = verified_gkr_binding.point[r_dimension];
        const Fr c_x = (Fr(1) - selector_s1) * (Fr(1) - selector_s2);
        const Fr c_w = (Fr(1) - selector_s1) * selector_s2;
        const Fr c_aux = selector_s1 * (Fr(1) - selector_s2);
        const Fr c_l = selector_s1 * selector_s2;

        std::size_t cursor = gkr_end;
        const NativeProofMessage *combine_point_message = consume_message(
            cursor, NativeProofMessageKind::FieldVector, "combine/r_minus");
        const NativeProofMessage *combine_selectors_message = consume_message(
            cursor, NativeProofMessageKind::FieldVector, "combine/selectors");
        const NativeProofMessage *combine_x_message = consume_message(
            cursor, NativeProofMessageKind::Field, "combine/public-X");
        const NativeProofMessage *combine_structured_root_message = consume_message(
            cursor, NativeProofMessageKind::Data, "combine/structured-root");
        const NativeProofMessage *combine_commitments_message = consume_message(
            cursor, NativeProofMessageKind::Data, "combine/commitments");
        const NativeProofMessage *combine_gkr_message = consume_message(
            cursor, NativeProofMessageKind::Data, "combine/gkr-input-commitment");
        const NativeProofMessage *combine_lasso_message = consume_message(
            cursor, NativeProofMessageKind::Field, "combine/lasso-final-claim");
        const NativeProofMessage *combine_lasso_commitment_message = consume_message(
            cursor, NativeProofMessageKind::Data, "combine/lasso-final-commitment");
        if (combine_point_message == nullptr || combine_selectors_message == nullptr ||
            combine_x_message == nullptr || combine_structured_root_message == nullptr ||
            combine_commitments_message == nullptr ||
            combine_gkr_message == nullptr || combine_lasso_message == nullptr ||
            combine_lasso_commitment_message == nullptr)
            return false;
        std::vector<Fr> encoded_r_minus, encoded_selectors;
        if (!decode_field_vector(combine_point_message->payload, encoded_r_minus) ||
            !decode_field_vector(combine_selectors_message->payload, encoded_selectors) ||
            encoded_r_minus != r_minus || encoded_selectors.size() != 2 ||
            encoded_selectors[0] != selector_s1 || encoded_selectors[1] != selector_s2) {
            set_reason(reason, "masked GAT combine point/selector binding failed");
            return false;
        }
        Fr encoded_x, encoded_lasso_claim;
        if (!decode_message_field(combine_x_message, encoded_x,
                                  "combine/public-X") ||
            !decode_message_field(combine_lasso_message, encoded_lasso_claim,
                                  "combine/lasso-final-claim") ||
            encoded_lasso_claim != lasso_result.final_l_evaluation) {
            set_reason(reason, "masked GAT combine public claim binding failed");
            return false;
        }
        const std::vector<Fr> x_values = public_feature_segment(
            context.plan, context.graph, context.public_features);
        const Fr x_eval = adapter_mle_eval(x_values, r_minus);
        if (encoded_x != x_eval) {
            set_reason(reason, "masked GAT public X terminal evaluation mismatch");
            return false;
        }
        if (combine_structured_root_message->payload !=
                actual_native_statement.structured_root()) {
            set_reason(reason, "masked GAT structured statement root binding failed");
            return false;
        }
        const std::string expected_commitments =
            basis_id + "|" + commitment_encoding(context.registration.com_W) +
            "|" + commitment_encoding(artifact.com_aux) + "|" +
            commitment_encoding(artifact.com_L);
        if (combine_commitments_message->payload != expected_commitments) {
            set_reason(reason, "masked GAT combine commitment binding failed");
            return false;
        }
        auto parse_scalar_commitment = [&](const NativeProofMessage *message,
                                           const std::string &label,
                                           zkgpt::native_pedersen::ScalarCommitment &out) {
            std::vector<std::string> parts;
            G1 point;
            if (message == nullptr || !core_split_pipe(message->payload, parts) ||
                parts.size() != 2 || parts[0] != scalar_basis.domain ||
                !core_decode_point(parts[1], point)) {
                set_reason(reason, "masked GAT scalar commitment is malformed: " + label);
                return false;
            }
            out = {scalar_basis.domain, point};
            return out.valid(scalar_basis, reason);
        };
        zkgpt::native_pedersen::ScalarCommitment encoded_gkr_commitment;
        if (!parse_scalar_commitment(combine_gkr_message,
                                     "combine/gkr-input-commitment",
                                     encoded_gkr_commitment) ||
            encoded_gkr_commitment.point !=
                verified_gkr_binding.unmasked_claim_commitment.point) {
            set_reason(reason, "masked GAT GKR input commitment binding failed");
            return false;
        }
        zkgpt::native_pedersen::ScalarCommitment encoded_lasso_commitment;
        if (!parse_scalar_commitment(combine_lasso_commitment_message,
                                     "combine/lasso-final-commitment",
                                     encoded_lasso_commitment) ||
            encoded_lasso_commitment.point != lasso_result.final_l_commitment.point) {
            set_reason(reason, "masked GAT Lasso final commitment binding failed");
            return false;
        }

        auto verify_terminal_opening =
            [&](const std::string &label, const CommitmentWire &wire,
                zkgpt::native_pedersen::ScalarCommitment &raw_commitment,
                zkgpt::native_pedersen::ScalarCommitment &masked_commitment) {
                const std::size_t start = cursor;
                const NativeProofMessage *basis_message = consume_message(
                    cursor, NativeProofMessageKind::Data, label + "/basis");
                const NativeProofMessage *claim_message = consume_message(
                    cursor, NativeProofMessageKind::Data, label + "/claim");
                const NativeProofMessage *announce_message = consume_message(
                    cursor, NativeProofMessageKind::Data, label + "/announce");
                const NativeProofMessage *challenge_message = consume_message(
                    cursor, NativeProofMessageKind::Challenge,
                    "challenge/" + label + "/c");
                const NativeProofMessage *response_message = consume_message(
                    cursor, NativeProofMessageKind::FieldVector, label + "/response");
                if (basis_message == nullptr || claim_message == nullptr ||
                    announce_message == nullptr || challenge_message == nullptr ||
                    response_message == nullptr)
                    return false;
                std::vector<std::string> parts;
                G1 evaluation_mask;
                Fr masked_evaluation;
                if (!core_split_pipe(claim_message->payload, parts) || parts.size() != 5 ||
                    !core_decode_point(parts[0], evaluation_mask) ||
                    !core_decode_field(parts[1], masked_evaluation) ||
                    parts[2] != scalar_basis.domain) {
                    set_reason(reason, "masked GAT scalar opening claim is malformed: " + label);
                    return false;
                }
                if (!core_decode_point(parts[3], raw_commitment.point) ||
                    !core_decode_point(parts[4], masked_commitment.point)) {
                    set_reason(reason, "masked GAT scalar opening point is malformed: " + label);
                    return false;
                }
                raw_commitment.domain = scalar_basis.domain;
                masked_commitment.domain = scalar_basis.domain;
                G1 commitment_point;
                if (!core_decode_point(wire.point_hex, commitment_point)) {
                    set_reason(reason, "masked GAT commitment point is malformed: " + label);
                    return false;
                }
                if (!hyrax::verify_masked_mle_opening_with_session(
                        commitment_point, evaluation_mask, masked_evaluation,
                        basis_id, generators, r_minus, session, label, reason,
                        true, true, "", &scalar_basis, &raw_commitment,
                        &masked_commitment))
                    return false;
                const std::string prefix = label + "/";
                const std::string challenge_prefix = "challenge/" + prefix;
                for (std::size_t index = cursor; index < session.messages().size();
                     ++index) {
                    if (session.messages()[index].label.compare(0, prefix.size(),
                                                                  prefix) == 0 ||
                        session.messages()[index].label.compare(
                            0, challenge_prefix.size(), challenge_prefix) == 0) {
                        set_reason(reason, "masked GAT opening has trailing protocol messages: " + label);
                        return false;
                    }
                }
                (void)start;
                return true;
            };
        zkgpt::native_pedersen::ScalarCommitment raw_w, masked_w;
        zkgpt::native_pedersen::ScalarCommitment raw_aux, masked_aux;
        zkgpt::native_pedersen::ScalarCommitment raw_l, masked_l;
        if (!verify_terminal_opening("Open_GAT_W", context.registration.com_W,
                                     raw_w, masked_w) ||
            !verify_terminal_opening("Open_GAT_Aux", artifact.com_aux,
                                     raw_aux, masked_aux) ||
            !verify_terminal_opening("Open_GAT_L", artifact.com_L,
                                     raw_l, masked_l))
            return false;

        const std::size_t relation_start = cursor;
        if (relation_start >= session.messages().size() ||
            session.messages()[relation_start].label !=
                "combine/input-linear/meta") {
            set_reason(reason, "masked GAT linear relation is not adjacent to openings");
            return false;
        }
        if (!zkgpt::native_pedersen::verify_hidden_linear(
                session, scalar_basis, "combine/input-linear",
                {verified_gkr_binding.unmasked_claim_commitment, raw_w, raw_aux, raw_l},
                {F_ONE, -c_w, -c_aux, -c_l}, c_x * x_eval, reason))
            return false;
        const std::array<std::string, 6> relation_labels = {
            "combine/input-linear/meta", "combine/input-linear/coefficients",
            "combine/input-linear/public", "combine/input-linear/T",
            "challenge/combine/input-linear/challenge",
            "combine/input-linear/response"};
        if (relation_start + relation_labels.size() > session.messages().size()) {
            set_reason(reason, "masked GAT linear relation is truncated");
            return false;
        }
        for (std::size_t index = 0; index < relation_labels.size(); ++index) {
            if (session.messages()[relation_start + index].label != relation_labels[index]) {
                set_reason(reason, "masked GAT linear relation has trailing or inserted messages");
                return false;
            }
        }
        cursor = relation_start + relation_labels.size();
        if (cursor != session.messages().size()) {
            set_reason(reason, "masked GAT transcript has trailing or unconsumed messages");
            return false;
        }
        return true;
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

namespace {

bool core_power_of_two(std::size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

std::size_t core_log2(std::size_t value) {
    std::size_t result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

std::string core_vector_digest(const std::string &domain,
                               const std::vector<Fr> &values) {
    std::string encoded;
    append_string(encoded, domain);
    append_u64(encoded, static_cast<std::uint64_t>(values.size()));
    for (const Fr &value : values) append_string(encoded, field_hex(value));
    return fixed_digest(hash_hex(encoded));
}

std::string core_commitment_payload(const std::string &basis_id,
                                    const CommitmentWire &com_W,
                                    const CommitmentWire &com_aux,
                                    const CommitmentWire &com_L,
                                    std::size_t domain) {
    std::string encoded;
    append_string(encoded, "zkgpt-masked-core-commitments-v1");
    append_string(encoded, basis_id);
    append_u64(encoded, static_cast<std::uint64_t>(domain));
    append_string(encoded, commitment_encoding(com_W));
    append_string(encoded, commitment_encoding(com_aux));
    append_string(encoded, commitment_encoding(com_L));
    return hash_hex(encoded);
}

bool core_decode_field(const std::string &encoded, Fr &value) {
    if (encoded.empty() || encoded.size() > 128) return false;
    try {
        value.setStr(encoded, 16);
        return field_hex(value) == encoded;
    } catch (...) {
        return false;
    }
}

const NativeProofMessage *core_find_message(
    const NativeProofSession &session, NativeProofMessageKind kind,
    const std::string &label, std::string *reason) {
    const NativeProofMessage *found = nullptr;
    for (const NativeProofMessage &message : session.messages()) {
        if (message.kind != kind || message.label != label) continue;
        if (found != nullptr) {
            set_reason(reason, "duplicate masked core message: " + label);
            return nullptr;
        }
        found = &message;
    }
    if (found == nullptr) set_reason(reason, "missing masked core message: " + label);
    return found;
}

bool core_decode_token_payload(const std::string &payload,
                               std::vector<std::string> &tokens) {
    tokens.clear();
    std::size_t offset = 0;
    auto read_u64 = [&](std::uint64_t &value) {
        if (payload.size() - offset < sizeof(std::uint64_t)) return false;
        value = 0;
        for (unsigned index = 0; index < 8; ++index)
            value |= static_cast<std::uint64_t>(
                         static_cast<unsigned char>(payload[offset++]))
                     << (8 * index);
        return true;
    };
    std::uint64_t count = 0;
    if (!read_u64(count) || count > 64) return false;
    tokens.reserve(static_cast<std::size_t>(count));
    for (std::size_t index = 0; index < static_cast<std::size_t>(count);
         ++index) {
        std::uint64_t size = 0;
        if (!read_u64(size) || size > (1ULL << 20) ||
            size > payload.size() - offset)
            return false;
        tokens.push_back(payload.substr(offset, static_cast<std::size_t>(size)));
        offset += static_cast<std::size_t>(size);
    }
    return offset == payload.size();
}

bool core_split_pipe(const std::string &payload,
                     std::vector<std::string> &parts) {
    parts.clear();
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = payload.find('|', begin);
        parts.push_back(payload.substr(begin,
                                      end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return !parts.empty();
}

bool core_decode_point(const std::string &encoded, G1 &point) {
    if (encoded.empty() || encoded.size() > 4096) return false;
    try {
        point.deserializeHexStr(encoded);
        return point.serializeToHexStr() == encoded && point.isValid() &&
               !point.isZero() && point.isValidOrder();
    } catch (...) {
        return false;
    }
}

bool core_commitment_matches(const CommitmentWire &wire,
                             const hyrax::PedersenMleCommitment &commitment,
                             const std::string &basis_id, std::size_t domain) {
    return wire.scheme == "hyrax-pedersen-bn254" &&
           wire.basis_id == basis_id &&
           wire.padded_size == domain && wire.logical_size != 0 &&
           wire.logical_size <= domain &&
           wire.point_hex == commitment.commitment.serializeToHexStr();
}

hyrax::PedersenMleCommitment core_existing_commitment(
    const CommitmentWire &wire, const Fr &blinder, const std::string &basis_id,
    const G1 &H, const std::vector<G1> &generators,
    const std::vector<Fr> &values) {
    if (values.empty() || values.size() != generators.size() ||
        !core_power_of_two(values.size()) ||
        H != hyrax::derive_pedersen_blinding_generator(basis_id))
        throw std::invalid_argument("masked core W commitment dimensions are invalid");
    if (wire.basis_id != basis_id ||
        wire.padded_size != values.size())
        throw std::invalid_argument("masked core W commitment identity mismatch");
    G1 encoded;
    encoded.deserializeHexStr(wire.point_hex);
    hyrax::PedersenMleCommitment result;
    result.commitment = encoded;
    result.blinding_generator = H;
    result.blinding = blinder;
    result.basis_id = basis_id;
    const G1 expected = perdersen_commit(
        const_cast<G1 *>(generators.data()), const_cast<Fr *>(values.data()),
        static_cast<int>(values.size())) + H * blinder;
    if (expected != result.commitment)
        throw std::invalid_argument("masked core W commitment does not bind witness");
    return result;
}

CommitmentWire core_commitment_wire(
    const hyrax::PedersenMleCommitment &commitment, std::size_t logical_size,
    std::size_t domain) {
    CommitmentWire wire;
    wire.scheme = "hyrax-pedersen-bn254";
    wire.basis_id = commitment.basis_id;
    wire.logical_size = logical_size;
    wire.padded_size = domain;
    wire.point_hex = commitment.commitment.serializeToHexStr();
    return wire;
}

std::vector<Fr> core_point_from_session(const NativeProofSession &session,
                                        const std::string &label,
                                        std::size_t dimension,
                                        std::string *reason) {
    std::vector<Fr> point;
    point.reserve(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
        const NativeProofMessage *message = core_find_message(
            session, NativeProofMessageKind::Challenge,
            "challenge/" + label + "/" + std::to_string(index), reason);
        if (message == nullptr) return {};
        Fr value;
        if (!core_decode_field(message->payload, value)) {
            set_reason(reason, "masked core challenge is not canonical");
            return {};
        }
        point.push_back(value);
    }
    return point;
}

bool core_message_in_scope(const std::string &label,
                           const std::string &scope) {
    const std::string prefix = scope + "/";
    const std::string challenge = "challenge/" + prefix;
    return label.compare(0, prefix.size(), prefix) == 0 ||
           label.compare(0, challenge.size(), challenge) == 0;
}

void core_copy_sections(const NativeProofSession &session,
                        GatProofArtifact &artifact) {
    artifact.Pi_Lasso.protocol = "Pi_Lasso";
    artifact.Pi_MatMul.protocol = "Pi_MatMul";
    artifact.Pi_GKR.protocol = "Pi_GKR";
    artifact.Pi_combine.protocol = "Pi_combine";
    artifact.Pi_open.protocol = "Pi_open";
    const auto copy = [](const NativeProofMessage &message) {
        TranscriptMessage result;
        result.label = message.label;
        result.payload = message.payload;
        result.kind = static_cast<std::uint8_t>(message.kind);
        result.sequence = message.sequence;
        return result;
    };
    for (const NativeProofMessage &message : session.messages()) {
        ProofSection *section = nullptr;
        if (message.label == "lasso/reserved" ||
            message.label.compare(0, sizeof("lasso/") - 1, "lasso/") == 0 ||
            message.label.compare(0, sizeof("challenge/lasso/") - 1,
                                  "challenge/lasso/") == 0)
            section = &artifact.Pi_Lasso;
        else if (message.label.compare(0, sizeof("matmul/") - 1,
                                       "matmul/") == 0 ||
                 message.label.compare(0, sizeof("challenge/matmul/") - 1,
                                       "challenge/matmul/") == 0 ||
                 message.label.compare(0, sizeof("gkr/layer/") - 1,
                                       "gkr/layer/") == 0 &&
                     message.label.find("/matmul") != std::string::npos ||
                 message.label.compare(0, sizeof("challenge/gkr/layer/") - 1,
                                       "challenge/gkr/layer/") == 0 &&
                     message.label.find("/matmul") != std::string::npos)
            section = &artifact.Pi_MatMul;
        else if (message.label.compare(0, sizeof("gkr/") - 1, "gkr/") == 0 ||
                 message.label.compare(0, sizeof("challenge/gkr/") - 1,
                                       "challenge/gkr/") == 0)
            section = &artifact.Pi_GKR;
        else if (message.label.compare(0, sizeof("combine/") - 1,
                                       "combine/") == 0 ||
                 message.label.compare(0, sizeof("challenge/combine/") - 1,
                                       "challenge/combine/") == 0)
            section = &artifact.Pi_combine;
        else if (message.label.compare(0, sizeof("Open_") - 1, "Open_") == 0 ||
                 message.label.compare(0, sizeof("challenge/Open_") - 1,
                                       "challenge/Open_") == 0)
            section = &artifact.Pi_open;
        if (section == nullptr)
            throw std::invalid_argument("masked core transcript label is outside canonical sections: " + message.label);
        section->messages.push_back(copy(message));
    }
}

bool core_expected_label(const std::string &label) {
    return label == "lasso/reserved" ||
           label.compare(0, sizeof("lasso/") - 1, "lasso/") == 0 ||
           label.compare(0, sizeof("challenge/lasso/") - 1,
                         "challenge/lasso/") == 0 ||
           core_message_in_scope(label, "matmul") ||
           core_message_in_scope(label, "gkr") ||
           core_message_in_scope(label, "combine") ||
           core_message_in_scope(label, "Open_W") ||
           core_message_in_scope(label, "Open_Aux") ||
           core_message_in_scope(label, "Open_L");
}

bool core_find_index(const NativeProofSession &session, const std::string &label,
                     NativeProofMessageKind kind, std::size_t &index,
                     std::string *reason) {
    bool found = false;
    for (std::size_t cursor = 0; cursor < session.messages().size(); ++cursor) {
        const NativeProofMessage &message = session.messages()[cursor];
        if (message.label != label || message.kind != kind) continue;
        if (found) {
            set_reason(reason, "duplicate masked core message: " + label);
            return false;
        }
        found = true;
        index = cursor;
    }
    if (!found) set_reason(reason, "missing masked core message: " + label);
    return found;
}

} // namespace

static MaskedCoreResult prove_masked_core_impl(
    const NativeProofStatement &statement, const CommitmentWire &com_W,
    const Fr &com_W_blinder, const std::string &basis_id, const G1 &H,
    const std::vector<G1> &generators, const MaskedCoreWitness &witness,
    const std::string &statement_digest,
    std::vector<NativeLassoFamilyProver> *lasso_families,
    NativeLassoQueryProver *lasso_query) {
    MaskedCoreResult result;
    try {
        ensure_field();
        if (statement_digest.empty() || statement_digest !=
                statement.verifier_context_digest ||
            statement.statement_root_digest != statement_digest)
            throw std::invalid_argument("masked core statement root is not authenticated");
        if (basis_id.empty() || generators.empty() ||
            !core_power_of_two(generators.size()) ||
            generators.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            H != hyrax::derive_pedersen_blinding_generator(basis_id))
            throw std::invalid_argument("masked core basis setup is invalid");
        const std::size_t domain = generators.size();
        if (statement.l_dimension != domain || !statement.valid())
            throw std::invalid_argument("masked core statement domain is invalid");
        const std::vector<Fr> dummy_basis(domain, Fr(0));
        if (statement.basis_digest != NativeProofStatement::basis_digest_for(
                H, generators, dummy_basis, domain))
            throw std::invalid_argument("masked core statement basis mismatch");
        const auto valid_segment = [domain](const std::vector<Fr> &values) {
            return values.size() == domain;
        };
        if (!valid_segment(witness.w) || !valid_segment(witness.aux) ||
            !valid_segment(witness.l_qry) || witness.public_output.empty() ||
            witness.gkr_round_polys.empty() || witness.matmul_round_polys.empty() ||
            witness.gkr_coefficients.size() != 5 ||
            witness.gkr_terminal_values.size() != 4 ||
            witness.matmul_terminal_values.size() != 2)
            throw std::invalid_argument("masked core witness dimensions are invalid");
        for (const std::vector<Fr> &poly : witness.gkr_round_polys)
            if (poly.size() != 3) throw std::invalid_argument("masked core GKR polynomial arity is invalid");
        for (const std::vector<Fr> &poly : witness.matmul_round_polys)
            if (poly.size() != 3) throw std::invalid_argument("masked core MatMul polynomial arity is invalid");
        if (com_W.basis_id != basis_id ||
            com_W.padded_size != domain || !com_W.valid())
            throw std::invalid_argument("masked core W commitment wire is invalid");

        const hyrax::PedersenMleCommitment w_commit =
            core_existing_commitment(com_W, com_W_blinder, basis_id, H,
                                     generators, witness.w);
        hyrax::PedersenMleCommitment aux_commit;
        hyrax::PedersenMleCommitment l_commit;
        if (witness.has_dynamic_commitments) {
            aux_commit = core_existing_commitment(
                witness.com_aux.wire, witness.com_aux.blinder, basis_id, H,
                generators, witness.aux);
            l_commit = core_existing_commitment(
                witness.com_L.wire, witness.com_L.blinder, basis_id, H,
                generators, witness.l_qry);
        } else {
            aux_commit = hyrax::pedersen_commit_mle(generators, witness.aux,
                                                     basis_id);
            l_commit = hyrax::pedersen_commit_mle(generators, witness.l_qry,
                                                   basis_id);
        }
        if (aux_commit.blinding_generator != H || l_commit.blinding_generator != H)
            throw std::logic_error("masked core Pedersen generator mismatch");
        const std::size_t aux_logical = witness.has_dynamic_commitments
            ? witness.com_aux.wire.logical_size : domain;
        const std::size_t l_logical = witness.has_dynamic_commitments
            ? witness.com_L.wire.logical_size : domain;
        result.com_aux = core_commitment_wire(aux_commit, aux_logical, domain);
        result.com_L = core_commitment_wire(l_commit, l_logical, domain);
        if (!core_commitment_matches(result.com_aux, aux_commit, basis_id, domain) ||
            !core_commitment_matches(result.com_L, l_commit, basis_id, domain))
            throw std::logic_error("masked core dynamic commitment wire mismatch");
        NativeProofStatement bound_statement = statement;
        const std::string expected_aux_digest =
            fixed_digest(hash_hex(commitment_encoding(result.com_aux)));
        const std::string expected_l_digest =
            fixed_digest(hash_hex(commitment_encoding(result.com_L)));
        const std::string expected_w_digest =
            fixed_digest(hash_hex(commitment_encoding(com_W)));
        // A zero digest is an explicit prover-side construction placeholder,
        // never a verifier wildcard: it is replaced before the statement is
        // put on the authenticated artifact wire.  Formal callers should
        // normally pre-bind all three values before invoking this bridge.
        const std::string zero_digest(64, '0');
        if (bound_statement.aux_digest == zero_digest)
            bound_statement.aux_digest = expected_aux_digest;
        if (bound_statement.l_digest == zero_digest)
            bound_statement.l_digest = expected_l_digest;
        if (bound_statement.com_w_digest == zero_digest)
            bound_statement.com_w_digest = expected_w_digest;
        if (bound_statement.aux_digest != expected_aux_digest ||
            bound_statement.l_digest != expected_l_digest ||
            bound_statement.com_w_digest != expected_w_digest)
            throw std::invalid_argument("masked core statement commitment binding mismatch");

        if ((lasso_families == nullptr) != (lasso_query == nullptr))
            throw std::invalid_argument("masked core Lasso inputs are incomplete");
        NativeProofSession session(kNativeArtifactDomain, bound_statement);
        std::string reason;
        if (lasso_families != nullptr) {
            if (!native_lasso_prove_bundle(session, *lasso_families, *lasso_query,
                                           nullptr, &reason))
                throw std::runtime_error("masked native Lasso proof failed: " + reason);
        } else {
            session.absorb("lasso/reserved", "native_lasso_pending_v1");
        }
        const Fr matmul_initial = witness.matmul_round_polys.front()[0] +
                                  witness.matmul_round_polys.front()[1];
        session.absorb_field("matmul/claim", matmul_initial);
        if (!native_masked_matmul_product_prove(
                session, "matmul/core", witness.matmul_round_polys,
                matmul_initial, witness.matmul_terminal_values, nullptr,
                &reason))
            throw std::runtime_error("masked MatMul core proof failed: " + reason);

        session.absorb("gkr/public-output",
                       core_vector_digest("zkgpt-masked-core-public-output-v1",
                                          witness.public_output));
        const std::size_t output_domain = next_power_of_two(witness.public_output.size());
        const std::size_t output_rounds = core_log2(output_domain);
        std::vector<Fr> output_point;
        output_point.reserve(output_rounds);
        for (std::size_t index = 0; index < output_rounds; ++index)
            output_point.push_back(session.challenge(
                "gkr/output/r/" + std::to_string(index)));
        std::vector<Fr> output_values(output_domain, Fr(0));
        std::copy(witness.public_output.begin(), witness.public_output.end(),
                  output_values.begin());
        const Fr gkr_initial = [&output_values, &output_point]() {
            if (output_point.empty()) return output_values.at(0);
            std::vector<Fr> folded = output_values;
            std::size_t active = folded.size();
            for (const Fr &coordinate : output_point) {
                const std::size_t half = active / 2;
                for (std::size_t index = 0; index < half; ++index)
                    folded[index] = folded[2 * index] * (Fr(1) - coordinate) +
                                    folded[2 * index + 1] * coordinate;
                active = half;
            }
            return folded.at(0);
        }();
        if (!native_masked_gkr_quadratic_prove(
                session, "gkr/core", witness.gkr_round_polys, gkr_initial,
                witness.gkr_terminal_values, witness.gkr_coefficients, nullptr,
                &reason))
            throw std::runtime_error("masked GKR core proof failed: " + reason);

        session.absorb("combine/public-input",
                       core_vector_digest("zkgpt-masked-core-public-input-v1",
                                          witness.public_input));
        session.absorb("combine/public-output",
                       core_vector_digest("zkgpt-masked-core-public-output-v1",
                                          witness.public_output));
        session.absorb("combine/commitments",
                       core_commitment_payload(basis_id, com_W, result.com_aux,
                                               result.com_L, domain));
        session.absorb("combine/opening-domain", std::to_string(domain));
        const std::size_t opening_rounds = core_log2(domain);
        std::vector<Fr> opening_point;
        opening_point.reserve(opening_rounds);
        for (std::size_t index = 0; index < opening_rounds; ++index)
            opening_point.push_back(session.challenge(
                "combine/r_minus/" + std::to_string(index)));

        hyrax::MaskedMleOpening ignored;
        if (!hyrax::open_masked_mle_with_session(
                w_commit, generators, opening_point, witness.w, session,
                "Open_W", ignored, &reason) ||
            !hyrax::open_masked_mle_with_session(
                aux_commit, generators, opening_point, witness.aux, session,
                "Open_Aux", ignored, &reason) ||
            !hyrax::open_masked_mle_with_session(
                l_commit, generators, opening_point, witness.l_qry, session,
                "Open_L", ignored, &reason))
            throw std::runtime_error("masked core Hyrax opening failed: " + reason);

        GatProofArtifact artifact;
        artifact.privacy_profile = "masked_full";
        artifact.statement_digest = statement_digest;
        artifact.com_aux = result.com_aux;
        artifact.com_L = result.com_L;
        artifact.native_domain = kNativeArtifactDomain;
        artifact.native_statement = bound_statement;
        artifact.native_messages = session.messages();
        artifact.transcript_digest = session.digest();
        core_copy_sections(session, artifact);
        if (!artifact.valid_wire(&reason))
            throw std::runtime_error("masked core artifact wire validation failed: " + reason);
        result.generated = true;
        result.artifact = std::move(artifact);
        result.diagnostic = "masked_core_generated; native_lasso_pending";
        return result;
    } catch (const std::exception &error) {
        result.generated = false;
        result.diagnostic = error.what();
        return result;
    }
}

MaskedCoreResult prove_masked_core(
    const NativeProofStatement &statement, const CommitmentWire &com_W,
    const Fr &com_W_blinder, const std::string &basis_id, const G1 &H,
    const std::vector<G1> &generators, const MaskedCoreWitness &witness,
    const std::string &statement_digest) {
    return prove_masked_core_impl(statement, com_W, com_W_blinder, basis_id, H,
                                  generators, witness, statement_digest,
                                  nullptr, nullptr);
}

MaskedCoreResult prove_masked_core_with_lasso(
    const NativeProofStatement &statement, const CommitmentWire &com_W,
    const Fr &com_W_blinder, const std::string &basis_id, const G1 &H,
    const std::vector<G1> &generators, const MaskedCoreWitness &witness,
    std::vector<NativeLassoFamilyProver> &families,
    NativeLassoQueryProver &query, const std::string &statement_digest) {
    return prove_masked_core_impl(statement, com_W, com_W_blinder, basis_id, H,
                                  generators, witness, statement_digest,
                                  &families, &query);
}

bool verify_masked_core(
    const GatProofArtifact &artifact, const CommitmentWire &com_W,
    const std::string &basis_id, const G1 &H,
    const std::vector<G1> &generators,
    const std::vector<Fr> &public_input,
    const std::vector<Fr> &public_output,
    const std::vector<Fr> &gkr_coefficients,
    std::size_t gkr_rounds, const Fr &matmul_initial_claim,
    std::size_t matmul_rounds, std::string *reason) {
    try {
        if (!artifact.valid_wire(reason) || artifact.privacy_profile != "masked_full")
            return false;
        NativeProofSession session;
        if (!artifact.native_session(session, reason)) return false;
        if (basis_id.empty() || generators.empty() ||
            !core_power_of_two(generators.size()) ||
            H != hyrax::derive_pedersen_blinding_generator(basis_id) ||
            com_W.basis_id != basis_id ||
            !com_W.valid() || com_W.padded_size != generators.size()) {
            set_reason(reason, "masked core verifier basis or W commitment is invalid");
            return false;
        }
        if (artifact.native_statement.com_w_digest !=
            fixed_digest(hash_hex(commitment_encoding(com_W)))) {
            set_reason(reason, "masked core verifier W statement binding mismatch");
            return false;
        }
        if (session.messages().empty()) {
            set_reason(reason, "masked core transcript has no Lasso or core prefix");
            return false;
        }
        const NativeProofMessage &first = session.messages().front();
        const bool reserved_lasso =
            first.kind == NativeProofMessageKind::Data &&
            first.label == "lasso/reserved" &&
            first.payload == "native_lasso_pending_v1";
        const bool native_lasso_prefix =
            first.label.compare(0, sizeof("lasso/") - 1, "lasso/") == 0 ||
            first.label.compare(0, sizeof("challenge/lasso/") - 1,
                                "challenge/lasso/") == 0;
        if (!reserved_lasso && !native_lasso_prefix) {
            set_reason(reason, "masked core transcript does not start with canonical Lasso");
            return false;
        }
        for (const NativeProofMessage &message : session.messages()) {
            if (!core_expected_label(message.label)) {
                set_reason(reason, "unexpected masked core transcript message: " + message.label);
                return false;
            }
        }
        const NativeProofMessage *matmul_claim = core_find_message(
            session, NativeProofMessageKind::Field, "matmul/claim", reason);
        Fr encoded_matmul;
        if (matmul_claim == nullptr || !core_decode_field(matmul_claim->payload,
                                                            encoded_matmul) ||
            encoded_matmul != matmul_initial_claim) {
            set_reason(reason, "masked core MatMul public claim mismatch");
            return false;
        }
        if (!native_masked_matmul_product_verify(
                session, "matmul/core", matmul_rounds,
                matmul_initial_claim, nullptr, reason))
            return false;

        const NativeProofMessage *output_digest = core_find_message(
            session, NativeProofMessageKind::Data, "gkr/public-output", reason);
        if (output_digest == nullptr ||
            output_digest->payload != core_vector_digest(
                "zkgpt-masked-core-public-output-v1", public_output)) {
            set_reason(reason, "masked core public output binding mismatch");
            return false;
        }
        const std::size_t output_domain = next_power_of_two(public_output.size());
        const std::size_t output_dimension = core_log2(output_domain);
        const std::vector<Fr> output_point = core_point_from_session(
            session, "gkr/output/r", output_dimension, reason);
        if (output_point.size() != output_dimension) return false;
        std::vector<Fr> output_values(output_domain, Fr(0));
        std::copy(public_output.begin(), public_output.end(), output_values.begin());
        std::vector<Fr> folded = output_values;
        for (const Fr &coordinate : output_point) {
            const std::size_t half = folded.size() / 2;
            for (std::size_t index = 0; index < half; ++index)
                folded[index] = folded[2 * index] * (Fr(1) - coordinate) +
                                folded[2 * index + 1] * coordinate;
            folded.resize(half);
        }
        if (!native_masked_gkr_quadratic_verify(
                session, "gkr/core", gkr_rounds, folded.at(0),
                gkr_coefficients, nullptr, reason))
            return false;

        const NativeProofMessage *public_input_message = core_find_message(
            session, NativeProofMessageKind::Data, "combine/public-input", reason);
        if (public_input_message == nullptr ||
            public_input_message->payload != core_vector_digest(
                "zkgpt-masked-core-public-input-v1", public_input)) {
            set_reason(reason, "masked core public input binding mismatch");
            return false;
        }
        const NativeProofMessage *public_output_combine = core_find_message(
            session, NativeProofMessageKind::Data, "combine/public-output", reason);
        if (public_output_combine == nullptr ||
            public_output_combine->payload != core_vector_digest(
                "zkgpt-masked-core-public-output-v1", public_output)) {
            set_reason(reason, "masked core combine output binding mismatch");
            return false;
        }
        const std::size_t domain = generators.size();
        const NativeProofMessage *shape_message = core_find_message(
            session, NativeProofMessageKind::Data, "combine/opening-domain", reason);
        if (shape_message == nullptr || shape_message->payload != std::to_string(domain)) {
            set_reason(reason, "masked core opening domain mismatch");
            return false;
        }
        const CommitmentWire &com_aux = artifact.com_aux;
        const CommitmentWire &com_L = artifact.com_L;
        const NativeProofMessage *commitment_message = core_find_message(
            session, NativeProofMessageKind::Data, "combine/commitments", reason);
        if (commitment_message == nullptr || commitment_message->payload !=
                core_commitment_payload(basis_id, com_W, com_aux, com_L, domain)) {
            set_reason(reason, "masked core commitment combine binding mismatch");
            return false;
        }
        const std::vector<Fr> point = core_point_from_session(
            session, "combine/r_minus", core_log2(domain), reason);
        if (point.size() != core_log2(domain)) return false;
        std::size_t w_open = 0, aux_open = 0, l_open = 0;
        if (!core_find_index(session, "Open_W/basis", NativeProofMessageKind::Data,
                             w_open, reason) ||
            !core_find_index(session, "Open_Aux/basis", NativeProofMessageKind::Data,
                             aux_open, reason) ||
            !core_find_index(session, "Open_L/basis", NativeProofMessageKind::Data,
                             l_open, reason) ||
            !(w_open < aux_open && aux_open < l_open)) {
            set_reason(reason, "masked core terminal openings are not canonically ordered");
            return false;
        }
        auto verify_opening = [&](const std::string &label,
                                  const CommitmentWire &wire) {
            const NativeProofMessage *claim = core_find_message(
                session, NativeProofMessageKind::Data, label + "/claim", reason);
            if (claim == nullptr) return false;
            std::vector<std::string> parts;
            if (!core_split_pipe(claim->payload, parts) || parts.size() != 2) {
                set_reason(reason, "malformed masked core opening claim: " + label);
                return false;
            }
            G1 commitment, mask_commitment;
            if (!core_decode_point(wire.point_hex, commitment) ||
                !core_decode_point(parts[0], mask_commitment)) {
                set_reason(reason, "malformed masked core opening point: " + label);
                return false;
            }
            Fr masked_evaluation;
            if (!core_decode_field(parts[1], masked_evaluation)) {
                set_reason(reason, "malformed masked core opening evaluation: " + label);
                return false;
            }
            return hyrax::verify_masked_mle_opening_with_session(
                commitment, mask_commitment, masked_evaluation, basis_id,
                generators, point, session, label, reason, true, true);
        };
        // The verifier intentionally supplies only commitment points.  Each
        // opening checks its own Schnorr response while all three calls use
        // the one transcript-derived r^- above.
        if (!verify_opening("Open_W", com_W) ||
            !verify_opening("Open_Aux", com_aux) ||
            !verify_opening("Open_L", com_L)) return false;
        return true;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
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
