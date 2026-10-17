#include "gat_proof.hpp"
#include "gat_formal_internal.hpp"

#include "hyrax_util.hpp"
#include "native_lasso.hpp"
#include "utils.hpp"
#include "verifier.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cybozu/sha2.hpp>
#include <iomanip>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <streambuf>
#include <stdexcept>
#include <set>
#include <tuple>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

mcl::bn::G1 perdersen_commit(mcl::bn::G1*, mcl::bn::Fr*, int);

namespace zkgpt {
namespace gat {
using zkgpt::NativeFieldVector;
namespace {

using namespace mcl::bn;

std::once_flag field_once;

const std::size_t kArtifactMaxString = 1ULL << 26;
const std::size_t kArtifactMaxMessages = 1ULL << 20;
const std::uint64_t kArtifactMaxWireBytes = 1ULL << 30;
const std::uint64_t kArtifactHardMaxWireBytes = 4ULL << 30;
const std::uint64_t kArtifactRowPointBytesPerCopy = 284;
const std::uint64_t kArtifactRowHeaderBytesPerFamily = 64ULL << 10;
const std::uint64_t kArtifactRowChunkBytesPerCopy = 2048;

const std::uint64_t kContextMaxWireBytes = 2ULL << 30;
const char kCanonicalGeneratorDerivationId[] = "hashAndMapToG1(canonical_basis_label+/coordinate/+canonical_absolute_index_decimal);absolute-index-u64;v1";
const std::uint64_t kContextMaxElements = 1ULL << 30;
const std::uint64_t kContextMaxLayers = 1ULL << 20;
const std::uint64_t kContextMaxDescriptors = 1ULL << 20;
const std::uint64_t kContextMaxRecipeBytes = 64ULL << 20;

const std::uint64_t kContextMaxShapeProfile = 128;
const std::uint64_t kContextMaxShapeDimensions = 16;
const char kNativeArtifactDomain[] = "zkgpt-native-gat-fs-v7";
const char kVerifierContextMagic[] = "ZKGPTVC5";
constexpr std::uint32_t kVerifierContextVersion = 5;
const char kVerifierContextSchema[] = "zkgpt-gat-verifier-context-v5";
const char kProofArtifactMagic[] = "ZKGPTGAT8";

void core_copy_sections(const NativeProofSession& session, GatProofArtifact& artifact);
bool core_split_pipe(const std::string& payload, std::vector<std::string>& parts);
bool core_decode_field(const std::string& encoded, Fr& value);
bool core_decode_point(const std::string& encoded, G1& point);
bool core_power_of_two(std::size_t value);
std::size_t core_log2(std::size_t value);
const NativeProofMessage* core_find_message(const NativeProofSession& session, NativeProofMessageKind kind, const std::string& label, std::string* reason);
bool is_lower_hex_digest(const std::string& value);

std::string plan_encoding(const GatCircuitPlan& plan);
std::string canonical_graph_topology_digest(const Graph& graph);
CanonicalModelShape canonical_model_shape(const Model& model);
Model make_shape_only_model(const CanonicalModelShape& shape);
NativeFieldVector flatten_aux_values(const ForwardResult& forward);
Wide abs_wide_adapter(Int value);
void append_int_vector(std::string& out, const std::vector<Int>& values);
bool read_compact_plan_recipe(class ContextReader& reader, GatCircuitPlan& plan);
bool materialize_compact_plan(const Graph& graph, const QuantConfig& quant, const GatCircuitPlan& recipe, GatCircuitPlan& materialized, std::string* reason);
bool context_layer_metadata_valid(const GatCircuitPlan& plan, std::string* reason);

void ensure_field() {
    std::call_once(field_once, []() { mcl::bn::initPairing(mcl::BN254); });
}

void append_u64(std::string& out, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) { out.push_back(static_cast<char>((value >> (i * 8)) & 0xffU)); }
}

void append_u32(std::string& out, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) { out.push_back(static_cast<char>((value >> (i * 8)) & 0xffU)); }
}

void append_string(std::string& out, const std::string& value) {
    append_u64(out, static_cast<std::uint64_t>(value.size()));
    out.append(value);
}

void append_int(std::string& out, Int value) {
    const bool negative = value < 0;
    const Wide absolute = negative ? -static_cast<Wide>(value) : static_cast<Wide>(value);
    out.push_back(static_cast<char>(negative ? 1 : 0));
    append_u64(out, static_cast<std::uint64_t>(absolute));
}

void append_shape(std::string& out, const Shape& shape) {
    append_u64(out, static_cast<std::uint64_t>(shape.dims.size()));
    for (std::size_t dimension : shape.dims) append_u64(out, static_cast<std::uint64_t>(dimension));
}

std::string field_hex(const Field& value) {
    std::string result = value.getStr(16);
    if (result.empty()) result = "0";
    return result;
}

std::string hash_hex(const std::string& bytes) {
    ensure_field();
    Field value;
    value.setHashOf(bytes.data(), bytes.size());
    return field_hex(value);
}

std::string fixed_digest(std::string value) {
    if (value.empty() || value.size() > 64) throw std::invalid_argument("digest is outside the fixed-width field");
    for (char& character : value) {
        if (character >= 'A' && character <= 'F') character = static_cast<char>(character - 'A' + 'a');
        if (!((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f')))
            throw std::invalid_argument("digest is not lowercase hexadecimal");
    }
    return std::string(64 - value.size(), '0') + value;
}

std::string normalize_digest(const std::string& value) {
    if (value.size() == 64) {
        try {
            return fixed_digest(value);
        } catch (...) {}
    }
    return fixed_digest(hash_hex(value));
}

std::string point_hex(const G1& point) { return point.serializeToHexStr(); }

std::size_t next_power_of_two(std::size_t value) {
    std::size_t result = 1;
    while (result < std::max<std::size_t>(1, value)) {
        if (result > std::numeric_limits<std::size_t>::max() / 2) throw std::overflow_error("zkGPT input layout overflow");
        result <<= 1;
    }
    return result;
}

hyrax::HyraxMleBasisView canonical_input_opening_view(
    const std::string &basis_label, const std::string &basis_id,
    const std::string &canonical_id, const std::string &role,
    std::uint64_t full_domain, std::uint64_t logical_length,
    const std::vector<Fr> &point) {
    if (basis_label.empty() || basis_id.empty() || canonical_id.empty() ||
        role.empty() || full_domain == 0 ||
        (full_domain & (full_domain - 1)) != 0 || logical_length == 0 ||
        logical_length > full_domain || point.size() >= 64 ||
        (std::uint64_t(1) << point.size()) != full_domain)
        throw std::invalid_argument("canonical masked input opening view is invalid");
    hyrax::HyraxMleBasisView view;
    view.full_domain = full_domain;
    view.logical_length = logical_length;
    view.active_domain = next_power_of_two(static_cast<std::size_t>(logical_length));
    view.role = role;
    view.basis_id = basis_id;
    view.point = point;
    view.generators.size = full_domain;
    view.generators.canonical_id = canonical_id;
    view.generators.scheme = hyrax::HyraxGeneratorProviderScheme::CanonicalAbsoluteHashToCurveV1;
    view.generators.at = [basis_label, full_domain](std::uint64_t index) {
        if (index >= full_domain)
            throw std::out_of_range("canonical masked input generator index is invalid");
        G1 generator;
        hashAndMapToG1(generator,
                       basis_label + "/coordinate/" + std::to_string(index));
        return generator;
    };
    return view;
}

hyrax::HyraxTensorMleView canonical_input_tensor_view(
    const std::string &basis_label, const std::string &basis_id,
    const std::string &role, std::uint64_t full_domain,
    std::uint64_t logical_length) {
    if (basis_label.empty() || basis_id.empty() || role.empty() ||
        full_domain == 0 || full_domain > (std::uint64_t(1) << 62) ||
        (full_domain & (full_domain - 1)) != 0 || logical_length == 0 ||
        logical_length > full_domain)
        throw std::invalid_argument("canonical tensor input view is invalid");

    std::uint64_t active_domain = 1;
    std::size_t active_bits = 0;
    while (active_domain < logical_length) {
        active_domain <<= 1;
        ++active_bits;
    }
    const std::size_t column_bits = active_bits / 2;
    const std::uint64_t columns = std::uint64_t(1) << column_bits;
    const std::uint64_t rows = active_domain / columns;
    const std::string generator_id =
        basis_label + "/tensor/" + role + "/columns-v2";
    if (generator_id.size() > 128)
        throw std::invalid_argument("canonical tensor generator descriptor is too long");

    hyrax::HyraxTensorMleView view;
    view.full_domain = full_domain;
    view.logical_length = logical_length;
    view.active_domain = active_domain;
    view.rows = rows;
    view.columns = columns;
    view.role = role;
    view.basis_id = basis_id;
    view.column_generators.size = columns;
    view.column_generators.canonical_id = generator_id;
    view.column_generators.scheme =
        hyrax::HyraxGeneratorProviderScheme::CanonicalAbsoluteHashToCurveV1;
    view.column_generators.at = [generator_id, columns](std::uint64_t index) {
        if (index >= columns)
            throw std::out_of_range("canonical tensor column generator index is invalid");
        G1 generator;
        hashAndMapToG1(generator, generator_id + "/" + std::to_string(index));
        return generator;
    };
    return view;
}

hyrax::HyraxScalarProvider canonical_input_tensor_scalar_provider(
    const SegmentedInputOracle &values, std::size_t segment,
    std::uint64_t logical_length) {
    if (segment >= 4 || logical_length == 0 ||
        values.layout().segments.at(segment).logical_length != logical_length)
        throw std::invalid_argument("canonical tensor input scalar provider is invalid");
    hyrax::HyraxScalarProvider provider;
    provider.size = logical_length;
    provider.at = [&values, segment, logical_length](std::uint64_t index) {
        if (index >= logical_length)
            throw std::out_of_range("canonical tensor input scalar index is invalid");
        return values.segment_value(segment, static_cast<std::size_t>(index));
    };
    return provider;
}

hyrax::HyraxScalarProvider canonical_input_scalar_provider(
    const NativeFieldVector &values, std::uint64_t full_domain,
    std::uint64_t segment_offset, std::uint64_t logical_length) {
    if (full_domain == 0 || segment_offset > values.size() ||
        logical_length > full_domain ||
        logical_length > values.size() - segment_offset)
        throw std::invalid_argument("canonical masked input scalar provider is invalid");
    hyrax::HyraxScalarProvider provider;
    provider.size = full_domain;
    provider.at = [&values, full_domain, segment_offset, logical_length](
                      std::uint64_t index) {
        if (index >= full_domain)
            throw std::out_of_range("canonical masked input scalar index is invalid");
        if (index >= logical_length) return Fr(0);
        return values.at(static_cast<std::size_t>(segment_offset + index));
    };
    return provider;
}

hyrax::HyraxScalarProvider canonical_input_scalar_provider(
    const SegmentedInputOracle &values, std::size_t segment,
    std::uint64_t full_domain, std::uint64_t logical_length) {
    if (segment >= 4 || full_domain == 0 ||
        logical_length > full_domain ||
        values.layout().n_in != full_domain ||
        values.layout().segments.at(segment).logical_length != logical_length)
        throw std::invalid_argument("canonical segmented input scalar provider is invalid");
    hyrax::HyraxScalarProvider provider;
    provider.size = full_domain;
    provider.at = [&values, segment, full_domain, logical_length](
                      std::uint64_t index) {
        if (index >= full_domain)
            throw std::out_of_range("canonical segmented input scalar index is invalid");
        return index < logical_length
                   ? values.segment_value(segment, static_cast<std::size_t>(index))
                   : Fr(0);
    };
    return provider;
}

NativeGkrInputSupport canonical_gkr_input_support(const GatCircuitPlan& plan) {
    if (plan.input_layout.segments.size() != 4 || plan.input_layout.n_in == 0)
        throw std::invalid_argument("canonical GKR input support needs the four-segment layout");
    NativeGkrInputSupport support;
    for (std::size_t segment = 0; segment < 3; ++segment) {
        const std::uint64_t length = plan.input_layout.segments.at(segment).logical_length;
        if (length != 0) support[segment].push_back({0, length});
    }
    const InputSegment& l_segment = plan.input_layout.segments.at(3);
    for (const LassoFamilyDescriptor& family : plan.lasso) {
        if (family.query_offset < l_segment.offset || family.padded_queries == 0)
            throw std::invalid_argument("canonical L_qry family support has an invalid offset");
        const std::uint64_t begin = family.query_offset - l_segment.offset;
        const std::uint64_t length = family.padded_queries;
        if (begin > l_segment.logical_length || length > l_segment.logical_length - begin)
            throw std::invalid_argument("canonical L_qry family support exceeds its logical extent");
        support[3].push_back({begin, begin + length});
    }
    std::sort(support[3].begin(), support[3].end(), [](const NativeGkrInputSupportInterval& left, const NativeGkrInputSupportInterval& right) {
        return left.begin < right.begin;
    });
    std::uint64_t previous_end = 0;
    for (const NativeGkrInputSupportInterval& interval : support[3]) {
        if (interval.begin < previous_end) throw std::invalid_argument("canonical L_qry family support overlaps");
        previous_end = interval.end;
    }
    return support;
}

std::vector<hyrax::HyraxTensorZeroInterval> canonical_l_query_zero_intervals(
    const GatCircuitPlan& plan) {
    if (plan.input_layout.segments.size() != 4)
        throw std::invalid_argument("canonical L_qry zero intervals need the four-segment layout");
    const std::uint64_t logical_length = plan.input_layout.segments.at(3).logical_length;
    const NativeGkrInputSupport support = canonical_gkr_input_support(plan);
    std::vector<hyrax::HyraxTensorZeroInterval> intervals;
    std::uint64_t cursor = 0;
    for (const NativeGkrInputSupportInterval& live : support[3]) {
        if (cursor < live.begin) intervals.push_back({cursor, live.begin});
        cursor = live.end;
    }
    if (cursor < logical_length) intervals.push_back({cursor, logical_length});
    return intervals;
}

std::size_t checked_add(std::size_t left, std::size_t right, const char* what) {
    if (right > std::numeric_limits<std::size_t>::max() - left) throw std::overflow_error(std::string("zkGPT size overflow: ") + what);
    return left + right;
}

std::size_t checked_mul(std::size_t left, std::size_t right, const char* what) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) throw std::overflow_error(std::string("zkGPT size overflow: ") + what);
    return left * right;
}

SparseGkrRelationKind sparse_relation_kind_for_id(const std::string& relation_id) {
    if (relation_id.find("attention.score/") == 0) return SparseGkrRelationKind::AttentionScore;
    if (relation_id.find("attention.softmax/") == 0) return SparseGkrRelationKind::MaskedSoftmax;
    if (relation_id.find("aggregation.") == 0) return SparseGkrRelationKind::Aggregation;
    if (relation_id.find("rounding/") == 0 || relation_id.find("signed_round_div") != std::string::npos) return SparseGkrRelationKind::SignedRoundDiv;
    if (relation_id.find("fconn") != std::string::npos || relation_id.find("projection.") == 0) return SparseGkrRelationKind::Fconn;
    if (relation_id.find("input") != std::string::npos) return SparseGkrRelationKind::InputBinding;
    if (relation_id.find("output") != std::string::npos) return SparseGkrRelationKind::OutputBinding;
    return SparseGkrRelationKind::Residual;
}

std::uint64_t as_u64(std::size_t value, const char* what) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (value > std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error(std::string(what) + " exceeds u64 coordinates");
    }
    return static_cast<std::uint64_t>(value);
}

char bit_length(std::size_t value) {
    std::size_t n = std::max<std::size_t>(1, value);
    char bits = 0;
    std::size_t domain = 1;
    while (domain < n) {
        if (domain > (std::numeric_limits<std::size_t>::max() >> 1)) throw std::overflow_error("zkGPT layer MLE domain overflow");
        domain <<= 1;
        ++bits;
    }
    return bits;
}

std::size_t fconn_axis_domain(std::size_t logical_extent, const char* what) {
    if (logical_extent == 0) throw std::invalid_argument(std::string(what) + " is zero");
    std::size_t domain = 1;
    while (domain < logical_extent) {
        if (domain > (std::numeric_limits<std::size_t>::max() >> 1))
            throw std::overflow_error(std::string("zkGPT FCONN padded axis overflows: ") + what);
        domain <<= 1;
    }
    return domain;
}

std::size_t fconn_output_domain(std::size_t rows, std::size_t cols) {
    return checked_mul(fconn_axis_domain(rows, "rows"),
                       fconn_axis_domain(cols, "columns"),
                       "FCONN padded output domain");
}

std::size_t fconn_padded_output_index(std::size_t rows, std::size_t cols,
                                      std::size_t logical_index) {
    if (logical_index >= checked_mul(rows, cols, "FCONN logical output coordinates"))
        throw std::out_of_range("FCONN logical output coordinate is invalid");
    const std::size_t padded_cols = fconn_axis_domain(cols, "columns");
    const std::size_t row = logical_index / cols;
    const std::size_t col = logical_index % cols;
    return checked_add(checked_mul(row, padded_cols, "FCONN padded output row"),
                       col, "FCONN padded output index");
}

std::string hidden_tensor_name(std::size_t layer, std::size_t head, const char* kind) {
    std::ostringstream out;
    out << "hidden/layer" << layer << "/head" << head << "/" << kind;
    return out.str();
}

std::string output_tensor_name(std::size_t head, const char* kind) {
    std::ostringstream out;
    out << "output/head" << head << "/" << kind;
    return out.str();
}

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
    if (rounded > static_cast<long double>(std::numeric_limits<Int>::max()) || rounded < static_cast<long double>(std::numeric_limits<Int>::min()))
        throw std::overflow_error("model quantization overflow");
    return static_cast<Int>(rounded);
}

std::vector<Int> flatten_model(const Model& model, const QuantConfig& quant) {
    model.validate_schema();
    quant.validate();
    std::vector<Int> result;
    const auto append_tensor = [&result](const Tensor& tensor, Int scale) {
        for (double value : tensor.values) result.push_back(quantize_model_value(value, scale));
    };
    for (std::size_t layer = 0; layer < 2; ++layer) {
        for (std::size_t head = 0; head < model.hidden_heads(); ++head) {
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "seq/kernel")), quant.s_w_hidden[layer][head]);
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "attn_src/kernel")), quant.s_a_src_hidden[layer][head]);
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "attn_dst/kernel")), quant.s_a_dst_hidden[layer][head]);
        }
    }
    for (std::size_t head = 0; head < model.manifest.output_heads; ++head) {
        append_tensor(model.tensor(output_tensor_name(head, "seq/kernel")), quant.s_w_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "attn_src/kernel")), quant.s_a_src_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "attn_dst/kernel")), quant.s_a_dst_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "output_bias")), quant.s_b_output[head]);
    }
    return result;
}

std::vector<std::string> canonical_model_tensor_names(std::size_t hidden_heads, std::size_t output_heads) {
    std::vector<std::string> names;
    names.reserve(2 * hidden_heads * 3 + output_heads * 4);
    for (std::size_t layer = 0; layer < 2; ++layer) {
        for (std::size_t head = 0; head < hidden_heads; ++head) {
            names.push_back(hidden_tensor_name(layer, head, "seq/kernel"));
            names.push_back(hidden_tensor_name(layer, head, "attn_src/kernel"));
            names.push_back(hidden_tensor_name(layer, head, "attn_dst/kernel"));
        }
    }
    for (std::size_t head = 0; head < output_heads; ++head) {
        names.push_back(output_tensor_name(head, "seq/kernel"));
        names.push_back(output_tensor_name(head, "attn_src/kernel"));
        names.push_back(output_tensor_name(head, "attn_dst/kernel"));
        names.push_back(output_tensor_name(head, "output_bias"));
    }
    return names;
}

CanonicalModelShape canonical_model_shape(const Model& model) {
    model.validate_schema();
    CanonicalModelShape result;
    result.model_arch_id = model.manifest.model_arch_id;
    result.tensor_count = model.tensors.size();
    result.layers = model.manifest.layers;
    result.output_heads = model.manifest.output_heads;
    result.classes = model.classes();
    result.input_dim = model.input_dim();
    result.hidden_dim = model.hidden_dim();
    const std::vector<std::string> names = canonical_model_tensor_names(model.hidden_heads(), model.manifest.output_heads);
    result.tensors.reserve(names.size());
    for (const std::string& name : names) result.tensors.push_back({name, model.tensor(name).shape});
    return result;
}

Model make_shape_only_model(const CanonicalModelShape& shape) {
    Model model;
    model.manifest.dataset = "verifier-reconstructed";
    model.manifest.model_arch_id = shape.model_arch_id;
    model.manifest.quant_cfg_id = "gat_l3_scale_chain";
    model.manifest.tensor_count = shape.tensor_count;
    model.manifest.layers = shape.layers;
    model.manifest.output_heads = shape.output_heads;
    model.manifest.classes = shape.classes;
    model.manifest.model_param_id = "public-shape-only";
    model.manifest.static_table_id = "tables:lrelu+elu+exp+range";
    model.manifest.input_preprocess_id = "canonical_public_features";
    model.manifest.batching_rule = "single_graph_or_ppi_offsets";
    model.manifest.edge_sort_rule = "edge_gid_then_dst_stable";
    model.manifest.self_loop_rule = "per_node";
    model.manifest.label_format = "single_label";
    model.manifest.output_average_rule = "per_head_bias_then_arithmetic_mean";
    model.manifest.quantization_scale_chain = "gat_l3_scale_chain";
    const std::size_t hidden_tensor_count = shape.tensor_count - shape.output_heads * 4;
    const std::size_t hidden_heads = hidden_tensor_count / ((shape.layers - 1) * 3);
    for (const CanonicalTensorShape& entry : shape.tensors) {
        Tensor tensor;
        tensor.name = entry.name;
        tensor.shape = entry.shape;
        tensor.values.assign(tensor.shape.elements(), 0.0);
        model.tensors.emplace(tensor.name, std::move(tensor));
    }
    model.validate_schema();
    return model;
}

bool canonical_model_shape_valid(const CanonicalModelShape& shape, std::string* reason = nullptr) {
    const auto fail = [reason](const std::string& message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    try {
        if (shape.model_arch_id.empty() || shape.tensor_count < 10 || shape.layers != 3 || shape.output_heads == 0 || shape.classes == 0 ||
            shape.input_dim == 0 || shape.hidden_dim == 0 || shape.tensors.size() != shape.tensor_count) {
            return fail("canonical model shape profile header is invalid");
        }
        if (shape.tensor_count < shape.output_heads * 4 || (shape.tensor_count - shape.output_heads * 4) % ((shape.layers - 1) * 3) != 0) {
            return fail("canonical model shape head profile is invalid");
        }
        const std::size_t hidden_heads = (shape.tensor_count - shape.output_heads * 4) / ((shape.layers - 1) * 3);
        if (hidden_heads == 0) return fail("canonical model shape has no hidden heads");
        const std::vector<std::string> names = canonical_model_tensor_names(hidden_heads, shape.output_heads);
        if (names.size() != shape.tensors.size()) return fail("canonical model shape profile count is invalid");
        std::size_t total_elements = 0;
        for (std::size_t i = 0; i < shape.tensors.size(); ++i) {
            const CanonicalTensorShape& entry = shape.tensors[i];
            if (entry.name != names[i] || entry.shape.dims.empty() || entry.shape.dims.size() > kContextMaxShapeDimensions) {
                return fail("canonical model shape tensor identity is invalid");
            }
            const std::size_t elements = entry.shape.elements();
            if (elements == 0 || elements > (1ULL << 26) || total_elements > (1ULL << 28) - elements) {
                return fail("canonical model shape tensor bounds are invalid");
            }
            total_elements += elements;
        }

        const Model shape_model = make_shape_only_model(shape);
        if (shape_model.input_dim() != shape.input_dim || shape_model.hidden_dim() != shape.hidden_dim || shape_model.classes() != shape.classes) {
            return fail("canonical model shape summary does not match tensor profile");
        }
        return true;
    } catch (const std::exception& error) { return fail(error.what()); }
}

std::string canonical_graph_topology_digest(const Graph& graph) {
    std::string encoded;
    append_string(encoded, "zkgpt-gat-canonical-graph-topology-v1");
    append_u64(encoded, graph.num_nodes);
    append_u64(encoded, graph.num_features);
    append_u64(encoded, graph.graph_count);
    append_u64(encoded, graph.num_classes);
    append_int_vector(encoded, graph.src);
    append_int_vector(encoded, graph.dst);
    append_int_vector(encoded, graph.edge_gid);
    append_u64(encoded, graph.node_ptr.size());
    for (std::size_t value : graph.node_ptr) append_u64(encoded, value);
    append_u64(encoded, graph.edge_ptr.size());
    for (std::size_t value : graph.edge_ptr) append_u64(encoded, value);
    return canonical_digest64(encoded);
}

const char* canonical_selector() { return "00:X_pub,01:W,10:Aux,11:L_qry"; }

const char* canonical_opening_policy() { return "same_r_three_openings_v1"; }

void append_double_bits(std::string& out, double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected double width");
    std::memcpy(&bits, &value, sizeof(bits));
    append_u64(out, bits);
}

void append_int_vector(std::string& out, const std::vector<Int>& values) {
    append_u64(out, static_cast<std::uint64_t>(values.size()));
    for (Int value : values) append_int(out, value);
}

void append_manifest_encoding(std::string& out, const ModelManifest& manifest) {
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

void append_quant_encoding(std::string& out, const QuantConfig& quant) {
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

void append_layout_encoding(std::string& out, const InputLayout& layout) {
    append_string(out, "zkgpt-gat-four-segment-layout-v1");
    append_u64(out, static_cast<std::uint64_t>(layout.n_raw));
    append_u64(out, static_cast<std::uint64_t>(layout.n_in));
    append_string(out, layout.selector);
    append_u64(out, static_cast<std::uint64_t>(layout.segments.size()));
    for (const InputSegment& segment : layout.segments) {
        append_string(out, segment.name);
        append_u64(out, static_cast<std::uint64_t>(segment.offset));
        append_u64(out, static_cast<std::uint64_t>(segment.logical_length));
        append_u64(out, static_cast<std::uint64_t>(segment.padded_length));
    }
}

bool canonical_layout_valid(const InputLayout& layout) {
    if (layout.n_in == 0 || layout.selector != canonical_selector() || layout.segments.size() != 4) return false;
    const char* names[] = {"X_pub", "W", "Aux", "L_qry"};
    std::size_t max_logical = 1;
    for (std::size_t index = 0; index < layout.segments.size(); ++index) {
        const InputSegment& segment = layout.segments[index];
        if (segment.name != names[index] || segment.padded_length != layout.n_in || segment.logical_length > layout.n_in ||
            segment.offset != checked_mul(layout.n_in, index, "canonical segment offset"))
            return false;
        max_logical = std::max(max_logical, segment.logical_length);
    }
    if (layout.n_raw != max_logical || next_power_of_two(layout.n_raw) != layout.n_in) return false;
    return true;
}

std::string canonical_manifest_digest(const Model& model) {
    std::string encoded;
    append_manifest_encoding(encoded, model.manifest);
    return fixed_digest(hash_hex(encoded));
}

void append_canonical_model_encoding(std::string& out, const Model& model, const QuantConfig& quant, bool include_values) {
    append_string(out, "zkgpt-gat-canonical-model-v1");
    append_u64(out, static_cast<std::uint64_t>(model.tensors.size()));
    const auto append_tensor = [&out, include_values](const Tensor& tensor, Int scale) {
        append_string(out, tensor.name);
        append_shape(out, tensor.shape);
        append_int(out, scale);
        append_u64(out, static_cast<std::uint64_t>(tensor.values.size()));
        if (include_values) {
            for (double value : tensor.values) append_double_bits(out, value);
        }
    };
    for (std::size_t layer = 0; layer < 2; ++layer) {
        for (std::size_t head = 0; head < model.hidden_heads(); ++head) {
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "seq/kernel")), quant.s_w_hidden[layer][head]);
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "attn_src/kernel")), quant.s_a_src_hidden[layer][head]);
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "attn_dst/kernel")), quant.s_a_dst_hidden[layer][head]);
        }
    }
    for (std::size_t head = 0; head < model.manifest.output_heads; ++head) {
        append_tensor(model.tensor(output_tensor_name(head, "seq/kernel")), quant.s_w_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "attn_src/kernel")), quant.s_a_src_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "attn_dst/kernel")), quant.s_a_dst_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "output_bias")), quant.s_b_output[head]);
    }
}

std::string canonical_model_digest(const Model& model, const QuantConfig& quant) {
    std::string encoded;
    append_canonical_model_encoding(encoded, model, quant, true);
    return fixed_digest(hash_hex(encoded));
}

std::string canonical_bounds_digest(const Model& model, const QuantConfig& quant) {
    std::string encoded;
    append_string(encoded, "zkgpt-gat-per-tensor-bounds-v1");
    const auto append_tensor = [&encoded](const Tensor& tensor, Int scale) {
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
        for (std::size_t head = 0; head < model.hidden_heads(); ++head) {
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "seq/kernel")), quant.s_w_hidden[layer][head]);
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "attn_src/kernel")), quant.s_a_src_hidden[layer][head]);
            append_tensor(model.tensor(hidden_tensor_name(layer, head, "attn_dst/kernel")), quant.s_a_dst_hidden[layer][head]);
        }
    }
    for (std::size_t head = 0; head < model.manifest.output_heads; ++head) {
        append_tensor(model.tensor(output_tensor_name(head, "seq/kernel")), quant.s_w_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "attn_src/kernel")), quant.s_a_src_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "attn_dst/kernel")), quant.s_a_dst_output[head]);
        append_tensor(model.tensor(output_tensor_name(head, "output_bias")), quant.s_b_output[head]);
    }
    return fixed_digest(hash_hex(encoded));
}

std::string canonical_quant_digest(const QuantConfig& quant) {
    std::string encoded;
    append_quant_encoding(encoded, quant);
    return fixed_digest(hash_hex(encoded));
}

std::string commitment_encoding(const CommitmentWire& commitment);

std::string registration_material_encoding(const Model& model, const QuantConfig& quant, const InputLayout& layout, const std::string& model_arch_id) {
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

std::string registration_basis_label(const Model& model, const QuantConfig& quant, const InputLayout& layout, const std::string& model_arch_id) {
    return "zkGPT/model/v2/" + hash_hex(registration_material_encoding(model, quant, layout, model_arch_id));
}

std::string registration_binding_encoding(const ModelRegistration& registration) {
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

void populate_registration_metadata(ModelRegistration& registration, const Model& model, const QuantConfig& quant, const InputLayout& layout,
                                    const std::string& model_arch_id) {
    ensure_field();
    if (!canonical_layout_valid(layout)) throw std::invalid_argument("model registration requires the canonical four-segment layout");
    if (model_arch_id.empty() || model.manifest.model_arch_id != model_arch_id)
        throw std::invalid_argument("model registration architecture identity mismatch");
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
    registration.layout_digest = canonical_digest64(layout_encoded);
    registration.selector = layout.selector;
    registration.n_raw = layout.n_raw;
    registration.n_in = layout.n_in;
    registration.input_segments = layout.segments;
    registration.opening_policy = canonical_opening_policy();
}

bool registration_metadata_matches(const ModelRegistration& actual, const ModelRegistration& expected) {
    if (actual.input_segments.size() != expected.input_segments.size()) return false;
    for (std::size_t index = 0; index < actual.input_segments.size(); ++index) {
        const InputSegment& left = actual.input_segments[index];
        const InputSegment& right = expected.input_segments[index];
        if (left.name != right.name || left.offset != right.offset || left.logical_length != right.logical_length || left.padded_length != right.padded_length)
            return false;
    }
    return actual.dataset == expected.dataset && actual.model_arch_id == expected.model_arch_id && actual.quant_cfg_id == expected.quant_cfg_id &&
           actual.static_tables_id == expected.static_tables_id && actual.field_id == expected.field_id && actual.field_modulus == expected.field_modulus &&
           actual.manifest_digest == expected.manifest_digest && actual.model_digest == expected.model_digest &&
           actual.bounds_digest == expected.bounds_digest && actual.quant_digest == expected.quant_digest && actual.layout_digest == expected.layout_digest &&
           actual.selector == expected.selector && actual.n_raw == expected.n_raw && actual.n_in == expected.n_in &&
           actual.opening_policy == expected.opening_policy;
}

InputLayout registration_layout(const ModelRegistration& registration) {
    InputLayout layout;
    layout.n_raw = registration.n_raw;
    layout.n_in = registration.n_in;
    layout.selector = registration.selector;
    layout.segments = registration.input_segments;
    return layout;
}

bool registration_wire_valid(const ModelRegistration& registration) {
    ensure_field();
    const InputLayout layout = registration_layout(registration);
    if (!canonical_layout_valid(layout) || registration.status != "registered_long_lived_com_W" || registration.dataset.empty() ||
        registration.model_arch_id.empty() || registration.quant_cfg_id.empty() || registration.static_tables_id.empty() || registration.field_id != "BN254" ||
        registration.field_modulus != Field::getModulo() || registration.manifest_digest.empty() || registration.model_digest.empty() ||
        registration.bounds_digest.empty() || registration.quant_digest.empty() || registration.opening_policy != canonical_opening_policy() ||
        !registration.com_W.valid() || registration.com_W.logical_size != layout.segments.at(1).logical_length || registration.com_W.padded_size != layout.n_in)
        return false;
    std::string layout_encoded;
    append_layout_encoding(layout_encoded, layout);
    if (registration.layout_digest != canonical_digest64(layout_encoded)) return false;
    if (registration.binding_digest != canonical_digest64(registration_binding_encoding(registration))) return false;
    return registration.registration_digest ==
           canonical_digest64("zkgpt-gat-registration-v2/" + registration.binding_digest + "/" + commitment_encoding(registration.com_W));
}

std::size_t attention_aux_size(const AttentionTrace& trace) {
    std::size_t result = 0;
    const auto add = [&result](std::size_t value, const char* what) { result = checked_add(result, value, what); };
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

std::size_t trace_aux_size(const GatTrace* trace) {
    if (trace == nullptr) return 1;
    std::size_t result = 0;
    const auto add = [&result](std::size_t value, const char* what) { result = checked_add(result, value, what); };
    add(trace->input.values.size(), "input trace");
    for (const HiddenLayerTrace& layer : trace->hidden_layers) {
        add(layer.concatenated.values.size(), "concat trace");
        for (const AttentionTrace& head : layer.heads) add(attention_aux_size(head), "hidden attention trace");
    }
    for (const OutputHeadTrace& head : trace->output_heads) {
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


std::size_t canonical_attention_aux_size(const Graph& graph, std::size_t output_dim) {
    const std::size_t nodes = graph.num_nodes;
    const std::size_t edges = graph.edges();
    const std::size_t coordinates = checked_mul(nodes, output_dim, "canonical attention coordinates");
    std::size_t result = 0;
    const auto add = [&result](std::size_t value, const char* what) { result = checked_add(result, value, what); };
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

std::size_t canonical_aux_size(const Graph& graph, const Model& model) {
    const std::size_t hidden0 = model.tensor(hidden_tensor_name(0, 0, "seq/kernel")).shape.dims.at(2);
    const std::size_t hidden1 = model.tensor(hidden_tensor_name(1, 0, "seq/kernel")).shape.dims.at(2);
    const std::size_t classes = model.classes();
    std::size_t result = checked_mul(graph.num_nodes, model.input_dim(), "canonical input trace coordinates");
    for (std::size_t output_dim : {hidden0, hidden1}) {
        result = checked_add(
            result, checked_mul(graph.num_nodes, checked_mul(output_dim, model.hidden_heads(), "canonical concat width"), "canonical concat coordinates"),
            "canonical concat coordinates");
        for (std::size_t head = 0; head < model.hidden_heads(); ++head)
            result = checked_add(result, canonical_attention_aux_size(graph, output_dim), "canonical hidden attention coordinates");
    }
    for (std::size_t head = 0; head < model.manifest.output_heads; ++head) {
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

struct StaticLookupBounds {
    std::pair<Int, Int> lrelu;
    std::pair<Int, Int> exp;
    std::pair<Int, Int> elu;
};

unsigned nonnegative_bit_width(Wide bound) {
    if (bound < 0) throw std::invalid_argument("negative public range bound");
    unsigned bits = 1;
    while (bits < 63 && bound >= (static_cast<Wide>(1) << bits)) ++bits;
    return bits;
}

std::pair<Int, Int> lookup_rectangle(const std::vector<TablePoint>& points);
StaticLookupBounds static_lookup_bounds(const StaticTables& tables);
const std::vector<TablePoint>& table_points_for(const StaticTables& tables, const std::string& family);
const std::pair<Int, Int>& table_bounds_for(const StaticLookupBounds& bounds, const std::string& family);

std::vector<std::array<unsigned, 5>> canonical_division_range_bits(const Graph& graph, const Model& model, const QuantConfig& quant, const StaticLookupBounds& bounds) {
    std::vector<std::size_t> degrees(graph.num_nodes, 0);
    for (Int destination : graph.dst) ++degrees.at(static_cast<std::size_t>(destination));
    const std::size_t degree_bound = *std::max_element(degrees.begin(), degrees.end());
    const Wide exp_bound = bounds.exp.second;
    const Wide hidden_q_bound = bounds.elu.first;
    const auto profile = [](Wide d, Wide q, bool hidden) { const unsigned db = nonnegative_bit_width(d), rb = nonnegative_bit_width(2 * d - 1); return std::array<unsigned, 5>{{hidden ? nonnegative_bit_width(d * (q + 1)) : 63, hidden ? nonnegative_bit_width(q) : 63, rb, rb, db}}; };
    std::vector<std::array<unsigned, 5>> result;
    for (std::size_t layer = 0; layer < 2; ++layer) for (std::size_t head = 0; head < model.hidden_heads(); ++head) result.push_back(profile(static_cast<Wide>(degree_bound) * exp_bound * quant.s_hprime_hidden[layer][head], hidden_q_bound, true));
    for (std::size_t head = 0; head < model.manifest.output_heads; ++head) { result.push_back(profile(static_cast<Wide>(degree_bound) * exp_bound * quant.s_yprime_output[head], 0, false)); result.push_back(profile(quant.s_out_pre, 0, false)); }
    result.push_back(profile(model.manifest.output_heads, 0, false));
    return result;
}

std::vector<std::array<unsigned, 5>> canonical_division_range_bits(const Graph& graph, const Model& model, const QuantConfig& quant) {
    const StaticTables tables = StaticTables::build(quant);
    return canonical_division_range_bits(graph, model, quant, static_lookup_bounds(tables));
}

QueryCounts canonical_query_counts(const Graph& graph, const Model& model, const QuantConfig& quant, const StaticLookupBounds& bounds) {
    QueryCounts result;
    const auto& lr = bounds.lrelu;
    const auto& ex = bounds.exp;
    const auto& el = bounds.elu;
    const auto tuple_bits = [](const std::pair<Int, Int>& b) { return 2 * (nonnegative_bit_width(2 * static_cast<Wide>(b.first)) + nonnegative_bit_width(2 * static_cast<Wide>(b.second))); };
    const auto profiles = canonical_division_range_bits(graph, model, quant, bounds);
    std::size_t profile_index = 0;
    const auto division_bits = [](const std::array<unsigned, 5>& profile) { return std::accumulate(profile.begin(), profile.end(), std::size_t(0)); };
    const auto add_attention = [&](std::size_t dim, bool hidden) {
        const std::size_t coordinates = checked_mul(graph.num_nodes, dim, "canonical query coordinates");
        result.lrelu = checked_add(result.lrelu, graph.edges(), "canonical LeakyReLU queries");
        result.exp = checked_add(result.exp, graph.edges(), "canonical ExpMap queries");
        if (hidden) result.elu = checked_add(result.elu, coordinates, "canonical ELU queries");
        const std::size_t edge_bits = tuple_bits(lr) + tuple_bits(ex) + nonnegative_bit_width(2 * static_cast<Wide>(lr.second));
        const std::size_t node_bits = division_bits(profiles.at(profile_index++)) + (hidden ? tuple_bits(el) : 0);
        result.range = checked_add(result.range, checked_add(checked_mul(graph.edges(), edge_bits, "canonical edge range bits"), checked_mul(coordinates, node_bits, "canonical node range bits"), "canonical range bits"), "canonical range queries");
    };
    for (std::size_t layer = 0; layer < 2; ++layer) for (std::size_t head = 0; head < model.hidden_heads(); ++head) add_attention(model.tensor(hidden_tensor_name(layer, head, "seq/kernel")).shape.dims.at(2), true);
    for (std::size_t head = 0; head < model.manifest.output_heads; ++head) { add_attention(model.classes(), false); result.range = checked_add(result.range, checked_mul(graph.num_nodes * model.classes(), division_bits(profiles.at(profile_index++)), "alignment range bits"), "canonical range queries"); }
    result.range = checked_add(result.range, checked_mul(graph.num_nodes * model.classes(), division_bits(profiles.at(profile_index)), "average range bits"), "canonical range queries");
    return result;
}

QueryCounts canonical_query_counts(const Graph& graph, const Model& model, const QuantConfig& quant) {
    const StaticTables tables = StaticTables::build(quant);
    return canonical_query_counts(graph, model, quant, static_lookup_bounds(tables));
}

std::size_t canonical_query_size(const Graph& graph, const Model& model, const QuantConfig& quant) {
    const QueryCounts counts = canonical_query_counts(graph, model, quant);
    const std::array<std::size_t, 4> blocks = {next_power_of_two(counts.lrelu), next_power_of_two(counts.exp), next_power_of_two(counts.elu),
                                               next_power_of_two(counts.range)};
    std::size_t result = 0;
    for (std::size_t block : blocks) {
        const std::size_t remainder = result % block;
        if (remainder != 0) result = checked_add(result, block - remainder, "canonical aligned query gap");
        result = checked_add(result, block, "canonical padded query size");
    }
    return result;
}

InputLayout make_layout(std::size_t public_size, std::size_t model_size, std::size_t aux_size, std::size_t query_size) {
    InputLayout layout;
    layout.n_raw = std::max<std::size_t>(1, std::max(std::max(public_size, model_size), std::max(aux_size, query_size)));
    layout.n_in = next_power_of_two(layout.n_raw);
    layout.segments = {{"X_pub", 0, public_size, layout.n_in},
                       {"W", layout.n_in, model_size, layout.n_in},
                       {"Aux", checked_mul(layout.n_in, 2, "Aux segment offset"), aux_size, layout.n_in},
                       {"L_qry", checked_mul(layout.n_in, 3, "Lasso segment offset"), query_size, layout.n_in}};
    return layout;
}

InputLayout canonical_input_layout_impl(const Graph& graph, const Model& model, const QuantConfig& quant) {
    model.validate_schema();
    quant.validate();
    const std::size_t model_size = flatten_model(model, quant).size();
    const std::size_t public_size = checked_mul(graph.num_nodes, model.input_dim(), "public input size");
    const std::size_t aux_size = canonical_aux_size(graph, model);
    const std::size_t query_size = canonical_query_size(graph, model, quant);
    return make_layout(public_size, model_size, aux_size, query_size);
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

AttentionCoordinates assign_attention_coordinates(std::size_t& cursor, const Graph& graph, std::size_t input_dim, std::size_t output_dim) {
    AttentionCoordinates result;
    result.input_dim = input_dim;
    result.output_dim = output_dim;
    const std::size_t nodes = graph.num_nodes;
    const std::size_t edges = graph.edges();
    const std::size_t coordinates = checked_mul(nodes, output_dim, "attention projection coordinates");
    const auto take = [&cursor](std::size_t count, const char* what) {
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

TraceCoordinates make_trace_coordinates(const Graph& graph, const Model& model, const InputLayout& layout) {
    if (layout.segments.size() != 4) throw std::invalid_argument("trace coordinates require four input segments");
    TraceCoordinates result;
    std::size_t cursor =
        checked_add(layout.segments.at(2).offset, checked_mul(graph.num_nodes, model.input_dim(), "input trace coordinates"), "trace coordinate cursor");
    result.hidden.resize(2);
    result.hidden_concat.resize(2);
    for (std::size_t layer = 0; layer < 2; ++layer) {
        const std::size_t output_dim = model.tensor(hidden_tensor_name(layer, 0, "seq/kernel")).shape.dims.at(2);
        result.hidden_concat[layer] = cursor;
        cursor = checked_add(cursor, checked_mul(graph.num_nodes, checked_mul(output_dim, model.hidden_heads(), "concat width"), "concat coordinates"),
                             "trace coordinate cursor");
        result.hidden[layer].reserve(model.hidden_heads());
        const std::size_t input_dim = model.tensor(hidden_tensor_name(layer, 0, "seq/kernel")).shape.dims.at(1);
        for (std::size_t head = 0; head < model.hidden_heads(); ++head)
            result.hidden[layer].push_back(assign_attention_coordinates(cursor, graph, input_dim, output_dim));
    }
    result.output.resize(model.manifest.output_heads);
    result.output_bias.resize(model.manifest.output_heads);
    result.output_aggregate.resize(model.manifest.output_heads);
    result.output_aligned.resize(model.manifest.output_heads);
    result.output_logits.resize(model.manifest.output_heads);
    result.output_alignment_divisions.resize(model.manifest.output_heads);
    const std::size_t classes = model.classes();
    const std::size_t output_coordinates = checked_mul(graph.num_nodes, classes, "output coordinates");
    for (std::size_t head = 0; head < model.manifest.output_heads; ++head) {
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
    const std::size_t expected_end = checked_add(layout.segments.at(2).offset, layout.segments.at(2).logical_length, "Aux segment end");
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

QueryCoordinates make_query_coordinates(const Graph& graph, const Model& model, const QuantConfig& quant, const InputLayout& layout) {
    QueryCoordinates result;
    result.counts = canonical_query_counts(graph, model, quant);
    std::size_t cursor = layout.segments.at(3).offset;
    const auto start_block = [&cursor](std::size_t block, const char* what) {
        if (block == 0 || (block & (block - 1)) != 0) throw std::invalid_argument("canonical query block is not power-of-two");
        const std::size_t remainder = cursor % block;
        if (remainder != 0) cursor = checked_add(cursor, block - remainder, "canonical aligned query gap");
        const std::size_t start = cursor;
        cursor = checked_add(cursor, block, what);
        return start;
    };
    result.lrelu = start_block(next_power_of_two(result.counts.lrelu), "LReLU query block");
    result.exp = start_block(next_power_of_two(result.counts.exp), "ExpMap query block");
    result.elu = start_block(next_power_of_two(result.counts.elu), "ELU query block");
    result.range = start_block(next_power_of_two(result.counts.range), "range query block");
    result.end = cursor;
    const std::size_t expected_end = checked_add(layout.segments.at(3).offset, layout.segments.at(3).logical_length, "L_qry segment end");
    if (cursor != expected_end) throw std::invalid_argument("canonical query coordinates do not cover L_qry");
    return result;
}

std::string family_name_from_index(std::size_t index) {
    std::ostringstream out;
    out << "family/" << index;
    return out.str();
}

void append_checkpoint(GatCircuitPlan& plan, const std::string& id, const std::string& relation, const std::string& segment, std::size_t coordinates,
                       std::size_t coordinate_offset = 0, const std::string& coordinate_rule = "canonical_row_major") {
    CheckpointDescriptor checkpoint;
    checkpoint.id = id;
    checkpoint.relation_id = relation;
    checkpoint.segment = segment;
    checkpoint.coordinates = coordinates;
    checkpoint.coordinate_offset = coordinate_offset;
    checkpoint.coordinate_rule = coordinate_rule;
    plan.checkpoints.push_back(std::move(checkpoint));
}

void require_input_region(const GatCircuitPlan& plan, std::size_t offset, std::size_t length, const char* what) {
    const std::size_t input_size = plan.circuit.circuit.front().size;
    if (offset > input_size || length > input_size - offset) {
        throw std::invalid_argument(std::string("canonical circuit coordinate exceeds input layer: ") + what);
    }
}

const InputSegment* canonical_segment_for_span(const InputLayout& layout, std::size_t offset, std::size_t length) {
    const InputSegment* matched = nullptr;
    for (const InputSegment& segment : layout.segments) {
        if (offset < segment.offset) continue;
        const std::size_t local = offset - segment.offset;
        if (local > segment.logical_length || length > segment.logical_length - local) continue;
        if (matched != nullptr) return nullptr;
        matched = &segment;
    }
    return matched;
}

void bind_matmul_segments(MatMulDescriptor& descriptor, const InputLayout& layout) {
    const std::size_t lhs_span = checked_mul(descriptor.rows, descriptor.inner, "MatMul canonical LHS span");
    const std::size_t rhs_span = checked_mul(descriptor.inner, descriptor.cols, "MatMul canonical RHS span");
    const InputSegment* lhs = canonical_segment_for_span(layout, descriptor.lhs_offset, lhs_span);
    const InputSegment* rhs = canonical_segment_for_span(layout, descriptor.rhs_offset, rhs_span);
    const InputSegment* output = canonical_segment_for_span(layout, descriptor.output_offset, descriptor.output_coordinates);
    if (lhs == nullptr || rhs == nullptr || output == nullptr)
        throw std::invalid_argument("MatMul operand/output span is not inside a canonical logical segment");
    descriptor.lhs_segment = lhs->name;
    descriptor.rhs_segment = rhs->name;
    descriptor.output_segment = output->name;
}

NativeGkrMatmulShape native_gkr_shape_from_descriptor(const MatMulDescriptor& descriptor, const InputLayout& layout) {
    NativeGkrMatmulShape shape;
    shape.rows = descriptor.rows;
    shape.inner = descriptor.inner;
    shape.cols = descriptor.cols;
    shape.relation_id = descriptor.relation_id;
    shape.family_id = descriptor.family_id;
    shape.lhs_offset = descriptor.lhs_offset;
    shape.rhs_offset = descriptor.rhs_offset;
    shape.output_offset = descriptor.output_offset;
    shape.input_segment_domain = layout.n_in;
    for (const InputSegment& segment : layout.segments) shape.input_segment_logical_lengths.push_back(segment.logical_length);
    const InputSegment* lhs =
        canonical_segment_for_span(layout, descriptor.lhs_offset, checked_mul(descriptor.rows, descriptor.inner, "native GKR canonical LHS span"));
    const InputSegment* rhs =
        canonical_segment_for_span(layout, descriptor.rhs_offset, checked_mul(descriptor.inner, descriptor.cols, "native GKR canonical RHS span"));
    const InputSegment* output = canonical_segment_for_span(layout, descriptor.output_offset, descriptor.output_coordinates);
    if (lhs == nullptr || rhs == nullptr || output == nullptr || lhs->name != descriptor.lhs_segment || rhs->name != descriptor.rhs_segment ||
        output->name != descriptor.output_segment)
        throw std::invalid_argument("native GKR MatMul descriptor has a stale canonical segment mapping");
    shape.lhs_segment = lhs->name;
    shape.rhs_segment = rhs->name;
    shape.lhs_segment_offset = descriptor.lhs_offset - lhs->offset;
    shape.rhs_segment_offset = descriptor.rhs_offset - rhs->offset;
    shape.output_segment = output->name;
    shape.output_segment_offset = descriptor.output_offset - output->offset;
    return shape;
}

std::string gkr_point_shape_digest(const layeredCircuit& circuit,
                                  const std::vector<MatMulDescriptor>& matmuls,
                                  const InputLayout& layout) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-gkr-point-shape-v1");
    append_u64(encoded, circuit.circuit.size());
    for (std::size_t index = 0; index < circuit.circuit.size(); ++index) {
        const layer& current = circuit.circuit[index];
        append_u64(encoded, index);
        append_u64(encoded, static_cast<std::uint64_t>(current.ty));
        append_u64(encoded, current.size);
        append_u64(encoded, static_cast<std::uint64_t>(static_cast<std::int64_t>(current.bit_length)));
        append_u64(encoded, current.size_u[0]);
        append_u64(encoded, current.size_u[1]);
        append_u64(encoded, current.size_v[0]);
        append_u64(encoded, current.size_v[1]);
        append_u64(encoded, static_cast<std::uint64_t>(static_cast<std::int64_t>(current.bit_length_u[0])));
        append_u64(encoded, static_cast<std::uint64_t>(static_cast<std::int64_t>(current.bit_length_u[1])));
        append_u64(encoded, static_cast<std::uint64_t>(static_cast<std::int64_t>(current.bit_length_v[0])));
        append_u64(encoded, static_cast<std::uint64_t>(static_cast<std::int64_t>(current.bit_length_v[1])));
        append_u64(encoded, static_cast<std::uint64_t>(static_cast<std::int64_t>(current.max_bl_u)));
        append_u64(encoded, static_cast<std::uint64_t>(static_cast<std::int64_t>(current.max_bl_v)));
        append_u64(encoded, current.need_phase2 ? 1 : 0);
        append_string(encoded, current.native_relation_id);
        append_string(encoded, current.native_family_id);
        append_u64(encoded, current.native_rows);
        append_u64(encoded, current.native_inner);
        append_u64(encoded, current.native_cols);
        append_u64(encoded, current.native_lhs_offset);
        append_u64(encoded, current.native_rhs_offset);
        append_u64(encoded, current.native_output_offset);
    }
    append_u64(encoded, matmuls.size());
    for (const MatMulDescriptor& descriptor : matmuls) {
        const NativeGkrMatmulShape shape = native_gkr_shape_from_descriptor(descriptor, layout);
        append_string(encoded, shape.relation_id);
        append_string(encoded, shape.family_id);
        append_u64(encoded, shape.rows);
        append_u64(encoded, shape.inner);
        append_u64(encoded, shape.cols);
        append_u64(encoded, shape.lhs_offset);
        append_u64(encoded, shape.rhs_offset);
        append_u64(encoded, shape.output_offset);
        append_string(encoded, shape.lhs_segment);
        append_u64(encoded, shape.lhs_segment_offset);
        append_string(encoded, shape.rhs_segment);
        append_u64(encoded, shape.rhs_segment_offset);
        append_string(encoded, shape.output_segment);
        append_u64(encoded, shape.output_segment_offset);
    }
    return canonical_digest64(encoded);
}

void append_unary_layer(GatCircuitPlan& plan, layerType type, std::size_t output_size, std::size_t input_offset, std::size_t input_stride = 1) {
    if (plan.circuit.circuit.size() >= std::numeric_limits<u8>::max()) throw std::overflow_error("zkGPT circuit has too many layers");
    layer current;
    current.ty = type;
    current.size = as_u64(std::max<std::size_t>(1, output_size), "layer size");
    current.zero_start_id = current.size;
    current.bit_length = bit_length(current.size);
    current.need_phase2 = false;
    const std::size_t count = std::max<std::size_t>(1, output_size);
    if (output_size == 0)
        require_input_region(plan, input_offset, 1, "empty unary layer");
    else
        require_input_region(plan, input_offset, checked_add(checked_mul(output_size - 1, input_stride, "unary coordinate stride"), 1, "unary coordinate span"),
                             "unary layer");
    current.reserve_gate_arrays(count, 0);
    current.reserve_coefficients(1);
    for (std::size_t g = 0; g < count; ++g) {
        const std::size_t source = checked_add(input_offset, checked_mul(g, input_stride, "unary coordinate"), "unary coordinate");
        current.add_unary(as_u64(g, "unary gate output"), as_u64(source, "unary gate input"), 0, 1);
    }
    plan.circuit.circuit.push_back(std::move(current));
}

void append_binary_layer(GatCircuitPlan& plan, layerType type, std::size_t output_size, std::size_t lhs_offset, std::size_t rhs_offset,
                         std::size_t lhs_stride = 1, std::size_t rhs_stride = 1, std::uint32_t l = 0, ll sc = 1) {
    if (plan.circuit.circuit.size() >= std::numeric_limits<u8>::max()) throw std::overflow_error("zkGPT circuit has too many layers");
    layer current;
    current.ty = type;
    current.size = as_u64(std::max<std::size_t>(1, output_size), "binary layer size");
    current.zero_start_id = current.size;
    current.bit_length = bit_length(current.size);
    current.need_phase2 = false;
    const std::size_t count = std::max<std::size_t>(1, output_size);
    const std::size_t span = output_size == 0 ? 1 : checked_add(checked_mul(output_size - 1, lhs_stride, "binary lhs stride"), 1, "binary lhs span");
    require_input_region(plan, lhs_offset, span, "binary lhs");
    const std::size_t rhs_span = output_size == 0 ? 1 : checked_add(checked_mul(output_size - 1, rhs_stride, "binary rhs stride"), 1, "binary rhs span");
    require_input_region(plan, rhs_offset, rhs_span, "binary rhs");
    current.reserve_gate_arrays(0, count);
    current.reserve_coefficients(1);
    for (std::size_t g = 0; g < count; ++g) {
        const std::size_t lhs = checked_add(lhs_offset, checked_mul(g, lhs_stride, "binary lhs coordinate"), "binary lhs coordinate");
        const std::size_t rhs = checked_add(rhs_offset, checked_mul(g, rhs_stride, "binary rhs coordinate"), "binary rhs coordinate");
        current.add_binary(as_u64(g, "binary gate output"), as_u64(lhs, "binary gate lhs"), as_u64(rhs, "binary gate rhs"), sc, static_cast<u8>(l));
    }
    plan.circuit.circuit.push_back(std::move(current));
}

struct ExplicitUnaryGate {
    std::size_t g = 0;
    std::size_t u = 0;
    u8 lu = 0;
    ll sc = 1;
};
static_assert(sizeof(ExplicitUnaryGate) >=
                  2 * sizeof(std::size_t) + sizeof(u8) + sizeof(ll),
              "GAT explicit unary temporary must retain all coordinate and coefficient fields");

struct ExplicitBinaryGate {
    std::size_t g = 0;
    std::size_t u = 0;
    std::size_t v = 0;
    u8 l = 0;
    ll sc = 1;
};

class GkrPointShapeCollector {
  public:
    GkrPointShapeCollector(layeredCircuit& circuit, const InputLayout& layout, bool symbolic = false)
        : circuit_(circuit), symbolic_(symbolic),
          progress_started_(std::chrono::steady_clock::now()),
          progress_last_report_(progress_started_) {
        layer_shapes_.reserve(256);
        append_layer_shape(circuit_.circuit.front(), 0, 0);
        std::uint64_t support_entries = 0;
        std::uint64_t previous_end = 0;
        for (const InputSegment& segment : layout.segments) {
            if (segment.logical_length == 0) continue;
            if (segment.offset < previous_end ||
                segment.offset > circuit_.circuit.front().size ||
                segment.logical_length > circuit_.circuit.front().size - segment.offset ||
                segment.logical_length > std::numeric_limits<std::uint64_t>::max() - support_entries)
                throw std::invalid_argument("shape-only GKR source support ranges are invalid");
            ranges_.emplace_back(static_cast<u64>(segment.offset),
                                 static_cast<u64>(segment.logical_length));
            support_entries += static_cast<std::uint64_t>(segment.logical_length);
            prefixes_.push_back(support_entries);
            previous_end = static_cast<std::uint64_t>(segment.offset + segment.logical_length);
        }
        support_entries_ = support_entries;
        input_source_entries_upper_bound_ = support_entries;
        if (!symbolic_) {
            const std::uint64_t words64 = support_entries / 64 + (support_entries % 64 != 0);
            if (words64 > std::numeric_limits<std::size_t>::max() / (2 * sizeof(std::uint64_t)))
                throw std::length_error("shape-only GKR source bitmap exceeds native size_t");
            word_count_ = static_cast<std::size_t>(words64);
            if (word_count_ > std::numeric_limits<std::size_t>::max() / 2)
                throw std::length_error("shape-only GKR dual-lane bitmap exceeds native size_t");
            seen_.resize(word_count_ * 2);
            bitmap_bytes_ = static_cast<std::uint64_t>(seen_.size()) * sizeof(std::uint64_t);
        }
        if (prefixes_.empty()) prefixes_.push_back(0);
        if (!symbolic_) {
            touched_index_capacity_ = seen_.size();
            if (touched_index_capacity_ > std::numeric_limits<std::size_t>::max() / sizeof(std::size_t))
                throw std::length_error("shape-only GKR touched-index capacity exceeds native size_t");
            touched_index_bytes_upper_bound_ =
                static_cast<std::uint64_t>(touched_index_capacity_) * sizeof(std::size_t);
            if (touched_index_capacity_ != 0)
                touched_indices_ = touched_index_allocator_.allocate(touched_index_capacity_);
        }
    }

    ~GkrPointShapeCollector() {
        if (touched_indices_ != nullptr)
            touched_index_allocator_.deallocate(touched_indices_, touched_index_capacity_);
    }
    GkrPointShapeCollector(const GkrPointShapeCollector&) = delete;
    GkrPointShapeCollector& operator=(const GkrPointShapeCollector&) = delete;

    std::uint64_t bitmap_bytes() const noexcept { return bitmap_bytes_; }
    std::uint64_t touched_index_bytes_upper_bound() const noexcept {
        return touched_index_bytes_upper_bound_;
    }
    const std::vector<GatGkrLayerShapeProfile>& layer_shapes() const noexcept { return layer_shapes_; }
    bool symbolic() const noexcept { return symbolic_; }

    void set_symbolic_input_source_entries_upper_bound(std::uint64_t bound) {
        if (!symbolic_ || bound == 0 || bound > support_entries_)
            throw std::invalid_argument("symbolic GKR active input-source bound is invalid");
        input_source_entries_upper_bound_ = bound;
    }

    void append_unary(std::size_t output, std::size_t input, std::uint8_t lu) {
        const std::size_t current_layer = circuit_.circuit.size();
        if (symbolic_) {
            observe_output(output);
            has_pre_layer_u_ = has_pre_layer_u_ || lu != 0;
            ++unary_count_;
            (void)input;
            return;
        }
        const std::size_t source_layer = lu == 0 ? 0 : current_layer - 1;
        observe_output(output);
        observe_source(0, source_layer, input, current_layer);
        has_pre_layer_u_ = has_pre_layer_u_ || lu != 0;
        ++unary_count_;
        note_shape_progress(1);
    }

    void append_binary(std::size_t output, std::size_t lhs, std::size_t rhs,
                       std::uint8_t l) {
        const std::size_t current_layer = circuit_.circuit.size();
        if (symbolic_) {
            observe_output(output);
            has_pre_layer_u_ = has_pre_layer_u_ || l != 0;
            has_pre_layer_v_ = has_pre_layer_v_ || (l & 1) != 0;
            has_v_operand_ = true;
            ++binary_count_;
            (void)lhs;
            (void)rhs;
            return;
        }
        const std::size_t source_u = l == 0 ? 0 : current_layer - 1;
        const std::size_t source_v = (l & 1) == 0 ? 0 : current_layer - 1;
        observe_output(output);
        observe_source(0, source_u, lhs, current_layer);
        observe_source(1, source_v, rhs, current_layer);
        has_v_operand_ = true;
        ++binary_count_;
        note_shape_progress(1);
    }

    void append_range_decomposition(std::size_t first_output,
                                    std::size_t source,
                                    std::size_t range_begin,
                                    std::size_t width,
                                    std::size_t unary_count,
                                    std::size_t binary_count) {
        const std::size_t current_layer = circuit_.circuit.size();
        if (width == 0 || first_output > std::numeric_limits<std::size_t>::max() - width ||
            range_begin > std::numeric_limits<std::size_t>::max() - width)
            throw std::overflow_error("shape-only GKR range decomposition overflows size_t");
        if (symbolic_) {
            observe_output(first_output + width);
            (void)support_index(static_cast<std::uint64_t>(source));
            validate_support_interval(static_cast<std::uint64_t>(range_begin),
                                      static_cast<std::uint64_t>(width));
            if (width > (std::numeric_limits<std::size_t>::max() - 1) / 2 ||
                unary_count != 2 * width + 1 || binary_count != width)
                throw std::logic_error("symbolic GKR range gate profile is inconsistent");
            unary_count_ += unary_count;
            binary_count_ += binary_count;
            has_v_operand_ = true;
            return;
        }
        observe_output(first_output + width);
        observe_source(0, 0, source, current_layer);
        mark_support_interval(0, static_cast<std::uint64_t>(range_begin),
                              static_cast<std::uint64_t>(width));
        mark_support_interval(1, static_cast<std::uint64_t>(range_begin),
                              static_cast<std::uint64_t>(width));
        if (width > (std::numeric_limits<std::size_t>::max() - 1) / 2)
            throw std::overflow_error("shape-only GKR range gate count overflows size_t");
        const std::size_t expected_unary = 2 * width + 1;
        if (unary_count != expected_unary || binary_count != width)
            throw std::logic_error("shape-only GKR range gate profile is inconsistent");
        unary_count_ += unary_count;
        binary_count_ += binary_count;
        has_v_operand_ = true;
        if (binary_count > std::numeric_limits<std::size_t>::max() - unary_count)
            throw std::overflow_error("GKR shape progress gate delta overflows size_t");
        note_shape_progress(static_cast<std::uint64_t>(unary_count + binary_count));
    }

    void append_explicit_layer(layerType type, std::size_t output_size,
                               std::size_t unary_count, std::size_t binary_count) {
        if (symbolic_) {
            const std::uint64_t u_refs = static_cast<std::uint64_t>(unary_count) + binary_count;
            const std::uint64_t v_refs = static_cast<std::uint64_t>(binary_count);
            append_symbolic_explicit_layer(type, output_size, unary_count, binary_count,
                                           std::min(input_source_entries_upper_bound_, u_refs),
                                           std::min(input_source_entries_upper_bound_, v_refs),
                                           has_pre_layer_u_, has_pre_layer_v_, has_v_operand_);
            return;
        }
        if (unary_count != unary_count_ || binary_count != binary_count_ ||
            (unary_count_ == 0 && binary_count_ == 0))
            throw std::logic_error("shape-only GKR gate visitor count mismatch");
        if (output_size == 0 || maximum_output_ >= output_size)
            throw std::invalid_argument("shape-only GKR gate output exceeds its layer domain");

        layer current;
        current.ty = type;
        current.size = as_u64(std::max<std::size_t>(1, output_size), "shape-only explicit layer size");
        current.zero_start_id = current.size;
        current.bit_length = bit_length(current.size);
        const layer& previous = circuit_.circuit.back();
        current.size_u[0] = count_lane(0);
        current.size_v[0] = count_lane(1);
        current.bit_length_u[0] = ceilPow2BitLength(current.size_u[0]);
        current.bit_length_v[0] = ceilPow2BitLength(current.size_v[0]);
        if (has_pre_layer_u_ || type == layerType::FFT || type == layerType::IFFT) {
            current.size_u[1] = previous.size;
            current.bit_length_u[1] = previous.bit_length;
        }
        if (has_pre_layer_v_) {
            current.size_v[1] = previous.size;
            current.bit_length_v[1] = previous.bit_length;
        }
        current.need_phase2 = has_v_operand_;
        current.updateSize();
        append_layer_shape(current, unary_count_, binary_count_);
        circuit_.circuit.push_back(std::move(current));
        circuit_.size = static_cast<u8>(circuit_.circuit.size());
        clear_lanes();
        reset_layer_counters();
    }

    void append_symbolic_explicit_layer(layerType type, std::size_t output_size,
                                        std::size_t unary_count, std::size_t binary_count,
                                        std::uint64_t input_u_upper, std::uint64_t input_v_upper,
                                        bool has_pre_layer_u, bool has_pre_layer_v,
                                        bool need_phase2) {
        if (!symbolic_ || (unary_count == 0 && binary_count == 0) || output_size == 0 ||
            circuit_.circuit.size() >= std::numeric_limits<u8>::max())
            throw std::invalid_argument("symbolic GKR layer shape is empty or exceeds native layer limits");
        layer current;
        current.ty = type;
        current.size = as_u64(output_size, "symbolic explicit layer size");
        current.zero_start_id = current.size;
        current.bit_length = bit_length(current.size);
        current.size_u[0] = as_u64(std::min<std::uint64_t>(input_source_entries_upper_bound_, input_u_upper), "symbolic U input source bound");
        current.size_v[0] = as_u64(std::min<std::uint64_t>(input_source_entries_upper_bound_, input_v_upper), "symbolic V input source bound");
        current.bit_length_u[0] = current.size_u[0] == 0 ? -1 : ceilPow2BitLength(current.size_u[0]);
        current.bit_length_v[0] = current.size_v[0] == 0 ? -1 : ceilPow2BitLength(current.size_v[0]);
        if (has_pre_layer_u) {
            current.size_u[1] = circuit_.circuit.back().size;
            current.bit_length_u[1] = circuit_.circuit.back().bit_length;
        }
        if (has_pre_layer_v) {
            current.size_v[1] = circuit_.circuit.back().size;
            current.bit_length_v[1] = circuit_.circuit.back().bit_length;
        }
        current.need_phase2 = need_phase2;
        current.updateSize();
        append_layer_shape(current, unary_count, binary_count);
        circuit_.circuit.push_back(std::move(current));
        circuit_.size = static_cast<u8>(circuit_.circuit.size());
        reset_layer_counters();
    }

    void append_fconn(const layer& current) {
        if (unary_count_ != 0 || binary_count_ != 0)
            throw std::logic_error("shape-only GKR FCONN overlaps an unfinished gate layer");
        layer shape = current;
        shape.size_u[0] = shape.size_v[0] = 0;
        shape.size_u[1] = shape.size_v[1] = 0;
        shape.bit_length_u[0] = shape.bit_length_v[0] = -1;
        shape.bit_length_u[1] = shape.bit_length_v[1] = -1;
        shape.updateSize();
        append_layer_shape(shape, 0, 0);
        circuit_.circuit.push_back(std::move(shape));
        circuit_.size = static_cast<u8>(circuit_.circuit.size());
    }

  private:
    void note_shape_progress(std::uint64_t gate_delta) {
        if (gate_delta > std::numeric_limits<std::uint64_t>::max() - progress_total_gates_)
            throw std::overflow_error("GKR shape progress gate count overflows u64");
        progress_total_gates_ += gate_delta;
        constexpr std::uint64_t kGateProgressInterval = 1ULL << 20;
        if (progress_total_gates_ < next_progress_gate_threshold_) return;
        const std::uint64_t next_multiple = progress_total_gates_ / kGateProgressInterval + 1;
        next_progress_gate_threshold_ = next_multiple >
                std::numeric_limits<std::uint64_t>::max() / kGateProgressInterval
            ? std::numeric_limits<std::uint64_t>::max()
            : next_multiple * kGateProgressInterval;
        const auto now = std::chrono::steady_clock::now();
        if (now - progress_last_report_ < std::chrono::seconds(15)) return;
        progress_last_report_ = now;
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - progress_started_).count();
        std::cerr << "NATIVE_GKR_SHAPE_CENSUS_PROGRESS gates=" << progress_total_gates_
                  << " layer_unary=" << unary_count_
                  << " layer_binary=" << binary_count_
                  << " current_layer=" << circuit_.circuit.size()
                  << " elapsed_ms=" << elapsed_ms << '\n' << std::flush;
    }

    void validate_support_interval(std::uint64_t begin, std::uint64_t count) const {
        if (count > std::numeric_limits<std::uint64_t>::max() - begin)
            throw std::overflow_error("symbolic GKR source interval overflows u64");
        std::uint64_t coordinate = begin;
        const std::uint64_t end = begin + count;
        while (coordinate < end) {
            const auto after = std::upper_bound(
                ranges_.begin(), ranges_.end(), coordinate,
                [](std::uint64_t value, const auto& range) { return value < range.first; });
            if (after == ranges_.begin())
                throw std::out_of_range("symbolic GKR source interval is outside logical support");
            const auto& range = *(after - 1);
            if (coordinate < range.first || coordinate - range.first >= range.second)
                throw std::out_of_range("symbolic GKR source interval crosses a logical-support gap");
            const std::uint64_t range_end = static_cast<std::uint64_t>(range.first) +
                static_cast<std::uint64_t>(range.second);
            coordinate = std::min<std::uint64_t>(end, range_end);
        }
    }

    void append_layer_shape(const layer& current, std::size_t unary, std::size_t binary) {
        GatGkrLayerShapeProfile shape;
        shape.layer_type = static_cast<std::uint32_t>(current.ty);
        shape.size = current.size;
        shape.size_u0 = current.size_u[0];
        shape.size_u1 = current.size_u[1];
        shape.size_v0 = current.size_v[0];
        shape.size_v1 = current.size_v[1];
        shape.unary_gates = static_cast<std::uint64_t>(unary);
        shape.binary_gates = static_cast<std::uint64_t>(binary);
        shape.native_rows = current.native_rows;
        shape.native_inner = current.native_inner;
        shape.native_cols = current.native_cols;
        shape.bit_length = current.bit_length;
        shape.max_bl_u = current.max_bl_u;
        shape.max_bl_v = current.max_bl_v;
        shape.need_phase2 = current.need_phase2;
        layer_shapes_.push_back(shape);
    }

    void observe_output(std::size_t output) {
        if (output == std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("shape-only GKR output coordinate overflows size_t");
        maximum_output_ = std::max(maximum_output_, output);
    }

    std::size_t support_index(std::uint64_t coordinate) const {
        const auto after = std::upper_bound(
            ranges_.begin(), ranges_.end(), coordinate,
            [](std::uint64_t value, const auto& range) { return value < range.first; });
        if (after == ranges_.begin())
            throw std::out_of_range("shape-only GKR source is outside logical input support");
        const std::size_t index = static_cast<std::size_t>((after - ranges_.begin()) - 1);
        const auto& range = ranges_[index];
        if (coordinate < range.first || coordinate - range.first >= range.second)
            throw std::out_of_range("shape-only GKR source is outside logical input support");
        const std::uint64_t prefix = index == 0 ? 0 : prefixes_[index - 1];
        const std::uint64_t result = prefix + coordinate - range.first;
        if (result >= support_entries_)
            throw std::logic_error("shape-only GKR source rank exceeds logical support");
        return static_cast<std::size_t>(result);
    }

    void set_support_bits(std::size_t lane, std::uint64_t begin,
                          std::uint64_t count) {
        std::uint64_t end = 0;
        if (count > std::numeric_limits<std::uint64_t>::max() - begin)
            throw std::overflow_error("shape-only GKR source interval overflows u64");
        end = begin + count;
        std::uint64_t coordinate = begin;
        while (coordinate < end) {
            const auto after = std::upper_bound(
                ranges_.begin(), ranges_.end(), coordinate,
                [](std::uint64_t value, const auto& range) { return value < range.first; });
            if (after == ranges_.begin())
                throw std::out_of_range("shape-only GKR source interval is outside logical support");
            const std::size_t range_index = static_cast<std::size_t>((after - ranges_.begin()) - 1);
            const auto& range = ranges_[range_index];
            if (coordinate < range.first || coordinate - range.first >= range.second)
                throw std::out_of_range("shape-only GKR source interval crosses a logical-support gap");
            const std::uint64_t range_end = range.first + range.second;
            const std::uint64_t part_end = std::min(end, range_end);
            const std::uint64_t prefix = range_index == 0 ? 0 : prefixes_[range_index - 1];
            std::uint64_t support = prefix + coordinate - range.first;
            const std::uint64_t part_count = part_end - coordinate;
            const std::size_t plane = lane * word_count_;
            const std::uint64_t support_end = support + part_count;
            while (support < support_end) {
                const std::size_t word = static_cast<std::size_t>(support >> 6);
                const unsigned first_bit = static_cast<unsigned>(support & 63);
                const std::uint64_t available = 64 - first_bit;
                const std::uint64_t take = std::min(available, support_end - support);
                const std::uint64_t mask = take == 64
                    ? ~std::uint64_t(0)
                    : ((std::uint64_t(1) << take) - 1) << first_bit;
                mark_seen(plane + word, mask);
                support += take;
            }
            coordinate = part_end;
        }
    }

    void mark_support_interval(std::size_t lane, std::uint64_t begin,
                               std::uint64_t count) {
        set_support_bits(lane, begin, count);
    }

    void observe_source(std::size_t lane, std::size_t source_layer,
                        std::size_t coordinate, std::size_t current_layer) {
        if (source_layer >= current_layer || source_layer >= circuit_.circuit.size())
            throw std::out_of_range("shape-only GKR source layer is invalid");
        if (coordinate >= circuit_.circuit[source_layer].size)
            throw std::out_of_range("shape-only GKR source coordinate exceeds its layer domain");
        if (source_layer != 0) {
            if (lane == 0) has_pre_layer_u_ = true;
            else has_pre_layer_v_ = true;
            return;
        }
        const std::size_t index = support_index(static_cast<std::uint64_t>(coordinate));
        const std::size_t plane = lane * word_count_;
        mark_seen(plane + (index >> 6), std::uint64_t(1) << (index & 63));
    }

    void mark_seen(std::size_t absolute_word, std::uint64_t mask) {
        if (absolute_word >= seen_.size() || mask == 0)
            throw std::out_of_range("shape-only GKR bitmap word is invalid");
        if (seen_[absolute_word] == 0) {
            if (touched_index_count_ >= touched_index_capacity_ || touched_indices_ == nullptr)
                throw std::length_error("shape-only GKR touched-index storage exceeded its exact word bound");
            touched_indices_[touched_index_count_++] = absolute_word;
        }
        seen_[absolute_word] |= mask;
    }

    std::uint64_t count_lane(std::size_t lane) const {
        std::uint64_t count = 0;
        if (lane > 1)
            throw std::out_of_range("shape-only GKR lane index is invalid");
        const std::size_t lane_begin = lane * word_count_;
        const std::size_t lane_end = lane_begin + word_count_;
        for (std::size_t offset = 0; offset < touched_index_count_; ++offset) {
            const std::size_t absolute_word = touched_indices_[offset];
            if (absolute_word < lane_begin || absolute_word >= lane_end) continue;
            count += static_cast<std::uint64_t>(__builtin_popcountll(seen_[absolute_word]));
        }
        return count;
    }

    void clear_lanes() {
        for (std::size_t offset = 0; offset < touched_index_count_; ++offset)
            seen_[touched_indices_[offset]] = 0;
        touched_index_count_ = 0;
    }

    void reset_layer_counters() {
        unary_count_ = 0;
        binary_count_ = 0;
        maximum_output_ = 0;
        has_pre_layer_u_ = false;
        has_pre_layer_v_ = false;
        has_v_operand_ = false;
    }

    layeredCircuit& circuit_;
    std::vector<std::pair<u64, u64>> ranges_;
    std::vector<std::uint64_t> prefixes_;
    zkgpt::NativeVector<std::uint64_t> seen_;
    std::vector<GatGkrLayerShapeProfile> layer_shapes_;
    std::uint64_t support_entries_ = 0;
    std::uint64_t input_source_entries_upper_bound_ = 0;
    std::uint64_t bitmap_bytes_ = 0;
    std::size_t word_count_ = 0;
    std::size_t unary_count_ = 0;
    std::size_t binary_count_ = 0;
    std::size_t maximum_output_ = 0;
    zkgpt::NativeScratchAllocator<std::size_t> touched_index_allocator_;
    std::size_t* touched_indices_ = nullptr;
    std::size_t touched_index_capacity_ = 0;
    std::size_t touched_index_count_ = 0;
    std::uint64_t touched_index_bytes_upper_bound_ = 0;
    bool has_pre_layer_u_ = false;
    bool has_pre_layer_v_ = false;
    bool has_v_operand_ = false;
    bool symbolic_ = false;
    std::uint64_t progress_total_gates_ = 0;
    std::uint64_t next_progress_gate_threshold_ = 1ULL << 20;
    std::chrono::steady_clock::time_point progress_started_;
    std::chrono::steady_clock::time_point progress_last_report_;
};

thread_local GkrPointShapeCollector* active_gkr_point_shape_collector = nullptr;

class GkrPointShapeCollectorScope {
  public:
    explicit GkrPointShapeCollectorScope(GkrPointShapeCollector* collector)
        : previous_(active_gkr_point_shape_collector) {
        active_gkr_point_shape_collector = collector;
    }
    ~GkrPointShapeCollectorScope() { active_gkr_point_shape_collector = previous_; }
    GkrPointShapeCollectorScope(const GkrPointShapeCollectorScope&) = delete;
    GkrPointShapeCollectorScope& operator=(const GkrPointShapeCollectorScope&) = delete;
  private:
    GkrPointShapeCollector* previous_;
};

class ExplicitUnaryGateBuffer {
  public:
    void reserve(std::size_t count) {
        if (active_gkr_point_shape_collector == nullptr) {
            gates_.reserve(count);
            if (count != 0U && gates_.capacity() != count)
                throw std::length_error("explicit unary gate buffer capacity differs from its exact source bound");
        }
    }
    void push_back(const ExplicitUnaryGate& gate) {
        ++count_;
        if (active_gkr_point_shape_collector != nullptr)
            active_gkr_point_shape_collector->append_unary(gate.g, gate.u, gate.lu);
        else
            gates_.push_back(gate);
    }
    std::size_t size() const noexcept { return count_; }
    bool empty() const noexcept { return count_ == 0; }
    auto begin() const { return gates_.begin(); }
    auto end() const { return gates_.end(); }
  private:
    zkgpt::NativeVector<ExplicitUnaryGate> gates_;
    std::size_t count_ = 0;
};

class ExplicitBinaryGateBuffer {
  public:
    void reserve(std::size_t count) {
        if (active_gkr_point_shape_collector == nullptr) gates_.reserve(count);
    }
    void push_back(const ExplicitBinaryGate& gate) {
        ++count_;
        if (active_gkr_point_shape_collector != nullptr)
            active_gkr_point_shape_collector->append_binary(gate.g, gate.u, gate.v, gate.l);
        else
            gates_.push_back(gate);
    }
    std::size_t size() const noexcept { return count_; }
    bool empty() const noexcept { return count_ == 0; }
    auto begin() const { return gates_.begin(); }
    auto end() const { return gates_.end(); }
  private:
    zkgpt::NativeVector<ExplicitBinaryGate> gates_;
    std::size_t count_ = 0;
};

const char* explicit_layer_type_name(layerType type) {
    switch (type) {
        case layerType::INPUT: return "INPUT";
        case layerType::FFT: return "FFT";
        case layerType::IFFT: return "IFFT";
        case layerType::ADD_BIAS: return "ADD_BIAS";
        case layerType::RELU: return "RELU";
        case layerType::Sqr: return "Sqr";
        case layerType::OPT_AVG_POOL: return "OPT_AVG_POOL";
        case layerType::MAX_POOL: return "MAX_POOL";
        case layerType::AVG_POOL: return "AVG_POOL";
        case layerType::DOT_PROD: return "DOT_PROD";
        case layerType::PADDING: return "PADDING";
        case layerType::FCONN: return "FCONN";
        case layerType::LAYER_NORM_1: return "LAYER_NORM_1";
        case layerType::LAYER_NORM_2: return "LAYER_NORM_2";
        case layerType::LAYER_NORM_3: return "LAYER_NORM_3";
        case layerType::GELU_1: return "GELU_1";
        case layerType::GELU_2: return "GELU_2";
        case layerType::GELU_3: return "GELU_3";
        case layerType::MHA_QK: return "MHA_QK";
        case layerType::SOFTMAX_1: return "SOFTMAX_1";
        case layerType::SOFTMAX_2: return "SOFTMAX_2";
        case layerType::SOFTMAX_3: return "SOFTMAX_3";
        case layerType::PLACE_HOLDER: return "PLACE_HOLDER";
    }
    return "UNKNOWN";
}

void append_explicit_layer(GatCircuitPlan& plan, layerType type, std::size_t output_size, const ExplicitUnaryGateBuffer& unary,
                           const ExplicitBinaryGateBuffer& binary, bool need_phase2 = false) {
    if (plan.circuit.circuit.size() >= std::numeric_limits<u8>::max()) throw std::overflow_error("zkGPT circuit has too many layers");
    if (active_gkr_point_shape_collector != nullptr) {
        active_gkr_point_shape_collector->append_explicit_layer(
            type, output_size, unary.size(), binary.size());
        return;
    }
    layer current;
    current.ty = type;
    current.size = as_u64(std::max<std::size_t>(1, output_size), "explicit layer size");
    current.zero_start_id = current.size;
    current.bit_length = bit_length(current.size);
    current.need_phase2 = need_phase2;
    const std::size_t input_size = plan.circuit.circuit.front().size;
    current.reserve_gate_arrays(unary.size(), binary.size());
    current.reserve_coefficients(2);
    for (const ExplicitUnaryGate& gate : unary) {
        if (gate.g >= output_size || gate.u >= input_size) {
            std::ostringstream detail;
            const std::size_t source_layer_size = gate.lu == 0 ? input_size : plan.circuit.circuit.back().size;
            detail << "explicit unary gate coordinate is out of bounds: layer=" << plan.circuit.circuit.size()
                   << " name=" << explicit_layer_type_name(type) << " output_size=" << output_size
                   << " source_layer_size=" << source_layer_size << " gate=(g=" << gate.g << ",u=" << gate.u
                   << ",lu=" << static_cast<unsigned>(gate.lu) << ')';
            throw std::invalid_argument(detail.str());
        }
        current.add_unary(as_u64(gate.g, "explicit unary output"), as_u64(gate.u, "explicit unary input"), gate.lu, gate.sc);
    }
    for (const ExplicitBinaryGate& gate : binary) {
        if (gate.g >= output_size || gate.u >= input_size || gate.v >= input_size)
            throw std::invalid_argument("explicit binary gate coordinate is out of bounds");
        current.add_binary(as_u64(gate.g, "explicit binary output"), as_u64(gate.u, "explicit binary lhs"), as_u64(gate.v, "explicit binary rhs"),
                           gate.sc, gate.l);
    }
    if (current.uni_gates.empty() && current.bin_gates.empty()) throw std::invalid_argument("explicit layered relation has no gates");
    plan.circuit.circuit.push_back(std::move(current));
}

void append_fconn_claim_link(GatCircuitPlan& plan, std::size_t descriptor_index,
                             const std::string& relation_id, const std::string& family_id,
                             const std::string& lhs_segment, const std::string& rhs_segment,
                             const std::string& output_segment, std::size_t lhs_offset,
                             std::size_t rhs_offset, std::size_t output_offset,
                             std::size_t rows, std::size_t inner, std::size_t cols) {
    if (relation_id.empty() || family_id.empty() || lhs_segment.empty() || rhs_segment.empty() ||
        output_segment != "Aux" || rows == 0 || inner == 0 || cols == 0)
        throw std::invalid_argument("canonical FCONN claim link identity or dimensions are invalid");
    FconnClaimLink link;
    link.descriptor_index = descriptor_index;
    link.relation_id = relation_id;
    link.family_id = family_id;
    link.lhs_segment = lhs_segment;
    link.rhs_segment = rhs_segment;
    link.output_segment = output_segment;
    link.lhs_offset = lhs_offset;
    link.rhs_offset = rhs_offset;
    link.output_offset = output_offset;
    link.rows = rows;
    link.inner = inner;
    link.cols = cols;
    plan.fconn_claim_links.push_back(std::move(link));
}

void append_fconn_claim_link(GatCircuitPlan& plan, const MatMulDescriptor& descriptor) {
    if (plan.matmuls.empty() || plan.matmuls.size() != plan.fconn_claim_links.size() + 1 ||
        plan.matmuls.back().relation_id != descriptor.relation_id)
        throw std::logic_error("canonical FCONN claim link is not in MatMul descriptor order");
    append_fconn_claim_link(plan, plan.matmuls.size() - 1, descriptor.relation_id,
                            descriptor.family_id, descriptor.lhs_segment,
                            descriptor.rhs_segment, descriptor.output_segment,
                            descriptor.lhs_offset, descriptor.rhs_offset,
                            descriptor.output_offset, descriptor.rows,
                            descriptor.inner, descriptor.cols);
}

void append_projection_layer(GatCircuitPlan& plan, const MatMulDescriptor& descriptor) {
    layer current;
    current.ty = layerType::FCONN;
    current.size = as_u64(fconn_output_domain(descriptor.rows, descriptor.cols), "projection layer size");
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

    current.native_relation_id = descriptor.relation_id;
    current.native_family_id = descriptor.family_id;
    current.native_rows = as_u64(descriptor.rows, "projection rows");
    current.native_inner = as_u64(descriptor.inner, "projection inner");
    current.native_cols = as_u64(descriptor.cols, "projection cols");
    current.native_lhs_offset = descriptor.lhs_offset;
    current.native_rhs_offset = descriptor.rhs_offset;
    current.native_output_offset = descriptor.output_offset;
    if (active_gkr_point_shape_collector != nullptr) {
        active_gkr_point_shape_collector->append_fconn(current);
        if (active_gkr_point_shape_collector->symbolic()) {
            active_gkr_point_shape_collector->append_symbolic_explicit_layer(
                layerType::ADD_BIAS, descriptor.output_coordinates,
                checked_mul(descriptor.output_coordinates, 2, "symbolic FCONN binding gate count"),
                0, descriptor.output_coordinates, 0, true, false, false);
            return;
        }
        ExplicitUnaryGateBuffer unary;
        ExplicitBinaryGateBuffer binary;
        unary.reserve(checked_mul(descriptor.output_coordinates, 2,
                                  "shape-only FCONN binding gate count"));
        for (std::size_t index = 0; index < descriptor.output_coordinates; ++index) {
            unary.push_back({index, fconn_padded_output_index(descriptor.rows, descriptor.cols, index), 1, 1});
            unary.push_back({index, descriptor.output_offset + index, 0, -1});
        }
        append_explicit_layer(plan, layerType::ADD_BIAS,
                              std::max<std::size_t>(1, descriptor.output_coordinates),
                              unary, binary, false);
        return;
    }
    plan.circuit.circuit.push_back(std::move(current));

    layer binding;
    binding.ty = layerType::ADD_BIAS;
    binding.size = as_u64(std::max<std::size_t>(1, descriptor.output_coordinates), "FCONN binding layer size");
    binding.zero_start_id = binding.size;
    binding.bit_length = bit_length(binding.size);
    binding.need_phase2 = false;
    binding.reserve_gate_arrays(checked_mul(descriptor.output_coordinates, 2, "FCONN binding gate count"), 0);
    binding.reserve_coefficients(2);
    for (std::size_t index = 0; index < descriptor.output_coordinates; ++index) {
        binding.add_unary(as_u64(index, "FCONN binding output"),
                          as_u64(fconn_padded_output_index(descriptor.rows, descriptor.cols, index),
                                 "FCONN binding previous output"),
                          1, 1);
        binding.add_unary(as_u64(index, "FCONN binding output"), as_u64(descriptor.output_offset + index, "FCONN binding input coordinate"), 0, -1);
    }
    plan.circuit.circuit.push_back(std::move(binding));
}

using LinearTerm = std::pair<std::size_t, ll>;

struct ProductTerm {
    std::size_t left = 0;
    std::size_t right = 0;
    ll coefficient = 1;
};

struct RelationLayerBuilder {
    std::size_t next_output = 0;
    std::size_t output_capacity = 0;
    std::size_t target_capacity = 0;
    std::size_t unary_capacity = 0;
    std::size_t binary_capacity = 0;
    std::size_t coefficient_capacity = 0;
    std::size_t emitted_targets = 0;
    std::size_t emitted_unary = 0;
    std::size_t emitted_binary = 0;
    bool shape_only = false;
    PackedResidualTargets targets;
    layer relation;

    void reserve(std::size_t output_count, std::size_t residual_count, std::size_t unary_count, std::size_t binary_count,
                 std::size_t coefficient_count) {
        output_capacity = output_count;
        target_capacity = residual_count;
        unary_capacity = unary_count;
        binary_capacity = binary_count;
        coefficient_capacity = coefficient_count;
        shape_only = active_gkr_point_shape_collector != nullptr;
        if (shape_only) return;
        targets.reserve(residual_count);
        if (residual_count != 0 && targets.capacity() != residual_count)
            throw std::length_error("canonical GAT residual target capacity differs from its exact source count");
        relation.reserve_gate_arrays(unary_count, binary_count);
        relation.reserve_coefficients(coefficient_count);
    }

    std::size_t add_public_output(std::size_t input) {
        if (next_output >= output_capacity || emitted_unary >= unary_capacity)
            throw std::length_error("canonical GAT relation exceeded its reserved public-output profile");
        const std::size_t output = next_output++;
        if (shape_only) {
            active_gkr_point_shape_collector->append_unary(output, input, 0);
        } else {
            relation.add_unary(as_u64(output, "GAT public output residual"), as_u64(input, "GAT public output input"), 0, 1);
            if (relation.coefficient_pool.size() > coefficient_capacity)
                throw std::length_error("canonical GAT relation exceeded its reserved coefficient profile");
        }
        ++emitted_unary;
        return output;
    }

    std::size_t add_residual(const std::vector<LinearTerm>& linear, const std::vector<ProductTerm>& products, Int target = 0) {
        if (next_output >= output_capacity || emitted_targets >= target_capacity ||
            emitted_unary > unary_capacity || linear.size() > unary_capacity - emitted_unary ||
            emitted_binary > binary_capacity || products.size() > binary_capacity - emitted_binary)
            throw std::length_error("canonical GAT relation exceeded its reserved residual profile");
        const std::size_t output = next_output++;
        ++emitted_targets;
        if (linear.empty() && products.empty()) throw std::invalid_argument("empty GAT arithmetic residual");
        if (shape_only) {
            for (const LinearTerm& term : linear)
                active_gkr_point_shape_collector->append_unary(output, term.first, 0);
            for (const ProductTerm& term : products)
                active_gkr_point_shape_collector->append_binary(output, term.left, term.right, 0);
        } else {
            targets.push_back(target);
            for (const LinearTerm& term : linear)
                relation.add_unary(as_u64(output, "GAT residual output"), as_u64(term.first, "GAT residual input"), 0, term.second);
            for (const ProductTerm& term : products)
                relation.add_binary(as_u64(output, "GAT residual output"), as_u64(term.left, "GAT residual lhs"), as_u64(term.right, "GAT residual rhs"), term.coefficient, 0);
            if (relation.coefficient_pool.size() > coefficient_capacity)
                throw std::length_error("canonical GAT relation exceeded its reserved coefficient profile");
        }
        emitted_unary += linear.size();
        emitted_binary += products.size();
        (void)target;
        return output;
    }

    void add_range_decomposition(std::size_t source, std::size_t range_begin,
                                 unsigned width) {
        if (!shape_only) throw std::logic_error("range decomposition shortcut used outside shape admission");
        if (width == 0 || width > (std::numeric_limits<std::size_t>::max() - 1) / 2)
            throw std::length_error("shape-only range decomposition width is invalid");
        const std::size_t unary_count = 2 * static_cast<std::size_t>(width) + 1;
        const std::size_t binary_count = static_cast<std::size_t>(width);
        const std::size_t target_count = static_cast<std::size_t>(width) + 1;
        if (next_output > output_capacity || target_count > output_capacity - next_output ||
            emitted_targets > target_capacity || target_count > target_capacity - emitted_targets ||
            emitted_unary > unary_capacity || unary_count > unary_capacity - emitted_unary ||
            emitted_binary > binary_capacity || binary_count > binary_capacity - emitted_binary)
            throw std::length_error("canonical GAT relation exceeded its reserved range profile");
        active_gkr_point_shape_collector->append_range_decomposition(
            next_output, source, range_begin, width, unary_count, binary_count);
        next_output += target_count;
        emitted_targets += target_count;
        emitted_unary += unary_count;
        emitted_binary += binary_count;
    }

    void set_symbolic_counts() {
        if (!shape_only || active_gkr_point_shape_collector == nullptr ||
            !active_gkr_point_shape_collector->symbolic())
            throw std::logic_error("symbolic relation counts require symbolic shape-only planning");
        next_output = output_capacity;
        emitted_targets = target_capacity;
        emitted_unary = unary_capacity;
        emitted_binary = binary_capacity;
    }

    std::size_t add_unary(std::size_t input, ll coefficient = 1) { return add_residual({{input, coefficient}}, {}); }
};

std::size_t relation_coefficient_candidate_count(const QuantConfig& quant, const StaticLookupBounds& bounds) {
    std::set<ll> candidates{0, 1, -1, 2, -2};
    for (unsigned bit = 0; bit < 63; ++bit) candidates.insert(static_cast<ll>(std::uint64_t(1) << bit));
    const auto add_signed_pair = [&candidates](Int value) {
        const ll coefficient = static_cast<ll>(value);
        candidates.insert(coefficient);
        if (coefficient != std::numeric_limits<ll>::min()) candidates.insert(-coefficient);
    };
    add_signed_pair(quant.s_input);
    add_signed_pair(quant.s_delta);
    add_signed_pair(quant.s_exp);
    for (Int value : quant.s_h) add_signed_pair(value);
    for (const auto& values : quant.s_w_hidden) for (Int value : values) add_signed_pair(value);
    for (const auto& values : quant.s_hprime_hidden) for (Int value : values) add_signed_pair(value);
    for (const auto& values : quant.s_a_src_hidden) for (Int value : values) add_signed_pair(value);
    for (const auto& values : quant.s_a_dst_hidden) for (Int value : values) add_signed_pair(value);
    for (Int value : quant.s_agg_pre) add_signed_pair(value);
    for (Int value : quant.s_agg) add_signed_pair(value);
    for (Int value : quant.s_cat) add_signed_pair(value);
    add_signed_pair(quant.s_out_pre);
    add_signed_pair(quant.s_out_head);
    add_signed_pair(quant.s_out);
    for (Int value : quant.s_w_output) add_signed_pair(value);
    for (Int value : quant.s_yprime_output) add_signed_pair(value);
    for (Int value : quant.s_a_src_output) add_signed_pair(value);
    for (Int value : quant.s_a_dst_output) add_signed_pair(value);
    for (Int value : quant.s_b_output) add_signed_pair(value);
    add_signed_pair(quant.range_bits);
    add_signed_pair(quant.static_lrelu_min);
    add_signed_pair(quant.static_lrelu_max);
    add_signed_pair(quant.static_exp_min);
    add_signed_pair(quant.static_exp_max);
    add_signed_pair(quant.static_elu_min);
    add_signed_pair(quant.static_elu_max);
    add_signed_pair(quant.lrelu_slope_num);
    add_signed_pair(quant.lrelu_slope_den);
    const auto add_lookup_base = [&add_signed_pair](Int by) {
        const Wide base = 2 * static_cast<Wide>(by) + 1;
        if (base > std::numeric_limits<Int>::max()) throw std::overflow_error("lookup coefficient base exceeds the integer coefficient domain");
        add_signed_pair(static_cast<Int>(base));
    };
    add_lookup_base(bounds.lrelu.second);
    add_lookup_base(bounds.exp.second);
    add_lookup_base(bounds.elu.second);
    return candidates.size();
}

std::vector<std::pair<std::size_t, std::size_t>> incoming_ranges_for_plan(const Graph& graph) {
    std::vector<std::pair<std::size_t, std::size_t>> ranges(graph.num_nodes, {graph.edges(), graph.edges()});
    for (std::size_t edge = 0; edge < graph.edges(); ++edge) {
        const std::size_t destination = static_cast<std::size_t>(graph.dst[edge]);
        if (ranges[destination].first == graph.edges()) ranges[destination].first = edge;
        ranges[destination].second = edge + 1;
    }
    for (const auto& range : ranges)
        if (range.first == graph.edges()) throw std::invalid_argument("canonical GAT circuit has an empty destination group");
    return ranges;
}

void add_division_residuals(RelationLayerBuilder& builder, std::size_t division_offset, std::size_t output_offset, const std::vector<LinearTerm>& dividend,
                            const std::vector<LinearTerm>& denominator, const std::string& rounding) {
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

    if (rounding == "nearest_ties_away") {
        builder.add_residual({{a, 2}, {d, 1}, {r, -1}}, {{d, q, -2}});
    } else if (rounding == "toward_zero") {
        builder.add_residual({{a, 1}, {r, -1}}, {{d, q, -1}});
    } else {
        throw std::invalid_argument("unsupported canonical GAT rounding mode");
    }

    builder.add_residual({{sigma, -1}}, {{sigma, sigma, 1}});
    builder.add_residual({{zero, -1}}, {{zero, zero, 1}});
    builder.add_residual({}, {{zero, a, 1}});
    builder.add_residual({}, {{zero, sigma, 1}});
    builder.add_residual({}, {{zero, a_inverse, 1}});
    builder.add_residual({{zero, 1}}, {{a, a_inverse, 1}}, 1);
    builder.add_residual({}, {{d, d_inverse, 1}}, 1);
    builder.add_residual({{r, 1}, {delta, 1}, {d, rounding == "nearest_ties_away" ? -2 : -1}}, {}, -1);
}

void add_attention_residuals(RelationLayerBuilder& builder, const Graph& graph, const std::vector<std::pair<std::size_t, std::size_t>>& incoming,
                             const AttentionCoordinates& a, bool hidden, const QuantConfig& quant) {
    const std::size_t edges = graph.edges();
    const std::size_t output_dim = a.output_dim;
    const std::size_t coordinates = graph.num_nodes * output_dim;

    for (std::size_t edge = 0; edge < edges; ++edge) {
        const std::size_t source = static_cast<std::size_t>(graph.src[edge]);
        const std::size_t destination = static_cast<std::size_t>(graph.dst[edge]);
        builder.add_residual({{a.source_score + source, 1}, {a.destination_score + destination, 1}, {a.raw_score + edge, -1}}, {});
        builder.add_residual({{a.group_max + destination, 1}, {a.leaky_score + edge, -1}, {a.delta + edge, -1}}, {});
        builder.add_residual({}, {{a.max_selector + edge, a.delta + edge, 1}});
        builder.add_residual({{a.max_selector + edge, -1}}, {{a.max_selector + edge, a.max_selector + edge, 1}});
        builder.add_residual({{a.exp + edge, -1}}, {{a.alpha + edge, a.group_sum + destination, 1}});
        builder.add_residual({{a.alpha + edge, -1}}, {{a.exp + edge, a.group_inverse + destination, 1}});
    }

    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        std::vector<LinearTerm> sum_exp = {{a.group_sum + node, -1}};
        std::vector<ProductTerm> selected_max;
        std::vector<ProductTerm> selected_sum;
        for (std::size_t edge = incoming[node].first; edge < incoming[node].second; ++edge) {
            sum_exp.push_back({a.exp + edge, 1});
            selected_max.push_back({a.max_selector + edge, a.leaky_score + edge, 1});
            selected_sum.push_back({a.max_selector + edge, a.group_sum + node, 1});
        }
        builder.add_residual(sum_exp, {});
        builder.add_residual({}, {{a.group_sum + node, a.group_inverse + node, 1}}, 1);
        builder.add_residual({{a.group_max + node, -1}}, selected_max);
        builder.add_residual({{a.group_sum + node, -1}}, selected_sum);
        for (std::size_t col = 0; col < output_dim; ++col) {
            const std::size_t coordinate = node * output_dim + col;
            std::vector<LinearTerm> integer_relation = {{a.integer_numerator + coordinate, -1}};
            std::vector<ProductTerm> integer_products;
            std::vector<ProductTerm> field_products;
            for (std::size_t edge = incoming[node].first; edge < incoming[node].second; ++edge) {
                const std::size_t source = static_cast<std::size_t>(graph.src[edge]);
                integer_products.push_back({a.exp + edge, a.projection + source * output_dim + col, 1});
                field_products.push_back({a.alpha + edge, a.projection + source * output_dim + col, 1});
            }
            builder.add_residual(integer_relation, integer_products);
            builder.add_residual({{a.field_numerator + coordinate, 1}, {a.integer_numerator + coordinate, -1}}, {});
            std::vector<LinearTerm> field_relation = {{a.field_numerator + coordinate, -1}};
            builder.add_residual(field_relation, {{a.field_aggregate + coordinate, a.group_sum + node, 1}});
        }
    }

    (void)hidden;
    (void)quant;
    (void)coordinates;
}

void add_nonnegative_range_residuals(RelationLayerBuilder& builder, std::size_t source, ll coefficient, Int constant, std::size_t& range_cursor, unsigned width = 63) {
    if (builder.shape_only) {
        builder.add_range_decomposition(source, range_cursor, width);
        range_cursor = checked_add(range_cursor, static_cast<std::size_t>(width),
                                   "shape-only range query cursor");
        (void)coefficient;
        (void)constant;
        return;
    }
    std::vector<LinearTerm> decomposition = {{source, -coefficient}};
    for (unsigned bit = 0; bit < width; ++bit) {
        decomposition.emplace_back(range_cursor, static_cast<ll>(1ULL << bit));
        builder.add_residual({{range_cursor, -1}}, {{range_cursor, range_cursor, 1}});
        ++range_cursor;
    }
    builder.add_residual(decomposition, {}, constant);
}

void add_lookup_tuple_residuals(RelationLayerBuilder& builder, std::size_t x, std::size_t y, std::size_t query, const std::pair<Int, Int>& bounds, std::size_t& range_cursor) {
    const Int bx = bounds.first, by = bounds.second;
    const Wide wide_base = 2 * static_cast<Wide>(by) + 1, wide_shift = static_cast<Wide>(bx) * wide_base + by;
    if (wide_base > std::numeric_limits<Int>::max() || wide_shift > std::numeric_limits<Int>::max()) throw std::overflow_error("authenticated lookup packing exceeds the integer coefficient domain");
    const Int base = static_cast<Int>(wide_base), shift = static_cast<Int>(wide_shift);
    builder.add_residual({{query, 1}, {x, -base}, {y, -1}}, {}, shift);
    add_nonnegative_range_residuals(builder, x, 1, bx, range_cursor, nonnegative_bit_width(2 * static_cast<Wide>(bx)));
    add_nonnegative_range_residuals(builder, x, -1, bx, range_cursor, nonnegative_bit_width(2 * static_cast<Wide>(bx)));
    add_nonnegative_range_residuals(builder, y, 1, by, range_cursor, nonnegative_bit_width(2 * static_cast<Wide>(by)));
    add_nonnegative_range_residuals(builder, y, -1, by, range_cursor, nonnegative_bit_width(2 * static_cast<Wide>(by)));
}

void add_division_range_residuals(RelationLayerBuilder& builder, std::size_t offset, std::size_t count, std::size_t& range_cursor, const std::array<unsigned, 5>& profile) {
    const std::array<std::size_t, 5> components{{2, 3, 4, 5, 8}};
    for (std::size_t index = 0; index < count; ++index) for (std::size_t item = 0; item < components.size(); ++item) add_nonnegative_range_residuals(builder, offset + index * 9 + components[item], 1, 0, range_cursor, profile[item]);
}

void add_attention_lookup_residuals(RelationLayerBuilder& builder, const Graph& graph, const AttentionCoordinates& a, bool hidden, const StaticLookupBounds& bounds, std::size_t& lrelu_cursor, std::size_t& exp_cursor, std::size_t& elu_cursor, std::size_t& range_cursor, const std::array<unsigned, 5>& profile) {
    for (std::size_t edge = 0; edge < graph.edges(); ++edge) add_lookup_tuple_residuals(builder, a.raw_score + edge, a.leaky_score + edge, lrelu_cursor++, bounds.lrelu, range_cursor);
    for (std::size_t edge = 0; edge < graph.edges(); ++edge) add_lookup_tuple_residuals(builder, a.delta + edge, a.exp + edge, exp_cursor++, bounds.exp, range_cursor);
    if (hidden) for (std::size_t index = 0; index < graph.num_nodes * a.output_dim; ++index) add_lookup_tuple_residuals(builder, a.recovered + index, a.activated + index, elu_cursor++, bounds.elu, range_cursor);
    else for (std::size_t index = 0; index < graph.num_nodes * a.output_dim; ++index) builder.add_residual({{a.activated + index, 1}, {a.recovered + index, -1}}, {});
    for (std::size_t edge = 0; edge < graph.edges(); ++edge) add_nonnegative_range_residuals(builder, a.delta + edge, 1, 0, range_cursor, nonnegative_bit_width(2 * static_cast<Wide>(bounds.lrelu.second)));
    add_division_range_residuals(builder, a.divisions, graph.num_nodes * a.output_dim, range_cursor, profile);
}

void add_attention_recovery_residuals(RelationLayerBuilder& builder, const Graph& graph, const AttentionCoordinates& a, bool hidden, std::size_t layer_id,
                                      std::size_t head_id, const QuantConfig& quant) {
    const Int dividend_scale = hidden ? quant.s_agg_pre[layer_id] : quant.s_out_pre;
    const Int denominator_scale = hidden ? quant.s_hprime_hidden[layer_id][head_id] : quant.s_yprime_output[head_id];
    const std::size_t coordinates = graph.num_nodes * a.output_dim;
    for (std::size_t index = 0; index < coordinates; ++index) {
        add_division_residuals(builder, a.divisions + index * 9, a.recovered + index, {{a.integer_numerator + index, dividend_scale}},
                               {{a.group_sum + index / a.output_dim, denominator_scale}}, quant.rounding);
    }
}

void finish_relation_layer(GatCircuitPlan& plan, RelationLayerBuilder& builder, std::size_t public_output_coordinates) {
    if (builder.next_output < public_output_coordinates) throw std::logic_error("GAT relation layer output count underflow");
    if (builder.targets.size() != builder.next_output - public_output_coordinates)
        throw std::logic_error("GAT relation residual target count does not match the public-output prefix");

    layer relation = std::move(builder.relation);
    relation.ty = layerType::RELU;
    relation.size = as_u64(builder.next_output, "GAT relation layer size");
    relation.zero_start_id = relation.size;
    relation.bit_length = bit_length(relation.size);
    relation.need_phase2 = false;
    plan.circuit.circuit.push_back(std::move(relation));
    plan.circuit.size = static_cast<u8>(plan.circuit.circuit.size());
    plan.public_output_coordinates = public_output_coordinates;
    plan.residual_coordinates = builder.next_output - public_output_coordinates;
    plan.residual_targets = std::move(builder.targets);
    plan.residual_targets.freeze();
    cybozu::Sha256 target_hasher;
    const std::string domain = "zkgpt-gat-public-residual-targets-v1";
    target_hasher.update(domain.data(), domain.size());
    for (Int target : plan.residual_targets) { std::array<unsigned char, 8> wire; for (unsigned bit = 0; bit < 8; ++bit) wire[bit] = static_cast<unsigned char>(static_cast<std::uint64_t>(target) >> (8 * bit)); target_hasher.update(wire.data(), wire.size()); }
    std::array<unsigned char, 32> digest;
    target_hasher.digest(digest.data(), digest.size(), nullptr, 0);
    std::ostringstream encoded;
    encoded << std::hex << std::setfill('0');
    for (unsigned char byte : digest) encoded << std::setw(2) << static_cast<unsigned>(byte);
    plan.residual_target_digest = encoded.str();
}

std::string plan_encoding(const GatCircuitPlan& plan) {
    if (!plan.native_proof_backing_is_resident())
        throw std::logic_error("cannot encode a GAT plan while its native proof backing is released");
    std::string encoded;
    append_string(encoded, plan.recipe_schema);
    append_string(encoded, plan.relation_schema);
    append_string(encoded, plan.sparse_relation_schema);
    append_u64(encoded, plan.sparse_relation_version);
    append_u64(encoded, plan.sparse_chunk_size);
    append_string(encoded, plan.sparse_plan_digest);
    encoded.push_back(static_cast<char>(plan.virtual_sparse ? 1 : 0));
    append_string(encoded, plan.model_arch_id);
    append_string(encoded, plan.model_shape.model_arch_id);
    append_u64(encoded, plan.model_shape.tensor_count);
    append_u64(encoded, plan.model_shape.layers);
    append_u64(encoded, plan.model_shape.output_heads);
    append_u64(encoded, plan.model_shape.classes);
    append_u64(encoded, plan.model_shape.input_dim);
    append_u64(encoded, plan.model_shape.hidden_dim);
    append_u64(encoded, plan.model_shape.tensors.size());
    for (const CanonicalTensorShape& tensor : plan.model_shape.tensors) {
        append_string(encoded, tensor.name);
        append_shape(encoded, tensor.shape);
    }
    append_u64(encoded, plan.graph_num_nodes);
    append_u64(encoded, plan.graph_num_features);
    append_u64(encoded, plan.graph_count);
    append_u64(encoded, plan.graph_num_classes);
    append_u64(encoded, plan.graph_edge_count);
    append_string(encoded, plan.graph_topology_digest);
    append_string(encoded, plan.quant_cfg_id);
    append_string(encoded, plan.quant_digest);
    append_string(encoded, plan.static_tables_id);
    append_string(encoded, plan.registry.registry_id);
    append_u64(encoded, static_cast<std::uint64_t>(plan.registry.descriptors.size()));
    for (const RelationDescriptor& descriptor : plan.registry.descriptors) {
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
    append_string(encoded, plan.residual_target_digest);
    append_u64(encoded, plan.division_range_bits.size());
    for (const auto& profile : plan.division_range_bits) for (unsigned width : profile) append_u64(encoded, width);
    append_u64(encoded, static_cast<std::uint64_t>(plan.terminal_openings.size()));
    for (const std::string& opening : plan.terminal_openings) append_string(encoded, opening);
    append_u64(encoded, plan.input_layout.n_raw);
    append_u64(encoded, plan.input_layout.n_in);
    append_u64(encoded, plan.input_layout.segments.size());
    for (const InputSegment& segment : plan.input_layout.segments) {
        append_string(encoded, segment.name);
        append_u64(encoded, segment.offset);
        append_u64(encoded, segment.logical_length);
        append_u64(encoded, segment.padded_length);
    }
    append_u64(encoded, static_cast<std::uint64_t>(plan.matmuls.size()));
    for (const MatMulDescriptor& descriptor : plan.matmuls) {
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
        append_string(encoded, descriptor.relation_schema);
        append_string(encoded, descriptor.lhs_segment);
        append_string(encoded, descriptor.rhs_segment);
        append_string(encoded, descriptor.output_segment);
        append_string(encoded, descriptor.lhs_coordinate_rule);
        append_string(encoded, descriptor.rhs_coordinate_rule);
        append_string(encoded, descriptor.output_coordinate_rule);
        append_string(encoded, descriptor.terminal_claim_rule);
    }
    append_u64(encoded, static_cast<std::uint64_t>(plan.fconn_claim_links.size()));
    for (const FconnClaimLink& link : plan.fconn_claim_links) {
        append_u64(encoded, link.descriptor_index);
        append_string(encoded, link.relation_id);
        append_string(encoded, link.family_id);
        append_string(encoded, link.lhs_segment);
        append_string(encoded, link.rhs_segment);
        append_string(encoded, link.output_segment);
        append_u64(encoded, link.lhs_offset);
        append_u64(encoded, link.rhs_offset);
        append_u64(encoded, link.output_offset);
        append_u64(encoded, link.rows);
        append_u64(encoded, link.inner);
        append_u64(encoded, link.cols);
        append_string(encoded, link.lhs_point_rule);
        append_string(encoded, link.rhs_point_rule);
        append_string(encoded, link.output_point_rule);
        append_string(encoded, link.source_rule);
    }
    append_u64(encoded, static_cast<std::uint64_t>(plan.sparse_gkr.size()));
    for (const SparseGkrDescriptor& descriptor : plan.sparse_gkr) {
        append_string(encoded, descriptor.relation_id);
        append_string(encoded, descriptor.family_id);
        encoded.push_back(static_cast<char>(descriptor.relation_kind));
        append_string(encoded, descriptor.relation_schema);
        append_u64(encoded, descriptor.edge_count);
        append_u64(encoded, descriptor.value_width);
        append_u64(encoded, descriptor.chunk_size);
        append_u64(encoded, descriptor.input_offset);
        append_u64(encoded, descriptor.source_offset);
        append_u64(encoded, descriptor.destination_offset);
        append_u64(encoded, descriptor.output_offset);
        append_string(encoded, descriptor.coordinate_rule);
        append_string(encoded, descriptor.group_rule);
        append_string(encoded, descriptor.arithmetic_rule);
        append_string(encoded, descriptor.overflow_rule);
        append_string(encoded, descriptor.terminal_rule);
        append_u64(encoded, static_cast<std::uint64_t>(descriptor.constraints.size()));
        for (const std::string& constraint : descriptor.constraints) append_string(encoded, constraint);
    }
    append_u64(encoded, static_cast<std::uint64_t>(plan.native_relations.size()));
    for (const NativeRelationDescriptor& descriptor : plan.native_relations) {
        append_string(encoded, descriptor.relation_id);
        append_string(encoded, descriptor.family_id);
        append_string(encoded, descriptor.relation_schema);
        append_string(encoded, descriptor.primitive);
        append_string(encoded, descriptor.domain_rule);
        append_string(encoded, descriptor.selector_rule);
        append_string(encoded, descriptor.coordinate_rule);
        append_string(encoded, descriptor.arithmetic_rule);
        append_string(encoded, descriptor.terminal_rule);
        append_u64(encoded, descriptor.edge_count);
        append_u64(encoded, descriptor.value_width);
        append_u64(encoded, descriptor.chunk_size);
        append_u64(encoded, descriptor.input_offset);
        append_u64(encoded, descriptor.output_offset);
        append_u64(encoded, static_cast<std::uint64_t>(descriptor.equations.size()));
        for (const std::string& equation : descriptor.equations) append_string(encoded, equation);
    }
    append_u64(encoded, static_cast<std::uint64_t>(plan.lasso.size()));
    for (const LassoFamilyDescriptor& descriptor : plan.lasso) {
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
    for (const CheckpointDescriptor& checkpoint : plan.checkpoints) {
        append_string(encoded, checkpoint.id);
        append_string(encoded, checkpoint.relation_id);
        append_string(encoded, checkpoint.segment);
        append_u64(encoded, checkpoint.coordinates);
        append_u64(encoded, checkpoint.coordinate_offset);
        append_string(encoded, checkpoint.coordinate_rule);
    }
    append_u64(encoded, static_cast<std::uint64_t>(plan.native_api.size()));
    for (const std::string& api : plan.native_api) append_string(encoded, api);
    return encoded;
}

bool validate_gat_circuit_plan_impl(const GatCircuitPlan& plan, std::string* reason);

GatCircuitPlan build_virtual_sparse_plan(const Graph& graph, const Model& model, const QuantConfig& quant) {
    graph.validate();
    model.validate_schema();
    quant.validate();
    GatCircuitPlan plan;
    plan.recipe_schema = "canonical_plan_recipe_v3";
    plan.relation_schema = "zkgpt-gat-relation-v1";
    plan.sparse_relation_schema = kSparseGkrRelationSchema;
    plan.sparse_relation_version = kSparseGkrRelationVersion;
    plan.sparse_chunk_size = kSparseGkrChunkEntries;
    plan.virtual_sparse = true;
    plan.model_arch_id = model.manifest.model_arch_id;
    plan.model_shape = canonical_model_shape(model);
    plan.graph_num_nodes = graph.num_nodes;
    plan.graph_num_features = graph.num_features;
    plan.graph_count = graph.graph_count;
    plan.graph_num_classes = graph.num_classes;
    plan.graph_edge_count = graph.edges();
    plan.graph_topology_digest = canonical_graph_topology_digest(graph);
    plan.quant_cfg_id = quant.quant_cfg_id;
    plan.quant_digest = canonical_quant_digest(quant);
    plan.static_tables_id = StaticTables::build(quant).table_id;
    plan.registry = RelationRegistry::canonical();
    plan.input_layout = canonical_input_layout_impl(graph, model, quant);
    plan.opening_policy = "same_r_three_terminal_openings_v1";
    plan.terminal_openings = {"W@r", "Aux@r", "L_qry@r"};
    const TraceCoordinates coordinates = make_trace_coordinates(graph, model, plan.input_layout);
    const std::size_t classes = model.classes();
    std::vector<SparseGkrRelationSpec> sparse_specs;
    sparse_specs.reserve(3 * (2 * model.hidden_heads() + model.manifest.output_heads) + 5);
    auto add_spec = [&](const std::string& relation_id, const std::string& family_id, SparseGkrRelationKind kind, std::size_t items, std::size_t width,
                        std::size_t input_offset, std::size_t source_offset, std::size_t destination_offset, std::size_t output_offset,
                        const std::string& coordinate_rule, const std::string& group_rule, std::vector<std::string> equations) {
        SparseGkrRelationSpec spec;
        spec.relation_id = relation_id;
        spec.family_id = family_id;
        spec.kind = kind;
        spec.item_count = items;
        spec.value_width = width;
        spec.input_offset = input_offset;
        spec.source_offset = source_offset;
        spec.destination_offset = destination_offset;
        spec.output_offset = output_offset;
        spec.chunk_size = kSparseGkrChunkEntries;
        spec.coordinate_rule = coordinate_rule;
        spec.group_rule = group_rule;
        spec.equations = std::move(equations);
        sparse_specs.push_back(spec);
        SparseGkrDescriptor descriptor;
        descriptor.relation_id = relation_id;
        descriptor.family_id = family_id;
        descriptor.relation_kind = kind;
        descriptor.relation_schema = kSparseGkrRelationSchema;
        descriptor.edge_count = items;
        descriptor.value_width = width;
        descriptor.chunk_size = kSparseGkrChunkEntries;
        descriptor.input_offset = input_offset;
        descriptor.source_offset = source_offset;
        descriptor.destination_offset = destination_offset;
        descriptor.output_offset = output_offset;
        descriptor.coordinate_rule = coordinate_rule;
        descriptor.group_rule = group_rule;
        descriptor.arithmetic_rule = kSparseGkrArithmeticRule;
        descriptor.overflow_rule = kSparseGkrOverflowRule;
        descriptor.terminal_rule = "terminal-claim-chain/input-output-binding-v1";
        descriptor.constraints = sparse_specs.back().equations;
        plan.sparse_gkr.push_back(std::move(descriptor));
    };
    add_spec("binding/input_segments", "input", SparseGkrRelationKind::InputBinding, 4, plan.input_layout.n_in, plan.input_layout.segments[0].offset,
             plan.input_layout.segments[1].offset, plan.input_layout.segments[2].offset, plan.input_layout.segments[3].offset,
             "segment_selector_low_bits_then_local_u64_coordinate", plan.input_layout.selector,
             {"logical segment values plus implicit zero padding", "SegmentedInputOracle chunked fold binding"});
    auto add_fconn = [&](const std::string& family, std::size_t rows, std::size_t inner, std::size_t cols, std::size_t lhs_offset, std::size_t rhs_offset,
                         std::size_t output_offset) {
        const std::string id = "fconn/" + family;
        SparseGkrRelationSpec spec;
        spec.relation_id = id;
        spec.family_id = family;
        spec.kind = SparseGkrRelationKind::Fconn;
        spec.item_count = rows;
        spec.value_width = checked_mul(inner, cols, "virtual FCONN relation width");
        spec.input_offset = lhs_offset;
        spec.source_offset = rhs_offset;
        spec.output_offset = output_offset;
        spec.chunk_size = kSparseGkrChunkEntries;
        spec.coordinate_rule = "row_major_exact[row,term,col]";
        spec.group_rule = "descriptor_index_ordered_fconn_claims";
        spec.equations = {"FCONN lhs/rhs/output segment-local MLE claims", "FCONN terminal claim consumed by output binding"};
        sparse_specs.push_back(std::move(spec));
        SparseGkrDescriptor descriptor;
        descriptor.relation_id = id;
        descriptor.family_id = family;
        descriptor.relation_kind = SparseGkrRelationKind::Fconn;
        descriptor.relation_schema = kSparseGkrRelationSchema;
        descriptor.edge_count = rows;
        descriptor.value_width = checked_mul(inner, cols, "virtual FCONN descriptor width");
        descriptor.chunk_size = kSparseGkrChunkEntries;
        descriptor.input_offset = lhs_offset;
        descriptor.source_offset = rhs_offset;
        descriptor.output_offset = output_offset;
        descriptor.coordinate_rule = "row_major_exact[row,term,col]";
        descriptor.group_rule = "descriptor_index_ordered_fconn_claims";
        descriptor.arithmetic_rule = kSparseGkrArithmeticRule;
        descriptor.overflow_rule = kSparseGkrOverflowRule;
        descriptor.terminal_rule = "terminal-claim-chain/input-output-binding-v1";
        descriptor.constraints = sparse_specs.back().equations;
        plan.sparse_gkr.push_back(std::move(descriptor));
        const InputSegment* lhs_segment = canonical_segment_for_span(
            plan.input_layout, lhs_offset, checked_mul(rows, inner, "virtual FCONN lhs span"));
        const InputSegment* rhs_segment = canonical_segment_for_span(
            plan.input_layout, rhs_offset, checked_mul(inner, cols, "virtual FCONN rhs span"));
        const InputSegment* output_segment = canonical_segment_for_span(
            plan.input_layout, output_offset, checked_mul(rows, cols, "virtual FCONN output span"));
        if (lhs_segment == nullptr || rhs_segment == nullptr || output_segment == nullptr)
            throw std::invalid_argument("virtual FCONN claim link exceeds a canonical input segment");
        append_fconn_claim_link(plan, plan.fconn_claim_links.size(), family, family,
                                lhs_segment->name, rhs_segment->name,
                                output_segment->name, lhs_offset, rhs_offset,
                                output_offset, rows, inner, cols);
    };

    std::size_t virtual_weight_offset = plan.input_layout.segments[1].offset;
    for (std::size_t layer = 0; layer < 2; ++layer) {
        const std::size_t dim = model.tensor(hidden_tensor_name(layer, 0, "seq/kernel")).shape.dims.at(2);
        const std::size_t inner = model.tensor(hidden_tensor_name(layer, 0, "seq/kernel")).shape.dims.at(1);
        const std::size_t lhs = layer == 0 ? plan.input_layout.segments[0].offset : coordinates.hidden_concat[layer - 1];
        for (std::size_t head = 0; head < model.hidden_heads(); ++head) {
            const std::string family = hidden_family(layer, head);
            add_fconn(family + "/projection", graph.num_nodes, inner, dim, lhs, virtual_weight_offset, coordinates.hidden[layer][head].projection);
            virtual_weight_offset = checked_add(virtual_weight_offset, checked_mul(inner, dim, "virtual hidden weight cursor"), "virtual hidden weight cursor");
            add_spec("attention.score/" + family, family, SparseGkrRelationKind::AttentionScore, graph.edges(), dim, coordinates.hidden[layer][head].raw_score,
                     coordinates.hidden[layer][head].source_score, coordinates.hidden[layer][head].destination_score, coordinates.hidden[layer][head].raw_score,
                     "edge_gid_then_dst_stable[e].src=src[e],dst=dst[e]", "Q_grp_new/Q_grp_end public graph recurrence",
                     {"raw_score=E_src[src]+E_dst[dst]", "leaky_score=T_LReLU(raw_score)"});
            add_spec("attention.softmax/" + family, family, SparseGkrRelationKind::MaskedSoftmax, graph.edges(), dim,
                     coordinates.hidden[layer][head].leaky_score, coordinates.hidden[layer][head].group_max, coordinates.hidden[layer][head].group_sum,
                     coordinates.hidden[layer][head].alpha, "edge_gid_then_dst_stable[e].dst=dst[e]", "Q_grp_new/Q_grp_end public destination recurrence",
                     {"group_max=max(leaky_score) by dst", "exp=T_exp(group_max-leaky_score)", "alpha=exp*group_inverse[dst]"});
            add_spec("aggregation.hidden/" + family, family, SparseGkrRelationKind::Aggregation, graph.edges(), dim, coordinates.hidden[layer][head].alpha,
                     coordinates.hidden[layer][head].projection, coordinates.hidden[layer][head].integer_numerator, coordinates.hidden[layer][head].recovered,
                     "edge_gid_then_dst_stable[e].src=src[e],dst=dst[e],col-major-inner", "Q_grp_new/Q_grp_end public destination recurrence",
                     {"integer_numerator=sum(alpha*projection[src])", "field_aggregate*group_sum=field_numerator", "recovered=SignedRoundDiv(...)"});
        }
    }
    for (std::size_t head = 0; head < model.manifest.output_heads; ++head) {
        const std::string family = output_family(head);
        const std::size_t dim = classes;
        const std::size_t inner = model.tensor(output_tensor_name(head, "seq/kernel")).shape.dims.at(1);
        add_fconn(family + "/projection", graph.num_nodes, inner, dim, coordinates.hidden_concat[1], virtual_weight_offset,
                  coordinates.output[head].projection);
        virtual_weight_offset = checked_add(virtual_weight_offset, checked_mul(inner, dim, "virtual output weight cursor"), "virtual output weight cursor");
        add_spec("attention.score/" + family, family, SparseGkrRelationKind::AttentionScore, graph.edges(), dim, coordinates.output[head].raw_score,
                 coordinates.output[head].source_score, coordinates.output[head].destination_score, coordinates.output[head].raw_score,
                 "edge_gid_then_dst_stable[e].src=src[e],dst=dst[e]", "Q_grp_new/Q_grp_end public graph recurrence",
                 {"raw_score=E_src[src]+E_dst[dst]", "leaky_score=T_LReLU(raw_score)"});
        add_spec("attention.softmax/" + family, family, SparseGkrRelationKind::MaskedSoftmax, graph.edges(), dim, coordinates.output[head].leaky_score,
                 coordinates.output[head].group_max, coordinates.output[head].group_sum, coordinates.output[head].alpha,
                 "edge_gid_then_dst_stable[e].dst=dst[e]", "Q_grp_new/Q_grp_end public destination recurrence",
                 {"group_max=max(leaky_score) by dst", "exp=T_exp(group_max-leaky_score)", "alpha=exp*group_inverse[dst]"});
        add_spec("aggregation.output/" + family, family, SparseGkrRelationKind::Aggregation, graph.edges(), dim, coordinates.output[head].alpha,
                 coordinates.output[head].projection, coordinates.output[head].integer_numerator, coordinates.output[head].recovered,
                 "edge_gid_then_dst_stable[e].src=src[e],dst=dst[e],col-major-inner", "Q_grp_new/Q_grp_end public destination recurrence",
                 {"integer_numerator=sum(alpha*projection[src])", "field_aggregate*group_sum=field_numerator", "recovered=SignedRoundDiv(...)"});
    }
    add_spec("rounding/signed_round_div", "rounding", SparseGkrRelationKind::SignedRoundDiv, graph.num_nodes, classes, coordinates.final_average_divisions,
             coordinates.head_sum, coordinates.final_logits, coordinates.final_logits, "node_gid_then_class_stable[node,class,witness_component]",
             "node-local-fixed-width", {"sigma,z boolean", "inverse and zero-product", "SignedRoundDiv nearest-ties-away terminal equation"});
    add_spec("binding/output_aux", "output", SparseGkrRelationKind::OutputBinding, graph.num_nodes, classes, coordinates.final_logits, coordinates.head_sum,
             coordinates.final_logits, coordinates.final_logits, "Aux/output[row_major_exact,node,class]", "public-Y-prefix-plus-residual-tail",
             {"Aux terminal claims bind public output Y", "residual terminal claims bind canonical zero tail"});
    SparseGkrPlan sparse_plan = build_sparse_gkr_plan(graph, plan.graph_topology_digest, plan.static_tables_id, sparse_specs);
    sparse_plan.claim_links.clear();
    for (std::size_t index = 1; index < sparse_plan.relations.size(); ++index) {
        SparseGkrClaimLink link;
        link.from_relation = sparse_plan.relations[index - 1].relation_id;
        link.to_relation = sparse_plan.relations[index].relation_id;
        link.claim_name = "terminal->initial/" + link.from_relation;
        sparse_plan.claim_links.push_back(std::move(link));
    }
    for (const FconnClaimLink& link : plan.fconn_claim_links) {
        SparseGkrClaimLink binding;
        binding.from_relation = "fconn/" + link.relation_id;
        binding.to_relation = "binding/output_aux";
        binding.claim_name = "FCONN:" + link.family_id + ":" + std::to_string(link.output_offset);
        binding.from_terminal = static_cast<std::uint32_t>(link.descriptor_index);
        sparse_plan.binding_links.push_back(std::move(binding));
    }

    const QueryCounts virtual_query_counts = canonical_query_counts(graph, model, quant);
    const QueryCoordinates virtual_query_coordinates = make_query_coordinates(graph, model, quant, plan.input_layout);
    const StaticTables virtual_tables = StaticTables::build(quant);
    const auto add_virtual_lasso = [&plan](const std::string& family, const std::string& table, std::size_t real, std::size_t offset, std::size_t table_domain,
                                           Int padding) {
        LassoFamilyDescriptor descriptor;
        descriptor.relation_id = "lasso/" + family;
        descriptor.family_id = family;
        descriptor.table_id = table;
        descriptor.real_queries = real;
        descriptor.padded_queries = next_power_of_two(real);
        descriptor.table_domain = next_power_of_two(table_domain);
        descriptor.padding_value = padding;
        descriptor.query_offset = offset;
        descriptor.incidence_rule = "L_qry/" + family + "/canonical_operation_index_row_major";
        descriptor.challenge_policy = "LC_before_challenge";
        descriptor.query_encoding = "lookup_tuple_encoding_id:packed_signed_pair_v1";
        plan.lasso.push_back(std::move(descriptor));
    };
    add_virtual_lasso("function/lrelu", "T_LReLU", virtual_query_counts.lrelu, virtual_query_coordinates.lrelu, virtual_tables.lrelu.size(),
                      virtual_tables.lrelu.front().x);
    add_virtual_lasso("function/exp", "T_exp", virtual_query_counts.exp, virtual_query_coordinates.exp, virtual_tables.exp.size(),
                      virtual_tables.exp.front().x);
    add_virtual_lasso("function/elu", "T_ELU", virtual_query_counts.elu, virtual_query_coordinates.elu, virtual_tables.elu.size(),
                      virtual_tables.elu.front().x);
    add_virtual_lasso("range/all", "T_range", virtual_query_counts.range, virtual_query_coordinates.range, 2, 0);
    plan.sparse_plan_digest = canonical_digest64(sparse_plan.encoding());
    plan.public_output_coordinates = checked_mul(graph.num_nodes, classes, "virtual public output coordinates");
    plan.residual_coordinates = plan.public_output_coordinates;
    plan.native_api = {"SparseGKR::prove_stream", "SparseGKR::verify_stream", "SegmentedInputOracle::fold_chunks", "NativeMaskedSumcheckProver",
                       "NativeMaskedSumcheckVerifier"};
    plan.plan_digest = canonical_digest64(plan_encoding(plan));
    plan.circuit_id = "R_GAT-zkGPT/" + plan.plan_digest;
    plan.official_layered = true;
    plan.has_placeholder = false;
    std::string reason;
    if (!validate_gat_circuit_plan_impl(plan, &reason)) throw std::invalid_argument("virtual sparse GAT plan validation failed: " + reason);
    return plan;
}

SparseGkrPlan sparse_plan_from_virtual_recipe(const Graph& graph, const GatCircuitPlan& plan) {
    if (!plan.virtual_sparse) throw std::invalid_argument("sparse relation recipe is not virtual");
    std::vector<SparseGkrRelationSpec> specs;
    specs.reserve(plan.sparse_gkr.size());
    for (const SparseGkrDescriptor& descriptor : plan.sparse_gkr) {
        SparseGkrRelationSpec spec;
        spec.relation_id = descriptor.relation_id;
        spec.family_id = descriptor.family_id;
        spec.kind = descriptor.relation_kind;
        spec.item_count = descriptor.edge_count;
        spec.value_width = descriptor.value_width;
        spec.input_offset = descriptor.input_offset;
        spec.source_offset = descriptor.source_offset;
        spec.destination_offset = descriptor.destination_offset;
        spec.output_offset = descriptor.output_offset;
        spec.chunk_size = descriptor.chunk_size;
        spec.coordinate_rule = descriptor.coordinate_rule;
        spec.group_rule = descriptor.group_rule;
        spec.arithmetic_rule = descriptor.arithmetic_rule;
        spec.overflow_rule = descriptor.overflow_rule;
        spec.terminal_rule = descriptor.terminal_rule;
        spec.equations = descriptor.constraints;
        specs.push_back(std::move(spec));
    }
    SparseGkrPlan sparse = build_sparse_gkr_plan(graph, plan.graph_topology_digest, plan.static_tables_id, specs);
    for (const FconnClaimLink& link : plan.fconn_claim_links) {
        SparseGkrClaimLink binding;
        binding.from_relation = "fconn/" + link.relation_id;
        binding.to_relation = "binding/output_aux";
        binding.claim_name = "FCONN:" + link.family_id + ":" + std::to_string(link.output_offset);
        binding.from_terminal = static_cast<std::uint32_t>(link.descriptor_index);
        sparse.binding_links.push_back(std::move(binding));
    }
    if (canonical_digest64(sparse.encoding()) != plan.sparse_plan_digest)
        throw std::invalid_argument("virtual sparse recipe digest does not match relation schedule");
    return sparse;
}

const AttentionTrace* sparse_find_attention(const ForwardResult& forward, const std::string& family) {
    for (const HiddenLayerTrace& layer : forward.trace.hidden_layers)
        for (const AttentionTrace& trace : layer.heads)
            if (trace.family_id == family) return &trace;
    for (const OutputHeadTrace& head : forward.trace.output_heads)
        if (head.attention.family_id == family) return &head.attention;
    return nullptr;
}

struct SparseRelationValueProvider {
    const Graph& graph;
    const Model& model;
    const QuantConfig& quant;
    const ForwardResult& forward;
    const GatCircuitPlan& plan;
    SegmentedInputOracle input_oracle;

    SparseRelationValueProvider(const Graph& graph_in, const Model& model_in, const QuantConfig& quant_in, const ForwardResult& forward_in,
                                const GatCircuitPlan& plan_in)
        : graph(graph_in), model(model_in), quant(quant_in), forward(forward_in), plan(plan_in), input_oracle(make_input_oracle()) {}

    SegmentedInputOracle make_input_oracle() const {
        std::array<std::vector<Fr>, 4> segments;
        for (Int value : forward.trace.input.values) segments[0].push_back(Fr(value));
        for (Int value : flatten_model(model, quant)) segments[1].push_back(Fr(value));
        { const NativeFieldVector aux = flatten_aux_values(forward); segments[2].assign(aux.begin(), aux.end()); }
        const InputSegment& l_segment = plan.input_layout.segments.at(3);
        segments[3].assign(l_segment.logical_length, Fr(0));
        return SegmentedInputOracle(plan.input_layout, std::move(segments));
    }

    const F& trace_field(const std::vector<F>& values, std::uint64_t index) const {
        static const F zero = F_ZERO;
        return index < values.size() ? values[static_cast<std::size_t>(index)] : zero;
    }

    F value(const SparseGkrRelationSpec& relation, std::uint64_t index) const {
        const std::uint64_t width = std::max<std::uint64_t>(1, relation.value_width);
        if (relation.kind == SparseGkrRelationKind::InputBinding) {
            const std::size_t segment = static_cast<std::size_t>(index / plan.input_layout.n_in);
            const std::size_t local = static_cast<std::size_t>(index % plan.input_layout.n_in);
            return segment < 4 ? input_oracle.segment_value(segment, local) : F_ZERO;
        }
        const std::uint64_t item = index / width;
        const std::uint64_t component = index % width;
        if (relation.kind == SparseGkrRelationKind::Fconn) {
            const std::size_t row = static_cast<std::size_t>(item);
            const std::size_t col = static_cast<std::size_t>(component);

            for (const FconnClaimLink& link : plan.fconn_claim_links) {
                if ("fconn/" + link.relation_id != relation.relation_id) continue;
                const std::size_t term = col / link.cols;
                const std::size_t out_col = col % link.cols;
                const F lhs = input_oracle.value_at(relation.input_offset + row * link.inner + term);
                return lhs * input_oracle.value_at(relation.source_offset + term * link.cols + out_col);
            }
            return F_ZERO;
        }
        const AttentionTrace* trace = sparse_find_attention(forward, relation.family_id);
        if (trace != nullptr) {
            switch (relation.kind) {
            case SparseGkrRelationKind::AttentionScore:
                return item * width + component < trace->raw_score.size() ? F(trace->raw_score[static_cast<std::size_t>(item * width + component)]) : F_ZERO;
            case SparseGkrRelationKind::MaskedSoftmax:
                return trace_field(trace->alpha, item * width + component);
            case SparseGkrRelationKind::Aggregation:
                return trace_field(trace->field_aggregate, item * width + component);
            default:
                break;
            }
        }
        if (relation.kind == SparseGkrRelationKind::SignedRoundDiv || relation.kind == SparseGkrRelationKind::OutputBinding) {
            const std::size_t coordinate = static_cast<std::size_t>(item * width + component);
            return coordinate < forward.trace.logits.values.size() ? F(forward.trace.logits.values[coordinate]) : F_ZERO;
        }
        return F_ZERO;
    }

    F sum(const SparseGkrRelationSpec& relation) const {
        std::uint64_t entries = 0;
        if (!sparse_gkr_checked_mul(relation.item_count, std::max<std::uint64_t>(1, relation.value_width), &entries))
            throw std::overflow_error("sparse relation sum domain overflows u64");
        F total = F_ZERO;
        for (std::uint64_t index = 0; index < entries; ++index) total += value(relation, index);
        return total;
    }

    F folded(const SparseGkrRelationSpec& relation, const std::vector<F>& prefix, F correction) const {
        std::uint64_t entries = 0;
        if (!sparse_gkr_checked_mul(relation.item_count, std::max<std::uint64_t>(1, relation.value_width), &entries))
            throw std::overflow_error("sparse relation fold domain overflows u64");
        std::uint64_t domain = 0;
        if (!sparse_gkr_next_power_of_two(entries, &domain)) throw std::overflow_error("sparse relation fold domain is not representable");
        F total = F_ZERO;
        for (std::uint64_t index = 0; index < domain; ++index) {
            F weight = F_ONE;
            for (std::size_t bit = 0; bit < prefix.size(); ++bit) {
                const bool set = ((index >> bit) & 1U) != 0;
                weight *= set ? prefix[bit] : (F_ONE - prefix[bit]);
            }
            F sample = index < entries ? value(relation, index) : F_ZERO;
            if (index == 0) sample += correction;
            total += sample * weight;
        }
        return total;
    }
};

struct SparseRelationRuntime {
    F initial_claim = F_ZERO;
    F correction = F_ZERO;
    std::vector<F> prefix;
    zkgpt::native_pedersen::ScalarOpening initial_opening;
    bool have_initial = false;
};

struct SparseProverRuntime {
    const SparseRelationValueProvider& provider;
    zkgpt::native_pedersen::PedersenBasis basis;
    std::map<std::string, SparseRelationRuntime> relations;
    F previous_terminal = F_ZERO;
    bool have_previous = false;
};

SparseGkrProverCallbacks make_sparse_prover_callbacks(SparseProverRuntime& runtime) {
    SparseGkrProverCallbacks callbacks;
    callbacks.initial_claim = [&runtime](const SparseGkrRelationSpec& relation, F& initial, std::string* reason) {
        try {
            SparseRelationRuntime& state = runtime.relations[relation.relation_id];
            const F raw = runtime.provider.sum(relation);
            state.initial_claim = runtime.have_previous ? runtime.previous_terminal : raw;
            state.correction = state.initial_claim - raw;
            state.initial_opening = zkgpt::native_pedersen::commit_fresh(runtime.basis, state.initial_claim);
            state.have_initial = true;
            initial = state.initial_claim;
            return true;
        } catch (const std::exception& error) {
            if (reason != nullptr) *reason = error.what();
            return false;
        }
    };
    callbacks.terminal_spec = [&runtime](const SparseGkrRelationSpec& relation, NativeMaskedTerminalSpec& spec, std::string* reason) {
        const auto found = runtime.relations.find(relation.relation_id);
        if (found == runtime.relations.end() || !found->second.have_initial) {
            if (reason != nullptr) *reason = "sparse relation initial claim is missing";
            return false;
        }
        spec = NativeMaskedTerminalSpec();
        spec.kind = NativeMaskedTerminalKind::Scalar;
        spec.has_external_binding = true;
        spec.public_initial_claim = true;
        spec.initial_opening = found->second.initial_opening;
        spec.initial_commitment = found->second.initial_opening.commitment;
        return true;
    };
    callbacks.round_polynomial = [&runtime](const SparseGkrRelationSpec& relation, const SparseGkrChunk& chunk, std::size_t round, const std::vector<F>& prefix,
                                            std::vector<F>& polynomial, std::string* reason) {
        try {
            if (round >= 64) throw std::overflow_error("sparse relation round exceeds u64 domain");
            SparseRelationRuntime& state = runtime.relations[relation.relation_id];
            state.prefix = prefix;
            F p0 = F_ZERO;
            F p1 = F_ZERO;
            for (std::uint64_t local = 0; local < chunk.length; ++local) {
                const std::uint64_t index = chunk.offset + local;
                F weight = F_ONE;
                for (std::size_t bit = 0; bit < prefix.size(); ++bit) {
                    const bool set = ((index >> bit) & 1U) != 0;
                    weight *= set ? prefix[bit] : (F_ONE - prefix[bit]);
                }
                F sample = runtime.provider.value(relation, index);
                if (index == 0) sample += state.correction;
                if (((index >> round) & 1U) == 0)
                    p0 += sample * weight;
                else
                    p1 += sample * weight;
            }
            polynomial = {p0, p1, (F(2) * p1) - p0};
            return true;
        } catch (const std::exception& error) {
            if (reason != nullptr) *reason = error.what();
            return false;
        }
    };
    callbacks.terminal_values = [&runtime](const SparseGkrRelationSpec& relation, std::vector<F>& terminal, std::string* reason) {
        try {
            const auto found = runtime.relations.find(relation.relation_id);
            if (found == runtime.relations.end() || !found->second.have_initial) {
                if (reason != nullptr) *reason = "sparse relation runtime is missing";
                return false;
            }
            terminal = {runtime.provider.folded(relation, found->second.prefix, found->second.correction)};
            return true;
        } catch (const std::exception& error) {
            if (reason != nullptr) *reason = error.what();
            return false;
        }
    };
    callbacks.relation_finished = [&runtime](const SparseGkrRelationResult& result) {
        if (!result.sumcheck.terminal_values.empty()) {
            runtime.previous_terminal = result.sumcheck.terminal_values.front();
            runtime.have_previous = true;
        }
    };
    return callbacks;
}

CommitmentWire sparse_wire_from_opening(const zkgpt::native_pedersen::ScalarOpening& opening) {
    CommitmentWire wire;
    wire.scheme = "hyrax-pedersen-bn254";
    wire.basis_id = fixed_digest(hash_hex(opening.commitment.domain));
    wire.logical_size = 1;
    wire.padded_size = 1;
    wire.point_hex = point_hex(opening.commitment.point);
    return wire;
}

bool sparse_wire_field(const std::string& encoded, F& value, std::string* reason) {
    if (!core_decode_field(encoded, value)) {
        if (reason != nullptr) { *reason = "sparse GKR artifact field is not canonical"; }
        return false;
    }
    return true;
}

SparseGkrProofWire make_sparse_proof_wire(const GatCircuitPlan& plan, const SparseGkrPlan& sparse_plan, const std::vector<SparseGkrRelationResult>& results,
                                          const SparseProverRuntime& runtime, const NativeProofSession& session) {
    if (results.size() != sparse_plan.relations.size()) throw std::invalid_argument("sparse GKR result count does not match relation plan");
    SparseGkrProofWire wire;
    wire.plan_digest = plan.sparse_plan_digest;
    wire.schedule_digest = canonical_digest64(sparse_plan.encoding());
    for (std::size_t index = 0; index < sparse_plan.relations.size(); ++index) {
        const SparseGkrRelationSpec& relation = sparse_plan.relations[index];
        const SparseGkrRelationResult& result = results[index];
        const auto runtime_it = runtime.relations.find(relation.relation_id);
        if (runtime_it == runtime.relations.end() || !runtime_it->second.have_initial)
            throw std::invalid_argument("sparse GKR relation runtime endpoint is missing");
        SparseGkrRelationWire encoded;
        encoded.relation_id = relation.relation_id;
        encoded.initial_claim = field_hex(runtime_it->second.initial_claim);
        encoded.final_claim = field_hex(result.sumcheck.final_claim);
        for (const F& value : result.sumcheck.terminal_values) encoded.terminal_values.push_back(field_hex(value));
        encoded.initial_commitment = point_hex(result.sumcheck.initial_commitment.point);
        for (const auto& commitment : result.sumcheck.terminal_commitments) encoded.terminal_commitments.push_back(point_hex(commitment.point));
        const std::string prefix = "sparse/gkr/" + relation.relation_id + "/";
        const std::string challenge_prefix = "challenge/" + prefix;
        bool found = false;
        for (std::size_t cursor = 0; cursor < session.messages().size(); ++cursor) {
            const std::string& label = session.messages()[cursor].label;
            if (label.compare(0, prefix.size(), prefix) == 0 || label.compare(0, challenge_prefix.size(), challenge_prefix) == 0) {
                if (!found) {
                    encoded.transcript_begin = cursor;
                    found = true;
                }
                encoded.transcript_end = cursor + 1;
            }
        }
        if (!found) throw std::invalid_argument("sparse GKR relation transcript range is missing");
        wire.relations.push_back(std::move(encoded));
    }
    const auto relation_wire = [&wire](const std::string& id) -> const SparseGkrRelationWire& {
        for (const SparseGkrRelationWire& relation : wire.relations)
            if (relation.relation_id == id) return relation;
        throw std::invalid_argument("sparse GKR relation wire lookup failed");
    };
    for (const SparseGkrClaimLink& link : sparse_plan.claim_links) {
        SparseGkrLinkWire encoded;
        encoded.from_relation = link.from_relation;
        encoded.to_relation = link.to_relation;
        encoded.claim_name = link.claim_name;
        encoded.from_terminal = link.from_terminal;
        encoded.to_initial = link.to_initial;
        encoded.from_claim = relation_wire(link.from_relation).final_claim;
        encoded.to_claim = relation_wire(link.to_relation).initial_claim;
        wire.terminal_links.push_back(std::move(encoded));
    }
    for (const SparseGkrClaimLink& link : sparse_plan.binding_links) {
        SparseGkrLinkWire encoded;
        encoded.from_relation = link.from_relation;
        encoded.to_relation = link.to_relation;
        encoded.claim_name = link.claim_name;
        encoded.from_terminal = link.from_terminal;
        encoded.to_initial = link.to_initial;
        encoded.from_claim = relation_wire(link.from_relation).final_claim;
        encoded.to_claim = relation_wire(link.to_relation).initial_claim;
        wire.binding_links.push_back(std::move(encoded));
    }
    wire.input_binding = canonical_digest64("sparse/input-binding-v1/" + wire.relations.front().initial_claim);
    wire.output_binding = canonical_digest64("sparse/output-binding-v1/" + wire.relations.back().final_claim);

    wire.hyrax_lasso_complete = true;
    return wire;
}

GatCircuitPlan build_plan_impl(const Graph& graph, const Model& model, const QuantConfig& quant, const ForwardResult* forward,
                               bool shape_only = false, const GatCapacityReport* capacity_override = nullptr,
                               bool forward_prevalidated = false, bool symbolic_shape_only = false) {
    graph.validate();
    model.validate_schema();
    quant.validate();
    if (forward != nullptr && !forward_prevalidated &&
        !validate_relation(graph, forward->trace.input, model, quant, *forward))
        throw std::invalid_argument("cannot build proof plan from an invalid GAT trace");

    const GatCapacityReport computed_capacity = capacity_override == nullptr ? inspect_gat_capacity(graph, model, quant) : GatCapacityReport{};
    const GatCapacityReport& capacity = capacity_override == nullptr ? computed_capacity : *capacity_override;
    const bool explicit_input_ok = capacity.flattened_input_domain_entries <= capacity.flattened_input_domain_limit_u64;
    const bool explicit_coordinate_ok = capacity.input_layout.n_in <= capacity.coordinate_domain_limit_u64;
    const bool explicit_storage_ok =
        capacity.estimated_native_scratch_bytes <= std::numeric_limits<std::size_t>::max() &&
        capacity.estimated_native_scratch_bytes <= static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
    const bool explicit_path_ready = explicit_input_ok && explicit_coordinate_ok && explicit_storage_ok;
    if (!explicit_path_ready) throw std::length_error("native layered GKR plan exceeds checked input-coordinate or addressable storage bounds");
    if (!capacity.ppi_node_ptr_boundary || !capacity.ppi_edge_ptr_boundary) throw std::length_error("canonical GAT graph batch boundaries are invalid");

    const InputLayout canonical_layout = canonical_input_layout_impl(graph, model, quant);
    const std::size_t aux_size = canonical_layout.segments.at(2).logical_length;
    const std::size_t query_size = canonical_layout.segments.at(3).logical_length;
    if (forward != nullptr && trace_aux_size(&forward->trace) != aux_size) {
        throw std::invalid_argument("forward trace Aux shape does not match canonical plan coordinates");
    }

    GatCircuitPlan plan;
    plan.recipe_schema = "canonical_plan_recipe_v3";
    plan.relation_schema = "zkgpt-gat-relation-v1";
    plan.model_arch_id = model.manifest.model_arch_id;
    plan.model_shape = canonical_model_shape(model);
    plan.graph_num_nodes = graph.num_nodes;
    plan.graph_num_features = graph.num_features;
    plan.graph_count = graph.graph_count;
    plan.graph_num_classes = graph.num_classes;
    plan.graph_edge_count = graph.edges();
    plan.graph_topology_digest = canonical_graph_topology_digest(graph);
    plan.quant_cfg_id = quant.quant_cfg_id;
    plan.quant_digest = canonical_quant_digest(quant);
    plan.static_tables_id = StaticTables::build(quant).table_id;
    plan.registry = RelationRegistry::canonical();
    plan.opening_policy = "same_r_three_terminal_openings_v1";
    plan.terminal_openings = {"W@r", "Aux@r", "L_qry@r"};
    plan.input_layout = canonical_layout;
    plan.division_range_bits = canonical_division_range_bits(graph, model, quant);
    const std::size_t circuit_input_size = checked_mul(plan.input_layout.n_in, 4, "four-segment input layer");

    plan.circuit.init(0, 1);
    plan.circuit.circuit[0].ty = layerType::INPUT;
    plan.circuit.circuit[0].size = as_u64(circuit_input_size, "input layer size");
    plan.circuit.circuit[0].zero_start_id = plan.circuit.circuit[0].size;
    plan.circuit.circuit[0].bit_length = bit_length(circuit_input_size);
    std::vector<std::pair<u64, u64>> source_support_ranges;
    source_support_ranges.reserve(plan.input_layout.segments.size());
    for (const InputSegment& segment : plan.input_layout.segments) {
        if (segment.logical_length == 0) continue;
        source_support_ranges.emplace_back(as_u64(segment.offset, "source support offset"),
                                           as_u64(segment.logical_length, "source support length"));
    }
    plan.circuit.set_source_support_ranges(std::move(source_support_ranges));
    std::unique_ptr<GkrPointShapeCollector> shape_collector;
    std::unique_ptr<GkrPointShapeCollectorScope> shape_collector_scope;
    if (shape_only) {
        shape_collector = std::make_unique<GkrPointShapeCollector>(plan.circuit,
                                                                  plan.input_layout, symbolic_shape_only);
        shape_collector_scope = std::make_unique<GkrPointShapeCollectorScope>(
            shape_collector.get());
    }

    const std::size_t hidden0_dim = model.tensor(hidden_tensor_name(0, 0, "seq/kernel")).shape.dims.at(2);
    const std::size_t hidden1_dim = model.tensor(hidden_tensor_name(1, 0, "seq/kernel")).shape.dims.at(2);
    const std::size_t classes = model.classes();
    const std::size_t first_hidden_projection = checked_mul(graph.num_nodes, hidden0_dim, "hidden projection coordinates");
    const std::size_t second_hidden_projection = checked_mul(graph.num_nodes, hidden1_dim, "hidden projection coordinates");

    const TraceCoordinates coordinates = make_trace_coordinates(graph, model, plan.input_layout);
    const QueryCoordinates query_coordinates = make_query_coordinates(graph, model, quant, plan.input_layout);
    std::size_t w_cursor = plan.input_layout.segments.at(1).offset;
    std::size_t lrelu_query_cursor = query_coordinates.lrelu;
    std::size_t exp_query_cursor = query_coordinates.exp;
    std::size_t elu_query_cursor = query_coordinates.elu;
    std::size_t range_query_cursor = query_coordinates.range;
    const auto add_attention_descriptors = [&plan, &graph](const std::string& family, const AttentionCoordinates& a, std::size_t output_dim, std::size_t seq_w,
                                                           std::size_t src_w, std::size_t dst_w, std::size_t input_dim, std::size_t lhs_offset,
                                                           const std::string& projection_scale, const std::string& attention_scale) {
        const std::size_t projection_coordinates = checked_mul(graph.num_nodes, output_dim, "projection coordinates");
        const auto add_matmul = [&plan, &graph](const std::string& relation, const std::string& lhs, const std::string& rhs, const std::string& output,
                                                std::size_t rows, std::size_t inner, std::size_t cols, std::size_t lhs_offset, std::size_t rhs_offset,
                                                std::size_t output_offset, const std::string& scale_rule) {
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
            bind_matmul_segments(descriptor, plan.input_layout);
            descriptor.coordinate_rule = "row_major_exact[row,term,col]";
            descriptor.scale_rule = scale_rule;
            plan.matmuls.push_back(descriptor);
            append_fconn_claim_link(plan, plan.matmuls.back());
            append_projection_layer(plan, descriptor);
        };
        add_matmul(family + "/projection", family + ".input", family + ".seq/kernel", family + ".H_prime", graph.num_nodes, input_dim, output_dim, lhs_offset,
                   seq_w, a.projection, projection_scale);
        add_matmul(family + "/attention/source_dot", family + ".H_prime", family + ".attn_src/kernel", family + ".E_src", graph.num_nodes, output_dim, 1,
                   a.projection, src_w, a.source_score, attention_scale);
        add_matmul(family + "/attention/destination_dot", family + ".H_prime", family + ".attn_dst/kernel", family + ".E_dst", graph.num_nodes, output_dim, 1,
                   a.projection, dst_w, a.destination_score, attention_scale);
    };
    for (std::size_t layer_id = 0; layer_id < 2; ++layer_id) {
        const std::size_t output_dim = layer_id == 0 ? hidden0_dim : hidden1_dim;
        const std::size_t input_dim = model.tensor(hidden_tensor_name(layer_id, 0, "seq/kernel")).shape.dims.at(1);
        const std::size_t projection_coordinates = checked_mul(graph.num_nodes, output_dim, "hidden projection coordinates");
        for (std::size_t head = 0; head < model.hidden_heads(); ++head) {
            const AttentionCoordinates& a = coordinates.hidden[layer_id][head];
            const std::string family = hidden_family(layer_id, head);
            const std::size_t seq_w = w_cursor;
            w_cursor = checked_add(w_cursor, checked_mul(input_dim, output_dim, "hidden sequence weights"), "hidden weight cursor");
            const std::size_t src_w = w_cursor;
            w_cursor = checked_add(w_cursor, output_dim, "hidden source weights");
            const std::size_t dst_w = w_cursor;
            w_cursor = checked_add(w_cursor, output_dim, "hidden destination weights");
            add_attention_descriptors(family, a, output_dim, seq_w, src_w, dst_w, input_dim,
                                      layer_id == 0 ? plan.input_layout.segments.at(0).offset : coordinates.hidden_concat[layer_id - 1],
                                      "s_input*s_w_hidden[layer][head]->s_hprime_hidden[layer][head]", "s_hprime_hidden[layer][head]*s_a_{src,dst}->s_h");
            append_checkpoint(plan, family + "/projection", family + "/projection", "Aux", projection_coordinates, a.projection,
                              "Aux[row_major_exact[node,col]]");
            append_checkpoint(plan, family + "/source_score", family + "/attention/source_dot", "Aux", graph.num_nodes, a.source_score,
                              "Aux[row_major_exact[node]]");
            append_checkpoint(plan, family + "/destination_score", family + "/attention/destination_dot", "Aux", graph.num_nodes, a.destination_score,
                              "Aux[row_major_exact[node]]");

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
            score.constraints = {"E_src[node]=dot(H_prime[node,row],a_src[row])",
                                 "E_dst[node]=dot(H_prime[node,row],a_dst[row])",
                                 "raw_score[e]=E_src[src[e]]+E_dst[dst[e]]",
                                 "Q_grp_new[e]=(e==edge_ptr[gid] or dst[e]!=dst[e-1])",
                                 "Q_grp_end[e]=(e+1==edge_ptr[gid+1] or dst[e+1]!=dst[e])",
                                 "leaky_score[e]=T_LReLU(raw_score[e])"};
            plan.sparse_gkr.push_back(score);
            lrelu_query_cursor = checked_add(lrelu_query_cursor, graph.edges(), "LeakyReLU query cursor");
            append_checkpoint(plan, score.relation_id + ".raw_score", score.relation_id, "Aux", graph.edges(), a.raw_score, "Aux[row_major_exact[edge]]");

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
            softmax.constraints = {"group_max[dst[e]]=max(leaky_score[group])",
                                   "delta[e]=group_max[dst[e]]-leaky_score[e]",
                                   "exp[e]=T_exp(delta[e])",
                                   "group_sum[node]=sum(exp[e]) over incoming canonical edges",
                                   "group_inverse[node]*group_sum[node]=1 in BN254",
                                   "max_selector[e] in {0,1} and sum_group(max_selector)=1",
                                   "alpha[e]=exp[e]*group_inverse[dst[e]]",
                                   "Q_grp_new and Q_grp_end recurrence is explicit"};
            plan.sparse_gkr.push_back(softmax);
            const std::size_t exp_offset = exp_query_cursor;
            exp_query_cursor = checked_add(exp_query_cursor, graph.edges(), "ExpMap query cursor");
            append_checkpoint(plan, softmax.relation_id + ".delta", softmax.relation_id, "Aux", graph.edges(), a.delta, "Aux[row_major_exact[edge]]");
            append_checkpoint(plan, softmax.relation_id + ".group_inverse", softmax.relation_id, "Aux", graph.num_nodes, a.group_inverse,
                              "Aux[row_major_exact[node]]");
            append_checkpoint(plan, softmax.relation_id + ".alpha", softmax.relation_id, "Aux", graph.edges(), a.alpha, "Aux[row_major_exact[edge]]");

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
            aggregate.constraints = {"Q_grp_new and Q_grp_end drive destination group recurrence",
                                     "integer_numerator[node,col]=sum(exp[e]*H_prime[src[e],col])",
                                     "field_numerator=integer_numerator embedded in BN254",
                                     "field_aggregate[node,col]*group_sum[node]=field_numerator[node,col]",
                                     "field_aggregate and integer_numerator have dual aggregate consistency",
                                     "recovered=SignedRoundDiv(s_agg_pre*integer_numerator,s_hprime*group_sum)",
                                     "destination gather uses canonical src[e],dst[e] coordinates"};
            plan.sparse_gkr.push_back(aggregate);
            append_checkpoint(plan, aggregate.relation_id + ".integer_numerator", aggregate.relation_id, "Aux", projection_coordinates, a.integer_numerator,
                              "Aux[row_major_exact[node,col]]");
            append_checkpoint(plan, aggregate.relation_id + ".field_aggregate", aggregate.relation_id, "Aux", projection_coordinates, a.field_aggregate,
                              "Aux[row_major_exact[node,col]]");
            append_checkpoint(plan, aggregate.relation_id + ".recovered", aggregate.relation_id, "Aux", projection_coordinates, a.recovered,
                              "Aux[row_major_exact[node,col]]");

            append_checkpoint(plan, "rounding/hidden/" + family, "rounding", "Aux", checked_mul(projection_coordinates, 9, "hidden division coordinates"),
                              a.divisions, "Aux[row_major_exact[node,col,witness_component]]");

            elu_query_cursor = checked_add(elu_query_cursor, projection_coordinates, "ELU query cursor");
            append_checkpoint(plan, "activation.hidden/" + family, "activation.hidden", "Aux", projection_coordinates, a.activated,
                              "Aux[row_major_exact[node,col]]");
        }
        const std::size_t concat_coordinates =
            checked_mul(graph.num_nodes, checked_mul(output_dim, model.hidden_heads(), "concat width"), "concat coordinates");
        if (active_gkr_point_shape_collector != nullptr && active_gkr_point_shape_collector->symbolic()) {
            active_gkr_point_shape_collector->append_symbolic_explicit_layer(
                layerType::ADD_BIAS, std::max<std::size_t>(1, concat_coordinates),
                concat_coordinates, 0, concat_coordinates, 0, false, false, false);
        } else {
            ExplicitUnaryGateBuffer concat_unary;
            concat_unary.reserve(concat_coordinates);
            for (std::size_t node = 0; node < graph.num_nodes; ++node) {
                for (std::size_t head = 0; head < model.hidden_heads(); ++head) {
                    const AttentionCoordinates& a = coordinates.hidden[layer_id][head];
                    for (std::size_t col = 0; col < output_dim; ++col) {
                        const std::size_t output_index = node * output_dim * model.hidden_heads() + head * output_dim + col;
                        concat_unary.push_back({output_index, a.activated + node * output_dim + col, 0, 1});
                    }
                }
            }
            append_explicit_layer(plan, layerType::ADD_BIAS, std::max<std::size_t>(1, concat_coordinates), concat_unary, {}, false);
        }
        append_checkpoint(plan, "concat/hidden/layer" + std::to_string(layer_id), "concat", "Aux", concat_coordinates, coordinates.hidden_concat[layer_id],
                          "Aux[row_major_exact[node,head,col]]");
    }

    for (std::size_t head = 0; head < model.manifest.output_heads; ++head) {
        const std::string family = output_family(head);
        const AttentionCoordinates& a = coordinates.output[head];
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
        bind_matmul_segments(projection, plan.input_layout);
        projection.coordinate_rule = "row_major_exact[row,term,col]";
        projection.scale_rule = "s_cat*s_w_output[head]->s_yprime_output[head]";
        plan.matmuls.push_back(projection);
        append_fconn_claim_link(plan, plan.matmuls.back());
        append_projection_layer(plan, projection);
        append_checkpoint(plan, projection.relation_id + ".Y_prime", projection.relation_id, "Aux", projection_coordinates, a.projection,
                          "Aux[row_major_exact[node,col]]");
        const std::string attention_scale = "s_yprime_output[head]*s_a_{src,dst}->s_out";
        const auto add_dot = [&plan, &graph, &family](const std::string& suffix, std::size_t lhs_offset, const std::string& rhs, const std::string& output,
                                                      std::size_t rhs_offset, std::size_t output_offset, std::size_t input_dim, const std::string& scale) {
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
            bind_matmul_segments(descriptor, plan.input_layout);
            descriptor.coordinate_rule = "row_major_exact[row,term,col]";
            descriptor.scale_rule = scale;
            plan.matmuls.push_back(descriptor);
            append_fconn_claim_link(plan, plan.matmuls.back());
            append_projection_layer(plan, descriptor);
            append_checkpoint(plan, descriptor.relation_id, descriptor.relation_id, "Aux", graph.num_nodes, output_offset, "Aux[row_major_exact[node]]");
        };
        add_dot("source_dot", a.projection, family + ".attn_src/kernel", family + ".E_src", src_w, a.source_score, classes, attention_scale);
        add_dot("destination_dot", a.projection, family + ".attn_dst/kernel", family + ".E_dst", dst_w, a.destination_score, classes, attention_scale);
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
        lrelu_query_cursor = checked_add(lrelu_query_cursor, graph.edges(), "output LeakyReLU query cursor");
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
        softmax.constraints = {"group_max=max(leaky_score) by dst",
                               "delta=group_max-leaky_score",
                               "exp=T_exp(delta)",
                               "group_sum=sum(exp)",
                               "group_inverse*group_sum=1",
                               "max_selector one-hot by group",
                               "alpha=exp*group_inverse[dst]",
                               "Q_grp_new/Q_grp_end destination-group recurrence"};
        plan.sparse_gkr.push_back(softmax);
        append_checkpoint(plan, softmax.relation_id + ".delta", softmax.relation_id, "Aux", graph.edges(), a.delta, "Aux[row_major_exact[edge]]");
        append_checkpoint(plan, softmax.relation_id + ".alpha", softmax.relation_id, "Aux", graph.edges(), a.alpha, "Aux[row_major_exact[edge]]");
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
        aggregate.constraints = {"integer numerator=sum(exp*Y_prime[src])", "field_numerator=integer_numerator", "field_aggregate*group_sum=field_numerator",
                                 "recovered=SignedRoundDiv(s_out_pre*integer_numerator,s_yprime*group_sum)", "dual integer/field aggregate consistency"};
        plan.sparse_gkr.push_back(aggregate);
        append_checkpoint(plan, aggregate.relation_id + ".recovered", aggregate.relation_id, "Aux", projection_coordinates, a.recovered,
                          "Aux[row_major_exact[node,col]]");
        append_checkpoint(plan, "rounding/output/" + family, "rounding", "Aux", checked_mul(projection_coordinates, 9, "output division coordinates"),
                          a.divisions, "Aux[row_major_exact[node,col,witness_component]]");
        append_checkpoint(plan, "output/aggregate/" + family, "aggregation.output", "Aux", projection_coordinates, coordinates.output_aggregate[head],
                          "Aux[row_major_exact[node,col]]");
        append_checkpoint(plan, "rounding/output_alignment/" + family, "rounding", "Aux", checked_mul(projection_coordinates, 9, "output alignment division coordinates"), coordinates.output_alignment_divisions[head], "Aux[row_major_exact[node,col,witness_component]]");
        append_checkpoint(plan, "output/aligned/" + family, "rounding", "Aux", projection_coordinates, coordinates.output_aligned[head],
                          "Aux[row_major_exact[node,col]]");
        append_checkpoint(plan, "output/bias/" + family, "output.public_logits", "W", classes, bias_w, "W[row_major_exact[class]]");
        append_checkpoint(plan, "output/logits/" + family, "output.public_logits", "Aux", projection_coordinates, coordinates.output_logits[head],
                          "Aux[row_major_exact[node,col]]");
    }
    const std::size_t output_coordinates = checked_mul(graph.num_nodes, classes, "final output coordinates");
    append_checkpoint(plan, "output/head_sum", "output.public_logits", "Aux", output_coordinates, coordinates.head_sum, "Aux[row_major_exact[node,class]]");
    append_checkpoint(plan, "output/final_average", "rounding", "Aux", checked_mul(output_coordinates, 9, "average division coordinates"),
                      coordinates.final_average_divisions, "Aux[row_major_exact[node,class,witness_component]]");
    if (active_gkr_point_shape_collector != nullptr && active_gkr_point_shape_collector->symbolic()) {
        active_gkr_point_shape_collector->append_symbolic_explicit_layer(
            layerType::ADD_BIAS, std::max<std::size_t>(1, output_coordinates),
            output_coordinates, 0, output_coordinates, 0, false, false, false);
    } else {
        ExplicitUnaryGateBuffer final_unary;
        final_unary.reserve(output_coordinates);
        for (std::size_t index = 0; index < output_coordinates; ++index) final_unary.push_back({index, coordinates.final_logits + index, 0, 1});
        append_explicit_layer(plan, layerType::ADD_BIAS, std::max<std::size_t>(1, output_coordinates), final_unary, {}, false);
    }
    append_checkpoint(plan, "output/public_logits", "output.public_logits", "Aux", output_coordinates, coordinates.final_logits,
                      "Aux[row_major_exact[node,class]]");

    const GatTrace* trace = forward == nullptr ? nullptr : &forward->trace;
    const StaticTables canonical_tables = StaticTables::build(quant);
    const StaticLookupBounds canonical_lookup_bounds = static_lookup_bounds(canonical_tables);
    const QueryCounts query_counts = query_coordinates.counts;
    const auto add_lasso = [&plan](const std::string& family, const std::string& table, std::size_t real, std::size_t offset, std::size_t domain, Int padding) {
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
        append_checkpoint(plan, descriptor.relation_id, descriptor.relation_id, "L_qry", descriptor.padded_queries, offset, "L_qry[row_major_exact[query]]");
    };
    add_lasso("function/lrelu", "T_LReLU", query_counts.lrelu, query_coordinates.lrelu, canonical_tables.lrelu.size(), canonical_tables.lrelu.front().x);
    add_lasso("function/exp", "T_exp", query_counts.exp, query_coordinates.exp, canonical_tables.exp.size(), canonical_tables.exp.front().x);
    add_lasso("function/elu", "T_ELU", query_counts.elu, query_coordinates.elu, canonical_tables.elu.size(), canonical_tables.elu.front().x);
    add_lasso("range/all", "T_range", query_counts.range, query_coordinates.range, 2, 0);
    if (symbolic_shape_only) {
        std::uint64_t source_entries = 0;
        source_entries = checked_add(source_entries,
            plan.input_layout.segments.at(0).logical_length,
            "symbolic active X_pub source span");
        source_entries = checked_add(source_entries,
            plan.input_layout.segments.at(1).logical_length,
            "symbolic active W source span");
        source_entries = checked_add(source_entries,
            plan.input_layout.segments.at(2).logical_length,
            "symbolic active Aux source span");
        for (const LassoFamilyDescriptor& family : plan.lasso)
            source_entries = checked_add(source_entries, family.padded_queries,
                "symbolic active padded L_qry source blocks");
        std::uint64_t all_logical_source_entries = 0;
        for (const InputSegment& segment : plan.input_layout.segments)
            all_logical_source_entries = checked_add(all_logical_source_entries,
                segment.logical_length, "symbolic logical source support");
        if (source_entries == 0 || source_entries > all_logical_source_entries ||
            shape_collector == nullptr)
            throw std::logic_error("symbolic active layer-0 source bound exceeds canonical logical support");
        shape_collector->set_symbolic_input_source_entries_upper_bound(source_entries);
        plan.gkr_layer0_source_domain_upper_bound = source_entries;
    }
    if (trace != nullptr && trace->tables.digest != canonical_tables.digest)
        throw std::invalid_argument("forward trace static table digest does not match canonical plan tables");
    if (w_cursor != checked_add(plan.input_layout.segments.at(1).offset, plan.input_layout.segments.at(1).logical_length, "model weight end")) {
        throw std::invalid_argument("canonical model weight coordinates do not cover W segment");
    }
    if (lrelu_query_cursor != checked_add(query_coordinates.lrelu, query_counts.lrelu, "LeakyReLU query end") ||
        exp_query_cursor != checked_add(query_coordinates.exp, query_counts.exp, "ExpMap query end") ||
        elu_query_cursor != checked_add(query_coordinates.elu, query_counts.elu, "ELU query end") ||
        range_query_cursor != query_coordinates.range) {
        throw std::invalid_argument("canonical L_qry wiring does not cover all query coordinates");
    }

    RelationLayerBuilder relation_builder;
    if (capacity.relation_residual_entries > std::numeric_limits<std::size_t>::max() ||
        capacity.relation_gate_output_entries > std::numeric_limits<std::size_t>::max() ||
        capacity.relation_gate_unary_entries > std::numeric_limits<std::size_t>::max() ||
        capacity.relation_gate_binary_entries > std::numeric_limits<std::size_t>::max())
        throw std::length_error("canonical GAT relation buffers exceed native size_t");
    relation_builder.reserve(static_cast<std::size_t>(capacity.relation_gate_output_entries),
                             static_cast<std::size_t>(capacity.relation_residual_entries),
                             static_cast<std::size_t>(capacity.relation_gate_unary_entries),
                             static_cast<std::size_t>(capacity.relation_gate_binary_entries),
                             relation_coefficient_candidate_count(quant, canonical_lookup_bounds));
    const std::size_t public_output_coordinates = checked_mul(graph.num_nodes, classes, "public output coordinates");
    if (symbolic_shape_only) relation_builder.set_symbolic_counts();
    else {
    std::size_t bound_profile_index = 0;
    std::size_t bound_lrelu_cursor = query_coordinates.lrelu, bound_exp_cursor = query_coordinates.exp, bound_elu_cursor = query_coordinates.elu, bound_range_cursor = query_coordinates.range;
    for (std::size_t index = 0; index < public_output_coordinates; ++index) relation_builder.add_public_output(coordinates.final_logits + index);

    for (std::size_t index = 0; index < plan.input_layout.segments.front().logical_length; ++index) relation_builder.add_residual({{plan.input_layout.segments.at(2).offset + index, 1}, {index, -1}}, {});
    const auto incoming = incoming_ranges_for_plan(graph);
    for (std::size_t layer_id = 0; layer_id < 2; ++layer_id) {
        const std::size_t output_dim = model.tensor(hidden_tensor_name(layer_id, 0, "seq/kernel")).shape.dims.at(2);
        for (std::size_t head = 0; head < model.hidden_heads(); ++head) {
            const AttentionCoordinates& a = coordinates.hidden[layer_id][head];
            add_attention_residuals(relation_builder, graph, incoming, a, true, quant);
            add_attention_lookup_residuals(relation_builder, graph, a, true, canonical_lookup_bounds, bound_lrelu_cursor, bound_exp_cursor, bound_elu_cursor, bound_range_cursor, plan.division_range_bits.at(bound_profile_index++));
            add_attention_recovery_residuals(relation_builder, graph, a, true, layer_id, head, quant);
        }
        for (std::size_t node = 0; node < graph.num_nodes; ++node) {
            for (std::size_t head = 0; head < model.hidden_heads(); ++head) {
                const AttentionCoordinates& a = coordinates.hidden[layer_id][head];
                for (std::size_t col = 0; col < output_dim; ++col) {
                    const std::size_t output_index = coordinates.hidden_concat[layer_id] + node * output_dim * model.hidden_heads() + head * output_dim + col;
                    const std::size_t source = a.activated + node * output_dim + col;
                    relation_builder.add_residual({{output_index, 1}, {source, -1}}, {});
                }
            }
        }
    }

    std::vector<std::size_t> output_bias_offsets(model.manifest.output_heads, 0);
    for (const CheckpointDescriptor& checkpoint : plan.checkpoints) {
        for (std::size_t head = 0; head < model.manifest.output_heads; ++head) {
            const std::string expected = "output/bias/output/head" + std::to_string(head);
            if (checkpoint.id == expected) output_bias_offsets[head] = checkpoint.coordinate_offset;
        }
    }
    for (std::size_t head = 0; head < model.manifest.output_heads; ++head) {
        const AttentionCoordinates& a = coordinates.output[head];
        const std::size_t output_coordinates = checked_mul(graph.num_nodes, classes, "output relation coordinates");
        add_attention_residuals(relation_builder, graph, incoming, a, false, quant);
        add_attention_lookup_residuals(relation_builder, graph, a, false, canonical_lookup_bounds, bound_lrelu_cursor, bound_exp_cursor, bound_elu_cursor, bound_range_cursor, plan.division_range_bits.at(bound_profile_index++));
        add_division_range_residuals(relation_builder, coordinates.output_alignment_divisions[head], output_coordinates, bound_range_cursor, plan.division_range_bits.at(bound_profile_index++));
        add_attention_recovery_residuals(relation_builder, graph, a, false, 0, head, quant);
        for (std::size_t index = 0; index < output_coordinates; ++index) {
            relation_builder.add_residual({{coordinates.output_aggregate[head] + index, 1}, {a.recovered + index, -1}}, {});
            add_division_residuals(relation_builder, coordinates.output_alignment_divisions[head] + index * 9, coordinates.output_aligned[head] + index,
                                   {{coordinates.output_aggregate[head] + index, quant.s_out_head}}, {}, quant.rounding);
            relation_builder.add_residual({{coordinates.output_alignment_divisions[head] + index * 9 + 8, 1}}, {}, quant.s_out_pre);
            relation_builder.add_residual({{coordinates.output_logits[head] + index, -1},
                                           {coordinates.output_aligned[head] + index, 1},
                                           {output_bias_offsets[head] + index % classes, 1}},
                                          {});
        }
    }
    for (std::size_t index = 0; index < public_output_coordinates; ++index) {
        std::vector<LinearTerm> average_terms;
        average_terms.reserve(model.manifest.output_heads + 1);
        average_terms.emplace_back(coordinates.head_sum + index, -1);
        for (std::size_t head = 0; head < model.manifest.output_heads; ++head) average_terms.emplace_back(coordinates.output_logits[head] + index, 1);
        relation_builder.add_residual(std::move(average_terms), {});
        add_division_residuals(relation_builder, coordinates.final_average_divisions + index * 9, coordinates.final_logits + index,
                               {{coordinates.head_sum + index, 1}}, {}, quant.rounding);
        relation_builder.add_residual({{coordinates.final_average_divisions + index * 9 + 8, 1}}, {}, static_cast<Int>(model.manifest.output_heads));
    }
    add_division_range_residuals(relation_builder, coordinates.final_average_divisions, public_output_coordinates, bound_range_cursor, plan.division_range_bits.at(bound_profile_index++));
    if (bound_lrelu_cursor != query_coordinates.lrelu + query_counts.lrelu || bound_exp_cursor != query_coordinates.exp + query_counts.exp || bound_elu_cursor != query_coordinates.elu + query_counts.elu || bound_range_cursor != query_coordinates.range + query_counts.range) throw std::logic_error("canonical lookup/range binding does not cover the registered query coordinates");
    for (const LassoFamilyDescriptor& family : plan.lasso) {
        Fr padding(0);
        if (family.family_id != "range/all") {
            const std::vector<TablePoint>& points = table_points_for(canonical_tables, family.family_id);
            const auto& bounds = table_bounds_for(canonical_lookup_bounds, family.family_id);
            padding = Fr((points.front().x + bounds.first) * (2 * bounds.second + 1) + points.front().y + bounds.second);
        }
        Int pad = 0; if (!padding.isZero()) pad = std::stoll(padding.getStr());
        for (std::size_t index = family.real_queries; index < family.padded_queries; ++index) relation_builder.add_residual({{family.query_offset + index, 1}}, {}, pad);
    }
    }
    if (relation_builder.next_output != capacity.relation_gate_output_entries ||
        relation_builder.emitted_targets != capacity.relation_residual_entries ||
        relation_builder.emitted_unary != capacity.relation_gate_unary_entries ||
        relation_builder.emitted_binary != capacity.relation_gate_binary_entries) {
        throw std::logic_error("canonical GAT relation emission does not match its exact capacity profile (outputs=" +
            std::to_string(relation_builder.next_output) + "/" + std::to_string(capacity.relation_gate_output_entries) +
            ", targets=" + std::to_string(relation_builder.emitted_targets) + "/" + std::to_string(capacity.relation_residual_entries) +
            ", unary=" + std::to_string(relation_builder.emitted_unary) + "/" + std::to_string(capacity.relation_gate_unary_entries) +
            ", binary=" + std::to_string(relation_builder.emitted_binary) + "/" + std::to_string(capacity.relation_gate_binary_entries) + ")");
    }
    if (!shape_only)
        finish_relation_layer(plan, relation_builder, public_output_coordinates);

    plan.native_api = {"layeredCircuit::initSubset",
                       "prover::sumcheckInitAll / sumcheckInit / sumcheckInitPhase1 / sumcheckUpdate1 / sumcheckFinalize1",
                       "prover::sumcheckInitPhase2 / sumcheckUpdate2 / sumcheckFinalize2",
                       "range_prover::sumcheck_deg1 / range_prover::sumcheck_deg3",
                       "perdersen_commit / hyrax::open",
                       "verifier::verifyGKR / verifier::verifyLasso / verifier::openCommit"};
    plan.circuit.size = static_cast<u8>(plan.circuit.circuit.size());
    if (shape_only) {
        if (!relation_builder.shape_only || relation_builder.emitted_targets != capacity.relation_residual_entries ||
            relation_builder.emitted_unary != capacity.relation_gate_unary_entries ||
            relation_builder.emitted_binary != capacity.relation_gate_binary_entries)
            throw std::logic_error("shape-only GAT relation emission disagrees with its exact capacity profile");
        if (symbolic_shape_only) {
            const std::uint64_t unary_refs = static_cast<std::uint64_t>(relation_builder.emitted_unary) + relation_builder.emitted_binary;
            active_gkr_point_shape_collector->append_symbolic_explicit_layer(
                layerType::RELU, relation_builder.next_output,
                relation_builder.emitted_unary, relation_builder.emitted_binary,
                unary_refs, relation_builder.emitted_binary, false, false,
                relation_builder.emitted_binary != 0);
        } else {
            active_gkr_point_shape_collector->append_explicit_layer(
                layerType::RELU, relation_builder.next_output,
                relation_builder.emitted_unary, relation_builder.emitted_binary);
        }
        std::vector<NativeGkrMatmulShape> shapes;
        shapes.reserve(plan.matmuls.size());
        for (const MatMulDescriptor& descriptor : plan.matmuls)
            shapes.push_back(native_gkr_shape_from_descriptor(descriptor,
                                                               plan.input_layout));
        std::string shape_reason;
        plan.gkr_point_domain_upper_bound_valid = validate_native_gkr_point_domains(
            plan.circuit, shapes, plan.circuit.circuit.front().size, &shape_reason);
        plan.gkr_point_domain_upper_bound_reason = shape_reason;
        if (!plan.gkr_point_domain_upper_bound_valid && !symbolic_shape_only)
            throw std::invalid_argument("shape-only GKR point-domain admission failed: " + shape_reason);
        if (!symbolic_shape_only)
            plan.gkr_point_shape_digest = gkr_point_shape_digest(
                plan.circuit, plan.matmuls, plan.input_layout);
        plan.gkr_layer_shape_profile = shape_collector->layer_shapes();
        plan.gkr_point_shape_bitmap_bytes = shape_collector->bitmap_bytes();
        plan.gkr_point_shape_touched_index_bytes_upper_bound =
            shape_collector->touched_index_bytes_upper_bound();
        return plan;
    }

    const auto subset_begin = std::chrono::steady_clock::now();
    std::cerr << "gat_adapter_prover_stage: native-initSubset begin" << std::endl;
    plan.circuit.initSubset();
    std::cerr << "gat_adapter_prover_stage: native-initSubset complete elapsed_ms="
              << std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - subset_begin).count()
              << std::endl;
    std::vector<NativeGkrMatmulShape> materialized_point_shapes;
    materialized_point_shapes.reserve(plan.matmuls.size());
    for (const MatMulDescriptor& descriptor : plan.matmuls)
        materialized_point_shapes.push_back(native_gkr_shape_from_descriptor(
            descriptor, plan.input_layout));
    std::string materialized_shape_reason;
    if (!validate_native_gkr_point_domains(
            plan.circuit, materialized_point_shapes,
            plan.circuit.circuit.front().size, &materialized_shape_reason))
        throw std::invalid_argument("materialized GKR point-domain admission failed: " +
                                    materialized_shape_reason);
    plan.gkr_point_shape_digest = gkr_point_shape_digest(
        plan.circuit, plan.matmuls, plan.input_layout);
    plan.sparse_relation_schema = kSparseGkrRelationSchema;
    plan.sparse_relation_version = kSparseGkrRelationVersion;
    plan.sparse_chunk_size = kSparseGkrChunkEntries;
    plan.virtual_sparse = false;
    std::vector<SparseGkrRelationSpec> sparse_specs;
    sparse_specs.reserve(plan.sparse_gkr.size() + plan.matmuls.size() + 3);
    SparseGkrRelationSpec input_binding;
    input_binding.relation_id = "binding/input_segments";
    input_binding.family_id = "input";
    input_binding.kind = SparseGkrRelationKind::InputBinding;
    input_binding.item_count = 4;
    input_binding.value_width = plan.input_layout.n_in;
    input_binding.chunk_size = kSparseGkrChunkEntries;
    input_binding.coordinate_rule = "segment_selector_low_bits_then_local_u64_coordinate";
    input_binding.group_rule = "00:X_pub,01:W,10:Aux,11:L_qry";
    input_binding.equations = {"logical segment values plus implicit zero padding", "input oracle fold uses checked-u64 chunk offsets"};
    sparse_specs.push_back(std::move(input_binding));
    for (const MatMulDescriptor& matmul : plan.matmuls) {
        SparseGkrRelationSpec fconn;
        fconn.relation_id = "fconn/" + matmul.relation_id;
        fconn.family_id = matmul.family_id;
        fconn.kind = SparseGkrRelationKind::Fconn;
        fconn.item_count = matmul.rows;
        fconn.value_width = checked_mul(matmul.inner, matmul.cols, "sparse FCONN relation width");
        fconn.input_offset = matmul.lhs_offset;
        fconn.source_offset = matmul.rhs_offset;
        fconn.output_offset = matmul.output_offset;
        fconn.chunk_size = kSparseGkrChunkEntries;
        fconn.coordinate_rule = matmul.coordinate_rule;
        fconn.group_rule = "descriptor_index_ordered_fconn_claims";
        fconn.equations = {"FCONN lhs/rhs/output MLE claims use descriptor offsets", "FCONN terminal claim links to output binding"};
        sparse_specs.push_back(std::move(fconn));
    }
    for (SparseGkrDescriptor& descriptor : plan.sparse_gkr) {
        descriptor.relation_kind = sparse_relation_kind_for_id(descriptor.relation_id);
        descriptor.relation_schema = kSparseGkrRelationSchema;
        descriptor.chunk_size = kSparseGkrChunkEntries;
        descriptor.arithmetic_rule = kSparseGkrArithmeticRule;
        descriptor.overflow_rule = kSparseGkrOverflowRule;
        descriptor.terminal_rule = "terminal-claim-chain/input-output-binding-v1";
        SparseGkrRelationSpec spec;
        spec.relation_id = descriptor.relation_id;
        spec.family_id = descriptor.family_id;
        spec.kind = descriptor.relation_kind;
        spec.item_count = descriptor.edge_count;
        spec.value_width = descriptor.value_width;
        spec.input_offset = descriptor.input_offset;
        spec.source_offset = descriptor.source_offset;
        spec.destination_offset = descriptor.destination_offset;
        spec.output_offset = descriptor.output_offset;
        spec.chunk_size = descriptor.chunk_size;
        spec.coordinate_rule = descriptor.coordinate_rule;
        spec.group_rule = descriptor.group_rule;
        spec.arithmetic_rule = descriptor.arithmetic_rule;
        spec.overflow_rule = descriptor.overflow_rule;
        spec.terminal_rule = descriptor.terminal_rule;
        spec.equations = descriptor.constraints;
        sparse_specs.push_back(std::move(spec));
    }
    SparseGkrRelationSpec rounding_spec;
    rounding_spec.relation_id = "rounding/signed_round_div";
    rounding_spec.family_id = "rounding";
    rounding_spec.kind = SparseGkrRelationKind::SignedRoundDiv;
    rounding_spec.item_count = graph.num_nodes;
    rounding_spec.value_width = classes;
    rounding_spec.chunk_size = kSparseGkrChunkEntries;
    rounding_spec.coordinate_rule = "node_gid_then_class_stable[node,class,witness_component]";
    rounding_spec.group_rule = "node-local-fixed-width";
    rounding_spec.equations = {"sigma,z boolean", "zero-product and inverse", "SignedRoundDiv nearest-ties-away equation",
                               "terminal claim binds Aux/output checkpoints"};
    sparse_specs.push_back(std::move(rounding_spec));
    SparseGkrPlan sparse_plan = build_sparse_gkr_plan(graph, plan.graph_topology_digest, plan.static_tables_id, sparse_specs);
    SparseGkrRelationSpec output_binding;
    output_binding.relation_id = "binding/output_aux";
    output_binding.family_id = "output";
    output_binding.kind = SparseGkrRelationKind::OutputBinding;
    output_binding.item_count = graph.num_nodes;
    output_binding.value_width = classes;
    output_binding.chunk_size = kSparseGkrChunkEntries;
    output_binding.coordinate_rule = "Aux/output[row_major_exact,node,class]";
    output_binding.group_rule = "public-Y-prefix-plus-residual-tail";
    output_binding.equations = {"Aux terminal claims bind public output coordinates", "residual terminal claims bind canonical zero tail"};
    sparse_plan.relations.push_back(std::move(output_binding));
    sparse_plan.claim_links.clear();
    for (std::size_t index = 1; index < sparse_plan.relations.size(); ++index) {
        SparseGkrClaimLink link;
        link.from_relation = sparse_plan.relations[index - 1].relation_id;
        link.to_relation = sparse_plan.relations[index].relation_id;
        link.claim_name = "terminal->initial/" + link.from_relation;
        sparse_plan.claim_links.push_back(std::move(link));
    }
    for (const FconnClaimLink& link : plan.fconn_claim_links) {
        SparseGkrClaimLink binding;
        binding.from_relation = "fconn/" + link.relation_id;
        binding.to_relation = "binding/output_aux";
        binding.claim_name = "FCONN:" + link.family_id + ":" + std::to_string(link.output_offset);
        binding.from_terminal = static_cast<std::uint32_t>(link.descriptor_index);
        sparse_plan.binding_links.push_back(std::move(binding));
    }
    plan.sparse_plan_digest = canonical_digest64(sparse_plan.encoding());

    plan.native_relations.clear();
    plan.native_relations.reserve(plan.matmuls.size() + plan.sparse_gkr.size() + 1);
    for (const MatMulDescriptor& matmul : plan.matmuls) {
        NativeRelationDescriptor native;
        native.relation_id = matmul.relation_id;
        native.family_id = matmul.family_id;
        native.primitive = "masked_fconn_sumcheck";
        native.domain_rule = "rows*inner*cols product relation over canonical MLE points";
        native.selector_rule = "descriptor_index_ordered";
        native.coordinate_rule = matmul.output_coordinate_rule;
        native.arithmetic_rule = "C[row,col]=sum_term A[row,term]*B[term,col]";
        native.terminal_rule = matmul.terminal_claim_rule;
        native.value_width = matmul.output_coordinates;
        native.chunk_size = 1ULL << 12;
        native.input_offset = matmul.lhs_offset;
        native.output_offset = matmul.output_offset;
        native.equations = {"lhs_offset/rhs_offset/output_offset bound to descriptor", "FCONN transition order bound to plan.matmuls"};
        plan.native_relations.push_back(std::move(native));
    }
    for (const SparseGkrDescriptor& sparse : plan.sparse_gkr) {
        NativeRelationDescriptor native;
        native.relation_id = sparse.relation_id;
        native.family_id = sparse.family_id;
        native.relation_schema = sparse.relation_schema;
        native.primitive = sparse.relation_id.find("softmax") != std::string::npos ? "masked_sparse_attention_sumcheck" : "masked_sparse_aggregation_sumcheck";
        native.domain_rule = "edge_gid_then_dst_stable_stream";
        native.selector_rule = sparse.group_rule;
        native.coordinate_rule = sparse.coordinate_rule;
        native.arithmetic_rule = sparse.arithmetic_rule;
        native.terminal_rule = sparse.terminal_rule;
        native.edge_count = sparse.edge_count;
        native.value_width = sparse.value_width;
        native.chunk_size = sparse.chunk_size;
        native.input_offset = sparse.input_offset;
        native.output_offset = sparse.output_offset;
        native.equations = sparse.constraints;
        plan.native_relations.push_back(std::move(native));
    }
    NativeRelationDescriptor rounding_relation;
    rounding_relation.relation_id = "rounding/signed_round_div";
    rounding_relation.family_id = "rounding";
    rounding_relation.primitive = "masked_signed_round_div_sumcheck";
    rounding_relation.domain_rule = "canonical checkpoint coordinates";
    rounding_relation.selector_rule = "rounding_recipe_v1";
    rounding_relation.coordinate_rule = "Aux[row_major_exact[node,class,witness_component]]";
    rounding_relation.arithmetic_rule = "signed dividend/denominator, boolean, reciprocal, quotient and remainder equations";
    rounding_relation.terminal_rule = "rounding checkpoints bind to Aux and output coordinates";
    rounding_relation.value_width = plan.residual_coordinates;
    rounding_relation.chunk_size = 1ULL << 12;
    rounding_relation.equations = {"sigma,z boolean", "zero-product and inverse", "toward-zero or nearest-ties-away equation", "nonnegative range limbs"};
    plan.native_relations.push_back(std::move(rounding_relation));
    plan.plan_digest = canonical_digest64(plan_encoding(plan));

    plan.circuit_id = "R_GAT-zkGPT/" + plan.plan_digest;
    plan.official_layered = true;
    plan.has_placeholder = false;
    if (plan.registry.descriptors.empty() || !plan.registry.covers_all_segments()) plan.official_layered = false;
    std::string plan_reason;
    if (!validate_gat_circuit_plan_impl(plan, &plan_reason)) throw std::invalid_argument("canonical GAT proof plan validation failed: " + plan_reason);
    return plan;
}

bool validate_gat_circuit_plan_impl(const GatCircuitPlan& plan, std::string* reason) {
    const auto fail = [reason](const std::string& message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    try {
        if (!plan.native_proof_backing_is_resident())
            return fail("native GAT proof backing is not resident");
        if (plan.virtual_sparse) {
            if (!plan.official_layered || plan.has_placeholder || plan.recipe_schema != "canonical_plan_recipe_v3" ||
                plan.sparse_relation_schema != kSparseGkrRelationSchema || plan.sparse_relation_version != kSparseGkrRelationVersion ||
                plan.sparse_chunk_size != kSparseGkrChunkEntries || !is_lower_hex_digest(plan.sparse_plan_digest) || plan.model_arch_id.empty() ||
                plan.model_arch_id != plan.model_shape.model_arch_id || !canonical_model_shape_valid(plan.model_shape, reason) || plan.graph_num_nodes == 0 ||
                plan.graph_num_features == 0 || plan.graph_count == 0 || plan.graph_num_classes == 0 || plan.graph_edge_count == 0 ||
                !is_lower_hex_digest(plan.graph_topology_digest) || plan.quant_cfg_id.empty() || plan.quant_digest.empty() || plan.static_tables_id.empty() ||
                plan.registry.descriptors.empty() || !plan.registry.covers_all_segments() || !canonical_layout_valid(plan.input_layout) ||
                plan.opening_policy != "same_r_three_terminal_openings_v1" || plan.terminal_openings != std::vector<std::string>{"W@r", "Aux@r", "L_qry@r"} ||
                plan.public_output_coordinates == 0 || plan.residual_coordinates == 0 || !plan.circuit.circuit.empty())
                return fail("virtual sparse plan identity/layout is invalid");
            std::set<std::string> relation_ids;
            bool has_score = false, has_softmax = false, has_aggregate = false;
            bool has_round = false, has_input = false, has_output = false;
            for (const SparseGkrDescriptor& descriptor : plan.sparse_gkr) {
                if (descriptor.relation_id.empty() || !relation_ids.insert(descriptor.relation_id).second || descriptor.edge_count == 0 ||
                    descriptor.value_width == 0 || descriptor.relation_schema != kSparseGkrRelationSchema || descriptor.chunk_size != kSparseGkrChunkEntries ||
                    !sparse_gkr_relation_kind_valid(static_cast<std::uint8_t>(descriptor.relation_kind)) ||
                    descriptor.arithmetic_rule != kSparseGkrArithmeticRule || descriptor.overflow_rule != kSparseGkrOverflowRule ||
                    descriptor.coordinate_rule.empty() || descriptor.terminal_rule.empty() || descriptor.constraints.empty())
                    return fail("virtual sparse descriptor is not canonical");
                has_score = has_score || descriptor.relation_kind == SparseGkrRelationKind::AttentionScore;
                has_softmax = has_softmax || descriptor.relation_kind == SparseGkrRelationKind::MaskedSoftmax;
                has_aggregate = has_aggregate || descriptor.relation_kind == SparseGkrRelationKind::Aggregation;
                has_round = has_round || descriptor.relation_kind == SparseGkrRelationKind::SignedRoundDiv;
                has_input = has_input || descriptor.relation_kind == SparseGkrRelationKind::InputBinding;
                has_output = has_output || descriptor.relation_kind == SparseGkrRelationKind::OutputBinding;
            }
            std::size_t fconn_link_index = 0;
            for (const SparseGkrDescriptor& descriptor : plan.sparse_gkr) {
                if (descriptor.relation_kind != SparseGkrRelationKind::Fconn) continue;
                if (fconn_link_index >= plan.fconn_claim_links.size())
                    return fail("virtual FCONN claim link count does not cover every MatMul relation");
                const FconnClaimLink& link = plan.fconn_claim_links[fconn_link_index];
                const InputSegment* lhs_segment = canonical_segment_for_span(
                    plan.input_layout, descriptor.input_offset,
                    checked_mul(link.rows, link.inner, "virtual FCONN canonical LHS span"));
                const InputSegment* rhs_segment = canonical_segment_for_span(
                    plan.input_layout, descriptor.source_offset,
                    checked_mul(link.inner, link.cols, "virtual FCONN canonical RHS span"));
                const InputSegment* output_segment = canonical_segment_for_span(
                    plan.input_layout, descriptor.output_offset,
                    checked_mul(link.rows, link.cols, "virtual FCONN canonical output span"));
                if (descriptor.relation_id.rfind("fconn/", 0) != 0 ||
                    link.descriptor_index != fconn_link_index ||
                    link.relation_id != descriptor.relation_id.substr(sizeof("fconn/") - 1) ||
                    link.family_id != descriptor.family_id || link.rows != descriptor.edge_count ||
                    checked_mul(link.inner, link.cols, "virtual FCONN descriptor width") != descriptor.value_width ||
                    link.lhs_offset != descriptor.input_offset || link.rhs_offset != descriptor.source_offset ||
                    link.output_offset != descriptor.output_offset || lhs_segment == nullptr || rhs_segment == nullptr ||
                    output_segment == nullptr || link.lhs_segment != lhs_segment->name ||
                    link.rhs_segment != rhs_segment->name || link.output_segment != output_segment->name ||
                    link.output_segment != "Aux" ||
                    link.lhs_point_rule != "fconn/lhs/row-major-mle-v1" ||
                    link.rhs_point_rule != "fconn/rhs/column-major-padded-mle-v1" ||
                    link.output_point_rule != "fconn/output/row-major-axis-padded-mle-v1" ||
                    link.source_rule != "canonical_input_oracle")
                    return fail("virtual FCONN claim link does not match its canonical relation descriptor");
                ++fconn_link_index;
            }
            if (fconn_link_index == 0 || fconn_link_index != plan.fconn_claim_links.size())
                return fail("virtual FCONN claim link count does not match MatMul relations");
            if (plan.sparse_gkr.empty() || !has_score || !has_softmax || !has_aggregate || !has_round || !has_input || !has_output ||
                plan.fconn_claim_links.empty() || plan.native_api.empty())
                return fail("virtual sparse plan omits a required relation family or FCONN binding");
            const std::string digest = canonical_digest64(plan_encoding(plan));
            if (plan.plan_digest != digest || plan.circuit_id != "R_GAT-zkGPT/" + digest) return fail("virtual sparse plan digest is not canonical");
            return true;
        }
        if (!plan.official_layered || plan.has_placeholder) return fail("plan is not an official layered circuit");
        if (plan.recipe_schema != "canonical_plan_recipe_v3" || plan.sparse_relation_schema != kSparseGkrRelationSchema ||
            plan.sparse_relation_version != kSparseGkrRelationVersion || plan.sparse_chunk_size != kSparseGkrChunkEntries ||
            !is_lower_hex_digest(plan.sparse_plan_digest) || plan.model_arch_id.empty() || plan.model_arch_id != plan.model_shape.model_arch_id ||
            !canonical_model_shape_valid(plan.model_shape, reason) || plan.graph_num_nodes == 0 || plan.graph_num_features == 0 || plan.graph_count == 0 ||
            plan.graph_num_classes == 0 || plan.graph_edge_count == 0 || !is_lower_hex_digest(plan.graph_topology_digest) || plan.quant_cfg_id.empty() ||
            plan.quant_digest.empty() || plan.static_tables_id.empty()) {
            return fail("compact canonical plan recipe identity is invalid");
        }
        if (plan.registry.descriptors.empty() || !plan.registry.covers_all_segments()) return fail("relation registry does not cover the canonical relations");
        if (!canonical_layout_valid(plan.input_layout)) return fail("input layout is not the canonical four-segment layout");
        if (plan.opening_policy != "same_r_three_terminal_openings_v1" || plan.terminal_openings != std::vector<std::string>{"W@r", "Aux@r", "L_qry@r"}) {
            return fail("terminal opening policy is not same-r W/Aux/L_qry");
        }
        if (plan.circuit.circuit.size() <= 1 || plan.circuit.size != plan.circuit.circuit.size() || plan.circuit.circuit.front().ty != layerType::INPUT ||
            plan.circuit.circuit.front().size == 0) {
            return fail("layered circuit has an invalid input or layer count");
        }
        if (plan.public_output_coordinates == 0 || plan.residual_coordinates == 0 ||
            plan.circuit.circuit.back().size != plan.public_output_coordinates + plan.residual_coordinates) {
            return fail("final relation layer does not expose Y plus residual coordinates");
        }
        const std::size_t input_size = plan.circuit.circuit.front().size;
        std::size_t fconn_index = 0;
        for (std::size_t layer_id = 1; layer_id < plan.circuit.circuit.size(); ++layer_id) {
            const layer& current = plan.circuit.circuit[layer_id];
            if (current.ty == layerType::PLACE_HOLDER || current.size == 0) return fail("placeholder or empty circuit layer");
            if (current.ty == layerType::FCONN) {
                if (fconn_index >= plan.matmuls.size() || !current.uni_gates.empty() || !current.bin_gates.empty() || !current.ori_id_u.empty() ||
                    !current.ori_id_v.empty() || current.native_relation_id.empty() || current.native_family_id.empty())
                    return fail("FCONN layer is not a compact native MatMul layer");
                const MatMulDescriptor& descriptor = plan.matmuls[fconn_index++];
                if (current.native_relation_id != descriptor.relation_id || current.native_family_id != descriptor.family_id ||
                    current.native_rows != descriptor.rows || current.native_inner != descriptor.inner || current.native_cols != descriptor.cols ||
                    current.native_lhs_offset != descriptor.lhs_offset || current.native_rhs_offset != descriptor.rhs_offset ||
                    current.native_output_offset != descriptor.output_offset ||
                    current.size != fconn_output_domain(descriptor.rows, descriptor.cols))
                    return fail("FCONN layer metadata does not match canonical MatMul order");
                continue;
            }
            const layer& previous = plan.circuit.circuit[layer_id - 1];
            if (previous.ty == layerType::FCONN) {
                const MatMulDescriptor* descriptor = nullptr;
                for (const MatMulDescriptor& candidate : plan.matmuls)
                    if (candidate.relation_id == previous.native_relation_id) {
                        descriptor = &candidate;
                        break;
                    }
                if (descriptor == nullptr) return fail("FCONN binding has no descriptor");
                if (current.size != descriptor->output_coordinates || !current.bin_gates.empty() ||
                    current.uni_gates.size() != checked_mul(static_cast<std::size_t>(current.size), 2, "FCONN binding gate count"))
                    return fail("FCONN output binding layer shape is invalid");
                for (std::size_t index = 0; index < current.size; ++index) {
                    const uniGate& previous_gate = current.uni_gates[index * 2];
                    const uniGate& input_gate = current.uni_gates[index * 2 + 1];
                    const std::size_t padded_index = fconn_padded_output_index(
                        descriptor->rows, descriptor->cols, index);
                    if (previous_gate.g != index || previous_gate.u != padded_index || previous_gate.lu != 1 || current.coefficient(previous_gate) != 1 || input_gate.g != index ||
                        input_gate.lu != 0 || current.coefficient(input_gate) != -1 || input_gate.u >= current.ori_id_u.size() ||
                        current.ori_id_u[input_gate.u] != descriptor->output_offset + index)
                        return fail("FCONN output binding coordinate is not canonical");
                }
                continue;
            }
            for (const uniGate& gate : current.uni_gates) {
                if (gate.g >= current.size || (gate.lu == 0 ? gate.u >= input_size : (gate.lu != 1 || gate.u >= previous.size)))
                    return fail("unary gate coordinate is out of bounds");
            }
            for (const binGate& gate : current.bin_gates) {
                if (gate.g >= current.size || gate.u >= input_size || gate.v >= input_size) return fail("binary gate coordinate is out of bounds");
                if (gate.l != 0) return fail("canonical GAT relation unexpectedly reads a non-input binary layer");
            }
            if (current.uni_gates.empty() && current.bin_gates.empty()) return fail("empty canonical circuit layer");
        }
        if (fconn_index != plan.matmuls.size()) return fail("canonical MatMul descriptor count does not match FCONN layers");
        const auto require_span = [input_size](std::size_t offset, std::size_t length) { return offset <= input_size && length <= input_size - offset; };
        for (const MatMulDescriptor& descriptor : plan.matmuls) {
            const InputSegment* lhs_segment = canonical_segment_for_span(plan.input_layout, descriptor.lhs_offset,
                                                                         checked_mul(descriptor.rows, descriptor.inner, "plan MatMul canonical LHS span"));
            const InputSegment* rhs_segment = canonical_segment_for_span(plan.input_layout, descriptor.rhs_offset,
                                                                         checked_mul(descriptor.inner, descriptor.cols, "plan MatMul canonical RHS span"));
            const InputSegment* output_segment = canonical_segment_for_span(plan.input_layout, descriptor.output_offset, descriptor.output_coordinates);
            if (descriptor.relation_id.empty() || descriptor.rows == 0 || descriptor.inner == 0 || descriptor.cols == 0 ||
                descriptor.output_coordinates != checked_mul(descriptor.rows, descriptor.cols, "plan MatMul output") ||
                !require_span(descriptor.lhs_offset, checked_mul(descriptor.rows, descriptor.inner, "plan MatMul lhs")) ||
                !require_span(descriptor.rhs_offset, checked_mul(descriptor.inner, descriptor.cols, "plan MatMul rhs")) ||
                !require_span(descriptor.output_offset, descriptor.output_coordinates) || lhs_segment == nullptr || rhs_segment == nullptr ||
                output_segment == nullptr || descriptor.lhs_segment != lhs_segment->name || descriptor.rhs_segment != rhs_segment->name ||
                descriptor.output_segment != output_segment->name || descriptor.output_segment != "Aux" ||
                descriptor.coordinate_rule.find("row_major_exact") == std::string::npos || descriptor.scale_rule.empty() || descriptor.native_path.empty()) {
                return fail("MatMul descriptor shape, coordinate, or scale binding is invalid");
            }
        }
        if (plan.matmuls.empty()) return fail("canonical plan has no MatMul descriptors");
        if (plan.fconn_claim_links.size() != plan.matmuls.size())
            return fail("canonical FCONN claim link count does not match MatMul descriptors");
        for (std::size_t index = 0; index < plan.matmuls.size(); ++index) {
            const MatMulDescriptor& descriptor = plan.matmuls[index];
            const FconnClaimLink& link = plan.fconn_claim_links[index];
            if (link.descriptor_index != index || link.relation_id != descriptor.relation_id ||
                link.family_id != descriptor.family_id || link.lhs_segment != descriptor.lhs_segment ||
                link.rhs_segment != descriptor.rhs_segment || link.output_segment != descriptor.output_segment ||
                link.lhs_offset != descriptor.lhs_offset || link.rhs_offset != descriptor.rhs_offset ||
                link.output_offset != descriptor.output_offset || link.rows != descriptor.rows ||
                link.inner != descriptor.inner || link.cols != descriptor.cols ||
                link.lhs_point_rule != "fconn/lhs/row-major-mle-v1" ||
                link.rhs_point_rule != "fconn/rhs/column-major-padded-mle-v1" ||
                link.output_point_rule != "fconn/output/row-major-axis-padded-mle-v1" ||
                link.source_rule != "canonical_input_oracle")
                return fail("canonical FCONN claim link does not match its MatMul descriptor");
        }
        for (const SparseGkrDescriptor& descriptor : plan.sparse_gkr) {
            if (descriptor.relation_id.empty() || descriptor.edge_count == 0 || descriptor.value_width == 0 ||
                descriptor.relation_schema != kSparseGkrRelationSchema || descriptor.chunk_size != kSparseGkrChunkEntries ||
                !sparse_gkr_relation_kind_valid(static_cast<std::uint8_t>(descriptor.relation_kind)) ||
                descriptor.arithmetic_rule != kSparseGkrArithmeticRule || descriptor.overflow_rule != kSparseGkrOverflowRule ||
                descriptor.terminal_rule.empty() || !require_span(descriptor.input_offset, 1) || !require_span(descriptor.source_offset, 1) ||
                !require_span(descriptor.destination_offset, 1) || !require_span(descriptor.output_offset, 1) ||
                descriptor.coordinate_rule.find("edge_gid_then_dst_stable") == std::string::npos ||
                descriptor.group_rule.find("Q_grp_new") == std::string::npos || descriptor.group_rule.find("Q_grp_end") == std::string::npos ||
                descriptor.constraints.empty()) {
                return fail("sparse GKR descriptor is not canonically bound");
            }
            bool has_group_recurrence = false;
            bool has_field_relation = false;
            bool has_round_relation = false;
            for (const std::string& constraint : descriptor.constraints) {
                if (constraint.find("Q_grp_new") != std::string::npos && constraint.find("Q_grp_end") != std::string::npos) has_group_recurrence = true;
                if (constraint.find("field") != std::string::npos) has_field_relation = true;
                if (constraint.find("SignedRoundDiv") != std::string::npos) has_round_relation = true;
                if (constraint.find("modulo") != std::string::npos || constraint.find("neighbor") != std::string::npos)
                    return fail("sparse descriptor contains a placeholder coordinate rule");
            }
            if (descriptor.relation_id.find("softmax") != std::string::npos && !has_group_recurrence) return fail("softmax descriptor lacks group recurrence");
            if (descriptor.relation_id.find("aggregation") != std::string::npos && (!has_field_relation || !has_round_relation))
                return fail("aggregation descriptor lacks field or SignedRoundDiv relation");
        }
        if (plan.sparse_gkr.empty()) return fail("canonical plan has no sparse GKR descriptors");
        const InputSegment& query_segment = plan.input_layout.segments.at(3);
        std::size_t query_cursor = query_segment.offset;
        if (plan.lasso.size() != 4) return fail("canonical plan must contain four Lasso families");
        for (const LassoFamilyDescriptor& descriptor : plan.lasso) {
            if (descriptor.padded_queries != 0) {
                const std::size_t local_offset = query_cursor - query_segment.offset;
                const std::size_t remainder = local_offset % descriptor.padded_queries;
                if (remainder != 0) query_cursor = checked_add(query_cursor, descriptor.padded_queries - remainder, "validated aligned Lasso query gap");
            }
            if (descriptor.relation_id.empty() || descriptor.real_queries == 0 || descriptor.padded_queries != next_power_of_two(descriptor.real_queries) ||
                descriptor.table_domain == 0 || (descriptor.table_domain & (descriptor.table_domain - 1)) != 0 || descriptor.query_offset != query_cursor ||
                ((descriptor.query_offset - query_segment.offset) % descriptor.padded_queries) != 0 ||
                !require_span(descriptor.query_offset, descriptor.padded_queries) ||
                descriptor.query_offset + descriptor.padded_queries > query_segment.offset + query_segment.logical_length ||
                descriptor.incidence_rule.find("L_qry/") != 0 || descriptor.incidence_rule.find("canonical_operation_index_row_major") == std::string::npos ||
                descriptor.challenge_policy != "LC_before_challenge" || descriptor.query_encoding.empty()) {
                std::ostringstream detail;
                detail << "Lasso query block is not canonical or not contiguous"
                       << " family=" << descriptor.family_id << " off=" << descriptor.query_offset << " cursor=" << query_cursor
                       << " seg=" << query_segment.offset << " logical=" << query_segment.logical_length << " real=" << descriptor.real_queries
                       << " padded=" << descriptor.padded_queries;
                return fail(detail.str());
            }
            query_cursor = checked_add(query_cursor, descriptor.padded_queries, "validated Lasso query cursor");
        }
        if (query_cursor != query_segment.offset + query_segment.logical_length) return fail("Lasso blocks do not cover L_qry logical length");
        std::set<std::string> checkpoint_ids;
        const auto find_segment = [&plan](const std::string& name) -> const InputSegment* {
            for (const InputSegment& segment : plan.input_layout.segments)
                if (segment.name == name) return &segment;
            return nullptr;
        };
        if (plan.checkpoints.empty()) return fail("canonical plan has no checkpoints");
        for (const CheckpointDescriptor& checkpoint : plan.checkpoints) {
            if (checkpoint.id.empty() || !checkpoint_ids.insert(checkpoint.id).second || checkpoint.coordinates == 0 ||
                checkpoint.coordinate_rule.find("row_major_exact") == std::string::npos)
                return fail("checkpoint identity or coordinate rule is invalid");
            const InputSegment* segment = find_segment(checkpoint.segment);
            if (segment == nullptr || checkpoint.coordinate_offset < segment->offset || checkpoint.coordinates > segment->logical_length ||
                checkpoint.coordinate_offset - segment->offset > segment->logical_length - checkpoint.coordinates ||
                checkpoint.coordinate_offset + checkpoint.coordinates > segment->offset + segment->logical_length)
                return fail("checkpoint exceeds its canonical segment");
        }
        const std::string digest = canonical_digest64(plan_encoding(plan));
        if (plan.plan_digest != digest || plan.circuit_id != "R_GAT-zkGPT/" + digest) return fail("plan digest does not bind canonical descriptors and wiring");
        return true;
    } catch (const std::exception& error) { return fail(error.what()); }
}

std::string commitment_encoding(const CommitmentWire& commitment) {
    std::string encoded;
    if (commitment.has_tensor_commitment) {
        const hyrax::HyraxTensorMleCommitment& tensor =
            commitment.tensor_commitment;
        append_string(encoded, "zkgpt-gat-balanced-tensor-commitment-wire-v2");
        append_string(encoded, commitment.scheme);
        append_string(encoded, commitment.basis_id);
        append_u64(encoded, commitment.logical_size);
        append_u64(encoded, commitment.padded_size);
        append_string(encoded, tensor.schema);
        append_string(encoded, tensor.method);
        append_u64(encoded, tensor.version);
        append_string(encoded, tensor.role);
        append_string(encoded, tensor.basis_id);
        append_u64(encoded,
                    static_cast<std::uint64_t>(tensor.column_generator_scheme));
        append_string(encoded, tensor.column_generator_id);
        append_u64(encoded, tensor.full_domain);
        append_u64(encoded, tensor.logical_length);
        append_u64(encoded, tensor.active_domain);
        append_u64(encoded, tensor.rows);
        append_u64(encoded, tensor.columns);
        append_string(encoded, tensor.root_digest);
        append_u64(encoded, tensor.row_commitments.size());
        for (const G1& row_commitment : tensor.row_commitments)
            append_string(encoded, row_commitment.serializeToHexStr());
        return encoded;
    }
    append_string(encoded, "zkgpt-gat-commitment-wire-v2");
    append_string(encoded, commitment.scheme);
    append_string(encoded, commitment.basis_id);
    append_u64(encoded, commitment.logical_size);
    append_u64(encoded, commitment.padded_size);
    append_string(encoded, commitment.point_hex);
    return encoded;
}

std::string commitment_statement_digest(const CommitmentWire& commitment) {
    if (commitment.has_tensor_commitment &&
        commitment.tensor_commitment.role == "canonical/L_qry") {
        hyrax::HyraxTensorMleView view;
        view.schema = commitment.tensor_commitment.schema;
        view.method = commitment.tensor_commitment.method;
        view.version = commitment.tensor_commitment.version;
        view.full_domain = commitment.tensor_commitment.full_domain;
        view.logical_length = commitment.tensor_commitment.logical_length;
        view.active_domain = commitment.tensor_commitment.active_domain;
        view.rows = commitment.tensor_commitment.rows;
        view.columns = commitment.tensor_commitment.columns;
        view.role = commitment.tensor_commitment.role;
        view.basis_id = commitment.tensor_commitment.basis_id;
        view.column_generators.size = commitment.tensor_commitment.columns;
        view.column_generators.canonical_id =
            commitment.tensor_commitment.column_generator_id;
        view.column_generators.scheme =
            commitment.tensor_commitment.column_generator_scheme;
        return native_lasso_tensor_query_digest(view,
                                                commitment.tensor_commitment);
    }
    return fixed_digest(hash_hex(commitment_encoding(commitment)));
}

void append_registration_encoding(std::string& encoded, const ModelRegistration& registration) {
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
    for (const InputSegment& segment : registration.input_segments) {
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

std::string section_encoding(const ProofSection& section) {
    std::string encoded;
    append_string(encoded, section.protocol);
    append_u64(encoded, section.messages.size());
    for (const TranscriptMessage& message : section.messages) {
        append_string(encoded, message.label);
        append_string(encoded, message.payload);
        append_u64(encoded, message.kind);
        append_u64(encoded, message.sequence);
    }
    return encoded;
}

std::string artifact_transcript_digest(const GatProofArtifact& artifact) {
    NativeProofSession session;
    std::string reason;
    if (!NativeProofSession::from_messages(artifact.native_domain, artifact.native_statement, artifact.native_messages, session, &reason)) {
        return std::string();
    }
    return session.digest();
}

CommitmentWire commit_segment_with_mask(const std::vector<Int>& values, const std::string& basis_label, std::size_t padded_domain, Fr* mask_out,
                                        std::vector<Fr>* masked_values_out) {
    ensure_field();
    if (values.empty() || basis_label.empty() || padded_domain == 0 || (padded_domain & (padded_domain - 1)) != 0 || values.size() > padded_domain) {
        throw std::invalid_argument("invalid masked Hyrax segment domain");
    }
    const std::string basis_id = fixed_digest(hash_hex(basis_label));
    const G1 H = hyrax::derive_pedersen_blinding_generator(basis_id);
    const std::size_t point_bits = core_log2(padded_domain);
    std::vector<Fr> zero_point(point_bits, Fr(0));
    const std::string canonical_id = NativeProofStatement::basis_descriptor_digest_for(H, basis_label, basis_id, kCanonicalGeneratorDerivationId, padded_domain);
    const hyrax::HyraxMleBasisView view = canonical_input_opening_view(basis_label, basis_id, canonical_id, "canonical/segment", padded_domain, values.size(), zero_point);
    hyrax::HyraxScalarProvider scalars;
    scalars.size = padded_domain;
    scalars.at = [&values, padded_domain](std::uint64_t index) {
        if (index >= padded_domain) throw std::out_of_range("canonical segment scalar index is invalid");
        return index < values.size() ? Fr(values[static_cast<std::size_t>(index)]) : Fr(0);
    };
    const hyrax::PedersenMleCommitment commitment = hyrax::pedersen_commit_mle(view, scalars);
    if (mask_out != nullptr) *mask_out = commitment.blinding;
    if (masked_values_out != nullptr) {
        masked_values_out->assign(padded_domain, Fr(0));
        for (std::size_t index = 0; index < values.size(); ++index) (*masked_values_out)[index] = Fr(values[index]);
    }
    CommitmentWire result;
    result.basis_id = basis_id;
    result.logical_size = values.size();
    result.padded_size = padded_domain;
    result.point_hex = point_hex(commitment.commitment);
    return result;
}

CommitmentWire commit_field_segment_with_mask(const std::vector<Fr>& values, const std::string& basis_label, std::size_t padded_domain,
                                              std::vector<Fr>* masked_values_out) {
    ensure_field();
    if (values.empty() || basis_label.empty() || padded_domain == 0 || (padded_domain & (padded_domain - 1)) != 0 || values.size() > padded_domain)
        throw std::invalid_argument("invalid field segment commitment domain");
    const std::string basis_id = fixed_digest(hash_hex(basis_label));
    const G1 H = hyrax::derive_pedersen_blinding_generator(basis_id);
    const std::size_t point_bits = core_log2(padded_domain);
    std::vector<Fr> zero_point(point_bits, Fr(0));
    const std::string canonical_id = NativeProofStatement::basis_descriptor_digest_for(H, basis_label, basis_id, kCanonicalGeneratorDerivationId, padded_domain);
    const hyrax::HyraxMleBasisView view = canonical_input_opening_view(basis_label, basis_id, canonical_id, "canonical/segment", padded_domain, values.size(), zero_point);
    hyrax::HyraxScalarProvider scalars;
    scalars.size = padded_domain;
    scalars.at = [&values, padded_domain](std::uint64_t index) {
        if (index >= padded_domain) throw std::out_of_range("canonical field-segment scalar index is invalid");
        return index < values.size() ? values[static_cast<std::size_t>(index)] : Fr(0);
    };
    const hyrax::PedersenMleCommitment commitment = hyrax::pedersen_commit_mle(view, scalars);
    if (masked_values_out != nullptr) {
        masked_values_out->assign(padded_domain, Fr(0));
        std::copy(values.begin(), values.end(), masked_values_out->begin());
    }
    CommitmentWire result;
    result.basis_id = basis_id;
    result.logical_size = values.size();
    result.padded_size = padded_domain;
    result.point_hex = point_hex(commitment.commitment);
    return result;
}

std::string registration_basis_label_public(const ModelRegistration& registration, const InputLayout& layout) {
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

std::vector<Fr> adapter_mle_basis(const std::vector<Fr>& point) {
    if (point.empty() || point.size() >= sizeof(std::size_t) * 8) throw std::invalid_argument("adapter MLE point dimension is invalid");
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

void append_division_components(NativeFieldVector& values, const DivisionWitness& division) {
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

void append_attention_aux_values(NativeFieldVector& values, const AttentionTrace& trace) {
    const auto append_ints = [&values](const std::vector<Int>& source) {
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
    for (const DivisionWitness& division : trace.divisions) append_division_components(values, division);
}

NativeFieldVector flatten_aux_values(const ForwardResult& forward) {
    NativeFieldVector values;
    values.reserve(trace_aux_size(&forward.trace));
    for (Int value : forward.trace.input.values) values.push_back(Fr(value));
    for (const HiddenLayerTrace& layer : forward.trace.hidden_layers) {
        for (Int value : layer.concatenated.values) values.push_back(Fr(value));
        for (const AttentionTrace& head : layer.heads) append_attention_aux_values(values, head);
    }
    for (const OutputHeadTrace& head : forward.trace.output_heads) {
        append_attention_aux_values(values, head.attention);
        for (Int value : head.bias) values.push_back(Fr(value));
        for (Int value : head.aggregate) values.push_back(Fr(value));
        for (Int value : head.aligned) values.push_back(Fr(value));
        for (Int value : head.logits) values.push_back(Fr(value));
        for (const DivisionWitness& division : head.alignment_divisions) append_division_components(values, division);
    }
    for (Int value : forward.trace.head_sum) values.push_back(Fr(value));
    for (const DivisionWitness& division : forward.trace.final_average_divisions) append_division_components(values, division);
    for (Int value : forward.trace.logits.values) values.push_back(Fr(value));
    return values;
}

struct AuxValueSpan {
    enum class Kind { Integers, Fields, Divisions } kind = Kind::Integers;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    const std::vector<Int>* integers = nullptr;
    const std::vector<Field>* fields = nullptr;
    const std::vector<DivisionWitness>* divisions = nullptr;
};

struct AuxValueProviderState {
    static constexpr std::uint64_t chunk_entries = 1ULL << 12;
    std::uint64_t size = 0;
    std::vector<AuxValueSpan> spans;
    std::vector<std::size_t> chunk_spans;

    Fr value_at(std::uint64_t index) const {
        if (index >= size) throw std::out_of_range("borrowed canonical Aux coordinate is invalid");
        std::size_t span_index = chunk_spans.at(static_cast<std::size_t>(index / chunk_entries));
        while (span_index < spans.size() && index >= spans[span_index].end) ++span_index;
        if (span_index >= spans.size() || index < spans[span_index].begin) throw std::logic_error("borrowed canonical Aux span map is inconsistent");
        const AuxValueSpan& span = spans[span_index];
        const std::uint64_t local = index - span.begin;
        if (span.kind == AuxValueSpan::Kind::Integers) return Fr(span.integers->at(static_cast<std::size_t>(local)));
        if (span.kind == AuxValueSpan::Kind::Fields) return span.fields->at(static_cast<std::size_t>(local));
        const DivisionWitness& division = span.divisions->at(static_cast<std::size_t>(local / 9));
        switch (local % 9) {
        case 0: return Fr(division.sigma);
        case 1: return Fr(division.z);
        case 2: return Fr(division.a);
        case 3: return Fr(division.q);
        case 4: return Fr(division.r);
        case 5: return Fr(division.delta);
        case 6: return division.a_inverse;
        case 7: return division.d_inverse;
        default: return Fr(division.denominator);
        }
    }
};

std::shared_ptr<const AuxValueProviderState> make_borrowed_aux_value_provider(const ForwardResult& forward) {
    auto state = std::make_shared<AuxValueProviderState>();
    const auto add_span = [&state](AuxValueSpan span, std::uint64_t count) {
        if (count == 0) return;
        if (count > std::numeric_limits<std::uint64_t>::max() - state->size) throw std::overflow_error("borrowed canonical Aux extent overflows u64");
        span.begin = state->size;
        span.end = state->size + count;
        state->spans.push_back(span);
        state->size = span.end;
    };
    const auto add_integers = [&add_span](const std::vector<Int>& values) { AuxValueSpan span; span.kind = AuxValueSpan::Kind::Integers; span.integers = &values; add_span(span, values.size()); };
    const auto add_fields = [&add_span](const std::vector<Field>& values) { AuxValueSpan span; span.kind = AuxValueSpan::Kind::Fields; span.fields = &values; add_span(span, values.size()); };
    const auto add_divisions = [&add_span](const std::vector<DivisionWitness>& values) {
        if (values.size() > std::numeric_limits<std::uint64_t>::max() / 9) throw std::overflow_error("borrowed canonical Aux division extent overflows u64");
        AuxValueSpan span; span.kind = AuxValueSpan::Kind::Divisions; span.divisions = &values; add_span(span, static_cast<std::uint64_t>(values.size()) * 9);
    };
    const auto add_attention = [&add_integers, &add_fields, &add_divisions](const AttentionTrace& trace) {
        add_integers(trace.projection); add_integers(trace.source_score); add_integers(trace.destination_score); add_integers(trace.raw_score);
        add_integers(trace.leaky_score); add_integers(trace.group_max); add_integers(trace.delta); add_integers(trace.exp); add_integers(trace.group_sum);
        add_fields(trace.group_inverse); add_integers(trace.max_selector); add_fields(trace.alpha); add_integers(trace.integer_numerator);
        add_fields(trace.field_numerator); add_fields(trace.field_aggregate); add_integers(trace.recovered); add_integers(trace.activated); add_divisions(trace.divisions);
    };
    add_integers(forward.trace.input.values);
    for (const HiddenLayerTrace& layer : forward.trace.hidden_layers) {
        add_integers(layer.concatenated.values);
        for (const AttentionTrace& head : layer.heads) add_attention(head);
    }
    for (const OutputHeadTrace& head : forward.trace.output_heads) {
        add_attention(head.attention); add_integers(head.bias); add_integers(head.aggregate); add_integers(head.aligned); add_integers(head.logits); add_divisions(head.alignment_divisions);
    }
    add_integers(forward.trace.head_sum); add_divisions(forward.trace.final_average_divisions); add_integers(forward.trace.logits.values);
    if (state->size != trace_aux_size(&forward.trace)) throw std::logic_error("borrowed canonical Aux span total differs from trace descriptor");
    const std::uint64_t chunk_count = state->size / AuxValueProviderState::chunk_entries + (state->size % AuxValueProviderState::chunk_entries != 0);
    if (chunk_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) throw std::length_error("borrowed canonical Aux span map exceeds host size");
    state->chunk_spans.resize(static_cast<std::size_t>(chunk_count));
    std::size_t span_index = 0;
    for (std::uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
        const std::uint64_t chunk_begin = chunk * AuxValueProviderState::chunk_entries;
        while (span_index + 1 < state->spans.size() && state->spans[span_index].end <= chunk_begin) ++span_index;
        if (span_index >= state->spans.size() || state->spans[span_index].begin > chunk_begin) throw std::logic_error("borrowed canonical Aux chunk map has a gap");
        state->chunk_spans[static_cast<std::size_t>(chunk)] = span_index;
    }
    return state;
}

Wide abs_wide_adapter(Int value) { return value < 0 ? -static_cast<Wide>(value) : static_cast<Wide>(value); }

Fr packed_table_value(Int x, Int y, Int bx, Int by) {
    const Wide base = static_cast<Wide>(2) * static_cast<Wide>(by) + 1;
    const Wide packed = (static_cast<Wide>(x) + bx) * base + (static_cast<Wide>(y) + by);
    if (packed < 0 || packed > static_cast<Wide>(std::numeric_limits<Int>::max()))
        throw std::overflow_error("packed lookup tuple exceeds adapter integer range");
    return Fr(static_cast<Int>(packed));
}

void append_nonnegative_bits(std::vector<Fr>& values, Wide input, unsigned width) {
    if (input < 0 || input > static_cast<Wide>(std::numeric_limits<Int>::max())) throw std::invalid_argument("canonical range value is outside the nonnegative signed-63-bit domain");
    if (input >= (static_cast<Wide>(1) << width)) throw std::invalid_argument("canonical range value exceeds its authenticated public bit bound");
    for (unsigned bit = 0; bit < width; ++bit) values.push_back(Fr(static_cast<Int>((input >> bit) & 1)));
}

std::pair<Int, Int> lookup_rectangle(const std::vector<TablePoint>& points) {
    Int bx = 0, by = 0;
    for (const TablePoint& point : points) { bx = std::max<Int>(bx, static_cast<Int>(abs_wide_adapter(point.x))); by = std::max<Int>(by, static_cast<Int>(abs_wide_adapter(point.y))); }
    return {bx, by};
}

StaticLookupBounds static_lookup_bounds(const StaticTables& tables) {
    return {lookup_rectangle(tables.lrelu), lookup_rectangle(tables.exp), lookup_rectangle(tables.elu)};
}

void append_tuple_range_queries(std::vector<Fr>& values, Int x, Int y, const std::pair<Int, Int>& bounds) {
    append_nonnegative_bits(values, static_cast<Wide>(x) + bounds.first, nonnegative_bit_width(2 * static_cast<Wide>(bounds.first)));
    append_nonnegative_bits(values, static_cast<Wide>(bounds.first) - x, nonnegative_bit_width(2 * static_cast<Wide>(bounds.first)));
    append_nonnegative_bits(values, static_cast<Wide>(y) + bounds.second, nonnegative_bit_width(2 * static_cast<Wide>(bounds.second)));
    append_nonnegative_bits(values, static_cast<Wide>(bounds.second) - y, nonnegative_bit_width(2 * static_cast<Wide>(bounds.second)));
}

void append_division_range_queries(std::vector<Fr>& values, const DivisionWitness& division, const std::array<unsigned, 5>& profile) {
    const std::array<Int, 5> values_to_range{{division.a, division.q, division.r, division.delta, division.denominator}};
    for (std::size_t item = 0; item < values_to_range.size(); ++item) append_nonnegative_bits(values, values_to_range[item], profile[item]);
}

void append_range_attention_queries(std::vector<Fr>& values, const AttentionTrace& trace, bool hidden, const StaticLookupBounds& bounds, const std::array<unsigned, 5>& profile) {
    for (std::size_t index = 0; index < trace.raw_score.size(); ++index) append_tuple_range_queries(values, trace.raw_score[index], trace.leaky_score[index], bounds.lrelu);
    for (std::size_t index = 0; index < trace.delta.size(); ++index) append_tuple_range_queries(values, trace.delta[index], trace.exp[index], bounds.exp);
    if (hidden) for (std::size_t index = 0; index < trace.recovered.size(); ++index) append_tuple_range_queries(values, trace.recovered[index], trace.activated[index], bounds.elu);
    for (Int value : trace.delta) append_nonnegative_bits(values, value, nonnegative_bit_width(2 * static_cast<Wide>(bounds.lrelu.second)));
    for (const DivisionWitness& division : trace.divisions) append_division_range_queries(values, division, profile);
}

std::vector<Fr> canonical_range_query_values(const GatCircuitPlan& plan, const ForwardResult& forward, const StaticLookupBounds& bounds) {
    std::vector<Fr> values;
    std::size_t profile_index = 0;
    for (const HiddenLayerTrace& layer : forward.trace.hidden_layers) for (const AttentionTrace& head : layer.heads) append_range_attention_queries(values, head, true, bounds, plan.division_range_bits.at(profile_index++));
    for (const OutputHeadTrace& head : forward.trace.output_heads) {
        append_range_attention_queries(values, head.attention, false, bounds, plan.division_range_bits.at(profile_index++));
        const auto& profile = plan.division_range_bits.at(profile_index++);
        for (const DivisionWitness& division : head.alignment_divisions) append_division_range_queries(values, division, profile);
    }
    const auto& profile = plan.division_range_bits.at(profile_index++);
    for (const DivisionWitness& division : forward.trace.final_average_divisions) append_division_range_queries(values, division, profile);
    if (profile_index != plan.division_range_bits.size()) throw std::logic_error("canonical range profile coverage is incomplete");
    return values;
}

template <class IndexVector>
void append_nonnegative_bits_indices(IndexVector& indices, Wide input,
                                     unsigned width) {
    if (input < 0 || input > static_cast<Wide>(std::numeric_limits<Int>::max()))
        throw std::invalid_argument("canonical range value is outside the nonnegative signed-63-bit domain");
    if (input >= (static_cast<Wide>(1) << width))
        throw std::invalid_argument("canonical range value exceeds its authenticated public bit bound");
    for (unsigned bit = 0; bit < width; ++bit)
        indices.push_back(static_cast<std::uint64_t>((input >> bit) & 1));
}

template <class IndexVector>
void append_tuple_range_indices(IndexVector& indices, Int x, Int y,
                                const std::pair<Int, Int>& bounds) {
    append_nonnegative_bits_indices(indices, static_cast<Wide>(x) + bounds.first,
                                    nonnegative_bit_width(2 * static_cast<Wide>(bounds.first)));
    append_nonnegative_bits_indices(indices, static_cast<Wide>(bounds.first) - x,
                                    nonnegative_bit_width(2 * static_cast<Wide>(bounds.first)));
    append_nonnegative_bits_indices(indices, static_cast<Wide>(y) + bounds.second,
                                    nonnegative_bit_width(2 * static_cast<Wide>(bounds.second)));
    append_nonnegative_bits_indices(indices, static_cast<Wide>(bounds.second) - y,
                                    nonnegative_bit_width(2 * static_cast<Wide>(bounds.second)));
}

template <class IndexVector>
void append_division_range_indices(IndexVector& indices,
                                   const DivisionWitness& division,
                                   const std::array<unsigned, 5>& profile) {
    const std::array<Int, 5> values_to_range{{division.a, division.q, division.r,
                                              division.delta, division.denominator}};
    for (std::size_t item = 0; item < values_to_range.size(); ++item)
        append_nonnegative_bits_indices(indices, values_to_range[item], profile[item]);
}

template <class IndexVector>
void append_range_attention_indices(IndexVector& indices,
                                    const AttentionTrace& trace, bool hidden,
                                    const StaticLookupBounds& bounds,
                                    const std::array<unsigned, 5>& profile) {
    for (std::size_t index = 0; index < trace.raw_score.size(); ++index)
        append_tuple_range_indices(indices, trace.raw_score[index], trace.leaky_score[index], bounds.lrelu);
    for (std::size_t index = 0; index < trace.delta.size(); ++index)
        append_tuple_range_indices(indices, trace.delta[index], trace.exp[index], bounds.exp);
    if (hidden)
        for (std::size_t index = 0; index < trace.recovered.size(); ++index)
            append_tuple_range_indices(indices, trace.recovered[index], trace.activated[index], bounds.elu);
    const unsigned delta_width = nonnegative_bit_width(2 * static_cast<Wide>(bounds.lrelu.second));
    for (Int value : trace.delta) append_nonnegative_bits_indices(indices, value, delta_width);
    for (const DivisionWitness& division : trace.divisions)
        append_division_range_indices(indices, division, profile);
}

template <class IndexVector>
void append_canonical_range_query_indices(IndexVector& indices,
                                          const GatCircuitPlan& plan,
                                          const ForwardResult& forward,
                                          const StaticTables& tables) {
    const StaticLookupBounds bounds = static_lookup_bounds(tables);
    std::size_t profile_index = 0;
    for (const HiddenLayerTrace& layer : forward.trace.hidden_layers)
        for (const AttentionTrace& head : layer.heads)
            append_range_attention_indices(indices, head, true, bounds,
                                           plan.division_range_bits.at(profile_index++));
    for (const OutputHeadTrace& head : forward.trace.output_heads) {
        append_range_attention_indices(indices, head.attention, false, bounds,
                                       plan.division_range_bits.at(profile_index++));
        const auto& profile = plan.division_range_bits.at(profile_index++);
        for (const DivisionWitness& division : head.alignment_divisions)
            append_division_range_indices(indices, division, profile);
    }
    const auto& profile = plan.division_range_bits.at(profile_index++);
    for (const DivisionWitness& division : forward.trace.final_average_divisions)
        append_division_range_indices(indices, division, profile);
    if (profile_index != plan.division_range_bits.size())
        throw std::logic_error("canonical range profile coverage is incomplete");
}

struct CanonicalLassoPairBlock {
    std::uint64_t begin = 0;
    std::uint64_t count = 0;
    const std::vector<Int>* x = nullptr;
    const std::vector<Int>* y = nullptr;
};

std::uint64_t canonical_lasso_pair_row(
    const std::vector<CanonicalLassoPairBlock>& blocks,
    const std::map<std::pair<Int, Int>, std::uint64_t>& table_indices,
    std::uint64_t real_queries, std::uint64_t padded_queries,
    std::uint64_t padding_index, std::uint64_t row) {
    if (row >= padded_queries)
        throw std::out_of_range("canonical Lasso selected query row is invalid");
    if (row >= real_queries) return padding_index;
    const auto found_block = std::upper_bound(
        blocks.begin(), blocks.end(), row,
        [](std::uint64_t index, const CanonicalLassoPairBlock& block) {
            return index < block.begin;
        });
    if (found_block == blocks.begin())
        throw std::out_of_range("canonical function Lasso query row has no source block");
    const CanonicalLassoPairBlock& block = *std::prev(found_block);
    const std::uint64_t local = row - block.begin;
    if (local >= block.count || block.x == nullptr || block.y == nullptr ||
        local >= block.x->size() || local >= block.y->size())
        throw std::out_of_range("canonical function Lasso query source row is invalid");
    const auto found = table_indices.find({block.x->at(static_cast<std::size_t>(local)),
                                           block.y->at(static_cast<std::size_t>(local))});
    if (found == table_indices.end())
        throw std::invalid_argument("canonical Lasso query is absent from static table");
    return found->second;
}

struct CanonicalRangeBitBlock {
    enum class Kind { Tuple, Vector, Division } kind = Kind::Vector;
    std::uint64_t begin = 0;
    std::uint64_t count = 0;
    const std::vector<Int>* first = nullptr;
    const std::vector<Int>* second = nullptr;
    const std::vector<DivisionWitness>* divisions = nullptr;
    Int bound_x = 0;
    Int bound_y = 0;
    unsigned width_x = 0;
    unsigned width_y = 0;
    std::array<unsigned, 5> division_widths{};
};

std::array<Int, 5> canonical_division_values(const DivisionWitness& division) {
    return {{division.a, division.q, division.r, division.delta, division.denominator}};
}

void append_canonical_range_tuple_block(
    std::vector<CanonicalRangeBitBlock>& blocks, std::uint64_t& cursor,
    const std::vector<Int>& x, const std::vector<Int>& y,
    const std::pair<Int, Int>& bounds) {
    if (x.size() != y.size())
        throw std::logic_error("canonical range tuple source dimensions differ");
    const unsigned width_x = nonnegative_bit_width(2 * static_cast<Wide>(bounds.first));
    const unsigned width_y = nonnegative_bit_width(2 * static_cast<Wide>(bounds.second));
    const std::uint64_t bits_per_tuple = 2ULL * width_x + 2ULL * width_y;
    if (bits_per_tuple == 0 || x.size() > std::numeric_limits<std::uint64_t>::max() / bits_per_tuple)
        throw std::overflow_error("canonical range tuple query block overflows u64");
    CanonicalRangeBitBlock block;
    block.kind = CanonicalRangeBitBlock::Kind::Tuple;
    block.begin = cursor;
    block.count = static_cast<std::uint64_t>(x.size()) * bits_per_tuple;
    block.first = &x;
    block.second = &y;
    block.bound_x = bounds.first;
    block.bound_y = bounds.second;
    block.width_x = width_x;
    block.width_y = width_y;
    if (block.count > std::numeric_limits<std::uint64_t>::max() - cursor)
        throw std::overflow_error("canonical range tuple query cursor overflows u64");
    cursor += block.count;
    if (block.count != 0) blocks.push_back(block);
}

void append_canonical_range_vector_block(
    std::vector<CanonicalRangeBitBlock>& blocks, std::uint64_t& cursor,
    const std::vector<Int>& values, unsigned width) {
    if (width == 0 || values.size() > std::numeric_limits<std::uint64_t>::max() / width)
        throw std::overflow_error("canonical range vector query block overflows u64");
    CanonicalRangeBitBlock block;
    block.kind = CanonicalRangeBitBlock::Kind::Vector;
    block.begin = cursor;
    block.count = static_cast<std::uint64_t>(values.size()) * width;
    block.first = &values;
    block.width_x = width;
    if (block.count > std::numeric_limits<std::uint64_t>::max() - cursor)
        throw std::overflow_error("canonical range vector query cursor overflows u64");
    cursor += block.count;
    if (block.count != 0) blocks.push_back(block);
}

void append_canonical_range_division_block(
    std::vector<CanonicalRangeBitBlock>& blocks, std::uint64_t& cursor,
    const std::vector<DivisionWitness>& divisions,
    const std::array<unsigned, 5>& profile) {
    const std::uint64_t bits_per_division = std::accumulate(
        profile.begin(), profile.end(), std::uint64_t(0));
    if (bits_per_division == 0 || divisions.size() > std::numeric_limits<std::uint64_t>::max() / bits_per_division)
        throw std::overflow_error("canonical range division query block overflows u64");
    CanonicalRangeBitBlock block;
    block.kind = CanonicalRangeBitBlock::Kind::Division;
    block.begin = cursor;
    block.count = static_cast<std::uint64_t>(divisions.size()) * bits_per_division;
    block.divisions = &divisions;
    block.division_widths = profile;
    if (block.count > std::numeric_limits<std::uint64_t>::max() - cursor)
        throw std::overflow_error("canonical range division query cursor overflows u64");
    cursor += block.count;
    if (block.count != 0) blocks.push_back(block);
}

void append_canonical_range_attention_blocks(
    std::vector<CanonicalRangeBitBlock>& blocks, std::uint64_t& cursor,
    const AttentionTrace& trace, bool hidden, const StaticLookupBounds& bounds,
    const std::array<unsigned, 5>& profile) {
    append_canonical_range_tuple_block(blocks, cursor, trace.raw_score,
                                       trace.leaky_score, bounds.lrelu);
    append_canonical_range_tuple_block(blocks, cursor, trace.delta,
                                       trace.exp, bounds.exp);
    if (hidden)
        append_canonical_range_tuple_block(blocks, cursor, trace.recovered,
                                           trace.activated, bounds.elu);
    append_canonical_range_vector_block(
        blocks, cursor, trace.delta,
        nonnegative_bit_width(2 * static_cast<Wide>(bounds.lrelu.second)));
    append_canonical_range_division_block(blocks, cursor, trace.divisions,
                                          profile);
}

std::uint64_t canonical_range_bit_value(
    const std::vector<CanonicalRangeBitBlock>& blocks,
    std::uint64_t real_queries, std::uint64_t padded_queries,
    std::uint64_t padding_index, std::uint64_t row) {
    if (row >= padded_queries)
        throw std::out_of_range("canonical range Lasso query row is invalid");
    if (row >= real_queries) return padding_index;
    const auto found = std::upper_bound(
        blocks.begin(), blocks.end(), row,
        [](std::uint64_t index, const CanonicalRangeBitBlock& block) {
            return index < block.begin;
        });
    if (found == blocks.begin())
        throw std::out_of_range("canonical range Lasso query row has no source block");
    const CanonicalRangeBitBlock& block = *std::prev(found);
    const std::uint64_t local = row - block.begin;
    if (local >= block.count)
        throw std::out_of_range("canonical range Lasso query source row is invalid");
    Wide value = 0;
    unsigned bit = 0;
    unsigned width = 0;
    if (block.kind == CanonicalRangeBitBlock::Kind::Tuple) {
        const std::uint64_t tuple_bits = 2ULL * block.width_x + 2ULL * block.width_y;
        const std::uint64_t tuple = local / tuple_bits;
        const std::uint64_t within = local % tuple_bits;
        if (tuple >= block.first->size() || tuple >= block.second->size())
            throw std::out_of_range("canonical range tuple source row is invalid");
        const Wide x = block.first->at(static_cast<std::size_t>(tuple));
        const Wide y = block.second->at(static_cast<std::size_t>(tuple));
        const Wide x_plus = x + block.bound_x;
        const Wide x_minus = static_cast<Wide>(block.bound_x) - x;
        const Wide y_plus = y + block.bound_y;
        const Wide y_minus = static_cast<Wide>(block.bound_y) - y;
        if (within < block.width_x) { value = x_plus; bit = static_cast<unsigned>(within); width = block.width_x; }
        else if (within < 2ULL * block.width_x) { value = x_minus; bit = static_cast<unsigned>(within - block.width_x); width = block.width_x; }
        else if (within < 2ULL * block.width_x + block.width_y) { value = y_plus; bit = static_cast<unsigned>(within - 2ULL * block.width_x); width = block.width_y; }
        else { value = y_minus; bit = static_cast<unsigned>(within - 2ULL * block.width_x - block.width_y); width = block.width_y; }
    } else if (block.kind == CanonicalRangeBitBlock::Kind::Vector) {
        const std::uint64_t element = local / block.width_x;
        bit = static_cast<unsigned>(local % block.width_x);
        width = block.width_x;
        if (element >= block.first->size())
            throw std::out_of_range("canonical range scalar source row is invalid");
        value = block.first->at(static_cast<std::size_t>(element));
    } else {
        std::uint64_t bits_per_division = 0;
        for (unsigned width : block.division_widths) bits_per_division += width;
        const std::uint64_t division_index = local / bits_per_division;
        std::uint64_t within = local % bits_per_division;
        if (division_index >= block.divisions->size())
            throw std::out_of_range("canonical range division source row is invalid");
        const std::array<Int, 5> values = canonical_division_values(
            block.divisions->at(static_cast<std::size_t>(division_index)));
        std::size_t field = 0;
        while (field < block.division_widths.size() && within >= block.division_widths[field]) {
            within -= block.division_widths[field];
            ++field;
        }
        if (field == block.division_widths.size())
            throw std::out_of_range("canonical range division bit selector is invalid");
        value = values[field];
        bit = static_cast<unsigned>(within);
        width = block.division_widths[field];
    }
    if (value < 0 || value > static_cast<Wide>(std::numeric_limits<Int>::max()))
        throw std::invalid_argument("canonical range value is outside the nonnegative signed-63-bit domain");
    if (width == 0 || width > 63 || bit >= width ||
        (width < 63 && value >= (static_cast<Wide>(1) << width)))
        throw std::logic_error("canonical range query bit exceeds its authenticated width");
    return static_cast<std::uint64_t>((value >> bit) & 1);
}

struct LassoAdapterBundle {
    std::shared_ptr<std::vector<NativeLassoFamilyProver>> families =
        std::make_shared<std::vector<NativeLassoFamilyProver>>();
    NativeLassoQueryProver query;

    CommitmentWire com_L;
    double commitment_ms = 0.0;
    double query_blinding_prepare_ms = 0.0;
    double family_generator_derive_ms = 0.0;
};

std::string native_family_id(const std::string& family) {
    if (family == "function/lrelu") return "function_lrelu";
    if (family == "function/exp") return "function_exp";
    if (family == "function/elu") return "function_elu";
    if (family == "range/all") return "range_all";
    throw std::invalid_argument("unknown canonical Lasso family: " + family);
}

const std::vector<TablePoint>& table_points_for(const StaticTables& tables, const std::string& family) {
    if (family == "function/lrelu") return tables.lrelu;
    if (family == "function/exp") return tables.exp;
    if (family == "function/elu") return tables.elu;
    throw std::invalid_argument("function table requested for non-function family");
}

const std::pair<Int, Int>& table_bounds_for(const StaticLookupBounds& bounds, const std::string& family) {
    if (family == "function/lrelu") return bounds.lrelu;
    if (family == "function/exp") return bounds.exp;
    if (family == "function/elu") return bounds.elu;
    throw std::invalid_argument("function bounds requested for non-function family");
}

LassoAdapterBundle build_lasso_bundle(const GatCircuitPlan& plan, const ForwardResult& forward, const StaticTables& tables, const std::string& basis_label) {
    if (plan.input_layout.segments.size() != 4) throw std::invalid_argument("Lasso bundle requires canonical four-segment layout");
    const StaticLookupBounds lookup_bounds = static_lookup_bounds(tables);
    const InputSegment& l_segment = plan.input_layout.segments.at(3);
    LassoAdapterBundle result;
    for (const LassoFamilyDescriptor& descriptor : plan.lasso) {
        const std::string family_id = native_family_id(descriptor.family_id);
        NativeLassoFamilyProver family;
        family.family_id = family_id;
        family.table_id = "table_" + family_id;
        family.resource_id = "resource_" + family_id;
        family.registration_id = "registration_" + family_id;
        family.segment_offset = descriptor.query_offset - l_segment.offset;
        family.logical_queries = descriptor.real_queries;
        family.padded_queries = descriptor.padded_queries;
        if (family.padded_queries == 0 || family.segment_offset % family.padded_queries != 0)
            throw std::invalid_argument("Lasso family segment is not power-of-two aligned for masked opening");
        if (family.segment_offset > l_segment.logical_length ||
            family.padded_queries > l_segment.logical_length - family.segment_offset)
            throw std::invalid_argument("Lasso family exceeds canonical L_qry logical domain");

        if (descriptor.family_id == "range/all") {
            family.table = {Fr(0), Fr(1)};
            family.padding_index = 0;
            family.padding_value = Fr(0);
            std::vector<CanonicalRangeBitBlock> blocks;
            std::uint64_t cursor = 0;
            std::size_t profile_index = 0;
            for (const HiddenLayerTrace& layer : forward.trace.hidden_layers)
                for (const AttentionTrace& head : layer.heads)
                    append_canonical_range_attention_blocks(
                        blocks, cursor, head, true, lookup_bounds,
                        plan.division_range_bits.at(profile_index++));
            for (const OutputHeadTrace& head : forward.trace.output_heads) {
                append_canonical_range_attention_blocks(
                    blocks, cursor, head.attention, false, lookup_bounds,
                    plan.division_range_bits.at(profile_index++));
                append_canonical_range_division_block(
                    blocks, cursor, head.alignment_divisions,
                    plan.division_range_bits.at(profile_index++));
            }
            append_canonical_range_division_block(
                blocks, cursor, forward.trace.final_average_divisions,
                plan.division_range_bits.at(profile_index++));
            if (profile_index != plan.division_range_bits.size() ||
                cursor != descriptor.real_queries)
                throw std::invalid_argument("canonical range Lasso query count does not match plan");
            auto range_blocks = std::make_shared<std::vector<CanonicalRangeBitBlock>>(std::move(blocks));
            const std::uint64_t real_queries = descriptor.real_queries;
            const std::uint64_t padded_queries = family.padded_queries;
            const std::uint64_t padding_index = family.padding_index;
            family.selected_table_row_at = [range_blocks, real_queries,
                                            padded_queries, padding_index](std::uint64_t row) {
                return canonical_range_bit_value(*range_blocks, real_queries,
                                                 padded_queries, padding_index, row);
            };
        } else {
            const std::vector<TablePoint>& points = table_points_for(tables, descriptor.family_id);
            const auto& bounds = table_bounds_for(lookup_bounds, descriptor.family_id);
            family.table.reserve(descriptor.table_domain);
            for (const TablePoint& point : points) family.table.push_back(packed_table_value(point.x, point.y, bounds.first, bounds.second));
            if (family.table.empty()) throw std::invalid_argument("empty static Lasso table");
            while (family.table.size() < descriptor.table_domain) family.table.push_back(family.table.front());
            family.padding_index = 0;
            family.padding_value = family.table[0];
            auto table_indices = std::make_shared<std::map<std::pair<Int, Int>, std::uint64_t>>();
            for (std::uint64_t index = 0; index < points.size(); ++index)
                table_indices->emplace(std::make_pair(points[static_cast<std::size_t>(index)].x,
                                                      points[static_cast<std::size_t>(index)].y),
                                       index);
            std::vector<CanonicalLassoPairBlock> query_blocks;
            std::uint64_t query_cursor = 0;
            const auto append_attention_queries = [&](const AttentionTrace& trace,
                                                       bool hidden) {
                const std::vector<Int>* x = nullptr;
                const std::vector<Int>* y = nullptr;
                if (descriptor.family_id == "function/lrelu") { x = &trace.raw_score; y = &trace.leaky_score; }
                else if (descriptor.family_id == "function/exp") { x = &trace.delta; y = &trace.exp; }
                else if (descriptor.family_id == "function/elu" && hidden) { x = &trace.recovered; y = &trace.activated; }
                if (x == nullptr) return;
                if (x->size() != y->size() || x->size() > std::numeric_limits<std::uint64_t>::max() - query_cursor)
                    throw std::overflow_error("canonical function Lasso query source extent overflows");
                if (!x->empty()) query_blocks.push_back({query_cursor, static_cast<std::uint64_t>(x->size()), x, y});
                query_cursor += static_cast<std::uint64_t>(x->size());
            };
            for (const HiddenLayerTrace& layer : forward.trace.hidden_layers)
                for (const AttentionTrace& head : layer.heads)
                    append_attention_queries(head, true);
            if (descriptor.family_id != "function/elu")
                for (const OutputHeadTrace& head : forward.trace.output_heads)
                    append_attention_queries(head.attention, false);
            if (query_cursor != descriptor.real_queries)
                throw std::invalid_argument("canonical function Lasso query count does not match plan");
            const std::uint64_t real_queries = descriptor.real_queries;
            const std::uint64_t padded_queries = family.padded_queries;
            const std::uint64_t padding_index = family.padding_index;
            family.selected_table_row_at = [query_blocks = std::move(query_blocks),
                                            table_indices, real_queries,
                                            padded_queries, padding_index](std::uint64_t row) {
                return canonical_lasso_pair_row(query_blocks, *table_indices,
                                                real_queries, padded_queries,
                                                padding_index, row);
            };
        }
        if (!family.selected_table_rows.empty() || !family.queries.empty() ||
            !family.legacy_clear_matrix.empty() ||
            !family.legacy_clear_matrix_generators.empty() ||
            !family.selected_table_row_at)
            throw std::invalid_argument("canonical Lasso formal selected-row provider retained a private backing copy or lacks its lazy callback");
        family.matrix_layout = native_lasso_balanced_matrix_layout(
            family.padded_queries, family.table.size());
        family.matrix_layout.trusted_generator_id =
            basis_label + "/lasso/" + family.family_id + "/matrix-column-v2";
        const auto generator_derive_begin = std::chrono::steady_clock::now();
        family.matrix_layout.column_generators.resize(
            static_cast<std::size_t>(family.matrix_layout.columns));
        for (std::uint64_t column = 0; column < family.matrix_layout.columns; ++column)
            hashAndMapToG1(
                family.matrix_layout.column_generators[static_cast<std::size_t>(column)],
                family.matrix_layout.trusted_generator_id + "/" +
                    std::to_string(column));
        result.family_generator_derive_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - generator_derive_begin).count();
        result.families->push_back(std::move(family));
    }
    std::sort(result.families->begin(), result.families->end(),
              [](const NativeLassoFamilyProver& left, const NativeLassoFamilyProver& right) { return left.family_id < right.family_id; });
    const std::string query_basis_id = fixed_digest(hash_hex(basis_label));
    std::vector<std::pair<std::uint64_t, std::uint64_t>> family_spans;
    family_spans.reserve(result.families->size());
    for (const NativeLassoFamilyProver& family : *result.families)
        family_spans.emplace_back(family.segment_offset, family.padded_queries);
    const std::shared_ptr<std::vector<NativeLassoFamilyProver>> family_storage =
        result.families;
    result.query.value_provider.size = l_segment.logical_length;
    result.query.value_provider.at =
        [family_storage, family_spans,
         domain = static_cast<std::uint64_t>(l_segment.logical_length)](
            std::uint64_t index) {
            if (index >= domain)
                throw std::out_of_range("canonical L_qry scalar coordinate is invalid");
            for (std::size_t family_index = 0;
                 family_index < family_storage->size(); ++family_index) {
                const std::uint64_t begin = family_spans[family_index].first;
                const std::uint64_t count = family_spans[family_index].second;
                if (index < begin || index - begin >= count) continue;
                const NativeLassoFamilyProver& family =
                    family_storage->at(family_index);
                const std::uint64_t row = index - begin;
                const std::uint64_t selected = family.selected_table_row_at
                    ? family.selected_table_row_at(row)
                    : family.selected_table_rows.at(static_cast<std::size_t>(row));
                if (selected >= family.table.size())
                    throw std::out_of_range("canonical L_qry selected table row is invalid");
                return family.table[static_cast<std::size_t>(selected)];
            }
            return Fr(0);
        };
    result.query.basis_id = query_basis_id;
    result.query.tensor_view = canonical_input_tensor_view(
        basis_label, query_basis_id, "canonical/L_qry",
        plan.input_layout.n_in, l_segment.logical_length);
    if (result.query.tensor_view.rows > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("canonical L_qry row blinding extent exceeds size_t");
    const auto query_blinding_prepare_begin = std::chrono::steady_clock::now();
    result.query.tensor_commitment_blindings.resize(
        static_cast<std::size_t>(result.query.tensor_view.rows));
    for (Fr& blinding : result.query.tensor_commitment_blindings)
        blinding.setByCSPRNG();
    result.query_blinding_prepare_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - query_blinding_prepare_begin).count();
    result.query.value_provider.size = l_segment.logical_length;
    const hyrax::HyraxScalarProvider query_values = result.query.value_provider;
    const auto commitment_begin = std::chrono::steady_clock::now();
    std::string tensor_reason;
    if (!hyrax::commit_hyrax_tensor_mle_with_blindings(
            result.query.tensor_view, query_values,
            result.query.tensor_commitment, result.query.tensor_state,
            std::move(result.query.tensor_commitment_blindings),
            &tensor_reason))
        throw std::runtime_error("canonical L_qry tensor commitment failed: " +
                                 tensor_reason);
    result.commitment_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - commitment_begin).count();
    result.com_L.scheme = hyrax::kHyraxTensorMleSchema;
    result.com_L.basis_id = result.query.tensor_view.basis_id;
    result.com_L.logical_size = l_segment.logical_length;
    result.com_L.padded_size = plan.input_layout.n_in;
    result.com_L.point_hex.clear();
    result.com_L.has_tensor_commitment = true;
    result.com_L.tensor_commitment = result.query.tensor_commitment;
    return result;
}

std::vector<NativeLassoChunkFamilyProver> build_lasso_chunk_families(const GatCircuitPlan& plan, const ForwardResult& forward, const StaticTables& tables) {
    std::map<std::string, std::vector<std::pair<Int, Int>>> function_queries;
    std::map<std::string, std::vector<Fr>> range_queries;
    const StaticLookupBounds lookup_bounds = static_lookup_bounds(tables);
    range_queries["range/all"] = canonical_range_query_values(plan, forward, lookup_bounds);
    const auto append_attention = [&](const AttentionTrace& trace, bool hidden) {
        auto& lrelu = function_queries["function/lrelu"];
        auto& exp = function_queries["function/exp"];
        for (std::size_t index = 0; index < trace.raw_score.size(); ++index) lrelu.emplace_back(trace.raw_score[index], trace.leaky_score[index]);
        for (std::size_t index = 0; index < trace.delta.size(); ++index) exp.emplace_back(trace.delta[index], trace.exp[index]);
        if (hidden) {
            auto& elu = function_queries["function/elu"];
            for (std::size_t index = 0; index < trace.recovered.size(); ++index) elu.emplace_back(trace.recovered[index], trace.activated[index]);
        }
    };
    for (const HiddenLayerTrace& layer : forward.trace.hidden_layers)
        for (const AttentionTrace& head : layer.heads) append_attention(head, true);
    for (const OutputHeadTrace& head : forward.trace.output_heads) {
        append_attention(head.attention, false);
    }

    const auto table_for = [&](const std::string& family) -> const std::vector<TablePoint>& {
        if (family == "function/lrelu") return tables.lrelu;
        if (family == "function/exp") return tables.exp;
        if (family == "function/elu") return tables.elu;
        throw std::invalid_argument("unknown virtual Lasso table family");
    };
    std::vector<NativeLassoChunkFamilyProver> result;
    result.reserve(plan.lasso.size());
    for (const LassoFamilyDescriptor& descriptor : plan.lasso) {
        NativeLassoChunkFamilyProver family;
        family.family_id = native_family_id(descriptor.family_id);
        family.table_id = descriptor.table_id;
        family.resource_id = "resource_" + family.family_id;
        family.registration_id = "registration_" + family.family_id;
        family.segment_offset = descriptor.query_offset - plan.input_layout.segments.at(3).offset;
        family.logical_queries = descriptor.real_queries;
        family.padded_queries = descriptor.padded_queries;
        family.padding_index = 0;
        family.chunk_size = std::min<std::uint64_t>(kNativeLassoChunkEntries, static_cast<std::uint64_t>(std::max<std::size_t>(
                                                                                  2, next_power_of_two(std::max<std::size_t>(2, descriptor.padded_queries)))));
        if (family.padded_queries >= kNativeLassoChunkEntries) family.chunk_size = kNativeLassoChunkEntries;
        if (family.padded_queries % family.chunk_size != 0) family.chunk_size = family.padded_queries;
        std::vector<Fr> table_values;
        if (descriptor.family_id == "range/all") {
            table_values = {F(0), F(1)};
            family.padding_value = F(0);
        } else {
            const auto& points = table_for(descriptor.family_id);
            const auto& bounds = table_bounds_for(lookup_bounds, descriptor.family_id);
            table_values.reserve(descriptor.table_domain);
            for (const TablePoint& point : points) table_values.push_back(packed_table_value(point.x, point.y, bounds.first, bounds.second));
            while (table_values.size() < descriptor.table_domain) table_values.push_back(table_values.front());
            family.padding_value = table_values.front();
        }
        family.table = table_values;
        std::vector<Fr> query_values;
        if (descriptor.family_id == "range/all") {
            query_values = range_queries[descriptor.family_id];
        } else {
            const auto& pairs = function_queries[descriptor.family_id];
            const auto& bounds = table_bounds_for(lookup_bounds, descriptor.family_id);
            for (const auto& pair : pairs) query_values.push_back(packed_table_value(pair.first, pair.second, bounds.first, bounds.second));
        }
        if (query_values.size() != family.logical_queries) throw std::invalid_argument("virtual Lasso query count mismatch");
        auto values = std::make_shared<std::vector<Fr>>(std::move(query_values));
        family.query_value = [values](std::uint64_t index) { return index < values->size() ? (*values)[static_cast<std::size_t>(index)] : F(0); };
        result.push_back(std::move(family));
    }
    std::sort(result.begin(), result.end(),
              [](const NativeLassoChunkFamilyProver& left, const NativeLassoChunkFamilyProver& right) { return left.family_id < right.family_id; });
    const auto legacy_public = [](const NativeLassoChunkFamilyPublic& family) {
        NativeLassoFamilyPublic legacy;
        legacy.family_id = family.family_id;
        legacy.table_id = family.table_id;
        legacy.resource_id = family.resource_id;
        legacy.registration_id = family.registration_id;
        legacy.table = family.table;
        legacy.segment_offset = family.segment_offset;
        legacy.logical_queries = family.logical_queries;
        legacy.padded_queries = family.padded_queries;
        legacy.padding_index = family.padding_index;
        legacy.padding_value = family.padding_value;
        return legacy;
    };
    std::vector<NativeLassoFamilyPublic> public_families;
    public_families.reserve(result.size());
    for (const NativeLassoChunkFamilyPublic& family : result) public_families.push_back(legacy_public(family));
    const std::string tables_digest = native_lasso_tables_digest(public_families);
    const std::string resources_digest = native_lasso_resources_digest(public_families);
    for (NativeLassoChunkFamilyProver& family : result) {
        family.statement_tables_digest = tables_digest;
        family.statement_resources_digest = resources_digest;
    }
    return result;
}

std::vector<NativeLassoChunkFamilyPublic> build_lasso_chunk_public_families(const GatCircuitPlan& plan, const StaticTables& tables) {
    std::vector<NativeLassoChunkFamilyPublic> result;
    const StaticLookupBounds lookup_bounds = static_lookup_bounds(tables);
    result.reserve(plan.lasso.size());
    for (const LassoFamilyDescriptor& descriptor : plan.lasso) {
        NativeLassoChunkFamilyPublic family;
        family.family_id = native_family_id(descriptor.family_id);
        family.table_id = descriptor.table_id;
        family.resource_id = "resource_" + family.family_id;
        family.registration_id = "registration_" + family.family_id;
        family.segment_offset = descriptor.query_offset - plan.input_layout.segments.at(3).offset;
        family.logical_queries = descriptor.real_queries;
        family.padded_queries = descriptor.padded_queries;
        family.padding_index = 0;
        family.chunk_size = family.padded_queries >= kNativeLassoChunkEntries ? kNativeLassoChunkEntries : family.padded_queries;
        if (family.chunk_size < 2) family.chunk_size = 2;
        if (descriptor.family_id == "range/all") {
            family.table = {F(0), F(1)};
            family.padding_value = F(0);
        } else {
            const std::vector<TablePoint>* points = nullptr;
            if (descriptor.family_id == "function/lrelu")
                points = &tables.lrelu;
            else if (descriptor.family_id == "function/exp")
                points = &tables.exp;
            else if (descriptor.family_id == "function/elu")
                points = &tables.elu;
            if (points == nullptr || points->empty()) throw std::invalid_argument("public virtual Lasso table is empty");
            const auto& bounds = table_bounds_for(lookup_bounds, descriptor.family_id);
            for (const TablePoint& point : *points) family.table.push_back(packed_table_value(point.x, point.y, bounds.first, bounds.second));
            while (family.table.size() < descriptor.table_domain) family.table.push_back(family.table.front());
            family.padding_value = family.table.front();
        }
        result.push_back(std::move(family));
    }
    std::sort(result.begin(), result.end(),
              [](const NativeLassoChunkFamilyPublic& left, const NativeLassoChunkFamilyPublic& right) { return left.family_id < right.family_id; });
    return result;
}

std::vector<NativeLassoFamilyPublic> build_lasso_public_families(const GatCircuitPlan& plan, const StaticTables& tables, const std::string& basis_label,
                                                                 const std::vector<NativeLassoFamilyProver>* cached_families = nullptr) {
    if (plan.input_layout.segments.size() != 4) throw std::invalid_argument("public Lasso registry requires four segments");
    const StaticLookupBounds lookup_bounds = static_lookup_bounds(tables);
    const InputSegment& l_segment = plan.input_layout.segments.at(3);
    std::vector<NativeLassoFamilyPublic> result;
    result.reserve(plan.lasso.size());
    for (const LassoFamilyDescriptor& descriptor : plan.lasso) {
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
            const std::vector<TablePoint>& points = table_points_for(tables, descriptor.family_id);
            const auto& bounds = table_bounds_for(lookup_bounds, descriptor.family_id);
            family.table.reserve(descriptor.table_domain);
            for (const TablePoint& point : points) family.table.push_back(packed_table_value(point.x, point.y, bounds.first, bounds.second));
            if (family.table.empty()) throw std::invalid_argument("empty public Lasso table");
            while (family.table.size() < descriptor.table_domain) family.table.push_back(family.table.front());
            family.padding_value = family.table.front();
        }
        if (family.segment_offset + family.padded_queries > l_segment.padded_length || family.table.empty() ||
            (family.padded_queries & (family.padded_queries - 1)) != 0)
            throw std::invalid_argument("public Lasso family exceeds canonical domain");
        family.matrix_layout = native_lasso_balanced_matrix_layout(
            family.padded_queries, family.table.size());
        family.matrix_layout.trusted_generator_id =
            basis_label + "/lasso/" + family.family_id + "/matrix-column-v2";
        family.matrix_layout.column_generators.resize(
            static_cast<std::size_t>(family.matrix_layout.columns));
        for (std::uint64_t column = 0; column < family.matrix_layout.columns; ++column)
            hashAndMapToG1(
                family.matrix_layout.column_generators[static_cast<std::size_t>(column)],
                family.matrix_layout.trusted_generator_id + "/" +
                    std::to_string(column));
        if (cached_families == nullptr) {
            // The verifier reconstructs the same trusted column basis from this
            // canonical descriptor; no per-coordinate matrix basis is carried.
        } else {
            const NativeLassoFamilyProver* cached = nullptr;
            for (const NativeLassoFamilyProver& candidate : *cached_families) {
                if (candidate.family_id == family.family_id) {
                    cached = &candidate;
                    break;
                }
            }
            if (cached == nullptr || cached->table != family.table || cached->table_id != family.table_id || cached->resource_id != family.resource_id ||
                cached->registration_id != family.registration_id || cached->segment_offset != family.segment_offset ||
                cached->logical_queries != family.logical_queries || cached->padded_queries != family.padded_queries ||
                cached->padding_index != family.padding_index || cached->padding_value != family.padding_value ||
                native_lasso_matrix_layout_digest(cached->matrix_layout) !=
                    native_lasso_matrix_layout_digest(family.matrix_layout)) {
                throw std::invalid_argument("cached Lasso matrix-generator domain or identity mismatch");
            }
            family.matrix_layout = cached->matrix_layout;
        }
        result.push_back(std::move(family));
    }
    std::sort(result.begin(), result.end(),
              [](const NativeLassoFamilyPublic& left, const NativeLassoFamilyPublic& right) { return left.family_id < right.family_id; });
    const std::string tables_digest = native_lasso_tables_digest(result);
    const std::string resources_digest = native_lasso_resources_digest(result);
    for (NativeLassoFamilyPublic& family : result) {
        family.statement_tables_digest = tables_digest;
        family.statement_resources_digest = resources_digest;
    }
    return result;
}

NativeLassoQueryPublic build_lasso_public_query(const GatCircuitPlan& plan, const CommitmentWire& com_L, const std::string& basis_label,
                                                const NativeLassoQueryProver* cached_query = nullptr) {
    const InputSegment& l_segment = plan.input_layout.segments.at(3);
    if (!com_L.valid() || !com_L.has_tensor_commitment ||
        com_L.padded_size != plan.input_layout.n_in ||
        com_L.logical_size != l_segment.logical_length)
        throw std::invalid_argument("public typed Lasso query commitment is invalid");
    NativeLassoQueryPublic query;
    query.tensor_view = canonical_input_tensor_view(
        basis_label, com_L.basis_id, "canonical/L_qry",
        plan.input_layout.n_in, l_segment.logical_length);
    query.tensor_commitment = com_L.tensor_commitment;
    if (cached_query != nullptr) {
        if (cached_query->tensor_view.full_domain != query.tensor_view.full_domain ||
            cached_query->tensor_view.logical_length != query.tensor_view.logical_length ||
            cached_query->tensor_view.active_domain != query.tensor_view.active_domain ||
            cached_query->tensor_view.rows != query.tensor_view.rows ||
            cached_query->tensor_view.columns != query.tensor_view.columns ||
            cached_query->tensor_view.role != query.tensor_view.role ||
            cached_query->tensor_view.basis_id != query.tensor_view.basis_id ||
            cached_query->tensor_view.column_generators.canonical_id !=
                query.tensor_view.column_generators.canonical_id ||
            native_lasso_tensor_query_digest(cached_query->tensor_view,
                                             cached_query->tensor_commitment) !=
                native_lasso_tensor_query_digest(query.tensor_view,
                                                 query.tensor_commitment))
            throw std::invalid_argument("cached Lasso typed tensor descriptor or commitment mismatch");
    }
    query.basis_id = com_L.basis_id;
    query.statement_l_digest = native_lasso_tensor_query_digest(
        query.tensor_view, query.tensor_commitment);
    return query;
}

NativeFieldVector input_segment(const GatCircuitPlan& plan, const NativeFieldVector& input, std::size_t segment_index) {
    if (segment_index >= plan.input_layout.segments.size()) throw std::invalid_argument("canonical input segment index is invalid");
    const InputSegment& segment = plan.input_layout.segments.at(segment_index);
    const std::size_t domain = plan.input_layout.n_in;
    if (segment.padded_length != domain || segment.offset + domain > input.size()) throw std::invalid_argument("canonical input segment domain is invalid");
    return NativeFieldVector(input.begin() + segment.offset, input.begin() + segment.offset + domain);
}

Fr adapter_mle_eval(const std::vector<Fr>& values, const std::vector<Fr>& point) {
    if (values.empty() || point.size() >= sizeof(std::size_t) * 8 || values.size() > (std::size_t(1) << point.size()))
        throw std::invalid_argument("adapter MLE dimensions are invalid");
    std::vector<Fr> folded(std::size_t(1) << point.size(), Fr(0));
    std::copy(values.begin(), values.end(), folded.begin());
    std::size_t active = folded.size();
    for (const Fr& coordinate : point) {
        const std::size_t half = active / 2;
        for (std::size_t index = 0; index < half; ++index) folded[index] = folded[2 * index] * (Fr(1) - coordinate) + folded[2 * index + 1] * coordinate;
        active = half;
    }
    return folded.front();
}

std::vector<Fr> public_feature_segment(const GatCircuitPlan& plan, const Graph& graph, const Tensor2D& features) {
    if (plan.input_layout.segments.empty() || features.values.size() > plan.input_layout.segments.front().logical_length)
        throw std::invalid_argument("public feature segment exceeds X_pub");
    std::vector<Fr> result(plan.input_layout.n_in, Fr(0));
    for (std::size_t index = 0; index < features.values.size(); ++index) result[index] = Fr(features.values[index]);
    (void)graph;
    return result;
}

std::vector<Fr> canonical_l_query_values(const GatCircuitPlan& plan, const ForwardResult& forward, const StaticTables& tables) {
    if (plan.input_layout.segments.size() != 4) throw std::invalid_argument("canonical L_qry requires four input segments");
    const StaticLookupBounds lookup_bounds = static_lookup_bounds(tables);
    const InputSegment& l_segment = plan.input_layout.segments.at(3);
    std::map<std::string, std::vector<std::pair<Int, Int>>> function_queries;
    std::map<std::string, std::vector<Fr>> range_queries;
    range_queries["range/all"] = canonical_range_query_values(plan, forward, lookup_bounds);
    const auto append_attention = [&](const AttentionTrace& trace, bool hidden) {
        auto& lrelu = function_queries["function/lrelu"];
        auto& exp = function_queries["function/exp"];
        for (std::size_t index = 0; index < trace.raw_score.size(); ++index) lrelu.emplace_back(trace.raw_score[index], trace.leaky_score[index]);
        for (std::size_t index = 0; index < trace.delta.size(); ++index) exp.emplace_back(trace.delta[index], trace.exp[index]);
        if (hidden) {
            auto& elu = function_queries["function/elu"];
            for (std::size_t index = 0; index < trace.recovered.size(); ++index) elu.emplace_back(trace.recovered[index], trace.activated[index]);
        }
    };
    for (const HiddenLayerTrace& layer : forward.trace.hidden_layers)
        for (const AttentionTrace& head : layer.heads) append_attention(head, true);
    for (const OutputHeadTrace& head : forward.trace.output_heads) {
        append_attention(head.attention, false);
    }

    std::vector<Fr> values(plan.input_layout.n_in, Fr(0));
    for (const LassoFamilyDescriptor& descriptor : plan.lasso) {
        const std::size_t family_offset = descriptor.query_offset - l_segment.offset;
        if (family_offset + descriptor.padded_queries > values.size()) throw std::invalid_argument("canonical L_qry family exceeds its segment");
        std::vector<Fr> real_queries;
        Fr padding = Fr(descriptor.padding_value);
        if (descriptor.family_id == "range/all") {
            const auto& queries = range_queries[descriptor.family_id];
            real_queries = queries;
            padding = Fr(0);
        } else {
            const auto& points = table_points_for(tables, descriptor.family_id);
            const auto& bounds = table_bounds_for(lookup_bounds, descriptor.family_id);
            padding = packed_table_value(points.front().x, points.front().y, bounds.first, bounds.second);
            for (const auto& pair : function_queries[descriptor.family_id]) real_queries.push_back(packed_table_value(pair.first, pair.second, bounds.first, bounds.second));
        }
        if (real_queries.size() != descriptor.real_queries) throw std::invalid_argument("canonical L_qry query count mismatch");
        for (std::size_t index = 0; index < descriptor.padded_queries; ++index) {
            values[family_offset + index] = index < real_queries.size() ? real_queries[index] : padding;
        }
    }
    return values;
}

std::shared_ptr<SegmentedInputOracle> canonical_input_oracle(
    const GatCircuitPlan& plan, const Graph& graph,
    const Tensor2D& public_features, const Model& model,
    const QuantConfig& quant, const ForwardResult& forward,
    const hyrax::HyraxScalarProvider& l_query_values, bool borrow_aux) {
    const std::size_t input_size = plan.circuit.circuit.front().size;
    const std::size_t n_in = plan.input_layout.n_in;
    if (input_size != checked_mul(n_in, 4, "canonical input witness")) throw std::invalid_argument("canonical input witness size does not match plan");
    if (public_features.rows != graph.num_nodes || public_features.cols != graph.num_features ||
        public_features.values.size() != public_features.rows * public_features.cols)
        throw std::invalid_argument("public feature shape does not match GAT witness");
    std::array<NativeFieldVector, 4> segments;
    segments[0].reserve(public_features.values.size());
    for (Int value : public_features.values) segments[0].emplace_back(value);
    const std::vector<Int> model_values = flatten_model(model, quant);
    segments[1].reserve(model_values.size());
    for (Int value : model_values) segments[1].emplace_back(value);
    std::array<SegmentedInputOracle::SegmentValueProvider, 4> providers{};
    if (borrow_aux) {
        const std::shared_ptr<const AuxValueProviderState> aux_values = make_borrowed_aux_value_provider(forward);
        providers[2] = [aux_values](std::uint64_t index) { return aux_values->value_at(index); };
    } else {
        segments[2] = flatten_aux_values(forward);
    }
    const InputSegment& aux_segment = plan.input_layout.segments.at(2);
    if ((borrow_aux ? trace_aux_size(&forward.trace) : segments[2].size()) != aux_segment.logical_length) throw std::invalid_argument("canonical Aux witness size does not match plan");
    const InputSegment& l_segment = plan.input_layout.segments.at(3);
    if (l_query_values.size != n_in || !l_query_values.at)
        throw std::invalid_argument("canonical L_qry input provider does not match the padded segment domain");
    providers[3] = [l_query_values](std::uint64_t index) {
        return l_query_values.at(index);
    };
    std::shared_ptr<SegmentedInputOracle> result = std::make_shared<SegmentedInputOracle>(
        plan.input_layout, std::move(segments), std::move(providers));
    for (std::size_t segment = 0; segment < plan.input_layout.segments.size(); ++segment) {
        if (result->layout().segments[segment].logical_length !=
            (segment == 0 ? public_features.values.size()
             : segment == 1 ? model_values.size()
             : segment == 2 ? aux_segment.logical_length
                            : l_segment.logical_length))
            throw std::invalid_argument("canonical input segment provider length does not match the plan");
    }
    return result;
}

zkgpt::NativeFieldView canonical_public_output_view(
    const GatCircuitPlan& plan, const Tensor2D& public_logits,
    const void* lifetime_token) {
    if (!plan.native_proof_backing_is_resident())
        throw std::logic_error("canonical public output view requires resident native proof backing");
    if (plan.public_output_coordinates != public_logits.values.size() ||
        plan.residual_targets.size() != plan.residual_coordinates)
        throw std::invalid_argument("canonical public output view shape does not match the plan");
    if (plan.public_output_coordinates >
        std::numeric_limits<std::size_t>::max() - plan.residual_coordinates)
        throw std::length_error("canonical public output view size overflows native size_t");
    const std::size_t logical_size =
        plan.public_output_coordinates + plan.residual_coordinates;
    return zkgpt::NativeFieldView::borrowed(
        logical_size, lifetime_token,
        [&plan, &public_logits, logical_size](std::size_t index) {
            if (index >= logical_size)
                throw std::out_of_range("canonical public output coordinate is invalid");
            if (index < plan.public_output_coordinates)
                return Fr(public_logits.values.at(index));
            return Fr(plan.residual_targets.at(index - plan.public_output_coordinates));
        });
}

bool evaluate_layered_values(
    const GatCircuitPlan& plan, std::uint64_t input_size,
    const std::function<Fr(std::uint64_t)>& input_at,
    std::vector<NativeFieldVector>& values, std::string* reason,
    const std::vector<NativeFieldVector>* expected_values = nullptr,
    const zkgpt::NativeFieldView* terminal_expected = nullptr) {
    const auto fail = [reason](const std::string& message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    try {
        if (!plan.native_proof_backing_is_resident())
            return fail("native GAT proof backing is not resident");
        if (plan.circuit.circuit.empty() || input_at == nullptr ||
            input_size != plan.circuit.circuit.front().size)
            return fail("layered witness input provider/domain is invalid");
        if (expected_values != nullptr && (expected_values->size() != plan.circuit.circuit.size() || !expected_values->front().empty()))
            return fail("GAT circuit witness layer count is invalid");
        if (terminal_expected != nullptr && !terminal_expected->valid())
            return fail("GAT terminal output view is not live");
        values.clear();
        if (expected_values == nullptr) { values.resize(plan.circuit.circuit.size()); values[0].clear(); }
        std::size_t next_matmul = 0;
        for (std::size_t layer_id = 1; layer_id < plan.circuit.circuit.size(); ++layer_id) {
            const layer& current = plan.circuit.circuit[layer_id];
            const bool streamed_terminal = terminal_expected != nullptr &&
                layer_id + 1 == plan.circuit.circuit.size();
            if (streamed_terminal && terminal_expected->size() != current.size)
                return fail("GAT terminal output view has the wrong logical size");
            if (streamed_terminal && expected_values != nullptr &&
                !expected_values->at(layer_id).empty())
                return fail("GAT terminal layer is both retained and streamed");
            // In validation mode every preceding retained witness layer has
            // already passed this same full gate evaluation. Read that checked
            // layer directly and keep only the current expected output buffer.
            const NativeFieldVector& previous_values = expected_values == nullptr
                ? values.at(layer_id - 1)
                : expected_values->at(layer_id - 1);
            NativeFieldVector output;
            if (!streamed_terminal) output.assign(current.size, Fr(0));
            if (current.ty == layerType::FCONN) {
                if (next_matmul >= plan.matmuls.size()) return fail("FCONN layer has no canonical MatMul descriptor");
                const MatMulDescriptor& descriptor = plan.matmuls[next_matmul++];
                if (current.native_relation_id != descriptor.relation_id || current.native_family_id != descriptor.family_id ||
                    current.native_rows != descriptor.rows || current.native_inner != descriptor.inner || current.native_cols != descriptor.cols ||
                    current.native_lhs_offset != descriptor.lhs_offset || current.native_rhs_offset != descriptor.rhs_offset ||
                    current.native_output_offset != descriptor.output_offset ||
                    current.size != fconn_output_domain(descriptor.rows, descriptor.cols) ||
                    (!streamed_terminal && current.size != output.size()))
                    return fail("FCONN layer metadata does not match canonical MatMul descriptor");
                const std::size_t lhs_count = checked_mul(descriptor.rows, descriptor.inner, "FCONN lhs evaluation");
                const std::size_t rhs_count = checked_mul(descriptor.inner, descriptor.cols, "FCONN rhs evaluation");
                if (descriptor.lhs_offset > input_size || lhs_count > input_size - descriptor.lhs_offset || descriptor.rhs_offset > input_size ||
                    rhs_count > input_size - descriptor.rhs_offset)
                    return fail("FCONN input span exceeds canonical input layer");
                for (std::size_t row = 0; row < descriptor.rows; ++row) {
                    for (std::size_t col = 0; col < descriptor.cols; ++col) {
                        Fr value = Fr(0);
                        for (std::size_t term = 0; term < descriptor.inner; ++term) {
                            const std::size_t lhs = descriptor.lhs_offset + row * descriptor.inner + term;
                            const std::size_t rhs = descriptor.rhs_offset + term * descriptor.cols + col;
                            value += input_at(lhs) * input_at(rhs);
                        }
                        const std::size_t output_index = fconn_padded_output_index(
                            descriptor.rows, descriptor.cols,
                            row * descriptor.cols + col);
                        if (streamed_terminal) {
                            if (terminal_expected->at(output_index) != value)
                                return fail("GAT terminal FCONN output mismatch at row " + std::to_string(output_index));
                        } else {
                            output[output_index] = value;
                        }
                    }
                }
                if (streamed_terminal) {
                    const std::size_t padded_rows = fconn_axis_domain(descriptor.rows, "rows");
                    const std::size_t padded_cols = fconn_axis_domain(descriptor.cols, "columns");
                    for (std::size_t row = 0; row < padded_rows; ++row) {
                        for (std::size_t col = 0; col < padded_cols; ++col) {
                            if (row < descriptor.rows && col < descriptor.cols) continue;
                            const std::size_t index = row * padded_cols + col;
                            if (terminal_expected->at(index) != Fr(0))
                                return fail("GAT terminal FCONN padding is not zero at coordinate " + std::to_string(index));
                        }
                    }
                    if (expected_values == nullptr) values[layer_id].clear();
                } else {
                    if (expected_values != nullptr && output != expected_values->at(layer_id)) return fail("GAT circuit witness layer does not match official gate evaluation");
                    if (expected_values == nullptr) values[layer_id] = std::move(output);
                }
                continue;
            }
            if (streamed_terminal) {
                std::uint64_t previous_g = 0;
                bool have_previous_g = false;
                for (const uniGate& gate : current.uni_gates) {
                    if (gate.g >= current.size || (have_previous_g && gate.g < previous_g))
                        return fail("GAT terminal unary gates are out of bounds or not ordered by output row");
                    previous_g = gate.g;
                    have_previous_g = true;
                }
                previous_g = 0;
                have_previous_g = false;
                for (const binGate& gate : current.bin_gates) {
                    if (gate.g >= current.size || (have_previous_g && gate.g < previous_g))
                        return fail("GAT terminal binary gates are out of bounds or not ordered by output row");
                    previous_g = gate.g;
                    have_previous_g = true;
                }
                std::size_t unary_index = 0;
                std::size_t binary_index = 0;
                for (std::size_t row = 0; row < current.size; ++row) {
                    Fr value = Fr(0);
                    while (unary_index < current.uni_gates.size() && current.uni_gates[unary_index].g == row) {
                        const uniGate& gate = current.uni_gates[unary_index++];
                        const Fr source = gate.lu == 0 ? input_at(current.ori_id_u.at(gate.u)) : previous_values.at(gate.u);
                        value += source * Fr(current.coefficient(gate));
                    }
                    while (binary_index < current.bin_gates.size() && current.bin_gates[binary_index].g == row) {
                        const binGate& gate = current.bin_gates[binary_index++];
                        const std::size_t lhs_layer = gate.getLayerIdU(static_cast<u8>(layer_id));
                        const std::size_t rhs_layer = gate.getLayerIdV(static_cast<u8>(layer_id));
                        const Fr lhs = lhs_layer == 0 ? input_at(current.ori_id_u.at(gate.u)) : previous_values.at(gate.u);
                        const Fr rhs = rhs_layer == 0 ? input_at(current.ori_id_v.at(gate.v)) : previous_values.at(gate.v);
                        value += lhs * rhs * Fr(current.coefficient(gate));
                    }
                    if (terminal_expected->at(row) != value)
                        return fail("GAT terminal gate output mismatch at row " + std::to_string(row));
                }
                if (unary_index != current.uni_gates.size() || binary_index != current.bin_gates.size())
                    return fail("GAT terminal gate stream did not consume every gate row");
                if (expected_values == nullptr) values[layer_id].clear();
                continue;
            }
            for (const uniGate& gate : current.uni_gates) {
                if (gate.g >= output.size()) return fail("unary gate output is out of bounds");
                const Fr source = gate.lu == 0 ? input_at(current.ori_id_u.at(gate.u)) : previous_values.at(gate.u);
                output[gate.g] += source * Fr(current.coefficient(gate));
            }
            for (const binGate& gate : current.bin_gates) {
                if (gate.g >= output.size()) return fail("binary gate output is out of bounds");
                const std::size_t lhs_layer = gate.getLayerIdU(static_cast<u8>(layer_id));
                const std::size_t rhs_layer = gate.getLayerIdV(static_cast<u8>(layer_id));
                const Fr lhs = lhs_layer == 0 ? input_at(current.ori_id_u.at(gate.u)) : previous_values.at(gate.u);
                const Fr rhs = rhs_layer == 0 ? input_at(current.ori_id_v.at(gate.v)) : previous_values.at(gate.v);
                output[gate.g] += lhs * rhs * Fr(current.coefficient(gate));
            }
            if (expected_values != nullptr && output != expected_values->at(layer_id)) return fail("GAT circuit witness layer does not match official gate evaluation");
            if (expected_values == nullptr) values[layer_id] = std::move(output);
        }
        if (next_matmul != plan.matmuls.size()) return fail("canonical MatMul descriptor has no FCONN layer");
        return true;
    } catch (const std::exception& error) { return fail(error.what()); }
}

bool evaluate_layered_values(const GatCircuitPlan& plan,
                             const NativeFieldVector& input,
                             std::vector<NativeFieldVector>& values,
                             std::string* reason,
                             const std::vector<NativeFieldVector>* expected_values = nullptr,
                             const zkgpt::NativeFieldView* terminal_expected = nullptr) {
    return evaluate_layered_values(
        plan, static_cast<std::uint64_t>(input.size()),
        [&input](std::uint64_t index) {
            if (index >= input.size())
                throw std::out_of_range("layered witness input coordinate is invalid");
            return input[static_cast<std::size_t>(index)];
        },
        values, reason, expected_values, terminal_expected);
}

std::string context_encoding(const VerifierContext& context) {
    std::string encoded;
    append_string(encoded, kVerifierContextSchema);
    append_string(encoded, context.relation_id);
    append_string(encoded, context.backend);
    append_string(encoded, context.model_arch_id);
    append_string(encoded, context.privacy_extension_profile_id);
    append_string(encoded, context.quant_cfg_id);
    append_string(encoded, context.static_tables_id);
    append_string(encoded, context.full_dataset_identity);
    append_string(encoded, context.checkpoint_identity);
    append_string(encoded, context.cache_identity);
    append_registration_encoding(encoded, context.registration);
    append_string(encoded, commitment_encoding(context.com_W));

    append_string(encoded, plan_encoding(context.plan_ref()));
    append_string(encoded, "public-graph-v3");
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
    append_string(encoded, "public-features-v3");
    append_u64(encoded, context.public_features.rows);
    append_u64(encoded, context.public_features.cols);
    append_u64(encoded, static_cast<std::uint64_t>(context.public_features.values.size()));
    for (Int value : context.public_features.values) append_int(encoded, value);
    append_string(encoded, "public-logits-v3");
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
    const auto append_table = [&encoded](const std::vector<TablePoint>& table) {
        append_u64(encoded, static_cast<std::uint64_t>(table.size()));
        for (const TablePoint& point : table) {
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
    const auto append_vector = [&encoded](const std::vector<Int>& values) {
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

void write_u32(std::ostream& out, std::uint32_t value) {
    char bytes[4];
    for (unsigned i = 0; i < 4; ++i) bytes[i] = static_cast<char>((value >> (i * 8)) & 0xffU);
    out.write(bytes, sizeof(bytes));
}

void write_u64(std::ostream& out, std::uint64_t value) {
    char bytes[8];
    for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<char>((value >> (i * 8)) & 0xffU);
    out.write(bytes, sizeof(bytes));
}

void write_string(std::ostream& out, const std::string& value) {
    write_u64(out, static_cast<std::uint64_t>(value.size()));
    if (!value.empty()) out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

bool read_exact(std::istream& in, char* buffer, std::size_t size) {
    if (size == 0) return true;
    in.read(buffer, static_cast<std::streamsize>(size));
    return in.good() && static_cast<std::size_t>(in.gcount()) == size;
}

void write_u8(std::ostream& out, std::uint8_t value) {
    const char byte = static_cast<char>(value);
    out.write(&byte, 1);
}

bool read_u32(std::istream& in, std::uint32_t& value) {
    char bytes[4];
    if (!read_exact(in, bytes, sizeof(bytes))) return false;
    value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[i])) << (i * 8);
    return true;
}

bool read_u64(std::istream& in, std::uint64_t& value) {
    char bytes[8];
    if (!read_exact(in, bytes, sizeof(bytes))) return false;
    value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i])) << (i * 8);
    return true;
}

bool read_u8(std::istream& in, std::uint8_t& value) {
    char byte = 0;
    if (!read_exact(in, &byte, 1)) return false;
    value = static_cast<std::uint8_t>(static_cast<unsigned char>(byte));
    return true;
}

bool read_string(std::istream& in, std::string& value) {
    std::uint64_t size = 0;
    if (!read_u64(in, size) || size > kArtifactMaxString) return false;
    value.assign(static_cast<std::size_t>(size), '\0');
    return read_exact(in, value.empty() ? nullptr : &value[0], value.size());
}

void write_native_statement(std::ostream& out, const NativeProofStatement& statement) {
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
    write_string(out, statement.relation_schema);
    write_string(out, statement.relation_digest);
    write_u64(out, static_cast<std::uint64_t>(statement.relation_order.size()));
    for (const std::string& relation : statement.relation_order) write_string(out, relation);
}

bool read_native_statement(std::istream& in, NativeProofStatement& statement) {
    std::uint64_t relation_count = 0;
    if (!read_string(in, statement.verifier_context_digest) || !read_string(in, statement.field_id) || !read_string(in, statement.plan_digest) ||
        !read_string(in, statement.resources_digest) || !read_string(in, statement.tables_digest) || !read_string(in, statement.com_w_digest) ||
        !read_string(in, statement.aux_digest) || !read_string(in, statement.l_digest) || !read_string(in, statement.basis_digest) ||
        !read_u64(in, statement.l_dimension) || !read_string(in, statement.statement_root_digest) || !read_string(in, statement.relation_schema) ||
        !read_string(in, statement.relation_digest) || !read_u64(in, relation_count) || relation_count > kArtifactMaxMessages)
        return false;
    statement.relation_order.clear();
    statement.relation_order.reserve(static_cast<std::size_t>(relation_count));
    for (std::uint64_t index = 0; index < relation_count; ++index) {
        std::string relation;
        if (!read_string(in, relation)) return false;
        statement.relation_order.push_back(std::move(relation));
    }
    return true;
}

void write_commitment(std::ostream& out, const CommitmentWire& commitment) {
    write_string(out, commitment.scheme);
    write_string(out, commitment.basis_id);
    write_u64(out, commitment.logical_size);
    write_u64(out, commitment.padded_size);
    write_string(out, commitment.point_hex);
    write_u8(out, commitment.has_tensor_commitment ? 1 : 0);
    if (!commitment.has_tensor_commitment) return;

    const hyrax::HyraxTensorMleCommitment& tensor = commitment.tensor_commitment;
    write_string(out, tensor.schema);
    write_string(out, tensor.method);
    write_u32(out, tensor.version);
    write_string(out, tensor.role);
    write_string(out, tensor.basis_id);
    write_u8(out, static_cast<std::uint8_t>(tensor.column_generator_scheme));
    write_string(out, tensor.column_generator_id);
    write_u64(out, tensor.full_domain);
    write_u64(out, tensor.logical_length);
    write_u64(out, tensor.active_domain);
    write_u64(out, tensor.rows);
    write_u64(out, tensor.columns);
    write_string(out, tensor.root_digest);
    write_u64(out, tensor.row_commitments.size());
    for (const G1& row_commitment : tensor.row_commitments)
        write_string(out, row_commitment.serializeToHexStr());
}

bool read_commitment(std::istream& in, CommitmentWire& commitment) {
    std::uint64_t logical_size = 0;
    std::uint64_t padded_size = 0;
    std::uint8_t has_tensor = 0;
    if (!read_string(in, commitment.scheme) || !read_string(in, commitment.basis_id) || !read_u64(in, logical_size) || !read_u64(in, padded_size) ||
        !read_string(in, commitment.point_hex) || !read_u8(in, has_tensor) || has_tensor > 1)
        return false;
    if (logical_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        padded_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        return false;
    commitment.logical_size = logical_size;
    commitment.padded_size = padded_size;
    commitment.has_tensor_commitment = has_tensor != 0;
    if (!commitment.has_tensor_commitment) return true;

    hyrax::HyraxTensorMleCommitment& tensor = commitment.tensor_commitment;
    std::uint32_t version = 0;
    std::uint8_t generator_scheme = 0;
    std::uint64_t row_count = 0;
    if (!read_string(in, tensor.schema) || !read_string(in, tensor.method) ||
        !read_u32(in, version) || !read_string(in, tensor.role) ||
        !read_string(in, tensor.basis_id) || !read_u8(in, generator_scheme) ||
        generator_scheme > static_cast<std::uint8_t>(
                               hyrax::HyraxGeneratorProviderScheme::CanonicalAbsoluteHashToCurveV1) ||
        !read_string(in, tensor.column_generator_id) ||
        !read_u64(in, tensor.full_domain) ||
        !read_u64(in, tensor.logical_length) ||
        !read_u64(in, tensor.active_domain) || !read_u64(in, tensor.rows) ||
        !read_u64(in, tensor.columns) || !read_string(in, tensor.root_digest) ||
        !read_u64(in, row_count) || row_count > kArtifactMaxMessages ||
        row_count > static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max()))
        return false;
    tensor.version = version;
    tensor.column_generator_scheme =
        static_cast<hyrax::HyraxGeneratorProviderScheme>(generator_scheme);
    tensor.row_commitments.clear();
    tensor.row_commitments.reserve(static_cast<std::size_t>(row_count));
    for (std::uint64_t index = 0; index < row_count; ++index) {
        std::string point_hex;
        if (!read_string(in, point_hex)) return false;
        try {
            G1 point;
            point.deserializeHexStr(point_hex);
            if (point.serializeToHexStr() != point_hex) return false;
            tensor.row_commitments.push_back(point);
        } catch (...) {
            return false;
        }
    }
    return true;
}

void write_section(std::ostream& out, const ProofSection& section) {
    write_string(out, section.protocol);
    write_u64(out, static_cast<std::uint64_t>(section.messages.size()));
    for (const TranscriptMessage& message : section.messages) {
        write_string(out, message.label);
        write_string(out, message.payload);
        write_u8(out, message.kind);
        write_u64(out, message.sequence);
    }
}

bool read_section(std::istream& in, ProofSection& section) {
    std::uint64_t count = 0;
    if (!read_string(in, section.protocol) || !read_u64(in, count) || count > kArtifactMaxMessages) return false;
    section.messages.clear();
    section.messages.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        TranscriptMessage message;
        std::uint8_t kind = 0;
        if (!read_string(in, message.label) || !read_string(in, message.payload) || !read_u8(in, kind) || !read_u64(in, message.sequence) ||
            kind < static_cast<std::uint8_t>(NativeProofMessageKind::Data) || kind > static_cast<std::uint8_t>(NativeProofMessageKind::Challenge))
            return false;
        message.kind = kind;
        section.messages.push_back(std::move(message));
    }
    return true;
}

void write_chunked_mle_wire(std::ostream& out, const hyrax::ChunkedMleWire& wire) {
    write_string(out, wire.schema);
    write_u32(out, wire.version);
    write_string(out, wire.label);
    write_string(out, wire.basis_id);
    write_string(out, wire.scalar_domain);
    write_u64(out, wire.logical_length);
    write_u64(out, wire.padded_length);
    write_u64(out, wire.chunk_size);
    write_string(out, wire.point_digest);
    write_string(out, wire.root_digest);
    write_string(out, wire.aggregate_commitment_hex);
    write_u64(out, static_cast<std::uint64_t>(wire.chunks.size()));
    for (const hyrax::ChunkedMleChunkWire& chunk : wire.chunks) {
        write_u64(out, chunk.offset);
        write_u64(out, chunk.actual_length);
        write_u64(out, chunk.padded_length);
        write_string(out, chunk.label);
        write_string(out, chunk.basis_id);
        write_string(out, chunk.commitment_hex);
        write_string(out, chunk.evaluation_mask_commitment_hex);
        write_string(out, chunk.masked_evaluation);
        write_string(out, chunk.raw_evaluation_commitment_hex);
        write_string(out, chunk.masked_evaluation_commitment_hex);
    }
}

bool read_chunked_mle_wire(std::istream& in, hyrax::ChunkedMleWire& wire) {
    if (!read_string(in, wire.schema) || !read_u32(in, wire.version) || !read_string(in, wire.label) || !read_string(in, wire.basis_id) ||
        !read_string(in, wire.scalar_domain) || !read_u64(in, wire.logical_length) || !read_u64(in, wire.padded_length) || !read_u64(in, wire.chunk_size) ||
        !read_string(in, wire.point_digest) || !read_string(in, wire.root_digest) || !read_string(in, wire.aggregate_commitment_hex))
        return false;
    std::uint64_t count = 0;
    if (!read_u64(in, count) || count > kArtifactMaxMessages) return false;
    wire.chunks.clear();
    wire.chunks.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        hyrax::ChunkedMleChunkWire chunk;
        if (!read_u64(in, chunk.offset) || !read_u64(in, chunk.actual_length) || !read_u64(in, chunk.padded_length) || !read_string(in, chunk.label) ||
            !read_string(in, chunk.basis_id) || !read_string(in, chunk.commitment_hex) || !read_string(in, chunk.evaluation_mask_commitment_hex) ||
            !read_string(in, chunk.masked_evaluation) || !read_string(in, chunk.raw_evaluation_commitment_hex) ||
            !read_string(in, chunk.masked_evaluation_commitment_hex))
            return false;
        wire.chunks.push_back(std::move(chunk));
    }
    return true;
}

void write_chunk_lasso_public(std::ostream& out, const NativeLassoChunkFamilyPublic& family) {
    write_string(out, family.family_id);
    write_string(out, family.table_id);
    write_string(out, family.resource_id);
    write_string(out, family.registration_id);
    write_u64(out, family.segment_offset);
    write_u64(out, family.logical_queries);
    write_u64(out, family.padded_queries);
    write_u64(out, family.padding_index);
    write_string(out, family.padding_value.getStr(16));
    write_u64(out, family.chunk_size);
    write_string(out, family.statement_tables_digest);
    write_string(out, family.statement_resources_digest);
    write_u64(out, static_cast<std::uint64_t>(family.table.size()));
    for (const F& value : family.table) write_string(out, value.getStr(16));
}

bool read_chunk_lasso_public(std::istream& in, NativeLassoChunkFamilyPublic& family) {
    std::uint64_t segment = 0, logical = 0, padded = 0, padding = 0, chunk = 0, table_count = 0;
    std::string padding_value;
    if (!read_string(in, family.family_id) || !read_string(in, family.table_id) || !read_string(in, family.resource_id) ||
        !read_string(in, family.registration_id) || !read_u64(in, segment) || !read_u64(in, logical) || !read_u64(in, padded) || !read_u64(in, padding) ||
        !read_string(in, padding_value) || !read_u64(in, chunk) || !read_string(in, family.statement_tables_digest) ||
        !read_string(in, family.statement_resources_digest) || !read_u64(in, table_count) || table_count > (1ULL << 16))
        return false;
    if (segment > std::numeric_limits<std::size_t>::max() || logical > std::numeric_limits<std::size_t>::max() ||
        padded > std::numeric_limits<std::size_t>::max() || padding > std::numeric_limits<std::size_t>::max())
        return false;
    family.segment_offset = static_cast<std::size_t>(segment);
    family.logical_queries = static_cast<std::size_t>(logical);
    family.padded_queries = static_cast<std::size_t>(padded);
    family.padding_index = static_cast<std::size_t>(padding);
    family.chunk_size = chunk;
    if (!core_decode_field(padding_value, family.padding_value)) return false;
    family.table.clear();
    family.table.reserve(static_cast<std::size_t>(table_count));
    for (std::uint64_t index = 0; index < table_count; ++index) {
        std::string encoded;
        F value;
        if (!read_string(in, encoded) || !core_decode_field(encoded, value)) return false;
        family.table.push_back(value);
    }
    return true;
}

void write_chunk_lasso_wire(std::ostream& out, const NativeLassoChunkFamilyWire& family) {
    write_chunk_lasso_public(out, family.public_metadata);
    write_string(out, family.row_point_digest);
    write_chunked_mle_wire(out, family.query);
    write_u64(out, static_cast<std::uint64_t>(family.matrix_columns.size()));
    for (const hyrax::ChunkedMleWire& column : family.matrix_columns) write_chunked_mle_wire(out, column);
    write_string(out, family.lookup_terminal_commitment_hex);
    write_string(out, family.onehot_terminal_commitment_hex);
}

bool read_chunk_lasso_wire(std::istream& in, NativeLassoChunkFamilyWire& family) {
    if (!read_chunk_lasso_public(in, family.public_metadata) || !read_string(in, family.row_point_digest) || !read_chunked_mle_wire(in, family.query))
        return false;
    std::uint64_t count = 0;
    if (!read_u64(in, count) || count > kArtifactMaxMessages) return false;
    family.matrix_columns.clear();
    family.matrix_columns.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        hyrax::ChunkedMleWire column;
        if (!read_chunked_mle_wire(in, column)) return false;
        family.matrix_columns.push_back(std::move(column));
    }
    return read_string(in, family.lookup_terminal_commitment_hex) && read_string(in, family.onehot_terminal_commitment_hex);
}

void write_sparse_proof_wire(std::ostream& out, const SparseGkrProofWire& wire) {
    write_string(out, wire.schema);
    write_u32(out, wire.version);
    write_string(out, wire.plan_digest);
    write_string(out, wire.schedule_digest);
    write_u64(out, static_cast<std::uint64_t>(wire.relations.size()));
    for (const SparseGkrRelationWire& relation : wire.relations) {
        write_string(out, relation.relation_id);
        write_string(out, relation.initial_claim);
        write_string(out, relation.final_claim);
        write_u64(out, static_cast<std::uint64_t>(relation.terminal_values.size()));
        for (const std::string& value : relation.terminal_values) write_string(out, value);
        write_string(out, relation.initial_commitment);
        write_u64(out, static_cast<std::uint64_t>(relation.terminal_commitments.size()));
        for (const std::string& commitment : relation.terminal_commitments) write_string(out, commitment);
        write_u64(out, relation.transcript_begin);
        write_u64(out, relation.transcript_end);
    }
    const auto write_links = [&out](const std::vector<SparseGkrLinkWire>& links) {
        write_u64(out, static_cast<std::uint64_t>(links.size()));
        for (const SparseGkrLinkWire& link : links) {
            write_string(out, link.from_relation);
            write_string(out, link.to_relation);
            write_string(out, link.claim_name);
            write_u32(out, link.from_terminal);
            write_u32(out, link.to_initial);
            write_string(out, link.from_claim);
            write_string(out, link.to_claim);
        }
    };
    write_links(wire.terminal_links);
    write_links(wire.binding_links);
    write_string(out, wire.input_binding);
    write_string(out, wire.output_binding);
    write_string(out, wire.lasso_chunks.schema);
    write_u32(out, wire.lasso_chunks.version);
    write_string(out, wire.lasso_chunks.tables_digest);
    write_string(out, wire.lasso_chunks.resources_digest);
    write_string(out, wire.lasso_chunks.root_digest);
    write_u64(out, static_cast<std::uint64_t>(wire.lasso_chunks.families.size()));
    for (const NativeLassoChunkFamilyWire& family : wire.lasso_chunks.families) write_chunk_lasso_wire(out, family);
    write_string(out, wire.lasso_chunks.final_commitment_hex);
    write_u64(out, static_cast<std::uint64_t>(wire.input_segment_openings.size()));
    for (const hyrax::ChunkedMleWire& opening : wire.input_segment_openings) write_chunked_mle_wire(out, opening);
    write_chunked_mle_wire(out, wire.output_opening);
    write_string(out, wire.input_terminal_commitment);
    write_string(out, wire.output_terminal_commitment);
    write_u8(out, wire.hyrax_lasso_complete ? 1 : 0);
}

bool read_sparse_proof_wire(std::istream& in, SparseGkrProofWire& wire) {
    if (!read_string(in, wire.schema) || !read_u32(in, wire.version) || !read_string(in, wire.plan_digest) || !read_string(in, wire.schedule_digest))
        return false;
    std::uint64_t count = 0;
    if (!read_u64(in, count) || count > kArtifactMaxMessages) return false;
    wire.relations.clear();
    wire.relations.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        SparseGkrRelationWire relation;
        std::uint64_t value_count = 0;
        if (!read_string(in, relation.relation_id) || !read_string(in, relation.initial_claim) || !read_string(in, relation.final_claim) ||
            !read_u64(in, value_count) || value_count > kArtifactMaxMessages)
            return false;
        relation.terminal_values.reserve(static_cast<std::size_t>(value_count));
        for (std::uint64_t j = 0; j < value_count; ++j) {
            std::string value;
            if (!read_string(in, value)) return false;
            relation.terminal_values.push_back(std::move(value));
        }
        std::uint64_t commitment_count = 0;
        if (!read_string(in, relation.initial_commitment) || !read_u64(in, commitment_count) || commitment_count > kArtifactMaxMessages) return false;
        relation.terminal_commitments.reserve(static_cast<std::size_t>(commitment_count));
        for (std::uint64_t j = 0; j < commitment_count; ++j) {
            std::string commitment;
            if (!read_string(in, commitment)) return false;
            relation.terminal_commitments.push_back(std::move(commitment));
        }
        if (!read_u64(in, relation.transcript_begin) || !read_u64(in, relation.transcript_end)) return false;
        wire.relations.push_back(std::move(relation));
    }
    const auto read_links = [&in](std::vector<SparseGkrLinkWire>& links) {
        std::uint64_t link_count = 0;
        if (!read_u64(in, link_count) || link_count > kArtifactMaxMessages) return false;
        links.clear();
        links.reserve(static_cast<std::size_t>(link_count));
        for (std::uint64_t i = 0; i < link_count; ++i) {
            SparseGkrLinkWire link;
            if (!read_string(in, link.from_relation) || !read_string(in, link.to_relation) || !read_string(in, link.claim_name) ||
                !read_u32(in, link.from_terminal) || !read_u32(in, link.to_initial) || !read_string(in, link.from_claim) || !read_string(in, link.to_claim))
                return false;
            links.push_back(std::move(link));
        }
        return true;
    };
    if (!read_links(wire.terminal_links) || !read_links(wire.binding_links) || !read_string(in, wire.input_binding) || !read_string(in, wire.output_binding) ||
        !read_string(in, wire.lasso_chunks.schema) || !read_u32(in, wire.lasso_chunks.version) || !read_string(in, wire.lasso_chunks.tables_digest) ||
        !read_string(in, wire.lasso_chunks.resources_digest) || !read_string(in, wire.lasso_chunks.root_digest))
        return false;
    std::uint64_t family_count = 0;
    if (!read_u64(in, family_count) || family_count > kArtifactMaxMessages) return false;
    wire.lasso_chunks.families.clear();
    wire.lasso_chunks.families.reserve(static_cast<std::size_t>(family_count));
    for (std::uint64_t index = 0; index < family_count; ++index) {
        NativeLassoChunkFamilyWire family;
        if (!read_chunk_lasso_wire(in, family)) return false;
        wire.lasso_chunks.families.push_back(std::move(family));
    }
    if (!read_string(in, wire.lasso_chunks.final_commitment_hex)) return false;
    std::uint64_t opening_count = 0;
    if (!read_u64(in, opening_count) || opening_count > 4) return false;
    wire.input_segment_openings.clear();
    wire.input_segment_openings.reserve(static_cast<std::size_t>(opening_count));
    for (std::uint64_t index = 0; index < opening_count; ++index) {
        hyrax::ChunkedMleWire opening;
        if (!read_chunked_mle_wire(in, opening)) return false;
        wire.input_segment_openings.push_back(std::move(opening));
    }
    if (!read_chunked_mle_wire(in, wire.output_opening) || !read_string(in, wire.input_terminal_commitment) ||
        !read_string(in, wire.output_terminal_commitment))
        return false;
    std::uint8_t complete = 0;
    if (!read_u8(in, complete) || complete > 1) return false;
    wire.hyrax_lasso_complete = complete != 0;
    return true;
}

void set_reason(std::string* reason, const std::string& value);
bool is_lower_hex_digest(const std::string& value);

void append_u8_wire(std::string& out, std::uint8_t value) { out.push_back(static_cast<char>(value)); }

void append_i8_wire(std::string& out, std::int8_t value) { append_u8_wire(out, static_cast<std::uint8_t>(value)); }

void append_bool_wire(std::string& out, bool value) { append_u8_wire(out, value ? 1 : 0); }

class ContextReader {
  public:
    ContextReader(const std::string& wire, std::size_t begin, std::size_t end) : wire_(wire), position_(begin), end_(end) {}

    bool ok() const { return error_.empty(); }
    const std::string& error() const { return error_; }
    bool at_end() const { return ok() && position_ == end_; }
    std::size_t remaining() const { return position_ <= end_ ? end_ - position_ : 0; }

    void fail_external(const std::string& message) { fail(message); }

    bool read_u8(std::uint8_t& value) {
        if (!take(1)) return false;
        value = static_cast<std::uint8_t>(static_cast<unsigned char>(wire_[position_++]));
        return true;
    }

    bool read_u32(std::uint32_t& value) {
        if (!take(4)) return false;
        value = 0;
        for (unsigned i = 0; i < 4; ++i) value |= static_cast<std::uint32_t>(static_cast<unsigned char>(wire_[position_++])) << (i * 8);
        return true;
    }

    bool read_u64(std::uint64_t& value) {
        if (!take(8)) return false;
        value = 0;
        for (unsigned i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(static_cast<unsigned char>(wire_[position_++])) << (i * 8);
        return true;
    }

    bool read_bool(bool& value) {
        std::uint8_t raw = 0;
        if (!read_u8(raw) || raw > 1) {
            fail("non-canonical boolean");
            return false;
        }
        value = raw != 0;
        return true;
    }

    bool read_i8(std::int8_t& value) {
        std::uint8_t raw = 0;
        if (!read_u8(raw)) return false;
        value = raw <= 127 ? static_cast<std::int8_t>(raw) : static_cast<std::int8_t>(static_cast<int>(raw) - 256);
        return true;
    }

    bool read_size(std::size_t& value, const char* what) {
        std::uint64_t raw = 0;
        if (!read_u64(raw) || raw > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            fail(std::string("invalid size for ") + what);
            return false;
        }
        value = static_cast<std::size_t>(raw);
        return true;
    }

    bool read_count(std::size_t& value, const char* what, std::uint64_t limit = kContextMaxElements) {
        if (!read_size(value, what) || static_cast<std::uint64_t>(value) > limit) {
            fail(std::string("oversized count for ") + what);
            return false;
        }

        if (static_cast<std::uint64_t>(value) > remaining()) {
            fail(std::string("count exceeds remaining wire for ") + what);
            return false;
        }
        return true;
    }

    bool read_int(Int& value) {
        std::uint8_t sign = 0;
        std::uint64_t magnitude = 0;
        if (!read_u8(sign) || !read_u64(magnitude) || sign > 1) {
            fail("invalid signed integer wire");
            return false;
        }
        const std::uint64_t max_positive = static_cast<std::uint64_t>(std::numeric_limits<Int>::max());
        if (magnitude > max_positive + (sign == 1 ? 1ULL : 0ULL) || (sign == 1 && magnitude == 0)) {
            fail("signed integer is outside int64 canonical range");
            return false;
        }
        if (sign == 0)
            value = static_cast<Int>(magnitude);
        else if (magnitude == max_positive + 1ULL)
            value = std::numeric_limits<Int>::min();
        else
            value = -static_cast<Int>(magnitude);
        return true;
    }

    bool read_string(std::string& value, const char* what, std::uint64_t limit = kArtifactMaxString) {
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

    bool read_blob(std::string& value, const char* what, std::uint64_t limit = kContextMaxRecipeBytes) {
        std::uint64_t raw = 0;
        if (!read_u64(raw) || raw > limit || raw > remaining()) {
            fail(std::string("invalid or oversized blob for ") + what);
            return false;
        }
        value.assign(wire_.data() + position_, static_cast<std::size_t>(raw));
        position_ += static_cast<std::size_t>(raw);
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

    void fail(const std::string& message) {
        if (error_.empty()) error_ = message;
    }

    const std::string& wire_;
    std::size_t position_ = 0;
    std::size_t end_ = 0;
    std::string error_;
};

void append_context_commitment(std::string& out, const CommitmentWire& commitment) {
    append_string(out, commitment.scheme);
    append_string(out, commitment.basis_id);
    append_u64(out, static_cast<std::uint64_t>(commitment.logical_size));
    append_u64(out, static_cast<std::uint64_t>(commitment.padded_size));
    append_string(out, commitment.point_hex);
}

bool read_context_commitment(ContextReader& reader, CommitmentWire& commitment) {
    std::size_t logical = 0, padded = 0;
    return reader.read_string(commitment.scheme, "commitment scheme") && reader.read_string(commitment.basis_id, "commitment basis") &&
           reader.read_size(logical, "commitment logical size") && reader.read_size(padded, "commitment padded size") &&
           reader.read_string(commitment.point_hex, "commitment point") && (commitment.logical_size = logical, commitment.padded_size = padded, true);
}

void append_context_segment(std::string& out, const InputSegment& segment) {
    append_string(out, segment.name);
    append_u64(out, static_cast<std::uint64_t>(segment.offset));
    append_u64(out, static_cast<std::uint64_t>(segment.logical_length));
    append_u64(out, static_cast<std::uint64_t>(segment.padded_length));
}

bool read_context_segment(ContextReader& reader, InputSegment& segment) {
    return reader.read_string(segment.name, "input segment name") && reader.read_size(segment.offset, "input segment offset") &&
           reader.read_size(segment.logical_length, "input segment logical length") && reader.read_size(segment.padded_length, "input segment padded length");
}

void append_context_int_vector(std::string& out, const std::vector<Int>& values) {
    append_u64(out, static_cast<std::uint64_t>(values.size()));
    for (Int value : values) append_int(out, value);
}

bool read_context_int_vector(ContextReader& reader, std::vector<Int>& values, const char* what, std::uint64_t limit = kContextMaxElements) {
    std::size_t count = 0;
    if (!reader.read_count(count, what, limit)) return false;
    values.resize(count);
    for (Int& value : values)
        if (!reader.read_int(value)) return false;
    return true;
}

void append_context_quant(std::string& out, const QuantConfig& quant) {
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

bool read_context_quant(ContextReader& reader, QuantConfig& quant) {
    if (!reader.read_string(quant.schema, "quant schema") || !reader.read_string(quant.quant_cfg_id, "quant id") || !reader.read_int(quant.s_input) ||
        !reader.read_int(quant.s_delta) || !reader.read_int(quant.s_exp))
        return false;
    for (Int& value : quant.s_h)
        if (!reader.read_int(value)) return false;
    for (std::size_t layer = 0; layer < quant.s_w_hidden.size(); ++layer) {
        if (!read_context_int_vector(reader, quant.s_w_hidden[layer], "hidden weight scales", 64) ||
            !read_context_int_vector(reader, quant.s_hprime_hidden[layer], "hidden projection scales", 64) ||
            !read_context_int_vector(reader, quant.s_a_src_hidden[layer], "hidden source scales", 64) ||
            !read_context_int_vector(reader, quant.s_a_dst_hidden[layer], "hidden destination scales", 64))
            return false;
    }
    for (Int& value : quant.s_agg_pre)
        if (!reader.read_int(value)) return false;
    for (Int& value : quant.s_agg)
        if (!reader.read_int(value)) return false;
    for (Int& value : quant.s_cat)
        if (!reader.read_int(value)) return false;
    if (!reader.read_int(quant.s_out_pre) || !reader.read_int(quant.s_out_head) || !reader.read_int(quant.s_out)) return false;
    for (Int& value : quant.s_w_output)
        if (!reader.read_int(value)) return false;
    for (Int& value : quant.s_yprime_output)
        if (!reader.read_int(value)) return false;
    for (Int& value : quant.s_a_src_output)
        if (!reader.read_int(value)) return false;
    for (Int& value : quant.s_a_dst_output)
        if (!reader.read_int(value)) return false;
    for (Int& value : quant.s_b_output)
        if (!reader.read_int(value)) return false;
    return reader.read_string(quant.rounding, "rounding policy") && reader.read_string(quant.rescale, "rescale policy") &&
           reader.read_string(quant.output_average, "output average policy") && reader.read_int(quant.range_bits) &&
           reader.read_string(quant.static_tables_id, "static tables id") && reader.read_string(quant.static_table_generator_id, "static table generator") &&
           reader.read_string(quant.static_row_order, "static table row order") && reader.read_string(quant.exp_approx_id, "exp approximation id") &&
           reader.read_string(quant.elu_approx_id, "elu approximation id") && reader.read_int(quant.static_lrelu_min) &&
           reader.read_int(quant.static_lrelu_max) && reader.read_int(quant.static_exp_min) && reader.read_int(quant.static_exp_max) &&
           reader.read_int(quant.static_elu_min) && reader.read_int(quant.static_elu_max) && reader.read_int(quant.lrelu_slope_num) &&
           reader.read_int(quant.lrelu_slope_den);
}

void append_context_tables(std::string& out, const StaticTables& tables) {
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
    const auto append_table = [&out](const std::vector<TablePoint>& table) {
        append_u64(out, static_cast<std::uint64_t>(table.size()));
        for (const TablePoint& point : table) {
            append_int(out, point.x);
            append_int(out, point.y);
        }
    };
    append_table(tables.lrelu);
    append_table(tables.exp);
    append_table(tables.elu);
}

bool read_context_tables(ContextReader& reader, StaticTables& tables) {
    if (!reader.read_string(tables.table_id, "table id") || !reader.read_string(tables.generator_id, "table generator") ||
        !reader.read_string(tables.digest, "table digest") || !reader.read_string(tables.quant_cfg_id, "table quant id") ||
        !reader.read_string(tables.rounding, "table rounding") || !reader.read_string(tables.rescale, "table rescale") ||
        !reader.read_string(tables.output_average, "table output average") || !reader.read_string(tables.row_order, "table row order") ||
        !reader.read_string(tables.exp_approx_id, "table exp approximation") || !reader.read_string(tables.elu_approx_id, "table elu approximation") ||
        !reader.read_int(tables.lrelu_slope_num) || !reader.read_int(tables.lrelu_slope_den) || !reader.read_int(tables.s_delta) ||
        !reader.read_int(tables.s_exp) || !reader.read_int(tables.s_elu_in) || !reader.read_int(tables.s_elu_out) || !reader.read_int(tables.lrelu_min) ||
        !reader.read_int(tables.lrelu_max) || !reader.read_int(tables.exp_min) || !reader.read_int(tables.exp_max) || !reader.read_int(tables.elu_min) ||
        !reader.read_int(tables.elu_max))
        return false;
    const auto read_table = [&reader](std::vector<TablePoint>& table, const char* what) -> bool {
        std::size_t count = 0;
        if (!reader.read_count(count, what, 1ULL << 24)) return false;
        table.resize(count);
        for (TablePoint& point : table)
            if (!reader.read_int(point.x) || !reader.read_int(point.y)) return false;
        return true;
    };
    return read_table(tables.lrelu, "LeakyReLU table") && read_table(tables.exp, "Exp table") && read_table(tables.elu, "ELU table");
}

void append_context_registration(std::string& out, const ModelRegistration& registration) {
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
    for (const InputSegment& segment : registration.input_segments) append_context_segment(out, segment);
    append_string(out, registration.opening_policy);
    append_string(out, registration.binding_digest);
    append_string(out, registration.registration_digest);
    append_string(out, registration.status);
}

bool read_context_registration(ContextReader& reader, ModelRegistration& registration) {
    if (!read_context_commitment(reader, registration.com_W) || !reader.read_string(registration.dataset, "registration dataset") ||
        !reader.read_string(registration.model_arch_id, "registration architecture") ||
        !reader.read_string(registration.quant_cfg_id, "registration quant id") ||
        !reader.read_string(registration.static_tables_id, "registration table id") || !reader.read_string(registration.field_id, "registration field id") ||
        !reader.read_string(registration.field_modulus, "registration field modulus") ||
        !reader.read_string(registration.manifest_digest, "registration manifest digest") ||
        !reader.read_string(registration.model_digest, "registration model digest") ||
        !reader.read_string(registration.bounds_digest, "registration bounds digest") ||
        !reader.read_string(registration.quant_digest, "registration quant digest") ||
        !reader.read_string(registration.layout_digest, "registration layout digest") || !reader.read_string(registration.selector, "registration selector") ||
        !reader.read_size(registration.n_raw, "registration raw layout size") || !reader.read_size(registration.n_in, "registration padded layout size"))
        return false;
    std::size_t count = 0;
    if (!reader.read_count(count, "registration input segments", 8)) return false;
    registration.input_segments.resize(count);
    for (InputSegment& segment : registration.input_segments)
        if (!read_context_segment(reader, segment)) return false;
    if (!reader.read_string(registration.opening_policy, "registration opening policy") ||
        !reader.read_string(registration.binding_digest, "registration binding digest") ||
        !reader.read_string(registration.registration_digest, "registration digest") || !reader.read_string(registration.status, "registration status"))
        return false;
    registration.com_W_padding_mask = Fr(0);
    registration.has_private_padding_mask = false;
    return true;
}

#if 0
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
    append_u64(out, plan.residual_targets.size());
    for (Int target : plan.residual_targets) append_u64(out, static_cast<std::uint64_t>(target));
    append_u64(out, plan.division_range_bits.size());
    for (const auto& profile : plan.division_range_bits) for (unsigned width : profile) append_u64(out, width);
    append_bool_wire(out, plan.official_layered);
    append_bool_wire(out, plan.has_placeholder);
    append_u8_wire(out, plan.circuit.size);
    append_u64(out, static_cast<std::uint64_t>(plan.circuit.circuit.size()));
    for (const layer &current : plan.circuit.circuit) {
        append_u32(out, static_cast<std::uint32_t>(current.ty));
        append_u64(out, current.size);
        for (unsigned i = 0; i < 2; ++i) append_u64(out, current.size_u[i]);
        for (unsigned i = 0; i < 2; ++i) append_u64(out, current.size_v[i]);
        for (unsigned i = 0; i < 2; ++i) append_i8_wire(out, current.bit_length_u[i]);
        for (unsigned i = 0; i < 2; ++i) append_i8_wire(out, current.bit_length_v[i]);
        append_i8_wire(out, current.bit_length);
        append_i8_wire(out, current.max_bl_u);
        append_i8_wire(out, current.max_bl_v);
        append_bool_wire(out, current.need_phase2);
        append_u64(out, current.zero_start_id);
        append_i8_wire(out, current.fft_bit_length);
        append_u64(out, static_cast<std::uint64_t>(current.uni_interval.size()));
        for (const auto &interval : current.uni_interval) {
            append_u64(out, interval.first);
            append_u64(out, interval.second);
        }
        append_u64(out, static_cast<std::uint64_t>(current.bin_interval.size()));
        for (const auto &interval : current.bin_interval) {
            append_u64(out, interval.first);
            append_u64(out, interval.second);
        }
        append_u64(out, static_cast<std::uint64_t>(current.ori_id_u.size()));
        for (u64 value : current.ori_id_u) append_u64(out, value);
        append_u64(out, static_cast<std::uint64_t>(current.ori_id_v.size()));
        for (u64 value : current.ori_id_v) append_u64(out, value);
        append_u64(out, static_cast<std::uint64_t>(current.uni_gates.size()));
        for (const uniGate &gate : current.uni_gates) {
            append_u64(out, gate.g);
            append_u64(out, gate.u);
            append_u8_wire(out, gate.lu);
            append_int(out, current.coefficient(gate));
        }
        append_u64(out, static_cast<std::uint64_t>(current.bin_gates.size()));
        for (const binGate &gate : current.bin_gates) {
            append_u64(out, gate.g);
            append_u64(out, gate.u);
            append_u64(out, gate.v);
            append_u8_wire(out, gate.l);
            append_int(out, current.coefficient(gate));
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
        !reader.read_size(plan.residual_coordinates, "residual coordinates") || !reader.read_count(count, "residual targets")) return false;
    plan.residual_targets.resize(count);
    for (std::size_t index = 0; index < count; ++index) { std::uint64_t encoded = 0; if (!reader.read_u64(encoded)) return false; plan.residual_targets.set(index, static_cast<Int>(encoded)); }
    plan.residual_targets.freeze();
    if (!reader.read_count(count, "division range profiles", kContextMaxDescriptors)) return false;
    plan.division_range_bits.resize(count);
    for (auto& profile : plan.division_range_bits) for (unsigned& width : profile) { std::uint64_t encoded = 0; if (!reader.read_u64(encoded) || encoded == 0 || encoded > 63) return false; width = static_cast<unsigned>(encoded); }
    if (!reader.read_bool(plan.official_layered) || !reader.read_bool(plan.has_placeholder)) return false;
    std::uint8_t circuit_size = 0;
    if (!reader.read_u8(circuit_size) ||
        !reader.read_count(count, "layered circuit layers", kContextMaxLayers)) return false;
    plan.circuit.size = circuit_size;
    plan.circuit.circuit.resize(count);
    for (layer &current : plan.circuit.circuit) {
        std::uint32_t type = 0; std::uint64_t size = 0;
        if (!reader.read_u32(type) || type > static_cast<std::uint32_t>(layerType::PLACE_HOLDER) ||
            !reader.read_u64(size)) return false;
        current.ty = static_cast<layerType>(type);
        current.size = size;
        for (unsigned i = 0; i < 2; ++i) if (!reader.read_u64(current.size_u[i])) return false;
        for (unsigned i = 0; i < 2; ++i) if (!reader.read_u64(current.size_v[i])) return false;
        for (unsigned i = 0; i < 2; ++i) if (!reader.read_i8(current.bit_length_u[i])) return false;
        for (unsigned i = 0; i < 2; ++i) if (!reader.read_i8(current.bit_length_v[i])) return false;
        if (!reader.read_i8(current.bit_length) || !reader.read_i8(current.max_bl_u) ||
            !reader.read_i8(current.max_bl_v) || !reader.read_bool(current.need_phase2) ||
            !reader.read_u64(current.zero_start_id) || !reader.read_i8(current.fft_bit_length)) return false;
        if (!reader.read_count(count, "unary interval entries", kContextMaxElements)) return false;
        current.uni_interval.resize(count);
        for (auto &interval : current.uni_interval) {
            std::uint64_t first = 0, second = 0;
            if (!reader.read_u64(first) || !reader.read_u64(second)) return false;
            interval = {first, second};
        }
        if (!reader.read_count(count, "binary interval entries", kContextMaxElements)) return false;
        current.bin_interval.resize(count);
        for (auto &interval : current.bin_interval) {
            std::uint64_t first = 0, second = 0;
            if (!reader.read_u64(first) || !reader.read_u64(second)) return false;
            interval = {first, second};
        }
        if (!reader.read_count(count, "unary original ids", kContextMaxElements)) return false;
        current.ori_id_u.resize(count);
        for (PackedU40 &value : current.ori_id_u) {
            u64 decoded = 0;
            if (!reader.read_u64(decoded) || decoded > PackedU40::kMaxValue) return false;
            value = decoded;
        }
        if (!reader.read_count(count, "binary original ids", kContextMaxElements)) return false;
        current.ori_id_v.resize(count);
        for (PackedU40 &value : current.ori_id_v) {
            u64 decoded = 0;
            if (!reader.read_u64(decoded) || decoded > PackedU40::kMaxValue) return false;
            value = decoded;
        }
        if (!reader.read_count(count, "unary gates", kContextMaxElements)) return false;
        current.reserve_gate_arrays(count, 0);
        for (std::size_t i = 0; i < count; ++i) {
            std::uint64_t g = 0, u = 0;
            std::uint8_t lu = 0;
            Int sc = 0;
            if (!reader.read_u64(g) || !reader.read_u64(u) || !reader.read_u8(lu) ||
                !reader.read_int(sc)) return false;
            current.add_unary(g, u, lu, sc);
        }
        if (!reader.read_count(count, "binary gates", kContextMaxElements)) return false;
        current.reserve_gate_arrays(0, count);
        for (std::size_t i = 0; i < count; ++i) {
            std::uint64_t g = 0, u = 0, v = 0;
            std::uint8_t l = 0;
            Int sc = 0;
            if (!reader.read_u64(g) || !reader.read_u64(u) || !reader.read_u64(v) ||
                !reader.read_u8(l) || !reader.read_int(sc)) return false;
            current.add_binary(g, u, v, sc, l);
        }
    }
    return true;
}
#endif

bool materialize_compact_plan(const Graph& graph, const QuantConfig& quant, const GatCircuitPlan& recipe, GatCircuitPlan& materialized, std::string* reason) {
    const auto materialize_begin = std::chrono::steady_clock::now();
    std::cerr << "gat_adapter_prover_stage: fullplan-materialize begin" << std::endl;
    const auto fail = [reason](const std::string& message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    try {
        if (!canonical_model_shape_valid(recipe.model_shape, reason)) return false;
        if (recipe.model_arch_id != recipe.model_shape.model_arch_id || recipe.graph_num_nodes != graph.num_nodes ||
            recipe.graph_num_features != graph.num_features || recipe.graph_count != graph.graph_count || recipe.graph_num_classes != graph.num_classes ||
            recipe.graph_edge_count != graph.edges() || recipe.graph_topology_digest != canonical_graph_topology_digest(graph) ||
            recipe.quant_cfg_id != quant.quant_cfg_id || recipe.quant_digest != canonical_quant_digest(quant) ||
            recipe.static_tables_id != StaticTables::build(quant).table_id) {
            return fail("compact plan recipe does not match authenticated public graph or quantization");
        }

        const Model shape_model = make_shape_only_model(recipe.model_shape);
        GatCircuitPlan generated =
            recipe.virtual_sparse ? build_virtual_sparse_plan(graph, shape_model, quant) : build_plan_impl(graph, shape_model, quant, nullptr);
        if (plan_encoding(generated) != plan_encoding(recipe)) { return fail("compact plan recipe is not canonical for the shared materializer"); }

        std::string validation_reason;
        if (!validate_gat_circuit_plan_impl(generated, &validation_reason) || !context_layer_metadata_valid(generated, &validation_reason)) {
            return fail(validation_reason.empty() ? "materialized compact plan failed validation" : validation_reason);
        }
        materialized = std::move(generated);
        std::cerr << "gat_adapter_prover_stage: fullplan-materialize complete elapsed_ms="
                  << std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - materialize_begin).count()
                  << std::endl;
        return true;
    } catch (const std::exception& error) { return fail(error.what()); }
}

void append_context_plan(std::string& out, const GatCircuitPlan& plan) { append_string(out, plan_encoding(plan)); }

bool read_compact_plan_recipe(ContextReader& reader, GatCircuitPlan& plan) {
    std::uint64_t sparse_version = 0;
    if (!reader.read_string(plan.recipe_schema, "plan recipe schema") || plan.recipe_schema != "canonical_plan_recipe_v3" ||
        !reader.read_string(plan.relation_schema, "plan relation schema") || plan.relation_schema != "zkgpt-gat-relation-v1" ||
        !reader.read_string(plan.sparse_relation_schema, "sparse relation schema") || plan.sparse_relation_schema != kSparseGkrRelationSchema ||
        !reader.read_u64(sparse_version) || sparse_version != kSparseGkrRelationVersion ||
        !reader.read_size(plan.sparse_chunk_size, "sparse relation chunk size") || plan.sparse_chunk_size != kSparseGkrChunkEntries ||
        !reader.read_string(plan.sparse_plan_digest, "sparse plan digest") || !reader.read_bool(plan.virtual_sparse) ||
        !reader.read_string(plan.model_arch_id, "plan model architecture") || !reader.read_string(plan.model_shape.model_arch_id, "model shape architecture") ||
        !reader.read_size(plan.model_shape.tensor_count, "model shape tensor count") || !reader.read_size(plan.model_shape.layers, "model shape layer count") ||
        !reader.read_size(plan.model_shape.output_heads, "model shape output head count") ||
        !reader.read_size(plan.model_shape.classes, "model shape class count") ||
        !reader.read_size(plan.model_shape.input_dim, "model shape input dimension") ||
        !reader.read_size(plan.model_shape.hidden_dim, "model shape hidden dimension"))
        return false;
    plan.sparse_relation_version = static_cast<std::uint32_t>(sparse_version);
    std::size_t count = 0;
    if (!reader.read_count(count, "model shape tensor profile", kContextMaxShapeProfile)) return false;
    plan.model_shape.tensors.resize(count);
    for (CanonicalTensorShape& tensor : plan.model_shape.tensors) {
        if (!reader.read_string(tensor.name, "model shape tensor name")) return false;
        std::size_t dims = 0;
        if (!reader.read_count(dims, "model shape dimensions", kContextMaxShapeDimensions)) return false;
        tensor.shape.dims.resize(dims);
        for (std::size_t& dimension : tensor.shape.dims)
            if (!reader.read_size(dimension, "model shape dimension")) return false;
    }
    if (!reader.read_size(plan.graph_num_nodes, "plan graph node count") || !reader.read_size(plan.graph_num_features, "plan graph feature count") ||
        !reader.read_size(plan.graph_count, "plan graph count") || !reader.read_size(plan.graph_num_classes, "plan graph class count") ||
        !reader.read_size(plan.graph_edge_count, "plan graph edge count") || !reader.read_string(plan.graph_topology_digest, "plan graph topology digest") ||
        !reader.read_string(plan.quant_cfg_id, "plan quantization id") || !reader.read_string(plan.quant_digest, "plan quantization digest") ||
        !reader.read_string(plan.static_tables_id, "plan static tables id") || !reader.read_string(plan.registry.registry_id, "registry id"))
        return false;
    if (!reader.read_count(count, "registry descriptors", kContextMaxDescriptors)) return false;
    plan.registry.descriptors.resize(count);
    for (RelationDescriptor& descriptor : plan.registry.descriptors) {
        if (!reader.read_string(descriptor.id, "relation id") || !reader.read_string(descriptor.primitive, "relation primitive") ||
            !reader.read_string(descriptor.segment, "relation segment") || !reader.read_string(descriptor.relation, "relation description") ||
            !reader.read_string(descriptor.public_claim, "relation public claim") || !reader.read_string(descriptor.witness, "relation witness"))
            return false;
    }
    if (!reader.read_string(plan.input_layout.selector, "layout selector") || !reader.read_string(plan.opening_policy, "plan opening policy") ||
        !reader.read_size(plan.public_output_coordinates, "public output coordinates") ||
        !reader.read_size(plan.residual_coordinates, "residual coordinates") || !reader.read_string(plan.residual_target_digest, "public residual target digest") || !reader.read_count(count, "division range profiles", kContextMaxDescriptors)) return false;
    plan.division_range_bits.resize(count);
    for (auto& profile : plan.division_range_bits) for (unsigned& width : profile) { std::uint64_t encoded = 0; if (!reader.read_u64(encoded) || encoded == 0 || encoded > 63) return false; width = static_cast<unsigned>(encoded); }
    if (!reader.read_count(count, "terminal openings", 16)) return false;
    plan.terminal_openings.resize(count);
    for (std::string& opening : plan.terminal_openings)
        if (!reader.read_string(opening, "terminal opening")) return false;
    if (!reader.read_size(plan.input_layout.n_raw, "layout raw size") || !reader.read_size(plan.input_layout.n_in, "layout padded size") ||
        !reader.read_count(count, "layout segments", 8))
        return false;
    plan.input_layout.segments.resize(count);
    for (InputSegment& segment : plan.input_layout.segments)
        if (!read_context_segment(reader, segment)) return false;
    if (!reader.read_count(count, "MatMul descriptors", kContextMaxDescriptors)) return false;
    plan.matmuls.resize(count);
    for (MatMulDescriptor& descriptor : plan.matmuls) {
        if (!reader.read_string(descriptor.relation_id, "MatMul relation") || !reader.read_string(descriptor.family_id, "MatMul family") ||
            !reader.read_string(descriptor.lhs, "MatMul lhs") || !reader.read_string(descriptor.rhs, "MatMul rhs") ||
            !reader.read_string(descriptor.output, "MatMul output") || !reader.read_size(descriptor.rows, "MatMul rows") ||
            !reader.read_size(descriptor.inner, "MatMul inner") || !reader.read_size(descriptor.cols, "MatMul cols") ||
            !reader.read_size(descriptor.output_coordinates, "MatMul output coordinates") || !reader.read_size(descriptor.lhs_offset, "MatMul lhs offset") ||
            !reader.read_size(descriptor.rhs_offset, "MatMul rhs offset") || !reader.read_size(descriptor.output_offset, "MatMul output offset") ||
            !reader.read_string(descriptor.coordinate_rule, "MatMul coordinate rule") || !reader.read_string(descriptor.scale_rule, "MatMul scale rule") ||
            !reader.read_string(descriptor.native_path, "MatMul native path") || !reader.read_string(descriptor.relation_schema, "MatMul relation schema") ||
            !reader.read_string(descriptor.lhs_segment, "MatMul lhs segment") || !reader.read_string(descriptor.rhs_segment, "MatMul rhs segment") ||
            !reader.read_string(descriptor.output_segment, "MatMul output segment") ||
            !reader.read_string(descriptor.lhs_coordinate_rule, "MatMul lhs coordinate rule") ||
            !reader.read_string(descriptor.rhs_coordinate_rule, "MatMul rhs coordinate rule") ||
            !reader.read_string(descriptor.output_coordinate_rule, "MatMul output coordinate rule") ||
            !reader.read_string(descriptor.terminal_claim_rule, "MatMul terminal claim rule"))
            return false;
    }
    if (!reader.read_count(count, "FCONN claim links", kContextMaxDescriptors)) return false;
    plan.fconn_claim_links.resize(count);
    for (FconnClaimLink& link : plan.fconn_claim_links) {
        if (!reader.read_size(link.descriptor_index, "FCONN descriptor index") || !reader.read_string(link.relation_id, "FCONN link relation") ||
            !reader.read_string(link.family_id, "FCONN link family") || !reader.read_string(link.lhs_segment, "FCONN lhs segment") ||
            !reader.read_string(link.rhs_segment, "FCONN rhs segment") || !reader.read_string(link.output_segment, "FCONN output segment") ||
            !reader.read_size(link.lhs_offset, "FCONN lhs offset") || !reader.read_size(link.rhs_offset, "FCONN rhs offset") ||
            !reader.read_size(link.output_offset, "FCONN output offset") || !reader.read_size(link.rows, "FCONN rows") ||
            !reader.read_size(link.inner, "FCONN inner") || !reader.read_size(link.cols, "FCONN cols") ||
            !reader.read_string(link.lhs_point_rule, "FCONN lhs point rule") || !reader.read_string(link.rhs_point_rule, "FCONN rhs point rule") ||
            !reader.read_string(link.output_point_rule, "FCONN output point rule") || !reader.read_string(link.source_rule, "FCONN source rule"))
            return false;
    }
    if (!reader.read_count(count, "sparse GKR descriptors", kContextMaxDescriptors)) return false;
    plan.sparse_gkr.resize(count);
    for (SparseGkrDescriptor& descriptor : plan.sparse_gkr) {
        std::uint8_t relation_kind = 0;
        if (!reader.read_string(descriptor.relation_id, "sparse relation") || !reader.read_string(descriptor.family_id, "sparse family") ||
            !reader.read_u8(relation_kind) || !sparse_gkr_relation_kind_valid(relation_kind) ||
            !reader.read_string(descriptor.relation_schema, "sparse relation schema") || descriptor.relation_schema != kSparseGkrRelationSchema ||
            !reader.read_size(descriptor.edge_count, "sparse edge count") || !reader.read_size(descriptor.value_width, "sparse value width") ||
            !reader.read_size(descriptor.chunk_size, "sparse chunk size") || !reader.read_size(descriptor.input_offset, "sparse input offset") ||
            !reader.read_size(descriptor.source_offset, "sparse source offset") ||
            !reader.read_size(descriptor.destination_offset, "sparse destination offset") ||
            !reader.read_size(descriptor.output_offset, "sparse output offset") || !reader.read_string(descriptor.coordinate_rule, "sparse coordinate rule") ||
            !reader.read_string(descriptor.group_rule, "sparse group rule") || !reader.read_string(descriptor.arithmetic_rule, "sparse arithmetic rule") ||
            !reader.read_string(descriptor.overflow_rule, "sparse overflow rule") || !reader.read_string(descriptor.terminal_rule, "sparse terminal rule") ||
            !reader.read_count(count, "sparse constraints", kContextMaxDescriptors))
            return false;
        descriptor.relation_kind = static_cast<SparseGkrRelationKind>(relation_kind);
        descriptor.constraints.resize(count);
        for (std::string& constraint : descriptor.constraints)
            if (!reader.read_string(constraint, "sparse constraint")) return false;
    }
    if (!reader.read_count(count, "native relation descriptors", kContextMaxDescriptors)) return false;
    plan.native_relations.resize(count);
    for (NativeRelationDescriptor& descriptor : plan.native_relations) {
        if (!reader.read_string(descriptor.relation_id, "native relation id") || !reader.read_string(descriptor.family_id, "native relation family") ||
            !reader.read_string(descriptor.relation_schema, "native relation schema") ||
            !reader.read_string(descriptor.primitive, "native relation primitive") ||
            !reader.read_string(descriptor.domain_rule, "native relation domain rule") ||
            !reader.read_string(descriptor.selector_rule, "native relation selector rule") ||
            !reader.read_string(descriptor.coordinate_rule, "native relation coordinate rule") ||
            !reader.read_string(descriptor.arithmetic_rule, "native relation arithmetic rule") ||
            !reader.read_string(descriptor.terminal_rule, "native relation terminal rule") ||
            !reader.read_size(descriptor.edge_count, "native relation edge count") ||
            !reader.read_size(descriptor.value_width, "native relation value width") ||
            !reader.read_size(descriptor.chunk_size, "native relation chunk size") ||
            !reader.read_size(descriptor.input_offset, "native relation input offset") ||
            !reader.read_size(descriptor.output_offset, "native relation output offset") ||
            !reader.read_count(count, "native relation equations", kContextMaxDescriptors))
            return false;
        descriptor.equations.resize(count);
        for (std::string& equation : descriptor.equations)
            if (!reader.read_string(equation, "native relation equation")) return false;
    }
    if (!reader.read_count(count, "Lasso descriptors", kContextMaxDescriptors)) return false;
    plan.lasso.resize(count);
    for (LassoFamilyDescriptor& descriptor : plan.lasso) {
        if (!reader.read_string(descriptor.relation_id, "Lasso relation") || !reader.read_string(descriptor.family_id, "Lasso family") ||
            !reader.read_string(descriptor.table_id, "Lasso table") || !reader.read_size(descriptor.real_queries, "Lasso real query count") ||
            !reader.read_size(descriptor.padded_queries, "Lasso padded query count") || !reader.read_size(descriptor.table_domain, "Lasso table domain") ||
            !reader.read_int(descriptor.padding_value) || !reader.read_size(descriptor.query_offset, "Lasso query offset") ||
            !reader.read_string(descriptor.incidence_rule, "Lasso incidence rule") ||
            !reader.read_string(descriptor.challenge_policy, "Lasso challenge policy") ||
            !reader.read_string(descriptor.query_encoding, "Lasso query encoding"))
            return false;
    }
    if (!reader.read_count(count, "checkpoint descriptors", kContextMaxDescriptors)) return false;
    plan.checkpoints.resize(count);
    for (CheckpointDescriptor& checkpoint : plan.checkpoints) {
        if (!reader.read_string(checkpoint.id, "checkpoint id") || !reader.read_string(checkpoint.relation_id, "checkpoint relation") ||
            !reader.read_string(checkpoint.segment, "checkpoint segment") || !reader.read_size(checkpoint.coordinates, "checkpoint coordinates") ||
            !reader.read_size(checkpoint.coordinate_offset, "checkpoint offset") ||
            !reader.read_string(checkpoint.coordinate_rule, "checkpoint coordinate rule"))
            return false;
    }
    if (!reader.read_count(count, "native API entries", kContextMaxDescriptors)) return false;
    plan.native_api.resize(count);
    for (std::string& api : plan.native_api)
        if (!reader.read_string(api, "native API entry")) return false;

    plan.circuit = layeredCircuit();
    plan.official_layered = false;
    plan.has_placeholder = false;
    return true;
}

bool read_context_plan(ContextReader& reader, GatCircuitPlan& plan) {
    std::string recipe;
    if (!reader.read_blob(recipe, "compact plan recipe", kContextMaxRecipeBytes)) return false;
    ContextReader recipe_reader(recipe, 0, recipe.size());
    if (!read_compact_plan_recipe(recipe_reader, plan) || !recipe_reader.at_end()) {
        reader.fail_external(recipe_reader.error().empty() ? "malformed compact plan recipe" : recipe_reader.error());
        return false;
    }
    return true;
}

void append_context_graph(std::string& out, const Graph& graph) {
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

bool read_context_graph(ContextReader& reader, Graph& graph) {
    if (!reader.read_size(graph.num_nodes, "graph node count") || !reader.read_size(graph.num_features, "graph feature count") ||
        !reader.read_size(graph.graph_count, "graph count") || !reader.read_size(graph.num_classes, "graph class count") ||
        !read_context_int_vector(reader, graph.src, "graph source edges") || !read_context_int_vector(reader, graph.dst, "graph destination edges") ||
        !read_context_int_vector(reader, graph.edge_gid, "graph edge graph ids"))
        return false;
    std::size_t count = 0;
    if (!reader.read_count(count, "graph node pointers")) return false;
    graph.node_ptr.resize(count);
    for (std::size_t& value : graph.node_ptr)
        if (!reader.read_size(value, "graph node pointer")) return false;
    if (!reader.read_count(count, "graph edge pointers")) return false;
    graph.edge_ptr.resize(count);
    for (std::size_t& value : graph.edge_ptr)
        if (!reader.read_size(value, "graph edge pointer")) return false;
    return true;
}

void append_context_tensor(std::string& out, const Tensor2D& tensor) {
    append_u64(out, static_cast<std::uint64_t>(tensor.rows));
    append_u64(out, static_cast<std::uint64_t>(tensor.cols));
    append_context_int_vector(out, tensor.values);
}

bool read_context_tensor(ContextReader& reader, Tensor2D& tensor, const char* what) {
    if (!reader.read_size(tensor.rows, what) || !reader.read_size(tensor.cols, what)) return false;
    const std::size_t expected = checked_mul(tensor.rows, tensor.cols, "verifier context tensor dimensions");
    std::size_t count = 0;
    if (!reader.read_count(count, what) || count != expected) return false;
    tensor.values.resize(count);
    for (Int& value : tensor.values)
        if (!reader.read_int(value)) return false;
    return true;
}

bool static_tables_equal(const StaticTables& left, const StaticTables& right) {
    const auto equal_rows = [](const std::vector<TablePoint>& a, const std::vector<TablePoint>& b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i].x != b[i].x || a[i].y != b[i].y) return false;
        return true;
    };
    return left.table_id == right.table_id && left.generator_id == right.generator_id && left.digest == right.digest &&
           left.quant_cfg_id == right.quant_cfg_id && left.rounding == right.rounding && left.rescale == right.rescale &&
           left.output_average == right.output_average && left.row_order == right.row_order && left.exp_approx_id == right.exp_approx_id &&
           left.elu_approx_id == right.elu_approx_id && left.lrelu_slope_num == right.lrelu_slope_num && left.lrelu_slope_den == right.lrelu_slope_den &&
           left.s_delta == right.s_delta && left.s_exp == right.s_exp && left.s_elu_in == right.s_elu_in && left.s_elu_out == right.s_elu_out &&
           left.lrelu_min == right.lrelu_min && left.lrelu_max == right.lrelu_max && left.exp_min == right.exp_min && left.exp_max == right.exp_max &&
           left.elu_min == right.elu_min && left.elu_max == right.elu_max && equal_rows(left.lrelu, right.lrelu) && equal_rows(left.exp, right.exp) &&
           equal_rows(left.elu, right.elu);
}

bool registry_equal(const RelationRegistry& left, const RelationRegistry& right) {
    if (left.registry_id != right.registry_id || left.descriptors.size() != right.descriptors.size()) return false;
    for (std::size_t i = 0; i < left.descriptors.size(); ++i) {
        const RelationDescriptor& a = left.descriptors[i];
        const RelationDescriptor& b = right.descriptors[i];
        if (a.id != b.id || a.primitive != b.primitive || a.segment != b.segment || a.relation != b.relation || a.public_claim != b.public_claim ||
            a.witness != b.witness)
            return false;
    }
    return true;
}

bool context_layer_metadata_valid(const GatCircuitPlan& plan, std::string* reason) {
    const auto fail = [reason](const std::string& message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    if (!plan.native_proof_backing_is_resident()) return fail("native GAT proof backing is not resident");
    if (plan.virtual_sparse) {
        if (!validate_gat_circuit_plan_impl(plan, reason)) return false;
        return true;
    }
    if (plan.circuit.circuit.size() != plan.circuit.size) return fail("layered circuit size field mismatch");
    std::size_t fconn_index = 0;
    for (std::size_t i = 0; i < plan.circuit.circuit.size(); ++i) {
        const layer& current = plan.circuit.circuit[i];
        if (current.zero_start_id != current.size) return fail("layer zero_start_id is not canonical");
        if (i == 0) {
            if (!current.ori_id_u.empty() || !current.ori_id_v.empty() || !current.uni_gates.empty() || !current.bin_gates.empty())
                return fail("input layer contains derived/gate state");
            continue;
        }
        if (current.ty == layerType::FCONN) {
            if (fconn_index >= plan.matmuls.size() || !current.uni_gates.empty() || !current.bin_gates.empty() || !current.ori_id_u.empty() ||
                !current.ori_id_v.empty() || current.native_relation_id.empty() || current.native_family_id.empty())
                return fail("FCONN layer metadata is not compact/canonical");
            const MatMulDescriptor& descriptor = plan.matmuls[fconn_index++];
            if (current.native_relation_id != descriptor.relation_id || current.native_family_id != descriptor.family_id ||
                current.native_rows != descriptor.rows || current.native_inner != descriptor.inner || current.native_cols != descriptor.cols ||
                current.native_lhs_offset != descriptor.lhs_offset || current.native_rhs_offset != descriptor.rhs_offset ||
                current.native_output_offset != descriptor.output_offset ||
                current.size != fconn_output_domain(descriptor.rows, descriptor.cols))
                return fail("FCONN layer does not match compact MatMul descriptor");
            continue;
        }
        if (i > 0 && plan.circuit.circuit[i - 1].ty == layerType::FCONN) {
            const layer& previous = plan.circuit.circuit[i - 1];
            const MatMulDescriptor* descriptor = nullptr;
            for (const MatMulDescriptor& candidate : plan.matmuls)
                if (candidate.relation_id == previous.native_relation_id) {
                    descriptor = &candidate;
                    break;
                }
            if (descriptor == nullptr) return fail("FCONN binding metadata lacks descriptor");
            if (current.size != descriptor->output_coordinates || !current.bin_gates.empty() ||
                current.uni_gates.size() != checked_mul(static_cast<std::size_t>(current.size), 2, "FCONN binding metadata gate count"))
                return fail("FCONN binding metadata shape is invalid");
            for (std::size_t index = 0; index < current.size; ++index) {
                const uniGate& previous_gate = current.uni_gates[index * 2];
                const uniGate& input_gate = current.uni_gates[index * 2 + 1];
                const std::size_t padded_index = fconn_padded_output_index(
                    descriptor->rows, descriptor->cols, index);
                if (previous_gate.g != index || previous_gate.u != padded_index || previous_gate.lu != 1 || current.coefficient(previous_gate) != 1 || input_gate.g != index ||
                    input_gate.lu != 0 || current.coefficient(input_gate) != -1 || input_gate.u >= current.ori_id_u.size() ||
                    current.ori_id_u[input_gate.u] != descriptor->output_offset + index)
                    return fail("FCONN binding metadata coordinate is not canonical");
            }
        }
        if (current.max_bl_u != std::max(current.bit_length_u[0], current.bit_length_u[1])) return fail("layer max_bl_u is not derived canonically");
        const i8 expected_max_v = current.need_phase2 ? std::max(current.bit_length_v[0], current.bit_length_v[1]) : 0;
        if (current.max_bl_v != expected_max_v) return fail("layer max_bl_v is not derived canonically");
        if (current.size_u[0] != current.ori_id_u.size() || current.size_v[0] != current.ori_id_v.size())
            return fail("layer subset sizes do not match initSubset maps");
        if (current.bit_length_u[0] != ceilPow2BitLength(current.size_u[0]) || current.bit_length_v[0] != ceilPow2BitLength(current.size_v[0])) {
            std::ostringstream detail;
            detail << "layer subset bit lengths are not canonical at layer " << i << " u=" << static_cast<int>(current.bit_length_u[0]) << "/"
                   << static_cast<int>(ceilPow2BitLength(current.size_u[0])) << " v=" << static_cast<int>(current.bit_length_v[0]) << "/"
                   << static_cast<int>(ceilPow2BitLength(current.size_v[0]));
            return fail(detail.str());
        }
        for (const uniGate& gate : current.uni_gates)
            if (gate.lu == 0 && gate.u >= current.ori_id_u.size()) return fail("unary gate subset coordinate exceeds initSubset map");
        for (const binGate& gate : current.bin_gates) {
            if (gate.getLayerIdU(static_cast<u8>(i)) == 0 && gate.u >= current.ori_id_u.size())
                return fail("binary lhs subset coordinate exceeds initSubset map");
            if (gate.getLayerIdV(static_cast<u8>(i)) == 0 && gate.v >= current.ori_id_v.size())
                return fail("binary rhs subset coordinate exceeds initSubset map");
        }
    }
    if (fconn_index != plan.matmuls.size()) return fail("FCONN layer/MatMul descriptor count mismatch");
    return true;
}

bool validate_verifier_context_object(const VerifierContext& context, std::string* reason) {
    const auto fail = [reason](const std::string& message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    try {
        context.graph.validate();
        context.quant.validate();
        const StaticTables canonical_tables = StaticTables::build(context.quant);
        if (!static_tables_equal(context.tables, canonical_tables) || !context.tables.authenticate())
            return fail("static tables do not match independently regenerated canonical tables");
        if (!registry_equal(context.plan_ref().registry, RelationRegistry::canonical())) return fail("relation registry is not the canonical zkGPT registry");
        if (!validate_gat_circuit_plan(context.plan_ref(), reason)) return false;
        if (!context_layer_metadata_valid(context.plan_ref(), reason)) return false;
        if (context.public_features.rows != context.graph.num_nodes || context.public_features.cols != context.graph.num_features ||
            context.public_features.cols == 0 ||
            context.public_features.values.size() != checked_mul(context.public_features.rows, context.public_features.cols, "verifier context feature tensor"))
            return fail("public feature shape is invalid");
        if (context.public_logits.rows != context.graph.num_nodes || context.public_logits.cols != context.graph.num_classes ||
            context.public_logits.cols == 0 ||
            context.public_logits.values.size() != checked_mul(context.public_logits.rows, context.public_logits.cols, "verifier context logits tensor"))
            return fail("public logits shape is invalid");
        if (!registration_wire_valid(context.registration)) return fail("model registration wire is invalid");
        if (context.com_W.scheme != context.registration.com_W.scheme || context.com_W.basis_id != context.registration.com_W.basis_id ||
            context.com_W.logical_size != context.registration.com_W.logical_size || context.com_W.padded_size != context.registration.com_W.padded_size ||
            context.com_W.point_hex != context.registration.com_W.point_hex)
            return fail("context com_W is inconsistent with model registration");
        if (context.relation_id != "R_GAT-zkGPT" || (context.backend != "cpu" && context.backend != "gpu") ||
            context.model_arch_id != context.registration.model_arch_id || context.quant_cfg_id != context.quant.quant_cfg_id ||
            context.static_tables_id != context.tables.table_id || context.privacy_extension_profile_id != "masked_full" ||
            context.tables.quant_cfg_id != context.quant.quant_cfg_id || context.plan_ref().model_arch_id != context.registration.model_arch_id ||
            context.plan_ref().graph_num_nodes != context.graph.num_nodes || context.plan_ref().graph_num_features != context.graph.num_features ||
            context.plan_ref().graph_count != context.graph.graph_count || context.plan_ref().graph_num_classes != context.graph.num_classes ||
            context.plan_ref().graph_edge_count != context.graph.edges() || context.plan_ref().graph_topology_digest != canonical_graph_topology_digest(context.graph) ||
            context.plan_ref().quant_cfg_id != context.quant.quant_cfg_id || context.plan_ref().quant_digest != canonical_quant_digest(context.quant) ||
            context.plan_ref().static_tables_id != context.tables.table_id)
            return fail("verifier context identity is not canonical");
        return true;
    } catch (const std::exception& error) { return fail(error.what()); }
}

void set_reason(std::string* reason, const std::string& value) {
    if (reason != nullptr) *reason = value;
}

bool forbidden_payload(const std::string& value) {
    static const char* forbidden[] = {"WITNESS", "witness", "PRIVATE_TRACE", "private_trace", "raw_model", "raw_aux", "raw_lookup"};
    for (const char* needle : forbidden)
        if (value.find(needle) != std::string::npos) return true;
    return false;
}

bool is_lower_hex_digest(const std::string& value) {
    if (value.size() != 64) return false;
    for (char character : value) {
        if (!((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f'))) return false;
    }
    return true;
}

bool canonical_field_wire(const std::string& value) {
    if (value.empty() || value.size() > 128) return false;
    try {
        ensure_field();
        Field parsed;
        parsed.setStr(value, 16);
        return field_hex(parsed) == value;
    } catch (...) { return false; }
}

bool valid_native_kind(std::uint8_t kind) {
    return kind >= static_cast<std::uint8_t>(NativeProofMessageKind::Data) && kind <= static_cast<std::uint8_t>(NativeProofMessageKind::Challenge);
}

}

std::string canonical_digest64(const std::string& bytes) { return fixed_digest(hash_hex(bytes)); }

InputLayout build_gat_input_layout(const Graph& graph, const Model& model, const QuantConfig& quant) {
    graph.validate();
    return canonical_input_layout_impl(graph, model, quant);
}

GatCapacityReport inspect_gat_capacity(const Graph& graph, const Model& model, const QuantConfig& quant,
                                      std::uint64_t feature_source_file_bytes,
                                      bool symbolic_profile_only) {
    GatCapacityReport report;
    report.num_nodes = graph.num_nodes;
    report.num_edges = graph.edges();
    report.num_features = graph.num_features;
    report.num_classes = graph.num_classes;
    report.graph_count = graph.graph_count;
    const bool graph_count_extent_valid = graph.graph_count < std::numeric_limits<std::size_t>::max();
    report.ppi_node_ptr_boundary = graph_count_extent_valid && graph.node_ptr.size() == graph.graph_count + 1;
    report.ppi_edge_ptr_boundary = graph_count_extent_valid && graph.edge_ptr.size() == graph.graph_count + 1;
    report.explicit_matmul_gate_expansion = false;
    try {
        graph.validate();
        model.validate_schema();
        quant.validate();
        report.model_shape = canonical_model_shape(model);
        report.input_layout = canonical_input_layout_impl(graph, model, quant);
        const std::size_t nodes = graph.num_nodes;
        const std::size_t edges = graph.edges();
        const std::size_t h0 = model.tensor(hidden_tensor_name(0, 0, "seq/kernel")).shape.dims.at(2);
        const std::size_t h1 = model.tensor(hidden_tensor_name(1, 0, "seq/kernel")).shape.dims.at(2);
        const std::size_t classes = model.classes();
        const std::size_t hidden_heads = model.hidden_heads();
        const std::size_t output_heads = model.manifest.output_heads;
        const auto mul = [](std::size_t a, std::size_t b, const char* what) { return checked_mul(a, b, what); };
        const auto add = [](std::size_t a, std::size_t b, const char* what) { return checked_add(a, b, what); };
        const auto uadd = [](std::uint64_t a, std::uint64_t b, const char* what) {
            if (b > std::numeric_limits<std::uint64_t>::max() - a) throw std::overflow_error(std::string("capacity u64 sum overflow: ") + what);
            return a + b;
        };
        const auto umul = [](std::uint64_t a, std::uint64_t b, const char* what) {
            if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) throw std::overflow_error(std::string("capacity u64 product overflow: ") + what);
            return a * b;
        };
        const auto ubytes = [&umul](std::uint64_t count, std::size_t width, const char* what) {
            return umul(count, static_cast<std::uint64_t>(width), what);
        };
        const auto pow2 = [](std::uint64_t value) {
            if (value == 0) return std::uint64_t(1);
            std::uint64_t result = 1;
            while (result < value) {
                if (result > (std::numeric_limits<std::uint64_t>::max() >> 1)) throw std::overflow_error("capacity power-of-two overflow");
                result <<= 1;
            }
            return result;
        };
        const auto bits_for = [&pow2](std::uint64_t value) {
            const std::uint64_t domain = pow2(std::max<std::uint64_t>(1, value));
            std::uint64_t bits = 0;
            for (std::uint64_t cursor = domain; cursor > 1; cursor >>= 1) ++bits;
            return bits;
        };
        const auto u64_from_size = [](std::size_t value, const char* what) {
            if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
                if (value > std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error(std::string("capacity size exceeds u64: ") + what);
            }
            return static_cast<std::uint64_t>(value);
        };
        const auto tensor_shape = [&pow2, &bits_for](std::uint64_t logical) {
            const std::uint64_t active = pow2(std::max<std::uint64_t>(1, logical));
            const std::uint64_t columns = std::uint64_t(1) << (bits_for(active) / 2);
            return std::array<std::uint64_t, 3>{{active, active / columns, columns}};
        };
        std::size_t explicit_gates = 0;
        const std::size_t output_coordinates = mul(nodes, classes, "capacity output coordinates");
        const std::size_t fconn_output_coordinates =
            add(mul(hidden_heads, mul(nodes, add(h0, 2, "capacity hidden FCONN width"), "capacity hidden FCONN outputs"), "capacity hidden FCONN outputs"),
                add(mul(hidden_heads, mul(nodes, add(h1, 2, "capacity hidden FCONN width"), "capacity hidden FCONN outputs"), "capacity FCONN outputs"),
                    mul(output_heads, mul(nodes, add(classes, 2, "capacity output FCONN width"), "capacity output FCONN outputs"), "capacity FCONN outputs"),
                    "capacity FCONN outputs"),
                "capacity FCONN outputs");

        report.max_materialized_gate = 0;
        report.fconn_count = (2 * hidden_heads + output_heads) * 3;
        report.attention_relation_count = 2 * hidden_heads + output_heads;
        report.division_relation_count = 0;
        report.context_recipe_bytes_bound = 4ULL << 20;
        report.context_payload_bytes_bound = add(mul(64, edges, "capacity context graph bytes"),
                                                 add(mul(16, mul(nodes, graph.num_features, "capacity context features"), "capacity context feature bytes"),
                                                     mul(16, output_coordinates, "capacity context logits"), "capacity context payload bound"),
                                                 "capacity context payload bound");
        report.scratch_chunk_entries = static_cast<std::size_t>(kSparseGkrChunkEntries);
        report.native_sparse_relation_count = 3 * (2 * hidden_heads + output_heads) + 2;
        std::uint64_t sparse_edge_domain = 0;
        std::uint64_t sparse_node_domain = 0;
        std::uint64_t sparse_input_domain = 0;
        const std::uint64_t max_attention_width = static_cast<std::uint64_t>(std::max({h0, h1, classes}));
        if (!sparse_gkr_checked_mul(static_cast<std::uint64_t>(edges), max_attention_width, &sparse_edge_domain) ||
            !sparse_gkr_checked_mul(static_cast<std::uint64_t>(nodes), max_attention_width, &sparse_node_domain) ||
            !sparse_gkr_checked_mul(4, static_cast<std::uint64_t>(report.input_layout.n_in), &sparse_input_domain))
            throw std::overflow_error("sparse GKR u64 domain overflow");
        report.sparse_domain_entries = std::max(sparse_input_domain, std::max(sparse_edge_domain, sparse_node_domain));
        const std::uint64_t relation_count = static_cast<std::uint64_t>(report.native_sparse_relation_count);
        if (!sparse_gkr_checked_mul(static_cast<std::uint64_t>(kSparseGkrChunkEntries), std::max<std::uint64_t>(1, relation_count),
                                    &report.sparse_working_set_entries) ||
            !sparse_gkr_checked_mul(report.sparse_working_set_entries, static_cast<std::uint64_t>(sizeof(Fr)), &report.sparse_working_set_bytes))
            throw std::overflow_error("sparse GKR working-set overflow");

        const StaticTables native_tables = StaticTables::build(quant);
        const StaticLookupBounds native_lookup_bounds = static_lookup_bounds(native_tables);
        const QueryCounts query_counts = canonical_query_counts(graph, model, quant, native_lookup_bounds);
        const std::array<std::size_t, 4> query_counts_ordered{{query_counts.lrelu, query_counts.exp, query_counts.elu, query_counts.range}};
        const std::array<std::size_t, 4> table_sizes{{native_tables.lrelu.size(), native_tables.exp.size(), native_tables.elu.size(), 2}};
        const std::array<const char*, 4> family_ids{{"lrelu", "exp", "elu", "range"}};
        constexpr std::uint64_t kRowChunkEntries = 4096;
        constexpr std::uint64_t kG1HexChars = 64;
        constexpr std::uint64_t kTranscriptLabelBytesUpper = 128;
        constexpr std::uint64_t kSmallFamilyMessageBytes = 1024;
        const std::uint64_t n_in = u64_from_size(report.input_layout.n_in, "input domain");
        report.flattened_input_domain_entries = umul(4, n_in, "flattened input domain");
        report.virtual_input_domain = report.flattened_input_domain_entries;
        report.max_materialized_vector = 0;
        report.lasso_logical_domain_entries = 0;
        report.maximum_masked_response_entries = 0;

        std::uint64_t family_persistent_bytes = 0;
        std::uint64_t family_column_generator_bytes = 0;
        std::uint64_t family_row_commitment_bytes = 0;
        std::uint64_t lasso_heap_transient_peak_bytes = 0;
        std::uint64_t largest_family_transient_bytes = 0;
        std::uint64_t family_wire_bytes_sum = 0;
        std::uint64_t family_message_count_sum = 0;
        std::uint64_t maximum_family_response_entries = 0;
        std::uint64_t maximum_family_fr_buffer_bytes = 0;
        std::uint64_t maximum_family_g1_buffer_bytes = 0;
        std::uint64_t maximum_family_columns = 0;
        std::uint64_t maximum_family_field_entries = 0;
        const auto log2_pow2 = [&bits_for](std::uint64_t value) {
            const std::uint64_t bits = bits_for(value);
            if ((std::uint64_t(1) << bits) != value) throw std::invalid_argument("capacity logarithm input is not a power of two");
            return bits;
        };
        for (std::size_t family = 0; family < query_counts_ordered.size(); ++family) {
            const std::uint64_t real_queries = u64_from_size(query_counts_ordered[family], "Lasso query count");
            const std::uint64_t padded_queries = pow2(real_queries);
            report.lasso_real_query_entries = uadd(report.lasso_real_query_entries,
                real_queries, "total real Lasso query entries");
            report.lasso_padded_query_entries = uadd(report.lasso_padded_query_entries,
                padded_queries, "total padded Lasso query entries");
            report.lasso_family_zero_pad_entries = uadd(
                report.lasso_family_zero_pad_entries, padded_queries - real_queries,
                "total Lasso family zero padding");
            const std::uint64_t table_domain = pow2(u64_from_size(table_sizes[family], "Lasso table size"));
            const NativeLassoCapacityProfile workspace = native_lasso_capacity_profile(padded_queries, table_domain);
            const std::uint64_t logical_entries = workspace.full_domain;
            const std::uint64_t query_challenges = log2_pow2(padded_queries);
            const std::uint64_t row_chunks = workspace.row_commitment_chunk_count;
            const std::uint64_t matrix_bits = log2_pow2(workspace.columns);
            const std::uint64_t l_query_bits = bits_for(report.input_layout.segments.at(3).logical_length);
            std::uint64_t family_messages = 1;
            family_messages = uadd(family_messages, row_chunks, "Lasso family messages");
            family_messages = uadd(family_messages, workspace.booleanity_rounds, "Lasso family messages");
            family_messages = uadd(family_messages, query_challenges, "Lasso family messages");
            family_messages = uadd(family_messages, 4, "Lasso family messages");
            family_messages = uadd(family_messages, uadd(114, umul(139, workspace.booleanity_rounds, "Lasso Booleanity messages"), "Lasso Booleanity messages"), "Lasso family messages");
            family_messages = uadd(family_messages, uadd(72, umul(103, workspace.onehot_rounds, "Lasso onehot messages"), "Lasso onehot messages"), "Lasso family messages");
            family_messages = uadd(family_messages, uadd(86, umul(121, workspace.lookup_rounds, "Lasso lookup messages"), "Lasso lookup messages"), "Lasso family messages");
            family_messages = uadd(family_messages, umul(4, uadd(9, umul(2, matrix_bits, "Lasso matrix opening rounds"), "Lasso matrix opening messages"), "Lasso matrix opening messages"), "Lasso family messages");
            family_messages = uadd(family_messages, uadd(10, umul(2, l_query_bits / 2, "Lasso L_qry tensor rounds"), "Lasso L_qry tensor messages"), "Lasso family messages");

            const std::uint64_t row_payload_bytes = uadd(umul(8, row_chunks, "Lasso row chunk framing"), umul(workspace.rows, 8 + kG1HexChars, "Lasso row commitment points"), "Lasso row commitment wire bound");
            const std::uint64_t max_family_vector_entries = std::max({workspace.booleanity_rounds, workspace.onehot_rounds, workspace.lookup_rounds,
                                                                       query_challenges, matrix_bits, l_query_bits / 2, std::uint64_t(9)});
            maximum_family_field_entries = std::max(maximum_family_field_entries, max_family_vector_entries);
            const std::uint64_t field_vector_bytes = uadd(8, umul(max_family_vector_entries, 8 + 64, "Lasso field vector wire bound"), "Lasso field vector wire bound");
            const std::uint64_t non_chunk_messages = family_messages >= row_chunks ? family_messages - row_chunks : 0;
            const std::uint64_t row_message_frames = umul(row_chunks, 25 + kTranscriptLabelBytesUpper, "Lasso row-chunk message frames");
            std::uint64_t family_wire = uadd(row_payload_bytes,
                uadd(row_message_frames, umul(non_chunk_messages, kSmallFamilyMessageBytes, "Lasso small message wire bound"), "Lasso row and small-message frames"),
                "Lasso family wire bound");
            family_wire = uadd(family_wire, umul(4, field_vector_bytes, "Lasso field-vector wire bound"), "Lasso family wire bound");

            GatLassoWorkspaceProfile entry;
            entry.family_id = family_ids[family];
            entry.workspace = workspace;
            entry.logical_domain_entries = logical_entries;
            entry.row_commitment_chunk_count = row_chunks;
            entry.row_commitment_chunk_entries = kRowChunkEntries;
            entry.booleanity_rounds = workspace.booleanity_rounds;
            entry.a_query_challenges = query_challenges;
            entry.onehot_rounds = workspace.onehot_rounds;
            entry.lookup_rounds = workspace.lookup_rounds;
            entry.hyrax_matrix_masked_openings = workspace.hyrax_matrix_masked_openings;
            entry.hyrax_l_qry_tensor_openings = workspace.hyrax_l_qry_tensor_openings;
            entry.row_commitment_wire_bytes_upper_bound = row_payload_bytes;
            entry.family_message_count_upper_bound = family_messages;
            entry.family_wire_bytes_upper_bound = family_wire;
            report.lasso_workspace_profile.push_back(std::move(entry));
            report.lasso_total_rows = uadd(report.lasso_total_rows, workspace.rows,
                "total Lasso row commitments");
            report.lasso_total_row_commitment_chunks = uadd(report.lasso_total_row_commitment_chunks,
                row_chunks, "total Lasso row commitment chunks");
            report.hyrax_matrix_masked_openings = uadd(report.hyrax_matrix_masked_openings,
                workspace.hyrax_matrix_masked_openings, "total Hyrax matrix openings");
            report.hyrax_l_qry_tensor_openings = uadd(report.hyrax_l_qry_tensor_openings,
                workspace.hyrax_l_qry_tensor_openings, "total Hyrax L_qry tensor openings");
            report.lasso_family_domains.push_back({{workspace.rows, workspace.columns}});
            report.lasso_logical_domain_entries = uadd(report.lasso_logical_domain_entries, logical_entries, "Lasso logical domain entries");
            family_persistent_bytes = uadd(family_persistent_bytes, workspace.row_ledger_retained_bytes, "Lasso retained workspace");
            family_column_generator_bytes = uadd(family_column_generator_bytes, workspace.column_generator_bytes,
                "Lasso retained column generators");
            family_row_commitment_bytes = uadd(family_row_commitment_bytes, workspace.row_commitment_bytes,
                "Lasso simultaneous row commitments");
            const std::uint64_t family_row_fold_heap = uadd(workspace.row_fold_weights_bytes,
                uadd(workspace.row_fold_values_bytes, workspace.onehot_query_bytes, "Lasso row-fold heap workspace"),
                "Lasso row-fold heap workspace");
            lasso_heap_transient_peak_bytes = std::max(lasso_heap_transient_peak_bytes,
                std::max(family_row_fold_heap, workspace.onehot_heap_peak_bytes));
            largest_family_transient_bytes = std::max(largest_family_transient_bytes,
                workspace.family_peak_bytes > workspace.row_ledger_retained_bytes ? workspace.family_peak_bytes - workspace.row_ledger_retained_bytes : std::uint64_t(0));
            family_wire_bytes_sum = uadd(family_wire_bytes_sum, family_wire, "Lasso family wire total");
            family_message_count_sum = uadd(family_message_count_sum, family_messages, "Lasso family messages total");
            maximum_family_response_entries = std::max(maximum_family_response_entries, workspace.columns);
            maximum_family_columns = std::max(maximum_family_columns, workspace.columns);
            maximum_family_fr_buffer_bytes = std::max({maximum_family_fr_buffer_bytes, workspace.private_witness_bytes,
                workspace.row_blinding_bytes, workspace.row_fold_weights_bytes, workspace.row_fold_values_bytes,
                workspace.onehot_query_bytes, workspace.t2_column_one_bytes, workspace.onehot_heap_peak_bytes,
                workspace.masked_opening_coefficient_bytes, workspace.masked_opening_response_bytes});
            maximum_family_g1_buffer_bytes = std::max({maximum_family_g1_buffer_bytes, workspace.row_commitment_bytes,
                workspace.column_generator_bytes, workspace.masked_opening_generator_bytes});
        }
        const std::uint64_t lqry_logical_length = u64_from_size(
            report.input_layout.segments.at(3).logical_length, "L_qry tensor length");
        if (report.lasso_padded_query_entries > lqry_logical_length)
            throw std::logic_error("padded Lasso query blocks exceed canonical L_qry support");
        report.gkr_layer0_lqry_padded_block_entries_upper_bound =
            report.lasso_padded_query_entries;
        report.gkr_layer0_lqry_zero_gap_entries =
            lqry_logical_length - report.lasso_padded_query_entries;
        report.lasso_private_selected_row_workspace_peak_bytes = uadd(family_persistent_bytes,
            std::max(family_row_commitment_bytes, largest_family_transient_bytes), "Lasso private workspace peak");
        report.lasso_heap_workspace_peak_bytes = uadd(family_persistent_bytes,
            uadd(umul(2, family_column_generator_bytes, "Lasso public identity generator copies"),
                std::max(family_row_commitment_bytes, lasso_heap_transient_peak_bytes),
                "Lasso ordinary vector workspace peak"), "Lasso ordinary vector workspace peak");
        report.hyrax_active_domain_workspace_entries = maximum_family_columns;
        report.hyrax_active_domain_workspace_bytes = 0;
        report.hyrax_folded_columns_workspace_bytes = 0;
        for (const GatLassoWorkspaceProfile& family : report.lasso_workspace_profile) {
            report.hyrax_active_domain_workspace_bytes = std::max(report.hyrax_active_domain_workspace_bytes,
                uadd(family.workspace.masked_opening_generator_bytes,
                     uadd(family.workspace.masked_opening_coefficient_bytes, family.workspace.masked_opening_response_bytes, "Hyrax masked opening workspace"),
                     "Hyrax masked opening workspace"));
            report.hyrax_folded_columns_workspace_bytes = std::max(report.hyrax_folded_columns_workspace_bytes,
                uadd(family.workspace.row_fold_values_bytes, family.workspace.onehot_query_bytes, "Hyrax folded column workspace"));
        }

        const std::uint64_t aux_length = u64_from_size(report.input_layout.segments.at(2).logical_length, "Aux tensor length");
        const std::uint64_t l_query_length = u64_from_size(report.input_layout.segments.at(3).logical_length, "L_qry tensor length");
        const auto aux_tensor = tensor_shape(aux_length);
        const auto l_query_tensor = tensor_shape(l_query_length);
        report.aux_logical_entries_upper_bound = aux_length;
        report.aux_active_domain_entries_upper_bound = aux_tensor[0];
        report.hyrax_l_qry_tensor_columns = l_query_tensor[2];
        const auto tensor_open_peak = [&umul, &uadd](const std::array<std::uint64_t, 3>& shape) {
            const std::uint64_t row_retained_and_weights = umul(shape[1], sizeof(G1) + 2 * sizeof(Fr), "tensor row workspace");
            const std::uint64_t column_workspace = umul(shape[2], 2 * sizeof(G1) + 3 * sizeof(Fr), "tensor column workspace");
            return uadd(row_retained_and_weights, column_workspace, "tensor open workspace");
        };
        const auto tensor_retained = [&umul](const std::array<std::uint64_t, 3>& shape) {
            return umul(shape[1], sizeof(G1) + sizeof(Fr), "tensor retained row state");
        };
        const auto tensor_transient = [&umul, &uadd](const std::array<std::uint64_t, 3>& shape) {
            return uadd(umul(shape[1], sizeof(Fr), "tensor row weights"),
                umul(shape[2], 2 * sizeof(G1) + 3 * sizeof(Fr), "tensor column transient"), "tensor transient workspace");
        };
        report.hyrax_aux_tensor_workspace_bytes = tensor_open_peak(aux_tensor);
        report.hyrax_l_qry_tensor_workspace_bytes = tensor_open_peak(l_query_tensor);
        report.hyrax_tensor_retained_workspace_bytes = uadd(tensor_retained(aux_tensor), tensor_retained(l_query_tensor), "tensor retained live set");
        report.hyrax_tensor_opening_transient_peak_workspace_bytes = std::max(tensor_transient(aux_tensor), tensor_transient(l_query_tensor));
        report.hyrax_tensor_combined_peak_workspace_bytes = uadd(report.hyrax_tensor_retained_workspace_bytes,
            report.hyrax_tensor_opening_transient_peak_workspace_bytes, "tensor simultaneous opening peak");
        const std::uint64_t w_active = pow2(u64_from_size(report.input_layout.segments.at(1).logical_length, "W MLE length"));
        report.hyrax_w_mle_workspace_bytes = ubytes(w_active, sizeof(G1) + 2 * sizeof(Fr), "W masked MLE workspace");

        const std::uint64_t graph_nodes = u64_from_size(nodes, "graph nodes");
        const std::uint64_t graph_edges = u64_from_size(edges, "graph edges");
        const std::uint64_t h0_u = u64_from_size(h0, "hidden width 0");
        const std::uint64_t h1_u = u64_from_size(h1, "hidden width 1");
        const std::uint64_t classes_u = u64_from_size(classes, "class count");
        const std::uint64_t hidden_heads_u = u64_from_size(hidden_heads, "hidden head count");
        const std::uint64_t output_heads_u = u64_from_size(output_heads, "output head count");
        const std::uint64_t n0 = umul(graph_nodes, h0_u, "hidden layer 0 values");
        const std::uint64_t n1 = umul(graph_nodes, h1_u, "hidden layer 1 values");
        const std::uint64_t nc = umul(graph_nodes, classes_u, "output values");

        struct RelationCapacityCounts {
            std::uint64_t outputs = 0;
            std::uint64_t unary = 0;
            std::uint64_t binary = 0;
        };
        RelationCapacityCounts relation_counts;
        const auto add_relation_counts = [&uadd](RelationCapacityCounts& destination, std::uint64_t outputs,
                                                   std::uint64_t unary, std::uint64_t binary) {
            destination.outputs = uadd(destination.outputs, outputs, "relation output count");
            destination.unary = uadd(destination.unary, unary, "relation unary gate count");
            destination.binary = uadd(destination.binary, binary, "relation binary gate count");
        };
        const auto add_relation_range = [&umul, &uadd, &add_relation_counts](RelationCapacityCounts& destination,
                                                                            std::uint64_t count, std::uint64_t width) {
            const std::uint64_t rows_per_range = uadd(width, 1, "range residual count");
            const std::uint64_t unary_per_range = uadd(umul(2, width, "range unary gates"), 1, "range unary gates");
            add_relation_counts(destination, umul(count, rows_per_range, "range residual count"),
                umul(count, unary_per_range, "range unary gates"), umul(count, width, "range binary gates"));
        };
        const auto add_relation_lookup = [&umul, &uadd, &add_relation_counts](RelationCapacityCounts& destination,
                                                                              std::uint64_t count, std::uint64_t x_width,
                                                                              std::uint64_t y_width) {
            const std::uint64_t rows_per_lookup = uadd(5, uadd(umul(2, x_width, "lookup range residuals"),
                umul(2, y_width, "lookup range residuals"), "lookup range residuals"), "lookup residual count");
            const std::uint64_t unary_per_lookup = uadd(7, uadd(umul(4, x_width, "lookup unary gates"),
                umul(4, y_width, "lookup unary gates"), "lookup unary gates"), "lookup unary gates");
            const std::uint64_t binary_per_lookup = uadd(umul(2, x_width, "lookup binary gates"),
                umul(2, y_width, "lookup binary gates"), "lookup binary gates");
            add_relation_counts(destination, umul(count, rows_per_lookup, "lookup residual count"),
                umul(count, unary_per_lookup, "lookup unary gates"), umul(count, binary_per_lookup, "lookup binary gates"));
        };
        const auto add_relation_attention = [&umul, &uadd, &add_relation_counts, graph_nodes, graph_edges](
                                                RelationCapacityCounts& destination, std::uint64_t dim) {
            const std::uint64_t coordinates = umul(graph_nodes, dim, "attention relation coordinates");
            const std::uint64_t outputs = uadd(uadd(umul(6, graph_edges, "attention edge residuals"),
                umul(4, graph_nodes, "attention node residuals"), "attention residual count"),
                umul(3, coordinates, "attention coordinate residuals"), "attention residual count");
            const std::uint64_t unary = uadd(uadd(umul(10, graph_edges, "attention edge unary gates"),
                umul(3, graph_nodes, "attention node unary gates"), "attention unary gates"),
                umul(4, coordinates, "attention coordinate unary gates"), "attention unary gates");
            const std::uint64_t binary = uadd(uadd(umul(6, graph_edges, "attention edge binary gates"), graph_nodes,
                "attention binary gates"), uadd(umul(graph_edges, dim, "attention aggregation binary gates"), coordinates,
                "attention binary gates"), "attention binary gates");
            add_relation_counts(destination, outputs, unary, binary);
        };
        const auto add_relation_division = [&umul, &uadd, &add_relation_counts, &quant](
                                               RelationCapacityCounts& destination, std::uint64_t count, bool has_denominator) {
            const bool nearest = quant.rounding == "nearest_ties_away";
            const std::uint64_t rows_per_division = has_denominator ? 12 : 11;
            const std::uint64_t unary_nonempty = nearest ? 15 : 14;
            const std::uint64_t unary_empty = nearest ? 13 : 12;
            const std::uint64_t unary_per_division = has_denominator ? unary_nonempty : unary_empty;
            add_relation_counts(destination, umul(count, rows_per_division, "division residual count"),
                umul(count, unary_per_division, "division unary gate count"), umul(count, 10, "division binary gate count"));
        };
        const auto add_relation_division_ranges = [&umul, &uadd, &add_relation_counts](
                                                        RelationCapacityCounts& destination, std::uint64_t count,
                                                        const std::array<unsigned, 5>& profile) {
            std::uint64_t width_sum = 0;
            for (unsigned width : profile) width_sum = uadd(width_sum, width, "division range width sum");
            add_relation_counts(destination, umul(count, uadd(5, width_sum, "division range residuals"), "division range residuals"),
                umul(count, uadd(5, umul(2, width_sum, "division range unary gates"), "division range unary gates"),
                     "division range unary gates"), umul(count, width_sum, "division range binary gates"));
        };

        const auto& lrelu_bounds = native_lookup_bounds.lrelu;
        const auto& exp_bounds = native_lookup_bounds.exp;
        const auto& elu_bounds = native_lookup_bounds.elu;
        const std::uint64_t lrelu_x_width = nonnegative_bit_width(2 * static_cast<Wide>(lrelu_bounds.first));
        const std::uint64_t lrelu_y_width = nonnegative_bit_width(2 * static_cast<Wide>(lrelu_bounds.second));
        const std::uint64_t exp_x_width = nonnegative_bit_width(2 * static_cast<Wide>(exp_bounds.first));
        const std::uint64_t exp_y_width = nonnegative_bit_width(2 * static_cast<Wide>(exp_bounds.second));
        const std::uint64_t elu_x_width = nonnegative_bit_width(2 * static_cast<Wide>(elu_bounds.first));
        const std::uint64_t elu_y_width = nonnegative_bit_width(2 * static_cast<Wide>(elu_bounds.second));
        const std::uint64_t delta_width = nonnegative_bit_width(2 * static_cast<Wide>(lrelu_bounds.second));
        const auto division_profiles = canonical_division_range_bits(graph, model, quant, native_lookup_bounds);
        std::size_t relation_profile_index = 0;
        const auto add_attention_relation_group = [&](std::uint64_t dim, bool hidden,
                                                      const std::array<unsigned, 5>& profile) {
            const std::uint64_t coordinates = umul(graph_nodes, dim, "attention relation coordinates");
            add_relation_attention(relation_counts, dim);
            add_relation_lookup(relation_counts, graph_edges, lrelu_x_width, lrelu_y_width);
            add_relation_lookup(relation_counts, graph_edges, exp_x_width, exp_y_width);
            if (hidden) add_relation_lookup(relation_counts, coordinates, elu_x_width, elu_y_width);
            else add_relation_counts(relation_counts, coordinates, umul(2, coordinates, "output activation identity gates"), 0);
            add_relation_range(relation_counts, graph_edges, delta_width);
            add_relation_division_ranges(relation_counts, coordinates, profile);
            add_relation_division(relation_counts, coordinates, true);
        };

        add_relation_counts(relation_counts, nc, nc, 0); // public output bindings
        const std::uint64_t public_feature_entries = u64_from_size(report.input_layout.segments.front().logical_length,
            "public input binding entries");
        add_relation_counts(relation_counts, public_feature_entries, umul(2, public_feature_entries, "public input binding gates"), 0);
        for (std::uint64_t dim : {h0_u, h1_u}) {
            for (std::uint64_t head = 0; head < hidden_heads_u; ++head) {
                if (relation_profile_index >= division_profiles.size()) throw std::logic_error("hidden relation range profile is missing");
                add_attention_relation_group(dim, true, division_profiles[relation_profile_index++]);
            }
            const std::uint64_t concat_coordinates = umul(graph_nodes, umul(dim, hidden_heads_u, "hidden concat relation width"),
                "hidden concat relation coordinates");
            add_relation_counts(relation_counts, concat_coordinates, umul(2, concat_coordinates, "hidden concat relation gates"), 0);
        }
        for (std::uint64_t head = 0; head < output_heads_u; ++head) {
            if (relation_profile_index + 1 >= division_profiles.size()) throw std::logic_error("output relation range profile is missing");
            add_attention_relation_group(classes_u, false, division_profiles[relation_profile_index++]);
            add_relation_division_ranges(relation_counts, nc, division_profiles[relation_profile_index++]);
            const std::uint64_t alignment_unary_per_coordinate = quant.rounding == "nearest_ties_away" ? 19 : 18;
            add_relation_counts(relation_counts, umul(14, nc, "output binding residuals"),
                umul(alignment_unary_per_coordinate, nc, "output binding unary gates"), umul(10, nc, "output binding binary gates"));
        }
        if (relation_profile_index >= division_profiles.size()) throw std::logic_error("final relation range profile is missing");
        const std::uint64_t average_unary_per_coordinate = uadd(output_heads_u, quant.rounding == "nearest_ties_away" ? 15 : 14,
            "final output average unary gates");
        add_relation_counts(relation_counts, umul(13, nc, "final output average residuals"),
            umul(average_unary_per_coordinate, nc, "final output average unary gates"), umul(10, nc, "final output average binary gates"));
        add_relation_division_ranges(relation_counts, nc, division_profiles[relation_profile_index++]);
        if (relation_profile_index != division_profiles.size()) throw std::logic_error("canonical relation range profile coverage is incomplete");
        for (std::size_t family = 0; family < query_counts_ordered.size(); ++family) {
            const std::uint64_t real = u64_from_size(query_counts_ordered[family], "Lasso real query count");
            const std::uint64_t padded = pow2(real);
            if (padded > real) add_relation_counts(relation_counts, padded - real, padded - real, 0);
        }

        const std::uint64_t exact_relation_output_count = relation_counts.outputs;
        const std::uint64_t relation_residual_entries = exact_relation_output_count - u64_from_size(output_coordinates, "public relation output coordinates");
        report.relation_gate_unary_entries = relation_counts.unary;
        report.relation_gate_binary_entries = relation_counts.binary;
        report.relation_gate_output_entries = relation_counts.outputs;
        report.retained_relation_witness_entries = 0;
        report.retained_relation_witness_bytes_upper_bound = 0;
        // The final relation outputs are a borrowed Y||residual-target view.
        // They remain logical outputs for binding, but production stores no
        // terminal NativeFieldVector or residual field-vector copy.
        report.relation_public_output_entries = relation_counts.outputs;
        report.relation_public_output_bytes_upper_bound = 0;
        report.relation_residual_entries = relation_residual_entries;
        report.relation_residual_bytes_upper_bound = 0;
        GatCircuitPlan exact_shape_plan = build_plan_impl(graph, model, quant, nullptr, true, &report, false, symbolic_profile_only);
        std::vector<GatGkrLayerShapeProfile>& layer_shapes = exact_shape_plan.gkr_layer_shape_profile;
        if (layer_shapes.size() != exact_shape_plan.circuit.circuit.size() || layer_shapes.size() < 2 ||
            layer_shapes.back().size != relation_counts.outputs ||
            layer_shapes.back().unary_gates != relation_counts.unary ||
            layer_shapes.back().binary_gates != relation_counts.binary)
            throw std::logic_error("actual GKR shape visitor profile disagrees with canonical relation counts");
        std::uint64_t actual_unary_gates = 0, actual_binary_gates = 0, direct_unary_count = 0, direct_binary_count = 0;
        std::uint64_t layer_value_entries = 0, maximum_layer_values = 1, max_layer_outputs = 1;
        std::uint64_t max_source_refs = 1, maximum_unary_layer_gates = 0, maximum_binary_layer_gates = 0;
        std::uint64_t max_input_u_entries = 0, max_input_v_entries = 0, input_map_entries = 0, input_map_count = 0;
        std::uint64_t canonical_copy_entries = 0, max_canonical_copy_layer_entries = 1;
        std::uint64_t fconn_layers = 0;
        std::uint64_t fconn_output_bit_sum = 0, nonfconn_output_bit_sum = 0, gkr_sumcheck_rounds = 0;
        for (std::size_t index = 1; index < layer_shapes.size(); ++index) {
            GatGkrLayerShapeProfile& shape = layer_shapes[index];
            shape.source_domains_exact = !symbolic_profile_only ||
                shape.layer_type == static_cast<std::uint32_t>(layerType::FCONN);
            actual_unary_gates = uadd(actual_unary_gates, shape.unary_gates, "actual GKR unary gates");
            actual_binary_gates = uadd(actual_binary_gates, shape.binary_gates, "actual GKR binary gates");
            const std::uint64_t layer_source_refs = uadd(shape.unary_gates, umul(2, shape.binary_gates, "actual layer operand references"), "actual layer operand references");
            max_source_refs = std::max(max_source_refs, layer_source_refs);
            maximum_unary_layer_gates = std::max(maximum_unary_layer_gates, shape.unary_gates);
            maximum_binary_layer_gates = std::max(maximum_binary_layer_gates, shape.binary_gates);
            max_input_u_entries = std::max(max_input_u_entries, shape.size_u0);
            max_input_v_entries = std::max(max_input_v_entries, shape.size_v0);
            input_map_entries = uadd(input_map_entries, uadd(shape.size_u0, shape.size_v0, "actual input source maps"), "actual input source maps");
            input_map_count = uadd(input_map_count, static_cast<std::uint64_t>(shape.size_u0 != 0) + static_cast<std::uint64_t>(shape.size_v0 != 0), "actual input source map count");
            max_layer_outputs = std::max(max_layer_outputs, shape.size);
            if (index + 1 < layer_shapes.size()) {
                layer_value_entries = uadd(layer_value_entries, shape.size, "actual retained GKR layer values");
                maximum_layer_values = std::max(maximum_layer_values, shape.size);
            }
            if (shape.layer_type == static_cast<std::uint32_t>(layerType::FCONN)) {
                ++fconn_layers;
                const std::uint64_t inner_bits = bits_for(shape.native_inner);
                fconn_output_bit_sum = uadd(fconn_output_bit_sum, static_cast<std::uint64_t>(shape.bit_length), "actual FCONN output bits");
                gkr_sumcheck_rounds = uadd(gkr_sumcheck_rounds, inner_bits, "actual GKR Sumcheck rounds");
                report.fconn_scalar_multiply_terms = uadd(report.fconn_scalar_multiply_terms,
                    umul(umul(shape.native_rows, shape.native_inner, "FCONN row-inner terms"),
                         shape.native_cols, "FCONN scalar multiply terms"),
                    "FCONN scalar multiply terms");
                report.fconn_output_coordinates = uadd(report.fconn_output_coordinates,
                    umul(shape.native_rows, shape.native_cols, "FCONN output coordinates"),
                    "FCONN output coordinates");
            } else {
                if (shape.max_bl_u < 0 || (shape.need_phase2 && shape.max_bl_v < 0))
                    throw std::logic_error("actual GKR layer visitor omitted a required sumcheck domain");
                const std::uint64_t u_rounds = static_cast<std::uint64_t>(shape.max_bl_u);
                const std::uint64_t v_rounds = shape.need_phase2 ? static_cast<std::uint64_t>(shape.max_bl_v) : 0;
                nonfconn_output_bit_sum = uadd(nonfconn_output_bit_sum, static_cast<std::uint64_t>(shape.bit_length), "actual GKR output bits");
                gkr_sumcheck_rounds = uadd(gkr_sumcheck_rounds, uadd(u_rounds, v_rounds, "actual GKR layer rounds"), "actual GKR Sumcheck rounds");

                const auto fold_work = [&uadd, &umul](std::uint64_t extent, std::uint64_t rounds,
                                               std::uint64_t& first_pairs,
                                               std::uint64_t& followup_pairs,
                                               std::uint64_t& total_pairs,
                                               std::uint64_t& extent_visits,
                                               std::uint64_t& followup_extent_visits,
                                               std::uint64_t& first_adds,
                                               std::uint64_t& first_subs,
                                               std::uint64_t& first_muls,
                                               std::uint64_t& followup_adds,
                                               std::uint64_t& followup_subs,
                                               std::uint64_t& followup_muls) {
                    for (std::uint64_t round = 0; round < rounds; ++round) {
                        const std::uint64_t pairs = extent / 2U + extent % 2U;
                        const std::uint64_t workers = pairs == 0 ? 0 : pairs < (1U << 15U)
                            ? 1U : std::min<std::uint64_t>(32U, std::min<std::uint64_t>(pairs, 1U << 16U));
                        const std::uint64_t chunks = pairs / (1U << 16U) +
                            static_cast<std::uint64_t>(pairs % (1U << 16U) != 0);
                        const std::uint64_t accumulator_adds = uadd(
                            umul(3U, umul(workers, chunks, "GKR fold partial chunk additions"),
                                 "GKR fold partial chunk additions"),
                            umul(3U, workers, "GKR fold partial reduction additions"),
                            "GKR fold accumulator additions");
                        if (round == 0)
                            first_pairs = uadd(first_pairs, pairs, "first-round packed GKR fold pairs");
                        else {
                            followup_pairs = uadd(followup_pairs, pairs, "follow-up packed GKR fold pairs");
                            followup_extent_visits = uadd(followup_extent_visits, extent,
                                "follow-up packed GKR active extent visits");
                        }
                        total_pairs = uadd(total_pairs, pairs, "packed GKR fold pairs");
                        extent_visits = uadd(extent_visits, extent, "packed GKR active extent visits");
                        const std::uint64_t pair_subtractions = umul(2U, pairs,
                            "GKR fold linear coefficient subtractions");
                        const std::uint64_t pair_multiplications = umul(3U, pairs,
                            "GKR fold quadratic coefficient multiplications");
                        const std::uint64_t pair_additions = uadd(umul(4U, pairs,
                            "GKR fold polynomial additions"), accumulator_adds,
                            "GKR fold polynomial and partial additions");
                        if (round == 0) {
                            first_subs = uadd(first_subs, pair_subtractions,
                                "first-round GKR fold field subtractions");
                            first_muls = uadd(first_muls, pair_multiplications,
                                "first-round GKR fold field multiplications");
                            first_adds = uadd(first_adds, pair_additions,
                                "first-round GKR fold field additions");
                        } else {
                            const std::uint64_t affine_coordinate_ops = umul(2U, extent,
                                "follow-up GKR affine coordinate operations");
                            followup_subs = uadd(followup_subs, pair_subtractions,
                                "follow-up GKR fold field subtractions");
                            followup_muls = uadd(followup_muls,
                                uadd(pair_multiplications, affine_coordinate_ops,
                                     "follow-up GKR fold field multiplications"),
                                "follow-up GKR fold field multiplications");
                            followup_adds = uadd(followup_adds,
                                uadd(pair_additions, affine_coordinate_ops,
                                     "follow-up GKR fold field additions"),
                                "follow-up GKR fold field additions");
                        }
                        extent = pairs;
                    }
                };
                shape.phase1_rounds = u_rounds;
                fold_work(shape.size_u0, u_rounds, shape.phase1_first_round_pair_visits,
                          shape.phase1_followup_pair_visits, shape.phase1_total_pair_visits,
                          shape.phase1_extent_visits, shape.phase1_followup_extent_visits,
                          shape.phase1_first_round_field_adds_upper_bound,
                          shape.phase1_first_round_field_subs_upper_bound,
                          shape.phase1_first_round_field_muls_upper_bound,
                          shape.phase1_followup_field_adds_upper_bound,
                          shape.phase1_followup_field_subs_upper_bound,
                          shape.phase1_followup_field_muls_upper_bound);
                fold_work(shape.size_u1, u_rounds, shape.phase1_first_round_pair_visits,
                          shape.phase1_followup_pair_visits, shape.phase1_total_pair_visits,
                          shape.phase1_extent_visits, shape.phase1_followup_extent_visits,
                          shape.phase1_first_round_field_adds_upper_bound,
                          shape.phase1_first_round_field_subs_upper_bound,
                          shape.phase1_first_round_field_muls_upper_bound,
                          shape.phase1_followup_field_adds_upper_bound,
                          shape.phase1_followup_field_subs_upper_bound,
                          shape.phase1_followup_field_muls_upper_bound);
                const auto packed_arena_init_fields = [&uadd](std::uint64_t extent, const char* what) {
                    return std::max<std::uint64_t>(2U, uadd(extent, extent & 1U, what));
                };
                shape.phase1_init_field_writes = uadd(
                    packed_arena_init_fields(shape.size_u0, "GKR phase-one lane 0 initialization capacity"),
                    packed_arena_init_fields(shape.size_u1, "GKR phase-one lane 1 initialization capacity"),
                    "GKR phase-one packed scalar initializations");
                shape.phase2_rounds = v_rounds;
                fold_work(shape.size_v0, v_rounds, shape.phase2_first_round_pair_visits,
                          shape.phase2_followup_pair_visits, shape.phase2_total_pair_visits,
                          shape.phase2_extent_visits, shape.phase2_followup_extent_visits,
                          shape.phase2_first_round_field_adds_upper_bound,
                          shape.phase2_first_round_field_subs_upper_bound,
                          shape.phase2_first_round_field_muls_upper_bound,
                          shape.phase2_followup_field_adds_upper_bound,
                          shape.phase2_followup_field_subs_upper_bound,
                          shape.phase2_followup_field_muls_upper_bound);
                fold_work(shape.size_v1, v_rounds, shape.phase2_first_round_pair_visits,
                          shape.phase2_followup_pair_visits, shape.phase2_total_pair_visits,
                          shape.phase2_extent_visits, shape.phase2_followup_extent_visits,
                          shape.phase2_first_round_field_adds_upper_bound,
                          shape.phase2_first_round_field_subs_upper_bound,
                          shape.phase2_first_round_field_muls_upper_bound,
                          shape.phase2_followup_field_adds_upper_bound,
                          shape.phase2_followup_field_subs_upper_bound,
                          shape.phase2_followup_field_muls_upper_bound);
                shape.phase2_init_field_writes = uadd(
                    packed_arena_init_fields(shape.size_v0, "GKR phase-two lane 0 initialization capacity"),
                    packed_arena_init_fields(shape.size_v1, "GKR phase-two lane 1 initialization capacity"),
                    "GKR phase-two packed scalar initializations");
                report.gkr_phase1_rounds_upper_bound = uadd(report.gkr_phase1_rounds_upper_bound,
                    u_rounds, "GKR phase-one rounds");
                report.gkr_phase2_rounds_upper_bound = uadd(report.gkr_phase2_rounds_upper_bound,
                    v_rounds, "GKR phase-two rounds");
                report.gkr_phase1_first_round_pair_visits_upper_bound = uadd(
                    report.gkr_phase1_first_round_pair_visits_upper_bound,
                    shape.phase1_first_round_pair_visits, "GKR phase-one first-round pairs");
                report.gkr_phase1_followup_pair_visits_upper_bound = uadd(
                    report.gkr_phase1_followup_pair_visits_upper_bound,
                    shape.phase1_followup_pair_visits, "GKR phase-one follow-up pairs");
                report.gkr_phase1_total_pair_visits_upper_bound = uadd(
                    report.gkr_phase1_total_pair_visits_upper_bound,
                    shape.phase1_total_pair_visits, "GKR phase-one total fold pairs");
                report.gkr_phase1_extent_visits_upper_bound = uadd(
                    report.gkr_phase1_extent_visits_upper_bound,
                    shape.phase1_extent_visits, "GKR phase-one active extent visits");
                report.gkr_phase1_followup_extent_visits_upper_bound = uadd(
                    report.gkr_phase1_followup_extent_visits_upper_bound,
                    shape.phase1_followup_extent_visits, "GKR phase-one follow-up extent visits");
                report.gkr_phase1_first_round_field_adds_upper_bound = uadd(
                    report.gkr_phase1_first_round_field_adds_upper_bound,
                    shape.phase1_first_round_field_adds_upper_bound, "GKR phase-one first-round field additions");
                report.gkr_phase1_first_round_field_subs_upper_bound = uadd(
                    report.gkr_phase1_first_round_field_subs_upper_bound,
                    shape.phase1_first_round_field_subs_upper_bound, "GKR phase-one first-round field subtractions");
                report.gkr_phase1_first_round_field_muls_upper_bound = uadd(
                    report.gkr_phase1_first_round_field_muls_upper_bound,
                    shape.phase1_first_round_field_muls_upper_bound, "GKR phase-one first-round field multiplications");
                report.gkr_phase1_followup_field_adds_upper_bound = uadd(
                    report.gkr_phase1_followup_field_adds_upper_bound,
                    shape.phase1_followup_field_adds_upper_bound, "GKR phase-one follow-up field additions");
                report.gkr_phase1_followup_field_subs_upper_bound = uadd(
                    report.gkr_phase1_followup_field_subs_upper_bound,
                    shape.phase1_followup_field_subs_upper_bound, "GKR phase-one follow-up field subtractions");
                report.gkr_phase1_followup_field_muls_upper_bound = uadd(
                    report.gkr_phase1_followup_field_muls_upper_bound,
                    shape.phase1_followup_field_muls_upper_bound, "GKR phase-one follow-up field multiplications");
                report.gkr_phase1_init_field_writes = uadd(report.gkr_phase1_init_field_writes,
                    shape.phase1_init_field_writes, "GKR phase-one active scalar initializations");
                report.gkr_phase2_first_round_pair_visits_upper_bound = uadd(
                    report.gkr_phase2_first_round_pair_visits_upper_bound,
                    shape.phase2_first_round_pair_visits, "GKR phase-two first-round pairs");
                report.gkr_phase2_followup_pair_visits_upper_bound = uadd(
                    report.gkr_phase2_followup_pair_visits_upper_bound,
                    shape.phase2_followup_pair_visits, "GKR phase-two follow-up pairs");
                report.gkr_phase2_total_pair_visits_upper_bound = uadd(
                    report.gkr_phase2_total_pair_visits_upper_bound,
                    shape.phase2_total_pair_visits, "GKR phase-two total fold pairs");
                report.gkr_phase2_extent_visits_upper_bound = uadd(
                    report.gkr_phase2_extent_visits_upper_bound,
                    shape.phase2_extent_visits, "GKR phase-two active extent visits");
                report.gkr_phase2_followup_extent_visits_upper_bound = uadd(
                    report.gkr_phase2_followup_extent_visits_upper_bound,
                    shape.phase2_followup_extent_visits, "GKR phase-two follow-up extent visits");
                report.gkr_phase2_first_round_field_adds_upper_bound = uadd(
                    report.gkr_phase2_first_round_field_adds_upper_bound,
                    shape.phase2_first_round_field_adds_upper_bound, "GKR phase-two first-round field additions");
                report.gkr_phase2_first_round_field_subs_upper_bound = uadd(
                    report.gkr_phase2_first_round_field_subs_upper_bound,
                    shape.phase2_first_round_field_subs_upper_bound, "GKR phase-two first-round field subtractions");
                report.gkr_phase2_first_round_field_muls_upper_bound = uadd(
                    report.gkr_phase2_first_round_field_muls_upper_bound,
                    shape.phase2_first_round_field_muls_upper_bound, "GKR phase-two first-round field multiplications");
                report.gkr_phase2_followup_field_adds_upper_bound = uadd(
                    report.gkr_phase2_followup_field_adds_upper_bound,
                    shape.phase2_followup_field_adds_upper_bound, "GKR phase-two follow-up field additions");
                report.gkr_phase2_followup_field_subs_upper_bound = uadd(
                    report.gkr_phase2_followup_field_subs_upper_bound,
                    shape.phase2_followup_field_subs_upper_bound, "GKR phase-two follow-up field subtractions");
                report.gkr_phase2_followup_field_muls_upper_bound = uadd(
                    report.gkr_phase2_followup_field_muls_upper_bound,
                    shape.phase2_followup_field_muls_upper_bound, "GKR phase-two follow-up field multiplications");
                report.gkr_phase2_init_field_writes = uadd(report.gkr_phase2_init_field_writes,
                    shape.phase2_init_field_writes, "GKR phase-two active scalar initializations");
            }
        }
        if (actual_unary_gates < relation_counts.unary || actual_binary_gates < relation_counts.binary ||
            actual_unary_gates > std::numeric_limits<std::size_t>::max() || actual_binary_gates > std::numeric_limits<std::size_t>::max() ||
            relation_counts.outputs > std::numeric_limits<std::size_t>::max())
            throw std::length_error("actual GKR gate profile exceeds native size_t or relation bounds");
        direct_unary_count = actual_unary_gates - relation_counts.unary;
        direct_binary_count = actual_binary_gates - relation_counts.binary;
        const std::uint64_t exact_explicit_gate_count = uadd(actual_unary_gates, actual_binary_gates, "actual explicit gate count");
        if (exact_explicit_gate_count > std::numeric_limits<std::size_t>::max()) throw std::length_error("actual explicit gate count exceeds native size_t");
        explicit_gates = static_cast<std::size_t>(exact_explicit_gate_count);
        report.max_materialized_gate = explicit_gates;
        report.explicit_gate_count = explicit_gates;
        report.division_relation_count = static_cast<std::size_t>(relation_counts.outputs);
        report.retained_layer_witness_entries_upper_bound = layer_value_entries;
        report.retained_layer_witness_bytes_upper_bound = ubytes(layer_value_entries, sizeof(Fr), "retained GKR layer witnesses");
        report.evaluated_layer_witness_entries_upper_bound = maximum_layer_values;
        report.evaluated_layer_witness_bytes_upper_bound = ubytes(maximum_layer_values, sizeof(Fr), "streamed GKR witness validation layer");
        report.max_materialized_vector = static_cast<std::size_t>(std::min<std::uint64_t>(maximum_layer_values, std::numeric_limits<std::size_t>::max()));
        report.max_materialized_vector_bytes = static_cast<std::size_t>(std::min<std::uint64_t>(ubytes(maximum_layer_values, sizeof(Fr), "maximum layer vector bytes"), std::numeric_limits<std::size_t>::max()));
        report.materialized_fr = report.max_materialized_vector;
        report.max_materialized_fr_bytes = std::max(ubytes(maximum_layer_values, sizeof(Fr), "maximum materialized field vector"), maximum_family_fr_buffer_bytes);
        report.materialized_g1 = static_cast<std::size_t>(std::min<std::uint64_t>(maximum_family_columns, std::numeric_limits<std::size_t>::max()));
        report.max_materialized_g1_bytes = maximum_family_g1_buffer_bytes;
        report.gkr_point_shape_ready = !symbolic_profile_only;
        report.gkr_point_shape_exact = !symbolic_profile_only;
        report.gkr_point_domain_upper_bound_valid = exact_shape_plan.gkr_point_domain_upper_bound_valid;
        report.symbolic_analysis_profile_complete = symbolic_profile_only;
        report.gkr_point_shape_digest = symbolic_profile_only ? std::string{} : exact_shape_plan.gkr_point_shape_digest;
        report.gkr_point_shape_layer_count = static_cast<std::uint64_t>(layer_shapes.size());
        report.gkr_point_shape_fconn_count = fconn_layers;
        report.gkr_point_shape_bitmap_scratch_bytes = symbolic_profile_only ? 0 : exact_shape_plan.gkr_point_shape_bitmap_bytes;
        report.gkr_point_shape_touched_index_scratch_bytes_upper_bound = symbolic_profile_only
            ? 0 : exact_shape_plan.gkr_point_shape_touched_index_bytes_upper_bound;
        report.gkr_layer_count_upper_bound = static_cast<std::uint64_t>(layer_shapes.size());
        report.gkr_fconn_layer_count_upper_bound = fconn_layers;
        report.gkr_work_counts_exact = !symbolic_profile_only;
        report.gkr_layer_work_profile = layer_shapes;
        report.gkr_layer0_source_domain_entries_upper_bound = symbolic_profile_only
            ? exact_shape_plan.gkr_layer0_source_domain_upper_bound
            : std::max(max_input_u_entries, max_input_v_entries);
        bool previous_lane_active = true;
        std::size_t next_matmul = exact_shape_plan.matmuls.size();
        std::uint64_t gkr_input_source_count = 0;
        for (std::size_t index = layer_shapes.size(); index-- > 1;) {
            const GatGkrLayerShapeProfile& shape = layer_shapes[index];
            const bool fconn_residual_branch = index > 1 && next_matmul > 0 &&
                layer_shapes[index - 1].layer_type == static_cast<std::uint32_t>(layerType::FCONN);
            if (!previous_lane_active && !fconn_residual_branch) {
                if (shape.layer_type != static_cast<std::uint32_t>(layerType::ADD_BIAS) || shape.need_phase2 ||
                    shape.binary_gates != 0 || shape.unary_gates != shape.size)
                    throw std::logic_error("actual GKR inactive-lane branch is not a canonical unary copy layer at index " +
                        std::to_string(index) + " type=" + std::to_string(shape.layer_type) +
                        " lower_type=" + std::to_string(layer_shapes[index - 1].layer_type) +
                        " next_matmul=" + std::to_string(next_matmul) +
                        " size=" + std::to_string(shape.size) + " bits=" + std::to_string(shape.bit_length) +
                        " unary=" + std::to_string(shape.unary_gates) +
                        " binary=" + std::to_string(shape.binary_gates) +
                        " phase2=" + std::to_string(shape.need_phase2));
                gkr_input_source_count = uadd(gkr_input_source_count, 1, "actual canonical-copy input source count");
                canonical_copy_entries = uadd(canonical_copy_entries, shape.size, "actual canonical-copy input entries");
                max_canonical_copy_layer_entries = std::max(max_canonical_copy_layer_entries, shape.size);
            }
            if (shape.layer_type == static_cast<std::uint32_t>(layerType::FCONN)) {
                if (next_matmul == 0) throw std::logic_error("actual GKR source emission exceeds the MatMul descriptor count");
                --next_matmul;
                gkr_input_source_count = uadd(gkr_input_source_count, 2, "actual FCONN input source count");
                previous_lane_active = false;
            } else {
                gkr_input_source_count = uadd(gkr_input_source_count, static_cast<std::uint64_t>(shape.size_u0 != 0), "actual direct U input source count");
                if (shape.size_v0 != 0 && !shape.need_phase2) throw std::logic_error("actual GKR V input source has no phase-two lane");
                if (shape.need_phase2) gkr_input_source_count = uadd(gkr_input_source_count, static_cast<std::uint64_t>(shape.size_v0 != 0), "actual direct V input source count");
                previous_lane_active = true;
            }
        }
        if (next_matmul != 0)
            throw std::logic_error("actual GKR source emission did not consume every MatMul descriptor");
        report.gkr_input_source_count_upper_bound = gkr_input_source_count;
        std::uint64_t gkr_layer_messages = 0;
        for (std::size_t index = 1; index < layer_shapes.size(); ++index) {
            const GatGkrLayerShapeProfile& shape = layer_shapes[index];
            if (shape.layer_type == static_cast<std::uint32_t>(layerType::FCONN))
                gkr_layer_messages = uadd(gkr_layer_messages, uadd(114, umul(122, bits_for(shape.native_inner), "actual FCONN messages"), "actual FCONN messages"), "actual GKR layer messages");
            else if (shape.need_phase2)
                gkr_layer_messages = uadd(gkr_layer_messages, uadd(336, umul(121, uadd(static_cast<std::uint64_t>(shape.max_bl_u), static_cast<std::uint64_t>(shape.max_bl_v), "actual phase-two messages"), "actual phase-two messages"), "actual phase-two messages"), "actual GKR layer messages");
            else gkr_layer_messages = uadd(gkr_layer_messages, uadd(211, umul(121, static_cast<std::uint64_t>(shape.max_bl_u), "actual phase-one messages"), "actual phase-one messages"), "actual GKR layer messages");
        }

        const std::uint64_t unary_gate_entry_bytes = sizeof(uniGate);
        const std::uint64_t binary_gate_entry_bytes = sizeof(binGate);
        const std::uint64_t maximum_unary_gate_array_entries = maximum_unary_layer_gates;
        const std::uint64_t maximum_binary_gate_array_entries = maximum_binary_layer_gates;
        const std::uint64_t maximum_single_gate_array_bytes = std::max(ubytes(maximum_unary_gate_array_entries, unary_gate_entry_bytes, "maximum unary gate array"),
            ubytes(maximum_binary_gate_array_entries, binary_gate_entry_bytes, "maximum binary gate array"));
        report.max_single_gate_array_bytes_upper_bound = maximum_single_gate_array_bytes;
        // The production-wide ExplicitUnaryGate buffers exist only for the
        // hidden concatenation layers and final logits copy. Relation layers
        // and FCONN bindings write directly into their compact native arrays;
        // using maximum_unary_layer_gates here incorrectly attributed the
        // largest relation layer to a builder that never materializes it.
        const std::uint64_t maximum_explicit_unary_builder_entries = std::max({
            umul(n0, hidden_heads_u, "hidden concatenation explicit unary builder"),
            umul(n1, hidden_heads_u, "hidden concatenation explicit unary builder"),
            nc});
        const std::uint64_t explicit_unary_gate_buffer_bytes =
            ubytes(maximum_explicit_unary_builder_entries, sizeof(ExplicitUnaryGate),
                   "maximum production explicit unary gate builder buffer");
        const std::uint64_t explicit_unary_compact_coexist_bytes = ubytes(
            maximum_explicit_unary_builder_entries, sizeof(uniGate),
            "compact unary gate array coexisting with production explicit builder");
        report.explicit_unary_gate_buffer_entry_bytes =
            static_cast<std::uint64_t>(sizeof(ExplicitUnaryGate));
        report.explicit_unary_gate_buffer_entries_upper_bound =
            maximum_explicit_unary_builder_entries;
        report.max_explicit_unary_gate_buffer_bytes_upper_bound =
            explicit_unary_gate_buffer_bytes;
        report.explicit_unary_gate_buffer_compact_coexist_bytes_upper_bound =
            uadd(explicit_unary_gate_buffer_bytes, explicit_unary_compact_coexist_bytes,
                 "explicit unary raw/compact coexistence");
        report.max_materialized_gate_bytes = static_cast<std::size_t>(std::min<std::uint64_t>(maximum_single_gate_array_bytes, std::numeric_limits<std::size_t>::max()));
        // Each active phase borrows its round-zero value prefix and allocates
        // only the multiplier arenas. After each native challenge, affine
        // pairs compact to scalars in place and mapped tails are punched before
        // the half-sized value arena is allocated. Original mappings remain
        // reserved; mapped accounting includes active page and filesystem block
        // rounding, including a partial final block after the punched extent.
        const std::uint64_t prefix_heap_limit = static_cast<std::uint64_t>(zkgpt::native_storage::heap_limit);
        const std::uint64_t prefix_page_bytes = zkgpt::native_storage::page_size();
        const std::uint64_t prefix_allocation_unit_bytes =
            zkgpt::native_storage::filesystem_allocation_unit_bytes(zkgpt::native_storage::scratch_directory());
        report.gkr_prefix_filesystem_allocation_unit_bytes = prefix_allocation_unit_bytes;
        report.gkr_input_fold_cache_page_size_bytes = prefix_page_bytes;
        const auto prefix_round_to_allocation = [&uadd, &umul, prefix_allocation_unit_bytes](std::uint64_t bytes, const char* what) {
            if (bytes == 0U) return std::uint64_t(0);
            return umul(uadd(bytes, prefix_allocation_unit_bytes - 1U, what) / prefix_allocation_unit_bytes,
                        prefix_allocation_unit_bytes, what);
        };
        const auto prefix_phase = [&umul, &uadd, &ubytes, prefix_heap_limit, prefix_page_bytes,
                                   &prefix_round_to_allocation](std::uint64_t lane0,
                                                                                                 std::uint64_t lane1,
                                                                                                 std::uint64_t rounds) {
            const auto arena_capacity = [&uadd](std::uint64_t extent, const char* what) {
                return std::max<std::uint64_t>(2U,
                    uadd(extent, extent & 1U, what));
            };
            const std::uint64_t capacity0 = arena_capacity(lane0, "GKR lane 0 packed arena capacity");
            const std::uint64_t capacity1 = arena_capacity(lane1, "GKR lane 1 packed arena capacity");
            const std::uint64_t arena0_bytes = ubytes(capacity0, sizeof(Fr), "GKR lane 0 packed arena");
            const std::uint64_t arena1_bytes = ubytes(capacity1, sizeof(Fr), "GKR lane 1 packed arena");
            const std::uint64_t mapped0 = arena0_bytes > prefix_heap_limit
                ? prefix_round_to_allocation(arena0_bytes, "GKR lane 0 allocated arena blocks") : 0U;
            const std::uint64_t mapped1 = arena1_bytes > prefix_heap_limit
                ? prefix_round_to_allocation(arena1_bytes, "GKR lane 1 allocated arena blocks") : 0U;
            const std::uint64_t heap0 = arena0_bytes > prefix_heap_limit ? 0U : arena0_bytes;
            const std::uint64_t heap1 = arena1_bytes > prefix_heap_limit ? 0U : arena1_bytes;
            const std::uint64_t initial_mapped = uadd(mapped0, mapped1, "GKR round-zero multiplier arenas");
            const std::uint64_t initial_heap = uadd(heap0, heap1, "GKR round-zero multiplier heap");

            constexpr std::uint64_t kChunkPairs = 1U << 16U;
            const std::uint64_t fold0 = lane0 / 2U + lane0 % 2U;
            const std::uint64_t fold1 = lane1 / 2U + lane1 % 2U;
            const std::uint64_t largest_fold = std::max(fold0, fold1);
            const std::uint64_t scratch_pairs = std::min(largest_fold, kChunkPairs);
            const std::uint64_t workers = largest_fold < (1U << 15U)
                ? 1U : std::min<std::uint64_t>(32U, scratch_pairs);
            const std::uint64_t scratch_bytes = largest_fold == 0 ? 0 : uadd(
                ubytes(umul(2, scratch_pairs, "GKR bounded prefix scratch entries"), sizeof(linear_poly),
                       "GKR bounded prefix fold scratch arrays"),
                ubytes(workers, sizeof(quadratic_poly), "GKR prefix worker partials"),
                "GKR bounded prefix fold scratch and partials");
            const auto mapped_live_bound = [prefix_heap_limit, prefix_page_bytes,
                                             &uadd, &umul, &ubytes, &prefix_round_to_allocation](std::uint64_t capacity,
                                                                                                          std::uint64_t active,
                                                                                                          const char* what) {
                const std::uint64_t file_bytes = ubytes(capacity, sizeof(Fr), what);
                if (file_bytes <= prefix_heap_limit) return std::uint64_t(0);
                const std::uint64_t active_bytes = ubytes(active, sizeof(Fr), what);
                const std::uint64_t page_live = std::min(file_bytes, umul(
                    uadd(active_bytes, prefix_page_bytes - 1U, what) / prefix_page_bytes,
                    prefix_page_bytes, what));
                const std::uint64_t tail = file_bytes % prefix_page_bytes;
                const std::uint64_t blocks_live = prefix_round_to_allocation(page_live, what);
                const std::uint64_t blocks_tail = prefix_round_to_allocation(tail, what);
                const std::uint64_t maximum_file_blocks = prefix_round_to_allocation(file_bytes, what);
                return std::min(maximum_file_blocks, uadd(blocks_live, blocks_tail, what));
            };
            std::uint64_t mapped_peak = initial_mapped;
            std::uint64_t heap_peak = uadd(initial_heap, scratch_bytes, "GKR round-zero multiplier and fold scratch heap");
            std::uint64_t total_peak = uadd(mapped_peak, heap_peak, "GKR round-zero multiplier and fold workspace");
            if (rounds > 0U && (lane0 > 0U || lane1 > 0U)) {
                const std::uint64_t after0 = lane0 == 0U ? 0U : arena_capacity(fold0, "GKR lane 0 half-size value arena capacity");
                const std::uint64_t after1 = lane1 == 0U ? 0U : arena_capacity(fold1, "GKR lane 1 half-size value arena capacity");
                const std::uint64_t after0_bytes = ubytes(after0, sizeof(Fr), "GKR lane 0 half-size value arena");
                const std::uint64_t after1_bytes = ubytes(after1, sizeof(Fr), "GKR lane 1 half-size value arena");
                const std::uint64_t after_mapped = uadd(
                    mapped_live_bound(capacity0, fold0, "GKR lane 0 punched multiplier tail"),
                    mapped_live_bound(capacity1, fold1, "GKR lane 1 punched multiplier tail"),
                    "GKR compact multiplier physical blocks");
                const std::uint64_t after_mapped_values = uadd(
                    after0_bytes > prefix_heap_limit ? prefix_round_to_allocation(after0_bytes, "GKR lane 0 allocated value blocks") : 0U,
                    after1_bytes > prefix_heap_limit ? prefix_round_to_allocation(after1_bytes, "GKR lane 1 allocated value blocks") : 0U,
                    "GKR half-size mapped value arenas");
                const std::uint64_t after_heap_values = uadd(
                    after0_bytes > prefix_heap_limit ? 0U : after0_bytes,
                    after1_bytes > prefix_heap_limit ? 0U : after1_bytes,
                    "GKR half-size value heap");
                const std::uint64_t after_heap = uadd(initial_heap, after_heap_values,
                    "GKR retained multiplier and half-size value heap");
                const std::uint64_t later_heap = rounds > 1U
                    ? uadd(after_heap, scratch_bytes, "GKR later compact-prefix fold workspace") : after_heap;
                const std::uint64_t after_mapped_total = uadd(after_mapped, after_mapped_values,
                    "GKR first-challenge mapped prefix peak");
                const std::uint64_t after_total = uadd(after_mapped_total,
                    rounds > 1U ? later_heap : after_heap,
                    "GKR compact prefix live workspace");
                mapped_peak = std::max(mapped_peak, after_mapped_total);
                heap_peak = std::max(heap_peak, later_heap);
                total_peak = std::max(total_peak, after_total);
            }
            return std::array<std::uint64_t, 3>{total_peak, mapped_peak, heap_peak};
        };
        std::uint64_t max_prefix_phase_bytes = 0;
        std::uint64_t max_prefix_mapped_bytes = 0;
        std::uint64_t max_prefix_heap_bytes = 0;
        for (std::size_t index = 1; index < layer_shapes.size(); ++index) {
            const GatGkrLayerShapeProfile& shape = layer_shapes[index];
            if (shape.layer_type == static_cast<std::uint32_t>(layerType::FCONN)) continue;
            const auto phase1 = prefix_phase(shape.size_u0, shape.size_u1, static_cast<std::uint64_t>(std::max<i8>(0, shape.max_bl_u)));
            max_prefix_phase_bytes = std::max(max_prefix_phase_bytes, phase1[0]);
            max_prefix_mapped_bytes = std::max(max_prefix_mapped_bytes, phase1[1]);
            max_prefix_heap_bytes = std::max(max_prefix_heap_bytes, phase1[2]);
            if (shape.need_phase2) {
                const auto phase2 = prefix_phase(shape.size_v0, shape.size_v1, static_cast<std::uint64_t>(std::max<i8>(0, shape.max_bl_v)));
                max_prefix_phase_bytes = std::max(max_prefix_phase_bytes, phase2[0]);
                max_prefix_mapped_bytes = std::max(max_prefix_mapped_bytes, phase2[1]);
                max_prefix_heap_bytes = std::max(max_prefix_heap_bytes, phase2[2]);
            }
        }
        report.gkr_prefix_workspace_bytes_upper_bound = max_prefix_phase_bytes;

        const std::uint64_t gkr_layers = report.gkr_layer_count_upper_bound;
        const std::uint64_t input_domain = report.flattened_input_domain_entries;
        const std::uint64_t input_bits = bits_for(input_domain);
        if (report.input_layout.segments.size() != 4)
            throw std::logic_error("GKR input-fold cache requires four canonical segments");
        std::array<std::uint64_t, 4> input_segment_logical_lengths{};
        for (std::size_t segment = 0; segment < input_segment_logical_lengths.size(); ++segment)
            input_segment_logical_lengths[segment] = u64_from_size(
                report.input_layout.segments[segment].logical_length,
                "GKR input-fold cache logical segment length");
        const NativeMaskedGkrInputFoldCachePlan input_fold_cache_plan =
            native_masked_gkr_input_fold_cache_plan(
                u64_from_size(report.input_layout.n_in,
                              "GKR input-fold cache segment domain"),
                input_segment_logical_lengths,
                std::max(prefix_allocation_unit_bytes, prefix_page_bytes),
                prefix_heap_limit);
        report.gkr_input_fold_cache_late_rounds =
            input_fold_cache_plan.late_fold_rounds;
        report.gkr_input_fold_cache_scalar_bytes =
            input_fold_cache_plan.scalar_bytes;
        report.gkr_input_fold_cache_raw_bytes_upper_bound =
            input_fold_cache_plan.raw_bytes_upper_bound;
        report.gkr_input_fold_cache_mapped_bytes_upper_bound =
            input_fold_cache_plan.mapped_bytes_upper_bound;
        report.gkr_input_fold_cache_heap_bytes_upper_bound =
            input_fold_cache_plan.heap_bytes_upper_bound;
        const std::uint64_t input_fold_heap_bytes = uadd(
            native_masked_gkr_input_fold_storage_bytes(
                canonical_copy_entries,
                report.gkr_input_source_count_upper_bound,
                input_bits),
            ubytes(umul(4, canonical_copy_entries,
                        "canonical-copy source/index transition envelope"),
                   sizeof(std::uint64_t),
                   "canonical-copy source/index transition heap envelope"),
            "bounded input-fold source and metadata heap");
        const std::uint64_t output_bits = bits_for(layer_shapes.back().size);
        const std::uint64_t gkr_b_sum = uadd(fconn_output_bit_sum, nonfconn_output_bit_sum, "GKR layer output-bit sum");
        report.gkr_message_count_upper_bound = uadd(137,
            uadd(output_bits, uadd(report.gkr_input_source_count_upper_bound,
                uadd(umul(121, input_bits, "GKR input rounds"), uadd(gkr_layer_messages, gkr_b_sum, "GKR layer transcript"), "GKR transcript"), "GKR transcript"), "GKR transcript"), "GKR transcript");
        const std::uint64_t gkr_max_fields = std::max({std::uint64_t(9), uadd(report.gkr_input_source_count_upper_bound, 1, "GKR field vector width"), input_bits});
        const std::uint64_t gkr_aggregate_entries = uadd(report.gkr_input_source_count_upper_bound, 1, "GKR aggregate coefficient count");
        const std::uint64_t gkr_aggregate_relation_meta_payload = uadd(
            uadd(40 + (sizeof("zkgpt-native-pedersen-relation-v1") - 1) + 6 + 256 + 2 * 64, 8, "GKR relation metadata header"),
            umul(uadd(gkr_aggregate_entries, 1, "GKR aggregate relation points"), 8 + 64, "GKR aggregate relation points"),
            "GKR aggregate relation metadata");
        const std::uint64_t gkr_aggregate_coefficients_payload = uadd(8,
            umul(gkr_aggregate_entries, 8 + 64, "GKR aggregate coefficients"), "GKR aggregate coefficients");
        const std::uint64_t gkr_input_point_payload = uadd(8,
            umul(input_bits, 8 + 64, "GKR input point payload"), "GKR input point payload");

        // A native masked Sumcheck round has a source-derived 121-message shape:
        // 13 Pedersen openings, 11 hidden-linear relations, two three-field
        // polynomial vectors, and one challenge. Bound those message classes by
        // their actual fixed encodings. The only O(S) messages are two aggregate
        // linear-relation metadata/coefficients pairs; the only O(I) vector is
        // the final input point. The source ledger absorbs a fixed 64-byte digest.
        constexpr std::uint64_t kGkrLabelBytesUpper = 128;
        constexpr std::uint64_t kGkrMessageFrameBytes = 25 + kGkrLabelBytesUpper;
        constexpr std::uint64_t kGkrSmallMessageWireBytes = 1536;
        constexpr std::uint64_t kPedersenPointHexBytes = 64;
        const std::uint64_t pedersen_header_bytes = 40 + (sizeof("zkgpt-native-pedersen-relation-v1") - 1) + 256 + 2 * kPedersenPointHexBytes;
        const std::uint64_t opening_meta_payload = uadd(
            uadd(pedersen_header_bytes, sizeof("opening") - 1, "GKR opening header"),
            8 + kPedersenPointHexBytes, "GKR opening metadata");
        const auto opening_wire = [&uadd, &umul, kGkrMessageFrameBytes, opening_meta_payload, kPedersenPointHexBytes](std::uint64_t count) {
            const std::uint64_t one = uadd(umul(4, kGkrMessageFrameBytes, "GKR opening frame"),
                uadd(opening_meta_payload,
                    uadd(kPedersenPointHexBytes,
                        uadd(64, 8 + 2 * (8 + 64), "GKR opening response payload"), "GKR opening point and response"),
                    "GKR opening payload"), "GKR opening wire");
            return umul(count, one, "GKR opening round wire");
        };
        const auto linear_wire = [&uadd, &umul, pedersen_header_bytes, kGkrMessageFrameBytes, kPedersenPointHexBytes](std::uint64_t operands) {
            const std::uint64_t meta = uadd(
                uadd(pedersen_header_bytes, sizeof("linear") - 1, "GKR linear header"),
                uadd(8, umul(uadd(operands, 1, "GKR linear relation points"), 8 + kPedersenPointHexBytes, "GKR linear relation points"),
                     "GKR linear relation metadata"), "GKR linear relation metadata");
            const std::uint64_t coefficients = uadd(8, umul(operands, 8 + 64, "GKR linear coefficients"), "GKR linear coefficients");
            const std::uint64_t payloads = uadd(meta,
                uadd(coefficients,
                    uadd(64, uadd(kPedersenPointHexBytes, uadd(64, 64, "GKR linear scalar payloads"), "GKR linear point and scalar payloads"),
                         "GKR linear payloads"), "GKR linear payloads"), "GKR linear payloads");
            return uadd(umul(6, kGkrMessageFrameBytes, "GKR linear message framing"), payloads, "GKR linear relation wire");
        };
        // Opening and linear helpers above already include their response and
        // coefficient FieldVectors. The round has two further three-field
        // vectors (q-values and q-transcript), plus the Sumcheck challenge.
        const std::uint64_t round_tail_wire = uadd(
            uadd(kGkrMessageFrameBytes, 64, "GKR Sumcheck challenge wire"),
            umul(2, uadd(kGkrMessageFrameBytes, 8 + 3 * (8 + 64), "GKR q-vector wire"), "GKR q-vectors wire"),
            "GKR round tail wire");
        const std::uint64_t gkr_round_wire = uadd(opening_wire(13),
            uadd(linear_wire(9),
                uadd(umul(10, linear_wire(4), "GKR small linear relation round wire"),
                    round_tail_wire,
                    "GKR linear and field-vector round wire"), "GKR linear and opening round wire"), "GKR masked Sumcheck round wire");
        const std::uint64_t gkr_round_messages = umul(gkr_sumcheck_rounds, 121, "GKR Sumcheck round messages");
        if (gkr_round_messages > report.gkr_message_count_upper_bound)
            throw std::logic_error("GKR per-round message formula exceeds its protocol message bound");
        const std::uint64_t gkr_fixed_messages = report.gkr_message_count_upper_bound - gkr_round_messages;
        std::uint64_t gkr_wire = uadd(umul(gkr_sumcheck_rounds, gkr_round_wire, "GKR source-derived round wire"),
            umul(gkr_fixed_messages, kGkrSmallMessageWireBytes, "GKR fixed scalar/point message wire"), "GKR one-copy transcript wire");
        const auto excess_over_small_message = [kGkrMessageFrameBytes, kGkrSmallMessageWireBytes](std::uint64_t payload) {
            const std::uint64_t wire = kGkrMessageFrameBytes + payload;
            return wire > kGkrSmallMessageWireBytes ? wire - kGkrSmallMessageWireBytes : std::uint64_t(0);
        };
        gkr_wire = uadd(gkr_wire, umul(2, excess_over_small_message(gkr_aggregate_relation_meta_payload), "GKR aggregate relation metadata excess"),
            "GKR aggregate relation wire");
        gkr_wire = uadd(gkr_wire, umul(2, excess_over_small_message(gkr_aggregate_coefficients_payload), "GKR aggregate coefficient excess"),
            "GKR aggregate coefficient wire");
        gkr_wire = uadd(gkr_wire, excess_over_small_message(gkr_input_point_payload), "GKR input point wire");
        report.gkr_wire_bytes_upper_bound = gkr_wire;
        const std::uint64_t gkr_small_payload_upper = kGkrSmallMessageWireBytes - kGkrMessageFrameBytes;
        const std::uint64_t maximum_gkr_payload = std::max({gkr_small_payload_upper, gkr_aggregate_relation_meta_payload,
            gkr_aggregate_coefficients_payload, gkr_input_point_payload, std::uint64_t(64)});

        const auto tail_message_count = [&uadd, &umul, &bits_for](const std::array<std::uint64_t, 3>& shape) {
            const std::uint64_t a = bits_for(shape[0]);
            const std::uint64_t c = bits_for(shape[2]);
            return uadd(83, uadd(umul(122, a, "tensor tail rounds"), umul(2, c, "tensor tail columns"), "tensor tail messages"), "tensor tail messages");
        };
        std::uint64_t zero_intervals = 0;
        std::uint64_t query_support_cursor = 0;
        for (const GatLassoWorkspaceProfile& family : report.lasso_workspace_profile) {
            const std::uint64_t block = family.workspace.padded_queries;
            const std::uint64_t remainder = query_support_cursor % block;
            if (remainder != 0) {
                ++zero_intervals;
                query_support_cursor = uadd(query_support_cursor, block - remainder, "L_qry zero-interval alignment");
            }
            query_support_cursor = uadd(query_support_cursor, block, "L_qry zero-interval live support");
        }
        if (query_support_cursor > l_query_length)
            throw std::logic_error("capacity L_qry zero-interval support exceeds its logical segment");
        if (query_support_cursor < l_query_length) ++zero_intervals;
        report.l_query_zero_interval_count_upper_bound = zero_intervals;
        const std::uint64_t l_query_zero_messages_per_interval = uadd(82,
            uadd(umul(122, bits_for(l_query_tensor[0]), "L_qry zero-interval rounds"),
                umul(2, bits_for(l_query_tensor[2]), "L_qry zero-interval columns"), "L_qry zero-interval messages"),
            "L_qry zero-interval messages");
        const std::uint64_t l_query_zero_messages = uadd(1,
            umul(zero_intervals, l_query_zero_messages_per_interval, "L_qry zero-interval messages"),
            "L_qry zero-interval messages");
        const std::uint64_t input_open_messages = uadd(uadd(9, umul(2, bits_for(w_active), "W MLE opening messages"), "W MLE opening messages"),
            uadd(uadd(10, umul(2, bits_for(aux_tensor[2]), "Aux tensor opening messages"), "Aux tensor opening messages"),
                 uadd(10, umul(2, bits_for(l_query_tensor[2]), "L_qry tensor opening messages"), "L_qry tensor opening messages"), "input tensor openings"), "input opening messages");
        const std::uint64_t zero_tail_certificates = static_cast<std::uint64_t>(aux_length < aux_tensor[0]) +
            static_cast<std::uint64_t>(l_query_length < l_query_tensor[0]);
        report.zero_tail_certificate_count_upper_bound = zero_tail_certificates;
        report.l_query_zero_interval_message_count_upper_bound = l_query_zero_messages;
        const std::uint64_t tail_messages = uadd(
            zero_tail_certificates == 0 ? 0 : (aux_length < aux_tensor[0] ? tail_message_count(aux_tensor) : 0) +
                (l_query_length < l_query_tensor[0] ? tail_message_count(l_query_tensor) : 0),
            l_query_zero_messages, "tensor tail messages");
        const std::uint64_t fixed_protocol_messages = 256;
        report.proof_message_count_upper_bound = uadd(family_message_count_sum,
            uadd(report.gkr_message_count_upper_bound, uadd(tail_messages, uadd(input_open_messages, fixed_protocol_messages, "top-level transcript messages"), "protocol transcript messages"), "protocol transcript messages"), "proof transcript messages");
        report.maximum_transcript_field_entries_upper_bound = std::max(gkr_max_fields, maximum_family_field_entries);
        report.total_transcript_field_entries_upper_bound = umul(report.proof_message_count_upper_bound,
            report.maximum_transcript_field_entries_upper_bound, "total transcript field entries");
        report.maximum_masked_response_entries = maximum_family_response_entries;
        std::uint64_t max_row_chunk_payload = 0;
        for (const GatLassoWorkspaceProfile& family : report.lasso_workspace_profile) {
            const std::uint64_t chunk_count = family.row_commitment_chunk_count;
            const std::uint64_t rows = family.workspace.rows;
            const std::uint64_t row_chunk_payload = rows == 0 ? 0 : uadd(8, umul(std::min<std::uint64_t>(rows, kRowChunkEntries), 8 + kG1HexChars, "maximum Lasso row chunk payload"), "maximum Lasso row chunk payload");
            max_row_chunk_payload = std::max(max_row_chunk_payload, row_chunk_payload);
            (void)chunk_count;
        }
        const std::uint64_t maximum_family_payload = uadd(8,
            umul(maximum_family_field_entries, 8 + 64, "maximum Lasso field vector payload"), "maximum Lasso field vector payload");
        report.maximum_message_string_bytes_upper_bound = std::max({max_row_chunk_payload, maximum_gkr_payload,
            maximum_family_payload});
        std::uint64_t small_wire_messages = report.proof_message_count_upper_bound - family_message_count_sum - report.gkr_message_count_upper_bound;
        const std::uint64_t one_copy_transcript_wire = uadd(family_wire_bytes_sum,
            uadd(report.gkr_wire_bytes_upper_bound, umul(small_wire_messages, 2048, "non-family protocol wire"), "one-copy transcript wire"),
            "one-copy transcript wire");
        report.transcript_wire_bytes_upper_bound = one_copy_transcript_wire;
        const std::uint64_t transcript_framing_bytes = umul(report.proof_message_count_upper_bound, 25, "native transcript message framing");
        report.transcript_content_bytes_upper_bound = one_copy_transcript_wire > transcript_framing_bytes
            ? one_copy_transcript_wire - transcript_framing_bytes : one_copy_transcript_wire;
        const std::uint64_t structured_tensor_rows = uadd(aux_tensor[1], l_query_tensor[1], "structured Aux/L_qry commitment rows");
        const std::uint64_t structured_tensor_wire = umul(structured_tensor_rows, 8 + kG1HexChars,
            "structured Aux/L_qry commitment wire");
        report.proof_artifact_wire_bytes_upper_bound = uadd(umul(2, one_copy_transcript_wire, "proof sections and native transcript copies"),
            uadd(structured_tensor_wire, 1ULL << 20, "proof artifact structured payload and framing"), "proof artifact wire bound");
        report.artifact_disk_bytes_upper_bound = uadd(report.proof_artifact_wire_bytes_upper_bound,
            uadd(report.context_recipe_bytes_bound, uadd(report.context_payload_bytes_bound, 1ULL << 20, "artifact disk framing"), "artifact disk context"), "artifact disk bound");

        const std::uint64_t raw_tables = uadd(uadd(u64_from_size(native_tables.lrelu.size(), "lrelu table entries"),
            u64_from_size(native_tables.exp.size(), "exp table entries"), "Lasso table entries"),
            uadd(u64_from_size(native_tables.elu.size(), "elu table entries"), 2, "Lasso table entries"), "Lasso table entries");
        const std::uint64_t padded_tables = uadd(uadd(pow2(u64_from_size(native_tables.lrelu.size(), "lrelu table size")),
            pow2(u64_from_size(native_tables.exp.size(), "exp table size")), "padded Lasso table sizes"),
            uadd(pow2(u64_from_size(native_tables.elu.size(), "elu table size")), 2, "padded Lasso table sizes"), "padded Lasso table sizes");
        const std::uint64_t table_value_bytes = ubytes(padded_tables, sizeof(Fr), "Lasso table values");
        const std::uint64_t table_map_bytes = ubytes(raw_tables, 256, "Lasso table map nodes");
        const std::uint64_t query_block_count = uadd(umul(6, hidden_heads_u, "Lasso function query blocks"),
            umul(2, output_heads_u, "Lasso function query blocks"), "Lasso function query blocks");
        const std::uint64_t range_block_count = uadd(umul(10, hidden_heads_u, "Lasso range attention blocks"),
            uadd(umul(6, output_heads_u, "Lasso range output blocks"), 1, "Lasso range output blocks"), "Lasso range callback blocks");
        report.lasso_table_and_index_bytes_upper_bound = uadd(table_value_bytes,
            uadd(table_map_bytes, ubytes(uadd(query_block_count, range_block_count, "Lasso block metadata count"), 256, "Lasso callback block metadata"), "Lasso index and metadata bytes"), "Lasso table and index bytes");

        const std::uint64_t graph_bytes = uadd(umul(graph_edges, 32, "graph edge backing"),
            uadd(umul(graph_nodes, uadd(128, umul(u64_from_size(graph.num_features, "feature width"), 64, "graph feature backing"), "graph node backing"), "graph backing"),
                 ubytes(uadd(graph_nodes, uadd(u64_from_size(graph.graph_count, "graph count"), 1, "graph pointer count"), "graph pointer count"), sizeof(std::size_t), "graph pointer backing"), "graph backing"), "graph host bytes");
        std::uint64_t model_elements = 0;
        for (const auto& item : model.tensors) {
            const Tensor& tensor = item.second;
            std::uint64_t elements = 1;
            for (std::size_t dimension : tensor.shape.dims) elements = umul(elements, u64_from_size(dimension, "model tensor dimension"), "model parameter count");
            model_elements = uadd(model_elements, elements, "model parameter count");
        }
        const std::uint64_t model_bytes = ubytes(model_elements, 64, "model resident workspace");
        const std::uint64_t aux_span_count = uadd(6,
            uadd(umul(36, hidden_heads_u, "borrowed Aux hidden span count"), umul(23, output_heads_u, "borrowed Aux output span count"), "borrowed Aux span count"),
            "borrowed Aux span count");
        const std::uint64_t aux_chunk_count = aux_length / AuxValueProviderState::chunk_entries +
            (aux_length % AuxValueProviderState::chunk_entries != 0);
        // Bound the actual typed vectors in one canonical trace tree. Int is
        // int64, Field is BN254 Fr, and an unreserved DivisionWitness push_back
        // vector can retain up to twice its logical size. All other vectors
        // below are sized/reserved from their exact public graph/model shapes.
        const std::uint64_t attention_per_coordinate_bytes = uadd(
            uadd(umul(4, sizeof(Int), "attention Int coordinate vectors"),
                 umul(2, sizeof(Fr), "attention Field coordinate vectors"), "attention coordinate vectors"),
            umul(2, sizeof(DivisionWitness), "unreserved attention division capacity"),
            "typed attention coordinate bytes");
        const std::uint64_t attention_per_node_bytes = uadd(umul(4, sizeof(Int), "attention node Int vectors"),
            sizeof(Fr), "attention node bytes");
        const std::uint64_t attention_per_edge_bytes = uadd(umul(5, sizeof(Int), "attention edge Int vectors"),
            sizeof(Fr), "attention edge bytes");
        const auto attention_trace_bytes = [&umul, &uadd, &ubytes, &graph_nodes, &graph_edges,
                                             attention_per_coordinate_bytes,
                                             attention_per_node_bytes, attention_per_edge_bytes](std::uint64_t dim) {
            const std::uint64_t coordinates = umul(graph_nodes, dim, "typed attention coordinates");
            return uadd(ubytes(coordinates, static_cast<std::size_t>(attention_per_coordinate_bytes),
                               "typed attention coordinate backing"),
                uadd(ubytes(graph_nodes, static_cast<std::size_t>(attention_per_node_bytes),
                            "typed attention node backing"),
                     ubytes(graph_edges, static_cast<std::size_t>(attention_per_edge_bytes),
                            "typed attention edge backing"), "typed attention node/edge backing"),
                "typed attention trace backing");
        };
        const std::uint64_t hidden_attention_trace_bytes = umul(hidden_heads_u,
            uadd(attention_trace_bytes(h0_u), attention_trace_bytes(h1_u), "two hidden attention layers"),
            "hidden attention heads");
        const std::uint64_t output_attention_trace_bytes = umul(output_heads_u,
            attention_trace_bytes(classes_u), "output attention heads");
        const std::uint64_t trace_input_bytes = ubytes(umul(graph_nodes,
            u64_from_size(graph.num_features, "typed trace feature width"), "typed trace input coordinates"),
            sizeof(Int), "typed trace input backing");
        const std::uint64_t trace_hidden_concat_bytes = ubytes(umul(graph_nodes,
            umul(hidden_heads_u, uadd(h0_u, h1_u, "typed hidden concatenation widths"),
                 "typed hidden concatenation entries"), "typed hidden concatenation entries"),
            sizeof(Int), "typed hidden concatenation backing");
        const std::uint64_t per_output_head_trace_bytes = uadd(
            ubytes(umul(152, nc, "typed output per-head arrays"), 1, "typed output per-head arrays"),
            ubytes(classes_u, sizeof(Int), "typed output bias backing"), "typed output head backing");
        const std::uint64_t trace_output_head_bytes = umul(output_heads_u, per_output_head_trace_bytes,
            "typed output-head trace backing");
        const std::uint64_t trace_global_output_bytes = ubytes(umul(152, nc,
            "typed global output arrays and reserved division witnesses"), 1,
            "typed global output arrays and reserved division witnesses");
        // run_forward and validate_relation each retain a local current tensor.
        // During assignment the old layer and new layer buffers may overlap;
        // charge input plus both hidden concatenation widths for each tree.
        const std::uint64_t trace_current_copy_bytes = ubytes(umul(graph_nodes,
            uadd(u64_from_size(graph.num_features, "current input width"),
                 umul(hidden_heads_u, uadd(h0_u, h1_u, "current hidden widths"),
                      "current hidden widths"), "current tensor source/destination envelope"),
            "current tensor source/destination entries"), sizeof(Int),
            "current tensor source/destination backing");
        const std::uint64_t one_typed_trace_tree_bytes = uadd(hidden_attention_trace_bytes,
            uadd(output_attention_trace_bytes,
                uadd(trace_input_bytes,
                    uadd(trace_hidden_concat_bytes,
                        uadd(trace_output_head_bytes,
                            uadd(trace_global_output_bytes, trace_current_copy_bytes,
                                 "typed trace global outputs and current tensor"),
                            "typed trace output heads and global outputs"),
                        "typed trace hidden concatenations and outputs"),
                    "typed trace input and concatenations"),
                "typed trace attention and owned vectors"), "one typed canonical trace tree");
        const std::uint64_t simultaneous_forward_trace_backing = umul(2, one_typed_trace_tree_bytes,
            "simultaneous forward and relation-validation typed trace trees");
        const std::uint64_t borrowed_aux_metadata = uadd(
            ubytes(aux_span_count, sizeof(AuxValueSpan), "borrowed Aux spans"),
            ubytes(aux_chunk_count, sizeof(std::size_t), "borrowed Aux chunk map"), "borrowed Aux provider metadata");
        // compute_attention builds one incoming-range vector at a time while
        // both trees are live. make_bound_family scans trace arrays in place.
        const std::uint64_t forward_helper_bytes = ubytes(graph_nodes, 64, "forward range/helper vector workspace");
        // A second StaticTables instance is built by validate_relation. The
        // existing Lasso table/index bound is larger than two native raw-table
        // copies and therefore safely covers that duplicate.
        // The proof retains one trace tree. The forward self-validation peak
        // below separately accounts for both trees and its duplicate tables.
        report.borrowed_trace_bytes_upper_bound = uadd(one_typed_trace_tree_bytes,
            uadd(borrowed_aux_metadata, uadd(forward_helper_bytes, 1ULL << 20,
                "proof trace helper metadata allowance"), "proof trace provider/helper metadata"),
            "one typed borrowed trace and provider state");
        const std::uint64_t forward_relation_validation_peak = uadd(graph_bytes,
            uadd(model_bytes,
                uadd(simultaneous_forward_trace_backing,
                    uadd(report.lasso_table_and_index_bytes_upper_bound,
                        uadd(borrowed_aux_metadata,
                            uadd(forward_helper_bytes, 1ULL << 20,
                                 "dual-trace forward helper metadata allowance"),
                            "dual-trace provider and helper metadata"),
                        "dual-trace table/index and provider state"),
                    "dual-trace relation validation and tables"),
                "dual-trace validation model"), "dual-trace validation graph");
        const std::uint64_t heap_limit = static_cast<std::uint64_t>(zkgpt::native_storage::heap_limit);
        const auto mapped_allocation = [heap_limit](std::uint64_t bytes) { return bytes > heap_limit ? bytes : std::uint64_t(0); };
        const auto mapped_growth = [&mapped_allocation, &umul, &ubytes](std::uint64_t count, std::size_t width, const char* what) {
            return mapped_allocation(umul(2, ubytes(count, width, what), what));
        };
        const auto mapped_reserved_collection = [&umul, heap_limit](std::uint64_t logical_bytes,
                                                                    std::uint64_t max_allocation,
                                                                    std::uint64_t allocation_count,
                                                                    const char* what) {
            if (logical_bytes == 0 || max_allocation == 0 || allocation_count == 0 || max_allocation <= heap_limit) return std::uint64_t(0);
            const std::uint64_t mapped_count = std::min(allocation_count, logical_bytes / (heap_limit + 1));
            if (mapped_count == 0) return std::uint64_t(0);
            return std::min(logical_bytes, umul(mapped_count, max_allocation, what));
        };

        // The allocator sends each individual buffer <=8 MiB to aligned heap
        // storage and each larger buffer to its private mmap-backed scratch file.
        // Bound mapped bytes by buffer class, while host heap uses a conservative
        // live NativeVector count times the per-allocation threshold.
        const std::uint64_t layer_array_count = std::max<std::uint64_t>(1, gkr_layers);
        const std::uint64_t direct_unary_gate_array_bytes = ubytes(maximum_unary_layer_gates, unary_gate_entry_bytes, "maximum direct unary gate array");
        const std::uint64_t direct_binary_gate_array_bytes = ubytes(maximum_binary_layer_gates, binary_gate_entry_bytes, "maximum direct binary gate array");
        const std::size_t coefficient_candidates = relation_coefficient_candidate_count(quant, native_lookup_bounds);
        if (coefficient_candidates > std::numeric_limits<std::size_t>::max() / 2)
            throw std::length_error("native coefficient-pool capacity exceeds native size_t");
        // Context decoding may grow its small per-layer pool incrementally;
        // charge twice the finite coefficient-candidate set to cover vector
        // growth, while direct plan builders reserve their known profile.
        const std::uint64_t coefficient_pool_per_layer_bytes = layer::coefficient_storage_bytes_for(coefficient_candidates * 2);
        const std::uint64_t coefficient_pools_heap_bytes = umul(layer_array_count, coefficient_pool_per_layer_bytes,
            "per-layer native coefficient pools and lookup maps");
        const std::uint64_t coefficient_pools_mapped_bytes = coefficient_pool_per_layer_bytes > heap_limit
            ? mapped_allocation(coefficient_pools_heap_bytes)
            : 0;
        const bool wide_coefficient_ids = coefficient_candidates > 256U;
        const std::uint64_t wide_id_entry_bytes = sizeof(u32);
        const std::uint64_t mapped_direct_gate_arrays = uadd(
            mapped_reserved_collection(ubytes(direct_unary_count, unary_gate_entry_bytes, "direct unary gate arrays"), direct_unary_gate_array_bytes,
                layer_array_count, "mapped direct unary gate arrays"),
            mapped_reserved_collection(ubytes(direct_binary_count, binary_gate_entry_bytes, "direct binary gate arrays"), direct_binary_gate_array_bytes,
                layer_array_count, "mapped direct binary gate arrays"), "mapped direct GKR gate arrays");
        const std::uint64_t mapped_direct_wide_ids = wide_coefficient_ids ? uadd(
            mapped_reserved_collection(ubytes(direct_unary_count, wide_id_entry_bytes, "direct wide unary coefficient IDs"),
                ubytes(maximum_unary_layer_gates, wide_id_entry_bytes, "maximum direct wide unary coefficient IDs"),
                layer_array_count, "mapped direct wide unary coefficient IDs"),
            mapped_reserved_collection(ubytes(direct_binary_count, wide_id_entry_bytes, "direct wide binary coefficient IDs"),
                ubytes(maximum_binary_layer_gates, wide_id_entry_bytes, "maximum direct wide binary coefficient IDs"),
                layer_array_count, "mapped direct wide binary coefficient IDs"),
            "mapped direct wide coefficient IDs") : 0;
        const std::uint64_t mapped_relation_gate_arrays = uadd(
            mapped_allocation(ubytes(relation_counts.unary, unary_gate_entry_bytes, "reserved relation unary gates")),
            mapped_allocation(ubytes(relation_counts.binary, binary_gate_entry_bytes, "reserved relation binary gates")),
            "mapped reserved relation gate arrays");
        const std::uint64_t mapped_relation_wide_ids = wide_coefficient_ids ? uadd(
            mapped_allocation(ubytes(relation_counts.unary, wide_id_entry_bytes, "reserved wide relation unary coefficient IDs")),
            mapped_allocation(ubytes(relation_counts.binary, wide_id_entry_bytes, "reserved wide relation binary coefficient IDs")),
            "mapped reserved relation wide coefficient IDs") : 0;
        const std::uint64_t mapped_gate_arrays = uadd(
            uadd(uadd(mapped_direct_gate_arrays, mapped_relation_gate_arrays, "mapped GKR gate arrays"),
                uadd(mapped_direct_wide_ids, mapped_relation_wide_ids, "mapped wide coefficient sidecars"),
                "mapped GKR gate arrays and sidecars"),
            coefficient_pools_mapped_bytes, "mapped GKR coefficient pools and gates");
        const std::uint64_t origin_map_reference_bytes = ubytes(input_map_entries,
            sizeof(PackedU40), "actual GKR input source-coordinate maps");
        const std::uint64_t origin_map_count = input_map_count;
        std::uint64_t origin_map_domain = 0;
        for (const InputSegment& segment : report.input_layout.segments)
            origin_map_domain = uadd(origin_map_domain, u64_from_size(segment.logical_length, "logical source support segment"),
                                     "logical source support entries");
        if (origin_map_domain > report.flattened_input_domain_entries)
            throw std::logic_error("canonical logical source support exceeds its actual layer-0 domain");
        report.gkr_subset_source_support_entries_upper_bound = origin_map_domain;
        // initSubset now builds each layer-0 source map from one packed bitmap,
        // emits the exact sorted unique coordinates, and reserves that exact
        // count. Bound each lane by both its source references and the actual
        // source layer-0 domain; bound all retained map vectors by the same
        // source-derived domain for every possible map lane.
        const std::uint64_t max_unique_map_entries = std::max(max_input_u_entries, max_input_v_entries);
        if (max_unique_map_entries > origin_map_domain ||
            input_map_entries > umul(origin_map_count, origin_map_domain, "actual GKR input map domain bound"))
            throw std::logic_error("actual GKR input source maps exceed their logical input support");
        const std::uint64_t origin_map_max_bytes = ubytes(max_unique_map_entries, sizeof(PackedU40), "maximum exact unique source-coordinate map");
        const std::uint64_t origin_map_logical_bytes = origin_map_reference_bytes;
        const std::uint64_t mapped_origin_maps = mapped_reserved_collection(origin_map_logical_bytes, origin_map_max_bytes,
            origin_map_count, "mapped exact unique source-coordinate maps");
        const std::uint64_t builder_target_bytes = ubytes(relation_residual_entries, sizeof(u8), "reserved packed relation residual target IDs");
        const std::uint64_t builder_target_dictionary_heap_bytes = PackedResidualTargets::dictionary_heap_upper_bound();
        const std::uint64_t builder_unary_bytes = ubytes(relation_counts.unary, sizeof(uniGate), "reserved relation unary gates");
        const std::uint64_t builder_binary_bytes = ubytes(relation_counts.binary, sizeof(binGate), "reserved relation binary gates");
        const std::uint64_t builder_wide_unary_bytes = wide_coefficient_ids
            ? ubytes(relation_counts.unary, wide_id_entry_bytes, "reserved wide relation unary coefficient IDs") : 0;
        const std::uint64_t builder_wide_binary_bytes = wide_coefficient_ids
            ? ubytes(relation_counts.binary, wide_id_entry_bytes, "reserved wide relation binary coefficient IDs") : 0;
        const std::uint64_t builder_target_capacity_bytes = builder_target_bytes;
        const std::uint64_t builder_unary_capacity_bytes = builder_unary_bytes;
        const std::uint64_t builder_binary_capacity_bytes = builder_binary_bytes;
        const std::uint64_t builder_max_bytes = std::max({builder_target_capacity_bytes, builder_unary_capacity_bytes,
            builder_binary_capacity_bytes, builder_wide_unary_bytes, builder_wide_binary_bytes});
        report.max_single_relation_builder_array_bytes_upper_bound = builder_max_bytes;
        const std::uint64_t mapped_gate_builder = uadd(
            mapped_allocation(builder_target_capacity_bytes),
            uadd(uadd(mapped_allocation(builder_unary_capacity_bytes), mapped_allocation(builder_binary_capacity_bytes),
                    "mapped relation builder gate arrays"),
                uadd(mapped_allocation(builder_wide_unary_bytes), mapped_allocation(builder_wide_binary_bytes),
                    "mapped relation builder wide coefficient IDs"),
                "mapped relation builder gates and sidecars"), "mapped relation builder reserved arrays");
        const std::uint64_t mapped_residual_targets = mapped_allocation(builder_target_capacity_bytes);
        const std::uint64_t subset_u_entries = max_input_u_entries;
        const std::uint64_t subset_v_entries = max_input_v_entries;
        const std::uint64_t subset_u_bytes = ubytes(subset_u_entries, sizeof(PackedU40), "GKR U exact unique source map capacity");
        const std::uint64_t subset_v_bytes = ubytes(subset_v_entries, sizeof(PackedU40), "GKR V exact unique source map capacity");
        report.gkr_subset_source_map_bytes_upper_bound = std::max(subset_u_bytes, subset_v_bytes);
        const std::uint64_t bitmap_word_count = origin_map_domain / 64 + (origin_map_domain % 64 != 0);
        const std::uint64_t subset_bitmap_bytes = ubytes(bitmap_word_count, sizeof(std::uint64_t), "one packed source-map bitmap");
        const std::uint64_t subset_rank_bytes = ubytes(bitmap_word_count, sizeof(std::uint64_t), "one packed source-map rank prefix");
        const std::uint64_t mapped_subset_bitmap = mapped_allocation(subset_bitmap_bytes);
        const std::uint64_t mapped_subset_rank = mapped_allocation(subset_rank_bytes);
        const std::uint64_t mapped_subset_scratch_peak = uadd(mapped_subset_bitmap, mapped_subset_rank,
            "mapped source-map bitmap and rank prefixes");
        report.gkr_subset_sort_scratch_bytes_upper_bound = 0;
        report.gkr_subset_bitmap_scratch_bytes_upper_bound = subset_bitmap_bytes;
        report.gkr_subset_rank_scratch_bytes_upper_bound = subset_rank_bytes;
        const std::uint64_t gate_builder_mapped_peak = uadd(
            uadd(uadd(mapped_direct_gate_arrays, mapped_direct_wide_ids, "gate-builder direct gates and coefficient IDs"),
                coefficient_pools_mapped_bytes, "gate-builder coefficient pools"),
            mapped_gate_builder, "gate-builder mapped buffers before relation-array transfer");
        // append_explicit_layer retains the raw ExplicitUnaryGate vector while
        // it allocates/fills the compact uniGate array. Bound that distinct
        // plan-construction live phase from the exact maximum canonical unary
        // layer; direct compact arrays above already include that layer.
        const std::uint64_t mapped_explicit_unary_gate_buffer =
            mapped_allocation(explicit_unary_gate_buffer_bytes);
        const std::uint64_t explicit_unary_plan_mapped_peak = uadd(
            uadd(uadd(mapped_direct_gate_arrays, mapped_direct_wide_ids,
                       "explicit unary plan direct gates and coefficient IDs"),
                 coefficient_pools_mapped_bytes,
                 "explicit unary plan coefficient pools"),
            mapped_explicit_unary_gate_buffer,
            "explicit unary raw builder coexisting with compact plan gates");
        const std::uint64_t gate_subset_mapped_peak = uadd(mapped_gate_arrays,
            uadd(mapped_residual_targets, uadd(mapped_origin_maps, mapped_subset_scratch_peak, "gate-subset mapped buffers"),
                 "gate-subset targets and maps"), "gate-subset mapped buffers");
        const std::uint64_t gate_plan_mapped_peak =
            std::max({gate_builder_mapped_peak, explicit_unary_plan_mapped_peak,
                      gate_subset_mapped_peak});

        const std::uint64_t layer_value_bytes = ubytes(layer_value_entries, sizeof(Fr), "layer witness values");
        const std::uint64_t max_layer_value_bytes = ubytes(maximum_layer_values, sizeof(Fr), "maximum layer witness allocation");
        // Each retained layer is constructed at its exact size and moved into
        // its empty per-layer vector, so there is no geometric growth buffer.
        // Bound mapped backing by the total logical bytes (which also safely
        // covers any per-vector allocator rounding).
        const std::uint64_t mapped_layer_values = mapped_allocation(layer_value_bytes);
        const std::uint64_t mapped_stream_layer = mapped_allocation(max_layer_value_bytes);
        const std::uint64_t mapped_stream_validation = mapped_stream_layer;
        const std::uint64_t mapped_gkr_witness = uadd(mapped_layer_values, mapped_stream_validation,
            "mapped retained and single-layer validated GKR witness");
        report.native_gkr_witness_mapped_bytes_upper_bound = mapped_gkr_witness;
        const std::uint64_t layer_heap_capacity_upper = std::min(layer_value_bytes, umul(layer_array_count, heap_limit, "GKR layer-vector heap capacity bound"));
        const std::uint64_t stream_layer_heap_capacity = std::min(max_layer_value_bytes, heap_limit);
        report.native_gkr_witness_heap_bytes_upper_bound = uadd(layer_heap_capacity_upper,
            stream_layer_heap_capacity, "native GKR witness and one validation layer heap upper bound");
        const std::uint64_t feature_segment_entries = u64_from_size(report.input_layout.segments.at(0).logical_length, "public feature segment entries");
        const std::uint64_t model_segment_entries = model_elements;
        const std::uint64_t mapped_input_segments = uadd(mapped_growth(feature_segment_entries, sizeof(Fr), "mapped public feature segment"),
            mapped_growth(model_segment_entries, sizeof(Fr), "mapped model input segment"), "mapped canonical input segments");
        // Canonical-copy source coordinates persist through the input fold;
        // only one temporary output-seen vector is live while each source is
        // constructed. Both use native backing and are counted separately from
        // the heap-resident lane index/cursors in the bounded fold helper.
        const std::uint64_t mapped_input_oracle_fold = uadd(
            mapped_growth(canonical_copy_entries, sizeof(std::uint64_t),
                          "mapped canonical-copy source-coordinate vectors"),
            mapped_growth(max_canonical_copy_layer_entries,
                          sizeof(unsigned char),
                          "mapped canonical-copy construction seen vector"),
            "mapped canonical-copy source workspace");
        const std::uint64_t first_hidden_input = u64_from_size(model.tensor(hidden_tensor_name(0, 0, "seq/kernel")).shape.dims.at(1), "first hidden input width");
        const std::uint64_t max_inner_domain = pow2(std::max({first_hidden_input, h0_u, umul(h0_u, hidden_heads_u, "hidden FCONN inner width"),
            h1_u, umul(h1_u, hidden_heads_u, "output FCONN inner width"), classes_u}));
        const std::uint64_t max_outer_domain = pow2(std::max({graph_nodes, graph_edges, h0_u, h1_u, classes_u}));
        const std::uint64_t mapped_fconn_fold = umul(4, mapped_growth(max_inner_domain, sizeof(Fr), "mapped FCONN fold buffer"),
            "mapped FCONN fold buffers");
        const std::uint64_t mapped_fconn_source_weights = uadd(mapped_growth(max_inner_domain, sizeof(Fr), "mapped FCONN inner equality weights"),
            mapped_growth(max_outer_domain, sizeof(Fr), "mapped FCONN outer equality weights"), "mapped FCONN source weights");
        const std::uint64_t mapped_gkr_prefix = max_prefix_mapped_bytes;

        std::uint64_t mapped_lasso_persistent = 0;
        std::uint64_t mapped_lasso_transient_peak = 0;
        for (const GatLassoWorkspaceProfile& family : report.lasso_workspace_profile) {
            const NativeLassoCapacityProfile& workspace = family.workspace;
            // The row ledger, row commitments and row-fold vectors are ordinary
            // std::vector allocations. Only the sparse/T2 witness and prepared
            // Hyrax opening buffers use NativeVector backing here.
            const std::uint64_t booleanity_peak = mapped_allocation(workspace.private_witness_bytes);
            const std::uint64_t opening_peak = uadd(mapped_allocation(workspace.masked_opening_generator_bytes),
                uadd(mapped_allocation(workspace.masked_opening_coefficient_bytes),
                    mapped_allocation(workspace.masked_opening_response_bytes), "mapped Lasso opening workspace"),
                "mapped Lasso opening workspace");
            mapped_lasso_transient_peak = std::max(mapped_lasso_transient_peak,
                std::max(booleanity_peak, opening_peak));
        }
        const auto mapped_tensor_transient = [&mapped_allocation, &ubytes, &uadd, &umul](const std::array<std::uint64_t, 3>& shape) {
            return uadd(mapped_allocation(ubytes(shape[1], sizeof(Fr), "mapped tensor row weights")),
                uadd(umul(2, mapped_allocation(ubytes(shape[2], sizeof(G1), "mapped tensor column generators")), "mapped tensor generator columns"),
                    umul(3, mapped_allocation(ubytes(shape[2], sizeof(Fr), "mapped tensor column coefficients")), "mapped tensor coefficient columns"),
                    "mapped tensor column workspace"), "mapped tensor transient workspace");
        };
        // Tensor row commitments/blindings use ordinary std::vector backing,
        // already covered by the host workspace bound; they are not mmap scratch.
        const std::uint64_t mapped_tensor_retained_bytes = 0;
        const std::uint64_t mapped_tensor_opening_peak = std::max(mapped_tensor_transient(aux_tensor), mapped_tensor_transient(l_query_tensor));
        const std::uint64_t mapped_w_active = mapped_allocation(ubytes(w_active, sizeof(G1), "mapped W MLE generators"));
        const std::uint64_t mapped_w_mle_peak = uadd(mapped_w_active,
            umul(2, mapped_allocation(ubytes(w_active, sizeof(Fr), "mapped W MLE coefficients")), "mapped W MLE field buffers"), "mapped W MLE workspace");

        const std::uint64_t proof_persistent_mapped = uadd(mapped_gate_arrays,
            uadd(mapped_origin_maps,
                uadd(mapped_residual_targets,
                    uadd(mapped_layer_values, mapped_input_segments, "mapped retained witness and input backing"), "mapped retained witness and input backing"),
                "mapped proof plan backing"), "mapped proof plan backing");
        const std::uint64_t lasso_phase_persistent_mapped = uadd(mapped_layer_values, mapped_input_segments,
            "mapped retained witness/input backing during Lasso cache release");
        const std::uint64_t plan_restore_mapped_peak = uadd(gate_plan_mapped_peak,
            uadd(lasso_phase_persistent_mapped,
                uadd(mapped_lasso_persistent, mapped_tensor_retained_bytes,
                    "mapped retained Lasso/tensor backing during native plan restore"),
                "mapped proof backing during native plan restore"),
            "native plan restore live peak");
        const std::uint64_t gkr_validation_mapped_peak = uadd(proof_persistent_mapped, mapped_stream_validation,
            "mapped GKR witness-validation live peak");
        const std::uint64_t gkr_proving_mapped_peak = uadd(proof_persistent_mapped,
            uadd(report.gkr_input_fold_cache_mapped_bytes_upper_bound,
                uadd(mapped_input_oracle_fold,
                    uadd(mapped_fconn_fold,
                        uadd(mapped_fconn_source_weights, mapped_gkr_prefix, "mapped GKR prefix and FCONN"), "mapped GKR workspaces"), "mapped input oracle and GKR workspaces"),
                "mapped GKR input-fold caches and workspaces"),
            "mapped GKR live peak");
        const std::uint64_t gkr_mapped_peak = std::max(gkr_validation_mapped_peak, gkr_proving_mapped_peak);
        const std::uint64_t lasso_tensor_mapped_peak = uadd(lasso_phase_persistent_mapped,
            uadd(mapped_lasso_persistent,
                uadd(mapped_tensor_retained_bytes, std::max(mapped_lasso_transient_peak, mapped_tensor_opening_peak), "mapped Lasso/tensor transient peak"),
                "mapped Lasso/tensor retained workspace"), "mapped Lasso proof live peak");
        const std::uint64_t input_open_mapped_peak = uadd(proof_persistent_mapped,
            uadd(mapped_lasso_persistent,
                uadd(mapped_tensor_retained_bytes, std::max(mapped_tensor_opening_peak, mapped_w_mle_peak), "mapped input opening transient peak"),
                "mapped input opening retained workspace"), "mapped input opening live peak");
        report.native_mapped_phase_breakdown = {
            {"gate_plan", gate_plan_mapped_peak},
            {"plan_restore", plan_restore_mapped_peak},
            {"gkr_validation", gkr_validation_mapped_peak},
            {"gkr_proving", gkr_proving_mapped_peak},
            {"lasso_tensor", lasso_tensor_mapped_peak},
            {"input_opening", input_open_mapped_peak},
            {"component:proof_persistent", proof_persistent_mapped},
            {"component:gate_arrays", mapped_gate_arrays},
            {"component:source_maps", mapped_origin_maps},
            {"component:residual_targets", mapped_residual_targets},
            {"component:layer_values", mapped_layer_values},
            {"component:input_segments", mapped_input_segments},
            {"component:input_oracle_fold", mapped_input_oracle_fold},
            {"component:input_fold_cache", report.gkr_input_fold_cache_mapped_bytes_upper_bound},
            {"component:fconn_fold", mapped_fconn_fold},
            {"component:fconn_source_weights", mapped_fconn_source_weights},
            {"component:gkr_prefix", mapped_gkr_prefix},
            {"component:lasso_persistent", mapped_lasso_persistent},
            {"component:tensor_retained", mapped_tensor_retained_bytes},
            {"component:lasso_transient", mapped_lasso_transient_peak},
            {"component:tensor_opening", mapped_tensor_opening_peak},
            {"component:w_mle", mapped_w_mle_peak},
            {"component:source_bitmap", mapped_subset_bitmap},
            {"component:source_rank", mapped_subset_rank},
            {"component:explicit_unary_gate_buffer", mapped_explicit_unary_gate_buffer},
        };
        const std::uint64_t gate_plan_scratch_peak = gate_plan_mapped_peak;
        report.estimated_native_mapped_bytes_upper_bound = std::max({gate_plan_scratch_peak, plan_restore_mapped_peak,
            gkr_mapped_peak, lasso_tensor_mapped_peak, input_open_mapped_peak});
        report.estimated_native_scratch_bytes = report.estimated_native_mapped_bytes_upper_bound;

        const std::uint64_t live_layer_vector_count = umul(
            wide_coefficient_ids ? 8U : 6U, layer_array_count,
            "live native per-layer vector allocation count");
        const std::uint64_t native_live_allocation_count = uadd(live_layer_vector_count, 130,
            "live native vector allocation count");
        report.estimated_native_heap_bytes_upper_bound = uadd(
            umul(native_live_allocation_count, heap_limit, "native heap allocation upper bound"),
            uadd(max_prefix_heap_bytes,
                uadd(uadd(uadd(coefficient_pools_heap_bytes, input_fold_heap_bytes,
                         "native coefficient-pool and bounded input-fold heap"),
                    report.gkr_input_fold_cache_heap_bytes_upper_bound,
                    "native input-fold cache and input-fold helper heap"),
                    uadd(builder_target_dictionary_heap_bytes,
                         explicit_unary_gate_buffer_bytes <= prefix_heap_limit
                             ? explicit_unary_gate_buffer_bytes : 0,
                         "packed target dictionary and heap-backed explicit unary builder"),
                    "native coefficient, input-fold, target-dictionary, and explicit unary builder heap"),
                "GKR prefix and bounded native heap workspace"),
            "native heap and bounded GKR/helper workspaces upper bound");
        const std::uint64_t source_map_heap_scratch = uadd(
            subset_bitmap_bytes <= heap_limit ? subset_bitmap_bytes : 0,
            subset_rank_bytes <= heap_limit ? subset_rank_bytes : 0,
            "heap-backed source-map bitmap and rank prefixes");
        report.estimated_native_heap_bytes_upper_bound = uadd(report.estimated_native_heap_bytes_upper_bound,
            source_map_heap_scratch, "native heap plus source-map bitmap and rank prefixes");
        const std::uint64_t source_ledger_temporary_host_bytes = uadd(320,
            umul(report.gkr_input_source_count_upper_bound,
                uadd(2048, umul(144, input_bits, "GKR source-ledger point serialization"), "GKR source-ledger per-source metadata"),
                "GKR source-ledger construction temporary"),
            "GKR source-ledger construction temporary");
        const auto encoded_string_bytes = [&uadd](std::uint64_t length, const char* what) {
            return uadd(8, length, what);
        };
        const std::uint64_t statement_prefix_bytes = uadd(encoded_string_bytes(sizeof("zkgpt-native-statement-v4") - 1, "native statement prefix"),
            uadd(umul(10, encoded_string_bytes(64, "native statement digest fields"), "native statement digest fields"),
                uadd(encoded_string_bytes(sizeof("bn254-fr-v1") - 1, "native statement field id"),
                    uadd(8, uadd(encoded_string_bytes(sizeof("zkgpt-gat-relation-v1") - 1, "native statement relation schema"),
                        uadd(encoded_string_bytes(64, "native statement relation digest"), 8, "native statement relation order"),
                        "native statement relation metadata"), "native statement dimensions"), "native statement fixed fields"),
                "native statement encoding"), "native statement encoding");
        report.transcript_prefix_bytes_upper_bound = uadd(encoded_string_bytes(sizeof("zkgpt-native-fs-v4") - 1, "native transcript prefix"),
            uadd(encoded_string_bytes(sizeof(kNativeArtifactDomain) - 1, "native transcript domain"),
                uadd(statement_prefix_bytes, 8, "native transcript message count prefix"), "native transcript statement prefix"),
            "native transcript prefix bytes");
        report.maximum_challenge_suffix_bytes_upper_bound = kTranscriptLabelBytesUpper;
        const std::uint64_t transcript_content_bytes = report.transcript_content_bytes_upper_bound;
        const std::uint64_t transcript_wire_bytes = report.transcript_wire_bytes_upper_bound;
        const std::uint64_t replay_encoded_bytes = uadd(report.transcript_prefix_bytes_upper_bound,
            uadd(transcript_wire_bytes, report.maximum_challenge_suffix_bytes_upper_bound, "challenge replay encoding"),
            "challenge replay encoding");
        const std::uint64_t transcript_payload_heap = uadd(umul(5, transcript_content_bytes, "artifact and verifier message strings"),
            uadd(umul(4, transcript_wire_bytes, "transcript body/replay cache capacity"),
                uadd(umul(3, replay_encoded_bytes, "challenge replay string growth"), report.maximum_challenge_suffix_bytes_upper_bound,
                    "challenge replay temporary strings"), "transcript cache and replay strings"), "native transcript string heap");
        const std::uint64_t message_count = report.proof_message_count_upper_bound;
        using LabelPositionsValue = std::pair<const std::string, std::vector<std::size_t>>;
        using ChallengeRoundValue = std::pair<const std::string, std::uint64_t>;
        const std::uint64_t hash_node_overhead = sizeof(void*) + sizeof(std::size_t) + alignof(std::max_align_t) + 32;
        const std::uint64_t hash_node_bytes = uadd(sizeof(LabelPositionsValue),
            uadd(umul(3, sizeof(ChallengeRoundValue), "native transcript challenge-map entries"),
                uadd(umul(3, sizeof(std::string), "native transcript set entries"), umul(7, hash_node_overhead, "native transcript hash nodes"),
                    "native transcript hash nodes"), "native transcript hash nodes"), "native transcript hash node storage");
        const std::uint64_t hash_nodes = umul(message_count, hash_node_bytes, "native transcript map/set nodes");
        const std::uint64_t hash_buckets = umul(umul(7, umul(4, message_count, "native transcript bucket growth"),
            "native transcript bucket growth"), sizeof(void*), "native transcript bucket arrays");
        const std::uint64_t label_key_capacity = umul(2,
            umul(message_count, uadd(umul(7, report.maximum_challenge_suffix_bytes_upper_bound, "native transcript label keys"), 320,
                "native transcript opening cache keys"), "native transcript key bytes"), "native transcript key growth");
        const std::uint64_t label_position_indices = uadd(umul(3, umul(message_count, sizeof(std::size_t), "native transcript label positions"),
            "native transcript label-position capacity growth"), umul(32, message_count, "native transcript label-position allocations"),
            "native transcript label-position indices");
        const std::uint64_t message_vector_metadata = uadd(
            uadd(umul(3, umul(message_count, sizeof(NativeProofMessage), "native transcript message vectors"), "native transcript message vectors"),
                umul(2, umul(message_count, sizeof(TranscriptMessage), "artifact section message vectors"), "artifact section message vectors"),
                "artifact and verifier message vectors"),
            uadd(umul(message_count, sizeof(const TranscriptMessage*), "artifact sequence index"),
                uadd(5 * sizeof(ProofSection), sizeof(NativeProofSession) + sizeof(GatProofArtifact), "artifact/session metadata"),
                "artifact/session and sequence metadata"), "native transcript vector metadata");
        const std::uint64_t statement_string_bytes = umul(2,
            uadd(umul(10, 64, "native statement digest strings"),
                uadd(sizeof("bn254-fr-v1") - 1, sizeof("zkgpt-gat-relation-v1") - 1, "native statement identifiers"),
                "native statement string contents"), "native statement copies");
        const std::uint64_t field_decode_bytes = ubytes(report.maximum_transcript_field_entries_upper_bound, sizeof(Fr),
            "native transcript field-vector decode workspace");
        const std::uint64_t transcript_metadata_bytes = uadd(message_vector_metadata,
            uadd(hash_nodes, uadd(hash_buckets, uadd(label_key_capacity,
                uadd(label_position_indices, uadd(statement_string_bytes, field_decode_bytes, "transcript statement/decode storage"),
                    "transcript keys/indexes/decode storage"), "transcript keys and indexes"), "transcript hash metadata"),
                "transcript hash maps and buckets"), "native transcript metadata");
        report.transcript_metadata_bytes_upper_bound = transcript_metadata_bytes;
        report.native_transcript_heap_bytes_upper_bound = uadd(transcript_payload_heap,
            uadd(transcript_metadata_bytes, source_ledger_temporary_host_bytes, "transcript metadata and source-ledger temporary"),
            "native transcript peak heap");
        const std::uint64_t transcript_host_bytes = report.native_transcript_heap_bytes_upper_bound;
        const std::uint64_t plan_metadata_bytes = ubytes(layer_array_count, 4096, "GKR layer metadata");
        const std::uint64_t proof_phase_host_bytes = uadd(graph_bytes,
            uadd(model_bytes,
                uadd(report.borrowed_trace_bytes_upper_bound,
                    uadd(report.lasso_table_and_index_bytes_upper_bound,
                        uadd(report.estimated_native_heap_bytes_upper_bound,
                            uadd(report.lasso_heap_workspace_peak_bytes,
                                uadd(plan_metadata_bytes, transcript_host_bytes, "plan and transcript host workspace"), "Lasso heap and proof session"),
                            "native heap and proof session"),
                        "tables and native heap"), "borrowed trace and proof state"), "model and proof state"), "graph and proof state");
        const std::uint64_t feature_entries = umul(graph_nodes, u64_from_size(graph.num_features, "forward admission feature width"),
                                                   "forward admission feature entries");
        const std::uint64_t feature_int_bytes = ubytes(feature_entries, sizeof(Int), "forward admission quantized features");
        const std::uint64_t feature_double_bytes = ubytes(feature_entries, sizeof(double), "forward admission decoded feature values");
        const std::uint64_t hidden_concat_entries = uadd(umul(n0, hidden_heads_u, "forward admission hidden layer 0 concat"),
                                                        umul(n1, hidden_heads_u, "forward admission hidden layer 1 concat"),
                                                        "forward admission hidden layer buffers");
        std::vector<std::size_t> admission_degrees(nodes, 0);
        for (Int destination : graph.dst) ++admission_degrees.at(static_cast<std::size_t>(destination));
        const std::size_t max_admission_degree = *std::max_element(admission_degrees.begin(), admission_degrees.end());
        const std::uint64_t projection_entries = umul(graph_nodes, u64_from_size(std::max({h0, h1, classes}), "forward admission projection width"),
                                                       "forward admission projection entries");
        const std::uint64_t forward_buffer_entries = uadd(feature_entries,
            uadd(hidden_concat_entries,
                uadd(projection_entries,
                    uadd(umul(graph_nodes, 4, "forward admission score and group arrays"),
                        uadd(u64_from_size(max_admission_degree, "forward admission maximum degree"),
                            umul(2, nc, "forward admission output/head buffers"), "forward admission edge/output buffers"),
                        "forward admission single-head buffers"), "forward admission matrices and head buffers"),
                "forward admission all live integer vectors"), "forward admission live vectors");
        // std::vector growth capacity is not part of the C++ contract. Keep a
        // two-times source bound for every simultaneously live admission vector.
        const std::uint64_t forward_dynamic_bytes = umul(2, ubytes(forward_buffer_entries, sizeof(Int),
                                                                    "forward admission live vector backing"),
                                                          "forward admission vector capacity envelope");
        const std::uint64_t forward_family_metadata = 64ULL << 10;
        const std::uint64_t forward_execution_peak = uadd(forward_dynamic_bytes,
            uadd(report.lasso_table_and_index_bytes_upper_bound, forward_family_metadata,
                "forward admission static table and family metadata"), "forward admission computation peak");
        const std::uint64_t feature_decode_peak = uadd(feature_int_bytes,
            uadd(feature_source_file_bytes,
                uadd(feature_double_bytes,
                    ubytes(u64_from_size(graph.num_features, "feature row width"), 128,
                           "forward admission maximum text row/map workspace"),
                    "forward admission source and decode buffer"), "forward admission input decode peak"),
            "forward admission input decode with quantized features");
        const std::uint64_t admission_phase_host_bytes = uadd(graph_bytes,
            uadd(model_bytes, std::max({forward_execution_peak, feature_decode_peak, forward_relation_validation_peak - graph_bytes - model_bytes}),
                "forward admission model and workspaces"), "forward admission graph, model, and workspaces");
        report.forward_admission_peak_host_memory_bytes_upper_bound = admission_phase_host_bytes;
        const std::uint64_t shape_admission_storage_bytes = uadd(
            report.gkr_point_shape_bitmap_scratch_bytes,
            report.gkr_point_shape_touched_index_scratch_bytes_upper_bound,
            "shape-admission bitmap and touched-index storage");
        const std::uint64_t shape_admission_peak_host_bytes = uadd(graph_bytes,
            uadd(model_bytes,
                uadd(shape_admission_storage_bytes,
                    uadd(umul(layer_array_count, 4096, "shape-admission layer metadata"), 1ULL << 20,
                         "shape-admission profile metadata"), "shape-admission storage and metadata"),
                "shape-admission model and workspaces"), "shape-admission graph and workspaces");
        report.estimated_peak_host_memory_bytes = std::max({proof_phase_host_bytes, admission_phase_host_bytes,
                                                             shape_admission_peak_host_bytes});

        const bool pointer_ok = report.ppi_node_ptr_boundary && report.ppi_edge_ptr_boundary;
        report.sparse_ready = pointer_ok && report.sparse_domain_entries != 0;
        report.proof_path = "formal_native_masked_lasso_hyrax+native_masked_layered_gkr";
        report.full_relation_bound = true;
        report.ready = false;
        report.reason = pointer_ok ? "runtime resource headroom has not been finalized" : "graph batch node_ptr/edge_ptr boundary is not canonical";
        if (symbolic_profile_only && exact_shape_plan.gkr_point_domain_upper_bound_valid)
            report.reason = "analysis-only symbolic profile; per-gate point source census and point digest were intentionally omitted";
        else if (symbolic_profile_only)
            report.reason = "analysis-only symbolic profile has an unvalidated conservative point-domain upper bound: " + exact_shape_plan.gkr_point_domain_upper_bound_reason;
    } catch (const std::exception& error) {
        report.ready = false;
        report.reason = error.what();
    }
    return report;
}

std::vector<std::array<unsigned, 5>> canonical_division_range_profiles(const Graph& graph, const Model& model, const QuantConfig& quant) {
    return canonical_division_range_bits(graph, model, quant);
}

void finalize_gat_capacity_report(GatCapacityReport& report) {
    report.ready = false;
    const std::uint64_t trusted_base = 1ULL << 30;
    const std::uint64_t trusted_hard_max = 4ULL << 30;
    const std::uint64_t family_count = report.lasso_workspace_profile.size();
    const auto add = [](std::uint64_t left, std::uint64_t right, const char* what) {
        if (right > std::numeric_limits<std::uint64_t>::max() - left) throw std::overflow_error(std::string("capacity u64 sum overflow: ") + what);
        return left + right;
    };
    const auto mul = [](std::uint64_t left, std::uint64_t right, const char* what) {
        if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) throw std::overflow_error(std::string("capacity u64 product overflow: ") + what);
        return left * right;
    };
    try {
        std::uint64_t trusted_limit = trusted_base;
        for (const GatLassoWorkspaceProfile& family : report.lasso_workspace_profile) {
            trusted_limit = add(trusted_limit, mul(2, mul(family.workspace.rows, 284, "trusted row artifact overhead"), "trusted row artifact overhead"), "trusted artifact limit");
            trusted_limit = add(trusted_limit, mul(2, mul(family.row_commitment_chunk_count, 2048, "trusted row chunk overhead"), "trusted row chunk overhead"), "trusted artifact limit");
        }
        trusted_limit = add(trusted_limit, mul(family_count, 64ULL << 10, "trusted family headers"), "trusted artifact limit");
        report.proof_artifact_wire_limit_bytes = trusted_limit <= trusted_hard_max ? trusted_limit : 0;
        report.disk_requirement_bytes_upper_bound = add(report.estimated_native_scratch_bytes, report.artifact_disk_bytes_upper_bound, "scratch plus artifact disk bound");
        const bool pointer_ok = report.ppi_node_ptr_boundary && report.ppi_edge_ptr_boundary;
        if (!pointer_ok) report.reason = "graph batch node_ptr/edge_ptr boundary is not canonical";
        else if (!report.gkr_point_shape_ready || !is_lower_hex_digest(report.gkr_point_shape_digest) ||
                 report.gkr_point_shape_layer_count == 0 || report.gkr_point_shape_fconn_count == 0) {
            if (report.reason.empty() || report.reason == "runtime resource headroom has not been finalized")
                report.reason = "native GKR point-domain shape admission is incomplete";
        }
        else if (report.gkr_point_shape_layer_count != report.gkr_layer_count_upper_bound ||
                 report.gkr_point_shape_fconn_count != report.gkr_fconn_layer_count_upper_bound)
            report.reason = "native GKR point-domain shape admission differs from the capacity circuit profile";
        else if (report.gkr_subset_source_support_entries_upper_bound >
                     std::numeric_limits<std::uint64_t>::max() - 63 ||
                 report.gkr_point_shape_bitmap_scratch_bytes !=
                     2 * ((report.gkr_subset_source_support_entries_upper_bound + 63) / 64) * sizeof(std::uint64_t) ||
                 report.gkr_point_shape_touched_index_scratch_bytes_upper_bound !=
                     2 * ((report.gkr_subset_source_support_entries_upper_bound + 63) / 64) * sizeof(std::size_t) ||
                 report.gkr_subset_bitmap_scratch_bytes_upper_bound !=
                     ((report.gkr_subset_source_support_entries_upper_bound + 63) / 64) * sizeof(std::uint64_t) ||
                 report.gkr_subset_rank_scratch_bytes_upper_bound !=
                     ((report.gkr_subset_source_support_entries_upper_bound + 63) / 64) * sizeof(std::uint64_t))
            report.reason = "native GKR point-domain scratch does not match exact logical source support";
        else if (report.flattened_input_domain_entries == 0 || report.flattened_input_domain_entries > report.flattened_input_domain_limit_u64 ||
                 report.input_layout.n_in > report.coordinate_domain_limit_u64)
            report.reason = "flattened input domain exceeds the checked u64 and MLE-coordinate limits";
        else if (!report.full_relation_bound || report.proof_path != "formal_native_masked_lasso_hyrax+native_masked_layered_gkr")
            report.reason = "formal masked Lasso/Hyrax/GKR relation path is not fully bound";
        else if (!report.forward_value_admission_ready || report.forward_admission_peak_host_memory_bytes_upper_bound == 0) {
            if (report.reason.empty() || report.reason == "runtime resource headroom has not been finalized")
                report.reason = "canonical all-value forward admission is incomplete";
        }
        else if (report.lasso_workspace_profile.size() != 4 || report.lasso_logical_domain_entries == 0 || report.hyrax_active_domain_workspace_entries == 0)
            report.reason = "formal Lasso family profile is incomplete";
        else if (report.proof_message_count_upper_bound == 0 || report.proof_message_count_upper_bound > report.proof_message_count_limit)
            report.reason = "proof message-count upper bound exceeds the authenticated message limit";
        else if (report.maximum_message_string_bytes_upper_bound == 0 || report.maximum_message_string_bytes_upper_bound > report.maximum_message_string_bytes_limit)
            report.reason = "proof message-string upper bound exceeds the authenticated string limit";
        else if (report.proof_artifact_wire_limit_bytes == 0 || report.proof_artifact_wire_bytes_upper_bound > report.proof_artifact_wire_limit_bytes)
            report.reason = "proof artifact upper bound exceeds the trusted dynamic wire limit";
        else if (report.estimated_peak_host_memory_bytes == 0 || report.host_memory_limit_bytes == 0 || report.estimated_peak_host_memory_bytes > report.host_memory_limit_bytes)
            report.reason = "estimated live host working set exceeds effective available memory headroom";
        else if (report.estimated_native_scratch_bytes > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
                 report.estimated_native_scratch_bytes > report.native_scratch_available_bytes)
            report.reason = "estimated native scratch working set exceeds file-offset or scratch-filesystem headroom";
        else if (report.artifact_disk_bytes_upper_bound == 0 || report.artifact_disk_available_bytes == 0)
            report.reason = "artifact output filesystem headroom is unavailable";
        else if (report.scratch_artifact_same_filesystem && report.disk_requirement_bytes_upper_bound > report.shared_scratch_artifact_available_bytes)
            report.reason = "combined native scratch and artifact output exceed shared filesystem headroom";
        else if (!report.scratch_artifact_same_filesystem && report.artifact_disk_bytes_upper_bound > report.artifact_disk_available_bytes)
            report.reason = "proof artifact and context exceed artifact output filesystem headroom";
        else if (report.estimated_native_scratch_bytes > std::numeric_limits<std::size_t>::max() ||
                 report.artifact_disk_bytes_upper_bound > std::numeric_limits<std::size_t>::max())
            report.reason = "capacity upper bound exceeds host size_t allocation limits";
        else {
            report.ready = true;
            report.reason.clear();
        }
    } catch (const std::exception& error) {
        report.ready = false;
        report.reason = error.what();
    }
}

void initialize_gat_field() { ensure_field(); }

NativeFieldVector pack_native_gkr_matmul_rhs(const MatMulDescriptor& descriptor, const NativeFieldVector& canonical_input_layer) {
    return pack_native_gkr_matmul_rhs(
        descriptor, canonical_input_layer.size(),
        [&canonical_input_layer](std::uint64_t index) {
            if (index >= canonical_input_layer.size())
                throw std::out_of_range("native GKR MatMul input coordinate is invalid");
            return canonical_input_layer[static_cast<std::size_t>(index)];
        });
}

NativeFieldVector pack_native_gkr_matmul_rhs(
    const MatMulDescriptor& descriptor, std::uint64_t canonical_input_size,
    const std::function<Fr(std::uint64_t)>& canonical_input_value_at) {
    if (descriptor.rows == 0 || descriptor.inner == 0 || descriptor.cols == 0) throw std::invalid_argument("native GKR MatMul RHS has an empty shape");
    if (!canonical_input_value_at)
        throw std::invalid_argument("native GKR MatMul input provider is empty");
    const std::size_t padded_inner = next_power_of_two(descriptor.inner);
    const std::size_t padded_cols = next_power_of_two(descriptor.cols);
    const std::size_t rhs_size = checked_mul(descriptor.inner, descriptor.cols, "canonical GKR MatMul RHS size");
    const std::size_t native_rhs_size = checked_mul(padded_inner, padded_cols, "padded native GKR MatMul RHS size");
    const std::size_t output_size = checked_mul(descriptor.rows, descriptor.cols, "native GKR MatMul output size");
    if (descriptor.output_coordinates != output_size) throw std::invalid_argument("native GKR MatMul RHS descriptor output shape is inconsistent");
    if (descriptor.coordinate_rule.find("row_major_exact") == std::string::npos)
        throw std::invalid_argument("native GKR MatMul RHS requires canonical row-major coordinates");
    if (descriptor.rhs_offset > canonical_input_size || rhs_size > canonical_input_size - descriptor.rhs_offset)
        throw std::invalid_argument("native GKR MatMul RHS span exceeds canonical input layer");

    NativeFieldVector packed(native_rhs_size, Fr(0));
    for (std::size_t col = 0; col < descriptor.cols; ++col) {
        const std::size_t native_column_base = checked_mul(col, padded_inner, "native GKR MatMul RHS column base");
        for (std::size_t term = 0; term < descriptor.inner; ++term) {
            const std::size_t canonical_relative =
                checked_add(checked_mul(term, descriptor.cols, "canonical MatMul RHS row-major stride"), col, "canonical MatMul RHS row-major coordinate");
            const std::size_t native_index = checked_add(native_column_base, term, "native MatMul RHS column-major coordinate");
            const std::size_t canonical_index = checked_add(descriptor.rhs_offset, canonical_relative, "canonical MatMul RHS input coordinate");
            packed.at(native_index) = canonical_input_value_at(canonical_index);
        }
    }
    return packed;
}

bool validate_gat_circuit_plan(const GatCircuitPlan& plan, std::string* reason) { return validate_gat_circuit_plan_impl(plan, reason); }

void GatCircuitPlan::release_native_proof_backing() const {
    if (!native_proof_backing_resident)
        throw std::logic_error("native GAT proof backing is already released");
    if (virtual_sparse || circuit.circuit.empty() || circuit.size != circuit.circuit.size() ||
        residual_targets.size() != residual_coordinates)
        throw std::logic_error("native GAT proof backing is incomplete and cannot be released");
    for (layer& current : circuit.circuit) current.release_native_proof_backing();
    PackedResidualTargets().swap(residual_targets);
    native_proof_backing_resident = false;
}

void GatCircuitPlan::restore_native_proof_backing(GatCircuitPlan&& canonical_rebuilt) const {
    if (native_proof_backing_resident)
        throw std::logic_error("native GAT proof backing is already resident");
    if (!canonical_rebuilt.native_proof_backing_resident || canonical_rebuilt.virtual_sparse ||
        canonical_rebuilt.plan_digest != plan_digest || canonical_rebuilt.circuit_id != circuit_id ||
        canonical_rebuilt.residual_target_digest != residual_target_digest ||
        canonical_rebuilt.public_output_coordinates != public_output_coordinates ||
        canonical_rebuilt.residual_coordinates != residual_coordinates ||
        canonical_rebuilt.input_layout.n_raw != input_layout.n_raw ||
        canonical_rebuilt.input_layout.n_in != input_layout.n_in ||
        canonical_rebuilt.input_layout.selector != input_layout.selector ||
        canonical_rebuilt.input_layout.segments.size() != input_layout.segments.size())
        throw std::invalid_argument("rebuilt native GAT proof backing does not match the released plan identity");
    for (std::size_t index = 0; index < input_layout.segments.size(); ++index) {
        const InputSegment& expected = input_layout.segments[index];
        const InputSegment& actual = canonical_rebuilt.input_layout.segments[index];
        if (actual.name != expected.name || actual.offset != expected.offset ||
            actual.logical_length != expected.logical_length || actual.padded_length != expected.padded_length)
            throw std::invalid_argument("rebuilt native GAT proof backing has a different canonical input layout");
    }
    std::string validation_reason;
    if (!validate_gat_circuit_plan(canonical_rebuilt, &validation_reason) ||
        !context_layer_metadata_valid(canonical_rebuilt, &validation_reason))
        throw std::invalid_argument("rebuilt native GAT proof backing failed canonical validation: " + validation_reason);
    if (canonical_rebuilt.residual_targets.size() != residual_coordinates ||
        canonical_rebuilt.circuit.size != circuit.size ||
        canonical_rebuilt.circuit.circuit.size() != circuit.circuit.size())
        throw std::invalid_argument("rebuilt native GAT proof backing has a different relation shape");
    for (std::size_t index = 0; index < circuit.circuit.size(); ++index)
        if (!circuit.circuit[index].native_metadata_matches(canonical_rebuilt.circuit.circuit[index]))
            throw std::invalid_argument("rebuilt native GAT proof backing changes retained layer metadata");
    for (std::size_t index = 0; index < circuit.circuit.size(); ++index)
        circuit.circuit[index].restore_native_proof_backing(std::move(canonical_rebuilt.circuit.circuit[index]));
    residual_targets = std::move(canonical_rebuilt.residual_targets);
    native_proof_backing_resident = true;
}

GatCircuitWitness build_gat_circuit_witness(const GatCircuitPlan& plan, const Graph& graph, const Tensor2D& public_features, const Model& model,
                                            const QuantConfig& quant, const ForwardResult& forward) {
    const std::vector<Fr> legacy_l_values =
        canonical_l_query_values(plan, forward, StaticTables::build(quant));
    const std::shared_ptr<std::vector<Fr>> legacy_l_storage =
        std::make_shared<std::vector<Fr>>(legacy_l_values);
    hyrax::HyraxScalarProvider legacy_l_provider;
    legacy_l_provider.size = plan.input_layout.n_in;
    legacy_l_provider.at = [legacy_l_storage](std::uint64_t index) {
        if (index >= legacy_l_storage->size())
            return Fr(0);
        return legacy_l_storage->at(static_cast<std::size_t>(index));
    };
    GatCircuitWitness witness = build_gat_circuit_witness(
        plan, graph, public_features, model, quant, forward,
        legacy_l_provider);
    witness.input_layer.resize(static_cast<std::size_t>(witness.input_layer_size));
    for (std::uint64_t coordinate = 0; coordinate < witness.input_layer_size;
         ++coordinate)
        witness.input_layer[static_cast<std::size_t>(coordinate)] =
            witness.input_oracle->value_at(
                static_cast<std::size_t>(coordinate));
    return witness;
}

GatCircuitWitness build_gat_circuit_witness_impl(
    const GatCircuitPlan& plan, const Graph& graph,
    const Tensor2D& public_features, const Model& model,
    const QuantConfig& quant, const ForwardResult& forward,
    const hyrax::HyraxScalarProvider& l_query_values, bool borrow_aux,
    bool copy_terminal_vectors, bool forward_prevalidated = false) {
    std::string reason;
    if (!validate_gat_circuit_plan(plan, &reason)) throw std::invalid_argument("cannot build GAT circuit witness: " + reason);
    graph.validate();
    model.validate_schema();
    quant.validate();
    if (!forward_prevalidated &&
        !validate_relation(graph, public_features, model, quant, forward))
        throw std::invalid_argument("cannot build GAT circuit witness from an invalid trace");

    GatCircuitWitness witness;
    witness.input_layer_size = plan.circuit.circuit.front().size;
    witness.input_oracle = canonical_input_oracle(
        plan, graph, public_features, model, quant, forward, l_query_values, borrow_aux);
    const std::shared_ptr<SegmentedInputOracle> input_oracle =
        witness.input_oracle;
    const auto input_at = [input_oracle](std::uint64_t coordinate) {
        return input_oracle->value_at(static_cast<std::size_t>(coordinate));
    };
    zkgpt::NativeFieldView terminal_expected;
    if (!copy_terminal_vectors)
        terminal_expected = canonical_public_output_view(plan, forward.logits, &witness);
    if (!evaluate_layered_values(
            plan, witness.input_layer_size, input_at, witness.values, &reason,
            nullptr, copy_terminal_vectors ? nullptr : &terminal_expected))
        throw std::invalid_argument("canonical GAT circuit evaluation failed: " + reason);
    const std::size_t expected_output_size =
        plan.public_output_coordinates + plan.residual_coordinates;
    if (witness.values.size() != plan.circuit.circuit.size() ||
        witness.values.empty() ||
        (copy_terminal_vectors && witness.values.back().size() != expected_output_size) ||
        (!copy_terminal_vectors && !witness.values.back().empty()))
        throw std::invalid_argument("canonical GAT circuit output shape is invalid");
    if (copy_terminal_vectors) {
        const NativeFieldVector& terminal = witness.values.back();
        witness.public_output = terminal;
        witness.residuals.assign(terminal.begin() + plan.public_output_coordinates,
                                 terminal.end());
        const zkgpt::NativeFieldView expected =
            canonical_public_output_view(plan, forward.logits, &witness);
        for (std::size_t index = 0; index < expected_output_size; ++index)
            if (terminal[index] != expected.at(index))
                throw std::invalid_argument("canonical GAT circuit public output binding failed at " + std::to_string(index));
    }
    witness.relation_ok = true;
    return witness;
}

GatCircuitWitness build_gat_circuit_witness(
    const GatCircuitPlan& plan, const Graph& graph,
    const Tensor2D& public_features, const Model& model,
    const QuantConfig& quant, const ForwardResult& forward,
    const hyrax::HyraxScalarProvider& l_query_values) {
    return build_gat_circuit_witness_impl(plan, graph, public_features, model, quant, forward, l_query_values, false, true);
}

GatCircuitWitness build_gat_circuit_witness_borrowed_aux(
    const GatCircuitPlan& plan, const Graph& graph,
    const Tensor2D& public_features, const Model& model,
    const QuantConfig& quant, const ForwardResult& forward,
    const hyrax::HyraxScalarProvider& l_query_values, bool copy_terminal_vectors = true) {
    return build_gat_circuit_witness_impl(plan, graph, public_features, model, quant, forward, l_query_values, true, copy_terminal_vectors);
}

bool evaluate_signed_round_div_gate_assignment(Int dividend, Int denominator, const DivisionResult& result, RoundingMode mode, std::string* reason) {
    ensure_field();
    NativeFieldVector input = {Fr(dividend), Fr(result.value), Fr(denominator)};
    append_division_components(input, result.witness);
    const std::array<unsigned, 5> profile{{63, 63, 63, 63, 63}};
    for (Int value : {result.witness.a, result.witness.q, result.witness.r, result.witness.delta, result.witness.denominator})
        for (unsigned bit = 0; bit < 63; ++bit) input.push_back(Fr(static_cast<Int>((static_cast<std::uint64_t>(value) >> bit) & 1ULL)));
    input.resize(next_power_of_two(input.size()), Fr(0));
    GatCircuitPlan plan;
    layer initial;
    initial.ty = layerType::INPUT;
    initial.size = input.size();
    initial.zero_start_id = input.size();
    initial.bit_length = bit_length(input.size());
    plan.circuit.circuit.push_back(std::move(initial));
    RelationLayerBuilder builder;
    const std::size_t range_width_sum = std::accumulate(profile.begin(), profile.end(), std::size_t(0));
    const std::size_t residual_count = 12 + 5 + range_width_sum;
    const std::size_t division_unary_count = mode == RoundingMode::TOWARD_ZERO ? 14 : 15;
    const std::size_t unary_count = division_unary_count + 5 + 2 * range_width_sum;
    const std::size_t binary_count = 10 + range_width_sum;
    constexpr std::size_t signed_round_coefficient_upper_bound = 70;
    builder.reserve(residual_count, residual_count, unary_count, binary_count,
                    std::min(signed_round_coefficient_upper_bound, unary_count + binary_count));
    add_division_residuals(builder, 3, 1, {{0, 1}}, {{2, 1}}, mode == RoundingMode::TOWARD_ZERO ? "toward_zero" : "nearest_ties_away");
    std::size_t cursor = 12;
    add_division_range_residuals(builder, 3, 1, cursor, profile);
    if (builder.targets.size() != builder.next_output) throw std::logic_error("SignedRoundDiv residual target count mismatch");
    layer relation = std::move(builder.relation);
    relation.ty = layerType::ADD_BIAS;
    relation.size = as_u64(std::max<std::size_t>(1, builder.next_output), "SignedRoundDiv relation size");
    relation.zero_start_id = relation.size;
    relation.bit_length = bit_length(relation.size);
    relation.need_phase2 = true;
    const std::size_t input_size = plan.circuit.circuit.front().size;
    for (const uniGate& gate : relation.uni_gates)
        if (gate.g >= builder.next_output || gate.u >= input_size) throw std::invalid_argument("SignedRoundDiv unary gate coordinate is out of bounds");
    for (const binGate& gate : relation.bin_gates)
        if (gate.g >= builder.next_output || gate.u >= input_size || gate.v >= input_size)
            throw std::invalid_argument("SignedRoundDiv binary gate coordinate is out of bounds");
    if (relation.uni_gates.empty() && relation.bin_gates.empty()) throw std::invalid_argument("SignedRoundDiv relation has no gates");
    plan.circuit.circuit.push_back(std::move(relation));
    plan.circuit.size = static_cast<u8>(plan.circuit.circuit.size());
    const auto subset_begin = std::chrono::steady_clock::now();
    std::cerr << "gat_adapter_prover_stage: native-initSubset begin" << std::endl;
    plan.circuit.initSubset();
    std::cerr << "gat_adapter_prover_stage: native-initSubset complete elapsed_ms="
              << std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - subset_begin).count()
              << std::endl;
    std::vector<NativeFieldVector> values;
    if (!evaluate_layered_values(plan, input, values, reason)) return false;
    for (std::size_t index = 0; index < builder.next_output; ++index) if (values.back()[index] != Fr(builder.targets[index])) {
        set_reason(reason, "native SignedRoundDiv gate residual mismatches public target");
        return false;
    }
    return true;
}

bool evaluate_gat_circuit_assignment(const GatCircuitPlan& plan, const NativeFieldVector& input, const Tensor2D& public_logits, std::string* reason) {
    std::vector<NativeFieldVector> values;
    zkgpt::NativeFieldView terminal_expected;
    try {
        terminal_expected = canonical_public_output_view(plan, public_logits, &values);
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
    if (!evaluate_layered_values(plan, input, values, reason, nullptr,
                                 &terminal_expected)) return false;
    if (values.empty() || values.size() != plan.circuit.circuit.size() ||
        !values.back().empty()) {
        set_reason(reason, "canonical native gate assignment shape mismatch");
        return false;
    }
    return true;
}

static bool evaluate_gat_circuit_witness_with_terminal(
    const GatCircuitPlan& plan, const GatCircuitWitness& witness,
    const zkgpt::NativeFieldView& terminal_expected, std::string* reason) {
    const auto fail = [reason](const std::string& message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    try {
        std::string plan_reason;
        if (!validate_gat_circuit_plan(plan, &plan_reason)) return fail("GAT circuit plan is invalid: " + plan_reason);
        if (witness.input_oracle == nullptr ||
            witness.input_layer_size != plan.circuit.circuit.front().size)
            return fail("GAT circuit witness input provider has the wrong size");
        const std::size_t terminal_size =
            plan.public_output_coordinates + plan.residual_coordinates;
        if (!terminal_expected.valid() || terminal_expected.size() != terminal_size ||
            witness.values.size() != plan.circuit.circuit.size() ||
            witness.values.empty() || !witness.values.front().empty())
            return fail("GAT circuit witness terminal output view has the wrong shape");
        const bool streamed_terminal = witness.values.back().empty();

        std::vector<NativeFieldVector> evaluated;
        const std::shared_ptr<SegmentedInputOracle> input_oracle =
            witness.input_oracle;
        const NativeFieldVector *legacy_input = &witness.input_layer;
        const auto input_at = [input_oracle, legacy_input](
                                  std::uint64_t coordinate) {
            if (!legacy_input->empty()) {
                if (coordinate >= legacy_input->size())
                    throw std::out_of_range("legacy GAT input coordinate is invalid");
                return legacy_input->at(static_cast<std::size_t>(coordinate));
            }
            return input_oracle->value_at(static_cast<std::size_t>(coordinate));
        };
        if (!evaluate_layered_values(
                plan, witness.input_layer_size, input_at, evaluated, reason,
                &witness.values,
                streamed_terminal ? &terminal_expected : nullptr))
            return false;
        if (!streamed_terminal &&
            witness.values.back().size() != terminal_size)
            return fail("GAT circuit witness public output shape is invalid");
        const zkgpt::NativeFieldView terminal_values = streamed_terminal
            ? terminal_expected
            : zkgpt::NativeFieldView::from_vector(witness.values.back());
        if (!streamed_terminal) {
            for (std::size_t index = 0; index < terminal_size; ++index)
                if (terminal_values.at(index) != terminal_expected.at(index))
                    return fail("GAT circuit witness public output does not match the canonical terminal view");
        }
        if (!witness.public_output.empty()) {
            if (witness.public_output.size() != terminal_size)
                return fail("GAT circuit witness public output has the wrong size");
            for (std::size_t index = 0; index < terminal_size; ++index)
                if (witness.public_output[index] != terminal_values.at(index))
                    return fail("GAT circuit witness public output does not match final layer");
        }
        if (!witness.residuals.empty() && witness.residuals.size() != plan.residual_coordinates) return fail("GAT circuit witness residual shape is invalid");
        for (std::size_t index = 0; index < plan.residual_coordinates; ++index) {
            if (!witness.residuals.empty() && witness.residuals[index] != terminal_values.at(plan.public_output_coordinates + index))
                return fail("GAT circuit witness residual is not output-bound");
            if (terminal_values.at(plan.public_output_coordinates + index) != Fr(plan.residual_targets.at(index))) return fail("GAT circuit relation residual does not match its public target");
        }
        if (!witness.relation_ok) return fail("GAT circuit witness was not marked relation-valid");
        return true;
    } catch (const std::exception& error) { return fail(error.what()); }
}

bool evaluate_gat_circuit_witness(const GatCircuitPlan& plan,
                                  const GatCircuitWitness& witness,
                                  std::string* reason) {
    if (witness.values.empty() || witness.values.back().empty()) {
        if (reason != nullptr)
            *reason = "streamed GAT circuit witness requires its canonical terminal output view";
        return false;
    }
    const zkgpt::NativeFieldView terminal_values =
        zkgpt::NativeFieldView::from_vector(witness.values.back());
    return evaluate_gat_circuit_witness_with_terminal(
        plan, witness, terminal_values, reason);
}

GatCircuitPlan build_gat_circuit_plan(const Graph& graph, const Model& model, const QuantConfig& quant, const ForwardResult& forward) {
    return build_plan_impl(graph, model, quant, &forward);
}

GatPointDomainAdmission inspect_gat_point_domain_admission(
    const Graph& graph, const Model& model, const QuantConfig& quant) {
    GatPointDomainAdmission admission;
    try {
        GatCircuitPlan plan = build_plan_impl(graph, model, quant, nullptr, true);
        admission.ready = true;
        admission.digest = plan.gkr_point_shape_digest;
        admission.layer_count = plan.circuit.circuit.size();
        admission.fconn_count = static_cast<std::uint64_t>(std::count_if(
            plan.circuit.circuit.begin(), plan.circuit.circuit.end(),
            [](const layer& current) { return current.ty == layerType::FCONN; }));
        admission.bitmap_bytes = plan.gkr_point_shape_bitmap_bytes;
        admission.touched_index_bytes_upper_bound =
            plan.gkr_point_shape_touched_index_bytes_upper_bound;
    } catch (const std::exception& error) {
        admission.ready = false;
        admission.reason = error.what();
    }
    return admission;
}

GatCircuitPlan build_gat_circuit_plan(const Graph& graph, const Model& model, const QuantConfig& quant) {
    return build_plan_impl(graph, model, quant, nullptr);
}

detail::ValidatedFormalInputs::ValidatedFormalInputs(
    Graph&& graph, Tensor2D&& features, Model&& model, QuantConfig&& quant,
    ForwardResult&& forward)
    : graph_(std::move(graph)), features_(std::move(features)),
      model_(std::move(model)), quant_(std::move(quant)),
      forward_(std::move(forward)) {}

detail::ValidatedFormalInputs detail::ValidatedFormalInputs::consume(
    Graph&& graph, Tensor2D&& features, Model&& model, QuantConfig&& quant,
    ForwardResult&& forward) {
    ValidatedFormalInputs owned(std::move(graph), std::move(features),
                                std::move(model), std::move(quant),
                                std::move(forward));
    const auto validation_begin = std::chrono::steady_clock::now();
    owned.canonical_validation_calls_ = 1;
    if (!validate_relation(owned.graph_, owned.features_, owned.model_,
                           owned.quant_, owned.forward_))
        throw std::invalid_argument("canonical GAT relation validation failed for move-owned formal inputs");
    owned.canonical_validation_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - validation_begin).count();
    owned.capacity_quant_digest_ = canonical_quant_digest(owned.quant_);
    if (owned.capacity_census_calls_ != 0U)
        throw std::logic_error("private formal capacity census was already performed");
    ++owned.capacity_census_calls_;
    owned.capacity_ = inspect_gat_capacity(owned.graph_, owned.model_,
                                           owned.quant_);
    owned.plan_ = std::make_shared<GatCircuitPlan>(
        build_plan_impl(owned.graph_, owned.model_, owned.quant_,
                        &owned.forward_, false, &owned.capacity_, true));
    if (owned.capacity_quant_digest_ != canonical_quant_digest(owned.quant_) ||
        !detail::formal_capacity_report_matches_plan(
            owned.capacity_, owned.graph_, owned.model_, owned.quant_,
            *owned.plan_))
        throw std::logic_error("private formal capacity report does not match its move-owned inputs and canonical plan");
    return owned;
}

bool detail::formal_capacity_report_matches_plan(
    const GatCapacityReport& capacity, const Graph& graph, const Model& model,
    const QuantConfig& quant, const GatCircuitPlan& plan) {
    const auto same_model_shape = [](const CanonicalModelShape& left,
                                     const CanonicalModelShape& right) {
        if (left.model_arch_id != right.model_arch_id ||
            left.tensor_count != right.tensor_count || left.layers != right.layers ||
            left.output_heads != right.output_heads || left.classes != right.classes ||
            left.input_dim != right.input_dim || left.hidden_dim != right.hidden_dim ||
            left.tensors.size() != right.tensors.size())
            return false;
        for (std::size_t index = 0; index < left.tensors.size(); ++index) {
            if (left.tensors[index].name != right.tensors[index].name ||
                !(left.tensors[index].shape == right.tensors[index].shape))
                return false;
        }
        return true;
    };
    const auto same_layout = [](const InputLayout& left, const InputLayout& right) {
        if (left.n_raw != right.n_raw || left.n_in != right.n_in ||
            left.selector != right.selector ||
            left.segments.size() != right.segments.size())
            return false;
        for (std::size_t index = 0; index < left.segments.size(); ++index) {
            const InputSegment& a = left.segments[index];
            const InputSegment& b = right.segments[index];
            if (a.name != b.name || a.offset != b.offset ||
                a.logical_length != b.logical_length ||
                a.padded_length != b.padded_length)
                return false;
        }
        return true;
    };
    try {
        const CanonicalModelShape expected_model = canonical_model_shape(model);
        const InputLayout expected_layout =
            canonical_input_layout_impl(graph, model, quant);
        const std::string quant_digest = canonical_quant_digest(quant);
        const std::string formal_path =
            "formal_native_masked_lasso_hyrax+native_masked_layered_gkr";
        if (capacity.schema != "zkgpt-gat-capacity-v8" ||
            capacity.symbolic_analysis_profile_complete ||
            !capacity.gkr_point_shape_ready || !capacity.gkr_point_shape_exact ||
            !is_lower_hex_digest(capacity.gkr_point_shape_digest) ||
            !capacity.full_relation_bound || capacity.proof_path != formal_path ||
            capacity.explicit_matmul_gate_expansion ||
            capacity.num_nodes != graph.num_nodes ||
            capacity.num_edges != graph.edges() ||
            capacity.num_features != graph.num_features ||
            capacity.num_classes != graph.num_classes ||
            capacity.graph_count != graph.graph_count ||
            !same_model_shape(capacity.model_shape, expected_model) ||
            !same_model_shape(capacity.model_shape, plan.model_shape) ||
            !same_layout(capacity.input_layout, expected_layout) ||
            !same_layout(capacity.input_layout, plan.input_layout) ||
            plan.model_arch_id != model.manifest.model_arch_id ||
            plan.graph_num_nodes != graph.num_nodes ||
            plan.graph_num_features != graph.num_features ||
            plan.graph_count != graph.graph_count ||
            plan.graph_num_classes != graph.num_classes ||
            plan.graph_edge_count != graph.edges() ||
            plan.quant_cfg_id != quant.quant_cfg_id ||
            plan.quant_digest != quant_digest ||
            capacity.gkr_point_shape_digest != plan.gkr_point_shape_digest ||
            plan.recipe_schema != "canonical_plan_recipe_v3" ||
            plan.relation_schema != "zkgpt-gat-relation-v1" ||
            plan.opening_policy != "same_r_three_terminal_openings_v1" ||
            plan.terminal_openings != std::vector<std::string>{"W@r", "Aux@r", "L_qry@r"} ||
            !plan.official_layered || plan.has_placeholder)
            return false;
        return true;
    } catch (...) {
        return false;
    }
}

GatCircuitPlan detail::rebuild_formal_plan_for_restore(
    const detail::ValidatedFormalInputs& inputs) {
    if (inputs.capacity_census_calls() != 1U ||
        canonical_quant_digest(inputs.quant()) != inputs.capacity_quant_digest() ||
        !detail::formal_capacity_report_matches_plan(
            inputs.capacity(), inputs.graph(), inputs.model(), inputs.quant(),
            inputs.plan()))
        throw std::invalid_argument("move-owned formal capacity report is stale or mismatched");
    ++inputs.capacity_restore_rebuild_calls_;
    return build_plan_impl(inputs.graph(), inputs.model(), inputs.quant(),
                           &inputs.forward(), false, &inputs.capacity(), true);
}

bool CommitmentWire::valid() const {
    const std::uint64_t padded = padded_size;
    if (!is_lower_hex_digest(basis_id) || logical_size == 0 || padded_size == 0 ||
        padded > (std::uint64_t(1) << 62) || (padded & (padded - 1)) != 0 ||
        logical_size > padded_size)
        return false;

    if (has_tensor_commitment) {
        const hyrax::HyraxTensorMleCommitment& tensor = tensor_commitment;
        if (scheme != hyrax::kHyraxTensorMleSchema || !point_hex.empty() ||
            tensor.schema != hyrax::kHyraxTensorMleSchema ||
            tensor.method != hyrax::kHyraxTensorMleMethod ||
            tensor.version != hyrax::kHyraxTensorMleVersion ||
            tensor.basis_id != basis_id || tensor.full_domain != padded_size ||
            tensor.logical_length != logical_size ||
            tensor.column_generator_scheme !=
                hyrax::HyraxGeneratorProviderScheme::CanonicalAbsoluteHashToCurveV1 ||
            tensor.column_generator_id.empty() ||
            tensor.column_generator_id.size() > 128 ||
            !is_lower_hex_digest(tensor.root_digest))
            return false;
        std::uint64_t active = 1;
        std::size_t active_bits = 0;
        while (active < logical_size) {
            active <<= 1;
            ++active_bits;
        }
        const std::uint64_t columns = std::uint64_t(1) << (active_bits / 2);
        const std::uint64_t rows = active / columns;
        if (tensor.active_domain != active || tensor.rows != rows ||
            tensor.columns != columns ||
            tensor.row_commitments.size() != rows || rows > kArtifactMaxMessages)
            return false;
        try {
            ensure_field();
            for (const G1& point : tensor.row_commitments) {
                if (!point.isValid()) return false;
                const std::string encoded = point.serializeToHexStr();
                G1 decoded;
                decoded.deserializeHexStr(encoded);
                if (decoded != point) return false;
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    if (scheme != "hyrax-pedersen-bn254" || point_hex.empty()) return false;
    try {
        ensure_field();
        G1 point;
        point.deserializeHexStr(point_hex);
        return point.serializeToHexStr() == point_hex;
    } catch (...) { return false; }
}

CommitmentWire commit_private_segment(const std::vector<Int>& values, const std::string& basis_label) {
    if (values.empty()) throw std::invalid_argument("cannot commit an empty zkGPT segment");
    return commit_private_segment(values, basis_label, next_power_of_two(values.size()));
}

CommitmentWire commit_private_segment(const std::vector<Int>& values, const std::string& basis_label, std::size_t padded_domain) {
    Fr blind;
    return commit_segment_with_mask(values, basis_label, padded_domain, &blind, nullptr);
}

ProverCommitment commit_private_segment_for_prover(const std::vector<Int>& values, const std::string& basis_label, std::size_t padded_domain) {
    ProverCommitment result;
    result.wire = commit_segment_with_mask(values, basis_label, padded_domain, &result.blinder, nullptr);
    return result;
}

namespace {

const char kComWSecretMagic[] = "zkgpt-gat-com-w-secret-v1";

bool read_secret_file(const std::string& path, std::string& payload, std::string* reason) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        set_reason(reason, "cannot open com_W prover secret");
        return false;
    }
    struct stat metadata{};
    if (::fstat(fd, &metadata) != 0 || !S_ISREG(metadata.st_mode) || (metadata.st_mode & (S_IRWXG | S_IRWXO)) != 0 || metadata.st_size < 0 ||
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

bool parse_secret_payload(const ModelRegistration& registration, const std::string& payload, Fr& blinder, std::string* reason) {
    std::istringstream in(payload);
    std::string magic, version, registration_digest, basis_id;
    std::string logical_size, padded_size, point, blinder_hex, trailing;
    if (!(in >> magic >> version >> registration_digest >> basis_id >> logical_size >> padded_size >> point >> blinder_hex) || (in >> trailing) ||
        magic != kComWSecretMagic || version != "1" || registration_digest != registration.registration_digest || basis_id != registration.com_W.basis_id ||
        logical_size != std::to_string(registration.com_W.logical_size) || padded_size != std::to_string(registration.com_W.padded_size) ||
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

}

bool write_prover_commitment_secret(const ModelRegistration& registration, const std::string& path, std::string* reason) {
    try {
        ensure_field();
        if (path.empty() || !registration_wire_valid(registration) || !registration.has_private_padding_mask) {
            set_reason(reason, "com_W prover secret requires an in-process registered model");
            return false;
        }
        if (field_hex(registration.com_W_padding_mask).empty()) {
            set_reason(reason, "com_W prover secret blinder is unavailable");
            return false;
        }
        std::ostringstream encoded;
        encoded << kComWSecretMagic << " 1 " << registration.registration_digest << ' ' << registration.com_W.basis_id << ' ' << registration.com_W.logical_size
                << ' ' << registration.com_W.padded_size << ' ' << registration.com_W.point_hex << ' ' << field_hex(registration.com_W_padding_mask) << '\n';
        const std::string payload = encoded.str();
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
        if (fd < 0) {
            set_reason(reason, errno == EEXIST ? "com_W prover secret already exists" : "cannot create com_W prover secret");
            return false;
        }
        bool ok = true;
        std::size_t offset = 0;
        while (offset < payload.size()) {
            const ssize_t written = ::write(fd, payload.data() + offset, payload.size() - offset);
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
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool read_prover_commitment_secret(const ModelRegistration& registration, const std::string& path, Fr& blinder, std::string* reason) {
    try {
        ensure_field();
        if (path.empty() || !registration_wire_valid(registration)) {
            set_reason(reason, "com_W prover secret requires a valid registration wire");
            return false;
        }
        std::string payload;
        if (!read_secret_file(path, payload, reason)) return false;
        return parse_secret_payload(registration, payload, blinder, reason);
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}

ModelRegistration register_model(const Model& model, const QuantConfig& quant, const std::string& model_arch_id) {
    (void)model;
    (void)quant;
    (void)model_arch_id;
    throw std::invalid_argument("canonical four-segment InputLayout is required for model registration");
}

ModelRegistration register_model(const Model& model, const QuantConfig& quant, const InputLayout& layout, const std::string& model_arch_id) {
    model.validate_schema();
    quant.validate();
    if (model_arch_id.empty()) throw std::invalid_argument("model architecture id is empty");
    const std::vector<Int> flattened = flatten_model(model, quant);
    ModelRegistration registration;
    populate_registration_metadata(registration, model, quant, layout, model_arch_id);
    const std::string basis_label = registration_basis_label(model, quant, layout, model_arch_id);
    registration.com_W = commit_segment_with_mask(flattened, basis_label, layout.n_in, &registration.com_W_padding_mask, nullptr);
    registration.has_private_padding_mask = true;
    if (registration.com_W.logical_size != layout.segments.at(1).logical_length || registration.com_W.padded_size != layout.n_in) {
        throw std::invalid_argument("model commitment does not use the canonical W/N_in domain");
    }
    registration.binding_digest = canonical_digest64(registration_binding_encoding(registration));
    registration.registration_digest = canonical_digest64("zkgpt-gat-registration-v2/" + registration.binding_digest + "/" + commitment_encoding(registration.com_W));
    registration.status = "registered_long_lived_com_W";
    return registration;
}

bool registration_matches_model(const Model& model, const QuantConfig& quant, const InputLayout& layout, const ModelRegistration& registration) {
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
        if (registration.com_W.scheme != "hyrax-pedersen-bn254" || registration.com_W.basis_id != fixed_digest(hash_hex(basis_label)) ||
            registration.com_W.logical_size != layout.segments.at(1).logical_length || registration.com_W.padded_size != layout.n_in)
            return false;
        const std::string expected_binding = canonical_digest64(registration_binding_encoding(registration));
        if (registration.binding_digest != expected_binding) return false;
        const std::string expected_registration =
            canonical_digest64("zkgpt-gat-registration-v2/" + registration.binding_digest + "/" + commitment_encoding(registration.com_W));
        return registration.registration_digest == expected_registration;
    } catch (const std::exception&) { return false; }
}

FiatShamirTranscript::FiatShamirTranscript(const std::string& domain) : domain_(domain) {
    if (domain_.empty()) throw std::invalid_argument("Fiat-Shamir domain is empty");
}

void FiatShamirTranscript::absorb(const std::string& label, const std::string& payload) {
    if (label.empty()) throw std::invalid_argument("Fiat-Shamir message label is empty");
    messages_.push_back(TranscriptMessage{label, payload});
}

Field FiatShamirTranscript::challenge(const std::string& label) {
    if (label.empty()) throw std::invalid_argument("Fiat-Shamir challenge label is empty");
    std::string encoded;
    append_string(encoded, domain_);
    append_u64(encoded, static_cast<std::uint64_t>(messages_.size()));
    for (const TranscriptMessage& message : messages_) {
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
    for (const TranscriptMessage& message : messages_) {
        append_string(encoded, message.label);
        append_string(encoded, message.payload);
    }
    return hash_hex(encoded);
}

bool GatProofArtifact::has_private_payload() const {
    if (forbidden_payload(privacy_profile) || forbidden_payload(statement_digest) || forbidden_payload(transcript_digest)) return true;
    const ProofSection* sections[] = {&Pi_Lasso, &Pi_MatMul, &Pi_GKR, &Pi_combine, &Pi_open};
    for (const ProofSection* section : sections) {
        if (forbidden_payload(section->protocol)) return true;
        for (const TranscriptMessage& message : section->messages)
            if (forbidden_payload(message.label) || forbidden_payload(message.payload)) return true;
    }
    if (forbidden_payload(native_domain) || forbidden_payload(native_statement.field_id)) return true;
    for (const NativeProofMessage& message : native_messages) {
        if (forbidden_payload(message.label) || forbidden_payload(message.payload)) return true;
    }
    if (forbidden_payload(sparse_gkr.plan_digest) || forbidden_payload(sparse_gkr.schedule_digest) || forbidden_payload(sparse_gkr.input_binding) ||
        forbidden_payload(sparse_gkr.output_binding) || forbidden_payload(sparse_gkr.lasso_chunks.root_digest) ||
        forbidden_payload(sparse_gkr.lasso_chunks.final_commitment_hex) || forbidden_payload(sparse_gkr.input_terminal_commitment) ||
        forbidden_payload(sparse_gkr.output_terminal_commitment))
        return true;
    for (const SparseGkrRelationWire& relation : sparse_gkr.relations) {
        if (forbidden_payload(relation.relation_id) || forbidden_payload(relation.initial_claim) || forbidden_payload(relation.final_claim) ||
            forbidden_payload(relation.initial_commitment))
            return true;
        for (const std::string& value : relation.terminal_values)
            if (forbidden_payload(value)) return true;
        for (const std::string& commitment : relation.terminal_commitments)
            if (forbidden_payload(commitment)) return true;
    }
    for (const NativeLassoChunkFamilyWire& family : sparse_gkr.lasso_chunks.families) {
        if (forbidden_payload(family.row_point_digest) || forbidden_payload(family.lookup_terminal_commitment_hex) ||
            forbidden_payload(family.onehot_terminal_commitment_hex))
            return true;
    }
    return false;
}

bool GatProofArtifact::native_session(NativeProofSession& out, std::string* reason) const {
    if (native_domain != kNativeArtifactDomain) {
        set_reason(reason, "unsupported native proof transcript domain");
        return false;
    }
    return NativeProofSession::from_messages(native_domain, native_statement, native_messages, out, reason);
}

bool sparse_proof_wire_valid(const SparseGkrProofWire& wire, std::size_t native_message_count, std::string* reason) {
    const auto fail = [reason](const std::string& message) {
        set_reason(reason, message);
        return false;
    };
    if (wire.schema != "zkgpt-gat-sparse-proof-v2" || wire.version != 2 || !is_lower_hex_digest(wire.plan_digest) ||
        !is_lower_hex_digest(wire.schedule_digest) || wire.relations.empty() || wire.relations.size() > kArtifactMaxMessages ||
        !is_lower_hex_digest(wire.input_binding) || !is_lower_hex_digest(wire.output_binding) || !wire.hyrax_lasso_complete ||
        wire.lasso_chunks.schema != "zkgpt-native-lasso-chunked-v1" || wire.lasso_chunks.version != kNativeLassoChunkVersion ||
        !is_lower_hex_digest(wire.lasso_chunks.tables_digest) || !is_lower_hex_digest(wire.lasso_chunks.resources_digest) ||
        !is_lower_hex_digest(wire.lasso_chunks.root_digest) || wire.lasso_chunks.families.empty() || wire.input_segment_openings.size() != 4 ||
        wire.output_opening.schema != hyrax::kChunkedMleSchema || wire.output_opening.version != hyrax::kChunkedMleVersion ||
        wire.input_terminal_commitment.empty() || wire.output_terminal_commitment.empty())
        return fail("sparse GKR proof wire schema or chunk completion binding is invalid");
    G1 input_terminal_point;
    G1 output_terminal_point;
    if (!core_decode_point(wire.input_terminal_commitment, input_terminal_point) || !core_decode_point(wire.output_terminal_commitment, output_terminal_point))
        return fail("sparse GKR chunk terminal commitments are not canonical points");
    std::set<std::string> ids;
    std::uint64_t previous_end = 0;
    for (const SparseGkrRelationWire& relation : wire.relations) {
        Fr initial_claim;
        Fr final_claim;
        G1 initial_commitment;
        if (relation.relation_id.empty() || !ids.insert(relation.relation_id).second || !core_decode_field(relation.initial_claim, initial_claim) ||
            !core_decode_field(relation.final_claim, final_claim) || relation.terminal_values.empty() || relation.terminal_values.size() > 4 ||
            relation.initial_commitment.empty() || relation.terminal_commitments.size() != relation.terminal_values.size() ||
            relation.transcript_begin >= relation.transcript_end || relation.transcript_end > native_message_count ||
            (!wire.relations.empty() && relation.transcript_begin < previous_end))
            return fail("sparse GKR relation endpoint wire is not canonical");
        Fr decoded;
        for (const std::string& value : relation.terminal_values)
            if (!core_decode_field(value, decoded)) return fail("sparse GKR terminal value is not canonical");
        if (!core_decode_point(relation.initial_commitment, initial_commitment)) return fail("sparse GKR initial commitment is not canonical");
        for (const std::string& commitment : relation.terminal_commitments) {
            G1 point;
            if (!core_decode_point(commitment, point)) return fail("sparse GKR terminal commitment is not canonical");
        }
        previous_end = relation.transcript_end;
    }
    if (wire.terminal_links.size() != wire.relations.size() - 1) return fail("sparse GKR terminal link count is incomplete");
    std::set<std::string> link_ids;
    for (std::size_t index = 0; index < wire.terminal_links.size(); ++index) {
        const SparseGkrLinkWire& link = wire.terminal_links[index];
        if (link.from_relation != wire.relations[index].relation_id || link.to_relation != wire.relations[index + 1].relation_id || link.claim_name.empty() ||
            link.from_claim != wire.relations[index].final_claim || link.to_claim != wire.relations[index + 1].initial_claim ||
            link.from_claim != link.to_claim || !link_ids.insert(link.from_relation + "->" + link.to_relation + "/" + link.claim_name).second)
            return fail("sparse GKR terminal link is missing or inconsistent");
    }
    for (const SparseGkrLinkWire& link : wire.binding_links) {
        if (ids.find(link.from_relation) == ids.end() || ids.find(link.to_relation) == ids.end() || link.claim_name.empty() ||
            !is_lower_hex_digest(link.from_claim) || !is_lower_hex_digest(link.to_claim) ||
            !link_ids.insert(link.from_relation + "->" + link.to_relation + "/" + link.claim_name).second)
            return fail("sparse GKR FCONN binding link is missing or duplicated");
    }
    return true;
}

bool GatProofArtifact::valid_wire(std::string* reason) const {
    NativeProofSession session;
    if (!native_session(session, reason)) return false;
    return valid_wire_with_session(reason, session);
}

bool GatProofArtifact::valid_wire_with_session(std::string* reason, const NativeProofSession& session) const {
    try {
        if (version != 8 || schema != "zkgpt-gat-proof-v8") {
            set_reason(reason, "unsupported proof wire version");
            return false;
        }
        if (relation_id != "R_GAT-zkGPT") {
            set_reason(reason, "proof relation id mismatch");
            return false;
        }
        const bool diagnostic_gkr = privacy_profile == "gkr_only_diagnostic";
        const bool sparse_complete = !sparse_gkr.plan_digest.empty();
        if (privacy_profile != "masked_full" && !diagnostic_gkr) {
            set_reason(reason, "proof does not declare a supported privacy profile");
            return false;
        }
        if (!is_lower_hex_digest(statement_digest) || !canonical_field_wire(transcript_digest)) {
            set_reason(reason, "proof statement or transcript digest is not canonical");
            return false;
        }
        if (!com_aux.valid() || !com_L.valid()) {
            set_reason(reason, "dynamic Hyrax commitment wire is invalid");
            return false;
        }
        if (!native_statement.valid(reason) || native_statement.field_id != "bn254-fr-v1" || native_statement.verifier_context_digest != statement_digest ||
            native_statement.statement_root_digest != statement_digest) {
            set_reason(reason, "native statement is not bound to the public context digest");
            return false;
        }
        if (native_statement.aux_digest != commitment_statement_digest(com_aux) ||
            native_statement.l_digest != commitment_statement_digest(com_L)) {
            set_reason(reason, "native statement commitment binding mismatch");
            return false;
        }

        if (sparse_complete && !sparse_proof_wire_valid(sparse_gkr, native_messages.size(), reason)) return false;

        const ProofSection* sections[] = {&Pi_Lasso, &Pi_MatMul, &Pi_GKR, &Pi_combine, &Pi_open};
        const char* names[] = {"Pi_Lasso", "Pi_MatMul", "Pi_GKR", "Pi_combine", "Pi_open"};
        std::size_t section_message_count = 0;
        for (unsigned i = 0; i < 5; ++i) {
            if (sections[i]->protocol != names[i] || (!diagnostic_gkr && !sparse_complete && sections[i]->messages.empty()) ||
                (sparse_complete && i == 2 && sections[i]->messages.empty())) {
                set_reason(reason, std::string("missing ") + names[i] + " messages");
                return false;
            }
            if (sections[i]->messages.size() > kArtifactMaxMessages || section_message_count > kArtifactMaxMessages - sections[i]->messages.size()) {
                set_reason(reason, "proof section message count exceeds wire limit");
                return false;
            }
            section_message_count += sections[i]->messages.size();
            for (const TranscriptMessage& message : sections[i]->messages) {
                if (message.label.empty() || message.label.size() > kArtifactMaxString || message.payload.size() > kArtifactMaxString ||
                    !valid_native_kind(message.kind)) {
                    set_reason(reason, "proof message exceeds canonical wire limits");
                    return false;
                }
            }
        }
        if (native_messages.empty() || native_messages.size() > kArtifactMaxMessages || native_messages.size() != section_message_count) {
            set_reason(reason, "native transcript and proof sections have different message counts");
            return false;
        }

        std::vector<const TranscriptMessage*> by_sequence(native_messages.size(), nullptr);
        for (const ProofSection* section : sections) {
            std::uint64_t previous = 0;
            bool have_previous = false;
            for (const TranscriptMessage& summary : section->messages) {
                if (summary.sequence >= native_messages.size() || (have_previous && summary.sequence <= previous)) {
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
            const TranscriptMessage* summary = by_sequence[index];
            const NativeProofMessage& native = native_messages[index];
            if (summary == nullptr || native.sequence != static_cast<std::uint64_t>(index) || static_cast<std::uint8_t>(native.kind) != summary->kind ||
                native.label != summary->label || native.payload != summary->payload) {
                set_reason(reason, "proof sections do not exactly map to native transcript");
                return false;
            }
        }

        if (has_private_payload()) {
            set_reason(reason, "proof wire contains a private payload marker");
            return false;
        }
        if (session.domain() != native_domain) {
            set_reason(reason, "provided native proof session domain mismatch");
            return false;
        }
        if (!session.valid(reason) || !session.replay(reason)) return false;
        if (session.statement_digest() != statement_digest || session.digest() != transcript_digest) {
            set_reason(reason, "Fiat-Shamir transcript or statement binding mismatch");
            return false;
        }
        return true;
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool GatProofArtifact::write(const std::string& path, std::string* reason) const {
    return write(path, kArtifactMaxWireBytes, reason);
}

bool GatProofArtifact::serialize(std::string& bytes,
                                 std::uint64_t max_wire_bytes,
                                 std::string* reason) const {
    if (max_wire_bytes == 0 || max_wire_bytes > kArtifactHardMaxWireBytes) {
        set_reason(reason, "proof artifact serialization byte limit is invalid");
        return false;
    }
    if (!valid_wire(reason)) return false;
    return serialize_after_wire_validation(bytes, max_wire_bytes, reason);
}

bool GatProofArtifact::serialize_after_wire_validation(
    std::string& bytes, std::uint64_t max_wire_bytes,
    std::string* reason) const {
    if (max_wire_bytes == 0 || max_wire_bytes > kArtifactHardMaxWireBytes) {
        set_reason(reason, "proof artifact serialization byte limit is invalid");
        return false;
    }
    const auto emit = [this](std::ostream& out) {
        out.write(kProofArtifactMagic, sizeof(kProofArtifactMagic) - 1);
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
        for (const NativeProofMessage& message : native_messages) {
            write_u8(out, static_cast<std::uint8_t>(message.kind));
            write_u64(out, message.sequence);
            write_string(out, message.label);
            write_string(out, message.payload);
        }
        write_string(out, transcript_digest);
        write_sparse_proof_wire(out, sparse_gkr);
        out.flush();
        return out.good();
    };

    class CountingBuffer final : public std::streambuf {
    public:
        explicit CountingBuffer(std::uint64_t limit) : limit_(limit) {}
        std::uint64_t size() const noexcept { return size_; }
        bool exceeded() const noexcept { return exceeded_; }
    protected:
        std::streamsize xsputn(const char*, std::streamsize count) override {
            if (count < 0 || static_cast<std::uint64_t>(count) > limit_ - size_) {
                exceeded_ = true;
                return 0;
            }
            size_ += static_cast<std::uint64_t>(count);
            return count;
        }
        int_type overflow(int_type value) override {
            if (traits_type::eq_int_type(value, traits_type::eof()))
                return traits_type::not_eof(value);
            if (size_ == limit_) {
                exceeded_ = true;
                return traits_type::eof();
            }
            ++size_;
            return value;
        }
    private:
        std::uint64_t limit_;
        std::uint64_t size_ = 0;
        bool exceeded_ = false;
    };

    CountingBuffer counter(max_wire_bytes);
    std::ostream sizing_stream(&counter);
    if (!emit(sizing_stream) || counter.exceeded() || counter.size() == 0 ||
        counter.size() > std::numeric_limits<std::size_t>::max()) {
        set_reason(reason, "proof artifact exceeds trusted-context byte limit while sizing wire");
        return false;
    }
    try {
        bytes.resize(static_cast<std::size_t>(counter.size()));
    } catch (const std::exception& error) {
        bytes.clear();
        set_reason(reason, std::string("cannot allocate exact proof artifact wire buffer: ") + error.what());
        return false;
    }
    if (bytes.capacity() != bytes.size()) {
        bytes.clear();
        set_reason(reason, "proof artifact wire string allocation exceeded its exact requested capacity");
        return false;
    }

    class FixedBuffer final : public std::streambuf {
    public:
        FixedBuffer(char* data, std::size_t size) : data_(data), size_(size) {}
        std::size_t written() const noexcept { return written_; }
        bool exceeded() const noexcept { return exceeded_; }
    protected:
        std::streamsize xsputn(const char* source, std::streamsize count) override {
            if (count < 0 || static_cast<std::uint64_t>(count) > size_ - written_) {
                exceeded_ = true;
                return 0;
            }
            const std::size_t amount = static_cast<std::size_t>(count);
            if (amount != 0) std::memcpy(data_ + written_, source, amount);
            written_ += amount;
            return count;
        }
        int_type overflow(int_type value) override {
            if (traits_type::eq_int_type(value, traits_type::eof()))
                return traits_type::not_eof(value);
            if (written_ == size_) {
                exceeded_ = true;
                return traits_type::eof();
            }
            data_[written_++] = traits_type::to_char_type(value);
            return value;
        }
    private:
        char* data_;
        std::size_t size_;
        std::size_t written_ = 0;
        bool exceeded_ = false;
    };

    FixedBuffer output(bytes.data(), bytes.size());
    std::ostream output_stream(&output);
    if (!emit(output_stream) || output.exceeded() || output.written() != bytes.size()) {
        bytes.clear();
        set_reason(reason, "proof artifact wire changed between sizing and serialization");
        return false;
    }
    return true;
}

bool GatProofArtifact::write(const std::string& path,
                             std::uint64_t max_wire_bytes,
                             std::string* reason) const {
    std::string bytes;
    if (!serialize(bytes, max_wire_bytes, reason)) return false;
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!out) {
        set_reason(reason, "cannot open proof artifact for writing");
        return false;
    }
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.flush();
    if (!out.good()) {
        out.close();
        std::remove(path.c_str());
        set_reason(reason, "proof artifact write failed");
        return false;
    }
    out.close();
    if (out.fail()) {
        std::remove(path.c_str());
        set_reason(reason, "proof artifact close failed");
        return false;
    }
    return true;
}

bool GatProofArtifact::read(const std::string& path, GatProofArtifact& out, std::string* reason) {
    return read(path, out, kArtifactMaxWireBytes, reason);
}

bool GatProofArtifact::read_wire_bytes(const std::string& path,
                                       std::string& bytes,
                                       std::uint64_t max_wire_bytes,
                                       std::string* reason) {
    bytes.clear();
    if (max_wire_bytes == 0 || max_wire_bytes > kArtifactHardMaxWireBytes) {
        set_reason(reason, "proof artifact reader byte limit is invalid");
        return false;
    }
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        set_reason(reason, "cannot open proof artifact without following links");
        return false;
    }
    struct stat info{};
    if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_nlink != 1 ||
        info.st_size < 0 || static_cast<std::uint64_t>(info.st_size) > max_wire_bytes ||
        static_cast<std::uint64_t>(info.st_size) > kArtifactHardMaxWireBytes ||
        static_cast<std::uint64_t>(info.st_size) > std::numeric_limits<std::size_t>::max()) {
        ::close(fd);
        set_reason(reason, "proof artifact exceeds trusted-context size limit");
        return false;
    }
    const std::size_t size = static_cast<std::size_t>(info.st_size);
    try {
        bytes.resize(size);
    } catch (const std::exception& error) {
        ::close(fd);
        set_reason(reason, std::string("cannot allocate exact proof artifact input buffer: ") + error.what());
        return false;
    }
    if (bytes.capacity() != bytes.size()) {
        bytes.clear();
        ::close(fd);
        set_reason(reason, "proof artifact input buffer exceeded its exact requested capacity");
        return false;
    }
    std::size_t offset = 0;
    while (offset < size) {
        const ssize_t count = ::read(fd, &bytes[offset], size - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            bytes.clear();
            ::close(fd);
            set_reason(reason, "proof artifact read was truncated or failed");
            return false;
        }
        offset += static_cast<std::size_t>(count);
    }
    char trailing = 0;
    ssize_t trailing_count = 0;
    do {
        trailing_count = ::read(fd, &trailing, 1);
    } while (trailing_count < 0 && errno == EINTR);
    const int close_result = ::close(fd);
    if (trailing_count != 0 || close_result != 0) {
        bytes.clear();
        set_reason(reason, trailing_count > 0 ? "proof artifact changed while being read" : "cannot close proof artifact input");
        return false;
    }
    return true;
}

bool GatProofArtifact::parse_wire(const std::string& bytes,
                                  GatProofArtifact& out,
                                  std::uint64_t max_wire_bytes,
                                  std::string* reason) {
    if (max_wire_bytes == 0 || max_wire_bytes > kArtifactHardMaxWireBytes ||
        bytes.size() > max_wire_bytes || bytes.size() > kArtifactHardMaxWireBytes) {
        set_reason(reason, "proof artifact wire exceeds trusted-context size limit");
        return false;
    }

    class FixedInputBuffer final : public std::streambuf {
    public:
        explicit FixedInputBuffer(const std::string& source) {
            char* begin = const_cast<char*>(source.data());
            setg(begin, begin, begin + source.size());
        }
    } input_buffer(bytes);
    std::istream in(&input_buffer);

    char magic[sizeof(kProofArtifactMagic) - 1];
    if (!read_exact(in, magic, sizeof(magic)) || std::memcmp(magic, kProofArtifactMagic, sizeof(magic)) != 0) {
        set_reason(reason, "invalid proof artifact magic");
        return false;
    }
    GatProofArtifact parsed;
    std::uint32_t version = 0;
    if (!read_u32(in, version) || !read_string(in, parsed.schema) || !read_string(in, parsed.relation_id) || !read_string(in, parsed.privacy_profile) ||
        !read_string(in, parsed.statement_digest) || !read_commitment(in, parsed.com_aux) || !read_commitment(in, parsed.com_L) ||
        !read_section(in, parsed.Pi_Lasso) || !read_section(in, parsed.Pi_MatMul) || !read_section(in, parsed.Pi_GKR) || !read_section(in, parsed.Pi_combine) ||
        !read_section(in, parsed.Pi_open) || !read_string(in, parsed.native_domain) || !read_native_statement(in, parsed.native_statement)) {
        set_reason(reason, "truncated proof artifact");
        return false;
    }
    parsed.version = version;
    std::uint64_t native_count = 0;
    if (!read_u64(in, native_count) || native_count == 0 || native_count > kArtifactMaxMessages) {
        set_reason(reason, "invalid native proof message count");
        return false;
    }
    parsed.native_messages.reserve(static_cast<std::size_t>(native_count));
    for (std::uint64_t i = 0; i < native_count; ++i) {
        NativeProofMessage message;
        std::uint8_t kind = 0;
        if (!read_u8(in, kind) || !valid_native_kind(kind) || !read_u64(in, message.sequence) || !read_string(in, message.label) ||
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
    if (!read_sparse_proof_wire(in, parsed.sparse_gkr)) {
        set_reason(reason, "truncated sparse GKR proof wire");
        return false;
    }
    char trailing = 0;
    if (in.read(&trailing, 1)) {
        set_reason(reason, "proof artifact has trailing bytes");
        return false;
    }
    if (!in.eof()) {
        set_reason(reason, "proof artifact decoder failed before end of wire");
        return false;
    }
    out = std::move(parsed);
    return true;
}

bool GatProofArtifact::decode_wire(const std::string& bytes,
                                  GatProofArtifact& out,
                                  std::uint64_t max_wire_bytes,
                                  std::string* reason) {
    GatProofArtifact parsed;
    if (!parse_wire(bytes, parsed, max_wire_bytes, reason) ||
        !parsed.valid_wire(reason))
        return false;
    out = std::move(parsed);
    return true;
}

bool GatProofArtifact::read(const std::string& path, GatProofArtifact& out,
                            std::uint64_t max_wire_bytes,
                            std::string* reason) {
    std::string bytes;
    if (!read_wire_bytes(path, bytes, max_wire_bytes, reason)) return false;
    return decode_wire(bytes, out, max_wire_bytes, reason);
}

VerifierContext make_verifier_context(const Graph& graph, const Tensor2D& public_features, const QuantConfig& quant, const StaticTables& tables,
                                      const Tensor2D& public_logits, const CommitmentWire& com_W, const GatCircuitPlan& plan, const std::string& model_arch_id,
                                      const std::string& privacy_profile) {
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

static VerifierContext make_verifier_context_impl(const Graph& graph, const Tensor2D& public_features, const QuantConfig& quant,
                                                  const StaticTables& tables, const Tensor2D& public_logits,
                                                  const ModelRegistration& registration, const GatCircuitPlan& plan,
                                                  std::shared_ptr<const GatCircuitPlan> shared_plan,
                                                  const std::string& privacy_profile) {
    ensure_field();
    graph.validate();
    quant.validate();
    if (!tables.authenticate()) throw std::invalid_argument("unauthenticated static tables in verifier context");
    if (public_features.rows != graph.num_nodes || public_features.cols != graph.num_features || public_features.cols == 0 ||
        public_features.values.size() != public_features.rows * public_features.cols)
        throw std::invalid_argument("invalid public feature shape in verifier context");
    if (public_logits.rows != graph.num_nodes || public_logits.cols != graph.num_classes || public_logits.cols == 0 ||
        public_logits.values.size() != public_logits.rows * public_logits.cols)
        throw std::invalid_argument("invalid public logit shape in verifier context");
    if (registration.model_arch_id.empty() || privacy_profile.empty()) throw std::invalid_argument("verifier context identity is incomplete");
    if (tables.quant_cfg_id != quant.quant_cfg_id || tables.table_id.empty())
        throw std::invalid_argument("static table identity does not match quantization config");
    if (!registration_wire_valid(registration)) throw std::invalid_argument("invalid bound model registration artifact");
    if (!plan.official_layered || plan.has_placeholder || plan.circuit_id.empty() || plan.plan_digest.empty())
        throw std::invalid_argument("verifier context requires an official layered plan");
    if (!canonical_layout_valid(plan.input_layout) || registration.layout_digest != canonical_digest64([&]() {
                                                          std::string encoded;
                                                          append_layout_encoding(encoded, plan.input_layout);
                                                          return encoded;
                                                      }()) ||
        registration.n_in != plan.input_layout.n_in || registration.input_segments.size() != plan.input_layout.segments.size()) {
        throw std::invalid_argument("bound model registration does not match the canonical four-segment plan layout");
    }
    for (std::size_t index = 0; index < plan.input_layout.segments.size(); ++index) {
        const InputSegment& left = registration.input_segments[index];
        const InputSegment& right = plan.input_layout.segments[index];
        if (left.name != right.name || left.offset != right.offset || left.logical_length != right.logical_length ||
            left.padded_length != right.padded_length) {
            throw std::invalid_argument("bound model registration segment does not match plan layout");
        }
    }
    if (registration.quant_digest != canonical_quant_digest(quant) || registration.static_tables_id != tables.table_id || registration.field_id != "BN254" ||
        registration.field_modulus != Field::getModulo())
        throw std::invalid_argument("bound model registration policy identity mismatch");
    VerifierContext context;
    context.graph = graph;
    context.public_features = public_features;
    context.quant = quant;
    context.tables = tables;
    context.public_logits = public_logits;
    context.com_W = registration.com_W;
    context.registration = registration;
    if (shared_plan) {
        if (shared_plan.get() != &plan)
            throw std::invalid_argument("shared verifier plan owner does not refer to the validated plan");
        context.shared_plan = std::move(shared_plan);
    } else {
        context.plan = plan;
    }

    context.backend = "cpu";
    context.model_arch_id = registration.model_arch_id;
    context.privacy_extension_profile_id = privacy_profile;
    context.quant_cfg_id = quant.quant_cfg_id;
    context.static_tables_id = tables.table_id;
    if (!registration.dataset.empty()) {
        context.full_dataset_identity = "canonical:" + registration.dataset;
        context.checkpoint_identity = "baseline-local/checkpoints/" + registration.dataset + "_gat/integrity.sha256";
        context.cache_identity = "baseline-local/data/cache/" + registration.dataset + "/integrity.sha256";
    }
    return context;
}

VerifierContext make_verifier_context(const Graph& graph, const Tensor2D& public_features, const QuantConfig& quant,
                                      const StaticTables& tables, const Tensor2D& public_logits,
                                      const ModelRegistration& registration, const GatCircuitPlan& plan,
                                      const std::string& privacy_profile) {
    return make_verifier_context_impl(graph, public_features, quant, tables, public_logits, registration, plan, {}, privacy_profile);
}

VerifierContext make_verifier_context(const Graph& graph, const Tensor2D& public_features, const QuantConfig& quant,
                                      const StaticTables& tables, const Tensor2D& public_logits,
                                      const ModelRegistration& registration,
                                      std::shared_ptr<const GatCircuitPlan> plan,
                                      const std::string& privacy_profile) {
    if (!plan) throw std::invalid_argument("shared verifier plan owner is null");
    return make_verifier_context_impl(graph, public_features, quant, tables, public_logits, registration, *plan,
                                      std::move(plan), privacy_profile);
}

std::string verifier_context_digest(const VerifierContext& context) { return canonical_digest64(context_encoding(context)); }

std::uint64_t gat_proof_artifact_wire_limit(const VerifierContext& context,
                                            std::string* reason) {
    try {
        unsigned __int128 rows = 0;
        unsigned __int128 chunks = 0;
        for (const LassoFamilyDescriptor& family : context.plan_ref().lasso) {
            const std::uint64_t padded_queries =
                static_cast<std::uint64_t>(family.padded_queries);
            const std::uint64_t table_domain =
                static_cast<std::uint64_t>(family.table_domain);
            if (static_cast<std::size_t>(padded_queries) !=
                    family.padded_queries ||
                static_cast<std::size_t>(table_domain) !=
                    family.table_domain)
                throw std::overflow_error(
                    "trusted Lasso dimensions exceed u64 artifact bound");
            const NativeLassoFamilyPublic::MatrixDescriptor layout =
                native_lasso_balanced_matrix_layout(padded_queries,
                                                    table_domain);
            rows += layout.rows;
            chunks += layout.rows / 4096 +
                      static_cast<std::uint64_t>(layout.rows % 4096 != 0);
        }
        const unsigned __int128 max_wire =
            static_cast<unsigned __int128>(kArtifactMaxWireBytes) +
            rows * kArtifactRowPointBytesPerCopy * 2 +
            chunks * kArtifactRowChunkBytesPerCopy * 2 +
            static_cast<unsigned __int128>(context.plan_ref().lasso.size()) *
                kArtifactRowHeaderBytesPerFamily;
        if (max_wire > kArtifactHardMaxWireBytes ||
            max_wire > std::numeric_limits<std::size_t>::max()) {
            set_reason(reason,
                       "trusted Q/T row commitments exceed the absolute 4 GiB proof artifact limit");
            return 0;
        }
        return static_cast<std::uint64_t>(max_wire);
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return 0;
    }
}

namespace {

std::string serialize_verifier_context_payload(const VerifierContext& context) {
    std::string payload;
    append_string(payload, kVerifierContextSchema);
    append_string(payload, context.relation_id);
    append_string(payload, context.backend);
    append_string(payload, context.model_arch_id);
    append_string(payload, context.privacy_extension_profile_id);
    append_string(payload, context.quant_cfg_id);
    append_string(payload, context.static_tables_id);
    append_string(payload, context.full_dataset_identity);
    append_string(payload, context.checkpoint_identity);
    append_string(payload, context.cache_identity);
    append_context_graph(payload, context.graph);
    append_context_tensor(payload, context.public_features);
    append_context_tensor(payload, context.public_logits);
    append_context_quant(payload, context.quant);
    append_context_tables(payload, context.tables);

    append_context_registration(payload, context.registration);
    append_context_commitment(payload, context.com_W);
    append_context_plan(payload, context.plan_ref());
    const std::string digest = verifier_context_digest(context);
    append_string(payload, digest);

    append_string(payload, digest);
    return payload;
}

bool read_context_file(const std::string& path, std::string& wire, std::string* reason) {
    struct stat info{};
    if (::lstat(path.c_str(), &info) != 0 || S_ISLNK(info.st_mode) || !S_ISREG(info.st_mode) || info.st_nlink != 1) {
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

bool read_header_u32(const std::string& wire, std::size_t offset, std::uint32_t& value) {
    if (offset > wire.size() || wire.size() - offset < 4) return false;
    value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= static_cast<std::uint32_t>(static_cast<unsigned char>(wire[offset + i])) << (i * 8);
    return true;
}

bool read_header_u64(const std::string& wire, std::size_t offset, std::uint64_t& value) {
    if (offset > wire.size() || wire.size() - offset < 8) return false;
    value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(static_cast<unsigned char>(wire[offset + i])) << (i * 8);
    return true;
}

}

std::size_t compact_plan_recipe_bytes(const GatCircuitPlan& plan) { return plan_encoding(plan).size(); }

std::size_t verifier_context_payload_bytes(const VerifierContext& context) { return serialize_verifier_context_payload(context).size(); }

bool write_verifier_context(const VerifierContext& context, const std::string& path, std::string* reason) {
    try {
        if (!validate_verifier_context_object(context, reason)) return false;
        std::string payload = serialize_verifier_context_payload(context);
        if (payload.size() > kContextMaxWireBytes) {
            set_reason(reason, "serialized verifier context exceeds the wire-size limit");
            return false;
        }
        struct stat existing{};
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
        write_u32(output, kVerifierContextVersion);
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
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool read_verifier_context(const std::string& path, VerifierContext& context, std::string* reason) {
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
        if (!read_header_u32(wire, 8, version) || !read_header_u64(wire, 12, payload_size) || version != kVerifierContextVersion ||
            payload_size != wire.size() - header_size || payload_size > kContextMaxWireBytes) {
            set_reason(reason, "unsupported, oversized, or non-canonical verifier context header");
            return false;
        }
        ContextReader reader(wire, header_size, wire.size());
        VerifierContext parsed;
        std::string schema;
        if (!reader.read_string(schema, "context schema") || schema != kVerifierContextSchema || !reader.read_string(parsed.relation_id, "context relation") ||
            !reader.read_string(parsed.backend, "context backend") || !reader.read_string(parsed.model_arch_id, "context model architecture") ||
            !reader.read_string(parsed.privacy_extension_profile_id, "context privacy profile") ||
            !reader.read_string(parsed.quant_cfg_id, "context quant id") || !reader.read_string(parsed.static_tables_id, "context table id") ||
            !reader.read_string(parsed.full_dataset_identity, "context dataset identity") ||
            !reader.read_string(parsed.checkpoint_identity, "context checkpoint identity") ||
            !reader.read_string(parsed.cache_identity, "context cache identity") || !read_context_graph(reader, parsed.graph) ||
            !read_context_tensor(reader, parsed.public_features, "public feature tensor") ||
            !read_context_tensor(reader, parsed.public_logits, "public logits tensor") || !read_context_quant(reader, parsed.quant) ||
            !read_context_tables(reader, parsed.tables) || !read_context_registration(reader, parsed.registration) ||
            !read_context_commitment(reader, parsed.com_W) || !read_context_plan(reader, parsed.plan)) {
            set_reason(reason, reader.error().empty() ? "malformed verifier context" : reader.error());
            return false;
        }
        std::string stored_digest;
        std::string stored_root;
        if (!reader.read_string(stored_digest, "context digest") || !reader.read_string(stored_root, "context structured root") || !reader.at_end()) {
            set_reason(reason, reader.error().empty() ? "verifier context has trailing or unknown bytes" : reader.error());
            return false;
        }

        try {
            parsed.graph.validate();
            parsed.quant.validate();
            const StaticTables canonical_tables = StaticTables::build(parsed.quant);
            if (!static_tables_equal(parsed.tables, canonical_tables) || !parsed.tables.authenticate()) {
                set_reason(reason, "verifier context static tables are not canonical");
                return false;
            }
            if (!registration_wire_valid(parsed.registration) || parsed.com_W.scheme != parsed.registration.com_W.scheme ||
                parsed.com_W.basis_id != parsed.registration.com_W.basis_id || parsed.com_W.logical_size != parsed.registration.com_W.logical_size ||
                parsed.com_W.padded_size != parsed.registration.com_W.padded_size || parsed.com_W.point_hex != parsed.registration.com_W.point_hex) {
                set_reason(reason, "verifier context registration or com_W binding is invalid");
                return false;
            }
            if (parsed.relation_id != "R_GAT-zkGPT" || (parsed.backend != "cpu" && parsed.backend != "gpu") ||
                parsed.model_arch_id != parsed.registration.model_arch_id || parsed.quant_cfg_id != parsed.quant.quant_cfg_id ||
                parsed.static_tables_id != parsed.tables.table_id || parsed.privacy_extension_profile_id != "masked_full" ||
                parsed.tables.quant_cfg_id != parsed.quant.quant_cfg_id) {
                set_reason(reason, "verifier context public identity is not canonical");
                return false;
            }
            std::string compact_reason;
            if (!canonical_model_shape_valid(parsed.plan.model_shape, &compact_reason) || parsed.plan.recipe_schema != "canonical_plan_recipe_v3" ||
                parsed.plan.model_arch_id != parsed.registration.model_arch_id || parsed.plan.graph_num_nodes != parsed.graph.num_nodes ||
                parsed.plan.graph_num_features != parsed.graph.num_features || parsed.plan.graph_count != parsed.graph.graph_count ||
                parsed.plan.graph_num_classes != parsed.graph.num_classes || parsed.plan.graph_edge_count != parsed.graph.edges() ||
                parsed.plan.graph_topology_digest != canonical_graph_topology_digest(parsed.graph) || parsed.plan.quant_cfg_id != parsed.quant.quant_cfg_id ||
                parsed.plan.quant_digest != canonical_quant_digest(parsed.quant) || parsed.plan.static_tables_id != parsed.tables.table_id) {
                set_reason(reason, compact_reason.empty() ? "compact plan recipe identity is not canonical" : compact_reason);
                return false;
            }
        } catch (const std::exception& error) {
            set_reason(reason, error.what());
            return false;
        }
        GatCircuitPlan compact_recipe = std::move(parsed.plan);
        GatCircuitPlan materialized_plan;
        std::string materialize_reason;
        if (!materialize_compact_plan(parsed.graph, parsed.quant, compact_recipe, materialized_plan, &materialize_reason)) {
            set_reason(reason, materialize_reason.empty() ? "compact plan materialization failed" : materialize_reason);
            return false;
        }
        parsed.plan = std::move(materialized_plan);
        std::string validation_reason;
        if (!validate_verifier_context_object(parsed, &validation_reason)) {
            set_reason(reason, validation_reason);
            return false;
        }
        const std::string expected_digest = verifier_context_digest(parsed);
        if (!is_lower_hex_digest(stored_digest) || !is_lower_hex_digest(stored_root) || stored_digest != expected_digest || stored_root != expected_digest) {
            set_reason(reason, "verifier context digest or structured root does not match canonical content");
            return false;
        }
        context = std::move(parsed);
        return true;
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}

static NativeProofStatement make_native_proof_statement_with_basis_digest(const VerifierContext& context, const CommitmentWire& com_aux,
                                                                           const CommitmentWire& com_L, std::size_t l_dimension,
                                                                           const std::string& basis_digest) {
    ensure_field();
    if (l_dimension == 0 || !context.plan_ref().official_layered || context.plan_ref().has_placeholder || !registration_wire_valid(context.registration) ||
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
    append_string(resource_material, context.plan_ref().plan_digest);

    NativeProofStatement statement;
    statement.verifier_context_digest = verifier_context_digest(context);
    statement.field_id = "bn254-fr-v1";
    if (!is_lower_hex_digest(context.plan_ref().plan_digest)) throw std::invalid_argument("native statement plan digest is not canonical");
    statement.plan_digest = context.plan_ref().plan_digest;
    statement.resources_digest = canonical_digest64(resource_material);
    statement.tables_digest = normalize_digest(context.tables.digest);
    statement.com_w_digest = fixed_digest(hash_hex(commitment_encoding(context.com_W)));
    statement.aux_digest = commitment_statement_digest(com_aux);
    statement.l_digest = commitment_statement_digest(com_L);
    statement.l_dimension = l_dimension;
    statement.basis_digest = basis_digest;

    statement.statement_root_digest = statement.verifier_context_digest;
    std::string reason;
    if (!statement.valid(&reason)) throw std::invalid_argument(reason);
    return statement;
}

NativeProofStatement make_native_proof_statement(const VerifierContext& context, const CommitmentWire& com_aux, const CommitmentWire& com_L,
                                                 const G1& G, const std::vector<G1>& generators, const std::vector<Fr>& basis,
                                                 std::size_t l_dimension) {
    return make_native_proof_statement_with_basis_digest(
        context, com_aux, com_L, l_dimension,
        NativeProofStatement::basis_digest_for(G, generators, basis, l_dimension));
}

NativeProofStatement make_native_proof_statement(const VerifierContext& context, const CommitmentWire& com_aux, const CommitmentWire& com_L,
                                                 const G1& H, const std::string& canonical_basis_label,
                                                 const std::string& canonical_basis_id, std::uint64_t full_domain) {
    return make_native_proof_statement_with_basis_digest(
        context, com_aux, com_L, static_cast<std::size_t>(full_domain),
        NativeProofStatement::basis_descriptor_digest_for(
            H, canonical_basis_label, canonical_basis_id, kCanonicalGeneratorDerivationId,
            full_domain));
}

GkrOnlyArtifactResult prove_gkr_only_artifact(prover& native_prover, const VerifierContext& context, const CommitmentWire& com_aux, const CommitmentWire& com_L,
                                              const G1& G, const std::vector<G1>& generators, const std::vector<Fr>& basis, std::size_t l_dimension,
                                              const std::vector<F>& public_output, const std::vector<NativeGkrMatmulShape>& matmuls) {
    GkrOnlyArtifactResult result;
    try {
        if (public_output.empty() || public_output.size() != context.public_logits.values.size()) {
            throw std::invalid_argument("GKR diagnostic output does not match the public Y shape");
        }
        for (std::size_t index = 0; index < public_output.size(); ++index) {
            if (public_output[index] != Fr(context.public_logits.values[index])) {
                throw std::invalid_argument("GKR diagnostic output is not bound to public Y");
            }
        }
        const zkgpt::NativeFieldView public_output_view =
            zkgpt::NativeFieldView::from_vector(public_output);
        if (native_prover.C.size < 2 || native_prover.C.circuit.empty() || native_prover.C.circuit.back().size != public_output_view.size()) {
            throw std::invalid_argument("GKR diagnostic circuit output does not match public Y");
        }
        if (!com_aux.valid() || !com_L.valid()) { throw std::invalid_argument("GKR diagnostic commitment wire is invalid"); }

        const NativeProofStatement statement = make_native_proof_statement(context, com_aux, com_L, G, generators, basis, l_dimension);
        NativeProofSession session(kNativeArtifactDomain, statement);

        session.absorb("adapter/context", statement.verifier_context_digest);
        session.absorb("adapter/com_aux", statement.aux_digest);

        verifier native_verifier(&native_prover, native_prover.C);
        if (!native_verifier.prove_gkr_with_session(session)) { throw std::runtime_error("native GKR prover/verifier loop rejected the synthetic relation"); }

        const auto public_output_digest = [&public_output_view]() {
            std::string encoded;
            append_string(encoded, "zkgpt-native-public-output-v1");
            append_u64(encoded, static_cast<std::uint64_t>(public_output_view.size()));
            for (std::size_t index = 0; index < public_output_view.size(); ++index)
                append_string(encoded, field_hex(public_output_view.at(index)));
            return fixed_digest(hash_hex(encoded));
        }();
        session.absorb("adapter/public_output", public_output_digest);
        session.absorb("adapter/com_L", statement.l_digest);

        std::string reason;
        if (!verify_gkr_transcript(session, native_prover.C, public_output_view, matmuls, &reason)) {
            throw std::runtime_error("independent native GKR transcript replay rejected: " + reason);
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

        const auto copy_message = [](const NativeProofMessage& message) {
            TranscriptMessage summary;
            summary.label = message.label;
            summary.payload = message.payload;
            summary.kind = static_cast<std::uint8_t>(message.kind);
            summary.sequence = message.sequence;
            return summary;
        };
        const std::vector<NativeProofMessage>& messages = session.messages();
        if (messages.size() < 4) { throw std::logic_error("native GKR session did not contain the adapter binding messages"); }
        artifact.Pi_Lasso.messages.push_back(copy_message(messages[0]));
        artifact.Pi_MatMul.messages.push_back(copy_message(messages[1]));
        artifact.Pi_Lasso.protocol = "Pi_Lasso";
        artifact.Pi_MatMul.protocol = "Pi_MatMul";
        artifact.Pi_GKR.protocol = "Pi_GKR";
        artifact.Pi_combine.protocol = "Pi_combine";
        artifact.Pi_open.protocol = "Pi_open";
        for (std::size_t index = 2; index + 2 < messages.size(); ++index) artifact.Pi_GKR.messages.push_back(copy_message(messages[index]));
        artifact.Pi_combine.messages.push_back(copy_message(messages[messages.size() - 2]));
        artifact.Pi_open.messages.push_back(copy_message(messages[messages.size() - 1]));

        if (!artifact.valid_wire(&reason)) { throw std::runtime_error("GKR diagnostic artifact failed canonical wire validation: " + reason); }
        result.generated = true;
        result.artifact = std::move(artifact);
        result.diagnostic = "gkr_only_diagnostic: native GKR transcript generated; masked_full remains unsupported";
        return result;
    } catch (const std::exception& error) {
        result.generated = false;
        result.diagnostic = error.what();
        return result;
    }
}

const char* prove_status_name(ProveStatus status) {
    switch (status) {
    case ProveStatus::GENERATED:
        return "generated";
    case ProveStatus::UNSUPPORTED_PRIVACY_PROFILE:
        return "unsupported_privacy_profile";
    case ProveStatus::REJECTED:
        return "rejected";
    }
    return "rejected";
}

ProveResult prove_gat(const GatCircuitPlan& plan, const Graph& graph, const Tensor2D& public_features, const Model& model, const QuantConfig& quant,
                      const ForwardResult& forward, const CommitmentWire& com_W, const std::string& privacy_profile) {
    ProveResult result;
    result.status = ProveStatus::REJECTED;
    result.diagnostic = "bound model registration is required by the production prover";
    return result;
}

static ProveResult prove_gat_impl(const GatCircuitPlan& plan, std::shared_ptr<const GatCircuitPlan> shared_plan,
                                  const VerifierContext* provided_context, const Graph& graph,
                                  const Tensor2D& public_features, const Model& model, const QuantConfig& quant,
                                  const ForwardResult& forward, const ModelRegistration& registration,
                                  const std::string& privacy_profile,
                                  const detail::ValidatedFormalInputs* validated_owner = nullptr) {
    ProveResult result;
    const auto preparation_begin = std::chrono::steady_clock::now();
    try {
        const auto stage = [](const char* name) { std::cerr << "gat_adapter_prover_stage: " << name << '\n'; };
        stage("begin");
        graph.validate();
        model.validate_schema();
        quant.validate();
        if (validated_owner != nullptr &&
            (&graph != &validated_owner->graph() ||
             &public_features != &validated_owner->features() ||
             &model != &validated_owner->model() ||
             &quant != &validated_owner->quant() ||
             &forward != &validated_owner->forward() ||
             !shared_plan || shared_plan.get() != &validated_owner->plan()))
            throw std::invalid_argument("trusted formal proof inputs do not match their move-owned validated context");
        if (provided_context != nullptr) {
            if (!shared_plan || provided_context->shared_plan != shared_plan ||
                &provided_context->plan_ref() != shared_plan.get())
                throw std::invalid_argument("prover context does not share the supplied immutable plan owner");
            std::string context_reason;
            if (!validate_verifier_context_object(*provided_context, &context_reason))
                throw std::invalid_argument("prover context is invalid: " + context_reason);
            if (provided_context->registration.binding_digest != registration.binding_digest ||
                provided_context->registration.registration_digest != registration.registration_digest ||
                provided_context->graph.num_nodes != graph.num_nodes ||
                provided_context->graph.num_features != graph.num_features ||
                provided_context->graph.graph_count != graph.graph_count ||
                provided_context->graph.num_classes != graph.num_classes ||
                provided_context->graph.src != graph.src || provided_context->graph.dst != graph.dst ||
                provided_context->graph.edge_gid != graph.edge_gid ||
                provided_context->public_features.rows != public_features.rows ||
                provided_context->public_features.cols != public_features.cols ||
                provided_context->public_features.values != public_features.values ||
                provided_context->public_logits.rows != forward.logits.rows ||
                provided_context->public_logits.cols != forward.logits.cols ||
                provided_context->public_logits.values != forward.logits.values ||
                canonical_quant_digest(provided_context->quant) != canonical_quant_digest(quant) ||
                provided_context->plan_ref().plan_digest != plan.plan_digest)
                throw std::invalid_argument("prover context does not match the supplied public statement and plan");
        }
        std::unique_ptr<VerifierContext> owned_context;
        const auto proof_context = [&](const StaticTables& tables) -> const VerifierContext& {
            if (provided_context != nullptr) return *provided_context;
            if (!owned_context) {
                if (shared_plan)
                    owned_context = std::make_unique<VerifierContext>(make_verifier_context(
                        graph, public_features, quant, tables, forward.logits, registration, shared_plan, privacy_profile));
                else
                    owned_context = std::make_unique<VerifierContext>(make_verifier_context(
                        graph, public_features, quant, tables, forward.logits, registration, plan, privacy_profile));
            }
            return *owned_context;
        };
        if (!plan.official_layered || plan.has_placeholder) throw std::invalid_argument("official layered circuit plan is incomplete");
        if (!registration_matches_model(model, quant, plan.input_layout, registration))
            throw std::invalid_argument("private model does not match the bound model registration");
        if (validated_owner == nullptr &&
            !validate_relation(graph, public_features, model, quant, forward))
            throw std::invalid_argument("canonical GAT relation is invalid");
        if (plan.virtual_sparse) throw std::invalid_argument("virtual sparse proof is unsupported: canonical GAT arithmetic and registered W/Aux/L commitment binding are incomplete");
        if (plan.virtual_sparse) {
            if (privacy_profile != "masked_full") {
                result.status = ProveStatus::UNSUPPORTED_PRIVACY_PROFILE;
                result.diagnostic = "virtual sparse path only accepts the masked_full public profile";
                return result;
            }
            const StaticTables tables = StaticTables::build(quant);
            if (!tables.authenticate()) throw std::invalid_argument("canonical static tables are not authenticated");
            const VerifierContext& context = proof_context(tables);
            const SparseGkrPlan sparse_plan = sparse_plan_from_virtual_recipe(graph, plan);
            const std::string sparse_basis_label = "zkgpt/gat-sparse-proof/v1/" + verifier_context_digest(context);
            const zkgpt::native_pedersen::PedersenBasis wire_basis = zkgpt::native_pedersen::derive_basis(sparse_basis_label);
            const CommitmentWire com_aux = sparse_wire_from_opening(zkgpt::native_pedersen::commit_fresh(wire_basis, F_ZERO));
            const CommitmentWire com_L = sparse_wire_from_opening(zkgpt::native_pedersen::commit_fresh(wire_basis, F_ZERO));
            std::vector<G1> statement_generators{wire_basis.G};
            NativeProofStatement statement =
                make_native_proof_statement(context, com_aux, com_L, wire_basis.G, statement_generators, std::vector<F>{F_ZERO}, 1);
            statement.relation_schema = kSparseGkrRelationSchema;
            statement.relation_digest = plan.sparse_plan_digest;
            for (const SparseGkrRelationSpec& relation : sparse_plan.relations) statement.relation_order.push_back(relation.relation_id);
            std::string statement_reason;
            if (!statement.valid(&statement_reason)) throw std::invalid_argument("sparse GKR native statement is invalid: " + statement_reason);
            NativeProofSession session(kNativeArtifactDomain, statement);
            SparseRelationValueProvider provider(graph, model, quant, forward, plan);
            SparseProverRuntime runtime{provider, zkgpt::native_pedersen::derive_basis("zkgpt/masked-sumcheck/v2/" + session.statement_digest())};
            const SparseGkrProverCallbacks callbacks = make_sparse_prover_callbacks(runtime);
            std::vector<SparseGkrRelationResult> relation_results;
            std::string sparse_reason;
            if (!prove_sparse_gkr_plan(session, graph, sparse_plan, callbacks, &relation_results, &sparse_reason))
                throw std::runtime_error("sparse GKR proof failed: " + sparse_reason);
            if (relation_results.size() != sparse_plan.relations.size() || relation_results.empty() ||
                relation_results.front().sumcheck.terminal_openings.empty() || relation_results.back().sumcheck.terminal_openings.empty())
                throw std::runtime_error("sparse GKR terminal opening chain is incomplete");

            const SparseRelationRuntime& input_runtime = runtime.relations.at("binding/input_segments");
            const std::size_t input_bits = 0 < plan.input_layout.n_in ? static_cast<std::size_t>(std::log2(plan.input_layout.n_in)) : 0;
            if (input_runtime.prefix.size() < input_bits + 2) throw std::runtime_error("sparse InputBinding point is incomplete");
            const std::vector<F> input_point(input_runtime.prefix.begin(), input_runtime.prefix.begin() + input_bits);
            const F selector_s2 = input_runtime.prefix[input_bits];
            const F selector_s1 = input_runtime.prefix[input_bits + 1];
            const std::array<F, 4> segment_coefficients = {(F_ONE - selector_s1) * (F_ONE - selector_s2), (F_ONE - selector_s1) * selector_s2,
                                                           selector_s1 * (F_ONE - selector_s2), selector_s1 * selector_s2};
            const std::uint64_t input_chunk_size =
                plan.input_layout.n_in >= kChunkedMleEntries ? kChunkedMleEntries : std::max<std::uint64_t>(2, plan.input_layout.n_in);
            std::vector<hyrax::ChunkedMleWire> input_segment_wires;
            std::vector<zkgpt::native_pedersen::ScalarOpening> input_segment_openings;
            input_segment_wires.reserve(4);
            input_segment_openings.reserve(4);
            for (std::size_t segment = 0; segment < 4; ++segment) {
                const InputSegment& metadata = plan.input_layout.segments.at(segment);
                hyrax::ChunkedMleProverResult opening;
                if (!hyrax::prove_chunked_mle_opening(
                        session, "input/chunked/" + metadata.name, metadata.logical_length, metadata.padded_length, input_chunk_size, input_point,
                        [&](std::uint64_t index) { return provider.input_oracle.segment_value(segment, static_cast<std::size_t>(index)); }, runtime.basis,
                        opening, &sparse_reason))
                    throw std::runtime_error("chunked input opening failed: " + sparse_reason);
                input_segment_wires.push_back(opening.wire);
                input_segment_openings.push_back(opening.aggregate_opening);
            }
            std::vector<zkgpt::native_pedersen::ScalarOpening> input_link_operands = input_segment_openings;
            std::vector<F> input_link_coefficients(segment_coefficients.begin(), segment_coefficients.end());
            input_link_operands.push_back(relation_results.front().sumcheck.terminal_openings.front());
            input_link_coefficients.push_back(-F_ONE);
            if (!zkgpt::native_pedersen::prove_hidden_linear(session, runtime.basis, "input/chunked/terminal-link", input_link_operands,
                                                             input_link_coefficients, F_ZERO, nullptr, &sparse_reason))
                throw std::runtime_error("chunked input terminal link failed: " + sparse_reason);

            const SparseGkrRelationSpec& output_relation = sparse_plan.relations.back();
            const SparseRelationRuntime& output_runtime = runtime.relations.at(output_relation.relation_id);
            std::uint64_t output_entries = 0;
            if (!sparse_gkr_checked_mul(output_relation.item_count, std::max<std::uint64_t>(1, output_relation.value_width), &output_entries))
                throw std::runtime_error("output chunk domain overflows u64");
            std::uint64_t output_padded = 0;
            if (!sparse_gkr_next_power_of_two(output_entries, &output_padded)) throw std::runtime_error("output chunk domain is not representable");
            const std::uint64_t output_chunk_size = output_padded >= kChunkedMleEntries ? kChunkedMleEntries : std::max<std::uint64_t>(2, output_padded);
            hyrax::ChunkedMleProverResult output_opening;
            if (!hyrax::prove_chunked_mle_opening(
                    session, "output/chunked/logits", output_entries, output_padded, output_chunk_size, output_runtime.prefix,
                    [&](std::uint64_t index) { return provider.value(output_relation, index); }, runtime.basis, output_opening, &sparse_reason))
                throw std::runtime_error("chunked output opening failed: " + sparse_reason);
            std::vector<zkgpt::native_pedersen::ScalarOpening> output_link_operands = {output_opening.aggregate_opening,
                                                                                       relation_results.back().sumcheck.terminal_openings.front()};
            if (!zkgpt::native_pedersen::prove_hidden_linear(session, runtime.basis, "output/chunked/terminal-link", output_link_operands, {F_ONE, -F_ONE},
                                                             F_ZERO, nullptr, &sparse_reason))
                throw std::runtime_error("chunked output terminal link failed: " + sparse_reason);

            std::vector<NativeLassoChunkFamilyProver> lasso_families = build_lasso_chunk_families(plan, forward, tables);
            NativeLassoChunkBundleProverResult lasso_chunks;
            if (!native_lasso_prove_chunked_bundle(session, lasso_families, runtime.basis, &input_point, &lasso_chunks, &sparse_reason))
                throw std::runtime_error("chunked Lasso proof failed: " + sparse_reason);

            if (!zkgpt::native_pedersen::prove_hidden_linear(session, runtime.basis, "lasso/chunked/input-link",
                                                             {lasso_chunks.final_opening, input_segment_openings[3]}, {F_ONE, -F_ONE}, F_ZERO, nullptr,
                                                             &sparse_reason))
                throw std::runtime_error("chunked Lasso input terminal link failed: " + sparse_reason);
            GatProofArtifact artifact;
            artifact.privacy_profile = "masked_full";
            artifact.statement_digest = verifier_context_digest(context);
            artifact.com_aux = com_aux;
            artifact.com_L = com_L;
            artifact.native_domain = kNativeArtifactDomain;
            artifact.native_statement = statement;
            artifact.sparse_gkr = make_sparse_proof_wire(plan, sparse_plan, relation_results, runtime, session);
            artifact.sparse_gkr.hyrax_lasso_complete = true;
            artifact.sparse_gkr.lasso_chunks = std::move(lasso_chunks.wire);
            artifact.sparse_gkr.input_segment_openings = std::move(input_segment_wires);
            artifact.sparse_gkr.output_opening = std::move(output_opening.wire);
            artifact.sparse_gkr.input_terminal_commitment = point_hex(relation_results.front().sumcheck.terminal_openings.front().commitment.point);
            artifact.sparse_gkr.output_terminal_commitment = point_hex(relation_results.back().sumcheck.terminal_openings.front().commitment.point);
            artifact.native_messages = session.messages();
            artifact.transcript_digest = session.digest();
            core_copy_sections(session, artifact);
            if (!artifact.valid_wire(&sparse_reason)) throw std::runtime_error("sparse GKR artifact is invalid: " + sparse_reason);
            result.status = ProveStatus::GENERATED;
            result.has_artifact = true;
            result.artifact = std::move(artifact);
            result.diagnostic = "sparse GKR plus chunked Hyrax/Lasso artifact generated and independently consumable";
            return result;
        }
        if (privacy_profile != "masked_full") {
            result.status = ProveStatus::UNSUPPORTED_PRIVACY_PROFILE;
            result.diagnostic = "unsupported_privacy_profile: zkGPT formal adapter only exposes masked_full";
            return result;
        }
        if (!registration.has_private_padding_mask) throw std::invalid_argument("com_W prover secret is unavailable in this process");

        const StaticTables tables = StaticTables::build(quant);
        if (!tables.authenticate()) throw std::invalid_argument("canonical static tables are not authenticated");
        const VerifierContext& context = proof_context(tables);
        const std::string basis_label = registration_basis_label(model, quant, plan.input_layout, registration.model_arch_id);
        const std::string basis_id = fixed_digest(hash_hex(basis_label));
        const std::size_t domain = plan.input_layout.n_in;
        std::vector<NativeGkrMatmulShape> point_domain_shapes;
        point_domain_shapes.reserve(plan.matmuls.size());
        for (const MatMulDescriptor &descriptor : plan.matmuls)
            point_domain_shapes.push_back(
                native_gkr_shape_from_descriptor(descriptor, plan.input_layout));
        std::string point_domain_reason;
        if (!validate_native_gkr_point_domains(
                plan.circuit, point_domain_shapes,
                plan.circuit.circuit.empty()
                    ? 0
                    : static_cast<std::size_t>(plan.circuit.circuit.front().size),
                &point_domain_reason))
            throw std::invalid_argument(
                "canonical GKR point/domain admission failed before Lasso: " +
                point_domain_reason);
        stage("lasso-build");
        LassoAdapterBundle lasso = build_lasso_bundle(plan, forward, tables, basis_label);
        result.initial_commitment_blinding_prepare_ms +=
            lasso.query_blinding_prepare_ms;
        result.lasso_family_generator_derive_ms =
            lasso.family_generator_derive_ms;
        GatCircuitWitness circuit_witness = validated_owner != nullptr
            ? build_gat_circuit_witness_impl(
                  plan, graph, public_features, model, quant, forward,
                  lasso.query.value_provider, true, false, true)
            : build_gat_circuit_witness_borrowed_aux(
                  plan, graph, public_features, model, quant, forward,
                  lasso.query.value_provider, false);
        stage("witness");
        if (validated_owner == nullptr) {
            std::string witness_reason;
            const zkgpt::NativeFieldView validation_terminal_output =
                canonical_public_output_view(plan, forward.logits, &circuit_witness);
            if (!evaluate_gat_circuit_witness_with_terminal(
                    plan, circuit_witness, validation_terminal_output, &witness_reason))
                throw std::invalid_argument("canonical GAT witness validation failed: " + witness_reason);
        }

        const InputSegment& aux_segment = plan.input_layout.segments.at(2);
        const hyrax::HyraxTensorMleView aux_commit_view =
            canonical_input_tensor_view(basis_label, basis_id, "canonical/Aux",
                                        domain, aux_segment.logical_length);
        const hyrax::HyraxScalarProvider aux_commit_values =
            canonical_input_tensor_scalar_provider(
                *circuit_witness.input_oracle, 2, aux_segment.logical_length);
        if (aux_commit_view.rows > std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("canonical Aux row blinding extent exceeds size_t");
        const auto aux_blinding_prepare_begin =
            std::chrono::steady_clock::now();
        std::vector<Fr> aux_commitment_blindings(
            static_cast<std::size_t>(aux_commit_view.rows));
        for (Fr& blinding : aux_commitment_blindings)
            blinding.setByCSPRNG();
        result.initial_commitment_blinding_prepare_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - aux_blinding_prepare_begin).count();

        result.preparation_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - preparation_begin).count();

        stage("generators");
        const G1 H = hyrax::derive_pedersen_blinding_generator(basis_id);
        hyrax::HyraxTensorMleCommitment aux_tensor_commitment;
        hyrax::HyraxTensorMleProverState aux_tensor_state;
        const auto aux_commitment_begin = std::chrono::steady_clock::now();
        std::string aux_commitment_reason;
        if (!hyrax::commit_hyrax_tensor_mle_with_blindings(
                aux_commit_view, aux_commit_values, aux_tensor_commitment,
                aux_tensor_state, std::move(aux_commitment_blindings),
                &aux_commitment_reason))
            throw std::runtime_error("canonical Aux tensor commitment failed: " +
                                     aux_commitment_reason);
        result.aux_commitment_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - aux_commitment_begin).count();
        CommitmentWire com_aux;
        com_aux.scheme = hyrax::kHyraxTensorMleSchema;
        com_aux.basis_id = basis_id;
        com_aux.logical_size = plan.input_layout.segments.at(2).logical_length;
        com_aux.padded_size = domain;
        com_aux.has_tensor_commitment = true;
        com_aux.tensor_commitment = aux_tensor_commitment;

        result.lasso_query_commitment_ms = lasso.commitment_ms;
        result.commitment_ms = result.aux_commitment_ms +
            result.lasso_query_commitment_ms +
            result.lasso_family_generator_derive_ms;
        const std::vector<NativeLassoFamilyPublic> lasso_public = build_lasso_public_families(plan, tables, basis_label, lasso.families.get());
        NativeLassoQueryPublic lasso_query_public = build_lasso_public_query(plan, lasso.com_L, basis_label, &lasso.query);
        const std::string lasso_tables_digest = native_lasso_tables_digest(lasso_public);
        const std::string lasso_resources_digest = native_lasso_resources_digest(lasso_public);
        const std::string lasso_digest = native_lasso_tensor_query_digest(
            lasso.query.tensor_view, lasso.query.tensor_commitment);
        for (NativeLassoFamilyProver& family : *lasso.families) {
            family.statement_tables_digest = lasso_tables_digest;
            family.statement_resources_digest = lasso_resources_digest;
        }
        lasso.query.statement_l_digest = lasso_digest;
        lasso_query_public.statement_l_digest = lasso_digest;

        NativeProofStatement statement = make_native_proof_statement(context, com_aux, lasso.com_L, H, basis_label, basis_id, domain);

        statement.tables_digest = lasso_tables_digest;
        statement.resources_digest = lasso_resources_digest;
        statement.l_digest = lasso_digest;
        std::string statement_reason;
        if (!statement.valid(&statement_reason)) throw std::invalid_argument("masked GAT statement is invalid: " + statement_reason);
        NativeProofSession session(kNativeArtifactDomain, statement);
        std::string reason;
        const zkgpt::native_pedersen::PedersenBasis scalar_basis =
            zkgpt::native_pedersen::derive_basis(
                "zkgpt/masked-sumcheck/v2/" + session.statement_digest());
        const std::vector<hyrax::HyraxTensorZeroInterval> l_query_zero_intervals =
            canonical_l_query_zero_intervals(plan);
        const NativeLassoTensorTailProveHook tensor_tail_stage =
            [&aux_commit_view, &aux_tensor_commitment, &aux_tensor_state,
             &aux_commit_values, &scalar_basis, &lasso_digest,
             l_query_zero_intervals](
                NativeProofSession& proof_session,
                const NativeLassoQueryProver& query,
                std::string* tail_reason) {
                if (proof_session.statement_digest().empty() ||
                    query.tensor_view.role != "canonical/L_qry" ||
                    native_lasso_tensor_query_digest(query.tensor_view,
                                                     query.tensor_commitment) !=
                        lasso_digest) {
                    set_reason(tail_reason,
                               "shared tensor-tail hook received an unbound L_qry root");
                    return false;
                }
                std::size_t exact_position = proof_session.checkpoint();
                std::vector<Fr> beta;
                std::vector<Fr> terminal_point;
                NativeMaskedSumcheckResult sumcheck;
                hyrax::MaskedMleOpening terminal_opening;
                if (!hyrax::prove_hyrax_tensor_zero_tail_certificate_with_session(
                        aux_commit_view, aux_tensor_commitment, aux_tensor_state,
                        aux_commit_values, proof_session, exact_position,
                        scalar_basis, beta, terminal_point, sumcheck,
                        terminal_opening, tail_reason))
                    return false;
                beta.clear();
                terminal_point.clear();
                sumcheck = NativeMaskedSumcheckResult{};
                terminal_opening = hyrax::MaskedMleOpening{};
                if (!hyrax::prove_hyrax_tensor_zero_tail_certificate_with_session(
                        query.tensor_view, query.tensor_commitment,
                        query.tensor_state, query.value_provider, proof_session,
                        exact_position, scalar_basis, beta, terminal_point,
                        sumcheck, terminal_opening, tail_reason))
                    return false;
                beta.clear();
                terminal_point.clear();
                sumcheck = NativeMaskedSumcheckResult{};
                std::vector<hyrax::HyraxTensorZeroIntervalCertificate> gap_certificates;
                if (!hyrax::prove_hyrax_tensor_zero_interval_certificates_with_session(
                        query.tensor_view, query.tensor_commitment,
                        query.tensor_state, query.value_provider,
                        l_query_zero_intervals, proof_session, exact_position,
                        scalar_basis, gap_certificates, tail_reason))
                    return false;
                return exact_position == proof_session.checkpoint();
            };
        NativeLassoBundleResult lasso_result;
        zkgpt::NativeFieldView terminal_output;
        stage("lasso-prove");
        const auto restore_native_plan = [&]() {
            if (plan.native_proof_backing_is_resident()) return;
            stage("restore-native-plan");
            GatCircuitPlan rebuilt = validated_owner != nullptr
                ? detail::rebuild_formal_plan_for_restore(*validated_owner)
                : build_gat_circuit_plan(graph, model, quant, forward);
            plan.restore_native_proof_backing(std::move(rebuilt));
        };
        plan.release_native_proof_backing();
        try {
            double family_initial_blinding_prepare_ms = 0.0;
            const bool lasso_ok = native_lasso_prove_bundle(
                session, *lasso.families, lasso.query, &lasso_result, &reason,
                tensor_tail_stage, &result.lasso_family_commitment_ms,
                &family_initial_blinding_prepare_ms,
                &result.lasso_family_selected_row_mapping_ms);
            result.initial_commitment_blinding_prepare_ms +=
                family_initial_blinding_prepare_ms;
            result.preparation_ms += family_initial_blinding_prepare_ms;
            result.commitment_ms = result.aux_commitment_ms +
                result.lasso_query_commitment_ms +
                result.lasso_family_generator_derive_ms +
                result.lasso_family_commitment_ms;
            lasso.families.reset();
            if (!lasso_ok)
                throw std::runtime_error("native masked Lasso proof failed: " + reason);
            if (!lasso_result.has_final_l_commitment ||
                !lasso_result.final_l_commitment.valid(zkgpt::native_pedersen::derive_basis("zkgpt/masked-sumcheck/v2/" + session.statement_digest()), &reason) ||
                lasso_result.final_query_point.empty())
                throw std::runtime_error("native masked Lasso did not return an authenticated final L_qry binding");
        } catch (const std::exception& prove_error) {
            lasso.families.reset();
            try {
                restore_native_plan();
            } catch (const std::exception& restore_error) {
                throw std::runtime_error(std::string("native Lasso proof failed and canonical GAT plan restoration also failed; proof: ") +
                                         prove_error.what() + "; restore: " + restore_error.what());
            }
            throw;
        } catch (...) {
            lasso.families.reset();
            try {
                restore_native_plan();
            } catch (const std::exception& restore_error) {
                throw std::runtime_error(std::string("native Lasso proof failed with an unknown error and canonical GAT plan restoration also failed: ") +
                                         restore_error.what());
            }
            throw;
        }
        restore_native_plan();
        terminal_output = canonical_public_output_view(plan, forward.logits, &circuit_witness);

        NativeLayeredGkrWitness gkr_witness;
        gkr_witness.layer_values_view = &circuit_witness.values;
        gkr_witness.terminal_output = terminal_output;
        auto input_provider = std::make_shared<NativeGkrInputProvider>();
        input_provider->full_size = circuit_witness.input_layer_size;
        input_provider->segment_domain = plan.input_layout.n_in;
        input_provider->segment_support = canonical_gkr_input_support(plan);
        input_provider->support_complete = true;
        for (std::size_t segment = 0; segment < 4; ++segment)
            input_provider->logical_lengths[segment] =
                plan.input_layout.segments.at(segment).logical_length;
        input_provider->value_at = [input_oracle = circuit_witness.input_oracle](
                                       std::uint64_t coordinate) {
            return input_oracle->value_at(static_cast<std::size_t>(coordinate));
        };
        input_provider->value_at_read_only_thread_safe = true;
        gkr_witness.input_provider = input_provider;
        gkr_witness.matmuls.reserve(plan.matmuls.size());
        for (const MatMulDescriptor& descriptor : plan.matmuls) {
            NativeGkrMatmulWitness matmul;
            matmul.shape = native_gkr_shape_from_descriptor(descriptor, plan.input_layout);
            const std::size_t lhs_size = checked_mul(descriptor.rows, descriptor.inner, "native GKR MatMul LHS size");
            const std::size_t rhs_size = checked_mul(descriptor.inner, descriptor.cols, "native GKR MatMul RHS size");
            if (descriptor.output_coordinates != checked_mul(descriptor.rows, descriptor.cols, "native GKR MatMul output size"))
                throw std::invalid_argument("native GKR MatMul descriptor output shape is inconsistent");
            if (descriptor.lhs_offset > circuit_witness.input_layer_size || lhs_size > circuit_witness.input_layer_size - descriptor.lhs_offset)
                throw std::invalid_argument("native GKR MatMul LHS span exceeds canonical input layer");
            if (descriptor.rhs_offset > circuit_witness.input_layer_size || rhs_size > circuit_witness.input_layer_size - descriptor.rhs_offset)
                throw std::invalid_argument("native GKR MatMul RHS span exceeds canonical input layer");
            matmul.left_value_at = [provider = input_provider,
                                    offset = static_cast<std::uint64_t>(descriptor.lhs_offset),
                                    size = static_cast<std::uint64_t>(lhs_size)](
                                       std::uint64_t index) {
                if (index >= size)
                    throw std::out_of_range("native GKR MatMul LHS provider index is invalid");
                return provider->value_at(offset + index);
            };
            const std::uint64_t inner_domain = next_power_of_two(descriptor.inner);
            const std::uint64_t cols_domain = next_power_of_two(descriptor.cols);
            const std::uint64_t padded_rhs_size = checked_mul(
                inner_domain, cols_domain, "native GKR MatMul padded RHS size");
            matmul.right_value_at = [provider = input_provider,
                                     offset = static_cast<std::uint64_t>(descriptor.rhs_offset),
                                     shape = matmul.shape, inner_domain,
                                     padded_rhs_size](std::uint64_t index) {
                if (index >= padded_rhs_size)
                    throw std::out_of_range("native GKR MatMul RHS provider index is invalid");
                const std::uint64_t col = index / inner_domain;
                const std::uint64_t term = index % inner_domain;
                if (col >= shape.cols || term >= shape.inner) return Fr(0);
                return provider->value_at(offset + term * shape.cols + col);
            };
            gkr_witness.matmuls.push_back(std::move(matmul));
        }
        NativeGkrInputBinding gkr_binding;
        NativeGkrProverResult gkr_result;
        stage("gkr-prove");
        if (!prove_masked_gkr_transcript(session, plan.circuit, gkr_witness, terminal_output, &reason, &gkr_binding, &gkr_result))
            throw std::runtime_error("native masked GKR proof failed: " + reason);

        if (gkr_binding.point.size() != core_log2(domain) + 2 || !gkr_binding.claim_commitment.valid(scalar_basis, &reason) ||
            !gkr_binding.unmasked_claim_commitment.valid(scalar_basis, &reason) || !gkr_binding.mask_commitment.valid(scalar_basis, &reason) ||
            !gkr_binding.unmasked_input_value_commitment.valid(scalar_basis, &reason) ||
            !gkr_binding.masked_input_value_commitment.valid(scalar_basis, &reason) || !gkr_result.has_unmasked_input_claim_opening ||
            !gkr_result.unmasked_input_claim_opening.valid(scalar_basis, &reason) ||
            gkr_result.unmasked_input_claim_opening.commitment.point != gkr_binding.unmasked_claim_commitment.point ||
            !gkr_result.has_unmasked_input_value_opening || !gkr_result.unmasked_input_value_opening.valid(scalar_basis, &reason) ||
            gkr_result.unmasked_input_value_opening.commitment.point != gkr_binding.unmasked_input_value_commitment.point)
            throw std::runtime_error("native masked GKR input binding is malformed");

        const std::size_t r_dimension = core_log2(domain);
        stage("input-openings");
        const std::vector<Fr> r_minus(gkr_binding.point.begin(), gkr_binding.point.begin() + r_dimension);

        const Fr selector_s1 = gkr_binding.point[r_dimension + 1];
        const Fr selector_s2 = gkr_binding.point[r_dimension];
        session.absorb_fields("combine/r_minus", r_minus);
        session.absorb_fields("combine/selectors", {selector_s1, selector_s2});
        const Fr x_eval = circuit_witness.input_oracle->evaluate_mle(0, r_minus);
        session.absorb_field("combine/public-X", x_eval);
        session.absorb("combine/structured-root", statement.structured_root());
        session.absorb("combine/commitments",
                       basis_id + "|" + commitment_encoding(registration.com_W) + "|" + commitment_encoding(com_aux) + "|" + commitment_encoding(lasso.com_L));
        session.absorb("combine/gkr-input-commitment", scalar_basis.domain + "|" + point_hex(gkr_binding.unmasked_claim_commitment.point));
        session.absorb_field("combine/lasso-final-claim", lasso_result.final_l_evaluation);
        session.absorb("combine/lasso-final-commitment", scalar_basis.domain + "|" + point_hex(lasso_result.final_l_commitment.point));

        const Fr c_x = (Fr(1) - selector_s1) * (Fr(1) - selector_s2);
        const Fr c_w = (Fr(1) - selector_s1) * selector_s2;
        const Fr c_aux = selector_s1 * (Fr(1) - selector_s2);
        const Fr c_l = selector_s1 * selector_s2;
        const InputSegment &w_segment = plan.input_layout.segments.at(1);
        const std::string canonical_generator_id =
            session.statement().basis_digest;
        const hyrax::HyraxMleBasisView w_view = canonical_input_opening_view(
            basis_label, basis_id, canonical_generator_id, "canonical/W",
            domain, w_segment.logical_length, r_minus);
        const hyrax::HyraxTensorMleView aux_view =
            canonical_input_tensor_view(basis_label, basis_id,
                                        "canonical/Aux", domain,
                                        aux_segment.logical_length);
        const hyrax::HyraxScalarProvider w_values =
            canonical_input_scalar_provider(*circuit_witness.input_oracle, 1,
                                            domain, w_segment.logical_length);
        const hyrax::HyraxScalarProvider aux_open_values =
            canonical_input_tensor_scalar_provider(
                *circuit_witness.input_oracle, 2, aux_segment.logical_length);

        hyrax::PedersenMleCommitment w_commitment;
        w_commitment.commitment.deserializeHexStr(registration.com_W.point_hex);
        w_commitment.blinding_generator = H;
        w_commitment.blinding = registration.com_W_padding_mask;
        w_commitment.basis_id = basis_id;
        hyrax::MaskedMleOpening w_opening;
        hyrax::MaskedMleOpening aux_opening;
        hyrax::MaskedMleOpening l_opening;
        if (!hyrax::open_masked_mle_with_session(w_commitment, w_view, w_values,
                                                 session, "Open_GAT_W", w_opening,
                                                 &reason, true, "", &scalar_basis) ||
            !hyrax::open_hyrax_tensor_mle_with_session(
                aux_view, aux_tensor_commitment, aux_tensor_state, r_minus,
                aux_open_values, session, "Open_GAT_Aux", aux_opening,
                &reason, "", &scalar_basis) ||
            !hyrax::open_hyrax_tensor_mle_with_session(
                lasso.query.tensor_view, lasso.query.tensor_commitment,
                lasso.query.tensor_state, r_minus,
                lasso.query.value_provider, session, "Open_GAT_L",
                l_opening, &reason, "", &scalar_basis))
            throw std::runtime_error("masked input terminal openings failed: " + reason);
        if (!w_opening.has_scalar_binding || !aux_opening.has_scalar_binding || !l_opening.has_scalar_binding)
            throw std::runtime_error("masked input opening omitted scalar binding");
        if (!zkgpt::native_pedersen::prove_hidden_linear(session, scalar_basis, "combine/input-linear",
                                                         {gkr_result.unmasked_input_value_opening, w_opening.raw_evaluation_opening,
                                                          aux_opening.raw_evaluation_opening, l_opening.raw_evaluation_opening},
                                                         {F_ONE, -c_w, -c_aux, -c_l}, c_x * x_eval, nullptr, &reason))
            throw std::runtime_error("masked GAT input linear relation failed: " + reason);

        GatProofArtifact artifact;
        stage("artifact");
        artifact.privacy_profile = "masked_full";
        artifact.statement_digest = verifier_context_digest(context);
        artifact.com_aux = com_aux;
        artifact.com_L = lasso.com_L;
        artifact.native_domain = kNativeArtifactDomain;
        artifact.native_statement = statement;
        artifact.native_messages = session.messages();
        artifact.transcript_digest = session.digest();
        core_copy_sections(session, artifact);
        session = NativeProofSession{};
        std::string verify_reason;
        if (validated_owner != nullptr) {
            // The formal CLI defers full artifact validation to its independent
            // verifier after checking serialized readback integrity. Keep public
            // proof API defaults self-verifying while avoiding this duplicate pass.
            result.self_verify_ms = 0.0;
            stage("selfverify-deferred-to-independent-verifier");
        } else {
            const auto self_verify_begin = std::chrono::steady_clock::now();
            const bool self_verified = verify_gat(context, artifact, &verify_reason);
            result.self_verify_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - self_verify_begin).count();
            if (!self_verified) throw std::runtime_error("formal masked GAT self-verification failed: " + verify_reason);
        }
        result.status = ProveStatus::GENERATED;
        result.has_artifact = true;
        result.artifact = std::move(artifact);
        result.diagnostic = validated_owner != nullptr
            ? "masked_full canonical GAT proof generated; serialized readback integrity and independent verification remain required"
            : "masked_full canonical GAT proof generated and independently verified";
        return result;
    } catch (const std::exception& error) {
        result.status = ProveStatus::REJECTED;
        result.diagnostic = error.what();
        result.has_artifact = false;
        return result;
    }
}

ProveResult prove_gat(const GatCircuitPlan& plan, const Graph& graph, const Tensor2D& public_features,
                      const Model& model, const QuantConfig& quant, const ForwardResult& forward,
                      const ModelRegistration& registration, const std::string& privacy_profile) {
    return prove_gat_impl(plan, {}, nullptr, graph, public_features, model, quant, forward, registration, privacy_profile);
}

ProveResult prove_gat(std::shared_ptr<const GatCircuitPlan> plan, const Graph& graph,
                      const Tensor2D& public_features, const Model& model, const QuantConfig& quant,
                      const ForwardResult& forward, const ModelRegistration& registration,
                      const VerifierContext& context, const std::string& privacy_profile) {
    if (!plan) {
        ProveResult result;
        result.status = ProveStatus::REJECTED;
        result.diagnostic = "shared immutable plan owner is null";
        return result;
    }
    const GatCircuitPlan& plan_ref = *plan;
    return prove_gat_impl(plan_ref, std::move(plan), &context, graph, public_features, model, quant, forward,
                          registration, privacy_profile);
}

ProveResult detail::prove_gat_formal(
    detail::ValidatedFormalInputs& inputs,
    const ModelRegistration& registration, const VerifierContext& context,
    const std::string& privacy_profile) {
    std::shared_ptr<const GatCircuitPlan> plan_owner = inputs.plan_;
    return prove_gat_impl(inputs.plan(), std::move(plan_owner), &context,
                          inputs.graph(), inputs.features(), inputs.model(),
                          inputs.quant(), inputs.forward(), registration,
                          privacy_profile, &inputs);
}

struct SparseVerifierRuntime {
    const SparseGkrProofWire& wire;
    const NativeProofSession& session;
    zkgpt::native_pedersen::PedersenBasis basis;
    F previous_terminal = F_ZERO;
    bool have_previous = false;
    std::string error;

    const SparseGkrRelationWire* find(const std::string& relation_id) const {
        for (const SparseGkrRelationWire& relation : wire.relations)
            if (relation.relation_id == relation_id) return &relation;
        return nullptr;
    }
};

bool sparse_commitment_from_wire(const std::string& encoded, const zkgpt::native_pedersen::PedersenBasis& basis,
                                 zkgpt::native_pedersen::ScalarCommitment& commitment, std::string* reason) {
    G1 point;
    if (!core_decode_point(encoded, point)) {
        set_reason(reason, "sparse GKR commitment point is malformed");
        return false;
    }
    commitment.domain = basis.domain;
    commitment.point = point;
    return commitment.valid(basis, reason);
}

bool sparse_verify_relation_range(const NativeProofSession& session, const std::string& relation_id, std::uint64_t& begin, std::uint64_t& end) {
    const std::string prefix = "sparse/gkr/" + relation_id + "/";
    const std::string challenge_prefix = "challenge/" + prefix;
    bool found = false;
    for (std::size_t index = 0; index < session.messages().size(); ++index) {
        const std::string& label = session.messages()[index].label;
        if (label.compare(0, prefix.size(), prefix) == 0 || label.compare(0, challenge_prefix.size(), challenge_prefix) == 0) {
            if (!found) {
                begin = index;
                found = true;
            }
            end = index + 1;
        }
    }
    return found;
}

bool verify_virtual_sparse_artifact(const VerifierContext& context, const GatProofArtifact& artifact, std::string* reason) {
    try {
        context.graph.validate();
        context.quant.validate();
        if (!context.plan_ref().virtual_sparse || !context.tables.authenticate()) {
            set_reason(reason, "virtual sparse verifier context is not canonical");
            return false;
        }
        if (artifact.privacy_profile != "masked_full" || !artifact.valid_wire(reason)) return false;
        if (artifact.statement_digest != verifier_context_digest(context)) {
            set_reason(reason, "sparse GKR artifact statement digest mismatch");
            return false;
        }
        const SparseGkrPlan sparse_plan = sparse_plan_from_virtual_recipe(context.graph, context.plan_ref());
        if (artifact.sparse_gkr.plan_digest != context.plan_ref().sparse_plan_digest ||
            artifact.sparse_gkr.schedule_digest != canonical_digest64(sparse_plan.encoding())) {
            set_reason(reason, "sparse GKR artifact plan digest mismatch");
            return false;
        }
        const std::string sparse_basis_label = "zkgpt/gat-sparse-proof/v1/" + verifier_context_digest(context);
        const zkgpt::native_pedersen::PedersenBasis wire_basis = zkgpt::native_pedersen::derive_basis(sparse_basis_label);
        NativeProofStatement expected =
            make_native_proof_statement(context, artifact.com_aux, artifact.com_L, wire_basis.G, std::vector<G1>{wire_basis.G}, std::vector<F>{F_ZERO}, 1);
        expected.relation_schema = kSparseGkrRelationSchema;
        expected.relation_digest = context.plan_ref().sparse_plan_digest;
        for (const SparseGkrRelationSpec& relation : sparse_plan.relations) expected.relation_order.push_back(relation.relation_id);
        const NativeProofStatement& actual = artifact.native_statement;
        if (actual.verifier_context_digest != expected.verifier_context_digest || actual.plan_digest != expected.plan_digest ||
            actual.resources_digest != expected.resources_digest || actual.tables_digest != expected.tables_digest ||
            actual.com_w_digest != expected.com_w_digest || actual.aux_digest != expected.aux_digest || actual.l_digest != expected.l_digest ||
            actual.basis_digest != expected.basis_digest || actual.l_dimension != expected.l_dimension ||
            actual.statement_root_digest != expected.statement_root_digest || actual.relation_schema != expected.relation_schema ||
            actual.relation_digest != expected.relation_digest || actual.relation_order != expected.relation_order) {
            set_reason(reason, "sparse GKR native statement relation binding mismatch");
            return false;
        }
        NativeProofSession session;
        if (!artifact.native_session(session, reason)) return false;
        const zkgpt::native_pedersen::PedersenBasis scalar_basis =
            zkgpt::native_pedersen::derive_basis("zkgpt/masked-sumcheck/v2/" + session.statement_digest());
        SparseVerifierRuntime runtime{artifact.sparse_gkr, session, scalar_basis};
        SparseGkrVerifierCallbacks callbacks;
        callbacks.initial_claim = [&runtime](const SparseGkrRelationSpec& relation, F& initial, std::string* callback_reason) {
            const SparseGkrRelationWire* wire = runtime.find(relation.relation_id);
            if (wire == nullptr || !sparse_wire_field(wire->initial_claim, initial, callback_reason)) return false;
            if (runtime.have_previous && initial != runtime.previous_terminal) {
                if (callback_reason != nullptr) *callback_reason = "sparse GKR artifact terminal chain mismatch";
                return false;
            }
            return true;
        };
        callbacks.terminal_spec = [&runtime](const SparseGkrRelationSpec& relation, NativeMaskedTerminalSpec& spec, std::string* callback_reason) {
            const SparseGkrRelationWire* wire = runtime.find(relation.relation_id);
            if (wire == nullptr) {
                if (callback_reason != nullptr) *callback_reason = "sparse GKR relation endpoint is missing";
                return false;
            }
            spec = NativeMaskedTerminalSpec();
            spec.kind = NativeMaskedTerminalKind::Scalar;
            spec.has_external_binding = true;
            spec.public_initial_claim = true;
            return sparse_commitment_from_wire(wire->initial_commitment, runtime.basis, spec.initial_commitment, callback_reason);
        };
        callbacks.relation_finished = [&runtime](const SparseGkrRelationResult& result) {
            const SparseGkrRelationWire* wire = runtime.find(result.relation_id);
            if (wire == nullptr) {
                runtime.error = "sparse GKR relation result is not serialized";
                return;
            }
            F expected_final;
            if (!core_decode_field(wire->final_claim, expected_final) || expected_final != result.sumcheck.final_claim ||
                result.sumcheck.terminal_values.size() != wire->terminal_values.size()) {
                runtime.error = "sparse GKR relation terminal endpoint mismatch";
                return;
            }
            for (std::size_t index = 0; index < wire->terminal_values.size(); ++index) {
                F expected_value;
                if (!core_decode_field(wire->terminal_values[index], expected_value) || expected_value != result.sumcheck.terminal_values[index] ||
                    index >= result.sumcheck.terminal_commitments.size() ||
                    point_hex(result.sumcheck.terminal_commitments[index].point) != wire->terminal_commitments[index]) {
                    runtime.error = "sparse GKR terminal commitment/value mismatch";
                    return;
                }
            }
            std::uint64_t begin = 0, end = 0;
            if (!sparse_verify_relation_range(runtime.session, result.relation_id, begin, end) || begin != wire->transcript_begin ||
                end != wire->transcript_end) {
                runtime.error = "sparse GKR transcript range mismatch";
                return;
            }
            if (!result.sumcheck.terminal_values.empty()) {
                runtime.previous_terminal = result.sumcheck.terminal_values.front();
                runtime.have_previous = true;
            }
        };
        std::vector<SparseGkrRelationResult> verified;
        if (!verify_sparse_gkr_plan(session, context.graph, sparse_plan, callbacks, &verified, reason)) return false;
        if (!runtime.error.empty()) {
            set_reason(reason, runtime.error);
            return false;
        }
        for (std::size_t index = 0; index < sparse_plan.claim_links.size(); ++index) {
            const SparseGkrClaimLink& expected_link = sparse_plan.claim_links[index];
            const SparseGkrLinkWire& actual_link = artifact.sparse_gkr.terminal_links[index];
            if (actual_link.from_relation != expected_link.from_relation || actual_link.to_relation != expected_link.to_relation ||
                actual_link.claim_name != expected_link.claim_name || actual_link.from_terminal != expected_link.from_terminal ||
                actual_link.to_initial != expected_link.to_initial) {
                set_reason(reason, "sparse GKR terminal link order mismatch");
                return false;
            }
        }
        if (artifact.sparse_gkr.binding_links.size() != sparse_plan.binding_links.size()) {
            set_reason(reason, "sparse GKR FCONN binding link count mismatch");
            return false;
        }
        for (std::size_t index = 0; index < sparse_plan.binding_links.size(); ++index) {
            const SparseGkrClaimLink& expected_link = sparse_plan.binding_links[index];
            const SparseGkrLinkWire& actual_link = artifact.sparse_gkr.binding_links[index];
            if (actual_link.from_relation != expected_link.from_relation || actual_link.to_relation != expected_link.to_relation ||
                actual_link.claim_name != expected_link.claim_name || actual_link.from_terminal != expected_link.from_terminal ||
                actual_link.to_initial != expected_link.to_initial) {
                set_reason(reason, "sparse GKR FCONN binding link order mismatch");
                return false;
            }
        }

        const auto find_verified = [&verified](const std::string& relation_id) -> const SparseGkrRelationResult* {
            for (const SparseGkrRelationResult& entry : verified)
                if (entry.relation_id == relation_id) return &entry;
            return nullptr;
        };
        const SparseGkrRelationResult* input_result = find_verified("binding/input_segments");
        const SparseGkrRelationResult* output_result = find_verified("binding/output_aux");
        if (input_result == nullptr || output_result == nullptr || input_result->sumcheck.challenges.empty() || output_result->sumcheck.challenges.empty() ||
            input_result->sumcheck.terminal_commitments.empty() || output_result->sumcheck.terminal_commitments.empty()) {
            set_reason(reason, "sparse GKR input/output verifier terminals are incomplete");
            return false;
        }
        const std::size_t input_bits = core_log2(context.plan_ref().input_layout.n_in);
        if (input_result->sumcheck.challenges.size() < input_bits + 2) {
            set_reason(reason, "sparse GKR input binding point has the wrong dimension");
            return false;
        }
        const std::vector<F> input_point(input_result->sumcheck.challenges.begin(), input_result->sumcheck.challenges.begin() + input_bits);
        const F selector_s2 = input_result->sumcheck.challenges[input_bits];
        const F selector_s1 = input_result->sumcheck.challenges[input_bits + 1];
        const std::array<F, 4> segment_coefficients = {(F_ONE - selector_s1) * (F_ONE - selector_s2), (F_ONE - selector_s1) * selector_s2,
                                                       selector_s1 * (F_ONE - selector_s2), selector_s1 * selector_s2};
        const std::uint64_t input_chunk_size =
            context.plan_ref().input_layout.n_in >= kChunkedMleEntries ? kChunkedMleEntries : std::max<std::uint64_t>(2, context.plan_ref().input_layout.n_in);
        std::vector<zkgpt::native_pedersen::ScalarCommitment> input_commitments;
        input_commitments.reserve(4);
        for (std::size_t segment = 0; segment < 4; ++segment) {
            const InputSegment& metadata = context.plan_ref().input_layout.segments.at(segment);
            const hyrax::ChunkedMleWire& opening = artifact.sparse_gkr.input_segment_openings[segment];
            if (!hyrax::verify_chunked_mle_opening(session, opening, "input/chunked/" + metadata.name, metadata.logical_length, metadata.padded_length,
                                                   input_chunk_size, input_point, runtime.basis, reason))
                return false;
            zkgpt::native_pedersen::ScalarCommitment commitment;
            if (!sparse_commitment_from_wire(opening.aggregate_commitment_hex, runtime.basis, commitment, reason)) return false;
            input_commitments.push_back(commitment);
        }
        if (point_hex(input_result->sumcheck.terminal_commitments.front().point) != artifact.sparse_gkr.input_terminal_commitment) {
            set_reason(reason, "sparse GKR input terminal commitment binding mismatch");
            return false;
        }
        input_commitments.push_back(input_result->sumcheck.terminal_commitments.front());
        if (!zkgpt::native_pedersen::verify_hidden_linear(
                session, runtime.basis, "input/chunked/terminal-link", input_commitments,
                {segment_coefficients[0], segment_coefficients[1], segment_coefficients[2], segment_coefficients[3], -F_ONE}, F_ZERO, reason))
            return false;

        const SparseGkrRelationSpec& output_relation = sparse_plan.relations.back();
        std::uint64_t output_entries = 0;
        if (!sparse_gkr_checked_mul(output_relation.item_count, std::max<std::uint64_t>(1, output_relation.value_width), &output_entries)) {
            set_reason(reason, "output chunk domain overflows u64");
            return false;
        }
        std::uint64_t output_padded = 0;
        if (!sparse_gkr_next_power_of_two(output_entries, &output_padded)) {
            set_reason(reason, "output chunk domain is not representable");
            return false;
        }
        const std::uint64_t output_chunk_size = output_padded >= kChunkedMleEntries ? kChunkedMleEntries : std::max<std::uint64_t>(2, output_padded);
        if (!hyrax::verify_chunked_mle_opening(session, artifact.sparse_gkr.output_opening, "output/chunked/logits", output_entries, output_padded,
                                               output_chunk_size, output_result->sumcheck.challenges, runtime.basis, reason))
            return false;
        zkgpt::native_pedersen::ScalarCommitment output_commitment;
        if (!sparse_commitment_from_wire(artifact.sparse_gkr.output_opening.aggregate_commitment_hex, runtime.basis, output_commitment, reason)) return false;
        if (point_hex(output_result->sumcheck.terminal_commitments.front().point) != artifact.sparse_gkr.output_terminal_commitment) {
            set_reason(reason, "sparse GKR output terminal commitment binding mismatch");
            return false;
        }
        if (!zkgpt::native_pedersen::verify_hidden_linear(session, runtime.basis, "output/chunked/terminal-link",
                                                          {output_commitment, output_result->sumcheck.terminal_commitments.front()}, {F_ONE, -F_ONE}, F_ZERO,
                                                          reason))
            return false;

        const std::vector<NativeLassoChunkFamilyPublic> lasso_families = build_lasso_chunk_public_families(context.plan_ref(), context.tables);
        if (!native_lasso_verify_chunked_bundle(session, lasso_families, artifact.sparse_gkr.lasso_chunks, runtime.basis, &input_point, reason)) return false;
        zkgpt::native_pedersen::ScalarCommitment lasso_commitment;
        if (!sparse_commitment_from_wire(artifact.sparse_gkr.lasso_chunks.final_commitment_hex, runtime.basis, lasso_commitment, reason)) return false;
        if (!zkgpt::native_pedersen::verify_hidden_linear(session, runtime.basis, "lasso/chunked/input-link", {lasso_commitment, input_commitments[3]},
                                                          {F_ONE, -F_ONE}, F_ZERO, reason))
            return false;
        return true;
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool verify_gat(const VerifierContext& context, const GatProofArtifact& artifact, std::string* reason) {
    try {
        context.graph.validate();
        context.quant.validate();
        if (context.plan_ref().virtual_sparse) { set_reason(reason, "virtual sparse proof is unsupported: canonical GAT arithmetic and registered W/Aux/L commitment binding are incomplete"); return false; }
        if (!context.tables.authenticate()) {
            set_reason(reason, "static tables are not authenticated");
            return false;
        }
        if (context.public_features.rows != context.graph.num_nodes || context.public_features.cols != context.graph.num_features ||
            context.public_features.cols == 0 || context.public_features.values.size() != context.public_features.rows * context.public_features.cols) {
            set_reason(reason, "public feature dimensions are invalid");
            return false;
        }
        if (context.public_logits.rows != context.graph.num_nodes || context.public_logits.cols != context.graph.num_classes ||
            context.public_logits.cols == 0 || context.public_logits.values.size() != context.public_logits.rows * context.public_logits.cols) {
            set_reason(reason, "public logit dimensions are invalid");
            return false;
        }
        if (!registration_wire_valid(context.registration)) {
            set_reason(reason, "bound model registration is invalid");
            return false;
        }
        if (context.com_W.scheme != context.registration.com_W.scheme || context.com_W.basis_id != context.registration.com_W.basis_id ||
            context.com_W.logical_size != context.registration.com_W.logical_size || context.com_W.padded_size != context.registration.com_W.padded_size ||
            context.com_W.point_hex != context.registration.com_W.point_hex) {
            set_reason(reason, "verifier context com_W does not match the registration artifact");
            return false;
        }
        if (!context.plan_ref().official_layered || context.plan_ref().has_placeholder) {
            set_reason(reason, "verifier plan is not an official layered circuit");
            return false;
        }
        if (!canonical_layout_valid(context.plan_ref().input_layout) || context.registration.n_in != context.plan_ref().input_layout.n_in ||
            context.registration.input_segments.size() != context.plan_ref().input_layout.segments.size()) {
            set_reason(reason, "registration layout does not match the canonical four-segment plan");
            return false;
        }
        for (std::size_t index = 0; index < context.plan_ref().input_layout.segments.size(); ++index) {
            const InputSegment& left = context.registration.input_segments[index];
            const InputSegment& right = context.plan_ref().input_layout.segments[index];
            if (left.name != right.name || left.offset != right.offset || left.logical_length != right.logical_length ||
                left.padded_length != right.padded_length) {
                set_reason(reason, "registration segment does not match the canonical plan");
                return false;
            }
        }
        if (context.tables.quant_cfg_id != context.quant.quant_cfg_id || context.tables.table_id.empty()) {
            set_reason(reason, "static table identity mismatch");
            return false;
        }
        if (context.relation_id != "R_GAT-zkGPT" || context.quant_cfg_id != context.quant.quant_cfg_id || context.static_tables_id != context.tables.table_id ||
            context.model_arch_id != context.registration.model_arch_id) {
            set_reason(reason, "public verifier context identity mismatch");
            return false;
        }
        if (context.privacy_extension_profile_id != "masked_full") {
            set_reason(reason, "unsupported_privacy_profile");
            return false;
        }
        if (artifact.privacy_profile != "masked_full") {
            set_reason(reason, "unsupported_privacy_profile");
            return false;
        }
        const std::string expected_statement = verifier_context_digest(context);
        if (artifact.statement_digest != expected_statement || artifact.native_statement.verifier_context_digest != expected_statement ||
            artifact.native_statement.statement_root_digest != expected_statement) {
            set_reason(reason, "proof statement digest does not match independent public context");
            return false;
        }
        const std::string expected_com_w_digest = fixed_digest(hash_hex(commitment_encoding(context.com_W)));
        if (artifact.native_statement.com_w_digest != expected_com_w_digest ||
            artifact.native_statement.aux_digest != commitment_statement_digest(artifact.com_aux) ||
            artifact.native_statement.l_digest != commitment_statement_digest(artifact.com_L)) {
            set_reason(reason, "native proof commitment binding does not match the public context");
            return false;
        }
        NativeProofSession session;
        if (!artifact.native_session(session, reason) || !artifact.valid_wire_with_session(reason, session)) return false;

        const InputLayout verifier_layout = registration_layout(context.registration);
        const std::string basis_label = registration_basis_label_public(context.registration, verifier_layout);
        const std::string basis_id = fixed_digest(hash_hex(basis_label));
        const std::size_t domain = context.plan_ref().input_layout.n_in;
        const InputSegment& trusted_w_segment =
            context.plan_ref().input_layout.segments.at(1);
        const InputSegment& trusted_aux_segment =
            context.plan_ref().input_layout.segments.at(2);
        const InputSegment& trusted_l_segment =
            context.plan_ref().input_layout.segments.at(3);
        if (domain == 0 || !core_power_of_two(domain) || context.registration.com_W.basis_id != basis_id || artifact.com_aux.basis_id != basis_id ||
            artifact.com_L.basis_id != basis_id || artifact.com_aux.padded_size != domain || artifact.com_L.padded_size != domain ||
            !artifact.com_aux.has_tensor_commitment ||
            !artifact.com_L.has_tensor_commitment ||
            context.registration.com_W.logical_size != trusted_w_segment.logical_length ||
            artifact.com_aux.logical_size != trusted_aux_segment.logical_length ||
            artifact.com_L.logical_size != trusted_l_segment.logical_length) {
            set_reason(reason, "masked GAT verifier basis or commitment domain mismatch");
            return false;
        }
        const hyrax::HyraxTensorMleView trusted_aux_tensor_view =
            canonical_input_tensor_view(basis_label, basis_id, "canonical/Aux",
                                        domain, trusted_aux_segment.logical_length);
        const hyrax::HyraxTensorMleView trusted_l_tensor_view =
            canonical_input_tensor_view(basis_label, basis_id,
                                        "canonical/L_qry", domain,
                                        trusted_l_segment.logical_length);

        const G1 H = hyrax::derive_pedersen_blinding_generator(basis_id);
        NativeProofStatement expected_native_statement =
            make_native_proof_statement(context, artifact.com_aux, artifact.com_L, H, basis_label, basis_id, domain);
        const std::vector<NativeLassoFamilyPublic> expected_lasso_families = build_lasso_public_families(context.plan_ref(), context.tables, basis_label);
        expected_native_statement.tables_digest = native_lasso_tables_digest(expected_lasso_families);
        expected_native_statement.resources_digest = native_lasso_resources_digest(expected_lasso_families);
        expected_native_statement.l_digest = commitment_statement_digest(artifact.com_L);
        const NativeProofStatement& actual_native_statement = artifact.native_statement;
        if (actual_native_statement.verifier_context_digest != expected_native_statement.verifier_context_digest ||
            actual_native_statement.field_id != expected_native_statement.field_id ||
            actual_native_statement.plan_digest != expected_native_statement.plan_digest ||
            actual_native_statement.resources_digest != expected_native_statement.resources_digest ||
            actual_native_statement.tables_digest != expected_native_statement.tables_digest ||
            actual_native_statement.com_w_digest != expected_native_statement.com_w_digest ||
            actual_native_statement.aux_digest != expected_native_statement.aux_digest ||
            actual_native_statement.l_digest != expected_native_statement.l_digest ||
            actual_native_statement.basis_digest != expected_native_statement.basis_digest ||
            actual_native_statement.l_dimension != expected_native_statement.l_dimension ||
            actual_native_statement.statement_root_digest != expected_native_statement.statement_root_digest) {
            set_reason(reason, "native proof statement structured fields do not match public context");
            return false;
        }

        const auto has_prefix = [](const std::string& value, const std::string& prefix) { return value.compare(0, prefix.size(), prefix) == 0; };
        const auto is_lasso_label = [&has_prefix](const std::string& label) {
            return has_prefix(label, "lasso/") ||
                   has_prefix(label, "challenge/lasso/") ||
                   has_prefix(label, "tensor/") ||
                   has_prefix(label, "challenge/tensor/");
        };
        const auto is_gkr_label = [&has_prefix](const std::string& label) { return has_prefix(label, "gkr/") || has_prefix(label, "challenge/gkr/"); };
        const auto is_combine_label = [&has_prefix](const std::string& label) {
            return has_prefix(label, "combine/") || has_prefix(label, "challenge/combine/");
        };
        const auto is_open_label = [&has_prefix](const std::string& label) {
            return has_prefix(label, "Open_W/") || has_prefix(label, "Open_Aux/") || has_prefix(label, "Open_L/") || has_prefix(label, "Open_GAT_W/") ||
                   has_prefix(label, "Open_GAT_Aux/") || has_prefix(label, "Open_GAT_L/") || has_prefix(label, "challenge/Open_W/") ||
                   has_prefix(label, "challenge/Open_Aux/") || has_prefix(label, "challenge/Open_L/") || has_prefix(label, "challenge/Open_GAT_W/") ||
                   has_prefix(label, "challenge/Open_GAT_Aux/") || has_prefix(label, "challenge/Open_GAT_L/");
        };
        for (const NativeProofMessage& message : session.messages()) {
            if (!is_lasso_label(message.label) && !is_gkr_label(message.label) && !is_combine_label(message.label) && !is_open_label(message.label)) {
                set_reason(reason, "masked GAT transcript has an unknown message label: " + message.label);
                return false;
            }
        }

        const std::vector<NativeLassoFamilyPublic>& lasso_families = expected_lasso_families;
        const NativeLassoQueryPublic lasso_query = build_lasso_public_query(context.plan_ref(), artifact.com_L, basis_label);
        const zkgpt::native_pedersen::PedersenBasis scalar_basis =
            zkgpt::native_pedersen::derive_basis(
                "zkgpt/masked-sumcheck/v2/" + session.statement_digest());
        const NativeLassoTensorTailVerifyHook tensor_tail_stage =
            [&trusted_aux_tensor_view, &artifact, &scalar_basis,
             l_query_zero_intervals = canonical_l_query_zero_intervals(context.plan_ref())](
                const NativeProofSession& proof_session,
                const NativeLassoQueryPublic& query,
                std::size_t& exact_position,
                std::string* tail_reason) {
                if (query.tensor_view.role != "canonical/L_qry" ||
                    native_lasso_tensor_query_digest(query.tensor_view,
                                                     query.tensor_commitment) !=
                        commitment_statement_digest(artifact.com_L)) {
                    set_reason(tail_reason,
                               "shared tensor-tail verifier received an unbound L_qry root");
                    return false;
                }
                std::vector<Fr> beta;
                std::vector<Fr> terminal_point;
                NativeMaskedSumcheckResult sumcheck;
                if (!hyrax::verify_hyrax_tensor_zero_tail_certificate_with_session(
                        trusted_aux_tensor_view,
                        artifact.com_aux.tensor_commitment, proof_session,
                        exact_position, scalar_basis, beta, terminal_point,
                        sumcheck, tail_reason))
                    return false;
                beta.clear();
                terminal_point.clear();
                sumcheck = NativeMaskedSumcheckResult{};
                if (!hyrax::verify_hyrax_tensor_zero_tail_certificate_with_session(
                        query.tensor_view, query.tensor_commitment,
                        proof_session, exact_position, scalar_basis, beta,
                        terminal_point, sumcheck, tail_reason))
                    return false;
                beta.clear();
                terminal_point.clear();
                sumcheck = NativeMaskedSumcheckResult{};
                std::vector<hyrax::HyraxTensorZeroIntervalCertificate> gap_certificates;
                if (!hyrax::verify_hyrax_tensor_zero_interval_certificates_with_session(
                        query.tensor_view, query.tensor_commitment,
                        l_query_zero_intervals, proof_session, exact_position,
                        scalar_basis, gap_certificates, tail_reason))
                    return false;
                return true;
            };
        NativeLassoBundleResult lasso_result;
        if (!native_lasso_verify_bundle(session, lasso_families, lasso_query,
                                        &lasso_result, reason,
                                        tensor_tail_stage)) {
            if (reason != nullptr && reason->empty()) set_reason(reason, "native Lasso verifier rejected without a diagnostic");
            return false;
        }

        const zkgpt::NativeFieldView public_output =
            canonical_public_output_view(context.plan_ref(), context.public_logits, &context);
        if (context.plan_ref().circuit.circuit.empty() ||
            context.plan_ref().circuit.circuit.back().size != public_output.size()) {
            set_reason(reason, "masked GAT public output/residual shape is invalid");
            return false;
        }

        std::vector<NativeGkrMatmulShape> matmuls;
        matmuls.reserve(context.plan_ref().matmuls.size());
        for (const MatMulDescriptor& descriptor : context.plan_ref().matmuls)
            matmuls.push_back(native_gkr_shape_from_descriptor(descriptor, context.plan_ref().input_layout));

        std::size_t gkr_start = session.messages().size();
        for (std::size_t index = 0; index < session.messages().size(); ++index) {
            const NativeProofMessage& message = session.messages()[index];
            if (message.label == "gkr/public-output-digest-v2") {
                if (message.kind != NativeProofMessageKind::Data || gkr_start != session.messages().size()) {
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
        const NativeGkrInputSupport trusted_gkr_input_support =
            canonical_gkr_input_support(context.plan_ref());
        if (!verify_masked_gkr_transcript(session, context.plan_ref().circuit, public_output, matmuls, gkr_start, &gkr_end, reason,
                                          &verified_gkr_binding, &trusted_gkr_input_support)) {
            if (reason != nullptr && reason->empty()) set_reason(reason, "native GKR verifier rejected without a diagnostic");
            return false;
        }

        if (verified_gkr_binding.point.size() != core_log2(domain) + 2 || !verified_gkr_binding.claim_commitment.valid(scalar_basis, reason) ||
            !verified_gkr_binding.unmasked_claim_commitment.valid(scalar_basis, reason) || !verified_gkr_binding.mask_commitment.valid(scalar_basis, reason) ||
            !verified_gkr_binding.unmasked_input_value_commitment.valid(scalar_basis, reason) ||
            !verified_gkr_binding.masked_input_value_commitment.valid(scalar_basis, reason) || !lasso_result.has_final_l_commitment ||
            !lasso_result.final_l_commitment.valid(scalar_basis, reason) || lasso_result.final_query_point.empty()) {
            set_reason(reason, "masked GAT v2 authenticated input/final binding is malformed");
            return false;
        }

        auto decode_field_vector = [](const std::string& payload, std::vector<Fr>& values) {
            values.clear();
            std::size_t offset = 0;
            auto read_u64 = [&](std::uint64_t& value) {
                if (payload.size() - offset < sizeof(std::uint64_t)) return false;
                value = 0;
                for (unsigned index = 0; index < 8; ++index) value |= static_cast<std::uint64_t>(static_cast<unsigned char>(payload[offset++])) << (8 * index);
                return true;
            };
            std::uint64_t count = 0;
            if (!read_u64(count) || count > (1ULL << 20)) return false;
            values.reserve(static_cast<std::size_t>(count));
            for (std::size_t index = 0; index < static_cast<std::size_t>(count); ++index) {
                std::uint64_t size = 0;
                if (!read_u64(size) || size == 0 || size > 128 || size > payload.size() - offset) return false;
                const std::string encoded = payload.substr(offset, static_cast<std::size_t>(size));
                offset += static_cast<std::size_t>(size);
                Fr value;
                if (!core_decode_field(encoded, value)) return false;
                values.push_back(value);
            }
            return offset == payload.size();
        };
        auto consume_message = [&](std::size_t& cursor, NativeProofMessageKind kind, const std::string& label) -> const NativeProofMessage* {
            if (cursor >= session.messages().size()) {
                set_reason(reason, "masked GAT transcript ended before " + label);
                return nullptr;
            }
            const NativeProofMessage& message = session.messages()[cursor];
            if (message.kind != kind || message.label != label) {
                set_reason(reason, "masked GAT transcript expected " + label);
                return nullptr;
            }
            ++cursor;
            return &message;
        };
        auto decode_message_field = [&](const NativeProofMessage* message, Fr& value, const std::string& label) {
            if (message == nullptr || !core_decode_field(message->payload, value)) {
                set_reason(reason, "masked GAT field message is malformed: " + label);
                return false;
            }
            return true;
        };

        if (gkr_end >= session.messages().size() || !is_combine_label(session.messages()[gkr_end].label)) {
            set_reason(reason, "masked GAT GKR scope is not followed by combine");
            return false;
        }
        const std::size_t r_dimension = core_log2(domain);
        const std::vector<Fr> r_minus(verified_gkr_binding.point.begin(), verified_gkr_binding.point.begin() + r_dimension);
        const Fr selector_s1 = verified_gkr_binding.point[r_dimension + 1];
        const Fr selector_s2 = verified_gkr_binding.point[r_dimension];
        const Fr c_x = (Fr(1) - selector_s1) * (Fr(1) - selector_s2);
        const Fr c_w = (Fr(1) - selector_s1) * selector_s2;
        const Fr c_aux = selector_s1 * (Fr(1) - selector_s2);
        const Fr c_l = selector_s1 * selector_s2;

        std::size_t cursor = gkr_end;
        const NativeProofMessage* combine_point_message = consume_message(cursor, NativeProofMessageKind::FieldVector, "combine/r_minus");
        const NativeProofMessage* combine_selectors_message = consume_message(cursor, NativeProofMessageKind::FieldVector, "combine/selectors");
        const NativeProofMessage* combine_x_message = consume_message(cursor, NativeProofMessageKind::Field, "combine/public-X");
        const NativeProofMessage* combine_structured_root_message = consume_message(cursor, NativeProofMessageKind::Data, "combine/structured-root");
        const NativeProofMessage* combine_commitments_message = consume_message(cursor, NativeProofMessageKind::Data, "combine/commitments");
        const NativeProofMessage* combine_gkr_message = consume_message(cursor, NativeProofMessageKind::Data, "combine/gkr-input-commitment");
        const NativeProofMessage* combine_lasso_message = consume_message(cursor, NativeProofMessageKind::Field, "combine/lasso-final-claim");
        const NativeProofMessage* combine_lasso_commitment_message = consume_message(cursor, NativeProofMessageKind::Data, "combine/lasso-final-commitment");
        if (combine_point_message == nullptr || combine_selectors_message == nullptr || combine_x_message == nullptr ||
            combine_structured_root_message == nullptr || combine_commitments_message == nullptr || combine_gkr_message == nullptr ||
            combine_lasso_message == nullptr || combine_lasso_commitment_message == nullptr)
            return false;
        std::vector<Fr> encoded_r_minus, encoded_selectors;
        if (!decode_field_vector(combine_point_message->payload, encoded_r_minus) ||
            !decode_field_vector(combine_selectors_message->payload, encoded_selectors) || encoded_r_minus != r_minus || encoded_selectors.size() != 2 ||
            encoded_selectors[0] != selector_s1 || encoded_selectors[1] != selector_s2) {
            set_reason(reason, "masked GAT combine point/selector binding failed");
            return false;
        }
        Fr encoded_x, encoded_lasso_claim;
        if (!decode_message_field(combine_x_message, encoded_x, "combine/public-X") ||
            !decode_message_field(combine_lasso_message, encoded_lasso_claim, "combine/lasso-final-claim") ||
            encoded_lasso_claim != lasso_result.final_l_evaluation) {
            set_reason(reason, "masked GAT combine public claim binding failed");
            return false;
        }
        const std::vector<Fr> x_values = public_feature_segment(context.plan_ref(), context.graph, context.public_features);
        const Fr x_eval = adapter_mle_eval(x_values, r_minus);
        if (encoded_x != x_eval) {
            set_reason(reason, "masked GAT public X terminal evaluation mismatch");
            return false;
        }
        if (combine_structured_root_message->payload != actual_native_statement.structured_root()) {
            set_reason(reason, "masked GAT structured statement root binding failed");
            return false;
        }
        const std::string expected_commitments = basis_id + "|" + commitment_encoding(context.registration.com_W) + "|" +
                                                 commitment_encoding(artifact.com_aux) + "|" + commitment_encoding(artifact.com_L);
        if (combine_commitments_message->payload != expected_commitments) {
            set_reason(reason, "masked GAT combine commitment binding failed");
            return false;
        }
        auto parse_scalar_commitment = [&](const NativeProofMessage* message, const std::string& label, zkgpt::native_pedersen::ScalarCommitment& out) {
            std::vector<std::string> parts;
            G1 point;
            if (message == nullptr || !core_split_pipe(message->payload, parts) || parts.size() != 2 || parts[0] != scalar_basis.domain ||
                !core_decode_point(parts[1], point)) {
                set_reason(reason, "masked GAT scalar commitment is malformed: " + label);
                return false;
            }
            out = {scalar_basis.domain, point};
            return out.valid(scalar_basis, reason);
        };
        zkgpt::native_pedersen::ScalarCommitment encoded_gkr_commitment;
        if (!parse_scalar_commitment(combine_gkr_message, "combine/gkr-input-commitment", encoded_gkr_commitment) ||
            encoded_gkr_commitment.point != verified_gkr_binding.unmasked_claim_commitment.point) {
            set_reason(reason, "masked GAT GKR input commitment binding failed");
            return false;
        }
        zkgpt::native_pedersen::ScalarCommitment encoded_lasso_commitment;
        if (!parse_scalar_commitment(combine_lasso_commitment_message, "combine/lasso-final-commitment", encoded_lasso_commitment) ||
            encoded_lasso_commitment.point != lasso_result.final_l_commitment.point) {
            set_reason(reason, "masked GAT Lasso final commitment binding failed");
            return false;
        }

        auto verify_terminal_opening = [&](const std::string& label,
                                           const std::string &role,
                                           std::uint64_t logical_length,
                                           const CommitmentWire& wire,
                                           const hyrax::HyraxTensorMleView* tensor_view,
                                           zkgpt::native_pedersen::ScalarCommitment& raw_commitment,
                                           zkgpt::native_pedersen::ScalarCommitment& masked_commitment) {
            const std::size_t start = cursor;
            if (wire.has_tensor_commitment != (tensor_view != nullptr)) {
                set_reason(reason, "masked GAT opening method differs from trusted commitment: " + label);
                return false;
            }
            if (tensor_view != nullptr &&
                consume_message(cursor, NativeProofMessageKind::Data,
                                label + "/tensor/public") == nullptr)
                return false;
            const std::string inner_label =
                tensor_view == nullptr ? label : label + "/column";
            const NativeProofMessage* basis_message = consume_message(cursor, NativeProofMessageKind::Data, inner_label + "/basis");
            const NativeProofMessage* claim_message = consume_message(cursor, NativeProofMessageKind::Data, inner_label + "/claim");
            const NativeProofMessage* announce_message = consume_message(cursor, NativeProofMessageKind::Data, inner_label + "/announce");
            const NativeProofMessage* challenge_message = consume_message(cursor, NativeProofMessageKind::Challenge, "challenge/" + inner_label + "/c");
            const NativeProofMessage* response_message = consume_message(cursor, NativeProofMessageKind::FieldVector, inner_label + "/response");
            if (basis_message == nullptr || claim_message == nullptr || announce_message == nullptr || challenge_message == nullptr ||
                response_message == nullptr)
                return false;
            std::vector<std::string> parts;
            G1 evaluation_mask;
            Fr masked_evaluation;
            if (!core_split_pipe(claim_message->payload, parts) || parts.size() != 5 || !core_decode_point(parts[0], evaluation_mask) ||
                !core_decode_field(parts[1], masked_evaluation) || parts[2] != scalar_basis.domain) {
                set_reason(reason, "masked GAT scalar opening claim is malformed: " + label);
                return false;
            }
            if (!core_decode_point(parts[3], raw_commitment.point) || !core_decode_point(parts[4], masked_commitment.point)) {
                set_reason(reason, "masked GAT scalar opening point is malformed: " + label);
                return false;
            }
            raw_commitment.domain = scalar_basis.domain;
            masked_commitment.domain = scalar_basis.domain;
            std::uint64_t opening_domain = 0;
            if (tensor_view != nullptr) {
                opening_domain = tensor_view->columns;
                if (!hyrax::verify_hyrax_tensor_mle_opening_with_session(
                        *tensor_view, wire.tensor_commitment, r_minus,
                        evaluation_mask, masked_evaluation, session, label,
                        reason, true, "", &scalar_basis, &raw_commitment,
                        &masked_commitment))
                    return false;
            } else {
                G1 commitment_point;
                if (!core_decode_point(wire.point_hex, commitment_point)) {
                    set_reason(reason, "masked GAT commitment point is malformed: " + label);
                    return false;
                }
                const hyrax::HyraxMleBasisView view = canonical_input_opening_view(
                    basis_label, basis_id, actual_native_statement.basis_digest,
                    role, domain, logical_length, r_minus);
                if (!hyrax::verify_masked_mle_opening_with_session(
                        commitment_point, evaluation_mask, masked_evaluation, view,
                        session, label, reason, true, true, "", &scalar_basis,
                        &raw_commitment, &masked_commitment))
                    return false;
                opening_domain = next_power_of_two(logical_length);
            }
            std::size_t active_rounds = 0;
            for (std::uint64_t active = opening_domain; active > 1; active >>= 1)
                ++active_rounds;
            if (consume_message(cursor, NativeProofMessageKind::Data,
                                inner_label + "/response-ipa/basis_ref") == nullptr ||
                consume_message(cursor, NativeProofMessageKind::Data,
                                inner_label + "/response-ipa/evaluation") == nullptr ||
                consume_message(cursor, NativeProofMessageKind::Challenge,
                                "challenge/" + inner_label +
                                    "/response-ipa/evaluation-base") == nullptr)
                return false;
            for (std::size_t round = 0; round < active_rounds; ++round) {
                const std::string round_label =
                    std::to_string(round);
                if (consume_message(cursor, NativeProofMessageKind::Data,
                                    inner_label + "/response-ipa/fold/" + round_label) == nullptr ||
                    consume_message(cursor, NativeProofMessageKind::Challenge,
                                    "challenge/" + inner_label +
                                        "/response-ipa/c/" + round_label) == nullptr)
                    return false;
            }
            if (consume_message(cursor, NativeProofMessageKind::Data,
                                inner_label + "/response-ipa/terminal") == nullptr)
                return false;
            const std::string prefix = label + "/";
            const std::string challenge_prefix = "challenge/" + prefix;
            for (std::size_t index = cursor; index < session.messages().size(); ++index) {
                if (session.messages()[index].label.compare(0, prefix.size(), prefix) == 0 ||
                    session.messages()[index].label.compare(0, challenge_prefix.size(), challenge_prefix) == 0) {
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
        if (!verify_terminal_opening(
                "Open_GAT_W", "canonical/W",
                trusted_w_segment.logical_length,
                context.registration.com_W, nullptr, raw_w, masked_w) ||
            !verify_terminal_opening(
                "Open_GAT_Aux", "canonical/Aux", trusted_aux_segment.logical_length,
                artifact.com_aux, &trusted_aux_tensor_view, raw_aux,
                masked_aux) ||
            !verify_terminal_opening(
                "Open_GAT_L", "canonical/L_qry", trusted_l_segment.logical_length,
                artifact.com_L, &trusted_l_tensor_view, raw_l, masked_l))
            return false;

        const std::size_t relation_start = cursor;
        if (relation_start >= session.messages().size() || session.messages()[relation_start].label != "combine/input-linear/meta") {
            set_reason(reason, "masked GAT linear relation is not adjacent to openings");
            return false;
        }
        if (!zkgpt::native_pedersen::verify_hidden_linear(session, scalar_basis, "combine/input-linear",
                                                          {verified_gkr_binding.unmasked_input_value_commitment, raw_w, raw_aux, raw_l},
                                                          {F_ONE, -c_w, -c_aux, -c_l}, c_x * x_eval, reason))
            return false;
        const std::array<std::string, 6> relation_labels = {
            "combine/input-linear/meta", "combine/input-linear/coefficients",        "combine/input-linear/public",
            "combine/input-linear/T",    "challenge/combine/input-linear/challenge", "combine/input-linear/response"};
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
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}

MaskingSession begin_masking_session(const std::string& profile_id, const std::string& statement_digest) {
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

bool core_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

std::size_t core_log2(std::size_t value) {
    std::size_t result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

template<class Allocator>
std::string core_vector_digest(const std::string& domain, const std::vector<Fr, Allocator>& values) {
    std::string encoded;
    append_string(encoded, domain);
    append_u64(encoded, static_cast<std::uint64_t>(values.size()));
    for (const Fr& value : values) append_string(encoded, field_hex(value));
    return fixed_digest(hash_hex(encoded));
}

std::string core_commitment_payload(const std::string& basis_id, const CommitmentWire& com_W, const CommitmentWire& com_aux, const CommitmentWire& com_L,
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

bool core_decode_field(const std::string& encoded, Fr& value) {
    if (encoded.empty() || encoded.size() > 128) return false;
    try {
        value.setStr(encoded, 16);
        return field_hex(value) == encoded;
    } catch (...) { return false; }
}

const NativeProofMessage* core_find_message(const NativeProofSession& session, NativeProofMessageKind kind, const std::string& label, std::string* reason) {
    const NativeProofMessage* found = nullptr;
    const std::vector<std::size_t>* positions = session.message_positions(label);
    if (positions == nullptr) {
        set_reason(reason, "missing masked core message: " + label);
        return nullptr;
    }
    for (std::size_t position : *positions) {
        if (position >= session.messages().size()) {
            set_reason(reason, "masked core message index is out of range: " + label);
            return nullptr;
        }
        const NativeProofMessage& message = session.messages()[position];
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

bool core_decode_token_payload(const std::string& payload, std::vector<std::string>& tokens) {
    tokens.clear();
    std::size_t offset = 0;
    auto read_u64 = [&](std::uint64_t& value) {
        if (payload.size() - offset < sizeof(std::uint64_t)) return false;
        value = 0;
        for (unsigned index = 0; index < 8; ++index) value |= static_cast<std::uint64_t>(static_cast<unsigned char>(payload[offset++])) << (8 * index);
        return true;
    };
    std::uint64_t count = 0;
    if (!read_u64(count) || count > 64) return false;
    tokens.reserve(static_cast<std::size_t>(count));
    for (std::size_t index = 0; index < static_cast<std::size_t>(count); ++index) {
        std::uint64_t size = 0;
        if (!read_u64(size) || size > (1ULL << 20) || size > payload.size() - offset) return false;
        tokens.push_back(payload.substr(offset, static_cast<std::size_t>(size)));
        offset += static_cast<std::size_t>(size);
    }
    return offset == payload.size();
}

bool core_split_pipe(const std::string& payload, std::vector<std::string>& parts) {
    parts.clear();
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = payload.find('|', begin);
        parts.push_back(payload.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return !parts.empty();
}

bool core_decode_point(const std::string& encoded, G1& point) {
    if (encoded.empty() || encoded.size() > 4096) return false;
    try {
        point.deserializeHexStr(encoded);
        return point.serializeToHexStr() == encoded && point.isValid() && !point.isZero() && point.isValidOrder();
    } catch (...) { return false; }
}

bool core_commitment_matches(const CommitmentWire& wire, const hyrax::PedersenMleCommitment& commitment, const std::string& basis_id, std::size_t domain) {
    return wire.scheme == "hyrax-pedersen-bn254" && wire.basis_id == basis_id && wire.padded_size == domain && wire.logical_size != 0 &&
           wire.logical_size <= domain && wire.point_hex == commitment.commitment.serializeToHexStr();
}

hyrax::PedersenMleCommitment core_existing_commitment(const CommitmentWire& wire, const Fr& blinder, const std::string& basis_id, const G1& H,
                                                      const std::vector<G1>& generators, const std::vector<Fr>& values) {
    if (values.empty() || values.size() != generators.size() || !core_power_of_two(values.size()) || H != hyrax::derive_pedersen_blinding_generator(basis_id))
        throw std::invalid_argument("masked core W commitment dimensions are invalid");
    if (wire.basis_id != basis_id || wire.padded_size != values.size()) throw std::invalid_argument("masked core W commitment identity mismatch");
    G1 encoded;
    encoded.deserializeHexStr(wire.point_hex);
    hyrax::PedersenMleCommitment result;
    result.commitment = encoded;
    result.blinding_generator = H;
    result.blinding = blinder;
    result.basis_id = basis_id;
    const G1 expected = perdersen_commit(const_cast<G1*>(generators.data()), const_cast<Fr*>(values.data()), static_cast<int>(values.size())) + H * blinder;
    if (expected != result.commitment) throw std::invalid_argument("masked core W commitment does not bind witness");
    return result;
}

CommitmentWire core_commitment_wire(const hyrax::PedersenMleCommitment& commitment, std::size_t logical_size, std::size_t domain) {
    CommitmentWire wire;
    wire.scheme = "hyrax-pedersen-bn254";
    wire.basis_id = commitment.basis_id;
    wire.logical_size = logical_size;
    wire.padded_size = domain;
    wire.point_hex = commitment.commitment.serializeToHexStr();
    return wire;
}

std::vector<Fr> core_point_from_session(const NativeProofSession& session, const std::string& label, std::size_t dimension, std::string* reason) {
    std::vector<Fr> point;
    point.reserve(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
        const NativeProofMessage* message =
            core_find_message(session, NativeProofMessageKind::Challenge, "challenge/" + label + "/" + std::to_string(index), reason);
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

bool core_message_in_scope(const std::string& label, const std::string& scope) {
    const std::string prefix = scope + "/";
    const std::string challenge = "challenge/" + prefix;
    return label.compare(0, prefix.size(), prefix) == 0 || label.compare(0, challenge.size(), challenge) == 0;
}

void core_copy_sections(const NativeProofSession& session, GatProofArtifact& artifact) {
    artifact.Pi_Lasso.protocol = "Pi_Lasso";
    artifact.Pi_MatMul.protocol = "Pi_MatMul";
    artifact.Pi_GKR.protocol = "Pi_GKR";
    artifact.Pi_combine.protocol = "Pi_combine";
    artifact.Pi_open.protocol = "Pi_open";
    const auto copy = [](const NativeProofMessage& message) {
        TranscriptMessage result;
        result.label = message.label;
        result.payload = message.payload;
        result.kind = static_cast<std::uint8_t>(message.kind);
        result.sequence = message.sequence;
        return result;
    };
    for (const NativeProofMessage& message : session.messages()) {
        ProofSection* section = nullptr;
        if (message.label == "lasso/reserved" || message.label.compare(0, sizeof("lasso/") - 1, "lasso/") == 0 ||
            message.label.compare(0, sizeof("challenge/lasso/") - 1, "challenge/lasso/") == 0 ||
            core_message_in_scope(message.label, "tensor/aux/tail") ||
            core_message_in_scope(message.label, "tensor/L_qry/tail") ||
            core_message_in_scope(message.label, "tensor/L_qry/zero-gap"))
            section = &artifact.Pi_Lasso;
        else if (message.label.compare(0, sizeof("matmul/") - 1, "matmul/") == 0 ||
                 message.label.compare(0, sizeof("challenge/matmul/") - 1, "challenge/matmul/") == 0 ||
                 message.label.compare(0, sizeof("gkr/layer/") - 1, "gkr/layer/") == 0 && message.label.find("/matmul") != std::string::npos ||
                 message.label.compare(0, sizeof("challenge/gkr/layer/") - 1, "challenge/gkr/layer/") == 0 &&
                     message.label.find("/matmul") != std::string::npos)
            section = &artifact.Pi_MatMul;
        else if (message.label.compare(0, sizeof("gkr/") - 1, "gkr/") == 0 || message.label.compare(0, sizeof("challenge/gkr/") - 1, "challenge/gkr/") == 0)
            section = &artifact.Pi_GKR;
        else if (message.label.compare(0, sizeof("sparse/") - 1, "sparse/") == 0 ||
                 message.label.compare(0, sizeof("challenge/sparse/") - 1, "challenge/sparse/") == 0)
            section = &artifact.Pi_GKR;
        else if (message.label.compare(0, sizeof("combine/") - 1, "combine/") == 0 ||
                 message.label.compare(0, sizeof("challenge/combine/") - 1, "challenge/combine/") == 0)
            section = &artifact.Pi_combine;
        else if (message.label.compare(0, sizeof("Open_") - 1, "Open_") == 0 || message.label.compare(0, sizeof("challenge/Open_") - 1, "challenge/Open_") == 0)
            section = &artifact.Pi_open;
        if (section == nullptr) throw std::invalid_argument("masked core transcript label is outside canonical sections: " + message.label);
        section->messages.push_back(copy(message));
    }
}

bool core_expected_label(const std::string& label) {
    return label == "lasso/reserved" || label.compare(0, sizeof("lasso/") - 1, "lasso/") == 0 ||
           label.compare(0, sizeof("challenge/lasso/") - 1, "challenge/lasso/") == 0 || core_message_in_scope(label, "matmul") ||
           core_message_in_scope(label, "tensor/aux/tail") || core_message_in_scope(label, "tensor/L_qry/tail") ||
           core_message_in_scope(label, "tensor/L_qry/zero-gap") || core_message_in_scope(label, "gkr") ||
           core_message_in_scope(label, "combine") || core_message_in_scope(label, "Open_W") ||
           core_message_in_scope(label, "Open_Aux") || core_message_in_scope(label, "Open_L");
}

bool core_find_index(const NativeProofSession& session, const std::string& label, NativeProofMessageKind kind, std::size_t& index, std::string* reason) {
    bool found = false;
    for (std::size_t cursor = 0; cursor < session.messages().size(); ++cursor) {
        const NativeProofMessage& message = session.messages()[cursor];
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

}

static MaskedCoreResult prove_masked_core_impl(const NativeProofStatement& statement, const CommitmentWire& com_W, const Fr& com_W_blinder,
                                               const std::string& basis_id, const G1& H, const std::vector<G1>& generators, const MaskedCoreWitness& witness,
                                               const std::string& statement_digest, std::vector<NativeLassoFamilyProver>* lasso_families,
                                               NativeLassoQueryProver* lasso_query) {
    MaskedCoreResult result;
    try {
        ensure_field();
        if (statement_digest.empty() || statement_digest != statement.verifier_context_digest || statement.statement_root_digest != statement_digest)
            throw std::invalid_argument("masked core statement root is not authenticated");
        if (basis_id.empty() || generators.empty() || !core_power_of_two(generators.size()) ||
            generators.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) || H != hyrax::derive_pedersen_blinding_generator(basis_id))
            throw std::invalid_argument("masked core basis setup is invalid");
        const std::size_t domain = generators.size();
        if (statement.l_dimension != domain || !statement.valid()) throw std::invalid_argument("masked core statement domain is invalid");
        const std::vector<Fr> dummy_basis(domain, Fr(0));
        if (statement.basis_digest != NativeProofStatement::basis_digest_for(H, generators, dummy_basis, domain))
            throw std::invalid_argument("masked core statement basis mismatch");
        const auto valid_segment = [domain](const std::vector<Fr>& values) { return values.size() == domain; };
        if (!valid_segment(witness.w) || !valid_segment(witness.aux) || !valid_segment(witness.l_qry) || witness.public_output.empty() ||
            witness.gkr_round_polys.empty() || witness.matmul_round_polys.empty() || witness.gkr_coefficients.size() != 5 ||
            witness.gkr_terminal_values.size() != 4 || witness.matmul_terminal_values.size() != 2)
            throw std::invalid_argument("masked core witness dimensions are invalid");
        for (const std::vector<Fr>& poly : witness.gkr_round_polys)
            if (poly.size() != 3) throw std::invalid_argument("masked core GKR polynomial arity is invalid");
        for (const std::vector<Fr>& poly : witness.matmul_round_polys)
            if (poly.size() != 3) throw std::invalid_argument("masked core MatMul polynomial arity is invalid");
        if (com_W.basis_id != basis_id || com_W.padded_size != domain || !com_W.valid())
            throw std::invalid_argument("masked core W commitment wire is invalid");

        const hyrax::PedersenMleCommitment w_commit = core_existing_commitment(com_W, com_W_blinder, basis_id, H, generators, witness.w);
        hyrax::PedersenMleCommitment aux_commit;
        hyrax::PedersenMleCommitment l_commit;
        if (witness.has_dynamic_commitments) {
            aux_commit = core_existing_commitment(witness.com_aux.wire, witness.com_aux.blinder, basis_id, H, generators, witness.aux);
            l_commit = core_existing_commitment(witness.com_L.wire, witness.com_L.blinder, basis_id, H, generators, witness.l_qry);
        } else {
            aux_commit = hyrax::pedersen_commit_mle(generators, witness.aux, basis_id);
            l_commit = hyrax::pedersen_commit_mle(generators, witness.l_qry, basis_id);
        }
        if (aux_commit.blinding_generator != H || l_commit.blinding_generator != H) throw std::logic_error("masked core Pedersen generator mismatch");
        const std::size_t aux_logical = witness.has_dynamic_commitments ? witness.com_aux.wire.logical_size : domain;
        const std::size_t l_logical = witness.has_dynamic_commitments ? witness.com_L.wire.logical_size : domain;
        result.com_aux = core_commitment_wire(aux_commit, aux_logical, domain);
        result.com_L = core_commitment_wire(l_commit, l_logical, domain);
        if (!core_commitment_matches(result.com_aux, aux_commit, basis_id, domain) || !core_commitment_matches(result.com_L, l_commit, basis_id, domain))
            throw std::logic_error("masked core dynamic commitment wire mismatch");
        NativeProofStatement bound_statement = statement;
        const std::string expected_aux_digest = commitment_statement_digest(result.com_aux);
        const std::string expected_l_digest = commitment_statement_digest(result.com_L);
        const std::string expected_w_digest = fixed_digest(hash_hex(commitment_encoding(com_W)));

        const std::string zero_digest(64, '0');
        if (bound_statement.aux_digest == zero_digest) bound_statement.aux_digest = expected_aux_digest;
        if (bound_statement.l_digest == zero_digest) bound_statement.l_digest = expected_l_digest;
        if (bound_statement.com_w_digest == zero_digest) bound_statement.com_w_digest = expected_w_digest;
        if (bound_statement.aux_digest != expected_aux_digest || bound_statement.l_digest != expected_l_digest ||
            bound_statement.com_w_digest != expected_w_digest)
            throw std::invalid_argument("masked core statement commitment binding mismatch");

        if ((lasso_families == nullptr) != (lasso_query == nullptr)) throw std::invalid_argument("masked core Lasso inputs are incomplete");
        NativeProofSession session(kNativeArtifactDomain, bound_statement);
        std::string reason;
        if (lasso_families != nullptr) {
            if (!native_lasso_prove_bundle(session, *lasso_families, *lasso_query, nullptr, &reason))
                throw std::runtime_error("masked native Lasso proof failed: " + reason);
        } else {
            session.absorb("lasso/reserved", "native_lasso_pending_v1");
        }
        const Fr matmul_initial = witness.matmul_round_polys.front()[0] + witness.matmul_round_polys.front()[1];
        session.absorb_field("matmul/claim", matmul_initial);
        if (!native_masked_matmul_product_prove(session, "matmul/core", witness.matmul_round_polys, matmul_initial, witness.matmul_terminal_values, nullptr,
                                                &reason))
            throw std::runtime_error("masked MatMul core proof failed: " + reason);

        session.absorb("gkr/public-output", core_vector_digest("zkgpt-masked-core-public-output-v1", witness.public_output));
        const std::size_t output_domain = next_power_of_two(witness.public_output.size());
        const std::size_t output_rounds = core_log2(output_domain);
        std::vector<Fr> output_point;
        output_point.reserve(output_rounds);
        for (std::size_t index = 0; index < output_rounds; ++index) output_point.push_back(session.challenge("gkr/output/r/" + std::to_string(index)));
        std::vector<Fr> output_values(output_domain, Fr(0));
        std::copy(witness.public_output.begin(), witness.public_output.end(), output_values.begin());
        const Fr gkr_initial = [&output_values, &output_point]() {
            if (output_point.empty()) return output_values.at(0);
            std::vector<Fr> folded = output_values;
            std::size_t active = folded.size();
            for (const Fr& coordinate : output_point) {
                const std::size_t half = active / 2;
                for (std::size_t index = 0; index < half; ++index)
                    folded[index] = folded[2 * index] * (Fr(1) - coordinate) + folded[2 * index + 1] * coordinate;
                active = half;
            }
            return folded.at(0);
        }();
        if (!native_masked_gkr_quadratic_prove(session, "gkr/core", witness.gkr_round_polys, gkr_initial, witness.gkr_terminal_values, witness.gkr_coefficients,
                                               nullptr, &reason))
            throw std::runtime_error("masked GKR core proof failed: " + reason);

        session.absorb("combine/public-input", core_vector_digest("zkgpt-masked-core-public-input-v1", witness.public_input));
        session.absorb("combine/public-output", core_vector_digest("zkgpt-masked-core-public-output-v1", witness.public_output));
        session.absorb("combine/commitments", core_commitment_payload(basis_id, com_W, result.com_aux, result.com_L, domain));
        session.absorb("combine/opening-domain", std::to_string(domain));
        const std::size_t opening_rounds = core_log2(domain);
        std::vector<Fr> opening_point;
        opening_point.reserve(opening_rounds);
        for (std::size_t index = 0; index < opening_rounds; ++index) opening_point.push_back(session.challenge("combine/r_minus/" + std::to_string(index)));

        hyrax::MaskedMleOpening ignored;
        if (!hyrax::open_masked_mle_with_session(w_commit, generators, opening_point, witness.w, session, "Open_W", ignored, &reason) ||
            !hyrax::open_masked_mle_with_session(aux_commit, generators, opening_point, witness.aux, session, "Open_Aux", ignored, &reason) ||
            !hyrax::open_masked_mle_with_session(l_commit, generators, opening_point, witness.l_qry, session, "Open_L", ignored, &reason))
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
        if (!artifact.valid_wire(&reason)) throw std::runtime_error("masked core artifact wire validation failed: " + reason);
        result.generated = true;
        result.artifact = std::move(artifact);
        result.diagnostic = "masked_core_generated; native_lasso_pending";
        return result;
    } catch (const std::exception& error) {
        result.generated = false;
        result.diagnostic = error.what();
        return result;
    }
}

MaskedCoreResult prove_masked_core(const NativeProofStatement& statement, const CommitmentWire& com_W, const Fr& com_W_blinder, const std::string& basis_id,
                                   const G1& H, const std::vector<G1>& generators, const MaskedCoreWitness& witness, const std::string& statement_digest) {
    return prove_masked_core_impl(statement, com_W, com_W_blinder, basis_id, H, generators, witness, statement_digest, nullptr, nullptr);
}

MaskedCoreResult prove_masked_core_with_lasso(const NativeProofStatement& statement, const CommitmentWire& com_W, const Fr& com_W_blinder,
                                              const std::string& basis_id, const G1& H, const std::vector<G1>& generators, const MaskedCoreWitness& witness,
                                              std::vector<NativeLassoFamilyProver>& families, NativeLassoQueryProver& query,
                                              const std::string& statement_digest) {
    return prove_masked_core_impl(statement, com_W, com_W_blinder, basis_id, H, generators, witness, statement_digest, &families, &query);
}

bool verify_masked_core(const GatProofArtifact& artifact, const CommitmentWire& com_W, const std::string& basis_id, const G1& H,
                        const std::vector<G1>& generators, const std::vector<Fr>& public_input, const std::vector<Fr>& public_output,
                        const std::vector<Fr>& gkr_coefficients, std::size_t gkr_rounds, const Fr& matmul_initial_claim, std::size_t matmul_rounds,
                        std::string* reason) {
    try {
        if (!artifact.valid_wire(reason) || artifact.privacy_profile != "masked_full") return false;
        NativeProofSession session;
        if (!artifact.native_session(session, reason)) return false;
        if (basis_id.empty() || generators.empty() || !core_power_of_two(generators.size()) || H != hyrax::derive_pedersen_blinding_generator(basis_id) ||
            com_W.basis_id != basis_id || !com_W.valid() || com_W.padded_size != generators.size()) {
            set_reason(reason, "masked core verifier basis or W commitment is invalid");
            return false;
        }
        if (artifact.native_statement.com_w_digest != fixed_digest(hash_hex(commitment_encoding(com_W)))) {
            set_reason(reason, "masked core verifier W statement binding mismatch");
            return false;
        }
        if (session.messages().empty()) {
            set_reason(reason, "masked core transcript has no Lasso or core prefix");
            return false;
        }
        const NativeProofMessage& first = session.messages().front();
        const bool reserved_lasso = first.kind == NativeProofMessageKind::Data && first.label == "lasso/reserved" && first.payload == "native_lasso_pending_v1";
        const bool native_lasso_prefix =
            first.label.compare(0, sizeof("lasso/") - 1, "lasso/") == 0 || first.label.compare(0, sizeof("challenge/lasso/") - 1, "challenge/lasso/") == 0;
        if (!reserved_lasso && !native_lasso_prefix) {
            set_reason(reason, "masked core transcript does not start with canonical Lasso");
            return false;
        }
        for (const NativeProofMessage& message : session.messages()) {
            if (!core_expected_label(message.label)) {
                set_reason(reason, "unexpected masked core transcript message: " + message.label);
                return false;
            }
        }
        const NativeProofMessage* matmul_claim = core_find_message(session, NativeProofMessageKind::Field, "matmul/claim", reason);
        Fr encoded_matmul;
        if (matmul_claim == nullptr || !core_decode_field(matmul_claim->payload, encoded_matmul) || encoded_matmul != matmul_initial_claim) {
            set_reason(reason, "masked core MatMul public claim mismatch");
            return false;
        }
        if (!native_masked_matmul_product_verify(session, "matmul/core", matmul_rounds, matmul_initial_claim, nullptr, reason)) return false;

        const NativeProofMessage* output_digest = core_find_message(session, NativeProofMessageKind::Data, "gkr/public-output", reason);
        if (output_digest == nullptr || output_digest->payload != core_vector_digest("zkgpt-masked-core-public-output-v1", public_output)) {
            set_reason(reason, "masked core public output binding mismatch");
            return false;
        }
        const std::size_t output_domain = next_power_of_two(public_output.size());
        const std::size_t output_dimension = core_log2(output_domain);
        const std::vector<Fr> output_point = core_point_from_session(session, "gkr/output/r", output_dimension, reason);
        if (output_point.size() != output_dimension) return false;
        std::vector<Fr> output_values(output_domain, Fr(0));
        std::copy(public_output.begin(), public_output.end(), output_values.begin());
        std::vector<Fr> folded = output_values;
        for (const Fr& coordinate : output_point) {
            const std::size_t half = folded.size() / 2;
            for (std::size_t index = 0; index < half; ++index) folded[index] = folded[2 * index] * (Fr(1) - coordinate) + folded[2 * index + 1] * coordinate;
            folded.resize(half);
        }
        if (!native_masked_gkr_quadratic_verify(session, "gkr/core", gkr_rounds, folded.at(0), gkr_coefficients, nullptr, reason)) return false;

        const NativeProofMessage* public_input_message = core_find_message(session, NativeProofMessageKind::Data, "combine/public-input", reason);
        if (public_input_message == nullptr || public_input_message->payload != core_vector_digest("zkgpt-masked-core-public-input-v1", public_input)) {
            set_reason(reason, "masked core public input binding mismatch");
            return false;
        }
        const NativeProofMessage* public_output_combine = core_find_message(session, NativeProofMessageKind::Data, "combine/public-output", reason);
        if (public_output_combine == nullptr || public_output_combine->payload != core_vector_digest("zkgpt-masked-core-public-output-v1", public_output)) {
            set_reason(reason, "masked core combine output binding mismatch");
            return false;
        }
        const std::size_t domain = generators.size();
        const NativeProofMessage* shape_message = core_find_message(session, NativeProofMessageKind::Data, "combine/opening-domain", reason);
        if (shape_message == nullptr || shape_message->payload != std::to_string(domain)) {
            set_reason(reason, "masked core opening domain mismatch");
            return false;
        }
        const CommitmentWire& com_aux = artifact.com_aux;
        const CommitmentWire& com_L = artifact.com_L;
        const NativeProofMessage* commitment_message = core_find_message(session, NativeProofMessageKind::Data, "combine/commitments", reason);
        if (commitment_message == nullptr || commitment_message->payload != core_commitment_payload(basis_id, com_W, com_aux, com_L, domain)) {
            set_reason(reason, "masked core commitment combine binding mismatch");
            return false;
        }
        const std::vector<Fr> point = core_point_from_session(session, "combine/r_minus", core_log2(domain), reason);
        if (point.size() != core_log2(domain)) return false;
        std::size_t w_open = 0, aux_open = 0, l_open = 0;
        if (!core_find_index(session, "Open_W/basis", NativeProofMessageKind::Data, w_open, reason) ||
            !core_find_index(session, "Open_Aux/basis", NativeProofMessageKind::Data, aux_open, reason) ||
            !core_find_index(session, "Open_L/basis", NativeProofMessageKind::Data, l_open, reason) || !(w_open < aux_open && aux_open < l_open)) {
            set_reason(reason, "masked core terminal openings are not canonically ordered");
            return false;
        }
        auto verify_opening = [&](const std::string& label, const CommitmentWire& wire) {
            const NativeProofMessage* claim = core_find_message(session, NativeProofMessageKind::Data, label + "/claim", reason);
            if (claim == nullptr) return false;
            std::vector<std::string> parts;
            if (!core_split_pipe(claim->payload, parts) || parts.size() != 2) {
                set_reason(reason, "malformed masked core opening claim: " + label);
                return false;
            }
            G1 commitment, mask_commitment;
            if (!core_decode_point(wire.point_hex, commitment) || !core_decode_point(parts[0], mask_commitment)) {
                set_reason(reason, "malformed masked core opening point: " + label);
                return false;
            }
            Fr masked_evaluation;
            if (!core_decode_field(parts[1], masked_evaluation)) {
                set_reason(reason, "malformed masked core opening evaluation: " + label);
                return false;
            }
            return hyrax::verify_masked_mle_opening_with_session(commitment, mask_commitment, masked_evaluation, basis_id, generators, point, session, label,
                                                                 reason, true, true);
        };

        if (!verify_opening("Open_W", com_W) || !verify_opening("Open_Aux", com_aux) || !verify_opening("Open_L", com_L)) return false;
        return true;
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}

NativePrimitiveEvidence exercise_official_primitives(const std::string& label) {
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

    native_prover.C.circuit[1].add_binary(0, 0, 1, 1, 0);
    native_prover.C.circuit[1].add_binary(0, 1, 0, 0, 0);
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

}
}
