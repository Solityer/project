#pragma once

#include "circuit.h"

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace zkgpt {
namespace gat {

using Int = std::int64_t;
using Wide = __int128_t;
using Field = F;

struct Shape {
    std::vector<std::size_t> dims;
    std::size_t elements() const;
    bool operator==(const Shape& other) const { return dims == other.dims; }
};

struct Tensor2D {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<Int> values;

    Tensor2D() = default;
    Tensor2D(std::size_t r, std::size_t c) : rows(r), cols(c), values(r * c, 0) {}
    Int& at(std::size_t r, std::size_t c) { return values.at(r * cols + c); }
    Int at(std::size_t r, std::size_t c) const { return values.at(r * cols + c); }
    bool same_shape(const Tensor2D& other) const { return rows == other.rows && cols == other.cols; }
};

struct CanonicalConfig {
    std::map<std::string, std::string> values;
    std::string source_path;

    static CanonicalConfig load(const std::string& path);
    std::string get(const std::string& key, const std::string& fallback = "") const;
    Int get_int(const std::string& key, Int fallback = 0) const;
    std::vector<Int> get_int_list(const std::string& key) const;
};

struct QuantConfig {
    std::string schema;
    std::string quant_cfg_id;
    Int s_input = 0;
    Int s_delta = 0;
    Int s_exp = 0;
    std::array<Int, 3> s_h{{0, 0, 0}};
    std::array<std::vector<Int>, 2> s_w_hidden;
    std::array<std::vector<Int>, 2> s_hprime_hidden;
    std::array<std::vector<Int>, 2> s_a_src_hidden;
    std::array<std::vector<Int>, 2> s_a_dst_hidden;
    std::array<Int, 2> s_agg_pre{{0, 0}};
    std::array<Int, 2> s_agg{{0, 0}};
    std::array<Int, 2> s_cat{{0, 0}};
    Int s_out_pre = 0;
    Int s_out_head = 0;
    Int s_out = 0;
    std::array<Int, 2> s_w_output{{0, 0}};
    std::array<Int, 2> s_yprime_output{{0, 0}};
    std::array<Int, 2> s_a_src_output{{0, 0}};
    std::array<Int, 2> s_a_dst_output{{0, 0}};
    std::array<Int, 2> s_b_output{{0, 0}};
    std::string rounding;
    std::string rescale;
    std::string output_average;

    Int range_bits = 12;

    std::string static_tables_id = "tables:lrelu+elu+exp+range";
    std::string static_table_generator_id = "zkgpt-gat-static-tables-v2";
    std::string static_row_order = "ascending_input";
    std::string exp_approx_id = "exp_neg_quantized";
    std::string elu_approx_id = "elu_expm1_quantized";
    Int static_lrelu_min = -(Int(1) << 12);
    Int static_lrelu_max = (Int(1) << 12);
    Int static_exp_min = 0;
    Int static_exp_max = (Int(1) << 13);
    Int static_elu_min = -(Int(1) << 12);
    Int static_elu_max = (Int(1) << 12);
    Int lrelu_slope_num = 1;
    Int lrelu_slope_den = 5;

    static QuantConfig load(const std::string& path);
    void validate() const;
};

struct Graph {
    std::size_t num_nodes = 0;
    std::size_t num_features = 0;
    std::size_t graph_count = 1;
    std::size_t num_classes = 0;
    std::vector<Int> src;
    std::vector<Int> dst;
    std::vector<Int> edge_gid;

    std::vector<std::size_t> node_ptr;
    std::vector<std::size_t> edge_ptr;

    std::size_t edges() const { return src.size(); }
    std::pair<std::size_t, std::size_t> node_range(std::size_t gid) const;
    std::pair<std::size_t, std::size_t> edge_range(std::size_t gid) const;
    void validate() const;
};

struct Tensor {
    std::string name;
    Shape shape;
    std::vector<double> values;
};

struct ModelManifest {
    std::string source_path;
    std::string dataset;
    std::string model_arch_id;
    std::string quant_cfg_id;
    std::size_t tensor_count = 0;
    std::size_t layers = 0;
    std::size_t output_heads = 0;
    std::size_t classes = 0;
    std::string model_param_id;
    std::string static_table_id;
    std::string input_preprocess_id;
    std::string batching_rule;
    std::string edge_sort_rule;
    std::string self_loop_rule;
    std::string label_format;
    std::string output_average_rule;
    std::string quantization_scale_chain;
};

struct Model {
    ModelManifest manifest;
    std::map<std::string, Tensor> tensors;

    static Model load(const std::string& bundle_dir);

    static Model load_shape_only(const std::string& bundle_dir);
    const Tensor& tensor(const std::string& name) const;
    void validate_schema() const;
    void validate_identity(const CanonicalConfig& config) const;
    std::size_t input_dim() const;
    std::size_t hidden_dim() const;

    std::size_t hidden_heads() const;
    std::size_t classes() const;
};

Tensor2D load_features(const std::string& cache_dir, const CanonicalConfig& config, const QuantConfig& quant, const Graph& graph);
std::vector<Int> load_labels(const std::string& cache_dir, const Graph& graph);
Graph load_graph(const std::string& cache_dir, const CanonicalConfig& config);
std::string comparison_canonical_graph_digest(const Graph& graph);
std::string sha256_file_digest(const std::string& path);

enum class RoundingMode { TOWARD_ZERO, NEAREST_TIES_AWAY };

struct DivisionWitness {
    Int sigma = 0;
    Int z = 0;
    Int a = 0;
    Int q = 0;
    Int r = 0;
    Int delta = 0;
    Field a_inverse = Field(0);
    Field d_inverse = Field(0);
    Int dividend = 0;
    Int denominator = 0;

    bool valid(RoundingMode mode) const;
};

struct DivisionResult {
    Int value = 0;
    DivisionWitness witness;
};

DivisionResult signed_round_div(Int dividend, Int denominator, RoundingMode mode = RoundingMode::NEAREST_TIES_AWAY);

struct TablePoint {
    Int x = 0;
    Int y = 0;
};

struct StaticTables {
    std::string table_id;
    std::string generator_id;
    std::string digest;
    std::string quant_cfg_id;
    std::string rounding;
    std::string rescale;
    std::string output_average;
    std::string row_order;
    std::string exp_approx_id;
    std::string elu_approx_id;
    Int lrelu_slope_num = 1;
    Int lrelu_slope_den = 5;
    Int s_delta = 0;
    Int s_exp = 0;
    Int s_elu_in = 0;
    Int s_elu_out = 0;
    Int lrelu_min = 0;
    Int lrelu_max = 0;
    Int exp_min = 0;
    Int exp_max = 0;
    Int elu_min = 0;
    Int elu_max = 0;
    std::vector<TablePoint> lrelu;
    std::vector<TablePoint> exp;
    std::vector<TablePoint> elu;

    static StaticTables build(const QuantConfig& quant);

    static StaticTables build(const QuantConfig& quant, Int observed_score_abs, Int observed_delta_max, Int observed_elu_abs);
    Int lookup_lrelu(Int x) const;
    Int lookup_exp(Int delta) const;
    Int lookup_elu(Int x) const;
    bool authenticate() const;
};

struct BoundFamily {
    std::string id;
    Wide model_abs = 0;
    Wide input_abs = 0;
    Wide weight_abs = 0;
    Wide attention_abs = 0;
    Wide score_abs = 0;
    Wide table_input_abs = 0;
    Wide table_output_abs = 0;
    Wide delta_abs = 0;
    Wide exp_abs = 0;
    Wide sum_abs = 0;
    Wide numerator_abs = 0;
    Wide division_x_abs = 0;
    Wide division_d_abs = 0;
    Wide packed_tuple_abs = 0;
    Wide q_abs = 0;
    Wide r_abs = 0;
    Wide delta_aux_abs = 0;
    Wide product_abs = 0;
    Wide input_dim = 0;
    Wide output_dim = 0;
    Wide max_in_degree = 0;
    Wide projection_abs = 0;
    Wide source_score_abs = 0;
    Wide destination_score_abs = 0;
    Wide alignment_x_abs = 0;
    Wide average_x_abs = 0;
    Wide elu_input_abs = 0;
    std::string field_modulus;
    std::string field_half_modulus;
    bool inverse_field_valid = false;
    bool no_wrap = false;
};

struct IntegerBoundLedger {
    std::string backend = "BN254";
    std::string table_id;
    std::string field_modulus;
    std::string field_half_modulus;
    Int lrelu_min = 0;
    Int lrelu_max = 0;
    Int exp_min = 0;
    Int exp_max = 0;
    Int elu_min = 0;
    Int elu_max = 0;
    std::vector<BoundFamily> families;

    void add(const BoundFamily& family);
    bool validate() const;
};

struct AttentionTrace {
    std::string family_id;
    std::size_t input_dim = 0;
    std::size_t output_dim = 0;
    std::vector<Int> projection;
    std::vector<Int> source_score;
    std::vector<Int> destination_score;
    std::vector<Int> raw_score;
    std::vector<Int> leaky_score;
    std::vector<Int> group_max;
    std::vector<Int> delta;
    std::vector<Int> exp;
    std::vector<Int> group_sum;
    std::vector<Field> group_inverse;
    std::vector<Int> max_selector;
    std::vector<Field> alpha;
    std::vector<Int> integer_numerator;
    std::vector<Field> field_numerator;
    std::vector<Field> field_aggregate;
    std::vector<Int> recovered;
    std::vector<Int> activated;
    std::vector<DivisionWitness> divisions;
    bool field_integer_consistent = false;
    bool relation_ok = false;
};

struct HiddenLayerTrace {
    std::size_t layer = 0;
    std::vector<AttentionTrace> heads;
    Tensor2D concatenated;
};

struct OutputHeadTrace {
    std::size_t head = 0;
    AttentionTrace attention;
    std::vector<Int> bias;
    std::vector<Int> aggregate;
    std::vector<Int> aligned;
    std::vector<Int> logits;
    std::vector<DivisionWitness> alignment_divisions;
};

struct GatTrace {
    Tensor2D input;
    std::vector<HiddenLayerTrace> hidden_layers;
    std::vector<OutputHeadTrace> output_heads;
    std::vector<Int> head_sum;
    std::vector<DivisionWitness> final_average_divisions;
    Tensor2D logits;
    StaticTables tables;
    IntegerBoundLedger ledger;
    bool relation_ok = false;
};

struct ForwardResult {
    Tensor2D logits;
    GatTrace trace;
};

using ForwardAdmissionProgress = void (*)(const char* phase, std::size_t layer,
                                         std::size_t head, bool complete);

// Bounded, value-exact preflight of the canonical forward relation. The
// implementation streams one attention head at a time and retains no proof
// witness trace; optional reference arguments are for native equivalence tests
// only and are never serialized by the formal runner.
bool run_forward_admission(const Graph& graph, const Tensor2D& features, const Model& model, const QuantConfig& quant,
                           const std::vector<std::array<unsigned, 5>>& division_range_profiles, std::string* reason = nullptr,
                           const ForwardResult* retained_reference = nullptr,
                           ForwardAdmissionProgress progress = nullptr);

ForwardResult run_forward(const Graph& graph, const Tensor2D& features, const Model& model, const QuantConfig& quant);
bool validate_relation(const Graph& graph, const Tensor2D& features, const Model& model, const QuantConfig& quant, const ForwardResult& result);

struct RelationDescriptor {
    std::string id;
    std::string primitive;
    std::string segment;
    std::string relation;
    std::string public_claim;
    std::string witness;
};

struct RelationRegistry {
    std::string registry_id = "zkgpt-gat-native-relations-v1";
    std::vector<RelationDescriptor> descriptors;

    static RelationRegistry canonical();
    const RelationDescriptor* find(const std::string& id) const;
    bool covers_all_segments() const;
};

struct OfficialCircuitPlan {
    layeredCircuit circuit;
    RelationRegistry registry;
    std::vector<std::string> native_api;
    bool ready_for_official_consumers = false;
};

OfficialCircuitPlan build_official_circuit_plan(const Graph& graph, const Model& model, const QuantConfig& quant);

}
}
