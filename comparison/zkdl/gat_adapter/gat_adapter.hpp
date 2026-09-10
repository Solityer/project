#pragma once

// GAT adaptation data model for the official CUDA/C++ zkDL tree.
//
// This header deliberately contains no prover implementation.  It describes
// the canonical witness, public statement, and operation-family views that
// the native zkDL Sumcheck/Hyrax implementation consumes.  The implementation
// lives in gat_runner.cu and keeps all GAT-specific logic in this directory.

#include <cstdint>
#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace zkdl_gat {

enum class Backend { Cpu, Gpu };

const char* backend_name(Backend backend) noexcept;
Backend parse_backend(const std::string& value);

struct Matrix {
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::vector<std::int64_t> values;

    std::int64_t& at(std::uint64_t row, std::uint64_t col) {
        return values.at(static_cast<std::size_t>(row * cols + col));
    }
    const std::int64_t& at(std::uint64_t row, std::uint64_t col) const {
        return values.at(static_cast<std::size_t>(row * cols + col));
    }
};

struct FieldValue {
    // Canonical little-endian 32-byte encoding of a BLS12-381 scalar.  The
    // native bridge converts this encoding to its Fr_t representation before
    // committing or opening; keeping it byte-oriented here avoids a second
    // field implementation in the adapter.
    std::array<std::uint8_t, 32> bytes{};
};

struct FieldMatrix {
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::vector<FieldValue> values;
};

struct Quantization {
    std::uint64_t input = 16;
    std::uint64_t delta = 256;
    std::uint64_t exp = 16;
    std::vector<std::uint64_t> hidden_input{16, 16, 16};
    std::vector<std::uint64_t> hidden_aggregate_pre{16, 16};
    std::vector<std::uint64_t> hidden_aggregate{16, 16};
    std::vector<std::vector<std::uint64_t>> hidden_weight;
    std::vector<std::vector<std::uint64_t>> hidden_h_prime;
    std::vector<std::vector<std::uint64_t>> hidden_attention_src;
    std::vector<std::vector<std::uint64_t>> hidden_attention_dst;
    std::vector<std::uint64_t> output_weight{4, 4};
    std::vector<std::uint64_t> output_y_prime{64, 64};
    std::vector<std::uint64_t> output_attention_src{4, 4};
    std::vector<std::uint64_t> output_attention_dst{4, 4};
    std::vector<std::uint64_t> output_bias{16, 16};
    std::vector<std::uint64_t> hidden_concat{16, 16};
    std::uint64_t output_pre = 16;
    std::uint64_t output_head = 16;
    std::uint64_t output = 16;
    std::string rounding = "nearest_ties_away";
    std::string rescale = "none";
    std::string output_average = "signed_round_div";
};

struct Architecture {
    std::uint64_t layers = 0;
    std::vector<std::uint64_t> hidden_heads;
    std::vector<std::uint64_t> hidden_dims;
    std::vector<std::uint64_t> input_dims;
    std::uint64_t output_heads = 0;
    std::uint64_t classes = 0;
};

struct Config {
    std::filesystem::path baseline_root;
    std::filesystem::path config_path;
    std::filesystem::path data_root;
    std::filesystem::path cache_root;
    std::filesystem::path checkpoint_bundle;
    std::filesystem::path run_root;
    std::filesystem::path asset_manifest_path;
    std::filesystem::path static_table_descriptor_path;
    std::filesystem::path integer_bound_ledger_path;
    std::filesystem::path proof_path;
    std::filesystem::path statement_path;
    std::filesystem::path output_path;
    std::string dataset;
    std::string model_arch_id;
    std::string model_param_id;
    std::string quant_cfg_id;
    std::string static_table_id;
    std::string input_preprocess_id;
    std::string proof_task;
    std::string privacy_policy;
    std::string input_privacy;
    std::string output_mode;
    std::string batching_rule;
    std::string self_loop_rule;
    std::string edge_sort_rule;
    std::string task_type;
    std::string report_unit;
    std::string config_digest;
    std::string asset_manifest_digest;
    std::string chunking_rule;
    std::string degree_bound_id;
    std::string srs_type;
    std::string srs_id;
    std::string srs_path;
    std::string opening_mode;
    std::string binding_mode;
    std::string security_profile_id;
    std::string verifier_registry_digest;
    std::string scale_chain_digest;
    std::uint64_t range_bits = 0;
    std::uint64_t batch_graphs = 0;
    std::uint64_t seed = 0;
    bool symmetrize_edges = false;
    bool deduplicate_edges = true;
    bool auto_prepare_dataset = false;
    bool dump_trace = false;
    // Independent verification may authenticate public metadata and relation
    // resources without opening private checkpoint tensor payloads.
    bool public_verifier_mode = false;
    Architecture architecture;
    Quantization quantization;
};

struct Graph {
    std::uint64_t total_nodes = 0;
    std::uint64_t nodes = 0;
    std::uint64_t edges = 0;
    std::uint64_t graphs = 0;
    std::uint64_t features = 0;
    std::vector<std::uint64_t> absolute_ids;
    std::vector<std::uint32_t> src;
    std::vector<std::uint32_t> dst;
    // Protocol edge_ptr is the graph-batch prefix pointer.  The destination
    // CSR pointer is kept separately because it is a derived selector used by
    // the native sparse/group relations, not a replacement for the public
    // batch boundary.
    std::vector<std::uint64_t> node_ptr;
    std::vector<std::uint64_t> edge_ptr;
    std::vector<std::uint64_t> node_edge_ptr;
    Matrix feature_values;
};

struct ModelTensor {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::vector<double> values;
};

struct ModelBundle {
    std::filesystem::path root;
    std::string dataset;
    std::string model_arch_id;
    std::string model_param_id;
    std::string quant_cfg_id;
    std::string static_table_id;
    std::string manifest_digest;
    std::string tensors_text_digest;
    std::string tensors_npz_digest;
    std::string best_model_digest;
    std::uint64_t tensor_count_manifest = 0;
    std::map<std::string, ModelTensor> tensors;
};

struct StaticTables {
    std::int64_t lrelu_min = 0;
    std::int64_t lrelu_max = 0;
    std::int64_t elu_min = 0;
    std::int64_t elu_max = 0;
    std::uint64_t exp_max = 0;
    std::vector<std::int64_t> lrelu;
    std::vector<std::int64_t> elu;
    std::vector<std::int64_t> exp;
    std::string generator_id;
    std::string version;
    std::string scale_id;
    std::string domain_id;
    std::string row_order;
    std::string spec_hash;
    std::string digest;
    std::string descriptor_digest;
    std::string quant_cfg_id;
    bool authenticated = false;
};

struct IntegerBoundFamily {
    std::uint64_t layer = 0;
    std::uint64_t head = 0;
    bool output = false;
    std::int64_t max_input_abs = 0;
    std::int64_t max_weight_abs = 0;
    std::int64_t max_attention_abs = 0;
    std::int64_t max_bias_abs = 0;
    std::int64_t max_score_abs = 0;
    std::int64_t max_table_output_abs = 0;
    std::int64_t max_delta = 0;
    std::int64_t max_exp = 0;
    std::int64_t max_sum = 0;
    std::int64_t max_aggregate_num = 0;
    std::int64_t max_dividend_abs = 0;
    std::int64_t max_aux_abs = 0;
    std::int64_t max_divisor = 0;
    std::int64_t max_field_product_abs = 0;
};

struct IntegerBoundLedger {
    std::string schema = "zkDL_gat_integer_bound_ledger_v1";
    std::string ledger_id;
    std::string quant_cfg_id;
    std::string static_table_id;
    std::string field_modulus;
    std::string scale_chain_digest;
    std::string static_table_descriptor_digest;
    std::string digest;
    std::string descriptor_digest;
    std::int64_t max_score_abs = 0;
    std::int64_t max_table_output_abs = 0;
    std::int64_t max_delta = 0;
    std::int64_t max_exp = 0;
    std::int64_t max_sum = 0;
    std::int64_t max_aggregate_num = 0;
    std::int64_t max_dividend_abs = 0;
    std::int64_t max_aux_abs = 0;
    std::int64_t max_divisor = 0;
    std::vector<IntegerBoundFamily> families;
    bool authenticated = false;
};

struct SignedRoundDivWitness {
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::vector<std::uint8_t> sigma;
    std::vector<std::uint8_t> z;
    std::vector<std::int64_t> a;
    std::vector<std::int64_t> q;
    std::vector<std::int64_t> r;
    std::vector<std::int64_t> delta;
    std::vector<std::int64_t> divisor;
    std::vector<FieldValue> a_inverse;
    std::vector<FieldValue> divisor_inverse;
};

struct HiddenHeadTrace {
    Matrix h_prime;
    std::vector<std::int64_t> e_src;
    std::vector<std::int64_t> e_dst;
    std::vector<std::int64_t> score;
    std::vector<std::int64_t> leaky_relu;
    std::vector<std::int64_t> maximum;
    std::vector<std::int64_t> delta;
    std::vector<std::int64_t> exp_weight;
    std::vector<std::int64_t> denominator;
    std::vector<FieldValue> inverse_field;
    std::vector<FieldValue> alpha_field;
    Matrix routed_h_prime;
    Matrix product_u_h_prime;
    FieldMatrix product_alpha_h_prime_field;
    Matrix aggregate_pre;
    FieldMatrix aggregate_pre_field;
    FieldMatrix consistency_residual_field;
    Matrix aggregate_num;
    Matrix elu_input;
    Matrix aggregate;
    SignedRoundDivWitness aggregate_round_div;
};

struct HiddenLayerTrace {
    Matrix input;
    std::vector<HiddenHeadTrace> heads;
    Matrix concat;
};

struct OutputHeadTrace {
    Matrix y_prime;
    std::vector<std::int64_t> e_src;
    std::vector<std::int64_t> e_dst;
    std::vector<std::int64_t> score;
    std::vector<std::int64_t> leaky_relu;
    std::vector<std::int64_t> maximum;
    std::vector<std::int64_t> delta;
    std::vector<std::int64_t> exp_weight;
    std::vector<std::int64_t> denominator;
    std::vector<FieldValue> inverse_field;
    std::vector<FieldValue> alpha_field;
    Matrix routed_y_prime;
    Matrix product_u_y_prime;
    FieldMatrix product_alpha_y_prime_field;
    Matrix aggregate_pre;
    FieldMatrix aggregate_pre_field;
    FieldMatrix consistency_residual_field;
    Matrix aggregate_num;
    Matrix y_quantized;
    Matrix aligned;
    std::vector<std::int64_t> bias;
    Matrix output;
    SignedRoundDivWitness aggregate_round_div;
    SignedRoundDivWitness alignment_round_div;
};

struct GatTrace {
    Matrix first_input;
    std::vector<HiddenLayerTrace> hidden_layers;
    std::vector<OutputHeadTrace> output_heads;
    Matrix output_sum;
    Matrix logits;
    SignedRoundDivWitness final_average_round_div;
};

struct TensorFamily {
    std::string id;
    std::string visibility;
    std::string mle_order;
    std::string stack_axis;
    std::vector<std::uint64_t> logical_shape;
    std::vector<std::uint64_t> padded_shape;
    std::vector<std::string> instances;
    std::string instance_order;
    std::string mapping;
    std::string zero_selector;
    std::string terminal_claim;
    std::string selector_id;
    std::string padding_rule;
    std::string terminal_point;
    std::string terminal_value;
    std::uint64_t degree_bound = 0;
};

struct OperationFamily {
    std::string id;
    std::string kind;
    std::string degree_bound;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> views;
    bool requires_reindex = false;
    bool public_route = false;
    std::string instance_order;
    std::string mapping;
    std::vector<std::string> auxiliary;
    std::string selector_id;
    std::string terminal_claim;
    std::string terminal_point;
    std::string terminal_value;
    std::uint64_t degree = 0;
};

struct RelationPlan {
    std::string schema_id = "zkDL_gat_inference_v1";
    std::string proof_scope = "full_gat_forward";
    std::string binding_rule = "fac4dnn_reindex_v1";
    std::string registry_digest;
    std::string terminal_claims_digest;
    std::string schema_digest;
    std::string mapping_digest;
    std::string selector_digest;
    std::string degree_digest;
    std::string static_table_descriptor_digest;
    std::string scale_chain_digest;
    std::string integer_bound_ledger_digest;
    std::string integer_bound_ledger_descriptor_digest;
    IntegerBoundLedger integer_bound_ledger;
    std::vector<TensorFamily> tensor_families;
    std::vector<OperationFamily> operation_families;
};

struct PublicStatement {
    std::string statement_id;
    std::string dataset;
    std::string model_arch_id;
    std::string model_param_id;
    std::string quant_cfg_id;
    std::string static_table_id;
    std::string input_preprocess_id;
    std::string topology_id;
    std::string feature_id;
    std::string output_id;
    std::string public_output_path = "public_Y.txt";
    std::string public_output_digest;
    std::string relation_registry_digest;
    std::string relation_terminal_claims_digest;
    std::string relation_mapping_digest;
    std::string relation_selector_digest;
    std::string relation_degree_digest;
    std::string static_table_generator_id;
    std::string static_table_version;
    std::string static_table_scale_id;
    std::string static_table_domain_id;
    std::string static_table_row_order;
    std::string static_table_spec_hash;
    std::string static_table_digest;
    std::string static_table_descriptor_digest;
    std::string scale_chain_digest;
    std::string integer_bound_ledger_id;
    std::string integer_bound_ledger_digest;
    std::string integer_bound_ledger_descriptor_digest;
    std::string config_digest;
    std::string asset_manifest_digest;
    std::string checkpoint_manifest_digest;
    std::string checkpoint_tensors_txt_digest;
    std::string checkpoint_tensors_npz_digest;
    std::string topology_boundary_digest;
    std::string verifier_registry_digest;
    std::uint64_t total_nodes = 0;
    std::uint64_t nodes = 0;
    std::uint64_t edges = 0;
    std::uint64_t graphs = 0;
    std::uint64_t classes = 0;
    std::uint64_t hidden_layer_count = 0;
    std::uint64_t hidden_head_count = 0;
    std::uint64_t output_head_count = 0;
    std::string proof_scope = "full_gat_forward";
    std::string privacy_policy = "model_and_trace_hidden";
    bool verifier_reads_witness = false;
};

struct ProofEnvelope {
    std::string schema_id = "zkDL_gat_proof_envelope_v1";
    std::string statement_id;
    std::string proof_id;
    std::string backend;
    std::string proof_scope = "full_gat_forward";
    std::string native_core_api = "zkdl_native_gat_bridge_v1";
    std::string status = "UNAVAILABLE";
    std::string static_table_descriptor_digest;
    std::string scale_chain_digest;
    std::string integer_bound_ledger_id;
    std::string integer_bound_ledger_digest;
    std::string integer_bound_ledger_descriptor_digest;
    std::uint64_t proof_size_bytes = 0;
    std::uint64_t commitment_size_bytes = 0;
    double commitment_time_seconds = 0.0;
    double prove_time_seconds = 0.0;
    double verify_time_seconds = 0.0;
    bool verifier_reads_witness = false;
    bool gpu_kernel_executed = false;
    bool silent_fallback = false;
    bool output_bound = false;
    std::string error;
};

struct RunRecord {
    std::string dataset;
    std::string backend;
    std::string statement_id;
    std::string proof_id;
    std::string verification_status = "NOT_RUN";
    std::string proof_scope = "full_gat_forward";
    std::string static_table_descriptor_digest;
    std::string scale_chain_digest;
    std::string integer_bound_ledger_id;
    std::string integer_bound_ledger_digest;
    std::string integer_bound_ledger_descriptor_digest;
    std::uint64_t proof_size_bytes = 0;
    std::uint64_t commitment_size_bytes = 0;
    double commitment_time_seconds = 0.0;
    double prove_time_seconds = 0.0;
    double verify_time_seconds = 0.0;
    bool verifier_reads_witness = false;
    bool gpu_kernel_executed = false;
    bool silent_fallback = false;
    bool output_bound = false;
    std::string error;
};

Config load_config(const std::filesystem::path& path);
struct AssetManifestEntry {
    std::string path;
    std::uint64_t bytes = 0;
    std::string sha256;
};

struct AssetManifest {
    std::string schema;
    std::string dataset;
    std::string quant_cfg_id;
    std::string config;
    std::string scale_chain_digest;
    std::string static_table_descriptor;
    std::string static_table_descriptor_digest;
    std::string integer_bound_ledger;
    std::string integer_bound_ledger_descriptor_digest;
    std::string digest;
    std::vector<AssetManifestEntry> files;
};

AssetManifest load_asset_manifest(const Config& config);
ModelBundle load_model_bundle(const Config& config);
ModelBundle load_model_metadata(const Config& config);
Graph load_graph(const Config& config);
StaticTables build_static_tables(const Config& config, const Graph& graph);
RelationPlan build_relation_plan(const Config& config, const Graph& graph,
                                 const StaticTables& tables);
GatTrace execute_cpu(const Config& config, const Graph& graph,
                     const ModelBundle& model, const StaticTables& tables);
GatTrace execute_gpu(const Config& config, const Graph& graph,
                     const ModelBundle& model, const StaticTables& tables,
                     bool* kernel_executed);
PublicStatement make_statement(const Config& config, const Graph& graph,
                               const ModelBundle& model, const StaticTables& tables,
                               const RelationPlan& relation, const GatTrace& trace);
void write_statement(const std::filesystem::path& path,
                     const PublicStatement& statement);
bool read_statement(const std::filesystem::path& path,
                    PublicStatement* statement, std::string* error);
void write_run_record(const std::filesystem::path& path, const RunRecord& record);

// Public output is a verifier input, not a value trusted from a proof
// envelope.  The production verifier reads this canonical integer matrix
// independently and checks the statement digest before invoking native code.
Matrix read_public_output(const std::filesystem::path& path,
                          std::uint64_t expected_rows,
                          std::uint64_t expected_cols);

// Structural validation shared by the production bridge and short negative
// tests.  It deliberately rejects missing/duplicate/non-canonical registry
// entries before any native proving call.
bool relation_plan_complete(const RelationPlan& relation, std::string* error = nullptr);

}  // namespace zkdl_gat
